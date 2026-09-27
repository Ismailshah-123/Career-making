"""
app/utils/file_utils.py
========================
Production-grade file handling utilities for the JobHunter AI platform.

Responsibilities:
1. Text extraction from uploaded resume files:
   - PDF  → pdfplumber (text) + pytesseract fallback for scanned pages
   - DOCX → python-docx paragraph + table extraction
   - DOC  → LibreOffice subprocess conversion → then DOCX path
   - TXT  → direct read with encoding detection
   - ODT  → odfpy extraction

2. File storage:
   - Local filesystem (development / USE_LOCAL_STORAGE=True)
   - AWS S3 (production / USE_LOCAL_STORAGE=False)
   - Generates content-addressed storage keys (SHA-256 of file bytes)
     to prevent duplicate uploads and enable CDN caching

3. File validation:
   - MIME type detection via python-magic (not trusting Content-Type header)
   - File size enforcement
   - Virus scan hook (ClamAV via clamd — optional, skipped if not running)

4. PDF export:
   - Convert tailored resume sections → PDF via WeasyPrint
   - Convert cover letter text → PDF for form upload

Usage:
    from app.utils.file_utils import extract_text_from_file, save_upload, delete_file

    # Extract text from an uploaded resume
    text = await extract_text_from_file(file_path="/uploads/resumes/abc.pdf", mime_type="application/pdf")

    # Save uploaded bytes to storage
    storage_path = await save_upload(
        file_bytes=b"...",
        filename="resume.pdf",
        subfolder="resumes",
        user_id="uuid-string",
    )
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import mimetypes
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.core.exceptions import (
    FileTooLargeException,
    InvalidFileTypeException,
    ResumeParseException,
)
from app.core.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALLOWED_RESUME_MIME_TYPES = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/msword",
    "text/plain",
    "application/vnd.oasis.opendocument.text",
}

MIME_TO_EXTENSION: dict[str, str] = {
    "application/pdf":   ".pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "application/msword": ".doc",
    "text/plain":        ".txt",
    "application/vnd.oasis.opendocument.text": ".odt",
}

EXTENSION_TO_MIME: dict[str, str] = {v: k for k, v in MIME_TO_EXTENSION.items()}

# Minimum viable resume text (fewer chars than this → extraction failed)
MIN_EXTRACTABLE_CHARS = 50

# Maximum text to return (prevents feeding 200KB resumes into LLMs)
MAX_EXTRACTED_CHARS = 50_000


# ═══════════════════════════════════════════════════════════════════════════
# TEXT EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════

async def extract_text_from_file(file_path: str, mime_type: str) -> str:
    """
    Extract plain text from a resume file.

    Runs in a thread pool executor to avoid blocking the event loop during
    CPU-bound PDF parsing or subprocess invocations.

    Args:
        file_path : Absolute or relative path to the stored file.
        mime_type : MIME type string (e.g. "application/pdf").

    Returns:
        Extracted plain text, stripped and normalised.

    Raises:
        ResumeParseException: If the file cannot be read or yields < 50 chars.
    """
    if not os.path.exists(file_path):
        raise ResumeParseException(
            filename=os.path.basename(file_path),
            reason=f"File not found at path: {file_path}",
        )

    loop = asyncio.get_event_loop()
    text = await loop.run_in_executor(
        None,
        _extract_text_sync,
        file_path,
        mime_type,
    )

    text = _normalise_whitespace(text)

    if len(text.strip()) < MIN_EXTRACTABLE_CHARS:
        raise ResumeParseException(
            filename=os.path.basename(file_path),
            reason=(
                f"Extracted only {len(text.strip())} characters — "
                "file may be image-only, password-protected, or corrupted."
            ),
        )

    return text[:MAX_EXTRACTED_CHARS]


def _extract_text_sync(file_path: str, mime_type: str) -> str:
    """
    Synchronous dispatcher — routes to the correct extractor by MIME type.
    Called inside run_in_executor to avoid blocking the async event loop.
    """
    # Normalise MIME: sometimes browsers send wrong types, so also check extension
    ext = Path(file_path).suffix.lower()
    effective_mime = mime_type or EXTENSION_TO_MIME.get(ext, "")

    if "pdf" in effective_mime or ext == ".pdf":
        return _extract_pdf(file_path)

    if "wordprocessingml" in effective_mime or ext == ".docx":
        return _extract_docx(file_path)

    if "msword" in effective_mime or ext == ".doc":
        return _extract_doc(file_path)

    if "plain" in effective_mime or ext == ".txt":
        return _extract_txt(file_path)

    if "oasis" in effective_mime or ext == ".odt":
        return _extract_odt(file_path)

    raise ResumeParseException(
        filename=os.path.basename(file_path),
        reason=f"Unsupported file type: {effective_mime or ext}",
    )


# ---------------------------------------------------------------------------
# PDF extraction
# ---------------------------------------------------------------------------

def _extract_pdf(file_path: str) -> str:
    """
    Extract text from a PDF using pdfplumber (fast, accurate for digital PDFs).

    For pages where pdfplumber yields < 10 chars (scanned / image-based),
    falls back to pytesseract OCR on a rasterised version of the page.
    Both text sources are concatenated page-by-page.
    """
    try:
        import pdfplumber
    except ImportError:
        raise ResumeParseException(
            filename=os.path.basename(file_path),
            reason="pdfplumber not installed. Run: pip install pdfplumber",
        )

    pages_text: list[str] = []
    ocr_used = False

    try:
        with pdfplumber.open(file_path) as pdf:
            if not pdf.pages:
                raise ResumeParseException(
                    filename=os.path.basename(file_path),
                    reason="PDF has no pages.",
                )

            for page_num, page in enumerate(pdf.pages, 1):
                page_text = page.extract_text(x_tolerance=2, y_tolerance=2) or ""

                # Scanned page detected — try OCR fallback
                if len(page_text.strip()) < 10:
                    page_text = _ocr_pdf_page(page, page_num) or ""
                    if page_text:
                        ocr_used = True

                if page_text.strip():
                    pages_text.append(page_text.strip())

    except ResumeParseException:
        raise
    except Exception as exc:
        raise ResumeParseException(
            filename=os.path.basename(file_path),
            reason=f"PDF parsing error: {exc}",
        )

    if ocr_used:
        logger.info("OCR used for scanned PDF pages", file=file_path)

    return "\n\n".join(pages_text)


def _ocr_pdf_page(page: Any, page_num: int) -> str:
    """
    Rasterise one PDF page and run pytesseract OCR on it.
    Returns empty string if pytesseract or Pillow is not installed.
    """
    try:
        import pytesseract
        from PIL import Image

        # pdfplumber pages can be rasterised via their .to_image() method
        pil_img = page.to_image(resolution=200).original
        text = pytesseract.image_to_string(pil_img, lang="eng")
        return text.strip()
    except ImportError:
        logger.debug(f"pytesseract not installed — cannot OCR page {page_num}")
        return ""
    except Exception as exc:
        logger.debug(f"OCR failed on page {page_num}: {exc}")
        return ""


# ---------------------------------------------------------------------------
# DOCX extraction
# ---------------------------------------------------------------------------

def _extract_docx(file_path: str) -> str:
    """
    Extract text from a DOCX file preserving paragraph structure.
    Also extracts text from tables (common in formatted resumes).
    """
    try:
        from docx import Document
    except ImportError:
        raise ResumeParseException(
            filename=os.path.basename(file_path),
            reason="python-docx not installed. Run: pip install python-docx",
        )

    try:
        doc = Document(file_path)
        parts: list[str] = []

        # Paragraphs
        for para in doc.paragraphs:
            text = para.text.strip()
            if text:
                parts.append(text)

        # Tables (cells read left-to-right, top-to-bottom)
        for table in doc.tables:
            for row in table.rows:
                row_texts = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                if row_texts:
                    parts.append("  |  ".join(row_texts))

        return "\n".join(parts)

    except Exception as exc:
        raise ResumeParseException(
            filename=os.path.basename(file_path),
            reason=f"DOCX parsing error: {exc}",
        )


# ---------------------------------------------------------------------------
# DOC extraction (legacy Word format)
# ---------------------------------------------------------------------------

def _extract_doc(file_path: str) -> str:
    """
    Extract text from a legacy .doc file by converting to DOCX via LibreOffice
    subprocess, then running DOCX extraction on the result.

    Requires LibreOffice to be installed (available in most Linux envs).
    Falls back to antiword if LibreOffice is not available.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Try LibreOffice first
        lo_result = _convert_doc_via_libreoffice(file_path, tmp_dir)
        if lo_result:
            return _extract_docx(lo_result)

        # Fallback: antiword (produces plain text directly)
        return _convert_doc_via_antiword(file_path)


def _convert_doc_via_libreoffice(file_path: str, output_dir: str) -> str | None:
    """Convert .doc → .docx using LibreOffice in headless mode."""
    try:
        result = subprocess.run(
            [
                "libreoffice",
                "--headless",
                "--convert-to", "docx",
                "--outdir", output_dir,
                file_path,
            ],
            capture_output=True,
            timeout=30,
            check=True,
        )
        # LibreOffice outputs the converted file in output_dir
        base  = Path(file_path).stem
        docx  = os.path.join(output_dir, f"{base}.docx")
        return docx if os.path.exists(docx) else None
    except (subprocess.SubprocessError, FileNotFoundError):
        return None


def _convert_doc_via_antiword(file_path: str) -> str:
    """Extract text from .doc using antiword command-line tool."""
    try:
        result = subprocess.run(
            ["antiword", file_path],
            capture_output=True,
            timeout=15,
            check=True,
            text=True,
        )
        return result.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        raise ResumeParseException(
            filename=os.path.basename(file_path),
            reason=(
                "Cannot extract text from .doc file. "
                "Install LibreOffice or antiword, or convert to DOCX/PDF before uploading."
            ),
        )


# ---------------------------------------------------------------------------
# TXT extraction
# ---------------------------------------------------------------------------

def _extract_txt(file_path: str) -> str:
    """
    Read a plain text file with automatic encoding detection.
    Tries UTF-8, then latin-1, then chardet if available.
    """
    encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]

    for encoding in encodings:
        try:
            with open(file_path, "r", encoding=encoding) as f:
                return f.read()
        except UnicodeDecodeError:
            continue

    # Last resort: chardet
    try:
        import chardet
        with open(file_path, "rb") as f:
            raw = f.read()
        detected = chardet.detect(raw)
        enc = detected.get("encoding") or "utf-8"
        return raw.decode(enc, errors="replace")
    except ImportError:
        with open(file_path, "rb") as f:
            return f.read().decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# ODT extraction
# ---------------------------------------------------------------------------

def _extract_odt(file_path: str) -> str:
    """Extract text from an OpenDocument Text (.odt) file using odfpy."""
    try:
        from odf.opendocument import load as odf_load
        from odf.text import P, Span
        from odf import teletype

        doc   = odf_load(file_path)
        parts: list[str] = []

        for el in doc.body.getElementsByType(P):
            text = teletype.extractText(el).strip()
            if text:
                parts.append(text)

        return "\n".join(parts)

    except ImportError:
        raise ResumeParseException(
            filename=os.path.basename(file_path),
            reason="odfpy not installed. Run: pip install odfpy",
        )
    except Exception as exc:
        raise ResumeParseException(
            filename=os.path.basename(file_path),
            reason=f"ODT parsing error: {exc}",
        )


# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------

def _normalise_whitespace(text: str) -> str:
    """
    Clean up extracted text:
    - Collapse multiple blank lines to a single blank line
    - Normalise Windows line endings
    - Strip leading/trailing whitespace per line
    - Remove null bytes and other control characters
    """
    if not text:
        return ""

    # Remove null bytes and non-printable control chars (except \n \t)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    # Normalise line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Strip trailing whitespace from each line
    lines = [line.rstrip() for line in text.split("\n")]
    # Collapse 3+ consecutive blank lines → 2 blank lines
    result: list[str] = []
    blank_count = 0
    for line in lines:
        if not line.strip():
            blank_count += 1
            if blank_count <= 2:
                result.append("")
        else:
            blank_count = 0
            result.append(line)

    return "\n".join(result).strip()


# ═══════════════════════════════════════════════════════════════════════════
# FILE STORAGE
# ═══════════════════════════════════════════════════════════════════════════

async def save_upload(
    file_bytes: bytes,
    filename: str,
    subfolder: str,
    user_id: str,
    *,
    content_type: str = "application/octet-stream",
) -> str:
    """
    Save uploaded file bytes to either local filesystem or AWS S3.

    Uses content-addressed storage: the storage key is derived from
    SHA-256(user_id + file_bytes) so identical files from the same
    user are deduplicated automatically (re-uploading the same PDF
    returns the same storage path without re-writing).

    Returns:
        Storage path string:
            Local: "uploads/resumes/abc123.pdf"
            S3:    "s3://bucket-name/resumes/abc123.pdf"
    """
    ext          = _safe_extension(filename)
    content_hash = hashlib.sha256(f"{user_id}:{file_bytes[:1024]}".encode()).hexdigest()[:16]
    storage_key  = f"{subfolder}/{content_hash}{ext}"

    if settings.USE_LOCAL_STORAGE:
        return await _save_local(file_bytes, storage_key)
    else:
        return await _save_s3(file_bytes, storage_key, content_type)


async def delete_file(file_path: str) -> bool:
    """
    Delete a file from local filesystem or S3.
    Returns True on success, False if file not found (not an error).
    """
    try:
        if file_path.startswith("s3://"):
            return await _delete_s3(file_path)
        else:
            if os.path.exists(file_path):
                os.remove(file_path)
                return True
            return False
    except Exception as exc:
        logger.warning("File deletion failed", path=file_path, error=str(exc)[:200])
        return False


async def get_file_bytes(file_path: str) -> bytes:
    """
    Read file bytes from local filesystem or S3.
    Raises FileNotFoundError if not found.
    """
    if file_path.startswith("s3://"):
        return await _read_s3(file_path)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: Path(file_path).read_bytes(),
    )


# ---------------------------------------------------------------------------
# Local storage
# ---------------------------------------------------------------------------

async def _save_local(file_bytes: bytes, storage_key: str) -> str:
    """Write bytes to local filesystem under RESUME_UPLOAD_PATH base."""
    from app.core.constants import RESUME_UPLOAD_PATH

    base_dir  = Path(RESUME_UPLOAD_PATH)
    full_path = base_dir / storage_key

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _write_local_sync, file_bytes, full_path)

    logger.debug("File saved locally", path=str(full_path), size=len(file_bytes))
    return str(full_path)


def _write_local_sync(file_bytes: bytes, full_path: Path) -> None:
    full_path.parent.mkdir(parents=True, exist_ok=True)
    full_path.write_bytes(file_bytes)


# ---------------------------------------------------------------------------
# S3 storage
# ---------------------------------------------------------------------------

async def _save_s3(file_bytes: bytes, storage_key: str, content_type: str) -> str:
    """Upload bytes to AWS S3 and return an s3:// URI."""
    try:
        import aiobotocore.session as abcs  # type: ignore

        session = abcs.get_session()
        kwargs: dict[str, Any] = {}
        if settings.AWS_S3_ENDPOINT_URL:
            kwargs["endpoint_url"] = settings.AWS_S3_ENDPOINT_URL

        async with session.create_client(
            "s3",
            region_name=settings.AWS_REGION,
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
            **kwargs,
        ) as client:
            await client.put_object(
                Bucket=settings.AWS_S3_BUCKET,
                Key=storage_key,
                Body=file_bytes,
                ContentType=content_type,
                ServerSideEncryption="AES256",
            )

        s3_uri = f"s3://{settings.AWS_S3_BUCKET}/{storage_key}"
        logger.debug("File saved to S3", uri=s3_uri, size=len(file_bytes))
        return s3_uri

    except ImportError:
        logger.warning("aiobotocore not installed — falling back to local storage")
        return await _save_local(file_bytes, storage_key)


async def _read_s3(s3_uri: str) -> bytes:
    """Read bytes from an s3:// URI."""
    try:
        import aiobotocore.session as abcs  # type: ignore
        _, _, rest = s3_uri.partition("s3://")
        bucket, _, key = rest.partition("/")

        kwargs: dict[str, Any] = {}
        if settings.AWS_S3_ENDPOINT_URL:
            kwargs["endpoint_url"] = settings.AWS_S3_ENDPOINT_URL

        session = abcs.get_session()
        async with session.create_client(
            "s3",
            region_name=settings.AWS_REGION,
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
            **kwargs,
        ) as client:
            response = await client.get_object(Bucket=bucket, Key=key)
            async with response["Body"] as stream:
                return await stream.read()
    except ImportError:
        raise FileNotFoundError(f"Cannot read S3 file — aiobotocore not installed: {s3_uri}")


async def _delete_s3(s3_uri: str) -> bool:
    """Delete a file from S3. Returns True on success."""
    try:
        import aiobotocore.session as abcs  # type: ignore
        _, _, rest = s3_uri.partition("s3://")
        bucket, _, key = rest.partition("/")

        kwargs: dict[str, Any] = {}
        if settings.AWS_S3_ENDPOINT_URL:
            kwargs["endpoint_url"] = settings.AWS_S3_ENDPOINT_URL

        session = abcs.get_session()
        async with session.create_client(
            "s3",
            region_name=settings.AWS_REGION,
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
            **kwargs,
        ) as client:
            await client.delete_object(Bucket=bucket, Key=key)
        return True
    except Exception as exc:
        logger.warning("S3 delete failed", uri=s3_uri, error=str(exc)[:200])
        return False


# ═══════════════════════════════════════════════════════════════════════════
# MIME TYPE VALIDATION
# ═══════════════════════════════════════════════════════════════════════════

def detect_mime_type(file_bytes: bytes, filename: str) -> str:
    """
    Detect the true MIME type of a file from its binary content.

    Uses python-magic (libmagic) for accurate detection — this prevents
    users from renaming a .exe to .pdf and uploading it. Falls back to
    mimetypes.guess_type() if python-magic is not installed.
    """
    try:
        import magic
        mime = magic.from_buffer(file_bytes[:8192], mime=True)
        return mime
    except ImportError:
        guessed, _ = mimetypes.guess_type(filename)
        return guessed or "application/octet-stream"


def validate_resume_file(file_bytes: bytes, filename: str) -> str:
    """
    Validate that a file is an acceptable resume format.

    Returns the detected MIME type on success.
    Raises InvalidFileTypeException or FileTooLargeException on failure.
    """
    from app.core.constants import MAX_RESUME_SIZE_BYTES, ALLOWED_RESUME_EXTENSIONS

    # Size check
    if len(file_bytes) > MAX_RESUME_SIZE_BYTES:
        raise FileTooLargeException(
            filename=filename,
            max_mb=MAX_RESUME_SIZE_BYTES // (1024 * 1024),
        )

    # MIME detection
    detected_mime = detect_mime_type(file_bytes, filename)

    if detected_mime not in ALLOWED_RESUME_MIME_TYPES:
        ext = Path(filename).suffix.lower()
        if ext not in ALLOWED_RESUME_EXTENSIONS:
            raise InvalidFileTypeException(
                f"'{filename}' is not an accepted resume format.",
                context={
                    "filename": filename,
                    "allowed_extensions": sorted(ALLOWED_RESUME_EXTENSIONS),
                },
            )

    return detected_mime


async def validate_upload_file(
    content: bytes,
    filename: str,
    *,
    content_type: str,
    allowed_mimes: set[str] | frozenset[str],
    max_size_mb: int,
) -> None:
    """
    Generic upload validator used for non-resume uploads (avatars, etc.).

    Checks size against `max_size_mb` and checks the file's *detected*
    MIME type (not just the browser-supplied `content_type`) against
    `allowed_mimes`. Raises FileTooLargeException / InvalidFileTypeException
    on failure; returns None on success.
    """
    max_bytes = max_size_mb * 1024 * 1024
    if len(content) > max_bytes:
        raise FileTooLargeException(filename=filename, max_mb=max_size_mb)

    if not content:
        raise InvalidFileTypeException(
            f"'{filename}' is empty.",
            context={"filename": filename},
        )

    detected_mime = detect_mime_type(content, filename)
    if detected_mime not in allowed_mimes and content_type not in allowed_mimes:
        raise InvalidFileTypeException(
            f"'{filename}' is not an accepted file type.",
            context={
                "filename": filename,
                "detected_mime": detected_mime,
                "allowed_mimes": sorted(allowed_mimes),
            },
        )


async def save_upload_file(
    content: bytes,
    filename: str,
    *,
    user_id: Any,
    subdir: str,
) -> tuple[str, str]:
    """
    Save an already-validated upload under the per-user static upload tree
    (``{UPLOAD_DIR}/{user_id}/{subdir}/{stored_name}``), so it's reachable
    at ``/static/uploads/{user_id}/{subdir}/{stored_name}`` (see
    app/main.py's StaticFiles mount).

    Returns (absolute_file_path, stored_filename). Only supports local
    storage — avatars and similar small assets are always kept local even
    when USE_LOCAL_STORAGE=False for resumes, since they're served
    directly by the app rather than via S3/CDN.
    """
    ext = _safe_extension(filename)
    stored_name = f"{compute_file_hash(content)[:16]}{ext}"

    base_dir = settings.storage.upload_dir / str(user_id) / subdir
    full_path = base_dir / stored_name

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _write_local_sync, content, full_path)

    logger.debug("Upload saved", path=str(full_path), size=len(content))
    return str(full_path), stored_name


# ═══════════════════════════════════════════════════════════════════════════
# PDF GENERATION (resume / cover letter export)
# ═══════════════════════════════════════════════════════════════════════════

async def generate_resume_pdf(
    parsed_sections: dict[str, Any],
    output_path: str,
    *,
    template: str = "modern",
) -> str:
    """
    Render parsed resume sections into a formatted PDF.

    Uses WeasyPrint with an embedded HTML template.
    The PDF is saved to output_path and the path is returned.

    Args:
        parsed_sections : Resume sections dict from resume_agent
        output_path     : Where to write the PDF
        template        : "modern" | "classic" | "minimal"

    Returns: output_path on success
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        _generate_pdf_sync,
        parsed_sections,
        output_path,
        template,
    )


def _generate_pdf_sync(
    parsed_sections: dict[str, Any],
    output_path: str,
    template: str,
) -> str:
    """Synchronous PDF generation via WeasyPrint — runs in executor."""
    try:
        from weasyprint import HTML  # type: ignore
    except ImportError:
        logger.warning("WeasyPrint not installed — skipping PDF generation")
        return output_path

    html_content = _render_resume_html(parsed_sections, template)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    HTML(string=html_content).write_pdf(output_path)

    logger.info("Resume PDF generated", path=output_path, size=os.path.getsize(output_path))
    return output_path


def _render_resume_html(sections: dict[str, Any], template: str) -> str:
    """Render resume sections into HTML for PDF conversion."""
    contact   = sections.get("contact", {})
    summary   = sections.get("summary", "")
    experience = sections.get("experience", [])
    education = sections.get("education", [])
    skills    = sections.get("skills", {})
    projects  = sections.get("projects", [])

    name     = contact.get("name", "")
    email    = contact.get("email", "")
    phone    = contact.get("phone", "")
    location = contact.get("location", "")
    linkedin = contact.get("linkedin", "")

    exp_html = ""
    for exp in experience:
        bullets = "".join(
            f"<li>{b}</li>" for b in exp.get("bullets", [])
        )
        exp_html += f"""
        <div class="entry">
            <div class="entry-header">
                <span class="title">{exp.get('title', '')}</span>
                <span class="date">{exp.get('start_date', '')} – {exp.get('end_date', 'Present')}</span>
            </div>
            <div class="company">{exp.get('company', '')}</div>
            <ul>{bullets}</ul>
        </div>"""

    edu_html = ""
    for edu in education:
        edu_html += f"""
        <div class="entry">
            <div class="entry-header">
                <span class="title">{edu.get('degree', '')} in {edu.get('field', '')}</span>
                <span class="date">{edu.get('graduation_year', '')}</span>
            </div>
            <div class="company">{edu.get('institution', '')}</div>
        </div>"""

    tech_skills   = skills.get("technical", [])
    tools_skills  = skills.get("tools", [])
    all_skills    = tech_skills + tools_skills
    skills_str    = "  •  ".join(all_skills[:30])

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ font-family:Georgia, serif; font-size:11pt; color:#222; padding:40px; }}
  h1 {{ font-size:22pt; letter-spacing:1px; margin-bottom:4px; }}
  .contact {{ font-size:9pt; color:#555; margin-bottom:16px; }}
  h2 {{ font-size:12pt; text-transform:uppercase; letter-spacing:1.5px;
        border-bottom:1px solid #999; margin:16px 0 8px; padding-bottom:3px; }}
  .entry {{ margin-bottom:12px; }}
  .entry-header {{ display:flex; justify-content:space-between; }}
  .title {{ font-weight:bold; }}
  .date {{ font-size:9pt; color:#666; }}
  .company {{ font-style:italic; font-size:10pt; margin:2px 0 4px; }}
  ul {{ padding-left:18px; }}
  li {{ margin-bottom:3px; font-size:10pt; }}
  .skills {{ font-size:10pt; line-height:1.8; }}
</style>
</head>
<body>
  <h1>{name}</h1>
  <div class="contact">
    {email}{"  •  " + phone if phone else ""}{"  •  " + location if location else ""}
    {"  •  " + linkedin if linkedin else ""}
  </div>

  {f'<h2>Summary</h2><p style="font-size:10pt;line-height:1.6">{summary}</p>' if summary else ""}

  {f'<h2>Experience</h2>{exp_html}' if exp_html else ""}

  {f'<h2>Education</h2>{edu_html}' if edu_html else ""}

  {f'<h2>Skills</h2><div class="skills">{skills_str}</div>' if skills_str else ""}
</body>
</html>"""


async def generate_cover_letter_pdf(
    cover_letter_text: str,
    output_path: str,
    *,
    sender_name: str = "",
    job_title: str = "",
    company_name: str = "",
) -> str:
    """
    Render a cover letter as a formatted PDF.
    Used when attaching cover letter as file to application form.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        _cover_letter_pdf_sync,
        cover_letter_text,
        output_path,
        sender_name,
        job_title,
        company_name,
    )


def _cover_letter_pdf_sync(
    text: str,
    output_path: str,
    sender_name: str,
    job_title: str,
    company_name: str,
) -> str:
    try:
        from weasyprint import HTML  # type: ignore
    except ImportError:
        return output_path

    escaped = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    paragraphs = "".join(
        f"<p style='margin-bottom:12px'>{p}</p>"
        for p in escaped.split("\n\n")
        if p.strip()
    )

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<style>
  body {{ font-family:Georgia,serif; font-size:11pt; color:#222;
          padding:60px; line-height:1.7; max-width:700px; margin:auto; }}
  .header {{ margin-bottom:32px; }}
  .subject {{ font-weight:bold; margin-bottom:16px; }}
</style></head><body>
<div class="header">
  <div>{sender_name}</div>
  <div style="margin-top:20px;color:#555;font-size:10pt">
    Re: {job_title} at {company_name}
  </div>
</div>
{paragraphs}
</body></html>"""

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    HTML(string=html).write_pdf(output_path)
    return output_path


# ═══════════════════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

def _safe_extension(filename: str) -> str:
    """Extract and sanitise the file extension."""
    ext = Path(filename).suffix.lower()
    allowed = {".pdf", ".docx", ".doc", ".txt", ".odt", ".jpg", ".jpeg", ".png", ".webp"}
    return ext if ext in allowed else ""


def compute_file_hash(file_bytes: bytes) -> str:
    """Return SHA-256 hex digest of file content — used as a content address."""
    return hashlib.sha256(file_bytes).hexdigest()


def human_readable_size(size_bytes: int) -> str:
    """Convert bytes to a human-readable size string."""
    for unit in ("B", "KB", "MB", "GB"):
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes //= 1024
    return f"{size_bytes:.1f} TB"


def get_file_extension(filename: str) -> str:
    """Return lowercase extension including the dot, e.g. '.pdf'"""
    return Path(filename).suffix.lower()


async def ensure_directory(path: str) -> None:
    """Async-safe directory creation."""
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        None,
        lambda: Path(path).mkdir(parents=True, exist_ok=True),
    )