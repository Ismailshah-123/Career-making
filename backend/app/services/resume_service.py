"""
CareerGPT — Resume Service
============================
PAGE SUMMARY:
  Core resume processing engine. Orchestrates the full CV lifecycle.
  Handles: file upload + validation, text extraction (PDF/DOCX),
  AI-powered structured parsing, ATS scoring, vector embedding + Qdrant storage,
  resume tailoring for specific jobs (calls ResumeAgent), analysis reports,
  PDF generation of tailored resumes, download URL signing, and bulk operations.

  USED BY: app/api/v1/resumes.py (all resume endpoints)
  USES:    ResumeRepository, JobRepository, EmbeddingService, QdrantService,
           GroqService (AI parsing + tailoring), ResumeAgent (tailor workflow)

  KEY OPERATIONS:
    upload_master_resume() → validate → extract text → AI parse → ATS score
                           → embed → store Qdrant → save DB record
    tailor_for_job()       → load master + job → ResumeAgent.tailor()
                           → save tailored record → update application
    analyze_resume()       → AI deep analysis → return scored report
    get_ats_score()        → fast ATS compatibility check
    export_resume_pdf()    → generate clean PDF from tailored data
    delete_resume()        → soft delete DB + soft delete Qdrant point

  TEXT EXTRACTION PIPELINE:
    PDF  → pdfplumber (primary, layout-aware) → PyMuPDF (fallback for scanned)
    DOCX → python-docx (paragraphs + tables)
    Both → clean whitespace → truncate to RESUME_EXTRACT_CHAR_LIMIT

  ATS SCORING (0-100):
    25 pts → Standard sections present (Experience, Education, Skills, Contact)
    25 pts → Keyword density and relevance
    20 pts → Formatting signals (no tables detected, clean structure)
    15 pts → Action verbs + quantified achievements
    10 pts → Word count in optimal range (400-800)
     5 pts → Complete contact info (email, phone, LinkedIn)
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.constants import (
    ALLOWED_RESUME_MIMES,
    RESUME_EMBED_CHAR_LIMIT,
    RESUME_EXTRACT_CHAR_LIMIT,
    RESUME_MAX_SIZE_MB,
    UPLOAD_DIR_RESUMES,
    UPLOAD_DIR_TAILORED,
    OptimizationLevel,
)
from app.core.exceptions import (
    FileTooLargeError,
    InvalidFileTypeError,
    ResumeExportError,
    ResumeNotFoundError,
    ResumeParseError,
    ResumeTailorError,
)
from app.core.logging import log_context, logger
from app.repositories.resume_repository import ResumeRepository
from app.services.embedding_service import get_embedding_service
from app.services.groq_service import GroqService, get_groq_service
from app.services.qdrant_service import QdrantService, get_qdrant_service

settings = get_settings()


class ResumeService:
    """
    Resume processing service.
    All resume-related business logic lives here.
    """

    def __init__(self, db: AsyncSession) -> None:
        self.db = db
        self.resume_repo = ResumeRepository(db)
        self.llm: GroqService = get_groq_service()
        self.embedder = get_embedding_service()
        self.qdrant: QdrantService = get_qdrant_service()

    # ── Upload & Processing ───────────────────────────────────────────────────

    async def upload_master_resume(
        self,
        *,
        user_id: uuid.UUID,
        file_content: bytes,
        original_filename: str,
        content_type: str,
    ) -> dict[str, Any]:
        """
        Full master resume upload pipeline.

        Steps:
          1. Validate file type and size
          2. Save raw file to disk
          3. Extract full text (PDF/DOCX)
          4. AI-parse structured fields (name, email, skills, etc.)
          5. Calculate ATS score
          6. Generate embedding vector
          7. Upsert to Qdrant
          8. Deactivate previous master resume (only one master per user)
          9. Save new Resume record to DB

        Returns complete resume dict ready for API response.
        """
        with log_context(user_id=str(user_id), filename=original_filename):
            # ── 1. Validate ───────────────────────────────────────────────────
            await self._validate_file(file_content, content_type, original_filename)

            # ── 2. Save file ──────────────────────────────────────────────────
            file_path, stored_name = await self._save_file(
                user_id=user_id,
                content=file_content,
                original_filename=original_filename,
                subdir=UPLOAD_DIR_RESUMES,
            )

            # ── 3. Extract text ───────────────────────────────────────────────
            raw_text = await self._extract_text(file_path, content_type)
            if not raw_text.strip():
                raise ResumeParseError(
                    "No text could be extracted from the file. "
                    "Ensure it is not a scanned image without OCR.",
                    context={"filename": original_filename},
                )

            # ── 4. AI Parse ───────────────────────────────────────────────────
            parsed = await self._ai_parse_resume(raw_text)

            # ── 5. ATS Score ──────────────────────────────────────────────────
            ats_score = await self._calculate_ats_score(raw_text)

            # ── 6. Embed ──────────────────────────────────────────────────────
            embed_text = self._build_embed_text(raw_text, parsed)
            vector = await self.embedder.embed(embed_text)

            # ── 7. Qdrant upsert ──────────────────────────────────────────────
            qdrant_payload = {
                "user_id":          str(user_id),
                "is_master":        True,
                "skills":           " ".join(parsed.get("skills_list", [])),
                "experience_years": parsed.get("experience_years"),
                "ats_score":        ats_score,
                "education_level":  parsed.get("education_level"),
            }
            qdrant_point_id = await self.qdrant.upsert_resume(
                uuid.UUID(str(user_id)),  # use user_id as placeholder
                vector,
                qdrant_payload,
            )

            # ── 8. Deactivate old masters ──────────────────────────────────────
            old_masters = await self.resume_repo.get_master_resumes(user_id)
            for old in old_masters:
                await self.resume_repo.update(old.id, is_master=False)
                if old.qdrant_point_id:
                    await self.qdrant.soft_delete_resume(old.qdrant_point_id)

            # ── 9. Create DB record ───────────────────────────────────────────
            resume = await self.resume_repo.create(
                user_id=user_id,
                original_filename=original_filename,
                stored_filename=stored_name,
                file_path=str(file_path),
                file_size_bytes=len(file_content),
                mime_type=content_type,
                is_master=True,
                raw_text=raw_text,
                ats_score=ats_score,
                qdrant_point_id=qdrant_point_id,
                embedding_model=settings.llm.embedding_model,
                **{k: v for k, v in parsed.items() if k != "skills_list"},
            )

            logger.info(
                "Master resume uploaded and processed",
                user_id=str(user_id),
                resume_id=str(resume.id),
                ats_score=ats_score,
                experience_years=parsed.get("experience_years"),
            )

            return self._resume_to_dict(resume)

    # ── AI Resume Analysis ────────────────────────────────────────────────────

    async def analyze_resume(
        self,
        *,
        resume_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> dict[str, Any]:
        """
        Deep AI analysis of a resume. Returns comprehensive quality report.

        Analyzes:
          - Overall quality score (0-100)
          - ATS compatibility score
          - Impact score (quantified achievements)
          - Readability score
          - Strengths (specific, not generic)
          - Improvements (actionable)
          - Missing sections
          - Top skills detected
          - Recommended target roles
          - Keyword density for top job categories
        """
        resume = await self.resume_repo.get_by_id_or_raise(resume_id)

        if resume.user_id != user_id:
            raise ResumeNotFoundError()

        if not resume.raw_text:
            raise ResumeParseError("Resume has no extracted text. Please re-upload.")

        from app.prompts.resume_prompts import RESUME_ANALYZE_PROMPT

        system, user_msg = RESUME_ANALYZE_PROMPT.render(
            resume_text=resume.raw_text[:8000]
        )

        result = await self.llm.complete_json(
            prompt=user_msg,
            system=system,
            temperature=0.15,
            max_tokens=1500,
        )

        # Recalculate ATS score live (always fresh)
        fresh_ats = await self._calculate_ats_score(resume.raw_text)

        # Update DB with fresh score
        if abs((resume.ats_score or 0) - fresh_ats) > 2:
            await self.resume_repo.update(resume_id, ats_score=fresh_ats)

        return {
            "resume_id":           str(resume_id),
            "overall_score":       float(result.get("overall_score", fresh_ats)),
            "ats_score":           fresh_ats,
            "impact_score":        float(result.get("impact_score", 50)),
            "readability_score":   float(result.get("readability_score", 70)),
            "strengths":           result.get("strengths", []),
            "improvements":        result.get("improvements", []),
            "missing_sections":    result.get("missing_sections", []),
            "top_skills":          result.get("top_skills", []),
            "experience_years":    resume.experience_years or result.get("experience_years", 0),
            "education_level":     resume.education_level or result.get("education_level"),
            "word_count":          len(resume.raw_text.split()),
            "recommended_roles":   result.get("recommended_roles", []),
            "analyzed_at":         datetime.now(UTC).isoformat(),
        }

    async def get_ats_score(
        self,
        *,
        resume_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> dict[str, Any]:
        """Quick ATS score endpoint — faster than full analyze."""
        resume = await self.resume_repo.get_by_id_or_raise(resume_id)
        if resume.user_id != user_id:
            raise ResumeNotFoundError()
        if not resume.raw_text:
            raise ResumeParseError("No text available for scoring.")

        score = await self._calculate_ats_score(resume.raw_text)
        await self.resume_repo.update(resume_id, ats_score=score)

        return {
            "resume_id":  str(resume_id),
            "ats_score":  score,
            "grade":      self._score_to_grade(score),
            "scored_at":  datetime.now(UTC).isoformat(),
        }

    # ── Tailoring ─────────────────────────────────────────────────────────────

    async def tailor_for_job(
        self,
        *,
        user_id: uuid.UUID,
        job_id: uuid.UUID,
        master_resume_id: uuid.UUID | None = None,
        optimization_level: str = OptimizationLevel.AGGRESSIVE.value,
    ) -> dict[str, Any]:
        """
        AI-tailor master resume for a specific job.
        Delegates heavy lifting to ResumeAgent (agent.py).
        Saves tailored resume PDF + DB record.
        Updates application match_score if application exists.
        """
        from app.agents.resume_agent.agent import ResumeAgent
        from app.repositories.job_repository import JobRepository

        job_repo = JobRepository(self.db)

        # Load master resume
        if master_resume_id:
            master = await self.resume_repo.get_by_id_or_raise(master_resume_id)
            if master.user_id != user_id:
                raise ResumeNotFoundError()
        else:
            masters = await self.resume_repo.get_master_resumes(user_id)
            if not masters:
                raise ResumeNotFoundError(
                    "No master resume found. Please upload your CV first."
                )
            master = masters[0]

        if not master.raw_text:
            raise ResumeParseError("Master resume has no extracted text.")

        # Load job
        job = await job_repo.get_by_id_or_raise(job_id)

        logger.info(
            "Starting resume tailoring",
            user_id=str(user_id),
            job=f"{job.title} @ {job.company}",
            level=optimization_level,
        )

        # Call the ResumeAgent
        agent = ResumeAgent(self.db)
        result = await agent.tailor(
            master_resume=master,
            job=job,
            optimization_level=optimization_level,
        )

        # Save tailored resume record
        tailored_path = result.get("pdf_path", "")
        stored_name = Path(tailored_path).name if tailored_path else f"tailored_{uuid.uuid4().hex}.pdf"

        tailored_resume = await self.resume_repo.create(
            user_id=user_id,
            original_filename=f"resume_{job.company.lower().replace(' ', '_')}.pdf",
            stored_filename=stored_name,
            file_path=tailored_path,
            file_size_bytes=Path(tailored_path).stat().st_size if tailored_path and Path(tailored_path).exists() else 0,
            mime_type="application/pdf",
            is_master=False,
            tailored_for_job_id=job_id,
            optimization_level=optimization_level,
            raw_text=result.get("tailored_text", ""),
            ats_score=result.get("ats_score", 0.0),
            skills=json.dumps(result.get("skills_section", [])),
        )

        # Update application record if it exists
        try:
            from app.repositories.application_repository import ApplicationRepository
            app_repo = ApplicationRepository(self.db)
            app = await app_repo.get_by_user_and_job(user_id, job_id)
            if app:
                await app_repo.update(
                    app.id,
                    resume_id=tailored_resume.id,
                    tailored_resume_path=tailored_path,
                    match_score=result.get("match_score", 0.0),
                    match_analysis=json.dumps({
                        "keywords_added":   result.get("keywords_added", []),
                        "keywords_missing": result.get("keywords_missing", []),
                        "ats_score":        result.get("ats_score", 0.0),
                        "improvement_summary": result.get("improvement_summary", ""),
                    }),
                )
        except Exception as exc:
            logger.warning("Application update after tailor failed (non-critical)", error=str(exc))

        logger.info(
            "Resume tailored successfully",
            resume_id=str(tailored_resume.id),
            match_score=result.get("match_score"),
            ats_score=result.get("ats_score"),
        )

        return {
            "resume_id":        str(tailored_resume.id),
            "job_id":           str(job_id),
            "match_score":      result.get("match_score", 0.0),
            "ats_score":        result.get("ats_score", 0.0),
            "keywords_added":   result.get("keywords_added", []),
            "keywords_missing": result.get("keywords_missing", []),
            "improvement_summary": result.get("improvement_summary", ""),
            "download_url":     f"/api/v1/resumes/{tailored_resume.id}/download",
            "tailored_at":      datetime.now(UTC).isoformat(),
        }

    # ── List & Delete ─────────────────────────────────────────────────────────

    async def list_user_resumes(
        self,
        user_id: uuid.UUID,
        *,
        master_only: bool = False,
        skip: int = 0,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """List all resumes for a user with pagination."""
        resumes = await self.resume_repo.get_user_resumes(
            user_id,
            master_only=master_only,
            skip=skip,
            limit=limit,
        )
        return [self._resume_to_dict(r) for r in resumes]

    async def delete_resume(
        self,
        *,
        resume_id: uuid.UUID,
        user_id: uuid.UUID,
        hard: bool = False,
    ) -> dict[str, str]:
        """
        Delete a resume. Soft-delete by default (GDPR: use hard=True for erasure).
        Also removes Qdrant point.
        """
        resume = await self.resume_repo.get_by_id_or_raise(resume_id)
        if resume.user_id != user_id:
            raise ResumeNotFoundError()

        # Remove from Qdrant
        if resume.qdrant_point_id:
            if hard:
                await self.qdrant.hard_delete_resume(resume.qdrant_point_id)
            else:
                await self.qdrant.soft_delete_resume(resume.qdrant_point_id)

        # Remove file from disk (only on hard delete)
        if hard and resume.file_path:
            try:
                Path(resume.file_path).unlink(missing_ok=True)
            except Exception as exc:
                logger.warning("File deletion failed", path=resume.file_path, error=str(exc))

        await self.resume_repo.delete(resume_id, hard=hard)

        logger.info(
            "Resume deleted",
            resume_id=str(resume_id),
            user_id=str(user_id),
            hard=hard,
        )

        return {"message": "Resume deleted successfully."}

    async def get_download_path(
        self,
        *,
        resume_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> tuple[str, str]:
        """
        Return (file_path, filename) for resume download.
        Validates ownership. Raises ResumeNotFoundError if missing/wrong user.
        """
        resume = await self.resume_repo.get_by_id_or_raise(resume_id)
        if resume.user_id != user_id:
            raise ResumeNotFoundError()

        if not resume.file_path or not Path(resume.file_path).exists():
            raise ResumeNotFoundError("Resume file not found on disk. Please re-upload.")

        return resume.file_path, resume.original_filename

    # ── Private: Text Extraction ──────────────────────────────────────────────

    async def _extract_text(self, file_path: Path, content_type: str) -> str:
        """
        Extract full text from PDF or DOCX.
        PDF:  pdfplumber (primary) → PyMuPDF (fallback for scanned)
        DOCX: python-docx
        """
        suffix = file_path.suffix.lower()

        if suffix == ".pdf" or "pdf" in content_type:
            return await self._extract_pdf(file_path)
        elif suffix in (".docx", ".doc") or "word" in content_type:
            return await self._extract_docx(file_path)
        else:
            raise InvalidFileTypeError(f"Unsupported file type: {suffix}")

    async def _extract_pdf(self, file_path: Path) -> str:
        """Extract text from PDF using pdfplumber with PyMuPDF fallback."""
        try:
            import pdfplumber

            pages: list[str] = []
            with pdfplumber.open(str(file_path)) as pdf:
                for page in pdf.pages:
                    text = page.extract_text(
                        x_tolerance=3,
                        y_tolerance=3,
                        layout=True,
                        x_density=7.25,
                        y_density=13,
                    )
                    if text:
                        pages.append(text)

            full_text = "\n\n".join(pages).strip()

            if not full_text:
                logger.info("pdfplumber extracted empty text, trying PyMuPDF")
                full_text = await self._extract_pdf_pymupdf(file_path)

            logger.debug(
                "PDF text extracted",
                path=str(file_path),
                pages=len(pages),
                chars=len(full_text),
            )
            return full_text[:RESUME_EXTRACT_CHAR_LIMIT]

        except ImportError:
            logger.warning("pdfplumber not available, falling back to PyMuPDF")
            return await self._extract_pdf_pymupdf(file_path)
        except Exception as exc:
            raise ResumeParseError(
                f"PDF extraction failed: {exc}",
                context={"file": str(file_path)},
            ) from exc

    async def _extract_pdf_pymupdf(self, file_path: Path) -> str:
        """PyMuPDF fallback for PDFs that pdfplumber can't read."""
        try:
            import fitz  # PyMuPDF

            doc = fitz.open(str(file_path))
            pages = []
            for page in doc:
                text = page.get_text("text", sort=True)
                if text.strip():
                    pages.append(text)
            doc.close()

            full_text = "\n\n".join(pages)
            logger.debug("PyMuPDF extraction", chars=len(full_text))
            return full_text[:RESUME_EXTRACT_CHAR_LIMIT]

        except ImportError as exc:
            raise ResumeParseError("Neither pdfplumber nor PyMuPDF is installed.") from exc
        except Exception as exc:
            raise ResumeParseError(f"PyMuPDF extraction failed: {exc}") from exc

    async def _extract_docx(self, file_path: Path) -> str:
        """Extract text from DOCX preserving paragraph + table order."""
        try:
            from docx import Document

            doc = Document(str(file_path))
            parts: list[str] = []

            for para in doc.paragraphs:
                text = para.text.strip()
                if text:
                    parts.append(text)

            for table in doc.tables:
                for row in table.rows:
                    cells = [c.text.strip() for c in row.cells if c.text.strip()]
                    if cells:
                        parts.append(" | ".join(cells))

            full_text = "\n".join(parts)
            logger.debug("DOCX extracted", chars=len(full_text))
            return full_text[:RESUME_EXTRACT_CHAR_LIMIT]

        except ImportError as exc:
            raise ResumeParseError("python-docx not installed. Run: pip install python-docx") from exc
        except Exception as exc:
            raise ResumeParseError(f"DOCX extraction failed: {exc}") from exc

    # ── Private: AI Parsing ───────────────────────────────────────────────────

    async def _ai_parse_resume(self, raw_text: str) -> dict[str, Any]:
        """
        Use LLM to extract structured fields from raw resume text.
        Returns dict of ORM-compatible fields.
        """
        from app.prompts.resume_prompts import RESUME_PARSE_PROMPT

        system, user_msg = RESUME_PARSE_PROMPT.render(
            resume_text=raw_text[:RESUME_EXTRACT_CHAR_LIMIT]
        )

        try:
            parsed = await self.llm.complete_json(
                prompt=user_msg,
                system=system,
                temperature=0.05,
                max_tokens=1200,
            )
        except Exception as exc:
            logger.warning("AI resume parsing failed, using basic extraction", error=str(exc))
            return self._basic_extract(raw_text)

        skills_list: list[str] = parsed.get("skills", []) or []

        return {
            "name":             parsed.get("name"),
            "email":            parsed.get("email"),
            "phone":            parsed.get("phone"),
            "location":         parsed.get("location"),
            "linkedin_url":     parsed.get("linkedin_url"),
            "github_url":       parsed.get("github_url"),
            "portfolio_url":    parsed.get("portfolio_url"),
            "summary":          parsed.get("summary"),
            "skills":           json.dumps(skills_list),
            "soft_skills":      json.dumps(parsed.get("soft_skills", [])),
            "certifications":   json.dumps(parsed.get("certifications", [])),
            "languages":        json.dumps(parsed.get("languages", ["English"])),
            "experience_years": parsed.get("experience_years"),
            "education_level":  parsed.get("education_level"),
            "skills_list":      skills_list,   # for embedding (not stored in DB)
        }

    async def _calculate_ats_score(self, raw_text: str) -> float:
        """Calculate ATS compatibility score 0-100 using LLM."""
        from app.prompts.resume_prompts import RESUME_ATS_SCORE_PROMPT

        system, user_msg = RESUME_ATS_SCORE_PROMPT.render(
            resume_text=raw_text[:6000]
        )

        try:
            result = await self.llm.complete_json(
                prompt=user_msg,
                system=system,
                temperature=0.05,
                max_tokens=400,
            )
            raw_score = float(result.get("score", 60))
            return max(0.0, min(100.0, raw_score))
        except Exception as exc:
            logger.warning("ATS scoring failed, using default", error=str(exc))
            return 60.0

    # ── Private: File Operations ──────────────────────────────────────────────

    async def _validate_file(
        self,
        content: bytes,
        content_type: str,
        filename: str,
    ) -> None:
        """Validate file type and size. Raises typed exceptions."""
        if content_type not in ALLOWED_RESUME_MIMES:
            # Also check by extension as fallback
            ext = Path(filename).suffix.lower()
            if ext not in (".pdf", ".doc", ".docx"):
                raise InvalidFileTypeError(
                    context={"content_type": content_type, "extension": ext}
                )

        max_bytes = RESUME_MAX_SIZE_MB * 1024 * 1024
        if len(content) > max_bytes:
            raise FileTooLargeError(
                context={
                    "size_mb":   round(len(content) / 1024 / 1024, 2),
                    "max_mb":    RESUME_MAX_SIZE_MB,
                }
            )

    async def _save_file(
        self,
        *,
        user_id: uuid.UUID,
        content: bytes,
        original_filename: str,
        subdir: str,
    ) -> tuple[Path, str]:
        """Save uploaded file to disk. Returns (absolute_path, stored_filename)."""
        ext = Path(original_filename).suffix.lower() or ".pdf"
        stored_name = f"{uuid.uuid4().hex}{ext}"

        user_dir = settings.storage.upload_dir / str(user_id) / subdir
        user_dir.mkdir(parents=True, exist_ok=True)

        file_path = user_dir / stored_name
        file_path.write_bytes(content)

        logger.debug(
            "File saved",
            path=str(file_path),
            size=len(content),
        )
        return file_path, stored_name

    # ── Private: Utilities ────────────────────────────────────────────────────

    @staticmethod
    def _build_embed_text(raw_text: str, parsed: dict[str, Any]) -> str:
        """Build optimized text for embedding (skills + summary weighted)."""
        skills = parsed.get("skills_list", [])
        summary = parsed.get("summary", "")
        # Repeat skills 3x to boost their semantic weight
        skill_block = " ".join(skills * 3)
        return f"{summary}\n\n{skill_block}\n\n{raw_text}"[:RESUME_EMBED_CHAR_LIMIT]

    @staticmethod
    def _basic_extract(raw_text: str) -> dict[str, Any]:
        """Fallback extraction without AI: simple regex-based field detection."""
        import re

        email_match = re.search(r"[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}", raw_text)
        phone_match = re.search(r"[\+]?[\d\s\-\(\)]{10,15}", raw_text)
        linkedin_match = re.search(r"linkedin\.com/in/[\w\-]+", raw_text)

        return {
            "name":             None,
            "email":            email_match.group() if email_match else None,
            "phone":            phone_match.group().strip() if phone_match else None,
            "location":         None,
            "linkedin_url":     f"https://{linkedin_match.group()}" if linkedin_match else None,
            "github_url":       None,
            "portfolio_url":    None,
            "summary":          None,
            "skills":           json.dumps([]),
            "soft_skills":      json.dumps([]),
            "certifications":   json.dumps([]),
            "languages":        json.dumps(["English"]),
            "experience_years": None,
            "education_level":  None,
            "skills_list":      [],
        }

    @staticmethod
    def _score_to_grade(score: float) -> str:
        if score >= 90:  return "A+"
        if score >= 80:  return "A"
        if score >= 70:  return "B"
        if score >= 60:  return "C"
        if score >= 50:  return "D"
        return "F"

    @staticmethod
    def _resume_to_dict(resume: Any) -> dict[str, Any]:
        """Safe serialization of Resume ORM object."""
        def _parse_json(val: str | None) -> list:
            if not val:
                return []
            try:
                return json.loads(val) if isinstance(val, str) else val
            except Exception:
                return []

        return {
            "id":               str(resume.id),
            "user_id":          str(resume.user_id),
            "original_filename": resume.original_filename,
            "is_master":        resume.is_master,
            "tailored_for_job_id": str(resume.tailored_for_job_id) if resume.tailored_for_job_id else None,
            "optimization_level": resume.optimization_level,
            "file_size_bytes":  resume.file_size_bytes,
            "mime_type":        resume.mime_type,
            "name":             resume.name,
            "email":            resume.email,
            "phone":            resume.phone,
            "location":         resume.location,
            "linkedin_url":     resume.linkedin_url,
            "github_url":       resume.github_url,
            "summary":          resume.summary,
            "skills":           _parse_json(resume.skills),
            "certifications":   _parse_json(resume.certifications),
            "languages":        _parse_json(resume.languages),
            "experience_years": resume.experience_years,
            "education_level":  resume.education_level,
            "ats_score":        resume.ats_score,
            "created_at":       resume.created_at.isoformat() if resume.created_at else None,
        }


__all__ = ["ResumeService"]