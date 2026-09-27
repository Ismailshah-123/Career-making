"""
CareerGPT — Text Chunking Engine
===================================
PAGE SUMMARY:
  Intelligent text chunking for long documents before embedding.
  Handles: resume sections, job description chunking, sliding window,
  semantic boundary detection, and chunk assembly for retrieval.

  WHY CHUNKING MATTERS:
    sentence-transformers/all-MiniLM-L6-v2 has a 512 token context limit.
    A 5-page resume = ~3,000 tokens → must be chunked for full coverage.
    Bad chunking: split mid-sentence → incoherent chunks → poor matching.
    Good chunking: split at semantic boundaries → coherent chunks → better recall.

  STRATEGIES:
    FIXED_CHAR:   Split every N characters (simple, fast)
    SENTENCE:     Split at sentence boundaries (better coherence)
    PARAGRAPH:    Split at paragraph/section breaks (best for resumes)
    SLIDING_WINDOW: Overlapping chunks for maximum recall
    SECTION_AWARE:  Resume-specific: split by section headers

  RESUME SECTIONS DETECTED:
    Summary / Profile / Objective
    Experience / Work History / Employment
    Education / Academic Background
    Skills / Technical Skills / Competencies
    Projects / Portfolio
    Certifications / Awards
    Languages / Interests

  CHUNK ASSEMBLY FOR RETRIEVAL (RAG):
    For downstream RAG workflows, chunks store their:
    - Source document ID
    - Section name
    - Position in document
    - Overlap with adjacent chunks
    This allows retrieving context around a matching chunk.

  USED BY:
    ResumeService   → chunk long resumes before embedding
    JobScraperAgent → chunk long job descriptions
    Future RAG      → chunk knowledge base documents
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterator


# ══════════════════════════════════════════════════════════════════════════════
# Data Models
# ══════════════════════════════════════════════════════════════════════════════

class ChunkStrategy(str, Enum):
    FIXED_CHAR       = "fixed_char"
    SENTENCE         = "sentence"
    PARAGRAPH        = "paragraph"
    SLIDING_WINDOW   = "sliding_window"
    SECTION_AWARE    = "section_aware"


@dataclass
class TextChunk:
    """A single chunk of text with metadata for retrieval."""
    text:            str
    chunk_index:     int
    total_chunks:    int
    start_char:      int
    end_char:        int
    strategy:        str
    section_name:    str | None     = None
    source_doc_id:   str | None     = None
    char_count:      int            = field(init=False)
    word_count:      int            = field(init=False)
    has_overlap:     bool           = False
    overlap_with:    list[int]      = field(default_factory=list)

    def __post_init__(self) -> None:
        self.char_count = len(self.text)
        self.word_count = len(self.text.split())

    @property
    def is_meaningful(self) -> bool:
        """True if chunk has enough content to embed usefully."""
        return self.word_count >= 5 and self.char_count >= 20

    def to_embed_text(self, *, prepend_section: bool = True) -> str:
        """Build final text for embedding, optionally prepending section name."""
        if prepend_section and self.section_name:
            return f"{self.section_name}:\n{self.text}"
        return self.text


# ══════════════════════════════════════════════════════════════════════════════
# Resume Section Detector
# ══════════════════════════════════════════════════════════════════════════════

RESUME_SECTION_PATTERNS: dict[str, list[str]] = {
    "summary":        ["summary", "profile", "objective", "about", "overview", "bio", "introduction"],
    "experience":     ["experience", "work history", "employment", "work experience", "professional experience", "career history"],
    "education":      ["education", "academic", "qualifications", "degrees", "schooling", "university", "college"],
    "skills":         ["skills", "technical skills", "competencies", "technologies", "stack", "tools", "expertise", "proficiencies"],
    "projects":       ["projects", "portfolio", "personal projects", "open source", "github projects", "side projects"],
    "certifications": ["certifications", "certificates", "awards", "achievements", "honors", "licenses", "accreditations"],
    "languages":      ["languages", "spoken languages", "linguistic skills"],
    "volunteer":      ["volunteer", "volunteering", "community", "nonprofit"],
    "publications":   ["publications", "papers", "research", "articles", "books"],
    "references":     ["references", "referees"],
}

_SECTION_HEADER_RE = re.compile(
    r"^(?:" + "|".join(
        re.escape(pattern)
        for patterns in RESUME_SECTION_PATTERNS.values()
        for pattern in patterns
    ) + r")\s*[:\-–—]?\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def detect_section(line: str) -> str | None:
    """
    Detect which resume section a line header belongs to.
    Returns normalized section name or None if not a section header.
    """
    clean = line.strip().lower().rstrip(":–—-").strip()
    for section_name, patterns in RESUME_SECTION_PATTERNS.items():
        if clean in patterns:
            return section_name
        for pattern in patterns:
            if clean.startswith(pattern) or clean.endswith(pattern):
                return section_name
    return None


# ══════════════════════════════════════════════════════════════════════════════
# Chunker Class
# ══════════════════════════════════════════════════════════════════════════════

class TextChunker:
    """
    Multi-strategy text chunker for document preprocessing.

    Usage:
        chunker = TextChunker()
        chunks = chunker.chunk_resume(raw_text, source_doc_id="resume-uuid")
        embed_texts = [c.to_embed_text() for c in chunks if c.is_meaningful]
    """

    def __init__(
        self,
        *,
        max_chunk_chars:  int   = 1500,
        min_chunk_chars:  int   = 100,
        overlap_chars:    int   = 200,
        sentence_end_re:  str   = r"[.!?]+",
    ) -> None:
        self.max_chunk_chars  = max_chunk_chars
        self.min_chunk_chars  = min_chunk_chars
        self.overlap_chars    = overlap_chars
        self._sentence_re     = re.compile(sentence_end_re)

    # ── Public Interface ──────────────────────────────────────────────────────

    def chunk_resume(
        self,
        text: str,
        *,
        source_doc_id: str | None = None,
    ) -> list[TextChunk]:
        """
        Chunk a resume using section-aware strategy.
        Best for resumes because it preserves section context.
        Each chunk stays within one section.
        """
        return self._section_aware_chunk(
            text,
            source_doc_id=source_doc_id,
        )

    def chunk_job_description(
        self,
        text: str,
        *,
        source_doc_id: str | None = None,
    ) -> list[TextChunk]:
        """
        Chunk a job description using paragraph strategy with sliding window.
        Job descriptions tend to have less defined sections.
        """
        if len(text) <= self.max_chunk_chars:
            return [
                TextChunk(
                    text=text,
                    chunk_index=0,
                    total_chunks=1,
                    start_char=0,
                    end_char=len(text),
                    strategy=ChunkStrategy.FIXED_CHAR.value,
                    source_doc_id=source_doc_id,
                )
            ]

        return self._sliding_window_chunk(
            text,
            source_doc_id=source_doc_id,
        )

    def chunk_document(
        self,
        text: str,
        *,
        strategy: ChunkStrategy = ChunkStrategy.PARAGRAPH,
        source_doc_id: str | None = None,
    ) -> list[TextChunk]:
        """Generic document chunking with configurable strategy."""
        if strategy == ChunkStrategy.FIXED_CHAR:
            return list(self._fixed_char_chunk(text, source_doc_id=source_doc_id))
        elif strategy == ChunkStrategy.SENTENCE:
            return list(self._sentence_chunk(text, source_doc_id=source_doc_id))
        elif strategy == ChunkStrategy.PARAGRAPH:
            return list(self._paragraph_chunk(text, source_doc_id=source_doc_id))
        elif strategy == ChunkStrategy.SLIDING_WINDOW:
            return self._sliding_window_chunk(text, source_doc_id=source_doc_id)
        elif strategy == ChunkStrategy.SECTION_AWARE:
            return self._section_aware_chunk(text, source_doc_id=source_doc_id)
        else:
            return list(self._paragraph_chunk(text, source_doc_id=source_doc_id))

    def get_representative_chunk(
        self,
        text: str,
        *,
        max_chars: int = 4000,
    ) -> str:
        """
        Return the most representative portion of a document for single-vector embedding.
        Prioritizes: summary section > skills section > first N chars.
        Used when we only want ONE vector per document (not chunk-level retrieval).
        """
        if len(text) <= max_chars:
            return text

        # Try to find and prioritize key sections
        sections = self._extract_sections(text)
        priority_sections = ["summary", "skills", "experience"]

        parts: list[str] = []
        chars_used = 0

        for section_name in priority_sections:
            if section_name in sections:
                content = sections[section_name]
                if chars_used + len(content) < max_chars:
                    parts.append(content)
                    chars_used += len(content)

        if chars_used < max_chars // 2:
            remaining = max_chars - chars_used
            parts.append(text[:remaining])

        return "\n\n".join(parts)[:max_chars]

    # ── Strategy Implementations ──────────────────────────────────────────────

    def _fixed_char_chunk(
        self,
        text: str,
        *,
        source_doc_id: str | None = None,
    ) -> Iterator[TextChunk]:
        """Split text into fixed-size character chunks."""
        step = self.max_chunk_chars
        positions = list(range(0, len(text), step))
        total = len(positions)

        for i, start in enumerate(positions):
            end = min(start + step, len(text))
            chunk_text = text[start:end].strip()
            if len(chunk_text) >= self.min_chunk_chars:
                yield TextChunk(
                    text=chunk_text,
                    chunk_index=i,
                    total_chunks=total,
                    start_char=start,
                    end_char=end,
                    strategy=ChunkStrategy.FIXED_CHAR.value,
                    source_doc_id=source_doc_id,
                )

    def _sentence_chunk(
        self,
        text: str,
        *,
        source_doc_id: str | None = None,
    ) -> Iterator[TextChunk]:
        """Split at sentence boundaries."""
        sentences = self._split_sentences(text)
        current: list[str] = []
        current_len = 0
        chunk_index = 0
        start_char = 0
        chunks_list: list[tuple[str, int, int]] = []

        for sentence in sentences:
            if current_len + len(sentence) > self.max_chunk_chars and current:
                chunk_text = " ".join(current)
                end_char = start_char + len(chunk_text)
                chunks_list.append((chunk_text, start_char, end_char))
                start_char = end_char
                current = [sentence]
                current_len = len(sentence)
                chunk_index += 1
            else:
                current.append(sentence)
                current_len += len(sentence)

        if current:
            chunk_text = " ".join(current)
            chunks_list.append((chunk_text, start_char, start_char + len(chunk_text)))

        total = len(chunks_list)
        for idx, (chunk_text, s, e) in enumerate(chunks_list):
            if len(chunk_text) >= self.min_chunk_chars:
                yield TextChunk(
                    text=chunk_text.strip(),
                    chunk_index=idx,
                    total_chunks=total,
                    start_char=s,
                    end_char=e,
                    strategy=ChunkStrategy.SENTENCE.value,
                    source_doc_id=source_doc_id,
                )

    def _paragraph_chunk(
        self,
        text: str,
        *,
        source_doc_id: str | None = None,
    ) -> Iterator[TextChunk]:
        """Split at paragraph breaks (\n\n)."""
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        current: list[str] = []
        current_len = 0
        chunk_index = 0
        all_chunks: list[str] = []

        for para in paragraphs:
            if current_len + len(para) > self.max_chunk_chars and current:
                all_chunks.append("\n\n".join(current))
                current = [para]
                current_len = len(para)
            else:
                current.append(para)
                current_len += len(para)

        if current:
            all_chunks.append("\n\n".join(current))

        total = len(all_chunks)
        char_pos = 0
        for idx, chunk_text in enumerate(all_chunks):
            if len(chunk_text) >= self.min_chunk_chars:
                yield TextChunk(
                    text=chunk_text,
                    chunk_index=idx,
                    total_chunks=total,
                    start_char=char_pos,
                    end_char=char_pos + len(chunk_text),
                    strategy=ChunkStrategy.PARAGRAPH.value,
                    source_doc_id=source_doc_id,
                )
            char_pos += len(chunk_text) + 2  # +2 for \n\n

    def _sliding_window_chunk(
        self,
        text: str,
        *,
        source_doc_id: str | None = None,
    ) -> list[TextChunk]:
        """Overlapping chunks for maximum recall. Good for dense technical text."""
        step = self.max_chunk_chars - self.overlap_chars
        positions = list(range(0, len(text), step))
        total = len(positions)
        chunks: list[TextChunk] = []

        for i, start in enumerate(positions):
            end = min(start + self.max_chunk_chars, len(text))
            chunk_text = text[start:end].strip()
            if len(chunk_text) < self.min_chunk_chars:
                continue

            has_overlap = i > 0
            overlap_with = []
            if i > 0:
                overlap_with.append(i - 1)
            if i < total - 1:
                overlap_with.append(i + 1)

            chunks.append(TextChunk(
                text=chunk_text,
                chunk_index=i,
                total_chunks=total,
                start_char=start,
                end_char=end,
                strategy=ChunkStrategy.SLIDING_WINDOW.value,
                source_doc_id=source_doc_id,
                has_overlap=has_overlap,
                overlap_with=overlap_with,
            ))

        return chunks

    def _section_aware_chunk(
        self,
        text: str,
        *,
        source_doc_id: str | None = None,
    ) -> list[TextChunk]:
        """
        Split resume by detected section headers.
        Each chunk stays within one section.
        Falls back to paragraph chunking within long sections.
        """
        sections = self._extract_sections(text)

        if not sections:
            return list(self._paragraph_chunk(text, source_doc_id=source_doc_id))

        all_chunks: list[TextChunk] = []
        total_sections = len(sections)
        chunk_index = 0

        for section_name, section_text in sections.items():
            if len(section_text) <= self.max_chunk_chars:
                if len(section_text) >= self.min_chunk_chars:
                    all_chunks.append(TextChunk(
                        text=section_text,
                        chunk_index=chunk_index,
                        total_chunks=total_sections,
                        start_char=0,
                        end_char=len(section_text),
                        strategy=ChunkStrategy.SECTION_AWARE.value,
                        section_name=section_name,
                        source_doc_id=source_doc_id,
                    ))
                    chunk_index += 1
            else:
                sub_chunks = list(
                    self._paragraph_chunk(section_text, source_doc_id=source_doc_id)
                )
                for sc in sub_chunks:
                    sc.section_name = section_name
                    sc.chunk_index = chunk_index
                    all_chunks.append(sc)
                    chunk_index += 1

        for chunk in all_chunks:
            chunk.total_chunks = len(all_chunks)

        return all_chunks

    # ── Private Helpers ───────────────────────────────────────────────────────

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences at . ! ? boundaries."""
        parts = self._sentence_re.split(text)
        return [p.strip() for p in parts if p.strip()]

    def _extract_sections(self, text: str) -> dict[str, str]:
        """
        Parse resume text into named sections.
        Returns {section_name: section_text} ordered dict.
        """
        lines = text.split("\n")
        sections: dict[str, str] = {}
        current_section = "header"
        current_lines: list[str] = []

        for line in lines:
            detected = detect_section(line)
            if detected and len(line.strip()) < 60:
                if current_lines:
                    existing = sections.get(current_section, "")
                    sections[current_section] = (
                        (existing + "\n" + "\n".join(current_lines)).strip()
                        if existing else "\n".join(current_lines).strip()
                    )
                current_section = detected
                current_lines = []
            else:
                current_lines.append(line)

        if current_lines:
            existing = sections.get(current_section, "")
            sections[current_section] = (
                (existing + "\n" + "\n".join(current_lines)).strip()
                if existing else "\n".join(current_lines).strip()
            )

        return {k: v for k, v in sections.items() if v.strip()}


# ── Singleton ─────────────────────────────────────────────────────────────────

_chunker: TextChunker | None = None


def get_chunker(**kwargs: int) -> TextChunker:
    """Return module-level TextChunker singleton."""
    global _chunker
    if _chunker is None:
        _chunker = TextChunker(**kwargs)
    return _chunker


__all__ = [
    "TextChunk",
    "TextChunker",
    "ChunkStrategy",
    "detect_section",
    "get_chunker",
    "build_resume_embed_text",
    "build_job_embed_text",
]