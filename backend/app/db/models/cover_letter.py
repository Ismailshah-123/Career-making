"""
app/db/models/cover_letter.py
==============================
Cover letter model for the JobHunter AI platform.

The cover_letter_agent generates personalised cover letters by cross-referencing:
1. The tailored resume sections (skills, experience, achievements)
2. The job description requirements
3. The company culture signals from the Company model
4. User tone preferences

Each application gets one primary cover letter, optionally with multiple
tone variants (formal / conversational / concise) for A/B testing.

Stores: content, tone, quality scores, generation metadata, file export paths,
keyword coverage analysis, and version history.

Relationships:
- user         → User
- job          → Job
- resume       → Resume
- application  → Application  (back-reference)
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

from sqlalchemy import (
    Boolean,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import BaseModel

if TYPE_CHECKING:
    from app.db.models.user import User
    from app.db.models.job import Job
    from app.db.models.resume import Resume
    from app.db.models.application import Application


class CoverLetter(BaseModel):
    """
    AI-generated cover letter for a specific job application.

    Inherits id (UUID PK), created_at, updated_at, is_deleted, deleted_at.
    """

    __tablename__ = "cover_letters"

    __table_args__ = (
        Index("ix_cover_letters_user_id", "user_id"),
        Index("ix_cover_letters_job_id", "job_id"),
        Index("ix_cover_letters_application_id", "application_id"),
        Index("ix_cover_letters_quality_score", "quality_score"),
        Index("ix_cover_letters_status", "status"),
    )

    # -----------------------------------------------------------------------
    # Ownership & Context
    # -----------------------------------------------------------------------

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="User who owns this cover letter.",
    )

    job_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("jobs.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
        comment="Job this cover letter was generated for.",
    )

    resume_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("resumes.id", ondelete="SET NULL"),
        nullable=True,
        comment="Tailored resume used as source for this cover letter.",
    )

    application_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("applications.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
        comment="Application this cover letter is attached to.",
    )

    # -----------------------------------------------------------------------
    # Content
    # -----------------------------------------------------------------------

    subject_line: Mapped[str | None] = mapped_column(
        String(512),
        nullable=True,
        comment="Email subject line for direct applications, e.g. 'Application for Senior ML Engineer'.",
    )

    salutation: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
        comment="Opening greeting: 'Dear Hiring Team,' or 'Hi Sarah,'.",
    )

    body: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="Full cover letter body text (plain text or Markdown).",
    )

    body_html: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="HTML-formatted body for rich email submissions.",
    )

    closing: Mapped[str | None] = mapped_column(
        String(512),
        nullable=True,
        comment="Closing sentence and sign-off.",
    )

    # -----------------------------------------------------------------------
    # Structure (parsed by the agent for quality scoring)
    # -----------------------------------------------------------------------

    paragraphs: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        server_default="{}",
        comment=(
            "Structured paragraph breakdown: "
            "{ opening: str, why_company: str, why_me: str, "
            "  key_achievements: str, cultural_fit: str, closing: str }"
        ),
    )

    word_count: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="Total word count of the body — enforced 150–500 range.",
    )

    # -----------------------------------------------------------------------
    # Tone & Style
    # -----------------------------------------------------------------------

    tone: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="professional",
        server_default="professional",
        comment="Tone variant: professional | conversational | concise | enthusiastic | formal.",
    )

    is_primary: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        server_default="true",
        comment="True = the main cover letter used in the application. Others are tone variants.",
    )

    version: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=1,
        server_default="1",
        comment="Version number — incremented when the letter is regenerated.",
    )

    # -----------------------------------------------------------------------
    # Keyword Coverage
    # -----------------------------------------------------------------------

    keywords_included: Mapped[list[str]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
        server_default="[]",
        comment="Job keywords found in the cover letter text.",
    )

    keywords_missing: Mapped[list[str]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
        server_default="[]",
        comment="Job keywords absent from the cover letter.",
    )

    keyword_coverage_rate: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment="Fraction of required job keywords covered (0–1).",
    )

    # -----------------------------------------------------------------------
    # Quality Scoring
    # -----------------------------------------------------------------------

    quality_score: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        index=True,
        comment="Overall quality score (0–100) from the cover_letter_agent evaluator.",
    )

    quality_breakdown: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        server_default="{}",
        comment=(
            "Sub-scores: "
            "{ relevance: float, specificity: float, tone_alignment: float, "
            "  ats_score: float, readability: float, keyword_coverage: float }"
        ),
    )

    quality_feedback: Mapped[list[str]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
        server_default="[]",
        comment="List of specific improvement suggestions from the quality evaluator.",
    )

    # -----------------------------------------------------------------------
    # Personalisation Signals Used
    # -----------------------------------------------------------------------

    personalisation_data: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        server_default="{}",
        comment=(
            "Inputs used for personalisation: "
            "{ company_mission: str, recent_news: str, hiring_manager_name: str, "
            "  mutual_connections: [str], tech_stack_overlap: [str] }"
        ),
    )

    # -----------------------------------------------------------------------
    # Status
    # -----------------------------------------------------------------------

    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="draft",
        server_default="draft",
        index=True,
        comment="draft | generated | approved | submitted | archived.",
    )

    is_user_edited: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default="false",
        comment="True if the user has manually edited the content after generation.",
    )

    user_rating: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="User satisfaction rating 1–5 — used to improve future generation prompts.",
    )

    # -----------------------------------------------------------------------
    # File Export
    # -----------------------------------------------------------------------

    pdf_path: Mapped[str | None] = mapped_column(
        String(1024),
        nullable=True,
        comment="S3 / local path to the exported PDF version.",
    )

    docx_path: Mapped[str | None] = mapped_column(
        String(1024),
        nullable=True,
        comment="S3 / local path to the exported DOCX version.",
    )

    # -----------------------------------------------------------------------
    # Generation Metadata
    # -----------------------------------------------------------------------

    generation_prompt: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="Full prompt sent to the LLM — stored for reproducibility and debugging.",
    )

    generation_model: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
        comment="LLM model used, e.g. 'llama-3.3-70b-versatile'.",
    )

    generation_tokens: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="Total tokens used in generation.",
    )

    generation_duration_ms: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="Time taken to generate this cover letter in milliseconds.",
    )

    # -----------------------------------------------------------------------
    # Relationships
    # -----------------------------------------------------------------------

    user: Mapped["User"] = relationship("User", lazy="select")
    job: Mapped["Job | None"] = relationship("Job", back_populates="cover_letters", lazy="select")
    resume: Mapped["Resume | None"] = relationship("Resume", back_populates="cover_letters", lazy="select")
    application: Mapped["Application | None"] = relationship(
        "Application",
        back_populates="cover_letter",
        foreign_keys=[application_id],
        lazy="select",
    )

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    @property
    def is_within_word_limit(self) -> bool:
        from app.core.constants import MIN_COVER_LETTER_WORDS, MAX_COVER_LETTER_WORDS
        if self.word_count is None:
            return True
        return MIN_COVER_LETTER_WORDS <= self.word_count <= MAX_COVER_LETTER_WORDS

    @property
    def is_high_quality(self) -> bool:
        return (self.quality_score or 0) >= 75

    def __repr__(self) -> str:
        return (
            f"<CoverLetter id={self.id} user_id={self.user_id} "
            f"job_id={self.job_id} quality={self.quality_score} tone={self.tone!r}>"
        )