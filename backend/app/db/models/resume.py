"""
app/db/models/resume.py
========================
Resume model for the JobHunter AI platform.

Stores: uploaded file metadata, parsed text content, structured sections
(experience, education, skills, etc.), embedding vector IDs (Qdrant
references), AI tailoring history, ATS scores, and per-application
tailored variants.

One user can have multiple resume versions (master + tailored).
Tailored resumes link back to the master via `parent_resume_id`.

Relationships:
- user           → User
- applications   → list[Application]  (which applications used this resume)
- cover_letters  → list[CoverLetter]
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
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import BaseModel

if TYPE_CHECKING:
    from app.db.models.user import User
    from app.db.models.application import Application
    from app.db.models.cover_letter import CoverLetter


class Resume(BaseModel):
    """
    A resume document owned by a User.

    Master resumes are uploaded by the user (is_master=True).
    Tailored resumes are auto-generated for specific jobs (is_master=False,
    parent_resume_id=<master_id>).
    """

    __tablename__ = "resumes"

    __table_args__ = (
        Index("ix_resumes_user_id", "user_id"),
        Index("ix_resumes_user_master", "user_id", "is_master"),
        Index("ix_resumes_parent", "parent_resume_id"),
        Index("ix_resumes_ats_score", "ats_score"),
    )

    # -----------------------------------------------------------------------
    # Ownership
    # -----------------------------------------------------------------------

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Owner of this resume.",
    )

    # -----------------------------------------------------------------------
    # Version control
    # -----------------------------------------------------------------------

    is_master: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        server_default="true",
        comment="True = original uploaded resume. False = AI-tailored variant.",
    )

    parent_resume_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("resumes.id", ondelete="SET NULL"),
        nullable=True,
        comment="For tailored resumes — points to the master resume.",
    )

    version: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=1,
        server_default="1",
        comment="Monotonically increasing version number within a master → tailored chain.",
    )

    title: Mapped[str] = mapped_column(
        String(256),
        nullable=False,
        default="My Resume",
        comment="User-assigned label, e.g. 'Senior Engineer Resume v2'.",
    )

    # -----------------------------------------------------------------------
    # Stored File
    # -----------------------------------------------------------------------

    original_filename: Mapped[str | None] = mapped_column(
        String(512),
        nullable=True,
        comment="Original upload filename, preserved for display.",
    )

    file_path: Mapped[str | None] = mapped_column(
        String(1024),
        nullable=True,
        comment="Storage path — S3 key or local filesystem path.",
    )

    file_size_bytes: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="Size of the stored file in bytes.",
    )

    mime_type: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
        comment="MIME type of the original file, e.g. 'application/pdf'.",
    )

    # -----------------------------------------------------------------------
    # Parsing Status
    # -----------------------------------------------------------------------

    is_parsed: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default="false",
        comment="True once the resume agent has successfully extracted content.",
    )

    parse_error: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="Error message if parsing failed; NULL on success.",
    )

    # -----------------------------------------------------------------------
    # Extracted Raw Text
    # -----------------------------------------------------------------------

    raw_text: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="Full plain-text extraction of the resume content.",
    )

    word_count: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="Word count of raw_text — used for brevity scoring.",
    )

    # -----------------------------------------------------------------------
    # Structured Sections (parsed by resume_agent)
    # -----------------------------------------------------------------------

    parsed_sections: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        server_default="{}",
        comment=(
            "Structured resume data extracted by the AI. "
            "Schema: { "
            "  contact: {name, email, phone, location, linkedin, github}, "
            "  summary: str, "
            "  experience: [{company, title, start_date, end_date, bullets: [str]}], "
            "  education: [{institution, degree, field, graduation_year, gpa}], "
            "  skills: {technical: [str], soft: [str], tools: [str]}, "
            "  projects: [{name, description, url, technologies: [str]}], "
            "  certifications: [{name, issuer, date, url}], "
            "  languages: [{name, proficiency}] "
            "}"
        ),
    )

    # -----------------------------------------------------------------------
    # Skills & Keywords (denormalised for fast matching)
    # -----------------------------------------------------------------------

    extracted_skills: Mapped[list[str]] = mapped_column(
        ARRAY(String(128)),
        nullable=False,
        default=list,
        server_default="{}",
        comment="Flat list of technical skills extracted by NLP — used for vector search pre-filter.",
    )

    extracted_job_titles: Mapped[list[str]] = mapped_column(
        ARRAY(String(256)),
        nullable=False,
        default=list,
        server_default="{}",
        comment="Job titles held or targeted — used to refine job discovery queries.",
    )

    years_of_experience: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment="Total estimated years of work experience computed by the resume agent.",
    )

    # -----------------------------------------------------------------------
    # Embedding / Vector Store
    # -----------------------------------------------------------------------

    is_embedded: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default="false",
        comment="True once this resume has been embedded and stored in Qdrant.",
    )

    embedding_model: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
        comment="Model used to generate the embedding, e.g. 'text-embedding-3-small'.",
    )

    qdrant_point_ids: Mapped[list[str]] = mapped_column(
        ARRAY(String(64)),
        nullable=False,
        default=list,
        server_default="{}",
        comment="Qdrant point UUIDs for each chunk of this resume.",
    )

    # -----------------------------------------------------------------------
    # ATS & Quality Scoring
    # -----------------------------------------------------------------------

    ats_score: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment="ATS compatibility score (0–100) computed by the resume agent.",
    )

    ats_feedback: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        server_default="{}",
        comment="Structured ATS feedback: {issues: [{type, field, message}], suggestions: [str]}.",
    )

    keyword_density_score: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment="Score (0–1) measuring alignment with target role keywords.",
    )

    # -----------------------------------------------------------------------
    # Tailoring Metadata (for tailored variants only)
    # -----------------------------------------------------------------------

    tailored_for_job_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("jobs.id", ondelete="SET NULL"),
        nullable=True,
        comment="Job this resume was tailored for (NULL for master resumes).",
    )

    tailoring_prompt_used: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="The exact AI prompt used to tailor this resume — stored for auditability.",
    )

    tailoring_changes: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        server_default="{}",
        comment="Diff summary: {added_keywords: [], modified_bullets: [], removed_sections: []}.",
    )

    # -----------------------------------------------------------------------
    # Sharing
    # -----------------------------------------------------------------------

    is_public: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default="false",
        comment="If True, the resume can be accessed via a public share link.",
    )

    share_token: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
        unique=True,
        comment="URL-safe token for public share links.",
    )

    # -----------------------------------------------------------------------
    # Relationships
    # -----------------------------------------------------------------------

    user: Mapped["User"] = relationship(
        "User",
        back_populates="resumes",
        lazy="select",
    )

    applications: Mapped[list["Application"]] = relationship(
        "Application",
        back_populates="resume",
        foreign_keys="Application.resume_id",
        lazy="select",
    )

    cover_letters: Mapped[list["CoverLetter"]] = relationship(
        "CoverLetter",
        back_populates="resume",
        lazy="select",
    )

    # Self-referential: tailored variants of this master
    tailored_versions: Mapped[list["Resume"]] = relationship(
        "Resume",
        foreign_keys=[parent_resume_id],
        lazy="select",
    )

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    @property
    def contact_info(self) -> dict[str, Any]:
        return self.parsed_sections.get("contact", {})

    @property
    def skills_list(self) -> list[str]:
        skills = self.parsed_sections.get("skills", {})
        return (
            skills.get("technical", [])
            + skills.get("soft", [])
            + skills.get("tools", [])
        )

    @property
    def experience_entries(self) -> list[dict[str, Any]]:
        return self.parsed_sections.get("experience", [])

    @property
    def is_ready_for_application(self) -> bool:
        """True if the resume is fully parsed and embedded."""
        return self.is_parsed and self.is_embedded and not self.parse_error

    def __repr__(self) -> str:
        return (
            f"<Resume id={self.id} user_id={self.user_id} "
            f"is_master={self.is_master} version={self.version}>"
        )