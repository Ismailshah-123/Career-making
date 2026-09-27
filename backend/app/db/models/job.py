"""
app/db/models/job.py
====================
Job posting model for the JobHunter AI platform.

Stores every scraped / discovered job posting with full metadata:
board source, compensation, work mode, requirements, company reference,
embedding IDs, match scoring fields, and deduplication fingerprint.

Relationships:
- company         → Company
- applications    → list[Application]
- cover_letters   → list[CoverLetter]  (generated for this job)

Deduplication:
- content_hash (MD5 of title+company+description) prevents re-processing
  the same posting scraped from multiple boards or on multiple runs.

Index strategy:
- job_board + external_id : UNIQUE — prevents duplicate scrapes
- work_mode, job_type, experience_level : for agent pre-filters
- posted_at DESC : chronological listing
- is_active : exclude expired postings
- content_hash : deduplication lookups
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import (
    Boolean,
    DateTime,
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
from app.core.constants import JobBoard, JobType, WorkMode, ExperienceLevel

if TYPE_CHECKING:
    from app.db.models.company import Company
    from app.db.models.application import Application
    from app.db.models.cover_letter import CoverLetter


class Job(BaseModel):
    """
    A job posting discovered by the discovery_agent or submitted manually.

    Inherits id (UUID PK), created_at, updated_at, is_deleted, deleted_at
    from BaseModel.
    """

    __tablename__ = "jobs"

    __table_args__ = (
        UniqueConstraint("job_board", "external_id", name="uq_jobs_board_external"),
        Index("ix_jobs_board", "job_board"),
        Index("ix_jobs_work_mode", "work_mode"),
        Index("ix_jobs_job_type", "job_type"),
        Index("ix_jobs_experience_level", "experience_level"),
        Index("ix_jobs_posted_at", "posted_at"),
        Index("ix_jobs_is_active", "is_active"),
        Index("ix_jobs_company_id", "company_id"),
        Index("ix_jobs_content_hash", "content_hash"),
        Index("ix_jobs_remote_friendly", "is_remote"),
        Index("ix_jobs_salary_range", "salary_min", "salary_max"),
    )

    # -----------------------------------------------------------------------
    # Source / Origin
    # -----------------------------------------------------------------------

    job_board: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        index=True,
        comment="Source board: linkedin | indeed | remoteok | wellfound | direct | other.",
    )

    external_id: Mapped[str | None] = mapped_column(
        String(512),
        nullable=True,
        comment="Unique ID on the source platform (e.g. LinkedIn job ID).",
    )

    source_url: Mapped[str | None] = mapped_column(
        String(2048),
        nullable=True,
        comment="Direct URL to the job posting on the source platform.",
    )

    apply_url: Mapped[str | None] = mapped_column(
        String(2048),
        nullable=True,
        comment="URL of the application form (may differ from source_url for ATS systems).",
    )

    ats_provider: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
        comment="ATS system detected: greenhouse | lever | workday | ashby | taleo | other.",
    )

    content_hash: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
        index=True,
        comment="MD5 of (title + company_name + description) for deduplication across boards.",
    )

    # -----------------------------------------------------------------------
    # Core Posting Fields
    # -----------------------------------------------------------------------

    title: Mapped[str] = mapped_column(
        String(512),
        nullable=False,
        comment="Job title as posted by the employer.",
    )

    description: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="Full job description HTML or plain text.",
    )

    description_cleaned: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="Cleaned / stripped plain-text description used for embedding.",
    )

    # -----------------------------------------------------------------------
    # Company Reference
    # -----------------------------------------------------------------------

    company_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("companys.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
        comment="FK to Company table (may be NULL for unrecognised companies).",
    )

    company_name: Mapped[str | None] = mapped_column(
        String(512),
        nullable=True,
        comment="Denormalised company name for display without a JOIN.",
    )

    company_logo_url: Mapped[str | None] = mapped_column(
        String(1024),
        nullable=True,
        comment="Company logo URL — used in UI job cards.",
    )

    company_website: Mapped[str | None] = mapped_column(
        String(1024),
        nullable=True,
        comment="Company website URL.",
    )

    company_size: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
        comment="Company size band: startup | 1-10 | 11-50 | 51-200 | 201-1000 | 1000+.",
    )

    company_industry: Mapped[str | None] = mapped_column(
        String(256),
        nullable=True,
        comment="Industry / sector of the hiring company.",
    )

    # -----------------------------------------------------------------------
    # Location & Work Mode
    # -----------------------------------------------------------------------

    location: Mapped[str | None] = mapped_column(
        String(512),
        nullable=True,
        comment="Raw location string from the job board.",
    )

    country: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
        comment="ISO 3166-1 alpha-2 country code, e.g. 'US'.",
    )

    city: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
        comment="City component of the job location.",
    )

    timezone_region: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
        comment="Required timezone region for remote roles, e.g. 'EST', 'EU'.",
    )

    work_mode: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default=WorkMode.ONSITE,
        comment="remote | hybrid | onsite.",
    )

    is_remote: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default="false",
        index=True,
        comment="Convenience flag: True if work_mode == 'remote'.",
    )

    # -----------------------------------------------------------------------
    # Job Classification
    # -----------------------------------------------------------------------

    job_type: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default=JobType.FULL_TIME,
        comment="full_time | part_time | contract | freelance | internship.",
    )

    experience_level: Mapped[str | None] = mapped_column(
        String(32),
        nullable=True,
        comment="entry | mid | senior | staff | principal | director | vp | c_level.",
    )

    department: Mapped[str | None] = mapped_column(
        String(256),
        nullable=True,
        comment="Engineering | Product | Design | Marketing | etc.",
    )

    # -----------------------------------------------------------------------
    # Compensation
    # -----------------------------------------------------------------------

    salary_min: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment="Lower bound of annual salary range (in USD).",
    )

    salary_max: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment="Upper bound of annual salary range (in USD).",
    )

    salary_currency: Mapped[str] = mapped_column(
        String(8),
        nullable=False,
        default="USD",
        server_default="USD",
        comment="ISO 4217 currency code.",
    )

    salary_period: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        default="annual",
        server_default="annual",
        comment="annual | monthly | hourly.",
    )

    has_equity: Mapped[bool | None] = mapped_column(
        Boolean,
        nullable=True,
        comment="Whether equity / stock options are mentioned.",
    )

    # -----------------------------------------------------------------------
    # Requirements (structured)
    # -----------------------------------------------------------------------

    required_skills: Mapped[list[str]] = mapped_column(
        ARRAY(String(128)),
        nullable=False,
        default=list,
        server_default="{}",
        comment="Hard-requirement skills extracted by the matching agent.",
    )

    preferred_skills: Mapped[list[str]] = mapped_column(
        ARRAY(String(128)),
        nullable=False,
        default=list,
        server_default="{}",
        comment="Nice-to-have skills extracted from the description.",
    )

    required_experience_years: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment="Minimum years of experience parsed from description.",
    )

    required_education: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
        comment="Required degree level: high_school | bachelor | master | phd | none.",
    )

    languages_required: Mapped[list[str]] = mapped_column(
        ARRAY(String(64)),
        nullable=False,
        default=list,
        server_default="{}",
        comment="Human languages required, e.g. ['English', 'Spanish'].",
    )

    structured_requirements: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        server_default="{}",
        comment=(
            "Full structured extraction: "
            "{ must_have: [str], nice_to_have: [str], "
            "  responsibilities: [str], benefits: [str], "
            "  tech_stack: [str] }"
        ),
    )

    # -----------------------------------------------------------------------
    # Embedding / Vector Store
    # -----------------------------------------------------------------------

    is_embedded: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default="false",
        comment="True once this job's description has been embedded into Qdrant.",
    )

    qdrant_point_id: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
        comment="Qdrant point UUID for this job's embedding.",
    )

    # -----------------------------------------------------------------------
    # Matching Metadata
    # -----------------------------------------------------------------------

    match_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default="0",
        comment="How many times this job has been matched against a user resume.",
    )

    avg_match_score: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment="Running average match score across all matched users.",
    )

    # -----------------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------------

    is_active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        server_default="true",
        index=True,
        comment="False if the posting has been closed or expired.",
    )

    posted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
        comment="When the job was originally posted on the source board.",
    )

    expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="When the posting closes (if known).",
    )

    scraped_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="When this job was collected by the scraping layer.",
    )

    last_seen_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="Most recent scrape run where this job was still visible.",
    )

    # -----------------------------------------------------------------------
    # Extra Metadata
    # -----------------------------------------------------------------------

    tags: Mapped[list[str]] = mapped_column(
        ARRAY(String(64)),
        nullable=False,
        default=list,
        server_default="{}",
        comment="Arbitrary tags added by scrapers or agents for filtering.",
    )

    raw_data: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        server_default="{}",
        comment="Original raw scraped payload — preserved for debugging and reprocessing.",
    )

    notes: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="Internal notes from agents or manual review.",
    )

    # -----------------------------------------------------------------------
    # Relationships
    # -----------------------------------------------------------------------

    company: Mapped["Company | None"] = relationship(
        "Company",
        back_populates="jobs",
        lazy="select",
    )

    applications: Mapped[list["Application"]] = relationship(
        "Application",
        back_populates="job",
        lazy="select",
    )

    cover_letters: Mapped[list["CoverLetter"]] = relationship(
        "CoverLetter",
        back_populates="job",
        lazy="select",
    )

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    @property
    def salary_range_display(self) -> str:
        if self.salary_min and self.salary_max:
            return f"{self.salary_currency} {int(self.salary_min):,} – {int(self.salary_max):,} / {self.salary_period}"
        if self.salary_min:
            return f"From {self.salary_currency} {int(self.salary_min):,}"
        if self.salary_max:
            return f"Up to {self.salary_currency} {int(self.salary_max):,}"
        return "Salary not disclosed"

    @property
    def all_skills(self) -> list[str]:
        return list(dict.fromkeys(self.required_skills + self.preferred_skills))

    def __repr__(self) -> str:
        return (
            f"<Job id={self.id} title={self.title!r} "
            f"board={self.job_board} active={self.is_active}>"
        )