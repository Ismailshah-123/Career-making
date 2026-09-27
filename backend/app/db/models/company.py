"""
app/db/models/company.py
=========================
Company entity for the JobHunter AI platform.

Stores firmographic data enriched by the discovery agent:
size, industry, tech stack, culture signals, Glassdoor/LinkedIn scores,
funding stage, and aggregated hiring pattern metadata.

Companies are shared across users — one company row, many job rows.
The discovery and matching agents reference company data to:
1. Filter out user-blacklisted companies
2. Enrich job cards with culture / compensation signals
3. Personalise outreach messages with company context

Relationships:
- jobs        → list[Job]
- recruiters  → list[Recruiter]
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sqlalchemy import (
    Boolean,
    Float,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import BaseModel

if TYPE_CHECKING:
    from app.db.models.job import Job
    from app.db.models.recruiter import Recruiter


class Company(BaseModel):
    """
    Hiring company entity — enriched by the discovery agent.

    Inherits id (UUID PK), created_at, updated_at, is_deleted, deleted_at.
    """

    __tablename__ = "companys"  # Matches auto __tablename__ from BaseModel

    __table_args__ = (
        UniqueConstraint("name", "website", name="uq_company_name_website"),
        Index("ix_company_name", "name"),
        Index("ix_company_industry", "industry"),
        Index("ix_company_size_band", "size_band"),
        Index("ix_company_is_verified", "is_verified"),
        Index("ix_company_hiring_score", "hiring_velocity_score"),
    )

    # -----------------------------------------------------------------------
    # Identity
    # -----------------------------------------------------------------------

    name: Mapped[str] = mapped_column(
        String(512),
        nullable=False,
        index=True,
        comment="Legal or commonly used company name.",
    )

    slug: Mapped[str | None] = mapped_column(
        String(256),
        nullable=True,
        unique=True,
        comment="URL-friendly slug, e.g. 'stripe' or 'openai'. Auto-generated.",
    )

    website: Mapped[str | None] = mapped_column(
        String(1024),
        nullable=True,
        comment="Primary company website URL.",
    )

    logo_url: Mapped[str | None] = mapped_column(
        String(1024),
        nullable=True,
        comment="CDN URL of the company logo.",
    )

    description: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="Brief company description — used in outreach personalisation.",
    )

    tagline: Mapped[str | None] = mapped_column(
        String(512),
        nullable=True,
        comment="Company mission / tagline from their website or LinkedIn.",
    )

    is_verified: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default="false",
        comment="True once a human or high-confidence source has verified the data.",
    )

    # -----------------------------------------------------------------------
    # Firmographic Data
    # -----------------------------------------------------------------------

    industry: Mapped[str | None] = mapped_column(
        String(256),
        nullable=True,
        index=True,
        comment="Primary industry, e.g. 'FinTech', 'HealthTech', 'SaaS'.",
    )

    sub_industry: Mapped[str | None] = mapped_column(
        String(256),
        nullable=True,
        comment="Sub-industry or vertical, e.g. 'Payments', 'EHR', 'DevTools'.",
    )

    size_band: Mapped[str | None] = mapped_column(
        String(32),
        nullable=True,
        index=True,
        comment="Employee count band: 1-10 | 11-50 | 51-200 | 201-1000 | 1001-5000 | 5000+.",
    )

    employee_count_exact: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="Exact employee count if known (e.g. from LinkedIn).",
    )

    founded_year: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="Year the company was founded.",
    )

    headquarters: Mapped[str | None] = mapped_column(
        String(256),
        nullable=True,
        comment="HQ city/country, e.g. 'San Francisco, CA, USA'.",
    )

    country: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
        comment="ISO 3166-1 alpha-2 country code of the HQ.",
    )

    is_public: Mapped[bool | None] = mapped_column(
        Boolean,
        nullable=True,
        comment="True if the company is publicly listed.",
    )

    stock_ticker: Mapped[str | None] = mapped_column(
        String(16),
        nullable=True,
        comment="Stock ticker symbol for public companies, e.g. 'MSFT'.",
    )

    # -----------------------------------------------------------------------
    # Funding / Stage
    # -----------------------------------------------------------------------

    funding_stage: Mapped[str | None] = mapped_column(
        String(32),
        nullable=True,
        comment="Funding stage: bootstrapped | pre-seed | seed | series-a | ... | ipo | acquired.",
    )

    total_funding_usd: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment="Total funding raised in USD (millions).",
    )

    last_funding_round: Mapped[str | None] = mapped_column(
        String(16),
        nullable=True,
        comment="Most recent round label, e.g. 'Series B'.",
    )

    last_funding_date: Mapped[str | None] = mapped_column(
        String(16),
        nullable=True,
        comment="Date of most recent funding round, e.g. '2024-03'.",
    )

    investors: Mapped[list[str]] = mapped_column(
        ARRAY(String(256)),
        nullable=False,
        default=list,
        server_default="{}",
        comment="Notable investor names — used in outreach personalisation.",
    )

    # -----------------------------------------------------------------------
    # Tech Stack & Culture
    # -----------------------------------------------------------------------

    tech_stack: Mapped[list[str]] = mapped_column(
        ARRAY(String(128)),
        nullable=False,
        default=list,
        server_default="{}",
        comment="Technologies used by the company (from job descriptions + Stackshare).",
    )

    engineering_blog_url: Mapped[str | None] = mapped_column(
        String(1024),
        nullable=True,
        comment="URL of the engineering / tech blog for content personalisation.",
    )

    culture_signals: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        server_default="{}",
        comment=(
            "Culture and workplace signals: "
            "{ remote_friendly: bool, async_culture: bool, "
            "  diversity_focus: bool, startup_pace: bool, "
            "  work_life_balance_rating: float, "
            "  glassdoor_rating: float, glassdoor_url: str, "
            "  values: [str], perks: [str] }"
        ),
    )

    # -----------------------------------------------------------------------
    # Social / Professional Presence
    # -----------------------------------------------------------------------

    linkedin_url: Mapped[str | None] = mapped_column(
        String(1024),
        nullable=True,
        comment="LinkedIn company page URL.",
    )

    linkedin_follower_count: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="LinkedIn follower count — proxy for brand recognition.",
    )

    twitter_handle: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
        comment="Twitter / X handle without @.",
    )

    github_org: Mapped[str | None] = mapped_column(
        String(256),
        nullable=True,
        comment="GitHub organisation name — used to check open-source activity.",
    )

    crunchbase_url: Mapped[str | None] = mapped_column(
        String(1024),
        nullable=True,
        comment="Crunchbase profile URL for funding data.",
    )

    # -----------------------------------------------------------------------
    # Hiring Pattern Metadata
    # -----------------------------------------------------------------------

    open_roles_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default="0",
        comment="Number of currently active job postings discovered.",
    )

    hiring_velocity_score: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        index=True,
        comment="Composite score (0–1) measuring hiring activity intensity.",
    )

    avg_application_response_days: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment="Average days to first recruiter response (from our application data).",
    )

    avg_offer_salary_usd: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment="Average offer salary seen in applications for this company.",
    )

    # -----------------------------------------------------------------------
    # Raw / External Data Cache
    # -----------------------------------------------------------------------

    raw_linkedin_data: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        server_default="{}",
        comment="Raw JSON from LinkedIn Company API or scraper — preserved for reprocessing.",
    )

    enrichment_sources: Mapped[list[str]] = mapped_column(
        ARRAY(String(64)),
        nullable=False,
        default=list,
        server_default="{}",
        comment="Sources used to enrich this record: linkedin | crunchbase | clearbit | manual.",
    )

    last_enriched_at: Mapped[str | None] = mapped_column(
        String(32),
        nullable=True,
        comment="ISO date of last enrichment run.",
    )

    # -----------------------------------------------------------------------
    # Relationships
    # -----------------------------------------------------------------------

    jobs: Mapped[list["Job"]] = relationship(
        "Job",
        back_populates="company",
        lazy="select",
        order_by="Job.posted_at.desc()",
    )

    recruiters: Mapped[list["Recruiter"]] = relationship(
        "Recruiter",
        back_populates="company",
        lazy="select",
    )

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    @property
    def is_hiring_actively(self) -> bool:
        return self.open_roles_count > 0

    @property
    def is_startup(self) -> bool:
        return self.funding_stage in ("pre-seed", "seed", "series-a", "series-b")

    @property
    def culture_rating(self) -> float | None:
        return self.culture_signals.get("glassdoor_rating")

    def __repr__(self) -> str:
        return f"<Company id={self.id} name={self.name!r} size={self.size_band}>"