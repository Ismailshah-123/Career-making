"""
app/db/models/recruiter.py
===========================
Recruiter contact model for the JobHunter AI platform.

Stores: contact details, LinkedIn profile, company association,
outreach sequence state, response history, and quality signals.

The outreach_agent discovers recruiters for target companies and populates
this table. The followup_agent reads it to schedule personalised outreach
and track conversation threads.

Relationships:
- company      → Company
- applications → list[Application]  (which applications this recruiter is linked to)
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

if TYPE_CHECKING:
    from app.db.models.company import Company
    from app.db.models.application import Application


class Recruiter(BaseModel):
    """
    A recruiter or hiring manager contact at a target company.

    Inherits id (UUID PK), created_at, updated_at, is_deleted, deleted_at.
    """

    __tablename__ = "recruiters"

    __table_args__ = (
        UniqueConstraint("linkedin_profile_id", name="uq_recruiter_linkedin_id"),
        Index("ix_recruiters_company_id", "company_id"),
        Index("ix_recruiters_email", "email"),
        Index("ix_recruiters_is_responsive", "is_responsive"),
        Index("ix_recruiters_outreach_status", "outreach_status"),
        Index("ix_recruiters_response_rate", "response_rate"),
    )

    # -----------------------------------------------------------------------
    # Identity
    # -----------------------------------------------------------------------

    full_name: Mapped[str | None] = mapped_column(
        String(256),
        nullable=True,
        comment="Recruiter's full name.",
    )

    first_name: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
        comment="First name — used in personalised outreach salutations.",
    )

    title: Mapped[str | None] = mapped_column(
        String(256),
        nullable=True,
        comment="Job title, e.g. 'Senior Technical Recruiter' or 'Head of Talent'.",
    )

    email: Mapped[str | None] = mapped_column(
        String(320),
        nullable=True,
        index=True,
        comment="Professional email address if discovered.",
    )

    email_verified: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default="false",
        comment="True if the email has been verified via a send or hunter.io.",
    )

    phone: Mapped[str | None] = mapped_column(
        String(32),
        nullable=True,
        comment="Phone number if publicly available.",
    )

    # -----------------------------------------------------------------------
    # LinkedIn
    # -----------------------------------------------------------------------

    linkedin_profile_id: Mapped[str | None] = mapped_column(
        String(256),
        nullable=True,
        unique=True,
        comment="LinkedIn member URN or profile ID for deduplication.",
    )

    linkedin_profile_url: Mapped[str | None] = mapped_column(
        String(1024),
        nullable=True,
        comment="Public LinkedIn profile URL.",
    )

    linkedin_connection_degree: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="1 = 1st connection, 2 = 2nd, 3 = 3rd. Lower = warmer outreach.",
    )

    linkedin_mutual_connections: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="Number of mutual LinkedIn connections — personalisation signal.",
    )

    linkedin_is_open_to_connect: Mapped[bool | None] = mapped_column(
        Boolean,
        nullable=True,
        comment="True if the recruiter has Open Profile or Open to Connect enabled.",
    )

    # -----------------------------------------------------------------------
    # Company Association
    # -----------------------------------------------------------------------

    company_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("companys.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
        comment="Company this recruiter currently works at.",
    )

    company_name: Mapped[str | None] = mapped_column(
        String(512),
        nullable=True,
        comment="Denormalised company name — used when company_id is NULL.",
    )

    specialisations: Mapped[list[str]] = mapped_column(
        ARRAY(String(128)),
        nullable=False,
        default=list,
        server_default="{}",
        comment="Roles or functions this recruiter specialises in, e.g. ['Engineering', 'Data'].",
    )

    seniority_focus: Mapped[list[str]] = mapped_column(
        ARRAY(String(32)),
        nullable=False,
        default=list,
        server_default="{}",
        comment="Experience levels they typically recruit for: ['mid', 'senior', 'staff'].",
    )

    # -----------------------------------------------------------------------
    # Outreach Sequencing
    # -----------------------------------------------------------------------

    outreach_status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="not_contacted",
        server_default="not_contacted",
        index=True,
        comment=(
            "Current outreach state: "
            "not_contacted | connection_sent | connected | message_sent | "
            "replied | meeting_booked | not_interested | opted_out."
        ),
    )

    outreach_channel: Mapped[str | None] = mapped_column(
        String(32),
        nullable=True,
        comment="Channel used: linkedin_message | linkedin_inmail | email | cold_call.",
    )

    connection_request_sent_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="When the LinkedIn connection request was sent.",
    )

    connected_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="When the LinkedIn connection was accepted.",
    )

    first_message_sent_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="When the first outreach message was sent.",
    )

    last_contacted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="Most recent contact attempt timestamp.",
    )

    next_contact_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="Scheduled time for the next contact in the outreach sequence.",
    )

    outreach_sequence_step: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default="0",
        comment="Current position in the outreach sequence (0 = not started).",
    )

    outreach_history: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
        server_default="[]",
        comment=(
            "Chronological outreach log: "
            "[{step, channel, sent_at, message_preview, response, response_at}]."
        ),
    )

    # -----------------------------------------------------------------------
    # Response Tracking
    # -----------------------------------------------------------------------

    has_responded: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default="false",
        comment="True if at least one response has been received from this recruiter.",
    )

    first_response_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="Timestamp of the first response received.",
    )

    last_response_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="Timestamp of the most recent response.",
    )

    response_sentiment: Mapped[str | None] = mapped_column(
        String(32),
        nullable=True,
        comment="Sentiment of the last response: positive | neutral | negative.",
    )

    conversation_thread: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
        server_default="[]",
        comment="Full conversation thread: [{direction: sent|received, message, timestamp}].",
    )

    # -----------------------------------------------------------------------
    # Quality Signals
    # -----------------------------------------------------------------------

    is_responsive: Mapped[bool | None] = mapped_column(
        Boolean,
        nullable=True,
        index=True,
        comment="True if this recruiter historically responds within 14 days.",
    )

    response_rate: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        index=True,
        comment="Platform-wide response rate for this recruiter (0–1), from aggregated data.",
    )

    quality_score: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment="Composite quality score (0–1): connection degree + response rate + seniority match.",
    )

    do_not_contact: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default="false",
        comment="True if the recruiter has opted out or the user marked them DNC.",
    )

    # -----------------------------------------------------------------------
    # Discovery Metadata
    # -----------------------------------------------------------------------

    discovery_source: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
        comment="How this recruiter was found: linkedin_search | job_posting | referral | manual.",
    )

    discovery_context: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        server_default="{}",
        comment="Context at discovery time: {job_id, search_query, search_date}.",
    )

    notes: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="Internal notes from the user or agents.",
    )

    # -----------------------------------------------------------------------
    # Relationships
    # -----------------------------------------------------------------------

    company: Mapped["Company | None"] = relationship(
        "Company",
        back_populates="recruiters",
        lazy="select",
    )

    applications: Mapped[list["Application"]] = relationship(
        "Application",
        back_populates="recruiter",
        lazy="select",
    )

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    @property
    def salutation(self) -> str:
        return self.first_name or self.full_name or "there"

    @property
    def is_warm_lead(self) -> bool:
        return (self.linkedin_connection_degree or 3) <= 2 and not self.do_not_contact

    def __repr__(self) -> str:
        return (
            f"<Recruiter id={self.id} name={self.full_name!r} "
            f"company={self.company_name!r} status={self.outreach_status!r}>"
        )