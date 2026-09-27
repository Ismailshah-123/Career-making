"""
app/db/models/application.py
=============================
Application model — tracks every job application from discovery to outcome.

This is the central workflow entity. Every automated or manual application
lives here, with a full audit trail of status transitions, automation
attempts, recruiter contact, and follow-up scheduling.

Relationships:
- user         → User
- job          → Job
- resume       → Resume  (the tailored resume used)
- cover_letter → CoverLetter

Status machine:
  discovered → queued → resume_tailored → cover_letter_generated
  → applying → applied → acknowledged → interview_scheduled
  → interviewed → offer_received / rejected / withdrawn

See constants.APPLICATION_STATUS_TRANSITIONS for the allowed edges.
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
from app.core.constants import ApplicationStatus

if TYPE_CHECKING:
    from app.db.models.user import User
    from app.db.models.job import Job
    from app.db.models.resume import Resume
    from app.db.models.cover_letter import CoverLetter
    from app.db.models.recruiter import Recruiter


class Application(BaseModel):
    """
    Represents one job application by a User for a Job posting.

    One user → one job → at most one active application (enforced by UNIQUE
    constraint on user_id + job_id where is_deleted = false).

    Inherits id (UUID PK), created_at, updated_at, is_deleted, deleted_at.
    """

    __tablename__ = "applications"

    __table_args__ = (
        UniqueConstraint(
            "user_id", "job_id",
            name="uq_applications_user_job",
        ),
        Index("ix_applications_user_id", "user_id"),
        Index("ix_applications_job_id", "job_id"),
        Index("ix_applications_status", "status"),
        Index("ix_applications_user_status", "user_id", "status"),
        Index("ix_applications_applied_at", "applied_at"),
        Index("ix_applications_next_followup", "next_followup_at"),
    )

    # -----------------------------------------------------------------------
    # Core FKs
    # -----------------------------------------------------------------------

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        comment="Applicant.",
    )

    job_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("jobs.id", ondelete="CASCADE"),
        nullable=False,
        comment="Job being applied to.",
    )

    resume_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("resumes.id", ondelete="SET NULL"),
        nullable=True,
        comment="Tailored resume used for this application (NULL until tailored).",
    )

    cover_letter_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("cover_letters.id", ondelete="SET NULL"),
        nullable=True,
        comment="Generated cover letter (NULL until generated).",
    )

    recruiter_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("recruiters.id", ondelete="SET NULL"),
        nullable=True,
        comment="Recruiter contact discovered by the outreach agent.",
    )

    # -----------------------------------------------------------------------
    # Status Machine
    # -----------------------------------------------------------------------

    status: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        default=ApplicationStatus.DISCOVERED,
        server_default=ApplicationStatus.DISCOVERED,
        index=True,
        comment="Current position in the application workflow.",
    )

    status_history: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
        server_default="[]",
        comment=(
            "Ordered list of status transitions: "
            "[{from_status, to_status, timestamp, reason, agent}]."
        ),
    )

    # -----------------------------------------------------------------------
    # Match Score
    # -----------------------------------------------------------------------

    match_score: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment="Cosine similarity score (0–1) between resume and job embedding.",
    )

    match_tier: Mapped[str | None] = mapped_column(
        String(16),
        nullable=True,
        comment="excellent | good | fair | poor — derived from match_score.",
    )

    match_explanation: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="Human-readable explanation of the match score from the matching agent.",
    )

    keyword_match_score: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment="Fraction of required job skills present in the resume (0–1).",
    )

    matched_skills: Mapped[list[str]] = mapped_column(
        ARRAY(String(128)),
        nullable=False,
        default=list,
        server_default="{}",
        comment="Skills in both the resume and job requirements.",
    )

    missing_skills: Mapped[list[str]] = mapped_column(
        ARRAY(String(128)),
        nullable=False,
        default=list,
        server_default="{}",
        comment="Required skills absent from the resume.",
    )

    # -----------------------------------------------------------------------
    # Application Submission
    # -----------------------------------------------------------------------

    is_auto_applied: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default="false",
        comment="True if the application was submitted by the automation layer.",
    )

    applied_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
        comment="UTC timestamp when the application form was successfully submitted.",
    )

    application_method: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
        comment="How it was submitted: playwright_automation | manual | linkedin_easy_apply | api.",
    )

    # -----------------------------------------------------------------------
    # Automation Tracking
    # -----------------------------------------------------------------------

    automation_attempts: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default="0",
        comment="Number of Playwright automation attempts made.",
    )

    last_automation_attempt_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="When the most recent automation attempt started.",
    )

    automation_error: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="Error message from the most recent failed automation attempt.",
    )

    automation_screenshot_path: Mapped[str | None] = mapped_column(
        String(1024),
        nullable=True,
        comment="S3 / local path to a screenshot captured on automation failure.",
    )

    form_data_used: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        server_default="{}",
        comment="Fields submitted in the application form — for debugging and resubmission.",
    )

    # -----------------------------------------------------------------------
    # Recruiter Outreach
    # -----------------------------------------------------------------------

    outreach_sent: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default="false",
        comment="True if a personalised LinkedIn / email outreach was sent.",
    )

    outreach_sent_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="When the outreach message was dispatched.",
    )

    outreach_message: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="The exact outreach message sent to the recruiter.",
    )

    outreach_response_received: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default="false",
        comment="True if the recruiter has responded to the outreach.",
    )

    # -----------------------------------------------------------------------
    # Follow-up Scheduling
    # -----------------------------------------------------------------------

    followup_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default="0",
        comment="Number of follow-up messages sent so far.",
    )

    next_followup_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
        comment="When the follow-up agent should send the next follow-up.",
    )

    last_followup_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="When the most recent follow-up was sent.",
    )

    followup_message: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="Text of the most recently generated/sent follow-up message.",
    )

    followup_history: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
        server_default="[]",
        comment="[{sent_at, message, channel, response}] — full follow-up history.",
    )

    # -----------------------------------------------------------------------
    # Interview Tracking
    # -----------------------------------------------------------------------

    interview_rounds: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
        server_default="[]",
        comment=(
            "Interview rounds: "
            "[{round_number, type, scheduled_at, duration_minutes, "
            "  interviewer_name, notes, outcome}]."
        ),
    )

    interview_prep_notes: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="AI-generated or user-written prep notes for interviews.",
    )

    # -----------------------------------------------------------------------
    # Offer
    # -----------------------------------------------------------------------

    offer_details: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        server_default="{}",
        comment=(
            "Offer details if received: "
            "{base_salary, equity, bonus, benefits, start_date, expires_at, accepted}."
        ),
    )

    # -----------------------------------------------------------------------
    # User Notes & Priority
    # -----------------------------------------------------------------------

    user_notes: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="Free-form notes added by the user in the dashboard.",
    )

    priority: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=5,
        server_default="5",
        comment="User-assigned priority 1 (highest) – 10 (lowest) for sorting.",
    )

    is_starred: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default="false",
        comment="User-starred for quick access in the dashboard.",
    )

    # -----------------------------------------------------------------------
    # Relationships
    # -----------------------------------------------------------------------

    user: Mapped["User"] = relationship("User", back_populates="applications", lazy="select")
    job: Mapped["Job"] = relationship("Job", back_populates="applications", lazy="select")
    resume: Mapped["Resume | None"] = relationship(
        "Resume",
        back_populates="applications",
        foreign_keys=[resume_id],
        lazy="select",
    )
    cover_letter: Mapped["CoverLetter | None"] = relationship(
        "CoverLetter",
        back_populates="application",
        foreign_keys=[cover_letter_id],
        lazy="select",
    )
    recruiter: Mapped["Recruiter | None"] = relationship(
        "Recruiter",
        back_populates="applications",
        lazy="select",
    )

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def record_status_change(
        self,
        from_status: str,
        to_status: str,
        *,
        reason: str | None = None,
        agent: str | None = None,
    ) -> None:
        """Append a status transition record to status_history."""
        from datetime import timezone
        entry: dict[str, Any] = {
            "from_status": from_status,
            "to_status": to_status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if reason:
            entry["reason"] = reason
        if agent:
            entry["agent"] = agent
        history = list(self.status_history)
        history.append(entry)
        self.status_history = history
        self.status = to_status

    @property
    def is_terminal(self) -> bool:
        from app.core.constants import TERMINAL_STATUSES, ApplicationStatus
        return ApplicationStatus(self.status) in TERMINAL_STATUSES

    @property
    def days_since_applied(self) -> int | None:
        if not self.applied_at:
            return None
        from datetime import timezone
        delta = datetime.now(timezone.utc) - self.applied_at
        return delta.days

    def __repr__(self) -> str:
        return (
            f"<Application id={self.id} user_id={self.user_id} "
            f"job_id={self.job_id} status={self.status!r}>"
        )