"""
app/db/models/user.py
=====================
Core User model for the JobHunter AI platform.

Stores: identity, authentication credentials, subscription plan, role-based
access control, OAuth tokens (LinkedIn), notification preferences, login
tracking, and profile metadata.

Relationships (lazy-loaded via relationship()):
- resumes         → list[Resume]
- applications    → list[Application]
- linkedin_posts  → list[LinkedInPost]
- agent_runs      → list[AgentRun]
- audit_logs      → list[AuditLog]

Index strategy:
- email: UNIQUE (login lookup)
- is_active + plan: composite (plan-gated feature queries)
- created_at: for admin analytics
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import BaseModel
from app.core.constants import UserPlan

if TYPE_CHECKING:
    from app.db.models.resume import Resume
    from app.db.models.application import Application
    from app.db.models.linkedin_post import LinkedInPost
    from app.db.models.agent_run import AgentRun
    from app.db.models.audit_log import AuditLog


class User(BaseModel):
    """
    Platform user entity.

    Inherits id (UUID PK), created_at, updated_at, is_deleted, deleted_at
    from BaseModel.
    """

    __tablename__ = "users"

    __table_args__ = (
        UniqueConstraint("email", name="uq_users_email"),
        Index("ix_users_email_active", "email", "is_active"),
        Index("ix_users_plan", "plan"),
        Index("ix_users_created_at", "created_at"),
    )

    # -----------------------------------------------------------------------
    # Identity
    # -----------------------------------------------------------------------

    email: Mapped[str] = mapped_column(
        String(320),
        nullable=False,
        index=True,
        comment="Primary email address — used for login and notifications.",
    )

    full_name: Mapped[str | None] = mapped_column(
        String(256),
        nullable=True,
        comment="Display name shown in the UI and outreach emails.",
    )

    avatar_url: Mapped[str | None] = mapped_column(
        String(1024),
        nullable=True,
        comment="URL of the user's profile picture (S3 or local).",
    )

    phone: Mapped[str | None] = mapped_column(
        String(32),
        nullable=True,
        comment="Optional phone number for SMS notifications.",
    )

    location: Mapped[str | None] = mapped_column(
        String(256),
        nullable=True,
        comment="City / country used to filter job search results.",
    )

    timezone: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        default="UTC",
        server_default="UTC",
        comment="IANA timezone string, e.g. 'America/New_York'.",
    )

    # -----------------------------------------------------------------------
    # Authentication
    # -----------------------------------------------------------------------

    hashed_password: Mapped[str] = mapped_column(
        String(256),
        nullable=False,
        comment="bcrypt hash of the user's password.",
    )

    is_active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        server_default="true",
        index=True,
        comment="False = account suspended or banned.",
    )

    is_verified: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default="false",
        comment="True once the email address has been verified.",
    )

    is_superuser: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default="false",
        comment="Superuser flag — bypasses all permission checks.",
    )

    roles: Mapped[list[str]] = mapped_column(
        ARRAY(String(64)),
        nullable=False,
        default=list,
        server_default="{}",
        comment="Role list, e.g. ['user', 'premium', 'admin'].",
    )

    email_verification_token: Mapped[str | None] = mapped_column(
        String(256),
        nullable=True,
        comment="One-time token sent to verify the email address.",
    )

    email_verification_sent_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="When the verification email was last sent.",
    )

    password_reset_token: Mapped[str | None] = mapped_column(
        String(256),
        nullable=True,
        comment="One-time token for password reset flow.",
    )

    password_reset_expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="Expiry of the active password reset token.",
    )

    # -----------------------------------------------------------------------
    # Login tracking / security
    # -----------------------------------------------------------------------

    last_login_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="Timestamp of the most recent successful login.",
    )

    last_login_ip: Mapped[str | None] = mapped_column(
        String(45),
        nullable=True,
        comment="IP address of the most recent successful login.",
    )

    failed_login_attempts: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default="0",
        comment="Consecutive failed login counter; reset on success.",
    )

    locked_until: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="Account is locked until this UTC timestamp.",
    )

    # -----------------------------------------------------------------------
    # Subscription / Plan
    # -----------------------------------------------------------------------

    plan: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default=UserPlan.FREE,
        server_default=UserPlan.FREE,
        index=True,
        comment="Subscription tier: free | pro | enterprise.",
    )

    plan_expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="When the current paid plan expires (NULL = free / perpetual).",
    )

    stripe_customer_id: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
        unique=True,
        comment="Stripe Customer ID for billing integration.",
    )

    stripe_subscription_id: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
        comment="Stripe Subscription ID for the active plan.",
    )

    # -----------------------------------------------------------------------
    # Monthly usage counters (reset by Celery beat at month start)
    # -----------------------------------------------------------------------

    applications_this_month: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default="0",
        comment="Number of job applications submitted in the current billing month.",
    )

    ai_rewrites_this_month: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default="0",
        comment="Number of AI-powered resume rewrites consumed this month.",
    )

    # -----------------------------------------------------------------------
    # LinkedIn OAuth
    # -----------------------------------------------------------------------

    linkedin_access_token: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="Encrypted LinkedIn OAuth access token.",
    )

    linkedin_refresh_token: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="Encrypted LinkedIn OAuth refresh token.",
    )

    linkedin_token_expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="Expiry of the LinkedIn access token.",
    )

    linkedin_profile_id: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
        comment="LinkedIn member URN, e.g. 'urn:li:person:ABC123'.",
    )

    linkedin_profile_url: Mapped[str | None] = mapped_column(
        String(512),
        nullable=True,
        comment="Public LinkedIn profile URL.",
    )

    # -----------------------------------------------------------------------
    # Job Search Preferences
    # -----------------------------------------------------------------------

    job_search_preferences: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        server_default="{}",
        comment=(
            "User job-search configuration. "
            "Schema: {desired_roles, preferred_boards, work_mode, locations, "
            "min_salary, max_salary, job_types, excluded_companies, "
            "auto_apply_enabled, cover_letter_enabled, linkedin_posting_enabled}"
        ),
    )

    # -----------------------------------------------------------------------
    # Notification Preferences
    # -----------------------------------------------------------------------

    notification_preferences: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=lambda: {
            "email_on_application": True,
            "email_on_status_change": True,
            "email_on_interview": True,
            "email_weekly_summary": True,
            "in_app_notifications": True,
        },
        server_default=(
            '{"email_on_application":true,"email_on_status_change":true,'
            '"email_on_interview":true,"email_weekly_summary":true,'
            '"in_app_notifications":true}'
        ),
        comment="Per-channel notification opt-ins.",
    )

    # -----------------------------------------------------------------------
    # Relationships
    # -----------------------------------------------------------------------

    resumes: Mapped[list["Resume"]] = relationship(
        "Resume",
        back_populates="user",
        cascade="all, delete-orphan",
        lazy="select",
        order_by="Resume.created_at.desc()",
    )

    applications: Mapped[list["Application"]] = relationship(
        "Application",
        back_populates="user",
        cascade="all, delete-orphan",
        lazy="select",
        order_by="Application.created_at.desc()",
    )

    linkedin_posts: Mapped[list["LinkedInPost"]] = relationship(
        "LinkedInPost",
        back_populates="user",
        cascade="all, delete-orphan",
        lazy="select",
        order_by="LinkedInPost.created_at.desc()",
    )

    agent_runs: Mapped[list["AgentRun"]] = relationship(
        "AgentRun",
        back_populates="user",
        cascade="all, delete-orphan",
        lazy="select",
        order_by="AgentRun.created_at.desc()",
    )

    audit_logs: Mapped[list["AuditLog"]] = relationship(
        "AuditLog",
        back_populates="user",
        lazy="select",
        order_by="AuditLog.created_at.desc()",
    )

    # -----------------------------------------------------------------------
    # Business logic helpers
    # -----------------------------------------------------------------------

    @property
    def is_locked(self) -> bool:
        """True if the account is currently locked."""
        if self.locked_until is None:
            return False
        return datetime.now(timezone.utc) < self.locked_until

    @property
    def is_plan_active(self) -> bool:
        """True if the user has an active paid plan or is on free tier."""
        if self.plan == UserPlan.FREE:
            return True
        if self.plan_expires_at is None:
            return True
        return datetime.now(timezone.utc) < self.plan_expires_at

    @property
    def has_linkedin_connected(self) -> bool:
        """True if a valid LinkedIn OAuth token is stored."""
        if not self.linkedin_access_token:
            return False
        if self.linkedin_token_expires_at is None:
            return True
        return datetime.now(timezone.utc) < self.linkedin_token_expires_at

    @property
    def display_name(self) -> str:
        """Best-effort display name — falls back to email prefix."""
        return self.full_name or self.email.split("@")[0]

    def increment_failed_login(self) -> None:
        self.failed_login_attempts += 1

    def reset_failed_login(self) -> None:
        self.failed_login_attempts = 0
        self.locked_until = None

    def get_preference(self, key: str, default: Any = None) -> Any:
        """Safe accessor for nested job_search_preferences keys."""
        return self.job_search_preferences.get(key, default)

    def __repr__(self) -> str:
        return f"<User id={self.id} email={self.email!r} plan={self.plan}>"