"""
app/db/models/audit_log.py
===========================
Immutable audit log for the JobHunter AI platform.

Every security-sensitive or data-mutating action is recorded here:
user auth events, resource CRUD, agent actions, billing changes,
admin overrides, and API key lifecycle.

Design principles:
- APPEND-ONLY: no UPDATE or DELETE ever occurs on this table
- No soft-delete mixin (is_deleted doesn't apply — logs are permanent)
- Stores before/after diffs for data mutation events
- IP address and user-agent captured for security forensics
- Indexed for compliance queries (user activity, event type, date range)

Used by: security audits, GDPR data exports, anomaly detection,
         admin dashboards, and customer support investigations.

Relationships:
- user → User (nullable — system events have no user)
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from sqlalchemy import (
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, UUIDMixin
from app.core.constants import AuditEvent

if TYPE_CHECKING:
    from app.db.models.user import User


class AuditLog(UUIDMixin, Base):
    """
    Immutable audit log entry.

    Intentionally does NOT inherit TimestampMixin or SoftDeleteMixin —
    audit records are written once, never modified, never deleted.

    The `created_at` column is set server-side to prevent client tampering.
    """

    __tablename__ = "audit_logs"

    __table_args__ = (
        Index("ix_audit_logs_user_id", "user_id"),
        Index("ix_audit_logs_event", "event"),
        Index("ix_audit_logs_resource", "resource_type", "resource_id"),
        Index("ix_audit_logs_created_at", "created_at"),
        Index("ix_audit_logs_ip_address", "ip_address"),
        Index("ix_audit_logs_user_event", "user_id", "event"),
        Index("ix_audit_logs_user_date", "user_id", "created_at"),
        Index("ix_audit_logs_session", "session_id"),
    )

    # -----------------------------------------------------------------------
    # Timestamp (server-side only — never set by Python to prevent tampering)
    # -----------------------------------------------------------------------

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        comment="UTC timestamp of the audit event — set by the database server.",
    )

    # -----------------------------------------------------------------------
    # Actor
    # -----------------------------------------------------------------------

    user_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
        comment="User who performed the action (NULL for system/background tasks).",
    )

    actor_type: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="user",
        server_default="user",
        comment="Who/what triggered the event: user | system | agent | admin | api_key.",
    )

    actor_id: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
        comment="Supplemental actor ID (e.g. API key ID, agent name, scheduler name).",
    )

    # -----------------------------------------------------------------------
    # Event
    # -----------------------------------------------------------------------

    event: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        index=True,
        comment=(
            "Audit event type (dot-separated namespace): "
            "user.login | resume.uploaded | application.submitted | agent.run_completed | etc. "
            "See AuditEvent enum in constants.py."
        ),
    )

    event_category: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="info",
        server_default="info",
        comment="Severity/category: info | warning | security | critical.",
    )

    description: Mapped[str | None] = mapped_column(
        String(1024),
        nullable=True,
        comment="Human-readable description of the event.",
    )

    # -----------------------------------------------------------------------
    # Resource (what was acted upon)
    # -----------------------------------------------------------------------

    resource_type: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
        index=True,
        comment="Type of the affected resource: User | Resume | Job | Application | etc.",
    )

    resource_id: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
        index=True,
        comment="UUID or identifier of the affected resource.",
    )

    # -----------------------------------------------------------------------
    # Change Diff (for CRUD events)
    # -----------------------------------------------------------------------

    before_state: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB,
        nullable=True,
        comment="Snapshot of the resource state BEFORE the mutation (NULL for CREATE events).",
    )

    after_state: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB,
        nullable=True,
        comment="Snapshot of the resource state AFTER the mutation (NULL for DELETE events).",
    )

    changed_fields: Mapped[list[str]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
        server_default="[]",
        comment="List of field names that changed in an UPDATE event.",
    )

    # -----------------------------------------------------------------------
    # Request Context
    # -----------------------------------------------------------------------

    ip_address: Mapped[str | None] = mapped_column(
        String(45),
        nullable=True,
        index=True,
        comment="Client IP address (IPv4 or IPv6). Null for background tasks.",
    )

    user_agent: Mapped[str | None] = mapped_column(
        String(1024),
        nullable=True,
        comment="HTTP User-Agent string from the request.",
    )

    request_id: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
        comment="X-Correlation-ID from the originating HTTP request.",
    )

    session_id: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
        index=True,
        comment="User session identifier for grouping events within a session.",
    )

    api_version: Mapped[str | None] = mapped_column(
        String(16),
        nullable=True,
        comment="API version used, e.g. 'v1'.",
    )

    endpoint: Mapped[str | None] = mapped_column(
        String(512),
        nullable=True,
        comment="HTTP method + path of the originating request, e.g. 'POST /api/v1/resumes'.",
    )

    # -----------------------------------------------------------------------
    # Outcome
    # -----------------------------------------------------------------------

    success: Mapped[bool] = mapped_column(
        JSONB,
        nullable=False,
        default=True,
        server_default="true",
        comment="True if the action completed successfully; False if it was blocked or failed.",
    )

    failure_reason: Mapped[str | None] = mapped_column(
        String(512),
        nullable=True,
        comment="Reason for failure if success=False (e.g. 'Rate limit exceeded').",
    )

    http_status_code: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="HTTP status code of the response, if applicable.",
    )

    # -----------------------------------------------------------------------
    # Extra Metadata
    # -----------------------------------------------------------------------

    # NOTE: named `event_metadata`, not `metadata` — SQLAlchemy's declarative
    # API reserves the attribute name `metadata` on every mapped class
    # (it's `Base.metadata`, the table registry). Using it as a column name
    # raises InvalidRequestError at class-definition time.
    event_metadata: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        server_default="{}",
        comment=(
            "Arbitrary extra data specific to the event type. "
            "For user.login: {failed_attempts, lockout_triggered}. "
            "For application.submitted: {job_board, auto_applied}. "
            "For agent.run_completed: {agent_name, tokens_used, duration_ms}."
        ),
    )

    tags: Mapped[list[str]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
        server_default="[]",
        comment="Free-form tags for grouping/filtering: ['security', 'billing', 'gdpr'].",
    )

    # -----------------------------------------------------------------------
    # Relationships
    # -----------------------------------------------------------------------

    user: Mapped["User | None"] = relationship(
        "User",
        back_populates="audit_logs",
        lazy="select",
    )

    # -----------------------------------------------------------------------
    # Class-level factory
    # -----------------------------------------------------------------------

    @classmethod
    def create(
        cls,
        *,
        event: str,
        description: str | None = None,
        user_id: uuid.UUID | None = None,
        actor_type: str = "user",
        actor_id: str | None = None,
        resource_type: str | None = None,
        resource_id: str | None = None,
        before_state: dict[str, Any] | None = None,
        after_state: dict[str, Any] | None = None,
        changed_fields: list[str] | None = None,
        ip_address: str | None = None,
        user_agent: str | None = None,
        request_id: str | None = None,
        session_id: str | None = None,
        endpoint: str | None = None,
        success: bool = True,
        failure_reason: str | None = None,
        http_status_code: int | None = None,
        metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        event_category: str = "info",
    ) -> "AuditLog":
        """
        Factory method for creating audit log entries.

        Usage:
            log = AuditLog.create(
                event=AuditEvent.USER_LOGIN,
                user_id=user.id,
                ip_address=request_ip,
                success=True,
            )
            db.add(log)
            await db.flush()   # No commit — let the transaction owner commit
        """
        return cls(
            event=event,
            description=description,
            user_id=user_id,
            actor_type=actor_type,
            actor_id=actor_id,
            event_category=event_category,
            resource_type=resource_type,
            resource_id=str(resource_id) if resource_id else None,
            before_state=before_state,
            after_state=after_state,
            changed_fields=changed_fields or [],
            ip_address=ip_address,
            user_agent=user_agent,
            request_id=request_id,
            session_id=session_id,
            endpoint=endpoint,
            success=success,
            failure_reason=failure_reason,
            http_status_code=http_status_code,
            event_metadata=metadata or {},
            tags=tags or [],
            api_version="v1",
        )

    def __repr__(self) -> str:
        return (
            f"<AuditLog id={self.id} event={self.event!r} "
            f"user_id={self.user_id} success={self.success}>"
        )