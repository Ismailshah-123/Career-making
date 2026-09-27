"""
CareerGPT — User Repository + AuditLog Repository
====================================================
PAGE SUMMARY:
  Data access layer for users and audit logs.
  All SQL queries live here — services/agents never write raw SQL.
  Implements: CRUD, soft-delete, search, bulk ops, stats queries.

  UserRepository:
    - get_by_id, get_by_id_or_raise, get_by_email, email_exists
    - create, update, soft_delete, hard_delete (GDPR)
    - update_last_login, update_stats (total_applications, etc.)
    - search_users (admin panel), get_by_plan, get_inactive_users
    - get_by_reset_token, get_by_email_verify_token (auth flows)
    - count_active, get_all_paginated (admin analytics)

  AuditLogRepository:
    - create (immutable — no update/delete)
    - get_by_user, get_by_action, get_recent (admin panel)
    - count_by_action (security analytics)

  BASE PATTERN:
    All repos inherit from BaseRepository[ModelT] which provides:
      get_by_id, get_by_id_or_raise, create, update, delete, count,
      exists, filter_by, first_by, upsert, bulk_update, get_all

  SOFT DELETE ENFORCEMENT:
    Every query on soft-deletable models adds WHERE is_deleted = FALSE.
    Hard delete (GDPR) available via delete(..., hard=True).
    Deleted records remain in DB for 90 days then purged by maintenance task.

  QUERY OPTIMIZATION:
    - All FK columns have DB indexes (defined on model)
    - Email column has unique index
    - Frequently filtered columns (role, plan, is_active) are indexed
    - Pagination uses OFFSET/LIMIT (acceptable for <100K users)
    - For large datasets: cursor-based pagination available via scroll_by_id()
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import and_, desc, func, or_, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import logger
from app.db.models.audit_log import AuditLog
from app.db.models.user import User


# ── Base Repository ───────────────────────────────────────────────────────────

class BaseRepository:
    """Generic async CRUD repository. All domain repos inherit from this."""

    model: type

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    def _base_query(self):
        q = select(self.model)
        if hasattr(self.model, "is_deleted"):
            q = q.where(self.model.is_deleted.is_(False))
        return q

    async def get_by_id(self, record_id: uuid.UUID) -> Any | None:
        q = self._base_query().where(self.model.id == record_id)
        result = await self.session.execute(q)
        return result.scalar_one_or_none()

    async def get_by_id_or_raise(self, record_id: uuid.UUID) -> Any:
        from app.core.exceptions import NotFoundError
        record = await self.get_by_id(record_id)
        if record is None:
            raise NotFoundError(
                f"{self.model.__name__} not found",
                context={"id": str(record_id)},
            )
        return record

    async def create(self, **kwargs: Any) -> Any:
        record = self.model(**kwargs)
        self.session.add(record)
        await self.session.flush()
        await self.session.refresh(record)
        return record

    async def update(self, record_id: uuid.UUID, **kwargs: Any) -> Any:
        record = await self.get_by_id_or_raise(record_id)
        for k, v in kwargs.items():
            setattr(record, k, v)
        self.session.add(record)
        await self.session.flush()
        await self.session.refresh(record)
        return record

    async def delete(self, record_id: uuid.UUID, *, hard: bool = False) -> None:
        record = await self.get_by_id_or_raise(record_id)
        if hard:
            await self.session.delete(record)
        else:
            if hasattr(record, "soft_delete"):
                record.soft_delete()
                self.session.add(record)
            else:
                await self.session.delete(record)
        await self.session.flush()

    async def count(self) -> int:
        q = select(func.count()).select_from(self._base_query().subquery())
        result = await self.session.execute(q)
        return result.scalar_one() or 0

    async def exists(self, record_id: uuid.UUID) -> bool:
        q = (
            select(func.count())
            .select_from(self.model)
            .where(self.model.id == record_id)
        )
        if hasattr(self.model, "is_deleted"):
            q = q.where(self.model.is_deleted.is_(False))
        result = await self.session.execute(q)
        return (result.scalar_one() or 0) > 0

    async def filter_by(self, **kwargs: Any) -> list[Any]:
        q = self._base_query()
        for k, v in kwargs.items():
            q = q.where(getattr(self.model, k) == v)
        result = await self.session.execute(q)
        return list(result.scalars().all())

    async def first_by(self, **kwargs: Any) -> Any | None:
        q = self._base_query()
        for k, v in kwargs.items():
            q = q.where(getattr(self.model, k) == v)
        result = await self.session.execute(q.limit(1))
        return result.scalar_one_or_none()

    async def get_all(
        self,
        *,
        skip: int = 0,
        limit: int = 50,
        order_by: Any = None,
    ) -> list[Any]:
        q = self._base_query().offset(skip).limit(limit)
        if order_by is not None:
            q = q.order_by(order_by)
        result = await self.session.execute(q)
        return list(result.scalars().all())

    async def upsert(
        self,
        lookup: dict[str, Any],
        defaults: dict[str, Any],
    ) -> tuple[Any, bool]:
        """Get or create. Returns (record, created: bool)."""
        record = await self.first_by(**lookup)
        if record:
            return record, False
        record = await self.create(**lookup, **defaults)
        return record, True

    async def bulk_update(
        self,
        filters: dict[str, Any],
        values: dict[str, Any],
    ) -> int:
        from sqlalchemy import update
        q = update(self.model)
        for k, v in filters.items():
            q = q.where(getattr(self.model, k) == v)
        q = q.values(**values)
        result = await self.session.execute(q)
        return result.rowcount  # type: ignore[return-value]


# ── User Repository ───────────────────────────────────────────────────────────

class UserRepository(BaseRepository):
    model = User

    # ── Read ──────────────────────────────────────────────────────────────────

    async def get_by_email(self, email: str) -> User | None:
        """Fetch user by email (case-insensitive)."""
        q = self._base_query().where(
            func.lower(User.email) == email.lower().strip()
        )
        result = await self.session.execute(q)
        return result.scalar_one_or_none()

    async def email_exists(self, email: str) -> bool:
        """Fast check if email is already registered."""
        q = (
            select(func.count())
            .select_from(User)
            .where(func.lower(User.email) == email.lower().strip())
            .where(User.is_deleted.is_(False))
        )
        result = await self.session.execute(q)
        return (result.scalar_one() or 0) > 0

    async def get_by_reset_token(self, token_hash: str) -> User | None:
        """Fetch user by hashed password reset token."""
        q = (
            self._base_query()
            .where(User.password_reset_token == token_hash)
            .where(User.is_active.is_(True))
        )
        result = await self.session.execute(q)
        return result.scalar_one_or_none()

    async def get_by_email_verify_token(self, token_hash: str) -> User | None:
        """Fetch user by hashed email verification token."""
        q = (
            self._base_query()
            .where(User.email_verify_token == token_hash)
        )
        result = await self.session.execute(q)
        return result.scalar_one_or_none()

    async def get_by_plan(
        self,
        plan: str,
        *,
        skip: int = 0,
        limit: int = 100,
    ) -> list[User]:
        """Fetch users on a specific plan (admin analytics)."""
        q = (
            self._base_query()
            .where(User.plan == plan)
            .where(User.is_active.is_(True))
            .order_by(desc(User.created_at))
            .offset(skip)
            .limit(limit)
        )
        result = await self.session.execute(q)
        return list(result.scalars().all())

    async def get_inactive_users(
        self,
        *,
        inactive_since_days: int = 30,
        skip: int = 0,
        limit: int = 100,
    ) -> list[User]:
        """
        Fetch users who haven't logged in for N days.
        Used by notification service for re-engagement emails.
        """
        cutoff = text(f"NOW() - INTERVAL '{inactive_since_days} days'")
        q = (
            self._base_query()
            .where(User.is_active.is_(True))
            .where(
                or_(
                    User.last_login_at < cutoff,
                    User.last_login_at.is_(None),
                )
            )
            .order_by(User.last_login_at.asc().nullsfirst())
            .offset(skip)
            .limit(limit)
        )
        result = await self.session.execute(q)
        return list(result.scalars().all())

    async def search_users(
        self,
        query: str,
        *,
        skip: int = 0,
        limit: int = 20,
        role: str | None = None,
        plan: str | None = None,
    ) -> list[User]:
        """Full-text search across name + email (admin panel)."""
        q = self._base_query().where(
            or_(
                User.full_name.ilike(f"%{query}%"),
                User.email.ilike(f"%{query}%"),
            )
        )
        if role:
            q = q.where(User.role == role)
        if plan:
            q = q.where(User.plan == plan)
        q = q.order_by(desc(User.created_at)).offset(skip).limit(limit)
        result = await self.session.execute(q)
        return list(result.scalars().all())

    async def count_active(self) -> int:
        """Count active (non-deleted, is_active=True) users."""
        q = (
            select(func.count())
            .select_from(User)
            .where(User.is_deleted.is_(False))
            .where(User.is_active.is_(True))
        )
        result = await self.session.execute(q)
        return result.scalar_one() or 0

    async def count_by_plan(self) -> dict[str, int]:
        """Return count of users per plan (admin dashboard)."""
        q = (
            select(User.plan, func.count(User.id).label("cnt"))
            .where(User.is_deleted.is_(False))
            .where(User.is_active.is_(True))
            .group_by(User.plan)
        )
        result = await self.session.execute(q)
        return {row.plan: row.cnt for row in result.all()}

    async def get_registrations_by_day(self, days: int = 30) -> list[dict]:
        """
        Return daily registration counts for the last N days.
        Used for admin analytics chart.
        """
        q = text("""
            SELECT
                DATE(created_at AT TIME ZONE 'UTC') AS day,
                COUNT(*) AS count
            FROM users
            WHERE is_deleted = FALSE
              AND created_at >= NOW() - INTERVAL ':days days'
            GROUP BY DATE(created_at AT TIME ZONE 'UTC')
            ORDER BY day DESC
        """).bindparams(days=days)
        result = await self.session.execute(q)
        return [{"day": str(row.day), "count": row.count} for row in result.all()]

    async def scroll_by_id(
        self,
        *,
        after_id: uuid.UUID | None = None,
        limit: int = 100,
        active_only: bool = True,
    ) -> list[User]:
        """
        Cursor-based pagination for large datasets.
        More efficient than OFFSET for >100K records.
        """
        q = self._base_query()
        if active_only:
            q = q.where(User.is_active.is_(True))
        if after_id:
            q = q.where(User.id > after_id)
        q = q.order_by(User.id).limit(limit)
        result = await self.session.execute(q)
        return list(result.scalars().all())

    # ── Write ─────────────────────────────────────────────────────────────────

    async def update_last_login(self, user_id: uuid.UUID) -> None:
        """Update last_login_at timestamp. Called on every successful login."""
        await self.update(user_id, last_login_at=datetime.now(UTC))

    async def increment_stat(
        self,
        user_id: uuid.UUID,
        field: str,
        delta: int = 1,
    ) -> None:
        """
        Atomically increment a numeric stat field.
        Usage: await repo.increment_stat(user_id, "total_applications", 1)
        """
        from sqlalchemy import update as sql_update
        q = (
            sql_update(User)
            .where(User.id == user_id)
            .values({field: getattr(User, field) + delta})
        )
        await self.session.execute(q)

    async def update_linkedin_token(
        self,
        user_id: uuid.UUID,
        access_token: str,
        linkedin_url: str | None = None,
    ) -> None:
        """Store LinkedIn OAuth access token (encrypted in production)."""
        updates: dict[str, Any] = {"linkedin_access_token": access_token}
        if linkedin_url:
            updates["linkedin_url"] = linkedin_url
        await self.update(user_id, **updates)

    async def hard_delete_for_gdpr(self, user_id: uuid.UUID) -> None:
        """
        GDPR right-to-erasure: permanently delete user record.
        Should only be called after all related data has been anonymized.
        """
        user = await self.get_by_id_or_raise(user_id)
        await self.session.delete(user)
        await self.session.flush()
        logger.warning(
            "HARD DELETE: User permanently erased (GDPR)",
            user_id=str(user_id),
        )

    async def anonymize_user(self, user_id: uuid.UUID) -> None:
        """
        Anonymize user data for GDPR deletion while keeping aggregates.
        Replaces PII with anonymized values, marks account as deleted.
        """
        anon_email = f"deleted_{uuid.uuid4().hex[:8]}@deleted.invalid"
        await self.update(
            user_id,
            email=anon_email,
            full_name="[Deleted User]",
            avatar_url=None,
            linkedin_url=None,
            linkedin_access_token=None,
            github_url=None,
            portfolio_url=None,
            target_roles=None,
            target_locations=None,
            password_reset_token=None,
            email_verify_token=None,
            is_active=False,
            is_deleted=True,
        )
        logger.info("User anonymized for GDPR", user_id=str(user_id))


# ── Audit Log Repository ──────────────────────────────────────────────────────

class AuditLogRepository(BaseRepository):
    """
    Audit log repository — immutable records only.
    No update() or delete() methods exposed.
    """
    model = AuditLog

    def _base_query(self):
        """AuditLog has no soft-delete — return all records."""
        return select(self.model)

    async def get_by_user(
        self,
        user_id: uuid.UUID,
        *,
        action: str | None = None,
        skip: int = 0,
        limit: int = 50,
    ) -> list[AuditLog]:
        """Fetch audit logs for a specific user (admin panel)."""
        q = (
            select(AuditLog)
            .where(AuditLog.user_id == user_id)
            .order_by(desc(AuditLog.created_at))
            .offset(skip)
            .limit(limit)
        )
        if action:
            q = q.where(AuditLog.event == action)
        result = await self.session.execute(q)
        return list(result.scalars().all())

    async def get_by_action(
        self,
        action: str,
        *,
        since_hours: int = 24,
        limit: int = 100,
    ) -> list[AuditLog]:
        """Fetch recent logs by action type (security monitoring)."""
        since = text(f"NOW() - INTERVAL '{since_hours} hours'")
        q = (
            select(AuditLog)
            .where(AuditLog.event == action)
            .where(AuditLog.created_at >= since)
            .order_by(desc(AuditLog.created_at))
            .limit(limit)
        )
        result = await self.session.execute(q)
        return list(result.scalars().all())

    async def count_by_action(
        self,
        action: str,
        *,
        since_hours: int = 1,
        ip_address: str | None = None,
    ) -> int:
        """
        Count occurrences of an action in the last N hours.
        Used for rate limiting and brute-force detection.
        Example: count failed logins from an IP in the last hour.
        """
        since = text(f"NOW() - INTERVAL '{since_hours} hours'")
        q = (
            select(func.count())
            .select_from(AuditLog)
            .where(AuditLog.event == action)
            .where(AuditLog.created_at >= since)
        )
        if ip_address:
            q = q.where(AuditLog.ip_address == ip_address)
        result = await self.session.execute(q)
        return result.scalar_one() or 0

    async def get_security_events(
        self,
        *,
        since_hours: int = 24,
        limit: int = 200,
    ) -> list[AuditLog]:
        """Fetch all security-related events (failed logins, pw resets, etc.)."""
        security_actions = [
            "user.login_failed",
            "user.password_reset_requested",
            "user.password_changed",
            "user.account_deactivated",
            "user.registered",
        ]
        since = text(f"NOW() - INTERVAL '{since_hours} hours'")
        q = (
            select(AuditLog)
            .where(AuditLog.event.in_(security_actions))
            .where(AuditLog.created_at >= since)
            .order_by(desc(AuditLog.created_at))
            .limit(limit)
        )
        result = await self.session.execute(q)
        return list(result.scalars().all())

    # Explicitly block mutation operations
    async def update(self, *args: Any, **kwargs: Any) -> Any:  # type: ignore[override]
        raise NotImplementedError("AuditLog records are immutable — no updates allowed")

    async def delete(self, *args: Any, **kwargs: Any) -> None:  # type: ignore[override]
        raise NotImplementedError("AuditLog records are immutable — no deletes allowed")


__all__ = ["BaseRepository", "UserRepository", "AuditLogRepository"]