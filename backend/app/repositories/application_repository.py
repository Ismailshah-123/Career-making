"""
CareerGPT — Application Repository
=====================================
PAGE SUMMARY:
  Data access layer for Application model.
  All application pipeline SQL lives here.

  KEY METHODS:
    get_by_user_and_job()      → duplicate check before create
    get_user_applications()    → paginated list with optional status filter
    get_pipeline_stats()       → count per status (Kanban board data)
    get_recent_activity()      → latest N applications (dashboard feed)
    get_auto_apply_queue()     → applications pending auto-apply
    get_stale_for_followup()   → applications ready for follow-up generation
    count_user_applications()  → total count for pagination
    update_status()            → status transition with timestamp
    get_with_job()             → application + eagerly loaded job
    get_match_score_stats()    → avg/min/max match scores per user
    get_applications_by_ats()  → filter by ATS platform (analytics)
    get_conversion_funnel()    → applied→viewed→interview→offer rates
    bulk_mark_applied()        → mark batch as applied after scraper run
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import and_, desc, func, or_, select, text, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.constants import ApplicationStatus, FOLLOWUP_AFTER_DAYS
from app.core.logging import logger
from app.db.models.application import Application
from app.db.models.job import Job
from app.repositories.user_repository import BaseRepository


class ApplicationRepository(BaseRepository):
    model = Application

    # ── Core Lookups ──────────────────────────────────────────────────────────

    async def get_by_user_and_job(
        self,
        user_id: uuid.UUID,
        job_id: uuid.UUID,
    ) -> Application | None:
        """
        Check if user already applied to a job.
        Used for duplicate prevention in ApplicationService.create_application().
        """
        q = (
            self._base_query()
            .where(Application.user_id == user_id)
            .where(Application.job_id == job_id)
        )
        result = await self.session.execute(q)
        return result.scalar_one_or_none()

    async def get_with_job(self, application_id: uuid.UUID) -> Application | None:
        """Fetch application with eagerly loaded Job relationship."""
        q = (
            select(Application)
            .options(selectinload(Application.job))
            .where(Application.id == application_id)
        )
        result = await self.session.execute(q)
        return result.scalar_one_or_none()

    async def get_by_id_or_raise(self, application_id: uuid.UUID) -> Application:
        """Fetch with job loaded. Raises NotFoundError if missing."""
        from app.core.exceptions import ApplicationNotFoundError
        app = await self.get_with_job(application_id)
        if not app:
            raise ApplicationNotFoundError(context={"id": str(application_id)})
        return app

    # ── List Queries ──────────────────────────────────────────────────────────

    async def get_user_applications(
        self,
        user_id: uuid.UUID,
        *,
        status: str | None = None,
        skip: int = 0,
        limit: int = 50,
        order_by_recent: bool = True,
    ) -> list[Application]:
        """
        Paginated list of user applications with optional status filter.
        Eagerly loads Job relationship for each application.
        """
        q = (
            select(Application)
            .options(selectinload(Application.job))
            .where(Application.user_id == user_id)
        )
        if status:
            q = q.where(Application.status == status)
        if order_by_recent:
            q = q.order_by(desc(Application.updated_at))
        q = q.offset(skip).limit(limit)
        result = await self.session.execute(q)
        return list(result.scalars().all())

    async def count_user_applications(
        self,
        user_id: uuid.UUID,
        *,
        status: str | None = None,
    ) -> int:
        """Total count of user applications (for pagination metadata)."""
        q = (
            select(func.count())
            .select_from(Application)
            .where(Application.user_id == user_id)
        )
        if status:
            q = q.where(Application.status == status)
        result = await self.session.execute(q)
        return result.scalar_one() or 0

    async def get_recent_activity(
        self,
        user_id: uuid.UUID,
        *,
        limit: int = 10,
    ) -> list[Application]:
        """
        Most recently updated applications.
        Used for: dashboard activity feed, notification generation.
        """
        q = (
            select(Application)
            .options(selectinload(Application.job))
            .where(Application.user_id == user_id)
            .order_by(desc(Application.updated_at))
            .limit(limit)
        )
        result = await self.session.execute(q)
        return list(result.scalars().all())

    # ── Pipeline & Stats ──────────────────────────────────────────────────────

    async def get_pipeline_stats(self, user_id: uuid.UUID) -> dict[str, int]:
        """
        Return application count per pipeline stage.
        Used by: dashboard Kanban view, pipeline stats widget.
        Returns: {pending: 5, applied: 12, viewed: 3, interview: 2, ...}
        """
        q = (
            select(
                Application.status,
                func.count(Application.id).label("cnt"),
            )
            .where(Application.user_id == user_id)
            .group_by(Application.status)
        )
        result = await self.session.execute(q)
        return {row.status: row.cnt for row in result.all()}

    async def get_match_score_stats(self, user_id: uuid.UUID) -> dict[str, float]:
        """Return match score statistics for a user's applications."""
        q = (
            select(
                func.avg(Application.match_score).label("avg"),
                func.min(Application.match_score).label("min"),
                func.max(Application.match_score).label("max"),
                func.count(Application.id).filter(
                    Application.match_score >= 0.8
                ).label("high_match_count"),
            )
            .where(Application.user_id == user_id)
            .where(Application.match_score.isnot(None))
        )
        result = await self.session.execute(q)
        row = result.one()
        return {
            "avg":              round(float(row.avg or 0), 3),
            "min":              round(float(row.min or 0), 3),
            "max":              round(float(row.max or 0), 3),
            "high_match_count": int(row.high_match_count or 0),
        }

    async def get_conversion_funnel(self, user_id: uuid.UUID) -> dict[str, Any]:
        """
        Calculate conversion rates through the application pipeline.
        Used by analytics page.
        Returns stage-by-stage conversion percentages.
        """
        stats = await self.get_pipeline_stats(user_id)
        total = sum(stats.values()) or 1

        applied_plus = (
            stats.get("applied", 0)
            + stats.get("viewed", 0)
            + stats.get("interview", 0)
            + stats.get("offer", 0)
        )
        viewed_plus = (
            stats.get("viewed", 0)
            + stats.get("interview", 0)
            + stats.get("offer", 0)
        )
        interview_plus = stats.get("interview", 0) + stats.get("offer", 0)

        return {
            "total_created":        total,
            "applied_rate":         round(applied_plus / total * 100, 1),
            "view_rate":            round(viewed_plus / max(applied_plus, 1) * 100, 1),
            "interview_rate":       round(interview_plus / max(applied_plus, 1) * 100, 1),
            "offer_rate":           round(stats.get("offer", 0) / max(interview_plus, 1) * 100, 1),
            "rejection_rate":       round(stats.get("rejected", 0) / total * 100, 1),
            "stages": stats,
        }

    async def get_weekly_activity(
        self,
        user_id: uuid.UUID,
        weeks: int = 8,
    ) -> list[dict[str, Any]]:
        """
        Return applications per week for the last N weeks.
        Used by: activity chart on analytics page.
        """
        q = text("""
            SELECT
                DATE_TRUNC('week', created_at AT TIME ZONE 'UTC') AS week_start,
                COUNT(*) AS count
            FROM applications
            WHERE user_id = :user_id
              AND created_at >= NOW() - INTERVAL ':weeks weeks'
            GROUP BY DATE_TRUNC('week', created_at AT TIME ZONE 'UTC')
            ORDER BY week_start DESC
        """).bindparams(user_id=str(user_id), weeks=weeks)

        try:
            result = await self.session.execute(q)
            return [
                {
                    "week":  row.week_start.strftime("%Y-%m-%d"),
                    "count": row.count,
                }
                for row in result.all()
            ]
        except Exception:
            return []

    # ── Follow-up Queries ─────────────────────────────────────────────────────

    async def get_stale_for_followup(
        self,
        user_id: uuid.UUID,
        *,
        min_days: int = FOLLOWUP_AFTER_DAYS,
        max_followups: int = 3,
    ) -> list[Application]:
        """
        Find applied applications that are stale and eligible for follow-up.
        Criteria:
          - Status = 'applied' or 'pending'
          - Applied at least min_days ago
          - followup_count < max_followups
          - No followup in last 7 days
        """
        cutoff = datetime.now(UTC) - timedelta(days=min_days)
        recent_followup_cutoff = datetime.now(UTC) - timedelta(days=7)

        q = (
            select(Application)
            .options(selectinload(Application.job))
            .where(Application.user_id == user_id)
            .where(
                Application.status.in_([
                    ApplicationStatus.APPLIED.value,
                    ApplicationStatus.PENDING.value,
                ])
            )
            .where(Application.applied_at <= cutoff)
            .where(Application.followup_count < max_followups)
            .where(
                or_(
                    Application.last_followup_at.is_(None),
                    Application.last_followup_at <= recent_followup_cutoff,
                )
            )
            .order_by(Application.applied_at.asc())
        )
        result = await self.session.execute(q)
        return list(result.scalars().all())

    # ── Auto-apply Queue ──────────────────────────────────────────────────────

    async def get_auto_apply_queue(
        self,
        user_id: uuid.UUID,
        *,
        limit: int = 10,
    ) -> list[Application]:
        """
        Get applications queued for auto-apply (status=pending, auto_applied=False).
        Used by ApplicationAgent to process the queue.
        """
        q = (
            select(Application)
            .options(selectinload(Application.job))
            .options(selectinload(Application.resume))
            .where(Application.user_id == user_id)
            .where(Application.status == ApplicationStatus.PENDING.value)
            .where(Application.auto_applied.is_(False))
            .where(Application.application_error.is_(None))
            .order_by(Application.created_at.asc())
            .limit(limit)
        )
        result = await self.session.execute(q)
        return list(result.scalars().all())

    # ── Status Updates ────────────────────────────────────────────────────────

    async def update_status(
        self,
        application_id: uuid.UUID,
        status: str,
        *,
        extra_fields: dict[str, Any] | None = None,
    ) -> Application:
        """Update status with automatic timestamp tracking."""
        updates: dict[str, Any] = {
            "status": status,
            "last_status_change_at": datetime.now(UTC),
        }
        if status == ApplicationStatus.APPLIED.value:
            updates["applied_at"] = datetime.now(UTC)
        if extra_fields:
            updates.update(extra_fields)
        return await self.update(application_id, **updates)

    async def mark_auto_applied(
        self,
        application_id: uuid.UUID,
        *,
        ats_platform: str,
        screenshots: list[str] | None = None,
    ) -> Application:
        """Mark application as successfully auto-applied."""
        import json
        return await self.update(
            application_id,
            status=ApplicationStatus.APPLIED.value,
            applied_at=datetime.now(UTC),
            last_status_change_at=datetime.now(UTC),
            auto_applied=True,
            ats_platform=ats_platform,
            screenshots=json.dumps(screenshots or []),
            application_error=None,
        )

    async def mark_apply_failed(
        self,
        application_id: uuid.UUID,
        error_message: str,
    ) -> Application:
        """Mark application auto-apply as failed with error message."""
        return await self.update(
            application_id,
            application_error=error_message[:1000],
        )

    async def bulk_mark_applied(
        self,
        application_ids: list[uuid.UUID],
    ) -> int:
        """Bulk update multiple applications to 'applied' status."""
        if not application_ids:
            return 0
        q = (
            update(Application)
            .where(Application.id.in_(application_ids))
            .values(
                status=ApplicationStatus.APPLIED.value,
                applied_at=datetime.now(UTC),
                last_status_change_at=datetime.now(UTC),
            )
        )
        result = await self.session.execute(q)
        return result.rowcount or 0


__all__ = ["ApplicationRepository"]