"""
CareerGPT — LinkedIn Repository
==================================
PAGE SUMMARY:
  Data access layer for LinkedInPost model.
  Manages: post creation, scheduling queue, engagement updates,
  publishing status tracking, analytics aggregation.

  KEY METHODS:
    get_user_posts()           → paginated post list with status filter
    get_scheduled_due()        → posts ready to publish (Celery beat polls this)
    update_engagement()        → sync impressions/likes from LinkedIn API
    get_engagement_stats()     → aggregated engagement analytics per user
    get_posts_by_status()      → filter by draft/scheduled/published/failed
    get_top_performing()       → sort by impressions+likes for best content insights
    count_published_this_week()→ dashboard stat: posts this week
    get_content_calendar()     → scheduled posts grouped by day (content calendar view)
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import and_, desc, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.constants import LinkedInPostStatus
from app.core.logging import logger
from app.db.models.linkedin_post import LinkedInPost
from app.repositories.user_repository import BaseRepository


class LinkedInRepository(BaseRepository):
    model = LinkedInPost

    async def get_user_posts(
        self,
        user_id: uuid.UUID,
        *,
        status: str | None = None,
        skip: int = 0,
        limit: int = 30,
    ) -> list[LinkedInPost]:
        """Paginated list of LinkedIn posts for a user."""
        q = (
            self._base_query()
            .where(LinkedInPost.user_id == user_id)
            .order_by(desc(LinkedInPost.created_at))
            .offset(skip)
            .limit(limit)
        )
        if status:
            q = q.where(LinkedInPost.status == status)
        result = await self.session.execute(q)
        return list(result.scalars().all())

    async def get_scheduled_due(
        self,
        *,
        before: datetime | None = None,
        limit: int = 20,
    ) -> list[LinkedInPost]:
        """
        Fetch scheduled posts that are due to be published.
        Called every 5 minutes by Celery beat task.
        """
        cutoff = before or datetime.now(UTC)
        q = (
            self._base_query()
            .where(LinkedInPost.status == LinkedInPostStatus.SCHEDULED.value)
            .where(LinkedInPost.scheduled_at <= cutoff)
            .order_by(LinkedInPost.scheduled_at.asc())
            .limit(limit)
        )
        result = await self.session.execute(q)
        return list(result.scalars().all())

    async def update_engagement(
        self,
        post_id: uuid.UUID,
        *,
        impressions: int | None = None,
        likes: int | None = None,
        comments: int | None = None,
        shares: int | None = None,
    ) -> LinkedInPost:
        """Update engagement metrics from LinkedIn API sync."""
        updates: dict[str, Any] = {}
        if impressions is not None:
            updates["impressions"] = impressions
        if likes is not None:
            updates["likes"] = likes
        if comments is not None:
            updates["comments"] = comments
        if shares is not None:
            updates["shares"] = shares
        return await self.update(post_id, **updates)

    async def mark_published(
        self,
        post_id: uuid.UUID,
        linkedin_post_id: str,
    ) -> LinkedInPost:
        """Mark post as published with LinkedIn's returned post ID."""
        return await self.update(
            post_id,
            status=LinkedInPostStatus.PUBLISHED.value,
            published_at=datetime.now(UTC),
            linkedin_post_id=linkedin_post_id,
            publish_error=None,
        )

    async def mark_failed(
        self,
        post_id: uuid.UUID,
        error: str,
    ) -> LinkedInPost:
        """Mark post publishing as failed."""
        return await self.update(
            post_id,
            status=LinkedInPostStatus.FAILED.value,
            publish_error=error[:1000],
        )

    async def get_engagement_stats(
        self,
        user_id: uuid.UUID,
    ) -> dict[str, Any]:
        """Aggregate engagement metrics across all published posts."""
        q = (
            select(
                func.count(LinkedInPost.id).label("total_posts"),
                func.sum(LinkedInPost.impressions).label("total_impressions"),
                func.sum(LinkedInPost.likes).label("total_likes"),
                func.sum(LinkedInPost.comments).label("total_comments"),
                func.sum(LinkedInPost.shares).label("total_shares"),
                func.avg(LinkedInPost.impressions).label("avg_impressions"),
                func.avg(LinkedInPost.likes).label("avg_likes"),
            )
            .where(LinkedInPost.user_id == user_id)
            .where(LinkedInPost.status == LinkedInPostStatus.PUBLISHED.value)
        )
        result = await self.session.execute(q)
        row = result.one()

        total_impressions = int(row.total_impressions or 0)
        total_likes = int(row.total_likes or 0)
        total_comments = int(row.total_comments or 0)
        total_shares = int(row.total_shares or 0)
        total_engagements = total_likes + total_comments + total_shares
        total_posts = int(row.total_posts or 0)

        return {
            "total_posts":        total_posts,
            "total_impressions":  total_impressions,
            "total_likes":        total_likes,
            "total_comments":     total_comments,
            "total_shares":       total_shares,
            "total_engagements":  total_engagements,
            "avg_impressions":    round(float(row.avg_impressions or 0), 1),
            "avg_likes":          round(float(row.avg_likes or 0), 1),
            "engagement_rate_pct": round(
                total_engagements / total_impressions * 100 if total_impressions > 0 else 0, 2
            ),
        }

    async def get_top_performing(
        self,
        user_id: uuid.UUID,
        *,
        limit: int = 5,
    ) -> list[LinkedInPost]:
        """Return top performing published posts by impressions."""
        q = (
            self._base_query()
            .where(LinkedInPost.user_id == user_id)
            .where(LinkedInPost.status == LinkedInPostStatus.PUBLISHED.value)
            .order_by(desc(LinkedInPost.impressions))
            .limit(limit)
        )
        result = await self.session.execute(q)
        return list(result.scalars().all())

    async def count_published_this_week(self, user_id: uuid.UUID) -> int:
        """Count posts published in the last 7 days."""
        cutoff = datetime.now(UTC) - timedelta(days=7)
        q = (
            select(func.count())
            .select_from(LinkedInPost)
            .where(LinkedInPost.user_id == user_id)
            .where(LinkedInPost.status == LinkedInPostStatus.PUBLISHED.value)
            .where(LinkedInPost.published_at >= cutoff)
        )
        result = await self.session.execute(q)
        return result.scalar_one() or 0

    async def get_content_calendar(
        self,
        user_id: uuid.UUID,
        *,
        days_ahead: int = 14,
    ) -> list[dict[str, Any]]:
        """Return scheduled posts grouped by day for content calendar view."""
        cutoff = datetime.now(UTC) + timedelta(days=days_ahead)
        q = (
            self._base_query()
            .where(LinkedInPost.user_id == user_id)
            .where(
                LinkedInPost.status.in_([
                    LinkedInPostStatus.SCHEDULED.value,
                    LinkedInPostStatus.DRAFT.value,
                ])
            )
            .where(
                LinkedInPost.scheduled_at <= cutoff
            )
            .order_by(LinkedInPost.scheduled_at.asc().nullslast())
        )
        result = await self.session.execute(q)
        posts = result.scalars().all()

        calendar: dict[str, list[dict]] = {}
        for post in posts:
            if post.scheduled_at:
                day_key = post.scheduled_at.strftime("%Y-%m-%d")
            else:
                day_key = "unscheduled"
            if day_key not in calendar:
                calendar[day_key] = []
            calendar[day_key].append({
                "id":           str(post.id),
                "hook":         post.hook,
                "status":       post.status,
                "scheduled_at": post.scheduled_at.isoformat() if post.scheduled_at else None,
                "tone":         post.tone,
            })

        return [{"date": k, "posts": v} for k, v in sorted(calendar.items())]

    async def retry_failed_posts(
        self,
        user_id: uuid.UUID,
        *,
        limit: int = 5,
    ) -> list[LinkedInPost]:
        """Return recently failed posts eligible for retry."""
        cutoff = datetime.now(UTC) - timedelta(hours=24)
        q = (
            self._base_query()
            .where(LinkedInPost.user_id == user_id)
            .where(LinkedInPost.status == LinkedInPostStatus.FAILED.value)
            .where(LinkedInPost.updated_at >= cutoff)
            .limit(limit)
        )
        result = await self.session.execute(q)
        return list(result.scalars().all())


__all__ = ["LinkedInRepository"]