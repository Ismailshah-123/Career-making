"""
CareerGPT — LinkedIn Service
==============================
PAGE SUMMARY:
  Orchestrates all LinkedIn-related operations through the LinkedInAgent.
  Handles post generation, carousel creation, scheduling, publishing,
  engagement sync, OAuth token management, and content calendar.

  PUBLIC API:
    generate_post()            → generate a new LinkedIn post draft
    generate_carousel()        → generate a multi-slide carousel post
    publish_post()             → publish a saved draft immediately
    schedule_post()            → schedule a post for future publishing
    regenerate_post()          → user didn't like draft, regenerate
    get_post()                 → fetch single post by ID
    list_posts()               → paginated post list with status filter
    delete_post()              → delete a draft (not published)
    sync_engagement()          → sync likes/impressions from LinkedIn API
    sync_all_engagement()      → sync all posts for a user (nightly task)
    get_analytics()            → post performance analytics
    get_content_calendar()     → upcoming scheduled posts calendar view
    connect_linkedin_oauth()   → initiate LinkedIn OAuth flow
    handle_oauth_callback()    → process OAuth callback, store token
    disconnect_linkedin()      → remove LinkedIn access token
    get_optimal_posting_time() → recommend best time to post

  IMAGE GENERATION IN POST FLOW:
    generate_post() has force_image parameter:
      None  = agent auto-decides (based on format/content signals)
      True  = always generate image
      False = never generate image
    Images are generated via LinkedInAgent → tools.generate_post_image()
    which tries: Pollinations.ai (FREE) → Together.ai → SVG fallback.
    Image path stored in _image_cache for publish step.

  SCHEDULING:
    scheduled_at field accepts UTC datetime.
    Celery beat task (linkedin_tasks.publish_due_scheduled) runs every
    5 minutes and publishes all posts where:
      status = "scheduled" AND scheduled_at <= now()
    get_optimal_posting_time() returns next Tue/Wed/Thu 8-10am UTC.

  USED BY:
    API routes: /api/v1/linkedin/*
    Celery workers: linkedin_tasks.py
    FollowupAgent: career milestone post suggestions
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.constants import LinkedInPostStatus
from app.core.exceptions import (
    LinkedInError,
    PostNotFoundError,
    UserNotFoundError,
    ValidationError,
)
from app.core.logging import log_context, logger


class LinkedInService:
    """
    LinkedIn content and publishing service.
    All heavy lifting delegated to LinkedInAgent.
    """

    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    # ── Post Generation ───────────────────────────────────────────────────────

    async def generate_post(
        self,
        user_id: uuid.UUID,
        *,
        topic_override: str | None = None,
        tone: str = "thought_leader",
        post_format: str = "insight",
        force_image: bool | None = None,
        schedule_at: datetime | None = None,
        custom_hook: str | None = None,
    ) -> dict[str, Any]:
        """
        Generate a LinkedIn post draft via LinkedInAgent.
        Returns full post data including quality score, image path if generated.
        """
        with log_context(service="linkedin", user_id=str(user_id)):
            await self._validate_user(user_id)

            from app.agents.linkedin_agent.agent import LinkedInAgent
            agent = LinkedInAgent(self.db)

            result = await agent.generate_post(
                user_id,
                topic_override=topic_override,
                tone=tone,
                post_format=post_format,
                force_image=force_image,
                schedule_at=schedule_at,
                custom_hook=custom_hook,
            )

            logger.info(
                "LinkedIn post generated",
                user_id=str(user_id),
                post_id=result.get("id"),
                quality_score=result.get("quality_score"),
                has_image=result.get("has_image"),
            )
            return result

    async def generate_carousel(
        self,
        user_id: uuid.UUID,
        *,
        topic: str,
        key_points: list[str],
        tone: str = "educational",
    ) -> dict[str, Any]:
        """
        Generate a LinkedIn carousel (multi-slide PDF post).
        Returns {id, title, slide_count, slides, pdf_path, caption, hashtags}.
        """
        await self._validate_user(user_id)

        from app.agents.linkedin_agent.agent import LinkedInAgent
        agent = LinkedInAgent(self.db)

        return await agent.generate_carousel(
            user_id,
            topic=topic,
            key_points=key_points,
            tone=tone,
        )

    # ── Publishing ────────────────────────────────────────────────────────────

    async def publish_post(
        self,
        post_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> dict[str, Any]:
        """
        Publish a saved draft post to LinkedIn immediately.
        Tries API → Playwright fallback → marks failed.
        Returns {post_id, linkedin_post_id, status, method, published_at}.
        """
        with log_context(service="linkedin_publish", post_id=str(post_id)):
            await self._validate_post_ownership(post_id, user_id)

            from app.agents.linkedin_agent.agent import LinkedInAgent
            agent = LinkedInAgent(self.db)
            result = await agent.publish_post(post_id, user_id)

            logger.info(
                "Post published",
                post_id=str(post_id),
                method=result.get("method"),
            )
            return result

    async def schedule_post(
        self,
        post_id: uuid.UUID,
        user_id: uuid.UUID,
        scheduled_at: datetime,
    ) -> dict[str, Any]:
        """Schedule an existing draft for future publication."""
        from app.repositories.linkedin_repository import LinkedInRepository
        await self._validate_post_ownership(post_id, user_id)

        repo = LinkedInRepository(self.db)
        post = await repo.update(
            post_id,
            status=LinkedInPostStatus.SCHEDULED.value,
            scheduled_at=scheduled_at,
        )

        logger.info(
            "Post scheduled",
            post_id=str(post_id),
            scheduled_at=scheduled_at.isoformat(),
        )
        return {
            "post_id":      str(post_id),
            "status":       "scheduled",
            "scheduled_at": scheduled_at.isoformat(),
        }

    # ── Regeneration ──────────────────────────────────────────────────────────

    async def regenerate_post(
        self,
        post_id: uuid.UUID,
        user_id: uuid.UUID,
        feedback: str,
    ) -> dict[str, Any]:
        """Regenerate a draft with user's specific feedback."""
        await self._validate_post_ownership(post_id, user_id)

        from app.agents.linkedin_agent.agent import LinkedInAgent
        agent = LinkedInAgent(self.db)
        return await agent.regenerate_with_feedback(post_id, user_id, feedback)

    # ── CRUD ──────────────────────────────────────────────────────────────────

    async def get_post(
        self,
        post_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> dict[str, Any]:
        """Fetch a single LinkedIn post by ID."""
        from app.repositories.linkedin_repository import LinkedInRepository
        repo = LinkedInRepository(self.db)
        post = await repo.get_by_id_or_raise(post_id)

        if post.user_id != user_id:
            raise PostNotFoundError()

        return self._serialize_post(post)

    async def list_posts(
        self,
        user_id: uuid.UUID,
        *,
        status: str | None = None,
        skip: int = 0,
        limit: int = 20,
    ) -> dict[str, Any]:
        """List user's LinkedIn posts with pagination and status filter."""
        from app.repositories.linkedin_repository import LinkedInRepository
        repo  = LinkedInRepository(self.db)
        posts = await repo.get_user_posts(user_id, status=status, limit=limit, offset=skip)
        total = await repo.count_user_posts(user_id, status=status)

        return {
            "posts": [self._serialize_post(p) for p in posts],
            "total": total,
            "skip":  skip,
            "limit": limit,
        }

    async def delete_post(
        self,
        post_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> None:
        """Delete a draft post. Cannot delete published posts."""
        from app.repositories.linkedin_repository import LinkedInRepository
        repo = LinkedInRepository(self.db)
        post = await repo.get_by_id_or_raise(post_id)

        if post.user_id != user_id:
            raise PostNotFoundError()

        if post.status == LinkedInPostStatus.PUBLISHED.value:
            raise ValidationError("Cannot delete a published LinkedIn post")

        await repo.soft_delete(post_id)

    # ── Engagement Sync ────────────────────────────────────────────────────────

    async def sync_engagement(
        self,
        post_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> dict[str, Any]:
        """Sync engagement metrics for a single post from LinkedIn API."""
        from app.repositories.linkedin_repository import LinkedInRepository
        from app.repositories.user_repository import UserRepository
        from app.agents.linkedin_agent.tools import sync_post_engagement

        user_repo = UserRepository(self.db)
        user      = await user_repo.get_by_id_or_raise(user_id)

        if not user.linkedin_access_token:
            return {"error": "LinkedIn not connected", "synced": False}

        li_repo = LinkedInRepository(self.db)
        post    = await li_repo.get_by_id_or_raise(post_id)

        if not post.linkedin_post_id:
            return {"error": "Post not yet published to LinkedIn", "synced": False}

        metrics = await sync_post_engagement(
            user.linkedin_access_token,
            post.linkedin_post_id,
        )
        await li_repo.update_engagement(
            post_id,
            likes=metrics.get("likes", 0),
            comments=metrics.get("comments", 0),
            shares=metrics.get("shares", 0),
        )
        return {**metrics, "synced": True, "post_id": str(post_id)}

    async def sync_all_engagement(self, user_id: uuid.UUID) -> dict[str, Any]:
        """Sync engagement for all published posts (called by nightly Celery task)."""
        from app.agents.linkedin_agent.agent import LinkedInAgent
        agent = LinkedInAgent(self.db)
        return await agent.sync_all_engagement(user_id)

    # ── Analytics ─────────────────────────────────────────────────────────────

    async def get_analytics(self, user_id: uuid.UUID) -> dict[str, Any]:
        """Return post performance analytics for the dashboard."""
        from app.repositories.linkedin_repository import LinkedInRepository
        repo = LinkedInRepository(self.db)

        stats      = await repo.get_engagement_stats(user_id)
        top_posts  = await repo.get_top_posts(user_id, limit=5)
        post_count = await repo.count_user_posts(user_id)

        return {
            "total_posts":       post_count,
            "posts_this_month":  stats.get("posts_this_month", 0),
            "total_impressions": stats.get("total_impressions", 0),
            "total_likes":       stats.get("total_likes", 0),
            "total_comments":    stats.get("total_comments", 0),
            "avg_engagement_rate": stats.get("avg_engagement_rate", 0),
            "top_posts":         [self._serialize_post_brief(p) for p in top_posts],
        }

    # ── Content Calendar ───────────────────────────────────────────────────────

    async def get_content_calendar(
        self,
        user_id: uuid.UUID,
        *,
        days: int = 30,
    ) -> list[dict[str, Any]]:
        """Return upcoming scheduled posts for calendar view."""
        from app.repositories.linkedin_repository import LinkedInRepository
        repo  = LinkedInRepository(self.db)
        posts = await repo.get_scheduled_posts(user_id, days=days)
        return [
            {
                "post_id":      str(p.id),
                "topic":        p.topic,
                "hook":         p.hook[:80] if p.hook else "",
                "scheduled_at": p.scheduled_at.isoformat() if p.scheduled_at else None,
                "status":       p.status,
                "has_image":    False,  # check from image_cache or stored field
            }
            for p in posts
        ]

    # ── OAuth ─────────────────────────────────────────────────────────────────

    async def get_oauth_url(self) -> dict[str, str]:
        """Return LinkedIn OAuth authorization URL."""
        from app.core.config import get_settings
        settings = get_settings()
        client_id    = settings.linkedin.client_id
        redirect_uri = settings.linkedin.redirect_uri
        scope        = "r_liteprofile r_emailaddress w_member_social"

        auth_url = (
            f"https://www.linkedin.com/oauth/v2/authorization"
            f"?response_type=code"
            f"&client_id={client_id}"
            f"&redirect_uri={redirect_uri}"
            f"&scope={scope.replace(' ', '%20')}"
        )
        return {"auth_url": auth_url}

    async def handle_oauth_callback(
        self,
        user_id: uuid.UUID,
        code: str,
    ) -> dict[str, Any]:
        """Exchange OAuth code for access token and store it."""
        import httpx
        from app.core.config import get_settings
        settings = get_settings()

        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                "https://www.linkedin.com/oauth/v2/accessToken",
                data={
                    "grant_type":    "authorization_code",
                    "code":          code,
                    "client_id":     settings.linkedin.client_id,
                    "client_secret": settings.linkedin.client_secret,
                    "redirect_uri":  settings.linkedin.redirect_uri,
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            if resp.status_code != 200:
                raise LinkedInError(
                    f"OAuth token exchange failed: {resp.status_code}",
                    context={"response": resp.text[:200]},
                )
            token_data = resp.json()

        from app.services.user_service import UserService
        user_svc = UserService(self.db)
        await user_svc.update_linkedin_token(
            user_id,
            token_data["access_token"],
            expires_in_seconds=token_data.get("expires_in", 5184000),
        )

        return {
            "connected":    True,
            "expires_in":   token_data.get("expires_in"),
        }

    async def get_optimal_posting_time(self, user_id: uuid.UUID) -> dict[str, Any]:
        """Return next optimal LinkedIn posting time for a user."""
        from app.utils.datetime_utils import get_optimal_send_time
        from app.repositories.user_repository import UserRepository

        repo = UserRepository(self.db)
        user = await repo.get_by_id_or_raise(user_id)

        optimal = get_optimal_send_time(
            send_type="linkedin",
            recipient_tz=user.timezone or "America/New_York",
        )
        return {
            "optimal_utc":   optimal.isoformat(),
            "optimal_local": optimal.strftime("%A, %B %d at %I:%M %p"),
            "reasoning":     "Tuesday-Thursday 8-10am has the highest LinkedIn engagement",
        }

    # ── Private ───────────────────────────────────────────────────────────────

    async def _validate_user(self, user_id: uuid.UUID) -> None:
        from app.repositories.user_repository import UserRepository
        repo = UserRepository(self.db)
        await repo.get_by_id_or_raise(user_id)

    async def _validate_post_ownership(
        self, post_id: uuid.UUID, user_id: uuid.UUID
    ) -> None:
        from app.repositories.linkedin_repository import LinkedInRepository
        repo = LinkedInRepository(self.db)
        post = await repo.get_by_id_or_raise(post_id)
        if post.user_id != user_id:
            raise PostNotFoundError()

    def _serialize_post(self, post: Any) -> dict[str, Any]:
        return {
            "id":               str(post.id),
            "topic":            post.topic,
            "hook":             post.hook,
            "body":             post.body,
            "full_post":        f"{post.hook}\n\n{post.body}" if post.hook and post.body else post.body or "",
            "hashtags":         json.loads(post.hashtags) if post.hashtags else [],
            "tone":             post.tone,
            "status":           post.status,
            "scheduled_at":     post.scheduled_at.isoformat() if post.scheduled_at else None,
            "published_at":     post.published_at.isoformat() if post.published_at else None,
            "linkedin_post_id": post.linkedin_post_id,
            "likes":            post.likes or 0,
            "comments":         post.comments or 0,
            "shares":           post.shares or 0,
            "impressions":      post.impressions or 0,
            "publish_error":    post.publish_error,
            "ai_model_used":    post.ai_model_used,
            "created_at":       post.created_at.isoformat() if post.created_at else None,
        }

    def _serialize_post_brief(self, post: Any) -> dict[str, Any]:
        return {
            "id":          str(post.id),
            "topic":       post.topic,
            "hook":        (post.hook or "")[:80],
            "status":      post.status,
            "likes":       post.likes or 0,
            "impressions": post.impressions or 0,
        }


__all__ = ["LinkedInService"]