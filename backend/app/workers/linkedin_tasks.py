"""
app/workers/linkedin_tasks.py
===============================
Celery task definitions for all LinkedIn content pipeline operations.

Tasks defined here:
    generate_single_post_task           — generate one AI post for a user + topic
    generate_daily_posts_task           — generate N-day content calendar (user-triggered)
    generate_daily_posts_for_all_users_task — Beat-scheduled fan-out at 02:00 UTC
    publish_scheduled_posts_task        — Beat every 10min: publish due posts via LinkedIn API
    publish_single_post_task            — publish one specific post immediately
    sync_post_analytics_task            — Beat every 6h: pull engagement metrics from LinkedIn
    sync_single_post_analytics_task     — pull metrics for one post on demand

Architecture:
    - generate_* tasks drive the linkedin_workflow.py LangGraph pipeline,
      which handles research → draft → quality score → approval gate.
    - publish_* tasks call services/linkedin_service.py directly — no graph
      needed, just a LinkedIn API call + DB update.
    - sync_* tasks call the LinkedIn analytics API and update LinkedInPost
      rows in bulk.
    - All tasks are wrapped with AgentRun lifecycle management.

LinkedIn rate limits respected:
    - Publishing: max 5 posts/day per user (enforced by plan limits +
      a Redis counter checked before each publish task executes)
    - Analytics API: max 100 requests/day (batched across all users)
    - Pause 30–90s between consecutive API calls to the same endpoint

Beat schedule integration (defined in celery_app.py):
    - 02:00 UTC daily  → generate_daily_posts_for_all_users_task
    - every 10 minutes → publish_scheduled_posts_task
    - every 6 hours    → sync_post_analytics_task
"""

from __future__ import annotations

import asyncio
import traceback
from datetime import datetime, timezone
from typing import Any

from celery import Task
from celery.exceptions import MaxRetriesExceededError, SoftTimeLimitExceeded

from app.workers.celery_app import celery_app, InstrumentedTask
from app.core.constants import AgentRunStatus, AGENT_LINKEDIN
from app.core.logging import get_logger, get_task_logger

logger = get_logger(__name__)


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


async def _update_agent_run(agent_run_id: str, **kwargs: Any) -> None:
    from app.db.session import get_db_context
    from app.db.models.agent_run import AgentRun
    from sqlalchemy import select
    import uuid

    async with get_db_context() as db:
        result = await db.execute(
            select(AgentRun).where(AgentRun.id == uuid.UUID(agent_run_id))
        )
        run = result.scalar_one_or_none()
        if not run:
            return
        for key, val in kwargs.items():
            if hasattr(run, key) and val is not None:
                setattr(run, key, val)
        if kwargs.get("status") == AgentRunStatus.RUNNING and not run.started_at:
            run.started_at = datetime.now(timezone.utc)
        if kwargs.get("status") in (
            AgentRunStatus.COMPLETED, AgentRunStatus.FAILED,
            AgentRunStatus.TIMED_OUT, AgentRunStatus.CANCELLED,
        ):
            run.completed_at = datetime.now(timezone.utc)
        await db.commit()


async def _create_agent_run(user_id: str, trigger: str = "api", input_payload: dict | None = None) -> str:
    from app.db.session import get_db_context
    from app.db.models.agent_run import AgentRun
    import uuid

    async with get_db_context() as db:
        run = AgentRun(
            user_id=uuid.UUID(user_id),
            agent_name=AGENT_LINKEDIN,
            trigger=trigger,
            status=AgentRunStatus.PENDING,
            input_payload=input_payload or {},
        )
        db.add(run)
        await db.flush()
        run_id = str(run.id)
        await db.commit()
    return run_id


# ---------------------------------------------------------------------------
# generate_single_post_task
# ---------------------------------------------------------------------------

@celery_app.task(
    base=InstrumentedTask,
    name="app.workers.linkedin_tasks.generate_single_post_task",
    bind=True,
    max_retries=3,
    default_retry_delay=60,
    soft_time_limit=300,
    time_limit=360,
    acks_late=True,
)
def generate_single_post_task(
    self: Task,
    *,
    user_id: str,
    agent_run_id: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    """
    Generate a single LinkedIn post for a user.

    config keys:
        topic              : str     — topic to write about
        category           : str     — LinkedInPostCategory value
        tone               : str     — thought-leader | professional | etc.
        requires_approval  : bool    — gate on user approval before publish
        include_hashtags   : bool
        include_cta        : bool
        max_length         : int     — max characters (≤ 3000)
        source_urls        : list    — URLs to ground the research in
        custom_instructions: str | None
        scheduled_at       : str | None — ISO 8601 datetime to schedule

    Returns the post_id of the created draft / scheduled post.
    """
    task_logger = get_task_logger("generate_single_post_task", task_id=self.request.id)
    task_logger.info(
        "LinkedIn post generation started",
        user_id=user_id,
        topic=config.get("topic", "")[:60],
    )
    start_ts = datetime.now(timezone.utc)

    try:
        _run(_update_agent_run(agent_run_id, status=AgentRunStatus.RUNNING))

        from app.workflows.linkedin_workflow import run_post_generation_workflow

        final_state = _run(
            run_post_generation_workflow(
                user_id=user_id,
                agent_run_id=agent_run_id,
                topic=config.get("topic", "AI and the future of work"),
                category=config.get("category", "ai_insights"),
                tone=config.get("tone", "thought-leader"),
                requires_approval=config.get("requires_approval", False),
                include_hashtags=config.get("include_hashtags", True),
                include_cta=config.get("include_cta", True),
                max_length=config.get("max_length", 1500),
                source_urls=config.get("source_urls", []),
                custom_instructions=config.get("custom_instructions"),
                scheduled_at=config.get("scheduled_at"),
            )
        )

        duration_ms = int((datetime.now(timezone.utc) - start_ts).total_seconds() * 1000)
        errors      = final_state.get("errors", [])
        post_id     = final_state.get("post_id")
        status      = AgentRunStatus.FAILED if (errors or not post_id) else AgentRunStatus.COMPLETED

        _run(_update_agent_run(
            agent_run_id,
            status=status,
            output_payload={
                "post_id":        post_id,
                "publish_status": final_state.get("publish_status"),
                "quality_score":  final_state.get("quality_score"),
                "tokens_used":    final_state.get("total_prompt_tokens", 0)
                                  + final_state.get("total_completion_tokens", 0),
                "errors":         errors[-2:] if errors else [],
            },
            duration_ms=duration_ms,
            error_message=errors[-1].get("message") if errors else None,
        ))

        task_logger.info(
            "LinkedIn post generation complete",
            post_id=post_id,
            quality_score=final_state.get("quality_score"),
            publish_status=final_state.get("publish_status"),
            duration_ms=duration_ms,
        )
        return {"status": status, "post_id": post_id}

    except SoftTimeLimitExceeded:
        task_logger.error("Post generation timed out", user_id=user_id)
        _run(_update_agent_run(agent_run_id, status=AgentRunStatus.TIMED_OUT,
                               error_message="Generation timed out after 5 minutes."))
        return {"status": "timed_out"}

    except Exception as exc:
        task_logger.error("Post generation failed", error=str(exc)[:400], user_id=user_id)
        is_rate_limit = "rate" in str(exc).lower() or "429" in str(exc)
        countdown = 60 if is_rate_limit else 30 * (self.request.retries + 1)
        try:
            raise self.retry(exc=exc, countdown=countdown)
        except MaxRetriesExceededError:
            _run(_update_agent_run(
                agent_run_id, status=AgentRunStatus.FAILED,
                error_message=str(exc)[:500], error_type=type(exc).__name__,
                error_traceback=traceback.format_exc(),
            ))
            return {"status": "failed", "error": str(exc)[:200]}


# ---------------------------------------------------------------------------
# generate_daily_posts_task (user-triggered via API)
# ---------------------------------------------------------------------------

@celery_app.task(
    base=InstrumentedTask,
    name="app.workers.linkedin_tasks.generate_daily_posts_task",
    bind=True,
    max_retries=2,
    default_retry_delay=120,
    soft_time_limit=1800,
    time_limit=2100,
    acks_late=True,
)
def generate_daily_posts_task(
    self: Task,
    *,
    user_id: str,
    agent_run_id: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    """
    Generate a content calendar of LinkedIn posts for the next N days.

    config keys:
        days_ahead    : int           — how many days forward to plan (1–30)
        categories    : list[str]     — categories to include (cycles if fewer than days_ahead)
        schedule_time : str | None    — "HH:MM" publish time in user's timezone

    Fan-out strategy: generates each day's post as an independent
    generate_single_post_task (not inline) — this allows:
    - Independent retry per day (day 3 failing doesn't cancel day 4)
    - Celery result tracking per post for UI progress display
    - Staggered timing to avoid hitting Groq rate limits

    Returns immediately with task IDs for each day's generation task.
    """
    task_logger = get_task_logger("generate_daily_posts_task", task_id=self.request.id)
    days_ahead = min(config.get("days_ahead", 7), 30)
    task_logger.info("Daily post calendar generation started", user_id=user_id, days=days_ahead)

    async def _plan_calendar() -> list[dict[str, Any]]:
        """Plan the content calendar using category rotation."""
        from app.services.linkedin_service import LinkedInService
        svc = LinkedInService()
        return await svc.plan_content_calendar(
            user_id=user_id,
            days_ahead=days_ahead,
            requested_categories=config.get("categories", []),
            schedule_time=config.get("schedule_time"),
        )

    try:
        _run(_update_agent_run(agent_run_id, status=AgentRunStatus.RUNNING))

        calendar = _run(_plan_calendar())
        task_logger.info(f"Content calendar planned: {len(calendar)} days")

        dispatched_task_ids: list[str] = []
        for i, day_plan in enumerate(calendar):
            # Stagger each day's generation by 30–90s to avoid rate-limit cascades
            countdown = i * 60 + (i * 30)

            # Create per-post AgentRun placeholder
            per_post_run_id = _run(_create_agent_run(
                user_id=user_id,
                trigger="daily_calendar",
                input_payload={"day": i + 1, "topic": day_plan.get("topic", ""), "parent_run_id": agent_run_id},
            ))

            task_result = generate_single_post_task.apply_async(
                kwargs={
                    "user_id":       user_id,
                    "agent_run_id":  per_post_run_id,
                    "config": {
                        "topic":             day_plan.get("topic", "AI trends"),
                        "category":          day_plan.get("category", "ai_insights"),
                        "tone":              day_plan.get("tone", "thought-leader"),
                        "requires_approval": day_plan.get("requires_approval", False),
                        "scheduled_at":      day_plan.get("scheduled_at"),
                        "include_hashtags":  True,
                        "include_cta":       True,
                        "max_length":        1500,
                    },
                },
                countdown=countdown,
            )
            dispatched_task_ids.append(task_result.id)

        _run(_update_agent_run(
            agent_run_id,
            status=AgentRunStatus.COMPLETED,
            output_payload={
                "days_planned":        len(calendar),
                "tasks_dispatched":    len(dispatched_task_ids),
                "task_ids":            dispatched_task_ids[:20],
            },
        ))

        task_logger.info(
            "Daily post calendar dispatched",
            days_planned=len(calendar),
            tasks_dispatched=len(dispatched_task_ids),
        )
        return {
            "status":         "dispatched",
            "days_planned":   len(calendar),
            "task_ids":       dispatched_task_ids,
        }

    except SoftTimeLimitExceeded:
        _run(_update_agent_run(agent_run_id, status=AgentRunStatus.TIMED_OUT))
        return {"status": "timed_out"}

    except Exception as exc:
        task_logger.error("Daily post calendar failed", error=str(exc)[:400])
        try:
            raise self.retry(exc=exc, countdown=120)
        except MaxRetriesExceededError:
            _run(_update_agent_run(
                agent_run_id, status=AgentRunStatus.FAILED,
                error_message=str(exc)[:500], error_type=type(exc).__name__,
            ))
            return {"status": "failed"}


# ---------------------------------------------------------------------------
# generate_daily_posts_for_all_users_task (Beat-scheduled 02:00 UTC)
# ---------------------------------------------------------------------------

@celery_app.task(
    name="app.workers.linkedin_tasks.generate_daily_posts_for_all_users_task",
    soft_time_limit=300,
    time_limit=360,
)
def generate_daily_posts_for_all_users_task() -> dict[str, Any]:
    """
    Beat-scheduled at 02:00 UTC daily.

    Finds all active users with linkedin_posting_enabled=True,
    checks how many posts they've published today, and enqueues
    generate_daily_posts_task for those who still need content.

    Staggered dispatch: each user's task fires 60s apart to spread
    the Groq API load across an hour rather than hitting it all at once.
    """
    task_logger = get_task_logger("generate_daily_posts_for_all_users")
    task_logger.info("Daily LinkedIn post fan-out started")

    async def _get_eligible_users() -> list[dict[str, Any]]:
        from app.db.session import get_db_context
        from app.db.models.user import User
        from sqlalchemy import select

        async with get_db_context() as db:
            result = await db.execute(
                select(User).where(
                    User.is_active.is_(True),
                    User.is_deleted.is_(False),
                    User.job_search_preferences["linkedin_posting_enabled"].astext == "true",
                    User.linkedin_access_token.isnot(None),
                )
            )
            users = result.scalars().all()
            return [
                {
                    "user_id": str(u.id),
                    "prefs":   u.job_search_preferences or {},
                    "plan":    u.plan,
                }
                for u in users
            ]

    async def _count_todays_posts(user_id: str) -> int:
        from app.db.session import get_db_context
        from app.db.models.linkedin_post import LinkedInPost
        from sqlalchemy import select, func
        from datetime import date

        today_start = datetime.combine(date.today(), datetime.min.time()).replace(tzinfo=timezone.utc)
        async with get_db_context() as db:
            result = await db.execute(
                select(func.count()).where(
                    LinkedInPost.user_id == __import__("uuid").UUID(user_id),
                    LinkedInPost.created_at >= today_start,
                    LinkedInPost.is_deleted.is_(False),
                )
            )
            return result.scalar_one()

    try:
        eligible_users = _run(_get_eligible_users())
        task_logger.info(f"Found {len(eligible_users)} eligible users")

        dispatched = 0
        for i, user_cfg in enumerate(eligible_users):
            user_id = user_cfg["user_id"]
            prefs   = user_cfg["prefs"]

            # Skip if already has posts today (idempotency)
            posts_today = _run(_count_todays_posts(user_id))
            if posts_today > 0:
                task_logger.debug(f"User {user_id} already has {posts_today} posts today — skipping")
                continue

            agent_run_id = _run(_create_agent_run(
                user_id=user_id,
                trigger="beat_scheduler",
                input_payload={"source": "daily_fan_out"},
            ))

            generate_daily_posts_task.apply_async(
                kwargs={
                    "user_id":      user_id,
                    "agent_run_id": agent_run_id,
                    "config": {
                        "days_ahead":    1,
                        "categories":    prefs.get("linkedin_categories", []),
                        "schedule_time": prefs.get("linkedin_post_time", "09:00"),
                    },
                },
                countdown=i * 60,  # stagger 60s per user
            )
            dispatched += 1

        task_logger.info(f"Daily fan-out complete: {dispatched} tasks dispatched")
        return {"eligible": len(eligible_users), "dispatched": dispatched}

    except Exception as exc:
        task_logger.error("Daily fan-out failed", error=str(exc)[:300])
        return {"status": "failed"}


# ---------------------------------------------------------------------------
# publish_scheduled_posts_task (Beat every 10 minutes)
# ---------------------------------------------------------------------------

@celery_app.task(
    name="app.workers.linkedin_tasks.publish_scheduled_posts_task",
    soft_time_limit=300,
    time_limit=360,
)
def publish_scheduled_posts_task() -> dict[str, Any]:
    """
    Beat-scheduled every 10 minutes.

    Finds all LinkedInPost rows where:
        status = 'scheduled'
        scheduled_at <= NOW()
        is_deleted = False
        publish_attempts < 3

    Dispatches publish_single_post_task for each.
    Uses SELECT FOR UPDATE SKIP LOCKED so parallel beat workers don't
    double-dispatch the same post.
    """
    task_logger = get_task_logger("publish_scheduled_posts")

    async def _find_and_dispatch() -> int:
        from app.db.session import get_db_context
        from app.db.models.linkedin_post import LinkedInPost
        from sqlalchemy import select

        now = datetime.now(timezone.utc)

        async with get_db_context() as db:
            result = await db.execute(
                select(LinkedInPost)
                .where(
                    LinkedInPost.status == "scheduled",
                    LinkedInPost.scheduled_at <= now,
                    LinkedInPost.is_deleted.is_(False),
                    LinkedInPost.publish_attempts < 3,
                )
                .limit(20)
                .with_for_update(skip_locked=True)
            )
            posts = result.scalars().all()

            dispatched = 0
            for post in posts:
                # Optimistically mark as 'publishing' to prevent double-dispatch
                post.status = "publishing"
                post.publish_attempts += 1
                await db.flush()

                publish_single_post_task.apply_async(
                    kwargs={"post_id": str(post.id), "user_id": str(post.user_id)},
                    priority=8,
                )
                dispatched += 1

            await db.commit()
            return dispatched

    try:
        dispatched = _run(_find_and_dispatch())
        if dispatched:
            task_logger.info(f"Dispatched {dispatched} publish tasks")
        return {"dispatched": dispatched}
    except Exception as exc:
        task_logger.error("Scheduled publish check failed", error=str(exc)[:300])
        return {"status": "failed"}


# ---------------------------------------------------------------------------
# publish_single_post_task
# ---------------------------------------------------------------------------

@celery_app.task(
    base=InstrumentedTask,
    name="app.workers.linkedin_tasks.publish_single_post_task",
    bind=True,
    max_retries=3,
    default_retry_delay=120,
    soft_time_limit=120,
    time_limit=180,
    acks_late=True,
)
def publish_single_post_task(
    self: Task,
    *,
    post_id: str,
    user_id: str,
) -> dict[str, Any]:
    """
    Publish one LinkedIn post to the LinkedIn Share API.

    Handles:
    - Loading the post and user's OAuth token
    - Token refresh if expired (via linkedin_service)
    - Calling the LinkedIn Share API (v2/ugcPosts)
    - Updating LinkedInPost.linkedin_post_id, .published_at, .status
    - Retrying on LinkedIn 5xx errors (not on 400/401/403 — those are permanent)
    """
    task_logger = get_task_logger("publish_single_post_task", task_id=self.request.id)
    task_logger.info("Publishing LinkedIn post", post_id=post_id)

    async def _publish() -> dict:
        from app.db.session import get_db_context
        from app.db.models.linkedin_post import LinkedInPost
        from app.db.models.user import User
        from app.services.linkedin_service import LinkedInService
        from sqlalchemy import select
        import uuid

        async with get_db_context() as db:
            post_result = await db.execute(
                select(LinkedInPost).where(LinkedInPost.id == uuid.UUID(post_id))
            )
            post = post_result.scalar_one_or_none()
            if not post:
                return {"status": "failed", "reason": "post_not_found"}
            if post.status == "published":
                return {"status": "skipped", "reason": "already_published"}

            user_result = await db.execute(select(User).where(User.id == post.user_id))
            user = user_result.scalar_one_or_none()
            if not user or not user.linkedin_access_token:
                post.status       = "failed"
                post.publish_error = "LinkedIn not connected — no access token."
                post.failed_at    = datetime.now(timezone.utc)
                await db.commit()
                return {"status": "failed", "reason": "no_linkedin_token"}

            svc = LinkedInService()
            try:
                linkedin_post_id, post_url = await svc.publish_post(user=user, post=post)

                post.status           = "published"
                post.published_at     = datetime.now(timezone.utc)
                post.linkedin_post_id = linkedin_post_id
                post.linkedin_post_url = post_url
                post.publish_error    = None
                await db.commit()

                return {
                    "status":          "published",
                    "linkedin_post_id": linkedin_post_id,
                    "post_url":        post_url,
                }

            except Exception as exc:
                error_msg = str(exc)[:500]
                post.status       = "failed"
                post.publish_error = error_msg
                post.failed_at    = datetime.now(timezone.utc)
                await db.commit()
                raise

    try:
        result = _run(_publish())
        task_logger.info("Post publish complete", post_id=post_id, status=result.get("status"))
        return result

    except Exception as exc:
        error_str = str(exc).lower()
        is_permanent = any(kw in error_str for kw in ["401", "403", "not found", "invalid token", "revoked"])
        task_logger.error("Post publish failed", post_id=post_id, error=str(exc)[:300], permanent=is_permanent)

        if not is_permanent:
            try:
                raise self.retry(exc=exc, countdown=120 * (self.request.retries + 1))
            except MaxRetriesExceededError:
                pass

        return {"status": "failed", "error": str(exc)[:200]}


# ---------------------------------------------------------------------------
# sync_post_analytics_task (Beat every 6h)
# ---------------------------------------------------------------------------

@celery_app.task(
    name="app.workers.linkedin_tasks.sync_post_analytics_task",
    soft_time_limit=600,
    time_limit=660,
)
def sync_post_analytics_task() -> dict[str, Any]:
    """
    Beat-scheduled every 6 hours.

    Fetches engagement metrics (impressions, likes, comments, shares, clicks)
    from the LinkedIn Analytics API for all published posts that:
    - Were published within the last 90 days (LinkedIn retains metrics for 90d)
    - Have not been synced within the last 6 hours

    Batches API calls per user to respect LinkedIn's per-token rate limits.
    Updates LinkedInPost rows with fresh metrics and computed engagement_rate.
    """
    task_logger = get_task_logger("sync_post_analytics")
    task_logger.info("LinkedIn analytics sync started")

    async def _sync_all() -> dict:
        from app.db.session import get_db_context
        from app.db.models.linkedin_post import LinkedInPost
        from app.db.models.user import User
        from app.services.linkedin_service import LinkedInService
        from sqlalchemy import select
        from datetime import timedelta
        import uuid

        ninety_days_ago = datetime.now(timezone.utc) - timedelta(days=90)
        six_hours_ago   = datetime.now(timezone.utc) - timedelta(hours=6)

        svc = LinkedInService()
        synced = 0
        failed = 0

        async with get_db_context() as db:
            # Get all posts needing sync, grouped implicitly by user
            posts_result = await db.execute(
                select(LinkedInPost)
                .where(
                    LinkedInPost.status == "published",
                    LinkedInPost.published_at >= ninety_days_ago,
                    LinkedInPost.linkedin_post_id.isnot(None),
                    LinkedInPost.is_deleted.is_(False),
                )
                .order_by(LinkedInPost.metrics_last_synced_at.asc().nullsfirst())
                .limit(200)
            )
            posts = posts_result.scalars().all()

            # Group by user for efficient token reuse
            user_posts: dict[str, list[Any]] = {}
            for post in posts:
                uid = str(post.user_id)
                user_posts.setdefault(uid, []).append(post)

            for uid, user_post_list in user_posts.items():
                user_result = await db.execute(
                    select(User).where(User.id == uuid.UUID(uid))
                )
                user = user_result.scalar_one_or_none()
                if not user or not user.linkedin_access_token:
                    continue

                for post in user_post_list:
                    try:
                        metrics = await svc.fetch_post_analytics(
                            user=user,
                            linkedin_post_id=post.linkedin_post_id,
                        )
                        post.impressions            = metrics.get("impressions", post.impressions)
                        post.likes                  = metrics.get("likes", post.likes)
                        post.comments               = metrics.get("comments", post.comments)
                        post.shares                 = metrics.get("shares", post.shares)
                        post.clicks                 = metrics.get("clicks", post.clicks)
                        post.profile_views_gained   = metrics.get("profile_views", post.profile_views_gained)
                        post.metrics_last_synced_at = datetime.now(timezone.utc)

                        if post.impressions and post.impressions > 0:
                            total_eng       = post.likes + post.comments + post.shares + post.clicks
                            post.engagement_rate = round(total_eng / post.impressions, 4)

                        synced += 1
                        await asyncio.sleep(0.5)  # small delay between API calls per user

                    except Exception as exc:
                        task_logger.warning(
                            "Analytics sync failed for one post",
                            post_id=str(post.id),
                            error=str(exc)[:200],
                        )
                        failed += 1

            await db.commit()

        return {"synced": synced, "failed": failed}

    try:
        result = _run(_sync_all())
        task_logger.info("Analytics sync complete", **result)
        return result
    except Exception as exc:
        task_logger.error("Analytics sync failed", error=str(exc)[:300])
        return {"status": "failed"}


# ---------------------------------------------------------------------------
# sync_single_post_analytics_task (on-demand)
# ---------------------------------------------------------------------------

@celery_app.task(
    name="app.workers.linkedin_tasks.sync_single_post_analytics_task",
    bind=True,
    max_retries=2,
    default_retry_delay=60,
    soft_time_limit=60,
    time_limit=90,
)
def sync_single_post_analytics_task(
    self: Task,
    *,
    post_id: str,
    user_id: str,
) -> dict[str, Any]:
    """
    Sync analytics for a single LinkedIn post on demand.

    Called from GET /linkedin/posts/{id}/analytics when the metrics are stale.
    Returns the updated metrics dict.
    """
    task_logger = get_task_logger("sync_single_post_analytics", task_id=self.request.id)

    async def _sync() -> dict:
        from app.db.session import get_db_context
        from app.db.models.linkedin_post import LinkedInPost
        from app.db.models.user import User
        from app.services.linkedin_service import LinkedInService
        from sqlalchemy import select
        import uuid

        async with get_db_context() as db:
            post_result = await db.execute(
                select(LinkedInPost).where(LinkedInPost.id == uuid.UUID(post_id))
            )
            post = post_result.scalar_one_or_none()
            if not post or not post.linkedin_post_id:
                return {"status": "skipped", "reason": "post_not_found_or_not_published"}

            user_result = await db.execute(select(User).where(User.id == uuid.UUID(user_id)))
            user = user_result.scalar_one_or_none()
            if not user or not user.linkedin_access_token:
                return {"status": "skipped", "reason": "no_linkedin_token"}

            svc     = LinkedInService()
            metrics = await svc.fetch_post_analytics(user=user, linkedin_post_id=post.linkedin_post_id)

            post.impressions          = metrics.get("impressions", post.impressions)
            post.likes                = metrics.get("likes", post.likes)
            post.comments             = metrics.get("comments", post.comments)
            post.shares               = metrics.get("shares", post.shares)
            post.clicks               = metrics.get("clicks", post.clicks)
            post.metrics_last_synced_at = datetime.now(timezone.utc)

            if post.impressions and post.impressions > 0:
                total_eng        = post.likes + post.comments + post.shares + post.clicks
                post.engagement_rate = round(total_eng / post.impressions, 4)

            await db.commit()

        return {
            "status":         "synced",
            "impressions":    post.impressions,
            "likes":          post.likes,
            "comments":       post.comments,
            "shares":         post.shares,
            "engagement_rate": post.engagement_rate,
        }

    try:
        result = _run(_sync())
        task_logger.info("Single post analytics synced", post_id=post_id)
        return result
    except Exception as exc:
        task_logger.error("Single post analytics sync failed", error=str(exc)[:300])
        try:
            raise self.retry(exc=exc, countdown=60)
        except MaxRetriesExceededError:
            return {"status": "failed"}

# ---------------------------------------------------------------------------
# sync_engagement_task
# ---------------------------------------------------------------------------

@celery_app.task(
    name="app.workers.linkedin_tasks.sync_engagement_task",
    bind=True,
    max_retries=2,
    default_retry_delay=60,
    soft_time_limit=60,
    time_limit=90,
)
def sync_engagement_task(
    self: Task,
    *,
    post_id: str,
    user_id: str,
    linkedin_post_id: str | None = None,
) -> dict[str, Any]:
    """
    Scheduled (T+24h) engagement sync for a just-published post, queued
    by linkedin_agent right after publish. `linkedin_post_id` is accepted
    for context/logging but not required — sync_single_post_analytics_task
    re-reads it from the DB row, which is the source of truth.
    """
    task_logger = get_task_logger("sync_engagement_task", task_id=self.request.id)
    task_logger.info("Scheduled engagement sync", post_id=post_id, linkedin_post_id=linkedin_post_id)
    return sync_single_post_analytics_task(post_id=post_id, user_id=user_id)
