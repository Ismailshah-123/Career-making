"""
app/workers/celery_app.py
==========================
Celery application factory and configuration for the JobHunter AI platform.

Celery handles all asynchronous, long-running, and scheduled work:
- Job discovery scraping (runs every 6 hours per user preference)
- Resume parsing and embedding (triggered on upload)
- Application form submission via Playwright (triggered by user action)
- LinkedIn post generation and publishing (daily schedule)
- Follow-up message sending (scheduled at application time + 7 days)
- Notification dispatch (email / webhook / in-app)
- Usage counter resets (1st of each month)
- Analytics aggregation (nightly)

Queue architecture:
- default          : General tasks, low priority
- ai_processing    : LLM calls (resume parsing, cover letters, posts)
                     Rate-limited to 1 worker to prevent Groq quota exhaustion
- scraping         : Web scraping tasks — CPU-light, I/O-heavy
- applications     : Playwright browser automation — resource-heavy (1 worker)
- notifications    : Email / webhook sends — fast, high priority
- linkedin         : LinkedIn API calls — rate-limited

Worker startup (from project root):
    # All queues on one machine (dev):
    celery -A app.workers.celery_app worker --loglevel=info -Q default,ai_processing,scraping,applications,notifications,linkedin

    # Production — separate workers per queue:
    celery -A app.workers.celery_app worker -Q applications -c 1 --loglevel=info
    celery -A app.workers.celery_app worker -Q ai_processing -c 2 --loglevel=info
    celery -A app.workers.celery_app worker -Q scraping -c 5 --loglevel=info
    celery -A app.workers.celery_app worker -Q notifications -c 4 --loglevel=info

Beat (scheduled tasks):
    celery -A app.workers.celery_app beat --loglevel=info

Flower monitoring:
    celery -A app.workers.celery_app flower --port=5555
"""

from __future__ import annotations

import logging
from typing import Any

from celery import Celery, Task
from celery.signals import (
    task_failure,
    task_postrun,
    task_prerun,
    task_retry,
    task_success,
    worker_ready,
    worker_shutdown,
)
from kombu import Exchange, Queue

from app.core.config import settings
from app.core.logging import get_logger, set_correlation_id
from app.core.constants import (
    QUEUE_DEFAULT,
    QUEUE_AI,
    QUEUE_SCRAPING,
    QUEUE_APPLICATIONS,
    QUEUE_NOTIFICATIONS,
    QUEUE_LINKEDIN,
)

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Celery application factory
# ---------------------------------------------------------------------------

def create_celery_app() -> Celery:
    """
    Create and configure the Celery application.

    Called once at module level — the resulting `celery_app` object is
    imported by all task modules and the Celery CLI entrypoint.
    """
    app = Celery(
        "jobhunter",
        broker=settings.CELERY_BROKER_URL,
        backend=settings.CELERY_RESULT_BACKEND,
        include=[
            "app.workers.job_tasks",
            "app.workers.resume_tasks",
            "app.workers.linkedin_tasks",
            "app.workers.notification_tasks",
        ],
    )

    # Apply configuration from settings
    app.config_from_object(_build_celery_config())

    # Register queues with explicit routing keys
    app.conf.task_queues = _build_queues()
    app.conf.task_default_queue = QUEUE_DEFAULT
    app.conf.task_routes = _build_task_routes()

    # Beat schedule for periodic tasks
    app.conf.beat_schedule = _build_beat_schedule()
    app.conf.beat_scheduler = "celery.beat:PersistentScheduler"

    return app


def _build_celery_config() -> dict[str, Any]:
    """Build the full Celery configuration dict from settings."""
    return {
        "broker_url": settings.CELERY_BROKER_URL,
        "result_backend": settings.CELERY_RESULT_BACKEND,
        "task_serializer": "json",
        "result_serializer": "json",
        "accept_content": ["json"],
        "timezone": "UTC",
        "enable_utc": True,
        "task_track_started": True,
        "task_send_sent_event": True,
        "task_soft_time_limit": settings.CELERY_TASK_SOFT_TIME_LIMIT,
        "task_time_limit": settings.CELERY_TASK_TIME_LIMIT,
        "worker_max_tasks_per_child": settings.CELERY_WORKER_MAX_TASKS_PER_CHILD,
        "worker_prefetch_multiplier": 1,    # Disable prefetch — tasks are long-running
        "result_expires": 60 * 60 * 24,     # Keep results for 24 hours
        "task_acks_late": True,             # Ack AFTER task completes (safe retry on worker crash)
        "task_reject_on_worker_lost": True,  # Re-queue if worker dies mid-task
        "broker_connection_retry_on_startup": True,
        "broker_connection_max_retries": 10,
        "task_compression": "gzip",         # Compress large payloads (resume text, etc.)
        "result_compression": "gzip",
        # Redis-specific: set visibility timeout > longest task time limit
        "broker_transport_options": {
            "visibility_timeout": 43200,    # 12 hours
            "fanout_prefix": True,
            "fanout_patterns": True,
        },
    }


def _build_queues() -> list[Queue]:
    """Define all named queues with their exchanges and routing keys."""
    direct_exchange = Exchange("direct", type="direct")
    return [
        Queue(
            QUEUE_DEFAULT,
            direct_exchange,
            routing_key=QUEUE_DEFAULT,
            queue_arguments={"x-max-priority": 10},
        ),
        Queue(
            QUEUE_AI,
            direct_exchange,
            routing_key=QUEUE_AI,
            queue_arguments={"x-max-priority": 5},
        ),
        Queue(
            QUEUE_SCRAPING,
            direct_exchange,
            routing_key=QUEUE_SCRAPING,
            queue_arguments={"x-max-priority": 3},
        ),
        Queue(
            QUEUE_APPLICATIONS,
            direct_exchange,
            routing_key=QUEUE_APPLICATIONS,
            queue_arguments={"x-max-priority": 8},
        ),
        Queue(
            QUEUE_NOTIFICATIONS,
            direct_exchange,
            routing_key=QUEUE_NOTIFICATIONS,
            queue_arguments={"x-max-priority": 10},
        ),
        Queue(
            QUEUE_LINKEDIN,
            direct_exchange,
            routing_key=QUEUE_LINKEDIN,
            queue_arguments={"x-max-priority": 5},
        ),
    ]


def _build_task_routes() -> dict[str, dict[str, str]]:
    """Map task names to their target queues."""
    return {
        # Job tasks
        "app.workers.job_tasks.discover_jobs_task":      {"queue": QUEUE_SCRAPING},
        "app.workers.job_tasks.match_jobs_task":         {"queue": QUEUE_AI},
        "app.workers.job_tasks.submit_application_task": {"queue": QUEUE_APPLICATIONS},
        "app.workers.job_tasks.send_followup_task":      {"queue": QUEUE_NOTIFICATIONS},
        "app.workers.job_tasks.send_outreach_task":      {"queue": QUEUE_LINKEDIN},
        "app.workers.job_tasks.discover_recruiters_task": {"queue": QUEUE_SCRAPING},
        "app.workers.job_tasks.embed_job_task":          {"queue": QUEUE_AI},

        # Resume tasks
        "app.workers.resume_tasks.process_resume_task":       {"queue": QUEUE_AI},
        "app.workers.resume_tasks.embed_resume_task":         {"queue": QUEUE_AI},
        "app.workers.resume_tasks.tailor_resume_task":        {"queue": QUEUE_AI},
        "app.workers.resume_tasks.generate_cover_letter_task": {"queue": QUEUE_AI},

        # LinkedIn tasks
        "app.workers.linkedin_tasks.generate_daily_posts_task": {"queue": QUEUE_LINKEDIN},
        "app.workers.linkedin_tasks.publish_scheduled_posts_task": {"queue": QUEUE_LINKEDIN},
        "app.workers.linkedin_tasks.sync_post_analytics_task": {"queue": QUEUE_LINKEDIN},
        "app.workers.linkedin_tasks.generate_single_post_task": {"queue": QUEUE_LINKEDIN},

        # Notification tasks
        "app.workers.notification_tasks.send_notification_task": {"queue": QUEUE_NOTIFICATIONS},
        "app.workers.notification_tasks.send_weekly_summary_task": {"queue": QUEUE_NOTIFICATIONS},
        "app.workers.notification_tasks.reset_monthly_counters_task": {"queue": QUEUE_DEFAULT},
        "app.workers.notification_tasks.write_audit_log_task": {"queue": QUEUE_DEFAULT},
    }


def _build_beat_schedule() -> dict[str, dict[str, Any]]:
    """
    Define all periodic tasks managed by Celery Beat.

    Schedules are defined in crontab / interval format.
    All times are UTC.
    """
    from celery.schedules import crontab

    return {
        # ── Discovery ────────────────────────────────────────────────────
        "discover-jobs-all-users": {
            "task": "app.workers.job_tasks.discover_jobs_for_all_users_task",
            "schedule": crontab(hour="*/6", minute="0"),  # Every 6 hours
            "options": {"queue": QUEUE_SCRAPING, "priority": 3},
        },

        # ── LinkedIn ─────────────────────────────────────────────────────
        "publish-scheduled-linkedin-posts": {
            "task": "app.workers.linkedin_tasks.publish_scheduled_posts_task",
            "schedule": crontab(minute="*/10"),             # Every 10 minutes
            "options": {"queue": QUEUE_LINKEDIN, "priority": 5},
        },
        "sync-linkedin-analytics": {
            "task": "app.workers.linkedin_tasks.sync_post_analytics_task",
            "schedule": crontab(hour="*/6", minute="30"),   # Every 6 hours, offset 30m
            "options": {"queue": QUEUE_LINKEDIN, "priority": 2},
        },
        "generate-daily-linkedin-posts": {
            "task": "app.workers.linkedin_tasks.generate_daily_posts_for_all_users_task",
            "schedule": crontab(hour="2", minute="0"),      # 02:00 UTC daily
            "options": {"queue": QUEUE_LINKEDIN, "priority": 4},
        },

        # ── Follow-ups ───────────────────────────────────────────────────
        "check-due-followups": {
            "task": "app.workers.job_tasks.check_due_followups_task",
            "schedule": crontab(hour="*/2", minute="15"),   # Every 2 hours, offset 15m
            "options": {"queue": QUEUE_NOTIFICATIONS, "priority": 7},
        },

        # ── Notifications ────────────────────────────────────────────────
        "send-weekly-summary": {
            "task": "app.workers.notification_tasks.send_weekly_summary_task",
            "schedule": crontab(day_of_week="1", hour="8", minute="0"),  # Mondays 08:00 UTC
            "options": {"queue": QUEUE_NOTIFICATIONS, "priority": 5},
        },

        # ── Maintenance ──────────────────────────────────────────────────
        "reset-monthly-usage-counters": {
            "task": "app.workers.notification_tasks.reset_monthly_counters_task",
            "schedule": crontab(day_of_month="1", hour="0", minute="5"),  # 1st of month 00:05 UTC
            "options": {"queue": QUEUE_DEFAULT, "priority": 2},
        },
        "cleanup-expired-agent-runs": {
            "task": "app.workers.notification_tasks.cleanup_old_agent_runs_task",
            "schedule": crontab(hour="3", minute="0"),      # 03:00 UTC daily
            "options": {"queue": QUEUE_DEFAULT, "priority": 1},
        },
    }


# ---------------------------------------------------------------------------
# Base Task class — shared instrumentation for all tasks
# ---------------------------------------------------------------------------

class InstrumentedTask(Task):
    """
    Custom Celery Task base class that adds:
    1. Structured logging on start/success/failure with task metadata
    2. AgentRun DB record lifecycle management (PENDING → RUNNING → DONE)
    3. Correlation ID injection into the async context
    4. Sentry breadcrumb creation per task
    5. Automatic retry for transient failures (configurable per task)

    Usage in a task module:
        @celery_app.task(
            base=InstrumentedTask,
            name="app.workers.job_tasks.discover_jobs_task",
            bind=True,
            max_retries=3,
            default_retry_delay=30,
        )
        def discover_jobs_task(self, *, user_id: str, agent_run_id: str, ...):
            ...
    """

    abstract = True

    # Subclass-overridable: which exception types trigger automatic retry
    retryable_exceptions: tuple[type[Exception], ...] = (
        ConnectionError,
        TimeoutError,
    )

    def on_failure(self, exc: Exception, task_id: str, args: Any, kwargs: Any, einfo: Any) -> None:
        logger.error(
            f"Task FAILED: {self.name}",
            task_id=task_id,
            error_type=type(exc).__name__,
            error=str(exc)[:500],
            exc_info=True,
        )
        super().on_failure(exc, task_id, args, kwargs, einfo)

    def on_retry(self, exc: Exception, task_id: str, args: Any, kwargs: Any, einfo: Any) -> None:
        logger.warning(
            f"Task RETRYING: {self.name}",
            task_id=task_id,
            attempt=self.request.retries + 1,
            max_retries=self.max_retries,
            error=str(exc)[:200],
        )
        super().on_retry(exc, task_id, args, kwargs, einfo)

    def on_success(self, retval: Any, task_id: str, args: Any, kwargs: Any) -> None:
        logger.info(f"Task SUCCESS: {self.name}", task_id=task_id)
        super().on_success(retval, task_id, args, kwargs)


# ---------------------------------------------------------------------------
# Signal handlers
# ---------------------------------------------------------------------------

@task_prerun.connect
def task_prerun_handler(task_id: str, task: Task, **kwargs: Any) -> None:
    """Inject a correlation ID matching the Celery task ID into the async context."""
    set_correlation_id(task_id)
    logger.info(f"Task started: {task.name}", task_id=task_id)


@task_postrun.connect
def task_postrun_handler(task_id: str, task: Task, state: str, **kwargs: Any) -> None:
    logger.debug(f"Task finished: {task.name}", task_id=task_id, state=state)


@task_failure.connect
def task_failure_handler(task_id: str, exception: Exception, **kwargs: Any) -> None:
    logger.error(
        f"Task failure signal",
        task_id=task_id,
        error=str(exception)[:300],
    )


@task_retry.connect
def task_retry_handler(request: Any, reason: Any, **kwargs: Any) -> None:
    logger.warning(
        f"Task retry signal",
        task_id=request.id,
        reason=str(reason)[:200],
    )


@worker_ready.connect
def worker_ready_handler(sender: Any, **kwargs: Any) -> None:
    logger.info(
        "Celery worker ready",
        hostname=sender.hostname,
        queues=list(sender.app.conf.task_queues or []),
    )


@worker_shutdown.connect
def worker_shutdown_handler(sender: Any, **kwargs: Any) -> None:
    logger.info("Celery worker shutting down", hostname=getattr(sender, "hostname", "unknown"))


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

celery_app: Celery = create_celery_app()

# Make the Celery app available as `app` for the CLI:
#   celery -A app.workers.celery_app worker ...
app = celery_app