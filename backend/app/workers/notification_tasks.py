"""
app/workers/notification_tasks.py
====================================
Celery task definitions for notifications, audit logging, and maintenance.

Tasks defined here:
    send_notification_task          — dispatch one notification (email/webhook/in-app)
    send_email_task                 — low-level SMTP email sender
    send_webhook_task               — POST to user-configured webhook URL
    send_weekly_summary_task        — Beat Monday 08:00 UTC: weekly digest emails
    write_audit_log_task            — async audit log write (fire-and-forget)
    reset_monthly_counters_task     — Beat 1st of month: reset usage counters
    cleanup_old_agent_runs_task     — Beat 03:00 UTC daily: prune old runs
    send_push_notification_task     — in-app WebSocket push (future expansion)

Notification routing:
    NotificationChannel.EMAIL       → send_email_task
    NotificationChannel.WEBHOOK     → send_webhook_task
    NotificationChannel.IN_APP      → send_push_notification_task

All email tasks use async SMTP via aiosmtplib for non-blocking I/O.
Webhook tasks use httpx with a 10-second timeout and 3 retries.
Audit log writes are fire-and-forget (max_retries=0, ignore_result=True).

GDPR note: Notification content is logged to the audit trail only as
metadata (event type, timestamp, user_id) — not the full message body —
to avoid storing PII in logs beyond what's necessary.
"""

from __future__ import annotations

import asyncio
import traceback
from datetime import datetime, timedelta, timezone
from typing import Any

from celery import Task
from celery.exceptions import MaxRetriesExceededError, SoftTimeLimitExceeded

from app.workers.celery_app import celery_app, InstrumentedTask
from app.core.constants import NotificationType, NotificationChannel
from app.core.config import settings
from app.core.logging import get_logger, get_task_logger

logger = get_logger(__name__)


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# send_notification_task
# ---------------------------------------------------------------------------

@celery_app.task(
    base=InstrumentedTask,
    name="app.workers.notification_tasks.send_notification_task",
    bind=True,
    max_retries=3,
    default_retry_delay=30,
    soft_time_limit=60,
    time_limit=90,
    acks_late=True,
    priority=9,
)
def send_notification_task(
    self: Task,
    *,
    user_id: str,
    notification_type: str,
    context: dict[str, Any],
    channels: list[str] | None = None,
) -> dict[str, Any]:
    """
    Route and dispatch a notification to one or more channels.

    The notification_type determines the template and subject line.
    Channels default to the user's notification_preferences if not specified.

    Dispatches channel-specific sub-tasks in parallel for efficiency:
        email   → send_email_task
        webhook → send_webhook_task
        in_app  → send_push_notification_task

    context is passed through to template rendering — never log it fully
    as it may contain PII (email addresses, job titles, company names).
    """
    task_logger = get_task_logger("send_notification_task", task_id=self.request.id)
    task_logger.info(
        "Notification dispatch started",
        user_id=user_id,
        notification_type=notification_type,
    )

    async def _dispatch() -> dict:
        from app.db.session import get_db_context
        from app.db.models.user import User
        from sqlalchemy import select
        import uuid

        async with get_db_context() as db:
            result = await db.execute(select(User).where(User.id == uuid.UUID(user_id)))
            user = result.scalar_one_or_none()
            if not user or not user.is_active:
                return {"status": "skipped", "reason": "user_not_found_or_inactive"}

        prefs           = user.notification_preferences or {}
        active_channels = channels or _resolve_channels(notification_type, prefs)
        dispatched: list[str] = []

        for channel in active_channels:
            try:
                if channel == NotificationChannel.EMAIL and settings.EMAIL_ENABLED:
                    subject, body_html, body_text = _render_email_template(
                        notification_type=notification_type,
                        context=context,
                        user_name=user.display_name,
                    )
                    send_email_task.apply_async(
                        kwargs={
                            "to_email":  user.email,
                            "to_name":   user.display_name,
                            "subject":   subject,
                            "body_html": body_html,
                            "body_text": body_text,
                        },
                        priority=8,
                    )
                    dispatched.append("email")

                elif channel == NotificationChannel.WEBHOOK:
                    webhook_url = prefs.get("webhook_url")
                    if webhook_url:
                        send_webhook_task.apply_async(
                            kwargs={
                                "webhook_url":       webhook_url,
                                "notification_type": notification_type,
                                "context":           context,
                                "user_id":           user_id,
                            },
                            priority=6,
                        )
                        dispatched.append("webhook")

                elif channel == NotificationChannel.IN_APP:
                    send_push_notification_task.apply_async(
                        kwargs={
                            "user_id":           user_id,
                            "notification_type": notification_type,
                            "context":           context,
                        },
                        priority=9,
                    )
                    dispatched.append("in_app")

            except Exception as exc:
                task_logger.warning(
                    f"Channel dispatch failed: {channel}",
                    error=str(exc)[:200],
                )

        return {"status": "dispatched", "channels": dispatched}

    try:
        result = _run(_dispatch())
        task_logger.info(
            "Notification dispatched",
            user_id=user_id,
            notification_type=notification_type,
            channels=result.get("channels"),
        )
        return result
    except Exception as exc:
        task_logger.error("Notification dispatch failed", error=str(exc)[:300])
        try:
            raise self.retry(exc=exc, countdown=30 * (self.request.retries + 1))
        except MaxRetriesExceededError:
            return {"status": "failed", "error": str(exc)[:200]}


# ---------------------------------------------------------------------------
# send_email_task
# ---------------------------------------------------------------------------

@celery_app.task(
    name="app.workers.notification_tasks.send_email_task",
    bind=True,
    max_retries=3,
    default_retry_delay=60,
    soft_time_limit=45,
    time_limit=60,
    priority=8,
)
def send_email_task(
    self: Task,
    *,
    to_email: str,
    to_name: str,
    subject: str,
    body_html: str,
    body_text: str,
) -> dict[str, Any]:
    """
    Send a single email via SMTP using aiosmtplib (async, non-blocking).

    Implements:
    - TLS/STARTTLS based on SMTP_USE_TLS setting
    - HTML + plain-text multipart/alternative message
    - Retry on connection errors (not on 550 permanent failures)
    - From name / Reply-To configuration from settings
    """
    task_logger = get_task_logger("send_email_task", task_id=self.request.id)

    if not settings.EMAIL_ENABLED:
        task_logger.debug("Email disabled in settings — skipping", to=to_email)
        return {"status": "skipped", "reason": "email_disabled"}

    async def _send() -> dict:
        import aiosmtplib
        from email.mime.multipart import MIMEMultipart
        from email.mime.text import MIMEText

        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = f"{settings.SMTP_FROM_NAME} <{settings.SMTP_FROM_EMAIL}>"
        msg["To"]      = f"{to_name} <{to_email}>"
        msg["Reply-To"] = str(settings.SMTP_FROM_EMAIL)

        msg.attach(MIMEText(body_text, "plain", "utf-8"))
        msg.attach(MIMEText(body_html, "html", "utf-8"))

        await aiosmtplib.send(
            msg,
            hostname=settings.SMTP_HOST,
            port=settings.SMTP_PORT,
            username=settings.SMTP_USER,
            password=settings.SMTP_PASSWORD,
            use_tls=settings.SMTP_USE_TLS,
            timeout=30,
        )
        return {"status": "sent", "to": to_email}

    try:
        result = _run(_send())
        task_logger.info("Email sent", to=to_email, subject=subject[:60])
        return result
    except Exception as exc:
        error_str = str(exc).lower()
        task_logger.error("Email send failed", to=to_email, error=str(exc)[:300])

        # Permanent SMTP failures — don't retry
        if any(kw in error_str for kw in ["550", "551", "553", "invalid address", "no such user"]):
            return {"status": "failed_permanent", "error": str(exc)[:200]}

        try:
            raise self.retry(exc=exc, countdown=60 * (self.request.retries + 1))
        except MaxRetriesExceededError:
            return {"status": "failed", "error": str(exc)[:200]}


# ---------------------------------------------------------------------------
# send_password_reset_email_task / send_email_verification_task
# ---------------------------------------------------------------------------

@celery_app.task(
    name="app.workers.notification_tasks.send_password_reset_email_task",
    bind=True,
    max_retries=3,
    default_retry_delay=60,
    soft_time_limit=45,
    time_limit=60,
    priority=9,
)
def send_password_reset_email_task(
    self: Task,
    *,
    user_id: str,
    email: str,
    full_name: str,
    reset_token: str,
) -> dict[str, Any]:
    """Send a 'reset your password' email containing a one-time link."""
    task_logger = get_task_logger("send_password_reset_email_task", task_id=self.request.id)
    reset_url = f"{settings.FRONTEND_URL}/reset-password?token={reset_token}"

    body_text = (
        f"Hi {full_name},\n\n"
        f"We received a request to reset your password.\n\n"
        f"Reset it here (valid for 1 hour):\n{reset_url}\n\n"
        f"If you didn't request this, you can safely ignore this email.\n\n"
        f"Best,\nThe CareerGPT Team"
    )
    body_html = (
        f"<p>Hi {full_name},</p>"
        f"<p>We received a request to reset your password.</p>"
        f"<p><a href=\"{reset_url}\">Reset your password</a> (valid for 1 hour).</p>"
        f"<p>If you didn't request this, you can safely ignore this email.</p>"
        f"<p>Best,<br/>The CareerGPT Team</p>"
    )

    try:
        return send_email_task(
            to_email=email,
            to_name=full_name,
            subject="Reset your CareerGPT password",
            body_html=body_html,
            body_text=body_text,
        )
    except Exception as exc:
        task_logger.error("Password reset email failed", user_id=user_id, error=str(exc)[:200])
        return {"status": "failed", "error": str(exc)[:200]}


@celery_app.task(
    name="app.workers.notification_tasks.send_email_verification_task",
    bind=True,
    max_retries=3,
    default_retry_delay=60,
    soft_time_limit=45,
    time_limit=60,
    priority=9,
)
def send_email_verification_task(
    self: Task,
    *,
    user_id: str,
    email: str,
    full_name: str,
    verify_token: str,
) -> dict[str, Any]:
    """Send an 'verify your email' message containing a one-time link."""
    task_logger = get_task_logger("send_email_verification_task", task_id=self.request.id)
    verify_url = f"{settings.FRONTEND_URL}/verify-email?token={verify_token}"

    body_text = (
        f"Hi {full_name},\n\n"
        f"Please confirm your email address to finish setting up your account:\n\n"
        f"{verify_url}\n\n"
        f"Best,\nThe CareerGPT Team"
    )
    body_html = (
        f"<p>Hi {full_name},</p>"
        f"<p>Please confirm your email address to finish setting up your account:</p>"
        f"<p><a href=\"{verify_url}\">Verify your email</a></p>"
        f"<p>Best,<br/>The CareerGPT Team</p>"
    )

    try:
        return send_email_task(
            to_email=email,
            to_name=full_name,
            subject="Verify your CareerGPT email address",
            body_html=body_html,
            body_text=body_text,
        )
    except Exception as exc:
        task_logger.error("Verification email failed", user_id=user_id, error=str(exc)[:200])
        return {"status": "failed", "error": str(exc)[:200]}


# ---------------------------------------------------------------------------
# send_job_alert_task
# ---------------------------------------------------------------------------

@celery_app.task(
    name="app.workers.notification_tasks.send_job_alert_task",
    bind=True,
    max_retries=2,
    default_retry_delay=30,
    soft_time_limit=30,
    time_limit=45,
    priority=7,
)
def send_job_alert_task(
    self: Task,
    *,
    user_id: str,
    job_id: str,
    alert_message: str,
    priority: str = "medium",
) -> dict[str, Any]:
    """
    Real-time in-app alert for a strong new job match, queued by
    discovery_agent right after a high-scoring job is found. Delegates to
    send_notification_task (in_app channel) so it goes through the same
    preference/channel routing as every other notification.
    """
    task_logger = get_task_logger("send_job_alert_task", task_id=self.request.id)
    try:
        return send_notification_task(
            user_id=user_id,
            notification_type=NotificationType.JOB_MATCHED,
            context={"job_id": job_id, "alert_message": alert_message, "priority": priority},
            channels=[NotificationChannel.IN_APP],
        )
    except Exception as exc:
        task_logger.warning("Job alert dispatch failed", user_id=user_id, error=str(exc)[:200])
        return {"status": "failed", "error": str(exc)[:200]}


# ---------------------------------------------------------------------------
# send_webhook_task
# ---------------------------------------------------------------------------

@celery_app.task(
    name="app.workers.notification_tasks.send_webhook_task",
    bind=True,
    max_retries=3,
    default_retry_delay=30,
    soft_time_limit=30,
    time_limit=45,
)
def send_webhook_task(
    self: Task,
    *,
    webhook_url: str,
    notification_type: str,
    context: dict[str, Any],
    user_id: str,
) -> dict[str, Any]:
    """
    POST a JSON webhook payload to the user's configured webhook URL.

    Payload schema:
    {
        "event":      "application.submitted",
        "timestamp":  "2025-01-01T12:00:00Z",
        "user_id":    "...",
        "data":       { ... context ... }
    }

    Validates that webhook_url uses HTTPS. Times out after 10 seconds.
    Retries on 5xx responses but not on 4xx (configuration errors).
    """
    task_logger = get_task_logger("send_webhook_task", task_id=self.request.id)

    if not webhook_url.startswith("https://"):
        task_logger.warning("Webhook URL rejected — must be HTTPS", url=webhook_url[:40])
        return {"status": "failed_permanent", "reason": "https_required"}

    async def _post() -> dict:
        import httpx

        payload = {
            "event":     notification_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "user_id":   user_id,
            "data":      context,
        }

        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                webhook_url,
                json=payload,
                headers={
                    "Content-Type": "application/json",
                    "User-Agent":   f"JobHunterAI-Webhook/1.0",
                    "X-Event-Type": notification_type,
                },
            )
            return {
                "status":      "sent" if response.status_code < 300 else "failed",
                "http_status": response.status_code,
                "url":         webhook_url[:40],
            }

    try:
        result = _run(_post())
        if result.get("status") == "sent":
            task_logger.info("Webhook delivered", url=webhook_url[:40], http_status=result.get("http_status"))
        else:
            task_logger.warning("Webhook delivery failed", http_status=result.get("http_status"))

        if result.get("http_status", 200) >= 500:
            raise Exception(f"Webhook returned HTTP {result['http_status']}")
        return result

    except Exception as exc:
        try:
            raise self.retry(exc=exc, countdown=30 * (self.request.retries + 1))
        except MaxRetriesExceededError:
            return {"status": "failed", "error": str(exc)[:200]}


# ---------------------------------------------------------------------------
# send_push_notification_task (in-app)
# ---------------------------------------------------------------------------

@celery_app.task(
    name="app.workers.notification_tasks.send_push_notification_task",
    soft_time_limit=15,
    time_limit=20,
    priority=9,
)
def send_push_notification_task(
    *,
    user_id: str,
    notification_type: str,
    context: dict[str, Any],
) -> dict[str, Any]:
    """
    Store an in-app notification record for display in the dashboard.

    In production this would also push via WebSocket to the connected
    browser session using Redis Pub/Sub or a dedicated push service.
    For now: write to a notifications table and return.
    """
    async def _write() -> dict:
        from app.db.session import get_db_context
        from sqlalchemy import text
        import uuid
        import json

        async with get_db_context() as db:
            # Write to a simple notifications JSONB store
            # (In a full schema, this would be its own Notification model)
            try:
                await db.execute(
                    text("""
                        INSERT INTO in_app_notifications
                        (id, user_id, notification_type, context, is_read, created_at)
                        VALUES (:id, :user_id, :type, :context, false, NOW())
                        ON CONFLICT DO NOTHING
                    """),
                    {
                        "id":      str(uuid.uuid4()),
                        "user_id": user_id,
                        "type":    notification_type,
                        "context": json.dumps(context),
                    },
                )
                await db.commit()
            except Exception:
                # Table may not exist yet — fail silently for in-app (non-critical)
                pass

        return {"status": "written"}

    try:
        return _run(_write())
    except Exception as exc:
        logger.warning("In-app notification write failed", error=str(exc)[:200])
        return {"status": "failed"}


# ---------------------------------------------------------------------------
# send_weekly_summary_task (Beat Monday 08:00 UTC)
# ---------------------------------------------------------------------------

@celery_app.task(
    name="app.workers.notification_tasks.send_weekly_summary_task",
    soft_time_limit=300,
    time_limit=360,
)
def send_weekly_summary_task() -> dict[str, Any]:
    """
    Beat-scheduled Monday 08:00 UTC.

    For each active user with email_weekly_summary=True:
    1. Compute last-7-day stats (applications, interviews, matches, posts)
    2. Render a summary email with the weekly dashboard snapshot
    3. Dispatch send_email_task for delivery

    Staggered by 5s per user to avoid SMTP rate limits.
    """
    task_logger = get_task_logger("send_weekly_summary")
    task_logger.info("Weekly summary dispatch started")

    async def _get_summary_users() -> list[dict]:
        from app.db.session import get_db_context
        from app.db.models.user import User
        from sqlalchemy import select

        async with get_db_context() as db:
            result = await db.execute(
                select(User).where(
                    User.is_active.is_(True),
                    User.is_verified.is_(True),
                    User.is_deleted.is_(False),
                    User.notification_preferences["email_weekly_summary"].astext == "true",
                )
            )
            return [
                {
                    "user_id":    str(u.id),
                    "email":      u.email,
                    "full_name":  u.display_name,
                }
                for u in result.scalars().all()
            ]

    async def _build_user_summary(user_id: str) -> dict:
        from app.db.session import get_db_context
        from app.db.models.application import Application
        from app.db.models.linkedin_post import LinkedInPost
        from app.core.constants import ApplicationStatus
        from sqlalchemy import select, func
        import uuid

        seven_days_ago = datetime.now(timezone.utc) - timedelta(days=7)
        async with get_db_context() as db:
            uid = uuid.UUID(user_id)

            apps = await db.execute(
                select(func.count()).where(
                    Application.user_id == uid,
                    Application.created_at >= seven_days_ago,
                    Application.is_deleted.is_(False),
                )
            )
            interviews = await db.execute(
                select(func.count()).where(
                    Application.user_id == uid,
                    Application.status == ApplicationStatus.INTERVIEW_SCHEDULED.value,
                    Application.is_deleted.is_(False),
                )
            )
            posts = await db.execute(
                select(func.count()).where(
                    LinkedInPost.user_id == uid,
                    LinkedInPost.created_at >= seven_days_ago,
                    LinkedInPost.status == "published",
                    LinkedInPost.is_deleted.is_(False),
                )
            )

        return {
            "applications_this_week": apps.scalar_one(),
            "active_interviews":      interviews.scalar_one(),
            "linkedin_posts":         posts.scalar_one(),
        }

    try:
        users = _run(_get_summary_users())
        task_logger.info(f"Sending weekly summary to {len(users)} users")

        sent = 0
        for i, user_data in enumerate(users):
            try:
                summary = _run(_build_user_summary(user_data["user_id"]))
                subject, body_html, body_text = _render_email_template(
                    notification_type=NotificationType.WEEKLY_SUMMARY,
                    context=summary,
                    user_name=user_data["full_name"],
                )
                send_email_task.apply_async(
                    kwargs={
                        "to_email":  user_data["email"],
                        "to_name":   user_data["full_name"],
                        "subject":   subject,
                        "body_html": body_html,
                        "body_text": body_text,
                    },
                    countdown=i * 5,
                    priority=5,
                )
                sent += 1
            except Exception as exc:
                task_logger.warning(
                    "Weekly summary failed for one user",
                    user_id=user_data.get("user_id"),
                    error=str(exc)[:200],
                )

        task_logger.info(f"Weekly summary dispatched: {sent}/{len(users)}")
        return {"sent": sent, "total": len(users)}

    except Exception as exc:
        task_logger.error("Weekly summary task failed", error=str(exc)[:300])
        return {"status": "failed"}


# ---------------------------------------------------------------------------
# write_audit_log_task (fire-and-forget)
# ---------------------------------------------------------------------------

@celery_app.task(
    name="app.workers.notification_tasks.write_audit_log_task",
    max_retries=0,
    ignore_result=True,
    soft_time_limit=15,
    time_limit=20,
)
def write_audit_log_task(*, log_data: dict[str, Any]) -> None:
    """
    Fire-and-forget audit log write.

    Accepts the same kwargs as AuditLog.create() and persists the record.
    Used for events triggered inside background tasks where no DB session
    is open in the calling context.
    """
    async def _write() -> None:
        from app.db.session import get_db_context
        from app.db.models.audit_log import AuditLog
        import uuid

        # Convert user_id string → UUID if present
        if isinstance(log_data.get("user_id"), str):
            try:
                log_data["user_id"] = uuid.UUID(log_data["user_id"])
            except ValueError:
                log_data.pop("user_id", None)

        async with get_db_context() as db:
            log = AuditLog.create(**log_data)
            db.add(log)
            await db.commit()

    try:
        _run(_write())
    except Exception as exc:
        logger.warning("Audit log write failed", error=str(exc)[:200])


# ---------------------------------------------------------------------------
# reset_monthly_counters_task (Beat 1st of month 00:05 UTC)
# ---------------------------------------------------------------------------

@celery_app.task(
    name="app.workers.notification_tasks.reset_monthly_counters_task",
    soft_time_limit=120,
    time_limit=180,
)
def reset_monthly_counters_task() -> dict[str, Any]:
    """
    Beat-scheduled 1st of every month at 00:05 UTC.

    Resets per-user monthly usage counters:
        - applications_this_month → 0
        - ai_rewrites_this_month  → 0

    Uses a bulk UPDATE rather than per-user updates for efficiency.
    """
    task_logger = get_task_logger("reset_monthly_counters")

    async def _reset() -> int:
        from app.db.session import get_db_context
        from sqlalchemy import update
        from app.db.models.user import User

        async with get_db_context() as db:
            result = await db.execute(
                update(User)
                .where(User.is_active.is_(True), User.is_deleted.is_(False))
                .values(applications_this_month=0, ai_rewrites_this_month=0)
            )
            await db.commit()
            return result.rowcount

    try:
        count = _run(_reset())
        task_logger.info(f"Monthly counters reset for {count} users")
        return {"reset_count": count}
    except Exception as exc:
        task_logger.error("Monthly counter reset failed", error=str(exc)[:300])
        return {"status": "failed"}


# ---------------------------------------------------------------------------
# cleanup_old_agent_runs_task (Beat 03:00 UTC daily)
# ---------------------------------------------------------------------------

@celery_app.task(
    name="app.workers.notification_tasks.cleanup_old_agent_runs_task",
    soft_time_limit=180,
    time_limit=240,
)
def cleanup_old_agent_runs_task() -> dict[str, Any]:
    """
    Beat-scheduled 03:00 UTC daily.

    Soft-deletes AgentRun rows older than 90 days that are in terminal
    states (COMPLETED, FAILED, CANCELLED, TIMED_OUT).

    Preserves recent runs and all PENDING/RUNNING runs regardless of age
    (running tasks should never be pruned).

    Uses batched DELETE (1000 rows at a time) to avoid long-running
    transactions that would hold table locks.
    """
    task_logger = get_task_logger("cleanup_old_agent_runs")

    async def _cleanup() -> int:
        from app.db.session import get_db_context
        from app.db.models.agent_run import AgentRun
        from sqlalchemy import select, update
        from datetime import timedelta

        cutoff = datetime.now(timezone.utc) - timedelta(days=90)
        terminal_statuses = [
            AgentRunStatus.COMPLETED,
            AgentRunStatus.FAILED,
            AgentRunStatus.CANCELLED,
            AgentRunStatus.TIMED_OUT,
        ]

        total_deleted = 0
        async with get_db_context() as db:
            while True:
                # Fetch a batch of IDs to soft-delete
                result = await db.execute(
                    select(AgentRun.id).where(
                        AgentRun.created_at < cutoff,
                        AgentRun.status.in_(terminal_statuses),
                        AgentRun.is_deleted.is_(False),
                    ).limit(1000)
                )
                ids = [row[0] for row in result.fetchall()]
                if not ids:
                    break

                await db.execute(
                    update(AgentRun)
                    .where(AgentRun.id.in_(ids))
                    .values(is_deleted=True)
                )
                await db.commit()
                total_deleted += len(ids)

                if len(ids) < 1000:
                    break

        return total_deleted

    try:
        count = _run(_cleanup())
        task_logger.info(f"Cleaned up {count} old agent runs")
        return {"deleted": count}
    except Exception as exc:
        task_logger.error("Agent run cleanup failed", error=str(exc)[:300])
        return {"status": "failed"}


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _resolve_channels(notification_type: str, prefs: dict[str, Any]) -> list[str]:
    """
    Determine which channels a notification should be sent to based on
    the user's notification_preferences and the notification type.
    """
    channels: list[str] = []

    channel_map: dict[str, str] = {
        NotificationType.APPLICATION_SUBMITTED:       "email_on_application",
        NotificationType.APPLICATION_STATUS_CHANGED:  "email_on_status_change",
        NotificationType.INTERVIEW_SCHEDULED:         "email_on_interview",
        NotificationType.JOB_MATCHED:                 "email_on_application",
        NotificationType.RESUME_PROCESSED:            "email_on_application",
        NotificationType.LINKEDIN_POST_PUBLISHED:     "in_app_notifications",
        NotificationType.AGENT_FAILED:                "email_on_application",
        NotificationType.WEEKLY_SUMMARY:              "email_weekly_summary",
    }

    pref_key = channel_map.get(notification_type, "in_app_notifications")

    if prefs.get(pref_key, False):
        channels.append(NotificationChannel.EMAIL)
    if prefs.get("in_app_notifications", True):
        channels.append(NotificationChannel.IN_APP)
    if prefs.get("webhook_url"):
        channels.append(NotificationChannel.WEBHOOK)

    return channels


def _render_email_template(
    notification_type: str,
    context: dict[str, Any],
    user_name: str,
) -> tuple[str, str, str]:
    """
    Render an email notification into (subject, html_body, plain_body).

    Production: replace with a proper template engine (Jinja2 + HTML templates).
    Currently returns well-structured plain-text approximations that render
    correctly in all email clients without a templating dependency.
    """
    templates: dict[str, dict[str, str]] = {
        NotificationType.APPLICATION_SUBMITTED: {
            "subject": "✅ Application Submitted — JobHunter AI",
            "body":    (
                "Hi {name},\n\n"
                "Your application has been automatically submitted.\n\n"
                "Job: {job_title} at {company_name}\n"
                "Board: {job_board}\n"
                "Applied at: {applied_at}\n\n"
                "Log in to your dashboard to track the status.\n\n"
                "Best,\nThe JobHunter AI Team"
            ),
        },
        NotificationType.APPLICATION_STATUS_CHANGED: {
            "subject": "📋 Application Update — JobHunter AI",
            "body":    (
                "Hi {name},\n\n"
                "Your application status has been updated.\n\n"
                "Status: {outcome}\n"
                "Reason: {reason}\n\n"
                "Action required: {action}\n\n"
                "Best,\nThe JobHunter AI Team"
            ),
        },
        NotificationType.INTERVIEW_SCHEDULED: {
            "subject": "🎉 Interview Scheduled — JobHunter AI",
            "body":    (
                "Hi {name},\n\n"
                "Congratulations! An interview has been scheduled.\n\n"
                "Please log in to your dashboard to view the details.\n\n"
                "Best,\nThe JobHunter AI Team"
            ),
        },
        NotificationType.AGENT_FAILED: {
            "subject": "⚠️ Action Required — JobHunter AI",
            "body":    (
                "Hi {name},\n\n"
                "One of your automated tasks encountered an issue and requires your attention.\n\n"
                "Please log in to your dashboard to review and retry.\n\n"
                "Best,\nThe JobHunter AI Team"
            ),
        },
        NotificationType.WEEKLY_SUMMARY: {
            "subject": "📊 Your Weekly Job Search Summary — JobHunter AI",
            "body":    (
                "Hi {name},\n\n"
                "Here's your weekly summary:\n\n"
                "Applications this week: {applications_this_week}\n"
                "Active interviews:      {active_interviews}\n"
                "LinkedIn posts:         {linkedin_posts}\n\n"
                "Keep it up! Log in to see the full dashboard.\n\n"
                "Best,\nThe JobHunter AI Team"
            ),
        },
    }

    template = templates.get(notification_type, {
        "subject": "Notification from JobHunter AI",
        "body":    "Hi {name},\n\nYou have a new notification. Please log in to your dashboard.\n\nBest,\nThe JobHunter AI Team",
    })

    body_text = template["body"].format(name=user_name, **{k: v for k, v in context.items() if isinstance(v, (str, int, float))})
    body_html = body_text.replace("\n", "<br>")
    subject   = template["subject"]

    return subject, f"<html><body><pre>{body_html}</pre></body></html>", body_text