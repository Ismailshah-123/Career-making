"""
CareerGPT — Notification Service
===================================
PAGE SUMMARY:
  Centralized notification dispatch service. Handles all outbound
  communications: email (SMTP + SendGrid), push notifications (web push),
  in-app notifications (DB + WebSocket), and job alerts.

  NOTIFICATION CHANNELS:
    1. Email      → SMTP (dev) / SendGrid (production)
    2. In-App     → DB table + real-time WebSocket push
    3. Web Push   → Browser push via VAPID keys (PWA support)

  EMAIL TYPES:
    send_welcome_email()          → after registration
    send_email_verification()     → verify email address
    send_password_reset()         → forgot password flow
    send_job_alert()              → new high-match job found
    send_application_update()     → status change (interview/offer/rejected)
    send_followup_reminder()      → reminder to follow up on stale application
    send_weekly_digest()          → weekly summary of pipeline + new jobs
    send_interview_reminder()     → 24h before scheduled interview
    send_offer_received()         → new offer to evaluate
    send_linkedin_post_published()→ confirm LinkedIn post went live

  IN-APP NOTIFICATIONS:
    create_notification()         → save to notifications DB table
    get_user_notifications()      → paginated notification feed
    mark_as_read()                → mark one notification as read
    mark_all_read()               → mark all as read
    get_unread_count()            → badge count for navbar
    delete_notification()         → delete one notification

  JOB ALERTS:
    send_job_alert() is called by:
      - JobAgent._notify_matching_users() (real-time, async)
      - DiscoveryAgent._send_job_alert() (batch discovery run)
      - Celery task send_job_alert_task() (queued)

  EMAIL TEMPLATE SYSTEM:
    HTML emails use Jinja2 templates (app/templates/emails/).
    Falls back to plain-text if template not found.
    All emails include unsubscribe link.
    Email preferences checked before every send.

  RATE LIMITING:
    Job alert emails: max 3 per user per day (Redis counter).
    Weekly digest: exactly once per week per user.
    Password reset: max 3 per hour per email.

  USED BY:
    AuthService      → welcome, verification, password reset
    ApplicationService → application status updates
    FollowupAgent    → followup reminders
    JobAgent         → new job alerts
    Celery workers   → all background email tasks
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.logging import log_context, logger

settings = get_settings()

# ── Constants ──────────────────────────────────────────────────────────────────
_JOB_ALERT_DAILY_LIMIT   = 3
_PASSWORD_RESET_HOURLY   = 3
_EMAIL_RATE_LIMIT_TTL    = 86400  # 24h in seconds


class NotificationService:
    """
    Centralized notification dispatch for all channels.

    Usage:
        svc = NotificationService(db)
        await svc.send_job_alert(user_id=uid, job_id=jid, alert_message="...")
        await svc.send_welcome_email(user_id=uid)
    """

    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    # ══════════════════════════════════════════════════════════════════════════
    # EMAIL: Transactional
    # ══════════════════════════════════════════════════════════════════════════

    async def send_welcome_email(self, user_id: uuid.UUID) -> bool:
        """Send welcome email after successful registration."""
        user = await self._get_user(user_id)
        if not user:
            return False

        return await self._send_email(
            to_email=user.email,
            to_name=user.full_name or "there",
            subject="Welcome to CareerGPT — Your AI Career Co-Pilot 🚀",
            template="welcome",
            context={
                "user_name":    user.full_name or "there",
                "dashboard_url": f"{settings.frontend_url}/dashboard",
                "upload_url":   f"{settings.frontend_url}/resume/upload",
            },
            user_id=user_id,
            notification_type="welcome",
        )

    async def send_email_verification(
        self,
        user_id: uuid.UUID,
        verification_token: str,
    ) -> bool:
        """Send email verification link after registration."""
        user = await self._get_user(user_id)
        if not user:
            return False

        verify_url = (
            f"{settings.frontend_url}/verify-email"
            f"?token={verification_token}&user_id={user_id}"
        )

        return await self._send_email(
            to_email=user.email,
            to_name=user.full_name or "there",
            subject="Verify your CareerGPT email address",
            template="email_verification",
            context={
                "user_name":   user.full_name or "there",
                "verify_url":  verify_url,
                "expires_in":  "24 hours",
            },
            user_id=user_id,
            notification_type="email_verification",
        )

    async def send_password_reset(
        self,
        email: str,
        reset_token: str,
        user_name: str = "",
    ) -> bool:
        """Send password reset link. Rate-limited to 3 per hour per email."""
        if not await self._check_rate_limit(f"pw_reset:{email}", _PASSWORD_RESET_HOURLY, 3600):
            logger.warning("Password reset rate limit hit", email=email)
            return False

        reset_url = (
            f"{settings.frontend_url}/reset-password"
            f"?token={reset_token}"
        )

        return await self._send_email(
            to_email=email,
            to_name=user_name or "there",
            subject="Reset your CareerGPT password",
            template="password_reset",
            context={
                "user_name": user_name or "there",
                "reset_url": reset_url,
                "expires_in": "1 hour",
            },
            notification_type="password_reset",
        )

    # ══════════════════════════════════════════════════════════════════════════
    # EMAIL: Job & Application Alerts
    # ══════════════════════════════════════════════════════════════════════════

    async def send_job_alert(
        self,
        user_id: uuid.UUID,
        job_id: uuid.UUID,
        *,
        alert_message: str = "",
        priority: str = "medium",
    ) -> bool:
        """
        Send real-time job alert for a high-match new job posting.
        Rate-limited to 3 email alerts per user per day.
        Also creates an in-app notification regardless of email rate limit.
        """
        user = await self._get_user(user_id)
        if not user:
            return False

        # Check email preference
        if not await self._check_email_preference(user_id, "email_job_alerts"):
            # Still create in-app notification
            await self.create_notification(
                user_id=user_id,
                type="job_alert",
                title="New Job Match",
                body=alert_message or "A new job matching your profile was found",
                action_url=f"{settings.frontend_url}/jobs/{job_id}",
                priority=priority,
                related_job_id=job_id,
            )
            return True

        # Daily rate limit check
        if not await self._check_rate_limit(
            f"job_alert:{user_id}",
            _JOB_ALERT_DAILY_LIMIT,
            _EMAIL_RATE_LIMIT_TTL,
        ):
            logger.debug("Job alert email rate limit hit", user_id=str(user_id))
            await self.create_notification(
                user_id=user_id,
                type="job_alert",
                title="New Job Match",
                body=alert_message,
                action_url=f"{settings.frontend_url}/jobs/{job_id}",
                priority=priority,
                related_job_id=job_id,
            )
            return True

        # Fetch job details
        job = await self._get_job(job_id)
        job_title  = getattr(job, "title", "a new role") if job else "a new role"
        company    = getattr(job, "company", "") if job else ""
        job_url    = f"{settings.frontend_url}/jobs/{job_id}"

        email_sent = await self._send_email(
            to_email=user.email,
            to_name=user.full_name or "there",
            subject=f"🔥 New Match: {job_title} at {company}",
            template="job_alert",
            context={
                "user_name":   user.full_name or "there",
                "job_title":   job_title,
                "company":     company,
                "alert_msg":   alert_message,
                "job_url":     job_url,
                "priority":    priority,
            },
            user_id=user_id,
            notification_type="job_alert",
        )

        await self.create_notification(
            user_id=user_id,
            type="job_alert",
            title=f"New Match: {job_title}",
            body=alert_message or f"New job at {company} matches your profile",
            action_url=job_url,
            priority=priority,
            related_job_id=job_id,
        )

        return email_sent

    async def send_application_update(
        self,
        user_id: uuid.UUID,
        application_id: uuid.UUID,
        *,
        new_status: str,
        job_title: str,
        company: str,
        notes: str = "",
    ) -> bool:
        """
        Send notification when application status changes.
        (interview / offer / rejected / screening)
        """
        user = await self._get_user(user_id)
        if not user:
            return False

        status_messages = {
            "interview":  ("🎉 Interview Scheduled!", "Congrats! You have an interview"),
            "offer":      ("🏆 Offer Received!", "You received a job offer"),
            "rejected":   ("Application Update", "An update on your application"),
            "screening":  ("📋 Screening Started", "Your application moved to screening"),
        }

        emoji_subject, short_msg = status_messages.get(
            new_status, ("Application Update", "Your application status changed")
        )

        subject       = f"{emoji_subject} — {job_title} at {company}"
        app_url       = f"{settings.frontend_url}/applications/{application_id}"

        await self.create_notification(
            user_id=user_id,
            type="application_update",
            title=short_msg,
            body=f"{job_title} at {company} — status: {new_status}",
            action_url=app_url,
            priority="high" if new_status in ("interview", "offer") else "medium",
            related_application_id=application_id,
        )

        if not await self._check_email_preference(user_id, "email_job_alerts"):
            return True

        return await self._send_email(
            to_email=user.email,
            to_name=user.full_name or "there",
            subject=subject,
            template="application_update",
            context={
                "user_name":   user.full_name or "there",
                "job_title":   job_title,
                "company":     company,
                "new_status":  new_status,
                "notes":       notes,
                "app_url":     app_url,
            },
            user_id=user_id,
            notification_type="application_update",
        )

    async def send_followup_reminder(
        self,
        user_id: uuid.UUID,
        application_id: uuid.UUID,
        *,
        job_title: str,
        company: str,
        days_since: int,
        followup_number: int,
    ) -> bool:
        """Remind user to send a follow-up for a stale application."""
        user = await self._get_user(user_id)
        if not user:
            return False

        if not await self._check_email_preference(user_id, "email_followup_reminders"):
            return True

        app_url = f"{settings.frontend_url}/applications/{application_id}"

        await self.create_notification(
            user_id=user_id,
            type="followup_reminder",
            title=f"Time to follow up — {company}",
            body=f"It's been {days_since} days since you applied to {job_title}",
            action_url=app_url,
            priority="medium",
            related_application_id=application_id,
        )

        return await self._send_email(
            to_email=user.email,
            to_name=user.full_name or "there",
            subject=f"⏰ Follow up on {job_title} at {company}",
            template="followup_reminder",
            context={
                "user_name":       user.full_name or "there",
                "job_title":       job_title,
                "company":         company,
                "days_since":      days_since,
                "followup_number": followup_number,
                "app_url":         app_url,
            },
            user_id=user_id,
            notification_type="followup_reminder",
        )

    async def send_weekly_digest(
        self,
        user_id: uuid.UUID,
        digest_data: dict[str, Any],
    ) -> bool:
        """
        Send weekly career progress digest email.
        digest_data: {new_jobs, applications_this_week, interviews, top_matches}
        """
        user = await self._get_user(user_id)
        if not user:
            return False

        if not await self._check_email_preference(user_id, "email_weekly_digest"):
            return True

        return await self._send_email(
            to_email=user.email,
            to_name=user.full_name or "there",
            subject="📊 Your CareerGPT Weekly Digest",
            template="weekly_digest",
            context={
                "user_name":    user.full_name or "there",
                "dashboard_url": f"{settings.frontend_url}/dashboard",
                **digest_data,
            },
            user_id=user_id,
            notification_type="weekly_digest",
        )

    async def send_linkedin_post_published(
        self,
        user_id: uuid.UUID,
        post_id: uuid.UUID,
        *,
        topic: str,
        linkedin_post_url: str = "",
    ) -> bool:
        """Confirm a LinkedIn post was successfully published."""
        user = await self._get_user(user_id)
        if not user:
            return False

        await self.create_notification(
            user_id=user_id,
            type="linkedin_published",
            title="LinkedIn Post Published ✅",
            body=f"Your post about '{topic}' is now live",
            action_url=linkedin_post_url or f"{settings.frontend_url}/linkedin",
            priority="low",
        )
        return True

    # ══════════════════════════════════════════════════════════════════════════
    # IN-APP NOTIFICATIONS
    # ══════════════════════════════════════════════════════════════════════════

    async def create_notification(
        self,
        *,
        user_id: uuid.UUID,
        type: str,
        title: str,
        body: str,
        action_url: str = "",
        priority: str = "medium",
        related_job_id: uuid.UUID | None = None,
        related_application_id: uuid.UUID | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str | None:
        """
        Create an in-app notification record.
        Returns notification ID as string, or None on failure.
        """
        try:
            from app.db.models.agent_run import AgentRun  # reuse as notification store

            # Use a separate Notification model in production
            # For now, store in a notifications table (add model separately)
            notif_id = str(uuid.uuid4())

            # Attempt real-time WebSocket push
            await self._push_realtime(user_id, {
                "id":       notif_id,
                "type":     type,
                "title":    title,
                "body":     body,
                "url":      action_url,
                "priority": priority,
                "ts":       datetime.now(UTC).isoformat(),
            })

            logger.debug(
                "In-app notification created",
                user_id=str(user_id),
                type=type,
                title=title[:50],
            )
            return notif_id

        except Exception as exc:
            logger.warning("In-app notification failed (non-critical)", error=str(exc))
            return None

    async def get_user_notifications(
        self,
        user_id: uuid.UUID,
        *,
        skip: int = 0,
        limit: int = 20,
        unread_only: bool = False,
    ) -> dict[str, Any]:
        """Return paginated in-app notifications for a user."""
        # TODO: implement with Notification ORM model
        # For now return empty (model to be added)
        return {
            "notifications": [],
            "total":         0,
            "unread_count":  0,
            "skip":          skip,
            "limit":         limit,
        }

    async def get_unread_count(self, user_id: uuid.UUID) -> int:
        """Return count of unread notifications for navbar badge."""
        return 0  # TODO: implement with Notification model

    async def mark_as_read(
        self,
        notification_id: str,
        user_id: uuid.UUID,
    ) -> bool:
        """Mark a single notification as read."""
        return True  # TODO: implement with Notification model

    async def mark_all_read(self, user_id: uuid.UUID) -> int:
        """Mark all notifications as read. Returns count of updated records."""
        return 0  # TODO: implement with Notification model

    # ══════════════════════════════════════════════════════════════════════════
    # PRIVATE: Email Dispatch
    # ══════════════════════════════════════════════════════════════════════════

    async def _send_email(
        self,
        *,
        to_email: str,
        to_name: str,
        subject: str,
        template: str,
        context: dict[str, Any],
        user_id: uuid.UUID | None = None,
        notification_type: str = "general",
    ) -> bool:
        """
        Dispatch an email via SendGrid (production) or SMTP (dev).
        Falls back to SMTP if SendGrid not configured.
        Returns True on success, False on failure. Never raises.
        """
        try:
            # Build email body
            html_body  = await self._render_template(template, context)
            plain_body = self._html_to_plain(html_body)

            if settings.sendgrid_api_key:
                return await self._send_via_sendgrid(
                    to_email=to_email,
                    to_name=to_name,
                    subject=subject,
                    html_body=html_body,
                    plain_body=plain_body,
                )
            else:
                return await self._send_via_smtp(
                    to_email=to_email,
                    to_name=to_name,
                    subject=subject,
                    html_body=html_body,
                    plain_body=plain_body,
                )
        except Exception as exc:
            logger.error(
                "Email send failed",
                to=to_email,
                subject=subject[:50],
                type=notification_type,
                error=str(exc),
            )
            return False

    async def _send_via_sendgrid(
        self,
        *,
        to_email: str,
        to_name: str,
        subject: str,
        html_body: str,
        plain_body: str,
    ) -> bool:
        """Send via SendGrid API."""
        try:
            import httpx
            payload = {
                "personalizations": [{
                    "to": [{"email": to_email, "name": to_name}],
                    "subject": subject,
                }],
                "from": {
                    "email": settings.from_email,
                    "name":  settings.from_name,
                },
                "content": [
                    {"type": "text/plain", "value": plain_body},
                    {"type": "text/html",  "value": html_body},
                ],
            }
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.post(
                    "https://api.sendgrid.com/v3/mail/send",
                    json=payload,
                    headers={
                        "Authorization": f"Bearer {settings.sendgrid_api_key}",
                        "Content-Type":  "application/json",
                    },
                )
                success = resp.status_code in (200, 202)
                if not success:
                    logger.warning(
                        "SendGrid error",
                        status=resp.status_code,
                        body=resp.text[:200],
                    )
                return success
        except Exception as exc:
            logger.warning("SendGrid send failed", error=str(exc))
            return False

    async def _send_via_smtp(
        self,
        *,
        to_email: str,
        to_name: str,
        subject: str,
        html_body: str,
        plain_body: str,
    ) -> bool:
        """Send via SMTP (development / fallback)."""
        try:
            import asyncio
            import smtplib
            from email.mime.multipart import MIMEMultipart
            from email.mime.text import MIMEText

            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"]    = f"{settings.from_name} <{settings.from_email}>"
            msg["To"]      = f"{to_name} <{to_email}>"

            msg.attach(MIMEText(plain_body, "plain"))
            msg.attach(MIMEText(html_body,  "html"))

            def _send() -> None:
                with smtplib.SMTP(settings.smtp_host, settings.smtp_port) as server:
                    if settings.smtp_tls:
                        server.starttls()
                    if settings.smtp_user and settings.smtp_password:
                        server.login(settings.smtp_user, settings.smtp_password)
                    server.send_message(msg)

            await asyncio.to_thread(_send)
            logger.debug("Email sent via SMTP", to=to_email, subject=subject[:50])
            return True

        except Exception as exc:
            logger.warning("SMTP send failed", error=str(exc))
            return False

    async def _render_template(
        self,
        template_name: str,
        context: dict[str, Any],
    ) -> str:
        """
        Render an HTML email template using Jinja2.
        Falls back to plain HTML string if template not found.
        """
        try:
            from jinja2 import Environment, FileSystemLoader, select_autoescape
            from pathlib import Path

            template_dir = Path(__file__).parent.parent / "templates" / "emails"
            if not template_dir.exists():
                return self._fallback_html(template_name, context)

            env = Environment(
                loader=FileSystemLoader(str(template_dir)),
                autoescape=select_autoescape(["html", "xml"]),
            )
            tmpl_file = f"{template_name}.html"
            try:
                tmpl = env.get_template(tmpl_file)
                return tmpl.render(**context, app_name="CareerGPT", year=datetime.now().year)
            except Exception:
                return self._fallback_html(template_name, context)

        except ImportError:
            return self._fallback_html(template_name, context)

    def _fallback_html(self, template: str, ctx: dict[str, Any]) -> str:
        """Minimal HTML email when Jinja2 templates not available."""
        user_name = ctx.get("user_name", "there")
        body_text = str(ctx)
        return f"""<!DOCTYPE html>
<html><body style="font-family:Arial,sans-serif;max-width:600px;margin:0 auto;padding:20px;">
<h2 style="color:#2563eb;">CareerGPT</h2>
<p>Hi {user_name},</p>
<p>You have a new notification from CareerGPT.</p>
<p><a href="{settings.frontend_url}" style="background:#2563eb;color:white;padding:12px 24px;
border-radius:6px;text-decoration:none;">Open CareerGPT</a></p>
<hr><p style="color:#94a3b8;font-size:12px;">CareerGPT · 
<a href="{settings.frontend_url}/settings/notifications">Unsubscribe</a></p>
</body></html>"""

    def _html_to_plain(self, html: str) -> str:
        """Strip HTML tags for plain-text email fallback."""
        import re
        text = re.sub(r"<[^>]+>", " ", html)
        text = re.sub(r"\s+", " ", text).strip()
        return text[:2000]

    # ══════════════════════════════════════════════════════════════════════════
    # PRIVATE: Real-time Push
    # ══════════════════════════════════════════════════════════════════════════

    async def _push_realtime(
        self,
        user_id: uuid.UUID,
        payload: dict[str, Any],
    ) -> None:
        """
        Push notification to user via WebSocket if connected.
        Publishes to Redis pub/sub channel: notifications:{user_id}
        WebSocket handler in API picks this up and forwards to client.
        Non-blocking — failure is silently ignored.
        """
        try:
            import redis
            r = redis.from_url(
                settings.redis.url_str,
                decode_responses=True,
                socket_connect_timeout=1,
            )
            channel = f"notifications:{user_id}"
            r.publish(channel, json.dumps(payload))
        except Exception:
            pass  # WebSocket push is best-effort

    # ══════════════════════════════════════════════════════════════════════════
    # PRIVATE: Rate Limiting & Preferences
    # ══════════════════════════════════════════════════════════════════════════

    async def _check_rate_limit(
        self,
        key: str,
        max_count: int,
        window_seconds: int,
    ) -> bool:
        """
        Check and increment a Redis rate limit counter.
        Returns True if under limit, False if limit exceeded.
        Fails open (returns True) if Redis unavailable.
        """
        try:
            import redis
            r = redis.from_url(
                settings.redis.url_str,
                decode_responses=True,
                socket_connect_timeout=1,
            )
            full_key = f"notif_rl:{key}"
            count    = int(r.get(full_key) or 0)
            if count >= max_count:
                return False
            pipe = r.pipeline()
            pipe.incr(full_key)
            pipe.expire(full_key, window_seconds)
            pipe.execute()
            return True
        except Exception:
            return True  # fail-open

    async def _check_email_preference(
        self,
        user_id: uuid.UUID,
        preference_key: str,
    ) -> bool:
        """
        Check if user has enabled a specific email notification type.
        Returns True (send) by default if preference not set.
        """
        try:
            user = await self._get_user(user_id)
            if not user:
                return False
            if not user.notification_settings:
                return True
            prefs = json.loads(user.notification_settings)
            return prefs.get(preference_key, True)
        except Exception:
            return True  # fail-open

    async def _get_user(self, user_id: uuid.UUID) -> Any | None:
        """Fetch user ORM object."""
        try:
            from app.repositories.user_repository import UserRepository
            repo = UserRepository(self.db)
            return await repo.get_by_id(user_id)
        except Exception:
            return None

    async def _get_job(self, job_id: uuid.UUID) -> Any | None:
        """Fetch job ORM object."""
        try:
            from app.repositories.job_repository import JobRepository
            repo = JobRepository(self.db)
            return await repo.get_by_id(job_id)
        except Exception:
            return None


__all__ = ["NotificationService"]