"""
CareerGPT — Followup Agent
=============================
PAGE SUMMARY:
  Orchestrates the tool functions in followup_agent/tools.py into the two
  entry points the rest of the app calls:

  CONTRACT:
    agent = FollowupAgent(db)                       # db optional
    message = await agent.generate(application=.., job=.., user=..)
      -> str  (the follow-up message body -- used by
               ApplicationService.send_manual_followup for an
               on-demand, user-triggered follow-up)

    agent = FollowupAgent()                          # db not required here
    result = await agent.send_followup(application=.., db=..)
      -> {"sent": bool, "eligible": bool, "reason": str,
          "followup_number": int, "message": str | None}
      (used by job_tasks.send_followup_task -- the scheduled, automatic
      T+7-day follow-up flow)

  Both paths share the same underlying tools: extract_application_brief()
  builds a flat dict from the ORM objects, check_followup_eligibility()
  enforces the timing/count rules, generate_followup_message() does the
  actual LLM write.

  DELIVERY:
    send_followup() attempts an email send via the recruiter contact on
    file (if one exists and EMAIL_ENABLED is set) through the same
    send_email_task used everywhere else. If no recruiter contact is on
    file, the message is generated and staged on the Application row for
    the user to see/send manually, and `sent` is False.

  AGENT RUN AUDIT:
    Every call writes a best-effort AgentRun row (agent_type="followup"),
    mirroring the pattern in cover_letter_agent / resume_agent.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.constants import AgentType, FOLLOWUP_WAIT_DAYS
from app.core.logging import logger
from app.agents.followup_agent import tools


class FollowupAgent:
    """
    Follow-up generation + delivery orchestrator.

    Usage (manual, on-demand -- has a db session up front):
        agent = FollowupAgent(db)
        message = await agent.generate(application=app, job=job, user=user)

    Usage (scheduled, via Celery -- receives db per-call):
        agent = FollowupAgent()
        result = await agent.send_followup(application=app, db=db)
    """

    def __init__(self, db: AsyncSession | None = None) -> None:
        self.db = db
        self._llm_calls = 0

    # -- Manual / on-demand ---------------------------------------------------

    async def generate(self, *, application: Any, job: Any, user: Any) -> str:
        """
        Generate a follow-up message body for an application. Does not
        check eligibility and does not send anything -- used when the
        user explicitly asks for a follow-up right now.
        """
        resume = await self._load_resume(application.resume_id) if self.db else None
        brief = tools.extract_application_brief(application, user, resume, job)

        self._llm_calls += 1
        result = await tools.generate_followup_message(brief)
        message = result.get("full_email") or result.get("body", "")

        if self.db is not None:
            await self._save_agent_run(
                user_id=user.id,
                status="success",
                input_data={"application_id": str(application.id), "mode": "manual"},
                output_data={"word_count": result.get("word_count", len(message.split()))},
                related_job_id=getattr(job, "id", None),
                related_resume_id=getattr(resume, "id", None),
            )

        return message

    # -- Scheduled --------------------------------------------------------------

    async def send_followup(self, *, application: Any, db: AsyncSession) -> dict[str, Any]:
        """
        Full scheduled follow-up pipeline: load context, check eligibility,
        generate the message, attempt delivery, and stage the resulting
        fields on `application` (caller is responsible for `db.commit()`).
        """
        job, user, resume = await self._load_context(application, db)
        brief = tools.extract_application_brief(application, user, resume, job)

        eligibility = tools.check_followup_eligibility(brief)
        if not eligibility.get("eligible"):
            return {
                "sent": False,
                "eligible": False,
                "reason": eligibility.get("reason", "Not eligible"),
                "followup_number": brief.get("followup_count", 0),
                "message": None,
            }

        self._llm_calls += 1
        result = await tools.generate_followup_message(brief)
        message = result.get("full_email") or result.get("body", "")
        followup_number = result.get("followup_number", eligibility.get("followup_number", 1))

        sent = await self._attempt_delivery(application, user, result, db)

        now = datetime.now(UTC)
        application.followup_count = (application.followup_count or 0) + 1
        application.last_followup_at = now
        application.next_followup_at = now + timedelta(days=FOLLOWUP_WAIT_DAYS)
        application.followup_message = message

        history = list(application.followup_history or [])
        history.append({
            "followup_number": followup_number,
            "sent_at": now.isoformat(),
            "sent": sent,
            "subject": result.get("subject", ""),
            "word_count": result.get("word_count", len(message.split())),
        })
        application.followup_history = history

        await self._save_agent_run(
            user_id=user.id,
            status="success",
            input_data={"application_id": str(application.id), "mode": "scheduled"},
            output_data={"sent": sent, "followup_number": followup_number},
            related_job_id=getattr(job, "id", None),
            related_resume_id=getattr(resume, "id", None),
            db=db,
        )

        return {
            "sent": sent,
            "eligible": True,
            "reason": "delivered" if sent else "generated_no_channel",
            "followup_number": followup_number,
            "message": message,
        }

    # -- Private helpers ----------------------------------------------------------

    async def _load_context(self, application: Any, db: AsyncSession) -> tuple[Any, Any, Any]:
        from app.db.models.job import Job
        from app.db.models.user import User
        from app.db.models.resume import Resume
        from sqlalchemy import select

        job = (await db.execute(select(Job).where(Job.id == application.job_id))).scalar_one_or_none()
        user = (await db.execute(select(User).where(User.id == application.user_id))).scalar_one_or_none()
        resume = None
        if application.resume_id:
            resume = (
                await db.execute(select(Resume).where(Resume.id == application.resume_id))
            ).scalar_one_or_none()
        return job, user, resume

    async def _load_resume(self, resume_id: "uuid.UUID | None") -> Any:
        if not resume_id or self.db is None:
            return None
        from app.db.models.resume import Resume
        from sqlalchemy import select

        result = await self.db.execute(select(Resume).where(Resume.id == resume_id))
        return result.scalar_one_or_none()

    async def _attempt_delivery(
        self,
        application: Any,
        user: Any,
        generated: dict[str, Any],
        db: AsyncSession,
    ) -> bool:
        """
        Best-effort delivery via email to the on-file recruiter contact.
        Returns False (never raises) if there's no channel available or
        the send fails -- the message is still staged on the application
        either way.
        """
        from app.core.config import settings

        if not settings.EMAIL_ENABLED or not application.recruiter_id:
            return False

        try:
            from app.db.models.recruiter import Recruiter
            from sqlalchemy import select

            recruiter = (
                await db.execute(select(Recruiter).where(Recruiter.id == application.recruiter_id))
            ).scalar_one_or_none()
            if not recruiter or not recruiter.email:
                return False

            from app.workers.notification_tasks import send_email_task

            send_email_task.apply_async(
                kwargs={
                    "to_email":  recruiter.email,
                    "to_name":   recruiter.full_name or "there",
                    "subject":   generated.get("subject") or f"Following up — {user.full_name}",
                    "body_html": f"<p>{generated.get('body', '').replace(chr(10), '<br/>')}</p>",
                    "body_text": generated.get("body", ""),
                },
                priority=6,
            )
            return True
        except Exception as exc:
            logger.warning("Follow-up email delivery failed (non-critical)", error=str(exc))
            return False

    async def _save_agent_run(
        self,
        *,
        user_id: "uuid.UUID",
        status: str,
        input_data: dict[str, Any],
        output_data: dict[str, Any],
        related_job_id: "uuid.UUID | None" = None,
        related_resume_id: "uuid.UUID | None" = None,
        db: AsyncSession | None = None,
    ) -> None:
        session = db or self.db
        if session is None:
            return
        try:
            from app.db.models.agent_run import AgentRun

            record = AgentRun(
                user_id=user_id,
                agent_type=AgentType.FOLLOWUP.value,
                status=status,
                started_at=datetime.now(UTC),
                completed_at=datetime.now(UTC),
                duration_ms=0,
                input_data=json.dumps(input_data)[:5000],
                output_data=json.dumps(output_data)[:5000],
                llm_calls=self._llm_calls,
                related_job_id=related_job_id,
                related_resume_id=related_resume_id,
            )
            session.add(record)
            await session.flush()
        except Exception as exc:
            logger.warning("AgentRun save failed (non-critical)", error=str(exc))


__all__ = ["FollowupAgent"]
