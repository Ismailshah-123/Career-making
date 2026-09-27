"""
CareerGPT — Outreach Agent
============================
PAGE SUMMARY:
  Outreach orchestrator. Generates ALL non-cover-letter outreach content:
  LinkedIn connection notes, InMail, recruiter cold emails, follow-ups,
  interview thank-you notes, referral requests, offer negotiations, and
  complete 3-touch outreach sequences.

  CONTRACT (ApplicationService reads these exact keys):
    agent.generate_linkedin_message(resume=, job=)  → {"message": str}
    agent.generate_recruiter_email(resume=, job=)   → {"body": str}

  ALL MODES:
    generate_linkedin_message()    → 300-char LinkedIn connection note
    generate_recruiter_email()     → cold email to recruiter/HM
    generate_inmessage()           → LinkedIn InMail (after connect accepted)
    generate_thankyou()            → post-interview thank-you note
    generate_referral_request()    → ask contact for internal referral
    generate_offer_response()      → professional offer negotiation
    generate_followup()            → follow-up for stale application
    generate_full_sequence()       → 3-touch sequence (initial + 2 follow-ups)

  CAREER FOCUS AWARENESS:
    All prompts receive job.title and job.company as context — they work
    identically for SAP Consultant, AI Engineer, DevOps Lead, or any other
    role. The agent never hard-codes job categories. Target roles come from
    the Job ORM object which was scraped from real job boards.

  RATE LIMITING:
    LinkedIn automation (connection notes) is rate-limited at 20/day via
    a Redis counter. Recruiter emails have no rate limit. The rate limiter
    is enforced in generate_linkedin_message() only — other methods skip it.

  AGENT RUN AUDIT:
    AgentRun rows written with agent_type="outreach" for billing tracking.
"""

from __future__ import annotations

import json
import time
import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.constants import AgentType
from app.core.logging import log_context, logger
from app.agents.outreach_agent import tools

_LINKEDIN_DAILY_LIMIT = 20
_LINKEDIN_LIMIT_TTL   = 60 * 60 * 24  # 24 hours


class OutreachAgent:
    """
    Outreach content generation orchestrator.

    Usage (called by ApplicationService):
        agent = OutreachAgent(db)
        li_result    = await agent.generate_linkedin_message(resume=r, job=j)
        email_result = await agent.generate_recruiter_email(resume=r, job=j)
    """

    def __init__(self, db: AsyncSession) -> None:
        self.db = db
        self._llm_calls = 0

    # ── Mode 1: LinkedIn Connection Note ──────────────────────────────────────

    async def generate_linkedin_message(
        self,
        *,
        resume: Any,
        job: Any,
        recipient_name: str = "",
        recipient_title: str = "",
        specific_hook: str = "",
        mutual_connection: str = "",
    ) -> dict[str, Any]:
        """
        Generate a 300-char LinkedIn connection request note.
        Called by ApplicationService._generate_linkedin_message().
        Returns: {"message": str, "character_count": int}
        """
        t_start = time.monotonic()

        with log_context(agent="outreach_linkedin", job_id=str(job.id)):
            # Rate limit check — protect user's LinkedIn account
            if not await self._check_linkedin_rate_limit(resume.user_id):
                logger.warning(
                    "LinkedIn daily connection limit reached",
                    user_id=str(resume.user_id),
                )
                return {
                    "message": (
                        f"Hi {recipient_name or 'there'}, I'm interested in the "
                        f"{job.title} role at {job.company}. Would love to connect. "
                        f"—{resume.name or 'Candidate'}"
                    )[:300],
                    "rate_limited": True,
                }

            candidate = tools.extract_candidate_outreach_brief(resume)
            job_brief  = tools.extract_job_outreach_brief(job)

            result = await tools.generate_linkedin_connection_note(
                candidate=candidate,
                job=job_brief,
                recipient_name=recipient_name,
                recipient_title=recipient_title,
                specific_hook=specific_hook or job.ai_summary or "",
                mutual_connection=mutual_connection,
            )
            self._llm_calls += 1

            await self._increment_linkedin_counter(resume.user_id)

            duration_ms = round((time.monotonic() - t_start) * 1000, 2)
            await self._save_agent_run(
                user_id=resume.user_id,
                status="success",
                input_data={"type": "linkedin_connection", "job_id": str(job.id)},
                output_data={"char_count": result.get("character_count", 0)},
                duration_ms=duration_ms,
                related_job_id=job.id,
            )

            return result

    # ── Mode 2: Recruiter Cold Email ──────────────────────────────────────────

    async def generate_recruiter_email(
        self,
        *,
        resume: Any,
        job: Any,
        recipient_name: str = "",
        recipient_title: str = "",
        company_context: str = "",
    ) -> dict[str, Any]:
        """
        Generate a cold email to a recruiter or hiring manager.
        Called by ApplicationService._generate_recruiter_email().
        Returns: {"body": str, "subject": str, "word_count": int}
        """
        t_start = time.monotonic()

        with log_context(agent="outreach_email", job_id=str(job.id)):
            candidate = tools.extract_candidate_outreach_brief(resume)
            job_brief  = tools.extract_job_outreach_brief(job)

            result = await tools.generate_recruiter_email(
                candidate=candidate,
                job=job_brief,
                recipient_name=recipient_name,
                recipient_title=recipient_title or "Hiring Manager",
                company_context=company_context or job.ai_summary or "",
            )
            self._llm_calls += 1

            duration_ms = round((time.monotonic() - t_start) * 1000, 2)
            await self._save_agent_run(
                user_id=resume.user_id,
                status="success",
                input_data={"type": "recruiter_email", "job_id": str(job.id)},
                output_data={"word_count": result.get("word_count", 0)},
                duration_ms=duration_ms,
                related_job_id=job.id,
            )

            # Normalize return to guarantee "body" key exists
            if "body" not in result and "message" in result:
                result["body"] = result["message"]

            return result

    # ── Mode 3: LinkedIn InMail ────────────────────────────────────────────────

    async def generate_inmessage(
        self,
        *,
        resume: Any,
        job: Any,
        recipient_name: str = "",
        recipient_title: str = "",
        specific_hook: str = "",
    ) -> dict[str, Any]:
        """
        Generate a LinkedIn InMail — used after connection is accepted,
        or for premium InMail to recruiters who aren't connected yet.
        """
        candidate = tools.extract_candidate_outreach_brief(resume)
        job_brief  = tools.extract_job_outreach_brief(job)

        result = await tools.generate_linkedin_inmessage(
            candidate=candidate,
            job=job_brief,
            recipient_name=recipient_name,
            recipient_title=recipient_title,
            specific_hook=specific_hook or job.ai_summary or "",
        )
        self._llm_calls += 1
        return result

    # ── Mode 4: Interview Thank-You ───────────────────────────────────────────

    async def generate_thankyou(
        self,
        *,
        resume: Any,
        job: Any,
        interviewer_name: str,
        interviewer_title: str = "",
        interview_date: str = "",
        specific_discussion_point: str = "",
        concern_to_address: str = "",
        additional_value: str = "",
    ) -> dict[str, Any]:
        """
        Generate a post-interview thank-you note.
        Best practice: send within 2-4 hours of interview.
        """
        t_start = time.monotonic()

        with log_context(agent="outreach_thankyou", job_id=str(job.id)):
            candidate = tools.extract_candidate_outreach_brief(resume)
            job_brief  = tools.extract_job_outreach_brief(job)

            result = await tools.generate_interview_thankyou(
                candidate=candidate,
                job=job_brief,
                interviewer_name=interviewer_name,
                interviewer_title=interviewer_title,
                interview_date=interview_date,
                specific_discussion_point=specific_discussion_point,
                concern_to_address=concern_to_address,
                additional_value=additional_value,
            )
            self._llm_calls += 1

            duration_ms = round((time.monotonic() - t_start) * 1000, 2)
            logger.info(
                "Thank-you note generated",
                word_count=result.get("word_count", 0),
                duration_ms=duration_ms,
            )
            return result

    # ── Mode 5: Referral Request ──────────────────────────────────────────────

    async def generate_referral_request(
        self,
        *,
        resume: Any,
        job: Any,
        contact_name: str,
        relationship: str = "former colleague",
        requester_pitch: str = "",
    ) -> dict[str, Any]:
        """
        Generate a message asking a contact to refer the candidate.
        Referrals have 4x interview rate — highest ROI outreach type.
        """
        candidate = tools.extract_candidate_outreach_brief(resume)
        job_brief  = tools.extract_job_outreach_brief(job)

        result = await tools.generate_referral_request(
            candidate=candidate,
            job=job_brief,
            contact_name=contact_name,
            relationship=relationship,
            requester_pitch=requester_pitch,
        )
        self._llm_calls += 1
        return result

    # ── Mode 6: Offer Negotiation ─────────────────────────────────────────────

    async def generate_offer_response(
        self,
        *,
        resume: Any,
        job: Any,
        offered_salary: int,
        offered_equity: str = "",
        target_salary: int = 0,
        market_data: str = "",
        competing_offer: str = "",
        priorities: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Generate a professional job offer negotiation response.
        85% of employers expect negotiation — this agent empowers users to do it.
        """
        t_start = time.monotonic()
        candidate = tools.extract_candidate_outreach_brief(resume)
        job_brief  = tools.extract_job_outreach_brief(job)
        prio = priorities or ["higher base salary", "more equity", "additional PTO"]

        result = await tools.generate_offer_negotiation(
            candidate=candidate,
            job=job_brief,
            offered_salary=offered_salary,
            offered_equity=offered_equity,
            target_salary=target_salary or int(offered_salary * 1.15),
            market_data=market_data,
            competing_offer=competing_offer,
            priority_1=prio[0] if len(prio) > 0 else "higher base salary",
            priority_2=prio[1] if len(prio) > 1 else "more equity",
            priority_3=prio[2] if len(prio) > 2 else "additional vacation days",
        )
        self._llm_calls += 1

        duration_ms = round((time.monotonic() - t_start) * 1000, 2)
        logger.info(
            "Offer negotiation generated",
            offered=offered_salary,
            target=target_salary,
            duration_ms=duration_ms,
        )
        return result

    # ── Mode 7: Follow-up Email ────────────────────────────────────────────────

    async def generate_followup(
        self,
        *,
        resume: Any,
        job: Any,
        applied_role: str,
        original_date: str,
        days_since: int,
        followup_number: int = 1,
        new_value_point: str = "",
    ) -> dict[str, Any]:
        """
        Generate a follow-up email for a stale application.
        followup_number 1→ gentle bump, 2→ new angle, 3→ graceful exit.
        Called by FollowupAgent which handles timing/scheduling logic.
        """
        candidate = tools.extract_candidate_outreach_brief(resume)
        job_brief  = tools.extract_job_outreach_brief(job)

        result = await tools.generate_followup_email(
            candidate=candidate,
            job=job_brief,
            applied_role=applied_role,
            original_date=original_date,
            days_since=days_since,
            followup_number=followup_number,
            new_value_point=new_value_point,
        )
        self._llm_calls += 1
        return result

    # ── Mode 8: Full 3-Touch Sequence ─────────────────────────────────────────

    async def generate_full_sequence(
        self,
        *,
        resume: Any,
        job: Any,
        recipient_name: str = "",
        recipient_title: str = "",
        channel: str = "email",
    ) -> dict[str, Any]:
        """
        Generate a complete 3-touch outreach sequence in one LLM call.
        Returns: {sequence: [{touch_number, send_day, subject, body}], ...}
        Used by: API endpoint POST /api/v1/applications/{id}/outreach-sequence
        """
        t_start = time.monotonic()

        with log_context(agent="outreach_sequence", job_id=str(job.id)):
            candidate = tools.extract_candidate_outreach_brief(resume)
            job_brief  = tools.extract_job_outreach_brief(job)

            result = await tools.generate_full_sequence(
                candidate=candidate,
                job=job_brief,
                recipient_name=recipient_name,
                recipient_title=recipient_title or "Hiring Manager",
                channel=channel,
            )
            self._llm_calls += 1

            duration_ms = round((time.monotonic() - t_start) * 1000, 2)
            sequence_count = len(result.get("sequence", []))

            await self._save_agent_run(
                user_id=resume.user_id,
                status="success",
                input_data={"type": "full_sequence", "channel": channel, "job_id": str(job.id)},
                output_data={"touch_count": sequence_count},
                duration_ms=duration_ms,
                related_job_id=job.id,
            )

            logger.info(
                "Outreach sequence generated",
                touches=sequence_count,
                channel=channel,
                duration_ms=duration_ms,
            )
            return result

    # ── Private: Rate Limiting ─────────────────────────────────────────────────

    async def _check_linkedin_rate_limit(self, user_id: uuid.UUID) -> bool:
        """Check if user is under LinkedIn daily connection note limit (20/day)."""
        try:
            import redis
            from app.core.config import get_settings
            settings = get_settings()
            r = redis.from_url(settings.redis.url_str, decode_responses=True, socket_connect_timeout=2)
            count = int(r.get(f"li_connections:{user_id}") or 0)
            return count < _LINKEDIN_DAILY_LIMIT
        except Exception:
            return True  # fail-open

    async def _increment_linkedin_counter(self, user_id: uuid.UUID) -> None:
        """Increment LinkedIn daily connection counter after sending a note."""
        try:
            import redis
            from app.core.config import get_settings
            settings = get_settings()
            r = redis.from_url(settings.redis.url_str, decode_responses=True, socket_connect_timeout=2)
            key = f"li_connections:{user_id}"
            pipe = r.pipeline()
            pipe.incr(key)
            pipe.expire(key, _LINKEDIN_LIMIT_TTL)
            pipe.execute()
        except Exception:
            pass

    # ── Private: Agent Run ────────────────────────────────────────────────────

    async def _save_agent_run(
        self,
        *,
        user_id: uuid.UUID,
        status: str,
        input_data: dict,
        output_data: dict,
        duration_ms: float,
        related_job_id: uuid.UUID | None = None,
    ) -> None:
        try:
            from app.db.models.agent_run import AgentRun
            record = AgentRun(
                user_id=user_id,
                agent_type=AgentType.OUTREACH.value,
                status=status,
                started_at=datetime.now(UTC),
                completed_at=datetime.now(UTC),
                duration_ms=duration_ms,
                input_data=json.dumps(input_data)[:5000],
                output_data=json.dumps(output_data)[:5000],
                llm_calls=self._llm_calls,
                related_job_id=related_job_id,
            )
            self.db.add(record)
            await self.db.flush()
        except Exception as exc:
            logger.warning("AgentRun save failed (non-critical)", error=str(exc))


__all__ = ["OutreachAgent"]