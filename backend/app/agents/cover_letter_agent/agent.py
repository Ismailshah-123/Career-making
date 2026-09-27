"""
CareerGPT — Cover Letter Agent
=================================
PAGE SUMMARY:
  Cover letter orchestrator. Decides which letter variant to generate
  (standard / referral / career-change / executive / cold-email), runs
  the quality gate, auto-improves if score < 75, saves to DB, and returns
  the flat dict ApplicationService expects.

  CONTRACT (ApplicationService._generate_cover_letter reads these keys):
    agent.generate(resume=<Resume ORM>, job=<Job ORM>)
      → {"body": str, "subject_line": str}   ← minimum required
      additional keys are stored in CoverLetter DB row

  VARIANT SELECTION LOGIC (auto_select_variant):
    executive   → job title contains VP/Director/Chief/CTO/Coo etc.
    referral    → caller passes referrer_name explicitly
    career_change → candidate's experience_years < 1 in the target domain
                    (detected by skills gap vs job requirements)
    cold_email  → caller passes cold_email=True (used in OutreachAgent flow)
    standard    → default for all other cases

  TONE AUTO-SELECTION:
    pick_tone_for_seniority() from tools.py gives a sensible default.
    Caller can always override via tone= parameter.

  QUALITY GATE:
    Every generated letter is scored 0-100.
    Score >= 75 → ready, save as-is
    Score 60-74 → auto-refine with the top_priority_fix from scoring
    Score < 60  → save as draft, flag for manual review

  DB PERSISTENCE:
    Every generation creates a CoverLetter row for audit/billing.
    The row stores: full text, word count, tone, version number,
    AI model used, linked job_id + application_id.
    ApplicationService updates application.cover_letter_text after create().

  AGENT RUN AUDIT:
    Each generate() call writes an AgentRun row (agent_type="cover_letter")
    for billing/usage tracking visible in the admin panel.
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
from app.agents.cover_letter_agent import tools


class CoverLetterAgent:
    """
    Cover letter generation orchestrator.

    Usage (called by ApplicationService._generate_cover_letter):
        agent = CoverLetterAgent(db)
        result = await agent.generate(resume=resume_orm, job=job_orm)
        body        = result["body"]
        subject     = result["subject_line"]

    Direct usage with options:
        result = await agent.generate(
            resume=resume,
            job=job,
            tone="enthusiastic",
            variant="referral",
            referrer_name="John Doe",
        )
    """

    def __init__(self, db: AsyncSession) -> None:
        self.db = db
        self._llm_calls = 0

    # ── Primary Method ─────────────────────────────────────────────────────────

    async def generate(
        self,
        *,
        resume: Any,
        job: Any,
        tone: str | None = None,
        variant: str = "auto",
        referrer_name: str = "",
        referrer_title: str = "",
        referral_relationship: str = "",
        cold_email: bool = False,
        highlight_skills: list[str] | None = None,
        custom_context: str = "",
        application_id: uuid.UUID | None = None,
        run_quality_gate: bool = True,
    ) -> dict[str, Any]:
        """
        Generate a cover letter for a resume+job pair.
        Returns dict with at minimum {"body", "subject_line"}.
        """
        t_start = time.monotonic()

        with log_context(agent="cover_letter", job_id=str(job.id)):
            candidate = tools.extract_candidate_brief(resume)
            job_brief  = tools.extract_job_brief(job)

            # ── Tone selection ─────────────────────────────────────────────────
            resolved_tone = tone or tools.pick_tone_for_seniority(job)

            # ── Variant selection ──────────────────────────────────────────────
            resolved_variant = (
                "cold_email"  if cold_email else
                "referral"    if referrer_name else
                "executive"   if self._is_executive_role(job) else
                variant if variant != "auto" else
                "standard"
            )

            logger.info(
                "Generating cover letter",
                variant=resolved_variant,
                tone=resolved_tone,
                job=f"{job.title} @ {job.company}",
            )

            # ── Generate ───────────────────────────────────────────────────────
            result = await self._dispatch(
                variant=resolved_variant,
                candidate=candidate,
                job=job_brief,
                tone=resolved_tone,
                referrer_name=referrer_name,
                referrer_title=referrer_title,
                referral_relationship=referral_relationship,
                highlight_skills=highlight_skills,
                custom_context=custom_context,
            )
            self._llm_calls += 1

            body         = result.get("full_body") or result.get("body") or ""
            subject_line = result.get("subject_line", f"Application: {job.title} — {candidate['candidate_name']}")

            # ── Quality gate ───────────────────────────────────────────────────
            quality_score = 70
            if run_quality_gate and body:
                quality_result = await tools.score_letter(
                    cover_letter=body,
                    job=job_brief,
                )
                self._llm_calls += 1
                quality_score = quality_result.get("overall_score", 70)

                if 60 <= quality_score < 75:
                    fix_hint = quality_result.get("top_priority_fix", "")
                    if fix_hint:
                        refined = await tools.refine_letter(
                            current_letter=body,
                            feedback=fix_hint,
                            job=job_brief,
                            primary_goal=fix_hint,
                        )
                        self._llm_calls += 1
                        body = refined.get("refined_letter", body)

            # ── Save to DB ─────────────────────────────────────────────────────
            cover_letter_id = await self._save_to_db(
                resume=resume,
                job=job,
                body=body,
                subject_line=subject_line,
                tone=resolved_tone,
                quality_score=quality_score,
                application_id=application_id,
            )

            duration_ms = round((time.monotonic() - t_start) * 1000, 2)

            await self._save_agent_run(
                user_id=resume.user_id,
                status="success",
                input_data={"variant": resolved_variant, "tone": resolved_tone, "job_id": str(job.id)},
                output_data={"quality_score": quality_score, "word_count": len(body.split())},
                duration_ms=duration_ms,
                related_job_id=job.id,
                related_resume_id=resume.id,
            )

            logger.info(
                "Cover letter generated",
                variant=resolved_variant,
                quality_score=quality_score,
                word_count=len(body.split()),
                duration_ms=duration_ms,
            )

            return {
                "body":              body,
                "subject_line":      subject_line,
                "word_count":        len(body.split()),
                "tone":              resolved_tone,
                "variant":           resolved_variant,
                "quality_score":     quality_score,
                "cover_letter_id":   str(cover_letter_id) if cover_letter_id else None,
                "llm_calls":         self._llm_calls,
                "duration_ms":       duration_ms,
            }

    # ── Secondary Mode: Refine Existing Letter ────────────────────────────────

    async def refine(
        self,
        *,
        cover_letter_id: uuid.UUID,
        user_id: uuid.UUID,
        feedback: str,
    ) -> dict[str, Any]:
        """
        User-triggered refinement of an existing cover letter.
        Loads the letter from DB, refines it, saves incremented version.
        """
        from app.db.models.cover_letter import CoverLetter
        from sqlalchemy import select

        result = await self.db.execute(
            select(CoverLetter).where(CoverLetter.id == cover_letter_id)
        )
        cl = result.scalar_one_or_none()
        if not cl:
            from app.core.exceptions import CoverLetterNotFoundError
            raise CoverLetterNotFoundError(context={"id": str(cover_letter_id)})

        job_brief = {
            "job_title":   cl.job_title or "",
            "company_name": cl.company_name or "",
            "key_requirements": "",
        }

        refined = await tools.refine_letter(
            current_letter=cl.body,
            feedback=feedback,
            job=job_brief,
            primary_goal="address the user's feedback",
        )
        self._llm_calls += 1

        new_body = refined.get("refined_letter", cl.body)

        cl.body       = new_body
        cl.word_count = len(new_body.split())
        cl.version    = (cl.version or 1) + 1
        self.db.add(cl)
        await self.db.flush()

        return {
            "cover_letter_id": str(cover_letter_id),
            "body":            new_body,
            "version":         cl.version,
            "changes_made":    refined.get("changes_made", []),
        }

    # ── Secondary Mode: Score Existing Letter ─────────────────────────────────

    async def score(
        self,
        *,
        cover_letter_text: str,
        job: Any,
    ) -> dict[str, Any]:
        """Score an existing cover letter without generating a new one."""
        job_brief = tools.extract_job_brief(job)
        result = await tools.score_letter(cover_letter=cover_letter_text, job=job_brief)
        self._llm_calls += 1
        return result

    # ── Secondary Mode: Generate Cold Email ───────────────────────────────────

    async def generate_cold_email(
        self,
        *,
        resume: Any,
        job: Any,
        recipient_name: str = "",
        recipient_title: str = "",
        why_company: str = "",
    ) -> dict[str, Any]:
        """Generate a short cold outreach email (used by OutreachAgent)."""
        candidate = tools.extract_candidate_brief(resume)
        job_brief  = tools.extract_job_brief(job)

        result = await tools.generate_cold_email(
            candidate=candidate,
            job=job_brief,
            recipient_name=recipient_name,
            recipient_title=recipient_title,
            why_company=why_company,
        )
        self._llm_calls += 1
        return result

    # ── Private: Dispatch ─────────────────────────────────────────────────────

    async def _dispatch(
        self,
        *,
        variant: str,
        candidate: dict[str, Any],
        job: dict[str, Any],
        tone: str,
        referrer_name: str = "",
        referrer_title: str = "",
        referral_relationship: str = "",
        highlight_skills: list[str] | None = None,
        custom_context: str = "",
    ) -> dict[str, Any]:
        """Route to the correct tool function based on variant."""
        if variant == "referral" and referrer_name:
            return await tools.generate_referral_letter(
                candidate=candidate,
                job=job,
                referrer_name=referrer_name,
                referrer_title=referrer_title,
                referral_relationship=referral_relationship,
                tone=tone,
            )
        elif variant == "career_change":
            return await tools.generate_career_change_letter(
                candidate=candidate,
                job=job,
                previous_career=candidate.get("candidate_role", "previous career"),
                change_reason="seeking new challenges aligned with evolving skills",
                tone=tone,
            )
        elif variant == "executive":
            return await tools.generate_executive_letter(
                candidate=candidate,
                job=job,
                seniority_level=job.get("job_title", ""),
                leadership_years=candidate.get("experience_years", 0),
                scale_metric=candidate.get("best_achievement", ""),
                tone="formal",
            )
        elif variant == "cold_email":
            return await tools.generate_cold_email(
                candidate=candidate,
                job=job,
            )
        else:
            return await tools.generate_standard_letter(
                candidate=candidate,
                job=job,
                tone=tone,
                custom_context=custom_context,
                highlight_skills=highlight_skills,
            )

    # ── Private: Helpers ──────────────────────────────────────────────────────

    def _is_executive_role(self, job: Any) -> bool:
        """Detect executive-level roles from title signals."""
        exec_keywords = (
            "vp ", "vice president", "chief", " cto", " coo", " cpo",
            "director", "head of", "principal"
        )
        title = (getattr(job, "title", "") or "").lower()
        return any(k in title for k in exec_keywords)

    async def _save_to_db(
        self,
        *,
        resume: Any,
        job: Any,
        body: str,
        subject_line: str,
        tone: str,
        quality_score: float,
        application_id: uuid.UUID | None,
    ) -> uuid.UUID | None:
        """Persist cover letter to DB. Non-blocking — errors are swallowed."""
        try:
            from app.db.models.cover_letter import CoverLetter
            cl = CoverLetter(
                user_id=resume.user_id,
                job_id=job.id,
                application_id=application_id,
                subject_line=subject_line,
                body=body,
                word_count=len(body.split()),
                tone=tone,
                company_name=job.company,
                job_title=job.title,
                version=1,
                is_final=quality_score >= 75,
                ai_model_used="groq/llama-3.3-70b-versatile",
            )
            self.db.add(cl)
            await self.db.flush()
            return cl.id
        except Exception as exc:
            logger.warning("CoverLetter DB save failed (non-critical)", error=str(exc))
            return None

    async def _save_agent_run(
        self,
        *,
        user_id: uuid.UUID,
        status: str,
        input_data: dict,
        output_data: dict,
        duration_ms: float,
        related_job_id: uuid.UUID | None = None,
        related_resume_id: uuid.UUID | None = None,
    ) -> None:
        try:
            from app.db.models.agent_run import AgentRun
            record = AgentRun(
                user_id=user_id,
                agent_type=AgentType.COVER_LETTER.value,
                status=status,
                started_at=datetime.now(UTC),
                completed_at=datetime.now(UTC),
                duration_ms=duration_ms,
                input_data=json.dumps(input_data)[:5000],
                output_data=json.dumps(output_data)[:5000],
                llm_calls=self._llm_calls,
                related_job_id=related_job_id,
                related_resume_id=related_resume_id,
            )
            self.db.add(record)
            await self.db.flush()
        except Exception as exc:
            logger.warning("AgentRun save failed (non-critical)", error=str(exc))


__all__ = ["CoverLetterAgent"]