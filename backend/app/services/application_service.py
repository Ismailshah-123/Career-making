"""
CareerGPT — Application Service
=================================
PAGE SUMMARY:
  Orchestrates the complete job application lifecycle end-to-end.
  This is the heaviest service in the platform — triggers all AI agents.

  USED BY: app/api/v1/applications.py
  USES:    ApplicationRepository, JobRepository, ResumeRepository,
           MatchingAgent (score), CoverLetterAgent (letter),
           OutreachAgent (LinkedIn msg + recruiter email),
           ApplicationAgent (Playwright auto-apply),
           FollowupAgent (follow-up scheduling)

  CREATE APPLICATION FLOW (create_application):
    1. Check duplicate (one application per user+job)
    2. Load job + master resume
    3. MatchingAgent → match_score, gaps, keywords
    4. CoverLetterAgent → personalized cover letter
    5. OutreachAgent → LinkedIn message + recruiter email draft
    6. Save Application record with all generated content
    7. If auto_tailor=True → queue resume tailor task (Celery)
    8. If auto_apply=True  → queue Playwright apply task (Celery)
    9. Increment user.total_applications counter

  PIPELINE STAGES:
    pending → applied → viewed → interview → offer → rejected | withdrawn

  ANALYTICS:
    pipeline_stats()    → count per status for dashboard kanban
    weekly_activity()   → applications per day for charts
    match_score_dist()  → distribution of match scores
    source_breakdown()  → which job boards produce most interviews

  AUTO-APPLY TRACKING:
    Each auto-apply session stores:
      - Celery task ID (for status polling)
      - ATS platform detected (greenhouse/lever/etc.)
      - Screenshots array (audit trail)
      - Error message on failure
    Client can poll GET /applications/{id}/apply-status for real-time updates.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.constants import ApplicationStatus, FOLLOWUP_AFTER_DAYS
from app.core.exceptions import (
    ApplicationNotFoundError,
    DuplicateApplicationError,
    JobNotFoundError,
    ResumeNotFoundError,
    ValidationError,
)
from app.core.logging import log_context, logger
from app.repositories.application_repository import ApplicationRepository
from app.repositories.job_repository import JobRepository
from app.repositories.resume_repository import ResumeRepository
from app.repositories.user_repository import UserRepository
from app.services.groq_service import get_groq_service


class ApplicationService:
    """
    Complete job application business logic service.
    Routes inject this with an AsyncSession via dependency injection.
    """

    def __init__(self, db: AsyncSession) -> None:
        self.db = db
        self.app_repo = ApplicationRepository(db)
        self.job_repo = JobRepository(db)
        self.resume_repo = ResumeRepository(db)
        self.user_repo = UserRepository(db)
        self.llm = get_groq_service()

    # ── Create Application ────────────────────────────────────────────────────

    async def create_application(
        self,
        *,
        user_id: uuid.UUID,
        job_id: uuid.UUID,
        resume_id: uuid.UUID | None = None,
        auto_tailor: bool = True,
        auto_cover_letter: bool = True,
        auto_apply: bool = False,
        notes: str | None = None,
        salary_expectation: int | None = None,
    ) -> dict[str, Any]:
        """
        Create a complete job application with AI-generated content.
        This is the core orchestration method — it triggers all relevant agents.

        All AI generation is synchronous here for immediate response,
        except auto-apply (Playwright) which is always async via Celery.
        Resume tailoring is queued to Celery if auto_tailor=True.

        Returns: Full application dict with all generated content.
        """
        with log_context(user_id=str(user_id), job_id=str(job_id)):
            # ── Duplicate check ────────────────────────────────────────────────
            existing = await self.app_repo.get_by_user_and_job(user_id, job_id)
            if existing:
                logger.info("Returning existing application", app_id=str(existing.id))
                return self._app_to_dict(existing)

            # ── Load job ───────────────────────────────────────────────────────
            job = await self.job_repo.get_by_id_or_raise(job_id)

            # ── Load resume ────────────────────────────────────────────────────
            if resume_id:
                resume = await self.resume_repo.get_by_id_or_raise(resume_id)
                if resume.user_id != user_id:
                    raise ResumeNotFoundError()
            else:
                masters = await self.resume_repo.get_master_resumes(user_id)
                resume = masters[0] if masters else None

            # ── AI: Match Analysis ─────────────────────────────────────────────
            match_score = 0.0
            match_analysis: dict[str, Any] = {}
            if resume and resume.raw_text:
                match_score, match_analysis = await self._run_match_analysis(resume, job)

            # ── AI: Cover Letter ───────────────────────────────────────────────
            cover_letter_text: str | None = None
            cover_letter_subject: str | None = None
            if auto_cover_letter and resume and resume.raw_text:
                cover_letter_text, cover_letter_subject = await self._generate_cover_letter(
                    resume=resume, job=job
                )

            # ── AI: LinkedIn Outreach Message ──────────────────────────────────
            linkedin_message: str | None = None
            if resume:
                linkedin_message = await self._generate_linkedin_message(
                    resume=resume, job=job
                )

            # ── AI: Recruiter Email ────────────────────────────────────────────
            recruiter_email_draft: str | None = None
            if resume:
                recruiter_email_draft = await self._generate_recruiter_email(
                    resume=resume, job=job
                )

            # ── Save Application ───────────────────────────────────────────────
            application = await self.app_repo.create(
                user_id=user_id,
                job_id=job_id,
                resume_id=resume.id if resume else None,
                status=ApplicationStatus.PENDING.value,
                match_score=match_score,
                match_analysis=json.dumps(match_analysis) if match_analysis else None,
                cover_letter_text=cover_letter_text,
                linkedin_message=linkedin_message,
                recruiter_email_draft=recruiter_email_draft,
                notes=notes,
                salary_expectation=salary_expectation,
                auto_applied=False,
            )

            # ── Queue: Resume Tailoring (Celery) ────────────────────────────────
            if auto_tailor and resume:
                try:
                    from app.workers.resume_tasks import tailor_resume_task
                    task = tailor_resume_task.delay(
                        user_id=str(user_id),
                        job_id=str(job_id),
                        master_resume_id=str(resume.id),
                        application_id=str(application.id),
                    )
                    logger.info("Resume tailor task queued", task_id=task.id)
                except Exception as exc:
                    logger.warning("Resume tailor queue failed (non-critical)", error=str(exc))

            # ── Queue: Auto Apply (Celery) ──────────────────────────────────────
            if auto_apply and resume:
                try:
                    from app.workers.job_tasks import auto_apply_task
                    task = auto_apply_task.delay(
                        application_id=str(application.id),
                        user_id=str(user_id),
                    )
                    await self.app_repo.update(
                        application.id,
                        playwright_session_id=task.id,
                    )
                    logger.info("Auto-apply task queued", task_id=task.id)
                except Exception as exc:
                    logger.warning("Auto-apply queue failed (non-critical)", error=str(exc))

            # ── Increment user counter ──────────────────────────────────────────
            user = await self.user_repo.get_by_id(user_id)
            if user:
                await self.user_repo.update(
                    user_id,
                    total_applications=user.total_applications + 1,
                )

            logger.info(
                "Application created",
                app_id=str(application.id),
                job=f"{job.title} @ {job.company}",
                match_score=match_score,
                auto_apply=auto_apply,
            )

            # Reload with job relationship
            application = await self.app_repo.get_by_id_or_raise(application.id)
            return self._app_to_dict(application)

    # ── Status Management ─────────────────────────────────────────────────────

    async def update_status(
        self,
        *,
        application_id: uuid.UUID,
        user_id: uuid.UUID,
        new_status: str,
        notes: str | None = None,
    ) -> dict[str, Any]:
        """
        Update application pipeline status.
        Validates ownership and status transition.
        If moving to 'interview', increments user.total_interviews.
        """
        valid_statuses = {s.value for s in ApplicationStatus}
        if new_status not in valid_statuses:
            raise ValidationError(
                f"Invalid status '{new_status}'",
                context={"valid": list(valid_statuses)},
            )

        application = await self.app_repo.get_by_id_or_raise(application_id)
        if application.user_id != user_id:
            raise ApplicationNotFoundError()

        old_status = application.status
        updates: dict[str, Any] = {
            "status": new_status,
            "last_status_change_at": datetime.now(UTC),
        }
        if notes:
            updates["notes"] = notes
        if new_status == ApplicationStatus.APPLIED.value and not application.applied_at:
            updates["applied_at"] = datetime.now(UTC)

        updated = await self.app_repo.update(application_id, **updates)

        # Increment interview counter
        if new_status == ApplicationStatus.INTERVIEW.value and old_status != ApplicationStatus.INTERVIEW.value:
            user = await self.user_repo.get_by_id(user_id)
            if user:
                await self.user_repo.update(
                    user_id,
                    total_interviews=user.total_interviews + 1,
                )

        logger.info(
            "Application status updated",
            app_id=str(application_id),
            old_status=old_status,
            new_status=new_status,
        )

        return self._app_to_dict(updated)

    # ── Follow-up Management ──────────────────────────────────────────────────

    async def generate_followup(
        self,
        *,
        application_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> dict[str, Any]:
        """
        Generate a follow-up message for a stale application.
        Only available if application is 'applied' and at least FOLLOWUP_AFTER_DAYS old.
        Increments followup_count. Max 3 follow-ups per application.
        """
        application = await self.app_repo.get_by_id_or_raise(application_id)
        if application.user_id != user_id:
            raise ApplicationNotFoundError()

        if application.status not in (
            ApplicationStatus.APPLIED.value,
            ApplicationStatus.PENDING.value,
        ):
            raise ValidationError(
                f"Cannot generate follow-up for application with status '{application.status}'"
            )

        if application.followup_count >= 3:
            raise ValidationError(
                "Maximum follow-ups (3) reached for this application.",
                context={"followup_count": application.followup_count},
            )

        # Check timing
        if application.applied_at:
            days_since = (datetime.now(UTC) - application.applied_at.replace(tzinfo=UTC)).days
            if days_since < FOLLOWUP_AFTER_DAYS:
                raise ValidationError(
                    f"Too early for follow-up. Wait {FOLLOWUP_AFTER_DAYS - days_since} more days.",
                    context={"days_since": days_since, "required": FOLLOWUP_AFTER_DAYS},
                )

        job = application.job
        user = await self.user_repo.get_by_id_or_raise(user_id)

        from app.agents.followup_agent.agent import FollowupAgent
        agent = FollowupAgent(self.db)
        followup_message = await agent.generate(
            application=application,
            job=job,
            user=user,
        )

        updated = await self.app_repo.update(
            application_id,
            followup_message=followup_message,
            followup_count=application.followup_count + 1,
            last_followup_at=datetime.now(UTC),
        )

        return {
            "application_id": str(application_id),
            "followup_message": followup_message,
            "followup_count": updated.followup_count,
            "generated_at": datetime.now(UTC).isoformat(),
        }

    # ── List & Stats ──────────────────────────────────────────────────────────

    async def list_applications(
        self,
        *,
        user_id: uuid.UUID,
        status: str | None = None,
        skip: int = 0,
        limit: int = 50,
    ) -> dict[str, Any]:
        """Paginated application list with optional status filter."""
        applications = await self.app_repo.get_user_applications(
            user_id, status=status, skip=skip, limit=limit
        )
        total = await self.app_repo.count_user_applications(user_id, status=status)

        return {
            "total": total,
            "skip":  skip,
            "limit": limit,
            "items": [self._app_to_dict(a) for a in applications],
        }

    async def get_pipeline_stats(self, user_id: uuid.UUID) -> dict[str, Any]:
        """
        Return application counts per pipeline stage.
        Used for the Kanban pipeline view on dashboard.
        """
        raw = await self.app_repo.get_pipeline_stats(user_id)
        total = sum(raw.values())
        return {
            "total":        total,
            "pending":      raw.get(ApplicationStatus.PENDING.value, 0),
            "applied":      raw.get(ApplicationStatus.APPLIED.value, 0),
            "viewed":       raw.get(ApplicationStatus.VIEWED.value, 0),
            "interview":    raw.get(ApplicationStatus.INTERVIEW.value, 0),
            "offer":        raw.get(ApplicationStatus.OFFER.value, 0),
            "rejected":     raw.get(ApplicationStatus.REJECTED.value, 0),
            "withdrawn":    raw.get(ApplicationStatus.WITHDRAWN.value, 0),
        }

    async def get_analytics(self, user_id: uuid.UUID) -> dict[str, Any]:
        """
        Comprehensive application analytics for the analytics page.
        Includes: weekly activity, avg match score, source breakdown,
        response rate, interview conversion rate, avg time to response.
        """
        pipeline = await self.get_pipeline_stats(user_id)
        all_apps = await self.app_repo.get_user_applications(user_id, limit=500)

        # Response rate (applied + beyond vs total)
        responded = sum([
            pipeline.get("viewed", 0),
            pipeline.get("interview", 0),
            pipeline.get("offer", 0),
            pipeline.get("rejected", 0),
        ])
        total_applied = pipeline.get("applied", 0) + responded
        response_rate = (responded / total_applied * 100) if total_applied > 0 else 0.0

        # Interview conversion rate
        interview_rate = (
            pipeline.get("interview", 0) / total_applied * 100
            if total_applied > 0 else 0.0
        )

        # Average match score
        scored = [a for a in all_apps if a.match_score is not None]
        avg_match = (
            sum(a.match_score for a in scored) / len(scored)
            if scored else 0.0
        )

        # Weekly activity (last 4 weeks)
        weekly: dict[str, int] = {}
        now = datetime.now(UTC)
        for app in all_apps:
            if not app.created_at:
                continue
            created = app.created_at.replace(tzinfo=UTC)
            days_ago = (now - created).days
            week_key = f"week_{days_ago // 7}"
            weekly[week_key] = weekly.get(week_key, 0) + 1

        # Source breakdown (from related jobs)
        sources: dict[str, int] = {}
        for app in all_apps:
            if app.job:
                src = app.job.source or "unknown"
                sources[src] = sources.get(src, 0) + 1

        # ATS platform breakdown (auto-apply only)
        ats_platforms: dict[str, int] = {}
        for app in all_apps:
            if app.ats_platform:
                ats_platforms[app.ats_platform] = ats_platforms.get(app.ats_platform, 0) + 1

        return {
            "pipeline":             pipeline,
            "response_rate_pct":    round(response_rate, 1),
            "interview_rate_pct":   round(interview_rate, 1),
            "avg_match_score":      round(avg_match, 3),
            "weekly_activity":      weekly,
            "source_breakdown":     sources,
            "ats_platform_breakdown": ats_platforms,
            "total_auto_applied":   sum(1 for a in all_apps if a.auto_applied),
            "total_manual_applied": sum(1 for a in all_apps if not a.auto_applied),
        }

    async def get_auto_apply_status(
        self,
        *,
        application_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> dict[str, Any]:
        """
        Poll the status of a Celery auto-apply task.
        Returns task state and any screenshots/errors.
        """
        application = await self.app_repo.get_by_id_or_raise(application_id)
        if application.user_id != user_id:
            raise ApplicationNotFoundError()

        task_status = "unknown"
        if application.playwright_session_id:
            try:
                from app.workers.celery_app import celery_app
                task = celery_app.AsyncResult(application.playwright_session_id)
                task_status = task.state
            except Exception:
                task_status = "unavailable"

        return {
            "application_id":   str(application_id),
            "auto_applied":     application.auto_applied,
            "status":           application.status,
            "task_id":          application.playwright_session_id,
            "task_status":      task_status,
            "ats_platform":     application.ats_platform,
            "screenshots":      json.loads(application.screenshots) if application.screenshots else [],
            "error":            application.application_error,
            "applied_at":       application.applied_at.isoformat() if application.applied_at else None,
        }

    # ── Private: AI Generation ────────────────────────────────────────────────

    async def _run_match_analysis(
        self,
        resume: Any,
        job: Any,
    ) -> tuple[float, dict[str, Any]]:
        """Run MatchingAgent to score resume vs job."""
        try:
            from app.agents.matching_agent.agent import MatchingAgent
            agent = MatchingAgent(self.db)
            result = await agent.analyze(resume=resume, job=job)
            score = float(result.get("score", 0.5))
            return score, result
        except Exception as exc:
            logger.warning("Match analysis failed (non-critical)", error=str(exc))
            return 0.5, {}

    async def _generate_cover_letter(
        self,
        resume: Any,
        job: Any,
    ) -> tuple[str | None, str | None]:
        """Generate personalized cover letter via CoverLetterAgent."""
        try:
            from app.agents.cover_letter_agent.agent import CoverLetterAgent
            agent = CoverLetterAgent(self.db)
            result = await agent.generate(resume=resume, job=job)
            return result.get("body"), result.get("subject_line")
        except Exception as exc:
            logger.warning("Cover letter generation failed (non-critical)", error=str(exc))
            return None, None

    async def _generate_linkedin_message(
        self,
        resume: Any,
        job: Any,
    ) -> str | None:
        """Generate LinkedIn outreach message via OutreachAgent."""
        try:
            from app.agents.outreach_agent.agent import OutreachAgent
            agent = OutreachAgent(self.db)
            result = await agent.generate_linkedin_message(resume=resume, job=job)
            return result.get("message")
        except Exception as exc:
            logger.warning("LinkedIn message generation failed (non-critical)", error=str(exc))
            return None

    async def _generate_recruiter_email(
        self,
        resume: Any,
        job: Any,
    ) -> str | None:
        """Generate recruiter cold email via OutreachAgent."""
        try:
            from app.agents.outreach_agent.agent import OutreachAgent
            agent = OutreachAgent(self.db)
            result = await agent.generate_recruiter_email(resume=resume, job=job)
            return result.get("body")
        except Exception as exc:
            logger.warning("Recruiter email generation failed (non-critical)", error=str(exc))
            return None

    # ── Serialization ─────────────────────────────────────────────────────────

    @staticmethod
    def _app_to_dict(app: Any) -> dict[str, Any]:
        """Safe serialization of Application ORM object."""
        def _parse(val: str | None) -> Any:
            if not val:
                return None
            try:
                return json.loads(val) if isinstance(val, str) else val
            except Exception:
                return val

        job_dict: dict[str, Any] | None = None
        if app.job:
            j = app.job
            job_dict = {
                "id":              str(j.id),
                "title":           j.title,
                "company":         j.company,
                "location":        j.location,
                "is_remote":       j.is_remote,
                "job_url":         j.job_url,
                "source":          j.source,
                "employment_type": j.employment_type,
                "salary_min":      j.salary_min,
                "salary_max":      j.salary_max,
                "salary_currency": j.salary_currency,
                "company_logo_url": j.company_logo_url,
                "posted_at":       j.posted_at.isoformat() if j.posted_at else None,
            }

        return {
            "id":                   str(app.id),
            "user_id":              str(app.user_id),
            "job_id":               str(app.job_id),
            "resume_id":            str(app.resume_id) if app.resume_id else None,
            "status":               app.status,
            "match_score":          app.match_score,
            "match_analysis":       _parse(app.match_analysis),
            "cover_letter_text":    app.cover_letter_text,
            "linkedin_message":     app.linkedin_message,
            "recruiter_email_draft": app.recruiter_email_draft,
            "followup_message":     app.followup_message,
            "followup_count":       app.followup_count,
            "tailored_resume_path": app.tailored_resume_path,
            "auto_applied":         app.auto_applied,
            "ats_platform":         app.ats_platform,
            "application_error":    app.application_error,
            "notes":                app.notes,
            "salary_expectation":   app.salary_expectation,
            "applied_at":           app.applied_at.isoformat() if app.applied_at else None,
            "last_status_change_at": app.last_status_change_at.isoformat() if app.last_status_change_at else None,
            "last_followup_at":     app.last_followup_at.isoformat() if app.last_followup_at else None,
            "created_at":           app.created_at.isoformat() if app.created_at else None,
            "updated_at":           app.updated_at.isoformat() if app.updated_at else None,
            "job":                  job_dict,
        }


__all__ = ["ApplicationService"]