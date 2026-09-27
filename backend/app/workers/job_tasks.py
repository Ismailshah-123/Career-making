"""
app/workers/job_tasks.py
==========================
Celery task definitions for all job-pipeline operations.

Tasks defined here:
    discover_jobs_task                — scrape boards, embed, match to resume
    discover_jobs_for_all_users_task  — Beat-scheduled fan-out across all active users
    match_jobs_task                   — run Qdrant similarity matching for a user
    embed_job_task                    — embed a single job into Qdrant
    submit_application_task           — run apply_workflow for one application
    send_followup_task                — send a follow-up message via outreach_agent
    check_due_followups_task          — Beat-scheduled: find & fire overdue follow-ups
    send_outreach_task                — dispatch outreach_agent for one application
    discover_recruiters_task          — run outreach_agent recruiter discovery
    update_agent_run_status_task      — utility: update AgentRun status from any task

All tasks are async-capable via asyncio.run() — Celery workers are sync
by default, so we wrap every async workflow entrypoint with asyncio.run().
Each task updates its corresponding AgentRun row at start, on success,
and on failure so the dashboard always reflects current state.

Retry policy:
    - Transient failures (network, DB connection): 3 retries, exponential backoff
    - AI service rate limits: 5 retries, 60s fixed delay
    - Browser automation failures: 2 retries, 30s delay
    - Permanent failures (job gone, URL 404): no retry, immediate failure
"""

from __future__ import annotations

import asyncio
import traceback
from datetime import datetime, timezone
from typing import Any

from celery import Task
from celery.exceptions import MaxRetriesExceededError, SoftTimeLimitExceeded

from app.workers.celery_app import celery_app, InstrumentedTask
from app.core.constants import AgentRunStatus, AGENT_DISCOVERY, AGENT_APPLICATION, AGENT_FOLLOWUP, AGENT_OUTREACH
from app.core.logging import get_logger, get_task_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Shared async helpers
# ---------------------------------------------------------------------------

def _run(coro: Any) -> Any:
    """Run an async coroutine in a new event loop inside a sync Celery task."""
    return asyncio.run(coro)


async def _update_agent_run(
    agent_run_id: str,
    *,
    status: str,
    output_payload: dict | None = None,
    error_message: str | None = None,
    error_type: str | None = None,
    error_traceback: str | None = None,
    duration_ms: int | None = None,
) -> None:
    """Persist AgentRun status update to the database."""
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
            logger.warning("AgentRun not found for update", agent_run_id=agent_run_id)
            return

        run.status = status
        if status == AgentRunStatus.RUNNING and not run.started_at:
            run.started_at = datetime.now(timezone.utc)
        if status in (AgentRunStatus.COMPLETED, AgentRunStatus.FAILED, AgentRunStatus.CANCELLED):
            run.completed_at = datetime.now(timezone.utc)
        if output_payload:
            run.output_payload = output_payload
        if error_message:
            run.error_message = error_message
        if error_type:
            run.error_type = error_type
        if error_traceback:
            run.error_traceback = error_traceback[:10_000]
        if duration_ms:
            run.duration_ms = duration_ms
        await db.commit()


# ---------------------------------------------------------------------------
# discover_jobs_task
# ---------------------------------------------------------------------------

@celery_app.task(
    base=InstrumentedTask,
    name="app.workers.job_tasks.discover_jobs_task",
    bind=True,
    max_retries=3,
    default_retry_delay=120,
    soft_time_limit=1800,     # 30-minute soft limit
    time_limit=2100,          # 35-minute hard limit
    acks_late=True,
)
def discover_jobs_task(
    self: Task,
    *,
    user_id: str,
    agent_run_id: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    """
    Run the job discovery workflow for one user.

    Scrapes configured boards, deduplicates, embeds new jobs into Qdrant,
    and optionally matches them against the user's active resume.

    config keys:
        job_boards          : list[str]  — boards to scrape
        keywords            : list[str]  — search keywords
        locations           : list[str]  — target locations
        work_modes          : list[str]  — remote | hybrid | onsite
        max_results_per_board: int
        match_to_resume_id  : str | None — resume UUID for post-discovery matching
    """
    task_logger = get_task_logger("discover_jobs_task", task_id=self.request.id)
    task_logger.info("Discovery task started", user_id=user_id, boards=config.get("job_boards"))

    start_ts = datetime.now(timezone.utc)

    try:
        _run(_update_agent_run(agent_run_id, status=AgentRunStatus.RUNNING))

        from app.workflows.job_workflow import run_discovery_workflow

        final_state = _run(
            run_discovery_workflow(
                user_id=user_id,
                agent_run_id=agent_run_id,
                job_boards=config.get("job_boards", ["linkedin", "indeed", "remoteok"]),
                keywords=config.get("keywords", []),
                locations=config.get("locations", []),
                work_modes=config.get("work_modes", ["remote"]),
                max_results_per_board=config.get("max_results_per_board", 50),
                match_to_resume_id=config.get("match_to_resume_id"),
            )
        )

        duration_ms = int((datetime.now(timezone.utc) - start_ts).total_seconds() * 1000)
        discovered  = len(final_state.get("discovered_job_ids", []))
        matched     = len(final_state.get("top_match_job_ids", []))
        errors      = final_state.get("errors", [])
        status      = AgentRunStatus.FAILED if errors else AgentRunStatus.COMPLETED

        _run(_update_agent_run(
            agent_run_id,
            status=status,
            output_payload={
                "discovered_jobs": discovered,
                "matched_jobs": matched,
                "boards_scraped": config.get("job_boards", []),
                "errors": errors[-3:] if errors else [],
            },
            duration_ms=duration_ms,
        ))

        task_logger.info(
            "Discovery task complete",
            discovered=discovered,
            matched=matched,
            duration_ms=duration_ms,
            status=status,
        )
        return {"discovered": discovered, "matched": matched, "status": status}

    except SoftTimeLimitExceeded:
        task_logger.error("Discovery task soft time limit exceeded")
        _run(_update_agent_run(
            agent_run_id,
            status=AgentRunStatus.TIMED_OUT,
            error_message="Task exceeded 30-minute soft time limit.",
            error_type="SoftTimeLimitExceeded",
        ))
        return {"status": "timed_out"}

    except Exception as exc:
        tb = traceback.format_exc()
        task_logger.error("Discovery task failed", error=str(exc)[:500], exc_info=True)

        try:
            countdown = min(120 * (2 ** self.request.retries), 600)
            raise self.retry(exc=exc, countdown=countdown)
        except MaxRetriesExceededError:
            _run(_update_agent_run(
                agent_run_id,
                status=AgentRunStatus.FAILED,
                error_message=str(exc)[:500],
                error_type=type(exc).__name__,
                error_traceback=tb,
            ))
            return {"status": "failed", "error": str(exc)[:200]}


# ---------------------------------------------------------------------------
# discover_jobs_for_all_users_task (Beat-scheduled fan-out)
# ---------------------------------------------------------------------------

@celery_app.task(
    name="app.workers.job_tasks.discover_jobs_for_all_users_task",
    soft_time_limit=300,
    time_limit=360,
)
def discover_jobs_for_all_users_task() -> dict[str, Any]:
    """
    Beat-scheduled task: find all active users with auto-discovery enabled
    and enqueue a discover_jobs_task for each.

    Runs every 6 hours (configured in celery_app.py beat_schedule).
    Uses staggered countdowns (5–60s) to spread the load rather than
    hitting all boards simultaneously.
    """
    task_logger = get_task_logger("discover_jobs_for_all_users")
    task_logger.info("Starting discovery fan-out for all active users")

    async def _get_eligible_users() -> list[dict[str, Any]]:
        from app.db.session import get_db_context
        from app.db.models.user import User
        from app.db.models.agent_run import AgentRun
        from sqlalchemy import select, func

        async with get_db_context() as db:
            # Load active users with auto-discover preference enabled
            result = await db.execute(
                select(User).where(
                    User.is_active.is_(True),
                    User.is_deleted.is_(False),
                    User.job_search_preferences["auto_apply_enabled"].astext == "true",
                )
            )
            users = result.scalars().all()

            eligible = []
            for user in users:
                prefs = user.job_search_preferences or {}
                eligible.append({
                    "user_id": str(user.id),
                    "job_boards": prefs.get("preferred_boards", ["linkedin", "remoteok"]),
                    "keywords":   prefs.get("desired_roles", []),
                    "locations":  prefs.get("locations", []),
                    "work_modes": prefs.get("work_mode", ["remote"]),
                })
            return eligible

    try:
        eligible_users = _run(_get_eligible_users())
        task_logger.info(f"Enqueueing discovery for {len(eligible_users)} users")

        dispatched = 0
        for i, user_config in enumerate(eligible_users):
            countdown_seconds = min(i * 5, 300)  # stagger up to 5 minutes

            # Create an AgentRun placeholder synchronously
            agent_run_id = _run(_create_agent_run_placeholder(
                user_id=user_config["user_id"],
                agent_name=AGENT_DISCOVERY,
            ))

            discover_jobs_task.apply_async(
                kwargs={
                    "user_id":       user_config["user_id"],
                    "agent_run_id":  agent_run_id,
                    "config":        user_config,
                },
                countdown=countdown_seconds,
            )
            dispatched += 1

        task_logger.info(f"Discovery fan-out complete: {dispatched} tasks enqueued")
        return {"dispatched": dispatched}

    except Exception as exc:
        task_logger.error("Discovery fan-out failed", error=str(exc)[:300])
        return {"status": "failed", "error": str(exc)[:200]}


# ---------------------------------------------------------------------------
# scrape_source_task
# ---------------------------------------------------------------------------

_SCRAPER_CLASSES: dict[str, str] = {
    "linkedin":  "app.automation.scraping.linkedin_scraper.LinkedInScraper",
    "indeed":    "app.automation.scraping.indeed_scraper.IndeedScraper",
    "remoteok":  "app.automation.scraping.remoteok_scraper.RemoteOKScraper",
    "wellfound": "app.automation.scraping.wellfound_scraper.WellfoundScraper",
}


def _load_scraper(source: str):
    import importlib

    path = _SCRAPER_CLASSES.get(source)
    if not path:
        raise ValueError(f"No scraper registered for source '{source}'")
    module_path, class_name = path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)()


@celery_app.task(
    name="app.workers.job_tasks.scrape_source_task",
    bind=True,
    max_retries=2,
    default_retry_delay=90,
    soft_time_limit=600,
    time_limit=720,
    acks_late=True,
)
def scrape_source_task(
    self: Task,
    *,
    source: str,
    keywords: list[str],
    locations: list[str],
    max_results: int = 50,
    work_modes: list[str] | None = None,
) -> dict[str, Any]:
    """
    Scrape ONE job board for a set of keywords/locations, independent of
    any specific user — populates the shared job pool that
    discovery_agent's matching step later searches against.

    Upserts each result into the Job table (dedup by source+external_id)
    and queues embed_job_task for every newly-created job.
    """
    task_logger = get_task_logger("scrape_source_task", task_id=self.request.id)
    task_logger.info("Source scrape started", source=source, keywords=keywords)

    async def _scrape() -> dict[str, Any]:
        from app.db.session import get_db_context
        from app.repositories.job_repository import JobRepository

        scraper = _load_scraper(source)
        raw_jobs = await scraper.search(
            keywords=keywords,
            locations=locations,
            work_modes=work_modes or ["remote"],
            max_results=max_results,
        )

        created_ids: list[str] = []
        async with get_db_context() as db:
            repo = JobRepository(db)
            for raw in raw_jobs:
                external_id = raw.get("external_id") or raw.get("source_url", "")
                if not external_id:
                    continue
                defaults = {k: v for k, v in raw.items() if k not in ("external_id",)}
                defaults.setdefault("job_board", source)
                job, created = await repo.upsert_job(source, external_id, defaults)
                if created:
                    created_ids.append(str(job.id))

        return {"found": len(raw_jobs), "created": len(created_ids), "job_ids": created_ids}

    try:
        result = _run(_scrape())
        for job_id in result["job_ids"]:
            embed_job_task.apply_async(kwargs={"job_id": job_id}, priority=5)

        task_logger.info(
            "Source scrape complete", source=source,
            found=result["found"], created=result["created"],
        )
        return {"status": "completed", **result}

    except SoftTimeLimitExceeded:
        task_logger.error("Source scrape timed out", source=source)
        return {"status": "timed_out", "source": source}

    except Exception as exc:
        task_logger.error("Source scrape failed", source=source, error=str(exc)[:300])
        try:
            raise self.retry(exc=exc, countdown=90 * (self.request.retries + 1))
        except MaxRetriesExceededError:
            return {"status": "failed", "source": source, "error": str(exc)[:200]}


# ---------------------------------------------------------------------------
# embed_job_task
# ---------------------------------------------------------------------------

@celery_app.task(
    name="app.workers.job_tasks.embed_job_task",
    bind=True,
    max_retries=3,
    default_retry_delay=30,
    soft_time_limit=120,
    time_limit=180,
)
def embed_job_task(self: Task, *, job_id: str) -> dict[str, Any]:
    """
    Embed a single job's description into the Qdrant jobs collection.

    Called after a new job is scraped and saved to the DB — the embedding
    step is deferred to a separate task because:
    1. It's I/O-bound (OpenAI API call) — shouldn't block the scraper
    2. It can fail independently and be retried without re-scraping
    3. Batch embedding is more efficient — this task can be grouped
       via celery.group() when embedding many new jobs at once.
    """
    task_logger = get_task_logger("embed_job_task", task_id=self.request.id)

    async def _embed() -> dict:
        from app.services.embedding_service import EmbeddingService
        from app.services.qdrant_service import QdrantService
        from app.db.session import get_db_context
        from app.db.models.job import Job
        from sqlalchemy import select
        import uuid

        async with get_db_context() as db:
            result = await db.execute(select(Job).where(Job.id == uuid.UUID(job_id)))
            job = result.scalar_one_or_none()
            if not job:
                return {"status": "skipped", "reason": "Job not found"}
            if job.is_embedded:
                return {"status": "skipped", "reason": "Already embedded"}

            text = job.description_cleaned or job.description or job.title
            if not text or len(text.strip()) < 20:
                return {"status": "skipped", "reason": "Description too short to embed"}

            emb_svc = EmbeddingService()
            qdrant  = QdrantService()

            vector = await emb_svc.embed_text(text[:8000])
            point_id = await qdrant.upsert_job(
                job_id=job_id,
                vector=vector,
                payload={
                    "job_id":        job_id,
                    "title":         job.title,
                    "company":       job.company_name,
                    "work_mode":     job.work_mode,
                    "job_type":      job.job_type,
                    "job_board":     job.job_board,
                    "required_skills": job.required_skills or [],
                },
            )
            job.is_embedded     = True
            job.qdrant_point_id = point_id
            await db.commit()

        return {"status": "embedded", "job_id": job_id, "point_id": point_id}

    try:
        result = _run(_embed())
        task_logger.info("Job embedded", job_id=job_id, status=result.get("status"))
        return result
    except Exception as exc:
        task_logger.error("Job embed failed", job_id=job_id, error=str(exc)[:300])
        try:
            raise self.retry(exc=exc, countdown=30 * (2 ** self.request.retries))
        except MaxRetriesExceededError:
            return {"status": "failed", "error": str(exc)[:200]}


# ---------------------------------------------------------------------------
# submit_application_task
# ---------------------------------------------------------------------------

@celery_app.task(
    base=InstrumentedTask,
    name="app.workers.job_tasks.submit_application_task",
    bind=True,
    max_retries=2,
    default_retry_delay=60,
    soft_time_limit=600,      # 10-minute soft limit — browser automation is slow
    time_limit=720,           # 12-minute hard limit
    acks_late=True,
)
def submit_application_task(
    self: Task,
    *,
    application_id: str,
    agent_run_id: str,
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Execute Playwright form automation for one job application.

    Delegates to the apply_workflow.run_apply_workflow() LangGraph graph,
    which handles:
    - Context loading (application, job, resume, cover letter from DB)
    - ATS provider detection
    - Form field detection and filling
    - Submission and confirmation capture
    - DB update (status → APPLIED)
    - Outreach and follow-up scheduling

    Returns a result dict that's stored in the Celery result backend
    for 24 hours (queryable via task_id from the API response).
    """
    task_logger = get_task_logger("submit_application_task", task_id=self.request.id)
    task_logger.info("Application submission task started", application_id=application_id)

    start_ts = datetime.now(timezone.utc)

    try:
        _run(_update_agent_run(agent_run_id, status=AgentRunStatus.RUNNING))

        # Load application + job context for the run
        context = _run(_load_application_context(application_id))
        if not context:
            _run(_update_agent_run(
                agent_run_id, status=AgentRunStatus.FAILED,
                error_message=f"Application {application_id} not found or not eligible."
            ))
            return {"status": "failed", "reason": "application_not_eligible"}

        from app.workflows.apply_workflow import run_apply_workflow

        final_state = _run(
            run_apply_workflow(
                user_id=context["user_id"],
                agent_run_id=agent_run_id,
                application_id=application_id,
                job_id=context["job_id"],
                base_resume_id=context["base_resume_id"],
                apply_url=context.get("apply_url"),
                ats_provider=context.get("ats_provider"),
            )
        )

        duration_ms = int((datetime.now(timezone.utc) - start_ts).total_seconds() * 1000)
        success     = final_state.get("submission_success", False)
        manual      = final_state.get("requires_manual_review", False)
        errors      = final_state.get("errors", [])

        status = AgentRunStatus.COMPLETED if (success or manual) else AgentRunStatus.FAILED

        _run(_update_agent_run(
            agent_run_id,
            status=status,
            output_payload={
                "application_id":       application_id,
                "submission_success":   success,
                "requires_manual_review": manual,
                "ats_provider":         final_state.get("ats_provider"),
                "fields_filled":        len(final_state.get("form_fields_filled", [])),
                "confirmation_snippet": (final_state.get("submission_confirmation_text") or "")[:200],
                "errors":               errors[-2:],
            },
            duration_ms=duration_ms,
        ))

        task_logger.info(
            "Application submission task complete",
            application_id=application_id,
            success=success,
            manual_review=manual,
            duration_ms=duration_ms,
        )
        return {"success": success, "manual_review": manual, "status": status}

    except SoftTimeLimitExceeded:
        task_logger.error("Application task soft time limit exceeded", application_id=application_id)
        _run(_update_agent_run(
            agent_run_id, status=AgentRunStatus.TIMED_OUT,
            error_message="Playwright automation exceeded 10-minute limit.",
            error_type="SoftTimeLimitExceeded",
        ))
        return {"status": "timed_out"}

    except Exception as exc:
        tb = traceback.format_exc()
        task_logger.error("Application task failed", application_id=application_id, error=str(exc)[:500])

        is_retryable = not any(
            kw in str(exc).lower()
            for kw in ["not found", "already applied", "no apply_url", "terminal status"]
        )
        if is_retryable:
            try:
                raise self.retry(exc=exc, countdown=60 * (self.request.retries + 1))
            except MaxRetriesExceededError:
                pass

        _run(_update_agent_run(
            agent_run_id,
            status=AgentRunStatus.FAILED,
            error_message=str(exc)[:500],
            error_type=type(exc).__name__,
            error_traceback=tb,
        ))
        return {"status": "failed", "error": str(exc)[:200]}


# ---------------------------------------------------------------------------
# auto_apply_task
# ---------------------------------------------------------------------------

@celery_app.task(
    name="app.workers.job_tasks.auto_apply_task",
    bind=True,
    max_retries=1,
    soft_time_limit=630,
    time_limit=750,
    acks_late=True,
)
def auto_apply_task(self: Task, *, application_id: str, user_id: str) -> dict[str, Any]:
    """
    Convenience entry point used right after an Application row is
    created with auto_apply=True: creates the AgentRun bookkeeping row
    then delegates to submit_application_task for the actual Playwright
    automation.
    """
    task_logger = get_task_logger("auto_apply_task", task_id=self.request.id)
    try:
        agent_run_id = _run(_create_agent_run_placeholder(user_id, AGENT_APPLICATION))
        return submit_application_task(application_id=application_id, agent_run_id=agent_run_id)
    except Exception as exc:
        task_logger.error("Auto-apply setup failed", application_id=application_id, error=str(exc)[:300])
        return {"status": "failed", "error": str(exc)[:200]}


# ---------------------------------------------------------------------------
# send_followup_task
# ---------------------------------------------------------------------------

@celery_app.task(
    base=InstrumentedTask,
    name="app.workers.job_tasks.send_followup_task",
    bind=True,
    max_retries=2,
    default_retry_delay=300,
    soft_time_limit=180,
    time_limit=240,
)
def send_followup_task(
    self: Task,
    application_id: str,
    agent_run_id: str,
) -> dict[str, Any]:
    """
    Send a follow-up message for an application via the followup_agent.

    ETA is set at application time + FOLLOWUP_WAIT_DAYS (7 days), so this
    task fires automatically after the wait period elapses.

    The followup_agent:
    1. Checks the application is still in APPLIED / ACKNOWLEDGED status
       (skip if interview scheduled or rejected — no follow-up needed)
    2. Looks up the recruiter contact (if discovered by outreach_agent)
    3. Searches for recent company news to personalise the message
    4. Writes a brief, professional follow-up referencing the original apply date
    5. Sends via LinkedIn message or email
    6. Updates followup_count, last_followup_at, and next_followup_at on the Application
    7. Respects MAX_FOLLOWUPS_PER_APPLICATION (2) hard limit
    """
    task_logger = get_task_logger("send_followup_task", task_id=self.request.id)
    task_logger.info("Follow-up task started", application_id=application_id)

    async def _run_followup() -> dict:
        from app.db.session import get_db_context
        from app.db.models.application import Application
        from app.agents.followup_agent.agent import FollowupAgent
        from app.core.constants import (
            ApplicationStatus, TERMINAL_STATUSES, MAX_FOLLOWUPS_PER_APPLICATION
        )
        from sqlalchemy import select
        import uuid

        async with get_db_context() as db:
            result = await db.execute(
                select(Application).where(
                    Application.id == uuid.UUID(application_id),
                    Application.is_deleted.is_(False),
                )
            )
            application = result.scalar_one_or_none()
            if not application:
                return {"status": "skipped", "reason": "application_not_found"}

            current_status = ApplicationStatus(application.status)
            if current_status in TERMINAL_STATUSES:
                return {"status": "skipped", "reason": f"terminal_status:{application.status}"}
            if current_status in (ApplicationStatus.INTERVIEW_SCHEDULED, ApplicationStatus.INTERVIEWED):
                return {"status": "skipped", "reason": "interview_in_progress"}
            if application.followup_count >= MAX_FOLLOWUPS_PER_APPLICATION:
                return {"status": "skipped", "reason": "max_followups_reached"}
            if not application.applied_at:
                return {"status": "skipped", "reason": "not_yet_applied"}

            agent = FollowupAgent()
            result_data = await agent.send_followup(
                application=application,
                db=db,
            )
            await db.commit()
            return result_data

    try:
        _run(_update_agent_run(agent_run_id, status=AgentRunStatus.RUNNING))
        result = _run(_run_followup())

        status = AgentRunStatus.COMPLETED if result.get("sent") else AgentRunStatus.COMPLETED
        _run(_update_agent_run(
            agent_run_id,
            status=status,
            output_payload=result,
        ))
        task_logger.info("Follow-up task complete", application_id=application_id, result=result)
        return result

    except Exception as exc:
        task_logger.error("Follow-up task failed", application_id=application_id, error=str(exc)[:300])
        try:
            raise self.retry(exc=exc, countdown=300)
        except MaxRetriesExceededError:
            _run(_update_agent_run(
                agent_run_id,
                status=AgentRunStatus.FAILED,
                error_message=str(exc)[:500],
                error_type=type(exc).__name__,
            ))
            return {"status": "failed"}


# ---------------------------------------------------------------------------
# check_due_followups_task (Beat-scheduled)
# ---------------------------------------------------------------------------

@celery_app.task(
    name="app.workers.job_tasks.check_due_followups_task",
    soft_time_limit=120,
    time_limit=180,
)
def check_due_followups_task() -> dict[str, Any]:
    """
    Beat-scheduled task: scan for applications where next_followup_at has
    passed and no follow-up has been sent yet, then enqueue send_followup_task.

    Runs every 2 hours (see celery_app.py beat_schedule).
    Uses SELECT FOR UPDATE SKIP LOCKED for safe concurrent worker execution.
    """
    task_logger = get_task_logger("check_due_followups")

    async def _find_and_dispatch() -> int:
        from app.db.session import get_db_context
        from app.db.models.application import Application
        from app.db.models.agent_run import AgentRun
        from app.core.constants import ApplicationStatus, TERMINAL_STATUSES
        from sqlalchemy import select
        from sqlalchemy import update

        now = datetime.now(timezone.utc)

        async with get_db_context() as db:
            result = await db.execute(
                select(Application)
                .where(
                    Application.is_deleted.is_(False),
                    Application.next_followup_at <= now,
                    Application.applied_at.isnot(None),
                    ~Application.status.in_([s.value for s in TERMINAL_STATUSES]),
                    Application.status.notin_([
                        ApplicationStatus.INTERVIEW_SCHEDULED.value,
                        ApplicationStatus.INTERVIEWED.value,
                    ]),
                )
                .limit(50)
                .with_for_update(skip_locked=True)
            )
            due_applications = result.scalars().all()

            dispatched = 0
            for application in due_applications:
                # Create AgentRun placeholder
                run = AgentRun(
                    user_id=application.user_id,
                    agent_name=AGENT_FOLLOWUP,
                    trigger="beat_scheduler",
                    status=AgentRunStatus.PENDING,
                    input_payload={"application_id": str(application.id)},
                )
                db.add(run)
                await db.flush()

                send_followup_task.apply_async(
                    args=[str(application.id), str(run.id)],
                )

                # Null out next_followup_at to prevent double-dispatch
                application.next_followup_at = None
                dispatched += 1

            await db.commit()
            return dispatched

    try:
        dispatched = _run(_find_and_dispatch())
        task_logger.info(f"Dispatched {dispatched} follow-up tasks")
        return {"dispatched": dispatched}
    except Exception as exc:
        task_logger.error("Follow-up check failed", error=str(exc)[:300])
        return {"status": "failed"}


# ---------------------------------------------------------------------------
# send_outreach_task
# ---------------------------------------------------------------------------

@celery_app.task(
    base=InstrumentedTask,
    name="app.workers.job_tasks.send_outreach_task",
    bind=True,
    max_retries=2,
    default_retry_delay=120,
    soft_time_limit=300,
    time_limit=360,
)
def send_outreach_task(
    self: Task,
    *,
    application_id: str,
    agent_run_id: str,
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Send a personalised LinkedIn outreach message to a recruiter at the
    hiring company, driven by the outreach_agent.
    """
    task_logger = get_task_logger("send_outreach_task", task_id=self.request.id)
    task_logger.info("Outreach task started", application_id=application_id)

    async def _run_outreach() -> dict:
        from app.agents.outreach_agent.agent import OutreachAgent
        from app.db.session import get_db_context
        from app.db.models.application import Application
        from sqlalchemy import select
        import uuid

        opts = options or {}
        async with get_db_context() as db:
            result = await db.execute(
                select(Application).where(Application.id == uuid.UUID(application_id))
            )
            application = result.scalar_one_or_none()
            if not application:
                return {"status": "skipped", "reason": "application_not_found"}
            if application.outreach_sent:
                return {"status": "skipped", "reason": "outreach_already_sent"}

            agent = OutreachAgent()
            outreach_result = await agent.send_outreach(
                user_id=str(application.user_id),
                recruiter_id=opts.get("recruiter_id"),
                application_id=application_id,
                job_id=str(application.job_id),
                channel=opts.get("channel", "linkedin_message"),
                custom_message=opts.get("custom_message"),
            )

            application.outreach_sent    = outreach_result.get("sent", False)
            application.outreach_sent_at = datetime.now(timezone.utc) if outreach_result.get("sent") else None
            application.outreach_message = outreach_result.get("message_sent", "")[:2000]
            await db.commit()
            return outreach_result

    try:
        _run(_update_agent_run(agent_run_id, status=AgentRunStatus.RUNNING))
        result = _run(_run_outreach())
        _run(_update_agent_run(agent_run_id, status=AgentRunStatus.COMPLETED, output_payload=result))
        task_logger.info("Outreach task complete", application_id=application_id)
        return result
    except Exception as exc:
        task_logger.error("Outreach task failed", error=str(exc)[:300])
        try:
            raise self.retry(exc=exc, countdown=120)
        except MaxRetriesExceededError:
            _run(_update_agent_run(
                agent_run_id, status=AgentRunStatus.FAILED,
                error_message=str(exc)[:500], error_type=type(exc).__name__,
            ))
            return {"status": "failed"}


# ---------------------------------------------------------------------------
# discover_recruiters_task
# ---------------------------------------------------------------------------

@celery_app.task(
    base=InstrumentedTask,
    name="app.workers.job_tasks.discover_recruiters_task",
    bind=True,
    max_retries=2,
    default_retry_delay=180,
    soft_time_limit=300,
    time_limit=360,
)
def discover_recruiters_task(
    self: Task,
    *,
    user_id: str,
    agent_run_id: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    """
    Use the outreach_agent to discover recruiters at a target company
    via LinkedIn search and save them to the Recruiter table.
    """
    task_logger = get_task_logger("discover_recruiters_task", task_id=self.request.id)

    async def _discover() -> dict:
        from app.agents.outreach_agent.agent import OutreachAgent
        agent = OutreachAgent()
        return await agent.discover_recruiters(
            company_id=config.get("company_id"),
            company_name=config.get("company_name"),
            job_title_keywords=config.get("job_title_keywords", ["recruiter", "talent"]),
            max_results=config.get("max_results", 10),
            user_id=user_id,
        )

    try:
        _run(_update_agent_run(agent_run_id, status=AgentRunStatus.RUNNING))
        result = _run(_discover())
        _run(_update_agent_run(
            agent_run_id, status=AgentRunStatus.COMPLETED, output_payload=result
        ))
        task_logger.info("Recruiter discovery complete", found=result.get("found", 0))
        return result
    except Exception as exc:
        task_logger.error("Recruiter discovery failed", error=str(exc)[:300])
        try:
            raise self.retry(exc=exc, countdown=180)
        except MaxRetriesExceededError:
            _run(_update_agent_run(
                agent_run_id, status=AgentRunStatus.FAILED,
                error_message=str(exc)[:500], error_type=type(exc).__name__,
            ))
            return {"status": "failed"}


# ---------------------------------------------------------------------------
# update_agent_run_status_task (utility)
# ---------------------------------------------------------------------------

@celery_app.task(
    name="app.workers.job_tasks.update_agent_run_status_task",
    soft_time_limit=30,
    time_limit=45,
)
def update_agent_run_status_task(
    agent_run_id: str,
    status: str,
    output_payload: dict | None = None,
) -> None:
    """Utility task: update an AgentRun status from any other task or workflow."""
    _run(_update_agent_run(agent_run_id, status=status, output_payload=output_payload))


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

async def _load_application_context(application_id: str) -> dict[str, Any] | None:
    """Load minimal application context for submit_application_task."""
    from app.db.session import get_db_context
    from app.db.models.application import Application
    from app.db.models.job import Job
    from sqlalchemy import select
    import uuid

    async with get_db_context() as db:
        result = await db.execute(
            select(Application).where(
                Application.id == uuid.UUID(application_id),
                Application.is_deleted.is_(False),
            )
        )
        application = result.scalar_one_or_none()
        if not application:
            return None

        job_result = await db.execute(select(Job).where(Job.id == application.job_id))
        job = job_result.scalar_one_or_none()
        if not job:
            return None

        return {
            "user_id":       str(application.user_id),
            "job_id":        str(application.job_id),
            "base_resume_id": str(application.resume_id) if application.resume_id else "",
            "apply_url":     job.apply_url,
            "ats_provider":  job.ats_provider,
        }


async def _create_agent_run_placeholder(user_id: str, agent_name: str) -> str:
    """Create a pending AgentRun row and return its UUID string."""
    from app.db.session import get_db_context
    from app.db.models.agent_run import AgentRun

    async with get_db_context() as db:
        run = AgentRun(
            user_id=__import__("uuid").UUID(user_id),
            agent_name=agent_name,
            trigger="beat_scheduler",
            status=AgentRunStatus.PENDING,
            input_payload={},
        )
        db.add(run)
        await db.flush()
        run_id = str(run.id)
        await db.commit()
    return run_id