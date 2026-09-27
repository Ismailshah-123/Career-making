"""
app/workers/resume_tasks.py
=============================
Celery task definitions for all resume pipeline operations.

Tasks defined here:
    process_resume_task        — parse + embed a freshly uploaded resume
    embed_resume_task          — (re-)embed an already-parsed resume
    tailor_resume_task         — generate a tailored resume variant for a job
    generate_cover_letter_task — generate a cover letter for an application

All tasks are driven by the resume_workflow.py and cover_letter_agent,
wrapped with AgentRun lifecycle management (PENDING → RUNNING → DONE/FAILED).

Idempotency:
- process_resume_task checks is_parsed + is_embedded before starting work.
- tailor_resume_task checks if a tailored variant for this job already exists.
- generate_cover_letter_task checks if a cover letter already exists for
  this application — if so, returns the existing one rather than regenerating.

Error handling:
- LLM rate limit (Groq 429): retry up to 5 times with 60s fixed delay
- File not found: fail permanently (no retry) — user must re-upload
- Qdrant connection error: retry 3 times with exponential backoff
- Parse failures (corrupt file, image-only PDF): fail permanently, update
  resume.parse_error with human-readable explanation
"""

from __future__ import annotations

import asyncio
import traceback
from datetime import datetime, timezone
from typing import Any

from celery import Task
from celery.exceptions import MaxRetriesExceededError, SoftTimeLimitExceeded

from app.workers.celery_app import celery_app, InstrumentedTask
from app.core.constants import AgentRunStatus
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
        if kwargs.get("status") in (AgentRunStatus.COMPLETED, AgentRunStatus.FAILED):
            run.completed_at = datetime.now(timezone.utc)
        await db.commit()


# ---------------------------------------------------------------------------
# process_resume_task
# ---------------------------------------------------------------------------

@celery_app.task(
    base=InstrumentedTask,
    name="app.workers.resume_tasks.process_resume_task",
    bind=True,
    max_retries=3,
    default_retry_delay=30,
    soft_time_limit=300,
    time_limit=360,
    acks_late=True,
)
def process_resume_task(
    self: Task,
    *,
    resume_id: str,
    agent_run_id: str,
) -> dict[str, Any]:
    """
    Full resume processing pipeline: extract text → parse sections →
    compute ATS score → generate embedding → persist all results.

    Triggered immediately after a user uploads a new resume.
    The UI polls GET /resumes/{id} watching for is_parsed=True + is_embedded=True.

    On success:
        - resume.is_parsed = True
        - resume.parsed_sections populated with structured contact/experience/skills
        - resume.ats_score set
        - resume.is_embedded = True
        - resume.qdrant_point_ids populated
        - resume.parse_error = None

    On failure:
        - resume.parse_error set with the human-readable reason
        - resume.is_parsed remains False
        - AgentRun status = FAILED
    """
    task_logger = get_task_logger("process_resume_task", task_id=self.request.id)
    task_logger.info("Resume processing started", resume_id=resume_id)
    start_ts = datetime.now(timezone.utc)

    async def _load_resume_metadata() -> dict | None:
        from app.db.session import get_db_context
        from app.db.models.resume import Resume
        from sqlalchemy import select
        import uuid

        async with get_db_context() as db:
            result = await db.execute(select(Resume).where(Resume.id == uuid.UUID(resume_id)))
            resume = result.scalar_one_or_none()
            if not resume:
                return None
            if resume.is_parsed and resume.is_embedded:
                return {"status": "already_processed"}
            return {
                "file_path": resume.file_path,
                "mime_type": resume.mime_type or "application/pdf",
                "user_id":   str(resume.user_id),
            }

    try:
        _run(_update_agent_run(agent_run_id, status=AgentRunStatus.RUNNING))

        meta = _run(_load_resume_metadata())
        if not meta:
            _run(_update_agent_run(
                agent_run_id, status=AgentRunStatus.FAILED,
                error_message=f"Resume {resume_id} not found.",
            ))
            return {"status": "failed", "reason": "not_found"}

        if meta.get("status") == "already_processed":
            _run(_update_agent_run(agent_run_id, status=AgentRunStatus.COMPLETED))
            task_logger.info("Resume already processed — skipping", resume_id=resume_id)
            return {"status": "skipped", "reason": "already_processed"}

        if not meta.get("file_path"):
            _run(_update_agent_run(
                agent_run_id, status=AgentRunStatus.FAILED,
                error_message="Resume has no file_path — cannot process.",
            ))
            return {"status": "failed", "reason": "no_file_path"}

        from app.workflows.resume_workflow import run_parse_workflow

        final_state = _run(
            run_parse_workflow(
                user_id=meta["user_id"],
                agent_run_id=agent_run_id,
                resume_id=resume_id,
                file_path=meta["file_path"],
                mime_type=meta["mime_type"],
            )
        )

        duration_ms = int((datetime.now(timezone.utc) - start_ts).total_seconds() * 1000)
        errors      = final_state.get("errors", [])
        status      = AgentRunStatus.FAILED if errors else AgentRunStatus.COMPLETED

        ats_score = final_state.get("ats_score")
        skills    = final_state.get("extracted_skills", [])

        _run(_update_agent_run(
            agent_run_id,
            status=status,
            output_payload={
                "resume_id":     resume_id,
                "ats_score":     ats_score,
                "skills_count":  len(skills),
                "word_count":    final_state.get("word_count"),
                "years_exp":     final_state.get("years_of_experience"),
                "errors":        errors[-2:] if errors else [],
            },
            duration_ms=duration_ms,
            error_message=errors[-1].get("message") if errors else None,
            error_type=errors[-1].get("error_type") if errors else None,
        ))

        task_logger.info(
            "Resume processing complete",
            resume_id=resume_id,
            ats_score=ats_score,
            skills_count=len(skills),
            duration_ms=duration_ms,
            status=status,
        )
        return {"status": status, "ats_score": ats_score, "skills": skills[:10]}

    except SoftTimeLimitExceeded:
        task_logger.error("Resume processing soft time limit exceeded", resume_id=resume_id)
        _run(_mark_resume_parse_error(resume_id, "Processing timed out after 5 minutes."))
        _run(_update_agent_run(
            agent_run_id, status=AgentRunStatus.TIMED_OUT,
            error_message="Processing timed out.", error_type="SoftTimeLimitExceeded",
        ))
        return {"status": "timed_out"}

    except Exception as exc:
        tb = traceback.format_exc()
        task_logger.error("Resume processing failed", resume_id=resume_id, error=str(exc)[:400])

        # Permanent failures — don't retry
        permanent_error_signals = ["file not found", "corrupt", "password protected", "image-only", "cannot extract"]
        is_permanent = any(sig in str(exc).lower() for sig in permanent_error_signals)

        if is_permanent:
            _run(_mark_resume_parse_error(resume_id, str(exc)[:500]))
            _run(_update_agent_run(
                agent_run_id, status=AgentRunStatus.FAILED,
                error_message=str(exc)[:500], error_type=type(exc).__name__,
            ))
            return {"status": "failed", "error": str(exc)[:200]}

        try:
            countdown = min(30 * (2 ** self.request.retries), 300)
            raise self.retry(exc=exc, countdown=countdown)
        except MaxRetriesExceededError:
            _run(_mark_resume_parse_error(resume_id, str(exc)[:500]))
            _run(_update_agent_run(
                agent_run_id, status=AgentRunStatus.FAILED,
                error_message=str(exc)[:500], error_type=type(exc).__name__,
                error_traceback=tb,
            ))
            return {"status": "failed", "error": str(exc)[:200]}


# ---------------------------------------------------------------------------
# embed_resume_task
# ---------------------------------------------------------------------------

@celery_app.task(
    name="app.workers.resume_tasks.embed_resume_task",
    bind=True,
    max_retries=3,
    default_retry_delay=30,
    soft_time_limit=120,
    time_limit=180,
)
def embed_resume_task(self: Task, *, resume_id: str) -> dict[str, Any]:
    """
    (Re-)embed an already-parsed resume into Qdrant.

    Used when:
    - The embedding model changes (re-embed all resumes)
    - User edits their resume manually after upload
    - API call to POST /resumes/{id}/embed
    """
    task_logger = get_task_logger("embed_resume_task", task_id=self.request.id)

    async def _embed() -> dict:
        from app.db.session import get_db_context
        from app.db.models.resume import Resume
        from app.services.embedding_service import EmbeddingService
        from app.services.qdrant_service import QdrantService
        from sqlalchemy import select
        import uuid

        async with get_db_context() as db:
            result = await db.execute(select(Resume).where(Resume.id == uuid.UUID(resume_id)))
            resume = result.scalar_one_or_none()
            if not resume:
                return {"status": "skipped", "reason": "not_found"}
            if not resume.is_parsed or not resume.parsed_sections:
                return {"status": "skipped", "reason": "not_parsed_yet"}

            emb_svc = EmbeddingService()
            qdrant  = QdrantService()

            chunks  = emb_svc.chunk_resume_sections(resume.parsed_sections)
            vectors = await emb_svc.embed_batch([c["text"] for c in chunks])

            point_ids = await qdrant.upsert_resume_chunks(
                resume_id=resume_id,
                user_id=str(resume.user_id),
                chunks=chunks,
                vectors=vectors,
            )

            resume.is_embedded     = True
            resume.embedding_model = emb_svc.model_name
            resume.qdrant_point_ids = point_ids
            await db.commit()

        return {"status": "embedded", "point_ids": point_ids, "chunks": len(chunks)}

    try:
        result = _run(_embed())
        task_logger.info("Resume embedding complete", resume_id=resume_id, status=result.get("status"))
        return result
    except Exception as exc:
        task_logger.error("Resume embedding failed", resume_id=resume_id, error=str(exc)[:300])
        try:
            raise self.retry(exc=exc, countdown=30 * (2 ** self.request.retries))
        except MaxRetriesExceededError:
            return {"status": "failed", "error": str(exc)[:200]}


# ---------------------------------------------------------------------------
# tailor_resume_task
# ---------------------------------------------------------------------------

@celery_app.task(
    base=InstrumentedTask,
    name="app.workers.resume_tasks.tailor_resume_task",
    bind=True,
    max_retries=3,
    default_retry_delay=60,
    soft_time_limit=300,
    time_limit=360,
    acks_late=True,
)
def tailor_resume_task(
    self: Task,
    *,
    resume_id: str,
    job_id: str,
    agent_run_id: str,
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Generate a tailored resume variant for a specific job using the
    resume_workflow tailor pipeline.

    Returns the new tailored resume's ID for the caller to link to
    the Application row via Application.resume_id.
    """
    task_logger = get_task_logger("tailor_resume_task", task_id=self.request.id)
    task_logger.info("Resume tailoring started", resume_id=resume_id, job_id=job_id)
    start_ts = datetime.now(timezone.utc)
    opts = options or {}

    async def _load_master_resume() -> dict | None:
        from app.db.session import get_db_context
        from app.db.models.resume import Resume
        from sqlalchemy import select
        import uuid

        async with get_db_context() as db:
            res_result = await db.execute(select(Resume).where(Resume.id == uuid.UUID(resume_id)))
            resume = res_result.scalar_one_or_none()
            if not resume or not resume.is_parsed:
                return None

            # Check if tailored version already exists for this job
            dup_result = await db.execute(
                select(Resume).where(
                    Resume.parent_resume_id == uuid.UUID(resume_id),
                    Resume.tailored_for_job_id == uuid.UUID(job_id),
                    Resume.is_deleted.is_(False),
                )
            )
            existing = dup_result.scalar_one_or_none()
            if existing:
                return {"existing_id": str(existing.id)}

            return {
                "user_id":         str(resume.user_id),
                "parsed_sections": resume.parsed_sections or {},
                "raw_text":        resume.raw_text or "",
            }

    try:
        _run(_update_agent_run(agent_run_id, status=AgentRunStatus.RUNNING))

        master_data = _run(_load_master_resume())
        if not master_data:
            _run(_update_agent_run(
                agent_run_id, status=AgentRunStatus.FAILED,
                error_message="Master resume not found or not yet parsed.",
            ))
            return {"status": "failed", "reason": "master_not_ready"}

        if "existing_id" in master_data:
            task_logger.info("Reusing existing tailored resume", resume_id=master_data["existing_id"])
            _run(_update_agent_run(
                agent_run_id, status=AgentRunStatus.COMPLETED,
                output_payload={"tailored_resume_id": master_data["existing_id"], "reused": True},
            ))
            return {"status": "reused", "tailored_resume_id": master_data["existing_id"]}

        from app.workflows.resume_workflow import run_tailor_workflow

        final_state = _run(
            run_tailor_workflow(
                user_id=master_data["user_id"],
                agent_run_id=agent_run_id,
                resume_id=resume_id,
                target_job_id=job_id,
                parsed_sections=master_data["parsed_sections"],
                raw_text=master_data["raw_text"],
                tone=opts.get("tone", "professional"),
                emphasise_skills=opts.get("emphasise_skills", []),
            )
        )

        duration_ms = int((datetime.now(timezone.utc) - start_ts).total_seconds() * 1000)
        errors      = final_state.get("errors", [])
        new_id      = final_state.get("new_resume_id")
        status      = AgentRunStatus.FAILED if errors or not new_id else AgentRunStatus.COMPLETED

        _run(_update_agent_run(
            agent_run_id,
            status=status,
            output_payload={
                "tailored_resume_id": new_id,
                "ats_score":          final_state.get("ats_score"),
                "changes_summary":    final_state.get("tailoring_changes", {}),
                "errors":             errors[-2:] if errors else [],
            },
            duration_ms=duration_ms,
            error_message=errors[-1].get("message") if errors else None,
        ))

        task_logger.info(
            "Resume tailoring complete",
            tailored_resume_id=new_id,
            ats_score=final_state.get("ats_score"),
            duration_ms=duration_ms,
        )
        return {"status": status, "tailored_resume_id": new_id}

    except SoftTimeLimitExceeded:
        _run(_update_agent_run(agent_run_id, status=AgentRunStatus.TIMED_OUT))
        return {"status": "timed_out"}

    except Exception as exc:
        task_logger.error("Resume tailoring failed", error=str(exc)[:400])
        try:
            raise self.retry(exc=exc, countdown=60 * (self.request.retries + 1))
        except MaxRetriesExceededError:
            _run(_update_agent_run(
                agent_run_id, status=AgentRunStatus.FAILED,
                error_message=str(exc)[:500], error_type=type(exc).__name__,
            ))
            return {"status": "failed"}


# ---------------------------------------------------------------------------
# generate_cover_letter_task
# ---------------------------------------------------------------------------

@celery_app.task(
    base=InstrumentedTask,
    name="app.workers.resume_tasks.generate_cover_letter_task",
    bind=True,
    max_retries=5,
    default_retry_delay=60,
    soft_time_limit=180,
    time_limit=240,
    acks_late=True,
)
def generate_cover_letter_task(
    self: Task,
    *,
    application_id: str,
    agent_run_id: str,
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Generate a personalised cover letter for a job application.

    Uses the cover_letter_agent which:
    1. Loads the tailored resume + job description + company culture signals
    2. Identifies the top 3 relevant achievements from the resume
    3. Researches recent company news (via web search tool)
    4. Writes a structured letter: hook → why company → why me → CTA
    5. Scores for keyword coverage and quality
    6. Saves CoverLetter row and links to Application.cover_letter_id

    Retries up to 5 times for LLM rate-limit errors (Groq 429s).
    """
    task_logger = get_task_logger("generate_cover_letter_task", task_id=self.request.id)
    task_logger.info("Cover letter generation started", application_id=application_id)
    start_ts = datetime.now(timezone.utc)
    opts = options or {}

    async def _generate() -> dict:
        from app.db.session import get_db_context
        from app.db.models.application import Application
        from app.db.models.cover_letter import CoverLetter
        from app.agents.cover_letter_agent.agent import CoverLetterAgent
        from sqlalchemy import select
        import uuid

        async with get_db_context() as db:
            # Idempotency check — return existing if already generated
            app_result = await db.execute(
                select(Application).where(Application.id == uuid.UUID(application_id))
            )
            application = app_result.scalar_one_or_none()
            if not application:
                return {"status": "failed", "reason": "application_not_found"}

            if application.cover_letter_id:
                cl_result = await db.execute(
                    select(CoverLetter).where(CoverLetter.id == application.cover_letter_id)
                )
                existing_cl = cl_result.scalar_one_or_none()
                if existing_cl and existing_cl.status not in ("failed", "archived"):
                    task_logger.info("Reusing existing cover letter", cl_id=str(existing_cl.id))
                    return {"status": "reused", "cover_letter_id": str(existing_cl.id)}

            agent = CoverLetterAgent()
            letter = await agent.generate(
                user_id=application.user_id,
                job_id=application.job_id,
                resume_id=application.resume_id,
                application_id=uuid.UUID(application_id),
                tone=opts.get("tone", "professional"),
                highlight_skills=opts.get("highlight_skills", []),
                custom_opening=opts.get("custom_opening"),
                db=db,
            )
            await db.commit()

        return {
            "status": "generated",
            "cover_letter_id": str(letter.id),
            "quality_score":   letter.quality_score,
            "word_count":      letter.word_count,
        }

    try:
        _run(_update_agent_run(agent_run_id, status=AgentRunStatus.RUNNING))
        result = _run(_generate())

        duration_ms = int((datetime.now(timezone.utc) - start_ts).total_seconds() * 1000)
        status = AgentRunStatus.COMPLETED if result.get("status") in ("generated", "reused") else AgentRunStatus.FAILED

        _run(_update_agent_run(
            agent_run_id,
            status=status,
            output_payload=result,
            duration_ms=duration_ms,
        ))

        task_logger.info(
            "Cover letter generation complete",
            application_id=application_id,
            status=result.get("status"),
            quality_score=result.get("quality_score"),
        )
        return result

    except SoftTimeLimitExceeded:
        _run(_update_agent_run(agent_run_id, status=AgentRunStatus.TIMED_OUT))
        return {"status": "timed_out"}

    except Exception as exc:
        error_str = str(exc).lower()
        task_logger.error("Cover letter generation failed", error=str(exc)[:400])

        # Groq rate limit — longer retry delay
        if "rate limit" in error_str or "429" in error_str:
            try:
                raise self.retry(exc=exc, countdown=60)
            except MaxRetriesExceededError:
                pass
        else:
            try:
                raise self.retry(exc=exc, countdown=30 * (self.request.retries + 1))
            except MaxRetriesExceededError:
                pass

        _run(_update_agent_run(
            agent_run_id, status=AgentRunStatus.FAILED,
            error_message=str(exc)[:500], error_type=type(exc).__name__,
        ))
        return {"status": "failed", "error": str(exc)[:200]}


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

async def _mark_resume_parse_error(resume_id: str, error_message: str) -> None:
    """Persist a parse error message on the Resume row."""
    from app.db.session import get_db_context
    from app.db.models.resume import Resume
    from sqlalchemy import select
    import uuid

    async with get_db_context() as db:
        result = await db.execute(select(Resume).where(Resume.id == uuid.UUID(resume_id)))
        resume = result.scalar_one_or_none()
        if resume:
            resume.parse_error = error_message[:1000]
            resume.is_parsed   = False
            await db.commit()