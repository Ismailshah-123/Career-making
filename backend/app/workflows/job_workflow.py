"""
app/workflows/job_workflow.py
================================
Production LangGraph orchestration for the complete job pipeline.

Execution modes
---------------
1. DISCOVERY-ONLY  (target_job_id is None after match_jobs)
   Scrape boards → dedup → filter → semantic match → embed new jobs
   → persist match scores → notify user.
   Triggered by POST /jobs/discover and the Beat scheduler every 6 h.

2. SINGLE-JOB      (target_job_id provided in initial state)
   Skip discovery / dedup / filter / embed entirely (seeded in initial_state).
   Run: match (fast 1-item) → tailor resume → generate cover letter
        → submit via Playwright → outreach → schedule follow-up → notify.
   Triggered by POST /applications/ (auto_apply=True) or
               POST /applications/{id}/apply.

Graph topology (full)
---------------------

  [DISCOVERY MODE]
  discover_jobs ──asyncio.gather(N boards)──►  dedup_jobs
                                                    │
                                               filter_jobs
                                                    │
                                               match_jobs ──► [branch]
                                                    │ (no target_job_id)
                                            embed_new_jobs
                                                    │
                                           persist_match_scores
                                                    │
                                              notify_user ──► END

  [SINGLE-JOB MODE]  (enter at match_jobs with seeded state)
                                               match_jobs ──► [branch]
                                                    │ (target_job_id set)
                                            tailor_resume
                                                    │
                                        generate_cover_letter
                                                    │
                                          submit_application
                                                    │
                                       post_submit_outreach
                                                    │
                                        schedule_followup
                                                    │
                                              notify_user ──► END

  [ANY NODE FAILURE]
                                            handle_error ──► END

Checkpointing
-------------
Postgres-backed AsyncPostgresSaver checkpointer is attached at compile
time. Every node's state delta is checkpointed before the next node runs,
enabling resumable workflows across worker restarts.

Token / Cost Tracking
---------------------
All LLM-calling nodes (tailor, cover_letter) return total_prompt_tokens
and total_completion_tokens deltas which LangGraph's operator.add reducer
accumulates across the entire run. The final totals are written to
AgentRun.total_tokens and AgentRun.estimated_cost_usd on completion.

Per-node Step Logging
---------------------
Each node appends an entry to AgentRun.steps via _log_step() so the
dashboard can show a live "what's happening" progress view without
polling the graph state directly.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from datetime import datetime, timedelta, timezone
from typing import Any

from langgraph.graph import StateGraph, END

from app.core.constants import (
    NODE_DISCOVERY,
    NODE_FILTER,
    NODE_MATCH,
    NODE_TAILOR_RESUME,
    NODE_GENERATE_COVER,
    NODE_APPLY,
    NODE_NOTIFY,
    NODE_ERROR,
    SUPPORTED_JOB_BOARDS,
    MATCH_SCORE_FAIR,
    MATCH_SCORE_GOOD,
    MATCH_SCORE_EXCELLENT,
    get_match_tier,
    AgentRunStatus,
    ApplicationStatus,
    NotificationType,
    FOLLOWUP_WAIT_DAYS,
    AGENT_DISCOVERY,
)
from app.core.logging import get_logger
from app.workflows.graph import (
    node,
    build_graph,
    halt_on_error,
    get_invoke_config,
)
from app.workflows.state import JobWorkflowState, new_base_state

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Local node-name constants (not in global constants.py — internal to this graph)
# ---------------------------------------------------------------------------
NODE_DEDUP             = "dedup_jobs"
NODE_EMBED_NEW         = "embed_new_jobs"
NODE_PERSIST_SCORES    = "persist_match_scores"
NODE_OUTREACH          = "post_submit_outreach"
NODE_FOLLOWUP_SCH      = "schedule_followup"
NODE_ERROR_HANDLE      = "handle_error"

# ---------------------------------------------------------------------------
# AgentRun step logger (non-blocking fire-and-forget)
# ---------------------------------------------------------------------------

async def _log_step(
    agent_run_id: str,
    step_name: str,
    status: str,
    *,
    duration_ms: int | None = None,
    output_summary: str | None = None,
    error: str | None = None,
) -> None:
    """
    Append a step record to AgentRun.steps and update current_node.
    Swallows all exceptions — step logging must never kill a workflow node.
    """
    try:
        from app.db.session import get_db_context
        from app.db.models.agent_run import AgentRun
        from sqlalchemy import select
        import uuid

        async with get_db_context() as db:
            result = await db.execute(
                select(AgentRun).where(AgentRun.id == uuid.UUID(agent_run_id))
            )
            run = result.scalar_one_or_none()
            if run:
                run.add_step(
                    step_name,
                    status,
                    duration_ms=duration_ms,
                    output_summary=output_summary,
                    error=error,
                )
                run.current_node = step_name
                await db.commit()
    except Exception as exc:
        logger.debug(f"_log_step non-fatal error: {exc}")


# ═══════════════════════════════════════════════════════════════════════════
# NODE IMPLEMENTATIONS
# ═══════════════════════════════════════════════════════════════════════════

@node(NODE_DISCOVERY)
async def discover_jobs_node(state: JobWorkflowState) -> dict[str, Any]:
    """
    Fan out scraping across all requested job boards concurrently via
    asyncio.gather(return_exceptions=True).

    Each board scraper (LinkedInScraper, IndeedScraper, etc.) inherits
    BaseScraper which provides:
      - curl_cffi TLS fingerprint impersonation (anti-bot)
      - Rotating residential proxy support
      - Per-domain rate limiting + exponential backoff retry
      - CAPTCHA / block-page detection

    Per-board failures are captured in discovery_errors_by_board and logged
    as warnings — they never halt the overall pipeline. A run that scrapes
    3 of 4 boards successfully is far better than a run that aborts entirely.

    Raw job dicts are stashed in metadata['raw_jobs'] for the dedup node;
    they are NOT placed in discovered_job_ids (that's the dedup node's job).
    """
    t0 = time.perf_counter()

    from app.automation.scraping.linkedin_scraper  import LinkedInScraper
    from app.automation.scraping.indeed_scraper    import IndeedScraper
    from app.automation.scraping.remoteok_scraper  import RemoteOKScraper
    from app.automation.scraping.wellfound_scraper import WellfoundScraper

    SCRAPER_MAP: dict[str, Any] = {
        "linkedin":  LinkedInScraper,
        "indeed":    IndeedScraper,
        "remoteok":  RemoteOKScraper,
        "wellfound": WellfoundScraper,
    }

    boards      = [b for b in state.get("job_boards", []) if b in SCRAPER_MAP] \
                  or list(SCRAPER_MAP.keys())
    keywords    = state.get("search_keywords", [])
    locations   = state.get("search_locations", [])
    work_modes  = state.get("work_modes", ["remote"])
    max_results = state.get("max_results_per_board", 50)

    await _log_step(
        state["agent_run_id"], NODE_DISCOVERY, "running",
        output_summary=f"Scraping {len(boards)} board(s): {boards}"
    )

    async def _scrape_board(board: str) -> tuple[str, list[dict], str | None]:
        scraper = SCRAPER_MAP[board]()
        t_board = time.perf_counter()
        try:
            jobs = await scraper.search(
                keywords=keywords,
                locations=locations,
                work_modes=work_modes,
                max_results=max_results,
            )
            elapsed = int((time.perf_counter() - t_board) * 1000)
            logger.info(f"Board scraped: {board}", count=len(jobs), ms=elapsed)
            return board, jobs, None
        except Exception as exc:
            logger.warning("Board scrape failed", board=board, error=str(exc)[:300])
            return board, [], str(exc)[:300]
        finally:
            await scraper.close()

    # Fan-out — all boards run concurrently
    results = await asyncio.gather(
        *[_scrape_board(b) for b in boards],
        return_exceptions=False,
    )

    all_raw: list[dict[str, Any]] = []
    errors_by_board: dict[str, str] = {}
    board_counts: dict[str, int] = {}

    for board, jobs, error in results:
        if error:
            errors_by_board[board] = error
            board_counts[board] = 0
        else:
            # Tag each job with its source board for dedup node
            for job in jobs:
                job.setdefault("job_board", board)
            all_raw.extend(jobs)
            board_counts[board] = len(jobs)

    duration_ms = int((time.perf_counter() - t0) * 1000)

    await _log_step(
        state["agent_run_id"], NODE_DISCOVERY, "completed",
        duration_ms=duration_ms,
        output_summary=(
            f"Scraped {len(all_raw)} raw jobs across {len(boards)} board(s). "
            f"Per-board: {board_counts}. "
            f"Errors: {list(errors_by_board.keys()) or 'none'}"
        ),
    )

    logger.info(
        "Discovery complete",
        boards=boards,
        raw_count=len(all_raw),
        board_counts=board_counts,
        errors=list(errors_by_board.keys()),
        duration_ms=duration_ms,
    )

    return {
        "metadata": {
            "raw_jobs":       all_raw,
            "boards_scraped": boards,
            "board_counts":   board_counts,
        },
        "discovery_errors_by_board": errors_by_board,
    }


@node(NODE_DEDUP)
async def dedup_jobs_node(state: JobWorkflowState) -> dict[str, Any]:
    """
    Two-pass deduplication of the raw scraped jobs.

    Pass 1 — In-memory dedup by content_hash:
        Same job appearing on multiple boards in the same scrape run
        (e.g. "Senior Python Engineer at Stripe" on both LinkedIn and Indeed)
        is reduced to one entry before any DB I/O.

    Pass 2 — DB dedup by content_hash:
        For jobs that survived pass 1, check if a matching row already
        exists in the jobs table (from a prior scrape run). Existing jobs
        get last_seen_at updated for staleness tracking. Only brand-new
        jobs get INSERT'd and added to new_jobs_data for embedding.

    Returns:
        discovered_job_ids — ALL job IDs (new + existing) that survived dedup.
            These feed into filter_jobs_node.
        metadata.new_jobs_data — only newly created rows (for embed_new_jobs).
    """
    t0 = time.perf_counter()

    from app.db.session import get_db_context
    from app.repositories.job_repository import JobRepository

    raw_jobs: list[dict] = state.get("metadata", {}).get("raw_jobs", [])
    boards_scraped = state.get("metadata", {}).get("boards_scraped", [])

    if not raw_jobs:
        await _log_step(state["agent_run_id"], NODE_DEDUP, "completed",
                        output_summary="No raw jobs to deduplicate.")
        return {
            "discovered_job_ids": [],
            "metadata": {"new_job_count": 0, "boards_scraped": boards_scraped},
        }

    # ── Pass 1: in-memory dedup ──────────────────────────────────────
    seen_hashes: set[str] = set()
    unique_raw: list[dict] = []
    for job in raw_jobs:
        h = _content_hash(
            job.get("title", ""),
            job.get("company_name", ""),
            job.get("description", "") or job.get("description_cleaned", ""),
        )
        job["_content_hash"] = h
        if h not in seen_hashes:
            seen_hashes.add(h)
            unique_raw.append(job)

    in_memory_dupes = len(raw_jobs) - len(unique_raw)

    # ── Pass 2: DB dedup ─────────────────────────────────────────────
    discovered_ids: list[str] = []
    new_jobs_data: list[dict] = []
    db_dupes = 0

    import uuid as _uuid
    async with get_db_context() as db:
        repo = JobRepository(db)

        for job in unique_raw:
            content_hash = job["_content_hash"]
            existing = await repo.get_by_content_hash(content_hash)

            if existing:
                # Refresh last_seen_at so stale-job cleanup doesn't remove it
                existing.last_seen_at = datetime.now(timezone.utc)
                discovered_ids.append(str(existing.id))
                db_dupes += 1
            else:
                new_record = await repo.create_from_raw(
                    board=job.get("job_board", "unknown"),
                    raw=job,
                    content_hash=content_hash,
                )
                discovered_ids.append(str(new_record.id))
                new_jobs_data.append({
                    "id":  str(new_record.id),
                    "raw": job,
                })

        await db.commit()

    duration_ms = int((time.perf_counter() - t0) * 1000)
    summary = (
        f"{len(raw_jobs)} raw → {len(unique_raw)} unique → "
        f"{len(new_jobs_data)} new, {db_dupes} existing refreshed. "
        f"In-memory dupes: {in_memory_dupes}"
    )

    await _log_step(
        state["agent_run_id"], NODE_DEDUP, "completed",
        duration_ms=duration_ms, output_summary=summary
    )
    logger.info("Dedup complete", **{
        "raw": len(raw_jobs), "unique": len(unique_raw),
        "new": len(new_jobs_data), "existing": db_dupes,
        "duration_ms": duration_ms,
    })

    return {
        "discovered_job_ids": discovered_ids,
        "metadata": {
            "new_job_count":  len(new_jobs_data),
            "new_jobs_data":  new_jobs_data,
            "boards_scraped": boards_scraped,
        },
    }


@node(NODE_FILTER)
async def filter_jobs_node(state: JobWorkflowState) -> dict[str, Any]:
    """
    Apply six hard pre-filters that require no LLM call:

      F1  job.is_active = True
      F2  company NOT in user's excluded_companies list (case-insensitive)
      F3  user hasn't already applied to this job in any active Application row
      F4  salary_max >= user's min_salary preference (when both known)
      F5  work_mode matches user's preferred work modes (when specified)
      F6  job.posted_at within the last 60 days (suppress stale listings)

    Produces a full exclusion log {job_id: "reason_code:detail"} so the
    dashboard can show users *why* a job was filtered out.
    """
    t0 = time.perf_counter()
    from app.db.session import get_db_context
    from app.repositories.job_repository import JobRepository
    from app.repositories.user_repository import UserRepository
    from app.db.models.application import Application
    from sqlalchemy import select
    import uuid

    discovered_ids = state.get("discovered_job_ids", [])
    if not discovered_ids:
        await _log_step(state["agent_run_id"], NODE_FILTER, "completed",
                        output_summary="No jobs to filter.")
        return {
            "filtered_job_ids": [],
            "excluded_job_ids": [],
            "exclusion_reasons": {},
        }

    staleness_cutoff = datetime.now(timezone.utc) - timedelta(days=60)

    async with get_db_context() as db:
        # Load user preferences
        user_repo = UserRepository(db)
        user = await user_repo.get_by_id(uuid.UUID(state["user_id"]))
        prefs = (user.job_search_preferences if user else {}) or {}

        excluded_companies  = {c.lower() for c in prefs.get("excluded_companies", [])}
        min_salary          = prefs.get("min_salary")
        pref_work_modes     = {m.lower() for m in (prefs.get("work_mode") or [])}

        # Load existing applications for this user (to catch already-applied jobs)
        applied_result = await db.execute(
            select(Application.job_id).where(
                Application.user_id == uuid.UUID(state["user_id"]),
                Application.is_deleted.is_(False),
            )
        )
        already_applied: set[str] = {str(row[0]) for row in applied_result.fetchall()}

        # Bulk load all job rows
        job_repo = JobRepository(db)
        jobs = await job_repo.get_by_ids(
            [uuid.UUID(jid) for jid in discovered_ids]
        )

        filtered:  list[str] = []
        excluded:  list[str] = []
        reasons:   dict[str, str] = {}
        f_counts: dict[str, int] = {f"F{i}": 0 for i in range(1, 7)}

        for job in jobs:
            jid = str(job.id)

            # F1 – active
            if not job.is_active:
                excluded.append(jid); reasons[jid] = "F1:job_inactive"
                f_counts["F1"] += 1; continue

            # F2 – company blacklist
            if job.company_name and job.company_name.lower() in excluded_companies:
                excluded.append(jid)
                reasons[jid] = f"F2:company_excluded:{job.company_name}"
                f_counts["F2"] += 1; continue

            # F3 – already applied
            if jid in already_applied:
                excluded.append(jid); reasons[jid] = "F3:already_applied"
                f_counts["F3"] += 1; continue

            # F4 – salary floor
            if min_salary and job.salary_max and job.salary_max < min_salary:
                excluded.append(jid)
                reasons[jid] = f"F4:salary_below_min:{int(job.salary_max)}<{int(min_salary)}"
                f_counts["F4"] += 1; continue

            # F5 – work mode preference
            if pref_work_modes and job.work_mode and job.work_mode not in pref_work_modes:
                excluded.append(jid)
                reasons[jid] = f"F5:work_mode:{job.work_mode}"
                f_counts["F5"] += 1; continue

            # F6 – staleness (posted_at known and older than 60 days)
            if job.posted_at and job.posted_at < staleness_cutoff:
                excluded.append(jid)
                reasons[jid] = "F6:posting_stale"
                f_counts["F6"] += 1; continue

            filtered.append(jid)

    duration_ms = int((time.perf_counter() - t0) * 1000)
    summary = (
        f"Kept {len(filtered)}/{len(discovered_ids)}. "
        f"Excluded {len(excluded)} "
        f"({', '.join(f'{k}:{v}' for k, v in f_counts.items() if v)})"
    )
    await _log_step(state["agent_run_id"], NODE_FILTER, "completed",
                    duration_ms=duration_ms, output_summary=summary)
    logger.info("Filter complete", kept=len(filtered), excluded=len(excluded),
                breakdown=f_counts, duration_ms=duration_ms)

    return {
        "filtered_job_ids": filtered,
        "excluded_job_ids": excluded,
        "exclusion_reasons": reasons,
    }


@node(NODE_MATCH)
async def match_jobs_node(state: JobWorkflowState) -> dict[str, Any]:
    """
    Qdrant cosine similarity matching — resume embedding vs all filtered jobs.

    Algorithm:
    1. Resolve resume_id (use state['resume_id'] or latest master for user)
    2. Call QdrantService.bulk_match() — one batched query, not N separate queries
    3. Rank results by score DESC, filter by MATCH_SCORE_FAIR threshold (0.60)
    4. Return top_match_job_ids for the downstream branch routing

    Failure modes handled gracefully:
    - Resume not embedded yet → returns all filtered IDs with score=0 (dashboard
      will prompt user to wait for embedding)
    - Qdrant unavailable → same fallback with warning log; pipeline continues
    - Empty filtered_job_ids → returns empty lists immediately

    In single-job mode the state is pre-seeded with match_results and
    top_match_job_ids so this node effectively fast-paths through.
    """
    t0 = time.perf_counter()
    from app.services.qdrant_service import QdrantService
    from app.repositories.resume_repository import ResumeRepository
    from app.db.session import get_db_context
    import uuid

    filtered_ids = state.get("filtered_job_ids", [])

    # In single-job mode the initial state pre-seeds these — skip re-matching
    if state.get("metadata", {}).get("pre_seeded") and state.get("match_results"):
        await _log_step(state["agent_run_id"], NODE_MATCH, "completed",
                        output_summary="Pre-seeded single-job mode — skipping Qdrant query.")
        return {}

    if not filtered_ids:
        return {"match_results": [], "top_match_job_ids": [], "resume_id": state.get("resume_id")}

    # Resolve resume
    resume_id = state.get("resume_id")
    if not resume_id:
        async with get_db_context() as db:
            repo   = ResumeRepository(db)
            resume = await repo.get_latest_master(uuid.UUID(state["user_id"]))
            if resume and resume.is_embedded:
                resume_id = str(resume.id)

    if not resume_id:
        await _log_step(state["agent_run_id"], NODE_MATCH, "completed",
                        output_summary="No embedded resume found — returning unranked jobs.")
        return {
            "match_results":     [{"job_id": jid, "score": 0.0, "tier": "unknown"} for jid in filtered_ids],
            "top_match_job_ids": filtered_ids[:20],
            "resume_id":         None,
        }

    # Execute Qdrant bulk match
    qdrant = QdrantService()
    try:
        raw_matches = await qdrant.bulk_match(resume_id=resume_id, job_ids=filtered_ids)
    except Exception as exc:
        logger.warning("Qdrant bulk_match failed — unranked fallback", error=str(exc)[:200])
        raw_matches = [{"job_id": jid, "score": 0.0, "tier": "unknown"} for jid in filtered_ids]

    # Build typed match result list
    match_results = [
        {
            "job_id":            m["job_id"],
            "score":             round(m["score"], 4),
            "tier":              get_match_tier(m["score"]).value,
            "matched_skills":    m.get("matched_skills", []),
            "missing_skills":    m.get("missing_skills", []),
            "keyword_coverage":  round(m.get("keyword_coverage", 0.0), 3),
            "explanation":       m.get("explanation", ""),
        }
        for m in raw_matches
    ]

    sorted_matches  = sorted(match_results, key=lambda x: x["score"], reverse=True)
    top_match_ids   = [m["job_id"] for m in sorted_matches if m["score"] >= MATCH_SCORE_FAIR]

    excellent = sum(1 for m in sorted_matches if m["score"] >= MATCH_SCORE_EXCELLENT)
    good      = sum(1 for m in sorted_matches if MATCH_SCORE_GOOD <= m["score"] < MATCH_SCORE_EXCELLENT)

    duration_ms = int((time.perf_counter() - t0) * 1000)
    summary = (
        f"Matched {len(match_results)} jobs. "
        f"Above threshold: {len(top_match_ids)} "
        f"(excellent={excellent}, good={good}). "
        f"Top score: {sorted_matches[0]['score'] if sorted_matches else 0:.3f}"
    )
    await _log_step(state["agent_run_id"], NODE_MATCH, "completed",
                    duration_ms=duration_ms, output_summary=summary)
    logger.info("Match complete", total=len(match_results), above_threshold=len(top_match_ids),
                excellent=excellent, good=good, duration_ms=duration_ms)

    return {
        "resume_id":         resume_id,
        "match_results":     match_results,
        "top_match_job_ids": top_match_ids,
    }


@node(NODE_EMBED_NEW)
async def embed_new_jobs_node(state: JobWorkflowState) -> dict[str, Any]:
    """
    Embed newly scraped jobs into the Qdrant jobs collection.

    Only runs in discovery-only mode. Batches embedding API calls (50 per
    request) to stay within OpenAI token limits per request. For each
    batch:
      1. Build text from description_cleaned → description → title fallback
      2. Call EmbeddingService.embed_batch() — single OpenAI API call
      3. Call QdrantService.upsert_jobs_batch() — single Qdrant upsert
      4. Persist is_embedded=True + qdrant_point_id on each Job DB row

    Failures are non-fatal at the batch level — an embedding API rate-limit
    on batch 3 of 5 doesn't discard batches 1, 2, 4, 5.
    """
    t0 = time.perf_counter()
    from app.services.embedding_service import EmbeddingService
    from app.services.qdrant_service    import QdrantService
    from app.db.session import get_db_context
    from app.repositories.job_repository import JobRepository
    import uuid

    new_jobs: list[dict] = state.get("metadata", {}).get("new_jobs_data", [])
    if not new_jobs:
        await _log_step(state["agent_run_id"], NODE_EMBED_NEW, "completed",
                        output_summary="No new jobs to embed.")
        return {"metadata": {"embedded_count": 0}}

    emb_svc  = EmbeddingService()
    qdrant   = QdrantService()
    embedded = 0
    failed   = 0
    BATCH    = 50

    async with get_db_context() as db:
        repo = JobRepository(db)

        for i in range(0, len(new_jobs), BATCH):
            batch    = new_jobs[i:i + BATCH]
            texts:    list[str]       = []
            job_ids:  list[str]       = []
            payloads: list[dict]      = []

            for item in batch:
                raw  = item["raw"]
                text = (
                    raw.get("description_cleaned")
                    or raw.get("description")
                    or f"{raw.get('title', '')} at {raw.get('company_name', '')}"
                )[:8000]

                if not text or len(text.strip()) < 30:
                    continue

                texts.append(text)
                job_ids.append(item["id"])
                payloads.append({
                    "job_id":           item["id"],
                    "title":            raw.get("title", ""),
                    "company":          raw.get("company_name", ""),
                    "work_mode":        raw.get("work_mode", "onsite"),
                    "job_type":         raw.get("job_type", "full_time"),
                    "job_board":        raw.get("job_board", "unknown"),
                    "required_skills":  raw.get("required_skills", [])[:20],
                    "experience_level": raw.get("experience_level"),
                    "salary_min":       raw.get("salary_min"),
                    "salary_max":       raw.get("salary_max"),
                })

            if not texts:
                continue

            try:
                vectors   = await emb_svc.embed_batch(texts)
                point_ids = await qdrant.upsert_jobs_batch(
                    job_ids=job_ids,
                    vectors=vectors,
                    payloads=payloads,
                )

                for idx, job_id in enumerate(job_ids):
                    job = await repo.get_by_id(uuid.UUID(job_id))
                    if job:
                        job.is_embedded     = True
                        job.qdrant_point_id = point_ids[idx] if idx < len(point_ids) else None
                        job.embedding_model = emb_svc.model_name
                        embedded += 1

                await db.flush()
            except Exception as exc:
                logger.warning("Job batch embed failed", batch_start=i,
                               batch_size=len(texts), error=str(exc)[:300])
                failed += len(texts)

        await db.commit()

    duration_ms = int((time.perf_counter() - t0) * 1000)
    summary = f"Embedded {embedded} jobs. Failed: {failed}."
    await _log_step(state["agent_run_id"], NODE_EMBED_NEW, "completed",
                    duration_ms=duration_ms, output_summary=summary)
    logger.info("Embedding complete", embedded=embedded, failed=failed, duration_ms=duration_ms)

    return {"metadata": {"embedded_count": embedded, "embed_failed": failed}}


@node(NODE_PERSIST_SCORES)
async def persist_match_scores_node(state: JobWorkflowState) -> dict[str, Any]:
    """
    Write match scores back to the Qdrant job payloads and optionally to
    any existing Application rows for this user + job combination.

    This ensures the dashboard shows match scores for all top-matched jobs
    without needing a live Qdrant query on every page load — the score is
    denormalised onto the Application row where it exists.

    Also updates Job.match_count and Job.avg_match_score aggregates used
    by the analytics dashboard to surface "hottest" jobs across all users.
    """
    t0 = time.perf_counter()
    from app.db.session import get_db_context
    from app.repositories.job_repository import JobRepository
    from app.db.models.application import Application
    from sqlalchemy import select
    import uuid

    match_results = state.get("match_results", [])
    if not match_results:
        return {}

    async with get_db_context() as db:
        job_repo = JobRepository(db)
        user_uid = uuid.UUID(state["user_id"])

        for match in match_results:
            jid   = match["job_id"]
            score = match["score"]
            tier  = match["tier"]

            # Update job aggregate match stats
            try:
                job = await job_repo.get_by_id(uuid.UUID(jid))
                if job:
                    job.match_count += 1
                    prior_avg = job.avg_match_score or 0.0
                    job.avg_match_score = round(
                        (prior_avg * (job.match_count - 1) + score) / job.match_count, 4
                    )
            except Exception:
                pass

            # Update Application row if one exists for this user + job
            try:
                result = await db.execute(
                    select(Application).where(
                        Application.user_id    == user_uid,
                        Application.job_id     == uuid.UUID(jid),
                        Application.is_deleted.is_(False),
                    )
                )
                app = result.scalar_one_or_none()
                if app:
                    app.match_score      = score
                    app.match_tier       = tier
                    app.matched_skills   = match.get("matched_skills", [])
                    app.missing_skills   = match.get("missing_skills", [])
                    app.keyword_match_score = match.get("keyword_coverage", 0.0)
            except Exception:
                pass

        await db.commit()

    duration_ms = int((time.perf_counter() - t0) * 1000)
    await _log_step(
        state["agent_run_id"], NODE_PERSIST_SCORES, "completed",
        duration_ms=duration_ms,
        output_summary=f"Persisted match scores for {len(match_results)} jobs.",
    )
    return {}


@node(NODE_TAILOR_RESUME)
async def tailor_resume_node(state: JobWorkflowState) -> dict[str, Any]:
    """
    Generate a tailored resume variant for state['target_job_id'] by
    calling resume_workflow.run_tailor_workflow().

    The tailor workflow:
      1. Passes through extract_text / parse_sections (pre-seeded from master)
      2. Runs tailor_sections: rewrites experience bullets + summary to
         emphasise skills overlapping with the job requirements
      3. Computes ATS score for the new variant
      4. Embeds the tailored variant into Qdrant
      5. Creates a new Resume row (is_master=False, parent_resume_id=master.id)

    Idempotent: reuses an existing tailored variant for the same job rather
    than re-running the LLM if one already exists in the DB.
    """
    t0 = time.perf_counter()
    from app.db.session import get_db_context
    from app.db.models.resume import Resume
    from app.repositories.resume_repository import ResumeRepository
    from app.workflows.resume_workflow import run_tailor_workflow
    from sqlalchemy import select
    import uuid

    target_job_id = state.get("target_job_id")
    resume_id     = state.get("resume_id")

    if not target_job_id or not resume_id:
        await _log_step(state["agent_run_id"], NODE_TAILOR_RESUME, "completed",
                        output_summary="Skipped — no target_job_id or resume_id.")
        return {"tailored_resume_id": None}

    await _log_step(state["agent_run_id"], NODE_TAILOR_RESUME, "running",
                    output_summary=f"Tailoring resume for job {target_job_id}")

    async with get_db_context() as db:
        repo = ResumeRepository(db)

        # Idempotency check
        dup_result = await db.execute(
            select(Resume).where(
                Resume.parent_resume_id    == uuid.UUID(resume_id),
                Resume.tailored_for_job_id == uuid.UUID(target_job_id),
                Resume.is_deleted.is_(False),
                Resume.is_parsed.is_(True),
            )
        )
        existing = dup_result.scalar_one_or_none()
        if existing:
            logger.info("Reusing existing tailored resume", resume_id=str(existing.id))
            duration_ms = int((time.perf_counter() - t0) * 1000)
            await _log_step(state["agent_run_id"], NODE_TAILOR_RESUME, "completed",
                            duration_ms=duration_ms,
                            output_summary=f"Reused existing tailored resume {existing.id}")
            return {"tailored_resume_id": str(existing.id)}

        master = await repo.get_by_id(uuid.UUID(resume_id))
        if not master or not master.is_parsed:
            raise ValueError(
                f"Master resume {resume_id} is not yet parsed. "
                "Cannot tailor — wait for parse to complete."
            )
        raw_text        = master.raw_text or ""
        parsed_sections = master.parsed_sections or {}

    meta = state.get("metadata", {})
    result = await run_tailor_workflow(
        user_id=state["user_id"],
        agent_run_id=state["agent_run_id"],
        resume_id=resume_id,
        target_job_id=target_job_id,
        parsed_sections=parsed_sections,
        raw_text=raw_text,
        tone=meta.get("tailor_tone", "professional"),
        emphasise_skills=meta.get("emphasise_skills", []),
    )

    new_id      = result.get("new_resume_id")
    ats_score   = result.get("ats_score")
    prompt_tok  = result.get("total_prompt_tokens", 0)
    comp_tok    = result.get("total_completion_tokens", 0)
    duration_ms = int((time.perf_counter() - t0) * 1000)

    await _log_step(
        state["agent_run_id"], NODE_TAILOR_RESUME, "completed",
        duration_ms=duration_ms,
        output_summary=(
            f"Tailored resume created: {new_id}. "
            f"ATS score: {ats_score}. Tokens: {prompt_tok + comp_tok}"
        ),
    )
    logger.info("Resume tailored", new_resume_id=new_id, ats_score=ats_score, duration_ms=duration_ms)

    return {
        "tailored_resume_id":    new_id,
        "total_prompt_tokens":   prompt_tok,
        "total_completion_tokens": comp_tok,
        "total_llm_calls":       result.get("total_llm_calls", 0),
    }


@node(NODE_GENERATE_COVER)
async def generate_cover_letter_node(state: JobWorkflowState) -> dict[str, Any]:
    """
    Generate a personalised cover letter for state['target_job_id'] using
    the CoverLetterAgent.

    The agent:
      1. Loads the tailored resume + full job description + company culture data
      2. Searches for recent company news (web search tool) for personalisation
      3. Drafts a structured letter: hook → why_company → why_me → CTA
      4. Self-evaluates for keyword coverage and quality score
      5. Saves CoverLetter DB row and links it to the Application row

    Idempotent: returns an existing cover letter if application.cover_letter_id
    already points to one that is not in a failed state.
    """
    t0 = time.perf_counter()
    from app.agents.cover_letter_agent.agent import CoverLetterAgent
    from app.db.session import get_db_context
    from app.db.models.application import Application
    from app.db.models.cover_letter import CoverLetter
    from sqlalchemy import select
    import uuid

    target_job_id      = state.get("target_job_id")
    application_id     = state.get("application_id")
    tailored_resume_id = state.get("tailored_resume_id") or state.get("resume_id")

    if not target_job_id:
        await _log_step(state["agent_run_id"], NODE_GENERATE_COVER, "completed",
                        output_summary="Skipped — no target_job_id.")
        return {"cover_letter_id": None}

    await _log_step(state["agent_run_id"], NODE_GENERATE_COVER, "running",
                    output_summary=f"Generating cover letter for job {target_job_id}")

    async with get_db_context() as db:
        # Idempotency: check if application already has a non-failed cover letter
        if application_id:
            app_result = await db.execute(
                select(Application).where(Application.id == uuid.UUID(application_id))
            )
            app = app_result.scalar_one_or_none()
            if app and app.cover_letter_id:
                cl_result = await db.execute(
                    select(CoverLetter).where(CoverLetter.id == app.cover_letter_id)
                )
                existing_cl = cl_result.scalar_one_or_none()
                if existing_cl and existing_cl.status not in ("failed", "archived"):
                    logger.info("Reusing existing cover letter", cl_id=str(existing_cl.id))
                    return {
                        "cover_letter_id": str(existing_cl.id),
                        "cover_letter_text": existing_cl.body or "",
                    }

        agent  = CoverLetterAgent()
        letter = await agent.generate(
            user_id=uuid.UUID(state["user_id"]),
            job_id=uuid.UUID(target_job_id),
            resume_id=uuid.UUID(tailored_resume_id) if tailored_resume_id else None,
            application_id=uuid.UUID(application_id) if application_id else None,
            db=db,
        )
        await db.commit()

    duration_ms = int((time.perf_counter() - t0) * 1000)
    prompt_tok  = getattr(letter, "_prompt_tokens", 0)
    comp_tok    = getattr(letter, "_completion_tokens", 0)

    await _log_step(
        state["agent_run_id"], NODE_GENERATE_COVER, "completed",
        duration_ms=duration_ms,
        output_summary=(
            f"Cover letter generated: {letter.id}. "
            f"Quality: {letter.quality_score}. Words: {letter.word_count}."
        ),
    )
    logger.info("Cover letter generated", cl_id=str(letter.id),
                quality=letter.quality_score, words=letter.word_count, duration_ms=duration_ms)

    return {
        "cover_letter_id":         str(letter.id),
        "cover_letter_text":       letter.body or "",
        "total_prompt_tokens":     prompt_tok,
        "total_completion_tokens": comp_tok,
        "total_llm_calls":         1,
    }


@node(NODE_APPLY)
async def submit_application_node(state: JobWorkflowState) -> dict[str, Any]:
    """
    Execute Playwright browser automation by delegating to
    apply_workflow.run_apply_workflow() — the dedicated pure-submission
    sub-graph that handles:
      - ATS provider detection
      - Form structure detection
      - Adaptive field filling (text, file upload, selects, radios)
      - Pre-submit validation
      - Submit + confirmation capture
      - Screenshot at every stage
      - DB update (status → APPLIED)

    Returns submission outcome in state so routing can proceed to outreach
    or error handling accordingly.
    """
    t0 = time.perf_counter()
    from app.workflows.apply_workflow import run_apply_workflow

    application_id      = state.get("application_id")
    target_job_id       = state.get("target_job_id")
    tailored_resume_id  = state.get("tailored_resume_id") or state.get("resume_id", "")

    if not application_id or not target_job_id:
        raise ValueError("submit_application_node requires application_id and target_job_id.")

    await _log_step(
        state["agent_run_id"], NODE_APPLY, "running",
        output_summary=(
            f"Submitting application {application_id} via Playwright. "
            f"ATS: {state.get('ats_provider', 'auto-detect')}"
        ),
    )

    result = await run_apply_workflow(
        user_id=state["user_id"],
        agent_run_id=state["agent_run_id"],
        application_id=application_id,
        job_id=target_job_id,
        base_resume_id=tailored_resume_id,
        apply_url=state.get("apply_url"),
        ats_provider=state.get("ats_provider"),
    )

    success       = result.get("submission_success", False)
    manual_review = result.get("requires_manual_review", False)
    error_msg     = result.get("manual_review_reason") if not success else None
    duration_ms   = int((time.perf_counter() - t0) * 1000)

    await _log_step(
        state["agent_run_id"], NODE_APPLY,
        "completed" if success else "failed",
        duration_ms=duration_ms,
        output_summary=(
            f"Submission {'SUCCESS' if success else 'FAILED'}. "
            f"Manual review: {manual_review}. "
            f"ATS: {result.get('ats_provider')}. "
            f"Fields filled: {len(result.get('form_fields_filled', []))}."
        ),
        error=error_msg,
    )

    logger.info(
        "Application submission complete",
        application_id=application_id,
        success=success,
        manual_review=manual_review,
        ats=result.get("ats_provider"),
        duration_ms=duration_ms,
    )

    return {
        "application_submitted":  success,
        "submission_method":      result.get("ats_provider"),
        "submission_error":       error_msg,
        "requires_manual_review": manual_review,
    }


@node(NODE_OUTREACH)
async def post_submit_outreach_node(state: JobWorkflowState) -> dict[str, Any]:
    """
    After a confirmed submission, enqueue a non-blocking outreach Celery task.

    Gated on user.job_search_preferences.outreach_enabled (default False).
    The task fires 5 minutes after submission to let the confirmation email
    settle before we reach out — recruiter discovery + message drafting
    happens inside send_outreach_task asynchronously.

    Any failure here is logged and swallowed — it must never affect the
    submission outcome or trigger an error route.
    """
    if not state.get("application_submitted"):
        return {}

    try:
        from app.db.session import get_db_context
        from app.db.models.user import User
        from app.db.models.agent_run import AgentRun
        from sqlalchemy import select
        import uuid

        async with get_db_context() as db:
            result = await db.execute(
                select(User).where(User.id == uuid.UUID(state["user_id"]))
            )
            user = result.scalar_one_or_none()
            if not user:
                return {}

            prefs = user.job_search_preferences or {}
            if not prefs.get("outreach_enabled", False):
                logger.debug("Outreach disabled for user", user_id=state["user_id"])
                return {}

            # Create an AgentRun placeholder for the outreach task
            run = AgentRun(
                user_id=uuid.UUID(state["user_id"]),
                agent_name="outreach_agent",
                trigger="job_workflow",
                status=AgentRunStatus.PENDING,
                input_payload={
                    "application_id": state.get("application_id"),
                    "job_id":         state.get("target_job_id"),
                },
            )
            db.add(run)
            await db.flush()
            run_id = str(run.id)
            await db.commit()

        from app.workers.job_tasks import send_outreach_task
        send_outreach_task.apply_async(
            kwargs={
                "application_id": state.get("application_id", ""),
                "agent_run_id":   run_id,
            },
            countdown=300,  # 5-minute delay
        )
        logger.info("Post-submit outreach task enqueued",
                    application_id=state.get("application_id"), countdown_s=300)

    except Exception as exc:
        logger.warning("Post-submit outreach enqueue failed (non-fatal)", error=str(exc)[:200])

    return {}


@node(NODE_FOLLOWUP_SCH)
async def schedule_followup_node(state: JobWorkflowState) -> dict[str, Any]:
    """
    Schedule the first follow-up Celery ETA task and write next_followup_at
    on the Application row.

    The ETA task (send_followup_task) fires automatically after
    FOLLOWUP_WAIT_DAYS (7 days). The followup_agent checks current
    application status before sending — if the user has already been
    contacted by the employer (status changed to ACKNOWLEDGED or beyond)
    the follow-up is silently skipped.

    Only runs after a confirmed submission in single-job mode.
    """
    if not state.get("application_submitted") or not state.get("application_id"):
        return {}

    next_followup = datetime.now(timezone.utc) + timedelta(days=FOLLOWUP_WAIT_DAYS)

    try:
        from app.db.session import get_db_context
        from app.db.models.application import Application
        from sqlalchemy import select
        import uuid

        async with get_db_context() as db:
            result = await db.execute(
                select(Application).where(
                    Application.id == uuid.UUID(state["application_id"])
                )
            )
            app = result.scalar_one_or_none()
            if app:
                app.next_followup_at = next_followup
                await db.commit()

        from app.workers.job_tasks import send_followup_task
        send_followup_task.apply_async(
            args=[state["application_id"], state["agent_run_id"]],
            eta=next_followup,
        )
        logger.info(
            "Follow-up scheduled",
            application_id=state["application_id"],
            eta=next_followup.isoformat(),
        )

    except Exception as exc:
        logger.warning("Follow-up scheduling failed (non-fatal)", error=str(exc)[:200])

    return {"metadata": {"next_followup_at": next_followup.isoformat()}}


@node(NODE_NOTIFY)
async def notify_user_node(state: JobWorkflowState) -> dict[str, Any]:
    """
    Dispatch the final user notification summarising the pipeline outcome.

    Discovery-only:
        "X new jobs matched — view top matches in dashboard"
        (only sent if at least 1 job matched above threshold)

    Single-job (success):
        "Application submitted to {company} for {role}"

    Single-job (manual review required):
        "Could not auto-apply — please review and submit manually"

    Uses NotificationService which routes to email / in-app / webhook
    based on user.notification_preferences.
    """
    from app.services.notification_service import NotificationService
    import uuid

    svc = NotificationService()

    if state.get("application_submitted"):
        notif_type = NotificationType.APPLICATION_SUBMITTED
        context    = {
            "application_id":       state.get("application_id", ""),
            "job_title":            state.get("metadata", {}).get("job_title", ""),
            "company_name":         state.get("metadata", {}).get("company_name", ""),
            "apply_url":            state.get("apply_url", ""),
            "requires_manual_review": state.get("requires_manual_review", False),
        }
    elif state.get("metadata", {}).get("pre_seeded"):
        # Single-job mode where submission was not reached (unlikely — log only)
        return {"notification_sent": False}
    else:
        discovered = len(state.get("discovered_job_ids", []))
        matched    = len(state.get("top_match_job_ids", []))
        if matched == 0:
            # No matches above threshold — skip notification to avoid noise
            logger.info("No matches above threshold — skipping notification")
            return {"notification_sent": False}

        notif_type = NotificationType.JOB_MATCHED
        context    = {
            "discovered_count": discovered,
            "matched_count":    matched,
            "boards":           state.get("metadata", {}).get("boards_scraped", []),
        }

    try:
        await svc.notify(
            user_id=uuid.UUID(state["user_id"]),
            notification_type=notif_type,
            context=context,
        )
        await _log_step(
            state["agent_run_id"], NODE_NOTIFY, "completed",
            output_summary=f"Notification dispatched: {notif_type}",
        )
    except Exception as exc:
        logger.warning("User notification failed (non-fatal)", error=str(exc)[:200])

    return {"notification_sent": True}


async def handle_error_node(state: JobWorkflowState) -> dict[str, Any]:
    """
    Terminal error handler.

    Collates all accumulated errors from state['errors'], logs them fully,
    sends a user notification with the specific failure reason (not a generic
    "something went wrong"), and marks the AgentRun as FAILED via
    _log_step with error status.
    """
    from app.services.notification_service import NotificationService
    import uuid

    errors = state.get("errors", [])
    last_error = errors[-1] if errors else {}

    logger.error(
        "job_workflow halted",
        agent_run_id=state.get("agent_run_id"),
        user_id=state.get("user_id"),
        error_count=len(errors),
        last_node=last_error.get("node", "unknown"),
        last_error=last_error.get("message", ""),
    )

    await _log_step(
        state["agent_run_id"],
        last_error.get("node", "unknown"),
        "failed",
        error=last_error.get("message", "")[:500],
    )

    try:
        svc = NotificationService()
        await svc.notify(
            user_id=uuid.UUID(state["user_id"]),
            notification_type=NotificationType.AGENT_FAILED,
            context={
                "errors": [e.get("message", "")[:200] for e in errors[-3:]],
                "failed_node": last_error.get("node", "unknown"),
                "agent": "job_workflow",
            },
        )
    except Exception as exc:
        logger.error("Error notification dispatch failed", error=str(exc)[:200])

    return {"status": "failed"}


# ═══════════════════════════════════════════════════════════════════════════
# ROUTING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def _route_after_match(state: JobWorkflowState) -> str:
    """
    Critical branch after semantic matching:

    should_halt=True       → NODE_ERROR_HANDLE (node failure upstream)
    target_job_id set      → Single-job mode: proceed to tailor → cover → apply
    target_job_id not set  → Discovery mode: embed new jobs → persist scores → notify
    """
    if state.get("should_halt"):
        return NODE_ERROR_HANDLE
    if state.get("target_job_id"):
        return NODE_TAILOR_RESUME
    return NODE_EMBED_NEW


def _route_after_embed(state: JobWorkflowState) -> str:
    """After embedding, persist match scores then notify."""
    if state.get("should_halt"):
        return NODE_ERROR_HANDLE
    return NODE_PERSIST_SCORES


def _route_after_submit(state: JobWorkflowState) -> str:
    """
    After submission attempt, always proceed to outreach regardless of
    submission success/failure — outreach node checks state.application_submitted
    internally and no-ops if it was not submitted.
    """
    if state.get("should_halt"):
        return NODE_ERROR_HANDLE
    return NODE_OUTREACH


# ═══════════════════════════════════════════════════════════════════════════
# GRAPH CONSTRUCTION
# ═══════════════════════════════════════════════════════════════════════════

async def build_job_graph():
    """
    Compile the job pipeline StateGraph.

    Compiled fresh per-invocation to avoid asyncio event-loop conflicts
    across forked Celery worker processes. Compilation is ~50ms, negligible
    relative to the minutes of scraping / LLM / browser work.

    Postgres checkpointer attached for full workflow resumability — if a
    worker dies mid-run the next invocation with the same workflow_id will
    resume from the last completed node rather than starting over.
    """
    graph = StateGraph(JobWorkflowState)

    # ── Register nodes ──────────────────────────────────────────────────
    graph.add_node(NODE_DISCOVERY,      discover_jobs_node)
    graph.add_node(NODE_DEDUP,          dedup_jobs_node)
    graph.add_node(NODE_FILTER,         filter_jobs_node)
    graph.add_node(NODE_MATCH,          match_jobs_node)
    graph.add_node(NODE_EMBED_NEW,      embed_new_jobs_node)
    graph.add_node(NODE_PERSIST_SCORES, persist_match_scores_node)
    graph.add_node(NODE_TAILOR_RESUME,  tailor_resume_node)
    graph.add_node(NODE_GENERATE_COVER, generate_cover_letter_node)
    graph.add_node(NODE_APPLY,          submit_application_node)
    graph.add_node(NODE_OUTREACH,       post_submit_outreach_node)
    graph.add_node(NODE_FOLLOWUP_SCH,   schedule_followup_node)
    graph.add_node(NODE_NOTIFY,         notify_user_node)
    graph.add_node(NODE_ERROR_HANDLE,   handle_error_node)

    graph.set_entry_point(NODE_DISCOVERY)

    # ── Edge wiring ─────────────────────────────────────────────────────

    # Discovery pipeline
    graph.add_conditional_edges(
        NODE_DISCOVERY, halt_on_error(success_node=NODE_DEDUP, error_node=NODE_ERROR_HANDLE)
    )
    graph.add_conditional_edges(
        NODE_DEDUP, halt_on_error(success_node=NODE_FILTER, error_node=NODE_ERROR_HANDLE)
    )
    graph.add_conditional_edges(
        NODE_FILTER, halt_on_error(success_node=NODE_MATCH, error_node=NODE_ERROR_HANDLE)
    )

    # Branch point: discovery-only vs single-job
    graph.add_conditional_edges(NODE_MATCH, _route_after_match)

    # Discovery-only branch: embed → persist scores → notify
    graph.add_conditional_edges(NODE_EMBED_NEW, _route_after_embed)
    graph.add_conditional_edges(
        NODE_PERSIST_SCORES, halt_on_error(success_node=NODE_NOTIFY, error_node=NODE_ERROR_HANDLE)
    )

    # Single-job branch: tailor → cover → submit → outreach → followup → notify
    graph.add_conditional_edges(
        NODE_TAILOR_RESUME, halt_on_error(success_node=NODE_GENERATE_COVER, error_node=NODE_ERROR_HANDLE)
    )
    graph.add_conditional_edges(
        NODE_GENERATE_COVER, halt_on_error(success_node=NODE_APPLY, error_node=NODE_ERROR_HANDLE)
    )
    graph.add_conditional_edges(NODE_APPLY, _route_after_submit)
    graph.add_conditional_edges(
        NODE_OUTREACH, halt_on_error(success_node=NODE_FOLLOWUP_SCH, error_node=NODE_NOTIFY)
    )
    graph.add_conditional_edges(
        NODE_FOLLOWUP_SCH, halt_on_error(success_node=NODE_NOTIFY, error_node=NODE_NOTIFY)
    )

    # Terminal edges
    graph.add_edge(NODE_NOTIFY,       END)
    graph.add_edge(NODE_ERROR_HANDLE, END)

    return await build_graph(graph)


# ═══════════════════════════════════════════════════════════════════════════
# PUBLIC ENTRY POINTS
# ═══════════════════════════════════════════════════════════════════════════

async def run_discovery_workflow(
    *,
    user_id: str,
    agent_run_id: str,
    job_boards: list[str],
    keywords: list[str],
    locations: list[str],
    work_modes: list[str],
    max_results_per_board: int,
    match_to_resume_id: str | None = None,
) -> dict[str, Any]:
    """
    Discovery-only mode entry point.

    Executes: discover → dedup → filter → match → embed → persist_scores → notify.

    Called from:
        workers/job_tasks.py:discover_jobs_task  (Celery task)
        api/v1/jobs.py:discover_jobs             (manual API trigger)
    """
    compiled    = await build_job_graph()
    workflow_id = f"discovery-{agent_run_id}"

    initial_state: dict[str, Any] = {
        **new_base_state(
            user_id=user_id,
            agent_run_id=agent_run_id,
            workflow_id=workflow_id,
        ),
        "job_boards":            job_boards or [b.value for b in SUPPORTED_JOB_BOARDS],
        "search_keywords":       keywords,
        "search_locations":      locations,
        "work_modes":            work_modes,
        "max_results_per_board": max_results_per_board,
        "resume_id":             match_to_resume_id,
        "target_job_id":         None,
        "application_id":        None,
        "apply_url":             None,
        "ats_provider":          None,
    }

    config      = get_invoke_config(workflow_id=workflow_id, recursion_limit=50)
    final_state = await compiled.ainvoke(initial_state, config=config)

    logger.info(
        "Discovery workflow finished",
        agent_run_id=agent_run_id,
        discovered=len(final_state.get("discovered_job_ids", [])),
        matched=len(final_state.get("top_match_job_ids", [])),
        status=final_state.get("status", "completed"),
    )
    return final_state


async def run_application_workflow(
    *,
    user_id: str,
    agent_run_id: str,
    application_id: str,
    target_job_id: str,
    resume_id: str,
    apply_url: str | None = None,
    ats_provider: str | None = None,
) -> dict[str, Any]:
    """
    Single-job mode entry point.

    Seeds discovery / dedup / filter / match nodes with the known job so
    they pass through without doing any network I/O, then executes:
    tailor → cover_letter → submit → outreach → followup → notify.

    Called from:
        workers/job_tasks.py:submit_application_task  (Celery task)
        api/v1/applications.py:trigger_apply          (manual trigger)
    """
    compiled    = await build_job_graph()
    workflow_id = f"apply-job-{application_id}"

    initial_state: dict[str, Any] = {
        **new_base_state(
            user_id=user_id,
            agent_run_id=agent_run_id,
            workflow_id=workflow_id,
        ),
        # Pre-seed all scraping/dedup/filter/match stages with the known job
        "job_boards":            [],
        "search_keywords":       [],
        "search_locations":      [],
        "work_modes":            [],
        "max_results_per_board": 0,
        "discovered_job_ids":    [target_job_id],
        "filtered_job_ids":      [target_job_id],
        "match_results":         [{
            "job_id":           target_job_id,
            "score":            1.0,
            "tier":             "excellent",
            "matched_skills":   [],
            "missing_skills":   [],
            "keyword_coverage": 1.0,
        }],
        "top_match_job_ids":     [target_job_id],
        # Single-job targeting
        "target_job_id":  target_job_id,
        "resume_id":      resume_id,
        "application_id": application_id,
        "apply_url":      apply_url,
        "ats_provider":   ats_provider,
        "metadata": {
            "pre_seeded": True,
        },
    }

    config      = get_invoke_config(workflow_id=workflow_id, recursion_limit=80)
    final_state = await compiled.ainvoke(initial_state, config=config)

    logger.info(
        "Application workflow finished",
        agent_run_id=agent_run_id,
        application_id=application_id,
        success=final_state.get("application_submitted"),
        manual_review=final_state.get("requires_manual_review"),
        status=final_state.get("status", "completed"),
    )
    return final_state


# ═══════════════════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

def _content_hash(title: str, company: str, description: str) -> str:
    """
    MD5 fingerprint for cross-board job deduplication.

    Normalises to lowercase and trims before hashing to catch minor
    formatting differences between the same job scraped from two boards.
    Only the first 300 chars of description are used — enough for
    fingerprinting without being sensitive to truncation differences.
    """
    basis = (
        f"{title.lower().strip()}"
        f"|{company.lower().strip()}"
        f"|{description[:300].lower().strip()}"
    )
    return hashlib.md5(basis.encode("utf-8")).hexdigest()