"""
CareerGPT — Discovery Agent Tools
====================================
PAGE SUMMARY:
  Concrete tool functions called by DiscoveryAgent.
  Each tool is a self-contained async function that does one thing well.
  Tools interact with: JobRepository, QdrantService, EmbeddingService,
  JobScrapers, NotificationService.

  TOOLS:
    search_jobs_semantic()      → embed query → Qdrant search → fetch DB jobs
    search_jobs_keyword()       → DB full-text + filter search
    trigger_scrape()            → fire Celery scraping tasks for specific queries
    rerank_with_llm()           → call LLM to re-rank semantic results
    score_opportunity()         → call LLM to score hidden opportunity quality
    get_already_applied_ids()   → fetch user's already-applied job IDs (exclude)
    get_market_insights()       → LLM analysis of job batch for market trends
    send_job_alert()            → send real-time notification for hot match
    save_agent_run()            → persist AgentRun record for audit/billing
    get_user_resume_vector()    → fetch user's master resume Qdrant vector
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import logger


async def search_jobs_semantic(
    db: AsyncSession,
    *,
    query_text: str,
    user_id: uuid.UUID,
    top_k: int = 30,
    is_remote: bool | None = None,
    sources: list[str] | None = None,
    experience_level: str | None = None,
    salary_min: int | None = None,
    exclude_job_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    """
    Embed query text → search Qdrant → fetch full Job objects from DB.
    Returns list of {semantic_score, job} dicts sorted by score descending.
    """
    from app.embeddings.embedder import get_embedder
    from app.repositories.job_repository import JobRepository
    from app.services.qdrant_service import get_qdrant_service

    embedder = get_embedder()
    qdrant = get_qdrant_service()
    job_repo = JobRepository(db)

    vector = await embedder.embed(query_text)

    qdrant_results = await qdrant.search_jobs_for_resume(
        vector,
        top_k=top_k,
        is_remote=is_remote,
        sources=sources,
        experience_level=experience_level,
        salary_min=salary_min,
        exclude_job_ids=exclude_job_ids or [],
    )

    if not qdrant_results:
        return []

    enriched = await job_repo.get_jobs_from_qdrant_results(qdrant_results)
    logger.debug(
        "Semantic job search",
        query=query_text[:50],
        results=len(enriched),
    )
    return enriched


async def search_jobs_keyword(
    db: AsyncSession,
    *,
    keyword: str | None = None,
    location: str | None = None,
    is_remote: bool | None = None,
    sources: list[str] | None = None,
    experience_level: str | None = None,
    salary_min: int | None = None,
    skills: list[str] | None = None,
    skip: int = 0,
    limit: int = 50,
) -> list[Any]:
    """Full-featured keyword search using JobRepository.search()."""
    from app.repositories.job_repository import JobRepository
    repo = JobRepository(db)
    return await repo.search(
        keyword=keyword,
        location=location,
        is_remote=is_remote,
        sources=sources,
        experience_level=experience_level,
        salary_min=salary_min,
        skills=skills,
        skip=skip,
        limit=limit,
    )


async def trigger_scrape(
    queries: list[dict[str, Any]],
    *,
    max_per_source: int = 100,
) -> list[str]:
    """
    Fire Celery scraping tasks for a list of queries.
    Returns list of Celery task IDs.
    Deduplicates by source+keyword to avoid redundant scrapes.
    """
    from app.workers.job_tasks import scrape_source_task

    seen: set[str] = set()
    task_ids: list[str] = []

    for q in queries:
        source = q.get("source", "all")
        keyword = q.get("keyword", "")
        location = q.get("location", "remote")
        dedup_key = f"{source}:{keyword}:{location}"

        if dedup_key in seen:
            continue
        seen.add(dedup_key)

        sources = ["linkedin", "indeed", "remoteok", "wellfound"] if source == "all" else [source]

        try:
            for src in sources:
                task = scrape_source_task.delay(
                    source=src,
                    keywords=[keyword],
                    locations=[location],
                    max_results=max_per_source,
                )
                task_ids.append(task.id)
        except Exception as exc:
            logger.warning(
                "Scrape task queue failed (non-critical)",
                source=source,
                keyword=keyword,
                error=str(exc),
            )

    logger.info(f"Queued {len(task_ids)} scraping tasks")
    return task_ids


async def get_already_applied_ids(
    db: AsyncSession,
    user_id: uuid.UUID,
) -> list[str]:
    """Fetch job IDs user has already applied to (for exclusion from results)."""
    from app.repositories.application_repository import ApplicationRepository
    repo = ApplicationRepository(db)
    apps = await repo.get_user_applications(user_id, limit=500)
    return [str(a.job_id) for a in apps]


async def get_user_resume_vector(
    db: AsyncSession,
    user_id: uuid.UUID,
) -> list[float] | None:
    """
    Get user's master resume embedding vector directly from Qdrant.
    Returns None if user has no master resume with a vector.
    """
    from app.repositories.resume_repository import ResumeRepository
    from app.services.qdrant_service import get_qdrant_service
    from app.embeddings.embedder import get_embedder

    repo = ResumeRepository(db)
    masters = await repo.get_master_resumes(user_id)

    if not masters:
        return None

    master = masters[0]

    if not master.raw_text:
        return None

    embedder = get_embedder()
    vector = await embedder.embed(master.raw_text[:4000])
    return vector


async def rerank_with_llm(
    jobs: list[dict[str, Any]],
    *,
    candidate_skills: list[str],
    experience_years: float,
    career_goal: str,
    prefers_remote: bool,
    min_salary: int | None,
) -> list[dict[str, Any]]:
    """
    Use LLM to re-rank semantic search results by true fit quality.
    Adds LLM-based ranking to the semantic score.
    """
    from app.agents.discovery_agent.prompts import DISCOVERY_RERANK
    from app.services.groq_service import get_groq_service

    if not jobs:
        return []

    llm = get_groq_service()

    jobs_summary = [
        {
            "job_id":          j["job"].id if hasattr(j.get("job"), "id") else j.get("job_id"),
            "title":           getattr(j.get("job"), "title", j.get("title", "")),
            "company":         getattr(j.get("job"), "company", j.get("company", "")),
            "semantic_score":  j.get("semantic_score", 0),
            "is_remote":       getattr(j.get("job"), "is_remote", False),
            "experience_level": getattr(j.get("job"), "experience_level", ""),
            "salary_min":      getattr(j.get("job"), "salary_min", None),
        }
        for j in jobs[:20]  # LLM context limit
    ]

    _, user_msg = DISCOVERY_RERANK.render(
        candidate_skills=", ".join(candidate_skills),
        experience_years=experience_years,
        career_goal=career_goal,
        prefers_remote=prefers_remote,
        min_salary=min_salary or 0,
        jobs_json=json.dumps(jobs_summary, indent=2),
    )

    try:
        result = await llm.complete_json(
            prompt=user_msg,
            system=DISCOVERY_RERANK.system,
            temperature=DISCOVERY_RERANK.temperature,
            max_tokens=DISCOVERY_RERANK.max_tokens,
        )

        ranked = result.get("ranked_jobs", [])
        skip_ids = set(result.get("jobs_to_skip", []))

        reranked = []
        rank_map = {str(r.get("job_id")): r for r in ranked}

        for job_dict in jobs:
            job_obj = job_dict.get("job")
            jid = str(getattr(job_obj, "id", ""))
            if jid in skip_ids:
                continue
            llm_rank = rank_map.get(jid, {})
            job_dict["llm_fit_score"]     = llm_rank.get("fit_score", job_dict.get("semantic_score", 0))
            job_dict["ranking_reason"]    = llm_rank.get("ranking_reason", "")
            job_dict["apply_urgency"]     = llm_rank.get("apply_urgency", "medium")
            job_dict["new_rank"]          = llm_rank.get("new_rank", 999)
            reranked.append(job_dict)

        reranked.sort(key=lambda x: x.get("new_rank", 999))
        return reranked

    except Exception as exc:
        logger.warning("LLM re-ranking failed, using semantic order", error=str(exc))
        return jobs


async def score_opportunity(
    job: Any,
) -> dict[str, Any]:
    """Score a single job's hidden opportunity quality using LLM."""
    from app.agents.discovery_agent.prompts import DISCOVERY_OPPORTUNITY_SCORE
    from app.services.groq_service import get_groq_service

    llm = get_groq_service()
    _, user_msg = DISCOVERY_OPPORTUNITY_SCORE.render(
        job_title=job.title,
        company_name=job.company,
        company_stage=job.ai_summary or "",
        job_description=(job.description or "")[:2000],
    )

    try:
        return await llm.complete_json(
            prompt=user_msg,
            system=DISCOVERY_OPPORTUNITY_SCORE.system,
            temperature=DISCOVERY_OPPORTUNITY_SCORE.temperature,
            max_tokens=DISCOVERY_OPPORTUNITY_SCORE.max_tokens,
        )
    except Exception as exc:
        logger.debug("Opportunity scoring failed", error=str(exc))
        return {"opportunity_score": 0.5}


async def get_market_insights(
    jobs: list[Any],
) -> dict[str, Any]:
    """Extract market intelligence from a batch of job objects."""
    from app.agents.discovery_agent.prompts import DISCOVERY_MARKET_INSIGHTS
    from app.services.groq_service import get_groq_service

    if not jobs:
        return {}

    llm = get_groq_service()
    all_skills: list[str] = []
    salaries_min: list[int] = []
    salaries_max: list[int] = []

    for job in jobs:
        if job.skills_required:
            try:
                skills = json.loads(job.skills_required)
                all_skills.extend(skills)
            except Exception:
                pass
        if job.salary_min:
            salaries_min.append(job.salary_min)
        if job.salary_max:
            salaries_max.append(job.salary_max)

    from collections import Counter
    skill_freq = Counter(all_skills).most_common(20)

    _, user_msg = DISCOVERY_MARKET_INSIGHTS.render(
        job_count=len(jobs),
        sample_titles=", ".join(j.title for j in jobs[:10]),
        sample_companies=", ".join(j.company for j in jobs[:10]),
        common_skills=", ".join(s for s, _ in skill_freq[:15]),
        salary_range=f"${min(salaries_min or [0]):,}-${max(salaries_max or [0]):,}",
        sources=", ".join(set(j.source for j in jobs)),
    )

    try:
        return await llm.complete_json(
            prompt=user_msg,
            system=DISCOVERY_MARKET_INSIGHTS.system,
            temperature=DISCOVERY_MARKET_INSIGHTS.temperature,
            max_tokens=DISCOVERY_MARKET_INSIGHTS.max_tokens,
        )
    except Exception as exc:
        logger.warning("Market insights generation failed", error=str(exc))
        return {}


async def check_alert_worthy(
    job: Any,
    *,
    user_skills: list[str],
    min_salary: int | None,
    remote_preference: str,
    experience_years: float,
    match_score: float,
) -> dict[str, Any]:
    """Decide if a job should trigger a real-time user notification."""
    from app.agents.discovery_agent.prompts import DISCOVERY_ALERT_FILTER
    from app.services.groq_service import get_groq_service

    if match_score < 0.75:
        return {"should_alert": False, "alert_reason": "Match score below threshold"}

    llm = get_groq_service()
    _, user_msg = DISCOVERY_ALERT_FILTER.render(
        user_skills=", ".join(user_skills),
        min_salary=min_salary or 0,
        remote_preference=remote_preference,
        experience_years=experience_years,
        job_title=job.title,
        company_name=job.company,
        source=job.source,
        job_salary=f"${job.salary_min or 0:,}-${job.salary_max or 0:,}",
        is_remote=job.is_remote,
        required_skills=job.skills_required or "",
        match_score=round(match_score, 2),
    )

    try:
        return await llm.complete_json(
            prompt=user_msg,
            system=DISCOVERY_ALERT_FILTER.system,
            temperature=DISCOVERY_ALERT_FILTER.temperature,
            max_tokens=DISCOVERY_ALERT_FILTER.max_tokens,
        )
    except Exception:
        return {"should_alert": match_score >= 0.88}


async def save_agent_run(
    db: AsyncSession,
    *,
    user_id: uuid.UUID,
    agent_type: str,
    status: str,
    input_data: dict,
    output_data: dict,
    duration_ms: float,
    llm_calls: int = 0,
) -> None:
    """Persist AgentRun record for audit trail and billing tracking."""
    try:
        from app.repositories.user_repository import BaseRepository
        from app.db.models.agent_run import AgentRun

        record = AgentRun(
            user_id=user_id,
            agent_type=agent_type,
            status=status,
            started_at=datetime.now(UTC),
            completed_at=datetime.now(UTC),
            duration_ms=duration_ms,
            input_data=json.dumps(input_data)[:5000],
            output_data=json.dumps(output_data)[:5000],
            llm_calls=llm_calls,
        )
        db.add(record)
        await db.flush()
    except Exception as exc:
        logger.warning("AgentRun save failed (non-critical)", error=str(exc))


__all__ = [
    "search_jobs_semantic",
    "search_jobs_keyword",
    "trigger_scrape",
    "get_already_applied_ids",
    "get_user_resume_vector",
    "rerank_with_llm",
    "score_opportunity",
    "get_market_insights",
    "check_alert_worthy",
    "save_agent_run",
]