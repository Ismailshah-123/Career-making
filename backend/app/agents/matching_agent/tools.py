"""
CareerGPT -- Matching Agent Tools
====================================
PAGE SUMMARY:
  Tool functions for MatchingAgent. Each tool is a focused, testable
  async function that calls one LLM prompt (or a cheap embedding call)
  and returns a structured dict. The agent orchestrates these -- tools
  never touch the DB.

  TOOLS:
    score_match()               -> full or fast resume<->job score
    quick_score_batch()         -> score one resume against many jobs concurrently
    extract_job_skills()        -> structured required/preferred skills off a posting
    analyze_skills_gap()        -> candidate vs. job requirements gap analysis
    rank_jobs_for_user()        -> multi-factor ranking of several jobs
    estimate_salary()           -> pay range estimate when a posting has none
    generate_match_explanation() -> plain-English blurb for a match score
    get_semantic_similarity()   -> free/instant cosine similarity via embeddings

  USED BY: MatchingAgent (agent.py)
"""

from __future__ import annotations

import asyncio
import math
from typing import Any

from app.core.logging import logger


# ══════════════════════════════════════════════════════════════════════════════
# Scoring
# ══════════════════════════════════════════════════════════════════════════════

async def score_match(
    resume_text: str,
    *,
    job_title: str,
    company_name: str,
    job_text: str,
    fast_mode: bool = False,
) -> dict[str, Any]:
    """Score how well a resume matches a job. fast_mode uses a cheaper/shorter prompt."""
    from app.agents.matching_agent.prompts import MATCH_ANALYSIS_PROMPT, MATCH_QUICK_SCORE_PROMPT
    from app.services.groq_service import get_groq_service

    prompt = MATCH_QUICK_SCORE_PROMPT if fast_mode else MATCH_ANALYSIS_PROMPT
    llm = get_groq_service()
    system, user_msg = prompt.render(
        resume_text=resume_text,
        job_title=job_title or "",
        company_name=company_name or "",
        job_text=job_text or "",
    )

    try:
        result = await llm.complete_json(
            user_msg, system=system, temperature=prompt.temperature, max_tokens=prompt.max_tokens,
        )
        result.setdefault("score", 0.5)
        result.setdefault("rating", "Fair Match")
        return result
    except Exception as exc:
        logger.warning("score_match failed, returning fallback", error=str(exc))
        return {"score": 0.5, "rating": "Unknown", "strengths": [], "gaps": [], "error": str(exc)[:200]}


async def quick_score_batch(
    resume_text: str,
    jobs: list[Any],
    *,
    concurrency: int = 5,
) -> list[dict[str, Any]]:
    """Fast-score one resume against many jobs concurrently, sorted best-first."""
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def _score_one(job: Any) -> dict[str, Any]:
        async with semaphore:
            job_text = f"{(job.description or '')[:1200]}"
            result = await score_match(
                resume_text,
                job_title=job.title,
                company_name=job.company,
                job_text=job_text,
                fast_mode=True,
            )
            return {
                "job_id": str(job.id),
                "title": job.title,
                "company": job.company,
                "score": result.get("score", 0.0),
                "rating": result.get("rating", "Unknown"),
                "top_gap": result.get("top_gap", ""),
            }

    results = await asyncio.gather(*(_score_one(j) for j in jobs), return_exceptions=False)
    results.sort(key=lambda r: r.get("score", 0.0), reverse=True)
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Skills Extraction / Gap Analysis
# ══════════════════════════════════════════════════════════════════════════════

async def extract_job_skills(job_text: str) -> dict[str, Any]:
    """Extract structured required/preferred skills from raw job posting text."""
    from app.agents.matching_agent.prompts import JOB_SKILLS_EXTRACT_PROMPT
    from app.services.groq_service import get_groq_service

    llm = get_groq_service()
    system, user_msg = JOB_SKILLS_EXTRACT_PROMPT.render(job_text=(job_text or "")[:6000])

    try:
        result = await llm.complete_json(
            user_msg, system=system,
            temperature=JOB_SKILLS_EXTRACT_PROMPT.temperature,
            max_tokens=JOB_SKILLS_EXTRACT_PROMPT.max_tokens,
        )
        result.setdefault("required_skills", {})
        result.setdefault("preferred_skills", {})
        return result
    except Exception as exc:
        logger.warning("extract_job_skills failed", error=str(exc))
        return {"required_skills": {}, "preferred_skills": {}, "experience_years_required": 0, "education_required": "bachelor"}


async def analyze_skills_gap(
    candidate_skills: list[str],
    required_skills: list[str],
    preferred_skills: list[str],
    *,
    candidate_experience: float,
    experience_required: float,
    candidate_education: str,
    education_required: str,
) -> dict[str, Any]:
    """Compare a candidate's skills/experience/education against a job's requirements."""
    from app.agents.matching_agent.prompts import SKILLS_GAP_ANALYZE_PROMPT
    from app.services.groq_service import get_groq_service

    llm = get_groq_service()
    system, user_msg = SKILLS_GAP_ANALYZE_PROMPT.render(
        candidate_skills=", ".join(candidate_skills),
        candidate_experience=candidate_experience,
        candidate_education=candidate_education,
        required_skills=", ".join(required_skills),
        preferred_skills=", ".join(preferred_skills),
        experience_required=experience_required,
        education_required=education_required,
    )

    try:
        return await llm.complete_json(
            user_msg, system=system,
            temperature=SKILLS_GAP_ANALYZE_PROMPT.temperature,
            max_tokens=SKILLS_GAP_ANALYZE_PROMPT.max_tokens,
        )
    except Exception as exc:
        logger.warning("analyze_skills_gap failed", error=str(exc))
        overlap = set(s.lower() for s in candidate_skills) & set(s.lower() for s in required_skills)
        pct = (len(overlap) / len(required_skills)) if required_skills else 0.5
        return {
            "skills_match_pct": round(pct, 2),
            "matched_required": list(overlap),
            "missing_required": [s for s in required_skills if s.lower() not in overlap],
            "matched_preferred": [],
            "missing_preferred": preferred_skills,
            "experience_gap_years": max(0.0, experience_required - candidate_experience),
            "education_meets_requirement": True,
            "overall_readiness": "close",
            "closing_the_gap": [],
        }


# ══════════════════════════════════════════════════════════════════════════════
# Ranking / Salary / Explanation
# ══════════════════════════════════════════════════════════════════════════════

async def rank_jobs_for_user(
    jobs: list[Any],
    resume: Any,
    *,
    career_goal: str,
    prefers_remote: bool = True,
    min_salary: int | None = None,
) -> dict[str, Any]:
    """Multi-factor ranking of several jobs for one candidate's stated career goal."""
    from app.agents.matching_agent.prompts import JOB_RANK_PROMPT
    from app.services.groq_service import get_groq_service

    jobs_text = "\n".join(
        f"- id={job.id} | {job.title} at {job.company} | remote={job.is_remote} | "
        f"salary={job.salary_min}-{job.salary_max}"
        for job in jobs
    )

    llm = get_groq_service()
    system, user_msg = JOB_RANK_PROMPT.render(
        resume_text=(resume.raw_text or "")[:3000],
        career_goal=career_goal or "",
        prefers_remote=prefers_remote,
        min_salary=min_salary or 0,
        jobs_text=jobs_text,
    )

    try:
        result = await llm.complete_json(
            user_msg, system=system,
            temperature=JOB_RANK_PROMPT.temperature,
            max_tokens=JOB_RANK_PROMPT.max_tokens,
        )
        result.setdefault("ranked_job_ids", [str(j.id) for j in jobs])
        result.setdefault("rankings", [])
        return result
    except Exception as exc:
        logger.warning("rank_jobs_for_user failed, returning input order", error=str(exc))
        return {
            "ranked_job_ids": [str(j.id) for j in jobs],
            "rankings": [{"job_id": str(j.id), "rank": i + 1, "score": 0.5, "reasoning": ""} for i, j in enumerate(jobs)],
        }


async def estimate_salary(
    job_title: str,
    job_summary: str,
    location: str,
    *,
    is_remote: bool,
    seniority: str,
    tech_stack: list[str],
    industry: str = "technology",
) -> dict[str, Any]:
    """Estimate a USD salary range when a job posting doesn't list one."""
    from app.agents.matching_agent.prompts import SALARY_ESTIMATE_PROMPT
    from app.services.groq_service import get_groq_service

    llm = get_groq_service()
    system, user_msg = SALARY_ESTIMATE_PROMPT.render(
        job_title=job_title or "",
        job_summary=job_summary or "",
        location=location or "remote",
        is_remote=is_remote,
        seniority=seniority or "mid",
        tech_stack=", ".join(tech_stack),
        industry=industry,
    )

    try:
        result = await llm.complete_json(
            user_msg, system=system,
            temperature=SALARY_ESTIMATE_PROMPT.temperature,
            max_tokens=SALARY_ESTIMATE_PROMPT.max_tokens,
        )
        result.setdefault("confidence", "low")
        result.setdefault("source", "llm_estimate")
        return result
    except Exception as exc:
        logger.warning("estimate_salary failed", error=str(exc))
        return {"salary_min_usd": None, "salary_max_usd": None, "confidence": "low", "source": "unavailable"}


async def generate_match_explanation(
    candidate_name: str,
    job_title: str,
    company_name: str,
    score: float,
    strengths: list[str],
    gaps: list[str],
) -> dict[str, Any]:
    """Turn a raw match score + strengths/gaps into a short human-readable blurb."""
    from app.agents.matching_agent.prompts import MATCH_EXPLANATION_PROMPT
    from app.services.groq_service import get_groq_service

    llm = get_groq_service()
    system, user_msg = MATCH_EXPLANATION_PROMPT.render(
        candidate_name=candidate_name or "You",
        job_title=job_title or "",
        company_name=company_name or "",
        score=score,
        strengths=", ".join(strengths or []),
        gaps=", ".join(gaps or []),
    )

    try:
        return await llm.complete_json(
            user_msg, system=system,
            temperature=MATCH_EXPLANATION_PROMPT.temperature,
            max_tokens=MATCH_EXPLANATION_PROMPT.max_tokens,
        )
    except Exception as exc:
        logger.warning("generate_match_explanation failed", error=str(exc))
        return {"explanation": f"You're a {round(score * 100)}% match for {job_title} at {company_name}."}


# ══════════════════════════════════════════════════════════════════════════════
# Semantic Similarity (embeddings, no LLM cost)
# ══════════════════════════════════════════════════════════════════════════════

async def get_semantic_similarity(text_a: str, text_b: str) -> float:
    """Cosine similarity between two texts' embeddings. Free/instant, no LLM call."""
    from app.embeddings.embedder import get_embedder

    embedder = get_embedder()
    vec_a, vec_b = await asyncio.gather(embedder.embed(text_a), embedder.embed(text_b))

    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


__all__ = [
    "score_match",
    "quick_score_batch",
    "extract_job_skills",
    "analyze_skills_gap",
    "rank_jobs_for_user",
    "estimate_salary",
    "generate_match_explanation",
    "get_semantic_similarity",
]
