"""
CareerGPT — Matching Agent
=============================
PAGE SUMMARY:
  The scoring brain of the platform. Every match_score the user sees —
  on the dashboard, application page, job list, and "best fit" rankings —
  is produced by this agent. Wraps the tool functions in tools.py with
  orchestration, caching, ORM hydration, and AgentRun audit logging.

  AGENT MODES:
    analyze()                 → full rich match analysis for ONE resume+job
                                 (used when user opens a specific job / creates an application)
    quick_score()              → fast single score, no gap detail (used in lists)
    bulk_score_jobs()          → score a resume against MANY jobs (job list page)
    find_best_resume_for_job() → from user's resume versions, which fits best?
    gap_analysis()             → deep skills gap + learning path (resume improvement page)
    rank_jobs()                → full multi-factor ranking for "best jobs for me"
    estimate_job_salary()      → salary estimate when JD has no explicit number
    explain_match()            → plain-English UI explanation for a match score

  CAREER FOCUS AWARENESS:
    Matching is role-agnostic by design — it works identically whether the
    user's target_roles are ["SAP Consultant", "SAP FICO"], ["AI Engineer",
    "ML Engineer", "LLM Engineer"], or any other title. The agent reads
    target_roles from the User profile (set in onboarding/preferences) and
    passes them into LLM prompts as career_goal context — no hardcoded
    role list anywhere in this codebase. Changing a user's target roles in
    Settings instantly changes what "good match" means for them, with zero
    code changes required.

  CACHING STRATEGY:
    Match scores are deterministic-ish but LLM calls are not free.
    A (resume_id, job_id) match result is cached in Redis for 24h.
    Cache invalidated automatically when resume is re-tailored (new resume_id).

  TWO-TIER SCORING:
    Tier 1 (instant, free):  Qdrant semantic cosine similarity — used for
                              ranking/sorting hundreds of jobs in <50ms
    Tier 2 (rich, LLM cost): Full gap analysis — used only when user opens
                              a specific job or creates an application
    This keeps the job list page fast and cheap while application pages
    get the full, expensive, high-quality analysis.

  AGENT RUN AUDIT:
    Every analyze() and gap_analysis() call is logged to AgentRun for
    billing/usage tracking and admin visibility into LLM cost per user.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.constants import AgentType
from app.core.logging import log_context, logger
from app.agents.matching_agent import tools

_MATCH_CACHE_TTL_SECS = 60 * 60 * 24  # 24 hours


class MatchingAgent:
    """
    Resume-to-job matching orchestrator.

    Usage:
        agent = MatchingAgent(db)
        result = await agent.analyze(resume=resume_obj, job=job_obj)
        # result["score"], result["strengths"], result["gaps"], ...
    """

    def __init__(self, db: AsyncSession) -> None:
        self.db = db
        self._llm_calls = 0

    # ── Mode 1: Full Rich Analysis ────────────────────────────────────────────

    async def analyze(
        self,
        *,
        resume: Any,
        job: Any,
        use_cache: bool = True,
    ) -> dict[str, Any]:
        """
        Full LLM match analysis between one resume and one job.
        This is what powers the Application creation flow and job detail page.

        Returns the complete MATCH_ANALYSIS_PROMPT JSON shape:
          score, rating, score_breakdown, strengths, gaps, critical_missing,
          ats_keywords_present/missing, recommendation, application_tips, etc.
        """
        t_start = time.monotonic()

        with log_context(agent="matching", resume_id=str(resume.id), job_id=str(job.id)):
            cache_key = self._cache_key(resume.id, job.id, "full")

            if use_cache:
                cached = await self._get_cached(cache_key)
                if cached:
                    logger.debug("Match analysis cache hit", cache_key=cache_key)
                    return cached

            if not resume.raw_text:
                logger.warning("Resume has no raw_text — cannot run match analysis")
                return self._empty_result(reason="resume_has_no_text")

            job_text = self._build_job_text(job)

            result = await tools.score_match(
                resume.raw_text[:4500],
                job_title=job.title,
                company_name=job.company,
                job_text=job_text,
                fast_mode=False,
            )
            self._llm_calls += 1

            # Enrich with deterministic structured comparisons (no LLM cost)
            result["experience_years_candidate"] = resume.experience_years
            result["education_candidate"]        = resume.education_level

            duration_ms = round((time.monotonic() - t_start) * 1000, 2)
            result["duration_ms"] = duration_ms

            if use_cache:
                await self._set_cached(cache_key, result)

            await self._save_agent_run(
                user_id=resume.user_id,
                status="success",
                input_data={"job_id": str(job.id), "resume_id": str(resume.id)},
                output_data={"score": result.get("score", 0)},
                duration_ms=duration_ms,
                related_job_id=job.id,
                related_resume_id=resume.id,
            )

            logger.info(
                "Match analysis complete",
                score=result.get("score"),
                rating=result.get("rating"),
                duration_ms=duration_ms,
            )
            return result

    # ── Mode 2: Quick Score (single, fast) ────────────────────────────────────

    async def quick_score(
        self,
        *,
        resume: Any,
        job: Any,
    ) -> dict[str, Any]:
        """
        Fast single score without full gap analysis.
        Used for: job list badges, inline preview cards.
        """
        if not resume.raw_text:
            return {"score": 0.0, "rating": "Unknown"}

        job_text = self._build_job_text(job, max_chars=1500)

        result = await tools.score_match(
            resume.raw_text[:2500],
            job_title=job.title,
            company_name=job.company,
            job_text=job_text,
            fast_mode=True,
        )
        self._llm_calls += 1
        return result

    # ── Mode 3: Bulk Score (resume vs many jobs) ──────────────────────────────

    async def bulk_score_jobs(
        self,
        *,
        resume: Any,
        jobs: list[Any],
        concurrency: int = 5,
        use_semantic_prefilter: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Score one resume against many jobs efficiently.

        Two-tier strategy:
          1. If use_semantic_prefilter: rank all jobs by Qdrant cosine score first
             (instant, free), then only LLM-score the top candidates
          2. Otherwise: LLM-score every job directly (slower, more accurate)

        Returns sorted list of {job_id, title, company, score, rating, top_gap}.
        """
        if not resume.raw_text or not jobs:
            return []

        with log_context(agent="matching_bulk", resume_id=str(resume.id), job_count=len(jobs)):
            jobs_to_score = jobs

            if use_semantic_prefilter and len(jobs) > 15:
                jobs_to_score = await self._semantic_prefilter(
                    resume=resume, jobs=jobs, top_n=15
                )
                logger.debug(
                    f"Semantic prefilter: {len(jobs)} → {len(jobs_to_score)} jobs for LLM scoring"
                )

            results = await tools.quick_score_batch(
                resume.raw_text[:2500],
                jobs_to_score,
                concurrency=concurrency,
            )
            self._llm_calls += len(jobs_to_score)

            logger.info(
                "Bulk scoring complete",
                jobs_scored=len(results),
                top_score=results[0]["score"] if results else 0,
            )
            return results

    # ── Mode 4: Find Best Resume for a Job ────────────────────────────────────

    async def find_best_resume_for_job(
        self,
        *,
        user_resumes: list[Any],
        job: Any,
    ) -> dict[str, Any]:
        """
        Given a user's multiple resume versions (master + tailored variants),
        determine which one best matches a specific job.
        Useful when a user has tailored resumes for similar past roles.
        """
        if not user_resumes:
            return {"best_resume_id": None, "reason": "no_resumes_available"}

        candidates = [r for r in user_resumes if r.raw_text]
        if not candidates:
            return {"best_resume_id": None, "reason": "no_resumes_with_text"}

        job_text = self._build_job_text(job, max_chars=1500)
        scored: list[tuple[Any, float]] = []

        for resume in candidates:
            try:
                result = await tools.score_match(
                    resume.raw_text[:2500],
                    job_title=job.title,
                    company_name=job.company,
                    job_text=job_text,
                    fast_mode=True,
                )
                self._llm_calls += 1
                scored.append((resume, result.get("score", 0.0)))
            except Exception as exc:
                logger.debug(f"Resume scoring failed for {resume.id}", error=str(exc))
                scored.append((resume, 0.0))

        scored.sort(key=lambda x: x[1], reverse=True)
        best_resume, best_score = scored[0]

        return {
            "best_resume_id":   str(best_resume.id),
            "best_resume_label": "master" if best_resume.is_master else "tailored",
            "score":            best_score,
            "all_scores": [
                {"resume_id": str(r.id), "is_master": r.is_master, "score": s}
                for r, s in scored
            ],
        }

    # ── Mode 5: Deep Skills Gap Analysis ──────────────────────────────────────

    async def gap_analysis(
        self,
        *,
        resume: Any,
        job: Any,
    ) -> dict[str, Any]:
        """
        Deep skills gap analysis with learning paths and timeline.
        Used on: resume improvement page, "what do I need to learn" feature.
        """
        candidate_skills: list[str] = []
        if resume.skills:
            try:
                candidate_skills = json.loads(resume.skills)
            except Exception:
                pass

        job_skills = await tools.extract_job_skills(
            f"{job.description or ''}\n{job.requirements or ''}"
        )
        self._llm_calls += 1

        required  = job_skills.get("required_skills", {})
        preferred = job_skills.get("preferred_skills", {})

        required_flat  = self._flatten_skill_dict(required)
        preferred_flat = self._flatten_skill_dict(preferred)

        result = await tools.analyze_skills_gap(
            candidate_skills,
            required_flat,
            preferred_flat,
            candidate_experience=resume.experience_years or 0.0,
            experience_required=float(job_skills.get("experience_years_required", 0) or 0),
            candidate_education=resume.education_level or "bachelor",
            education_required=job_skills.get("education_required", "bachelor"),
        )
        self._llm_calls += 1

        result["job_skills_extracted"] = job_skills
        return result

    # ── Mode 6: Rank Multiple Jobs for User ───────────────────────────────────

    async def rank_jobs(
        self,
        *,
        resume: Any,
        jobs: list[Any],
        career_goal: str,
        prefers_remote: bool = True,
        min_salary: int | None = None,
    ) -> dict[str, Any]:
        """
        Full multi-factor ranking considering career growth, not just skill match.
        career_goal comes from User.target_roles — works identically for any
        target career (SAP, AI/ML Engineering, DevOps, etc.) since it's just
        free-text context fed to the LLM, never a hardcoded category.
        """
        result = await tools.rank_jobs_for_user(
            jobs,
            resume,
            career_goal=career_goal,
            prefers_remote=prefers_remote,
            min_salary=min_salary,
        )
        self._llm_calls += 1
        return result

    # ── Mode 7: Salary Estimation ─────────────────────────────────────────────

    async def estimate_job_salary(self, job: Any) -> dict[str, Any]:
        """Estimate salary range when job posting has no explicit number."""
        if job.salary_min and job.salary_max:
            return {
                "salary_min_usd": job.salary_min,
                "salary_max_usd": job.salary_max,
                "confidence":     "high",
                "source":         "explicit_in_posting",
            }

        result = await tools.estimate_salary(
            job.title,
            job.ai_summary or "",
            job.location or "remote",
            is_remote=job.is_remote,
            seniority=job.experience_level or "mid",
            tech_stack=json.loads(job.skills_required) if job.skills_required else [],
            industry="technology",
        )
        self._llm_calls += 1
        return result

    # ── Mode 8: Plain-English Explanation ─────────────────────────────────────

    async def explain_match(
        self,
        *,
        candidate_name: str,
        job: Any,
        match_result: dict[str, Any],
    ) -> dict[str, Any]:
        """Generate a human-readable explanation of why a candidate matches a job."""
        result = await tools.generate_match_explanation(
            candidate_name,
            job.title,
            job.company,
            match_result.get("score", 0.5),
            match_result.get("strengths", []),
            match_result.get("gaps", []),
        )
        self._llm_calls += 1
        return result

    # ── Private: Semantic Prefiltering ────────────────────────────────────────

    async def _semantic_prefilter(
        self,
        *,
        resume: Any,
        jobs: list[Any],
        top_n: int,
    ) -> list[Any]:
        """
        Use free, instant embedding similarity to narrow down which jobs
        are worth spending LLM tokens on. Falls back to all jobs if
        embedding fails for any reason.
        """
        try:
            scored: list[tuple[Any, float]] = []
            for job in jobs:
                job_text = f"{job.title} {job.company} {(job.description or '')[:500]}"
                sim = await tools.get_semantic_similarity(
                    resume.raw_text[:2000], job_text
                )
                scored.append((job, sim))
            scored.sort(key=lambda x: x[1], reverse=True)
            return [j for j, _ in scored[:top_n]]
        except Exception as exc:
            logger.warning("Semantic prefilter failed, scoring all jobs", error=str(exc))
            return jobs[:top_n]

    # ── Private: Caching ───────────────────────────────────────────────────────

    def _cache_key(self, resume_id: uuid.UUID, job_id: uuid.UUID, mode: str) -> str:
        raw = f"match:{mode}:{resume_id}:{job_id}"
        return hashlib.sha256(raw.encode()).hexdigest()[:40]

    async def _get_cached(self, key: str) -> dict[str, Any] | None:
        try:
            import redis
            from app.core.config import get_settings
            settings = get_settings()
            r = redis.from_url(settings.redis.url_str, decode_responses=True, socket_connect_timeout=2)
            cached = r.get(f"matchcache:{key}")
            return json.loads(cached) if cached else None
        except Exception:
            return None

    async def _set_cached(self, key: str, value: dict[str, Any]) -> None:
        try:
            import redis
            from app.core.config import get_settings
            settings = get_settings()
            r = redis.from_url(settings.redis.url_str, decode_responses=True, socket_connect_timeout=2)
            r.setex(f"matchcache:{key}", _MATCH_CACHE_TTL_SECS, json.dumps(value, default=str))
        except Exception:
            pass

    # ── Private: Helpers ──────────────────────────────────────────────────────

    def _build_job_text(self, job: Any, max_chars: int = 3000) -> str:
        """Build combined job text for LLM matching prompts."""
        parts = [
            job.description or "",
            job.requirements or "",
        ]
        combined = "\n\n".join(p for p in parts if p)
        return combined[:max_chars]

    def _flatten_skill_dict(self, skill_dict: dict[str, Any]) -> list[str]:
        """Flatten a categorized skills dict (from extract_job_skills) into a flat list."""
        if isinstance(skill_dict, list):
            return skill_dict
        flat: list[str] = []
        for v in skill_dict.values():
            if isinstance(v, list):
                flat.extend(v)
        return flat

    def _empty_result(self, *, reason: str) -> dict[str, Any]:
        return {
            "score": 0.0,
            "rating": "Unknown",
            "strengths": [],
            "gaps": [],
            "recommendation": "Unable to analyze — " + reason,
            "_error_reason": reason,
        }

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
                agent_type=AgentType.MATCHING.value,
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


__all__ = ["MatchingAgent"]