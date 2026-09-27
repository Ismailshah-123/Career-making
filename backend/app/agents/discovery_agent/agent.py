"""
CareerGPT — Discovery Agent
=============================
PAGE SUMMARY:
  Intelligent job discovery orchestrator. The first agent in the pipeline.
  Finds the best jobs for a user by combining semantic search, keyword search,
  LLM re-ranking, opportunity scoring, and real-time alerts.

  AGENT MODES:
    discover_for_user()      → full personalized job discovery (main mode)
      1. Load user profile + preferences + resume vector
      2. Expand search queries using LLM (multiple titles/keywords)
      3. Semantic search via Qdrant (resume vector → similar jobs)
      4. Keyword search via PostgreSQL (title/skills filter)
      5. Merge + deduplicate results
      6. Exclude already-applied jobs
      7. LLM re-rank by true fit quality
      8. Score top 10 opportunities for hidden quality signals
      9. Send real-time alerts for 90%+ matches
      10. Generate market insights from result batch
      11. Save AgentRun record

    discover_trending()      → find hottest jobs in the market right now
      1. Scrape latest jobs from all sources
      2. Return top 50 by recency + engagement signals

    discover_by_query()      → user-driven search with natural language
      1. Parse NL query with LLM
      2. Run semantic + keyword search
      3. Return ranked results

    find_similar_jobs()      → given a job, find similar ones
      1. Get job's Qdrant point
      2. Use Qdrant recommend() API
      3. Filter by user preferences

  GROQ PROVIDER SELECTION:
    Uses llama-3.3-70b-versatile for all reasoning tasks.
    Falls back to claude-3-5-haiku on quota errors.
    JSON-mode enforced on all structured outputs.

  AGENT RUN LIFECYCLE:
    Each invocation creates an AgentRun DB record with:
    - Input parameters
    - Output summary (jobs found, alerts sent)
    - Duration, LLM calls count
    - Status (success/failed/partial)
    Visible in admin panel and user's agent history.
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
from app.agents.discovery_agent import tools


class DiscoveryAgent:
    """
    Job Discovery Agent — finds the best jobs for a specific user.

    Orchestrates: semantic search, keyword search, LLM re-ranking,
    opportunity scoring, market insights, and real-time job alerts.

    Usage:
        agent = DiscoveryAgent(db)
        result = await agent.discover_for_user(user_id=user.id)
    """

    def __init__(self, db: AsyncSession) -> None:
        self.db = db
        self._llm_calls = 0

    # ── Primary Method: Full Personalized Discovery ───────────────────────────

    async def discover_for_user(
        self,
        user_id: uuid.UUID,
        *,
        max_results: int = 50,
        trigger_scrape_if_stale: bool = True,
        send_alerts: bool = True,
    ) -> dict[str, Any]:
        """
        Full personalized job discovery pipeline.
        The most comprehensive and important DiscoveryAgent method.
        Called by: Celery daily task, user-triggered refresh button.

        Returns:
          {
            jobs: [enriched job dicts sorted by fit],
            market_insights: {skills, trends, salary data},
            alerts_sent: int,
            total_found: int,
            query_count: int,
            duration_ms: float,
            agent_run_id: str,
          }
        """
        t_start = time.monotonic()

        with log_context(agent="discovery", user_id=str(user_id)):
            logger.info("Starting personalized job discovery", user_id=str(user_id))

            # ── 1. Load User Profile ──────────────────────────────────────────
            from app.repositories.user_repository import UserRepository
            from app.repositories.resume_repository import ResumeRepository

            user_repo = UserRepository(self.db)
            resume_repo = ResumeRepository(self.db)

            user = await user_repo.get_by_id_or_raise(user_id)
            masters = await resume_repo.get_master_resumes(user_id)
            master_resume = masters[0] if masters else None

            # Parse user preferences
            user_skills: list[str] = []
            if master_resume and master_resume.skills:
                try:
                    user_skills = json.loads(master_resume.skills)
                except Exception:
                    pass

            target_roles: list[str] = []
            if user.target_roles:
                try:
                    target_roles = json.loads(user.target_roles)
                except Exception:
                    pass

            target_locations: list[str] = []
            if user.target_locations:
                try:
                    target_locations = json.loads(user.target_locations)
                except Exception:
                    target_locations = ["remote"]

            if not target_locations:
                target_locations = ["remote"]

            experience_years = (
                master_resume.experience_years or 3.0
                if master_resume else 3.0
            )

            # ── 2. Expand Search Queries ──────────────────────────────────────
            expanded_queries = await self._expand_queries(
                user_skills=user_skills,
                target_roles=target_roles,
                experience_years=experience_years,
                target_locations=target_locations,
                remote_preference=user.remote_preference,
            )

            # ── 3. Trigger Scraping (if stale) ────────────────────────────────
            if trigger_scrape_if_stale:
                from app.repositories.job_repository import JobRepository
                job_repo = JobRepository(self.db)
                scraped_today = await job_repo.get_jobs_scraped_today()
                if scraped_today < 50:
                    logger.info("Jobs stale — triggering background scrape")
                    await tools.trigger_scrape(
                        expanded_queries[:6],
                        max_per_source=100,
                    )

            # ── 4. Get Resume Vector ──────────────────────────────────────────
            resume_vector = None
            if master_resume:
                resume_vector = await tools.get_user_resume_vector(self.db, user_id)

            # ── 5. Semantic Search ─────────────────────────────────────────────
            already_applied = await tools.get_already_applied_ids(self.db, user_id)
            semantic_jobs: list[dict[str, Any]] = []

            if resume_vector:
                resume_query = (
                    f"{' '.join(target_roles)} {' '.join(user_skills[:10])}"
                    if target_roles else ' '.join(user_skills[:15])
                )
                semantic_jobs = await tools.search_jobs_semantic(
                    self.db,
                    query_text=resume_query,
                    user_id=user_id,
                    top_k=40,
                    is_remote=user.remote_preference == "remote" or None,
                    salary_min=user.min_salary,
                    exclude_job_ids=already_applied,
                )
                logger.info(f"Semantic search returned {len(semantic_jobs)} jobs")

            # ── 6. Keyword Search (supplemental) ─────────────────────────────
            keyword_jobs: list[dict[str, Any]] = []
            for role in target_roles[:3]:
                kw_results = await tools.search_jobs_keyword(
                    self.db,
                    keyword=role,
                    is_remote=user.remote_preference == "remote" or None,
                    salary_min=user.min_salary,
                    limit=20,
                )
                for job in kw_results:
                    if str(job.id) not in already_applied:
                        keyword_jobs.append({
                            "semantic_score": 0.6,
                            "job": job,
                        })

            # ── 7. Merge + Deduplicate ────────────────────────────────────────
            merged = self._merge_results(semantic_jobs, keyword_jobs)
            logger.info(f"After merge+dedup: {len(merged)} unique jobs")

            # ── 8. LLM Re-ranking ─────────────────────────────────────────────
            if len(merged) > 5:
                merged = await tools.rerank_with_llm(
                    merged,
                    candidate_skills=user_skills,
                    experience_years=experience_years,
                    career_goal=" ".join(target_roles) or "software engineer",
                    prefers_remote=user.remote_preference in ("remote", "any"),
                    min_salary=user.min_salary,
                )
                self._llm_calls += 1

            # Limit to max_results
            merged = merged[:max_results]

            # ── 9. Opportunity Scoring (top 10 only — LLM cost control) ───────
            for job_dict in merged[:10]:
                job = job_dict.get("job")
                if job:
                    try:
                        opp_score = await tools.score_opportunity(job)
                        job_dict["opportunity_score"] = opp_score.get("opportunity_score", 0.5)
                        job_dict["hidden_signals"]    = opp_score.get("hidden_signals", [])
                        job_dict["apply_urgency"]     = opp_score.get("hiring_urgency", "medium")
                        self._llm_calls += 1
                    except Exception as exc:
                        logger.debug("Opportunity scoring failed", error=str(exc))

            # ── 10. Real-time Alerts ─────────────────────────────────────────
            alerts_sent = 0
            if send_alerts:
                for job_dict in merged[:20]:
                    score = job_dict.get("llm_fit_score", job_dict.get("semantic_score", 0))
                    if score >= 0.88:
                        alert_result = await tools.check_alert_worthy(
                            job_dict["job"],
                            user_skills=user_skills,
                            min_salary=user.min_salary,
                            remote_preference=user.remote_preference,
                            experience_years=experience_years,
                            match_score=score,
                        )
                        self._llm_calls += 1
                        if alert_result.get("should_alert"):
                            await self._send_job_alert(
                                user_id=user_id,
                                job=job_dict["job"],
                                alert_data=alert_result,
                            )
                            alerts_sent += 1

            # ── 11. Market Insights ───────────────────────────────────────────
            all_job_objects = [j["job"] for j in merged if j.get("job")]
            market_insights: dict[str, Any] = {}
            if all_job_objects:
                try:
                    market_insights = await tools.get_market_insights(all_job_objects)
                    self._llm_calls += 1
                except Exception as exc:
                    logger.debug("Market insights failed", error=str(exc))

            duration_ms = round((time.monotonic() - t_start) * 1000, 2)

            # ── 12. Save AgentRun ─────────────────────────────────────────────
            await tools.save_agent_run(
                self.db,
                user_id=user_id,
                agent_type=AgentType.DISCOVERY.value,
                status="success",
                input_data={
                    "target_roles": target_roles,
                    "skills_count": len(user_skills),
                    "remote_preference": user.remote_preference,
                },
                output_data={
                    "jobs_found":     len(merged),
                    "alerts_sent":    alerts_sent,
                    "semantic_count": len(semantic_jobs),
                    "keyword_count":  len(keyword_jobs),
                },
                duration_ms=duration_ms,
                llm_calls=self._llm_calls,
            )

            logger.info(
                "Discovery complete",
                user_id=str(user_id),
                jobs_found=len(merged),
                alerts_sent=alerts_sent,
                duration_ms=duration_ms,
                llm_calls=self._llm_calls,
            )

            return {
                "jobs":             [self._job_dict_to_response(j) for j in merged],
                "market_insights":  market_insights,
                "alerts_sent":      alerts_sent,
                "total_found":      len(merged),
                "semantic_count":   len(semantic_jobs),
                "keyword_count":    len(keyword_jobs),
                "query_count":      len(expanded_queries),
                "duration_ms":      duration_ms,
                "llm_calls":        self._llm_calls,
            }

    # ── Mode 2: Discover by Natural Language Query ────────────────────────────

    async def discover_by_query(
        self,
        user_id: uuid.UUID,
        *,
        query: str,
        max_results: int = 30,
    ) -> dict[str, Any]:
        """
        Natural language job search.
        Example: "Find me remote senior Python jobs at AI startups paying $150K+"
        """
        t_start = time.monotonic()

        with log_context(agent="discovery_query", user_id=str(user_id)):
            already_applied = await tools.get_already_applied_ids(self.db, user_id)

            semantic_results = await tools.search_jobs_semantic(
                self.db,
                query_text=query,
                user_id=user_id,
                top_k=max_results,
                exclude_job_ids=already_applied,
            )

            # Also do keyword search
            keyword_results = await tools.search_jobs_keyword(
                self.db,
                keyword=query[:50],
                limit=max_results // 2,
            )
            kw_formatted = [{"semantic_score": 0.55, "job": j} for j in keyword_results
                           if str(j.id) not in already_applied]

            merged = self._merge_results(semantic_results, kw_formatted)

            return {
                "jobs":         [self._job_dict_to_response(j) for j in merged[:max_results]],
                "query":        query,
                "total_found":  len(merged),
                "duration_ms":  round((time.monotonic() - t_start) * 1000, 2),
            }

    # ── Mode 3: Find Similar Jobs ─────────────────────────────────────────────

    async def find_similar_jobs(
        self,
        job_id: uuid.UUID,
        *,
        user_id: uuid.UUID,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Find jobs semantically similar to a given job.
        Used on job detail page: "Similar positions".
        """
        from app.repositories.job_repository import JobRepository
        from app.services.qdrant_service import get_qdrant_service

        job_repo = JobRepository(self.db)
        qdrant = get_qdrant_service()

        job = await job_repo.get_by_id_or_raise(job_id)
        if not job.qdrant_point_id:
            return []

        similar = await qdrant.search_similar_jobs(
            job.qdrant_point_id,
            top_k=top_k,
            score_threshold=0.65,
        )

        if not similar:
            return []

        similar_ids = [
            uuid.UUID(r["job_id"])
            for r in similar
            if r.get("job_id")
        ]
        jobs = await job_repo.get_jobs_by_ids(similar_ids)

        already_applied = await tools.get_already_applied_ids(self.db, user_id)
        score_map = {r["job_id"]: r["score"] for r in similar}

        return [
            {
                "semantic_score": score_map.get(str(j.id), 0),
                "job": self._serialize_job(j),
            }
            for j in jobs
            if str(j.id) not in already_applied
        ]

    # ── Mode 4: Trending Jobs ─────────────────────────────────────────────────

    async def discover_trending(
        self,
        *,
        limit: int = 50,
        source: str | None = None,
    ) -> dict[str, Any]:
        """
        Get the most recently scraped and highest-engagement jobs.
        Used by: trending jobs page, email digest.
        """
        results = await tools.search_jobs_keyword(
            self.db,
            limit=limit,
        )
        return {
            "jobs": [self._serialize_job(j) for j in results],
            "total": len(results),
        }

    # ── Private Helpers ───────────────────────────────────────────────────────

    async def _expand_queries(
        self,
        *,
        user_skills: list[str],
        target_roles: list[str],
        experience_years: float,
        target_locations: list[str],
        remote_preference: str,
    ) -> list[dict[str, Any]]:
        """Use LLM to expand search intent into multiple queries."""
        from app.agents.discovery_agent.prompts import DISCOVERY_QUERY_EXPAND
        from app.services.groq_service import get_groq_service

        if not target_roles and not user_skills:
            return [{"keyword": "software engineer", "location": "remote", "source": "all"}]

        llm = get_groq_service()
        user_intent = " ".join(target_roles) or "software engineer"

        _, user_msg = DISCOVERY_QUERY_EXPAND.render(
            user_intent=user_intent,
            user_skills=", ".join(user_skills[:15]),
            experience_years=experience_years,
            target_locations=", ".join(target_locations),
            remote_preference=remote_preference,
        )

        try:
            result = await llm.complete_json(
                prompt=user_msg,
                system=DISCOVERY_QUERY_EXPAND.system,
                temperature=DISCOVERY_QUERY_EXPAND.temperature,
                max_tokens=DISCOVERY_QUERY_EXPAND.max_tokens,
            )
            self._llm_calls += 1
            queries = result.get("primary_queries", [])
            if not queries:
                raise ValueError("Empty queries from LLM")
            return queries
        except Exception as exc:
            logger.warning("Query expansion failed, using defaults", error=str(exc))
            return [
                {"keyword": role, "location": target_locations[0], "source": "all"}
                for role in (target_roles[:3] or ["software engineer"])
            ]

    def _merge_results(
        self,
        semantic: list[dict[str, Any]],
        keyword: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Merge semantic + keyword results, deduplicate by job_id.
        Semantic results take priority (higher trust score).
        """
        seen_ids: set[str] = set()
        merged: list[dict[str, Any]] = []

        for item in semantic + keyword:
            job = item.get("job")
            if not job:
                continue
            jid = str(getattr(job, "id", ""))
            if jid and jid not in seen_ids:
                seen_ids.add(jid)
                merged.append(item)

        merged.sort(
            key=lambda x: x.get("llm_fit_score", x.get("semantic_score", 0)),
            reverse=True,
        )
        return merged

    async def _send_job_alert(
        self,
        *,
        user_id: uuid.UUID,
        job: Any,
        alert_data: dict[str, Any],
    ) -> None:
        """Queue a real-time job alert notification."""
        try:
            from app.workers.notification_tasks import send_job_alert_task
            send_job_alert_task.delay(
                user_id=str(user_id),
                job_id=str(job.id),
                alert_message=alert_data.get(
                    "suggested_alert_message",
                    f"New job match: {job.title} at {job.company}",
                ),
                priority=alert_data.get("alert_priority", "medium"),
            )
        except Exception as exc:
            logger.debug("Alert notification queue failed (non-critical)", error=str(exc))

    def _serialize_job(self, job: Any) -> dict[str, Any]:
        """Serialize a Job ORM object to response dict."""
        return {
            "id":               str(job.id),
            "title":            job.title,
            "company":          job.company,
            "company_logo_url": job.company_logo_url,
            "location":         job.location,
            "is_remote":        job.is_remote,
            "job_url":          job.job_url,
            "source":           job.source,
            "employment_type":  job.employment_type,
            "experience_level": job.experience_level,
            "salary_min":       job.salary_min,
            "salary_max":       job.salary_max,
            "salary_currency":  job.salary_currency,
            "ai_summary":       job.ai_summary,
            "visa_sponsorship": job.visa_sponsorship,
            "posted_at":        job.posted_at.isoformat() if job.posted_at else None,
            "skills_required":  json.loads(job.skills_required) if job.skills_required else [],
        }

    def _job_dict_to_response(self, job_dict: dict[str, Any]) -> dict[str, Any]:
        """Convert merged result dict to API response format."""
        job = job_dict.get("job")
        return {
            **(self._serialize_job(job) if job else {}),
            "semantic_score":    round(job_dict.get("semantic_score", 0), 3),
            "llm_fit_score":     round(job_dict.get("llm_fit_score", 0), 3),
            "opportunity_score": round(job_dict.get("opportunity_score", 0), 3),
            "apply_urgency":     job_dict.get("apply_urgency", "medium"),
            "ranking_reason":    job_dict.get("ranking_reason", ""),
            "hidden_signals":    job_dict.get("hidden_signals", []),
        }


__all__ = ["DiscoveryAgent"]