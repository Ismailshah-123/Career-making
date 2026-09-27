"""
CareerGPT — Job Repository
============================
PAGE SUMMARY:
  Data access layer for Job model. All job-related SQL lives here.
  Services and agents NEVER write raw SQL — they call methods here.

  CORE METHODS:
    get_by_id, get_by_id_or_raise    → single job fetch
    get_by_source_id                 → deduplication during scraping
    search()                         → full filter+pagination search (main job board)
    semantic_ids_to_jobs()           → convert Qdrant point payloads → Job objects
    count_by_source()                → dashboard stats
    get_recent()                     → latest scraped jobs
    get_active_for_source()          → per-source job list
    bulk_create()                    → batch insert for scrapers (fast)
    deactivate_old_jobs()            → nightly cleanup of stale listings
    upsert_job()                     → create or update on re-scrape
    mark_jobs_inactive_by_company()  → when company page returns 404
    get_company_distribution()       → analytics: top companies
    get_top_skills()                 → analytics: most requested skills
    get_salary_range_stats()         → analytics: salary distribution
    full_text_search()               → PostgreSQL tsvector search

  SEARCH FILTERS (search() method):
    keyword         → ilike on title + company + description
    location        → ilike on location
    is_remote       → boolean filter
    sources         → list of job board sources
    experience_level→ entry/mid/senior/lead/executive
    employment_type → full-time/contract/part-time/freelance
    salary_min      → salary_max >= value (job pays at least this)
    salary_max      → salary_min <= value (job is affordable)
    posted_after    → filter by posting date
    skills          → any of these skills in skills_required
    company         → exact or fuzzy company match
    visa_sponsorship→ boolean flag

  PERFORMANCE:
    - Source+external_id unique index for O(1) dedup
    - Company+title composite index for company search
    - Remote+active composite index for remote job filter
    - Posted_at index for date range queries
    - Full text search via PostgreSQL tsvector (created by migration)
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import and_, desc, func, or_, select, text, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import logger
from app.db.models.job import Job
from app.repositories.user_repository import BaseRepository


class JobRepository(BaseRepository):
    model = Job

    # ── Single Fetch ──────────────────────────────────────────────────────────

    async def get_by_source_id(
        self,
        source: str,
        external_id: str,
    ) -> Job | None:
        """
        Fetch job by source + external_id pair.
        Used by scrapers for deduplication before insert.
        O(1) via unique composite index on (source, external_id).
        """
        q = (
            self._base_query()
            .where(Job.source == source)
            .where(Job.external_id == external_id)
        )
        result = await self.session.execute(q)
        return result.scalar_one_or_none()

    async def get_by_url_hash(self, job_url: str) -> Job | None:
        """Fetch job by exact URL match (fallback deduplication)."""
        q = self._base_query().where(Job.job_url == job_url)
        result = await self.session.execute(q.limit(1))
        return result.scalar_one_or_none()

    # ── Search (main endpoint) ────────────────────────────────────────────────

    async def search(
        self,
        *,
        keyword: str | None = None,
        location: str | None = None,
        is_remote: bool | None = None,
        sources: list[str] | None = None,
        experience_level: str | None = None,
        employment_type: str | None = None,
        salary_min: int | None = None,
        salary_max: int | None = None,
        posted_after: datetime | None = None,
        company: str | None = None,
        visa_sponsorship: bool | None = None,
        skills: list[str] | None = None,
        skip: int = 0,
        limit: int = 50,
        order_by_recent: bool = True,
    ) -> list[Job]:
        """
        Full-featured job search with all filter combinations.
        Used by: GET /api/v1/jobs endpoint.

        Returns paginated list of active, non-deleted jobs.
        Filters are all optional and combinable.
        """
        q = (
            self._base_query()
            .where(Job.is_active.is_(True))
        )

        # ── Text search ───────────────────────────────────────────────────────
        if keyword:
            kw = f"%{keyword.lower()}%"
            q = q.where(
                or_(
                    Job.title.ilike(kw),
                    Job.company.ilike(kw),
                    Job.description.ilike(kw),
                    Job.skills_required.ilike(kw),
                )
            )

        # ── Location ──────────────────────────────────────────────────────────
        if location:
            q = q.where(Job.location.ilike(f"%{location}%"))

        # ── Remote ────────────────────────────────────────────────────────────
        if is_remote is not None:
            q = q.where(Job.is_remote.is_(is_remote))

        # ── Sources ───────────────────────────────────────────────────────────
        if sources:
            q = q.where(Job.source.in_(sources))

        # ── Experience level ──────────────────────────────────────────────────
        if experience_level:
            q = q.where(Job.experience_level == experience_level)

        # ── Employment type ───────────────────────────────────────────────────
        if employment_type:
            q = q.where(Job.employment_type == employment_type)

        # ── Salary range ──────────────────────────────────────────────────────
        if salary_min is not None:
            q = q.where(
                or_(
                    Job.salary_max >= salary_min,
                    Job.salary_min >= salary_min,
                )
            )
        if salary_max is not None:
            q = q.where(
                or_(
                    Job.salary_min.is_(None),
                    Job.salary_min <= salary_max,
                )
            )

        # ── Posted after ──────────────────────────────────────────────────────
        if posted_after:
            q = q.where(Job.posted_at >= posted_after)

        # ── Company ───────────────────────────────────────────────────────────
        if company:
            q = q.where(Job.company.ilike(f"%{company}%"))

        # ── Visa sponsorship ──────────────────────────────────────────────────
        if visa_sponsorship is not None:
            q = q.where(Job.visa_sponsorship.is_(visa_sponsorship))

        # ── Skills (any match) ────────────────────────────────────────────────
        if skills:
            skill_conditions = [
                Job.skills_required.ilike(f"%{skill}%")
                for skill in skills
            ]
            q = q.where(or_(*skill_conditions))

        # ── Ordering ──────────────────────────────────────────────────────────
        if order_by_recent:
            q = q.order_by(
                desc(Job.posted_at).nullslast(),
                desc(Job.created_at),
            )

        q = q.offset(skip).limit(limit)
        result = await self.session.execute(q)
        return list(result.scalars().all())

    async def search_count(
        self,
        *,
        keyword: str | None = None,
        location: str | None = None,
        is_remote: bool | None = None,
        sources: list[str] | None = None,
        experience_level: str | None = None,
        employment_type: str | None = None,
        salary_min: int | None = None,
        company: str | None = None,
    ) -> int:
        """Count results for the same filters as search() — for pagination."""
        q = (
            select(func.count())
            .select_from(Job)
            .where(Job.is_deleted.is_(False))
            .where(Job.is_active.is_(True))
        )
        if keyword:
            kw = f"%{keyword.lower()}%"
            q = q.where(or_(Job.title.ilike(kw), Job.company.ilike(kw)))
        if location:
            q = q.where(Job.location.ilike(f"%{location}%"))
        if is_remote is not None:
            q = q.where(Job.is_remote.is_(is_remote))
        if sources:
            q = q.where(Job.source.in_(sources))
        if experience_level:
            q = q.where(Job.experience_level == experience_level)
        if employment_type:
            q = q.where(Job.employment_type == employment_type)
        if salary_min is not None:
            q = q.where(or_(Job.salary_max >= salary_min, Job.salary_min >= salary_min))
        if company:
            q = q.where(Job.company.ilike(f"%{company}%"))
        result = await self.session.execute(q)
        return result.scalar_one() or 0

    async def full_text_search(
        self,
        query: str,
        *,
        skip: int = 0,
        limit: int = 20,
    ) -> list[Job]:
        """
        PostgreSQL full-text search using tsvector.
        Falls back to ILIKE search if tsvector index doesn't exist.
        Requires migration: CREATE INDEX ix_jobs_fts ON jobs USING gin(search_vector).
        """
        try:
            fts_q = text("""
                SELECT id FROM jobs
                WHERE is_deleted = FALSE AND is_active = TRUE
                  AND search_vector @@ plainto_tsquery('english', :query)
                ORDER BY ts_rank(search_vector, plainto_tsquery('english', :query)) DESC
                LIMIT :limit OFFSET :skip
            """).bindparams(query=query, limit=limit, skip=skip)

            result = await self.session.execute(fts_q)
            ids = [row[0] for row in result.all()]

            if not ids:
                return []

            jobs_q = (
                self._base_query()
                .where(Job.id.in_(ids))
            )
            jobs_result = await self.session.execute(jobs_q)
            jobs_by_id = {j.id: j for j in jobs_result.scalars().all()}
            return [jobs_by_id[i] for i in ids if i in jobs_by_id]

        except Exception:
            # Fallback to ILIKE
            return await self.search(keyword=query, skip=skip, limit=limit)

    # ── Semantic Search Bridge ────────────────────────────────────────────────

    async def get_jobs_by_ids(self, job_ids: list[uuid.UUID]) -> list[Job]:
        """
        Fetch multiple jobs by ID list.
        Used after Qdrant semantic search returns job_ids from payload.
        Preserves semantic score order from Qdrant.
        """
        if not job_ids:
            return []

        q = (
            self._base_query()
            .where(Job.id.in_(job_ids))
            .where(Job.is_active.is_(True))
        )
        result = await self.session.execute(q)
        jobs_by_id = {j.id: j for j in result.scalars().all()}

        # Return in the same order as job_ids (Qdrant score order)
        return [jobs_by_id[jid] for jid in job_ids if jid in jobs_by_id]

    async def get_jobs_from_qdrant_results(
        self,
        qdrant_results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Convert Qdrant search results → enriched dicts with full Job data.
        Qdrant result: {point_id, score, job_id, title, company, ...}
        Returns combined dict with score + full DB job data.
        """
        if not qdrant_results:
            return []

        job_ids_str = [r.get("job_id") for r in qdrant_results if r.get("job_id")]
        job_ids = [uuid.UUID(jid) for jid in job_ids_str]
        jobs = await self.get_jobs_by_ids(job_ids)
        jobs_by_id = {str(j.id): j for j in jobs}

        enriched = []
        for qdrant_res in qdrant_results:
            jid = qdrant_res.get("job_id", "")
            job = jobs_by_id.get(jid)
            if job:
                enriched.append({
                    "semantic_score": qdrant_res.get("score", 0.0),
                    "point_id":       qdrant_res.get("point_id"),
                    "job":            job,
                })

        return enriched

    # ── Listing & Stats ───────────────────────────────────────────────────────

    async def get_recent(
        self,
        *,
        hours: int = 24,
        limit: int = 50,
        source: str | None = None,
    ) -> list[Job]:
        """Get jobs scraped/posted in the last N hours."""
        cutoff = datetime.now(UTC) - timedelta(hours=hours)
        q = (
            self._base_query()
            .where(Job.is_active.is_(True))
            .where(Job.created_at >= cutoff)
            .order_by(desc(Job.created_at))
            .limit(limit)
        )
        if source:
            q = q.where(Job.source == source)
        result = await self.session.execute(q)
        return list(result.scalars().all())

    async def count_by_source(self) -> dict[str, int]:
        """
        Return job count per source.
        Used by: dashboard stats, scraper monitoring.
        """
        q = (
            select(Job.source, func.count(Job.id).label("cnt"))
            .where(Job.is_deleted.is_(False))
            .where(Job.is_active.is_(True))
            .group_by(Job.source)
            .order_by(desc("cnt"))
        )
        result = await self.session.execute(q)
        return {row.source: row.cnt for row in result.all()}

    async def count_by_experience_level(self) -> dict[str, int]:
        """Return job counts per experience level (analytics)."""
        q = (
            select(Job.experience_level, func.count(Job.id).label("cnt"))
            .where(Job.is_deleted.is_(False))
            .where(Job.is_active.is_(True))
            .where(Job.experience_level.isnot(None))
            .group_by(Job.experience_level)
        )
        result = await self.session.execute(q)
        return {row.experience_level: row.cnt for row in result.all()}

    async def count_remote_vs_onsite(self) -> dict[str, int]:
        """Return remote vs onsite job counts."""
        q = (
            select(Job.is_remote, func.count(Job.id).label("cnt"))
            .where(Job.is_deleted.is_(False))
            .where(Job.is_active.is_(True))
            .group_by(Job.is_remote)
        )
        result = await self.session.execute(q)
        data: dict[str, int] = {}
        for row in result.all():
            key = "remote" if row.is_remote else "onsite"
            data[key] = row.cnt
        return data

    async def get_top_companies(self, limit: int = 20) -> list[dict[str, Any]]:
        """Return companies with most active job listings."""
        q = (
            select(Job.company, func.count(Job.id).label("cnt"))
            .where(Job.is_deleted.is_(False))
            .where(Job.is_active.is_(True))
            .group_by(Job.company)
            .order_by(desc("cnt"))
            .limit(limit)
        )
        result = await self.session.execute(q)
        return [{"company": row.company, "count": row.cnt} for row in result.all()]

    async def get_salary_stats(self) -> dict[str, Any]:
        """Return salary distribution stats for jobs with salary data."""
        q = (
            select(
                func.avg(Job.salary_min).label("avg_min"),
                func.avg(Job.salary_max).label("avg_max"),
                func.min(Job.salary_min).label("min_salary"),
                func.max(Job.salary_max).label("max_salary"),
                func.percentile_cont(0.5).within_group(Job.salary_min).label("median_min"),
            )
            .where(Job.is_deleted.is_(False))
            .where(Job.is_active.is_(True))
            .where(Job.salary_min.isnot(None))
        )
        result = await self.session.execute(q)
        row = result.one()
        return {
            "avg_min":    round(float(row.avg_min or 0)),
            "avg_max":    round(float(row.avg_max or 0)),
            "min_salary": round(float(row.min_salary or 0)),
            "max_salary": round(float(row.max_salary or 0)),
            "median_min": round(float(row.median_min or 0)),
        }

    async def get_jobs_scraped_today(self) -> int:
        """Count jobs created today (UTC). Used by dashboard stats."""
        today_start = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
        q = (
            select(func.count())
            .select_from(Job)
            .where(Job.is_deleted.is_(False))
            .where(Job.created_at >= today_start)
        )
        result = await self.session.execute(q)
        return result.scalar_one() or 0

    # ── Bulk Operations (for scrapers) ────────────────────────────────────────

    async def bulk_create(self, jobs_data: list[dict[str, Any]]) -> list[Job]:
        """
        Batch insert multiple jobs. Used by scrapers for efficiency.
        Skips duplicates (checks source+external_id before each insert).
        Returns list of newly created Job objects.
        """
        created: list[Job] = []
        for data in jobs_data:
            source = data.get("source", "")
            ext_id = data.get("external_id", "")
            if source and ext_id:
                existing = await self.get_by_source_id(source, ext_id)
                if existing:
                    continue
            job = await self.create(**data)
            created.append(job)
        logger.info(
            "Bulk job create complete",
            total_input=len(jobs_data),
            newly_created=len(created),
        )
        return created

    async def upsert_job(
        self,
        source: str,
        external_id: str,
        defaults: dict[str, Any],
    ) -> tuple[Job, bool]:
        """
        Create job if new, update if already exists.
        Returns (job, created: bool).
        Updates: description, applicant_count, is_active, ai_summary.
        Does NOT re-embed unless description changed significantly.
        """
        existing = await self.get_by_source_id(source, external_id)
        if existing:
            # Update mutable fields only
            updatable = {
                k: v for k, v in defaults.items()
                if k in ("description", "requirements", "benefits", "applicant_count",
                         "salary_min", "salary_max", "is_active", "expires_at",
                         "ai_summary", "ai_keywords", "ai_green_flags", "ai_red_flags")
            }
            if updatable:
                updated = await self.update(existing.id, **updatable)
                return updated, False
            return existing, False

        job = await self.create(
            source=source,
            external_id=external_id,
            **defaults,
        )
        return job, True

    # ── Maintenance ───────────────────────────────────────────────────────────

    async def deactivate_old_jobs(self, days: int = 60) -> int:
        """
        Mark jobs older than N days as inactive.
        Called nightly by Celery beat task.
        Returns count of deactivated jobs.
        """
        cutoff = datetime.now(UTC) - timedelta(days=days)
        q = (
            update(Job)
            .where(Job.is_deleted.is_(False))
            .where(Job.is_active.is_(True))
            .where(
                or_(
                    and_(Job.posted_at.isnot(None), Job.posted_at < cutoff),
                    and_(Job.posted_at.is_(None), Job.created_at < cutoff),
                )
            )
            .values(is_active=False)
        )
        result = await self.session.execute(q)
        count = result.rowcount or 0
        logger.info(f"Deactivated {count} old jobs (>{days} days)")
        return count

    async def deactivate_by_company(self, company_name: str) -> int:
        """
        Mark all jobs from a company as inactive.
        Used when company's career page returns 404 or company is closed.
        """
        q = (
            update(Job)
            .where(Job.company.ilike(f"%{company_name}%"))
            .where(Job.is_active.is_(True))
            .where(Job.is_deleted.is_(False))
            .values(is_active=False)
        )
        result = await self.session.execute(q)
        count = result.rowcount or 0
        logger.info(f"Deactivated {count} jobs for company: {company_name}")
        return count

    async def delete_old_inactive_jobs(self, days: int = 90) -> int:
        """
        Hard-delete jobs that have been inactive for N days.
        Frees disk space. Called by weekly maintenance task.
        """
        cutoff = datetime.now(UTC) - timedelta(days=days)
        q = text("""
            DELETE FROM jobs
            WHERE is_active = FALSE
              AND is_deleted = FALSE
              AND updated_at < :cutoff
        """).bindparams(cutoff=cutoff)
        result = await self.session.execute(q)
        count = result.rowcount or 0
        logger.info(f"Hard-deleted {count} old inactive jobs")
        return count

    async def update_qdrant_point_id(
        self,
        job_id: uuid.UUID,
        point_id: str,
    ) -> None:
        """Store Qdrant point ID after embedding. Non-blocking update."""
        await self.update(job_id, qdrant_point_id=point_id)

    async def get_jobs_without_embeddings(
        self,
        limit: int = 100,
    ) -> list[Job]:
        """
        Fetch active jobs that haven't been embedded yet.
        Used by backfill task to embed historical jobs.
        """
        q = (
            self._base_query()
            .where(Job.is_active.is_(True))
            .where(Job.qdrant_point_id.is_(None))
            .where(Job.description.isnot(None))
            .order_by(Job.created_at.desc())
            .limit(limit)
        )
        result = await self.session.execute(q)
        return list(result.scalars().all())

    async def get_jobs_for_source_cleanup(
        self,
        source: str,
        keep_ids: list[str],
    ) -> list[Job]:
        """
        Find jobs from a source that are no longer in the latest scrape.
        Used to deactivate removed job listings.
        keep_ids: external_ids that are still active on the source.
        """
        if not keep_ids:
            return []

        q = (
            self._base_query()
            .where(Job.source == source)
            .where(Job.is_active.is_(True))
            .where(Job.external_id.notin_(keep_ids))
        )
        result = await self.session.execute(q)
        return list(result.scalars().all())


__all__ = ["JobRepository"] 