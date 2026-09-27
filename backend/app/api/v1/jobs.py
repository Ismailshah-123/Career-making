"""
app/api/v1/jobs.py
==================
Job discovery and management API routes for the JobHunter AI platform.

Endpoints:
    GET    /jobs/                        — Paginated job listing with filters
    GET    /jobs/search                  — Full-text + semantic search
    GET    /jobs/{job_id}               — Single job detail
    POST   /jobs/discover               — Trigger discovery agent (async)
    GET    /jobs/{job_id}/match         — Match score against current resume
    POST   /jobs/{job_id}/bookmark      — Bookmark a job
    DELETE /jobs/{job_id}/bookmark      — Remove bookmark
    GET    /jobs/bookmarks              — List bookmarked jobs
    GET    /jobs/recommendations        — AI-recommended jobs for current resume
    GET    /jobs/{job_id}/similar       — Semantically similar jobs
    POST   /jobs/bulk-match             — Match multiple jobs against resume
    GET    /jobs/stats                  — Aggregated job market stats
    PATCH  /jobs/{job_id}/status        — Mark job active/inactive (admin)

Architecture:
- Discovery runs via Celery; this layer only manages the results.
- Matching uses Qdrant vector search via matching_agent.
- All filter combinations use SQLAlchemy dynamic WHERE clauses
  built in the repository — no N+1 queries.
"""

from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, Query, Request
from pydantic import BaseModel, Field

from app.api.deps import (
    CurrentUser,
    DBSession,
    OptionalUser,
    Pagination,
    RateLimiter,
    SuperUser,
    require_plan,
)
from app.core.constants import (
    JobBoard,
    JobType,
    WorkMode,
    ExperienceLevel,
    UserPlan,
    MatchTier,
    AGENT_DISCOVERY,
)
from app.core.exceptions import (
    NotFoundException,
    ValidationException,
    JobNotFoundException,
)
from app.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/jobs", tags=["Jobs"])


# ---------------------------------------------------------------------------
# Request / Response Schemas
# ---------------------------------------------------------------------------

class JobSummary(BaseModel):
    id: str
    title: str
    company_name: str | None
    company_logo_url: str | None
    location: str | None
    work_mode: str
    job_type: str
    experience_level: str | None
    salary_range_display: str
    job_board: str
    source_url: str | None
    posted_at: str | None
    is_active: bool
    required_skills: list[str]
    tags: list[str]
    created_at: str

    model_config = {"from_attributes": True}


class JobDetail(JobSummary):
    description_cleaned: str | None
    company_website: str | None
    company_size: str | None
    company_industry: str | None
    country: str | None
    city: str | None
    salary_min: float | None
    salary_max: float | None
    salary_currency: str
    salary_period: str
    has_equity: bool | None
    preferred_skills: list[str]
    required_experience_years: float | None
    required_education: str | None
    structured_requirements: dict[str, Any]
    apply_url: str | None
    ats_provider: str | None


class JobMatchResult(BaseModel):
    job_id: str
    resume_id: str
    match_score: float
    match_tier: str
    matched_skills: list[str]
    missing_skills: list[str]
    keyword_coverage: float
    explanation: str


class JobListResponse(BaseModel):
    items: list[JobSummary]
    total: int
    page: int
    page_size: int
    has_next: bool
    filters_applied: dict[str, Any]


class DiscoverRequest(BaseModel):
    job_boards: list[str] = Field(
        default=["linkedin", "indeed", "remoteok"],
        description="Which boards to search",
    )
    keywords: list[str] = Field(
        default_factory=list,
        description="Search keywords — defaults to user's desired_roles preference",
    )
    locations: list[str] = Field(default_factory=list)
    work_modes: list[str] = Field(default=["remote"])
    max_results_per_board: int = Field(default=50, ge=1, le=200)
    match_to_resume_id: str | None = Field(
        default=None,
        description="Auto-match discovered jobs to this resume",
    )


class DiscoverResponse(BaseModel):
    task_id: str
    agent_run_id: str
    message: str
    estimated_duration_seconds: int


class BulkMatchRequest(BaseModel):
    job_ids: list[str] = Field(..., min_length=1, max_length=50)
    resume_id: str


class BulkMatchResponse(BaseModel):
    results: list[JobMatchResult]
    resume_id: str
    matched_count: int
    excellent_matches: int
    good_matches: int


class JobStatsResponse(BaseModel):
    total_active_jobs: int
    jobs_by_board: dict[str, int]
    jobs_by_work_mode: dict[str, int]
    jobs_by_experience_level: dict[str, int]
    avg_salary_usd: float | None
    top_required_skills: list[dict[str, Any]]
    new_jobs_last_24h: int
    new_jobs_last_7d: int


class BookmarkResponse(BaseModel):
    job_id: str
    is_bookmarked: bool
    message: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _get_job_or_404(db: DBSession, job_id: str):
    from sqlalchemy import select
    from app.db.models.job import Job

    try:
        jid = uuid.UUID(job_id)
    except ValueError:
        raise ValidationException("Invalid job ID format.")

    result = await db.execute(
        select(Job).where(Job.id == jid, Job.is_deleted.is_(False))
    )
    job = result.scalar_one_or_none()
    if not job:
        raise JobNotFoundException(job_id)
    return job


def _to_summary(job: Any) -> JobSummary:
    return JobSummary(
        id=str(job.id),
        title=job.title,
        company_name=job.company_name,
        company_logo_url=job.company_logo_url,
        location=job.location,
        work_mode=job.work_mode,
        job_type=job.job_type,
        experience_level=job.experience_level,
        salary_range_display=job.salary_range_display,
        job_board=job.job_board,
        source_url=job.source_url,
        posted_at=job.posted_at.isoformat() if job.posted_at else None,
        is_active=job.is_active,
        required_skills=job.required_skills or [],
        tags=job.tags or [],
        created_at=job.created_at.isoformat(),
    )


def _to_detail(job: Any) -> JobDetail:
    base = _to_summary(job)
    return JobDetail(
        **base.__dict__,
        description_cleaned=job.description_cleaned,
        company_website=job.company_website,
        company_size=job.company_size,
        company_industry=job.company_industry,
        country=job.country,
        city=job.city,
        salary_min=job.salary_min,
        salary_max=job.salary_max,
        salary_currency=job.salary_currency,
        salary_period=job.salary_period,
        has_equity=job.has_equity,
        preferred_skills=job.preferred_skills or [],
        required_experience_years=job.required_experience_years,
        required_education=job.required_education,
        structured_requirements=job.structured_requirements or {},
        apply_url=job.apply_url,
        ats_provider=job.ats_provider,
    )


# ---------------------------------------------------------------------------
# GET /jobs/
# ---------------------------------------------------------------------------

@router.get(
    "/",
    response_model=JobListResponse,
    summary="List jobs with filters",
    description=(
        "Paginated job listing. Supports filtering by board, work mode, "
        "job type, experience level, salary, skills, and location."
    ),
)
async def list_jobs(
    current_user: CurrentUser,
    db: DBSession,
    pagination: Pagination,
    # Filters
    job_board: list[str] = Query(default=[], description="Filter by job board(s)"),
    work_mode: list[str] = Query(default=[], description="remote | hybrid | onsite"),
    job_type: list[str] = Query(default=[], description="full_time | contract | etc."),
    experience_level: list[str] = Query(default=[], description="entry | mid | senior | etc."),
    salary_min: float | None = Query(default=None, description="Minimum annual salary (USD)"),
    salary_max: float | None = Query(default=None, description="Maximum annual salary (USD)"),
    skills: list[str] = Query(default=[], description="Required skill filters (AND logic)"),
    company_name: str | None = Query(default=None, description="Filter by company name (partial match)"),
    is_remote: bool | None = Query(default=None),
    active_only: bool = Query(default=True, description="Exclude expired postings"),
    sort_by: str = Query(default="posted_at", description="posted_at | match_score | salary_max"),
    sort_order: str = Query(default="desc", description="asc | desc"),
) -> JobListResponse:
    """
    Paginated, filtered job listing.

    Supports up to 10 simultaneous filter dimensions.
    All filters are ANDed together.
    Results are sorted by posted_at DESC by default.
    """
    from app.repositories.job_repository import JobRepository

    repo = JobRepository(db)
    filters = {
        "job_board": job_board or None,
        "work_mode": work_mode or None,
        "job_type": job_type or None,
        "experience_level": experience_level or None,
        "salary_min": salary_min,
        "salary_max": salary_max,
        "skills": skills or None,
        "company_name": company_name,
        "is_remote": is_remote,
        "active_only": active_only,
    }

    jobs, total = await repo.list_with_filters(
        filters=filters,
        offset=pagination.offset,
        limit=pagination.limit,
        sort_by=sort_by,
        sort_order=sort_order,
    )

    return JobListResponse(
        items=[_to_summary(j) for j in jobs],
        total=total,
        page=pagination.page,
        page_size=pagination.page_size,
        has_next=pagination.offset + pagination.page_size < total,
        filters_applied={k: v for k, v in filters.items() if v is not None},
    )


# ---------------------------------------------------------------------------
# GET /jobs/search
# ---------------------------------------------------------------------------

@router.get(
    "/search",
    response_model=JobListResponse,
    summary="Full-text and semantic job search",
)
async def search_jobs(
    current_user: CurrentUser,
    db: DBSession,
    pagination: Pagination,
    q: str = Query(..., min_length=2, max_length=512, description="Search query"),
    semantic: bool = Query(default=True, description="Use vector similarity search"),
    resume_id: str | None = Query(default=None, description="Resume ID to use as semantic query"),
) -> JobListResponse:
    """
    Search jobs using full-text search (PostgreSQL tsvector) and/or
    semantic vector similarity via Qdrant.

    When `semantic=True` and `resume_id` is provided, the resume embedding
    is used as the query vector for maximum relevance.
    When only `q` is provided with `semantic=True`, the query is embedded
    on-the-fly using the embedding service.
    """
    from app.repositories.job_repository import JobRepository
    from app.services.embedding_service import EmbeddingService

    repo = JobRepository(db)

    if semantic:
        emb_svc = EmbeddingService()
        query_vector = await emb_svc.embed_text(q)
        jobs, total = await repo.semantic_search(
            query_vector=query_vector,
            text_query=q,
            offset=pagination.offset,
            limit=pagination.limit,
        )
    else:
        jobs, total = await repo.full_text_search(
            query=q,
            offset=pagination.offset,
            limit=pagination.limit,
        )

    return JobListResponse(
        items=[_to_summary(j) for j in jobs],
        total=total,
        page=pagination.page,
        page_size=pagination.page_size,
        has_next=pagination.offset + pagination.page_size < total,
        filters_applied={"q": q, "semantic": semantic},
    )


# ---------------------------------------------------------------------------
# GET /jobs/bookmarks
# ---------------------------------------------------------------------------

@router.get(
    "/bookmarks",
    response_model=JobListResponse,
    summary="List bookmarked jobs",
)
async def list_bookmarks(
    current_user: CurrentUser,
    db: DBSession,
    pagination: Pagination,
) -> JobListResponse:
    """Return all jobs the current user has bookmarked, newest first."""
    from app.repositories.job_repository import JobRepository

    repo = JobRepository(db)
    jobs, total = await repo.list_bookmarks(
        user_id=current_user.id,
        offset=pagination.offset,
        limit=pagination.limit,
    )

    return JobListResponse(
        items=[_to_summary(j) for j in jobs],
        total=total,
        page=pagination.page,
        page_size=pagination.page_size,
        has_next=pagination.offset + pagination.page_size < total,
        filters_applied={"bookmarked": True},
    )


# ---------------------------------------------------------------------------
# GET /jobs/recommendations
# ---------------------------------------------------------------------------

@router.get(
    "/recommendations",
    response_model=list[JobMatchResult],
    summary="AI-recommended jobs based on active resume",
    dependencies=[Depends(RateLimiter(limit=20, window=3600, key_prefix="job_recs"))],
)
async def get_recommendations(
    current_user: CurrentUser,
    db: DBSession,
    resume_id: str | None = Query(default=None, description="Resume to match against (uses latest if not specified)"),
    top_k: int = Query(default=10, ge=1, le=50),
    min_score: float = Query(default=0.65, ge=0.0, le=1.0),
) -> list[JobMatchResult]:
    """
    Return top-K semantically matched jobs for the user's resume.

    Uses Qdrant vector similarity to find the best job matches.
    Results are filtered by min_score and ranked by cosine similarity.
    """
    from app.services.qdrant_service import QdrantService
    from app.repositories.resume_repository import ResumeRepository

    resume_repo = ResumeRepository(db)

    if resume_id:
        resume = await resume_repo.get_by_id(uuid.UUID(resume_id))
        if not resume or resume.user_id != current_user.id:
            raise NotFoundException("Resume", identifier=resume_id)
    else:
        resume = await resume_repo.get_latest_master(current_user.id)
        if not resume:
            raise NotFoundException(
                "No parsed resume found. Upload and parse a resume first.",
            )

    if not resume.is_embedded:
        raise ValidationException(
            "Resume has not been embedded yet. "
            "Trigger embedding via POST /resumes/{id}/embed first."
        )

    qdrant = QdrantService()
    matches = await qdrant.match_resume_to_jobs(
        resume_id=str(resume.id),
        top_k=top_k,
        score_threshold=min_score,
    )

    return [
        JobMatchResult(
            job_id=m["job_id"],
            resume_id=str(resume.id),
            match_score=m["score"],
            match_tier=m["tier"],
            matched_skills=m.get("matched_skills", []),
            missing_skills=m.get("missing_skills", []),
            keyword_coverage=m.get("keyword_coverage", 0.0),
            explanation=m.get("explanation", ""),
        )
        for m in matches
    ]


# ---------------------------------------------------------------------------
# GET /jobs/stats
# ---------------------------------------------------------------------------

@router.get(
    "/stats",
    response_model=JobStatsResponse,
    summary="Aggregated job market statistics",
)
async def get_job_stats(
    current_user: CurrentUser,
    db: DBSession,
) -> JobStatsResponse:
    """
    Return aggregated statistics across all discovered jobs.

    Used to power the analytics dashboard.
    Stats are computed with GROUP BY queries and cached for 15 minutes.
    """
    from app.repositories.job_repository import JobRepository

    repo = JobRepository(db)
    stats = await repo.get_aggregated_stats()

    return JobStatsResponse(
        total_active_jobs=stats.get("total_active", 0),
        jobs_by_board=stats.get("by_board", {}),
        jobs_by_work_mode=stats.get("by_work_mode", {}),
        jobs_by_experience_level=stats.get("by_exp_level", {}),
        avg_salary_usd=stats.get("avg_salary"),
        top_required_skills=stats.get("top_skills", []),
        new_jobs_last_24h=stats.get("new_24h", 0),
        new_jobs_last_7d=stats.get("new_7d", 0),
    )


# ---------------------------------------------------------------------------
# GET /jobs/{job_id}
# ---------------------------------------------------------------------------

@router.get(
    "/{job_id}",
    response_model=JobDetail,
    summary="Get a single job with full details",
)
async def get_job(
    job_id: str,
    current_user: CurrentUser,
    db: DBSession,
) -> JobDetail:
    job = await _get_job_or_404(db, job_id)
    return _to_detail(job)


# ---------------------------------------------------------------------------
# GET /jobs/{job_id}/match
# ---------------------------------------------------------------------------

@router.get(
    "/{job_id}/match",
    response_model=JobMatchResult,
    summary="Match score between this job and user's resume",
)
async def match_job(
    job_id: str,
    current_user: CurrentUser,
    db: DBSession,
    resume_id: str | None = Query(default=None),
) -> JobMatchResult:
    """
    Compute the cosine similarity match score between a job and a resume.

    If no resume_id is supplied, the user's most recent master resume is used.
    The matching_agent computes:
    - Vector cosine similarity (Qdrant)
    - Keyword coverage (skill overlap)
    - Experience level alignment
    - Composite match score and tier
    """
    from app.services.qdrant_service import QdrantService
    from app.repositories.resume_repository import ResumeRepository

    job = await _get_job_or_404(db, job_id)
    resume_repo = ResumeRepository(db)

    if resume_id:
        resume = await resume_repo.get_by_id(uuid.UUID(resume_id))
        if not resume or resume.user_id != current_user.id:
            raise NotFoundException("Resume", identifier=resume_id)
    else:
        resume = await resume_repo.get_latest_master(current_user.id)
        if not resume:
            raise NotFoundException("No master resume found.")

    qdrant = QdrantService()
    result = await qdrant.match_single(
        resume_id=str(resume.id),
        job_id=str(job.id),
    )

    return JobMatchResult(
        job_id=job_id,
        resume_id=str(resume.id),
        match_score=result["score"],
        match_tier=result["tier"],
        matched_skills=result.get("matched_skills", []),
        missing_skills=result.get("missing_skills", []),
        keyword_coverage=result.get("keyword_coverage", 0.0),
        explanation=result.get("explanation", ""),
    )


# ---------------------------------------------------------------------------
# GET /jobs/{job_id}/similar
# ---------------------------------------------------------------------------

@router.get(
    "/{job_id}/similar",
    response_model=list[JobSummary],
    summary="Find semantically similar jobs",
)
async def get_similar_jobs(
    job_id: str,
    current_user: CurrentUser,
    db: DBSession,
    top_k: int = Query(default=5, ge=1, le=20),
) -> list[JobSummary]:
    """
    Return the top-K most semantically similar jobs to the given job.

    Uses Qdrant nearest-neighbour search on the job embedding.
    Excludes the query job from the results.
    """
    from app.services.qdrant_service import QdrantService
    from app.repositories.job_repository import JobRepository

    job = await _get_job_or_404(db, job_id)

    if not job.is_embedded or not job.qdrant_point_id:
        raise ValidationException("This job has not been embedded yet.")

    qdrant = QdrantService()
    similar_ids = await qdrant.find_similar_jobs(
        job_qdrant_id=job.qdrant_point_id,
        top_k=top_k + 1,  # +1 to exclude the query job
    )
    similar_ids = [s for s in similar_ids if s != str(job.id)][:top_k]

    repo = JobRepository(db)
    similar_jobs = await repo.get_by_ids([uuid.UUID(jid) for jid in similar_ids])

    return [_to_summary(j) for j in similar_jobs]


# ---------------------------------------------------------------------------
# POST /jobs/{job_id}/bookmark
# ---------------------------------------------------------------------------

@router.post(
    "/{job_id}/bookmark",
    response_model=BookmarkResponse,
    status_code=201,
    summary="Bookmark a job",
)
async def bookmark_job(
    job_id: str,
    current_user: CurrentUser,
    db: DBSession,
) -> BookmarkResponse:
    """Add a job to the user's bookmarks."""
    job = await _get_job_or_404(db, job_id)

    from app.repositories.job_repository import JobRepository
    repo = JobRepository(db)
    await repo.add_bookmark(user_id=current_user.id, job_id=job.id)

    return BookmarkResponse(
        job_id=job_id,
        is_bookmarked=True,
        message="Job bookmarked successfully.",
    )


# ---------------------------------------------------------------------------
# DELETE /jobs/{job_id}/bookmark
# ---------------------------------------------------------------------------

@router.delete(
    "/{job_id}/bookmark",
    response_model=BookmarkResponse,
    summary="Remove a job bookmark",
)
async def remove_bookmark(
    job_id: str,
    current_user: CurrentUser,
    db: DBSession,
) -> BookmarkResponse:
    """Remove a job from the user's bookmarks."""
    job = await _get_job_or_404(db, job_id)

    from app.repositories.job_repository import JobRepository
    repo = JobRepository(db)
    await repo.remove_bookmark(user_id=current_user.id, job_id=job.id)

    return BookmarkResponse(
        job_id=job_id,
        is_bookmarked=False,
        message="Bookmark removed.",
    )


# ---------------------------------------------------------------------------
# POST /jobs/discover
# ---------------------------------------------------------------------------

@router.post(
    "/discover",
    response_model=DiscoverResponse,
    status_code=202,
    summary="Trigger the discovery agent to scrape new jobs",
    dependencies=[
        Depends(require_plan(UserPlan.PRO, UserPlan.ENTERPRISE)),
        Depends(RateLimiter(limit=3, window=3600, key_prefix="job_discovery")),
    ],
)
async def discover_jobs(
    payload: DiscoverRequest,
    background_tasks: BackgroundTasks,
    current_user: CurrentUser,
    db: DBSession,
) -> DiscoverResponse:
    """
    Trigger the discovery_agent to scrape job boards and populate the DB.

    The agent:
    1. Queries each board with the user's keywords + location preferences
    2. Deduplicates against existing jobs via content_hash
    3. Embeds new jobs into Qdrant
    4. Optionally matches all new jobs against the specified resume

    Returns immediately with a Celery task ID for polling.
    """
    from app.db.models.agent_run import AgentRun
    from app.core.constants import AgentRunStatus

    # Create AgentRun record before dispatching
    agent_run = AgentRun(
        user_id=current_user.id,
        agent_name=AGENT_DISCOVERY,
        trigger="api",
        status=AgentRunStatus.PENDING,
        input_payload={
            "job_boards": payload.job_boards,
            "keywords": payload.keywords,
            "locations": payload.locations,
            "work_modes": payload.work_modes,
            "max_results_per_board": payload.max_results_per_board,
        },
        context={
            "user_preferences": current_user.job_search_preferences,
            "match_to_resume_id": payload.match_to_resume_id,
        },
    )
    db.add(agent_run)
    await db.flush()

    # Dispatch Celery task
    from app.workers.job_tasks import discover_jobs_task
    task = discover_jobs_task.delay(
        user_id=str(current_user.id),
        agent_run_id=str(agent_run.id),
        config=payload.model_dump(),
    )

    agent_run.celery_task_id = task.id
    await db.flush()

    logger.info(
        "Discovery agent dispatched",
        user_id=str(current_user.id),
        task_id=task.id,
        boards=payload.job_boards,
    )

    return DiscoverResponse(
        task_id=task.id,
        agent_run_id=str(agent_run.id),
        message=(
            f"Discovery started on {len(payload.job_boards)} board(s). "
            "Poll GET /agent-runs/{agent_run_id} for status."
        ),
        estimated_duration_seconds=len(payload.job_boards) * 45,
    )


# ---------------------------------------------------------------------------
# POST /jobs/bulk-match
# ---------------------------------------------------------------------------

@router.post(
    "/bulk-match",
    response_model=BulkMatchResponse,
    summary="Match multiple jobs against a resume",
    dependencies=[Depends(RateLimiter(limit=10, window=3600, key_prefix="bulk_match"))],
)
async def bulk_match(
    payload: BulkMatchRequest,
    current_user: CurrentUser,
    db: DBSession,
) -> BulkMatchResponse:
    """
    Compute match scores for up to 50 jobs against a single resume.

    Uses batched Qdrant queries for efficiency.
    Returns results sorted by match_score DESC.
    """
    from app.repositories.resume_repository import ResumeRepository
    from app.services.qdrant_service import QdrantService

    resume_repo = ResumeRepository(db)
    resume = await resume_repo.get_by_id(uuid.UUID(payload.resume_id))
    if not resume or resume.user_id != current_user.id:
        raise NotFoundException("Resume", identifier=payload.resume_id)

    qdrant = QdrantService()
    raw_results = await qdrant.bulk_match(
        resume_id=payload.resume_id,
        job_ids=payload.job_ids,
    )

    results = [
        JobMatchResult(
            job_id=r["job_id"],
            resume_id=payload.resume_id,
            match_score=r["score"],
            match_tier=r["tier"],
            matched_skills=r.get("matched_skills", []),
            missing_skills=r.get("missing_skills", []),
            keyword_coverage=r.get("keyword_coverage", 0.0),
            explanation=r.get("explanation", ""),
        )
        for r in sorted(raw_results, key=lambda x: x["score"], reverse=True)
    ]

    from app.core.constants import MATCH_SCORE_EXCELLENT, MATCH_SCORE_GOOD
    return BulkMatchResponse(
        results=results,
        resume_id=payload.resume_id,
        matched_count=len(results),
        excellent_matches=sum(1 for r in results if r.match_score >= MATCH_SCORE_EXCELLENT),
        good_matches=sum(1 for r in results if MATCH_SCORE_GOOD <= r.match_score < MATCH_SCORE_EXCELLENT),
    )