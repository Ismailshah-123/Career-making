"""
app/api/v1/applications.py
===========================
Application lifecycle management API for the JobHunter AI platform.

Endpoints:
    POST   /applications/                        — Create / queue a new application
    GET    /applications/                        — Paginated pipeline view with filters
    GET    /applications/{id}                    — Full application detail
    PATCH  /applications/{id}/status             — Manual status update
    DELETE /applications/{id}                    — Withdraw / soft-delete
    POST   /applications/{id}/apply              — Trigger auto-apply via Playwright
    POST   /applications/{id}/cover-letter       — Generate cover letter
    POST   /applications/{id}/outreach           — Send recruiter outreach
    POST   /applications/{id}/followup           — Schedule / send follow-up
    GET    /applications/{id}/timeline           — Full event timeline
    GET    /applications/pipeline                — Kanban-style pipeline summary
    GET    /applications/stats                   — Conversion funnel metrics
    POST   /applications/bulk-apply             — Queue multiple jobs for auto-apply
    PATCH  /applications/{id}/notes              — Update user notes and priority
    POST   /applications/{id}/interview          — Log an interview round
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, Query, Request
from pydantic import BaseModel, Field, field_validator

from app.api.deps import (
    CurrentUser,
    DBSession,
    Pagination,
    RateLimiter,
    check_plan_limit,
    require_plan,
)
from app.core.constants import (
    ApplicationStatus,
    APPLICATION_STATUS_TRANSITIONS,
    TERMINAL_STATUSES,
    UserPlan,
    AGENT_APPLICATION,
    AGENT_COVER_LETTER,
    AGENT_OUTREACH,
    AGENT_FOLLOWUP,
    AgentRunStatus,
)
from app.core.exceptions import (
    ApplicationNotFoundException,
    ConflictException,
    InvalidStatusTransition,
    NotFoundException,
    OwnershipException,
    ValidationException,
    DuplicateApplicationException,
)
from app.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/applications", tags=["Applications"])


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class ApplicationCreate(BaseModel):
    job_id: str = Field(..., description="UUID of the job to apply to")
    resume_id: str | None = Field(
        default=None,
        description="Resume to use — defaults to latest master resume",
    )
    priority: int = Field(default=5, ge=1, le=10, description="1=highest, 10=lowest")
    notes: str | None = Field(default=None, max_length=2000)
    auto_apply: bool = Field(
        default=False,
        description="Immediately queue for automated application submission",
    )


class StatusUpdateRequest(BaseModel):
    status: str = Field(..., description="New application status")
    reason: str | None = Field(default=None, max_length=512)

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: str) -> str:
        try:
            ApplicationStatus(v)
        except ValueError:
            valid = [s.value for s in ApplicationStatus]
            raise ValueError(f"Invalid status. Must be one of: {valid}")
        return v


class NotesUpdateRequest(BaseModel):
    notes: str | None = Field(default=None, max_length=2000)
    priority: int | None = Field(default=None, ge=1, le=10)
    is_starred: bool | None = None


class InterviewLogRequest(BaseModel):
    round_number: int = Field(..., ge=1)
    interview_type: str = Field(
        ...,
        description="phone | video | technical | system_design | behavioral | onsite | final",
    )
    scheduled_at: str = Field(..., description="ISO 8601 datetime string")
    duration_minutes: int = Field(default=60, ge=15, le=480)
    interviewer_name: str | None = Field(default=None, max_length=256)
    notes: str | None = Field(default=None, max_length=2000)
    outcome: str | None = Field(
        default=None,
        description="passed | failed | pending — set after the interview",
    )


class CoverLetterGenerateRequest(BaseModel):
    tone: str = Field(
        default="professional",
        description="professional | conversational | concise | enthusiastic | formal",
    )
    highlight_skills: list[str] = Field(default_factory=list, max_length=10)
    custom_opening: str | None = Field(default=None, max_length=512)


class OutreachRequest(BaseModel):
    recruiter_id: str | None = Field(
        default=None,
        description="Target recruiter UUID — auto-discovered if not supplied",
    )
    channel: str = Field(
        default="linkedin_message",
        description="linkedin_message | linkedin_inmail | email",
    )
    custom_message: str | None = Field(default=None, max_length=1500)


class FollowUpRequest(BaseModel):
    send_now: bool = Field(default=False, description="Send immediately vs schedule")
    scheduled_for: str | None = Field(
        default=None,
        description="ISO 8601 datetime — required if send_now=False",
    )
    channel: str = Field(default="linkedin_message")
    custom_message: str | None = Field(default=None, max_length=1000)


class BulkApplyRequest(BaseModel):
    job_ids: list[str] = Field(..., min_length=1, max_length=20)
    resume_id: str | None = None
    generate_cover_letters: bool = True
    priority: int = Field(default=5, ge=1, le=10)


class ApplicationSummary(BaseModel):
    id: str
    job_id: str
    job_title: str | None
    company_name: str | None
    company_logo_url: str | None
    job_board: str | None
    status: str
    match_score: float | None
    match_tier: str | None
    is_starred: bool
    priority: int
    applied_at: str | None
    next_followup_at: str | None
    followup_count: int
    is_auto_applied: bool
    created_at: str


class ApplicationDetail(ApplicationSummary):
    resume_id: str | None
    cover_letter_id: str | None
    recruiter_id: str | None
    matched_skills: list[str]
    missing_skills: list[str]
    keyword_match_score: float | None
    match_explanation: str | None
    status_history: list[dict[str, Any]]
    automation_attempts: int
    automation_error: str | None
    outreach_sent: bool
    outreach_sent_at: str | None
    outreach_response_received: bool
    followup_history: list[dict[str, Any]]
    interview_rounds: list[dict[str, Any]]
    offer_details: dict[str, Any]
    user_notes: str | None
    form_data_used: dict[str, Any]


class PipelineColumn(BaseModel):
    status: str
    label: str
    count: int
    applications: list[ApplicationSummary]


class PipelineResponse(BaseModel):
    columns: list[PipelineColumn]
    total: int


class ApplicationStats(BaseModel):
    total: int
    by_status: dict[str, int]
    applied_this_month: int
    response_rate: float
    interview_rate: float
    offer_rate: float
    avg_days_to_response: float | None
    top_companies_applied: list[str]
    top_boards_used: dict[str, int]


class ApplicationListResponse(BaseModel):
    items: list[ApplicationSummary]
    total: int
    page: int
    page_size: int
    has_next: bool


class AgentTaskResponse(BaseModel):
    task_id: str
    agent_run_id: str
    application_id: str
    message: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _get_application_or_404(
    db: DBSession,
    application_id: str,
    user_id: uuid.UUID,
):
    from sqlalchemy import select
    from app.db.models.application import Application

    try:
        aid = uuid.UUID(application_id)
    except ValueError:
        raise ValidationException("Invalid application ID format.")

    result = await db.execute(
        select(Application).where(
            Application.id == aid,
            Application.is_deleted.is_(False),
        )
    )
    app = result.scalar_one_or_none()
    if not app:
        raise ApplicationNotFoundException(application_id)
    if app.user_id != user_id:
        raise OwnershipException("Application")
    return app


def _to_summary(app: Any, job: Any = None) -> ApplicationSummary:
    return ApplicationSummary(
        id=str(app.id),
        job_id=str(app.job_id),
        job_title=getattr(job, "title", None) if job else None,
        company_name=getattr(job, "company_name", None) if job else None,
        company_logo_url=getattr(job, "company_logo_url", None) if job else None,
        job_board=getattr(job, "job_board", None) if job else None,
        status=app.status,
        match_score=app.match_score,
        match_tier=app.match_tier,
        is_starred=app.is_starred,
        priority=app.priority,
        applied_at=app.applied_at.isoformat() if app.applied_at else None,
        next_followup_at=app.next_followup_at.isoformat() if app.next_followup_at else None,
        followup_count=app.followup_count,
        is_auto_applied=app.is_auto_applied,
        created_at=app.created_at.isoformat(),
    )


def _to_detail(app: Any, job: Any = None) -> ApplicationDetail:
    base = _to_summary(app, job)
    return ApplicationDetail(
        **base.__dict__,
        resume_id=str(app.resume_id) if app.resume_id else None,
        cover_letter_id=str(app.cover_letter_id) if app.cover_letter_id else None,
        recruiter_id=str(app.recruiter_id) if app.recruiter_id else None,
        matched_skills=app.matched_skills or [],
        missing_skills=app.missing_skills or [],
        keyword_match_score=app.keyword_match_score,
        match_explanation=app.match_explanation,
        status_history=app.status_history or [],
        automation_attempts=app.automation_attempts,
        automation_error=app.automation_error,
        outreach_sent=app.outreach_sent,
        outreach_sent_at=app.outreach_sent_at.isoformat() if app.outreach_sent_at else None,
        outreach_response_received=app.outreach_response_received,
        followup_history=app.followup_history or [],
        interview_rounds=app.interview_rounds or [],
        offer_details=app.offer_details or {},
        user_notes=app.user_notes,
        form_data_used=app.form_data_used or {},
    )


async def _create_agent_run(
    db: DBSession,
    *,
    user_id: uuid.UUID,
    agent_name: str,
    application_id: str,
    input_payload: dict,
) -> Any:
    from app.db.models.agent_run import AgentRun
    run = AgentRun(
        user_id=user_id,
        agent_name=agent_name,
        trigger="api",
        status=AgentRunStatus.PENDING,
        input_payload=input_payload,
        context={"application_id": application_id},
    )
    db.add(run)
    await db.flush()
    return run


# ---------------------------------------------------------------------------
# POST /applications/
# ---------------------------------------------------------------------------

@router.post(
    "/",
    response_model=ApplicationDetail,
    status_code=201,
    summary="Create and queue a new job application",
    dependencies=[Depends(check_plan_limit("monthly_applications"))],
)
async def create_application(
    payload: ApplicationCreate,
    background_tasks: BackgroundTasks,
    current_user: CurrentUser,
    db: DBSession,
) -> ApplicationDetail:
    """
    Queue a new application for a job.

    Flow:
    1. Validate job exists and is still active
    2. Check for duplicate application (same user + job)
    3. Resolve resume (use supplied ID or latest master)
    4. Create Application row with status=QUEUED
    5. Increment monthly counter
    6. If auto_apply=True, dispatch application_agent task

    Returns the created application immediately.
    """
    from sqlalchemy import select
    from app.db.models.job import Job
    from app.db.models.application import Application
    from app.repositories.resume_repository import ResumeRepository

    # Validate job
    try:
        job_id = uuid.UUID(payload.job_id)
    except ValueError:
        raise ValidationException("Invalid job_id format.")

    result = await db.execute(
        select(Job).where(Job.id == job_id, Job.is_deleted.is_(False))
    )
    job = result.scalar_one_or_none()
    if not job:
        raise NotFoundException("Job", identifier=payload.job_id)
    if not job.is_active:
        raise ValidationException(
            "This job posting is no longer active.",
            field="job_id",
        )

    # Duplicate check
    dup_result = await db.execute(
        select(Application).where(
            Application.user_id == current_user.id,
            Application.job_id == job_id,
            Application.is_deleted.is_(False),
        )
    )
    if dup_result.scalar_one_or_none():
        raise DuplicateApplicationException(payload.job_id)

    # Resolve resume
    resume_repo = ResumeRepository(db)
    if payload.resume_id:
        resume = await resume_repo.get_by_id(uuid.UUID(payload.resume_id))
        if not resume or resume.user_id != current_user.id:
            raise NotFoundException("Resume", identifier=payload.resume_id)
    else:
        resume = await resume_repo.get_latest_master(current_user.id)

    # Create application
    application = Application(
        user_id=current_user.id,
        job_id=job_id,
        resume_id=resume.id if resume else None,
        status=ApplicationStatus.QUEUED,
        priority=payload.priority,
        user_notes=payload.notes,
        status_history=[{
            "from_status": ApplicationStatus.DISCOVERED,
            "to_status": ApplicationStatus.QUEUED,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "reason": "Created via API",
            "agent": "api",
        }],
    )
    db.add(application)

    # Increment monthly counter
    current_user.applications_this_month += 1
    await db.flush()

    # Optionally dispatch auto-apply
    if payload.auto_apply and job.apply_url:
        agent_run = await _create_agent_run(
            db,
            user_id=current_user.id,
            agent_name=AGENT_APPLICATION,
            application_id=str(application.id),
            input_payload={"application_id": str(application.id)},
        )
        from app.workers.job_tasks import submit_application_task
        task = submit_application_task.delay(
            application_id=str(application.id),
            agent_run_id=str(agent_run.id),
        )
        agent_run.celery_task_id = task.id
        await db.flush()

    logger.info(
        "Application created",
        application_id=str(application.id),
        job_id=payload.job_id,
        user_id=str(current_user.id),
        auto_apply=payload.auto_apply,
    )

    return _to_detail(application, job)


# ---------------------------------------------------------------------------
# GET /applications/
# ---------------------------------------------------------------------------

@router.get(
    "/",
    response_model=ApplicationListResponse,
    summary="List applications with filters",
)
async def list_applications(
    current_user: CurrentUser,
    db: DBSession,
    pagination: Pagination,
    status: list[str] = Query(default=[], description="Filter by one or more statuses"),
    job_board: list[str] = Query(default=[]),
    is_starred: bool | None = Query(default=None),
    priority_max: int | None = Query(default=None, ge=1, le=10),
    has_interview: bool | None = Query(default=None),
    sort_by: str = Query(default="created_at", description="created_at | applied_at | match_score | priority"),
    sort_order: str = Query(default="desc"),
) -> ApplicationListResponse:
    """
    Return the user's application pipeline with optional filters.

    Supports filtering by status, board, starred state, and priority.
    Sorted by created_at DESC by default for a newest-first pipeline view.
    """
    from app.repositories.application_repository import ApplicationRepository

    repo = ApplicationRepository(db)
    apps, total = await repo.list_with_filters(
        user_id=current_user.id,
        status_filter=status or None,
        job_board_filter=job_board or None,
        is_starred=is_starred,
        priority_max=priority_max,
        has_interview=has_interview,
        offset=pagination.offset,
        limit=pagination.limit,
        sort_by=sort_by,
        sort_order=sort_order,
    )

    return ApplicationListResponse(
        items=[_to_summary(a, getattr(a, "job", None)) for a in apps],
        total=total,
        page=pagination.page,
        page_size=pagination.page_size,
        has_next=pagination.offset + pagination.page_size < total,
    )


# ---------------------------------------------------------------------------
# GET /applications/pipeline
# ---------------------------------------------------------------------------

@router.get(
    "/pipeline",
    response_model=PipelineResponse,
    summary="Kanban pipeline view grouped by status",
)
async def get_pipeline(
    current_user: CurrentUser,
    db: DBSession,
) -> PipelineResponse:
    """
    Return all applications grouped into pipeline columns by status.

    Returns a maximum of 10 applications per column for performance.
    Clients can fetch more via the paginated /applications/?status= endpoint.
    """
    from app.repositories.application_repository import ApplicationRepository

    repo = ApplicationRepository(db)
    grouped = await repo.get_pipeline_grouped(user_id=current_user.id, per_column=10)

    column_order = [
        (ApplicationStatus.QUEUED, "Queued"),
        (ApplicationStatus.RESUME_TAILORED, "Resume Tailored"),
        (ApplicationStatus.COVER_LETTER_GENERATED, "Cover Letter Ready"),
        (ApplicationStatus.APPLYING, "Applying"),
        (ApplicationStatus.APPLIED, "Applied"),
        (ApplicationStatus.ACKNOWLEDGED, "Acknowledged"),
        (ApplicationStatus.INTERVIEW_SCHEDULED, "Interview Scheduled"),
        (ApplicationStatus.INTERVIEWED, "Interviewed"),
        (ApplicationStatus.OFFER_RECEIVED, "Offer Received"),
        (ApplicationStatus.REJECTED, "Rejected"),
    ]

    columns = []
    total = 0
    for status_enum, label in column_order:
        bucket = grouped.get(status_enum.value, {"apps": [], "count": 0})
        count = bucket["count"]
        total += count
        columns.append(
            PipelineColumn(
                status=status_enum.value,
                label=label,
                count=count,
                applications=[
                    _to_summary(a, getattr(a, "job", None))
                    for a in bucket["apps"]
                ],
            )
        )

    return PipelineResponse(columns=columns, total=total)


# ---------------------------------------------------------------------------
# GET /applications/stats
# ---------------------------------------------------------------------------

@router.get(
    "/stats",
    response_model=ApplicationStats,
    summary="Application conversion funnel metrics",
)
async def get_application_stats(
    current_user: CurrentUser,
    db: DBSession,
) -> ApplicationStats:
    """
    Return conversion funnel metrics for the dashboard.

    Computes: total, by-status counts, response rate, interview rate,
    offer rate, average days to response, and top companies / boards.
    """
    from app.repositories.application_repository import ApplicationRepository

    repo = ApplicationRepository(db)
    stats = await repo.get_user_stats(current_user.id)

    total = stats.get("total", 0)
    applied = stats.get("by_status", {}).get(ApplicationStatus.APPLIED, 0)
    interviewed = stats.get("by_status", {}).get(ApplicationStatus.INTERVIEWED, 0)
    offered = stats.get("by_status", {}).get(ApplicationStatus.OFFER_RECEIVED, 0)

    return ApplicationStats(
        total=total,
        by_status=stats.get("by_status", {}),
        applied_this_month=stats.get("applied_this_month", 0),
        response_rate=round(interviewed / applied, 3) if applied else 0.0,
        interview_rate=round(interviewed / applied, 3) if applied else 0.0,
        offer_rate=round(offered / interviewed, 3) if interviewed else 0.0,
        avg_days_to_response=stats.get("avg_days_to_response"),
        top_companies_applied=stats.get("top_companies", []),
        top_boards_used=stats.get("top_boards", {}),
    )


# ---------------------------------------------------------------------------
# GET /applications/{id}
# ---------------------------------------------------------------------------

@router.get(
    "/{application_id}",
    response_model=ApplicationDetail,
    summary="Full application detail",
)
async def get_application(
    application_id: str,
    current_user: CurrentUser,
    db: DBSession,
) -> ApplicationDetail:
    app = await _get_application_or_404(db, application_id, current_user.id)

    # Eagerly load job for display fields
    from sqlalchemy import select
    from app.db.models.job import Job
    job_result = await db.execute(select(Job).where(Job.id == app.job_id))
    job = job_result.scalar_one_or_none()

    return _to_detail(app, job)


# ---------------------------------------------------------------------------
# PATCH /applications/{id}/status
# ---------------------------------------------------------------------------

@router.patch(
    "/{application_id}/status",
    response_model=ApplicationDetail,
    summary="Manually update application status",
)
async def update_status(
    application_id: str,
    payload: StatusUpdateRequest,
    current_user: CurrentUser,
    db: DBSession,
) -> ApplicationDetail:
    """
    Manually transition an application to a new status.

    Validates that the transition is allowed per the state machine
    defined in APPLICATION_STATUS_TRANSITIONS.
    """
    app = await _get_application_or_404(db, application_id, current_user.id)

    current_status = ApplicationStatus(app.status)
    new_status = ApplicationStatus(payload.status)
    allowed_transitions = APPLICATION_STATUS_TRANSITIONS.get(current_status, [])

    if new_status not in allowed_transitions:
        raise InvalidStatusTransition(
            from_status=current_status.value,
            to_status=new_status.value,
        )

    app.record_status_change(
        from_status=current_status.value,
        to_status=new_status.value,
        reason=payload.reason,
        agent="user_manual",
    )

    if new_status == ApplicationStatus.APPLIED and not app.applied_at:
        app.applied_at = datetime.now(timezone.utc)

    await db.flush()

    logger.info(
        "Application status updated",
        application_id=application_id,
        from_status=current_status.value,
        to_status=new_status.value,
    )

    from sqlalchemy import select
    from app.db.models.job import Job
    job_result = await db.execute(select(Job).where(Job.id == app.job_id))
    job = job_result.scalar_one_or_none()
    return _to_detail(app, job)


# ---------------------------------------------------------------------------
# DELETE /applications/{id}
# ---------------------------------------------------------------------------

@router.delete(
    "/{application_id}",
    status_code=204,
    response_model=None,
    summary="Withdraw and soft-delete an application",
)
async def withdraw_application(
    application_id: str,
    current_user: CurrentUser,
    db: DBSession,
) -> None:
    """
    Withdraw an application and mark it as soft-deleted.

    Terminal applications (offer_received) cannot be withdrawn — only
    active pipeline applications can be removed.
    """
    app = await _get_application_or_404(db, application_id, current_user.id)

    if app.status == ApplicationStatus.OFFER_RECEIVED:
        raise ValidationException(
            "Cannot withdraw an application with an accepted offer.",
            field="status",
        )

    if not app.is_terminal:
        app.record_status_change(
            from_status=app.status,
            to_status=ApplicationStatus.WITHDRAWN.value,
            reason="Withdrawn by user",
            agent="user_manual",
        )
    app.soft_delete()
    await db.flush()

    logger.info("Application withdrawn", application_id=application_id)


# ---------------------------------------------------------------------------
# POST /applications/{id}/apply
# ---------------------------------------------------------------------------

@router.post(
    "/{application_id}/apply",
    response_model=AgentTaskResponse,
    status_code=202,
    summary="Trigger automated application submission",
    dependencies=[
        Depends(require_plan(UserPlan.PRO, UserPlan.ENTERPRISE)),
        Depends(RateLimiter(limit=10, window=3600, key_prefix="auto_apply")),
    ],
)
async def trigger_apply(
    application_id: str,
    current_user: CurrentUser,
    db: DBSession,
) -> AgentTaskResponse:
    """
    Dispatch the application_agent to submit this application via Playwright.

    The agent will:
    1. Navigate to the apply_url or ATS system
    2. Fill all form fields from the resume and user profile
    3. Attach the tailored resume PDF
    4. Paste the generated cover letter
    5. Submit and capture a confirmation screenshot
    6. Update application status to APPLIED on success

    Returns immediately — poll GET /agent-runs/{agent_run_id} for progress.
    """
    app = await _get_application_or_404(db, application_id, current_user.id)

    if app.is_terminal:
        raise ValidationException(
            f"Cannot re-apply to a {app.status} application.",
            field="status",
        )

    if app.automation_attempts >= 3:
        raise ValidationException(
            "Maximum automation attempts (3) reached. Apply manually via the job URL.",
            field="automation_attempts",
        )

    agent_run = await _create_agent_run(
        db,
        user_id=current_user.id,
        agent_name=AGENT_APPLICATION,
        application_id=application_id,
        input_payload={"application_id": application_id},
    )

    from app.workers.job_tasks import submit_application_task
    task = submit_application_task.delay(
        application_id=application_id,
        agent_run_id=str(agent_run.id),
    )
    agent_run.celery_task_id = task.id
    app.automation_attempts += 1
    app.last_automation_attempt_at = datetime.now(timezone.utc)
    await db.flush()

    return AgentTaskResponse(
        task_id=task.id,
        agent_run_id=str(agent_run.id),
        application_id=application_id,
        message="Auto-apply agent dispatched. Monitor progress via the agent run endpoint.",
    )


# ---------------------------------------------------------------------------
# POST /applications/{id}/cover-letter
# ---------------------------------------------------------------------------

@router.post(
    "/{application_id}/cover-letter",
    response_model=AgentTaskResponse,
    status_code=202,
    summary="Generate a cover letter for this application",
    dependencies=[Depends(check_plan_limit("ai_rewrites"))],
)
async def generate_cover_letter(
    application_id: str,
    payload: CoverLetterGenerateRequest,
    current_user: CurrentUser,
    db: DBSession,
) -> AgentTaskResponse:
    """
    Dispatch the cover_letter_agent to generate a personalised cover letter.

    The agent uses:
    - Tailored resume content (skills, experience bullets)
    - Full job description requirements
    - Company culture signals from the Company model
    - User tone preference and highlighted skills

    The generated letter is saved as a CoverLetter row and linked to this
    application via cover_letter_id.
    """
    app = await _get_application_or_404(db, application_id, current_user.id)

    agent_run = await _create_agent_run(
        db,
        user_id=current_user.id,
        agent_name=AGENT_COVER_LETTER,
        application_id=application_id,
        input_payload={
            "application_id": application_id,
            "tone": payload.tone,
            "highlight_skills": payload.highlight_skills,
            "custom_opening": payload.custom_opening,
        },
    )

    from app.workers.resume_tasks import generate_cover_letter_task
    task = generate_cover_letter_task.delay(
        application_id=application_id,
        agent_run_id=str(agent_run.id),
        options=payload.model_dump(),
    )
    agent_run.celery_task_id = task.id
    current_user.ai_rewrites_this_month += 1
    await db.flush()

    return AgentTaskResponse(
        task_id=task.id,
        agent_run_id=str(agent_run.id),
        application_id=application_id,
        message="Cover letter generation queued. Poll GET /agent-runs/{id} for status.",
    )


# ---------------------------------------------------------------------------
# POST /applications/{id}/outreach
# ---------------------------------------------------------------------------

@router.post(
    "/{application_id}/outreach",
    response_model=AgentTaskResponse,
    status_code=202,
    summary="Send recruiter outreach message",
    dependencies=[
        Depends(require_plan(UserPlan.PRO, UserPlan.ENTERPRISE)),
        Depends(RateLimiter(limit=5, window=3600, key_prefix="outreach")),
    ],
)
async def send_outreach(
    application_id: str,
    payload: OutreachRequest,
    current_user: CurrentUser,
    db: DBSession,
) -> AgentTaskResponse:
    """
    Dispatch the outreach_agent to find a recruiter and send a personalised message.

    If recruiter_id is supplied, uses that contact directly.
    Otherwise, the agent searches LinkedIn for recruiters at the company
    with the highest quality score and connection degree.

    Message is personalised using company culture signals, mutual connections,
    recent company news, and the user's relevant experience.
    """
    app = await _get_application_or_404(db, application_id, current_user.id)

    if app.outreach_sent:
        raise ConflictException(
            "Outreach has already been sent for this application. "
            "Use the follow-up endpoint for subsequent messages.",
            detail={"application_id": application_id},
        )

    agent_run = await _create_agent_run(
        db,
        user_id=current_user.id,
        agent_name=AGENT_OUTREACH,
        application_id=application_id,
        input_payload={
            "application_id": application_id,
            "recruiter_id": payload.recruiter_id,
            "channel": payload.channel,
            "custom_message": payload.custom_message,
        },
    )

    from app.workers.job_tasks import send_outreach_task
    task = send_outreach_task.delay(
        application_id=application_id,
        agent_run_id=str(agent_run.id),
        options=payload.model_dump(),
    )
    agent_run.celery_task_id = task.id
    await db.flush()

    return AgentTaskResponse(
        task_id=task.id,
        agent_run_id=str(agent_run.id),
        application_id=application_id,
        message="Outreach agent dispatched. Message will be sent shortly.",
    )


# ---------------------------------------------------------------------------
# POST /applications/{id}/followup
# ---------------------------------------------------------------------------

@router.post(
    "/{application_id}/followup",
    response_model=AgentTaskResponse,
    status_code=202,
    summary="Schedule or send a follow-up message",
)
async def schedule_followup(
    application_id: str,
    payload: FollowUpRequest,
    current_user: CurrentUser,
    db: DBSession,
) -> AgentTaskResponse:
    """
    Schedule or immediately send a follow-up for this application.

    Enforces MAX_FOLLOWUPS_PER_APPLICATION (2) to prevent spam.
    The followup_agent crafts a brief, professional follow-up referencing
    the original application and any recent company news.
    """
    from app.core.constants import MAX_FOLLOWUPS_PER_APPLICATION

    app = await _get_application_or_404(db, application_id, current_user.id)

    if app.followup_count >= MAX_FOLLOWUPS_PER_APPLICATION:
        raise ValidationException(
            f"Maximum follow-ups ({MAX_FOLLOWUPS_PER_APPLICATION}) reached for this application.",
            field="followup_count",
        )

    if not payload.send_now and not payload.scheduled_for:
        raise ValidationException(
            "Either send_now=true or scheduled_for must be provided.",
        )

    scheduled_dt = None
    if payload.scheduled_for:
        try:
            scheduled_dt = datetime.fromisoformat(payload.scheduled_for)
        except ValueError:
            raise ValidationException(
                "Invalid scheduled_for format. Use ISO 8601.",
                field="scheduled_for",
            )

    agent_run = await _create_agent_run(
        db,
        user_id=current_user.id,
        agent_name=AGENT_FOLLOWUP,
        application_id=application_id,
        input_payload={
            "application_id": application_id,
            "send_now": payload.send_now,
            "scheduled_for": payload.scheduled_for,
            "channel": payload.channel,
            "custom_message": payload.custom_message,
        },
    )

    from app.workers.job_tasks import send_followup_task
    if payload.send_now:
        task = send_followup_task.delay(
            application_id=application_id,
            agent_run_id=str(agent_run.id),
        )
    else:
        task = send_followup_task.apply_async(
            args=[application_id, str(agent_run.id)],
            eta=scheduled_dt,
        )

    agent_run.celery_task_id = task.id
    if scheduled_dt:
        app.next_followup_at = scheduled_dt
    await db.flush()

    return AgentTaskResponse(
        task_id=task.id,
        agent_run_id=str(agent_run.id),
        application_id=application_id,
        message=(
            "Follow-up sent immediately."
            if payload.send_now
            else f"Follow-up scheduled for {payload.scheduled_for}."
        ),
    )


# ---------------------------------------------------------------------------
# POST /applications/{id}/interview
# ---------------------------------------------------------------------------

@router.post(
    "/{application_id}/interview",
    response_model=ApplicationDetail,
    summary="Log an interview round",
)
async def log_interview(
    application_id: str,
    payload: InterviewLogRequest,
    current_user: CurrentUser,
    db: DBSession,
) -> ApplicationDetail:
    """
    Log a new or update an existing interview round for this application.

    Automatically transitions status to INTERVIEW_SCHEDULED if the
    current status allows it and an outcome has not been set.
    """
    app = await _get_application_or_404(db, application_id, current_user.id)

    rounds = list(app.interview_rounds)
    # Update existing round if same round_number
    existing_idx = next(
        (i for i, r in enumerate(rounds) if r.get("round_number") == payload.round_number),
        None,
    )
    round_data = {
        "round_number": payload.round_number,
        "type": payload.interview_type,
        "scheduled_at": payload.scheduled_at,
        "duration_minutes": payload.duration_minutes,
        "interviewer_name": payload.interviewer_name,
        "notes": payload.notes,
        "outcome": payload.outcome,
        "logged_at": datetime.now(timezone.utc).isoformat(),
    }

    if existing_idx is not None:
        rounds[existing_idx] = round_data
    else:
        rounds.append(round_data)
    app.interview_rounds = rounds

    # Auto-transition status
    current_st = ApplicationStatus(app.status)
    if (
        ApplicationStatus.INTERVIEW_SCHEDULED in APPLICATION_STATUS_TRANSITIONS.get(current_st, [])
        and not payload.outcome
    ):
        app.record_status_change(
            from_status=current_st.value,
            to_status=ApplicationStatus.INTERVIEW_SCHEDULED.value,
            reason="Interview logged via API",
            agent="user_manual",
        )
    elif payload.outcome == "passed" and current_st == ApplicationStatus.INTERVIEW_SCHEDULED:
        app.record_status_change(
            from_status=current_st.value,
            to_status=ApplicationStatus.INTERVIEWED.value,
            reason=f"Interview round {payload.round_number} passed",
            agent="user_manual",
        )

    await db.flush()

    from sqlalchemy import select
    from app.db.models.job import Job
    job_result = await db.execute(select(Job).where(Job.id == app.job_id))
    job = job_result.scalar_one_or_none()
    return _to_detail(app, job)


# ---------------------------------------------------------------------------
# PATCH /applications/{id}/notes
# ---------------------------------------------------------------------------

@router.patch(
    "/{application_id}/notes",
    response_model=ApplicationDetail,
    summary="Update user notes and priority",
)
async def update_notes(
    application_id: str,
    payload: NotesUpdateRequest,
    current_user: CurrentUser,
    db: DBSession,
) -> ApplicationDetail:
    app = await _get_application_or_404(db, application_id, current_user.id)

    if payload.notes is not None:
        app.user_notes = payload.notes
    if payload.priority is not None:
        app.priority = payload.priority
    if payload.is_starred is not None:
        app.is_starred = payload.is_starred

    await db.flush()

    from sqlalchemy import select
    from app.db.models.job import Job
    job_result = await db.execute(select(Job).where(Job.id == app.job_id))
    job = job_result.scalar_one_or_none()
    return _to_detail(app, job)


# ---------------------------------------------------------------------------
# GET /applications/{id}/timeline
# ---------------------------------------------------------------------------

@router.get(
    "/{application_id}/timeline",
    response_model=list[dict],
    summary="Full event timeline for an application",
)
async def get_timeline(
    application_id: str,
    current_user: CurrentUser,
    db: DBSession,
) -> list[dict]:
    """
    Return a merged, chronologically sorted timeline of all events:
    status changes, automation attempts, outreach sends, follow-ups,
    interview rounds, and agent runs.
    """
    app = await _get_application_or_404(db, application_id, current_user.id)

    timeline: list[dict] = []

    for entry in (app.status_history or []):
        timeline.append({
            "type": "status_change",
            "timestamp": entry.get("timestamp"),
            "from": entry.get("from_status"),
            "to": entry.get("to_status"),
            "reason": entry.get("reason"),
            "agent": entry.get("agent"),
        })

    for entry in (app.followup_history or []):
        timeline.append({
            "type": "followup",
            "timestamp": entry.get("sent_at"),
            "channel": entry.get("channel"),
            "response": entry.get("response"),
        })

    for entry in (app.interview_rounds or []):
        timeline.append({
            "type": "interview",
            "timestamp": entry.get("scheduled_at"),
            "round": entry.get("round_number"),
            "interview_type": entry.get("type"),
            "outcome": entry.get("outcome"),
        })

    if app.outreach_sent and app.outreach_sent_at:
        timeline.append({
            "type": "outreach_sent",
            "timestamp": app.outreach_sent_at.isoformat(),
            "responded": app.outreach_response_received,
        })

    timeline.sort(key=lambda x: x.get("timestamp") or "", reverse=False)
    return timeline


# ---------------------------------------------------------------------------
# POST /applications/bulk-apply
# ---------------------------------------------------------------------------

@router.post(
    "/bulk-apply",
    response_model=list[AgentTaskResponse],
    status_code=202,
    summary="Queue multiple jobs for automated application",
    dependencies=[
        Depends(require_plan(UserPlan.PRO, UserPlan.ENTERPRISE)),
        Depends(RateLimiter(limit=2, window=3600, key_prefix="bulk_apply")),
    ],
)
async def bulk_apply(
    payload: BulkApplyRequest,
    background_tasks: BackgroundTasks,
    current_user: CurrentUser,
    db: DBSession,
) -> list[AgentTaskResponse]:
    """
    Create and queue applications for up to 20 jobs in a single request.

    For each job:
    1. Creates an Application row
    2. Optionally generates a cover letter
    3. Dispatches the application_agent

    Returns one AgentTaskResponse per job.
    Duplicate jobs (already applied) are silently skipped.
    """
    from sqlalchemy import select
    from app.db.models.job import Job
    from app.db.models.application import Application
    from app.workers.job_tasks import submit_application_task

    results: list[AgentTaskResponse] = []

    for job_id_str in payload.job_ids:
        try:
            job_id = uuid.UUID(job_id_str)
        except ValueError:
            continue

        # Check duplicate
        dup = await db.execute(
            select(Application).where(
                Application.user_id == current_user.id,
                Application.job_id == job_id,
                Application.is_deleted.is_(False),
            )
        )
        if dup.scalar_one_or_none():
            continue

        job = (
            await db.execute(select(Job).where(Job.id == job_id, Job.is_deleted.is_(False)))
        ).scalar_one_or_none()
        if not job or not job.is_active:
            continue

        app = Application(
            user_id=current_user.id,
            job_id=job_id,
            resume_id=uuid.UUID(payload.resume_id) if payload.resume_id else None,
            status=ApplicationStatus.QUEUED,
            priority=payload.priority,
            status_history=[{
                "from_status": ApplicationStatus.DISCOVERED.value,
                "to_status": ApplicationStatus.QUEUED.value,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "reason": "Bulk apply",
                "agent": "api",
            }],
        )
        db.add(app)
        current_user.applications_this_month += 1
        await db.flush()

        agent_run = await _create_agent_run(
            db,
            user_id=current_user.id,
            agent_name=AGENT_APPLICATION,
            application_id=str(app.id),
            input_payload={"application_id": str(app.id), "bulk": True},
        )

        task = submit_application_task.delay(
            application_id=str(app.id),
            agent_run_id=str(agent_run.id),
        )
        agent_run.celery_task_id = task.id
        await db.flush()

        results.append(
            AgentTaskResponse(
                task_id=task.id,
                agent_run_id=str(agent_run.id),
                application_id=str(app.id),
                message=f"Queued application for: {job.title} at {job.company_name}",
            )
        )

    logger.info(
        "Bulk apply dispatched",
        user_id=str(current_user.id),
        queued=len(results),
        requested=len(payload.job_ids),
    )
    return results