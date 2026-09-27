"""
app/api/v1/recruiters.py
=========================
Recruiter contact management API for the JobHunter AI platform.

Endpoints:
    GET    /recruiters/                      — List all recruiter contacts
    POST   /recruiters/                      — Manually add a recruiter
    GET    /recruiters/{id}                  — Recruiter detail with outreach history
    PATCH  /recruiters/{id}                  — Update recruiter notes/status
    DELETE /recruiters/{id}                  — Remove / mark DNC
    POST   /recruiters/discover              — Discover recruiters for a company
    GET    /recruiters/{id}/outreach-history — Full conversation thread
    POST   /recruiters/{id}/message          — Send a direct message
    PATCH  /recruiters/{id}/response         — Record a received response
    GET    /recruiters/stats                 — Outreach funnel analytics
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Query, Request
from pydantic import BaseModel, EmailStr, Field

from app.api.deps import (
    CurrentUser,
    DBSession,
    Pagination,
    RateLimiter,
    require_plan,
)
from app.core.constants import UserPlan, AGENT_OUTREACH, AgentRunStatus
from app.core.exceptions import (
    ConflictException,
    NotFoundException,
    OwnershipException,
    ValidationException,
)
from app.core.logging import get_logger
from fastapi import Depends

logger = get_logger(__name__)
router = APIRouter(prefix="/recruiters", tags=["Recruiters"])


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class RecruiterSummary(BaseModel):
    id: str
    full_name: str | None
    title: str | None
    email: str | None
    linkedin_profile_url: str | None
    linkedin_connection_degree: int | None
    company_name: str | None
    outreach_status: str
    has_responded: bool
    quality_score: float | None
    response_rate: float | None
    is_warm_lead: bool
    do_not_contact: bool
    discovery_source: str | None
    created_at: str


class RecruiterDetail(RecruiterSummary):
    phone: str | None
    specialisations: list[str]
    seniority_focus: list[str]
    outreach_channel: str | None
    outreach_sequence_step: int
    first_message_sent_at: str | None
    last_contacted_at: str | None
    next_contact_at: str | None
    outreach_history: list[dict[str, Any]]
    conversation_thread: list[dict[str, Any]]
    response_sentiment: str | None
    linkedin_mutual_connections: int | None
    linkedin_is_open_to_connect: bool | None
    notes: str | None


class CreateRecruiterRequest(BaseModel):
    full_name: str | None = Field(default=None, max_length=256)
    title: str | None = Field(default=None, max_length=256)
    email: EmailStr | None = None
    linkedin_profile_url: str | None = Field(default=None, max_length=1024)
    company_id: str | None = None
    company_name: str | None = Field(default=None, max_length=512)
    specialisations: list[str] = Field(default_factory=list)
    notes: str | None = Field(default=None, max_length=2000)


class UpdateRecruiterRequest(BaseModel):
    notes: str | None = Field(default=None, max_length=2000)
    do_not_contact: bool | None = None
    outreach_status: str | None = None
    response_sentiment: str | None = None


class DiscoverRecruitersRequest(BaseModel):
    company_id: str | None = None
    company_name: str | None = Field(default=None, max_length=512)
    job_title_keywords: list[str] = Field(
        default=["recruiter", "talent", "hiring"],
        description="Keywords to find relevant contacts",
    )
    max_results: int = Field(default=10, ge=1, le=50)
    connection_degree: int | None = Field(default=2, ge=1, le=3)


class MessageRequest(BaseModel):
    message: str = Field(..., min_length=10, max_length=1500)
    channel: str = Field(
        default="linkedin_message",
        description="linkedin_message | linkedin_inmail | email",
    )


class ResponseRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=5000)
    received_at: str | None = None
    sentiment: str | None = Field(
        default=None,
        description="positive | neutral | negative",
    )


class RecruiterStats(BaseModel):
    total_contacts: int
    by_status: dict[str, int]
    response_rate: float
    avg_response_days: float | None
    warm_leads: int
    do_not_contact: int
    messages_sent: int
    meetings_booked: int


class AgentTaskResponse(BaseModel):
    task_id: str
    agent_run_id: str
    message: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _get_recruiter_or_404(db: DBSession, recruiter_id: str, user_id: uuid.UUID) -> Any:
    """Load a recruiter — access is verified via the linked application's user."""
    from sqlalchemy import select
    from app.db.models.recruiter import Recruiter

    try:
        rid = uuid.UUID(recruiter_id)
    except ValueError:
        raise ValidationException("Invalid recruiter ID.")

    result = await db.execute(
        select(Recruiter).where(Recruiter.id == rid, Recruiter.is_deleted.is_(False))
    )
    recruiter = result.scalar_one_or_none()
    if not recruiter:
        raise NotFoundException("Recruiter", identifier=recruiter_id)
    return recruiter


def _to_summary(r: Any) -> RecruiterSummary:
    return RecruiterSummary(
        id=str(r.id),
        full_name=r.full_name,
        title=r.title,
        email=r.email,
        linkedin_profile_url=r.linkedin_profile_url,
        linkedin_connection_degree=r.linkedin_connection_degree,
        company_name=r.company_name,
        outreach_status=r.outreach_status,
        has_responded=r.has_responded,
        quality_score=r.quality_score,
        response_rate=r.response_rate,
        is_warm_lead=r.is_warm_lead,
        do_not_contact=r.do_not_contact,
        discovery_source=r.discovery_source,
        created_at=r.created_at.isoformat(),
    )


def _to_detail(r: Any) -> RecruiterDetail:
    base = _to_summary(r)
    return RecruiterDetail(
        **base.__dict__,
        phone=r.phone,
        specialisations=r.specialisations or [],
        seniority_focus=r.seniority_focus or [],
        outreach_channel=r.outreach_channel,
        outreach_sequence_step=r.outreach_sequence_step,
        first_message_sent_at=r.first_message_sent_at.isoformat() if r.first_message_sent_at else None,
        last_contacted_at=r.last_contacted_at.isoformat() if r.last_contacted_at else None,
        next_contact_at=r.next_contact_at.isoformat() if r.next_contact_at else None,
        outreach_history=r.outreach_history or [],
        conversation_thread=r.conversation_thread or [],
        response_sentiment=r.response_sentiment,
        linkedin_mutual_connections=r.linkedin_mutual_connections,
        linkedin_is_open_to_connect=r.linkedin_is_open_to_connect,
        notes=r.notes,
    )


# ---------------------------------------------------------------------------
# GET /recruiters/
# ---------------------------------------------------------------------------

@router.get("/", summary="List all recruiter contacts")
async def list_recruiters(
    current_user: CurrentUser,
    db: DBSession,
    pagination: Pagination,
    status: str | None = Query(default=None, description="Filter by outreach_status"),
    company_name: str | None = Query(default=None),
    warm_only: bool = Query(default=False, description="Only return warm leads"),
    sort_by: str = Query(default="created_at", description="created_at | quality_score | response_rate"),
    sort_order: str = Query(default="desc"),
) -> dict:
    from sqlalchemy import select, func
    from app.db.models.recruiter import Recruiter

    stmt = select(Recruiter).where(Recruiter.is_deleted.is_(False))
    if status:
        stmt = stmt.where(Recruiter.outreach_status == status)
    if company_name:
        stmt = stmt.where(Recruiter.company_name.ilike(f"%{company_name}%"))
    if warm_only:
        stmt = stmt.where(Recruiter.is_responsive.is_(True))

    count_stmt = select(func.count()).select_from(stmt.subquery())
    total = (await db.execute(count_stmt)).scalar_one()

    col_map = {
        "created_at": Recruiter.created_at,
        "quality_score": Recruiter.quality_score,
        "response_rate": Recruiter.response_rate,
    }
    sort_col = col_map.get(sort_by, Recruiter.created_at)
    stmt = stmt.order_by(sort_col.desc() if sort_order == "desc" else sort_col.asc())
    stmt = stmt.offset(pagination.offset).limit(pagination.limit)

    result = await db.execute(stmt)
    recruiters = result.scalars().all()

    return {
        "items": [_to_summary(r) for r in recruiters],
        "total": total,
        "page": pagination.page,
        "page_size": pagination.page_size,
        "has_next": pagination.offset + pagination.page_size < total,
    }


# ---------------------------------------------------------------------------
# POST /recruiters/
# ---------------------------------------------------------------------------

@router.post("/", status_code=201, summary="Manually add a recruiter contact")
async def create_recruiter(
    payload: CreateRecruiterRequest,
    current_user: CurrentUser,
    db: DBSession,
) -> RecruiterDetail:
    from app.db.models.recruiter import Recruiter

    if payload.linkedin_profile_url:
        from sqlalchemy import select
        dup = await db.execute(
            select(Recruiter).where(
                Recruiter.linkedin_profile_url == payload.linkedin_profile_url,
                Recruiter.is_deleted.is_(False),
            )
        )
        if dup.scalar_one_or_none():
            raise ConflictException(
                "A recruiter with this LinkedIn URL already exists.",
                detail={"linkedin_profile_url": payload.linkedin_profile_url},
            )

    recruiter = Recruiter(
        full_name=payload.full_name,
        title=payload.title,
        email=str(payload.email) if payload.email else None,
        linkedin_profile_url=payload.linkedin_profile_url,
        company_id=uuid.UUID(payload.company_id) if payload.company_id else None,
        company_name=payload.company_name,
        specialisations=payload.specialisations,
        notes=payload.notes,
        discovery_source="manual",
        outreach_status="not_contacted",
    )
    db.add(recruiter)
    await db.flush()

    logger.info("Recruiter created manually", recruiter_id=str(recruiter.id))
    return _to_detail(recruiter)


# ---------------------------------------------------------------------------
# GET /recruiters/stats
# ---------------------------------------------------------------------------

@router.get("/stats", response_model=RecruiterStats, summary="Outreach funnel analytics")
async def get_recruiter_stats(current_user: CurrentUser, db: DBSession) -> RecruiterStats:
    from sqlalchemy import select, func
    from app.db.models.recruiter import Recruiter

    result = await db.execute(
        select(
            Recruiter.outreach_status,
            func.count().label("count"),
        )
        .where(Recruiter.is_deleted.is_(False))
        .group_by(Recruiter.outreach_status)
    )
    rows = result.all()
    by_status = {row[0]: row[1] for row in rows}
    total = sum(by_status.values())

    responded = (
        await db.execute(
            select(func.count()).where(
                Recruiter.is_deleted.is_(False),
                Recruiter.has_responded.is_(True),
            )
        )
    ).scalar_one()

    warm = (
        await db.execute(
            select(func.count()).where(
                Recruiter.is_deleted.is_(False),
                Recruiter.is_responsive.is_(True),
            )
        )
    ).scalar_one()

    dnc = (
        await db.execute(
            select(func.count()).where(
                Recruiter.is_deleted.is_(False),
                Recruiter.do_not_contact.is_(True),
            )
        )
    ).scalar_one()

    msgs_sent = by_status.get("message_sent", 0) + by_status.get("replied", 0) + by_status.get("meeting_booked", 0)

    return RecruiterStats(
        total_contacts=total,
        by_status=by_status,
        response_rate=round(responded / max(msgs_sent, 1), 3),
        avg_response_days=None,
        warm_leads=warm,
        do_not_contact=dnc,
        messages_sent=msgs_sent,
        meetings_booked=by_status.get("meeting_booked", 0),
    )


# ---------------------------------------------------------------------------
# GET /recruiters/{id}
# ---------------------------------------------------------------------------

@router.get("/{recruiter_id}", response_model=RecruiterDetail, summary="Recruiter detail")
async def get_recruiter(recruiter_id: str, current_user: CurrentUser, db: DBSession) -> RecruiterDetail:
    r = await _get_recruiter_or_404(db, recruiter_id, current_user.id)
    return _to_detail(r)


# ---------------------------------------------------------------------------
# PATCH /recruiters/{id}
# ---------------------------------------------------------------------------

@router.patch("/{recruiter_id}", response_model=RecruiterDetail, summary="Update recruiter")
async def update_recruiter(
    recruiter_id: str,
    payload: UpdateRecruiterRequest,
    current_user: CurrentUser,
    db: DBSession,
) -> RecruiterDetail:
    r = await _get_recruiter_or_404(db, recruiter_id, current_user.id)
    if payload.notes is not None:
        r.notes = payload.notes
    if payload.do_not_contact is not None:
        r.do_not_contact = payload.do_not_contact
    if payload.outreach_status is not None:
        r.outreach_status = payload.outreach_status
    if payload.response_sentiment is not None:
        r.response_sentiment = payload.response_sentiment
    await db.flush()
    return _to_detail(r)


# ---------------------------------------------------------------------------
# DELETE /recruiters/{id}
# ---------------------------------------------------------------------------

@router.delete("/{recruiter_id}", status_code=204, response_model=None, summary="Remove or DNC a recruiter")
async def delete_recruiter(
    recruiter_id: str,
    current_user: CurrentUser,
    db: DBSession,
) -> None:
    r = await _get_recruiter_or_404(db, recruiter_id, current_user.id)
    r.soft_delete()
    r.do_not_contact = True
    await db.flush()


# ---------------------------------------------------------------------------
# POST /recruiters/discover
# ---------------------------------------------------------------------------

@router.post(
    "/discover",
    response_model=AgentTaskResponse,
    status_code=202,
    summary="Discover recruiters for a company using the outreach agent",
    dependencies=[
        Depends(require_plan(UserPlan.PRO, UserPlan.ENTERPRISE)),
        Depends(RateLimiter(limit=5, window=3600, key_prefix="rec_discover")),
    ],
)
async def discover_recruiters(
    payload: DiscoverRecruitersRequest,
    current_user: CurrentUser,
    db: DBSession,
) -> AgentTaskResponse:
    """
    Dispatch the outreach_agent to find recruiters at a target company via LinkedIn.

    Discovered recruiters are scored by quality (connection degree × response rate)
    and saved to the Recruiter table. High-quality contacts are flagged
    as warm leads automatically.
    """
    from app.db.models.agent_run import AgentRun

    if not payload.company_id and not payload.company_name:
        raise ValidationException("Either company_id or company_name is required.")

    run = AgentRun(
        user_id=current_user.id,
        agent_name=AGENT_OUTREACH,
        trigger="api",
        status=AgentRunStatus.PENDING,
        input_payload=payload.model_dump(),
    )
    db.add(run)
    await db.flush()

    from app.workers.job_tasks import discover_recruiters_task
    task = discover_recruiters_task.delay(
        user_id=str(current_user.id),
        agent_run_id=str(run.id),
        config=payload.model_dump(),
    )
    run.celery_task_id = task.id
    await db.flush()

    return AgentTaskResponse(
        task_id=task.id,
        agent_run_id=str(run.id),
        message=f"Recruiter discovery started for '{payload.company_name or payload.company_id}'.",
    )


# ---------------------------------------------------------------------------
# POST /recruiters/{id}/message
# ---------------------------------------------------------------------------

@router.post(
    "/{recruiter_id}/message",
    response_model=RecruiterDetail,
    summary="Send a direct message to a recruiter",
    dependencies=[Depends(RateLimiter(limit=10, window=3600, key_prefix="rec_msg"))],
)
async def send_message(
    recruiter_id: str,
    payload: MessageRequest,
    current_user: CurrentUser,
    db: DBSession,
) -> RecruiterDetail:
    r = await _get_recruiter_or_404(db, recruiter_id, current_user.id)

    if r.do_not_contact:
        raise ValidationException("This contact is marked as Do Not Contact.")

    thread = list(r.conversation_thread)
    thread.append({
        "direction": "sent",
        "message": payload.message,
        "channel": payload.channel,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })
    r.conversation_thread = thread

    history = list(r.outreach_history)
    history.append({
        "step": r.outreach_sequence_step + 1,
        "channel": payload.channel,
        "sent_at": datetime.now(timezone.utc).isoformat(),
        "message_preview": payload.message[:100],
    })
    r.outreach_history = history
    r.outreach_sequence_step += 1
    r.outreach_status = "message_sent"
    r.outreach_channel = payload.channel
    r.last_contacted_at = datetime.now(timezone.utc)
    await db.flush()

    return _to_detail(r)


# ---------------------------------------------------------------------------
# PATCH /recruiters/{id}/response
# ---------------------------------------------------------------------------

@router.patch(
    "/{recruiter_id}/response",
    response_model=RecruiterDetail,
    summary="Record a response received from a recruiter",
)
async def record_response(
    recruiter_id: str,
    payload: ResponseRequest,
    current_user: CurrentUser,
    db: DBSession,
) -> RecruiterDetail:
    r = await _get_recruiter_or_404(db, recruiter_id, current_user.id)

    received_at = datetime.now(timezone.utc)
    if payload.received_at:
        try:
            received_at = datetime.fromisoformat(payload.received_at.replace("Z", "+00:00"))
        except ValueError:
            pass

    thread = list(r.conversation_thread)
    thread.append({
        "direction": "received",
        "message": payload.message,
        "timestamp": received_at.isoformat(),
    })
    r.conversation_thread = thread

    if not r.has_responded:
        r.has_responded = True
        r.first_response_at = received_at
    r.last_response_at = received_at
    r.response_sentiment = payload.sentiment
    r.outreach_status = "replied"
    r.is_responsive = True
    await db.flush()

    return _to_detail(r)