"""
app/api/v1/resumes.py
======================
Resume management API routes for the JobHunter AI platform.

Endpoints:
    POST   /resumes/upload          — Upload a resume file (PDF, DOCX, TXT)
    GET    /resumes/                 — List all resumes for current user
    GET    /resumes/{resume_id}      — Get a single resume with full parsed data
    DELETE /resumes/{resume_id}      — Soft-delete a resume
    POST   /resumes/{resume_id}/parse — Re-parse an existing resume (async)
    POST   /resumes/{resume_id}/tailor — AI-tailor resume for a specific job
    GET    /resumes/{resume_id}/ats-score — Get ATS compatibility analysis
    POST   /resumes/{resume_id}/embed — (Re-)generate and store vector embedding
    GET    /resumes/{resume_id}/download — Download the stored resume file
    POST   /resumes/{resume_id}/share — Generate a public share link
    GET    /resumes/{resume_id}/versions — List tailored versions of a master resume
    PATCH  /resumes/{resume_id}      — Update resume title or preferences
"""

from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, Query, UploadFile
from fastapi.responses import FileResponse, StreamingResponse

from app.api.deps import (
    CurrentUser,
    DBSession,
    Pagination,
    RateLimiter,
    check_plan_limit,
    require_plan,
)
from app.core.constants import (
    ALLOWED_RESUME_EXTENSIONS,
    MAX_RESUME_SIZE_BYTES,
    UserPlan,
)
from app.core.exceptions import (
    FileTooLargeException,
    InvalidFileTypeException,
    NotFoundException,
    OwnershipException,
    ResumeNotFoundException,
    ValidationException,
)
from app.core.logging import get_logger
from pydantic import BaseModel, Field

logger = get_logger(__name__)
router = APIRouter(prefix="/resumes", tags=["Resumes"])


# ---------------------------------------------------------------------------
# Response Schemas
# ---------------------------------------------------------------------------

class ResumeBase(BaseModel):
    id: str
    title: str
    is_master: bool
    version: int
    is_parsed: bool
    is_embedded: bool
    ats_score: float | None
    original_filename: str | None
    file_size_bytes: int | None
    word_count: int | None
    years_of_experience: float | None
    extracted_skills: list[str]
    created_at: str
    updated_at: str

    model_config = {"from_attributes": True}


class ResumeDetail(ResumeBase):
    parsed_sections: dict[str, Any]
    ats_feedback: dict[str, Any]
    tailoring_changes: dict[str, Any]
    tailored_for_job_id: str | None
    parent_resume_id: str | None
    is_public: bool
    share_token: str | None


class ATSScoreResponse(BaseModel):
    resume_id: str
    ats_score: float
    keyword_density_score: float | None
    feedback: dict[str, Any]
    suggestions: list[str]
    passed_checks: list[str]
    failed_checks: list[str]


class TailorRequest(BaseModel):
    job_id: str = Field(..., description="UUID of the job to tailor the resume for")
    tone: str = Field(default="professional", description="professional | concise | detailed")
    emphasise_skills: list[str] = Field(default_factory=list)


class TailorResponse(BaseModel):
    tailored_resume_id: str
    original_resume_id: str
    job_id: str
    changes_summary: dict[str, Any]
    ats_score: float | None
    task_id: str | None


class ResumeListResponse(BaseModel):
    items: list[ResumeBase]
    total: int
    page: int
    page_size: int
    has_next: bool


class UpdateResumeRequest(BaseModel):
    title: str | None = Field(default=None, max_length=256)
    is_public: bool | None = None


class ShareLinkResponse(BaseModel):
    share_url: str
    share_token: str
    expires_at: str | None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _validate_upload(file: UploadFile) -> None:
    """Validate file extension and size before processing."""
    if not file.filename:
        raise ValidationException("Uploaded file has no filename.")

    import os
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_RESUME_EXTENSIONS:
        raise InvalidFileTypeException(file.filename, ALLOWED_RESUME_EXTENSIONS)

    if file.size and file.size > MAX_RESUME_SIZE_BYTES:
        raise FileTooLargeException(file.filename, MAX_RESUME_SIZE_BYTES // (1024 * 1024))


async def _get_resume_or_404(db: DBSession, resume_id: str, user_id: uuid.UUID):
    """Load a resume and verify ownership."""
    from sqlalchemy import select
    from app.db.models.resume import Resume

    try:
        rid = uuid.UUID(resume_id)
    except ValueError:
        raise ValidationException("Invalid resume ID format.")

    from sqlalchemy.ext.asyncio import AsyncSession
    result = await db.execute(
        select(Resume).where(
            Resume.id == rid,
            Resume.is_deleted.is_(False),
        )
    )
    resume = result.scalar_one_or_none()
    if not resume:
        raise ResumeNotFoundException(resume_id)
    if resume.user_id != user_id:
        raise OwnershipException("Resume")
    return resume


# ---------------------------------------------------------------------------
# POST /resumes/upload
# ---------------------------------------------------------------------------

@router.post(
    "/upload",
    response_model=ResumeDetail,
    status_code=201,
    summary="Upload a resume file",
    description=(
        "Accepts PDF, DOCX, DOC, TXT, or ODT files up to 10MB. "
        "Triggers async parsing and embedding in the background."
    ),
)
async def upload_resume(
    background_tasks: BackgroundTasks,
    current_user: CurrentUser,
    db: DBSession,
    file: UploadFile = File(..., description="Resume file (PDF, DOCX, TXT)"),
    title: str | None = Form(default=None, max_length=256),
) -> ResumeDetail:
    """
    Upload a resume file.

    1. Validates file type and size
    2. Saves to S3 / local storage
    3. Creates Resume DB row with is_parsed=False
    4. Dispatches async parse + embed task via Celery

    Returns the Resume immediately — poll GET /resumes/{id} for parsed state.
    """
    _validate_upload(file)

    from app.services.resume_service import ResumeService
    svc = ResumeService(db)

    file_bytes = await file.read()
    if len(file_bytes) > MAX_RESUME_SIZE_BYTES:
        raise FileTooLargeException(file.filename or "file", MAX_RESUME_SIZE_BYTES // (1024 * 1024))

    resume = await svc.create_from_upload(
        user_id=current_user.id,
        filename=file.filename or "resume",
        content_type=file.content_type or "application/octet-stream",
        file_bytes=file_bytes,
        title=title or (file.filename.rsplit(".", 1)[0] if file.filename else "My Resume"),
    )

    # Dispatch async parse task
    background_tasks.add_task(svc.trigger_parse_task, str(resume.id))

    logger.info("Resume uploaded", resume_id=str(resume.id), user_id=str(current_user.id))

    return _to_detail(resume)


# ---------------------------------------------------------------------------
# GET /resumes/
# ---------------------------------------------------------------------------

@router.get(
    "/",
    response_model=ResumeListResponse,
    summary="List all resumes for the current user",
)
async def list_resumes(
    current_user: CurrentUser,
    db: DBSession,
    pagination: Pagination,
    master_only: bool = Query(default=False, description="Return only master resumes"),
) -> ResumeListResponse:
    """
    Return paginated list of resumes owned by the current user.

    Use `master_only=true` to exclude AI-tailored variants from the list.
    """
    from app.repositories.resume_repository import ResumeRepository

    repo = ResumeRepository(db)
    resumes, total = await repo.list_by_user(
        user_id=current_user.id,
        offset=pagination.offset,
        limit=pagination.limit,
        master_only=master_only,
    )

    return ResumeListResponse(
        items=[_to_base(r) for r in resumes],
        total=total,
        page=pagination.page,
        page_size=pagination.page_size,
        has_next=pagination.offset + pagination.page_size < total,
    )


# ---------------------------------------------------------------------------
# GET /resumes/{resume_id}
# ---------------------------------------------------------------------------

@router.get(
    "/{resume_id}",
    response_model=ResumeDetail,
    summary="Get a single resume with full parsed data",
)
async def get_resume(
    resume_id: str,
    current_user: CurrentUser,
    db: DBSession,
) -> ResumeDetail:
    resume = await _get_resume_or_404(db, resume_id, current_user.id)
    return _to_detail(resume)


# ---------------------------------------------------------------------------
# PATCH /resumes/{resume_id}
# ---------------------------------------------------------------------------

@router.patch(
    "/{resume_id}",
    response_model=ResumeDetail,
    summary="Update resume metadata",
)
async def update_resume(
    resume_id: str,
    payload: UpdateResumeRequest,
    current_user: CurrentUser,
    db: DBSession,
) -> ResumeDetail:
    resume = await _get_resume_or_404(db, resume_id, current_user.id)

    if payload.title is not None:
        resume.title = payload.title
    if payload.is_public is not None:
        resume.is_public = payload.is_public
        if payload.is_public and not resume.share_token:
            from app.core.security import generate_secure_token
            resume.share_token = generate_secure_token(24)

    await db.flush()
    return _to_detail(resume)


# ---------------------------------------------------------------------------
# DELETE /resumes/{resume_id}
# ---------------------------------------------------------------------------

@router.delete(
    "/{resume_id}",
    status_code=204,
    response_model=None,
    summary="Soft-delete a resume",
)
async def delete_resume(
    resume_id: str,
    current_user: CurrentUser,
    db: DBSession,
) -> None:
    resume = await _get_resume_or_404(db, resume_id, current_user.id)
    resume.soft_delete()
    await db.flush()
    logger.info("Resume deleted", resume_id=resume_id, user_id=str(current_user.id))


# ---------------------------------------------------------------------------
# POST /resumes/{resume_id}/parse
# ---------------------------------------------------------------------------

@router.post(
    "/{resume_id}/parse",
    status_code=202,
    summary="Re-parse a resume (async)",
    response_model=dict,
)
async def parse_resume(
    resume_id: str,
    background_tasks: BackgroundTasks,
    current_user: CurrentUser,
    db: DBSession,
) -> dict:
    """
    Trigger re-parsing of an existing resume.

    Returns immediately with task_id; client polls GET /resumes/{id}
    until is_parsed=True.
    """
    resume = await _get_resume_or_404(db, resume_id, current_user.id)

    from app.services.resume_service import ResumeService
    svc = ResumeService(db)
    background_tasks.add_task(svc.trigger_parse_task, str(resume.id))

    return {"message": "Parse job queued.", "resume_id": resume_id}


# ---------------------------------------------------------------------------
# POST /resumes/{resume_id}/tailor
# ---------------------------------------------------------------------------

@router.post(
    "/{resume_id}/tailor",
    response_model=TailorResponse,
    status_code=202,
    summary="AI-tailor a resume for a specific job",
    dependencies=[Depends(check_plan_limit("ai_rewrites"))],
)
async def tailor_resume(
    resume_id: str,
    payload: TailorRequest,
    background_tasks: BackgroundTasks,
    current_user: CurrentUser,
    db: DBSession,
) -> TailorResponse:
    """
    Generate a tailored version of the master resume for a specific job.

    The resume_agent:
    1. Extracts required skills from the job description
    2. Rewrites experience bullets to emphasise relevant skills
    3. Adjusts the summary to align with the role
    4. Adds missing keywords from the job posting
    5. Scores the result with ATS analysis

    The tailored resume is saved as a new Resume row (is_master=False)
    linked back to the original via parent_resume_id.
    """
    master_resume = await _get_resume_or_404(db, resume_id, current_user.id)

    if not master_resume.is_parsed:
        raise ValidationException(
            "Resume must be fully parsed before tailoring. "
            "Wait for parse to complete or re-trigger it.",
            field="is_parsed",
        )

    from app.services.resume_service import ResumeService
    svc = ResumeService(db)

    tailored_resume, task_id = await svc.tailor_resume(
        master_resume=master_resume,
        job_id=uuid.UUID(payload.job_id),
        user_id=current_user.id,
        tone=payload.tone,
        emphasise_skills=payload.emphasise_skills,
    )

    # Increment usage counter
    current_user.ai_rewrites_this_month += 1
    await db.flush()

    return TailorResponse(
        tailored_resume_id=str(tailored_resume.id),
        original_resume_id=resume_id,
        job_id=payload.job_id,
        changes_summary=tailored_resume.tailoring_changes,
        ats_score=tailored_resume.ats_score,
        task_id=task_id,
    )


# ---------------------------------------------------------------------------
# GET /resumes/{resume_id}/ats-score
# ---------------------------------------------------------------------------

@router.get(
    "/{resume_id}/ats-score",
    response_model=ATSScoreResponse,
    summary="Get ATS compatibility analysis for a resume",
)
async def get_ats_score(
    resume_id: str,
    current_user: CurrentUser,
    db: DBSession,
    job_id: str | None = Query(default=None, description="Optional job ID to score against"),
) -> ATSScoreResponse:
    """
    Return the ATS compatibility score and detailed feedback.

    If job_id is provided, the score is computed against that job's
    requirements for higher precision.
    """
    resume = await _get_resume_or_404(db, resume_id, current_user.id)

    from app.services.resume_service import ResumeService
    svc = ResumeService(db)

    score_data = await svc.compute_ats_score(
        resume=resume,
        job_id=uuid.UUID(job_id) if job_id else None,
    )

    return ATSScoreResponse(
        resume_id=resume_id,
        ats_score=score_data["ats_score"],
        keyword_density_score=score_data.get("keyword_density_score"),
        feedback=score_data.get("feedback", {}),
        suggestions=score_data.get("suggestions", []),
        passed_checks=score_data.get("passed_checks", []),
        failed_checks=score_data.get("failed_checks", []),
    )


# ---------------------------------------------------------------------------
# POST /resumes/{resume_id}/embed
# ---------------------------------------------------------------------------

@router.post(
    "/{resume_id}/embed",
    status_code=202,
    summary="Generate / refresh vector embedding for a resume",
    response_model=dict,
)
async def embed_resume(
    resume_id: str,
    background_tasks: BackgroundTasks,
    current_user: CurrentUser,
    db: DBSession,
) -> dict:
    """
    Generate or refresh the Qdrant vector embedding for this resume.

    Useful after manual edits or when switching embedding models.
    """
    resume = await _get_resume_or_404(db, resume_id, current_user.id)

    if not resume.is_parsed:
        raise ValidationException("Resume must be parsed before embedding.")

    from app.services.resume_service import ResumeService
    svc = ResumeService(db)
    background_tasks.add_task(svc.trigger_embed_task, str(resume.id))

    return {"message": "Embedding job queued.", "resume_id": resume_id}


# ---------------------------------------------------------------------------
# GET /resumes/{resume_id}/versions
# ---------------------------------------------------------------------------

@router.get(
    "/{resume_id}/versions",
    response_model=list[ResumeBase],
    summary="List all tailored versions of a master resume",
)
async def list_resume_versions(
    resume_id: str,
    current_user: CurrentUser,
    db: DBSession,
) -> list[ResumeBase]:
    master = await _get_resume_or_404(db, resume_id, current_user.id)

    from app.repositories.resume_repository import ResumeRepository
    repo = ResumeRepository(db)
    versions = await repo.list_tailored_versions(master.id)

    return [_to_base(r) for r in versions]


# ---------------------------------------------------------------------------
# POST /resumes/{resume_id}/share
# ---------------------------------------------------------------------------

@router.post(
    "/{resume_id}/share",
    response_model=ShareLinkResponse,
    summary="Generate a public share link",
    dependencies=[Depends(require_plan(UserPlan.PRO, UserPlan.ENTERPRISE))],
)
async def create_share_link(
    resume_id: str,
    current_user: CurrentUser,
    db: DBSession,
) -> ShareLinkResponse:
    resume = await _get_resume_or_404(db, resume_id, current_user.id)

    if not resume.share_token:
        from app.core.security import generate_secure_token
        resume.share_token = generate_secure_token(24)
        resume.is_public = True
        await db.flush()

    share_url = f"https://app.jobhunter.ai/r/{resume.share_token}"
    return ShareLinkResponse(
        share_url=share_url,
        share_token=resume.share_token,
        expires_at=None,
    )


# ---------------------------------------------------------------------------
# Response serialisers
# ---------------------------------------------------------------------------

def _to_base(r: Any) -> ResumeBase:
    return ResumeBase(
        id=str(r.id),
        title=r.title,
        is_master=r.is_master,
        version=r.version,
        is_parsed=r.is_parsed,
        is_embedded=r.is_embedded,
        ats_score=r.ats_score,
        original_filename=r.original_filename,
        file_size_bytes=r.file_size_bytes,
        word_count=r.word_count,
        years_of_experience=r.years_of_experience,
        extracted_skills=r.extracted_skills or [],
        created_at=r.created_at.isoformat(),
        updated_at=r.updated_at.isoformat(),
    )


def _to_detail(r: Any) -> ResumeDetail:
    return ResumeDetail(
        **_to_base(r).__dict__,
        parsed_sections=r.parsed_sections or {},
        ats_feedback=r.ats_feedback or {},
        tailoring_changes=r.tailoring_changes or {},
        tailored_for_job_id=str(r.tailored_for_job_id) if r.tailored_for_job_id else None,
        parent_resume_id=str(r.parent_resume_id) if r.parent_resume_id else None,
        is_public=r.is_public,
        share_token=r.share_token,
    )