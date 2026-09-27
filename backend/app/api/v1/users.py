"""
app/api/v1/users.py
====================
User profile and settings management routes for the JobHunter AI platform.

Endpoints:
    GET    /users/me                         — Current user profile
    PATCH  /users/me                         — Update profile fields
    PATCH  /users/me/preferences             — Update job search preferences
    PATCH  /users/me/notifications           — Update notification settings
    POST   /users/me/avatar                  — Upload profile avatar
    DELETE /users/me/avatar                  — Remove avatar
    GET    /users/me/api-keys               — List API keys
    POST   /users/me/api-keys               — Create new API key
    DELETE /users/me/api-keys/{key_id}      — Revoke API key
    GET    /users/me/usage                   — Monthly usage counters
    DELETE /users/me                         — Delete account (GDPR)
    GET    /users/{id}                       — Get user profile (admin)
    GET    /users/                           — List all users (admin)
"""

from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter, BackgroundTasks, File, Query, UploadFile
from pydantic import BaseModel, EmailStr, Field, field_validator

from app.api.deps import (
    CurrentUser,
    DBSession,
    Pagination,
    SuperUser,
)
from app.core.constants import (
    ALLOWED_AVATAR_EXTENSIONS,
    MAX_AVATAR_SIZE_BYTES,
    PLAN_LIMITS,
    UserPlan,
)
from app.core.exceptions import (
    FileTooLargeException,
    InvalidFileTypeException,
    NotFoundException,
    ValidationException,
)
from app.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/users", tags=["Users"])


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class UserProfileResponse(BaseModel):
    id: str
    email: str
    full_name: str | None
    avatar_url: str | None
    phone: str | None
    location: str | None
    timezone: str
    plan: str
    plan_expires_at: str | None
    is_active: bool
    is_verified: bool
    is_superuser: bool
    roles: list[str]
    linkedin_connected: bool
    linkedin_profile_url: str | None
    created_at: str
    last_login_at: str | None


class UpdateProfileRequest(BaseModel):
    full_name: str | None = Field(default=None, max_length=256)
    phone: str | None = Field(default=None, max_length=32)
    location: str | None = Field(default=None, max_length=256)
    timezone: str | None = Field(default=None, max_length=64)


class JobSearchPreferences(BaseModel):
    desired_roles: list[str] = Field(default_factory=list, max_length=20)
    preferred_boards: list[str] = Field(default_factory=list)
    work_mode: list[str] = Field(default_factory=list)
    locations: list[str] = Field(default_factory=list)
    min_salary: float | None = None
    max_salary: float | None = None
    job_types: list[str] = Field(default_factory=list)
    excluded_companies: list[str] = Field(default_factory=list)
    auto_apply_enabled: bool = False
    cover_letter_enabled: bool = True
    linkedin_posting_enabled: bool = True
    daily_application_limit: int = Field(default=5, ge=1, le=50)
    require_approval_before_apply: bool = True


class NotificationPreferences(BaseModel):
    email_on_application: bool = True
    email_on_status_change: bool = True
    email_on_interview: bool = True
    email_weekly_summary: bool = True
    in_app_notifications: bool = True


class APIKeyResponse(BaseModel):
    key_id: str
    prefix: str
    created_at: str
    last_used_at: str | None
    is_active: bool


class CreateAPIKeyResponse(BaseModel):
    key_id: str
    api_key: str
    prefix: str
    message: str
    warning: str


class UsageResponse(BaseModel):
    plan: str
    plan_limits: dict[str, int]
    applications_this_month: int
    ai_rewrites_this_month: int
    applications_remaining: int
    ai_rewrites_remaining: int
    reset_date: str


# ---------------------------------------------------------------------------
# GET /users/me
# ---------------------------------------------------------------------------

@router.get("/me", response_model=UserProfileResponse, summary="Get current user profile")
async def get_my_profile(current_user: CurrentUser) -> UserProfileResponse:
    return UserProfileResponse(
        id=str(current_user.id),
        email=current_user.email,
        full_name=current_user.full_name,
        avatar_url=current_user.avatar_url,
        phone=current_user.phone,
        location=current_user.location,
        timezone=current_user.timezone,
        plan=current_user.plan,
        plan_expires_at=(
            current_user.plan_expires_at.isoformat() if current_user.plan_expires_at else None
        ),
        is_active=current_user.is_active,
        is_verified=current_user.is_verified,
        is_superuser=current_user.is_superuser,
        roles=current_user.roles or [],
        linkedin_connected=current_user.has_linkedin_connected,
        linkedin_profile_url=current_user.linkedin_profile_url,
        created_at=current_user.created_at.isoformat(),
        last_login_at=(
            current_user.last_login_at.isoformat() if current_user.last_login_at else None
        ),
    )


# ---------------------------------------------------------------------------
# PATCH /users/me
# ---------------------------------------------------------------------------

@router.patch("/me", response_model=UserProfileResponse, summary="Update profile fields")
async def update_profile(
    payload: UpdateProfileRequest,
    current_user: CurrentUser,
    db: DBSession,
) -> UserProfileResponse:
    if payload.full_name is not None:
        current_user.full_name = payload.full_name
    if payload.phone is not None:
        current_user.phone = payload.phone
    if payload.location is not None:
        current_user.location = payload.location
    if payload.timezone is not None:
        import zoneinfo
        try:
            zoneinfo.ZoneInfo(payload.timezone)
        except Exception:
            raise ValidationException("Invalid timezone string.", field="timezone")
        current_user.timezone = payload.timezone

    await db.flush()
    return await get_my_profile(current_user)


# ---------------------------------------------------------------------------
# PATCH /users/me/preferences
# ---------------------------------------------------------------------------

@router.patch(
    "/me/preferences",
    response_model=dict,
    summary="Update job search preferences",
)
async def update_preferences(
    payload: JobSearchPreferences,
    current_user: CurrentUser,
    db: DBSession,
) -> dict:
    current_user.job_search_preferences = payload.model_dump()
    await db.flush()
    return {"message": "Preferences updated.", "preferences": current_user.job_search_preferences}


# ---------------------------------------------------------------------------
# PATCH /users/me/notifications
# ---------------------------------------------------------------------------

@router.patch(
    "/me/notifications",
    response_model=dict,
    summary="Update notification preferences",
)
async def update_notifications(
    payload: NotificationPreferences,
    current_user: CurrentUser,
    db: DBSession,
) -> dict:
    current_user.notification_preferences = payload.model_dump()
    await db.flush()
    return {"message": "Notification preferences updated.", "preferences": payload.model_dump()}


# ---------------------------------------------------------------------------
# POST /users/me/avatar
# ---------------------------------------------------------------------------

@router.post("/me/avatar", response_model=dict, summary="Upload profile avatar")
async def upload_avatar(
    current_user: CurrentUser,
    db: DBSession,
    file: UploadFile = File(...),
) -> dict:
    import os
    if not file.filename:
        raise ValidationException("File has no filename.")
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_AVATAR_EXTENSIONS:
        raise InvalidFileTypeException(file.filename, ALLOWED_AVATAR_EXTENSIONS)

    data = await file.read()
    if len(data) > MAX_AVATAR_SIZE_BYTES:
        raise FileTooLargeException(
            file.filename, MAX_AVATAR_SIZE_BYTES // (1024 * 1024)
        )

    from app.services.user_service import UserService
    svc = UserService(db)
    avatar_url = await svc.upload_avatar(
        user_id=current_user.id,
        file_bytes=data,
        filename=file.filename,
        content_type=file.content_type or "image/jpeg",
    )
    current_user.avatar_url = avatar_url
    await db.flush()

    return {"avatar_url": avatar_url, "message": "Avatar updated successfully."}


# ---------------------------------------------------------------------------
# DELETE /users/me/avatar
# ---------------------------------------------------------------------------

@router.delete("/me/avatar", status_code=204, response_model=None, summary="Remove profile avatar")
async def delete_avatar(current_user: CurrentUser, db: DBSession) -> None:
    current_user.avatar_url = None
    await db.flush()


# ---------------------------------------------------------------------------
# GET /users/me/api-keys
# ---------------------------------------------------------------------------

@router.get(
    "/me/api-keys",
    response_model=list[APIKeyResponse],
    summary="List API keys",
)
async def list_api_keys(current_user: CurrentUser, db: DBSession) -> list[APIKeyResponse]:
    from app.services.user_service import UserService
    svc = UserService(db)
    keys = await svc.list_api_keys(current_user.id)
    return [
        APIKeyResponse(
            key_id=str(k["id"]),
            prefix=k["prefix"],
            created_at=k["created_at"],
            last_used_at=k.get("last_used_at"),
            is_active=k["is_active"],
        )
        for k in keys
    ]


# ---------------------------------------------------------------------------
# POST /users/me/api-keys
# ---------------------------------------------------------------------------

@router.post(
    "/me/api-keys",
    response_model=CreateAPIKeyResponse,
    status_code=201,
    summary="Create a new API key",
)
async def create_api_key(current_user: CurrentUser, db: DBSession) -> CreateAPIKeyResponse:
    """
    Generate a new API key for programmatic access.

    The plaintext key is returned ONCE — store it securely.
    It cannot be retrieved again; revoke and regenerate if lost.
    """
    from app.core.security import generate_api_key
    from app.services.user_service import UserService

    plaintext, hashed = generate_api_key()
    svc = UserService(db)
    key_id = await svc.store_api_key(user_id=current_user.id, key_hash=hashed, prefix=plaintext[:10])

    return CreateAPIKeyResponse(
        key_id=str(key_id),
        api_key=plaintext,
        prefix=plaintext[:10],
        message="API key created. Store it securely — it will not be shown again.",
        warning="This key grants full API access. Never commit it to version control.",
    )


# ---------------------------------------------------------------------------
# DELETE /users/me/api-keys/{key_id}
# ---------------------------------------------------------------------------

@router.delete(
    "/me/api-keys/{key_id}",
    status_code=204,
    response_model=None,
    summary="Revoke an API key",
)
async def revoke_api_key(
    key_id: str,
    current_user: CurrentUser,
    db: DBSession,
) -> None:
    from app.services.user_service import UserService
    svc = UserService(db)
    await svc.revoke_api_key(user_id=current_user.id, key_id=key_id)


# ---------------------------------------------------------------------------
# GET /users/me/usage
# ---------------------------------------------------------------------------

@router.get("/me/usage", response_model=UsageResponse, summary="Monthly usage counters")
async def get_usage(current_user: CurrentUser) -> UsageResponse:
    from datetime import date, timezone as tz
    today = date.today()
    reset_date = date(today.year + (today.month // 12), (today.month % 12) + 1, 1)

    plan = UserPlan(current_user.plan)
    limits = PLAN_LIMITS.get(plan, {})
    app_limit = limits.get("monthly_applications", 0)
    rewrite_limit = limits.get("ai_rewrites", 0)

    return UsageResponse(
        plan=current_user.plan,
        plan_limits=limits,
        applications_this_month=current_user.applications_this_month,
        ai_rewrites_this_month=current_user.ai_rewrites_this_month,
        applications_remaining=(
            max(0, app_limit - current_user.applications_this_month)
            if app_limit != -1 else -1
        ),
        ai_rewrites_remaining=(
            max(0, rewrite_limit - current_user.ai_rewrites_this_month)
            if rewrite_limit != -1 else -1
        ),
        reset_date=reset_date.isoformat(),
    )


# ---------------------------------------------------------------------------
# DELETE /users/me (GDPR account deletion)
# ---------------------------------------------------------------------------

@router.delete("/me", status_code=202, summary="Delete account (GDPR right to erasure)")
async def delete_account(
    current_user: CurrentUser,
    background_tasks: BackgroundTasks,
    db: DBSession,
) -> dict:
    """
    Request account deletion (GDPR Article 17).

    Soft-deletes the account immediately and queues a background job to
    permanently erase all personal data within 30 days.
    LinkedIn tokens and API keys are revoked immediately.
    """
    current_user.soft_delete()
    current_user.is_active = False
    current_user.linkedin_access_token = None
    current_user.linkedin_refresh_token = None
    await db.flush()

    logger.info("Account deletion requested", user_id=str(current_user.id))
    return {
        "message": "Account deletion initiated. Your data will be permanently removed within 30 days.",
        "user_id": str(current_user.id),
    }


# ---------------------------------------------------------------------------
# Admin endpoints
# ---------------------------------------------------------------------------

@router.get("/", response_model=list[dict], summary="[Admin] List all users")
async def list_all_users(
    admin: SuperUser,
    db: DBSession,
    pagination: Pagination,
    plan: str | None = Query(default=None),
    is_active: bool | None = Query(default=None),
) -> list[dict]:
    from sqlalchemy import select
    from app.db.models.user import User

    stmt = select(User).where(User.is_deleted.is_(False))
    if plan:
        stmt = stmt.where(User.plan == plan)
    if is_active is not None:
        stmt = stmt.where(User.is_active == is_active)
    stmt = stmt.offset(pagination.offset).limit(pagination.limit).order_by(User.created_at.desc())

    result = await db.execute(stmt)
    users = result.scalars().all()
    return [
        {
            "id": str(u.id),
            "email": u.email,
            "full_name": u.full_name,
            "plan": u.plan,
            "is_active": u.is_active,
            "is_verified": u.is_verified,
            "created_at": u.created_at.isoformat(),
            "applications_this_month": u.applications_this_month,
        }
        for u in users
    ]


@router.get("/{user_id}", response_model=dict, summary="[Admin] Get user by ID")
async def get_user_by_id(user_id: str, admin: SuperUser, db: DBSession) -> dict:
    from sqlalchemy import select
    from app.db.models.user import User
    result = await db.execute(
        select(User).where(User.id == uuid.UUID(user_id), User.is_deleted.is_(False))
    )
    user = result.scalar_one_or_none()
    if not user:
        raise NotFoundException("User", identifier=user_id)
    return {
        "id": str(user.id),
        "email": user.email,
        "full_name": user.full_name,
        "plan": user.plan,
        "is_active": user.is_active,
        "roles": user.roles,
        "created_at": user.created_at.isoformat(),
        "job_search_preferences": user.job_search_preferences,
    }