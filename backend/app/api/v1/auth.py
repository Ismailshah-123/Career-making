"""
app/api/v1/auth.py
==================
Authentication API routes for the JobHunter AI platform.

Endpoints:
    POST   /auth/register          — Create a new user account
    POST   /auth/login             — Email/password login → JWT pair
    POST   /auth/refresh           — Rotate refresh token → new access token
    POST   /auth/logout            — Blacklist refresh token
    POST   /auth/forgot-password   — Send password reset email
    POST   /auth/reset-password    — Consume reset token, set new password
    POST   /auth/verify-email      — Verify email with OTP/token
    POST   /auth/resend-verification — Re-send email verification
    GET    /auth/me                — Return current authenticated user
    POST   /auth/change-password   — Change password (authenticated)

Security:
- Access tokens expire in 24h (configurable)
- Refresh tokens rotate on every use (family invalidation on reuse)
- Failed login counter + lockout enforced in service layer
- All sensitive events written to AuditLog
"""

from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Depends, Request, Response
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import CurrentUser, DBSession, get_db
from app.core.constants import AuditEvent
from app.core.exceptions import AuthenticationException, ValidationException
from app.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/auth", tags=["Authentication"])


# ---------------------------------------------------------------------------
# Schemas (inline for self-containment; production: import from app.schemas.auth)
# ---------------------------------------------------------------------------

from pydantic import BaseModel, EmailStr, Field, field_validator
from typing import Any
import re


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=10, max_length=128)
    full_name: str | None = Field(default=None, max_length=256)
    timezone: str = "UTC"

    @field_validator("password")
    @classmethod
    def validate_password_strength(cls, v: str) -> str:
        from app.core.security import is_password_strong
        passed, violations = is_password_strong(v)
        if not passed:
            raise ValueError(f"Password too weak: {'; '.join(violations)}")
        return v


class LoginRequest(BaseModel):
    email: EmailStr
    password: str
    remember_me: bool = False


class RefreshRequest(BaseModel):
    refresh_token: str


class ForgotPasswordRequest(BaseModel):
    email: EmailStr


class ResetPasswordRequest(BaseModel):
    token: str
    new_password: str = Field(..., min_length=10, max_length=128)

    @field_validator("new_password")
    @classmethod
    def validate_strength(cls, v: str) -> str:
        from app.core.security import is_password_strong
        passed, violations = is_password_strong(v)
        if not passed:
            raise ValueError(f"Password too weak: {'; '.join(violations)}")
        return v


class VerifyEmailRequest(BaseModel):
    token: str


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str = Field(..., min_length=10, max_length=128)

    @field_validator("new_password")
    @classmethod
    def validate_strength(cls, v: str) -> str:
        from app.core.security import is_password_strong
        passed, violations = is_password_strong(v)
        if not passed:
            raise ValueError(f"Password too weak: {'; '.join(violations)}")
        return v


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    user_id: str
    email: str
    plan: str


class UserResponse(BaseModel):
    id: str
    email: str
    full_name: str | None
    is_active: bool
    is_verified: bool
    plan: str
    avatar_url: str | None
    linkedin_connected: bool
    created_at: str

    model_config = {"from_attributes": True}


class MessageResponse(BaseModel):
    message: str
    detail: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Helper: Write audit log in background
# ---------------------------------------------------------------------------

async def _write_audit(
    db: AsyncSession,
    *,
    event: str,
    user_id: Any = None,
    success: bool = True,
    ip_address: str | None = None,
    request_id: str | None = None,
    metadata: dict | None = None,
    failure_reason: str | None = None,
) -> None:
    from app.db.models.audit_log import AuditLog
    log = AuditLog.create(
        event=event,
        user_id=user_id,
        success=success,
        ip_address=ip_address,
        request_id=request_id,
        metadata=metadata or {},
        failure_reason=failure_reason,
    )
    db.add(log)
    await db.flush()


def _get_ip(request: Request) -> str:
    fwd = request.headers.get("X-Forwarded-For")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


# ---------------------------------------------------------------------------
# POST /auth/register
# ---------------------------------------------------------------------------

@router.post(
    "/register",
    response_model=TokenResponse,
    status_code=201,
    summary="Register a new user account",
    responses={
        201: {"description": "Account created, JWT pair returned"},
        409: {"description": "Email already registered"},
        422: {"description": "Password too weak or invalid email"},
    },
)
async def register(
    payload: RegisterRequest,
    request: Request,
    background_tasks: BackgroundTasks,
    db: DBSession,
) -> TokenResponse:
    """
    Create a new user account.

    Returns a JWT access/refresh token pair immediately — no email
    verification required to log in, but some features are gated behind
    is_verified=True.

    A verification email is dispatched in the background.
    """
    from app.services.auth_service import AuthService

    svc = AuthService(db)
    user, access_token, refresh_token = await svc.register(
        email=payload.email,
        password=payload.password,
        full_name=payload.full_name,
        timezone=payload.timezone,
    )

    background_tasks.add_task(
        _write_audit,
        db,
        event=AuditEvent.USER_REGISTERED,
        user_id=user.id,
        ip_address=_get_ip(request),
        metadata={"email": payload.email},
    )

    logger.info("User registered", user_id=str(user.id), email=user.email)

    from app.core.constants import ACCESS_TOKEN_EXPIRE_MINUTES
    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        user_id=str(user.id),
        email=user.email,
        plan=user.plan,
    )


# ---------------------------------------------------------------------------
# POST /auth/login
# ---------------------------------------------------------------------------

@router.post(
    "/login",
    response_model=TokenResponse,
    summary="Login with email and password",
    responses={
        200: {"description": "JWT pair returned"},
        401: {"description": "Invalid credentials"},
        403: {"description": "Account locked or suspended"},
    },
)
async def login(
    payload: LoginRequest,
    request: Request,
    db: DBSession,
) -> TokenResponse:
    """
    Authenticate with email and password.

    Returns a JWT access token (24h) and refresh token (30 days).
    Failed attempts are counted; the account is locked after 5 consecutive
    failures for 15 minutes.
    """
    from app.services.auth_service import AuthService

    svc = AuthService(db)
    ip = _get_ip(request)

    user, access_token, refresh_token = await svc.login(
        email=payload.email,
        password=payload.password,
        ip_address=ip,
    )

    await _write_audit(
        db,
        event=AuditEvent.USER_LOGIN,
        user_id=user.id,
        ip_address=ip,
        metadata={"remember_me": payload.remember_me},
    )

    logger.info("User logged in", user_id=str(user.id), ip=ip)

    from app.core.constants import ACCESS_TOKEN_EXPIRE_MINUTES
    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        user_id=str(user.id),
        email=user.email,
        plan=user.plan,
    )


# ---------------------------------------------------------------------------
# POST /auth/refresh
# ---------------------------------------------------------------------------

@router.post(
    "/refresh",
    response_model=TokenResponse,
    summary="Rotate refresh token and get new access token",
    responses={
        200: {"description": "New token pair returned"},
        401: {"description": "Refresh token invalid or expired"},
    },
)
async def refresh_token(
    payload: RefreshRequest,
    db: DBSession,
) -> TokenResponse:
    """
    Exchange a valid refresh token for a new access + refresh token pair.

    The provided refresh token is immediately invalidated (rotation).
    If a reused token is detected, the entire token family is invalidated.
    """
    from app.services.auth_service import AuthService

    svc = AuthService(db)
    user, access_token, new_refresh = await svc.refresh(payload.refresh_token)

    from app.core.constants import ACCESS_TOKEN_EXPIRE_MINUTES
    return TokenResponse(
        access_token=access_token,
        refresh_token=new_refresh,
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        user_id=str(user.id),
        email=user.email,
        plan=user.plan,
    )


# ---------------------------------------------------------------------------
# POST /auth/logout
# ---------------------------------------------------------------------------

@router.post(
    "/logout",
    response_model=MessageResponse,
    summary="Logout — invalidate refresh token",
)
async def logout(
    payload: RefreshRequest,
    request: Request,
    current_user: CurrentUser,
    db: DBSession,
) -> MessageResponse:
    """
    Logout the current user by blacklisting their refresh token.

    The access token expires naturally (it cannot be revoked without a
    Redis blacklist per JTI — enable in auth_service if needed for security).
    """
    from app.services.auth_service import AuthService

    svc = AuthService(db)
    await svc.logout(payload.refresh_token)

    await _write_audit(
        db,
        event=AuditEvent.USER_LOGOUT,
        user_id=current_user.id,
        ip_address=_get_ip(request),
    )

    return MessageResponse(message="Logged out successfully.")


# ---------------------------------------------------------------------------
# POST /auth/forgot-password
# ---------------------------------------------------------------------------

@router.post(
    "/forgot-password",
    response_model=MessageResponse,
    summary="Request a password reset email",
)
async def forgot_password(
    payload: ForgotPasswordRequest,
    background_tasks: BackgroundTasks,
    db: DBSession,
) -> MessageResponse:
    """
    Send a password reset link to the given email address.

    Always returns 200 — never reveals whether the email is registered
    (prevents user enumeration attacks).
    """
    from app.services.auth_service import AuthService

    svc = AuthService(db)
    # Fire-and-forget — does nothing if email not found
    background_tasks.add_task(svc.initiate_password_reset, payload.email)

    return MessageResponse(
        message=(
            "If an account with that email exists, a password reset link "
            "has been sent. Check your inbox."
        )
    )


# ---------------------------------------------------------------------------
# POST /auth/reset-password
# ---------------------------------------------------------------------------

@router.post(
    "/reset-password",
    response_model=MessageResponse,
    summary="Reset password using a reset token",
    responses={
        200: {"description": "Password changed"},
        400: {"description": "Token invalid or expired"},
    },
)
async def reset_password(
    payload: ResetPasswordRequest,
    db: DBSession,
) -> MessageResponse:
    """
    Consume a password reset token and set a new password.

    Tokens are single-use and expire after 30 minutes.
    On success, all active sessions are invalidated.
    """
    from app.services.auth_service import AuthService

    svc = AuthService(db)
    await svc.reset_password(
        token=payload.token,
        new_password=payload.new_password,
    )

    return MessageResponse(
        message="Password reset successfully. Please log in with your new password."
    )


# ---------------------------------------------------------------------------
# POST /auth/verify-email
# ---------------------------------------------------------------------------

@router.post(
    "/verify-email",
    response_model=MessageResponse,
    summary="Verify email address with token",
)
async def verify_email(
    payload: VerifyEmailRequest,
    db: DBSession,
) -> MessageResponse:
    """
    Confirm email ownership using the token sent to the user's inbox.

    Marks is_verified=True on success, unlocking gated features.
    """
    from app.services.auth_service import AuthService

    svc = AuthService(db)
    await svc.verify_email(payload.token)

    return MessageResponse(message="Email verified successfully. All features are now unlocked.")


# ---------------------------------------------------------------------------
# POST /auth/resend-verification
# ---------------------------------------------------------------------------

@router.post(
    "/resend-verification",
    response_model=MessageResponse,
    summary="Re-send the email verification link",
)
async def resend_verification(
    background_tasks: BackgroundTasks,
    current_user: CurrentUser,
    db: DBSession,
) -> MessageResponse:
    """
    Re-send the email verification link to the authenticated user.

    Limited to 3 sends per hour (enforced in auth_service).
    """
    from app.services.auth_service import AuthService

    if current_user.is_verified:
        return MessageResponse(message="Your email is already verified.")

    svc = AuthService(db)
    background_tasks.add_task(svc.resend_verification_email, current_user)

    return MessageResponse(
        message=f"Verification email resent to {current_user.email}. Check your inbox."
    )


# ---------------------------------------------------------------------------
# GET /auth/me
# ---------------------------------------------------------------------------

@router.get(
    "/me",
    response_model=UserResponse,
    summary="Get the currently authenticated user",
)
async def get_me(current_user: CurrentUser) -> UserResponse:
    """
    Return the profile of the currently authenticated user.

    No DB query — the user object is already resolved by the dependency.
    """
    return UserResponse(
        id=str(current_user.id),
        email=current_user.email,
        full_name=current_user.full_name,
        is_active=current_user.is_active,
        is_verified=current_user.is_verified,
        plan=current_user.plan,
        avatar_url=current_user.avatar_url,
        linkedin_connected=current_user.has_linkedin_connected,
        created_at=current_user.created_at.isoformat(),
    )


# ---------------------------------------------------------------------------
# POST /auth/change-password
# ---------------------------------------------------------------------------

@router.post(
    "/change-password",
    response_model=MessageResponse,
    summary="Change password (authenticated)",
    responses={
        200: {"description": "Password changed"},
        401: {"description": "Current password incorrect"},
    },
)
async def change_password(
    payload: ChangePasswordRequest,
    request: Request,
    current_user: CurrentUser,
    db: DBSession,
) -> MessageResponse:
    """
    Change the authenticated user's password.

    Requires the current password for verification.
    On success, issues an audit log entry and optionally invalidates
    all other sessions.
    """
    from app.services.auth_service import AuthService

    svc = AuthService(db)
    await svc.change_password(
        user=current_user,
        current_password=payload.current_password,
        new_password=payload.new_password,
    )

    await _write_audit(
        db,
        event=AuditEvent.USER_PASSWORD_CHANGED,
        user_id=current_user.id,
        ip_address=_get_ip(request),
    )

    logger.info("Password changed", user_id=str(current_user.id))

    return MessageResponse(
        message="Password changed successfully. Please log in again on all devices."
    )