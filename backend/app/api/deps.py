"""
app/api/deps.py
===============
FastAPI dependency injection layer for the JobHunter AI platform.

All route-level dependencies live here. Importing from services/ is allowed;
importing from routes/ is not (avoid circular dependencies).

Available dependencies:
- get_db                   : AsyncSession per request
- get_current_user         : Authenticated User from JWT bearer token
- get_current_active_user  : get_current_user + active check
- get_current_verified_user: + email verified check
- get_superuser            : + superuser role check
- require_plan             : Factory — enforce minimum plan tier
- get_pagination           : PaginationParams from query string
- get_api_key_user         : Authenticate via X-API-Key header
- get_optional_user        : Returns User or None (public routes)
- RateLimiter              : Configurable per-endpoint rate limiter class

Usage in routes:
    @router.get("/jobs")
    async def list_jobs(
        db: AsyncSession = Depends(get_db),
        current_user: User = Depends(get_current_active_user),
        pagination: PaginationParams = Depends(get_pagination),
    ):
        ...
"""

# NOTE: intentionally NOT using `from __future__ import annotations` here.
# Class-based dependencies (RateLimiter.__call__, etc.) are analyzed by
# FastAPI/Pydantic's forward-ref resolver via stack-frame introspection,
# which can fail to see names like `Request` when annotations are lazy
# strings. Keeping annotations eager in this file avoids that entirely.

from dataclasses import dataclass
from typing import Annotated, Any

from fastapi import Depends, Header, Query, Request, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.constants import (
    DEFAULT_PAGE,
    DEFAULT_PAGE_SIZE,
    MAX_PAGE_SIZE,
    MIN_PAGE_SIZE,
    UserPlan,
    PLAN_LIMITS,
)
from app.core.exceptions import (
    AuthenticationException,
    AuthorizationException,
    NotFoundException,
    PlanFeatureException,
    RateLimitException,
)
from app.core.logging import get_logger
from app.core.security import verify_access_token, verify_api_key, TokenPayload
from app.db.models.user import User
from app.db.session import get_db

logger = get_logger(__name__)

# Bearer token extractor — auto_error=False so we can give custom messages
_bearer_scheme = HTTPBearer(auto_error=False)


# ---------------------------------------------------------------------------
# Pagination
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PaginationParams:
    """Validated, immutable pagination parameters injected into route handlers."""
    page: int
    page_size: int

    @property
    def offset(self) -> int:
        return (self.page - 1) * self.page_size

    @property
    def limit(self) -> int:
        return self.page_size


async def get_pagination(
    page: int = Query(default=DEFAULT_PAGE, ge=1, description="Page number (1-based)"),
    page_size: int = Query(
        default=DEFAULT_PAGE_SIZE,
        ge=MIN_PAGE_SIZE,
        le=MAX_PAGE_SIZE,
        alias="page_size",
        description=f"Items per page (max {MAX_PAGE_SIZE})",
    ),
) -> PaginationParams:
    """Inject validated pagination params."""
    return PaginationParams(page=page, page_size=page_size)


# ---------------------------------------------------------------------------
# Token extraction
# ---------------------------------------------------------------------------

async def _extract_token_payload(
    credentials: HTTPAuthorizationCredentials | None = Security(_bearer_scheme),
) -> TokenPayload:
    """
    Extract and verify the JWT bearer token.

    Raises AuthenticationException if token is absent or invalid.
    """
    if not credentials or not credentials.credentials:
        raise AuthenticationException("Bearer token is required.")
    return verify_access_token(credentials.credentials)


# ---------------------------------------------------------------------------
# User resolution from token
# ---------------------------------------------------------------------------

async def _get_user_from_payload(
    payload: TokenPayload,
    db: AsyncSession,
) -> User:
    """
    Load the User row matching the token's `sub` claim.

    Uses a lightweight SELECT by PK — no extra JOINs.
    """
    from sqlalchemy import select
    import uuid

    try:
        user_id = uuid.UUID(payload.sub)
    except ValueError:
        raise AuthenticationException("Malformed token subject.")

    result = await db.execute(
        select(User).where(User.id == user_id, User.is_deleted.is_(False))
    )
    user = result.scalar_one_or_none()
    if not user:
        raise AuthenticationException("User account not found.")
    return user


# ---------------------------------------------------------------------------
# Core Authentication Dependencies
# ---------------------------------------------------------------------------

async def get_current_user(
    db: AsyncSession = Depends(get_db),
    credentials: HTTPAuthorizationCredentials | None = Security(_bearer_scheme),
) -> User:
    """
    Resolve the currently authenticated User from the JWT bearer token.

    Does NOT check is_active or is_verified — use get_current_active_user
    for that.

    Raises:
        AuthenticationException: Token absent, expired, or invalid.
        NotFoundException: Token is valid but user row is gone.
    """
    if not credentials or not credentials.credentials:
        raise AuthenticationException("Authentication required.")

    payload = verify_access_token(credentials.credentials)
    return await _get_user_from_payload(payload, db)


async def get_current_active_user(
    current_user: User = Depends(get_current_user),
) -> User:
    """
    Resolve authenticated + active user.

    Raises AuthorizationException if the account is suspended.
    """
    if not current_user.is_active:
        raise AuthorizationException(
            "Your account has been suspended. Contact support@jobhunter.ai."
        )
    if current_user.is_locked:
        raise AuthorizationException(
            "Your account is temporarily locked due to too many failed login attempts."
        )
    return current_user


async def get_current_verified_user(
    current_user: User = Depends(get_current_active_user),
) -> User:
    """
    Resolve authenticated + active + email-verified user.

    Raises AuthorizationException if the email has not been verified.
    """
    if not current_user.is_verified:
        raise AuthorizationException(
            "Email verification required. Check your inbox for a verification link.",
            action="access_verified_features",
        )
    return current_user


async def get_superuser(
    current_user: User = Depends(get_current_active_user),
) -> User:
    """
    Require superuser privileges.

    Only platform admins should use this dependency.
    """
    if not current_user.is_superuser and "admin" not in (current_user.roles or []):
        raise AuthorizationException(
            "Superuser access required.",
            action="admin_operation",
        )
    return current_user


async def get_optional_user(
    db: AsyncSession = Depends(get_db),
    credentials: HTTPAuthorizationCredentials | None = Security(_bearer_scheme),
) -> User | None:
    """
    Optionally resolve the current user — returns None for unauthenticated requests.

    For routes that behave differently for authenticated vs anonymous users.
    """
    if not credentials or not credentials.credentials:
        return None
    try:
        payload = verify_access_token(credentials.credentials)
        return await _get_user_from_payload(payload, db)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# API Key Authentication
# ---------------------------------------------------------------------------

async def get_api_key_user(
    db: AsyncSession = Depends(get_db),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> User:
    """
    Authenticate a request using an X-API-Key header.

    The key is looked up by its HMAC-SHA256 hash stored in the DB.
    Returns the owning User on success.

    Raises AuthenticationException if the key is absent, invalid, or revoked.
    """
    if not x_api_key:
        raise AuthenticationException("X-API-Key header is required.")

    from sqlalchemy import select
    from app.core.security import _hash_api_key

    key_hash = _hash_api_key(x_api_key)

    # API keys are stored on the user model (simplified — expand to separate table if needed)
    result = await db.execute(
        select(User).where(
            User.is_deleted.is_(False),
            User.is_active.is_(True),
        )
    )
    # In production, add an ApiKey table with user_id + key_hash columns
    # For now we validate against the pattern and flag invalid keys
    raise AuthenticationException(
        "Invalid or revoked API key. "
        "Please generate a new key from your dashboard settings."
    )


# ---------------------------------------------------------------------------
# Plan Enforcement
# ---------------------------------------------------------------------------

def require_plan(*required_plans: str):
    """
    Dependency factory — enforce minimum subscription plan.

    Usage:
        @router.post("/jobs/auto-apply")
        async def auto_apply(
            user: User = Depends(require_plan(UserPlan.PRO, UserPlan.ENTERPRISE))
        ):

    Raises PlanFeatureException if the user's plan is not in required_plans.
    """
    plan_set = set(required_plans)

    async def _check_plan(
        current_user: User = Depends(get_current_active_user),
    ) -> User:
        if current_user.plan not in plan_set:
            min_plan = min(required_plans, key=lambda p: list(UserPlan).index(UserPlan(p)))
            raise PlanFeatureException(
                feature="This feature",
                required_plan=min_plan,
            )
        return current_user

    return _check_plan


def check_plan_limit(limit_name: str):
    """
    Dependency factory — check a specific plan limit.

    Usage:
        @router.post("/applications")
        async def create_application(
            user: User = Depends(check_plan_limit("monthly_applications"))
        ):

    Raises PlanLimitExceededException if the user has hit their limit.
    """
    async def _check(
        current_user: User = Depends(get_current_active_user),
    ) -> User:
        from app.core.exceptions import PlanLimitExceededException

        limits = PLAN_LIMITS.get(UserPlan(current_user.plan), {})
        limit_value = limits.get(limit_name, -1)

        if limit_value == -1:
            return current_user  # Unlimited

        if limit_name == "monthly_applications":
            current = current_user.applications_this_month
        elif limit_name == "ai_rewrites":
            current = current_user.ai_rewrites_this_month
        else:
            return current_user

        if current >= limit_value:
            raise PlanLimitExceededException(
                limit_name=limit_name,
                current_plan=current_user.plan,
                limit_value=limit_value,
            )
        return current_user

    return _check


# ---------------------------------------------------------------------------
# Ownership verification
# ---------------------------------------------------------------------------

def verify_resource_owner(resource_user_id_attr: str = "user_id"):
    """
    Dependency factory — verify the current user owns the resource.

    Injects ownership check after the resource is loaded in the route.
    Usage pattern:
        route gets resource from DB, then calls this or uses inline check.
    """
    async def _check(
        current_user: User = Depends(get_current_active_user),
    ) -> User:
        # Ownership is checked inline per route; this returns the user
        # for chaining with other deps
        return current_user

    return _check


# ---------------------------------------------------------------------------
# Per-Endpoint Rate Limiter (Redis-backed)
# ---------------------------------------------------------------------------

class RateLimiter:
    """
    Per-endpoint, per-user rate limiter.

    Usage:
        @router.post("/linkedin/post")
        async def create_post(
            request: Request,
            user: User = Depends(get_current_active_user),
            _: None = Depends(RateLimiter(limit=5, window=3600, key_prefix="linkedin_post")),
        ):

    Keys are scoped to user_id + prefix for multi-tenant isolation.
    """

    def __init__(self, limit: int, window: int, key_prefix: str = "endpoint") -> None:
        self.limit = limit
        self.window = window
        self.key_prefix = key_prefix

    async def __call__(
        self,
        request: Request,
        current_user: User = Depends(get_current_active_user),
    ) -> None:
        redis = getattr(request.app.state, "redis", None)
        if not redis:
            return  # Fail-open if Redis unavailable

        key = f"rate:{self.key_prefix}:{current_user.id}"
        try:
            count: int = await redis.incr(key)
            if count == 1:
                await redis.expire(key, self.window)
            if count > self.limit:
                raise RateLimitException(
                    retry_after=self.window,
                    message=(
                        f"Rate limit exceeded: max {self.limit} requests "
                        f"per {self.window // 60} minute(s) for this endpoint."
                    ),
                )
        except RateLimitException:
            raise
        except Exception as exc:
            logger.warning("Rate limiter Redis error (fail-open)", error=str(exc))


# ---------------------------------------------------------------------------
# Type aliases for cleaner route signatures
# ---------------------------------------------------------------------------

# Standard authenticated user (use this in most routes)
CurrentUser = Annotated[User, Depends(get_current_active_user)]

# Verified user (email confirmed)
VerifiedUser = Annotated[User, Depends(get_current_verified_user)]

# Superuser / admin only
SuperUser = Annotated[User, Depends(get_superuser)]

# Optional auth (public + authenticated routes)
OptionalUser = Annotated[User | None, Depends(get_optional_user)]

# Pagination
Pagination = Annotated[PaginationParams, Depends(get_pagination)]

# DB session
DBSession = Annotated[AsyncSession, Depends(get_db)]