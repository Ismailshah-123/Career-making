"""
app/api/v1/linkedin.py
=======================
LinkedIn integration API routes for the JobHunter AI platform.

Endpoints:
    GET    /linkedin/auth/url              — Get LinkedIn OAuth authorization URL
    GET    /linkedin/auth/callback         — OAuth callback (token exchange)
    DELETE /linkedin/auth/disconnect       — Revoke LinkedIn connection
    GET    /linkedin/profile               — Get connected LinkedIn profile
    POST   /linkedin/posts/generate        — Generate an AI post for a topic
    GET    /linkedin/posts/                — List all posts (draft/scheduled/published)
    GET    /linkedin/posts/{id}            — Get single post detail
    PATCH  /linkedin/posts/{id}            — Update post content/schedule
    DELETE /linkedin/posts/{id}            — Delete draft / cancel scheduled post
    POST   /linkedin/posts/{id}/publish    — Publish immediately
    POST   /linkedin/posts/{id}/schedule   — Schedule for a specific time
    POST   /linkedin/posts/{id}/approve    — Approve draft for publishing
    GET    /linkedin/posts/{id}/analytics  — Per-post engagement analytics
    GET    /linkedin/analytics/overview    — Account-level analytics summary
    POST   /linkedin/posts/generate-daily  — Trigger daily AI content pipeline
    GET    /linkedin/topics/suggestions    — AI-suggested topics for next posts
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
    LinkedInPostCategory,
    UserPlan,
    AGENT_LINKEDIN,
    AgentRunStatus,
    MAX_LINKEDIN_POST_CHARS,
    MIN_LINKEDIN_POST_CHARS,
    LINKEDIN_POST_HASHTAG_LIMIT,
)
from app.core.exceptions import (
    AuthorizationException,
    ConflictException,
    NotFoundException,
    ValidationException,
    OwnershipException,
)
from app.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/linkedin", tags=["LinkedIn"])


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class OAuthURLResponse(BaseModel):
    authorization_url: str
    state: str
    scope: str


class LinkedInProfileResponse(BaseModel):
    profile_id: str | None
    profile_url: str | None
    full_name: str | None
    headline: str | None
    is_connected: bool
    token_expires_at: str | None
    follower_count: int | None


class GeneratePostRequest(BaseModel):
    topic: str = Field(..., min_length=5, max_length=256)
    category: str = Field(
        default=LinkedInPostCategory.AI_INSIGHTS,
        description="Post category for scheduling diversity",
    )
    tone: str = Field(
        default="thought-leader",
        description="professional | conversational | thought-leader | educational | storytelling",
    )
    include_hashtags: bool = True
    include_cta: bool = True
    max_length: int = Field(default=1500, ge=100, le=3000)
    source_urls: list[str] = Field(
        default_factory=list,
        description="Source articles to base the post on",
        max_length=5,
    )
    custom_instructions: str | None = Field(default=None, max_length=512)

    @field_validator("category")
    @classmethod
    def validate_category(cls, v: str) -> str:
        try:
            LinkedInPostCategory(v)
        except ValueError:
            valid = [c.value for c in LinkedInPostCategory]
            raise ValueError(f"Invalid category. Must be one of: {valid}")
        return v


class UpdatePostRequest(BaseModel):
    content: str | None = Field(default=None, min_length=100, max_length=3000)
    title: str | None = Field(default=None, max_length=256)
    hashtags: list[str] | None = Field(default=None, max_length=5)
    scheduled_at: str | None = None
    requires_approval: bool | None = None

    @field_validator("content")
    @classmethod
    def validate_length(cls, v: str | None) -> str | None:
        if v is not None and len(v) > MAX_LINKEDIN_POST_CHARS:
            raise ValueError(f"Content exceeds {MAX_LINKEDIN_POST_CHARS} character limit.")
        return v


class SchedulePostRequest(BaseModel):
    scheduled_at: str = Field(..., description="ISO 8601 UTC datetime for publishing")

    @field_validator("scheduled_at")
    @classmethod
    def validate_future(cls, v: str) -> str:
        try:
            dt = datetime.fromisoformat(v.replace("Z", "+00:00"))
        except ValueError:
            raise ValueError("Invalid datetime format. Use ISO 8601.")
        if dt <= datetime.now(timezone.utc):
            raise ValueError("scheduled_at must be in the future.")
        return v


class DailyPipelineRequest(BaseModel):
    categories: list[str] = Field(
        default_factory=list,
        description="Categories to generate (defaults to user config)",
    )
    schedule_time: str | None = Field(
        default=None,
        description="Time to publish daily (HH:MM in user's timezone), e.g. '08:00'",
    )
    days_ahead: int = Field(default=7, ge=1, le=30)


class PostSummary(BaseModel):
    id: str
    title: str | None
    content_preview: str
    category: str
    tone: str | None
    status: str
    scheduled_at: str | None
    published_at: str | None
    character_count: int | None
    hashtags: list[str]
    likes: int
    comments: int
    shares: int
    impressions: int
    engagement_rate: float | None
    created_at: str


class PostDetail(PostSummary):
    content: str
    hook: str | None
    call_to_action: str | None
    generation_model: str | None
    generation_tokens_used: int | None
    source_urls: list[str]
    mentioned_companies: list[str]
    linkedin_post_id: str | None
    linkedin_post_url: str | None
    publish_error: str | None
    publish_attempts: int
    ab_test_group: str | None


class PostListResponse(BaseModel):
    items: list[PostSummary]
    total: int
    page: int
    page_size: int
    has_next: bool


class PostAnalyticsResponse(BaseModel):
    post_id: str
    impressions: int
    likes: int
    comments: int
    shares: int
    clicks: int
    profile_views_gained: int
    engagement_rate: float
    total_engagement: int
    metrics_last_synced_at: str | None
    benchmark_engagement_rate: float


class OverviewAnalyticsResponse(BaseModel):
    total_posts: int
    published_posts: int
    total_impressions: int
    total_likes: int
    total_comments: int
    total_shares: int
    avg_engagement_rate: float
    best_performing_post_id: str | None
    best_category: str | None
    posts_by_status: dict[str, int]
    posts_by_category: dict[str, int]
    weekly_trend: list[dict[str, Any]]


class TopicSuggestion(BaseModel):
    topic: str
    category: str
    reason: str
    trending_score: float
    suggested_hashtags: list[str]


class AgentTaskResponse(BaseModel):
    task_id: str
    agent_run_id: str
    message: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_linkedin_connected(user: Any) -> None:
    if not user.has_linkedin_connected:
        raise AuthorizationException(
            "LinkedIn account is not connected. "
            "Connect via GET /linkedin/auth/url first.",
            action="linkedin_post",
        )


async def _get_post_or_404(db: DBSession, post_id: str, user_id: uuid.UUID) -> Any:
    from sqlalchemy import select
    from app.db.models.linkedin_post import LinkedInPost

    try:
        pid = uuid.UUID(post_id)
    except ValueError:
        raise ValidationException("Invalid post ID format.")

    result = await db.execute(
        select(LinkedInPost).where(
            LinkedInPost.id == pid,
            LinkedInPost.is_deleted.is_(False),
        )
    )
    post = result.scalar_one_or_none()
    if not post:
        raise NotFoundException("LinkedIn post", identifier=post_id)
    if post.user_id != user_id:
        raise OwnershipException("LinkedIn post")
    return post


def _to_summary(post: Any) -> PostSummary:
    preview = post.content[:150] + "..." if len(post.content) > 150 else post.content
    return PostSummary(
        id=str(post.id),
        title=post.title,
        content_preview=preview,
        category=post.category,
        tone=post.tone,
        status=post.status,
        scheduled_at=post.scheduled_at.isoformat() if post.scheduled_at else None,
        published_at=post.published_at.isoformat() if post.published_at else None,
        character_count=post.character_count,
        hashtags=post.hashtags or [],
        likes=post.likes,
        comments=post.comments,
        shares=post.shares,
        impressions=post.impressions,
        engagement_rate=post.engagement_rate,
        created_at=post.created_at.isoformat(),
    )


def _to_detail(post: Any) -> PostDetail:
    base = _to_summary(post)
    return PostDetail(
        **base.__dict__,
        content=post.content,
        hook=post.hook,
        call_to_action=post.call_to_action,
        generation_model=post.generation_model,
        generation_tokens_used=post.generation_tokens_used,
        source_urls=post.source_urls or [],
        mentioned_companies=post.mentioned_companies or [],
        linkedin_post_id=post.linkedin_post_id,
        linkedin_post_url=post.linkedin_post_url,
        publish_error=post.publish_error,
        publish_attempts=post.publish_attempts,
        ab_test_group=post.ab_test_group,
    )


async def _create_agent_run(db: DBSession, *, user_id: uuid.UUID, input_payload: dict) -> Any:
    from app.db.models.agent_run import AgentRun
    run = AgentRun(
        user_id=user_id,
        agent_name=AGENT_LINKEDIN,
        trigger="api",
        status=AgentRunStatus.PENDING,
        input_payload=input_payload,
    )
    db.add(run)
    await db.flush()
    return run


# ---------------------------------------------------------------------------
# GET /linkedin/auth/url
# ---------------------------------------------------------------------------

@router.get(
    "/auth/url",
    response_model=OAuthURLResponse,
    summary="Get LinkedIn OAuth authorization URL",
)
async def get_oauth_url(current_user: CurrentUser) -> OAuthURLResponse:
    """
    Generate a LinkedIn OAuth 2.0 authorization URL.

    The user is redirected here to grant the platform access to their
    LinkedIn profile and posting permissions.

    State is a signed JWT to prevent CSRF during the OAuth flow.
    """
    from app.services.linkedin_service import LinkedInService
    svc = LinkedInService()
    url, state = svc.build_auth_url(user_id=str(current_user.id))

    return OAuthURLResponse(
        authorization_url=url,
        state=state,
        scope=svc.scope,
    )


# ---------------------------------------------------------------------------
# GET /linkedin/auth/callback
# ---------------------------------------------------------------------------

@router.get(
    "/auth/callback",
    response_model=dict,
    summary="LinkedIn OAuth callback — exchange code for tokens",
    include_in_schema=False,  # Hidden from public docs — called by LinkedIn
)
async def oauth_callback(
    db: DBSession,
    code: str = Query(...),
    state: str = Query(...),
) -> dict:
    """
    LinkedIn redirects here after the user approves permissions.

    Validates state (CSRF), exchanges code for access + refresh tokens,
    fetches basic profile data, and persists tokens encrypted in the DB.
    """
    from app.services.linkedin_service import LinkedInService
    from app.core.security import verify_access_token

    svc = LinkedInService()

    # Validate state JWT to extract user_id
    try:
        payload = verify_access_token(state)
        user_id = uuid.UUID(payload.sub)
    except Exception:
        raise ValidationException("Invalid or expired OAuth state parameter.")

    tokens = await svc.exchange_code(code=code)

    # Load user and persist tokens
    from sqlalchemy import select
    from app.db.models.user import User
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if not user:
        raise NotFoundException("User", identifier=str(user_id))

    profile = await svc.fetch_profile(tokens["access_token"])
    user.linkedin_access_token = tokens["access_token"]
    user.linkedin_refresh_token = tokens.get("refresh_token")
    user.linkedin_profile_id = profile.get("id")
    user.linkedin_profile_url = profile.get("profile_url")
    if "expires_in" in tokens:
        from datetime import timedelta
        user.linkedin_token_expires_at = datetime.now(timezone.utc) + timedelta(
            seconds=tokens["expires_in"]
        )
    await db.flush()

    logger.info("LinkedIn connected", user_id=str(user_id), profile_id=profile.get("id"))
    return {"message": "LinkedIn connected successfully.", "profile_id": profile.get("id")}


# ---------------------------------------------------------------------------
# DELETE /linkedin/auth/disconnect
# ---------------------------------------------------------------------------

@router.delete(
    "/auth/disconnect",
    response_model=dict,
    summary="Revoke LinkedIn connection",
)
async def disconnect_linkedin(
    current_user: CurrentUser,
    db: DBSession,
) -> dict:
    """Remove stored LinkedIn OAuth tokens from the user's account."""
    current_user.linkedin_access_token = None
    current_user.linkedin_refresh_token = None
    current_user.linkedin_token_expires_at = None
    current_user.linkedin_profile_id = None
    current_user.linkedin_profile_url = None
    await db.flush()

    logger.info("LinkedIn disconnected", user_id=str(current_user.id))
    return {"message": "LinkedIn account disconnected successfully."}


# ---------------------------------------------------------------------------
# GET /linkedin/profile
# ---------------------------------------------------------------------------

@router.get(
    "/profile",
    response_model=LinkedInProfileResponse,
    summary="Get connected LinkedIn profile details",
)
async def get_linkedin_profile(current_user: CurrentUser) -> LinkedInProfileResponse:
    """Return the LinkedIn profile details stored for the current user."""
    if not current_user.has_linkedin_connected:
        return LinkedInProfileResponse(
            profile_id=None,
            profile_url=None,
            full_name=None,
            headline=None,
            is_connected=False,
            token_expires_at=None,
            follower_count=None,
        )

    return LinkedInProfileResponse(
        profile_id=current_user.linkedin_profile_id,
        profile_url=current_user.linkedin_profile_url,
        full_name=current_user.full_name,
        headline=None,
        is_connected=True,
        token_expires_at=(
            current_user.linkedin_token_expires_at.isoformat()
            if current_user.linkedin_token_expires_at
            else None
        ),
        follower_count=None,
    )


# ---------------------------------------------------------------------------
# POST /linkedin/posts/generate
# ---------------------------------------------------------------------------

@router.post(
    "/posts/generate",
    response_model=PostDetail,
    status_code=201,
    summary="Generate an AI-written LinkedIn post",
    dependencies=[
        Depends(RateLimiter(limit=10, window=3600, key_prefix="li_generate")),
        Depends(check_plan_limit("linkedin_posts_per_day")),
    ],
)
async def generate_post(
    payload: GeneratePostRequest,
    current_user: CurrentUser,
    db: DBSession,
) -> PostDetail:
    """
    Generate a LinkedIn post using the linkedin_agent.

    The agent:
    1. Searches for recent news / research on the topic (web search tool)
    2. Extracts key insights and data points
    3. Writes a hook, body, and call-to-action in the specified tone
    4. Selects relevant hashtags (max 5 per LinkedIn best practices)
    5. Scores the post for engagement potential
    6. Saves as a draft — user can review before publishing

    Returns the generated post immediately (synchronous LLM call).
    """
    from app.services.linkedin_service import LinkedInService

    svc = LinkedInService()
    post = await svc.generate_post(
        user_id=current_user.id,
        topic=payload.topic,
        category=payload.category,
        tone=payload.tone,
        include_hashtags=payload.include_hashtags,
        include_cta=payload.include_cta,
        max_length=payload.max_length,
        source_urls=payload.source_urls,
        custom_instructions=payload.custom_instructions,
        db=db,
    )

    logger.info(
        "LinkedIn post generated",
        post_id=str(post.id),
        user_id=str(current_user.id),
        topic=payload.topic,
    )
    return _to_detail(post)


# ---------------------------------------------------------------------------
# GET /linkedin/posts/
# ---------------------------------------------------------------------------

@router.get(
    "/posts/",
    response_model=PostListResponse,
    summary="List all LinkedIn posts",
)
async def list_posts(
    current_user: CurrentUser,
    db: DBSession,
    pagination: Pagination,
    status: list[str] = Query(default=[], description="Filter by status"),
    category: list[str] = Query(default=[]),
    sort_by: str = Query(default="created_at", description="created_at | scheduled_at | published_at | likes"),
    sort_order: str = Query(default="desc"),
) -> PostListResponse:
    """Return paginated list of LinkedIn posts for the current user."""
    from app.repositories.linkedin_repository import LinkedInRepository

    repo = LinkedInRepository(db)
    posts, total = await repo.list_by_user(
        user_id=current_user.id,
        status_filter=status or None,
        category_filter=category or None,
        offset=pagination.offset,
        limit=pagination.limit,
        sort_by=sort_by,
        sort_order=sort_order,
    )

    return PostListResponse(
        items=[_to_summary(p) for p in posts],
        total=total,
        page=pagination.page,
        page_size=pagination.page_size,
        has_next=pagination.offset + pagination.page_size < total,
    )


# ---------------------------------------------------------------------------
# GET /linkedin/posts/{id}
# ---------------------------------------------------------------------------

@router.get(
    "/posts/{post_id}",
    response_model=PostDetail,
    summary="Get single LinkedIn post detail",
)
async def get_post(post_id: str, current_user: CurrentUser, db: DBSession) -> PostDetail:
    post = await _get_post_or_404(db, post_id, current_user.id)
    return _to_detail(post)


# ---------------------------------------------------------------------------
# PATCH /linkedin/posts/{id}
# ---------------------------------------------------------------------------

@router.patch(
    "/posts/{post_id}",
    response_model=PostDetail,
    summary="Update post content or schedule",
)
async def update_post(
    post_id: str,
    payload: UpdatePostRequest,
    current_user: CurrentUser,
    db: DBSession,
) -> PostDetail:
    """Update a draft or scheduled post. Published posts cannot be edited."""
    post = await _get_post_or_404(db, post_id, current_user.id)

    if post.status == "published":
        raise ValidationException(
            "Published posts cannot be edited. Create a new post instead.",
            field="status",
        )

    if payload.content is not None:
        post.content = payload.content
        post.character_count = len(payload.content)
        post.word_count = len(payload.content.split())

    if payload.title is not None:
        post.title = payload.title

    if payload.hashtags is not None:
        if len(payload.hashtags) > LINKEDIN_POST_HASHTAG_LIMIT:
            raise ValidationException(
                f"Maximum {LINKEDIN_POST_HASHTAG_LIMIT} hashtags allowed.",
                field="hashtags",
            )
        post.hashtags = payload.hashtags

    if payload.scheduled_at is not None:
        try:
            dt = datetime.fromisoformat(payload.scheduled_at.replace("Z", "+00:00"))
        except ValueError:
            raise ValidationException("Invalid scheduled_at format.", field="scheduled_at")
        post.scheduled_at = dt
        post.status = "scheduled"

    if payload.requires_approval is not None:
        post.requires_approval = payload.requires_approval

    await db.flush()
    return _to_detail(post)


# ---------------------------------------------------------------------------
# DELETE /linkedin/posts/{id}
# ---------------------------------------------------------------------------

@router.delete(
    "/posts/{post_id}",
    status_code=204,
    response_model=None,
    summary="Delete draft or cancel scheduled post",
)
async def delete_post(post_id: str, current_user: CurrentUser, db: DBSession) -> None:
    post = await _get_post_or_404(db, post_id, current_user.id)

    if post.status == "published":
        raise ValidationException("Published posts cannot be deleted from this API.")

    post.soft_delete()
    post.status = "cancelled"
    await db.flush()


# ---------------------------------------------------------------------------
# POST /linkedin/posts/{id}/publish
# ---------------------------------------------------------------------------

@router.post(
    "/posts/{post_id}/publish",
    response_model=PostDetail,
    summary="Publish a post immediately to LinkedIn",
    dependencies=[Depends(RateLimiter(limit=5, window=3600, key_prefix="li_publish"))],
)
async def publish_post(
    post_id: str,
    current_user: CurrentUser,
    db: DBSession,
) -> PostDetail:
    """
    Publish a draft or approved post immediately to LinkedIn.

    Requires LinkedIn to be connected (OAuth tokens present and valid).
    The post is submitted via the LinkedIn Share API or Playwright fallback.
    Engagement metrics are synced periodically after publication.
    """
    _require_linkedin_connected(current_user)

    post = await _get_post_or_404(db, post_id, current_user.id)

    if post.status == "published":
        raise ConflictException("This post is already published.")

    if post.requires_approval and not post.approved_at:
        raise ValidationException(
            "This post requires approval before publishing. "
            "Use POST /linkedin/posts/{id}/approve first.",
        )

    from app.services.linkedin_service import LinkedInService
    svc = LinkedInService()

    post.status = "publishing"
    post.publish_attempts += 1
    await db.flush()

    try:
        linkedin_post_id, post_url = await svc.publish_post(
            user=current_user,
            post=post,
        )
        post.status = "published"
        post.published_at = datetime.now(timezone.utc)
        post.linkedin_post_id = linkedin_post_id
        post.linkedin_post_url = post_url
        post.publish_error = None
    except Exception as exc:
        post.status = "failed"
        post.failed_at = datetime.now(timezone.utc)
        post.publish_error = str(exc)
        await db.flush()
        raise

    await db.flush()
    logger.info(
        "LinkedIn post published",
        post_id=post_id,
        linkedin_id=linkedin_post_id,
        user_id=str(current_user.id),
    )
    return _to_detail(post)


# ---------------------------------------------------------------------------
# POST /linkedin/posts/{id}/schedule
# ---------------------------------------------------------------------------

@router.post(
    "/posts/{post_id}/schedule",
    response_model=PostDetail,
    summary="Schedule a post for a specific time",
)
async def schedule_post(
    post_id: str,
    payload: SchedulePostRequest,
    current_user: CurrentUser,
    db: DBSession,
) -> PostDetail:
    """Schedule a draft post for publishing at a specific future time."""
    _require_linkedin_connected(current_user)

    post = await _get_post_or_404(db, post_id, current_user.id)

    if post.status == "published":
        raise ConflictException("Post is already published.")

    dt = datetime.fromisoformat(payload.scheduled_at.replace("Z", "+00:00"))
    post.scheduled_at = dt
    post.status = "scheduled"
    await db.flush()

    logger.info("LinkedIn post scheduled", post_id=post_id, scheduled_at=payload.scheduled_at)
    return _to_detail(post)


# ---------------------------------------------------------------------------
# POST /linkedin/posts/{id}/approve
# ---------------------------------------------------------------------------

@router.post(
    "/posts/{post_id}/approve",
    response_model=PostDetail,
    summary="Approve a draft post for publishing",
)
async def approve_post(
    post_id: str,
    current_user: CurrentUser,
    db: DBSession,
) -> PostDetail:
    """Mark a post as approved by the user, enabling auto-publish on schedule."""
    post = await _get_post_or_404(db, post_id, current_user.id)

    if post.status not in ("draft", "pending_approval"):
        raise ValidationException(f"Cannot approve a post with status '{post.status}'.")

    post.approved_at = datetime.now(timezone.utc)
    post.status = "scheduled" if post.scheduled_at else "draft"
    await db.flush()

    return _to_detail(post)


# ---------------------------------------------------------------------------
# GET /linkedin/posts/{id}/analytics
# ---------------------------------------------------------------------------

@router.get(
    "/posts/{post_id}/analytics",
    response_model=PostAnalyticsResponse,
    summary="Per-post engagement analytics",
)
async def get_post_analytics(
    post_id: str,
    current_user: CurrentUser,
    db: DBSession,
) -> PostAnalyticsResponse:
    """
    Return engagement metrics for a specific post.

    Metrics are synced from LinkedIn periodically (every 6h for posts < 7 days old).
    On-demand sync is triggered if last sync > 1h ago.
    """
    post = await _get_post_or_404(db, post_id, current_user.id)

    if not post.is_published:
        raise ValidationException("Analytics are only available for published posts.")

    # Trigger sync if stale
    from app.services.linkedin_service import LinkedInService
    svc = LinkedInService()
    from datetime import timedelta

    should_sync = (
        post.metrics_last_synced_at is None
        or (datetime.now(timezone.utc) - post.metrics_last_synced_at) > timedelta(hours=1)
    )
    if should_sync and current_user.has_linkedin_connected:
        try:
            metrics = await svc.fetch_post_analytics(
                user=current_user,
                linkedin_post_id=post.linkedin_post_id,
            )
            post.impressions = metrics.get("impressions", post.impressions)
            post.likes = metrics.get("likes", post.likes)
            post.comments = metrics.get("comments", post.comments)
            post.shares = metrics.get("shares", post.shares)
            post.clicks = metrics.get("clicks", post.clicks)
            post.metrics_last_synced_at = datetime.now(timezone.utc)
            if post.impressions:
                post.engagement_rate = round(
                    (post.likes + post.comments + post.shares) / post.impressions, 4
                )
            await db.flush()
        except Exception as exc:
            logger.warning("Analytics sync failed", post_id=post_id, error=str(exc))

    return PostAnalyticsResponse(
        post_id=post_id,
        impressions=post.impressions,
        likes=post.likes,
        comments=post.comments,
        shares=post.shares,
        clicks=post.clicks,
        profile_views_gained=post.profile_views_gained,
        engagement_rate=post.computed_engagement_rate,
        total_engagement=post.total_engagement,
        metrics_last_synced_at=(
            post.metrics_last_synced_at.isoformat() if post.metrics_last_synced_at else None
        ),
        benchmark_engagement_rate=0.035,  # LinkedIn avg ~3.5%
    )


# ---------------------------------------------------------------------------
# GET /linkedin/analytics/overview
# ---------------------------------------------------------------------------

@router.get(
    "/analytics/overview",
    response_model=OverviewAnalyticsResponse,
    summary="Account-level LinkedIn analytics summary",
)
async def get_analytics_overview(
    current_user: CurrentUser,
    db: DBSession,
    days: int = Query(default=30, ge=7, le=365),
) -> OverviewAnalyticsResponse:
    """
    Return aggregated analytics across all published posts in the given period.

    Includes weekly trend data for the dashboard sparkline chart.
    """
    from app.repositories.linkedin_repository import LinkedInRepository

    repo = LinkedInRepository(db)
    stats = await repo.get_analytics_overview(user_id=current_user.id, days=days)

    return OverviewAnalyticsResponse(
        total_posts=stats.get("total_posts", 0),
        published_posts=stats.get("published_posts", 0),
        total_impressions=stats.get("total_impressions", 0),
        total_likes=stats.get("total_likes", 0),
        total_comments=stats.get("total_comments", 0),
        total_shares=stats.get("total_shares", 0),
        avg_engagement_rate=stats.get("avg_engagement_rate", 0.0),
        best_performing_post_id=stats.get("best_post_id"),
        best_category=stats.get("best_category"),
        posts_by_status=stats.get("by_status", {}),
        posts_by_category=stats.get("by_category", {}),
        weekly_trend=stats.get("weekly_trend", []),
    )


# ---------------------------------------------------------------------------
# POST /linkedin/posts/generate-daily
# ---------------------------------------------------------------------------

@router.post(
    "/posts/generate-daily",
    response_model=AgentTaskResponse,
    status_code=202,
    summary="Trigger daily AI content pipeline",
    dependencies=[
        Depends(require_plan(UserPlan.PRO, UserPlan.ENTERPRISE)),
        Depends(RateLimiter(limit=2, window=86400, key_prefix="daily_pipeline")),
    ],
)
async def generate_daily_pipeline(
    payload: DailyPipelineRequest,
    current_user: CurrentUser,
    db: DBSession,
) -> AgentTaskResponse:
    """
    Trigger the linkedin_agent to generate and schedule posts for the next N days.

    The agent:
    1. Plans a content calendar covering all configured categories
    2. Searches for trending topics and news for each post
    3. Generates each post with category-appropriate tone
    4. Schedules posts at the configured optimal time
    5. Flags posts that require approval before publishing
    """
    agent_run = await _create_agent_run(
        db,
        user_id=current_user.id,
        input_payload={
            "categories": payload.categories,
            "schedule_time": payload.schedule_time,
            "days_ahead": payload.days_ahead,
        },
    )

    from app.workers.linkedin_tasks import generate_daily_posts_task
    task = generate_daily_posts_task.delay(
        user_id=str(current_user.id),
        agent_run_id=str(agent_run.id),
        config=payload.model_dump(),
    )
    agent_run.celery_task_id = task.id
    await db.flush()

    return AgentTaskResponse(
        task_id=task.id,
        agent_run_id=str(agent_run.id),
        message=(
            f"Daily content pipeline started for {payload.days_ahead} days. "
            "Posts will appear in draft/scheduled state once generated."
        ),
    )


# ---------------------------------------------------------------------------
# GET /linkedin/topics/suggestions
# ---------------------------------------------------------------------------

@router.get(
    "/topics/suggestions",
    response_model=list[TopicSuggestion],
    summary="AI-suggested post topics based on trends",
    dependencies=[Depends(RateLimiter(limit=10, window=3600, key_prefix="li_topics"))],
)
async def get_topic_suggestions(
    current_user: CurrentUser,
    count: int = Query(default=5, ge=1, le=15),
    category: str | None = Query(default=None),
) -> list[TopicSuggestion]:
    """
    Return AI-suggested LinkedIn post topics based on:
    - Current AI / tech industry trends (via web search)
    - User's past post categories and performance
    - Optimal content calendar diversity

    Results are ranked by trending score (0-1).
    """
    from app.services.linkedin_service import LinkedInService

    svc = LinkedInService()
    suggestions = await svc.get_topic_suggestions(
        user_id=current_user.id,
        count=count,
        category_filter=category,
    )

    return [
        TopicSuggestion(
            topic=s["topic"],
            category=s["category"],
            reason=s["reason"],
            trending_score=s["trending_score"],
            suggested_hashtags=s.get("hashtags", []),
        )
        for s in suggestions
    ]