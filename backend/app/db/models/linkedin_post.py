"""
app/db/models/linkedin_post.py
===============================
LinkedIn post model for the JobHunter AI platform.

The linkedin_agent generates one AI post per day on configured topics
(AI trends, career advice, tech news, productivity, etc.).
Posts are scheduled, drafted, reviewed (optional), and published
via the LinkedIn API or Playwright automation.

Stores: content, scheduling state, engagement metrics, hashtags,
media attachments, performance analytics, and A/B test metadata.

Relationships:
- user → User
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import BaseModel
from app.core.constants import LinkedInPostCategory

if TYPE_CHECKING:
    from app.db.models.user import User


class LinkedInPost(BaseModel):
    """
    A LinkedIn post generated and managed by the linkedin_agent.

    Lifecycle: draft → scheduled → publishing → published / failed

    Inherits id (UUID PK), created_at, updated_at, is_deleted, deleted_at.
    """

    __tablename__ = "linked_in_posts"

    __table_args__ = (
        Index("ix_linkedin_posts_user_id", "user_id"),
        Index("ix_linkedin_posts_status", "status"),
        Index("ix_linkedin_posts_scheduled_at", "scheduled_at"),
        Index("ix_linkedin_posts_published_at", "published_at"),
        Index("ix_linkedin_posts_category", "category"),
        Index("ix_linkedin_posts_user_scheduled", "user_id", "scheduled_at"),
    )

    # -----------------------------------------------------------------------
    # Ownership
    # -----------------------------------------------------------------------

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="User this post belongs to.",
    )

    # -----------------------------------------------------------------------
    # Content
    # -----------------------------------------------------------------------

    title: Mapped[str | None] = mapped_column(
        String(256),
        nullable=True,
        comment="Internal title for the post — not published, used for dashboard display.",
    )

    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="Full post text including emojis, line breaks, and hashtags.",
    )

    content_draft: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="Draft version before user approval (differs from published content).",
    )

    content_approved: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="User-edited / approved final content (if manual review is enabled).",
    )

    hook: Mapped[str | None] = mapped_column(
        String(512),
        nullable=True,
        comment="First line / attention hook — the most critical part for LinkedIn reach.",
    )

    call_to_action: Mapped[str | None] = mapped_column(
        String(512),
        nullable=True,
        comment="Call-to-action appended at the end, e.g. 'What do you think? Comment below.'",
    )

    character_count: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="Character count of the published content (LinkedIn limit: 3000).",
    )

    word_count: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="Word count of the published content.",
    )

    # -----------------------------------------------------------------------
    # Categorisation
    # -----------------------------------------------------------------------

    category: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        default=LinkedInPostCategory.AI_INSIGHTS,
        index=True,
        comment="Post topic category — used for content scheduling diversity.",
    )

    topic: Mapped[str | None] = mapped_column(
        String(256),
        nullable=True,
        comment="Specific topic or angle, e.g. 'GPT-4o multimodal capabilities'.",
    )

    hashtags: Mapped[list[str]] = mapped_column(
        ARRAY(String(64)),
        nullable=False,
        default=list,
        server_default="{}",
        comment="Hashtags included in the post (max 5 per LinkedIn best practices).",
    )

    mentioned_companies: Mapped[list[str]] = mapped_column(
        ARRAY(String(256)),
        nullable=False,
        default=list,
        server_default="{}",
        comment="Company names mentioned in the content — for tracking and personalisation.",
    )

    source_urls: Mapped[list[str]] = mapped_column(
        ARRAY(String(2048)),
        nullable=False,
        default=list,
        server_default="{}",
        comment="Source articles or research used to inform the post content.",
    )

    # -----------------------------------------------------------------------
    # Media / Attachments
    # -----------------------------------------------------------------------

    has_image: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default="false",
        comment="True if an image is attached to the post.",
    )

    image_url: Mapped[str | None] = mapped_column(
        String(1024),
        nullable=True,
        comment="S3 / CDN URL of the attached image.",
    )

    has_document: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default="false",
        comment="True if a PDF carousel / document is attached.",
    )

    document_url: Mapped[str | None] = mapped_column(
        String(1024),
        nullable=True,
        comment="S3 / CDN URL of the attached document.",
    )

    media_alt_text: Mapped[str | None] = mapped_column(
        String(512),
        nullable=True,
        comment="Accessibility alt text for attached media.",
    )

    # -----------------------------------------------------------------------
    # Publishing State Machine
    # -----------------------------------------------------------------------

    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="draft",
        server_default="draft",
        index=True,
        comment="draft | pending_approval | scheduled | publishing | published | failed | cancelled.",
    )

    requires_approval: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default="false",
        comment="If True, the post waits for user approval before publishing.",
    )

    approved_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="When the user approved the post for publishing.",
    )

    scheduled_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
        comment="UTC timestamp when this post is scheduled to be published.",
    )

    published_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
        comment="UTC timestamp when the post was successfully published to LinkedIn.",
    )

    failed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="UTC timestamp of last publishing failure.",
    )

    publish_attempts: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default="0",
        comment="Number of publishing attempts (for retry tracking).",
    )

    publish_error: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="Error message from the most recent failed publish attempt.",
    )

    # -----------------------------------------------------------------------
    # LinkedIn Platform IDs
    # -----------------------------------------------------------------------

    linkedin_post_id: Mapped[str | None] = mapped_column(
        String(256),
        nullable=True,
        unique=True,
        comment="LinkedIn activity URN returned after successful publish, e.g. 'urn:li:share:...'.",
    )

    linkedin_post_url: Mapped[str | None] = mapped_column(
        String(1024),
        nullable=True,
        comment="Public URL of the published post.",
    )

    linkedin_author_urn: Mapped[str | None] = mapped_column(
        String(256),
        nullable=True,
        comment="LinkedIn author URN used when publishing.",
    )

    # -----------------------------------------------------------------------
    # Engagement Metrics (updated by scheduled sync job)
    # -----------------------------------------------------------------------

    impressions: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default="0",
        comment="Number of times the post was shown in feeds.",
    )

    likes: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default="0",
        comment="Total reactions (likes, celebrates, etc.).",
    )

    comments: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default="0",
        comment="Number of comments.",
    )

    shares: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default="0",
        comment="Number of reshares / reposts.",
    )

    clicks: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default="0",
        comment="Link click count (if post includes a URL).",
    )

    profile_views_gained: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default="0",
        comment="Profile view lift attributable to this post (from LinkedIn Analytics).",
    )

    engagement_rate: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment="(likes + comments + shares) / impressions — updated on each sync.",
    )

    metrics_last_synced_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="When engagement metrics were last fetched from the LinkedIn API.",
    )

    # -----------------------------------------------------------------------
    # Generation Metadata
    # -----------------------------------------------------------------------

    generation_prompt: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="The exact prompt sent to the LLM to generate this post.",
    )

    generation_model: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
        comment="LLM model used for generation, e.g. 'llama-3.3-70b-versatile'.",
    )

    generation_tokens_used: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="Total tokens consumed in generation (prompt + completion).",
    )

    tone: Mapped[str | None] = mapped_column(
        String(32),
        nullable=True,
        comment="Tone used: professional | conversational | thought-leader | educational | storytelling.",
    )

    # -----------------------------------------------------------------------
    # A/B Testing
    # -----------------------------------------------------------------------

    ab_test_group: Mapped[str | None] = mapped_column(
        String(8),
        nullable=True,
        comment="A/B test group if running split content tests: 'A' | 'B'.",
    )

    ab_test_id: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
        comment="Identifier linking paired A/B test posts.",
    )

    # -----------------------------------------------------------------------
    # Relationships
    # -----------------------------------------------------------------------

    user: Mapped["User"] = relationship(
        "User",
        back_populates="linkedin_posts",
        lazy="select",
    )

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    @property
    def is_published(self) -> bool:
        return self.status == "published" and self.published_at is not None

    @property
    def is_scheduled(self) -> bool:
        return self.status == "scheduled"

    @property
    def total_engagement(self) -> int:
        return self.likes + self.comments + self.shares + self.clicks

    @property
    def computed_engagement_rate(self) -> float:
        if self.impressions == 0:
            return 0.0
        return round(self.total_engagement / self.impressions, 4)

    @property
    def is_within_char_limit(self) -> bool:
        return len(self.content) <= 3000

    def __repr__(self) -> str:
        return (
            f"<LinkedInPost id={self.id} user_id={self.user_id} "
            f"status={self.status!r} category={self.category!r}>"
        )