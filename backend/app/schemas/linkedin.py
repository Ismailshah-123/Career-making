"""
CareerGPT — LinkedIn Schemas
==============================
PAGE SUMMARY:
  Pydantic schemas for LinkedIn content generation and publishing.
  Used by: app/api/v1/linkedin.py
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import Field, field_validator

from app.schemas.common import APIModel, parse_json_list


class LinkedInPostRequest(APIModel):
    topic:       str | None = Field(default=None, max_length=500)
    tone:        str = Field(
        default="thought_leader",
        pattern="^(thought_leader|personal|educational|motivational)$",
    )
    include_hook: bool = True
    include_cta:  bool = True
    schedule_at:  datetime | None = None
    custom_hook:  str | None = Field(default=None, max_length=300)


class LinkedInPostUpdate(APIModel):
    body:         str | None = None
    hook:         str | None = None
    hashtags:     list[str] | None = None
    scheduled_at: datetime | None = None
    status:       str | None = Field(
        default=None,
        pattern="^(draft|scheduled)$",
    )


class LinkedInPostResponse(APIModel):
    id:              uuid.UUID
    user_id:         uuid.UUID
    topic:           str
    hook:            str
    body:            str
    hashtags:        list[str] = Field(default_factory=list)
    emoji_set:       str | None
    tone:            str | None
    status:          str
    scheduled_at:    datetime | None
    published_at:    datetime | None
    linkedin_post_id: str | None
    impressions:     int
    likes:           int
    comments:        int
    shares:          int
    publish_error:   str | None
    created_at:      datetime

    @field_validator("hashtags", mode="before")
    @classmethod
    def parse_hashtags(cls, v: Any) -> list:
        return parse_json_list(v)


class LinkedInEngagementStats(APIModel):
    total_posts:        int
    total_impressions:  int
    total_likes:        int
    total_comments:     int
    total_shares:       int
    total_engagements:  int
    avg_impressions:    float
    avg_likes:          float
    engagement_rate_pct: float


class ContentCalendarDay(APIModel):
    date:  str
    posts: list[dict]


class LinkedInPublishResponse(APIModel):
    post_id:         uuid.UUID
    linkedin_post_id: str | None
    status:          str
    published_at:    str | None
    method:          str