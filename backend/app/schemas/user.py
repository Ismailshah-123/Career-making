"""
CareerGPT — User Schemas
==========================
PAGE SUMMARY:
  Pydantic schemas for user profile, preferences, stats, and admin operations.
  Used by: app/api/v1/users.py, app/api/v1/analytics.py
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import EmailStr, Field, field_validator

from app.schemas.common import APIModel, parse_json_list


class UserProfileUpdate(APIModel):
    full_name:   str | None = Field(default=None, min_length=2, max_length=200)
    avatar_url:  str | None = None
    github_url:  str | None = None
    portfolio_url: str | None = None


class UserPreferencesUpdate(APIModel):
    target_roles:       list[str] | None = None
    target_locations:   list[str] | None = None
    min_salary:         int | None = Field(default=None, ge=0, le=10_000_000)
    remote_preference:  str | None = Field(
        default=None,
        pattern="^(remote|hybrid|onsite|any)$",
    )
    linkedin_url:       str | None = None


class UserResponse(APIModel):
    id:                 uuid.UUID
    email:              str
    full_name:          str
    avatar_url:         str | None
    role:               str
    plan:               str
    is_active:          bool
    is_email_verified:  bool
    linkedin_url:       str | None
    github_url:         str | None
    portfolio_url:      str | None
    target_roles:       list[str] = Field(default_factory=list)
    target_locations:   list[str] = Field(default_factory=list)
    min_salary:         int | None
    remote_preference:  str
    total_applications: int
    total_interviews:   int
    linkedin_followers: int
    created_at:         datetime
    last_login_at:      datetime | None

    @field_validator("target_roles", "target_locations", mode="before")
    @classmethod
    def parse_json(cls, v: Any) -> list:
        return parse_json_list(v)


class UserStatsResponse(APIModel):
    total_applications:     int
    applications_this_week: int
    interviews_scheduled:   int
    offers_received:        int
    linkedin_posts_published: int
    linkedin_followers:     int
    avg_match_score:        float | None
    response_rate_pct:      float
    interview_rate_pct:     float


class UserAdminResponse(UserResponse):
    """Extended user info for admin panel."""
    plan_expires_at:      datetime | None
    email_verified_at:    datetime | None
    total_resumes:        int = 0
    total_applications:   int = 0


class LinkedInTokenUpdate(APIModel):
    access_token: str
    linkedin_url: str | None = None