"""
CareerGPT — Job Schemas
=========================
PAGE SUMMARY:
  Pydantic schemas for job search, detail, scrape trigger, and analytics.
  Used by: app/api/v1/jobs.py
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import Field, field_validator

from app.schemas.common import APIModel, parse_json_list


class JobSearchRequest(APIModel):
    keyword:          str | None = None
    location:         str | None = None
    is_remote:        bool | None = None
    sources:          list[str] | None = None
    experience_level: str | None = None
    employment_type:  str | None = None
    salary_min:       int | None = Field(default=None, ge=0)
    salary_max:       int | None = Field(default=None, ge=0)
    company:          str | None = None
    visa_sponsorship: bool | None = None
    skills:           list[str] | None = None
    posted_after:     datetime | None = None
    skip:             int = Field(default=0, ge=0)
    limit:            int = Field(default=50, ge=1, le=200)


class JobScrapeRequest(APIModel):
    sources:        list[str] = Field(default=["remoteok", "indeed", "wellfound"])
    keywords:       list[str] = Field(default=["software engineer", "python developer"])
    locations:      list[str] = Field(default=["remote", "Pakistan", "United States"])
    max_per_source: int = Field(default=50, ge=1, le=500)


class JobResponse(APIModel):
    id:               uuid.UUID
    source:           str
    job_url:          str
    title:            str
    company:          str
    company_logo_url: str | None
    location:         str | None
    is_remote:        bool
    employment_type:  str | None
    experience_level: str | None
    salary_min:       int | None
    salary_max:       int | None
    salary_currency:  str | None
    skills_required:  list[str] = Field(default_factory=list)
    ai_summary:       str | None
    posted_at:        datetime | None
    is_active:        bool
    visa_sponsorship: bool

    @field_validator("skills_required", mode="before")
    @classmethod
    def parse_skills(cls, v: Any) -> list:
        return parse_json_list(v)


class JobDetailResponse(JobResponse):
    description:    str | None
    requirements:   str | None
    benefits:       str | None
    ai_keywords:    list[str] = Field(default_factory=list)
    ai_green_flags: list[str] = Field(default_factory=list)
    ai_red_flags:   list[str] = Field(default_factory=list)
    applicant_count: int | None

    @field_validator("ai_keywords", "ai_green_flags", "ai_red_flags", mode="before")
    @classmethod
    def parse_json(cls, v: Any) -> list:
        return parse_json_list(v)


class SemanticJobSearchRequest(APIModel):
    """Natural language job search powered by vector similarity."""
    query:            str = Field(min_length=3, max_length=500)
    top_k:            int = Field(default=20, ge=1, le=100)
    is_remote:        bool | None = None
    sources:          list[str] | None = None
    experience_level: str | None = None
    salary_min:       int | None = None
    exclude_applied:  bool = True


class SemanticJobResult(APIModel):
    semantic_score: float
    job:            JobResponse


class JobScrapeResponse(APIModel):
    task_id:        str
    sources:        list[str]
    status:         str = "queued"
    estimated_jobs: int
    poll_url:       str