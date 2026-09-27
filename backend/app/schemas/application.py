"""
CareerGPT — Application Schemas
==================================
PAGE SUMMARY:
  Pydantic schemas for full application lifecycle management.
  Used by: app/api/v1/applications.py
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import Field, field_validator

from app.schemas.common import APIModel, parse_json_dict, parse_json_list
from app.schemas.job import JobResponse


class ApplicationCreateRequest(APIModel):
    job_id:             uuid.UUID
    resume_id:          uuid.UUID | None = None
    auto_tailor:        bool = True
    auto_cover_letter:  bool = True
    auto_apply:         bool = False
    notes:              str | None = Field(default=None, max_length=2000)
    salary_expectation: int | None = Field(default=None, ge=0)


class ApplicationStatusUpdate(APIModel):
    status: str = Field(
        pattern="^(pending|applied|viewed|interview|offer|rejected|withdrawn)$"
    )
    notes: str | None = Field(default=None, max_length=2000)


class ApplicationResponse(APIModel):
    id:                    uuid.UUID
    user_id:               uuid.UUID
    job_id:                uuid.UUID
    resume_id:             uuid.UUID | None
    status:                str
    match_score:           float | None
    match_analysis:        dict | None
    cover_letter_text:     str | None
    linkedin_message:      str | None
    recruiter_email_draft: str | None
    followup_message:      str | None
    followup_count:        int
    tailored_resume_path:  str | None
    auto_applied:          bool
    ats_platform:          str | None
    application_error:     str | None
    notes:                 str | None
    salary_expectation:    int | None
    applied_at:            datetime | None
    last_status_change_at: datetime | None
    last_followup_at:      datetime | None
    created_at:            datetime
    updated_at:            datetime
    job:                   JobResponse | None

    @field_validator("match_analysis", mode="before")
    @classmethod
    def parse_analysis(cls, v: Any) -> dict:
        return parse_json_dict(v)


class PipelineStats(APIModel):
    total:     int = 0
    pending:   int = 0
    applied:   int = 0
    viewed:    int = 0
    interview: int = 0
    offer:     int = 0
    rejected:  int = 0
    withdrawn: int = 0


class ApplicationAnalytics(APIModel):
    pipeline:               PipelineStats
    response_rate_pct:      float
    interview_rate_pct:     float
    avg_match_score:        float
    weekly_activity:        dict[str, int]
    source_breakdown:       dict[str, int]
    ats_platform_breakdown: dict[str, int]
    total_auto_applied:     int
    total_manual_applied:   int


class FollowupResponse(APIModel):
    application_id:  uuid.UUID
    followup_message: str
    followup_count:  int
    generated_at:    str


class AutoApplyStatusResponse(APIModel):
    application_id: uuid.UUID
    auto_applied:   bool
    status:         str
    task_id:        str | None
    task_status:    str
    ats_platform:   str | None
    screenshots:    list[str]
    error:          str | None
    applied_at:     str | None