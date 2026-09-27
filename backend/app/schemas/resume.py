"""
CareerGPT — Resume Schemas
============================
PAGE SUMMARY:
  Pydantic schemas for resume upload, analysis, tailoring, and export.
  Used by: app/api/v1/resumes.py
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import Field, field_validator

from app.schemas.common import APIModel, parse_json_list


class ResumeTailorRequest(APIModel):
    job_id:              uuid.UUID
    master_resume_id:    uuid.UUID | None = None
    optimization_level:  str = Field(
        default="aggressive",
        pattern="^(conservative|balanced|aggressive)$",
    )


class ResumeResponse(APIModel):
    id:               uuid.UUID
    user_id:          uuid.UUID
    original_filename: str
    is_master:        bool
    tailored_for_job_id: uuid.UUID | None
    optimization_level:  str | None
    file_size_bytes:  int
    mime_type:        str
    name:             str | None
    email:            str | None
    phone:            str | None
    location:         str | None
    linkedin_url:     str | None
    github_url:       str | None
    summary:          str | None
    skills:           list[str] = Field(default_factory=list)
    certifications:   list[str] = Field(default_factory=list)
    languages:        list[str] = Field(default_factory=list)
    experience_years: float | None
    education_level:  str | None
    ats_score:        float | None
    created_at:       datetime

    @field_validator("skills", "certifications", "languages", mode="before")
    @classmethod
    def parse_json(cls, v: Any) -> list:
        return parse_json_list(v)


class ResumeAnalysisResponse(APIModel):
    resume_id:           uuid.UUID
    overall_score:       float
    ats_score:           float
    impact_score:        float
    readability_score:   float
    strengths:           list[str]
    improvements:        list[str]
    missing_sections:    list[str]
    top_skills:          list[str]
    experience_years:    float
    education_level:     str | None
    word_count:          int
    recommended_roles:   list[str]
    analyzed_at:         str


class ResumeTailorResponse(APIModel):
    resume_id:           uuid.UUID
    job_id:              uuid.UUID
    match_score:         float
    ats_score:           float
    keywords_added:      list[str]
    keywords_missing:    list[str]
    improvement_summary: str
    download_url:        str
    tailored_at:         str


class ResumeATSResponse(APIModel):
    resume_id:  uuid.UUID
    ats_score:  float
    grade:      str
    scored_at:  str