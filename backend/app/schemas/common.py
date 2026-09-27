"""
CareerGPT — Common Schemas
============================
PAGE SUMMARY:
  Shared Pydantic v2 base models, pagination, error responses, and utilities.
  All domain schemas import from here for consistency.
  Never return raw ORM objects from routes — always use these schemas.

  BASE CLASSES:
    APIModel        → orm_mode=True, populate_by_name=True (all response schemas inherit)
    PaginatedResponse → standard {total, skip, limit, items} wrapper
    MessageResponse → {message, detail} for success confirmations
    ErrorResponse   → {error, message, context, request_id} matches exception handler output

  VALIDATORS:
    parse_json_field()  → parse JSON strings from DB to Python lists/dicts
    parse_uuid()        → consistent UUID string handling
    sanitize_string()   → strip whitespace, truncate dangerous inputs
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field, field_validator

T = TypeVar("T")


class APIModel(BaseModel):
    """Base schema for all ORM-backed response models."""
    model_config = ConfigDict(
        from_attributes=True,
        populate_by_name=True,
        str_strip_whitespace=True,
    )


class PaginatedResponse(BaseModel, Generic[T]):
    """Standard paginated API response wrapper."""
    total: int = Field(description="Total records matching the query")
    skip:  int = Field(description="Records skipped (offset)")
    limit: int = Field(description="Records per page")
    items: list[T] = Field(description="Page items")


class MessageResponse(BaseModel):
    """Simple success/info message response."""
    message: str
    detail:  str | None = None


class ErrorResponse(BaseModel):
    """Structured error response (mirrors exception handler output)."""
    error:      str
    message:    str
    context:    dict[str, Any] = Field(default_factory=dict)
    request_id: str | None = None


class TaskResponse(BaseModel):
    """Response for async Celery task triggers."""
    task_id:    str
    status:     str = "queued"
    message:    str = "Task queued successfully"
    poll_url:   str | None = None


class HealthResponse(BaseModel):
    """Health check response."""
    status:     str
    version:    str
    env:        str
    database:   str
    redis:      str
    qdrant:     str


def parse_json_list(v: Any) -> list:
    """Parse JSON string from DB column to Python list."""
    if v is None:
        return []
    if isinstance(v, list):
        return v
    if isinstance(v, str):
        try:
            parsed = json.loads(v)
            return parsed if isinstance(parsed, list) else []
        except Exception:
            return []
    return []


def parse_json_dict(v: Any) -> dict:
    """Parse JSON string from DB column to Python dict."""
    if v is None:
        return {}
    if isinstance(v, dict):
        return v
    if isinstance(v, str):
        try:
            parsed = json.loads(v)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}