"""
CareerGPT — Business-Logic Validators
========================================
Input validators used by the service layer for values that go beyond
what Pydantic's schema-level typing already enforces (length limits,
character sets, list de-duplication, sane numeric ranges).

Every validator either returns a cleaned/normalized value or raises
app.core.exceptions.ValidationError — never a bare ValueError — so the
API layer's exception handler can turn it into a consistent 400
response.
"""

from __future__ import annotations

import re
from urllib.parse import urlparse

from app.core.constants import (
    FULL_NAME_MAX_LENGTH,
    FULL_NAME_MIN_LENGTH,
    MAX_SKILLS_PER_RESUME,
)
from app.core.exceptions import ValidationError

# ── Name ──────────────────────────────────────────────────────────────────────

_NAME_RE = re.compile(r"^[A-Za-z\u00C0-\u024F\u4E00-\u9FFF\s.'\-]+$")


def validate_full_name(name: str) -> str:
    """
    Clean and validate a person's display name.
    Raises ValidationError if the name is empty, too short/long, or
    contains characters that clearly aren't part of a name (digits,
    symbols other than spaces/periods/hyphens/apostrophes).
    """
    cleaned = " ".join((name or "").split())  # collapse internal whitespace

    if len(cleaned) < FULL_NAME_MIN_LENGTH:
        raise ValidationError(
            f"Full name must be at least {FULL_NAME_MIN_LENGTH} characters.",
            field="full_name",
        )
    if len(cleaned) > FULL_NAME_MAX_LENGTH:
        raise ValidationError(
            f"Full name must be under {FULL_NAME_MAX_LENGTH} characters.",
            field="full_name",
        )
    if not _NAME_RE.match(cleaned):
        raise ValidationError(
            "Full name may only contain letters, spaces, periods, hyphens and apostrophes.",
            field="full_name",
        )
    return cleaned


# ── URLs ──────────────────────────────────────────────────────────────────────

def normalize_url(url: str) -> str:
    """
    Normalize a user-supplied URL (LinkedIn/GitHub/portfolio profile link).

    - Empty input returns "" (these fields are optional).
    - A bare domain/handle ("linkedin.com/in/x") gets "https://" prepended.
    - Raises ValidationError if the result still isn't a plausible URL.
    """
    raw = (url or "").strip()
    if not raw:
        return ""

    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://", raw):
        raw = f"https://{raw}"

    parsed = urlparse(raw)
    if not parsed.netloc or "." not in parsed.netloc:
        raise ValidationError(f"'{url}' does not look like a valid URL.", field="url")

    return raw


# ── List fields (roles / locations / skills) ──────────────────────────────────

def _clean_string_list(
    values: list[str] | None,
    *,
    field: str,
    max_items: int,
    max_item_length: int = 100,
) -> list[str]:
    if not values:
        return []

    cleaned: list[str] = []
    seen: set[str] = set()
    for raw in values:
        item = " ".join((raw or "").split())
        if not item:
            continue
        if len(item) > max_item_length:
            item = item[:max_item_length]
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(item)

    if len(cleaned) > max_items:
        raise ValidationError(
            f"A maximum of {max_items} {field} is supported.",
            field=field,
        )
    return cleaned


def validate_target_roles(roles: list[str] | None) -> list[str]:
    """Clean and de-duplicate a list of desired job-title strings."""
    return _clean_string_list(roles, field="target_roles", max_items=25, max_item_length=100)


def validate_locations_list(locations: list[str] | None) -> list[str]:
    """Clean and de-duplicate a list of desired work locations."""
    return _clean_string_list(locations, field="target_locations", max_items=25, max_item_length=100)


def validate_skills_list(skills: list[str] | None) -> list[str]:
    """Clean and de-duplicate a list of skill strings."""
    return _clean_string_list(
        skills, field="skills", max_items=MAX_SKILLS_PER_RESUME, max_item_length=60
    )


# ── Numeric fields ─────────────────────────────────────────────────────────────

def validate_salary(value: int | None, min_value: int = 0, max_value: int = 100_000_000) -> int | None:
    """Validate a salary figure is a sane non-negative integer."""
    if value is None:
        return None
    if value < min_value:
        raise ValidationError(f"Salary cannot be below {min_value}.", field="salary")
    if value > max_value:
        raise ValidationError(f"Salary of {value} looks implausible.", field="salary")
    return int(value)


def validate_experience_years(years: float | None, min_value: float = 0, max_value: float = 60) -> float | None:
    """Validate a years-of-experience figure is within a plausible range."""
    if years is None:
        return None
    if years < min_value or years > max_value:
        raise ValidationError(
            f"Experience years must be between {min_value} and {max_value}.",
            field="experience_years",
        )
    return float(years)
