"""
app/utils/datetime_utils.py
============================
Production-grade datetime utilities for the JobHunter AI platform.

All datetimes in the platform are stored as UTC in PostgreSQL (DateTime(timezone=True)).
This module provides:

1. TIMEZONE OPERATIONS
   - now_utc()              — always-UTC replacement for datetime.utcnow()
   - to_utc()               — convert any tz-aware dt to UTC
   - to_user_tz()           — convert UTC dt to a user's local timezone
   - as_utc()               — make a naive dt timezone-aware (assume UTC)

2. PARSING
   - parse_iso()            — parse ISO 8601 strings robustly (with/without Z/offset)
   - parse_date_flexible()  — parse human date strings ("Jan 2023", "2023-01", etc.)
                              used when normalising resume experience dates from LLM output
   - parse_relative_date()  — parse "3 days ago", "posted yesterday", "2h ago"
                              from job board scrapers into absolute UTC datetimes

3. FORMATTING
   - format_iso()           — consistent ISO 8601 output with Z suffix
   - format_display()       — "January 15, 2025 at 3:42 PM EST"
   - format_relative()      — "2 hours ago", "in 3 days", "yesterday"
   - format_duration_ms()   — "1m 23s", "45s", "2h 3m" — for AgentRun timing display

4. SCHEDULING HELPERS
   - next_occurrence()      — given "HH:MM" in a timezone, return the next UTC dt
                              that falls on that time (for LinkedIn post scheduling)
   - business_days_from()   — add N business days (Mon–Fri) to a date
   - is_business_hours()    — check if a UTC dt falls in business hours for a timezone
   - followup_due_date()    — compute FOLLOWUP_WAIT_DAYS from a given applied_at

5. RANGE HELPERS
   - start_of_day()         — midnight UTC for a given date
   - start_of_week()        — Monday midnight UTC
   - start_of_month()       — 1st of month midnight UTC
   - date_range()           — generate list of dates between two dates

6. DURATION COMPUTATION
   - days_since()           — how many days since a past datetime (positive int)
   - hours_until()          — hours until a future datetime (can be negative if past)
   - business_hours_between() — count business hours between two datetimes

Usage:
    from app.utils.datetime_utils import (
        now_utc, parse_iso, format_relative, next_occurrence,
        followup_due_date, parse_relative_date,
    )

    applied_at = now_utc()
    followup   = followup_due_date(applied_at)          # → UTC dt 7 days later
    display    = format_relative(applied_at)             # → "just now"
    scheduled  = next_occurrence("09:00", "America/New_York")  # → next 9am ET in UTC
"""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta, timezone, time as dt_time
from typing import Iterator

# ---------------------------------------------------------------------------
# Timezone import — zoneinfo (Python 3.9+) with backports.zoneinfo fallback
# ---------------------------------------------------------------------------
try:
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError  # type: ignore
except ImportError:
    from backports.zoneinfo import ZoneInfo, ZoneInfoNotFoundError  # type: ignore

UTC = timezone.utc
_DEFAULT_USER_TZ = "UTC"


# ═══════════════════════════════════════════════════════════════════════════
# 1. CORE TIMEZONE OPERATIONS
# ═══════════════════════════════════════════════════════════════════════════

def now_utc() -> datetime:
    """
    Return the current UTC datetime — always timezone-aware.

    Preferred over datetime.utcnow() which returns a naive datetime,
    and over datetime.now() which returns local time. Use this everywhere.
    """
    return datetime.now(UTC)


def today_utc() -> date:
    """Return today's date in UTC."""
    return datetime.now(UTC).date()


def to_utc(dt: datetime) -> datetime:
    """
    Convert any timezone-aware datetime to UTC.

    Args:
        dt: A timezone-aware datetime (any tz).

    Returns:
        The equivalent UTC datetime.

    Raises:
        ValueError: If dt is timezone-naive (ambiguous — cannot convert).
    """
    if dt.tzinfo is None:
        raise ValueError(
            "Cannot convert naive datetime to UTC. "
            "Use as_utc() to first attach a timezone assumption."
        )
    return dt.astimezone(UTC)


def as_utc(dt: datetime) -> datetime:
    """
    Attach UTC timezone to a naive datetime without conversion.

    Use when you *know* a naive datetime is already UTC (e.g. values
    read from a database that was configured to store UTC but didn't
    attach tzinfo on retrieval).
    """
    if dt.tzinfo is not None:
        return to_utc(dt)
    return dt.replace(tzinfo=UTC)


def to_user_tz(dt: datetime, timezone_str: str) -> datetime:
    """
    Convert a UTC datetime to the user's local timezone.

    Args:
        dt           : UTC datetime (tz-aware preferred; naive assumed UTC).
        timezone_str : IANA timezone string e.g. "America/New_York", "Europe/Berlin".

    Returns:
        Datetime in the user's timezone.

    Falls back to UTC if the timezone string is invalid.
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)

    try:
        user_tz = ZoneInfo(timezone_str or _DEFAULT_USER_TZ)
        return dt.astimezone(user_tz)
    except (ZoneInfoNotFoundError, KeyError):
        return dt.astimezone(UTC)


def strip_tz(dt: datetime) -> datetime:
    """Remove timezone info from a datetime (make it naive — use with care)."""
    return dt.replace(tzinfo=None)


# ═══════════════════════════════════════════════════════════════════════════
# 2. PARSING
# ═══════════════════════════════════════════════════════════════════════════

def parse_iso(dt_string: str | None) -> datetime | None:
    """
    Parse an ISO 8601 datetime string robustly.

    Handles these formats (all common in API responses and DB reads):
        "2025-01-15T09:30:00Z"
        "2025-01-15T09:30:00+00:00"
        "2025-01-15T09:30:00.123456Z"
        "2025-01-15T09:30:00"          ← naive (assumed UTC)
        "2025-01-15"                    ← date-only (assumed midnight UTC)

    Returns None for empty/None input rather than raising.
    Returns UTC datetime always.
    """
    if not dt_string:
        return None

    s = dt_string.strip()

    # Replace 'Z' suffix with '+00:00' for fromisoformat compatibility
    s = s.replace("Z", "+00:00")

    try:
        dt = datetime.fromisoformat(s)
        return as_utc(dt) if dt.tzinfo is None else to_utc(dt)
    except ValueError:
        pass

    # Date-only format "YYYY-MM-DD"
    try:
        d = date.fromisoformat(s[:10])
        return datetime(d.year, d.month, d.day, tzinfo=UTC)
    except ValueError:
        pass

    # Last-resort: try common non-ISO formats
    for fmt in (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%d/%m/%Y",
        "%m/%d/%Y",
        "%B %d, %Y",
        "%b %d, %Y",
    ):
        try:
            dt = datetime.strptime(s[:len(fmt) + 5], fmt)
            return dt.replace(tzinfo=UTC)
        except ValueError:
            continue

    return None


def parse_date_flexible(date_str: str | None) -> date | None:
    """
    Parse flexible date strings produced by LLMs and resume parsers.

    Handles:
        "2023-01"          → date(2023, 1, 1)
        "January 2023"     → date(2023, 1, 1)
        "Jan 2023"         → date(2023, 1, 1)
        "2023"             → date(2023, 1, 1)
        "Present"          → None (caller interprets as current)
        "Current"          → None
        "Now"              → None
        "2023-06-15"       → date(2023, 6, 15)

    Used when normalising resume experience start_date / end_date fields.
    """
    if not date_str:
        return None

    s = date_str.strip()

    # "Present", "Current", "Now", "Ongoing"
    if s.lower() in ("present", "current", "now", "ongoing", "today", "n/a", "-"):
        return None

    # Full ISO date
    try:
        return date.fromisoformat(s[:10])
    except ValueError:
        pass

    # YYYY-MM format
    if re.match(r"^\d{4}-\d{2}$", s):
        try:
            return date(int(s[:4]), int(s[5:7]), 1)
        except ValueError:
            pass

    # "Month YYYY" or "Mon YYYY"
    month_patterns = [
        ("%B %Y", r"^\w+ \d{4}$"),   # "January 2023"
        ("%b %Y", r"^\w{3} \d{4}$"), # "Jan 2023"
        ("%b. %Y", r"^\w{3}\. \d{4}$"),
    ]
    for fmt, pattern in month_patterns:
        if re.match(pattern, s, re.IGNORECASE):
            try:
                return datetime.strptime(s, fmt).date().replace(day=1)
            except ValueError:
                continue

    # Year only "2023"
    if re.match(r"^\d{4}$", s):
        try:
            return date(int(s), 1, 1)
        except ValueError:
            pass

    # "Q1 2023", "Q3 2024"
    q_match = re.match(r"^Q([1-4])\s+(\d{4})$", s, re.IGNORECASE)
    if q_match:
        quarter = int(q_match.group(1))
        year    = int(q_match.group(2))
        month   = (quarter - 1) * 3 + 1
        return date(year, month, 1)

    return None


def parse_relative_date(relative_str: str | None, *, reference: datetime | None = None) -> datetime | None:
    """
    Parse relative date expressions from job board scrapers into absolute UTC datetimes.

    Handles patterns like:
        "just now", "moments ago"         → reference - 1 minute
        "1 minute ago", "2 minutes ago"   → reference - N minutes
        "1 hour ago", "3h ago", "3h"      → reference - N hours
        "yesterday"                        → reference - 1 day
        "1 day ago", "2 days ago"         → reference - N days
        "posted 3 days ago"               → reference - 3 days
        "1 week ago", "2 weeks ago"       → reference - N weeks
        "1 month ago", "2 months ago"     → reference - N * 30 days
        "30+ days ago"                    → reference - 30 days

    Args:
        relative_str : The raw relative date string from a job board.
        reference    : Base datetime for "ago" calculations (defaults to now_utc()).

    Returns:
        UTC datetime approximation, or None if parsing fails.
    """
    if not relative_str:
        return None

    ref  = reference or now_utc()
    s    = relative_str.lower().strip()

    # Remove common prefixes/suffixes
    s = re.sub(r"^(posted|updated|listed|added)\s+", "", s)
    s = re.sub(r"\s+ago\s*$", "", s)
    s = s.strip()

    # "just now", "moments ago", "now"
    if s in ("just now", "moments", "now", "today", "an hour", "a minute"):
        deltas = {"just now": 1, "moments": 2, "now": 0, "today": 0,
                  "an hour": 60, "a minute": 1}
        return ref - timedelta(minutes=deltas.get(s, 0))

    # "yesterday"
    if s == "yesterday":
        return ref - timedelta(days=1)

    # "X+ days ago" (LinkedIn often shows "30+ days ago")
    plus_match = re.match(r"(\d+)\+\s*(day|week|month|hour|minute|min|hr|h|d|w|m)", s)
    if plus_match:
        n    = int(plus_match.group(1))
        unit = plus_match.group(2)
        return ref - _unit_to_delta(n, unit)

    # Standard "N unit" patterns (e.g., "3 days", "2 hours", "1 week")
    patterns = [
        r"^(\d+)\s*(second|sec|s)\b",
        r"^(\d+)\s*(minute|min|m)\b",
        r"^(\d+)\s*(hour|hr|h)\b",
        r"^(\d+)\s*(day|d)\b",
        r"^(\d+)\s*(week|wk|w)\b",
        r"^(\d+)\s*(month|mo)\b",
        r"^(\d+)\s*(year|yr|y)\b",
        r"^(a|an)\s+(second|minute|hour|day|week|month|year)\b",
    ]

    for pattern in patterns:
        match = re.match(pattern, s, re.IGNORECASE)
        if match:
            if match.group(1).lower() in ("a", "an"):
                n    = 1
                unit = match.group(2)
            else:
                n    = int(match.group(1))
                unit = match.group(2)
            return ref - _unit_to_delta(n, unit)

    return None


def _unit_to_delta(n: int, unit: str) -> timedelta:
    """Convert (N, unit_string) → timedelta. Unit string is case-insensitive."""
    u = unit.lower()
    if u in ("second", "sec", "s"):
        return timedelta(seconds=n)
    if u in ("minute", "min", "m"):
        return timedelta(minutes=n)
    if u in ("hour", "hr", "h"):
        return timedelta(hours=n)
    if u in ("day", "d"):
        return timedelta(days=n)
    if u in ("week", "wk", "w"):
        return timedelta(weeks=n)
    if u in ("month", "mo"):
        return timedelta(days=n * 30)
    if u in ("year", "yr", "y"):
        return timedelta(days=n * 365)
    return timedelta(days=n)


# ═══════════════════════════════════════════════════════════════════════════
# 3. FORMATTING
# ═══════════════════════════════════════════════════════════════════════════

def format_iso(dt: datetime | None) -> str | None:
    """
    Format a datetime as an ISO 8601 string with UTC 'Z' suffix.

    Example: datetime(2025, 1, 15, 9, 30, 0, tzinfo=UTC) → "2025-01-15T09:30:00Z"

    Returns None for None input (propagates nulls cleanly in API responses).
    """
    if dt is None:
        return None
    utc_dt = to_utc(dt) if dt.tzinfo else dt.replace(tzinfo=UTC)
    return utc_dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def format_iso_ms(dt: datetime | None) -> str | None:
    """Format with millisecond precision: "2025-01-15T09:30:00.123Z" """
    if dt is None:
        return None
    utc_dt = to_utc(dt) if dt.tzinfo else dt.replace(tzinfo=UTC)
    return utc_dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{utc_dt.microsecond // 1000:03d}Z"


def format_display(
    dt: datetime | None,
    *,
    user_timezone: str = "UTC",
    include_time: bool = True,
    include_tz_label: bool = True,
) -> str:
    """
    Format a datetime for human display in a user's timezone.

    Examples:
        format_display(dt, user_timezone="America/New_York")
        → "January 15, 2025 at 3:42 PM EST"

        format_display(dt, include_time=False)
        → "January 15, 2025"
    """
    if dt is None:
        return "—"

    local_dt = to_user_tz(dt, user_timezone)

    if include_time:
        time_str = local_dt.strftime("%-I:%M %p")          # "3:42 PM"
        date_str = local_dt.strftime("%B %-d, %Y")         # "January 15, 2025"
        result   = f"{date_str} at {time_str}"
        if include_tz_label:
            tz_abbr = local_dt.strftime("%Z")
            result  = f"{result} {tz_abbr}"
        return result
    else:
        return local_dt.strftime("%B %-d, %Y")


def format_relative(dt: datetime | None, *, reference: datetime | None = None) -> str:
    """
    Format a past or future datetime as a human-friendly relative string.

    Past:   "just now", "2 minutes ago", "3 hours ago",
            "yesterday", "3 days ago", "last week", "2 months ago"
    Future: "in 5 minutes", "in 2 hours", "in 3 days", "in 2 weeks"
    Far:    "on January 15, 2024" (more than 1 year away/past)
    """
    if dt is None:
        return "unknown"

    ref   = reference or now_utc()
    dt_u  = as_utc(dt) if dt.tzinfo is None else to_utc(dt)
    delta = dt_u - ref
    total_seconds = int(delta.total_seconds())
    is_future     = total_seconds > 0
    abs_seconds   = abs(total_seconds)

    def _ago(val: int, unit: str) -> str:
        label = f"{val} {unit}{'s' if val != 1 else ''}"
        return f"in {label}" if is_future else f"{label} ago"

    if abs_seconds < 30:
        return "just now"
    if abs_seconds < 90:
        return "in a minute" if is_future else "a minute ago"
    if abs_seconds < 3600:
        return _ago(abs_seconds // 60, "minute")
    if abs_seconds < 5400:
        return "in an hour" if is_future else "an hour ago"
    if abs_seconds < 86400:
        return _ago(abs_seconds // 3600, "hour")
    if abs_seconds < 172800:
        return "tomorrow" if is_future else "yesterday"
    if abs_seconds < 604800:
        return _ago(abs_seconds // 86400, "day")
    if abs_seconds < 1_209_600:
        return "next week" if is_future else "last week"
    if abs_seconds < 2_592_000:
        return _ago(abs_seconds // 604800, "week")
    if abs_seconds < 31_536_000:
        months = abs_seconds // 2_592_000
        return "next month" if months == 1 and is_future else \
               "last month" if months == 1 else _ago(months, "month")

    return dt_u.strftime("on %B %-d, %Y")


def format_duration_ms(milliseconds: int | float | None) -> str:
    """
    Format a duration in milliseconds as a human-readable string.

    Examples:
        450      → "450ms"
        1_234    → "1.2s"
        65_000   → "1m 5s"
        3_720_000 → "1h 2m"
        None     → "—"
    """
    if milliseconds is None:
        return "—"

    ms = int(milliseconds)
    if ms < 0:
        return "—"
    if ms < 1000:
        return f"{ms}ms"

    seconds = ms // 1000
    if seconds < 60:
        frac = ms / 1000
        return f"{frac:.1f}s"

    minutes = seconds // 60
    rem_sec = seconds % 60
    if minutes < 60:
        return f"{minutes}m {rem_sec}s" if rem_sec else f"{minutes}m"

    hours   = minutes // 60
    rem_min = minutes % 60
    return f"{hours}h {rem_min}m" if rem_min else f"{hours}h"


def format_date_range(start: date | None, end: date | None) -> str:
    """
    Format a start/end date pair as a display range.

    Examples:
        (date(2022, 3, 1), date(2024, 1, 1)) → "March 2022 – January 2024"
        (date(2023, 6, 1), None)              → "June 2023 – Present"
        (None, None)                          → "—"
    """
    def _fmt(d: date | None) -> str:
        if d is None:
            return "Present"
        return d.strftime("%B %Y")

    if start is None and end is None:
        return "—"
    return f"{_fmt(start)} – {_fmt(end)}"


# ═══════════════════════════════════════════════════════════════════════════
# 4. SCHEDULING HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def next_occurrence(
    time_str: str,
    user_timezone: str,
    *,
    reference: datetime | None = None,
    min_advance_minutes: int = 5,
) -> datetime:
    """
    Given "HH:MM" in a user's timezone, return the next UTC datetime when
    that clock time occurs.

    Used for scheduling LinkedIn posts at the user's preferred publish time
    (e.g. "09:00" in "America/New_York") and converting to UTC for Celery ETA.

    Args:
        time_str             : "HH:MM" 24-hour time string.
        user_timezone        : IANA timezone name.
        reference            : Base datetime (defaults to now_utc()).
        min_advance_minutes  : If the target time is within this many minutes
                               from reference, push to the NEXT day's occurrence.

    Returns:
        UTC datetime of the next occurrence of the specified local time.
    """
    ref = reference or now_utc()

    try:
        h, m = [int(x) for x in time_str.split(":")]
    except (ValueError, AttributeError):
        h, m = 9, 0  # fallback to 09:00

    try:
        tz = ZoneInfo(user_timezone)
    except (ZoneInfoNotFoundError, KeyError):
        tz = ZoneInfo("UTC")

    ref_local   = ref.astimezone(tz)
    target_time = dt_time(h, m, 0, tzinfo=tz)

    # Build today's occurrence in user tz
    candidate = ref_local.replace(hour=h, minute=m, second=0, microsecond=0)

    # If today's occurrence is too close or already past, push to tomorrow
    if (candidate - ref_local).total_seconds() < min_advance_minutes * 60:
        candidate = candidate + timedelta(days=1)

    return to_utc(candidate)


def business_days_from(start_date: date, n: int) -> date:
    """
    Add N business days (Monday–Friday, no holiday awareness) to start_date.

    Used for follow-up scheduling and SLA calculations.

    Args:
        start_date : Base date.
        n          : Number of business days to add (can be negative).

    Returns: Resulting date (always a weekday).
    """
    if n == 0:
        return start_date

    direction = 1 if n > 0 else -1
    remaining = abs(n)
    current   = start_date

    while remaining > 0:
        current   = current + timedelta(days=direction)
        if current.weekday() < 5:   # Monday=0, Friday=4, Saturday=5, Sunday=6
            remaining -= 1

    return current


def is_business_hours(
    dt: datetime,
    *,
    user_timezone: str = "America/New_York",
    start_hour: int = 9,
    end_hour: int = 18,
) -> bool:
    """
    Return True if dt falls within business hours (Mon–Fri, start_hour–end_hour)
    in the specified timezone.

    Used to decide whether to send outreach messages immediately or queue
    them for the next business morning (to avoid 2am LinkedIn messages).
    """
    local = to_user_tz(dt, user_timezone)
    if local.weekday() >= 5:   # Weekend
        return False
    return start_hour <= local.hour < end_hour


def next_business_hours(
    dt: datetime | None = None,
    *,
    user_timezone: str = "America/New_York",
    start_hour: int = 9,
) -> datetime:
    """
    Return the next business hours opening after dt (or now if dt is None).

    If dt is already within business hours, returns dt unchanged.
    Otherwise returns the next Monday–Friday at start_hour in the given timezone.

    Used by the outreach_agent to schedule LinkedIn messages during optimal hours.
    """
    ref = dt or now_utc()
    if is_business_hours(ref, user_timezone=user_timezone, start_hour=start_hour):
        return ref

    # Convert to user tz and advance to next business start
    try:
        tz = ZoneInfo(user_timezone)
    except (ZoneInfoNotFoundError, KeyError):
        tz = ZoneInfo("UTC")

    local = ref.astimezone(tz)

    # Move to start_hour today first
    candidate = local.replace(hour=start_hour, minute=0, second=0, microsecond=0)

    # If we're past today's start already, move to next day
    if candidate <= local:
        candidate = candidate + timedelta(days=1)

    # Skip weekends
    while candidate.weekday() >= 5:
        candidate = candidate + timedelta(days=1)

    return to_utc(candidate)


def get_optimal_send_time(
    send_type: str,
    recipient_tz: str = "UTC",
    *,
    reference: datetime | None = None,
) -> datetime:
    """
    Suggest the next good UTC datetime to send an outbound message.

    send_type="linkedin"  → next Tue/Wed/Thu at 9am local time
                             (Tuesday-Thursday 8-10am has the highest
                             LinkedIn engagement).
    send_type="email"/other → next business-hours opening (9am local,
                             Mon-Fri), via next_business_hours().
    """
    try:
        tz = ZoneInfo(recipient_tz)
    except (ZoneInfoNotFoundError, KeyError):
        tz = ZoneInfo("UTC")

    ref = reference or now_utc()

    if send_type != "linkedin":
        return next_business_hours(ref, user_timezone=recipient_tz, start_hour=9)

    local = ref.astimezone(tz)
    candidate = local.replace(hour=9, minute=0, second=0, microsecond=0)
    if candidate <= local:
        candidate += timedelta(days=1)

    # Tuesday=1, Wednesday=2, Thursday=3
    while candidate.weekday() not in (1, 2, 3):
        candidate += timedelta(days=1)

    return to_utc(candidate)


def followup_due_date(applied_at: datetime | None) -> datetime:
    """
    Compute the follow-up due datetime from an application submission time.

    Uses FOLLOWUP_WAIT_DAYS from constants (default 7 calendar days).
    If applied_at is None, returns now + FOLLOWUP_WAIT_DAYS.

    Returns: UTC datetime when the follow-up should be sent.
    """
    from app.core.constants import FOLLOWUP_WAIT_DAYS

    base = applied_at or now_utc()
    if base.tzinfo is None:
        base = base.replace(tzinfo=UTC)
    return base + timedelta(days=FOLLOWUP_WAIT_DAYS)


def stagger_datetime(
    base: datetime,
    index: int,
    *,
    stagger_seconds: int = 60,
    max_stagger_seconds: int = 3600,
) -> datetime:
    """
    Return base + (index * stagger_seconds), capped at max_stagger_seconds.

    Used when scheduling N tasks spread over time to avoid thundering-herd
    effects (e.g. 50 users' discovery tasks spread over 1 hour).
    """
    offset = min(index * stagger_seconds, max_stagger_seconds)
    return base + timedelta(seconds=offset)


# ═══════════════════════════════════════════════════════════════════════════
# 5. RANGE HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def start_of_day(dt: datetime | None = None) -> datetime:
    """Return midnight UTC for the date component of dt (or today)."""
    d = (dt or now_utc()).date()
    return datetime(d.year, d.month, d.day, tzinfo=UTC)


def end_of_day(dt: datetime | None = None) -> datetime:
    """Return 23:59:59.999999 UTC for the date component of dt (or today)."""
    d = (dt or now_utc()).date()
    return datetime(d.year, d.month, d.day, 23, 59, 59, 999_999, tzinfo=UTC)


def start_of_week(dt: datetime | None = None) -> datetime:
    """Return Monday midnight UTC for the ISO week containing dt (or today)."""
    d = (dt or now_utc()).date()
    monday = d - timedelta(days=d.weekday())
    return datetime(monday.year, monday.month, monday.day, tzinfo=UTC)


def start_of_month(dt: datetime | None = None) -> datetime:
    """Return the 1st of the month midnight UTC for dt (or today)."""
    d = (dt or now_utc()).date()
    return datetime(d.year, d.month, 1, tzinfo=UTC)


def start_of_year(dt: datetime | None = None) -> datetime:
    """Return January 1st midnight UTC for the year of dt (or today)."""
    year = (dt or now_utc()).year
    return datetime(year, 1, 1, tzinfo=UTC)


def date_range(
    start: date,
    end: date,
    *,
    inclusive: bool = True,
) -> Iterator[date]:
    """
    Generate a sequence of dates from start to end (inclusive by default).

    Usage:
        for d in date_range(date(2025, 1, 1), date(2025, 1, 7)):
            print(d)
        # 2025-01-01 through 2025-01-07
    """
    current = start
    stop    = end if inclusive else end - timedelta(days=1)
    while current <= stop:
        yield current
        current += timedelta(days=1)


def week_dates(dt: datetime | None = None) -> list[date]:
    """Return all 7 dates in the ISO week containing dt (Mon–Sun)."""
    monday = start_of_week(dt).date()
    return [monday + timedelta(days=i) for i in range(7)]


def month_boundaries(year: int, month: int) -> tuple[datetime, datetime]:
    """
    Return (start_of_month_utc, start_of_next_month_utc) for a given year/month.

    Useful for building date-range WHERE clauses in analytics queries.
    """
    from calendar import monthrange
    _, last_day = monthrange(year, month)
    start = datetime(year, month, 1, tzinfo=UTC)
    end   = datetime(year, month, last_day, 23, 59, 59, 999_999, tzinfo=UTC)
    return start, end


# ═══════════════════════════════════════════════════════════════════════════
# 6. DURATION COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════

def days_since(dt: datetime | None) -> int | None:
    """
    Return how many full days have passed since dt.

    Returns None if dt is None or in the future.
    """
    if dt is None:
        return None
    dt_u = as_utc(dt) if dt.tzinfo is None else to_utc(dt)
    delta = now_utc() - dt_u
    if delta.total_seconds() < 0:
        return None
    return delta.days


def hours_until(dt: datetime | None) -> float | None:
    """
    Return hours until dt from now. Negative means dt is in the past.
    Returns None if dt is None.
    """
    if dt is None:
        return None
    dt_u  = as_utc(dt) if dt.tzinfo is None else to_utc(dt)
    delta = dt_u - now_utc()
    return round(delta.total_seconds() / 3600, 2)


def total_seconds_between(start: datetime, end: datetime) -> float:
    """Return total seconds between two datetimes (end - start). Can be negative."""
    s = as_utc(start) if start.tzinfo is None else to_utc(start)
    e = as_utc(end)   if end.tzinfo   is None else to_utc(end)
    return (e - s).total_seconds()


def business_hours_between(
    start: datetime,
    end: datetime,
    *,
    user_timezone: str = "America/New_York",
    work_start: int = 9,
    work_end: int = 18,
) -> float:
    """
    Count the number of business hours between start and end.

    Skips weekends and hours outside work_start–work_end range.
    Used for SLA tracking (e.g. recruiter response time in business hours).

    Returns: Business hours as a float.
    """
    if start >= end:
        return 0.0

    try:
        tz = ZoneInfo(user_timezone)
    except (ZoneInfoNotFoundError, KeyError):
        tz = ZoneInfo("UTC")

    s = as_utc(start).astimezone(tz)
    e = as_utc(end).astimezone(tz)

    total_hours = 0.0
    cursor      = s

    while cursor < e:
        # Skip to next business day if on weekend
        if cursor.weekday() >= 5:
            days_ahead = 7 - cursor.weekday()
            cursor = cursor.replace(hour=work_start, minute=0, second=0, microsecond=0)
            cursor = cursor + timedelta(days=days_ahead)
            continue

        # Start of business today
        day_start = cursor.replace(hour=work_start, minute=0, second=0, microsecond=0)
        day_end   = cursor.replace(hour=work_end, minute=0, second=0, microsecond=0)

        # Clamp cursor to business window
        effective_start = max(cursor, day_start)
        effective_end   = min(e, day_end)

        if effective_start < effective_end:
            total_hours += (effective_end - effective_start).total_seconds() / 3600

        # Advance to next day business start
        cursor = day_end + timedelta(days=1)
        cursor = cursor.replace(hour=work_start, minute=0, second=0, microsecond=0)

    return round(total_hours, 2)


def estimate_years_experience(
    start_date: date | None,
    end_date: date | None,
) -> float:
    """
    Estimate years of experience for one job entry.

    Args:
        start_date : Employment start date.
        end_date   : Employment end date (None = still employed → use today).

    Returns: Years as a float, rounded to 1 decimal place. 0.0 if undetermined.
    """
    if start_date is None:
        return 0.0

    end = end_date or today_utc()
    delta_days = (end - start_date).days
    return round(max(0.0, delta_days / 365.25), 1)


def total_experience_years(
    experience_entries: list[dict],
) -> float:
    """
    Compute total career experience from a list of experience dicts.

    Handles overlapping date ranges (e.g. consulting + part-time) by
    computing the union of date intervals rather than summing durations.

    Each dict must have 'start_date' and 'end_date' keys as date objects
    or parseable strings.

    Returns: Total unique years of experience as a float.
    """
    if not experience_entries:
        return 0.0

    intervals: list[tuple[date, date]] = []
    today = today_utc()

    for entry in experience_entries:
        raw_start = entry.get("start_date")
        raw_end   = entry.get("end_date")

        start = parse_date_flexible(raw_start) if isinstance(raw_start, str) else raw_start
        end   = parse_date_flexible(raw_end)   if isinstance(raw_end, str) else raw_end

        if start is None:
            continue
        end = end or today
        if start > end:
            start, end = end, start   # swap if reversed

        intervals.append((start, end))

    if not intervals:
        return 0.0

    # Merge overlapping intervals
    intervals.sort(key=lambda x: x[0])
    merged: list[tuple[date, date]] = [intervals[0]]

    for start, end in intervals[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))

    total_days = sum((e - s).days for s, e in merged)
    return round(total_days / 365.25, 1)


# ═══════════════════════════════════════════════════════════════════════════
# CONVENIENCE EXPORTS
# ═══════════════════════════════════════════════════════════════════════════

def datetime_to_epoch(dt: datetime | None) -> int | None:
    """Convert a UTC datetime to a Unix epoch timestamp (seconds)."""
    if dt is None:
        return None
    utc = as_utc(dt) if dt.tzinfo is None else to_utc(dt)
    return int(utc.timestamp())


def epoch_to_datetime(epoch: int | float | None) -> datetime | None:
    """Convert a Unix epoch timestamp to a UTC datetime."""
    if epoch is None:
        return None
    try:
        return datetime.fromtimestamp(float(epoch), tz=UTC)
    except (ValueError, OSError, OverflowError):
        return None


def is_past(dt: datetime | None) -> bool:
    """Return True if dt is in the past (or None)."""
    if dt is None:
        return True
    return (as_utc(dt) if dt.tzinfo is None else to_utc(dt)) < now_utc()


def is_future(dt: datetime | None) -> bool:
    """Return True if dt is in the future."""
    return not is_past(dt)


def clamp_datetime(
    dt: datetime,
    *,
    minimum: datetime | None = None,
    maximum: datetime | None = None,
) -> datetime:
    """Clamp a datetime between optional minimum and maximum bounds."""
    result = dt
    if minimum is not None and result < minimum:
        result = minimum
    if maximum is not None and result > maximum:
        result = maximum
    return result