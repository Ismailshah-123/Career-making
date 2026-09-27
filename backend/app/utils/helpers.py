"""
CareerGPT — General Purpose Helpers
======================================
PAGE SUMMARY:
  Miscellaneous production utilities that don't fit neatly into other
  util modules. Used across agents, services, workers, and API routes.

  FUNCTIONS:
    Retry & Resilience:
      retry_async()            → retry an async function N times with backoff
      with_timeout()           → wrap coroutine with asyncio timeout
      safe_run()               → run coroutine, return default on any error

    Data Transformation:
      flatten_list()           → [[1,2],[3,4]] → [1,2,3,4]
      chunk_list()             → split list into N-sized chunks
      deduplicate_list()       → remove duplicates preserving order
      merge_dicts()            → deep merge two dicts
      pick_keys()              → {a:1,b:2,c:3} + [a,c] → {a:1,c:3}
      omit_keys()              → remove keys from dict
      safe_get()               → nested dict access: safe_get(d,"a.b.c")
      flatten_dict()           → {"a":{"b":1}} → {"a.b":1}
      group_by()               → group list of dicts by a key

    String Helpers:
      slugify()                → "Hello World!" → "hello-world"
      camel_to_snake()         → "CamelCase" → "camel_case"
      snake_to_camel()         → "snake_case" → "snakeCase"
      truncate_middle()        → "very long...string" for display
      mask_sensitive()         → "secret" → "se***et" (for logging)
      extract_numbers()        → "salary: $120K" → [120000]
      pluralize()              → "1 job" / "5 jobs"
      title_case_smart()       → "ai engineer" → "AI Engineer"

    JSON Helpers:
      safe_json_loads()        → parse JSON, return default on error
      safe_json_dumps()        → serialize to JSON string, handle datetime
      parse_json_field()       → DB JSON string → Python object
      ensure_list()            → "a" / None / ["a"] → ["a"]
      ensure_dict()            → None / {} / {...} → dict

    Numeric Helpers:
      clamp()                  → clamp value between min and max
      percentage()             → safe_percentage(3, 10) → 30.0
      round_to()               → round to N significant figures
      format_currency()        → 120000 → "$120,000"
      format_number_short()    → 1500000 → "1.5M", 25000 → "25K"

    Async Helpers:
      run_async()              → run coroutine in sync context (Celery)
      gather_with_errors()     → gather with per-item error capture
      async_map()              → async equivalent of map() with concurrency

    Hashing & ID:
      short_id()               → generate short readable ID "a3k9m"
      deterministic_id()       → stable UUID from input string
      hash_string()            → SHA256 hex of string
      generate_otp()           → 6-digit one-time password

    Environment:
      is_production()          → True if APP_ENV=production
      is_development()         → True if APP_ENV=development
      get_version()            → app version string from settings

  USED BY: Every layer — agents, services, workers, API routes, scrapers
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
import re
import string
import unicodedata
import uuid
from datetime import datetime
from typing import Any, Callable, Iterable, TypeVar

from app.core.logging import logger

T = TypeVar("T")


# ══════════════════════════════════════════════════════════════════════════════
# Retry & Resilience
# ══════════════════════════════════════════════════════════════════════════════

async def retry_async(
    coro_fn: Callable,
    *args: Any,
    retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
    **kwargs: Any,
) -> Any:
    """
    Retry an async coroutine function with exponential backoff.

    Usage:
        result = await retry_async(some_async_fn, arg1, arg2, retries=3, delay=1.0)

    Args:
        coro_fn:    Async function to call
        retries:    Maximum retry attempts (not counting first attempt)
        delay:      Initial delay between retries in seconds
        backoff:    Multiplier applied to delay after each retry
        exceptions: Exception types that trigger retry (others propagate immediately)
    """
    last_exc: Exception | None = None
    current_delay = delay

    for attempt in range(retries + 1):
        try:
            return await coro_fn(*args, **kwargs)
        except exceptions as exc:
            last_exc = exc
            if attempt < retries:
                logger.debug(
                    f"Retry {attempt + 1}/{retries}",
                    fn=getattr(coro_fn, "__name__", str(coro_fn)),
                    error=str(exc),
                    next_delay=current_delay,
                )
                await asyncio.sleep(current_delay)
                current_delay *= backoff
            else:
                logger.warning(
                    f"All {retries + 1} attempts failed",
                    fn=getattr(coro_fn, "__name__", str(coro_fn)),
                    error=str(exc),
                )

    if last_exc:
        raise last_exc


async def with_timeout(
    coro: Any,
    timeout_seconds: float,
    *,
    default: Any = None,
    raise_on_timeout: bool = False,
) -> Any:
    """
    Run a coroutine with a timeout.
    Returns default value if timed out (or raises TimeoutError if raise_on_timeout=True).

    Usage:
        result = await with_timeout(fetch_data(), timeout_seconds=5.0, default=[])
    """
    try:
        return await asyncio.wait_for(coro, timeout=timeout_seconds)
    except asyncio.TimeoutError:
        if raise_on_timeout:
            raise
        logger.debug(f"Coroutine timed out after {timeout_seconds}s")
        return default


async def safe_run(
    coro: Any,
    *,
    default: Any = None,
    log_errors: bool = True,
) -> Any:
    """
    Run a coroutine and return default on any exception. Never raises.

    Usage:
        result = await safe_run(risky_operation(), default={})
    """
    try:
        return await coro
    except Exception as exc:
        if log_errors:
            logger.debug("safe_run caught exception", error=str(exc))
        return default


# ══════════════════════════════════════════════════════════════════════════════
# List Helpers
# ══════════════════════════════════════════════════════════════════════════════

def flatten_list(nested: list[list[T]]) -> list[T]:
    """Flatten one level of nesting: [[1,2],[3,4]] → [1,2,3,4]."""
    return [item for sublist in nested for item in sublist]


def chunk_list(lst: list[T], size: int) -> list[list[T]]:
    """
    Split a list into chunks of max `size` elements.
    Last chunk may be smaller.
    Usage: chunk_list([1,2,3,4,5], 2) → [[1,2],[3,4],[5]]
    """
    if size <= 0:
        raise ValueError("Chunk size must be positive")
    return [lst[i : i + size] for i in range(0, len(lst), size)]


def deduplicate_list(lst: list[T], *, key: Callable[[T], Any] | None = None) -> list[T]:
    """
    Remove duplicates from a list while preserving order.
    Optional key function for custom equality (e.g., by dict field).

    Usage:
        deduplicate_list([1, 2, 1, 3]) → [1, 2, 3]
        deduplicate_list(jobs, key=lambda j: j["id"]) → unique by id
    """
    seen: set[Any] = set()
    result: list[T] = []
    for item in lst:
        k = key(item) if key else item
        if k not in seen:
            seen.add(k)
            result.append(item)
    return result


def group_by(
    items: list[dict[str, Any]],
    key: str,
) -> dict[str, list[dict[str, Any]]]:
    """
    Group a list of dicts by a key value.
    Usage: group_by(applications, "status") → {"applied": [...], "pending": [...]}
    """
    result: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        k = str(item.get(key, ""))
        result.setdefault(k, []).append(item)
    return result


def batch_items(
    items: list[T],
    batch_size: int,
) -> Iterable[list[T]]:
    """
    Yield successive batches from a list.
    Usage: for batch in batch_items(jobs, 50): process(batch)
    """
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


# ══════════════════════════════════════════════════════════════════════════════
# Dict Helpers
# ══════════════════════════════════════════════════════════════════════════════

def merge_dicts(base: dict, override: dict) -> dict:
    """
    Deep merge two dicts. Override values take precedence.
    Nested dicts are merged recursively (not replaced entirely).

    Usage:
        merge_dicts({"a": {"x": 1}}, {"a": {"y": 2}}) → {"a": {"x": 1, "y": 2}}
    """
    result = dict(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = merge_dicts(result[key], val)
        else:
            result[key] = val
    return result


def pick_keys(d: dict, keys: list[str]) -> dict:
    """Return a new dict with only the specified keys. Missing keys are skipped."""
    return {k: v for k, v in d.items() if k in keys}


def omit_keys(d: dict, keys: list[str]) -> dict:
    """Return a new dict with the specified keys removed."""
    return {k: v for k, v in d.items() if k not in keys}


def safe_get(d: dict | None, path: str, default: Any = None) -> Any:
    """
    Safely access nested dict values using dot notation.
    Usage: safe_get(user, "profile.address.city", "Unknown")
    """
    if d is None:
        return default
    parts = path.split(".")
    current: Any = d
    for part in parts:
        if not isinstance(current, dict):
            return default
        current = current.get(part)
        if current is None:
            return default
    return current


def flatten_dict(
    d: dict,
    *,
    prefix: str = "",
    separator: str = ".",
) -> dict[str, Any]:
    """
    Flatten nested dict to dot-notation keys.
    Usage: flatten_dict({"a": {"b": 1}}) → {"a.b": 1}
    """
    items: dict[str, Any] = {}
    for key, val in d.items():
        new_key = f"{prefix}{separator}{key}" if prefix else key
        if isinstance(val, dict):
            items.update(flatten_dict(val, prefix=new_key, separator=separator))
        else:
            items[new_key] = val
    return items


def remove_none_values(d: dict) -> dict:
    """Remove keys with None values from a dict (shallow)."""
    return {k: v for k, v in d.items() if v is not None}


def remove_empty_values(d: dict) -> dict:
    """Remove keys with None, empty string, or empty list values."""
    return {
        k: v for k, v in d.items()
        if v is not None and v != "" and v != [] and v != {}
    }


# ══════════════════════════════════════════════════════════════════════════════
# JSON Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _json_default(obj: Any) -> Any:
    """JSON serializer for objects not serializable by default json encoder."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, uuid.UUID):
        return str(obj)
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def safe_json_loads(
    text: str | None,
    *,
    default: Any = None,
) -> Any:
    """
    Parse JSON string, returning default on any parse error.
    Never raises. Use when parsing untrusted/optional JSON data.
    """
    if not text:
        return default
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return default


def safe_json_dumps(obj: Any, *, indent: int | None = None) -> str:
    """
    Serialize to JSON string. Handles datetime, UUID, and ORM objects.
    Returns "{}" on serialization failure (never raises).
    """
    try:
        return json.dumps(obj, default=_json_default, indent=indent, ensure_ascii=False)
    except Exception as exc:
        logger.debug("JSON serialization failed", error=str(exc))
        return "{}"


def parse_json_field(value: Any, *, default: Any = None) -> Any:
    """
    Parse a value that might be a JSON string or already a Python object.
    Handles DB columns that store JSON as TEXT.

    Usage:
        skills = parse_json_field(resume.skills, default=[])
        # Works whether resume.skills is '["Python"]' or ["Python"] or None
    """
    if value is None:
        return default
    if isinstance(value, (list, dict)):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, ValueError):
            return default
    return default


def ensure_list(value: Any) -> list:
    """
    Ensure value is a list. Wraps scalars, parses JSON strings, returns [] for None.
    Usage: ensure_list(None) → [], ensure_list("a") → ["a"], ensure_list([1,2]) → [1,2]
    """
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        parsed = safe_json_loads(value)
        if isinstance(parsed, list):
            return parsed
        return [value] if value else []
    if isinstance(value, (tuple, set)):
        return list(value)
    return [value]


def ensure_dict(value: Any) -> dict:
    """Ensure value is a dict. Returns {} for None and invalid types."""
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        parsed = safe_json_loads(value)
        if isinstance(parsed, dict):
            return parsed
    return {}


# ══════════════════════════════════════════════════════════════════════════════
# String Helpers
# ══════════════════════════════════════════════════════════════════════════════

def slugify(text: str, *, separator: str = "-", max_length: int = 100) -> str:
    """
    Convert text to URL-safe slug.
    "Hello World! This is Cool" → "hello-world-this-is-cool"
    """
    if not text:
        return ""
    # Unicode normalization → ASCII fallback
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[-_\s]+", separator, text)
    text = text.strip(separator)
    return text[:max_length]


def camel_to_snake(text: str) -> str:
    """Convert CamelCase to snake_case: "CamelCase" → "camel_case"."""
    s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", text)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def snake_to_camel(text: str, *, capitalize_first: bool = False) -> str:
    """Convert snake_case to camelCase: "snake_case" → "snakeCase"."""
    parts = text.split("_")
    if not parts:
        return text
    if capitalize_first:
        return "".join(p.capitalize() for p in parts)
    return parts[0] + "".join(p.capitalize() for p in parts[1:])


def truncate_middle(
    text: str,
    max_length: int,
    *,
    placeholder: str = "...",
) -> str:
    """
    Truncate text by removing the middle section.
    Used for displaying long file paths or URLs in logs.
    "very-long-string" → "very...ring" (preserves start and end)
    """
    if not text or len(text) <= max_length:
        return text
    half = (max_length - len(placeholder)) // 2
    return text[:half] + placeholder + text[-half:]


def mask_sensitive(
    value: str,
    *,
    visible_chars: int = 2,
    mask_char: str = "*",
) -> str:
    """
    Mask a sensitive string for safe logging.
    "secret_token_here" → "se***************"
    Used when logging API keys, passwords, tokens.
    """
    if not value:
        return ""
    if len(value) <= visible_chars * 2:
        return mask_char * len(value)
    visible = min(visible_chars, len(value) // 4)
    masked_len = len(value) - visible * 2
    return value[:visible] + mask_char * masked_len + value[-visible:]


def pluralize(count: int, singular: str, plural: str | None = None) -> str:
    """
    Return singular or plural form based on count.
    Usage: pluralize(1, "job") → "1 job", pluralize(5, "job") → "5 jobs"
    """
    plural_form = plural or f"{singular}s"
    return f"{count} {singular if count == 1 else plural_form}"


def title_case_smart(text: str) -> str:
    """
    Title case with smart capitalization for known acronyms.
    "ai engineer" → "AI Engineer", "sap consultant" → "SAP Consultant"
    """
    always_upper = {
        "ai", "ml", "api", "sdk", "cli", "ui", "ux", "sap", "aws", "gcp",
        "azure", "sql", "nosql", "etl", "erp", "crm", "bi", "ci", "cd",
        "llm", "nlp", "rag", "saas", "paas", "iaas", "devops", "secops",
        "qa", "qe", "sre", "dba", "cto", "ceo", "coo", "cpo", "vp",
    }
    always_lower = {"a", "an", "the", "and", "but", "or", "for", "at", "by",
                    "in", "of", "on", "to", "up", "as"}

    words = text.lower().split()
    result: list[str] = []
    for i, word in enumerate(words):
        clean = word.strip(",.!?;:")
        suffix = word[len(clean):]
        if clean in always_upper:
            result.append(clean.upper() + suffix)
        elif clean in always_lower and i > 0:
            result.append(clean + suffix)
        else:
            result.append(clean.capitalize() + suffix)
    return " ".join(result)


def extract_numbers(text: str) -> list[float]:
    """
    Extract all numbers from a text string.
    Handles K/M suffixes: "$120K" → 120000.0, "1.5M" → 1500000.0
    """
    results: list[float] = []
    pattern = re.compile(r"\$?(\d+(?:\.\d+)?)\s*([KkMm])?")
    for match in pattern.finditer(text):
        value = float(match.group(1))
        suffix = (match.group(2) or "").upper()
        if suffix == "K":
            value *= 1_000
        elif suffix == "M":
            value *= 1_000_000
        results.append(value)
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Numeric Helpers
# ══════════════════════════════════════════════════════════════════════════════

def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp a value between min and max inclusive."""
    return max(min_val, min(max_val, value))


def safe_percentage(numerator: float, denominator: float, *, decimals: int = 1) -> float:
    """
    Calculate percentage safely (handles zero denominator).
    safe_percentage(3, 10) → 30.0
    safe_percentage(5, 0) → 0.0 (no ZeroDivisionError)
    """
    if denominator == 0:
        return 0.0
    return round((numerator / denominator) * 100, decimals)


def round_to_significant(value: float, sig_figs: int = 2) -> float:
    """Round to N significant figures."""
    if value == 0:
        return 0.0
    import math
    magnitude = math.floor(math.log10(abs(value)))
    factor = 10 ** (sig_figs - 1 - magnitude)
    return round(value * factor) / factor


def format_currency(
    amount: int | float,
    *,
    currency: str = "USD",
    decimals: int = 0,
) -> str:
    """
    Format a number as currency string.
    format_currency(120000) → "$120,000"
    format_currency(1500.50, decimals=2) → "$1,500.50"
    """
    symbols = {"USD": "$", "GBP": "£", "EUR": "€", "PKR": "₨"}
    symbol = symbols.get(currency, currency + " ")
    if decimals == 0:
        return f"{symbol}{int(amount):,}"
    return f"{symbol}{amount:,.{decimals}f}"


def format_number_short(value: int | float) -> str:
    """
    Format large numbers with K/M suffix.
    1_500_000 → "1.5M", 25_000 → "25K", 500 → "500"
    """
    val = float(value)
    if val >= 1_000_000:
        return f"{val / 1_000_000:.1f}M".rstrip("0").rstrip(".")
    if val >= 1_000:
        return f"{val / 1_000:.1f}K".rstrip("0").rstrip(".")
    return str(int(val))


# ══════════════════════════════════════════════════════════════════════════════
# Async Helpers
# ══════════════════════════════════════════════════════════════════════════════

def run_async(coro: Any) -> Any:
    """
    Run an async coroutine from synchronous context (e.g., Celery tasks).
    Creates a new event loop if none exists in the current thread.
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Already in async context — use nest_asyncio if available
            import nest_asyncio
            nest_asyncio.apply()
            return loop.run_until_complete(coro)
        return loop.run_until_complete(coro)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()


async def gather_with_errors(
    coros: list[Any],
    *,
    return_exceptions: bool = True,
) -> list[tuple[Any, Exception | None]]:
    """
    Gather multiple coroutines and capture per-item errors.
    Returns list of (result, error) tuples.
    result is None if error occurred; error is None if succeeded.

    Usage:
        results = await gather_with_errors([coro1(), coro2(), coro3()])
        for result, error in results:
            if error:
                handle_error(error)
            else:
                process(result)
    """
    raw = await asyncio.gather(*coros, return_exceptions=True)
    return [
        (item, None) if not isinstance(item, Exception)
        else (None, item)
        for item in raw
    ]


async def async_map(
    fn: Callable,
    items: list[T],
    *,
    concurrency: int = 10,
) -> list[Any]:
    """
    Async map with controlled concurrency.
    Applies async function fn to each item with max N concurrent executions.

    Usage:
        results = await async_map(process_job, jobs, concurrency=5)
    """
    semaphore = asyncio.Semaphore(concurrency)

    async def _run(item: T) -> Any:
        async with semaphore:
            return await fn(item)

    return list(await asyncio.gather(*[_run(item) for item in items]))


# ══════════════════════════════════════════════════════════════════════════════
# Hashing & ID Generation
# ══════════════════════════════════════════════════════════════════════════════

def short_id(length: int = 8) -> str:
    """
    Generate a short, URL-safe random ID.
    Uses lowercase letters + digits for readability.
    short_id(6) → "a3k9mz"
    """
    alphabet = string.ascii_lowercase + string.digits
    return "".join(random.SystemRandom().choices(alphabet, k=length))


def deterministic_id(seed: str) -> str:
    """
    Generate a deterministic UUID from an input string.
    Same input always produces same UUID (UUID v5 with DNS namespace).
    Used for stable external IDs from URLs or compound keys.
    deterministic_id("linkedin:123456") → "550e8400-e29b-41d4-a716-446655440000"
    """
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, seed))


def hash_string(text: str, *, length: int | None = None) -> str:
    """
    SHA-256 hash of a string. Returns hex digest.
    Optional length param truncates the hex output.
    hash_string("hello") → "2cf24dba5..."
    hash_string("hello", length=8) → "2cf24dba"
    """
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return digest[:length] if length else digest


def generate_otp(length: int = 6) -> str:
    """
    Generate a cryptographically secure numeric OTP.
    Returns string of N digits: "847263"
    """
    return "".join(str(random.SystemRandom().randint(0, 9)) for _ in range(length))


# ══════════════════════════════════════════════════════════════════════════════
# Environment Helpers
# ══════════════════════════════════════════════════════════════════════════════

def is_production() -> bool:
    """Return True if APP_ENV=production."""
    from app.core.config import get_settings
    return get_settings().is_production


def is_development() -> bool:
    """Return True if APP_ENV=development."""
    from app.core.config import get_settings
    return get_settings().is_development


def get_version() -> str:
    """Return the application version string from settings."""
    from app.core.config import get_settings
    return get_settings().app_version


def get_app_name() -> str:
    """Return the application name from settings."""
    from app.core.config import get_settings
    return get_settings().app_name


# ══════════════════════════════════════════════════════════════════════════════
# Miscellaneous
# ══════════════════════════════════════════════════════════════════════════════

def deep_copy_dict(d: dict) -> dict:
    """Deep copy a dict via JSON round-trip (safe for JSON-serializable data)."""
    return json.loads(json.dumps(d, default=_json_default))


def normalize_score(
    value: float,
    *,
    in_min: float = 0.0,
    in_max: float = 100.0,
    out_min: float = 0.0,
    out_max: float = 1.0,
) -> float:
    """
    Normalize a value from one range to another.
    normalize_score(75, in_min=0, in_max=100, out_min=0, out_max=1) → 0.75
    """
    if in_max == in_min:
        return out_min
    normalized = (value - in_min) / (in_max - in_min)
    return clamp(out_min + normalized * (out_max - out_min), out_min, out_max)


def build_search_query(
    keyword: str | None = None,
    *,
    fields: list[str] | None = None,
) -> dict[str, Any]:
    """
    Build a standardized search query dict for DB repositories.
    Used by API route handlers to pass search params consistently.
    """
    query: dict[str, Any] = {}
    if keyword:
        query["keyword"] = keyword.strip()
    if fields:
        query["fields"] = fields
    return query


def _json_default(obj: Any) -> Any:
    """JSON serializer for non-serializable types."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, uuid.UUID):
        return str(obj)
    raise TypeError(f"Type {type(obj).__name__} not serializable")


__all__ = [
    "retry_async",
    "with_timeout",
    "safe_run",
    "flatten_list",
    "chunk_list",
    "deduplicate_list",
    "group_by",
    "batch_items",
    "merge_dicts",
    "pick_keys",
    "omit_keys",
    "safe_get",
    "flatten_dict",
    "remove_none_values",
    "remove_empty_values",
    "safe_json_loads",
    "safe_json_dumps",
    "parse_json_field",
    "ensure_list",
    "ensure_dict",
    "slugify",
    "camel_to_snake",
    "snake_to_camel",
    "truncate_middle",
    "mask_sensitive",
    "pluralize",
    "title_case_smart",
    "extract_numbers",
    "clamp",
    "safe_percentage",
    "round_to_significant",
    "format_currency",
    "format_number_short",
    "run_async",
    "gather_with_errors",
    "async_map",
    "short_id",
    "deterministic_id",
    "hash_string",
    "generate_otp",
    "is_production",
    "is_development",
    "get_version",
    "get_app_name",
    "deep_copy_dict",
    "normalize_score",
    "build_search_query",
]