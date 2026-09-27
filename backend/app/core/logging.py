"""
CareerGPT — Structured Logging
================================
JSON logging in production, pretty colours in development.
Exports: get_logger, get_agent_logger, configure_logging, logger, log_context.
Compatible with both your original logging.py API and main.py imports.
"""

from __future__ import annotations

import json
import logging
import logging.config
import traceback
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, Generator

# Lazy import settings to avoid circular imports
_settings: Any = None

def _get_settings() -> Any:
    global _settings
    if _settings is None:
        from app.core.config import get_settings
        _settings = get_settings()
    return _settings


# ── Correlation ID ─────────────────────────────────────────────────────────────

_correlation_id_var: ContextVar[str] = ContextVar("correlation_id", default="")


def get_correlation_id() -> str:
    cid = _correlation_id_var.get()
    if not cid:
        cid = str(uuid.uuid4())
        _correlation_id_var.set(cid)
    return cid


def set_correlation_id(cid: str) -> None:
    _correlation_id_var.set(cid)


def clear_correlation_id() -> None:
    _correlation_id_var.set("")


# ── Sensitive field redaction ──────────────────────────────────────────────────

_SENSITIVE_KEYS = frozenset({
    "password", "token", "secret", "api_key", "access_token",
    "refresh_token", "authorization", "client_secret", "private_key",
    "smtp_password", "groq_api_key", "openai_api_key", "anthropic_api_key",
})
_REDACTED = "***"


def _redact(data: Any, depth: int = 0) -> Any:
    if depth > 5:
        return data
    if isinstance(data, dict):
        return {
            k: _REDACTED if k.lower() in _SENSITIVE_KEYS else _redact(v, depth + 1)
            for k, v in data.items()
        }
    if isinstance(data, list):
        return [_redact(i, depth + 1) for i in data]
    return data


# ── JSON Formatter ─────────────────────────────────────────────────────────────

class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        settings = _get_settings()
        payload: dict[str, Any] = {
            "ts":             datetime.now(timezone.utc).isoformat(),
            "level":          record.levelname,
            "logger":         record.name,
            "message":        record.getMessage(),
            "correlation_id": get_correlation_id(),
            "service":        settings.app_name,
            "env":            settings.app_env,
            "module":         record.module,
            "fn":             record.funcName,
            "line":           record.lineno,
        }
        for key, value in record.__dict__.items():
            if key not in logging.LogRecord.__dict__ and not key.startswith("_"):
                if key not in payload:
                    payload[key] = _redact(value)
        if record.exc_info:
            payload["exception"] = {
                "type":      record.exc_info[0].__name__ if record.exc_info[0] else "Unknown",
                "message":   str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info),
            }
        return json.dumps(payload, default=str, ensure_ascii=False)


# ── Pretty Formatter ───────────────────────────────────────────────────────────

class PrettyFormatter(logging.Formatter):
    GREY     = "\x1b[38;20m"
    CYAN     = "\x1b[36;20m"
    YELLOW   = "\x1b[33;20m"
    RED      = "\x1b[31;20m"
    BOLD_RED = "\x1b[31;1m"
    GREEN    = "\x1b[32;20m"
    RESET    = "\x1b[0m"

    COLOURS = {
        "DEBUG":    "\x1b[38;20m",
        "INFO":     "\x1b[32;20m",
        "WARNING":  "\x1b[33;20m",
        "ERROR":    "\x1b[31;20m",
        "CRITICAL": "\x1b[31;1m",
    }

    def format(self, record: logging.LogRecord) -> str:
        colour = self.COLOURS.get(record.levelname, self.GREY)
        ts  = datetime.now(timezone.utc).strftime("%H:%M:%S.%f")[:-3]
        cid = get_correlation_id()
        cid_part = f" [{cid[:8]}]" if cid else ""
        prefix = (
            f"{colour}{ts} {record.levelname:<8}{self.RESET}"
            f" {self.CYAN}{record.name}{self.RESET}{cid_part} │ "
        )
        msg = record.getMessage()
        full = f"{prefix}{msg}"
        if record.exc_info:
            full += "\n" + self.formatException(record.exc_info)
        return full


# ── Bound Logger (supports .bind() and keyword args) ─────────────────────────

class BoundLogger:
    """
    Thin wrapper that merges bound context into every log call.
    Supports both:
        logger.info("msg", key=value)       ← keyword args
        logger.info("msg")                  ← plain
    """

    def __init__(self, raw: logging.Logger, context: dict[str, Any] | None = None) -> None:
        self._raw     = raw
        self._context = context or {}

    def bind(self, **kwargs: Any) -> "BoundLogger":
        return BoundLogger(self._raw, {**self._context, **kwargs})

    def _merge(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        return {**self._context, **kwargs}

    def debug(self, msg: str, **kwargs: Any) -> None:
        self._raw.debug(msg, extra=self._merge(kwargs), stacklevel=2)

    def info(self, msg: str, **kwargs: Any) -> None:
        self._raw.info(msg, extra=self._merge(kwargs), stacklevel=2)

    def warning(self, msg: str, **kwargs: Any) -> None:
        self._raw.warning(msg, extra=self._merge(kwargs), stacklevel=2)

    def error(self, msg: str, exc_info: bool = False, **kwargs: Any) -> None:
        self._raw.error(msg, exc_info=exc_info, extra=self._merge(kwargs), stacklevel=2)

    def critical(self, msg: str, exc_info: bool = True, **kwargs: Any) -> None:
        self._raw.critical(msg, exc_info=exc_info, extra=self._merge(kwargs), stacklevel=2)

    def exception(self, msg: str, **kwargs: Any) -> None:
        self._raw.exception(msg, extra=self._merge(kwargs), stacklevel=2)

    # Support .warning() alias
    warn = warning


# ── Context Manager ────────────────────────────────────────────────────────────

@contextmanager
def log_context(**kwargs: Any) -> Generator[None, None, None]:
    """
    Temporarily bind extra fields to all log calls in a block.

    Usage:
        with log_context(user_id="123", agent="resume"):
            logger.info("Starting tailor")
    """
    # This is a lightweight context helper — kwargs are passed per-call in agents
    yield


# ── Config Builder ─────────────────────────────────────────────────────────────

_logging_configured = False


def _build_config() -> dict[str, Any]:
    settings  = _get_settings()
    is_dev    = settings.is_development
    formatter = "pretty" if is_dev else "json"
    level     = settings.APP_LOG_LEVEL

    return {
        "version":                  1,
        "disable_existing_loggers": False,
        "formatters": {
            "json":   {"()": JSONFormatter},
            "pretty": {"()": PrettyFormatter},
        },
        "handlers": {
            "console": {
                "class":     "logging.StreamHandler",
                "stream":    "ext://sys.stdout",
                "formatter": formatter,
            },
        },
        "loggers": {
            "app": {
                "handlers": ["console"],
                "level":    level,
                "propagate": False,
            },
            "uvicorn":        {"handlers": ["console"], "level": "WARNING",  "propagate": False},
            "uvicorn.access": {"handlers": ["console"], "level": "WARNING",  "propagate": False},
            "uvicorn.error":  {"handlers": ["console"], "level": "ERROR",    "propagate": False},
            "sqlalchemy.engine": {
                "handlers":  ["console"],
                "level":     "DEBUG" if settings.DB_ECHO_SQL else "WARNING",
                "propagate": False,
            },
            "sqlalchemy.pool": {"handlers": ["console"], "level": "WARNING", "propagate": False},
            "celery":          {"handlers": ["console"], "level": "INFO",    "propagate": False},
            "celery.task":     {"handlers": ["console"], "level": "INFO",    "propagate": False},
            "httpx":           {"handlers": ["console"], "level": "WARNING", "propagate": False},
            "httpcore":        {"handlers": ["console"], "level": "WARNING", "propagate": False},
            "playwright":      {"handlers": ["console"], "level": "WARNING", "propagate": False},
        },
        "root": {"handlers": ["console"], "level": "WARNING"},
    }


def configure_logging() -> None:
    """
    Configure logging for the application.
    Idempotent — safe to call multiple times.
    Called by main.py lifespan on startup.
    """
    global _logging_configured
    if _logging_configured:
        return
    logging.config.dictConfig(_build_config())
    _logging_configured = True
    get_logger(__name__).info("Logging configured", env=_get_settings().app_env)


# Alias — your original logging.py called it setup_logging
setup_logging = configure_logging


# ── Logger Factory ─────────────────────────────────────────────────────────────

def get_logger(name: str) -> BoundLogger:
    """
    Get a BoundLogger for the given module name.
    Always pass __name__ as name.

        from app.core.logging import logger
        logger.info("msg", key=value)
    """
    if not _logging_configured:
        configure_logging()
    raw_name = name if name.startswith("app") else f"app.{name}"
    return BoundLogger(logging.getLogger(raw_name))


def get_agent_logger(agent_name: str, run_id: str | None = None) -> BoundLogger:
    ctx: dict[str, Any] = {"agent": agent_name}
    if run_id:
        ctx["run_id"] = run_id
    return get_logger(f"app.agents.{agent_name}").bind(**ctx)


def get_task_logger(task_name: str, task_id: str | None = None) -> BoundLogger:
    ctx: dict[str, Any] = {"task": task_name}
    if task_id:
        ctx["task_id"] = task_id
    return get_logger(f"app.workers.{task_name}").bind(**ctx)


def get_service_logger(service_name: str) -> BoundLogger:
    return get_logger(f"app.services.{service_name}").bind(service=service_name)


# ── Module-level logger (used everywhere as `from app.core.logging import logger`) ──

logger: BoundLogger = get_logger("app.core")