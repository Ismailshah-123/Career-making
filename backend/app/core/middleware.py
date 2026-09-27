"""
CareerGPT — Middleware Stack
==============================
All middleware classes imported by main.py:
  - RequestLoggingMiddleware
  - SecurityHeadersMiddleware
  - RateLimitMiddleware
  - CorrelationIDMiddleware
  - ProcessTimeMiddleware

Also exports register_middleware() for convenience.
Redis rate-limit is fail-open — if Redis is down, requests pass through.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Callable

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.logging import get_logger, set_correlation_id, clear_correlation_id

logger = get_logger(__name__)

# Paths that skip rate-limit + noisy access logs
_SKIP_PATHS = frozenset({
    "/health", "/ping", "/metrics",
    "/docs", "/redoc", "/openapi.json",
    "/favicon.ico",
})

CORRELATION_ID_HEADER = "X-Correlation-ID"


# ══════════════════════════════════════════════════════════════════════════════
# 1. Correlation ID
# ══════════════════════════════════════════════════════════════════════════════

class CorrelationIDMiddleware(BaseHTTPMiddleware):
    """
    Inject/propagate X-Correlation-ID on every request.
    Reads from incoming header or generates a fresh UUID4.
    Echoes the same ID back in the response.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        cid = (
            request.headers.get(CORRELATION_ID_HEADER)
            or request.headers.get("X-Request-ID")
            or str(uuid.uuid4())
        )
        set_correlation_id(cid)
        try:
            response = await call_next(request)
        finally:
            clear_correlation_id()
        response.headers[CORRELATION_ID_HEADER] = cid
        return response


# ══════════════════════════════════════════════════════════════════════════════
# 2. Request Logging
# ══════════════════════════════════════════════════════════════════════════════

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Emit one structured log line per HTTP request with latency + status.
    Health/metrics paths are logged at DEBUG to reduce noise.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start = time.perf_counter()
        path  = request.url.path

        response: Response | None = None
        try:
            response = await call_next(request)
            return response
        except Exception as exc:
            logger.error(
                "Unhandled exception during request",
                method=request.method,
                path=path,
                error=str(exc),
                exc_info=True,
            )
            raise
        finally:
            duration_ms = round((time.perf_counter() - start) * 1000, 2)
            status      = response.status_code if response else 500

            if path in _SKIP_PATHS:
                log_fn = logger.debug
            elif status >= 500:
                log_fn = logger.error
            elif status >= 400:
                log_fn = logger.warning
            else:
                log_fn = logger.info

            log_fn(
                f"{request.method} {path} → {status}",
                method=request.method,
                path=path,
                status_code=status,
                duration_ms=duration_ms,
                client_ip=_client_ip(request),
                user_agent=request.headers.get("User-Agent", "")[:120],
            )


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


# ══════════════════════════════════════════════════════════════════════════════
# 3. Security Headers
# ══════════════════════════════════════════════════════════════════════════════

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Add OWASP-recommended security response headers to every response.
    HSTS only added in production to avoid breaking local HTTP dev.
    """

    _HEADERS: dict[str, str] = {
        "X-Content-Type-Options":  "nosniff",
        "X-Frame-Options":         "DENY",
        "X-XSS-Protection":        "1; mode=block",
        "Referrer-Policy":         "strict-origin-when-cross-origin",
        "Permissions-Policy":      "geolocation=(), microphone=(), camera=()",
    }

    _HSTS = "max-age=31536000; includeSubDomains; preload"

    _CSP = (
        "default-src 'self'; "
        "script-src 'self'; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data: https:; "
        "font-src 'self'; "
        "connect-src 'self'; "
        "frame-ancestors 'none';"
    )

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)

        for header, value in self._HEADERS.items():
            response.headers[header] = value

        response.headers["Content-Security-Policy"] = self._CSP

        # HSTS only in production (HTTPS only)
        try:
            from app.core.config import get_settings
            if get_settings().is_production:
                response.headers["Strict-Transport-Security"] = self._HSTS
        except Exception:
            pass

        return response


# ══════════════════════════════════════════════════════════════════════════════
# 4. Process Time Header
# ══════════════════════════════════════════════════════════════════════════════

class ProcessTimeMiddleware(BaseHTTPMiddleware):
    """Append X-Process-Time (ms) to all responses."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start    = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Process-Time"] = f"{(time.perf_counter() - start) * 1000:.2f}ms"
        return response


# ══════════════════════════════════════════════════════════════════════════════
# 5. Rate Limiting (Redis sliding window — fail-open)
# ══════════════════════════════════════════════════════════════════════════════

class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Redis-backed sliding window rate limiter.

    Limits per minute (configurable):
        /api/v1/auth/*  → 10 req/min  (brute-force protection)
        /api/v1/*       → settings.RATE_LIMIT_REQUESTS_PER_MINUTE
        other paths     → 200 req/min

    Fail-open: if Redis is unavailable, all requests pass through.
    Excluded paths (health, docs) always pass through.
    """

    def __init__(
        self,
        app: Any,
        *,
        requests_per_minute: int = 100,
        window_seconds: int = 60,
        exclude_paths: list[str] | None = None,
    ) -> None:
        super().__init__(app)
        self._limit          = requests_per_minute
        self._window         = window_seconds
        self._exclude_paths  = set(exclude_paths or []) | _SKIP_PATHS
        self._redis: Any     = None
        self._redis_ok       = True   # flip to False on first connect failure

    def _get_redis(self) -> Any:
        if not self._redis_ok:
            return None
        if self._redis is None:
            try:
                import redis as _redis
                from app.core.config import get_settings
                self._redis = _redis.from_url(
                    get_settings().REDIS_URL,
                    decode_responses=True,
                    socket_connect_timeout=1,
                    socket_timeout=1,
                )
                self._redis.ping()
            except Exception as exc:
                logger.debug("Redis unavailable — rate limiting disabled", error=str(exc))
                self._redis    = None
                self._redis_ok = False
        return self._redis

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        path = request.url.path

        # Skip excluded paths
        if path in self._exclude_paths:
            return await call_next(request)

        # Choose limit
        if "/auth/" in path:
            limit = 10
        else:
            limit = self._limit

        # Redis check (fail-open)
        r = self._get_redis()
        if r is not None:
            try:
                ip  = _client_ip(request)
                # Bucket key: IP + path prefix (not full path to group endpoints)
                bucket = path.split("/")[2] if path.count("/") >= 2 else "root"
                key    = f"rl:{ip}:{bucket}"

                count: int = r.incr(key)
                if count == 1:
                    r.expire(key, self._window)

                if count > limit:
                    logger.warning(
                        "Rate limit exceeded",
                        ip=ip,
                        path=path,
                        count=count,
                        limit=limit,
                    )
                    return JSONResponse(
                        status_code=429,
                        content={
                            "error":   True,
                            "code":    "RATE_LIMIT_EXCEEDED",
                            "message": "Too many requests. Please slow down.",
                            "context": {"retry_after_seconds": self._window},
                        },
                        headers={
                            "Retry-After":        str(self._window),
                            "X-RateLimit-Limit":  str(limit),
                            "X-RateLimit-Remaining": "0",
                        },
                    )

                # Attach remaining header
                response = await call_next(request)
                response.headers["X-RateLimit-Limit"]     = str(limit)
                response.headers["X-RateLimit-Remaining"] = str(max(0, limit - count))
                return response

            except Exception as exc:
                # Redis error — fail-open
                logger.debug("Rate limit check error (fail-open)", error=str(exc))

        return await call_next(request)


# ══════════════════════════════════════════════════════════════════════════════
# Exception Handlers (registered by main.py)
# ══════════════════════════════════════════════════════════════════════════════

async def app_exception_handler(request: Request, exc: Any) -> JSONResponse:
    """Handle CareerGPTError subclasses → clean JSON response."""
    from app.core.logging import get_correlation_id

    payload = {
        "error":          True,
        "code":           exc.error_code,
        "message":        exc.message,
        "context":        exc.context,
        "correlation_id": get_correlation_id(),
    }

    log_level = "warning" if exc.http_status < 500 else "error"
    getattr(logger, log_level)(
        f"App error: {exc.error_code}",
        status_code=exc.http_status,
        message=exc.message,
    )

    return JSONResponse(
        status_code=exc.http_status,
        content=payload,
        headers=exc.headers or None,
    )


async def validation_exception_handler(request: Request, exc: Any) -> JSONResponse:
    """Handle Pydantic RequestValidationError."""
    errors = []
    for error in exc.errors():
        errors.append({
            "field":   ".".join(str(loc) for loc in error["loc"]),
            "message": error["msg"],
            "type":    error["type"],
        })
    logger.warning("Request validation failed", path=request.url.path, errors=len(errors))
    return JSONResponse(
        status_code=422,
        content={
            "error":   True,
            "code":    "VALIDATION_ERROR",
            "message": "Request validation failed.",
            "context": {"errors": errors},
        },
    )


async def http_exception_handler(request: Request, exc: Any) -> JSONResponse:
    """Handle Starlette HTTPException."""
    logger.warning("HTTP exception", status_code=exc.status_code, path=request.url.path)
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error":   True,
            "code":    "HTTP_ERROR",
            "message": str(exc.detail),
            "context": {},
        },
    )


async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all for unhandled exceptions — never leak internals."""
    logger.error(
        "Unhandled exception",
        path=request.url.path,
        method=request.method,
        error=str(exc),
        exc_info=True,
    )
    return JSONResponse(
        status_code=500,
        content={
            "error":   True,
            "code":    "INTERNAL_ERROR",
            "message": "An unexpected error occurred. Our team has been notified.",
            "context": {},
        },
    )


# ══════════════════════════════════════════════════════════════════════════════
# Registration helper (used by main.py)
# ══════════════════════════════════════════════════════════════════════════════

def register_middleware(app: FastAPI) -> None:
    """
    Attach all middleware + exception handlers.
    Call from main.py create_app().
    """
    from fastapi.exceptions import RequestValidationError
    from starlette.exceptions import HTTPException as StarletteHTTPException
    from app.core.exceptions import CareerGPTError
    from app.core.config import get_settings

    settings = get_settings()

    # Exception handlers (registered first)
    app.add_exception_handler(CareerGPTError, app_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(Exception, generic_exception_handler)

    # Middleware stack (last added = outermost)
    app.add_middleware(ProcessTimeMiddleware)
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(CorrelationIDMiddleware)
    app.add_middleware(
        RateLimitMiddleware,
        requests_per_minute=settings.RATE_LIMIT_REQUESTS_PER_MINUTE,
        window_seconds=60,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.BACKEND_CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=[
            "X-Correlation-ID",
            "X-Process-Time",
            "X-RateLimit-Limit",
            "X-RateLimit-Remaining",
            "Retry-After",
        ],
    )

    logger.info(
        "Middleware registered",
        cors_origins=settings.BACKEND_CORS_ORIGINS,
        rate_limit=settings.RATE_LIMIT_REQUESTS_PER_MINUTE,
        env=settings.APP_ENV,
    )