"""
app/api/v1/health.py
=====================
Health check routes for load balancers, Kubernetes probes, and monitoring.

Endpoints:
    GET /health                      — Basic liveness probe (no dependency checks)
    GET /health/live                 — Kubernetes liveness probe alias
    GET /health/ready                — Kubernetes readiness probe (checks DB + Redis)
    GET /health/detailed             — Full dependency status (admin only)

Design notes:
- /health and /health/live respond instantly with zero I/O — used by load
  balancers and Kubernetes liveness probes that fire every few seconds.
  These must never touch the database; a slow DB should not kill the pod.
- /health/ready performs real connectivity checks (DB ping, Redis ping) and
  returns 503 if either is down — Kubernetes stops routing traffic to this
  pod until it recovers, but does NOT restart it.
- /health/detailed additionally checks Qdrant and is gated behind
  superuser auth to avoid leaking infrastructure topology publicly.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.api.deps import SuperUser
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/health", tags=["Health"])

API_VERSION = "1.0.0"


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class ServiceStatus(BaseModel):
    name: str
    status: str
    latency_ms: float | None
    detail: str | None


class HealthResponse(BaseModel):
    status: str
    version: str
    environment: str
    timestamp: str
    services: list[ServiceStatus] | None = None


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

async def _timed_check(name: str, coro: Awaitable[Any]) -> tuple[ServiceStatus, bool]:
    """Run a single dependency check, capturing latency and outcome."""
    t0 = time.monotonic()
    try:
        result = await coro
        latency = round((time.monotonic() - t0) * 1000, 1)
        return (
            ServiceStatus(name=name, status="healthy", latency_ms=latency, detail=str(result) if result else None),
            True,
        )
    except Exception as exc:
        latency = round((time.monotonic() - t0) * 1000, 1)
        logger.warning(f"Health check failed: {name}", error=str(exc))
        return (
            ServiceStatus(name=name, status="unhealthy", latency_ms=latency, detail=str(exc)[:256]),
            False,
        )


async def _check_postgresql() -> str:
    from app.db.session import ping_database
    result = await ping_database()
    return result.get("status", "ok")


async def _check_redis() -> str:
    import redis.asyncio as aioredis
    r = aioredis.from_url(settings.REDIS_URL, socket_connect_timeout=2)
    try:
        await r.ping()
    finally:
        await r.aclose()
    return "pong"


async def _check_qdrant() -> str:
    from qdrant_client import AsyncQdrantClient
    client = AsyncQdrantClient(
        url=settings.QDRANT_URL,
        api_key=settings.QDRANT_API_KEY,
        timeout=3,
    )
    try:
        info = await client.get_collections()
        return f"{len(info.collections)} collections"
    finally:
        await client.close()


async def _check_groq() -> str:
    """Lightweight Groq reachability check — does not consume tokens."""
    import httpx
    async with httpx.AsyncClient(timeout=3) as client:
        resp = await client.get(
            "https://api.groq.com/openai/v1/models",
            headers={"Authorization": f"Bearer {settings.GROQ_API_KEY}"},
        )
        resp.raise_for_status()
        return f"HTTP {resp.status_code}"


# ---------------------------------------------------------------------------
# GET /health — basic liveness, zero I/O
# ---------------------------------------------------------------------------

@router.get(
    "",
    response_model=HealthResponse,
    summary="Basic liveness probe (no dependency checks)",
)
async def health_basic() -> HealthResponse:
    """
    Lightweight liveness check.

    Returns 200 immediately without touching the database, Redis, or any
    external service. This is what load balancers and uptime monitors
    should hit every few seconds — it must never be slow or flaky.
    """
    return HealthResponse(
        status="healthy",
        version=API_VERSION,
        environment=settings.APP_ENV,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


# ---------------------------------------------------------------------------
# GET /health/live — Kubernetes liveness probe alias
# ---------------------------------------------------------------------------

@router.get(
    "/live",
    response_model=HealthResponse,
    summary="Kubernetes liveness probe",
    include_in_schema=False,
)
async def liveness() -> HealthResponse:
    """
    Kubernetes liveness probe.

    Identical to /health — kept as a separate path so the K8s manifest's
    livenessProbe can target a stable, dedicated route independent of any
    future changes to the public /health endpoint's semantics.
    """
    return await health_basic()


# ---------------------------------------------------------------------------
# GET /health/ready — Kubernetes readiness probe
# ---------------------------------------------------------------------------

@router.get(
    "/ready",
    response_model=HealthResponse,
    summary="Kubernetes readiness probe — verifies DB and Redis connectivity",
    responses={
        200: {"description": "All critical dependencies are reachable"},
        503: {"description": "One or more critical dependencies are unreachable"},
    },
)
async def readiness() -> HealthResponse:
    """
    Readiness probe — checks that PostgreSQL and Redis are reachable.

    Returns 503 if either is down. Kubernetes will stop routing new traffic
    to this pod (but will NOT restart it) until both checks pass again.

    Qdrant and Groq are intentionally excluded here: they're used for
    AI features, not core request serving, so a Qdrant blip shouldn't
    take auth/CRUD traffic out of rotation. Use /health/detailed for that.
    """
    db_status, db_ok = await _timed_check("postgresql", _check_postgresql())
    redis_status, redis_ok = await _timed_check("redis", _check_redis())

    services = [db_status, redis_status]
    all_healthy = db_ok and redis_ok

    response = HealthResponse(
        status="healthy" if all_healthy else "degraded",
        version=API_VERSION,
        environment=settings.APP_ENV,
        timestamp=datetime.now(timezone.utc).isoformat(),
        services=services,
    )

    if not all_healthy:
        raise HTTPException(status_code=503, detail=response.model_dump())

    return response


# ---------------------------------------------------------------------------
# GET /health/detailed — full dependency status (admin only)
# ---------------------------------------------------------------------------

@router.get(
    "/detailed",
    response_model=HealthResponse,
    summary="[Admin] Full dependency health check",
)
async def detailed_health(admin: SuperUser) -> HealthResponse:
    """
    Full health check across every external dependency:
    PostgreSQL, Redis, Qdrant, and the Groq LLM API.

    Gated behind superuser auth — exposing infrastructure topology
    (which services exist, their latency, failure reasons) to anonymous
    callers would aid reconnaissance, so this is intentionally not public.

    Useful for on-call debugging and the internal admin dashboard.
    """
    results = await asyncio.gather(
        _timed_check("postgresql", _check_postgresql()),
        _timed_check("redis", _check_redis()),
        _timed_check("qdrant", _check_qdrant()),
        _timed_check("groq", _check_groq()),
    )

    services = [r[0] for r in results]
    all_healthy = all(r[1] for r in results)

    return HealthResponse(
        status="healthy" if all_healthy else "degraded",
        version=API_VERSION,
        environment=settings.APP_ENV,
        timestamp=datetime.now(timezone.utc).isoformat(),
        services=services,
    )