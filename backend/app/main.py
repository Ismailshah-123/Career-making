"""
CareerGPT — FastAPI Application Entry Point
=============================================
Run:  uvicorn main:app --host 0.0.0.0 --port 8000 --reload
Prod: gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
"""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.core.config import get_settings
from app.core.exceptions import (
    CareerGPTError,
    AuthenticationError,
    AuthorizationError,
    NotFoundError,
    ValidationError as AppValidationError,
    RateLimitError,
    LLMError,
)
from app.core.logging import configure_logging, logger
from app.core.middleware import (
    RateLimitMiddleware,
    SecurityHeadersMiddleware,
    RequestLoggingMiddleware,
    CorrelationIDMiddleware,
    ProcessTimeMiddleware,
)

settings = get_settings()


# ══════════════════════════════════════════════════════════════════════════════
# Lifespan
# ══════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    startup_t = time.monotonic()

    # 1. Configure logging first
    configure_logging()
    logger.info(
        "CareerGPT starting",
        version=settings.app_version,
        env=settings.app_env,
        db=settings.DB_NAME,
    )

    # 2. PostgreSQL check
    try:
        from app.db.init_db import verify_db_connection
        await verify_db_connection()
        logger.info("✅ PostgreSQL connected", db=settings.DB_NAME)
    except Exception as exc:
        logger.critical(f"❌ PostgreSQL failed: {exc}")
        raise RuntimeError(f"Database unreachable: {exc}") from exc

    # 3. Alembic migrations (optional — controlled by env var)
    if settings.app_env != "test" and settings.run_migrations_on_startup:
        try:
            from alembic.config import Config as AlembicConfig
            from alembic import command as alembic_command
            import asyncio as _asyncio
            alembic_cfg = AlembicConfig("alembic.ini")
            await _asyncio.to_thread(alembic_command.upgrade, alembic_cfg, "head")
            logger.info("✅ Migrations applied")
        except Exception as exc:
            logger.warning(f"⚠️  Migration skipped: {exc}")

    # 4. Qdrant collections
    try:
        from app.vectorstore.qdrant_manager import get_qdrant_manager
        mgr     = get_qdrant_manager()
        results = await mgr.ensure_collections()
        logger.info("✅ Qdrant ready", collections=list(results.keys()))
    except Exception as exc:
        logger.warning(f"⚠️  Qdrant unavailable (vector search degraded): {exc}")

    # 5. Warm up embedding model
    if settings.warmup_embedding_model:
        try:
            from app.embeddings.embedder import get_embedder
            embedder = get_embedder()
            await embedder.embed("warmup")
            logger.info(
                "✅ Embedding model loaded",
                model=embedder.model_name,
                device="gpu" if embedder._use_gpu else "cpu",
            )
        except Exception as exc:
            logger.warning(f"⚠️  Embedding warmup failed: {exc}")

    # 6. Groq LLM check
    try:
        from app.services.groq_service import get_groq_service
        health = await get_groq_service().health_check()
        if health.get("status") == "ok":
            logger.info(f"✅ LLM ready", provider=health.get("provider"), model=health.get("model"))
        else:
            logger.warning("⚠️  LLM health check failed", result=health)
    except Exception as exc:
        logger.warning(f"⚠️  LLM check failed: {exc}")

    # 7. Sentry
    if settings.sentry_dsn and settings.is_production:
        try:
            import sentry_sdk
            sentry_sdk.init(
                dsn=settings.sentry_dsn,
                environment=settings.app_env,
                traces_sample_rate=0.1,
            )
            logger.info("✅ Sentry initialized")
        except ImportError:
            logger.warning("⚠️  sentry-sdk not installed")

    startup_ms = round((time.monotonic() - startup_t) * 1000)
    logger.info(
        "🚀 CareerGPT started",
        startup_ms=startup_ms,
        routes=len(app.routes),
        docs=f"http://{settings.APP_HOST}:{settings.APP_PORT}/docs",
    )

    # ── App is running ────────────────────────────────────────────────────────
    yield

    # ── Shutdown ──────────────────────────────────────────────────────────────
    logger.info("CareerGPT shutting down...")
    try:
        from app.db.session import async_engine
        await async_engine.dispose()
        logger.info("✅ DB pool closed")
    except Exception as exc:
        logger.warning(f"DB dispose warning: {exc}")
    logger.info("✅ Shutdown complete")


# ══════════════════════════════════════════════════════════════════════════════
# App Factory
# ══════════════════════════════════════════════════════════════════════════════

def create_app() -> FastAPI:
    app = FastAPI(
        title="CareerGPT API",
        description=(
            "Enterprise AI Career Platform. "
            "Automates job discovery, resume tailoring, auto-apply, "
            "LinkedIn content, and follow-up sequences."
        ),
        version=settings.app_version,
        docs_url="/docs"        if not settings.is_production else None,
        redoc_url="/redoc"      if not settings.is_production else None,
        openapi_url="/openapi.json" if not settings.is_production else None,
        lifespan=lifespan,
    )

    _add_middleware(app)
    _add_exception_handlers(app)
    _include_routers(app)
    _mount_static(app)
    _add_websocket_routes(app)
    _add_root_routes(app)

    return app


# ══════════════════════════════════════════════════════════════════════════════
# Middleware
# ══════════════════════════════════════════════════════════════════════════════

def _add_middleware(app: FastAPI) -> None:
    # Innermost first (CORS wraps everything)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.BACKEND_CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Correlation-ID", "X-Process-Time", "X-RateLimit-Remaining"],
    )
    app.add_middleware(
        RateLimitMiddleware,
        requests_per_minute=settings.RATE_LIMIT_REQUESTS_PER_MINUTE,
        window_seconds=60,
        exclude_paths=["/health", "/ping", "/metrics", "/docs", "/redoc", "/openapi.json"],
    )
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(CorrelationIDMiddleware)
    app.add_middleware(ProcessTimeMiddleware)


# ══════════════════════════════════════════════════════════════════════════════
# Exception Handlers
# ══════════════════════════════════════════════════════════════════════════════

def _add_exception_handlers(app: FastAPI) -> None:

    @app.exception_handler(AuthenticationError)
    async def auth_handler(_: Request, exc: AuthenticationError) -> JSONResponse:
        return JSONResponse(
            status_code=401,
            content=_err("UNAUTHORIZED", str(exc), exc.context),
            headers={"WWW-Authenticate": "Bearer"},
        )

    @app.exception_handler(AuthorizationError)
    async def authz_handler(_: Request, exc: AuthorizationError) -> JSONResponse:
        return JSONResponse(status_code=403, content=_err("FORBIDDEN", str(exc), exc.context))

    @app.exception_handler(NotFoundError)
    async def not_found_handler(_: Request, exc: NotFoundError) -> JSONResponse:
        return JSONResponse(status_code=404, content=_err("NOT_FOUND", str(exc), exc.context))

    @app.exception_handler(AppValidationError)
    async def validation_handler(_: Request, exc: AppValidationError) -> JSONResponse:
        return JSONResponse(status_code=422, content=_err("VALIDATION_ERROR", str(exc), exc.context))

    @app.exception_handler(RateLimitError)
    async def rate_limit_handler(_: Request, exc: RateLimitError) -> JSONResponse:
        return JSONResponse(
            status_code=429,
            content=_err("RATE_LIMIT_EXCEEDED", str(exc), exc.context),
            headers={"Retry-After": "60"},
        )

    @app.exception_handler(LLMError)
    async def llm_error_handler(_: Request, exc: LLMError) -> JSONResponse:
        return JSONResponse(
            status_code=503,
            content=_err(
                "AI_SERVICE_UNAVAILABLE",
                "AI service temporarily unavailable. Please retry.",
                {"detail": str(exc)} if not settings.is_production else {},
            ),
        )

    @app.exception_handler(CareerGPTError)
    async def app_error_handler(_: Request, exc: CareerGPTError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.http_status,
            content=_err(
                exc.error_code or "APP_ERROR",
                str(exc),
                exc.context if not settings.is_production else {},
            ),
        )

    @app.exception_handler(Exception)
    async def generic_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.error(
            "Unhandled exception",
            path=request.url.path,
            method=request.method,
            error=str(exc),
            exc_info=True,
        )
        return JSONResponse(
            status_code=500,
            content=_err(
                "INTERNAL_ERROR",
                "An unexpected error occurred." if settings.is_production else str(exc),
                {},
            ),
        )


# ══════════════════════════════════════════════════════════════════════════════
# Routers
# ══════════════════════════════════════════════════════════════════════════════

def _include_routers(app: FastAPI) -> None:
    """
    Mount the single aggregated v1 router (app/api/v1/__init__.py).

    Each route module (auth.py, users.py, ...) already declares its own
    ``prefix=`` and ``tags=`` on its ``router = APIRouter(...)``, and the
    v1 aggregator includes them as-is — so we must NOT pass prefix/tags
    again here, or every path ends up double-prefixed
    (e.g. ``/api/v1/auth/auth/login``).
    """
    from app.api.v1 import api_router

    app.include_router(api_router, prefix="/api/v1")


# ══════════════════════════════════════════════════════════════════════════════
# Static Files
# ══════════════════════════════════════════════════════════════════════════════

def _mount_static(app: FastAPI) -> None:
    try:
        upload_dir = settings.storage.upload_dir
        upload_dir.mkdir(parents=True, exist_ok=True)
        app.mount(
            "/static/uploads",
            StaticFiles(directory=str(upload_dir)),
            name="uploads",
        )
        logger.debug("Static files mounted", dir=str(upload_dir))
    except Exception as exc:
        logger.warning(f"Static files mount failed: {exc}")


# ══════════════════════════════════════════════════════════════════════════════
# WebSocket — Real-time Notifications
# ══════════════════════════════════════════════════════════════════════════════

def _add_websocket_routes(app: FastAPI) -> None:

    @app.websocket("/ws/notifications/{user_id}")
    async def notification_ws(websocket: WebSocket, user_id: str) -> None:
        """
        Real-time notification channel via Redis pub/sub.
        Frontend connects here to receive instant job alerts + app updates.
        """
        await websocket.accept()
        logger.debug("WebSocket connected", user_id=user_id)

        try:
            import asyncio
            import json as _json

            try:
                import redis.asyncio as aioredis
                r      = await aioredis.from_url(settings.REDIS_URL, decode_responses=True)
                pubsub = r.pubsub()
                await pubsub.subscribe(f"notifications:{user_id}")

                async def _listen() -> None:
                    async for msg in pubsub.listen():
                        if msg["type"] == "message":
                            await websocket.send_text(msg["data"])

                async def _heartbeat() -> None:
                    while True:
                        await asyncio.sleep(30)
                        await websocket.send_json({"type": "ping"})

                listen_task    = asyncio.create_task(_listen())
                heartbeat_task = asyncio.create_task(_heartbeat())

                try:
                    await websocket.receive_text()
                except WebSocketDisconnect:
                    pass
                finally:
                    listen_task.cancel()
                    heartbeat_task.cancel()
                    await pubsub.unsubscribe(f"notifications:{user_id}")
                    await r.aclose()

            except ImportError:
                # redis.asyncio not installed — simple keep-alive only
                logger.warning("redis.asyncio not installed — WebSocket is heartbeat-only")
                while True:
                    await asyncio.sleep(30)
                    try:
                        await websocket.send_json({"type": "ping"})
                    except Exception:
                        break

        except WebSocketDisconnect:
            pass
        except Exception as exc:
            logger.debug("WebSocket error", user_id=user_id, error=str(exc))
            try:
                await websocket.close()
            except Exception:
                pass


# ══════════════════════════════════════════════════════════════════════════════
# Root Routes
# ══════════════════════════════════════════════════════════════════════════════

def _add_root_routes(app: FastAPI) -> None:

    @app.get("/", include_in_schema=False)
    async def root() -> dict[str, Any]:
        return {
            "name":    settings.app_name,
            "version": settings.app_version,
            "status":  "running",
            "env":     settings.app_env,
            "docs":    "/docs" if not settings.is_production else None,
        }

    @app.get("/ping", include_in_schema=False)
    async def ping() -> dict[str, str]:
        """Liveness probe — always returns 200 instantly."""
        return {"status": "ok", "ts": str(time.time())}

    @app.get("/health", include_in_schema=False)
    async def health() -> dict[str, Any]:
        """Readiness probe — checks all critical dependencies."""
        from datetime import UTC, datetime
        result: dict[str, Any] = {
            "status":  "ok",
            "version": settings.app_version,
            "ts":      datetime.now(UTC).isoformat(),
            "db":      settings.DB_NAME,
            "checks":  {},
        }

        # PostgreSQL
        try:
            from app.db.init_db import verify_db_connection
            await verify_db_connection()
            result["checks"]["database"] = "ok"
        except Exception as exc:
            result["checks"]["database"] = f"error: {exc}"
            result["status"] = "degraded"

        # Redis
        try:
            import redis
            r = redis.from_url(settings.REDIS_URL, socket_connect_timeout=2)
            r.ping()
            result["checks"]["redis"] = "ok"
        except Exception as exc:
            result["checks"]["redis"] = f"error: {exc}"
            result["status"] = "degraded"

        # Qdrant (non-critical)
        try:
            from app.vectorstore.qdrant_manager import get_qdrant_manager
            report = await get_qdrant_manager().get_health_report()
            result["checks"]["qdrant"] = report.get("status", "unknown")
        except Exception as exc:
            result["checks"]["qdrant"] = f"unavailable: {exc}"

        return result


# ══════════════════════════════════════════════════════════════════════════════
# Error Response Builder
# ══════════════════════════════════════════════════════════════════════════════

def _err(
    code: str,
    message: str,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "error":      True,
        "code":       code,
        "message":    message,
        "context":    context or {},
        "request_id": str(uuid.uuid4()),
    }


# ══════════════════════════════════════════════════════════════════════════════
# App Instance
# ══════════════════════════════════════════════════════════════════════════════

app = create_app()


# ══════════════════════════════════════════════════════════════════════════════
# Dev Entry Point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.APP_HOST,
        port=settings.APP_PORT,
        reload=settings.APP_DEBUG,
        log_level="debug" if settings.APP_DEBUG else "info",
        access_log=False,   # We handle access logging in RequestLoggingMiddleware
        workers=1,
    )