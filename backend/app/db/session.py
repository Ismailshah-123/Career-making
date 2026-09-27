"""
app/db/session.py
=================
Async SQLAlchemy session management for the JobHunter AI platform.

Provides:
- async_engine     : Shared AsyncEngine with configured connection pool
- AsyncSessionLocal: Session factory (async context manager)
- get_db()         : FastAPI dependency — yields an AsyncSession per request
- get_db_context() : Standalone async context manager for use outside FastAPI
                     (agents, Celery tasks, CLI scripts)
- ping_database()  : Health check — raises if DB is unreachable

Session lifecycle:
- Each HTTP request gets its own session (from FastAPI dependency injection)
- Each Celery task or agent run gets its own session via get_db_context()
- Sessions are closed and returned to the pool automatically
- Commits are explicit — nothing is auto-committed

Connection pool settings come from settings.get_db_kwargs() to allow
environment-specific tuning without code changes.

Usage (FastAPI route):
    @router.get("/jobs")
    async def list_jobs(db: AsyncSession = Depends(get_db)):
        return await job_repo.list_all(db)

Usage (agent / task):
    async with get_db_context() as db:
        user = await user_repo.get_by_id(db, user_id)
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import NullPool, AsyncAdaptedQueuePool

from app.core.config import settings
from app.core.exceptions import DatabaseException
from app.core.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Engine creation
# ---------------------------------------------------------------------------

def _build_engine() -> AsyncEngine:
    """
    Create the shared AsyncEngine.

    Pool strategy:
    - NullPool for test environments (each connection is closed immediately,
      avoiding interference between test cases).
    - AsyncAdaptedQueuePool for all other environments with tuned parameters.
    """
    pool_class = NullPool if settings.APP_ENV == "test" else AsyncAdaptedQueuePool

    engine_kwargs = {
        "url": settings.DATABASE_URL,
        "echo": settings.DB_ECHO_SQL,
        "pool_pre_ping": True,               # Verify connection before use
        "future": True,
    }

    if pool_class is not NullPool:
        engine_kwargs.update(
            {
                "poolclass": pool_class,
                "pool_size": settings.DB_POOL_SIZE,
                "max_overflow": settings.DB_MAX_OVERFLOW,
                "pool_timeout": settings.DB_POOL_TIMEOUT,
                "pool_recycle": settings.DB_POOL_RECYCLE,
            }
        )
    else:
        engine_kwargs["poolclass"] = NullPool

    engine = create_async_engine(**engine_kwargs)
    logger.info(
        "Database engine created",
        url=settings.DATABASE_URL.split("@")[-1],   # hide credentials
        pool_size=settings.DB_POOL_SIZE,
        max_overflow=settings.DB_MAX_OVERFLOW,
        environment=settings.APP_ENV,
    )
    return engine


# Module-level engine singleton
async_engine: AsyncEngine = _build_engine()

# ---------------------------------------------------------------------------
# Session factory
# ---------------------------------------------------------------------------

AsyncSessionLocal: async_sessionmaker[AsyncSession] = async_sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,      # Avoid lazy-load after commit in async context
)

# ---------------------------------------------------------------------------
# FastAPI dependency
# ---------------------------------------------------------------------------

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency that provides a database session per request.

    Automatically commits on success, rolls back on exception, and always
    closes the session — even if an exception is raised mid-request.

    Usage:
        async def my_route(db: AsyncSession = Depends(get_db)):
            ...
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


# ---------------------------------------------------------------------------
# Standalone context manager (agents, tasks, CLI)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def get_db_context() -> AsyncGenerator[AsyncSession, None]:
    """
    Async context manager providing a database session outside FastAPI.

    Identical commit/rollback semantics to get_db().

    Usage:
        async with get_db_context() as db:
            result = await db.execute(select(User))
            users = result.scalars().all()
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            logger.error("Database session rolled back due to error", exc_info=True)
            raise
        finally:
            await session.close()


# ---------------------------------------------------------------------------
# Read-only session (for analytics / reporting queries)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def get_read_only_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Provide a session configured for read-only operations.

    Sets the PostgreSQL transaction to READ ONLY to prevent accidental writes
    and allow query routing to read replicas in the future.
    """
    async with AsyncSessionLocal() as session:
        try:
            await session.execute(text("SET TRANSACTION READ ONLY"))
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


# ---------------------------------------------------------------------------
# Bulk operation session (disable autoflush for batch inserts)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def get_bulk_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Session optimised for bulk insert / update operations.

    Disables autoflush and increases chunk size hints.
    Caller is responsible for explicit flush() calls between chunks.
    """
    async with AsyncSessionLocal() as session:
        session.sync_session.autoflush = False
        try:
            yield session
            await session.flush()
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


# ---------------------------------------------------------------------------
# Health / connectivity helpers
# ---------------------------------------------------------------------------

async def ping_database() -> dict[str, str]:
    """
    Execute a lightweight query to verify DB connectivity.

    Returns:
        dict with 'status' and 'version' on success.

    Raises:
        DatabaseException: If the database is unreachable.
    """
    try:
        async with async_engine.connect() as conn:
            result = await conn.execute(text("SELECT version()"))
            version = result.scalar_one()
        return {"status": "healthy", "version": str(version)}
    except Exception as exc:
        logger.error("Database ping failed", error=str(exc))
        raise DatabaseException(operation="ping", reason=str(exc)) from exc


async def get_pool_status() -> dict[str, int]:
    """Return current connection pool statistics."""
    pool = async_engine.pool
    return {
        "pool_size": pool.size(),
        "checked_in": pool.checkedin(),
        "checked_out": pool.checkedout(),
        "overflow": pool.overflow(),
        "invalid": pool.invalid(),
    }


# ---------------------------------------------------------------------------
# Engine teardown (called on application shutdown)
# ---------------------------------------------------------------------------

async def close_engine() -> None:
    """
    Dispose the engine and close all pooled connections.

    Call from the FastAPI shutdown event handler.
    """
    await async_engine.dispose()
    logger.info("Database engine disposed.")


# ---------------------------------------------------------------------------
# Transactional helper (for complex multi-step operations)
# ---------------------------------------------------------------------------

class UnitOfWork:
    """
    Explicit unit-of-work pattern for operations spanning multiple repositories.

    Usage:
        async with UnitOfWork() as uow:
            user = await uow.users.get_by_id(uow.session, user_id)
            resume = await uow.resumes.create(uow.session, data)
            # Both are in the same transaction; commit on exit.

    Note: Repositories are injected at runtime to avoid circular imports.
    """

    def __init__(self) -> None:
        self._session: AsyncSession | None = None

    @property
    def session(self) -> AsyncSession:
        if self._session is None:
            raise RuntimeError("UnitOfWork session not initialised. Use as async context manager.")
        return self._session

    async def __aenter__(self) -> "UnitOfWork":
        self._session = AsyncSessionLocal()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._session is None:
            return
        try:
            if exc_type is None:
                await self._session.commit()
            else:
                await self._session.rollback()
        finally:
            await self._session.close()
            self._session = None

    async def commit(self) -> None:
        """Explicit mid-transaction commit (use with care)."""
        if self._session:
            await self._session.commit()

    async def rollback(self) -> None:
        """Explicit rollback."""
        if self._session:
            await self._session.rollback()

    async def flush(self) -> None:
        """Flush without committing."""
        if self._session:
            await self._session.flush()


from typing import Any  # noqa: E402 (avoids circular import in type annotations above)