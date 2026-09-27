"""
CareerGPT — Database Initialisation
======================================
Called by main.py lifespan on startup.
verify_db_connection() → raises RuntimeError if DB unreachable.
"""

from __future__ import annotations

from app.core.logging import get_logger

logger = get_logger(__name__)


async def verify_db_connection() -> None:
    """
    Verify PostgreSQL is reachable and the database exists.
    Raises RuntimeError with a clear message if connection fails.
    Called once at startup — fast fail before accepting traffic.
    """
    try:
        from sqlalchemy.ext.asyncio import create_async_engine
        from sqlalchemy import text
        from app.core.config import get_settings

        settings = get_settings()
        engine   = create_async_engine(
            settings.DATABASE_URL,
            echo=False,
            pool_pre_ping=True,
            connect_args={"timeout": 5},
        )
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        await engine.dispose()
        logger.debug("DB connectivity verified", db=settings.DB_NAME)

    except Exception as exc:
        raise RuntimeError(
            f"Cannot connect to PostgreSQL database '{get_settings().DB_NAME}'. "
            f"Make sure Docker is running and .env has correct DB_* values. "
            f"Error: {exc}"
        ) from exc