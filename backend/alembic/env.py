"""
Alembic environment script.

Wires Alembic to the app's own settings (app.core.config) and full model
metadata (app.db.base_import) instead of a hardcoded sqlalchemy.url in
alembic.ini, so migrations always target whatever DB the app itself is
configured for -- one source of truth, no drift between .env and alembic.ini.

Runs migrations synchronously (via psycopg2) even though the app serves
requests through an async engine (asyncpg) -- this is standard practice:
migrations are a one-off operation, and sync keeps this script simple.
"""

from __future__ import annotations

import sys
from logging.config import fileConfig
from pathlib import Path

from sqlalchemy import engine_from_config, pool

from alembic import context

# Make `app` importable when Alembic is invoked from the backend/ directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import get_settings  # noqa: E402
from app.db.base_import import Base  # noqa: E402  (imports every model)

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Full schema -- populated because app.db.base_import imports every model.
target_metadata = Base.metadata

# Override whatever's in alembic.ini with the app's own settings.
settings = get_settings()
config.set_main_option("sqlalchemy.url", settings.DATABASE_URL_SYNC)


def run_migrations_offline() -> None:
    """Emit SQL to stdout without a live DB connection (`alembic upgrade head --sql`)."""
    context.configure(
        url=settings.DATABASE_URL_SYNC,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations against a live DB connection (the normal case)."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            compare_server_default=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
