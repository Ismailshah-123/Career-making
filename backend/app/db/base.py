"""
app/db/base.py
==============
SQLAlchemy ORM foundation for the JobHunter AI platform.

Exports:
- Base              : Declarative base that all models inherit from
- TimestampMixin    : created_at / updated_at auto-managed columns
- UUIDMixin         : UUID primary key (uuid4, server-default)
- SoftDeleteMixin   : is_deleted / deleted_at non-destructive deletion
- AuditMixin        : created_by_id / updated_by_id FK columns
- BaseModel         : Combines UUID + Timestamp + SoftDelete (most models use this)

All imports of app models must go through app/db/base_import.py (not here)
to avoid circular import issues with Alembic.

Column conventions:
- UUID PKs use gen_random_uuid() on PostgreSQL (no Python-side generation needed)
- JSONB preferred over JSON for PostgreSQL-native indexing
- All text fields have explicit length limits to prevent abuse
- Indexes declared inline for clarity
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    Boolean,
    DateTime,
    MetaData,
    String,
    event,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    MappedColumn,
    declared_attr,
    mapped_column,
)

# ---------------------------------------------------------------------------
# Naming conventions for Alembic-generated constraint names
# This prevents unnamed constraints and makes migrations deterministic.
# ---------------------------------------------------------------------------

NAMING_CONVENTION: dict[str, str] = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

metadata = MetaData(naming_convention=NAMING_CONVENTION)


# ---------------------------------------------------------------------------
# Declarative Base
# ---------------------------------------------------------------------------

class Base(DeclarativeBase):
    """
    Root declarative base for all ORM models.

    All models should inherit from BaseModel (below) rather than directly
    from Base, unless they have a non-standard primary key strategy.
    """

    metadata = metadata

    # Automatically derive __tablename__ from class name
    @declared_attr.directive
    @classmethod
    def __tablename__(cls) -> str:
        # Convert CamelCase to snake_case and pluralise
        import re
        name = re.sub(r"(?<!^)(?=[A-Z])", "_", cls.__name__).lower()
        # Simple pluralisation — override in model if needed
        return name + "s" if not name.endswith("s") else name

    def to_dict(self, exclude: set[str] | None = None) -> dict[str, Any]:
        """Serialise model instance to a plain dict (non-recursive)."""
        exclude_cols = exclude or set()
        result: dict[str, Any] = {}
        for col in self.__table__.columns:
            if col.name in exclude_cols:
                continue
            value = getattr(self, col.name)
            if isinstance(value, uuid.UUID):
                result[col.name] = str(value)
            elif isinstance(value, datetime):
                result[col.name] = value.isoformat()
            else:
                result[col.name] = value
        return result

    def update_from_dict(self, data: dict[str, Any], exclude: set[str] | None = None) -> None:
        """Update model attributes from a dict, skipping excluded fields."""
        skip = (exclude or set()) | {"id", "created_at"}
        for key, value in data.items():
            if key not in skip and hasattr(self, key):
                setattr(self, key, value)

    def __repr__(self) -> str:
        pk = getattr(self, "id", None)
        return f"<{self.__class__.__name__} id={pk}>"


# ---------------------------------------------------------------------------
# UUID Primary Key Mixin
# ---------------------------------------------------------------------------

class UUIDMixin:
    """
    Provides a UUID v4 primary key column.

    Uses PostgreSQL's gen_random_uuid() as the server default so the DB
    assigns IDs, reducing round-trips for bulk inserts.
    Python also generates IDs via default= for in-memory access before flush.
    """

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=text("gen_random_uuid()"),
        nullable=False,
        index=True,
        comment="Primary key — UUID v4.",
    )


# ---------------------------------------------------------------------------
# Timestamp Mixin
# ---------------------------------------------------------------------------

class TimestampMixin:
    """
    Automatic created_at / updated_at management.

    created_at: Set on INSERT, never updated.
    updated_at: Updated on every UPDATE via SQLAlchemy event + onupdate.
    Both store timezone-aware UTC datetimes.
    """

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        server_default=func.now(),
        comment="UTC timestamp when this record was created.",
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        server_default=func.now(),
        onupdate=lambda: datetime.now(timezone.utc),
        comment="UTC timestamp when this record was last modified.",
    )


# ---------------------------------------------------------------------------
# Soft Delete Mixin
# ---------------------------------------------------------------------------

class SoftDeleteMixin:
    """
    Non-destructive deletion support.

    is_deleted: Boolean flag — True means the record is logically deleted.
    deleted_at: UTC timestamp of deletion (NULL if not deleted).

    Repositories should filter WHERE is_deleted = FALSE by default.
    Physical deletion (VACUUM) should be handled by a separate cleanup job.
    """

    is_deleted: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default="false",
        index=True,
        comment="Soft-delete flag. Logically deleted records are excluded from queries.",
    )

    deleted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        default=None,
        comment="UTC timestamp of logical deletion.",
    )

    def soft_delete(self) -> None:
        """Mark this record as logically deleted."""
        self.is_deleted = True
        self.deleted_at = datetime.now(timezone.utc)

    def restore(self) -> None:
        """Restore a soft-deleted record."""
        self.is_deleted = False
        self.deleted_at = None


# ---------------------------------------------------------------------------
# Audit Mixin (who created / updated)
# ---------------------------------------------------------------------------

class AuditMixin:
    """
    Lightweight audit trail — records which user created or last modified a row.

    References user IDs as strings (not FK constraints) to avoid cascade
    complexity and to support service/system actors that have no user row.
    """

    created_by_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
        default=None,
        comment="User UUID who created this record (NULL = system/background task).",
    )

    updated_by_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
        default=None,
        comment="User UUID who last modified this record.",
    )


# ---------------------------------------------------------------------------
# Composite BaseModel (used by almost all domain models)
# ---------------------------------------------------------------------------

class BaseModel(UUIDMixin, TimestampMixin, SoftDeleteMixin, Base):
    """
    Standard model base combining UUID PK + timestamps + soft delete.

    Inherit from this instead of Base directly unless you have special needs.

    Example:
        class Job(BaseModel):
            __tablename__ = "jobs"
            title: Mapped[str] = mapped_column(String(256))
    """

    __abstract__ = True


# ---------------------------------------------------------------------------
# Slim BaseModel (for lookup / reference tables — no soft delete)
# ---------------------------------------------------------------------------

class SlimBaseModel(UUIDMixin, TimestampMixin, Base):
    """
    Minimal model base: UUID PK + timestamps only.

    Use for reference/lookup tables that are never soft-deleted
    (e.g. company tags, skill categories).
    """

    __abstract__ = True


# ---------------------------------------------------------------------------
# SQLAlchemy Event Listeners
# ---------------------------------------------------------------------------

@event.listens_for(TimestampMixin, "before_update", propagate=True)
def set_updated_at(mapper: Any, connection: Any, target: TimestampMixin) -> None:
    """
    Ensure updated_at is refreshed on every UPDATE even without ORM-tracked
    attribute changes (e.g. when using bulk_update_mappings).
    """
    target.updated_at = datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Type aliases for common column patterns
# ---------------------------------------------------------------------------

def uuid_column(**kwargs: Any) -> MappedColumn:
    """Helper to declare a UUID FK column cleanly."""
    return mapped_column(UUID(as_uuid=True), **kwargs)


def short_string(length: int = 128, **kwargs: Any) -> MappedColumn:
    """Convenience for short VARCHAR columns."""
    return mapped_column(String(length), **kwargs)


def long_string(**kwargs: Any) -> MappedColumn:
    """Convenience for text-length VARCHAR(1024) columns."""
    return mapped_column(String(1024), **kwargs)