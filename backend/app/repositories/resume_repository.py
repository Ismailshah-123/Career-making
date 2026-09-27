"""
CareerGPT — Resume Repository
================================
PAGE SUMMARY:
  Data access layer for Resume model.
  Handles: master vs tailored resume queries, Qdrant point tracking,
  ownership validation helpers, bulk operations for maintenance tasks.

  KEY METHODS:
    get_master_resumes()          → user's current master CVs (sorted newest first)
    get_user_resumes()            → all resumes (master + tailored) paginated
    get_tailored_for_job()        → specific tailored resume for a job
    get_latest_master()           → single most-recent master resume
    count_by_user()               → total resume count per user
    update_qdrant_point()         → store Qdrant point ID after embedding
    get_resumes_without_embeddings() → backfill task: resumes not yet embedded
    get_resumes_by_ats_score()    → analytics: distribution of ATS scores
"""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import and_, desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.resume import Resume
from app.repositories.user_repository import BaseRepository


class ResumeRepository(BaseRepository):
    model = Resume

    async def get_master_resumes(
        self,
        user_id: uuid.UUID,
    ) -> list[Resume]:
        """Return all master resumes for a user, newest first."""
        q = (
            self._base_query()
            .where(Resume.user_id == user_id)
            .where(Resume.is_master.is_(True))
            .order_by(desc(Resume.created_at))
        )
        result = await self.session.execute(q)
        return list(result.scalars().all())

    async def get_latest_master(self, user_id: uuid.UUID) -> Resume | None:
        """Return the single most recent master resume."""
        q = (
            self._base_query()
            .where(Resume.user_id == user_id)
            .where(Resume.is_master.is_(True))
            .order_by(desc(Resume.created_at))
            .limit(1)
        )
        result = await self.session.execute(q)
        return result.scalar_one_or_none()

    async def get_user_resumes(
        self,
        user_id: uuid.UUID,
        *,
        master_only: bool = False,
        skip: int = 0,
        limit: int = 50,
    ) -> list[Resume]:
        """List all resumes for a user with optional master-only filter."""
        q = (
            self._base_query()
            .where(Resume.user_id == user_id)
            .order_by(desc(Resume.created_at))
            .offset(skip)
            .limit(limit)
        )
        if master_only:
            q = q.where(Resume.is_master.is_(True))
        result = await self.session.execute(q)
        return list(result.scalars().all())

    async def get_tailored_for_job(
        self,
        user_id: uuid.UUID,
        job_id: uuid.UUID,
    ) -> Resume | None:
        """Fetch the tailored resume for a specific user+job combination."""
        q = (
            self._base_query()
            .where(Resume.user_id == user_id)
            .where(Resume.tailored_for_job_id == job_id)
            .where(Resume.is_master.is_(False))
            .order_by(desc(Resume.created_at))
            .limit(1)
        )
        result = await self.session.execute(q)
        return result.scalar_one_or_none()

    async def count_by_user(self, user_id: uuid.UUID) -> int:
        """Total non-deleted resume count for a user."""
        q = (
            select(func.count())
            .select_from(Resume)
            .where(Resume.user_id == user_id)
            .where(Resume.is_deleted.is_(False))
        )
        result = await self.session.execute(q)
        return result.scalar_one() or 0

    async def update_qdrant_point(
        self,
        resume_id: uuid.UUID,
        point_id: str,
        model: str,
    ) -> None:
        """Store Qdrant point ID and embedding model after vector upsert."""
        await self.update(
            resume_id,
            qdrant_point_id=point_id,
            embedding_model=model,
        )

    async def get_resumes_without_embeddings(
        self,
        limit: int = 50,
    ) -> list[Resume]:
        """Fetch resumes not yet embedded in Qdrant (for backfill task)."""
        q = (
            self._base_query()
            .where(Resume.qdrant_point_id.is_(None))
            .where(Resume.raw_text.isnot(None))
            .where(Resume.is_master.is_(True))
            .order_by(desc(Resume.created_at))
            .limit(limit)
        )
        result = await self.session.execute(q)
        return list(result.scalars().all())

    async def get_ats_score_distribution(self) -> list[dict[str, Any]]:
        """
        Return ATS score distribution in buckets of 10.
        Used by analytics page: histogram of user resume quality.
        """
        from sqlalchemy import case, cast, Integer
        bucket_expr = (func.floor(Resume.ats_score / 10) * 10).cast(Integer)
        q = (
            select(
                bucket_expr.label("bucket"),
                func.count(Resume.id).label("count"),
            )
            .where(Resume.is_deleted.is_(False))
            .where(Resume.ats_score.isnot(None))
            .where(Resume.is_master.is_(True))
            .group_by(bucket_expr)
            .order_by(bucket_expr)
        )
        result = await self.session.execute(q)
        return [
            {"range": f"{row.bucket}-{row.bucket + 9}", "count": row.count}
            for row in result.all()
        ]

    async def deactivate_user_masters(self, user_id: uuid.UUID) -> None:
        """Mark all user's master resumes as non-master (before new upload)."""
        from sqlalchemy import update
        q = (
            update(Resume)
            .where(Resume.user_id == user_id)
            .where(Resume.is_master.is_(True))
            .where(Resume.is_deleted.is_(False))
            .values(is_master=False)
        )
        await self.session.execute(q)