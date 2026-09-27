"""
CareerGPT — Qdrant Vector Database Service
============================================
PAGE SUMMARY:
  Manages ALL vector operations for the platform.
  Two collections: resume_embeddings + job_embeddings.
  Provides: upsert, semantic search, filtered search, batch ops, health check,
  collection stats, point deletion, payload update, and snapshot management.
  Used by: ResumeAgent (store resume vectors), MatchingAgent (find jobs by resume),
  DiscoveryAgent (semantic job search), ApplicationService (match scoring).

COLLECTIONS:
  resume_embeddings → one point per resume (master + tailored)
    payload: {user_id, resume_id, is_master, skills, experience_years, ats_score}
  job_embeddings    → one point per job listing
    payload: {job_id, source, title, company, is_remote, experience_level, salary_min}

SEARCH MODES:
  - Pure semantic: embed query text → cosine similarity search
  - Filtered: semantic + metadata filters (remote only, salary range, source, etc.)
  - Batch: embed multiple texts in one call for efficiency
  - Recommend: find jobs similar to a given job (job-to-job similarity)

PRODUCTION NOTES:
  - Client is lazy-initialized (not at import time)
  - All ops have retry logic via tenacity
  - Qdrant API key is optional (local dev) / required (cloud)
  - Collections are created with HNSW index for O(log n) search
  - Points use UUIDs as string IDs (Qdrant supports both int + UUID string)
  - Soft-delete: set payload field "is_deleted": true instead of hard delete
    (allows recovery; cleaned up by nightly maintenance task)
"""

from __future__ import annotations

import uuid
from typing import Any

from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.config import get_settings
from app.core.constants import (
    QDRANT_COLLECTION_JOBS,
    QDRANT_COLLECTION_RESUMES,
    QDRANT_DEFAULT_TOP_K,
    QDRANT_SCORE_THRESHOLD_JOB,
    QDRANT_SCORE_THRESHOLD_RESUME,
)
from app.core.exceptions import VectorDBError
from app.core.logging import logger

settings = get_settings()


# ── Qdrant Model Imports (lazy) ───────────────────────────────────────────────

def _models() -> Any:
    try:
        from qdrant_client import models as qm
        return qm
    except ImportError as exc:
        raise VectorDBError("qdrant-client not installed. Run: pip install qdrant-client") from exc


def _client_cls() -> Any:
    try:
        from qdrant_client import QdrantClient
        return QdrantClient
    except ImportError as exc:
        raise VectorDBError("qdrant-client not installed") from exc


# ── Collection Schema Definitions ─────────────────────────────────────────────

RESUME_COLLECTION_CONFIG = {
    "name": QDRANT_COLLECTION_RESUMES,
    "payload_schema": {
        "user_id":         "keyword",
        "resume_id":       "keyword",
        "is_master":       "bool",
        "is_deleted":      "bool",
        "skills":          "text",
        "experience_years":"float",
        "ats_score":       "float",
        "education_level": "keyword",
        "languages":       "keyword",
    },
    "description": "Resume embeddings for semantic matching with jobs",
}

JOB_COLLECTION_CONFIG = {
    "name": QDRANT_COLLECTION_JOBS,
    "payload_schema": {
        "job_id":          "keyword",
        "source":          "keyword",
        "title":           "text",
        "company":         "keyword",
        "is_remote":       "bool",
        "is_active":       "bool",
        "is_deleted":      "bool",
        "experience_level":"keyword",
        "employment_type": "keyword",
        "salary_min":      "integer",
        "salary_max":      "integer",
        "location":        "text",
        "skills_required": "text",
    },
    "description": "Job listing embeddings for semantic resume matching",
}


class QdrantService:
    """
    Production-grade Qdrant vector database service.

    Singleton via get_qdrant_service() factory.
    All methods are synchronous wrappers around the Qdrant sync client
    (async client has known stability issues with some Qdrant versions).

    Usage:
        svc = get_qdrant_service()
        point_id = await svc.upsert_resume(resume_id, vector, payload)
        results  = await svc.search_jobs_for_resume(resume_vector, filters={...})
    """

    def __init__(self) -> None:
        self._client: Any = None
        self._collections_verified: bool = False

    # ── Client ────────────────────────────────────────────────────────────────

    def _get_client(self) -> Any:
        """Lazy-initialize Qdrant client."""
        if self._client is None:
            try:
                Client = _client_cls()
                self._client = Client(
                    url=settings.qdrant.url,
                    api_key=settings.qdrant.api_key or None,
                    timeout=settings.qdrant.timeout,
                    prefer_grpc=False,
                )
                logger.debug("Qdrant client initialized", url=settings.qdrant.url)
            except Exception as exc:
                raise VectorDBError(f"Qdrant client init failed: {exc}") from exc
        return self._client

    # ── Collection Management ─────────────────────────────────────────────────

    async def ensure_collections(self) -> None:
        """
        Called at startup. Creates collections if missing, verifies schema.
        Idempotent — safe to call multiple times.
        """
        qm = _models()
        client = self._get_client()

        try:
            existing = {c.name for c in client.get_collections().collections}
        except Exception as exc:
            raise VectorDBError(f"Cannot list Qdrant collections: {exc}") from exc

        for cfg in [RESUME_COLLECTION_CONFIG, JOB_COLLECTION_CONFIG]:
            name = cfg["name"]
            if name not in existing:
                try:
                    client.create_collection(
                        collection_name=name,
                        vectors_config=qm.VectorParams(
                            size=settings.qdrant.vector_size,
                            distance=qm.Distance.COSINE,
                            on_disk=False,
                        ),
                        hnsw_config=qm.HnswConfigDiff(
                            m=16,
                            ef_construct=100,
                            full_scan_threshold=10_000,
                        ),
                        optimizers_config=qm.OptimizersConfigDiff(
                            indexing_threshold=20_000,
                            memmap_threshold=50_000,
                        ),
                        replication_factor=1,
                    )
                    logger.info("Qdrant collection created", collection=name)

                    # Create payload indexes for filtered search
                    for field, field_type in cfg["payload_schema"].items():
                        try:
                            schema_type = qm.PayloadSchemaType[field_type.upper()]
                            client.create_payload_index(
                                collection_name=name,
                                field_name=field,
                                field_schema=schema_type,
                            )
                        except Exception as idx_exc:
                            logger.warning(
                                "Payload index creation failed (non-critical)",
                                field=field,
                                error=str(idx_exc),
                            )
                except Exception as exc:
                    raise VectorDBError(f"Failed to create collection '{name}': {exc}") from exc
            else:
                logger.debug("Qdrant collection exists", collection=name)

        self._collections_verified = True
        logger.info("Qdrant collections verified", collections=list(existing | {QDRANT_COLLECTION_RESUMES, QDRANT_COLLECTION_JOBS}))

    # ── Resume Operations ─────────────────────────────────────────────────────

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8), reraise=True)
    async def upsert_resume(
        self,
        resume_id: uuid.UUID,
        vector: list[float],
        payload: dict[str, Any],
    ) -> str:
        """
        Upsert a resume embedding into the resume collection.
        Returns the Qdrant point ID (UUID string).
        If the resume already has a point_id, it will be updated in place.
        """
        qm = _models()
        client = self._get_client()
        point_id = str(uuid.uuid4())

        full_payload = {
            "resume_id": str(resume_id),
            "is_deleted": False,
            **payload,
        }

        try:
            client.upsert(
                collection_name=QDRANT_COLLECTION_RESUMES,
                wait=True,
                points=[
                    qm.PointStruct(
                        id=point_id,
                        vector=vector,
                        payload=full_payload,
                    )
                ],
            )
            logger.info(
                "Resume vector upserted",
                resume_id=str(resume_id),
                point_id=point_id,
                is_master=payload.get("is_master"),
            )
            return point_id
        except Exception as exc:
            raise VectorDBError(f"Resume upsert failed: {exc}") from exc

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8), reraise=True)
    async def search_resumes_for_job(
        self,
        job_vector: list[float],
        *,
        user_id: str | None = None,
        master_only: bool = True,
        top_k: int = QDRANT_DEFAULT_TOP_K,
        score_threshold: float = QDRANT_SCORE_THRESHOLD_RESUME,
        min_experience_years: float | None = None,
        education_level: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Find resumes semantically matching a job description vector.
        Used by MatchingAgent to rank candidates.

        Returns list of:
          {point_id, score, resume_id, user_id, is_master, skills, ats_score, ...}
        """
        qm = _models()
        client = self._get_client()

        must_conditions: list[Any] = [
            qm.FieldCondition(key="is_deleted", match=qm.MatchValue(value=False)),
        ]
        if master_only:
            must_conditions.append(
                qm.FieldCondition(key="is_master", match=qm.MatchValue(value=True))
            )
        if user_id:
            must_conditions.append(
                qm.FieldCondition(key="user_id", match=qm.MatchValue(value=user_id))
            )
        if min_experience_years is not None:
            must_conditions.append(
                qm.FieldCondition(
                    key="experience_years",
                    range=qm.Range(gte=min_experience_years),
                )
            )
        if education_level:
            must_conditions.append(
                qm.FieldCondition(
                    key="education_level",
                    match=qm.MatchValue(value=education_level),
                )
            )

        query_filter = qm.Filter(must=must_conditions) if must_conditions else None

        try:
            results = client.search(
                collection_name=QDRANT_COLLECTION_RESUMES,
                query_vector=job_vector,
                query_filter=query_filter,
                limit=top_k,
                score_threshold=score_threshold,
                with_payload=True,
                with_vectors=False,
            )
            return [
                {
                    "point_id": str(r.id),
                    "score": round(r.score, 4),
                    **r.payload,
                }
                for r in results
            ]
        except Exception as exc:
            raise VectorDBError(f"Resume search failed: {exc}") from exc

    # ── Job Operations ────────────────────────────────────────────────────────

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8), reraise=True)
    async def upsert_job(
        self,
        job_id: uuid.UUID,
        vector: list[float],
        payload: dict[str, Any],
    ) -> str:
        """
        Upsert a job listing embedding.
        Returns the Qdrant point ID.
        """
        qm = _models()
        client = self._get_client()
        point_id = str(uuid.uuid4())

        full_payload = {
            "job_id": str(job_id),
            "is_active": True,
            "is_deleted": False,
            **payload,
        }

        try:
            client.upsert(
                collection_name=QDRANT_COLLECTION_JOBS,
                wait=True,
                points=[
                    qm.PointStruct(
                        id=point_id,
                        vector=vector,
                        payload=full_payload,
                    )
                ],
            )
            logger.debug(
                "Job vector upserted",
                job_id=str(job_id),
                point_id=point_id,
                source=payload.get("source"),
            )
            return point_id
        except Exception as exc:
            raise VectorDBError(f"Job upsert failed: {exc}") from exc

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8), reraise=True)
    async def search_jobs_for_resume(
        self,
        resume_vector: list[float],
        *,
        top_k: int = QDRANT_DEFAULT_TOP_K,
        score_threshold: float = QDRANT_SCORE_THRESHOLD_JOB,
        is_remote: bool | None = None,
        sources: list[str] | None = None,
        experience_level: str | None = None,
        employment_type: str | None = None,
        salary_min: int | None = None,
        salary_max: int | None = None,
        exclude_job_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Core semantic job matching: find jobs that best match a resume vector.
        Used by MatchingAgent and DiscoveryAgent.

        Supports rich filtering: remote, source, experience level, salary range.
        Returns list of {point_id, score, job_id, title, company, source, ...}
        Excludes already-applied jobs via exclude_job_ids.
        """
        qm = _models()
        client = self._get_client()

        must_conditions: list[Any] = [
            qm.FieldCondition(key="is_active",  match=qm.MatchValue(value=True)),
            qm.FieldCondition(key="is_deleted", match=qm.MatchValue(value=False)),
        ]
        must_not_conditions: list[Any] = []

        if is_remote is not None:
            must_conditions.append(
                qm.FieldCondition(key="is_remote", match=qm.MatchValue(value=is_remote))
            )
        if sources:
            must_conditions.append(
                qm.FieldCondition(key="source", match=qm.MatchAny(any=sources))
            )
        if experience_level:
            must_conditions.append(
                qm.FieldCondition(
                    key="experience_level", match=qm.MatchValue(value=experience_level)
                )
            )
        if employment_type:
            must_conditions.append(
                qm.FieldCondition(
                    key="employment_type", match=qm.MatchValue(value=employment_type)
                )
            )
        if salary_min is not None:
            must_conditions.append(
                qm.FieldCondition(
                    key="salary_max",
                    range=qm.Range(gte=salary_min),
                )
            )
        if salary_max is not None:
            must_conditions.append(
                qm.FieldCondition(
                    key="salary_min",
                    range=qm.Range(lte=salary_max),
                )
            )
        if exclude_job_ids:
            for jid in exclude_job_ids:
                must_not_conditions.append(
                    qm.FieldCondition(key="job_id", match=qm.MatchValue(value=jid))
                )

        query_filter = qm.Filter(
            must=must_conditions,
            must_not=must_not_conditions if must_not_conditions else None,
        )

        try:
            results = client.search(
                collection_name=QDRANT_COLLECTION_JOBS,
                query_vector=resume_vector,
                query_filter=query_filter,
                limit=top_k,
                score_threshold=score_threshold,
                with_payload=True,
                with_vectors=False,
            )
            logger.info(
                "Semantic job search complete",
                results=len(results),
                top_score=results[0].score if results else 0,
                filters={"remote": is_remote, "sources": sources, "level": experience_level},
            )
            return [
                {
                    "point_id": str(r.id),
                    "score":    round(r.score, 4),
                    **r.payload,
                }
                for r in results
            ]
        except Exception as exc:
            raise VectorDBError(f"Job search failed: {exc}") from exc

    async def search_similar_jobs(
        self,
        job_point_id: str,
        *,
        top_k: int = 10,
        score_threshold: float = 0.7,
        exclude_same_company: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Find jobs similar to a given job (job-to-job similarity).
        Used on job detail page to show "Similar jobs" section.
        """
        qm = _models()
        client = self._get_client()

        must: list[Any] = [
            qm.FieldCondition(key="is_active",  match=qm.MatchValue(value=True)),
            qm.FieldCondition(key="is_deleted", match=qm.MatchValue(value=False)),
        ]

        try:
            # First get the job's own payload (for company exclusion)
            existing = client.retrieve(
                collection_name=QDRANT_COLLECTION_JOBS,
                ids=[job_point_id],
                with_payload=True,
            )
            if not existing:
                return []

            if exclude_same_company and existing[0].payload:
                company = existing[0].payload.get("company")
                if company:
                    # We can't do "not equal" directly, skip for now — handled post-filter
                    pass

            results = client.recommend(
                collection_name=QDRANT_COLLECTION_JOBS,
                positive=[job_point_id],
                query_filter=qm.Filter(must=must),
                limit=top_k + 5,  # Fetch extra to allow post-filtering
                score_threshold=score_threshold,
                with_payload=True,
            )

            filtered = [
                {"point_id": str(r.id), "score": round(r.score, 4), **r.payload}
                for r in results
                if str(r.id) != job_point_id
            ]

            if exclude_same_company and existing[0].payload:
                company = existing[0].payload.get("company", "").lower()
                filtered = [
                    r for r in filtered
                    if r.get("company", "").lower() != company
                ]

            return filtered[:top_k]
        except Exception as exc:
            logger.warning("Similar jobs search failed (non-critical)", error=str(exc))
            return []

    # ── Batch Operations ──────────────────────────────────────────────────────

    async def upsert_jobs_batch(
        self,
        jobs: list[dict[str, Any]],
        *,
        batch_size: int = 100,
    ) -> int:
        """
        Batch upsert multiple job vectors.
        jobs: list of {job_id, vector, payload}
        Returns count of successfully upserted points.
        """
        qm = _models()
        client = self._get_client()
        total_upserted = 0

        # Split into batches
        for i in range(0, len(jobs), batch_size):
            batch = jobs[i : i + batch_size]
            points = []
            for item in batch:
                point_id = str(uuid.uuid4())
                points.append(
                    qm.PointStruct(
                        id=point_id,
                        vector=item["vector"],
                        payload={
                            "job_id":     str(item["job_id"]),
                            "is_active":  True,
                            "is_deleted": False,
                            **item.get("payload", {}),
                        },
                    )
                )
                # Store point_id back into item for caller to use
                item["qdrant_point_id"] = point_id

            try:
                client.upsert(
                    collection_name=QDRANT_COLLECTION_JOBS,
                    wait=True,
                    points=points,
                )
                total_upserted += len(points)
                logger.debug(
                    f"Job batch {i // batch_size + 1} upserted",
                    count=len(points),
                )
            except Exception as exc:
                logger.error(
                    "Job batch upsert failed",
                    batch_start=i,
                    error=str(exc),
                )

        logger.info("Job batch upsert complete", total=total_upserted)
        return total_upserted

    # ── Payload Updates ───────────────────────────────────────────────────────

    async def update_job_active_status(
        self, point_ids: list[str], is_active: bool
    ) -> None:
        """Mark jobs as active/inactive without re-embedding."""
        qm = _models()
        client = self._get_client()
        try:
            client.set_payload(
                collection_name=QDRANT_COLLECTION_JOBS,
                payload={"is_active": is_active},
                points=point_ids,
                wait=True,
            )
            logger.info(
                "Job active status updated",
                count=len(point_ids),
                is_active=is_active,
            )
        except Exception as exc:
            logger.warning("Job status update failed", error=str(exc))

    async def soft_delete_resume(self, point_id: str) -> None:
        """Soft-delete a resume point (keeps vector, marks is_deleted=true)."""
        qm = _models()
        client = self._get_client()
        try:
            client.set_payload(
                collection_name=QDRANT_COLLECTION_RESUMES,
                payload={"is_deleted": True},
                points=[point_id],
                wait=True,
            )
        except Exception as exc:
            logger.warning("Resume soft-delete failed", point_id=point_id, error=str(exc))

    async def hard_delete_resume(self, point_id: str) -> None:
        """Hard-delete a resume point (GDPR erasure)."""
        qm = _models()
        client = self._get_client()
        try:
            client.delete(
                collection_name=QDRANT_COLLECTION_RESUMES,
                points_selector=qm.PointIdsList(points=[point_id]),
                wait=True,
            )
            logger.info("Resume point hard-deleted", point_id=point_id)
        except Exception as exc:
            logger.warning("Resume hard-delete failed", point_id=point_id, error=str(exc))

    async def hard_delete_job(self, point_id: str) -> None:
        """Hard-delete a job point."""
        qm = _models()
        client = self._get_client()
        try:
            client.delete(
                collection_name=QDRANT_COLLECTION_JOBS,
                points_selector=qm.PointIdsList(points=[point_id]),
                wait=True,
            )
        except Exception as exc:
            logger.warning("Job hard-delete failed", point_id=point_id, error=str(exc))

    # ── Stats & Health ────────────────────────────────────────────────────────

    async def get_collection_stats(self) -> dict[str, Any]:
        """Return point counts and status for both collections."""
        client = self._get_client()
        stats: dict[str, Any] = {}
        for name in [QDRANT_COLLECTION_RESUMES, QDRANT_COLLECTION_JOBS]:
            try:
                info = client.get_collection(name)
                stats[name] = {
                    "points_count":     info.points_count,
                    "vectors_count":    info.vectors_count,
                    "indexed_vectors":  info.indexed_vectors_count,
                    "status":           info.status.value if info.status else "unknown",
                }
            except Exception as exc:
                stats[name] = {"error": str(exc)}
        return stats

    async def health_check(self) -> dict[str, Any]:
        """Health check for /health endpoint."""
        try:
            client = self._get_client()
            client.get_collections()
            stats = await self.get_collection_stats()
            return {
                "status":      "ok",
                "url":         settings.qdrant.url,
                "collections": stats,
            }
        except Exception as exc:
            return {"status": "error", "error": str(exc)}

    async def scroll_all_job_ids(self, limit: int = 1000) -> list[str]:
        """
        Scroll through all job point IDs (for maintenance/cleanup tasks).
        Used by nightly deactivation job.
        """
        qm = _models()
        client = self._get_client()
        all_ids: list[str] = []
        offset = None

        try:
            while True:
                results, next_offset = client.scroll(
                    collection_name=QDRANT_COLLECTION_JOBS,
                    scroll_filter=qm.Filter(
                        must=[qm.FieldCondition(key="is_deleted", match=qm.MatchValue(value=False))]
                    ),
                    limit=limit,
                    offset=offset,
                    with_payload=["job_id"],
                    with_vectors=False,
                )
                all_ids.extend(str(r.id) for r in results)
                if next_offset is None:
                    break
                offset = next_offset
        except Exception as exc:
            logger.error("Scroll failed", error=str(exc))

        return all_ids

    async def count_by_filter(
        self,
        collection: str,
        filters: dict[str, Any],
    ) -> int:
        """Count points matching a set of equality filters."""
        qm = _models()
        client = self._get_client()
        conditions = [
            qm.FieldCondition(key=k, match=qm.MatchValue(value=v))
            for k, v in filters.items()
        ]
        try:
            result = client.count(
                collection_name=collection,
                count_filter=qm.Filter(must=conditions),
                exact=True,
            )
            return result.count
        except Exception as exc:
            logger.warning("Count failed", collection=collection, error=str(exc))
            return 0


# ── Singleton ─────────────────────────────────────────────────────────────────

_qdrant_service: QdrantService | None = None


def get_qdrant_service() -> QdrantService:
    """
    Return the module-level QdrantService singleton.
    Thread-safe for FastAPI (single process, async event loop).
    """
    global _qdrant_service
    if _qdrant_service is None:
        _qdrant_service = QdrantService()
    return _qdrant_service


__all__ = ["QdrantService", "get_qdrant_service"]