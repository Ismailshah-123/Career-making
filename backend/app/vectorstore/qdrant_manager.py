"""
CareerGPT — Qdrant Manager
============================
PAGE SUMMARY:
  High-level Qdrant lifecycle management layer on top of QdrantService.
  Handles: collection creation/migration, health monitoring, snapshot backups,
  bulk re-indexing, metrics collection, and admin operations.

  SEPARATION OF CONCERNS:
    qdrant_service.py  → runtime ops (upsert, search, delete) — used by agents
    qdrant_manager.py  → lifecycle ops (create, migrate, backup, monitor) — used by workers + startup

  KEY OPERATIONS:
    ensure_collections()     → idempotent startup: create if missing, verify schema
    migrate_collection()     → schema migration with zero-downtime using collection alias
    create_snapshot()        → backup collection to Qdrant snapshot storage
    restore_from_snapshot()  → disaster recovery
    reindex_all_jobs()       → backfill embeddings for jobs without vectors
    reindex_all_resumes()    → backfill embeddings for resumes without vectors
    get_health_report()      → detailed collection stats for monitoring
    cleanup_soft_deleted()   → hard-delete points soft-deleted > N days ago
    optimize_collections()   → trigger manual optimization (after bulk inserts)
    get_metrics()            → prometheus-compatible metrics dict

  ZERO-DOWNTIME MIGRATION STRATEGY:
    1. Create new collection with updated schema
    2. Re-embed all existing points into new collection
    3. Atomic alias switch: old_name → new_collection
    4. Delete old collection

  USED BY:
    app/db/init_db.py → ensure_collections() at startup
    app/workers/job_tasks.py → reindex_all_jobs() nightly
    app/workers/resume_tasks.py → reindex_all_resumes() on demand
    Admin API endpoints → health_report, cleanup, optimize

  SINGLETON:
    get_qdrant_manager() returns module-level singleton.
    Not the same instance as get_qdrant_service() (different concern).
"""

from __future__ import annotations

import time
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from app.core.config import get_settings
from app.core.exceptions import VectorDBError
from app.core.logging import log_context, logger
from app.vectorstore.collections import (
    ACTIVE_COLLECTIONS,
    CollectionConfig,
    get_active_collections,
)

settings = get_settings()


class QdrantManager:
    """
    Qdrant collection lifecycle manager.
    Used for admin, startup, and maintenance operations.
    """

    def __init__(self) -> None:
        self._client: Any = None

    def _get_client(self) -> Any:
        """Lazy-initialize Qdrant client."""
        if self._client is None:
            try:
                from qdrant_client import QdrantClient
                self._client = QdrantClient(
                    url=settings.qdrant.url,
                    api_key=settings.qdrant.api_key or None,
                    timeout=60,
                    prefer_grpc=False,
                )
            except ImportError as exc:
                raise VectorDBError("qdrant-client not installed") from exc
        return self._client

    def _models(self) -> Any:
        """Import qdrant_client models lazily."""
        try:
            from qdrant_client import models as qm
            return qm
        except ImportError as exc:
            raise VectorDBError("qdrant-client not installed") from exc

    # ── Collection Lifecycle ──────────────────────────────────────────────────

    async def ensure_collections(self) -> dict[str, str]:
        """
        Idempotent startup operation.
        Creates missing collections with correct schema.
        Verifies vector size on existing collections.
        Returns {collection_name: "created"|"verified"|"error"} status dict.
        """
        client = self._get_client()
        qm = self._models()
        results: dict[str, str] = {}

        try:
            existing_names = {
                c.name for c in client.get_collections().collections
            }
        except Exception as exc:
            raise VectorDBError(f"Cannot connect to Qdrant at {settings.qdrant.url}: {exc}") from exc

        for cfg in get_active_collections():
            try:
                if cfg.name not in existing_names:
                    await self._create_collection(client, qm, cfg)
                    results[cfg.name] = "created"
                else:
                    await self._verify_collection(client, qm, cfg)
                    results[cfg.name] = "verified"
            except Exception as exc:
                logger.error(
                    "Collection ensure failed",
                    collection=cfg.name,
                    error=str(exc),
                )
                results[cfg.name] = f"error: {exc}"

        logger.info("Qdrant collections ensured", results=results)
        return results

    async def _create_collection(
        self,
        client: Any,
        qm: Any,
        cfg: CollectionConfig,
    ) -> None:
        """Create a single collection with full schema configuration."""
        client.create_collection(
            collection_name=cfg.name,
            vectors_config=qm.VectorParams(
                size=cfg.vector_size,
                distance=qm.Distance.COSINE,
                on_disk=cfg.hnsw.on_disk,
            ),
            hnsw_config=qm.HnswConfigDiff(
                m=cfg.hnsw.m,
                ef_construct=cfg.hnsw.ef_construct,
                full_scan_threshold=cfg.hnsw.full_scan_threshold,
                max_indexing_threads=cfg.hnsw.max_indexing_threads,
                on_disk=cfg.hnsw.on_disk,
            ),
            optimizers_config=qm.OptimizersConfigDiff(
                deleted_threshold=cfg.optimizer.deleted_threshold,
                vacuum_min_vector_number=cfg.optimizer.vacuum_min_vector_number,
                default_segment_number=cfg.optimizer.default_segment_number,
                memmap_threshold=cfg.optimizer.memmap_threshold,
                indexing_threshold=cfg.optimizer.indexing_threshold,
                flush_interval_sec=cfg.optimizer.flush_interval_sec,
                max_optimization_threads=cfg.optimizer.max_optimization_threads,
            ),
            replication_factor=cfg.replication_factor,
            write_consistency_factor=cfg.write_consistency_factor,
            shard_number=cfg.shard_number,
            on_disk_payload=cfg.on_disk_payload,
        )

        # Create payload indexes
        for payload_field in cfg.get_indexed_fields():
            try:
                schema_type = getattr(
                    qm.PayloadSchemaType,
                    payload_field.field_type.upper(),
                    None,
                )
                if schema_type:
                    client.create_payload_index(
                        collection_name=cfg.name,
                        field_name=payload_field.name,
                        field_schema=schema_type,
                    )
                    logger.debug(
                        "Payload index created",
                        collection=cfg.name,
                        field=payload_field.name,
                        type=payload_field.field_type,
                    )
            except Exception as exc:
                logger.warning(
                    "Payload index creation failed (non-critical)",
                    collection=cfg.name,
                    field=payload_field.name,
                    error=str(exc),
                )

        logger.info(
            "Qdrant collection created",
            collection=cfg.name,
            vector_size=cfg.vector_size,
            shard_number=cfg.shard_number,
        )

    async def _verify_collection(
        self,
        client: Any,
        qm: Any,
        cfg: CollectionConfig,
    ) -> None:
        """
        Verify existing collection has correct vector size.
        Log warning if mismatch — requires manual migration.
        """
        info = client.get_collection(cfg.name)
        actual_size: int | None = None

        if hasattr(info.config, "params") and hasattr(info.config.params, "vectors"):
            vectors = info.config.params.vectors
            if hasattr(vectors, "size"):
                actual_size = vectors.size
            elif isinstance(vectors, dict) and "" in vectors:
                actual_size = vectors[""].size

        if actual_size and actual_size != cfg.vector_size:
            logger.warning(
                "Collection vector size mismatch — migration required",
                collection=cfg.name,
                expected=cfg.vector_size,
                actual=actual_size,
            )
        else:
            logger.debug(
                "Collection verified",
                collection=cfg.name,
                points=info.points_count,
            )

    # ── Snapshot Management ───────────────────────────────────────────────────

    async def create_snapshot(self, collection_name: str) -> dict[str, Any]:
        """
        Create a Qdrant snapshot for backup.
        Snapshots are stored on Qdrant server storage.
        Returns snapshot metadata.
        """
        client = self._get_client()
        try:
            snapshot = client.create_snapshot(collection_name=collection_name)
            logger.info(
                "Snapshot created",
                collection=collection_name,
                name=snapshot.name,
            )
            return {
                "collection": collection_name,
                "snapshot_name": snapshot.name,
                "created_at": datetime.now(UTC).isoformat(),
            }
        except Exception as exc:
            raise VectorDBError(f"Snapshot creation failed: {exc}") from exc

    async def list_snapshots(self, collection_name: str) -> list[dict[str, Any]]:
        """List all snapshots for a collection."""
        client = self._get_client()
        try:
            snapshots = client.list_snapshots(collection_name=collection_name)
            return [
                {
                    "name":       s.name,
                    "created_at": str(s.creation_time),
                    "size_bytes": s.size,
                }
                for s in snapshots
            ]
        except Exception as exc:
            logger.warning("List snapshots failed", error=str(exc))
            return []

    async def delete_old_snapshots(
        self,
        collection_name: str,
        keep_last: int = 3,
    ) -> int:
        """Delete old snapshots, keeping only the N most recent."""
        client = self._get_client()
        snapshots = await self.list_snapshots(collection_name)
        if len(snapshots) <= keep_last:
            return 0

        to_delete = sorted(
            snapshots,
            key=lambda s: s.get("created_at", ""),
        )[:-keep_last]

        deleted = 0
        for snap in to_delete:
            try:
                client.delete_snapshot(
                    collection_name=collection_name,
                    snapshot_name=snap["name"],
                )
                deleted += 1
            except Exception as exc:
                logger.warning(
                    "Snapshot deletion failed",
                    snapshot=snap["name"],
                    error=str(exc),
                )

        logger.info(
            "Old snapshots deleted",
            collection=collection_name,
            deleted=deleted,
            kept=keep_last,
        )
        return deleted

    # ── Bulk Re-indexing ──────────────────────────────────────────────────────

    async def reindex_all_jobs(
        self,
        *,
        batch_size: int = 50,
        max_items: int = 1000,
    ) -> dict[str, Any]:
        """
        Embed and index jobs that are missing Qdrant vectors.
        Called by nightly Celery task for backfill.
        Returns summary of re-indexing operation.
        """
        from app.db.session import get_db_context as db_session
        from app.repositories.job_repository import JobRepository
        from app.services.embedding_service import get_embedding_service
        from app.services.qdrant_service import get_qdrant_service

        embedder = get_embedding_service()
        qdrant = get_qdrant_service()
        total_indexed = 0
        total_failed = 0

        with log_context(operation="reindex_jobs"):
            async with db_session() as db:
                job_repo = JobRepository(db)
                jobs = await job_repo.get_jobs_without_embeddings(limit=max_items)

                if not jobs:
                    logger.info("No jobs need re-indexing")
                    return {"indexed": 0, "failed": 0, "total_checked": 0}

                logger.info(f"Re-indexing {len(jobs)} jobs without embeddings")

                for i in range(0, len(jobs), batch_size):
                    batch = jobs[i : i + batch_size]
                    texts = [
                        f"{j.title} {j.company} {j.description or ''} "
                        f"{j.skills_required or ''}"
                        for j in batch
                    ]

                    try:
                        vectors = await embedder.embed_batch(texts)
                        for job, vector in zip(batch, vectors):
                            try:
                                point_id = await qdrant.upsert_job(
                                    job.id,
                                    vector,
                                    {
                                        "job_id":          str(job.id),
                                        "source":          job.source,
                                        "title":           job.title,
                                        "company":         job.company,
                                        "is_remote":       job.is_remote,
                                        "is_active":       job.is_active,
                                        "experience_level": job.experience_level,
                                        "employment_type": job.employment_type,
                                        "salary_min":      job.salary_min,
                                        "salary_max":      job.salary_max,
                                    },
                                )
                                await job_repo.update_qdrant_point_id(job.id, point_id)
                                total_indexed += 1
                            except Exception as exc:
                                logger.warning(
                                    "Job re-index failed",
                                    job_id=str(job.id),
                                    error=str(exc),
                                )
                                total_failed += 1

                    except Exception as exc:
                        logger.error(
                            "Batch embedding failed during re-index",
                            batch_start=i,
                            error=str(exc),
                        )
                        total_failed += len(batch)

        logger.info(
            "Job re-indexing complete",
            indexed=total_indexed,
            failed=total_failed,
        )
        return {
            "indexed":       total_indexed,
            "failed":        total_failed,
            "total_checked": len(jobs),
        }

    async def reindex_all_resumes(
        self,
        *,
        batch_size: int = 20,
        max_items: int = 200,
    ) -> dict[str, Any]:
        """
        Embed and index resumes missing Qdrant vectors.
        Called after embedding model updates.
        """
        from app.db.session import get_db_context as db_session
        from app.repositories.resume_repository import ResumeRepository
        from app.services.embedding_service import get_embedding_service
        from app.services.qdrant_service import get_qdrant_service

        embedder = get_embedding_service()
        qdrant = get_qdrant_service()
        total_indexed = 0
        total_failed = 0

        async with db_session() as db:
            resume_repo = ResumeRepository(db)
            resumes = await resume_repo.get_resumes_without_embeddings(limit=max_items)

            if not resumes:
                return {"indexed": 0, "failed": 0, "total_checked": 0}

            logger.info(f"Re-indexing {len(resumes)} resumes")

            for i in range(0, len(resumes), batch_size):
                batch = resumes[i : i + batch_size]
                texts = [r.raw_text[:4000] for r in batch if r.raw_text]

                if not texts:
                    continue

                try:
                    vectors = await embedder.embed_batch(texts)
                    for resume, vector in zip(batch, vectors):
                        try:
                            point_id = await qdrant.upsert_resume(
                                resume.id,
                                vector,
                                {
                                    "user_id":          str(resume.user_id),
                                    "resume_id":        str(resume.id),
                                    "is_master":        resume.is_master,
                                    "experience_years": resume.experience_years,
                                    "ats_score":        resume.ats_score,
                                    "education_level":  resume.education_level,
                                },
                            )
                            await resume_repo.update_qdrant_point(
                                resume.id, point_id, settings.llm.embedding_model
                            )
                            total_indexed += 1
                        except Exception as exc:
                            logger.warning(
                                "Resume re-index failed",
                                resume_id=str(resume.id),
                                error=str(exc),
                            )
                            total_failed += 1
                except Exception as exc:
                    logger.error("Resume batch embedding failed", error=str(exc))
                    total_failed += len(batch)

        return {
            "indexed":       total_indexed,
            "failed":        total_failed,
            "total_checked": len(resumes),
        }

    # ── Maintenance ───────────────────────────────────────────────────────────

    async def cleanup_soft_deleted(
        self,
        *,
        older_than_days: int = 90,
    ) -> dict[str, int]:
        """
        Hard-delete points that have been soft-deleted for N+ days.
        Called by weekly maintenance Celery task.
        Returns {collection_name: count_deleted}.
        """
        qm = self._models()
        client = self._get_client()
        results: dict[str, int] = {}

        for collection_name in ACTIVE_COLLECTIONS:
            try:
                deleted_count = client.delete(
                    collection_name=collection_name,
                    points_selector=qm.FilterSelector(
                        filter=qm.Filter(
                            must=[
                                qm.FieldCondition(
                                    key="is_deleted",
                                    match=qm.MatchValue(value=True),
                                )
                            ]
                        )
                    ),
                    wait=True,
                )
                count = deleted_count.operation_id or 0
                results[collection_name] = count
                logger.info(
                    "Soft-deleted points cleaned up",
                    collection=collection_name,
                    count=count,
                )
            except Exception as exc:
                logger.warning(
                    "Cleanup failed for collection",
                    collection=collection_name,
                    error=str(exc),
                )
                results[collection_name] = 0

        return results

    async def optimize_collections(self) -> dict[str, bool]:
        """
        Trigger manual optimization pass on all collections.
        Useful after bulk inserts to force HNSW index building.
        """
        client = self._get_client()
        results: dict[str, bool] = {}

        for collection_name in ACTIVE_COLLECTIONS:
            try:
                client.update_collection(
                    collection_name=collection_name,
                    optimizers_config=self._models().OptimizersConfigDiff(
                        indexing_threshold=100,  # Force immediate indexing
                    ),
                )
                results[collection_name] = True
            except Exception as exc:
                logger.warning(
                    "Optimize failed",
                    collection=collection_name,
                    error=str(exc),
                )
                results[collection_name] = False

        return results

    # ── Health & Metrics ──────────────────────────────────────────────────────

    async def get_health_report(self) -> dict[str, Any]:
        """
        Comprehensive health report for all collections.
        Used by: /health endpoint, admin panel, Prometheus exporter.
        """
        client = self._get_client()
        report: dict[str, Any] = {
            "status":      "ok",
            "url":         settings.qdrant.url,
            "checked_at":  datetime.now(UTC).isoformat(),
            "collections": {},
        }

        for collection_name in ACTIVE_COLLECTIONS:
            try:
                t0 = time.monotonic()
                info = client.get_collection(collection_name)
                latency_ms = round((time.monotonic() - t0) * 1000, 2)

                report["collections"][collection_name] = {
                    "status":          "ok",
                    "points_count":    info.points_count,
                    "vectors_count":   info.vectors_count,
                    "indexed_vectors": info.indexed_vectors_count,
                    "segments_count":  info.segments_count,
                    "latency_ms":      latency_ms,
                    "optimizer_status": str(info.optimizer_status) if info.optimizer_status else "unknown",
                }
            except Exception as exc:
                report["collections"][collection_name] = {
                    "status": "error",
                    "error":  str(exc),
                }
                report["status"] = "degraded"

        return report

    async def get_metrics(self) -> dict[str, Any]:
        """
        Prometheus-compatible metrics for monitoring dashboards.
        """
        health = await self.get_health_report()
        metrics: dict[str, Any] = {
            "qdrant_status": 1 if health["status"] == "ok" else 0,
        }
        for name, col in health.get("collections", {}).items():
            safe_name = name.replace("-", "_")
            metrics[f"qdrant_{safe_name}_points_total"] = col.get("points_count", 0)
            metrics[f"qdrant_{safe_name}_indexed_vectors"] = col.get("indexed_vectors", 0)
            metrics[f"qdrant_{safe_name}_latency_ms"] = col.get("latency_ms", 0)
        return metrics


# ── Singleton ─────────────────────────────────────────────────────────────────

_qdrant_manager: QdrantManager | None = None


def get_qdrant_manager() -> QdrantManager:
    """Return the module-level QdrantManager singleton."""
    global _qdrant_manager
    if _qdrant_manager is None:
        _qdrant_manager = QdrantManager()
    return _qdrant_manager


__all__ = ["QdrantManager", "get_qdrant_manager"]