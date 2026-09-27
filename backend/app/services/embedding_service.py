"""
CareerGPT — Embedding Service
================================
PAGE SUMMARY:
  Thin service-layer wrapper around app/embeddings/embedder.py.
  Provides a consistent interface for all vector embedding operations
  across services and agents. Adds: health checking, Redis caching
  coordination, batch size management, and usage metrics tracking.

  WHY A WRAPPER OVER DIRECT EMBEDDER IMPORT?
    Services and agents import get_embedding_service() — not get_embedder().
    This layer means we can swap embedding models, add caching tiers,
    or route to an external embedding API (OpenAI, Cohere) without
    changing any agent or service code.

  PUBLIC API:
    embed()             → embed one text string → list[float]
    embed_batch()       → embed multiple texts → list[list[float]]
    embed_resume()      → embed with resume-optimized preprocessing
    embed_job()         → embed with job-optimized preprocessing
    similarity()        → cosine similarity between two texts
    health_check()      → verify model is loaded and working
    get_vector_size()   → return embedding dimension (768 for MiniLM)
    get_cache_stats()   → Redis + in-memory cache statistics

  USED BY:
    ResumeService    → embed resume text after upload/tailor
    JobAgent         → embed job description after enrichment
    MatchingAgent    → embed query for semantic search
    QdrantService    → called before every upsert/search
    QdrantManager    → bulk re-indexing operations
"""

from __future__ import annotations

from typing import Any

from app.core.logging import logger


class EmbeddingService:
    """
    Service-layer wrapper around the Embedder singleton.
    Provides all embedding operations with health checking
    and metrics tracking built in.
    """

    def __init__(self) -> None:
        self._embedder: Any = None
        self._total_calls   = 0
        self._cache_hits    = 0

    def _get_embedder(self) -> Any:
        """Lazy-load the embedder singleton."""
        if self._embedder is None:
            from app.embeddings.embedder import get_embedder
            self._embedder = get_embedder()
        return self._embedder

    # ── Core Embedding Operations ─────────────────────────────────────────────

    async def embed(self, text: str) -> list[float]:
        """
        Embed a single text string.
        Returns a normalized float vector of dimension get_vector_size().
        Raises EmbeddingError if text is empty after preprocessing.
        """
        self._total_calls += 1
        embedder = self._get_embedder()
        return await embedder.embed(text)

    async def embed_batch(
        self,
        texts: list[str],
        *,
        show_progress: bool = False,
    ) -> list[list[float]]:
        """
        Embed multiple texts efficiently using batch inference.
        Cache-aware: cached texts skip model inference.
        Preserves input order in output.
        """
        self._total_calls += len(texts)
        embedder = self._get_embedder()
        return await embedder.embed_batch(texts, show_progress=show_progress)

    def embed_sync(self, text: str) -> list[float]:
        """
        Synchronous embedding for Celery task context.
        Checks cache first, falls back to model inference.
        """
        self._total_calls += 1
        embedder = self._get_embedder()
        return embedder.embed_sync(text)

    # ── Optimized Domain-Specific Embedders ──────────────────────────────────

    async def embed_resume(
        self,
        raw_text: str,
        *,
        skills: list[str] | None = None,
        summary: str | None = None,
        experience_titles: list[str] | None = None,
    ) -> list[float]:
        """
        Embed a resume with domain-specific preprocessing.
        Skills section is boosted 3x for better job matching recall.
        """
        from app.embeddings.embedder import build_resume_embed_text
        optimized_text = build_resume_embed_text(
            raw_text,
            skills=skills,
            summary=summary,
            experience_titles=experience_titles,
        )
        return await self.embed(optimized_text)

    async def embed_job(
        self,
        title: str,
        company: str,
        description: str | None,
        *,
        skills_required: list[str] | None = None,
        requirements: str | None = None,
    ) -> list[float]:
        """
        Embed a job listing with domain-specific preprocessing.
        Title is boosted 3x, required skills 2x for better resume matching.
        """
        from app.embeddings.embedder import build_job_embed_text
        optimized_text = build_job_embed_text(
            title,
            company,
            description,
            skills_required=skills_required,
            requirements=requirements,
        )
        return await self.embed(optimized_text)

    # ── Similarity ────────────────────────────────────────────────────────────

    async def similarity(self, text_a: str, text_b: str) -> float:
        """
        Compute semantic similarity between two texts.
        Returns float 0.0-1.0 (0=unrelated, 1=identical meaning).
        """
        embedder = self._get_embedder()
        return await embedder.similarity(text_a, text_b)

    def cosine_similarity(
        self,
        vec_a: list[float],
        vec_b: list[float],
    ) -> float:
        """Compute cosine similarity between two pre-computed vectors."""
        embedder = self._get_embedder()
        return embedder.cosine_similarity(vec_a, vec_b)

    # ── Health & Diagnostics ──────────────────────────────────────────────────

    async def health_check(self) -> dict[str, Any]:
        """
        Verify embedding model is loaded and producing valid output.
        Used by GET /api/v1/health endpoint.
        """
        import time
        t_start = time.monotonic()
        try:
            test_text   = "Senior Python Engineer with 5 years FastAPI experience"
            vector      = await self.embed(test_text)
            vector_size = len(vector)
            latency_ms  = round((time.monotonic() - t_start) * 1000, 1)

            if vector_size < 100:
                return {
                    "status":      "error",
                    "error":       f"Unexpected vector size: {vector_size}",
                    "latency_ms":  latency_ms,
                }

            return {
                "status":       "ok",
                "model":        self._get_embedder().model_name,
                "vector_size":  vector_size,
                "latency_ms":   latency_ms,
                "device":       "gpu" if self._get_embedder()._use_gpu else "cpu",
            }
        except Exception as exc:
            return {
                "status":    "error",
                "error":     str(exc)[:200],
                "latency_ms": round((time.monotonic() - t_start) * 1000, 1),
            }

    def get_vector_size(self) -> int:
        """Return the embedding vector dimension (768 for all-MiniLM-L6-v2)."""
        return self._get_embedder().get_vector_size()

    def get_cache_stats(self) -> dict[str, Any]:
        """Return cache hit statistics for monitoring."""
        embedder = self._get_embedder()
        cache_stats = embedder.get_cache_stats()
        return {
            **cache_stats,
            "total_service_calls": self._total_calls,
        }

    async def benchmark(self, n_texts: int = 50) -> dict[str, Any]:
        """Run embedding throughput benchmark. Used by admin panel."""
        embedder = self._get_embedder()
        return await embedder.benchmark(n_texts)


# ── Singleton ──────────────────────────────────────────────────────────────────

_embedding_service: EmbeddingService | None = None


def get_embedding_service() -> EmbeddingService:
    """Return the module-level EmbeddingService singleton."""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service


__all__ = ["EmbeddingService", "get_embedding_service"]