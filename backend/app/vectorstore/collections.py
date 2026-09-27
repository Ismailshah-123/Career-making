"""
CareerGPT — Qdrant Collection Definitions
==========================================
PAGE SUMMARY:
  Central registry of ALL Qdrant collection schemas, payload index configs,
  HNSW parameters, optimizer settings, and shard/replication configs.
  This is the single source of truth for vector DB structure.

  COLLECTIONS:
    resume_embeddings   → one point per resume (master + tailored)
    job_embeddings      → one point per active job listing
    company_embeddings  → company culture + description vectors (future)
    post_embeddings     → LinkedIn post vectors for topic dedup (future)

  PAYLOAD SCHEMAS:
    Every searchable field has a typed index.
    This converts O(n) payload scans → O(log n) indexed lookups.
    Critical for performance at scale (100K+ jobs, 10K+ resumes).

  HNSW TUNING:
    m=16:              connections per node (higher = better recall, more memory)
    ef_construct=100:  construction quality (higher = better index, slower build)
    full_scan_threshold=10000: switch to brute-force below 10K points (faster)

  DISTANCE METRIC:
    COSINE used for all collections.
    Sentence-transformers output normalized vectors → cosine = dot product.
    This means scores are already in [0,1] range (no rescaling needed).

  USED BY:
    QdrantService.ensure_collections() → called at startup
    qdrant_manager.py → collection lifecycle management
    migration scripts → for schema updates

  PRODUCTION NOTES:
    - Collections are created idempotently (safe to call multiple times)
    - Payload index creation is also idempotent
    - Changing vector size requires recreating collection + re-embedding all points
    - Changing HNSW params triggers background re-indexing (no downtime)
    - Shard count should equal CPU count on Qdrant node for optimal perf
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ══════════════════════════════════════════════════════════════════════════════
# Collection Field Type Registry
# ══════════════════════════════════════════════════════════════════════════════

class FieldType:
    """Qdrant payload schema type constants."""
    KEYWORD  = "keyword"   # Exact match, enum values
    INTEGER  = "integer"   # Range queries, min/max
    FLOAT    = "float"     # Range queries, scores
    BOOL     = "bool"      # True/False filters
    TEXT     = "text"      # Full-text search
    GEO      = "geo"       # Geographic coordinates
    DATETIME = "datetime"  # Date range queries


# ══════════════════════════════════════════════════════════════════════════════
# Collection Schema Dataclasses
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class PayloadField:
    """Definition of a single indexed payload field."""
    name:        str
    field_type:  str
    required:    bool  = False
    description: str   = ""
    indexed:     bool  = True


@dataclass(frozen=True)
class HNSWConfig:
    """HNSW index configuration for a collection."""
    m:                     int   = 16
    ef_construct:          int   = 100
    full_scan_threshold:   int   = 10_000
    max_indexing_threads:  int   = 0     # 0 = use all available threads
    on_disk:               bool  = False
    payload_m:             int | None = None  # HNSW M for payload index


@dataclass(frozen=True)
class OptimizerConfig:
    """Qdrant optimizer settings for a collection."""
    deleted_threshold:    float = 0.2   # Trigger vacuum when 20% deleted
    vacuum_min_vector_number: int = 1000
    default_segment_number:   int = 0   # 0 = auto
    max_segment_size:     int | None = None
    memmap_threshold:     int = 50_000  # Switch to mmap above this count
    indexing_threshold:   int = 20_000  # Start HNSW indexing above this count
    flush_interval_sec:   int = 5
    max_optimization_threads: int = 1


@dataclass(frozen=True)
class CollectionConfig:
    """Complete collection definition including schema and tuning params."""
    name:              str
    description:       str
    vector_size:       int
    payload_fields:    tuple[PayloadField, ...]
    hnsw:              HNSWConfig          = field(default_factory=HNSWConfig)
    optimizer:         OptimizerConfig     = field(default_factory=OptimizerConfig)
    replication_factor: int                = 1
    write_consistency_factor: int          = 1
    shard_number:      int                 = 1
    on_disk_payload:   bool                = False

    def get_indexed_fields(self) -> list[PayloadField]:
        """Return only fields that need a payload index."""
        return [f for f in self.payload_fields if f.indexed]

    def to_schema_dict(self) -> dict[str, str]:
        """Return {field_name: field_type} for index creation."""
        return {f.name: f.field_type for f in self.get_indexed_fields()}


# ══════════════════════════════════════════════════════════════════════════════
# RESUME EMBEDDINGS COLLECTION
# ══════════════════════════════════════════════════════════════════════════════

RESUME_COLLECTION = CollectionConfig(
    name="resume_embeddings",
    description=(
        "Resume vectors for semantic matching with job descriptions. "
        "One point per resume (both master and tailored versions). "
        "Used by MatchingAgent to find best-fit candidates and "
        "by DiscoveryAgent to find best-fit jobs for a user."
    ),
    vector_size=768,  # sentence-transformers/all-MiniLM-L6-v2 output size
    payload_fields=(
        # ── Ownership ─────────────────────────────────────────────────────
        PayloadField("user_id",          FieldType.KEYWORD, required=True,  description="Owner user UUID"),
        PayloadField("resume_id",        FieldType.KEYWORD, required=True,  description="Resume UUID"),
        # ── Type ──────────────────────────────────────────────────────────
        PayloadField("is_master",        FieldType.BOOL,    required=True,  description="True = original uploaded CV"),
        PayloadField("is_deleted",       FieldType.BOOL,    required=True,  description="Soft delete flag"),
        # ── Content signals (for filtered search) ─────────────────────────
        PayloadField("skills",           FieldType.TEXT,    required=False, description="Space-separated skill list for text search"),
        PayloadField("experience_years", FieldType.FLOAT,   required=False, description="Years of professional experience"),
        PayloadField("ats_score",        FieldType.FLOAT,   required=False, description="ATS compatibility score 0-100"),
        PayloadField("education_level",  FieldType.KEYWORD, required=False, description="high_school|associate|bachelor|master|phd"),
        PayloadField("languages",        FieldType.KEYWORD, required=False, description="Primary language of resume"),
        # ── Metadata ──────────────────────────────────────────────────────
        PayloadField("created_at_ts",    FieldType.INTEGER, required=False, description="Unix timestamp for recency sorting", indexed=False),
    ),
    hnsw=HNSWConfig(
        m=16,
        ef_construct=128,
        full_scan_threshold=5_000,  # Lower threshold → use brute force for small user sets
    ),
    optimizer=OptimizerConfig(
        indexing_threshold=10_000,
        memmap_threshold=30_000,
    ),
)


# ══════════════════════════════════════════════════════════════════════════════
# JOB EMBEDDINGS COLLECTION
# ══════════════════════════════════════════════════════════════════════════════

JOB_COLLECTION = CollectionConfig(
    name="job_embeddings",
    description=(
        "Job listing vectors for semantic resume-to-job matching. "
        "One point per active job. Points are soft-deleted (is_active=false) "
        "when jobs expire or are removed. Hard-deleted during weekly cleanup. "
        "Used by DiscoveryAgent for 'find best jobs for my resume' feature."
    ),
    vector_size=768,
    payload_fields=(
        # ── Identity ──────────────────────────────────────────────────────
        PayloadField("job_id",           FieldType.KEYWORD, required=True,  description="Job UUID"),
        PayloadField("source",           FieldType.KEYWORD, required=True,  description="linkedin|indeed|remoteok|wellfound|rozee|company"),
        # ── Status ────────────────────────────────────────────────────────
        PayloadField("is_active",        FieldType.BOOL,    required=True,  description="Is job still open?"),
        PayloadField("is_deleted",       FieldType.BOOL,    required=True,  description="Soft delete flag"),
        # ── Job attributes (for filtered search) ──────────────────────────
        PayloadField("title",            FieldType.TEXT,    required=True,  description="Job title for text search"),
        PayloadField("company",          FieldType.KEYWORD, required=True,  description="Company name exact match"),
        PayloadField("is_remote",        FieldType.BOOL,    required=False, description="Remote work available?"),
        PayloadField("experience_level", FieldType.KEYWORD, required=False, description="entry|mid|senior|lead|executive"),
        PayloadField("employment_type",  FieldType.KEYWORD, required=False, description="full-time|contract|part-time|freelance"),
        PayloadField("salary_min",       FieldType.INTEGER, required=False, description="Minimum salary (USD/year)"),
        PayloadField("salary_max",       FieldType.INTEGER, required=False, description="Maximum salary (USD/year)"),
        PayloadField("location",         FieldType.TEXT,    required=False, description="Job location for text search"),
        PayloadField("skills_required",  FieldType.TEXT,    required=False, description="Space-separated required skills"),
        PayloadField("visa_sponsorship", FieldType.BOOL,    required=False, description="Visa sponsorship available?"),
        # ── Timestamps ────────────────────────────────────────────────────
        PayloadField("posted_at_ts",     FieldType.INTEGER, required=False, description="Job posting Unix timestamp", indexed=True),
    ),
    hnsw=HNSWConfig(
        m=16,
        ef_construct=100,
        full_scan_threshold=10_000,
        on_disk=False,  # Keep in RAM for fast matching
    ),
    optimizer=OptimizerConfig(
        deleted_threshold=0.2,  # Vacuum after 20% of points are soft-deleted
        indexing_threshold=20_000,
        memmap_threshold=100_000,
        flush_interval_sec=5,
    ),
    shard_number=2,  # 2 shards for better write throughput during bulk scraping
)


# ══════════════════════════════════════════════════════════════════════════════
# FUTURE COLLECTIONS (defined but not yet active)
# ══════════════════════════════════════════════════════════════════════════════

COMPANY_COLLECTION = CollectionConfig(
    name="company_embeddings",
    description=(
        "Company culture and description vectors. "
        "Used for company-culture matching in future 'find companies I'd thrive at' feature."
    ),
    vector_size=768,
    payload_fields=(
        PayloadField("company_id",    FieldType.KEYWORD, required=True),
        PayloadField("company_name",  FieldType.KEYWORD, required=True),
        PayloadField("industry",      FieldType.KEYWORD, required=False),
        PayloadField("stage",         FieldType.KEYWORD, required=False),
        PayloadField("remote_policy", FieldType.KEYWORD, required=False),
        PayloadField("is_deleted",    FieldType.BOOL,    required=True),
    ),
)

POST_COLLECTION = CollectionConfig(
    name="linkedin_post_embeddings",
    description=(
        "LinkedIn post content vectors. "
        "Used for topic deduplication — avoid posting too-similar content within 30 days."
    ),
    vector_size=768,
    payload_fields=(
        PayloadField("post_id",     FieldType.KEYWORD, required=True),
        PayloadField("user_id",     FieldType.KEYWORD, required=True),
        PayloadField("topic",       FieldType.TEXT,    required=True),
        PayloadField("tone",        FieldType.KEYWORD, required=False),
        PayloadField("posted_at_ts",FieldType.INTEGER, required=False),
        PayloadField("is_deleted",  FieldType.BOOL,    required=True),
    ),
)


# ══════════════════════════════════════════════════════════════════════════════
# Active Collections Registry
# ══════════════════════════════════════════════════════════════════════════════

ACTIVE_COLLECTIONS: dict[str, CollectionConfig] = {
    RESUME_COLLECTION.name: RESUME_COLLECTION,
    JOB_COLLECTION.name:    JOB_COLLECTION,
}

ALL_COLLECTIONS: dict[str, CollectionConfig] = {
    RESUME_COLLECTION.name:  RESUME_COLLECTION,
    JOB_COLLECTION.name:     JOB_COLLECTION,
    COMPANY_COLLECTION.name: COMPANY_COLLECTION,
    POST_COLLECTION.name:    POST_COLLECTION,
}


def get_collection(name: str) -> CollectionConfig:
    """Fetch collection config by name. Raises KeyError if not found."""
    if name not in ALL_COLLECTIONS:
        raise KeyError(
            f"Collection '{name}' not defined. "
            f"Available: {list(ALL_COLLECTIONS)}"
        )
    return ALL_COLLECTIONS[name]


def get_active_collections() -> list[CollectionConfig]:
    """Return only collections that should be created at startup."""
    return list(ACTIVE_COLLECTIONS.values())


# ══════════════════════════════════════════════════════════════════════════════
# Query Presets (named filter combinations for common search patterns)
# ══════════════════════════════════════════════════════════════════════════════

RESUME_QUERY_PRESETS: dict[str, dict[str, Any]] = {
    "master_only": {
        "description": "Only master (original) resumes",
        "filters": {"is_master": True, "is_deleted": False},
    },
    "active_all": {
        "description": "All non-deleted resumes (master + tailored)",
        "filters": {"is_deleted": False},
    },
    "senior_engineers": {
        "description": "Senior+ engineers (5+ years)",
        "filters": {"is_deleted": False, "is_master": True},
        "range_filters": {"experience_years": {"gte": 5.0}},
    },
}

JOB_QUERY_PRESETS: dict[str, dict[str, Any]] = {
    "active_remote": {
        "description": "Active remote jobs only",
        "filters": {"is_active": True, "is_deleted": False, "is_remote": True},
    },
    "active_all": {
        "description": "All active jobs regardless of location",
        "filters": {"is_active": True, "is_deleted": False},
    },
    "senior_remote": {
        "description": "Senior remote roles",
        "filters": {
            "is_active": True, "is_deleted": False,
            "is_remote": True, "experience_level": "senior",
        },
    },
    "pakistan_jobs": {
        "description": "Jobs sourced from Pakistan job boards",
        "filters": {"is_active": True, "is_deleted": False, "source": "rozee"},
    },
}


__all__ = [
    "FieldType",
    "PayloadField",
    "HNSWConfig",
    "OptimizerConfig",
    "CollectionConfig",
    "RESUME_COLLECTION",
    "JOB_COLLECTION",
    "COMPANY_COLLECTION",
    "POST_COLLECTION",
    "ACTIVE_COLLECTIONS",
    "ALL_COLLECTIONS",
    "RESUME_QUERY_PRESETS",
    "JOB_QUERY_PRESETS",
    "get_collection",
    "get_active_collections",
]