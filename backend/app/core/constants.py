"""
CareerGPT — Unified Constants
================================
All platform-wide enums, limits, and identifiers.
Includes every value imported by agents, services, workers, and main.py.
Zero dependencies on other app modules.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

# ══════════════════════════════════════════════════════════════════════════════
# API
# ══════════════════════════════════════════════════════════════════════════════

API_V1_PREFIX   = "/api/v1"
API_TITLE       = "CareerGPT API"
API_VERSION     = "1.0.0"
DOCS_URL        = "/docs"
REDOC_URL       = "/redoc"
OPENAPI_URL     = "/openapi.json"

# ══════════════════════════════════════════════════════════════════════════════
# Authentication
# ══════════════════════════════════════════════════════════════════════════════

ACCESS_TOKEN_EXPIRE_MINUTES    = 1440   # 24h
REFRESH_TOKEN_EXPIRE_DAYS      = 30
ALGORITHM                      = "HS256"
PASSWORD_RESET_TOKEN_EXPIRE_MINUTES = 30
EMAIL_VERIFICATION_TOKEN_EXPIRE_HOURS = 48
MIN_PASSWORD_LENGTH            = 8
MAX_PASSWORD_LENGTH            = 128
PASSWORD_MIN_LENGTH            = 8   # alias
PASSWORD_MAX_LENGTH            = 128  # alias
BCRYPT_ROUNDS                  = 12
MAX_LOGIN_ATTEMPTS             = 5
LOCKOUT_DURATION_MINUTES       = 15
API_KEY_PREFIX                 = "cgpt_"
API_KEY_LENGTH                 = 48

# ══════════════════════════════════════════════════════════════════════════════
# User fields
# ══════════════════════════════════════════════════════════════════════════════

EMAIL_MAX_LENGTH    = 254
FULL_NAME_MAX_LENGTH = 200
FULL_NAME_MIN_LENGTH = 2


class UserRole(str, Enum):
    USER       = "user"
    ADMIN      = "admin"
    SUPERUSER  = "superuser"

# ══════════════════════════════════════════════════════════════════════════════
# Pagination
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT_PAGE      = 1
DEFAULT_PAGE_SIZE = 20
MAX_PAGE_SIZE     = 100
MIN_PAGE_SIZE     = 1

# ══════════════════════════════════════════════════════════════════════════════
# File Uploads
# ══════════════════════════════════════════════════════════════════════════════

ALLOWED_RESUME_EXTENSIONS = {".pdf", ".docx", ".doc", ".txt", ".odt"}
ALLOWED_AVATAR_EXTENSIONS  = {".jpg", ".jpeg", ".png", ".webp"}
ALLOWED_RESUME_MIMES = frozenset({
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/msword",
    "text/plain",
})

MAX_RESUME_SIZE_MB    = 10
MAX_AVATAR_SIZE_MB    = 5
MAX_RESUME_SIZE_BYTES = MAX_RESUME_SIZE_MB * 1024 * 1024
MAX_AVATAR_SIZE_BYTES = MAX_AVATAR_SIZE_MB * 1024 * 1024
RESUME_MAX_SIZE_MB    = MAX_RESUME_SIZE_MB  # alias

RESUME_EMBED_CHAR_LIMIT = 8000
RESUME_EXTRACT_CHAR_LIMIT = 12000   # cap applied when pulling raw text out of an uploaded file

# Local disk path (relative to UPLOAD_DIR) where original resume uploads are kept
# Local disk base path where uploaded files are written by
# app/utils/file_utils.py's _save_local() (matches Settings.UPLOAD_DIR's
# default — kept as its own constant since constants.py has zero
# dependencies on app.core.config).
RESUME_UPLOAD_PATH = "./uploads"

# Upload subdirectory names
UPLOAD_DIR_TAILORED    = "tailored"
UPLOAD_DIR_SCREENSHOTS = "screenshots"
UPLOAD_DIR_RESUMES     = "resumes"
UPLOAD_DIR_AVATARS     = "avatars"
UPLOAD_DIR_LINKEDIN    = "linkedin_images"

# ══════════════════════════════════════════════════════════════════════════════
# Embedding & Vector Store
# ══════════════════════════════════════════════════════════════════════════════

EMBEDDING_DIMENSION     = 768    # all-MiniLM-L6-v2 output
EMBEDDING_MODEL         = "all-MiniLM-L6-v2"
EMBEDDING_BATCH_SIZE    = 32
MAX_TOKENS_PER_CHUNK    = 512
CHUNK_OVERLAP           = 64

QDRANT_RESUME_COLLECTION     = "resume_embeddings"
QDRANT_JOB_COLLECTION        = "job_embeddings"
QDRANT_COVER_LETTER_COLLECTION = "cover_letter_embeddings"
QDRANT_LINKEDIN_COLLECTION   = "linkedin_post_embeddings"

# Aliases used by qdrant_service.py (kept in sync with the names above)
QDRANT_COLLECTION_RESUMES = QDRANT_RESUME_COLLECTION
QDRANT_COLLECTION_JOBS    = QDRANT_JOB_COLLECTION

SIMILARITY_SCORE_THRESHOLD = 0.72
TOP_K_MATCHES              = 20

QDRANT_DEFAULT_TOP_K          = TOP_K_MATCHES
QDRANT_SCORE_THRESHOLD_JOB    = SIMILARITY_SCORE_THRESHOLD
QDRANT_SCORE_THRESHOLD_RESUME = SIMILARITY_SCORE_THRESHOLD

# ══════════════════════════════════════════════════════════════════════════════
# HTTP Timeouts
# ══════════════════════════════════════════════════════════════════════════════

HTTP_TIMEOUT_SCRAPER  = 30.0
HTTP_TIMEOUT_LLM      = 60.0
HTTP_TIMEOUT_LINKEDIN = 30.0

# ══════════════════════════════════════════════════════════════════════════════
# Rate Limits
# ══════════════════════════════════════════════════════════════════════════════

RATE_LIMIT_AUTH             = "10/minute"
RATE_LIMIT_API_DEFAULT      = "100/minute"
RATE_LIMIT_AI_ENDPOINT      = "30/minute"
RATE_LIMIT_UPLOAD           = "20/hour"
RATE_LIMIT_SCRAPER          = "60/hour"
RATE_LIMIT_LINKEDIN_POST    = "5/hour"
RATE_LIMIT_REQUESTS_PER_MINUTE = 100

# ══════════════════════════════════════════════════════════════════════════════
# Celery Queue & Task Names
# ══════════════════════════════════════════════════════════════════════════════

QUEUE_DEFAULT       = "default"
QUEUE_AI            = "ai_processing"
QUEUE_SCRAPING      = "scraping"
QUEUE_APPLICATIONS  = "applications"
QUEUE_NOTIFICATIONS = "notifications"
QUEUE_LINKEDIN      = "linkedin"

TASK_PROCESS_RESUME        = "workers.resume_tasks.process_resume"
TASK_DISCOVER_JOBS         = "workers.job_tasks.discover_jobs"
TASK_MATCH_JOBS            = "workers.job_tasks.match_jobs_to_resume"
TASK_GENERATE_COVER_LETTER = "workers.resume_tasks.generate_cover_letter"
TASK_SUBMIT_APPLICATION    = "workers.job_tasks.submit_application"
TASK_POST_LINKEDIN         = "workers.linkedin_tasks.post_to_linkedin"
TASK_SEND_NOTIFICATION     = "workers.notification_tasks.send_notification"
TASK_FOLLOW_UP             = "workers.job_tasks.send_followup"
TASK_EMBED_RESUME          = "workers.resume_tasks.embed_resume"
TASK_EMBED_JOB             = "workers.job_tasks.embed_job"

# ══════════════════════════════════════════════════════════════════════════════
# Agent Names
# ══════════════════════════════════════════════════════════════════════════════

AGENT_DISCOVERY    = "discovery_agent"
AGENT_RESUME       = "resume_agent"
AGENT_MATCHING     = "matching_agent"
AGENT_COVER_LETTER = "cover_letter_agent"
AGENT_OUTREACH     = "outreach_agent"
AGENT_LINKEDIN     = "linkedin_agent"
AGENT_APPLICATION  = "application_agent"
AGENT_FOLLOWUP     = "followup_agent"

ALL_AGENTS = [
    AGENT_DISCOVERY, AGENT_RESUME, AGENT_MATCHING, AGENT_COVER_LETTER,
    AGENT_OUTREACH, AGENT_LINKEDIN, AGENT_APPLICATION, AGENT_FOLLOWUP,
]


class AgentType(str, Enum):
    DISCOVERY    = "discovery"
    RESUME       = "resume"
    MATCHING     = "matching"
    COVER_LETTER = "cover_letter"
    OUTREACH     = "outreach"
    LINKEDIN     = "linkedin"
    APPLICATION  = "application"
    FOLLOWUP     = "followup"

# ══════════════════════════════════════════════════════════════════════════════
# Workflow Node Names (LangGraph)
# ══════════════════════════════════════════════════════════════════════════════
# Shared node-name identifiers used by app/workflows/*.py when building each
# StateGraph. Kept centralized so every workflow refers to the same string
# for a given step (node registration, @node() telemetry, conditional edges).

NODE_DISCOVERY      = "discovery"
NODE_FILTER         = "filter"
NODE_MATCH          = "match"
NODE_TAILOR_RESUME  = "tailor_resume"
NODE_GENERATE_COVER = "generate_cover_letter"
NODE_APPLY          = "apply"
NODE_NOTIFY         = "notify"
NODE_ERROR          = "handle_error"
NODE_END            = "__end__"   # mirrors langgraph.graph.END's sentinel value

# ══════════════════════════════════════════════════════════════════════════════
# Job Source / Board
# ══════════════════════════════════════════════════════════════════════════════

class JobSource(str, Enum):
    LINKEDIN   = "linkedin"
    INDEED     = "indeed"
    REMOTEOK   = "remoteok"
    WELLFOUND  = "wellfound"
    ROZEE      = "rozee"
    GREENHOUSE = "greenhouse"
    LEVER      = "lever"
    WORKDAY    = "workday"
    ASHBY      = "ashby"
    COMPANY    = "company"
    MANUAL     = "manual"
    OTHER      = "other"


# Alias used by some agents
JobBoard = JobSource

SUPPORTED_JOB_BOARDS = [
    JobSource.LINKEDIN,
    JobSource.INDEED,
    JobSource.REMOTEOK,
    JobSource.WELLFOUND,
    JobSource.ROZEE,
]

# ══════════════════════════════════════════════════════════════════════════════
# ATS Platform Detection
# ══════════════════════════════════════════════════════════════════════════════

class ATSPlatform(str, Enum):
    GREENHOUSE = "greenhouse"
    LEVER      = "lever"
    WORKDAY    = "workday"
    ASHBY      = "ashby"
    LINKEDIN   = "linkedin"
    INDEED     = "indeed"
    GENERIC    = "generic"


# Domain → ATS platform mapping for URL-based detection
ATS_DOMAIN_MAP: dict[str, ATSPlatform] = {
    "greenhouse.io":        ATSPlatform.GREENHOUSE,
    "boards.greenhouse.io": ATSPlatform.GREENHOUSE,
    "lever.co":             ATSPlatform.LEVER,
    "jobs.lever.co":        ATSPlatform.LEVER,
    "myworkdayjobs.com":    ATSPlatform.WORKDAY,
    "workday.com":          ATSPlatform.WORKDAY,
    "ashbyhq.com":          ATSPlatform.ASHBY,
    "jobs.ashbyhq.com":     ATSPlatform.ASHBY,
    "linkedin.com":         ATSPlatform.LINKEDIN,
    "indeed.com":           ATSPlatform.INDEED,
    "apply.indeed.com":     ATSPlatform.INDEED,
}

# ══════════════════════════════════════════════════════════════════════════════
# Application Status
# ══════════════════════════════════════════════════════════════════════════════

class ApplicationStatus(str, Enum):
    # Pipeline stages (before the application is actually submitted)
    DISCOVERED             = "discovered"
    QUEUED                 = "queued"
    RESUME_TAILORED        = "resume_tailored"
    COVER_LETTER_GENERATED = "cover_letter_generated"
    APPLYING                = "applying"
    # Submitted / legacy-simple stage (kept for services using the short flow)
    PENDING     = "pending"
    APPLIED     = "applied"
    # Post-submission engagement
    VIEWED               = "viewed"
    ACKNOWLEDGED          = "acknowledged"
    SCREENING             = "screening"
    INTERVIEW             = "interview"
    INTERVIEW_SCHEDULED   = "interview_scheduled"
    INTERVIEWED           = "interviewed"
    OFFER                 = "offer"
    OFFER_RECEIVED        = "offer_received"
    # Terminal
    REJECTED    = "rejected"
    WITHDRAWN   = "withdrawn"
    FAILED      = "failed"

TERMINAL_STATUSES = {
    ApplicationStatus.REJECTED,
    ApplicationStatus.WITHDRAWN,
    ApplicationStatus.OFFER,
    ApplicationStatus.OFFER_RECEIVED,
    ApplicationStatus.FAILED,
}

# Allowed forward transitions for each status. Used to validate manual
# status updates (e.g. PATCH /applications/{id}/status) and by the
# application/apply workflows to decide what can happen next.
APPLICATION_STATUS_TRANSITIONS: dict[ApplicationStatus, list[ApplicationStatus]] = {
    ApplicationStatus.DISCOVERED:             [ApplicationStatus.QUEUED, ApplicationStatus.WITHDRAWN, ApplicationStatus.FAILED],
    ApplicationStatus.QUEUED:                 [ApplicationStatus.RESUME_TAILORED, ApplicationStatus.APPLYING, ApplicationStatus.WITHDRAWN, ApplicationStatus.FAILED],
    ApplicationStatus.RESUME_TAILORED:        [ApplicationStatus.COVER_LETTER_GENERATED, ApplicationStatus.APPLYING, ApplicationStatus.WITHDRAWN, ApplicationStatus.FAILED],
    ApplicationStatus.COVER_LETTER_GENERATED: [ApplicationStatus.APPLYING, ApplicationStatus.WITHDRAWN, ApplicationStatus.FAILED],
    ApplicationStatus.APPLYING:               [ApplicationStatus.APPLIED, ApplicationStatus.QUEUED, ApplicationStatus.WITHDRAWN, ApplicationStatus.FAILED],
    ApplicationStatus.PENDING:                [ApplicationStatus.APPLIED, ApplicationStatus.WITHDRAWN, ApplicationStatus.FAILED],
    ApplicationStatus.APPLIED:                [ApplicationStatus.VIEWED, ApplicationStatus.ACKNOWLEDGED, ApplicationStatus.SCREENING, ApplicationStatus.INTERVIEW_SCHEDULED, ApplicationStatus.REJECTED, ApplicationStatus.WITHDRAWN],
    ApplicationStatus.VIEWED:                 [ApplicationStatus.ACKNOWLEDGED, ApplicationStatus.SCREENING, ApplicationStatus.INTERVIEW_SCHEDULED, ApplicationStatus.REJECTED, ApplicationStatus.WITHDRAWN],
    ApplicationStatus.ACKNOWLEDGED:           [ApplicationStatus.SCREENING, ApplicationStatus.INTERVIEW_SCHEDULED, ApplicationStatus.REJECTED, ApplicationStatus.WITHDRAWN],
    ApplicationStatus.SCREENING:              [ApplicationStatus.INTERVIEW_SCHEDULED, ApplicationStatus.INTERVIEW, ApplicationStatus.REJECTED, ApplicationStatus.WITHDRAWN],
    ApplicationStatus.INTERVIEW_SCHEDULED:    [ApplicationStatus.INTERVIEWED, ApplicationStatus.INTERVIEW, ApplicationStatus.REJECTED, ApplicationStatus.WITHDRAWN],
    ApplicationStatus.INTERVIEW:              [ApplicationStatus.INTERVIEWED, ApplicationStatus.OFFER, ApplicationStatus.OFFER_RECEIVED, ApplicationStatus.REJECTED, ApplicationStatus.WITHDRAWN],
    ApplicationStatus.INTERVIEWED:            [ApplicationStatus.INTERVIEW_SCHEDULED, ApplicationStatus.OFFER, ApplicationStatus.OFFER_RECEIVED, ApplicationStatus.REJECTED, ApplicationStatus.WITHDRAWN],
    ApplicationStatus.OFFER:                  [ApplicationStatus.WITHDRAWN],
    ApplicationStatus.OFFER_RECEIVED:         [ApplicationStatus.WITHDRAWN],
    ApplicationStatus.REJECTED:               [],
    ApplicationStatus.WITHDRAWN:              [],
    ApplicationStatus.FAILED:                 [ApplicationStatus.QUEUED],
}

# ══════════════════════════════════════════════════════════════════════════════
# Job Type / Work Mode / Experience Level
# ══════════════════════════════════════════════════════════════════════════════

class JobType(str, Enum):
    FULL_TIME  = "full-time"
    PART_TIME  = "part-time"
    CONTRACT   = "contract"
    FREELANCE  = "freelance"
    INTERNSHIP = "internship"


class WorkMode(str, Enum):
    REMOTE  = "remote"
    HYBRID  = "hybrid"
    ONSITE  = "onsite"


class ExperienceLevel(str, Enum):
    ENTRY     = "entry"
    MID       = "mid"
    SENIOR    = "senior"
    LEAD      = "lead"
    EXECUTIVE = "executive"

# ══════════════════════════════════════════════════════════════════════════════
# LinkedIn Post
# ══════════════════════════════════════════════════════════════════════════════

class LinkedInPostStatus(str, Enum):
    DRAFT      = "draft"
    SCHEDULED  = "scheduled"
    PUBLISHED  = "published"
    FAILED     = "failed"


class LinkedInPostTone(str, Enum):
    THOUGHT_LEADER = "thought_leader"
    PROFESSIONAL   = "professional"
    ENTHUSIASTIC   = "enthusiastic"
    FORMAL         = "formal"
    EDUCATIONAL    = "educational"
    PERSONAL       = "personal"

# ══════════════════════════════════════════════════════════════════════════════
# Agent Run Status
# ══════════════════════════════════════════════════════════════════════════════

class AgentRunStatus(str, Enum):
    PENDING   = "pending"
    RUNNING   = "running"
    COMPLETED = "completed"
    FAILED    = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"

# ══════════════════════════════════════════════════════════════════════════════
# Notification Types
# ══════════════════════════════════════════════════════════════════════════════

class NotificationType(str, Enum):
    APPLICATION_SUBMITTED    = "application_submitted"
    APPLICATION_STATUS_CHANGED = "application_status_changed"
    INTERVIEW_SCHEDULED      = "interview_scheduled"
    JOB_MATCHED              = "job_matched"
    RESUME_PROCESSED         = "resume_processed"
    LINKEDIN_POST_PUBLISHED  = "linkedin_post_published"
    AGENT_FAILED             = "agent_failed"
    WEEKLY_SUMMARY           = "weekly_summary"


class NotificationChannel(str, Enum):
    EMAIL   = "email"
    WEBHOOK = "webhook"
    IN_APP  = "in_app"

# ══════════════════════════════════════════════════════════════════════════════
# Audit Events
# ══════════════════════════════════════════════════════════════════════════════

class AuditEvent(str, Enum):
    USER_REGISTERED     = "user.registered"
    USER_LOGIN          = "user.login"
    USER_LOGOUT         = "user.logout"
    RESUME_UPLOADED     = "resume.uploaded"
    RESUME_TAILORED     = "resume.tailored"
    JOB_DISCOVERED      = "job.discovered"
    APPLICATION_CREATED = "application.created"
    APPLICATION_SUBMITTED = "application.submitted"
    LINKEDIN_POST_PUBLISHED = "linkedin.post_published"
    AGENT_RUN_COMPLETED = "agent.run_completed"
    AGENT_RUN_FAILED    = "agent.run_failed"

# ══════════════════════════════════════════════════════════════════════════════
# LLM / Groq
# ══════════════════════════════════════════════════════════════════════════════

GROQ_DEFAULT_MODEL          = "llama-3.3-70b-versatile"
GROQ_FAST_MODEL             = "llama-3.1-8b-instant"
GROQ_MAX_TOKENS             = 8192
GROQ_TEMPERATURE_CREATIVE   = 0.8
GROQ_TEMPERATURE_PRECISE    = 0.1
GROQ_TEMPERATURE_BALANCED   = 0.4
GROQ_MAX_RETRIES            = 3
GROQ_RETRY_DELAY_SECONDS    = 2.0
GROQ_TIMEOUT_SECONDS        = 60

# ══════════════════════════════════════════════════════════════════════════════
# Playwright
# ══════════════════════════════════════════════════════════════════════════════

PLAYWRIGHT_TIMEOUT_MS           = 30_000
PLAYWRIGHT_NAVIGATION_TIMEOUT_MS = 60_000
PLAYWRIGHT_HEADLESS             = True
PLAYWRIGHT_SLOW_MO_MS           = 0
MAX_FORM_FILL_RETRIES           = 3
PLAYWRIGHT_MAX_APPLY_STEPS      = 10
HUMAN_TYPING_DELAY_MS           = (50, 150)

# ══════════════════════════════════════════════════════════════════════════════
# Scraping
# ══════════════════════════════════════════════════════════════════════════════

SCRAPE_REQUEST_TIMEOUT_SECONDS = 30
SCRAPE_MAX_CONCURRENT          = 5
SCRAPE_RETRY_ATTEMPTS          = 3
SCRAPE_RETRY_BACKOFF_SECONDS   = 5

# ══════════════════════════════════════════════════════════════════════════════
# Match Score Thresholds
# ══════════════════════════════════════════════════════════════════════════════

MATCH_SCORE_EXCELLENT = 0.90
MATCH_SCORE_GOOD      = 0.75
MATCH_SCORE_FAIR      = 0.60
MATCH_SCORE_POOR      = 0.45


class MatchTier(str, Enum):
    EXCELLENT = "excellent"
    GOOD      = "good"
    FAIR      = "fair"
    POOR      = "poor"


def get_match_tier(score: float) -> MatchTier:
    if score >= MATCH_SCORE_EXCELLENT: return MatchTier.EXCELLENT
    if score >= MATCH_SCORE_GOOD:      return MatchTier.GOOD
    if score >= MATCH_SCORE_FAIR:      return MatchTier.FAIR
    return MatchTier.POOR

# ══════════════════════════════════════════════════════════════════════════════
# User Plan
# ══════════════════════════════════════════════════════════════════════════════

class UserPlan(str, Enum):
    FREE       = "free"
    PRO        = "pro"
    ENTERPRISE = "enterprise"


PLAN_LIMITS: dict[UserPlan, dict[str, Any]] = {
    UserPlan.FREE: {
        "monthly_applications":     10,
        "resume_versions":          2,
        "linkedin_posts_per_day":   1,
        "job_alerts":               3,
        "ai_rewrites":              5,
    },
    UserPlan.PRO: {
        "monthly_applications":     200,
        "resume_versions":          20,
        "linkedin_posts_per_day":   5,
        "job_alerts":               50,
        "ai_rewrites":              100,
    },
    UserPlan.ENTERPRISE: {
        "monthly_applications":     -1,
        "resume_versions":          -1,
        "linkedin_posts_per_day":   -1,
        "job_alerts":               -1,
        "ai_rewrites":              -1,
    },
}

# ══════════════════════════════════════════════════════════════════════════════
# Followup Rules
# ══════════════════════════════════════════════════════════════════════════════

FOLLOWUP_AFTER_DAYS          = 7
FOLLOWUP_WAIT_DAYS           = 7   # alias used by datetime_utils / workflows
MAX_FOLLOWUPS_PER_APP        = 3
MAX_FOLLOWUPS_PER_APPLICATION = 2  # hard cap enforced by the API + worker layer
OUTREACH_SEQUENCE_STEPS      = 3

# ══════════════════════════════════════════════════════════════════════════════
# HTTP Status Codes (aliases)
# ══════════════════════════════════════════════════════════════════════════════

HTTP_200_OK                   = 200
HTTP_201_CREATED              = 201
HTTP_204_NO_CONTENT           = 204
HTTP_400_BAD_REQUEST          = 400
HTTP_401_UNAUTHORIZED         = 401
HTTP_403_FORBIDDEN            = 403
HTTP_404_NOT_FOUND            = 404
HTTP_409_CONFLICT             = 409
HTTP_422_UNPROCESSABLE_ENTITY = 422
HTTP_429_TOO_MANY_REQUESTS    = 429
HTTP_500_INTERNAL_SERVER_ERROR = 500
HTTP_502_BAD_GATEWAY          = 502
HTTP_503_SERVICE_UNAVAILABLE  = 503

# ══════════════════════════════════════════════════════════════════════════════
# Cache TTLs (seconds)
# ══════════════════════════════════════════════════════════════════════════════

CACHE_TTL_JOB_LISTING  = 60 * 30
CACHE_TTL_USER_PROFILE = 60 * 5
CACHE_TTL_MATCH_RESULTS = 60 * 60
CACHE_TTL_ANALYTICS    = 60 * 15
CACHE_TTL_LINKEDIN_POST = 60 * 60 * 6

# ══════════════════════════════════════════════════════════════════════════════
# Error Codes
# ══════════════════════════════════════════════════════════════════════════════

ERR_VALIDATION              = "VALIDATION_ERROR"
ERR_NOT_FOUND               = "NOT_FOUND"
ERR_UNAUTHORIZED            = "UNAUTHORIZED"
ERR_FORBIDDEN               = "FORBIDDEN"
ERR_CONFLICT                = "CONFLICT"
ERR_RATE_LIMITED            = "RATE_LIMITED"
ERR_AI_SERVICE              = "AI_SERVICE_ERROR"
ERR_VECTOR_STORE            = "VECTOR_STORE_ERROR"
ERR_SCRAPER                 = "SCRAPER_ERROR"
ERR_AUTOMATION              = "AUTOMATION_ERROR"
ERR_INTERNAL                = "INTERNAL_SERVER_ERROR"
ERR_INVALID_STATUS_TRANSITION = "INVALID_STATUS_TRANSITION"
ERR_PLAN_LIMIT_EXCEEDED     = "PLAN_LIMIT_EXCEEDED"

# ══════════════════════════════════════════════════════════════════════════════
# Resume Sections
# ══════════════════════════════════════════════════════════════════════════════

class ResumeSection(str, Enum):
    SUMMARY        = "summary"
    EXPERIENCE     = "experience"
    EDUCATION      = "education"
    SKILLS         = "skills"
    PROJECTS       = "projects"
    CERTIFICATIONS = "certifications"
    PUBLICATIONS   = "publications"
    AWARDS         = "awards"
    LANGUAGES      = "languages"
    VOLUNTEERING   = "volunteering"
    REFERENCES     = "references"


class OptimizationLevel(str, Enum):
    """How aggressively the resume agent rewrites content when tailoring."""
    CONSERVATIVE = "conservative"
    BALANCED     = "balanced"
    AGGRESSIVE   = "aggressive"

# ══════════════════════════════════════════════════════════════════════════════
# LinkedIn Post Category
# ══════════════════════════════════════════════════════════════════════════════

class LinkedInPostCategory(str, Enum):
    AI_INSIGHTS       = "ai_insights"
    CAREER_TIPS       = "career_tips"
    TECH_TRENDS       = "tech_trends"
    PRODUCTIVITY      = "productivity"
    INDUSTRY_NEWS     = "industry_news"
    PERSONAL_BRAND    = "personal_brand"
    JOB_SEARCH_TIPS   = "job_search_tips"
    LEADERSHIP        = "leadership"

# ══════════════════════════════════════════════════════════════════════════════
# Date / Misc
# ══════════════════════════════════════════════════════════════════════════════

DATE_FORMAT             = "%Y-%m-%d"
DATETIME_FORMAT         = "%Y-%m-%dT%H:%M:%SZ"
DATETIME_FORMAT_DISPLAY = "%B %d, %Y %I:%M %p"
TIMEZONE_UTC            = "UTC"
DEFAULT_COUNTRY         = "US"
DEFAULT_CURRENCY        = "USD"
PLATFORM_NAME           = "CareerGPT"
SUPPORT_EMAIL           = "support@careergpt.ai"
MAX_SKILLS_PER_RESUME   = 100
MAX_COVER_LETTER_WORDS  = 500
MIN_COVER_LETTER_WORDS  = 150
MAX_LINKEDIN_POST_CHARS = 3000
MIN_LINKEDIN_POST_CHARS = 100
LINKEDIN_POST_HASHTAG_LIMIT = 5