"""
CareerGPT — Unified Configuration
====================================
Production-grade Pydantic-Settings v2 config.
Supports BOTH uppercase (your original config.py) and
lowercase (what main.py + all agents expect) attribute access.

All settings read from environment variables / .env file.
"""

from __future__ import annotations

import secrets
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import Field, EmailStr, computed_field, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,   # Allow both APP_ENV and app_env
        extra="ignore",
        validate_default=True,
    )

    # ── Application ───────────────────────────────────────────────────────────
    APP_NAME: str = "CareerGPT"
    APP_ENV: Literal["development", "staging", "production", "test"] = "development"
    APP_VERSION: str = "1.0.0"
    APP_DEBUG: bool = False
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000
    APP_RELOAD: bool = False
    APP_WORKERS: int = 4
    APP_LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    APP_SECRET_KEY: str = Field(
        default_factory=lambda: secrets.token_urlsafe(64),
        repr=False,
    )

    # CORS — comma-separated string or list
    BACKEND_CORS_ORIGINS: Annotated[list[str], NoDecode] = ["http://localhost:3000", "http://localhost:3001"]

    @field_validator("BACKEND_CORS_ORIGINS", mode="before")
    @classmethod
    def parse_cors(cls, v: Any) -> list[str]:
        if isinstance(v, str):
            return [o.strip() for o in v.split(",") if o.strip()]
        return list(v) if v else ["http://localhost:3000"]

    # Feature flags
    RUN_MIGRATIONS_ON_STARTUP: bool = False
    WARMUP_EMBEDDING_MODEL: bool = True

    # ── Database ──────────────────────────────────────────────────────────────
    DB_HOST: str = "localhost"
    DB_PORT: int = 5432
    DB_USER: str = "postgres"
    DB_PASSWORD: str = Field(default="password", repr=False)
    DB_NAME: str = "careergpt"
    DB_POOL_SIZE: int = 10
    DB_MAX_OVERFLOW: int = 10
    DB_POOL_TIMEOUT: int = 30
    DB_POOL_RECYCLE: int = 1800
    DB_ECHO_SQL: bool = False

    @computed_field
    @property
    def DATABASE_URL(self) -> str:
        return (
            f"postgresql+asyncpg://{self.DB_USER}:{self.DB_PASSWORD}"
            f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        )

    @computed_field
    @property
    def DATABASE_URL_SYNC(self) -> str:
        return (
            f"postgresql+psycopg2://{self.DB_USER}:{self.DB_PASSWORD}"
            f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        )

    # ── Redis ─────────────────────────────────────────────────────────────────
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: str | None = Field(default=None, repr=False)
    REDIS_DB: int = 0
    REDIS_MAX_CONNECTIONS: int = 50

    @computed_field
    @property
    def REDIS_URL(self) -> str:
        auth = f":{self.REDIS_PASSWORD}@" if self.REDIS_PASSWORD else ""
        return f"redis://{auth}{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"

    # ── JWT ───────────────────────────────────────────────────────────────────
    JWT_SECRET_KEY: str = Field(
        default_factory=lambda: secrets.token_urlsafe(64),
        repr=False,
    )
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440   # 24h
    JWT_REFRESH_TOKEN_EXPIRE_DAYS: int = 30

    # ── LLM: Groq (FREE primary) ──────────────────────────────────────────────
    GROQ_API_KEY: str = Field(default="", repr=False)
    GROQ_DEFAULT_MODEL: str = "llama-3.3-70b-versatile"
    GROQ_FAST_MODEL: str = "llama-3.1-8b-instant"
    GROQ_MAX_TOKENS: int = 8192
    GROQ_TEMPERATURE: float = 0.4
    GROQ_MAX_RETRIES: int = 3
    GROQ_TIMEOUT: int = 60

    # ── LLM: Anthropic (Fallback) ─────────────────────────────────────────────
    ANTHROPIC_API_KEY: str | None = Field(default=None, repr=False)

    # ── LLM: OpenAI (Last resort) ─────────────────────────────────────────────
    OPENAI_API_KEY: str | None = Field(default=None, repr=False)

    # ── Image Generation ──────────────────────────────────────────────────────
    # Pollinations.ai = FREE, no key needed
    TOGETHER_API_KEY: str | None = Field(default=None, repr=False)

    # ── Qdrant ────────────────────────────────────────────────────────────────
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_API_KEY: str | None = Field(default=None, repr=False)
    QDRANT_USE_TLS: bool = False
    QDRANT_GRPC_PORT: int = 6334
    QDRANT_PREFER_GRPC: bool = False
    QDRANT_TIMEOUT: int = 30
    QDRANT_VECTOR_SIZE: int = 384

    @computed_field
    @property
    def QDRANT_URL(self) -> str:
        scheme = "https" if self.QDRANT_USE_TLS else "http"
        return f"{scheme}://{self.QDRANT_HOST}:{self.QDRANT_PORT}"

    # ── Storage ───────────────────────────────────────────────────────────────
    UPLOAD_DIR: str = "./uploads"
    USE_LOCAL_STORAGE: bool = True
    AWS_ACCESS_KEY_ID: str | None = Field(default=None, repr=False)
    AWS_SECRET_ACCESS_KEY: str | None = Field(default=None, repr=False)
    AWS_REGION: str = "us-east-1"
    AWS_S3_BUCKET: str | None = None
    AWS_S3_ENDPOINT_URL: str | None = None

    # ── LinkedIn ──────────────────────────────────────────────────────────────
    LINKEDIN_CLIENT_ID: str | None = Field(default=None, repr=False)
    LINKEDIN_CLIENT_SECRET: str | None = Field(default=None, repr=False)
    LINKEDIN_REDIRECT_URI: str = "http://localhost:8000/api/v1/linkedin/oauth/callback"
    LINKEDIN_EMAIL: str | None = None   # For Playwright fallback
    LINKEDIN_PASSWORD: str | None = Field(default=None, repr=False)

    # ── Email ─────────────────────────────────────────────────────────────────
    SMTP_HOST: str = "localhost"
    SMTP_PORT: int = 587
    SMTP_USER: str | None = None
    SMTP_PASSWORD: str | None = Field(default=None, repr=False)
    SMTP_USE_TLS: bool = True
    SMTP_FROM_EMAIL: str = "noreply@careergpt.ai"
    SMTP_FROM_NAME: str = "CareerGPT"
    EMAIL_ENABLED: bool = False
    SENDGRID_API_KEY: str | None = Field(default=None, repr=False)

    # ── Celery ────────────────────────────────────────────────────────────────
    CELERY_BROKER_URL: str | None = None
    CELERY_RESULT_BACKEND: str | None = None
    CELERY_TASK_SERIALIZER: str = "json"
    CELERY_RESULT_SERIALIZER: str = "json"
    CELERY_ACCEPT_CONTENT: list[str] = ["json"]
    CELERY_TIMEZONE: str = "UTC"
    CELERY_TASK_SOFT_TIME_LIMIT: int = 300
    CELERY_TASK_TIME_LIMIT: int = 600
    CELERY_WORKER_MAX_TASKS_PER_CHILD: int = 50

    @model_validator(mode="after")
    def set_celery_defaults(self) -> "Settings":
        if self.CELERY_BROKER_URL is None:
            self.CELERY_BROKER_URL = self.REDIS_URL
        if self.CELERY_RESULT_BACKEND is None:
            self.CELERY_RESULT_BACKEND = self.REDIS_URL
        return self

    # ── Playwright ────────────────────────────────────────────────────────────
    PLAYWRIGHT_HEADLESS: bool = True
    PLAYWRIGHT_TIMEOUT_MS: int = 30_000
    PLAYWRIGHT_SLOW_MO_MS: int = 0
    PLAYWRIGHT_PROXY_URL: str | None = None

    # ── Scraping ──────────────────────────────────────────────────────────────
    SCRAPING_PROXY_URL: str | None = None
    SCRAPE_REQUEST_TIMEOUT_SECONDS: int = 30

    # ── Sentry ────────────────────────────────────────────────────────────────
    SENTRY_DSN: str | None = Field(default=None, repr=False)

    # ── Rate Limiting ─────────────────────────────────────────────────────────
    RATE_LIMIT_REQUESTS_PER_MINUTE: int = 100

    # ── Frontend ──────────────────────────────────────────────────────────────
    FRONTEND_URL: str = "http://localhost:3000"

    # ── Embedding ─────────────────────────────────────────────────────────────
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

    # ══════════════════════════════════════════════════════════════════════════
    # LOWERCASE ALIASES — what main.py + all agents expect
    # ══════════════════════════════════════════════════════════════════════════

    @property
    def app_name(self) -> str:
        return self.APP_NAME

    @property
    def app_env(self) -> str:
        return self.APP_ENV

    @property
    def app_version(self) -> str:
        return self.APP_VERSION

    @property
    def debug(self) -> bool:
        return self.APP_DEBUG

    @property
    def host(self) -> str:
        return self.APP_HOST

    @property
    def port(self) -> int:
        return self.APP_PORT

    # app_-prefixed aliases — this is what main.py's lifespan/uvicorn.run
    # actually reads, kept alongside the un-prefixed ones above for safety.
    @property
    def app_debug(self) -> bool:
        return self.APP_DEBUG

    @property
    def app_host(self) -> str:
        return self.APP_HOST

    @property
    def app_port(self) -> int:
        return self.APP_PORT

    @property
    def app_workers(self) -> int:
        return self.APP_WORKERS

    @property
    def app_secret_key(self) -> str:
        return self.APP_SECRET_KEY

    @property
    def is_production(self) -> bool:
        return self.APP_ENV == "production"

    @property
    def is_development(self) -> bool:
        return self.APP_ENV == "development"

    @property
    def cors_origins(self) -> list[str]:
        return self.BACKEND_CORS_ORIGINS

    @property
    def rate_limit_requests_per_minute(self) -> int:
        return self.RATE_LIMIT_REQUESTS_PER_MINUTE

    @property
    def run_migrations_on_startup(self) -> bool:
        return self.RUN_MIGRATIONS_ON_STARTUP

    @property
    def warmup_embedding_model(self) -> bool:
        return self.WARMUP_EMBEDDING_MODEL

    @property
    def groq_api_key(self) -> str:
        return self.GROQ_API_KEY

    @property
    def anthropic_api_key(self) -> str | None:
        return self.ANTHROPIC_API_KEY

    @property
    def openai_api_key(self) -> str | None:
        return self.OPENAI_API_KEY

    @property
    def sentry_dsn(self) -> str | None:
        return self.SENTRY_DSN

    @property
    def frontend_url(self) -> str:
        return self.FRONTEND_URL

    @property
    def from_email(self) -> str:
        return self.SMTP_FROM_EMAIL

    @property
    def from_name(self) -> str:
        return self.SMTP_FROM_NAME

    @property
    def sendgrid_api_key(self) -> str | None:
        return self.SENDGRID_API_KEY

    @property
    def smtp_host(self) -> str:
        return self.SMTP_HOST

    @property
    def smtp_port(self) -> int:
        return self.SMTP_PORT

    @property
    def smtp_user(self) -> str | None:
        return self.SMTP_USER

    @property
    def smtp_password(self) -> str | None:
        return self.SMTP_PASSWORD

    @property
    def smtp_tls(self) -> bool:
        return self.SMTP_USE_TLS

    # ── Nested-style accessors (what middleware + agents use) ─────────────────

    @property
    def redis(self) -> "_RedisConfig":
        return _RedisConfig(url_str=self.REDIS_URL)

    @property
    def qdrant(self) -> "_QdrantConfig":
        return _QdrantConfig(
            url=self.QDRANT_URL,
            api_key=self.QDRANT_API_KEY,
            timeout=self.QDRANT_TIMEOUT,
            vector_size=self.QDRANT_VECTOR_SIZE,
        )

    @property
    def storage(self) -> "_StorageConfig":
        return _StorageConfig(upload_dir=Path(self.UPLOAD_DIR))

    @property
    def llm(self) -> "_LLMConfig":
        return _LLMConfig(embedding_model=self.EMBEDDING_MODEL)

    @property
    def linkedin(self) -> "_LinkedInConfig":
        return _LinkedInConfig(
            client_id=self.LINKEDIN_CLIENT_ID or "",
            client_secret=self.LINKEDIN_CLIENT_SECRET or "",
            redirect_uri=self.LINKEDIN_REDIRECT_URI,
            email=self.LINKEDIN_EMAIL or "",
            password=self.LINKEDIN_PASSWORD or "",
        )

    @property
    def playwright(self) -> "_PlaywrightConfig":
        return _PlaywrightConfig(
            headless=self.PLAYWRIGHT_HEADLESS,
            timeout_ms=self.PLAYWRIGHT_TIMEOUT_MS,
            slow_mo=self.PLAYWRIGHT_SLOW_MO_MS,
            proxy_url=self.PLAYWRIGHT_PROXY_URL,
        )

    # ── Celery config dict ────────────────────────────────────────────────────

    def get_celery_config(self) -> dict[str, Any]:
        return {
            "broker_url":                  self.CELERY_BROKER_URL,
            "result_backend":              self.CELERY_RESULT_BACKEND,
            "task_serializer":             self.CELERY_TASK_SERIALIZER,
            "result_serializer":           self.CELERY_RESULT_SERIALIZER,
            "accept_content":              self.CELERY_ACCEPT_CONTENT,
            "timezone":                    self.CELERY_TIMEZONE,
            "task_soft_time_limit":        self.CELERY_TASK_SOFT_TIME_LIMIT,
            "task_time_limit":             self.CELERY_TASK_TIME_LIMIT,
        }

    def get_db_kwargs(self) -> dict[str, Any]:
        return {
            "pool_size":     self.DB_POOL_SIZE,
            "max_overflow":  self.DB_MAX_OVERFLOW,
            "pool_timeout":  self.DB_POOL_TIMEOUT,
            "pool_recycle":  self.DB_POOL_RECYCLE,
            "echo":          self.DB_ECHO_SQL,
            "pool_pre_ping": True,
        }


# ── Nested config dataclasses ─────────────────────────────────────────────────

class _RedisConfig:
    def __init__(self, url_str: str) -> None:
        self.url_str = url_str
        # Alias — some callers (main.py) read `.url` instead of `.url_str`.
        self.url = url_str


class _QdrantConfig:
    def __init__(self, url: str, api_key: str | None, timeout: int = 30, vector_size: int = 384) -> None:
        self.url         = url
        self.api_key     = api_key
        self.timeout     = timeout
        self.vector_size = vector_size


class _StorageConfig:
    def __init__(self, upload_dir: Path) -> None:
        self.upload_dir = upload_dir
        upload_dir.mkdir(parents=True, exist_ok=True)


class _LLMConfig:
    def __init__(self, embedding_model: str) -> None:
        self.embedding_model = embedding_model


class _LinkedInConfig:
    def __init__(
        self,
        client_id: str,
        client_secret: str,
        redirect_uri: str,
        email: str,
        password: str,
    ) -> None:
        self.client_id     = client_id
        self.client_secret = client_secret
        self.redirect_uri  = redirect_uri
        self.email         = email
        self.password      = password


class _PlaywrightConfig:
    def __init__(self, headless: bool, timeout_ms: int, slow_mo: int, proxy_url: str | None = None) -> None:
        self.headless   = headless
        self.timeout_ms = timeout_ms
        self.slow_mo    = slow_mo
        self.proxy_url  = proxy_url


# ── Singleton ─────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings: Settings = get_settings()