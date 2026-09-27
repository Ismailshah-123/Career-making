"""
CareerGPT — Exception Hierarchy
==================================
All application exceptions with exact names main.py + all agents import.
Every exception carries: error_code, http_status, context dict.
"""

from __future__ import annotations
from typing import Any


# ══════════════════════════════════════════════════════════════════════════════
# Base
# ══════════════════════════════════════════════════════════════════════════════

class CareerGPTError(Exception):
    """Root exception for all CareerGPT errors."""
    http_status: int = 500
    error_code:  str = "INTERNAL_ERROR"

    def __init__(
        self,
        message: str = "An unexpected error occurred.",
        *,
        error_code: str | None = None,
        context: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.message    = message
        self.error_code = error_code or self.__class__.error_code
        self.context    = context or {}
        self.headers    = headers or {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "error_code": self.error_code,
            "message":    self.message,
            "context":    self.context,
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(code={self.error_code!r}, msg={self.message!r})"


# ── Alias used by middleware.py (your file uses BaseAppException) ─────────────
BaseAppException = CareerGPTError


# ══════════════════════════════════════════════════════════════════════════════
# 400 — Validation
# ══════════════════════════════════════════════════════════════════════════════

class ValidationError(CareerGPTError):
    """Input validation failed (business logic, not Pydantic schema)."""
    http_status = 400
    error_code  = "VALIDATION_ERROR"

    def __init__(
        self,
        message: str = "Validation failed.",
        *,
        field: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        ctx = context or {}
        if field:
            ctx["field"] = field
        super().__init__(message, context=ctx)


# Alias used by your middleware.py (ValidationException)
ValidationException = ValidationError


class FileTooLargeError(ValidationError):
    http_status = 400
    error_code  = "FILE_TOO_LARGE"

    def __init__(self, filename: str = "", max_mb: int = 10) -> None:
        super().__init__(
            f"File '{filename}' exceeds the {max_mb} MB size limit.",
            context={"filename": filename, "max_size_mb": max_mb},
        )


# Alias
FileTooLargeException = FileTooLargeError


class InvalidFileTypeError(ValidationError):
    http_status = 400
    error_code  = "INVALID_FILE_TYPE"

    def __init__(self, message: str = "File type not allowed.", context: dict | None = None) -> None:
        super().__init__(message, context=context)


InvalidFileTypeException = InvalidFileTypeError


class ResumeParseError(ValidationError):
    error_code = "RESUME_PARSE_ERROR"

    def __init__(self, filename: str = "", reason: str | None = None) -> None:
        super().__init__(
            f"Could not parse resume '{filename}'.",
            context={"filename": filename, "reason": reason},
        )


ResumeParseException = ResumeParseError


# ══════════════════════════════════════════════════════════════════════════════
# 401 — Authentication
# ══════════════════════════════════════════════════════════════════════════════

class AuthenticationError(CareerGPTError):
    """Request cannot be authenticated."""
    http_status = 401
    error_code  = "UNAUTHORIZED"

    def __init__(
        self,
        message: str = "Authentication required.",
        *,
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message,
            context=context,
            headers={"WWW-Authenticate": "Bearer"},
        )


AuthenticationException = AuthenticationError


class InvalidTokenError(AuthenticationError):
    error_code = "INVALID_TOKEN"

    def __init__(self, reason: str = "Token is invalid or expired.") -> None:
        super().__init__(reason, context={"reason": reason})


InvalidTokenException = InvalidTokenError


class TokenExpiredError(AuthenticationError):
    error_code = "TOKEN_EXPIRED"

    def __init__(self) -> None:
        super().__init__(
            "Access token has expired.",
            context={"hint": "Use the refresh endpoint to get a new token."},
        )


TokenExpiredException = TokenExpiredError


class InvalidCredentialsError(AuthenticationError):
    error_code = "INVALID_CREDENTIALS"

    def __init__(self) -> None:
        super().__init__("Invalid email or password.")


InvalidCredentialsException = InvalidCredentialsError


class AccountLockedError(AuthenticationError):
    error_code = "ACCOUNT_LOCKED"

    def __init__(self, unlock_at: str | None = None) -> None:
        ctx: dict[str, Any] = {}
        if unlock_at:
            ctx["unlock_at"] = unlock_at
        super().__init__("Account is temporarily locked.", context=ctx)


AccountLockedException = AccountLockedError


class AccountDeactivatedError(AuthenticationError):
    """Raised when a user with is_active=False attempts to authenticate."""
    error_code = "ACCOUNT_DEACTIVATED"

    def __init__(self, context: dict[str, Any] | None = None) -> None:
        super().__init__("This account has been deactivated.", context=context)


# Alias — auth_service.py's refresh/verification flows raise this name for
# an invalid token; it's the same failure InvalidTokenError already models.
TokenInvalidError = InvalidTokenError


class AuthError(CareerGPTError):
    """
    Generic authentication-flow error (password reset, email verification,
    etc.) where the caller supplies its own error_code per failure reason.
    """
    http_status = 400
    error_code  = "AUTH_ERROR"


# ══════════════════════════════════════════════════════════════════════════════
# 403 — Authorization
# ══════════════════════════════════════════════════════════════════════════════

class AuthorizationError(CareerGPTError):
    """Authenticated user lacks permission."""
    http_status = 403
    error_code  = "FORBIDDEN"

    def __init__(
        self,
        message: str = "You do not have permission to perform this action.",
        *,
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, context=context)


AuthorizationException = AuthorizationError


class OwnershipError(AuthorizationError):
    def __init__(self, resource: str) -> None:
        super().__init__(f"You do not own this {resource}.", context={"resource": resource})


OwnershipException = OwnershipError


class PlanFeatureError(AuthorizationError):
    def __init__(self, feature: str, required_plan: str) -> None:
        super().__init__(
            f"'{feature}' requires the {required_plan} plan.",
            context={"feature": feature, "required_plan": required_plan},
        )


PlanFeatureException = PlanFeatureError


# ══════════════════════════════════════════════════════════════════════════════
# 404 — Not Found
# ══════════════════════════════════════════════════════════════════════════════

class NotFoundError(CareerGPTError):
    """Resource does not exist."""
    http_status = 404
    error_code  = "NOT_FOUND"

    def __init__(
        self,
        resource: str = "Resource",
        identifier: str | None = None,
        message: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        msg = message or (
            f"{resource} '{identifier}' not found."
            if identifier else f"{resource} not found."
        )
        ctx = context or {}
        ctx["resource"] = resource
        if identifier:
            ctx["identifier"] = identifier
        super().__init__(msg, context=ctx)


NotFoundException = NotFoundError


class UserNotFoundError(NotFoundError):
    def __init__(self, user_id: str = "") -> None:
        super().__init__("User", user_id)


UserNotFoundException = UserNotFoundError


class JobNotFoundError(NotFoundError):
    def __init__(self, job_id: str = "") -> None:
        super().__init__("Job", job_id)


JobNotFoundException = JobNotFoundError


class ResumeNotFoundError(NotFoundError):
    def __init__(self, resume_id: str = "") -> None:
        super().__init__("Resume", resume_id)


ResumeNotFoundException = ResumeNotFoundError


class ApplicationNotFoundError(NotFoundError):
    def __init__(self, application_id: str = "") -> None:
        super().__init__("Application", application_id)


ApplicationNotFoundException = ApplicationNotFoundError


class PostNotFoundError(NotFoundError):
    def __init__(self, post_id: str = "") -> None:
        super().__init__("LinkedIn Post", post_id)


# ══════════════════════════════════════════════════════════════════════════════
# 409 — Conflict
# ══════════════════════════════════════════════════════════════════════════════

class ConflictError(CareerGPTError):
    http_status = 409
    error_code  = "CONFLICT"

    def __init__(
        self,
        message: str = "A conflict occurred.",
        *,
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, context=context)


ConflictException = ConflictError


class DuplicateEmailError(ConflictError):
    def __init__(self, email: str) -> None:
        super().__init__(
            f"Account with email '{email}' already exists.",
            context={"field": "email", "value": email},
        )


DuplicateEmailException = DuplicateEmailError


class DuplicateApplicationError(ConflictError):
    def __init__(self, job_id: str = "") -> None:
        super().__init__(
            "You have already applied to this job.",
            context={"job_id": job_id},
        )


DuplicateApplicationException = DuplicateApplicationError


class InvalidStatusTransitionError(ConflictError):
    error_code = "INVALID_STATUS_TRANSITION"

    def __init__(self, from_status: str, to_status: str) -> None:
        super().__init__(
            f"Cannot transition from '{from_status}' to '{to_status}'.",
            context={"from_status": from_status, "to_status": to_status},
        )


InvalidStatusTransition = InvalidStatusTransitionError


# ══════════════════════════════════════════════════════════════════════════════
# 429 — Rate Limit
# ══════════════════════════════════════════════════════════════════════════════

class RateLimitError(CareerGPTError):
    http_status = 429
    error_code  = "RATE_LIMIT_EXCEEDED"

    def __init__(
        self,
        message: str = "Too many requests. Please slow down.",
        *,
        retry_after: int | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        ctx = context or {}
        headers: dict[str, str] = {}
        if retry_after is not None:
            ctx["retry_after_seconds"] = retry_after
            headers["Retry-After"]     = str(retry_after)
        super().__init__(message, context=ctx, headers=headers)


RateLimitException = RateLimitError


# ══════════════════════════════════════════════════════════════════════════════
# 402 — Plan Limit
# ══════════════════════════════════════════════════════════════════════════════

class PlanLimitExceededError(CareerGPTError):
    http_status = 402
    error_code  = "PLAN_LIMIT_EXCEEDED"

    def __init__(self, limit_name: str, current_plan: str, limit_value: int) -> None:
        super().__init__(
            f"Reached '{limit_name}' limit ({limit_value}) on {current_plan} plan.",
            context={"limit": limit_name, "plan": current_plan, "limit_value": limit_value},
        )


PlanLimitExceededException = PlanLimitExceededError


# ══════════════════════════════════════════════════════════════════════════════
# 502 — External Services
# ══════════════════════════════════════════════════════════════════════════════

class LLMError(CareerGPTError):
    """Groq / LLM provider error."""
    http_status = 503
    error_code  = "LLM_ERROR"

    def __init__(
        self,
        message: str = "AI service temporarily unavailable.",
        *,
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, context=context)


# Aliases
AIServiceError      = LLMError
AIServiceException  = LLMError


class LLMQuotaError(LLMError):
    error_code = "LLM_QUOTA_EXCEEDED"

    def __init__(self, message: str = "LLM rate limit exceeded.", context: dict | None = None) -> None:
        super().__init__(message, context=context)


GroqRateLimitException = LLMQuotaError


class LLMTimeoutError(LLMError):
    error_code = "LLM_TIMEOUT"

    def __init__(self, message: str = "LLM request timed out.", context: dict | None = None) -> None:
        super().__init__(message, context=context)


class VectorDBError(CareerGPTError):
    http_status = 502
    error_code  = "VECTOR_DB_ERROR"

    def __init__(self, message: str = "Vector database error.", context: dict | None = None) -> None:
        super().__init__(message, context=context)


VectorStoreException = VectorDBError
VectorStoreError     = VectorDBError


class ScraperError(CareerGPTError):
    http_status = 502
    error_code  = "SCRAPER_ERROR"

    def __init__(self, message: str = "Scraping failed.", context: dict | None = None) -> None:
        super().__init__(message, context=context)


ScraperException = ScraperError


class PlaywrightError(CareerGPTError):
    http_status = 502
    error_code  = "PLAYWRIGHT_ERROR"

    def __init__(self, message: str = "Browser automation failed.", context: dict | None = None) -> None:
        super().__init__(message, context=context)


AutomationError      = PlaywrightError
AutomationException  = PlaywrightError


class LinkedInError(CareerGPTError):
    http_status = 502
    error_code  = "LINKEDIN_ERROR"

    def __init__(self, message: str = "LinkedIn operation failed.", context: dict | None = None) -> None:
        super().__init__(message, context=context)


class LinkedInPublishError(LinkedInError):
    error_code = "LINKEDIN_PUBLISH_ERROR"


# ══════════════════════════════════════════════════════════════════════════════
# 500 — Internal
# ══════════════════════════════════════════════════════════════════════════════

class InternalError(CareerGPTError):
    http_status = 500
    error_code  = "INTERNAL_ERROR"


InternalServerException = InternalError


class DatabaseError(InternalError):
    error_code = "DATABASE_ERROR"

    def __init__(self, operation: str = "", reason: str | None = None) -> None:
        super().__init__(
            f"Database operation '{operation}' failed.",
            context={"operation": operation, "reason": reason},
        )


DatabaseException = DatabaseError


class StorageError(InternalError):
    error_code = "STORAGE_ERROR"

    def __init__(self, message: str = "Storage operation failed.", context: dict | None = None) -> None:
        super().__init__(message, context=context)


class EmbeddingError(InternalError):
    error_code = "EMBEDDING_ERROR"

    def __init__(self, message: str = "Embedding failed.", context: dict | None = None) -> None:
        super().__init__(message, context=context)


class ConfigurationError(InternalError):
    error_code = "CONFIGURATION_ERROR"

    def __init__(self, setting: str, reason: str) -> None:
        super().__init__(
            f"Configuration error for '{setting}': {reason}",
            context={"setting": setting, "reason": reason},
        )


ConfigurationException = ConfigurationError


# ══════════════════════════════════════════════════════════════════════════════
# Agent Errors
# ══════════════════════════════════════════════════════════════════════════════

class AgentError(CareerGPTError):
    http_status = 500
    error_code  = "AGENT_ERROR"

    def __init__(
        self,
        message: str = "Agent execution failed.",
        *,
        error_code: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, error_code=error_code, context=context)


AgentException = AgentError


class AgentTimeoutError(AgentError):
    error_code = "AGENT_TIMEOUT"


AgentTimeoutException = AgentTimeoutError


# Domain-specific agent errors
class ResumeTailorError(AgentError):
    error_code = "RESUME_TAILOR_ERROR"


class ResumeExportError(AgentError):
    error_code = "RESUME_EXPORT_ERROR"


class ApplicationSubmissionError(AgentError):
    error_code = "APPLICATION_SUBMISSION_ERROR"


class CoverLetterNotFoundError(NotFoundError):
    def __init__(self, context: dict | None = None) -> None:
        super().__init__("CoverLetter", context=context)