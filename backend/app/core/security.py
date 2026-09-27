"""
CareerGPT — Security & Auth Utilities
========================================
JWT creation/verification, bcrypt password hashing,
API key generation, CSRF, and role checks.
Compatible with your original security.py + all agent imports.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

import bcrypt
from jose import JWTError, ExpiredSignatureError, jwt
from pydantic import BaseModel, Field

from app.core.constants import (
    ACCESS_TOKEN_EXPIRE_MINUTES,
    REFRESH_TOKEN_EXPIRE_DAYS,
    ALGORITHM,
    API_KEY_PREFIX,
    API_KEY_LENGTH,
    BCRYPT_ROUNDS,
    MIN_PASSWORD_LENGTH,
    MAX_PASSWORD_LENGTH,
)
from app.core.exceptions import (
    InvalidTokenError,
    TokenExpiredError,
    AuthenticationError,
)
from app.core.logging import get_logger

logger = get_logger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Token Payload Models
# ══════════════════════════════════════════════════════════════════════════════

class TokenPayload(BaseModel):
    sub:   str
    jti:   str
    type:  str = "access"
    exp:   int
    iat:   int
    plan:  str        = "free"
    roles: list[str]  = Field(default_factory=list)
    email: str | None = None

    @property
    def user_id(self) -> str:
        """Alias for `sub` — the subject/user id claim."""
        return self.sub


class RefreshTokenPayload(BaseModel):
    sub:    str
    jti:    str
    type:   str = "refresh"
    exp:    int
    iat:    int
    family: str

    @property
    def user_id(self) -> str:
        """Alias for `sub` — the subject/user id claim."""
        return self.sub


class APIKeyPayload(BaseModel):
    user_id:    str
    key_id:     str
    prefix:     str
    created_at: datetime
    is_active:  bool


# ══════════════════════════════════════════════════════════════════════════════
# Password Hashing
# ══════════════════════════════════════════════════════════════════════════════

def hash_password(plain_password: str) -> str:
    """
    Hash plain-text password with bcrypt.
    Returns the full hash string (includes salt + cost factor).
    """
    if not plain_password:
        raise ValueError("Password cannot be empty.")
    password_bytes = plain_password.encode("utf-8")
    if len(password_bytes) > 72:
        raise ValueError("Password exceeds 72-byte bcrypt limit.")
    salt   = bcrypt.gensalt(rounds=BCRYPT_ROUNDS)
    hashed = bcrypt.hashpw(password_bytes, salt)
    return hashed.decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Constant-time bcrypt verification.
    Returns False (never raises) on mismatch.
    """
    if not plain_password or not hashed_password:
        return False
    try:
        return bcrypt.checkpw(
            plain_password.encode("utf-8"),
            hashed_password.encode("utf-8"),
        )
    except Exception:
        return False


def validate_password_strength(password: str) -> None:
    """
    Enforce password policy. Raises ValueError with reason on failure.
    Policy: min 8 chars, at least 1 uppercase, at least 1 digit.
    """
    if not password:
        raise ValueError("Password is required")
    if len(password) < MIN_PASSWORD_LENGTH:
        raise ValueError(f"Password must be at least {MIN_PASSWORD_LENGTH} characters")
    if len(password) > MAX_PASSWORD_LENGTH:
        raise ValueError(f"Password must be under {MAX_PASSWORD_LENGTH} characters")
    if not any(c.isupper() for c in password):
        raise ValueError("Password must contain at least one uppercase letter")
    if not any(c.isdigit() for c in password):
        raise ValueError("Password must contain at least one digit")


def is_password_strong(password: str) -> tuple[bool, list[str]]:
    """
    Return (passed, list_of_violations).
    Non-raising version for UI feedback.
    """
    violations: list[str] = []
    if len(password) < MIN_PASSWORD_LENGTH:
        violations.append(f"Must be at least {MIN_PASSWORD_LENGTH} characters.")
    if not any(c.isupper() for c in password):
        violations.append("Must contain at least one uppercase letter.")
    if not any(c.islower() for c in password):
        violations.append("Must contain at least one lowercase letter.")
    if not any(c.isdigit() for c in password):
        violations.append("Must contain at least one digit.")
    if not any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
        violations.append("Must contain at least one special character.")
    return len(violations) == 0, violations


# ══════════════════════════════════════════════════════════════════════════════
# JWT Access Token
# ══════════════════════════════════════════════════════════════════════════════

def _get_secret() -> str:
    from app.core.config import get_settings
    return get_settings().JWT_SECRET_KEY


def create_access_token(
    user_id: str,
    *,
    email: str | None = None,
    plan: str = "free",
    roles: list[str] | None = None,
    expires_delta: timedelta | None = None,
    extra_claims: dict[str, Any] | None = None,
) -> str:
    """
    Create a signed JWT access token.
    Returns signed JWT string.
    """
    now    = datetime.now(timezone.utc)
    expire = now + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    jti    = str(uuid.uuid4())

    payload: dict[str, Any] = {
        "sub":   user_id,
        "jti":   jti,
        "type":  "access",
        "exp":   int(expire.timestamp()),
        "iat":   int(now.timestamp()),
        "nbf":   int(now.timestamp()),
        "plan":  plan,
        "roles": roles or [],
    }
    if email:
        payload["email"] = email
    if extra_claims:
        payload.update(extra_claims)

    token = jwt.encode(payload, _get_secret(), algorithm=ALGORITHM)
    logger.debug("Access token created", user_id=user_id, jti=jti)
    return token


def verify_access_token(token: str) -> TokenPayload:
    """
    Decode and validate a JWT access token.
    Raises TokenExpiredError or InvalidTokenError on failure.
    """
    try:
        payload = jwt.decode(
            token,
            _get_secret(),
            algorithms=[ALGORITHM],
            options={"verify_exp": True, "verify_nbf": True},
        )
    except ExpiredSignatureError:
        raise TokenExpiredError()
    except JWTError as exc:
        raise InvalidTokenError(str(exc))

    if payload.get("type") != "access":
        raise InvalidTokenError(f"Expected access token, got '{payload.get('type')}'")

    try:
        return TokenPayload(**payload)
    except Exception as exc:
        raise InvalidTokenError(f"Malformed token payload: {exc}")


# ══════════════════════════════════════════════════════════════════════════════
# JWT Refresh Token
# ══════════════════════════════════════════════════════════════════════════════

def create_refresh_token(
    user_id: str,
    *,
    family: str | None = None,
    expires_delta: timedelta | None = None,
) -> tuple[str, str]:
    """
    Create a signed JWT refresh token.
    Returns (token_string, jti).
    """
    now          = datetime.now(timezone.utc)
    expire       = now + (expires_delta or timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS))
    jti          = str(uuid.uuid4())
    token_family = family or str(uuid.uuid4())

    payload: dict[str, Any] = {
        "sub":    user_id,
        "jti":    jti,
        "type":   "refresh",
        "exp":    int(expire.timestamp()),
        "iat":    int(now.timestamp()),
        "family": token_family,
    }

    token = jwt.encode(payload, _get_secret(), algorithm=ALGORITHM)
    return token, jti


def verify_refresh_token(token: str) -> RefreshTokenPayload:
    """Decode and validate a JWT refresh token."""
    try:
        payload = jwt.decode(
            token,
            _get_secret(),
            algorithms=[ALGORITHM],
            options={"verify_exp": True},
        )
    except ExpiredSignatureError:
        raise TokenExpiredError()
    except JWTError as exc:
        raise InvalidTokenError(str(exc))

    if payload.get("type") != "refresh":
        raise InvalidTokenError("Not a refresh token.")

    try:
        return RefreshTokenPayload(**payload)
    except Exception as exc:
        raise InvalidTokenError(f"Malformed refresh token: {exc}")


# ══════════════════════════════════════════════════════════════════════════════
# Token Pair Helper (used by auth_service.py)
# ══════════════════════════════════════════════════════════════════════════════

def create_token_pair(
    user_id: Any,
    email: str | None = None,
    role: str | None = None,
) -> dict[str, Any]:
    """
    Create a matching access + refresh token pair for a freshly
    authenticated/registered user.

    Returns a dict shaped for direct use in a token response:
        {"access_token", "refresh_token", "token_type", "expires_in"}
    """
    uid = str(user_id)
    access_token = create_access_token(
        uid,
        email=email,
        roles=[role] if role else [],
    )
    refresh_token, _jti = create_refresh_token(uid)
    return {
        "access_token":  access_token,
        "refresh_token": refresh_token,
        "token_type":    "bearer",
        "expires_in":    ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    }


def decode_token(
    token: str,
    expected_type: str = "access",
) -> TokenPayload | RefreshTokenPayload:
    """
    Decode either an access or a refresh token, selected by
    `expected_type`. Both returned payload types expose `.user_id`.

    Raises TokenExpiredError or InvalidTokenError on failure — same
    exceptions raised by verify_access_token / verify_refresh_token.
    """
    if expected_type == "refresh":
        return verify_refresh_token(token)
    return verify_access_token(token)


# ══════════════════════════════════════════════════════════════════════════════
# API Key Generation
# ══════════════════════════════════════════════════════════════════════════════

def _hmac_key(plaintext_key: str) -> str:
    return hmac.new(
        _get_secret().encode("utf-8"),
        plaintext_key.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


# Alias — app/api/deps.py looks up an API key by hashing the presented
# plaintext key and comparing against the stored hash.
_hash_api_key = _hmac_key


def generate_api_key() -> tuple[str, str]:
    """
    Generate a new API key.
    Returns (plaintext_key, hashed_key).
    Show plaintext_key to user ONCE. Store hashed_key in DB.
    """
    raw       = secrets.token_urlsafe(API_KEY_LENGTH)
    plaintext = f"{API_KEY_PREFIX}{raw}"
    hashed    = _hmac_key(plaintext)
    return plaintext, hashed


def verify_api_key(plaintext_key: str, stored_hash: str) -> bool:
    """Constant-time API key verification."""
    if not plaintext_key.startswith(API_KEY_PREFIX):
        return False
    expected = _hmac_key(plaintext_key)
    return hmac.compare_digest(expected.encode(), stored_hash.encode())


def extract_api_key_from_header(authorization: str) -> str | None:
    """Extract raw API key from Authorization header. Returns None if not found."""
    if not authorization:
        return None
    for prefix in ("Bearer ", "ApiKey ", "bearer ", "apikey "):
        if authorization.startswith(prefix):
            candidate = authorization[len(prefix):].strip()
            if candidate.startswith(API_KEY_PREFIX):
                return candidate
    if authorization.startswith(API_KEY_PREFIX):
        return authorization.strip()
    return None


# ══════════════════════════════════════════════════════════════════════════════
# CSRF (Stateless double-submit cookie pattern)
# ══════════════════════════════════════════════════════════════════════════════

def generate_csrf_token(session_id: str) -> str:
    nonce  = secrets.token_hex(16)
    digest = hmac.new(
        _get_secret().encode(),
        f"{nonce}:{session_id}".encode(),
        hashlib.sha256,
    ).hexdigest()
    return f"{nonce}:{digest}"


def verify_csrf_token(token: str, session_id: str) -> bool:
    try:
        nonce, digest = token.split(":", 1)
    except ValueError:
        return False
    expected = hmac.new(
        _get_secret().encode(),
        f"{nonce}:{session_id}".encode(),
        hashlib.sha256,
    ).hexdigest()
    return hmac.compare_digest(expected.encode(), digest.encode())


# ══════════════════════════════════════════════════════════════════════════════
# Role / Permission Checks
# ══════════════════════════════════════════════════════════════════════════════

ROLE_WEIGHTS: dict[str, int] = {
    "guest":     0,
    "user":      10,
    "premium":   20,
    "moderator": 30,
    "admin":     50,
    "superuser": 100,
}


def has_role(token_payload: TokenPayload, required_role: str) -> bool:
    max_weight = max(
        (ROLE_WEIGHTS.get(r, 0) for r in token_payload.roles),
        default=0,
    )
    return max_weight >= ROLE_WEIGHTS.get(required_role, 0)


def is_superuser(token_payload: TokenPayload) -> bool:
    return "superuser" in token_payload.roles


def is_admin_or_above(token_payload: TokenPayload) -> bool:
    return has_role(token_payload, "admin")


def owns_resource(token_payload: TokenPayload, resource_owner_id: str) -> bool:
    return token_payload.sub == resource_owner_id or is_admin_or_above(token_payload)


# ══════════════════════════════════════════════════════════════════════════════
# Utilities
# ══════════════════════════════════════════════════════════════════════════════

def generate_secure_token(nbytes: int = 32) -> str:
    return secrets.token_urlsafe(nbytes)


def generate_numeric_otp(length: int = 6) -> str:
    return "".join(str(secrets.randbelow(10)) for _ in range(length))


def mask_email(email: str) -> str:
    try:
        local, domain = email.split("@", 1)
        masked = local[0] + "***" if len(local) > 1 else "***"
        return f"{masked}@{domain}"
    except ValueError:
        return "***"


def mask_token(token: str, visible_chars: int = 8) -> str:
    return token[:visible_chars] + "***" if len(token) > visible_chars else "***"


def constant_time_compare(val1: str, val2: str) -> bool:
    return hmac.compare_digest(val1.encode("utf-8"), val2.encode("utf-8"))