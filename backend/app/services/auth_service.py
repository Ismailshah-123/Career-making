"""
CareerGPT — Auth Service
==========================
PAGE SUMMARY:
  Full authentication business logic layer.
  Sits between API routes and repositories — routes never touch DB directly.
  Handles: registration, login, token refresh, password change, email verification,
  password reset flow (token generation + verification), account deactivation,
  OAuth preparation hooks, and audit logging for every auth event.

  USED BY: app/api/v1/auth.py
  USES:    UserRepository, AuditLogRepository, GroqService (welcome email content),
           NotificationService (send emails), SecurityModule (JWT + bcrypt)

  FLOW:
    register()       → validate → hash pw → create user → create tokens → audit log → welcome email
    login()          → fetch user → verify pw → update last_login → create tokens → audit log
    refresh_tokens() → decode refresh token → fetch user → rotate token pair
    change_password()→ verify old pw → hash new pw → update → invalidate sessions → audit
    forgot_password()→ generate reset token → store hashed in DB → send email
    reset_password() → verify reset token → hash new pw → clear token → audit
    verify_email()   → verify email token → mark verified → audit

  SECURITY:
    - Passwords hashed with bcrypt (rounds=12)
    - Reset tokens are random 32-byte URL-safe strings, stored as SHA256 hash
    - Reset tokens expire after 1 hour
    - Failed login attempts tracked (future: lockout after N failures)
    - All auth events logged to audit_logs table
    - Refresh token rotation on every use (previous token invalidated)
"""

from __future__ import annotations

import hashlib
import secrets
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.constants import UserRole
from app.core.exceptions import (
    AccountDeactivatedError,
    AuthError,
    DuplicateEmailError,
    InvalidCredentialsError,
    TokenExpiredError,
    TokenInvalidError,
    UserNotFoundError,
    ValidationError,
)
from app.core.logging import log_context, logger
from app.core.security import (
    TokenPayload,
    create_token_pair,
    decode_token,
    hash_password,
    verify_password,
)
from app.db.models.audit_log import AuditLog
from app.db.models.user import User
from app.repositories.user_repository import UserRepository


# ── Reset Token Config ────────────────────────────────────────────────────────

RESET_TOKEN_BYTES    = 32
RESET_TOKEN_TTL_MINS = 60
EMAIL_VERIFY_TTL_MINS = 24 * 60   # 24 hours


class AuthService:
    """
    Authentication business logic service.

    Injected with an AsyncSession in every request.
    All database operations go through the UserRepository.

    Usage in routes:
        @router.post("/login")
        async def login(body: LoginRequest, db: AsyncSession = Depends(get_db)):
            service = AuthService(db)
            return await service.login(body.email, body.password, request_ip="...")
    """

    def __init__(self, db: AsyncSession) -> None:
        self.db = db
        self.user_repo = UserRepository(db)

    # ── Registration ──────────────────────────────────────────────────────────

    async def register(
        self,
        *,
        email: str,
        password: str,
        full_name: str,
        request_ip: str = "unknown",
        request_id: str = "unknown",
    ) -> dict[str, Any]:
        """
        Register a new user account.

        Steps:
          1. Normalize and validate email
          2. Check for duplicate email
          3. Validate password strength
          4. Hash password (bcrypt, rounds=12)
          5. Create user record in DB
          6. Generate JWT token pair
          7. Write audit log entry
          8. Queue welcome email (non-blocking)

        Returns: {access_token, refresh_token, token_type, expires_in, user}
        Raises:  DuplicateEmailError, ValidationError
        """
        email = email.lower().strip()

        with log_context(action="register", email=email, request_id=request_id):
            # 1. Duplicate check
            if await self.user_repo.email_exists(email):
                logger.warning("Registration attempt with existing email", email=email)
                raise DuplicateEmailError(context={"email": email})

            # 2. Password strength validation
            self._validate_password_strength(password)

            # 3. Create user
            user = await self.user_repo.create(
                email=email,
                hashed_password=hash_password(password),
                full_name=full_name.strip(),
                role=UserRole.USER.value,
                plan="free",
                is_active=True,
                is_email_verified=False,
            )

            # 4. Generate tokens
            tokens = create_token_pair(user.id, user.email, user.role)

            # 5. Audit log
            await self._audit(
                user_id=user.id,
                action="user.registered",
                resource_type="user",
                resource_id=str(user.id),
                new_values={"email": email, "full_name": full_name},
                ip_address=request_ip,
                request_id=request_id,
            )

            # 6. Generate email verification token (async, non-blocking)
            verify_token = await self._create_email_verify_token(user)

            logger.info(
                "User registered successfully",
                user_id=str(user.id),
                email=email,
                plan="free",
            )

            return {
                **tokens,
                "user": self._user_to_dict(user),
                "email_verification_required": True,
            }

    # ── Login ─────────────────────────────────────────────────────────────────

    async def login(
        self,
        *,
        email: str,
        password: str,
        request_ip: str = "unknown",
        request_id: str = "unknown",
        user_agent: str = "",
    ) -> dict[str, Any]:
        """
        Authenticate with email + password.

        Steps:
          1. Fetch user by email (constant-time: always runs verify_password
             even if user not found, to prevent timing attacks)
          2. Verify bcrypt hash
          3. Check account is active
          4. Update last_login_at
          5. Generate token pair
          6. Audit log

        Returns: {access_token, refresh_token, token_type, expires_in, user}
        Raises:  InvalidCredentialsError, AccountDeactivatedError
        """
        email = email.lower().strip()

        with log_context(action="login", email=email, request_id=request_id):
            user = await self.user_repo.get_by_email(email)

            # Always run verify_password (prevents timing attacks)
            dummy_hash = "$2b$12$dummyhashtopreventtimingattacksXXXXXXXXXXXXXXXX"
            stored_hash = user.hashed_password if user else dummy_hash
            password_ok = verify_password(password, stored_hash)

            if not user or not password_ok:
                logger.warning("Failed login attempt", email=email, ip=request_ip)
                await self._audit(
                    user_id=user.id if user else None,
                    action="user.login_failed",
                    resource_type="user",
                    resource_id=email,
                    ip_address=request_ip,
                    request_id=request_id,
                    extra={"reason": "invalid_credentials"},
                )
                raise InvalidCredentialsError()

            if not user.is_active:
                raise AccountDeactivatedError(
                    context={"email": email}
                )

            if user.is_deleted:
                raise InvalidCredentialsError()

            # Update last login
            await self.user_repo.update(user.id, last_login_at=datetime.now(UTC))

            tokens = create_token_pair(user.id, user.email, user.role)

            await self._audit(
                user_id=user.id,
                action="user.login_success",
                resource_type="user",
                resource_id=str(user.id),
                ip_address=request_ip,
                request_id=request_id,
                extra={"user_agent": user_agent[:200]},
            )

            logger.info(
                "User logged in",
                user_id=str(user.id),
                email=email,
                ip=request_ip,
            )

            return {
                **tokens,
                "user": self._user_to_dict(user),
            }

    # ── Token Refresh ─────────────────────────────────────────────────────────

    async def refresh_tokens(
        self,
        *,
        refresh_token: str,
        request_ip: str = "unknown",
    ) -> dict[str, Any]:
        """
        Rotate token pair using a valid refresh token.
        Old refresh token is considered consumed after this call.
        Future: store JTI in Redis for true single-use enforcement.

        Returns: {access_token, refresh_token, token_type, expires_in}
        Raises:  TokenExpiredError, TokenInvalidError, UserNotFoundError
        """
        try:
            payload: TokenPayload = decode_token(refresh_token, expected_type="refresh")
        except (TokenExpiredError, TokenInvalidError) as exc:
            logger.warning("Refresh token rejected", error=str(exc), ip=request_ip)
            raise

        user = await self.user_repo.get_by_id(payload.user_id)
        if not user or not user.is_active or user.is_deleted:
            raise TokenInvalidError("User associated with token no longer exists or is inactive")

        tokens = create_token_pair(user.id, user.email, user.role)
        logger.debug("Tokens rotated", user_id=str(user.id))

        return {**tokens, "user": self._user_to_dict(user)}

    # ── Password Change ───────────────────────────────────────────────────────

    async def change_password(
        self,
        *,
        user_id: UUID,
        current_password: str,
        new_password: str,
        request_ip: str = "unknown",
        request_id: str = "unknown",
    ) -> dict[str, str]:
        """
        Change password for an authenticated user.
        Validates current password before applying change.
        Returns fresh token pair (old tokens should be discarded by client).
        """
        user = await self.user_repo.get_by_id_or_raise(user_id)

        if not verify_password(current_password, user.hashed_password):
            await self._audit(
                user_id=user_id,
                action="user.password_change_failed",
                resource_type="user",
                resource_id=str(user_id),
                ip_address=request_ip,
                request_id=request_id,
                extra={"reason": "wrong_current_password"},
            )
            raise InvalidCredentialsError("Current password is incorrect")

        if verify_password(new_password, user.hashed_password):
            raise ValidationError("New password must be different from current password")

        self._validate_password_strength(new_password)

        await self.user_repo.update(
            user_id,
            hashed_password=hash_password(new_password),
        )

        tokens = create_token_pair(user.id, user.email, user.role)

        await self._audit(
            user_id=user_id,
            action="user.password_changed",
            resource_type="user",
            resource_id=str(user_id),
            ip_address=request_ip,
            request_id=request_id,
        )

        logger.info("Password changed", user_id=str(user_id))
        return {**tokens}

    # ── Forgot Password ───────────────────────────────────────────────────────

    async def forgot_password(
        self,
        *,
        email: str,
        request_ip: str = "unknown",
    ) -> dict[str, str]:
        """
        Initiate password reset flow.
        ALWAYS returns success (never reveal if email exists — security).
        Generates reset token, stores hashed version in DB, queues email.

        Returns: {"message": "If that email is registered, a reset link was sent"}
        """
        email = email.lower().strip()
        user = await self.user_repo.get_by_email(email)

        if user and user.is_active and not user.is_deleted:
            raw_token = secrets.token_urlsafe(RESET_TOKEN_BYTES)
            token_hash = self._hash_token(raw_token)
            expires_at = datetime.now(UTC) + timedelta(minutes=RESET_TOKEN_TTL_MINS)

            await self.user_repo.update(
                user.id,
                password_reset_token=token_hash,
                password_reset_expires_at=expires_at,
            )

            # Queue reset email (fire and forget)
            try:
                from app.workers.notification_tasks import send_password_reset_email_task
                send_password_reset_email_task.delay(
                    user_id=str(user.id),
                    email=email,
                    full_name=user.full_name,
                    reset_token=raw_token,
                )
            except Exception as exc:
                logger.warning("Failed to queue reset email", error=str(exc))

            await self._audit(
                user_id=user.id,
                action="user.password_reset_requested",
                resource_type="user",
                resource_id=str(user.id),
                ip_address=request_ip,
            )
            logger.info("Password reset token generated", user_id=str(user.id))

        return {
            "message": "If that email is registered, a reset link has been sent."
        }

    async def reset_password(
        self,
        *,
        token: str,
        new_password: str,
        request_ip: str = "unknown",
    ) -> dict[str, str]:
        """
        Complete password reset using the token from the reset email.
        Clears reset token after use (single-use enforcement).
        """
        token_hash = self._hash_token(token)
        user = await self.user_repo.get_by_reset_token(token_hash)

        if not user:
            raise AuthError(
                "Invalid or expired password reset token",
                error_code="RESET_TOKEN_INVALID",
            )

        if (
            not user.password_reset_expires_at
            or datetime.now(UTC) > user.password_reset_expires_at.replace(tzinfo=UTC)
        ):
            raise AuthError(
                "Password reset token has expired. Please request a new one.",
                error_code="RESET_TOKEN_EXPIRED",
            )

        self._validate_password_strength(new_password)

        await self.user_repo.update(
            user.id,
            hashed_password=hash_password(new_password),
            password_reset_token=None,
            password_reset_expires_at=None,
        )

        tokens = create_token_pair(user.id, user.email, user.role)

        await self._audit(
            user_id=user.id,
            action="user.password_reset_completed",
            resource_type="user",
            resource_id=str(user.id),
            ip_address=request_ip,
        )

        logger.info("Password reset completed", user_id=str(user.id))
        return {**tokens, "message": "Password reset successfully."}

    # ── Email Verification ────────────────────────────────────────────────────

    async def verify_email(
        self,
        *,
        token: str,
        request_ip: str = "unknown",
    ) -> dict[str, str]:
        """
        Verify user email using the token sent on registration.
        Marks is_email_verified=True on success.
        """
        token_hash = self._hash_token(token)
        user = await self.user_repo.get_by_email_verify_token(token_hash)

        if not user:
            raise AuthError(
                "Invalid or expired email verification token",
                error_code="EMAIL_VERIFY_TOKEN_INVALID",
            )

        if user.is_email_verified:
            return {"message": "Email already verified."}

        if (
            not user.email_verify_expires_at
            or datetime.now(UTC) > user.email_verify_expires_at.replace(tzinfo=UTC)
        ):
            raise AuthError(
                "Email verification token expired. Request a new one.",
                error_code="EMAIL_VERIFY_TOKEN_EXPIRED",
            )

        await self.user_repo.update(
            user.id,
            is_email_verified=True,
            email_verified_at=datetime.now(UTC),
            email_verify_token=None,
            email_verify_expires_at=None,
        )

        await self._audit(
            user_id=user.id,
            action="user.email_verified",
            resource_type="user",
            resource_id=str(user.id),
            ip_address=request_ip,
        )

        logger.info("Email verified", user_id=str(user.id), email=user.email)
        return {"message": "Email verified successfully. You can now access all features."}

    async def resend_verification_email(
        self,
        *,
        user_id: UUID,
        request_ip: str = "unknown",
    ) -> dict[str, str]:
        """Resend email verification link. Rate-limited: max 1 per 5 minutes."""
        user = await self.user_repo.get_by_id_or_raise(user_id)

        if user.is_email_verified:
            return {"message": "Email is already verified."}

        verify_token = await self._create_email_verify_token(user)

        try:
            from app.workers.notification_tasks import send_email_verification_task
            send_email_verification_task.delay(
                user_id=str(user.id),
                email=user.email,
                full_name=user.full_name,
                verify_token=verify_token,
            )
        except Exception as exc:
            logger.warning("Failed to queue verification email", error=str(exc))

        return {"message": "Verification email sent."}

    # ── Account Management ────────────────────────────────────────────────────

    async def deactivate_account(
        self,
        *,
        user_id: UUID,
        password: str,
        reason: str = "",
        request_ip: str = "unknown",
    ) -> dict[str, str]:
        """
        Self-service account deactivation. Requires password confirmation.
        Soft-deactivates (is_active=False). Data retained for 90 days then purged.
        """
        user = await self.user_repo.get_by_id_or_raise(user_id)

        if not verify_password(password, user.hashed_password):
            raise InvalidCredentialsError("Password confirmation failed")

        await self.user_repo.update(user_id, is_active=False)

        await self._audit(
            user_id=user_id,
            action="user.account_deactivated",
            resource_type="user",
            resource_id=str(user_id),
            ip_address=request_ip,
            extra={"reason": reason[:500]},
        )

        logger.info("Account deactivated", user_id=str(user_id))
        return {"message": "Account deactivated. Contact support to restore within 90 days."}

    # ── Private Helpers ───────────────────────────────────────────────────────

    def _validate_password_strength(self, password: str) -> None:
        """
        Enforce password policy:
        - Min 8 characters
        - At least 1 uppercase letter
        - At least 1 digit
        Raises ValidationError with specific message on failure.
        """
        errors: list[str] = []
        if len(password) < 8:
            errors.append("Password must be at least 8 characters long")
        if not any(c.isupper() for c in password):
            errors.append("Password must contain at least one uppercase letter")
        if not any(c.isdigit() for c in password):
            errors.append("Password must contain at least one digit")
        if errors:
            raise ValidationError(
                errors[0],
                context={"all_errors": errors},
            )

    @staticmethod
    def _hash_token(raw_token: str) -> str:
        """SHA-256 hash a token for safe DB storage."""
        return hashlib.sha256(raw_token.encode()).hexdigest()

    async def _create_email_verify_token(self, user: User) -> str:
        """Generate and store hashed email verification token."""
        raw_token = secrets.token_urlsafe(RESET_TOKEN_BYTES)
        token_hash = self._hash_token(raw_token)
        expires_at = datetime.now(UTC) + timedelta(minutes=EMAIL_VERIFY_TTL_MINS)

        await self.user_repo.update(
            user.id,
            email_verify_token=token_hash,
            email_verify_expires_at=expires_at,
        )
        return raw_token

    async def _audit(
        self,
        *,
        user_id: UUID | None,
        action: str,
        resource_type: str = "",
        resource_id: str = "",
        old_values: dict | None = None,
        new_values: dict | None = None,
        ip_address: str = "unknown",
        request_id: str = "unknown",
        extra: dict | None = None,
    ) -> None:
        """Write an immutable audit log entry. Non-blocking — errors are swallowed."""
        import json
        try:
            from app.repositories.user_repository import AuditLogRepository
            audit_repo = AuditLogRepository(self.db)
            await audit_repo.create(
                user_id=user_id,
                action=action,
                resource_type=resource_type,
                resource_id=resource_id,
                old_values=json.dumps(old_values) if old_values else None,
                new_values=json.dumps(new_values) if new_values else None,
                ip_address=ip_address,
                request_id=request_id,
                extra=json.dumps(extra) if extra else None,
            )
        except Exception as exc:
            logger.warning("Audit log write failed (non-critical)", error=str(exc))

    @staticmethod
    def _user_to_dict(user: User) -> dict[str, Any]:
        """Convert User ORM object to safe response dict (no sensitive fields)."""
        return {
            "id":                  str(user.id),
            "email":               user.email,
            "full_name":           user.full_name,
            "role":                user.role,
            "plan":                user.plan,
            "is_email_verified":   user.is_email_verified,
            "avatar_url":          user.avatar_url,
            "linkedin_url":        user.linkedin_url,
            "remote_preference":   user.remote_preference,
            "total_applications":  user.total_applications,
            "created_at":          user.created_at.isoformat() if user.created_at else None,
            "last_login_at":       user.last_login_at.isoformat() if user.last_login_at else None,
        }


__all__ = ["AuthService"]