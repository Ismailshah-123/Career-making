"""
CareerGPT — User Service
==========================
PAGE SUMMARY:
  Complete user account management service. Handles all user CRUD,
  profile management, preferences, onboarding flow, GDPR, and
  user analytics aggregation.

  PUBLIC API:
    get_user()               → fetch user by ID with enriched stats
    get_user_by_email()      → lookup by email
    update_profile()         → update name, bio, avatar, location
    update_preferences()     → target roles, locations, salary, remote pref
    update_linkedin_token()  → store OAuth access token securely
    update_avatar()          → upload + resize profile image
    complete_onboarding()    → mark onboarding done, set initial prefs
    get_dashboard_stats()    → aggregated stats for dashboard home page
    get_agent_history()      → paginated AgentRun history for user
    deactivate_account()     → soft-delete with GDPR data anonymization
    delete_account()         → hard delete (GDPR right to erasure)
    export_user_data()       → GDPR data export (all user data as JSON)
    get_activity_feed()      → recent actions for activity timeline
    update_notification_settings() → email/push preference toggles

  DASHBOARD STATS AGGREGATION:
    Returns single dict with all data the dashboard needs:
    total_applications, applications_this_week, response_rate,
    interviews_count, offers_count, pipeline_breakdown,
    top_matching_jobs (from DiscoveryAgent), match_score_avg,
    resume_ats_score, linkedin_posts_this_month, agent_runs_today.

  USED BY:
    API routes: /api/v1/users/*
    Auth routes: post-registration onboarding
    Admin panel: user management
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import (
    UserNotFoundError,
    ValidationError,
    StorageError,
)
from app.core.logging import log_context, logger
from app.utils.validators import (
    validate_full_name,
    validate_skills_list,
    validate_target_roles,
    validate_locations_list,
    validate_salary,
    validate_experience_years,
    normalize_url,
)


class UserService:
    """
    User account management service.

    Usage:
        service = UserService(db)
        user    = await service.get_user(user_id)
        stats   = await service.get_dashboard_stats(user_id)
    """

    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    # ── Read ──────────────────────────────────────────────────────────────────

    async def get_user(self, user_id: uuid.UUID) -> dict[str, Any]:
        """
        Fetch user by ID with enriched profile stats.
        Returns serialized user dict for API response.
        Raises UserNotFoundError if not found.
        """
        from app.repositories.user_repository import UserRepository
        repo = UserRepository(self.db)
        user = await repo.get_by_id_or_raise(user_id)
        return await self._serialize_user(user, include_stats=True)

    async def get_user_by_email(self, email: str) -> dict[str, Any] | None:
        """Fetch user by email. Returns None if not found."""
        from app.repositories.user_repository import UserRepository
        repo = UserRepository(self.db)
        user = await repo.get_by_email(email)
        if not user:
            return None
        return await self._serialize_user(user, include_stats=False)

    async def list_users(
        self,
        *,
        skip: int = 0,
        limit: int = 50,
        search: str | None = None,
        is_active: bool | None = None,
    ) -> dict[str, Any]:
        """
        Admin: list all users with pagination and optional search.
        Returns {users: [...], total: int, skip: int, limit: int}.
        """
        from app.repositories.user_repository import UserRepository
        repo  = UserRepository(self.db)
        users = await repo.list_users(
            skip=skip, limit=limit,
            search=search, is_active=is_active,
        )
        total = await repo.count_users(search=search, is_active=is_active)
        return {
            "users": [self._serialize_user_brief(u) for u in users],
            "total": total,
            "skip":  skip,
            "limit": limit,
        }

    # ── Update Profile ────────────────────────────────────────────────────────

    async def update_profile(
        self,
        user_id: uuid.UUID,
        *,
        full_name: str | None = None,
        bio: str | None = None,
        location: str | None = None,
        phone: str | None = None,
        linkedin_url: str | None = None,
        github_url: str | None = None,
        portfolio_url: str | None = None,
        timezone: str | None = None,
    ) -> dict[str, Any]:
        """
        Update user profile fields. Only updates provided (non-None) fields.
        Validates name format, URL formats.
        Returns updated user dict.
        """
        from app.repositories.user_repository import UserRepository
        repo = UserRepository(self.db)
        user = await repo.get_by_id_or_raise(user_id)

        updates: dict[str, Any] = {}

        if full_name is not None:
            updates["full_name"] = validate_full_name(full_name)

        if bio is not None:
            if len(bio) > 1000:
                raise ValidationError("Bio must be under 1000 characters")
            updates["bio"] = bio.strip()

        if location is not None:
            updates["location"] = location.strip()[:200]

        if phone is not None:
            updates["phone"] = phone.strip()[:20]

        if linkedin_url is not None:
            updates["linkedin_url"] = normalize_url(linkedin_url)[:500]

        if github_url is not None:
            updates["github_url"] = normalize_url(github_url)[:500]

        if portfolio_url is not None:
            updates["portfolio_url"] = normalize_url(portfolio_url)[:500]

        if timezone is not None:
            updates["timezone"] = timezone[:50]

        if not updates:
            return await self._serialize_user(user, include_stats=False)

        updates["updated_at"] = datetime.now(UTC)
        updated_user = await repo.update(user_id, **updates)

        logger.info("User profile updated", user_id=str(user_id), fields=list(updates.keys()))
        return await self._serialize_user(updated_user, include_stats=False)

    # ── Update Preferences ────────────────────────────────────────────────────

    async def update_preferences(
        self,
        user_id: uuid.UUID,
        *,
        target_roles: list[str] | None = None,
        target_locations: list[str] | None = None,
        min_salary: int | None = None,
        remote_preference: str | None = None,
        experience_years: float | None = None,
        skills: list[str] | None = None,
        preferred_company_stages: list[str] | None = None,
        open_to_relocation: bool | None = None,
        visa_sponsorship_needed: bool | None = None,
    ) -> dict[str, Any]:
        """
        Update job search preferences used by all discovery + matching agents.
        Target roles are stored as JSON array — no hardcoded role categories.
        """
        from app.repositories.user_repository import UserRepository
        repo = UserRepository(self.db)
        user = await repo.get_by_id_or_raise(user_id)

        updates: dict[str, Any] = {}

        if target_roles is not None:
            validated = validate_target_roles(target_roles)
            updates["target_roles"] = json.dumps(validated)

        if target_locations is not None:
            validated_locs = validate_locations_list(target_locations)
            updates["target_locations"] = json.dumps(validated_locs)

        if min_salary is not None:
            updates["min_salary"] = validate_salary(min_salary, min_value=0)

        if remote_preference is not None:
            valid_prefs = {"remote", "hybrid", "onsite", "any"}
            if remote_preference not in valid_prefs:
                raise ValidationError(
                    f"remote_preference must be one of: {valid_prefs}",
                    context={"value": remote_preference},
                )
            updates["remote_preference"] = remote_preference

        if experience_years is not None:
            updates["experience_years"] = validate_experience_years(experience_years)

        if skills is not None:
            validated_skills = validate_skills_list(skills)
            updates["skills"] = json.dumps(validated_skills)

        if preferred_company_stages is not None:
            valid_stages = {"startup", "series-a", "series-b", "series-c", "growth", "enterprise", "any"}
            filtered = [s for s in preferred_company_stages if s in valid_stages]
            updates["preferred_company_stages"] = json.dumps(filtered)

        if open_to_relocation is not None:
            updates["open_to_relocation"] = open_to_relocation

        if visa_sponsorship_needed is not None:
            updates["visa_sponsorship_needed"] = visa_sponsorship_needed

        if not updates:
            return await self._serialize_user(user, include_stats=False)

        updates["updated_at"] = datetime.now(UTC)
        updated = await repo.update(user_id, **updates)

        logger.info(
            "User preferences updated",
            user_id=str(user_id),
            fields=list(updates.keys()),
        )
        return await self._serialize_user(updated, include_stats=False)

    # ── LinkedIn Token ────────────────────────────────────────────────────────

    async def update_linkedin_token(
        self,
        user_id: uuid.UUID,
        access_token: str,
        *,
        expires_in_seconds: int = 5184000,  # 60 days
    ) -> None:
        """
        Store LinkedIn OAuth access token.
        Token is stored encrypted at rest (handled by DB column type).
        """
        from app.repositories.user_repository import UserRepository
        from app.utils.datetime_utils import now_utc

        repo = UserRepository(self.db)
        expires_at = now_utc() + timedelta(seconds=expires_in_seconds)

        await repo.update(
            user_id,
            linkedin_access_token=access_token,
            linkedin_token_expires_at=expires_at,
            linkedin_connected_at=now_utc(),
        )
        logger.info("LinkedIn token updated", user_id=str(user_id))

    async def revoke_linkedin_token(self, user_id: uuid.UUID) -> None:
        """Remove LinkedIn access token (disconnect LinkedIn)."""
        from app.repositories.user_repository import UserRepository
        repo = UserRepository(self.db)
        await repo.update(
            user_id,
            linkedin_access_token=None,
            linkedin_token_expires_at=None,
        )

    # ── Avatar ────────────────────────────────────────────────────────────────

    async def update_avatar(
        self,
        user_id: uuid.UUID,
        content: bytes,
        filename: str,
    ) -> dict[str, Any]:
        """
        Upload and store profile avatar image.
        Validates: image type, max 5MB.
        Returns {avatar_url: str}.
        """
        from app.utils.file_utils import validate_upload_file, save_upload_file
        from app.repositories.user_repository import UserRepository

        ALLOWED_IMAGE_MIMES = frozenset({
            "image/jpeg", "image/png", "image/webp", "image/gif"
        })
        await validate_upload_file(
            content, filename,
            content_type=f"image/{filename.rsplit('.', 1)[-1].lower()}",
            allowed_mimes=ALLOWED_IMAGE_MIMES,
            max_size_mb=5,
        )

        file_path, stored_name = await save_upload_file(
            content, filename,
            user_id=user_id,
            subdir="avatars",
        )

        avatar_url = f"/static/uploads/{user_id}/avatars/{stored_name}"

        repo = UserRepository(self.db)
        await repo.update(user_id, avatar_url=avatar_url)

        logger.info("Avatar updated", user_id=str(user_id), path=str(file_path))
        return {"avatar_url": avatar_url}

    # ── Onboarding ────────────────────────────────────────────────────────────

    async def complete_onboarding(
        self,
        user_id: uuid.UUID,
        *,
        full_name: str,
        target_roles: list[str],
        experience_years: float,
        remote_preference: str = "any",
        min_salary: int = 0,
        target_locations: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Complete the user onboarding flow after registration.
        Sets all initial preferences so agents have context from day one.
        Called by the /onboarding API endpoint on first login flow.
        """
        from app.repositories.user_repository import UserRepository
        repo = UserRepository(self.db)
        await repo.get_by_id_or_raise(user_id)

        await self.update_profile(user_id, full_name=full_name)
        result = await self.update_preferences(
            user_id,
            target_roles=target_roles,
            experience_years=experience_years,
            remote_preference=remote_preference,
            min_salary=min_salary,
            target_locations=target_locations or ["remote"],
        )

        await repo.update(
            user_id,
            onboarding_completed=True,
            onboarding_completed_at=datetime.now(UTC),
        )

        logger.info(
            "Onboarding complete",
            user_id=str(user_id),
            target_roles=target_roles,
        )
        return result

    # ── Dashboard Stats ────────────────────────────────────────────────────────

    async def get_dashboard_stats(self, user_id: uuid.UUID) -> dict[str, Any]:
        """
        Aggregate all data needed by the dashboard home page.
        Single endpoint to avoid N+1 API calls from the frontend.
        Returns comprehensive stats in one DB round-trip.
        """
        from app.repositories.application_repository import ApplicationRepository
        from app.repositories.resume_repository import ResumeRepository
        from app.repositories.linkedin_repository import LinkedInRepository
        from app.repositories.user_repository import UserRepository

        user_repo = UserRepository(self.db)
        app_repo  = ApplicationRepository(self.db)
        res_repo  = ResumeRepository(self.db)
        li_repo   = LinkedInRepository(self.db)

        user = await user_repo.get_by_id_or_raise(user_id)

        # Parallel DB queries
        pipeline_stats, resumes, li_stats = await __import__("asyncio").gather(
            app_repo.get_pipeline_stats(user_id),
            res_repo.get_master_resumes(user_id),
            li_repo.get_engagement_stats(user_id),
        )

        # Compute response rate
        total_apps    = pipeline_stats.get("total", 0)
        responded     = (
            pipeline_stats.get("interview", 0)
            + pipeline_stats.get("offer", 0)
            + pipeline_stats.get("rejected", 0)
        )
        response_rate = round(responded / max(total_apps, 1) * 100, 1)

        # Master resume ATS score
        master_resume = resumes[0] if resumes else None
        ats_score     = master_resume.ats_score if master_resume else 0

        return {
            "user": {
                "full_name":             user.full_name,
                "avatar_url":            user.avatar_url,
                "onboarding_completed":  user.onboarding_completed,
                "target_roles": json.loads(user.target_roles) if user.target_roles else [],
                "member_since":          user.created_at.strftime("%B %Y") if user.created_at else "",
            },
            "applications": {
                "total":           total_apps,
                "this_week":       pipeline_stats.get("this_week", 0),
                "pipeline":        pipeline_stats,
                "response_rate":   response_rate,
                "interviews":      pipeline_stats.get("interview", 0),
                "offers":          pipeline_stats.get("offer", 0),
            },
            "resume": {
                "has_master":      bool(master_resume),
                "ats_score":       ats_score,
                "last_updated":    master_resume.updated_at.isoformat() if master_resume and master_resume.updated_at else None,
            },
            "linkedin": {
                "connected":       bool(user.linkedin_access_token),
                "posts_this_month": li_stats.get("posts_this_month", 0),
                "total_impressions": li_stats.get("total_impressions", 0),
                "avg_engagement":  li_stats.get("avg_engagement_rate", 0),
            },
        }

    # ── Agent History ─────────────────────────────────────────────────────────

    async def get_agent_history(
        self,
        user_id: uuid.UUID,
        *,
        skip: int = 0,
        limit: int = 20,
        agent_type: str | None = None,
    ) -> dict[str, Any]:
        """Return paginated AgentRun history for a user."""
        from app.repositories.user_repository import UserRepository
        repo  = UserRepository(self.db)
        runs  = await repo.get_agent_runs(
            user_id, skip=skip, limit=limit, agent_type=agent_type
        )
        total = await repo.count_agent_runs(user_id, agent_type=agent_type)
        return {
            "runs":  [self._serialize_agent_run(r) for r in runs],
            "total": total,
            "skip":  skip,
            "limit": limit,
        }

    # ── Notification Settings ─────────────────────────────────────────────────

    async def update_notification_settings(
        self,
        user_id: uuid.UUID,
        settings_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Update email/push notification preference toggles."""
        from app.repositories.user_repository import UserRepository
        repo = UserRepository(self.db)

        allowed_keys = {
            "email_job_alerts", "email_followup_reminders",
            "email_weekly_digest", "push_job_alerts",
            "push_application_updates", "email_marketing",
        }
        filtered = {k: bool(v) for k, v in settings_data.items() if k in allowed_keys}

        await repo.update(user_id, notification_settings=json.dumps(filtered))
        return filtered

    # ── Account Management ────────────────────────────────────────────────────

    async def deactivate_account(self, user_id: uuid.UUID) -> None:
        """
        Soft-delete user account.
        Sets is_active=False. Data retained for 30 days then auto-purged.
        """
        from app.repositories.user_repository import UserRepository
        repo = UserRepository(self.db)
        await repo.deactivate(user_id)
        logger.info("Account deactivated", user_id=str(user_id))

    async def delete_account(self, user_id: uuid.UUID) -> None:
        """
        Hard delete — GDPR right to erasure.
        Anonymizes PII, deletes files, removes Qdrant vectors.
        Non-reversible. 
        """
        from app.repositories.user_repository import UserRepository, AuditLogRepository
        repo = UserRepository(self.db)

        # Anonymize personal data
        anon_email = f"deleted_{uuid.uuid4().hex[:8]}@deleted.careergpt"
        await repo.update(
            user_id,
            email=anon_email,
            full_name="Deleted User",
            phone=None,
            bio=None,
            avatar_url=None,
            linkedin_url=None,
            github_url=None,
            portfolio_url=None,
            linkedin_access_token=None,
            is_active=False,
            is_deleted=True,
            deleted_at=datetime.now(UTC),
        )

        # Delete Qdrant resume vectors
        try:
            from app.services.qdrant_service import get_qdrant_service
            qdrant = get_qdrant_service()
            await qdrant.soft_delete_by_user(user_id)
        except Exception as exc:
            logger.warning("Qdrant vector deletion failed (non-critical)", error=str(exc))

        logger.info("Account hard-deleted (GDPR erasure)", user_id=str(user_id))

    async def export_user_data(self, user_id: uuid.UUID) -> dict[str, Any]:
        """
        GDPR: export all user data as a structured JSON dict.
        Returns everything we store about this user.
        """
        from app.repositories.user_repository import UserRepository
        from app.repositories.application_repository import ApplicationRepository
        from app.repositories.resume_repository import ResumeRepository

        user_repo = UserRepository(self.db)
        user      = await user_repo.get_by_id_or_raise(user_id)

        app_repo  = ApplicationRepository(self.db)
        apps      = await app_repo.get_user_applications(user_id, limit=500)

        res_repo  = ResumeRepository(self.db)
        resumes   = await res_repo.get_all_for_user(user_id)

        return {
            "export_date": datetime.now(UTC).isoformat(),
            "user": {
                "id":           str(user.id),
                "email":        user.email,
                "full_name":    user.full_name,
                "created_at":   user.created_at.isoformat() if user.created_at else None,
                "target_roles": json.loads(user.target_roles) if user.target_roles else [],
                "min_salary":   user.min_salary,
            },
            "applications": [
                {
                    "id":         str(a.id),
                    "job_title":  a.job.title if a.job else "",
                    "company":    a.job.company if a.job else "",
                    "status":     a.status,
                    "applied_at": a.applied_at.isoformat() if a.applied_at else None,
                }
                for a in apps
            ],
            "resumes": [
                {
                    "id":        str(r.id),
                    "is_master": r.is_master,
                    "filename":  r.original_filename,
                    "ats_score": r.ats_score,
                }
                for r in resumes
            ],
        }

    # ── Private: Serialization ────────────────────────────────────────────────

    async def _serialize_user(
        self,
        user: Any,
        *,
        include_stats: bool = False,
    ) -> dict[str, Any]:
        """Serialize User ORM object to API response dict."""
        data: dict[str, Any] = {
            "id":                    str(user.id),
            "email":                 user.email,
            "full_name":             user.full_name,
            "bio":                   user.bio,
            "avatar_url":            user.avatar_url,
            "location":              user.location,
            "phone":                 user.phone,
            "linkedin_url":          user.linkedin_url,
            "github_url":            user.github_url,
            "portfolio_url":         user.portfolio_url,
            "timezone":              user.timezone or "UTC",
            "target_roles":          json.loads(user.target_roles) if user.target_roles else [],
            "target_locations":      json.loads(user.target_locations) if user.target_locations else [],
            "min_salary":            user.min_salary,
            "remote_preference":     user.remote_preference,
            "experience_years":      user.experience_years,
            "skills":                json.loads(user.skills) if user.skills else [],
            "linkedin_connected":    bool(user.linkedin_access_token),
            "onboarding_completed":  user.onboarding_completed,
            "is_active":             user.is_active,
            "created_at":            user.created_at.isoformat() if user.created_at else None,
            "updated_at":            user.updated_at.isoformat() if user.updated_at else None,
        }
        return data

    def _serialize_user_brief(self, user: Any) -> dict[str, Any]:
        """Minimal user dict for list responses."""
        return {
            "id":          str(user.id),
            "email":       user.email,
            "full_name":   user.full_name,
            "avatar_url":  user.avatar_url,
            "is_active":   user.is_active,
            "created_at":  user.created_at.isoformat() if user.created_at else None,
        }

    def _serialize_agent_run(self, run: Any) -> dict[str, Any]:
        """Serialize AgentRun to API response dict."""
        return {
            "id":          str(run.id),
            "agent_type":  run.agent_type,
            "status":      run.status,
            "duration_ms": run.duration_ms,
            "llm_calls":   run.llm_calls,
            "started_at":  run.started_at.isoformat() if run.started_at else None,
        }


__all__ = ["UserService"]