"""
CareerGPT — LinkedIn Agent
============================
PAGE SUMMARY:
  The personal branding agent. Runs the full LinkedIn content pipeline:
  topic research → post writing → image generation → quality scoring
  → hook improvement → carousel creation → scheduling → publishing.

  AGENT MODES:
    generate_post()         → full AI post with optional image (main flow)
    generate_with_image()   → post + AI-generated visual (Pollinations / SVG)
    generate_carousel()     → multi-slide carousel PDF post
    publish_post()          → publish a draft post (API or Playwright)
    schedule_post()         → schedule for optimal posting time
    sync_engagement()       → pull latest metrics from LinkedIn
    batch_generate()        → create a week's worth of posts at once

  FULL GENERATE FLOW (generate_post):
    1. Research trending topic (if not provided)
    2. Check recent posts — avoid repeating topics from last 30 days
    3. Generate post text via GroqService
    4. Score quality (0-100) — reject if < 65
    5. If score < 75 → improve the hook automatically
    6. If image requested → generate image prompt → generate image
    7. Optimize hashtags (3-5 most relevant)
    8. Save LinkedInPost DB record (status=draft)
    9. If schedule_now=True → schedule for next optimal time
    10. Return full post dict with content + image path + schedule time

  QUALITY GATE:
    Minimum score to publish: 65/100
    If score < 75: auto-improve hook before saving
    If score < 65: regenerate with different format (max 2 retries)

  IMAGE GENERATION:
    Default: ON for posts (users can disable per request)
    Provider order: Pollinations.ai (free) → Together.ai → SVG fallback
    Image saved locally → uploaded to LinkedIn Media API on publish
    Supports: 1200x627 feed images + 1080x1080 carousel slides

  GROQ MODEL SELECTION:
    Post writing:    llama-3.3-70b-versatile (creative, high quality)
    Topic research:  llama-3.3-70b-versatile (knowledge + creativity)
    Quality scoring: llama-3.1-8b-instant    (fast, deterministic)
    Hook improvement: llama-3.3-70b-versatile (creative rewrite)

  USED BY:
    app/api/v1/linkedin.py → all LinkedIn endpoints
    app/workers/linkedin_tasks.py → scheduled daily post generation
"""

from __future__ import annotations

import json
import time
import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.constants import AgentType, LinkedInPostStatus, LinkedInPostTone
from app.core.exceptions import AgentError
from app.core.logging import log_context, logger
from app.agents.linkedin_agent import tools


class LinkedInAgent:
    """
    LinkedIn personal branding agent.
    Orchestrates the complete content creation and publishing pipeline.

    Usage:
        agent = LinkedInAgent(db)
        result = await agent.generate_post(
            user_id=user.id,
            tone="thought_leader",
            with_image=True,
        )
    """

    def __init__(self, db: AsyncSession) -> None:
        self.db = db
        self._llm_calls = 0

    # ── Mode 1: Generate Post (main flow) ─────────────────────────────────────

    async def generate_post(
        self,
        user_id: uuid.UUID,
        *,
        topic: str | None = None,
        tone: str = LinkedInPostTone.THOUGHT_LEADER.value,
        post_format: str = "insight",
        with_image: bool = True,
        schedule_now: bool = False,
        schedule_at: datetime | None = None,
        custom_hook: str = "",
        include_cta: bool = True,
    ) -> dict[str, Any]:
        """
        Full LinkedIn post generation pipeline.
        Returns complete post data ready for preview, editing, and publishing.
        """
        t_start = time.monotonic()

        with log_context(agent="linkedin", user_id=str(user_id)):
            logger.info("Starting LinkedIn post generation", tone=tone, with_image=with_image)

            # ── 1. Load user context ──────────────────────────────────────────
            from app.repositories.user_repository import UserRepository
            from app.repositories.resume_repository import ResumeRepository

            user_repo   = UserRepository(self.db)
            resume_repo = ResumeRepository(self.db)

            user   = await user_repo.get_by_id_or_raise(user_id)
            master = await resume_repo.get_latest_master(user_id)

            user_skills: list[str] = []
            if master and master.skills:
                try:
                    user_skills = json.loads(master.skills)
                except Exception:
                    pass

            user_expertise = (
                f"{master.summary or ''} Skills: {', '.join(user_skills[:10])}"
                if master else "Software engineer and tech professional"
            )

            # ── 2. Get recent topics (avoid repetition) ───────────────────────
            recent_topics = await tools.get_recent_post_topics(self.db, user_id)

            # ── 3. Research topic if not provided ─────────────────────────────
            topic_data: dict[str, Any] = {}
            if not topic:
                topic_data = await tools.research_trending_topic(
                    user_expertise=user_expertise,
                    tech_stack=user_skills[:8],
                    recent_post_topics=recent_topics,
                )
                self._llm_calls += 1
                topic = topic_data.get("recommended_topic", "AI trends in software engineering")

            key_points: list[str] = topic_data.get("key_points", [])
            cta_question: str = topic_data.get("cta_question", "What has been your experience with this?")
            hook_ideas: list[str] = topic_data.get("hook_ideas", [])

            # ── 4. Generate post text ─────────────────────────────────────────
            post_data = await tools.generate_post_text(
                topic=topic,
                key_points=key_points,
                tone=tone,
                post_format=post_format,
                user_expertise=user_expertise,
                include_hook=True,
                include_cta=include_cta,
                cta_question=cta_question,
                custom_hook=custom_hook or (hook_ideas[0] if hook_ideas else ""),
            )
            self._llm_calls += 1

            hook       = post_data.get("hook", "")
            body       = post_data.get("body", "")
            full_post  = post_data.get("full_post", f"{hook}\n\n{body}")
            hashtags   = post_data.get("hashtags", ["#SoftwareEngineering", "#Python"])
            char_count = post_data.get("character_count", len(full_post))

            # ── 5. Quality scoring ────────────────────────────────────────────
            quality_result = await tools.score_post_quality(full_post)
            self._llm_calls += 1
            quality_score  = quality_result.get("overall_score", 70)

            # ── 6. Improve hook if quality < 75 ───────────────────────────────
            retry_count = 0
            while quality_score < 75 and retry_count < 2:
                logger.info(
                    f"Post quality {quality_score} < 75, improving hook",
                    attempt=retry_count + 1,
                )
                if quality_score < 65:
                    # Regenerate entirely
                    post_data = await tools.generate_post_text(
                        topic=topic,
                        key_points=key_points,
                        tone=tone,
                        post_format="case_study" if post_format == "insight" else "list",
                        user_expertise=user_expertise,
                        include_cta=include_cta,
                        cta_question=cta_question,
                    )
                    self._llm_calls += 1
                    hook      = post_data.get("hook", hook)
                    body      = post_data.get("body", body)
                    full_post = post_data.get("full_post", f"{hook}\n\n{body}")
                else:
                    # Just improve the hook
                    full_post = await tools.improve_hook(full_post, topic)
                    self._llm_calls += 1
                    hook = full_post.split("\n")[0] if "\n" in full_post else full_post[:150]

                quality_result = await tools.score_post_quality(full_post)
                self._llm_calls += 1
                quality_score  = quality_result.get("overall_score", 70)
                retry_count   += 1

            logger.info(
                "Post quality gate passed",
                score=quality_score,
                grade=quality_result.get("grade", "B"),
                retries=retry_count,
            )

            # ── 7. Image generation ───────────────────────────────────────────
            image_path:   str | None = None
            image_prompt: str | None = None

            if with_image:
                img_prompt_data = await tools.generate_image_prompt(
                    post_topic=topic,
                    post_hook=hook,
                    key_message=body[:200] if body else topic,
                    visual_style="professional",
                )
                self._llm_calls += 1
                image_prompt = img_prompt_data.get("image_prompt", "")

                if image_prompt:
                    image_path = await tools.generate_post_image(
                        image_prompt,
                        width=1200,
                        height=627,
                        user_id=user_id,
                    )
                    logger.info(
                        "Post image generated",
                        path=image_path,
                        method="pollinations" if image_path and "png" in image_path else "svg",
                    )

            # ── 8. Hashtag optimization ───────────────────────────────────────
            from app.services.groq_service import get_groq_service
            from app.prompts.linkedin_prompts import LINKEDIN_HASHTAG_SELECT

            llm = get_groq_service()
            _, htag_msg = LINKEDIN_HASHTAG_SELECT.render(
                post_topic=topic,
                content_preview=hook,
                target_audience="software engineers, developers, tech professionals",
                tech_mentioned=", ".join(user_skills[:5]),
            )
            try:
                htag_result = await llm.complete_json(
                    prompt=htag_msg,
                    system=LINKEDIN_HASHTAG_SELECT.system,
                    temperature=LINKEDIN_HASHTAG_SELECT.temperature,
                    max_tokens=LINKEDIN_HASHTAG_SELECT.max_tokens,
                )
                self._llm_calls += 1
                optimized_hashtags = htag_result.get("recommended_hashtags", hashtags)
            except Exception:
                optimized_hashtags = hashtags

            # ── 9. Determine schedule time ────────────────────────────────────
            final_schedule_at: datetime | None = None
            if schedule_at:
                final_schedule_at = schedule_at
            elif schedule_now:
                final_schedule_at = tools.get_optimal_post_time()

            # ── 10. Save to DB ────────────────────────────────────────────────
            from app.repositories.linkedin_repository import LinkedInRepository

            li_repo = LinkedInRepository(self.db)
            post = await li_repo.create(
                user_id=user_id,
                topic=topic,
                hook=hook,
                body=body,
                hashtags=json.dumps(optimized_hashtags),
                emoji_set=post_data.get("emojis_used", ["💡", "🚀"]) and " ".join(
                    post_data.get("emojis_used", [])
                ),
                tone=tone,
                status=(
                    LinkedInPostStatus.SCHEDULED.value
                    if final_schedule_at
                    else LinkedInPostStatus.DRAFT.value
                ),
                scheduled_at=final_schedule_at,
                ai_model_used="groq/llama-3.3-70b-versatile",
            )

            duration_ms = round((time.monotonic() - t_start) * 1000, 2)

            # ── 11. Save AgentRun ─────────────────────────────────────────────
            from app.agents.discovery_agent.tools import save_agent_run
            await save_agent_run(
                self.db,
                user_id=user_id,
                agent_type=AgentType.LINKEDIN.value,
                status="success",
                input_data={"topic": topic, "tone": tone, "with_image": with_image},
                output_data={
                    "post_id":      str(post.id),
                    "quality_score": quality_score,
                    "has_image":    bool(image_path),
                    "char_count":   char_count,
                },
                duration_ms=duration_ms,
                llm_calls=self._llm_calls,
            )

            logger.info(
                "LinkedIn post generated",
                post_id=str(post.id),
                quality_score=quality_score,
                has_image=bool(image_path),
                scheduled_at=final_schedule_at.isoformat() if final_schedule_at else None,
                duration_ms=duration_ms,
            )

            return {
                "post_id":            str(post.id),
                "topic":              topic,
                "hook":               hook,
                "body":               body,
                "full_post":          full_post,
                "hashtags":           optimized_hashtags,
                "status":             post.status,
                "quality_score":      quality_score,
                "quality_grade":      quality_result.get("grade", "B"),
                "has_image":          bool(image_path),
                "image_path":         image_path,
                "image_prompt":       image_prompt,
                "character_count":    char_count,
                "scheduled_at":       final_schedule_at.isoformat() if final_schedule_at else None,
                "best_posting_time":  post_data.get("best_posting_time", "Tuesday-Thursday 8-10am"),
                "duration_ms":        duration_ms,
                "llm_calls":          self._llm_calls,
            }

    # ── Mode 2: Generate with Image (explicit) ────────────────────────────────

    async def generate_with_image(
        self,
        user_id: uuid.UUID,
        *,
        topic: str | None = None,
        tone: str = LinkedInPostTone.THOUGHT_LEADER.value,
        visual_style: str = "professional",
    ) -> dict[str, Any]:
        """Generate a post with AI image. Convenience wrapper over generate_post."""
        return await self.generate_post(
            user_id,
            topic=topic,
            tone=tone,
            with_image=True,
        )

    # ── Mode 3: Generate Carousel ─────────────────────────────────────────────

    async def generate_carousel(
        self,
        user_id: uuid.UUID,
        *,
        topic: str | None = None,
        slide_count: int = 7,
        tone: str = "educational",
    ) -> dict[str, Any]:
        """
        Generate a multi-slide LinkedIn carousel PDF post.
        Carousels get 3x more engagement than single images.
        """
        t_start = time.monotonic()

        with log_context(agent="linkedin_carousel", user_id=str(user_id)):
            # Research topic if not provided
            if not topic:
                recent_topics = await tools.get_recent_post_topics(self.db, user_id)
                topic_data    = await tools.research_trending_topic(
                    user_expertise="software engineer",
                    tech_stack=["Python", "FastAPI", "AI"],
                    recent_post_topics=recent_topics,
                )
                self._llm_calls += 1
                topic = topic_data.get("recommended_topic", "Python best practices")
                key_points = topic_data.get("key_points", [])
            else:
                key_points = []

            # Generate carousel outline
            from app.agents.linkedin_agent.prompts import CAROUSEL_OUTLINE
            from app.services.groq_service import get_groq_service

            llm = get_groq_service()
            _, user_msg = CAROUSEL_OUTLINE.render(
                topic=topic,
                key_points="\n".join(f"- {p}" for p in key_points) if key_points else topic,
                audience="software engineers and developers",
                tone=tone,
            )

            outline = await llm.complete_json(
                prompt=user_msg,
                system=CAROUSEL_OUTLINE.system,
                temperature=CAROUSEL_OUTLINE.temperature,
                max_tokens=CAROUSEL_OUTLINE.max_tokens,
            )
            self._llm_calls += 1

            # Generate PDF
            pdf_path = await tools.create_carousel_pdf(outline, user_id=user_id)

            # Save as LinkedIn post
            from app.repositories.linkedin_repository import LinkedInRepository
            li_repo = LinkedInRepository(self.db)

            caption_hook = outline.get("slides", [{}])[0].get("headline", topic)
            hashtags     = outline.get("hashtags", ["#SoftwareEngineering"])

            post = await li_repo.create(
                user_id=user_id,
                topic=topic,
                hook=caption_hook,
                body=f"Carousel: {outline.get('title', topic)}",
                hashtags=json.dumps(hashtags),
                tone=tone,
                status=LinkedInPostStatus.DRAFT.value,
                ai_model_used="groq/llama-3.3-70b-versatile",
            )

            duration_ms = round((time.monotonic() - t_start) * 1000, 2)

            logger.info(
                "Carousel generated",
                post_id=str(post.id),
                slides=len(outline.get("slides", [])),
                pdf_path=pdf_path,
            )

            return {
                "post_id":       str(post.id),
                "topic":         topic,
                "title":         outline.get("title", topic),
                "slide_count":   len(outline.get("slides", [])),
                "pdf_path":      pdf_path,
                "hashtags":      hashtags,
                "status":        post.status,
                "outline":       outline,
                "duration_ms":   duration_ms,
            }

    # ── Mode 4: Publish Post ──────────────────────────────────────────────────

    async def publish_post(
        self,
        post_id: uuid.UUID,
        user_id: uuid.UUID,
        *,
        image_path: str | None = None,
    ) -> dict[str, Any]:
        """
        Publish a draft/scheduled post to LinkedIn.
        Tries LinkedIn API first, falls back to Playwright.
        """
        with log_context(agent="linkedin_publish", user_id=str(user_id), post_id=str(post_id)):
            from app.repositories.linkedin_repository import LinkedInRepository
            from app.repositories.user_repository import UserRepository

            li_repo   = LinkedInRepository(self.db)
            user_repo = UserRepository(self.db)

            post = await li_repo.get_by_id_or_raise(post_id)
            user = await user_repo.get_by_id_or_raise(user_id)

            # Build full post content
            hashtags: list[str] = []
            if post.hashtags:
                try:
                    hashtags = json.loads(post.hashtags)
                except Exception:
                    pass

            full_content = f"{post.hook}\n\n{post.body}"
            if hashtags:
                full_content += "\n\n" + " ".join(hashtags)

            publish_result: dict[str, Any] = {}

            # ── Try API first ──────────────────────────────────────────────────
            if user.linkedin_access_token:
                try:
                    publish_result = await tools.publish_via_linkedin_api(
                        access_token=user.linkedin_access_token,
                        content=full_content,
                        image_path=image_path,
                    )
                except Exception as exc:
                    logger.warning(
                        "LinkedIn API publish failed, trying Playwright",
                        error=str(exc),
                    )

            # ── Playwright fallback ────────────────────────────────────────────
            if not publish_result.get("success"):
                if settings.linkedin.email and settings.linkedin.password:
                    publish_result = await tools.publish_via_playwright(
                        settings.linkedin.email,
                        settings.linkedin.password,
                        content=full_content,
                        image_path=image_path,
                    )
                else:
                    raise AgentError(
                        "LinkedIn publishing failed: no API token and no Playwright credentials configured. "
                        "Add LINKEDIN_EMAIL + LINKEDIN_PASSWORD or connect your LinkedIn account."
                    )

            # ── Update DB ──────────────────────────────────────────────────────
            linkedin_post_id = publish_result.get("linkedin_post_id", "")
            await li_repo.mark_published(post_id, linkedin_post_id)

            # Queue engagement sync in 24h
            try:
                from app.workers.linkedin_tasks import sync_engagement_task
                sync_engagement_task.apply_async(
                    kwargs={
                        "post_id":         str(post_id),
                        "user_id":         str(user_id),
                        "linkedin_post_id": linkedin_post_id,
                    },
                    countdown=86400,  # 24 hours
                )
            except Exception as exc:
                logger.debug("Engagement sync scheduling failed (non-critical)", error=str(exc))

            logger.info(
                "Post published",
                post_id=str(post_id),
                linkedin_post_id=linkedin_post_id,
                method=publish_result.get("method"),
                has_image=publish_result.get("has_image"),
            )

            return {
                "post_id":          str(post_id),
                "linkedin_post_id": linkedin_post_id,
                "status":           "published",
                "method":           publish_result.get("method", "unknown"),
                "has_image":        publish_result.get("has_image", False),
                "published_at":     datetime.now(UTC).isoformat(),
            }

    # ── Mode 5: Sync Engagement ───────────────────────────────────────────────

    async def sync_engagement(
        self,
        post_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> dict[str, Any]:
        """Pull latest engagement metrics from LinkedIn and update DB."""
        from app.repositories.linkedin_repository import LinkedInRepository
        from app.repositories.user_repository import UserRepository

        li_repo   = LinkedInRepository(self.db)
        user_repo = UserRepository(self.db)

        post = await li_repo.get_by_id_or_raise(post_id)
        user = await user_repo.get_by_id_or_raise(user_id)

        if not post.linkedin_post_id or not user.linkedin_access_token:
            return {"synced": False, "reason": "No LinkedIn post ID or access token"}

        metrics = await tools.sync_post_engagement(
            user.linkedin_access_token,
            post.linkedin_post_id,
        )

        await li_repo.update_engagement(
            post_id,
            impressions=metrics.get("impressions", 0),
            likes=metrics.get("likes", 0),
            comments=metrics.get("comments", 0),
            shares=metrics.get("shares", 0),
        )

        logger.info("Engagement synced", post_id=str(post_id), metrics=metrics)
        return {"synced": True, "metrics": metrics}

    # ── Mode 6: Batch Generate (week's worth) ─────────────────────────────────

    async def batch_generate(
        self,
        user_id: uuid.UUID,
        *,
        count: int = 5,
        tones: list[str] | None = None,
        with_images: bool = True,
        auto_schedule: bool = True,
    ) -> dict[str, Any]:
        """
        Generate multiple posts at once — a week's content calendar.
        Auto-schedules across Tue/Wed/Thu at optimal times.
        """
        if count > 10:
            count = 10  # Safety cap

        tones_list = tones or [
            LinkedInPostTone.THOUGHT_LEADER.value,
            LinkedInPostTone.PERSONAL.value,
            LinkedInPostTone.EDUCATIONAL.value,
            LinkedInPostTone.MOTIVATIONAL.value,
            LinkedInPostTone.THOUGHT_LEADER.value,
        ]

        formats = ["insight", "story", "list", "case_study", "question"]
        generated: list[dict[str, Any]] = []
        errors: list[str] = []

        from datetime import timedelta
        base_time = tools.get_optimal_post_time()

        for i in range(count):
            try:
                # Space posts 2 days apart
                schedule_at = base_time + timedelta(days=i * 2) if auto_schedule else None
                tone        = tones_list[i % len(tones_list)]
                fmt         = formats[i % len(formats)]

                result = await self.generate_post(
                    user_id,
                    tone=tone,
                    post_format=fmt,
                    with_image=with_images,
                    schedule_at=schedule_at,
                )
                generated.append(result)
                self._llm_calls = 0  # Reset counter per post

            except Exception as exc:
                logger.error(f"Batch post {i+1} failed", error=str(exc))
                errors.append(str(exc))

        return {
            "generated_count":  len(generated),
            "error_count":      len(errors),
            "posts":            generated,
            "errors":           errors,
            "scheduled_dates":  [p.get("scheduled_at") for p in generated if p.get("scheduled_at")],
        }


__all__ = ["LinkedInAgent"]