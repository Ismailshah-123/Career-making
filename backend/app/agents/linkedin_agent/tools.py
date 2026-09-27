"""
CareerGPT — LinkedIn Agent Tools
===================================
PAGE SUMMARY:
  All tool functions used by LinkedInAgent.
  Handles: AI image generation (Stable Diffusion / DALL-E / Pollinations),
  post text generation, hashtag optimization, carousel PDF creation,
  LinkedIn API publishing, Playwright fallback publishing, engagement sync,
  scheduling queue management, and topic research via web search.

  TOOLS:
    research_trending_topic()    → search web for trending AI/tech news
    generate_post_text()         → call GroqService to write post content
    generate_image_prompt()      → create image generation prompt from post
    generate_post_image()        → generate image via free/paid API
    create_carousel_pdf()        → build carousel PDF from slide outline
    optimize_hashtags()          → select best 3-5 hashtags for reach
    score_post_quality()         → LLM quality gate before publishing
    improve_post_hook()          → rewrite hook if quality score < 7
    publish_via_linkedin_api()   → publish via LinkedIn REST API v2
    publish_via_playwright()     → Playwright fallback (no token needed)
    sync_post_engagement()       → pull latest likes/impressions from API
    schedule_post()              → add post to scheduling queue
    get_optimal_post_time()      → return best day/time for audience
    save_post_to_db()            → persist LinkedInPost record
    get_recent_post_topics()     → avoid repeating topics in last 30 days

  IMAGE GENERATION STRATEGY:
    1. Primary: Pollinations.ai (FREE, no API key, high quality)
       GET https://image.pollinations.ai/prompt/{encoded_prompt}
    2. Fallback: Together.ai FLUX (cheap, fast, high quality)
    3. Fallback: DALL-E 3 via OpenAI API (most expensive, best quality)
    4. Last resort: Generate SVG diagram via code (zero cost, always available)

  IMAGE SPECS FOR LINKEDIN:
    Feed post image:   1200×627px (1.91:1 ratio)
    Profile banner:    1584×396px
    Carousel slide:    1080×1080px (square)
    Article cover:     1280×720px

  PUBLISHING FLOW:
    1. Generate text → score → improve if needed
    2. If image requested → generate image → upload to LinkedIn media API
    3. Publish post with media_id
    4. Update DB with LinkedIn post ID
    5. Schedule engagement sync in 24h (Celery)
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx

from app.core.config import get_settings
from app.core.exceptions import LinkedInError, LinkedInPublishError, AgentError
from app.core.logging import logger

settings = get_settings()


# ══════════════════════════════════════════════════════════════════════════════
# Topic Research
# ══════════════════════════════════════════════════════════════════════════════

async def research_trending_topic(
    user_expertise: str,
    tech_stack: list[str],
    recent_post_topics: list[str],
) -> dict[str, Any]:
    """
    Research trending AI/tech topics using LLM + contextual awareness.
    Avoids recently-posted topics. Returns best topic for today's post.
    """
    from app.prompts.linkedin_prompts import LINKEDIN_TOPIC_RESEARCH
    from app.services.groq_service import get_groq_service

    llm = get_groq_service()
    _, user_msg = LINKEDIN_TOPIC_RESEARCH.render(
        user_expertise=user_expertise,
        current_projects="building AI-powered career platform",
        tech_stack=", ".join(tech_stack),
        recent_themes="LLMs, RAG, agentic AI, Python performance, DevOps automation",
        recent_posts=", ".join(recent_post_topics[-5:]) if recent_post_topics else "none",
    )

    try:
        result = await llm.complete_json(
            prompt=user_msg,
            system=LINKEDIN_TOPIC_RESEARCH.system,
            temperature=LINKEDIN_TOPIC_RESEARCH.temperature,
            max_tokens=LINKEDIN_TOPIC_RESEARCH.max_tokens,
        )
        logger.info("Topic researched", topic=result.get("recommended_topic", "")[:60])
        return result
    except Exception as exc:
        logger.warning("Topic research failed, using fallback topic", error=str(exc))
        return {
            "recommended_topic": f"How {tech_stack[0] if tech_stack else 'Python'} changed the way I approach backend architecture",
            "key_points": ["Speed of development", "Ecosystem quality", "Performance gains"],
            "hook_ideas": ["I rewrote our backend in Python last year.", "No one talks about this Python feature."],
            "cta_question": "What's your experience with this?",
        }


# ══════════════════════════════════════════════════════════════════════════════
# Post Text Generation
# ══════════════════════════════════════════════════════════════════════════════

async def generate_post_text(
    *,
    topic: str,
    key_points: list[str],
    tone: str,
    post_format: str,
    user_expertise: str,
    include_hook: bool = True,
    include_cta: bool = True,
    cta_question: str = "",
    custom_hook: str = "",
) -> dict[str, Any]:
    """Generate LinkedIn post text using LLM."""
    from app.prompts.linkedin_prompts import LINKEDIN_POST_WRITE
    from app.services.groq_service import get_groq_service

    llm = get_groq_service()
    _, user_msg = LINKEDIN_POST_WRITE.render(
        topic=topic,
        key_points="\n".join(f"- {p}" for p in key_points),
        tone=tone,
        format=post_format,
        include_hook=include_hook,
        include_cta=include_cta,
        cta_question=cta_question or "What has been your experience with this?",
        user_expertise=user_expertise,
        custom_hook=custom_hook,
    )

    result = await llm.complete_json(
        prompt=user_msg,
        system=LINKEDIN_POST_WRITE.system,
        temperature=LINKEDIN_POST_WRITE.temperature,
        max_tokens=LINKEDIN_POST_WRITE.max_tokens,
    )

    logger.info(
        "Post text generated",
        char_count=result.get("character_count", 0),
        predicted_engagement=result.get("predicted_engagement_score", 0),
    )
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Image Generation
# ══════════════════════════════════════════════════════════════════════════════

async def generate_image_prompt(
    *,
    post_topic: str,
    post_hook: str,
    key_message: str,
    visual_style: str = "professional",
) -> dict[str, Any]:
    """Generate an optimized image creation prompt from post content."""
    from app.agents.linkedin_agent.prompts import IMAGE_PROMPT_GENERATE
    from app.services.groq_service import get_groq_service

    llm = get_groq_service()
    _, user_msg = IMAGE_PROMPT_GENERATE.render(
        post_topic=post_topic,
        post_hook=post_hook,
        key_message=key_message,
        visual_style=visual_style,
    )

    try:
        return await llm.complete_json(
            prompt=user_msg,
            system=IMAGE_PROMPT_GENERATE.system,
            temperature=IMAGE_PROMPT_GENERATE.temperature,
            max_tokens=IMAGE_PROMPT_GENERATE.max_tokens,
        )
    except Exception as exc:
        logger.warning("Image prompt generation failed", error=str(exc))
        return {
            "image_prompt": f"Professional LinkedIn post image about {post_topic}. Clean minimal design, dark background, white text, tech aesthetic.",
            "image_style": "minimal_professional",
            "dimensions": "1200x627",
            "should_include_text": True,
            "text_overlay": post_hook[:80],
            "color_scheme": "dark_professional",
            "alt_text": f"LinkedIn post image: {post_topic}",
        }


async def generate_post_image(
    prompt: str,
    *,
    width: int = 1200,
    height: int = 627,
    user_id: uuid.UUID,
    post_id: uuid.UUID | None = None,
) -> str | None:
    """
    Generate an image for a LinkedIn post.

    Tries providers in order:
      1. Pollinations.ai (FREE — no API key needed, high quality)
      2. Together.ai FLUX (cheap, fast)
      3. SVG fallback (always available, zero cost)

    Returns: local file path of saved image, or None on failure.
    """
    # ── 1. Pollinations.ai (FREE) ─────────────────────────────────────────────
    image_path = await _generate_pollinations(prompt, width=width, height=height, user_id=user_id)
    if image_path:
        return image_path

    # ── 2. Together.ai FLUX ────────────────────────────────────────────────────
    together_key = os.getenv("TOGETHER_API_KEY", "")
    if together_key:
        image_path = await _generate_together(
            prompt, width=width, height=height,
            api_key=together_key, user_id=user_id
        )
        if image_path:
            return image_path

    # ── 3. SVG Fallback ────────────────────────────────────────────────────────
    logger.info("Using SVG fallback for post image")
    return await _generate_svg_fallback(prompt, user_id=user_id)


async def _generate_pollinations(
    prompt: str,
    *,
    width: int,
    height: int,
    user_id: uuid.UUID,
) -> str | None:
    """
    Generate image via Pollinations.ai — completely free, no API key.
    GET https://image.pollinations.ai/prompt/{encoded_prompt}?width=1200&height=627
    """
    try:
        encoded = quote(prompt, safe="")
        url = (
            f"https://image.pollinations.ai/prompt/{encoded}"
            f"?width={width}&height={height}&nologo=true&enhance=true"
            f"&model=flux&seed={abs(hash(prompt)) % 99999}"
        )

        async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
            response = await client.get(url)
            if response.status_code == 200 and response.headers.get("content-type", "").startswith("image/"):
                return await _save_image(response.content, user_id=user_id, ext=".png")

        logger.debug("Pollinations returned non-image response")
        return None

    except Exception as exc:
        logger.warning("Pollinations image generation failed", error=str(exc))
        return None


async def _generate_together(
    prompt: str,
    *,
    width: int,
    height: int,
    api_key: str,
    user_id: uuid.UUID,
) -> str | None:
    """Generate image via Together.ai FLUX model (cheap, high quality)."""
    try:
        payload = {
            "model":   "black-forest-labs/FLUX.1-schnell-Free",
            "prompt":  prompt,
            "width":   width,
            "height":  height,
            "steps":   4,
            "n":       1,
        }
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                "https://api.together.xyz/v1/images/generations",
                json=payload,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
            )
            if response.status_code == 200:
                data = response.json()
                images = data.get("data", [])
                if images:
                    b64 = images[0].get("b64_json") or ""
                    url  = images[0].get("url") or ""
                    if b64:
                        img_bytes = base64.b64decode(b64)
                        return await _save_image(img_bytes, user_id=user_id, ext=".png")
                    elif url:
                        img_resp = await client.get(url)
                        if img_resp.status_code == 200:
                            return await _save_image(img_resp.content, user_id=user_id, ext=".png")

        return None
    except Exception as exc:
        logger.warning("Together.ai image generation failed", error=str(exc))
        return None


async def _generate_svg_fallback(
    prompt: str,
    *,
    user_id: uuid.UUID,
) -> str:
    """
    Generate a professional SVG image as a zero-cost fallback.
    Creates a clean, branded post image with the post topic as text overlay.
    """
    title = prompt[:80] if len(prompt) > 80 else prompt
    title_lines = _wrap_text(title, max_chars=35)

    line_elements = "".join(
        f'<text x="600" y="{260 + i * 60}" '
        f'font-family="Inter, Helvetica, sans-serif" '
        f'font-size="36" font-weight="600" fill="white" text-anchor="middle">'
        f'{line}</text>'
        for i, line in enumerate(title_lines[:4])
    )

    svg_content = f"""<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="627" viewBox="0 0 1200 627">
  <defs>
    <linearGradient id="bg" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%"   style="stop-color:#0f172a;stop-opacity:1"/>
      <stop offset="50%"  style="stop-color:#1e293b;stop-opacity:1"/>
      <stop offset="100%" style="stop-color:#0f172a;stop-opacity:1"/>
    </linearGradient>
    <linearGradient id="accent" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%"   style="stop-color:#3b82f6;stop-opacity:1"/>
      <stop offset="100%" style="stop-color:#8b5cf6;stop-opacity:1"/>
    </linearGradient>
  </defs>
  <!-- Background -->
  <rect width="1200" height="627" fill="url(#bg)"/>
  <!-- Accent bar -->
  <rect x="0" y="0" width="1200" height="6" fill="url(#accent)"/>
  <!-- Grid decoration -->
  <g opacity="0.05">
    {"".join(f'<line x1="{x}" y1="0" x2="{x}" y2="627" stroke="white" stroke-width="1"/>' for x in range(0, 1201, 80))}
    {"".join(f'<line x1="0" y1="{y}" x2="1200" y2="{y}" stroke="white" stroke-width="1"/>' for y in range(0, 628, 80))}
  </g>
  <!-- CareerGPT branding -->
  <text x="60" y="60" font-family="Inter, Helvetica, sans-serif" font-size="20"
        font-weight="700" fill="#3b82f6" letter-spacing="2">CAREERGPT</text>
  <!-- Decorative circle -->
  <circle cx="600" cy="200" r="120" fill="none" stroke="url(#accent)" stroke-width="2" opacity="0.3"/>
  <circle cx="600" cy="200" r="80"  fill="none" stroke="url(#accent)" stroke-width="1" opacity="0.2"/>
  <!-- Title text -->
  {line_elements}
  <!-- Bottom bar -->
  <rect x="0" y="591" width="1200" height="36" fill="url(#accent)" opacity="0.15"/>
  <text x="600" y="614" font-family="Inter, Helvetica, sans-serif" font-size="16"
        fill="#94a3b8" text-anchor="middle">careergpt.ai</text>
</svg>"""

    save_dir = settings.storage.upload_dir / str(user_id) / "linkedin_images"
    save_dir.mkdir(parents=True, exist_ok=True)
    file_path = save_dir / f"post_{uuid.uuid4().hex[:8]}.svg"
    file_path.write_text(svg_content, encoding="utf-8")
    logger.info("SVG post image generated", path=str(file_path))
    return str(file_path)


async def _save_image(
    content: bytes,
    *,
    user_id: uuid.UUID,
    ext: str = ".png",
) -> str:
    """Save raw image bytes to disk. Returns file path."""
    save_dir = settings.storage.upload_dir / str(user_id) / "linkedin_images"
    save_dir.mkdir(parents=True, exist_ok=True)
    filename = f"post_{uuid.uuid4().hex[:12]}{ext}"
    file_path = save_dir / filename
    file_path.write_bytes(content)
    logger.debug("Image saved", path=str(file_path), size_kb=len(content) // 1024)
    return str(file_path)


def _wrap_text(text: str, max_chars: int = 35) -> list[str]:
    """Wrap text into lines for SVG rendering."""
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        if len(current) + len(word) + 1 <= max_chars:
            current = f"{current} {word}".strip()
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


# ══════════════════════════════════════════════════════════════════════════════
# Carousel PDF Creation
# ══════════════════════════════════════════════════════════════════════════════

async def create_carousel_pdf(
    outline: dict[str, Any],
    *,
    user_id: uuid.UUID,
) -> str | None:
    """
    Create a LinkedIn carousel PDF from the slide outline.
    Uses reportlab for PDF generation.
    Each slide = one page = 1080x1080px.
    Returns file path of generated PDF.
    """
    try:
        from reportlab.lib.pagesizes import landscape
        from reportlab.lib.units import mm
        from reportlab.pdfgen import canvas as rl_canvas
        from reportlab.lib.colors import HexColor

        slides = outline.get("slides", [])
        if not slides:
            return None

        save_dir = settings.storage.upload_dir / str(user_id) / "linkedin_carousels"
        save_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = save_dir / f"carousel_{uuid.uuid4().hex[:8]}.pdf"

        PAGE_SIZE = (1080, 1080)
        c = rl_canvas.Canvas(str(pdf_path), pagesize=PAGE_SIZE)

        DARK_BG    = HexColor("#0f172a")
        ACCENT     = HexColor("#3b82f6")
        WHITE      = HexColor("#ffffff")
        GRAY       = HexColor("#94a3b8")
        PURPLE     = HexColor("#8b5cf6")

        for slide in slides:
            slide_num  = slide.get("slide_number", 1)
            headline   = slide.get("headline", "")
            subtext    = slide.get("subtext", "")
            slide_type = slide.get("type", "content")
            emoji      = slide.get("emoji", "")
            code       = slide.get("code_snippet", "")

            # Background
            c.setFillColor(DARK_BG)
            c.rect(0, 0, 1080, 1080, fill=1, stroke=0)

            # Top accent bar
            c.setFillColor(ACCENT)
            c.rect(0, 1074, 1080, 6, fill=1, stroke=0)

            # Slide number indicator
            c.setFillColor(GRAY)
            c.setFont("Helvetica", 16)
            total = outline.get("slide_count", len(slides))
            c.drawString(60, 1040, f"{slide_num}/{total}")

            # Branding
            c.setFillColor(ACCENT)
            c.setFont("Helvetica-Bold", 18)
            c.drawRightString(1020, 1040, "CareerGPT")

            # Emoji (large, centered)
            if emoji and slide_type in ("hook", "cta"):
                c.setFont("Helvetica", 80)
                c.setFillColor(WHITE)
                c.drawCentredString(540, 750, emoji)

            # Slide number badge (for content slides)
            if slide_type == "content" and "#" in headline:
                c.setFillColor(ACCENT)
                c.roundRect(60, 680, 180, 60, 12, fill=1, stroke=0)
                c.setFillColor(WHITE)
                c.setFont("Helvetica-Bold", 28)
                badge_text = headline.split("#")[1].split()[0] if "#" in headline else str(slide_num)
                c.drawCentredString(150, 700, f"#{badge_text}")

            # Headline
            c.setFillColor(WHITE)
            clean_headline = headline.replace(f"#{badge_text}" if "#" in headline else "", "").strip() if slide_type == "content" else headline
            c.setFont("Helvetica-Bold", 44 if len(clean_headline) < 40 else 36)
            _draw_wrapped_text(c, clean_headline, x=60, y=640, max_width=960, line_height=52, fill_color=WHITE)

            # Subtext
            if subtext:
                c.setFillColor(GRAY)
                c.setFont("Helvetica", 28)
                _draw_wrapped_text(c, subtext, x=60, y=460, max_width=960, line_height=38, fill_color=GRAY)

            # Code snippet
            if code:
                c.setFillColor(HexColor("#1e293b"))
                c.roundRect(60, 200, 960, 220, 12, fill=1, stroke=0)
                c.setFillColor(HexColor("#e2e8f0"))
                c.setFont("Courier", 20)
                code_lines = code.replace("\\n", "\n").split("\n")
                for i, line in enumerate(code_lines[:6]):
                    c.drawString(90, 400 - i * 32, line)

            # Bottom CTA (last slide)
            if slide_type in ("cta", "follow"):
                c.setFillColor(ACCENT)
                c.roundRect(290, 80, 500, 72, 36, fill=1, stroke=0)
                c.setFillColor(WHITE)
                c.setFont("Helvetica-Bold", 26)
                c.drawCentredString(540, 108, "Follow for more →")

            c.showPage()

        c.save()
        logger.info("Carousel PDF created", path=str(pdf_path), slides=len(slides))
        return str(pdf_path)

    except ImportError:
        logger.warning("reportlab not installed — carousel PDF skipped")
        return None
    except Exception as exc:
        logger.error("Carousel PDF creation failed", error=str(exc))
        return None


def _draw_wrapped_text(
    canvas: Any,
    text: str,
    *,
    x: float,
    y: float,
    max_width: float,
    line_height: float,
    fill_color: Any,
) -> None:
    """Draw text with word wrapping on a reportlab canvas."""
    canvas.setFillColor(fill_color)
    words = text.split()
    line = ""
    current_y = y

    for word in words:
        test_line = f"{line} {word}".strip()
        if canvas.stringWidth(test_line) <= max_width:
            line = test_line
        else:
            if line:
                canvas.drawString(x, current_y, line)
                current_y -= line_height
            line = word
    if line:
        canvas.drawString(x, current_y, line)


# ══════════════════════════════════════════════════════════════════════════════
# Post Quality Gate
# ══════════════════════════════════════════════════════════════════════════════

async def score_post_quality(post_text: str) -> dict[str, Any]:
    """Score post quality. Returns score + decision to publish or revise."""
    from app.prompts.linkedin_prompts import LINKEDIN_POST_SCORE
    from app.services.groq_service import get_groq_service

    llm = get_groq_service()
    _, user_msg = LINKEDIN_POST_SCORE.render(linkedin_post=post_text)

    try:
        result = await llm.complete_json(
            prompt=user_msg,
            system=LINKEDIN_POST_SCORE.system,
            temperature=LINKEDIN_POST_SCORE.temperature,
            max_tokens=LINKEDIN_POST_SCORE.max_tokens,
        )
        logger.info("Post scored", score=result.get("overall_score"), grade=result.get("grade"))
        return result
    except Exception as exc:
        logger.warning("Post scoring failed", error=str(exc))
        return {"overall_score": 70, "grade": "B", "post_or_revise": "post_with_minor_edits"}


async def improve_hook(post_text: str, topic: str) -> str:
    """Rewrite the first line of a post to improve engagement."""
    from app.prompts.linkedin_prompts import LINKEDIN_POST_HOOK_IMPROVE
    from app.services.groq_service import get_groq_service

    first_line = post_text.split("\n")[0] if "\n" in post_text else post_text[:150]
    llm = get_groq_service()
    _, user_msg = LINKEDIN_POST_HOOK_IMPROVE.render(
        current_hook=first_line,
        topic=topic,
        audience="software engineers and tech professionals",
    )

    try:
        result = await llm.complete_json(
            prompt=user_msg,
            system=LINKEDIN_POST_HOOK_IMPROVE.system,
            temperature=LINKEDIN_POST_HOOK_IMPROVE.temperature,
            max_tokens=LINKEDIN_POST_HOOK_IMPROVE.max_tokens,
        )
        best_hook = result.get("recommended", first_line)
        return post_text.replace(first_line, best_hook, 1)
    except Exception:
        return post_text


# ══════════════════════════════════════════════════════════════════════════════
# LinkedIn Publishing
# ══════════════════════════════════════════════════════════════════════════════

async def upload_image_to_linkedin(
    access_token: str,
    image_path: str,
    person_urn: str,
) -> str | None:
    """
    Upload image to LinkedIn media API.
    Returns asset URN for use in post creation.
    LinkedIn image upload is a 3-step process:
      1. Register upload → get upload URL + asset URN
      2. PUT image bytes to upload URL
      3. Use asset URN in post creation
    """
    if not Path(image_path).exists():
        logger.warning("Image file not found for upload", path=image_path)
        return None

    headers = {
        "Authorization":             f"Bearer {access_token}",
        "X-Restli-Protocol-Version": "2.0.0",
        "Content-Type":              "application/json",
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        # Step 1: Register upload
        register_payload = {
            "registerUploadRequest": {
                "recipes":     ["urn:li:digitalmediaRecipe:feedshare-image"],
                "owner":       person_urn,
                "serviceRelationships": [{
                    "relationshipType": "OWNER",
                    "identifier":       "urn:li:userGeneratedContent",
                }],
            }
        }

        try:
            reg_resp = await client.post(
                "https://api.linkedin.com/v2/assets?action=registerUpload",
                json=register_payload,
                headers=headers,
            )
            if reg_resp.status_code not in (200, 201):
                logger.warning("LinkedIn register upload failed", status=reg_resp.status_code)
                return None

            reg_data     = reg_resp.json()
            upload_url   = reg_data["value"]["uploadMechanism"]["com.linkedin.digitalmedia.uploading.MediaUploadHttpRequest"]["uploadUrl"]
            asset_urn    = reg_data["value"]["asset"]

            # Step 2: Upload binary image
            img_bytes = Path(image_path).read_bytes()
            content_type = "image/svg+xml" if image_path.endswith(".svg") else "image/png"

            upload_resp = await client.put(
                upload_url,
                content=img_bytes,
                headers={
                    "Authorization": f"Bearer {access_token}",
                    "Content-Type":  content_type,
                },
            )
            if upload_resp.status_code not in (200, 201):
                logger.warning("LinkedIn image upload failed", status=upload_resp.status_code)
                return None

            logger.info("Image uploaded to LinkedIn", asset_urn=asset_urn)
            return asset_urn

        except Exception as exc:
            logger.warning("LinkedIn image upload error", error=str(exc))
            return None


async def publish_via_linkedin_api(
    access_token: str,
    *,
    content: str,
    image_path: str | None = None,
    scheduled_publish_time: int | None = None,
) -> dict[str, Any]:
    """
    Publish a post to LinkedIn via REST API v2.
    Supports: text-only, text+image, scheduled posts.
    Returns: {success, linkedin_post_id, method, error}
    """
    headers = {
        "Authorization":             f"Bearer {access_token}",
        "X-Restli-Protocol-Version": "2.0.0",
        "Content-Type":              "application/json",
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        # Get person URN
        try:
            profile_resp = await client.get(
                "https://api.linkedin.com/v2/userinfo",
                headers=headers,
            )
            if profile_resp.status_code == 401:
                raise LinkedInError("LinkedIn access token expired or invalid")
            profile      = profile_resp.json()
            person_urn   = f"urn:li:person:{profile.get('sub', '')}"
        except LinkedInError:
            raise
        except Exception as exc:
            raise LinkedInError(f"Failed to get LinkedIn profile: {exc}") from exc

        # Upload image if provided
        media_asset_urn: str | None = None
        if image_path:
            media_asset_urn = await upload_image_to_linkedin(
                access_token, image_path, person_urn
            )

        # Build post payload
        specific_content: dict[str, Any] = {
            "com.linkedin.ugc.ShareContent": {
                "shareCommentary": {"text": content},
                "shareMediaCategory": "NONE" if not media_asset_urn else "IMAGE",
            }
        }

        if media_asset_urn:
            specific_content["com.linkedin.ugc.ShareContent"]["media"] = [{
                "status":      "READY",
                "description": {"text": ""},
                "media":       media_asset_urn,
                "title":       {"text": ""},
            }]

        post_payload: dict[str, Any] = {
            "author":         person_urn,
            "lifecycleState": "PUBLISHED",
            "specificContent": specific_content,
            "visibility": {
                "com.linkedin.ugc.MemberNetworkVisibility": "PUBLIC"
            },
        }

        if scheduled_publish_time:
            post_payload["lifecycleState"] = "DRAFT"
            post_payload["scheduledPublishTime"] = scheduled_publish_time

        try:
            post_resp = await client.post(
                "https://api.linkedin.com/v2/ugcPosts",
                json=post_payload,
                headers=headers,
            )

            if post_resp.status_code not in (200, 201):
                raise LinkedInPublishError(
                    f"LinkedIn post creation failed: {post_resp.status_code} — {post_resp.text[:300]}"
                )

            post_id = post_resp.headers.get("x-restli-id", "")
            logger.info("Post published via LinkedIn API", post_id=post_id, has_image=bool(media_asset_urn))

            return {
                "success":         True,
                "linkedin_post_id": post_id,
                "method":          "api",
                "has_image":       bool(media_asset_urn),
                "person_urn":      person_urn,
            }

        except LinkedInPublishError:
            raise
        except Exception as exc:
            raise LinkedInPublishError(f"Post publish failed: {exc}") from exc


async def publish_via_playwright(
    email: str,
    password: str,
    *,
    content: str,
    image_path: str | None = None,
) -> dict[str, Any]:
    """
    Playwright-based LinkedIn publishing fallback.
    Used when no OAuth access token is available.
    Automates: login → open post composer → type content → attach image → post.
    """
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        raise AgentError("playwright not installed — cannot publish via browser automation")

    if not email or not password:
        raise LinkedInError("LinkedIn credentials not configured for Playwright publishing")

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=settings.playwright.headless)
        ctx     = await browser.new_context(
            viewport={"width": 1280, "height": 800},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        )
        page = await ctx.new_page()

        try:
            # Login
            await page.goto("https://www.linkedin.com/login", wait_until="networkidle")
            await page.fill("#username", email)
            await page.fill("#password", password)
            await page.click("button[type='submit']")
            await page.wait_for_url("**/feed/**", timeout=20000)

            # Open post composer
            start_post_btn = await page.wait_for_selector(
                "button.share-box-feed-entry__trigger, button:has-text('Start a post')",
                timeout=10000,
            )
            await start_post_btn.click()
            await page.wait_for_timeout(1500)

            # Type content
            editor = await page.wait_for_selector(
                "div.ql-editor, div[contenteditable='true'][role='textbox']",
                timeout=5000,
            )
            await editor.click()
            await editor.type(content, delay=10)
            await page.wait_for_timeout(800)

            # Attach image if provided
            if image_path and Path(image_path).exists():
                try:
                    media_btn = await page.query_selector(
                        "button[aria-label*='media'], button[aria-label*='image'], "
                        "button[aria-label*='photo'], .share-creation-state__media-button"
                    )
                    if media_btn:
                        # Find file input via JavaScript
                        file_input = await page.query_selector("input[type='file'][accept*='image']")
                        if file_input:
                            await file_input.set_input_files(image_path)
                            await page.wait_for_timeout(3000)
                            logger.info("Image attached via Playwright")
                except Exception as exc:
                    logger.warning("Image attach via Playwright failed (continuing without)", error=str(exc))

            # Click Post
            post_btn = await page.wait_for_selector(
                "button:has-text('Post'), "
                "button.share-actions__primary-action, "
                "button[aria-label='Post']",
                timeout=5000,
            )
            await post_btn.click()
            await page.wait_for_timeout(4000)

            logger.info("Post published via Playwright")
            return {
                "success":          True,
                "linkedin_post_id": "",
                "method":           "playwright",
                "has_image":        bool(image_path),
            }

        except Exception as exc:
            raise LinkedInPublishError(f"Playwright publish failed: {exc}") from exc
        finally:
            await browser.close()


# ══════════════════════════════════════════════════════════════════════════════
# Engagement Sync
# ══════════════════════════════════════════════════════════════════════════════

async def sync_post_engagement(
    access_token: str,
    linkedin_post_id: str,
) -> dict[str, int]:
    """
    Sync post engagement metrics from LinkedIn Analytics API.
    Returns: {impressions, likes, comments, shares}
    """
    if not access_token or not linkedin_post_id:
        return {"impressions": 0, "likes": 0, "comments": 0, "shares": 0}

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                f"https://api.linkedin.com/v2/socialActions/{linkedin_post_id}",
                headers={
                    "Authorization":             f"Bearer {access_token}",
                    "X-Restli-Protocol-Version": "2.0.0",
                },
            )
            if resp.status_code == 200:
                data = resp.json()
                return {
                    "likes":    data.get("likesSummary", {}).get("totalLikes", 0),
                    "comments": data.get("commentsSummary", {}).get("totalFirstLevelComments", 0),
                    "shares":   data.get("sharesSummary", {}).get("totalShares", 0),
                    "impressions": 0,  # Requires separate analytics API call
                }
    except Exception as exc:
        logger.debug("Engagement sync failed", error=str(exc))

    return {"impressions": 0, "likes": 0, "comments": 0, "shares": 0}


# ══════════════════════════════════════════════════════════════════════════════
# Utility
# ══════════════════════════════════════════════════════════════════════════════

def get_optimal_post_time() -> datetime:
    """
    Return the next optimal posting time based on LinkedIn engagement research.
    Best times: Tue/Wed/Thu 8-10am local time.
    Avoids: weekends, Monday mornings, Friday afternoons.
    """
    now = datetime.now(UTC)
    optimal_hour = 9  # 9am UTC (adjust per user timezone in future)

    days_to_add = 0
    target = now.replace(hour=optimal_hour, minute=0, second=0, microsecond=0)

    if now.hour >= optimal_hour:
        days_to_add = 1

    for _ in range(7):
        candidate = target + timedelta(days=days_to_add)
        weekday = candidate.weekday()
        if weekday in (1, 2, 3):  # Tue, Wed, Thu
            return candidate
        days_to_add += 1

    return target + timedelta(days=1)


async def get_recent_post_topics(
    db: Any,
    user_id: uuid.UUID,
    days: int = 30,
) -> list[str]:
    """Fetch topics of posts created in last N days (avoid repetition)."""
    try:
        from app.repositories.linkedin_repository import LinkedInRepository
        repo = LinkedInRepository(db)
        posts = await repo.get_user_posts(user_id, limit=20)
        cutoff = datetime.now(UTC) - timedelta(days=days)
        return [
            p.topic for p in posts
            if p.created_at and p.created_at.replace(tzinfo=UTC) >= cutoff
        ]
    except Exception:
        return []


__all__ = [
    "research_trending_topic",
    "generate_post_text",
    "generate_image_prompt",
    "generate_post_image",
    "create_carousel_pdf",
    "score_post_quality",
    "improve_hook",
    "publish_via_linkedin_api",
    "publish_via_playwright",
    "sync_post_engagement",
    "get_optimal_post_time",
    "get_recent_post_topics",
    "upload_image_to_linkedin",
]