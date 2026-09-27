"""
CareerGPT — LinkedIn Agent Prompts
=====================================
PAGE SUMMARY:
  Prompts for the LinkedIn agent — the most creative and complex agent.
  Handles text posts, image-enhanced posts, carousels, polls, and scheduling.

  USED BY: LinkedInAgent (agent.py)
"""

from __future__ import annotations
from app.prompts.resume_prompts import Prompt
from app.prompts.linkedin_prompts import (
    LINKEDIN_TOPIC_RESEARCH,
    LINKEDIN_POST_WRITE,
    LINKEDIN_POST_HOOK_IMPROVE,
    LINKEDIN_HASHTAG_SELECT,
    LINKEDIN_POST_SCORE,
)

# Re-export shared prompts for agent use
__all__ = [
    "LINKEDIN_TOPIC_RESEARCH",
    "LINKEDIN_POST_WRITE",
    "LINKEDIN_POST_HOOK_IMPROVE",
    "LINKEDIN_HASHTAG_SELECT",
    "LINKEDIN_POST_SCORE",
]

# ── Agent-specific additional prompts ─────────────────────────────────────────

IMAGE_PROMPT_GENERATE = Prompt(
    name="linkedin_image_prompt",
    version="v2",
    temperature=0.7,
    max_tokens=400,
    system="""You generate image prompts for LinkedIn post visuals.
The image should visually reinforce the post's key message.
For technical posts: diagrams, code visualizations, architecture images.
For career posts: professional workspace, growth charts, people collaborating.
Keep prompts under 100 words. Make them specific and visually distinctive.
Return ONLY valid JSON.""",
    user_template="""Generate an image prompt for this LinkedIn post:

POST TOPIC: $post_topic
POST HOOK: $post_hook
KEY MESSAGE: $key_message
VISUAL STYLE: $visual_style (professional / minimal / bold / data-viz)

Return:
{
  "image_prompt": "Clean minimal diagram showing FastAPI request lifecycle with color-coded layers: client (blue), router (green), service (orange), database (red). Dark background, white text labels, tech aesthetic.",
  "image_style": "technical_diagram",
  "dimensions": "1200x627",
  "should_include_text": true,
  "text_overlay": "Why 90% of FastAPI apps have the same bottleneck",
  "color_scheme": "dark_professional",
  "alt_text": "FastAPI request lifecycle diagram showing performance bottleneck at database layer"
}""",
)

CAROUSEL_OUTLINE = Prompt(
    name="linkedin_carousel_outline",
    version="v1",
    temperature=0.7,
    max_tokens=800,
    system="""You create LinkedIn carousel outlines (multi-image PDF posts).
Carousels get 3x more engagement than single-image posts on LinkedIn.
Each slide should have ONE key point, minimal text, and a clear visual concept.
Format: 5-10 slides. First slide = hook. Last slide = CTA + follow prompt.
Return ONLY valid JSON.""",
    user_template="""Create a LinkedIn carousel outline:

TOPIC: $topic
KEY POINTS: $key_points
AUDIENCE: $audience
TONE: $tone

Return:
{
  "title": "7 Python Mistakes Senior Engineers Make",
  "slide_count": 7,
  "slides": [
    {
      "slide_number": 1,
      "type": "hook",
      "headline": "7 Python mistakes I see in 90% of codebases",
      "subtext": "Even from senior engineers",
      "visual_concept": "Bold text on dark background, number 7 prominently displayed",
      "emoji": "🐍"
    },
    {
      "slide_number": 2,
      "type": "content",
      "headline": "Mistake #1: Using async def for CPU work",
      "subtext": "async doesn't make CPU-bound code faster. It makes it slower.",
      "visual_concept": "Before/after code comparison, red X on async, green check on threading",
      "code_snippet": "# Wrong\\nasync def compute(): ...\\n\\n# Right\\ndef compute(): ..."
    }
  ],
  "cover_image_prompt": "Bold, high-contrast slide with Python logo and number 7",
  "hashtags": ["#Python", "#SoftwareEngineering", "#BackendDev"],
  "estimated_engagement": "high"
}""",
)

LINKEDIN_AGENT_PROMPTS = {
    "image_prompt":     IMAGE_PROMPT_GENERATE,
    "carousel_outline": CAROUSEL_OUTLINE,
}