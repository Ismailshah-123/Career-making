"""
CareerGPT — Resume Agent Prompts
===================================
PAGE SUMMARY:
  Re-exports the resume prompt library for agent-local imports.
  Mirrors the pattern used by discovery_agent/prompts.py and
  matching_agent/prompts.py — agents import from their own namespace,
  the actual prompt definitions live centrally in app/prompts/.

  USED BY: ResumeAgent (agent.py) via tools.py
"""

from __future__ import annotations

from app.prompts.resume_prompts import (
    RESUME_PARSE_PROMPT,
    RESUME_ATS_SCORE_PROMPT,
    RESUME_TAILOR_PROMPT,
    RESUME_ANALYZE_PROMPT,
    RESUME_BULLET_IMPROVE,
    RESUME_SUMMARY_WRITE,
    get_resume_prompt,
)

__all__ = [
    "RESUME_PARSE_PROMPT",
    "RESUME_ATS_SCORE_PROMPT",
    "RESUME_TAILOR_PROMPT",
    "RESUME_ANALYZE_PROMPT",
    "RESUME_BULLET_IMPROVE",
    "RESUME_SUMMARY_WRITE",
    "get_resume_prompt",
]