"""
CareerGPT — Cover Letter Agent Prompts
==========================================
PAGE SUMMARY:
  Re-exports the cover letter prompt library for agent-local imports.
  Actual prompt definitions live centrally in app/prompts/cover_letter_prompts.py.

  USED BY: CoverLetterAgent (agent.py) via tools.py
"""

from __future__ import annotations

from app.prompts.cover_letter_prompts import (
    COVER_LETTER_GENERATE,
    COVER_LETTER_REFINE,
    COVER_LETTER_SUBJECT,
    COVER_LETTER_COLD_EMAIL,
    COVER_LETTER_REFERRAL,
    COVER_LETTER_CAREER_CHANGE,
    COVER_LETTER_EXECUTIVE,
    COVER_LETTER_SCORE,
    get_cover_letter_prompt,
)

__all__ = [
    "COVER_LETTER_GENERATE",
    "COVER_LETTER_REFINE",
    "COVER_LETTER_SUBJECT",
    "COVER_LETTER_COLD_EMAIL",
    "COVER_LETTER_REFERRAL",
    "COVER_LETTER_CAREER_CHANGE",
    "COVER_LETTER_EXECUTIVE",
    "COVER_LETTER_SCORE",
    "get_cover_letter_prompt",
]