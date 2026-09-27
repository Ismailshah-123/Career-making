"""
CareerGPT — Outreach Agent Prompts
=====================================
PAGE SUMMARY:
  Re-exports the outreach prompt library for agent-local imports.
  Actual definitions live in app/prompts/outreach_prompts.py.
  USED BY: OutreachAgent (agent.py) via tools.py
"""

from __future__ import annotations

from app.prompts.outreach_prompts import (
    LINKEDIN_CONNECTION_REQUEST,
    LINKEDIN_INMESSAGE,
    RECRUITER_COLD_EMAIL,
    FOLLOWUP_EMAIL_PROMPT,
    INTERVIEW_THANKYOU_PROMPT,
    REFERRAL_REQUEST_PROMPT,
    OFFER_NEGOTIATION_PROMPT,
    OUTREACH_SEQUENCE_PROMPT,
    get_outreach_prompt,
)

__all__ = [
    "LINKEDIN_CONNECTION_REQUEST",
    "LINKEDIN_INMESSAGE",
    "RECRUITER_COLD_EMAIL",
    "FOLLOWUP_EMAIL_PROMPT",
    "INTERVIEW_THANKYOU_PROMPT",
    "REFERRAL_REQUEST_PROMPT",
    "OFFER_NEGOTIATION_PROMPT",
    "OUTREACH_SEQUENCE_PROMPT",
    "get_outreach_prompt",
]