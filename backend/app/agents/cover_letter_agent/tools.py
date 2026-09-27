"""
CareerGPT — Cover Letter Agent Tools
=======================================
PAGE SUMMARY:
  Tool functions for the CoverLetterAgent. Wraps each cover letter prompt
  variant (standard, referral, career change, executive, cold email) in a
  self-contained async function with input normalization and graceful
  fallback if the LLM call fails.

  TOOLS:
    generate_standard_letter()  → main 4-tone cover letter generator
    generate_referral_letter()  → when candidate has an internal referral
    generate_career_change_letter() → for career switchers
    generate_executive_letter() → C-suite / VP / Director level
    generate_cold_email()       → short, direct outreach (not a formal letter)
    score_letter()              → quality gate, 0-100 score
    refine_letter()             → iterative improvement from user feedback
    pick_tone_for_seniority()   → auto-select tone based on job seniority signal
    extract_candidate_brief()   → pull structured fields off a Resume ORM object
    extract_job_brief()         → pull structured fields off a Job ORM object
"""

from __future__ import annotations

import json
from typing import Any

from app.core.logging import logger


# ══════════════════════════════════════════════════════════════════════════════
# Brief Extraction (Resume/Job ORM → flat prompt-ready dicts)
# ══════════════════════════════════════════════════════════════════════════════

def extract_candidate_brief(resume: Any) -> dict[str, Any]:
    """Pull the fields cover letter prompts need off a Resume ORM object."""
    skills: list[str] = []
    if resume.skills:
        try:
            skills = json.loads(resume.skills)
        except Exception:
            pass

    best_achievement = ""
    if resume.summary:
        best_achievement = resume.summary[:200]

    return {
        "candidate_name":   resume.name or "Candidate",
        "candidate_role":   "",  # filled by caller from most recent experience if available
        "candidate_summary": resume.summary or (resume.raw_text or "")[:600],
        "top_skills":       skills[:6],
        "best_achievement":  best_achievement,
        "experience_years":  resume.experience_years or 0.0,
    }


def extract_job_brief(job: Any) -> dict[str, Any]:
    """Pull the fields cover letter prompts need off a Job ORM object."""
    required_skills: list[str] = []
    if job.skills_required:
        try:
            required_skills = json.loads(job.skills_required)
        except Exception:
            pass

    return {
        "job_title":         job.title,
        "company_name":      job.company,
        "company_context":   job.ai_summary or "",
        "key_requirements":  (job.requirements or job.description or "")[:600],
        "tech_stack":        ", ".join(required_skills[:8]),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Standard Cover Letter
# ══════════════════════════════════════════════════════════════════════════════

async def generate_standard_letter(
    *,
    candidate: dict[str, Any],
    job: dict[str, Any],
    tone: str = "professional",
    custom_context: str = "",
    highlight_skills: list[str] | None = None,
) -> dict[str, Any]:
    """
    Generate a standard cover letter using the 4-tone main prompt.
    Returns the full COVER_LETTER_GENERATE JSON shape — caller reads
    "full_body" as the deliverable text and "subject_line" for email use.
    """
    from app.agents.cover_letter_agent.prompts import COVER_LETTER_GENERATE
    from app.services.groq_service import get_groq_service

    llm = get_groq_service()
    _, user_msg = COVER_LETTER_GENERATE.render(
        candidate_name=candidate.get("candidate_name", "Candidate"),
        candidate_role=candidate.get("candidate_role", ""),
        candidate_summary=candidate.get("candidate_summary", ""),
        top_skills=", ".join(candidate.get("top_skills", [])),
        best_achievement=candidate.get("best_achievement", ""),
        job_title=job.get("job_title", ""),
        company_name=job.get("company_name", ""),
        company_context=job.get("company_context", ""),
        key_requirements=job.get("key_requirements", ""),
        tech_stack=job.get("tech_stack", ""),
        tone=tone,
        custom_context=custom_context or "none",
        highlight_skills=", ".join(highlight_skills) if highlight_skills else "none specified",
    )

    try:
        result = await llm.complete_json(
            prompt=user_msg,
            system=COVER_LETTER_GENERATE.system,
            temperature=COVER_LETTER_GENERATE.temperature,
            max_tokens=COVER_LETTER_GENERATE.max_tokens,
        )
        result.setdefault("full_body", _assemble_fallback(result))
        return result
    except Exception as exc:
        logger.warning("Standard cover letter generation failed", error=str(exc))
        return _fallback_letter(candidate, job)


# ══════════════════════════════════════════════════════════════════════════════
# Referral Cover Letter
# ══════════════════════════════════════════════════════════════════════════════

async def generate_referral_letter(
    *,
    candidate: dict[str, Any],
    job: dict[str, Any],
    referrer_name: str,
    referrer_title: str = "",
    referral_relationship: str = "",
    referrer_quote: str = "",
    tone: str = "professional",
) -> dict[str, Any]:
    """Generate a cover letter that leads with an internal referral."""
    from app.agents.cover_letter_agent.prompts import COVER_LETTER_REFERRAL
    from app.services.groq_service import get_groq_service

    llm = get_groq_service()
    _, user_msg = COVER_LETTER_REFERRAL.render(
        candidate_name=candidate.get("candidate_name", "Candidate"),
        candidate_summary=candidate.get("candidate_summary", ""),
        best_achievement=candidate.get("best_achievement", ""),
        relevant_skills=", ".join(candidate.get("top_skills", [])),
        referrer_name=referrer_name,
        referrer_title=referrer_title,
        referral_relationship=referral_relationship or "former colleague",
        referrer_quote=referrer_quote or "none provided",
        job_title=job.get("job_title", ""),
        company_name=job.get("company_name", ""),
        key_requirements=job.get("key_requirements", ""),
        tone=tone,
    )

    try:
        result = await llm.complete_json(
            prompt=user_msg,
            system=COVER_LETTER_REFERRAL.system,
            temperature=COVER_LETTER_REFERRAL.temperature,
            max_tokens=COVER_LETTER_REFERRAL.max_tokens,
        )
        result.setdefault("full_body", _assemble_fallback(result, hook_key="opening_with_referral", body_key="qualification_paragraph"))
        return result
    except Exception as exc:
        logger.warning("Referral cover letter generation failed", error=str(exc))
        return _fallback_letter(candidate, job)


# ══════════════════════════════════════════════════════════════════════════════
# Career Change Cover Letter
# ══════════════════════════════════════════════════════════════════════════════

async def generate_career_change_letter(
    *,
    candidate: dict[str, Any],
    job: dict[str, Any],
    previous_career: str,
    change_reason: str,
    upskilling: str = "",
    transferable_achievement: str = "",
    tone: str = "enthusiastic",
) -> dict[str, Any]:
    """Generate a cover letter framing a career transition as an asset."""
    from app.agents.cover_letter_agent.prompts import COVER_LETTER_CAREER_CHANGE
    from app.services.groq_service import get_groq_service

    llm = get_groq_service()
    _, user_msg = COVER_LETTER_CAREER_CHANGE.render(
        candidate_name=candidate.get("candidate_name", "Candidate"),
        previous_career=previous_career,
        target_career=job.get("job_title", ""),
        change_reason=change_reason,
        upskilling=upskilling or "actively building relevant project portfolio",
        transferable_skills=", ".join(candidate.get("top_skills", [])),
        transferable_achievement=transferable_achievement or candidate.get("best_achievement", ""),
        job_title=job.get("job_title", ""),
        company_name=job.get("company_name", ""),
        company_values=job.get("company_context", ""),
        tech_requirements=job.get("tech_stack", ""),
        tone=tone,
    )

    try:
        result = await llm.complete_json(
            prompt=user_msg,
            system=COVER_LETTER_CAREER_CHANGE.system,
            temperature=COVER_LETTER_CAREER_CHANGE.temperature,
            max_tokens=COVER_LETTER_CAREER_CHANGE.max_tokens,
        )
        result.setdefault("full_body", _assemble_fallback(result, hook_key="opening_narrative", body_key="bridge_paragraph"))
        return result
    except Exception as exc:
        logger.warning("Career change letter generation failed", error=str(exc))
        return _fallback_letter(candidate, job)


# ══════════════════════════════════════════════════════════════════════════════
# Executive Cover Letter
# ══════════════════════════════════════════════════════════════════════════════

async def generate_executive_letter(
    *,
    candidate: dict[str, Any],
    job: dict[str, Any],
    seniority_level: str,
    leadership_years: float,
    scale_metric: str,
    strategic_win: str = "",
    board_experience: str = "",
    notable_companies: str = "",
    company_stage: str = "",
    investors: str = "",
    business_challenges: str = "",
) -> dict[str, Any]:
    """Generate a C-suite / VP / Director-level cover letter."""
    from app.agents.cover_letter_agent.prompts import COVER_LETTER_EXECUTIVE
    from app.services.groq_service import get_groq_service

    llm = get_groq_service()
    _, user_msg = COVER_LETTER_EXECUTIVE.render(
        candidate_name=candidate.get("candidate_name", "Candidate"),
        seniority_level=seniority_level,
        leadership_years=leadership_years,
        scale_metric=scale_metric,
        strategic_win=strategic_win or candidate.get("best_achievement", ""),
        board_experience=board_experience or "none specified",
        notable_companies=notable_companies or "none specified",
        job_title=job.get("job_title", ""),
        company_name=job.get("company_name", ""),
        company_stage=company_stage or "established",
        investors=investors or "not specified",
        business_challenges=business_challenges or job.get("company_context", ""),
        reports_to="Leadership Team",
    )

    try:
        result = await llm.complete_json(
            prompt=user_msg,
            system=COVER_LETTER_EXECUTIVE.system,
            temperature=COVER_LETTER_EXECUTIVE.temperature,
            max_tokens=COVER_LETTER_EXECUTIVE.max_tokens,
        )
        result.setdefault("full_body", _assemble_fallback(result, hook_key="opening_strategic", body_key="leadership_track_record"))
        return result
    except Exception as exc:
        logger.warning("Executive letter generation failed", error=str(exc))
        return _fallback_letter(candidate, job)


# ══════════════════════════════════════════════════════════════════════════════
# Cold Email (not a formal letter)
# ══════════════════════════════════════════════════════════════════════════════

async def generate_cold_email(
    *,
    candidate: dict[str, Any],
    job: dict[str, Any],
    recipient_name: str = "",
    recipient_title: str = "",
    recipient_context: str = "",
    why_company: str = "",
) -> dict[str, Any]:
    """Generate a short, direct cold outreach email (distinct from a formal cover letter)."""
    from app.agents.cover_letter_agent.prompts import COVER_LETTER_COLD_EMAIL
    from app.services.groq_service import get_groq_service

    llm = get_groq_service()
    _, user_msg = COVER_LETTER_COLD_EMAIL.render(
        sender_name=candidate.get("candidate_name", "Candidate"),
        sender_title=candidate.get("candidate_role", "") or "Software Professional",
        sender_achievement=candidate.get("best_achievement", ""),
        sender_experience=candidate.get("experience_years", 0),
        sender_skills=", ".join(candidate.get("top_skills", [])),
        recipient_name=recipient_name or "there",
        recipient_title=recipient_title,
        company_name=job.get("company_name", ""),
        recipient_context=recipient_context or job.get("company_context", ""),
        target_role=job.get("job_title", ""),
        why_company=why_company or job.get("company_context", ""),
    )

    try:
        return await llm.complete_json(
            prompt=user_msg,
            system=COVER_LETTER_COLD_EMAIL.system,
            temperature=COVER_LETTER_COLD_EMAIL.temperature,
            max_tokens=COVER_LETTER_COLD_EMAIL.max_tokens,
        )
    except Exception as exc:
        logger.warning("Cold email generation failed", error=str(exc))
        return {
            "subject": f"Interest in {job.get('job_title', 'the role')} at {job.get('company_name', '')}",
            "body": (
                f"Hi {recipient_name or 'there'},\n\n"
                f"I'm reaching out about the {job.get('job_title', '')} role at "
                f"{job.get('company_name', '')}. {candidate.get('best_achievement', '')}\n\n"
                f"Would you have 15 minutes this week to connect?\n\n"
                f"Best,\n{candidate.get('candidate_name', '')}"
            ),
            "word_count": 60,
        }


# ══════════════════════════════════════════════════════════════════════════════
# Quality Gate & Refinement
# ══════════════════════════════════════════════════════════════════════════════

async def score_letter(
    *,
    cover_letter: str,
    job: dict[str, Any],
) -> dict[str, Any]:
    """Score an existing cover letter 0-100 with actionable feedback."""
    from app.agents.cover_letter_agent.prompts import COVER_LETTER_SCORE
    from app.services.groq_service import get_groq_service

    llm = get_groq_service()
    _, user_msg = COVER_LETTER_SCORE.render(
        cover_letter=cover_letter,
        job_title=job.get("job_title", ""),
        company_name=job.get("company_name", ""),
        key_requirements=job.get("key_requirements", ""),
    )

    try:
        return await llm.complete_json(
            prompt=user_msg,
            system=COVER_LETTER_SCORE.system,
            temperature=COVER_LETTER_SCORE.temperature,
            max_tokens=COVER_LETTER_SCORE.max_tokens,
        )
    except Exception as exc:
        logger.warning("Cover letter scoring failed", error=str(exc))
        return {"overall_score": 70, "grade": "B", "send_or_revise": "revise"}


async def refine_letter(
    *,
    current_letter: str,
    feedback: str,
    job: dict[str, Any],
    primary_goal: str = "",
    secondary_goal: str = "",
) -> dict[str, Any]:
    """Iteratively improve a cover letter based on specific user feedback."""
    from app.agents.cover_letter_agent.prompts import COVER_LETTER_REFINE
    from app.services.groq_service import get_groq_service

    llm = get_groq_service()
    _, user_msg = COVER_LETTER_REFINE.render(
        current_letter=current_letter,
        feedback=feedback,
        primary_goal=primary_goal or "address the feedback",
        secondary_goal=secondary_goal or "maintain authentic voice",
        company_name=job.get("company_name", ""),
        job_title=job.get("job_title", ""),
        key_requirements=job.get("key_requirements", ""),
    )

    try:
        return await llm.complete_json(
            prompt=user_msg,
            system=COVER_LETTER_REFINE.system,
            temperature=COVER_LETTER_REFINE.temperature,
            max_tokens=COVER_LETTER_REFINE.max_tokens,
        )
    except Exception as exc:
        logger.warning("Cover letter refinement failed", error=str(exc))
        return {"refined_letter": current_letter, "changes_made": []}


# ══════════════════════════════════════════════════════════════════════════════
# Tone Selection Heuristic
# ══════════════════════════════════════════════════════════════════════════════

def pick_tone_for_seniority(job: Any) -> str:
    """
    Cheap, deterministic tone pre-selection based on job title/level signals
    before any LLM call. Used as a sensible default the user can override.
    """
    title = (getattr(job, "title", "") or "").lower()
    level = (getattr(job, "experience_level", "") or "").lower()

    if any(k in title for k in ("vp ", "vice president", "chief", "cto", "ceo", "director")):
        return "formal"
    if level in ("lead", "executive"):
        return "formal"
    if any(k in title for k in ("startup", "founding")):
        return "enthusiastic"
    return "professional"


# ══════════════════════════════════════════════════════════════════════════════
# Internal Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _assemble_fallback(
    result: dict[str, Any],
    *,
    hook_key: str = "opening_hook",
    body_key: str = "body_paragraph_1",
) -> str:
    """
    Defensive assembly of full_body if the LLM didn't populate it directly.
    Stitches together whatever paragraph fields ARE present.
    """
    parts = [
        result.get("salutation", ""),
        result.get(hook_key, ""),
        result.get(body_key, ""),
        result.get("body_paragraph_2", ""),
        result.get("closing", ""),
        result.get("sign_off", ""),
    ]
    return "\n\n".join(p for p in parts if p)


def _fallback_letter(candidate: dict[str, Any], job: dict[str, Any]) -> dict[str, Any]:
    """Last-resort hardcoded letter if every LLM call fails — never leave the user with nothing."""
    name = candidate.get("candidate_name", "Candidate")
    title = job.get("job_title", "this role")
    company = job.get("company_name", "your company")
    body = (
        f"Dear Hiring Manager,\n\n"
        f"I'm writing to express my interest in the {title} position at {company}. "
        f"With {candidate.get('experience_years', 0)} years of experience and skills in "
        f"{', '.join(candidate.get('top_skills', [])[:3])}, I believe I would be a strong "
        f"addition to your team.\n\n"
        f"I would welcome the opportunity to discuss how my background aligns with your needs.\n\n"
        f"Best regards,\n{name}"
    )
    return {
        "subject_line": f"Application: {title} — {name}",
        "full_body": body,
        "word_count": len(body.split()),
        "tone_achieved": "professional",
    }


__all__ = [
    "extract_candidate_brief",
    "extract_job_brief",
    "generate_standard_letter",
    "generate_referral_letter",
    "generate_career_change_letter",
    "generate_executive_letter",
    "generate_cold_email",
    "score_letter",
    "refine_letter",
    "pick_tone_for_seniority",
]