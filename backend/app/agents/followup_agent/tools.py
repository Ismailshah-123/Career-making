"""
CareerGPT — Followup Agent Tools
===================================
PAGE SUMMARY:
  Tool functions for the FollowupAgent. Each tool is a focused, testable
  async function. Tools call LLM prompts and return structured dicts.
  The FollowupAgent orchestrates these tools — tools never touch the DB.

  TOOLS:
    generate_followup_message()      → write follow-up #1/#2/#3 email
    plan_followup_sequence()         → plan all 3 touches upfront at apply time
    analyze_pipeline_health()        → assess user's full application pipeline
    diagnose_ghost_application()     → diagnose why application got no response
    generate_reactivation_email()    → reactivate a 30+ day cold application
    generate_interview_prep()        → create interview prep notes for a role
    analyze_job_offer()              → comprehensive offer evaluation + advice
    generate_withdrawal_email()      → professional withdrawal from consideration
    get_new_value_point()            → suggest fresh value hook for follow-up
    extract_application_brief()      → pull all relevant fields off Application ORM
    check_followup_eligibility()     → validate timing + count before generating

  USED BY: FollowupAgent (agent.py)

  VALUE POINT ROTATION LOGIC:
    Every follow-up must add NEW value — never repeat the same hook.
    The agent tracks which value points have been used via the DB.
    get_new_value_point() generates a fresh hook based on:
      1. Candidate's recent achievements (from resume)
      2. Company news / industry trends (searched if possible)
      3. New project / certification completed since applying
      4. Relevant article / blog post the candidate could reference
    This prevents follow-ups from sounding like copy-paste spam.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from app.core.logging import logger


# ══════════════════════════════════════════════════════════════════════════════
# Application Brief Extraction
# ══════════════════════════════════════════════════════════════════════════════

def extract_application_brief(
    application: Any,
    user: Any,
    resume: Any,
    job: Any,
) -> dict[str, Any]:
    """
    Pull all relevant fields off ORM objects into a flat dict for prompt rendering.
    Centralizes all ORM access so prompt functions only deal with plain dicts.
    """
    skills: list[str] = []
    if resume and resume.skills:
        try:
            skills = json.loads(resume.skills)
        except Exception:
            pass

    applied_date = ""
    days_since = 0
    if application.applied_at:
        applied_dt = application.applied_at.replace(tzinfo=UTC)
        applied_date = applied_dt.strftime("%B %d, %Y")
        days_since = (datetime.now(UTC) - applied_dt).days

    return {
        "application_id":    str(application.id),
        "candidate_name":    user.full_name or "Candidate",
        "candidate_email":   user.email or "",
        "job_title":         job.title if job else "",
        "company_name":      job.company if job else "",
        "job_url":           job.job_url if job else "",
        "company_context":   job.ai_summary if job else "",
        "source":            job.source if job else "",
        "posted_date":       job.posted_at.strftime("%B %d, %Y") if job and job.posted_at else "unknown",
        "applied_date":      applied_date,
        "days_since":        days_since,
        "followup_count":    application.followup_count or 0,
        "last_followup_at":  application.last_followup_at.isoformat() if application.last_followup_at else None,
        "status":            application.status,
        "match_score":       application.match_score or 0.0,
        "ats_score":         resume.ats_score if resume else 0.0,
        "top_skills":        ", ".join(skills[:8]),
        "resume_skills":     ", ".join(skills),
        "best_achievement":  resume.summary[:200] if resume and resume.summary else "",
        "cover_letter_text": application.cover_letter_text or "",
        "original_subject":  f"{job.title} Application — {user.full_name}" if job else "",
        "required_skills":   json.loads(job.skills_required) if job and job.skills_required else [],
    }


def check_followup_eligibility(brief: dict[str, Any]) -> dict[str, Any]:
    """
    Validate that timing and count rules allow a new follow-up.
    Returns {"eligible": bool, "reason": str, "next_allowed_day": int}.
    Rules:
      - Must be at least 7 days since application or last follow-up
      - Max 3 follow-ups total per application
      - Status must be "applied" or "pending" (not "interview"/"rejected"/etc.)
    """
    from app.core.constants import FOLLOWUP_AFTER_DAYS, ApplicationStatus

    days_since      = brief.get("days_since", 0)
    followup_count  = brief.get("followup_count", 0)
    status          = brief.get("status", "")
    last_followup   = brief.get("last_followup_at")

    # Status check
    if status not in (ApplicationStatus.APPLIED.value, ApplicationStatus.PENDING.value):
        return {
            "eligible": False,
            "reason": f"Status is '{status}' — follow-ups only for applied/pending",
            "next_allowed_day": None,
        }

    # Count check
    if followup_count >= 3:
        return {
            "eligible": False,
            "reason": "Maximum 3 follow-ups reached for this application",
            "next_allowed_day": None,
        }

    # Timing check — at least 7 days since apply
    if days_since < FOLLOWUP_AFTER_DAYS:
        return {
            "eligible": False,
            "reason": f"Only {days_since} days since applying — wait until day {FOLLOWUP_AFTER_DAYS}",
            "next_allowed_day": FOLLOWUP_AFTER_DAYS - days_since,
        }

    # Timing check — at least 7 days since last follow-up
    if last_followup:
        try:
            last_dt   = datetime.fromisoformat(last_followup).replace(tzinfo=UTC)
            days_since_last = (datetime.now(UTC) - last_dt).days
            if days_since_last < 7:
                return {
                    "eligible": False,
                    "reason": f"Last follow-up was {days_since_last} days ago — wait 7 days between follow-ups",
                    "next_allowed_day": 7 - days_since_last,
                }
        except Exception:
            pass

    return {
        "eligible": True,
        "reason": "All timing and count rules satisfied",
        "followup_number": followup_count + 1,
        "next_allowed_day": 0,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Follow-up Message Generator
# ══════════════════════════════════════════════════════════════════════════════

async def generate_followup_message(brief: dict[str, Any]) -> dict[str, Any]:
    """
    Generate a follow-up email for a stale application.
    Auto-selects follow-up #1, #2, or #3 based on followup_count.
    Each follow-up uses a different angle and new value point.
    """
    from app.agents.followup_agent.prompts import FOLLOWUP_MESSAGE_GENERATE
    from app.services.groq_service import get_groq_service

    llm = get_groq_service()
    followup_number = brief.get("followup_count", 0) + 1
    new_value = await get_new_value_point(brief, followup_number=followup_number)

    _, user_msg = FOLLOWUP_MESSAGE_GENERATE.render(
        candidate_name=brief.get("candidate_name", ""),
        job_title=brief.get("job_title", ""),
        company_name=brief.get("company_name", ""),
        applied_date=brief.get("applied_date", ""),
        days_since=brief.get("days_since", 0),
        followup_number=followup_number,
        top_skills=brief.get("top_skills", ""),
        best_achievement=brief.get("best_achievement", ""),
        new_value_point=new_value,
        original_subject=brief.get("original_subject", ""),
    )

    try:
        result = await llm.complete_json(
            prompt=user_msg,
            system=FOLLOWUP_MESSAGE_GENERATE.system,
            temperature=FOLLOWUP_MESSAGE_GENERATE.temperature,
            max_tokens=FOLLOWUP_MESSAGE_GENERATE.max_tokens,
        )
        result["followup_number"] = followup_number
        result["new_value_hook"]  = new_value
        logger.info(
            "Follow-up message generated",
            company=brief.get("company_name"),
            followup_number=followup_number,
            word_count=result.get("word_count", 0),
        )
        return result
    except Exception as exc:
        logger.warning("Follow-up generation failed", error=str(exc))
        return _fallback_followup(brief, followup_number)


def _fallback_followup(brief: dict[str, Any], followup_number: int) -> dict[str, Any]:
    """Hardcoded fallback if all LLM calls fail — user always gets something."""
    name = brief.get("candidate_name", "")
    role = brief.get("job_title", "the role")
    company = brief.get("company_name", "your company")

    bodies = {
        1: (
            f"Hi there,\n\nI wanted to follow up on my {role} application submitted "
            f"{brief.get('days_since', 7)} days ago. I remain very interested and "
            f"believe my background would be a strong fit.\n\n"
            f"Happy to provide any additional information. Best, {name}"
        ),
        2: (
            f"Hi there,\n\nBriefly following up on my {role} application at {company}. "
            f"Still very interested in this opportunity. No worries if timing isn't right. "
            f"Best, {name}"
        ),
        3: (
            f"Hi there,\n\nJust closing the loop on my {role} application. "
            f"I appreciate your time and would welcome the opportunity to connect "
            f"in the future. Best, {name}"
        ),
    }

    body = bodies.get(followup_number, bodies[1])
    return {
        "subject":        f"Re: {role} Application — {name}",
        "body":           body,
        "full_email":     body,
        "word_count":     len(body.split()),
        "followup_number": followup_number,
        "tone":           "professional",
    }


# ══════════════════════════════════════════════════════════════════════════════
# New Value Point Generator
# ══════════════════════════════════════════════════════════════════════════════

async def get_new_value_point(
    brief: dict[str, Any],
    followup_number: int = 1,
) -> str:
    """
    Generate a fresh, non-repetitive value hook for a follow-up email.
    Rotates between: achievement update, company insight, skill progression.
    """
    from app.services.groq_service import get_groq_service

    llm = get_groq_service()
    angle_by_number = {
        1: "recent_achievement",
        2: "company_insight_or_news",
        3: "graceful_closure",
    }
    angle = angle_by_number.get(followup_number, "recent_achievement")

    system = """Generate a single fresh value-add hook for a follow-up email.
Return ONE sentence (max 25 words) that adds value and justifies the follow-up.
Return ONLY plain text — no JSON, no explanation."""

    prompt = (
        f"Candidate: {brief.get('candidate_name')}, {brief.get('top_skills')}\n"
        f"Following up on: {brief.get('job_title')} at {brief.get('company_name')}\n"
        f"Follow-up #{followup_number} — angle: {angle}\n"
        f"Previous achievement: {brief.get('best_achievement', '')[:100]}\n"
        f"Generate ONE fresh value hook sentence."
    )

    try:
        result = await llm.complete(
            prompt=prompt,
            system=system,
            max_tokens=60,
            temperature=0.85,
        )
        return result.strip().strip('"').strip("'")
    except Exception:
        defaults = {
            1: f"I recently completed a project in {brief.get('top_skills', '').split(',')[0].strip()} that reinforced my interest in this role.",
            2: f"I noticed {brief.get('company_name', 'your company')} recently — it further confirmed my enthusiasm for this opportunity.",
            3: "I just wanted to close the loop professionally.",
        }
        return defaults.get(followup_number, defaults[1])


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline Health Analyzer
# ══════════════════════════════════════════════════════════════════════════════

async def analyze_pipeline_health(
    user: Any,
    pipeline_stats: dict[str, int],
    stale_applications: list[dict[str, Any]],
    total_count: int,
) -> dict[str, Any]:
    """
    Analyze the user's full application pipeline and recommend actions.
    Returns prioritized list of applications needing immediate attention.
    """
    from app.agents.followup_agent.prompts import FOLLOWUP_STRATEGY_ANALYZE
    from app.services.groq_service import get_groq_service

    llm = get_groq_service()
    _, user_msg = FOLLOWUP_STRATEGY_ANALYZE.render(
        candidate_name=user.full_name or "Candidate",
        total_count=total_count,
        pipeline_json=json.dumps(pipeline_stats, indent=2),
        stale_applications_json=json.dumps(stale_applications[:10], indent=2),
    )

    try:
        result = await llm.complete_json(
            prompt=user_msg,
            system=FOLLOWUP_STRATEGY_ANALYZE.system,
            temperature=FOLLOWUP_STRATEGY_ANALYZE.temperature,
            max_tokens=FOLLOWUP_STRATEGY_ANALYZE.max_tokens,
        )
        logger.info(
            "Pipeline health analyzed",
            health_score=result.get("pipeline_health_score"),
            urgent_count=len(result.get("urgent_actions", [])),
        )
        return result
    except Exception as exc:
        logger.warning("Pipeline analysis failed", error=str(exc))
        return {
            "pipeline_health_score": 60,
            "health_assessment": "Analysis unavailable. Review stale applications manually.",
            "urgent_actions": [],
            "pipeline_advice": f"You have {total_count} total applications. Focus on roles with high match scores.",
        }


# ══════════════════════════════════════════════════════════════════════════════
# Ghost Application Diagnosis
# ══════════════════════════════════════════════════════════════════════════════

async def diagnose_ghost_application(brief: dict[str, Any]) -> dict[str, Any]:
    """
    Diagnose why an application received zero response after 21+ days.
    Returns root cause, recovery action, and probability of late response.
    """
    from app.agents.followup_agent.prompts import FOLLOWUP_GHOST_DIAGNOSIS
    from app.services.groq_service import get_groq_service

    llm = get_groq_service()
    required_skills = brief.get("required_skills", [])
    _, user_msg = FOLLOWUP_GHOST_DIAGNOSIS.render(
        job_title=brief.get("job_title", ""),
        company_name=brief.get("company_name", ""),
        applied_date=brief.get("applied_date", ""),
        days_since=brief.get("days_since", 0),
        match_score=brief.get("match_score", 0),
        ats_score=brief.get("ats_score", 0),
        source=brief.get("source", ""),
        posted_date=brief.get("posted_date", ""),
        followup_count=brief.get("followup_count", 0),
        company_news="Not available",
        resume_skills=brief.get("resume_skills", ""),
        required_skills=", ".join(required_skills) if isinstance(required_skills, list) else str(required_skills),
    )

    try:
        result = await llm.complete_json(
            prompt=user_msg,
            system=FOLLOWUP_GHOST_DIAGNOSIS.system,
            temperature=FOLLOWUP_GHOST_DIAGNOSIS.temperature,
            max_tokens=FOLLOWUP_GHOST_DIAGNOSIS.max_tokens,
        )
        logger.info(
            "Ghost application diagnosed",
            company=brief.get("company_name"),
            root_cause=result.get("primary_diagnosis"),
        )
        return result
    except Exception as exc:
        logger.warning("Ghost diagnosis failed", error=str(exc))
        return {
            "primary_diagnosis":        "unknown",
            "diagnosis_description":    "Unable to determine root cause automatically.",
            "recovery_action":          "re_apply_tailored",
            "recovery_instructions":    "Tailor your resume with aggressive optimization and re-apply if job is still open.",
            "probability_of_late_response": 0.05,
            "time_to_move_on":          True,
        }


# ══════════════════════════════════════════════════════════════════════════════
# Reactivation Email
# ══════════════════════════════════════════════════════════════════════════════

async def generate_reactivation_email(
    brief: dict[str, Any],
    reactivation_hook: str = "",
    new_achievement: str = "",
    company_news: str = "",
) -> dict[str, Any]:
    """Generate a reactivation email for a cold application (30+ days stale)."""
    from app.agents.followup_agent.prompts import FOLLOWUP_REACTIVATE_DRAFT
    from app.services.groq_service import get_groq_service

    llm = get_groq_service()
    _, user_msg = FOLLOWUP_REACTIVATE_DRAFT.render(
        candidate_name=brief.get("candidate_name", ""),
        job_title=brief.get("job_title", ""),
        company_name=brief.get("company_name", ""),
        days_since=brief.get("days_since", 30),
        reactivation_hook=reactivation_hook or "continued interest in the role",
        new_achievement=new_achievement or brief.get("best_achievement", ""),
        company_news=company_news or "not specified",
    )

    try:
        return await llm.complete_json(
            prompt=user_msg,
            system=FOLLOWUP_REACTIVATE_DRAFT.system,
            temperature=FOLLOWUP_REACTIVATE_DRAFT.temperature,
            max_tokens=FOLLOWUP_REACTIVATE_DRAFT.max_tokens,
        )
    except Exception as exc:
        logger.warning("Reactivation email generation failed", error=str(exc))
        name = brief.get("candidate_name", "")
        role = brief.get("job_title", "the role")
        company = brief.get("company_name", "")
        return {
            "subject": f"Re: {role} Application — Update from {name}",
            "body": (
                f"Hi there,\n\nI wanted to reach back out regarding my {role} application "
                f"at {company}. I've {new_achievement or 'continued developing my skills'} "
                f"since we last connected and remain very interested in this opportunity.\n\n"
                f"Would love to reconnect if timing is right. Best, {name}"
            ),
            "word_count": 55,
        }


# ══════════════════════════════════════════════════════════════════════════════
# Interview Prep Generator
# ══════════════════════════════════════════════════════════════════════════════

async def generate_interview_prep(
    brief: dict[str, Any],
    interview_type: str = "technical",
    interviewer_info: str = "",
) -> dict[str, Any]:
    """
    Generate comprehensive interview prep notes.
    Called when application status moves to "interview".
    """
    from app.agents.followup_agent.prompts import FOLLOWUP_INTERVIEW_PREP
    from app.services.groq_service import get_groq_service

    llm = get_groq_service()
    _, user_msg = FOLLOWUP_INTERVIEW_PREP.render(
        resume_summary=f"{brief.get('top_skills')} — {brief.get('best_achievement', '')}",
        job_title=brief.get("job_title", ""),
        company_name=brief.get("company_name", ""),
        job_description=brief.get("company_context", ""),
        interview_type=interview_type,
        interviewer_info=interviewer_info or "not specified",
    )

    try:
        result = await llm.complete_json(
            prompt=user_msg,
            system=FOLLOWUP_INTERVIEW_PREP.system,
            temperature=FOLLOWUP_INTERVIEW_PREP.temperature,
            max_tokens=FOLLOWUP_INTERVIEW_PREP.max_tokens,
        )
        logger.info(
            "Interview prep generated",
            company=brief.get("company_name"),
            interview_type=interview_type,
        )
        return result
    except Exception as exc:
        logger.warning("Interview prep generation failed", error=str(exc))
        return {
            "interview_type": interview_type,
            "prep_time_recommended_hours": 4,
            "company_research": {"product": "Research the company website and recent news"},
            "likely_technical_questions": [],
            "likely_behavioral_questions": [],
            "questions_to_ask_interviewer": [
                "What does the first 90 days look like for this role?",
                "How does the team approach technical debt?",
                "What does success look like in year one?",
            ],
            "prep_checklist": [
                "Research company LinkedIn page",
                "Read last 3 blog posts",
                "Prepare 3 STAR behavioral stories",
                "Practice explaining your most complex project",
            ],
        }


# ══════════════════════════════════════════════════════════════════════════════
# Offer Analysis
# ══════════════════════════════════════════════════════════════════════════════

async def analyze_job_offer(
    brief: dict[str, Any],
    *,
    base_salary: int,
    equity_details: str = "",
    signing_bonus: int = 0,
    annual_bonus: str = "0%",
    benefits_summary: str = "",
    remote_policy: str = "",
    start_date: str = "",
    current_salary: int = 0,
    target_salary: int = 0,
    other_offers: str = "none",
    market_salary_range: str = "",
) -> dict[str, Any]:
    """Comprehensive job offer analysis with negotiation recommendations."""
    from app.agents.followup_agent.prompts import FOLLOWUP_OFFER_ANALYSIS
    from app.services.groq_service import get_groq_service

    llm = get_groq_service()
    _, user_msg = FOLLOWUP_OFFER_ANALYSIS.render(
        candidate_name=brief.get("candidate_name", ""),
        current_salary=f"${current_salary:,}" if current_salary else "not specified",
        target_salary=f"${target_salary:,}" if target_salary else "not specified",
        career_goal="grow as a software engineering professional",
        other_offers=other_offers,
        company_name=brief.get("company_name", ""),
        job_title=brief.get("job_title", ""),
        base_salary=f"${base_salary:,}",
        equity_details=equity_details or "not specified",
        signing_bonus=f"${signing_bonus:,}" if signing_bonus else "none",
        annual_bonus=annual_bonus,
        benefits_summary=benefits_summary or "standard package",
        remote_policy=remote_policy or "not specified",
        start_date=start_date or "flexible",
        market_salary_range=market_salary_range or "market competitive",
    )

    try:
        result = await llm.complete_json(
            prompt=user_msg,
            system=FOLLOWUP_OFFER_ANALYSIS.system,
            temperature=FOLLOWUP_OFFER_ANALYSIS.temperature,
            max_tokens=FOLLOWUP_OFFER_ANALYSIS.max_tokens,
        )
        logger.info(
            "Offer analyzed",
            company=brief.get("company_name"),
            base=base_salary,
            recommendation=result.get("accept_recommendation"),
        )
        return result
    except Exception as exc:
        logger.warning("Offer analysis failed", error=str(exc))
        return {
            "total_comp_year_1": base_salary + (signing_bonus // 4),
            "offer_strength": "unknown",
            "negotiation_recommended": True,
            "accept_recommendation": "negotiate_then_decide",
            "final_verdict": "Consult with a career advisor for a comprehensive offer analysis.",
        }


# ══════════════════════════════════════════════════════════════════════════════
# Withdrawal Email
# ══════════════════════════════════════════════════════════════════════════════

async def generate_withdrawal_email(
    brief: dict[str, Any],
    withdrawal_reason: str = "accepted another offer",
    contact_name: str = "",
) -> dict[str, Any]:
    """Generate a professional withdrawal-from-consideration email."""
    from app.agents.followup_agent.prompts import FOLLOWUP_WITHDRAWAL_DRAFT
    from app.services.groq_service import get_groq_service

    llm = get_groq_service()
    _, user_msg = FOLLOWUP_WITHDRAWAL_DRAFT.render(
        candidate_name=brief.get("candidate_name", ""),
        job_title=brief.get("job_title", ""),
        company_name=brief.get("company_name", ""),
        withdrawal_reason=withdrawal_reason,
        contact_name=contact_name or "Hiring Manager",
    )

    try:
        return await llm.complete_json(
            prompt=user_msg,
            system=FOLLOWUP_WITHDRAWAL_DRAFT.system,
            temperature=FOLLOWUP_WITHDRAWAL_DRAFT.temperature,
            max_tokens=FOLLOWUP_WITHDRAWAL_DRAFT.max_tokens,
        )
    except Exception as exc:
        logger.warning("Withdrawal email generation failed", error=str(exc))
        name = brief.get("candidate_name", "")
        role = brief.get("job_title", "the role")
        company = brief.get("company_name", "")
        return {
            "subject": f"Re: {role} Application — Withdrawal",
            "body": (
                f"Hi {contact_name or 'there'},\n\nThank you so much for considering me "
                f"for the {role} position at {company}. After careful consideration, "
                f"I need to withdraw my application at this time. I hope we'll have the "
                f"opportunity to connect in the future. Best wishes, {name}"
            ),
            "word_count": 52,
            "keeps_door_open": True,
        }


# ══════════════════════════════════════════════════════════════════════════════
# Follow-up Sequence Planner
# ══════════════════════════════════════════════════════════════════════════════

async def plan_followup_sequence(brief: dict[str, Any]) -> dict[str, Any]:
    """
    Plan all 3 follow-up touches upfront when an application is created.
    Enables automatic scheduling without repeated LLM calls later.
    """
    from app.agents.followup_agent.prompts import FOLLOWUP_SEQUENCE_PLAN
    from app.services.groq_service import get_groq_service

    llm = get_groq_service()
    _, user_msg = FOLLOWUP_SEQUENCE_PLAN.render(
        candidate_name=brief.get("candidate_name", ""),
        job_title=brief.get("job_title", ""),
        company_name=brief.get("company_name", ""),
        applied_date=brief.get("applied_date", datetime.now(UTC).strftime("%B %d, %Y")),
        match_score=brief.get("match_score", 0.7),
        key_strengths=brief.get("top_skills", ""),
        company_context=brief.get("company_context", ""),
    )

    try:
        return await llm.complete_json(
            prompt=user_msg,
            system=FOLLOWUP_SEQUENCE_PLAN.system,
            temperature=FOLLOWUP_SEQUENCE_PLAN.temperature,
            max_tokens=FOLLOWUP_SEQUENCE_PLAN.max_tokens,
        )
    except Exception as exc:
        logger.warning("Sequence planning failed, using defaults", error=str(exc))
        return {
            "sequence": [
                {"touch_number": 1, "send_on_day": 7,  "angle": "value_add",       "auto_schedule": True},
                {"touch_number": 2, "send_on_day": 14, "angle": "company_insight",  "auto_schedule": True},
                {"touch_number": 3, "send_on_day": 21, "angle": "graceful_exit",    "auto_schedule": True},
            ],
            "total_sequence_days": 21,
            "expected_response_probability": 0.45,
        }


__all__ = [
    "extract_application_brief",
    "check_followup_eligibility",
    "generate_followup_message",
    "get_new_value_point",
    "plan_followup_sequence",
    "analyze_pipeline_health",
    "diagnose_ghost_application",
    "generate_reactivation_email",
    "generate_interview_prep",
    "analyze_job_offer",
    "generate_withdrawal_email",
]