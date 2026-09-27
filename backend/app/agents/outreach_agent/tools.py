"""
CareerGPT — Outreach Agent Tools
===================================
PAGE SUMMARY:
  Tool functions for the OutreachAgent. Each function wraps one outreach
  prompt variant in a clean async call with input normalization,
  graceful fallback text, and typed return shapes.

  TOOLS:
    generate_linkedin_connection_note()  → 300-char LinkedIn connection note
    generate_linkedin_inmessage()        → longer LinkedIn InMail after connect
    generate_recruiter_email()           → cold email to recruiter/hiring manager
    generate_followup_email()            → follow-up after no response (1/2/3)
    generate_interview_thankyou()        → thank-you note after interview round
    generate_referral_request()          → ask a contact for internal referral
    generate_offer_negotiation()         → professional counter-offer email
    generate_full_sequence()             → 3-touch outreach sequence in one call
    extract_candidate_outreach_brief()   → pull fields off Resume ORM object
    extract_job_outreach_brief()         → pull fields off Job ORM object

  ALL TOOLS RETURN AT MINIMUM:
    {"message": str} or {"body": str} — the outreach text itself.
    Additional fields (word_count, subject, cta, etc.) are informational.

  CONTRACT WITH ApplicationService:
    ApplicationService._generate_linkedin_message() calls
      OutreachAgent.generate_linkedin_message(resume=, job=)
        which calls tools.generate_linkedin_connection_note() here.
    ApplicationService._generate_recruiter_email() calls
      OutreachAgent.generate_recruiter_email(resume=, job=)
        which calls tools.generate_recruiter_email() here.
"""

from __future__ import annotations

import json
from typing import Any

from app.core.logging import logger


def extract_candidate_outreach_brief(resume: Any) -> dict[str, Any]:
    """Pull the fields outreach prompts need off a Resume ORM object."""
    skills: list[str] = []
    if resume.skills:
        try:
            skills = json.loads(resume.skills)
        except Exception:
            pass

    return {
        "sender_name":       resume.name or "Candidate",
        "sender_title":      "",
        "sender_role":       "",
        "key_achievement":   resume.summary[:180] if resume.summary else "",
        "experience_summary": (resume.raw_text or "")[:400],
        "experience_years":  resume.experience_years or 0.0,
        "primary_skills":    skills[:6],
        "top_skills":        ", ".join(skills[:6]),
        "linkedin_url":      resume.linkedin_url or "",
    }


def extract_job_outreach_brief(job: Any) -> dict[str, Any]:
    """Pull the fields outreach prompts need off a Job ORM object."""
    return {
        "company_name":    job.company,
        "target_role":     job.title,
        "job_reference":   job.job_url or f"Job ID: {job.id}",
        "company_context": job.ai_summary or "",
        "source":          job.source,
    }


# ══════════════════════════════════════════════════════════════════════════════
# LinkedIn Connection Note (300 chars max)
# ══════════════════════════════════════════════════════════════════════════════

async def generate_linkedin_connection_note(
    *,
    candidate: dict[str, Any],
    job: dict[str, Any],
    recipient_name: str = "",
    recipient_title: str = "",
    specific_hook: str = "",
    mutual_connection: str = "",
) -> dict[str, Any]:
    """
    Generate a LinkedIn connection request note.
    LinkedIn enforces 300 character limit — the LLM prompt enforces this too.
    """
    from app.agents.outreach_agent.prompts import LINKEDIN_CONNECTION_REQUEST
    from app.services.groq_service import get_groq_service

    llm = get_groq_service()
    _, user_msg = LINKEDIN_CONNECTION_REQUEST.render(
        sender_name=candidate.get("sender_name", ""),
        sender_role=candidate.get("sender_role", "") or candidate.get("top_skills", ""),
        sender_experience=candidate.get("experience_years", 0),
        recipient_name=recipient_name or "there",
        recipient_title=recipient_title or "Recruiter",
        company_name=job.get("company_name", ""),
        target_role=job.get("target_role", ""),
        specific_hook=specific_hook or job.get("company_context", ""),
        mutual_connection=mutual_connection or "none",
    )

    try:
        result = await llm.complete_json(
            prompt=user_msg,
            system=LINKEDIN_CONNECTION_REQUEST.system,
            temperature=LINKEDIN_CONNECTION_REQUEST.temperature,
            max_tokens=LINKEDIN_CONNECTION_REQUEST.max_tokens,
        )
        msg = result.get("message", "")
        if len(msg) > 300:
            msg = msg[:297] + "..."
        result["message"] = msg
        return result
    except Exception as exc:
        logger.warning("LinkedIn connection note failed", error=str(exc))
        fallback = (
            f"Hi {recipient_name or 'there'}, I noticed {job.get('company_name', '')} is hiring "
            f"for {job.get('target_role', 'a role')} and would love to connect. "
            f"—{candidate.get('sender_name', '')}"
        )
        return {
            "message": fallback[:300],
            "character_count": len(fallback),
            "within_limit": True,
        }


# ══════════════════════════════════════════════════════════════════════════════
# LinkedIn InMail (longer, after connection accepted)
# ══════════════════════════════════════════════════════════════════════════════

async def generate_linkedin_inmessage(
    *,
    candidate: dict[str, Any],
    job: dict[str, Any],
    recipient_name: str = "",
    recipient_title: str = "",
    specific_hook: str = "",
) -> dict[str, Any]:
    """Generate a LinkedIn InMail message — used after connection is accepted."""
    from app.agents.outreach_agent.prompts import LINKEDIN_INMESSAGE
    from app.services.groq_service import get_groq_service

    llm = get_groq_service()
    _, user_msg = LINKEDIN_INMESSAGE.render(
        sender_name=candidate.get("sender_name", ""),
        sender_role=candidate.get("sender_role", "") or "Software Professional",
        key_achievement=candidate.get("key_achievement", ""),
        experience_summary=candidate.get("experience_summary", ""),
        top_skills=candidate.get("top_skills", ""),
        recipient_name=recipient_name or "there",
        recipient_title=recipient_title or "Recruiter",
        company_name=job.get("company_name", ""),
        specific_hook=specific_hook or job.get("company_context", ""),
        job_title=job.get("target_role", ""),
        job_reference=job.get("job_reference", ""),
        why_company=job.get("company_context", ""),
    )

    try:
        return await llm.complete_json(
            prompt=user_msg,
            system=LINKEDIN_INMESSAGE.system,
            temperature=LINKEDIN_INMESSAGE.temperature,
            max_tokens=LINKEDIN_INMESSAGE.max_tokens,
        )
    except Exception as exc:
        logger.warning("LinkedIn InMail generation failed", error=str(exc))
        return {
            "subject": f"{job.get('target_role', 'Role')} Interest — {candidate.get('sender_name', '')}",
            "body": (
                f"Hi {recipient_name or 'there'},\n\nI came across the {job.get('target_role', '')} "
                f"role at {job.get('company_name', '')} and I'm very interested. "
                f"With experience in {candidate.get('top_skills', '')}, I believe I'd be a strong fit.\n\n"
                f"Would you have 15 minutes to connect this week?\n\nBest,\n{candidate.get('sender_name', '')}"
            ),
        }


# ══════════════════════════════════════════════════════════════════════════════
# Recruiter Cold Email
# ══════════════════════════════════════════════════════════════════════════════

async def generate_recruiter_email(
    *,
    candidate: dict[str, Any],
    job: dict[str, Any],
    recipient_name: str = "",
    recipient_title: str = "",
    company_context: str = "",
) -> dict[str, Any]:
    """Generate a cold email to a recruiter or hiring manager."""
    from app.agents.outreach_agent.prompts import RECRUITER_COLD_EMAIL
    from app.services.groq_service import get_groq_service

    llm = get_groq_service()
    _, user_msg = RECRUITER_COLD_EMAIL.render(
        sender_name=candidate.get("sender_name", ""),
        sender_title=candidate.get("sender_title", "") or candidate.get("top_skills", ""),
        experience_years=candidate.get("experience_years", 0),
        key_achievement=candidate.get("key_achievement", ""),
        primary_skills=candidate.get("top_skills", ""),
        recipient_name=recipient_name or "there",
        recipient_title=recipient_title or "Hiring Manager",
        company_name=job.get("company_name", ""),
        company_context=company_context or job.get("company_context", ""),
        target_role=job.get("target_role", ""),
        job_reference=job.get("job_reference", ""),
        why_company=job.get("company_context", ""),
    )

    try:
        return await llm.complete_json(
            prompt=user_msg,
            system=RECRUITER_COLD_EMAIL.system,
            temperature=RECRUITER_COLD_EMAIL.temperature,
            max_tokens=RECRUITER_COLD_EMAIL.max_tokens,
        )
    except Exception as exc:
        logger.warning("Recruiter email generation failed", error=str(exc))
        return {
            "subject": f"{candidate.get('experience_years', 0)} YOE Engineer — {job.get('target_role', 'Role')} Interest",
            "body": (
                f"Hi {recipient_name or 'there'},\n\n"
                f"I'm reaching out about the {job.get('target_role', '')} role at "
                f"{job.get('company_name', '')}.\n\n"
                f"{candidate.get('key_achievement', '')}\n\n"
                f"Would you have 15 minutes this week?\n\n"
                f"Best,\n{candidate.get('sender_name', '')}"
            ),
            "word_count": 60,
        }


# ══════════════════════════════════════════════════════════════════════════════
# Follow-up Email
# ══════════════════════════════════════════════════════════════════════════════

async def generate_followup_email(
    *,
    candidate: dict[str, Any],
    job: dict[str, Any],
    applied_role: str,
    original_date: str,
    days_since: int,
    followup_number: int = 1,
    new_value_point: str = "",
) -> dict[str, Any]:
    """
    Generate a follow-up email for a stale job application.
    followup_number: 1 = gentle bump, 2 = new angle, 3 = graceful exit.
    """
    from app.agents.outreach_agent.prompts import FOLLOWUP_EMAIL_PROMPT
    from app.services.groq_service import get_groq_service

    llm = get_groq_service()
    _, user_msg = FOLLOWUP_EMAIL_PROMPT.render(
        sender_name=candidate.get("sender_name", ""),
        applied_role=applied_role,
        company_name=job.get("company_name", ""),
        original_date=original_date,
        days_since=days_since,
        followup_number=followup_number,
        new_value_point=new_value_point or "continuing to grow my skills in this area",
    )

    try:
        return await llm.complete_json(
            prompt=user_msg,
            system=FOLLOWUP_EMAIL_PROMPT.system,
            temperature=FOLLOWUP_EMAIL_PROMPT.temperature,
            max_tokens=FOLLOWUP_EMAIL_PROMPT.max_tokens,
        )
    except Exception as exc:
        logger.warning("Follow-up email generation failed", error=str(exc))
        return {
            "subject": f"Re: {applied_role} Application — {candidate.get('sender_name', '')}",
            "body": (
                f"Hi there,\n\nI wanted to follow up on my {applied_role} application "
                f"submitted {days_since} days ago.\n\n"
                f"I remain very interested in this opportunity and would welcome "
                f"any update on the status.\n\n"
                f"Best,\n{candidate.get('sender_name', '')}"
            ),
            "word_count": 55,
        }


# ══════════════════════════════════════════════════════════════════════════════
# Interview Thank-You
# ══════════════════════════════════════════════════════════════════════════════

async def generate_interview_thankyou(
    *,
    candidate: dict[str, Any],
    job: dict[str, Any],
    interviewer_name: str,
    interviewer_title: str = "",
    interview_date: str = "",
    specific_discussion_point: str = "",
    concern_to_address: str = "",
    additional_value: str = "",
) -> dict[str, Any]:
    """Generate a post-interview thank-you note. Send within 2-4 hours of interview."""
    from app.agents.outreach_agent.prompts import INTERVIEW_THANKYOU_PROMPT
    from app.services.groq_service import get_groq_service

    llm = get_groq_service()
    _, user_msg = INTERVIEW_THANKYOU_PROMPT.render(
        candidate_name=candidate.get("sender_name", ""),
        interviewer_name=interviewer_name,
        interviewer_title=interviewer_title,
        company_name=job.get("company_name", ""),
        job_title=job.get("target_role", ""),
        interview_date=interview_date or "today",
        specific_discussion_point=specific_discussion_point or "our technical discussion",
        concern_to_address=concern_to_address or "none raised",
        additional_value=additional_value or "none",
    )

    try:
        return await llm.complete_json(
            prompt=user_msg,
            system=INTERVIEW_THANKYOU_PROMPT.system,
            temperature=INTERVIEW_THANKYOU_PROMPT.temperature,
            max_tokens=INTERVIEW_THANKYOU_PROMPT.max_tokens,
        )
    except Exception as exc:
        logger.warning("Thank-you note generation failed", error=str(exc))
        return {
            "subject": f"Thank you — {job.get('target_role', '')} interview | {candidate.get('sender_name', '')}",
            "body": (
                f"Hi {interviewer_name},\n\nThank you for taking the time to speak with me "
                f"today about the {job.get('target_role', '')} role. I enjoyed our conversation "
                f"and remain very excited about the opportunity.\n\n"
                f"Best,\n{candidate.get('sender_name', '')}"
            ),
            "word_count": 45,
        }


# ══════════════════════════════════════════════════════════════════════════════
# Referral Request
# ══════════════════════════════════════════════════════════════════════════════

async def generate_referral_request(
    *,
    candidate: dict[str, Any],
    job: dict[str, Any],
    contact_name: str,
    relationship: str = "former colleague",
    requester_pitch: str = "",
) -> dict[str, Any]:
    """Ask a contact to refer the candidate to their company's job opening."""
    from app.agents.outreach_agent.prompts import REFERRAL_REQUEST_PROMPT
    from app.services.groq_service import get_groq_service

    llm = get_groq_service()
    _, user_msg = REFERRAL_REQUEST_PROMPT.render(
        requester_name=candidate.get("sender_name", ""),
        contact_name=contact_name,
        relationship=relationship,
        company_name=job.get("company_name", ""),
        job_title=job.get("target_role", ""),
        job_url=job.get("job_reference", ""),
        requester_pitch=requester_pitch or (
            f"{candidate.get('experience_years', 0)} years experience in "
            f"{candidate.get('top_skills', '')}. {candidate.get('key_achievement', '')}"
        ),
    )

    try:
        return await llm.complete_json(
            prompt=user_msg,
            system=REFERRAL_REQUEST_PROMPT.system,
            temperature=REFERRAL_REQUEST_PROMPT.temperature,
            max_tokens=REFERRAL_REQUEST_PROMPT.max_tokens,
        )
    except Exception as exc:
        logger.warning("Referral request generation failed", error=str(exc))
        return {
            "subject": f"Quick favor — {job.get('company_name', '')} referral?",
            "body": (
                f"Hi {contact_name},\n\nHope you're doing well! I noticed "
                f"{job.get('company_name', '')} is hiring for {job.get('target_role', '')} "
                f"and I'm very interested. Would you be comfortable referring me? "
                f"No worries if it's not a fit.\n\nThanks,\n{candidate.get('sender_name', '')}"
            ),
        }


# ══════════════════════════════════════════════════════════════════════════════
# Offer Negotiation
# ══════════════════════════════════════════════════════════════════════════════

async def generate_offer_negotiation(
    *,
    candidate: dict[str, Any],
    job: dict[str, Any],
    offered_salary: int,
    offered_equity: str = "",
    target_salary: int = 0,
    market_data: str = "",
    competing_offer: str = "",
    priority_1: str = "higher base salary",
    priority_2: str = "more equity",
    priority_3: str = "additional vacation days",
) -> dict[str, Any]:
    """Generate a professional job offer negotiation response."""
    from app.agents.outreach_agent.prompts import OFFER_NEGOTIATION_PROMPT
    from app.services.groq_service import get_groq_service

    llm = get_groq_service()
    _, user_msg = OFFER_NEGOTIATION_PROMPT.render(
        candidate_name=candidate.get("sender_name", ""),
        company_name=job.get("company_name", ""),
        job_title=job.get("target_role", ""),
        offered_salary=f"${offered_salary:,}",
        offered_equity=offered_equity or "not specified",
        target_salary=f"${target_salary:,}" if target_salary else "market rate",
        market_data=market_data or f"Similar roles at comparable companies: ${offered_salary:,}-${offered_salary + 20000:,}",
        competing_offer=competing_offer or "none",
        priority_1=priority_1,
        priority_2=priority_2,
        priority_3=priority_3,
    )

    try:
        return await llm.complete_json(
            prompt=user_msg,
            system=OFFER_NEGOTIATION_PROMPT.system,
            temperature=OFFER_NEGOTIATION_PROMPT.temperature,
            max_tokens=OFFER_NEGOTIATION_PROMPT.max_tokens,
        )
    except Exception as exc:
        logger.warning("Offer negotiation generation failed", error=str(exc))
        return {
            "subject": f"Re: {job.get('target_role', '')} Offer — {candidate.get('sender_name', '')}",
            "body": (
                f"Thank you so much for the offer. I'm very excited about joining "
                f"{job.get('company_name', '')}. After reviewing the details, I was hoping "
                f"we could discuss the compensation package. Based on my experience and "
                f"market research, I was expecting closer to "
                f"${target_salary:,}. Would there be flexibility there?\n\n"
                f"Best,\n{candidate.get('sender_name', '')}"
            ),
        }


# ══════════════════════════════════════════════════════════════════════════════
# Full 3-Touch Outreach Sequence
# ══════════════════════════════════════════════════════════════════════════════

async def generate_full_sequence(
    *,
    candidate: dict[str, Any],
    job: dict[str, Any],
    recipient_name: str = "",
    recipient_title: str = "",
    channel: str = "email",
) -> dict[str, Any]:
    """
    Generate a complete 3-touch outreach sequence (initial + 2 follow-ups)
    in a single LLM call for efficiency.
    """
    from app.agents.outreach_agent.prompts import OUTREACH_SEQUENCE_PROMPT
    from app.services.groq_service import get_groq_service

    llm = get_groq_service()
    _, user_msg = OUTREACH_SEQUENCE_PROMPT.render(
        sender_name=candidate.get("sender_name", ""),
        sender_title=candidate.get("sender_role", "") or "Software Professional",
        key_achievement=candidate.get("key_achievement", ""),
        experience_years=candidate.get("experience_years", 0),
        recipient_name=recipient_name or "there",
        company_name=job.get("company_name", ""),
        recipient_title=recipient_title or "Hiring Manager",
        target_role=job.get("target_role", ""),
        channel=channel,
    )

    try:
        return await llm.complete_json(
            prompt=user_msg,
            system=OUTREACH_SEQUENCE_PROMPT.system,
            temperature=OUTREACH_SEQUENCE_PROMPT.temperature,
            max_tokens=OUTREACH_SEQUENCE_PROMPT.max_tokens,
        )
    except Exception as exc:
        logger.warning("Outreach sequence generation failed", error=str(exc))
        return {"sequence": [], "total_sequence_duration_days": 15}


__all__ = [
    "extract_candidate_outreach_brief",
    "extract_job_outreach_brief",
    "generate_linkedin_connection_note",
    "generate_linkedin_inmessage",
    "generate_recruiter_email",
    "generate_followup_email",
    "generate_interview_thankyou",
    "generate_referral_request",
    "generate_offer_negotiation",
    "generate_full_sequence",
]