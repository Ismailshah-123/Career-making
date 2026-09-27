"""
CareerGPT — Followup Agent Prompts
=====================================
PAGE SUMMARY:
  All LLM prompts used by the FollowupAgent for generating stale-application
  follow-up messages, analyzing application pipeline health, suggesting
  intervention strategies, and creating re-engagement content for ghosted
  applications.

  PROMPTS IN THIS FILE:
    FOLLOWUP_MESSAGE_GENERATE   → main follow-up email/message generator
    FOLLOWUP_STRATEGY_ANALYZE   → analyze which applications need attention
    FOLLOWUP_SEQUENCE_PLAN      → plan a 3-touch follow-up sequence upfront
    FOLLOWUP_GHOST_DIAGNOSIS    → diagnose why an application got no response
    FOLLOWUP_REACTIVATE_DRAFT   → reactivate a cold application after 30+ days
    FOLLOWUP_INTERVIEW_PREP     → generate interview prep notes from job analysis
    FOLLOWUP_OFFER_ANALYSIS     → analyze a received job offer comprehensively
    FOLLOWUP_WITHDRAWAL_DRAFT   → professional withdrawal from consideration

  USED BY: FollowupAgent (agent.py) via tools.py

  FOLLOW-UP TIMING SCIENCE:
    Research data on email follow-up response rates:
      Day 0 (apply):            0% response (just submitted)
      Day 7 (follow-up #1):    42% of total responses come after this
      Day 14 (follow-up #2):   28% of total responses come after this
      Day 21 (follow-up #3):   15% of total responses come after this
      Day 28+ (follow-up #4+): < 5% — stop following up, move on

    The FollowupAgent enforces these timing windows.
    FOLLOWUP_AFTER_DAYS = 7 is set in app/core/constants.py.
    MAX 3 follow-ups per application — hardcoded safety limit.

  GHOST DIAGNOSIS:
    When an application gets zero response for 21+ days, the agent
    diagnoses likely reasons and suggests interventions:
      - Job filled internally
      - ATS resume never parsed correctly (false negative)
      - Missing required keyword → filtered at screening
      - Salary mismatch → auto-rejected
      - Application volume too high → not reviewed yet
    Each diagnosis comes with a specific recovery action.
"""

from __future__ import annotations

from app.prompts.resume_prompts import Prompt


# ══════════════════════════════════════════════════════════════════════════════
# MAIN FOLLOW-UP MESSAGE GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

FOLLOWUP_MESSAGE_GENERATE = Prompt(
    name="followup_message_generate",
    version="v3",
    temperature=0.70,
    max_tokens=600,
    system="""You write job application follow-up emails that get responses.
You have a 42% response rate on first follow-ups compared to the industry average of 12%.

FOLLOW-UP RULES BY NUMBER:
  #1 (Day 7):  Short, professional. Reference application. Add ONE new value point.
               Never apologetic. Confident and brief (< 80 words body).
  #2 (Day 14): Even shorter. Different angle. Mention company news or achievement.
               End with easy out: "If timing isn't right, no worries."
  #3 (Day 21): Graceful exit email. Thank them for their time. Leave door open.
               < 60 words. No questions. Just closes the loop professionally.

NEVER SAY: "I hope this finds you well", "circling back", "just checking in",
           "I know you're busy", "I don't want to bother you", "any updates?"

ALWAYS INCLUDE: ONE new piece of value (achievement, company insight, project update)
SUBJECT LINE: Always reply to the original thread (Re: original subject)

Return ONLY valid JSON.""",
    user_template="""Write a follow-up email for this application:

CANDIDATE: $candidate_name
ROLE APPLIED FOR: $job_title at $company_name
APPLICATION DATE: $applied_date
DAYS SINCE APPLYING: $days_since
FOLLOW-UP NUMBER: $followup_number (1, 2, or 3)

CANDIDATE'S TOP SKILLS: $top_skills
CANDIDATE'S BEST ACHIEVEMENT: $best_achievement

NEW VALUE POINT TO ADD THIS FOLLOW-UP:
$new_value_point

ORIGINAL EMAIL SUBJECT (for Re: threading):
$original_subject

Return:
{
  "subject": "Re: $job_title Application — $candidate_name",
  "salutation": "Hi [Name],",
  "body": "Complete email body (under 80 words for #1, 60 words for #2/#3)",
  "sign_off": "Best regards,",
  "full_email": "Complete assembled email as single string",
  "word_count": 67,
  "new_value_used": "What new value was added vs the original email",
  "tone": "professional_confident",
  "followup_strategy": "value_add",
  "next_followup_day": 14,
  "send_time_recommendation": "Tuesday-Thursday, 8-10am recipient timezone"
}""",
)


# ══════════════════════════════════════════════════════════════════════════════
# APPLICATION PIPELINE HEALTH ANALYZER
# ══════════════════════════════════════════════════════════════════════════════

FOLLOWUP_STRATEGY_ANALYZE = Prompt(
    name="followup_strategy_analyze",
    version="v2",
    temperature=0.20,
    max_tokens=800,
    system="""You are a career strategist analyzing a job seeker's application pipeline.
Identify which applications need immediate attention, which are healthy,
and which should be abandoned. Provide specific, prioritized action items.

PRIORITIZE BY:
  1. High match score + recent posting → push hard for response
  2. Interview stage → always follow up within 24h of any interaction
  3. Stale 7-14 days → send follow-up #1
  4. Stale 14-21 days → send follow-up #2
  5. Stale 21+ days with no response → consider withdrawal or final note
  6. Rejected → send graceful thank-you, ask to be kept in mind for future

Return ONLY valid JSON.""",
    user_template="""Analyze this application pipeline and recommend actions:

CANDIDATE: $candidate_name
TOTAL APPLICATIONS: $total_count

PIPELINE BREAKDOWN:
$pipeline_json

APPLICATIONS NEEDING ATTENTION (stale, no response):
$stale_applications_json

Return:
{
  "pipeline_health_score": 72,
  "health_assessment": "Active pipeline with good mix. 3 applications need immediate follow-up.",
  "urgent_actions": [
    {
      "application_id": "uuid",
      "company": "Stripe",
      "role": "Senior Backend Engineer",
      "action": "send_followup_1",
      "reason": "7 days since application, strong 87% match — don't let this go cold",
      "priority": "high",
      "deadline": "Today"
    }
  ],
  "healthy_applications": ["uuid1", "uuid2"],
  "abandon_recommendations": [
    {
      "application_id": "uuid",
      "reason": "28 days no response, company announced hiring freeze last week"
    }
  ],
  "weekly_apply_target": 8,
  "pipeline_advice": "Your response rate is 18% — industry average is 10%. Keep applying at this pace.",
  "skill_gaps_costing_interviews": ["Kubernetes", "Terraform"],
  "top_performing_job_source": "remoteok"
}""",
)


# ══════════════════════════════════════════════════════════════════════════════
# FOLLOW-UP SEQUENCE PLANNER
# Plan full 3-touch sequence at application creation time
# ══════════════════════════════════════════════════════════════════════════════

FOLLOWUP_SEQUENCE_PLAN = Prompt(
    name="followup_sequence_plan",
    version="v1",
    temperature=0.30,
    max_tokens=700,
    system="""You create a proactive 3-touch follow-up plan at the time of application.
Planning upfront means follow-ups are sent at optimal times automatically.
Each touch should have a DIFFERENT angle and add NEW value.
Return ONLY valid JSON.""",
    user_template="""Plan a 3-touch follow-up sequence for this application:

CANDIDATE: $candidate_name
ROLE: $job_title at $company_name
APPLIED DATE: $applied_date
MATCH SCORE: $match_score
CANDIDATE KEY STRENGTHS: $key_strengths
COMPANY CONTEXT: $company_context

Return:
{
  "sequence": [
    {
      "touch_number": 1,
      "send_on_day": 7,
      "send_date": "2024-01-15",
      "angle": "value_add",
      "new_value_hook": "Just shipped a FastAPI performance improvement that reduced our API response time by 40%",
      "email_preview": "Hi [Name], following up on my Senior Engineer application...",
      "subject": "Re: Senior Backend Engineer Application",
      "recommended_send_time": "Tuesday 9am"
    },
    {
      "touch_number": 2,
      "send_on_day": 14,
      "send_date": "2024-01-22",
      "angle": "company_insight",
      "new_value_hook": "Noticed your team just announced the new AI features — my background in LLM integration is directly relevant",
      "email_preview": "Hi [Name], your recent announcement about...",
      "subject": "Re: Senior Backend Engineer Application",
      "recommended_send_time": "Wednesday 8:30am"
    },
    {
      "touch_number": 3,
      "send_on_day": 21,
      "send_date": "2024-01-29",
      "angle": "graceful_exit",
      "new_value_hook": "Keeping it brief — just wanted to close the loop professionally",
      "email_preview": "Hi [Name], I understand timing may not be right...",
      "subject": "Re: Senior Backend Engineer Application",
      "recommended_send_time": "Thursday 9am"
    }
  ],
  "total_sequence_days": 21,
  "expected_response_probability": 0.58,
  "auto_schedule": true
}""",
)


# ══════════════════════════════════════════════════════════════════════════════
# GHOST DIAGNOSIS
# Why did this application get no response?
# ══════════════════════════════════════════════════════════════════════════════

FOLLOWUP_GHOST_DIAGNOSIS = Prompt(
    name="followup_ghost_diagnosis",
    version="v2",
    temperature=0.25,
    max_tokens=600,
    system="""You diagnose why a job application received zero response.
Be specific and honest — avoid vague answers like "keep trying".
Provide one primary cause and one concrete recovery action.
Return ONLY valid JSON.""",
    user_template="""Diagnose why this application received no response after $days_since days:

APPLICATION DETAILS:
Role: $job_title at $company_name
Applied: $applied_date
Days Since: $days_since
Match Score: $match_score
Resume ATS Score: $ats_score
Applied Via: $source
Job Posted: $posted_date
Follow-ups Sent: $followup_count
Company Recent News: $company_news

RESUME SKILLS: $resume_skills
JOB REQUIRED SKILLS: $required_skills

Return:
{
  "primary_diagnosis": "ats_keyword_miss",
  "diagnosis_description": "Your resume has 72% keyword overlap but is missing 'Kubernetes' which appears 4 times in the JD. Most ATS filter resumes that miss required keywords even once — this likely caused auto-rejection before human review.",
  "confidence": "high",
  "supporting_evidence": ["'Kubernetes' appears 4x in JD", "Your ATS score of 68 is below the typical 75+ threshold", "Job posted 35 days ago — likely already filled if no response by now"],
  "secondary_factors": ["High application volume for this role (applicant count: 847)", "Remote role attracts 3x more applicants"],
  "recovery_action": "tailor_resume",
  "recovery_instructions": "Re-tailor your resume with aggressive optimization — specifically add Kubernetes experience (even Docker Swarm counts as container orchestration) and re-apply if the job is still open.",
  "probability_of_late_response": 0.08,
  "recommendation": "Move on — low probability of response. Apply the Kubernetes lesson to the next 3 applications immediately.",
  "time_to_move_on": true
}""",
)


# ══════════════════════════════════════════════════════════════════════════════
# COLD APPLICATION REACTIVATION (30+ days stale)
# ══════════════════════════════════════════════════════════════════════════════

FOLLOWUP_REACTIVATE_DRAFT = Prompt(
    name="followup_reactivate_draft",
    version="v1",
    temperature=0.65,
    max_tokens=500,
    system="""You write reactivation emails for cold applications (30+ days old).
These are different from standard follow-ups — they need to justify why
the candidate is reaching out again after a long silence.
Good reactivation hooks: new achievement, company news, role re-posted,
mutual connection introduced, or a relevant article/event.
Return ONLY valid JSON.""",
    user_template="""Write a reactivation email for this cold application:

CANDIDATE: $candidate_name
ROLE: $job_title at $company_name
DAYS SINCE LAST CONTACT: $days_since
REACTIVATION HOOK: $reactivation_hook
NEW ACHIEVEMENT SINCE APPLYING: $new_achievement
COMPANY NEWS/TRIGGER: $company_news

Return:
{
  "subject": "Re: $job_title Application — Update from $candidate_name",
  "body": "Complete reactivation email body (under 100 words)",
  "reactivation_hook_used": "Company announced expansion into new market last week",
  "tone": "professional_refreshed",
  "word_count": 87,
  "send_recommendation": "Send Monday or Tuesday morning for best open rates"
}""",
)


# ══════════════════════════════════════════════════════════════════════════════
# INTERVIEW PREP GENERATOR
# When application moves to "interview" status
# ══════════════════════════════════════════════════════════════════════════════

FOLLOWUP_INTERVIEW_PREP = Prompt(
    name="followup_interview_prep",
    version="v2",
    temperature=0.30,
    max_tokens=1200,
    system="""You are a senior technical interview coach who has prepared 5,000+
candidates for software engineering interviews at top companies.
Create specific, actionable interview prep based on the job description and resume.
Focus on: likely questions, talking points, red flags to address, company research.
Return ONLY valid JSON.""",
    user_template="""Create interview prep for this candidate:

CANDIDATE BACKGROUND:
$resume_summary

JOB: $job_title at $company_name
JOB DESCRIPTION: $job_description

INTERVIEW TYPE: $interview_type (phone_screen / technical / system_design / behavioral / final)
INTERVIEWER INFO (if known): $interviewer_info

Return:
{
  "interview_type": "technical",
  "prep_time_recommended_hours": 4,
  "company_research": {
    "product": "What $company_name's main product does and who uses it",
    "tech_stack": ["Python", "FastAPI", "PostgreSQL", "Kubernetes"],
    "recent_news": "Series C raised in Q3, expanding AI team",
    "culture_signals": ["async-first", "strong engineering culture", "high autonomy"]
  },
  "likely_technical_questions": [
    {
      "question": "How would you design a rate limiter for our API?",
      "why_likely": "Core to their infrastructure based on JD",
      "your_talking_points": ["Redis sliding window algorithm", "Token bucket vs leaky bucket tradeoffs", "Distributed rate limiting across services"],
      "sample_answer_structure": "Start with requirements clarification, then explain Redis ZSET approach with O(log n) complexity..."
    }
  ],
  "likely_behavioral_questions": [
    {
      "question": "Tell me about a time you reduced system latency significantly.",
      "star_answer": "S: Our API was averaging 800ms... T: We needed < 200ms... A: I profiled with py-spy and found N+1 queries... R: Reduced to 85ms, 10x improvement."
    }
  ],
  "your_strengths_to_highlight": ["FastAPI expertise matches their exact stack", "PostgreSQL optimization experience"],
  "gaps_to_address_proactively": "You don't have Kubernetes production experience — acknowledge this early and explain your Docker expertise as foundation",
  "questions_to_ask_interviewer": [
    "What does the on-call rotation look like for this team?",
    "How does the team approach technical debt?",
    "What does the first 90 days look like for this role?"
  ],
  "red_flags_to_watch_for": ["Unclear team structure", "Vague answers about tech stack", "No mention of code review process"],
  "prep_checklist": ["Research CEO and CTO LinkedIn", "Read last 3 company blog posts", "Practice rate limiter design on paper", "Prepare 3 STAR stories"]
}""",
)


# ══════════════════════════════════════════════════════════════════════════════
# OFFER ANALYSIS
# Comprehensive offer evaluation when user receives an offer
# ══════════════════════════════════════════════════════════════════════════════

FOLLOWUP_OFFER_ANALYSIS = Prompt(
    name="followup_offer_analysis",
    version="v2",
    temperature=0.20,
    max_tokens=900,
    system="""You are a compensation and career strategy expert.
Analyze a job offer comprehensively — not just salary, but total compensation,
career trajectory, culture fit, and opportunity cost.
Be direct with your recommendation. Don't hedge excessively.
Return ONLY valid JSON.""",
    user_template="""Analyze this job offer:

CANDIDATE:
Current Salary: $current_salary
Target Salary: $target_salary
Career Goal: $career_goal
Other Offers: $other_offers

OFFER DETAILS:
Company: $company_name
Role: $job_title
Base Salary: $base_salary
Equity: $equity_details
Signing Bonus: $signing_bonus
Annual Bonus: $annual_bonus
Benefits: $benefits_summary
Remote Policy: $remote_policy
Start Date: $start_date

MARKET DATA:
Similar roles: $market_salary_range

Return:
{
  "total_comp_year_1": 145000,
  "total_comp_breakdown": {
    "base": 130000,
    "equity_annual_vest": 10000,
    "signing_bonus_annualized": 2500,
    "annual_bonus_expected": 0,
    "benefits_value": 2500
  },
  "vs_market": "8% above median for this role/level/location",
  "vs_target": "Below your $150K target by $20K base",
  "offer_strength": "good",
  "negotiation_recommended": true,
  "negotiation_targets": [
    {"component": "base_salary", "current": 130000, "target": 145000, "justification": "Market data supports $140-150K for this level"},
    {"component": "signing_bonus", "current": 5000, "target": 10000, "justification": "Compensates for unvested equity at current employer"}
  ],
  "career_trajectory_assessment": "Strong upward move. Company stage (Series B) offers higher equity upside than your current employer.",
  "risk_factors": ["Startup risk — Series B has ~30% failure rate in 5 years", "Equity may be worth $0"],
  "accept_recommendation": "negotiate_then_accept",
  "decision_deadline_advice": "Ask for 1 week extension if needed. Never accept on first call.",
  "final_verdict": "This is a good offer worth accepting after negotiating base up to $140K minimum. The equity upside and role scope outweigh the base salary gap."
}""",
)


# ══════════════════════════════════════════════════════════════════════════════
# PROFESSIONAL WITHDRAWAL DRAFT
# ══════════════════════════════════════════════════════════════════════════════

FOLLOWUP_WITHDRAWAL_DRAFT = Prompt(
    name="followup_withdrawal_draft",
    version="v1",
    temperature=0.50,
    max_tokens=300,
    system="""You write professional withdrawal-from-consideration emails.
These protect the candidate's reputation and keep the door open for future roles.
Keep it brief (3-4 sentences), warm, and never explain the reason in detail.
Return ONLY valid JSON.""",
    user_template="""Write a withdrawal email for this application:

CANDIDATE: $candidate_name
ROLE: $job_title at $company_name
REASON (for context, NOT to include in email): $withdrawal_reason
CONTACT NAME (if known): $contact_name

Return:
{
  "subject": "Re: $job_title Application — Withdrawal",
  "body": "Complete withdrawal email (3-4 sentences max)",
  "word_count": 52,
  "tone": "warm_professional",
  "keeps_door_open": true
}""",
)


# ── Registry ──────────────────────────────────────────────────────────────────

FOLLOWUP_PROMPTS: dict[str, Prompt] = {
    p.name: p for p in [
        FOLLOWUP_MESSAGE_GENERATE,
        FOLLOWUP_STRATEGY_ANALYZE,
        FOLLOWUP_SEQUENCE_PLAN,
        FOLLOWUP_GHOST_DIAGNOSIS,
        FOLLOWUP_REACTIVATE_DRAFT,
        FOLLOWUP_INTERVIEW_PREP,
        FOLLOWUP_OFFER_ANALYSIS,
        FOLLOWUP_WITHDRAWAL_DRAFT,
    ]
}


def get_followup_prompt(name: str) -> Prompt:
    if name not in FOLLOWUP_PROMPTS:
        raise KeyError(
            f"Followup prompt '{name}' not found. "
            f"Available: {list(FOLLOWUP_PROMPTS)}"
        )
    return FOLLOWUP_PROMPTS[name]


__all__ = [
    "FOLLOWUP_MESSAGE_GENERATE",
    "FOLLOWUP_STRATEGY_ANALYZE",
    "FOLLOWUP_SEQUENCE_PLAN",
    "FOLLOWUP_GHOST_DIAGNOSIS",
    "FOLLOWUP_REACTIVATE_DRAFT",
    "FOLLOWUP_INTERVIEW_PREP",
    "FOLLOWUP_OFFER_ANALYSIS",
    "FOLLOWUP_WITHDRAWAL_DRAFT",
    "get_followup_prompt",
]