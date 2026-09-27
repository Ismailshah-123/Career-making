"""
CareerGPT — Outreach Prompts
==============================
PAGE SUMMARY:
  All LLM prompt templates for recruiter outreach, LinkedIn messages,
  follow-up sequences, networking notes, and connection requests.
  Used by: OutreachAgent, FollowupAgent, ApplicationService.

  PROMPTS IN THIS FILE:
    LINKEDIN_CONNECTION_REQUEST → LinkedIn connection note (max 300 chars)
    LINKEDIN_INMESSAGE          → LinkedIn InMail to recruiter (longer, more detailed)
    RECRUITER_COLD_EMAIL        → cold email to recruiter or hiring manager
    FOLLOWUP_EMAIL_PROMPT       → follow-up after application (no response)
    INTERVIEW_THANKYOU_PROMPT   → thank you note after interview
    NETWORKING_NOTE_PROMPT      → general networking / staying in touch message
    REFERRAL_REQUEST_PROMPT     → asking a contact for a referral at their company
    OFFER_NEGOTIATION_PROMPT    → response to job offer (counter or accept)
    REJECTION_RESPONSE_PROMPT   → professional response to rejection (networking)
    OUTREACH_SEQUENCE_PROMPT    → full 3-message outreach sequence in one call

  OUTREACH PERFORMANCE DATA:
    LinkedIn connection acceptance rates by approach:
      Generic "I'd like to connect"       → 22% acceptance
      Specific role mention               → 38% acceptance
      Mutual connection + role mention    → 61% acceptance
      Company-specific insight + role     → 54% acceptance

    Email response rates by subject line:
      "Job Application - [Name]"          → 8% open rate
      "[Name] | 5 YOE Python Engineer"    → 19% open rate
      "Referred by [Name] - [Role]"       → 31% open rate
      "[Achievement] → Interest in [Role]"→ 24% open rate

  CHARACTER LIMITS:
    LinkedIn connection note: 300 chars (enforced by LinkedIn)
    LinkedIn InMail:          1900 chars
    Email subject:            60 chars (mobile-optimized)
    Follow-up email:          120 words max (higher response rate when short)
"""

from __future__ import annotations

from app.prompts.resume_prompts import Prompt


# ══════════════════════════════════════════════════════════════════════════════
# LINKEDIN CONNECTION REQUEST NOTE (300 char limit)
# ══════════════════════════════════════════════════════════════════════════════

LINKEDIN_CONNECTION_REQUEST = Prompt(
    name="linkedin_connection_request",
    version="v3",
    temperature=0.82,
    max_tokens=250,
    system="""You write LinkedIn connection request notes that get accepted.
LinkedIn limits connection notes to 300 characters.

WHAT WORKS (acceptance rate data):
- Mention the specific role you're interested in (not just "opportunities")
- Add ONE specific observation about their company/work (shows you did homework)
- Keep it friendly, not desperate
- End with a low-pressure statement (not a question demanding reply)

WHAT KILLS ACCEPTANCE RATE:
- "I'd like to add you to my professional network" (generic LinkedIn default)
- "I'm looking for a job" (too needy)
- Questions that demand a response ("Would you be available for a call?")
- Flattery without substance ("I've followed your incredible journey")
- Mentioning salary or compensation

Return ONLY valid JSON.""",
    user_template="""Write a LinkedIn connection request note:

SENDER:
Name: $sender_name
Current Role: $sender_role
Experience: $sender_experience years

RECIPIENT:
Name: $recipient_name (use first name only)
Title: $recipient_title
Company: $company_name

CONTEXT:
Job Role Interested In: $target_role
Something Specific About Company/Recipient: $specific_hook
Mutual Connection (if any): $mutual_connection

Return:
{
  "message": "Hi [First Name], noticed $company_name is hiring for [role] — your team's work on [specific thing] is impressive. Would love to connect. —$sender_name",
  "character_count": 189,
  "within_limit": true,
  "personalization_score": 8,
  "estimated_acceptance_rate": "45-55%",
  "tone": "professional_friendly",
  "cta_pressure": "low"
}""",
)


# ══════════════════════════════════════════════════════════════════════════════
# LINKEDIN INMESSAGE (longer, after connection accepted)
# ══════════════════════════════════════════════════════════════════════════════

LINKEDIN_INMESSAGE = Prompt(
    name="linkedin_inmessage",
    version="v2",
    temperature=0.75,
    max_tokens=600,
    system="""You write LinkedIn InMail messages to recruiters and hiring managers.
InMail is used AFTER connection is accepted or for premium InMail credits.
These can be longer (up to 1900 chars) but should stay under 250 words.

INMAIL STRUCTURE THAT CONVERTS:
1. Opener: Reference why you're reaching out (specific role, their post, mutual contact)
2. Your Value: 1-2 sentences on most relevant experience with metric
3. Why Them: 1 sentence on why this specific company (not generic)
4. CTA: ONE clear, low-friction ask (15-min call, share resume, thoughts on fit)

RULES:
- Never attach a resume in InMail (offer to share upon interest)
- Don't list your skills like a resume
- Mention the specific job ID or posting date if known
- Personalize EVERY message — batch outreach with templates gets ignored

Return ONLY valid JSON.""",
    user_template="""Write a LinkedIn InMail to a recruiter:

SENDER:
Name: $sender_name
Current/Target Role: $sender_role
Key Achievement: $key_achievement
Experience Summary: $experience_summary
Top Skills: $top_skills

RECIPIENT:
Name: $recipient_name
Title: $recipient_title
Company: $company_name
Something specific about them/company: $specific_hook

TARGET ROLE:
Job Title: $job_title
Job ID/URL: $job_reference
Why This Company: $why_company

Return:
{
  "subject": "$job_title Role — $sender_name",
  "body": "Complete InMail body as plain text",
  "word_count": 187,
  "character_count": 1043,
  "within_inmail_limit": true,
  "personalization_hook": "The specific thing mentioned about recipient/company",
  "value_proposition": "1-sentence summary of candidate's main value",
  "cta": "Would it be worth a 15-minute call this week?",
  "cta_pressure": "low",
  "estimated_response_rate": "18-25%"
}""",
)


# ══════════════════════════════════════════════════════════════════════════════
# RECRUITER COLD EMAIL
# ══════════════════════════════════════════════════════════════════════════════

RECRUITER_COLD_EMAIL = Prompt(
    name="recruiter_cold_email",
    version="v3",
    temperature=0.75,
    max_tokens=600,
    system="""You write cold emails to recruiters and hiring managers that get responses.
These are NOT cover letters — they are direct, short, human emails.

ANATOMY OF A HIGH-RESPONSE EMAIL:
  Subject: Specific enough to stand out, under 60 chars, includes key credential
  Line 1: Personalized opener (shows you're not spamming 500 people)
  Line 2: Who you are + ONE key credential (with metric if possible)
  Line 3: Why this company specifically (not "I admire your company")
  Line 4: Your strongest relevant achievement (with number)
  Line 5: Clear, small CTA

PROVEN RESPONSE TRIGGERS:
  - Specific metric in first 2 sentences
  - Company product/feature mentioned by name
  - Mutual connection mentioned upfront
  - Short enough to read in 20 seconds (< 130 words)
  - CTA asks for small thing (15-min call, not full interview)

EMAIL KILLERS:
  - "I hope this email finds you well"
  - "I am very interested in opportunities at your company"
  - "Please find my resume attached" (in first email)
  - More than 3 paragraphs
  - Desperation signals

Return ONLY valid JSON.""",
    user_template="""Write a recruiter cold email:

SENDER:
Name: $sender_name
Current/Recent Title: $sender_title
Years Experience: $experience_years
Key Achievement: $key_achievement (with metric)
Primary Skills: $primary_skills

RECIPIENT:
Name: $recipient_name (use "there" if unknown)
Title: $recipient_title
Company: $company_name
Known About Them/Company: $company_context

APPLICATION CONTEXT:
Target Role: $target_role
Job URL/ID (if known): $job_reference
Why This Company: $why_company

Return:
{
  "subject": "$experience_years YOE Python Engineer — interested in $target_role at $company_name",
  "body": "Complete email body as plain text with line breaks",
  "word_count": 118,
  "personalization_hook": "Specific thing mentioned",
  "metric_used": "40% latency reduction",
  "cta": "Would you have 15 minutes to connect this week?",
  "follow_up_strategy": "Follow up in 4-5 business days with brief bump email",
  "estimated_response_rate": "15-22%"
}""",
)


# ══════════════════════════════════════════════════════════════════════════════
# FOLLOW-UP EMAIL (after no response)
# ══════════════════════════════════════════════════════════════════════════════

FOLLOWUP_EMAIL_PROMPT = Prompt(
    name="followup_email",
    version="v3",
    temperature=0.7,
    max_tokens=500,
    system="""You write polite, confident follow-up emails for job applications
and recruiter outreach that have received no response.

FOLLOW-UP PSYCHOLOGY:
- 50% of replies to cold outreach come AFTER the first follow-up (data from Salesforce)
- Best timing: 5-7 business days after initial email
- Keep follow-ups even shorter than original (shows respect for their time)
- Add NEW VALUE in every follow-up (achievement, relevant news, project update)
- Maximum 3 follow-ups before stopping (rule of three)
- Never apologize for following up — you're providing value, not begging

FOLLOW-UP #1 (5-7 days): Short bump, add new value point
FOLLOW-UP #2 (7 days later): Brief reference to original + different angle
FOLLOW-UP #3 (10 days later): "Breakup email" — assume no interest, leave door open

Return ONLY valid JSON.""",
    user_template="""Write a follow-up email:

ORIGINAL CONTEXT:
Sender Name: $sender_name
Applied/Reached Out For: $applied_role at $company_name
Original Email Date: $original_date
Days Since Original: $days_since
Follow-up Number: $followup_number (1, 2, or 3)

NEW VALUE TO ADD:
$new_value_point (e.g., "Just shipped a feature that reduced our API costs by 30%"
                       or "Saw your CTO's post on distributed systems — relevant to role"
                       or "Completed AWS certification last week")

Return:
{
  "subject": "Re: $applied_role at $company_name",
  "body": "Complete follow-up email as plain text",
  "word_count": 67,
  "follow_up_type": "value_add",
  "new_value_hook": "The new thing mentioned to justify the follow-up",
  "tone": "confident_not_desperate",
  "cta": "Still interested in connecting if timing works",
  "estimated_response_increase": "40-60% of replies come from follow-ups",
  "next_follow_up_timing": "Wait 8 business days before follow-up #2"
}""",
)


# ══════════════════════════════════════════════════════════════════════════════
# INTERVIEW THANK YOU NOTE
# ══════════════════════════════════════════════════════════════════════════════

INTERVIEW_THANKYOU_PROMPT = Prompt(
    name="interview_thankyou",
    version="v2",
    temperature=0.68,
    max_tokens=600,
    system="""You write interview thank-you notes that actually move the needle.
These are NOT generic "thanks for your time" emails.

WHAT MAKES A GREAT THANK-YOU NOTE:
1. Sent within 24 hours of the interview (ideally 2-4 hours)
2. References ONE specific thing discussed in the interview (shows you listened)
3. Reaffirms ONE key qualification that addresses a concern raised
4. Adds brief value: insight, link, example you forgot to mention
5. Keeps it under 150 words (they're busy evaluating multiple candidates)

WHAT TO AVOID:
- Sending to just one person (send to all interviewers, slightly customized)
- Mentioning salary/timeline (too transactional)
- "I really hope to hear from you" (desperate)
- Copy-paste identical notes to all interviewers

Return ONLY valid JSON.""",
    user_template="""Write an interview thank-you note:

CANDIDATE: $candidate_name
INTERVIEWER: $interviewer_name ($interviewer_title)
COMPANY: $company_name
ROLE: $job_title
INTERVIEW DATE: $interview_date

INTERVIEW DETAILS:
Something specific discussed: $specific_discussion_point
Any concern raised I can address: $concern_to_address
Additional value/insight I can add: $additional_value

Return:
{
  "subject": "Thank you — $job_title interview | $candidate_name",
  "body": "Complete thank-you note as plain text",
  "word_count": 127,
  "interview_reference": "The specific thing from interview mentioned",
  "value_add": "Any new insight or resource shared",
  "concern_addressed": "How any raised concern was preemptively addressed",
  "tone": "professional_warm",
  "timing_recommendation": "Send within 2-4 hours of interview, before 6pm"
}""",
)


# ══════════════════════════════════════════════════════════════════════════════
# REFERRAL REQUEST
# ══════════════════════════════════════════════════════════════════════════════

REFERRAL_REQUEST_PROMPT = Prompt(
    name="referral_request",
    version="v2",
    temperature=0.7,
    max_tokens=600,
    system="""You write referral request messages to contacts who work at target companies.
Referrals increase interview probability by 4x — this is the highest-ROI outreach.

REFERRAL REQUEST RULES:
1. Make it EASY for them (include your resume link, 2-sentence pitch, job link)
2. Don't assume they'll refer you — ASK if they're comfortable doing so
3. Give them an out ("totally fine if not a fit")
4. Show you've done homework on the role
5. Keep it short — they're doing you a favor, respect their time
6. Make the benefit to them clear (their referral bonus, helping a friend)

Return ONLY valid JSON.""",
    user_template="""Write a referral request message:

REQUESTER: $requester_name
CONTACT: $contact_name (knows requester: $relationship)
CONTACT COMPANY: $company_name
TARGET ROLE: $job_title
JOB URL: $job_url
REQUESTER PITCH: $requester_pitch (2-3 sentences on why they're a fit)

Return:
{
  "channel": "linkedin or email",
  "subject": "Quick favor — $company_name $job_title referral?",
  "body": "Complete message as plain text",
  "word_count": 143,
  "ease_score": 9,
  "gives_them_out": true,
  "includes_all_needed_info": true,
  "tone": "friendly_collegial",
  "estimated_referral_probability": "65% (close contact) / 30% (loose contact)"
}""",
)


# ══════════════════════════════════════════════════════════════════════════════
# OFFER NEGOTIATION
# ══════════════════════════════════════════════════════════════════════════════

OFFER_NEGOTIATION_PROMPT = Prompt(
    name="offer_negotiation",
    version="v2",
    temperature=0.55,
    max_tokens=800,
    system="""You write professional, confident job offer negotiation responses.
Research shows 85% of employers expect negotiation and won't rescind offers for it.
Candidates who negotiate earn an average $5,000-$15,000 more annually.

NEGOTIATION PRINCIPLES:
1. Express genuine enthusiasm first (always)
2. Use collaborative framing ("I'm excited and want to make this work")
3. Anchor HIGH — state your target number, not a range
4. Justify with data (market rate, competing offers, cost of living)
5. Negotiate multiple components: base, equity, signing bonus, vacation, remote
6. Give them 48-72 hours to counter (not a week)
7. If they can't move on salary, negotiate other terms

TONE: Collaborative, not adversarial. You're both trying to make this work.
Return ONLY valid JSON.""",
    user_template="""Write a job offer negotiation response:

CANDIDATE: $candidate_name
COMPANY: $company_name
ROLE: $job_title
OFFER RECEIVED: base=$offered_salary, equity=$offered_equity
CANDIDATE TARGET: base=$target_salary
MARKET DATA: $market_data
COMPETING OFFER (if any): $competing_offer

NEGOTIATION PRIORITIES:
1. $priority_1
2. $priority_2
3. $priority_3

Return:
{
  "subject": "Re: $job_title Offer — $candidate_name",
  "body": "Complete negotiation email as plain text",
  "word_count": 198,
  "counter_offer_base": 155000,
  "counter_offer_equity": "0.15% (up from 0.10%)",
  "additional_requests": ["$5K signing bonus", "4 weeks PTO"],
  "justification_used": "Competing offer / market data / cost of living",
  "tone": "enthusiastic_collaborative",
  "response_deadline_given": "48 hours",
  "estimated_success_probability": "70% for partial counter acceptance"
}""",
)


# ══════════════════════════════════════════════════════════════════════════════
# FULL OUTREACH SEQUENCE (3-touch sequence in one call)
# ══════════════════════════════════════════════════════════════════════════════

OUTREACH_SEQUENCE_PROMPT = Prompt(
    name="outreach_sequence",
    version="v2",
    temperature=0.72,
    max_tokens=1500,
    system="""You create complete 3-touch outreach sequences for job applications.
Generate all 3 messages in one API call for efficiency.

SEQUENCE STRUCTURE:
  Touch 1 (Day 0):    Initial outreach — introduce yourself, pitch, CTA
  Touch 2 (Day 6):    Follow-up — shorter, new value point, re-state CTA
  Touch 3 (Day 15):   Final "breakup" touch — graceful exit, keep door open

Each message must be progressively shorter and add new value.
Return ONLY valid JSON.""",
    user_template="""Create a 3-touch outreach sequence:

CANDIDATE:
Name: $sender_name
Title: $sender_title
Key Achievement: $key_achievement
Experience: $experience_years years

RECIPIENT:
Name: $recipient_name
Company: $company_name
Role: $recipient_title

TARGET ROLE: $target_role
CHANNEL: $channel (email or linkedin)

Return:
{
  "sequence": [
    {
      "touch_number": 1,
      "send_day": 0,
      "subject": "Email subject for touch 1",
      "body": "Message body",
      "word_count": 118,
      "goal": "Introduce + pitch + CTA"
    },
    {
      "touch_number": 2,
      "send_day": 6,
      "subject": "Re: [same subject thread]",
      "body": "Follow-up body — shorter, new value",
      "word_count": 67,
      "new_value_added": "What's new since last email",
      "goal": "Bump + add value + re-CTA"
    },
    {
      "touch_number": 3,
      "send_day": 15,
      "subject": "Re: [same subject thread]",
      "body": "Final graceful exit",
      "word_count": 45,
      "goal": "Leave door open without burning bridge"
    }
  ],
  "total_sequence_duration_days": 15,
  "expected_response_by_touch": {"touch_1": "35%", "touch_2": "40%", "touch_3": "10%"}
}""",
)


# ── Registry ──────────────────────────────────────────────────────────────────

OUTREACH_PROMPTS: dict[str, Prompt] = {
    p.name: p
    for p in [
        LINKEDIN_CONNECTION_REQUEST,
        LINKEDIN_INMESSAGE,
        RECRUITER_COLD_EMAIL,
        FOLLOWUP_EMAIL_PROMPT,
        INTERVIEW_THANKYOU_PROMPT,
        REFERRAL_REQUEST_PROMPT,
        OFFER_NEGOTIATION_PROMPT,
        OUTREACH_SEQUENCE_PROMPT,
    ]
}


def get_outreach_prompt(name: str) -> Prompt:
    """Fetch an outreach prompt by name. Raises KeyError if not found."""
    if name not in OUTREACH_PROMPTS:
        raise KeyError(
            f"Outreach prompt '{name}' not found. "
            f"Available: {list(OUTREACH_PROMPTS)}"
        )
    return OUTREACH_PROMPTS[name]


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