"""
CareerGPT — Application Agent Prompts
========================================
PAGE SUMMARY:
  All LLM prompts used by the ApplicationAgent for browser form analysis,
  ATS platform detection, field mapping, CAPTCHA handling strategies,
  and auto-apply result interpretation.

  PROMPTS IN THIS FILE:
    APPLICATION_FORM_ANALYZE   → analyze page HTML to identify form fields
    APPLICATION_FIELD_MAP      → map candidate data to detected form fields
    APPLICATION_QUESTIONS_FILL → answer screening questions (cover + custom q's)
    APPLICATION_ERROR_DIAGNOSE → diagnose why an auto-apply attempt failed
    APPLICATION_SUCCESS_DETECT → detect if submission was successful

  USED BY: ApplicationAgent (agent.py) via tools.py

  DESIGN PRINCIPLE:
    Playwright automation handles the mechanics (click, type, upload).
    LLM handles the intelligence (which field is which, what to answer,
    did this page indicate success or failure).
    This separation means the LLM never sees credentials or session cookies
    — it only sees sanitized HTML and decides what text to put where.

  ATS PLATFORMS SUPPORTED:
    Greenhouse  → greenhouse.io   — most common for Series B+ startups
    Lever       → lever.co        — second most common
    Workday     → myworkdayjobs.com — large enterprises (SAP, banks, etc.)
    Ashby       → ashbyhq.com     — growing in YC/AI companies
    LinkedIn Easy Apply → linkedin.com — fastest path, 1-click for many
    Indeed Apply → indeed.com     — high volume, simple forms
    Generic     → company careers pages, custom ATS systems

  NOTE ON SAP/ENTERPRISE ROLES:
    Workday is used by virtually every SAP customer (Fortune 500).
    The APPLICATION_FORM_ANALYZE prompt specifically handles Workday's
    complex multi-step form patterns (profile import, equal opportunity,
    work authorization questions).
"""

from __future__ import annotations

from app.prompts.resume_prompts import Prompt


# ══════════════════════════════════════════════════════════════════════════════
# FORM ANALYSIS PROMPT
# Identifies ATS platform + all fillable fields from page HTML
# ══════════════════════════════════════════════════════════════════════════════

APPLICATION_FORM_ANALYZE = Prompt(
    name="application_form_analyze",
    version="v3",
    temperature=0.05,
    max_tokens=800,
    system="""You are an expert at analyzing job application form HTML.
Identify: ATS platform, all fillable fields with CSS selectors, file upload inputs,
multi-step indicators, CAPTCHA presence, and the submit button.

ATS DETECTION RULES:
  greenhouse.io   → look for 'greenhouse', '#application', class 'field--input'
  lever.co        → look for 'lever', 'lever-apply', input name patterns 'urls[LinkedIn]'
  myworkdayjobs   → look for 'workday', 'wd-' class prefixes, 'WDAY' in scripts
  ashbyhq.com     → look for 'ashby', 'ashby-job-posting'
  linkedin.com    → look for 'jobs-easy-apply', 'artdeco-modal'
  indeed.com      → look for 'indeed', 'ia-IndeedApplyButton'

SELECTOR PRIORITY: prefer id > name attr > data-testid > class (most stable to least stable)
Return ONLY valid JSON.""",
    user_template="""Analyze this job application form HTML:

URL: $page_url
HTML (truncated to 3000 chars):
$page_html

Candidate data available:
  name: $candidate_name
  email: $candidate_email
  phone: $candidate_phone
  location: $candidate_location
  linkedin: $linkedin_url
  years_experience: $experience_years

Return:
{
  "ats_platform": "greenhouse",
  "confidence": "high",
  "is_multi_step": false,
  "estimated_steps": 1,
  "captcha_detected": false,
  "file_upload_present": true,
  "fields": [
    {
      "selector": "#first_name",
      "field_type": "text",
      "label": "First Name",
      "maps_to": "first_name",
      "required": true,
      "value_to_fill": "$candidate_first_name"
    },
    {
      "selector": "input[name='job_application[email]']",
      "field_type": "email",
      "label": "Email",
      "maps_to": "email",
      "required": true,
      "value_to_fill": "$candidate_email"
    },
    {
      "selector": "input[type='file']",
      "field_type": "file_upload",
      "label": "Resume",
      "maps_to": "resume_file",
      "required": true,
      "value_to_fill": "RESUME_PATH"
    }
  ],
  "submit_selector": "input[type='submit'], button[type='submit']",
  "next_step_selector": null,
  "estimated_completion_time_seconds": 45
}""",
)


# ══════════════════════════════════════════════════════════════════════════════
# SCREENING QUESTIONS ANSWERER
# Handles custom screening questions + EEO/work authorization
# ══════════════════════════════════════════════════════════════════════════════

APPLICATION_QUESTIONS_FILL = Prompt(
    name="application_questions_fill",
    version="v2",
    temperature=0.15,
    max_tokens=1000,
    system="""You answer job application screening questions on behalf of a candidate.

STRICT RULES:
1. NEVER lie or fabricate: if candidate doesn't have a skill, answer honestly
2. For yes/no eligibility questions: answer based on provided candidate facts
3. For salary questions: use the provided salary_expectation or leave blank
4. For "why do you want to work here" type questions: write 2-3 genuine sentences
5. For EEO/diversity questions: always select "prefer not to answer" unless candidate specified
6. For work authorization: answer strictly based on candidate's actual location/status
7. Keep essay answers under 300 words — hiring managers don't read more than that
8. Return ONLY valid JSON""",
    user_template="""Answer these job application screening questions for this candidate:

CANDIDATE PROFILE:
Name: $candidate_name
Location: $candidate_location
Experience: $experience_years years
Skills: $candidate_skills
Work Authorization: $work_authorization
Salary Expectation: $salary_expectation
LinkedIn: $linkedin_url

JOB: $job_title at $company_name
COMPANY MISSION/CONTEXT: $company_context

QUESTIONS TO ANSWER:
$questions_json

Return:
{
  "answers": [
    {
      "question_id": "q1",
      "question_text": "Why do you want to work at $company_name?",
      "answer_type": "textarea",
      "answer": "I'm excited about $company_name's mission to... My background in $candidate_skills aligns directly with...",
      "selector": "textarea#question_1",
      "confidence": "high"
    },
    {
      "question_id": "q2",
      "question_text": "Are you authorized to work in the US?",
      "answer_type": "select",
      "answer": "Yes",
      "selector": "select#work_auth",
      "confidence": "high"
    }
  ],
  "unanswerable_questions": [],
  "notes": "Salary question left blank as per strategy — will negotiate after offer"
}""",
)


# ══════════════════════════════════════════════════════════════════════════════
# ERROR DIAGNOSIS PROMPT
# When auto-apply fails, diagnose why and suggest recovery
# ══════════════════════════════════════════════════════════════════════════════

APPLICATION_ERROR_DIAGNOSE = Prompt(
    name="application_error_diagnose",
    version="v1",
    temperature=0.1,
    max_tokens=400,
    system="""You diagnose job application automation failures.
Given the error context, identify the root cause and recommend a recovery action.
Return ONLY valid JSON.""",
    user_template="""Diagnose this auto-apply failure:

ATS PLATFORM: $ats_platform
ERROR MESSAGE: $error_message
PAGE URL AT FAILURE: $page_url
LAST SUCCESSFUL ACTION: $last_action
PAGE TITLE AT FAILURE: $page_title
SCREENSHOT AVAILABLE: $has_screenshot

Return:
{
  "root_cause": "captcha_triggered",
  "root_cause_description": "Cloudflare CAPTCHA appeared after navigating to application form",
  "severity": "blocking",
  "recovery_action": "manual_apply",
  "recovery_instructions": "Please complete this application manually. Direct URL: $page_url",
  "can_retry_automatically": false,
  "retry_delay_seconds": 0,
  "user_action_required": true,
  "user_message": "This company's application requires human verification. We've saved your resume and cover letter — click the link to apply manually in 2 minutes."
}""",
)


# ══════════════════════════════════════════════════════════════════════════════
# SUCCESS DETECTION PROMPT
# Confirms whether application was actually submitted
# ══════════════════════════════════════════════════════════════════════════════

APPLICATION_SUCCESS_DETECT = Prompt(
    name="application_success_detect",
    version="v2",
    temperature=0.05,
    max_tokens=200,
    system="""Analyze page content to determine if a job application was successfully submitted.
Look for: confirmation text, "application received", "thank you for applying",
redirect to confirmation page, confirmation email mention.
Return ONLY valid JSON.""",
    user_template="""Was this job application successfully submitted?

PAGE URL: $page_url
PAGE TITLE: $page_title
PAGE TEXT (first 1000 chars): $page_text
HTTP STATUS: $http_status

Return:
{
  "submitted": true,
  "confidence": "high",
  "confirmation_signal": "Page title says 'Application Submitted' and body contains 'Thank you for applying'",
  "confirmation_number": "APP-2024-98765",
  "next_steps_mentioned": "You will receive an email confirmation within 24 hours"
}""",
)


# ══════════════════════════════════════════════════════════════════════════════
# WORKDAY-SPECIFIC PROMPT
# Workday has complex multi-step forms — needs dedicated handling
# Used for SAP, enterprise, Fortune 500 companies
# ══════════════════════════════════════════════════════════════════════════════

APPLICATION_WORKDAY_STEPS = Prompt(
    name="application_workday_steps",
    version="v1",
    temperature=0.05,
    max_tokens=600,
    system="""You analyze Workday job application steps.
Workday uses a consistent multi-step pattern:
  Step 1: My Information (name, contact, address, source)
  Step 2: My Experience (resume upload, work history, education)
  Step 3: Application Questions (custom screening questions)
  Step 4: Voluntary Disclosures (EEO, veteran, disability)
  Step 5: Review & Submit

Identify current step and required actions.
Return ONLY valid JSON.""",
    user_template="""Identify the current Workday application step and actions needed:

CURRENT PAGE URL: $page_url
PAGE HEADING: $page_heading
VISIBLE FORM FIELDS: $visible_fields
STEP INDICATOR TEXT: $step_text

Return:
{
  "current_step": 1,
  "step_name": "My Information",
  "total_steps": 5,
  "fields_to_fill": [
    {"label": "First Name", "selector": "input[data-automation-id='legalNameSection_firstName']", "value": "$first_name"},
    {"label": "Last Name",  "selector": "input[data-automation-id='legalNameSection_lastName']",  "value": "$last_name"}
  ],
  "next_button_selector": "button[data-automation-id='bottom-navigation-next-button']",
  "save_for_later_selector": "button[data-automation-id='saveForLaterButton']",
  "can_import_linkedin": true,
  "can_import_resume": false
}""",
)


# ── Registry ──────────────────────────────────────────────────────────────────

APPLICATION_PROMPTS: dict[str, Prompt] = {
    p.name: p for p in [
        APPLICATION_FORM_ANALYZE,
        APPLICATION_QUESTIONS_FILL,
        APPLICATION_ERROR_DIAGNOSE,
        APPLICATION_SUCCESS_DETECT,
        APPLICATION_WORKDAY_STEPS,
    ]
}


def get_application_prompt(name: str) -> Prompt:
    if name not in APPLICATION_PROMPTS:
        raise KeyError(
            f"Application prompt '{name}' not found. "
            f"Available: {list(APPLICATION_PROMPTS)}"
        )
    return APPLICATION_PROMPTS[name]


__all__ = [
    "APPLICATION_FORM_ANALYZE",
    "APPLICATION_QUESTIONS_FILL",
    "APPLICATION_ERROR_DIAGNOSE",
    "APPLICATION_SUCCESS_DETECT",
    "APPLICATION_WORKDAY_STEPS",
    "get_application_prompt",
]