"""
CareerGPT — Matching Prompts
==============================
PAGE SUMMARY:
  All LLM prompt templates for resume-to-job matching, semantic scoring,
  gap analysis, skills extraction, and recommendation ranking.
  Used by: MatchingAgent, DiscoveryAgent, ApplicationService.

  PROMPTS IN THIS FILE:
    MATCH_ANALYSIS_PROMPT       → full resume-job match score + gap analysis
    MATCH_QUICK_SCORE_PROMPT    → fast 0-1 score (no explanation, low latency)
    SKILLS_EXTRACT_PROMPT       → extract skills from job description
    SKILLS_GAP_PROMPT           → identify gaps between resume and JD skills
    JOB_ENRICH_PROMPT           → AI enrichment of scraped job posting
    JOB_REQUIREMENTS_PARSE      → parse structured requirements from JD text
    CANDIDATE_RANK_PROMPT       → rank multiple jobs for a given resume
    SALARY_ESTIMATE_PROMPT      → estimate salary range from job posting context
    MATCH_EXPLANATION_PROMPT    → plain-English explanation of why candidate matches

  MATCH SCORING METHODOLOGY:
    score = weighted average of:
      40% → Skills overlap (required skills in resume / required skills in JD)
      20% → Experience years match (meets minimum requirement)
      15% → Education match (meets minimum level)
      15% → Semantic similarity (via Qdrant cosine score)
      10% → Location/remote match

    score range: 0.0 – 1.0
    interpretation:
      0.85+ → Strong Match (apply immediately, auto-tailor)
      0.70-0.84 → Good Match (apply with tailoring)
      0.55-0.69 → Partial Match (significant gaps, apply only if interested)
      Below 0.55 → Poor Match (not recommended)

  LATENCY TIERS:
    Tier 1 (fast,  <2s): MATCH_QUICK_SCORE — used for bulk job list scoring
    Tier 2 (med,   <5s): MATCH_ANALYSIS — used when opening a specific job
    Tier 3 (thorough, <10s): CANDIDATE_RANK — used for "find best jobs for me"
"""

from __future__ import annotations

from app.prompts.resume_prompts import Prompt


# ══════════════════════════════════════════════════════════════════════════════
# FULL MATCH ANALYSIS (medium latency, rich output)
# ══════════════════════════════════════════════════════════════════════════════

MATCH_ANALYSIS_PROMPT = Prompt(
    name="match_analysis",
    version="v3",
    temperature=0.1,
    max_tokens=1500,
    system="""You are a senior technical recruiter with 15 years of experience
at top tech companies (Google, Meta, Stripe). You specialize in precise
candidate-to-job matching with NO false positives.

SCORING CRITERIA (return score as 0.0-1.0 float):
  40%: Required skills overlap — how many JD required skills appear in resume?
  20%: Experience years — does candidate meet the minimum?
  15%: Education — does candidate meet the minimum level?
  15%: Role title and seniority alignment
  10%: Location / remote policy match (if specified)

OBJECTIVITY RULES:
- Do NOT inflate scores to make candidates feel good
- A 0.85+ score means "I would forward this resume to the hiring manager TODAY"
- A 0.5 score means "There are significant gaps but potential"
- Missing a critical required skill drops the score by 0.1 minimum per skill
- Having nice-to-have skills adds max 0.05 to score (don't over-weight bonus skills)

Return ONLY valid JSON — no preamble, no explanation outside the JSON.""",
    user_template="""Analyze this candidate's match for the job:

===== CANDIDATE RESUME =====
$resume_text

===== JOB DESCRIPTION =====
Title: $job_title
Company: $company_name
$job_text

Return:
{
  "score": 0.78,
  "rating": "Good Match",
  "confidence": "high",
  "score_breakdown": {
    "skills_overlap": 0.82,
    "experience_match": 0.75,
    "education_match": 1.0,
    "seniority_match": 0.80,
    "location_match": 1.0
  },
  "strengths": [
    "Python 3.10+ expertise directly matches primary stack requirement",
    "FastAPI experience matches their exact framework (mentioned 3x in JD)",
    "5 years experience exceeds their 3+ year minimum by comfortable margin",
    "PostgreSQL + Redis experience matches their data layer perfectly"
  ],
  "gaps": [
    "Missing Kubernetes experience listed as 'required' in JD",
    "No GraphQL mentioned (listed as 'preferred' — not critical)",
    "No mention of team leadership (JD says 'may mentor juniors')"
  ],
  "critical_missing": ["Kubernetes"],
  "nice_to_have_missing": ["GraphQL", "Terraform"],
  "ats_keywords_present": ["Python", "FastAPI", "PostgreSQL", "Docker", "REST API", "microservices"],
  "ats_keywords_missing": ["Kubernetes", "GraphQL", "CI/CD", "Terraform"],
  "experience_years_required": 3,
  "experience_years_candidate": 5,
  "experience_match": true,
  "education_required": "bachelor",
  "education_candidate": "bachelor",
  "education_match": true,
  "location_compatible": true,
  "remote_compatible": true,
  "salary_in_range": null,
  "recommendation": "Apply — strong technical match with one gap in Kubernetes that can be addressed in cover letter",
  "application_tips": [
    "Address Kubernetes gap directly in cover letter: mention any container orchestration exposure",
    "Lead with FastAPI experience in resume summary — it's their exact stack",
    "Quantify the scale of your PostgreSQL work (users served, data volume)"
  ],
  "estimated_interview_probability": 0.62
}""",
)


# ══════════════════════════════════════════════════════════════════════════════
# QUICK SCORE (low latency, for bulk scoring)
# ══════════════════════════════════════════════════════════════════════════════

MATCH_QUICK_SCORE_PROMPT = Prompt(
    name="match_quick_score",
    version="v2",
    temperature=0.05,
    max_tokens=200,
    system="""You are a precise job matching engine. Given a resume and job description,
return ONLY a match score from 0.0 to 1.0 with minimal explanation.
This is used for bulk scoring — speed is critical.
Return ONLY valid JSON with exactly these 4 fields.""",
    user_template="""Score this resume-job match (0.0-1.0):

RESUME SUMMARY: $resume_summary
RESUME SKILLS: $resume_skills

JOB: $job_title at $company_name
JOB REQUIRED SKILLS: $required_skills
JOB EXPERIENCE REQUIRED: $experience_required
JOB EDUCATION REQUIRED: $education_required

Return:
{
  "score": 0.76,
  "rating": "Good Match",
  "top_match_reason": "Python + FastAPI experience directly matches primary requirements",
  "top_gap": "Missing Kubernetes (required)"
}""",
)


# ══════════════════════════════════════════════════════════════════════════════
# SKILLS EXTRACTION FROM JD
# ══════════════════════════════════════════════════════════════════════════════

SKILLS_EXTRACT_PROMPT = Prompt(
    name="skills_extract",
    version="v2",
    temperature=0.05,
    max_tokens=600,
    system="""You are an expert at extracting and categorizing skills from job descriptions.
Extract ALL skills mentioned and categorize them accurately.
Do NOT add skills that are not mentioned in the text.
Return ONLY valid JSON.""",
    user_template="""Extract all skills from this job description:

$job_description

Return:
{
  "required_skills": {
    "programming_languages": ["Python", "TypeScript"],
    "frameworks": ["FastAPI", "React"],
    "databases": ["PostgreSQL", "Redis"],
    "cloud": ["AWS", "GCP"],
    "devops": ["Docker", "Kubernetes", "CI/CD"],
    "tools": ["Git", "Jira"],
    "methodologies": ["Agile", "TDD"],
    "soft_skills": ["communication", "collaboration"]
  },
  "preferred_skills": {
    "programming_languages": ["Rust", "Go"],
    "frameworks": ["GraphQL"],
    "databases": ["MongoDB"],
    "cloud": ["Terraform"],
    "other": ["ML experience"]
  },
  "all_skills_flat": ["Python", "TypeScript", "FastAPI", "React", "PostgreSQL"],
  "experience_years_required": 4,
  "experience_years_preferred": 6,
  "seniority_level": "senior",
  "education_required": "bachelor",
  "domain_knowledge": ["fintech", "payments", "distributed systems"],
  "certifications_mentioned": ["AWS Solutions Architect"],
  "visa_sponsorship": false,
  "remote_policy": "fully_remote"
}""",
)


# ══════════════════════════════════════════════════════════════════════════════
# SKILLS GAP ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

SKILLS_GAP_PROMPT = Prompt(
    name="skills_gap",
    version="v2",
    temperature=0.1,
    max_tokens=800,
    system="""You are a skills gap analyst. Compare a candidate's skills with job requirements
and produce a detailed gap analysis with learning recommendations.

CATEGORIZE GAPS BY SEVERITY:
  critical:    Required skill completely absent → likely rejection
  significant: Required skill partially covered → needs explicit mention in interview
  minor:       Preferred skill absent → small disadvantage
  irrelevant:  Preferred skill, candidate has a similar alternative

Return ONLY valid JSON.""",
    user_template="""Analyze the skills gap between this candidate and job:

CANDIDATE SKILLS: $candidate_skills
CANDIDATE EXPERIENCE: $candidate_experience years
CANDIDATE EDUCATION: $candidate_education

JOB REQUIRED SKILLS: $required_skills
JOB PREFERRED SKILLS: $preferred_skills
JOB EXPERIENCE REQUIRED: $experience_required years
JOB EDUCATION REQUIRED: $education_required

Return:
{
  "overall_skills_match_pct": 78,
  "critical_gaps": [
    {
      "skill": "Kubernetes",
      "severity": "critical",
      "why_matters": "Listed as required for their container orchestration workflow",
      "learning_path": "Kubernetes.io official tutorial + CKA certification (4-6 weeks)",
      "alternative": "Docker Swarm experience may partially compensate — mention explicitly"
    }
  ],
  "significant_gaps": [
    {
      "skill": "GraphQL",
      "severity": "significant",
      "why_matters": "Their API gateway uses GraphQL (mentioned 3x in JD)",
      "learning_path": "Apollo GraphQL docs + build a small project (1-2 weeks)",
      "alternative": "REST API expertise shows API design understanding"
    }
  ],
  "minor_gaps": ["Terraform", "Datadog"],
  "skills_candidate_has_that_match": ["Python", "FastAPI", "PostgreSQL", "Docker", "Redis", "AWS"],
  "bonus_skills_candidate_has": ["Celery", "SQLAlchemy", "pytest"],
  "gap_closure_timeline": {
    "critical_gaps_weeks": 6,
    "significant_gaps_weeks": 2,
    "total_ready_in_weeks": 8
  },
  "apply_now_recommendation": true,
  "apply_now_reasoning": "Strong 78% match. Critical Kubernetes gap is learnable. Apply now and address gap honestly in interview."
}""",
)


# ══════════════════════════════════════════════════════════════════════════════
# JOB ENRICHMENT (AI enhancement of scraped raw job)
# ══════════════════════════════════════════════════════════════════════════════

JOB_ENRICH_PROMPT = Prompt(
    name="job_enrich",
    version="v3",
    temperature=0.15,
    max_tokens=700,
    system="""You are a job posting analyst. Enrich raw job posting data with
structured analysis. Be accurate — do NOT invent information not in the posting.

Your enrichment helps candidates quickly evaluate fit and helps our matching system.

Return ONLY valid JSON.""",
    user_template="""Enrich this job posting:

Title: $job_title
Company: $company_name
Raw Description: $job_description

Return:
{
  "summary": "3-sentence summary: (1) what the role does, (2) required tech stack, (3) one key requirement or benefit",
  "keywords": ["Python", "FastAPI", "PostgreSQL", "Remote", "Senior", "Microservices"],
  "required_skills": ["Python 3.10+", "FastAPI", "PostgreSQL", "Docker"],
  "preferred_skills": ["Kubernetes", "GraphQL", "Terraform"],
  "tech_stack": ["Python", "FastAPI", "PostgreSQL", "Redis", "Docker", "AWS"],
  "experience_required": "4-6 years",
  "seniority": "senior",
  "company_stage": "Series B startup",
  "company_size": "50-200 employees",
  "role_type": "individual_contributor",
  "remote_policy": "fully_remote",
  "visa_sponsorship": false,
  "estimated_salary_usd": {"min": 120000, "max": 160000, "confidence": "medium"},
  "domain": ["fintech", "payments"],
  "green_flags": [
    "Fully remote with async culture",
    "Strong engineering team (ex-Stripe, Airbnb mentioned)",
    "Equity package for all employees"
  ],
  "red_flags": [
    "On-call rotation required (every 4 weeks)",
    "Scope seems very broad for one engineer"
  ],
  "culture_signals": ["async-first", "documentation-heavy", "autonomy-focused"],
  "interview_process": "Take-home project + 3 technical rounds (if mentioned, else null)",
  "urgency": "actively_hiring"
}""",
)


# ══════════════════════════════════════════════════════════════════════════════
# MULTI-JOB RANKING FOR CANDIDATE
# ══════════════════════════════════════════════════════════════════════════════

CANDIDATE_RANK_PROMPT = Prompt(
    name="candidate_rank",
    version="v2",
    temperature=0.1,
    max_tokens=1200,
    system="""You are a career advisor ranking job opportunities for a specific candidate.
Consider not just skill match but also: career growth, compensation, culture fit,
and the candidate's stated preferences.

Rank jobs from best to worst fit for THIS specific candidate.
Be specific about WHY each job ranks where it does.
Return ONLY valid JSON.""",
    user_template="""Rank these jobs for this candidate:

CANDIDATE PROFILE:
Background: $candidate_summary
Skills: $candidate_skills
Experience: $experience_years years
Career Goal: $career_goal
Preferences: remote=$prefers_remote, min_salary=$min_salary, target_roles=$target_roles

JOBS TO RANK:
$jobs_json

Return:
{
  "ranked_jobs": [
    {
      "rank": 1,
      "job_id": "job-uuid-here",
      "title": "Senior Python Engineer",
      "company": "Stripe",
      "match_score": 0.91,
      "why_top_ranked": "Best skills match (94%), exceeds salary preference, fully remote, career growth to Staff level visible",
      "pros": ["Exact Python/FastAPI stack", "Fully remote", "$150K-180K salary range", "Strong eng brand"],
      "cons": ["Large company culture", "On-call every 6 weeks"],
      "recommended_action": "Apply immediately with tailored resume + cover letter"
    }
  ],
  "summary": "Top 2 jobs are strong matches. Jobs 3-4 have significant skill gaps. Skip jobs 5-6.",
  "next_steps": [
    "Apply to jobs #1 and #2 today with tailored resumes",
    "Upskill in Kubernetes (addresses gap in job #3)",
    "Skip job #5 — salary below minimum preference by 30%"
  ]
}""",
)


# ══════════════════════════════════════════════════════════════════════════════
# SALARY ESTIMATION
# ══════════════════════════════════════════════════════════════════════════════

SALARY_ESTIMATE_PROMPT = Prompt(
    name="salary_estimate",
    version="v1",
    temperature=0.05,
    max_tokens=400,
    system="""You are a compensation analyst. Estimate the salary range for a job based on
all available signals: explicit salary if mentioned, company size/stage, location,
seniority level, tech stack, industry, and market data knowledge.

Be conservative and accurate — candidates should not be misled.
Confidence: high (explicit salary mentioned), medium (strong signals), low (guessing).
Return ONLY valid JSON.""",
    user_template="""Estimate salary for this role:

Job Title: $job_title
Company: $company_name
Company Stage: $company_stage
Location: $location
Remote: $is_remote
Seniority: $seniority
Tech Stack: $tech_stack
Industry: $industry
Explicit Salary in Posting: $explicit_salary

Return:
{
  "salary_min_usd": 120000,
  "salary_max_usd": 160000,
  "salary_midpoint_usd": 140000,
  "equity_likely": true,
  "equity_estimate": "0.05-0.15% for Series B",
  "confidence": "medium",
  "rationale": "Senior Python engineer at Series B fintech, remote US. Levels.fyi data for similar roles 2024: $125K-$165K total comp.",
  "currency": "USD",
  "compensation_type": "salary + equity",
  "bonus_likely": false,
  "data_sources": ["Levels.fyi comparable roles", "company stage benchmarks", "location adjustment"]
}""",
)


# ══════════════════════════════════════════════════════════════════════════════
# MATCH EXPLANATION (plain English for UI)
# ══════════════════════════════════════════════════════════════════════════════

MATCH_EXPLANATION_PROMPT = Prompt(
    name="match_explanation",
    version="v1",
    temperature=0.4,
    max_tokens=400,
    system="""You write clear, plain-English explanations of why a candidate
is or isn't a good match for a job. These explanations appear in the UI
next to a match score. Keep it friendly, honest, and actionable.
Return ONLY valid JSON.""",
    user_template="""Write a plain-English match explanation:

Candidate: $candidate_name
Job: $job_title at $company_name
Match Score: $match_score
Strengths: $strengths
Gaps: $gaps

Return:
{
  "headline": "Strong match — your Python + FastAPI stack is exactly what they need",
  "explanation": "You're a great fit for the core technical requirements. Your 5 years of Python experience exceeds their 3-year minimum, and your FastAPI work directly matches their backend stack. The main gap is Kubernetes — they list it as required, but your Docker experience shows container fundamentals. Worth applying and addressing this in your cover letter.",
  "action_recommendation": "Apply now with tailored resume",
  "urgency": "high",
  "one_liner": "87% match — apply today, address Kubernetes gap in cover letter"
}""",
)


# ── Registry ──────────────────────────────────────────────────────────────────

MATCHING_PROMPTS: dict[str, Prompt] = {
    p.name: p
    for p in [
        MATCH_ANALYSIS_PROMPT,
        MATCH_QUICK_SCORE_PROMPT,
        SKILLS_EXTRACT_PROMPT,
        SKILLS_GAP_PROMPT,
        JOB_ENRICH_PROMPT,
        CANDIDATE_RANK_PROMPT,
        SALARY_ESTIMATE_PROMPT,
        MATCH_EXPLANATION_PROMPT,
    ]
}


def get_matching_prompt(name: str) -> Prompt:
    """Fetch a matching prompt by name. Raises KeyError if not found."""
    if name not in MATCHING_PROMPTS:
        raise KeyError(
            f"Matching prompt '{name}' not found. "
            f"Available: {list(MATCHING_PROMPTS)}"
        )
    return MATCHING_PROMPTS[name]


__all__ = [
    "MATCH_ANALYSIS_PROMPT",
    "MATCH_QUICK_SCORE_PROMPT",
    "SKILLS_EXTRACT_PROMPT",
    "SKILLS_GAP_PROMPT",
    "JOB_ENRICH_PROMPT",
    "CANDIDATE_RANK_PROMPT",
    "SALARY_ESTIMATE_PROMPT",
    "MATCH_EXPLANATION_PROMPT",
    "get_matching_prompt",
]