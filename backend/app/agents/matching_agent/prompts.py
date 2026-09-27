"""
CareerGPT -- Matching Agent Prompts
======================================
PAGE SUMMARY:
  LLM prompts used by MatchingAgent (agent.py) via tools.py for scoring
  resume/job fit, extracting structured job skill requirements, gap
  analysis, multi-job ranking, salary estimation and plain-English
  match explanations.

  PROMPTS IN THIS FILE:
    MATCH_ANALYSIS_PROMPT     -> full resume<->job scoring (score_match, full mode)
    MATCH_QUICK_SCORE_PROMPT  -> cheap/fast single score (score_match, fast mode)
    JOB_SKILLS_EXTRACT_PROMPT -> pull required/preferred skills off a job posting
    SKILLS_GAP_ANALYZE_PROMPT -> compare candidate skills against a job's requirements
    JOB_RANK_PROMPT           -> multi-factor ranking of several jobs for one candidate
    SALARY_ESTIMATE_PROMPT    -> estimate a pay range when a posting has none
    MATCH_EXPLANATION_PROMPT  -> turn a raw match score into a human-readable blurb

  USED BY: MatchingAgent (agent.py) via tools.py
"""

from __future__ import annotations

from app.prompts.resume_prompts import Prompt


MATCH_ANALYSIS_PROMPT = Prompt(
    name="match_analysis",
    version="v2",
    temperature=0.15,
    max_tokens=1600,
    system="""You are a senior technical recruiter with 15 years of experience matching
candidates to roles. You give honest, specific, evidence-based assessments -- never
generic praise, never a fabricated skill or fabricated experience.

Score on genuine fit: skills overlap, seniority match, domain relevance, and what's
realistically missing. Be direct about gaps -- a candidate benefits far more from an
honest 55 than an inflated 85.

Return ONLY valid JSON, no markdown, no commentary.""",
    user_template="""CANDIDATE RESUME:
$resume_text

JOB: $job_title at $company_name
JOB DESCRIPTION:
$job_text

Score this match and return this exact JSON:
{
  "score": 0.78,
  "rating": "Strong Match",
  "score_breakdown": {"skills": 0.8, "experience": 0.75, "domain": 0.7, "seniority": 0.85},
  "strengths": ["Direct experience with the core tech stack", "Seniority matches the role"],
  "gaps": ["No direct experience with X", "Limited exposure to Y"],
  "critical_missing": ["Required certification not present"],
  "ats_keywords_present": ["Python", "FastAPI", "PostgreSQL"],
  "ats_keywords_missing": ["Kubernetes", "Terraform"],
  "recommendation": "apply",
  "application_tips": ["Lead with the microservices project", "Quantify the team-size impact"]
}
"rating" must be one of: "Excellent Match", "Strong Match", "Good Match", "Fair Match", "Weak Match".
"recommendation" must be one of: "apply", "apply_with_tailoring", "skip".""",
)


MATCH_QUICK_SCORE_PROMPT = Prompt(
    name="match_quick_score",
    version="v1",
    temperature=0.1,
    max_tokens=300,
    system="""You are a fast resume/job matching engine. Give a quick, honest fit score
with minimal explanation. Return ONLY valid JSON.""",
    user_template="""CANDIDATE RESUME (excerpt):
$resume_text

JOB: $job_title at $company_name
$job_text

Return ONLY:
{"score": 0.72, "rating": "Good Match", "top_gap": "No cloud infra experience"}""",
)


JOB_SKILLS_EXTRACT_PROMPT = Prompt(
    name="job_skills_extract",
    version="v1",
    temperature=0.05,
    max_tokens=700,
    system="""Extract structured skill requirements from a job posting.
Group skills by category (languages, frameworks, tools, cloud, soft_skills).
Return ONLY valid JSON.""",
    user_template="""JOB POSTING TEXT:
$job_text

Return this exact JSON:
{
  "required_skills": {"languages": ["Python"], "frameworks": ["FastAPI"], "tools": ["Docker"], "cloud": ["AWS"], "soft_skills": []},
  "preferred_skills": {"languages": [], "frameworks": [], "tools": [], "cloud": [], "soft_skills": []},
  "experience_years_required": 3,
  "education_required": "bachelor"
}
"education_required" must be one of: high_school, associate, bachelor, master, phd, other.""",
)


SKILLS_GAP_ANALYZE_PROMPT = Prompt(
    name="skills_gap_analyze",
    version="v1",
    temperature=0.1,
    max_tokens=900,
    system="""You are a career coach analyzing a skills gap between a candidate and a
job's requirements. Be specific and actionable. Return ONLY valid JSON.""",
    user_template="""CANDIDATE SKILLS: $candidate_skills
CANDIDATE EXPERIENCE: $candidate_experience years, education: $candidate_education

JOB REQUIRED SKILLS: $required_skills
JOB PREFERRED SKILLS: $preferred_skills
JOB REQUIRES: $experience_required years experience, education: $education_required

Return this exact JSON:
{
  "skills_match_pct": 0.72,
  "matched_required": ["Python", "FastAPI"],
  "missing_required": ["Kubernetes"],
  "matched_preferred": ["Docker"],
  "missing_preferred": ["Terraform"],
  "experience_gap_years": 0,
  "education_meets_requirement": true,
  "overall_readiness": "ready",
  "closing_the_gap": ["Take a 1-week Kubernetes crash course", "Highlight the Docker project prominently"]
}
"overall_readiness" must be one of: ready, close, stretch, not_ready.""",
)


JOB_RANK_PROMPT = Prompt(
    name="job_rank",
    version="v1",
    temperature=0.2,
    max_tokens=1200,
    system="""You are a career strategist ranking job opportunities for a candidate
based on genuine career growth potential, not just keyword overlap. Consider the
candidate's stated career goal, remote preference and salary floor. Return ONLY
valid JSON.""",
    user_template="""CANDIDATE RESUME:
$resume_text
CAREER GOAL: $career_goal
PREFERS REMOTE: $prefers_remote
MINIMUM SALARY: $min_salary

JOBS TO RANK:
$jobs_text

Return this exact JSON:
{
  "ranked_job_ids": ["<job_id_1>", "<job_id_2>"],
  "rankings": [
    {"job_id": "<job_id_1>", "rank": 1, "score": 0.88, "reasoning": "Closest match to stated goal, remote, above floor"}
  ]
}""",
)


SALARY_ESTIMATE_PROMPT = Prompt(
    name="salary_estimate",
    version="v1",
    temperature=0.2,
    max_tokens=400,
    system="""You are a compensation analyst. Estimate a realistic USD salary range for
a role using title, seniority, location, remote status and tech stack. Base the
estimate on typical market data through your training; state your confidence
honestly. Return ONLY valid JSON.""",
    user_template="""JOB TITLE: $job_title
SUMMARY: $job_summary
LOCATION: $location (remote: $is_remote)
SENIORITY: $seniority
TECH STACK: $tech_stack
INDUSTRY: $industry

Return this exact JSON:
{
  "salary_min_usd": 95000,
  "salary_max_usd": 130000,
  "confidence": "medium",
  "source": "llm_estimate",
  "reasoning": "Mid-level backend role in a competitive remote market"
}
"confidence" must be one of: low, medium, high.""",
)


MATCH_EXPLANATION_PROMPT = Prompt(
    name="match_explanation",
    version="v1",
    temperature=0.4,
    max_tokens=350,
    system="""Write a short, warm, plain-English explanation of why a candidate is (or
isn't) a good fit for a role, for the candidate to read directly. 2-4 sentences.
Return ONLY valid JSON.""",
    user_template="""CANDIDATE: $candidate_name
JOB: $job_title at $company_name
MATCH SCORE: $score
STRENGTHS: $strengths
GAPS: $gaps

Return this exact JSON:
{"explanation": "You're a strong fit for this Backend Engineer role at Acme -- your FastAPI and PostgreSQL experience lines up directly with their stack. The main gap is Kubernetes, which is worth mentioning you're actively learning if you apply."}""",
)


__all__ = [
    "MATCH_ANALYSIS_PROMPT",
    "MATCH_QUICK_SCORE_PROMPT",
    "JOB_SKILLS_EXTRACT_PROMPT",
    "SKILLS_GAP_ANALYZE_PROMPT",
    "JOB_RANK_PROMPT",
    "SALARY_ESTIMATE_PROMPT",
    "MATCH_EXPLANATION_PROMPT",
]
