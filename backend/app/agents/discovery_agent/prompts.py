"""
CareerGPT — Discovery Agent Prompts
======================================
PAGE SUMMARY:
  All LLM prompts used by the DiscoveryAgent for intelligent job discovery,
  query expansion, relevance re-ranking, and opportunity scoring.

  PROMPTS:
    DISCOVERY_QUERY_EXPAND    → expand user intent into multiple search queries
    DISCOVERY_RERANK          → re-rank semantic search results by fit quality
    DISCOVERY_OPPORTUNITY_SCORE → score a job's hidden opportunity potential
    DISCOVERY_MARKET_INSIGHTS  → extract market insights from job batch
    DISCOVERY_ALERT_FILTER    → decide if a new job warrants a user notification

  USED BY: DiscoveryAgent (agent.py) via tools.py
"""

from __future__ import annotations
from app.prompts.resume_prompts import Prompt


DISCOVERY_QUERY_EXPAND = Prompt(
    name="discovery_query_expand",
    version="v2",
    temperature=0.3,
    max_tokens=600,
    system="""You are a job search expert. Expand a user's job search intent into
multiple specific search queries that maximize relevant job discovery across
different job boards and terminologies.

Different companies use different titles for the same role:
  "Backend Engineer" = "Server-Side Developer" = "API Engineer" = "Platform Engineer"
Include regional variations: US terms differ from UK/Pakistan/India terms.
Return ONLY valid JSON.""",
    user_template="""Expand this job search for maximum discovery:

USER INTENT: $user_intent
USER SKILLS: $user_skills
USER EXPERIENCE_YEARS: $experience_years
TARGET LOCATIONS: $target_locations
REMOTE PREFERENCE: $remote_preference

Return:
{
  "primary_queries": [
    {"keyword": "Senior Python Engineer", "location": "remote", "source": "all"},
    {"keyword": "Backend Developer Python FastAPI", "location": "United States", "source": "linkedin"}
  ],
  "alternative_titles": [
    "API Engineer", "Platform Engineer", "Server-Side Developer", "Python Developer"
  ],
  "exclude_keywords": ["Java", "PHP", "Ruby on Rails"],
  "salary_signals": ["$120K+", "competitive salary", "equity"],
  "company_type_signals": ["startup", "Series B", "Series C", "scale-up"],
  "total_query_count": 12,
  "estimated_jobs_per_run": 300
}""",
)

DISCOVERY_RERANK = Prompt(
    name="discovery_rerank",
    version="v2",
    temperature=0.1,
    max_tokens=1000,
    system="""You are a senior technical recruiter re-ranking job results for a specific candidate.
Re-rank these jobs from best to worst match for this candidate's profile and goals.
Consider: skill match, career growth, compensation signals, company quality, remote policy.
Return ONLY valid JSON.""",
    user_template="""Re-rank these jobs for this candidate:

CANDIDATE:
Skills: $candidate_skills
Experience: $experience_years years
Goal: $career_goal
Preferences: remote=$prefers_remote, min_salary=$min_salary

JOBS (semantic search results):
$jobs_json

Return:
{
  "ranked_jobs": [
    {
      "job_id": "uuid",
      "original_rank": 3,
      "new_rank": 1,
      "fit_score": 0.91,
      "ranking_reason": "Perfect Python/FastAPI stack match, fully remote, Series B with growth potential",
      "apply_urgency": "high",
      "estimated_competition": "medium"
    }
  ],
  "top_pick_reasoning": "Why the #1 job stands out above all others",
  "jobs_to_skip": ["job_id_1", "job_id_2"],
  "skip_reasons": {"job_id_1": "Salary signals below minimum", "job_id_2": "Requires Salesforce (deal breaker)"}
}""",
)

DISCOVERY_OPPORTUNITY_SCORE = Prompt(
    name="discovery_opportunity_score",
    version="v1",
    temperature=0.1,
    max_tokens=500,
    system="""You score a job posting's hidden opportunity quality beyond skill match.
Consider: company growth trajectory, role seniority ceiling, team quality signals,
equity upside, hiring urgency, competition level.
Return ONLY valid JSON with scores 0.0-1.0.""",
    user_template="""Score the opportunity quality of this job:

JOB: $job_title at $company_name
COMPANY STAGE: $company_stage
DESCRIPTION: $job_description

Return:
{
  "opportunity_score": 0.84,
  "growth_potential": 0.9,
  "compensation_potential": 0.8,
  "company_quality": 0.85,
  "role_impact": 0.82,
  "hiring_urgency": "high",
  "estimated_competition_level": "medium",
  "hidden_signals": [
    "Mentions 'leading the architecture' — this is a senior/lead scope",
    "Series B + YC-backed = strong equity upside",
    "Posted 2 days ago = low competition window"
  ],
  "red_flags": ["On-call every 3 weeks mentioned in benefits"],
  "opportunity_summary": "High-growth Series B with lead-level scope. Apply in next 48h before competition builds."
}""",
)

DISCOVERY_MARKET_INSIGHTS = Prompt(
    name="discovery_market_insights",
    version="v1",
    temperature=0.2,
    max_tokens=700,
    system="""You extract market intelligence from a batch of job postings.
Identify trends, in-demand skills, salary signals, and hiring patterns.
Return ONLY valid JSON.""",
    user_template="""Analyze these $job_count job postings for market insights:

SAMPLE TITLES: $sample_titles
SAMPLE COMPANIES: $sample_companies
COMMON SKILLS SEEN: $common_skills
SALARY RANGE SEEN: $salary_range
SOURCES: $sources

Return:
{
  "hottest_skills": ["Python", "FastAPI", "Kubernetes", "LLM integration"],
  "declining_skills": ["Django monoliths", "jQuery"],
  "avg_salary_range": {"min": 110000, "max": 160000, "currency": "USD"},
  "top_hiring_companies": ["Stripe", "Anthropic", "OpenAI"],
  "remote_job_pct": 68,
  "contract_vs_fulltime_ratio": "22% contract / 78% full-time",
  "experience_demand": {"entry": 15, "mid": 35, "senior": 40, "lead_plus": 10},
  "market_summary": "Strong demand for Python engineers with AI/ML integration experience. Kubernetes becoming required, not preferred.",
  "candidate_advice": "Add even basic LLM/OpenAI API experience to resume immediately — appearing in 40% of senior Python JDs"
}""",
)

DISCOVERY_ALERT_FILTER = Prompt(
    name="discovery_alert_filter",
    version="v1",
    temperature=0.05,
    max_tokens=200,
    system="""You decide if a newly scraped job should trigger a real-time notification
for a specific user. Be conservative — only alert for genuinely strong matches.
Return ONLY valid JSON.""",
    user_template="""Should this job trigger a notification for this user?

USER SKILLS: $user_skills
USER MIN_SALARY: $min_salary
USER REMOTE_PREF: $remote_preference
USER EXPERIENCE: $experience_years years

JOB: $job_title at $company_name ($source)
JOB SALARY: $job_salary
JOB REMOTE: $is_remote
JOB SKILLS REQUIRED: $required_skills
SEMANTIC MATCH SCORE: $match_score

Return:
{
  "should_alert": true,
  "alert_priority": "high",
  "alert_reason": "95% skill match + remote + salary above minimum. High-urgency: posted 1 hour ago.",
  "suggested_alert_message": "🔥 New job: Senior Python Engineer at Stripe — 95% match"
}""",
)

DISCOVERY_PROMPTS = {
    p.name: p for p in [
        DISCOVERY_QUERY_EXPAND,
        DISCOVERY_RERANK,
        DISCOVERY_OPPORTUNITY_SCORE,
        DISCOVERY_MARKET_INSIGHTS,
        DISCOVERY_ALERT_FILTER,
    ]
}

def get_discovery_prompt(name: str) -> Prompt:
    if name not in DISCOVERY_PROMPTS:
        raise KeyError(f"Discovery prompt '{name}' not found. Available: {list(DISCOVERY_PROMPTS)}")
    return DISCOVERY_PROMPTS[name]

__all__ = ["DISCOVERY_PROMPTS", "get_discovery_prompt",
           "DISCOVERY_QUERY_EXPAND", "DISCOVERY_RERANK",
           "DISCOVERY_OPPORTUNITY_SCORE", "DISCOVERY_MARKET_INSIGHTS",
           "DISCOVERY_ALERT_FILTER"]