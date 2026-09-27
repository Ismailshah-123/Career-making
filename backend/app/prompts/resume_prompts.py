"""
CareerGPT — Resume Prompts
============================
PAGE SUMMARY:
  All LLM prompt templates for resume-related operations.
  Used by: ResumeService, ResumeAgent, MatchingAgent.

  PROMPTS:
    RESUME_PARSE_PROMPT     → extract structured fields from raw text
    RESUME_ATS_SCORE_PROMPT → score 0-100 for ATS compatibility
    RESUME_TAILOR_PROMPT    → rewrite resume to match job description
    RESUME_ANALYZE_PROMPT   → deep quality analysis with recommendations
    RESUME_BULLET_IMPROVE   → rewrite individual bullet points with metrics
    RESUME_SUMMARY_WRITE    → write targeted professional summary for a role

  PROMPT DESIGN PHILOSOPHY:
    1. System prompt = persona + strict output format + rules
    2. User prompt = data injection via Template substitution
    3. All prompts return valid JSON (enforced in system prompt)
    4. Temperature calibrated per task (0.05 for extraction, 0.7 for writing)
    5. Token limits set conservatively to avoid truncation
    6. Schema hints provided so LLM knows exact output structure

  VERSIONING:
    Each Prompt dataclass has a version field.
    When prompts are updated, bump version so A/B tests can track impact.
    AgentRun table stores which prompt version was used for each generation.
"""

from __future__ import annotations

from dataclasses import dataclass
from string import Template


@dataclass(frozen=True)
class Prompt:
    """Immutable prompt definition with typed render methods."""
    name:         str
    system:       str
    user_template: str
    version:      str  = "v1"
    max_tokens:   int  = 2048
    temperature:  float = 0.2

    def render(self, **kwargs: object) -> tuple[str, str]:
        """Return (system_prompt, user_message) with variables substituted."""
        user = Template(self.user_template).safe_substitute(**kwargs)
        return self.system, user

    def render_user(self, **kwargs: object) -> str:
        return Template(self.user_template).safe_substitute(**kwargs)


# ══════════════════════════════════════════════════════════════════════════════
# RESUME PARSE PROMPT
# ══════════════════════════════════════════════════════════════════════════════

RESUME_PARSE_PROMPT = Prompt(
    name="resume_parse",
    version="v3",
    temperature=0.05,
    max_tokens=1500,
    system="""You are an expert resume parser with 15 years of HR and ATS experience.
Extract ALL structured information from the resume text with 100% accuracy.

CRITICAL RULES:
- Return ONLY valid JSON — no markdown, no code fences, no explanation whatsoever
- If a field is not present, use null (not empty string, not "N/A")
- skills and languages MUST be JSON arrays (even if only one item)
- experience_years must be a float (e.g., 4.5 for 4.5 years), estimated from job dates
- education_level must be one of: high_school, associate, bachelor, master, phd, other
- Extract ALL skills mentioned anywhere in the resume (technologies, tools, methodologies)
- Soft skills go in soft_skills array separately from technical skills""",
    user_template="""Parse this resume and extract all structured information:

$resume_text

Return this exact JSON structure (no extra fields, no missing fields):
{
  "name": "Full Name or null",
  "email": "email@example.com or null",
  "phone": "+1234567890 or null",
  "location": "City, Country or null",
  "linkedin_url": "https://linkedin.com/in/... or null",
  "github_url": "https://github.com/... or null",
  "portfolio_url": "https://... or null",
  "summary": "Professional summary paragraph or null",
  "skills": ["Python", "FastAPI", "PostgreSQL", "Docker", "AWS"],
  "soft_skills": ["Leadership", "Communication", "Problem Solving"],
  "certifications": ["AWS Solutions Architect", "Google Cloud Professional"],
  "languages": ["English", "Urdu"],
  "experience_years": 5.5,
  "education_level": "bachelor",
  "education": [
    {"degree": "BSc Computer Science", "institution": "MIT", "year": "2019", "gpa": null}
  ],
  "work_experience": [
    {
      "company": "TechCorp",
      "title": "Senior Software Engineer",
      "start_date": "2021-03",
      "end_date": "2024-01",
      "is_current": false,
      "location": "Remote",
      "bullets": ["Led team of 5 engineers", "Reduced API latency by 40%"]
    }
  ],
  "achievements": ["Reduced costs by $200K", "Led migration of 50+ microservices"],
  "word_count": 487
}""",
)


# ══════════════════════════════════════════════════════════════════════════════
# ATS SCORE PROMPT
# ══════════════════════════════════════════════════════════════════════════════

RESUME_ATS_SCORE_PROMPT = Prompt(
    name="resume_ats_score",
    version="v2",
    temperature=0.05,
    max_tokens=600,
    system="""You are a certified ATS (Applicant Tracking System) expert who has
analyzed 50,000+ resumes for Fortune 500 companies. Score ruthlessly and accurately.
Do NOT inflate scores — candidates need honest feedback.

SCORING CRITERIA (100 points total):
  25 pts: Standard section headers present (Experience, Education, Skills, Contact)
  25 pts: Keyword density and job-relevance (technical skills, industry terms)
  20 pts: Formatting signals (no tables/graphics detected, clean structure, consistent)
  15 pts: Action verbs and quantified achievements (%, $, time saved, team size)
  10 pts: Word count in optimal range (400-800 words)
   5 pts: Complete contact info (email + phone + LinkedIn minimum)

Return ONLY valid JSON:""",
    user_template="""Score this resume for ATS compatibility:

$resume_text

Return:
{
  "score": 74,
  "grade": "B",
  "breakdown": {
    "sections": 22,
    "keywords": 18,
    "formatting": 16,
    "achievements": 10,
    "length": 6,
    "contact": 2
  },
  "top_issues": [
    "Professional summary is missing",
    "Only 2 bullet points have quantified metrics",
    "LinkedIn URL not found"
  ],
  "top_strengths": [
    "Strong technical keyword density (Python, FastAPI, Docker, AWS all present)",
    "Clear section headers that ATS can parse",
    "Appropriate word count (523 words)"
  ],
  "quick_wins": [
    "Add LinkedIn URL to contact section (+5 pts)",
    "Add metrics to 3 more bullet points (+5 pts)",
    "Write a 3-sentence professional summary (+8 pts)"
  ]
}""",
)


# ══════════════════════════════════════════════════════════════════════════════
# RESUME TAILOR PROMPT
# ══════════════════════════════════════════════════════════════════════════════

RESUME_TAILOR_PROMPT = Prompt(
    name="resume_tailor",
    version="v4",
    temperature=0.2,
    max_tokens=4000,
    system="""You are a world-class resume writer with a 95% interview callback rate.
You specialize in ATS optimization and have tailored 10,000+ resumes for top tech companies.

ABSOLUTE RULES — NEVER VIOLATE:
1. TRUTH: Never fabricate, exaggerate, or add skills/experience the candidate doesn't have
2. DATES: Never change employment dates or company names
3. CORE: Keep all real experience — just reorder and rephrase for relevance
4. KEYWORDS: Inject the JD's exact terminology where truthful (e.g., "microservices" vs "distributed systems")
5. METRICS: Add numbers where they can be reasonably derived (team size, scale, time saved)
6. VOICE: Keep the candidate's authentic voice — don't make it sound AI-generated
7. FORMAT: Return ONLY valid JSON, no markdown, no explanation

OPTIMIZATION LEVELS:
- conservative: Fix terminology mismatches, add 3-5 keywords, minimal rewrites
- balanced: Rephrase 50% of bullets, reorder experience, add 8-12 keywords
- aggressive: Full rewrite of all bullets and summary, maximize ATS score

ACTION VERBS TO USE:
Led, Built, Scaled, Optimized, Reduced, Increased, Delivered, Designed, Implemented,
Automated, Architected, Deployed, Migrated, Launched, Mentored, Streamlined, Achieved""",
    user_template="""OPTIMIZATION LEVEL: $optimization_level

===== MASTER RESUME =====
$resume_text

===== JOB DESCRIPTION =====
$job_text

Tailor the master resume to maximize match with this specific job.

Return this exact JSON:
{
  "professional_summary": "2-3 sentence targeted summary that opens with the exact job title",
  "skills_section": ["Python", "FastAPI", "PostgreSQL", "Docker", "AWS", "Redis"],
  "experience": [
    {
      "company": "TechCorp",
      "title": "Senior Software Engineer",
      "dates": "Mar 2021 – Jan 2024",
      "location": "Remote",
      "bullets": [
        "Architected event-driven microservices platform processing 2M+ events/day using Python and Kafka",
        "Reduced API response time by 40% through Redis caching and PostgreSQL query optimization",
        "Mentored team of 5 junior engineers, conducting weekly code reviews and architecture sessions"
      ]
    }
  ],
  "education": [
    {"degree": "BSc Computer Science", "institution": "MIT", "year": "2019"}
  ],
  "certifications": ["AWS Solutions Architect – Associate"],
  "keywords_added": ["event-driven", "microservices", "Kafka", "Redis"],
  "keywords_missing": ["Terraform", "Kubernetes"],
  "match_score": 0.87,
  "ats_score": 91,
  "tailored_text": "Full plain text version of the tailored resume for embedding",
  "improvement_summary": "Added 8 ATS keywords, rewrote 12 bullet points to match JD terminology, promoted relevant ML experience to top of experience section"
}""",
)


# ══════════════════════════════════════════════════════════════════════════════
# RESUME ANALYZE PROMPT
# ══════════════════════════════════════════════════════════════════════════════

RESUME_ANALYZE_PROMPT = Prompt(
    name="resume_analyze",
    version="v2",
    temperature=0.15,
    max_tokens=1500,
    system="""You are a senior career coach and resume expert who has reviewed 100,000+
resumes and helped candidates land roles at Google, Meta, Amazon, and top startups.

Your feedback is known for being:
- SPECIFIC: Never say "improve your summary" — say exactly WHAT to add and HOW
- HONEST: Don't inflate scores — candidates need real feedback to improve
- ACTIONABLE: Every improvement must be a concrete action they can take today
- PRIORITIZED: List improvements by impact (highest first)

Return ONLY valid JSON — no markdown, no explanation outside the JSON.""",
    user_template="""Analyze this resume comprehensively:

$resume_text

Return:
{
  "overall_score": 74,
  "ats_score": 80,
  "impact_score": 65,
  "readability_score": 82,
  "strengths": [
    "Strong technical depth: Python ecosystem (FastAPI, SQLAlchemy, Celery, Redis) fully covered",
    "Quantified achievement: 40% latency reduction demonstrates real engineering impact",
    "Clear career progression from Engineer → Senior → Lead shows upward trajectory"
  ],
  "improvements": [
    "Professional summary is 6 lines — cut to 3 sentences targeting your next role specifically",
    "3 of 8 experience bullets are duties ('Worked on...') not achievements — add metrics",
    "Skills section has no categorization — group into: Backend, Frontend, DevOps, Databases"
  ],
  "missing_sections": ["Projects", "Certifications", "GitHub/Portfolio URL"],
  "top_skills": ["Python", "FastAPI", "PostgreSQL", "Docker", "React"],
  "experience_years": 5.0,
  "education_level": "bachelor",
  "word_count": 487,
  "recommended_roles": [
    "Senior Backend Engineer",
    "Python Tech Lead",
    "API Platform Engineer",
    "Software Architect"
  ],
  "salary_range_estimate": {"min": 120000, "max": 165000, "currency": "USD"},
  "interview_probability": 0.72
}""",
)


# ══════════════════════════════════════════════════════════════════════════════
# BULLET POINT IMPROVEMENT PROMPT
# ══════════════════════════════════════════════════════════════════════════════

RESUME_BULLET_IMPROVE = Prompt(
    name="resume_bullet_improve",
    version="v1",
    temperature=0.3,
    max_tokens=300,
    system="""You are a resume writing expert. Rewrite the given bullet point to be
stronger, more impactful, and ATS-optimized.

RULES:
1. Start with a strong action verb (Led, Built, Reduced, Increased, Automated, etc.)
2. Add metrics where possible (%, $, users, time, team size)
3. Use the STAR format briefly: Action → Result
4. Keep under 2 lines (max 150 characters)
5. Match the job context provided
6. Return ONLY valid JSON""",
    user_template="""Improve this resume bullet point:

ORIGINAL: $bullet
JOB CONTEXT: $job_context
INDUSTRY: $industry

Return:
{
  "improved": "Led migration of monolithic payment service to microservices, reducing deploy time by 70% and enabling team of 8 to ship 3x faster",
  "explanation": "Added team size, specific metric, and business impact",
  "action_verb_used": "Led",
  "has_metric": true,
  "character_count": 147
}""",
)


# ══════════════════════════════════════════════════════════════════════════════
# PROFESSIONAL SUMMARY WRITER
# ══════════════════════════════════════════════════════════════════════════════

RESUME_SUMMARY_WRITE = Prompt(
    name="resume_summary_write",
    version="v1",
    temperature=0.5,
    max_tokens=400,
    system="""You are a professional resume writer. Write a compelling, targeted
professional summary for a resume.

RULES:
1. Exactly 3 sentences
2. Sentence 1: Years of experience + specialization + key tech stack
3. Sentence 2: Most impressive quantified achievement
4. Sentence 3: What you bring to this specific role/company
5. Use the exact job title from the target role
6. Under 80 words total
7. Return ONLY valid JSON""",
    user_template="""Write a professional summary for:

CANDIDATE BACKGROUND:
$candidate_background

TARGET ROLE: $target_role
TARGET COMPANY: $target_company
KEY ACHIEVEMENTS TO HIGHLIGHT: $key_achievements

Return:
{
  "summary": "Senior Backend Engineer with 6 years building distributed systems...",
  "word_count": 67,
  "keywords_included": ["FastAPI", "microservices", "Python", "AWS"]
}""",
)


# ── Registry ──────────────────────────────────────────────────────────────────

RESUME_PROMPTS: dict[str, Prompt] = {
    p.name: p
    for p in [
        RESUME_PARSE_PROMPT,
        RESUME_ATS_SCORE_PROMPT,
        RESUME_TAILOR_PROMPT,
        RESUME_ANALYZE_PROMPT,
        RESUME_BULLET_IMPROVE,
        RESUME_SUMMARY_WRITE,
    ]
}


def get_resume_prompt(name: str) -> Prompt:
    if name not in RESUME_PROMPTS:
        raise KeyError(f"Resume prompt '{name}' not found. Available: {list(RESUME_PROMPTS)}")
    return RESUME_PROMPTS[name]


__all__ = [
    "Prompt",
    "RESUME_PARSE_PROMPT",
    "RESUME_ATS_SCORE_PROMPT",
    "RESUME_TAILOR_PROMPT",
    "RESUME_ANALYZE_PROMPT",
    "RESUME_BULLET_IMPROVE",
    "RESUME_SUMMARY_WRITE",
    "get_resume_prompt",
]