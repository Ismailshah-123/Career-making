"""
CareerGPT — Resume Agent
===========================
PAGE SUMMARY:
  Orchestrates resume_agent/tools.py (LLM calls + PDF rendering) into the
  three entry points the rest of the app calls:

  CONTRACT:
    agent = ResumeAgent(db)                 # db optional for parse/tailor_sections
    result = await agent.parse_raw_text(raw_text=...)
      -> {"sections": dict, "skills": [str], "job_titles": [str],
          "years_of_experience": float, "usage": {"prompt_tokens", "completion_tokens"}}

    result = await agent.tailor_sections(
        original_sections=dict, job_id=UUID, tone=str, emphasise_skills=[str]
    )
      -> {"sections": dict, "changes_summary": str, "prompt_used": str,
          "usage": {"prompt_tokens", "completion_tokens"}}

    agent = ResumeAgent(db)                 # or ResumeAgent() -- db optional
    result = await agent.tailor(master_resume=<Resume ORM>, job=<Job ORM>,
                                 optimization_level=str)
      -> {"pdf_path": str, "tailored_text": str, "ats_score": int,
          "skills_section": [str]}

  Both parse_raw_text/tailor_sections back resume_workflow.py's LangGraph
  nodes; tailor() backs resume_service.py's simpler one-shot flow.

  TOKEN USAGE:
    groq_service.complete_json() doesn't return token counts, so usage is
    estimated via GroqService.count_tokens() on the prompt/result text --
    good enough for the cost dashboards, not billing-grade precision.
"""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import logger
from app.agents.resume_agent import tools


class ResumeAgent:
    """
    Resume parsing / tailoring orchestrator.

    Usage:
        agent = ResumeAgent(db)
        parsed = await agent.parse_raw_text(raw_text=extracted_text)
        tailored = await agent.tailor_sections(
            original_sections=parsed["sections"], job_id=job.id, tone="professional",
        )

    One-shot usage (resume_service.py):
        agent = ResumeAgent(db)
        result = await agent.tailor(master_resume=resume, job=job, optimization_level="balanced")
    """

    def __init__(self, db: AsyncSession | None = None) -> None:
        self.db = db

    # -- Parsing ----------------------------------------------------------------

    async def parse_raw_text(self, *, raw_text: str) -> dict[str, Any]:
        """Extract structured fields from raw extracted resume text."""
        from app.prompts.resume_prompts import RESUME_PARSE_PROMPT
        from app.services.groq_service import get_groq_service
        from app.core.constants import RESUME_EXTRACT_CHAR_LIMIT

        llm = get_groq_service()
        text = (raw_text or "")[:RESUME_EXTRACT_CHAR_LIMIT]
        system, user_msg = RESUME_PARSE_PROMPT.render(resume_text=text)

        sections = await llm.complete_json(
            user_msg,
            system=system,
            temperature=RESUME_PARSE_PROMPT.temperature,
            max_tokens=RESUME_PARSE_PROMPT.max_tokens,
        )

        work_experience = sections.get("work_experience") or []
        job_titles = [w.get("title") for w in work_experience if w.get("title")]

        prompt_tokens = llm.count_tokens(user_msg)
        completion_tokens = llm.count_tokens(str(sections))

        return {
            "sections": sections,
            "skills": sections.get("skills", []) or [],
            "job_titles": job_titles,
            "years_of_experience": sections.get("experience_years", 0) or 0,
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
        }

    # -- Tailoring (workflow node) -----------------------------------------------

    async def tailor_sections(
        self,
        *,
        original_sections: dict[str, Any],
        job_id: uuid.UUID,
        tone: str = "professional",
        emphasise_skills: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Tailor a parsed-sections dict against a specific job. Used by
        resume_workflow.py's tailoring node, which already has `job_id`
        rather than a loaded Job ORM object.
        """
        from app.prompts.resume_prompts import RESUME_TAILOR_PROMPT
        from app.services.groq_service import get_groq_service

        job_title, company_name, job_text = "", "", ""
        if self.db is not None:
            from app.db.models.job import Job
            from sqlalchemy import select

            job = (await self.db.execute(select(Job).where(Job.id == job_id))).scalar_one_or_none()
            if job:
                job_title = job.title or ""
                company_name = job.company or ""
                job_text = job.description or ""

        resume_text = self._sections_to_text(original_sections)
        if emphasise_skills:
            resume_text += "\n\nPRIORITY SKILLS TO EMPHASISE: " + ", ".join(emphasise_skills)

        llm = get_groq_service()
        system, user_msg = RESUME_TAILOR_PROMPT.render(
            optimization_level=tone or "balanced",
            resume_text=resume_text,
            job_text=job_text or "(no job description available)",
        )
        result = await llm.complete_json(
            user_msg,
            system=system,
            temperature=RESUME_TAILOR_PROMPT.temperature,
            max_tokens=RESUME_TAILOR_PROMPT.max_tokens,
        )

        prompt_tokens = llm.count_tokens(user_msg)
        completion_tokens = llm.count_tokens(str(result))

        return {
            "sections": result,
            "changes_summary": result.get("improvement_summary", ""),
            "prompt_used": RESUME_TAILOR_PROMPT.name,
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
        }

    # -- One-shot tailor (resume_service.py) -------------------------------------

    async def tailor(
        self,
        *,
        master_resume: Any,
        job: Any,
        optimization_level: str = "aggressive",
    ) -> dict[str, Any]:
        """
        End-to-end: tailor the master resume against `job` and render a PDF.
        Returns the flat dict resume_service.py stores on the Resume row.
        """
        resume_text = master_resume.raw_text or self._sections_to_text(
            getattr(master_resume, "parsed_sections", None) or {}
        )

        result = await tools.tailor_resume_content(
            resume_text=resume_text,
            job_title=getattr(job, "title", "") or "",
            company_name=getattr(job, "company", "") or "",
            job_text=getattr(job, "description", "") or "",
            optimization_level=optimization_level,
        )

        contact = tools.extract_contact_block(master_resume)
        try:
            pdf_path = await tools.render_tailored_pdf(
                user_id=master_resume.user_id,
                job_id=job.id,
                tailored_data=result,
                contact=contact,
            )
        except Exception as exc:
            logger.warning("PDF render failed, falling back to text", error=str(exc))
            pdf_path = await tools.render_tailored_text_fallback(
                user_id=master_resume.user_id,
                job_id=job.id,
                tailored_data=result,
                contact=contact,
            )

        return {
            "pdf_path": pdf_path,
            "tailored_text": result.get("tailored_text", ""),
            "ats_score": result.get("ats_score", 0),
            "skills_section": result.get("skills_section", []),
        }

    # -- Private helpers ----------------------------------------------------------

    @staticmethod
    def _sections_to_text(sections: dict[str, Any]) -> str:
        """Flatten a parsed-sections dict back into plain resume text."""
        if not sections:
            return ""

        parts: list[str] = []
        if summary := sections.get("summary"):
            parts.append(f"SUMMARY\n{summary}")

        skills = sections.get("skills")
        if isinstance(skills, dict):
            flat_skills = [s for group in skills.values() for s in (group or [])]
        else:
            flat_skills = skills or []
        if flat_skills:
            parts.append("SKILLS\n" + ", ".join(flat_skills))

        for job in sections.get("work_experience", []) or []:
            header = f"{job.get('title', '')} at {job.get('company', '')} ({job.get('start_date', '')} - {job.get('end_date', 'Present')})"
            bullets = "\n".join(f"- {b}" for b in job.get("bullets", []) or [])
            parts.append(f"{header}\n{bullets}")

        for edu in sections.get("education", []) or []:
            parts.append(f"{edu.get('degree', '')}, {edu.get('institution', '')} ({edu.get('year', '')})")

        if certs := sections.get("certifications"):
            parts.append("CERTIFICATIONS\n" + ", ".join(certs))

        return "\n\n".join(parts)


__all__ = ["ResumeAgent"]
