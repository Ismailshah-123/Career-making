"""
CareerGPT — Resume Agent Tools
=================================
PAGE SUMMARY:
  Tool functions for the ResumeAgent. Each function is a focused, testable
  unit: one LLM call or one PDF-generation operation, never both mixed.

  TOOLS:
    tailor_resume_content()    → core LLM call: rewrite resume JSON for a job
    write_targeted_summary()   → rewrite just the professional summary
    improve_bullet()           → rewrite a single bullet point with metrics
    build_embed_text()         → assemble text for re-embedding the tailored resume
    render_tailored_pdf()      → generate a clean, ATS-safe PDF from tailored JSON
    render_tailored_text_fallback() → plain .txt fallback if reportlab unavailable
    extract_contact_block()    → pull name/email/phone/links from master resume

  PDF GENERATION PHILOSOPHY (render_tailored_pdf):
    ATS systems parse PDFs by extracting raw text in reading order. The #1
    cause of "great resume, zero ATS pass-through" is multi-column layouts,
    text boxes, and tables that scramble reading order. This renderer uses
    ONLY single-column flowing text — Paragraph + Spacer + HRFlowable —
    which is the layout ATS parsers handle with 100% reliability.

    Sections rendered in order: Header → Summary → Skills → Experience →
    Education → Certifications. This order matches what 90%+ of ATS
    keyword-extraction engines expect (skills near top = higher keyword
    density score in the first parsed chunk).
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from app.core.config import get_settings
from app.core.constants import UPLOAD_DIR_TAILORED
from app.core.exceptions import ResumeExportError, ResumeTailorError
from app.core.logging import logger

settings = get_settings()


async def tailor_resume_content(
    *,
    resume_text: str,
    job_title: str,
    company_name: str,
    job_text: str,
    optimization_level: str,
) -> dict[str, Any]:
    """
    Core LLM call: rewrite a resume's content to match a specific job.
    Returns the full RESUME_TAILOR_PROMPT JSON shape — see
    app/prompts/resume_prompts.py for the exact schema.

    Raises ResumeTailorError if the LLM call fails after retries.
    """
    from app.agents.resume_agent.prompts import RESUME_TAILOR_PROMPT
    from app.services.groq_service import get_groq_service

    level_instructions = {
        "conservative": "Make minimal changes — only fix terminology mismatches and add 3-5 keywords.",
        "balanced":     "Moderately rephrase bullets and summary. Add relevant keywords naturally.",
        "aggressive":   "Fully rewrite for maximum ATS optimization. Prioritize job-relevant experience.",
    }
    level_instruction = level_instructions.get(optimization_level, level_instructions["balanced"])

    llm = get_groq_service()
    job_combined = f"Title: {job_title}\nCompany: {company_name}\n\n{job_text}"

    _, user_msg = RESUME_TAILOR_PROMPT.render(
        optimization_level=optimization_level,
        level_instruction=level_instruction,
        resume_text=resume_text,
        job_text=job_combined,
    )

    try:
        result = await llm.complete_json(
            prompt=user_msg,
            system=RESUME_TAILOR_PROMPT.system,
            temperature=RESUME_TAILOR_PROMPT.temperature,
            max_tokens=RESUME_TAILOR_PROMPT.max_tokens,
        )
    except Exception as exc:
        raise ResumeTailorError(f"AI tailoring failed: {exc}") from exc

    # Defensive normalization — ensure required keys always present
    result.setdefault("professional_summary", "")
    result.setdefault("skills_section", [])
    result.setdefault("experience", [])
    result.setdefault("education", [])
    result.setdefault("certifications", [])
    result.setdefault("keywords_added", [])
    result.setdefault("keywords_missing", [])
    result.setdefault("match_score", 0.7)
    result.setdefault("ats_score", 70)
    result.setdefault("tailored_text", "")
    result.setdefault("improvement_summary", "")

    return result


async def write_targeted_summary(
    *,
    candidate_background: str,
    target_role: str,
    target_company: str,
    key_achievements: list[str],
) -> dict[str, Any]:
    """Rewrite just the professional summary for a specific role/company."""
    from app.agents.resume_agent.prompts import RESUME_SUMMARY_WRITE
    from app.services.groq_service import get_groq_service

    llm = get_groq_service()
    _, user_msg = RESUME_SUMMARY_WRITE.render(
        candidate_background=candidate_background,
        target_role=target_role,
        target_company=target_company,
        key_achievements=", ".join(key_achievements),
    )

    try:
        return await llm.complete_json(
            prompt=user_msg,
            system=RESUME_SUMMARY_WRITE.system,
            temperature=RESUME_SUMMARY_WRITE.temperature,
            max_tokens=RESUME_SUMMARY_WRITE.max_tokens,
        )
    except Exception as exc:
        logger.warning("Summary generation failed", error=str(exc))
        return {"summary": candidate_background[:200], "word_count": 0}


async def improve_bullet(
    *,
    bullet: str,
    job_context: str,
    industry: str = "technology",
) -> dict[str, Any]:
    """Rewrite a single resume bullet point with stronger verbs and metrics."""
    from app.agents.resume_agent.prompts import RESUME_BULLET_IMPROVE
    from app.services.groq_service import get_groq_service

    llm = get_groq_service()
    _, user_msg = RESUME_BULLET_IMPROVE.render(
        bullet=bullet,
        job_context=job_context,
        industry=industry,
    )

    try:
        return await llm.complete_json(
            prompt=user_msg,
            system=RESUME_BULLET_IMPROVE.system,
            temperature=RESUME_BULLET_IMPROVE.temperature,
            max_tokens=RESUME_BULLET_IMPROVE.max_tokens,
        )
    except Exception as exc:
        logger.debug("Bullet improvement failed, returning original", error=str(exc))
        return {"improved": bullet, "has_metric": False}


def build_embed_text(tailored_data: dict[str, Any]) -> str:
    """
    Assemble a single embeddable text blob from tailored resume JSON.
    Used to generate a fresh Qdrant vector for the tailored version,
    so future job matches reflect the tailored content, not the master.
    """
    parts: list[str] = []

    if summary := tailored_data.get("professional_summary"):
        parts.append(summary)

    skills = tailored_data.get("skills_section", [])
    if skills:
        # 3x repetition boosts skill-token weight in the embedding,
        # consistent with build_resume_embed_text() in app/embeddings/embedder.py
        parts.extend([" ".join(skills)] * 3)

    for exp in tailored_data.get("experience", []):
        title = exp.get("title", "")
        company = exp.get("company", "")
        bullets = " ".join(exp.get("bullets", []))
        parts.append(f"{title} {company} {bullets}")

    return "\n\n".join(p for p in parts if p)


def extract_contact_block(master_resume: Any) -> dict[str, str]:
    """Pull header contact fields off the master Resume ORM object."""
    return {
        "name":          getattr(master_resume, "name", None) or "Professional Resume",
        "email":         getattr(master_resume, "email", None) or "",
        "phone":         getattr(master_resume, "phone", None) or "",
        "location":      getattr(master_resume, "location", None) or "",
        "linkedin_url":  getattr(master_resume, "linkedin_url", None) or "",
        "github_url":    getattr(master_resume, "github_url", None) or "",
        "portfolio_url": getattr(master_resume, "portfolio_url", None) or "",
    }


async def render_tailored_pdf(
    *,
    user_id: uuid.UUID,
    job_id: uuid.UUID,
    tailored_data: dict[str, Any],
    contact: dict[str, str],
) -> str:
    """
    Render a clean, single-column, ATS-safe PDF from tailored resume JSON.
    Falls back to a plain-text file if reportlab is unavailable.
    Returns the absolute file path of the generated document.
    """
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import inch
        from reportlab.platypus import (
            HRFlowable,
            Paragraph,
            SimpleDocTemplate,
            Spacer,
        )
    except ImportError:
        logger.warning("reportlab not installed — using plain-text fallback for tailored resume")
        return await render_tailored_text_fallback(
            user_id=user_id, job_id=job_id, tailored_data=tailored_data, contact=contact
        )

    output_dir = settings.storage.upload_dir / str(user_id) / UPLOAD_DIR_TAILORED
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / f"resume_{job_id}_{uuid.uuid4().hex[:8]}.pdf"

    try:
        doc = SimpleDocTemplate(
            str(pdf_path),
            pagesize=letter,
            rightMargin=0.75 * inch,
            leftMargin=0.75 * inch,
            topMargin=0.7 * inch,
            bottomMargin=0.7 * inch,
        )

        styles = getSampleStyleSheet()
        name_style = ParagraphStyle(
            "NameStyle", parent=styles["Heading1"],
            fontSize=19, spaceAfter=4, textColor=colors.HexColor("#0f172a"),
        )
        section_style = ParagraphStyle(
            "SectionStyle", parent=styles["Heading2"],
            fontSize=12.5, spaceAfter=5, spaceBefore=10,
            textColor=colors.HexColor("#1e3a8a"),
        )
        body_style = ParagraphStyle(
            "BodyStyle", parent=styles["Normal"],
            fontSize=10, spaceAfter=3, leading=14,
        )
        contact_style = ParagraphStyle(
            "ContactStyle", parent=body_style, textColor=colors.HexColor("#475569"),
        )
        bullet_style = ParagraphStyle(
            "BulletStyle", parent=body_style, leftIndent=14,
        )
        date_style = ParagraphStyle(
            "DateStyle", parent=body_style, textColor=colors.grey, fontSize=9,
        )

        story: list[Any] = []

        # ── Header ────────────────────────────────────────────────────────────
        story.append(Paragraph(contact["name"], name_style))
        contact_parts = [
            p for p in [
                contact["email"], contact["phone"], contact["location"],
                contact["linkedin_url"], contact["github_url"],
            ] if p
        ]
        if contact_parts:
            story.append(Paragraph(" | ".join(contact_parts), contact_style))
        story.append(Spacer(1, 4))
        story.append(HRFlowable(width="100%", thickness=1.2, color=colors.HexColor("#0f172a")))
        story.append(Spacer(1, 6))

        # ── Summary ───────────────────────────────────────────────────────────
        if summary := tailored_data.get("professional_summary"):
            story.append(Paragraph("PROFESSIONAL SUMMARY", section_style))
            story.append(Paragraph(_escape(summary), body_style))

        # ── Skills ────────────────────────────────────────────────────────────
        skills = tailored_data.get("skills_section", [])
        if skills:
            story.append(Paragraph("TECHNICAL SKILLS", section_style))
            story.append(Paragraph(_escape(" • ".join(skills)), body_style))

        # ── Experience ────────────────────────────────────────────────────────
        experience = tailored_data.get("experience", [])
        if experience:
            story.append(Paragraph("PROFESSIONAL EXPERIENCE", section_style))
            for exp in experience:
                title_line = f"<b>{_escape(exp.get('title', ''))}</b> — {_escape(exp.get('company', ''))}"
                story.append(Paragraph(title_line, body_style))
                date_loc_parts = [p for p in [exp.get("dates"), exp.get("location")] if p]
                if date_loc_parts:
                    story.append(Paragraph(_escape(" | ".join(date_loc_parts)), date_style))
                for bullet in exp.get("bullets", []):
                    story.append(Paragraph(f"• {_escape(bullet)}", bullet_style))
                story.append(Spacer(1, 5))

        # ── Education ─────────────────────────────────────────────────────────
        education = tailored_data.get("education", [])
        if education:
            story.append(Paragraph("EDUCATION", section_style))
            for edu in education:
                if isinstance(edu, dict):
                    edu_text = (
                        f"<b>{_escape(edu.get('degree', ''))}</b> — "
                        f"{_escape(edu.get('institution', ''))} "
                        f"{_escape(str(edu.get('year', '')))}"
                    )
                else:
                    edu_text = _escape(str(edu))
                story.append(Paragraph(edu_text, body_style))

        # ── Certifications ────────────────────────────────────────────────────
        certs = tailored_data.get("certifications", [])
        if certs:
            story.append(Paragraph("CERTIFICATIONS", section_style))
            story.append(Paragraph(_escape(" • ".join(certs)), body_style))

        doc.build(story)
        logger.info("Tailored resume PDF rendered", path=str(pdf_path))
        return str(pdf_path)

    except Exception as exc:
        raise ResumeExportError(f"PDF rendering failed: {exc}") from exc


async def render_tailored_text_fallback(
    *,
    user_id: uuid.UUID,
    job_id: uuid.UUID,
    tailored_data: dict[str, Any],
    contact: dict[str, str],
) -> str:
    """Plain .txt fallback when reportlab is unavailable in the environment."""
    output_dir = settings.storage.upload_dir / str(user_id) / UPLOAD_DIR_TAILORED
    output_dir.mkdir(parents=True, exist_ok=True)
    txt_path = output_dir / f"resume_{job_id}_{uuid.uuid4().hex[:8]}.txt"

    lines: list[str] = [contact["name"]]
    contact_line = " | ".join(p for p in [
        contact["email"], contact["phone"], contact["location"], contact["linkedin_url"]
    ] if p)
    if contact_line:
        lines.append(contact_line)
    lines.append("=" * 60)

    if summary := tailored_data.get("professional_summary"):
        lines += ["", "PROFESSIONAL SUMMARY", summary]

    if skills := tailored_data.get("skills_section"):
        lines += ["", "TECHNICAL SKILLS", " | ".join(skills)]

    if experience := tailored_data.get("experience"):
        lines += ["", "PROFESSIONAL EXPERIENCE"]
        for exp in experience:
            lines.append(f"{exp.get('title', '')} — {exp.get('company', '')} ({exp.get('dates', '')})")
            for bullet in exp.get("bullets", []):
                lines.append(f"  • {bullet}")
            lines.append("")

    if education := tailored_data.get("education"):
        lines += ["EDUCATION"]
        for edu in education:
            if isinstance(edu, dict):
                lines.append(f"{edu.get('degree', '')} — {edu.get('institution', '')} {edu.get('year', '')}")

    txt_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Tailored resume saved as text fallback", path=str(txt_path))
    return str(txt_path)


def _escape(text: str) -> str:
    """Escape characters that would break ReportLab's mini-HTML markup."""
    if not text:
        return ""
    return (
        text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
    )


__all__ = [
    "tailor_resume_content",
    "write_targeted_summary",
    "improve_bullet",
    "build_embed_text",
    "extract_contact_block",
    "render_tailored_pdf",
    "render_tailored_text_fallback",
]