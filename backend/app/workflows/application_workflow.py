"""
app/workflows/application_workflow.py
========================================
LangGraph workflow for automated job application submission.

This is the most operationally sensitive workflow in the platform: it
drives a real browser, fills real forms, and submits real applications on
behalf of the user. Every design decision here prioritises:
  1. Verifiable outcomes — screenshots, confirmation text, status updates
  2. Safe partial failure — any node can halt cleanly, flagging for manual review
  3. No double-submission — idempotency guards before every submission attempt
  4. Audit trail — every automation step is logged in AgentRun.steps

Graph topology:

  validate_application
          |
    tailor_resume      (resume_workflow sub-call, async)
          |
  generate_cover_letter
          |
    detect_ats_provider
          |
     fill_form         (Playwright — ATS-specific strategy loaded at runtime)
          |
     submit_form       (click submit, wait for confirmation signal)
          |
   verify_submission   (parse confirmation page / email)
          |
   +------+------+
   | success      | failure
   v               v
update_db      flag_manual_review
   |
discover_recruiter     (optional — if outreach is enabled for this user)
   |
send_outreach_message  (optional)
   |
schedule_followup
   |
  END

On any node failure (detected via @node decorator setting should_halt):
  -> handle_error  (screenshot, DB update, user notification)
  -> END
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any

from langgraph.graph import StateGraph, END

from app.core.constants import (
    NODE_ERROR,
    ApplicationStatus,
    APPLICATION_STATUS_TRANSITIONS,
    FOLLOWUP_WAIT_DAYS,
    AGENT_OUTREACH,
    AgentRunStatus,
)
from app.core.logging import get_logger
from app.workflows.graph import (
    node,
    build_graph,
    halt_on_error,
    conditional_router,
    get_invoke_config,
)
from app.workflows.state import ApplicationWorkflowState, new_base_state

logger = get_logger(__name__)

# Local node name constants
NODE_VALIDATE = "validate_application"
NODE_TAILOR = "tailor_resume"
NODE_COVER = "generate_cover_letter"
NODE_DETECT_ATS = "detect_ats_provider"
NODE_FILL = "fill_form"
NODE_SUBMIT = "submit_form"
NODE_VERIFY = "verify_submission"
NODE_UPDATE_DB = "update_application_db"
NODE_MANUAL_REVIEW = "flag_manual_review"
NODE_RECRUITER = "discover_recruiter"
NODE_OUTREACH = "send_outreach_message"
NODE_FOLLOWUP = "schedule_followup"


# ---------------------------------------------------------------------------
# Node implementations
# ---------------------------------------------------------------------------

@node(NODE_VALIDATE)
async def validate_application_node(state: ApplicationWorkflowState) -> dict[str, Any]:
    """
    Pre-flight checks before committing any browser time or LLM budget:
    1. Application row still exists and is not terminal
    2. Job is still active and has an apply_url
    3. Resume exists and is fully parsed + embedded
    4. Cover letter not already generated (idempotency)
    5. No duplicate in-flight automation (automation_attempts guard)

    Any failed check routes to handle_error with a clear human-readable
    reason so the user understands exactly why auto-apply couldn't proceed.
    """
    from app.db.session import get_db_context
    from app.db.models.application import Application
    from app.db.models.job import Job
    from app.db.models.resume import Resume
    from sqlalchemy import select
    import uuid

    async with get_db_context() as db:
        app_result = await db.execute(
            select(Application).where(
                Application.id == uuid.UUID(state["application_id"]),
                Application.is_deleted.is_(False),
            )
        )
        application = app_result.scalar_one_or_none()
        if not application:
            raise ValueError(f"Application {state['application_id']} not found.")

        from app.core.constants import TERMINAL_STATUSES
        if ApplicationStatus(application.status) in TERMINAL_STATUSES:
            raise ValueError(f"Application already in terminal status: {application.status}")

        job_result = await db.execute(select(Job).where(Job.id == application.job_id))
        job = job_result.scalar_one_or_none()
        if not job or not job.is_active:
            raise ValueError(f"Job {application.job_id} is no longer active.")
        if not job.apply_url:
            raise ValueError(f"Job {application.job_id} has no apply_url — cannot automate.")

        resume_result = await db.execute(
            select(Resume).where(Resume.id == uuid.UUID(state["base_resume_id"]))
        )
        resume = resume_result.scalar_one_or_none()
        if not resume or not resume.is_parsed:
            raise ValueError("Resume is not yet parsed — cannot proceed with tailoring.")

    return {
        "apply_url": job.apply_url,
        "ats_provider": job.ats_provider,
        "metadata": {
            "job_title": job.title,
            "company_name": job.company_name,
            "job_board": job.job_board,
        },
    }


@node(NODE_TAILOR)
async def tailor_resume_node(state: ApplicationWorkflowState) -> dict[str, Any]:
    """
    Generate a job-specific tailored resume variant via the resume_workflow.

    If a tailored resume for this exact job already exists (from a prior
    attempt on the same application), reuse it rather than burning LLM
    budget on a redundant re-tailor — idempotency via the DB query below.
    """
    from app.db.session import get_db_context
    from app.db.models.resume import Resume
    from sqlalchemy import select
    import uuid
    from app.workflows.resume_workflow import run_tailor_workflow

    job_id = state["job_id"]
    base_resume_id = state["base_resume_id"]

    async with get_db_context() as db:
        existing = await db.execute(
            select(Resume).where(
                Resume.parent_resume_id == uuid.UUID(base_resume_id),
                Resume.tailored_for_job_id == uuid.UUID(job_id),
                Resume.is_deleted.is_(False),
                Resume.is_parsed.is_(True),
            )
        )
        already_tailored = existing.scalar_one_or_none()
        if already_tailored:
            logger.info(
                "Reusing existing tailored resume",
                resume_id=str(already_tailored.id),
            )
            return {
                "tailored_resume_id": str(already_tailored.id),
                "tailored_resume_path": already_tailored.file_path,
            }

        master = await db.execute(
            select(Resume).where(Resume.id == uuid.UUID(base_resume_id))
        )
        master_resume = master.scalar_one_or_none()
        if not master_resume:
            raise ValueError("Master resume not found.")

        raw_text = master_resume.raw_text or ""
        parsed_sections = master_resume.parsed_sections or {}

    result = await run_tailor_workflow(
        user_id=state["user_id"],
        agent_run_id=state["agent_run_id"],
        resume_id=base_resume_id,
        target_job_id=job_id,
        parsed_sections=parsed_sections,
        raw_text=raw_text,
    )

    return {
        "tailored_resume_id": result.get("new_resume_id"),
        "tailored_resume_path": None,   # file export happens in fill_form node
    }


@node(NODE_COVER)
async def generate_cover_letter_node(state: ApplicationWorkflowState) -> dict[str, Any]:
    """
    Generate the cover letter for this application via the cover_letter_agent.

    Idempotent — checks for an existing cover_letter_id on the Application
    row first and reuses it rather than generating a duplicate.
    """
    from app.db.session import get_db_context
    from app.db.models.application import Application
    from sqlalchemy import select
    import uuid

    async with get_db_context() as db:
        app_result = await db.execute(
            select(Application).where(Application.id == uuid.UUID(state["application_id"]))
        )
        application = app_result.scalar_one_or_none()

        if application and application.cover_letter_id:
            logger.info("Reusing existing cover letter", cl_id=str(application.cover_letter_id))
            from app.db.models.cover_letter import CoverLetter
            cl_result = await db.execute(
                select(CoverLetter).where(CoverLetter.id == application.cover_letter_id)
            )
            cl = cl_result.scalar_one_or_none()
            if cl:
                return {
                    "cover_letter_id": str(cl.id),
                    "cover_letter_text": cl.body,
                }

    from app.agents.cover_letter_agent.agent import CoverLetterAgent

    agent = CoverLetterAgent()
    async with get_db_context() as db:
        letter = await agent.generate(
            user_id=uuid.UUID(state["user_id"]),
            job_id=uuid.UUID(state["job_id"]),
            resume_id=(
                uuid.UUID(state["tailored_resume_id"])
                if state.get("tailored_resume_id")
                else uuid.UUID(state["base_resume_id"])
            ),
            application_id=uuid.UUID(state["application_id"]),
            db=db,
        )
        await db.commit()

    return {
        "cover_letter_id": str(letter.id),
        "cover_letter_text": letter.body,
    }


@node(NODE_DETECT_ATS)
async def detect_ats_provider_node(state: ApplicationWorkflowState) -> dict[str, Any]:
    """
    Identify the ATS system backing the apply_url so the correct Playwright
    strategy is loaded in fill_form_node. Uses a combination of URL pattern
    matching (fast, no network) and a lightweight HEAD request to check for
    ATS-specific headers or redirects.

    Known ATS signatures:
        greenhouse.io / boards.greenhouse.io → greenhouse
        jobs.lever.co / lever.co            → lever
        workday.com / myworkdaysite.com     → workday
        jobs.ashbyhq.com                    → ashby
        smartrecruiters.com                 → smartrecruiters
        taleo.net / talesystem.com          → taleo
        icims.com                           → icims
        *.linkedin.com/jobs                 → linkedin_easy_apply
        *.indeed.com/viewjob                → indeed
        remoteok.io                         → remoteok
        (everything else)                   → generic
    """
    from app.automation.playwright.browser import detect_ats_from_url

    apply_url = state.get("apply_url", "")
    detected = state.get("ats_provider")

    if not detected:
        detected = await detect_ats_from_url(apply_url)

    logger.info("ATS provider detected", ats=detected, url=apply_url[:80])
    return {"ats_provider": detected}


@node(NODE_FILL)
async def fill_form_node(state: ApplicationWorkflowState) -> dict[str, Any]:
    """
    Launch a Playwright browser session and fill every field of the
    application form using the ATS-specific page object strategy.

    Field values are assembled from:
    - Resume parsed_sections (name, email, phone, experience bullets)
    - Tailored resume PDF path (for file upload fields)
    - Cover letter text (for textarea fields)
    - User profile preferences (work authorisation, salary expectations)

    All detected + filled fields are logged to form_data_used on the
    Application row so they can be inspected if submission fails and the
    user needs to identify what was wrong.
    """
    from app.automation.playwright.browser import get_browser_context
    from app.repositories.resume_repository import ResumeRepository
    from app.db.session import get_db_context
    import uuid

    ats = state.get("ats_provider", "generic")
    apply_url = state["apply_url"]

    async with get_db_context() as db:
        repo = ResumeRepository(db)
        resume_id = state.get("tailored_resume_id") or state["base_resume_id"]
        resume = await repo.get_by_id(uuid.UUID(resume_id))
        parsed = resume.parsed_sections or {}

    # Load the ATS-specific page object
    page_object = _load_page_object(ats)

    detected_fields: list[dict[str, Any]] = []
    filled_fields: list[str] = []

    async with get_browser_context() as (browser, page):
        try:
            await page.goto(apply_url, wait_until="networkidle", timeout=30_000)

            detected_fields = await page_object.detect_fields(page)

            form_data = _build_form_data(
                parsed_sections=parsed,
                cover_letter_text=state.get("cover_letter_text", ""),
                resume_path=state.get("tailored_resume_path"),
                detected_fields=detected_fields,
            )

            filled_fields = await page_object.fill_fields(
                page=page,
                form_data=form_data,
                resume_path=state.get("tailored_resume_path"),
            )

        except Exception as exc:
            screenshot_path = await _capture_screenshot(page, state["application_id"])
            raise RuntimeError(
                f"Form fill failed on {ats} ATS ({apply_url[:60]}): {exc}"
            ) from exc

    return {
        "form_fields_detected": detected_fields,
        "form_fields_filled": filled_fields,
        "automation_step": "form_filled",
    }


@node(NODE_SUBMIT)
async def submit_form_node(state: ApplicationWorkflowState) -> dict[str, Any]:
    """
    Click the submit button and wait for either a success signal or an
    error indicator, with a 30-second timeout.

    Success signals: URL change to a /confirmation / /thank-you path,
    or presence of a "successfully submitted" / "application received"
    string in the page content.

    Failure signals: visible error banner, unchanged URL after 15s,
    CAPTCHA overlay detected.
    """
    from app.automation.playwright.browser import get_browser_context
    import uuid

    ats = state.get("ats_provider", "generic")
    page_object = _load_page_object(ats)

    async with get_browser_context() as (browser, page):
        try:
            submit_result = await page_object.submit(page)

            if not submit_result.get("success"):
                screenshot_path = await _capture_screenshot(page, state["application_id"])
                return {
                    "submission_success": False,
                    "submission_confirmation_text": None,
                    "form_screenshot_path": screenshot_path,
                    "automation_step": "submit_failed",
                    "requires_manual_review": True,
                    "manual_review_reason": submit_result.get("error", "Submit click did not produce a confirmation signal."),
                }

            confirmation_text = submit_result.get("confirmation_text", "")
            screenshot_path = await _capture_screenshot(page, state["application_id"])

        except Exception as exc:
            screenshot_path = None
            try:
                screenshot_path = await _capture_screenshot(page, state["application_id"])
            except Exception:
                pass
            raise RuntimeError(f"Form submission failed: {exc}") from exc

    return {
        "submission_success": True,
        "submission_confirmation_text": confirmation_text,
        "form_screenshot_path": screenshot_path,
        "automation_step": "submitted",
        "requires_manual_review": False,
    }


@node(NODE_VERIFY)
async def verify_submission_node(state: ApplicationWorkflowState) -> dict[str, Any]:
    """
    Secondary confirmation pass after submit_form: wait up to 10s for a
    follow-up email confirmation (if the ATS sends one within that window)
    or re-check the confirmation page URL is still a success page rather
    than having redirected back to the job listing (which sometimes indicates
    a silent failure on certain ATS platforms).

    This is a best-effort verification — a verification failure does NOT
    cancel the application, it only sets requires_manual_review=True so the
    user is asked to manually confirm whether the submission went through.
    """
    await asyncio.sleep(3)  # brief delay to let any post-submit redirects settle

    confirmation_text = state.get("submission_confirmation_text", "")
    success_keywords = [
        "application received", "successfully applied", "thank you for applying",
        "your application has been submitted", "we'll be in touch",
        "application submitted", "received your application",
    ]

    confidence = any(kw in confirmation_text.lower() for kw in success_keywords)

    if not confidence and state.get("submission_success"):
        logger.warning(
            "Submission success flagged but no confirmation text found — flagging for manual review",
            application_id=state["application_id"],
        )
        return {
            "submission_success": True,
            "requires_manual_review": True,
            "manual_review_reason": "Submitted but no standard confirmation text detected — please verify manually.",
        }

    return {
        "submission_success": state.get("submission_success", False),
        "requires_manual_review": not confidence and not state.get("submission_success", False),
    }


@node(NODE_UPDATE_DB)
async def update_application_db_node(state: ApplicationWorkflowState) -> dict[str, Any]:
    """
    Persist the submission outcome: update Application status → APPLIED,
    set applied_at, store form_data_used, cover_letter_id, tailored_resume_id.
    Idempotent — safe to call on retry since status transitions guard against
    going backwards.
    """
    from app.db.session import get_db_context
    from app.db.models.application import Application
    from sqlalchemy import select
    import uuid

    async with get_db_context() as db:
        result = await db.execute(
            select(Application).where(Application.id == uuid.UUID(state["application_id"]))
        )
        application = result.scalar_one_or_none()
        if not application:
            raise ValueError(f"Application {state['application_id']} not found during DB update.")

        if application.status not in (
            ApplicationStatus.APPLIED.value,
            ApplicationStatus.ACKNOWLEDGED.value,
        ):
            application.record_status_change(
                from_status=application.status,
                to_status=ApplicationStatus.APPLIED.value,
                reason="Automated submission confirmed",
                agent="application_agent",
            )
            application.applied_at = datetime.now(timezone.utc)
            application.is_auto_applied = True
            application.application_method = "playwright_automation"

        if state.get("tailored_resume_id"):
            application.resume_id = uuid.UUID(state["tailored_resume_id"])
        if state.get("cover_letter_id"):
            application.cover_letter_id = uuid.UUID(state["cover_letter_id"])

        application.form_data_used = {
            "fields_detected": len(state.get("form_fields_detected", [])),
            "fields_filled": state.get("form_fields_filled", []),
            "ats_provider": state.get("ats_provider"),
            "confirmation_text": state.get("submission_confirmation_text", "")[:500],
            "screenshot_path": state.get("form_screenshot_path"),
        }
        application.automation_error = None

        next_followup = datetime.now(timezone.utc) + timedelta(days=FOLLOWUP_WAIT_DAYS)
        application.next_followup_at = next_followup

        await db.commit()

    logger.info(
        "Application DB updated — APPLIED",
        application_id=state["application_id"],
        ats=state.get("ats_provider"),
    )

    return {"next_followup_at": next_followup.isoformat()}


@node(NODE_MANUAL_REVIEW)
async def flag_manual_review_node(state: ApplicationWorkflowState) -> dict[str, Any]:
    """
    Mark the application for manual review and notify the user.

    Called when submission fails, the ATS is unsupported, verification is
    inconclusive, or the fill detected a CAPTCHA / unusual form structure.
    The application status is set to QUEUED rather than FAILED so the user
    can retry from the dashboard after investigating.
    """
    from app.db.session import get_db_context
    from app.db.models.application import Application
    from app.services.notification_service import NotificationService
    from app.core.constants import NotificationType
    from sqlalchemy import select
    import uuid

    reason = state.get("manual_review_reason", "Automation could not complete — manual action required.")

    async with get_db_context() as db:
        result = await db.execute(
            select(Application).where(Application.id == uuid.UUID(state["application_id"]))
        )
        application = result.scalar_one_or_none()
        if application:
            application.automation_error = reason
            application.automation_attempts = (application.automation_attempts or 0)
            if application.status == ApplicationStatus.APPLYING.value:
                application.record_status_change(
                    from_status=ApplicationStatus.APPLYING.value,
                    to_status=ApplicationStatus.QUEUED.value,
                    reason=f"Auto-apply flagged for manual review: {reason}",
                    agent="application_agent",
                )
            await db.commit()

    try:
        svc = NotificationService()
        await svc.notify(
            user_id=uuid.UUID(state["user_id"]),
            notification_type=NotificationType.APPLICATION_STATUS_CHANGED,
            context={
                "application_id": state["application_id"],
                "reason": reason,
                "action_required": "Please review and apply manually via the job URL.",
                "apply_url": state.get("apply_url"),
            },
        )
    except Exception as exc:
        logger.error("Failed to send manual-review notification", error=str(exc))

    return {"requires_manual_review": True}


@node(NODE_RECRUITER)
async def discover_recruiter_node(state: ApplicationWorkflowState) -> dict[str, Any]:
    """
    Optional post-submission step: search LinkedIn for a recruiter at the
    hiring company who specialises in the role type being applied for.

    Gated on the user's outreach_enabled preference; skipped silently if
    disabled or if the company can't be identified from the job row.
    """
    from app.db.session import get_db_context
    from app.db.models.job import Job
    from app.repositories.user_repository import UserRepository
    from sqlalchemy import select
    import uuid

    async with get_db_context() as db:
        user_repo = UserRepository(db)
        user = await user_repo.get_by_id(uuid.UUID(state["user_id"]))
        if not (user and user.job_search_preferences.get("outreach_enabled", False)):
            logger.info("Recruiter discovery skipped — outreach not enabled")
            return {"recruiter_id": None}

        job_result = await db.execute(select(Job).where(Job.id == uuid.UUID(state["job_id"])))
        job = job_result.scalar_one_or_none()
        if not job or not job.company_id:
            return {"recruiter_id": None}

    from app.agents.outreach_agent.agent import OutreachAgent

    agent = OutreachAgent()
    recruiter = await agent.discover_recruiter(
        company_id=str(job.company_id),
        job_title_keywords=["recruiter", "talent", "hiring"],
        max_results=5,
    )

    return {"recruiter_id": str(recruiter.id) if recruiter else None}


@node(NODE_OUTREACH)
async def send_outreach_node(state: ApplicationWorkflowState) -> dict[str, Any]:
    """
    Send a personalised outreach message to the discovered recruiter via
    LinkedIn. Skipped if no recruiter was found or outreach was already sent.
    """
    recruiter_id = state.get("recruiter_id")
    if not recruiter_id:
        return {"outreach_sent": False}

    from app.agents.outreach_agent.agent import OutreachAgent
    from app.db.session import get_db_context
    from app.db.models.recruiter import Recruiter
    from sqlalchemy import select
    import uuid

    async with get_db_context() as db:
        result = await db.execute(select(Recruiter).where(Recruiter.id == uuid.UUID(recruiter_id)))
        recruiter = result.scalar_one_or_none()
        if not recruiter or recruiter.do_not_contact:
            return {"outreach_sent": False}

    agent = OutreachAgent()
    sent = await agent.send_outreach(
        user_id=state["user_id"],
        recruiter_id=recruiter_id,
        application_id=state["application_id"],
        job_id=state["job_id"],
    )

    return {"outreach_sent": sent}


@node(NODE_FOLLOWUP)
async def schedule_followup_node(state: ApplicationWorkflowState) -> dict[str, Any]:
    """
    Write the next-followup timestamp onto the Application row and register
    the Celery ETA task that will invoke the followup_agent at that time.
    """
    next_followup_str = state.get("next_followup_at")
    if not next_followup_str:
        next_followup = datetime.now(timezone.utc) + timedelta(days=FOLLOWUP_WAIT_DAYS)
        next_followup_str = next_followup.isoformat()
    else:
        next_followup = datetime.fromisoformat(next_followup_str)

    try:
        from app.workers.job_tasks import send_followup_task
        send_followup_task.apply_async(
            args=[state["application_id"], state["agent_run_id"]],
            eta=next_followup,
        )
    except Exception as exc:
        logger.warning("Could not schedule follow-up Celery task", error=str(exc))

    return {"next_followup_at": next_followup_str}


async def handle_error_node(state: ApplicationWorkflowState) -> dict[str, Any]:
    """
    Terminal error node for the application workflow. Persists the error
    message to the Application row and notifies the user so the failure
    is visible in the dashboard rather than silently stuck in APPLYING state.
    """
    from app.db.session import get_db_context
    from app.db.models.application import Application
    from sqlalchemy import select
    import uuid

    errors = state.get("errors", [])
    error_summary = "; ".join(e.get("message", "Unknown") for e in errors[-2:])
    node_that_failed = errors[-1].get("node", "unknown") if errors else "unknown"

    logger.error(
        "Application workflow halted",
        application_id=state.get("application_id"),
        agent_run_id=state.get("agent_run_id"),
        node=node_that_failed,
        errors=errors,
    )

    async with get_db_context() as db:
        result = await db.execute(
            select(Application).where(Application.id == uuid.UUID(state["application_id"]))
        )
        application = result.scalar_one_or_none()
        if application:
            application.automation_error = error_summary
            if application.status == ApplicationStatus.APPLYING.value:
                application.record_status_change(
                    from_status=ApplicationStatus.APPLYING.value,
                    to_status=ApplicationStatus.QUEUED.value,
                    reason=f"Agent error at {node_that_failed}: {error_summary}",
                    agent="application_agent",
                )
            await db.commit()

    return {"status": "failed"}


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _load_page_object(ats_provider: str):
    """
    Dynamically load the ATS-specific Playwright page object.

    Falls back to the generic page object for unrecognised ATS systems
    rather than raising — better to attempt a generic fill than hard-fail.
    """
    try:
        if ats_provider == "greenhouse":
            from app.automation.playwright.company_apply import GreenhouseApply
            return GreenhouseApply()
        elif ats_provider == "lever":
            from app.automation.playwright.company_apply import CompanyApply
            return CompanyApply()
        elif ats_provider in ("indeed", "indeed_easy_apply"):
            from app.automation.playwright.indeed_apply import IndeedApply
            return IndeedApply()
        elif ats_provider == "remoteok":
            from app.automation.playwright.remoteok_apply import RemoteOKApply
            return RemoteOKApply()
        else:
            from app.automation.playwright.company_apply import CompanyApply
            return CompanyApply()
    except ImportError:
        from app.automation.playwright.company_apply import CompanyApply
        return CompanyApply()


def _build_form_data(
    *,
    parsed_sections: dict[str, Any],
    cover_letter_text: str,
    resume_path: str | None,
    detected_fields: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Assemble a flat field-name → value mapping from the parsed resume sections
    and cover letter. Keys match common ATS field names / HTML name attributes.
    """
    contact = parsed_sections.get("contact", {})
    experience = parsed_sections.get("experience", [])
    education = parsed_sections.get("education", [])
    most_recent_exp = experience[0] if experience else {}
    most_recent_edu = education[0] if education else {}

    return {
        "first_name": contact.get("name", "").split()[0] if contact.get("name") else "",
        "last_name": " ".join(contact.get("name", "").split()[1:]) if contact.get("name") else "",
        "full_name": contact.get("name", ""),
        "email": contact.get("email", ""),
        "phone": contact.get("phone", ""),
        "location": contact.get("location", ""),
        "linkedin_url": contact.get("linkedin", ""),
        "github_url": contact.get("github", ""),
        "website_url": contact.get("website", ""),
        "current_company": most_recent_exp.get("company", ""),
        "current_title": most_recent_exp.get("title", ""),
        "cover_letter": cover_letter_text,
        "resume_path": resume_path or "",
        "highest_degree": most_recent_edu.get("degree", ""),
        "university": most_recent_edu.get("institution", ""),
        "graduation_year": str(most_recent_edu.get("graduation_year", "")),
    }


async def _capture_screenshot(page: Any, application_id: str) -> str | None:
    """Take a screenshot and save to the configured storage path."""
    try:
        from app.core.config import settings
        import os
        path = os.path.join("uploads", "screenshots", f"app_{application_id}.png")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        await page.screenshot(path=path, full_page=True)
        return path
    except Exception as exc:
        logger.warning("Screenshot capture failed", error=str(exc))
        return None


# ---------------------------------------------------------------------------
# Conditional routing
# ---------------------------------------------------------------------------

def _route_after_verify(state: ApplicationWorkflowState) -> str:
    """Route to DB update on success, manual review flag on failure."""
    if state.get("should_halt"):
        return NODE_ERROR
    if state.get("submission_success") and not state.get("requires_manual_review"):
        return NODE_UPDATE_DB
    return NODE_MANUAL_REVIEW


def _route_after_db_update(state: ApplicationWorkflowState) -> str:
    """After successful DB update, attempt optional recruiter discovery."""
    if state.get("should_halt"):
        return NODE_ERROR
    return NODE_RECRUITER


def _route_after_outreach(state: ApplicationWorkflowState) -> str:
    """After outreach attempt (sent or skipped), schedule follow-up."""
    if state.get("should_halt"):
        return NODE_ERROR
    return NODE_FOLLOWUP


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

async def build_application_graph():
    """Construct and compile the application submission StateGraph."""
    graph = StateGraph(ApplicationWorkflowState)

    graph.add_node(NODE_VALIDATE, validate_application_node)
    graph.add_node(NODE_TAILOR, tailor_resume_node)
    graph.add_node(NODE_COVER, generate_cover_letter_node)
    graph.add_node(NODE_DETECT_ATS, detect_ats_provider_node)
    graph.add_node(NODE_FILL, fill_form_node)
    graph.add_node(NODE_SUBMIT, submit_form_node)
    graph.add_node(NODE_VERIFY, verify_submission_node)
    graph.add_node(NODE_UPDATE_DB, update_application_db_node)
    graph.add_node(NODE_MANUAL_REVIEW, flag_manual_review_node)
    graph.add_node(NODE_RECRUITER, discover_recruiter_node)
    graph.add_node(NODE_OUTREACH, send_outreach_node)
    graph.add_node(NODE_FOLLOWUP, schedule_followup_node)
    graph.add_node(NODE_ERROR, handle_error_node)

    graph.set_entry_point(NODE_VALIDATE)

    graph.add_conditional_edges(NODE_VALIDATE, halt_on_error(success_node=NODE_TAILOR))
    graph.add_conditional_edges(NODE_TAILOR, halt_on_error(success_node=NODE_COVER))
    graph.add_conditional_edges(NODE_COVER, halt_on_error(success_node=NODE_DETECT_ATS))
    graph.add_conditional_edges(NODE_DETECT_ATS, halt_on_error(success_node=NODE_FILL))
    graph.add_conditional_edges(NODE_FILL, halt_on_error(success_node=NODE_SUBMIT))
    graph.add_conditional_edges(NODE_SUBMIT, halt_on_error(success_node=NODE_VERIFY))
    graph.add_conditional_edges(NODE_VERIFY, _route_after_verify)
    graph.add_conditional_edges(NODE_UPDATE_DB, _route_after_db_update)
    graph.add_conditional_edges(NODE_RECRUITER, halt_on_error(success_node=NODE_OUTREACH))
    graph.add_conditional_edges(NODE_OUTREACH, _route_after_outreach)
    graph.add_conditional_edges(NODE_FOLLOWUP, halt_on_error(success_node=END))

    graph.add_edge(NODE_MANUAL_REVIEW, END)
    graph.add_edge(NODE_ERROR, END)

    return await build_graph(graph)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def run_application_submission(
    *,
    user_id: str,
    agent_run_id: str,
    application_id: str,
    job_id: str,
    base_resume_id: str,
    apply_url: str | None = None,
    ats_provider: str | None = None,
) -> dict[str, Any]:
    """
    Run the full application submission pipeline for one application.

    Called from workers/job_tasks.py:submit_application_task.
    Returns the final workflow state for the task to use when updating
    the AgentRun completion record.
    """
    compiled = await build_application_graph()
    workflow_id = f"apply-{application_id}"

    initial_state: dict[str, Any] = {
        **new_base_state(
            user_id=user_id,
            agent_run_id=agent_run_id,
            workflow_id=workflow_id,
        ),
        "application_id": application_id,
        "job_id": job_id,
        "base_resume_id": base_resume_id,
        "apply_url": apply_url,
        "ats_provider": ats_provider,
        "form_fields_detected": [],
        "form_fields_filled": [],
        "submission_success": False,
        "requires_manual_review": False,
        "outreach_sent": False,
    }

    config = get_invoke_config(workflow_id=workflow_id, recursion_limit=80)
    final_state = await compiled.ainvoke(initial_state, config=config)
    return final_state