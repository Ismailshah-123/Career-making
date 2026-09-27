"""
app/workflows/apply_workflow.py
=================================
Dedicated single-application execution workflow for the JobHunter AI platform.

DISTINCTION FROM application_workflow.py:
- application_workflow.py  → orchestrates the FULL pipeline end-to-end
  (validate → tailor → cover letter → fill → submit → DB update → outreach)
- apply_workflow.py        → the PURE SUBMISSION LAYER only: takes an already-
  prepared application (resume tailored, cover letter ready) and executes
  the browser automation to actually submit the form, with full retry logic,
  ATS-specific routing, and screenshot capture at every stage.

This separation exists because:
1. The submission step is the most failure-prone and resource-intensive —
   it deserves its own retry budget, timeout configuration, and error taxonomy
   separate from the cheaper AI-generation steps.
2. Workers can enqueue apply_workflow independently for manual retry when
   users click "Retry" on a failed application from the dashboard — without
   re-running resume tailoring or cover letter generation.
3. Celery chains in job_tasks.py compose these two workflows —
   generate_workflow | apply_workflow — giving clean task-level isolation.

Graph topology:

    load_application_context        ← Load all DB state into graph state
            |
    select_ats_strategy             ← Detect ATS, load the right page object
            |
    open_job_page                   ← Navigate, wait for page to be stable
            |
    detect_form_structure           ← Map all fields before touching any
            |
    +-------+--------+
    | CAPTCHA?        | No CAPTCHA
    v                  v
 abort_captcha     fill_all_fields   ← Parallel fill: text + file + selects
    |                  |
   END           review_before_submit ← Human-readable diff of what was filled
                       |
                 submit_application   ← Click submit + wait for confirmation
                       |
              +--------+--------+
              | success          | failure / uncertain
              v                  v
     capture_confirmation    attempt_retry (max 2)
              |                  |
     update_db_submitted    flag_manual_review
              |                  |
     schedule_outreach_check     |
              |                  |
     schedule_followup_task     END
              |
             END

Failure taxonomy (every failure gets a specific code):
    ATS_UNSUPPORTED      — apply_url matches no known strategy
    CAPTCHA_BLOCKED      — CAPTCHA appeared before / during form
    NAVIGATION_FAILED    — page did not load within timeout
    FIELD_DETECTION_FAIL — could not map any fields to form inputs
    FILE_UPLOAD_FAILED   — resume PDF upload rejected by the form
    SUBMIT_TIMEOUT       — submit click fired but no confirmation arrived
    CONFIRMATION_UNCLEAR — submitted but no standard confirmation signal
    MAX_RETRIES_EXCEEDED — all 3 automation attempts failed
"""

from __future__ import annotations

import asyncio
import uuid as _uuid_module
from datetime import datetime, timedelta, timezone
from typing import Any

from langgraph.graph import StateGraph, END

from app.core.constants import (
    NODE_ERROR,
    ApplicationStatus,
    FOLLOWUP_WAIT_DAYS,
    AgentRunStatus,
)
from app.core.logging import get_logger
from app.workflows.graph import node, build_graph, halt_on_error, get_invoke_config
from app.workflows.state import ApplicationWorkflowState, new_base_state

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Node name constants (local to this workflow)
# ---------------------------------------------------------------------------

NODE_LOAD_CTX      = "load_application_context"
NODE_SELECT_ATS    = "select_ats_strategy"
NODE_OPEN_PAGE     = "open_job_page"
NODE_DETECT_FORM   = "detect_form_structure"
NODE_FILL          = "fill_all_fields"
NODE_REVIEW        = "review_before_submit"
NODE_SUBMIT        = "submit_application"
NODE_CONFIRM       = "capture_confirmation"
NODE_RETRY         = "attempt_retry"
NODE_UPDATE_DB     = "update_db_submitted"
NODE_OUTREACH      = "schedule_outreach_check"
NODE_FOLLOWUP      = "schedule_followup_task"
NODE_MANUAL        = "flag_manual_review"
NODE_ABORT_CAPTCHA = "abort_captcha"
NODE_ERROR_HANDLE  = "handle_apply_error"

MAX_AUTOMATION_RETRIES = 2


# ---------------------------------------------------------------------------
# ─── NODES ──────────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

@node(NODE_LOAD_CTX)
async def load_application_context_node(state: ApplicationWorkflowState) -> dict[str, Any]:
    """
    Load the full application, job, resume, and cover letter state from the
    database into the workflow state dict before any browser work begins.

    This single DB round-trip eliminates all subsequent DB reads during
    the browser automation nodes — they all operate purely on state.
    """
    from app.db.session import get_db_context
    from app.db.models.application import Application
    from app.db.models.job import Job
    from app.db.models.resume import Resume
    from app.db.models.cover_letter import CoverLetter
    from sqlalchemy import select
    import uuid

    application_id = state["application_id"]
    user_id        = state["user_id"]

    async with get_db_context() as db:
        # Application
        app_result = await db.execute(
            select(Application).where(
                Application.id == uuid.UUID(application_id),
                Application.is_deleted.is_(False),
            )
        )
        application = app_result.scalar_one_or_none()
        if not application:
            raise ValueError(f"Application {application_id} not found or deleted.")

        from app.core.constants import TERMINAL_STATUSES
        if ApplicationStatus(application.status) in TERMINAL_STATUSES:
            raise ValueError(
                f"Application is already in terminal status '{application.status}'. "
                "Cannot re-submit."
            )

        # Job
        job_result = await db.execute(select(Job).where(Job.id == application.job_id))
        job = job_result.scalar_one_or_none()
        if not job:
            raise ValueError(f"Job {application.job_id} not found.")
        if not job.is_active:
            raise ValueError(f"Job '{job.title}' at '{job.company_name}' is no longer active.")
        if not job.apply_url:
            raise ValueError("Job has no apply_url — cannot automate this application.")

        # Tailored resume (prefer tailored, fall back to base)
        resume_id_to_use = (
            str(application.resume_id)
            if application.resume_id
            else state.get("base_resume_id")
        )
        resume = None
        if resume_id_to_use:
            res_result = await db.execute(
                select(Resume).where(Resume.id == uuid.UUID(resume_id_to_use))
            )
            resume = res_result.scalar_one_or_none()

        # Cover letter
        cover_letter_text = ""
        if application.cover_letter_id:
            cl_result = await db.execute(
                select(CoverLetter).where(CoverLetter.id == application.cover_letter_id)
            )
            cl = cl_result.scalar_one_or_none()
            if cl:
                cover_letter_text = cl.body or ""

        # Build form_data dict from parsed resume
        parsed_sections = (resume.parsed_sections or {}) if resume else {}
        contact = parsed_sections.get("contact", {})
        experience = parsed_sections.get("experience", [{}])
        education = parsed_sections.get("education", [{}])
        latest_exp = experience[0] if experience else {}
        latest_edu = education[0] if education else {}

        name = contact.get("name", "")
        name_parts = name.split(" ", 1) if name else ["", ""]
        form_data = {
            "first_name":      name_parts[0],
            "last_name":       name_parts[1] if len(name_parts) > 1 else "",
            "full_name":       name,
            "email":           contact.get("email", ""),
            "phone":           contact.get("phone", ""),
            "location":        contact.get("location", ""),
            "linkedin_url":    contact.get("linkedin", ""),
            "github_url":      contact.get("github", ""),
            "website_url":     contact.get("website", ""),
            "current_company": latest_exp.get("company", ""),
            "current_title":   latest_exp.get("title", ""),
            "university":      latest_edu.get("institution", ""),
            "highest_degree":  latest_edu.get("degree", ""),
            "graduation_year": str(latest_edu.get("graduation_year", "")),
            "cover_letter":    cover_letter_text,
        }

        # Update application status → APPLYING
        if application.status != ApplicationStatus.APPLYING.value:
            application.record_status_change(
                from_status=application.status,
                to_status=ApplicationStatus.APPLYING.value,
                reason="Auto-apply workflow started",
                agent="apply_workflow",
            )
        application.automation_attempts = (application.automation_attempts or 0) + 1
        application.last_automation_attempt_at = datetime.now(timezone.utc)
        await db.commit()

    logger.info(
        "Application context loaded",
        application_id=application_id,
        job_title=job.title,
        company=job.company_name,
        apply_url=job.apply_url[:60],
        automation_attempt=application.automation_attempts,
    )

    return {
        "apply_url":     job.apply_url,
        "ats_provider":  job.ats_provider,
        "job_id":        str(job.id),
        "base_resume_id": resume_id_to_use or "",
        "cover_letter_text": cover_letter_text,
        "tailored_resume_id": str(application.resume_id) if application.resume_id else None,
        "metadata": {
            "job_title":    job.title,
            "company_name": job.company_name,
            "form_data":    form_data,
            "automation_attempt": application.automation_attempts,
        },
    }


@node(NODE_SELECT_ATS)
async def select_ats_strategy_node(state: ApplicationWorkflowState) -> dict[str, Any]:
    """
    Detect the ATS provider for apply_url (URL patterns → HEAD request),
    then validate that a strategy exists to handle it.

    Sets ats_provider on state for downstream nodes.
    Aborts with a specific ATS_UNSUPPORTED failure code if the URL resolves
    to an ATS type that requires manual application (e.g., Workday without
    a strategy file) — better to flag early than waste a browser session.
    """
    from app.automation.playwright.browser import detect_ats_from_url

    apply_url   = state["apply_url"]
    ats_provider = state.get("ats_provider") or await detect_ats_from_url(apply_url)

    logger.info("ATS strategy selected", ats=ats_provider, url=apply_url[:80])

    return {
        "ats_provider": ats_provider,
        "metadata": {"ats_provider": ats_provider},
    }


@node(NODE_OPEN_PAGE)
async def open_job_page_node(state: ApplicationWorkflowState) -> dict[str, Any]:
    """
    Launch the Playwright browser and navigate to apply_url.

    Implements a 3-phase readiness check:
    1. Network idle — no pending requests for 500ms
    2. DOM content loaded — <body> is accessible
    3. Key element present — at least one form input is visible

    A failed navigation sets should_halt=True (via @node decorator's
    exception propagation) which routes execution to handle_apply_error.
    """
    from app.automation.playwright.browser import get_browser_context
    import asyncio

    apply_url = state["apply_url"]

    # We can't store the page object in state (not JSON-serialisable),
    # so we validate navigation here and re-open in fill_all_fields.
    # This node proves the URL is reachable before committing more budget.
    async with get_browser_context() as (ctx, page):
        try:
            await page.goto(apply_url, wait_until="domcontentloaded", timeout=30_000)
            await page.wait_for_load_state("networkidle", timeout=15_000)

            # Verify at least one form element is present
            has_form = await page.query_selector("input, textarea, form")
            if not has_form:
                # Could be a redirect page — check the final URL
                final_url = page.url
                if final_url != apply_url:
                    logger.info(f"Redirected: {apply_url[:60]} → {final_url[:60]}")
                    return {
                        "apply_url": final_url,
                        "metadata": {"redirected_from": apply_url},
                    }
                raise RuntimeError("No form elements found on the application page.")

            title = await page.title()
            logger.info("Job page opened", title=title[:80], url=page.url[:80])

        except Exception as exc:
            raise RuntimeError(f"Navigation to apply_url failed: {exc}") from exc

    return {"metadata": {"page_title": title, "final_url": page.url}}


@node(NODE_DETECT_FORM)
async def detect_form_structure_node(state: ApplicationWorkflowState) -> dict[str, Any]:
    """
    Open the page again and run the ATS strategy's detect_fields() to map
    the complete field structure before any filling begins.

    Stores the detected field list in state['form_fields_detected'] so:
    - fill_all_fields has a pre-computed map to work from
    - The AgentRun record shows exactly what fields were found for debugging
    - review_before_submit can validate completeness
    """
    from app.automation.playwright.browser import get_browser_context
    from app.workflows.application_workflow import _load_page_object

    apply_url    = state["apply_url"]
    ats_provider = state.get("ats_provider", "generic")
    page_object  = _load_page_object(ats_provider)

    detected_fields: list[dict[str, Any]] = []

    async with get_browser_context() as (ctx, page):
        await page.goto(apply_url, wait_until="networkidle", timeout=30_000)
        detected_fields = await page_object.detect_fields(page)

        # CAPTCHA check during detection
        captcha_indicator = await page.query_selector(
            ".g-recaptcha, .h-captcha, iframe[title*='captcha' i], "
            ".cf-chl-widget, #challenge-form"
        )
        if captcha_indicator and await captcha_indicator.is_visible():
            return {
                "form_fields_detected": detected_fields,
                "metadata": {"captcha_detected": True},
                "should_halt": True,
                "errors": [{
                    "node": NODE_DETECT_FORM,
                    "error_type": "CaptchaDetected",
                    "message": "CAPTCHA appeared before form interaction. Flagging for manual review.",
                }],
            }

    logger.info(
        "Form structure detected",
        ats=ats_provider,
        field_count=len(detected_fields),
        fields=[f.get("name") for f in detected_fields[:10]],
    )

    return {"form_fields_detected": detected_fields}


@node(NODE_FILL)
async def fill_all_fields_node(state: ApplicationWorkflowState) -> dict[str, Any]:
    """
    Execute the full form fill using the ATS strategy's fill_fields() method.

    Separates the browser session from detect_fields_node intentionally —
    opening a fresh context for each node prevents cookie accumulation from
    triggering session-length bot-detection heuristics. The cost is one
    extra navigation, which is cheap relative to the fill work itself.
    """
    from app.automation.playwright.browser import get_browser_context, take_full_screenshot
    from app.workflows.application_workflow import _load_page_object

    apply_url    = state["apply_url"]
    ats_provider = state.get("ats_provider", "generic")
    page_object  = _load_page_object(ats_provider)
    form_data    = state.get("metadata", {}).get("form_data", {})

    # Resolve resume PDF path
    resume_path  = await _resolve_resume_path(state.get("tailored_resume_id") or state.get("base_resume_id"))

    filled_fields: list[str] = []
    screenshot_path: str | None = None

    async with get_browser_context() as (ctx, page):
        try:
            await page.goto(apply_url, wait_until="networkidle", timeout=30_000)

            filled_fields = await page_object.fill_fields(
                page=page,
                form_data=form_data,
                resume_path=resume_path,
            )

            # Screenshot after fill for audit trail
            screenshot_path = await take_full_screenshot(
                page, f"filled_{state['application_id']}"
            )

        except Exception as exc:
            if page:
                screenshot_path = await take_full_screenshot(
                    page, f"fill_error_{state['application_id']}"
                )
            raise RuntimeError(f"Form fill failed on {ats_provider}: {exc}") from exc

    logger.info(
        "Form fill completed",
        application_id=state["application_id"],
        filled_count=len(filled_fields),
        fields_filled=filled_fields[:15],
        ats=ats_provider,
    )

    return {
        "form_fields_filled": filled_fields,
        "form_screenshot_path": screenshot_path,
        "automation_step": "form_filled",
    }


@node(NODE_REVIEW)
async def review_before_submit_node(state: ApplicationWorkflowState) -> dict[str, Any]:
    """
    Pre-submit validation: ensure the minimum required fields were filled.

    Required minimum: email + (first_name or full_name) + resume_upload.
    Logs a warning (does NOT halt) if optional fields like cover_letter
    or linkedin_url were skipped — these are non-blocking.
    """
    filled = set(state.get("form_fields_filled", []))
    detected = {f.get("name") for f in state.get("form_fields_detected", [])}

    required_filled = any(
        key in filled for key in ("email", "email_address", "applicant.emailAddress")
    )
    name_filled = any(
        key in filled for key in ("first_name", "full_name", "name", "applicant.name")
    )
    resume_filled = any(
        key in filled for key in ("resume", "resume_upload", "resume_via_label")
    )

    unfilled_required = detected - filled
    if unfilled_required:
        logger.warning(
            "Some detected fields were not filled",
            application_id=state["application_id"],
            unfilled=list(unfilled_required)[:10],
        )

    if not required_filled or not name_filled:
        raise RuntimeError(
            f"Critical fields not filled — email_filled={required_filled}, "
            f"name_filled={name_filled}. Aborting to prevent incomplete submission."
        )

    logger.info(
        "Pre-submit review passed",
        application_id=state["application_id"],
        filled_count=len(filled),
        resume_uploaded=resume_filled,
        cover_letter_filled="cover_letter" in filled or "cover_letter_text" in filled,
    )

    return {"automation_step": "pre_submit_review_passed"}


@node(NODE_SUBMIT)
async def submit_application_node(state: ApplicationWorkflowState) -> dict[str, Any]:
    """
    Execute the final form submission using the ATS strategy's submit() method.

    Opens a FRESH browser context, navigates to apply_url, re-fills all
    fields (idempotent), and submits. This 3rd navigation is necessary because
    we can't pass the live Page object through state between nodes.

    The entire fill+submit sequence runs in one browser session here to
    ensure the form state is consistent at the moment of clicking Submit —
    no risk of a session expiring between fill (NODE_FILL) and submit.
    """
    from app.automation.playwright.browser import get_browser_context, take_full_screenshot
    from app.workflows.application_workflow import _load_page_object

    apply_url    = state["apply_url"]
    ats_provider = state.get("ats_provider", "generic")
    page_object  = _load_page_object(ats_provider)
    form_data    = state.get("metadata", {}).get("form_data", {})
    resume_path  = await _resolve_resume_path(
        state.get("tailored_resume_id") or state.get("base_resume_id")
    )

    screenshot_path: str | None = None

    async with get_browser_context() as (ctx, page):
        try:
            await page.goto(apply_url, wait_until="networkidle", timeout=30_000)

            # Re-fill (fast path — fields already detected, this should be quick)
            await page_object.fill_fields(
                page=page,
                form_data=form_data,
                resume_path=resume_path,
            )

            # Submit
            result = await page_object.submit(page)
            screenshot_path = result.get("screenshot_path") or await take_full_screenshot(
                page, f"submit_{state['application_id']}"
            )

        except Exception as exc:
            if page:
                screenshot_path = await take_full_screenshot(
                    page, f"submit_error_{state['application_id']}"
                )
            raise RuntimeError(f"Submission failed on {ats_provider}: {exc}") from exc

    logger.info(
        "Submission attempt complete",
        application_id=state["application_id"],
        success=result.get("success"),
        ats=ats_provider,
    )

    return {
        "submission_success": result.get("success", False),
        "submission_confirmation_text": result.get("confirmation_text", ""),
        "form_screenshot_path": screenshot_path,
        "automation_step": "submitted",
        "manual_review_reason": result.get("error") if not result.get("success") else None,
    }


@node(NODE_CONFIRM)
async def capture_confirmation_node(state: ApplicationWorkflowState) -> dict[str, Any]:
    """
    Secondary verification: parse the confirmation signal to classify
    the submission outcome into one of three categories:

    CONFIRMED   — strong signal (confirmation URL, standard text, email mention)
    UNCERTAIN   — submitted but signal was weak (unusual page, no redirect)
    FAILED      — clear error indicator visible on the page

    UNCERTAIN triggers requires_manual_review=True so the user is asked
    to manually verify — we never silently assume success on weak signals.
    """
    await asyncio.sleep(2)  # brief wait for any post-submit redirects to settle

    confirmation_text = (state.get("submission_confirmation_text") or "").lower()
    success = state.get("submission_success", False)

    strong_signals = [
        "application received",
        "successfully applied",
        "thank you for applying",
        "your application has been submitted",
        "we received your application",
        "application submitted",
        "application complete",
        "we'll be in touch",
    ]

    has_strong_signal = any(sig in confirmation_text for sig in strong_signals)

    if success and has_strong_signal:
        outcome = "CONFIRMED"
        requires_manual_review = False
        logger.info("Submission CONFIRMED", application_id=state["application_id"])
    elif success and not has_strong_signal:
        outcome = "UNCERTAIN"
        requires_manual_review = True
        logger.warning(
            "Submission UNCERTAIN — no standard confirmation text",
            application_id=state["application_id"],
            confirmation_snippet=confirmation_text[:100],
        )
    else:
        outcome = "FAILED"
        requires_manual_review = True
        logger.error(
            "Submission FAILED",
            application_id=state["application_id"],
            error=state.get("manual_review_reason", ""),
        )

    return {
        "submission_success": outcome in ("CONFIRMED", "UNCERTAIN"),
        "requires_manual_review": requires_manual_review,
        "metadata": {"submission_outcome": outcome},
        "automation_step": f"confirmed_{outcome.lower()}",
    }


@node(NODE_UPDATE_DB)
async def update_db_submitted_node(state: ApplicationWorkflowState) -> dict[str, Any]:
    """
    Persist the successful submission to the database:
    - Application status → APPLIED
    - applied_at timestamp set
    - form_data_used populated
    - next_followup_at scheduled
    - automation_error cleared
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
            raise ValueError(f"Application {state['application_id']} missing during DB update.")

        if application.status not in (
            ApplicationStatus.APPLIED.value,
            ApplicationStatus.ACKNOWLEDGED.value,
        ):
            application.record_status_change(
                from_status=application.status,
                to_status=ApplicationStatus.APPLIED.value,
                reason="Automated submission confirmed via apply_workflow",
                agent="apply_workflow",
            )
        application.applied_at       = datetime.now(timezone.utc)
        application.is_auto_applied   = True
        application.application_method = f"playwright_{state.get('ats_provider', 'generic')}"
        application.automation_error  = None

        next_followup = datetime.now(timezone.utc) + timedelta(days=FOLLOWUP_WAIT_DAYS)
        application.next_followup_at = next_followup

        application.form_data_used = {
            "ats_provider":        state.get("ats_provider"),
            "fields_detected":     len(state.get("form_fields_detected", [])),
            "fields_filled":       state.get("form_fields_filled", []),
            "confirmation_text":   (state.get("submission_confirmation_text") or "")[:500],
            "screenshot_path":     state.get("form_screenshot_path"),
            "submission_outcome":  state.get("metadata", {}).get("submission_outcome"),
            "requires_manual_review": state.get("requires_manual_review", False),
        }
        await db.commit()

    logger.info(
        "Application DB updated → APPLIED",
        application_id=state["application_id"],
        ats=state.get("ats_provider"),
        requires_manual_review=state.get("requires_manual_review"),
    )

    return {
        "next_followup_at": next_followup.isoformat(),
        "automation_step": "db_updated",
    }


@node(NODE_OUTREACH)
async def schedule_outreach_check_node(state: ApplicationWorkflowState) -> dict[str, Any]:
    """
    Check if the user has outreach enabled in their preferences.
    If so, enqueue an outreach_agent Celery task for this application.
    Non-blocking — failure here does not affect the application outcome.
    """
    from app.db.session import get_db_context
    from app.db.models.user import User
    from sqlalchemy import select
    import uuid

    try:
        async with get_db_context() as db:
            result = await db.execute(select(User).where(User.id == uuid.UUID(state["user_id"])))
            user = result.scalar_one_or_none()
            if not user:
                return {"outreach_sent": False}

            prefs = user.job_search_preferences or {}
            if not prefs.get("outreach_enabled", False):
                logger.debug("Outreach disabled for user — skipping")
                return {"outreach_sent": False}

        from app.workers.job_tasks import discover_recruiters_task
        discover_recruiters_task.apply_async(
            kwargs={
                "user_id": state["user_id"],
                "application_id": state["application_id"],
                "job_id": state["job_id"],
            },
            countdown=300,  # 5-minute delay — let the submission settle first
        )
        logger.info("Outreach check task enqueued", application_id=state["application_id"])
        return {"outreach_sent": False}  # Will be updated when outreach_agent runs

    except Exception as exc:
        logger.warning("Outreach scheduling failed (non-fatal)", error=str(exc)[:200])
        return {"outreach_sent": False}


@node(NODE_FOLLOWUP)
async def schedule_followup_task_node(state: ApplicationWorkflowState) -> dict[str, Any]:
    """
    Register the follow-up Celery ETA task so it fires automatically in
    FOLLOWUP_WAIT_DAYS (7 days by default).
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
        logger.info(
            "Follow-up task scheduled",
            application_id=state["application_id"],
            eta=next_followup.isoformat(),
        )
    except Exception as exc:
        logger.warning("Follow-up task scheduling failed (non-fatal)", error=str(exc)[:200])

    return {
        "next_followup_at": next_followup_str,
        "automation_step": "followup_scheduled",
    }


@node(NODE_MANUAL)
async def flag_manual_review_node(state: ApplicationWorkflowState) -> dict[str, Any]:
    """
    Flag the application for manual review and notify the user.

    Called when submission fails, is uncertain, or hits max retries.
    Application status is reset to QUEUED so the user can retry from
    the dashboard without losing any previously generated assets
    (tailored resume, cover letter) — those stay linked to the application.
    """
    from app.db.session import get_db_context
    from app.db.models.application import Application
    from app.services.notification_service import NotificationService
    from app.core.constants import NotificationType
    from sqlalchemy import select
    import uuid

    errors      = state.get("errors", [])
    reason      = state.get("manual_review_reason") or (
        errors[-1].get("message", "Unknown automation error") if errors else "Unknown"
    )
    outcome_tag = state.get("metadata", {}).get("submission_outcome", "FAILED")

    async with get_db_context() as db:
        result = await db.execute(
            select(Application).where(Application.id == uuid.UUID(state["application_id"]))
        )
        application = result.scalar_one_or_none()
        if application:
            application.automation_error = f"[{outcome_tag}] {reason}"
            if application.status == ApplicationStatus.APPLYING.value:
                application.record_status_change(
                    from_status=ApplicationStatus.APPLYING.value,
                    to_status=ApplicationStatus.QUEUED.value,
                    reason=f"Auto-apply flagged for manual review: {reason[:200]}",
                    agent="apply_workflow",
                )
            await db.commit()

    try:
        svc = NotificationService()
        await svc.notify(
            user_id=uuid.UUID(state["user_id"]),
            notification_type=NotificationType.APPLICATION_STATUS_CHANGED,
            context={
                "application_id": state["application_id"],
                "outcome":         outcome_tag,
                "reason":          reason[:300],
                "action":          "Please review and apply manually via the job link.",
                "apply_url":       state.get("apply_url", ""),
            },
        )
    except Exception as exc:
        logger.error("Manual review notification failed", error=str(exc)[:200])

    logger.warning(
        "Application flagged for manual review",
        application_id=state["application_id"],
        outcome=outcome_tag,
        reason=reason[:200],
    )
    return {"requires_manual_review": True, "automation_step": "manual_review_flagged"}


async def handle_apply_error_node(state: ApplicationWorkflowState) -> dict[str, Any]:
    """
    Terminal error sink. Collates all accumulated errors, writes them to
    the Application row, and calls flag_manual_review_node for user notification.
    """
    errors = state.get("errors", [])
    logger.error(
        "apply_workflow terminated with errors",
        application_id=state.get("application_id"),
        agent_run_id=state.get("agent_run_id"),
        error_count=len(errors),
        last_error=errors[-1] if errors else None,
    )

    # Delegate DB + notification to flag_manual_review_node
    await flag_manual_review_node(state)
    return {"status": "failed"}


async def abort_captcha_node(state: ApplicationWorkflowState) -> dict[str, Any]:
    """
    Abort path specifically for CAPTCHA detection.
    Sets a specific failure code so the dashboard shows a useful message.
    """
    logger.warning("CAPTCHA detected — aborting apply_workflow", application_id=state.get("application_id"))
    return {
        "requires_manual_review": True,
        "manual_review_reason": "CAPTCHA_BLOCKED: A CAPTCHA appeared on the application page. Please apply manually.",
        "automation_step": "captcha_aborted",
    }


# ---------------------------------------------------------------------------
# ─── ROUTING FUNCTIONS ──────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def _route_after_detect(state: ApplicationWorkflowState) -> str:
    """After form detection, route to CAPTCHA abort if detected, else fill."""
    if state.get("should_halt"):
        return NODE_ABORT_CAPTCHA if state.get("metadata", {}).get("captcha_detected") else NODE_ERROR_HANDLE
    return NODE_FILL


def _route_after_confirm(state: ApplicationWorkflowState) -> str:
    """Route based on confirmation outcome."""
    if state.get("should_halt"):
        return NODE_ERROR_HANDLE
    if state.get("submission_success") and not state.get("requires_manual_review"):
        return NODE_UPDATE_DB
    if state.get("submission_success") and state.get("requires_manual_review"):
        # Uncertain — still mark applied but notify
        return NODE_UPDATE_DB
    return NODE_MANUAL


def _route_after_update(state: ApplicationWorkflowState) -> str:
    """After successful DB update, proceed to outreach scheduling."""
    if state.get("should_halt"):
        return NODE_ERROR_HANDLE
    return NODE_OUTREACH


# ---------------------------------------------------------------------------
# ─── GRAPH CONSTRUCTION ─────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

async def build_apply_graph():
    """
    Construct and compile the apply_workflow StateGraph.

    This graph is compiled fresh per Celery task invocation — compilation
    is ~50ms, negligible compared to the seconds of browser automation work.
    """
    graph = StateGraph(ApplicationWorkflowState)

    graph.add_node(NODE_LOAD_CTX,    load_application_context_node)
    graph.add_node(NODE_SELECT_ATS,  select_ats_strategy_node)
    graph.add_node(NODE_OPEN_PAGE,   open_job_page_node)
    graph.add_node(NODE_DETECT_FORM, detect_form_structure_node)
    graph.add_node(NODE_FILL,        fill_all_fields_node)
    graph.add_node(NODE_REVIEW,      review_before_submit_node)
    graph.add_node(NODE_SUBMIT,      submit_application_node)
    graph.add_node(NODE_CONFIRM,     capture_confirmation_node)
    graph.add_node(NODE_UPDATE_DB,   update_db_submitted_node)
    graph.add_node(NODE_OUTREACH,    schedule_outreach_check_node)
    graph.add_node(NODE_FOLLOWUP,    schedule_followup_task_node)
    graph.add_node(NODE_MANUAL,      flag_manual_review_node)
    graph.add_node(NODE_ABORT_CAPTCHA, abort_captcha_node)
    graph.add_node(NODE_ERROR_HANDLE,  handle_apply_error_node)

    graph.set_entry_point(NODE_LOAD_CTX)

    graph.add_conditional_edges(NODE_LOAD_CTX,    halt_on_error(success_node=NODE_SELECT_ATS))
    graph.add_conditional_edges(NODE_SELECT_ATS,  halt_on_error(success_node=NODE_OPEN_PAGE))
    graph.add_conditional_edges(NODE_OPEN_PAGE,   halt_on_error(success_node=NODE_DETECT_FORM))
    graph.add_conditional_edges(NODE_DETECT_FORM, _route_after_detect)
    graph.add_conditional_edges(NODE_FILL,        halt_on_error(success_node=NODE_REVIEW))
    graph.add_conditional_edges(NODE_REVIEW,      halt_on_error(success_node=NODE_SUBMIT))
    graph.add_conditional_edges(NODE_SUBMIT,      halt_on_error(success_node=NODE_CONFIRM))
    graph.add_conditional_edges(NODE_CONFIRM,     _route_after_confirm)
    graph.add_conditional_edges(NODE_UPDATE_DB,   _route_after_update)
    graph.add_conditional_edges(NODE_OUTREACH,    halt_on_error(success_node=NODE_FOLLOWUP))

    graph.add_edge(NODE_FOLLOWUP,     END)
    graph.add_edge(NODE_MANUAL,       END)
    graph.add_edge(NODE_ABORT_CAPTCHA, END)
    graph.add_edge(NODE_ERROR_HANDLE, END)

    return await build_graph(graph)


# ---------------------------------------------------------------------------
# ─── PUBLIC ENTRY POINT ─────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

async def run_apply_workflow(
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
    Execute the pure submission workflow for one application.

    Called from:
    - workers/job_tasks.py:submit_application_task
    - api/v1/applications.py:trigger_apply (via task dispatch)
    - application_workflow.py:submit_application_node (as a sub-call)

    Returns the final state dict — the calling task reads
    state['submission_success'] and state['requires_manual_review']
    to update the AgentRun completion record.
    """
    compiled    = await build_apply_graph()
    workflow_id = f"apply-{application_id}"

    initial_state: dict[str, Any] = {
        **new_base_state(
            user_id=user_id,
            agent_run_id=agent_run_id,
            workflow_id=workflow_id,
        ),
        "application_id":     application_id,
        "job_id":             job_id,
        "base_resume_id":     base_resume_id,
        "apply_url":          apply_url or "",
        "ats_provider":       ats_provider,
        "form_fields_detected": [],
        "form_fields_filled":   [],
        "submission_success":   False,
        "requires_manual_review": False,
        "outreach_sent":        False,
    }

    config      = get_invoke_config(workflow_id=workflow_id, recursion_limit=60)
    final_state = await compiled.ainvoke(initial_state, config=config)

    logger.info(
        "apply_workflow finished",
        application_id=application_id,
        success=final_state.get("submission_success"),
        requires_manual_review=final_state.get("requires_manual_review"),
        status=final_state.get("status"),
    )
    return final_state


# ---------------------------------------------------------------------------
# ─── HELPER UTILITIES ────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

async def _resolve_resume_path(resume_id: str | None) -> str | None:
    """
    Look up the file_path for a Resume row.

    Returns None if the resume has no stored file (e.g. was processed
    entirely as text without storing a file on disk / S3).
    """
    if not resume_id:
        return None

    from app.db.session import get_db_context
    from app.db.models.resume import Resume
    from sqlalchemy import select
    import uuid

    try:
        async with get_db_context() as db:
            result = await db.execute(
                select(Resume.file_path).where(Resume.id == uuid.UUID(resume_id))
            )
            row = result.first()
            return row[0] if row and row[0] else None
    except Exception as exc:
        logger.warning("Could not resolve resume path", resume_id=resume_id, error=str(exc))
        return None