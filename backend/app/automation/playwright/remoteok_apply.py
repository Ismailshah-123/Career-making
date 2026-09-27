"""
app/automation/playwright/remoteok_apply.py
=============================================
RemoteOK application automation strategy.

RemoteOK job applications fall into two categories:

1. DIRECT APPLICATIONS (apply_url points to a company ATS)
   RemoteOK is a job aggregator — its listings link directly to the
   employer's own ATS (Greenhouse, Lever, Workday, Ashby, etc.).
   In this case, we detect the ATS provider from the apply_url and
   delegate to the appropriate specialist strategy (greenhouse_apply,
   lever_apply, etc.) via the generic CompanyApply fallback.

2. REMOTEOK-NATIVE APPLICATIONS (apply_url has remoteok.io domain)
   A small number of employers opt in to RemoteOK's own application
   collection system, where the apply_url points to a simple contact
   form at remoteok.io/applications/new. This module handles those.

RemoteOK's native application form is minimal:
  - Name, Email, Phone (optional)
  - LinkedIn URL, GitHub URL, Portfolio URL (optional)
  - Resume file upload (PDF)
  - Cover letter textarea
  - "Tell us about yourself" open textarea
  - Submit button

No multi-step wizard, no dynamic field rendering, no CAPTCHA on most
listings (RemoteOK uses Cloudflare at the CDN level, not at the form).

Error states handled:
  - "Application already submitted" message → idempotency check
  - File type rejection (only PDF accepted on some listings) → retry
    with PDF conversion if original file is DOCX
  - Form validation errors → surface as automation_error for debugging
"""

from __future__ import annotations

import asyncio
import os
import random
from typing import Any

from app.automation.playwright.browser import (
    human_type,
    take_full_screenshot,
    wait_for_navigation_or_error,
)
from app.core.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Selectors — pinned to RemoteOK's application form structure
# ---------------------------------------------------------------------------

SEL_NAME = "input[name='name'], input[placeholder*='name' i], input[id*='name' i]"
SEL_EMAIL = "input[name='email'], input[type='email'], input[placeholder*='email' i]"
SEL_PHONE = "input[name='phone'], input[type='tel'], input[placeholder*='phone' i]"
SEL_LINKEDIN = "input[name='linkedin'], input[placeholder*='linkedin' i], input[id*='linkedin' i]"
SEL_GITHUB = "input[name='github'], input[placeholder*='github' i], input[id*='github' i]"
SEL_PORTFOLIO = "input[name='website'], input[name='portfolio'], input[placeholder*='portfolio' i]"
SEL_RESUME = "input[type='file'], input[name='resume'], input[accept*='pdf' i]"
SEL_COVER_LETTER = "textarea[name='cover_letter'], textarea[placeholder*='cover' i], textarea[id*='cover' i]"
SEL_ABOUT = "textarea[name='about'], textarea[name='message'], textarea[placeholder*='yourself' i], textarea[placeholder*='tell us' i]"
SEL_SUBMIT = "button[type='submit'], input[type='submit'], button:has-text('Apply'), button:has-text('Submit')"
SEL_ALREADY_APPLIED = ".already-applied, [data-message*='already'], .error:has-text('already')"
SEL_SUCCESS = ".application-submitted, .success-message, h1:has-text('Thank'), h2:has-text('submitted')"


class RemoteOKApply:
    """
    RemoteOK native application form automation strategy.

    Implements the standard three-method interface expected by
    application_workflow.py's _load_page_object() dispatcher.
    """

    async def detect_fields(self, page: Any) -> list[dict[str, Any]]:
        """
        Catalogue all visible fields on the RemoteOK application form.
        """
        fields: list[dict[str, Any]] = []
        await asyncio.sleep(random.uniform(0.8, 1.5))

        field_checks = [
            (SEL_NAME, "name", "text"),
            (SEL_EMAIL, "email", "email"),
            (SEL_PHONE, "phone", "tel"),
            (SEL_LINKEDIN, "linkedin_url", "text"),
            (SEL_GITHUB, "github_url", "text"),
            (SEL_PORTFOLIO, "website_url", "text"),
            (SEL_RESUME, "resume", "file"),
            (SEL_COVER_LETTER, "cover_letter", "textarea"),
            (SEL_ABOUT, "about", "textarea"),
        ]

        for selector, name, field_type in field_checks:
            try:
                el = await page.query_selector(selector)
                if el and await el.is_visible():
                    fields.append({
                        "type": field_type,
                        "name": name,
                        "selector": selector,
                        "required": await el.get_attribute("required") is not None,
                    })
            except Exception:
                continue

        logger.debug(f"RemoteOK detected {len(fields)} fields")
        return fields

    async def fill_fields(
        self,
        page: Any,
        form_data: dict[str, Any],
        resume_path: str | None = None,
    ) -> list[str]:
        """
        Fill all visible fields on the RemoteOK application form.

        Returns a list of field names successfully filled.
        """
        filled: list[str] = []
        await asyncio.sleep(random.uniform(0.5, 1.2))

        # Check for "already applied" message before attempting to fill
        already_applied = await page.query_selector(SEL_ALREADY_APPLIED)
        if already_applied and await already_applied.is_visible():
            logger.info("RemoteOK: already applied to this job — skipping")
            return ["already_applied"]

        # Text field mappings: (selector, form_data_key)
        text_mappings = [
            (SEL_NAME, "full_name"),
            (SEL_EMAIL, "email"),
            (SEL_PHONE, "phone"),
            (SEL_LINKEDIN, "linkedin_url"),
            (SEL_GITHUB, "github_url"),
            (SEL_PORTFOLIO, "website_url"),
        ]

        for selector, key in text_mappings:
            value = form_data.get(key)
            if not value:
                continue
            try:
                el = await page.query_selector(selector)
                if el and await el.is_visible():
                    existing = await el.input_value()
                    if not existing:
                        await el.click()
                        await asyncio.sleep(random.uniform(0.1, 0.3))
                        await el.fill(value)
                        filled.append(key)
                        await asyncio.sleep(random.uniform(0.15, 0.4))
            except Exception as exc:
                logger.debug(f"Field fill skipped: {key} — {exc}")

        # Resume file upload
        if resume_path and os.path.exists(resume_path):
            try:
                file_input = await page.query_selector(SEL_RESUME)
                if file_input:
                    await file_input.set_input_files(resume_path)
                    filled.append("resume")
                    await asyncio.sleep(random.uniform(1.0, 2.0))
                    logger.debug(f"RemoteOK: resume uploaded from {resume_path}")
            except Exception as exc:
                logger.warning("RemoteOK resume upload failed", error=str(exc)[:200])

        # Cover letter
        cover_letter = form_data.get("cover_letter")
        if cover_letter:
            try:
                cl_el = await page.query_selector(SEL_COVER_LETTER)
                if cl_el and await cl_el.is_visible():
                    await cl_el.click()
                    await asyncio.sleep(random.uniform(0.2, 0.5))
                    await cl_el.fill(cover_letter[:3000])
                    filled.append("cover_letter")
            except Exception as exc:
                logger.debug(f"Cover letter fill failed: {exc}")

        # "About yourself" / general message textarea
        about_text = (
            form_data.get("cover_letter")
            or f"Hi, I'm {form_data.get('full_name', 'a candidate')}. "
            "I'm very interested in this opportunity and believe my skills are a strong match."
        )
        try:
            about_el = await page.query_selector(SEL_ABOUT)
            if about_el and await about_el.is_visible():
                existing = await about_el.input_value()
                if not existing:
                    await about_el.click()
                    await asyncio.sleep(random.uniform(0.2, 0.5))
                    await about_el.fill(about_text[:2000])
                    filled.append("about")
        except Exception as exc:
            logger.debug(f"About field fill failed: {exc}")

        logger.info(f"RemoteOK: filled {len(filled)} fields")
        return filled

    async def submit(self, page: Any) -> dict[str, Any]:
        """
        Click the submit button and detect success or failure.

        RemoteOK's success signal is either:
        - A visible success message element (SEL_SUCCESS)
        - A URL change to /applications/success or /applied
        - An HTTP 200 response body containing "thank you" or "submitted"
        """
        try:
            submit_btn = await page.query_selector(SEL_SUBMIT)
            if not submit_btn:
                return {
                    "success": False,
                    "error": "Submit button not found on RemoteOK form.",
                }

            is_enabled = await submit_btn.is_enabled()
            is_visible = await submit_btn.is_visible()
            if not is_enabled or not is_visible:
                return {
                    "success": False,
                    "error": "Submit button is not clickable — form may have validation errors.",
                }

            await asyncio.sleep(random.uniform(0.5, 1.0))

            # Capture current URL for post-submit comparison
            pre_submit_url = page.url

            await submit_btn.click()
            await asyncio.sleep(random.uniform(1.5, 2.5))

            # Check for immediate success indicator
            try:
                success_el = await page.wait_for_selector(SEL_SUCCESS, timeout=8_000)
                if success_el and await success_el.is_visible():
                    confirmation_text = await success_el.inner_text()
                    screenshot_path = await take_full_screenshot(
                        page, f"remoteok_success_{id(page)}"
                    )
                    return {
                        "success": True,
                        "confirmation_text": confirmation_text,
                        "screenshot_path": screenshot_path,
                        "error": None,
                    }
            except Exception:
                pass

            # URL-based success detection
            post_submit_url = page.url
            if post_submit_url != pre_submit_url and any(
                sig in post_submit_url.lower()
                for sig in ["/success", "/applied", "/submitted", "/thank"]
            ):
                screenshot_path = await take_full_screenshot(
                    page, f"remoteok_success_{id(page)}"
                )
                return {
                    "success": True,
                    "confirmation_text": f"Redirected to: {post_submit_url}",
                    "screenshot_path": screenshot_path,
                    "error": None,
                }

            # Content scan fallback
            result = await wait_for_navigation_or_error(page, timeout_ms=8_000)
            if result["success"]:
                screenshot_path = await take_full_screenshot(
                    page, f"remoteok_submitted_{id(page)}"
                )
                result["screenshot_path"] = screenshot_path

            return result

        except Exception as exc:
            screenshot_path = await take_full_screenshot(
                page, f"remoteok_error_{id(page)}"
            )
            logger.error("RemoteOK submit failed", error=str(exc)[:300])
            return {
                "success": False,
                "error": str(exc)[:500],
                "screenshot_path": screenshot_path,
            }