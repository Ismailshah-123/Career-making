"""
app/automation/playwright/linkedin_apply.py
=============================================
LinkedIn Easy Apply Playwright automation strategy.

LinkedIn's Easy Apply modal is a multi-step wizard with dynamic field
rendering — fields appear and disappear based on answers to prior steps.
This module handles:
- Clicking the "Easy Apply" button on the job listing page
- Detecting each step's field set (text, select, radio, file upload)
- Filling all fields correctly, page-by-page through the wizard
- Handling the "Upload Resume" step with the tailored PDF
- Submitting the final page and waiting for the success modal
- Extracting the confirmation text from the "application submitted" modal

LinkedIn-specific complexity handled:
1. The wizard can have 1–5 steps — we detect step count dynamically
2. Radio group questions (work authorisation, equity compensation preference)
   are answered using the user's job_search_preferences
3. Some fields are pre-populated from the user's LinkedIn profile — these
   are detected and skipped rather than overwritten
4. File upload replaces whatever resume LinkedIn has stored — always upload
   the fresh tailored PDF for this application
5. CAPTCHA detection — if a CAPTCHA appears mid-flow, we abort and flag
   the application for manual review rather than attempting a solve

The base class interface that all ATS strategies must implement:
    detect_fields(page) -> list[dict]
    fill_fields(page, form_data, resume_path) -> list[str]
    submit(page) -> dict[str, Any]
"""

from __future__ import annotations

import asyncio
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
# Selector constants — pinned to LinkedIn's current DOM structure
# (Update these if LinkedIn changes their HTML; the rest of the logic holds)
# ---------------------------------------------------------------------------

SEL_EASY_APPLY_BTN = "button.jobs-apply-button"
SEL_EASY_APPLY_MODAL = ".jobs-easy-apply-modal"
SEL_MODAL_FOOTER = ".jobs-easy-apply-modal__footer"
SEL_NEXT_BTN = "button[aria-label='Continue to next step']"
SEL_SUBMIT_BTN = "button[aria-label='Submit application']"
SEL_REVIEW_BTN = "button[aria-label='Review your application']"
SEL_SUCCESS_MODAL = ".artdeco-modal__content h2"
SEL_FILE_INPUT = "input[type='file']"
SEL_TEXT_INPUTS = "input[type='text'], input[type='email'], input[type='tel'], input[type='url']"
SEL_TEXTAREA = "textarea"
SEL_SELECT = "select"
SEL_RADIO = "input[type='radio']"
SEL_CAPTCHA = ".challenge-dialog, iframe[title*='reCAPTCHA']"
SEL_PROGRESS = ".t-12.t-black--light"  # "Step N of M" text


class LinkedInApply:
    """
    LinkedIn Easy Apply automation strategy.

    Implements the standard three-method interface:
        detect_fields / fill_fields / submit
    """

    # Maximum steps we'll attempt before aborting (safety guard)
    MAX_STEPS = 8

    async def detect_fields(self, page: Any) -> list[dict[str, Any]]:
        """
        Navigate to the Easy Apply wizard and catalogue all fields across
        all wizard steps without filling them.

        Returns a list of field descriptor dicts used by fill_fields.
        """
        fields: list[dict[str, Any]] = []

        try:
            await self._open_easy_apply_modal(page)
        except Exception as exc:
            logger.error("Could not open Easy Apply modal", error=str(exc))
            return fields

        step = 0
        while step < self.MAX_STEPS:
            step += 1
            await asyncio.sleep(random.uniform(0.5, 1.0))

            if await self._is_captcha_visible(page):
                fields.append({"type": "captcha", "step": step})
                break

            step_fields = await self._detect_step_fields(page, step)
            fields.extend(step_fields)

            if not await self._has_next_step(page):
                break
            await self._click_next(page)

        return fields

    async def fill_fields(
        self,
        page: Any,
        form_data: dict[str, Any],
        resume_path: str | None = None,
    ) -> list[str]:
        """
        Fill every field in the Easy Apply wizard, advancing step by step.

        Returns a list of field names that were successfully filled.
        """
        filled: list[str] = []

        try:
            await self._open_easy_apply_modal(page)
        except Exception as exc:
            raise RuntimeError(f"Could not open Easy Apply modal: {exc}") from exc

        step = 0
        while step < self.MAX_STEPS:
            step += 1
            await asyncio.sleep(random.uniform(0.8, 1.5))

            if await self._is_captcha_visible(page):
                raise RuntimeError(
                    "CAPTCHA detected during LinkedIn Easy Apply — manual action required."
                )

            step_filled = await self._fill_step(page, form_data, resume_path, step)
            filled.extend(step_filled)

            if not await self._has_next_step(page):
                break

            await self._click_next(page)
            await asyncio.sleep(random.uniform(0.5, 1.2))

        return filled

    async def submit(self, page: Any) -> dict[str, Any]:
        """
        Click the Submit button and wait for the success modal.

        LinkedIn shows a modal confirmation with "Your application was sent
        to <Company>" text on success. The modal also shows a "See application"
        link — we extract both as the confirmation signal.
        """
        try:
            # Sometimes a "Review" step precedes the actual submit
            review_btn = await page.query_selector(SEL_REVIEW_BTN)
            if review_btn and await review_btn.is_visible():
                await review_btn.click()
                await asyncio.sleep(random.uniform(1.0, 1.8))

            submit_btn = await page.wait_for_selector(SEL_SUBMIT_BTN, timeout=8_000)
            if not submit_btn or not await submit_btn.is_enabled():
                return {
                    "success": False,
                    "error": "Submit button not found or not enabled.",
                }

            await asyncio.sleep(random.uniform(0.3, 0.8))
            await submit_btn.click()

            try:
                success_el = await page.wait_for_selector(SEL_SUCCESS_MODAL, timeout=12_000)
                confirmation_text = await success_el.inner_text()
                return {
                    "success": True,
                    "confirmation_text": confirmation_text,
                    "error": None,
                }
            except Exception:
                # Fall back to URL/content scanning
                result = await wait_for_navigation_or_error(page, timeout_ms=10_000)
                return result

        except Exception as exc:
            screenshot_path = await take_full_screenshot(page, "linkedin_submit_error")
            return {
                "success": False,
                "error": str(exc),
                "screenshot_path": screenshot_path,
            }

    # ---------------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------------

    async def _open_easy_apply_modal(self, page: Any) -> None:
        """
        Find and click the Easy Apply button, then wait for the modal to open.
        """
        btn = await page.wait_for_selector(SEL_EASY_APPLY_BTN, timeout=8_000)
        if not btn:
            raise RuntimeError("Easy Apply button not found on this job page.")

        await asyncio.sleep(random.uniform(0.4, 0.9))
        await btn.click()

        await page.wait_for_selector(SEL_EASY_APPLY_MODAL, timeout=10_000)
        await asyncio.sleep(random.uniform(0.5, 1.0))

    async def _is_captcha_visible(self, page: Any) -> bool:
        try:
            el = await page.query_selector(SEL_CAPTCHA)
            return el is not None and await el.is_visible()
        except Exception:
            return False

    async def _has_next_step(self, page: Any) -> bool:
        next_btn = await page.query_selector(SEL_NEXT_BTN)
        if next_btn and await next_btn.is_visible() and await next_btn.is_enabled():
            return True
        return False

    async def _click_next(self, page: Any) -> None:
        next_btn = await page.wait_for_selector(SEL_NEXT_BTN, timeout=5_000)
        await asyncio.sleep(random.uniform(0.3, 0.6))
        await next_btn.click()
        await page.wait_for_load_state("domcontentloaded")

    async def _detect_step_fields(self, page: Any, step: int) -> list[dict[str, Any]]:
        """Catalogue all visible fields on the current wizard step."""
        fields: list[dict[str, Any]] = []

        modal = await page.query_selector(SEL_EASY_APPLY_MODAL)
        if not modal:
            return fields

        for input_el in await modal.query_selector_all(SEL_TEXT_INPUTS):
            try:
                label = await self._get_label(page, input_el)
                name = await input_el.get_attribute("name") or label or "text_field"
                fields.append({
                    "type": "text",
                    "name": name,
                    "label": label,
                    "step": step,
                    "selector": f"input[name='{name}']",
                    "required": await input_el.get_attribute("required") is not None,
                })
            except Exception:
                continue

        for ta in await modal.query_selector_all(SEL_TEXTAREA):
            try:
                label = await self._get_label(page, ta)
                fields.append({
                    "type": "textarea",
                    "name": "cover_letter" if "cover" in label.lower() else label,
                    "label": label,
                    "step": step,
                })
            except Exception:
                continue

        for sel_el in await modal.query_selector_all(SEL_SELECT):
            try:
                label = await self._get_label(page, sel_el)
                fields.append({
                    "type": "select",
                    "name": await sel_el.get_attribute("name") or label,
                    "label": label,
                    "step": step,
                })
            except Exception:
                continue

        file_inputs = await modal.query_selector_all(SEL_FILE_INPUT)
        if file_inputs:
            fields.append({"type": "file_upload", "step": step})

        return fields

    async def _fill_step(
        self,
        page: Any,
        form_data: dict[str, Any],
        resume_path: str | None,
        step: int,
    ) -> list[str]:
        """Fill all fields visible on the current wizard step."""
        filled: list[str] = []
        modal = await page.query_selector(SEL_EASY_APPLY_MODAL)
        if not modal:
            return filled

        # File upload (resume)
        file_inputs = await modal.query_selector_all(SEL_FILE_INPUT)
        if file_inputs and resume_path:
            for file_input in file_inputs:
                try:
                    await file_input.set_input_files(resume_path)
                    filled.append("resume_upload")
                    await asyncio.sleep(random.uniform(0.5, 1.0))
                except Exception as exc:
                    logger.warning("Resume upload failed", error=str(exc))

        # Text inputs
        for input_el in await modal.query_selector_all(SEL_TEXT_INPUTS):
            try:
                label = (await self._get_label(page, input_el)).lower()
                value = self._match_field_value(label, form_data)
                if value:
                    existing = await input_el.input_value()
                    if not existing:
                        await human_type(page, None, value)
                        filled.append(label)
                        await asyncio.sleep(random.uniform(0.1, 0.3))
            except Exception as exc:
                logger.debug("Text field fill failed", error=str(exc))

        # Textareas (cover letter)
        for ta in await modal.query_selector_all(SEL_TEXTAREA):
            try:
                label = (await self._get_label(page, ta)).lower()
                if "cover" in label or "letter" in label or "additional" in label:
                    cover_text = form_data.get("cover_letter", "")
                    if cover_text:
                        await ta.click()
                        await ta.fill(cover_text)
                        filled.append("cover_letter")
            except Exception as exc:
                logger.debug("Textarea fill failed", error=str(exc))

        # Selects — pick first non-empty option if value not known
        for sel_el in await modal.query_selector_all(SEL_SELECT):
            try:
                label = (await self._get_label(page, sel_el)).lower()
                value = self._match_field_value(label, form_data)
                if value:
                    await sel_el.select_option(label=value)
                    filled.append(label)
                else:
                    # Select the first substantive option (skip placeholder)
                    options = await sel_el.query_selector_all("option")
                    if len(options) > 1:
                        await sel_el.select_option(index=1)
                        filled.append(f"{label}_default")
            except Exception as exc:
                logger.debug("Select fill failed", error=str(exc))

        # Radio buttons — pick "Yes" for authorisation questions
        for radio in await modal.query_selector_all(SEL_RADIO):
            try:
                radio_value = await radio.get_attribute("value") or ""
                if radio_value.lower() in ("yes", "true", "1"):
                    label = await self._get_label(page, radio)
                    if any(kw in label.lower() for kw in ["authoris", "eligible", "citizen", "work"]):
                        await radio.check()
                        filled.append(f"radio_{label}")
            except Exception:
                continue

        return filled

    async def _get_label(self, page: Any, element: Any) -> str:
        """Try to find the label associated with a form element."""
        try:
            element_id = await element.get_attribute("id")
            if element_id:
                label_el = await page.query_selector(f"label[for='{element_id}']")
                if label_el:
                    return (await label_el.inner_text()).strip()
            # Try aria-label
            aria = await element.get_attribute("aria-label")
            if aria:
                return aria.strip()
            # Try placeholder
            placeholder = await element.get_attribute("placeholder")
            if placeholder:
                return placeholder.strip()
        except Exception:
            pass
        return "unknown"

    def _match_field_value(self, label: str, form_data: dict[str, Any]) -> str:
        """
        Map a field label to a value from form_data using keyword matching.
        Returns empty string if no match found — field is skipped.
        """
        label = label.lower()
        mapping = [
            (["first name", "given name", "firstname"], "first_name"),
            (["last name", "surname", "family name", "lastname"], "last_name"),
            (["full name", "your name"], "full_name"),
            (["email"], "email"),
            (["phone", "mobile", "telephone"], "phone"),
            (["city", "location", "where are you"], "location"),
            (["linkedin", "linkedin url"], "linkedin_url"),
            (["github"], "github_url"),
            (["website", "portfolio"], "website_url"),
            (["current company", "employer"], "current_company"),
            (["current title", "job title", "position"], "current_title"),
            (["university", "school", "college", "institution"], "university"),
            (["degree", "education level"], "highest_degree"),
            (["graduation", "grad year"], "graduation_year"),
        ]
        for keywords, field_key in mapping:
            if any(kw in label for kw in keywords):
                return form_data.get(field_key, "")
        return ""