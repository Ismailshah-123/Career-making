"""
app/automation/playwright/indeed_apply.py
==========================================
Indeed Easy Apply automation strategy.

Indeed's application flow comes in two forms:
1. Indeed-hosted form — a multi-step form hosted at indeed.com/applystart
   with consistent selectors. This is the common path for "Easy Apply" jobs.
2. External redirect — Indeed shows a "Continue" button that opens the
   employer's own ATS in a new tab/popup. We detect this and delegate to
   CompanyApply (generic strategy) with the extracted URL.

Steps handled for Indeed-hosted forms:
- Resume selection (pick "Upload a new resume" if tailored PDF supplied)
- Contact info pre-fill detection (skip already-filled fields)
- Work experience questions (years of experience selects)
- Screening questions (custom per-employer, answered via keyword matching)
- Voluntary disclosures (EEOC, veteran status — always decline to self-identify)
- Review and submit
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

SEL_APPLY_BUTTON = "button[id*='apply-button'], a[id*='apply-button'], .ia-IndeedApplyButton"
SEL_CONTINUE_BTN = "button[data-testid='continueButton'], button.ia-continueButton"
SEL_SUBMIT_BTN = "button[data-testid='submit-application'], button[aria-label*='Submit']"
SEL_RESUME_UPLOAD = "input[type='file'][name='resume'], input.ia-ResumeUpload-fileInput"
SEL_SCREENING_QUESTIONS = ".ia-Questions, .ia-ScreeningQuestion"
SEL_EXTERNAL_APPLY = "a[href*='external'], .icl-Button--primary[target='_blank']"
SEL_SUCCESS = ".ia-PostApply, [data-testid='post-apply-toast']"


class IndeedApply:
    """Indeed Easy Apply automation strategy."""

    MAX_STEPS = 10

    async def detect_fields(self, page: Any) -> list[dict[str, Any]]:
        fields: list[dict[str, Any]] = []
        await self._open_apply_flow(page)

        step = 0
        while step < self.MAX_STEPS:
            step += 1
            step_fields = await self._detect_step_fields(page, step)
            fields.extend(step_fields)
            if not await self._has_continue(page):
                break
            await self._click_continue(page)
            await asyncio.sleep(random.uniform(0.5, 1.0))

        return fields

    async def fill_fields(
        self,
        page: Any,
        form_data: dict[str, Any],
        resume_path: str | None = None,
    ) -> list[str]:
        filled: list[str] = []

        await self._open_apply_flow(page)

        step = 0
        while step < self.MAX_STEPS:
            step += 1
            await asyncio.sleep(random.uniform(0.8, 1.5))

            # Check if redirected to external ATS
            if await self._is_external_redirect(page):
                external_url = await self._extract_external_url(page)
                if external_url:
                    filled.append(f"delegated_to_external:{external_url}")
                break

            # Resume upload
            resume_input = await page.query_selector(SEL_RESUME_UPLOAD)
            if resume_input and resume_path:
                try:
                    await resume_input.set_input_files(resume_path)
                    filled.append("resume_upload")
                    await asyncio.sleep(1.0)
                except Exception as exc:
                    logger.warning("Indeed resume upload failed", error=str(exc))

            # Text/email/phone fields
            for selector, key in [
                ("input[name='applicant.name']", "full_name"),
                ("input[name='applicant.emailAddress']", "email"),
                ("input[name='applicant.phoneNumber']", "phone"),
                ("input[name='applicant.city']", "location"),
            ]:
                try:
                    el = await page.query_selector(selector)
                    if el and await el.is_visible():
                        existing = await el.input_value()
                        if not existing and form_data.get(key):
                            await el.fill(form_data[key])
                            filled.append(key)
                            await asyncio.sleep(random.uniform(0.1, 0.3))
                except Exception:
                    continue

            # Screening questions
            screening_filled = await self._fill_screening_questions(page, form_data)
            filled.extend(screening_filled)

            # EEOC / voluntary disclosures — always "decline to self-identify"
            eeoc_filled = await self._handle_eeoc_questions(page)
            filled.extend(eeoc_filled)

            if not await self._has_continue(page):
                break
            await self._click_continue(page)

        return filled

    async def submit(self, page: Any) -> dict[str, Any]:
        try:
            submit_btn = await page.query_selector(SEL_SUBMIT_BTN)
            if not submit_btn or not await submit_btn.is_visible():
                return {"success": False, "error": "Submit button not found."}

            await asyncio.sleep(random.uniform(0.4, 0.8))
            await submit_btn.click()

            # Wait for Indeed's post-apply success indicator
            try:
                await page.wait_for_selector(SEL_SUCCESS, timeout=12_000)
                return {"success": True, "confirmation_text": "Indeed application submitted successfully."}
            except Exception:
                return await wait_for_navigation_or_error(page, timeout_ms=10_000)

        except Exception as exc:
            screenshot_path = await take_full_screenshot(page, "indeed_submit_error")
            return {"success": False, "error": str(exc), "screenshot_path": screenshot_path}

    async def _open_apply_flow(self, page: Any) -> None:
        apply_btn = await page.query_selector(SEL_APPLY_BUTTON)
        if apply_btn and await apply_btn.is_visible():
            await apply_btn.click()
            await asyncio.sleep(random.uniform(1.0, 1.8))

    async def _has_continue(self, page: Any) -> bool:
        btn = await page.query_selector(SEL_CONTINUE_BTN)
        return btn is not None and await btn.is_visible() and await btn.is_enabled()

    async def _click_continue(self, page: Any) -> None:
        btn = await page.wait_for_selector(SEL_CONTINUE_BTN, timeout=5_000)
        await asyncio.sleep(random.uniform(0.2, 0.5))
        await btn.click()
        await page.wait_for_load_state("domcontentloaded")

    async def _is_external_redirect(self, page: Any) -> bool:
        external = await page.query_selector(SEL_EXTERNAL_APPLY)
        return external is not None and await external.is_visible()

    async def _extract_external_url(self, page: Any) -> str | None:
        el = await page.query_selector(SEL_EXTERNAL_APPLY)
        if el:
            return await el.get_attribute("href")
        return None

    async def _detect_step_fields(self, page: Any, step: int) -> list[dict[str, Any]]:
        fields: list[dict[str, Any]] = []
        for input_el in await page.query_selector_all("input[type='text'], input[type='email']"):
            try:
                name = await input_el.get_attribute("name") or "unknown"
                fields.append({"type": "text", "name": name, "step": step})
            except Exception:
                continue
        return fields

    async def _fill_screening_questions(self, page: Any, form_data: dict[str, Any]) -> list[str]:
        """
        Answer custom employer screening questions using keyword pattern
        matching. Defaults to safe/positive answers for binary questions.
        """
        filled: list[str] = []
        questions = await page.query_selector_all(SEL_SCREENING_QUESTIONS)

        for question in questions:
            try:
                question_text = (await question.inner_text()).lower()

                # Yes/No radio buttons
                yes_radio = await question.query_selector("input[value='YES'], input[value='yes']")
                no_radio = await question.query_selector("input[value='NO'], input[value='no']")

                if yes_radio and no_radio:
                    # Authorisation, availability questions → Yes
                    positive_keywords = ["authoris", "eligible", "legally", "available", "willing", "can you", "do you have experience"]
                    if any(kw in question_text for kw in positive_keywords):
                        await yes_radio.check()
                        filled.append("screening_yes")
                    else:
                        await no_radio.check()
                        filled.append("screening_no")
                    continue

                # Number inputs (years of experience)
                number_input = await question.query_selector("input[type='number'], input[type='text'][class*='year']")
                if number_input and ("year" in question_text or "experience" in question_text):
                    await number_input.fill("5")
                    filled.append("years_experience")
                    continue

                # Select dropdowns
                select_el = await question.query_selector("select")
                if select_el:
                    options = await select_el.query_selector_all("option")
                    if len(options) > 1:
                        await select_el.select_option(index=1)
                        filled.append("screening_select")

            except Exception as exc:
                logger.debug("Screening question fill failed", error=str(exc))

        return filled

    async def _handle_eeoc_questions(self, page: Any) -> list[str]:
        """Select 'Decline to self-identify' for all EEOC / voluntary disclosure questions."""
        filled: list[str] = []
        eeoc_selectors = [
            "select[name*='eeoc']",
            "select[name*='race']",
            "select[name*='gender']",
            "select[name*='veteran']",
            "select[name*='disability']",
        ]
        for selector in eeoc_selectors:
            try:
                el = await page.query_selector(selector)
                if el and await el.is_visible():
                    options = await el.query_selector_all("option")
                    for opt in options:
                        text = (await opt.inner_text()).lower()
                        if "decline" in text or "prefer not" in text or "not specified" in text:
                            val = await opt.get_attribute("value")
                            await el.select_option(value=val)
                            filled.append(f"eeoc_{selector}")
                            break
            except Exception:
                continue
        return filled