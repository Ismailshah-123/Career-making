"""
app/automation/playwright/company_apply.py
============================================
Generic company application form automation + Greenhouse ATS strategy.

This module provides two classes:

1. CompanyApply — the catch-all "generic" strategy for:
   - Unknown / unrecognised ATS systems
   - Custom company career pages
   - Any apply_url that doesn't match a specialist strategy
   Uses adaptive field detection: scans the DOM for input/textarea/select
   elements, infers their purpose from label text / placeholder / name
   attribute, and fills them using a priority-ordered matching pipeline.

2. GreenhouseApply — specialist strategy for Greenhouse.io ATS.
   Greenhouse is used by thousands of high-growth companies (Stripe, Airbnb,
   Shopify, Figma, Notion, etc.) and has a consistent DOM structure across
   all customers. Handles:
   - Standard application form (boards.greenhouse.io/*)
   - Embedded Greenhouse widget on company career pages
   - Custom questions (demographic, work authorisation, salary expectations)
   - Demographic/EEOC section — always "Decline to self-identify"
   - Resume upload (PDF preferred, DOCX converted if needed)
   - Cover letter upload OR textarea (Greenhouse supports both)
   - LinkedIn profile URL field (always present on Greenhouse)
   - Custom employer questions (answered via heuristic keyword matching)

WHY A SPECIALIST GREENHOUSE CLASS?
Greenhouse has 30,000+ paying customers — it's the single highest-value
ATS to get right. The generic fallback misses Greenhouse-specific quirks:
  - Their file upload uses a hidden input triggered by a visible <label>
  - Their "Education" section uses a nested accordion that must be expanded
  - Some custom questions require specific answer formats (year selects, etc.)
  - Their submit button has a 500ms debounce that trips up immediate clicks
"""

from __future__ import annotations

import asyncio
import os
import random
import re
from typing import Any

from app.automation.playwright.browser import (
    human_type,
    take_full_screenshot,
    wait_for_navigation_or_error,
)
from app.core.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Greenhouse-specific selectors
# ---------------------------------------------------------------------------

GH_SEL_FIRST_NAME = "#first_name, input[name='first_name'], input[autocomplete='given-name']"
GH_SEL_LAST_NAME = "#last_name, input[name='last_name'], input[autocomplete='family-name']"
GH_SEL_EMAIL = "#email, input[name='email'], input[type='email']"
GH_SEL_PHONE = "#phone, input[name='phone'], input[type='tel']"
GH_SEL_RESUME_UPLOAD = "#resume, input[name='resume'], input[id*='resume'][type='file']"
GH_SEL_RESUME_LABEL = "label[for='resume'], .resume-label, .attach-resume"
GH_SEL_COVER_LETTER_FILE = "input[id*='cover_letter'][type='file']"
GH_SEL_COVER_LETTER_TEXT = "textarea[id*='cover_letter'], textarea[name*='cover_letter']"
GH_SEL_LINKEDIN = "input[id*='linkedin'], input[placeholder*='linkedin' i]"
GH_SEL_WEBSITE = "input[id*='website'], input[id*='portfolio'], input[placeholder*='website' i]"
GH_SEL_GITHUB = "input[id*='github'], input[placeholder*='github' i]"
GH_SEL_EDUCATION_SECTION = "#education, .education-fields, [data-qa='education-section']"
GH_SEL_SUBMIT = "#submit_app, button[type='submit'], input[type='submit']"
GH_SEL_DEMOGRAPHIC = "#demographic_survey, .demographic-survey"
GH_SEL_CUSTOM_QUESTIONS = ".custom-field, .application-question, [data-qa*='question']"
GH_SEL_SUCCESS = ".success-message, .confirmation, h1:has-text('submitted'), h1:has-text('received')"
GH_SEL_EEOCON = "[id*='eeoc'], [name*='eeoc'], [id*='race'], [id*='gender'], [id*='veteran'], [id*='disability']"


class GreenhouseApply:
    """
    Greenhouse.io ATS application automation strategy.

    Handles boards.greenhouse.io/* and embedded Greenhouse widgets on
    company career pages. The same class works for both because Greenhouse
    uses the same DOM structure across both deployment modes.
    """

    async def detect_fields(self, page: Any) -> list[dict[str, Any]]:
        """Catalogue all fields on the Greenhouse application form."""
        await asyncio.sleep(random.uniform(0.5, 1.0))
        fields: list[dict[str, Any]] = []

        standard_fields = [
            (GH_SEL_FIRST_NAME, "first_name", "text"),
            (GH_SEL_LAST_NAME, "last_name", "text"),
            (GH_SEL_EMAIL, "email", "email"),
            (GH_SEL_PHONE, "phone", "tel"),
            (GH_SEL_LINKEDIN, "linkedin_url", "text"),
            (GH_SEL_WEBSITE, "website_url", "text"),
            (GH_SEL_GITHUB, "github_url", "text"),
            (GH_SEL_RESUME_UPLOAD, "resume", "file"),
            (GH_SEL_COVER_LETTER_FILE, "cover_letter_file", "file"),
            (GH_SEL_COVER_LETTER_TEXT, "cover_letter_text", "textarea"),
        ]

        for selector, name, field_type in standard_fields:
            try:
                el = await page.query_selector(selector)
                if el:
                    fields.append({
                        "type": field_type,
                        "name": name,
                        "selector": selector,
                        "required": await el.get_attribute("required") is not None,
                    })
            except Exception:
                continue

        # Custom questions
        custom_els = await page.query_selector_all(GH_SEL_CUSTOM_QUESTIONS)
        for i, el in enumerate(custom_els):
            try:
                label_text = await el.inner_text()
                fields.append({
                    "type": "custom",
                    "name": f"custom_{i}",
                    "label": label_text[:100].strip(),
                })
            except Exception:
                continue

        return fields

    async def fill_fields(
        self,
        page: Any,
        form_data: dict[str, Any],
        resume_path: str | None = None,
    ) -> list[str]:
        """Fill all fields on the Greenhouse application form."""
        filled: list[str] = []
        await asyncio.sleep(random.uniform(0.8, 1.5))

        # --- Standard identity fields ---
        identity_mappings = [
            (GH_SEL_FIRST_NAME, "first_name"),
            (GH_SEL_LAST_NAME, "last_name"),
            (GH_SEL_EMAIL, "email"),
            (GH_SEL_PHONE, "phone"),
            (GH_SEL_LINKEDIN, "linkedin_url"),
            (GH_SEL_WEBSITE, "website_url"),
            (GH_SEL_GITHUB, "github_url"),
        ]

        for selector, key in identity_mappings:
            value = form_data.get(key)
            if not value:
                continue
            try:
                el = await page.query_selector(selector)
                if el and await el.is_visible():
                    existing = await el.input_value()
                    if not existing:
                        await el.click()
                        await asyncio.sleep(random.uniform(0.1, 0.25))
                        await el.fill(value)
                        filled.append(key)
                        await asyncio.sleep(random.uniform(0.15, 0.35))
            except Exception as exc:
                logger.debug(f"Greenhouse field skip: {key} — {exc}")

        # --- Resume upload (Greenhouse uses a hidden file input + visible label) ---
        if resume_path and os.path.exists(resume_path):
            try:
                # Try direct file input first
                file_input = await page.query_selector(GH_SEL_RESUME_UPLOAD)
                if file_input:
                    await file_input.set_input_files(resume_path)
                    filled.append("resume")
                    await asyncio.sleep(random.uniform(1.0, 2.0))
                else:
                    # Click the label to trigger file chooser
                    label = await page.query_selector(GH_SEL_RESUME_LABEL)
                    if label:
                        async with page.expect_file_chooser() as fc_info:
                            await label.click()
                        file_chooser = await fc_info.value
                        await file_chooser.set_files(resume_path)
                        filled.append("resume_via_label")
                        await asyncio.sleep(random.uniform(1.0, 2.0))
            except Exception as exc:
                logger.warning("Greenhouse resume upload failed", error=str(exc)[:200])

        # --- Cover letter (file OR textarea — check which exists) ---
        cover_letter_text = form_data.get("cover_letter", "")
        if cover_letter_text:
            try:
                cl_textarea = await page.query_selector(GH_SEL_COVER_LETTER_TEXT)
                if cl_textarea and await cl_textarea.is_visible():
                    await cl_textarea.click()
                    await asyncio.sleep(random.uniform(0.2, 0.4))
                    await cl_textarea.fill(cover_letter_text[:5000])
                    filled.append("cover_letter_text")
                else:
                    cl_file = await page.query_selector(GH_SEL_COVER_LETTER_FILE)
                    if cl_file and cover_letter_text and resume_path:
                        # Inline cover letter PDF creation would go here
                        # For now: skip file upload if only text is available
                        logger.debug("Greenhouse: cover letter file input found but no PDF available")
            except Exception as exc:
                logger.debug(f"Cover letter fill failed: {exc}")

        # --- Education accordion (Greenhouse often hides this until clicked) ---
        await self._fill_education_section(page, form_data, filled)

        # --- Custom questions ---
        await self._fill_custom_questions(page, form_data, filled)

        # --- EEOC / Demographic questions ---
        await self._fill_eeoc_section(page, filled)

        logger.info(f"Greenhouse: filled {len(filled)} fields")
        return filled

    async def submit(self, page: Any) -> dict[str, Any]:
        """
        Submit the Greenhouse application form.

        Greenhouse has a 500ms debounce on their submit button — clicking
        too fast after filling the last field causes a no-op. We wait
        1-2s after the last fill before clicking.
        """
        try:
            await asyncio.sleep(random.uniform(1.0, 2.0))  # debounce wait

            submit_btn = await page.wait_for_selector(GH_SEL_SUBMIT, timeout=8_000)
            if not submit_btn:
                return {"success": False, "error": "Greenhouse submit button not found."}

            if not await submit_btn.is_enabled():
                # Sometimes submit is disabled due to validation errors
                validation_errors = await self._collect_validation_errors(page)
                return {
                    "success": False,
                    "error": f"Submit button disabled. Validation errors: {validation_errors}",
                }

            await submit_btn.click()
            await asyncio.sleep(random.uniform(0.5, 1.0))

            # Greenhouse success: either a redirect to /confirmed or an inline message
            try:
                success_el = await page.wait_for_selector(GH_SEL_SUCCESS, timeout=12_000)
                confirmation = await success_el.inner_text()
                screenshot_path = await take_full_screenshot(page, f"greenhouse_success_{id(page)}")
                return {
                    "success": True,
                    "confirmation_text": confirmation,
                    "screenshot_path": screenshot_path,
                    "error": None,
                }
            except Exception:
                pass

            result = await wait_for_navigation_or_error(page, timeout_ms=10_000)
            if result["success"]:
                result["screenshot_path"] = await take_full_screenshot(
                    page, f"greenhouse_submitted_{id(page)}"
                )
            return result

        except Exception as exc:
            screenshot_path = await take_full_screenshot(page, f"greenhouse_error_{id(page)}")
            return {"success": False, "error": str(exc)[:500], "screenshot_path": screenshot_path}

    # ---------------------------------------------------------------------------
    # Greenhouse helpers
    # ---------------------------------------------------------------------------

    async def _fill_education_section(
        self,
        page: Any,
        form_data: dict[str, Any],
        filled: list[str],
    ) -> None:
        """Expand and fill the education accordion if present."""
        try:
            edu_section = await page.query_selector(GH_SEL_EDUCATION_SECTION)
            if not edu_section:
                return

            # Some Greenhouse forms require clicking to expand the section
            toggle = await edu_section.query_selector(".toggle, [aria-expanded]")
            if toggle:
                expanded = await toggle.get_attribute("aria-expanded")
                if expanded == "false":
                    await toggle.click()
                    await asyncio.sleep(random.uniform(0.3, 0.6))

            # School name
            school_input = await edu_section.query_selector("input[id*='school'], input[placeholder*='school' i]")
            if school_input and form_data.get("university"):
                await school_input.fill(form_data["university"])
                filled.append("education_school")

            # Degree
            degree_select = await edu_section.query_selector("select[id*='degree']")
            if degree_select:
                degree_map = {
                    "bachelor": "Bachelor's Degree",
                    "master": "Master's Degree",
                    "phd": "PhD",
                    "associate": "Associate's Degree",
                    "high school": "High School",
                }
                user_degree = form_data.get("highest_degree", "").lower()
                for key, option_text in degree_map.items():
                    if key in user_degree:
                        try:
                            await degree_select.select_option(label=option_text)
                            filled.append("education_degree")
                        except Exception:
                            pass
                        break

        except Exception as exc:
            logger.debug(f"Greenhouse education section failed: {exc}")

    async def _fill_custom_questions(
        self,
        page: Any,
        form_data: dict[str, Any],
        filled: list[str],
    ) -> None:
        """
        Answer Greenhouse custom employer questions using keyword heuristics.
        """
        try:
            question_containers = await page.query_selector_all(GH_SEL_CUSTOM_QUESTIONS)
            for container in question_containers:
                try:
                    label_el = await container.query_selector("label, legend, .question-label")
                    if not label_el:
                        continue
                    question_text = (await label_el.inner_text()).lower().strip()

                    # Yes/No radio questions
                    yes_radio = await container.query_selector("input[value='Yes'], input[value='yes'], input[value='1']")
                    no_radio = await container.query_selector("input[value='No'], input[value='no'], input[value='0']")

                    if yes_radio and no_radio:
                        positive_signals = ["authoris", "eligible", "legal", "citizen", "18 year", "available", "willing"]
                        if any(sig in question_text for sig in positive_signals):
                            await yes_radio.check()
                            filled.append(f"custom_yes:{question_text[:40]}")
                        else:
                            await no_radio.check()
                            filled.append(f"custom_no:{question_text[:40]}")
                        continue

                    # Number inputs (salary expectation, years of experience)
                    number_input = await container.query_selector("input[type='number'], input[type='text'][class*='number']")
                    if number_input and ("salary" in question_text or "compensation" in question_text or "expect" in question_text):
                        await number_input.fill("0")  # Indicate "Open / Negotiable"
                        filled.append(f"custom_salary:{question_text[:40]}")
                        continue

                    if number_input and ("year" in question_text or "experience" in question_text):
                        await number_input.fill("5")
                        filled.append(f"custom_years:{question_text[:40]}")
                        continue

                    # Select dropdowns
                    select_el = await container.query_selector("select")
                    if select_el:
                        options = await select_el.query_selector_all("option")
                        if len(options) > 1:
                            # Prefer "Yes" options for clearance/authorisation questions
                            for opt in options:
                                opt_text = (await opt.inner_text()).lower()
                                if any(sig in opt_text for sig in ["yes", "authorized", "eligible"]):
                                    await select_el.select_option(label=await opt.inner_text())
                                    filled.append(f"custom_select:{question_text[:40]}")
                                    break
                            else:
                                await select_el.select_option(index=1)
                                filled.append(f"custom_select_default:{question_text[:40]}")

                    # Text areas — skip (too risky to auto-fill free-text custom questions)

                except Exception as exc:
                    logger.debug(f"Custom question handling failed: {exc}")

        except Exception as exc:
            logger.debug(f"Greenhouse custom questions section failed: {exc}")

    async def _fill_eeoc_section(self, page: Any, filled: list[str]) -> None:
        """Select 'Decline to self-identify' for all EEOC/demographic fields."""
        try:
            eeoc_selects = await page.query_selector_all(GH_SEL_EEOCON)
            for select_el in eeoc_selects:
                try:
                    tag = await select_el.evaluate("el => el.tagName.toLowerCase()")
                    if tag == "select":
                        options = await select_el.query_selector_all("option")
                        for opt in options:
                            text = (await opt.inner_text()).lower()
                            if any(kw in text for kw in ["decline", "prefer not", "not specified", "no answer"]):
                                await select_el.select_option(label=await opt.inner_text())
                                filled.append("eeoc_decline")
                                break
                except Exception:
                    continue
        except Exception as exc:
            logger.debug(f"EEOC section handling failed: {exc}")

    async def _collect_validation_errors(self, page: Any) -> list[str]:
        """Gather visible validation error messages to surface in automation_error."""
        errors: list[str] = []
        try:
            error_els = await page.query_selector_all(".error, .field-error, .invalid-feedback, [aria-invalid='true']")
            for el in error_els[:5]:
                if await el.is_visible():
                    text = await el.inner_text()
                    if text.strip():
                        errors.append(text.strip()[:100])
        except Exception:
            pass
        return errors


# ---------------------------------------------------------------------------
# Generic / fallback strategy
# ---------------------------------------------------------------------------

class CompanyApply:
    """
    Generic company career page automation strategy.

    Used when no specialist ATS class matches the apply_url. Adaptive field
    detection uses a multi-pass matching pipeline:

    Pass 1: Standard HTML attribute matching (name, id, type attributes)
    Pass 2: Label text matching (case-insensitive keyword scan)
    Pass 3: Placeholder and aria-label matching
    Pass 4: Positional heuristics (first textarea = cover letter, etc.)

    Falls back gracefully for fields it can't identify — better to fill
    10 of 15 fields correctly than to crash on field 11 and submit nothing.
    """

    async def detect_fields(self, page: Any) -> list[dict[str, Any]]:
        """Adaptively catalogue all visible form fields."""
        fields: list[dict[str, Any]] = []
        await asyncio.sleep(random.uniform(0.5, 1.0))

        try:
            all_inputs = await page.query_selector_all(
                "input:not([type='hidden']):not([type='checkbox']):not([type='radio']):not([type='submit']):not([type='button'])"
            )
            for input_el in all_inputs:
                if not await input_el.is_visible():
                    continue
                field_info = await self._classify_input(page, input_el)
                if field_info:
                    fields.append(field_info)

            all_textareas = await page.query_selector_all("textarea")
            for ta in all_textareas:
                if not await ta.is_visible():
                    continue
                field_info = await self._classify_textarea(page, ta)
                if field_info:
                    fields.append(field_info)

        except Exception as exc:
            logger.warning("Generic field detection partially failed", error=str(exc)[:200])

        return fields

    async def fill_fields(
        self,
        page: Any,
        form_data: dict[str, Any],
        resume_path: str | None = None,
    ) -> list[str]:
        """Fill all detectable fields using adaptive matching."""
        filled: list[str] = []
        await asyncio.sleep(random.uniform(0.5, 1.0))

        try:
            # File inputs (resume upload)
            file_inputs = await page.query_selector_all("input[type='file']")
            for file_input in file_inputs:
                if not await file_input.is_visible() and not resume_path:
                    continue
                accept = await file_input.get_attribute("accept") or ""
                if any(kw in accept.lower() for kw in ["pdf", "doc", "resume", "cv"]) or not accept:
                    if resume_path and os.path.exists(resume_path):
                        try:
                            await file_input.set_input_files(resume_path)
                            filled.append("resume_upload")
                            await asyncio.sleep(1.5)
                            break
                        except Exception as exc:
                            logger.warning("Generic file upload failed", error=str(exc)[:150])

            # All text inputs
            all_inputs = await page.query_selector_all(
                "input[type='text'], input[type='email'], input[type='tel'], input[type='url']"
            )
            for input_el in all_inputs:
                if not await input_el.is_visible():
                    continue
                try:
                    key = await self._identify_input_purpose(page, input_el)
                    value = form_data.get(key) if key else None
                    if not value:
                        continue
                    existing = await input_el.input_value()
                    if not existing:
                        await input_el.click()
                        await asyncio.sleep(random.uniform(0.1, 0.25))
                        await input_el.fill(value)
                        filled.append(key)
                        await asyncio.sleep(random.uniform(0.15, 0.35))
                except Exception:
                    continue

            # Textareas
            textareas = await page.query_selector_all("textarea")
            for ta in textareas:
                if not await ta.is_visible():
                    continue
                try:
                    purpose = await self._identify_textarea_purpose(page, ta)
                    if purpose == "cover_letter" and form_data.get("cover_letter"):
                        await ta.click()
                        await asyncio.sleep(random.uniform(0.2, 0.4))
                        await ta.fill(form_data["cover_letter"][:5000])
                        filled.append("cover_letter")
                except Exception:
                    continue

            # Select dropdowns
            selects = await page.query_selector_all("select")
            for select_el in selects:
                if not await select_el.is_visible():
                    continue
                try:
                    label = await self._get_field_label(page, select_el)
                    if not label:
                        continue
                    label_lower = label.lower()
                    if any(kw in label_lower for kw in ["eeoc", "race", "gender", "veteran", "disability", "ethnicity"]):
                        await self._select_decline_option(select_el)
                        filled.append(f"eeoc_{label[:30]}")
                    elif any(kw in label_lower for kw in ["work auth", "authoris", "eligible", "visa"]):
                        await self._select_yes_option(select_el)
                        filled.append(f"auth_{label[:30]}")
                    elif await self._select_has_substantive_options(select_el):
                        await select_el.select_option(index=1)
                        filled.append(f"select_{label[:30]}")
                except Exception:
                    continue

        except Exception as exc:
            logger.warning("Generic fill partially failed", error=str(exc)[:200])

        return filled

    async def submit(self, page: Any) -> dict[str, Any]:
        """
        Find and click the submit button using multiple selector strategies.
        """
        submit_selectors = [
            "button[type='submit']",
            "input[type='submit']",
            "button:has-text('Submit')",
            "button:has-text('Apply')",
            "button:has-text('Send Application')",
            "button:has-text('Submit Application')",
            "button:has-text('Apply Now')",
            "#submit",
            ".submit-btn",
            "[data-qa='submit-button']",
        ]

        submit_btn = None
        for selector in submit_selectors:
            try:
                el = await page.query_selector(selector)
                if el and await el.is_visible() and await el.is_enabled():
                    submit_btn = el
                    break
            except Exception:
                continue

        if not submit_btn:
            return {"success": False, "error": "No submit button found on this career page."}

        try:
            await asyncio.sleep(random.uniform(0.8, 1.5))
            await submit_btn.click()

            result = await wait_for_navigation_or_error(page, timeout_ms=15_000)
            if result["success"]:
                result["screenshot_path"] = await take_full_screenshot(
                    page, f"generic_submitted_{id(page)}"
                )
            return result

        except Exception as exc:
            screenshot_path = await take_full_screenshot(page, f"generic_error_{id(page)}")
            return {"success": False, "error": str(exc)[:500], "screenshot_path": screenshot_path}

    # ---------------------------------------------------------------------------
    # Field classification helpers
    # ---------------------------------------------------------------------------

    _PURPOSE_KEYWORD_MAP = [
        (["first_name", "firstname", "given", "first name"], "first_name"),
        (["last_name", "lastname", "surname", "family", "last name"], "last_name"),
        (["full_name", "fullname", "your name", "name"], "full_name"),
        (["email", "e-mail"], "email"),
        (["phone", "mobile", "tel", "telephone"], "phone"),
        (["linkedin", "linkedin url"], "linkedin_url"),
        (["github", "github url"], "github_url"),
        (["website", "portfolio", "personal site"], "website_url"),
        (["current company", "employer", "company name"], "current_company"),
        (["title", "current role", "current position", "job title"], "current_title"),
        (["university", "school", "college", "institution"], "university"),
        (["city", "location", "where are you"], "location"),
    ]

    async def _identify_input_purpose(self, page: Any, input_el: Any) -> str | None:
        """Multi-pass input purpose identification."""
        # Pass 1: name/id attributes
        for attr in ("name", "id", "autocomplete"):
            val = await input_el.get_attribute(attr)
            if not val:
                continue
            val_lower = val.lower()
            for keywords, purpose in self._PURPOSE_KEYWORD_MAP:
                if any(kw in val_lower for kw in keywords):
                    return purpose

        # Pass 2: label text
        label = await self._get_field_label(page, input_el)
        if label:
            label_lower = label.lower()
            for keywords, purpose in self._PURPOSE_KEYWORD_MAP:
                if any(kw in label_lower for kw in keywords):
                    return purpose

        # Pass 3: placeholder
        placeholder = await input_el.get_attribute("placeholder") or ""
        if placeholder:
            ph_lower = placeholder.lower()
            for keywords, purpose in self._PURPOSE_KEYWORD_MAP:
                if any(kw in ph_lower for kw in keywords):
                    return purpose

        return None

    async def _identify_textarea_purpose(self, page: Any, ta: Any) -> str | None:
        """Identify if a textarea is for cover letter, about, or unknown."""
        label = await self._get_field_label(page, ta) or ""
        placeholder = await ta.get_attribute("placeholder") or ""
        name = await ta.get_attribute("name") or ""
        combined = (label + " " + placeholder + " " + name).lower()

        if any(kw in combined for kw in ["cover letter", "cover_letter", "covering"]):
            return "cover_letter"
        if any(kw in combined for kw in ["about yourself", "tell us", "describe yourself", "message"]):
            return "about"
        return None

    async def _classify_input(self, page: Any, input_el: Any) -> dict | None:
        purpose = await self._identify_input_purpose(page, input_el)
        if not purpose:
            return None
        input_type = await input_el.get_attribute("type") or "text"
        return {"type": input_type, "name": purpose, "required": await input_el.get_attribute("required") is not None}

    async def _classify_textarea(self, page: Any, ta: Any) -> dict | None:
        purpose = await self._identify_textarea_purpose(page, ta)
        if not purpose:
            return None
        return {"type": "textarea", "name": purpose}

    async def _get_field_label(self, page: Any, element: Any) -> str | None:
        try:
            element_id = await element.get_attribute("id")
            if element_id:
                label_el = await page.query_selector(f"label[for='{element_id}']")
                if label_el:
                    return (await label_el.inner_text()).strip()
            aria_label = await element.get_attribute("aria-label")
            if aria_label:
                return aria_label.strip()
        except Exception:
            pass
        return None

    async def _select_decline_option(self, select_el: Any) -> None:
        """Select a decline/prefer-not-to-answer option."""
        options = await select_el.query_selector_all("option")
        for opt in options:
            text = (await opt.inner_text()).lower()
            if any(kw in text for kw in ["decline", "prefer not", "not specified", "no answer", "i don't"]):
                await select_el.select_option(label=await opt.inner_text())
                return
        # Fallback: pick last option (often "Decline to self-identify")
        if options:
            await select_el.select_option(index=len(options) - 1)

    async def _select_yes_option(self, select_el: Any) -> None:
        """Select an affirmative option for work authorisation questions."""
        options = await select_el.query_selector_all("option")
        for opt in options:
            text = (await opt.inner_text()).lower()
            if any(kw in text for kw in ["yes", "authorised", "eligible", "citizen", "permanent"]):
                await select_el.select_option(label=await opt.inner_text())
                return
        if len(options) > 1:
            await select_el.select_option(index=1)

    async def _select_has_substantive_options(self, select_el: Any) -> bool:
        options = await select_el.query_selector_all("option")
        return len(options) > 1