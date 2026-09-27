"""
app/automation/playwright/browser.py
=======================================
Shared Playwright browser management for the JobHunter AI automation layer.

Provides:
- get_browser_context()    : Async context manager — launches a stealth
                              Chromium browser, yields (browser, page), closes
                              on exit. Used by all form-fill modules.
- detect_ats_from_url()    : Classify an apply_url into a known ATS provider
                              using URL pattern matching + lightweight HEAD
                              request inspection. No browser needed.
- BrowserPool              : Singleton that reuses a single browser process
                              across multiple concurrent page tasks, saving
                              ~2s startup overhead per application.
- take_full_screenshot()   : Utility — captures and saves a full-page PNG.
- human_type()             : Simulates human-paced keystroke input to reduce
                              bot-detection triggers on ATS systems.
- wait_for_navigation_or_error() : Waits for either a success URL/element or
                              an error indicator, whichever comes first.

Stealth measures applied (layered, not relying on any single bypass):
1. Custom Chromium args: disable automation flags, randomise viewport
2. navigator.webdriver = false injection via page.add_init_script
3. Random user-agent rotation from a curated modern-browser list
4. Human-paced typing with variance (50–150ms per keystroke)
5. Random micro-delays between form interactions (200–800ms)
6. No headless flag in the User-Agent string
7. Proxy support (configured via settings.PLAYWRIGHT_PROXY_URL)

Security note: these measures reduce detection but do not eliminate it.
Some ATS platforms (notably Workday and some iCIMS instances) run
Akamai / PerimeterX bot detection that can still trigger. The verify_submission
node in application_workflow.py handles this by routing to manual review
rather than crashing the pipeline.
"""

from __future__ import annotations

import asyncio
import os
import random
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Browser configuration
# ---------------------------------------------------------------------------

_MODERN_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:126.0) Gecko/20100101 Firefox/126.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_5) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
]

_CHROMIUM_LAUNCH_ARGS = [
    "--no-sandbox",
    "--disable-setuid-sandbox",
    "--disable-blink-features=AutomationControlled",
    "--disable-infobars",
    "--disable-dev-shm-usage",
    "--disable-gpu",
    "--no-first-run",
    "--no-default-browser-check",
    "--disable-extensions",
    "--disable-default-apps",
    "--disable-background-timer-throttling",
    "--disable-renderer-backgrounding",
    "--disable-backgrounding-occluded-windows",
    "--disable-ipc-flooding-protection",
]

_STEALTH_INIT_SCRIPT = """
Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
Object.defineProperty(navigator, 'plugins', { get: () => [1, 2, 3] });
Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en'] });
window.chrome = { runtime: {} };
"""

_VIEWPORTS = [
    {"width": 1920, "height": 1080},
    {"width": 1440, "height": 900},
    {"width": 1366, "height": 768},
    {"width": 2560, "height": 1440},
]


# ---------------------------------------------------------------------------
# Browser Pool (shared browser process across concurrent page tasks)
# ---------------------------------------------------------------------------

class BrowserPool:
    """
    Singleton that owns a single Playwright Chromium browser process and
    dispenses new BrowserContext (isolated session) instances on demand.

    Each application gets its own context (separate cookies, localStorage,
    network interception scope) but shares the browser process startup cost.
    The pool is initialised lazily on first use and shut down during
    application lifespan cleanup.
    """

    _instance: "BrowserPool | None" = None
    _playwright: Any = None
    _browser: Any = None
    _lock = asyncio.Lock()

    @classmethod
    async def get(cls) -> "BrowserPool":
        if cls._instance is None:
            async with cls._lock:
                if cls._instance is None:
                    instance = cls()
                    await instance._start()
                    cls._instance = instance
        return cls._instance

    async def _start(self) -> None:
        from playwright.async_api import async_playwright

        self._playwright = await async_playwright().start()

        launch_kwargs: dict[str, Any] = {
            "args": _CHROMIUM_LAUNCH_ARGS,
            "headless": settings.PLAYWRIGHT_HEADLESS,
            "slow_mo": settings.PLAYWRIGHT_SLOW_MO_MS,
        }
        if settings.PLAYWRIGHT_PROXY_URL:
            launch_kwargs["proxy"] = {"server": settings.PLAYWRIGHT_PROXY_URL}

        self._browser = await self._playwright.chromium.launch(**launch_kwargs)
        logger.info("Playwright browser pool started", headless=settings.PLAYWRIGHT_HEADLESS)

    async def new_context(self) -> Any:
        """Return a new isolated BrowserContext with stealth settings applied."""
        viewport = random.choice(_VIEWPORTS)
        user_agent = random.choice(_MODERN_USER_AGENTS)

        context = await self._browser.new_context(
            viewport=viewport,
            user_agent=user_agent,
            locale="en-US",
            timezone_id="America/New_York",
            permissions=["geolocation"],
            accept_downloads=True,
            java_script_enabled=True,
            ignore_https_errors=False,
        )
        context.set_default_timeout(settings.PLAYWRIGHT_TIMEOUT_MS)
        context.set_default_navigation_timeout(settings.PLAYWRIGHT_TIMEOUT_MS * 2)
        return context

    async def close(self) -> None:
        """Shutdown the browser pool cleanly. Called from lifespan cleanup."""
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()
        BrowserPool._instance = None
        logger.info("Playwright browser pool closed.")


# ---------------------------------------------------------------------------
# Context manager for per-task browser sessions
# ---------------------------------------------------------------------------

@asynccontextmanager
async def get_browser_context() -> AsyncGenerator[tuple[Any, Any], None]:
    """
    Async context manager — yields (browser_context, page) for one automation
    task. The context is closed on exit regardless of success or failure,
    ensuring no session state leaks between applications.

    The stealth init script is injected before any navigation so it's
    present from the very first network request.

    Usage:
        async with get_browser_context() as (ctx, page):
            await page.goto("https://jobs.greenhouse.io/...")
            ...
    """
    pool = await BrowserPool.get()
    ctx = await pool.new_context()
    page = await ctx.new_page()

    try:
        await page.add_init_script(_STEALTH_INIT_SCRIPT)
        yield ctx, page
    finally:
        try:
            await page.close()
        except Exception:
            pass
        try:
            await ctx.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# ATS detection
# ---------------------------------------------------------------------------

_ATS_URL_PATTERNS: list[tuple[list[str], str]] = [
    (["boards.greenhouse.io", "app.greenhouse.io"], "greenhouse"),
    (["jobs.lever.co", "lever.co/"], "lever"),
    (["myworkdayjobs.com", "workday.com"], "workday"),
    (["jobs.ashbyhq.com", "ashbyhq.com"], "ashby"),
    (["smartrecruiters.com"], "smartrecruiters"),
    (["taleo.net", "talesystem.com", "oracle.taleo"], "taleo"),
    (["icims.com"], "icims"),
    (["linkedin.com/jobs/view", "linkedin.com/jobs/apply"], "linkedin_easy_apply"),
    (["indeed.com/viewjob", "indeed.com/applystart"], "indeed"),
    (["remoteok.io", "remoteok.com"], "remoteok"),
    (["wellfound.com/jobs"], "wellfound"),
    (["bamboohr.com"], "bamboohr"),
    (["workable.com"], "workable"),
    (["recruitee.com"], "recruitee"),
    (["jobvite.com"], "jobvite"),
]


async def detect_ats_from_url(url: str) -> str:
    """
    Classify an apply_url into a known ATS provider.

    Uses URL pattern matching first (fast, no network). If no pattern
    matches, issues a single HEAD request and checks for ATS-specific
    response headers (X-Greenhouse-Job, Lever-Version, x-workday-*).
    Falls back to 'generic' if nothing is detected.
    """
    url_lower = url.lower()

    # Fast path: URL pattern matching
    for patterns, ats in _ATS_URL_PATTERNS:
        if any(p in url_lower for p in patterns):
            logger.debug("ATS detected via URL pattern", ats=ats, url=url[:60])
            return ats

    # Slow path: HEAD request for response headers
    try:
        import httpx
        async with httpx.AsyncClient(timeout=5, follow_redirects=True) as client:
            response = await client.head(url, headers={"User-Agent": random.choice(_MODERN_USER_AGENTS)})
            headers_lower = {k.lower(): v.lower() for k, v in response.headers.items()}
            final_url = str(response.url).lower()

            # Re-check patterns against the final redirect destination
            for patterns, ats in _ATS_URL_PATTERNS:
                if any(p in final_url for p in patterns):
                    return ats

            # Check ATS-specific response headers
            if "x-greenhouse" in str(headers_lower):
                return "greenhouse"
            if "lever" in str(headers_lower):
                return "lever"
            if "workday" in str(headers_lower):
                return "workday"
    except Exception as exc:
        logger.debug("ATS HEAD request failed (non-fatal)", error=str(exc))

    return "generic"


# ---------------------------------------------------------------------------
# Human typing simulation
# ---------------------------------------------------------------------------

async def human_type(page: Any, selector: str, text: str, *, clear_first: bool = True) -> None:
    """
    Type text into a form field with human-paced keystroke timing.

    min/max delay between keystrokes is set in constants.HUMAN_TYPING_DELAY_MS.
    An occasional longer pause (simulating a brief think) is injected every
    8–15 characters to break up statistically detectable uniform intervals.
    """
    from app.core.constants import HUMAN_TYPING_DELAY_MS

    element = await page.wait_for_selector(selector, timeout=10_000)
    await element.click()
    await asyncio.sleep(random.uniform(0.1, 0.3))

    if clear_first:
        await element.select_text()
        await page.keyboard.press("Backspace")
        await asyncio.sleep(random.uniform(0.1, 0.2))

    min_delay, max_delay = HUMAN_TYPING_DELAY_MS
    think_pause_interval = random.randint(8, 15)

    for i, char in enumerate(text):
        await element.type(char, delay=random.randint(min_delay, max_delay))
        if (i + 1) % think_pause_interval == 0:
            await asyncio.sleep(random.uniform(0.3, 0.9))


# ---------------------------------------------------------------------------
# Navigation outcome detection
# ---------------------------------------------------------------------------

_SUCCESS_URL_PATTERNS = [
    "/confirmation", "/confirm", "/thank-you", "/thankyou",
    "/success", "/submitted", "/complete", "/done",
    "/application-submitted", "/apply-success",
]

_SUCCESS_TEXT_PATTERNS = [
    "application received",
    "successfully applied",
    "thank you for applying",
    "your application has been submitted",
    "we received your application",
    "application submitted",
    "application complete",
]

_ERROR_SELECTORS = [
    ".error-message",
    ".alert-danger",
    "[data-error]",
    ".form-error",
    "#error-banner",
    ".validation-error",
]


async def wait_for_navigation_or_error(
    page: Any,
    *,
    timeout_ms: int = 15_000,
) -> dict[str, Any]:
    """
    Wait up to timeout_ms for a success or error signal after form submission.

    Returns:
        {"success": bool, "confirmation_text": str, "error": str | None}
    """
    deadline = asyncio.get_event_loop().time() + (timeout_ms / 1000)

    while asyncio.get_event_loop().time() < deadline:
        await asyncio.sleep(0.5)

        current_url = page.url.lower()
        if any(pattern in current_url for pattern in _SUCCESS_URL_PATTERNS):
            return {
                "success": True,
                "confirmation_text": await page.title() or "Application submitted.",
                "error": None,
            }

        try:
            content = await page.content()
            content_lower = content.lower()
            for pattern in _SUCCESS_TEXT_PATTERNS:
                if pattern in content_lower:
                    return {
                        "success": True,
                        "confirmation_text": pattern,
                        "error": None,
                    }
        except Exception:
            pass

        for selector in _ERROR_SELECTORS:
            try:
                error_el = await page.query_selector(selector)
                if error_el and await error_el.is_visible():
                    error_text = await error_el.inner_text()
                    return {"success": False, "confirmation_text": "", "error": error_text}
            except Exception:
                continue

    return {
        "success": False,
        "confirmation_text": "",
        "error": f"Timed out after {timeout_ms}ms waiting for submission confirmation.",
    }


# ---------------------------------------------------------------------------
# Screenshot utility
# ---------------------------------------------------------------------------

async def take_full_screenshot(page: Any, name: str) -> str | None:
    """
    Capture a full-page screenshot and save it under uploads/screenshots/.

    Returns the file path on success, None on failure (non-fatal — we don't
    want a screenshot failure to cascade into an application failure).
    """
    try:
        screenshots_dir = os.path.join("uploads", "screenshots")
        os.makedirs(screenshots_dir, exist_ok=True)
        path = os.path.join(screenshots_dir, f"{name}.png")
        await page.screenshot(path=path, full_page=True)
        return path
    except Exception as exc:
        logger.warning("Screenshot failed", name=name, error=str(exc))
        return None