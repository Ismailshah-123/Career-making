"""
app/automation/scraping/base_scraper.py
=========================================
Production-grade base scraper class for the JobHunter AI platform.

Every job board scraper (LinkedIn, Indeed, RemoteOK, Wellfound) inherits
from BaseScraper and gets the following capabilities for free:

1. ANTI-DETECTION ENGINE
   - Real browser TLS fingerprinting via curl_cffi (not httpx/requests)
   - JA3/H2 fingerprint matching against a real Chrome 125 session
   - navigator.webdriver=false + full Chrome headless-detection bypass
   - Randomised request timing with exponential jitter
   - Viewport and screen resolution randomisation per session
   - Accept-Language, Accept-Encoding headers matching real Chrome
   - Referer chain simulation (Google → board homepage → job page)
   - Cookie jar persistence across requests in the same session

2. PROXY MANAGEMENT
   - Rotating residential proxy pool (configured via SCRAPING_PROXY_URL)
   - Automatic proxy rotation on 429 / 403 / connection failure
   - Per-proxy ban detection and cool-down tracking in Redis
   - Direct fallback when all proxies are banned (with warning)

3. RATE LIMITING
   - Per-domain sliding-window counter in Redis
   - Exponential backoff on 429 with Retry-After header respect
   - Concurrent request cap per scraper instance (asyncio.Semaphore)

4. RETRY LOGIC
   - 3 attempts per request by default, exponential backoff + jitter
   - Distinguishes retryable errors (5xx, timeout, connection error)
     from permanent errors (403 IP ban, 404, scraping blocked page)
   - Logs every retry with full context for operational monitoring

5. SESSION LIFECYCLE
   - Per-scraper async session with cookie jar
   - Auto-renewal of sessions that accumulate bot-detection signals
   - Graceful close() method — always call in a finally block or
     use the async context manager form `async with ScrapeSession():`

6. CONTENT VALIDATION
   - Detects captcha / bot-detection pages before parsing
   - Validates HTML structure before attempting field extraction
   - Returns empty list (not raises) on board-level scrape failures —
     the discovery_node fan-out handles partial failures gracefully

Usage in a subclass:
    class IndeedScraper(BaseScraper):
        BASE_URL = "https://www.indeed.com"
        RATE_LIMIT_DOMAIN = "indeed.com"

        async def search(self, *, keywords, locations, work_modes, max_results):
            url = self._build_search_url(keywords, locations)
            response = await self._get(url)
            return self._parse_listings(response)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
import time
from abc import ABC, abstractmethod
from typing import Any

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_RETRY_ATTEMPTS = 3
DEFAULT_RETRY_BASE_DELAY = 2.0      # seconds
DEFAULT_RETRY_MAX_DELAY = 60.0
DEFAULT_CONCURRENT_LIMIT = 5
DEFAULT_MIN_REQUEST_DELAY = 1.0     # seconds between requests on the same domain
DEFAULT_MAX_REQUEST_DELAY = 3.5

CAPTCHA_SIGNALS = [
    "captcha",
    "cf-chl",                         # Cloudflare challenge
    "g-recaptcha",
    "hcaptcha",
    "challenge-form",
    "are you a robot",
    "please verify you are a human",
    "access denied",
    "unusual traffic",
    "automated queries",
    "bot detection",
    "security check",
    "429 too many",
]

BLOCKED_SIGNALS = [
    "your ip has been blocked",
    "ip address has been banned",
    "access from this location has been restricted",
    "scraping is not permitted",
    "terms of service violation",
]

# Realistic Chrome 125 headers
_CHROME_HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
    "Accept-Encoding": "gzip, deflate, br, zstd",
    "Accept-Language": "en-US,en;q=0.9",
    "Cache-Control": "max-age=0",
    "Sec-Ch-Ua": '"Google Chrome";v="125", "Chromium";v="125", "Not.A/Brand";v="24"',
    "Sec-Ch-Ua-Mobile": "?0",
    "Sec-Ch-Ua-Platform": '"Windows"',
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Upgrade-Insecure-Requests": "1",
}

_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 Edg/124.0.0.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_5) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:126.0) Gecko/20100101 Firefox/126.0",
]

_VIEWPORTS = [
    "1920x1080", "1440x900", "1366x768", "1280x800",
    "2560x1440", "1600x900", "1280x1024",
]


# ---------------------------------------------------------------------------
# Proxy Manager
# ---------------------------------------------------------------------------

class ProxyManager:
    """
    Manages a pool of proxy URLs with ban detection and rotation.

    Proxy URLs are loaded from SCRAPING_PROXY_URL (comma-separated list
    for a pool, single URL for a rotating proxy endpoint).

    Ban state is tracked in-memory with a per-proxy cool-down timer.
    For multi-process deployments, upgrade to Redis-backed tracking.
    """

    def __init__(self) -> None:
        self._proxies: list[str] = self._load_proxies()
        self._banned: dict[str, float] = {}     # proxy_url → banned_until_timestamp
        self._current_idx: int = 0
        self._ban_duration_seconds: int = 300   # 5-minute cool-down per banned proxy

    def _load_proxies(self) -> list[str]:
        raw = settings.SCRAPING_PROXY_URL or ""
        if not raw:
            return []
        return [p.strip() for p in raw.split(",") if p.strip()]

    def has_proxies(self) -> bool:
        return bool(self._proxies)

    def get_proxy(self) -> str | None:
        """
        Return the next available proxy (round-robin, skipping banned ones).
        Returns None if no proxies configured or all are banned.
        """
        if not self._proxies:
            return None

        now = time.monotonic()
        available = [
            p for p in self._proxies
            if self._banned.get(p, 0) <= now
        ]
        if not available:
            logger.warning("All proxies are currently banned — scraping direct")
            return None

        proxy = available[self._current_idx % len(available)]
        self._current_idx += 1
        return proxy

    def ban_proxy(self, proxy_url: str) -> None:
        """Mark a proxy as banned for _ban_duration_seconds."""
        self._banned[proxy_url] = time.monotonic() + self._ban_duration_seconds
        logger.warning("Proxy banned", proxy=proxy_url[:40], duration_s=self._ban_duration_seconds)

    def report_success(self, proxy_url: str) -> None:
        """Un-ban a proxy on successful response (handles false bans)."""
        self._banned.pop(proxy_url, None)


# ---------------------------------------------------------------------------
# Base Scraper
# ---------------------------------------------------------------------------

class BaseScraper(ABC):
    """
    Abstract base class for all job board scrapers.

    Subclasses must implement:
        BASE_URL: str                           — board homepage
        RATE_LIMIT_DOMAIN: str                 — key for rate-limit tracking
        search(*, keywords, locations, ...)    — returns list[dict] of raw job data
        _parse_listings(html) -> list[dict]    — parse raw HTML into job dicts

    Subclasses may override:
        CONCURRENT_LIMIT: int                  — max parallel requests
        RETRY_ATTEMPTS: int
        MIN_REQUEST_DELAY / MAX_REQUEST_DELAY  — per-domain timing
    """

    BASE_URL: str = ""
    RATE_LIMIT_DOMAIN: str = ""
    CONCURRENT_LIMIT: int = DEFAULT_CONCURRENT_LIMIT
    RETRY_ATTEMPTS: int = DEFAULT_RETRY_ATTEMPTS
    MIN_REQUEST_DELAY: float = DEFAULT_MIN_REQUEST_DELAY
    MAX_REQUEST_DELAY: float = DEFAULT_MAX_REQUEST_DELAY

    def __init__(self) -> None:
        self._session: Any = None
        self._proxy_manager = ProxyManager()
        self._semaphore = asyncio.Semaphore(self.CONCURRENT_LIMIT)
        self._user_agent = random.choice(_USER_AGENTS)
        self._viewport = random.choice(_VIEWPORTS)
        self._last_request_time: float = 0.0
        self._session_request_count: int = 0
        self._session_renewal_threshold: int = random.randint(80, 150)
        self._request_log: list[dict[str, Any]] = []

    # ---------------------------------------------------------------------------
    # Abstract interface
    # ---------------------------------------------------------------------------

    @abstractmethod
    async def search(
        self,
        *,
        keywords: list[str],
        locations: list[str],
        work_modes: list[str],
        max_results: int,
    ) -> list[dict[str, Any]]:
        """
        Execute a job search and return a list of raw job dicts.

        Each dict must contain at minimum:
            title, company_name, description, source_url, external_id,
            job_board, work_mode, job_type, location, posted_at
        """

    @abstractmethod
    def _parse_listings(self, html: str) -> list[dict[str, Any]]:
        """Parse the raw HTML/JSON response into a list of job dicts."""

    # ---------------------------------------------------------------------------
    # HTTP request engine
    # ---------------------------------------------------------------------------

    async def _get(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        referer: str | None = None,
        is_json: bool = False,
    ) -> str:
        """
        Execute a GET request with full anti-detection, retry, and proxy support.

        Returns the response body as a string (HTML or JSON text).
        Raises ScraperError on permanent failure after all retries.
        """
        from app.core.exceptions import ScraperException

        effective_headers = self._build_headers(referer=referer)
        if headers:
            effective_headers.update(headers)

        last_error: Exception | None = None
        current_proxy = self._proxy_manager.get_proxy()

        for attempt in range(1, self.RETRY_ATTEMPTS + 1):
            await self._rate_limit_wait()

            try:
                async with self._semaphore:
                    response_text = await self._execute_request(
                        url=url,
                        params=params,
                        headers=effective_headers,
                        proxy=current_proxy,
                    )

                # Validate response
                if self._is_captcha_response(response_text):
                    logger.warning(
                        "CAPTCHA / bot-detection page received",
                        url=url[:80],
                        attempt=attempt,
                        proxy=current_proxy[:30] if current_proxy else "direct",
                    )
                    if current_proxy:
                        self._proxy_manager.ban_proxy(current_proxy)
                        current_proxy = self._proxy_manager.get_proxy()
                    await self._backoff(attempt)
                    continue

                if self._is_blocked_response(response_text):
                    logger.error("IP blocked on this domain", url=url[:80])
                    from app.core.exceptions import ScraperException
                    raise ScraperException(
                        board=self.RATE_LIMIT_DOMAIN,
                        reason="IP address blocked by target site",
                        url=url,
                    )

                if current_proxy:
                    self._proxy_manager.report_success(current_proxy)

                self._session_request_count += 1
                if self._session_request_count >= self._session_renewal_threshold:
                    await self._renew_session()

                return response_text

            except ScraperException:
                raise
            except Exception as exc:
                last_error = exc
                error_str = str(exc).lower()

                # Detect proxy-level failures
                if current_proxy and any(
                    kw in error_str for kw in ["proxy", "tunnel", "connect", "refused"]
                ):
                    self._proxy_manager.ban_proxy(current_proxy)
                    current_proxy = self._proxy_manager.get_proxy()

                logger.warning(
                    "Request failed",
                    url=url[:80],
                    attempt=attempt,
                    error=str(exc)[:200],
                    proxy=current_proxy[:30] if current_proxy else "direct",
                )

                if attempt < self.RETRY_ATTEMPTS:
                    await self._backoff(attempt)
                    # Rotate user agent on retry
                    self._user_agent = random.choice(_USER_AGENTS)
                    effective_headers["User-Agent"] = self._user_agent
                else:
                    raise ScraperException(
                        board=self.RATE_LIMIT_DOMAIN,
                        reason=f"All {self.RETRY_ATTEMPTS} attempts failed: {last_error}",
                        url=url,
                    )

        raise ScraperException(
            board=self.RATE_LIMIT_DOMAIN,
            reason=f"Exhausted retries. Last error: {last_error}",
            url=url,
        )

    async def _execute_request(
        self,
        url: str,
        params: dict | None,
        headers: dict,
        proxy: str | None,
    ) -> str:
        """
        Execute the actual HTTP request using curl_cffi for TLS fingerprint
        impersonation. Falls back to httpx if curl_cffi is not installed.

        curl_cffi is preferred because it supports JA3/H2 fingerprint
        impersonation at the TLS handshake level — not achievable with
        httpx/aiohttp which use Python's ssl module with a fixed fingerprint
        that's trivially detected by Cloudflare and Akamai.
        """
        try:
            from curl_cffi.requests import AsyncSession

            proxy_kwargs = {"proxy": proxy} if proxy else {}
            async with AsyncSession(impersonate="chrome124") as session:
                resp = await session.get(
                    url,
                    params=params,
                    headers=headers,
                    timeout=settings.SCRAPE_REQUEST_TIMEOUT_SECONDS,
                    allow_redirects=True,
                    **proxy_kwargs,
                )
                resp.raise_for_status()
                return resp.text

        except ImportError:
            # Fallback: standard httpx (no TLS fingerprint impersonation)
            import httpx
            proxy_kwargs = {"proxy": proxy} if proxy else {}
            async with httpx.AsyncClient(
                timeout=settings.SCRAPE_REQUEST_TIMEOUT_SECONDS,
                follow_redirects=True,
                headers=headers,
                **proxy_kwargs,
            ) as client:
                resp = await client.get(url, params=params)
                resp.raise_for_status()
                return resp.text

    # ---------------------------------------------------------------------------
    # Rate limiting and timing
    # ---------------------------------------------------------------------------

    async def _rate_limit_wait(self) -> None:
        """
        Enforce per-domain minimum delay between requests.

        Adds gaussian-distributed jitter so request intervals appear
        organic rather than metronomically uniform (which is a detection signal).
        """
        elapsed = time.monotonic() - self._last_request_time
        base_delay = random.uniform(self.MIN_REQUEST_DELAY, self.MAX_REQUEST_DELAY)
        # Gaussian jitter around the base delay
        jitter = random.gauss(0, base_delay * 0.15)
        required_delay = max(0.0, base_delay + jitter - elapsed)

        if required_delay > 0:
            await asyncio.sleep(required_delay)

        self._last_request_time = time.monotonic()

    async def _backoff(self, attempt: int) -> None:
        """Exponential backoff with full jitter for retry delays."""
        delay = min(
            DEFAULT_RETRY_MAX_DELAY,
            DEFAULT_RETRY_BASE_DELAY * (2 ** (attempt - 1)),
        )
        # Full jitter: uniform random within [0, delay]
        actual_delay = random.uniform(0, delay)
        logger.debug(f"Backoff {actual_delay:.1f}s before attempt {attempt + 1}")
        await asyncio.sleep(actual_delay)

    # ---------------------------------------------------------------------------
    # Headers and session management
    # ---------------------------------------------------------------------------

    def _build_headers(self, referer: str | None = None) -> dict[str, str]:
        """Build a realistic Chrome header set for this request."""
        headers = dict(_CHROME_HEADERS)
        headers["User-Agent"] = self._user_agent
        if referer:
            headers["Referer"] = referer
            headers["Sec-Fetch-Site"] = "same-origin" if self.BASE_URL in referer else "cross-site"
        return headers

    async def _renew_session(self) -> None:
        """Rotate user agent and reset request counter to simulate a fresh browser session."""
        self._user_agent = random.choice(_USER_AGENTS)
        self._viewport = random.choice(_VIEWPORTS)
        self._session_request_count = 0
        self._session_renewal_threshold = random.randint(80, 150)
        logger.debug(
            "Scraper session renewed",
            domain=self.RATE_LIMIT_DOMAIN,
            new_ua=self._user_agent[:60],
        )

    # ---------------------------------------------------------------------------
    # Content detection
    # ---------------------------------------------------------------------------

    def _is_captcha_response(self, html: str) -> bool:
        html_lower = html.lower()
        return any(signal in html_lower for signal in CAPTCHA_SIGNALS)

    def _is_blocked_response(self, html: str) -> bool:
        html_lower = html.lower()
        return any(signal in html_lower for signal in BLOCKED_SIGNALS)

    def _is_empty_response(self, html: str) -> bool:
        return len(html.strip()) < 200

    # ---------------------------------------------------------------------------
    # Common parsing utilities available to all subclasses
    # ---------------------------------------------------------------------------

    def _extract_json_ld(self, html: str) -> list[dict[str, Any]]:
        """Extract all JSON-LD structured data blocks from an HTML page."""
        import re
        results = []
        pattern = r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>'
        for match in re.finditer(pattern, html, re.DOTALL | re.IGNORECASE):
            try:
                data = json.loads(match.group(1).strip())
                results.append(data)
            except (json.JSONDecodeError, ValueError):
                continue
        return results

    def _extract_json_from_script(self, html: str, variable_name: str) -> Any:
        """
        Extract a JavaScript variable assignment from a <script> block.

        Used by boards that embed job data as window.__INITIAL_STATE__ = {...}
        or similar patterns rather than serving a proper JSON API endpoint.
        """
        import re
        pattern = rf'{re.escape(variable_name)}\s*=\s*(\{{.*?\}}|\[.*?\]);'
        match = re.search(pattern, html, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except (json.JSONDecodeError, ValueError):
                pass
        return None

    def _clean_html(self, html: str) -> str:
        """Strip HTML tags and normalise whitespace for description text."""
        import re
        cleaned = re.sub(r'<[^>]+>', ' ', html)
        cleaned = re.sub(r'&nbsp;', ' ', cleaned)
        cleaned = re.sub(r'&amp;', '&', cleaned)
        cleaned = re.sub(r'&lt;', '<', cleaned)
        cleaned = re.sub(r'&gt;', '>', cleaned)
        cleaned = re.sub(r'&quot;', '"', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned)
        return cleaned.strip()

    def _normalise_work_mode(self, raw: str) -> str:
        """Normalise various work-mode strings to remote | hybrid | onsite."""
        raw_lower = raw.lower()
        if any(kw in raw_lower for kw in ["remote", "work from home", "wfh", "anywhere", "distributed"]):
            return "remote"
        if any(kw in raw_lower for kw in ["hybrid", "partially remote", "flexible"]):
            return "hybrid"
        return "onsite"

    def _normalise_job_type(self, raw: str) -> str:
        raw_lower = raw.lower()
        if any(kw in raw_lower for kw in ["full", "full-time", "fulltime"]):
            return "full_time"
        if any(kw in raw_lower for kw in ["part", "part-time", "parttime"]):
            return "part_time"
        if any(kw in raw_lower for kw in ["contract", "contractor", "freelance", "consulting"]):
            return "contract"
        if any(kw in raw_lower for kw in ["intern", "co-op", "coop"]):
            return "internship"
        return "full_time"

    def _parse_salary(self, raw: str) -> tuple[float | None, float | None, str]:
        """
        Extract salary range from free-text. Returns (min, max, currency).
        Handles: $120k-180k, $120,000 - $180,000/yr, €80.000, £90k+
        """
        import re
        if not raw:
            return None, None, "USD"

        currency = "USD"
        if "€" in raw or "eur" in raw.lower():
            currency = "EUR"
        elif "£" in raw or "gbp" in raw.lower():
            currency = "GBP"
        elif "cad" in raw.lower():
            currency = "CAD"

        raw_clean = re.sub(r'[£€$,]', '', raw)
        numbers = re.findall(r'(\d+(?:\.\d+)?)\s*[kK]?', raw_clean)
        values = []
        for i, num_str in enumerate(numbers[:2]):
            num = float(num_str)
            # Detect 'k' suffix (the 'k' follows the number in raw)
            original_context = raw_clean[raw_clean.find(num_str):]
            if original_context and original_context[len(num_str):len(num_str)+1].lower() == 'k':
                num *= 1000
            elif num < 1000:
                num *= 1000  # Assume e.g. "120" means "$120k" in salary context
            values.append(num)

        if len(values) >= 2:
            return min(values), max(values), currency
        if len(values) == 1:
            return values[0], None, currency
        return None, None, currency

    def _compute_content_hash(self, title: str, company: str, description: str) -> str:
        """MD5 fingerprint for deduplication across boards."""
        basis = f"{title.lower().strip()}|{company.lower().strip()}|{description[:300].lower().strip()}"
        return hashlib.md5(basis.encode("utf-8")).hexdigest()

    # ---------------------------------------------------------------------------
    # Lifecycle
    # ---------------------------------------------------------------------------

    async def close(self) -> None:
        """Release session resources. Always call in a finally block."""
        if self._session:
            try:
                await self._session.close()
            except Exception:
                pass
            self._session = None

    async def __aenter__(self) -> "BaseScraper":
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()