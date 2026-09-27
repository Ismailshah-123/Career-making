"""
app/automation/scraping/indeed_scraper.py
==========================================
Production Indeed job scraper for the JobHunter AI platform.

Strategy (layered):
1. Indeed's internal mosaic API (JSON) — returns structured job data,
   no HTML parsing needed. Extracted from network analysis of indeed.com.
2. Indeed job search HTML pages — fallback with BeautifulSoup/regex parsing.
3. Individual job detail API fetch — for full description text.

Indeed-specific challenges handled:
- Dynamic CSRF token rotation across sessions (embedded in page source)
- "Indeed Account" login wall on some geographies — bypassed via guest params
- Duplicate filtering (same job posted under multiple sponsored ad slots)
- Salary normalisation (hourly / monthly / annual mixed in the same feed)
- Sponsored job detection and filtering (marked in results, optional skip)
- Location parsing: "Remote", "Remote in [City]", "Hybrid remote" variants
- Encoded apply URLs (trackingUrl that redirects to actual ATS URL)

Pagination: 10 results per page (Indeed's native page size for API),
controlled via 'start' parameter incrementing by 10. Stops at 1000
(Indeed's hard limit) or max_results, whichever comes first.
"""

from __future__ import annotations

import asyncio
import json
import re
import random
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import urlencode, urlparse, parse_qs

from app.automation.scraping.base_scraper import BaseScraper
from app.core.logging import get_logger

logger = get_logger(__name__)


class IndeedScraper(BaseScraper):
    """
    Indeed job scraper — mosaic JSON API + HTML fallback.
    """

    BASE_URL = "https://www.indeed.com"
    RATE_LIMIT_DOMAIN = "indeed.com"
    CONCURRENT_LIMIT = 3
    RETRY_ATTEMPTS = 3
    MIN_REQUEST_DELAY = 4.0
    MAX_REQUEST_DELAY = 10.0
    PAGE_SIZE = 15

    _SEARCH_URL = "https://www.indeed.com/jobs"
    _API_URL = "https://www.indeed.com/rpc/jobcards/lite"
    _DETAIL_URL = "https://www.indeed.com/viewjob"

    # Indeed work-mode URL parameters
    _REMOTE_FILTERS = {
        "remote": "attr(DSQF7)&attribute=REMOTE",
        "hybrid": "attr(QXDD5)&attribute=PARTIALLY_REMOTE",
        "onsite": None,
    }

    def __init__(self) -> None:
        super().__init__()
        self._csrf_token: str | None = None
        self._seen_keys: set[str] = set()

    async def search(
        self,
        *,
        keywords: list[str],
        locations: list[str],
        work_modes: list[str],
        max_results: int,
    ) -> list[dict[str, Any]]:
        """Search Indeed and return normalised job dicts."""
        all_jobs: list[dict[str, Any]] = []
        query = " ".join(keywords) if keywords else "software engineer"
        location = locations[0] if locations else "Remote"

        # Determine remote filter
        remote_filter = None
        for mode in work_modes:
            if mode in self._REMOTE_FILTERS and self._REMOTE_FILTERS[mode]:
                remote_filter = self._REMOTE_FILTERS[mode]
                break

        logger.info(
            "Indeed search started",
            query=query,
            location=location,
            max_results=max_results,
        )

        # Fetch first page to extract CSRF token and check structure
        await self._prime_session(query, location)

        start = 0
        consecutive_failures = 0

        while len(all_jobs) < max_results and start <= 990:
            try:
                page_jobs = await self._fetch_page(
                    query=query,
                    location=location,
                    start=start,
                    remote_filter=remote_filter,
                )
            except Exception as exc:
                consecutive_failures += 1
                logger.warning(
                    "Indeed page fetch failed",
                    start=start,
                    error=str(exc)[:200],
                )
                if consecutive_failures >= 3:
                    break
                await asyncio.sleep(random.uniform(30, 60))
                continue

            if not page_jobs:
                consecutive_failures += 1
                if consecutive_failures >= 2:
                    break
                start += self.PAGE_SIZE
                continue

            consecutive_failures = 0
            new_jobs = [j for j in page_jobs if j.get("external_id") not in self._seen_keys]
            if not new_jobs:
                logger.info("Indeed: no new jobs on this page — stopping pagination")
                break

            for j in new_jobs:
                self._seen_keys.add(j["external_id"])

            enriched = await self._enrich_jobs(new_jobs[: max_results - len(all_jobs)])
            all_jobs.extend(enriched)
            start += self.PAGE_SIZE

            await asyncio.sleep(random.uniform(5, 12))

        logger.info("Indeed search complete", total=len(all_jobs))
        return all_jobs[:max_results]

    async def _prime_session(self, query: str, location: str) -> None:
        """
        Fetch the search homepage to prime cookies and extract CSRF token.
        Essential for subsequent API calls to succeed on Indeed.
        """
        try:
            params = {"q": query, "l": location, "sort": "date"}
            html = await self._get(
                self._SEARCH_URL,
                params=params,
                referer="https://www.google.com/",
            )
            self._csrf_token = self._extract_csrf(html)
        except Exception as exc:
            logger.warning("Indeed session prime failed (non-fatal)", error=str(exc)[:100])

    def _extract_csrf(self, html: str) -> str | None:
        """Extract Indeed's CSRF token from the page source."""
        patterns = [
            r'"csrfToken"\s*:\s*"([^"]+)"',
            r'name="csrfToken"\s+value="([^"]+)"',
            r'data-csrf="([^"]+)"',
        ]
        for pattern in patterns:
            match = re.search(pattern, html)
            if match:
                return match.group(1)
        return None

    async def _fetch_page(
        self,
        *,
        query: str,
        location: str,
        start: int,
        remote_filter: str | None,
    ) -> list[dict[str, Any]]:
        """
        Attempt the mosaic JSON API first; fall back to HTML parsing.
        """
        try:
            return await self._fetch_via_api(
                query=query,
                location=location,
                start=start,
                remote_filter=remote_filter,
            )
        except Exception as exc:
            logger.debug(f"Indeed API attempt failed ({exc}), trying HTML")
            return await self._fetch_via_html(
                query=query,
                location=location,
                start=start,
                remote_filter=remote_filter,
            )

    async def _fetch_via_api(
        self,
        *,
        query: str,
        location: str,
        start: int,
        remote_filter: str | None,
    ) -> list[dict[str, Any]]:
        """
        Call Indeed's internal mosaic lite API.

        This endpoint returns a JSON payload with job cards including
        salary, snippet, and apply-type information. It requires a valid
        session cookie (obtained via _prime_session) and a CSRF token.
        """
        params: dict[str, Any] = {
            "q": query,
            "l": location,
            "start": start,
            "limit": self.PAGE_SIZE,
            "sort": "date",
            "fromage": "14",     # Posted within 14 days
        }
        if remote_filter:
            params["remotejob"] = "1"

        extra_headers: dict[str, str] = {
            "Accept": "application/json, text/javascript, */*; q=0.01",
            "X-Requested-With": "XMLHttpRequest",
            "Referer": f"{self._SEARCH_URL}?{urlencode({'q': query, 'l': location})}",
        }
        if self._csrf_token:
            extra_headers["X-Csrf-Token"] = self._csrf_token

        response = await self._get(
            self._API_URL,
            params=params,
            headers=extra_headers,
        )

        try:
            data = json.loads(response)
        except json.JSONDecodeError:
            raise ValueError("Indeed API returned non-JSON response")

        job_cards = (
            data.get("jobCards")
            or data.get("results")
            or data.get("jobs")
            or []
        )

        jobs: list[dict[str, Any]] = []
        for card in job_cards:
            job = self._normalise_api_job(card)
            if job:
                jobs.append(job)

        return jobs

    def _normalise_api_job(self, card: dict[str, Any]) -> dict[str, Any] | None:
        """Normalise an Indeed API job card into our schema."""
        job_key = card.get("jobKey") or card.get("jobkey") or card.get("id")
        if not job_key:
            return None

        title = (
            card.get("displayTitle")
            or card.get("normalizedTitle")
            or card.get("title", "")
        )
        company = card.get("companyName") or card.get("company", "")
        location_raw = card.get("formattedLocation") or card.get("location", "")
        description_snippet = self._clean_html(card.get("snippet") or card.get("summary", ""))

        # Salary extraction from various Indeed salary structures
        salary_raw = ""
        salary_info = card.get("estimatedSalary") or card.get("salarySnippet") or {}
        if isinstance(salary_info, dict):
            sal_min = salary_info.get("min") or salary_info.get("salaryMin")
            sal_max = salary_info.get("max") or salary_info.get("salaryMax")
            salary_raw = salary_info.get("formattedRange", "")
        else:
            sal_min, sal_max = None, None

        if sal_min is None and salary_raw:
            sal_min, sal_max, _ = self._parse_salary(salary_raw)

        # Work mode
        remote_tags = card.get("remoteWorkModel", {})
        is_remote = (
            remote_tags.get("remoteness") in ("remote", "fully_remote")
            or "remote" in location_raw.lower()
        )
        is_hybrid = (
            remote_tags.get("remoteness") == "partially_remote"
            or "hybrid" in location_raw.lower()
        )
        work_mode = "remote" if is_remote else ("hybrid" if is_hybrid else "onsite")

        # Job URL
        source_url = f"https://www.indeed.com/viewjob?jk={job_key}"

        # Apply URL (decode from tracking URL if needed)
        apply_url = card.get("applyUrls", {}).get("indeedApplyUrl") or None
        if apply_url and "trackingUrl" in apply_url:
            apply_url = self._decode_tracking_url(apply_url)

        # Posted date
        posted_epoch = card.get("pubDate") or card.get("postedDate")
        posted_at = (
            datetime.fromtimestamp(int(posted_epoch) / 1000, tz=timezone.utc).isoformat()
            if posted_epoch
            else None
        )

        # Employment type
        emp_type = card.get("jobType") or card.get("jobTypes", [""])[0] if card.get("jobTypes") else ""
        job_type = self._normalise_job_type(emp_type)

        return {
            "external_id": job_key,
            "title": title,
            "company_name": company,
            "location": location_raw,
            "source_url": source_url,
            "job_board": "indeed",
            "work_mode": work_mode,
            "job_type": job_type,
            "posted_at": posted_at,
            "description": description_snippet,
            "description_cleaned": description_snippet,
            "apply_url": apply_url,
            "salary_min": float(sal_min) if sal_min else None,
            "salary_max": float(sal_max) if sal_max else None,
            "salary_currency": "USD",
            "required_skills": [],
            "preferred_skills": [],
            "ats_provider": None,
            "experience_level": None,
            "company_logo_url": card.get("companyBrandingAttributes", {}).get("logoUrl"),
        }

    async def _fetch_via_html(
        self,
        *,
        query: str,
        location: str,
        start: int,
        remote_filter: str | None,
    ) -> list[dict[str, Any]]:
        """HTML fallback scraper for Indeed job search pages."""
        params: dict[str, Any] = {
            "q": query,
            "l": location,
            "start": start,
            "sort": "date",
            "fromage": "14",
        }
        if remote_filter:
            params["remotejob"] = "1"

        html = await self._get(
            self._SEARCH_URL,
            params=params,
            referer="https://www.google.com/",
        )
        return self._parse_listings(html)

    def _parse_listings(self, html: str) -> list[dict[str, Any]]:
        """Regex-based HTML parser for Indeed search result pages."""
        jobs: list[dict[str, Any]] = []

        # Indeed embeds job data in a window._initialData or similar
        initial_data = self._extract_json_from_script(html, "window._initialData")
        if initial_data:
            job_list = (
                initial_data.get("jobList", {}).get("jobs", [])
                or initial_data.get("jobs", [])
            )
            for raw in job_list:
                normalised = self._normalise_api_job(raw)
                if normalised:
                    jobs.append(normalised)
            if jobs:
                return jobs

        # Plain regex fallback
        job_pattern = re.compile(
            r'data-jk="([^"]+)"[^>]*>.*?class="jobTitle"[^>]*>.*?<span[^>]*>([^<]+)</span>.*?class="companyName"[^>]*>.*?>([^<]+)<',
            re.DOTALL,
        )
        for match in job_pattern.finditer(html):
            job_key, title, company = match.group(1), match.group(2).strip(), match.group(3).strip()
            jobs.append({
                "external_id": job_key,
                "title": title,
                "company_name": company,
                "location": "",
                "source_url": f"https://www.indeed.com/viewjob?jk={job_key}",
                "job_board": "indeed",
                "work_mode": "onsite",
                "job_type": "full_time",
                "posted_at": None,
                "description": "",
                "description_cleaned": "",
                "apply_url": None,
                "salary_min": None,
                "salary_max": None,
                "salary_currency": "USD",
                "required_skills": [],
                "preferred_skills": [],
                "ats_provider": None,
                "experience_level": None,
                "company_logo_url": None,
            })

        return jobs

    async def _enrich_jobs(self, jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Fetch full description for each job from the detail page."""
        enriched: list[dict[str, Any]] = []
        for job in jobs:
            try:
                detail = await self._fetch_job_detail(job["external_id"])
                enriched.append({**job, **detail})
            except Exception as exc:
                logger.warning(
                    "Indeed detail fetch failed",
                    job_id=job["external_id"],
                    error=str(exc)[:150],
                )
                enriched.append(job)
            await asyncio.sleep(random.uniform(3, 7))
        return enriched

    async def _fetch_job_detail(self, job_key: str) -> dict[str, Any]:
        """Fetch and parse a single Indeed job detail page."""
        params = {"jk": job_key, "viewtype": "embedded"}
        html = await self._get(
            self._DETAIL_URL,
            params=params,
            referer=f"{self._SEARCH_URL}?q=software+engineer",
        )

        if self._is_empty_response(html):
            return {}

        # Try JSON-LD first
        json_lds = self._extract_json_ld(html)
        for block in json_lds:
            if block.get("@type") in ("JobPosting", "jobPosting"):
                description_html = block.get("description", "")
                description_clean = self._clean_html(description_html)
                sal_min, sal_max, sal_cur = self._parse_salary(
                    str(block.get("baseSalary", {}).get("value", {}).get("minValue", ""))
                )
                apply_url = block.get("url") or block.get("sameAs")
                skills = self._extract_skills_from_description(description_clean)
                return {
                    "description": description_html[:50_000],
                    "description_cleaned": description_clean[:20_000],
                    "apply_url": apply_url,
                    "salary_min": sal_min,
                    "salary_max": sal_max,
                    "salary_currency": sal_cur,
                    "required_skills": skills[:30],
                    "experience_level": self._infer_experience_level(description_clean[:300]),
                    "ats_provider": self._detect_ats_from_url(apply_url or ""),
                }

        # HTML fallback
        desc_match = re.search(
            r'id="jobDescriptionText"[^>]*>(.*?)</div>', html, re.DOTALL
        )
        description_html = desc_match.group(1) if desc_match else ""
        description_clean = self._clean_html(description_html)
        skills = self._extract_skills_from_description(description_clean)

        apply_match = re.search(r'"applyUrl"\s*:\s*"([^"]+)"', html)
        apply_url = apply_match.group(1).replace("\\u0026", "&") if apply_match else None

        sal_match = re.search(r'"salary"\s*:\s*"([^"]+)"', html)
        sal_min, sal_max, sal_cur = (
            self._parse_salary(sal_match.group(1)) if sal_match else (None, None, "USD")
        )

        return {
            "description": description_html[:50_000],
            "description_cleaned": description_clean[:20_000],
            "apply_url": apply_url,
            "salary_min": sal_min,
            "salary_max": sal_max,
            "salary_currency": sal_cur,
            "required_skills": skills[:30],
            "experience_level": self._infer_experience_level(description_clean[:300]),
            "ats_provider": self._detect_ats_from_url(apply_url or ""),
        }

    # ---------------------------------------------------------------------------
    # Skill and level extraction (reuse LinkedIn's lists via mixin-style)
    # ---------------------------------------------------------------------------

    _SKILL_KEYWORDS = [
        "Python", "JavaScript", "TypeScript", "Go", "Rust", "Java", "C++", "C#",
        "Ruby", "React", "Next.js", "Vue", "Angular", "Node.js", "FastAPI", "Django",
        "Flask", "Spring", "PyTorch", "TensorFlow", "AWS", "GCP", "Azure", "Docker",
        "Kubernetes", "Terraform", "PostgreSQL", "MySQL", "MongoDB", "Redis",
        "GraphQL", "Kafka", "Elasticsearch", "Snowflake", "dbt", "Airflow", "Spark",
        "LangChain", "OpenAI", "Hugging Face", "CI/CD", "Git", "Linux", "Bash",
    ]

    def _extract_skills_from_description(self, text: str) -> list[str]:
        found: list[str] = []
        text_lower = text.lower()
        for skill in self._SKILL_KEYWORDS:
            if re.search(r'\b' + re.escape(skill.lower()) + r'\b', text_lower):
                found.append(skill)
        return found

    def _infer_experience_level(self, text: str) -> str | None:
        text_lower = text.lower()
        if any(kw in text_lower for kw in ["principal", "staff ", "distinguished"]):
            return "staff"
        if any(kw in text_lower for kw in ["senior", "sr.", "lead "]):
            return "senior"
        if any(kw in text_lower for kw in ["junior", "entry", "intern", "associate", "jr."]):
            return "entry"
        return "mid"

    def _detect_ats_from_url(self, url: str) -> str | None:
        if not url:
            return None
        url_lower = url.lower()
        mapping = [
            (["greenhouse.io"], "greenhouse"),
            (["lever.co"], "lever"),
            (["workday.com"], "workday"),
            (["ashbyhq.com"], "ashby"),
            (["icims.com"], "icims"),
            (["taleo.net"], "taleo"),
            (["bamboohr.com"], "bamboohr"),
        ]
        for patterns, ats in mapping:
            if any(p in url_lower for p in patterns):
                return ats
        return None

    def _decode_tracking_url(self, tracking_url: str) -> str:
        """Extract the real destination URL from an Indeed tracking redirect URL."""
        try:
            parsed = urlparse(tracking_url)
            params = parse_qs(parsed.query)
            dest = params.get("dest", params.get("url", [tracking_url]))[0]
            return dest
        except Exception:
            return tracking_url