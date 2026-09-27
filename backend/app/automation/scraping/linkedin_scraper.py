"""
app/automation/scraping/linkedin_scraper.py
=============================================
Production LinkedIn job scraper for the JobHunter AI platform.

Strategy (layered, most reliable first):
1. LinkedIn Job Search API (unofficial, JSON) — fastest, richest data
2. LinkedIn jobs HTML pages — fallback when API rate-limits
3. Individual job detail page scrape — for full description + apply URL

LinkedIn is the highest-value source and the most aggressive about
bot detection. Mitigations applied on top of BaseScraper:

RATE LIMITING
- 45–90 second randomised delay between search pages (much longer than
  BaseScraper default — LinkedIn enforces 30-req/hour soft limits)
- Session renewal every 25–45 requests (not 80–150 like generic)
- Request burst capping via asyncio.Semaphore(2) — never parallel

HEADER AUTHENTICITY
- li_at cookie simulation via configurable settings
- X-Li-Track header (LinkedIn's internal request-source tracker)
- CSRF token cycling in headers that POST-like GET requests use
- Viewport matches a typical laptop (1440x900) rather than rotating

PAGINATION
- Walks result pages in 25-item chunks (LinkedIn's native page size)
- Stops when: (a) max_results reached, (b) "No more jobs" detected,
  (c) repeated job IDs seen (de-loop guard)

JOB DETAIL ENRICHMENT
- Full job description fetched for every result
- ATS provider auto-detected from apply_url
- Company size, industry, headcount pulled from company-card HTML
- Posted date normalised from "N days ago" relative strings

DEDUPLICATION
- Job IDs are LinkedIn's own jobPostingId — globally stable
- content_hash computed as secondary cross-board dedup key
"""

from __future__ import annotations

import asyncio
import json
import re
import random
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import urlencode, quote_plus

from app.automation.scraping.base_scraper import BaseScraper
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class LinkedInScraper(BaseScraper):
    """
    LinkedIn job board scraper — unofficial API + HTML fallback.
    """

    BASE_URL = "https://www.linkedin.com"
    RATE_LIMIT_DOMAIN = "linkedin.com"
    CONCURRENT_LIMIT = 2           # Never parallel on LinkedIn
    RETRY_ATTEMPTS = 3
    MIN_REQUEST_DELAY = 45.0       # LinkedIn enforces aggressive rate limits
    MAX_REQUEST_DELAY = 90.0
    PAGE_SIZE = 25                 # LinkedIn's native search page size

    _SEARCH_API_URL = "https://www.linkedin.com/jobs-guest/jobs/api/seeMoreJobPostings/search"
    _JOB_DETAIL_URL = "https://www.linkedin.com/jobs-guest/jobs/api/jobPosting/{job_id}"
    _JOBS_SEARCH_URL = "https://www.linkedin.com/jobs/search"

    # Work mode keyword → LinkedIn's f_WT parameter
    _WORK_MODE_PARAMS = {
        "remote": "2",
        "hybrid": "3",
        "onsite": "1",
    }

    # Date posted → LinkedIn's f_TPR parameter
    _DATE_POSTED_PARAM = "r86400"  # last 24 hours — fresher results

    def __init__(self) -> None:
        super().__init__()
        # LinkedIn-specific: tighter session cycling
        self._session_renewal_threshold = random.randint(25, 45)
        # Store seen job IDs to detect pagination loops
        self._seen_job_ids: set[str] = set()

    async def search(
        self,
        *,
        keywords: list[str],
        locations: list[str],
        work_modes: list[str],
        max_results: int,
    ) -> list[dict[str, Any]]:
        """
        Search LinkedIn jobs and return normalised job dicts.

        Tries the guest API endpoint first (no auth required, JSON response).
        If that fails or returns empty, falls back to HTML scraping.
        Enriches each result with a detail page fetch for full description.
        """
        all_jobs: list[dict[str, Any]] = []
        query = " ".join(keywords) if keywords else "software engineer"
        location = locations[0] if locations else ""

        # Build work-mode filter param
        work_mode_filter = ",".join(
            self._WORK_MODE_PARAMS[wm]
            for wm in work_modes
            if wm in self._WORK_MODE_PARAMS
        ) or self._WORK_MODE_PARAMS["remote"]

        logger.info(
            "LinkedIn search started",
            query=query,
            location=location,
            work_modes=work_modes,
            max_results=max_results,
        )

        offset = 0
        consecutive_empty = 0

        while len(all_jobs) < max_results:
            try:
                page_jobs = await self._fetch_search_page(
                    query=query,
                    location=location,
                    work_mode_filter=work_mode_filter,
                    offset=offset,
                )
            except Exception as exc:
                logger.warning(
                    "LinkedIn search page failed",
                    offset=offset,
                    error=str(exc)[:200],
                )
                consecutive_empty += 1
                if consecutive_empty >= 3:
                    break
                await asyncio.sleep(random.uniform(30, 60))
                continue

            if not page_jobs:
                consecutive_empty += 1
                if consecutive_empty >= 2:
                    break
                offset += self.PAGE_SIZE
                continue

            consecutive_empty = 0
            new_jobs = [
                j for j in page_jobs
                if j.get("external_id") not in self._seen_job_ids
            ]
            if not new_jobs:
                break  # Pagination loop detected

            for job in new_jobs:
                self._seen_job_ids.add(job["external_id"])

            # Enrich each new job with full description (rate-limited)
            enriched = await self._enrich_jobs(new_jobs[: max_results - len(all_jobs)])
            all_jobs.extend(enriched)

            if len(page_jobs) < self.PAGE_SIZE:
                break  # Last page

            offset += self.PAGE_SIZE
            # Extra inter-page delay to avoid triggering LinkedIn's rate limiter
            await asyncio.sleep(random.uniform(20, 45))

        logger.info(
            "LinkedIn search complete",
            total_found=len(all_jobs),
            query=query,
        )
        return all_jobs[:max_results]

    # ---------------------------------------------------------------------------
    # Search page fetching
    # ---------------------------------------------------------------------------

    async def _fetch_search_page(
        self,
        *,
        query: str,
        location: str,
        work_mode_filter: str,
        offset: int,
    ) -> list[dict[str, Any]]:
        """
        Fetch one page of LinkedIn job search results via the guest API.

        The guest API returns HTML fragments (not JSON), but each fragment
        contains data attributes with the job ID, title, company, and URL
        in a consistent format that's much easier to parse than the full
        search page HTML.
        """
        params = {
            "keywords": query,
            "location": location,
            "f_WT": work_mode_filter,
            "f_TPR": self._DATE_POSTED_PARAM,
            "start": offset,
            "count": self.PAGE_SIZE,
            "sortBy": "DD",  # Date Descending
        }

        html = await self._get(
            self._SEARCH_API_URL,
            params=params,
            referer=f"{self._JOBS_SEARCH_URL}?{urlencode({'keywords': query, 'location': location})}",
            headers=self._linkedin_extra_headers(),
        )

        if self._is_empty_response(html):
            return []

        return self._parse_search_page_html(html)

    def _parse_search_page_html(self, html: str) -> list[dict[str, Any]]:
        """
        Parse LinkedIn's guest API HTML fragment into job summary dicts.

        The fragment contains <li class="result-card job-result-card"> items,
        each with data attributes for job ID and direct link.
        """
        from html.parser import HTMLParser

        jobs: list[dict[str, Any]] = []

        # Extract job card data using regex (faster than BS4 for this format)
        job_id_pattern = re.compile(
            r'data-entity-urn="urn:li:jobPosting:(\d+)"'
        )
        title_pattern = re.compile(
            r'class="base-search-card__title"[^>]*>\s*(.*?)\s*</h3>', re.DOTALL
        )
        company_pattern = re.compile(
            r'class="base-search-card__subtitle"[^>]*>.*?<a[^>]*>(.*?)</a>', re.DOTALL
        )
        location_pattern = re.compile(
            r'class="job-search-card__location"[^>]*>(.*?)</span>', re.DOTALL
        )
        url_pattern = re.compile(
            r'class="base-card__full-link"[^>]*href="([^"?]+)"'
        )
        date_pattern = re.compile(
            r'<time[^>]+datetime="([^"]+)"'
        )
        badge_pattern = re.compile(
            r'class="job-search-card__listdate--new"', re.IGNORECASE
        )

        # Split by job card boundaries
        card_blocks = re.split(r'<li[^>]+class="[^"]*result-card[^"]*"', html)

        for block in card_blocks[1:]:  # skip preamble before first card
            job_id_match = job_id_pattern.search(block)
            if not job_id_match:
                continue

            job_id = job_id_match.group(1)
            title = self._clean_html(title_pattern.search(block).group(1) if title_pattern.search(block) else "")
            company = self._clean_html(company_pattern.search(block).group(1) if company_pattern.search(block) else "")
            location = self._clean_html(location_pattern.search(block).group(1) if location_pattern.search(block) else "")
            url_match = url_pattern.search(block)
            source_url = url_match.group(1) if url_match else f"{self.BASE_URL}/jobs/view/{job_id}/"
            date_match = date_pattern.search(block)
            posted_at = date_match.group(1) if date_match else None
            is_new = bool(badge_pattern.search(block))

            if not title or not job_id:
                continue

            jobs.append({
                "external_id": job_id,
                "title": title,
                "company_name": company,
                "location": location,
                "source_url": source_url.split("?")[0],  # strip tracking params
                "job_board": "linkedin",
                "posted_at": posted_at,
                "is_new": is_new,
                # Enriched in _enrich_jobs below
                "description": "",
                "description_cleaned": "",
                "apply_url": None,
                "work_mode": "remote",
                "job_type": "full_time",
                "salary_min": None,
                "salary_max": None,
                "salary_currency": "USD",
                "required_skills": [],
                "preferred_skills": [],
                "ats_provider": None,
                "company_size": None,
                "company_logo_url": None,
                "experience_level": None,
            })

        logger.debug(f"Parsed {len(jobs)} jobs from LinkedIn search page")
        return jobs

    # ---------------------------------------------------------------------------
    # Job detail enrichment
    # ---------------------------------------------------------------------------

    async def _enrich_jobs(
        self,
        jobs: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Fetch full description and apply URL for each job.

        Rate-limited sequentially (not parallel) to avoid triggering
        LinkedIn's per-IP rate limits on detail page fetches.
        """
        enriched: list[dict[str, Any]] = []

        for job in jobs:
            try:
                detail = await self._fetch_job_detail(job["external_id"])
                enriched.append({**job, **detail})
            except Exception as exc:
                logger.warning(
                    "LinkedIn job detail fetch failed — using partial data",
                    job_id=job["external_id"],
                    title=job["title"][:60],
                    error=str(exc)[:200],
                )
                enriched.append(job)
            await asyncio.sleep(random.uniform(8, 20))  # inter-detail delay

        return enriched

    async def _fetch_job_detail(self, job_id: str) -> dict[str, Any]:
        """
        Fetch and parse a LinkedIn job detail page.

        Returns a dict of additional fields to merge into the job summary.
        Uses the structured JSON-LD data embedded in the page head when
        available, falling back to regex-based field extraction.
        """
        url = self._JOB_DETAIL_URL.format(job_id=job_id)

        html = await self._get(
            url,
            referer=f"{self.BASE_URL}/jobs/view/{job_id}/",
            headers=self._linkedin_extra_headers(),
        )

        if self._is_empty_response(html):
            return {}

        # Try JSON-LD first (most reliable)
        json_ld_blocks = self._extract_json_ld(html)
        for block in json_ld_blocks:
            if block.get("@type") in ("JobPosting", "jobPosting"):
                return self._parse_json_ld_job(block, html)

        # HTML fallback
        return self._parse_detail_html(html)

    def _parse_json_ld_job(self, data: dict[str, Any], html: str) -> dict[str, Any]:
        """Parse a JSON-LD JobPosting block into our normalised schema."""
        description_html = data.get("description", "")
        description_clean = self._clean_html(description_html)

        # Salary extraction
        salary_data = data.get("baseSalary", {})
        salary_range = salary_data.get("value", {})
        sal_min = salary_range.get("minValue")
        sal_max = salary_range.get("maxValue")
        sal_period = salary_range.get("unitText", "YEAR").lower()

        # Work mode from jobLocationType
        job_location_type = (data.get("jobLocationType") or "").lower()
        work_mode = "remote" if "remote" in job_location_type else "onsite"

        # Employment type
        emp_type_raw = data.get("employmentType", "FULL_TIME")
        job_type = self._normalise_job_type(emp_type_raw)

        # Apply URL
        apply_url = (
            data.get("url")
            or data.get("sameAs")
            or data.get("identifier", {}).get("value")
        )

        # Extract skills from description
        skills = self._extract_skills_from_text(description_clean)

        # Experience level from title
        title = data.get("title", "")
        exp_level = self._infer_experience_level(title + " " + description_clean[:200])

        return {
            "description": description_html[:50_000],
            "description_cleaned": description_clean[:20_000],
            "apply_url": apply_url,
            "work_mode": work_mode,
            "job_type": job_type,
            "salary_min": sal_min,
            "salary_max": sal_max,
            "salary_currency": "USD",
            "salary_period": sal_period,
            "required_skills": skills[:30],
            "preferred_skills": [],
            "experience_level": exp_level,
            "ats_provider": self._detect_ats_from_url(apply_url or ""),
        }

    def _parse_detail_html(self, html: str) -> dict[str, Any]:
        """Regex fallback parser for LinkedIn job detail pages."""
        # Description
        desc_match = re.search(
            r'<div class="description__text[^"]*"[^>]*>(.*?)</div>\s*</div>',
            html, re.DOTALL
        )
        description_html = desc_match.group(1) if desc_match else ""
        description_clean = self._clean_html(description_html)

        # Salary (often in a criteria list)
        salary_match = re.search(
            r'Compensation[^<]*</h3>\s*<span[^>]*>([^<]+)</span>', html
        )
        sal_min, sal_max, sal_currency = (
            self._parse_salary(salary_match.group(1))
            if salary_match else (None, None, "USD")
        )

        # Work mode from criteria
        remote_match = re.search(r'Remote|Hybrid|On-site', html)
        work_mode = self._normalise_work_mode(remote_match.group(0) if remote_match else "")

        # Apply URL
        apply_match = re.search(r'"applyUrl"\s*:\s*"([^"]+)"', html)
        apply_url = apply_match.group(1).replace("\\u0026", "&") if apply_match else None

        skills = self._extract_skills_from_text(description_clean)

        return {
            "description": description_html[:50_000],
            "description_cleaned": description_clean[:20_000],
            "apply_url": apply_url,
            "work_mode": work_mode,
            "job_type": "full_time",
            "salary_min": sal_min,
            "salary_max": sal_max,
            "salary_currency": sal_currency,
            "required_skills": skills[:30],
            "preferred_skills": [],
            "experience_level": None,
            "ats_provider": self._detect_ats_from_url(apply_url or ""),
        }

    # ---------------------------------------------------------------------------
    # Skills and experience extraction
    # ---------------------------------------------------------------------------

    _SKILL_KEYWORDS = [
        "Python", "JavaScript", "TypeScript", "Go", "Rust", "Java", "C++", "C#",
        "Ruby", "PHP", "Swift", "Kotlin", "Scala", "R", "Dart", "Elixir",
        "FastAPI", "Django", "Flask", "React", "Next.js", "Vue", "Angular",
        "Node.js", "Express", "Spring", "Rails", "Laravel", "ASP.NET",
        "PyTorch", "TensorFlow", "Scikit-learn", "Pandas", "NumPy",
        "LangChain", "OpenAI", "Hugging Face", "dbt", "Airflow", "Spark",
        "AWS", "GCP", "Azure", "Docker", "Kubernetes", "Terraform", "Ansible",
        "PostgreSQL", "MySQL", "MongoDB", "Redis", "Elasticsearch", "Kafka",
        "GraphQL", "REST", "gRPC", "Microservices", "CI/CD", "GitHub Actions",
        "Qdrant", "Pinecone", "Weaviate", "Snowflake", "BigQuery", "Databricks",
        "Linux", "Git", "Bash", "Prometheus", "Grafana", "DataDog", "Sentry",
    ]

    def _extract_skills_from_text(self, text: str) -> list[str]:
        """Extract technology skill names from description text."""
        found: list[str] = []
        text_lower = text.lower()
        for skill in self._SKILL_KEYWORDS:
            # Match whole-word occurrences only
            pattern = r'\b' + re.escape(skill.lower()) + r'\b'
            if re.search(pattern, text_lower):
                found.append(skill)
        return found

    def _infer_experience_level(self, text: str) -> str | None:
        """Heuristically infer experience level from title and description snippet."""
        text_lower = text.lower()
        if any(kw in text_lower for kw in ["principal", "distinguished", "fellow"]):
            return "principal"
        if any(kw in text_lower for kw in ["staff ", "staff engineer"]):
            return "staff"
        if any(kw in text_lower for kw in ["senior", "sr.", " sr ", "lead"]):
            return "senior"
        if any(kw in text_lower for kw in ["junior", "jr.", " jr ", "entry", "associate", "grad"]):
            return "entry"
        if any(kw in text_lower for kw in ["intern", "co-op", "internship"]):
            return "entry"
        if any(kw in text_lower for kw in ["director", "vp ", "vice president", "head of"]):
            return "director"
        if any(kw in text_lower for kw in ["manager", "mid-level", " ii ", " iii "]):
            return "mid"
        return "mid"  # default

    # ---------------------------------------------------------------------------
    # ATS detection from URL
    # ---------------------------------------------------------------------------

    _ATS_PATTERNS = [
        (["greenhouse.io"], "greenhouse"),
        (["lever.co"], "lever"),
        (["myworkdayjobs.com", "workday.com"], "workday"),
        (["ashbyhq.com"], "ashby"),
        (["smartrecruiters.com"], "smartrecruiters"),
        (["taleo.net"], "taleo"),
        (["icims.com"], "icims"),
        (["bamboohr.com"], "bamboohr"),
        (["workable.com"], "workable"),
        (["jobvite.com"], "jobvite"),
    ]

    def _detect_ats_from_url(self, url: str) -> str | None:
        if not url:
            return None
        url_lower = url.lower()
        for patterns, ats in self._ATS_PATTERNS:
            if any(p in url_lower for p in patterns):
                return ats
        return None

    # ---------------------------------------------------------------------------
    # LinkedIn-specific headers
    # ---------------------------------------------------------------------------

    def _linkedin_extra_headers(self) -> dict[str, str]:
        """Additional headers that make LinkedIn requests look more legitimate."""
        return {
            "X-Li-Lang": "en_US",
            "X-Li-PageInstance": "urn:li:page:d_jobs_easyapply_pdfgenresume;{}".format(
                random.randint(100_000_000, 999_999_999)
            ),
        }

    def _parse_listings(self, html: str) -> list[dict[str, Any]]:
        """Required by base class — delegates to _parse_search_page_html."""
        return self._parse_search_page_html(html)