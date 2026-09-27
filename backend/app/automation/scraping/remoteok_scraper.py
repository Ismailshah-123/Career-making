"""
app/automation/scraping/remoteok_scraper.py
=============================================
Production RemoteOK job scraper for the JobHunter AI platform.

RemoteOK is the simplest board to scrape — it provides a public JSON API
at remoteok.io/remote-jobs.json that returns all active listings with full
structured data, no authentication required.

However, simplicity doesn't mean easy in production:

RATE LIMITING
- RemoteOK has an undocumented 1-request/minute soft limit on their API.
  Exceeding it returns a Cloudflare 429 with a 5-minute block.
- We respect this with a minimum 65-second delay between requests.
- The full listing is returned in one call (no pagination) so we only
  need one API call per search run — we filter client-side.

CLOUDFLARE PROTECTION
- RemoteOK sits behind Cloudflare, which uses JA3 fingerprinting.
- curl_cffi impersonation is critical here (see BaseScraper._execute_request).
- If curl_cffi is absent, many requests get 403'd mid-session.
- We also send Cf-Visitor and Cf-Clearance headers to look legitimate.

DATA QUALITY
- Tags are RemoteOK's native taxonomy (e.g. "Python", "React", "DevOps")
  and map directly to our required_skills field — no NLP extraction needed.
- Salary is often provided as {min, max} integers in USD — highest quality
  salary data of any board we scrape.
- Company logos are hosted on remoteok.io CDN — always accessible.
- Location is always "Worldwide" / "100% Remote" (it's a remote-only board).
- apply_url points directly to the company's ATS — no redirect chain.

FILTERING
- Since the full listing is returned in one JSON call, all keyword/location/
  work-mode filtering happens client-side after parsing — we download once
  and filter, rather than making N parameterised search requests.
- Work mode is always "remote" for every listing (it's a remote-only board).
- keyword matching is done against title + tags (not full description) for
  performance — descriptions can be 50KB per job.

DEDUPLICATION
- RemoteOK provides a stable integer 'id' field — used as external_id.
- content_hash is still computed for cross-board deduplication (same job
  sometimes appears on both RemoteOK and LinkedIn).
"""

from __future__ import annotations

import asyncio
import json
import re
import random
from datetime import datetime, timezone
from typing import Any

from app.automation.scraping.base_scraper import BaseScraper
from app.core.logging import get_logger

logger = get_logger(__name__)


class RemoteOKScraper(BaseScraper):
    """
    RemoteOK job scraper — single public JSON API endpoint.
    """

    BASE_URL = "https://remoteok.io"
    RATE_LIMIT_DOMAIN = "remoteok.io"
    CONCURRENT_LIMIT = 1              # One request at a time — strict rate limit
    RETRY_ATTEMPTS = 3
    MIN_REQUEST_DELAY = 65.0          # RemoteOK 1-req/min soft limit
    MAX_REQUEST_DELAY = 90.0

    _API_URL = "https://remoteok.io/api"
    _JOB_BASE_URL = "https://remoteok.io"

    # Curated list of RemoteOK's native tech tags to extract as skills
    _KNOWN_TECH_TAGS = frozenset({
        "python", "javascript", "typescript", "go", "golang", "rust", "java",
        "c++", "csharp", "ruby", "php", "swift", "kotlin", "scala", "elixir",
        "react", "nextjs", "vue", "angular", "node", "nodejs", "express",
        "fastapi", "django", "flask", "rails", "laravel", "spring",
        "pytorch", "tensorflow", "ml", "ai", "machine-learning", "deep-learning",
        "llm", "gpt", "langchain", "nlp", "computer-vision",
        "aws", "gcp", "azure", "cloud", "devops", "docker", "kubernetes", "k8s",
        "terraform", "ansible", "ci-cd", "github-actions", "jenkins",
        "postgres", "postgresql", "mysql", "mongodb", "redis", "elasticsearch",
        "kafka", "rabbitmq", "graphql", "rest", "grpc", "microservices",
        "solidity", "web3", "blockchain", "ethereum",
        "ios", "android", "react-native", "flutter",
        "data-science", "data-engineering", "analytics", "sql", "dbt",
        "snowflake", "databricks", "spark", "airflow",
        "linux", "bash", "security", "sre", "platform-engineering",
    })

    # Tags that describe role categories (not individual skills)
    _CATEGORY_TAGS = frozenset({
        "backend", "frontend", "fullstack", "full-stack", "mobile",
        "senior", "junior", "lead", "engineer", "developer", "programmer",
        "executive", "non-tech", "marketing", "design", "finance",
        "hr", "legal", "operations", "sales", "customer-support",
        "remote", "worldwide", "usa-only", "europe", "latam", "async",
    })

    def __init__(self) -> None:
        super().__init__()
        # Cache the full listing for the duration of one search run
        # to avoid making multiple API calls for different keyword filters
        self._cached_listings: list[dict[str, Any]] | None = None
        self._cache_fetched_at: float = 0.0
        self._cache_ttl_seconds: float = 300.0  # 5-minute cache

    async def search(
        self,
        *,
        keywords: list[str],
        locations: list[str],
        work_modes: list[str],
        max_results: int,
    ) -> list[dict[str, Any]]:
        """
        Fetch all RemoteOK listings, then filter client-side by keywords.

        Since all jobs on RemoteOK are remote, work_mode and location
        filters are effectively ignored here — all results are remote.
        Keyword matching is case-insensitive against title + tag list.
        """
        logger.info(
            "RemoteOK search started",
            keywords=keywords,
            max_results=max_results,
        )

        all_listings = await self._fetch_all_listings()

        if not all_listings:
            logger.warning("RemoteOK returned empty listing")
            return []

        # Client-side keyword filtering
        matched = self._filter_by_keywords(all_listings, keywords)

        # Sort by date posted (most recent first)
        matched.sort(
            key=lambda j: j.get("epoch", 0),
            reverse=True,
        )

        results = [self._normalise_job(j) for j in matched[:max_results]]
        results = [r for r in results if r is not None]

        logger.info(
            "RemoteOK search complete",
            total_listings=len(all_listings),
            matched=len(results),
            keywords=keywords,
        )
        return results

    async def _fetch_all_listings(self) -> list[dict[str, Any]]:
        """
        Fetch the full RemoteOK JSON listing.

        Uses an in-memory cache to avoid hitting the API more than once
        per 5 minutes in the same process — particularly important when
        multiple keywords are searched in the same discovery run (each
        keyword set doesn't need a separate API call for RemoteOK).
        """
        import time as _time

        now = _time.monotonic()
        if (
            self._cached_listings is not None
            and (now - self._cache_fetched_at) < self._cache_ttl_seconds
        ):
            logger.debug("RemoteOK: using cached listings")
            return self._cached_listings

        for attempt in range(1, self.RETRY_ATTEMPTS + 1):
            try:
                response = await self._get(
                    self._API_URL,
                    headers=self._remoteok_headers(),
                    referer=self.BASE_URL,
                )

                if self._is_empty_response(response):
                    raise ValueError("RemoteOK returned empty response")

                listings = self._parse_api_response(response)

                if not listings:
                    raise ValueError(f"RemoteOK returned 0 jobs after parsing")

                self._cached_listings = listings
                self._cache_fetched_at = now
                logger.info(f"RemoteOK: fetched {len(listings)} listings from API")
                return listings

            except Exception as exc:
                logger.warning(
                    f"RemoteOK API fetch attempt {attempt}/{self.RETRY_ATTEMPTS} failed",
                    error=str(exc)[:200],
                )
                if attempt < self.RETRY_ATTEMPTS:
                    delay = random.uniform(65, 120)  # respect rate limit on retry
                    logger.info(f"Waiting {delay:.0f}s before RemoteOK retry")
                    await asyncio.sleep(delay)

        logger.error("RemoteOK: all fetch attempts failed")
        return []

    def _parse_api_response(self, raw_response: str) -> list[dict[str, Any]]:
        """
        Parse the RemoteOK JSON API response.

        The API returns an array where the first element is always a legal
        notice / metadata dict (no 'slug' field) — this must be skipped.
        All subsequent elements are job listings.
        """
        try:
            data = json.loads(raw_response)
        except json.JSONDecodeError as exc:
            # Sometimes Cloudflare returns a challenge page that looks HTML
            if "<!DOCTYPE" in raw_response or "<html" in raw_response.lower():
                logger.warning("RemoteOK returned HTML challenge page instead of JSON")
                raise ValueError("Bot challenge detected on RemoteOK") from exc
            raise

        if not isinstance(data, list):
            raise ValueError(f"Expected list from RemoteOK API, got {type(data)}")

        # First element is always the legal/metadata entry — skip it
        listings = [
            item for item in data
            if isinstance(item, dict) and item.get("slug")  # jobs always have a slug
        ]

        return listings

    def _filter_by_keywords(
        self,
        listings: list[dict[str, Any]],
        keywords: list[str],
    ) -> list[dict[str, Any]]:
        """
        Client-side keyword filtering for RemoteOK listings.

        Matches against: job title, tag list (exact match), and position field.
        An empty keywords list returns all listings (no filter).
        Matching is OR-based across keywords — any keyword hit returns the job.
        """
        if not keywords:
            return listings

        keywords_lower = [kw.lower().strip() for kw in keywords if kw.strip()]
        if not keywords_lower:
            return listings

        matched: list[dict[str, Any]] = []

        for listing in listings:
            # Build the searchable text for this listing
            title = (listing.get("position") or listing.get("title") or "").lower()
            tags = " ".join(str(t).lower() for t in (listing.get("tags") or []))
            company = (listing.get("company") or "").lower()
            search_text = f"{title} {tags} {company}"

            if any(kw in search_text for kw in keywords_lower):
                matched.append(listing)

        return matched

    def _normalise_job(self, raw: dict[str, Any]) -> dict[str, Any] | None:
        """
        Transform a raw RemoteOK API job dict into our normalised schema.

        RemoteOK's field names:
            id, slug, position, company, logo, url, apply_url,
            tags (list of strings), description, salary_min, salary_max,
            equity, date (ISO 8601), epoch (unix timestamp)
        """
        job_id = str(raw.get("id") or raw.get("slug") or "")
        if not job_id:
            return None

        title = raw.get("position") or raw.get("title") or ""
        company = raw.get("company") or ""
        description_html = raw.get("description") or ""
        description_clean = self._clean_html(description_html)

        # Tags — split into tech skills vs category tags
        all_tags: list[str] = [str(t).lower() for t in (raw.get("tags") or [])]
        tech_skills = [
            t.title().replace("-", " ")
            for t in all_tags
            if t in self._KNOWN_TECH_TAGS
        ]
        category_tags = [
            t for t in all_tags
            if t in self._CATEGORY_TAGS
        ]

        # Experience level inference from tags and title
        exp_level = self._infer_experience_level_from_tags(category_tags, title)

        # Salary — RemoteOK provides clean integer min/max when available
        sal_min_raw = raw.get("salary_min")
        sal_max_raw = raw.get("salary_max")
        sal_min = float(sal_min_raw) if sal_min_raw else None
        sal_max = float(sal_max_raw) if sal_max_raw else None

        # If only a range string is provided (older API format)
        if sal_min is None and sal_max is None:
            salary_str = raw.get("salary") or ""
            if salary_str:
                sal_min, sal_max, _ = self._parse_salary(salary_str)

        # Apply URL — RemoteOK provides it directly (no redirect)
        apply_url = raw.get("apply_url") or raw.get("url")
        ats_provider = self._detect_ats_from_url(apply_url or "")

        # Posted date from epoch or ISO string
        posted_at = None
        epoch = raw.get("epoch")
        if epoch:
            try:
                posted_at = datetime.fromtimestamp(int(epoch), tz=timezone.utc).isoformat()
            except (ValueError, OSError, OverflowError):
                pass
        if not posted_at:
            date_str = raw.get("date")
            if date_str:
                posted_at = date_str

        # Logo URL
        logo_url = raw.get("logo")
        if logo_url and not logo_url.startswith("http"):
            logo_url = f"https://remoteok.io/{logo_url.lstrip('/')}"

        # Source URL — always the remoteok.io listing page
        source_url = raw.get("url") or f"{self.BASE_URL}/remote-jobs/{raw.get('slug', job_id)}"

        # Equity flag
        has_equity = raw.get("equity") is True

        # Company size from RemoteOK tags
        company_size = None
        for tag in all_tags:
            if "startup" in tag or "small" in tag:
                company_size = "1-50"
                break
            if "enterprise" in tag or "large" in tag:
                company_size = "1000+"
                break

        return {
            "external_id": job_id,
            "title": title,
            "company_name": company,
            "company_logo_url": logo_url,
            "company_size": company_size,
            "location": "Worldwide",
            "country": None,
            "city": None,
            "source_url": source_url,
            "apply_url": apply_url,
            "ats_provider": ats_provider,
            "job_board": "remoteok",
            "work_mode": "remote",
            "is_remote": True,
            "job_type": self._infer_job_type_from_tags(all_tags),
            "experience_level": exp_level,
            "posted_at": posted_at,
            "description": description_html[:50_000],
            "description_cleaned": description_clean[:20_000],
            "salary_min": sal_min,
            "salary_max": sal_max,
            "salary_currency": "USD",
            "salary_period": "annual",
            "has_equity": has_equity,
            "required_skills": tech_skills[:30],
            "preferred_skills": [],
            "tags": all_tags[:20],
            "raw_data": {
                "remoteok_id": job_id,
                "slug": raw.get("slug"),
                "original_tags": raw.get("tags", []),
            },
        }

    def _infer_experience_level_from_tags(
        self,
        category_tags: list[str],
        title: str,
    ) -> str | None:
        """Infer experience level from RemoteOK category tags and title."""
        title_lower = title.lower()
        all_text = " ".join(category_tags) + " " + title_lower

        if "senior" in all_text or "sr." in all_text or "lead" in all_text:
            return "senior"
        if "junior" in all_text or "jr." in all_text or "entry" in all_text:
            return "entry"
        if "principal" in all_text or "staff" in all_text:
            return "principal"
        if "director" in all_text or "head" in title_lower or "vp" in title_lower:
            return "director"
        return "mid"

    def _infer_job_type_from_tags(self, tags: list[str]) -> str:
        """Infer job type from RemoteOK tags."""
        tags_str = " ".join(tags).lower()
        if "contract" in tags_str or "freelance" in tags_str:
            return "contract"
        if "part-time" in tags_str or "parttime" in tags_str:
            return "part_time"
        if "intern" in tags_str:
            return "internship"
        return "full_time"

    def _detect_ats_from_url(self, url: str) -> str | None:
        if not url:
            return None
        url_lower = url.lower()
        ats_patterns = [
            (["greenhouse.io"], "greenhouse"),
            (["lever.co"], "lever"),
            (["workday.com", "myworkdayjobs.com"], "workday"),
            (["ashbyhq.com"], "ashby"),
            (["smartrecruiters.com"], "smartrecruiters"),
            (["icims.com"], "icims"),
            (["bamboohr.com"], "bamboohr"),
            (["workable.com"], "workable"),
            (["jobvite.com"], "jobvite"),
            (["taleo.net"], "taleo"),
        ]
        for patterns, ats in ats_patterns:
            if any(p in url_lower for p in patterns):
                return ats
        return None

    def _remoteok_headers(self) -> dict[str, str]:
        """Headers specifically tuned for RemoteOK's Cloudflare configuration."""
        return {
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": f"{self.BASE_URL}/",
            "Origin": self.BASE_URL,
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-origin",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
        }

    def _parse_listings(self, html: str) -> list[dict[str, Any]]:
        """BaseScraper abstract method — delegates to _parse_api_response."""
        return self._parse_api_response(html)