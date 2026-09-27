"""
app/automation/scraping/wellfound_scraper.py
=============================================
Production Wellfound (formerly AngelList Talent) job scraper.

Wellfound is the premier board for startup jobs and is uniquely valuable
for startup-focused job seekers because it includes:
- Company funding stage and total raised
- Team size and growth metrics
- Equity range (% and/or dollar value)
- Investor names (Y Combinator, a16z, Sequoia, etc.)
- Company culture signals and values
- Real-time hiring activity indicators

Strategy (layered, most reliable first):
1. Wellfound's internal GraphQL API — returns structured JSON with all
   company and job fields. Discovered via browser network analysis.
   No formal auth required for public jobs, but session cookies help.
2. Wellfound HTML job listing pages — fallback with regex/Next.js hydration
   data extraction from window.__NEXT_DATA__ JSON blob.

AUTHENTICATION
- Wellfound shows more results and full salary to authenticated users.
- The scraper operates in unauthenticated mode by default.
- If WELLFOUND_SESSION_COOKIE is configured in settings, it's injected
  to unlock fuller data — this is a user-provided personal session cookie,
  not OAuth-based, so handle with care.

GRAPHQL PAGINATION
- Wellfound's GraphQL API uses cursor-based pagination (not offset).
- pageInfo.endCursor is extracted and passed as 'after' in the next query.
- We stop when hasNextPage is False or max_results is reached.

ANTI-DETECTION
- Wellfound uses DataDome bot protection (not Cloudflare).
- DataDome analyses mouse movements, scroll depth, and timing patterns.
- Since we're calling the API endpoint (not interacting with the DOM),
  DataDome fingerprinting is largely bypassed — it focuses on Playwright
  interactions more than raw HTTP requests to API endpoints.
- We still rotate User-Agent and add realistic request timing.

RATE LIMITS
- Wellfound imposes ~100 requests/hour for unauthenticated sessions.
- Our minimum 8-second delay keeps us well under this threshold.
- The GraphQL endpoint is separate from the HTML pages — each counts
  separately against their rate limits.
"""

from __future__ import annotations

import asyncio
import json
import re
import random
from datetime import datetime, timezone
from typing import Any

from app.automation.scraping.base_scraper import BaseScraper
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# GraphQL query for Wellfound job search
# ---------------------------------------------------------------------------

_JOB_SEARCH_QUERY = """
query JobSearchResults(
  $query: String,
  $locationSlugs: [String],
  $roleTypes: [String],
  $jobTypes: [String],
  $first: Int,
  $after: String
) {
  talent {
    jobListings(
      query: $query,
      locationSlugs: $locationSlugs,
      roleTypes: $roleTypes,
      jobTypes: $jobTypes,
      first: $first,
      after: $after
    ) {
      pageInfo {
        hasNextPage
        endCursor
      }
      totalCount
      edges {
        node {
          id
          slug
          title
          createdAt
          remote
          locationNames
          jobType
          equity
          minCompensation
          maxCompensation
          currency
          description
          applyUrl
          atsSource
          startupId
          startup {
            id
            name
            slug
            logoUrl
            oneLiner
            website
            twitterUrl
            linkedInUrl
            crunchbaseUrl
            fundingRoundsByFunding {
              amount
              roundType
              date
            }
            totalFunding
            stage
            teamSize
            highlightedJobFunctions {
              displayName
            }
            markets {
              displayName
            }
            locations {
              displayName
            }
            companySize
            hiring
            investors {
              name
            }
            badges {
              type
              label
            }
          }
          jobSkills {
            skill {
              name
              slug
            }
          }
          requiredJobSkills {
            skill {
              name
              slug
            }
          }
          preferredJobSkills {
            skill {
              name
              slug
            }
          }
          seniorityLevel
        }
      }
    }
  }
}
"""


class WellfoundScraper(BaseScraper):
    """
    Wellfound (AngelList Talent) job scraper — GraphQL API + HTML fallback.
    """

    BASE_URL = "https://wellfound.com"
    RATE_LIMIT_DOMAIN = "wellfound.com"
    CONCURRENT_LIMIT = 2
    RETRY_ATTEMPTS = 3
    MIN_REQUEST_DELAY = 8.0
    MAX_REQUEST_DELAY = 20.0
    PAGE_SIZE = 20      # Wellfound GraphQL max per page

    _GRAPHQL_URL = "https://wellfound.com/graphql"
    _JOBS_URL = "https://wellfound.com/jobs"

    # Wellfound role type slugs for common tech roles
    _ROLE_TYPE_MAP: dict[str, list[str]] = {
        "software engineer": ["eng-software"],
        "backend": ["eng-backend"],
        "frontend": ["eng-frontend"],
        "fullstack": ["eng-fullstack"],
        "data science": ["data-scientist"],
        "machine learning": ["eng-ml"],
        "devops": ["devops"],
        "mobile": ["eng-mobile"],
        "product manager": ["pm-product"],
        "designer": ["design-product"],
    }

    # Wellfound job type slugs
    _JOB_TYPE_MAP: dict[str, str] = {
        "full_time": "fulltime",
        "part_time": "parttime",
        "contract": "contract",
        "internship": "internship",
    }

    def __init__(self) -> None:
        super().__init__()
        self._seen_ids: set[str] = set()
        self._wellfound_session = getattr(settings, "WELLFOUND_SESSION_COOKIE", None)

    async def search(
        self,
        *,
        keywords: list[str],
        locations: list[str],
        work_modes: list[str],
        max_results: int,
    ) -> list[dict[str, Any]]:
        """
        Search Wellfound and return normalised job dicts.

        Tries GraphQL API first. On any failure, attempts HTML scraping
        of the Wellfound jobs page as a fallback.
        """
        logger.info(
            "Wellfound search started",
            keywords=keywords,
            locations=locations,
            max_results=max_results,
        )

        try:
            results = await self._search_via_graphql(
                keywords=keywords,
                locations=locations,
                work_modes=work_modes,
                max_results=max_results,
            )
        except Exception as exc:
            logger.warning(
                "Wellfound GraphQL failed — trying HTML fallback",
                error=str(exc)[:200],
            )
            results = await self._search_via_html(
                keywords=keywords,
                locations=locations,
                max_results=max_results,
            )

        logger.info("Wellfound search complete", total=len(results))
        return results

    # ---------------------------------------------------------------------------
    # GraphQL search
    # ---------------------------------------------------------------------------

    async def _search_via_graphql(
        self,
        *,
        keywords: list[str],
        locations: list[str],
        work_modes: list[str],
        max_results: int,
    ) -> list[dict[str, Any]]:
        """
        Execute paginated GraphQL queries against the Wellfound jobs API.
        """
        all_jobs: list[dict[str, Any]] = []
        cursor: str | None = None
        has_next = True
        query_str = " ".join(keywords) if keywords else "software engineer"

        # Build location slugs from location names
        location_slugs = self._build_location_slugs(locations, work_modes)

        # Build role types from keywords
        role_types = self._build_role_types(keywords)

        while has_next and len(all_jobs) < max_results:
            variables: dict[str, Any] = {
                "query": query_str,
                "locationSlugs": location_slugs,
                "roleTypes": role_types,
                "jobTypes": ["fulltime"],
                "first": min(self.PAGE_SIZE, max_results - len(all_jobs)),
            }
            if cursor:
                variables["after"] = cursor

            try:
                response_data = await self._graphql_request(
                    query=_JOB_SEARCH_QUERY,
                    variables=variables,
                )
            except Exception as exc:
                logger.warning("Wellfound GraphQL page failed", cursor=cursor, error=str(exc)[:200])
                break

            # Navigate the response
            job_listings = (
                response_data
                .get("data", {})
                .get("talent", {})
                .get("jobListings", {})
            )

            if not job_listings:
                logger.warning("Wellfound: empty jobListings in GraphQL response")
                break

            page_info = job_listings.get("pageInfo", {})
            has_next = page_info.get("hasNextPage", False)
            cursor = page_info.get("endCursor")

            edges = job_listings.get("edges", [])
            for edge in edges:
                node = edge.get("node", {})
                job_id = str(node.get("id") or "")
                if not job_id or job_id in self._seen_ids:
                    continue
                self._seen_ids.add(job_id)
                normalised = self._normalise_graphql_job(node)
                if normalised:
                    all_jobs.append(normalised)

            logger.debug(
                "Wellfound page fetched",
                page_size=len(edges),
                total_so_far=len(all_jobs),
                has_next=has_next,
            )

            if not has_next:
                break

            await asyncio.sleep(random.uniform(self.MIN_REQUEST_DELAY, self.MAX_REQUEST_DELAY))

        return all_jobs[:max_results]

    async def _graphql_request(
        self,
        query: str,
        variables: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute a single GraphQL request against the Wellfound API."""
        payload = json.dumps({"query": query, "variables": variables})

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "X-Requested-With": "XMLHttpRequest",
            "Referer": f"{self._JOBS_URL}?q=software-engineer",
            "Origin": self.BASE_URL,
        }
        if self._wellfound_session:
            headers["Cookie"] = f"remember_user_token={self._wellfound_session}"

        # Use POST via a custom execute method since _get only does GET
        response_text = await self._post(
            self._GRAPHQL_URL,
            body=payload,
            headers=headers,
        )

        try:
            data = json.loads(response_text)
        except json.JSONDecodeError:
            raise ValueError(f"Wellfound GraphQL returned non-JSON: {response_text[:200]}")

        if "errors" in data:
            errors = data["errors"]
            raise ValueError(f"Wellfound GraphQL errors: {errors}")

        return data

    async def _post(
        self,
        url: str,
        body: str,
        headers: dict[str, str],
    ) -> str:
        """
        Execute a POST request. Extended from BaseScraper's GET-only engine.
        Uses the same anti-detection and retry logic as _get().
        """
        from app.core.exceptions import ScraperException

        await self._rate_limit_wait()

        for attempt in range(1, self.RETRY_ATTEMPTS + 1):
            try:
                from curl_cffi.requests import AsyncSession
                proxy = self._proxy_manager.get_proxy()
                proxy_kwargs = {"proxy": proxy} if proxy else {}

                async with AsyncSession(impersonate="chrome124") as session:
                    resp = await session.post(
                        url,
                        data=body,
                        headers=headers,
                        timeout=settings.SCRAPE_REQUEST_TIMEOUT_SECONDS,
                        **proxy_kwargs,
                    )
                    resp.raise_for_status()
                    return resp.text

            except ImportError:
                import httpx
                proxy = self._proxy_manager.get_proxy()
                proxy_kwargs = {"proxy": proxy} if proxy else {}
                async with httpx.AsyncClient(
                    timeout=settings.SCRAPE_REQUEST_TIMEOUT_SECONDS,
                    follow_redirects=True,
                    **proxy_kwargs,
                ) as client:
                    resp = await client.post(url, content=body, headers=headers)
                    resp.raise_for_status()
                    return resp.text

            except Exception as exc:
                if attempt == self.RETRY_ATTEMPTS:
                    raise ScraperException(
                        board="wellfound",
                        reason=f"POST failed after {self.RETRY_ATTEMPTS} attempts: {exc}",
                        url=url,
                    ) from exc
                await self._backoff(attempt)

        raise ScraperException(board="wellfound", reason="POST exhausted retries", url=url)

    # ---------------------------------------------------------------------------
    # Job normalisation
    # ---------------------------------------------------------------------------

    def _normalise_graphql_job(self, node: dict[str, Any]) -> dict[str, Any] | None:
        """Transform a Wellfound GraphQL job node into our normalised schema."""
        job_id = str(node.get("id") or "")
        slug = node.get("slug") or ""
        title = node.get("title") or ""
        if not title or not job_id:
            return None

        startup = node.get("startup") or {}
        company_name = startup.get("name") or ""
        company_logo = startup.get("logoUrl")
        company_website = startup.get("website")
        company_linkedin = startup.get("linkedInUrl")

        # Description
        description_html = node.get("description") or ""
        description_clean = self._clean_html(description_html)

        # Location / work mode
        location_names: list[str] = node.get("locationNames") or []
        is_remote = node.get("remote") or any(
            "remote" in loc.lower() for loc in location_names
        )
        work_mode = "remote" if is_remote else "onsite"
        location_str = ", ".join(location_names) if location_names else ("Worldwide" if is_remote else "")

        # Job type
        job_type_raw = node.get("jobType") or "fulltime"
        job_type = self._normalise_job_type(job_type_raw)

        # Skills
        required_skills = [
            s["skill"]["name"]
            for s in (node.get("requiredJobSkills") or [])
            if s.get("skill", {}).get("name")
        ]
        preferred_skills = [
            s["skill"]["name"]
            for s in (node.get("preferredJobSkills") or [])
            if s.get("skill", {}).get("name")
        ]
        all_skills = [
            s["skill"]["name"]
            for s in (node.get("jobSkills") or [])
            if s.get("skill", {}).get("name")
        ]
        if not required_skills:
            required_skills = all_skills[:20]

        # Salary
        min_comp = node.get("minCompensation")
        max_comp = node.get("maxCompensation")
        salary_currency = node.get("currency") or "USD"

        # Equity
        equity_raw = node.get("equity") or ""
        has_equity = bool(equity_raw)

        # Apply URL
        apply_url = node.get("applyUrl")
        ats_provider = node.get("atsSource") or self._detect_ats_from_url(apply_url or "")

        # Source URL
        if slug:
            source_url = f"{self.BASE_URL}/jobs/{slug}"
        else:
            source_url = f"{self.BASE_URL}/l/job/{job_id}"

        # Posted date
        created_at = node.get("createdAt")
        posted_at = None
        if created_at:
            try:
                if isinstance(created_at, (int, float)):
                    posted_at = datetime.fromtimestamp(created_at / 1000, tz=timezone.utc).isoformat()
                else:
                    posted_at = created_at
            except Exception:
                posted_at = None

        # Experience level
        seniority = node.get("seniorityLevel") or ""
        exp_level = self._normalise_seniority(seniority, title)

        # Company / startup enrichment fields
        funding_stage = startup.get("stage") or ""
        team_size = startup.get("teamSize")
        total_funding = startup.get("totalFunding")
        investors = [i.get("name") for i in (startup.get("investors") or []) if i.get("name")]
        markets = [m.get("displayName") for m in (startup.get("markets") or []) if m.get("displayName")]

        # Compute content hash for cross-board dedup
        content_hash = self._compute_content_hash(title, company_name, description_clean)

        return {
            "external_id": job_id,
            "title": title,
            "company_name": company_name,
            "company_logo_url": company_logo,
            "company_website": company_website,
            "location": location_str,
            "country": None,
            "city": None,
            "source_url": source_url,
            "apply_url": apply_url,
            "ats_provider": ats_provider,
            "job_board": "wellfound",
            "work_mode": work_mode,
            "is_remote": is_remote,
            "job_type": job_type,
            "experience_level": exp_level,
            "posted_at": posted_at,
            "description": description_html[:50_000],
            "description_cleaned": description_clean[:20_000],
            "salary_min": float(min_comp) if min_comp else None,
            "salary_max": float(max_comp) if max_comp else None,
            "salary_currency": salary_currency,
            "salary_period": "annual",
            "has_equity": has_equity,
            "required_skills": required_skills[:30],
            "preferred_skills": preferred_skills[:20],
            "tags": markets[:10],
            "content_hash": content_hash,
            "raw_data": {
                "wellfound_id": job_id,
                "slug": slug,
                "equity_range": equity_raw,
                "funding_stage": funding_stage,
                "team_size": team_size,
                "total_funding_usd": total_funding,
                "investors": investors[:10],
                "company_linkedin_url": company_linkedin,
            },
        }

    def _normalise_seniority(self, seniority: str, title: str) -> str | None:
        """Normalise Wellfound seniority field to our experience_level enum."""
        seniority_lower = (seniority or "").lower()
        title_lower = (title or "").lower()

        mapping = [
            (["intern", "internship"], "entry"),
            (["junior", "jr", "entry", "associate"], "entry"),
            (["mid", "intermediate", "regular"], "mid"),
            (["senior", "sr", "lead"], "senior"),
            (["staff"], "staff"),
            (["principal"], "principal"),
            (["director", "vp", "vice president", "head of"], "director"),
            (["c-level", "cto", "ceo", "cpo", "vp of"], "c_level"),
        ]
        all_text = f"{seniority_lower} {title_lower}"
        for keywords, level in mapping:
            if any(kw in all_text for kw in keywords):
                return level
        return "mid"

    def _detect_ats_from_url(self, url: str) -> str | None:
        if not url:
            return None
        url_lower = url.lower()
        for patterns, ats in [
            (["greenhouse.io"], "greenhouse"),
            (["lever.co"], "lever"),
            (["workday.com"], "workday"),
            (["ashbyhq.com"], "ashby"),
            (["smartrecruiters.com"], "smartrecruiters"),
            (["icims.com"], "icims"),
            (["bamboohr.com"], "bamboohr"),
        ]:
            if any(p in url_lower for p in patterns):
                return ats
        return None

    def _build_location_slugs(
        self,
        locations: list[str],
        work_modes: list[str],
    ) -> list[str]:
        """Build Wellfound location slugs from location names and work modes."""
        slugs: list[str] = []
        if "remote" in work_modes or not locations:
            slugs.append("remote")
        for loc in locations:
            loc_lower = loc.lower().strip()
            slug_mapping = {
                "new york": "new-york-city-metro",
                "nyc": "new-york-city-metro",
                "san francisco": "san-francisco-bay-area",
                "sf": "san-francisco-bay-area",
                "los angeles": "los-angeles-ca",
                "london": "london-uk",
                "berlin": "berlin-germany",
                "toronto": "toronto-ontario",
                "austin": "austin-tx",
                "seattle": "seattle-wa",
                "boston": "boston-ma",
                "chicago": "chicago-il",
                "miami": "miami-fl",
                "denver": "denver-colorado",
            }
            for city, slug in slug_mapping.items():
                if city in loc_lower:
                    slugs.append(slug)
                    break
        return slugs or ["remote"]

    def _build_role_types(self, keywords: list[str]) -> list[str]:
        """Map keyword list to Wellfound role type slugs."""
        roles: list[str] = []
        keywords_lower = [kw.lower() for kw in keywords]
        for keyword, role_list in self._ROLE_TYPE_MAP.items():
            if any(keyword in kw for kw in keywords_lower):
                roles.extend(role_list)
        return roles or ["eng-software"]

    # ---------------------------------------------------------------------------
    # HTML fallback
    # ---------------------------------------------------------------------------

    async def _search_via_html(
        self,
        *,
        keywords: list[str],
        locations: list[str],
        max_results: int,
    ) -> list[dict[str, Any]]:
        """
        Fallback HTML scraper using Wellfound's Next.js hydration data.
        Wellfound embeds all job data into window.__NEXT_DATA__ as a JSON
        blob — no complex HTML parsing needed once we extract that blob.
        """
        query = "-".join(kw.lower().replace(" ", "-") for kw in keywords) or "software-engineer"
        url = f"{self._JOBS_URL}?q={query}&remote=true"

        try:
            html = await self._get(url, referer="https://www.google.com/")
        except Exception as exc:
            logger.error("Wellfound HTML fallback failed", error=str(exc)[:200])
            return []

        next_data = self._extract_next_data(html)
        if not next_data:
            logger.warning("Wellfound: could not extract __NEXT_DATA__ from HTML")
            return []

        return self._parse_next_data_jobs(next_data, max_results)

    def _extract_next_data(self, html: str) -> dict | None:
        """Extract the __NEXT_DATA__ JSON blob from a Next.js rendered page."""
        match = re.search(
            r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>',
            html, re.DOTALL
        )
        if not match:
            return None
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            return None

    def _parse_next_data_jobs(
        self,
        next_data: dict,
        max_results: int,
    ) -> list[dict[str, Any]]:
        """Extract job listings from the Next.js page props."""
        jobs: list[dict[str, Any]] = []
        try:
            # Navigate the Next.js data structure (path may vary by page version)
            page_props = next_data.get("props", {}).get("pageProps", {})
            job_data = (
                page_props.get("jobListings")
                or page_props.get("jobs")
                or page_props.get("initialJobListings")
                or []
            )
            for raw in job_data[:max_results]:
                normalised = self._normalise_graphql_job(raw)
                if normalised:
                    jobs.append(normalised)
        except Exception as exc:
            logger.warning("Wellfound __NEXT_DATA__ parsing failed", error=str(exc)[:200])
        return jobs

    def _parse_listings(self, html: str) -> list[dict[str, Any]]:
        """BaseScraper abstract method — delegates to Next.js extraction."""
        next_data = self._extract_next_data(html)
        if next_data:
            return self._parse_next_data_jobs(next_data, 100)
        return []