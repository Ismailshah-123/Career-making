"""
CareerGPT -- Backend API Client
===================================
Thin wrapper around httpx for every endpoint the Streamlit frontend
calls. Centralizing this here means:
  - one place to handle the base URL / timeouts / auth headers
  - one place to handle token refresh on 401
  - pages stay free of raw HTTP/request-shape details

All methods raise APIError on a non-2xx response (with the backend's own
error message when available) so pages can catch one exception type.
"""

from __future__ import annotations

import os
from typing import Any

import httpx
import streamlit as st

API_BASE_URL = os.environ.get("CAREERGPT_API_URL", "http://localhost:8000/api/v1")
REQUEST_TIMEOUT = 30.0


class APIError(Exception):
    """Raised for any non-2xx response from the backend."""

    def __init__(self, status_code: int, message: str, detail: Any = None) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(message)


class APIClient:
    """
    Stateless-ish HTTP client. Reads/writes the JWT access token from
    st.session_state so every page shares one logged-in session.
    """

    def __init__(self, base_url: str = API_BASE_URL) -> None:
        self.base_url = base_url.rstrip("/")

    # -- token helpers ------------------------------------------------------------

    @property
    def _access_token(self) -> str | None:
        return st.session_state.get("access_token")

    @property
    def _refresh_token(self) -> str | None:
        return st.session_state.get("refresh_token")

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        if self._access_token:
            headers["Authorization"] = f"Bearer {self._access_token}"
        return headers

    def _store_tokens(self, data: dict[str, Any]) -> None:
        st.session_state["access_token"] = data.get("access_token")
        st.session_state["refresh_token"] = data.get("refresh_token")
        st.session_state["user_id"] = data.get("user_id")
        st.session_state["user_email"] = data.get("email")
        st.session_state["user_plan"] = data.get("plan")

    def logout(self) -> None:
        for key in ("access_token", "refresh_token", "user_id", "user_email", "user_plan"):
            st.session_state.pop(key, None)

    @property
    def is_authenticated(self) -> bool:
        return bool(self._access_token)

    # -- low-level request with one automatic refresh-and-retry --------------------

    def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: dict | None = None,
        params: dict | None = None,
        files: dict | None = None,
        data: dict | None = None,
        _retried: bool = False,
    ) -> Any:
        url = f"{self.base_url}{path}"
        try:
            with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
                resp = client.request(
                    method, url, json=json_body, params=params,
                    files=files, data=data, headers=self._headers(),
                )
        except httpx.ConnectError as exc:
            raise APIError(0, f"Can't reach the API at {self.base_url}. Is the backend running?") from exc
        except httpx.TimeoutException as exc:
            raise APIError(0, "Request timed out. The backend took too long to respond.") from exc

        if resp.status_code == 401 and not _retried and self._refresh_token:
            if self._try_refresh():
                return self._request(method, path, json_body=json_body, params=params,
                                      files=files, data=data, _retried=True)

        if resp.status_code >= 400:
            detail = None
            message = f"Request failed ({resp.status_code})"
            try:
                body = resp.json()
                detail = body
                message = body.get("message") or body.get("detail") or message
            except Exception:
                pass
            raise APIError(resp.status_code, message, detail)

        if resp.status_code == 204 or not resp.content:
            return None
        return resp.json()

    def _try_refresh(self) -> bool:
        try:
            with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
                resp = client.post(
                    f"{self.base_url}/auth/refresh",
                    json={"refresh_token": self._refresh_token},
                )
            if resp.status_code == 200:
                self._store_tokens(resp.json())
                return True
        except Exception:
            pass
        self.logout()
        return False

    # ── Auth ─────────────────────────────────────────────────────────────────────

    def register(self, email: str, password: str, full_name: str, timezone: str = "UTC") -> dict:
        data = self._request("POST", "/auth/register", json_body={
            "email": email, "password": password, "full_name": full_name, "timezone": timezone,
        })
        self._store_tokens(data)
        return data

    def login(self, email: str, password: str, remember_me: bool = False) -> dict:
        data = self._request("POST", "/auth/login", json_body={
            "email": email, "password": password, "remember_me": remember_me,
        })
        self._store_tokens(data)
        return data

    def forgot_password(self, email: str) -> dict:
        return self._request("POST", "/auth/forgot-password", json_body={"email": email})

    # ── Users / Profile ──────────────────────────────────────────────────────────

    def get_profile(self) -> dict:
        return self._request("GET", "/users/me")

    def update_profile(self, **fields: Any) -> dict:
        return self._request("PATCH", "/users/me", json_body={k: v for k, v in fields.items() if v is not None})

    def get_usage(self) -> dict:
        return self._request("GET", "/users/me/usage")

    # ── Resumes ──────────────────────────────────────────────────────────────────

    def list_resumes(self) -> list[dict]:
        result = self._request("GET", "/resumes/")
        return result if isinstance(result, list) else result.get("items", [])

    def upload_resume(self, filename: str, content: bytes, content_type: str) -> dict:
        return self._request("POST", "/resumes/upload", files={"file": (filename, content, content_type)})

    def get_resume(self, resume_id: str) -> dict:
        return self._request("GET", f"/resumes/{resume_id}")

    def delete_resume(self, resume_id: str) -> None:
        return self._request("DELETE", f"/resumes/{resume_id}")

    def tailor_resume(self, resume_id: str, job_id: str, optimization_level: str = "balanced") -> dict:
        return self._request("POST", f"/resumes/{resume_id}/tailor", json_body={
            "job_id": job_id, "optimization_level": optimization_level,
        })

    def ats_score(self, resume_id: str) -> dict:
        return self._request("GET", f"/resumes/{resume_id}/ats-score")

    # ── Jobs ─────────────────────────────────────────────────────────────────────

    def search_jobs(self, **params: Any) -> dict:
        return self._request("GET", "/jobs/search", params={k: v for k, v in params.items() if v not in (None, "")})

    def get_job(self, job_id: str) -> dict:
        return self._request("GET", f"/jobs/{job_id}")

    def match_job(self, job_id: str, resume_id: str | None = None) -> dict:
        body = {"resume_id": resume_id} if resume_id else {}
        return self._request("POST", f"/jobs/{job_id}/match", json_body=body)

    def get_recommendations(self, limit: int = 20) -> list[dict]:
        result = self._request("GET", "/jobs/recommendations", params={"limit": limit})
        return result if isinstance(result, list) else result.get("items", [])

    def bookmark_job(self, job_id: str) -> dict:
        return self._request("POST", f"/jobs/{job_id}/bookmark")

    def get_job_stats(self) -> dict:
        return self._request("GET", "/jobs/stats")

    def discover_jobs(self, keywords: list[str], locations: list[str]) -> dict:
        return self._request("POST", "/jobs/discover", json_body={"keywords": keywords, "locations": locations})

    # ── Applications ──────────────────────────────────────────────────────────────

    def list_applications(self, status: str | None = None) -> list[dict]:
        params = {"status": status} if status else {}
        result = self._request("GET", "/applications/", params=params)
        return result if isinstance(result, list) else result.get("items", [])

    def create_application(self, job_id: str, resume_id: str | None = None, auto_apply: bool = False) -> dict:
        return self._request("POST", "/applications/", json_body={
            "job_id": job_id, "resume_id": resume_id, "auto_apply": auto_apply,
        })

    def get_application(self, application_id: str) -> dict:
        return self._request("GET", f"/applications/{application_id}")

    def update_application_status(self, application_id: str, status: str, note: str | None = None) -> dict:
        return self._request("PATCH", f"/applications/{application_id}/status", json_body={
            "status": status, "note": note,
        })

    def get_pipeline(self) -> dict:
        return self._request("GET", "/applications/pipeline")

    def get_application_stats(self) -> dict:
        return self._request("GET", "/applications/stats")

    def trigger_followup(self, application_id: str) -> dict:
        return self._request("POST", f"/applications/{application_id}/followup")

    def add_note(self, application_id: str, note: str) -> dict:
        return self._request("POST", f"/applications/{application_id}/notes", json_body={"note": note})

    def get_timeline(self, application_id: str) -> list[dict]:
        result = self._request("GET", f"/applications/{application_id}/timeline")
        return result if isinstance(result, list) else result.get("items", [])

    # ── LinkedIn ─────────────────────────────────────────────────────────────────

    def linkedin_profile(self) -> dict:
        return self._request("GET", "/linkedin/profile")

    def linkedin_auth_url(self) -> dict:
        return self._request("GET", "/linkedin/auth/url")

    def linkedin_disconnect(self) -> None:
        return self._request("DELETE", "/linkedin/auth/disconnect")

    def list_linkedin_posts(self) -> list[dict]:
        result = self._request("GET", "/linkedin/posts/")
        return result if isinstance(result, list) else result.get("items", [])

    def generate_linkedin_post(self, topic: str, tone: str = "professional") -> dict:
        return self._request("POST", "/linkedin/posts/generate", json_body={"topic": topic, "tone": tone})

    def approve_linkedin_post(self, post_id: str) -> dict:
        return self._request("POST", f"/linkedin/posts/{post_id}/approve")

    def publish_linkedin_post(self, post_id: str) -> dict:
        return self._request("POST", f"/linkedin/posts/{post_id}/publish")

    def linkedin_analytics_overview(self) -> dict:
        return self._request("GET", "/linkedin/analytics/overview")

    # ── Health ───────────────────────────────────────────────────────────────────

    def health(self) -> dict:
        with httpx.Client(timeout=5.0) as client:
            resp = client.get(f"{self.base_url.rsplit('/api', 1)[0]}/health")
            return resp.json()


@st.cache_resource
def get_client() -> APIClient:
    return APIClient()
