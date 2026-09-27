"""
app/workflows/state.py
========================
Shared state schemas for every LangGraph workflow in the platform.

LangGraph passes a single mutable state object through every node in a
graph. Each node reads what it needs and returns a partial dict of updates
that LangGraph merges back into the running state via the reducers defined
below — nodes never mutate state directly, they return deltas.

Design rules followed throughout this module:
1. Every field has an explicit reducer (Annotated[..., operator]) so
   concurrent branches (e.g. parallel job-board scraping) merge predictably
   instead of one branch silently overwriting another's writes.
2. List fields use `operator.add` (append/concat) — safe for fan-out nodes.
3. Scalar status/flag fields use a "last write wins" reducer (no operator
   annotation) since only one node should be setting them at a time.
4. Nothing here imports from db/models — state must stay JSON-serialisable
   (Celery serialises it between task hops), so we pass UUIDs as str.
5. TypedDict doesn't support clean field inheritance across total=False
   variants, so each workflow state below re-declares the base fields
   explicitly rather than inheriting — this is intentional, not duplication
   by accident.
"""

from __future__ import annotations

import operator
from datetime import datetime, timezone
from typing import Annotated, Any, Literal, TypedDict


# ---------------------------------------------------------------------------
# Reducer helpers
# ---------------------------------------------------------------------------

def _last_write_wins(_old: Any, new: Any) -> Any:
    """Explicit 'last write wins' reducer — makes intent visible vs omitting an operator."""
    return new


def _merge_dicts(old: dict | None, new: dict | None) -> dict:
    """Shallow-merge two dicts — used for accumulating metadata across nodes."""
    merged = dict(old or {})
    merged.update(new or {})
    return merged


# ---------------------------------------------------------------------------
# Base state factory — used to seed every workflow's initial state dict
# ---------------------------------------------------------------------------

def new_base_state(
    *,
    user_id: str,
    agent_run_id: str,
    workflow_id: str,
    celery_task_id: str | None = None,
) -> dict[str, Any]:
    """Factory for the initial state dict passed into graph.ainvoke()."""
    return {
        "user_id": user_id,
        "agent_run_id": agent_run_id,
        "workflow_id": workflow_id,
        "celery_task_id": celery_task_id,
        "current_node": "start",
        "completed_nodes": [],
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "errors": [],
        "retry_count": 0,
        "should_halt": False,
        "total_prompt_tokens": 0,
        "total_completion_tokens": 0,
        "total_llm_calls": 0,
        "metadata": {},
    }


# ---------------------------------------------------------------------------
# JobWorkflowState — discovery -> filter -> match -> tailor -> apply -> notify
# ---------------------------------------------------------------------------

class JobWorkflowState(TypedDict, total=False):
    """
    State for the end-to-end job pipeline graph (job_workflow.py).

    Flow: discover_jobs -> filter_jobs -> match_jobs -> tailor_resume
          -> generate_cover_letter -> submit_application -> send_notification
    """

    # --- base fields ---
    user_id: str
    agent_run_id: str
    workflow_id: str
    celery_task_id: str | None
    current_node: str
    completed_nodes: Annotated[list[str], operator.add]
    started_at: str
    status: Annotated[Literal["pending", "running", "completed", "failed"], _last_write_wins]
    errors: Annotated[list[dict[str, Any]], operator.add]
    retry_count: Annotated[int, _last_write_wins]
    should_halt: Annotated[bool, _last_write_wins]
    total_prompt_tokens: Annotated[int, operator.add]
    total_completion_tokens: Annotated[int, operator.add]
    total_llm_calls: Annotated[int, operator.add]
    metadata: Annotated[dict[str, Any], _merge_dicts]

    # --- discovery inputs ---
    job_boards: list[str]
    search_keywords: list[str]
    search_locations: list[str]
    work_modes: list[str]
    max_results_per_board: int

    # --- discovery outputs (fan-out: each board scraper appends here) ---
    discovered_job_ids: Annotated[list[str], operator.add]
    discovery_errors_by_board: Annotated[dict[str, str], _merge_dicts]

    # --- filter stage ---
    filtered_job_ids: list[str]
    excluded_job_ids: Annotated[list[str], operator.add]
    exclusion_reasons: Annotated[dict[str, str], _merge_dicts]

    # --- matching stage ---
    resume_id: str | None
    match_results: Annotated[list[dict[str, Any]], operator.add]
    top_match_job_ids: list[str]

    # --- per-job application processing (single-job mode) ---
    target_job_id: str | None
    application_id: str | None
    tailored_resume_id: str | None
    cover_letter_id: str | None

    # --- submission stage ---
    application_submitted: bool
    submission_method: str | None
    submission_error: str | None

    # --- notification stage ---
    notification_sent: bool


# ---------------------------------------------------------------------------
# ResumeWorkflowState — upload -> parse -> embed -> ATS score -> (tailor)
# ---------------------------------------------------------------------------

class ResumeWorkflowState(TypedDict, total=False):
    """
    State for the resume processing graph (resume_workflow.py).

    Flow: extract_text -> parse_sections -> extract_skills
          -> compute_ats_score -> generate_embedding -> finalize
    """

    user_id: str
    agent_run_id: str
    workflow_id: str
    celery_task_id: str | None
    current_node: str
    completed_nodes: Annotated[list[str], operator.add]
    started_at: str
    status: Annotated[Literal["pending", "running", "completed", "failed"], _last_write_wins]
    errors: Annotated[list[dict[str, Any]], operator.add]
    retry_count: Annotated[int, _last_write_wins]
    should_halt: Annotated[bool, _last_write_wins]
    total_prompt_tokens: Annotated[int, operator.add]
    total_completion_tokens: Annotated[int, operator.add]
    total_llm_calls: Annotated[int, operator.add]
    metadata: Annotated[dict[str, Any], _merge_dicts]

    # --- inputs ---
    resume_id: str
    file_path: str
    mime_type: str
    is_tailoring: bool
    target_job_id: str | None
    tailor_tone: str | None
    emphasise_skills: list[str]

    # --- extraction stage ---
    raw_text: str | None
    word_count: int | None
    extraction_error: str | None

    # --- parsing stage ---
    parsed_sections: dict[str, Any] | None
    extracted_skills: list[str]
    extracted_job_titles: list[str]
    years_of_experience: float | None

    # --- tailoring stage (only when is_tailoring=True) ---
    tailored_sections: dict[str, Any] | None
    tailoring_changes: dict[str, Any] | None
    new_resume_id: str | None

    # --- ATS scoring stage ---
    ats_score: float | None
    ats_feedback: dict[str, Any] | None
    keyword_density_score: float | None

    # --- embedding stage ---
    embedding_vector: list[float] | None
    qdrant_point_ids: list[str]


# ---------------------------------------------------------------------------
# LinkedInWorkflowState — topic research -> draft -> review -> publish
# ---------------------------------------------------------------------------

class LinkedInWorkflowState(TypedDict, total=False):
    """
    State for LinkedIn content generation graph (linkedin_workflow.py).

    Flow: plan_topics -> research_topic -> draft_post -> score_quality
          -> (requires_approval ? wait_for_approval : auto_approve)
          -> schedule_or_publish
    """

    user_id: str
    agent_run_id: str
    workflow_id: str
    celery_task_id: str | None
    current_node: str
    completed_nodes: Annotated[list[str], operator.add]
    started_at: str
    status: Annotated[Literal["pending", "running", "completed", "failed"], _last_write_wins]
    errors: Annotated[list[dict[str, Any]], operator.add]
    retry_count: Annotated[int, _last_write_wins]
    should_halt: Annotated[bool, _last_write_wins]
    total_prompt_tokens: Annotated[int, operator.add]
    total_completion_tokens: Annotated[int, operator.add]
    total_llm_calls: Annotated[int, operator.add]
    metadata: Annotated[dict[str, Any], _merge_dicts]

    # --- planning inputs ---
    requested_categories: list[str]
    days_ahead: int
    schedule_time: str | None

    # --- planning outputs (fan-out: one entry per day to generate) ---
    content_calendar: Annotated[list[dict[str, Any]], operator.add]

    # --- per-post generation (single-post mode) ---
    topic: str | None
    category: str | None
    tone: str | None
    source_urls: list[str]

    # --- research stage ---
    research_findings: list[dict[str, Any]] | None
    research_citations: list[str]

    # --- drafting stage ---
    draft_content: str | None
    draft_hook: str | None
    draft_hashtags: list[str]
    draft_cta: str | None

    # --- quality scoring stage ---
    quality_score: float | None
    quality_feedback: list[str]
    needs_revision: bool

    # --- approval gate ---
    requires_approval: bool
    is_approved: bool

    # --- output ---
    post_id: str | None
    publish_status: str | None


# ---------------------------------------------------------------------------
# ApplicationWorkflowState — tailor -> cover letter -> automate -> outreach
# ---------------------------------------------------------------------------

class ApplicationWorkflowState(TypedDict, total=False):
    """
    State for the per-application submission graph (application_workflow.py).

    Flow: validate_application -> tailor_resume -> generate_cover_letter
          -> fill_application_form -> submit_form -> verify_submission
          -> (success ? discover_recruiter -> send_outreach : flag_for_manual_review)
    """

    user_id: str
    agent_run_id: str
    workflow_id: str
    celery_task_id: str | None
    current_node: str
    completed_nodes: Annotated[list[str], operator.add]
    started_at: str
    status: Annotated[Literal["pending", "running", "completed", "failed"], _last_write_wins]
    errors: Annotated[list[dict[str, Any]], operator.add]
    retry_count: Annotated[int, _last_write_wins]
    should_halt: Annotated[bool, _last_write_wins]
    total_prompt_tokens: Annotated[int, operator.add]
    total_completion_tokens: Annotated[int, operator.add]
    total_llm_calls: Annotated[int, operator.add]
    metadata: Annotated[dict[str, Any], _merge_dicts]

    # --- inputs ---
    application_id: str
    job_id: str
    base_resume_id: str
    apply_url: str | None
    ats_provider: str | None

    # --- resume tailoring stage ---
    tailored_resume_id: str | None
    tailored_resume_path: str | None

    # --- cover letter stage ---
    cover_letter_id: str | None
    cover_letter_text: str | None

    # --- form automation stage ---
    form_fields_detected: list[dict[str, Any]]
    form_fields_filled: Annotated[list[str], operator.add]
    form_screenshot_path: str | None
    automation_step: str | None

    # --- submission result ---
    submission_success: bool
    submission_confirmation_text: str | None
    requires_manual_review: bool
    manual_review_reason: str | None

    # --- post-submission outreach (optional continuation) ---
    recruiter_id: str | None
    outreach_sent: bool

    # --- next follow-up scheduling ---
    next_followup_at: str | None


# ---------------------------------------------------------------------------
# Type union for generic graph utilities (graph.py uses this for typing)
# ---------------------------------------------------------------------------

AnyWorkflowState = (
    JobWorkflowState
    | ResumeWorkflowState
    | LinkedInWorkflowState
    | ApplicationWorkflowState
)