"""
app/db/models/agent_run.py
===========================
AgentRun model — execution ledger for every multi-agent workflow invocation.

Every time an agent runs (discovery, resume, matching, cover_letter,
outreach, linkedin, application, followup), a row is created here before
execution begins and updated through completion, failure, or cancellation.

This provides:
1. Full audit trail of AI actions taken on behalf of each user
2. Token / cost tracking per run for billing and rate limiting
3. Retry state management for failed runs
4. Performance benchmarking (latency, step timings)
5. Input/output snapshots for debugging and reproducibility

Relationships:
- user → User
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import BaseModel
from app.core.constants import AgentRunStatus

if TYPE_CHECKING:
    from app.db.models.user import User


class AgentRun(BaseModel):
    """
    Execution record for a single agent invocation.

    Created BEFORE the agent starts — status begins as 'pending'.
    Updated atomically as the agent progresses through its workflow.

    Inherits id (UUID PK), created_at, updated_at, is_deleted, deleted_at.
    """

    __tablename__ = "agent_runs"

    __table_args__ = (
        Index("ix_agent_runs_user_id", "user_id"),
        Index("ix_agent_runs_agent_name", "agent_name"),
        Index("ix_agent_runs_status", "status"),
        Index("ix_agent_runs_user_agent", "user_id", "agent_name"),
        Index("ix_agent_runs_started_at", "started_at"),
        Index("ix_agent_runs_workflow_id", "workflow_id"),
        Index("ix_agent_runs_celery_task_id", "celery_task_id"),
    )

    # -----------------------------------------------------------------------
    # Ownership & Identity
    # -----------------------------------------------------------------------

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="User on whose behalf this agent ran.",
    )

    agent_name: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        index=True,
        comment=(
            "Agent identifier: discovery_agent | resume_agent | matching_agent | "
            "cover_letter_agent | outreach_agent | linkedin_agent | "
            "application_agent | followup_agent."
        ),
    )

    workflow_id: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
        index=True,
        comment="LangGraph workflow run ID grouping multiple agent steps.",
    )

    celery_task_id: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
        index=True,
        comment="Celery task ID for this run — used for status polling and revocation.",
    )

    parent_run_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("agent_runs.id", ondelete="SET NULL"),
        nullable=True,
        comment="Parent AgentRun if this is a sub-agent invoked by an orchestrator.",
    )

    # -----------------------------------------------------------------------
    # Execution Context
    # -----------------------------------------------------------------------

    trigger: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        default="api",
        server_default="api",
        comment="What triggered this run: api | celery_schedule | user_action | webhook | test.",
    )

    input_payload: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        server_default="{}",
        comment="Input parameters passed to the agent at invocation.",
    )

    context: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        server_default="{}",
        comment="Runtime context: user preferences, active resume ID, target job boards, etc.",
    )

    # -----------------------------------------------------------------------
    # Status Machine
    # -----------------------------------------------------------------------

    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default=AgentRunStatus.PENDING,
        server_default=AgentRunStatus.PENDING,
        index=True,
        comment="pending | running | completed | failed | cancelled | timed_out.",
    )

    # -----------------------------------------------------------------------
    # Timing
    # -----------------------------------------------------------------------

    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
        comment="UTC timestamp when execution actually began (after queuing).",
    )

    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="UTC timestamp when execution finished (success or failure).",
    )

    duration_ms: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="Total execution time in milliseconds.",
    )

    queue_wait_ms: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="Time spent waiting in the Celery queue before execution started.",
    )

    # -----------------------------------------------------------------------
    # Step-by-Step Execution Log
    # -----------------------------------------------------------------------

    steps: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
        server_default="[]",
        comment=(
            "Ordered step-by-step log: "
            "[{ step_name: str, status: str, started_at: str, "
            "   completed_at: str, duration_ms: int, "
            "   input_summary: str, output_summary: str, error: str | null }]."
        ),
    )

    current_step: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
        comment="Name of the step currently being executed (useful for progress display).",
    )

    total_steps: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="Total number of steps planned for this run.",
    )

    completed_steps: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default="0",
        comment="Number of steps completed so far.",
    )

    # -----------------------------------------------------------------------
    # LLM Token Usage & Cost Tracking
    # -----------------------------------------------------------------------

    prompt_tokens: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default="0",
        comment="Total prompt tokens consumed across all LLM calls in this run.",
    )

    completion_tokens: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default="0",
        comment="Total completion tokens generated across all LLM calls.",
    )

    total_tokens: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default="0",
        comment="prompt_tokens + completion_tokens.",
    )

    llm_calls: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default="0",
        comment="Number of individual LLM API calls made.",
    )

    estimated_cost_usd: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment="Estimated API cost in USD based on token usage and model pricing.",
    )

    models_used: Mapped[list[str]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
        server_default="[]",
        comment="List of LLM model names used in this run.",
    )

    # -----------------------------------------------------------------------
    # Tool Call Tracking
    # -----------------------------------------------------------------------

    tool_calls: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
        server_default="[]",
        comment=(
            "Tool invocations made by the agent: "
            "[{ tool_name: str, called_at: str, duration_ms: int, "
            "   success: bool, error: str | null }]."
        ),
    )

    tool_calls_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default="0",
        comment="Total number of tool invocations.",
    )

    # -----------------------------------------------------------------------
    # Output & Results
    # -----------------------------------------------------------------------

    output_payload: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        server_default="{}",
        comment=(
            "Structured results produced by the agent. "
            "Schema varies by agent: e.g. for discovery_agent: {jobs_found: int, jobs_embedded: int}; "
            "for application_agent: {application_id: str, submitted: bool}."
        ),
    )

    artifacts_created: Mapped[list[str]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
        server_default="[]",
        comment="IDs of DB records created by this run, e.g. [resume_id, cover_letter_id].",
    )

    # -----------------------------------------------------------------------
    # Error Handling
    # -----------------------------------------------------------------------

    error_message: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="Human-readable error message if the run failed.",
    )

    error_type: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
        comment="Exception class name, e.g. 'AIServiceException' or 'PlaywrightTimeoutError'.",
    )

    error_traceback: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="Full Python traceback for debugging (truncated to 10k chars).",
    )

    retry_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default="0",
        comment="Number of times this run has been retried.",
    )

    max_retries: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=3,
        server_default="3",
        comment="Maximum retries allowed before marking as permanently failed.",
    )

    is_retryable: Mapped[bool] = mapped_column(
        JSONB,  # Using JSONB for portability — stores boolean
        nullable=False,
        default=True,
        server_default="true",
        comment="Whether this run can be retried on failure.",
    )

    # -----------------------------------------------------------------------
    # Performance Metadata
    # -----------------------------------------------------------------------

    memory_usage_mb: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment="Peak memory usage during this run in MB (recorded by the worker).",
    )

    scraping_requests_made: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default="0",
        comment="Number of HTTP/browser requests made by scrapers or Playwright.",
    )

    # -----------------------------------------------------------------------
    # Relationships
    # -----------------------------------------------------------------------

    user: Mapped["User"] = relationship(
        "User",
        back_populates="agent_runs",
        lazy="select",
    )

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def add_step(
        self,
        step_name: str,
        status: str,
        *,
        duration_ms: int | None = None,
        input_summary: str | None = None,
        output_summary: str | None = None,
        error: str | None = None,
    ) -> None:
        """Append a step record to the execution log."""
        from datetime import timezone
        step: dict[str, Any] = {
            "step_name": step_name,
            "status": status,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        if duration_ms is not None:
            step["duration_ms"] = duration_ms
        if input_summary:
            step["input_summary"] = input_summary
        if output_summary:
            step["output_summary"] = output_summary
        if error:
            step["error"] = error
        steps = list(self.steps)
        steps.append(step)
        self.steps = steps
        self.completed_steps = len([s for s in steps if s.get("status") == "completed"])

    def add_tool_call(self, tool_name: str, *, duration_ms: int, success: bool, error: str | None = None) -> None:
        from datetime import timezone
        calls = list(self.tool_calls)
        calls.append({
            "tool_name": tool_name,
            "called_at": datetime.now(timezone.utc).isoformat(),
            "duration_ms": duration_ms,
            "success": success,
            "error": error,
        })
        self.tool_calls = calls
        self.tool_calls_count = len(calls)

    def add_token_usage(self, prompt: int, completion: int, model: str) -> None:
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.total_tokens = self.prompt_tokens + self.completion_tokens
        self.llm_calls += 1
        models = list(self.models_used)
        if model not in models:
            models.append(model)
            self.models_used = models

    @property
    def progress_percent(self) -> float:
        if not self.total_steps:
            return 0.0
        return round(self.completed_steps / self.total_steps * 100, 1)

    @property
    def is_running(self) -> bool:
        return self.status == AgentRunStatus.RUNNING

    @property
    def is_done(self) -> bool:
        return self.status in (
            AgentRunStatus.COMPLETED,
            AgentRunStatus.FAILED,
            AgentRunStatus.CANCELLED,
            AgentRunStatus.TIMED_OUT,
        )

    def __repr__(self) -> str:
        return (
            f"<AgentRun id={self.id} agent={self.agent_name!r} "
            f"status={self.status!r} tokens={self.total_tokens}>"
        )