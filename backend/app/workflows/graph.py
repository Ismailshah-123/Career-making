"""
app/workflows/graph.py
========================
Shared LangGraph construction utilities used by every workflow module.

Provides:
- get_checkpointer()        : Async Postgres-backed checkpointer for resumable runs
- node()                    : Decorator wrapping every node function with
                               timing, AgentRun.steps logging, and error capture
- conditional_router()      : Factory for building edge-routing functions that
                               read state["should_halt"] / state["errors"] uniformly
- build_graph()              : Thin wrapper around StateGraph().compile() that
                               attaches the checkpointer and a recursion limit
- halt_on_error()            : Standard conditional edge — routes to "handle_error"
                               whenever a node sets should_halt=True

Why this file exists separately from each *_workflow.py:
Every one of the 4 workflow graphs (job, resume, linkedin, application) needs
identical plumbing — checkpointing, per-node telemetry, and the same
halt-on-error routing pattern. Without this module, that plumbing would be
copy-pasted 4 times and drift out of sync. Workflow files import from here
and focus purely on their own node logic and graph topology.
"""

from __future__ import annotations

import functools
import time
import traceback
from typing import Any, Awaitable, Callable, TypeVar

from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.core.config import settings
from app.core.constants import NODE_ERROR, NODE_END
from app.core.logging import get_logger

logger = get_logger(__name__)

StateT = TypeVar("StateT", bound=dict)
NodeFn = Callable[[StateT], Awaitable[dict[str, Any]]]

# ---------------------------------------------------------------------------
# Checkpointer — persists graph state to Postgres so long-running workflows
# (e.g. a LinkedIn post sitting at "wait_for_approval" for hours) survive
# worker restarts and can be resumed from the exact node they paused at.
# ---------------------------------------------------------------------------

_checkpointer_instance: Any = None


async def get_checkpointer() -> Any:
    """
    Return a shared AsyncPostgresSaver checkpointer instance.

    Lazily initialised on first call and cached for the process lifetime.
    Falls back to an in-memory checkpointer (no persistence across restarts)
    if langgraph-checkpoint-postgres is not installed — logged as a warning
    since this degrades resumability but should never crash the app.
    """
    global _checkpointer_instance
    if _checkpointer_instance is not None:
        return _checkpointer_instance

    try:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

        _checkpointer_instance = AsyncPostgresSaver.from_conn_string(
            settings.DATABASE_URL_SYNC.replace("postgresql+psycopg2", "postgresql")
        )
        # Ensure the checkpoint tables exist (idempotent)
        async with _checkpointer_instance as cp:
            await cp.setup()
        logger.info("LangGraph Postgres checkpointer initialised.")
    except ImportError:
        from langgraph.checkpoint.memory import MemorySaver

        logger.warning(
            "langgraph-checkpoint-postgres not installed — using in-memory "
            "checkpointer. Workflow state will NOT survive worker restarts."
        )
        _checkpointer_instance = MemorySaver()
    except Exception as exc:
        from langgraph.checkpoint.memory import MemorySaver

        logger.error(
            "Postgres checkpointer setup failed — falling back to in-memory.",
            error=str(exc),
            exc_info=True,
        )
        _checkpointer_instance = MemorySaver()

    return _checkpointer_instance


# ---------------------------------------------------------------------------
# Node decorator — wraps every node function with consistent instrumentation
# ---------------------------------------------------------------------------

def node(node_name: str) -> Callable[[NodeFn], NodeFn]:
    """
    Decorator applied to every LangGraph node function across all workflows.

    Wraps the node with:
    1. Timing — measures wall-clock duration
    2. Structured logging — one log line per node entry/exit
    3. completed_nodes tracking — appends node_name so the audit trail in
       AgentRun.steps reflects exactly which nodes ran, in order
    4. Error capture — any exception raised inside the node is caught,
       converted into a state["errors"] entry, and should_halt is set
       True instead of letting the exception propagate and kill the
       entire Celery task ungracefully. The router then sends execution
       to the "handle_error" node for cleanup/notification.

    Usage:
        @node("discover_jobs")
        async def discover_jobs_node(state: JobWorkflowState) -> dict:
            ...
            return {"discovered_job_ids": [...]}
    """

    def decorator(fn: NodeFn) -> NodeFn:
        @functools.wraps(fn)
        async def wrapper(state: StateT) -> dict[str, Any]:
            run_id = state.get("agent_run_id", "unknown")
            t0 = time.perf_counter()

            logger.info(
                f"Node started: {node_name}",
                agent_run_id=run_id,
                node=node_name,
            )

            # If a prior node already requested a halt, short-circuit
            # immediately without executing this node's body at all.
            if state.get("should_halt"):
                return {"current_node": node_name}

            try:
                result = await fn(state)
                duration_ms = round((time.perf_counter() - t0) * 1000, 1)

                logger.info(
                    f"Node completed: {node_name}",
                    agent_run_id=run_id,
                    node=node_name,
                    duration_ms=duration_ms,
                )

                # Always stamp current_node and append to completed_nodes,
                # merging in whatever the node itself returned.
                return {
                    **result,
                    "current_node": node_name,
                    "completed_nodes": [node_name],
                }

            except Exception as exc:
                duration_ms = round((time.perf_counter() - t0) * 1000, 1)
                tb = traceback.format_exc()

                logger.error(
                    f"Node failed: {node_name}",
                    agent_run_id=run_id,
                    node=node_name,
                    duration_ms=duration_ms,
                    error=str(exc),
                    exc_info=True,
                )

                error_entry = {
                    "node": node_name,
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": tb[-4000:],  # cap traceback length for storage
                    "duration_ms": duration_ms,
                }

                return {
                    "current_node": node_name,
                    "completed_nodes": [node_name],
                    "errors": [error_entry],
                    "should_halt": True,
                    "status": "failed",
                }

        return wrapper

    return decorator


# ---------------------------------------------------------------------------
# Conditional routing helpers
# ---------------------------------------------------------------------------

def halt_on_error(success_node: str, error_node: str = NODE_ERROR) -> Callable[[StateT], str]:
    """
    Standard conditional-edge function: routes to error_node if the previous
    node set should_halt=True, otherwise continues to success_node.

    Usage in a workflow file:
        graph.add_conditional_edges(
            "discover_jobs",
            halt_on_error(success_node="filter_jobs"),
        )
    """

    def _router(state: StateT) -> str:
        if state.get("should_halt"):
            return error_node
        return success_node

    return _router


def conditional_router(
    condition_key: str,
    routes: dict[Any, str],
    default: str,
    *,
    error_node: str = NODE_ERROR,
) -> Callable[[StateT], str]:
    """
    General-purpose conditional edge factory for branching on any state field.

    Checks should_halt FIRST (always takes priority over business-logic
    branching), then looks up state[condition_key] in `routes`, falling back
    to `default` if the value isn't found.

    Usage:
        graph.add_conditional_edges(
            "score_quality",
            conditional_router(
                condition_key="needs_revision",
                routes={True: "draft_post", False: "wait_for_approval"},
                default="wait_for_approval",
            ),
        )
    """

    def _router(state: StateT) -> str:
        if state.get("should_halt"):
            return error_node
        value = state.get(condition_key)
        return routes.get(value, default)

    return _router


def approval_gate_router(
    approval_key: str = "requires_approval",
    approved_key: str = "is_approved",
    *,
    approved_node: str,
    pending_node: str,
    error_node: str = NODE_ERROR,
) -> Callable[[StateT], str]:
    """
    Specialised router for the human-approval gate pattern used in the
    LinkedIn workflow: if a post doesn't require approval, or it does and
    has already been approved, proceed; otherwise route to a node that
    pauses the graph (relying on the checkpointer) until approval arrives.
    """

    def _router(state: StateT) -> str:
        if state.get("should_halt"):
            return error_node
        if not state.get(approval_key):
            return approved_node
        if state.get(approved_key):
            return approved_node
        return pending_node

    return _router


# ---------------------------------------------------------------------------
# Graph compilation wrapper
# ---------------------------------------------------------------------------

DEFAULT_RECURSION_LIMIT = 50


async def build_graph(
    graph: StateGraph,
    *,
    use_checkpointer: bool = True,
    interrupt_before: list[str] | None = None,
) -> CompiledStateGraph:
    """
    Compile a StateGraph with the shared Postgres checkpointer attached.

    Every workflow module calls this instead of graph.compile() directly,
    so checkpointing behaviour stays consistent and is configured in one
    place. Set use_checkpointer=False for short-lived sub-graphs that don't
    need resumability (rare — most graphs should checkpoint).

    interrupt_before: list of node names where execution halts BEFORE the
    named node runs — the checkpointer persists state at that point, and
    ainvoke() with the same thread_id resumes execution from there on the
    next call. Used by linkedin_workflow for the human-approval pause.
    """
    compile_kwargs: dict[str, Any] = {}
    if interrupt_before:
        compile_kwargs["interrupt_before"] = interrupt_before

    if use_checkpointer:
        checkpointer = await get_checkpointer()
        return graph.compile(checkpointer=checkpointer, **compile_kwargs)
    return graph.compile(**compile_kwargs)


def get_invoke_config(
    *,
    workflow_id: str,
    recursion_limit: int = DEFAULT_RECURSION_LIMIT,
) -> dict[str, Any]:
    """
    Build the config dict passed to compiled_graph.ainvoke(state, config=...).

    `thread_id` (mapped from workflow_id) is what LangGraph's checkpointer
    uses as the persistence key — invoking with the same thread_id resumes
    an existing run instead of starting fresh, which is how the LinkedIn
    approval-gate pause/resume pattern works in practice.
    """
    return {
        "configurable": {"thread_id": workflow_id},
        "recursion_limit": recursion_limit,
    }