"""
app/workflows/linkedin_workflow.py
=====================================
LangGraph workflow for AI-generated LinkedIn content: plan -> research ->
draft -> score -> (approval gate) -> publish.

Graph topology (single-post mode — the common case):

                       research_topic
                            |
                       draft_post  <-------------------+
                            |                            | needs_revision
                       score_quality ----------------------+
                            |
              (max 1 revision pass, then proceeds regardless of score)
                            |
                  requires_approval AND NOT is_approved ?
                       /                              \
                     yes                                no
                      |                                  |
              wait_for_approval                  publish_or_schedule
              (graph pauses here                          |
               via checkpointer,                         END
               resumed by
               POST /linkedin/
               posts/{id}/approve)
                      |
              publish_or_schedule
                      |
                     END

Calendar-planning mode (generate_daily_pipeline) wraps this same per-post
subgraph: the Celery task layer (workers/linkedin_tasks.py) fans out N
independent invocations of run_post_generation_workflow, one per day in the
requested window, each producing its own LinkedInPost draft row. The
per-day fan-out happens at the Celery level rather than inside this graph
so each day's post is independently retryable — a rate-limit failure on
day 3 doesn't take down days 1-7.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from langgraph.graph import StateGraph, END

from app.core.constants import (
    NODE_ERROR,
    MAX_LINKEDIN_POST_CHARS,
    LINKEDIN_POST_HASHTAG_LIMIT,
)
from app.core.logging import get_logger
from app.workflows.graph import (
    node,
    build_graph,
    halt_on_error,
    approval_gate_router,
    get_invoke_config,
)
from app.workflows.state import LinkedInWorkflowState, new_base_state

logger = get_logger(__name__)

NODE_RESEARCH = "research_topic"
NODE_DRAFT = "draft_post"
NODE_SCORE = "score_quality"
NODE_WAIT_APPROVAL = "wait_for_approval"
NODE_PUBLISH = "publish_or_schedule"

QUALITY_SCORE_THRESHOLD = 65.0
MAX_REVISION_ATTEMPTS = 1


# ---------------------------------------------------------------------------
# Node implementations
# ---------------------------------------------------------------------------

@node(NODE_RESEARCH)
async def research_topic_node(state: LinkedInWorkflowState) -> dict[str, Any]:
    """
    Gather supporting facts, statistics, or recent news for the topic via
    the linkedin_agent's web-search-backed research tool.

    Skipped (pass-through) on a revision-loop re-entry — the @node decorator
    re-runs this node every time the graph re-enters it, but research from
    the first pass remains valid; only the draft needs reworking based on
    score_quality's feedback, so we short-circuit if findings already exist.
    """
    if state.get("research_findings") is not None:
        return {
            "research_findings": state["research_findings"],
            "research_citations": state.get("research_citations", []),
        }

    from app.agents.linkedin_agent.agent import LinkedInAgent

    agent = LinkedInAgent()
    result = await agent.research_topic(
        topic=state["topic"],
        category=state.get("category"),
        source_urls=state.get("source_urls", []),
    )

    return {
        "research_findings": result["findings"],
        "research_citations": result.get("citations", []),
        "total_llm_calls": result["usage"].get("llm_calls", 0),
        "total_prompt_tokens": result["usage"].get("prompt_tokens", 0),
        "total_completion_tokens": result["usage"].get("completion_tokens", 0),
    }


@node(NODE_DRAFT)
async def draft_post_node(state: LinkedInWorkflowState) -> dict[str, Any]:
    """
    Write the actual post: hook, body, hashtags, and call-to-action, in the
    requested tone, grounded in research_findings.

    On a revision loop (entered a second time after score_quality flagged
    needs_revision=True), quality_feedback from the prior attempt is passed
    back into the agent so the rewrite specifically addresses the flagged
    issues rather than generating a fresh, equally-flawed draft.
    """
    from app.agents.linkedin_agent.agent import LinkedInAgent

    agent = LinkedInAgent()
    meta = state.get("metadata", {})

    result = await agent.draft_post(
        topic=state["topic"],
        category=state.get("category", "ai_insights"),
        tone=state.get("tone", "thought-leader"),
        research_findings=state.get("research_findings", []),
        max_length=min(meta.get("max_length", 1500), MAX_LINKEDIN_POST_CHARS),
        include_hashtags=meta.get("include_hashtags", True),
        include_cta=meta.get("include_cta", True),
        custom_instructions=meta.get("custom_instructions"),
        revision_feedback=state.get("quality_feedback") if state.get("needs_revision") else None,
    )

    return {
        "draft_content": result["content"],
        "draft_hook": result.get("hook"),
        "draft_hashtags": result.get("hashtags", [])[:LINKEDIN_POST_HASHTAG_LIMIT],
        "draft_cta": result.get("call_to_action"),
        "needs_revision": False,  # reset — score_quality re-evaluates fresh
        "total_llm_calls": result["usage"].get("llm_calls", 0),
        "total_prompt_tokens": result["usage"].get("prompt_tokens", 0),
        "total_completion_tokens": result["usage"].get("completion_tokens", 0),
    }


@node(NODE_SCORE)
async def score_quality_node(state: LinkedInWorkflowState) -> dict[str, Any]:
    """
    Score the draft for engagement potential: hook strength, specificity,
    readability, hashtag relevance, and length appropriateness.

    Flags needs_revision=True at most once, bounded by retry_count against
    MAX_REVISION_ATTEMPTS, so a stubbornly low-scoring topic can never loop
    research->draft->score forever — after one revision pass the post
    proceeds regardless of score and the pipeline always terminates.
    """
    from app.agents.linkedin_agent.agent import LinkedInAgent

    agent = LinkedInAgent()
    result = await agent.score_post_quality(
        content=state["draft_content"],
        hook=state.get("draft_hook"),
        category=state.get("category", "ai_insights"),
    )

    current_retry = state.get("retry_count", 0)
    is_low_quality = result["score"] < QUALITY_SCORE_THRESHOLD
    should_revise = is_low_quality and current_retry < MAX_REVISION_ATTEMPTS

    return {
        "quality_score": result["score"],
        "quality_feedback": result.get("feedback", []),
        "needs_revision": should_revise,
        "retry_count": current_retry + 1 if should_revise else current_retry,
    }


async def wait_for_approval_node(state: LinkedInWorkflowState) -> dict[str, Any]:
    """
    Pause point for posts requiring manual approval before publishing.

    Deliberately NOT wrapped in @node — this is a checkpoint target, not a
    unit of work with its own success/failure semantics. The compiled
    graph's checkpointer persists state here; execution naturally stalls
    because approval_gate_router sends unapproved state to END rather than
    looping, leaving the run resumable rather than busy-waiting.

    api/v1/linkedin.py:approve_post re-invokes the compiled graph with the
    SAME workflow_id (thread_id) via resume_after_approval below, setting
    is_approved=True in the resumed state, which routes past this node to
    NODE_PUBLISH on the next pass through approval_gate_router.
    """
    logger.info(
        "Post paused for approval",
        agent_run_id=state.get("agent_run_id"),
        post_id=state.get("post_id"),
    )

    if not state.get("post_id"):
        from app.repositories.linkedin_repository import LinkedInRepository
        from app.db.session import get_db_context
        import uuid

        async with get_db_context() as db:
            repo = LinkedInRepository(db)
            post = await repo.create_draft(
                user_id=uuid.UUID(state["user_id"]),
                content=state["draft_content"],
                hook=state.get("draft_hook"),
                hashtags=state.get("draft_hashtags", []),
                call_to_action=state.get("draft_cta"),
                category=state.get("category", "ai_insights"),
                tone=state.get("tone"),
                status="pending_approval",
                requires_approval=True,
            )
            await db.commit()
        return {"post_id": str(post.id), "publish_status": "pending_approval"}

    return {"publish_status": "pending_approval"}


@node(NODE_PUBLISH)
async def publish_or_schedule_node(state: LinkedInWorkflowState) -> dict[str, Any]:
    """
    Final stage: either save as a publish-ready draft or schedule for the
    requested time, persisting (or updating, if this is a post-approval
    resume) the LinkedInPost row.

    Note: this node prepares the post record but does not call the actual
    LinkedIn publish API — that happens via the explicit
    POST /linkedin/posts/{id}/publish endpoint (api/v1/linkedin.py), which
    gives the user a final manual trigger point even for non-approval-gated
    posts, rather than auto-publishing the instant generation finishes.
    """
    from app.repositories.linkedin_repository import LinkedInRepository
    from app.db.session import get_db_context
    import uuid

    schedule_time = state.get("metadata", {}).get("scheduled_at")

    async with get_db_context() as db:
        repo = LinkedInRepository(db)

        if state.get("post_id"):
            post = await repo.get_by_id(uuid.UUID(state["post_id"]))
            post.content = state["draft_content"]
            post.hook = state.get("draft_hook")
            post.hashtags = state.get("draft_hashtags", [])
            post.call_to_action = state.get("draft_cta")
        else:
            post = await repo.create_draft(
                user_id=uuid.UUID(state["user_id"]),
                content=state["draft_content"],
                hook=state.get("draft_hook"),
                hashtags=state.get("draft_hashtags", []),
                call_to_action=state.get("draft_cta"),
                category=state.get("category", "ai_insights"),
                tone=state.get("tone"),
                status="draft",
                requires_approval=False,
            )

        post.character_count = len(state["draft_content"])
        post.word_count = len(state["draft_content"].split())
        post.source_urls = state.get("research_citations", [])
        post.generation_prompt = state.get("metadata", {}).get("generation_prompt")
        post.generation_model = state.get("metadata", {}).get("generation_model")
        post.generation_tokens_used = (
            state.get("total_prompt_tokens", 0) + state.get("total_completion_tokens", 0)
        )

        if schedule_time:
            post.scheduled_at = datetime.fromisoformat(schedule_time)
            post.status = "scheduled"
            publish_status = "scheduled"
        else:
            post.status = "draft"
            publish_status = "draft"

        await db.commit()
        post_id = str(post.id)

    logger.info("Post finalized", post_id=post_id, status=publish_status)

    return {"post_id": post_id, "publish_status": publish_status}


async def handle_error_node(state: LinkedInWorkflowState) -> dict[str, Any]:
    """
    Terminal error node. Logs the failure; deliberately does NOT notify the
    user by default since a single failed post-generation attempt is
    low-severity — they can simply retry from the dashboard. Repeated
    failures across many days (calendar mode) are surfaced via the
    AgentRun status in the dashboard rather than per-post notifications,
    to avoid spamming the user with N failure emails for an N-day pipeline.
    """
    errors = state.get("errors", [])
    logger.error(
        "LinkedIn workflow halted with errors",
        agent_run_id=state.get("agent_run_id"),
        topic=state.get("topic"),
        errors=errors,
    )
    return {"status": "failed", "publish_status": "failed"}


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def _route_after_score(state: LinkedInWorkflowState) -> str:
    """
    Three-way branch after scoring: error / revise / proceed-to-approval-check.

    Kept as a named function (rather than an inline lambda) so the routing
    logic is unit-testable in isolation and visible in stack traces if a
    routing bug ever needs debugging.
    """
    if state.get("should_halt"):
        return NODE_ERROR
    if state.get("needs_revision"):
        return NODE_DRAFT
    if state.get("requires_approval") and not state.get("is_approved"):
        return NODE_WAIT_APPROVAL
    return NODE_PUBLISH


async def build_linkedin_graph():
    """
    Construct and compile the LinkedIn content generation StateGraph.

    Approval pausing uses LangGraph's interrupt_before mechanism rather than
    a routing trick: the graph genuinely halts execution before NODE_PUBLISH
    runs, with the checkpointer persisting state at exactly that point.
    Re-invoking ainvoke() with the same thread_id and no new input resumes
    from there and proceeds into NODE_PUBLISH — this is the documented
    LangGraph pattern for human-in-the-loop pauses, and is what makes
    resume_after_approval below actually resume rather than re-run the
    graph from scratch.

    wait_for_approval_node itself still exists as a real node (not bypassed)
    so it can persist the draft-pending-approval LinkedInPost row before the
    interrupt fires — the interrupt is configured to fire AFTER this node
    runs and BEFORE NODE_PUBLISH, via interrupt_before=[NODE_PUBLISH]
    combined with routing that sends unapproved state to NODE_WAIT_APPROVAL
    only (never reaching NODE_PUBLISH in the same invocation).
    """
    graph = StateGraph(LinkedInWorkflowState)

    graph.add_node(NODE_RESEARCH, research_topic_node)
    graph.add_node(NODE_DRAFT, draft_post_node)
    graph.add_node(NODE_SCORE, score_quality_node)
    graph.add_node(NODE_WAIT_APPROVAL, wait_for_approval_node)
    graph.add_node(NODE_PUBLISH, publish_or_schedule_node)
    graph.add_node(NODE_ERROR, handle_error_node)

    graph.set_entry_point(NODE_RESEARCH)

    graph.add_conditional_edges(NODE_RESEARCH, halt_on_error(success_node=NODE_DRAFT))
    graph.add_conditional_edges(NODE_DRAFT, halt_on_error(success_node=NODE_SCORE))
    graph.add_conditional_edges(NODE_SCORE, _route_after_score)

    # Both branches converge on NODE_PUBLISH as their only forward edge.
    # On the initial run, _route_after_score sends approval-required posts
    # to NODE_WAIT_APPROVAL, which writes the draft row and then the graph
    # is invoked with interrupt_before=[NODE_PUBLISH], so it stops right
    # here without running NODE_PUBLISH. The approved branch from
    # _route_after_score skips NODE_WAIT_APPROVAL entirely and proceeds
    # straight through to NODE_PUBLISH uninterrupted in the same call.
    graph.add_edge(NODE_WAIT_APPROVAL, NODE_PUBLISH)
    graph.add_conditional_edges(NODE_PUBLISH, halt_on_error(success_node=END))

    graph.add_edge(NODE_ERROR, END)

    return await build_graph(graph, interrupt_before=[NODE_PUBLISH])


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

async def run_post_generation_workflow(
    *,
    user_id: str,
    agent_run_id: str,
    topic: str,
    category: str,
    tone: str,
    requires_approval: bool,
    include_hashtags: bool = True,
    include_cta: bool = True,
    max_length: int = 1500,
    source_urls: list[str] | None = None,
    custom_instructions: str | None = None,
    scheduled_at: str | None = None,
) -> dict[str, Any]:
    """
    Run the single-post generation pipeline end to end.

    Called from services/linkedin_service.py:generate_post (synchronous
    single-post path) and from workers/linkedin_tasks.py per-day fan-out
    in the calendar-planning flow.
    """
    compiled = await build_linkedin_graph()
    workflow_id = f"linkedin-post-{agent_run_id}"

    initial_state: dict[str, Any] = {
        **new_base_state(
            user_id=user_id,
            agent_run_id=agent_run_id,
            workflow_id=workflow_id,
        ),
        "topic": topic,
        "category": category,
        "tone": tone,
        "source_urls": source_urls or [],
        "requires_approval": requires_approval,
        "is_approved": False,
        "metadata": {
            "max_length": max_length,
            "include_hashtags": include_hashtags,
            "include_cta": include_cta,
            "custom_instructions": custom_instructions,
            "scheduled_at": scheduled_at,
            "generation_model": None,
        },
    }

    config = get_invoke_config(workflow_id=workflow_id)
    return await compiled.ainvoke(initial_state, config=config)


async def resume_after_approval(
    *,
    workflow_id: str,
    post_id: str,
) -> dict[str, Any]:
    """
    Resume a workflow paused at wait_for_approval after the user approves
    the draft via POST /linkedin/posts/{id}/approve.

    Re-invokes the SAME compiled graph with the SAME thread_id (workflow_id)
    so the checkpointer restores the exact prior state; the partial state
    dict passed here merges is_approved=True on top of it, which
    _route_after_score's NODE_WAIT_APPROVAL branch was waiting on, sending
    execution to NODE_PUBLISH.
    """
    compiled = await build_linkedin_graph()
    config = get_invoke_config(workflow_id=workflow_id)

    resume_state = {"is_approved": True, "post_id": post_id}
    return await compiled.ainvoke(resume_state, config=config)