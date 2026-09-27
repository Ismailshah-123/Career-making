"""
app/workflows/resume_workflow.py
===================================
LangGraph workflow for resume processing: upload -> parse -> (tailor) -> score -> embed.

Graph topology:

                      extract_text
                           |
                      parse_sections
                           |
                  +--------+--------+
                  | is_tailoring     | is_tailoring
                  | == False         | == True
                  v                  v
            compute_ats_score   tailor_sections
                  |                  |
                  |             compute_ats_score
                  |                  |
                  +--------+---------+
                           |
                    generate_embedding
                           |
                          END

Two entry modes compiled from the same graph:
1. Initial parse — a freshly uploaded master resume goes through extraction,
   parsing, scoring, and embedding. is_tailoring=False.
2. Tailor-for-job — an existing parsed master resume is rewritten to better
   match a specific job, producing a NEW Resume row (is_master=False), then
   that new variant is scored and embedded independently. is_tailoring=True.

Both modes share extract_text/parse_sections/compute_ats_score/
generate_embedding so there's exactly one implementation of each stage to
maintain — the tailoring path just inserts tailor_sections in between
parsing and scoring, and writes to new_resume_id instead of resume_id.
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import StateGraph, END

from app.core.constants import NODE_ERROR
from app.core.logging import get_logger
from app.workflows.graph import node, build_graph, halt_on_error, get_invoke_config
from app.workflows.state import ResumeWorkflowState, new_base_state

logger = get_logger(__name__)

# Node name constants local to this workflow (not in the global constants.py
# NODE_* set since they're resume-pipeline-specific, not shared across graphs)
NODE_EXTRACT = "extract_text"
NODE_PARSE = "parse_sections"
NODE_TAILOR = "tailor_sections"
NODE_SCORE = "compute_ats_score"
NODE_EMBED = "generate_embedding"


# ---------------------------------------------------------------------------
# Node implementations
# ---------------------------------------------------------------------------

@node(NODE_EXTRACT)
async def extract_text_node(state: ResumeWorkflowState) -> dict[str, Any]:
    """
    Pull raw text out of the uploaded file (PDF/DOCX/TXT/ODT).

    In tailoring mode (is_tailoring=True), the caller has already extracted
    text from the master resume at upload time and seeds it directly into
    initial_state — this node detects that pre-seeded raw_text and passes
    it through unchanged instead of re-reading a file_path that won't exist
    for a tailor-for-job run (there is no new upload, only a re-write of
    already-parsed content).

    For a real upload (is_tailoring=False), delegates to file-format-specific
    extractors in utils/file_utils.py. A failed extraction (corrupt file,
    password-protected PDF, scanned image with no text layer) raises
    ResumeParseException, which the @node decorator's exception handling
    converts into a halted state — no manual halt needed here.
    """
    if state.get("is_tailoring") and state.get("raw_text"):
        existing_text = state["raw_text"]
        return {
            "raw_text": existing_text,
            "word_count": state.get("word_count") or len(existing_text.split()),
        }

    from app.utils.file_utils import extract_text_from_file

    raw_text = await extract_text_from_file(
        file_path=state["file_path"],
        mime_type=state["mime_type"],
    )

    if not raw_text or len(raw_text.strip()) < 50:
        from app.core.exceptions import ResumeParseException
        raise ResumeParseException(
            filename=state["file_path"],
            reason="Extracted text is too short — file may be image-only or corrupted.",
        )

    return {
        "raw_text": raw_text,
        "word_count": len(raw_text.split()),
    }


@node(NODE_PARSE)
async def parse_sections_node(state: ResumeWorkflowState) -> dict[str, Any]:
    """
    Invoke the resume_agent's structured extraction tool to turn raw text
    into the parsed_sections schema (contact, summary, experience, education,
    skills, projects, certifications, languages).

    In tailoring mode, parsed_sections is already seeded from the master
    resume row by the caller (the master was already parsed at upload
    time) — this node passes it through unchanged rather than burning an
    LLM call to re-derive identical output. This mirrors extract_text_node's
    pass-through behaviour for raw_text in the same mode.

    For a real initial parse, this is the first LLM call in the pipeline —
    token usage is tracked on the returned state delta so it accumulates
    correctly via the operator.add reducer on total_prompt_tokens /
    total_completion_tokens.
    """
    if state.get("is_tailoring") and state.get("parsed_sections"):
        return {
            "parsed_sections": state["parsed_sections"],
            "extracted_skills": state.get("extracted_skills", []),
            "extracted_job_titles": state.get("extracted_job_titles", []),
            "years_of_experience": state.get("years_of_experience"),
        }

    from app.agents.resume_agent.agent import ResumeAgent

    agent = ResumeAgent()
    result = await agent.parse_raw_text(raw_text=state["raw_text"])

    return {
        "parsed_sections": result["sections"],
        "extracted_skills": result["skills"],
        "extracted_job_titles": result["job_titles"],
        "years_of_experience": result.get("years_of_experience"),
        "total_prompt_tokens": result["usage"]["prompt_tokens"],
        "total_completion_tokens": result["usage"]["completion_tokens"],
        "total_llm_calls": 1,
    }


@node(NODE_TAILOR)
async def tailor_sections_node(state: ResumeWorkflowState) -> dict[str, Any]:
    """
    Tailoring-only stage: rewrite experience bullets, summary, and skill
    ordering to align with state['target_job_id']'s requirements, while
    preserving factual accuracy (no fabricated experience).

    Creates a NEW Resume DB row immediately (is_master=False,
    parent_resume_id=master.id) so downstream scoring/embedding nodes
    operate on new_resume_id rather than mutating the master.
    """
    from app.agents.resume_agent.agent import ResumeAgent
    from app.repositories.resume_repository import ResumeRepository
    from app.db.session import get_db_context
    import uuid

    agent = ResumeAgent()
    result = await agent.tailor_sections(
        original_sections=state["parsed_sections"],
        job_id=uuid.UUID(state["target_job_id"]),
        tone=state.get("tailor_tone", "professional"),
        emphasise_skills=state.get("emphasise_skills", []),
    )

    async with get_db_context() as db:
        repo = ResumeRepository(db)
        new_resume = await repo.create_tailored_variant(
            master_resume_id=uuid.UUID(state["resume_id"]),
            tailored_sections=result["sections"],
            tailoring_changes=result["changes_summary"],
            tailored_for_job_id=uuid.UUID(state["target_job_id"]),
            tailoring_prompt_used=result.get("prompt_used"),
        )
        await db.commit()

    return {
        "tailored_sections": result["sections"],
        "tailoring_changes": result["changes_summary"],
        "new_resume_id": str(new_resume.id),
        "total_prompt_tokens": result["usage"]["prompt_tokens"],
        "total_completion_tokens": result["usage"]["completion_tokens"],
        "total_llm_calls": 1,
    }


@node(NODE_SCORE)
async def compute_ats_score_node(state: ResumeWorkflowState) -> dict[str, Any]:
    """
    Run the deterministic + LLM-assisted ATS compatibility scorer.

    Operates on whichever resume is "current" for this run: new_resume_id
    if tailoring just happened, otherwise resume_id from the initial parse.
    Persists the score directly onto the Resume row since this is the
    canonical place that score gets computed.
    """
    from app.agents.resume_agent.agent import ResumeAgent
    from app.repositories.resume_repository import ResumeRepository
    from app.db.session import get_db_context
    import uuid

    target_resume_id = state.get("new_resume_id") or state["resume_id"]
    sections = state.get("tailored_sections") or state["parsed_sections"]

    agent = ResumeAgent()
    score_result = await agent.score_ats_compatibility(
        sections=sections,
        raw_text=state.get("raw_text", ""),
        job_id=uuid.UUID(state["target_job_id"]) if state.get("target_job_id") else None,
    )

    async with get_db_context() as db:
        repo = ResumeRepository(db)
        resume = await repo.get_by_id(uuid.UUID(target_resume_id))
        resume.ats_score = score_result["ats_score"]
        resume.ats_feedback = score_result["feedback"]
        resume.keyword_density_score = score_result.get("keyword_density_score")
        if not resume.is_parsed:
            resume.is_parsed = True
            resume.parsed_sections = sections
            resume.extracted_skills = state.get("extracted_skills", [])
            resume.extracted_job_titles = state.get("extracted_job_titles", [])
            resume.years_of_experience = state.get("years_of_experience")
            resume.raw_text = state.get("raw_text")
            resume.word_count = state.get("word_count")
        await db.commit()

    return {
        "ats_score": score_result["ats_score"],
        "ats_feedback": score_result["feedback"],
        "keyword_density_score": score_result.get("keyword_density_score"),
    }


@node(NODE_EMBED)
async def generate_embedding_node(state: ResumeWorkflowState) -> dict[str, Any]:
    """
    Final stage: chunk the resume content, embed each chunk via the
    embedding_service, and upsert vectors into the Qdrant resumes collection.

    Marks is_embedded=True on the Resume row so the matching pipeline
    (job_workflow.py:match_jobs_node) knows this resume is ready for
    similarity search.
    """
    from app.services.embedding_service import EmbeddingService
    from app.services.qdrant_service import QdrantService
    from app.repositories.resume_repository import ResumeRepository
    from app.db.session import get_db_context
    import uuid

    target_resume_id = state.get("new_resume_id") or state["resume_id"]
    sections = state.get("tailored_sections") or state.get("parsed_sections", {})

    embedding_svc = EmbeddingService()
    qdrant_svc = QdrantService()

    chunks = embedding_svc.chunk_resume_sections(sections)
    vectors = await embedding_svc.embed_batch([c["text"] for c in chunks])

    point_ids = await qdrant_svc.upsert_resume_chunks(
        resume_id=target_resume_id,
        user_id=state["user_id"],
        chunks=chunks,
        vectors=vectors,
    )

    async with get_db_context() as db:
        repo = ResumeRepository(db)
        resume = await repo.get_by_id(uuid.UUID(target_resume_id))
        resume.is_embedded = True
        resume.embedding_model = embedding_svc.model_name
        resume.qdrant_point_ids = point_ids
        await db.commit()

    return {
        "embedding_vector": vectors[0] if vectors else None,
        "qdrant_point_ids": point_ids,
    }


async def handle_error_node(state: ResumeWorkflowState) -> dict[str, Any]:
    """
    Terminal error node: marks the resume row with the parse error so the
    user sees a clear failure reason in the UI rather than a silently
    stuck "processing" state.
    """
    from app.repositories.resume_repository import ResumeRepository
    from app.db.session import get_db_context
    import uuid

    errors = state.get("errors", [])
    error_summary = "; ".join(e.get("message", "Unknown error") for e in errors[-2:])

    logger.error(
        "Resume workflow halted with errors",
        agent_run_id=state.get("agent_run_id"),
        resume_id=state.get("resume_id"),
        errors=errors,
    )

    target_resume_id = state.get("new_resume_id") or state.get("resume_id")
    if target_resume_id:
        try:
            async with get_db_context() as db:
                repo = ResumeRepository(db)
                resume = await repo.get_by_id(uuid.UUID(target_resume_id))
                if resume:
                    resume.parse_error = error_summary
                    await db.commit()
        except Exception as exc:
            logger.error("Failed to persist parse_error on resume row", error=str(exc))

    return {"status": "failed"}


# ---------------------------------------------------------------------------
# Conditional routing
# ---------------------------------------------------------------------------

def _route_after_parse(state: ResumeWorkflowState) -> str:
    """Branch to tailoring if this is a tailor-for-job run, otherwise score directly."""
    if state.get("should_halt"):
        return NODE_ERROR
    if state.get("is_tailoring"):
        return NODE_TAILOR
    return NODE_SCORE


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

async def build_resume_graph():
    """Construct and compile the resume processing StateGraph."""
    graph = StateGraph(ResumeWorkflowState)

    graph.add_node(NODE_EXTRACT, extract_text_node)
    graph.add_node(NODE_PARSE, parse_sections_node)
    graph.add_node(NODE_TAILOR, tailor_sections_node)
    graph.add_node(NODE_SCORE, compute_ats_score_node)
    graph.add_node(NODE_EMBED, generate_embedding_node)
    graph.add_node(NODE_ERROR, handle_error_node)

    graph.set_entry_point(NODE_EXTRACT)

    graph.add_conditional_edges(NODE_EXTRACT, halt_on_error(success_node=NODE_PARSE))
    graph.add_conditional_edges(NODE_PARSE, _route_after_parse)
    graph.add_conditional_edges(NODE_TAILOR, halt_on_error(success_node=NODE_SCORE))
    graph.add_conditional_edges(NODE_SCORE, halt_on_error(success_node=NODE_EMBED))
    graph.add_conditional_edges(NODE_EMBED, halt_on_error(success_node=END, error_node=NODE_ERROR))

    graph.add_edge(NODE_ERROR, END)

    return await build_graph(graph)


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

async def run_parse_workflow(
    *,
    user_id: str,
    agent_run_id: str,
    resume_id: str,
    file_path: str,
    mime_type: str,
) -> dict[str, Any]:
    """
    Run the initial parse pipeline for a freshly uploaded master resume.

    Called from workers/resume_tasks.py:process_resume_task.
    """
    compiled = await build_resume_graph()
    workflow_id = f"resume-parse-{resume_id}"

    initial_state: dict[str, Any] = {
        **new_base_state(
            user_id=user_id,
            agent_run_id=agent_run_id,
            workflow_id=workflow_id,
        ),
        "resume_id": resume_id,
        "file_path": file_path,
        "mime_type": mime_type,
        "is_tailoring": False,
        "target_job_id": None,
        "tailor_tone": None,
        "emphasise_skills": [],
    }

    config = get_invoke_config(workflow_id=workflow_id)
    return await compiled.ainvoke(initial_state, config=config)


async def run_tailor_workflow(
    *,
    user_id: str,
    agent_run_id: str,
    resume_id: str,
    target_job_id: str,
    parsed_sections: dict[str, Any],
    raw_text: str,
    tone: str = "professional",
    emphasise_skills: list[str] | None = None,
) -> dict[str, Any]:
    """
    Run the tailor-for-job pipeline against an already-parsed master resume.

    The caller (a master resume that was already extracted/parsed at upload
    time) seeds raw_text and parsed_sections directly into initial state.
    The graph still enters at NODE_EXTRACT/NODE_PARSE for topology
    consistency with the initial-parse mode, but both nodes detect
    is_tailoring=True with pre-seeded data and pass it through unchanged
    instead of re-extracting from a nonexistent file or re-running the
    parsing LLM call on already-parsed content — see extract_text_node and
    parse_sections_node for the pass-through logic. Real work begins at
    NODE_TAILOR, which rewrites sections against the target job.

    Called from workers/resume_tasks.py:generate_cover_letter_task and from
    the application_workflow.py tailoring step.
    """
    compiled = await build_resume_graph()
    workflow_id = f"resume-tailor-{resume_id}-{target_job_id}"

    initial_state: dict[str, Any] = {
        **new_base_state(
            user_id=user_id,
            agent_run_id=agent_run_id,
            workflow_id=workflow_id,
        ),
        "resume_id": resume_id,
        "file_path": "",
        "mime_type": "",
        "is_tailoring": True,
        "target_job_id": target_job_id,
        "tailor_tone": tone,
        "emphasise_skills": emphasise_skills or [],
        # Pre-seed parse outputs so NODE_EXTRACT/NODE_PARSE are skipped
        # functionally (entry point still runs them, but they read from
        # an already-set raw_text/parsed_sections and pass through fast)
        "raw_text": raw_text,
        "parsed_sections": parsed_sections,
        "word_count": len(raw_text.split()) if raw_text else 0,
    }

    config = get_invoke_config(workflow_id=workflow_id)
    return await compiled.ainvoke(initial_state, config=config)