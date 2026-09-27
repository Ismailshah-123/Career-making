"""
Contract tests for the agent layer: every agent class instantiates and
exposes the methods the rest of the app calls on it. These don't call
the LLM (no network) -- they exist to catch the class of bug this
codebase had before its QA pass, where two agent modules were entirely
empty files that crashed the moment anything tried to import them.
"""

from __future__ import annotations


def test_resume_agent_has_expected_interface():
    from app.agents.resume_agent.agent import ResumeAgent

    agent = ResumeAgent()
    for method in ("parse_raw_text", "tailor_sections", "tailor"):
        assert callable(getattr(agent, method, None)), f"ResumeAgent.{method} missing"


def test_followup_agent_has_expected_interface():
    from app.agents.followup_agent.agent import FollowupAgent

    agent = FollowupAgent()
    for method in ("generate", "send_followup"):
        assert callable(getattr(agent, method, None)), f"FollowupAgent.{method} missing"


def test_matching_agent_tools_are_importable():
    from app.agents.matching_agent import tools

    for fn in (
        "score_match", "quick_score_batch", "extract_job_skills",
        "analyze_skills_gap", "rank_jobs_for_user", "estimate_salary",
        "generate_match_explanation", "get_semantic_similarity",
    ):
        assert callable(getattr(tools, fn, None)), f"matching_agent.tools.{fn} missing"


def test_every_agent_module_imports_cleanly():
    import importlib

    agent_modules = [
        "app.agents.discovery_agent.agent",
        "app.agents.resume_agent.agent",
        "app.agents.matching_agent.agent",
        "app.agents.cover_letter_agent.agent",
        "app.agents.outreach_agent.agent",
        "app.agents.linkedin_agent.agent",
        "app.agents.application_agent.agent",
        "app.agents.followup_agent.agent",
    ]
    for mod in agent_modules:
        importlib.import_module(mod)  # raises on failure
