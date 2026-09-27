"""CareerGPT -- Job search, discovery, and match scoring."""

from __future__ import annotations

import streamlit as st

from utils.api_client import APIError, get_client
from utils.auth import require_login, render_sidebar
from utils.theme import inject_css, hero, card_open, card_close

st.set_page_config(page_title="Job Search — CareerGPT", page_icon="🔍", layout="wide")
inject_css()
require_login()
client = get_client()

hero("Job Search & Matching", "Search live postings and see an AI-generated fit score against your resume.")

with st.form("search_form"):
    c1, c2, c3, c4 = st.columns([2, 2, 1.4, 1])
    keywords = c1.text_input("Keywords", placeholder="e.g. Backend Engineer, Python")
    location = c2.text_input("Location", placeholder="Remote / City")
    remote_only = c3.checkbox("Remote only", value=True)
    search_clicked = c4.form_submit_button("Search", type="primary", use_container_width=True)

if search_clicked:
    try:
        with st.spinner("Searching..."):
            results = client.search_jobs(q=keywords, location=location, remote=remote_only)
        jobs = results if isinstance(results, list) else results.get("items", [])
        st.session_state["_job_search_results"] = jobs
    except APIError as e:
        st.error(f"Search failed: {e}")
        st.session_state["_job_search_results"] = []

jobs = st.session_state.get("_job_search_results", [])

if not jobs:
    st.info("Search above, or discover fresh postings from the sources CareerGPT scrapes.")
    if st.button("🔎 Run a fresh discovery scan", type="secondary"):
        try:
            with st.spinner("Kicking off discovery — this runs in the background..."):
                client.discover_jobs(keywords=[keywords] if keywords else ["software engineer"], locations=[location] if location else ["remote"])
            st.success("Discovery started. Check back in a minute, or refresh Recommendations on the Dashboard.")
        except APIError as e:
            st.error(str(e))
else:
    st.caption(f"{len(jobs)} results")
    for job in jobs:
        card_open()
        top = st.columns([3, 1, 1])
        top[0].markdown(f"### {job.get('title', 'Untitled role')}")
        top[0].caption(f"{job.get('company', '')} · {job.get('location', 'Remote')} · {job.get('job_board', job.get('source', ''))}")
        salary_min, salary_max = job.get("salary_min"), job.get("salary_max")
        if salary_min or salary_max:
            top[1].metric("Salary", f"${salary_min or '?'}–${salary_max or '?'}")
        if top[2].button("Match Score", key=f"match_{job.get('id')}"):
            try:
                with st.spinner("Scoring fit..."):
                    active_resume = st.session_state.get("active_resume_id")
                    result = client.match_job(job["id"], resume_id=active_resume)
                score = result.get("score", 0)
                st.progress(min(max(float(score), 0.0), 1.0), text=f"{round(float(score) * 100)}% match — {result.get('rating', '')}")
                strengths = result.get("strengths") or []
                gaps = result.get("gaps") or []
                if strengths:
                    st.markdown("**Strengths:** " + "; ".join(strengths[:4]))
                if gaps:
                    st.markdown("**Gaps:** " + "; ".join(gaps[:4]))
            except APIError as e:
                st.error(str(e))

        desc = job.get("description", "")
        if desc:
            with st.expander("Job description"):
                st.write(desc[:2000] + ("..." if len(desc) > 2000 else ""))

        action_cols = st.columns(3)
        if action_cols[0].button("⭐ Bookmark", key=f"bm_{job.get('id')}"):
            try:
                client.bookmark_job(job["id"])
                st.success("Bookmarked.")
            except APIError as e:
                st.error(str(e))
        if action_cols[1].button("📨 Apply", key=f"apply_{job.get('id')}"):
            try:
                active_resume = st.session_state.get("active_resume_id")
                client.create_application(job["id"], resume_id=active_resume, auto_apply=False)
                st.success("Added to your applications pipeline.")
            except APIError as e:
                st.error(str(e))
        if action_cols[2].button("🤖 Auto-Apply", key=f"auto_{job.get('id')}"):
            try:
                active_resume = st.session_state.get("active_resume_id")
                client.create_application(job["id"], resume_id=active_resume, auto_apply=True)
                st.success("Queued for automated application.")
            except APIError as e:
                st.error(str(e))
        card_close()

render_sidebar(client)
