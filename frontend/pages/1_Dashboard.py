"""CareerGPT -- Dashboard: pipeline overview, quick stats, recent activity."""

from __future__ import annotations

import streamlit as st

from utils.api_client import APIError, get_client
from utils.auth import require_login, render_sidebar
from utils.theme import inject_css, hero, card_open, card_close, status_badge

st.set_page_config(page_title="Dashboard — CareerGPT", page_icon="📊", layout="wide")
inject_css()
require_login()
client = get_client()

hero("Dashboard", f"Welcome back, {st.session_state.get('user_email', 'there')} — here's where your search stands.")

# ── Top-line stats ────────────────────────────────────────────────────────────
try:
    app_stats = client.get_application_stats()
except APIError:
    app_stats = {}
try:
    job_stats = client.get_job_stats()
except APIError:
    job_stats = {}

c1, c2, c3, c4 = st.columns(4)
c1.metric("Applications Sent", app_stats.get("total_applications", app_stats.get("total", "—")))
c2.metric("Active Pipeline", app_stats.get("active_count", "—"))
c3.metric("Interviews", app_stats.get("interview_count", "—"))
c4.metric("Jobs Matched", job_stats.get("total_matched", job_stats.get("total", "—")))

st.write("")

left, right = st.columns([1.4, 1], gap="large")

# ── Pipeline breakdown ─────────────────────────────────────────────────────────
with left:
    card_open()
    st.subheader("Pipeline by Stage")
    try:
        pipeline = client.get_pipeline()
        stages = pipeline.get("stages") or pipeline
        if isinstance(stages, dict) and stages:
            for stage, items in stages.items():
                count = len(items) if isinstance(items, list) else items
                bcol, ncol = st.columns([4, 1])
                bcol.markdown(status_badge(stage), unsafe_allow_html=True)
                ncol.markdown(f"**{count}**")
        else:
            st.info("No applications yet — head to Job Search to find your first match.")
    except APIError as e:
        st.warning(f"Couldn't load pipeline: {e}")
    card_close()

    card_open()
    st.subheader("Recent Applications")
    try:
        apps = client.list_applications()[:6]
        if apps:
            for a in apps:
                row = st.columns([3, 2, 2, 1])
                row[0].markdown(f"**{a.get('job_title', a.get('title', 'Untitled role'))}**")
                row[1].markdown(a.get("company", a.get("company_name", "")))
                row[2].markdown(status_badge(a.get("status", "")), unsafe_allow_html=True)
                if row[3].button("View", key=f"view_{a.get('id')}"):
                    st.session_state["selected_application_id"] = a.get("id")
                    st.switch_page("pages/4_Applications.py")
        else:
            st.info("Nothing here yet.")
    except APIError as e:
        st.warning(f"Couldn't load applications: {e}")
    card_close()

# ── Recommended jobs ─────────────────────────────────────────────────────────
with right:
    card_open()
    st.subheader("Top Matches For You")
    try:
        recs = client.get_recommendations(limit=5)
        if recs:
            for job in recs:
                st.markdown(f"**{job.get('title')}**")
                st.caption(f"{job.get('company', '')} · {job.get('location', 'Remote')}")
                score = job.get("match_score") or job.get("score")
                if score is not None:
                    st.progress(min(max(float(score), 0.0), 1.0), text=f"{round(float(score) * 100)}% match")
                st.markdown("---")
        else:
            st.info("Run a job search to get personalized matches.")
            if st.button("Go to Job Search", use_container_width=True):
                st.switch_page("pages/3_Job_Search.py")
    except APIError as e:
        st.info("Recommendations will appear once you've uploaded a resume and matched a few jobs.")
    card_close()

    card_open()
    st.subheader("Quick Actions")
    qa1, qa2 = st.columns(2)
    if qa1.button("📄 Upload Resume", use_container_width=True):
        st.switch_page("pages/2_Resumes.py")
    if qa2.button("🔍 Find Jobs", use_container_width=True):
        st.switch_page("pages/3_Job_Search.py")
    card_close()

render_sidebar(client)
