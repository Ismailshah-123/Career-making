"""CareerGPT -- Applications pipeline tracker (kanban-style by status)."""

from __future__ import annotations

import streamlit as st

from utils.api_client import APIError, get_client
from utils.auth import require_login, render_sidebar
from utils.theme import inject_css, hero, card_open, card_close, status_badge

st.set_page_config(page_title="Applications — CareerGPT", page_icon="📋", layout="wide")
inject_css()
require_login()
client = get_client()

hero("Applications", "Every application, one pipeline — from discovered to offer.")

STAGE_GROUPS = {
    "In Progress": ["discovered", "queued", "resume_tailored", "cover_letter_generated", "applying"],
    "Submitted": ["pending", "applied", "viewed", "acknowledged"],
    "Interviewing": ["screening", "interview", "interview_scheduled", "interviewed"],
    "Outcome": ["offer", "offer_received", "rejected", "withdrawn", "failed"],
}

try:
    applications = client.list_applications()
except APIError as e:
    applications = []
    st.warning(f"Couldn't load applications: {e}")

if not applications:
    st.info("No applications yet — find a job you like and apply from the Job Search page.")
    if st.button("Go to Job Search"):
        st.switch_page("pages/3_Job_Search.py")
else:
    cols = st.columns(4)
    for col, (group_name, statuses) in zip(cols, STAGE_GROUPS.items()):
        with col:
            st.markdown(f"#### {group_name}")
            group_apps = [a for a in applications if (a.get("status") or "").lower() in statuses]
            st.caption(f"{len(group_apps)} application(s)")
            for a in group_apps:
                card_open()
                st.markdown(f"**{a.get('job_title', a.get('title', 'Role'))}**")
                st.caption(a.get("company", a.get("company_name", "")))
                st.markdown(status_badge(a.get("status", "")), unsafe_allow_html=True)
                if st.button("Details", key=f"detail_{a.get('id')}", use_container_width=True):
                    st.session_state["selected_application_id"] = a.get("id")
                card_close()

st.markdown("---")

selected_id = st.session_state.get("selected_application_id")
if selected_id:
    card_open()
    try:
        app = client.get_application(selected_id)
        st.subheader(f"{app.get('job_title', 'Application')} — {app.get('company', '')}")
        st.markdown(status_badge(app.get("status", "")), unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            st.write(f"**Applied:** {app.get('applied_at', '—')}")
            st.write(f"**Follow-ups sent:** {app.get('followup_count', 0)}")
        with c2:
            st.write(f"**Next follow-up:** {app.get('next_followup_at', '—')}")
            st.write(f"**Auto-applied:** {'Yes' if app.get('auto_applied') else 'No'}")

        new_status = st.selectbox(
            "Update status",
            options=[s for group in STAGE_GROUPS.values() for s in group],
            index=0,
            key="status_select",
        )
        note = st.text_input("Note (optional)")
        if st.button("Update Status", type="primary"):
            try:
                client.update_application_status(selected_id, new_status, note or None)
                st.success("Status updated.")
                st.rerun()
            except APIError as e:
                st.error(str(e))

        action_cols = st.columns(2)
        if action_cols[0].button("✉️ Send Follow-up Now"):
            try:
                result = client.trigger_followup(selected_id)
                st.success("Follow-up generated" + (" and sent." if result.get("sent") else "; review it before sending."))
                if result.get("message"):
                    st.text_area("Follow-up message", result["message"], height=150)
            except APIError as e:
                st.error(str(e))

        with st.expander("Timeline"):
            try:
                timeline = client.get_timeline(selected_id)
                for event in timeline:
                    st.write(f"**{event.get('created_at', '')}** — {event.get('description', event.get('event', ''))}")
            except APIError as e:
                st.caption("Timeline unavailable.")
    except APIError as e:
        st.error(f"Couldn't load application: {e}")
    card_close()

render_sidebar(client)
