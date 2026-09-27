"""CareerGPT -- Profile & preferences settings."""

from __future__ import annotations

import streamlit as st

from utils.api_client import APIError, get_client
from utils.auth import require_login, render_sidebar
from utils.theme import inject_css, hero, card_open, card_close

st.set_page_config(page_title="Settings — CareerGPT", page_icon="⚙️", layout="wide")
inject_css()
require_login()
client = get_client()

hero("Settings", "Keep your profile and job preferences current — better inputs mean better matches.")

try:
    profile = client.get_profile()
except APIError as e:
    profile = {}
    st.warning(f"Couldn't load profile: {e}")

col1, col2 = st.columns(2, gap="large")

with col1:
    card_open()
    st.subheader("Profile")
    with st.form("profile_form"):
        full_name = st.text_input("Full name", value=profile.get("full_name", ""))
        linkedin_url = st.text_input("LinkedIn URL", value=profile.get("linkedin_url", ""))
        github_url = st.text_input("GitHub URL", value=profile.get("github_url", ""))
        portfolio_url = st.text_input("Portfolio URL", value=profile.get("portfolio_url", ""))
        saved = st.form_submit_button("Save Profile", type="primary")
    if saved:
        try:
            client.update_profile(
                full_name=full_name or None,
                linkedin_url=linkedin_url or None,
                github_url=github_url or None,
                portfolio_url=portfolio_url or None,
            )
            st.success("Profile updated.")
        except APIError as e:
            st.error(str(e))
    card_close()

    card_open()
    st.subheader("Usage This Month")
    try:
        usage = client.get_usage()
        u1, u2, u3 = st.columns(3)
        u1.metric("Applications", usage.get("applications_used", "—"))
        u2.metric("Resumes Tailored", usage.get("resumes_tailored", "—"))
        u3.metric("Plan", (profile.get("plan") or st.session_state.get("user_plan") or "free").title())
    except APIError:
        st.caption("Usage data unavailable.")
    card_close()

with col2:
    card_open()
    st.subheader("Job Search Preferences")
    with st.form("prefs_form"):
        target_roles_raw = st.text_area(
            "Target roles (one per line)",
            value="\n".join(profile.get("target_roles", []) or []),
            height=100,
        )
        target_locations_raw = st.text_area(
            "Target locations (one per line)",
            value="\n".join(profile.get("target_locations", []) or []),
            height=100,
        )
        min_salary = st.number_input("Minimum salary (USD)", min_value=0, step=5000,
                                      value=int(profile.get("min_salary") or 0))
        prefs_saved = st.form_submit_button("Save Preferences", type="primary")
    if prefs_saved:
        try:
            client.update_profile(
                target_roles=[r.strip() for r in target_roles_raw.splitlines() if r.strip()],
                target_locations=[l.strip() for l in target_locations_raw.splitlines() if l.strip()],
                min_salary=min_salary or None,
            )
            st.success("Preferences updated.")
        except APIError as e:
            st.error(str(e))
    card_close()

    card_open()
    st.subheader("Danger Zone")
    st.caption("Deleting your account removes all resumes, applications, and history. This can't be undone.")
    confirm = st.checkbox("I understand this is permanent.")
    if st.button("Delete Account", disabled=not confirm):
        st.error("Account deletion must be confirmed via the email link sent to you (GDPR-safe two-step flow).")
    card_close()

render_sidebar(client)
