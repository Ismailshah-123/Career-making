"""
CareerGPT -- Streamlit Frontend Entrypoint
==============================================
Landing / login / register screen. Once authenticated, the sidebar
navigation (Streamlit's automatic multi-page nav, driven by pages/)
takes over.
"""

from __future__ import annotations

import streamlit as st

from utils.api_client import APIError, get_client
from utils.theme import inject_css, hero, card_open, card_close, NAVY, BLUE

st.set_page_config(
    page_title="CareerGPT — AI Job Application Platform",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="collapsed" if not st.session_state.get("access_token") else "expanded",
)
inject_css()

client = get_client()

# ── Already logged in -> send to dashboard ───────────────────────────────────
if client.is_authenticated:
    st.switch_page("pages/1_Dashboard.py")

# ── Landing / auth screen ─────────────────────────────────────────────────────

hero(
    "CareerGPT",
    "Your AI co-pilot for the entire job search — discovery, tailored resumes, "
    "smart matching, and auto-apply, all in one place.",
)

left, right = st.columns([1.1, 1], gap="large")

with left:
    card_open()
    st.subheader("Why CareerGPT")
    st.markdown(
        f"""
- 🔍 **Smart discovery** across LinkedIn, Indeed, RemoteOK & more
- 🎯 **AI match scoring** so you only spend time on real fits
- 📄 **Resume tailoring** per job, with live ATS scoring
- ✍️ **Cover letters & recruiter outreach**, drafted for you
- 🤖 **Auto-apply** with human-in-the-loop review
- 📊 **One pipeline view** of every application, from discovered to offer
        """
    )
    st.caption("Connects to your CareerGPT backend — configure `CAREERGPT_API_URL` to point at your deployment.")
    card_close()

with right:
    card_open()
    tab_login, tab_register = st.tabs(["Log In", "Create Account"])

    with tab_login:
        with st.form("login_form", border=False):
            email = st.text_input("Email", key="login_email")
            password = st.text_input("Password", type="password", key="login_password")
            remember = st.checkbox("Remember me", value=True)
            submitted = st.form_submit_button("Log In", use_container_width=True, type="primary")

        if submitted:
            if not email or not password:
                st.error("Enter both your email and password.")
            else:
                try:
                    with st.spinner("Signing you in..."):
                        client.login(email, password, remember_me=remember)
                    st.success("Welcome back!")
                    st.rerun()
                except APIError as e:
                    st.error(e.args[0] if e.args else "Login failed.")

    with tab_register:
        with st.form("register_form", border=False):
            r_name = st.text_input("Full name")
            r_email = st.text_input("Email", key="reg_email")
            r_password = st.text_input(
                "Password", type="password", key="reg_password",
                help="At least 10 characters, mixing case/numbers/symbols.",
            )
            r_submitted = st.form_submit_button("Create Account", use_container_width=True, type="primary")

        if r_submitted:
            if not r_email or not r_password:
                st.error("Email and password are required.")
            else:
                try:
                    with st.spinner("Creating your account..."):
                        client.register(r_email, r_password, r_name or "")
                    st.success("Account created! Redirecting to your dashboard...")
                    st.rerun()
                except APIError as e:
                    st.error(e.args[0] if e.args else "Could not create account.")

    card_close()

st.markdown(
    f"<p style='text-align:center;color:#5B6B85;margin-top:2rem;'>"
    f"CareerGPT · Built for candidates who want their search run like a pipeline, not a chore.</p>",
    unsafe_allow_html=True,
)
