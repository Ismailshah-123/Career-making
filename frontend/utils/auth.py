"""
CareerGPT -- Page-level auth guard.
Call require_login() at the top of every page under pages/ so an
unauthenticated visitor is redirected back to the login screen instead
of seeing a broken, data-less page.
"""

from __future__ import annotations

import streamlit as st


def require_login() -> None:
    if not st.session_state.get("access_token"):
        st.warning("Please log in to continue.")
        st.page_link("streamlit_app.py", label="Go to Login", icon="🔐")
        st.stop()


def current_user_label() -> str:
    return st.session_state.get("user_email") or "Account"


def render_sidebar(client) -> None:
    """Shared branded sidebar footer: user chip + logout. Call after require_login()."""
    with st.sidebar:
        st.markdown("---")
        st.caption(f"Signed in as **{current_user_label()}**")
        if st.button("Log out", use_container_width=True, key="sidebar_logout"):
            client.logout()
            st.switch_page("streamlit_app.py")
