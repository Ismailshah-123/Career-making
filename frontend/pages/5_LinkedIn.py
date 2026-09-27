"""CareerGPT -- LinkedIn content: connect account, generate/approve/publish posts."""

from __future__ import annotations

import streamlit as st

from utils.api_client import APIError, get_client
from utils.auth import require_login, render_sidebar
from utils.theme import inject_css, hero, card_open, card_close

st.set_page_config(page_title="LinkedIn — CareerGPT", page_icon="💼", layout="wide")
inject_css()
require_login()
client = get_client()

hero("LinkedIn Presence", "Stay visible to recruiters with AI-drafted posts you approve before they go live.")

try:
    profile = client.linkedin_profile()
    connected = bool(profile)
except APIError:
    profile = None
    connected = False

if not connected:
    card_open()
    st.subheader("Connect your LinkedIn account")
    st.write("Connect once, then CareerGPT can draft and (with your approval) publish posts for you.")
    if st.button("Connect LinkedIn", type="primary"):
        try:
            auth_data = client.linkedin_auth_url()
            url = auth_data.get("url") or auth_data.get("auth_url")
            if url:
                st.link_button("Continue to LinkedIn", url, type="primary")
            else:
                st.info("Auth URL unavailable — check backend LinkedIn OAuth configuration.")
        except APIError as e:
            st.error(str(e))
    card_close()
else:
    top1, top2 = st.columns([2, 1])
    with top1:
        card_open()
        st.subheader("Generate a New Post")
        topic = st.text_input("Topic", placeholder="e.g. What I learned shipping a side project")
        tone = st.selectbox("Tone", ["professional", "casual", "thought_leadership", "celebratory"])
        if st.button("Generate Draft", type="primary"):
            try:
                with st.spinner("Drafting..."):
                    post = client.generate_linkedin_post(topic, tone)
                st.session_state["_draft_post"] = post
            except APIError as e:
                st.error(str(e))

        draft = st.session_state.get("_draft_post")
        if draft:
            st.text_area("Draft", draft.get("content", draft.get("text", "")), height=180, key="draft_text")
            dc1, dc2 = st.columns(2)
            if dc1.button("✅ Approve"):
                try:
                    client.approve_linkedin_post(draft["id"])
                    st.success("Approved — ready to publish.")
                except APIError as e:
                    st.error(str(e))
            if dc2.button("🚀 Publish Now"):
                try:
                    client.publish_linkedin_post(draft["id"])
                    st.success("Published to LinkedIn.")
                    st.session_state.pop("_draft_post", None)
                except APIError as e:
                    st.error(str(e))
        card_close()

    with top2:
        card_open()
        st.subheader("Analytics")
        try:
            overview = client.linkedin_analytics_overview()
            st.metric("Impressions (30d)", overview.get("impressions", "—"))
            st.metric("Engagement Rate", f"{overview.get('engagement_rate', 0)}%")
            st.metric("Posts Published", overview.get("posts_published", "—"))
        except APIError:
            st.caption("Analytics will appear after your first published post.")
        card_close()

    card_open()
    st.subheader("Post History")
    try:
        posts = client.list_linkedin_posts()
        if posts:
            for p in posts:
                with st.expander(f"{p.get('status', 'draft').title()} — {p.get('created_at', '')}"):
                    st.write(p.get("content", p.get("text", "")))
        else:
            st.caption("No posts yet.")
    except APIError as e:
        st.caption("Couldn't load post history.")
    card_close()

render_sidebar(client)
