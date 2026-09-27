"""CareerGPT -- Resume manager: upload, view, ATS score, tailor per job."""

from __future__ import annotations

import streamlit as st

from utils.api_client import APIError, get_client
from utils.auth import require_login, render_sidebar
from utils.theme import inject_css, hero, card_open, card_close

st.set_page_config(page_title="Resumes — CareerGPT", page_icon="📄", layout="wide")
inject_css()
require_login()
client = get_client()

hero("Resumes", "Upload your master resume once — CareerGPT tailors a fresh version for every job.")

upload_col, list_col = st.columns([1, 1.6], gap="large")

with upload_col:
    card_open()
    st.subheader("Upload a Resume")
    uploaded = st.file_uploader("PDF, DOCX, or TXT", type=["pdf", "docx", "doc", "txt"])
    if uploaded is not None and st.button("Upload", type="primary", use_container_width=True):
        try:
            with st.spinner("Uploading and parsing..."):
                result = client.upload_resume(uploaded.name, uploaded.getvalue(), uploaded.type or "application/octet-stream")
            st.success("Resume uploaded and parsed.")
            st.session_state.pop("_resume_list_cache", None)
            st.rerun()
        except APIError as e:
            st.error(f"Upload failed: {e}")
    st.caption("Max 10MB. Your resume is parsed automatically so it can be matched and tailored per job.")
    card_close()

with list_col:
    card_open()
    st.subheader("Your Resumes")
    try:
        resumes = client.list_resumes()
    except APIError as e:
        resumes = []
        st.warning(f"Couldn't load resumes: {e}")

    if not resumes:
        st.info("No resumes uploaded yet.")
    else:
        for r in resumes:
            with st.expander(f"📄 {r.get('title', r.get('filename', 'Resume'))} · v{r.get('version', 1)}"):
                meta_cols = st.columns(3)
                meta_cols[0].metric("ATS Score", r.get("ats_score", "—"))
                meta_cols[1].metric("Skills Found", len(r.get("extracted_skills", []) or []))
                meta_cols[2].metric("Words", r.get("word_count", "—"))

                skills = r.get("extracted_skills") or []
                if skills:
                    st.write(", ".join(skills[:20]) + (" ..." if len(skills) > 20 else ""))

                btn_cols = st.columns(3)
                if btn_cols[0].button("Refresh ATS Score", key=f"ats_{r['id']}"):
                    try:
                        with st.spinner("Scoring..."):
                            score = client.ats_score(r["id"])
                        st.success(f"ATS score: {score.get('ats_score', score.get('score', '—'))}")
                    except APIError as e:
                        st.error(str(e))
                if btn_cols[1].button("Use for Job Search", key=f"use_{r['id']}"):
                    st.session_state["active_resume_id"] = r["id"]
                    st.success("Set as your active resume for matching.")
                if btn_cols[2].button("Delete", key=f"del_{r['id']}"):
                    try:
                        client.delete_resume(r["id"])
                        st.success("Deleted.")
                        st.rerun()
                    except APIError as e:
                        st.error(str(e))
    card_close()

    if st.session_state.get("active_resume_id"):
        st.caption(f"Active resume for matching/tailoring: `{st.session_state['active_resume_id']}`")

render_sidebar(client)
