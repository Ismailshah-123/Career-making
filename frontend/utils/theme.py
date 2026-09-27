"""
CareerGPT -- Design System
=============================
Navy / blue / white design tokens + a single CSS injection helper used by
every page. Keep this the ONE place that defines colors so the app stays
visually consistent as pages are added.
"""

from __future__ import annotations

import streamlit as st

# ── Design tokens ────────────────────────────────────────────────────────────
NAVY        = "#0B1D3A"   # primary dark navy (headers, sidebar, text)
NAVY_DEEP   = "#071227"   # near-black navy (hero backgrounds)
BLUE        = "#2563EB"   # primary accent (buttons, links, active states)
BLUE_LIGHT  = "#EFF4FF"   # light blue tint (cards, hover states)
BLUE_SOFT   = "#DCE7FB"   # borders / dividers
WHITE       = "#FFFFFF"
GRAY_BG     = "#F4F7FB"   # page background
GRAY_TEXT   = "#5B6B85"   # secondary text
SUCCESS     = "#16A34A"
WARNING     = "#D97706"
DANGER      = "#DC2626"

STATUS_COLORS = {
    "discovered": "#94A3B8", "queued": "#94A3B8",
    "resume_tailored": "#60A5FA", "cover_letter_generated": "#60A5FA",
    "applying": "#3B82F6", "pending": "#3B82F6", "applied": BLUE,
    "viewed": "#8B5CF6", "acknowledged": "#8B5CF6",
    "screening": "#F59E0B", "interview": "#F59E0B",
    "interview_scheduled": "#F59E0B", "interviewed": "#F59E0B",
    "offer": SUCCESS, "offer_received": SUCCESS,
    "rejected": DANGER, "withdrawn": "#94A3B8", "failed": DANGER,
}


def inject_css() -> None:
    """Call once per page (top of the script) to apply the shared theme."""
    st.markdown(
        f"""
        <style>
        html, body, [class*="css"]  {{
            font-family: -apple-system, "Segoe UI", Roboto, Inter, sans-serif;
        }}

        .stApp {{
            background-color: {GRAY_BG};
        }}

        section[data-testid="stSidebar"] {{
            background-color: {NAVY};
        }}
        section[data-testid="stSidebar"] * {{
            color: {WHITE} !important;
        }}
        section[data-testid="stSidebar"] .stButton button {{
            background-color: transparent;
            border: 1px solid rgba(255,255,255,0.25);
            color: {WHITE} !important;
        }}
        section[data-testid="stSidebar"] .stButton button:hover {{
            background-color: {BLUE};
            border-color: {BLUE};
        }}

        h1, h2, h3 {{
            color: {NAVY};
            font-weight: 700;
        }}

        .cg-hero {{
            background: linear-gradient(135deg, {NAVY_DEEP} 0%, {NAVY} 55%, {BLUE} 140%);
            color: {WHITE};
            padding: 2.25rem 2rem;
            border-radius: 16px;
            margin-bottom: 1.5rem;
        }}
        .cg-hero h1 {{ color: {WHITE}; margin-bottom: 0.25rem; }}
        .cg-hero p {{ color: #C7D6F5; font-size: 1.02rem; margin: 0; }}

        .cg-card {{
            background: {WHITE};
            border: 1px solid {BLUE_SOFT};
            border-radius: 14px;
            padding: 1.25rem 1.4rem;
            box-shadow: 0 1px 3px rgba(11,29,58,0.06);
            margin-bottom: 1rem;
        }}

        .cg-metric-label {{
            color: {GRAY_TEXT};
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            font-weight: 600;
        }}
        .cg-metric-value {{
            color: {NAVY};
            font-size: 2rem;
            font-weight: 800;
            line-height: 1.1;
        }}

        .cg-badge {{
            display: inline-block;
            padding: 0.18rem 0.65rem;
            border-radius: 999px;
            font-size: 0.75rem;
            font-weight: 700;
            color: {WHITE};
        }}

        .stButton>button[kind="primary"], .stButton>button:not([kind]) {{
            background-color: {BLUE};
            color: {WHITE};
            border-radius: 8px;
            border: none;
            font-weight: 600;
        }}
        .stButton>button:hover {{
            background-color: {NAVY};
            color: {WHITE};
        }}

        div[data-testid="stMetric"] {{
            background: {WHITE};
            border: 1px solid {BLUE_SOFT};
            border-radius: 12px;
            padding: 0.85rem 1rem;
        }}

        .cg-pill-nav a {{
            text-decoration: none;
        }}

        footer {{visibility: hidden;}}
        #MainMenu {{visibility: hidden;}}
        </style>
        """,
        unsafe_allow_html=True,
    )


def status_badge(status: str) -> str:
    color = STATUS_COLORS.get((status or "").lower(), GRAY_TEXT)
    label = (status or "unknown").replace("_", " ").title()
    return f'<span class="cg-badge" style="background:{color}">{label}</span>'


def hero(title: str, subtitle: str) -> None:
    st.markdown(
        f"""<div class="cg-hero"><h1>{title}</h1><p>{subtitle}</p></div>""",
        unsafe_allow_html=True,
    )


def card_open(extra_style: str = "") -> None:
    st.markdown(f'<div class="cg-card" style="{extra_style}">', unsafe_allow_html=True)


def card_close() -> None:
    st.markdown("</div>", unsafe_allow_html=True)
