"""
app.py  —  MarketMind AI  (Saudi Arabia GTM Intelligence Platform)
------------------------------------------------------------------
Fixes in this version:
  1. PDF "not enough horizontal space" — fixed in pdf_export.py (explicit
     content width on every multi_cell call; table rows handled separately).
  2. Nested expander crash — all "Copy" blocks inside show_content_output()
     replaced with st.container(border=True) instead of st.expander().
  3. Background image opacity — _bg_css() uses background-blend-mode:overlay
     with a white rgba() background-color so the image fades cleanly without
     affecting the sidebar.
  4. Budget label — Arabic text removed; label is English-only.
  5. Sidebar API-key inputs removed — keys come from env vars only. Sidebar
     shows pipeline progress + a single Restart button at the bottom.

Pipeline order (sequential):
  Agent 1: Research Agent         — Saudi Arabia market & competitor analysis
  Agent 2: Strategy Agent         — ICP, budget-aware GTM, 30-day calendar
  Agent 3: Content Agent          — posts, email, ads, video script
  Agent 4: Brand Alignment Agent  — FINAL quality gate, refines all content

Run:  streamlit run app.py
"""

from __future__ import annotations

import base64
import os
import sys
import tempfile
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# STEP 1 — Load .env FIRST, before any LangChain/local import.
# LangChain reads tracing env vars at import time, so dotenv must fire here.
# ---------------------------------------------------------------------------
from dotenv import load_dotenv
load_dotenv()

# ---------------------------------------------------------------------------
# STEP 2 — Set the correct LangChain tracing variables.
# Use LANGCHAIN_TRACING_V2 and LANGCHAIN_PROJECT (current SDK naming).
# Do NOT set LANGCHAIN_ENDPOINT — let the SDK use its default, which avoids
# the 404 "path does not exist" error caused by stale endpoint overrides.
# ---------------------------------------------------------------------------
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"]    = os.getenv("LANGSMITH_PROJECT", "MarketMindAI_Track")

# Forward the API key under both names the SDK checks
_ls_key = os.getenv("LANGSMITH_API_KEY", "")
if _ls_key:
    os.environ["LANGCHAIN_API_KEY"] = _ls_key

# ---------------------------------------------------------------------------
# STEP 3 — Optional connectivity check (silent — never crashes the app).
# Verifies the LangSmith client can reach the API before agents run.
# ---------------------------------------------------------------------------
try:
    from langsmith import Client as _LSClient
    from langsmith import traceable as _traceable
    _ls_client = _LSClient()          # uses LANGCHAIN_API_KEY automatically
    _ls_client.list_projects()        # lightweight ping — no data written
    _LANGSMITH_CONNECTED = True
except Exception:
    _LANGSMITH_CONNECTED = False      # tracing unavailable; app continues normally
    # Fallback: identity decorator so wrappers below work even without langsmith
    def _traceable(*args, **kwargs):
        def decorator(fn):
            return fn
        return decorator if args and callable(args[0]) else decorator

# ---------------------------------------------------------------------------
# STEP 4 — Project root on sys.path so local agent/util imports resolve.
# Must come before any "from agents.*" or "from utils.*" import.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# STEP 5 — All remaining imports (LangChain already tracing at this point).
# ---------------------------------------------------------------------------
import streamlit as st
from openai import OpenAI
from tavily import TavilyClient

from agents.brand_alignment_agent import BrandAlignmentAgent, RefinedContent
from agents.content_agent import ContentAgent, ContentPackage
from agents.research_agent import CompanyProfile, ResearchAgent
from agents.strategy_agent import StrategyAgent
from utils.pdf_export import build_pdf
from utils.pdf_handler import load_and_validate_pdf
from utils.security import validate_input_text

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="MarketMind AI — Saudi Arabia GTM",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Fix 3 — Background CSS with blend-mode fade, sidebar unaffected
# ---------------------------------------------------------------------------
def _bg_css() -> str:
    bg_path = _PROJECT_ROOT / "assets" / "background.jpg"
    if bg_path.exists():
        with open(bg_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        return (
            "<style>"
            ".stApp {"
            # هنا نضع لوناً ثابتاً (ليس تدرجاً) فوق الصورة
            f"  background-image: linear-gradient(rgba(245, 243, 250, 0.8), rgba(245, 243, 250, 0.8)), "
            f"  url(\"data:image/jpeg;base64,{b64}\");"
            "  background-size: cover;"         # تغطية كاملة
            "  background-position: center;"    # توسيط في المنتصف
            "  background-repeat: no-repeat;"
            "  background-attachment: fixed;"
            "}"
            "[data-testid=\"stSidebar\"] {"
            "  background: #EAE6FA !important; background-image: none !important;"
            "}"
            "</style>"
        )
    return ""

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
# Inject background as a full <style> block (returned by _bg_css())
st.markdown(_bg_css(), unsafe_allow_html=True)

st.markdown("""
<style>
:root {
    --mm-primary:#C5B4EC; --mm-accent:#E5A4CB;
    --mm-dark:#5B4F7B;    --mm-surface:#fffffff2;
    --mm-border:#E1DBF4;  --mm-success:#88C3A4;
    --mm-text:#4A3E65;
}

/* Content area */
.block-container {
    position:relative; z-index:1;
    padding-top:1.4rem !important;
    max-width:1180px;
}

/* ── Sidebar — isolated dark panel ── */
[data-testid="stSidebar"] {
    background:#EAE6FA !important;
    background-image:none !important;
    z-index:100;
}
[data-testid="stSidebar"] * { color:#5B4F7B !important; }
[data-testid="stSidebar"] .stMarkdown h2 {
    color:#9D84D9 !important; font-size:.82rem;
    letter-spacing:.12em; text-transform:uppercase;
}

/* ── Stage pills ── */
.stage-pill {
    display:inline-block; padding:3px 11px; border-radius:20px;
    font-size:.71rem; font-weight:700; letter-spacing:.06em;
    text-transform:uppercase; margin-bottom:4px;
}
.pill-pending { background:#E4E2F1; color:#A49EC6; }
.pill-active  { background:#D6CAFA; color:#5B4F7B; border:1px solid #B09EEB; }
.pill-done    { background:#D1EAD9; color:#407D56; }

/* ── Result banner ── */
.result-banner {
    background:linear-gradient(90deg,#C5B4EC,#E5A4CB);
    color:#5B4F7B; padding:16px 22px; border-radius:12px;
    margin-bottom:18px; font-size:1.1rem; font-weight:700;
}

/* ── Content cards ── */
.post-card {
    border-left:4px solid var(--mm-primary); background:var(--mm-surface);
    border-radius:0 10px 10px 0; padding:15px 19px; margin-bottom:13px;
    box-shadow:0 1px 5px rgba(197,180,236,.15);
}
.ad-card {
    border-left:4px solid var(--mm-accent); background:var(--mm-surface);
    border-radius:0 10px 10px 0; padding:15px 19px; margin-bottom:13px;
    box-shadow:0 1px 5px rgba(229,164,203,.15);
}
.video-card {
    border-left:4px solid #A8C2FA; background:var(--mm-surface);
    border-radius:0 10px 10px 0; padding:15px 19px; margin-bottom:13px;
    box-shadow:0 1px 5px rgba(168,194,250,.15);
}
.platform-tag {
    font-size:.67rem; font-weight:700; text-transform:uppercase;
    letter-spacing:.1em; color:#9D84D9; margin-bottom:5px;
}
.email-preview {
    background:var(--mm-surface); border:1px solid var(--mm-border);
    border-radius:10px; padding:22px 26px;
    font-family:Georgia,serif; line-height:1.75; color:var(--mm-text);
}
.email-subject  { font-size:1.05rem; font-weight:700; color:var(--mm-dark); margin-bottom:3px; }
.email-preview-text { font-size:.79rem; color:#A49EC6; margin-bottom:14px; font-style:italic; }
.email-cta {
    display:inline-block; margin-top:12px; padding:9px 22px;
    background:var(--mm-primary); color:#5B4F7B !important;
    border-radius:6px; font-weight:700; font-size:.86rem;
}
.guardrail-item {
    background:#FFF0F5; border-left:3px solid #E5A4CB;
    padding:5px 11px; margin-bottom:3px; border-radius:0 4px 4px 0; font-size:.79rem;
    color:#5B4F7B;
}
.mm-divider { border:none; border-top:1px solid var(--mm-border); margin:20px 0; }
#MainMenu,footer { visibility:hidden; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
DEFAULTS: dict = {
    "stage": 1,
    "substage": "form",         # form | hitl | running
    "profile":          None,
    "research_report":  None,
    "research_doc_id":  None,
    "gtm_strategy":     None,
    "calendar_md":      None,
    "strategy_doc_id":  None,
    "content_package":  None,
    "content_doc_id":   None,
    "refined_content":  None,
    "brand_doc_id":     None,
    "form_company_name":   "",
    "form_products":       "",
    "form_brand_voice":    "",
    "form_past_marketing": "",
    "form_pdf_text":       "",
    "form_budget":         "SAR 20,000 per month",
    # Token / cost tracking
    "total_prompt_tokens":     0,
    "total_completion_tokens": 0,
    "total_tokens":            0,
    "total_cost_usd":          0.0,
    "call_log":                [],
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ---------------------------------------------------------------------------
# Fix 5 — API clients from environment only
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def _get_clients(openai_key: str, tavily_key: str):
    from langchain_openai import ChatOpenAI
    openai_client  = OpenAI(api_key=openai_key)
    tavily_client  = TavilyClient(api_key=tavily_key)
    # Shared ChatOpenAI instance — all agents share this so LangSmith
    # sees one unified trace tree with token counts per call.
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.4,
        api_key=openai_key,
    )
    return openai_client, tavily_client, llm

def resolve_clients():
    """Read API keys exclusively from environment variables."""
    ok = os.environ.get("OPENAI_API_KEY", "").strip()
    tk = os.environ.get("TAVILY_API_KEY", "").strip()
    if not ok or not tk:
        st.error(
            "⚠️ API keys not found in environment.\n\n"
            "Add them to your `.env` file:\n```\n"
            "OPENAI_API_KEY=sk-...\nTAVILY_API_KEY=tvly-...\n```"
        )
        st.stop()
    return _get_clients(ok, tk)  # returns (openai_client, tavily_client, llm)

# ---------------------------------------------------------------------------
# PDF download button helper
# ---------------------------------------------------------------------------
def pdf_download_button(label: str, agent_label: str, content: str,
                         filename: str, key: str) -> None:
    company_name = ""
    if st.session_state.profile:
        company_name = st.session_state.profile.company_name
    try:
        pdf_bytes = build_pdf(agent_label, company_name, content)
        st.download_button(
            label=label,
            data=pdf_bytes,
            file_name=filename,
            mime="application/pdf",
            key=key,
        )
    except Exception as exc:
        st.warning(f"PDF generation note: {exc}")



# ---------------------------------------------------------------------------
# Token & cost tracking helpers
# GPT-4o-mini: $0.15/1M prompt tokens, $0.60/1M completion tokens
# ---------------------------------------------------------------------------
_COST_PROMPT     = 0.15 / 1_000_000
_COST_COMPLETION = 0.60 / 1_000_000

def _track_tokens(agent_name, response):
    try:
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        p  = getattr(usage, "prompt_tokens",     0) or 0
        c  = getattr(usage, "completion_tokens", 0) or 0
        cost = p * _COST_PROMPT + c * _COST_COMPLETION
        st.session_state.total_prompt_tokens     += p
        st.session_state.total_completion_tokens += c
        st.session_state.total_tokens            += p + c
        st.session_state.total_cost_usd          += cost
        st.session_state.call_log.append(
            {"agent": agent_name, "prompt": p, "completion": c, "cost": cost}
        )
    except Exception:
        pass


class _CC:
    def __init__(self, real, name):
        self._r = real
        self._n = name
    def create(self, *a, **kw):
        resp = self._r.chat.completions.create(*a, **kw)
        _track_tokens(self._n, resp)
        return resp
    def parse(self, *a, **kw):
        resp = self._r.beta.chat.completions.parse(*a, **kw)
        _track_tokens(self._n, resp)
        return resp

class _CB:
    def __init__(self, real, name):
        self._r = real
        self._n = name
    @property
    def chat(self):
        return type("_CC2", (), {
            "completions": _CC(self._r, self._n)
        })()

class _TrackingClient:
    def __init__(self, real, name):
        self._r  = real
        self.beta = _CB(real, name)
    def __getattr__(self, item):
        return getattr(self._r, item)

def _inject(agent, name):
    agent.openai_client = _TrackingClient(agent.openai_client, name)
    return agent

# ---------------------------------------------------------------------------
# Fix 5 — Sidebar: pipeline progress + Restart only (no API key inputs)
# ---------------------------------------------------------------------------
def render_sidebar() -> None:
    with st.sidebar:
        st.markdown(
            "<h1 style='color:#9D84D9;font-size:1.5rem;font-weight:800;text-align: center;"
            "letter-spacing:.04em;margin-bottom:0'>🧠 MarketMind AI</h1>"
            "<p style='font-size:.73rem;color:#A49EC6;margin-top:3px;text-align: center;'>"
            "Saudi Arabia GTM Intelligence Platform</p>",
            unsafe_allow_html=True,
        )
        st.divider()

        # ── Pipeline tracker ──────────────────────────────────────────────
        st.markdown("## Pipeline")
        current = st.session_state.stage
        for num, icon, title, subtitle in [
            (1, "🔍", "Research Agent",   "Saudi Arabia Market Analysis"),
            (2, "🗺️", "Strategy Agent",   "ICP · Budget · Calendar"),
            (3, "✍️", "Content Agent",     "Posts · Email · Ads · Video"),
            (4, "✅", "Brand Alignment",   "Final Quality Gate"),
        ]:
            done   = current > num
            active = current == num
            cls = "pill-done" if done else ("pill-active" if active else "pill-pending")
            txt = "✓ Done" if done else ("▶ Active" if active else "Pending")
            
            st.markdown(
                f'<span class="stage-pill {cls}">{txt}</span><br>'
                f'<span style="font-weight:700;font-size:.87rem">{icon} {title}</span><br>'
                f'<span style="font-size:.73rem;color:#A49EC6">{subtitle}</span>',
                unsafe_allow_html=True,
            )
            st.markdown("<div style='margin-bottom:10px'></div>", unsafe_allow_html=True)

        # ── API key status (read-only indicator, no input) ────────────────
        st.divider()
        has_ok = bool(os.environ.get("OPENAI_API_KEY","").strip())
        has_tk = bool(os.environ.get("TAVILY_API_KEY","").strip())
        both   = has_ok and has_tk
        st.markdown(
            f"<span style='font-size:.77rem;color:{'#407D56' if both else '#E5A4CB'}'>"
            f"{'● API keys loaded from environment' if both else '● API keys missing — check .env'}"
            "</span>",
            unsafe_allow_html=True,
        )

        # ── Token & cost panel ───────────────────────────────────────────
        if st.session_state.total_tokens > 0:
            st.divider()
            st.markdown("## 💰 LLM Usage")
            st.markdown(
                f"<div style='font-size:.78rem;color:#A49EC6;line-height:1.9'>"
                f"<b style='color:#9D84D9'>Prompt tokens</b><br>"
                f"{st.session_state.total_prompt_tokens:,}<br>"
                f"<b style='color:#9D84D9'>Completion tokens</b><br>"
                f"{st.session_state.total_completion_tokens:,}<br>"
                f"<b style='color:#9D84D9'>Total tokens</b><br>"
                f"{st.session_state.total_tokens:,}<br>"
                f"<b style='color:#9D84D9'>Est. cost (USD)</b><br>"
                f"<span style='font-size:.92rem;font-weight:800;color:#88C3A4'>"
                f"${st.session_state.total_cost_usd:.4f}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
            with st.expander("📋 Call breakdown", expanded=False):
                for entry in st.session_state.call_log:
                    st.markdown(
                        f"<div style='font-size:.72rem;color:#A49EC6;margin-bottom:6px'>"
                        f"<b style='color:#9D84D9'>{entry['agent']}</b><br>"
                        f"↳ {entry['prompt']:,} + {entry['completion']:,} tok"
                        f" &nbsp;|&nbsp; ${entry['cost']:.5f}"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

        # ── Restart button at the bottom ──────────────────────────────────
        st.divider()
        if st.button("🔄 Restart Pipeline", use_container_width=True):
            for k, v in DEFAULTS.items():
                st.session_state[k] = v
            st.rerun()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def hr():
    st.markdown('<hr class="mm-divider">', unsafe_allow_html=True)

def section_header(icon: str, title: str, subtitle: str = "") -> None:
    st.markdown(
        f"<div style='margin-bottom:4px'>"
        f"<span style='font-size:1.7rem'>{icon}</span> "  # تم تكبير حجم الأيقونة بالتناسب
        f"<span style='font-size:1.45rem; font-weight:800; color:#5B4F7B'>{title}</span></div>"  # تم تعديل اللون والخط
        + (f"<p style='color:#A49EC6; margin:0 0 10px 0; font-size:.95rem'>{subtitle}</p>"  # تم تعديل لون وحجم الخط الفرعي
           if subtitle else ""),
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Output display functions — results shown at TOP of each stage
# ---------------------------------------------------------------------------

def show_research_output() -> None:
    if not st.session_state.research_report:
        return
    with st.expander("📊 Research Agent Output", expanded=True):
        st.markdown(st.session_state.research_report)
        hr()
        pdf_download_button(
            "📥 Download Research Report (PDF)",
            "Research Agent Output",
            st.session_state.research_report,
            "research_report.pdf",
            "dl_research",
        )


def show_strategy_output() -> None:
    if not st.session_state.gtm_strategy:
        return
    with st.expander("🗺️ Strategy Agent Output", expanded=True):
        st.markdown(st.session_state.gtm_strategy)

        if st.session_state.calendar_md:
            hr()
            st.markdown("### 📅 30-Day Marketing Calendar")
            st.markdown(st.session_state.calendar_md)
            hr()
            cal_text = "# 30-Day Marketing Calendar\n\n" + st.session_state.calendar_md
            pdf_download_button(
                "🖨️ Download / Print Calendar (PDF)",
                "30-Day Marketing Calendar",
                cal_text,
                "marketing_calendar.pdf",
                "dl_calendar",
            )

        hr()
        full_strategy = st.session_state.gtm_strategy
        if st.session_state.calendar_md:
            full_strategy += "\n\n## 30-Day Marketing Calendar\n" + st.session_state.calendar_md
        pdf_download_button(
            "📥 Download GTM Strategy (PDF)",
            "Strategy Agent Output",
            full_strategy,
            "gtm_strategy.pdf",
            "dl_strategy",
        )


def show_content_output() -> None:
    """
    Fix 2 — nested expander replaced:
    All internal 'Copy' blocks use st.container(border=True) instead of
    st.expander() to comply with Streamlit's no-nested-expander rule.
    """
    pkg: ContentPackage | None = st.session_state.content_package
    if not pkg:
        return

    refined: RefinedContent | None = st.session_state.refined_content
    posts = refined.refined_posts        if refined else pkg.linkedin_x_posts
    email = refined.refined_email        if refined else pkg.marketing_email
    ads   = refined.refined_ads          if refined else pkg.ad_copy_sets
    vs    = refined.refined_video_script if refined else pkg.video_script

    with st.expander("✍️ Content Agent Output", expanded=True):
        tab_posts, tab_email, tab_ads, tab_video = st.tabs(
            ["📱 Social Posts", "✉️ Email", "📣 Ad Copy", "🎬 Video Script"]
        )

        # ── Social Posts ──────────────────────────────────────────────────
        with tab_posts:
            for i, post in enumerate(posts, 1):
                st.markdown(
                    f'<div class="post-card">'
                    f'<div class="platform-tag">Post {i} — {post.platform} · {post.post_type}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(f"**🪝 Hook**\n\n{post.hook}")
                st.markdown(f"**📝 Body**\n\n{post.body}")
                st.markdown(f"**🎯 CTA**\n\n{post.call_to_action}")
                if post.hashtags:
                    st.markdown(" ".join(f"`{h}`" for h in post.hashtags))

                plain = f"{post.hook}\n\n{post.body}\n\n{post.call_to_action}"
                if post.hashtags:
                    plain += "\n\n" + " ".join(post.hashtags)

                # ✅ Fix 2: container(border=True) instead of expander()
                st.caption(f"📋 Copy Post {i}:")
                with st.container(border=True):
                    st.text_area(
                        "", value=plain, height=130,
                        label_visibility="collapsed",
                        key=f"cp_post_{i}",
                    )
                st.divider()

        # ── Email ─────────────────────────────────────────────────────────
        with tab_email:
            cm, cp = st.columns([1, 2])
            with cm:
                st.markdown(f"**Subject:** {email.subject_line}")
                st.markdown(f"**Preview:** *{email.preview_text}*")
                st.markdown(f"**CTA:** `{email.primary_cta}`")
            with cp:
                body_html = "".join(f"<p>{p}</p>" for p in email.body_paragraphs)
                st.markdown(
                    f'<div class="email-preview">'
                    f'<div class="email-subject">{email.subject_line}</div>'
                    f'<div class="email-preview-text">{email.preview_text}</div>'
                    f"<p>{email.greeting}</p>{body_html}"
                    f'<a class="email-cta" href="#">{email.primary_cta}</a>'
                    f"<br><br><p>{email.sign_off}</p></div>",
                    unsafe_allow_html=True,
                )

            plain_email = (
                f"Subject: {email.subject_line}\nPreview: {email.preview_text}\n\n"
                f"{email.greeting}\n\n"
                + "\n\n".join(email.body_paragraphs)
                + f"\n\n{email.primary_cta}\n\n{email.sign_off}"
            )
            # ✅ Fix 2: container(border=True) instead of expander()
            st.caption("📋 Copy Email:")
            with st.container(border=True):
                st.text_area(
                    "", value=plain_email, height=180,
                    label_visibility="collapsed",
                    key="cp_email",
                )

        # ── Ad Copy ───────────────────────────────────────────────────────
        with tab_ads:
            for i, ad in enumerate(ads, 1):
                st.markdown(
                    f'<div class="ad-card">'
                    f'<div class="platform-tag">Ad {i} — {ad.placement}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(f"**Headline:** {ad.headline}")
                st.markdown(f"**Body:** {ad.body_copy}")
                st.markdown(f"**CTA:** `{ad.cta_button}`")

                ad_plain = f"Headline: {ad.headline}\nBody: {ad.body_copy}\nCTA: {ad.cta_button}"
                # ✅ Fix 2: container(border=True) instead of expander()
                st.caption(f"📋 Copy Ad {i}:")
                with st.container(border=True):
                    st.text_area(
                        "", value=ad_plain, height=90,
                        label_visibility="collapsed",
                        key=f"cp_ad_{i}",
                    )
                st.divider()

        # ── Video Script ──────────────────────────────────────────────────
        with tab_video:
            st.markdown(f"### 🎬 {vs.title}")
            c1, c2, c3 = st.columns(3)
            c1.metric("Platform", vs.target_platform)
            c2.metric("Duration", f"{vs.total_duration_seconds}s")
            c3.metric("Scenes",   len(vs.scenes))
            st.markdown(f"**Objective:** {vs.objective}")
            hr()
            for sc in vs.scenes:
                st.markdown(
                    f'<div class="video-card">'
                    f'<div class="platform-tag">Scene {sc.scene_number} — {sc.duration_seconds}s</div>'
                    f'<strong>📷 Visual Cue:</strong><br>{sc.visual_cue}<br><br>'
                    f'<strong>🎙️ Dialogue:</strong><br><em>{sc.spoken_dialogue}</em>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            st.markdown(f"**🎯 Closing CTA:** {vs.closing_cta}")

            script_text = (
                f"# VIDEO SCRIPT: {vs.title}\n"
                f"Platform: {vs.target_platform} | Duration: {vs.total_duration_seconds}s\n"
                f"Objective: {vs.objective}\n\n"
            )
            for sc in vs.scenes:
                script_text += (
                    f"## Scene {sc.scene_number} ({sc.duration_seconds}s)\n"
                    f"Visual: {sc.visual_cue}\nDialogue: {sc.spoken_dialogue}\n\n"
                )
            script_text += f"Closing CTA: {vs.closing_cta}"

            # ✅ Fix 2: container(border=True) instead of expander()
            st.caption("📋 Copy Full Script:")
            with st.container(border=True):
                st.text_area(
                    "", value=script_text, height=230,
                    label_visibility="collapsed",
                    key="cp_script",
                )

        # Content PDF download (outside tabs, still inside expander)
        hr()
        content_text = _content_to_text(posts, email, ads, vs)
        pdf_download_button(
            "📥 Download Content Package (PDF)",
            "Content Agent Output",
            content_text,
            "content_package.pdf",
            "dl_content",
        )


def show_brand_alignment_output() -> None:
    refined: RefinedContent | None = st.session_state.refined_content
    if not refined:
        return
    with st.expander("✅ Brand Alignment Agent Output", expanded=True):
        st.markdown(f"**Alignment Summary**\n\n{refined.alignment_summary}")
        hr()
        st.markdown(f"**Brand Voice Guidelines**\n\n{refined.brand_voice_guidelines}")
        hr()
        st.info(f"**Aligned Value Proposition:** {refined.aligned_value_proposition}")
        hr()
        st.markdown("**Content Guardrails Applied:**")
        for g in refined.content_guardrails:
            st.markdown(f'<div class="guardrail-item">🛡️ {g}</div>', unsafe_allow_html=True)
        hr()
        brand_text = (
            "# BRAND ALIGNMENT REPORT\n\n"
            f"## Alignment Summary\n{refined.alignment_summary}\n\n"
            f"## Brand Voice Guidelines\n{refined.brand_voice_guidelines}\n\n"
            f"## Value Proposition\n{refined.aligned_value_proposition}\n\n"
            "## Content Guardrails\n"
            + "\n".join(f"- {g}" for g in refined.content_guardrails)
        )
        pdf_download_button(
            "📥 Download Brand Alignment Report (PDF)",
            "Brand Alignment Agent Output",
            brand_text,
            "brand_alignment_report.pdf",
            "dl_brand",
        )


def _content_to_text(posts, email, ads, video_script) -> str:
    parts = ["# CONTENT PACKAGE\n\n## Social Posts\n"]
    for i, p in enumerate(posts, 1):
        parts.append(
            f"### Post {i} [{p.platform} / {p.post_type}]\n"
            f"Hook: {p.hook}\n\n{p.body}\n\nCTA: {p.call_to_action}\n"
            + (" ".join(p.hashtags) + "\n" if p.hashtags else "")
        )
    parts.append(
        f"\n## Marketing Email\nSubject: {email.subject_line}\n"
        f"Preview: {email.preview_text}\n\n{email.greeting}\n\n"
        + "\n\n".join(email.body_paragraphs)
        + f"\n\nCTA: {email.primary_cta}\n\n{email.sign_off}\n"
    )
    parts.append("\n## Ad Copy Sets\n")
    for i, a in enumerate(ads, 1):
        parts.append(
            f"### Ad {i} [{a.placement}]\n"
            f"Headline: {a.headline}\nBody: {a.body_copy}\nCTA: {a.cta_button}\n"
        )
    vs = video_script
    parts.append(
        f"\n## Video Script\nTitle: {vs.title}\nPlatform: {vs.target_platform}\n"
        f"Duration: {vs.total_duration_seconds}s\nObjective: {vs.objective}\n\n"
    )
    for sc in vs.scenes:
        parts.append(
            f"Scene {sc.scene_number} ({sc.duration_seconds}s)\n"
            f"Visual: {sc.visual_cue}\nDialogue: {sc.spoken_dialogue}\n\n"
        )
    parts.append(f"Closing CTA: {vs.closing_cta}")
    return "".join(parts)

# ---------------------------------------------------------------------------
# Stage 1a — Profile form
# ---------------------------------------------------------------------------
def render_form() -> None:
    section_header(
        "🔍", "Stage 1 — Research Agent",
        "Enter your company profile. The agent will execute live Saudi Arabia market research.",
    )
    hr()

    profile_source = st.radio(
        "Profile input method:", ["✏️ Manual Entry", "📄 Upload PDF"], horizontal=True
    )
    pdf_text = ""
    if profile_source == "📄 Upload PDF":
        uploaded = st.file_uploader("Upload company profile PDF", type=["pdf"])
        if uploaded:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded.read())
                tmp_path = tmp.name
            with st.spinner("Extracting PDF text…"):
                pdf_text = load_and_validate_pdf(tmp_path) or ""
            Path(tmp_path).unlink(missing_ok=True)
            if pdf_text:
                st.success(f"PDF parsed — {len(pdf_text)} chars extracted.")
            else:
                st.error("Could not extract text — use manual entry.")

    hr()
    with st.form("profile_form"):
        c1, c2 = st.columns(2)
        with c1:
            company_name = st.text_input(
                "Company Name *",
                placeholder="e.g. Acme Technologies Saudi Arabia",
            )
            brand_voice = st.text_area(
                "Brand Voice / Tone *", height=110,
                placeholder="e.g. Professional yet approachable, data-driven",
            )
        with c2:
            products = st.text_area(
                "Products / Services & Pricing *", height=110,
                placeholder="e.g. SaaS ERP — Starter SAR 1,800/mo, Pro SAR 5,400/mo",
            )
            past_mkt = st.text_area(
                "Past Successful Marketing (optional)", height=110,
                placeholder="e.g. LinkedIn series drove 3x pipeline in Q3 2023",
            )

        # Fix 4 — English-only budget label (Arabic text removed)
        st.markdown("**Marketing Budget**")
        bc1, bc2 = st.columns([2, 1])
        with bc1:
            budget_amount = st.number_input(
                "Amount (SAR)",
                min_value=1_000, max_value=10_000_000,
                value=20_000, step=1_000,
                help="Total marketing budget in Saudi Riyals (SAR).",
            )
        with bc2:
            budget_period = st.selectbox(
                "Period",
                ["per month", "per quarter", "per campaign", "per year"],
            )

        submitted = st.form_submit_button(
            "Next: Review & Approve Search →", type="primary", use_container_width=True
        )

    if not submitted:
        return

    errors: list[str] = []
    using_pdf = bool(profile_source == "📄 Upload PDF" and pdf_text)
    if not using_pdf:
        if not company_name.strip(): errors.append("Company Name is required.")
        if not products.strip():     errors.append("Products / Services is required.")
        if not brand_voice.strip():  errors.append("Brand Voice is required.")
    for label, val in [
        ("Company Name", company_name),
        ("Products", products),
        ("Brand Voice", brand_voice),
    ]:
        if val.strip() and not validate_input_text(val):
            errors.append(f"'{label}' contains potentially unsafe content.")
    if past_mkt.strip() and not validate_input_text(past_mkt):
        errors.append("'Past Marketing' contains potentially unsafe content.")
    if errors:
        for e in errors:
            st.error(e)
        return

    st.session_state.form_company_name   = company_name
    st.session_state.form_products       = products
    st.session_state.form_brand_voice    = brand_voice
    st.session_state.form_past_marketing = past_mkt
    st.session_state.form_pdf_text       = pdf_text
    st.session_state.form_budget         = f"SAR {budget_amount:,} {budget_period}"
    st.session_state.substage            = "hitl"
    st.rerun()
# ---------------------------------------------------------------------------
# Stage 1b — HITL gate
# ---------------------------------------------------------------------------
def render_hitl() -> None:
    section_header("🔐", "Human-in-the-Loop Approval")
    name   = st.session_state.form_company_name or "Company (from PDF)"
    budget = st.session_state.form_budget
    st.success(f"✅ Profile saved for: **{name}** |  Budget: **{budget}**")
    st.info(
        "The next step calls the **Tavily Search API** for live Saudi Arabia "
        "market and competitor intelligence. This will consume API credits.\n\n"
        f"**Company:** {name}  |  **Budget:** {budget}\n\n"
        "Approve to launch the full 4-agent pipeline."
    )
    c1, c2, _ = st.columns([1, 1, 4])
    with c1:
        if st.button("✅ Approve & Run", type="primary", use_container_width=True):
            st.session_state.substage = "running"
            st.rerun()
    with c2:
        if st.button("✏️ Edit Profile", use_container_width=True):
            st.session_state.substage = "form"
            st.rerun()


# ---------------------------------------------------------------------------
# LangSmith @traceable wrappers + token/cost tracking — ONE place only.
# ---------------------------------------------------------------------------

@_traceable(name="Stage1_ParseProfileFromPDF")
def _trace_parse_pdf(agent, pdf_text):
    return _inject(agent, "Stage1_ParseProfileFromPDF").parse_profile_from_text(pdf_text)

@_traceable(name="Stage1_BuildSearchQueries")
def _trace_build_queries(agent, profile):
    return agent.build_search_queries(profile)

@_traceable(name="Stage1_SynthesiseReport")
def _trace_synthesise(agent, profile, context):
    return _inject(agent, "Stage1_SynthesiseReport").synthesise_report(profile, context)

@_traceable(name="Stage1_PersistResearch")
def _trace_persist_research(agent, profile, research_report):
    return agent.persist_research(profile, research_report)

@_traceable(name="Stage2_StrategyAgent")
def _trace_strategy(agent, profile, research_doc_id):
    return _inject(agent, "Stage2_StrategyAgent").run(
        profile=profile, research_doc_id=research_doc_id,
    )

@_traceable(name="Stage3_ContentAgent")
def _trace_content(agent, profile, strategy_doc_id):
    return _inject(agent, "Stage3_ContentAgent").run(
        profile=profile, strategy_doc_id=strategy_doc_id,
    )

@_traceable(name="Stage4_BrandAlignmentAgent")
def _trace_brand(agent, profile, pkg, content_doc_id):
    return _inject(agent, "Stage4_BrandAlignmentAgent").run(
        profile=profile, pkg=pkg, content_doc_id=content_doc_id,
    )

# ---------------------------------------------------------------------------
# Stage 1c — Research Agent execution
# ---------------------------------------------------------------------------
def run_research_pipeline() -> None:
    section_header("🔍", "Research Agent Running…")
    openai_client, tavily_client, llm = resolve_clients()
    agent = ResearchAgent(openai_client=openai_client, tavily_client=tavily_client, llm=llm)

    company_name = st.session_state.form_company_name
    products     = st.session_state.form_products
    brand_voice  = st.session_state.form_brand_voice
    past_mkt     = st.session_state.form_past_marketing
    pdf_text     = st.session_state.form_pdf_text
    budget       = st.session_state.form_budget

    bar    = st.progress(0, text="Building company profile…")
    status = st.empty()

    try:
        if pdf_text:
            status.info("Parsing profile from PDF via LLM…")
            profile = _trace_parse_pdf(agent, pdf_text)
            if company_name.strip(): profile.company_name        = company_name.strip()
            if brand_voice.strip():  profile.brand_voice         = brand_voice.strip()
            if products.strip():     profile.products_and_prices = products.strip()
            profile.marketing_budget = budget
        else:
            profile = CompanyProfile(
                company_name=company_name.strip(),
                products_and_prices=products.strip(),
                brand_voice=brand_voice.strip(),
                past_successful_marketing=past_mkt.strip() or "No past marketing plans provided.",
                marketing_budget=budget,
            )

        queries  = _trace_build_queries(agent, profile)
        snippets: list[str] = []
        for i, q in enumerate(queries):
            status.info(f"Saudi Arabia search ({i+1}/{len(queries)}): {q[:60]}…")
            try:
                res = tavily_client.search(q, search_depth="advanced", max_results=5)
                for r in res.get("results", []):
                    s = r.get("content", "")
                    if s:
                        snippets.append(s[:500])
            except Exception as exc:
                status.warning(f"Search note: {exc}")
            bar.progress(
                10 + int((i + 1) / len(queries) * 45),
                text=f"Searching ({i+1}/{len(queries)})…",
            )

        context = (
            "\n\n---\n\n".join(snippets) if snippets
            else "No live results retrieved. Generating from parametric knowledge."
        )

        bar.progress(60, text="Synthesising Saudi Arabia research report…")
        status.info("Synthesising report…")
        research_report = _trace_synthesise(agent, profile, context)

        bar.progress(90, text="Persisting to ChromaDB…")
        research_doc_id = _trace_persist_research(agent, profile, research_report)

        bar.progress(100, text="✓ Research complete!")
        status.empty()
        time.sleep(0.3)

        st.session_state.profile         = profile
        st.session_state.research_report = research_report
        st.session_state.research_doc_id = research_doc_id
        st.session_state.substage        = "form"
        st.session_state.stage           = 2
        st.rerun()

    except Exception as exc:
        bar.empty()
        status.empty()
        st.error(f"❌ Research Agent error: {exc}")
        st.exception(exc)
        if st.button("← Go back"):
            st.session_state.substage = "hitl"
            st.rerun()

# ---------------------------------------------------------------------------
# Stage 2 — Strategy Agent
# ---------------------------------------------------------------------------
def render_stage_2() -> None:
    # Previous outputs at top
    show_research_output()
    hr()
    section_header(
        "🗺️", "Stage 2 — Strategy Agent",
        "Generating budget-aware ICP, GTM strategy, and 30-day marketing calendar.",
    )
    profile: CompanyProfile = st.session_state.profile
    st.info(f"**Marketing Budget:** {profile.marketing_budget}  — strategy will be designed around this budget.")

    if st.button("🗺️ Generate ICP, Strategy & Calendar", type="primary"):
        openai_client, _, llm = resolve_clients()
        with st.spinner("Building Saudi Arabia GTM strategy and 30-day calendar…"):
            try:
                agent = StrategyAgent(openai_client=openai_client, llm=llm)
                strategy_text, calendar_md, strategy_doc_id = _trace_strategy(
                    agent, profile, st.session_state.research_doc_id,
                )
                st.session_state.gtm_strategy    = strategy_text
                st.session_state.calendar_md     = calendar_md
                st.session_state.strategy_doc_id = strategy_doc_id
                st.session_state.stage           = 3
                st.rerun()
            except Exception as exc:
                st.error(f"❌ Strategy Agent error: {exc}")
                st.exception(exc)

# ---------------------------------------------------------------------------
# Stage 3 — Content Agent
# ---------------------------------------------------------------------------
def render_stage_3() -> None:
    show_research_output()
    show_strategy_output()
    hr()
    section_header(
        "✍️", "Stage 3 — Content Agent",
        "Generating social posts, marketing email, ad copy, and video script.",
    )

    if st.button("✍️ Generate Full Content Package", type="primary"):
        openai_client, _, llm = resolve_clients()
        with st.spinner("Generating Saudi Arabia content package…"):
            try:
                agent = ContentAgent(openai_client=openai_client, llm=llm)
                content_package, content_doc_id = _trace_content(
                    agent, st.session_state.profile, st.session_state.strategy_doc_id,
                )
                st.session_state.content_package = content_package
                st.session_state.content_doc_id  = content_doc_id
                st.session_state.stage           = 4
                st.rerun()
            except Exception as exc:
                st.error(f"❌ Content Agent error: {exc}")
                st.exception(exc)

# ---------------------------------------------------------------------------
# Stage 4 — Brand Alignment Agent (final gate)
# ---------------------------------------------------------------------------
def render_stage_4() -> None:
    show_research_output()
    show_strategy_output()
    show_content_output()
    hr()
    section_header(
        "✅", "Stage 4 — Brand Alignment Agent",
        "Final quality gate: auditing and refining ALL content for brand consistency and KSA compliance.",
    )

    if st.button("✅ Run Final Brand Alignment", type="primary"):
        openai_client, _, llm = resolve_clients()
        with st.spinner("Running final brand alignment audit…"):
            try:
                agent = BrandAlignmentAgent(openai_client=openai_client, llm=llm)
                refined, brand_doc_id = _trace_brand(
                    agent,
                    st.session_state.profile,
                    st.session_state.content_package,
                    st.session_state.content_doc_id,
                )
                st.session_state.refined_content = refined
                st.session_state.brand_doc_id    = brand_doc_id
                st.session_state.stage           = 5
                st.rerun()
            except Exception as exc:
                st.error(f"❌ Brand Alignment error: {exc}")
                st.exception(exc)

# ---------------------------------------------------------------------------
# Stage 5 — Final results dashboard (tabbed layout)
# ---------------------------------------------------------------------------
def render_final_results() -> None:
    profile: CompanyProfile = st.session_state.profile

    st.markdown(
        f'<div class="result-banner">'
        f'✅ Pipeline Complete — {profile.company_name} | Saudi Arabia GTM Intelligence'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Unified tab strip — one tab per agent, most valuable output first ──
    tab_brand, tab_content, tab_strategy, tab_research = st.tabs([
        "✅ Brand Alignment",
        "✍️ Content",
        "🎯 Strategy",
        "🔍 Research",
    ])

    with tab_brand:
        show_brand_alignment_output()

    with tab_content:
        show_content_output()

    with tab_strategy:
        show_strategy_output()

    with tab_research:
        show_research_output()

    hr()
    st.markdown(
        f"**Session IDs** &emsp; "
        f"Research: `{st.session_state.research_doc_id}` &emsp; "
        f"Strategy: `{st.session_state.strategy_doc_id}` &emsp; "
        f"Content: `{st.session_state.content_doc_id}` &emsp; "
        f"Brand: `{st.session_state.brand_doc_id}`"
    )

# ---------------------------------------------------------------------------
# Main router
# ---------------------------------------------------------------------------
def main() -> None:
    render_sidebar()

    st.markdown(
        """
        <div style="text-align: center;">
            <h1 style='font-size:3.5rem; font-weight:800; margin-bottom:2px; color:#5B4F7B'>
                🧠 MarketMind AI
            </h1>
            <p style='color:#A49EC6; margin:0 0 6px 0; font-size:1.2rem;'>
                Saudi Arabia GTM Intelligence Platform · 4 Agentic AI Specialists
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    stage    = st.session_state.stage
    substage = st.session_state.substage

    if stage == 1:
        if substage == "form":     render_form()
        elif substage == "hitl":    render_hitl()
        elif substage == "running": run_research_pipeline()
    elif stage == 2: render_stage_2()
    elif stage == 3: render_stage_3()
    elif stage == 4: render_stage_4()
    elif stage >= 5: render_final_results()

if __name__ == "__main__":
    main()