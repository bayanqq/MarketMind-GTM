"""
agents/strategy_agent.py
------------------------
Agent 2 — Strategy Agent | MarketMind AI
Architecture: v3 — ChatOpenAI Dependency Injection for LangSmith Tracing
 
Changes from v2
───────────────
• __init__ now accepts `llm: ChatOpenAI` alongside `openai_client`.
• beta.chat.completions.parse replaced with self.llm.with_structured_output().
• self.openai_client kept ONLY for embeddings.
• All Pydantic schemas, budget engine, and prompts .
"""


from __future__ import annotations

import logging
import re
import uuid

from openai import OpenAI
from pydantic import BaseModel, Field

from agents.research_agent import CompanyProfile
from database_setup import get_market_research_collection, get_icp_and_strategy_collection

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LangChain imports
# ---------------------------------------------------------------------------
try:
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI

    _STRATEGY_TEMPLATE = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a Senior Go-To-Market Strategist specialising in Saudi Arabia and GCC B2B "
            "markets with a strong background in financial modelling for marketing budgets.\n\n"
            "CRITICAL ICP RULES:\n"
            "- Define segments using ONLY: sectors, company sizes, revenue bands, job titles, "
            "  pain points, and behavioural traits.\n"
            "- DO NOT generate fictional names, social handles, or invented personas.\n"
            "- ALL budget allocations must use the exact SAR integer provided and sum correctly.\n"
            "- Use Vision 2030 terminology (NTP, NEOM, Nitaqat, ZATCA, GOSI) where relevant."
        )),
        ("human", "{user_prompt}"),
    ])
    _LANGCHAIN_AVAILABLE = True
    logger.info("[StrategyAgent] LangChain prompt template loaded.")
except ImportError:
    _LANGCHAIN_AVAILABLE = False
    ChatOpenAI = None
    logger.info("[StrategyAgent] LangChain not installed — using inline prompts.")


# ---------------------------------------------------------------------------
# Pydantic schemas 
# ---------------------------------------------------------------------------

class ICPSegment(BaseModel):
    segment_name: str        = Field(..., description="Descriptive label (e.g. 'Mid-Market CFOs in KSA').")
    firmographics: str       = Field(..., description="Sector, size, revenue band, KSA geography.")
    job_titles: list[str]    = Field(..., description="Specific decision-maker and influencer titles.")
    pain_points: list[str]   = Field(..., description="3-5 professional pain points.")
    behavioural_traits: str  = Field(..., description="Research, evaluation, and buying behaviour.")
    value_drivers: str       = Field(..., description="Business outcomes they seek.")


class MarketingCalendarEntry(BaseModel):
    day: int               = Field(..., description="Day number (1-30).")
    activity: str          = Field(..., description="Marketing activity type.")
    description: str       = Field(..., description="One-line description of what to do.")
    channel: str           = Field(..., description="LinkedIn / Email / Instagram / Snapchat / etc.")
    budget_allocation: str = Field(..., description="SAR amount for this activity (e.g. 'SAR 1,500').")


class GTMStrategyDocument(BaseModel):
    executive_summary: str         = Field(...)
    icp_segments: list[ICPSegment] = Field(...)
    value_proposition: str         = Field(...)
    messaging_pillars: list[str]   = Field(...)
    channel_strategy: str          = Field(...)
    budget_breakdown: str          = Field(
        ...,
        description="Mathematical budget allocation table: channel, SAR amount, %, rationale."
    )
    past_marketing_scaled: str     = Field(...)
    execution_roadmap: str         = Field(...)
    kpis: list[str]                = Field(...)
    risk_mitigation: str           = Field(...)
    content_calendar: list[MarketingCalendarEntry] = Field(
        ..., description="30-day content calendar with SAR budget per entry."
    )


# ---------------------------------------------------------------------------
# Budget parsing utility 
# ---------------------------------------------------------------------------

def _parse_budget_sar(budget_str: str) -> int:
    if not budget_str or budget_str.strip().lower() in ("not specified", ""):
        return 10_000
    cleaned = budget_str.replace(",", "").replace(".", "")
    numbers = re.findall(r"\d+", cleaned)
    if numbers:
        value = int(numbers[0])
        return max(1_000, min(10_000_000, value))
    return 10_000


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class StrategyAgent:
    """
    Agent 2 — Saudi Arabia GTM strategist.
    Uses ChatOpenAI.with_structured_output() for LangSmith-traced structured
    output generation. Falls back to openai_client.beta.parse() if needed.
    Embeddings always use openai_client directly.
    """

    CHAT_MODEL      = "gpt-4o-mini"
    EMBEDDING_MODEL = "text-embedding-3-small"

    def __init__(
        self,
        openai_client: OpenAI,
        llm: "ChatOpenAI | None" = None,
    ) -> None:
        self.openai_client       = openai_client    # embeddings only
        self.research_collection = get_market_research_collection()
        self.strategy_collection = get_icp_and_strategy_collection()
        if llm is not None:
            self.llm = llm
        elif _LANGCHAIN_AVAILABLE:
            self.llm = ChatOpenAI(model=self.CHAT_MODEL, temperature=0.4)
        else:
            self.llm = None

    # ------------------------------------------------------------------
    # ChromaDB retrieval 
    # ------------------------------------------------------------------

    def fetch_research(self, research_doc_id: str) -> str:
        result = self.research_collection.get(ids=[research_doc_id], include=["documents"])
        if not result["documents"] or not result["documents"][0]:
            raise ValueError(f"Research doc '{research_doc_id}' not found.")
        return result["documents"][0]

    # ------------------------------------------------------------------
    # Strategy generation — uses with_structured_output()
    # ------------------------------------------------------------------

    def generate_strategy(self, profile: CompanyProfile, research_report: str) -> GTMStrategyDocument:
        """Generate GTM strategy with mathematically correct budget distribution."""
        budget_sar   = _parse_budget_sar(profile.marketing_budget)
        budget_label = f"SAR {budget_sar:,}"

        channel_allocs = {
            "LinkedIn (content + sponsored)": int(budget_sar * 0.30),
            "Google Search / Display":         int(budget_sar * 0.20),
            "Snapchat Ads":                    int(budget_sar * 0.15),
            "Instagram / Reels":               int(budget_sar * 0.10),
            "WhatsApp Business API":           int(budget_sar * 0.05),
            "Events / Webinars":               int(budget_sar * 0.10),
            "Content Production":              int(budget_sar * 0.07),
            "Reserve / Testing":               int(budget_sar * 0.03),
        }
        alloc_text = "\n".join(
            f"  {ch}: SAR {amt:,} ({round(amt/budget_sar*100)}%%)"
            for ch, amt in channel_allocs.items()
        )
        daily_avg = budget_sar // 30

        past_note = (
            f"Past Marketing to Scale:\n{profile.past_successful_marketing}"
            if profile.past_successful_marketing != "No past marketing plans provided."
            else "No prior marketing history — build from first principles."
        )

        system_content = (
            "You are a Senior Go-To-Market Strategist specialising in Saudi Arabia and GCC B2B "
            "markets with expertise in financial modelling for marketing budgets.\n\n"
            "CRITICAL ICP RULES:\n"
            "- Define segments using ONLY: sectors, company sizes, revenue bands, job titles, "
            "  pain points, and behavioural traits.\n"
            "- DO NOT generate fictional names, social handles, or invented personas.\n"
            "- ALL budget figures must use the provided SAR amounts exactly — no random numbers.\n"
            "- Use Vision 2030 terminology (NTP, NEOM, Nitaqat, ZATCA, GOSI) where relevant."
        )

        user_content = (
            f"Company: {profile.company_name}\n"
            f"Products & Pricing: {profile.products_and_prices}\n"
            f"Brand Voice: {profile.brand_voice}\n"
            f"Total Monthly Marketing Budget: {budget_label}\n"
            f"{past_note}\n\n"
            "MATHEMATICAL BUDGET ALLOCATION (use these exact SAR figures throughout):\n"
            f"{alloc_text}\n"
            f"Daily average across 30-day calendar: SAR {daily_avg:,}\n\n"
            "Produce a complete Saudi Arabia GTM strategy:\n\n"
            "EXECUTIVE SUMMARY: Core strategic intent and primary KSA market opportunity.\n\n"
            "ICP SEGMENTS (2-3): firmographics, job titles, pain points, behavioural traits, value drivers.\n\n"
            "VALUE PROPOSITION: Localised for Saudi/GCC B2B buyers.\n\n"
            "MESSAGING PILLARS: 3-5 themes resonating with KSA B2B decision-makers.\n\n"
            "CHANNEL STRATEGY: Use the exact channel allocations above. Explain ROI rationale per channel.\n\n"
            "BUDGET BREAKDOWN: Render the allocation table above as a formatted breakdown. "
            f"Total must equal {budget_label}.\n\n"
            "SCALING PAST SUCCESSES: Which past tactics transfer to KSA and how to localise them.\n\n"
            "90-DAY ROADMAP: Month 1, 2, 3 with specific deliverables.\n\n"
            "KPIs: Quantifiable targets per phase.\n\n"
            "RISK MITIGATION: Top 3 KSA-specific risks with contingency plans.\n\n"
            "CONTENT CALENDAR: Generate exactly 30 entries. Each entry must include:\n"
            "  - day (1-30), activity type, one-line description, channel,\n"
            f"  - budget_allocation as a realistic SAR amount based on the daily average of SAR {daily_avg:,}.\n"
            "  Include: LinkedIn posts, cold emails, Instagram stories, Snapchat ads,\n"
            "  WhatsApp follow-ups, webinars, and engagement activities.\n\n"
            f"Saudi Arabia Market Research:\n{research_report}"
        )

        # ── Primary path: ChatOpenAI.with_structured_output() ─────────────
        if _LANGCHAIN_AVAILABLE and self.llm is not None:
            try:
                structured_llm = self.llm.with_structured_output(GTMStrategyDocument)
                if _LANGCHAIN_AVAILABLE:
                    msgs = _STRATEGY_TEMPLATE.format_messages(user_prompt=user_content)
                else:
                    from langchain_core.messages import SystemMessage, HumanMessage
                    msgs = [SystemMessage(content=system_content),
                            HumanMessage(content=user_content)]
                return structured_llm.invoke(msgs)
            except Exception as exc:
                logger.warning("[StrategyAgent] with_structured_output failed (%s) — fallback.", exc)

        # ── Fallback: raw openai parse() ──────────────────────────────────
        oai_msgs = [{"role": "system", "content": system_content},
                    {"role": "user",   "content": user_content}]
        completion = self.openai_client.beta.chat.completions.parse(
            model=self.CHAT_MODEL,
            messages=oai_msgs,
            response_format=GTMStrategyDocument,
            max_tokens=3500,
        )
        return completion.choices[0].message.parsed

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def strategy_to_text(doc: GTMStrategyDocument) -> str:
        lines: list[str] = ["# SAUDI ARABIA GTM STRATEGY\n"]
        lines += ["## Executive Summary", doc.executive_summary, ""]
        lines += ["## ICP Segments"]
        for i, seg in enumerate(doc.icp_segments, 1):
            lines += [
                f"\n### Segment {i}: {seg.segment_name}",
                f"**Firmographics:** {seg.firmographics}",
                f"**Job Titles:** {', '.join(seg.job_titles)}",
                "**Pain Points:**",
                *[f"  - {pp}" for pp in seg.pain_points],
                f"**Behavioural Traits:** {seg.behavioural_traits}",
                f"**Value Drivers:** {seg.value_drivers}",
            ]
        lines += ["", "## Localised Value Proposition", doc.value_proposition, ""]
        lines += ["## Messaging Pillars"]
        lines += [f"  {i}. {p}" for i, p in enumerate(doc.messaging_pillars, 1)]
        lines += ["", "## Channel Strategy", doc.channel_strategy, ""]
        lines += ["## Budget Breakdown", doc.budget_breakdown, ""]
        lines += ["## Scaling Past Marketing Successes", doc.past_marketing_scaled, ""]
        lines += ["## 90-Day Execution Roadmap", doc.execution_roadmap, ""]
        lines += ["## KPIs & Success Metrics"]
        lines += [f"  - {k}" for k in doc.kpis]
        lines += ["", "## Risk Mitigation", doc.risk_mitigation]
        return "\n".join(lines)

    @staticmethod
    def calendar_to_markdown(doc: GTMStrategyDocument) -> str:
        rows = ["| Day | Activity | Description | Channel | Budget |",
                "|-----|----------|-------------|---------|--------|"]
        for e in doc.content_calendar:
            desc = e.description.replace("|", "-")
            rows.append(f"| {e.day} | {e.activity} | {desc} | {e.channel} | {e.budget_allocation} |")
        return "\n".join(rows)

    # ------------------------------------------------------------------
    # Embedding & persistence — openai_client for embeddings ONLY
    # ------------------------------------------------------------------

    def embed_text(self, text: str) -> list[float]:
        return self.openai_client.embeddings.create(
            model=self.EMBEDDING_MODEL, input=text
        ).data[0].embedding

    def persist_strategy(self, profile: CompanyProfile, strategy_text: str,
                         calendar_md: str, research_doc_id: str) -> str:
        doc_id   = f"strategy_{uuid.uuid4().hex}"
        combined = (
            strategy_text
            + "\n\n---\n\n## 30-Day Content Calendar\n"
            + calendar_md
            + "\n\n---\n\n## Context Metadata\n"
            + f"Company: {profile.company_name}\n"
            + f"Budget: {profile.marketing_budget}\n"
            + "Region: Saudi Arabia / Middle East\n"
        )
        embedding = self.embed_text(combined[:8000])
        self.strategy_collection.upsert(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[combined],
            metadatas=[{
                "company_name":           profile.company_name,
                "brand_voice":            profile.brand_voice,
                "marketing_budget":       profile.marketing_budget,
                "budget_sar_integer":     str(_parse_budget_sar(profile.marketing_budget)),
                "source_research_doc_id": research_doc_id,
                "region":                 "Saudi Arabia / Middle East",
                "agent":                  "strategy",
                "has_calendar":           "true",
                "has_budget_breakdown":   "true",
            }],
        )
        return doc_id

    def run(self, profile: CompanyProfile, research_doc_id: str) -> tuple[str, str, str]:
        research    = self.fetch_research(research_doc_id)
        doc         = self.generate_strategy(profile, research)
        text        = self.strategy_to_text(doc)
        calendar_md = self.calendar_to_markdown(doc)
        doc_id      = self.persist_strategy(profile, text, calendar_md, research_doc_id)
        return text, calendar_md, doc_id