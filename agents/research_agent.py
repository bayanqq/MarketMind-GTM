"""
agents/research_agent.py
------------------------
Agent 1 — Research Agent | MarketMind AI
Architecture: v3 — ChatOpenAI Dependency Injection for LangSmith Tracing
 
Changes from v2
───────────────
• __init__ now accepts `llm: ChatOpenAI` alongside `openai_client`.
• All chat.completions.create calls replaced with self.llm.invoke().
• self.openai_client is kept ONLY for embeddings (text-embedding-3-small).
• Pydantic schemas and system prompts 
"""


from __future__ import annotations

import json
import logging
import uuid

from openai import OpenAI, RateLimitError, APIConnectionError, APIStatusError
from pydantic import BaseModel, Field
from tavily import TavilyClient

from database_setup import get_market_research_collection
from utils.security import sanitize_for_query

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LangChain imports
# ---------------------------------------------------------------------------
try:
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.messages import SystemMessage, HumanMessage
    from langchain_openai import ChatOpenAI

    _RESEARCH_SYSTEM_TEMPLATE = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a Principal Market Intelligence Analyst with 15 years of experience "
            "specialising in Saudi Arabia and the GCC under Vision 2030.\n\n"
            "VISION 2030 VOCABULARY — always use these terms natively:\n"
            "  Economic pillars: National Transformation Program (NTP), National Industrial "
            "  Development & Logistics Program (NIDLP), Vision Realization Programs (VRPs), "
            "  NEOM, Red Sea Project, Qiddiya, Diriyah Gate, ROSHN, PIF portfolio companies.\n"
            "  Regulatory bodies: ZATCA (Zakat/Tax/Customs), CITC (Telecom & IT), "
            "  SFDA (Food & Drug), GOSI (Social Insurance), MOMRA (Municipalities), "
            "  MISA (Investment), SAGIA successor rules, Nitaqat Saudisation quotas.\n"
            "  Cultural context: Wasta (relationship capital), Majlis (consultative gatherings), "
            "  Shura (consensus-building), halal certification requirements, gender-inclusive "
            "  workplace mandates post-2019, Ramadan marketing blackout windows.\n"
            "  Digital landscape: Snapchat #1 penetration in KSA, WhatsApp Business API "
            "  for B2B lead nurturing, X (Arabic) for thought leadership, TikTok for Gen-Z "
            "  consumer, LinkedIn for B2B enterprise, Google dominant for search intent.\n\n"
            "Your reports are used by C-suites for high-stakes market entry decisions. "
            "Always compare the client against named competitors in KSA/ME."
        )),
        ("human", "{user_prompt}"),
    ])

    _QUERY_EXPANSION_TEMPLATE = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a Saudi Arabia market research specialist. Your task is to generate "
            "precise Tavily web search queries that will fill knowledge gaps for a company "
            "entering the Saudi market. Each query must be under 250 characters and be "
            "optimised for retrieving live market data from English-language sources."
        )),
        ("human", (
            "Company: {company_name}\n"
            "Products: {products}\n"
            "Budget: {budget}\n\n"
            "Generate exactly 5 search queries that together cover:\n"
            "1. Saudi Arabia market size and growth for this product category\n"
            "2. Named direct competitors operating in KSA right now\n"
            "3. Vision 2030 regulatory or tender opportunities relevant to this product\n"
            "4. Saudi B2B buyer behaviour and decision-maker channels\n"
            "5. Pricing benchmarks and localization requirements in KSA\n\n"
            "Return ONLY a JSON array of 5 strings. No explanation, no markdown."
        )),
    ])

    _LANGCHAIN_AVAILABLE = True
    logger.info("[ResearchAgent] LangChain prompt templates loaded.")

except ImportError:
    _LANGCHAIN_AVAILABLE = False
    ChatOpenAI = None
    logger.info("[ResearchAgent] LangChain not installed — using inline prompts.")


# ---------------------------------------------------------------------------
# Pydantic schemas 
# ---------------------------------------------------------------------------

class CompanyProfile(BaseModel):
    company_name: str = Field(..., description="Full legal or trading name.")
    products_and_prices: str = Field(..., description="Products/services and pricing.")
    brand_voice: str = Field(..., description="Tone and communication style.")
    past_successful_marketing: str = Field(
        default="No past marketing plans provided.",
        description="Optional summary of past successful campaigns.",
    )
    marketing_budget: str = Field(
        default="Not specified",
        description="Marketing budget / cost scope (e.g. SAR 50,000/month).",
    )


class _ExpandedQueries(BaseModel):
    """Internal schema for LLM-generated adaptive query expansion."""
    queries: list[str] = Field(..., description="Exactly 5 search queries.")


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class ResearchAgent:
    """
    Agent 1 — Saudi Arabia market research specialist.
    Uses ChatOpenAI (langchain_openai) for all LLM calls so every completion
    is automatically traced in LangSmith with token counts and latency.
    Embeddings still use openai_client directly (ChatOpenAI does not embed).
    """

    EMBEDDING_MODEL = "text-embedding-3-small"
    CHAT_MODEL      = "gpt-4o-mini"

    _V2030_CONTEXT = (
        "VISION 2030 VOCABULARY — use these terms natively throughout your output:\n"
        "Economic pillars: National Transformation Program (NTP), NIDLP, VRPs, NEOM, "
        "Red Sea Project, Qiddiya, Diriyah Gate, ROSHN, PIF portfolio companies.\n"
        "Regulatory: ZATCA, CITC, SFDA, GOSI, MOMRA, MISA, Nitaqat Saudisation.\n"
        "Culture: Wasta, Majlis, Shura, halal certification, gender-inclusive workplaces "
        "(post-2019), Ramadan marketing windows, VAT at 15%%.\n"
        "Digital: Snapchat (highest KSA penetration), WhatsApp Business API, X (Arabic "
        "thought leadership), TikTok (Gen-Z), LinkedIn (B2B enterprise), Google (intent)."
    )

    def __init__(
        self,
        openai_client: OpenAI,
        tavily_client: TavilyClient,
        llm: "ChatOpenAI | None" = None,
    ) -> None:
        self.openai_client = openai_client          # kept for embeddings only
        self.tavily_client = tavily_client
        self.collection    = get_market_research_collection()
        # Use injected ChatOpenAI if provided; fall back to a local instance
        if llm is not None:
            self.llm = llm
        elif _LANGCHAIN_AVAILABLE:
            self.llm = ChatOpenAI(model=self.CHAT_MODEL, temperature=0.35)
        else:
            self.llm = None

    # ------------------------------------------------------------------
    # Internal helper: call LLM via ChatOpenAI or fallback to raw client
    # ------------------------------------------------------------------

    def _llm_invoke(self, messages: list[dict], temperature: float = 0.35,
                    max_tokens: int = 2200) -> str:
        """
        Invoke the LLM. Uses self.llm (ChatOpenAI) when available so the call
        appears in LangSmith with full token + cost metadata.
        Falls back to self.openai_client when LangChain is not installed.
        """
        if self.llm is not None:
            from langchain_core.messages import SystemMessage, HumanMessage
            lc_msgs = []
            for m in messages:
                if m["role"] == "system":
                    lc_msgs.append(SystemMessage(content=m["content"]))
                else:
                    lc_msgs.append(HumanMessage(content=m["content"]))
            # Override temperature for this call
            response = self.llm.invoke(lc_msgs)
            return response.content.strip()
        # Fallback
        response = self.openai_client.chat.completions.create(
            model=self.CHAT_MODEL,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()

    # ------------------------------------------------------------------
    # Profile parsing
    # ------------------------------------------------------------------

    def parse_profile_from_text(self, extracted_text: str) -> CompanyProfile:
        """Extract a CompanyProfile from raw PDF text using structured output."""
        try:
            completion = self.openai_client.beta.chat.completions.parse(
                model=self.CHAT_MODEL,
                messages=[
                    {"role": "system", "content": "You are a business analyst. Extract a structured company profile."},
                    {"role": "user", "content": (
                        f"Extract from the text below:\n\n{extracted_text}\n\n"
                        "Fields: company_name, products_and_prices, brand_voice, "
                        "past_successful_marketing, marketing_budget. "
                        "Use 'Not specified' for missing fields."
                    )},
                ],
                response_format=CompanyProfile,
                max_tokens=800,
            )
            return completion.choices[0].message.parsed
        except (RateLimitError, APIConnectionError, APIStatusError) as exc:
            logger.error("[ResearchAgent] parse_profile_from_text API error: %s", exc)
            raise

    # ------------------------------------------------------------------
    # Adaptive query expansion
    # ------------------------------------------------------------------

    def build_search_queries(self, profile: CompanyProfile) -> list[str]:
        """
        Adaptive Querying: ask the LLM to generate 5 targeted search queries.
        Uses self.llm (ChatOpenAI) for automatic LangSmith tracing.
        Falls back to static list on any failure.
        """
        company  = sanitize_for_query(profile.company_name, max_chars=55)
        products = sanitize_for_query(profile.products_and_prices, max_chars=70)

        try:
            if _LANGCHAIN_AVAILABLE and self.llm is not None:
                messages = _QUERY_EXPANSION_TEMPLATE.format_messages(
                    company_name=company,
                    products=products,
                    budget=profile.marketing_budget,
                )
                response = self.llm.invoke(messages)
                raw = response.content.strip()
            else:
                oai_msgs = [
                    {"role": "system", "content": (
                        "You are a Saudi Arabia market research specialist. Generate precise "
                        "Tavily web search queries to fill knowledge gaps for a company "
                        "entering the Saudi market. Each query must be under 250 characters."
                    )},
                    {"role": "user", "content": (
                        f"Company: {company}\nProducts: {products}\nBudget: {profile.marketing_budget}\n\n"
                        "Generate exactly 5 search queries covering:\n"
                        "1. Saudi Arabia market size and growth for this product category\n"
                        "2. Named direct competitors in KSA right now\n"
                        "3. Vision 2030 regulatory/tender opportunities for this product\n"
                        "4. Saudi B2B buyer behaviour and decision-maker channels\n"
                        "5. Pricing benchmarks and localisation requirements in KSA\n\n"
                        "Return ONLY a JSON array of 5 strings. No explanation, no markdown."
                    )},
                ]
                response = self.openai_client.chat.completions.create(
                    model=self.CHAT_MODEL, messages=oai_msgs,
                    temperature=0.3, max_tokens=400,
                )
                raw = response.choices[0].message.content.strip()

            raw = raw.strip("`").strip()
            if raw.startswith("json"):
                raw = raw[4:].strip()

            queries: list[str] = json.loads(raw)
            if isinstance(queries, list) and len(queries) >= 3:
                valid = [str(q)[:250] for q in queries if str(q).strip()]
                logger.info("[ResearchAgent] Adaptive queries generated: %d", len(valid))
                return valid[:7]

        except Exception as exc:
            logger.warning("[ResearchAgent] Adaptive query expansion failed (%s) — using static fallback.", exc)

        # Static fallback
        static = [
            f"{company} market opportunity Saudi Arabia Vision 2030",
            f"{products} B2B demand Saudi Arabia KSA market 2024",
            f"{company} competitors Saudi Arabia Middle East landscape",
            f"{products} competitors pricing Saudi Arabia UAE 2024",
            f"digital marketing trends Saudi Arabia B2B enterprise 2024",
            f"GCC consumer behaviour technology adoption Saudi Arabia 2024",
            f"Saudi Arabia {products} industry growth market size 2024",
        ]
        return [q[:250] for q in static]

    # ------------------------------------------------------------------
    # Tavily search 
    # ------------------------------------------------------------------

    def run_tavily_search(self, queries: list[str]) -> str:
        snippets: list[str] = []
        for q in queries:
            try:
                res = self.tavily_client.search(q, search_depth="advanced", max_results=5)
                for r in res.get("results", []):
                    s = r.get("content", "")
                    if s:
                        snippets.append(s[:500])
            except Exception as exc:
                snippets.append(f"[Search note: {exc}]")
        return "\n\n---\n\n".join(snippets) if snippets else (
            "No live results. Generating from parametric knowledge.")

    # ------------------------------------------------------------------
    # Report synthesis — now uses self.llm via _llm_invoke()
    # ------------------------------------------------------------------

    def synthesise_report(self, profile: CompanyProfile, context: str) -> str:
        """Synthesise Tavily context into a structured KSA market research report."""
        user_prompt = (
            f"Company: {profile.company_name}\n"
            f"Products & Pricing: {profile.products_and_prices}\n"
            f"Brand Voice: {profile.brand_voice}\n"
            f"Marketing Budget: {profile.marketing_budget}\n\n"
            "Produce a comprehensive Saudi Arabia-First Market Research Report:\n\n"
            "## 1. Saudi Arabia Market Landscape\n"
            "   - Market size, growth rate, Vision 2030 alignment, key macro drivers.\n"
            "   - Regulatory environment (SFDA, ZATCA, CITC, Nitaqat) as relevant.\n\n"
            "## 2. Competitive Analysis — Saudi Arabia & Middle East\n"
            "   - Name and profile 3-5 direct competitors operating in KSA/ME.\n"
            "   - Their positioning, pricing, strengths, weaknesses, and market share.\n"
            "   - Clear white-space opportunities the client can capture.\n\n"
            "## 3. Target Audience Segments in KSA\n"
            "   - 2-3 segments: sector, company size, decision-maker roles.\n"
            "   - Cultural and psychographic traits specific to Saudi buyers.\n\n"
            "## 4. Saudi Consumer & Business Behaviour\n"
            "   - Communication preferences, Wasta dynamics, Majlis decision-making,\n"
            "     purchasing process, and Ramadan/seasonal patterns.\n\n"
            "## 5. Digital Channel Intelligence — KSA\n"
            "   - Platform usage (Snapchat, Instagram, LinkedIn, X/Twitter, TikTok,\n"
            "     WhatsApp Business, Google) among KSA B2B audiences.\n"
            "   - Content format preferences and peak engagement windows.\n\n"
            "## 6. Key Opportunities & Risks\n"
            "   - Top 3 opportunities with Vision 2030 alignment notes.\n"
            "   - Top 3 risks (regulatory, cultural, competitive) with impact ratings.\n\n"
            "## 7. Strategic Implications\n"
            "   - Prioritised entry/expansion recommendations for the Saudi market.\n\n"
            f"Retrieved Intelligence:\n{context}"
        )

        system_prompt = (
            "You are a Principal Market Intelligence Analyst specialising in Saudi Arabia "
            "and the GCC under Vision 2030. You produce rigorous, evidence-based reports "
            "used by C-suites for high-stakes market entry and expansion decisions. "
            "Your analysis always compares the client against named competitors in KSA/ME.\n\n"
            + self._V2030_CONTEXT
        )

        if _LANGCHAIN_AVAILABLE and self.llm is not None:
            try:
                msgs = _RESEARCH_SYSTEM_TEMPLATE.format_messages(user_prompt=user_prompt)
                response = self.llm.invoke(msgs)
                return response.content.strip()
            except Exception:
                pass

        # Fallback to raw client
        response = self.openai_client.chat.completions.create(
            model=self.CHAT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.35,
            max_tokens=2200,
        )
        return response.choices[0].message.content.strip()

    # ------------------------------------------------------------------
    # Embedding & persistence — openai_client kept for embeddings
    # ------------------------------------------------------------------

    def embed_text(self, text: str) -> list[float]:
        return self.openai_client.embeddings.create(
            model=self.EMBEDDING_MODEL, input=text
        ).data[0].embedding

    def persist_research(self, profile: CompanyProfile, report: str) -> str:
        doc_id    = f"research_{uuid.uuid4().hex}"
        embedding = self.embed_text(report)
        self.collection.upsert(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[report],
            metadatas=[{
                "company_name":        profile.company_name,
                "brand_voice":         profile.brand_voice,
                "products_and_prices": profile.products_and_prices,
                "marketing_budget":    profile.marketing_budget,
                "agent":               "research",
            }],
        )
        return doc_id