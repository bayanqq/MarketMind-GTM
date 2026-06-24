"""
main.py
-------
MarketMind AI — Central Pipeline Orchestrator.

Boots all API clients from the local ``.env`` file and executes the four-
agent pipeline sequentially:

    Agent 1 (Research)  →  Agent 2 (Strategy)  →  Agent 3 (Content)
    →  Agent 4 (Brand Alignment)

Each agent's output is fed directly into the next, forming a seamless
end-to-end agentic marketing and market research workflow.
"""

from __future__ import annotations

import sys

from dotenv import load_dotenv
import os

from openai import OpenAI
from tavily import TavilyClient

from agents.research_agent import ResearchAgent
from agents.strategy_agent import StrategyAgent
from agents.content_agent import ContentAgent
from agents.brand_alignment_agent import BrandAlignmentAgent


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

def load_environment() -> tuple[str, str]:
    """Load and validate required environment variables.

    Returns:
        A tuple of ``(openai_api_key, tavily_api_key)``.

    Raises:
        SystemExit: When either required key is missing from the environment.
    """
    load_dotenv()

    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    tavily_key = os.getenv("TAVILY_API_KEY", "").strip()

    missing: list[str] = []
    if not openai_key:
        missing.append("OPENAI_API_KEY")
    if not tavily_key:
        missing.append("TAVILY_API_KEY")

    if missing:
        print(
            f"[MarketMind AI] Fatal: the following environment variables are not "
            f"set in your .env file: {', '.join(missing)}\n"
            "Please add them and restart."
        )
        sys.exit(1)

    return openai_key, tavily_key


# ---------------------------------------------------------------------------
# Pipeline banner
# ---------------------------------------------------------------------------

def print_banner() -> None:
    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║          M A R K E T M I N D   A I                          ║")
    print("║   Agentic Marketing & Market Research Platform               ║")
    print("║   Middle East GTM Intelligence System  v2.0                  ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()


def print_stage(stage_number: int, title: str) -> None:
    print()
    print(f"┌─────────────────────────────────────────────────────────────┐")
    print(f"│  STAGE {stage_number}: {title:<53}│")
    print(f"└─────────────────────────────────────────────────────────────┘")


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def main() -> None:
    """Execute the full MarketMind AI four-agent pipeline."""

    print_banner()

    # ── Environment & client initialisation ────────────────────────────────
    openai_key, tavily_key = load_environment()

    openai_client = OpenAI(api_key=openai_key)
    tavily_client = TavilyClient(api_key=tavily_key)

    print("[MarketMind AI] API clients initialised.")
    print(f"[MarketMind AI] ChromaDB persistence path: {os.getenv('CHROMA_PERSIST_PATH', './chroma_db')}")

    # ── Stage 1: Research Agent ─────────────────────────────────────────────
    print_stage(1, "Research Agent — Profile Ingestion & Market Analysis")

    research_agent = ResearchAgent(
        openai_client=openai_client,
        tavily_client=tavily_client,
    )

    try:
        profile, research_report, research_doc_id = research_agent.run()
    except RuntimeError as exc:
        print(f"\n[MarketMind AI] Pipeline aborted at Stage 1: {exc}")
        sys.exit(0)

    print(f"\n[MarketMind AI] Stage 1 complete. Research doc ID: {research_doc_id}")

    # ── Stage 2: Strategy Agent ─────────────────────────────────────────────
    print_stage(2, "Strategy Agent — Middle East GTM Strategy Generation")

    strategy_agent = StrategyAgent(openai_client=openai_client)

    try:
        gtm_strategy, strategy_doc_id = strategy_agent.run(
            profile=profile,
            research_doc_id=research_doc_id,
        )
    except (ValueError, RuntimeError) as exc:
        print(f"\n[MarketMind AI] Pipeline aborted at Stage 2: {exc}")
        sys.exit(1)

    print(f"\n[MarketMind AI] Stage 2 complete. Strategy doc ID: {strategy_doc_id}")

    # ── Stage 3: Content Agent ──────────────────────────────────────────────
    print_stage(3, "Content Agent — Marketing Content Synthesis")

    content_agent = ContentAgent(openai_client=openai_client)

    try:
        # run() returns (ContentPackage, content_doc_id) — both are needed
        # so Stage 4 can link the brand alignment result to its source document.
        content_package, content_doc_id = content_agent.run(
            profile=profile,
            strategy_doc_id=strategy_doc_id,
        )
    except (ValueError, RuntimeError) as exc:
        print(f"\n[MarketMind AI] Pipeline aborted at Stage 3: {exc}")
        sys.exit(1)

    print(f"\n[MarketMind AI] Stage 3 complete. Content doc ID: {content_doc_id}")

    # ── Stage 4: Brand Alignment Agent ─────────────────────────────────────
    print_stage(4, "Brand Alignment Agent — Compliance & Brand Voice Audit")

    brand_alignment_agent = BrandAlignmentAgent(openai_client=openai_client)

    try:
        # run() expects (profile, content_package, content_doc_id) and returns
        # (RefinedContent, brand_doc_id).  run_alignment() re-raises API errors
        # (RateLimitError, APIConnectionError, APIStatusError, RuntimeError) so
        # we catch the broad Exception here and log it without masking details.
        refined_content, brand_doc_id = brand_alignment_agent.run(
            profile=profile,
            pkg=content_package,
            content_doc_id=content_doc_id,
        )
    except Exception as exc:
        # Stage 4 failure is non-fatal: content was already generated and
        # persisted in Stage 3.  We log the error and continue to the summary
        # so the operator can inspect and re-run alignment independently.
        print(f"\n[MarketMind AI] Warning: Stage 4 encountered an error: {exc}")
        print("[MarketMind AI] Content from Stage 3 is preserved. Brand alignment skipped.")
        brand_doc_id = "N/A (stage 4 failed)"
    else:
        print(f"\n[MarketMind AI] Stage 4 complete. Brand alignment doc ID: {brand_doc_id}")

    # ── Pipeline summary ────────────────────────────────────────────────────
    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  ✓  MARKETMIND AI PIPELINE COMPLETED SUCCESSFULLY            ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    print(f"║  Company     : {profile.company_name:<46}║")
    print(f"║  Research ID : {research_doc_id:<46}║")
    print(f"║  Strategy ID : {strategy_doc_id:<46}║")
    print(f"║  Content  ID : {content_doc_id:<46}║")
    print(f"║  Brand    ID : {brand_doc_id:<46}║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()