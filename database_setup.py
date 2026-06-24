"""
database_setup.py
-----------------
ChromaDB persistent client initialization and collection management for MarketMind AI.
Four collections map to the four-agent pipeline:
  market_research       → Agent 1 (Research)
  icp_and_strategy      → Agent 2 (Strategy)
  brand_alignment       → Agent 3 (Brand Alignment)
  marketing_strategies  → Agent 4 (Content) — legacy name kept for compatibility
"""

import os
import chromadb
from chromadb.config import Settings


# ---------------------------------------------------------------------------
# Collection name constants
# ---------------------------------------------------------------------------

COLLECTION_MARKET_RESEARCH:      str = "market_research"
COLLECTION_ICP_AND_STRATEGY:     str = "icp_and_strategy"
COLLECTION_BRAND_ALIGNMENT:      str = "brand_alignment"
COLLECTION_MARKETING_STRATEGIES: str = "marketing_strategies"

_CHROMA_CLIENT: chromadb.ClientAPI | None = None


# ---------------------------------------------------------------------------
# Singleton client
# ---------------------------------------------------------------------------

def get_chroma_client() -> chromadb.ClientAPI:
    global _CHROMA_CLIENT
    if _CHROMA_CLIENT is None:
        # Resolve to an absolute path so ChromaDB always finds the same folder
        # regardless of the working directory Streamlit uses on Windows.
        _raw_path: str = os.getenv(
            "CHROMA_PERSIST_PATH",
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db"),
        )
        persist_path: str = os.path.abspath(_raw_path)
        os.makedirs(persist_path, exist_ok=True)
        _CHROMA_CLIENT = chromadb.PersistentClient(
            path=persist_path,
            settings=Settings(anonymized_telemetry=False),
        )
    return _CHROMA_CLIENT


def get_or_create_collection(
    collection_name: str,
    *,
    metadata: dict | None = None,
) -> chromadb.Collection:
    client = get_chroma_client()
    return client.get_or_create_collection(
        name=collection_name,
        metadata=metadata or {"hnsw:space": "cosine"},
    )


# ---------------------------------------------------------------------------
# Per-agent collection accessors
# ---------------------------------------------------------------------------

def get_market_research_collection() -> chromadb.Collection:
    """Agent 1 — raw market, competitor, and trend research."""
    return get_or_create_collection(
        COLLECTION_MARKET_RESEARCH,
        metadata={"hnsw:space": "cosine", "description": "Market research documents"},
    )


def get_icp_and_strategy_collection() -> chromadb.Collection:
    """Agent 2 — ICP definition, positioning, messaging, and channel strategy."""
    return get_or_create_collection(
        COLLECTION_ICP_AND_STRATEGY,
        metadata={"hnsw:space": "cosine", "description": "ICP and GTM strategy documents"},
    )


def get_brand_alignment_collection() -> chromadb.Collection:
    """Agent 3 — brand-aligned, compliance-checked strategy documents."""
    return get_or_create_collection(
        COLLECTION_BRAND_ALIGNMENT,
        metadata={"hnsw:space": "cosine", "description": "Brand-aligned strategy documents"},
    )


def get_marketing_strategies_collection() -> chromadb.Collection:
    """Agent 4 — final content packages ready for publishing."""
    return get_or_create_collection(
        COLLECTION_MARKETING_STRATEGIES,
        metadata={"hnsw:space": "cosine", "description": "Final marketing content packages"},
    )


def reset_collections() -> None:
    """Delete and recreate all collections. Use only during development."""
    client = get_chroma_client()
    for name in (
        COLLECTION_MARKET_RESEARCH,
        COLLECTION_ICP_AND_STRATEGY,
        COLLECTION_BRAND_ALIGNMENT,
        COLLECTION_MARKETING_STRATEGIES,
    ):
        try:
            client.delete_collection(name)
        except Exception:
            pass
    get_market_research_collection()
    get_icp_and_strategy_collection()
    get_brand_alignment_collection()
    get_marketing_strategies_collection()
    print("[DB] All four collections reset and recreated.")