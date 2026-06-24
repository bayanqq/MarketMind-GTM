"""
agents/brand_alignment_agent.py
--------------------------------
Agent 4 — Brand Alignment Agent | MarketMind AI
Architecture: v2 — Token Control + Robust Error Handling + CoT-Aware Audit
 
Enhancements in this version
─────────────────────────────
• Dynamic Token Budget: max_tokens is computed from the ContentPackage
  serialised length so we never request more tokens than the task requires,
  while guaranteeing a minimum of 2000 to avoid truncation. Capped at 3800.
 
• Robust Error Handling: The `openai_client.beta.chat.completions.parse`
  call is wrapped in a comprehensive try-except that catches:
    - RateLimitError      → logged, re-raised for caller retry logic
    - APIConnectionError  → logged, re-raised
    - APIStatusError      → logged, re-raised (context window overrun etc.)
    - Exception           → logged, re-raised with context
  The pipeline can then present a user-friendly error rather than crashing.
 
• CoT-Aware Audit: The audit prompt explicitly reads the ContentAgent's
  `reasoning` field to understand WHY tones and hooks were chosen before
  making changes, preventing the agent from needlessly reverting intentional
  creative decisions.
"""


from __future__ import annotations

import logging
import uuid

from openai import OpenAI, RateLimitError, APIConnectionError, APIStatusError
from pydantic import BaseModel, Field

from agents.research_agent import CompanyProfile
from agents.content_agent import ContentPackage, SocialPost, AdCopySet, MarketingEmail, VideoScript
from database_setup import get_marketing_strategies_collection, get_brand_alignment_collection

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LangChain prompt template (graceful fallback when not installed)
# ---------------------------------------------------------------------------
try:
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI

    _ALIGNMENT_TEMPLATE = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a Senior Brand Strategist and Compliance Officer — the absolute final "
            "quality gate before any content is published for Saudi Arabia.\n\n"
            "You audit content for:\n"
            "  1. Brand consistency — language, tone, terminology match the brand voice exactly.\n"
            "  2. Cultural appropriateness for Saudi Arabia / GCC professional audiences.\n"
            "  3. Full compliance with professional standards.\n\n"
            "MANDATORY PROHIBITIONS — enforce for ALL content:\n"
            "- Offensive, discriminatory, or derogatory content.\n"
            "- Political commentary or geopolitical references.\n"
            "- Religious endorsements, critiques, or insensitive references.\n"
            "- Culturally inappropriate content for Saudi Arabia / GCC.\n"
            "- Unverified statistical claims or misleading data.\n"
            "- Sexually suggestive, violent, or unprofessional content.\n\n"
            "IMPORTANT: Read the agent's reasoning (Chain of Thought) BEFORE making changes. "
            "Only override intentional creative decisions if they violate compliance rules or "
            "are clearly off-brand. Do not revert choices that are culturally sound."
        )),
        ("human", "{user_prompt}"),
    ])

    _LANGCHAIN_AVAILABLE = True
    logger.info("[BrandAlignmentAgent] LangChain prompt template loaded.")
except ImportError:
    _LANGCHAIN_AVAILABLE = False
    ChatOpenAI = None
    logger.info("[BrandAlignmentAgent] LangChain not installed — using inline prompts.")


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

class RefinedContent(BaseModel):
    """Brand-verified and refined version of the full content package."""
    alignment_summary: str           = Field(..., description="Overall audit: what was changed and why.")
    brand_voice_guidelines: str      = Field(..., description="Concrete do/don't rules for future campaigns.")
    content_guardrails: list[str]    = Field(..., description="Compliance rules applied in this audit.")
    refined_posts: list[SocialPost]  = Field(..., description="Brand-refined social posts.")
    refined_email: MarketingEmail    = Field(..., description="Brand-refined email.")
    refined_ads: list[AdCopySet]     = Field(..., description="Brand-refined ad copy sets.")
    refined_video_script: VideoScript = Field(..., description="Brand-refined video script.")
    aligned_value_proposition: str   = Field(..., description="Value prop in exact brand voice.")


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class BrandAlignmentAgent:
    """
    Backstory:
        Senior Brand Strategist and Compliance Officer, 10 years in GCC B2B.
        Absolute final quality gate. Reads the ContentAgent's CoT reasoning
        before auditing — preserving intentional creative decisions that are
        culturally sound while correcting genuine off-brand elements.

    Goal:
        Audit ALL generated content against brand voice and KSA compliance.
        Refine non-compliant elements. Return a brand-verified content package
        with explicit guardrails documented.
    """

    CHAT_MODEL      = "gpt-4o-mini"
    EMBEDDING_MODEL = "text-embedding-3-small"

    _SAFETY_PROHIBITIONS = (
        "MANDATORY PROHIBITIONS — enforce for ALL content:\n"
        "- PROHIBITED: Offensive, discriminatory, or derogatory content of any kind.\n"
        "- PROHIBITED: Political commentary or geopolitical references.\n"
        "- PROHIBITED: Religious endorsements, critiques, or insensitive references.\n"
        "- PROHIBITED: Culturally inappropriate content for Saudi Arabia / GCC.\n"
        "- PROHIBITED: Unverified statistical claims or misleading data.\n"
        "- PROHIBITED: Sexually suggestive, violent, or unprofessional content.\n"
        "- REQUIRED: All copy professional, factual, and business-focused."
    )

    def __init__(
        self,
        openai_client: OpenAI,
        llm: "ChatOpenAI | None" = None,
    ) -> None:
        self.openai_client        = openai_client    # embeddings only
        self.content_collection   = get_marketing_strategies_collection()
        self.alignment_collection = get_brand_alignment_collection()
        if llm is not None:
            self.llm = llm
        elif _LANGCHAIN_AVAILABLE:
            self.llm = ChatOpenAI(model=self.CHAT_MODEL, temperature=0.3)
        else:
            self.llm = None

    # ------------------------------------------------------------------
    # Content serialisation
    # ------------------------------------------------------------------

    def _serialize_package(self, pkg: ContentPackage) -> str:
        """Convert ContentPackage to a readable string including CoT reasoning."""
        parts = []

        # ── Include CoT reasoning so the agent can read it ────────────────
        cot = getattr(pkg, "reasoning", "")
        if cot:
            parts += [
                "=== CONTENT AGENT REASONING (Chain of Thought) ===",
                cot,
                "(Review this before making changes — only override if compliance/brand issue.)",
                "",
            ]

        parts.append("=== SOCIAL POSTS ===")
        for i, p in enumerate(pkg.linkedin_x_posts, 1):
            parts += [
                f"\nPost {i} [{p.platform} / {p.post_type}]",
                f"Hook: {p.hook}",
                f"Body: {p.body}",
                f"CTA: {p.call_to_action}",
                f"Hashtags: {' '.join(p.hashtags)}",
            ]

        e = pkg.marketing_email
        parts += [
            "\n=== EMAIL ===",
            f"Subject: {e.subject_line}",
            f"Preview: {e.preview_text}",
            f"Greeting: {e.greeting}",
            *e.body_paragraphs,
            f"CTA: {e.primary_cta}",
            f"Sign-off: {e.sign_off}",
        ]

        parts.append("\n=== AD COPY ===")
        for i, a in enumerate(pkg.ad_copy_sets, 1):
            parts += [
                f"\nAd {i} [{a.placement}]",
                f"Headline: {a.headline}",
                f"Body: {a.body_copy}",
                f"CTA: {a.cta_button}",
            ]

        vs = pkg.video_script
        parts += [
            "\n=== VIDEO SCRIPT ===",
            f"Title: {vs.title}",
            f"Platform: {vs.target_platform}",
            f"Objective: {vs.objective}",
            f"Duration: {vs.total_duration_seconds}s",
        ]
        for sc in vs.scenes:
            parts += [
                f"\nScene {sc.scene_number} ({sc.duration_seconds}s)",
                f"Visual: {sc.visual_cue}",
                f"Dialogue: {sc.spoken_dialogue}",
            ]
        parts.append(f"Closing CTA: {vs.closing_cta}")

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Dynamic token budget
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_max_tokens(content_str: str) -> int:
        """
        Scale max_tokens based on the ContentPackage size.
        Minimum 2000 (to avoid truncating RefinedContent schema).
        Maximum 3800 (safety cap for gpt-4o-mini context).
        """
        # Rough estimate: output is ~60% the length of input
        estimated_output_chars = int(len(content_str) * 0.60)
        # ~4 chars per token
        estimated_tokens = estimated_output_chars // 4
        return max(2000, min(3800, estimated_tokens))

    # ------------------------------------------------------------------
    # Brand alignment audit with robust error handling
    # ------------------------------------------------------------------

    def run_alignment(self, profile: CompanyProfile, pkg: ContentPackage) -> RefinedContent:
        """
        Audit and refine the ContentPackage.

        Raises:
            RateLimitError: when the API rate limit is hit (caller should retry).
            APIConnectionError: when the network connection fails.
            APIStatusError: when the API returns a 4xx/5xx (incl. context overflow).
            RuntimeError: for unexpected errors with context.
        """
        content_str = self._serialize_package(pkg)
        max_tokens  = self._compute_max_tokens(content_str)

        logger.info(
            "[BrandAlignmentAgent] Serialised content: %d chars. Max tokens: %d.",
            len(content_str), max_tokens,
        )

        user_prompt = (
            f"Company: {profile.company_name}\n"
            f"Brand Voice: {profile.brand_voice}\n"
            f"Products: {profile.products_and_prices}\n"
            f"Past Marketing: {profile.past_successful_marketing}\n\n"
            "AUDIT AND REFINE the following content package.\n"
            "IMPORTANT: Read the 'Content Agent Reasoning' section first. "
            "Only override creative decisions if they violate compliance rules or are off-brand.\n\n"
            "For each piece of content:\n"
            "  - Identify off-brand language, tone mismatches, or compliance issues.\n"
            "  - Rewrite to fix issues while preserving commercial impact.\n"
            "  - Document all changes in alignment_summary.\n\n"
            "Also produce:\n"
            "  - brand_voice_guidelines: concrete do/don't rules.\n"
            "  - content_guardrails: full list of rules applied.\n"
            "  - aligned_value_proposition: value prop in exact brand voice.\n\n"
            f"Content Package to Audit:\n{content_str}"
        )

        if _LANGCHAIN_AVAILABLE:
            try:
                msgs = _ALIGNMENT_TEMPLATE.format_messages(user_prompt=user_prompt)
                oai_msgs = [{"role": "system", "content": msgs[0].content},
                            {"role": "user",   "content": msgs[1].content}]
            except Exception:
                oai_msgs = self._build_fallback_messages(profile, user_prompt)
        else:
            oai_msgs = self._build_fallback_messages(profile, user_prompt)

        # ── Robust API call with comprehensive error handling ──────────────
        try:
            if self.llm is not None:
                structured_llm = self.llm.with_structured_output(RefinedContent)
                result: RefinedContent = structured_llm.invoke(oai_msgs)
            else:
                completion = self.openai_client.beta.chat.completions.parse(
                    model=self.CHAT_MODEL, messages=oai_msgs,
                    response_format=RefinedContent, max_tokens=max_tokens,
                )
                result: RefinedContent = completion.choices[0].message.parsed
            logger.info("[BrandAlignmentAgent] Brand alignment audit complete.")
            return result

        except RateLimitError as exc:
            logger.error(
                "[BrandAlignmentAgent] Rate limit hit. Retry after backoff. Error: %s", exc
            )
            raise

        except APIConnectionError as exc:
            logger.error(
                "[BrandAlignmentAgent] API connection error. Check network/proxy. Error: %s", exc
            )
            raise

        except APIStatusError as exc:
            # Covers 400 (context overflow), 500, 529, etc.
            logger.error(
                "[BrandAlignmentAgent] API status error (status=%s). "
                "May be context window overrun — content was %d chars. Error: %s",
                exc.status_code, len(content_str), exc,
            )
            raise

        except Exception as exc:
            logger.error(
                "[BrandAlignmentAgent] Unexpected error during alignment: %s", exc
            )
            raise RuntimeError(
                f"BrandAlignmentAgent encountered an unexpected error: {exc}"
            ) from exc

    def _build_fallback_messages(self, profile: CompanyProfile, user_prompt: str) -> list[dict]:
        """Inline system + user messages when LangChain is not available."""
        system_content = (
            "You are a Senior Brand Strategist and Compliance Officer — the absolute final "
            "quality gate before any content is published for Saudi Arabia. "
            "Read the agent's Chain-of-Thought reasoning before making changes.\n\n"
            + self._SAFETY_PROHIBITIONS
        )
        return [
            {"role": "system", "content": system_content},
            {"role": "user",   "content": user_prompt},
        ]

    # ------------------------------------------------------------------
    # Embedding & persistence
    # ------------------------------------------------------------------

    def embed_text(self, text: str) -> list[float]:
        return self.openai_client.embeddings.create(
            model=self.EMBEDDING_MODEL, input=text
        ).data[0].embedding

    def persist_alignment(
        self,
        profile: CompanyProfile,
        refined: RefinedContent,
        content_doc_id: str,
    ) -> str:
        doc_id         = f"brand_{uuid.uuid4().hex}"
        summary        = f"Brand alignment for {profile.company_name}. {refined.alignment_summary[:300]}"
        guardrails_str = " | ".join(refined.content_guardrails)[:800]

        self.alignment_collection.upsert(
            ids=[doc_id],
            embeddings=[self.embed_text(summary)],
            documents=[summary],
            metadatas=[{
                "company_name":          profile.company_name,
                "brand_voice":           profile.brand_voice,
                "alignment_summary":     refined.alignment_summary[:400],
                "content_guardrails":    guardrails_str,
                "aligned_value_prop":    refined.aligned_value_proposition[:400],
                "source_content_doc_id": content_doc_id,
                "cot_aware_audit":       "true",
                "agent":                 "brand_alignment",
            }],
        )
        return doc_id

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(
        self,
        profile: CompanyProfile,
        pkg: ContentPackage,
        content_doc_id: str,
    ) -> tuple[RefinedContent, str]:
        refined  = self.run_alignment(profile, pkg)
        brand_id = self.persist_alignment(profile, refined, content_doc_id)
        return refined, brand_id