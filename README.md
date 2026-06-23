
# 🚀 MarketMind AI 🇸🇦

> **A production-grade, multi-agent marketing automation platform built to generate end-to-end Go-To-Market (GTM) strategies and content for the Saudi Arabia and GCC market.**

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?logo=langchain&logoColor=white)
![OpenAI](https://img.shields.io/badge/GPT--4o--mini-412991?logo=openai&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-FF6B6B?logo=database&logoColor=white)

## 📖 Overview
MarketMind AI takes a company profile as input and autonomously produces market research, a localized GTM strategy, a 30-day marketing calendar, and a full suite of platform-ready marketing content (social posts, email drafts, ad copy, video scripts). 

It is designed to provide startups, SMEs, and enterprise marketing teams entering the Saudi market with a structured, culturally aligned, and budget-aware strategy without the need for expensive external consultancies.

---

## ✨ The 4-Agent Pipeline
The system is built on a sequential four-agent architecture:

1. **🔍 Research Agent:** Performs live Saudi Arabia market and competitor analysis using adaptive web search queries calibrated to **Saudi Vision 2030** terminology.
2. **♟️ Strategy Agent:** Generates a mathematically budget-constrained GTM strategy with a professional Ideal Customer Profile (ICP) based on real firmographic data, plus a 30-day marketing calendar.
3. **✍️ Content Agent:** Produces all marketing content with a self-reflection critique loop and Chain-of-Thought (CoT) reasoning.
4. **🛡️ Brand Alignment Agent:** Acts as the final quality gate, auditing every piece of content against brand guidelines and **7 strict Saudi cultural compliance rules**.

---

## 🛠️ Tech Stack & AI Components

| Component | Role in the System |
| :--- | :--- |
| **GPT-4o-mini** | Primary inference model for all four agents (via ChatOpenAI). |
| **LangChain / LangSmith** | Orchestration, Chain-of-Thought routing, and automatic tracing (token, cost, latency). |
| **ChromaDB** | Persistent local vector database (4 collections in pipeline order). |
| **text-embedding-3-small** | Generates 1536-dim vectors for all agent outputs. |
| **Tavily Search API** | Live web search for KSA market intelligence with adaptive querying. |
| **Pydantic V2** | Ensures strict schema compliance for output generation (`with_structured_output`). |
| **Streamlit** | Unified, interactive frontend dashboard with PDF export capabilities. |

---

## ⚙️ How It Works
The user enters a company profile via the Streamlit interface (company name, products, pricing, brand voice, past marketing history, and SAR budget). After a Human-in-the-Loop approval gate, the pipeline executes:

1. **Research Stage:** Fetches live KSA market data across 7 adaptive queries and synthesizes a 7-section report.
2. **Strategy Stage:** Computes exact SAR allocations mathematically. LLM generates the strategy using these figures verbatim.
3. **Content Stage:** Extracts strategy anchors (~65% token reduction). Generates a first-draft `ContentPackage`, self-critiques, and refines it.
4. **Brand Alignment Stage:** Reads the Content Agent's CoT reasoning and audits the content to preserve creative intent while enforcing KSA cultural compliance.
5. **Output:** All outputs are embedded, stored in ChromaDB, and displayed in the dashboard with per-agent PDF download buttons.

---

## 🚀 Key Achievements & Features
- **Budget Constraint Engine:** Python pre-computes exact SAR allocations to completely eliminate LLM financial hallucinations.
- **Self-Reflection Loop:** A critique-then-refine pattern that measurably improves content hook strength and CTA clarity.
- **Vision 2030 Vocabulary Injection:** System prompts are engineered with KSA-specific data (e.g., Snapchat penetration) for highly regionalized output.
- **Three-Layer Safety Architecture:** Input injection guard + generation-time prohibitions + final Brand Alignment audit.
- **Pipeline Persistence:** Incremental refinement without re-running upstream agents, thanks to ChromaDB.

---

## 🗺️ Roadmap & Future Improvements
- [ ] **Arabic Language Support:** Integrate Arabic-capable fonts (Amiri/Cairo) for PDF exports and add bilingual content generation modes.
- [ ] **LangGraph Orchestration:** Upgrade from a linear pipeline to a state machine for conditional, dynamic re-runs.
- [ ] **Competitor Monitoring:** Scheduled weekly Tavily searches to flag market shifts automatically.
- [ ] **Multi-Tenant Campaigns:** Extend pipeline to manage multiple companies with isolated ChromaDB namespaces.
- [ ] **CRM Integrations:** One-click export of marketing calendars to HubSpot or Google Sheets.

---

## 👥 Meet the Team
This project was developed by:
- **Areej Alharthi**
- **Bayan Alqarni**
- **Ghada Aljuhani**
- **Rawan Alghamdi**

*Developed as a graduation course project in Data Science and Artificial Intelligence.*
