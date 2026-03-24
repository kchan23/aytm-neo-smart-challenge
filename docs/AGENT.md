# AGENT.md — Architecture & Continuation Guide

This file exists to give any AI agent (or human developer) full context on what this project is, what has been built, and how to continue safely after context loss or agent switching.

---

## Project Overview

**Neo Smart Living × Aytm Joint Challenge — CPP AI Hackathon 2026**

A Streamlit multi-page app demonstrating a STAMP-validated synthetic market research pipeline for the Tahoe Mini prefab backyard unit ($23K, 117 sq ft). The pipeline replaces $6K–$48K field research with ~$2–$4 in LLM API calls via OpenRouter.

All API calls go through **OpenRouter** (`https://openrouter.ai/api/v1/chat/completions`). API key is loaded from `OPENROUTER_API_KEY` env var or a `.env` file.

---

## Repo Structure (`app/` is the git repo root)

```
app/
├── Home.py                        # Landing page — navigation hub, key stats, pipeline narrative
├── requirements.txt               # Streamlit Cloud dependencies
├── .env.example                   # API key template (copy to .env, never commit .env)
│
├── pages/
│   ├── 1_Interviews.py            # Stage 2: Qualitative interview insights (6-tab dashboard)
│   ├── 2_Survey.py                # Stage 4: Quantitative survey analytics (7-tab dashboard)
│   └── 3_Building_Codes.py        # Stage 3 extension: RAG chatbot + evaluation mode
│
├── utils/
│   ├── __init__.py
│   ├── analytics.py               # Shared stats module (descriptive, Mann-Whitney U)
│   ├── llm_caller.py              # OpenRouter API wrapper (dual-LLM parallel calls)
│   ├── rag_pipeline.py            # Document loading, chunking, TF-IDF retrieval
│   ├── judge.py                   # Judge LLM: compare answers, synthesize, classify disagreement
│   └── rag_evaluator.py           # Batch evaluation against ground-truth Q&A
│
├── data/
│   ├── building_codes/            # Drop PDFs/TXTs here — gitignored for large files
│   │   └── .gitkeep
│   └── eval/
│       └── sample_qa.json         # Seed Q&A pairs for evaluation mode (committed)
│
└── docs/
    ├── AGENT.md                   # This file
    └── DECISIONS.md               # Log of key architectural decisions
```

The `prototype/` directory (sibling to `app/`) contains the professor's reference implementation. It is **not part of the git repo** — it's for local reference only. Pages 1 and 2 read pre-generated CSVs from `../prototype/output/`.

---

## Three-Stage Pipeline

```
Stage 1 (not yet built): Client Discovery
  → LLM role-plays as Tony Koo (Neo Smart Living CEO) for practice discovery interviews

Stage 2 (BUILT): Qualitative Interviews  [pages/1_Interviews.py]
  → 30 synthetic depth interviews via GPT-4.1-mini + Gemini 2.5 Flash
  → 3-layer analysis: VADER sentiment, LDA topics, emotion classification
  → Data: ../prototype/output/interview_transcripts.csv + interview_analysis.csv

Stage 3 (not yet built as standalone): Qual → Quant Handoff
  → Interview themes feed segment definitions → survey design

Stage 4 (BUILT): Quantitative Survey    [pages/2_Survey.py]
  → 60 synthetic survey responses (5 segments × 2 LLMs × 6 per)
  → Mann-Whitney U cross-model validation
  → Data: ../prototype/output/synthetic_responses.csv

Stage 5/6 (BUILT): Analysis + Dashboard
  → Integrated into pages 1 and 2

EXTENSION (BUILT): Building Codes RAG   [pages/3_Building_Codes.py]
  → User asks building code questions
  → TF-IDF retrieval over loaded PDFs/text files
  → GPT-4.1-mini + Gemini 2.5 Flash answer independently (parallel)
  → Claude Sonnet judges: compare, synthesize, classify disagreement severity
  → Evaluation mode: batch ROUGE-1 scoring vs. sample_qa.json
```

---

## LLM Model Assignments

| Role | Model ID (OpenRouter) | Label |
|---|---|---|
| Dual LLM A | `openai/gpt-4.1-mini` | GPT-4.1-mini |
| Dual LLM B | `google/gemini-2.5-flash` | Gemini-2.5-Flash |
| Judge | `anthropic/claude-3.5-sonnet` | Claude Sonnet |

These are defined as constants in `utils/llm_caller.py` and `utils/judge.py`. To swap models, edit only those files.

---

## Key Patterns

### sys.path for page imports
Every page file adds `app/` to sys.path so `utils.*` is importable:
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.analytics import load_data
```

### API key loading
```python
from utils.llm_caller import load_api_key
api_key = load_api_key(Path(__file__).parent.parent)  # looks for app/.env
```

### Caching
- `@st.cache_data` — for CSV/JSON loading (reloads when file changes)
- `@st.cache_resource` — for the RAG retriever index (heavy, built once per session)

### Data paths (all relative to app/)
```python
BASE = Path(__file__).parent.parent          # app/
PROTO_OUTPUT = BASE.parent / "prototype" / "output"   # ../prototype/output/
DOCS_DIR = BASE / "data" / "building_codes"
EVAL_DIR = BASE / "data" / "eval"
```

---

## What Needs to Be Built Next

- [ ] **Stage 1 — Client Discovery page**: `pages/0_Client_Discovery.py`
  - LLM role-plays Tony Koo using context from `Customized_GPT_for_Learning_Business_Case.pdf`
  - Multi-turn chat interface
  - Compare simulated vs. real answers framework

- [ ] **Add actual building code documents** to `app/data/building_codes/`
  - California R-3/U occupancy codes for structures ≤120 sq ft
  - Local jurisdiction codes (Los Angeles, San Bernardino County)
  - ADU/permit exemption guidelines

- [ ] **Responsible AI note** (required deliverable): half-page doc distinguishing synthetic findings from decision-grade evidence

- [ ] **GenAI documentation** (required deliverable): list of tools, prompts, AI-generated components

---

## Running Locally

```bash
# Install dependencies
pip install -r app/requirements.txt

# Set API key
export OPENROUTER_API_KEY=sk-or-...   # or create app/.env

# Run the app
streamlit run app/Home.py

# Run individual prototype scripts (reference only, not part of repo)
streamlit run prototype/dashboard.py
streamlit run prototype/interview_dashboard.py
```

---

## Judging Criteria (50 pts total)

| Criterion | Pts | Key Signal |
|---|---|---|
| Innovation & Problem Framing | 10 | Go beyond API wrapper; show cross-model validation value |
| Impact & Relevance | 10 | Findings actionable for Tony Koo's real decisions |
| Technical Feasibility | 10 | Pipeline connects, validation works, responsible AI stated |
| User Need & Adoption | 10 | Modular, documented, usable by non-programmers |
| Storytelling | 10 | 5-min video, clear narrative, confidence calibration |
