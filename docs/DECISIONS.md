# DECISIONS.md — Architectural Decision Log

Each entry explains a key decision, why it was made, and any trade-offs. Update this file whenever a non-obvious choice is made.

---

## [2026-03-23] Use OpenRouter for all API calls

**Decision**: All LLM calls go through OpenRouter (`https://openrouter.ai/api/v1/chat/completions`), not provider SDKs directly.

**Why**: Consistency with the professor's prototype (`prototype/synthetic_respondents.py`). Single API key for all models. Easy to swap model IDs without changing calling code.

**Trade-off**: Adds ~50–200ms latency vs. direct API. Acceptable for a demo.

**Where**: `utils/llm_caller.py` — `OPENROUTER_URL`, `call_openrouter()`

---

## [2026-03-23] Dual LLM: GPT-4.1-mini + Gemini 2.5 Flash

**Decision**: The two "worker" LLMs are `openai/gpt-4.1-mini` and `google/gemini-2.5-flash`.

**Why**: Same models used in the professor's prototype (Stages 2 & 4). Maintains methodological consistency with the STAMP validation framework. Both are cheap (sub-$0.001/1K tokens), fast, and available on OpenRouter.

**Trade-off**: Neither is state-of-the-art. Disagreements between them may reflect model-specific biases rather than genuine ambiguity in source documents. This is acceptable and even useful as a signal.

**Where**: `utils/llm_caller.py` — `MODELS` dict

---

## [2026-03-23] Judge LLM: Claude Sonnet (anthropic/claude-3.5-sonnet)

**Decision**: The judge model is `anthropic/claude-3.5-sonnet` via OpenRouter.

**Why**: User specified "Sonnet or higher" for judgment. Sonnet is more capable at structured reasoning and JSON output than the worker models, while being cheaper than Opus. Using a different model family (Anthropic vs. OpenAI/Google) reduces the risk that the judge simply prefers one worker model due to shared training.

**Trade-off**: Costs ~3–5x more per call than the worker LLMs. Each user question costs ~3 API calls total. Add a "Disable Judge" toggle to control costs.

**Note**: Verify the exact OpenRouter model ID — it may be `anthropic/claude-3.5-sonnet`, `anthropic/claude-3-5-sonnet`, or `anthropic/claude-sonnet-4-5` depending on availability. Check https://openrouter.ai/models if the judge call fails.

**Where**: `utils/judge.py` — `JUDGE_MODEL_ID`

---

## [2026-03-23] TF-IDF retrieval by default (no vector DB)

**Decision**: Default retrieval uses `sklearn.TfidfVectorizer` + cosine similarity. `sentence-transformers` is optional.

**Why**: Zero additional dependencies beyond sklearn (already required for analytics). No GPU needed. Runs on any machine. For building code documents (precise, keyword-rich text), TF-IDF performs well — semantic embeddings matter more for conversational/ambiguous queries.

**Trade-off**: TF-IDF fails on paraphrased queries ("what's the height limit" vs. "maximum elevation of structure"). Mitigated by top_k slider (default k=5).

**Where**: `utils/rag_pipeline.py` — `BuildingCodeRetriever.build_index()`

---

## [2026-03-23] data/ folder inside app/ (git repo root)

**Decision**: `app/data/building_codes/` and `app/data/eval/` are inside the git repo.

**Why**: Streamlit Cloud requires all app assets to be in the repo. `sample_qa.json` needs to be committed. Building code PDFs can be gitignored but the folder structure is committed via `.gitkeep`.

**Trade-off**: Large PDFs must not be committed (add to `.gitignore`). The folder is a placeholder — users must add their own documents.

**Where**: `app/data/building_codes/.gitkeep`, `app/data/eval/sample_qa.json`

---

## [2026-03-23] sys.path insert pattern for page imports

**Decision**: All `pages/*.py` files insert `str(Path(__file__).parent.parent)` (= `app/`) into `sys.path[0]` before importing from `utils/`.

**Why**: Streamlit pages run in a subprocess context where the working directory may not be `app/`. This mirrors the pattern already used in `pages/2_Survey.py`. Avoids packaging `utils/` as a proper installable package.

**Trade-off**: Fragile if file locations change. Document clearly.

**Where**: All `pages/*.py` files, top of file before utils imports.

---

## [2026-03-23] ROUGE-1 F1 as evaluation metric (pure Python)

**Decision**: Evaluation scoring uses token-overlap ROUGE-1 F1 computed inline (no `rouge-score` package).

**Why**: Avoids an extra dependency. For short answers (1–4 sentences), ROUGE-1 F1 is a reasonable proxy for coverage. Exact match (all GT tokens present in prediction) provides a complementary strict signal.

**Trade-off**: Doesn't capture semantic similarity. A correct answer in different words will score low. Acceptable for a hackathon demo — the disagreement analysis is the more interesting output anyway.

**Where**: `utils/rag_evaluator.py` — `score_answer()`

---

## [2026-03-23] Evaluator accepts callables as parameters (no circular imports)

**Decision**: `rag_evaluator.run_evaluation()` accepts `dual_llm_fn` and `judge_fn` as callable parameters rather than importing them directly.

**Why**: Prevents circular imports between `utils/` modules. Makes the evaluator independently testable with mock functions. Caller (`3_Building_Codes.py`) is responsible for wiring the functions together.

**Where**: `utils/rag_evaluator.py` — `run_evaluation()` signature

---

## [2026-03-23] Home.py data path: parent / "data" (inside app/)

**Decision**: Changed `Path(__file__).parent.parent / "data"` to `Path(__file__).parent / "data"` for building codes and eval data.

**Why**: `parent.parent` points outside the git repo (to the workspace root). For Streamlit Cloud deployment, all paths must be inside `app/`. The prototype output paths remain at `parent.parent / "prototype" / "output"` since those are local-only reference files.

**Where**: `Home.py` — `bc_dir` variable; `utils/rag_pipeline.py` and `utils/rag_evaluator.py` path constants.

---

## [2026-03-23] STAMP methodology alignment in evaluator

**Decision**: The RAG evaluator follows the STAMP (Structured Taxonomy AI Measurement Protocol) methodology from Lin (under review, Journal of Marketing).

**Key choices:**
1. **Binary adequacy coding** — the judge (Claude Sonnet) assigns `answer_a_adequate` (0/1) and `answer_b_adequate` (0/1) per question, treating GPT-4.1-mini and Gemini as two independent coders.
2. **Krippendorff's alpha** — computed in `rag_evaluator.krippendorff_alpha()` from the binary adequacy pairs. Thresholds: α ≥ .80 = high reliability; α ≥ .67 = acceptable; α < .67 = unreliable → prompt needs refinement.
3. **Disagreement as diagnostic signal** — each disagreement type maps to a specific prompt-refinement action in `_generate_suggestions()`, not treated as generic noise.
4. **Model caveat** — STAMP Table W1 flags GPT-4.1-mini as "Avoid" (poor recall). We use it because (a) it matches the professor's prototype, (b) its disagreements with Gemini are informative signals, and (c) a hackathon budget does not permit the full GPT-4.1 model.

**Where**: `utils/judge.py` — `_JUDGE_USER_TEMPLATE` (adequacy fields); `utils/rag_evaluator.py` — `krippendorff_alpha()`, `build_disagreement_report()`, `_generate_suggestions()`

---

## [2026-03-23] PyMuPDF with fallback to pdfplumber

**Decision**: PDF loading tries `fitz` (PyMuPDF) first, falls back to `pdfplumber`, then raises a clear error.

**Why**: PyMuPDF (`pymupdf` package) is faster and more reliable but may have Windows install issues (requires build tools). `pdfplumber` is a pure-Python alternative. The fallback ensures the app degrades gracefully.

**Where**: `utils/rag_pipeline.py` — `BuildingCodeRetriever._load_pdf()`
