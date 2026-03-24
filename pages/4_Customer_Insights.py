"""Customer Intelligence Chat — STAMP-validated Q&A over synthetic customer data."""

import sys
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# --- Path setup ---
APP_DIR = Path(__file__).parent.parent.resolve()
PROJECT_ROOT = APP_DIR.parent.resolve()
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from utils.customer_retriever import CustomerDataRetriever, get_customer_retriever
from utils.llm_caller import load_api_key, call_dual_llm_parallel
from utils.judge import call_judge, classify_disagreement_severity
from utils.rag_evaluator import (
    load_sample_qa,
    run_evaluation,
    build_disagreement_report,
    results_to_dataframe,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Customer Intelligence",
    page_icon="💡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_DIR = PROJECT_ROOT / "prototype" / "output"
EVAL_DIR = APP_DIR / "data" / "eval"
DEFAULT_QA_PATH = EVAL_DIR / "customer_qa.json"

STARTER_QUESTIONS = [
    "Which customer segment should Neo Smart Living target first?",
    "What is the biggest barrier to purchasing the Tahoe Mini?",
    "Which positioning concept resonates most strongly?",
    "How does the permit-light feature affect purchase likelihood?",
    "What do Remote Professionals say about work-life boundaries?",
    "What are Property Maximizers' main objections?",
]

# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Building customer data index…")
def _build_retriever() -> CustomerDataRetriever:
    return get_customer_retriever(DATA_DIR)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar() -> dict:
    st.sidebar.title("⚙️ Settings")

    # API key
    st.sidebar.subheader("API Key")
    try:
        api_key = load_api_key(PROJECT_ROOT)
        st.sidebar.success(f"✅ Key loaded (…{api_key[-6:]})")
    except RuntimeError as e:
        st.sidebar.error("❌ No API key found")
        st.sidebar.caption(str(e))
        api_key = None

    st.sidebar.divider()

    # Retrieval settings
    st.sidebar.subheader("Retrieval")
    top_k = st.sidebar.slider("Top-K chunks", min_value=3, max_value=12, value=6,
                               help="Number of data chunks passed as context to each LLM")
    use_judge = st.sidebar.toggle("Use judge LLM", value=True,
                                   help="Disable to reduce API cost (shows raw answers only)")

    st.sidebar.divider()

    # Data status
    st.sidebar.subheader("Data Sources")
    interview_path = DATA_DIR / "interview_analysis.csv"
    survey_path = DATA_DIR / "synthetic_responses.csv"
    themes_path = DATA_DIR / "interview_themes.json"

    for label, path in [
        ("Interviews (30)", interview_path),
        ("Survey (60)", survey_path),
        ("Themes", themes_path),
    ]:
        if path.exists():
            st.sidebar.success(f"✅ {label}")
        else:
            st.sidebar.error(f"❌ {label} missing")

    st.sidebar.divider()

    if st.sidebar.button("🗑️ Clear chat history", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    return {"api_key": api_key, "top_k": top_k, "use_judge": use_judge}


# ---------------------------------------------------------------------------
# Chat tab
# ---------------------------------------------------------------------------

def render_chat_tab(config: dict, retriever: CustomerDataRetriever) -> None:
    st.header("Ask Your Customer Data")
    st.caption(
        f"Index: {retriever.source_summary} · "
        "Answers are grounded in synthetic interview quotes and survey statistics."
    )

    if retriever.is_empty:
        st.warning(
            "No customer data found. Ensure `prototype/output/` contains "
            "`interview_analysis.csv`, `synthetic_responses.csv`, and `interview_themes.json`."
        )

    # Chat history
    history = st.session_state.get("chat_history", [])
    for exchange in history:
        _render_exchange(exchange)

    # Starter question chips
    if not history:
        st.markdown("**Suggested questions:**")
        cols = st.columns(3)
        for i, q in enumerate(STARTER_QUESTIONS):
            if cols[i % 3].button(q, key=f"starter_{i}", use_container_width=True):
                st.session_state["_pending_question"] = q
                st.rerun()

    # Handle starter button click
    pending = st.session_state.pop("_pending_question", None)

    # Chat input
    question = st.chat_input(
        "Ask anything about your customers…",
        disabled=config["api_key"] is None or retriever.is_empty,
    ) or pending

    if question:
        with st.spinner("Retrieving customer data and querying models… (8–15 seconds)"):
            exchange = _run_pipeline(question, config, retriever)
        st.session_state.setdefault("chat_history", []).append(exchange)
        st.rerun()


def _run_pipeline(question: str, config: dict, retriever: CustomerDataRetriever) -> dict:
    api_key = config["api_key"]
    top_k = config["top_k"]
    use_judge = config["use_judge"]

    chunks = retriever.retrieve(question, top_k=top_k)
    context = retriever.format_context(chunks)

    system_prompt = (
        "You are a market research analyst for Neo Smart Living. "
        "Answer based ONLY on the provided synthetic customer data context. "
        "The data comes from 30 depth interviews and 60 survey responses with SoCal homeowners "
        "considering the Tahoe Mini — a 117 sq ft prefab backyard structure at $23,000. "
        "If the context does not contain enough information to answer, say so explicitly. "
        "Be specific: cite segment names, percentages, or direct quotes when available."
    )
    user_prompt = f"Customer Data Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"

    llm_answers = call_dual_llm_parallel(api_key, system_prompt, user_prompt, temperature=0.2)
    answer_gpt = llm_answers.get("GPT-4.1-mini", "No response")
    answer_gemini = llm_answers.get("Gemini-2.5-Flash", "No response")

    if use_judge and not answer_gpt.startswith("ERROR") and not answer_gemini.startswith("ERROR"):
        judge = call_judge(api_key, question, context, answer_gpt, answer_gemini)
        severity = classify_disagreement_severity(judge)
    else:
        judge = None
        severity = "none"

    return {
        "question": question,
        "answer_gpt": answer_gpt,
        "answer_gemini": answer_gemini,
        "judge": judge,
        "severity": severity,
        "chunks": [{"source": c.source, "chunk_index": c.chunk_index, "text": c.text} for c in chunks],
        "timestamp": datetime.now().strftime("%H:%M:%S"),
    }


def _render_exchange(exchange: dict) -> None:
    st.markdown(f"**Your question** _{exchange['timestamp']}_")
    st.info(exchange["question"])

    col_gpt, col_gemini = st.columns(2)
    with col_gpt:
        st.markdown("##### GPT-4.1-mini")
        answer = exchange["answer_gpt"]
        if answer.startswith("ERROR"):
            st.error(answer)
        else:
            st.markdown(answer)

    with col_gemini:
        st.markdown("##### Gemini 2.5 Flash")
        answer = exchange["answer_gemini"]
        if answer.startswith("ERROR"):
            st.error(answer)
        else:
            st.markdown(answer)

    judge = exchange.get("judge")
    if judge:
        st.divider()
        st.markdown("##### Judge Synthesis (Claude Sonnet)")
        st.markdown(judge.get("synthesis", ""))

        severity = exchange.get("severity", "none")
        dtype = judge.get("disagreement_type", "agreement")
        explanation = judge.get("disagreement_explanation", "")
        confidence = judge.get("confidence", "medium")

        if severity == "none":
            st.success(f"Models agreed · Confidence: {confidence}")
        elif severity == "minor":
            st.warning(f"Minor disagreement: **{dtype}** — {explanation}")
        else:
            st.error(f"Significant disagreement: **{dtype}** — {explanation}")

        col_exp1, col_exp2 = st.columns(2)
        with col_exp1:
            with st.expander("Retrieved customer data chunks"):
                for chunk in exchange.get("chunks", []):
                    st.caption(f"[{chunk['source']} | chunk {chunk['chunk_index']}]")
                    st.text(chunk["text"][:400] + ("…" if len(chunk["text"]) > 400 else ""))
        with col_exp2:
            with st.expander("Judge details (JSON)"):
                st.json(judge)

    st.divider()


# ---------------------------------------------------------------------------
# Evaluation tab
# ---------------------------------------------------------------------------

def render_eval_tab(config: dict, retriever: CustomerDataRetriever) -> None:
    st.header("Batch Accuracy Evaluation")
    st.markdown(
        "Run the full pipeline on a ground-truth Q&A set to measure accuracy "
        "and surface disagreement patterns using the **STAMP methodology**."
    )

    if config["api_key"] is None:
        st.error("API key required to run evaluation.")
        return

    qa_source = st.radio(
        "Q&A source",
        ["Use default customer_qa.json", "Upload my own"],
        horizontal=True,
    )

    qa_path = None
    if qa_source == "Use default customer_qa.json":
        if DEFAULT_QA_PATH.exists():
            st.success(f"✅ Using `{DEFAULT_QA_PATH.name}` ({_count_qa(DEFAULT_QA_PATH)} questions)")
            qa_path = DEFAULT_QA_PATH
        else:
            st.error(f"Default Q&A file not found at `{DEFAULT_QA_PATH}`")
    else:
        uploaded = st.file_uploader("Upload customer_qa.json", type=["json"])
        if uploaded:
            tmp_path = EVAL_DIR / "uploaded_customer_qa.json"
            tmp_path.write_bytes(uploaded.read())
            qa_path = tmp_path
            st.success(f"✅ Uploaded ({_count_qa(qa_path)} questions)")

    if qa_path is None:
        return

    col_run, col_info = st.columns([1, 3])
    with col_run:
        run_btn = st.button("▶ Run Evaluation", type="primary", use_container_width=True)
    with col_info:
        st.caption("Each question makes 3 API calls (2 LLMs + 1 judge). 12 questions ≈ 2–3 minutes.")

    if run_btn:
        progress_bar = st.progress(0, text="Starting evaluation…")
        status_text = st.empty()

        def on_progress(current, total, question):
            pct = current / total
            progress_bar.progress(pct, text=f"Question {current}/{total}")
            status_text.caption(f"Last: {question[:80]}…" if len(question) > 80 else question)

        results = run_evaluation(
            api_key=config["api_key"],
            qa_path=qa_path,
            retriever=retriever,
            progress_callback=on_progress,
        )
        progress_bar.progress(1.0, text="Complete!")
        status_text.empty()
        st.session_state["customer_eval_results"] = results
        st.rerun()

    if "customer_eval_results" in st.session_state:
        _render_eval_results(st.session_state["customer_eval_results"])


def _count_qa(path: Path) -> int:
    try:
        return len(json.loads(path.read_text(encoding="utf-8")))
    except Exception:
        return 0


def _render_eval_results(results) -> None:
    report = build_disagreement_report(results)
    if not results:
        st.warning("No results to display.")
        return

    st.subheader("Accuracy Metrics")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Synthesis ROUGE-1", f"{report.get('avg_synthesis_rouge1', 0):.3f}",
              help="Average ROUGE-1 F1 of judge synthesis vs. ground truth")
    c2.metric("GPT-4.1-mini ROUGE-1", f"{report.get('avg_gpt_rouge1', 0):.3f}")
    c3.metric("Gemini ROUGE-1", f"{report.get('avg_gemini_rouge1', 0):.3f}")
    c4.metric("Disagreement Rate", f"{report.get('disagreement_rate', 0):.0%}")
    c5.metric("Hallucination Rate", f"{report.get('hallucination_rate', 0):.0%}")

    # STAMP inter-rater reliability
    st.subheader("STAMP Inter-Rater Reliability")
    s1, s2, s3, s4 = st.columns(4)
    alpha_val = report.get("krippendorff_alpha", "N/A")
    s1.metric("Krippendorff's α", alpha_val,
              help="α ≥ .80 = high; α ≥ .67 = acceptable; α < .67 = refine prompt")
    s2.metric("STAMP Reliability", report.get("stamp_reliability_tier", "N/A"))
    gpt_adeq = report.get("gpt_adequacy_rate", "N/A")
    gem_adeq = report.get("gemini_adequacy_rate", "N/A")
    s3.metric("GPT-4.1-mini Adequacy",
              f"{gpt_adeq:.0%}" if isinstance(gpt_adeq, float) else gpt_adeq)
    s4.metric("Gemini Adequacy",
              f"{gem_adeq:.0%}" if isinstance(gem_adeq, float) else gem_adeq)

    st.divider()

    col_chart, col_suggestions = st.columns([1, 1])

    with col_chart:
        st.subheader("Disagreement Types")
        type_counts = report.get("disagreement_type_counts", {})
        if type_counts:
            fig, ax = plt.subplots(figsize=(5, 3))
            labels = list(type_counts.keys())
            values = list(type_counts.values())
            colors = [
                "#2ecc71" if l == "agreement"
                else "#e67e22" if l in ("hallucination", "factual_conflict")
                else "#3498db"
                for l in labels
            ]
            ax.barh(labels, values, color=colors)
            ax.set_xlabel("Count")
            ax.set_title("Disagreement Type Distribution")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    with col_suggestions:
        st.subheader("Prompt Engineering Suggestions")
        for suggestion in report.get("prompt_engineering_suggestions", []):
            st.info(suggestion)

    contested = report.get("most_contested_questions", [])
    if contested:
        st.subheader("Most Contested Questions")
        st.dataframe(pd.DataFrame(contested), use_container_width=True, hide_index=True)

    st.divider()

    st.subheader("Full Results")
    df = results_to_dataframe(results)
    display_cols = [
        "question_id", "topic", "difficulty", "disagreement_type",
        "disagreement_severity", "answer_a_adequate", "answer_b_adequate",
        "gpt_rouge1", "gemini_rouge1", "synthesis_rouge1", "error",
    ]
    st.dataframe(df[[c for c in display_cols if c in df.columns]],
                 use_container_width=True, hide_index=True)

    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇ Download results CSV", csv_bytes, "customer_eval_results.csv", "text/csv")
    with col_dl2:
        report_json = json.dumps(report, indent=2).encode("utf-8")
        st.download_button("⬇ Download report JSON", report_json, "customer_disagreement_report.json", "application/json")

    with st.expander("Full disagreement report (JSON)"):
        st.json(report)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    st.title("💡 Customer Intelligence Chat")
    st.markdown(
        "Ask questions about your customers — answered independently by two LLMs, "
        "synthesized by a judge, and validated using the **STAMP methodology**. "
        "Answers are grounded in 30 synthetic depth interviews and 60 survey responses."
    )
    st.divider()

    config = render_sidebar()
    retriever = _build_retriever()

    tab_chat, tab_eval = st.tabs(["💬 Ask Your Data", "📊 Evaluation Mode"])

    with tab_chat:
        render_chat_tab(config, retriever)

    with tab_eval:
        render_eval_tab(config, retriever)


if __name__ == "__main__":
    main()
else:
    main()
