"""Batch evaluation of the RAG chatbot against a ground-truth Q&A set."""

from __future__ import annotations

import json
import re
import string
from collections import Counter
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

from .llm_caller import call_dual_llm_parallel
from .judge import call_judge, classify_disagreement_severity
from .rag_pipeline import BuildingCodeRetriever


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    question_id: str
    question: str
    topic: str
    difficulty: str
    ground_truth: str

    # LLM answers
    answer_gpt: str = ""
    answer_gemini: str = ""
    judge_synthesis: str = ""

    # Disagreement
    disagreement_type: str = "agreement"
    disagreement_severity: str = "none"
    disagreement_explanation: str = ""
    judge_confidence: str = "medium"
    judge_citations: list = field(default_factory=list)

    # Accuracy scores
    gpt_rouge1: float = 0.0
    gemini_rouge1: float = 0.0
    synthesis_rouge1: float = 0.0
    gpt_exact: bool = False
    gemini_exact: bool = False
    synthesis_exact: bool = False

    # Error flag
    error: str = ""


# ---------------------------------------------------------------------------
# Q&A loading
# ---------------------------------------------------------------------------

def load_sample_qa(path: Path) -> list[dict]:
    """
    Load and validate sample Q&A from a JSON file.

    Expected schema:
      [{"id": str, "question": str, "ground_truth": str,
        "topic": str (optional), "difficulty": str (optional)}]
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Q&A file not found: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(data, list):
        raise ValueError("Q&A file must contain a JSON array at the top level.")

    for i, item in enumerate(data):
        for required in ("id", "question", "ground_truth"):
            if required not in item:
                raise ValueError(
                    f"Item {i} is missing required field '{required}'."
                )

    return data


# ---------------------------------------------------------------------------
# Scoring (pure Python ROUGE-1, no external library)
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    """Lowercase, remove punctuation, split into tokens."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text.split()


def score_answer(prediction: str, ground_truth: str) -> dict:
    """
    Compute ROUGE-1 F1 and normalized exact match between prediction and ground_truth.

    ROUGE-1: token overlap between prediction and ground truth.
    Exact match: normalized (lowercase, stripped punctuation) string equality.
    """
    pred_tokens = _tokenize(prediction)
    gt_tokens = _tokenize(ground_truth)

    if not pred_tokens or not gt_tokens:
        return {"rouge1": 0.0, "exact": False}

    pred_counts = Counter(pred_tokens)
    gt_counts = Counter(gt_tokens)

    overlap = sum((pred_counts & gt_counts).values())
    precision = overlap / len(pred_tokens) if pred_tokens else 0.0
    recall = overlap / len(gt_tokens) if gt_tokens else 0.0

    if precision + recall == 0:
        rouge1 = 0.0
    else:
        rouge1 = 2 * precision * recall / (precision + recall)

    # Normalized exact match
    pred_norm = " ".join(pred_tokens)
    gt_norm = " ".join(gt_tokens)
    exact = pred_norm == gt_norm

    return {"rouge1": round(rouge1, 4), "exact": exact}


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------

def run_evaluation(
    api_key: str,
    qa_path: Path,
    retriever: BuildingCodeRetriever,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> list[EvalResult]:
    """
    Run the full RAG + dual-LLM + judge pipeline on each Q&A pair.

    Runs sequentially (not parallel) to control API costs and rate limits.
    Calls progress_callback(current_index, total, question_text) after each item.
    """
    qa_items = load_sample_qa(qa_path)
    total = len(qa_items)
    results: list[EvalResult] = []

    for i, item in enumerate(qa_items):
        question = item["question"]
        ground_truth = item["ground_truth"]
        topic = item.get("topic", "")
        difficulty = item.get("difficulty", "")

        result = EvalResult(
            question_id=item["id"],
            question=question,
            topic=topic,
            difficulty=difficulty,
            ground_truth=ground_truth,
        )

        try:
            # 1. Retrieve context
            chunks = retriever.retrieve(question, top_k=5)
            context = retriever.format_context(chunks)

            # 2. Build prompts
            system_prompt = (
                "You are a building code expert. Answer based ONLY on the provided context. "
                "If the context does not contain the answer, say so explicitly. "
                "Be concise and cite specific sections or requirements when available."
            )
            user_prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"

            # 3. Dual LLM answers
            llm_answers = call_dual_llm_parallel(
                api_key, system_prompt, user_prompt, temperature=0.2
            )
            gpt_label = "GPT-4.1-mini"
            gemini_label = "Gemini-2.5-Flash"
            answer_gpt = llm_answers.get(gpt_label, "ERROR: no response")
            answer_gemini = llm_answers.get(gemini_label, "ERROR: no response")

            result.answer_gpt = answer_gpt
            result.answer_gemini = answer_gemini

            # 4. Judge
            judge = call_judge(api_key, question, context, answer_gpt, answer_gemini)
            result.judge_synthesis = judge.get("synthesis", "")
            result.disagreement_type = judge.get("disagreement_type", "agreement")
            result.disagreement_severity = classify_disagreement_severity(judge)
            result.disagreement_explanation = judge.get("disagreement_explanation", "")
            result.judge_confidence = judge.get("confidence", "medium")
            result.judge_citations = judge.get("citations", [])

            # 5. Score
            gpt_score = score_answer(answer_gpt, ground_truth)
            gemini_score = score_answer(answer_gemini, ground_truth)
            synth_score = score_answer(result.judge_synthesis, ground_truth)

            result.gpt_rouge1 = gpt_score["rouge1"]
            result.gemini_rouge1 = gemini_score["rouge1"]
            result.synthesis_rouge1 = synth_score["rouge1"]
            result.gpt_exact = gpt_score["exact"]
            result.gemini_exact = gemini_score["exact"]
            result.synthesis_exact = synth_score["exact"]

        except Exception as e:
            result.error = str(e)

        results.append(result)

        if progress_callback:
            progress_callback(i + 1, total, question)

    return results


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def build_disagreement_report(results: list[EvalResult]) -> dict:
    """
    Aggregate evaluation results into a summary report with prompt engineering suggestions.
    """
    total = len(results)
    if total == 0:
        return {}

    error_count = sum(1 for r in results if r.error)
    valid = [r for r in results if not r.error]
    valid_total = len(valid)

    if valid_total == 0:
        return {"error": "All evaluations failed.", "error_count": error_count}

    # Disagreement type distribution
    type_counts = Counter(r.disagreement_type for r in valid)

    # Severity distribution
    severity_counts = Counter(r.disagreement_severity for r in valid)

    # Accuracy
    avg_gpt_rouge1 = sum(r.gpt_rouge1 for r in valid) / valid_total
    avg_gemini_rouge1 = sum(r.gemini_rouge1 for r in valid) / valid_total
    avg_synthesis_rouge1 = sum(r.synthesis_rouge1 for r in valid) / valid_total

    hallucination_rate = type_counts.get("hallucination", 0) / valid_total
    missing_context_rate = type_counts.get("missing_context", 0) / valid_total
    disagreement_rate = (
        sum(1 for r in valid if r.disagreement_severity != "none") / valid_total
    )

    # Most contested questions (major disagreements with lowest synthesis ROUGE-1)
    major_disagreements = [
        r for r in valid if r.disagreement_severity == "major"
    ]
    major_disagreements.sort(key=lambda r: r.synthesis_rouge1)
    contested = [
        {
            "id": r.question_id,
            "question": r.question,
            "disagreement_type": r.disagreement_type,
            "synthesis_rouge1": r.synthesis_rouge1,
        }
        for r in major_disagreements[:5]
    ]

    # Topic-level accuracy
    topic_accuracy: dict[str, list[float]] = {}
    for r in valid:
        topic_accuracy.setdefault(r.topic, []).append(r.synthesis_rouge1)
    topic_avg = {
        t: round(sum(scores) / len(scores), 4)
        for t, scores in topic_accuracy.items()
    }

    # Prompt engineering suggestions
    suggestions = _generate_suggestions(
        hallucination_rate, missing_context_rate, disagreement_rate, type_counts
    )

    return {
        "total_questions": total,
        "valid_evaluations": valid_total,
        "error_count": error_count,
        "avg_gpt_rouge1": round(avg_gpt_rouge1, 4),
        "avg_gemini_rouge1": round(avg_gemini_rouge1, 4),
        "avg_synthesis_rouge1": round(avg_synthesis_rouge1, 4),
        "hallucination_rate": round(hallucination_rate, 4),
        "missing_context_rate": round(missing_context_rate, 4),
        "disagreement_rate": round(disagreement_rate, 4),
        "disagreement_type_counts": dict(type_counts),
        "severity_counts": dict(severity_counts),
        "most_contested_questions": contested,
        "accuracy_by_topic": topic_avg,
        "prompt_engineering_suggestions": suggestions,
    }


def _generate_suggestions(
    hallucination_rate: float,
    missing_context_rate: float,
    disagreement_rate: float,
    type_counts: Counter,
) -> list[str]:
    """Rule-based prompt engineering suggestions."""
    suggestions = []

    if missing_context_rate > 0.30:
        suggestions.append(
            f"High missing-context rate ({missing_context_rate:.0%}): Add more building code "
            "documents to data/building_codes/ to improve retrieval coverage."
        )

    if hallucination_rate > 0.20:
        suggestions.append(
            f"High hallucination rate ({hallucination_rate:.0%}): Strengthen the system prompt "
            "with 'Answer ONLY from the provided context. If the answer is not in the context, "
            "say I don\\'t know.' Consider reducing LLM temperature."
        )

    if type_counts.get("ambiguous_source", 0) > 2:
        suggestions.append(
            "Several ambiguous-source disagreements detected: The source documents may have "
            "conflicting or unclear language. Consider adding more specific section references "
            "to the source documents or increasing chunk overlap."
        )

    if type_counts.get("factual_conflict", 0) > 1:
        suggestions.append(
            "Factual conflicts detected between models: Review the contested questions and "
            "verify ground truth answers against authoritative source documents."
        )

    if disagreement_rate > 0.50:
        suggestions.append(
            f"Overall disagreement rate is high ({disagreement_rate:.0%}): Consider increasing "
            "top_k retrieved chunks to give models more context, or try the embedding "
            "retrieval method for more semantic search coverage."
        )

    if not suggestions:
        suggestions.append(
            "No major issues detected. Models are generally in agreement with good context coverage."
        )

    return suggestions


# ---------------------------------------------------------------------------
# DataFrame export
# ---------------------------------------------------------------------------

def results_to_dataframe(results: list[EvalResult]) -> pd.DataFrame:
    """Convert evaluation results to a flat DataFrame."""
    rows = []
    for r in results:
        d = asdict(r)
        d["judge_citations"] = "; ".join(r.judge_citations)
        rows.append(d)
    return pd.DataFrame(rows)


def save_report(results: list[EvalResult], output_dir: Path) -> None:
    """Save results CSV and disagreement report JSON to output_dir."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = results_to_dataframe(results)
    df.to_csv(output_dir / "eval_results.csv", index=False)

    report = build_disagreement_report(results)
    (output_dir / "disagreement_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
