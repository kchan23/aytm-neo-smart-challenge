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

    # STAMP binary adequacy coding (0 = inadequate, 1 = adequate, -1 = not evaluated)
    answer_a_adequate: int = -1   # GPT-4.1-mini
    answer_b_adequate: int = -1   # Gemini-2.5-Flash

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


def krippendorff_alpha(ratings_a: list[int], ratings_b: list[int]) -> float:
    """
    Compute Krippendorff's alpha for binary nominal data with two raters.

    Used per STAMP (Lin, under review) to measure inter-rater reliability between
    two LLM coders on binary adequacy judgements (0 = inadequate, 1 = adequate).

    Thresholds (STAMP / standard):
      α ≥ .80 → high reliability
      α ≥ .67 → acceptable (tentative conclusions)
      α < .67 → unreliable — prompt needs refinement

    Returns float in [-1, 1], or NaN if computation is undefined (e.g. all ratings identical).
    """
    if len(ratings_a) != len(ratings_b) or not ratings_a:
        return float("nan")

    n = len(ratings_a)
    # Observed disagreement: proportion of items where raters differ
    disagree = sum(a != b for a, b in zip(ratings_a, ratings_b))
    D_o = disagree / n

    # Expected disagreement using all 2n ratings (Krippendorff nominal metric)
    all_ratings = ratings_a + ratings_b
    N = len(all_ratings)  # = 2n
    n0 = all_ratings.count(0)
    n1 = all_ratings.count(1)

    # Guard against degenerate case (all ratings identical → D_e = 0)
    if N * (N - 1) == 0 or (n0 == 0 or n1 == 0):
        return float("nan")

    D_e = 2 * n0 * n1 / (N * (N - 1))

    return round(1.0 - D_o / D_e, 4)


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
            # STAMP binary adequacy coding
            result.answer_a_adequate = int(judge.get("answer_a_adequate", -1))
            result.answer_b_adequate = int(judge.get("answer_b_adequate", -1))

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

    # STAMP: Krippendorff's alpha from binary adequacy ratings
    adequate_pairs = [
        (r.answer_a_adequate, r.answer_b_adequate)
        for r in valid
        if r.answer_a_adequate in (0, 1) and r.answer_b_adequate in (0, 1)
    ]
    if adequate_pairs:
        ratings_a = [p[0] for p in adequate_pairs]
        ratings_b = [p[1] for p in adequate_pairs]
        alpha = krippendorff_alpha(ratings_a, ratings_b)
        # F1 per model: adequacy rate = proportion rated adequate
        gpt_adequacy_rate = sum(ratings_a) / len(ratings_a)
        gemini_adequacy_rate = sum(ratings_b) / len(ratings_b)
    else:
        alpha = float("nan")
        gpt_adequacy_rate = float("nan")
        gemini_adequacy_rate = float("nan")

    alpha_str = f"{alpha:.3f}" if alpha == alpha else "N/A"  # NaN check

    # STAMP reliability tier
    if alpha != alpha:  # NaN
        stamp_reliability = "undetermined"
    elif alpha >= 0.80:
        stamp_reliability = "high (α ≥ .80)"
    elif alpha >= 0.67:
        stamp_reliability = "acceptable (α ≥ .67)"
    else:
        stamp_reliability = "unreliable (α < .67) — refine prompt"

    # Prompt engineering suggestions
    suggestions = _generate_suggestions(
        hallucination_rate, missing_context_rate, disagreement_rate, type_counts, alpha
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
        # STAMP inter-rater reliability
        "krippendorff_alpha": alpha_str,
        "stamp_reliability_tier": stamp_reliability,
        "gpt_adequacy_rate": round(gpt_adequacy_rate, 4) if gpt_adequacy_rate == gpt_adequacy_rate else "N/A",
        "gemini_adequacy_rate": round(gemini_adequacy_rate, 4) if gemini_adequacy_rate == gemini_adequacy_rate else "N/A",
        "stamp_rated_pairs": len(adequate_pairs),
        "prompt_engineering_suggestions": suggestions,
    }


def _generate_suggestions(
    hallucination_rate: float,
    missing_context_rate: float,
    disagreement_rate: float,
    type_counts: Counter,
    alpha: float = float("nan"),
) -> list[str]:
    """
    STAMP-aligned prompt engineering suggestions.

    Per Lin (under review): inter-model disagreement is a diagnostic signal —
    each type maps to a specific prompt refinement strategy.
    """
    suggestions = []

    # STAMP: low inter-rater reliability → refine the coding prompt
    if alpha == alpha and alpha < 0.67:  # not NaN and below threshold
        suggestions.append(
            f"[STAMP] Low inter-rater reliability (α = {alpha:.3f}, threshold α ≥ .67): "
            "The two LLMs are producing divergent adequacy judgements. Per the STAMP methodology, "
            "this signals the system prompt needs a clearer definition of 'adequate answer', "
            "explicit inclusion/exclusion criteria, and grounded examples with chain-of-thought. "
            "Refine the system prompt before drawing conclusions from this evaluation."
        )
    elif alpha == alpha and alpha < 0.80:
        suggestions.append(
            f"[STAMP] Acceptable inter-rater reliability (α = {alpha:.3f}). "
            "Findings are tentatively valid. Aim for α ≥ .80 for high-confidence conclusions. "
            "Add 1–2 chain-of-thought examples to the system prompt to tighten agreement."
        )

    if missing_context_rate > 0.30:
        suggestions.append(
            f"[Missing context — {missing_context_rate:.0%}] Retrieval is failing to surface "
            "relevant chunks. Action: (1) Add more building code documents to data/building_codes/, "
            "(2) increase top_k, or (3) switch to embedding-based retrieval for semantic coverage."
        )

    if hallucination_rate > 0.20:
        suggestions.append(
            f"[Hallucination — {hallucination_rate:.0%}] Models are adding facts not present in "
            "retrieved context. Action: Strengthen the grounding instruction — "
            "'Answer ONLY from the provided context. If the answer is absent, say so explicitly.' "
            "Reduce temperature to 0.1. Consider a stricter no-fabrication example in the prompt."
        )

    if type_counts.get("ambiguous_source", 0) > 2:
        suggestions.append(
            "[Ambiguous source] Multiple questions triggered ambiguous-source disagreements. "
            "Action: The source documents contain unclear or conflicting language. "
            "Add explicit section-citation instructions to the prompt, increase chunk overlap, "
            "or annotate ambiguous passages in the source documents."
        )

    if type_counts.get("factual_conflict", 0) > 1:
        suggestions.append(
            "[Factual conflict] Direct factual contradictions detected between models. "
            "Action: Review the contested questions in 'Most Contested' table and verify "
            "ground truth against authoritative source documents. "
            "These items may expose errors in the source PDFs or the ground-truth Q&A set."
        )

    if disagreement_rate > 0.50:
        suggestions.append(
            f"[High disagreement — {disagreement_rate:.0%}] More than half of questions trigger "
            "model disagreement. Action: Increase top_k retrieved chunks, try embedding retrieval, "
            "or add a clearer answer-format template to the system prompt."
        )

    if not suggestions:
        suggestions.append(
            "No major issues detected. Models show strong inter-rater agreement with good "
            "context coverage. Per STAMP, this indicates the prompt and retrieval pipeline "
            "are well-calibrated for this question set."
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
