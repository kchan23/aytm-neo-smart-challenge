"""Judge LLM logic: compare two LLM answers, synthesize, and classify disagreement."""

from __future__ import annotations

from .llm_caller import call_openrouter, parse_json_response, JUDGE_MODEL

# ---------------------------------------------------------------------------
# Disagreement taxonomy
# ---------------------------------------------------------------------------

DISAGREEMENT_TYPES = [
    "agreement",          # Models are substantively aligned
    "ambiguous_source",   # Both answers plausible; source text is unclear
    "hallucination",      # One or both add facts not present in context
    "missing_context",    # Neither found sufficient info in retrieved chunks
    "factual_conflict",   # Direct contradiction on a specific fact or number
    "emphasis_difference",# Same facts, different framing or weight
]

SEVERITY_MAP = {
    "agreement": "none",
    "emphasis_difference": "minor",
    "ambiguous_source": "minor",
    "missing_context": "minor",
    "hallucination": "major",
    "factual_conflict": "major",
}

# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

_JUDGE_SYSTEM = """\
You are an expert building code analyst and LLM output evaluator.
Your job is to compare two independent answers to a building code question,
identify where they agree and disagree, synthesize the most accurate answer,
and classify the nature of any disagreement.

CRITICAL RULES:
- Base your synthesis ONLY on the provided context. Do not add external knowledge.
- If the context does not contain enough information to answer, say so explicitly.
- Be specific about which chunks (by source and chunk index) support your synthesis.
- Output ONLY valid JSON — no markdown, no preamble."""

_JUDGE_USER_TEMPLATE = """\
QUESTION:
{question}

RETRIEVED CONTEXT:
{context}

ANSWER A (GPT-4.1-mini):
{answer_a}

ANSWER B (Gemini-2.5-Flash):
{answer_b}

Analyze the two answers and return a JSON object with exactly these keys:

{{
  "agreements": ["<point they agree on>", ...],
  "disagreements": ["<specific point of disagreement>", ...],
  "disagreement_type": "<one of: agreement | ambiguous_source | hallucination | missing_context | factual_conflict | emphasis_difference>",
  "disagreement_explanation": "<1-2 sentence explanation of WHY they disagree, or empty string if agreement>",
  "synthesis": "<your authoritative answer grounded only in the provided context>",
  "confidence": "<high | medium | low>",
  "citations": ["<source filename and chunk index referenced>", ...]
}}

IMPORTANT:
- "agreements" and "disagreements" should be concise bullet-point strings.
- "disagreement_type" must be exactly one of the listed values.
- "synthesis" should be a complete, readable answer a user would find helpful.
- "confidence" = high if context clearly supports a complete answer;
                 medium if partial or somewhat ambiguous;
                 low if context is insufficient or models both appear to hallucinate.
"""


def build_judge_prompt(
    question: str, context: str, answer_a: str, answer_b: str
) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for the judge model."""
    user_prompt = _JUDGE_USER_TEMPLATE.format(
        question=question,
        context=context,
        answer_a=answer_a,
        answer_b=answer_b,
    )
    return _JUDGE_SYSTEM, user_prompt


# ---------------------------------------------------------------------------
# Judge call
# ---------------------------------------------------------------------------

def call_judge(
    api_key: str,
    question: str,
    context: str,
    answer_a: str,
    answer_b: str,
) -> dict:
    """
    Call the judge LLM to compare answer_a and answer_b.

    Returns a dict with keys: agreements, disagreements, disagreement_type,
    disagreement_explanation, synthesis, confidence, citations.

    On failure returns a structured error dict instead of raising.
    """
    system_prompt, user_prompt = build_judge_prompt(question, context, answer_a, answer_b)

    try:
        raw = call_openrouter(
            api_key=api_key,
            model=JUDGE_MODEL,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.1,
            max_tokens=1500,
        )
        result = parse_json_response(raw)
        # Ensure all expected keys are present
        return _validate_judge_result(result)
    except Exception as e:
        return _error_judge_result(str(e))


# ---------------------------------------------------------------------------
# Severity classification
# ---------------------------------------------------------------------------

def classify_disagreement_severity(judge_result: dict) -> str:
    """
    Returns "none", "minor", or "major" based on disagreement_type.
    """
    dtype = judge_result.get("disagreement_type", "agreement")
    return SEVERITY_MAP.get(dtype, "minor")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _validate_judge_result(result: dict) -> dict:
    """Ensure all required keys exist, filling defaults where missing."""
    defaults = {
        "agreements": [],
        "disagreements": [],
        "disagreement_type": "agreement",
        "disagreement_explanation": "",
        "synthesis": "",
        "confidence": "medium",
        "citations": [],
    }
    for key, default in defaults.items():
        if key not in result:
            result[key] = default

    # Normalise disagreement_type to known values
    if result["disagreement_type"] not in DISAGREEMENT_TYPES:
        result["disagreement_type"] = "ambiguous_source"

    return result


def _error_judge_result(error_message: str) -> dict:
    """Return a structured error dict when the judge call fails."""
    return {
        "agreements": [],
        "disagreements": [],
        "disagreement_type": "missing_context",
        "disagreement_explanation": f"Judge LLM call failed: {error_message}",
        "synthesis": "Unable to synthesize an answer — the judge model encountered an error. Please try again.",
        "confidence": "low",
        "citations": [],
        "_error": error_message,
    }
