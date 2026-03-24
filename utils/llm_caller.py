"""Shared OpenRouter LLM calling utilities for the Building Codes chatbot."""

from __future__ import annotations

import json
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

MODELS: dict[str, str] = {
    "openai/gpt-4.1-mini": "GPT-4.1-mini",
    "google/gemini-2.5-flash": "Gemini-2.5-Flash",
}

JUDGE_MODEL = "openai/gpt-4.1"

MAX_RETRIES = 3


# ---------------------------------------------------------------------------
# API key loading
# ---------------------------------------------------------------------------

def load_api_key(base_dir: Path | None = None) -> str:
    """
    Load OpenRouter API key from environment variable or .env file.

    Search order:
      1. OPENROUTER_API_KEY environment variable
      2. .env file in base_dir (if provided)
      3. .env file in the app/ directory (relative to this file)
      4. .env file in the project root (two levels up from this file)
    """
    key = os.environ.get("OPENROUTER_API_KEY")
    if key:
        return key

    search_dirs: list[Path] = []
    if base_dir:
        search_dirs.append(Path(base_dir))

    this_file = Path(__file__).resolve()
    search_dirs.append(this_file.parent)          # app/utils/
    search_dirs.append(this_file.parent.parent)   # app/
    search_dirs.append(this_file.parent.parent.parent)  # project root

    for d in search_dirs:
        env_path = d / ".env"
        if env_path.exists():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.startswith("OPENROUTER_API_KEY="):
                    key = line.split("=", 1)[1].strip().strip('"').strip("'")
                    if key:
                        return key

    raise RuntimeError(
        "No OPENROUTER_API_KEY found.\n"
        "  1. Set environment variable: OPENROUTER_API_KEY=sk-or-...\n"
        "  2. Create a .env file with: OPENROUTER_API_KEY=sk-or-...\n"
        "  Get your key at https://openrouter.ai/keys"
    )


# ---------------------------------------------------------------------------
# Core API call
# ---------------------------------------------------------------------------

def call_openrouter(
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.3,
    max_tokens: int = 2000,
) -> str:
    """
    Call OpenRouter API with retry logic and exponential backoff.
    Returns raw string content from the model.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    data: dict = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    # GPT models support response_format; Gemini does not
    if model.startswith("openai/"):
        data["response_format"] = {"type": "json_object"}

    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(
                OPENROUTER_URL, headers=headers, json=data, timeout=120
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                wait = min(65, (2 ** attempt) + random.uniform(0, 1))
                time.sleep(wait)
            else:
                raise RuntimeError(
                    f"OpenRouter call failed after {MAX_RETRIES} attempts "
                    f"(model={model}): {e}"
                ) from e


# ---------------------------------------------------------------------------
# Dual-LLM parallel call
# ---------------------------------------------------------------------------

def call_dual_llm_parallel(
    api_key: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.3,
    max_tokens: int = 2000,
) -> dict[str, str]:
    """
    Call both MODELS simultaneously using a thread pool.

    Returns a dict keyed by human-readable model label:
        {"GPT-4.1-mini": "<answer>", "Gemini-2.5-Flash": "<answer>"}

    If one model fails, its value will be an error string starting with "ERROR:".
    Raises RuntimeError only if BOTH models fail.
    """
    results: dict[str, str] = {}
    errors: dict[str, str] = {}

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            executor.submit(
                call_openrouter,
                api_key,
                model_id,
                system_prompt,
                user_prompt,
                temperature,
                max_tokens,
            ): label
            for model_id, label in MODELS.items()
        }

        for future in as_completed(futures):
            label = futures[future]
            try:
                results[label] = future.result()
            except Exception as e:
                errors[label] = f"ERROR: {e}"

    if errors and not results:
        raise RuntimeError(
            f"Both LLMs failed.\n"
            + "\n".join(f"  {k}: {v}" for k, v in errors.items())
        )

    # Merge errors into results so callers always get both keys
    results.update(errors)
    return results


# ---------------------------------------------------------------------------
# JSON parsing utility (shared with judge.py and evaluator)
# ---------------------------------------------------------------------------

def parse_json_response(raw_text: str) -> dict:
    """
    Parse a JSON object from an LLM response.
    Handles markdown fences and extra surrounding text.
    """
    # Direct parse
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        pass

    # Strip markdown fences
    cleaned = re.sub(r"^```(?:json)?\s*\n?", "", raw_text.strip())
    cleaned = re.sub(r"\n?```\s*$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Regex extract first {...}
    match = re.search(r"\{[\s\S]*\}", raw_text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not parse JSON from response: {raw_text[:300]}")
