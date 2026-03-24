"""Customer data retriever: build a TF-IDF index over synthetic interview + survey data."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .rag_pipeline import DocumentChunk


# ---------------------------------------------------------------------------
# Question labels for interview IQ columns
# ---------------------------------------------------------------------------

IQ_LABELS = {
    "IQ1": "Backyard Relationship",
    "IQ2": "Unmet Home Needs",
    "IQ3": "Prior Consideration of Backyard Structures",
    "IQ4": "Lifestyle Fantasy / Ideal Use",
    "IQ5": "Work-Life Boundaries",
    "IQ6": "Product Reaction (Tahoe Mini)",
    "IQ7": "Purchase Barriers & Drivers",
    "IQ8": "Discovery & Social Influence",
}

# ---------------------------------------------------------------------------
# Barrier, concept, and value-driver display names for survey summaries
# ---------------------------------------------------------------------------

BARRIER_COLS = {
    "Q5_cost": "Total cost (~$23,000)",
    "Q5_hoa": "HOA restrictions",
    "Q5_permit": "Permit uncertainty",
    "Q5_space": "Backyard space/access",
    "Q5_financing": "Financing options",
    "Q5_quality": "Build quality concerns",
    "Q5_resale": "Resale value impact",
}

CONCEPT_COLS = {
    "Q9a": "Home Office appeal",
    "Q10a": "Guest Suite / STR Income appeal",
    "Q11a": "Wellness Studio appeal",
    "Q12a": "Adventure Basecamp appeal",
    "Q13a": "Simplicity / Speed appeal",
    "Q9b": "Home Office purchase likelihood",
    "Q10b": "Guest Suite / STR Income purchase likelihood",
    "Q11b": "Wellness Studio purchase likelihood",
    "Q12b": "Adventure Basecamp purchase likelihood",
    "Q13b": "Simplicity / Speed purchase likelihood",
}

VALUE_COLS = {
    "Q15": "Permit-light sizing",
    "Q16": "One-day installation",
    "Q17": "Build quality / craftsmanship",
}


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

class CustomerDataRetriever:
    """
    Load synthetic customer data (interviews + survey) from prototype/output/,
    build a TF-IDF index over text chunks, and retrieve top-k relevant chunks
    for a natural-language query.

    Compatible interface with BuildingCodeRetriever:
      .retrieve(query, top_k) → list[DocumentChunk]
      .format_context(chunks, max_chars) → str
      .is_empty → bool
    """

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)  # prototype/output/

        self._chunks: list[DocumentChunk] = []
        self._index = None
        self._vectorizer = None
        self._built = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_and_build(self) -> None:
        """Load all data sources and build the TF-IDF index."""
        self._chunks = self._build_corpus()
        self._build_tfidf_index()
        self._built = True

    def retrieve(self, query: str, top_k: int = 6) -> list[DocumentChunk]:
        """Return top_k most relevant chunks for query."""
        if not self._built:
            self.load_and_build()
        if not self._chunks:
            return []
        return self._retrieve_tfidf(query, top_k)

    def format_context(self, chunks: list[DocumentChunk], max_chars: int = 4000) -> str:
        """Join chunks into a single context string with source attribution."""
        parts = []
        total = 0
        for chunk in chunks:
            header = f"[{chunk.source} | chunk {chunk.chunk_index}]"
            block = f"{header}\n{chunk.text}"
            if total + len(block) > max_chars:
                remaining = max_chars - total
                if remaining > len(header) + 20:
                    block = block[:remaining] + "…"
                    parts.append(block)
                break
            parts.append(block)
            total += len(block) + 2
        return "\n\n".join(parts)

    @property
    def is_empty(self) -> bool:
        return len(self._chunks) == 0

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    @property
    def source_summary(self) -> str:
        interview = sum(1 for c in self._chunks if "interview" in c.source.lower() and "theme" not in c.source.lower())
        themes = sum(1 for c in self._chunks if "theme" in c.source.lower())
        survey = sum(1 for c in self._chunks if "survey" in c.source.lower())
        return f"{interview} interview quotes, {themes} theme summaries, {survey} survey stat chunks"

    # ------------------------------------------------------------------
    # Corpus construction
    # ------------------------------------------------------------------

    def _build_corpus(self) -> list[DocumentChunk]:
        chunks: list[DocumentChunk] = []

        # 1. Interview quote chunks
        interview_path = self.data_dir / "interview_analysis.csv"
        if interview_path.exists():
            chunks.extend(self._interview_chunks(interview_path))

        # 2. Theme summary chunks
        themes_path = self.data_dir / "interview_themes.json"
        if themes_path.exists():
            chunks.extend(self._theme_chunks(themes_path))

        # 3. Survey statistic summary chunks
        survey_path = self.data_dir / "synthetic_responses.csv"
        if survey_path.exists():
            chunks.extend(self._survey_stat_chunks(survey_path))

        return chunks

    def _interview_chunks(self, path: Path) -> list[DocumentChunk]:
        df = pd.read_csv(path)
        chunks = []
        idx = 0

        for _, row in df.iterrows():
            persona = row.get("persona_name", "Unknown")
            work = row.get("work_arrangement", "")
            model = row.get("model", "")
            sentiment = row.get("sentiment_label", "")
            emotion = row.get("primary_emotion", "")

            source = f"Interview: {persona} ({model})"

            for iq_col, iq_label in IQ_LABELS.items():
                answer = row.get(iq_col, "")
                if not isinstance(answer, str) or not answer.strip():
                    continue

                text = (
                    f"Persona: {persona} | Work: {work}\n"
                    f"Question: {iq_label}\n"
                    f"{answer.strip()}\n"
                    f"[Sentiment: {sentiment} | Emotion: {emotion}]"
                )
                chunks.append(DocumentChunk(
                    text=text,
                    source=source,
                    chunk_index=idx,
                    metadata={"persona": persona, "question": iq_col, "model": model},
                ))
                idx += 1

        return chunks

    def _theme_chunks(self, path: Path) -> list[DocumentChunk]:
        data = json.loads(path.read_text(encoding="utf-8"))
        chunks = []

        themes = data.get("llm_themes", data.get("themes", []))
        for i, theme in enumerate(themes):
            name = theme.get("theme_name", theme.get("name", f"Theme {i+1}"))
            freq = theme.get("frequency", "")
            desc = theme.get("description", "")
            # quotes may be list of strings or list of dicts with "quote" key
            raw_quotes = theme.get("supporting_quotes", theme.get("quotes", []))
            quotes = [
                q["quote"] if isinstance(q, dict) else q
                for q in raw_quotes[:3]
            ]

            quote_block = "\n".join(f'  • "{q}"' for q in quotes)
            text = (
                f"Interview Theme: {name} (frequency: {freq})\n"
                f"{desc}\n"
                f"Representative quotes:\n{quote_block}"
            )
            chunks.append(DocumentChunk(
                text=text,
                source="Interview Themes",
                chunk_index=i,
                metadata={"theme": name},
            ))

        return chunks

    def _survey_stat_chunks(self, path: Path) -> list[DocumentChunk]:
        df = pd.read_csv(path)
        chunks = []
        idx = 0

        def add(text: str, label: str) -> None:
            nonlocal idx
            chunks.append(DocumentChunk(
                text=text,
                source=f"Survey Stats: {label}",
                chunk_index=idx,
                metadata={"topic": label},
            ))
            idx += 1

        # --- Purchase intent by segment ---
        pi = df.groupby("segment_name")[["Q1", "Q2"]].mean().round(2)
        lines = ["Purchase interest (Q1, 1–5) and likelihood (Q2, 1–5) by segment:"]
        for seg, row in pi.iterrows():
            lines.append(f"  {seg}: interest={row['Q1']}, likelihood={row['Q2']}")
        add("\n".join(lines), "Purchase Intent by Segment")

        # --- Purchase intent by model ---
        pm = df.groupby("model")[["Q1", "Q2"]].mean().round(2)
        lines = ["Purchase intent by LLM model:"]
        for model, row in pm.iterrows():
            lines.append(f"  {model}: interest={row['Q1']}, likelihood={row['Q2']}")
        add("\n".join(lines), "Purchase Intent by Model")

        # --- Intended use (Q3) ---
        q3 = df["Q3"].value_counts(normalize=True).mul(100).round(1)
        lines = ["Primary intended use (Q3) — % of respondents:"]
        for use, pct in q3.items():
            lines.append(f"  {use}: {pct}%")
        add("\n".join(lines), "Intended Use (Q3)")

        # --- Intended use by segment ---
        q3s = df.groupby("segment_name")["Q3"].agg(lambda x: x.value_counts().index[0])
        lines = ["Most common intended use per segment:"]
        for seg, use in q3s.items():
            lines.append(f"  {seg}: {use}")
        add("\n".join(lines), "Intended Use by Segment")

        # --- Barrier severity (Q5_*) ---
        b_means = df[[*BARRIER_COLS]].mean().round(2)
        lines = ["Barrier severity ratings (Q5, 1=not a barrier, 5=major barrier):"]
        for col, label in BARRIER_COLS.items():
            lines.append(f"  {label}: {b_means[col]}")
        add("\n".join(lines), "Barrier Severity Overall")

        # --- Barrier severity by segment ---
        b_seg = df.groupby("segment_name")[[*BARRIER_COLS]].mean().round(2)
        lines = ["Barrier severity by segment (mean, 1–5):"]
        for seg, row in b_seg.iterrows():
            top = max(BARRIER_COLS.items(), key=lambda kv: row[kv[0]])
            lines.append(f"  {seg} — top barrier: {top[1]} ({row[top[0]]})")
        add("\n".join(lines), "Barrier Severity by Segment")

        # --- Greatest barrier (Q6) ---
        q6 = df["Q6"].value_counts(normalize=True).mul(100).round(1)
        lines = ["Single greatest barrier cited (Q6):"]
        for barrier, pct in q6.items():
            lines.append(f"  {barrier}: {pct}%")
        add("\n".join(lines), "Greatest Barrier (Q6)")

        # --- Permit-light effect (Q7) ---
        q7 = df.groupby("segment_name")["Q7"].mean().round(2)
        lines = ["Effect of permit-light sizing on purchase likelihood (Q7, 1–5) by segment:"]
        for seg, val in q7.items():
            lines.append(f"  {seg}: {val}")
        add("\n".join(lines), "Permit-Light Effect (Q7)")

        # --- Concept appeal + likelihood ---
        appeal_cols = [c for c in CONCEPT_COLS if c.endswith("a")]
        like_cols = [c for c in CONCEPT_COLS if c.endswith("b")]
        concept_names = ["Home Office", "Guest Suite/STR", "Wellness Studio", "Adventure Basecamp", "Simplicity/Speed"]

        appeal_means = df[appeal_cols].mean().round(2).values
        like_means = df[like_cols].mean().round(2).values
        lines = ["Concept test ratings (appeal and purchase likelihood, 1–5):"]
        for name, appeal, like in zip(concept_names, appeal_means, like_means):
            lines.append(f"  {name}: appeal={appeal}, likelihood={like}")
        add("\n".join(lines), "Concept Test Ratings")

        # --- Preferred concept (Q14) ---
        q14 = df["Q14"].value_counts(normalize=True).mul(100).round(1)
        lines = ["Preferred concept overall (Q14):"]
        for concept, pct in q14.items():
            lines.append(f"  {concept}: {pct}%")
        add("\n".join(lines), "Preferred Concept (Q14)")

        # --- Value drivers (Q15–Q17) ---
        vd = df[[*VALUE_COLS]].mean().round(2)
        lines = ["Value driver ratings (1–5, higher = more important):"]
        for col, label in VALUE_COLS.items():
            lines.append(f"  {label}: {vd[col]}")
        add("\n".join(lines), "Value Drivers (Q15–Q17)")

        # --- Most persuasive driver (Q18) ---
        q18 = df["Q18"].value_counts(normalize=True).mul(100).round(1)
        lines = ["Most persuasive value driver (Q18):"]
        for driver, pct in q18.items():
            lines.append(f"  {driver}: {pct}%")
        add("\n".join(lines), "Most Persuasive Driver (Q18)")

        # --- Demographics overview ---
        age_dist = df["Q21"].value_counts(normalize=True).mul(100).round(1)
        income_dist = df["Q22"].value_counts(normalize=True).mul(100).round(1)
        work_dist = df["Q23"].value_counts(normalize=True).mul(100).round(1)
        lines = ["Respondent demographics:"]
        lines.append("  Age distribution:")
        for age, pct in age_dist.items():
            lines.append(f"    {age}: {pct}%")
        lines.append("  Income distribution:")
        for income, pct in income_dist.items():
            lines.append(f"    {income}: {pct}%")
        lines.append("  Work arrangement:")
        for work, pct in work_dist.items():
            lines.append(f"    {work}: {pct}%")
        add("\n".join(lines), "Demographics")

        # --- HOA status ---
        hoa = df["Q24"].value_counts(normalize=True).mul(100).round(1)
        lines = ["HOA membership status (Q24):"]
        for status, pct in hoa.items():
            lines.append(f"  {status}: {pct}%")
        add("\n".join(lines), "HOA Status (Q24)")

        # --- Outreach channels ---
        ch1 = df["Q20_1"].value_counts(normalize=True).mul(100).round(1)
        lines = ["Top outreach / discovery channels mentioned:"]
        for channel, pct in ch1.items():
            lines.append(f"  {channel}: {pct}%")
        add("\n".join(lines), "Outreach Channels (Q20)")

        return chunks

    # ------------------------------------------------------------------
    # TF-IDF index
    # ------------------------------------------------------------------

    def _build_tfidf_index(self) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        texts = [c.text for c in self._chunks]
        self._vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=15000,
        )
        self._index = self._vectorizer.fit_transform(texts)

    def _retrieve_tfidf(self, query: str, top_k: int) -> list[DocumentChunk]:
        from sklearn.metrics.pairwise import cosine_similarity
        q_vec = self._vectorizer.transform([query])
        scores = cosine_similarity(q_vec, self._index).flatten()
        top_indices = scores.argsort()[::-1][:top_k]
        return [self._chunks[i] for i in top_indices if scores[i] > 0]


# ---------------------------------------------------------------------------
# Factory (mirrors get_retriever() in rag_pipeline.py)
# ---------------------------------------------------------------------------

def get_customer_retriever(data_dir: Path) -> CustomerDataRetriever:
    """Build and return a ready CustomerDataRetriever."""
    retriever = CustomerDataRetriever(data_dir)
    retriever.load_and_build()
    return retriever
