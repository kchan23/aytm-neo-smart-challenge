"""Qualitative Interview Insights — 6-tab Streamlit page."""

import streamlit as st
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path

st.set_page_config(page_title="Interview Insights", page_icon="🎤", layout="wide")

DATA_DIR = Path(__file__).parent.parent.parent / "prototype" / "output"
TRANSCRIPT_PATH = DATA_DIR / "interview_transcripts.csv"
ANALYSIS_PATH = DATA_DIR / "interview_analysis.csv"
THEMES_PATH = DATA_DIR / "interview_themes.json"

QUESTION_KEYS = ["IQ1", "IQ2", "IQ3", "IQ4", "IQ5", "IQ6", "IQ7", "IQ8"]
QUESTION_LABELS = {
    "IQ1": "Backyard Relationship",
    "IQ2": "Unmet Home Needs",
    "IQ3": "Prior Consideration",
    "IQ4": "Lifestyle Fantasy",
    "IQ5": "Work-Life Boundaries",
    "IQ6": "Product Reaction",
    "IQ7": "Barriers & Drivers",
    "IQ8": "Social & Discovery",
}


@st.cache_data
def load_data():
    transcripts = pd.read_csv(TRANSCRIPT_PATH) if TRANSCRIPT_PATH.exists() else None
    analysis = pd.read_csv(ANALYSIS_PATH) if ANALYSIS_PATH.exists() else None
    themes = json.loads(THEMES_PATH.read_text()) if THEMES_PATH.exists() else None
    return transcripts, analysis, themes


def to_csv(df):
    return df.to_csv(index=False).encode("utf-8")


def tab_overview(df):
    st.header("Overview")

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Interviews", len(df))
    c2.metric("Models Used", df["model"].nunique())
    c3.metric("Unique Personas", df["persona_id"].nunique())

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Interviews by Model")
        st.bar_chart(df["model"].value_counts())

    with col2:
        st.subheader("Age Distribution")
        st.bar_chart(df["age"].value_counts().sort_index())

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Income Distribution")
        income_order = [
            "$50,000-$74,999", "$75,000-$99,999",
            "$100,000-$149,999", "$150,000-$199,999", "$200,000 or more",
        ]
        income_counts = df["income"].value_counts().reindex(income_order, fill_value=0)
        st.bar_chart(income_counts)

    with col4:
        st.subheader("Work Arrangement")
        st.bar_chart(df["work_arrangement"].value_counts())

    st.subheader("HOA Status")
    st.bar_chart(df["hoa_status"].value_counts())


def tab_sentiment(df):
    st.header("Sentiment Analysis")

    if "sentiment_IQ1" not in df.columns:
        st.warning("No sentiment data found. Run `python prototype/interview_analysis.py` first.")
        return

    st.subheader("Sentiment Heatmap (Respondent × Question)")
    sent_cols = [f"sentiment_{q}" for q in QUESTION_KEYS]
    heatmap_data = df.set_index("persona_id")[sent_cols].copy()
    heatmap_data.columns = [QUESTION_LABELS[q] for q in QUESTION_KEYS]

    fig, ax = plt.subplots(figsize=(12, max(6, len(df) * 0.3)))
    sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="RdYlGn", center=0,
                vmin=-1, vmax=1, ax=ax, linewidths=0.5)
    ax.set_ylabel("")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Mean Sentiment by Question")
        means = df[sent_cols].mean()
        means.index = [QUESTION_LABELS[q] for q in QUESTION_KEYS]
        fig, ax = plt.subplots(figsize=(8, 4))
        colors = ["#4CAF50" if v > 0.05 else ("#F44336" if v < -0.05 else "#9E9E9E") for v in means]
        means.plot.barh(ax=ax, color=colors)
        ax.axvline(x=0, color="black", linewidth=0.5)
        ax.set_xlabel("Mean VADER Compound Score")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("IQ6 (Product Reaction) Sentiment Distribution")
        fig, ax = plt.subplots(figsize=(8, 4))
        df["sentiment_IQ6"].hist(bins=15, ax=ax, color="steelblue", edgecolor="white")
        ax.axvline(x=0, color="red", linestyle="--", label="Neutral")
        ax.set_xlabel("Sentiment Score")
        ax.set_ylabel("Count")
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.subheader("Sentiment by Model (Box Plot + Mann-Whitney U)")
    models = df["model"].unique()
    if len(models) >= 2:
        fig, ax = plt.subplots(figsize=(10, 4))
        df.boxplot(column="sentiment_overall", by="model", ax=ax, grid=False)
        ax.set_title("")
        plt.suptitle("")
        ax.set_ylabel("Overall Sentiment")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        g1 = df.loc[df["model"] == models[0], "sentiment_overall"].dropna()
        g2 = df.loc[df["model"] == models[1], "sentiment_overall"].dropna()
        if len(g1) > 0 and len(g2) > 0:
            stat, p = stats.mannwhitneyu(g1, g2, alternative="two-sided")
            st.markdown(
                f"**Mann-Whitney U**: U={stat:.1f}, p={p:.4f} "
                f"{'(significant)' if p < 0.05 else '(not significant)'}"
            )

    st.download_button(
        "Download sentiment data",
        to_csv(df[["persona_id", "model"] + sent_cols + ["sentiment_overall"]]),
        "sentiment.csv", "text/csv",
    )


def tab_emotional_tone(df):
    st.header("Emotional Tone Assessment")

    if "primary_emotion" not in df.columns:
        st.warning("No emotion data found. Run `python prototype/interview_analysis.py` first.")
        return

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Primary Emotion Distribution")
        fig, ax = plt.subplots(figsize=(8, 5))
        df["primary_emotion"].value_counts().plot.barh(ax=ax, color="teal")
        ax.set_xlabel("Count")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("Emotion Intensity vs. Overall Sentiment")
        if "emotion_intensity" in df.columns and "sentiment_overall" in df.columns:
            fig, ax = plt.subplots(figsize=(8, 5))
            emotions = df["primary_emotion"].unique()
            ax.scatter(
                df["sentiment_overall"], df["emotion_intensity"],
                c=pd.Categorical(df["primary_emotion"]).codes, cmap="Set2",
                s=80, alpha=0.7, edgecolors="white",
            )
            ax.set_xlabel("Overall Sentiment")
            ax.set_ylabel("Emotion Intensity (1-5)")
            for i, em in enumerate(emotions):
                ax.scatter([], [], color=plt.cm.Set2(i / max(len(emotions) - 1, 1)), label=em, s=60)
            ax.legend(title="Emotion", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    st.subheader("Emotion by Age Group")
    if "age" in df.columns:
        st.dataframe(pd.crosstab(df["age"], df["primary_emotion"]), use_container_width=True)

    st.subheader("Emotion by Work Arrangement")
    if "work_arrangement" in df.columns:
        st.dataframe(pd.crosstab(df["work_arrangement"], df["primary_emotion"]), use_container_width=True)

    st.subheader("Emotion by Income")
    if "income" in df.columns:
        st.dataframe(pd.crosstab(df["income"], df["primary_emotion"]), use_container_width=True)


def tab_thematic(themes):
    st.header("Thematic Analysis")

    if not themes:
        st.warning("No themes data found. Run `python prototype/interview_analysis.py` first.")
        return

    lda = themes.get("lda_topics", {})
    if lda:
        st.subheader(
            f"LDA Topics (k={lda.get('num_topics', '?')}, coherence={lda.get('coherence_score', '?')})"
        )
        for topic in lda.get("topics", []):
            label = topic.get("label", f"Topic {topic['topic_id']}")
            keywords = ", ".join(topic.get("keywords", []))
            st.markdown(f"**{label}**: {keywords}")

        topic_labels = [t.get("label", f"Topic {t['topic_id']}") for t in lda.get("topics", [])]
        if topic_labels:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.barh(topic_labels, range(len(topic_labels), 0, -1), color="steelblue")
            ax.set_xlabel("Relative Prevalence")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    llm_themes = themes.get("llm_themes", [])
    if llm_themes:
        st.subheader(f"LLM-Identified Themes ({len(llm_themes)})")
        for theme in llm_themes:
            with st.expander(f"{theme['theme_name']} (n={theme.get('frequency', '?')})"):
                st.write(theme.get("description", ""))
                for q in theme.get("supporting_quotes", []):
                    st.markdown(f"> *\"{q['quote']}\"* — {q['respondent_id']}")


def tab_segment_discovery(themes):
    st.header("Segment Discovery")

    if not themes:
        st.warning("No themes data found. Run `python prototype/interview_analysis.py` first.")
        return

    segments = themes.get("segment_suggestions", [])
    if not segments:
        st.info("No segment suggestions found in themes data.")
        return

    st.subheader("Emergent Segments from Interview Data")
    for seg in segments:
        with st.expander(f"{seg['segment_name']} (est. {seg.get('estimated_size', '?')})"):
            st.write(seg.get("description", ""))
            st.markdown(f"**Key Driver:** {seg.get('key_driver', 'N/A')}")
            st.markdown(f"**Primary Barrier:** {seg.get('primary_barrier', 'N/A')}")
            reps = seg.get("representative_respondents", [])
            if reps:
                st.markdown(f"**Representative Respondents:** {', '.join(reps)}")

    mapping = themes.get("existing_segment_mapping", {})
    if mapping:
        st.subheader("Mapping to Existing Survey Segments")
        existing_segments = [
            "Remote Professional", "Active Adventurer", "Wellness Seeker",
            "Property Maximizer", "Budget-Conscious DIYer",
        ]
        map_data = [
            {
                "Interview Segment": k,
                "Survey Segment": v,
                "Alignment": "Direct Match" if v in existing_segments else "New",
            }
            for k, v in mapping.items()
        ]
        st.dataframe(pd.DataFrame(map_data), use_container_width=True)

        matched = sum(1 for v in mapping.values() if v in existing_segments)
        st.markdown(f"""
- **{matched}/{len(mapping)}** interview segments map directly to existing survey segments
- Interview data {'validates' if matched >= len(mapping) * 0.6 else 'partially validates'} the predetermined segmentation
- Consider {'refining segment boundaries' if matched < len(mapping) else 'proceeding with current segments'} based on qualitative nuances
""")


def tab_transcripts(df):
    st.header("Full Transcripts")

    col1, col2, col3 = st.columns(3)
    with col1:
        model_filter = st.multiselect("Model", df["model"].unique(), default=list(df["model"].unique()))
    with col2:
        age_filter = st.multiselect("Age", sorted(df["age"].unique()), default=list(df["age"].unique()))
    with col3:
        search = st.text_input("Search text (any question)")

    filtered = df[df["model"].isin(model_filter) & df["age"].isin(age_filter)]

    if search:
        mask = pd.Series(False, index=filtered.index)
        for q in QUESTION_KEYS + ["additional_thoughts"]:
            if q in filtered.columns:
                mask |= filtered[q].astype(str).str.contains(search, case=False, na=False)
        filtered = filtered[mask]

    st.markdown(f"**Showing {len(filtered)} of {len(df)} interviews**")

    for _, row in filtered.iterrows():
        with st.expander(
            f"{row['persona_id']} — {row['persona_name']} ({row['model']}) | {row['age']}, {row['income']}"
        ):
            st.markdown(f"**Home:** {row.get('home_situation', 'N/A')}")
            st.markdown(f"**Household:** {row.get('household', 'N/A')}")
            st.markdown(f"**Lifestyle:** {row.get('lifestyle_note', 'N/A')}")
            st.markdown(f"**HOA:** {row.get('hoa_status', 'N/A')}")
            st.markdown("---")
            for q in QUESTION_KEYS:
                label = QUESTION_LABELS.get(q, q)
                st.markdown(f"**{q} ({label}):**")
                st.write(row.get(q, "[No response]"))
            if pd.notna(row.get("additional_thoughts")):
                st.markdown("**Additional Thoughts:**")
                st.write(row["additional_thoughts"])


# --- Main ---
st.title("🎤 Qualitative Interview Insights")

transcripts, analysis, themes = load_data()

if transcripts is None:
    st.error(
        "No interview data found.\n\n"
        "Run `python prototype/generate_test_interviews.py` (no API key needed) "
        "or `python prototype/synthetic_interviews.py` first."
    )
    st.stop()

# Use analysis data if available (has sentiment + emotion columns), otherwise raw transcripts
df = analysis if analysis is not None else transcripts

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Overview",
    "Sentiment",
    "Emotional Tone",
    "Thematic Analysis",
    "Segment Discovery",
    "Full Transcripts",
])

with tab1:
    tab_overview(df)
with tab2:
    tab_sentiment(df)
with tab3:
    tab_emotional_tone(df)
with tab4:
    tab_thematic(themes)
with tab5:
    tab_segment_discovery(themes)
with tab6:
    tab_transcripts(df)
