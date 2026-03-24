"""Survey Analytics — 7-tab Streamlit page."""

import sys
from pathlib import Path

# Ensure utils/ is importable when running as a Streamlit page
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from utils.analytics import (
    load_data as _load_data, model_comparison_likert, barrier_heatmap_data,
    LIKERT_KEYS, BARRIER_KEYS, CONCEPT_APPEAL, DEMOGRAPHIC_KEYS, DATA_PATH,
)

st.set_page_config(page_title="Survey Analytics", page_icon="📊", layout="wide")


def to_csv(df):
    return df.to_csv(index=False).encode("utf-8")


@st.cache_data
def load_data():
    return _load_data()


def tab_overview(df):
    st.header("Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Respondents", len(df))
    q30_pass = (df["Q30"] == 3).sum()
    col2.metric("Q30 Attention Pass", f"{q30_pass}/{len(df)}")
    col3.metric("Models", df["model"].nunique())

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Respondents by Model")
        st.bar_chart(df["model"].value_counts())
    with c2:
        st.subheader("Respondents by Segment")
        st.bar_chart(df["segment_name"].value_counts())

    st.subheader("Model × Segment Cross-Tab")
    ct = pd.crosstab(df["segment_name"], df["model"])
    st.dataframe(ct, use_container_width=True)
    st.download_button("Download cross-tab CSV", to_csv(ct.reset_index()), "crosstab.csv", "text/csv")


def tab_purchase_interest(df):
    st.header("Purchase Interest (RQ1)")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Q1: Purchase Interest by Segment")
        fig, ax = plt.subplots(figsize=(8, 4))
        df.boxplot(column="Q1", by="segment_name", ax=ax, grid=False)
        ax.set_title("")
        plt.suptitle("")
        ax.set_xlabel("Segment")
        ax.set_ylabel("Interest (1-5)")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with c2:
        st.subheader("Q2: Purchase Likelihood by Segment")
        fig, ax = plt.subplots(figsize=(8, 4))
        df.boxplot(column="Q2", by="segment_name", ax=ax, grid=False)
        ax.set_title("")
        plt.suptitle("")
        ax.set_xlabel("Segment")
        ax.set_ylabel("Likelihood (1-5)")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.subheader("Cross-Tab: Q1 Mean by Work Arrangement (RQ5)")
    if "Q23" in df.columns:
        ct = df.groupby("Q23")[["Q1", "Q2"]].mean().round(2)
        st.dataframe(ct, use_container_width=True)

    st.subheader("Cross-Tab: Q1 Mean by Income (RQ6)")
    if "Q22" in df.columns:
        ct = df.groupby("Q22")[["Q1", "Q2"]].mean().round(2)
        st.dataframe(ct, use_container_width=True)

    dl = df.groupby(["segment_name", "model"])[["Q1", "Q2"]].describe().round(2).reset_index()
    st.download_button("Download purchase interest data", to_csv(dl), "purchase_interest.csv", "text/csv")


def tab_use_barriers(df):
    st.header("Use Case & Barriers (RQ2-4)")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Q3: Primary Intended Use")
        fig, ax = plt.subplots(figsize=(8, 5))
        df["Q3"].value_counts().plot.barh(ax=ax, color="steelblue")
        ax.set_xlabel("Count")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with c2:
        st.subheader("Q7: Permit-Light Effect")
        fig, ax = plt.subplots(figsize=(8, 5))
        df["Q7"].value_counts().sort_index().plot.bar(ax=ax, color="teal")
        ax.set_xlabel("Rating (1=Decrease, 5=Greatly Increase)")
        ax.set_ylabel("Count")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.subheader("Q5: Barrier Severity Heatmap (Mean by Segment)")
    heatmap_data = barrier_heatmap_data(df)
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="YlOrRd", ax=ax, vmin=1, vmax=5)
    ax.set_ylabel("")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.subheader("Q6: Single Greatest Barrier")
    st.bar_chart(df["Q6"].value_counts())

    st.download_button("Download barrier data", to_csv(heatmap_data.reset_index()), "barriers.csv", "text/csv")


def tab_concept_test(df):
    st.header("Positioning Concept Test (RQ9)")

    appeal_cols = ["Q9a", "Q10a", "Q11a", "Q12a", "Q13a"]
    likelihood_cols = ["Q9b", "Q10b", "Q11b", "Q12b", "Q13b"]
    concept_names = ["Home Office", "Guest Suite/STR", "Wellness Studio", "Adventure", "Simplicity"]

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Concept Appeal (Mean)")
        means = df[appeal_cols].mean()
        means.index = concept_names
        fig, ax = plt.subplots(figsize=(8, 4))
        means.plot.bar(ax=ax, color="steelblue")
        ax.set_ylabel("Mean Appeal (1-5)")
        ax.set_ylim(1, 5)
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with c2:
        st.subheader("Concept Purchase Likelihood (Mean)")
        means = df[likelihood_cols].mean()
        means.index = concept_names
        fig, ax = plt.subplots(figsize=(8, 4))
        means.plot.bar(ax=ax, color="darkorange")
        ax.set_ylabel("Mean Likelihood (1-5)")
        ax.set_ylim(1, 5)
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.subheader("Concept Appeal by Segment")
    appeal_by_seg = df.groupby("segment_name")[appeal_cols].mean()
    appeal_by_seg.columns = concept_names
    fig, ax = plt.subplots(figsize=(10, 5))
    appeal_by_seg.plot.bar(ax=ax)
    ax.set_ylabel("Mean Appeal (1-5)")
    ax.set_ylim(1, 5)
    ax.legend(title="Concept", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.subheader("Q14: Overall Concept Preference")
    fig, ax = plt.subplots(figsize=(8, 5))
    df["Q14"].value_counts().plot.pie(ax=ax, autopct="%1.0f%%", startangle=90)
    ax.set_ylabel("")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.download_button("Download concept data", to_csv(appeal_by_seg.reset_index()), "concepts.csv", "text/csv")


def tab_value_sponsorship(df):
    st.header("Value Drivers & Sponsorship")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Q15-Q17: Value Driver Ratings")
        vd = df[["Q15", "Q16", "Q17"]].mean()
        vd.index = ["Permit-Light", "Install Speed", "Build Quality"]
        fig, ax = plt.subplots(figsize=(6, 4))
        vd.plot.bar(ax=ax, color=["#2196F3", "#4CAF50", "#FF9800"])
        ax.set_ylabel("Mean Rating (1-5)")
        ax.set_ylim(1, 5)
        plt.xticks(rotation=0)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with c2:
        st.subheader("Q18: Most Persuasive Value Driver")
        fig, ax = plt.subplots(figsize=(6, 4))
        df["Q18"].value_counts().plot.pie(ax=ax, autopct="%1.0f%%", startangle=90)
        ax.set_ylabel("")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    c3, c4 = st.columns(2)
    with c3:
        st.subheader("Q19: Sponsorship Impact")
        fig, ax = plt.subplots(figsize=(6, 4))
        df["Q19"].value_counts().sort_index().plot.bar(ax=ax, color="teal")
        ax.set_xlabel("Rating (1=Decrease a lot, 5=Increase a lot)")
        ax.set_ylabel("Count")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with c4:
        st.subheader("Q20: Most Effective Outreach Channels")
        channels = pd.concat([df["Q20_1"], df["Q20_2"]]).dropna().replace("", np.nan).dropna()
        fig, ax = plt.subplots(figsize=(8, 4))
        channels.value_counts().plot.barh(ax=ax, color="steelblue")
        ax.set_xlabel("Mentions")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    vd_data = df.groupby(["segment_name", "model"])[["Q15", "Q16", "Q17", "Q19"]].mean().round(2).reset_index()
    st.download_button("Download value driver data", to_csv(vd_data), "value_drivers.csv", "text/csv")


def tab_model_comparison(df):
    st.header("Model Comparison")

    models = df["model"].unique()
    if len(models) < 2:
        st.warning("Need at least 2 models for comparison.")
        return

    compare_vars = ["Q1", "Q2", "Q7", "Q15", "Q16", "Q17", "Q19"]
    compare_labels = [
        "Purchase Interest", "Purchase Likelihood", "Permit-Light Effect",
        "Value: Permit-Light", "Value: Install Speed", "Value: Build Quality",
        "Sponsorship Impact",
    ]

    st.subheader("Mean Comparison (Paired Bar Chart)")
    means = df.groupby("model")[compare_vars].mean()
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(compare_vars))
    width = 0.35
    for i, model in enumerate(models):
        vals = means.loc[model, compare_vars].values
        ax.bar(x + i * width, vals, width, label=model)
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(compare_labels, rotation=30, ha="right")
    ax.set_ylabel("Mean Rating (1-5)")
    ax.set_ylim(1, 5)
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.subheader("Mann-Whitney U Tests (with Effect Size)")
    mc = model_comparison_likert(df)
    st.dataframe(mc, use_container_width=True)
    st.caption(
        "Effect size r: rank-biserial correlation (r = 1 − 2U/n₁n₂). "
        "|r| < 0.3 small, 0.3-0.5 medium, > 0.5 large."
    )

    st.subheader("Segment Profile Radar Chart")
    radar_vars = ["Q1", "Q2", "Q7", "Q15", "Q16", "Q17"]
    radar_labels = ["Purchase Int.", "Purchase Lik.", "Permit-Light", "V: Permit", "V: Speed", "V: Quality"]
    segments = df["segment_name"].unique()

    fig, axes = plt.subplots(
        1, len(segments), figsize=(4 * len(segments), 4),
        subplot_kw=dict(projection="polar"),
    )
    if len(segments) == 1:
        axes = [axes]

    angles = np.linspace(0, 2 * np.pi, len(radar_vars), endpoint=False).tolist()
    angles += angles[:1]

    for ax, seg in zip(axes, segments):
        ax.set_title(seg, size=10, pad=15)
        ax.set_ylim(1, 5)
        ax.set_yticks([1, 2, 3, 4, 5])
        ax.set_yticklabels(["1", "2", "3", "4", "5"], size=7)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(radar_labels, size=7)
        for model in models:
            subset = df[(df["model"] == model) & (df["segment_name"] == seg)]
            vals = subset[radar_vars].mean().tolist()
            vals += vals[:1]
            ax.plot(angles, vals, "o-", linewidth=1.5, label=model, markersize=3)
            ax.fill(angles, vals, alpha=0.1)
        ax.legend(loc="upper right", fontsize=7, bbox_to_anchor=(1.3, 1.1))

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.download_button("Download model comparison CSV", to_csv(mc), "model_comparison.csv", "text/csv")


def tab_demographics(df):
    st.header("Demographics")

    for col, label in DEMOGRAPHIC_KEYS.items():
        if col not in df.columns:
            continue
        st.subheader(f"{label} ({col})")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**By Model**")
            st.dataframe(pd.crosstab(df[col], df["model"]), use_container_width=True)
        with c2:
            st.markdown("**By Segment**")
            ct_seg = pd.crosstab(df[col], df["segment_name"])
            st.dataframe(ct_seg, use_container_width=True)

        fig, ax = plt.subplots(figsize=(10, 4))
        ct_seg.plot.bar(ax=ax)
        ax.set_title(f"{label} Distribution by Segment")
        ax.set_ylabel("Count")
        ax.legend(title="Segment", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.subheader("Mean Purchase Interest by Demographics")
    demo_summary = []
    for col, label in DEMOGRAPHIC_KEYS.items():
        if col not in df.columns:
            continue
        grp = df.groupby(col)[["Q1", "Q2"]].mean().round(2).reset_index().rename(columns={col: "Category"})
        grp.insert(0, "Demographic", label)
        demo_summary.append(grp)
    if demo_summary:
        demo_df = pd.concat(demo_summary, ignore_index=True)
        st.dataframe(demo_df, use_container_width=True)
        st.download_button("Download demographics CSV", to_csv(demo_df), "demographics.csv", "text/csv")


# --- Main ---
st.title("📊 Survey Analytics")

if not DATA_PATH.exists():
    st.error(
        f"Data file not found: `{DATA_PATH}`\n\n"
        "Run `python prototype/generate_test_data.py` (no API key needed) "
        "or `python prototype/synthetic_respondents.py` first."
    )
    st.stop()

df_full = load_data()

# --- Sidebar Filters ---
st.sidebar.header("Filters")
all_models = sorted(df_full["model"].unique())
all_segments = sorted(df_full["segment_name"].unique())

selected_models = st.sidebar.multiselect("Model", all_models, default=all_models)
selected_segments = st.sidebar.multiselect("Segment", all_segments, default=all_segments)

df = df_full[df_full["model"].isin(selected_models) & df_full["segment_name"].isin(selected_segments)]
st.sidebar.metric("Filtered Rows", len(df))

if df.empty:
    st.warning("No data matches the current filters.")
    st.stop()

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Overview",
    "Purchase Interest (RQ1)",
    "Use Case & Barriers (RQ2-4)",
    "Concept Test (RQ9)",
    "Value Drivers & Sponsorship",
    "Model Comparison",
    "Demographics",
])

with tab1:
    tab_overview(df)
with tab2:
    tab_purchase_interest(df)
with tab3:
    tab_use_barriers(df)
with tab4:
    tab_concept_test(df)
with tab5:
    tab_value_sponsorship(df)
with tab6:
    tab_model_comparison(df)
with tab7:
    tab_demographics(df)
