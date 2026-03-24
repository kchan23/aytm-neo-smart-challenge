"""Shared analytics module for Neo Smart Living synthetic respondent analysis."""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

# Data lives in prototype/output/ — resolve relative to this file's location
DATA_PATH = Path(__file__).parent.parent.parent / "prototype" / "output" / "synthetic_responses.csv"

LIKERT_KEYS = {
    "Q0b": "Category Interest",
    "Q1": "Purchase Interest ($23K)",
    "Q2": "Purchase Likelihood (24mo)",
    "Q7": "Permit-Light Effect",
    "Q15": "Value: Permit-Light",
    "Q16": "Value: Install Speed",
    "Q17": "Value: Build Quality",
    "Q19": "Sponsorship Impact",
}

BARRIER_KEYS = {
    "Q5_cost": "Cost (~$23K)",
    "Q5_hoa": "HOA Restrictions",
    "Q5_permit": "Permit Uncertainty",
    "Q5_space": "Limited Space",
    "Q5_financing": "Lack of Financing",
    "Q5_quality": "Build Quality Concerns",
    "Q5_resale": "Resale Value Uncertainty",
}

CONCEPT_APPEAL = {
    "Q9a": "Home Office",
    "Q9b": "Home Office (Likelihood)",
    "Q10a": "Guest Suite / STR",
    "Q10b": "Guest Suite (Likelihood)",
    "Q11a": "Wellness Studio",
    "Q11b": "Wellness (Likelihood)",
    "Q12a": "Adventure Basecamp",
    "Q12b": "Adventure (Likelihood)",
    "Q13a": "Simplicity",
    "Q13b": "Simplicity (Likelihood)",
}

CATEGORICAL_KEYS = ["Q3", "Q6", "Q14", "Q18", "Q20_1", "Q20_2"]

DEMOGRAPHIC_KEYS = {"Q21": "Age", "Q22": "Income", "Q23": "Work Arrangement"}

ALL_NUMERIC = list(LIKERT_KEYS.keys()) + list(BARRIER_KEYS.keys()) + list(CONCEPT_APPEAL.keys()) + ["Q30"]


def load_data(path=None):
    """Read CSV and coerce Likert columns to numeric."""
    df = pd.read_csv(path or DATA_PATH)
    for col in ALL_NUMERIC:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def descriptive_likert(df, group_cols):
    """Mean, SD, median, IQR per group for all Likert variables."""
    numeric_cols = [c for c in ALL_NUMERIC if c in df.columns and c != "Q30"]
    rows = []
    for name, grp in df.groupby(group_cols):
        if isinstance(name, str):
            name = (name,)
        for col in numeric_cols:
            vals = grp[col].dropna()
            label = LIKERT_KEYS.get(col) or BARRIER_KEYS.get(col) or CONCEPT_APPEAL.get(col, col)
            row = dict(zip(group_cols if isinstance(group_cols, list) else [group_cols], name))
            row.update({
                "Variable": col, "Label": label,
                "N": len(vals), "Mean": vals.mean(), "SD": vals.std(),
                "Median": vals.median(), "IQR": vals.quantile(0.75) - vals.quantile(0.25),
            })
            rows.append(row)
    return pd.DataFrame(rows)


def descriptive_categorical(df, group_cols):
    """Value counts + percentages for categorical variables."""
    rows = []
    for col in CATEGORICAL_KEYS:
        if col not in df.columns:
            continue
        ct = df.groupby(group_cols)[col].value_counts().reset_index(name="Count")
        totals = df.groupby(group_cols)[col].count().reset_index(name="Total")
        ct = ct.merge(totals, on=group_cols if isinstance(group_cols, list) else [group_cols])
        ct["Pct"] = (ct["Count"] / ct["Total"] * 100).round(1)
        ct["Variable"] = col
        rows.append(ct)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def model_comparison_likert(df):
    """Mann-Whitney U + rank-biserial effect size for all Likert vars between models."""
    models = df["model"].unique()
    if len(models) < 2:
        return pd.DataFrame()
    m1, m2 = models[0], models[1]
    numeric_cols = [c for c in ALL_NUMERIC if c in df.columns and c != "Q30"]
    rows = []
    for col in numeric_cols:
        g1 = df.loc[df["model"] == m1, col].dropna()
        g2 = df.loc[df["model"] == m2, col].dropna()
        if len(g1) == 0 or len(g2) == 0:
            continue
        label = LIKERT_KEYS.get(col) or BARRIER_KEYS.get(col) or CONCEPT_APPEAL.get(col, col)
        stat, p = stats.mannwhitneyu(g1, g2, alternative="two-sided")
        n1, n2 = len(g1), len(g2)
        r = 1 - (2 * stat) / (n1 * n2)  # rank-biserial
        rows.append({
            "Variable": col, "Label": label,
            f"Mean ({m1})": round(g1.mean(), 3), f"Mean ({m2})": round(g2.mean(), 3),
            "U": round(stat, 1), "p": round(p, 4),
            "Effect Size (r)": round(r, 3),
            "Significant (p<.05)": p < 0.05,
        })
    return pd.DataFrame(rows)


def model_comparison_categorical(df):
    """Chi-square or Fisher exact for categorical vars between models."""
    models = df["model"].unique()
    if len(models) < 2:
        return pd.DataFrame()
    rows = []
    for col in CATEGORICAL_KEYS:
        if col not in df.columns:
            continue
        ct = pd.crosstab(df["model"], df[col])
        if ct.shape[0] < 2:
            continue
        if ct.shape == (2, 2):
            stat, p = stats.fisher_exact(ct)
            test = "Fisher"
        else:
            stat, p, _, _ = stats.chi2_contingency(ct)
            test = "Chi-square"
        rows.append({"Variable": col, "Test": test, "Statistic": round(stat, 3), "p": round(p, 4)})
    return pd.DataFrame(rows)


def barrier_heatmap_data(df):
    """Pivot segment x barrier means."""
    barrier_cols = list(BARRIER_KEYS.keys())
    hm = df.groupby("segment_name")[barrier_cols].mean()
    hm.columns = [BARRIER_KEYS[c] for c in barrier_cols]
    return hm


def segment_profiles(df):
    """Key variable means per segment."""
    key_vars = ["Q1", "Q2", "Q7", "Q15", "Q16", "Q17", "Q19"]
    key_vars = [c for c in key_vars if c in df.columns]
    prof = df.groupby("segment_name")[key_vars].mean().round(3)
    prof.columns = [LIKERT_KEYS.get(c, c) for c in key_vars]
    return prof
