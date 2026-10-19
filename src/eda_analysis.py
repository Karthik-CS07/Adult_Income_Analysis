"""
Adult Census Income Analysis and Prediction
Module: eda_analysis.py
Description: Statistical analysis (univariate, bivariate, multivariate), distributions,
             correlations, and category income proportions.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd
import numpy as np


def run_comprehensive_eda(
    df: pd.DataFrame,
    metrics_dir: str = "outputs/metrics"
) -> Dict[str, Any]:
    """
    Performs full exploratory data analysis on the clean adult dataset:
    - Numerical summary (Mean, std, median, min, max, Q1, Q3, IQR, Skewness)
    - Categorical summary (Unique count, top mode, mode frequency, mode percentage)
    - Missing values analysis (with '?' converted to NaN)
    - Numerical correlation matrix
    - Bivariate income proportions across demographic categories

    Saves all resulting tables to metrics_dir as CSVs.
    """
    out_dir = Path(metrics_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Clean whitespace and replace '?' for accurate statistics
    working_df = df.copy()
    for col in working_df.select_dtypes(include=["object"]).columns:
        working_df[col] = working_df[col].astype(str).str.strip()
    working_df = working_df.replace("?", np.nan)

    # Identify columns
    target_col = [col for col in working_df.columns if "income" in col.lower()][0]
    num_cols = list(working_df.select_dtypes(include=[np.number]).columns)
    # Remove binary target if already encoded
    num_cols = [c for c in num_cols if c != "target"]
    cat_cols = [c for c in working_df.select_dtypes(include=["object", "category"]).columns if c != target_col]

    # Create binary indicator for computing rates: 1 if >50K else 0
    working_df["is_high_income"] = working_df[target_col].apply(lambda x: 1 if ">50k" in str(x).lower() else 0)

    # -------------------------------------------------------------
    # A. NUMERICAL DESCRIPTIVE STATISTICS
    # -------------------------------------------------------------
    num_stats = []
    for col in num_cols:
        s = working_df[col].dropna()
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        skew = s.skew()
        num_stats.append({
            "Feature": col,
            "Count": len(s),
            "Mean": round(s.mean(), 2),
            "Std": round(s.std(), 2),
            "Min": s.min(),
            "25% (Q1)": q1,
            "50% (Median)": s.median(),
            "75% (Q3)": q3,
            "Max": s.max(),
            "IQR": iqr,
            "Skewness": round(skew, 2),
        })
    num_summary_df = pd.DataFrame(num_stats)
    num_summary_df.to_csv(out_dir / "numerical_summary.csv", index=False)
    print(f"[EDA] Numerical summary saved: {len(num_stats)} features.")

    # -------------------------------------------------------------
    # B. CATEGORICAL SUMMARY
    # -------------------------------------------------------------
    cat_stats = []
    for col in cat_cols:
        s = working_df[col].dropna()
        top_cat = s.mode()[0] if not s.empty else "N/A"
        top_freq = s.value_counts().iloc[0] if not s.empty else 0
        top_pct = round((top_freq / len(s)) * 100, 2) if not s.empty else 0.0
        cat_stats.append({
            "Feature": col,
            "Unique Count": s.nunique(),
            "Total Non-Null": len(s),
            "Most Frequent": top_cat,
            "Top Freq Count": top_freq,
            "Top Freq Pct (%)": top_pct,
        })
    cat_summary_df = pd.DataFrame(cat_stats)
    cat_summary_df.to_csv(out_dir / "categorical_summary.csv", index=False)
    print(f"[EDA] Categorical summary saved: {len(cat_stats)} features.")

    # -------------------------------------------------------------
    # C. MISSING VALUES SUMMARY
    # -------------------------------------------------------------
    missing_records = []
    total_len = len(working_df)
    for col in working_df.columns:
        if col in ["is_high_income", "target"]:
            continue
        missing_count = int(working_df[col].isnull().sum())
        missing_pct = round((missing_count / total_len) * 100, 2)
        missing_records.append({
            "Feature": col,
            "Missing Count": missing_count,
            "Missing Pct (%)": missing_pct,
            "Data Type": str(working_df[col].dtype),
        })
    missing_df = pd.DataFrame(missing_records)
    missing_df.to_csv(out_dir / "missing_values_summary.csv", index=False)
    print(f"[EDA] Missing values summary saved.")

    # -------------------------------------------------------------
    # D. NUMERICAL CORRELATION MATRIX
    # -------------------------------------------------------------
    corr_cols = num_cols + ["is_high_income"]
    corr_matrix = working_df[corr_cols].corr().round(4)
    corr_matrix.to_csv(out_dir / "correlation_matrix.csv")
    print(f"[EDA] Correlation matrix saved.")

    # -------------------------------------------------------------
    # E. INCOME PROPORTIONS ACROSS DEMOGRAPHIC CATEGORIES
    # -------------------------------------------------------------
    proportions_dict = {}
    for col in ["education", "workclass", "occupation", "sex", "marital.status", "relationship", "race"]:
        if col in working_df.columns:
            group_summary = working_df.groupby(col).agg(
                Total_Count=("is_high_income", "count"),
                High_Income_Count=("is_high_income", "sum"),
                High_Income_Rate=("is_high_income", "mean"),
            ).reset_index()
            group_summary["Low_Income_Count"] = group_summary["Total_Count"] - group_summary["High_Income_Count"]
            group_summary["High_Income_Pct"] = (group_summary["High_Income_Rate"] * 100).round(2)
            group_summary["Low_Income_Pct"] = (100 - group_summary["High_Income_Pct"]).round(2)
            group_summary = group_summary.sort_values(by="High_Income_Rate", ascending=False)
            group_summary.to_csv(out_dir / f"income_proportions_{col.replace('.', '_')}.csv", index=False)
            proportions_dict[col] = group_summary

    print(f"[EDA] Category income proportions saved.")

    return {
        "numerical_summary": num_summary_df,
        "categorical_summary": cat_summary_df,
        "missing_summary": missing_df,
        "correlation_matrix": corr_matrix,
        "proportions": proportions_dict,
    }


if __name__ == "__main__":
    from data_loader import load_dataset
    df, _ = load_dataset()
    eda_results = run_comprehensive_eda(df)
    print("EDA analysis completed.")
