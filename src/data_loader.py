"""
Adult Census Income Analysis and Prediction
Module: data_loader.py
Description: Data acquisition, loading, structural verification, and initial dataset profiling.
"""

import os
import json
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import pandas as pd
import numpy as np


def find_dataset(custom_path: Optional[str] = None) -> Path:
    """
    Locates the adult dataset CSV file across potential search paths.
    """
    if custom_path and Path(custom_path).is_file():
        return Path(custom_path)

    possible_paths = [
        Path("data/adult.csv"),
        Path("../data/adult.csv"),
        Path("EDA_Adult_Income_Project/data/adult.csv"),
        Path("../EDA_Adult_Income_Project/data/adult.csv"),
        Path("data/adult-census-income.csv"),
        Path("data/adult_income.csv"),
    ]

    for p in possible_paths:
        if p.is_file():
            return p.resolve()

    # Search recursively in data/ or workspace
    for root_dir in [Path("data"), Path(".")]:
        if root_dir.exists():
            for p in root_dir.rglob("*.csv"):
                if "adult" in p.name.lower():
                    return p.resolve()

    raise FileNotFoundError(
        "Could not find adult census income dataset CSV. "
        "Please ensure 'adult.csv' is placed inside the 'data/' folder."
    )


def load_dataset(
    data_path: Optional[str] = None,
    output_metrics_dir: Optional[str] = None
) -> Tuple[pd.DataFrame, Dict]:
    """
    Loads the Adult Census Income dataset, performs initial structural verification,
    computes descriptive statistics, and saves a dataset summary.

    Parameters
    ----------
    data_path : Optional[str]
        Path to the dataset CSV file.
    output_metrics_dir : Optional[str]
        Directory where summary metrics will be saved.

    Returns
    -------
    Tuple[pd.DataFrame, Dict]
        The loaded raw DataFrame and a metadata summary dictionary.
    """
    csv_file = find_dataset(data_path)
    print(f"[DataLoader] Loading dataset from: {csv_file}")

    df = pd.read_csv(csv_file)

    # 1. Dimensions and columns verification
    n_rows, n_cols = df.shape
    columns = list(df.columns)
    print(f"[DataLoader] Dataset Loaded Successfully: {n_rows:,} rows, {n_cols} columns.")

    if n_cols < 15:
        raise ValueError(f"Expected at least 15 columns, but found {n_cols}: {columns}")

    # Check for target column
    target_candidates = [col for col in columns if "income" in col.lower() or "salary" in col.lower()]
    if not target_candidates:
        raise ValueError("Could not find income target column in dataset.")
    target_col = target_candidates[0]

    # 2. Separate numerical and categorical features
    numerical_cols = list(df.select_dtypes(include=[np.number]).columns)
    categorical_cols = [c for c in df.select_dtypes(include=["object", "category"]).columns if c != target_col]

    # 3. Missing values and question mark placeholders inspection
    missing_counts = {}
    question_mark_counts = {}
    for col in df.columns:
        null_count = int(df[col].isnull().sum())
        q_count = int((df[col].astype(str).str.strip() == "?").sum())
        missing_counts[col] = null_count
        question_mark_counts[col] = q_count

    # 4. Duplicate rows
    duplicate_count = int(df.duplicated().sum())

    # 5. Target class distribution
    target_counts = df[target_col].astype(str).str.strip().value_counts().to_dict()
    target_proportions = (df[target_col].astype(str).str.strip().value_counts(normalize=True) * 100).round(2).to_dict()

    summary = {
        "dataset_path": str(csv_file),
        "total_rows": n_rows,
        "total_columns": n_cols,
        "columns": columns,
        "dtypes": {k: str(v) for k, v in df.dtypes.items()},
        "numerical_features": numerical_cols,
        "categorical_features": categorical_cols,
        "target_feature": target_col,
        "null_counts": missing_counts,
        "question_mark_counts": question_mark_counts,
        "duplicate_rows_count": duplicate_count,
        "target_distribution_counts": target_counts,
        "target_distribution_percentages": target_proportions,
    }

    # Print summary
    print("\n--- DATASET OVERVIEW ---")
    print(f"Dimensions: {n_rows} rows x {n_cols} columns")
    print(f"Numerical Features ({len(numerical_cols)}): {numerical_cols}")
    print(f"Categorical Features ({len(categorical_cols)}): {categorical_cols}")
    print(f"Target Feature: '{target_col}' -> {target_counts} ({target_proportions})")
    print(f"Duplicate Rows: {duplicate_count}")
    print(f"Columns with '?' Placeholders: {[c for c, count in question_mark_counts.items() if count > 0]}")

    # Save summary if metrics directory specified
    if output_metrics_dir:
        out_dir = Path(output_metrics_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        summary_file = out_dir / "dataset_summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4)
        print(f"[DataLoader] Summary saved to: {summary_file}")

    return df, summary


if __name__ == "__main__":
    df, summary = load_dataset(output_metrics_dir="outputs/metrics")
    print("\nFirst 5 rows:")
    print(df.head())
