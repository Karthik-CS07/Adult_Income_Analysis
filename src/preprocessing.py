"""
Adult Census Income Analysis and Prediction
Module: preprocessing.py
Description: Data cleaning, missing value imputation, encoding, scaling, and stratified splitting
             strictly preventing data leakage.
"""

import os
import json
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Any
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def clean_raw_data(df: pd.DataFrame, drop_duplicates: bool = True) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Cleans the raw adult census income dataframe:
    - Replaces '?' placeholders with np.nan
    - Strips leading/trailing whitespaces from object columns
    - Drops duplicate rows and logs statistics
    - Encodes target column into binary (0: <=50K, 1: >50K)

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, Any]]
        Cleaned dataframe and a dictionary of cleaning logs.
    """
    cleaned_df = df.copy()

    # 1. Strip whitespace from string/object columns
    for col in cleaned_df.select_dtypes(include=["object"]).columns:
        cleaned_df[col] = cleaned_df[col].astype(str).str.strip()

    # 2. Replace '?' with np.nan
    cleaned_df = cleaned_df.replace("?", np.nan)

    # 3. Duplicate handling
    initial_rows = len(cleaned_df)
    dup_count = int(cleaned_df.duplicated().sum())
    if drop_duplicates and dup_count > 0:
        cleaned_df = cleaned_df.drop_duplicates().reset_index(drop=True)
    post_dedup_rows = len(cleaned_df)

    # 4. Standardize column names (remove spaces, replace dots with underscores if desired, or keep as is)
    # Target column identification
    target_col = [col for col in cleaned_df.columns if "income" in col.lower()][0]

    # Target normalization: 0 for <=50K, 1 for >50K
    target_series = cleaned_df[target_col].astype(str).str.strip().str.rstrip(".")
    cleaned_df["target"] = target_series.apply(lambda x: 1 if ">50k" in x.lower() else 0)

    # Missing counts after replacing '?'
    missing_after_clean = cleaned_df.isnull().sum().to_dict()

    cleaning_logs = {
        "initial_rows": initial_rows,
        "duplicates_removed": dup_count if drop_duplicates else 0,
        "cleaned_rows": post_dedup_rows,
        "target_distribution_clean": cleaned_df["target"].value_counts().to_dict(),
        "target_distribution_percent": (cleaned_df["target"].value_counts(normalize=True) * 100).round(2).to_dict(),
        "missing_values_per_column": {k: int(v) for k, v in missing_after_clean.items() if v > 0},
    }

    print(f"[Preprocessing] Cleaned dataset: {post_dedup_rows:,} rows (dropped {dup_count} duplicates).")
    print(f"[Preprocessing] Target distribution: 0 (<=50K): {cleaning_logs['target_distribution_clean'].get(0)}, 1 (>50K): {cleaning_logs['target_distribution_clean'].get(1)}")
    return cleaned_df, cleaning_logs


def compute_outlier_summary(df: pd.DataFrame, numerical_cols: List[str]) -> pd.DataFrame:
    """
    Computes IQR-based outlier statistics for numerical features.
    """
    records = []
    for col in numerical_cols:
        series = df[col].dropna()
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = series[(series < lower_bound) | (series > upper_bound)]
        records.append({
            "Feature": col,
            "Count": len(series),
            "Mean": round(series.mean(), 2),
            "Std": round(series.std(), 2),
            "Min": series.min(),
            "Q1 (25%)": q1,
            "Median (50%)": series.median(),
            "Q3 (75%)": q3,
            "Max": series.max(),
            "IQR": iqr,
            "Lower Bound": lower_bound,
            "Upper Bound": upper_bound,
            "Outlier Count": len(outliers),
            "Outlier Pct (%)": round((len(outliers) / len(series)) * 100, 2),
        })
    return pd.DataFrame(records)


def prepare_data_pipeline(
    df: pd.DataFrame,
    outputs_dir: str = "outputs",
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Executes end-to-end data preparation strictly preventing data leakage:
    1. Splits dataset into Train (70%), Validation (15%), and Test (15%) stratified by target.
    2. Fits numerical Imputer (median) + StandardScaler on Train ONLY.
    3. Fits categorical Imputer (most_frequent) + OneHotEncoder on Train ONLY.
    4. Transforms Train, Validation, and Test sets.
    5. Saves artifacts and preprocessed arrays.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing preprocessed arrays, feature names, and metadata.
    """
    outputs_path = Path(outputs_dir)
    processed_dir = outputs_path / "processed"
    metrics_dir = outputs_path / "metrics"
    processed_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # 1. Clean data
    clean_df, cleaning_logs = clean_raw_data(df, drop_duplicates=True)

    # 2. Identify feature types
    target_col = [col for col in clean_df.columns if "income" in col.lower()][0]
    predictor_cols = [c for c in clean_df.columns if c not in [target_col, "target"]]
    
    numerical_cols = list(clean_df[predictor_cols].select_dtypes(include=[np.number]).columns)
    categorical_cols = list(clean_df[predictor_cols].select_dtypes(include=["object", "category"]).columns)

    # 3. Compute outlier summary
    outlier_df = compute_outlier_summary(clean_df, numerical_cols)
    outlier_file = metrics_dir / "outlier_summary.csv"
    outlier_df.to_csv(outlier_file, index=False)
    print(f"[Preprocessing] Outlier summary saved to: {outlier_file}")

    # 4. Stratified 70 : 15 : 15 Split
    X = clean_df[predictor_cols]
    y = clean_df["target"].values

    # Step A: 70% Train, 30% Temp (Val + Test)
    X_train_df, X_temp_df, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=random_state, stratify=y
    )

    # Step B: Split Temp 50:50 -> 15% Val, 15% Test
    X_val_df, X_test_df, y_val, y_test = train_test_split(
        X_temp_df, y_temp, test_size=0.50, random_state=random_state, stratify=y_temp
    )

    print(f"[Preprocessing] Dataset Split Sizes:")
    print(f"  Train:      {len(X_train_df):,} rows ({len(X_train_df)/len(clean_df)*100:.1f}%) | Positives: {np.mean(y_train)*100:.2f}%")
    print(f"  Validation: {len(X_val_df):,} rows ({len(X_val_df)/len(clean_df)*100:.1f}%) | Positives: {np.mean(y_val)*100:.2f}%")
    print(f"  Test:       {len(X_test_df):,} rows ({len(X_test_df)/len(clean_df)*100:.1f}%) | Positives: {np.mean(y_test)*100:.2f}%")

    # 5. Build Preprocessor Pipelines
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, numerical_cols),
            ("cat", cat_pipeline, categorical_cols),
        ]
    )

    # 6. Fit ONLY on X_train to prevent leakage!
    print("[Preprocessing] Fitting ColumnTransformer strictly on Training set...")
    preprocessor.fit(X_train_df)

    # 7. Transform Train, Validation, and Test
    X_train_proc = preprocessor.transform(X_train_df)
    X_val_proc = preprocessor.transform(X_val_df)
    X_test_proc = preprocessor.transform(X_test_df)

    # 8. Extract feature names
    cat_encoder = preprocessor.named_transformers_["cat"].named_steps["encoder"]
    encoded_cat_features = list(cat_encoder.get_feature_names_out(categorical_cols))
    all_feature_names = numerical_cols + encoded_cat_features

    print(f"[Preprocessing] Processed Feature Count: {len(all_feature_names)} features.")

    # 9. Save preprocessed arrays
    np.save(processed_dir / "X_train.npy", X_train_proc)
    np.save(processed_dir / "X_val.npy", X_val_proc)
    np.save(processed_dir / "X_test.npy", X_test_proc)
    np.save(processed_dir / "y_train.npy", y_train)
    np.save(processed_dir / "y_val.npy", y_val)
    np.save(processed_dir / "y_test.npy", y_test)

    metadata = {
        "num_train_samples": int(len(X_train_proc)),
        "num_val_samples": int(len(X_val_proc)),
        "num_test_samples": int(len(X_test_proc)),
        "num_features": int(X_train_proc.shape[1]),
        "numerical_features": numerical_cols,
        "categorical_features": categorical_cols,
        "feature_names": all_feature_names,
        "cleaning_logs": cleaning_logs,
    }

    with open(processed_dir / "preprocessor_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    return {
        "X_train": X_train_proc,
        "X_val": X_val_proc,
        "X_test": X_test_proc,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "feature_names": all_feature_names,
        "numerical_cols": numerical_cols,
        "categorical_cols": categorical_cols,
        "clean_df": clean_df,
        "preprocessor": preprocessor,
        "metadata": metadata,
    }


if __name__ == "__main__":
    from data_loader import load_dataset
    df, _ = load_dataset()
    data_dict = prepare_data_pipeline(df)
    print("Preprocessing completed successfully!")
