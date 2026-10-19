"""
Adult Census Income Analysis and Prediction
Module: evaluation.py
Description: Evaluates the trained deep learning model on the held-out test set,
             calculates comprehensive classification metrics, generates the 4 required
             evaluation charts, and computes permutation feature importance.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
)
from tensorflow import keras


def setup_eval_plot_style():
    sns.set_theme(style="whitegrid")
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["savefig.bbox"] = "tight"


def evaluate_test_set(
    model: keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    metrics_dir: str = "outputs/metrics"
) -> Dict[str, Any]:
    """
    Evaluates the model strictly on the held-out test set.
    Computes accuracy, precision, recall, f1, specificity, ROC-AUC,
    confusion matrix, and classification report.
    """
    metrics_path = Path(metrics_dir)
    metrics_path.mkdir(parents=True, exist_ok=True)

    # 1. Predictions
    y_prob = model.predict(X_test, batch_size=256, verbose=0).ravel()
    y_pred = (y_prob >= 0.50).astype(int)

    # 2. Metric Calculations
    acc = float(accuracy_score(y_test, y_pred))
    prec_1 = float(precision_score(y_test, y_pred, pos_label=1))
    rec_1 = float(recall_score(y_test, y_pred, pos_label=1))
    f1_1 = float(f1_score(y_test, y_pred, pos_label=1))

    prec_0 = float(precision_score(y_test, y_pred, pos_label=0))
    rec_0 = float(recall_score(y_test, y_pred, pos_label=0))
    f1_0 = float(f1_score(y_test, y_pred, pos_label=0))

    macro_f1 = float(f1_score(y_test, y_pred, average="macro"))
    weighted_f1 = float(f1_score(y_test, y_pred, average="weighted"))
    auc = float(roc_auc_score(y_test, y_prob))

    # Confusion matrix breakdown
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    report_dict = classification_report(y_test, y_pred, target_names=["<=50K (0)", ">50K (1)"], output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    report_df.to_csv(metrics_path / "test_classification_report.csv")

    metrics_summary = {
        "test_sample_count": len(y_test),
        "accuracy": round(acc, 4),
        "roc_auc": round(auc, 4),
        "class_1_positive_high_income": {
            "precision": round(prec_1, 4),
            "recall": round(rec_1, 4),
            "f1_score": round(f1_1, 4),
        },
        "class_0_negative_low_income": {
            "precision": round(prec_0, 4),
            "recall": round(rec_0, 4),
            "f1_score": round(f1_0, 4),
        },
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
            "specificity": round(specificity, 4),
        },
    }

    # Save metrics JSON
    with open(metrics_path / "model_evaluation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=4)

    print("\n--- TEST SET EVALUATION RESULTS ---")
    print(f"Accuracy:    {acc:.4f} ({acc*100:.2f}%)")
    print(f"ROC-AUC:     {auc:.4f}")
    print(f"Class >50K:  Precision={prec_1:.4f}, Recall={rec_1:.4f}, F1={f1_1:.4f}")
    print(f"Class <=50K: Precision={prec_0:.4f}, Recall={rec_0:.4f}, F1={f1_0:.4f}")
    print(f"Macro F1:    {macro_f1:.4f} | Specificity: {specificity:.4f}")
    print(f"Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")

    return {
        "metrics": metrics_summary,
        "y_prob": y_prob,
        "y_pred": y_pred,
        "cm": cm,
    }


def plot_evaluation_charts(
    history: Dict[str, List[float]],
    y_test: np.ndarray,
    y_prob: np.ndarray,
    y_pred: np.ndarray,
    figures_dir: str = "outputs/figures"
):
    """
    Generates and saves the 4 required model evaluation figures:
    1. Training vs Validation Loss by Epoch
    2. Training vs Validation Accuracy by Epoch
    3. Confusion Matrix Heatmap
    4. ROC Curve with actual AUC value
    """
    setup_eval_plot_style()
    fig_path = Path(figures_dir)
    fig_path.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history["loss"]) + 1)

    # 1. Training vs Validation Loss
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history["loss"], "o-", color="#2563EB", linewidth=2, label="Training Loss")
    ax.plot(epochs, history["val_loss"], "s--", color="#DC2626", linewidth=2, label="Validation Loss")
    best_loss_epoch = int(np.argmin(history["val_loss"]) + 1)
    min_val_loss = min(history["val_loss"])
    ax.scatter([best_loss_epoch], [min_val_loss], color="#16A34A", s=120, zorder=5, label=f"Best Model (Epoch {best_loss_epoch})")
    ax.set_title("Evaluation Chart 1: Training vs. Validation Loss by Epoch", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Binary Cross-Entropy Loss", fontsize=11)
    ax.set_xticks(epochs)
    ax.legend(frameon=True)
    plt.tight_layout()
    chart1_path = fig_path / "eval1_loss_epochs.png"
    plt.savefig(chart1_path)
    plt.close()
    print(f"[Evaluation] Saved Chart 1: {chart1_path.name}")

    # 2. Training vs Validation Accuracy
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history["accuracy"], "o-", color="#0D9488", linewidth=2, label="Training Accuracy")
    ax.plot(epochs, history["val_accuracy"], "s--", color="#F59E0B", linewidth=2, label="Validation Accuracy")
    best_acc_epoch = int(np.argmax(history["val_accuracy"]) + 1)
    max_val_acc = max(history["val_accuracy"])
    ax.scatter([best_acc_epoch], [max_val_acc], color="#16A34A", s=120, zorder=5, label=f"Peak Accuracy ({max_val_acc*100:.2f}%)")
    ax.set_title("Evaluation Chart 2: Training vs. Validation Accuracy by Epoch", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_xticks(epochs)
    ax.legend(frameon=True)
    plt.tight_layout()
    chart2_path = fig_path / "eval2_accuracy_epochs.png"
    plt.savefig(chart2_path)
    plt.close()
    print(f"[Evaluation] Saved Chart 2: {chart2_path.name}")

    # 3. Confusion Matrix Heatmap
    cm = confusion_matrix(y_test, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(7, 6))
    annot_matrix = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot_matrix[i, j] = f"{cm[i, j]:,}\n({cm_norm[i, j]*100:.1f}%)"

    sns.heatmap(
        cm,
        annot=annot_matrix,
        fmt="",
        cmap="Blues",
        cbar=True,
        xticklabels=["Predicted <=50K", "Predicted >50K"],
        yticklabels=["Actual <=50K", "Actual >50K"],
        linewidths=1.2,
        ax=ax,
    )
    ax.set_title("Evaluation Chart 3: Test Set Confusion Matrix", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    chart3_path = fig_path / "eval3_confusion_matrix.png"
    plt.savefig(chart3_path)
    plt.close()
    print(f"[Evaluation] Saved Chart 3: {chart3_path.name}")

    # 4. ROC Curve with AUC Value
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_val = roc_auc_score(y_test, y_prob)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color="#2563EB", linewidth=2.5, label=f"MLP Classifier (AUC = {auc_val:.4f})")
    ax.plot([0, 1], [0, 1], color="#94A3B8", linestyle="--", linewidth=1.5, label="Random Guess (AUC = 0.5000)")
    ax.fill_between(fpr, tpr, alpha=0.15, color="#3B82F6")
    ax.set_title("Evaluation Chart 4: Receiver Operating Characteristic (ROC) Curve", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    ax.set_ylabel("True Positive Rate (Recall / Sensitivity)", fontsize=11)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.legend(loc="lower right", frameon=True, fontsize=11)
    plt.tight_layout()
    chart4_path = fig_path / "eval4_roc_curve.png"
    plt.savefig(chart4_path)
    plt.close()
    print(f"[Evaluation] Saved Chart 4: {chart4_path.name}")


def compute_permutation_importance(
    model: keras.Model,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: List[str],
    figures_dir: str = "outputs/figures",
    metrics_dir: str = "outputs/metrics",
    n_repeats: int = 3,
    top_k: int = 15,
) -> pd.DataFrame:
    """
    Computes model-agnostic permutation feature importance on validation data:
    Measures the drop in ROC-AUC when each feature column is randomly shuffled.
    Aggregates indicators and saves a bar chart.
    """
    print("\n--- COMPUTING PERMUTATION FEATURE IMPORTANCE ---")
    setup_eval_plot_style()
    fig_path = Path(figures_dir)
    metrics_path = Path(metrics_dir)

    # Baseline score
    baseline_probs = model.predict(X_val, batch_size=256, verbose=0).ravel()
    baseline_auc = roc_auc_score(y_val, baseline_probs)
    print(f"[Permutation Importance] Baseline Validation ROC-AUC: {baseline_auc:.4f}")

    rng = np.random.default_rng(42)
    importance_scores = []

    for idx, f_name in enumerate(feature_names):
        drops = []
        for _ in range(n_repeats):
            X_permuted = X_val.copy()
            X_permuted[:, idx] = rng.permutation(X_permuted[:, idx])
            perm_probs = model.predict(X_permuted, batch_size=256, verbose=0).ravel()
            perm_auc = roc_auc_score(y_val, perm_probs)
            drops.append(baseline_auc - perm_auc)
        
        mean_drop = float(np.mean(drops))
        std_drop = float(np.std(drops))
        importance_scores.append({
            "Feature": f_name,
            "AUC_Drop_Mean": mean_drop,
            "AUC_Drop_Std": std_drop,
        })

    imp_df = pd.DataFrame(importance_scores).sort_values(by="AUC_Drop_Mean", ascending=False)
    imp_df.to_csv(metrics_path / "feature_importance.csv", index=False)
    print(f"[Permutation Importance] Metrics saved to: {metrics_path / 'feature_importance.csv'}")

    # Plot top_k features
    top_df = imp_df.head(top_k).iloc[::-1]  # Invert so highest is on top
    fig, ax = plt.subplots(figsize=(10, 7))
    palette = sns.color_palette("mako_r", n_colors=len(top_df))
    ax.barh(
        top_df["Feature"],
        top_df["AUC_Drop_Mean"],
        xerr=top_df["AUC_Drop_Std"],
        color=palette,
        edgecolor="#1E293B",
        linewidth=0.8,
        height=0.65,
        capsize=3,
    )
    ax.set_title(f"Permutation Feature Importance (Top {top_k} Features by AUC Reduction)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Mean Drop in Validation ROC-AUC When Permuted", fontsize=11, labelpad=8)
    ax.set_ylabel("Engineered Feature", fontsize=11, labelpad=8)
    plt.tight_layout()
    chart5_path = fig_path / "eval5_feature_importance.png"
    plt.savefig(chart5_path)
    plt.close()
    print(f"[Permutation Importance] Saved Chart 5: {chart5_path.name}")

    return imp_df


if __name__ == "__main__":
    from data_loader import load_dataset
    from preprocessing import prepare_data_pipeline
    from model_training import train_production_model

    df, _ = load_dataset()
    data_dict = prepare_data_pipeline(df)

    cfg = {"hidden_units": [128, 64], "dropout_rates": [0.30, 0.20], "learning_rate": 0.001}
    model, history = train_production_model(
        data_dict["X_train"], data_dict["y_train"],
        data_dict["X_val"], data_dict["y_val"],
        cfg
    )

    eval_out = evaluate_test_set(model, data_dict["X_test"], data_dict["y_test"])
    plot_evaluation_charts(history, data_dict["y_test"], eval_out["y_prob"], eval_out["y_pred"])
    compute_permutation_importance(model, data_dict["X_val"], data_dict["y_val"], data_dict["feature_names"])
    print("Evaluation completed successfully.")
