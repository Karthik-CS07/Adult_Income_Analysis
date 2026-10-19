"""
Adult Census Income Analysis and Prediction
Module: visualizations.py
Description: Generates the 6 mandatory EDA visualizations + bonus domain plots,
             styled with high aesthetics, annotated with percentages, and saved at 300 DPI.
             Includes documentation of the Seven Stages of Visualization.
"""

import os
from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


# Set global clean aesthetics
plt.rcParams.update({
    "font.sans-serif": "DejaVu Sans",
    "axes.edgecolor": "#CBD5E1",
    "axes.linewidth": 1.2,
    "grid.color": "#E2E8F0",
    "grid.linestyle": "--",
    "grid.alpha": 0.6,
})


def setup_plot_style():
    """Configures high-quality styling for figures."""
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["savefig.bbox"] = "tight"


def plot_income_distribution(df: pd.DataFrame, save_path: Path):
    """
    Visualization 1: Income Distribution
    Shows class distribution with counts and percentages, highlighting the ~76:24 imbalance.
    """
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    target_col = [c for c in df.columns if "income" in c.lower()][0]
    data = df[target_col].astype(str).str.strip().str.rstrip(".")
    counts = data.value_counts()
    percentages = (counts / len(data) * 100).round(1)

    palette = ["#3B82F6", "#10B981"]
    bars = ax.bar(counts.index, counts.values, color=palette, width=0.55, edgecolor="#1E293B", linewidth=1.2)

    for bar, pct in zip(bars, percentages):
        height = bar.get_height()
        ax.annotate(
            f"{height:,}\n({pct}%)",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center", va="bottom",
            fontsize=11, fontweight="bold", color="#1E293B"
        )

    ax.set_title("Visualization 1: Target Class Income Distribution (<=50K vs >50K)", fontsize=13, fontweight="bold", pad=14)
    ax.set_xlabel("Annual Income Level", fontsize=11, labelpad=10)
    ax.set_ylabel("Number of Individuals", fontsize=11, labelpad=10)
    ax.set_ylim(0, max(counts.values) * 1.18)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[Visualizations] Saved Viz 1: {save_path.name}")


def plot_age_distribution(df: pd.DataFrame, save_path: Path):
    """
    Visualization 2: Age Distribution
    Displays histogram with KDE curve, noting the right-skew and peak working-age density.
    """
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(9, 5))

    age_data = df["age"].dropna()
    mean_val = age_data.mean()
    median_val = age_data.median()

    sns.histplot(
        age_data, kde=True, bins=35, color="#6366F1", edgecolor="#312E81", alpha=0.6, ax=ax, line_kws={"linewidth": 2.5}
    )

    ax.axvline(median_val, color="#EF4444", linestyle="--", linewidth=2, label=f"Median Age: {median_val:.0f} yrs")
    ax.axvline(mean_val, color="#F59E0B", linestyle="-.", linewidth=2, label=f"Mean Age: {mean_val:.1f} yrs")

    ax.set_title("Visualization 2: Age Distribution of Census Respondents", fontsize=13, fontweight="bold", pad=14)
    ax.set_xlabel("Age (Years)", fontsize=11, labelpad=10)
    ax.set_ylabel("Respondent Count", fontsize=11, labelpad=10)
    ax.legend(frameon=True, facecolor="white", edgecolor="#E2E8F0")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[Visualizations] Saved Viz 2: {save_path.name}")


def plot_education_frequency(df: pd.DataFrame, save_path: Path):
    """
    Visualization 3: Education Level Frequency
    Horizontal bar chart highlighting educational attainment among respondents.
    """
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    edu_counts = df["education"].astype(str).str.strip().value_counts()
    
    y_pos = np.arange(len(edu_counts))
    palette = sns.color_palette("Blues_r", n_colors=len(edu_counts))

    bars = ax.barh(y_pos, edu_counts.values, color=palette, edgecolor="#1E293B", linewidth=0.8, height=0.65)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(edu_counts.index, fontsize=10)
    ax.invert_yaxis()  # Top category at top

    for bar in bars:
        width = bar.get_width()
        pct = (width / len(df) * 100)
        ax.annotate(
            f"{width:,} ({pct:.1f}%)",
            xy=(width, bar.get_y() + bar.get_height() / 2),
            xytext=(6, 0),
            textcoords="offset points",
            ha="left", va="center",
            fontsize=9, color="#1E293B"
        )

    ax.set_title("Visualization 3: Educational Attainment Frequency", fontsize=13, fontweight="bold", pad=14)
    ax.set_xlabel("Number of Individuals", fontsize=11, labelpad=10)
    ax.set_ylabel("Education Level", fontsize=11, labelpad=10)
    ax.set_xlim(0, max(edu_counts.values) * 1.22)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[Visualizations] Saved Viz 3: {save_path.name}")


def plot_working_hours_by_income(df: pd.DataFrame, save_path: Path):
    """
    Visualization 4: Weekly Working Hours by Income Level
    Boxplot showing distributions, standard 40h mark, and overtime tendencies.
    """
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(8, 6))

    target_col = [c for c in df.columns if "income" in c.lower()][0]
    clean_df = df.copy()
    clean_df["income_clean"] = clean_df[target_col].astype(str).str.strip().str.rstrip(".")

    palette = {"<=50K": "#93C5FD", ">50K": "#34D399"}
    sns.boxplot(
        data=clean_df,
        x="income_clean",
        y="hours.per.week",
        hue="income_clean",
        legend=False,
        palette=palette,
        boxprops=dict(edgecolor="#1E293B", linewidth=1.5),
        whiskerprops=dict(color="#1E293B", linewidth=1.5),
        capprops=dict(color="#1E293B", linewidth=1.5),
        medianprops=dict(color="#EF4444", linewidth=2.5),
        flierprops=dict(marker="o", markersize=3, alpha=0.3, markeredgecolor="gray"),
        width=0.45,
        ax=ax
    )

    ax.axhline(40, color="#64748B", linestyle=":", linewidth=1.5, label="Standard 40-hr Workweek")

    ax.set_title("Visualization 4: Weekly Working Hours by Income Category", fontsize=13, fontweight="bold", pad=14)
    ax.set_xlabel("Income Class", fontsize=11, labelpad=10)
    ax.set_ylabel("Hours Worked Per Week", fontsize=11, labelpad=10)
    ax.legend(loc="upper right", frameon=True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[Visualizations] Saved Viz 4: {save_path.name}")


def plot_education_income_relationship(df: pd.DataFrame, save_path: Path):
    """
    Visualization 5: Education and Income Relationship
    Grouped 100% stacked bar chart showing income proportions across ordered education levels.
    """
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(12, 6))

    target_col = [c for c in df.columns if "income" in c.lower()][0]
    temp_df = df.copy()
    temp_df["income_clean"] = temp_df[target_col].astype(str).str.strip().str.rstrip(".")
    temp_df["edu_clean"] = temp_df["education"].astype(str).str.strip()

    # Sort education by mean education.num if available, else by >50K rate
    if "education.num" in temp_df.columns:
        edu_order = temp_df.groupby("edu_clean")["education.num"].mean().sort_values().index
    else:
        edu_order = temp_df.groupby("edu_clean")["income_clean"].apply(lambda s: (s == ">50K").mean()).sort_values().index

    ct = pd.crosstab(temp_df["edu_clean"], temp_df["income_clean"], normalize="index") * 100
    ct = ct.reindex(edu_order)

    cols = ["<=50K", ">50K"]
    for c in cols:
        if c not in ct.columns:
            ct[c] = 0.0

    p1 = ax.bar(ct.index, ct["<=50K"], label="<=50K", color="#94A3B8", width=0.6, edgecolor="#1E293B")
    p2 = ax.bar(ct.index, ct[">50K"], bottom=ct["<=50K"], label=">50K", color="#059669", width=0.6, edgecolor="#1E293B")

    # Annotate high income %
    for idx, high_pct in enumerate(ct[">50K"]):
        if high_pct > 5:
            ax.annotate(
                f"{high_pct:.1f}%",
                xy=(idx, 100 - high_pct / 2),
                ha="center", va="center",
                color="white", fontweight="bold", fontsize=9
            )

    ax.set_title("Visualization 5: Income Distribution Across Educational Attainment (Ordered)", fontsize=13, fontweight="bold", pad=14)
    ax.set_xlabel("Education Level", fontsize=11, labelpad=10)
    ax.set_ylabel("Proportion (%)", fontsize=11, labelpad=10)
    ax.set_ylim(0, 105)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    ax.legend(title="Income Class", loc="upper left", frameon=True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[Visualizations] Saved Viz 5: {save_path.name}")


def plot_correlation_heatmap(df: pd.DataFrame, save_path: Path):
    """
    Visualization 6: Numerical Correlation Heatmap
    Heatmap of Pearson correlation coefficients between continuous variables and binary target.
    """
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(8, 7))

    target_col = [c for c in df.columns if "income" in c.lower()][0]
    temp_df = df.copy()
    temp_df["high_income"] = temp_df[target_col].astype(str).str.strip().str.rstrip(".").apply(lambda x: 1 if ">50k" in x.lower() else 0)

    num_cols = list(temp_df.select_dtypes(include=[np.number]).columns)
    corr = temp_df[num_cols].corr()

    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        vmax=0.5,
        vmin=-0.2,
        center=0,
        annot=True,
        fmt=".2f",
        square=True,
        linewidths=0.8,
        cbar_kws={"shrink": 0.8, "label": "Pearson Correlation"},
        ax=ax
    )

    ax.set_title("Visualization 6: Correlation Heatmap of Numerical Features", fontsize=13, fontweight="bold", pad=14)
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[Visualizations] Saved Viz 6: {save_path.name}")


def plot_capital_gain_loss_by_income(df: pd.DataFrame, save_path: Path):
    """
    Bonus Visualization 7: Capital Gains and Capital Losses by Income Class
    Demonstrates the extreme skewness and zero-inflation of investment metrics.
    """
    setup_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    target_col = [c for c in df.columns if "income" in c.lower()][0]
    temp_df = df.copy()
    temp_df["income_clean"] = temp_df[target_col].astype(str).str.strip().str.rstrip(".")

    # Filter non-zero gains/losses to show conditional distribution
    gain_nonzero = temp_df[temp_df["capital.gain"] > 0]
    loss_nonzero = temp_df[temp_df["capital.loss"] > 0]

    sns.boxplot(
        data=gain_nonzero, x="income_clean", y="capital.gain",
        hue="income_clean", legend=False,
        palette={"<=50K": "#93C5FD", ">50K": "#34D399"},
        ax=axes[0]
    )
    axes[0].set_title("Non-Zero Capital Gains by Income", fontsize=11, fontweight="bold")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Capital Gain (USD, Log Scale)")
    axes[0].set_xlabel("Income Class")

    sns.boxplot(
        data=loss_nonzero, x="income_clean", y="capital.loss",
        hue="income_clean", legend=False,
        palette={"<=50K": "#93C5FD", ">50K": "#34D399"},
        ax=axes[1]
    )
    axes[1].set_title("Non-Zero Capital Losses by Income", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Capital Loss (USD)")
    axes[1].set_xlabel("Income Class")

    plt.suptitle("Bonus Visualization 7: Capital Gains and Losses Among Active Investors", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[Visualizations] Saved Viz 7: {save_path.name}")


def plot_occupation_income(df: pd.DataFrame, save_path: Path):
    """
    Bonus Visualization 8: High Income Rate Across Occupation Categories.
    """
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    target_col = [c for c in df.columns if "income" in c.lower()][0]
    temp_df = df.copy()
    temp_df["is_high"] = temp_df[target_col].astype(str).str.strip().str.rstrip(".").apply(lambda x: 1 if ">50k" in x.lower() else 0)
    temp_df["occ_clean"] = temp_df["occupation"].astype(str).str.strip().replace("?", "Missing/Other")

    occ_rates = temp_df.groupby("occ_clean")["is_high"].agg(["mean", "count"]).sort_values(by="mean", ascending=True)
    occ_rates["pct"] = occ_rates["mean"] * 100

    palette = sns.color_palette("viridis", n_colors=len(occ_rates))
    bars = ax.barh(occ_rates.index, occ_rates["pct"], color=palette, edgecolor="#1E293B", height=0.65)

    for bar, (_, row) in zip(bars, occ_rates.iterrows()):
        w = bar.get_width()
        ax.annotate(
            f"{w:.1f}% (N={int(row['count']):,})",
            xy=(w, bar.get_y() + bar.get_height() / 2),
            xytext=(6, 0),
            textcoords="offset points",
            ha="left", va="center",
            fontsize=9
        )

    ax.set_title("Bonus Visualization 8: High Income (>50K) Proportion by Occupation", fontsize=13, fontweight="bold", pad=14)
    ax.set_xlabel("Percentage Earning >50K (%)", fontsize=11, labelpad=10)
    ax.set_ylabel("Occupation", fontsize=11, labelpad=10)
    ax.set_xlim(0, max(occ_rates["pct"]) * 1.35)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[Visualizations] Saved Viz 8: {save_path.name}")


def generate_all_eda_visualizations(df: pd.DataFrame, figures_dir: str = "outputs/figures"):
    """
    Generates and persists all EDA figures required by the specification.
    """
    fig_path = Path(figures_dir)
    fig_path.mkdir(parents=True, exist_ok=True)

    print("\n--- GENERATING EDA VISUALIZATIONS ---")
    plot_income_distribution(df, fig_path / "viz1_income_distribution.png")
    plot_age_distribution(df, fig_path / "viz2_age_distribution.png")
    plot_education_frequency(df, fig_path / "viz3_education_frequency.png")
    plot_working_hours_by_income(df, fig_path / "viz4_working_hours_by_income.png")
    plot_education_income_relationship(df, fig_path / "viz5_education_income_relationship.png")
    plot_correlation_heatmap(df, fig_path / "viz6_correlation_heatmap.png")
    plot_capital_gain_loss_by_income(df, fig_path / "viz7_capital_gain_loss_by_income.png")
    plot_occupation_income(df, fig_path / "viz8_occupation_income.png")
    print(f"[Visualizations] All EDA visualizations successfully written to: {fig_path.resolve()}\n")


if __name__ == "__main__":
    from data_loader import load_dataset
    df, _ = load_dataset()
    generate_all_eda_visualizations(df)
