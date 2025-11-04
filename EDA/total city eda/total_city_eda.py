"""
Outputs:
    - Summary statistics, missing values, and coverage checks in CSV files.
    - Distribution plots for key variables.
    - Correlation heatmap across numeric features.
    - Outlier report using the IQR rule.

The script focuses on citywide trends across all community areas (1–77),
with explicit check for Altgeld Gardens (community area 54).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - guard rail for missing optional deps
    raise RuntimeError(
        "matplotlib is required to generate plots. "
        "Install it before running this script."
    ) from exc


REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "merged data sets" / \
    "aggregated" / "aggregated_chicago_dataset_2012plus.csv"
OUTPUT_DIR = Path(__file__).resolve().parent
PLOTS_DIR = OUTPUT_DIR / "plots"


def ensure_output_dirs() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print(
        f"Loaded dataset with {df.shape[0]:,} rows and {df.shape[1]} columns.")
    return df


def compute_summary_tables(df: pd.DataFrame) -> None:
    summary_stats = df.describe().T
    summary_stats.to_csv(OUTPUT_DIR / "summary_statistics_numeric.csv")
    print("Saved numeric summary statistics to summary_statistics_numeric.csv")

    missing = (
        df.isna()
        .sum()
        .rename("missing_count")
        .to_frame()
        .assign(missing_pct=lambda s: (s["missing_count"] / len(df)) * 100)
    )
    missing.to_csv(OUTPUT_DIR / "missing_values_summary.csv")
    print("Saved missing value summary to missing_values_summary.csv")


def check_community_area_coverage(df: pd.DataFrame) -> None:
    expected = set(range(1, 78))
    observed = set(df["community_area"].unique())
    missing = sorted(expected - observed)
    if missing:
        print(f"WARNING: Missing community areas in dataset: {missing}")
    else:
        print("All community areas (1–77) are represented in the dataset.")

    altgeld = df[df["community_area"] == 54]
    if altgeld.empty:
        print("WARNING: Altgeld Gardens (community area 54) is missing!")
    else:
        altgeld_summary = (
            altgeld.describe()
            .T[["mean", "min", "max"]]
            .rename(columns={"mean": "altgeld_mean", "min": "altgeld_min", "max": "altgeld_max"})
        )
        altgeld_summary.to_csv(OUTPUT_DIR / "altgeld_gardens_summary.csv")
        print("Altgeld Gardens profile saved to altgeld_gardens_summary.csv")


def plot_distributions(df: pd.DataFrame, columns: Iterable[str]) -> None:
    for column in columns:
        if column not in df.columns:
            print(f"Skipping {column}: column not found.")
            continue

        series = pd.to_numeric(df[column], errors="coerce").replace(
            [np.inf, -np.inf], np.nan).dropna()
        if series.empty:
            print(f"Skipping {column}: column has no data.")
            continue

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(series, bins=30, color="#1f77b4",
                edgecolor="black", alpha=0.75)
        ax.set_title(f"Distribution of {column}")
        ax.set_xlabel(column)
        ax.set_ylabel("Frequency")
        fig.tight_layout()
        output_path = PLOTS_DIR / f"{column}_distribution.png"
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        print(f"Saved distribution plot for {column} to {output_path.name}")

    cleaned_columns = []
    boxplot_data = []
    for column in columns:
        if column not in df.columns:
            continue
        series = pd.to_numeric(df[column], errors="coerce").replace(
            [np.inf, -np.inf], np.nan).dropna()
        if series.empty:
            continue
        cleaned_columns.append(column)
        boxplot_data.append(series)

    if cleaned_columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.boxplot(
            boxplot_data,
            tick_labels=cleaned_columns,
            patch_artist=True,
            boxprops=dict(facecolor="#aec7e8", color="#1f77b4"),
            medianprops=dict(color="#d62728"),
        )
        ax.set_title("Boxplots for key variables")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        boxplot_path = PLOTS_DIR / "key_variables_boxplot.png"
        fig.savefig(boxplot_path, dpi=150)
        plt.close(fig)
        print(f"Saved boxplot overview to {boxplot_path.name}")


def correlation_heatmap(df: pd.DataFrame) -> None:
    numeric_df = df.select_dtypes(include=[np.number]).replace([
        np.inf, -np.inf], np.nan)
    corr = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    cax = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_title("Correlation Heatmap - Numeric Features")
    ticks = range(len(corr.columns))
    ax.set_xticks(ticks)
    ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticks(ticks)
    ax.set_yticklabels(corr.columns)
    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04, label="Correlation")
    fig.tight_layout()
    output_path = PLOTS_DIR / "correlation_heatmap.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved correlation heatmap to {output_path.name}")

    corr.to_csv(OUTPUT_DIR / "correlation_matrix.csv")
    print("Saved correlation matrix to correlation_matrix.csv")


def detect_outliers(df: pd.DataFrame) -> None:
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    outlier_records = []

    for column in numeric_cols:
        series = pd.to_numeric(df[column], errors="coerce").replace(
            [np.inf, -np.inf], np.nan).dropna()
        if series.empty:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        mask = (df[column] < lower_bound) | (df[column] > upper_bound)
        outlier_count = int(mask.sum())
        if outlier_count == 0:
            continue

        outlier_records.append(
            {
                "column": column,
                "outlier_count": outlier_count,
                "outlier_pct": (outlier_count / len(df)) * 100,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
            }
        )

    if outlier_records:
        outliers_df = pd.DataFrame(outlier_records).sort_values(
            "outlier_pct", ascending=False)
        outliers_path = OUTPUT_DIR / "outlier_report.csv"
        outliers_df.to_csv(outliers_path, index=False)
        print(f"Saved outlier report to {outliers_path.name}")
    else:
        print("No outliers detected via IQR method.")


def citywide_trends(df: pd.DataFrame) -> None:
    trend_columns = [
        "homicide_count",
        "shooting_count",
        "homicide_arrests",
        "shooting_arrests",
        "homicide_domestic",
        "shooting_domestic",
        "pop_total",
        "per_capita_income",
        "hardship_index",
    ]
    existing_columns = [c for c in trend_columns if c in df.columns]

    yearly = (
        df.groupby("year")[existing_columns]
        .sum(min_count=1)
        .sort_index()
    )
    yearly.to_csv(OUTPUT_DIR / "citywide_totals_by_year.csv")
    print("Saved citywide totals by year to citywide_totals_by_year.csv")

    community_profile = (
        df.groupby("community_area")[existing_columns]
        .mean()
        .sort_values("homicide_count", ascending=False)
    )
    community_profile.to_csv(OUTPUT_DIR / "community_area_profile.csv")
    print("Saved average community area profile to community_area_profile.csv")


def main() -> int:
    ensure_output_dirs()
    df = load_dataset()
    df = df.replace([np.inf, -np.inf], np.nan)

    compute_summary_tables(df)
    check_community_area_coverage(df)
    citywide_trends(df)

    distribution_columns = [
        "homicide_count",
        "shooting_count",
        "homicide_arrests",
        "shooting_arrests",
        "pop_total",
        "per_capita_income",
        "hardship_index",
        "homicides_per_10k",
        "shootings_per_10k",
    ]
    plot_distributions(df, distribution_columns)
    correlation_heatmap(df)
    detect_outliers(df)

    print("EDA complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
