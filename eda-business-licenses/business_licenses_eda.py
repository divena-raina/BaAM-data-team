"""
Outputs:
    - Summary statistics, missing values, and coverage checks in CSV files.
    - Distribution plots for key variables.
    - Correlation heatmap across numeric features.
    - Outlier report using the IQR rule.
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
DATA_PATH = REPO_ROOT / "business_licenses_datasets" / \
    "cleaned_business_licenses.csv"
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
    # Numeric summary statistics
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        summary_stats = df[numeric_cols].describe().T
        summary_stats.to_csv(OUTPUT_DIR / "summary_statistics_numeric.csv")
        print("Saved numeric summary statistics to summary_statistics_numeric.csv")

    # Categorical summary statistics
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        cat_summary = []
        for col in categorical_cols:
            value_counts = df[col].value_counts()
            cat_summary.append({
                "column": col,
                "unique_count": df[col].nunique(),
                "most_common": value_counts.index[0] if len(value_counts) > 0 else "N/A",
                "most_common_count": value_counts.iloc[0] if len(value_counts) > 0 else 0,
                "most_common_pct": (value_counts.iloc[0] / len(df)) * 100 if len(value_counts) > 0 else 0,
                "top_5_values": " | ".join([f"{val} ({count})" for val, count in value_counts.head(5).items()])
            })
        cat_summary_df = pd.DataFrame(cat_summary)
        cat_summary_df.to_csv(OUTPUT_DIR / "summary_statistics_categorical.csv", index=False)
        print("Saved categorical summary statistics to summary_statistics_categorical.csv")

    # Missing values summary
    missing = (
        df.isna()
        .sum()
        .rename("missing_count")
        .to_frame()
        .assign(missing_pct=lambda s: (s["missing_count"] / len(df)) * 100)
        .sort_values("missing_pct", ascending=False)
    )
    missing.to_csv(OUTPUT_DIR / "missing_values_summary.csv")
    print("Saved missing value summary to missing_values_summary.csv")


def check_community_area_coverage(df: pd.DataFrame) -> None:
    ca_col = "COMMUNITY AREA" if "COMMUNITY AREA" in df.columns else "community_area"
    if ca_col not in df.columns:
        print("WARNING: Community area column not found in dataset.")
        return

    expected = set(range(1, 78))
    observed = set(pd.to_numeric(df[ca_col], errors="coerce").dropna().astype(int).unique())
    missing = sorted(expected - observed)
    if missing:
        print(f"WARNING: Missing community areas in dataset: {missing}")
    else:
        print("All community areas (1–77) are represented in the dataset.")


def plot_distributions(df: pd.DataFrame, numeric_columns: Iterable[str], categorical_columns: Iterable[str]) -> None:
    # Plot numeric distributions
    for column in numeric_columns:
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

    # Plot categorical distributions
    for column in categorical_columns:
        if column not in df.columns:
            print(f"Skipping {column}: column not found.")
            continue

        value_counts = df[column].value_counts()
        if len(value_counts) == 0:
            print(f"Skipping {column}: column has no data.")
            continue

        # Limit to top 20 categories for readability
        top_values = value_counts.head(20)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.barh(range(len(top_values)), top_values.values, color="#1f77b4", alpha=0.75)
        ax.set_yticks(range(len(top_values)))
        ax.set_yticklabels([str(v)[:50] for v in top_values.index])  # Truncate long labels
        ax.set_xlabel("Frequency")
        ax.set_ylabel(column)
        ax.set_title(f"Distribution of {column} (Top 20)")
        ax.invert_yaxis()
        fig.tight_layout()
        output_path = PLOTS_DIR / f"{column}_distribution.png"
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved distribution plot for {column} to {output_path.name}")

    # Combined boxplot for numeric variables
    cleaned_columns = []
    boxplot_data = []
    for column in numeric_columns:
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
        ax.set_title("Boxplots for key numeric variables")
        ax.set_ylim(0, 200)
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        boxplot_path = PLOTS_DIR / "key_variables_boxplot.png"
        fig.savefig(boxplot_path, dpi=150)
        plt.close(fig)
        print(f"Saved boxplot overview to {boxplot_path.name}")


def correlation_heatmap(df: pd.DataFrame) -> None:
    numeric_df = df.select_dtypes(include=[np.number]).replace([
        np.inf, -np.inf], np.nan)
    
    if len(numeric_df.columns) < 2:
        print("Not enough numeric columns for correlation analysis.")
        return
    
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
                "min_value": series.min(),
                "max_value": series.max(),
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


def main() -> int:
    ensure_output_dirs()
    df = load_dataset()
    df = df.replace([np.inf, -np.inf], np.nan)

    compute_summary_tables(df)
    check_community_area_coverage(df)

    # Numeric variables for distribution plots
    numeric_distribution_columns = [
        "WARD",
        "PRECINCT",
        "POLICE DISTRICT",
        "COMMUNITY AREA",
        "LATITUDE",
        "LONGITUDE",
        "LICENSE CODE",
    ]

    # Categorical variables for distribution plots
    categorical_distribution_columns = [
        "LICENSE DESCRIPTION",
        "BUSINESS ACTIVITY",
        "APPLICATION TYPE",
        "LICENSE STATUS",
        "COMMUNITY AREA NAME",
    ]

    plot_distributions(df, numeric_distribution_columns, categorical_distribution_columns)
    correlation_heatmap(df)
    detect_outliers(df)

    print("EDA complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
