"""
Utility script to clean the Violence Reduction shooting dataset so it is ready
for feature engineering and merging with other sources.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


RAW_DATA_PATH = Path(
    "cleaned_data/shooting dataset pt3/raw_data_Violence_Reduction_-_Victims_of_"
    "Homicides_and_Non-Fatal_Shootings_20251119.csv"
)
OUTPUT_CSV_PATH = Path(
    "cleaned_data/shooting dataset pt3/cleaned_shooting_dataset_pt3.csv"
)
DOC_PATH = Path(
    "cleaned_data/shooting dataset pt3/shooting_p3_documentation.md")

COLUMNS_TO_DROP = [
    "HOMICIDE_VICTIM_FIRST_NAME",
    "HOMICIDE_VICTIM_MI",
    "HOMICIDE_VICTIM_LAST_NAME",
    "STATE_HOUSE_DISTRICT",
    "STATE_SENATE_DISTRICT",
    "STREET_OUTREACH_ORGANIZATION",
]


def format_zip(zip_value: pd.Series) -> pd.Series:
    """Return a zero-padded ZIP Code string column."""
    zip_str = zip_value.astype("string").str.strip()
    zip_str = zip_str.str.replace(r"\.0$", "", regex=True)
    # keep numeric characters only so values like '60637-1234' keep base zip
    zip_str = zip_str.str.extract(r"(\d+)", expand=False)
    return zip_str.str.zfill(5)


def clean_dataset() -> None:
    """Clean the raw CSV, persist cleaned data and documentation."""
    df = pd.read_csv(RAW_DATA_PATH)
    doc_lines = []

    starting_rows = len(df)
    doc_lines.append(f"- Loaded {starting_rows} rows from raw dataset.")

    cols_present = [col for col in COLUMNS_TO_DROP if col in df.columns]
    if cols_present:
        df = df.drop(columns=cols_present)
        doc_lines.append(
            "- Dropped personally identifiable or duplicative columns "
            f"{', '.join(cols_present)}."
        )

    df["CASE_NUMBER"] = df["CASE_NUMBER"].astype(
        "string").str.strip().str.upper()
    df.loc[df["CASE_NUMBER"].eq("") | df["CASE_NUMBER"].isin(
        ["<NA>", "NAN"]), "CASE_NUMBER"] = pd.NA

    df["GUNSHOT_INJURY_I"] = (
        df["GUNSHOT_INJURY_I"].astype("string").str.strip().str.upper()
    )
    gunshot_mask = df["GUNSHOT_INJURY_I"].eq("YES")
    doc_lines.append(
        f"- Filtered to gunshot victims only: removed {len(df) - gunshot_mask.sum()} rows."
    )
    df = df[gunshot_mask]

    df["DATE"] = pd.to_datetime(
        df["DATE"], errors="coerce", format="%m/%d/%Y %I:%M:%S %p"
    )
    df["BLOCK"] = df["BLOCK"].astype("string").str.strip().str.upper()
    df["VICTIMIZATION_PRIMARY"] = (
        df["VICTIMIZATION_PRIMARY"].astype("string").str.strip()
    )
    df["INCIDENT_PRIMARY"] = df["INCIDENT_PRIMARY"].astype(
        "string").str.strip()
    df["ZIP_CODE"] = format_zip(df["ZIP_CODE"])
    doc_lines.append(
        "- Standardized case numbers, street blocks, primary classifications, "
        "and ZIP codes for consistent joins."
    )

    cutoff = pd.Timestamp("2012-01-01")
    year_mask = df["DATE"] >= cutoff
    doc_lines.append(
        f"- Filtered out incidents prior to 2012: removed {len(df) - year_mask.sum()} rows."
    )
    df = df[year_mask]

    essential_non_null = ["CASE_NUMBER", "DATE", "BLOCK", "ZIP_CODE"]
    before_drop_na = len(df)
    df = df.dropna(subset=essential_non_null)
    doc_lines.append(
        f"- Dropped {before_drop_na - len(df)} rows missing essential identifiers or dates."
    )

    df["DATE"] = df["DATE"].dt.strftime("%Y-%m-%d %H:%M:%S")

    df.to_csv(OUTPUT_CSV_PATH, index=False)
    doc_lines.append(
        f"- Final cleaned dataset saved to `{OUTPUT_CSV_PATH}` with {len(df)} rows."
    )

    documentation = "\n".join(
        [
            "# Shooting dataset cleaning summary",
            "",
            "Cleaning actions performed:",
            *doc_lines,
        ]
    )
    DOC_PATH.write_text(documentation)


if __name__ == "__main__":
    clean_dataset()
