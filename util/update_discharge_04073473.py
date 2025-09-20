"""Utility to replace discharge values in metadata/csv_filtered/04073473.csv.

Loads the CSV, computes the median discharge between 2013-01-01 and
2022-09-02 (inclusive), replaces values in that date range with the
median, and writes the updated data back to disk.
"""
from pathlib import Path

import pandas as pd

DATA_PATH = Path("metadata/csv_filtered/04073473.csv")
START_DATE = "2013-01-01"
END_DATE = "2022-09-02"


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"CSV file not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH, parse_dates=["date"])

    date_mask = (df["date"] >= START_DATE) & (df["date"] <= END_DATE)

    median_value = df.loc[date_mask, "discharge"].median(skipna=True)
    if pd.isna(median_value):
        raise ValueError(
            "Median discharge is NaN; ensure the selected date range has valid data.")

    df.loc[date_mask, "discharge"] = median_value

    df.to_csv(DATA_PATH, index=False)


if __name__ == "__main__":
    main()
