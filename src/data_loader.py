"""
Load the raw NOAA/CDC Dengue Forecasting Project surveillance data.

This module does ONE job: turn the two raw CSVs into a tidy two-column
time series (date, cases) for a single city. All cleaning/validation
lives in preprocessing.py so that loading stays boring and inspectable.
"""

from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

LABELS_CSV = RAW_DIR / "dengue_labels_train.csv"
FEATURES_CSV = RAW_DIR / "dengue_features_train.csv"

CITIES = {"sj": "San Juan, Puerto Rico", "iq": "Iquitos, Peru"}


def load_raw(city: str = "sj") -> pd.DataFrame:
    """
    Return a DataFrame with columns ['date', 'cases'] for one city.

    Why we merge two files:
      - dengue_labels_train.csv holds the target we care about (total_cases)
        but only identifies time as (year, weekofyear).
      - Those two columns have a known quirk in this dataset (a stray
        'week 53' that disagrees with the year field), so they are NOT a
        safe time index.
      - dengue_features_train.csv carries `week_start_date`, an actual
        calendar date. We take that one column and nothing else.

    We deliberately ignore the climate covariates. The research question is
    "how much HISTORY do I need", so the only predictor allowed is the
    disease's own past.
    """
    if city not in CITIES:
        raise ValueError(f"city must be one of {list(CITIES)}, got {city!r}")

    labels = pd.read_csv(LABELS_CSV)
    features = pd.read_csv(FEATURES_CSV, usecols=["city", "year", "weekofyear", "week_start_date"])

    keys = ["city", "year", "weekofyear"]
    # validate="one_to_one" makes pandas raise if the join keys are not unique
    # on both sides. That is a cheap guard against silently duplicating rows.
    merged = labels.merge(features, on=keys, how="left", validate="one_to_one")

    city_df = merged.loc[merged["city"] == city].copy()

    out = pd.DataFrame({
        "date": pd.to_datetime(city_df["week_start_date"]),
        "cases": city_df["total_cases"],
    })
    return out.reset_index(drop=True)


if __name__ == "__main__":
    for c in CITIES:
        df = load_raw(c)
        print(f"{c} ({CITIES[c]}): {len(df)} rows, {df['date'].min().date()} -> {df['date'].max().date()}")
        print(df.head(3).to_string(index=False), "\n")
