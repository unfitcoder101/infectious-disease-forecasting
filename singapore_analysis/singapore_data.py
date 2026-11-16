"""
Loader for the Singapore dengue fever series.

Returns the same two-column shape the existing pipeline expects
(`date`, `cases`), so the unmodified functions in src/ can be reused.
src/data_loader.py is DengAI-specific and is deliberately left untouched.

THE `date` COLUMN IS DERIVED, NOT READ
--------------------------------------
The CSV carries epi-week labels (e.g. 2014-W53) but no calendar dates.
`date` is the Sunday that starts each epi week under the rule the file was
shown to follow in singapore_data_audit.md: Sunday-Saturday weeks, week 1 is
the first week with at least 4 days in January. ISO-8601 is NOT used.

The dates only label rows. The models see lag values, never dates, and the
row order is identical to the file's `time_index` order, so no result
depends on this derivation. An assertion checks the derived dates are one
gapless 7-day sequence; nothing is inserted, dropped or reordered.
"""

import datetime as dt
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SINGAPORE_CSV = ROOT / "data" / "raw" / "singapore_dengue_fever_weekly_clean.csv"
TARGET_COLUMN = "dengue_fever_cases"
EXPECTED_ROWS = 574
SERIES_LABEL = "Singapore (MOH, Dengue Fever only)"


def epi_week_start(year: int, week: int) -> dt.date:
    jan4 = dt.date(year, 1, 4)
    week1 = jan4 - dt.timedelta(days=(jan4.weekday() + 1) % 7)   # Sunday on or before 4 Jan
    return week1 + dt.timedelta(days=7 * (week - 1))


def load_singapore_raw() -> pd.DataFrame:
    """The CSV exactly as supplied, plus the derived `date` column."""
    df = pd.read_csv(SINGAPORE_CSV)
    if len(df) != EXPECTED_ROWS:
        raise ValueError(f"expected {EXPECTED_ROWS} rows, found {len(df)}")
    if df[TARGET_COLUMN].isna().any() or (df[TARGET_COLUMN] < 0).any():
        raise ValueError("missing or negative case counts found; stop and re-audit")
    df["date"] = pd.to_datetime([epi_week_start(int(y), int(w))
                                 for y, w in zip(df["epi_year"], df["epi_week_number"])])
    gaps = df["date"].diff().dropna().dt.days
    if not (gaps == 7).all():
        raise ValueError("epi-week labels do not form a gapless weekly sequence; stop and re-audit")
    if not (df["time_index"].diff().dropna() == 1).all():
        raise ValueError("time_index is not consecutive")
    return df


def load_singapore() -> pd.DataFrame:
    """Two columns, `date` and `cases`, in file order. Drop-in for load_raw()."""
    df = load_singapore_raw()
    return pd.DataFrame({"date": df["date"], "cases": df[TARGET_COLUMN]}).reset_index(drop=True)


def epi_week_lookup() -> dict:
    """Map derived date -> epi-week label, for reporting fold boundaries."""
    df = load_singapore_raw()
    return dict(zip(df["date"], df["epi_week"]))
