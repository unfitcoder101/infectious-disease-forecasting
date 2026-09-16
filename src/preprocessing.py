"""
Validate and clean the weekly surveillance series.

Philosophy: this module AUDITS the data and reports what it finds, rather
than quietly patching things. Silent imputation is how you end up reporting
a result that is really an artifact of your own cleaning code.
"""

import pandas as pd


def audit_series(df: pd.DataFrame) -> dict:
    """
    Run the data-quality checks and return findings as a dict.
    Does not modify the data.

    Checks:
      1. chronological order
      2. duplicate dates
      3. missing target values
      4. spacing between consecutive weeks
    """
    dates = df["date"]
    gaps = dates.diff().dt.days.dropna()

    # The dataset uses an epidemiological-week convention: the final week of
    # each year is stretched to 8-9 days so the next year can start on Jan 1.
    # Those are NOT missing weeks. A genuine missing week would show up as a
    # gap of 14+ days.
    return {
        "n_rows": len(df),
        "date_min": dates.min(),
        "date_max": dates.max(),
        "is_sorted": bool(dates.is_monotonic_increasing),
        "n_duplicate_dates": int(dates.duplicated().sum()),
        "n_missing_cases": int(df["cases"].isna().sum()),
        "gap_counts": gaps.value_counts().sort_index().to_dict(),
        "n_true_missing_weeks": int((gaps >= 14).sum()),
        "n_negative_cases": int((df["cases"] < 0).sum()),
    }


def print_audit(findings: dict, city_label: str) -> None:
    print(f"\n=== DATA AUDIT: {city_label} ===")
    print(f"observations        : {findings['n_rows']}")
    print(f"date range          : {findings['date_min'].date()} -> {findings['date_max'].date()}")
    print(f"chronologically sorted: {findings['is_sorted']}")
    print(f"duplicate dates     : {findings['n_duplicate_dates']}")
    print(f"missing case values : {findings['n_missing_cases']}")
    print(f"negative case values: {findings['n_negative_cases']}")
    print(f"gaps between weeks  : {findings['gap_counts']}  (days -> count)")
    print(f"TRUE missing weeks (gap >= 14 days): {findings['n_true_missing_weeks']}")


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Produce the analysis-ready series.

    Steps, all explicit:
      1. sort chronologically  (the whole experiment depends on time order)
      2. drop duplicate dates, keeping the first
      3. cast cases to integer
      4. reset the index

    NOTE ON MISSING DATA: this series has zero missing targets, so no
    imputation is performed. If imputation were ever needed here, the only
    defensible choice for forecasting is a BACKWARD-looking fill (e.g.
    forward-fill / interpolation using past values only). A centred or
    interpolating fill would leak future information into past rows.
    """
    out = df.sort_values("date").drop_duplicates(subset="date", keep="first").copy()
    out["cases"] = out["cases"].astype(int)
    return out.reset_index(drop=True)


if __name__ == "__main__":
    from data_loader import load_raw, CITIES

    for city, label in CITIES.items():
        raw = load_raw(city)
        print_audit(audit_series(raw), label)
        clean = preprocess(raw)
        print(f"-> clean series: {len(clean)} rows")
