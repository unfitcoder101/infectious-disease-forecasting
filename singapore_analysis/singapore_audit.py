"""
Phase 2 data audit for the Singapore dengue fever series.

Read-only: this script never modifies the raw CSV and never fills, drops or
reorders observations. It writes audit tables to singapore_analysis/tables/.

Run with:  .venv/bin/python singapore_analysis/singapore_audit.py
"""

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "data" / "raw" / "singapore_dengue_fever_weekly_clean.csv"
TABLES = Path(__file__).resolve().parent / "tables"
TARGET = "dengue_fever_cases"


# --- Candidate week-numbering conventions -----------------------------------
# Each function returns the first day of week 1 of `year` under one rule.
def w1_sun_4day(year):
    """Sunday-Saturday weeks; week 1 has >= 4 days in the new year
    (equivalently: the week containing 4 January). US CDC MMWR rule."""
    jan4 = dt.date(year, 1, 4)
    return jan4 - dt.timedelta(days=(jan4.weekday() + 1) % 7)


def w1_sun_jan1(year):
    """Sunday-Saturday weeks; week 1 is the week containing 1 January."""
    jan1 = dt.date(year, 1, 1)
    return jan1 - dt.timedelta(days=(jan1.weekday() + 1) % 7)


def w1_sun_first_full(year):
    """Sunday-Saturday weeks; week 1 is the first week wholly inside the year."""
    jan1 = dt.date(year, 1, 1)
    return jan1 + dt.timedelta(days=(6 - jan1.weekday()) % 7)


def w1_iso(year):
    """ISO-8601: Monday-Sunday weeks; week 1 contains 4 January."""
    jan4 = dt.date(year, 1, 4)
    return jan4 - dt.timedelta(days=jan4.weekday())


CONVENTIONS = {
    "Sun-Sat, week 1 has >=4 days in January": w1_sun_4day,
    "Sun-Sat, week 1 contains 1 January": w1_sun_jan1,
    "Sun-Sat, week 1 is first full week": w1_sun_first_full,
    "ISO-8601 (Mon-Sun)": w1_iso,
}


def weeks_in_year(rule, year):
    return (rule(year + 1) - rule(year)).days // 7


def week_start(year, week):
    """Derived Sunday start date under the convention the file matches."""
    return w1_sun_4day(year) + dt.timedelta(days=7 * (week - 1))


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    raw = CSV.read_bytes()
    df = pd.read_csv(CSV)
    n = len(df)

    checks = []

    def add(check, result, detail=""):
        checks.append(dict(check=check, result=result, detail=detail))

    add("Total observations", n, "expected 574")
    add("Columns", ", ".join(df.columns))
    add("Data types", "; ".join(f"{c}={t}" for c, t in df.dtypes.astype(str).items()))
    add("File encoding", "no BOM, LF line endings" if raw[:3] != b"\xef\xbb\xbf" and b"\r\n" not in raw
        else "BOM or CRLF present")
    add("Missing values (any column)", int(df.isna().sum().sum()))
    add("Fully duplicated rows", int(df.duplicated().sum()))
    add("Duplicate epi_week labels", int(df["epi_week"].duplicated().sum()))
    add("Duplicate (epi_year, epi_week_number)", int(df.duplicated(["epi_year", "epi_week_number"]).sum()))
    add("Duplicate time_index", int(df["time_index"].duplicated().sum()))
    add("Negative case counts", int((df[TARGET] < 0).sum()))
    add("Zero case counts", int((df[TARGET] == 0).sum()))
    add("Non-integer case counts", int((df[TARGET] % 1 != 0).sum()))
    add("time_index is exactly 1..N", bool((df["time_index"] == np.arange(1, n + 1)).all()))

    label = df["epi_year"].astype(str) + "-W" + df["epi_week_number"].astype(str).str.zfill(2)
    add("epi_week label agrees with epi_year and epi_week_number", bool((label == df["epi_week"]).all()))

    key = df["epi_year"] * 100 + df["epi_week_number"]
    add("Chronologically ordered (year, week strictly increasing)", bool(key.is_monotonic_increasing and key.is_unique))

    broken = [int(y) for y, g in df.groupby("epi_year")
              if not (g["epi_week_number"].to_numpy() == np.arange(1, len(g) + 1)).all()]
    add("Years whose weeks are not contiguous from 1", len(broken), str(broken))
    add("Years covered", f"{df['epi_year'].min()}-{df['epi_year'].max()}",
        f"{df['epi_year'].nunique()} years, none missing: "
        f"{sorted(df['epi_year'].unique()) == list(range(df['epi_year'].min(), df['epi_year'].max() + 1))}")

    # --- Annual counts versus each candidate convention -----------------------
    annual = df.groupby("epi_year").agg(
        observed_weeks=("epi_week_number", "count"),
        first_week=("epi_week_number", "min"),
        last_week=("epi_week_number", "max"),
    ).reset_index()
    for name, rule in CONVENTIONS.items():
        annual[name] = [weeks_in_year(rule, int(y)) for y in annual["epi_year"]]
    annual.to_csv(TABLES / "audit_annual_week_counts.csv", index=False)

    conv_rows = []
    for name in CONVENTIONS:
        mismatch = annual.loc[annual["observed_weeks"] != annual[name], "epi_year"].tolist()
        conv_rows.append(dict(convention=name, years_matching=len(annual) - len(mismatch),
                              years_total=len(annual), mismatching_years=str(mismatch)))
    conv = pd.DataFrame(conv_rows)
    conv.to_csv(TABLES / "audit_week_convention_match.csv", index=False)

    # --- Derived dates under the matching convention --------------------------
    starts = pd.to_datetime([week_start(int(y), int(w))
                             for y, w in zip(df["epi_year"], df["epi_week_number"])])
    gaps = pd.Series(starts).diff().dt.days.dropna()
    add("Derived week-start gaps (Sun-Sat, >=4-day rule)", str(gaps.value_counts().to_dict()),
        "all 7 days means the labels form one gapless weekly sequence")
    add("Derived week starts all fall on a Sunday", bool((starts.dayofweek == 6).all()))
    add("Derived date range (first week start -> last week end)",
        f"{starts[0].date()} -> {(starts[-1] + pd.Timedelta(days=6)).date()}")

    # Year boundaries: last week of each year and first week of the next.
    b_rows = []
    for y in sorted(df["epi_year"].unique())[:-1]:
        last = df[df["epi_year"] == y].iloc[-1]
        first = df[df["epi_year"] == y + 1].iloc[0]
        s_last = week_start(int(y), int(last["epi_week_number"]))
        s_first = week_start(int(y + 1), 1)
        b_rows.append(dict(
            last_week=last["epi_week"], last_week_dates=f"{s_last} to {s_last + dt.timedelta(days=6)}",
            last_week_cases=int(last[TARGET]),
            next_week=first["epi_week"], next_week_dates=f"{s_first} to {s_first + dt.timedelta(days=6)}",
            next_week_cases=int(first[TARGET]),
            days_between_starts=(s_first - s_last).days,
        ))
    pd.DataFrame(b_rows).to_csv(TABLES / "audit_year_boundaries.csv", index=False)

    # Week 53 rows with their neighbours, to check they are ordinary weeks.
    w53 = []
    for i in df.index[df["epi_week_number"] == 53]:
        for j in (i - 1, i, i + 1):
            r = df.loc[j]
            s = week_start(int(r["epi_year"]), int(r["epi_week_number"]))
            w53.append(dict(time_index=int(r["time_index"]), epi_week=r["epi_week"],
                            derived_dates=f"{s} to {s + dt.timedelta(days=6)}", cases=int(r[TARGET])))
    pd.DataFrame(w53).to_csv(TABLES / "audit_week53_context.csv", index=False)

    out = pd.DataFrame(checks)
    out.to_csv(TABLES / "audit_checks.csv", index=False)

    pd.set_option("display.width", 220, "display.max_colwidth", 90)
    print(out.to_string(index=False))
    print("\nAnnual week counts vs conventions:\n", annual.to_string(index=False))
    print("\nConvention match:\n", conv.to_string(index=False))
    print("\nYear boundaries:\n", pd.DataFrame(b_rows).to_string(index=False))
    print("\nWeek 53 context:\n", pd.DataFrame(w53).to_string(index=False))


if __name__ == "__main__":
    main()
