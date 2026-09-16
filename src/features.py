"""
Turn a time series into a supervised learning problem using LAG FEATURES.

THE CORE IDEA
-------------
A model like XGBoost does not understand "time". It only understands
"here is a row of numbers X, predict y". So we reshape the series into
rows, where each row's inputs are the previous W weeks and the target is
the following week.

With a 4-week window, the row for time t is:

    X = [cases(t-3), cases(t-2), cases(t-1), cases(t)]      -> lag_4..lag_1
    y =  cases(t+1)

W (the window) is the experimental variable of this whole study.

NAMING CONVENTION (important, and a classic viva question)
----------------------------------------------------------
`lag_1` = the MOST RECENT week (t), `lag_2` = the week before it (t-1),
and so on. So lag_k means "k steps before the target". Under this
convention a window of size W always uses lag_1 .. lag_W, and the
features of a small window are a strict subset of a larger one. That is
what makes the 1 / 2 / 4 / 8 / 12 comparison a fair, nested comparison:
the ONLY thing changing is how far back the model can see.
"""

import pandas as pd

HISTORY_WINDOWS = [1, 2, 4, 8, 12]


def make_lag_features(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Build the supervised table for a given history window.

    Returns columns: date, lag_1..lag_window, target.
    `date` is the date of week t -- the most recent INPUT week. The target
    is the case count of the following week, t+1.

    Rows where any lag is undefined (the first `window` weeks, which have
    no complete history behind them) are dropped. This is why a longer
    window costs you training rows -- a real trade-off this study measures.
    """
    if window < 1:
        raise ValueError("window must be >= 1")

    s = df["cases"]
    out = pd.DataFrame({"date": df["date"]})

    # shift(k) moves values FORWARD in time, so row t receives the value
    # from row t-k. Shifting is always backward-looking: it can only ever
    # pull in the past. That is what makes this leakage-safe.
    for k in range(1, window + 1):
        out[f"lag_{k}"] = s.shift(k - 1)

    # shift(-1) pulls the NEXT week's value in as the target.
    out["target"] = s.shift(-1)

    return out.dropna().reset_index(drop=True)


def build_aligned_tables(df: pd.DataFrame, windows=HISTORY_WINDOWS) -> dict:
    """
    Build one supervised table per window, ALL SHARING THE SAME ROWS.

    WHY THIS MATTERS (this is an experimental-design fix, not a detail)
    -------------------------------------------------------------------
    A 1-week window yields 935 usable rows; a 12-week window yields only
    924, because the long window needs 12 weeks of run-up before its first
    complete row. If each window were evaluated on its own table, the
    windows would be scored on DIFFERENT test weeks -- and any difference in
    error could then be caused by the different test weeks rather than by
    the history length. That would confound the only variable we are
    studying.

    So we restrict every window to the rows the LONGEST window can support.
    After this, all five windows are trained and tested on exactly the same
    target weeks, and the only difference between them is how many past
    weeks each model is allowed to look at.
    """
    tables = {w: make_lag_features(df, w) for w in windows}
    common_dates = set(tables[max(windows)]["date"])
    return {
        w: t[t["date"].isin(common_dates)].reset_index(drop=True)
        for w, t in tables.items()
    }


def split_X_y(table: pd.DataFrame):
    """Separate the feature matrix from the target vector."""
    feature_cols = [c for c in table.columns if c.startswith("lag_")]
    return table[feature_cols], table["target"], feature_cols


if __name__ == "__main__":
    from data_loader import load_raw
    from preprocessing import preprocess

    clean = preprocess(load_raw("sj"))
    print("Original series (first 8 weeks):")
    print(clean.head(8).to_string(index=False))

    print("\n--- 4-week window, first 3 supervised rows ---")
    print(make_lag_features(clean, 4).head(3).to_string(index=False))

    print("\n--- usable rows per window, BEFORE alignment ---")
    for w in HISTORY_WINDOWS:
        print(f"  window={w:2d}  ->  {len(make_lag_features(clean, w))} supervised rows")

    print("\n--- AFTER aligning all windows to common target weeks ---")
    aligned = build_aligned_tables(clean)
    for w, t in aligned.items():
        print(f"  window={w:2d}  ->  {len(t)} rows | "
              f"{t['date'].min().date()} -> {t['date'].max().date()}")
    date_sets = [tuple(t["date"]) for t in aligned.values()]
    print(f"  all windows cover identical weeks: {all(d == date_sets[0] for d in date_sets)}")
