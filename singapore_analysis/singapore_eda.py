"""
Phase 4 exploratory data analysis for the Singapore dengue fever series.

Descriptive only. No model is trained here. Nothing in data/ or results/ is
written. Outputs go to singapore_analysis/figures/ and singapore_analysis/tables/.

Run with:  .venv/bin/python singapore_analysis/singapore_eda.py

Definitions used in this script (all descriptive, fixed before looking at
which weeks they select, and documented in singapore_eda_report.md):

  HIGH-ACTIVITY THRESHOLD  75th percentile of the full series.
  OUTBREAK PERIOD          a run of >= 8 consecutive weeks above that threshold.
                           Runs separated by <= 2 weeks below it are merged.
  EXTREME OBSERVATION      above Q3 + 1.5 * IQR (Tukey's upper fence).
"""

import datetime as dt
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CSV = ROOT / "data" / "raw" / "singapore_dengue_fever_weekly_clean.csv"
FIG = HERE / "figures"
TAB = HERE / "tables"
TARGET = "dengue_fever_cases"

# Same palette as src/visualization.py, so figures read as one project.
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#8a8984"
TEXT_PRIMARY, TEXT_SECONDARY, GRID, BG = "#0b0b0b", "#52514e", "#dcdbd6", "#fcfcfb"

MAX_WINDOW = 12          # longest history window in the study
TRAIN_FRAC = 0.70        # CV_TRAIN_FRAC in src/experiment.py
OUTBREAK_MIN_WEEKS = 8
OUTBREAK_MERGE_GAP = 2
MAX_LAG = 104


# --- helpers ------------------------------------------------------------------
def week_start(year, week):
    """Sunday start of an epi week: Sunday-Saturday weeks, week 1 has >= 4 days
    in January. This rule reproduces every year in the file (see the audit).
    Dates are used for plot axes and labels only, never as model inputs."""
    jan4 = dt.date(year, 1, 4)
    w1 = jan4 - dt.timedelta(days=(jan4.weekday() + 1) % 7)
    return w1 + dt.timedelta(days=7 * (week - 1))


def load():
    df = pd.read_csv(CSV)
    df["week_start"] = pd.to_datetime(
        [week_start(int(y), int(w)) for y, w in zip(df["epi_year"], df["epi_week_number"])])
    assert (df["week_start"].diff().dropna().dt.days == 7).all(), "series is not gapless"
    return df


def style(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=12, color=TEXT_PRIMARY, loc="left", fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, fontsize=10, color=TEXT_SECONDARY)
    ax.set_ylabel(ylabel, fontsize=10, color=TEXT_SECONDARY)
    ax.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.7)
    ax.set_axisbelow(True)
    ax.set_facecolor(BG)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)


def save(fig, name):
    fig.patch.set_facecolor(BG)
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=150)
    plt.close(fig)
    print(f"  wrote figures/{name}")


def acf(x, max_lag):
    x = np.asarray(x, dtype=float) - np.mean(x)
    denom = np.sum(x * x)
    return np.array([1.0] + [np.sum(x[k:] * x[:-k]) / denom for k in range(1, max_lag + 1)])


def pacf_from_acf(r, max_lag):
    """Partial autocorrelation by the Durbin-Levinson recursion."""
    pac = np.zeros(max_lag + 1)
    pac[0] = 1.0
    phi_prev = np.array([])
    for k in range(1, max_lag + 1):
        if k == 1:
            phi_kk = r[1]
            phi = np.array([phi_kk])
        else:
            num = r[k] - np.sum(phi_prev * r[k - 1:0:-1])
            den = 1.0 - np.sum(phi_prev * r[1:k])
            phi_kk = num / den
            phi = np.append(phi_prev - phi_kk * phi_prev[::-1], phi_kk)
        pac[k] = phi_kk
        phi_prev = phi
    return pac


def best_mean_split(y, min_seg):
    """Single change point that most reduces squared error around segment means."""
    n = len(y)
    total = np.sum((y - y.mean()) ** 2)
    best = (None, 0.0)
    for k in range(min_seg, n - min_seg + 1):
        sse = np.sum((y[:k] - y[:k].mean()) ** 2) + np.sum((y[k:] - y[k:].mean()) ** 2)
        if total - sse > best[1]:
            best = (k, total - sse)
    return best


def binary_segmentation(y, n_breaks, min_seg):
    """Greedy mean-shift segmentation. Descriptive: it always returns n_breaks
    breaks whether or not the series truly has any, so the gain column matters."""
    segments, breaks = [(0, len(y))], []
    total = np.sum((y - y.mean()) ** 2)
    for _ in range(n_breaks):
        cand = None
        for (a, b) in segments:
            if b - a < 2 * min_seg:
                continue
            k, gain = best_mean_split(y[a:b], min_seg)
            if k is not None and (cand is None or gain > cand[2]):
                cand = (a, b, gain, a + k)
        if cand is None:
            break
        a, b, gain, k = cand
        segments.remove((a, b))
        segments += [(a, k), (k, b)]
        breaks.append(dict(order=len(breaks) + 1, position=k, sse_reduction_share=gain / total))
    return breaks, sorted(segments)


# --- analyses -------------------------------------------------------------------
def main():
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 220)

    df = load()
    y = df[TARGET].to_numpy(dtype=float)
    logy = np.log(y)                      # safe: minimum count is 24, no zeros
    n = len(df)

    # Where the planned evaluation period starts (protocol, for reference lines).
    n_aligned = n - MAX_WINDOW
    min_train = int(n_aligned * TRAIN_FRAC)
    first_test_target_pos = min_train + MAX_WINDOW        # 0-based row of first test target
    eval_start = df["week_start"].iloc[first_test_target_pos]
    eval_label = df["epi_week"].iloc[first_test_target_pos]

    # ---- 1. full time series --------------------------------------------------
    print("1. full time series")
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.6), sharex=True)
    for ax, vals, lab in ((axes[0], y, "Weekly cases"), (axes[1], y, "Weekly cases (log scale)")):
        ax.plot(df["week_start"], vals, color=BLUE, linewidth=1.3)
        ax.axvline(eval_start, color=GREY, linestyle="--", linewidth=1)
        style(ax, "", "", lab)
    axes[1].set_yscale("log")
    axes[0].set_title("Singapore weekly dengue fever cases, 2012-W01 to 2022-W52",
                      fontsize=12, color=TEXT_PRIMARY, loc="left", fontweight="bold", pad=10)
    axes[0].annotate(f"planned evaluation period starts {eval_label}",
                     xy=(eval_start, y.max() * 0.93), xytext=(-8, 0), textcoords="offset points",
                     ha="right", fontsize=9, color=TEXT_SECONDARY)
    axes[1].set_xlabel("Week start (Sunday), derived from epi-week label", fontsize=10, color=TEXT_SECONDARY)
    save(fig, "01_weekly_time_series.png")

    # ---- 2-3. annual totals and weekly averages --------------------------------
    print("2-3. annual tables")
    g = df.groupby("epi_year")[TARGET]
    annual = pd.DataFrame({
        "weeks": g.count(), "total_cases": g.sum(), "weekly_mean": g.mean(),
        "weekly_median": g.median(), "weekly_std": g.std(), "weekly_min": g.min(),
        "weekly_max": g.max(),
    })
    annual["cv"] = annual["weekly_std"] / annual["weekly_mean"]
    annual["peak_week"] = [df.loc[idx, "epi_week"] for idx in g.idxmax()]
    annual["share_of_all_cases_pct"] = 100 * annual["total_cases"] / annual["total_cases"].sum()
    annual = annual.reset_index()
    annual.to_csv(TAB / "eda_annual_summary.csv", index=False)
    print(annual.round(1).to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].bar(annual["epi_year"].astype(str), annual["total_cases"], color=BLUE)
    for x, v in zip(annual["epi_year"].astype(str), annual["total_cases"]):
        axes[0].text(x, v, f"{v:,.0f}", ha="center", va="bottom", fontsize=7.5, color=TEXT_SECONDARY)
    style(axes[0], "Annual dengue fever cases", "Epidemiological year", "Cases")
    axes[1].bar(annual["epi_year"].astype(str), annual["weekly_mean"], color=BLUE, label="Mean")
    axes[1].plot(annual["epi_year"].astype(str), annual["weekly_median"], color=ORANGE, marker="o",
                 linewidth=1.5, label="Median")
    style(axes[1], "Average weekly cases by year", "Epidemiological year", "Cases per week")
    axes[1].legend(frameon=False, fontsize=9)
    for ax in axes:
        ax.tick_params(axis="x", labelsize=8)
    save(fig, "02_annual_totals_and_weekly_averages.png")

    # ---- 4. distribution and descriptive statistics ----------------------------
    print("4. distribution")
    s = pd.Series(y)
    q = s.quantile([0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
    q1, q3 = q[0.25], q[0.75]
    iqr = q3 - q1
    fence = q3 + 1.5 * iqr
    desc = {
        "n": n, "mean": s.mean(), "std": s.std(), "min": s.min(), "max": s.max(),
        **{f"p{int(p * 100):02d}": v for p, v in q.items()},
        "iqr": iqr, "tukey_upper_fence": fence,
        "skewness": s.skew(), "excess_kurtosis": s.kurt(),
        "coefficient_of_variation": s.std() / s.mean(),
        "variance_to_mean_ratio": s.var() / s.mean(),
        "log_mean": logy.mean(), "log_std": logy.std(ddof=1),
        "log_skewness": pd.Series(logy).skew(), "log_excess_kurtosis": pd.Series(logy).kurt(),
        "share_of_cases_in_top_10pct_weeks_pct": 100 * s.nlargest(int(round(n * 0.10))).sum() / s.sum(),
    }
    pd.DataFrame({"statistic": list(desc), "value": list(desc.values())}).to_csv(
        TAB / "eda_descriptive_statistics.csv", index=False)
    for k, v in desc.items():
        print(f"  {k:40s} {v:12.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].hist(y, bins=40, color=BLUE, edgecolor=BG)
    axes[0].axvline(s.median(), color=ORANGE, linewidth=1.5, label=f"median {s.median():.0f}")
    axes[0].axvline(s.mean(), color=GREY, linestyle="--", linewidth=1.5, label=f"mean {s.mean():.0f}")
    style(axes[0], "Distribution of weekly cases", "Cases per week", "Number of weeks")
    axes[0].legend(frameon=False, fontsize=9)
    axes[1].hist(logy, bins=40, color=BLUE, edgecolor=BG)
    style(axes[1], "Distribution of log weekly cases", "ln(cases per week)", "Number of weeks")
    save(fig, "03_distribution.png")

    # ---- 5. rolling averages ----------------------------------------------------
    print("5. rolling averages")
    roll = pd.DataFrame({"time_index": df["time_index"], "epi_week": df["epi_week"], TARGET: y})
    for w in (4, 13, 52):
        roll[f"rolling_mean_{w}w"] = s.rolling(w).mean()      # trailing: past weeks only
    roll["rolling_std_52w"] = s.rolling(52).std()
    roll["rolling_cv_52w"] = roll["rolling_std_52w"] / roll["rolling_mean_52w"]
    roll.to_csv(TAB / "eda_rolling_statistics.csv", index=False)

    fig, axes = plt.subplots(2, 1, figsize=(11, 6.6), sharex=True)
    axes[0].plot(df["week_start"], y, color=GRID, linewidth=1, label="Weekly")
    axes[0].plot(df["week_start"], roll["rolling_mean_4w"], color=GREY, linewidth=1, label="4-week mean")
    axes[0].plot(df["week_start"], roll["rolling_mean_13w"], color=BLUE, linewidth=1.6, label="13-week mean")
    axes[0].plot(df["week_start"], roll["rolling_mean_52w"], color=ORANGE, linewidth=1.8, label="52-week mean")
    style(axes[0], "Trailing rolling means", "", "Cases per week")
    axes[0].legend(frameon=False, fontsize=9, ncol=4, loc="upper left")
    axes[1].plot(df["week_start"], roll["rolling_std_52w"], color=BLUE, linewidth=1.6)
    style(axes[1], "Trailing 52-week standard deviation", "Week start (Sunday)", "Cases")
    for ax in axes:
        ax.axvline(eval_start, color=GREY, linestyle="--", linewidth=1)
    save(fig, "04_rolling_averages.png")

    # ---- 6. autocorrelation -----------------------------------------------------
    print("6. autocorrelation")
    r_raw, r_log = acf(y, MAX_LAG), acf(logy, MAX_LAG)
    r_diff, r_dlog = acf(np.diff(y), MAX_LAG), acf(np.diff(logy), MAX_LAG)
    p_raw, p_log = pacf_from_acf(r_raw, 26), pacf_from_acf(r_log, 26)
    acf_tab = pd.DataFrame({"lag": np.arange(MAX_LAG + 1), "acf_cases": r_raw, "acf_log_cases": r_log,
                            "acf_diff_cases": r_diff, "acf_diff_log_cases": r_dlog})
    acf_tab["pacf_cases"] = np.append(p_raw, [np.nan] * (MAX_LAG - 26))
    acf_tab["pacf_log_cases"] = np.append(p_log, [np.nan] * (MAX_LAG - 26))
    acf_tab.to_csv(TAB / "eda_autocorrelation.csv", index=False)
    band = 1.96 / np.sqrt(n)
    print(acf_tab.loc[[1, 2, 4, 8, 12, 13, 26, 39, 52, 78, 104]].round(3).to_string(index=False))
    print(f"  white-noise 95% band: +/-{band:.3f}")
    print("  pacf (cases) lags 1-13:", np.round(p_raw[1:14], 3))
    print("  pacf (log)   lags 1-13:", np.round(p_log[1:14], 3))

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    lags = np.arange(MAX_LAG + 1)
    for ax, r, title in ((axes[0, 0], r_raw, "Autocorrelation, weekly cases"),
                         (axes[0, 1], r_diff, "Autocorrelation, week-to-week change")):
        ax.bar(lags[1:], r[1:], color=BLUE, width=0.8)
        ax.axhline(band, color=GREY, linestyle="--", linewidth=1)
        ax.axhline(-band, color=GREY, linestyle="--", linewidth=1)
        for w in (1, 2, 4, 8, 12):
            ax.bar(w, r[w], color=ORANGE, width=0.8)
        style(ax, title, "Lag (weeks)", "Correlation")
    for ax, p, title in ((axes[1, 0], p_raw, "Partial autocorrelation, weekly cases"),
                         (axes[1, 1], p_log, "Partial autocorrelation, log weekly cases")):
        ax.bar(np.arange(1, 27), p[1:], color=BLUE, width=0.7)
        for w in (1, 2, 4, 8, 12):
            ax.bar(w, p[w], color=ORANGE, width=0.7)
        ax.axhline(band, color=GREY, linestyle="--", linewidth=1)
        ax.axhline(-band, color=GREY, linestyle="--", linewidth=1)
        style(ax, title, "Lag (weeks)", "Partial correlation")
    fig.suptitle("Orange bars mark the study's window lengths (1, 2, 4, 8, 12). Dashed: white-noise 95% band.",
                 fontsize=9, color=TEXT_SECONDARY, x=0.01, ha="left", y=0.995)
    save(fig, "05_autocorrelation.png")

    # ---- 7. seasonality ---------------------------------------------------------
    print("7. seasonality")
    wk = df[df["epi_week_number"] <= 52].copy()          # week 53 exists in only 2 years
    wk["log_cases"] = np.log(wk[TARGET])
    wk["share_of_year_pct"] = 100 * wk[TARGET] / wk.groupby("epi_year")[TARGET].transform("sum")
    wk["ratio_to_year_mean"] = wk[TARGET] / wk.groupby("epi_year")[TARGET].transform("mean")
    seas = wk.groupby("epi_week_number").agg(
        mean_cases=(TARGET, "mean"), median_cases=(TARGET, "median"),
        min_cases=(TARGET, "min"), max_cases=(TARGET, "max"),
        mean_ratio_to_year_mean=("ratio_to_year_mean", "mean"),
        median_ratio_to_year_mean=("ratio_to_year_mean", "median"),
    ).reset_index()
    seas.to_csv(TAB / "eda_seasonal_profile_by_week.csv", index=False)

    def r2_of_groups(values, groups):
        v = pd.Series(np.asarray(values, dtype=float))
        fitted = v.groupby(np.asarray(groups)).transform("mean")
        return 1 - np.sum((v - fitted) ** 2) / np.sum((v - v.mean()) ** 2)

    yr_dm_log = wk["log_cases"] - wk.groupby("epi_year")["log_cases"].transform("mean")
    seas_stats = {
        "r2_week_of_year_on_cases": r2_of_groups(wk[TARGET], wk["epi_week_number"]),
        "r2_year_on_cases": r2_of_groups(wk[TARGET], wk["epi_year"]),
        "r2_week_of_year_on_log_cases": r2_of_groups(wk["log_cases"], wk["epi_week_number"]),
        "r2_year_on_log_cases": r2_of_groups(wk["log_cases"], wk["epi_year"]),
        "r2_week_of_year_on_year_demeaned_log_cases": r2_of_groups(yr_dm_log, wk["epi_week_number"]),
        "peak_week_of_mean_profile": int(seas.loc[seas["mean_ratio_to_year_mean"].idxmax(), "epi_week_number"]),
        "trough_week_of_mean_profile": int(seas.loc[seas["mean_ratio_to_year_mean"].idxmin(), "epi_week_number"]),
        "peak_to_trough_ratio_of_mean_profile":
            seas["mean_ratio_to_year_mean"].max() / seas["mean_ratio_to_year_mean"].min(),
    }
    q_of_week = np.minimum((wk["epi_week_number"] - 1) // 13 + 1, 4)
    quarter = wk.groupby(q_of_week)[TARGET].sum()
    for k, v in quarter.items():
        seas_stats[f"share_of_cases_weeks_{(k - 1) * 13 + 1}_{k * 13}_pct"] = 100 * v / quarter.sum()
    peak_weeks = annual[["epi_year", "peak_week", "weekly_max"]].copy()
    peak_weeks["peak_week_number"] = peak_weeks["peak_week"].str[-2:].astype(int)
    seas_stats["peak_week_number_min"] = int(peak_weeks["peak_week_number"].min())
    seas_stats["peak_week_number_max"] = int(peak_weeks["peak_week_number"].max())
    seas_stats["years_peaking_in_weeks_23_to_35"] = int(peak_weeks["peak_week_number"].between(23, 35).sum())
    pd.DataFrame({"statistic": list(seas_stats), "value": list(seas_stats.values())}).to_csv(
        TAB / "eda_seasonality_statistics.csv", index=False)
    for k, v in seas_stats.items():
        print(f"  {k:48s} {v:10.3f}")
    print(peak_weeks.to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1, 1.15]})
    for yr, grp in wk.groupby("epi_year"):
        axes[0].plot(grp["epi_week_number"], grp["ratio_to_year_mean"], color=GRID, linewidth=1)
    axes[0].plot(seas["epi_week_number"], seas["mean_ratio_to_year_mean"], color=BLUE, linewidth=2,
                 label="Mean across years")
    axes[0].plot(seas["epi_week_number"], seas["median_ratio_to_year_mean"], color=ORANGE, linewidth=1.6,
                 label="Median across years")
    axes[0].axhline(1, color=GREY, linestyle="--", linewidth=1)
    style(axes[0], "Seasonal profile (each year scaled to its own mean)", "Epidemiological week",
          "Cases / that year's weekly mean")
    axes[0].legend(frameon=False, fontsize=9)
    heat = df.pivot(index="epi_year", columns="epi_week_number", values=TARGET)
    im = axes[1].imshow(np.log10(heat.to_numpy()), aspect="auto", cmap="Blues", interpolation="nearest")
    axes[1].set_yticks(range(len(heat.index)))
    axes[1].set_yticklabels(heat.index, fontsize=8, color=TEXT_SECONDARY)
    axes[1].set_xticks(range(0, 53, 4))
    axes[1].set_xticklabels(range(1, 54, 4), fontsize=8, color=TEXT_SECONDARY)
    axes[1].set_title("Weekly cases by year and week (log10 colour scale)", fontsize=12,
                      color=TEXT_PRIMARY, loc="left", fontweight="bold", pad=10)
    axes[1].set_xlabel("Epidemiological week (week 53 exists only in 2014 and 2020)",
                       fontsize=10, color=TEXT_SECONDARY)
    cb = fig.colorbar(im, ax=axes[1], fraction=0.04, pad=0.02)
    cb.set_ticks([np.log10(v) for v in (30, 100, 300, 1000)])
    cb.set_ticklabels(["30", "100", "300", "1,000"])
    cb.ax.tick_params(labelsize=8, colors=TEXT_SECONDARY)
    save(fig, "06_seasonality.png")

    # ---- 8. outbreak periods ------------------------------------------------------
    print("8. outbreak periods")
    thr = q3
    above = y > thr
    runs, start = [], None
    for i, flag in enumerate(above):
        if flag and start is None:
            start = i
        if not flag and start is not None:
            runs.append([start, i - 1])
            start = None
    if start is not None:
        runs.append([start, n - 1])
    merged = []
    for a, b in runs:
        if merged and a - merged[-1][1] - 1 <= OUTBREAK_MERGE_GAP:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    outbreaks = []
    for a, b in merged:
        if b - a + 1 < OUTBREAK_MIN_WEEKS:
            continue
        seg = df.iloc[a:b + 1]
        pk = seg[TARGET].idxmax()
        outbreaks.append(dict(
            start_week=df.loc[a, "epi_week"], end_week=df.loc[b, "epi_week"], duration_weeks=b - a + 1,
            peak_week=df.loc[pk, "epi_week"], peak_cases=int(df.loc[pk, TARGET]),
            total_cases=int(seg[TARGET].sum()), mean_weekly_cases=seg[TARGET].mean(),
            weeks_start_to_peak=int(pk - a), weeks_peak_to_end=int(b - pk),
            share_of_all_cases_pct=100 * seg[TARGET].sum() / y.sum(),
            start_time_index=int(df.loc[a, "time_index"]), end_time_index=int(df.loc[b, "time_index"]),
            in_planned_evaluation_period=bool(b >= first_test_target_pos),
        ))
    outbreaks = pd.DataFrame(outbreaks)
    outbreaks.to_csv(TAB / "eda_outbreak_periods.csv", index=False)
    print(f"  threshold (75th percentile) = {thr:.0f}; weeks above = {int(above.sum())}")
    print(outbreaks.round(1).to_string(index=False))
    short_runs = [(df.loc[a, "epi_week"], df.loc[b, "epi_week"], b - a + 1)
                  for a, b in merged if b - a + 1 < OUTBREAK_MIN_WEEKS]
    print("  runs above threshold shorter than the minimum duration:", short_runs)

    fig, ax = plt.subplots(figsize=(11, 4.4))
    ax.plot(df["week_start"], y, color=BLUE, linewidth=1.3)
    ax.axhline(thr, color=GREY, linestyle="--", linewidth=1)
    ax.axhline(fence, color=ORANGE, linestyle=":", linewidth=1.2)
    for r in outbreaks.itertuples():
        a, b = r.start_time_index - 1, r.end_time_index - 1
        ax.axvspan(df["week_start"].iloc[a], df["week_start"].iloc[b], color=ORANGE, alpha=0.13, linewidth=0)
        ax.annotate(f"{r.peak_week}\n{r.peak_cases:,}", xy=(df["week_start"].iloc[a + r.weeks_start_to_peak],
                    r.peak_cases), xytext=(0, 5), textcoords="offset points", ha="center",
                    fontsize=8, color=TEXT_SECONDARY)
    ax.axvline(eval_start, color=GREY, linestyle="--", linewidth=1)
    ax.set_ylim(0, y.max() * 1.15)
    style(ax, f"Outbreak periods: at least {OUTBREAK_MIN_WEEKS} consecutive weeks above the 75th percentile "
              f"({thr:.0f} cases)", "Week start (Sunday)", "Cases per week")
    ax.text(df["week_start"].iloc[232], fence, f" Tukey upper fence {fence:.0f}", va="bottom", fontsize=8,
            color=ORANGE)
    save(fig, "07_outbreak_periods.png")

    # ---- 9. extreme observations ----------------------------------------------------
    print("9. extreme observations")
    df["change"] = df[TARGET].diff()
    df["pct_change"] = 100 * df[TARGET].pct_change()
    df["log_change"] = np.log(df[TARGET]).diff()
    top = df.nlargest(15, TARGET)[["time_index", "epi_week", TARGET, "change", "pct_change"]]
    top.to_csv(TAB / "eda_top15_weeks.csv", index=False)
    ext = df[df[TARGET] > fence]
    ext_by_year = ext.groupby("epi_year")[TARGET].agg(["count", "min", "max"]).reset_index()
    ext_by_year.to_csv(TAB / "eda_extreme_weeks_by_year.csv", index=False)
    cols = ["time_index", "epi_week", TARGET, "change", "pct_change"]
    jumps = pd.concat([
        df.nlargest(8, "change")[cols].assign(kind="largest absolute rise"),
        df.nsmallest(8, "change")[cols].assign(kind="largest absolute fall"),
        df.nlargest(8, "pct_change")[cols].assign(kind="largest relative rise"),
        df.nsmallest(8, "pct_change")[cols].assign(kind="largest relative fall"),
    ])
    jumps.to_csv(TAB / "eda_largest_weekly_changes.csv", index=False)
    print(f"  Tukey upper fence = {fence:.1f}; weeks above = {len(ext)}")
    print(ext_by_year.to_string(index=False))
    print(top.round(1).to_string(index=False))
    print(jumps.round(1).to_string(index=False))

    change_stats = df.groupby("epi_year").agg(
        mean_abs_change=("change", lambda v: v.abs().mean()),
        median_abs_change=("change", lambda v: v.abs().median()),
        max_abs_change=("change", lambda v: v.abs().max()),
        mean_abs_pct_change=("pct_change", lambda v: v.abs().mean()),
        std_log_change=("log_change", "std"),
    ).reset_index()
    change_stats.to_csv(TAB / "eda_weekly_change_by_year.csv", index=False)
    overall_change = dict(
        mean_abs_change=df["change"].abs().mean(), median_abs_change=df["change"].abs().median(),
        rmse_of_change=float(np.sqrt((df["change"] ** 2).mean())),
        mean_abs_pct_change=df["pct_change"].abs().mean(), std_log_change=df["log_change"].std(),
        share_of_weeks_rising_pct=100 * (df["change"] > 0).sum() / df["change"].notna().sum(),
    )
    print("  week-to-week change, whole series:", {k: round(v, 3) for k, v in overall_change.items()})
    print(change_stats.round(2).to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].bar(df["week_start"], df["change"], color=BLUE, width=7)
    axes[0].axvline(eval_start, color=GREY, linestyle="--", linewidth=1)
    style(axes[0], "Week-to-week change in cases", "Week start (Sunday)", "Change from previous week")
    axes[1].bar(change_stats["epi_year"].astype(str), change_stats["mean_abs_change"], color=BLUE)
    style(axes[1], "Mean absolute week-to-week change by year", "Epidemiological year", "Cases")
    axes[1].tick_params(axis="x", labelsize=8)
    save(fig, "08_weekly_changes.png")

    # ---- 10. structural breaks --------------------------------------------------------
    print("10. structural breaks")
    breaks, segments = binary_segmentation(logy, n_breaks=5, min_seg=26)
    for b in breaks:
        b["first_week_of_new_segment"] = df["epi_week"].iloc[b["position"]]
    pd.DataFrame(breaks).to_csv(TAB / "eda_structural_break_candidates.csv", index=False)
    seg_rows = []
    for a, b in segments:
        seg_rows.append(dict(start_week=df["epi_week"].iloc[a], end_week=df["epi_week"].iloc[b - 1],
                             weeks=b - a, mean_cases=y[a:b].mean(), median_cases=np.median(y[a:b]),
                             geometric_mean_cases=float(np.exp(logy[a:b].mean())),
                             std_cases=y[a:b].std(ddof=1)))
    seg_tab = pd.DataFrame(seg_rows)
    seg_tab.to_csv(TAB / "eda_structural_break_segments.csv", index=False)
    print(pd.DataFrame(breaks).round(3).to_string(index=False))
    print(seg_tab.round(1).to_string(index=False))

    # Regime comparison at the planned train/evaluation boundary (protocol-relevant).
    pre, post = df.iloc[:first_test_target_pos], df.iloc[first_test_target_pos:]
    def block(d, name):
        v = d[TARGET]
        return dict(period=name, first_week=d["epi_week"].iloc[0], last_week=d["epi_week"].iloc[-1],
                    weeks=len(d), mean=v.mean(), median=v.median(), std=v.std(), min=v.min(), max=v.max(),
                    p90=v.quantile(0.90), mean_abs_change=d["change"].abs().mean(),
                    std_log_change=d["log_change"].std(),
                    lag1_autocorrelation=float(acf(v.to_numpy(dtype=float), 1)[1]),
                    weeks_above_pre_period_max=int((v > pre[TARGET].max()).sum()))
    regime = pd.DataFrame([block(pre, "before planned evaluation period"),
                           block(post, "planned evaluation period"),
                           block(df[df["epi_year"] <= 2019], "2012-2019"),
                           block(df[df["epi_year"].isin([2020, 2021])], "2020-2021 (COVID-19 period)"),
                           block(df[df["epi_year"] == 2022], "2022")])
    regime.to_csv(TAB / "eda_regime_comparison.csv", index=False)
    print(regime.round(2).to_string(index=False))

    fig, axes = plt.subplots(2, 1, figsize=(11, 6.6), sharex=True)
    axes[0].plot(df["week_start"], y, color=BLUE, linewidth=1.2)
    for a, b in segments:
        axes[0].hlines(np.exp(logy[a:b].mean()), df["week_start"].iloc[a], df["week_start"].iloc[b - 1],
                       color=ORANGE, linewidth=2)
    axes[0].set_yscale("log")
    style(axes[0], "Mean-shift segmentation of log cases (orange: segment geometric mean)", "",
          "Cases per week (log scale)")
    axes[1].plot(df["week_start"], roll["rolling_cv_52w"], color=BLUE, linewidth=1.6)
    style(axes[1], "Trailing 52-week coefficient of variation", "Week start (Sunday)", "Std / mean")
    for ax in axes:
        ax.axvline(eval_start, color=GREY, linestyle="--", linewidth=1)
    save(fig, "09_structural_breaks.png")

    print(f"\nplanned evaluation period: first target week {eval_label} "
          f"(time_index {first_test_target_pos + 1}); aligned rows {n_aligned}; min_train {min_train}")
    print("done")


if __name__ == "__main__":
    main()
