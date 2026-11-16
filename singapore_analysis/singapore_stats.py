"""
Fold-level statistics for the Singapore external validation (Phase 6).

These functions take the long CV table produced by the UNMODIFIED
src/experiment.run_cv_experiment() and add what the task asks for beyond the
original study: medians, confidence intervals and paired tests.

WHAT IS AN ADDITION RELATIVE TO THE ORIGINAL STUDY
--------------------------------------------------
The original reported mean, standard deviation, fold win counts and
descriptive paired differences. It reported no medians, no confidence
intervals and no hypothesis tests. Everything in this module beyond
mean/std is therefore new, and is labelled as such in the protocol.

CAVEAT THAT APPLIES TO EVERY INTERVAL AND P-VALUE HERE
------------------------------------------------------
Folds are consecutive blocks of one time series with a growing, shared
training set. They are not independent samples. Intervals and p-values are
computed as if they were, so they are likely too narrow / too small.
"""

import numpy as np
import pandas as pd
from scipy import stats

BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 42
CONFIDENCE = 0.95
PERSISTENCE = "Persistence (baseline)"
SHORT_WINDOWS = (1, 2, 4)
LONG_WINDOWS = (8, 12)


def _t_interval(values, confidence=CONFIDENCE):
    v = np.asarray(values, dtype=float)
    n = len(v)
    if n < 2:
        return np.nan, np.nan
    half = stats.t.ppf(0.5 + confidence / 2, n - 1) * v.std(ddof=1) / np.sqrt(n)
    return v.mean() - half, v.mean() + half


def _bootstrap_interval(values, confidence=CONFIDENCE):
    """Percentile bootstrap of the mean, resampling folds with replacement."""
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    means = v[rng.integers(0, len(v), size=(BOOTSTRAP_RESAMPLES, len(v)))].mean(axis=1)
    lo, hi = np.quantile(means, [(1 - confidence) / 2, 0.5 + confidence / 2])
    return float(lo), float(hi)


def _wilcoxon_p(diffs):
    """Two-sided Wilcoxon signed-rank p-value; NaN if every difference is zero."""
    d = np.asarray(diffs, dtype=float)
    if np.allclose(d, 0):
        return np.nan
    return float(stats.wilcoxon(d, alternative="two-sided").pvalue)


def holm_adjust(pvalues):
    """Holm step-down adjusted p-values. NaNs are left as NaN and not counted."""
    p = np.asarray(pvalues, dtype=float)
    out = np.full(len(p), np.nan)
    idx = np.where(~np.isnan(p))[0]
    m = len(idx)
    order = idx[np.argsort(p[idx])]
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        out[i] = running
    return out


def extended_cv_summary(cv: pd.DataFrame) -> pd.DataFrame:
    """Mean, median, std, range and a t-interval of the mean, per window and model."""
    rows = []
    for (window, model), g in cv.groupby(["history_window", "model"]):
        row = dict(history_window=int(window), model=model, n_folds=g["fold"].nunique())
        for metric in ("MAE", "RMSE"):
            v = g[metric].to_numpy()
            lo, hi = _t_interval(v)
            row.update({
                f"{metric}_mean": v.mean(), f"{metric}_median": float(np.median(v)),
                f"{metric}_std": v.std(ddof=1), f"{metric}_min": v.min(), f"{metric}_max": v.max(),
                f"{metric}_mean_ci95_low": lo, f"{metric}_mean_ci95_high": hi,
            })
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["model", "history_window"]).reset_index(drop=True)


def _describe_pair(a, b, label_a, label_b, metric):
    """Paired summary of a - b across folds. Negative means `a` had lower error."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    d = a - b
    t_lo, t_hi = _t_interval(d)
    b_lo, b_hi = _bootstrap_interval(d)
    return dict(
        a=label_a, b=label_b, metric=metric, n_folds=len(d),
        mean_a=a.mean(), mean_b=b.mean(),
        mean_diff=d.mean(), median_diff=float(np.median(d)), std_diff=d.std(ddof=1),
        mean_diff_pct_of_b=100 * d.mean() / b.mean(),
        t_ci95_low=t_lo, t_ci95_high=t_hi, bootstrap_ci95_low=b_lo, bootstrap_ci95_high=b_hi,
        folds_a_better=int((d < 0).sum()), folds_b_better=int((d > 0).sum()), ties=int((d == 0).sum()),
        wilcoxon_p=_wilcoxon_p(d),
    )


def window_pair_comparisons(cv: pd.DataFrame, metric="MAE") -> pd.DataFrame:
    """Every pair of windows, per model (shorter window listed first as `a`)."""
    rows = []
    models = [m for m in cv["model"].unique() if m != PERSISTENCE]
    for model in models:
        pivot = cv[cv["model"] == model].pivot(index="fold", columns="history_window", values=metric)
        windows = sorted(pivot.columns)
        for i, wa in enumerate(windows):
            for wb in windows[i + 1:]:
                rows.append(dict(model=model, family="secondary: window pairs",
                                 **_describe_pair(pivot[wa], pivot[wb], f"window {wa}", f"window {wb}", metric)))
    return pd.DataFrame(rows)


def short_vs_long_contrast(cv: pd.DataFrame, metric="MAE") -> pd.DataFrame:
    """PRIMARY contrast, one per model: per-fold mean error of the short
    windows {1, 2, 4} minus per-fold mean error of the long windows {8, 12}."""
    rows = []
    for model in [m for m in cv["model"].unique() if m != PERSISTENCE]:
        pivot = cv[cv["model"] == model].pivot(index="fold", columns="history_window", values=metric)
        if not set(SHORT_WINDOWS + LONG_WINDOWS) <= set(pivot.columns):
            continue
        short = pivot[list(SHORT_WINDOWS)].mean(axis=1)
        long_ = pivot[list(LONG_WINDOWS)].mean(axis=1)
        rows.append(dict(model=model, family="primary: short vs long",
                         **_describe_pair(short, long_, "mean of windows 1,2,4", "mean of windows 8,12", metric)))
    return pd.DataFrame(rows)


def persistence_comparisons(cv: pd.DataFrame, metric="MAE") -> pd.DataFrame:
    """Each model and window against persistence on the same folds."""
    base = cv[cv["model"] == PERSISTENCE].groupby("fold")[metric].first()
    rows = []
    for model in [m for m in cv["model"].unique() if m != PERSISTENCE]:
        pivot = cv[cv["model"] == model].pivot(index="fold", columns="history_window", values=metric)
        for w in sorted(pivot.columns):
            rows.append(dict(model=model, family="secondary: vs persistence",
                             **_describe_pair(pivot[w], base.loc[pivot.index], f"window {w}", "persistence", metric)))
    return pd.DataFrame(rows)


def all_comparisons(cv: pd.DataFrame, metric="MAE") -> pd.DataFrame:
    """All three families, with Holm adjustment applied WITHIN each family."""
    parts = [short_vs_long_contrast(cv, metric), window_pair_comparisons(cv, metric),
             persistence_comparisons(cv, metric)]
    out = pd.concat([p for p in parts if len(p)], ignore_index=True)
    out["wilcoxon_p_holm"] = np.nan
    for _, idx in out.groupby("family").groups.items():
        out.loc[idx, "wilcoxon_p_holm"] = holm_adjust(out.loc[idx, "wilcoxon_p"].to_numpy())
    out["min_attainable_two_sided_p"] = 2.0 / (2.0 ** out["n_folds"])
    return out


def near_optimal_table(summary: pd.DataFrame, tolerance: float) -> pd.DataFrame:
    """The 5% rule, shown window by window (the rule itself is unchanged)."""
    rows = []
    for model, g in summary.groupby("model"):
        best = g["MAE_mean"].min()
        for r in g.sort_values("history_window").itertuples():
            rows.append(dict(model=model, history_window=int(r.history_window), MAE_mean=r.MAE_mean,
                             best_MAE_mean=best, pct_above_best=100 * (r.MAE_mean / best - 1),
                             threshold=best * (1 + tolerance),
                             near_optimal=bool(r.MAE_mean <= best * (1 + tolerance))))
    return pd.DataFrame(rows)
