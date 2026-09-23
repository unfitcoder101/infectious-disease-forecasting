"""
The experiment: history window x model grid.

PRE-REGISTERED DECISION RULE
----------------------------
The rule for "minimum sufficient history" is fixed HERE, in code, before
any results are looked at. Choosing the rule after seeing the numbers would
let you pick whichever threshold flatters your preferred answer.

    A window is NEAR-OPTIMAL if its test MAE is within TOLERANCE (relative)
    of the best test MAE across all windows for that model.
    The MINIMUM SUFFICIENT WINDOW is the SHORTEST near-optimal window.

TOLERANCE = 0.05 means "within 5% of the best MAE".
"""

import time
import numpy as np
import pandas as pd

from evaluation import chronological_split, evaluate, naive_baseline, rolling_origin_splits
from features import HISTORY_WINDOWS, build_aligned_tables, split_X_y
from models import RANDOM_SEED, get_models

TOLERANCE = 0.05
N_TIMING_REPEATS = 5

# --- Rolling-origin cross-validation configuration -------------------------
# Derived from the ACTUAL aligned row count for San Juan (n=924), not chosen
# blind. Same 70% floor as chronological_split(), so the CV experiment's
# starting point matches the single-split experiment's train size.
#
#   min_train  = int(924 * 0.70)        = 646 weeks
#   available  = 924 - 646              = 278 weeks left for testing
#   test_block = 26 weeks (~half a year), the only block size in the
#                requested 20-26 week range whose fold count lands in the
#                requested 8-10 fold target:
#                  278 // 26 = 10 folds   (278 // 20..25 all give 11-13 folds)
#   leftover   = 278 - 10*26 = 18 weeks, unused (too short for one more
#                whole block; dropped rather than padded)
CV_TRAIN_FRAC = 0.70
CV_TEST_BLOCK = 26
CV_FOLD_TARGET = (8, 10)

# --- Fine-grained follow-up CV: resolving 1 vs 2 vs 4 weeks -----------------
# A SEPARATE, narrower rolling-origin CV pass, run ALONGSIDE (never replacing)
# the CV_TRAIN_FRAC/CV_TEST_BLOCK config above. The committed 10-fold/26-week
# run already resolved "short (1-4) vs long (8-12)"; it could NOT resolve
# which of {1, 2, 4} is shortest-sufficient (those three sit within one
# fold-level std of each other). This config trades block length for more
# folds -- i.e. a smaller standard error on the mean -- specifically to
# sharpen THAT remaining question. Same aligned base (n=924), same min_train
# (646, same 70% floor) as the committed run; only test_block and the window
# set actually trained change.
#
#   available  = 924 - 646 = 278 weeks (identical to the committed run)
#   test_block = 13 weeks (one calendar quarter -- exactly half the original
#                26-week block: a principled halving, not an arbitrary
#                shrink, and still a recognizable epidemiological unit)
#   n_folds    = 278 // 13 = 21   (21*13=273 <= 278 < 22*13=286)
#   leftover   = 278 - 273 = 5 weeks, unused
# Windows 8 and 12 are already resolved as worse and are excluded here --
# the sole purpose of this pass is to separate {1, 2, 4}.
FINE_CV_TEST_BLOCK = 13
FINE_CV_WINDOWS = [1, 2, 4]

def _median_timed(fn, repeats=N_TIMING_REPEATS):
    """Run fn repeatedly, return (result_of_last_run, median seconds)."""
    times, result = [], None
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - t0)
    return result, float(np.median(times))

def run_experiment(series: pd.DataFrame, windows=HISTORY_WINDOWS) -> pd.DataFrame:
    """Train every model on every window; return the tidy results table."""
    tables = build_aligned_tables(series, windows)
    rows = []

    for window in windows:
        table = tables[window]
        train, val, test = chronological_split(table)

        X_tr, y_tr, feat_cols = split_X_y(train)
        X_va, y_va, _ = split_X_y(val)
        X_te, y_te, _ = split_X_y(test)

        # Persistence baseline, recorded once per window for reference.
        base = naive_baseline(test)
        rows.append(dict(
            history_window=window, model="Persistence (baseline)",
            **base, val_MAE=naive_baseline(val)["MAE"],
            training_time=0.0, inference_time=0.0,
            n_observations=len(table), n_train=len(train), n_test=len(test),
            n_features=1,
        ))

        for name, model in get_models().items():
            # Fit repeatedly and take the MEDIAN time: a single measurement is
            # dominated by noise from other processes on the machine.
            _, train_time = _median_timed(lambda: model.fit(X_tr, y_tr))
            preds, infer_time = _median_timed(lambda: model.predict(X_te))

            scores = evaluate(y_te, preds)
            rows.append(dict(
                history_window=window, model=name, **scores,
                val_MAE=evaluate(y_va, model.predict(X_va))["MAE"],
                training_time=train_time, inference_time=infer_time,
                n_observations=len(table), n_train=len(train), n_test=len(test),
                n_features=len(feat_cols),
            ))

    return pd.DataFrame(rows)

def run_cv_experiment(series: pd.DataFrame, windows=HISTORY_WINDOWS,
                       train_frac=CV_TRAIN_FRAC, test_block=CV_TEST_BLOCK,
                       align_windows=None) -> pd.DataFrame:
    """
    Rolling-origin cross-validation: same window x model grid as
    run_experiment(), but scored on a SEQUENCE of non-overlapping future
    folds instead of one fixed test slice.

    Returns a LONG table with one row per (fold, history_window, model),
    same metric columns as run_experiment()'s output, plus `fold`,
    `fold_test_start`, `fold_test_end`.

    The fold boundaries (min_train, test_block, n_folds) are computed from
    the ACTUAL aligned row count of `series`, not hardcoded, so this stays
    correct if run on a different city with a different series length --
    though see research_notes.md for a caveat about short series where the
    8-10 fold target cannot be met at this block size (e.g. Iquitos).

    align_windows: the window set used ONLY to compute the common-date
    alignment (build_aligned_tables), separate from `windows` (the windows
    actually trained/evaluated). Defaults to `windows` itself -- i.e. by
    default this behaves EXACTLY as before, byte-for-byte, since passing
    nothing here makes align_windows == windows, same as the old
    single-argument alignment call. Pass a WIDER set (e.g. the full
    HISTORY_WINDOWS) to align against the same 924-row date range as a
    different experiment even when `windows` itself is a narrower subset --
    this is how the fine-grained {1, 2, 4} follow-up keeps the identical
    aligned base as the committed 10-fold/26-week run instead of silently
    re-aligning to a shorter run-up (which would shrink to 932 rows if
    aligned against max({1,2,4})=4 alone).
    """
    align_windows = list(windows) if align_windows is None else list(align_windows)
    tables_full = build_aligned_tables(series, align_windows)
    tables = {w: tables_full[w] for w in windows}
    n = len(tables[windows[0]])          # identical length for every window
    min_train = int(n * train_frac)

    rows = []
    for window in windows:
        table = tables[window]
        for fold_idx, train, test in rolling_origin_splits(table, min_train, test_block):
            X_tr, y_tr, feat_cols = split_X_y(train)
            X_te, y_te, _ = split_X_y(test)

            base = naive_baseline(test)
            rows.append(dict(
                fold=fold_idx, history_window=window, model="Persistence (baseline)",
                **base, training_time=0.0, inference_time=0.0,
                n_train=len(train), n_test=len(test),
                fold_test_start=test["date"].min(), fold_test_end=test["date"].max(),
            ))

            for name, model in get_models().items():
                _, train_time = _median_timed(lambda: model.fit(X_tr, y_tr))
                preds, infer_time = _median_timed(lambda: model.predict(X_te))
                scores = evaluate(y_te, preds)
                rows.append(dict(
                    fold=fold_idx, history_window=window, model=name, **scores,
                    training_time=train_time, inference_time=infer_time,
                    n_train=len(train), n_test=len(test),
                    fold_test_start=test["date"].min(), fold_test_end=test["date"].max(),
                ))

    return pd.DataFrame(rows)

def cv_summary(cv_results: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-fold CV scores into mean +/- std per (history_window, model).

    This is the headline CV table: instead of a single MAE per window, each
    window now has a MEAN and a STANDARD DEVIATION across folds. The std is
    the whole point -- it tells us whether a difference between two windows'
    mean MAE is bigger than the fold-to-fold noise, or lost inside it.
    """
    agg = cv_results.groupby(["history_window", "model"]).agg(
        n_folds=("fold", "nunique"),
        MAE_mean=("MAE", "mean"), MAE_std=("MAE", "std"),
        RMSE_mean=("RMSE", "mean"), RMSE_std=("RMSE", "std"),
        R2_mean=("R2", "mean"), R2_std=("R2", "std"),
        training_time_mean=("training_time", "mean"),
        inference_time_mean=("inference_time", "mean"),
    ).reset_index()
    return agg.sort_values(["model", "history_window"]).reset_index(drop=True)

def fold_win_counts(cv_results: pd.DataFrame) -> pd.DataFrame:
    """
    RANKING STABILITY: for each fold and model, which history_window had the
    lowest MAE? Count how often each window "wins" across folds.

    This is a direct, honest test of whether the single-split conclusion
    ("window X is best") is a stable property of the data or just noise from
    picking one test slice. A window that wins 8/10 folds is a real signal;
    a near-even split across all five windows means the folds disagree with
    each other about which window is best, i.e. the ranking is not resolved.
    """
    model_rows = cv_results[~cv_results["model"].str.startswith("Persistence")]
    out = []
    for model in model_rows["model"].unique():
        sub = model_rows[model_rows["model"] == model]
        winners = sub.loc[sub.groupby("fold")["MAE"].idxmin(), ["fold", "history_window"]]
        counts = winners["history_window"].value_counts().reindex(
            sorted(cv_results["history_window"].unique()), fill_value=0
        )
        n_folds = sub["fold"].nunique()
        for window, wins in counts.items():
            out.append(dict(model=model, history_window=window,
                             folds_won=int(wins), n_folds=n_folds,
                             win_rate_pct=100 * wins / n_folds))
    return pd.DataFrame(out)

def cv_minimum_sufficient_window(summary: pd.DataFrame, tolerance=TOLERANCE) -> pd.DataFrame:
    """
    Same pre-registered near-optimal / minimum-sufficient rule as
    minimum_sufficient_window(), applied to CV MEAN MAE instead of a single
    test-set MAE. Also reports whether the shortest near-optimal window's
    mean MAE is within ONE STANDARD DEVIATION of the best window's mean --
    a simple, transparent check for "is this difference even distinguishable
    from fold-to-fold noise", not just "is it within X% on average".
    """
    out = []
    for model in summary["model"].unique():
        sub = summary[summary["model"] == model].sort_values("history_window")
        best_row = sub.loc[sub["MAE_mean"].idxmin()]
        best_mae, best_std = best_row["MAE_mean"], best_row["MAE_std"]
        best_window = int(best_row["history_window"])
        threshold = best_mae * (1 + tolerance)
        near = sub[sub["MAE_mean"] <= threshold]
        min_window_row = near.loc[near["history_window"].idxmin()]
        within_1std = bool(min_window_row["MAE_mean"] <= best_mae + best_std)
        out.append(dict(
            model=model, best_window=best_window, best_MAE_mean=best_mae, best_MAE_std=best_std,
            tolerance=tolerance, mae_threshold=threshold,
            minimum_sufficient_window=int(min_window_row["history_window"]),
            near_optimal_windows=", ".join(str(int(w)) for w in near["history_window"]),
            min_sufficient_within_1std_of_best=within_1std,
        ))
    return pd.DataFrame(out)

def paired_fold_differences(cv_results: pd.DataFrame, pairs=((1, 2), (1, 4), (2, 4))) -> pd.DataFrame:
    """
    PAIRED per-fold MAE differences between two history windows.

    WHY PAIRED, NOT INDEPENDENT
    ----------------------------
    Every window in a given CV run is scored on the SAME fold test-weeks
    (guaranteed here by run_cv_experiment's align_windows mechanism, which
    keeps all windows on one shared aligned base with identical positional
    fold boundaries). Fold 5's MAE for window=1 and fold 5's MAE for
    window=4 are therefore measured on the exact same held-out weeks.

    Most of the huge fold-to-fold variance already documented (a quiet
    26-week block ~2.4 MAE vs an outbreak block ~20+ MAE) is a property of
    THAT FOLD, not of the window -- an outbreak in a fold makes every
    window's forecast harder in that fold. Comparing window MEANS as if
    they were independent samples leaves all of that shared noise in place.
    Comparing the PAIRED DIFFERENCE (window_a's MAE minus window_b's MAE,
    fold by fold) cancels the part of the noise common to both windows in
    that fold, isolating the window-specific effect -- the more targeted
    signal for a question this fine (observed differences of ~0.04-0.3 MAE,
    far smaller than the ~4-5 MAE fold-to-fold noise).

    Returns one row per (model, window_a, window_b, fold) with MAE_a,
    MAE_b, and diff = MAE_a - MAE_b. A negative diff means window_a (the
    shorter window, listed first in each pair) had LOWER error in that fold.
    """
    model_rows = cv_results[~cv_results["model"].str.startswith("Persistence")]
    out = []
    for model in model_rows["model"].unique():
        sub = model_rows[model_rows["model"] == model]
        pivot = sub.pivot(index="fold", columns="history_window", values="MAE")
        for wa, wb in pairs:
            if wa not in pivot.columns or wb not in pivot.columns:
                continue
            diff = pivot[wa] - pivot[wb]
            for fold in pivot.index:
                out.append(dict(
                    model=model, window_a=wa, window_b=wb, fold=int(fold),
                    MAE_a=pivot.loc[fold, wa], MAE_b=pivot.loc[fold, wb],
                    diff=diff.loc[fold],
                ))
    return pd.DataFrame(out)

def paired_diff_summary(paired: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize paired differences per (model, window_a, window_b): mean/std
    of the paired diff, and how often window_a beat window_b (diff < 0)
    across folds -- a simple, transparent sign-consistency check.

    A win rate close to 50% means the folds DISAGREE with each other about
    which window is better in that pair -- i.e. no consistent direction,
    which is direct evidence the comparison is not resolved (not merely
    "not yet significant"). A win rate strongly away from 50%, together
    with a mean_diff clearly outside +/-1 std_diff of zero, is what
    consistent, resolvable evidence would look like.
    """
    out = []
    for (model, wa, wb), g in paired.groupby(["model", "window_a", "window_b"]):
        n = len(g)
        wins_a = int((g["diff"] < 0).sum())   # window_a strictly lower MAE that fold
        ties = int((g["diff"] == 0).sum())
        out.append(dict(
            model=model, window_a=int(wa), window_b=int(wb), n_folds=n,
            mean_diff=g["diff"].mean(), std_diff=g["diff"].std(),
            wins_a=wins_a, wins_b=n - wins_a - ties, ties=ties,
            win_rate_a_pct=100 * wins_a / n,
        ))
    return pd.DataFrame(out)

def sensitivity_analysis(results: pd.DataFrame) -> pd.DataFrame:
    """
    Step-by-step change from each window to the next longer one.
    Negative delta_MAE_pct = the longer window IMPROVED accuracy.
    """
    out = []
    for model in results["model"].unique():
        if model.startswith("Persistence"):
            continue
        sub = results[results["model"] == model].sort_values("history_window")
        for prev, cur in zip(sub.itertuples(), sub.iloc[1:].itertuples()):
            out.append(dict(
                model=model,
                transition=f"{prev.history_window} -> {cur.history_window} weeks",
                MAE_before=prev.MAE, MAE_after=cur.MAE,
                delta_MAE=cur.MAE - prev.MAE,
                delta_MAE_pct=100 * (cur.MAE - prev.MAE) / prev.MAE,
                delta_RMSE_pct=100 * (cur.RMSE - prev.RMSE) / prev.RMSE,
                train_time_ratio=cur.training_time / prev.training_time,
            ))
    return pd.DataFrame(out)

def robustness_check(results: pd.DataFrame, tolerance=TOLERANCE) -> pd.DataFrame:
    """
    Apply the SAME pre-registered rule to a second, disjoint future period
    (the validation set) and check whether it gives the same answer.

    WHY THIS EXISTS
    ---------------
    A single test period gives a single ranking of windows, and it is very
    easy to mistake "the ranking on my one test set" for "the truth". If the
    minimum sufficient window is a real property of the disease dynamics, it
    should look similar on two different stretches of future. If it moves
    around, then the honest conclusion is that the data does not resolve the
    question -- and reporting a confident single number would be overclaiming.
    """
    out = []
    for model in results["model"].unique():
        if model.startswith("Persistence"):
            continue
        sub = results[results["model"] == model].sort_values("history_window")
        picks = {}
        for period, col in (("test", "MAE"), ("validation", "val_MAE")):
            best = sub[col].min()
            near = sub[sub[col] <= best * (1 + tolerance)]
            picks[period] = dict(
                best_window=int(sub.loc[sub[col].idxmin(), "history_window"]),
                min_sufficient=int(near["history_window"].min()),
                best_score=best,
                spread_pct=100 * (sub[col].max() - sub[col].min()) / sub[col].min(),
            )
        out.append(dict(
            model=model,
            test_best_window=picks["test"]["best_window"],
            val_best_window=picks["validation"]["best_window"],
            test_min_sufficient=picks["test"]["min_sufficient"],
            val_min_sufficient=picks["validation"]["min_sufficient"],
            test_spread_pct=picks["test"]["spread_pct"],
            val_spread_pct=picks["validation"]["spread_pct"],
            periods_agree=picks["test"]["min_sufficient"] == picks["validation"]["min_sufficient"],
        ))
    return pd.DataFrame(out)

def baseline_comparison(results: pd.DataFrame) -> pd.DataFrame:
    """
    How much does each model/window actually beat the persistence baseline?
    Negative improvement = WORSE than simply predicting 'next week = this week'.
    """
    base = results[results["model"].str.startswith("Persistence")].iloc[0]
    rows = []
    for r in results[~results["model"].str.startswith("Persistence")].itertuples():
        rows.append(dict(
            history_window=r.history_window, model=r.model,
            MAE=r.MAE, baseline_MAE=base.MAE,
            improvement_pct=100 * (base.MAE - r.MAE) / base.MAE,
            beats_baseline=r.MAE < base.MAE,
            val_improvement_pct=100 * (base.val_MAE - r.val_MAE) / base.val_MAE,
        ))
    return pd.DataFrame(rows)

def minimum_sufficient_window(results: pd.DataFrame, tolerance=TOLERANCE) -> pd.DataFrame:
    """Apply the pre-registered rule declared at the top of this module."""
    out = []
    for model in results["model"].unique():
        if model.startswith("Persistence"):
            continue
        sub = results[results["model"] == model].sort_values("history_window")
        best_mae = sub["MAE"].min()
        best_window = int(sub.loc[sub["MAE"].idxmin(), "history_window"])
        threshold = best_mae * (1 + tolerance)
        near = sub[sub["MAE"] <= threshold]
        out.append(dict(
            model=model, best_window=best_window, best_MAE=best_mae,
            tolerance=tolerance, mae_threshold=threshold,
            minimum_sufficient_window=int(near["history_window"].min()),
            near_optimal_windows=", ".join(str(w) for w in near["history_window"]),
        ))
    return pd.DataFrame(out)
