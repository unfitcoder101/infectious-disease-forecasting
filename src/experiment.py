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

from evaluation import chronological_split, evaluate, naive_baseline
from features import HISTORY_WINDOWS, build_aligned_tables, split_X_y
from models import RANDOM_SEED, get_models

TOLERANCE = 0.05
N_TIMING_REPEATS = 5


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
