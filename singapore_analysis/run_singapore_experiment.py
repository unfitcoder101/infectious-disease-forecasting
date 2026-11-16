"""
Singapore external validation of the history-window study.

    .venv/bin/python singapore_analysis/run_singapore_experiment.py            # DRY RUN (default)
    .venv/bin/python singapore_analysis/run_singapore_experiment.py --execute  # trains models

DRY RUN builds the aligned tables and the fold layout, verifies them, and
writes the fold plan to singapore_analysis/tables/. It trains nothing.

--execute runs the experiment defined in singapore_experiment_protocol.md.
It must not be used until that protocol has been reviewed and approved.

Everything that defines the method is imported UNCHANGED from src/:
windows, lag construction, alignment, models and hyperparameters, seed,
splits, metrics, persistence baseline and the 5% rule. This script changes
only the data source and the output location. It writes nothing under
results/ or data/, so the San Juan and Iquitos outputs cannot be touched.
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

from evaluation import chronological_split, rolling_origin_splits        # noqa: E402
from features import HISTORY_WINDOWS, build_aligned_tables               # noqa: E402
from preprocessing import audit_series, preprocess, print_audit          # noqa: E402
from singapore_data import SERIES_LABEL, epi_week_lookup, load_singapore  # noqa: E402

TABLES_DIR = HERE / "tables"
RESULTS_DIR = HERE / "results"          # NOT the project's results/ directory
FIGURES_DIR = HERE / "figures"

# Taken from src/experiment.py and repeated here only so the dry run does not
# need to import the model libraries. The --execute path imports the real
# constants and asserts these copies agree with them.
CV_TRAIN_FRAC = 0.70
BLOCK_PRIMARY = 26      # CV_TEST_BLOCK
BLOCK_SECONDARY = 13    # FINE_CV_TEST_BLOCK


def banner(text):
    print(f"\n{'=' * 78}\n{text}\n{'=' * 78}")


def fold_plan(tables: dict, test_block: int, labels: dict) -> pd.DataFrame:
    """Lay out and VERIFY the folds for one block size. Trains nothing."""
    ref = tables[max(tables)]
    n = len(ref)
    min_train = int(n * CV_TRAIN_FRAC)
    dates = list(ref["date"])
    pos = {d: i for i, d in enumerate(sorted(labels))}       # date -> row in the raw series
    ordered = sorted(labels)

    rows, prev_test_end = [], -1
    per_window = {w: rolling_origin_splits(t, min_train, test_block) for w, t in tables.items()}
    n_folds = len(per_window[max(tables)])
    for i in range(n_folds):
        _, train, test = per_window[max(tables)][i]
        # Every window must have identical rows in this fold.
        for w, folds in per_window.items():
            _, tr_w, te_w = folds[i]
            assert list(tr_w["date"]) == list(train["date"]), f"train rows differ for window {w}"
            assert list(te_w["date"]) == list(test["date"]), f"test rows differ for window {w}"
            assert np.array_equal(te_w["target"].to_numpy(), test["target"].to_numpy())
            assert np.array_equal(te_w["lag_1"].to_numpy(), test["lag_1"].to_numpy())
        train_end = dates.index(train["date"].iloc[-1])
        test_start, test_end = dates.index(test["date"].iloc[0]), dates.index(test["date"].iloc[-1])
        assert train_end + 1 == test_start, "gap or overlap between train and test"
        assert test_start > prev_test_end, "test blocks overlap"
        assert len(test) == test_block
        prev_test_end = test_end

        def target_label(input_date):          # the target is the week AFTER the input week
            return labels[ordered[pos[input_date] + 1]]

        # Leakage check on targets: the last training target must precede the first test target.
        assert pos[train["date"].iloc[-1]] + 1 < pos[test["date"].iloc[0]] + 1
        train_max = float(max(train["target"].max(), train[[c for c in train if c.startswith("lag_")]].max().max()))
        rows.append(dict(
            test_block_weeks=test_block, fold=i, n_train=len(train), n_test=len(test),
            first_train_target_week=target_label(train["date"].iloc[0]),
            last_train_target_week=target_label(train["date"].iloc[-1]),
            first_test_target_week=target_label(test["date"].iloc[0]),
            last_test_target_week=target_label(test["date"].iloc[-1]),
            test_target_mean=float(test["target"].mean()), test_target_min=float(test["target"].min()),
            test_target_max=float(test["target"].max()),
            max_value_seen_in_training=train_max,
            test_targets_above_training_max=int((test["target"] > train_max).sum()),
        ))
    plan = pd.DataFrame(rows)
    assert plan["n_train"].is_monotonic_increasing and plan["n_train"].is_unique
    plan.attrs.update(n_aligned=n, min_train=min_train, n_folds=n_folds,
                      unused_weeks=n - min_train - n_folds * test_block)
    return plan


def dry_run(series, tables, labels):
    banner("FOLD COUNT BY BLOCK SIZE (derived from the aligned row count, nothing forced)")
    n = len(tables[max(HISTORY_WINDOWS)])
    min_train = int(n * CV_TRAIN_FRAC)
    avail = n - min_train
    print(f"  observations={len(series)} | aligned rows={n} | min_train=int({n} x {CV_TRAIN_FRAC}) = {min_train} "
          f"| weeks available for testing={avail}")
    opts = pd.DataFrame([dict(test_block_weeks=b, n_folds=avail // b, unused_weeks=avail - (avail // b) * b,
                              in_original_8_to_10_target=8 <= avail // b <= 10)
                         for b in [13] + list(range(20, 27))])
    print(opts.to_string(index=False))
    opts.to_csv(TABLES_DIR / "protocol_fold_count_options.csv", index=False)

    banner("SINGLE CHRONOLOGICAL SPLIT (70 / 15 / 15, as in the original study)")
    tr, va, te = chronological_split(tables[max(HISTORY_WINDOWS)])
    ordered = sorted(labels)
    pos = {d: i for i, d in enumerate(ordered)}
    tl = lambda d: labels[ordered[pos[d] + 1]]                # noqa: E731
    split = pd.DataFrame([dict(part=name, rows=len(p), first_target_week=tl(p["date"].iloc[0]),
                               last_target_week=tl(p["date"].iloc[-1]), target_mean=p["target"].mean(),
                               target_max=p["target"].max())
                          for name, p in (("train", tr), ("validation", va), ("test", te))])
    print(split.round(1).to_string(index=False))
    split.to_csv(TABLES_DIR / "protocol_single_split.csv", index=False)

    for block, tag in ((BLOCK_PRIMARY, "primary"), (BLOCK_SECONDARY, "secondary")):
        plan = fold_plan(tables, block, labels)
        banner(f"FOLD PLAN, {tag.upper()}: {block}-week blocks -> {plan.attrs['n_folds']} folds, "
               f"{plan.attrs['unused_weeks']} weeks unused")
        print(plan.round(1).to_string(index=False))
        plan.to_csv(TABLES_DIR / f"protocol_fold_plan_{block}w.csv", index=False)
        print("  verified: identical rows across all windows in every fold; test blocks contiguous,")
        print("  non-overlapping and forward-chaining; training strictly expanding; no test target in training.")

    banner("DRY RUN COMPLETE. No model was trained. Nothing was written outside singapore_analysis/tables/.")


def execute(series, labels):
    """The experiment itself. Only reachable with --execute."""
    import experiment as ex
    from models import RANDOM_SEED
    from visualization import METRICS, plot_cv_fold_detail, plot_cv_summary, plot_metric, plot_robustness
    import singapore_stats as st

    assert ex.CV_TRAIN_FRAC == CV_TRAIN_FRAC and ex.CV_TEST_BLOCK == BLOCK_PRIMARY
    assert ex.FINE_CV_TEST_BLOCK == BLOCK_SECONDARY and ex.TOLERANCE == 0.05
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fmt = lambda v: f"{v:.4f}"                                # noqa: E731

    banner("SINGLE SPLIT: all windows x both models")
    results = ex.run_experiment(series, HISTORY_WINDOWS)
    results.to_csv(RESULTS_DIR / "sg_results_table.csv", index=False)
    ex.sensitivity_analysis(results).to_csv(RESULTS_DIR / "sg_sensitivity_analysis.csv", index=False)
    mins = ex.minimum_sufficient_window(results)
    mins.to_csv(RESULTS_DIR / "sg_minimum_sufficient_window.csv", index=False)
    robust = ex.robustness_check(results)
    robust.to_csv(RESULTS_DIR / "sg_robustness_check.csv", index=False)
    basecmp = ex.baseline_comparison(results)
    basecmp.to_csv(RESULTS_DIR / "sg_baseline_comparison.csv", index=False)
    print(results.to_string(index=False, float_format=fmt))
    print(mins.to_string(index=False, float_format=fmt))
    print(robust.to_string(index=False, float_format=fmt))
    print(basecmp.to_string(index=False, float_format=fmt))
    for metric, ylabel, title, is_cost in METRICS:
        plot_metric(results, metric, ylabel, f"Singapore: {title}", is_cost,
                    FIGURES_DIR / f"results_{metric.lower()}_vs_history_window.png")
    plot_robustness(results, FIGURES_DIR / "results_robustness_two_periods.png")

    ordered = sorted(labels)
    pos = {d: i for i, d in enumerate(ordered)}
    all_pairs = tuple((a, b) for i, a in enumerate(HISTORY_WINDOWS) for b in HISTORY_WINDOWS[i + 1:])

    for block, tag in ((BLOCK_PRIMARY, "cv26"), (BLOCK_SECONDARY, "cv13")):
        banner(f"ROLLING-ORIGIN CV: {block}-week blocks, all five windows")
        cv = ex.run_cv_experiment(series, HISTORY_WINDOWS, test_block=block)
        # fold_test_start/end are INPUT-week dates (as in the original). Add target-week labels.
        cv["first_test_target_week"] = [labels[ordered[pos[d] + 1]] for d in cv["fold_test_start"]]
        cv["last_test_target_week"] = [labels[ordered[pos[d] + 1]] for d in cv["fold_test_end"]]
        cv.to_csv(RESULTS_DIR / f"sg_{tag}_results.csv", index=False)

        summary = ex.cv_summary(cv)
        summary.to_csv(RESULTS_DIR / f"sg_{tag}_summary.csv", index=False)
        ex.fold_win_counts(cv).to_csv(RESULTS_DIR / f"sg_{tag}_fold_win_counts.csv", index=False)
        cv_mins = ex.cv_minimum_sufficient_window(summary)
        cv_mins.to_csv(RESULTS_DIR / f"sg_{tag}_minimum_sufficient_window.csv", index=False)
        paired = ex.paired_fold_differences(cv, pairs=all_pairs)
        paired.to_csv(RESULTS_DIR / f"sg_{tag}_paired_differences.csv", index=False)
        ex.paired_diff_summary(paired).to_csv(RESULTS_DIR / f"sg_{tag}_paired_diff_summary.csv", index=False)

        # Additions relative to the original study (see singapore_stats.py).
        st.extended_cv_summary(cv).to_csv(RESULTS_DIR / f"sg_{tag}_extended_summary.csv", index=False)
        st.near_optimal_table(summary, ex.TOLERANCE).to_csv(RESULTS_DIR / f"sg_{tag}_near_optimal.csv", index=False)
        for metric in ("MAE", "RMSE"):
            st.all_comparisons(cv, metric).to_csv(
                RESULTS_DIR / f"sg_{tag}_comparisons_{metric.lower()}.csv", index=False)

        print(summary.to_string(index=False, float_format=fmt))
        print(cv_mins.to_string(index=False, float_format=fmt))
        plot_cv_summary(summary, FIGURES_DIR / f"results_{tag}_mae_vs_history_window.png")
        plot_cv_fold_detail(cv, FIGURES_DIR / f"results_{tag}_fold_detail.png")

    banner(f"DONE. Results -> {RESULTS_DIR}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--execute", action="store_true",
                    help="train the models. Do not use before the protocol is approved.")
    args = ap.parse_args()
    pd.set_option("display.width", 250)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    banner(f"LOAD AND AUDIT: {SERIES_LABEL}")
    raw = load_singapore()
    print_audit(audit_series(raw), SERIES_LABEL)
    series = preprocess(raw)
    assert series.equals(raw.astype({"cases": int})), "preprocess() changed the series; it should be a no-op here"
    labels = epi_week_lookup()

    banner("LAG FEATURES (all windows aligned to identical weeks)")
    tables = build_aligned_tables(series, HISTORY_WINDOWS)
    for w, t in tables.items():
        print(f"  window={w:2d} | rows={len(t)} | features={w}")
    ref_dates = list(tables[max(HISTORY_WINDOWS)]["date"])
    assert all(list(t["date"]) == ref_dates for t in tables.values())
    # Target alignment check against the raw series: target of row with input week t is cases(t+1).
    cases = series.set_index("date")["cases"]
    ordered = list(series["date"])
    t12 = tables[12]
    for i in (0, len(t12) // 2, len(t12) - 1):
        j = ordered.index(t12["date"].iloc[i])
        assert t12["target"].iloc[i] == cases.iloc[j + 1] and t12["lag_1"].iloc[i] == cases.iloc[j]
        assert t12["lag_12"].iloc[i] == cases.iloc[j - 11]
    print("  verified: identical target weeks for every window; target = next week; lag_1 = current week.")

    if args.execute:
        execute(series, labels)
    else:
        dry_run(series, tables, labels)


if __name__ == "__main__":
    main()
