"""Run the unchanged history-window experiment on the fixed Brazil snapshot.

Usage:
  python brazil_analysis/run_brazil_experiment.py --plan   # audit + split plan only
  python brazil_analysis/run_brazil_experiment.py --execute  # fit and score models

All generated data, tables, and figures remain in brazil_analysis/.
"""
import argparse
import itertools
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

import experiment as ex  # noqa: E402
from brazil_data import SERIES_LABEL, audit, load_raw  # noqa: E402
from evaluation import chronological_split, rolling_origin_splits  # noqa: E402
from features import HISTORY_WINDOWS, build_aligned_tables  # noqa: E402
from models import RANDOM_SEED, get_models  # noqa: E402
from preprocessing import preprocess  # noqa: E402
from visualization import (METRICS, plot_cv_fold_detail, plot_cv_summary,
                           plot_metric, plot_robustness)  # noqa: E402

DATA_DIR = HERE / "data" / "processed"
TABLES_DIR = HERE / "tables"
RESULTS_DIR = HERE / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
BLOCKS = ((ex.CV_TEST_BLOCK, "cv26"), (ex.FINE_CV_TEST_BLOCK, "cv13"))


def build_plan(series: pd.DataFrame, tables: dict, block: int) -> pd.DataFrame:
    reference = tables[max(HISTORY_WINDOWS)]
    n = len(reference)
    min_train = int(n * ex.CV_TRAIN_FRAC)
    splits = rolling_origin_splits(reference, min_train, block)
    rows = []
    for fold, train, test in splits:
        rows.append({
            "fold": fold,
            "block_weeks": block,
            "n_train": len(train),
            "n_test": len(test),
            "first_train_target_week": train["date"].iloc[0] + pd.Timedelta(days=7),
            "last_train_target_week": train["date"].iloc[-1] + pd.Timedelta(days=7),
            "first_test_target_week": test["date"].iloc[0] + pd.Timedelta(days=7),
            "last_test_target_week": test["date"].iloc[-1] + pd.Timedelta(days=7),
            "test_mean_cases": float(test["target"].mean()),
            "test_max_cases": int(test["target"].max()),
            "test_cases_above_train_max": int((test["target"] > train["target"].max()).sum()),
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--plan", action="store_true")
    choice.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    findings = audit()
    print("BRAZIL SERIES AUDIT")
    for key, value in findings.items():
        print(f"{key}: {value}")

    raw = load_raw()
    series = preprocess(raw)
    if not series.equals(raw.astype({"cases": int})):
        raise ValueError("Preprocessing changed this already audited series")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    series.to_csv(DATA_DIR / "rio_weekly_cases.csv", index=False)
    tables = build_aligned_tables(series, HISTORY_WINDOWS)
    ref_dates = list(tables[max(HISTORY_WINDOWS)]["date"])
    if not all(list(table["date"]) == ref_dates for table in tables.values()):
        raise AssertionError("History windows do not share the same evaluation weeks")
    for window, table in tables.items():
        print(f"window={window:2d}: aligned_rows={len(table)}; features={window}")

    summary = []
    for block, tag in BLOCKS:
        plan = build_plan(series, tables, block)
        plan.to_csv(TABLES_DIR / f"protocol_fold_plan_{block}w.csv", index=False)
        n = len(tables[max(HISTORY_WINDOWS)])
        min_train = int(n * ex.CV_TRAIN_FRAC)
        summary.append({"block_weeks": block, "aligned_rows": n, "minimum_train_rows": min_train,
                        "folds": len(plan), "unused_rows": n - min_train - len(plan) * block})
        print(f"{block}-week rolling CV: {len(plan)} folds; {summary[-1]['unused_rows']} trailing rows unused")
    pd.DataFrame(summary).to_csv(TABLES_DIR / "protocol_fold_summary.csv", index=False)
    train, val, test = chronological_split(tables[max(HISTORY_WINDOWS)])
    pd.DataFrame([
        {"part": name, "rows": len(part), "first_target_week": part["date"].iloc[0] + pd.Timedelta(days=7),
         "last_target_week": part["date"].iloc[-1] + pd.Timedelta(days=7),
         "mean_cases": part["target"].mean(), "max_cases": part["target"].max()}
        for name, part in (("train", train), ("validation", val), ("test", test))
    ]).to_csv(TABLES_DIR / "protocol_single_split.csv", index=False)

    if args.plan:
        print("Plan written. No model was trained.")
        return

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("\nSINGLE CHRONOLOGICAL SPLIT")
    results = ex.run_experiment(series, HISTORY_WINDOWS)
    results.to_csv(RESULTS_DIR / "brazil_results_table.csv", index=False)
    ex.sensitivity_analysis(results).to_csv(RESULTS_DIR / "brazil_sensitivity_analysis.csv", index=False)
    mins = ex.minimum_sufficient_window(results)
    mins.to_csv(RESULTS_DIR / "brazil_minimum_sufficient_window.csv", index=False)
    robust = ex.robustness_check(results)
    robust.to_csv(RESULTS_DIR / "brazil_robustness_check.csv", index=False)
    basecmp = ex.baseline_comparison(results)
    basecmp.to_csv(RESULTS_DIR / "brazil_baseline_comparison.csv", index=False)
    print(results[["history_window", "model", "MAE", "RMSE", "R2", "val_MAE", "n_train", "n_test"]]
          .to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\n5% rule:\n", mins.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\nPersistence comparison:\n", basecmp.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    for metric, ylabel, title, is_cost in METRICS:
        plot_metric(results, metric, ylabel, f"Rio de Janeiro, Brazil: {title}", is_cost,
                    FIGURES_DIR / f"brazil_{metric.lower()}_vs_history_window.png")
    plot_robustness(results, FIGURES_DIR / "brazil_robustness_two_periods.png")

    all_pairs = tuple(itertools.combinations(HISTORY_WINDOWS, 2))
    for block, tag in BLOCKS:
        print(f"\nROLLING-ORIGIN {block}-WEEK CV")
        cv = ex.run_cv_experiment(series, HISTORY_WINDOWS, test_block=block)
        cv.to_csv(RESULTS_DIR / f"brazil_{tag}_results.csv", index=False)
        cv_summary = ex.cv_summary(cv)
        cv_summary.to_csv(RESULTS_DIR / f"brazil_{tag}_summary.csv", index=False)
        wins = ex.fold_win_counts(cv)
        wins.to_csv(RESULTS_DIR / f"brazil_{tag}_fold_win_counts.csv", index=False)
        cv_mins = ex.cv_minimum_sufficient_window(cv_summary)
        cv_mins.to_csv(RESULTS_DIR / f"brazil_{tag}_minimum_sufficient_window.csv", index=False)
        paired = ex.paired_fold_differences(cv, pairs=all_pairs)
        paired.to_csv(RESULTS_DIR / f"brazil_{tag}_paired_differences.csv", index=False)
        paired_summary = ex.paired_diff_summary(paired)
        paired_summary.to_csv(RESULTS_DIR / f"brazil_{tag}_paired_diff_summary.csv", index=False)
        print(cv_summary[["history_window", "model", "n_folds", "MAE_mean", "MAE_std", "RMSE_mean"]]
              .to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        print("5% rule:\n", cv_mins.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        print("fold wins:\n", wins.to_string(index=False))
        plot_cv_summary(cv_summary, FIGURES_DIR / f"brazil_{tag}_mae_vs_history_window.png")
        plot_cv_fold_detail(cv, FIGURES_DIR / f"brazil_{tag}_fold_detail.png")

    print(f"\nAll Brazil outputs are isolated in {HERE.relative_to(ROOT)}.")


if __name__ == "__main__":
    main()
