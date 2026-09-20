"""               
How Much Surveillance History Is Enough?
A temporal context sensitivity study for short-term infectious disease forecasting.
            
Run with:  python run_experiment.py  [--city sj|iq]
              
Pipeline: load -> audit -> preprocess -> lag features -> train -> evaluate
          -> sensitivity analysis -> save results -> plots
"""
                 
import argparse
import random               
import sys
from pathlib import Path                       
              
import numpy as np
import pandas as pd
             
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))                   

from data_loader import CITIES, load_raw                       # noqa: E402
from evaluation import chronological_split                      # noqa: E402
from experiment import (CV_TEST_BLOCK, CV_TRAIN_FRAC, FINE_CV_TEST_BLOCK,  # noqa: E402
                        FINE_CV_WINDOWS, TOLERANCE, baseline_comparison,
                        cv_minimum_sufficient_window, cv_summary, fold_win_counts,
                        minimum_sufficient_window, paired_diff_summary,
                        paired_fold_differences, robustness_check,
                        run_cv_experiment, run_experiment, sensitivity_analysis)
from features import HISTORY_WINDOWS, build_aligned_tables, split_X_y  # noqa: E402
from models import RANDOM_SEED, get_models                      # noqa: E402
from preprocessing import audit_series, preprocess, print_audit  # noqa: E402
from visualization import (FIG_DIR, generate_all,                 # noqa: E402
                           plot_cv_fold_detail, plot_cv_summary,
                           plot_forecast, plot_paired_differences,
                           plot_robustness)

RESULTS_DIR = Path(__file__).resolve().parent / "results"
PROCESSED_DIR = Path(__file__).resolve().parent / "data" / "processed"


def banner(text):
    print(f"\n{'=' * 72}\n{text}\n{'=' * 72}")


def main(city: str):
    # Fixed seeds everywhere, so a rerun reproduces these numbers exactly.
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    pd.set_option("display.width", 200)

    banner(f"STEP 1-2: LOAD AND AUDIT — {CITIES[city]}")
    raw = load_raw(city)
    print_audit(audit_series(raw), CITIES[city])
    series = preprocess(raw)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    series.to_csv(PROCESSED_DIR / f"{city}_weekly_cases.csv", index=False)

    banner("STEP 3: LAG FEATURES (all windows aligned to identical weeks)")
    tables = build_aligned_tables(series, HISTORY_WINDOWS)
    for w in HISTORY_WINDOWS:
        tr, va, te = chronological_split(tables[w])
        print(f"  window={w:2d} | rows={len(tables[w])} | train={len(tr)} val={len(va)} test={len(te)} "
              f"| test period {te['date'].min().date()} -> {te['date'].max().date()}")

    banner("STEP 4-5: TRAIN XGBoost AND LightGBM ACROSS ALL WINDOWS")
    results = run_experiment(series, HISTORY_WINDOWS)
    cols = ["history_window", "model", "MAE", "RMSE", "R2",
            "training_time", "inference_time", "n_observations", "n_train", "n_test", "n_features"]
    print(results[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    banner("STEP 6: SENSITIVITY ANALYSIS — what does extra history buy?")
    sens = sensitivity_analysis(results)
    print(sens.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    banner(f"STEP 7: MINIMUM SUFFICIENT HISTORY (pre-registered rule, tolerance={TOLERANCE:.0%})")
    mins = minimum_sufficient_window(results)
    print(mins.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    banner("STEP 7b: ROBUSTNESS — does the same rule give the same answer on a second period?")
    robust = robustness_check(results)
    print(robust.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    if not robust["periods_agree"].all():
        print("\n  *** The two evaluation periods DISAGREE. The minimum sufficient")
        print("      window is therefore NOT identified by this data. See research_notes.md.")

    banner("STEP 7c: DID WE EVEN BEAT THE PERSISTENCE BASELINE?")
    basecmp = baseline_comparison(results)
    print(basecmp.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    n_beat = int(basecmp["beats_baseline"].sum())
    print(f"\n  {n_beat} of {len(basecmp)} model/window configurations beat persistence on the test set.")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    robust.to_csv(RESULTS_DIR / "robustness_check.csv", index=False)
    basecmp.to_csv(RESULTS_DIR / "baseline_comparison.csv", index=False)
    results.to_csv(RESULTS_DIR / "results_table.csv", index=False)
    sens.to_csv(RESULTS_DIR / "sensitivity_analysis.csv", index=False)
    mins.to_csv(RESULTS_DIR / "minimum_sufficient_window.csv", index=False)

    banner("STEP 7d: ROLLING-ORIGIN CROSS-VALIDATION (the CV milestone)")
    n_aligned = len(build_aligned_tables(series, HISTORY_WINDOWS)[max(HISTORY_WINDOWS)])
    min_train = int(n_aligned * CV_TRAIN_FRAC)
    n_folds_expected = (n_aligned - min_train) // CV_TEST_BLOCK
    print(f"  aligned rows={n_aligned} | train_frac={CV_TRAIN_FRAC:.0%} -> min_train={min_train} "
          f"| test_block={CV_TEST_BLOCK} weeks | folds={n_folds_expected}")
    if not (8 <= n_folds_expected <= 10):
        print(f"  NOTE: fold count {n_folds_expected} is outside the 8-10 target for this series "
              f"length ({CITIES[city]}). Config was derived from San Juan; see research_notes.md.")

    cv_results = run_cv_experiment(series, HISTORY_WINDOWS)
    fold_bounds = (cv_results[["fold", "fold_test_start", "fold_test_end"]]
                   .drop_duplicates().sort_values("fold"))
    print("\n  fold test-block boundaries:")
    print(fold_bounds.to_string(index=False))

    summary = cv_summary(cv_results)
    print("\n  CV summary (mean +/- std across folds):")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    wins = fold_win_counts(cv_results)
    print("\n  fold win counts (how often each window had the lowest MAE):")
    print(wins.to_string(index=False, float_format=lambda v: f"{v:.1f}"))

    cv_mins = cv_minimum_sufficient_window(summary)
    print(f"\n  CV minimum sufficient window (tolerance={TOLERANCE:.0%}):")
    print(cv_mins.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    cv_results.to_csv(RESULTS_DIR / "cv_results.csv", index=False)
    summary.to_csv(RESULTS_DIR / "cv_summary.csv", index=False)
    wins.to_csv(RESULTS_DIR / "cv_fold_win_counts.csv", index=False)
    cv_mins.to_csv(RESULTS_DIR / "cv_minimum_sufficient_window.csv", index=False)

    banner("STEP 7e: DOES CV CHANGE THE CONCLUSION?")
    for model in mins["model"]:
        single_pick = int(mins.loc[mins["model"] == model, "minimum_sufficient_window"].iloc[0])
        cv_pick = int(cv_mins.loc[cv_mins["model"] == model, "minimum_sufficient_window"].iloc[0])
        agree = "SAME" if single_pick == cv_pick else "DIFFERENT"
        print(f"  {model:9s}: single-split pick={single_pick:2d} weeks | CV pick={cv_pick:2d} weeks -> {agree}")

    banner("STEP 8: FIGURES")
    for p in generate_all(results):
        print(f"  wrote {p.relative_to(Path.cwd())}")

    rob_out = FIG_DIR / "robustness_two_periods.png"
    plot_robustness(results, rob_out)
    print(f"  wrote {rob_out.relative_to(Path.cwd())}")

    cv_summary_out = FIG_DIR / "cv_mae_vs_history_window.png"
    plot_cv_summary(summary, cv_summary_out)
    print(f"  wrote {cv_summary_out.relative_to(Path.cwd())}")

    cv_detail_out = FIG_DIR / "cv_fold_detail.png"
    plot_cv_fold_detail(cv_results, cv_detail_out)
    print(f"  wrote {cv_detail_out.relative_to(Path.cwd())}")

    # Forecast overlay: shortest vs longest window, for the better model overall.
    model_rows = results[~results["model"].str.startswith("Persistence")]
    best_model = model_rows.groupby("model")["MAE"].mean().idxmin()
    preds = {}
    for w in (min(HISTORY_WINDOWS), max(HISTORY_WINDOWS)):
        tr, _, te = chronological_split(tables[w])
        X_tr, y_tr, _ = split_X_y(tr)
        X_te, _, _ = split_X_y(te)
        m = get_models()[best_model]
        m.fit(X_tr, y_tr)
        preds[w] = m.predict(X_te)
    _, _, te = chronological_split(tables[max(HISTORY_WINDOWS)])
    out = FIG_DIR / "forecast_actual_vs_predicted.png"
    plot_forecast((te["date"].values, te["target"].values), preds, out, best_model)
    print(f"  wrote {out.relative_to(Path.cwd())}")

    banner("STEP 9: TARGETED FOLLOW-UP — FINE-GRAINED CV FOR WINDOWS {1, 2, 4}")
    print("  Reuses the SAME aligned base (align_windows=HISTORY_WINDOWS, n=924,")
    print(f"  min_train={min_train}) as the committed 10-fold/26-week CV above.")
    print("  Only test_block and the trained window set change, to sharpen the")
    print("  still-unresolved 1-vs-2-vs-4 comparison. Windows 8/12 excluded: already")
    print("  resolved as worse. The committed cv_* outputs above are untouched.")
    fine_n_folds_expected = (n_aligned - min_train) // FINE_CV_TEST_BLOCK
    print(f"\n  windows={FINE_CV_WINDOWS} | test_block={FINE_CV_TEST_BLOCK} weeks "
          f"| folds={fine_n_folds_expected} (vs 10 in the committed run)")

    fine_cv_results = run_cv_experiment(series, FINE_CV_WINDOWS, test_block=FINE_CV_TEST_BLOCK,
                                         align_windows=HISTORY_WINDOWS)
    fine_bounds = (fine_cv_results[["fold", "fold_test_start", "fold_test_end"]]
                   .drop_duplicates().sort_values("fold"))
    print("\n  fold test-block boundaries (fine-grained run):")
    print(fine_bounds.to_string(index=False))

    fine_summary = cv_summary(fine_cv_results)
    print(f"\n  fine-grained CV summary (mean +/- std across {fine_n_folds_expected} folds):")
    print(fine_summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    fine_wins = fold_win_counts(fine_cv_results)
    print("\n  fine-grained fold win counts:")
    print(fine_wins.to_string(index=False, float_format=lambda v: f"{v:.1f}"))

    fine_mins = cv_minimum_sufficient_window(fine_summary)
    print(f"\n  fine-grained minimum sufficient window (tolerance={TOLERANCE:.0%}):")
    print(fine_mins.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    paired = paired_fold_differences(fine_cv_results)
    paired_summ = paired_diff_summary(paired)
    print("\n  paired per-fold MAE differences (A - B; negative = A better that fold):")
    print(paired_summ.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    fine_cv_results.to_csv(RESULTS_DIR / "fine_cv_results.csv", index=False)
    fine_summary.to_csv(RESULTS_DIR / "fine_cv_summary.csv", index=False)
    fine_wins.to_csv(RESULTS_DIR / "fine_cv_fold_win_counts.csv", index=False)
    fine_mins.to_csv(RESULTS_DIR / "fine_cv_minimum_sufficient_window.csv", index=False)
    paired.to_csv(RESULTS_DIR / "fine_cv_paired_differences.csv", index=False)
    paired_summ.to_csv(RESULTS_DIR / "fine_cv_paired_diff_summary.csv", index=False)

    fine_fig_out = FIG_DIR / "fine_cv_paired_differences.png"
    plot_paired_differences(paired_summ, fine_fig_out)
    print(f"  wrote {fine_fig_out.relative_to(Path.cwd())}")

    banner("DONE")
    print(f"Results  -> {RESULTS_DIR}")
    print(f"Figures  -> {FIG_DIR}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", default="sj", choices=list(CITIES))
    main(ap.parse_args().city)
