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
from experiment import (TOLERANCE, baseline_comparison,          # noqa: E402
                        minimum_sufficient_window, robustness_check,
                        run_experiment, sensitivity_analysis)
from features import HISTORY_WINDOWS, build_aligned_tables, split_X_y  # noqa: E402
from models import RANDOM_SEED, get_models                      # noqa: E402
from preprocessing import audit_series, preprocess, print_audit  # noqa: E402
from visualization import (FIG_DIR, generate_all,                 # noqa: E402
                           plot_forecast, plot_robustness)

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

    banner("STEP 8: FIGURES")
    for p in generate_all(results):
        print(f"  wrote {p.relative_to(Path.cwd())}")

    rob_out = FIG_DIR / "robustness_two_periods.png"
    plot_robustness(results, rob_out)
    print(f"  wrote {rob_out.relative_to(Path.cwd())}")

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

    banner("DONE")
    print(f"Results  -> {RESULTS_DIR}")
    print(f"Figures  -> {FIG_DIR}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", default="sj", choices=list(CITIES))
    main(ap.parse_args().city)
