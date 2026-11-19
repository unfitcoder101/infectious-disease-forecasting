# Level versus change forecast follow-up

This add-on follows the instructor's checklist. It does not overwrite the original `results/`, `singapore_analysis/results/`, or `brazil_analysis/results/` outputs.

## Run

From the repository root, install the existing pinned requirements plus the follow-up SciPy pin, then run:

```bash
python followup_experiments/run_change_forecast.py --execute
```

The script fixes Python/NumPy/model seeds at 42. It uses the existing model factory and hyperparameters, 1/2/4/8/12-week lag windows, an expanding 70% training window, and 13-week non-overlapping rolling-origin test blocks. All models and both target formulations share the same test points within each series. The number of folds follows from the series length; they are 21 for San Juan, 11 for Iquitos and 13 for Singapore.

## Outputs

- `results/run_manifest.json`: Python, package, platform, seed and fold configuration.
- `results/fold_plan_summary.csv`: aligned row counts and fold counts by series.
- `results/oos_predictions.csv`: each point-level forecast, its absolute/squared error and configuration relative MAE.
- `results/all_models_relative_mae_summary.csv`: case-scale MAE/RMSE and relative errors for persistence, all direct-count configurations and the selected seasonal configurations.
- `results/dm_tests_{san_juan,iquitos,singapore}.csv`: 20 comparisons against persistence plus four best-short versus best-long comparisons per series, pointwise absolute-error DM test, h=1, and Holm adjustment within series.
- `results/seasonality_sensitivity.csv`: reruns of only each target/model's best short and best long window with sine/cosine week-of-year inputs.
- `figures/{san_juan,iquitos,singapore}_relative_mae.png`: relative MAE by window for level and change targets.

Detailed findings and limitations are in `research_update.md`. The pointwise DM assumptions and post-selection limitation for the best-short/best-long comparisons are stated there and in the per-series result files.

## Environment

Use Python **3.12.5**. `requirements.txt` pins the original model stack; `requirements-followup.txt` adds the SciPy version used for the DM tests. The run manifest records the exact versions used in the saved run.

The prior result tables and figures were also regenerated and audited; see `reproduction_audit.md`. The Singapore source overlap outputs are reproducible with `python singapore_analysis/check_overlap.py`.
