# History-window and log-change forecasting follow-up

This reproducible follow-up compares history-window length for one-week-ahead dengue forecasts. It includes San Juan, Iquitos, Singapore, and Rio de Janeiro, Brazil in the same pipeline. The original result folders and inputs are preserved.

## Run

From the repository root:

```bash
python followup_experiments/run_change_forecast.py --execute
```

The run fixes Python, NumPy, and model seeds at 42; uses the existing XGBoost/LightGBM settings; compares 1/2/4/8/12 lag weeks; and evaluates an expanding 70% initial training set with non-overlapping 13-week test blocks. Direct-level, log-change, and persistence forecasts share the same test dates within each series.

## Statistical analysis

Diebold–Mariano tests use pointwise absolute-error differences, Newey–West Bartlett HAC variance with lag 4, the horizon-1 Harvey–Leybourne–Newbold correction, and Holm adjustment within each series. Each series has 28 tests: 20 model/target/window comparisons against persistence plus eight prespecified history contrasts (1 vs. 12 and 4 vs. 12 for each model and target). Seasonality is a sensitivity analysis on selected short and long configurations.

## Outputs

- `results/run_manifest.json`: runtime and package versions, seeds, folds, and test protocol.
- `results/fold_plan_summary.csv`: aligned rows and fold counts.
- `results/oos_predictions.csv`: point forecasts and errors.
- `results/all_models_relative_mae_summary.csv`: MAE, RMSE, and relative MAE for model configurations and persistence.
- `results/dm_tests_{san_juan,iquitos,singapore,rio_de_janeiro_brazil}.csv`: per-series HAC/HLN tests and Holm-adjusted p-values.
- `results/seasonality_sensitivity.csv`: selected seasonal sensitivity results.
- `figures/relative_mae_by_window_all_series.png`: relative MAE by window across the four settings.
- `figures/singapore_fold_errors_and_peaks.png`: Singapore fold errors and peak weeks.

The current findings and limitations are in `research_update.md`. Across all four series, no test was significant after correction; the conclusion is no detectable history-window gain and no significant improvement over persistence.
