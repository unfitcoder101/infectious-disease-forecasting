# Forecasting experiment update for instructor review

**Run date:** 2026-11-23
**Question:** For one-week-ahead dengue forecasting, does adding older weekly case counts improve performance over a one-week input window or persistence?

## Main result

The common pipeline now covers San Juan, Iquitos, Singapore, and Rio de Janeiro, Brazil. Brazil was feasible in the same run and is included as a fourth within-series evaluation setting. It is not cross-country model transfer: every series is trained and evaluated separately using the same procedure.

Across the four series, **0 of 28 Holm-adjusted tests per series were significant at 0.05**. This includes all 20 level/change model configurations versus persistence and the eight pre-specified comparisons (1 vs. 12 and 4 vs. 12 weeks, for each of two models and two targets). No history comparison was significant, and no configuration significantly outperformed persistence. This is a finding of no detectable difference under this design; it is not an equivalence or noninferiority result.

The conclusion for the paper is: **“Extending history beyond the most recent week gave no detectable improvement, and no configuration significantly outperformed persistence.”** The observed best point estimates vary by series, so they should not be presented as confirmed winners.

## Evaluation design

- **Series:** DengAI San Juan and Iquitos; Singapore Ministry of Health weekly dengue-fever series; Rio de Janeiro reported dengue cases from the dated InfoDengue snapshot.
- **Target:** direct next-week count, or log change `d = log(y[t+1]+1) - log(y[t]+1)`. Change forecasts are reconstructed as `max(0, exp(log(y[t]+1)+d_hat)-1)`. MAE and RMSE use original case counts.
- **Inputs:** lag windows of 1, 2, 4, 8, and 12 weeks.
- **Models:** existing XGBoost and LightGBM configurations; persistence predicts the latest observed count.
- **Validation:** expanding initial training set of 70%, then non-overlapping 13-week test blocks. Within each series all methods share the same test dates. Models are fitted at the start of each block, not updated each week inside the block.
- **Inference:** two-sided Diebold–Mariano tests on pointwise absolute-error differences, Newey–West Bartlett HAC variance with lag 4, horizon-1 Harvey–Leybourne–Newbold correction, and Holm correction over 28 comparisons within each series.
- **Pre-specified history comparisons:** 1 vs. 12 weeks and 4 vs. 12 weeks for both model families and both target types.
- **Scale comparison:** relative MAE equals configuration MAE divided by that series' persistence MAE. No raw case errors are pooled across series.
- **Seasonal sensitivity:** best short (1/2/4) and best long (8/12) windows were selected from the main results, then refit with sine/cosine week-of-year terms. This is exploratory because selection uses the evaluation outcomes.

## Data and fold counts

| Series | Weekly observations | Aligned rows | Initial training rows | Folds | Test weeks |
|---|---:|---:|---:|---:|---:|
| San Juan | 936 | 924 | 646 | 21 | 273 |
| Iquitos | 520 | 508 | 355 | 11 | 143 |
| Singapore | 574 | 562 | 393 | 13 | 169 |
| Rio de Janeiro, Brazil | 835 | 823 | 576 | 19 | 247 |

San Juan and Iquitos have five and ten final rows, respectively, that do not fill another 13-week test block; these are unused.

## Lowest observed MAE (descriptive)

| Series | Persistence MAE | Lowest model MAE | Relative MAE | Configuration |
|---|---:|---:|---:|---|
| San Juan | 6.161 | 5.971 | 0.969 | Change, LightGBM, 2 weeks |
| Iquitos | 4.664 | 4.597 | 0.986 | Direct level, LightGBM, 1 week |
| Singapore | 54.698 | 57.537 | 1.052 | Change, LightGBM, 4 weeks |
| Rio de Janeiro, Brazil | 186.717 | 151.251 | 0.810 | Change, LightGBM, 4 weeks |

None of these lowest point estimates is a statistically established winner. In Singapore, persistence had lower MAE than every tested model/window configuration. In Brazil, the best change-model point estimate was notably lower than persistence, but the corrected comparison was not significant.

## Seasonality sensitivity

The selected seasonal fits lowered MAE relative to their nonseasonal counterparts in 6/8 San Juan, 5/8 Iquitos, 3/8 Singapore, and 1/8 Brazil configurations. In Singapore, none of the eight selected configurations beat persistence. The direction is mixed across series and does not change the main conclusion.

## Singapore data-source note

The project series matches the broader MOH Weekly Infectious Disease Bulletin Cases dengue-fever series in 261/261 weeks from 2014-W01 through 2018-W52. The separate MOH Weekly Number of Dengue and Dengue Haemorrhagic Fever Cases source differs in 52/261 weeks (19.9%). The cause is unresolved; the analysis retains the project series and does not combine sources. Details and week-level mismatches are in `singapore_analysis/singapore_overlap_results.md`; the manuscript moves this source discrepancy to Appendix A.

## Reproducibility and files

The run fixes Python, NumPy, and model seeds at 42. Environment versions: Python 3.12.5, NumPy 2.5.3, pandas 3.0.5, SciPy 1.18.1, scikit-learn 1.9.1, XGBoost 3.4.1, and LightGBM 4.7.0. The exact settings are recorded in `results/run_manifest.json`.

- Main code: `run_change_forecast.py`
- OOS predictions: `results/oos_predictions.csv`
- Relative-MAE summary: `results/all_models_relative_mae_summary.csv`
- Fold plan: `results/fold_plan_summary.csv`
- DM/HAC/HLN tables: `results/dm_tests_{san_juan,iquitos,singapore,rio_de_janeiro_brazil}.csv`
- Seasonality sensitivity: `results/seasonality_sensitivity.csv`
- Main paper figure: `figures/relative_mae_by_window_all_series.png`
- Singapore peak/fold-error figure: `figures/singapore_fold_errors_and_peaks.png`
- IEEE-format draft: `outputs/history_window_dengue_forecasting_ieee_draft.docx`

Reproduce with `python followup_experiments/run_change_forecast.py --execute` from the repository root.
