# Forecasting experiment update for instructor review

**Run date:** 2026-11-19  
**Question:** does history length affect one-week-ahead dengue forecasts, and does predicting log change improve on direct case-count prediction or persistence?

## Executive result

**There is still no single winning history window across San Juan, Iquitos and Singapore.** The change target improved Singapore's lowest observed MAE substantially compared with direct count prediction, but it still did not beat persistence. In San Juan and Iquitos, the best change models were slightly below persistence by point estimate; none of the relevant DM comparisons survived Holm correction. Short windows often had lower MAE than long windows, but no best-short versus best-long contrast was significant after correction.

The evidence supports a qualified conclusion: **1–4 weeks was often competitive, but the preferred window depended on the series, model and evaluation design.** These tests do not establish that adding older weeks improves forecasts beyond the latest count.

## Methods and reproducibility

- **Forecast target:** level models predict `y[t+1]`. Change models predict `d = log(y[t+1]+1) - log(y[t]+1)` using the same lag features. Change forecasts are converted back as `max(0, exp(log(y[t]+1) + d_hat) - 1)`. MAE and RMSE are computed in original cases.
- **Windows and models:** 1, 2, 4, 8, 12 weekly case lags; the project's unchanged XGBoost and LightGBM settings; persistence predicts `y[t+1] = y[t]`.
- **Validation:** expanding 70% training window, then non-overlapping 13-week test folds. Within a series, exactly the same forecast weeks are used for level, change and persistence. Fold counts follow each series' length: San Juan 21 folds/273 out-of-sample weeks, Iquitos 11/143, Singapore 13/169. The last 5 San Juan and 10 Iquitos rows were unused because they did not fill another 13-week block.
- **Relative MAE:** model MAE divided by persistence MAE for that series. Below 1 means lower MAE than persistence. Counts are not pooled between series.
- **DM tests:** two-sided Diebold–Mariano on pointwise absolute-error differentials, horizon 1 with the Harvey–Leybourne–Newbold small-sample correction. Comparison A tests all 20 level/change × model × window configurations against persistence. Comparison B tests each model's best short window (selected from 1/2/4 by observed OOS MAE) against its best long window (selected from 8/12) for each target type. Holm correction covers all 24 p-values within each series. Best-window comparisons use the same OOS sample for selection and testing, so they are exploratory and may favor the selected configurations.
- **Seasonality sensitivity:** after selecting best short and best long configurations using the no-seasonality results, only those selected configurations were refit with `sin(2πw/52)` and `cos(2πw/52)` added. Singapore's original epidemiological week number is used; the DengAI series use ISO calendar week from their source date.
- **Fixed environment:** Python 3.12.5; NumPy 2.5.3; pandas 3.0.5; SciPy 1.18.1; scikit-learn 1.9.1; XGBoost 3.4.1; LightGBM 4.7.0. Python `random`, NumPy and model seeds are 42. Exact settings and versions are in `results/run_manifest.json`.

## Best observed configurations

MAE is shown in cases per week; relative MAE is in parentheses. Persistence MAE for each series is listed at the top. “Best short” and “best long” are chosen separately by target type and model.

| Series | Persistence MAE | Forecast target | Best short configuration | Best long configuration | Best observed overall |
|---|---:|---|---|---|---|
| San Juan | 6.16 | Direct count | LightGBM, 4 wk: 6.01 (0.975) | XGBoost, 12 wk: 6.59 (1.070) | LightGBM, 4 wk: 6.01 (0.975) |
| San Juan | 6.16 | Log change | LightGBM, 2 wk: 5.97 (0.969) | XGBoost, 8 wk: 6.35 (1.030) | LightGBM, 2 wk: 5.97 (0.969) |
| Iquitos | 4.66 | Direct count | LightGBM, 1 wk: 4.60 (0.986) | XGBoost, 12 wk: 5.27 (1.129) | LightGBM, 1 wk: 4.60 (0.986) |
| Iquitos | 4.66 | Log change | LightGBM, 1 wk: 4.60 (0.986) | XGBoost, 12 wk: 5.49 (1.178) | LightGBM, 1 wk: 4.60 (0.986) |
| Singapore | 54.70 | Direct count | XGBoost, 1 wk: 73.35 (1.341) | XGBoost, 8 wk: 86.87 (1.588) | XGBoost, 1 wk: 73.35 (1.341) |
| Singapore | 54.70 | Log change | LightGBM, 4 wk: 57.54 (1.052) | LightGBM, 8 wk: 60.88 (1.113) | LightGBM, 4 wk: 57.54 (1.052) |

Across all 20 model/window combinations per series (10 direct-count and 10 change), relative MAE was below 1 for 5 San Juan configurations, 2 Iquitos configurations, and **0 Singapore configurations**. In Singapore, the change target reduced the best observed MAE from 73.35 to 57.54 cases, but persistence remained lower at 54.70.

## Statistical test summary

| Series | DM tests significant after Holm (of 24) | Direction of significant results | Best short vs best long after Holm |
|---|---:|---|---|
| San Juan | 0 | — | None significant; adjusted p-values 0.323–1.000 |
| Iquitos | 0 | — | None significant; adjusted p-values 1.000 |
| Singapore | 11 | All 11 were worse than persistence | None significant; adjusted p-values 0.195–0.708 |

The 11 Singapore results are direct or change configurations with larger absolute errors than persistence. No model configuration significantly beat persistence in any series after within-series Holm correction. The “best short vs best long” comparisons also do not identify a clear winner.

DM horizon 1 uses the contemporaneous variance of the weekly loss differential; no additional autocorrelation lags are included. Weekly forecast losses can still be serially dependent, so p-values should be read cautiously. The selected best-short/best-long comparisons are post-selection and are descriptive rather than clean confirmatory tests.

## Seasonality sensitivity

The selected seasonal configurations improved MAE in 6/8 selected San Juan configurations, 5/8 Iquitos configurations, and 3/8 Singapore configurations. The direction was not consistent. In Singapore, every selected seasonal and non-seasonal configuration remained above persistence. Adding the two calendar features therefore did not change the project-level conclusion or establish a window winner, though it did alter some individual point estimates.

Full comparisons, including MAE/RMSE and relative MAE for every model/window, are in `results/all_models_relative_mae_summary.csv`; seasonal comparisons are in `results/seasonality_sensitivity.csv`.

## Singapore source overlap check

The selected Singapore series was checked against the two MOH sources available in the project materials. Over **2014-W01 to 2018-W52 (261 weeks)**:

- The broader 2012–2022 MOH *Weekly Infectious Disease Bulletin Cases* `Dengue Fever` series matches all **261/261** project values exactly.
- The separate MOH *Weekly Number of Dengue and Dengue Haemorrhagic Fever Cases* file has 209 exact `Dengue` matches and **52 differing weeks** (19.9%), with differences from -8 to +1 cases (dedicated file minus project value). Thirty-seven of those mismatches occur in 2016. Adding DHF does not reconcile all differences. The reason for the official-source discrepancy is not established from the available files.

There are no missing weeks. The selected series was not overwritten. See `singapore_overlap_results.md`, `tables/overlap_check_2014_2018.csv`, and `tables/overlap_mismatched_weeks_2014_2018.csv`.

## Reproduction check of previous results

The original pipeline was rerun in an isolated project copy for San Juan and Iquitos, and the Singapore external-validation runner was rerun from code. The 15 San Juan result CSVs and 25 Singapore result CSVs reproduced their reported metric values exactly (excluding measured run-time columns and the derived run-time ratio, which naturally vary). All 10 San Juan and 19 Singapore PNG figures were pixel-identical to the committed figures. The Iquitos single-split table reproduced the values in `research_notes.md` §4.4 to the reported two decimal places; its persistence MAE was 2.97.

The Brazil experiment was also rerun from its script: metric tables matched the earlier outputs (excluding timing fields) and all 10 generated figures were pixel-identical. All new tables and the three level/change relative-MAE figures are generated by `run_change_forecast.py` from the frozen inputs.

## Limitations and conclusion

1. The window comparisons concern one-week-ahead forecasts using two fixed tree models and univariate lags. They do not establish results for other models, longer horizons, other diseases or India.
2. CV test points are consecutive weeks from one series, and adjacent folds share expanding training histories. DM's standard horizon-1 variance assumption may not capture serial dependence in weekly loss differences.
3. The best-short/best-long configurations were selected by the same OOS MAE used for the test; this makes those p-values exploratory.
4. The Singapore source mismatch remains unexplained, although the broad weekly-bulletin source agrees exactly with the selected project series.

**Final conclusion:** no single history length (1, 2, or 4 weeks) wins consistently. Short windows were often more competitive than long ones, but the winning point estimate and whether a model beat persistence varied by series and target. Predicting log change helped Singapore but did not outperform persistence. The project should report “no clear universal winner; short history is often sufficient in these experiments,” not select one number as the answer.
