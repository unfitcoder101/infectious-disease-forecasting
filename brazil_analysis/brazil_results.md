# Brazil external validation: Rio de Janeiro dengue

## Research question and status

This is a post-hoc external validation of the project's history-window experiment, not a preregistered replication. It asks whether the San Juan/Iquitos/Singapore pattern—short histories often being competitive with 8–12 weeks—also appears in an independent Brazilian municipal surveillance series. All results remain in `brazil_analysis/`; no existing dataset, result, or conclusion was overwritten.

## Data and reproducibility

The series is InfoDengue reported dengue cases (`casos`) for Rio de Janeiro municipality (IBGE geocode 3304557), epidemiological weeks 2010–2025. The API snapshot contains 835 weekly observations (2010-01-03 to 2025-12-28), no duplicate dates, missing counts, negative counts, or gaps in weekly spacing. Counts range from 3 to 16,339 (median 171). InfoDengue revises observations retrospectively; this analysis uses the dated snapshot retrieved 2026-11-18, not the nowcast field `casos_est`. The API definition, query, and checksum are recorded in [`SOURCE.md`](SOURCE.md).

The model protocol holds the existing lag windows (1, 2, 4, 8, 12), XGBoost and LightGBM settings, and persistence baseline fixed. Forecast target is next week. Windows share aligned target weeks. No covariates, tuning, transformations, or cross-city pooling were added. One chronological 70/15/15 split and expanding-window CV with 26-week (primary; 9 folds) and 13-week (secondary; 19 folds) test blocks were run. The CV designs are descriptive: folds reuse training history and are not independent samples. The 5% minimum-sufficient rule is a practical tolerance, not a hypothesis test.

## Main rolling-origin results

Mean MAE, in Rio reported cases per week:

| History | XGBoost, 26 wk (9 folds) | LightGBM, 26 wk | XGBoost, 13 wk (19 folds) | LightGBM, 13 wk |
|---:|---:|---:|---:|---:|
| 1 | 217.91 | 210.45 | 207.37 | 199.31 |
| 2 | 244.52 | 208.66 | 229.04 | 198.76 |
| 4 | **186.11** | **202.73** | **186.64** | **195.38** |
| 8 | 198.35 | 232.49 | 192.75 | 220.64 |
| 12 | 200.34 | 231.72 | 195.08 | 217.01 |
| Persistence | 195.06 | 195.06 | 186.72 | 186.72 |

Four weeks gave the lowest mean MAE for both models in both fold designs. In both designs, the shortest window within 5% of each model's best mean was 4 weeks for XGBoost and 1 week for LightGBM. Thus, the rule does not identify one common minimum for both algorithms. On the 13-week folds, all three short LightGBM windows (1, 2, 4) were within 5%; XGBoost's 4, 8, and 12 were within 5%. Fold winners also varied: for 13-week folds, XGBoost's 12-week model won 8/19 folds, while LightGBM's 1-week model won 9/19. A favorable mean therefore does not mean a window consistently wins across time.

Dispersion was substantial: the standard deviation of fold MAE ranged from about 289 to 447 cases in the primary 26-week CV, and from about 400 to 566 in the secondary 13-week CV. These standard deviations are much larger than differences between the window means. The 13-week design's mean persistence MAE (186.72) was also lower than either model's mean at every window, although the 26-week XGBoost mean at 4 weeks (186.11) was marginally below persistence (195.06). The baseline should therefore remain central to any operational interpretation.

## Single chronological test split

On the held-out 124-week test period, persistence MAE was 319.77 cases. XGBoost at 4, 8, and 12 weeks beat it (MAEs 268.66, 302.99, and 307.05); its 4-week result was 15.98% lower than persistence. LightGBM at 4 weeks was nearly tied (318.77, 0.31% lower), while its 1-, 2-, 8-, and 12-week results were worse. Both models' validation-period MAEs were worse than persistence for every window. The split's most extreme test peak reached 16,339 cases. This regime dependence makes the test-period ranking a weak basis for claiming a generally best history length.

## Interpretation for the project question

Brazil supports the tentative short-history pattern: 8–12 weeks did not improve average error over 1–4 weeks in these rolling-origin designs, and 4 weeks had the lowest mean for both models. It does **not** establish that 4 weeks is best, nor that one minimum history length works across algorithms or datasets. The rule yields different minimum-sufficient windows by model, fold winners are mixed, the dispersion is very large, and persistence is strong. The 26-week and 13-week CV means agree on the lowest mean but differ in persistence comparisons; the held-out split further shows validation performance can differ from test performance.

Taken with the other settings, the defensible project-level statement is: **in these four dengue surveillance settings and under these fixed one-step forecasting designs, using more than 4 weeks of cases did not provide a consistent improvement; 1–4 weeks was often competitive, but the exact sufficient window varied by dataset, model, period, and evaluation design.** This is evidence about this experiment, not a general law about machine learning or infectious-disease forecasting. The four series use different surveillance definitions, revisions, periods, and scales. Keep all errors within their own series; do not pool raw errors or compare their absolute MAEs.

## Limitations

1. This Brazil analysis was added after examining the existing project; it was not preregistered.
2. InfoDengue counts are retrospectively revised, so the snapshot is a version of record, not an immutable final truth. The checksum preserves this exact input.
3. Rio is one municipality and one pathogen. This does not validate transfer to India, COVID-19, or other diseases.
4. The large epidemic peaks dominate raw-count error; tree models cannot extrapolate beyond observed target ranges in the same way a mechanistic or explicitly extrapolative model might.
5. CV folds share expanding training data. Their SDs describe temporal variability; they are not independent-sample uncertainty estimates. We do not claim inferential significance from mean rankings.
6. The 5% rule is a descriptive threshold inherited from the project, not a formal test.
7. Only one-week-ahead univariate forecasts with fixed model settings were evaluated. No covariates or tuning were used.

## Artifacts

- Frozen source: `data/raw/infodengue_rio_2010_2025_snapshot_20261118.csv`
- Processed weekly series: `data/processed/rio_weekly_cases.csv`
- Protocol: [`brazil_experiment_protocol.md`](brazil_experiment_protocol.md)
- Tables and fold-level results: `results/brazil_*.csv`
- Figures: `results/figures/`
- Runner: `run_brazil_experiment.py`

## References

- InfoDengue. *Tutorial: Using the API with Python (English).* https://info.dengue.mat.br/tutorial_api_python/locale-en (accessed 2026-11-18).
- InfoDengue. *API services and data dictionary.* https://info.dengue.mat.br/services/api (accessed 2026-11-18).
- InfoDengue API query used for the frozen Rio series: https://info.dengue.mat.br/api/alertcity?geocode=3304557&disease=dengue&format=csv&ew_start=1&ew_end=53&ey_start=2010&ey_end=2025 (retrieved 2026-11-18).
- The main project repository and code history: https://github.com/unfitcoder101/infectious-disease-forecasting; implementation review: https://github.com/unfitcoder101/infectious-disease-forecasting/pull/1 (branch `add-brazil-external-validation`).
