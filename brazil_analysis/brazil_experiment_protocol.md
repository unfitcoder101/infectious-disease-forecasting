# Brazil external validation experiment protocol

## Purpose

Apply the existing project method to one independent weekly dengue series to assess whether the short-history pattern is also observed in Rio de Janeiro. This is an additional external validation, not a pre-registered experiment and not evidence that the result generalizes to every dengue surveillance system.

## Fixed analysis choices

- Dataset: InfoDengue reported weekly dengue cases (`casos`) for Rio de Janeiro municipality, using the dated snapshot described in `SOURCE.md`.
- Forecast target: next epidemiological week's reported case count.
- History windows: 1, 2, 4, 8, and 12 previous weeks.
- Models: XGBoost and LightGBM with the unchanged settings in `src/models.py` and seed 42.
- Baseline: persistence, predicting next week with the most recent observed weekly count.
- Alignment: evaluate every history window on identical target weeks.
- Single split: chronological 70% train, 15% validation, 15% test, as in the existing pipeline.
- Primary rolling-origin evaluation: expanding training set with non-overlapping 26-week test blocks and minimum training size of 70% of aligned observations. The number of complete folds follows from the data length; leftover rows are not padded.
- Secondary rolling-origin evaluation: 13-week blocks, same minimum training fraction, to check ranking sensitivity at a finer block size. This is descriptive and its adjacent folds share expanding training data.
- Metrics and summaries: MAE, RMSE, R-squared, training/inference time, fold win counts, paired fold-level MAE differences, persistence comparison, and the project's existing 5% near-optimal/minimum-sufficient rule.
- No tuning, covariates, imputation, pooling, or changes to San Juan, Iquitos, or Singapore data/results.

The 5% rule is inherited from the original project. It is descriptive rather than a significance test: the shortest window in the 5% band is termed minimum sufficient, not a proven unique optimum. Report the full error pattern and the persistence comparison alongside that label.
