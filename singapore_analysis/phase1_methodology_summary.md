# Phase 1 — Methodology summary of the original study

Prepared 2026-11-17 from a full read of `research_notes.md`, `run_experiment.py`,
and every module in `src/`. Nothing in the original project was modified.
This is a summary of what the code and notes actually do; where the two differ
in wording, the code is treated as authoritative.

## 1. Research question and design principle

How many weeks of past surveillance data are needed to forecast next week's
dengue case count? History windows of **1, 2, 4, 8 and 12 weeks** are compared
(`features.HISTORY_WINDOWS`). Model settings are held fixed so that history
length is the only variable that changes. The study is explicitly not a search
for the best dengue model.

## 2. Data and preprocessing

| Item | Original study |
|---|---|
| Source | NOAA/CDC Dengue Forecasting Project (2015), DrivenData "DengAI" distribution |
| Primary series | San Juan, 936 weeks, 1990-04-30 → 2008-04-22 |
| Secondary series | Iquitos, 520 weeks, 2000–2010 |
| Target column | `total_cases` |
| Time index | `week_start_date` (the `year`/`weekofyear` columns were rejected because of a stray week 53) |
| Covariates | None. The only predictor is the series' own past |
| Transformation | None. Raw integer counts |
| Preprocessing (`preprocess`) | Sort by date, drop duplicate dates keeping the first, cast to int, reset index |
| Imputation | None (no missing values were found) |

`audit_series` checks ordering, duplicate dates, missing values, negative
values and the spacing between consecutive weeks. A gap of 14 days or more is
treated as a true missing week; the 8–9 day year-end gaps in DengAI are not.

## 3. Target construction and history windows

From `features.make_lag_features`:

- `lag_k = cases.shift(k − 1)`, so `lag_1` is the most recent observed week *t*.
- `target = cases.shift(−1)`, the count in week *t + 1*. Horizon is one week.
- The row's `date` is the date of week *t* (the last input week), not the target week.
- Window *W* uses `lag_1 … lag_W`. Windows are nested: a short window's
  features are a strict subset of a longer one's.
- Rows with any undefined lag or target are dropped.

**Alignment** (`build_aligned_tables`): every window is restricted to the rows
the longest window supports, so all windows are trained and scored on the same
target weeks. For a series of *N* weeks and a maximum window of 12 this leaves
*N − 12* rows (San Juan: 936 → 924).

## 4. Models and exact hyperparameters

From `models.get_models`, seed `RANDOM_SEED = 42`:

| Setting | XGBoost (`XGBRegressor`) | LightGBM (`LGBMRegressor`) |
|---|---|---|
| `n_estimators` | 300 | 300 |
| `learning_rate` | 0.05 | 0.05 |
| `max_depth` | 4 | 4 |
| `random_state` | 42 | 42 |
| `subsample` | 0.9 | 0.9 |
| `colsample_bytree` | 0.9 | 0.9 |
| `n_jobs` | 1 | 1 |
| `min_child_samples` | library default | 10 |
| Other | `verbosity=0` | `verbose=-1` |

Everything not listed is the library default (so the objective is squared
error, LightGBM `num_leaves` is 31, and `subsample` has no effect in LightGBM
because `subsample_freq` is left at 0). Pinned versions: xgboost 3.4.1,
lightgbm 4.7.0, scikit-learn 1.9.1, pandas 3.0.5, numpy 2.5.3, Python 3.14.7.
A fresh model object is created for every fit.

## 5. Persistence baseline

`evaluation.naive_baseline`: prediction = `lag_1`, scored against `target` on
the same test rows as the models. It uses only `lag_1`, so it is identical for
every window once windows are aligned.

## 6. Validation methodology

**Single chronological split** (`chronological_split`): 70% train, 15%
validation, 15% test by row position on the aligned table. San Juan: 646 / 139 / 139.
The validation block is not used for tuning; it serves as a second evaluation
period in `robustness_check`.

**Rolling-origin cross-validation** (`rolling_origin_splits`, `run_cv_experiment`):

- Expanding training window, non-overlapping test blocks, walk-forward.
- `min_train = int(n_aligned × 0.70)`.
- Fold *i*: train on rows `[0, min_train + i·block)`, test on the next `block` rows.
- Fold count = `(n_aligned − min_train) // block`. Leftover rows are dropped, not padded.
- Main run: `CV_TEST_BLOCK = 26` weeks. San Juan gives 10 folds, 18 weeks unused.
  The 26-week block was chosen as the only value in a 20–26 week range giving 8–10 folds *for San Juan*.
- Follow-up run: `FINE_CV_TEST_BLOCK = 13` weeks, windows {1, 2, 4} only,
  aligned against the full window set so the 924-row base is unchanged.
  San Juan gives 21 folds, 5 weeks unused.
- On Iquitos (508 aligned rows) the 8–10 fold target is not met; the pipeline reports this instead of forcing it.

Metrics (`evaluate`): MAE, RMSE and R². CV conclusions rest on MAE and RMSE
because per-fold R² was found unreliable at 26- and 13-week blocks.

## 7. Statistical comparison and the 5% rule

- **Pre-registered rule** (`TOLERANCE = 0.05`): a window is *near-optimal* if its
  MAE ≤ best MAE × 1.05 for that model. The *minimum sufficient window* is the
  shortest near-optimal window.
- **CV version** (`cv_minimum_sufficient_window`): the same rule on CV mean MAE,
  plus a flag for whether the chosen window's mean is within one fold-level
  standard deviation of the best window's mean.
- **CV summary** (`cv_summary`): mean and sample standard deviation across folds
  of MAE, RMSE, R² per (window, model).
- **Fold win counts** (`fold_win_counts`): per fold, which window had the lowest MAE.
- **Paired fold differences** (`paired_fold_differences`, `paired_diff_summary`):
  per-fold `MAE_a − MAE_b` for pairs (1,2), (1,4), (2,4); mean, standard
  deviation, wins and win rate.
- **No hypothesis tests, p-values or confidence intervals** were computed in the
  original study. The paired comparisons are stated to be descriptive, with no
  multiplicity correction (limitation 14). Medians were not reported.

## 8. Original findings

1. Single split, San Juan test period: longer windows were worse (4 → 8 weeks raised MAE by 18–22%).
2. That pattern did not repeat on the adjacent validation period.
3. Only 1 of 10 model/window configurations beat persistence on the San Juan test set; none did on Iquitos.
4. 10-fold CV: windows 1, 2 and 4 are within the 5% band for both models; 8 and 12 are outside it. Minimum sufficient window by the rule is 1 week for both models. Fold-level standard deviation is 62–89% of mean MAE.
5. 21-fold follow-up: windows 1, 2 and 4 could not be separated. No window is called optimal.

**Conclusion as stated:** a short-history regime (1–4 weeks) is at least as
good as long history (8–12 weeks) for this dataset and configuration; the exact
minimum within {1, 2, 4} is not identified.

## 9. Original limitations that carry over

One disease and one dataset family; fixed hyperparameters by design; one-step
horizon only; no climate covariates; raw untransformed counts; R² unreliable per
fold; CV block size calibrated to San Juan; folds are slices of one period, not
independent datasets; seasonal-phase heterogeneity between 13-week folds; paired
comparisons descriptive only; results specific to this series and configuration.

## 10. Points that matter for adding Singapore

| Point | Consequence |
|---|---|
| The pipeline expects a `date` column and a `cases` column | Singapore has epi-week labels but no dates. An adapter is needed (see the protocol) |
| `data_loader.load_raw` is DengAI-specific | Singapore needs its own loader. `data_loader.py` should not be edited |
| CV fold counts are derived from the aligned row count | Singapore's fold count will differ from 10 and 21 (see the protocol) |
| Confidence intervals, medians and formal tests are not in the original | Anything of that kind for Singapore is an addition and must be labelled as one |
| Outputs are written to `results/` with fixed filenames | Singapore outputs must go to a separate directory so nothing is overwritten |
| `SOURCE.md` describes DengAI counts as "laboratory-confirmed" | The upstream metadata is more specific (see the comparability document) |
