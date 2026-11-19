# Singapore external validation — experimental protocol (Phases 5 and 6)

**Written:** 2026-11-17, before any model was trained on Singapore data.
**Status:** awaiting review. **No model has been trained.**

This protocol fixes every analytic choice in advance, so that nothing is chosen
after the results are seen. The original study did the same for its 5% rule.

## 1. Purpose and claim

To test whether the original study's finding, that short surveillance histories
(1–4 weeks) are at least as good as long ones (8–12 weeks) for one-week-ahead
dengue forecasting, also holds on an independent surveillance system.

Singapore is an **external validation with a different operational case
definition**, not a replication (see `singapore_target_comparability.md`).
Results will be reported separately and never pooled with San Juan or Iquitos.

## 2. What is unchanged from the original study

All of the following are imported directly from the existing `src/` modules.
No file under `src/`, `results/` or `data/` is edited or overwritten.

| Element | Value | Source in code |
|---|---|---|
| History windows | 1, 2, 4, 8, 12 weeks | `features.HISTORY_WINDOWS` |
| Forecast horizon | One week ahead: target = cases(t + 1) | `features.make_lag_features` |
| Features | `lag_1 … lag_W`, raw counts, nested across windows | `features.make_lag_features` |
| Alignment | All windows restricted to the rows the 12-week window supports | `features.build_aligned_tables` |
| Covariates and transformations | None | — |
| XGBoost | 300 trees, learning rate 0.05, depth 4, subsample 0.9, colsample_bytree 0.9, seed 42, 1 thread | `models.get_models` |
| LightGBM | Same, plus `min_child_samples=10` | `models.get_models` |
| Baseline | Persistence: prediction = `lag_1`, scored on the same test rows | `evaluation.naive_baseline` |
| Metrics | MAE and RMSE (R² is computed by the code but not used for conclusions) | `evaluation.evaluate` |
| Validation | Expanding-window rolling-origin CV, non-overlapping test blocks | `evaluation.rolling_origin_splits` |
| Minimum training size | `int(aligned rows × 0.70)` | `experiment.CV_TRAIN_FRAC` |
| Near-optimal rule | MAE within 5% of the best window's MAE; minimum sufficient window = shortest near-optimal | `experiment.TOLERANCE`, `cv_minimum_sufficient_window` |
| Library versions | xgboost 3.4.1, lightgbm 4.7.0, scikit-learn 1.9.1, pandas 3.0.5, numpy 2.5.3 | `requirements.txt` |

**Hyperparameters are not tuned for Singapore.**

## 3. Data handling

| Step | Decision |
|---|---|
| Input | `data/raw/singapore_dengue_fever_weekly_clean.csv`, column `dengue_fever_cases`, 574 weeks, used as supplied |
| Loader | New module `singapore_analysis/singapore_data.py`. `src/data_loader.py` is DengAI-specific and is not edited |
| Time index | The pipeline needs a `date` column. It is derived as the Sunday starting each epi week (Sunday–Saturday weeks, week 1 has at least 4 days in January). The loader asserts the result is one gapless 7-day sequence |
| Does the derived date affect results? | No. Models receive lag values only, and row order equals the file's `time_index` order |
| Week 53 | 2014-W53 and 2020-W53 are kept as ordinary weeks |
| Cleaning | None. `preprocess()` is called for consistency and is asserted to change nothing |
| Imputation, outlier removal, smoothing | None |
| DHF | Not added |

## 4. Resulting data layout

Verified by the dry run (`run_singapore_experiment.py` without `--execute`),
which builds the tables and folds and trains nothing.

| Quantity | San Juan (original) | Singapore |
|---|---|---|
| Observations | 936 | 574 |
| Aligned rows (all windows) | 924 | **562** |
| First target week | — | 2012-W13 |
| Last target week | — | 2022-W52 |
| `min_train` | 646 | **393** |
| Weeks available for testing | 278 | **169** |

Checks passed in the dry run: all five windows have identical rows and identical
targets; `target` equals the next week's count; `lag_1` equals the current
week's count; `lag_12` equals the count 11 weeks before the current week.

## 5. How many folds the data supports

The original fold counts (10 and 21) came from San Juan's 278 test weeks.
Singapore has 169. Fold count = `169 // block`, with nothing padded or forced
(`tables/protocol_fold_count_options.csv`):

| Test block (weeks) | Folds | Unused weeks | Within the original 8–10 target |
|---|---|---|---|
| 13 | **13** | 0 | No |
| 20 | 8 | 9 | Yes |
| 21 | 8 | 1 | Yes |
| 22 | 7 | 15 | No |
| 23 | 7 | 8 | No |
| 24 | 7 | 1 | No |
| 25 | 6 | 19 | No |
| 26 | **6** | 13 | No |

**21 folds are not available.** Getting 21 folds from 169 weeks would need
8-week blocks, or a smaller `min_train`. Either would change the method. The
fold count is reported as it falls.

### Decision D1 (needs your approval): which block sizes to use

The original made two choices that give different answers here: a fixed 26-week
block in code, and a stated aim of 8–10 folds.

| Option | Blocks | Folds | For | Against |
|---|---|---|---|---|
| **A (proposed)** | 26 and 13 weeks, the code's constants unchanged | 6 and 13 | No parameter changes. Same block lengths as San Juan, so "a fold" means the same span of time. Same treatment the original gave Iquitos, which also missed the 8–10 target and was reported as such | Only 6 folds in the main design |
| B | 20 or 21 weeks, re-deriving the block from the 8–10 fold aim | 8 | Meets the stated fold aim | Changes `CV_TEST_BLOCK`. Two block sizes qualify, so one must be picked, which is a new free choice. Blocks no longer match San Juan's |

Option A is proposed. The 13-week design, with 13 folds and no unused weeks,
carries the statistical comparisons, so the low fold count of the 26-week design
matters less.

## 6. Experiments to run

All three use all five windows, both models and persistence.

### 6.1 Single chronological split (as in the original §4)

70 / 15 / 15 by row (`tables/protocol_single_split.csv`):

| Part | Rows | Target weeks | Mean target | Max target |
|---|---|---|---|---|
| Train | 393 | 2012-W13 to 2019-W40 | 222.0 | 888 |
| Validation | 84 | 2019-W41 to 2021-W19 | 489.8 | 1,791 |
| Test | 85 | 2021-W20 to 2022-W52 | 410.3 | 1,563 |

As in the original, the validation block is not used for tuning. It is a second
evaluation period for the robustness check. In this design the models are
trained once on data ending 2019-W40, so both evaluation periods contain values
far above anything in training. This run is included for continuity with the
original and is expected to be the least informative of the three.

### 6.2 Rolling-origin CV, 26-week blocks: 6 folds, 13 weeks unused

`tables/protocol_fold_plan_26w.csv`:

| Fold | Train rows | Test target weeks | Test mean | Test max | Max seen in training | Test weeks above training max |
|---|---|---|---|---|---|---|
| 0 | 393 | 2019-W41 to 2020-W14 | 325 | 402 | 888 | 0 |
| 1 | 419 | 2020-W15 to 2020-W40 | 988 | 1,791 | 888 | **13** |
| 2 | 445 | 2020-W41 to 2021-W13 | 245 | 591 | 1,791 | 0 |
| 3 | 471 | 2021-W14 to 2021-W39 | 96 | 137 | 1,791 | 0 |
| 4 | 497 | 2021-W40 to 2022-W13 | 162 | 510 | 1,791 | 0 |
| 5 | 523 | 2022-W14 to 2022-W39 | 952 | 1,563 | 1,791 | 0 |

Unused: target weeks 2022-W40 to 2022-W52.

### 6.3 Rolling-origin CV, 13-week blocks: 13 folds, 0 weeks unused

`tables/protocol_fold_plan_13w.csv`:

| Fold | Train rows | Test target weeks | Test mean | Test max | Max seen in training | Test weeks above training max |
|---|---|---|---|---|---|---|
| 0 | 393 | 2019-W41 to 2020-W01 | 283 | 372 | 888 | 0 |
| 1 | 406 | 2020-W02 to 2020-W14 | 367 | 402 | 888 | 0 |
| 2 | 419 | 2020-W15 to 2020-W27 | 781 | 1,460 | 888 | **4** |
| 3 | 432 | 2020-W28 to 2020-W40 | 1,194 | 1,791 | 1,460 | **4** |
| 4 | 445 | 2020-W41 to 2020-W53 | 346 | 591 | 1,791 | 0 |
| 5 | 458 | 2021-W01 to 2021-W13 | 144 | 194 | 1,791 | 0 |
| 6 | 471 | 2021-W14 to 2021-W26 | 116 | 137 | 1,791 | 0 |
| 7 | 484 | 2021-W27 to 2021-W39 | 77 | 109 | 1,791 | 0 |
| 8 | 497 | 2021-W40 to 2021-W52 | 68 | 104 | 1,791 | 0 |
| 9 | 510 | 2022-W01 to 2022-W13 | 256 | 510 | 1,791 | 0 |
| 10 | 523 | 2022-W14 to 2022-W26 | 1,141 | 1,563 | 1,791 | 0 |
| 11 | 536 | 2022-W27 to 2022-W39 | 762 | 1,164 | 1,791 | 0 |
| 12 | 549 | 2022-W40 to 2022-W52 | 312 | 397 | 1,791 | 0 |

Fold construction was verified programmatically for both designs: every window
has identical train and test rows in every fold; test blocks are contiguous,
non-overlapping and forward-chaining; training grows by exactly one block per
fold; no test target appears in any training set.

### Decision D2 (needs your approval): windows in the 13-week run

The original 13-week follow-up trained only windows {1, 2, 4}, because 8 and 12
had already been judged worse on San Juan. Nothing has been judged on Singapore,
so dropping 8 and 12 here would assume the conclusion under test. **The proposal
is to run all five windows in the 13-week design**, and to extend the paired
comparisons from the original three pairs to all ten. This is a deviation from
the original follow-up and is flagged as one.

## 7. Statistical evaluation design (Phase 6)

### 7.1 Carried over unchanged from the original

Computed by the original functions, per model and window:

- Fold-level MAE and RMSE.
- Mean and standard deviation across folds (`cv_summary`).
- Fold win counts (`fold_win_counts`).
- Near-optimal windows and minimum sufficient window by the 5% rule on CV mean
  MAE, with the "within one standard deviation of the best" flag
  (`cv_minimum_sufficient_window`).
- Paired fold-level MAE differences: mean, standard deviation, wins
  (`paired_fold_differences`, `paired_diff_summary`).
- Comparison with persistence on identical folds.

### 7.2 Additions, not part of the original study

The original reported no medians, no confidence intervals and no hypothesis
tests. The following are **new** and are implemented in
`singapore_analysis/singapore_stats.py`. The module was checked against the
existing San Juan fold file (read-only): it reproduces the original paired means,
standard deviations and win counts to four decimals.

| Addition | Definition |
|---|---|
| Median, minimum and maximum of fold MAE and RMSE | Per model and window |
| Per-window table for the 5% rule | Each window's percentage above the best mean MAE, and whether it is inside the band |
| Paired comparisons for all 10 window pairs | Per model, per-fold `MAE_shorter − MAE_longer`; also for RMSE |
| Paired comparison with persistence | Per model and window, per-fold `MAE_model − MAE_persistence` |
| Short-versus-long contrast | Per model, per-fold mean MAE of windows {1, 2, 4} minus mean MAE of windows {8, 12} |
| 95% confidence interval of each paired mean difference | Student-t interval across folds, and a percentile bootstrap resampling folds (10,000 resamples, seed 42) |
| Test | Two-sided Wilcoxon signed-rank test on the per-fold paired differences |
| Multiplicity | Holm adjustment within each family (below) |

### 7.3 Families of comparisons, fixed in advance

| Family | Comparisons | Count |
|---|---|---|
| **Primary** | Short-versus-long contrast, one per model | 2 |
| Secondary: window pairs | All pairs of the five windows, per model | 20 |
| Secondary: persistence | Each model at each window against persistence | 10 |

The primary family maps directly onto the original study's "confident"
conclusion. The secondary families are exploratory.

### 7.4 What may be called significant

- A difference is called statistically significant only if its **Holm-adjusted
  Wilcoxon p-value is below 0.05 in the 13-fold design**.
- **The 6-fold design cannot support any significance claim.** With 6 folds the
  smallest possible two-sided Wilcoxon p-value is 2/2⁶ = 0.031, which cannot
  pass a Holm adjustment over even two comparisons. Its tests will be computed
  and shown but treated as descriptive.
- With 13 folds the smallest possible p-value is 2/2¹³ = 0.00024, so the test
  can in principle detect a consistent effect in all three families.
- If nothing meets the criterion, the report will say the comparison is
  **not resolved**, as the original did. It will not say the windows are
  equivalent.
- "Lowest observed mean" will be kept distinct from "evidence of being better",
  following the original's wording.

### 7.5 Known weaknesses of these statistics

| Weakness | Consequence |
|---|---|
| Folds are consecutive blocks of one series and share most of their training data | They are not independent. Intervals are probably too narrow and p-values too small |
| 13 folds is a small sample; fold MAE will be very unequal (test means run from 68 to 1,194 cases) | The mean difference will be driven by three or four epidemic folds. The median and the win count are reported for that reason, and the Wilcoxon test is used in place of a t-test |
| The t-interval assumes roughly normal differences | Unlikely to hold. The bootstrap interval is reported alongside it; if they disagree materially, neither is relied on |
| The 13 folds cover one 3.25-year period | More folds would be a finer grid over the same period, not new information (the original's limitation 11) |
| The 5% rule is applied to mean MAE | It can be decided by one epidemic fold. The per-fold tables will show whether that happened |

### 7.6 Pre-specified reading of the outcome

| Outcome in Singapore | Reading |
|---|---|
| Minimum sufficient window is 1, 2 or 4 for both models in both CV designs, and the primary contrast does not favour long windows | Consistent with the original finding on an independent system |
| Minimum sufficient window is 8 or 12 for either model in both CV designs, and the primary contrast favours long windows with a Holm-adjusted p below 0.05 | Evidence against the finding transferring |
| The two CV designs disagree, or the models disagree, or neither condition above is met | Not identified. Reported as such |

The comparison with persistence is reported separately. Whether the models beat
persistence does not enter the reading above.

## 8. Risks identified before running

| # | Risk | Handling |
|---|---|---|
| R1 | **Extrapolation.** Tree models cannot predict above the largest training target. 13 of 26 test weeks in 26-week fold 1 (and 4 each in 13-week folds 2 and 3) exceed the training maximum | Not corrected, since any fix would change the method. Affects all windows alike, so it mainly disadvantages the models against persistence. The affected folds will be identified in every results table. No fold is excluded |
| R2 | **Non-representative evaluation period.** It holds two record epidemics and the COVID-19 years | Stated with every result. No claim of typical performance |
| R3 | **Fewer training rows** (393–549 against San Juan's 646–906) at the same model capacity | Hyperparameters stay fixed. Noted as a reason longer windows may overfit more here |
| R4 | **Scale.** Errors are in cases and Singapore's counts are about 11 times San Juan's | No cross-dataset comparison of MAE or RMSE |
| R5 | **Epidemic folds dominate mean MAE** | Medians, win counts and paired comparisons reported alongside means |
| R6 | **Reported fold dates.** The original's `fold_test_start` and `fold_test_end` are dates of the *input* week, one week before the target | Kept as they are. Target-week labels are added as extra columns in the Singapore output |

## 9. Deviations from the original, in one place

| # | Deviation | Why | Needs approval |
|---|---|---|---|
| V1 | New loader with a derived `date` column | The file has no dates and the DengAI loader cannot read it | No (no effect on results) |
| V2 | 6 and 13 folds, not 10 and 21 | Fewer test weeks; nothing forced | **Yes (D1)** |
| V3 | All five windows in the 13-week design, all ten pairs | Dropping 8 and 12 would presuppose the result | **Yes (D2)** |
| V4 | Medians, confidence intervals, Wilcoxon tests, Holm adjustment, primary contrast | Requested for this phase; absent from the original | **Yes (D3)** |
| V5 | Outputs go to `singapore_analysis/results/` with an `sg_` prefix | Prevents overwriting San Juan and Iquitos files | No |
| V6 | `visualization.generate_all` is not called | Its default path is `results/figures/`, which holds the San Juan figures. Individual plot functions are called with Singapore paths | No |

Nothing else differs. In particular there is no new feature, no transformation,
no tuning and no change to target alignment or window definitions.

## 10. Assumptions

| # | Assumption | Checked |
|---|---|---|
| A1 | The file is the MOH Dengue Fever series | Yes: all 574 values match the source |
| A2 | Epi weeks are Sunday–Saturday with the 4-day rule | Yes structurally (11 of 11 years) and against 2 bulletin headings from a secondary archive |
| A3 | Counts are final | Not directly; source last updated June 2024 |
| A4 | Weekly counts are laboratory-confirmed cases | Supported for 2014 only |
| A5 | Week assignment (notification or onset date) is the same throughout | Not established |
| A6 | No change in case definition or testing within 2012–2022 | Not established |
| A7 | Fixed seeds give reproducible accuracy numbers, as reported for San Juan | To be confirmed by running twice |

## 11. Execution

Not to be run until this protocol is approved.

```
.venv/bin/python singapore_analysis/run_singapore_experiment.py            # dry run, trains nothing
.venv/bin/python singapore_analysis/run_singapore_experiment.py --execute  # the experiment
```

`--execute` writes only to `singapore_analysis/results/` and
`singapore_analysis/figures/results_*.png`. Before trusting the output:

1. `git status` shows no change under `src/`, `results/` or `data/processed/`.
2. A second run reproduces every MAE and RMSE exactly.
3. Persistence MAE is identical across the five windows within each fold.

The execute path was subsequently run and its tables and figures were regenerated
and checked against the report. See `followup_experiments/reproduction_audit.md`
for the reproduction checks and `followup_experiments/research_update.md` for
the follow-up results.

## 12. Open items

| Item | Status |
|---|---|
| Second Singapore CSV (2014–2018 overlap) | **Completed.** All 261 weeks aligned; 52 weeks differ from the separate MOH Dengue/DHF file. The broader MOH weekly bulletin matches all 261 weeks. See `singapore_overlap_results.md`. |
| Decisions D1, D2, D3 | Awaiting your review |
| Dating basis of the weekly counts | Unresolved; would need MOH bulletin methodology notes |
