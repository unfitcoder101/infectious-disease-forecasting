# Singapore external validation — results

**Run:** 2026-11-17 · **Protocol:** `singapore_experiment_protocol.md`, approved
without changes (decisions D1, D2, D3) before the run · **Seed:** 42 ·
**Outputs:** `singapore_analysis/results/sg_*.csv`, `singapore_analysis/figures/results_*.png`

Singapore is an external validation on a different operational case definition
(MOH Dengue Fever notifications, DHF excluded). It is not a replication of San
Juan or Iquitos, and no error value here is comparable with theirs.

## 1. Outcome under the pre-specified reading

| Question | Answer |
|---|---|
| Minimum sufficient window (5% rule) | **1 week, for both models, in both CV designs and on both periods of the single split** |
| Do windows 8 and 12 fall inside the 5% band anywhere? | No |
| Primary contrast, short {1, 2, 4} against long {8, 12} | Favours short in mean and median for both models. **Not statistically significant** (Holm-adjusted p = 0.68 for both, 13 folds) |
| Reading fixed in the protocol (§7.6) | **Consistent with the original finding** on an independent system |
| Do the models beat persistence? | **No.** Persistence has lower mean and median MAE than every one of the 10 model/window configurations in both CV designs |

Two qualifications matter as much as the headline:

1. "Consistent with" rests on the 5% rule and the direction of the contrast. The
   paired test does not separate short from long windows.
2. The comparison is between variants of models that are all clearly worse than
   a one-line rule on this evaluation period.

## 2. Run checks

| Check | Result |
|---|---|
| Second full run reproduces every MAE and RMSE exactly | Yes (single split, 26-week CV, 13-week CV) |
| Persistence MAE identical across the five windows within each fold | Yes, both designs |
| Any tracked file changed (`src/`, `results/`, `data/processed/`, notes) | None |
| Folds obtained | 6 (26-week blocks) and 13 (13-week blocks), as planned |

## 3. Rolling-origin CV, 13-week blocks (13 folds)

This is the design that carries the statistical comparisons.

| Window | XGBoost MAE mean ± std | XGBoost median | LightGBM MAE mean ± std | LightGBM median |
|---|---|---|---|---|
| 1 | **73.35 ± 77.77** | 43.00 | **83.88 ± 102.60** | 38.64 |
| 2 | 86.45 ± 92.98 | 40.37 | 87.45 ± 105.40 | 40.87 |
| 4 | 82.20 ± 90.00 | 42.77 | 84.27 ± 100.75 | 39.52 |
| 8 | 86.87 ± 101.79 | 44.29 | 95.77 ± 124.60 | 47.10 |
| 12 | 87.09 ± 99.13 | 45.34 | 93.83 ± 124.83 | 43.29 |
| Persistence | 54.70 ± 51.23 | 33.00 | | |

**5% rule on mean MAE**

| Model | Best window | Near-optimal windows | Minimum sufficient window | Percentage above best: windows 2 / 4 / 8 / 12 |
|---|---|---|---|---|
| XGBoost | 1 | 1 only | **1** | 17.9% / 12.1% / 18.4% / 18.7% |
| LightGBM | 1 | 1, 2, 4 | **1** | 4.3% / 0.5% / 14.2% / 11.9% |

**Fold win counts (lowest MAE in the fold)**

| Window | XGBoost | LightGBM |
|---|---|---|
| 1 | 6 / 13 | 6 / 13 |
| 2 | 2 / 13 | 1 / 13 |
| 4 | 3 / 13 | 3 / 13 |
| 8 | 0 / 13 | 0 / 13 |
| 12 | 2 / 13 | 3 / 13 |

**Primary family: short {1, 2, 4} minus long {8, 12}, per-fold MAE**

| Model | Mean difference | Median difference | 95% t-interval | 95% bootstrap interval | Folds short better | Wilcoxon p | Holm-adjusted p |
|---|---|---|---|---|---|---|---|
| XGBoost | −6.32 | −1.12 | −24.41 to 11.78 | −24.12 to 6.98 | 9 of 13 | 0.588 | 0.679 |
| LightGBM | −9.60 | −1.37 | −25.06 to 5.86 | −24.29 to 1.54 | 7 of 13 | 0.340 | 0.679 |

Negative means the short windows had lower error. The same contrast on RMSE has
the same direction (−8.33 and −9.75) and is also not significant (Holm p 0.89 and 0.49).

**Secondary family: window pairs.** None of the 20 comparisons is significant
after Holm adjustment (smallest adjusted p = 0.34). The two smallest unadjusted
values are both window 1 against window 2: window 1 had lower MAE in 10 of 13
folds for each model (unadjusted p = 0.033 for XGBoost, 0.017 for LightGBM).
Full table: `results/sg_cv13_comparisons_mae.csv`.

**Secondary family: against persistence, per-fold MAE**

| Model | Window | Mean excess over persistence | Median excess | Folds where the model was better | Holm-adjusted p |
|---|---|---|---|---|---|
| XGBoost | 1 | +18.65 (+34%) | +5.48 | 2 of 13 | 0.051 |
| XGBoost | 2 | +31.75 (+58%) | +8.34 | 1 of 13 | **0.012** |
| XGBoost | 4 | +27.50 (+50%) | +4.84 | 1 of 13 | **0.012** |
| XGBoost | 8 | +32.17 (+59%) | +5.33 | 2 of 13 | **0.040** |
| XGBoost | 12 | +32.40 (+59%) | +7.41 | 2 of 13 | **0.012** |
| LightGBM | 1 | +29.18 (+53%) | +5.67 | 3 of 13 | **0.028** |
| LightGBM | 2 | +32.75 (+60%) | +6.27 | 1 of 13 | **0.012** |
| LightGBM | 4 | +29.58 (+54%) | +6.49 | 2 of 13 | **0.040** |
| LightGBM | 8 | +41.07 (+75%) | +4.68 | 2 of 13 | 0.051 |
| LightGBM | 12 | +39.13 (+72%) | +4.63 | 5 of 13 | 0.110 |

By the criterion fixed in the protocol, **7 of the 10 configurations are
statistically significantly worse than persistence**. This family was declared
exploratory in advance, and the fold-dependence caveat in section 7 applies.

## 4. Rolling-origin CV, 26-week blocks (6 folds)

Descriptive only: six folds cannot reach significance after adjustment.

| Window | XGBoost MAE mean ± std | XGBoost median | LightGBM MAE mean ± std | LightGBM median |
|---|---|---|---|---|
| 1 | **102.24 ± 128.60** | 36.44 | **99.49 ± 135.10** | 34.07 |
| 2 | 114.52 ± 138.73 | 39.06 | 104.51 ± 138.26 | 37.57 |
| 4 | 123.50 ± 173.38 | 35.85 | 106.90 ± 145.43 | 36.62 |
| 8 | 129.97 ± 185.26 | 37.43 | 120.39 ± 160.18 | 40.48 |
| 12 | 129.64 ± 182.26 | 42.32 | 113.59 ± 149.52 | 43.37 |
| Persistence | 56.80 ± 50.27 | 29.46 | | |

- Minimum sufficient window is 1 for both models, and window 1 is the only
  near-optimal window for both. LightGBM's window 2 misses the band by a hair
  (5.04% above best).
- The short-versus-long contrast favours short (−16.4 XGBoost, −13.4 LightGBM;
  4 of 6 folds each).
- Persistence was better than the model in 6 of 6 folds for 8 of the 10
  configurations and in 5 of 6 for the other two.

## 5. Single chronological split

Trained once on data ending 2019-W40; validation 2019-W41 to 2021-W19; test
2021-W20 to 2022-W52.

| Window | XGBoost test MAE | LightGBM test MAE | XGBoost validation MAE | LightGBM validation MAE |
|---|---|---|---|---|
| 1 | **98.30** | **106.14** | **134.58** | **138.07** |
| 2 | 101.20 | 109.35 | 138.12 | 142.79 |
| 4 | 140.13 | 116.02 | 165.41 | 149.45 |
| 8 | 144.58 | 122.61 | 172.78 | 162.39 |
| 12 | 147.25 | 113.29 | 177.50 | 154.28 |
| Persistence | 49.99 | | 59.46 | |

Minimum sufficient window is 1 for both models on both periods, so the two
periods agree (they did not for San Juan). No configuration beats persistence;
every one has roughly two to three times its error. As anticipated in the
protocol, this design is the least informative: the models never see a value
above 888 and both evaluation periods contain peaks near twice that.

## 6. Where the error comes from

Per-fold MAE, 13-week design (`results/sg_cv13_results.csv`):

| Fold | Target weeks | Persistence | XGBoost, windows 1 / 12 | LightGBM, windows 1 / 12 |
|---|---|---|---|---|
| 0 | 2019-W41 to 2020-W01 | 29.6 | 31.2 / 44.7 | 28.0 / 48.7 |
| 1 | 2020-W02 to 2020-W14 | 33.0 | 43.0 / 45.7 | 39.7 / 52.9 |
| 2 | 2020-W15 to 2020-W27 | 90.7 | **221.1 / 263.9** | **230.8 / 252.3** |
| 3 | 2020-W28 to 2020-W40 | 165.0 | **173.1 / 288.8** | **344.8 / 427.1** |
| 4 | 2020-W41 to 2020-W53 | 46.0 | 56.8 / 49.9 | 56.4 / 43.3 |
| 5 | 2021-W01 to 2021-W13 | 9.2 | 14.7 / 17.5 | 14.9 / 18.8 |
| 6 | 2021-W14 to 2021-W26 | 17.9 | 18.0 / 16.7 | 17.5 / 14.9 |
| 7 | 2021-W27 to 2021-W39 | 13.0 | 14.1 / 15.5 | 15.2 / 17.6 |
| 8 | 2021-W40 to 2021-W52 | 10.5 | 10.2 / 12.1 | 10.3 / 10.4 |
| 9 | 2022-W01 to 2022-W13 | 37.9 | 43.6 / 45.3 | 38.6 / 37.9 |
| 10 | 2022-W14 to 2022-W26 | 143.5 | 219.7 / 211.2 | 167.3 / 202.5 |
| 11 | 2022-W27 to 2022-W39 | 85.2 | 73.8 / 91.8 | 92.6 / 63.3 |
| 12 | 2022-W40 to 2022-W52 | 29.5 | 34.4 / 29.0 | 34.3 / 30.0 |

- Fold MAE spans 9 to 427 cases. Folds 2, 3 and 10 (the 2020 and 2022 peaks)
  supply most of every mean.
- Folds 2 and 3 are the ones flagged in the protocol as risk R1: their targets
  exceed anything in the training data, which tree models cannot predict. The
  models' error there is 1.0 to 2.9 times persistence's.
- The gap to persistence is not only an extrapolation effect. Persistence is
  also ahead in most quiet folds, which is why it wins 8 to 12 of 13 folds per
  configuration and the median excess is 5 to 8 cases.
- In the quiet folds (5 to 9) all windows are within a few cases of each other.
  The separation between windows comes from the epidemic folds and from folds 0 and 1.

## 7. Interpretation

*Separated from the observations above.*

- **The short-history finding transfers in direction.** On an independent
  surveillance system, in a different region and decade, adding history beyond a
  few weeks did not improve one-week-ahead forecasts at fixed model settings.
  Windows 8 and 12 were outside the 5% band in every design.
- **It transfers as a pattern, not as a proven difference.** The primary test
  does not distinguish short from long. The mean differences are driven by a
  few epidemic folds; the median differences are about one case.
- **Singapore points to 1 week more sharply than San Juan did.** San Juan's CV
  left windows 1, 2 and 4 inside the band for both models. Here window 1 is the
  only near-optimal window in three of the four model/design combinations, and
  for XGBoost in the 13-fold design windows 2, 4, 8 and 12 are all 12–19% above
  it. For XGBoost the pattern is "1 against the rest" more than "short against
  long". The window 1 versus window 2 comparison is the closest thing to a
  resolved pair but does not survive adjustment.
- **The EDA hypothesis was not borne out.** The autocorrelation structure
  suggested a 2 to 4 week window might average out reporting noise and beat
  persistence. It did not. More lags made the tree models worse, not better.
- **The strongest result is about persistence.** The original found that almost
  nothing beat the baseline. Singapore shows the same thing more strongly, with
  formal support in 7 of 10 configurations. As configured, these models are not
  competitive one-week-ahead forecasters for this series.
- A plausible reason more lags hurt is the one given in the original: with a few
  hundred training rows and fixed capacity, extra lag features add variance.
  Singapore has fewer training rows than San Juan (393 to 549 against 646 to
  906), which would make this worse. This was not tested.

## 8. Limitations specific to this run

1. **The evaluation period is unrepresentative.** It covers 2019-W41 to 2022-W52:
   two record epidemics and the COVID-19 years. Nothing here describes typical
   Singapore performance.
2. **Extrapolation handicaps the tree models against persistence** in the folds
   covering the 2020 rise. It was left uncorrected by design.
3. **Folds are not independent.** They are consecutive blocks of one series with
   shared training data. Intervals are probably too narrow and p-values too small.
   The persistence result is large and consistent across folds, so it is unlikely
   to hinge on this; the window comparisons are null either way.
4. **Thirteen folds, one 3.25-year period.** The t-intervals and bootstrap
   intervals disagree noticeably for several comparisons (for example the
   bootstrap interval for XGBoost window 1 against persistence excludes zero and
   the t-interval does not), which reflects heavy-tailed fold differences. Per the
   protocol, neither interval is relied on where they disagree; the Wilcoxon
   result and win counts are.
5. **The 5% rule is applied to mean MAE**, which the epidemic folds dominate.
   Medians rank the windows less cleanly: for XGBoost in the 13-fold design the
   lowest *median* MAE is window 2, not window 1.
6. **The medians, intervals, tests and the primary contrast are additions** to
   the original method and were not applied to San Juan or Iquitos.
7. All limitations of the original study still apply: fixed hyperparameters,
   one-step horizon, raw counts, no covariates.
8. **Cross-source counts differ in the separate MOH 2014–2018 Dengue/DHF file.** On 261 overlapping weeks, its Dengue count differs from the selected series in 52 weeks (range -8 to +1); the broader MOH Weekly Infectious Disease Bulletin source matches all 261 weeks exactly. The source-vintage difference is unresolved and does not change the forecast inputs used here. See `singapore_overlap_results.md`.

## 9. Conclusion

For Singapore weekly dengue fever notifications, 2012–2022, with XGBoost and
LightGBM at the original fixed settings, one week of history was the minimum
sufficient window under the pre-registered 5% rule in every design, and 8- and
12-week windows were never near-optimal. This is consistent with the original
study's short-history finding on an independent surveillance system. The
difference between short and long windows was not statistically significant, so
the finding is supported in direction and not established as an effect.

Neither model beat persistence in any configuration, and most were significantly
worse over an evaluation period dominated by two record epidemics.

No window is described as optimal, and no Singapore figure should be compared
numerically with the San Juan or Iquitos results.
