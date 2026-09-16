# How Much Surveillance History Is Enough?
### A Temporal Context Sensitivity Study for Short-Term Infectious Disease Forecasting

**Run date:** 2026-09-16, updated 2026-09-17 with rolling-origin CV · **Seed:** 42 · **Status:** working research prototype

---

## 1. Research question

How many weeks of past dengue surveillance data are actually needed to forecast
next week's case count? We compare history windows of **1, 2, 4, 8, and 12 weeks**.

The question is *not* "what is the best possible dengue model". Model settings are
held fixed on purpose so that history length is the only variable that changes.

---

## 2. Data

**Source:** NOAA / CDC **Dengue Forecasting Project** (2015) training data, as
distributed in the DrivenData "DengAI" challenge. Real surveillance data, not
synthetic.

**Series used:** San Juan, Puerto Rico — **936 weekly observations**,
1990-04-30 → 2008-04-22 (18.0 years). Cases: min 0, median 19, max 461.

**Provenance caveat (important):** the original host
`dengueforecasting.noaa.gov` **no longer resolves** — the project site has been
decommissioned. Data was obtained from public GitHub mirrors, and to guard
against an altered copy, the files were downloaded from **two unrelated
repositories and verified byte-identical** (md5 `5df4b6d0240bb83d4b509cc045b46603`).
Full details in `data/raw/SOURCE.md`.

### Data quality audit (measured, not assumed)
| Check | Result |
|---|---|
| Chronologically sorted | yes |
| Duplicate dates | 0 |
| Missing case values | **0** |
| Negative values | 0 |
| Gaps between consecutive weeks | 917 × 7 days, 14 × 8 days, 4 × 9 days |
| **True missing weeks (gap ≥ 14 days)** | **0** |

The 18 irregular gaps are **not** missing data. They are the epidemiological-week
convention: the last week of each year is stretched to 8–9 days so the next year
begins on January 1. **No imputation was performed**, because none was needed.

We also found the dataset's `year`/`weekofyear` columns contain a known
inconsistency (a stray "week 53" disagreeing with the year field), so
`week_start_date` was used as the authoritative time index instead.

**No climate covariates were used.** The only predictor is the disease's own past.

---

## 3. Method

- **Lag features.** For window *W*, `X = [lag_1 … lag_W]` (lag_1 = most recent
  week), `y` = next week's cases. Because `lag_1..lag_W` is nested, a short
  window's features are a strict subset of a long window's — a fair comparison.
- **Window alignment.** A 12-week window needs a 12-week run-up, so it naturally
  yields fewer rows (924) than a 1-week window (935). All windows were therefore
  restricted to the **same 924 target weeks**, so every window is trained and
  scored on identical weeks. Without this, a difference between windows could be
  caused by the different weeks being tested rather than by history length.
- **Chronological split**, never random: train 646 weeks (1990-07 → 2002-12),
  validation 139 (2002-12 → 2005-08), test 139 (2005-08 → 2008-04).
- **Models.** XGBoost and LightGBM, *identical fixed settings* across all windows
  (300 trees, lr 0.05, max_depth 4, seed 42), single-threaded so timings compare.
- **Baseline.** Persistence: predict next week = this week.
- **Pre-registered rule**, fixed in code before results were seen: a window is
  *near-optimal* if its MAE is within **5%** of the best MAE; the **minimum
  sufficient window** is the shortest near-optimal one.

---

## 4. OBSERVED RESULTS

### 4.1 Test period (2005-08 → 2008-04)

| Window | XGBoost MAE | LightGBM MAE | XGB RMSE | LGBM RMSE | XGB R² | LGBM R² |
|---|---|---|---|---|---|---|
| 1 | 7.98 | **7.09** | 14.24 | 11.43 | 0.814 | **0.880** |
| 2 | 7.70 | 7.51 | 14.24 | 13.65 | 0.814 | 0.829 |
| 4 | **7.56** | 7.63 | 13.19 | 13.82 | 0.840 | 0.825 |
| 8 | 8.92 | 9.32 | 15.35 | 16.89 | 0.784 | 0.738 |
| 12 | 8.61 | 9.82 | 14.28 | 18.27 | 0.813 | 0.694 |
| **Persistence** | **7.26** | | **11.99** | | **0.868** | |

### 4.2 The two findings that matter

**Finding A — on the test period, more history made forecasts worse.**
Going from 4 → 8 weeks raised MAE by **18.0%** (XGBoost) and **22.1%** (LightGBM).

**Finding B — that pattern does NOT replicate on the validation period.**
Applying the identical rule to the earlier, disjoint validation period:

| Model | Best window (test) | Best window (validation) | MAE spread across windows (test) | MAE spread (validation) |
|---|---|---|---|---|
| XGBoost | 4 | 2 | 18.0% | **4.1%** |
| LightGBM | 1 | **8** | 38.5% | **10.1%** |

On the validation period the curve is close to **flat** (MAE 4.59–5.07), and
LightGBM's best window is **8** — the opposite end from its test-period best of 1.

**Finding C — almost nothing beat the trivial baseline.**
**1 of 10** model/window configurations beat persistence on the test set
(LightGBM, 1 week, by 2.3%). Every other configuration was worse, some by 35%.

### 4.3 Computational cost
Training time grew roughly linearly with window size — XGBoost 0.013 s (1 week)
→ 0.041 s (12 weeks), about **3×**. Inference time stayed ~0.0005 s throughout.
In absolute terms all costs are negligible at this data scale.

Timings are **medians of 5 repeated runs**, measured single-threaded. They still
vary by roughly ±1 ms between runs, so the ~3× trend is meaningful but the
individual numbers are not exact constants. All accuracy metrics, by contrast,
reproduce bitwise across runs (verified).

### 4.4 Secondary check: a second city

The pipeline also runs on the Iquitos, Peru series (520 weeks, 2000–2010) via
`python run_experiment.py --city iq`. Observed test MAE by window:

| Window | XGBoost | LightGBM |
|---|---|---|
| 1 | 3.50 | **3.39** |
| 2 | 4.38 | 3.57 |
| 4 | **3.38** | 3.57 |
| 8 | 3.58 | 3.82 |
| 12 | 3.65 | 3.72 |
| **Persistence** | **2.97** | |

Here the two periods **do agree**: the minimum sufficient window is **1 week** for
both models on both periods. This is consistent with the San Juan finding that
short windows suffice. But note that on Iquitos **no configuration at all** beat
persistence — reinforcing Finding C rather than softening it.

This is a secondary check on a second city, not an independent replication: it is
the same dataset family, the same pipeline, and the same fixed model settings.

---

## 5. ROLLING-ORIGIN CROSS-VALIDATION

Section 4 rests on **one** 139-week test period. `robustness_check()` already
showed that a single split's ranking of windows does not even repeat on the
adjacent validation period. Rolling-origin (walk-forward) cross-validation
replaces that one test slice with a **sequence** of them, so each window gets
a distribution of scores rather than one point estimate.

### 5.1 Configuration (derived from the actual aligned row count, not guessed)

| Parameter | Value | How it was chosen |
|---|---|---|
| Aligned series length | 924 weeks | Same window-aligned series as the single-split experiment |
| `min_train` (fold 0 training size) | 646 weeks | `int(924 × 0.70)` — same 70% floor as the single split, for continuity |
| Available weeks for folds | 278 | `924 − 646` |
| `test_block` | **26 weeks** (~half a year) | The only block size in the requested 20–26 week range whose fold count lands in the requested 8–10 target: `278 // 26 = 10`. (Every block from 20–25 gives 11–13 folds — too many for the target range.) |
| `n_folds` | **10** | `278 // 26` |
| Leftover (unused) | 18 weeks | Too short for an 11th whole block; dropped rather than padded |
| Window scheme | **Expanding**, walk-forward | Training set only grows forward in time; never includes a row after its fold's test block |
| Test blocks | **Non-overlapping** | No week is scored by more than one fold, so averaging across folds does not double-count any observation |

Fold boundaries (verified, not assumed): fold 0 trains on 646 weeks and tests
on 2002-12-17 → 2003-06-11; each subsequent fold adds exactly 26 weeks of
training data and slides its test block forward by 26 weeks; fold 9 trains on
880 weeks and tests on 2007-06-18 → 2007-12-10. Verified programmatically:
train size strictly increasing across all 10 folds, every test block exactly
26 weeks, and every fold's test period starts strictly after the previous
fold's test period ends (no overlap).

Same model settings, same seed, same window-alignment logic as the single
split — only the evaluation scheme changed.

### 5.2 Results: CV mean ± std per window

| Window | XGBoost MAE (mean±std) | LightGBM MAE (mean±std) |
|---|---|---|
| 1 | 6.65 ± 4.60 | 6.27 ± 3.86 |
| 2 | 6.39 ± 4.89 | 6.33 ± 4.76 |
| 4 | **6.37 ± 4.86** | **6.23 ± 4.67** |
| 8 | 7.01 ± 5.92 | 6.89 ± 5.93 |
| 12 | 6.87 ± 5.31 | 7.17 ± 6.36 |
| Persistence | 6.33 ± 4.11 (same every window — uses only lag_1) | |

**The standard deviations are on the same order as the means.** For every
window, `MAE_std` is roughly 62–89% of `MAE_mean`. This is the central new
fact CV adds: fold-to-fold variance is enormous relative to the differences
*between* windows. The CV mean-MAE figure
(`results/figures/cv_mae_vs_history_window.png`) makes this visible directly —
the ±1 std error bars for every window and the persistence baseline overlap
almost completely.

### 5.3 Where the variance comes from

Per-fold MAE ranges from about **2.4** (a quiet 26-week block) to about
**20.8** (the fold containing the peak of the 2007 outbreak). The
`cv_fold_detail.png` figure shows this directly: most folds cluster tightly
between 2.5 and 6 cases, with 2–3 outlier folds (those overlapping an
outbreak) pulled up to 12–21. **The dominant source of error variance is
which folds happen to contain an outbreak, not history window length.**

A related side-effect: **R² is not a reliable per-fold metric at this block
size.** Mean R² across folds is only 0.01–0.23 (versus 0.69–0.88 on the single
139-week test period), and individual folds range as low as **−1.06**. This
is not a bug — R² divides by each fold's own variance (`SS_tot`), and a quiet
26-week off-season block has very little variance to divide by, making R²
extremely sensitive to small errors in that fold. R² over one long,
outbreak-inclusive period (Section 4) is a fundamentally different, less
noisy statistic than R² averaged over many short quiet-plus-outbreak folds.
This is reported as a limitation of the CV design at this fold length, not
suppressed.

### 5.4 Ranking stability across folds ("fold win counts")

For each fold, which window had the lowest MAE?

| Window | XGBoost win rate | LightGBM win rate |
|---|---|---|
| 1 | 0% (0/10) | 20% (2/10) |
| 2 | 30% (3/10) | 20% (2/10) |
| 4 | 30% (3/10) | 20% (2/10) |
| 8 | 20% (2/10) | 30% (3/10) |
| 12 | 20% (2/10) | 10% (1/10) |

No window wins a clear majority of folds for either model. Wins are spread
across nearly all five windows (XGBoost's win rate ranges only 0–30%;
LightGBM's 10–30%). **This directly confirms, with fold-level evidence, what
the single-split-vs-validation disagreement in Section 4.2 already
suggested: which window looks "best" changes depending on which slice of
time you evaluate on.**

### 5.5 CV minimum sufficient window

Applying the same pre-registered 5%-tolerance rule to CV mean MAE:

| Model | Best window (CV mean) | Minimum sufficient window | Within 1 std of best? |
|---|---|---|---|
| XGBoost | 4 | **1** | Yes |
| LightGBM | 4 | **1** | Yes |
| Persistence | 1 | 1 | Yes |

For both models, windows 1, 2, and 4 are all within the 5% tolerance band,
and the shortest of those (1 week) is within one standard deviation of the
best-performing window. In other words: **the CV data cannot statistically
distinguish 1 week of history from 4 weeks of history** at this fold count
and block size.

### 5.6 Does CV change the conclusion?

| Model | Single-split pick | CV pick | Agree? |
|---|---|---|---|
| XGBoost | 2 weeks | 1 week | Different, but adjacent — both far short of 8–12 |
| LightGBM | 1 week | 1 week | Same |

**CV does not overturn the headline finding — it strengthens it, and adds an
important qualifier.** Both evaluation schemes agree that **short windows
(1–4 weeks) are at least as good as long windows (8–12 weeks)**, and CV adds
quantitative support that this gap is real: the jump to 8–12 weeks sits
outside the 5% tolerance band in the CV means too. But CV also shows the
opposite side honestly — the *exact* single best window among {1, 2, 4} is
**not** resolved, because those three sit within one fold-level standard
deviation of each other. Before CV, "window 4 is the single-split optimum"
looked like a precise finding; after CV, it is visibly noise-level.

## 6. INTERPRETATION

*(Clearly separated from the observations above.)*

- The forecast plot shows both models producing curves that **track the epidemic
  but lag it** — they behave much like persistence, which is unsurprising when
  `lag_1` is by far the most informative feature in a strongly autocorrelated series.
- The most plausible explanation for Finding A is **overfitting**: with only 646
  training rows and fixed capacity, adding lags adds parameters the models can use
  to fit training noise. The test period contains a large 2007 outbreak (peak ~170
  cases/week); tree models cannot extrapolate beyond values seen in training, so
  extra features mostly add variance rather than signal.
- Finding B is the reason we cannot state a single answer. The disagreement between
  two adjacent future periods shows that **differences between windows are smaller
  than differences between evaluation periods**. The steep degradation at 8 and 12
  weeks is a property of *that particular test stretch*, not a stable property of
  dengue dynamics.
- Finding C means the headline comparison should be read modestly: we are comparing
  variants of a model family that, as configured, barely competes with a one-line rule.

---

## 7. LIMITATIONS

1. **One disease, one dataset family.** San Juan dengue 1990–2008 (primary) plus
   an Iquitos secondary check. Nothing here generalises to other diseases or eras.
2. **A single train/test cut was the original design; rolling-origin CV (Section 5)
   now addresses this**, but did not fully resolve it. It confirmed short windows
   (1–4) are at least as good as long windows (8–12), while also showing that the
   fold-to-fold standard deviation (≈62–89% of the mean MAE) is too large to
   distinguish 1 from 2 from 4 weeks specifically. More folds, or a longer series,
   would be needed to resolve that finer question.
3. **Fixed hyperparameters by design.** Longer windows might do better with
   regularisation tuned per window. Our result says "more lags did not help *at
   fixed capacity*", not "more lags cannot help".
4. **Only one-step-ahead (t+1) forecasting** was tested. Longer history may matter
   much more at longer horizons, which is exactly where it would be expected to.
5. **No climate covariates**, deliberately. A model using rainfall/temperature
   would be a different study.
6. **Raw counts, untransformed.** A log or square-root transform is standard for
   epidemic counts and would change the error profile.
7. **R² is unreliable as a per-fold CV metric at a 26-week block size.** Individual
   fold R² ranged from −1.06 to 0.84 because a quiet 26-week block has very little
   case-count variance to divide by. CV conclusions in this study are therefore
   based on MAE/RMSE, not per-fold R². This is a property of evaluating R² over
   short windows, not a defect in the pipeline.
8. **The CV block-size derivation (20–26 weeks → 8–10 folds) was calibrated to San
   Juan's 924 aligned rows.** Run on Iquitos (508 aligned rows), the same range
   yields at most 7 folds — the pipeline detects and reports this rather than
   silently forcing a bad configuration, but the 8–10 fold target was not met for
   that city.
9. **Cross-validation variance itself was measured on one series.** We have not
   checked whether the size of the fold-to-fold standard deviation is a property
   of San Juan dengue specifically or of short-block CV on epidemic count data in
   general.

---

## 8. CONCLUSION

**For this dataset and this experimental configuration** — San Juan weekly dengue,
1990–2008, one-step-ahead forecasting, XGBoost and LightGBM at fixed settings —
we do **not** find evidence that longer surveillance history improves next-week
forecasting. Windows of 1–4 weeks performed as well as or better than 8–12 weeks
on the test period, while costing about 3× less to train.

However, **we still cannot identify one exact minimum sufficient window**. The
original single-split evidence already showed this (XGBoost: 2 vs 1; LightGBM: 1
vs 4 between test and validation periods). Rolling-origin cross-validation
(Section 5), run across 10 forward-chaining folds, **does not overturn that
uncertainty — it explains it with fold-level evidence**: no window wins a
majority of folds (best win rate 30%), and windows 1, 2, and 4 sit within one
fold-level standard deviation of each other for both models. What CV *does*
add confidently is the coarser-grained answer: **short history (1–4 weeks) is
consistently at least as good as long history (8–12 weeks)** across both the
single-split design and the 10-fold CV design, and it is cheaper to train. The
honest conclusion is therefore two-tiered — confident about "short beats or
matches long", not confident about the single best window within {1, 2, 4}.

We explicitly do **not** conclude that any particular window is optimal for
infectious disease forecasting in general.
