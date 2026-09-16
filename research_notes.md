# How Much Surveillance History Is Enough?
### A Temporal Context Sensitivity Study for Short-Term Infectious Disease Forecasting

**Run date:** 2026-09-16 · **Seed:** 42 · **Status:** working research prototype

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
Training time grew roughly linearly with window size — XGBoost 0.014 s (1 week)
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

## 5. INTERPRETATION

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

## 6. LIMITATIONS

1. **One disease, one dataset family.** San Juan dengue 1990–2008 (primary) plus
   an Iquitos secondary check. Nothing here generalises to other diseases or eras.
2. **A single train/test cut.** With one 139-week test period, the ranking of
   windows is unstable — we demonstrated this directly. Rolling-origin
   (walk-forward) cross-validation would give a far more reliable answer and is the
   single most valuable next step.
3. **Fixed hyperparameters by design.** Longer windows might do better with
   regularisation tuned per window. Our result says "more lags did not help *at
   fixed capacity*", not "more lags cannot help".
4. **Only one-step-ahead (t+1) forecasting** was tested. Longer history may matter
   much more at longer horizons, which is exactly where it would be expected to.
5. **No climate covariates**, deliberately. A model using rainfall/temperature
   would be a different study.
6. **Raw counts, untransformed.** A log or square-root transform is standard for
   epidemic counts and would change the error profile.

---

## 7. CONCLUSION

**For this dataset and this experimental configuration** — San Juan weekly dengue,
1990–2008, one-step-ahead forecasting, XGBoost and LightGBM at fixed settings —
we do **not** find evidence that longer surveillance history improves next-week
forecasting. Windows of 1–4 weeks performed as well as or better than 8–12 weeks
on the test period, while costing about 3× less to train.

However, **we cannot identify a single minimum sufficient window**, because the
pre-registered rule gives different answers on two disjoint future periods
(XGBoost: 2 vs 1; LightGBM: 1 vs 4). The honest conclusion is that **the short
windows are not worse, and are cheaper** — which is a useful practical finding —
but the data as split here does not resolve the exact minimum.

We explicitly do **not** conclude that any particular window is optimal for
infectious disease forecasting in general.
