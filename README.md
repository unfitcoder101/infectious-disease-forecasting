# How Much Surveillance History Is Enough?                     

A temporal context sensitivity study for short-term infectious disease forecasting.

**Question:** how many weeks of past dengue case counts are actually needed to
forecast next week's count? Compares history windows of **1, 2, 4, 8, 12 weeks**
using XGBoost and LightGBM on real weekly dengue surveillance data.

## Headline result                    

For San Juan dengue (1990–2008), **longer history did not improve next-week
forecasts**. Short windows (1–4 weeks) matched or beat long windows (8–12), both
on a single chronological test split and on **10-fold rolling-origin
cross-validation**, while training ~3× faster. Only 1 of 10 configurations beat
a persistence baseline.             

## Final Research Finding

A **short-history regime (1–4 weeks) is well supported**, but the *exact*
minimum sufficient window within {1, 2, 4} is **not resolved**. A second,
targeted 21-fold/13-week CV experiment — built specifically to separate 1, 2,
and 4 weeks using a paired fold-level comparison — still found no consistent
winner: every paired comparison remained noisy, and the two models didn't even
agree on which window had the lowest observed error (LightGBM: window 4;
XGBoost: window 2). **No window is claimed to be optimal.** The honest,
final conclusion is that this study identifies a short-history regime rather
than an exact minimum sufficient window.
                 
See [`research_notes.md`](research_notes.md) §§5–11 for the full CV methodology,
the targeted follow-up, external validations, and limitations. The project-level
conclusion is now informed by four separately analysed dengue settings; errors
are never pooled across datasets.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python run_experiment.py              # San Juan (default)
python run_experiment.py --city iq    # Iquitos, Peru
```

Everything is seeded (seed 42). Reruns reproduce **all accuracy metrics bitwise
identically** (verified). The `training_time` / `inference_time` columns vary by
~1 ms between runs, as any wall-clock measurement does — they are medians of 5
repeats, and should be read as orders of magnitude, not exact constants.

## Output

**Single-split experiment:**
- `results/results_table.csv` — window × model × {MAE, RMSE, R², train/inference time, sample counts}
- `results/sensitivity_analysis.csv` — change at each 1→2→4→8→12 step
- `results/minimum_sufficient_window.csv` — the pre-registered 5%-tolerance rule
- `results/robustness_check.csv` — does the rule agree across two periods? (it does not)
- `results/baseline_comparison.csv` — improvement over persistence

**Rolling-origin cross-validation (10 folds, 26-week blocks):**
- `results/cv_results.csv` — one row per (fold × window × model)
- `results/cv_summary.csv` — mean ± std across folds, per window × model
- `results/cv_fold_win_counts.csv` — how often each window had the lowest MAE
- `results/cv_minimum_sufficient_window.csv` — the same 5%-tolerance rule applied to CV means

**Targeted follow-up (21 folds, 13-week blocks, windows {1, 2, 4} only):**
- `results/fine_cv_results.csv`, `fine_cv_summary.csv`, `fine_cv_fold_win_counts.csv`,
  `fine_cv_minimum_sufficient_window.csv` — same structure as above, finer fold grid
- `results/fine_cv_paired_differences.csv`, `fine_cv_paired_diff_summary.csv` —
  paired per-fold MAE differences for 1-vs-2, 1-vs-4, 2-vs-4

**Figures** (`results/figures/*.png`): MAE, RMSE, R², training time, inference
time vs window (single split); a two-period robustness panel; CV mean MAE with
±1 std error bars; every fold's MAE plotted individually; paired fold-level MAE
differences for the targeted follow-up; an actual-vs-predicted forecast overlay.

## Layout

```
data/raw/          real surveillance CSVs + SOURCE.md (provenance & checksums)
data/processed/    cleaned weekly series
src/data_loader.py     load + merge the raw files
src/preprocessing.py   audit and clean (audits rather than silently imputing)
src/features.py        lag features + window alignment
src/models.py          XGBoost / LightGBM, identical fixed settings
src/evaluation.py      chronological split, rolling-origin CV splits, MAE/RMSE/R², persistence baseline
src/experiment.py      window × model grid, sensitivity, robustness, CV runner + summary
src/visualization.py   figures, including CV error-bar and fold-detail plots
run_experiment.py      runs the whole pipeline
```

## Method notes
                              
- **No random splitting.** Chronological only — a random split would train the
  model on weeks after those it is tested on, and consecutive weeks are so similar
  that near-duplicate rows would land on both sides of the split.                        
- **All windows aligned to identical weeks** (924), so history length is the only
  variable that differs between conditions.                        
- **Model settings fixed across windows** — the study is about context length, not
  about tuning.                             
- **Decision rule pre-registered in code** before results were inspected.
- **Rolling-origin CV** (expanding training window, non-overlapping test blocks)
  answers "does the window ranking hold up across time", which a single split
  cannot. Fold size (26 weeks, 10 folds) was derived from the actual aligned row
  count, not guessed — see `research_notes.md` §5.1.                               

## Data                       
                                          
NOAA/CDC Dengue Forecasting Project (2015) via the DrivenData "DengAI" dataset.
Real data, not synthetic. The original NOAA host is offline; files were taken from
two independent mirrors and verified byte-identical. See `data/raw/SOURCE.md`.
                                

## Singapore external validation

**External validation, not replication.** The method was re-run on Singapore
Ministry of Health weekly dengue fever notifications (DHF excluded), 2012–2022. The
case definition differs from DengAI, so no Singapore error value is comparable with
the San Juan or Iquitos results, and the series are never pooled. Everything lives
in `singapore_analysis/`, separate from `results/`, and appears on the dashboard's
"Singapore External Validation" tab.

- **The 5% rule selected 1 week**, but no window is claimed optimal. "Lowest
  observed" error is not "meaningfully better": short and long history were **not
  statistically separable (p = 0.68)**.
- **Persistence beat every model.** Holm-adjusted significance against persistence
  is indicative only, because folds are not independent.
- The evaluation period (2019-W41 to 2022-W52) includes the 2020 and 2022
  epidemics, and the results are dominated by three epidemic folds.
- The overlap check against the second Singapore CSV is pending.

Details: `research_notes.md` §10 and `singapore_analysis/singapore_results.md`.
Source and licence: `data/raw/SOURCE.md`.


## Brazil external validation

A separate post-hoc run applies the same 1/2/4/8/12-week history experiment to
InfoDengue reported dengue cases in Rio de Janeiro (2010–2025). It is isolated
under `brazil_analysis/`; existing data and results are preserved. Four weeks
had the lowest rolling-origin mean MAE for both models in both fold designs,
but the 5% rule selected 4 weeks for XGBoost and 1 week for LightGBM. Fold
variation was large and persistence was competitive. This supports short history
as a plausible pattern in Rio, but establishes no unique optimum or universal
result. Brazil's raw errors are not compared with the other settings.

- Reproduce: `python brazil_analysis/run_brazil_experiment.py --execute`
- Data source, query, date, and checksum: [`brazil_analysis/SOURCE.md`](brazil_analysis/SOURCE.md)
- Protocol and detailed interpretation: [`brazil_analysis/brazil_experiment_protocol.md`](brazil_analysis/brazil_experiment_protocol.md), [`brazil_analysis/brazil_results.md`](brazil_analysis/brazil_results.md)
- Frozen API snapshot: `brazil_analysis/data/raw/infodengue_rio_2010_2025_snapshot_20261118.csv`
- Results and figures: `brazil_analysis/results/`

The final four-setting synthesis is in `research_notes.md` §11.4. This is still
a dengue study; Brazil does not constitute validation on COVID-19, India, or a
different disease.
