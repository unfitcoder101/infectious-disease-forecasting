# Singapore dengue source overlap check

## Result

The check is complete for **2014-W01 through 2018-W52** (261 weeks). There are no missing or unmatched weeks between the selected project series and the MOH dataset. However, the MOH dataset specifically titled *Weekly Number of Dengue and Dengue Haemorrhagic Fever Cases* does **not** have the same weekly counts as the project series in every week:

| Comparison | Exact matches | Mismatched weeks | Difference range |
|---|---:|---:|---:|
| MOH `Dengue` counts vs project `dengue_fever_cases` | 209 / 261 | **52 / 261 (19.9%)** | -8 to +1 cases |
| MOH `Dengue` + `DHF` vs project `dengue_fever_cases` | 161 / 261 | 100 / 261 | -8 to +3 cases |
| MOH *Weekly Infectious Disease Bulletin Cases* `Dengue Fever` vs project series | 261 / 261 | **0 / 261** | exactly equal |

For the dedicated Dengue/DHF file, 49 of the 52 mismatches are lower in that file than in the project series and 3 are higher by one case. Mismatches cluster in 2016 (37 weeks); there are 7 in 2015, 6 in 2017 and 2 in 2018. Adding DHF does not reconcile the difference: the combined count differs in 100 weeks, and the difference is not equal to the DHF count week by week. This is therefore not just an accidental DF-versus-DF+DHF comparison.

The broader 2012–2022 MOH bulletin source matches all 261 weeks exactly. The mismatch is specific to comparing the separate 2014–2018 MOH Dengue/DHF table against the project file. Both are official MOH data.gov.sg releases, but the files expose different vintages/series descriptions. The accessible files do not establish why their counts differ. We report the discrepancy as an unresolved source-version difference; we do not overwrite the selected series or silently combine sources.

## Mismatched weeks

The complete week-by-week join is `tables/overlap_check_2014_2018.csv`. The 52 weeks where the dedicated MOH `Dengue` count differs from the project input are in `tables/overlap_mismatched_weeks_2014_2018.csv`. Differences are defined as `MOH dedicated-file Dengue count - project count`; negative values mean the dedicated file is lower.

## Sources and reproducibility

- The dedicated 2014–2018 source is MOH, data.gov.sg dataset ID `d_ac1eecf0886ff0bceefbc51556247015`, accessed 2026-11-19. It reports 530 records, including a 2018-W53 pair with `NA`; that NA week is outside the agreed overlap interval. The full API response is frozen at `data/raw/moh_weekly_dengue_df_dhf_2014_2018_api_snapshot.json`.
- The broader source is MOH, data.gov.sg dataset ID `d_ca168b2cb763640d72c4600a68f9909e`, downloaded 2026-11-16 and frozen at `data/raw/WeeklyInfectiousDiseaseBulletinCases.csv`.
- The selected time series remains `../../data/raw/singapore_dengue_fever_weekly_clean.csv`.
- Checksums: `tables/overlap_check_summary.json`; provenance and citations: `data/raw/SOURCES.md`.

The overlap comparison is a source-consistency check, not an independent forecast validation dataset.
