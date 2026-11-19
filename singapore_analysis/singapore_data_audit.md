# Singapore dengue fever series — data audit (Phase 2)

**Audited:** 2026-11-17 · **File:** `data/raw/singapore_dengue_fever_weekly_clean.csv`
· **md5:** `055198432469886d428416e003a00b2a` · **Script:** `singapore_analysis/singapore_audit.py`

The raw file was not modified. No observation was added, dropped, filled or reordered.

## 1. Verdict

The file is structurally clean and matches the Ministry of Health source exactly.
It is one gapless sequence of 574 consecutive Sunday–Saturday weeks from
1 January 2012 to 31 December 2022. The week-53 pattern is fully explained by
MOH's week-numbering rule. No cleaning or imputation is needed.

The cross-source check against the separate 2014–2018 MOH dengue/DHF table is complete; the exact matches and count differences are recorded in section 7 and `singapore_overlap_results.md`.

## 2. Structural checks

| Check | Result |
|---|---|
| Total observations | **574** (as expected) |
| Columns | `time_index`, `epi_week`, `epi_year`, `epi_week_number`, `dengue_fever_cases` |
| Data types | `epi_week` is text; the other four are int64 |
| Encoding | UTF-8, no BOM, LF line endings |
| Missing values | 0 in every column |
| Fully duplicated rows | 0 |
| Duplicate `epi_week` labels | 0 |
| Duplicate (`epi_year`, `epi_week_number`) | 0 |
| Negative case counts | 0 |
| Zero case counts | 0 (minimum is 24) |
| Non-integer case counts | 0 |
| `time_index` runs exactly 1…574 | Yes |
| `epi_week` label agrees with `epi_year` and `epi_week_number` | Yes, all 574 rows |
| Chronological ordering | Yes, strictly increasing by (year, week) |
| Week numbers contiguous from 1 within every year | Yes, all 11 years |
| Years covered | 2012–2022, none missing |

Full output: `tables/audit_checks.csv`.

## 3. Verification against the primary source

The file was compared with the MOH dataset "Weekly Infectious Disease Bulletin
Cases" on data.gov.sg (resource `d_ca168b2cb763640d72c4600a68f9909e`, coverage
January 2012 to December 2022, last updated 6 June 2024), queried on 2026-11-17
for `disease = "Dengue Fever"`.

| Comparison | Result |
|---|---|
| Weeks in source | 574 |
| Weeks in file | 574 |
| Weeks in one but not the other | 0 |
| Weeks with a different case count | **0** |

So `dengue_fever_cases` is the unaltered MOH "Dengue Fever" series, and the
"clean" file differs from the source only by the added helper columns
(`time_index`, `epi_year`, `epi_week_number`).

The source also carries a separate "Dengue Haemorrhagic Fever" series for the
same 574 weeks. It is **not** part of the file and was not combined with it. Its
size is recorded for context only (`tables/context_df_vs_dhf_annual_from_moh_source.csv`):
424 DHF cases against 164,023 DF cases over 2012–2022, which is 0.26% of the
combined total, never more than 8 in a week.

## 4. Epidemiological week convention

### 4.1 What the source states

The data.gov.sg column description for `epi_week` reads: "Each epidemiological
week begins on a Sunday and ends on Saturday." The source does not state how
week 1 is chosen. ISO-8601 was not assumed; it was tested as one candidate.

### 4.2 What the file's structure shows

Four candidate rules were tested against the number of weeks per year in the file
(`tables/audit_annual_week_counts.csv`, `tables/audit_week_convention_match.csv`):

| Convention | Years matching (of 11) | Years that do not match |
|---|---|---|
| **Sunday–Saturday, week 1 has at least 4 days in January** | **11** | none |
| Sunday–Saturday, week 1 contains 1 January | 7 | 2014, 2016, 2020, 2022 |
| Sunday–Saturday, week 1 is the first full week | 7 | 2012, 2014, 2017, 2020 |
| ISO-8601 (Monday–Sunday) | 9 | 2014, 2015 |

Only the first rule reproduces all 11 years. It is the same rule as the US CDC
MMWR week ("week containing 4 January", Sunday start).

### 4.3 External anchors

Two MOH bulletin headings, as reproduced on a third-party archive (FluTrackers;
secondary source), agree with the dates this rule produces:

| Week | Bulletin heading | Derived by the rule |
|---|---|---|
| 2014-W01 | 29 Dec – 4 Jan 2014 | 2013-12-29 to 2014-01-04 |
| 2015-W01 | 4 – 10 Jan 2015 | 2015-01-04 to 2015-01-10 |

### 4.4 Week 53

| Question | Answer |
|---|---|
| Why does 2014 have a week 53? | 1 January 2015 was a Thursday. The week 28 Dec 2014 – 3 Jan 2015 has only 3 days in January, so it belongs to 2014 as week 53. |
| Why does 2020 have a week 53? | 1 January 2021 was a Friday. The week 27 Dec 2020 – 2 Jan 2021 has only 2 days in January, so it is 2020-W53. |
| Why is there no 2015-W53? | Under the Sunday rule 2015 runs from 4 Jan 2015 to 2 Jan 2016, which is exactly 52 weeks. 2015 has 53 weeks only under ISO-8601 (Monday start). Its absence is correct and is not a missing observation. |

Under ISO numbering the same calendar days would be labelled differently: the
53-week years would be 2015 and 2020, not 2014 and 2020. Re-labelling the file
with an ISO calendar library would therefore shift week labels by one for part
of the series. **ISO week functions must not be used on this file**
(for example `pandas.Series.dt.isocalendar()` or `%G-W%V`).

The two week-53 observations are ordinary 7-day weeks with values in line with
their neighbours (`tables/audit_week53_context.csv`):

| Week | Derived dates | Cases |
|---|---|---|
| 2014-W52 | 21–27 Dec 2014 | 188 |
| **2014-W53** | 28 Dec 2014 – 3 Jan 2015 | 158 |
| 2015-W01 | 4–10 Jan 2015 | 256 |
| 2020-W52 | 20–26 Dec 2020 | 196 |
| **2020-W53** | 27 Dec 2020 – 2 Jan 2021 | 193 |
| 2021-W01 | 3–9 Jan 2021 | 194 |

They are kept as they are. They are not dropped, merged into week 52 or
redistributed.

### 4.5 Continuity across year boundaries

Assigning each label its Sunday start date under the matching rule gives
**573 gaps of exactly 7 days and no other gap**. Every year boundary is a plain
7-day step (`tables/audit_year_boundaries.csv`). The series spans
2012-01-01 to 2022-12-31 with no missing and no repeated week.

This differs from DengAI, where the year-end week is stretched to 8–9 days.
Singapore has true 7-day weeks throughout.

## 5. Annual observation counts

| Year | Weeks | Year | Weeks |
|---|---|---|---|
| 2012 | 52 | 2018 | 52 |
| 2013 | 52 | 2019 | 52 |
| 2014 | **53** | 2020 | **53** |
| 2015 | 52 | 2021 | 52 |
| 2016 | 52 | 2022 | 52 |
| 2017 | 52 | **Total** | **574** |

9 × 52 + 2 × 53 = 574.

## 6. Assumptions and what was not verified

| # | Assumption or open point | Basis | Effect if wrong |
|---|---|---|---|
| A1 | Week 1 is the first Sunday–Saturday week with at least 4 days in January | Matches all 11 years; matches 2 bulletin headings; not stated by MOH in the dataset description | Derived calendar dates would shift. Row order, and therefore every model input, would not change |
| A2 | Derived week-start dates are used only for plot axes and for reporting fold boundaries | Design choice | None on results. Only the order of rows enters the models |
| A3 | Counts are final, not provisional | Source last updated 6 June 2024, 17 months after the last week | Late revisions would be small; not checkable from the file |
| A4 | The bulletin dates were read from a secondary archive, not from MOH PDFs | The MOH PDFs could not be opened as text during this audit | A1 would rest on the structural match alone, which is already unique among the rules tested |
| A5 | Whether cases are assigned to a week by notification date or by onset date | **Not established.** No source read states it | Affects comparability with DengAI (see `singapore_target_comparability.md`), not the audit |

## 7. Second Singapore CSV — completed 2026-11-19

The official MOH *Weekly Number of Dengue and Dengue Haemorrhagic Fever Cases*
dataset (data.gov.sg ID `d_ac1eecf0886ff0bceefbc51556247015`) was retrieved from
the API and compared with the project's selected `dengue_fever_cases` series over
2014-W01 to 2018-W52. All **261 expected weeks** appear in both files; there are
no coverage mismatches. The series values are not identical: the dedicated
MOH file's `Dengue` count matches 209 weeks and differs in **52 weeks** (19.9%),
by -8 to +1 cases (dedicated MOH count minus project count). Differences are
concentrated in 2016 (37 of 52 weeks); 2014 has no mismatches.

The broader MOH *Weekly Infectious Disease Bulletin Cases* file (dataset ID
`d_ca168b2cb763640d72c4600a68f9909e`) matches the selected project series exactly
for all 261 weeks. Thus the discrepancy is specific to the separate 2014–2018
Dengue/DHF release. Adding its DHF series to its Dengue values does not reconcile
the mismatch; the combined value still differs in 100 weeks, and the difference
is not equal to DHF week by week. Both files are official MOH releases. Available
files do not establish whether this reflects a data vintage, revision, or another
source difference, so we record it as unresolved and do not alter the selected
series.

The complete joined comparison is in `tables/overlap_check_2014_2018.csv`; all
52 mismatch rows are in `tables/overlap_mismatched_weeks_2014_2018.csv`. Frozen
source files, retrieval details, and checksums are in `data/raw/SOURCES.md` and
`tables/overlap_check_summary.json`. A longer interpretation is in
`singapore_overlap_results.md`. This is a source-consistency check, not an
independent validation series.

## 8. Sources

- [Weekly Infectious Disease Bulletin Cases — data.gov.sg](https://data.gov.sg/datasets/d_ca168b2cb763640d72c4600a68f9909e/view)
- [Weekly Number of Dengue and Dengue Haemorrhagic Fever Cases — data.gov.sg](https://data.gov.sg/collections/509/datasets/d_ac1eecf0886ff0bceefbc51556247015/view)
- [Singapore Weekly Infectious Disease Bulletin 2014 — FluTrackers archive](https://flutrackers.com/forum/forum/asia/seasonal-flu-2009-2014-including-h1n1-pandemic-2009-ab/singapore-aa/157811-singapore-weekly-infectious-disease-bulletin-2014)
- [Singapore Weekly Infectious Disease Bulletin 2015 — FluTrackers archive](https://flutrackers.com/forum/forum/asia/seasonal-flu-2009-2014-including-h1n1-pandemic-2009-ab/singapore-aa/721464-singapore-weekly-infectious-disease-bulletin-2015)
