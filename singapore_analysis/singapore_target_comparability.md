# Target comparability: DengAI `total_cases` vs Singapore `dengue_fever_cases` (Phase 3)

Prepared 2026-11-17. Statements are marked **Sourced** (read in a cited page),
**Measured** (computed from data) or **Not established** (no source found).

## 1. Conclusion

Singapore is an **external validation** of the study's method on a different
surveillance system. It is **not an exact replication**. The two targets are both
weekly dengue counts built largely on laboratory testing, but they differ in case
definition, in what is counted, in how weeks are dated, in ascertainment and in
scale. A result on Singapore can show whether the short-history finding
*transfers* to a different setting. It cannot be pooled with, or averaged
against, the San Juan and Iquitos numbers.

## 2. Side-by-side comparison

| Dimension | DengAI San Juan | DengAI Iquitos | Singapore |
|---|---|---|---|
| Column | `total_cases` | `total_cases` | `dengue_fever_cases` |
| Publisher | Puerto Rico Department of Health and US CDC | US NAMRU-6 and UC Davis | Singapore Ministry of Health |
| What is counted | Serotyped lab-confirmed cases + lab-positive cases without serotype + **estimated** positives among untested cases | Serotyped lab-confirmed cases + **estimated** additional cases | Notified cases classified as Dengue Fever |
| Contains modelled or imputed counts | **Yes** (`additional_cases`) | **Yes** (`additional_cases`) | No evidence of any |
| Severity scope | All lab-positive dengue; no split by severity | Same | **Dengue Fever only.** DHF is a separate series and is excluded |
| Laboratory basis | PCR and IgM; methods changed in 2005, 2007 and 2009 | IgM ELISA and RT-PCR, consistent over time | NS1 antigen, PCR, serology. Annual total matches MOH's "laboratory cases" figure for 2014 (see 3.3) |
| Capture mechanism | Surveillance system run with CDC; passive/active not stated in metadata | Enrolment of patients at clinics, 5–7 hours a day, five days a week | Statutory notification by doctors and laboratories within 24 hours |
| Week dating | By symptom onset; final data | By symptom onset; final data | **Not established** (notification or onset) |
| Week structure | Year-end week stretched to 8–9 days; 52 weeks every year | Same | True 7-day Sunday–Saturday weeks; week 53 in 2014 and 2020 |
| Period | 1990–2008 | 2000–2010 | 2012–2022 |
| Observations | 936 | 520 | 574 |
| Weekly count: min / median / max | 0 / 19 / 461 | 0 / 5 / 116 | 24 / 208 / 1,791 |
| Zero-case weeks | 4 of 936 | 96 of 520 | None |

## 3. Evidence

### 3.1 DengAI `total_cases`

- **Sourced** (San Juan metadata, CDC repository): `total_cases` is "the sum of
  all cases (denv1-4_cases + other_positive_cases + additional_cases)".
  `other_positive_cases` are "laboratory-positive cases without serotype
  identified (these include acute IgM positive and IgM conversions)".
  `additional_cases` are "estimated by multiplying the number of untested cases
  by the rate of laboratory-positive cases amongst those that were tested".
- **Sourced** (Iquitos metadata): `total_cases` is "the sum of all cases
  (denv1-4_cases + additional_cases)". Patients "were enrolled 5-7 hours a day,
  five days a week".
- **Sourced** (Johansson et al. 2019): "Weekly laboratory-confirmed and
  serotype-specific dengue surveillance data"; "All data were final, reflecting
  all cases with onset in each week regardless of reporting delays"; "data from
  31 December (30 December for leap years) were removed to ensure 52 wk/y"; and
  "The datasets also do not represent all infections or even all cases".
- **Not established:** whether the DengAI counts separate dengue fever from
  severe forms. No page read mentions DF, DHF or DSS, so `total_cases` is read
  here as all lab-positive dengue regardless of severity.

A note on the existing project documentation: `data/raw/SOURCE.md` describes the
DengAI counts as "laboratory-confirmed". The upstream metadata shows this is a
simplification, because `total_cases` also contains estimated positives among
untested cases. `SOURCE.md` has not been edited.

### 3.2 Singapore `dengue_fever_cases`

- **Measured:** the file equals the MOH "Dengue Fever" series from the Weekly
  Infectious Disease Bulletin dataset for all 574 weeks
  (see `singapore_data_audit.md`, section 3).
- **Sourced** (Communicable Diseases Agency, dengue page): notification is by
  "Medical practitioners" and "Laboratories", "On clinical suspicion or
  laboratory confirmation", "No later than 24 hours from the time of diagnosis".
  The same page directs readers to the bulletin "for the numbers of confirmed
  dengue cases in Singapore".
- **Sourced** (same page): DHF requires all four of fever, bleeding, platelet
  count below 100 × 10⁹/L and evidence of increased capillary permeability.
  Dengue Fever is the classification for cases that do not meet those criteria.
  The page does not name the WHO 1997 classification as its basis, although
  these four criteria correspond to it.
- **Not established:** whether weekly counts are dated by notification or onset;
  whether imported cases are included; whether historical weeks are revised.

### 3.3 A cross-check on the laboratory basis

**Measured:** the 2014 total in the file is 18,306 Dengue Fever cases, and the
source's separate DHF series totals 20 for 2014. Their sum is 18,326.
**Sourced:** MOH's 2014 surveillance report gives "18,326 laboratory cases of
DF/DHF reported". The exact match indicates that the bulletin's weekly series
counts laboratory-confirmed cases, at least for 2014. This is one year only; it
is evidence, not proof, for the other ten.

### 3.4 Size of the DHF exclusion

**Measured** from the MOH source (not from the selected file, and not combined
with it): 424 DHF cases against 164,023 Dengue Fever cases over 2012–2022. DHF is
0.26% of the combined total, between 0.11% and 0.79% in any year, and never more
than 8 cases in a week. Excluding DHF changes the level and shape of the series
negligibly. It still makes the *definition* different from DengAI's, which is
why it is documented and not ignored.

## 4. Why the differences matter for this study

| Difference | Consequence for a history-window experiment |
|---|---|
| **Scale.** Singapore's median week is about 11 times San Juan's | MAE and RMSE are in cases and are not comparable across datasets. Only within-dataset comparisons (window vs window, model vs persistence) and relative quantities (the 5% band, percentage differences) transfer |
| **No zero weeks, higher counts** | Relative noise from small counts is lower in Singapore. Week-to-week changes are a smaller fraction of the level, which favours persistence and short windows for reasons of counting statistics alone |
| **Week dating (onset vs possibly notification)** | A notification-dated series lags and smooths the onset curve by the reporting delay. The lag structure the models learn could differ for this reason and not because the disease dynamics differ |
| **Estimated counts in DengAI** | San Juan's `additional_cases` scale with the number of untested patients, which adds a component Singapore's series does not have |
| **Ascertainment** | Singapore has statutory dual notification, widespread point-of-care NS1 testing and intensive vector control. Iquitos is clinic enrolment with limited hours. The fraction of infections captured differs and may differ over time within each series |
| **Changing diagnostics in San Juan** | Three method changes fall inside the San Juan series. No equivalent break is documented for Singapore 2012–2022, though none was specifically searched for |
| **Era and context** | Singapore's window includes the COVID-19 period (2020–2021) and record epidemics in 2020 and 2022. Health-seeking and movement patterns changed in ways the DengAI series never experienced |
| **Week 53 and 7-day weeks** | Singapore's weeks are uniform. DengAI's year-end weeks are longer, which slightly inflates one count per year. Neither is corrected |
| **Severity scope** | Negligible numerically (section 3.4), but the targets are not the same quantity |

## 5. What may and may not be claimed

**May be claimed**, if the results support it:

- That the *qualitative* finding (short windows at least as good as long ones;
  how models compare with persistence) does or does not hold on an independent
  surveillance system in a different region and decade.
- Within-Singapore statements using the same pre-registered 5% rule.

**May not be claimed:**

- That Singapore "replicates" San Juan or Iquitos.
- Any comparison of absolute MAE or RMSE between Singapore and DengAI.
- Any pooled or averaged result across the three series.
- That a difference between Singapore and DengAI is caused by disease dynamics,
  since case definition, dating, ascertainment and era all differ at once.
- That the second Singapore CSV is an independent replication. It covers the same
  system and overlapping years.

## 6. Decisions

| # | Decision | Reason |
|---|---|---|
| D1 | Use `dengue_fever_cases` as supplied, as the sole target | It is the unaltered MOH series |
| D2 | Do not add DHF | Different target definition from the one supplied; instructed exclusion; numerically negligible |
| D3 | No rescaling, per-capita conversion or transformation to "match" DengAI | The original study uses raw counts; changing this would alter the method |
| D4 | Report Singapore results in their own tables and files, never merged with DengAI results | Prevents an implied pooled comparison |
| D5 | Describe the work as "external validation on a different operational case definition" in every write-up | Accuracy of the claim |

## 7. Sources

- [San Juan metadata — CDC dengue-forecasting-project-2015](https://raw.githubusercontent.com/cdcepi/dengue-forecasting-project-2015/master/Dengue_data/san_juan_metadata.md)
- [Iquitos metadata — CDC dengue-forecasting-project-2015](https://raw.githubusercontent.com/cdcepi/dengue-forecasting-project-2015/master/Dengue_data/iquitos_metadata.md)
- [Johansson et al. 2019, PNAS — An open challenge to advance probabilistic forecasting for dengue epidemics](https://pmc.ncbi.nlm.nih.gov/articles/PMC6883829/)
- [DengAI problem description — DrivenData](https://www.drivendata.org/competitions/44/dengai-predicting-disease-spread/page/82/)
- [Weekly Infectious Disease Bulletin Cases — data.gov.sg](https://data.gov.sg/datasets/d_ca168b2cb763640d72c4600a68f9909e/view)
- [Dengue fever, information for professionals — Communicable Diseases Agency Singapore](https://www.cda.gov.sg/professionals/diseases/dengue-fever/)
- [Communicable Diseases Surveillance in Singapore 2014 — CDA](https://www.cda.gov.sg/resources/communicable-diseases-surveillance-in-singapore-2014/)
- [Infectious disease notification — Ministry of Health Singapore](https://www.moh.gov.sg/seeking-healthcare/overview-of-diseases/communicable-diseases/infectious-disease-notification/)
