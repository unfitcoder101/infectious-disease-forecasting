# Data provenance

## Dataset
NOAA / CDC **Dengue Forecasting Project** (2015) training data, as distributed in the
DrivenData competition "DengAI: Predicting Disease Spread".

Weekly laboratory-confirmed dengue case counts for:
- `sj` = San Juan, Puerto Rico (936 weeks, 1990-04-30 to 2008-04-22)
- `iq` = Iquitos, Peru (520 weeks, 2000-07-01 to 2010-06-25)
                                
## Files
- `dengue_labels_train.csv` — city, year, weekofyear, total_cases   (the surveillance counts)
- `dengue_features_train.csv` — same keys plus `week_start_date` and climate covariates.
  This project uses ONLY `week_start_date` from this file, as the authoritative time index.
  No climate covariates are used: the research question is about history length, not predictors.
                                           
## How it was obtained (2026-09-16)
The original primary host, https://dengueforecasting.noaa.gov/ , NO LONGER RESOLVES
(DNS failure; the project site has been decommissioned). The data was therefore taken
from public GitHub mirrors.
                         
To guard against a corrupted or altered mirror, the files were downloaded from two
unrelated repositories and confirmed BYTE-IDENTICAL:
  - https://raw.githubusercontent.com/rushanshakya/PredictiveAnalysis-PredictDengueSpread/master/
  - https://raw.githubusercontent.com/Yasirurandeepa/DengAI-Time-Series-Analysis/master/

md5 checksums:
  dengue_labels_train.csv    5df4b6d0240bb83d4b509cc045b46603
  dengue_features_train.csv  a4614b3c80dbbba78d6d08b870b6eb20

## Status
This is REAL surveillance data, not synthetic. It is public and widely redistributed.

---

# Singapore external validation dataset

Added 2026-11-17. Used only by `singapore_analysis/`. It is an external validation
series with a different case definition from the DengAI data above, not a
replication of it.

## Dataset
Singapore Ministry of Health (MOH), **Weekly Infectious Disease Bulletin Cases**,
as published on data.gov.sg:
https://data.gov.sg/datasets/d_ca168b2cb763640d72c4600a68f9909e/view
(resource id `d_ca168b2cb763640d72c4600a68f9909e`; coverage January 2012 to
December 2022; source last updated 6 June 2024).

Weekly notified **Dengue Fever** cases for Singapore: 574 weeks, 2012-W01 to 2022-W52.
**Dengue Haemorrhagic Fever (DHF) is excluded.** MOH publishes it as a separate
series, and it has not been added to this file.

## File
- `singapore_dengue_fever_weekly_clean.csv` — time_index, epi_week, epi_year,
  epi_week_number, dengue_fever_cases.
  `dengue_fever_cases` is the MOH "Dengue Fever" series. `time_index`, `epi_year` and
  `epi_week_number` are helper columns derived from the `epi_week` label.
  The file has no calendar dates. Epidemiological weeks run Sunday to Saturday, and
  week 53 exists in 2014 and 2020 (not ISO-8601 numbering).

## Verification (2026-11-17)
All 574 weekly values were compared with the MOH dataset above, queried through the
data.gov.sg API for `disease = "Dengue Fever"`: 574 weeks on each side, none missing
from either, **0 values different**.

md5 checksum:
  singapore_dengue_fever_weekly_clean.csv    055198432469886d428416e003a00b2a

Full audit: `singapore_analysis/singapore_data_audit.md`.

## Licence
Singapore Open Data Licence, as stated on the dataset page. **The full licence text
has not been read**, so its attribution and reuse terms have not been checked.

## Status
This is REAL surveillance data, not synthetic.
