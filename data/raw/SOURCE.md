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
