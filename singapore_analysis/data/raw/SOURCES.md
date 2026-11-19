# Singapore overlap-check source snapshots

These snapshots support the comparison requested for the overlapping Singapore surveillance files. They are additional to, and do not replace, `data/raw/singapore_dengue_fever_weekly_clean.csv`.

1. `moh_weekly_dengue_df_dhf_2014_2018_api_snapshot.json` is the full 530-record data.gov.sg API response for MOH dataset `d_ac1eecf0886ff0bceefbc51556247015`, retrieved 2026-11-19 from:
   `https://data.gov.sg/api/action/datastore_search?resource_id=d_ac1eecf0886ff0bceefbc51556247015&limit=1000`
   The dataset page describes laboratory-confirmed Dengue and Dengue Haemorrhagic Fever counts for 2014–2018. Rows marked `NA` (2018-W53) are outside the comparison interval.
2. `WeeklyInfectiousDiseaseBulletinCases.csv` is the MOH data.gov.sg dataset `d_ca168b2cb763640d72c4600a68f9909e`, downloaded 2026-11-16, and is the broader weekly bulletin source present on the user's machine. It contains disease-specific weekly entries across years.

The exact file checksums are recorded in the project's overlap audit. The 2014-W01–2018-W52 comparison table and mismatched rows are in `singapore_analysis/tables/overlap_check_2014_2018.csv` and `overlap_mismatched_weeks_2014_2018.csv`.

## References

- Ministry of Health. *Weekly Number of Dengue and Dengue Haemorrhagic Fever Cases* (data.gov.sg dataset `d_ac1eecf0886ff0bceefbc51556247015`), https://data.gov.sg/collections/509/datasets/d_ac1eecf0886ff0bceefbc51556247015/view (accessed 2026-11-19).
- Ministry of Health. *Weekly Infectious Disease Bulletin Cases* (data.gov.sg dataset `d_ca168b2cb763640d72c4600a68f9909e`), https://data.gov.sg/datasets/d_ca168b2cb763640d72c4600a68f9909e/view (accessed 2026-11-19).

## SHA-256

- `moh_weekly_dengue_df_dhf_2014_2018_api_snapshot.json`: `9b0a8c130b4f32813a37db2dbc42af91157584fcbe4dbbfc96ddf37dba31d076`
- `WeeklyInfectiousDiseaseBulletinCases.csv`: `b782b6570e3e3ba4377b487b0c1c40905d0d7dd40db0e716c8a659bb53310a54`
