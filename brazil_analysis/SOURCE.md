# Brazil dataset source and audit

## Source

- Project: InfoDengue, the weekly arbovirus surveillance and nowcasting platform developed with Brazilian public-health partners.
- Series: reported dengue cases (`casos`) for the municipality of Rio de Janeiro, IBGE code `3304557`.
- API documentation: <https://info.dengue.mat.br/tutorial_api_python/locale-en>
- Data dictionary: <https://info.dengue.mat.br/services/api>
- API query used: <https://info.dengue.mat.br/api/alertcity?geocode=3304557&disease=dengue&format=csv&ew_start=1&ew_end=53&ey_start=2010&ey_end=2025>
- Retrieved: 2026-11-18 (client local date, Asia/Kolkata).
- Requested period: epidemiological weeks 1–53, 2010–2025.
- Saved source snapshot: `data/raw/infodengue_rio_2010_2025_snapshot_20261118.csv`.
- SHA-256: `e361cd41ba5a0a205179c15addd91e5126bf6ebf0f24a9122d792d125937fe11`.

The API describes `casos` as the number of reported cases per week and notes that values are revised retrospectively each week. The raw file records the API's `versao_modelo` field, whose latest value in this snapshot is 2026-11-17. This experiment uses `casos`; it does not use `casos_est` (InfoDengue's nowcast estimate), climate variables, reproduction-number estimates, incidence rates, or other covariates. Keep this dated snapshot with the project so the experiment is reproducible despite later source revisions.

## Initial structural audit

- Rows: 835 weekly observations.
- Date range: 2010-01-03 through 2025-12-28.
- Dates: strictly increasing after sorting; no duplicates.
- Spacing: all 834 consecutive intervals are exactly 7 days.
- Missing reported counts: 0.
- Negative reported counts: 0.
- Reported cases (`casos`): min 3, median 171, max 16,339.

The API's dates mark Sunday starts of epidemiological weeks. Its case counts are reported surveillance counts revised over time; their operational definition and ascertainment are not identical to DengAI or Singapore MOH. This is external validation on a distinct series, not a pooled comparison or a numerical comparison of MAE across locations.
