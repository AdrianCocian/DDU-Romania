# Results of the published run

`run_20260908_131817_6a83be93/` holds the outputs of the run reported in the manuscript (ERA5-Land, Romania,
2005–2024, 2,912 grid cells; 14 °C, 130 DDU or 3120 degree-hours in 30-day windows; 3-day season rule).
The run_id is the one written by step05 (`META_step05.json`).

| File | Content |
|---|---|
| `META_step05.json` | season settings of step05 (threshold, window, `min_gap_days` = 4, year wrap) |
| `META_step07_class_breaks.json` | suitability class limits (P25/P50/P75/P90 of hourly sum: 81, 108, 124, 135 days) |
| `META_step08.json` | statistics settings and library versions |
| `per_cell_classes_ALL_YEARS.csv` | season length (days) per cell and year for the four methods and the consensus, with suitability classes 0–5 (58,240 rows) |
| `T1_trend_MK_Sen_all_methods.csv` | Mann–Kendall and Sen's slope (95% CI), national median and mean series |
| `T2_distribution_diagnostics_summary.csv` | Shapiro–Wilk, Hartigan dip test, skewness (summary over years) |
| `T3_variance_ratio_national_and_percell.csv` | variance share of method choice (national series; per-cell values) |
| `T4_class_agreement_pooled.csv`, `T4b_class_agreement_by_year.csv` | agreement of each method with the hourly sum (same class, κ, ρ, differences) |
| `T5_national_summary_all_methods.csv` | national median, mean, IQR and share of cells with a season, per year and method |
| `T6b_pairwise_delta_by_year.csv` | paired differences method1 − method2 per year |
| `T7_suitability_class_counts_by_year.csv` | number and share of cells per class, year and method |
| `T8_bias_by_reference_class.csv` | difference from the hourly sum by reference class |
| `T9_median_gap_diagnostic.csv` | gap between national medians versus paired differences |

Variables: `season_length`, `start_doy`, `end_doy` (day numbers on a 365-day basis: in leap years 29 February
shares number 59; no day is left out of the season length) and `total_eip_days`.
Method keys: `midpoint` = min–max average, `daily_mean` = hourly average, `be_sine` = BE-sine,
`hourly_sum` = hourly sum (reference), `consensus` = median of the four.

Files written by `python tools/verify_release.py --data-root <folder of the published run>`:

| File | Content |
|---|---|
| `season_ALL_YEARS_methods.csv.gz` | per-cell file of step05, unchanged: onset, end, season length, total EIP days, number of runs and suitable days outside the season for every cell, year and method (2,912 × 20 × 4 = 232,960 rows) |
| `summary_ALL_YEARS_methods.csv` | per-year summary written by step05 |
| `qc_gaps_summary.csv` | completeness of the hourly input per year: hours without a value inside the study area, and whether the hourly file equals the output of step 3b (present when stage 3 was run) |
| `VERIFICATION.txt` | record of the verification: stages, software versions, result |

In the three `META_*.json` files the folder of the run on the author's computer is written as `<data root>`;
nothing else was changed.

All files in this folder are derived from ERA5-Land: contains modified Copernicus Climate Change Service
information 2026. Neither the European Commission nor ECMWF is responsible for any use that may be made of the
Copernicus information or data it contains.

`python tools/check_against_manuscript.py --pack <folder of a regenerated step09 pack>` compares a regenerated run
with these tables and with the numbers printed in the manuscript.
