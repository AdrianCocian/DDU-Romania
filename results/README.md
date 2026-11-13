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

Variables: `season_length`, `start_doy`, `end_doy` (DOY on a 365-day basis) and `total_eip_days`.
Method keys: `midpoint` = min–max average, `daily_mean` = hourly average, `be_sine` = BE-sine,
`hourly_sum` = hourly sum (reference), `consensus` = median of the four.

The per-cell season file of step05 (`season_ALL_YEARS_methods.csv.gz`, onset, end, length and total EIP days for
every cell, year and method) is added with `python tools/export_results.py`.

`python tools/check_against_manuscript.py` compares a regenerated run with these tables and with the numbers printed
in the manuscript.
