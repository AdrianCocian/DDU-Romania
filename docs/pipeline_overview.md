

# Pipeline overview

This document describes the workflow implemented by run_pipeline.py and scripts/step01–step08.

Inputs
- Hourly near-surface temperature (ERA5-Land) for the study period.
- Romania boundary polygon shapefile (boundaries/; EPSG:4326).
- Configuration in config.py.

Outputs
All generated outputs are written under out/ (gitignored). Key directories:
- out/08_season_swrap/         season metrics from step05
- out/12_method_compare/       method comparisons + maps + workbooks

Steps

Step 0: config.py
- Portable configuration for paths, thresholds, plotting defaults, and boundary auto-detection.

Step 01: scripts/step01_download_ro.py
- Downloads hourly temperature from CDS/ERA5-Land for the requested time range and ROI.

Step 02: scripts/step02_std_clip_ro.py
- Standardizes and clips the temperature data to Romania polygon.

Step 03: scripts/step03_concat_year.py
- Concatenates monthly files into yearly NetCDF.

Step 04: scripts/step04_tails.py
- Builds “tail” files to support year-wrap logic for rolling windows (e.g., 29-day tail for WIN30).

Step 05: scripts/step05_ddu_season_wrap.py
- Computes per-pixel season metrics under multiple DDU calculation methods.
- Outputs (per window):
  - out/08_season_swrap/win{W}/csv/season_ALL_YEARS_methods.csv
  - out/08_season_swrap/win{W}/csv/summary_ALL_YEARS_methods.csv

Step 06: scripts/step06_ddu_method_compare.py
- Builds consensus (median across methods) and pairwise deltas, and generates tables/plots/maps.
- Outputs:
  - out/12_method_compare/win{W}/6.compare/

Step 07 (WIN30 only): scripts/step07_risk_class_map.py
- Converts season_length into fixed discrete risk classes (0..5) and generates maps/plots.

Step 08 (WIN30 only): scripts/step08_per_year_per_pixel.py
- Produces yearly distribution matrices (DOY/month/length/EIP), national summaries, and mandatory stats workbooks:
  - Dist_<var>: skewness, Shapiro-Wilk, Hartigan dip test
  - Trend_<var>: Mann–Kendall, Sen slope, Sen intercept
  - VarRatio_<var>: method uncertainty vs interannual signal
  - NatSum_<var>: national median and IQR
