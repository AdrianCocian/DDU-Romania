# Pipeline overview

All settings live in `config.py` (thresholds, window, 3-day rule, method names and colours, class rule, folder
names). Statistics shared by steps 07–09 live in `ddu_stats.py`, so the same code produces the trend, class and
agreement numbers everywhere. `run_pipeline.py` runs the steps in order.

## Inputs

- ERA5-Land hourly 2 m air temperature (`reanalysis-era5-land`, variable `2m_temperature`), bounding box
  48.5 N, 19.9 E, 43.5 N, 30.1 E, 2005–2024.
- National boundary of Romania, `data/boundaries/ro_frontiera_poligon.shp` (ANCPI, version 23.12.2021, Stereo 70,
  EPSG:3844; see `data/boundaries/README.md`).

## Steps

**1. Download** (`step01_download_ro.py`). Monthly requests to the Copernicus Climate Data Store; NetCDF, with GRIB
conversion as a fallback. Needs `~/.cdsapirc`. Output: `out/01_download/{YYYY}_{MM}_raw.nc`.

**2. Standardise and clip** (`step02_std_clip_ro.py`). Kelvin to °C, common coordinate names, latitude ascending.
The boundary is reprojected to EPSG:4326 and every cell that it touches is kept (`CLIP_ALL_TOUCHED = True`); the
other cells are set to missing. For Romania the polygon touches 2,937 cells, of which 2,912 have an ERA5-Land value
(the 25 others lie on the Black Sea coast and the coastal lagoons). Output: `out/02_standardized/`, `out/03_clip_ro/`.

**3. Yearly files** (`step03_concat_year.py`). One hourly file per year, `out/04_yearly_raw/{YYYY}.nc`
(8,760 or 8,784 hours); no value is changed.

**3b. Completeness check** (`step03b_qc_fill_year.py`). Counts, for every year, the hours without a value inside the
study area and writes the files read by steps 4 and 5: `out/06_filled_yearly/{YYYY}.nc` with the temperature and a
quality flag (0 original, 1 short gap interpolated in time, 2 climatology of the same day and hour, 3 mean of the
neighbouring cells), and the report `out/07_reports/qc_gaps_summary.csv`. ERA5-Land is complete over land, so the
temperatures are copied unchanged and every flag is 0; the filling branches exist so that an incomplete input
cannot pass unnoticed. `--report-only` writes no NetCDF file and compares existing files with the output of the step.

**4. Tails** (`step04_tails.py`). The last 29 days of every year, used to complete the 30-day window at the start of
the next year (`out/06_filled_yearly_tails_29d/`).

**5. DDU and seasons** (`step05_ddu_season_wrap.py`).
- Daily DDU with four methods (manuscript Table 1), threshold 14 °C:
  min–max average `max(0, (Tmin + Tmax)/2 − 14)`; hourly average `max(0, Tmean − 14)`;
  Baskerville–Emin sine (closed form of a sine day between Tmin and Tmax);
  hourly sum `Σ max(0, T_h − 14)` in degree-hours (reference).
- A day is suitable when the 30-day sum ending on that day reaches 130 DDU (3120 degree-hours for the hourly sum).
- 3-day rule: runs of suitable days separated by 1–2 unsuitable days are joined; 3 or more consecutive unsuitable
  days end a run (`MIN_GAP_DAYS = 4`). The longest run is the season.
- Year wrap: from 2006 the window of the first 29 days of the year is completed with the tail of the previous year.
  If a tail is missing the year is computed without it and the script says so.
- Per cell, year and method: onset and end (dates, and day numbers on a 365-day basis: in leap years 29 February
  shares number 59, no day is dropped), season length (days from onset to end, both included), total EIP days (all
  suitable days of the year), number of runs and suitable days before and after the season.
- Output: `out/08_season_swrap/win30/csv/season_ALL_YEARS_methods.csv`, `summary_ALL_YEARS_methods.csv`,
  `META.json` (run_id, settings) and per-year files. Per-year files are reused only when their settings stamp
  (`{YYYY}_settings.json`) matches the current settings; otherwise the year is recomputed.

**6. Method comparison** (`step06_ddu_method_compare.py`). Consensus (per-cell median of the methods), pairwise
differences (method1 − method2, as in the manuscript tables), agreement thresholds, maps per year.
Output: `out/12_method_compare/win30/6_method_comparison/`.

**7. Suitability classes** (`step07_suitability_classes.py`). Class limits = P25/P50/P75/P90 of the hourly-sum
season length over all cell-years with a season (published run: 81, 108, 124, 135 days); the same limits for every
method and for the consensus. Output: `7_suitability_classes/` (limits in `00_meta/class_breaks_META_7.json`,
counts, maps). `--breaks 81,108,124,135` applies fixed limits to another period.

**8. Per-year statistics** (`step08_per_year_per_pixel.py`). Year-count matrices, national median and IQR per year,
Mann–Kendall and Sen's slope with 95% CI for the median and mean series (consensus and each method), variance share
of method choice, distribution diagnostics. Output: `8_statistics/`.

**9. Manuscript tables** (`step09_manuscript_statistics.py`). Tables T1–T9, per-cell classes, the manuscript tables
(Tables 3–5, S1–S7) and `key_numbers.json` with every number quoted in the text. Output: `00_MANUSCRIPT_PACK/`.

**10. Figures** (`step10_manuscript_figures.py`). Figure 1 (consensus season length, 2005/2011/2017/2024), Figure 2
(suitability classes, hourly sum), Figure 3 (national median season length per method), 300 dpi PNG and PDF.

## Method names

| Key in files | Name in the manuscript |
|---|---|
| `midpoint` | Min–max average |
| `daily_mean` | Hourly average |
| `be_sine` | BE-sine (Baskerville–Emin) |
| `hourly_sum` | Hourly sum (reference) |
| `consensus` | Consensus (median of the four) |
