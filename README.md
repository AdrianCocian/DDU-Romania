# DDU-Romania: Dirofilaria development units and transmission seasons in Romania, 2005–2024

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18843229.svg)](https://doi.org/10.5281/zenodo.18843229)

Code and outputs of the study *Comparative modelling of Dirofilaria spp. transmission risk (2005–2024) using DDU*
(A. I. Cocian, M. Petrić, A. D. Mihalca). ERA5-Land hourly 2 m air temperature is converted into Dirofilarial
Development Units (DDU) with four methods; for every grid cell, year and method the pipeline derives the period in
which larvae can reach the infective stage in the mosquito, compares the methods and reproduces the tables and figures
of the manuscript.

The model describes temperature-bounded larval development only. It does not represent vector abundance, species
composition, biting behaviour, host density, chemoprophylaxis or the movement of infected animals; the outputs are
climatic suitability indicators, not observed infection risk.

## Model rules (manuscript, sections 2.2–2.6)

| Item | Rule |
|---|---|
| Temperature | ERA5-Land hourly 2 m air temperature, ~0.1° grid, 2,912 cells inside Romania, 2005–2024 |
| Threshold | 14 °C |
| DDU methods (Table 1) | min–max average (`midpoint`), hourly average (`daily_mean`), Baskerville–Emin sine (`be_sine`), hourly sum in degree-hours (`hourly_sum`, reference) |
| Suitable day | DDU accumulated over the 30 days ending on that day ≥ 130 DDU (≥ 3120 degree-hours for the hourly sum); cold days add 0 and do not reset the window |
| Year wrap | the first 29 days of a year use the last 29 days of the previous year (from 2006) |
| Season (3-day rule) | a run of suitable days ends when 3 or more consecutive days are unsuitable; gaps of 1–2 days stay inside; the longest run of the year is the season |
| Metrics | onset (DOY), end (DOY), season length = end − onset + 1, total EIP days = all suitable days of the year |
| Day of year | 365-day basis (29 February omitted), so a DOY is the same date every year |
| Consensus | per-cell median of the four methods (onset and end: median over the methods with a season) |
| Suitability classes (Table 2) | limits = P25/P50/P75/P90 of hourly-sum season length over all cell-years with a season: 0, 1–80, 81–107, 108–123, 124–134, ≥135 days; relative within Romania |
| Statistics | Mann–Kendall (tie- and continuity-corrected), Sen's slope with 95% CI (Gilbert 1987), variance share of method choice, agreement with the reference (same class, Cohen's κ, Spearman ρ), Shapiro–Wilk and Hartigan dip test |

In the code the 3-day rule is `MIN_GAP_DAYS = 4` (`config.py`): a new run starts when the next suitable day is
4 or more days after the previous one. Earlier development versions used a 10-day gap; it can still be set with
`step05 --min-gap 10` for sensitivity runs.

## Repository layout

```
config.py                  paths, thresholds, method names, class rule (one place for all settings)
ddu_stats.py               statistics shared by steps 07-09 (trend, classes, agreement, variance partition)
run_pipeline.py            runs steps 1-10 in order
scripts/step01-step10      the pipeline steps (see below)
tools/                     check_against_manuscript.py, export_results.py
results/                   archived tables of the published run (run_20260908_131817_6a83be93)
data/boundaries/           Romania boundary (EPSG:4326)
tests/                     pytest suite (formulas, 3-day rule, statistics, synthetic end-to-end run)
docs/                      pipeline overview, data sources, reproducibility notes
out/                       all generated outputs (not in git)
```

| Step | Script | Output (under `out/`) |
|---|---|---|
| 1 | `step01_download_ro.py` | ERA5-Land monthly files from the Copernicus CDS |
| 2 | `step02_std_clip_ro.py` | standardised (°C) files clipped to Romania |
| 3 | `step03_concat_year.py` | `06_filled_yearly/{YYYY}.nc` |
| 4 | `step04_tails.py` | `06_filled_yearly_tails_29d/{YYYY}_tail29d.nc` |
| 5 | `step05_ddu_season_wrap.py` | `08_season_swrap/win30/csv/season_ALL_YEARS_methods.csv` (per cell, year, method) |
| 6 | `step06_ddu_method_compare.py` | `12_method_compare/win30/6_method_comparison/` consensus, pairwise differences, maps |
| 7 | `step07_suitability_classes.py` | `12_method_compare/win30/7_suitability_classes/` class limits, counts, maps |
| 8 | `step08_per_year_per_pixel.py` | `12_method_compare/win30/8_statistics/` national series, trends, variance shares |
| 9 | `step09_manuscript_statistics.py` | `12_method_compare/win30/00_MANUSCRIPT_PACK/` tables T1–T9, manuscript tables, `key_numbers.json` |
| 10 | `step10_manuscript_figures.py` | `00_MANUSCRIPT_PACK/figures/` Figures 1–3 (300 dpi PNG and PDF) |

## Installation

```bash
git clone https://github.com/AdrianCocian/DDU-Romania.git
cd DDU-Romania
conda env create -f environment.yml
conda activate ddu_romania
```

## Copernicus CDS key (only for step 1)

1. Create an account at https://cds.climate.copernicus.eu and accept the ERA5-Land licence.
2. Put your personal key in `~/.cdsapirc` (Windows: `C:\Users\<user>\.cdsapirc`), as described on the
   "How to API" page of the CDS. Never commit this file; `.gitignore` excludes it.

## Running

```bash
python run_pipeline.py                      # all steps, 2005-2024 (the download takes many hours)
python run_pipeline.py --from 5             # hourly yearly files already in out/06_filled_yearly*
python run_pipeline.py --from 9 --to 10     # tables and figures only, from the step05 output
python run_pipeline.py --dry-run            # show the commands
```

Outputs go to `<root>/out/`; the root is this folder unless `--root` or the environment variable `DDU_ROOT`
points elsewhere. Every step can also be run on its own (`python scripts/step07_suitability_classes.py --help`).

## Reproducing the numbers of the manuscript

Without downloading anything:

```bash
pytest -m "not slow"     # formulas, 3-day rule, and statistics against the archived tables of the published run
pytest -m slow           # steps 5-10 end to end on a small synthetic dataset (about 2 minutes)
```

From the per-cell output of the published run (`season_ALL_YEARS_methods.csv`, written by step 5):

```bash
python run_pipeline.py --from 9 --to 10
python tools/check_against_manuscript.py
```

The check recomputes the values quoted in the paper (class limits, agreement, variance shares, trends, class shares,
national medians) and compares every table with `results/run_20260908_131817_6a83be93/`; it ends with
`RESULT: all checks passed` when the run reproduces the paper.

## Citation

Please cite the article (in review) and the software archive: Cocian AI. DDU-Romania. Zenodo.
https://doi.org/10.5281/zenodo.18843229 (all versions; each release has its own DOI). See `CITATION.cff`.

## Licence and data

Code: MIT (see `LICENSE`). ERA5-Land data: Copernicus Climate Change Service (C3S) Climate Data Store; users must
accept the ERA5-Land licence. Questions and problems: please open an issue on GitHub.
