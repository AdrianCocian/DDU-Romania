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

## Model rules (manuscript, sections 2.1–2.4)

| Item | Rule |
|---|---|
| Temperature | ERA5-Land hourly 2 m air temperature, 0.1° grid, 2005–2024 |
| Study area | cells of the grid touched by the national boundary of Romania and having an ERA5-Land value: 2,912 cells |
| Threshold | 14 °C |
| DDU methods (Table 1) | min–max average (`midpoint`), hourly average (`daily_mean`), Baskerville–Emin sine (`be_sine`), hourly sum in degree-hours (`hourly_sum`, reference) |
| Suitable day | DDU accumulated over the 30 days ending on that day ≥ 130 DDU (≥ 3120 degree-hours for the hourly sum); days at or below 14 °C add 0 and do not reset the window |
| Year wrap | the window of the first 29 days of a year is completed with the last 29 days of the previous year (2006–2024; in 2005, the first year of data, no day before 30 January can be suitable) |
| Season (3-day rule) | a run of suitable days ends when 3 or more consecutive days are unsuitable; interruptions of 1–2 days stay inside the run; the longest run of the year is the season |
| Metrics | onset and end (day of year), season length = number of days from onset to end, both included (bridged days count), total EIP days = number of suitable days in the year, inside or outside the season |
| Day of year | numbered on a 365-day basis: in leap years 29 February shares number 59 with 28 February, so a number is the same calendar date every year. No day is left out of the calculation |
| Consensus | per-cell median of the four methods (onset and end: median over the methods that give a season) |
| Suitability classes (Table 2) | limits = P25/P50/P75/P90 of hourly-sum season length over all cell-years with a season: 0, 1–80, 81–107, 108–123, 124–134, ≥135 days; relative within Romania |
| Statistics | Mann–Kendall (tie- and continuity-corrected), Sen's slope with 95% CI (Gilbert 1987), variance share of method choice, agreement with the reference (same class, Cohen's κ, Spearman ρ), Shapiro–Wilk and Hartigan dip test |

Total EIP days can be smaller than the season length (a season may contain one or two unsuitable days between
suitable ones) or larger (suitable days outside the season).

In the code the 3-day rule is `MIN_GAP_DAYS = 4` (`config.py`): a new run starts when the next suitable day is
4 or more days after the previous one. Earlier development versions used a 10-day gap; it can still be set with
`step05 --min-gap 10` for sensitivity runs.

## Repository layout

```
config.py                  paths, thresholds, method names, class rule (one place for all settings)
ddu_stats.py               statistics shared by steps 07-09 (trend, classes, agreement, variance partition)
run_pipeline.py            runs steps 1-10 in order
scripts/                   the pipeline steps (table below)
tools/                     verify_release.py, check_against_manuscript.py, release_checklist.py
results/                   tables and per-cell values of the published run (run_20260908_131817_6a83be93)
data/boundaries/           national boundary of Romania (ANCPI; EPSG:3844) and its licence note
tests/                     pytest suite
docs/                      pipeline overview, data sources, reproducibility notes
out/                       all generated outputs (not in git)
```

| Step | Script | Output (under `out/`) |
|---|---|---|
| 1 | `step01_download_ro.py` | `01_download/{YYYY}_{MM}_raw.nc`, ERA5-Land monthly files from the Copernicus CDS |
| 2 | `step02_std_clip_ro.py` | `02_standardized/`, `03_clip_ro/`: °C, common coordinate names, cells outside the boundary set to missing |
| 3 | `step03_concat_year.py` | `04_yearly_raw/{YYYY}.nc`, one hourly file per year |
| 3b | `step03b_qc_fill_year.py` | `06_filled_yearly/{YYYY}.nc` (temperature and quality flag) and `07_reports/qc_gaps_summary.csv` (missing hours per year) |
| 4 | `step04_tails.py` | `06_filled_yearly_tails_29d/{YYYY}_tail29d.nc`, last 29 days of each year |
| 5 | `step05_ddu_season_wrap.py` | `08_season_swrap/win30/csv/season_ALL_YEARS_methods.csv` (per cell, year, method) |
| 6 | `step06_ddu_method_compare.py` | `12_method_compare/win30/6_method_comparison/` consensus, pairwise differences, maps |
| 7 | `step07_suitability_classes.py` | `12_method_compare/win30/7_suitability_classes/` class limits, counts, maps |
| 8 | `step08_per_year_per_pixel.py` | `12_method_compare/win30/8_statistics/` national series, trends, variance shares |
| 9 | `step09_manuscript_statistics.py` | `12_method_compare/win30/00_MANUSCRIPT_PACK/` tables T1–T9, manuscript tables, `key_numbers.json` |
| 10 | `step10_manuscript_figures.py` | `00_MANUSCRIPT_PACK/figures/` Figures 1–3 (300 dpi PNG and PDF) |

Step 3b is a completeness check. ERA5-Land has no missing hours over land, so it copies the temperatures unchanged
and marks every value as original; if hours were missing it would fill them and flag each filled value.

## Installation

```bash
git clone https://github.com/AdrianCocian/DDU-Romania.git
cd DDU-Romania
conda env create -f environment.yml
conda activate ddu_romania
```

## Copernicus CDS key (only for step 1)

1. Create an account at https://cds.climate.copernicus.eu and accept the licence of the ERA5-Land dataset.
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

Without any download:

```bash
pytest -m "not slow"     # about one minute
pytest -m slow           # about four minutes
```

The fast tests check the DDU formulas, the season rule on temperatures with a known answer, the seasons against a
second implementation written from the definitions in the article (`tests/independent.py`), and the statistics
against the archived tables of the published run, including every number quoted in the article. The slow tests run
the pipeline from monthly files in the form delivered by the CDS (steps 2–5), steps 5–10 on a synthetic data set,
and the verification tool.

From the per-cell values in `results/` (written by step 5 of the published run):

```bash
python scripts/step09_manuscript_statistics.py --season-csv results/run_20260908_131817_6a83be93/season_ALL_YEARS_methods.csv.gz --out out/pack --no-excel
python tools/check_against_manuscript.py --pack out/pack
```

The check recomputes the values quoted in the article (class limits, agreement, variance shares, trends, class
shares, national medians and means, paired differences) and compares every table with the archived tables; it ends
with `RESULT: all checks passed` when the run reproduces the article.

On a computer that holds the hourly files of a run, one command repeats the whole comparison (hourly files against
step 3b, seasons recomputed with step 5, steps 6–10, numbers of the article) without writing into the data folder:

```bash
python tools/verify_release.py --data-root <folder that contains out/>
```

Its record for the published run is `results/run_20260908_131817_6a83be93/VERIFICATION.txt`. Details:
`docs/reproducibility.md`.

## Citation

Please cite the article (in review) and the software archive: Cocian AI. DDU-Romania. Zenodo.
https://doi.org/10.5281/zenodo.18843229 (all versions; each release has its own DOI). See `CITATION.cff`.

## Licences and data

- Code: MIT (`LICENSE`).
- National boundary (`data/boundaries/`): ANCPI, reusable under the Romanian Open Government Licence OGL-ROU-1.0;
  not covered by the MIT licence. See `data/boundaries/README.md`.
- ERA5-Land: Muñoz Sabater J. ERA5-Land hourly data from 1950 to present. Copernicus Climate Change Service (C3S)
  Climate Data Store (CDS); 2019. https://doi.org/10.24381/cds.e2161bac. Licence: CC BY 4.0 (the licence of the
  Copernicus data stores since 2 July 2025). The repository holds no ERA5-Land file; step 1 downloads them.
- Tables and per-cell values in `results/` are derived from ERA5-Land: contains modified Copernicus Climate Change
  Service information 2026. Neither the European Commission nor ECMWF is responsible for any use that may be made
  of the Copernicus information or data it contains.

Questions and problems: please open an issue on GitHub.
