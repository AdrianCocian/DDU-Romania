# Reproducibility

## Environment

`environment.yml` pins the versions recorded for the published run (Python 3.11, NumPy 1.26.4, pandas 3.0.0,
SciPy 1.16.2, Matplotlib 3.10.6, diptest 0.10.0). Every step writes its settings to a `META*.json` file next to its
outputs; step08 also records the library versions.

## Determinism

- No random numbers enter the seasons, classes or trends. The Shapiro–Wilk test uses a random subsample of
  5,000 cells when a year has more (seed 42), so its p-values are reproducible.
- Per-year files of step05 are reused only when `{YYYY}_settings.json` matches the current settings (threshold,
  window, 3-day rule, year wrap); otherwise the year is recomputed. `--force` always recomputes.
- A run of step05 on some years writes `season_ALL_YEARS_methods.csv` for those years only; run it on the whole
  period before steps 6–10.

## Provenance of the published run

The run reported in the article has run_id `20260908_131817_6a83be93` (8 September 2026; settings in
`results/run_20260908_131817_6a83be93/META_step05.json`). It was produced on the first author's workstation with
the working copy of this pipeline as it stood at that date. The code released here is that pipeline after
clean-up: the same formulas, thresholds and season rule; method names as in the manuscript; one statistics module
shared by steps 7–9; the completeness step 3b; tests and the verification tool. Because the code was reorganised
after the run, its equivalence with the published run is demonstrated rather than assumed:

| What is compared | How | Where |
|---|---|---|
| Statistics, classes, trends, agreement, tables T1–T9 | recomputed from the archived per-cell values and compared with the archived tables | `tests/test_stats_published.py` |
| Every number quoted in the article | 165 values of the Results section recomputed from the tables | `tools/check_against_manuscript.py`, `tests/test_stats_published.py` |
| Study area | the 2,912 cells against the boundary file | `tests/test_stats_published.py` |
| DDU formulas and the season rule | temperatures with a known answer; a second implementation written from the definitions in the article | `tests/test_ddu_formulas.py`, `tests/test_season_rule.py`, `tests/test_known_answer_seasons.py`, `tests/test_independent_reimplementation.py` |
| Chain from the downloads to the seasons | monthly files in the form delivered by the CDS through steps 2–5 | `tests/test_from_raw_months.py` |
| Hourly files and per-cell seasons of the published run | step 3b and step 5 of this repository run on the original hourly files; every cell, year, method and column compared | `tools/verify_release.py`; record in `results/run_20260908_131817_6a83be93/VERIFICATION.txt` |

The last comparison needs the hourly files (about 4 GB), which are not in the repository; its outcome is the
`VERIFICATION.txt` file archived next to the per-cell values. That file lists the seven stages, the software
versions used and whether the verification was complete.

## Checking

1. `pytest -m "not slow"`: about one minute, no download.
2. `pytest -m slow`: about four minutes, no download. `tests/test_from_raw_months.py` needs the geospatial
   libraries of `environment.yml` and is skipped without them.
3. From the archived per-cell values:
   `python scripts/step09_manuscript_statistics.py --season-csv results/run_20260908_131817_6a83be93/season_ALL_YEARS_methods.csv.gz --out out/pack --no-excel`,
   then `python tools/check_against_manuscript.py --pack out/pack`. The script prints every number quoted in the
   manuscript next to the recomputed value, compares all tables with the published run, and ends with
   `RESULT: all checks passed` or the list of differences.
4. With the hourly files of a run: `python tools/verify_release.py --data-root <folder that contains out/>`.
   Stages: (1) per-cell file found; (2) its season lengths are those archived; (3) no hour is missing and the
   files in `out/06_filled_yearly` are what step 3b writes from `out/04_yearly_raw`; (4) step 5 gives the same
   per-cell file; (5) steps 6–10 run; (6) numbers of the article and tables; (7) per-cell file, summary,
   completeness report and `VERIFICATION.txt` written to `results/`. Nothing is written into the data folder; the
   work folder is `out/_verify_release`. A stage whose input is absent is reported as skipped and the result is
   marked incomplete.

## Another region or period

- Boundary: set the environment variable `DDU_SHP_RO` to a polygon file readable by GeoPandas (any coordinate
  system with a `.prj`).
- Temperature request: change `BBOX` (north, west, south, east) and, if needed, `YEARS` in `config.py`; `--years`
  selects the period on the command line. Set `YEAR_WRAP_START` to the second year of the period.
- Suitability classes are computed from the data (`step07`); `--breaks 81,108,124,135` applies the limits of the
  article instead. The classes are relative to the area analysed and are not transferable between regions.
- `tools/check_against_manuscript.py --only-tables` and `tools/verify_release.py --tables-only` compare tables
  without the numbers of the article, which apply to Romania 2005–2024 only.

## Archiving a release

1. `python tools/verify_release.py --data-root <folder of the published run>` and confirm
   `VERIFICATION PASSED - every stage passed`.
2. Commit the files it wrote to `results/`.
3. `python tools/release_checklist.py` and confirm `READY`.
4. Publish a GitHub release (for example `v1.0.0`). With the Zenodo GitHub integration switched on for this
   repository, Zenodo archives the release and assigns it a version DOI; the concept DOI 10.5281/zenodo.18843229
   always resolves to the latest version. When a repository has both `.zenodo.json` and `CITATION.cff`, Zenodo
   takes the metadata of the record from `.zenodo.json` only.
