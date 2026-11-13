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

## Three levels of checking

1. `pytest -m "not slow"`: DDU formulas, the 3-day rule, DOY handling, and the statistics against the archived
   tables of the published run (`results/`), in a few seconds.
2. `pytest -m slow`: steps 5–10 end to end on a small synthetic dataset (no download, about 2 minutes).
3. With the per-cell output of step05 (`out/08_season_swrap/win30/csv/season_ALL_YEARS_methods.csv`):
   `python run_pipeline.py --from 9 --to 10`, then `python tools/check_against_manuscript.py`. The script prints
   every number quoted in the manuscript next to the recomputed value, compares all tables with the published run,
   and ends with `RESULT: all checks passed` or the list of differences.

## Archiving a release

1. Run level 3 above and confirm that all checks pass.
2. `python tools/export_results.py` copies the per-cell season file (gzip) into `results/run_<run_id>/`.
3. Commit, push, and publish a GitHub release (for example `v1.0.0`). With the Zenodo GitHub integration switched
   on for this repository, Zenodo archives the release and assigns it a version DOI; the concept DOI
   10.5281/zenodo.18843229 always resolves to the latest version.
