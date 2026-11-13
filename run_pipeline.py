#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_pipeline.py - run the DDU pipeline (steps 1-10) in order, stopping at the first error.

  1  step01_download_ro.py           ERA5-Land hourly 2 m temperature from the Copernicus CDS (needs ~/.cdsapirc)
  2  step02_std_clip_ro.py           standardise units/coordinates, clip to Romania
  3  step03_concat_year.py           one hourly file per year
  4  step04_tails.py                 last 29 days of each year (input of the 30-day window of the next year)
  5  step05_ddu_season_wrap.py       daily DDU (4 methods), 30-day window, 3-day season rule, per-cell seasons
  6  step06_ddu_method_compare.py    method comparison, consensus (median of the methods), pairwise differences
  7  step07_suitability_classes.py   relative climatic suitability classes (P25/P50/P75/P90 of hourly sum)
  8  step08_per_year_per_pixel.py    per-year counts, national series, trend and variance statistics
  9  step09_manuscript_statistics.py manuscript tables and key numbers (00_MANUSCRIPT_PACK)
 10  step10_manuscript_figures.py    Figures 1-3 (300 dpi PNG + PDF)

Examples
  python run_pipeline.py                         # everything, 2005-2024 (download takes many hours)
  python run_pipeline.py --from 5                # hourly inputs of steps 1-4 already present
  python run_pipeline.py --from 9 --to 10        # tables and figures only (needs the step05 output)
  python run_pipeline.py --skip 6,8 --years 2005-2024
  python run_pipeline.py --dry-run               # print the commands only

Data root: --root or DDU_ROOT (default: this repository). All outputs go to <root>/out/, which git ignores.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List

HERE = Path(__file__).resolve().parent
S = HERE / "scripts"


def _years_span(years: str) -> List[int]:
    s = years.replace("–", "-").strip()
    if "-" in s:
        a, b = (int(x) for x in s.split("-", 1))
        return list(range(min(a, b), max(a, b) + 1))
    return sorted(int(x) for x in s.split(",") if x.strip())


def build_steps(a) -> dict:
    py = sys.executable
    y = a.years
    root = ["--root", str(a.root)]
    yrs = _years_span(y)
    map_years = a.map_years.strip()
    steps = {
        1: [py, str(S / "step01_download_ro.py"), "--years", y],
        2: [py, str(S / "step02_std_clip_ro.py"), "--years", y],
        3: [py, str(S / "step03_concat_year.py"), "--years", y],
        4: [py, str(S / "step04_tails.py"), "--years", y],
        5: [py, str(S / "step05_ddu_season_wrap.py"), *root, "--years", y, "--windows", "30"]
           + (["--excel"] if a.excel else []) + (["--force"] if a.force else []),
        6: [py, str(S / "step06_ddu_method_compare.py"), *root, "--years", y, "--windows", "30"]
           + ([] if a.excel else ["--no-excel-compact"]),
        7: [py, str(S / "step07_suitability_classes.py"), *root, "--years", y]
           + (["--excel"] if a.excel else []) + (["--no-maps"] if a.no_maps else []),
        8: [py, str(S / "step08_per_year_per_pixel.py"), *root, "--years", y]
           + ([] if a.excel else ["--no-excel"]) + (["--no-maps"] if a.no_maps else []),
        9: [py, str(S / "step09_manuscript_statistics.py"),
            "--season-csv", str(a.root / "out" / "08_season_swrap" / "win30" / "csv" / "season_ALL_YEARS_methods.csv"),
            "--out", str(a.root / "out" / "12_method_compare" / "win30" / "00_MANUSCRIPT_PACK")]
           + ([] if a.excel else ["--no-excel"]),
        10: [py, str(S / "step10_manuscript_figures.py"),
             "--pack", str(a.root / "out" / "12_method_compare" / "win30" / "00_MANUSCRIPT_PACK")]
            + (["--years", map_years] if map_years else []),
    }
    if len(yrs) < 3:
        print("[run] note: fewer than 3 years - trend statistics will be skipped")
    return steps


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Run the DDU pipeline, steps 1-10")
    ap.add_argument("--root", default=os.environ.get("DDU_ROOT", str(HERE)),
                    help="data root (parent of out/); default: DDU_ROOT or this repository")
    ap.add_argument("--years", default="2005-2024", help="e.g. 2005-2024 (default) or 2018,2020")
    ap.add_argument("--from", dest="first", type=int, default=1, help="first step (1-10)")
    ap.add_argument("--to", dest="last", type=int, default=10, help="last step (1-10)")
    ap.add_argument("--skip", default="", help="comma list of steps to skip, e.g. 6,8")
    ap.add_argument("--excel", action="store_true", help="also write the Excel workbooks of steps 5-9")
    ap.add_argument("--no-maps", action="store_true", help="steps 7-8 without per-year maps (faster)")
    ap.add_argument("--force", action="store_true", help="step05: recompute years even if per-year files exist")
    ap.add_argument("--map-years", default="", help="step10: four years for the map panels (default 2005,2011,2017,2024)")
    ap.add_argument("--dry-run", action="store_true", help="print the commands without running them")
    a = ap.parse_args(argv)

    a.root = Path(a.root).expanduser().resolve()
    skip = {int(s) for s in a.skip.split(",") if s.strip()}
    todo = [k for k in range(a.first, a.last + 1) if k not in skip]
    if not todo or min(todo) < 1 or max(todo) > 10:
        ap.error("steps must be within 1-10")

    env = dict(os.environ, DDU_ROOT=str(a.root), MPLBACKEND="Agg", PYTHONIOENCODING="utf-8")
    steps = build_steps(a)
    print(f"[run] root={a.root} years={a.years} steps={todo}")
    t_all = time.time()
    for k in todo:
        cmd = steps[k]
        print(f"\n[run] step {k}: {' '.join(cmd)}", flush=True)
        if a.dry_run:
            continue
        t0 = time.time()
        rc = subprocess.call(cmd, cwd=str(HERE), env=env)
        if rc != 0:
            print(f"[run] step {k} failed (exit code {rc}) - stopped")
            return rc
        print(f"[run] step {k} done in {time.time() - t0:.0f} s", flush=True)
    print(f"\n[run] finished in {time.time() - t_all:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
