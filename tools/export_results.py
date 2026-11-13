#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export_results.py - put the per-cell output of step05 next to the archived tables in results/.

Copies, for the run named in out/08_season_swrap/win30/csv/META.json (run_id):
  season_ALL_YEARS_methods.csv   -> results/run_<run_id>/season_ALL_YEARS_methods.csv.gz   (gzip, ~1/6 of the size)
  summary_ALL_YEARS_methods.csv  -> results/run_<run_id>/summary_ALL_YEARS_methods.csv
  META.json                      -> results/run_<run_id>/META_step05.json   (only if not already there)

The archived tables of the published run are not touched. Run tools/check_against_manuscript.py first.

Usage
  python tools/export_results.py                  # data root = DDU_ROOT or this repository
  python tools/export_results.py --root "E:/PROIECT DDU/BACK UP/DDU_PROJECT_UPDATED"
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Copy the step05 per-cell output into results/")
    ap.add_argument("--root", default=os.environ.get("DDU_ROOT", str(REPO)), help="data root (parent of out/)")
    a = ap.parse_args(argv)

    src = Path(a.root).expanduser().resolve() / "out" / "08_season_swrap" / "win30" / "csv"
    season = src / "season_ALL_YEARS_methods.csv"
    if not season.exists():
        print(f"not found: {season}")
        return 2
    meta_p = src / "META.json"
    run_id = json.loads(meta_p.read_text(encoding="utf-8")).get("run_id") if meta_p.exists() else None
    if not run_id:
        print(f"run_id not found in {meta_p}")
        return 2

    dst = REPO / "results" / f"run_{run_id}"
    dst.mkdir(parents=True, exist_ok=True)
    out_gz = dst / "season_ALL_YEARS_methods.csv.gz"
    with open(season, "rb") as fi, open(out_gz, "wb") as fraw, \
            gzip.GzipFile(filename="season_ALL_YEARS_methods.csv", mode="wb", fileobj=fraw, mtime=0) as fo:
        shutil.copyfileobj(fi, fo)
    print(f"[export] {out_gz}  ({out_gz.stat().st_size / 1e6:.1f} MB)")
    summ = src / "summary_ALL_YEARS_methods.csv"
    if summ.exists():
        shutil.copy2(summ, dst / summ.name)
        print(f"[export] {dst / summ.name}")
    if not (dst / "META_step05.json").exists():
        shutil.copy2(meta_p, dst / "META_step05.json")
        print(f"[export] {dst / 'META_step05.json'}")
    print("[export] done - review with 'git status', then commit the results/ folder")
    return 0


if __name__ == "__main__":
    sys.exit(main())
