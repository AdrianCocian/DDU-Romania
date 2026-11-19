#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_release.py - one command that checks, on the computer that holds the data of the published run, that this
code reproduces the article, and then adds the per-cell season file of that run to results/.

Nothing is written into the data folder: every output goes to a work folder inside this repository
(out/_verify_release, ignored by git) and, at the end, to results/run_<run_id>/.

Stages
  1  per-cell file of the published run found (out/08_season_swrap/win30/csv/season_ALL_YEARS_methods.csv)
  2  its season lengths are those archived in results/ (58,240 cell-years x 4 methods)
  3  hourly input: no hour is missing inside the national mask and the files out/06_filled_yearly of the data
     folder are what step 03b of this repository writes from out/04_yearly_raw      [skipped with --skip-inputs
     or when the data folder has no out/04_yearly_raw]
  4  step05 of this repository, run on the hourly files of the data folder, gives the same onset, end, season
     length and total EIP days for every cell, year and method (the other columns of the file are compared
     and reported)        [skipped with --skip-step05 or when the hourly files are not in the data folder]
  5  steps 06-10 of this repository run on the per-cell file of the published run
  6  tools/check_against_manuscript.py: the numbers printed in the article and the archived tables are
     reproduced from that file
  7  per-cell file (gzip), summary, completeness report and VERIFICATION.txt written to results/run_<run_id>/

Usage
  python tools/verify_release.py --data-root "E:/PROIECT DDU/BACK UP/DDU_PROJECT_UPDATED"
  python tools/verify_release.py --data-root <folder> --skip-inputs --skip-step05   # statistics only (fast)
  python tools/verify_release.py --data-root <folder> --full                        # steps 06-08 with all maps (slow)

Exit code 0 = "VERIFICATION PASSED"; 1 = a stage failed (nothing is written to results/); 2 = input missing.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import gzip
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
S = REPO / "scripts"
REFERENCE_RUN = REPO / "results" / "run_20260908_131817_6a83be93"
METHODS = ["midpoint", "daily_mean", "hourly_sum", "be_sine"]
PUBLISHED_VERSIONS = {"python": "3.11.13", "numpy": "1.26.4", "pandas": "3.0.0", "scipy": "1.16.2",
                      "matplotlib": "3.10.6", "diptest": "0.10.0"}
SEASON_REL = Path("out") / "08_season_swrap" / "win30" / "csv"
# columns of the per-cell file that describe the run, not the result
NOT_COMPARED = {"created_utc", "run_id", "timestamp", "timestamp_utc"}
# columns analysed in the article, with the tolerance of the comparison (days: exact; coordinates: 1e-6 degree)
CORE = {"lat": 1e-6, "lon": 1e-6, "season_start_doy": 1e-9, "season_end_doy": 1e-9, "season_length": 1e-9,
        "total_eip_days": 1e-9}


def _utf8_stdout() -> None:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def versions() -> dict:
    out = {"python": platform.python_version()}
    for mod in ("numpy", "pandas", "scipy", "matplotlib", "diptest", "xarray"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:
            out[mod] = "not installed"
    return out


def run(cmd: List[str], log: List[str], env: dict) -> int:
    """Run one pipeline step; its messages go straight to the console so that progress stays visible."""
    print(f"   $ python {' '.join(str(c) for c in cmd[1:])}", flush=True)
    rc = subprocess.call(cmd, cwd=str(REPO), env=env)
    if rc != 0:
        log.append(f"command failed ({rc}): {' '.join(str(c) for c in cmd[1:])}")
    return rc


# ------------------------------------------------------------------ comparisons
def lengths_match_archive(season_csv: Path, reference: Path) -> tuple:
    """Season lengths of the per-cell file against results/.../per_cell_classes_ALL_YEARS.csv."""
    ref = pd.read_csv(reference / "per_cell_classes_ALL_YEARS.csv", usecols=["year", "i", "j"] + METHODS)
    df = pd.read_csv(season_csv, usecols=["year", "i", "j", "method", "season_length"], low_memory=False)
    wide = (df.pivot_table(index=["year", "i", "j"], columns="method", values="season_length", aggfunc="first")
            .reindex(columns=METHODS).fillna(0.0).reset_index())
    m = ref.merge(wide, on=["year", "i", "j"], how="outer", suffixes=("_ref", ""), indicator=True)
    only = int((m["_merge"] != "both").sum())
    both = m[m["_merge"] == "both"]
    diff = {k: int((both[k].to_numpy(float) != both[k + "_ref"].to_numpy(float)).sum()) for k in METHODS}
    ok = only == 0 and sum(diff.values()) == 0 and len(both) == len(ref)
    return ok, f"{len(both)} cell-years compared; rows on one side only: {only}; differing values by method: {diff}"


def compare_season_files(new_csv: Path, old_csv: Path) -> tuple:
    """Per-cell file computed now against the per-cell file of the published run.

    The verdict rests on the columns the article analyses (CORE: position, onset, end, season length, total EIP
    days) for every cell, year and method. The other columns of the file describe the suitable days outside the
    season; they are compared too and differences are listed, without changing the verdict.
    """
    a = pd.read_csv(new_csv, low_memory=False)
    b = pd.read_csv(old_csv, low_memory=False)
    keys = ["year", "method", "i", "j"]
    common = [c for c in a.columns if c in b.columns and c not in keys and c not in NOT_COMPARED]
    missing = [c for c in CORE if c not in common]
    if missing:
        return False, f"columns missing in one of the files: {missing}", []
    m = a.merge(b, on=keys, how="outer", suffixes=("", "_pub"), indicator=True)
    only = int((m["_merge"] != "both").sum())
    m = m[m["_merge"] == "both"]

    def differing(c):
        x, y = m[c], m[c + "_pub"]
        xn, yn = pd.to_numeric(x, errors="coerce"), pd.to_numeric(y, errors="coerce")
        if xn.notna().sum() == x.notna().sum() and yn.notna().sum() == y.notna().sum():      # numeric column
            xv, yv = xn.to_numpy(float), yn.to_numpy(float)
            return ~(np.isclose(xv, yv, rtol=0, atol=CORE.get(c, 1e-9)) | (np.isnan(xv) & np.isnan(yv)))
        return (x.fillna("").astype(str) != y.fillna("").astype(str)).to_numpy()

    bad_core, bad_other = [], []
    for c in common:
        neq = differing(c)
        if neq.any():
            first = m.loc[neq, keys].iloc[0].to_dict()
            (bad_core if c in CORE else bad_other).append(f"{c}: {int(neq.sum())} rows differ (first: {first})")
    others = [c for c in common if c not in CORE]
    extra = sorted((set(a.columns) ^ set(b.columns)) - NOT_COMPARED)
    ok = not bad_core and only == 0 and len(m) == len(b)
    msg = (f"{len(m)} rows (cell x year x method); rows on one side only: {only}; "
           f"{len(CORE) - len(bad_core)} of {len(CORE)} analysed columns identical ({', '.join(CORE)}); "
           f"{len(others) - len(bad_other)} of {len(others)} other columns identical")
    notes = bad_core + [f"other column, not part of the verdict - {t}" for t in bad_other]
    if extra:
        notes.append(f"columns present in one file only (not compared): {extra}")
    return ok, msg, notes


# ------------------------------------------------------------------ main
def main(argv=None) -> int:
    _utf8_stdout()
    ap = argparse.ArgumentParser(description="Check that this code reproduces the published run, then archive its per-cell file")
    ap.add_argument("--data-root", required=True, help="folder of the published run (the parent of its out/ folder)")
    ap.add_argument("--work", default=str(REPO / "out" / "_verify_release"), help="work folder (default: out/_verify_release)")
    ap.add_argument("--years", default="2005-2024")
    ap.add_argument("--reference", default=str(REFERENCE_RUN), help="archived tables of the published run")
    ap.add_argument("--results-dir", default="", help="where to archive (default: results/run_<run_id>)")
    ap.add_argument("--skip-inputs", action="store_true", help="do not check the hourly files against step 03b")
    ap.add_argument("--skip-step05", action="store_true", help="do not recompute the seasons from the hourly files")
    ap.add_argument("--full", action="store_true", help="steps 06-08 with all maps (slow)")
    ap.add_argument("--tables-only", action="store_true",
                    help="stage 5 compares the tables only (for data other than the published run)")
    ap.add_argument("--no-export", action="store_true", help="do not write anything to results/")
    a = ap.parse_args(argv)

    data = Path(a.data_root).expanduser().resolve()
    work = Path(a.work).expanduser().resolve()
    reference = Path(a.reference).expanduser().resolve()
    pub_csv_dir = data / SEASON_REL
    pub_csv = pub_csv_dir / "season_ALL_YEARS_methods.csv"
    py = sys.executable
    env = dict(os.environ, MPLBACKEND="Agg", PYTHONIOENCODING="utf-8", TQDM_DISABLE="1")
    env.pop("DDU_ROOT", None)
    log, stages = [], []

    def stage(n, title, ok, detail=""):
        mark = "PASS" if ok is True else ("SKIP" if ok is None else "FAIL")
        line = f"[{mark}] stage {n}: {title}" + (f" - {detail}" if detail else "")
        print(line, flush=True)
        stages.append(line)
        return ok is not False

    if work == data:
        print("the work folder must not be the data folder (nothing may be written into the data of the published run)")
        return 2

    print("verify_release - " + _dt.datetime.now().strftime("%Y-%m-%d %H:%M"))
    v = versions()
    print("software:  " + ", ".join(f"{k} {x}" for k, x in v.items()))
    differ = {k: (v.get(k), x) for k, x in PUBLISHED_VERSIONS.items() if v.get(k) != x}
    print("published run: " + ", ".join(f"{k} {x}" for k, x in PUBLISHED_VERSIONS.items()))
    print("versions differ from the published run: " + (", ".join(f"{k} {a_} (published {b_})" for k, (a_, b_) in differ.items()) or "none"))
    print(f"data folder: {data}\nwork folder: {work}\n")

    # ---- stage 1
    if not pub_csv.exists():
        stage(1, "per-cell file of the published run", False, f"not found: {pub_csv}")
        print("\nVERIFICATION NOT STARTED: wrong --data-root?")
        return 2
    meta = {}
    if (pub_csv_dir / "META.json").exists():
        meta = json.loads((pub_csv_dir / "META.json").read_text(encoding="utf-8", errors="replace"))
    run_id = str(meta.get("run_id") or "unknown")
    gap = meta.get("min_gap_days")
    ok = stage(1, "per-cell file of the published run", gap in (4, None),
               f"{pub_csv.stat().st_size / 1e6:.1f} MB; run_id {run_id}; min_gap_days {gap}")
    if not ok:
        print("\nVERIFICATION FAILED: the per-cell file was not computed with the 3-day rule (min_gap_days = 4)")
        return 1

    # ---- stage 2
    try:
        ok2, detail = lengths_match_archive(pub_csv, reference)
    except Exception as e:
        ok2, detail = False, f"{type(e).__name__}: {e}"
    if not stage(2, "season lengths equal to the archived published run", ok2, detail):
        print("\nVERIFICATION FAILED: this is not the per-cell file of the published run; nothing written")
        return 1

    work_csv_dir = work / SEASON_REL
    work_csv = work_csv_dir / "season_ALL_YEARS_methods.csv"
    work.mkdir(parents=True, exist_ok=True)

    # ---- stage 3
    y0, y1 = (int(x) for x in a.years.split("-"))
    years = list(range(y0, y1 + 1))
    hourly = data / "out" / "06_filled_yearly"
    yearly = data / "out" / "04_yearly_raw"
    title3 = "hourly input complete and equal to the output of step 03b"
    qc_report: Optional[Path] = None
    ok_in: Optional[bool] = None
    both = [y for y in years if (yearly / f"{y}.nc").exists() and (hourly / f"{y}.nc").exists()]
    if a.skip_inputs:
        stage(3, title3, None, "--skip-inputs")
    elif not both:
        stage(3, title3, None, f"no pair of files out/04_yearly_raw/<year>.nc and out/06_filled_yearly/<year>.nc for {a.years}")
    else:
        print(f"stage 3: checking the hourly files of {len(both)} year(s) (reads every file once) ...", flush=True)
        rep = work / "out" / "07_reports" / "qc_gaps_summary.csv"
        if rep.exists():
            rep.unlink()
        run([py, str(S / "step03b_qc_fill_year.py"), "--root", str(work), "--input-root", str(data),
             "--years", ",".join(str(y) for y in both), "--report-only"], log, env)
        if not rep.exists():
            ok_in, detail = False, "step 03b did not write its report"
        else:
            q = pd.read_csv(rep)
            q = q[q["year"].isin(both)]
            chk = q["filled_file_check"].astype(str)
            missing = int(q["missing_hours_inside_before"].sum())
            same_t = chk.str.startswith("identical") | (chk == "DIFFERENT (t2m_qc)")
            notes = sorted(set(chk[chk != "identical"]))
            ok_in = bool(len(q) == len(both) and same_t.all() and missing == 0)
            detail = (f"{len(q)} of {len(years)} years checked; missing hours inside the mask: {missing}; "
                      f"temperature identical in {int(same_t.sum())} of {len(q)} files"
                      + (f"; notes: {notes}" if notes else ""))
            qc_report = rep if ok_in else None
        if not stage(3, title3, ok_in, detail):
            print("\nVERIFICATION FAILED at stage 3: the hourly files of the data folder are not what this repository "
                  "produces from the yearly files. Nothing written to results/.")
            return 1

    # ---- stage 4
    have_hourly = all((hourly / f"{y}.nc").exists() for y in years)
    ok3: Optional[bool] = None
    stage4_notes: List[str] = []
    if a.skip_step05:
        stage(4, "step05 from the hourly files reproduces the published seasons", None, "--skip-step05")
    elif not have_hourly:
        stage(4, "step05 from the hourly files reproduces the published seasons", None,
              f"hourly files {y0}.nc-{y1}.nc not all present in {hourly}")
    else:
        print(f"stage 4: recomputing the seasons of {a.years} from the hourly files (several minutes) ...", flush=True)
        if work_csv_dir.exists():
            shutil.rmtree(work_csv_dir.parent.parent)          # out/08_season_swrap of the work folder only
        rc = run([py, str(S / "step05_ddu_season_wrap.py"), "--root", str(work), "--input-root", str(data),
                  "--years", a.years, "--windows", "30", "--force"], log, env)
        if rc == 0 and work_csv.exists():
            ok3, detail, stage4_notes = compare_season_files(work_csv, pub_csv)
            for nline in stage4_notes:
                print("     " + nline)
        else:
            ok3, detail = False, "step05 did not finish"
        if not stage(4, "step05 from the hourly files reproduces the published seasons", ok3, detail):
            print("\nVERIFICATION FAILED at stage 4; nothing written to results/")
            return 1
    # stages 5 and 6 always use the per-cell file of the published run itself
    work_csv_dir.mkdir(parents=True, exist_ok=True)
    if work_csv.exists():
        work_csv.replace(work_csv_dir / "season_ALL_YEARS_methods.recomputed.csv")
    shutil.copy2(pub_csv, work_csv)

    # ---- stage 5
    pack = work / "out" / "12_method_compare" / "win30" / "00_MANUSCRIPT_PACK"
    root = ["--root", str(work)]
    fast = [] if a.full else ["--no-maps"]
    cmds = [
        [py, str(S / "step06_ddu_method_compare.py"), *root, "--years", a.years, "--windows", "30", "--no-excel-compact"]
        + ([] if a.full else ["--compare-only"]),
        [py, str(S / "step07_suitability_classes.py"), *root, "--years", a.years, *fast],
        [py, str(S / "step08_per_year_per_pixel.py"), *root, "--years", a.years, "--no-excel", *fast],
        [py, str(S / "step09_manuscript_statistics.py"), "--season-csv", str(work_csv), "--out", str(pack), "--no-excel"],
        [py, str(S / "step10_manuscript_figures.py"), "--pack", str(pack)],
    ]
    print("stage 5: steps 06-10 on the per-cell file of the published run ...", flush=True)
    failed = [Path(c[1]).name for c in cmds if run(c, log, env) != 0]
    if not stage(5, "steps 06-10 run on the per-cell file of the published run", not failed, ("failed: " + ", ".join(failed)) if failed else
                 "step06 comparison, step07 classes, step08 statistics, step09 tables, step10 figures"):
        print("\nVERIFICATION FAILED at stage 5; nothing written to results/")
        return 1

    # ---- stage 6
    cmd = [py, str(HERE / "check_against_manuscript.py"), "--pack", str(pack), "--reference", str(reference)]
    if a.tables_only:
        cmd.append("--only-tables")
    p = subprocess.run(cmd, cwd=str(REPO), env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
    report = p.stdout.strip().splitlines()
    for line in report:
        if "[OK ]" not in line:
            print("   " + line)
    summary = "; ".join(l.strip() for l in report if l.strip().startswith(("Part A:", "Part B:")))
    if not stage(6, "numbers printed in the article and archived tables reproduced", p.returncode == 0, summary):
        print("\nVERIFICATION FAILED at stage 6; nothing written to results/")
        return 1

    gaps = []
    if ok_in is not True:
        gaps.append("stage 3 not run (hourly files not compared with the output of step 03b)")
    if ok3 is not True:
        gaps.append("stage 4 not run (per-cell seasons not recomputed from the hourly files)")

    # ---- stage 7
    if a.no_export:
        stage(7, "archive", None, "--no-export")
    else:
        dst = Path(a.results_dir).expanduser().resolve() if a.results_dir else REPO / "results" / f"run_{run_id}"
        dst.mkdir(parents=True, exist_ok=True)
        out_gz = dst / "season_ALL_YEARS_methods.csv.gz"
        with open(pub_csv, "rb") as fi, open(out_gz, "wb") as fraw, \
                gzip.GzipFile(filename="season_ALL_YEARS_methods.csv", mode="wb", fileobj=fraw, mtime=0) as fo:
            shutil.copyfileobj(fi, fo)
        summ = pub_csv_dir / "summary_ALL_YEARS_methods.csv"
        if summ.exists():
            shutil.copy2(summ, dst / summ.name)
        if qc_report is not None:
            shutil.copy2(qc_report, dst / qc_report.name)
        stage(7, "archive", True, f"{out_gz.name} ({out_gz.stat().st_size / 1e6:.1f} MB) in {dst.name}")
        lines = [
            "Verification of the release against the published run",
            "======================================================",
            f"date: {_dt.datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"system: {platform.system()} {platform.release()}",
            "software: " + ", ".join(f"{k} {x}" for k, x in v.items()),
            "software of the published run: " + ", ".join(f"{k} {x}" for k, x in PUBLISHED_VERSIONS.items()),
            f"published run: run_id {run_id}, years {a.years}",
            "command: python tools/verify_release.py --data-root <folder of the published run>"
            + (" --skip-inputs" if a.skip_inputs else "") + (" --skip-step05" if a.skip_step05 else "")
            + (" --full" if a.full else "") + (" --tables-only" if a.tables_only else ""),
            "",
            *stages,
            "",
            "result: " + ("complete - every stage passed" if not gaps else "INCOMPLETE - " + "; ".join(gaps)),
            "",
            *(["Stage 4 in detail:", *["  " + n for n in stage4_notes], ""] if stage4_notes else []),
            "Stage 6 in detail:",
            *["  " + l for l in report if l.strip().startswith(("Part", "RESULT")) or "[BAD]" in l or "DIFFERENT" in l
              or (l.startswith("  ") and "rows" in l)],
        ]
        text = "\n".join(lines) + "\n"
        for path, label in ((reference, "<archived tables>"), (work, "<work folder>"), (data, "<data folder>"),
                            (REPO, "<repository>"), (Path.home(), "<home>")):   # no local paths in a public file
            text = text.replace(str(path), label)
        (dst / "VERIFICATION.txt").write_text(text, encoding="utf-8")
        print(f"   written: {dst.name}/VERIFICATION.txt")

    if gaps:
        print("\nVERIFICATION PASSED, BUT INCOMPLETE:")
        for g in gaps:
            print("  - " + g)
    else:
        print("\nVERIFICATION PASSED - every stage passed")
    if not a.no_export:
        print("review with 'git status', then commit the results/ folder")
    return 0


if __name__ == "__main__":
    sys.exit(main())
