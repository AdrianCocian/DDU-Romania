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
     and reported). A row that differs is recomputed in the single precision of step05 and in double precision;
     it is accepted only when rounding explains it: a day whose 30-day sum lies at the threshold (within
     0.001 %), counted as suitable in one computation and not in the other. Every such row is listed in
     VERIFICATION.txt     [skipped with --skip-step05 or when the hourly files are not in the data folder]
  5  steps 06-10 of this repository run on the per-cell file of the published run
  6  tools/check_against_manuscript.py: the numbers printed in the article and the archived tables are
     reproduced from that file
  7  per-cell file (gzip), summary, completeness report and VERIFICATION.txt written to results/run_<run_id>/

Usage
  python tools/verify_release.py --data-root <folder of the published run>
  python tools/verify_release.py --data-root <folder> --skip-inputs --skip-step05   # statistics only (fast)
  python tools/verify_release.py --data-root <folder> --full                        # steps 06-08 with all maps (slow)

Exit code 0 = "VERIFICATION PASSED"; 1 = a stage failed (nothing is written to results/); 2 = input missing.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import gzip
import itertools
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
SEASON_KEYS = ("season_start_doy", "season_end_doy", "season_length", "total_eip_days")
# A day is "at the threshold" when its 30-day sum, computed in double precision, lies within this fraction of the
# threshold (0.001 %, i.e. 0.03 degree-hours of 3120 or 0.0013 DDU of 130). Single-precision numbers carry about
# seven significant digits and a 30-day sum involves up to several hundred rounded operations, so rounding alone can
# decide such a day. The tolerance is at least three times the largest rounding error of the single-precision sums
# (tests/test_verify_release.py).
TIE_REL_TOL = 1e-5


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


def code_version() -> str:
    """Git commit of this repository, and whether tracked files differ from it."""
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(REPO), capture_output=True, text=True, timeout=60)
        if sha.returncode != 0:
            return "unknown (git did not report a commit)"
        changed = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=str(REPO),
                                 capture_output=True, text=True, timeout=120).stdout.strip()
        return f"git commit {sha.stdout.strip()[:12]}" + (" with local changes to tracked files" if changed
                                                          else ", no local changes to tracked files")
    except Exception:
        return "unknown (git not available)"


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


def _number_text(s) -> str:
    """'2.0' -> '2', '3;5.0' -> '3;5': the same numbers written the same way; other text unchanged."""
    out = []
    for t in str(s).split(";"):
        t = t.strip()
        try:
            f = float(t)
            out.append(str(int(f)) if np.isfinite(f) and f == int(f) else repr(f))
        except ValueError:
            out.append(t)
    return ";".join(out)


def compare_season_files(new_csv: Path, old_csv: Path) -> tuple:
    """Per-cell file computed now against the per-cell file of the published run.

    The verdict rests on the columns the article analyses (CORE: position, onset, end, season length, total EIP
    days) for every cell, year and method. The other columns of the file describe the suitable days outside the
    season; they are compared too and differences are listed, with an example, without changing the verdict.
    Returns (ok, message, notes, info); info["core_diff"] holds the rows that differ in an analysed column (keys and
    both values of each analysed column) when both files have the same rows, otherwise None.
    """
    a = pd.read_csv(new_csv, low_memory=False)
    b = pd.read_csv(old_csv, low_memory=False)
    keys = ["year", "method", "i", "j"]
    common = [c for c in a.columns if c in b.columns and c not in keys and c not in NOT_COMPARED]
    missing = [c for c in CORE if c not in common]
    if missing:
        return False, f"columns missing in one of the files: {missing}", [], {"core_diff": None}
    m = a.merge(b, on=keys, how="outer", suffixes=("", "_pub"), indicator=True)
    only = int((m["_merge"] != "both").sum())
    m = m[m["_merge"] == "both"].reset_index(drop=True)

    def differing(c):
        x, y = m[c], m[c + "_pub"]
        xn, yn = pd.to_numeric(x, errors="coerce"), pd.to_numeric(y, errors="coerce")
        if xn.notna().sum() == x.notna().sum() and yn.notna().sum() == y.notna().sum():      # numeric column
            xv, yv = xn.to_numpy(float), yn.to_numpy(float)
            return ~(np.isclose(xv, yv, rtol=0, atol=CORE.get(c, 1e-9)) | (np.isnan(xv) & np.isnan(yv)))
        return (x.fillna("").astype(str) != y.fillna("").astype(str)).to_numpy()

    bad_core, bad_other = [], []
    core_rows = np.zeros(len(m), dtype=bool)
    for c in common:
        neq = differing(c)
        if not neq.any():
            continue
        first = {k: (v.item() if hasattr(v, "item") else v) for k, v in m.loc[neq, keys].iloc[0].items()}
        if c in CORE:
            core_rows |= neq
            bad_core.append(f"{c}: {int(neq.sum())} rows differ (first: {first})")
            continue
        new, pub = m.loc[neq, c].fillna("").astype(str), m.loc[neq, c + "_pub"].fillna("").astype(str)
        how = (", all only in how the numbers are written"
               if all(_number_text(x) == _number_text(y) for x, y in zip(new, pub)) else "")
        bad_other.append(f"{c}: {int(neq.sum())} rows differ{how} (first: {first}; published '{pub.iloc[0]}', "
                         f"recomputed '{new.iloc[0]}')")
    others = [c for c in common if c not in CORE]
    extra = sorted((set(a.columns) ^ set(b.columns)) - NOT_COMPARED)
    same_rows = only == 0 and len(m) == len(b)
    ok = not bad_core and same_rows
    other_txt = f"{len(others) - len(bad_other)} of {len(others)} other columns identical"
    msg = (f"{len(m)} rows (cell x year x method); rows on one side only: {only}; "
           f"{len(CORE) - len(bad_core)} of {len(CORE)} analysed columns identical ({', '.join(CORE)}); " + other_txt)
    notes = bad_core + [f"other column, not part of the verdict - {t}" for t in bad_other]
    if extra:
        notes.append(f"columns present in one file only (not compared): {extra}")
    core_diff = None
    if bad_core and same_rows:
        core_diff = m.loc[core_rows, keys + [x for c in CORE for x in (c, c + "_pub")]].reset_index(drop=True)
    return ok, msg, notes, {"core_diff": core_diff, "n_rows": len(m), "other": other_txt}


# ------------------------------------------------------------------ rounding at the threshold (stage 4)
def load_step05():
    """scripts/step05_ddu_season_wrap.py as a module: its functions and its settings from config.py.

    Its start-up code reads --root from the command line, sets DDU_ROOT and extends sys.path; all three are
    restored, so the steps run later by this tool are not affected.
    """
    import importlib.util
    argv, root, path = sys.argv[:], os.environ.get("DDU_ROOT"), sys.path[:]
    sys.argv = sys.argv[:1]
    try:
        spec = importlib.util.spec_from_file_location("ddu_step05_for_verify", S / "step05_ddu_season_wrap.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        sys.argv = argv
        sys.path[:] = path
        if root is None:
            os.environ.pop("DDU_ROOT", None)
        else:
            os.environ["DDU_ROOT"] = root
    return mod


def window_sums(s5, year: int, method: str, double: bool) -> dict:
    """30-day sums of one year and method on the whole grid, for the days of that year.

    double=False: exactly as step05 computes them (the precision of the hourly files, float32, and its order of
    operations on the whole grid). double=True: the same steps in float64. The tail of the previous year is used as
    in step05 (s5.DIR_IN_FILLED and s5.DIR_TAILS_29 point to the data folder).
    """
    import xarray as xr
    window, pad = 30, 29
    thr_dd, thr_dh = s5._window_thresholds(window)
    dtype = "float64" if double else None
    with xr.open_dataset(s5.DIR_IN_FILLED / f"{year}.nc", decode_timedelta=True) as ds:
        t2m = ds["t2m"] if dtype is None else ds["t2m"].astype(dtype)
        lat, lon = s5._latlon(t2m)
        inside = (~t2m.isnull()).any(dim="time")
        Tmin, Tmax, Tavg = s5.daily_agg(t2m)
        t_daily = pd.to_datetime(Tavg["time"].values)
        ddu_day, thr, _ = s5.build_ddu(method, t2m, Tmin, Tmax, Tavg, thr_dd, thr_dh, dtype=dtype or "float32")
        ddu_day = ddu_day.where(inside)
        sums = None
        if s5.USE_YEAR_WRAP and year >= s5.YEAR_WRAP_START:
            tail, ok, _ = s5._prev_tail_ddu(year, method, lat, lon, ddu_day, t_daily[0], pad_days=pad,
                                            dir_tails=s5.DIR_TAILS_29, dtype=dtype)
            if ok:
                ext, _ = s5.roll_eip_window(xr.concat([tail, ddu_day], dim="time"), thr, window)
                sums = ext.isel(time=slice(pad, None))
        if sums is None:
            sums, _ = s5.roll_eip_window(ddu_day, thr, window)
        sums = sums.transpose("time", lat, lon).values
    return {"sums": sums, "thr": float(thr), "t_daily": t_daily}


def season_values(s5, flags, t_daily) -> dict:
    """The analysed columns that step05's season rule gives for one cell-year of suitable days."""
    m = s5.season_metrics_for_pixel(np.asarray(flags, dtype=bool), t_daily)
    return {"season_start_doy": s5.to_doy(m["season_start"]), "season_end_doy": s5.to_doy(m["season_end"]),
            "season_length": float(m["season_length"]), "total_eip_days": float(m["total_eip_days"])}


def _same_values(x: dict, y: dict) -> bool:
    def same(p, q):
        p, q = float(p), float(q)
        return (np.isnan(p) and np.isnan(q)) or abs(p - q) <= 1e-9
    return all(same(x[k], y[k]) for k in SEASON_KEYS)


def _season_text(v: dict) -> str:
    def n(x):
        return "none" if np.isnan(float(x)) else f"{float(x):g}"
    return (f"onset {n(v['season_start_doy'])}, end {n(v['season_end_doy'])}, length {n(v['season_length'])}, "
            f"total EIP days {n(v['total_eip_days'])}")


def classify_tie(s32, s64, thr: float, values_of, published: dict, recomputed: dict, day_label=str,
                 rel_tol: float = TIE_REL_TOL, max_days: int = 10) -> tuple:
    """Does rounding at the threshold explain why a published row differs from the recomputed one?

    s32: 30-day sums of the cell-year as step05 computes them (single precision); s64: the same in double
    precision; values_of(flags): the analysed columns that step05's season rule gives for a sequence of suitable
    days. Accepted only when (a) the season rule applied to s32 gives the recomputed row, (b) the two precisions
    disagree only on days whose 30-day sum lies within rel_tol of the threshold, and (c) the season rule gives the
    published row when each such day is counted as suitable or not. Returns (accepted, text).
    """
    no = "not explained by rounding at the threshold - "
    s32, s64 = np.asarray(s32, dtype=float), np.asarray(s64, dtype=float)
    e32, e64 = s32 >= thr, s64 >= thr
    if not _same_values(values_of(e32), recomputed):
        return False, no + ("internal check failed: the season rule applied to the recomputed sums does not give "
                            "the recomputed row")
    near = np.flatnonzero(np.abs(s64 - thr) <= rel_tol * thr)
    off = np.setdiff1d(np.flatnonzero(e32 != e64), near)
    if off.size:
        k = int(off[0])
        return False, no + (f"single and double precision disagree on {day_label(k)}, whose 30-day sum "
                            f"({s64[k]:.6f}) is not at the threshold {thr:g}")
    if near.size == 0:
        return False, no + f"no day of the year has a 30-day sum within {100 * rel_tol:g} % of the threshold {thr:g}"
    if near.size > max_days:
        return False, no + f"{near.size} days have a 30-day sum at the threshold, too many to test"
    tried = [tuple(e64[near]), tuple(e32[near])]
    for combo in tried + [c for c in itertools.product((False, True), repeat=near.size) if c not in tried]:
        flags = e64.copy()
        flags[near] = combo
        if not _same_values(values_of(flags), published):
            continue
        days = []
        for k, v in zip(near, combo):
            k = int(k)
            like = ("both precisions" if v == e64[k] == e32[k] else "double precision" if v == e64[k]
                    else "single precision" if v == e32[k] else "neither precision of this computer")
            days.append(f"30-day sum ending {day_label(k)} = {s64[k]:.6f} in double precision and {s32[k]:.6f} in "
                        f"single precision (threshold {thr:g}, difference {s64[k] - thr:+.6f}); counted as "
                        f"{'suitable' if v else 'not suitable'} in the published run, as in {like}")
        return True, ("rounding at the threshold - " + "; ".join(days) + f"; published row: {_season_text(published)}; "
                      f"recomputed row: {_season_text(recomputed)}")
    return False, no + (f"the published row ({_season_text(published)}) does not follow from either decision on the "
                        f"{near.size} day(s) at the threshold")


def explain_threshold_ties(core_diff: pd.DataFrame, data: Path, rel_tol: float = TIE_REL_TOL) -> tuple:
    """Recompute every row that differs in an analysed column (single and double precision, whole grid of its year
    and method) and accept it only when classify_tie does. Returns (all accepted, notes)."""
    s5 = load_step05()
    s5.DIR_IN_FILLED = data / "out" / "06_filled_yearly"
    s5.DIR_TAILS_29 = data / "out" / "06_filled_yearly_tails_29d"
    notes = [f"rows that differ in an analysed column: {len(core_diff)}. Each was recomputed in single precision (as "
             f"step05) and in double precision; a row is accepted only if the two precisions disagree on no day other "
             f"than days whose 30-day sum lies within {100 * rel_tol:g} % of the threshold, and the published values "
             f"follow from step05's season rule with each such day counted as suitable or not"]
    ok_all = True
    for (year, method), rows in core_diff.groupby(["year", "method"], sort=True):
        year = int(year)
        single = window_sums(s5, year, method, double=False)
        double = window_sums(s5, year, method, double=True)
        thr, t_daily = single["thr"], single["t_daily"]

        def values_of(flags, t_daily=t_daily):
            return season_values(s5, flags, t_daily)

        def day_label(k, t_daily=t_daily):
            d = pd.Timestamp(t_daily[k])
            return f"{d.date()} (DOY {s5.doy_leap_safe(d)})"

        for r in rows.to_dict("records"):
            i, j = int(r["i"]), int(r["j"])
            where = f"{year} {method} i={i} j={j} ({r['lat_pub']:.4f} N, {r['lon_pub']:.4f} E): "
            if abs(r["lat"] - r["lat_pub"]) > CORE["lat"] or abs(r["lon"] - r["lon_pub"]) > CORE["lon"]:
                ok_all = False
                notes.append(where + "not explained by rounding at the threshold - the position of the cell differs")
                continue
            ok, text = classify_tie(single["sums"][:, i, j], double["sums"][:, i, j], thr, values_of,
                                    {k: r[k + "_pub"] for k in SEASON_KEYS}, {k: r[k] for k in SEASON_KEYS},
                                    day_label, rel_tol)
            ok_all = ok_all and ok
            notes.append(where + text)
    return ok_all, notes


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
    code = code_version()
    print("code: " + code)
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
            ok3, detail, stage4_notes, info = compare_season_files(work_csv, pub_csv)
            for nline in stage4_notes:
                print("     " + nline)
            diff = info["core_diff"]
            if not ok3 and diff is not None:
                print(f"stage 4: {len(diff)} row(s) differ in an analysed column; recomputing them in single and double "
                      f"precision ...", flush=True)
                try:
                    tie_ok, tie_notes = explain_threshold_ties(diff, data)
                except Exception as e:
                    tie_ok, tie_notes = False, [f"rounding check not completed: {type(e).__name__}: {e}"]
                for nline in tie_notes:
                    print("     " + nline)
                stage4_notes += tie_notes
                if tie_ok:
                    ok3, n, k = True, info["n_rows"], len(diff)
                    detail = (f"{n} rows (cell x year x method); rows on one side only: 0; analysed columns "
                              f"({', '.join(CORE)}) identical in {n - k} rows; {k} "
                              + ("row differs" if k == 1 else "rows differ")
                              + f" only by rounding at the threshold (see Stage 4 in detail); {info['other']}")
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
            f"code: {code}",
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
