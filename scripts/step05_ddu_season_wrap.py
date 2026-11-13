#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
step05_ddu_season_wrap.py — RO-only, year-wrap, 4 methods — CSV + Excel (FINAL, repo/Git-ready)

What this does
--------------
For each YEAR × window_days × method, compute per-pixel transmission season metrics from
filled hourly temperature NetCDFs, using DDU (Tbase=14°C) and a rolling EIP window
(default 30d; also supports 15d).

Outputs (per window):
  <ROOT>/out/<outstep>/win{W}/csv/
    {YYYY}_season_{VAR}.csv
    {YYYY}_summary_{METHOD}.csv
    season_ALL_YEARS_methods.csv
    summary_ALL_YEARS_methods.csv
    RUN_LOG.csv
    META.json
    season_swrap_win{W}.xlsx  (optional --excel)

NEW: --final-only
- Writes ONLY the 2 final aggregated files per selected window(s):
    season_ALL_YEARS_methods.csv
    summary_ALL_YEARS_methods.csv
- No per-year/per-method files, no RUN_LOG.csv, no META.json, no Excel.
- If existing per-year season CSVs are present, it can reuse them (fast).

Repo portability
----------------
- Auto-detects repo ROOT by walking upward until finding config.py or .git.
- Sets DDU_ROOT automatically and fixes sys.path so `from config import cfg` works
  regardless of current working directory.
- No hardcoded C:\ paths.

Inputs (expected from previous steps)
-------------------------------------
cfg.DIR_FILLED                 : <ROOT>/out/06_filled_yearly/{YYYY}.nc
cfg.DIR_FILLED_TAILS           : <ROOT>/out/06_filled_yearly_tails_29d/{YYYY-1}_tail29d.nc  (win30)
cfg.DIR_FILLED_TAILS_14        : <ROOT>/out/06_filled_yearly_tails_14d/{YYYY-1}_tail14d.nc  (win15)

Season definition (manuscript, section 2.3)
--------------------------------------------
- Daily DDU by four methods (Table 1): min-max average ("midpoint"), hourly average
  ("daily_mean"), hourly sum ("hourly_sum", degree-hours) and Baskerville-Emin sine ("be_sine").
- A day is transmission-suitable when the DDU accumulated over the 30 days ending on that day
  is >= 130 DDU (daily methods) or >= 3120 degree-hours (hourly sum). Days at or below 14 C add
  nothing but do not reset the window. The first 29 days of a year use the last 29 days of the
  previous year (tails from step04; used from YEAR_WRAP_START = 2006 onward).
- 3-day rule: a season is a run of suitable days that ends when three or more consecutive days
  are unsuitable; gaps of one or two unsuitable days stay inside the season. In split_runs() a new
  run starts when the next suitable day is >= MIN_GAP_DAYS days after the previous one, so
  MIN_GAP_DAYS = 4 (config.py, default) bridges at most 2 unsuitable days. The longest run of the
  year is the season; season_length = number of days from the first to the last day of the run,
  both included (bridged days and, in leap years, 29 February count); total_eip_days counts every
  suitable day of the year, also outside the season.
- Day of year (DOY) is numbered on a 365-day basis: in leap years 29 February shares day 59 with
  28 February, so a given DOY is the same calendar date in every year. No day is left out of
  the calculation.

Usage
-----
python scripts/step05_ddu_season_wrap.py --years 2005-2024 --windows 30 --excel
python scripts/step05_ddu_season_wrap.py --years 2024-2024 --windows 30 --final-only
python scripts/step05_ddu_season_wrap.py --min-gap 4        (explicit 3-day rule; same as default)

ENV
---
DDU_ROOT can override repo root; --root also overrides (and sets DDU_ROOT).
"""

from __future__ import annotations

import os
import sys
import argparse
import time
import uuid
import datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


# ------------------------- repo bootstrap -------------------------
def _find_repo_root(start: Path) -> Path:
    """Walk upward from start until config.py or .git is found; fallback to start parent."""
    cur = start.resolve()
    for _ in range(10):
        if (cur / "config.py").exists() or (cur / ".git").exists():
            return cur
        cur = cur.parent
    return start.resolve()


def _preparse_root(argv) -> str | None:
    for i, a in enumerate(argv):
        if a == "--root" and i + 1 < len(argv):
            return argv[i + 1]
    return None


# Determine repo root early so config import works regardless of CWD.
_root_cli = _preparse_root(sys.argv)
if _root_cli:
    repo_root = Path(_root_cli).expanduser().resolve()
else:
    # If this file is under <repo>/scripts/, repo_root is parent of scripts.
    this = Path(__file__).resolve()
    guess = this.parent.parent if this.parent.name.lower() == "scripts" else this.parent
    repo_root = _find_repo_root(guess)

if _root_cli:
    os.environ["DDU_ROOT"] = str(repo_root)          # --root wins over an inherited DDU_ROOT
else:
    os.environ.setdefault("DDU_ROOT", str(repo_root))
_this_py = Path(__file__).resolve()
_code_root = _this_py.parent.parent if _this_py.parent.name.lower() == "scripts" else _this_py.parent
for _p in (str(repo_root), str(_code_root)):   # this repository's config.py / ddu_stats.py come first
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Import config after bootstrap
from config import cfg  # noqa: E402

try:
    from tqdm import tqdm  # type: ignore
except Exception:
    tqdm = lambda x, **k: x  # noqa: E731


# ------------------------- params from cfg -------------------------
TBASE = float(getattr(cfg, "TBASE_C", 14.0))

THRESH_DD_30D = float(getattr(cfg, "THRESH_DD_30D", 130.0))
THRESH_DH_30D = float(getattr(cfg, "THRESH_DH_30D", 3120.0))
THRESH_DD_15D = float(getattr(cfg, "THRESH_DD_15D", THRESH_DD_30D))
THRESH_DH_15D = float(getattr(cfg, "THRESH_DH_15D", THRESH_DH_30D))

MIN_GAP_DAYS = int(getattr(cfg, "MIN_GAP_DAYS", 4))   # 4 = 3-day rule (see docstring); override: --min-gap
USE_YEAR_WRAP = bool(getattr(cfg, "USE_YEAR_WRAP", True))
YEAR_WRAP_START = int(getattr(cfg, "YEAR_WRAP_START", 2006))
VALIDATE_TAIL = bool(getattr(cfg, "VALIDATE_TAIL", True))

DEFAULT_METHODS = ["midpoint", "daily_mean", "hourly_sum", "be_sine"]

ROOT = Path(getattr(cfg, "ROOT", repo_root)).resolve()

# Prefer cfg dirs if present (repo-portable)
DIR_IN_FILLED = Path(getattr(cfg, "DIR_FILLED", ROOT / "out" / "06_filled_yearly"))
DIR_TAILS_29 = Path(getattr(cfg, "DIR_FILLED_TAILS", ROOT / "out" / "06_filled_yearly_tails_29d"))
DIR_TAILS_14 = Path(getattr(cfg, "DIR_FILLED_TAILS_14", ROOT / "out" / "06_filled_yearly_tails_14d"))

RUN_ID = f"{datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

VAR_MAP = {
    "midpoint": "DDU_midpoint",
    "daily_mean": "DDU_daily_mean",
    "hourly_sum": "DDU_hourly_sum",
    "be_sine": "DDU_be_sine",
}

NEEDED_MEDIAN_COLS = {
    "median_start_doy",
    "median_end_doy",
    "median_length_days",
    "median_outside_eip_days",
    "mean_length_days_with_season",
    "median_length_days_with_season",
}

SEASON_COLS = [
    "year", "window_days", "method", "var_name",
    "lat", "lon", "i", "j",
    "season_start", "season_end",
    "season_start_doy", "season_end_doy",
    "season_start_angle", "season_end_angle",
    "start_month", "end_month",
    "season_length",
    "total_eip_days", "runs_count",
    "outside_eip_days", "outside_fraction",
    "preseason_eip_days", "postseason_eip_days",
    "outside_runs_count",
    "preseason_runs_ranges", "postseason_runs_ranges",
    "outside_runs_ranges", "outside_runs_lengths",
    "preseason_runs_ranges_doy", "postseason_runs_ranges_doy", "outside_runs_ranges_doy",
    "first_outside_date", "last_outside_date",
    "first_outside_doy", "last_outside_doy",
]

SUMMARY_COLS = [
    "year", "window_days", "method", "var_name",
    "n_pixels_grid", "n_pixels_inside", "n_rows_file",
    "n_pixels_transmission_runs_ge_1", "pct_pixels_transmission_inside",
    "n_with_season", "pct_with_season",
    "mean_start_doy", "mean_end_doy",
    "median_start_doy", "median_end_doy",
    "mean_length_days", "median_length_days",
    "mean_length_days_with_season", "median_length_days_with_season",
    "pct_multi_runs", "pct_with_extra_eip",
    "mean_outside_eip_days", "median_outside_eip_days",
    "year_wrap_used", "year_wrap_note",
    "created_utc",
]


# ------------------------- common utils -------------------------
def _latlon(da: xr.DataArray):
    dims = list(da.dims)
    lat = next((d for d in dims if d.lower() in ("latitude", "lat", "y")), None)
    lon = next((d for d in dims if d.lower() in ("longitude", "lon", "x")), None)
    if not lat or not lon:
        raise ValueError("Cannot detect lat/lon dimensions.")
    return lat, lon


def daily_agg(t2m: xr.DataArray):
    Tmin = t2m.resample(time="1D", label="left").min()
    Tmax = t2m.resample(time="1D", label="left").max()
    Tavg = t2m.resample(time="1D", label="left").mean()
    return Tmin, Tmax, Tavg


def ddu_midpoint(Tmin, Tmax):
    Tm = (Tmin + Tmax) / 2.0
    return xr.where(Tm > TBASE, Tm - TBASE, 0.0)


def ddu_daily_mean(Tavg):
    return xr.where(Tavg > TBASE, Tavg - TBASE, 0.0)


def ddu_hourly_sum(t2m_hourly):
    pos = xr.where(t2m_hourly > TBASE, t2m_hourly - TBASE, 0.0)
    return pos.resample(time="1D", label="left").sum()


def ddu_be_sine(Tmin, Tmax, Tbase=TBASE):
    Tm = (Tmin + Tmax) / 2.0
    A = (Tmax - Tmin) / 2.0
    DD = xr.zeros_like(Tm, dtype="float32")
    below = Tmax <= Tbase
    above = Tmin >= Tbase
    partial = (~below) & (~above) & (A > 0)

    DD = xr.where(above, (Tm - Tbase).astype("float32"), DD)

    y = xr.where(partial, (Tbase - Tm) / xr.where(A == 0, np.nan, A), 0.0)
    y = xr.where(partial, xr.apply_ufunc(np.clip, y, -1.0, 1.0, dask="allowed"), y)
    th = xr.where(partial, xr.apply_ufunc(np.arcsin, y, dask="allowed"), 0.0)
    part = ((Tm - Tbase) * (np.pi / 2 - th) + A * xr.apply_ufunc(np.cos, th, dask="allowed")) / np.pi
    part = xr.where(part > 0, part, 0.0).astype("float32")

    return xr.where(partial, part, DD)


def build_ddu(method: str, t2m, Tmin, Tmax, Tavg, thr_dd, thr_dh):
    if method == "midpoint":
        return ddu_midpoint(Tmin, Tmax), thr_dd, "DDU_midpoint"
    if method == "daily_mean":
        return ddu_daily_mean(Tavg), thr_dd, "DDU_daily_mean"
    if method == "hourly_sum":
        return ddu_hourly_sum(t2m), thr_dh, "DDU_hourly_sum"
    if method == "be_sine":
        return ddu_be_sine(Tmin, Tmax), thr_dd, "DDU_be_sine"
    raise ValueError(method)


def roll_eip_window(ddu_day, thr, window_days):
    sW = ddu_day.rolling(time=window_days, min_periods=window_days).sum()
    eip = (sW >= thr).astype(np.int8)
    return sW, eip


def split_runs(eip_bool: np.ndarray, min_gap: int):
    """Split suitable days into runs.

    A new run starts when the index difference between consecutive suitable days is >= min_gap,
    i.e. when >= (min_gap - 1) consecutive unsuitable days separate them. With min_gap = 4 a gap of
    1-2 unsuitable days is bridged and a gap of >= 3 days ends the run (3-day rule).
    """
    idx = np.flatnonzero(eip_bool)
    if idx.size == 0:
        return []
    cuts = np.where(np.diff(idx) >= min_gap)[0]
    starts = np.r_[idx[0], idx[cuts + 1]]
    ends = np.r_[idx[cuts], idx[-1]]
    return list(zip(starts, ends))


def doy_leap_safe(d: pd.Timestamp) -> int:
    doy = d.dayofyear
    if d.is_leap_year and doy > 59:
        return doy - 1
    return doy


def to_doy(date_str: str) -> float:
    if not date_str:
        return np.nan
    d = pd.to_datetime(date_str, errors="coerce")
    if pd.isna(d):
        return np.nan
    return float(doy_leap_safe(d))


def _ranges_to_doy_ranges(ranges_str: str) -> str:
    if not ranges_str:
        return ""
    out = []
    for piece in [p for p in ranges_str.split(";") if p.strip()]:
        if "–" in piece:
            a, b = piece.split("–", 1)
        elif "—" in piece:
            a, b = piece.split("—", 1)
        else:
            toks = piece.strip().split()
            if len(toks) >= 2:
                a, b = toks[0], toks[-1]
            else:
                continue
        da = pd.to_datetime(a, errors="coerce")
        db = pd.to_datetime(b, errors="coerce")
        if pd.isna(da) or pd.isna(db):
            continue
        out.append(f"{doy_leap_safe(da)}–{doy_leap_safe(db)}")
    return ";".join(out)


def season_metrics_for_pixel(eip_bool: np.ndarray, t_daily: np.ndarray):
    out = {k: 0 for k in (
        "runs_count", "season_length", "total_eip_days", "outside_eip_days",
        "preseason_eip_days", "postseason_eip_days", "outside_runs_count",
        "preseason_runs", "postseason_runs"
    )}
    out.update({k: "" for k in (
        "season_start", "season_end",
        "preseason_runs_ranges", "postseason_runs_ranges",
        "outside_runs_ranges", "outside_runs_lengths",
        "first_outside_date", "last_outside_date"
    )})

    tot = int(eip_bool.sum())
    if tot == 0:
        return out

    runs = split_runs(eip_bool, MIN_GAP_DAYS)
    out["runs_count"] = len(runs)
    out["total_eip_days"] = tot
    if not runs:
        return out

    lengths = [(e - s + 1) for s, e in runs]
    mi = int(np.argmax(lengths))
    season = runs[mi]

    t0, t1 = pd.Timestamp(t_daily[season[0]]), pd.Timestamp(t_daily[season[1]])
    out["season_start"], out["season_end"] = t0.date().isoformat(), t1.date().isoformat()
    out["season_length"] = (t1 - t0).days + 1

    pre, post = runs[:mi], runs[mi + 1:]
    pre_days = int(sum((e - s + 1) for s, e in pre))
    post_days = int(sum((e - s + 1) for s, e in post))
    out["preseason_eip_days"], out["postseason_eip_days"] = pre_days, post_days
    out["outside_eip_days"] = pre_days + post_days
    out["outside_runs_count"] = len(pre) + len(post)
    out["preseason_runs"], out["postseason_runs"] = len(pre), len(post)

    def _runs_to_ranges(runs_, t_):
        return ";".join(
            f"{pd.Timestamp(t_[s]).date().isoformat()}–{pd.Timestamp(t_[e]).date().isoformat()}"
            for s, e in runs_
        )

    out["preseason_runs_ranges"] = _runs_to_ranges(pre, t_daily)
    out["postseason_runs_ranges"] = _runs_to_ranges(post, t_daily)
    out["outside_runs_ranges"] = ";".join([x for x in (out["preseason_runs_ranges"], out["postseason_runs_ranges"]) if x]).strip(";")
    out["outside_runs_lengths"] = ";".join(str(e - s + 1) for s, e in (pre + post)) if (pre or post) else ""

    out["first_outside_date"] = (
        (pd.Timestamp(t_daily[pre[0][0]]).date().isoformat() if pre else
         (pd.Timestamp(t_daily[post[0][0]]).date().isoformat() if post else ""))
    )
    out["last_outside_date"] = (
        (pd.Timestamp(t_daily[post[-1][1]]).date().isoformat() if post else
         (pd.Timestamp(t_daily[pre[-1][1]]).date().isoformat() if pre else ""))
    )
    return out


def _safe_to_csv(df: pd.DataFrame, out: Path, retries=6, sleep=0.7):
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + f".tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}")
    df.to_csv(tmp, index=False, encoding="utf-8")
    for k in range(retries):
        try:
            os.replace(tmp, out)
            return
        except PermissionError:
            time.sleep(sleep * (k + 1))
    os.replace(tmp, out.with_name(out.stem + f"__{int(time.time())}" + out.suffix))


def _safe_unlink(p: Path, retries=6, sleep=0.7):
    if not p.exists():
        return
    for k in range(retries):
        try:
            p.unlink()
            return
        except PermissionError:
            time.sleep(sleep * (k + 1))
    p.unlink()


def _append_csv_chunk(df: pd.DataFrame, out_path: Path, header: bool, retries=6, sleep=0.7):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if header else "a"
    for k in range(retries):
        try:
            df.to_csv(out_path, index=False, mode=mode, header=header, encoding="utf-8")
            return True
        except PermissionError:
            time.sleep(sleep * (k + 1))
    df.to_csv(out_path, index=False, mode=mode, header=header, encoding="utf-8")
    return True


def _safe_to_excel(sheets: dict, out_xlsx: Path):
    out_xlsx.parent.mkdir(parents=True, exist_ok=True)
    last_err = None
    for eng in ("xlsxwriter", "openpyxl"):
        try:
            with pd.ExcelWriter(out_xlsx, engine=eng) as xlw:
                for name, df in sheets.items():
                    sname = (name or "Sheet")[:31]
                    if isinstance(df, pd.DataFrame):
                        df.to_excel(xlw, sheet_name=sname, index=False)
                    else:
                        pd.DataFrame(df).to_excel(xlw, sheet_name=sname, index=False)
            return True
        except Exception as e:
            last_err = e
    print("⚠ Excel write failed:", last_err)
    return False


def _infer_grid_from_ij(df: pd.DataFrame):
    if ("i" not in df.columns) or ("j" not in df.columns) or df.empty:
        return (np.nan, np.nan, np.nan)
    i = pd.to_numeric(df["i"], errors="coerce")
    j = pd.to_numeric(df["j"], errors="coerce")
    if not np.isfinite(i).any() or not np.isfinite(j).any():
        return (np.nan, np.nan, np.nan)
    Ny = int(i.max() + 1)
    Nx = int(j.max() + 1)
    n_grid = float(Ny * Nx)
    return (Ny, Nx, n_grid)


def _summary_row_from_season_df(df: pd.DataFrame, year: int, window_days: int, method: str, var_name: str,
                               Ny: float = np.nan, Nx: float = np.nan, n_grid: float = np.nan):
    if df is None or df.empty:
        return None

    d = df.copy()
    for c in ("runs_count", "season_length", "season_start_doy", "season_end_doy", "outside_eip_days"):
        if c not in d.columns:
            d[c] = np.nan

    runs = pd.to_numeric(d["runs_count"], errors="coerce").fillna(0)
    length = pd.to_numeric(d["season_length"], errors="coerce").fillna(0)
    start = pd.to_numeric(d["season_start_doy"], errors="coerce")
    end = pd.to_numeric(d["season_end_doy"], errors="coerce")
    outside = pd.to_numeric(d["outside_eip_days"], errors="coerce").fillna(0)

    n_inside = int(len(d))
    n_tx = int((runs >= 1).sum())
    pct_tx_inside = round((n_tx / n_inside * 100.0), 3) if n_inside else 0.0

    has_season = (length > 0)
    n_with_season = int(has_season.sum())
    pct_with_season = round(100.0 * (n_with_season / n_inside), 3) if n_inside else 0.0

    mean_start = float(start.mean(skipna=True)) if np.isfinite(start).any() else np.nan
    mean_end = float(end.mean(skipna=True)) if np.isfinite(end).any() else np.nan
    med_start = float(start.median(skipna=True)) if np.isfinite(start).any() else np.nan
    med_end = float(end.median(skipna=True)) if np.isfinite(end).any() else np.nan

    mean_len = float(length.mean(skipna=True)) if np.isfinite(length).any() else np.nan
    med_len = float(length.median(skipna=True)) if np.isfinite(length).any() else np.nan

    if has_season.any():
        mean_len_ws = float(length[has_season].mean())
        med_len_ws = float(length[has_season].median())
        pct_multi_runs = round(100.0 * (runs[has_season] > 1).mean(), 3)
    else:
        mean_len_ws = np.nan
        med_len_ws = np.nan
        pct_multi_runs = 0.0

    pct_with_extra_eip = round(100.0 * (outside > 0).mean(), 3) if n_inside else 0.0
    mean_outside = float(outside.mean()) if n_inside else np.nan
    med_outside = float(outside.median()) if n_inside else np.nan

    year_wrap_used = int(USE_YEAR_WRAP and (year >= YEAR_WRAP_START))
    year_wrap_note = "used" if year_wrap_used else "off"

    row = {
        "year": year,
        "window_days": window_days,
        "method": method,
        "var_name": var_name,
        "n_pixels_grid": n_grid,
        "n_pixels_inside": n_inside,
        "n_rows_file": n_inside,
        "n_pixels_transmission_runs_ge_1": n_tx,
        "pct_pixels_transmission_inside": pct_tx_inside,
        "n_with_season": n_with_season,
        "pct_with_season": pct_with_season,

        "mean_start_doy": round(mean_start, 3) if np.isfinite(mean_start) else np.nan,
        "mean_end_doy": round(mean_end, 3) if np.isfinite(mean_end) else np.nan,
        "median_start_doy": round(med_start, 3) if np.isfinite(med_start) else np.nan,
        "median_end_doy": round(med_end, 3) if np.isfinite(med_end) else np.nan,

        "mean_length_days": round(mean_len, 2) if np.isfinite(mean_len) else np.nan,
        "median_length_days": round(med_len, 2) if np.isfinite(med_len) else np.nan,
        "mean_length_days_with_season": round(mean_len_ws, 2) if np.isfinite(mean_len_ws) else np.nan,
        "median_length_days_with_season": round(med_len_ws, 2) if np.isfinite(med_len_ws) else np.nan,

        "pct_multi_runs": pct_multi_runs,
        "pct_with_extra_eip": pct_with_extra_eip,
        "mean_outside_eip_days": round(mean_outside, 2) if np.isfinite(mean_outside) else np.nan,
        "median_outside_eip_days": round(med_outside, 2) if np.isfinite(med_outside) else np.nan,

        "year_wrap_used": year_wrap_used,
        "year_wrap_note": year_wrap_note,
        "created_utc": datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds"),
    }
    return row


def _upgrade_per_year_summaries_from_seasons(year: int, window_days: int, methods: list[str], dir_csv: Path) -> pd.DataFrame:
    """Rebuild per-year summary_{method}.csv from existing per-year season_{VAR}.csv (FAST, no NetCDF)."""
    all_summ = []
    for meth in methods:
        if meth not in VAR_MAP:
            continue
        var_name = VAR_MAP[meth]
        season_path = dir_csv / f"{year}_season_{var_name}.csv"
        if not season_path.exists():
            continue
        dfm = pd.read_csv(season_path, low_memory=False)
        dfm["year"] = year
        dfm["window_days"] = window_days
        dfm["method"] = meth
        dfm["var_name"] = var_name
        dfm = dfm.reindex(columns=SEASON_COLS)

        Ny, Nx, n_grid = _infer_grid_from_ij(dfm)
        srow = _summary_row_from_season_df(dfm, year, window_days, meth, var_name, Ny, Nx, n_grid)
        if srow:
            df_s = pd.DataFrame([srow]).reindex(columns=SUMMARY_COLS)
            _safe_to_csv(df_s, dir_csv / f"{year}_summary_{meth}.csv")
            all_summ.append(df_s)

    return pd.concat(all_summ, ignore_index=True) if all_summ else pd.DataFrame(columns=SUMMARY_COLS)


# ------------------------- tails (year-wrap) -------------------------
def _prev_tail_ddu(year: int, method: str, lat: str, lon: str,
                   ref_da: xr.DataArray, first_time: pd.Timestamp,
                   pad_days: int, dir_tails: Path):
    # Tail filenames expected from your tails step:
    #   {YYYY-1}_tail29d.nc or {YYYY-1}_tail14d.nc
    suffix = f"tail{pad_days}d"
    p = dir_tails / f"{year-1}_{suffix}.nc"
    if not p.exists():
        tprev = pd.date_range(end=first_time - pd.Timedelta(days=1), periods=pad_days, freq="D")
        empty = xr.DataArray(
            np.zeros((pad_days, ref_da.sizes[lat], ref_da.sizes[lon]), dtype=float),
            coords={"time": tprev, lat: ref_da[lat].values, lon: ref_da[lon].values},
            dims=("time", lat, lon), name="DDU_tail"
        )
        return empty, False, "tail_missing"

    with xr.open_dataset(p, decode_timedelta=True) as dth:
        t2m_prev = dth["t2m"]
        Tmin_p, Tmax_p, Tavg_p = daily_agg(t2m_prev)
        if method == "midpoint":
            ddu_p = ddu_midpoint(Tmin_p, Tmax_p)
        elif method == "daily_mean":
            ddu_p = ddu_daily_mean(Tavg_p)
        elif method == "hourly_sum":
            ddu_p = ddu_hourly_sum(t2m_prev)
        elif method == "be_sine":
            ddu_p = ddu_be_sine(Tmin_p, Tmax_p)
        else:
            raise ValueError(method)

        if VALIDATE_TAIL:
            n = ddu_p.sizes["time"]
            if n < pad_days:
                return ddu_p, False, f"tail_too_short={n}"
            if n > pad_days:
                ddu_p = ddu_p.isel(time=slice(n - pad_days, n))
        return ddu_p, True, "ok"


# ------------------------- settings stamp (safe reuse of per-year files) -------------------------
def _settings_stamp(window_days: int, thr_dd: float, thr_dh: float) -> dict:
    return {"tbase_c": float(TBASE), "window_days": int(window_days), "thresh_dd_window": float(thr_dd),
            "thresh_dh_window": float(thr_dh), "min_gap_days": int(MIN_GAP_DAYS),
            "use_year_wrap": bool(USE_YEAR_WRAP), "year_wrap_start": int(YEAR_WRAP_START)}


def _stamp_path(dir_csv: Path, year: int) -> Path:
    return dir_csv / f"{year}_settings.json"


def _stamp_matches(dir_csv: Path, year: int, stamp: dict) -> bool:
    """Per-year files are reused only if they were written with exactly the current settings."""
    sp = _stamp_path(dir_csv, year)
    if not sp.exists():
        return False
    try:
        old = json.loads(sp.read_text(encoding="utf-8"))
    except Exception:
        return False
    return all(old.get(k) == v for k, v in stamp.items())


def _window_thresholds(window_days: int):
    """(DDU threshold, degree-hour threshold) of a window; 30 d: 130 DDU / 3120 degree-hours."""
    if window_days == 30:
        return THRESH_DD_30D, THRESH_DH_30D
    if window_days == 15:
        return THRESH_DD_15D, THRESH_DH_15D
    return THRESH_DD_30D * (window_days / 30.0), THRESH_DH_30D * (window_days / 30.0)


# ------------------------- processing core -------------------------
def process_year_window(year: int, methods, window_days: int, outstep: str,
                        save_parquet=False, force=False, final_only=False):
    # thresholds + tails per window
    if window_days == 30:
        thr_dd, thr_dh = THRESH_DD_30D, THRESH_DH_30D
        pad_days, dir_tails = 29, DIR_TAILS_29
    elif window_days == 15:
        thr_dd, thr_dh = THRESH_DD_15D, THRESH_DH_15D
        pad_days, dir_tails = 14, DIR_TAILS_14
    else:
        thr_dd = THRESH_DD_30D * (window_days / 30.0)
        thr_dh = THRESH_DH_30D * (window_days / 30.0)
        pad_days = window_days - 1
        dir_tails = ROOT / "out" / f"06_filled_yearly_tails_{pad_days}d"

    dir_root = ROOT / "out" / outstep / f"win{window_days}"
    dir_csv = dir_root / "csv"
    dir_csv.mkdir(parents=True, exist_ok=True)

    # Never reuse per-year files computed with other settings (e.g. the old 10-day gap rule).
    stamp = _settings_stamp(window_days, thr_dd, thr_dh)
    if not force and not _stamp_matches(dir_csv, year, stamp):
        if any(dir_csv.glob(f"{year}_season_*.csv")):
            print(f"[!] {year} (win{window_days}): existing per-year files have no/other settings stamp "
                  f"-> recomputing with MIN_GAP_DAYS={MIN_GAP_DAYS}")
        force = True

    # ---- FINAL-ONLY FAST REUSE (season CSV exists) ----
    if final_only and (not force):
        exp_rows = [dir_csv / f"{year}_season_{VAR_MAP[m]}.csv" for m in methods if m in VAR_MAP]
        if exp_rows and all(p.exists() for p in exp_rows):
            all_rows = []
            all_summ_rows = []
            for meth in methods:
                if meth not in VAR_MAP:
                    continue
                var_name = VAR_MAP[meth]
                season_path = dir_csv / f"{year}_season_{var_name}.csv"
                if not season_path.exists():
                    continue
                dfm = pd.read_csv(season_path, low_memory=False)
                dfm["year"] = year
                dfm["window_days"] = window_days
                dfm["method"] = meth
                dfm["var_name"] = var_name
                dfm = dfm.reindex(columns=SEASON_COLS)
                all_rows.append(dfm)

                Ny, Nx, n_grid = _infer_grid_from_ij(dfm)
                srow = _summary_row_from_season_df(dfm, year, window_days, meth, var_name, Ny, Nx, n_grid)
                if srow:
                    all_summ_rows.append(pd.DataFrame([srow]).reindex(columns=SUMMARY_COLS))

            df_year_all_rows = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame(columns=SEASON_COLS)
            df_year_all_summ = pd.concat(all_summ_rows, ignore_index=True) if all_summ_rows else pd.DataFrame(columns=SUMMARY_COLS)

            print(f"↪ final-only reuse {year} (win{window_days}): using existing per-year season CSVs")
            return dir_csv, df_year_all_rows, df_year_all_summ

    # ---- NORMAL SKIP PATH (requires per-year season+summary; auto-upgrade medians if needed) ----
    if (not final_only) and (not force):
        exp_rows = [dir_csv / f"{year}_season_{VAR_MAP[m]}.csv" for m in methods if m in VAR_MAP]
        exp_summ = [dir_csv / f"{year}_summary_{m}.csv" for m in methods if m in VAR_MAP]
        if exp_rows and all(p.exists() for p in exp_rows) and exp_summ and all(p.exists() for p in exp_summ):
            try:
                df_summ_probe = pd.concat([pd.read_csv(p) for p in exp_summ], ignore_index=True)
                missing_meds = not NEEDED_MEDIAN_COLS.issubset(set(df_summ_probe.columns))
                if missing_meds:
                    print(f"↪ upgrade summaries {year} (win{window_days}): recomputing medians from season CSVs")
                    df_summ_probe = _upgrade_per_year_summaries_from_seasons(year, window_days, methods, dir_csv)

                df_year_all_rows = pd.concat([pd.read_csv(p, low_memory=False) for p in exp_rows], ignore_index=True)
                print(f"↪ skip compute {year} (win{window_days}): per-year files already present")
                return dir_csv, df_year_all_rows.reindex(columns=SEASON_COLS), df_summ_probe.reindex(columns=SUMMARY_COLS)
            except Exception:
                pass

    # ---- FULL COMPUTE PATH ----
    src = DIR_IN_FILLED / f"{year}.nc"
    if not src.exists():
        print(f"[!] missing: {src}")
        return dir_csv, pd.DataFrame(columns=SEASON_COLS), pd.DataFrame(columns=SUMMARY_COLS)

    _safe_unlink(_stamp_path(dir_csv, year))   # a crash mid-year must not leave a valid stamp behind
    ds = xr.open_dataset(src, decode_timedelta=True)
    t2m = ds["t2m"]
    lat_name, lon_name = _latlon(t2m)

    inside = (~t2m.isnull()).any(dim="time")
    Ny, Nx = int(t2m.sizes[lat_name]), int(t2m.sizes[lon_name])
    n_inside = int(inside.sum().item())

    Tmin, Tmax, Tavg = daily_agg(t2m)
    t_daily = pd.to_datetime(Tavg["time"].values)

    year_rows, year_summ = [], []

    for meth in methods:
        ddu_day, thr, var_name = build_ddu(meth, t2m, Tmin, Tmax, Tavg, thr_dd, thr_dh)
        ddu_day = ddu_day.where(inside)

        if USE_YEAR_WRAP and (year >= YEAR_WRAP_START):
            tail, ok, wrap_note = _prev_tail_ddu(
                year, meth, lat_name, lon_name, ddu_day, t_daily[0],
                pad_days=pad_days, dir_tails=dir_tails
            )
            if ok:
                ext = xr.concat([tail, ddu_day], dim="time")
                _, eip_e = roll_eip_window(ext, thr, window_days)
                eip = eip_e.isel(time=slice(pad_days, None))
            else:
                if meth == methods[0]:
                    print(f"[!] {year} (win{window_days}): tail of {year - 1} not usable ({wrap_note}) in {dir_tails} "
                          f"-> no year wrap: the first {pad_days} days of {year} cannot be suitable. Run step04 first.")
                _, eip = roll_eip_window(ddu_day, thr, window_days)
        else:
            _, eip = roll_eip_window(ddu_day, thr, window_days)

        eip_np = eip.values
        lats, lons = ddu_day[lat_name].values, ddu_day[lon_name].values

        rows = []
        ii, jj = np.where(inside.values)
        for i, j in tqdm(list(zip(ii, jj)), desc=f"{year} win{window_days} {var_name}"):
            m = season_metrics_for_pixel(eip_np[:, i, j].astype(bool), t_daily)

            sd = to_doy(m["season_start"])
            ed = to_doy(m["season_end"])

            ang_start = float(2.0 * np.pi * (sd - 1.0) / 365.0) if np.isfinite(sd) else np.nan
            ang_end = float(2.0 * np.pi * (ed - 1.0) / 365.0) if np.isfinite(ed) else np.nan

            outside_frac = (m["outside_eip_days"] / m["total_eip_days"]) if m["total_eip_days"] > 0 else np.nan
            start_month = int(pd.to_datetime(m["season_start"]).month) if m["season_start"] else np.nan
            end_month = int(pd.to_datetime(m["season_end"]).month) if m["season_end"] else np.nan

            rows.append({
                "year": year, "window_days": window_days, "method": meth, "var_name": var_name,
                "lat": float(lats[i]), "lon": float(lons[j]), "i": int(i), "j": int(j),
                "season_start": m["season_start"], "season_end": m["season_end"],
                "season_start_doy": float(sd) if np.isfinite(sd) else np.nan,
                "season_end_doy": float(ed) if np.isfinite(ed) else np.nan,
                "season_start_angle": float(ang_start), "season_end_angle": float(ang_end),
                "start_month": start_month, "end_month": end_month,
                "season_length": m["season_length"],
                "total_eip_days": m["total_eip_days"], "runs_count": m["runs_count"],
                "outside_eip_days": m["outside_eip_days"], "outside_fraction": float(outside_frac),
                "preseason_eip_days": m["preseason_eip_days"], "postseason_eip_days": m["postseason_eip_days"],
                "outside_runs_count": m["outside_runs_count"],
                "preseason_runs_ranges": m["preseason_runs_ranges"],
                "postseason_runs_ranges": m["postseason_runs_ranges"],
                "outside_runs_ranges": m["outside_runs_ranges"],
                "outside_runs_lengths": m["outside_runs_lengths"],
                "preseason_runs_ranges_doy": _ranges_to_doy_ranges(m["preseason_runs_ranges"]),
                "postseason_runs_ranges_doy": _ranges_to_doy_ranges(m["postseason_runs_ranges"]),
                "outside_runs_ranges_doy": _ranges_to_doy_ranges(m["outside_runs_ranges"]),
                "first_outside_date": m["first_outside_date"], "last_outside_date": m["last_outside_date"],
                "first_outside_doy": float(to_doy(m["first_outside_date"])) if m["first_outside_date"] else np.nan,
                "last_outside_doy": float(to_doy(m["last_outside_date"])) if m["last_outside_date"] else np.nan,
            })

        df = pd.DataFrame(rows).reindex(columns=SEASON_COLS)

        if not final_only:
            csv_path = dir_csv / f"{year}_season_{var_name}.csv"
            _safe_to_csv(df, csv_path)
            if save_parquet:
                try:
                    df.to_parquet(csv_path.with_suffix(".parquet"), index=False)
                except Exception:
                    pass

        # summary row from df (consistent)
        Ny2, Nx2, n_grid = _infer_grid_from_ij(df)
        srow = _summary_row_from_season_df(df, year, window_days, meth, var_name, Ny2, Nx2, n_grid)
        df_summ = pd.DataFrame([srow]).reindex(columns=SUMMARY_COLS) if srow else pd.DataFrame(columns=SUMMARY_COLS)

        if not final_only and not df_summ.empty:
            _safe_to_csv(df_summ, dir_csv / f"{year}_summary_{meth}.csv")

        year_rows.append(df)
        year_summ.append(df_summ)

    ds.close()

    if not final_only:
        _stamp_path(dir_csv, year).write_text(json.dumps(stamp, indent=2), encoding="utf-8")
        meta = {
            "run_id": RUN_ID,
            "years": None,
            "methods": methods,
            "tbase_c": float(TBASE),
            "window_days": window_days,
            "thresh_dd_window": float(thr_dd),
            "thresh_dh_window": float(thr_dh),
            "min_gap_days": int(MIN_GAP_DAYS),
            "bridged_unsuitable_days_max": int(MIN_GAP_DAYS) - 2,
            "season_ends_after_consecutive_unsuitable_days": int(MIN_GAP_DAYS) - 1,
            "season_definition": ("longest run of transmission-suitable days after bridging; start/end = first/last "
                                  "day of that run; season_length = end - start + 1 (bridged days included); "
                                  "total_eip_days = all suitable days of the year"),
            "doy_convention": "365-day basis: in leap years 29 February shares day 59; no day is left out of the calculation",
            "use_year_wrap": bool(USE_YEAR_WRAP),
            "year_wrap_start": int(YEAR_WRAP_START),
            "dirs": {
                "root": str(ROOT),
                "in_filled": str(DIR_IN_FILLED),
                "in_tails": str(dir_tails),
                "out_csv": str(dir_csv),
            },
            "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds"),
        }
        with open(dir_csv / "META.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

    df_year_all_rows = pd.concat(year_rows, ignore_index=True) if year_rows else pd.DataFrame(columns=SEASON_COLS)
    df_year_all_summ = pd.concat(year_summ, ignore_index=True) if year_summ else pd.DataFrame(columns=SUMMARY_COLS)

    return dir_csv, df_year_all_rows, df_year_all_summ


def _write_excel_summary(win_dir_csv: Path, window_days: int, years, methods, run_log_win):
    summ_path = win_dir_csv / "summary_ALL_YEARS_methods.csv"
    if not summ_path.exists():
        print(f"[EXCEL win{window_days}] skip: missing {summ_path.name}")
        return

    df = pd.read_csv(summ_path)
    xlsx_path = win_dir_csv / f"season_swrap_win{window_days}.xlsx"

    def _safe_pivot(val):
        return (df.pivot_table(index="year", columns="method", values=val, aggfunc="mean")
                .sort_index().reset_index())

    piv = {}
    pivot_cols = [
        "pct_pixels_transmission_inside",
        "pct_with_season",
        "mean_start_doy", "median_start_doy",
        "mean_end_doy", "median_end_doy",
        "mean_length_days", "median_length_days",
        "mean_length_days_with_season", "median_length_days_with_season",
        "mean_outside_eip_days", "median_outside_eip_days",
    ]
    for col in pivot_cols:
        if col in df.columns:
            piv[col] = _safe_pivot(col)

    def _num(x): return pd.to_numeric(x, errors="coerce")

    df_stats = df.copy()
    numeric_cols = [c for c in pivot_cols if c in df_stats.columns]
    for c in numeric_cols:
        df_stats[c] = _num(df_stats[c])

    agg_spec = {c: ["mean", "std"] for c in numeric_cols}
    method_stats = (df_stats.groupby("method", as_index=False).agg(agg_spec)) if agg_spec else pd.DataFrame()
    if not method_stats.empty:
        method_stats.columns = ["_".join([c for c in col if c]).strip("_") for col in method_stats.columns.values]
        method_stats = method_stats.round(3)

    meta_df = pd.DataFrame([{
        "run_id": RUN_ID,
        "root": str(ROOT),
        "in_filled": str(DIR_IN_FILLED),
        "tails_29": str(DIR_TAILS_29),
        "tails_14": str(DIR_TAILS_14),
        "years": f"{years[0]}–{years[-1]}",
        "methods": ",".join(methods),
        "window_days": window_days,
        "tbase_c": TBASE,
        "thresh_dd_30d": THRESH_DD_30D,
        "thresh_dh_30d": THRESH_DH_30D,
        "thresh_dd_15d": THRESH_DD_15D,
        "thresh_dh_15d": THRESH_DH_15D,
        "min_gap_days": MIN_GAP_DAYS,
        "bridged_unsuitable_days_max": MIN_GAP_DAYS - 2,
        "season_ends_after_consecutive_unsuitable_days": MIN_GAP_DAYS - 1,
        "use_year_wrap": USE_YEAR_WRAP,
        "created_utc": datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds"),
    }])

    sheets = {
        "summary_all_years": df,
        "method_stats": method_stats,
        "run_log": pd.DataFrame(run_log_win),
        "meta": meta_df,
    }
    for name, pv in piv.items():
        sheets[f"pivot_{name}"] = pv

    _safe_to_excel(sheets, xlsx_path)
    print(f"[EXCEL win{window_days}] → {xlsx_path}")


# ------------------------- CLI -------------------------
def _expand_years(s, default_years):
    if not s:
        return list(default_years)
    s = s.replace("–", "-").replace("—", "-").strip()
    if "-" in s:
        a, b = map(int, s.split("-"))
        return list(range(min(a, b), max(a, b) + 1))
    return [int(x) for x in s.split(",") if x.strip()]


def parse_args():
    p = argparse.ArgumentParser(description="DDU→EIP Season Wrap — CSV + Excel; win30 + win15")
    p.add_argument("--root", type=str, default=None, help="override ROOT (also sets DDU_ROOT)")
    p.add_argument("--years", type=str, default=None, help="ex: 2015-2024 or 2018,2020,2022")
    p.add_argument("--methods", type=str, default=None, help="midpoint,daily_mean,hourly_sum,be_sine (default: all)")
    p.add_argument("--windows", type=str, default="30", help="windows (days): 30 (default) or 30,15")
    p.add_argument("--outstep", type=str, default="08_season_swrap", help="subfolder in out/ for results")
    p.add_argument("--parquet", action="store_true", help="also write .parquet next to CSV (ignored in --final-only)")
    p.add_argument("--force", action="store_true", help="recompute years even if per-year CSV exists")
    p.add_argument("--excel", action="store_true", help="write Excel per window (ignored in --final-only)")
    p.add_argument("--final-only", action="store_true",
                   help="write ONLY season_ALL_YEARS_methods.csv + summary_ALL_YEARS_methods.csv (no intermediates)")
    p.add_argument("--min-gap", type=int, default=None,
                   help="override MIN_GAP_DAYS (default from config.py: 4 = 3-day rule; must be >= 2)")
    p.add_argument("--input-root", type=str, default=None,
                   help="read the hourly files from <input-root>/out/06_filled_yearly (and the tails next to it) "
                        "while writing the results under --root; default: the same root")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.root:
        ROOT = Path(args.root).expanduser().resolve()
        os.environ["DDU_ROOT"] = str(ROOT)

    if args.min_gap is not None:
        if args.min_gap < 2:
            raise SystemExit("--min-gap must be >= 2 (2 = no bridging; 4 = 3-day rule)")
        MIN_GAP_DAYS = int(args.min_gap)

    if args.input_root:
        _in = Path(args.input_root).expanduser().resolve() / "out"
        DIR_IN_FILLED = _in / "06_filled_yearly"
        DIR_TAILS_29 = _in / "06_filled_yearly_tails_29d"
        DIR_TAILS_14 = _in / "06_filled_yearly_tails_14d"

    cfg_years = getattr(cfg, "YEARS", [])
    YEARS = _expand_years(args.years, cfg_years)
    METHODS = DEFAULT_METHODS[:] if not args.methods else [m.strip() for m in args.methods.split(",") if m.strip()]
    WINDOWS = [int(w) for w in (args.windows.replace(" ", "").split(",")) if w.strip()]

    if not YEARS:
        raise SystemExit("YEARS missing (use --years 2015-2024 or set YEARS in config.py).")

    print(f"[CFG] ROOT={ROOT}")
    print(f"[CFG] IN_FILLED={DIR_IN_FILLED}")
    print(f"[CFG] TAILS_29={DIR_TAILS_29}   TAILS_14={DIR_TAILS_14} (optional)")
    print(f"[CFG] YEARS={YEARS[0]}–{YEARS[-1]}  Tbase={TBASE}°C  Windows={WINDOWS}")
    print(f"[CFG] Methods={METHODS}  OUTSTEP={args.outstep}  FORCE={'ON' if args.force else 'OFF'}  FINAL_ONLY={'ON' if args.final_only else 'OFF'}")
    print(f"[CFG] MIN_GAP_DAYS={MIN_GAP_DAYS}: season ends after >= {MIN_GAP_DAYS - 1} consecutive unsuitable days "
          f"(bridges <= {MIN_GAP_DAYS - 2})")

    # Pre-flight: every year must be reusable (per-year files with the current settings stamp) or
    # computable (hourly NetCDF present). Otherwise stop here, before an existing
    # season_ALL_YEARS_methods.csv could be deleted (--final-only) or overwritten with fewer years.
    blocked = []
    for window_days in WINDOWS:
        dcsv = ROOT / "out" / args.outstep / f"win{window_days}" / "csv"
        stamp = _settings_stamp(window_days, *_window_thresholds(window_days))
        for y in YEARS:
            reusable = ((not args.force) and _stamp_matches(dcsv, y, stamp)
                        and any(dcsv.glob(f"{y}_season_*.csv")))
            if not reusable and not (DIR_IN_FILLED / f"{y}.nc").exists():
                blocked.append(f"win{window_days}/{y}")
    if USE_YEAR_WRAP and 30 in WINDOWS:
        no_tail = [y for y in YEARS if y >= YEAR_WRAP_START and (DIR_IN_FILLED / f"{y}.nc").exists()
                   and not (DIR_TAILS_29 / f"{y - 1}_tail29d.nc").exists()]
        if no_tail:
            print(f"[!] no tail file of the previous year in {DIR_TAILS_29} for: {no_tail}. These years are computed "
                  f"without the year wrap (no suitable day before 30 January). Run step04 to create the tails.")
    if blocked:
        raise SystemExit("[step05] stopped, nothing written. These years have neither an input file "
                         f"{DIR_IN_FILLED}/<YEAR>.nc nor per-year results with the current settings: "
                         + ", ".join(blocked))

    t0_all = time.time()

    for window_days in WINDOWS:
        win_dir_csv = ROOT / "out" / args.outstep / f"win{window_days}" / "csv"
        win_dir_csv.mkdir(parents=True, exist_ok=True)

        season_all_path = win_dir_csv / "season_ALL_YEARS_methods.csv"
        summary_all_path = win_dir_csv / "summary_ALL_YEARS_methods.csv"

        if args.final_only:
            _safe_unlink(season_all_path)
            _safe_unlink(summary_all_path)

            header_written = False
            summ_frames = []

            for y in YEARS:
                out_csv_dir, df_y_rows, df_y_summ = process_year_window(
                    y, METHODS, window_days, outstep=args.outstep,
                    save_parquet=False, force=args.force, final_only=True
                )

                if isinstance(df_y_rows, pd.DataFrame) and (not df_y_rows.empty):
                    df_chunk = df_y_rows.reindex(columns=SEASON_COLS)
                    _append_csv_chunk(df_chunk, season_all_path, header=(not header_written))
                    header_written = True

                if isinstance(df_y_summ, pd.DataFrame) and (not df_y_summ.empty):
                    summ_frames.append(df_y_summ.reindex(columns=SUMMARY_COLS))

            if summ_frames:
                df_all_summ = pd.concat(summ_frames, ignore_index=True)

                # rounding like normal mode
                cols_3 = [
                    "mean_start_doy", "mean_end_doy",
                    "median_start_doy", "median_end_doy",
                    "pct_with_season", "pct_pixels_transmission_inside",
                    "pct_multi_runs", "pct_with_extra_eip",
                ]
                for c in cols_3:
                    if c in df_all_summ.columns:
                        df_all_summ[c] = pd.to_numeric(df_all_summ[c], errors="coerce").round(3)

                cols_2 = [
                    "mean_length_days", "median_length_days",
                    "mean_length_days_with_season", "median_length_days_with_season",
                    "mean_outside_eip_days", "median_outside_eip_days",
                ]
                for c in cols_2:
                    if c in df_all_summ.columns:
                        df_all_summ[c] = pd.to_numeric(df_all_summ[c], errors="coerce").round(2)

                _safe_to_csv(df_all_summ, summary_all_path)

            print(f"[FINAL-ONLY win{window_days}] wrote:")
            print(f"  - {season_all_path}")
            print(f"  - {summary_all_path}")
            continue

        # -------- NORMAL MODE ----------
        agg_rows_win, agg_summ_win = [], []
        run_log_win = []

        for y in YEARS:
            t0 = time.time()
            try:
                out_csv_dir, df_y_rows, df_y_summ = process_year_window(
                    y, METHODS, window_days, outstep=args.outstep,
                    save_parquet=args.parquet, force=args.force, final_only=False
                )
                if isinstance(df_y_rows, pd.DataFrame) and not df_y_rows.empty:
                    agg_rows_win.append(df_y_rows)
                if isinstance(df_y_summ, pd.DataFrame) and not df_y_summ.empty:
                    agg_summ_win.append(df_y_summ)

                run_log_win.append({
                    "run_id": RUN_ID,
                    "year": y,
                    "window_days": window_days,
                    "methods": ",".join(METHODS),
                    "ok": 1,
                    "duration_sec": round(time.time() - t0, 2),
                    "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds"),
                    "out_csv": str(out_csv_dir),
                })
            except Exception as e:
                print(f"✖ YEAR {y} (win{window_days}): {e}")
                run_log_win.append({
                    "run_id": RUN_ID,
                    "year": y,
                    "window_days": window_days,
                    "methods": ",".join(METHODS),
                    "ok": 0,
                    "error": str(e),
                    "duration_sec": round(time.time() - t0, 2),
                    "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds"),
                })

        if run_log_win:
            _safe_to_csv(pd.DataFrame(run_log_win), win_dir_csv / "RUN_LOG.csv")

        df_all_rows = pd.concat(agg_rows_win, ignore_index=True) if agg_rows_win else pd.DataFrame(columns=SEASON_COLS)
        df_all_summ = pd.concat(agg_summ_win, ignore_index=True) if agg_summ_win else pd.DataFrame(columns=SUMMARY_COLS)

        if not df_all_rows.empty:
            _safe_to_csv(df_all_rows, season_all_path)

        if not df_all_summ.empty:
            cols_3 = [
                "mean_start_doy", "mean_end_doy",
                "median_start_doy", "median_end_doy",
                "pct_with_season", "pct_pixels_transmission_inside",
                "pct_multi_runs", "pct_with_extra_eip",
            ]
            for c in cols_3:
                if c in df_all_summ.columns:
                    df_all_summ[c] = pd.to_numeric(df_all_summ[c], errors="coerce").round(3)

            cols_2 = [
                "mean_length_days", "median_length_days",
                "mean_length_days_with_season", "median_length_days_with_season",
                "mean_outside_eip_days", "median_outside_eip_days",
            ]
            for c in cols_2:
                if c in df_all_summ.columns:
                    df_all_summ[c] = pd.to_numeric(df_all_summ[c], errors="coerce").round(2)

            _safe_to_csv(df_all_summ, summary_all_path)

        if args.excel and (not df_all_summ.empty):
            _write_excel_summary(win_dir_csv, window_days, YEARS, METHODS, run_log_win)

    print(f"\n✔ DONE in {time.time() - t0_all:.1f}s → {ROOT / 'out' / args.outstep}")
