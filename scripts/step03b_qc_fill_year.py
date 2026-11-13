#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
step03b_qc_fill_year.py — quality control of the yearly files and creation of the
"filled yearly" files read by step04 (tails) and step05 (DDU and seasons).

Inputs (per year Y):
  out/04_yearly_raw/{Y}.nc                 (written by step03)

Outputs:
  out/06_filled_yearly/{Y}.nc              t2m (float32, degC) + t2m_qc (uint8) + coordinates dayofyear, hour
  out/07_reports/qc_gaps_summary.csv       one row per year: missing hours inside the national mask, how they were
                                           treated, and the state of the filled file (column filled_file_check:
                                           "written"; with --report-only "identical", "identical (coordinate ...
                                           differs)", "DIFFERENT (...)" or "absent")

Quality-control flags (variable t2m_qc, CF status_flag):
  0  original               value taken unchanged from ERA5-Land
  1  interpolated_shortgap  linear interpolation in time, gap <= MAX_GAP_HOURS (default 24 h)
  2  climatology_doy_hod    mean of the same day-of-year and hour over the other available years
  3  spatial_interpolation  mean of the valid neighbouring cells (3 x 3 window) at the same hour

Note. A reanalysis is gap-free by construction, so inside the national mask every value is expected to carry
flag 0 and t2m to be identical to out/04_yearly_raw/{Y}.nc. The filling branches make the step complete; the QC
summary records, for every year, how many hours were missing and how they were treated. Cells outside the mask
stay NaN. Use --report-only to check existing files without writing anything.

CLI
---
  --root PATH             data root (parent of out/); default: DDU_ROOT or the repository
  --input-root PATH       read out/04_yearly_raw (and, with --report-only, the existing out/06_filled_yearly) from
                          PATH instead of --root; everything this step writes still goes under --root. Used to check
                          the data of an earlier run without writing anything into its folder.
  --years A-B | Y1,Y2     (default: cfg.YEARS)
  --max-gap-hours N       (default: 24)
  --overwrite             rebuild even if the output exists
  --report-only           write no NetCDF file: report the missing hours of every year and compare the existing
                          out/06_filled_yearly/{Y}.nc with what this step would write (t2m, QC flag, coordinates)
"""

from __future__ import annotations

import os
import sys
import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import xarray as xr

CODE_ROOT = Path(__file__).resolve().parents[1]


def _preparse(argv: List[str], name: str) -> Optional[str]:
    for i, a in enumerate(argv):
        if a == name and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith(name + "="):
            return a.split("=", 1)[1]
    return None


_root_cli = _preparse(sys.argv, "--root")
if _root_cli:
    os.environ["DDU_ROOT"] = str(Path(_root_cli).expanduser().resolve())   # --root wins over an inherited DDU_ROOT
else:
    os.environ.setdefault("DDU_ROOT", str(CODE_ROOT))
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from config import cfg  # noqa: E402

VAR, QC = "t2m", "t2m_qc"
DIR_SRC = Path(cfg.DIR_YEAR)          # yearly files written by step03
DIR_DST = Path(cfg.DIR_FILLED)        # filled yearly files written here
DIR_EXISTING = DIR_DST                # filled files that --report-only compares with
FLAG_VALUES = np.array([0, 1, 2, 3], dtype="uint8")
FLAG_MEANINGS = "original interpolated_shortgap climatology_doy_hod spatial_interpolation"


def _parse_years(s: Optional[str]) -> List[int]:
    if not s:
        return list(cfg.YEARS)
    s = s.replace("–", "-").strip()
    if "-" in s:
        a, b = s.split("-", 1)
        return list(range(int(a), int(b) + 1))
    return [int(t) for t in s.split(",") if t.strip()]


def _latlon_names(ds: xr.Dataset):
    lat = next(d for d in ds[VAR].dims if d.lower() in ("lat", "latitude", "y"))
    lon = next(d for d in ds[VAR].dims if d.lower() in ("lon", "longitude", "x"))
    return lat, lon


def _inside_mask(t2m: np.ndarray) -> np.ndarray:
    """A cell is inside the national mask if it has at least one valid hour in the year."""
    return np.isfinite(t2m).any(axis=0)


def _climatology(year: int, years_available: List[int], dir_year: Path, lat_n: str, lon_n: str) -> Optional[xr.DataArray]:
    """Mean t2m per (dayofyear, hour) over the other available years (29 February dropped)."""
    others = [y for y in years_available if y != year and (dir_year / f"{y}.nc").exists()]
    if not others:
        return None
    acc, n = None, 0
    for y in others:
        ds = xr.open_dataset(dir_year / f"{y}.nc")
        da = ds[VAR]
        t = pd.DatetimeIndex(da["time"].values)
        keep = ~((t.month == 2) & (t.day == 29))
        da = da.isel(time=np.flatnonzero(keep))
        t = t[keep]
        doy = t.dayofyear - ((t.is_leap_year) & (t.dayofyear > 59)).astype(int)
        key = doy * 24 + t.hour
        g = da.assign_coords(key=("time", key)).groupby("key").mean("time")
        acc = g if acc is None else acc + g
        n += 1
        ds.close()
    return acc / n


def _compare_existing(dst: Path, t2m: np.ndarray, qc: np.ndarray, doy: np.ndarray, hour: np.ndarray,
                      lat_n: str, lon_n: str) -> str:
    """Compare an existing filled file with the arrays this step would write.

    'identical'                        temperature, QC flag and the auxiliary coordinates are equal
    'identical (coordinate ... differs)'  temperature and QC flag are equal; dayofyear/hour, which no later step reads,
                                       are labelled differently
    'DIFFERENT (...)'                  temperature or QC flag differ
    """
    diff, aux = [], []
    with xr.open_dataset(dst) as ex:
        if VAR not in ex or ex[VAR].transpose("time", lat_n, lon_n).shape != t2m.shape:
            return "DIFFERENT (shape)"
        a = ex[VAR].transpose("time", lat_n, lon_n).values
        if not (np.array_equal(np.isnan(a), np.isnan(t2m)) and np.array_equal(np.nan_to_num(a), np.nan_to_num(t2m))):
            diff.append(VAR)
        if QC not in ex or not np.array_equal(ex[QC].transpose("time", lat_n, lon_n).values, qc):
            diff.append(QC)
        for name, ref in (("dayofyear", doy), ("hour", hour)):
            if name not in ex.variables or not np.array_equal(ex[name].values, ref):
                aux.append(name)
    if diff:
        return "DIFFERENT (" + ", ".join(diff) + ")"
    return "identical" if not aux else "identical (coordinate " + ", ".join(aux) + " differs)"


def process_year(year: int, years_available: List[int], max_gap_hours: int, overwrite: bool,
                 report_only: bool = False) -> dict:
    src = DIR_SRC / f"{year}.nc"
    dst = (DIR_EXISTING if report_only else DIR_DST) / f"{year}.nc"
    if not src.exists():
        raise FileNotFoundError(f"missing yearly file: {src}")
    if dst.exists() and not overwrite and not report_only:
        print(f"[SKIP] {dst.name} exists (use --overwrite)")
        return {"year": year, "status": "skipped"}

    ds = xr.open_dataset(src)
    lat_n, lon_n = _latlon_names(ds)
    da = ds[VAR].transpose("time", lat_n, lon_n)
    t2m = da.values.astype("float32")
    T, NY, NX = t2m.shape

    inside = _inside_mask(t2m)
    n_inside = int(inside.sum())
    missing = np.isnan(t2m) & inside[None, :, :]
    n_missing0 = int(missing.sum())
    qc = np.zeros_like(t2m, dtype="uint8")

    n1 = n2 = n3 = 0
    if n_missing0:
        # 1) short gaps: linear interpolation in time (gaps of at most max_gap_hours with valid values on both sides)
        f1 = np.zeros_like(missing)
        for iy, ix in zip(*np.where(missing.any(axis=0))):
            series = t2m[:, iy, ix]
            nan = np.isnan(series)
            edges = np.flatnonzero(np.diff(np.r_[False, nan, False]))
            for g0, g1 in zip(edges[::2], edges[1::2]):      # NaN run = [g0, g1)
                if g1 - g0 <= max_gap_hours and g0 > 0 and g1 < T:
                    x0, x1 = series[g0 - 1], series[g1]
                    series[g0:g1] = x0 + (x1 - x0) * (np.arange(1, g1 - g0 + 1) / (g1 - g0 + 1))
                    f1[g0:g1, iy, ix] = True
            t2m[:, iy, ix] = series
        qc[f1] = 1
        n1 = int(f1.sum())
        missing = np.isnan(t2m) & inside[None]
        # 2) longer gaps: day-of-year x hour climatology from the other years
        if missing.any():
            clim = _climatology(year, years_available, DIR_SRC, lat_n, lon_n)
            if clim is not None:
                t = pd.DatetimeIndex(da["time"].values)
                doy = t.dayofyear - ((t.is_leap_year) & (t.dayofyear > 59)).astype(int)
                key = doy * 24 + t.hour
                cl = clim.sel(key=xr.DataArray(key, dims="time")).transpose("time", lat_n, lon_n).values
                f2 = missing & np.isfinite(cl)
                t2m = np.where(f2, cl, t2m).astype("float32")
                qc[f2] = 2
                n2 = int(f2.sum())
                missing = np.isnan(t2m) & inside[None]
        # 3) still missing: mean of the valid 3 x 3 neighbours at the same hour
        if missing.any():
            for ti in np.flatnonzero(missing.any(axis=(1, 2))):
                sl = t2m[ti]
                for iy, ix in zip(*np.where(missing[ti])):
                    win = sl[max(iy - 1, 0):iy + 2, max(ix - 1, 0):ix + 2]
                    if np.isfinite(win).any():
                        t2m[ti, iy, ix] = np.nanmean(win)
                        qc[ti, iy, ix] = 3
                        n3 += 1
            missing = np.isnan(t2m) & inside[None]

    n_left = int(missing.sum())
    t = pd.DatetimeIndex(da["time"].values)
    doy = (t.dayofyear - ((t.is_leap_year) & (t.dayofyear > 59)).astype(int)).astype("int16")
    hour = t.hour.astype("int16")

    row = {"year": year, "status": "ok" if n_left == 0 else "GAPS_REMAIN", "hours": T,
           "cells_inside": n_inside, "missing_hours_inside_before": n_missing0,
           "filled_shortgap": n1, "filled_climatology": n2, "filled_spatial": n3,
           "missing_after": n_left,
           "pct_missing_before": round(100.0 * n_missing0 / max(T * n_inside, 1), 4)}
    if report_only:
        ds.close()
        row["filled_file_check"] = (_compare_existing(dst, t2m, qc, np.asarray(doy), np.asarray(hour), lat_n, lon_n)
                                    if dst.exists() else "absent")
        print(f"[CHECK] {year}: inside cells={n_inside} missing={n_missing0} -> would fill 1/2/3 = {n1}/{n2}/{n3}, "
              f"left={n_left} | existing {dst.name}: {row['filled_file_check']}")
        return row

    out = xr.Dataset(
        {
            VAR: ((("time", lat_n, lon_n)), t2m),
            QC: ((("time", lat_n, lon_n)), qc),
        },
        coords={"time": da["time"].values, lat_n: da[lat_n].values, lon_n: da[lon_n].values,
                "dayofyear": ("time", doy), "hour": ("time", hour)},
    )
    out[VAR].attrs = {"units": "degree_Celsius", "standard_name": "air_temperature", "long_name": "2m air temperature"}
    out[QC].attrs = {"standard_name": "status_flag", "long_name": "Data quality flag",
                     "flag_values": FLAG_VALUES, "flag_meanings": FLAG_MEANINGS}
    out.attrs.update(cfg.global_attrs())
    out.attrs["qc_max_gap_hours"] = int(max_gap_hours)
    out.attrs["qc_missing_inside_mask_before"] = n_missing0
    out.attrs["qc_missing_inside_mask_after"] = n_left
    dst.parent.mkdir(parents=True, exist_ok=True)
    enc = {VAR: {"zlib": True, "complevel": 4, "dtype": "float32", "_FillValue": np.float32(np.nan)},
           QC: {"zlib": True, "complevel": 4, "dtype": "uint8"}}
    out.to_netcdf(dst, encoding=enc)
    ds.close()

    row["filled_file_check"] = "written"
    print(f"[OK] {year}: inside cells={n_inside} missing={n_missing0} -> filled 1/2/3 = {n1}/{n2}/{n3}, left={n_left} -> {dst}")
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=str, default=None, help="data root (parent of out/)")
    ap.add_argument("--input-root", type=str, default=None,
                    help="read out/04_yearly_raw (and, with --report-only, out/06_filled_yearly) from this folder; "
                         "write under --root")
    ap.add_argument("--years", type=str, default=None)
    ap.add_argument("--max-gap-hours", type=int, default=24)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--report-only", action="store_true",
                    help="write no NetCDF file; report missing hours and compare the existing filled files")
    a = ap.parse_args()

    global DIR_SRC, DIR_EXISTING
    if a.input_root:
        _in = Path(a.input_root).expanduser().resolve() / "out"
        DIR_SRC = _in / "04_yearly_raw"
        DIR_EXISTING = _in / "06_filled_yearly"
        print(f"[CFG] input root: yearly files from {DIR_SRC}" + (f"; existing filled files from {DIR_EXISTING}" if a.report_only else ""))

    years = _parse_years(a.years)
    available = sorted(int(p.stem) for p in DIR_SRC.glob("*.nc") if p.stem.isdigit())
    rows = []
    for y in years:
        rows.append(process_year(y, available, a.max_gap_hours, a.overwrite, a.report_only))
    rep = Path(cfg.DIR_REP) / "qc_gaps_summary.csv"
    rep.parent.mkdir(parents=True, exist_ok=True)
    df_new = pd.DataFrame([r for r in rows if r.get("status") != "skipped"])
    if rep.exists() and not df_new.empty:
        old = pd.read_csv(rep)
        old = old[~old["year"].isin(df_new["year"])]
        df_new = pd.concat([old, df_new], ignore_index=True).sort_values("year")
    if not df_new.empty:
        df_new.to_csv(rep, index=False)
        print(f"[OUT] {rep}")
    bad = [r for r in rows if r.get("status") == "GAPS_REMAIN" or str(r.get("filled_file_check", "")).startswith("DIFFERENT")]
    if a.report_only and rows:
        n_missing = sum(int(r.get("missing_hours_inside_before", 0)) for r in rows)
        n_ident = sum(str(r.get("filled_file_check", "")).startswith("identical") for r in rows)
        print(f"[CHECK] {len(rows)} year(s): {n_missing} missing hour(s) inside the mask; existing filled files identical "
              f"to the output of this step: {n_ident}/{len(rows)}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
