#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step04_tails.py — build + sanity-check tails in one script (portable, GitHub-ready)

Build:
  - From out/06_filled_yearly/{YYYY}.nc
  - Extract last PAD_DAYS (default 29 => 696h)
  - Write to out/06_filled_yearly_tails_{PAD_DAYS}d/{YYYY}_tail{PAD_DAYS}d.nc

Check:
  - Validate tails length, time monotonicity, presence of t2m, optional t2m_qc
  - Write summary: out/07_reports/tails_sanity_summary.csv

CLI examples:
  python scripts/step04_tails.py
  python scripts/step04_tails.py --years 2015-2024 --pad-days 29
  python scripts/step04_tails.py --years 2018,2020,2022 --build-only --overwrite
  python scripts/step04_tails.py --pad-days 29 --check-only

ENV (optional):
  YEARS=2015-2024           (used if --years not provided)
  ALLOW_PARTIAL_YEAR=1      allow writing tails even if source year shorter than PAD_DAYS*24 (still checked)
"""

from __future__ import annotations

import os
import re
import glob
import argparse
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import xarray as xr
import pandas as pd

# -----------------------------
# Repo bootstrap (portable)
# -----------------------------
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("DDU_ROOT", str(REPO_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import cfg  # noqa: E402


VAR, QC = "t2m", "t2m_qc"
QC_ALLOWED = {0, 1, 2, 3, 255}


# ----------------------------
# Helpers
# ----------------------------
def _parse_years_spec(s: Optional[str]) -> Optional[List[int]]:
    if not s:
        return None
    s = s.strip().replace("–", "-").replace("—", "-")
    if not s:
        return None
    if "-" in s and "," not in s:
        a, b = s.split("-", 1)
        a, b = int(a.strip()), int(b.strip())
        if a > b:
            a, b = b, a
        return list(range(a, b + 1))
    out = []
    for tok in s.split(","):
        tok = tok.strip()
        if tok:
            out.append(int(tok))
    return sorted(set(out)) if out else None


def parse_years(arg: Optional[str]) -> List[int]:
    """
    Parse years from:
      - CLI: '2015-2024' or '2018,2020,2022'
      - ENV YEARS (if CLI missing)
      - Auto-detect from cfg.DIR_FILLED/*.nc (numeric stems)
    """
    spec = arg or os.environ.get("YEARS")
    yrs = _parse_years_spec(spec)
    if yrs:
        return yrs

    # auto-detect from files
    years = sorted({
        int(Path(p).stem)
        for p in glob.glob(str(Path(cfg.DIR_FILLED) / "*.nc"))
        if Path(p).stem.isdigit()
    })
    return years


def _latlon_keys(ds: xr.Dataset) -> Tuple[Optional[str], Optional[str]]:
    for la in ("latitude", "lat", "y"):
        for lo in ("longitude", "lon", "x"):
            if la in ds.dims and lo in ds.dims:
                return la, lo
    return None, None


def _first_var_name(ds: xr.Dataset) -> Optional[str]:
    for k in ds.data_vars:
        return k
    return None


def _ensure_var_t2m(ds: xr.Dataset) -> xr.Dataset:
    """If 't2m' missing but only one data_var exists, rename it to 't2m'."""
    if VAR in ds.data_vars:
        return ds
    if len(ds.data_vars) == 1:
        v = _first_var_name(ds)
        if v and v != VAR:
            return ds.rename({v: VAR})
    return ds


def _tails_base_dir_from_cfg() -> Optional[Path]:
    """
    If cfg.DIR_FILLED_TAILS exists, treat it as either:
      - base with suffix: out/..._tails_29d   -> base becomes out/..._tails
      - already base:     out/..._tails      -> base stays out/..._tails
      - already custom:   out/somewhere      -> base stays as-is
    """
    if not hasattr(cfg, "DIR_FILLED_TAILS"):
        return None
    d = Path(getattr(cfg, "DIR_FILLED_TAILS"))
    name = d.name

    # common pattern: *_tails_29d
    m = re.match(r"^(.*_tails)_(\d+)d$", name)
    if m:
        return d.with_name(m.group(1))
    return d


def _tails_dir_for(pad_days: int) -> Path:
    """
    Always returns a pad-specific directory:
      <base>_{pad_days}d
    where base is derived from cfg.DIR_FILLED_TAILS if present, else default:
      out/06_filled_yearly_tails
    """
    base = _tails_base_dir_from_cfg()
    if base is None:
        base = Path(cfg.ROOT) / "out" / "06_filled_yearly_tails"
    return base.with_name(f"{base.name}_{pad_days}d")


def _write_atomic(ds: xr.Dataset, dst: Path, encoding: Dict[str, Any]) -> None:
    dst = Path(dst)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    try:
        try:
            ds.to_netcdf(tmp, encoding=encoding, engine="h5netcdf", mode="w")
        except (ImportError, ModuleNotFoundError, ValueError):
            # h5netcdf or h5py not installed: same file content through the netCDF4 library
            if tmp.exists():
                tmp.unlink()
            ds.to_netcdf(tmp, encoding=encoding, engine="netcdf4", mode="w")
        if dst.exists():
            dst.unlink()
        tmp.replace(dst)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except Exception:
                pass


# ----------------------------
# BUILD
# ----------------------------
def _sanity_print_tail(path_nc: Path, need_hours: int, expect_qc: bool = True) -> None:
    """Short summary to stdout."""
    try:
        with xr.open_dataset(path_nc) as ds:
            la, lo = _latlon_keys(ds)
            T = ds.sizes.get("time", 0)
            size_str = (
                T,
                ds.sizes.get(la, "?") if la else "?",
                ds.sizes.get(lo, "?") if lo else "?",
            )
            if expect_qc and QC in ds:
                try:
                    qc_codes = np.unique(ds[QC].values)
                except Exception:
                    qc_codes = "QC:ERR"
            else:
                qc_codes = "QC:NA"
            time_ok = bool(ds["time"].to_index().is_monotonic_increasing) if "time" in ds else False
            status = "OK" if T == need_hours else f"LEN={T}/{need_hours}"
            print(
                f"  ↳ sanity: {path_nc.name} | {status} | size={size_str} | "
                f"codes={qc_codes} | time↑={time_ok}"
            )
    except Exception as e:
        print(f"  ↳ sanity: {path_nc.name} | open_error: {e}")


def build_tail_for_year(year: int, pad_days: int = 29, overwrite: bool = False) -> Optional[Path]:
    src = Path(cfg.DIR_FILLED) / f"{year}.nc"
    if not src.exists():
        print(f"[!] missing source: {src}")
        return None

    tails_dir = _tails_dir_for(pad_days)
    tails_dir.mkdir(parents=True, exist_ok=True)
    dst = tails_dir / f"{year}_tail{pad_days}d.nc"

    need_hours = int(pad_days) * 24

    if dst.exists() and not overwrite:
        print(f"↪ tail exists: {dst}")
        _sanity_print_tail(dst, need_hours)
        return dst

    with xr.open_dataset(src) as ds:
        ds = _ensure_var_t2m(ds)
        keep = [v for v in (VAR, QC) if v in ds.variables]
        if VAR not in keep:
            print(f"[!] {src.name} missing '{VAR}'. Skipping.")
            return None

        T = int(ds.sizes.get("time", 0))
        if T <= 0:
            print(f"[!] {src.name} has time=0. Skipping.")
            return None

        start = max(0, T - need_hours)
        tail = ds[keep].isel(time=slice(start, T))

        # keep attrs
        try:
            tail.attrs.update(ds.attrs or {})
        except Exception:
            pass
        for k in keep:
            try:
                tail[k].attrs.update(ds[k].attrs or {})
            except Exception:
                pass

        # optional: preserve CRS if rioxarray used upstream
        try:
            import rioxarray  # noqa: F401
            if getattr(ds, "rio", None) and ds.rio.crs:
                tail = tail.rio.write_crs(ds.rio.crs)
        except Exception:
            pass

        enc: Dict[str, Any] = {v: {"zlib": True, "complevel": 4, "shuffle": True} for v in keep}
        enc["time"] = {"zlib": True, "complevel": 4, "shuffle": True}

        _write_atomic(tail, dst, enc)

    print(f"✔ tail saved: {dst}")
    _sanity_print_tail(dst, need_hours)
    return dst


# ----------------------------
# CHECK
# ----------------------------
def _check_one(path_nc: Path, need_hours: int) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "file": path_nc.name,
        "need_hours": need_hours,
        "time_len": None,
        "time_monotonic": None,
        "has_t2m": False,
        "has_qc": False,
        "qc_len_ok": None,
        "qc_codes_ok": None,
        "qc_codes": None,
        "lat_dim": None,
        "lon_dim": None,
        "ny": None,
        "nx": None,
        "t_start": None,
        "t_end": None,
        "status": "FAIL",
        "notes": "",
    }

    try:
        with xr.open_dataset(path_nc) as ds:
            ds = _ensure_var_t2m(ds)
            if VAR not in ds:
                row["notes"] = f"missing {VAR}"
                return row

            row["has_t2m"] = True
            la, lo = _latlon_keys(ds)
            if la and lo:
                row["lat_dim"], row["lon_dim"] = la, lo
                row["ny"] = int(ds.sizes.get(la, -1))
                row["nx"] = int(ds.sizes.get(lo, -1))

            da = ds[VAR]
            T = int(da.sizes.get("time", 0))
            row["time_len"] = T

            if T > 0 and "time" in da.coords:
                row["t_start"] = str(da["time"].values[0])
                row["t_end"] = str(da["time"].values[-1])
                try:
                    row["time_monotonic"] = bool(da["time"].to_index().is_monotonic_increasing)
                except Exception:
                    row["time_monotonic"] = None

            if QC in ds:
                row["has_qc"] = True
                row["qc_len_ok"] = (int(ds[QC].sizes.get("time", -1)) == T)
                try:
                    codes = np.unique(ds[QC].values)
                    row["qc_codes"] = ",".join(map(str, np.sort(codes)))
                    row["qc_codes_ok"] = set(map(int, np.asarray(codes).tolist())).issubset(QC_ALLOWED)
                except Exception:
                    row["qc_codes_ok"] = None

            ok_len = (T == need_hours)
            ok_mon = (row["time_monotonic"] is True)
            ok_qc = (not row["has_qc"]) or (row["qc_len_ok"] and (row["qc_codes_ok"] in (True, None)))

            row["status"] = "OK" if (ok_len and ok_mon and ok_qc and row["has_t2m"]) else "CHECK"

            notes = []
            if not ok_len:
                notes.append(f"LEN={T}/{need_hours}")
            if ok_mon is False:
                notes.append("time not monotonic")
            if row["has_qc"] and not row["qc_len_ok"]:
                notes.append("qc_len!=time_len")
            if row["has_qc"] and row["qc_codes_ok"] is False:
                notes.append(f"bad_qc_codes={row['qc_codes']}")
            row["notes"] = "; ".join(notes)
            return row

    except Exception as e:
        row["notes"] = f"open_error: {e}"
        return row


def run_check(pad_days: int, only_these_files: Optional[List[Path]] = None) -> pd.DataFrame:
    need_hours = int(pad_days) * 24

    if only_these_files:
        tails = [Path(p) for p in only_these_files if Path(p).exists()]
    else:
        base = _tails_dir_for(pad_days)
        tails = sorted(base.glob(f"*_tail{pad_days}d.nc"))

        # backward compat (legacy folder for 29d)
        if not tails and pad_days == 29:
            base_old = Path(cfg.ROOT) / "out" / "06_filled_yearly_tails_29d"
            tails = sorted(base_old.glob("*_tail29d.nc"))

    if not tails:
        print("[!] No tails found for pad-days =", pad_days)
        return pd.DataFrame([])

    print(f"[TAIL-SANITY] Checking {len(tails)} files (pad-days={pad_days}, need_hours={need_hours})")
    rows = []
    for p in tails:
        r = _check_one(p, need_hours)
        rows.append(r)
        print(
            f"{r['file']} | {r['status']} | time={r['time_len']}/{need_hours} | "
            f"QC={'yes' if r['has_qc'] else 'no'}"
            f"{'' if r['qc_codes'] is None else ' | codes=' + r['qc_codes']} | "
            f"time↑={r['time_monotonic']} | {r['t_start']} → {r['t_end']}"
            f"{'' if not r['notes'] else ' | ' + r['notes']}"
        )

    df = pd.DataFrame(rows)
    try:
        Path(cfg.DIR_REP).mkdir(parents=True, exist_ok=True)
        out_csv = Path(cfg.DIR_REP) / "tails_sanity_summary.csv"
        df.to_csv(out_csv, index=False)
        print("✔ summary CSV →", out_csv)
    except Exception as e:
        print("⚠ could not write summary CSV:", e)

    return df


# ----------------------------
# MAIN
# ----------------------------
def main():
    ap = argparse.ArgumentParser(description="Build + Check tails in one script")
    ap.add_argument("--years", type=str, default=None,
                    help="2015-2024 or 2018,2020,2022 (otherwise auto-detect from DIR_FILLED)")
    ap.add_argument("--pad-days", type=int, default=29, help="Tail length in days (default 29 => 696h)")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing tails when building")
    ap.add_argument("--build-only", action="store_true", help="Run BUILD only")
    ap.add_argument("--check-only", action="store_true", help="Run CHECK only")
    args = ap.parse_args()

    do_build = True
    do_check = True
    if args.build_only and not args.check_only:
        do_check = False
    if args.check_only and not args.build_only:
        do_build = False

    try:
        cfg.save_run_meta()
    except Exception:
        pass

    built_paths: List[Path] = []

    if do_build:
        years = parse_years(args.years)
        if not years:
            print("[!] No years detected in", cfg.DIR_FILLED)
        else:
            tails_dir = _tails_dir_for(args.pad_days)
            print(f"[BUILD] years={years} | pad-days={args.pad_days} | overwrite={args.overwrite}")
            print(f"[BUILD] DIR_FILLED = {cfg.DIR_FILLED}")
            print(f"[BUILD] DIR_TAILS  = {tails_dir}")
            for y in years:
                try:
                    p = build_tail_for_year(int(y), pad_days=int(args.pad_days), overwrite=bool(args.overwrite))
                    if p:
                        built_paths.append(p)
                except Exception as e:
                    print(f"✖ tail {y}: {e}")

    if do_check:
        only_list = built_paths if built_paths else None
        run_check(int(args.pad_days), only_these_files=only_list)


if __name__ == "__main__":
    main()
