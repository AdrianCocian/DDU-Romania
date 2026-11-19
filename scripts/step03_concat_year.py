#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
step03_concat_year.py — concatenate monthly CLIP files into a single yearly RAW file (+ optional PNG QC)

Inputs (per year Y):
  out/03_clip_ro/{Y}_??_t2m_hourly_clip.nc

Outputs:
  out/04_yearly_raw/{Y}.nc
  out/04_yearly_raw/png/{Y}_t2m_first.png   (unless --no-png)
  out/04_yearly_raw/png/{Y}_t2m_mean.png    (unless --no-png)

Notes
-----
- NO masking/gap-fill/regridding/interpolation here (strict concat only).
- Enforces identical lat/lon grid across months; fails if grid differs.
- Portable for GitHub:
    • auto-sets DDU_ROOT to repo root (parent of /scripts) if not set
    • ensures repo root is on sys.path before importing config.py

CLI
---
  --years A-B            (default: cfg.YEARS or ENV YEARS)
  --allow-partial-year   allow incomplete years (default: off, same as ENV ALLOW_PARTIAL_YEAR=1)
  --no-png               disable PNGs (default: ON)
"""

from __future__ import annotations

import os
import sys
import re
import argparse
from pathlib import Path

import numpy as np
import xarray as xr

# -----------------------------
# Repo bootstrap (portable)
# -----------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("DDU_ROOT", str(REPO_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import cfg  # noqa: E402

# -----------------------------
# Optional Dask
# -----------------------------
_HAS_DASK = True
try:
    import dask  # noqa: F401
    dask.config.set(scheduler="threads", **{"array.slicing.split_large_chunks": True})
except Exception:
    _HAS_DASK = False

VAR = "t2m"

# -----------------------------
# Plotting (headless)
# -----------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _cmap_with_nodata():
    cmap = matplotlib.colormaps.get_cmap(getattr(cfg, "CMAP_NAME", "RdBu_r")).copy()
    cmap.set_bad(color=getattr(cfg, "NODATA_COLOR_HEX", "#FFF8B5"), alpha=1.0)
    return cmap


def _robust_limits(arr2d, qlow=2, qhigh=98):
    arr = np.asarray(arr2d)
    if not np.isfinite(arr).any():
        return (-10.0, 30.0)
    vmin = float(np.nanpercentile(arr, qlow))
    vmax = float(np.nanpercentile(arr, qhigh))
    if vmin == vmax:
        vmin, vmax = vmin - 0.1, vmax + 0.1
    return vmin, vmax


def _save_png_map(da2d, out_png: Path, title: str, label: str, lon=None, lat=None):
    out_png.parent.mkdir(parents=True, exist_ok=True)
    data = np.asarray(da2d)

    vmin, vmax = _robust_limits(data)
    fig, ax = plt.subplots(figsize=(7.2, 6.0), dpi=getattr(cfg, "PNG_DPI", 160))

    if lon is not None and lat is not None:
        extent = (float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max()))
        im = ax.imshow(
            data,
            origin="lower",
            extent=extent,
            interpolation=getattr(cfg, "IMAGE_INTERPOLATION", "nearest"),
            vmin=vmin,
            vmax=vmax,
            cmap=_cmap_with_nodata(),
            aspect="equal",
        )
        ax.set_xlabel("Longitude (°E)")
        ax.set_ylabel("Latitude (°N)")
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
    else:
        im = ax.imshow(data, origin="lower", vmin=vmin, vmax=vmax, cmap=_cmap_with_nodata())

    cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label(label)
    ax.set_title(title)

    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# Xarray helpers
# -----------------------------
def ensure_time_coord(ds: xr.Dataset) -> xr.Dataset:
    try:
        ds = xr.decode_cf(ds)
    except Exception:
        pass
    if "time" in ds.coords:
        ds = ds.sortby("time")
    if "step" in ds.dims and ds.sizes.get("step", 1) == 1:
        ds = ds.isel(step=0, drop=True)
    return ds


def ensure_unique_time(ds: xr.Dataset) -> xr.Dataset:
    if "time" not in ds.variables:
        return ds
    vals = ds["time"].values
    _, idx = np.unique(vals, return_index=True)
    if len(idx) != len(vals):
        ds = ds.isel(time=np.sort(idx))
    return ds


def ensure_var_t2m(ds: xr.Dataset) -> xr.Dataset:
    if VAR in ds.data_vars:
        return ds
    for v in ("t2m", "T2M", "t", "temperature", "air_temperature"):
        if v in ds.data_vars:
            return ds.rename({v: VAR})
    names = list(ds.data_vars)
    if not names:
        raise KeyError("Dataset has no data_vars.")
    return ds.rename({names[0]: VAR})


def detect_lat_lon_dims(ds: xr.Dataset) -> tuple[str, str]:
    dims = list(ds[VAR].dims) if VAR in ds.data_vars else list(next(iter(ds.data_vars.values())).dims)
    lat_dim = next((d for d in dims if d.lower() in ("latitude", "lat", "y")), None)
    lon_dim = next((d for d in dims if d.lower() in ("longitude", "lon", "x")), None)
    if not lat_dim or not lon_dim:
        raise ValueError(f"Cannot detect spatial dims (dims={dims})")
    return lat_dim, lon_dim


def _month_from_name(p: Path, year: int) -> int | None:
    m = re.search(rf"{year:04d}_(\d{{2}})_t2m_hourly_clip", p.name)
    if not m:
        return None
    return int(m.group(1))


def _write_year_netcdf(ds_year: xr.Dataset, out_nc: Path):
    out_nc.parent.mkdir(parents=True, exist_ok=True)
    enc = {
        VAR: {
            "zlib": True,
            "complevel": 4,
            "shuffle": True,
            "dtype": "float32",
            "_FillValue": np.float32(np.nan),
        },
        "time": {"zlib": True, "complevel": 4, "shuffle": True},
    }
    for engine in ("h5netcdf", "netcdf4"):
        try:
            ds_year.to_netcdf(out_nc, engine=engine, encoding=enc)
            return
        except Exception:
            pass
    ds_year.to_netcdf(out_nc)


# -----------------------------
# Year concat
# -----------------------------
def concat_one_year(year: int, allow_partial_year: bool, make_png: bool):
    dir_clip = Path(cfg.DIR_CLIP)
    dir_year = Path(cfg.DIR_YEAR)
    dir_png = dir_year / "png"

    dir_year.mkdir(parents=True, exist_ok=True)
    if make_png:
        dir_png.mkdir(parents=True, exist_ok=True)

    pattern = f"{year:04d}_??_t2m_hourly_clip.nc"
    files = sorted(dir_clip.glob(pattern))
    if not files:
        raise FileNotFoundError(f"[{year}] No monthly clip files in {dir_clip} (pattern={pattern})")

    months_present = sorted({m for m in (_month_from_name(p, year) for p in files) if m is not None})
    if len(months_present) != 12 and not allow_partial_year:
        raise RuntimeError(
            f"[{year}] Incomplete year: months={months_present} (n={len(months_present)}). "
            f"Use --allow-partial-year (or ENV ALLOW_PARTIAL_YEAR=1) only if intentional."
        )
    if len(months_present) != 12 and allow_partial_year:
        print(f"[WARN] {year}: incomplete year (months={months_present}) — continuing (allow_partial_year=True)")

    dss: list[xr.Dataset] = []
    lat_ref = lon_ref = None
    lat_dim = lon_dim = None

    print(f"[INFO] {year}: {len(files)} monthly files")
    for p in files:
        ds = xr.open_dataset(p, chunks="auto" if _HAS_DASK else None)
        ds = ensure_var_t2m(ensure_unique_time(ensure_time_coord(ds))).squeeze(drop=True)

        if lat_dim is None or lon_dim is None:
            lat_dim, lon_dim = detect_lat_lon_dims(ds)
            lat_ref = np.asarray(ds[lat_dim].values)
            lon_ref = np.asarray(ds[lon_dim].values)
        else:
            if (
                (ds.sizes[lat_dim] != lat_ref.size)
                or (ds.sizes[lon_dim] != lon_ref.size)
                or (not np.array_equal(np.asarray(ds[lat_dim].values), lat_ref))
                or (not np.array_equal(np.asarray(ds[lon_dim].values), lon_ref))
            ):
                ds.close()
                raise ValueError(
                    f"[{year}] Grid differs across months. Stopping to avoid modifying values.\n"
                    f"   Problem file: {p.name}"
                )

        dss.append(ds)

    ds_year = xr.concat(dss, dim="time", combine_attrs="override").sortby("time").squeeze(drop=True)
    ds_year = ensure_unique_time(ds_year)

    t2m = ds_year[VAR]

    try:
        non_nan = int(t2m.count().compute().values) if _HAS_DASK else int(t2m.count().values)
    except Exception:
        non_nan = int(np.isfinite(t2m.values).sum())

    total = int(t2m.size)
    print(f"[QC] {year} {VAR} shape={tuple(t2m.shape)}, non-NaN={non_nan}, NaN={total - non_nan}")

    out_nc = dir_year / f"{year}.nc"
    _write_year_netcdf(ds_year, out_nc)
    print(f"[OK] Yearly RAW saved: {out_nc}")

    if make_png:
        try:
            snap = t2m.isel(time=0)
            _save_png_map(
                snap,
                dir_png / f"{year}_t2m_first.png",
                f"{year} — {VAR} (first timestep)",
                f"{VAR} (°C)",
                lon=ds_year[lon_dim],
                lat=ds_year[lat_dim],
            )
            print(f"[OK] PNG first: {dir_png / f'{year}_t2m_first.png'}")
        except Exception as e:
            print(f"[WARN] PNG first failed: {e}")

        try:
            mean_year = t2m.mean("time")
            _save_png_map(
                mean_year,
                dir_png / f"{year}_t2m_mean.png",
                f"{year} — {VAR} (annual mean)",
                f"{VAR} (°C)",
                lon=ds_year[lon_dim],
                lat=ds_year[lat_dim],
            )
            print(f"[OK] PNG mean: {dir_png / f'{year}_t2m_mean.png'}")
        except Exception as e:
            print(f"[WARN] PNG mean failed: {e}")

    for d in dss:
        try:
            d.close()
        except Exception:
            pass
    try:
        ds_year.close()
    except Exception:
        pass

    print(f"[DONE] {year}\n")


# -----------------------------
# CLI
# -----------------------------
def _parse_years_spec(spec: str | None) -> list[int]:
    if not spec:
        return list(getattr(cfg, "YEARS", [2005]))
    s = spec.strip().replace("–", "-").replace("—", "-")
    if "-" in s and "," not in s:
        a, b = s.split("-", 1)
        a, b = int(a.strip()), int(b.strip())
        if a > b:
            a, b = b, a
        return list(range(a, b + 1))
    out: list[int] = []
    for tok in s.split(","):
        tok = tok.strip()
        if tok:
            out.append(int(tok))
    return out or list(getattr(cfg, "YEARS", [2005]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=str, default=None, help="A-B or list (e.g., 2005-2024 or 2020,2021)")
    ap.add_argument("--allow-partial-year", action="store_true", help="Allow incomplete years (default: off)")
    ap.add_argument("--no-png", action="store_true", help="Disable PNGs (default: ON)")
    args = ap.parse_args()

    years = _parse_years_spec(args.years or os.environ.get("YEARS"))
    allow_partial_year = bool(
        args.allow_partial_year or os.environ.get("ALLOW_PARTIAL_YEAR", "0").strip().lower() in ("1", "true", "yes")
    )
    make_png = not args.no_png

    try:
        cfg.save_run_meta()
    except Exception:
        pass

    print(f"[CFG] ROOT={cfg.ROOT}")
    print(f"[RUN] years={years[0]}–{years[-1]}  allow_partial_year={allow_partial_year}  make_png={make_png}")
    print(f"[IN ] {cfg.DIR_CLIP}")
    print(f"[OUT] {cfg.DIR_YEAR}")

    for y in years:
        concat_one_year(int(y), allow_partial_year=allow_partial_year, make_png=make_png)


if __name__ == "__main__":
    main()
