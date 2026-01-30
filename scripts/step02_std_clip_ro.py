# -*- coding: utf-8 -*-
r"""
step02_std_clip_ro.py — Standardize + STRICT RO polygon clip (ALL MONTHS by default) + PNG quicklook (default ON)

What it does
------------
- Reads monthly raw files from out/01_download (prefer {YYYY}_{MM}_raw.nc from step01).
- Standardizes to:
    • var name: t2m
    • units: degC
    • dims: time, lat, lon
    • lon normalized to [-180,180], lat ascending
- Clips STRICT to Romania polygon:
    • outside polygon becomes real NaN
    • optional assert on inside cell count (INSIDE_EXPECTED)
- Writes:
    out/02_standardized/{YYYY}_{MM}_t2m_hourly_std.nc
    out/03_clip_ro/{YYYY}_{MM}_t2m_hourly_clip.nc
    out/03_clip_ro/png/{YYYY}_{MM}_t2m_hourly_clip_tHH.png (unless --no-png)

Portability (GitHub-ready)
--------------------------
- Auto-sets DDU_ROOT to repo root (parent of /scripts) if not set.
- Ensures repo root is in sys.path, so `from config import cfg` works even when running the script directly.

CLI
---
  --years A-B            (default: cfg.YEARS)
  --months 01-12         (default: cfg.MONTHS)
  --only-month MM        alias for --months MM
  --time-index N         hour index for PNG (default 0)
  --no-png               disable PNG (default ON)
  --force-std            rebuild *_std.nc even if exists
  --force-clip           rebuild *_clip.nc even if exists
  --one-per-year         quick-test: process only ONE month per year (prefers selected month(s), else first available)

ENV (optional)
--------------
  FORCE_STD=1            same as --force-std
  FORCE_CLIP=1           same as --force-clip
  ALLOW_FALLBACK=1       allow bbox clip if polygon clip fails (NOT recommended)
  ASSERT_INSIDE=1        assert inside cells == INSIDE_EXPECTED (default off)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# -----------------------------
# Repo bootstrap (portable)
# -----------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("DDU_ROOT", str(REPO_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# --- standard libs ---
import shutil
import zipfile
import tarfile
import gzip
import re
import warnings
import subprocess
import json
import argparse
from typing import Tuple, Dict, List, Optional

import numpy as np
import pandas as pd
import xarray as xr

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# --- GDAL/PROJ (Windows/Conda) ---
os.environ.setdefault("SHAPE_RESTORE_SHX", "YES")
for _p in [Path(sys.prefix) / "Library" / "share" / "gdal", Path(sys.prefix) / "share" / "gdal"]:
    if _p.exists():
        os.environ.setdefault("GDAL_DATA", str(_p))
        break
for _p in [Path(sys.prefix) / "Library" / "share" / "proj", Path(sys.prefix) / "share" / "proj"]:
    if _p.exists():
        os.environ.setdefault("PROJ_LIB", str(_p))
        break

# --- GIS deps ---
import geopandas as gpd
from shapely.geometry import Point
from shapely.prepared import prep

try:
    from shapely import set_precision
except Exception:
    set_precision = None

try:
    from shapely.validation import make_valid as _make_valid
except Exception:
    _make_valid = None

import rioxarray  # noqa
from rasterio.transform import from_bounds

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl

from config import cfg  # noqa: E402


# --- ENV toggles ---
FORCE_STD_ENV = os.environ.get("FORCE_STD", "0").strip().lower() in ("1", "true", "yes")
FORCE_CLIP_ENV = os.environ.get("FORCE_CLIP", "0").strip().lower() in ("1", "true", "yes")
ALLOW_FALLBACK = os.environ.get("ALLOW_FALLBACK", "0").strip().lower() in ("1", "true", "yes")
ASSERT_INSIDE = os.environ.get("ASSERT_INSIDE", "0").strip().lower() in ("1", "true", "yes")


# ------------------------ helpers ------------------------
def _ensure_dir(p: Path):
    try:
        cfg.ensure(p, verbose=False)
    except Exception:
        p.mkdir(parents=True, exist_ok=True)


def _global_attrs() -> dict:
    try:
        return cfg.global_attrs()
    except Exception:
        return {"Conventions": "CF-1.10", "history": "std_clip"}


def _parse_years(spec: Optional[str], default: List[int]) -> List[int]:
    if not spec:
        return list(default)
    s = spec.strip().replace("–", "-").replace("—", "-")
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
    return out or list(default)


def _parse_months(spec: Optional[str], default: List[str]) -> List[str]:
    if not spec:
        return list(default)
    s = spec.strip()
    if "-" in s and "," not in s:
        a, b = s.split("-", 1)
        a, b = int(a.strip()), int(b.strip())
        if a > b:
            a, b = b, a
        return [f"{m:02d}" for m in range(a, b + 1)]
    out: List[str] = []
    for tok in s.split(","):
        tok = tok.strip()
        if tok:
            out.append(f"{int(tok):02d}")
    return out or list(default)


def _ym_from_stem(stem: str) -> Optional[Tuple[int, int]]:
    m = re.search(r"(20\d{2})[_-]?(\d{2})", stem)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def _count_inside_non_nan(da2d: xr.DataArray) -> int:
    return int(da2d.notnull().sum().item())


def _extent(da: xr.DataArray) -> Tuple[float, float, float, float]:
    return float(da["lon"].min()), float(da["lon"].max()), float(da["lat"].min()), float(da["lat"].max())


# --------------------- opener / std ---------------------
def sniff_magic(path: str) -> str:
    p = Path(path)
    try:
        with open(p, "rb") as f:
            head = f.read(8)
    except Exception:
        head = b""
    if head.startswith(b"PK\x03\x04"):
        return "zip"
    if head.startswith(b"\x1f\x8b"):
        return "gz"
    if head.startswith(b"\x89HDF"):
        return "netcdf4"
    if head.startswith(b"CDF"):
        return "netcdf3"
    if head.startswith(b"GRIB"):
        return "grib"
    try:
        if tarfile.is_tarfile(p):
            return "tar"
        if zipfile.is_zipfile(p):
            return "zip"
    except Exception:
        pass
    ext = p.suffix.lower()
    if ext in (".nc", ".cdf"):
        return "unknown-nc"
    if ext in (".grib", ".grb", ".grib2", ".grb2"):
        return "grib"
    if ext in (".zip",):
        return "zip"
    if ext in (".gz", ".gzip"):
        return "gz"
    if ext in (".tar", ".tgz", ".tar.gz"):
        return "tar"
    return "unknown"


def _safe_open_netcdf(path: str) -> xr.Dataset:
    for eng in ("h5netcdf", "netcdf4", "scipy"):
        try:
            return xr.open_dataset(path, engine=eng, decode_times=True)
        except Exception:
            pass
    raise OSError(f"Cannot open NetCDF with known engines: {path}")


def open_any(path_in: str) -> xr.Dataset:
    p = Path(path_in)
    fmt = sniff_magic(str(p))
    work_tmp = Path(cfg.DIR_DL) / "_extracted" / p.stem
    work_tmp.mkdir(parents=True, exist_ok=True)

    if fmt in ("netcdf3", "netcdf4", "unknown-nc"):
        return _safe_open_netcdf(str(p))

    if fmt == "grib":
        try:
            import cfgrib  # noqa
            return xr.open_dataset(str(p), engine="cfgrib", backend_kwargs={"indexpath": ""})
        except Exception:
            if shutil.which("grib_to_netcdf"):
                out_nc = work_tmp / (p.stem + "_fromgrib.nc")
                subprocess.run(["grib_to_netcdf", "-o", str(out_nc), str(p)], check=True, capture_output=True)
                return _safe_open_netcdf(str(out_nc))
            raise OSError("GRIB detected but cfgrib/eccodes missing.")

    if fmt == "gz":
        out_nc = work_tmp / (p.stem + "_extracted.nc")
        with gzip.open(p, "rb") as src, open(out_nc, "wb") as dst:
            shutil.copyfileobj(src, dst)
        return open_any(str(out_nc))

    if fmt == "tar":
        with tarfile.open(p, "r:*") as tf:
            names = tf.getnames()
            pick = next((n for n in names if n.endswith(".nc")), None) or next(
                (n for n in names if n.endswith((".grib", ".grb", ".grib2", ".grb2"))), None
            )
            if not pick:
                raise OSError(f"TAR contains no .nc/.grib: {p}")
            out_fp = work_tmp / f"{p.stem}__{Path(pick).name}"
            with tf.extractfile(pick) as s, open(out_fp, "wb") as d:
                shutil.copyfileobj(s, d)
        return open_any(str(out_fp))

    if fmt == "zip":
        with zipfile.ZipFile(p, "r") as zf:
            members = zf.namelist()

            def _largest(exts):
                c = [m for m in members if any(m.endswith(e) for e in exts)]
                return max(c, key=lambda m: zf.getinfo(m).file_size) if c else None

            pick = _largest([".nc"]) or _largest([".grib", ".grb", ".grib2", ".grb2"])
            if not pick:
                raise OSError(f"ZIP contains no .nc/.grib: {p}")
            out_fp = work_tmp / f"{p.stem}__{Path(pick).name}"
            with zf.open(pick) as s, open(out_fp, "wb") as d:
                shutil.copyfileobj(s, d)
        return open_any(str(out_fp))

    raise OSError(f"Unsupported format for {p} (magic={fmt})")


def find_temp_var(ds: xr.Dataset) -> str:
    for v in ["t2m", "2m_temperature", "T2M", "t", "temperature", "air_temperature"]:
        if v in ds.data_vars:
            return v
    for v in ds.data_vars:
        if ds[v].ndim >= 2:
            return v
    raise KeyError("No temperature-like variable found.")


def _ensure_datetime64ns(da: xr.DataArray) -> xr.DataArray:
    if np.issubdtype(da.dtype, np.datetime64):
        return da.astype("datetime64[ns]")
    return xr.DataArray(pd.to_datetime(np.asarray(da.values)), dims=da.dims)


def _to_timedelta64ns(da: xr.DataArray) -> xr.DataArray:
    if np.issubdtype(da.dtype, np.timedelta64):
        return da.astype("timedelta64[ns]")
    u = (da.attrs.get("units", "") or "").lower()
    vals = np.asarray(da.values)
    if "hour" in u:
        f = np.timedelta64(1, "h")
    elif "min" in u:
        f = np.timedelta64(1, "m")
    elif "day" in u:
        f = np.timedelta64(1, "D")
    elif "sec" in u:
        f = np.timedelta64(1, "s")
    else:
        f = np.timedelta64(1, "h") if np.nanmax(vals) <= 1000 else np.timedelta64(1, "s")
    return xr.DataArray(vals.astype("int64"), dims=da.dims) * f


def ensure_time_1d(ds: xr.Dataset, var: str) -> xr.Dataset:
    if ("valid_time" in ds.dims or "valid_time" in ds.coords) and "time" not in ds.dims:
        vt = _ensure_datetime64ns(ds["valid_time"])
        ds = ds.assign_coords(time=("valid_time", vt.values)).swap_dims({"valid_time": "time"})
        if "valid_time" in ds.coords:
            ds = ds.reset_coords("valid_time", drop=True)

    if "time" in ds.dims and "step" in ds.dims and ds.sizes.get("step", 1) > 1:
        t = _ensure_datetime64ns(ds["time"])
        s = _to_timedelta64ns(ds["step"])
        v = ds[var]
        if s.dims == ("time",):
            vt = t + s
            v = v.assign_coords(time=vt)
        else:
            vt = t.expand_dims({"step": s.sizes["step"]}) + s
            v = v.transpose("time", "step", ...).stack(time=("time", "step"))
            vt = vt.stack(time=("time", "step"))
            v = v.assign_coords(time=vt)
        ds = xr.Dataset({var: v})
        if "step" in ds.dims:
            try:
                ds = ds.drop_vars("step")
            except Exception:
                pass
        if "step" in ds.coords:
            ds = ds.reset_coords("step", drop=True)
    return ds


def to_celsius(da: xr.DataArray) -> xr.DataArray:
    u = (da.attrs.get("units", "") or "").lower()
    out = da - 273.15 if u in ("k", "kelvin") else da
    if u not in ("k", "kelvin", "c", "degc", "degree_celsius", "degrees_celsius", "°c"):
        med = float(np.nanmedian(da.values))
        out = da - 273.15 if med > 200 else da
    out.attrs.update({"units": "degC", "long_name": "2m air temperature"})
    return out


def to_lon_lat(da: xr.DataArray) -> xr.DataArray:
    rename = {}
    for a, b in [("longitude", "lon"), ("latitude", "lat"), ("x", "lon"), ("y", "lat")]:
        if a in da.dims:
            rename[a] = b
    if rename:
        da = da.rename(rename)
    if {"time", "lat", "lon"}.issubset(da.dims):
        da = da.transpose("time", "lat", "lon")

    if "lon" in da.coords:
        lon = da["lon"]
        try:
            if float(lon.max()) > 180:
                da = da.assign_coords(lon=((lon + 180) % 360) - 180).sortby("lon")
        except Exception:
            pass

    if "lat" in da.coords:
        lat = da["lat"]
        try:
            if float(lat[0]) > float(lat[-1]):
                da = da.reindex(lat=list(reversed(lat.values)))
        except Exception:
            pass
    return da


def standardize_one(src: str, out_std: str):
    ds = open_any(src)
    var = find_temp_var(ds)
    ds = ensure_time_1d(ds, var)
    da = to_celsius(ds[var].astype("float32")).where(lambda x: (x > -90) & (x < 70))
    da = to_lon_lat(da).rename("t2m")

    ds_out = da.to_dataset()
    cu = getattr(cfg, "COORD_UNITS", {"lat": "degrees_north", "lon": "degrees_east"})
    if "lat" in ds_out.coords:
        ds_out["lat"].attrs["units"] = cu.get("lat", "degrees_north")
    if "lon" in ds_out.coords:
        ds_out["lon"].attrs["units"] = cu.get("lon", "degrees_east")
    ds_out.attrs.update(_global_attrs())

    _ensure_dir(Path(out_std).parent)
    ds_out.to_netcdf(
        out_std,
        encoding={
            "t2m": {
                "zlib": True,
                "complevel": 4,
                "shuffle": True,
                "dtype": "float32",
                "_FillValue": np.float32(np.nan),
            }
        },
        engine="h5netcdf",
    )


# -------------------------- polygon prep --------------------------
def _geom_fix(gdf: "gpd.GeoDataFrame") -> "gpd.GeoDataFrame":
    try:
        gdf = gdf.explode(index_parts=False).reset_index(drop=True)
    except Exception:
        pass
    try:
        if _make_valid is not None:
            gdf["geometry"] = gdf["geometry"].map(_make_valid)
    except Exception:
        pass
    try:
        gdf["geometry"] = gdf.buffer(0)
    except Exception:
        pass
    eps = float(getattr(cfg, "CLIP_GEOM_PREC_EPS", 1e-9))
    if set_precision and eps > 0:
        try:
            gdf["geometry"] = gdf["geometry"].map(lambda g: set_precision(g, eps))
        except Exception:
            pass
    return gdf


def load_roi():
    shp_path = getattr(cfg, "SHP_RO", None) or getattr(cfg, "SHP", None)
    if not shp_path:
        raise RuntimeError("cfg.SHP_RO not set. Fix config.py or set DDU_SHP_RO.")
    gdf = gpd.read_file(str(shp_path))
    if gdf.empty:
        raise RuntimeError(f"Shapefile has no geometries: {shp_path}")
    if gdf.crs is None:
        forced = getattr(cfg, "SHP_CRS", None)
        gdf = gdf.set_crs(forced or f"EPSG:{getattr(cfg,'CRS_EPSG',4326)}")
    gdf = gdf.to_crs(epsg=getattr(cfg, "CRS_EPSG", 4326))
    gdf = _geom_fix(gdf)
    try:
        ge = gdf.union_all()
    except Exception:
        ge = gdf.unary_union
    if ge.is_empty:
        raise RuntimeError("Unioned geometry is empty.")
    return gdf, ge


# ------------------------------ CLIP ------------------------------
def _center_of_cell_mask(clipped: xr.DataArray, ge) -> xr.DataArray:
    lon2d, lat2d = np.meshgrid(clipped.lon.values, clipped.lat.values)
    P = prep(ge)
    flat = np.fromiter(
        (P.contains(Point(x, y)) for x, y in zip(lon2d.ravel(), lat2d.ravel())),
        dtype=bool,
        count=lon2d.size,
    ).reshape(lon2d.shape)
    return xr.DataArray(flat, dims=("lat", "lon"))


def clip_one(std_path: str, clip_path: str) -> int:
    ds = xr.open_dataset(std_path)
    da = ds["t2m"] if "t2m" in ds.data_vars else ds[find_temp_var(ds)]
    da = to_lon_lat(da)

    da.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
    da.rio.write_crs(f"EPSG:{getattr(cfg,'CRS_EPSG',4326)}", inplace=True)

    dlon = float(da.lon[1] - da.lon[0])
    dlat = float(da.lat[1] - da.lat[0])
    transform = from_bounds(
        float(da.lon.min()) - dlon / 2,
        float(da.lat.min()) - dlat / 2,
        float(da.lon.max()) + dlon / 2,
        float(da.lat.max()) + dlat / 2,
        da.sizes["lon"],
        da.sizes["lat"],
    )
    da.rio.write_transform(transform, inplace=True)
    da.rio.write_nodata(np.nan, inplace=True)

    gdf, ge = load_roi()
    all_touched = bool(getattr(cfg, "CLIP_ALL_TOUCHED", True))

    try:
        clipped = da.rio.clip([ge], gdf.crs, drop=True, all_touched=all_touched)
        if clipped.size == 0:
            raise ValueError("empty-after-clip")
    except Exception as e:
        if not ALLOW_FALLBACK:
            raise RuntimeError(f"[CLIP][FATAL] Shapefile clip failed: {e}. Fix shapefile/CRS.")
        N, W, S, E = cfg.BBOX
        clipped = da.rio.clip_box(minx=W, miny=S, maxx=E, maxy=N)
        if clipped.size == 0:
            raise RuntimeError("BBOX clip also produced empty result.")

    base = clipped.isel(time=0)
    inside_n = _count_inside_non_nan(base)

    if ASSERT_INSIDE and hasattr(cfg, "INSIDE_EXPECTED"):
        exp = int(getattr(cfg, "INSIDE_EXPECTED"))
        if inside_n != exp:
            mask_pip = _center_of_cell_mask(base, ge)
            clipped = clipped.where(mask_pip)
            base = clipped.isel(time=0)
            inside_n = _count_inside_non_nan(base)
            if inside_n != exp:
                raise AssertionError(f"[CLIP][ASSERT] inside={inside_n}, expected={exp} (ALL_TOUCHED={all_touched})")

    try:
        if not cfg.REF_MASK_PATH.exists():
            ref_mask = base.notnull().astype("uint8").rename("inside_mask_REF")
            _ensure_dir(cfg.DIR_MASKS)
            ref_mask.to_dataset().to_netcdf(cfg.REF_MASK_PATH)
    except Exception:
        pass

    ds_clip = clipped.to_dataset(name="t2m")
    ds_clip.attrs.update(_global_attrs())
    _ensure_dir(Path(clip_path).parent)
    ds_clip.to_netcdf(
        clip_path,
        encoding={
            "t2m": {
                "zlib": True,
                "complevel": 4,
                "shuffle": True,
                "dtype": "float32",
                "_FillValue": np.float32(np.nan),
            }
        },
        engine="h5netcdf",
    )
    return inside_n


# ---------------------------- PNG quicklook ----------------------------
def _save_quick_png(clip_nc: Path, out_png: Path, time_index: int = 0):
    ds = xr.open_dataset(clip_nc)
    da = ds["t2m"].isel(time=time_index).load()

    cmap = mpl.cm.get_cmap(getattr(cfg, "CMAP_NAME", "RdBu_r")).copy()
    cmap.set_bad(color=getattr(cfg, "NODATA_COLOR_HEX", "#FFF8B5"), alpha=1.0)

    data = da.values
    if np.isfinite(data).any():
        vmin = float(np.nanpercentile(data, 2))
        vmax = float(np.nanpercentile(data, 98))
        if vmin == vmax:
            vmin, vmax = vmin - 0.1, vmax + 0.1
    else:
        vmin, vmax = -10, 30

    lon = ds["lon"].values
    lat = ds["lat"].values
    extent = (float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max()))

    _ensure_dir(out_png.parent)
    fig, ax = plt.subplots(figsize=(7.2, 6.0), dpi=getattr(cfg, "PNG_DPI", 160))
    im = ax.imshow(
        data,
        origin="lower",
        extent=extent,
        interpolation=getattr(cfg, "IMAGE_INTERPOLATION", "nearest"),
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        aspect="equal",
    )
    cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label("2 m air temperature (°C)")

    when = pd.to_datetime(ds["time"].values[time_index]).strftime("%Y-%m-%d %H:%M UTC")
    ax.set_title(f"{clip_nc.stem} — {when}")
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])

    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


# -------------------- input selection (ALL months) --------------------
def _candidate_for_month(year: int, month: str) -> Optional[Path]:
    """
    Prefer step01 output: {YYYY}_{MM}_raw.nc
    Fallback: any file matching {YYYY}_{MM}* with known extensions (largest file).
    """
    m2 = f"{int(month):02d}"
    p_pref = Path(cfg.DIR_DL) / f"{year}_{m2}_raw.nc"
    if p_pref.exists():
        return p_pref

    exts = {".nc", ".grib", ".grb", ".grib2", ".zip", ".gz", ".gzip", ".tar"}
    cand = [p for p in Path(cfg.DIR_DL).glob(f"{year}_{m2}*") if p.suffix.lower() in exts and not p.name.endswith(".idx")]
    if not cand:
        return None
    # pick largest to avoid tiny index/partial files
    cand.sort(key=lambda p: p.stat().st_size if p.exists() else 0, reverse=True)
    return cand[0]


def collect_inputs(years: List[int], months: List[str], one_per_year: bool) -> List[Path]:
    out: List[Path] = []
    if one_per_year:
        # choose first available among selected months (or any if none found)
        for y in years:
            found = None
            for m in months:
                found = _candidate_for_month(y, m)
                if found:
                    break
            if not found:
                # try any month in download dir
                cand = sorted(Path(cfg.DIR_DL).glob(f"{y}_*_raw.nc"))
                found = cand[0] if cand else None
            if found:
                out.append(found)
        return out

    # ALL months
    for y in years:
        for m in months:
            p = _candidate_for_month(y, m)
            if p:
                out.append(p)
            else:
                print(f"[WARN] Missing input for {y}-{m}: expected {y}_{m}_raw.nc in {cfg.DIR_DL}")
    return out


# --------------------------------- main ---------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=str, default=None, help="A-B or list (e.g., 2005-2024 or 2020,2021)")
    ap.add_argument("--months", type=str, default=None, help="01-12 or list (e.g., 01,07,12)")
    ap.add_argument("--only-month", type=str, default=None, help="MM (alias for --months MM)")
    ap.add_argument("--one-per-year", action="store_true", help="process only one month per year (quick test)")
    ap.add_argument("--time-index", type=int, default=0, help="Hour index for PNG (default 0)")
    ap.add_argument("--no-png", action="store_true", help="Disable PNGs (default ON)")
    ap.add_argument("--force-std", action="store_true", help="Rebuild standardized NetCDF even if exists")
    ap.add_argument("--force-clip", action="store_true", help="Rebuild clipped NetCDF even if exists")
    args = ap.parse_args()

    years = _parse_years(args.years, list(cfg.YEARS))

    months_spec = args.months
    if (not months_spec) and args.only_month:
        months_spec = str(args.only_month).strip()
    months = _parse_months(months_spec, list(cfg.MONTHS))

    time_index = int(args.time_index)
    make_png = not args.no_png

    force_std = bool(args.force_std or FORCE_STD_ENV)
    force_clip = bool(args.force_clip or FORCE_CLIP_ENV)

    print(f"[CFG] ROOT={cfg.ROOT}")
    print(f"[CFG] SHP={cfg.SHP_RO}")
    print(f"[RUN] Years={years[0]}–{years[-1]}  Months={months[0]}..{months[-1]}  one_per_year={args.one_per_year}")
    print(f"[RUN] force_std={force_std}  force_clip={force_clip}  make_png={make_png}  time_index={time_index}")

    inputs = collect_inputs(years, months, args.one_per_year)
    if not inputs:
        print("✖ No inputs found. Run step01 first or place monthly files in out/01_download.")
        return

    meta_path = Path(cfg.DIR_CLIP) / "_clip_meta.json"
    first_inside = None

    for p in inputs:
        ym = _ym_from_stem(p.stem)
        if not ym:
            print(f"[WARN] Skip (cannot parse YYYY_MM): {p.name}")
            continue
        yy, mm = ym
        mm2 = f"{mm:02d}"

        out_std = Path(cfg.DIR_STD) / f"{yy}_{mm2}_t2m_hourly_std.nc"
        out_clip = Path(cfg.DIR_CLIP) / f"{yy}_{mm2}_t2m_hourly_clip.nc"

        # ---------- STD ----------
        if out_std.exists() and not force_std:
            print("• SKIP STD ", out_std.name)
        else:
            try:
                if out_std.exists() and force_std:
                    out_std.unlink(missing_ok=True)
                standardize_one(str(p), str(out_std))
                print("✔ STD  ", out_std.name)
            except Exception as e:
                print(f"✖ STD fail {p.name} → {e}")
                continue

        # ---------- CLIP ----------
        try:
            with xr.open_dataset(out_std) as dtmp:
                da = to_lon_lat(dtmp["t2m"])
                lonmin, lonmax, latmin, latmax = _extent(da)
                print(f"[DBG] {yy}-{mm2} data lon=[{lonmin:.3f},{lonmax:.3f}] lat=[{latmin:.3f},{latmax:.3f}]")

            if out_clip.exists() and not force_clip:
                print("• SKIP CLIP", out_clip.name)
                with xr.open_dataset(out_clip) as dsc:
                    inside_n = _count_inside_non_nan(dsc["t2m"].isel(time=0))
            else:
                if out_clip.exists() and force_clip:
                    out_clip.unlink(missing_ok=True)
                inside_n = clip_one(str(out_std), str(out_clip))
                print(f"✔ CLIP {out_clip.name}  inside_nonNaN={inside_n}")

            if first_inside is None:
                first_inside = inside_n
                _ensure_dir(Path(cfg.DIR_CLIP))
                meta = {
                    "inside_cells_first": inside_n,
                    "all_touched": bool(getattr(cfg, "CLIP_ALL_TOUCHED", True)),
                    "crs": f"EPSG:{getattr(cfg,'CRS_EPSG',4326)}",
                }
                try:
                    with open(meta_path, "w", encoding="utf-8") as f:
                        json.dump(meta, f, indent=2)
                except Exception:
                    pass

        except Exception as e:
            print(f"✖ CLIP fail {out_std.name} → {e}")
            continue

        # ---------- PNG (DEFAULT ON) ----------
        if make_png:
            try:
                png_dir = Path(cfg.DIR_CLIP) / "png"
                _ensure_dir(png_dir)
                png_path = png_dir / (out_clip.stem + f"_t{time_index:02d}.png")
                _save_quick_png(out_clip, png_path, time_index=time_index)
                print(f"✔ PNG  {png_path.name}")
            except Exception as e:
                print(f"✖ PNG fail {out_clip.name} → {e}")


if __name__ == "__main__":
    main()
