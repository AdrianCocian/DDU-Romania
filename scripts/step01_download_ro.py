# -*- coding: utf-8 -*-
"""
step01_download_ro.py — ERA5-Land hourly T2M downloader (CDS) + unzip/convert + merge + auto-repair

Key features
------------
- Portable repo bootstrap:
    • sets DDU_ROOT to repo root (parent of /scripts) if not set
    • ensures repo root is on sys.path so `from config import cfg` works from anywhere
- Requests NetCDF; if server returns a ZIP, detects via magic bytes and handles it as ZIP
- Extracts only .nc/.grib from archives; converts GRIB→NetCDF (grib_to_netcdf, cfgrib fallback)
- Concatenates month parts along time → {YYYY_MM}_raw.nc (lossless)
- Keeps only {YYYY_MM}_raw.nc in out/01_download; cleans other month artifacts
- Minimal QC: variable, units, expected hours per month, grid size (no value modification)

Requirements
------------
- ~/.cdsapirc and ERA5-Land license accepted in CDS account
- conda env packages: cdsapi, xarray, netcdf4/h5netcdf, cfgrib+eccodes (optional but recommended)
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

from config import cfg  # noqa: E402

# -----------------------------
# Imports (standard libs)
# -----------------------------
import argparse
import calendar
import shutil
import subprocess
import time
import zipfile

# ========== Optional (QC/merge) ==========
_HAS_XARRAY = False
try:
    import xarray as xr  # noqa: F401
    _HAS_XARRAY = True
except Exception:
    _HAS_XARRAY = False


# -----------------------------
# CLI / ENV parsing
# -----------------------------
def _parse_years_spec(s: str | None) -> list[int] | None:
    if not s:
        return None
    s = s.strip().replace("–", "-").replace("—", "-")
    if not s:
        return None
    if "-" in s:
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
    return out or None


def _parse_months_spec(s: str | None) -> list[str] | None:
    if not s:
        return None
    s = s.strip()
    if not s:
        return None
    # allow "01-12" or "1-12"
    if "-" in s and "," not in s:
        a, b = s.split("-", 1)
        a, b = int(a.strip()), int(b.strip())
        if a > b:
            a, b = b, a
        return [f"{m:02d}" for m in range(a, b + 1)]
    out: list[str] = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(f"{int(tok):02d}")
    return out or None


def parse_args():
    ap = argparse.ArgumentParser(description="ERA5-Land hourly T2M downloader (Romania bbox/grid from config)")
    ap.add_argument("--years", type=str, default=None, help="e.g. 2005-2024 or 2018,2020,2022 (overrides cfg/ENV)")
    ap.add_argument("--months", type=str, default=None, help="e.g. 01-12 or 01,07,12 (overrides cfg/ENV)")
    ap.add_argument("--force", action="store_true", help="force re-download (also sets nocache)")
    return ap.parse_args()


# ========== Utils ==========
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def month_days(year: int, month: str) -> list[str]:
    n = calendar.monthrange(year, int(month))[1]
    return [f"{d:02d}" for d in range(1, n + 1)]


def target_path(year: int, month: str, ext: str) -> Path:
    return cfg.DIR_DL / f"{year}_{month}{ext}"


def already_ok(p: Path) -> bool:
    try:
        return p.exists() and p.stat().st_size > 0
    except Exception:
        return False


def _sniff_format(path: Path) -> str:
    """
    'zip' | 'netcdf3' | 'netcdf4' | 'unknown'
    ZIP:   PK\x03\x04
    NC3:   CDF\001 / CDF\002
    HDF5:  \x89HDF\r\n\x1a\n  (NetCDF4)
    """
    try:
        with open(path, "rb") as f:
            sig = f.read(8)
    except Exception:
        return "unknown"
    if sig[:2] == b"PK":
        return "zip"
    if sig[:3] == b"CDF":
        return "netcdf3"
    if sig.startswith(b"\x89HDF"):
        return "netcdf4"
    return "unknown"


def _is_netcdf_file(p: Path) -> bool:
    return _sniff_format(p) in ("netcdf3", "netcdf4")


def month_done(year: int, month: str) -> bool:
    p = cfg.DIR_DL / f"{year}_{month}_raw.nc"
    return p.exists() and _is_netcdf_file(p)


def safe_replace(src: Path, dst: Path) -> None:
    if dst.exists():
        dst.unlink()
    src.replace(dst)


def _retrieve_with_retry(client, dataset, req, out_path, attempts=3, sleep_s=10) -> bool:
    last_err = None
    for k in range(attempts):
        try:
            client.retrieve(dataset, req, out_path)
            if already_ok(Path(out_path)):
                return True
        except Exception as e:
            last_err = e
        time.sleep(sleep_s * (k + 1))
    if last_err:
        print(f"ERROR: retrieve failed after {attempts} attempts: {last_err}")
    return False


def cleanup_keep_only_final(year: int, month: str) -> None:
    keep = f"{year}_{month}_raw.nc"
    for p in cfg.DIR_DL.glob(f"{year}_{month}*"):
        if p.name == keep:
            continue
        try:
            if p.is_file():
                p.unlink()
        except Exception:
            pass


# ========== QC/merge helpers ==========
def _first_var_name(ds):
    for v in ("t2m", "T2M", "t", "temperature", "air_temperature"):
        if v in ds.data_vars:
            return v
    return list(ds.data_vars)[0]


def _open_nc_any_engine(nc_path: Path):
    if not _HAS_XARRAY:
        raise RuntimeError("xarray is required to open NetCDF for merge/QC.")
    try:
        return xr.open_dataset(nc_path, engine="h5netcdf")
    except Exception:
        try:
            return xr.open_dataset(nc_path, engine="netcdf4")
        except Exception:
            return xr.open_dataset(nc_path, engine="scipy")


def _step_to_timedelta64(step_da):
    import numpy as np

    s = step_da
    try:
        if str(s.dtype).startswith("timedelta64"):
            return s.astype("timedelta64[ns]")
    except Exception:
        pass

    units = str(getattr(s, "attrs", {}).get("units", "")).lower()

    try:
        ivals = s.astype("int64").values
    except Exception:
        try:
            ivals = s.values.astype("int64")
        except Exception:
            ivals = np.arange(s.size, dtype="int64")

    if ("hour" in units) or (units == "h"):
        td = ivals * np.timedelta64(1, "h")
    elif ("min" in units) or (units == "m"):
        td = ivals * np.timedelta64(1, "m")
    elif ("sec" in units) or (units == "s"):
        td = ivals * np.timedelta64(1, "s")
    else:
        td = ivals * np.timedelta64(1, "h")

    return xr.DataArray(td.astype("timedelta64[ns]"), dims=s.dims, coords=s.coords, attrs=s.attrs)


def _ensure_time_coord(ds):
    if not _HAS_XARRAY:
        return ds

    try:
        ds = xr.decode_cf(ds)
    except Exception:
        pass

    if "time" not in ds.dims and "valid_time" in ds.dims:
        ds = ds.rename({"valid_time": "time"})
        if "valid_time" in ds.coords and "time" not in ds.coords:
            ds = ds.rename({"valid_time": "time"})

    if "time" in ds.dims and "step" in ds.dims and ds.sizes.get("step", 1) > 1:
        try:
            if "valid_time" in ds.coords and set(ds["valid_time"].dims) >= {"time", "step"}:
                vt2d = ds["valid_time"]
            else:
                step_td = _step_to_timedelta64(ds["step"])
                vt2d = ds["time"].astype("datetime64[ns]") + step_td

            vt1d = vt2d.stack(tstep=("time", "step")).values

            ds2 = ds.stack(tstep=("time", "step"))
            ds2 = ds2.reset_index("tstep", drop=True)
            ds2 = ds2.assign_coords(time=("tstep", vt1d))
            ds2 = ds2.swap_dims({"tstep": "time"})

            if "tstep" in ds2.variables:
                ds2 = ds2.drop_vars("tstep")

            for c in ("step", "valid_time"):
                if c in ds2.variables:
                    try:
                        ds2 = ds2.drop_vars(c)
                    except Exception:
                        pass

            ds = ds2
        except Exception:
            pass

    if "time" in ds.coords:
        import numpy as np
        try:
            ds = ds.sortby("time")
            tvals = ds["time"].values
            _, idx = np.unique(tvals, return_index=True)
            if len(idx) != len(tvals):
                ds = ds.isel(time=np.sort(idx))
        except Exception:
            pass

    return ds


def merge_month_parts(year: int, month: str, parts: list[Path], out_nc: Path) -> Path:
    if not _HAS_XARRAY:
        raise RuntimeError(f"xarray is required to merge multiple parts ({len(parts)}) for {year}-{month}")

    dsets = []
    for p in sorted(parts):
        try:
            ds = _open_nc_any_engine(p)
            ds = _ensure_time_coord(ds)
            v = _first_var_name(ds)
            if v != "t2m":
                ds = ds.rename({v: "t2m"})
            dsets.append(ds)
        except Exception as e:
            print(f"   WARN: skip part {p.name} (open fail): {e}")

    if not dsets:
        raise RuntimeError(f"No valid parts to merge for {year}-{month}")

    ds_all = xr.concat(dsets, dim="time", combine_attrs="override")
    ds_all = _ensure_time_coord(ds_all)

    tmp = out_nc.with_suffix(".nc.tmp")
    encoding = {
        "t2m": {"zlib": True, "complevel": 4, "shuffle": True},
        "time": {"zlib": True, "complevel": 4, "shuffle": True},
    }

    wrote = False
    for engine in ("h5netcdf", "netcdf4", None):
        try:
            if engine is None:
                ds_all.to_netcdf(tmp, encoding=encoding)
            else:
                ds_all.to_netcdf(tmp, engine=engine, encoding=encoding)
            wrote = True
            break
        except Exception:
            continue
    if not wrote:
        raise RuntimeError(f"Failed to write merged NetCDF for {year}-{month}: {out_nc.name}")

    safe_replace(tmp, out_nc)

    for ds in dsets:
        try:
            ds.close()
        except Exception:
            pass

    print(f"OK: merged -> {out_nc.name}")
    return out_nc


# ========== GRIB→NetCDF ==========
def _convert_grib_to_nc(grib_path: Path) -> Path:
    nc_path = grib_path.with_suffix(".nc")

    # ecCodes CLI
    try:
        subprocess.run(
            ["grib_to_netcdf", "-o", str(nc_path), str(grib_path)],
            check=True,
            capture_output=True,
        )
        grib_path.unlink(missing_ok=True)
        return nc_path
    except Exception:
        pass

    # xarray+cfgrib fallback
    if not _HAS_XARRAY:
        print(f"WARN: cfgrib fallback unavailable (xarray missing) for {grib_path.name}")
        return grib_path

    try:
        ds = xr.open_dataset(str(grib_path), engine="cfgrib", backend_kwargs={"indexpath": ""})
        ds.to_netcdf(str(nc_path))
        ds.close()
        grib_path.unlink(missing_ok=True)
        print(f"OK: converted {grib_path.name} -> {nc_path.name}")
        return nc_path
    except Exception as e:
        print(f"ERROR: GRIB->NC failed for {grib_path.name}: {e}")
        return grib_path


# ========== ZIP handling ==========
def unzip_and_collect(zip_path: Path) -> list[Path]:
    outs: list[Path] = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.namelist() or []
        if not members:
            print(f"WARN: empty archive: {zip_path.name}")
            return outs

        for k, name in enumerate(members):
            ext = Path(name).suffix.lower()
            if ext not in (".nc", ".grib", ".grb", ".grib2"):
                continue

            out_name = (
                f"{zip_path.stem}_{k:03d}_raw.nc"
                if ext == ".nc"
                else f"{zip_path.stem}_{k:03d}_raw{ext}"
            )
            dst = zip_path.with_name(out_name)

            if not already_ok(dst):
                with zf.open(name) as fi, open(dst, "wb") as fo:
                    shutil.copyfileobj(fi, fo)

            if ext in (".grib", ".grb", ".grib2"):
                dst = _convert_grib_to_nc(dst)

            outs.append(dst)

    outs_nc = [p for p in outs if p.suffix.lower() == ".nc" and already_ok(p)]

    for p in outs:
        if p.suffix.lower() in (".grib", ".grb", ".grib2") and p.exists():
            try:
                p.unlink()
            except Exception:
                pass

    print(f"OK: unzip {zip_path.name}: {len(outs_nc)} netcdf")
    return outs_nc


# ========== CDS download ==========
def cds_retrieve(year: int, month: str, nocache: bool = False) -> Path:
    try:
        import cdsapi
    except ImportError:
        raise SystemExit("Missing dependency: cdsapi (conda install -c conda-forge cdsapi)")

    ensure_dir(cfg.DIR_DL)

    nc_out = target_path(year, month, ".nc")
    zip_out = target_path(year, month, ".zip")

    if already_ok(nc_out):
        fmt = _sniff_format(nc_out)
        if fmt == "zip":
            if zip_out.exists():
                try:
                    zip_out.unlink()
                except Exception:
                    pass
            nc_out.rename(zip_out)
            return zip_out
        return nc_out

    if already_ok(zip_out):
        return zip_out

    bbox = cfg.rounded_bbox()  # [N, W, S, E]

    base_req = {
        "variable": [cfg.CDS_VAR],
        "year": [str(year)],
        "month": [month],
        "day": month_days(year, month),
        "time": [f"{h:02d}:00" for h in range(24)],
        "area": bbox,
        "grid": cfg.DL_GRID,
    }
    if nocache:
        base_req["nocache"] = str(int(time.time() * 1000))

    c = cdsapi.Client(
        timeout=int(getattr(cfg, "CDS_TIMEOUT", 600)),
        verify=bool(getattr(cfg, "CDS_VERIFY_SSL", True)),
    )

    req_new = {
        **base_req,
        "data_format": str(getattr(cfg, "CDS_FORMAT", "netcdf")),
        "download_format": "unarchived",
    }
    print(f"DOWN: CDS {year}-{month} (data_format={req_new['data_format']}) ...")
    if _retrieve_with_retry(c, cfg.CDS_DATASET, req_new, str(nc_out), attempts=3, sleep_s=10):
        fmt = _sniff_format(nc_out)
        if fmt == "zip":
            if zip_out.exists():
                try:
                    zip_out.unlink()
                except Exception:
                    pass
            nc_out.rename(zip_out)
            print(f"INFO: delivered ZIP; using {zip_out.name}")
            return zip_out
        print(f"OK: downloaded {nc_out.name}")
        return nc_out

    req_old_nc = {**base_req, "format": "netcdf"}
    print(f"DOWN: CDS {year}-{month} legacy (format=netcdf) ...")
    if _retrieve_with_retry(c, cfg.CDS_DATASET, req_old_nc, str(nc_out), attempts=3, sleep_s=10):
        fmt = _sniff_format(nc_out)
        if fmt == "zip":
            if zip_out.exists():
                try:
                    zip_out.unlink()
                except Exception:
                    pass
            nc_out.rename(zip_out)
            print(f"INFO: delivered ZIP; using {zip_out.name}")
            return zip_out
        print(f"OK: downloaded {nc_out.name}")
        return nc_out

    req_old_zip = {**base_req, "format": "zip"}
    print(f"DOWN: CDS {year}-{month} legacy (format=zip) ...")
    if _retrieve_with_retry(c, cfg.CDS_DATASET, req_old_zip, str(zip_out), attempts=3, sleep_s=10):
        print(f"OK: downloaded {zip_out.name}")
        return zip_out

    raise RuntimeError(f"CDS retrieval failed for {year}-{month}")


# ========== QC (non-invasive) ==========
def _qc_print(nc_path: Path, year: int, month: str):
    if not _HAS_XARRAY:
        return
    try:
        ds = _open_nc_any_engine(nc_path)
        v = _first_var_name(ds)
        units = ds[v].attrs.get("units") or ds[v].attrs.get("unit")

        la = next(d for d in ds.dims if d.lower() in ("lat", "latitude", "y"))
        lo = next(d for d in ds.dims if d.lower() in ("lon", "longitude", "x"))

        tdim = "time" if "time" in ds.dims else ("valid_time" if "valid_time" in ds.dims else None)
        ntime = ds.sizes.get(tdim, None) if tdim else None

        print(f"QC: {year}-{month} grid={ds.sizes[la]}x{ds.sizes[lo]} var='{v}' units='{units}' tdim={tdim} n={ntime}")

        if ntime is not None:
            expected = calendar.monthrange(year, int(month))[1] * 24
            if int(ntime) != int(expected):
                print(f"QC WARN: {year}-{month} expected_hours={expected} but got={ntime}")

        ds.close()
    except Exception as e:
        print(f"QC WARN: {year}-{month}: {e}")


# ========== MAIN ==========
def main():
    args = parse_args()

    # FORCE toggles: CLI > ENV > cfg
    force_env = str(os.environ.get("FORCE", "0")).strip().lower() in ("1", "true", "yes")
    FORCE = bool(args.force or getattr(cfg, "FORCE", False) or force_env)

    # Years/months: CLI > ENV > cfg
    years = _parse_years_spec(args.years) or _parse_years_spec(os.environ.get("YEARS")) or list(cfg.YEARS)
    months = _parse_months_spec(args.months) or _parse_months_spec(os.environ.get("MONTHS")) or list(cfg.MONTHS)

    try:
        cfg.save_run_meta()
    except Exception:
        pass

    ensure_dir(cfg.DIR_DL)

    for y in years:
        for m in months:
            month_nc_final = cfg.DIR_DL / f"{y}_{m}_raw.nc"

            if month_nc_final.exists() and not _is_netcdf_file(month_nc_final):
                print(f"WARN: repairing mislabeled file -> {month_nc_final.name}")
                try:
                    month_nc_final.unlink()
                except Exception:
                    pass

            if month_done(y, m) and not FORCE:
                cleanup_keep_only_final(y, m)
                print(f"SKIP: {y}-{m}: {month_nc_final.name} already present")
                continue

            try:
                out = cds_retrieve(y, m, nocache=FORCE)

                if out.suffix.lower() == ".nc":
                    raw_nc = out.with_name(f"{out.stem}_raw.nc")
                    if not raw_nc.exists():
                        safe_replace(out, raw_nc)
                        print(f"OK: saved {raw_nc.name}")
                    parts = [raw_nc]
                else:
                    parts = unzip_and_collect(out)
                    try:
                        if out.exists():
                            out.unlink()
                    except Exception:
                        pass

                if not parts:
                    print(f"WARN: no usable NetCDF for {y}-{m}")
                    continue

                if len(parts) == 1:
                    src = parts[0]
                    if month_nc_final.resolve() != src.resolve():
                        tmp = month_nc_final.with_suffix(".nc.tmp")
                        shutil.copyfile(src, tmp)
                        safe_replace(tmp, month_nc_final)
                else:
                    month_nc_final = merge_month_parts(y, m, parts, month_nc_final)

                _qc_print(month_nc_final, y, m)
                cleanup_keep_only_final(y, m)

            except Exception as e:
                print(f"ERROR: {y}-{m} failed: {e}")


if __name__ == "__main__":
    main()
