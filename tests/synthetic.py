"""Small synthetic ERA5-Land-like datasets for the tests (no download needed).

make_synthetic_root   smooth climate, 3 x 4 cells (one outside the mask); writes, under <root>/out/:
                        06_filled_yearly/{YYYY}.nc                     hourly 2 m temperature (deg C)
                        06_filled_yearly_tails_29d/{YYYY}_tail29d.nc   last 29 days of each year
make_varied_root      weather-like climate, 4 x 6 cells from cold to warm with warm and cold spells: cells without
                      a season, seasons made of several runs, runs joined across one or two unsuitable days;
                      writes 06_filled_yearly/{YYYY}.nc only (the tails are left to step04)
make_raw_months       monthly files in the form delivered by the Copernicus CDS (t2m in K, valid_time, latitude
                      descending) on a grid that crosses the southern border; writes 01_download/{YYYY}_{MM}_raw.nc
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import xarray as xr

LATS = np.array([45.2, 45.1, 45.0])          # descending, as in ERA5
LONS = np.array([25.0, 25.1, 25.2, 25.3])


def hourly_temperature(year: int, seed: int = 0) -> xr.DataArray:
    """Seasonal + diurnal cycle, warmer to the south-east, +0.3 deg C per year, small noise."""
    time = pd.date_range(f"{year}-01-01 00:00", f"{year}-12-31 23:00", freq="h")
    doy = time.dayofyear.to_numpy(dtype=float)
    hour = time.hour.to_numpy(dtype=float)
    seasonal = 10.0 + 14.0 * np.sin(2 * np.pi * (doy - 110.0) / 365.0)
    diurnal = 5.0 * np.sin(2 * np.pi * (hour - 9.0) / 24.0)
    base = (seasonal + diurnal + 0.3 * (year - 2005))[:, None, None]
    lat_off = (45.2 - LATS)[None, :, None] * 10.0          # 0, 1, 2 deg C from north to south
    lon_off = (LONS - 25.0)[None, None, :] * 5.0           # 0 ... 1.5 deg C from west to east
    rng = np.random.default_rng(seed + year)
    t = base + lat_off + lon_off + rng.normal(0.0, 0.5, size=(time.size, LATS.size, LONS.size))
    t[:, 0, 0] = np.nan                                   # outside the country mask
    return xr.DataArray(t.astype("float32"), coords={"time": time, "latitude": LATS, "longitude": LONS},
                        dims=("time", "latitude", "longitude"), name="t2m", attrs={"units": "degC"})


def make_synthetic_root(root: Path, years: Iterable[int] = (2005, 2006, 2007, 2008)) -> Path:
    root = Path(root)
    d_year = root / "out" / "06_filled_yearly"
    d_tail = root / "out" / "06_filled_yearly_tails_29d"
    d_year.mkdir(parents=True, exist_ok=True)
    d_tail.mkdir(parents=True, exist_ok=True)
    for y in years:
        da = hourly_temperature(int(y))
        da.to_dataset().to_netcdf(d_year / f"{y}.nc")
        tail = da.isel(time=slice(da.sizes["time"] - 29 * 24, None))
        tail.to_dataset().to_netcdf(d_tail / f"{y}_tail29d.nc")
    return root


# ------------------------------------------------------------------ varied climate
V_LATS = np.array([47.0, 46.5, 46.0, 45.5])              # north -> south
V_LONS = np.array([22.0, 22.5, 23.0, 23.5, 24.0, 24.5])


def _ar1(rng, n: int, phi: float, sd: float) -> np.ndarray:
    e = rng.normal(0.0, sd * np.sqrt(1.0 - phi ** 2), n)
    x = np.empty(n)
    x[0] = rng.normal(0.0, sd)
    for k in range(1, n):
        x[k] = phi * x[k - 1] + e[k]
    return x


def varied_temperature(year: int) -> xr.DataArray:
    """Seasonal and diurnal cycle plus persistent warm and cold spells; cold north-west to warm south-east."""
    time = pd.date_range(f"{year}-01-01 00:00", f"{year}-12-31 23:00", freq="h")
    nd = time.size // 24
    doy = np.repeat(np.arange(nd, dtype=float), 24)
    hour = time.hour.to_numpy(dtype=float)
    rng = np.random.default_rng(1000 + year)
    seasonal = 9.0 + 12.5 * np.sin(2 * np.pi * (doy - 105.0) / 365.0)
    spells = np.repeat(_ar1(rng, nd, 0.75, 3.2), 24)
    w = 2 * np.pi * (hour - 9.0) / 24.0
    # asymmetric daily cycle (short warm afternoon, long cool night): the mean of the 24 hours differs from (min + max) / 2
    diurnal = (4.5 + 2.0 * np.sin(2 * np.pi * (doy - 105.0) / 365.0)) * (np.sin(w) + 0.35 * np.sin(2 * w + 1.0))
    t = np.empty((time.size, V_LATS.size, V_LONS.size), dtype="float32")
    for a in range(V_LATS.size):
        for b in range(V_LONS.size):
            offset = -8.0 + 3.0 * a + 0.7 * b + 0.25 * (year - 2005)
            local = np.repeat(_ar1(rng, nd, 0.6, 1.2), 24)
            t[:, a, b] = seasonal + offset + spells + local + diurnal + rng.normal(0.0, 0.3, time.size)
    t[:, 0, 0] = np.nan                                   # outside the country mask
    return xr.DataArray(t, coords={"time": time, "latitude": V_LATS, "longitude": V_LONS},
                        dims=("time", "latitude", "longitude"), name="t2m", attrs={"units": "degree_Celsius"})


def make_varied_root(root: Path, years: Iterable[int] = (2005, 2006, 2007, 2008)) -> Path:
    root = Path(root)
    d_year = root / "out" / "06_filled_yearly"
    d_year.mkdir(parents=True, exist_ok=True)
    for y in years:
        varied_temperature(int(y)).to_dataset().to_netcdf(d_year / f"{y}.nc")
    return root


# ------------------------------------------------------------------ monthly files as downloaded
R_LATS = np.round(np.arange(44.4, 43.39, -0.1), 1)       # 44.4 ... 43.4: crosses the Danube (the southern border)
R_LONS = np.round(np.arange(25.0, 26.01, 0.1), 1)


def raw_month(year: int, month: int) -> xr.Dataset:
    time = pd.date_range(f"{year}-{month:02d}-01", periods=pd.Period(f"{year}-{month:02d}").days_in_month * 24, freq="h")
    doy, hour = time.dayofyear.to_numpy(float), time.hour.to_numpy(float)
    base = 11.0 + 13.0 * np.sin(2 * np.pi * (doy - 110.0) / 365.0) + 5.0 * np.sin(2 * np.pi * (hour - 9.0) / 24.0)
    rng = np.random.default_rng(year * 100 + month)
    arr = base[:, None, None] + (44.4 - R_LATS)[None, :, None] * 2.0 + rng.normal(0.0, 0.5, (time.size, R_LATS.size, R_LONS.size))
    ds = xr.Dataset({"t2m": (("valid_time", "latitude", "longitude"), (arr + 273.15).astype("float32"))},
                    coords={"valid_time": time, "latitude": R_LATS, "longitude": R_LONS})
    ds["t2m"].attrs.update(units="K", long_name="2 metre temperature")
    return ds


def make_raw_months(root: Path, years: Iterable[int] = (2005, 2006)) -> Path:
    root = Path(root)
    d = root / "out" / "01_download"
    d.mkdir(parents=True, exist_ok=True)
    for y in years:
        for m in range(1, 13):
            raw_month(int(y), m).to_netcdf(d / f"{y}_{m:02d}_raw.nc")
    return root
