"""Small synthetic ERA5-Land-like dataset for end-to-end tests (no download needed).

Writes, under <root>/out/:
  06_filled_yearly/{YYYY}.nc                hourly 2 m temperature (deg C), dims time x latitude x longitude
  06_filled_yearly_tails_29d/{YYYY}_tail29d.nc   last 29 days of each year (year-wrap input of the next year)

The grid (3 x 4 cells, 0.1 deg) lies inside Romania near 45 N, 25 E; one cell is NaN for the whole
period (outside the mask), as in the real clipped files.
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
