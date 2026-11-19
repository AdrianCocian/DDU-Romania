"""Steps 02 -> 03 -> 03b -> 04 -> 05 from monthly files in the form delivered by the Copernicus CDS.

Checks the chain that turns downloads into seasons: unit conversion, clipping to the national boundary, yearly
concatenation, completeness step, tails, and the seasons themselves against the independent implementation.
Needs the geospatial libraries of environment.yml (geopandas, rioxarray); skipped when they are not installed.
"""
import os

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import independent as ind
from conftest import REPO, SHP, run_script
from synthetic import R_LATS, R_LONS, make_raw_months, raw_month

pytestmark = pytest.mark.slow
gpd = pytest.importorskip("geopandas")
pytest.importorskip("rioxarray")

YEARS = (2005, 2006)


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    root = make_raw_months(tmp_path_factory.mktemp("raw_months"), YEARS)
    env = dict(os.environ, DDU_SHP_RO=str(SHP))
    r = run_script([REPO / "run_pipeline.py", "--root", root, "--years", "2005-2006", "--from", "2", "--to", "5"], env=env)
    assert r.returncode == 0, (r.stdout[-3000:], r.stderr[-3000:])
    return root


def _touched_by_boundary():
    """Cells of the test grid that the national boundary polygon touches (computed with shapely, not with step02)."""
    from shapely.geometry import box
    from shapely.prepared import prep
    polygon = prep(gpd.read_file(SHP).to_crs(4326).union_all())
    return {(float(y), float(x)) for y in R_LATS for x in R_LONS
            if polygon.intersects(box(x - 0.05, y - 0.05, x + 0.05, y + 0.05))}


def test_clip_keeps_the_cells_touched_by_the_boundary_and_converts_to_celsius(run):
    with xr.open_dataset(run / "out" / "06_filled_yearly" / "2005.nc") as ds:
        t = ds["t2m"].load()
        qc = ds["t2m_qc"].load()
    assert t.sizes["time"] == 8760 and str(t.attrs.get("units")) == "degree_Celsius"
    assert list(t["lat"].values) == sorted(t["lat"].values)                      # latitude ascending after step02
    has_data = {(round(float(y), 1), round(float(x), 1)) for y in t["lat"].values for x in t["lon"].values
                if bool(np.isfinite(t.sel(lat=y, lon=x).values).any())}
    touched = _touched_by_boundary()
    assert has_data == touched and 0 < len(touched) < R_LATS.size * R_LONS.size   # part of the grid lies outside Romania
    assert int((qc.values != 0).sum()) == 0                                       # nothing was filled

    january = raw_month(2005, 1)["t2m"]                                           # Kelvin, latitude descending
    y, x = sorted(touched)[len(touched) // 2]
    expected = january.sel(latitude=y, longitude=x, method="nearest").values.astype("float32") - np.float32(273.15)
    got = t.sel(lat=y, lon=x, method="nearest").values[: expected.size]
    assert np.allclose(got, expected, atol=1e-4)


def test_yearly_file_and_filled_file_hold_the_same_temperatures(run):
    for y in YEARS:
        with xr.open_dataset(run / "out" / "04_yearly_raw" / f"{y}.nc") as a, \
                xr.open_dataset(run / "out" / "06_filled_yearly" / f"{y}.nc") as b:
            va, vb = a["t2m"].values, b["t2m"].values
        assert va.shape == vb.shape
        assert np.array_equal(np.isnan(va), np.isnan(vb)) and np.array_equal(np.nan_to_num(va), np.nan_to_num(vb))
    rep = pd.read_csv(run / "out" / "07_reports" / "qc_gaps_summary.csv")
    assert rep["year"].tolist() == list(YEARS) and (rep["missing_hours_inside_before"] == 0).all()
    assert (run / "out" / "06_filled_yearly_tails_29d" / "2005_tail29d.nc").exists()


def test_seasons_equal_the_independent_implementation_applied_to_the_downloads(run):
    df = pd.read_csv(run / "out" / "08_season_swrap" / "win30" / "csv" / "season_ALL_YEARS_methods.csv")
    summary = pd.read_csv(run / "out" / "08_season_swrap" / "win30" / "csv" / "summary_ALL_YEARS_methods.csv")
    assert dict(zip(summary["year"], summary["year_wrap_used"].astype(int))) == {2005: 0, 2006: 1}
    touched = _touched_by_boundary()
    assert len(df) == len(YEARS) * 4 * len(touched)
    previous, wrong = None, []
    for y in YEARS:
        months = [raw_month(y, m)["t2m"] for m in range(1, 13)]                   # straight from the "downloads"
        kelvin = np.concatenate([m.values for m in months], axis=0)
        ddu = ind.daily_ddu(kelvin.astype("float32") - np.float32(273.15))
        for m, thr in ind.THRESHOLD.items():
            suit = ind.suitable_days(ddu[m], None if previous is None else previous[m], thr)
            for r in df[(df["year"] == y) & (df["method"] == m)].itertuples(index=False):
                a = int(np.argmin(np.abs(R_LATS - r.lat)))
                b = int(np.argmin(np.abs(R_LONS - r.lon)))
                got = None if pd.isna(r.season_start_doy) else (int(r.season_start_doy), int(r.season_end_doy),
                                                                int(r.season_length), int(r.total_eip_days), int(r.runs_count))
                exp = ind.season(suit[:, a, b], leap=False)
                if got != exp:
                    wrong.append((y, m, r.lat, r.lon, got, exp))
        previous = ddu
    assert not wrong, wrong[:5]
