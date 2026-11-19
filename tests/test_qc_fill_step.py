"""Step 03b: completeness check of the yearly files and creation of the "filled yearly" files.

For a gap-free reanalysis the step must change nothing: t2m identical to the yearly file and every flag 0.
The filling branches (short gap, climatology, neighbours) are tested on gaps introduced on purpose.
"""
import os

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from conftest import REPO, SHP, run_script

LAT, LON = np.array([45.0, 45.1, 45.2]), np.array([25.0, 25.1])


def _year(year, seed=1):
    time = pd.date_range(f"{year}-01-01", f"{year}-12-31 23:00", freq="h")
    rng = np.random.default_rng(seed)
    season = (10 + 10 * np.sin(np.arange(time.size) / 8760 * 2 * np.pi))[:, None, None]
    t2m = (season + rng.normal(0, 1, (time.size, 3, 2))).astype("float32")
    t2m[:, 0, 0] = np.nan                                   # outside the national mask
    return time, t2m


def _write(path, time, t2m):
    path.parent.mkdir(parents=True, exist_ok=True)
    xr.Dataset({"t2m": (("time", "lat", "lon"), t2m)}, coords={"time": time, "lat": LAT, "lon": LON}).to_netcdf(path)


def _read(path):
    with xr.open_dataset(path) as ds:                       # closed again at once (Windows cannot delete an open file)
        ds.load()
        return ds["t2m"].values, ds["t2m_qc"].values, ds["dayofyear"].values, ds["hour"].values


@pytest.fixture()
def dirs(step03b, tmp_path, monkeypatch):
    raw, filled = tmp_path / "out" / "04_yearly_raw", tmp_path / "out" / "06_filled_yearly"
    monkeypatch.setattr(step03b, "DIR_SRC", raw)
    monkeypatch.setattr(step03b, "DIR_DST", filled)
    monkeypatch.setattr(step03b, "DIR_EXISTING", filled)
    return raw, filled


def test_complete_year_is_left_unchanged(step03b, dirs):
    raw, filled = dirs
    time, t2m = _year(2008)                                  # leap year: 8,784 hours
    _write(raw / "2008.nc", time, t2m)
    row = step03b.process_year(2008, [2008], 24, True)
    t_out, qc, doy, hour = _read(filled / "2008.nc")
    assert row["status"] == "ok" and row["filled_file_check"] == "written"
    assert row["hours"] == 8784 and row["cells_inside"] == 5 and row["missing_hours_inside_before"] == 0
    assert np.array_equal(np.isnan(t_out), np.isnan(t2m)) and np.array_equal(np.nan_to_num(t_out), np.nan_to_num(t2m))
    assert (qc == 0).all()
    assert np.isnan(t_out[:, 0, 0]).all()                    # cells outside the mask stay empty
    assert (int(doy[0]), int(doy[-1]), int(hour[-1])) == (1, 365, 23)
    feb29, mar1 = time.get_loc("2008-02-29 00:00"), time.get_loc("2008-03-01 00:00")
    assert int(doy[feb29]) == 59 and int(doy[mar1]) == 60    # 29 February shares day 59; no hour is dropped


def test_short_gap_is_interpolated_in_time(step03b, dirs):
    raw, filled = dirs
    time, t2m = _year(2005)
    truth = t2m.copy()
    t2m[100:106, 1, 1] = np.nan                              # 6 missing hours
    _write(raw / "2005.nc", time, t2m)
    row = step03b.process_year(2005, [2005], 24, True)
    t_out, qc, _, _ = _read(filled / "2005.nc")
    assert (row["missing_hours_inside_before"], row["filled_shortgap"], row["missing_after"]) == (6, 6, 0)
    assert (qc[100:106, 1, 1] == 1).all() and int((qc != 0).sum()) == 6
    x0, x1 = float(truth[99, 1, 1]), float(truth[106, 1, 1])
    expected = x0 + (x1 - x0) * np.arange(1, 7) / 7.0        # straight line between the two neighbouring hours
    assert np.allclose(t_out[100:106, 1, 1], expected, atol=1e-5)
    rest = np.ones(t2m.shape, bool)
    rest[100:106, 1, 1] = False
    assert np.array_equal(np.nan_to_num(t_out[rest]), np.nan_to_num(truth[rest]))


def test_long_gap_uses_the_climatology_of_the_other_years(step03b, dirs):
    raw, filled = dirs
    time5, t5 = _year(2005, seed=1)
    time6, t6 = _year(2006, seed=2)
    time7, t7 = _year(2007, seed=3)
    t5[2000:2100, 2, 0] = np.nan                             # 100 missing hours: longer than the 24-hour limit
    _write(raw / "2005.nc", time5, t5)
    _write(raw / "2006.nc", time6, t6)
    _write(raw / "2007.nc", time7, t7)
    row = step03b.process_year(2005, [2005, 2006, 2007], 24, True)
    t_out, qc, _, _ = _read(filled / "2005.nc")
    assert (row["filled_shortgap"], row["filled_climatology"], row["filled_spatial"], row["missing_after"]) == (0, 100, 0, 0)
    assert (qc[2000:2100, 2, 0] == 2).all()
    expected = (t6[2000:2100, 2, 0].astype("float64") + t7[2000:2100, 2, 0]) / 2.0     # same day of year and hour
    assert np.allclose(t_out[2000:2100, 2, 0], expected, atol=1e-4)


def test_long_gap_without_other_years_uses_the_neighbouring_cells(step03b, dirs):
    raw, filled = dirs
    time, t2m = _year(2005)
    t2m[2000:2100, 2, 0] = np.nan
    _write(raw / "2005.nc", time, t2m)
    row = step03b.process_year(2005, [2005], 24, True)
    t_out, qc, _, _ = _read(filled / "2005.nc")
    assert (row["filled_climatology"], row["filled_spatial"], row["missing_after"]) == (0, 100, 0)
    assert (qc[2000:2100, 2, 0] == 3).all()
    neighbours = np.stack([t2m[2000:2100, 1, 0], t2m[2000:2100, 1, 1], t2m[2000:2100, 2, 1]]).astype("float64")
    assert np.allclose(t_out[2000:2100, 2, 0], neighbours.mean(axis=0), atol=1e-4)


def test_report_only_compares_and_writes_no_netcdf(step03b, dirs, tmp_path):
    raw, filled = dirs
    time, t2m = _year(2005)
    _write(raw / "2005.nc", time, t2m)
    assert step03b.process_year(2005, [2005], 24, False, report_only=True)["filled_file_check"] == "absent"
    assert not filled.exists()
    step03b.process_year(2005, [2005], 24, True)
    f = filled / "2005.nc"
    before = (f.stat().st_mtime_ns, f.stat().st_size)
    rep = step03b.process_year(2005, [2005], 24, False, report_only=True)
    assert rep["filled_file_check"] == "identical" and rep["missing_hours_inside_before"] == 0
    assert (f.stat().st_mtime_ns, f.stat().st_size) == before

    with xr.open_dataset(f) as ds:
        changed = ds.load()
    changed["t2m"].values[5000, 1, 1] += 0.5
    changed.to_netcdf(tmp_path / "changed.nc")
    changed.close()
    (tmp_path / "changed.nc").replace(f)
    assert step03b.process_year(2005, [2005], 24, False, report_only=True)["filled_file_check"] == "DIFFERENT (t2m)"


def test_input_root_checks_another_folder_without_writing_into_it(tmp_path):
    """The command used by tools/verify_release.py on the folder of the published run."""
    data, work = tmp_path / "data", tmp_path / "work"
    time, t2m = _year(2005)
    _write(data / "out" / "04_yearly_raw" / "2005.nc", time, t2m)
    env = dict(os.environ, DDU_SHP_RO=str(SHP))
    env.pop("DDU_ROOT", None)
    script = REPO / "scripts" / "step03b_qc_fill_year.py"
    # build the filled file inside the data folder (as the original run did), then check it from outside
    r = run_script([script, "--root", data, "--years", "2005"], env=env)
    assert r.returncode == 0, (r.stdout + r.stderr)[-1500:]
    snapshot = {p.relative_to(data).as_posix(): (p.stat().st_mtime_ns, p.stat().st_size) for p in data.rglob("*") if p.is_file()}

    r = run_script([script, "--root", work, "--input-root", data, "--years", "2005", "--report-only"], env=env)
    out = r.stdout + r.stderr
    assert r.returncode == 0, out[-1500:]
    assert "existing filled files identical to the output of this step: 1/1" in out
    assert "0 missing hour(s) inside the mask" in out
    after = {p.relative_to(data).as_posix(): (p.stat().st_mtime_ns, p.stat().st_size) for p in data.rglob("*") if p.is_file()}
    assert after == snapshot                                 # nothing created, changed or removed in the data folder
    rep = pd.read_csv(work / "out" / "07_reports" / "qc_gaps_summary.csv")
    assert rep.loc[0, "filled_file_check"] == "identical" and int(rep.loc[0, "missing_hours_inside_before"]) == 0
    assert not (work / "out" / "06_filled_yearly" / "2005.nc").exists()
