"""Steps 04 (tails) and 05 (DDU, 30-day window, 3-day rule, year wrap) on temperatures with a known answer.

Daily DDU at a constant 20 degC is 6 for every method (144 degree-hours for the hourly sum), so the 30-day sum
reaches 130 DDU (3120 degree-hours) when the window holds at least 22 warm days. The window needs 30 full days:
without the tail of the previous year the first possible suitable day is 30 January; with it, 1 January.

Years 2005-2008 (2008 is a leap year); grid 3 x 2:

cell (0, 0)  20 degC all year              2005: DOY 30-365 (336 d)      2006, 2007: DOY 1-365 (365 d)
                                           2008: DOY 1-365, 366 d  <- 29 February is a day like any other;
                                                 only the numbering of the days is on a 365-day basis
cell (0, 1)  10 degC all year              no season
cell (1, 0)  20 degC 1 June - 31 August    DOY 173-251 (79 d) in every year
cell (1, 1)  20 degC on 1-25 January and 7-31 December
             2005: runs DOY 30-33 and 362-365 (season 4 d, 8 suitable days)
             2006-2008: runs DOY 1-33 and 362-365 (season 33 d, 37 suitable days)  <- year wrap
cell (2, 0)  no data (outside the mask)
cell (2, 1)  20 degC all year              as cell (0, 0)
"""
import json
import os
import shutil

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from conftest import REPO, SHP, run_script

METHODS = ["midpoint", "daily_mean", "hourly_sum", "be_sine"]

# (year, row, column): (start_doy, end_doy, season_length, total_eip_days, runs_count)
EXPECTED = {
    (2005, 0, 0): (30, 365, 336, 336, 1), (2006, 0, 0): (1, 365, 365, 365, 1),
    (2005, 0, 1): (None, None, 0, 0, 0), (2006, 0, 1): (None, None, 0, 0, 0),
    (2005, 1, 0): (173, 251, 79, 79, 1), (2006, 1, 0): (173, 251, 79, 79, 1),
    (2005, 1, 1): (30, 33, 4, 8, 2), (2006, 1, 1): (1, 33, 33, 37, 2),
    (2005, 2, 1): (30, 365, 336, 336, 1), (2006, 2, 1): (1, 365, 365, 365, 1),
    (2007, 0, 0): (1, 365, 365, 365, 1), (2008, 0, 0): (1, 365, 366, 366, 1),
    (2007, 0, 1): (None, None, 0, 0, 0), (2008, 0, 1): (None, None, 0, 0, 0),
    (2007, 1, 0): (173, 251, 79, 79, 1), (2008, 1, 0): (173, 251, 79, 79, 1),
    (2007, 1, 1): (1, 33, 33, 37, 2), (2008, 1, 1): (1, 33, 33, 37, 2),
    (2007, 2, 1): (1, 365, 365, 365, 1), (2008, 2, 1): (1, 365, 366, 366, 1),
}


def _year_file(year, path):
    time = pd.date_range(f"{year}-01-01", f"{year}-12-31 23:00", freq="h")
    month, day = time.month.to_numpy(), time.day.to_numpy()
    t = np.full((time.size, 3, 2), 10.0, dtype="float32")
    t[:, 0, 0] = 20.0
    t[:, 1, 0] = np.where((month >= 6) & (month <= 8), 20.0, 10.0)
    t[:, 1, 1] = np.where(((month == 1) & (day <= 25)) | ((month == 12) & (day >= 7)), 20.0, 10.0)
    t[:, 2, 0] = np.nan
    t[:, 2, 1] = 20.0
    ds = xr.Dataset({"t2m": (("time", "lat", "lon"), t)},
                    coords={"time": time, "lat": np.array([45.0, 45.1, 45.2]), "lon": np.array([25.0, 25.1])})
    ds["t2m"].attrs["units"] = "degree_Celsius"
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path)
    ds.close()


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    root = tmp_path_factory.mktemp("known_answer")
    for y in (2005, 2006, 2007, 2008):
        _year_file(y, root / "out" / "06_filled_yearly" / f"{y}.nc")
    env = dict(os.environ, DDU_ROOT=str(root), DDU_SHP_RO=str(SHP))
    r = run_script([REPO / "scripts" / "step04_tails.py", "--years", "2005-2007", "--pad-days", "29"], env=env)
    assert r.returncode == 0, (r.stdout[-1500:], r.stderr[-1500:])
    r = run_script([REPO / "scripts" / "step05_ddu_season_wrap.py", "--years", "2005-2008", "--windows", "30", "--force"], env=env)
    assert r.returncode == 0, (r.stdout[-1500:], r.stderr[-1500:])
    return root, env


def _csv(root):
    return root / "out" / "08_season_swrap" / "win30" / "csv"


def test_onset_end_length_total_and_runs(run):
    root, _ = run
    df = pd.read_csv(_csv(root) / "season_ALL_YEARS_methods.csv")
    lat_order, lon_order = sorted(df["lat"].unique()), sorted(df["lon"].unique())
    wrong = []
    for (year, a, b), exp in EXPECTED.items():
        for m in METHODS:
            row = df[(df["year"] == year) & (df["method"] == m)
                     & np.isclose(df["lat"], lat_order[a]) & np.isclose(df["lon"], lon_order[b])]
            assert len(row) == 1, (year, a, b, m)
            r = row.iloc[0]
            got = (None if pd.isna(r["season_start_doy"]) else int(r["season_start_doy"]),
                   None if pd.isna(r["season_end_doy"]) else int(r["season_end_doy"]),
                   int(r["season_length"]), int(r["total_eip_days"]), int(r["runs_count"]))
            if got != exp:
                wrong.append((year, a, b, m, got, exp))
    assert not wrong, wrong[:5]


def test_cell_without_data_is_excluded(run):
    root, _ = run
    df = pd.read_csv(_csv(root) / "season_ALL_YEARS_methods.csv")
    assert df.groupby(["year", "method"]).size().unique().tolist() == [5]


def test_leap_year_dates_and_day_numbering(run):
    root, _ = run
    df = pd.read_csv(_csv(root) / "season_ALL_YEARS_methods.csv")
    lat1, lon0 = sorted(df["lat"].unique())[1], sorted(df["lon"].unique())[0]
    r = df[(df["year"] == 2008) & (df["method"] == "hourly_sum") & np.isclose(df["lat"], lat1) & np.isclose(df["lon"], lon0)].iloc[0]
    assert (r["season_start"], r["season_end"]) == ("2008-06-22", "2008-09-08")      # DOY 173-251 on the 365-day basis
    assert (int(r["season_start_doy"]), int(r["season_end_doy"])) == (173, 251)


def test_year_wrap_from_2006(run):
    root, _ = run
    summ = pd.read_csv(_csv(root) / "summary_ALL_YEARS_methods.csv")
    wrap = dict(zip(summ["year"], summ["year_wrap_used"]))
    assert int(wrap[2005]) == 0 and all(int(wrap[y]) == 1 for y in (2006, 2007, 2008))


def test_years_computed_with_other_settings_are_recomputed_not_reused(run, tmp_path):
    root, env = run
    shutil.copytree(root / "out" / "08_season_swrap", tmp_path / "out" / "08_season_swrap")   # work on a copy
    stamp = _csv(tmp_path) / "2005_settings.json"
    s = json.loads(stamp.read_text(encoding="utf-8"))
    assert s["min_gap_days"] == 4
    s["min_gap_days"] = 10                      # as if 2005 had been computed with the old 10-day rule
    stamp.write_text(json.dumps(s), encoding="utf-8")
    r = run_script([REPO / "scripts" / "step05_ddu_season_wrap.py", "--root", tmp_path, "--input-root", root,
                    "--years", "2005-2008", "--windows", "30"], env=dict(env, DDU_ROOT=str(tmp_path)))
    out = r.stdout + r.stderr
    assert r.returncode == 0, out[-1500:]
    assert "2005 (win30): existing per-year files have no/other settings stamp" in out
    for y in (2006, 2007, 2008):
        assert f"skip compute {y}" in out
    assert json.loads(stamp.read_text(encoding="utf-8"))["min_gap_days"] == 4
    pd.testing.assert_frame_equal(pd.read_csv(_csv(root) / "season_ALL_YEARS_methods.csv"),
                                  pd.read_csv(_csv(tmp_path) / "season_ALL_YEARS_methods.csv"))


def test_input_root_gives_the_same_result_in_another_folder(run, tmp_path):
    """--input-root reads the hourly files and tails from one folder and writes under --root (used by verify_release)."""
    root, env = run
    r = run_script([REPO / "scripts" / "step05_ddu_season_wrap.py", "--root", tmp_path, "--input-root", root,
                    "--years", "2005-2008", "--windows", "30"], env=dict(env, DDU_ROOT=str(tmp_path)))
    assert r.returncode == 0, (r.stdout + r.stderr)[-1500:]
    a = pd.read_csv(_csv(root) / "season_ALL_YEARS_methods.csv")
    b = pd.read_csv(_csv(tmp_path) / "season_ALL_YEARS_methods.csv")
    pd.testing.assert_frame_equal(a, b)


def test_missing_tail_is_reported(run, tmp_path):
    """Without the tail of the previous year a year is computed without the year wrap, and the script says so."""
    root, env = run
    data, work = tmp_path / "data", tmp_path / "work"
    (data / "out" / "06_filled_yearly").mkdir(parents=True)
    shutil.copy2(root / "out" / "06_filled_yearly" / "2008.nc", data / "out" / "06_filled_yearly" / "2008.nc")
    r = run_script([REPO / "scripts" / "step05_ddu_season_wrap.py", "--root", work, "--input-root", data,
                    "--years", "2008-2008", "--windows", "30"], env=dict(env, DDU_ROOT=str(work)))
    out = r.stdout + r.stderr
    assert r.returncode == 0, out[-1500:]
    assert "no tail file of the previous year" in out and "[2008]" in out
    assert "tail of 2007 not usable (tail_missing)" in out
    df = pd.read_csv(_csv(work) / "season_ALL_YEARS_methods.csv")
    assert int(df["season_start_doy"].min()) == 30                                # hence no suitable day before 30 January
