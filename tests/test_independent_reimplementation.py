"""Steps 04-05 against a second implementation written from the definitions in the article (tests/independent.py).

The data set has cells without a season, seasons made of several runs, runs joined across one or two unsuitable
days and a leap year, so the comparison covers every branch of the season definition.
"""
import os

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import independent as ind
from conftest import REPO, SHP, run_script
from synthetic import make_varied_root

YEARS = (2005, 2006, 2007, 2008)


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    root = make_varied_root(tmp_path_factory.mktemp("varied"), YEARS)
    env = dict(os.environ, DDU_ROOT=str(root), DDU_SHP_RO=str(SHP))
    r = run_script([REPO / "scripts" / "step04_tails.py", "--years", "2005-2007", "--pad-days", "29"], env=env)
    assert r.returncode == 0, (r.stdout[-1500:], r.stderr[-1500:])
    r = run_script([REPO / "scripts" / "step05_ddu_season_wrap.py", "--years", "2005-2008", "--windows", "30", "--force"], env=env)
    assert r.returncode == 0, (r.stdout[-1500:], r.stderr[-1500:])
    df = pd.read_csv(root / "out" / "08_season_swrap" / "win30" / "csv" / "season_ALL_YEARS_methods.csv")
    return root, df


def _independent(root):
    """{(year, method, lat, lon): season tuple or None} for every cell with data."""
    out, previous = {}, None
    for y in YEARS:
        with xr.open_dataset(root / "out" / "06_filled_yearly" / f"{y}.nc") as ds:
            t = ds["t2m"].transpose("time", "latitude", "longitude").load()
        lat, lon = t["latitude"].values, t["longitude"].values
        leap = bool(pd.Timestamp(t["time"].values[0]).is_leap_year)
        ddu = ind.daily_ddu(t.values)
        for m, thr in ind.THRESHOLD.items():
            suit = ind.suitable_days(ddu[m], None if previous is None else previous[m], thr)
            for a in range(lat.size):
                for b in range(lon.size):
                    if np.isnan(t.values[0, a, b]):
                        continue
                    out[(y, m, round(float(lat[a]), 4), round(float(lon[b]), 4))] = ind.season(suit[:, a, b], leap)
        previous = ddu
    return out


def test_every_cell_year_and_method_equals_the_independent_implementation(run):
    root, df = run
    expected = _independent(root)
    assert len(df) == len(expected) == 4 * 4 * 23            # 4 years x 4 methods x 23 cells with data
    wrong = []
    for r in df.itertuples(index=False):
        got = None if pd.isna(r.season_start_doy) else (int(r.season_start_doy), int(r.season_end_doy), int(r.season_length),
                                                        int(r.total_eip_days), int(r.runs_count))
        exp = expected[(r.year, r.method, round(float(r.lat), 4), round(float(r.lon), 4))]
        if got != exp:
            wrong.append((r.year, r.method, r.lat, r.lon, got, exp))
    assert not wrong, wrong[:5]


def test_the_data_set_exercises_every_branch(run):
    _, df = run
    inside = df["total_eip_days"] - df["outside_eip_days"]
    assert (df["season_length"] == 0).sum() > 10                                 # cell-years without a season
    assert (df["runs_count"] >= 2).sum() > 20                                    # several runs: the longest is the season
    assert (inside < df["season_length"]).sum() > 5                              # one or two unsuitable days inside a season
    assert (df["total_eip_days"] < df["season_length"]).sum() > 0                # hence total EIP days can be below length
    assert (df["total_eip_days"] > df["season_length"]).sum() > 20               # ... or above it (suitable days outside)
    assert set(df["method"]) == set(ind.THRESHOLD)
    by_method = df.groupby("method")["season_length"].mean()
    assert by_method.nunique() == 4                                              # the four methods give different seasons


def test_relations_between_the_columns(run):
    _, df = run
    s = df.dropna(subset=["season_start_doy"])
    days = (pd.to_datetime(s["season_end"]) - pd.to_datetime(s["season_start"])).dt.days + 1
    assert (days == s["season_length"]).all()                                    # length = days from onset to end, inclusive
    inside = df["total_eip_days"] - df["outside_eip_days"]
    assert ((inside >= 0) & (inside <= df["season_length"])).all()               # suitable days inside the season
    assert (df["outside_eip_days"] == df["preseason_eip_days"] + df["postseason_eip_days"]).all()
    assert ((df["season_length"] == 0) == (df["total_eip_days"] == 0)).all()     # no season <=> no suitable day
    assert (s["outside_runs_count"] == s["runs_count"] - 1).all()
