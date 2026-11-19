"""End-to-end: steps 5-10 on a small synthetic dataset (no download)."""
import json
import os

import numpy as np
import pandas as pd
import pytest

from conftest import REPO, SHP, run_script
from synthetic import make_synthetic_root

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    root = make_synthetic_root(tmp_path_factory.mktemp("synthetic"))
    env = dict(os.environ, DDU_ROOT=str(root), DDU_SHP_RO=str(SHP))
    r = run_script([REPO / "run_pipeline.py", "--root", root, "--years", "2005-2008", "--from", "5", "--to", "10",
                    "--no-maps", "--map-years", "2005,2006,2007,2008"], env=env)
    assert r.returncode == 0, (r.stdout[-3000:], r.stderr[-3000:])
    return root


def test_season_file(run):
    csv = run / "out" / "08_season_swrap" / "win30" / "csv"
    df = pd.read_csv(csv / "season_ALL_YEARS_methods.csv")
    assert set(df["method"]) == {"midpoint", "daily_mean", "hourly_sum", "be_sine"}
    assert df.groupby(["year", "method"]).size().eq(11).all()          # 12 cells, one outside the mask
    s = df.dropna(subset=["season_start_doy"])
    days = (pd.to_datetime(s["season_end"]) - pd.to_datetime(s["season_start"])).dt.days + 1
    assert (days == s["season_length"]).all()                          # length = days from onset to end, inclusive
    # total EIP days counts suitable days only: inside the season it cannot exceed the length (bridged days are
    # unsuitable), and suitable days outside the season add to it
    inside = df["total_eip_days"] - df["outside_eip_days"]
    assert ((inside >= 0) & (inside <= df["season_length"])).all()
    meta = json.loads((csv / "META.json").read_text(encoding="utf-8"))
    assert meta["min_gap_days"] == 4 and meta["season_ends_after_consecutive_unsuitable_days"] == 3


def test_step06_consensus_and_step08_trends_agree_with_step09(run):
    base = run / "out" / "12_method_compare" / "win30"
    assert (base / "6_method_comparison").is_dir()
    t1 = pd.read_csv(base / "00_MANUSCRIPT_PACK" / "T1_trend_MK_Sen_all_methods.csv")
    for var in ["season_length", "start_doy", "end_doy", "total_eip_days"]:
        t8 = pd.concat([pd.read_csv(base / "8_statistics" / "graphs" / "tables" / f"trend_stats_{var}.csv"),
                        pd.read_csv(base / "8_statistics" / "graphs" / "tables" / f"trend_stats_{var}_by_method.csv")])
        m = t8.merge(t1[t1.variable == var], on=["variable", "method", "series"], suffixes=("_8", "_9"))
        assert len(m) == 10, var
        for c in ["mk_S", "mk_p", "sen_slope", "sen_lo95", "sen_hi95", "sen_intercept"]:
            assert np.allclose(m[c + "_8"], m[c + "_9"], equal_nan=True), (var, c)


def test_classes_step07_equal_step09(run):
    base = run / "out" / "12_method_compare" / "win30"
    c7 = pd.read_csv(base / "7_suitability_classes" / "01_tables" / "suitability_class_counts_ALL_win30.csv")
    c9 = pd.read_csv(base / "00_MANUSCRIPT_PACK" / "T7_suitability_class_counts_by_year.csv")
    m = c7.merge(c9, on=["year", "method", "class"], suffixes=("_7", "_9"))
    assert len(m) == len(c9) and (m["count_7"] == m["count_9"]).all()


def test_figures_written(run):
    fig = run / "out" / "12_method_compare" / "win30" / "00_MANUSCRIPT_PACK" / "figures"
    for name in ["Fig1_season_length_consensus", "Fig2_suitability_classes_hourly_sum", "Fig3_season_length_timeseries"]:
        assert (fig / f"{name}.png").stat().st_size > 10_000
        assert (fig / f"{name}.pdf").exists()
