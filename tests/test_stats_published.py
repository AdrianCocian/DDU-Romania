"""Statistics in ddu_stats.py reproduce the archived tables of the published run."""
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

import ddu_stats as ds
from conftest import REPO


# ---------------------------------------------------------------- Mann-Kendall / Sen
def test_mann_kendall_basic_properties():
    up = ds.mann_kendall(np.arange(10.0))
    assert up["S"] == 45 and up["tau"] == 1.0 and up["trend"] == "increasing"
    flat = ds.mann_kendall(np.ones(10))
    assert flat["S"] == 0 and flat["trend"] == "no trend"


def test_trend_rows_equal_published_T1(published):
    t5 = pd.read_csv(published / "T5_national_summary_all_methods.csv")
    t1 = pd.read_csv(published / "T1_trend_MK_Sen_all_methods.csv")
    cols = ["mk_S", "mk_p", "mk_z", "mk_tau", "sen_slope", "sen_lo95", "sen_hi95", "sen_intercept", "acf1"]
    n = 0
    for r in t1.itertuples():
        s = t5[(t5.variable == r.variable) & (t5.method == r.method)].sort_values("year")
        got = ds.trend_row(s["year"].to_numpy(), s[r.series].to_numpy())
        for c in cols:
            assert got[c] == pytest.approx(getattr(r, c), rel=1e-9, abs=1e-12), (r.variable, r.method, r.series, c)
        assert got["mk_trend"] == r.mk_trend
        n += 1
    assert n == len(t1) >= 40                     # 4 variables x 5 series (4 methods + consensus) x 2


def test_variance_partition_equals_published_T3(published):
    t5 = pd.read_csv(published / "T5_national_summary_all_methods.csv")
    t3 = pd.read_csv(published / "T3_variance_ratio_national_and_percell.csv")
    t3 = t3[t3["level"] == "national_series"]
    methods = ["midpoint", "daily_mean", "hourly_sum", "be_sine"]
    for r in t3.itertuples():
        tab = (t5[(t5.variable == r.variable) & (t5.method.isin(methods))]
               .pivot(index="year", columns="method", values=r.series)[methods])
        got = ds.variance_partition(tab)
        assert got["percent_method_uncertainty"] == pytest.approx(r.percent_method_uncertainty, rel=1e-9)


# ---------------------------------------------------------------- suitability classes
def test_class_limits_and_classes_equal_published(published):
    pc = pd.read_csv(published / "per_cell_classes_ALL_YEARS.csv")
    breaks = ds.class_breaks(pc["hourly_sum"].to_numpy())
    assert list(breaks) == [81.0, 108.0, 124.0, 135.0]
    assert ds.class_ranges(breaks) == ["0", "1-80", "81-107", "108-123", "124-134", ">=135"]
    for m in ["midpoint", "daily_mean", "hourly_sum", "be_sine", "consensus"]:
        assert np.array_equal(ds.classify(pc[m].to_numpy(), breaks), pc[f"class_{m}"].to_numpy()), m


def test_classify_edges():
    b = [81, 108, 124, 135]
    got = ds.classify(np.array([0, 1, 80, 80.5, 81, 107, 108, 123, 124, 134, 135, 300, np.nan]), b)
    assert np.array_equal(got[:-1], [0, 1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]) and np.isnan(got[-1])


def test_cohen_kappa_perfect_and_chance():
    a = np.array([0, 1, 2, 3, 4, 5] * 10)
    assert ds.cohen_kappa(a, a) == pytest.approx(1.0)
    assert ds.cohen_kappa(a, np.roll(a, 1)) < 0


def test_step07_reproduces_published_class_counts(published, tmp_path):
    """step07 on the published season lengths: same limits (Table 2) and the same counts as T7."""
    pc = pd.read_csv(published / "per_cell_classes_ALL_YEARS.csv")
    methods = ["midpoint", "daily_mean", "hourly_sum", "be_sine"]
    long = pc.melt(id_vars=["year", "i", "j", "lat", "lon"], value_vars=methods,
                   var_name="method", value_name="season_length")
    season_csv = tmp_path / "season_ALL_YEARS_methods.csv"
    long.to_csv(season_csv, index=False)
    r = subprocess.run([sys.executable, str(REPO / "scripts" / "step07_suitability_classes.py"),
                        "--root", str(tmp_path), "--season-csv", str(season_csv), "--no-maps"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    out = tmp_path / "out" / "12_method_compare" / "win30" / "7_suitability_classes"
    import json
    meta = json.loads((out / "00_meta" / "class_breaks_META_7.json").read_text(encoding="utf-8"))
    assert meta["class_breaks_percentiles_P25_P50_P75_P90"] == [81.0, 108.0, 124.0, 135.0]
    assert meta["n_cell_years_reference_with_season"] == 53278
    mine = pd.read_csv(out / "01_tables" / "suitability_class_counts_ALL_win30.csv")
    ref = pd.read_csv(published / "T7_suitability_class_counts_by_year.csv")
    m = mine.merge(ref, on=["year", "method", "class"], suffixes=("", "_ref"))
    assert len(m) == len(ref) == 600
    assert (m["count"] == m["count_ref"]).all()
    assert np.allclose(m["pct"], m["pct_ref"])


def test_manuscript_check_passes_on_published_tables(published, tmp_path):
    """tools/check_against_manuscript.py, Part A, on the archived tables (season length numbers)."""
    import json
    pc = pd.read_csv(published / "per_cell_classes_ALL_YEARS.csv")
    methods = ["midpoint", "daily_mean", "hourly_sum", "be_sine"]
    long = pc.melt(id_vars=["year", "i", "j", "lat", "lon"], value_vars=methods,
                   var_name="method", value_name="season_length")
    season_csv = tmp_path / "season.csv"
    long.to_csv(season_csv, index=False)
    pack = tmp_path / "pack"
    r = subprocess.run([sys.executable, str(REPO / "scripts" / "step09_manuscript_statistics.py"),
                        "--season-csv", str(season_csv), "--out", str(pack), "--variables", "season_length",
                        "--no-excel"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    r = subprocess.run([sys.executable, str(REPO / "tools" / "check_against_manuscript.py"), "--pack", str(pack)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-3000:]
    assert "all checks passed" in r.stdout
    assert json.loads((pack / "key_numbers.json").read_text())["n_cell_years_with_season_hourly_sum"] == 53278
