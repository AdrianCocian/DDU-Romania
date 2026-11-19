"""Statistics in ddu_stats.py reproduce the archived tables of the published run."""
import shutil

import numpy as np
import pandas as pd
import pytest

import ddu_stats as ds
from conftest import REPO, run_script


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
    r = run_script([REPO / "scripts" / "step07_suitability_classes.py",
                    "--root", tmp_path, "--season-csv", season_csv, "--no-maps"])
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


def test_tables_from_published_lengths_equal_archive(published, tmp_path):
    """step09 on the published season lengths reproduces the archived tables of the published run."""
    import json
    pc = pd.read_csv(published / "per_cell_classes_ALL_YEARS.csv")
    methods = ["midpoint", "daily_mean", "hourly_sum", "be_sine"]
    long = pc.melt(id_vars=["year", "i", "j", "lat", "lon"], value_vars=methods,
                   var_name="method", value_name="season_length")
    season_csv = tmp_path / "season.csv"
    long.to_csv(season_csv, index=False)
    pack = tmp_path / "pack"
    r = run_script([REPO / "scripts" / "step09_manuscript_statistics.py", "--season-csv", season_csv, "--out", pack,
                    "--variables", "season_length", "--no-excel"])
    assert r.returncode == 0, r.stderr[-2000:]
    assert json.loads((pack / "key_numbers.json").read_text(encoding="utf-8"))["n_cell_years_with_season_hourly_sum"] == 53278
    # tables written by this code from the published season lengths = archived tables (season length rows)
    r = run_script([REPO / "tools" / "check_against_manuscript.py", "--pack", pack, "--only-tables"])
    lines = {l.split()[0]: l for l in r.stdout.splitlines() if l.startswith("  T") or l.startswith("  per_cell")}
    for name in ("T4_class_agreement_pooled", "T4b_class_agreement_by_year", "T7_suitability_class_counts_by_year",
                 "T8_bias_by_reference_class", "T9_median_gap_diagnostic", "per_cell_classes_ALL_YEARS"):
        assert lines[name].rstrip().endswith("OK"), lines[name]
    # only season length was given, so the tables that also hold onset, end and total EIP days are incomplete here
    for name in ("T1_trend_MK_Sen_all_methods", "T2_distribution_diagnostics_summary", "T3",
                 "T5_national_summary_all_methods", "T6b_pairwise_delta_by_year"):
        assert "rows present on one side only" in lines[name] and "max rel. diff" not in lines[name], lines[name]


# ---------------------------------------------------------------- numbers printed in the manuscript
def test_every_number_quoted_in_the_manuscript_is_in_the_archived_tables(published):
    """Part A of the check on the archived tables: all four variables, classes, agreement, trends."""
    r = run_script([REPO / "tools" / "check_against_manuscript.py", "--pack", published, "--only-manuscript"])
    assert r.returncode == 0, r.stdout[-3000:]
    assert "[BAD]" not in r.stdout
    n = [l for l in r.stdout.splitlines() if l.strip().startswith("Part A:")][0].split()
    assert n[2] == n[4] and int(n[4]) >= 160


def test_manuscript_check_detects_a_changed_value(published, tmp_path):
    """The check must fail when a table no longer supports the text: one national median changed by one day."""
    pack = tmp_path / "changed"
    shutil.copytree(published, pack)
    t5 = pd.read_csv(pack / "T5_national_summary_all_methods.csv")
    sel = (t5.variable == "season_length") & (t5.method == "consensus") & (t5.year == 2024)
    assert sel.sum() == 1
    t5.loc[sel, "median"] += 1.0
    t5.to_csv(pack / "T5_national_summary_all_methods.csv", index=False)
    r = run_script([REPO / "tools" / "check_against_manuscript.py", "--pack", pack])
    assert r.returncode == 1
    assert "[BAD] Consensus season length 2024, median (days)" in r.stdout
    assert "T5_national_summary_all_methods" in r.stdout and "DIFFERENT" in r.stdout


# ---------------------------------------------------------------- study domain
def test_published_cells_are_the_cells_touched_by_the_national_boundary(published):
    """2,912 cells = the 0.1-degree cells that the boundary polygon touches (2,937) minus 25 cells on the Black Sea
    coast and the coastal lagoons for which ERA5-Land has no value."""
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import box
    from shapely.prepared import prep
    from conftest import SHP

    cells = pd.read_csv(published / "per_cell_classes_ALL_YEARS.csv", usecols=["year", "lat", "lon"])
    cells = cells[cells["year"] == 2005]
    have = {(round(float(a), 1), round(float(b), 1)) for a, b in zip(cells["lat"], cells["lon"])}
    polygon = prep(gpd.read_file(SHP).to_crs(4326).union_all())
    lats, lons = np.round(np.arange(43.5, 48.5001, 0.1), 1), np.round(np.arange(19.9, 30.1001, 0.1), 1)
    touched = {(float(y), float(x)) for y in lats for x in lons if polygon.intersects(box(x - 0.05, y - 0.05, x + 0.05, y + 0.05))}
    assert len(have) == 2912 and len(touched) == 2937
    assert have <= touched
    without_data = touched - have
    assert len(without_data) == 25 and min(x for _, x in without_data) >= 28.6
