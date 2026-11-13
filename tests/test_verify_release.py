"""tools/verify_release.py on a small stand-in for the data folder of the published run.

The tool must pass on an intact folder, write nothing into it, and stop - without writing to results/ - when
a season length, an hourly temperature or only an onset date differs from what this repository computes.
"""
import gzip
import os
import shutil

import pandas as pd
import pytest
import xarray as xr

from conftest import REPO, SHP, run_script
from synthetic import make_varied_root

pytestmark = pytest.mark.slow
YEARS = "2005-2008"
CSV = ("out", "08_season_swrap", "win30", "csv")


def _snapshot(folder):
    return {p.relative_to(folder).as_posix(): (p.stat().st_mtime_ns, p.stat().st_size) for p in folder.rglob("*") if p.is_file()}


@pytest.fixture(scope="module")
def published_like(tmp_path_factory):
    """Data folder as left by a complete run: yearly files, filled files, tails, per-cell seasons; plus its tables."""
    base = tmp_path_factory.mktemp("verify")
    data = make_varied_root(base / "data")
    env = dict(os.environ, DDU_SHP_RO=str(SHP), DDU_ROOT=str(data))
    filled, yearly = data / "out" / "06_filled_yearly", data / "out" / "04_yearly_raw"
    filled.rename(yearly)                                        # the generator's files play the role of step03's output
    for script, args in (("step03b_qc_fill_year.py", ["--root", data, "--years", YEARS]),
                         ("step04_tails.py", ["--years", "2005-2007", "--pad-days", "29"]),
                         ("step05_ddu_season_wrap.py", ["--root", data, "--years", YEARS, "--windows", "30"])):
        r = run_script([REPO / "scripts" / script, *args], env=env)
        assert r.returncode == 0, (script, r.stdout[-2000:], r.stderr[-2000:])
    reference = base / "reference"
    r = run_script([REPO / "scripts" / "step09_manuscript_statistics.py", "--season-csv",
                    data.joinpath(*CSV, "season_ALL_YEARS_methods.csv"), "--out", reference, "--no-excel"], env=env)
    assert r.returncode == 0, (r.stdout[-2000:], r.stderr[-2000:])
    return base, data, reference


def _verify(data, reference, work, results, *extra):
    env = dict(os.environ, DDU_SHP_RO=str(SHP))
    env.pop("DDU_ROOT", None)
    return run_script([REPO / "tools" / "verify_release.py", "--data-root", data, "--reference", reference, "--work", work,
                       "--results-dir", results, "--years", YEARS, "--tables-only", *extra], env=env, timeout=1500)


def _copy(data, dst, hourly=True):
    """Copy of the data folder; without the hourly files it holds the per-cell results only."""
    if hourly:
        shutil.copytree(data, dst)
    else:
        shutil.copytree(data.joinpath(*CSV), dst.joinpath(*CSV))
    return dst


def _change(csv_file, column, delta, where_length_positive=True):
    df = pd.read_csv(csv_file)
    idx = df.index[(df["season_length"] > 20) & (df["method"] == "hourly_sum")][0] if where_length_positive else df.index[0]
    df.loc[idx, column] = df.loc[idx, column] + delta
    df.to_csv(csv_file, index=False)


def test_intact_folder_passes_every_stage_and_is_left_untouched(published_like):
    base, data, reference = published_like
    before = _snapshot(data)
    results = base / "results_ok"
    r = _verify(data, reference, base / "work_ok", results)
    out = r.stdout + r.stderr
    assert r.returncode == 0, out[-4000:]
    for n in range(1, 8):
        assert f"[PASS] stage {n}:" in out, (n, out[-4000:])
    assert "VERIFICATION PASSED - every stage passed" in out
    assert _snapshot(data) == before                              # nothing created, changed or removed in the data folder

    published_csv = data.joinpath(*CSV, "season_ALL_YEARS_methods.csv")
    with gzip.open(results / "season_ALL_YEARS_methods.csv.gz", "rb") as f:
        assert f.read() == published_csv.read_bytes()             # the archive holds the published file, byte for byte
    assert (results / "summary_ALL_YEARS_methods.csv").exists() and (results / "qc_gaps_summary.csv").exists()
    record = (results / "VERIFICATION.txt").read_text(encoding="utf-8")
    assert "result: complete - every stage passed" in record
    assert str(base) not in record and str(REPO) not in record    # no local path in a file meant to be public
    qc = pd.read_csv(results / "qc_gaps_summary.csv")
    assert len(qc) == 4 and (qc["missing_hours_inside_before"] == 0).all() and (qc["filled_file_check"] == "identical").all()

    # the documented way to rebuild the tables from the archived per-cell file
    env = dict(os.environ, DDU_SHP_RO=str(SHP), DDU_ROOT=str(base / "work_ok"))
    pack = base / "pack_from_archive"
    r = run_script([REPO / "scripts" / "step09_manuscript_statistics.py", "--season-csv",
                    results / "season_ALL_YEARS_methods.csv.gz", "--out", pack, "--no-excel"], env=env)
    assert r.returncode == 0, (r.stdout[-2000:], r.stderr[-2000:])
    r = run_script([REPO / "tools" / "check_against_manuscript.py", "--pack", pack, "--reference", reference, "--only-tables"], env=env)
    assert r.returncode == 0 and "RESULT: all checks passed" in r.stdout, r.stdout[-2000:]


def test_without_hourly_files_the_result_is_reported_as_incomplete(published_like):
    base, data, reference = published_like
    only_results = _copy(data, base / "data_results_only", hourly=False)
    r = _verify(only_results, reference, base / "work_partial", base / "results_partial")
    out = r.stdout + r.stderr
    assert r.returncode == 0, out[-4000:]
    assert "[SKIP] stage 3:" in out and "[SKIP] stage 4:" in out
    assert "VERIFICATION PASSED, BUT INCOMPLETE" in out and "every stage passed" not in out
    assert "result: INCOMPLETE" in (base / "results_partial" / "VERIFICATION.txt").read_text(encoding="utf-8")


def test_changed_season_length_stops_at_stage_2(published_like):
    base, data, reference = published_like
    bad = _copy(data, base / "data_bad_length", hourly=False)
    _change(bad.joinpath(*CSV, "season_ALL_YEARS_methods.csv"), "season_length", 1)
    results = base / "results_bad_length"
    r = _verify(bad, reference, base / "work_bad_length", results)
    out = r.stdout + r.stderr
    assert r.returncode == 1 and "[FAIL] stage 2:" in out and "VERIFICATION FAILED" in out
    assert not results.exists()


def test_changed_hourly_temperature_stops_at_stage_3(published_like):
    base, data, reference = published_like
    bad = _copy(data, base / "data_bad_hour")
    f = bad / "out" / "06_filled_yearly" / "2006.nc"
    with xr.open_dataset(f) as ds:
        changed = ds.load()
    changed["t2m"].values[4000, 2, 3] += 0.5                      # one hour, one cell
    changed.to_netcdf(bad / "changed.nc")
    changed.close()
    (bad / "changed.nc").replace(f)
    results = base / "results_bad_hour"
    r = _verify(bad, reference, base / "work_bad_hour", results)
    out = r.stdout + r.stderr
    assert r.returncode == 1 and "[FAIL] stage 3:" in out and "DIFFERENT (t2m)" in out
    assert not results.exists()


def test_changed_onset_with_unchanged_length_stops_at_stage_4(published_like):
    base, data, reference = published_like
    bad = _copy(data, base / "data_bad_onset")
    _change(bad.joinpath(*CSV, "season_ALL_YEARS_methods.csv"), "season_start_doy", 1)
    results = base / "results_bad_onset"
    r = _verify(bad, reference, base / "work_bad_onset", results)
    out = r.stdout + r.stderr
    assert r.returncode == 1, out[-4000:]
    assert "[PASS] stage 2:" in out and "[PASS] stage 3:" in out and "[FAIL] stage 4:" in out
    assert "season_start_doy: 1 rows differ" in out
    assert not results.exists()


def test_difference_outside_the_analysed_columns_is_reported_but_does_not_fail(published_like):
    base, data, reference = published_like
    other = _copy(data, base / "data_other_column")
    _change(other.joinpath(*CSV, "season_ALL_YEARS_methods.csv"), "outside_eip_days", 1)
    results = base / "results_other_column"
    r = _verify(other, reference, base / "work_other_column", results)
    out = r.stdout + r.stderr
    assert r.returncode == 0, out[-4000:]
    assert "[PASS] stage 4:" in out and "6 of 6 analysed columns identical" in out
    note = "other column, not part of the verdict - outside_eip_days: 1 rows differ"
    assert note in out and note in (results / "VERIFICATION.txt").read_text(encoding="utf-8")


def test_work_folder_must_not_be_the_data_folder(published_like):
    base, data, reference = published_like
    before = _snapshot(data)
    r = _verify(data, reference, data, base / "results_same")
    assert r.returncode == 2 and "must not be the data folder" in (r.stdout + r.stderr)
    assert _snapshot(data) == before
