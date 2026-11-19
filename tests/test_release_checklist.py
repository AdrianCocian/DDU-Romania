"""tools/release_checklist.py: the metadata of the repository stay valid, and the checks detect what they should."""
import importlib.util
import itertools

import numpy as np
import pandas as pd
import pytest

from conftest import REPO

spec = importlib.util.spec_from_file_location("release_checklist", REPO / "tools" / "release_checklist.py")
rc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rc)


def test_zenodo_metadata_is_valid():
    ok, detail = rc.check_zenodo_json()
    assert ok, detail


def test_citation_file_is_valid_and_names_the_same_authors():
    ok, detail = rc.check_citation_cff()
    assert ok, detail


def test_no_credential_local_path_or_email_in_tracked_files():
    if not rc.tracked_files():
        pytest.skip("not a git checkout")
    ok, detail = rc.check_tracked_content()
    assert ok, detail


def test_patterns():
    assert rc.SECRET.search("url: https://cds.climate.copernicus.eu/api\nkey: 1a2b3c4d-1111-2222-3333-444455556666\n")
    assert not rc.SECRET.search("key: value of the legend\n")
    assert rc.LOCAL_PATH.search(r'"root": "E:\\PROIECT DDU\\BACK UP"') and rc.LOCAL_PATH.search(r"C:\Users\somebody\file")
    assert not rc.LOCAL_PATH.search(r"Windows: `C:\Users\<user>\.cdsapirc`")
    assert rc.EMAIL.search("write to first.last@example.org") and not rc.EMAIL.search("@pytest.fixture(scope='module')")


def _fake_run(folder, change_one_length=False):
    """A stand-in for results/run_<id>: 2 cells x 20 years x 4 methods."""
    folder.mkdir(parents=True, exist_ok=True)
    rows, wide = [], []
    rng = np.random.default_rng(3)
    for year, (i, j) in itertools.product(range(2005, 2025), [(0, 0), (0, 1)]):
        lengths = {m: int(rng.integers(0, 150)) for m in rc.METHODS}
        wide.append({"year": year, "i": i, "j": j, **{m: float(v) for m, v in lengths.items()}})
        for m, v in lengths.items():
            rows.append({"year": year, "i": i, "j": j, "method": m, "season_start_doy": 150.0 if v else np.nan,
                         "season_end_doy": 150.0 + v - 1 if v else np.nan, "season_length": v, "total_eip_days": v})
    df = pd.DataFrame(rows)
    if change_one_length:
        df.loc[5, "season_length"] += 1
    df.to_csv(folder / "season_ALL_YEARS_methods.csv.gz", index=False)
    pd.DataFrame(wide).to_csv(folder / "per_cell_classes_ALL_YEARS.csv", index=False)


def test_per_cell_file_check(tmp_path, monkeypatch):
    monkeypatch.setattr(rc, "REPO", tmp_path)
    monkeypatch.setattr(rc, "RUN", tmp_path / "run")
    monkeypatch.setattr(rc, "EXPECTED_ROWS", 2 * 20 * 4)
    (tmp_path / "run").mkdir()
    ok, detail = rc.check_per_cell_file()
    assert not ok and "is missing" in detail
    _fake_run(tmp_path / "run")
    ok, detail = rc.check_per_cell_file()
    assert ok, detail
    _fake_run(tmp_path / "run", change_one_length=True)
    ok, detail = rc.check_per_cell_file()
    assert not ok and "season lengths differ" in detail


@pytest.mark.parametrize("text, expected", [
    ("[PASS] stage 1: x\nresult: complete - every stage passed\n", True),
    ("[PASS] stage 1: x\n[SKIP] stage 4: y\nresult: INCOMPLETE - stage 4 not run\n", False),
    ("[FAIL] stage 4: y\nresult: complete - every stage passed\n", False),
    ("no result here\n", False),
])
def test_verification_record_check(tmp_path, monkeypatch, text, expected):
    monkeypatch.setattr(rc, "RUN", tmp_path)
    (tmp_path / "VERIFICATION.txt").write_text(text, encoding="utf-8")
    ok, _ = rc.check_verification_record()
    assert ok is expected


# ------------------------------------------------------------------ the archived per-cell file, once it is there
ARCHIVED = REPO / "results" / "run_20260908_131817_6a83be93" / "season_ALL_YEARS_methods.csv.gz"
needs_archive = pytest.mark.skipif(not ARCHIVED.exists(),
                                   reason="per-cell file of the published run not archived yet (tools/verify_release.py adds it)")


@needs_archive
def test_archived_per_cell_file_is_complete_and_matches_the_tables():
    ok, detail = rc.check_per_cell_file()
    assert ok, detail


@needs_archive
@pytest.mark.slow
def test_article_numbers_and_tables_from_the_archived_per_cell_file():
    """Step 09 on the archived per-cell file: every number printed in the article and every archived table."""
    ok, detail = rc.check_article_numbers()
    assert ok, detail
