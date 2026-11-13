"""Shared test setup: repository on sys.path, headless plotting, a throw-away data root."""
from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
PUBLISHED = REPO / "results" / "run_20260908_131817_6a83be93"
SHP = REPO / "data" / "boundaries" / "ro_frontiera_poligon.shp"

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("DDU_SHP_RO", str(SHP))
# config.py creates its output folders under DDU_ROOT: keep them out of the repository
os.environ["DDU_ROOT"] = tempfile.mkdtemp(prefix="ddu_test_root_")
for p in (str(REPO / "tests"), str(REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="session")
def step05():
    return _load_script("step05_ddu_season_wrap")


@pytest.fixture(scope="session")
def published():
    if not PUBLISHED.exists():
        pytest.skip("archived tables of the published run not present")
    return PUBLISHED
