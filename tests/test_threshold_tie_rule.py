"""tools/verify_release.py, stage 4: a recomputed row that differs from the published one is accepted only when
rounding at the threshold explains it.

Single-precision sums carry about seven significant digits. On a day whose exact 30-day sum lies a fraction of a
degree-hour from the threshold, rounding alone can decide whether the day counts as suitable. Such a day may be
decided differently by two runs of the same code (another order of summation, another library version); any
other difference must stop the verification.
"""
import importlib.util

import numpy as np
import pandas as pd
import pytest

from conftest import REPO

spec = importlib.util.spec_from_file_location("verify_release", REPO / "tools" / "verify_release.py")
vr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vr)

THR = 3120.0
DAYS = pd.date_range("2018-01-01", periods=365, freq="D")


@pytest.fixture
def values_of(step05):
    return lambda flags: vr.season_values(step05, flags, DAYS)


def day_label(k):
    return f"{DAYS[k].date()} (DOY {k + 1})"


def sums_with_tie():
    """30-day sums above the threshold from 31 May to 2 October; on 30 May the double-precision sum is 0.000305
    below the threshold and the single-precision sum equals it."""
    s64 = np.full(365, 1000.0)
    s64[150:275] = 5000.0
    s64[149] = THR - 0.000305
    s32 = s64.copy()
    s32[149] = THR
    return s32, s64


def test_rounding_at_the_threshold_is_accepted_and_reported(values_of):
    s32, s64 = sums_with_tie()
    published, recomputed = values_of(s64 >= THR), values_of(s32 >= THR)
    assert published["season_start_doy"] == 151 and recomputed["season_start_doy"] == 150
    ok, text = vr.classify_tie(s32, s64, THR, values_of, published, recomputed, day_label)
    assert ok, text
    assert text.startswith("rounding at the threshold")
    assert "2018-05-30 (DOY 150) = 3119.999695 in double precision and 3120.000000 in single precision" in text
    assert "counted as not suitable in the published run, as in double precision" in text
    assert "published row: onset 151, end 275, length 125, total EIP days 125" in text


def test_published_row_that_no_decision_explains_is_rejected(values_of):
    s32, s64 = sums_with_tie()
    published = dict(values_of(s64 >= THR), season_end_doy=276.0)
    ok, text = vr.classify_tie(s32, s64, THR, values_of, published, values_of(s32 >= THR), day_label)
    assert not ok and "does not follow from either decision" in text


def test_disagreement_away_from_the_threshold_is_rejected(values_of):
    s32, s64 = sums_with_tie()
    s32[100] = 5000.0                                    # single precision far above, double precision far below
    ok, text = vr.classify_tie(s32, s64, THR, values_of, values_of(s64 >= THR), values_of(s32 >= THR), day_label)
    assert not ok and "disagree on 2018-04-11 (DOY 101)" in text and "is not at the threshold" in text


def test_difference_without_a_day_at_the_threshold_is_rejected(values_of):
    s64 = np.full(365, 1000.0)
    s64[150:275] = 5000.0
    published = dict(values_of(s64 >= THR), season_start_doy=152.0)
    ok, text = vr.classify_tie(s64, s64, THR, values_of, published, values_of(s64 >= THR), day_label)
    assert not ok and "no day of the year has a 30-day sum within 0.001 % of the threshold" in text


def test_recomputed_row_must_follow_from_the_recomputed_sums(values_of):
    s32, s64 = sums_with_tie()
    recomputed = dict(values_of(s32 >= THR), total_eip_days=1.0)
    ok, text = vr.classify_tie(s32, s64, THR, values_of, values_of(s64 >= THR), recomputed, day_label)
    assert not ok and "internal check failed" in text


def test_tolerance_is_relative_to_the_threshold(values_of):
    s32, s64 = sums_with_tie()
    s64[149] = THR - 0.04                               # 0.0013 % below 3120 degree-hours: not at the threshold
    ok, text = vr.classify_tie(s32, s64, THR, values_of, values_of(s64 >= THR), values_of(s32 >= THR), day_label)
    assert not ok and "is not at the threshold" in text


@pytest.mark.parametrize("a, b, same", [
    ("2.0", "2", True), ("3;5.0", "3;5", True), ("", "", True), ("2", "3", False), ("3;5", "5;3", False),
    ("2013-05-01–2013-05-03", "2013-05-01–2013-05-03", True), ("2013-05-01–2013-05-03", "2013-05-01–2013-05-04", False),
])
def test_number_text(a, b, same):
    assert (vr._number_text(a) == vr._number_text(b)) is same
