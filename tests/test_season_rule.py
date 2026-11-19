"""3-day season rule (manuscript section 2.3) and the 365-day DOY convention."""
import numpy as np
import pandas as pd

import ddu_stats as ds


def _eip(pattern: str) -> np.ndarray:
    """'1' = transmission-suitable day, '0' = unsuitable day."""
    return np.array([c == "1" for c in pattern])


def test_default_is_the_3_day_rule(step05):
    assert step05.MIN_GAP_DAYS == 4


def test_one_or_two_unsuitable_days_are_bridged(step05):
    assert step05.split_runs(_eip("1110111001111"), 4) == [(0, 12)]


def test_three_consecutive_unsuitable_days_end_the_season(step05):
    assert step05.split_runs(_eip("1110001111"), 4) == [(0, 2), (6, 9)]


def test_no_suitable_day_means_no_run(step05):
    assert step05.split_runs(_eip("0000"), 4) == []


def test_season_is_the_longest_run_with_bridged_days_included(step05):
    # run A: days 0-4 (5 d); 3 unsuitable days; run B: days 8-17 with a bridged 2-day hole (10 d)
    eip = _eip("11111" + "000" + "1111" + "00" + "1111" + "0000")
    t = pd.date_range("2021-06-01", periods=eip.size, freq="D").values
    m = step05.season_metrics_for_pixel(eip, t)
    assert m["runs_count"] == 2
    assert (m["season_start"], m["season_end"]) == ("2021-06-09", "2021-06-18")
    assert m["season_length"] == 10           # end - start + 1, bridged days included
    assert m["total_eip_days"] == 13          # every suitable day of the year, also outside the season
    assert m["outside_eip_days"] == 5


def test_no_season(step05):
    eip = _eip("0" * 30)
    m = step05.season_metrics_for_pixel(eip, pd.date_range("2021-01-01", periods=30, freq="D").values)
    assert m["season_length"] == 0 and m["total_eip_days"] == 0 and m["season_start"] == ""


def test_doy_omits_29_february(step05):
    f = step05.doy_leap_safe
    assert f(pd.Timestamp("2024-02-28")) == 59
    assert f(pd.Timestamp("2024-03-01")) == f(pd.Timestamp("2023-03-01")) == 60
    assert f(pd.Timestamp("2024-12-31")) == f(pd.Timestamp("2023-12-31")) == 365


def test_doy_labels_used_in_the_tables():
    assert ds.doy_to_date(170).strftime("%d %b") == "19 Jun"
    assert ds.doy_date_label(178.5) == "27–28 Jun"
    assert ds.doy_date_label(243.5) == "31 Aug–1 Sep"
