"""Daily DDU of the four methods (manuscript Table 1) and the 30-day window."""
import numpy as np
import pandas as pd
import pytest
import xarray as xr

_trapz = getattr(np, "trapezoid", None) or np.trapz


def _daily(values, start="2020-07-01"):
    t = pd.date_range(start, periods=len(values), freq="D")
    return xr.DataArray(np.asarray(values, dtype=float), coords={"time": t}, dims="time")


def test_base_temperature_and_thresholds(step05):
    assert step05.TBASE == 14.0
    assert step05.THRESH_DD_30D == 130.0
    assert step05.THRESH_DH_30D == 3120.0 == 130.0 * 24


def test_min_max_average(step05):
    tmin, tmax = _daily([10, 12, 16, 5]), _daily([20, 14, 30, 13])
    assert np.allclose(step05.ddu_midpoint(tmin, tmax).values, [1, 0, 9, 0])


def test_hourly_average(step05):
    assert np.allclose(step05.ddu_daily_mean(_daily([13, 14, 17.5, 20])).values, [0, 0, 3.5, 6])


def test_hourly_sum_counts_degree_hours(step05):
    t = pd.date_range("2020-07-01", periods=48, freq="h")
    temps = np.r_[np.full(24, 16.0), np.full(12, 10.0), np.full(12, 20.0)]
    out = step05.ddu_hourly_sum(xr.DataArray(temps, coords={"time": t}, dims="time")).values
    assert np.allclose(out, [24 * 2.0, 12 * 6.0])


@pytest.mark.parametrize("tmin,tmax", [(8, 18), (14, 26), (2, 10), (10, 30), (13.9, 14.1), (14, 14)])
def test_be_sine_equals_integral_of_a_sine_day(step05, tmin, tmax):
    x = np.linspace(-np.pi, np.pi, 400001)
    temp = (tmin + tmax) / 2 + (tmax - tmin) / 2 * np.sin(x)
    expected = _trapz(np.clip(temp - 14.0, 0, None), x) / (2 * np.pi)
    got = float(step05.ddu_be_sine(_daily([tmin]), _daily([tmax])).values[0])
    assert got == pytest.approx(expected, abs=1e-4)


def test_30_day_window_does_not_reset_on_cold_days(step05):
    ddu = np.full(45, 5.0)
    ddu[20:22] = 0.0                                   # two cold days inside the window
    s, eip = step05.roll_eip_window(_daily(ddu), 130.0, 30)
    s, eip = s.values, eip.values
    assert np.isnan(s[28]) and eip[28] == 0            # incomplete window: not suitable
    assert s[29] == pytest.approx(28 * 5.0)            # cold days add 0 but do not restart the count
    assert eip[29] == 1                                # 140 >= 130 (a reset would leave 8 days = 40)


def test_window_below_threshold_is_not_suitable(step05):
    s, eip = step05.roll_eip_window(_daily(np.full(35, 4.0)), 130.0, 30)
    assert s.values[29] == pytest.approx(120.0) and eip.values.max() == 0
