"""A second implementation of the method, written from the definitions in the article and not from the pipeline code.

It uses numpy only and shares no function with scripts/ or ddu_stats.py, so agreement between the two is evidence
that the pipeline computes what the article describes:

  - daily DDU (developmental threshold 14 degC): min-max average, hourly average, hourly sum, Baskerville-Emin sine;
  - a day is suitable when the sum over the 30 days ending on that day reaches 130 DDU (3120 degree-hours for the
    hourly sum); the window of the first days of January is completed with the last 29 days of the previous year;
  - a season is a run of suitable days that ends when three or more consecutive days are unsuitable; when a cell has
    several runs in a year the longest is the season (the earliest of equally long runs);
  - onset and end are day numbers on a 365-day basis (29 February shares day 59); season length counts every day
    from onset to end; total EIP days counts every suitable day of the year.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

TB = 14.0
THRESHOLD = {"hourly_sum": 3120.0, "daily_mean": 130.0, "midpoint": 130.0, "be_sine": 130.0}


def daily_ddu(hourly_degc: np.ndarray) -> Dict[str, np.ndarray]:
    """hourly_degc: (hours, ...) with a whole number of days -> {method: (days, ...)}."""
    n = hourly_degc.shape[0] // 24
    h = hourly_degc[: n * 24].reshape(n, 24, *hourly_degc.shape[1:]).astype("float64")
    tmin, tmax, tmean = h.min(axis=1), h.max(axis=1), h.mean(axis=1)
    avg, amp = (tmin + tmax) / 2.0, (tmax - tmin) / 2.0
    with np.errstate(invalid="ignore", divide="ignore"):
        theta = np.arcsin(np.clip((TB - avg) / np.where(amp > 0, amp, np.nan), -1.0, 1.0))
        partly = ((avg - TB) * (np.pi / 2.0 - theta) + amp * np.cos(theta)) / np.pi
    return {
        "hourly_sum": np.clip(h - TB, 0.0, None).sum(axis=1),
        "daily_mean": np.clip(tmean - TB, 0.0, None),
        "midpoint": np.clip(avg - TB, 0.0, None),
        "be_sine": np.where(tmin >= TB, avg - TB, np.where(tmax <= TB, 0.0, partly)),
    }


def suitable_days(ddu: np.ndarray, previous: Optional[np.ndarray], threshold: float) -> np.ndarray:
    """Boolean (days, ...): 30-day sum ending on the day >= threshold; `previous` = daily DDU of the year before."""
    ext = ddu if previous is None else np.concatenate([previous[-29:], ddu], axis=0)
    first = 0 if previous is None else 29
    csum = np.cumsum(np.concatenate([np.zeros((1,) + ext.shape[1:]), ext], axis=0), axis=0)
    rolling = np.full(ext.shape, np.nan)
    rolling[29:] = csum[30:] - csum[:-30]
    return rolling[first:] >= threshold


def season(suitable_1d: np.ndarray, leap: bool) -> Optional[Tuple[int, int, int, int, int]]:
    """(onset day, end day, length, total suitable days, number of runs) or None when no day is suitable."""
    days = np.flatnonzero(suitable_1d)                 # 0-based calendar days
    if days.size == 0:
        return None
    runs, start = [], int(days[0])
    for a, b in zip(days[:-1], days[1:]):
        if b - a >= 4:                                # three or more unsuitable days in between
            runs.append((start, int(a)))
            start = int(b)
    runs.append((start, int(days[-1])))
    first, last = max(runs, key=lambda r: (r[1] - r[0] + 1, -r[0]))

    def number(day: int) -> int:                      # 1-based, 365-day basis
        return day + 1 - (1 if (leap and day >= 59) else 0)

    return number(first), number(last), last - first + 1, int(days.size), len(runs)
