# -*- coding: utf-8 -*-
"""
ddu_stats.py - statistical and formatting helpers shared by steps 07-10.

All functions are pure (no file I/O) so that they can be unit-tested
(see tests/). The definitions follow section 2.4 of the manuscript:

- Mann-Kendall trend test with tie correction and continuity correction;
  Sen's slope with a 95 % confidence interval by the rank method of
  Gilbert (1987), with linear interpolation between ordered slopes when
  the ranks are not integers; intercept = median(x) - slope * median(t),
  t = 0 for the first year.
- Variance partition of method choice versus interannual climate on a
  years x methods table of national values.
- Relative climatic suitability classes: quantile bands (P25, P50, P75, P90)
  of the hourly-sum season length among cell-years with a season, pooled
  over all years; class 0 = no season.
- Day of year (DOY) is numbered on a 365-day basis (in leap years 29 February
  shares day 59 with 28 February), so a DOY always maps to the same calendar
  date. No day is left out of the calculation.
"""
from __future__ import annotations

import datetime as _dt
import math
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats as _st

MINUS = "−"   # typographic minus sign used in the manuscript tables
EN_DASH = "–"

# ---------------------------------------------------------------------------
# Day of year and number formatting
# ---------------------------------------------------------------------------
_NONLEAP = _dt.date(2001, 1, 1)   # any non-leap year: DOY n -> same date every year


def doy_to_date(doy: int) -> _dt.date:
    """Calendar date of an integer DOY on the 365-day basis (day 59 = 28 February in every year)."""
    d = int(doy)
    if not 1 <= d <= 365:
        raise ValueError(f"DOY out of range 1..365: {doy}")
    return _NONLEAP + _dt.timedelta(days=d - 1)


def _day_month(d: _dt.date) -> str:
    return f"{d.day} {d.strftime('%b')}"


def doy_date_label(doy: float) -> str:
    """'19 Jun' for an integer DOY; '27-28 Jun' or '31 Aug-1 Sep' (en dash) for a fractional one."""
    if doy is None or not np.isfinite(doy):
        return ""
    lo, hi = math.floor(doy), math.ceil(doy)
    if lo == hi:
        return _day_month(doy_to_date(lo))
    a, b = doy_to_date(lo), doy_to_date(hi)
    if a.month == b.month:
        return f"{a.day}{EN_DASH}{b.day} {b.strftime('%b')}"
    return f"{_day_month(a)}{EN_DASH}{_day_month(b)}"


def fmt_num(x: float, max_decimals: int = 2) -> str:
    """Up to `max_decimals` decimals, trailing zeros removed, typographic minus sign."""
    if x is None or not np.isfinite(x):
        return ""
    s = f"{float(x):.{max_decimals}f}".rstrip("0").rstrip(".")
    if s in ("-0", ""):
        s = "0"
    return s.replace("-", MINUS)


def fmt_fixed(x: float, decimals: int = 1) -> str:
    """Fixed number of decimals, half-up on the decimal value, typographic minus sign."""
    if x is None or not np.isfinite(x):
        return ""
    q = round_half_up(float(x), decimals)
    return f"{q:.{decimals}f}".replace("-", MINUS)


def round_half_up(x: float, decimals: int = 1) -> float:
    """Round half away from zero on the decimal representation (30.15 -> 30.2)."""
    from decimal import Decimal, ROUND_HALF_UP
    q = Decimal(1).scaleb(-decimals)
    return float(Decimal(repr(float(x))).quantize(q, rounding=ROUND_HALF_UP))


def fmt_doy(doy: float) -> str:
    """'170 (19 Jun)' or '178.5 (27-28 Jun)'."""
    if doy is None or not np.isfinite(doy):
        return ""
    return f"{fmt_num(doy)} ({doy_date_label(doy)})"


def fmt_p50_iqr(p50: float, p25: float, p75: float) -> str:
    """'-1 (-4, 0)' with typographic minus signs."""
    return f"{fmt_num(p50)} ({fmt_num(p25)}, {fmt_num(p75)})"


# ---------------------------------------------------------------------------
# Percentiles
# ---------------------------------------------------------------------------
def quantiles(x: Iterable[float], qs: Sequence[float] = (0.25, 0.5, 0.75)) -> List[float]:
    """Percentiles with linear interpolation (numpy/pandas default), NaN ignored."""
    v = np.asarray(list(x) if not isinstance(x, np.ndarray) else x, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return [float("nan")] * len(qs)
    return [float(np.quantile(v, q)) for q in qs]


# ---------------------------------------------------------------------------
# Trend: Mann-Kendall + Sen's slope
# ---------------------------------------------------------------------------
def mann_kendall(x: Sequence[float], alpha: float = 0.05) -> Dict[str, float]:
    """Two-sided Mann-Kendall test (tie-corrected variance, continuity-corrected z)."""
    v = np.asarray(x, dtype=float)
    v = v[np.isfinite(v)]
    n = int(v.size)
    if n < 3:
        return {"n": n}
    diff = v[None, :] - v[:, None]
    s = float(np.sum(np.sign(diff[np.triu_indices(n, k=1)])))
    _, counts = np.unique(v, return_counts=True)
    t = counts[counts > 1].astype(float)
    var_s = (n * (n - 1) * (2 * n + 5) - float(np.sum(t * (t - 1) * (2 * t + 5)))) / 18.0
    if var_s <= 0:
        z = 0.0
    elif s > 0:
        z = (s - 1.0) / math.sqrt(var_s)
    elif s < 0:
        z = (s + 1.0) / math.sqrt(var_s)
    else:
        z = 0.0
    p = 2.0 * (1.0 - _st.norm.cdf(abs(z)))
    tau = s / (0.5 * n * (n - 1))
    trend = ("increasing" if z > 0 else "decreasing") if p < alpha else "no trend"
    return {"n": n, "S": s, "var_S": var_s, "z": z, "p": float(p), "tau": tau, "trend": trend}


def sen_slope(x: Sequence[float], t: Optional[Sequence[float]] = None, var_s: Optional[float] = None,
              alpha: float = 0.05) -> Dict[str, float]:
    """Sen's slope with (1 - alpha) CI (Gilbert 1987 rank method, interpolated ranks)."""
    v = np.asarray(x, dtype=float)
    tt = np.arange(v.size, dtype=float) if t is None else np.asarray(t, dtype=float)
    m = np.isfinite(v) & np.isfinite(tt)
    v, tt = v[m], tt[m]
    n = int(v.size)
    if n < 2:
        return {"slope": float("nan"), "lo": float("nan"), "hi": float("nan"), "intercept": float("nan")}
    iu = np.triu_indices(n, k=1)
    dt = (tt[None, :] - tt[:, None])[iu]
    dv = (v[None, :] - v[:, None])[iu]
    good = dt != 0
    slopes = np.sort(dv[good] / dt[good])
    N = slopes.size
    slope = float(np.median(slopes))
    if var_s is None:
        var_s = mann_kendall(v).get("var_S", float("nan"))
    c = _st.norm.ppf(1.0 - alpha / 2.0) * math.sqrt(var_s) if var_s and var_s > 0 else float("nan")

    def _at(rank: float) -> float:          # 1-based rank, linear interpolation
        r = min(max(rank, 1.0), float(N))
        lo_i, hi_i = int(math.floor(r)), int(math.ceil(r))
        f = r - lo_i
        return float(slopes[lo_i - 1] * (1.0 - f) + slopes[hi_i - 1] * f)

    if np.isfinite(c):
        m1 = (N - c) / 2.0
        m2 = (N + c) / 2.0
        lo, hi = _at(m1), _at(m2 + 1.0)
    else:
        lo = hi = float("nan")
    t0 = tt - tt.min()
    intercept = float(np.median(v) - slope * np.median(t0))
    return {"slope": slope, "lo": lo, "hi": hi, "intercept": intercept}


def acf1(x: Sequence[float]) -> float:
    """Lag-1 sample autocorrelation (standard estimator)."""
    v = np.asarray(x, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 3:
        return float("nan")
    c = v - v.mean()
    den = float(np.sum(c * c))
    return float(np.sum(c[:-1] * c[1:]) / den) if den > 0 else float("nan")


def trend_row(years: Sequence[int], values: Sequence[float], alpha: float = 0.05) -> Dict[str, float]:
    """One row of table T1 (Mann-Kendall + Sen with 95 % CI) for a yearly series."""
    y = np.asarray(years, dtype=float)
    v = np.asarray(values, dtype=float)
    order = np.argsort(y)
    y, v = y[order], v[order]
    m = np.isfinite(v) & np.isfinite(y)
    if int(m.sum()) < 3:
        raise ValueError("trend_row needs at least 3 finite values")
    mk = mann_kendall(v[m], alpha=alpha)
    sn = sen_slope(v[m], t=(y[m] - y[m].min()), var_s=mk.get("var_S"), alpha=alpha)
    return {
        "n": int(m.sum()), "year_first": int(y[m].min()), "year_last": int(y[m].max()),
        "mk_S": int(round(mk["S"])), "mk_trend": mk.get("trend"), "mk_p": mk.get("p"), "mk_z": mk.get("z"),
        "mk_tau": mk.get("tau"), "sen_slope": sn["slope"], "sen_lo95": sn["lo"], "sen_hi95": sn["hi"],
        "sen_intercept": sn["intercept"], "acf1": acf1(v[m]),
    }


# ---------------------------------------------------------------------------
# Variance partition: method choice vs interannual climate
# ---------------------------------------------------------------------------
def variance_partition(table: pd.DataFrame) -> Dict[str, float]:
    """table: rows = years, columns = methods, values = national median (or mean).

    V_method = sample variance across methods within each year, averaged over years;
    V_year   = sample variance across years of the year-wise mean over methods;
    share    = V_method / (V_method + V_year).
    """
    t = table.astype(float)
    v_method = float(t.var(axis=1, ddof=1).mean())
    v_year = float(t.mean(axis=1).var(ddof=1))
    den = v_method + v_year
    ratio = v_method / den if den > 0 else float("nan")
    return {"var_method_noise": v_method, "var_year_signal": v_year, "variance_ratio": ratio,
            "percent_method_uncertainty": 100.0 * ratio if np.isfinite(ratio) else float("nan")}


# ---------------------------------------------------------------------------
# Relative climatic suitability classes
# ---------------------------------------------------------------------------
CLASS_PERCENTILES = (25, 50, 75, 90)
CLASS_LABELS = ["0 no season", "1 lowest", "2 below median", "3 above median", "4 high", "5 highest"]


def class_breaks(season_length: Iterable[float], percentiles: Sequence[float] = CLASS_PERCENTILES) -> np.ndarray:
    """P25/P50/P75/P90 of season length among cell-years with a season (length > 0)."""
    v = np.asarray(list(season_length) if not isinstance(season_length, np.ndarray) else season_length, dtype=float)
    v = v[np.isfinite(v) & (v > 0)]
    if v.size == 0:
        raise ValueError("no cell-year with a season: class breaks undefined")
    return np.percentile(v, percentiles)


def classify(season_length: Iterable[float], breaks: Sequence[float]) -> np.ndarray:
    """0 = no season; 1: < P25; 2: [P25, P50); 3: [P50, P75); 4: [P75, P90); 5: >= P90. NaN stays NaN."""
    v = np.asarray(list(season_length) if not isinstance(season_length, np.ndarray) else season_length, dtype=float)
    b = np.asarray(breaks, dtype=float)
    out = np.full(v.shape, np.nan)
    ok = np.isfinite(v)
    out[ok & (v <= 0)] = 0
    pos = ok & (v > 0)
    out[pos] = 1 + np.searchsorted(b, v[pos], side="right")
    return out


def class_ranges(breaks: Sequence[float]) -> List[str]:
    """Day ranges of classes 0-5 for integer season lengths, e.g. ['0', '1-80', ..., '>=135']."""
    b = [int(math.ceil(x)) for x in breaks]
    return ["0", f"1-{b[0] - 1}", f"{b[0]}-{b[1] - 1}", f"{b[1]}-{b[2] - 1}", f"{b[2]}-{b[3] - 1}", f">={b[3]}"]


# ---------------------------------------------------------------------------
# Agreement between a method and the reference
# ---------------------------------------------------------------------------
def cohen_kappa(a: np.ndarray, b: np.ndarray, n_classes: int = 6, weights: Optional[str] = None) -> float:
    """Cohen's kappa for integer classes 0..n_classes-1 (weights: None or 'linear')."""
    a = np.asarray(a, dtype=int)
    b = np.asarray(b, dtype=int)
    k = n_classes
    obs = np.zeros((k, k), dtype=float)
    np.add.at(obs, (a, b), 1.0)
    n = obs.sum()
    if n == 0:
        return float("nan")
    obs /= n
    exp = np.outer(obs.sum(axis=1), obs.sum(axis=0))
    i, j = np.indices((k, k))
    w = (np.abs(i - j) / (k - 1)) if weights == "linear" else (i != j).astype(float)
    den = float(np.sum(w * exp))
    return 1.0 - float(np.sum(w * obs)) / den if den > 0 else float("nan")


def agreement_row(method_len: np.ndarray, ref_len: np.ndarray, breaks: Sequence[float]) -> Dict[str, float]:
    """One row of table T4: agreement of a method with the reference (season length)."""
    m = np.asarray(method_len, dtype=float)
    r = np.asarray(ref_len, dtype=float)
    ok = np.isfinite(m) & np.isfinite(r)
    m, r = m[ok], r[ok]
    cm, cr = classify(m, breaks).astype(int), classify(r, breaks).astype(int)
    d = m - r
    dc = cm - cr
    rho, rho_p = _st.spearmanr(m, r)
    pr, pr_p = _st.pearsonr(m, r)
    p25, p50, p75 = quantiles(d)
    return {
        "n_cells": int(m.size),
        "pct_same_class": 100.0 * float(np.mean(dc == 0)),
        "pct_within_1_class": 100.0 * float(np.mean(np.abs(dc) <= 1)),
        "pct_class_higher": 100.0 * float(np.mean(dc > 0)),
        "pct_class_lower": 100.0 * float(np.mean(dc < 0)),
        "kappa": cohen_kappa(cm, cr),
        "kappa_linear_w": cohen_kappa(cm, cr, weights="linear"),
        "spearman_rho": float(rho), "spearman_p": float(rho_p),
        "pearson_r": float(pr), "pearson_p": float(pr_p),
        "median_delta_days": p50, "p25_delta": p25, "p75_delta": p75,
        "mean_abs_delta_days": float(np.mean(np.abs(d))),
    }


def bias_by_reference_class(method_len: np.ndarray, ref_len: np.ndarray, breaks: Sequence[float]) -> pd.DataFrame:
    """Table T8: difference method - reference by reference class (0-5 and ALL)."""
    m = np.asarray(method_len, dtype=float)
    r = np.asarray(ref_len, dtype=float)
    ok = np.isfinite(m) & np.isfinite(r)
    m, r = m[ok], r[ok]
    cm, cr = classify(m, breaks), classify(r, breaks)
    rows = []
    groups = [(str(int(k)), cr == k) for k in range(6)] + [("ALL", np.ones_like(cr, dtype=bool))]
    for name, sel in groups:
        d = m[sel] - r[sel]
        if d.size == 0:
            continue
        p25, p50, p75 = quantiles(d)
        dc = cm[sel] - cr[sel]
        rows.append({"reference_class": name, "n_cell_years": int(d.size), "mean_delta_days": float(d.mean()),
                     "median_delta_days": p50, "p25_delta": p25, "p75_delta": p75,
                     "pct_same_class": 100.0 * float(np.mean(dc == 0)),
                     "pct_class_lower": 100.0 * float(np.mean(dc < 0)),
                     "pct_class_higher": 100.0 * float(np.mean(dc > 0))})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Distribution diagnostics (skewness, Shapiro-Wilk, Hartigan's dip)
# ---------------------------------------------------------------------------
def distribution_diagnostics(values: Iterable[float], max_n: int = 5000, seed: int = 42) -> Dict[str, float]:
    v = np.asarray(list(values) if not isinstance(values, np.ndarray) else values, dtype=float)
    v = v[np.isfinite(v)]
    out = {"n": int(v.size), "skewness": float("nan"), "shapiro_stat": float("nan"), "shapiro_p": float("nan"),
           "dip_stat": float("nan"), "dip_p": float("nan")}
    if v.size < 3:
        return out
    out["skewness"] = float(_st.skew(v, bias=False))
    vt = v
    if v.size > max_n:
        vt = np.random.default_rng(seed).choice(v, size=max_n, replace=False)
    try:
        w, p = _st.shapiro(vt)
        out["shapiro_stat"], out["shapiro_p"] = float(w), float(p)
    except Exception:
        pass
    try:
        import diptest  # type: ignore
        dip, p = diptest.diptest(vt)
        out["dip_stat"], out["dip_p"] = float(dip), float(p)
    except Exception:
        pass
    return out
