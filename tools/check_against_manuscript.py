#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_against_manuscript.py - does a set of result tables reproduce the paper?

Part A  every number quoted in the Results text of the manuscript (season length, onset, end, total EIP days,
        suitability classes, agreement between methods, trends, variance shares) is recomputed from the tables
        and compared after rounding half-up to the decimals printed in the paper.
Part B  every table is compared with the tables archived from the published run
        (results/run_20260908_131817_6a83be93/), tolerance 1e-6 relative.

Usage
  python tools/check_against_manuscript.py                       # pack in <ROOT>/out/.../00_MANUSCRIPT_PACK
  python tools/check_against_manuscript.py --pack <folder>       # a pack written by step09, or results/run_*
  python tools/check_against_manuscript.py --only-manuscript     # Part A only
  python tools/check_against_manuscript.py --only-tables         # Part B only (e.g. for another period)

Exit code 0 = everything matches, 1 = at least one difference, 2 = input missing.
The tables of Part A are those written by step09 (T1-T9, per_cell_classes_ALL_YEARS.csv).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))
import ddu_stats as ds  # noqa: E402

REFERENCE_RUN = REPO / "results" / "run_20260908_131817_6a83be93"
METHODS = ["midpoint", "daily_mean", "hourly_sum", "be_sine"]
REF_YEARS = [2005, 2011, 2017, 2024]


def _utf8_stdout() -> None:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


# ------------------------------------------------------------------ tables
class Tables:
    """T1-T9 and the per-cell file, from a step09 pack or from results/run_*."""

    def __init__(self, folder: Path):
        self.folder = Path(folder)
        self._cache = {}

    def _read(self, *names: str) -> pd.DataFrame:
        for n in names:
            p = self.folder / n
            if p.exists():
                if n not in self._cache:
                    self._cache[n] = pd.read_csv(p)
                return self._cache[n]
        raise FileNotFoundError(f"{' / '.join(names)} not found in {self.folder}")

    @property
    def t1(self): return self._read("T1_trend_MK_Sen_all_methods.csv")
    @property
    def t2(self): return self._read("T2_distribution_diagnostics_summary.csv")
    @property
    def t4(self): return self._read("T4_class_agreement_pooled.csv")
    @property
    def t5(self): return self._read("T5_national_summary_all_methods.csv")
    @property
    def t6b(self): return self._read("T6b_pairwise_delta_by_year.csv")
    @property
    def t7(self): return self._read("T7_suitability_class_counts_by_year.csv")
    @property
    def t9(self): return self._read("T9_median_gap_diagnostic.csv")
    @property
    def cells(self): return self._read("per_cell_classes_ALL_YEARS.csv")

    @property
    def t3(self):
        t = self._read("T3_variance_ratio_national.csv", "T3_variance_ratio_national_and_percell.csv")
        return t[t["level"] == "national_series"] if "level" in t.columns else t

    # ---- single values
    def nat(self, var, method, year, col="median") -> float:
        q = self.t5[(self.t5.variable == var) & (self.t5.method == method) & (self.t5.year == year)]
        return float(q[col].iloc[0])

    def nat_series(self, var, method, col="median") -> pd.Series:
        q = self.t5[(self.t5.variable == var) & (self.t5.method == method)]
        return q.set_index("year")[col].astype(float).sort_index()

    def trend(self, var, method, series, col) -> float:
        q = self.t1[(self.t1.variable == var) & (self.t1.method == method) & (self.t1.series == series)]
        return float(q[col].iloc[0])

    def share(self, var, series) -> float:
        q = self.t3[(self.t3.variable == var) & (self.t3.series == series)]
        return float(q["percent_method_uncertainty"].iloc[0])

    def var_part(self, var, series, col) -> float:
        q = self.t3[(self.t3.variable == var) & (self.t3.series == series)]
        return float(q[col].iloc[0])

    def agree(self, method, col) -> float:
        return float(self.t4[self.t4.method == method][col].iloc[0])

    def delta(self, var, m1, m2, col="median_delta", years=None) -> pd.Series:
        q = self.t6b[(self.t6b.variable == var) & (self.t6b.method1 == m1) & (self.t6b.method2 == m2)]
        s = q.set_index("year")[col].astype(float).sort_index()
        return s if years is None else s.loc[list(years)]

    def class_pct(self, year, classes, method="hourly_sum") -> float:
        q = self.t7[(self.t7.method == method) & (self.t7.year == year)]
        return 100.0 * float(q[q["class"].isin(classes)]["count"].sum()) / float(q["count"].sum())

    def gap(self, method, year, col) -> float:
        q = self.t9[(self.t9.method == method) & (self.t9.year == year)]
        return float(q[col].iloc[0])

    def dist(self, var, method, col) -> float:
        q = self.t2[(self.t2.variable == var) & (self.t2.method == method)]
        return float(q[col].iloc[0])


# ------------------------------------------------------------------ Part A
def manuscript_checks(T: Tables):
    """(description, function returning the computed value, value printed in the manuscript, decimals)."""
    C = []

    def add(desc, fn, expected, dec=0):
        C.append((desc, fn, expected, dec))

    cells = lambda: T.cells                                                       # noqa: E731
    hs = lambda: cells()["hourly_sum"].to_numpy(float)                            # noqa: E731
    n_season = lambda: int((hs() > 0).sum())                                      # noqa: E731

    def lost(method):      # season under the hourly sum, none under the method
        return int(((cells()[method].to_numpy(float) <= 0) & (hs() > 0)).sum())

    def gained(method):    # season under the method, none under the hourly sum
        return int(((cells()[method].to_numpy(float) > 0) & (hs() <= 0)).sum())

    def lost_all_in_class1(method):
        d = cells()
        m = (d[method].to_numpy(float) <= 0) & (hs() > 0)
        return float((d.loc[m, "class_hourly_sum"] == 1).all())

    def one_class_lower(method):
        d = cells()
        return 100.0 * float(((d[f"class_{method}"] - d["class_hourly_sum"]) == -1).mean())

    # ---- data set and suitability classes (Table 2, section 3.2)
    add("Grid cells", lambda: cells().groupby("year").size().iloc[0], 2912)
    add("Cell-years", lambda: len(cells()), 58240)
    for name, k, val in (("P25", 0, 81), ("P50", 1, 108), ("P75", 2, 124), ("P90", 3, 135)):
        add(f"Table 2: class limit {name} of hourly-sum season length (days)",
            lambda k=k: ds.class_breaks(hs())[k], val)
    add("Cell-years with a season under the hourly sum", n_season, 53278)
    add("2005: classes 0-1 (%)", lambda: T.class_pct(2005, [0, 1]), 48.5, 1)
    add("2005: class 4 (%)", lambda: T.class_pct(2005, [4]), 8.1, 1)
    add("2005: class 5 (%)", lambda: T.class_pct(2005, [5]), 0.4, 1)
    add("2011: cells without a season (%)", lambda: T.class_pct(2011, [0]), 14.7, 1)
    add("2011: class 4 (%)", lambda: T.class_pct(2011, [4]), 31.4, 1)
    add("2011: class 5 (%)", lambda: T.class_pct(2011, [5]), 18.1, 1)
    add("2017: class 3 (%)", lambda: T.class_pct(2017, [3]), 34.5, 1)
    add("2017: class 4 (%)", lambda: T.class_pct(2017, [4]), 11.3, 1)
    add("2017: class 5 (%)", lambda: T.class_pct(2017, [5]), 0.0, 1)
    add("2024: cells without a season (%)", lambda: T.class_pct(2024, [0]), 1.4, 1)
    add("2024: classes 3-5 (%)", lambda: T.class_pct(2024, [3, 4, 5]), 62.1, 1)

    # ---- season onset and end (sections 3.1.1, 3.1.2)
    def per_method(var, year, col="median"):
        return [T.nat(var, m, year, col) for m in METHODS]

    add("Onset 2005, per-method medians: earliest (DOY)", lambda: min(per_method("start_doy", 2005)), 167)
    add("Onset 2005, per-method medians: latest (DOY)", lambda: max(per_method("start_doy", 2005)), 182)
    add("Onset 2024, per-method medians: earliest (DOY)", lambda: min(per_method("start_doy", 2024)), 160)
    add("Onset 2024, per-method medians: latest (DOY)", lambda: max(per_method("start_doy", 2024)), 163)
    add("End 2005, per-method medians: earliest (DOY)", lambda: min(per_method("end_doy", 2005)), 266)
    add("End 2005, per-method medians: latest (DOY)", lambda: max(per_method("end_doy", 2005)), 267)
    add("End 2024, per-method medians: earliest (DOY)", lambda: min(per_method("end_doy", 2024)), 274)
    add("End 2024, per-method medians: latest (DOY)", lambda: max(per_method("end_doy", 2024)), 275)
    add("Consensus onset 2005, median (DOY)", lambda: T.nat("start_doy", "consensus", 2005), 170, 1)
    add("Consensus onset 2005, P25 (DOY)", lambda: T.nat("start_doy", "consensus", 2005, "p25"), 164, 1)
    add("Consensus onset 2005, P75 (DOY)", lambda: T.nat("start_doy", "consensus", 2005, "p75"), 202, 1)
    add("Consensus onset 2024, median (DOY)", lambda: T.nat("start_doy", "consensus", 2024), 161.5, 1)
    add("Consensus onset, earliest yearly median (DOY)", lambda: T.nat_series("start_doy", "consensus").min(), 149.5, 1)
    add("Consensus onset, year of the earliest median", lambda: T.nat_series("start_doy", "consensus").idxmin(), 2018)
    add("Consensus onset, latest yearly median (DOY)", lambda: T.nat_series("start_doy", "consensus").max(), 178.5, 1)
    add("Consensus onset, year of the latest median", lambda: T.nat_series("start_doy", "consensus").idxmax(), 2006)
    add("Consensus onset, Mann-Kendall p (median series)", lambda: T.trend("start_doy", "consensus", "median", "mk_p"), 0.85, 2)
    add("Consensus end 2005, median (DOY)", lambda: T.nat("end_doy", "consensus", 2005), 266, 1)
    add("Consensus end 2005, P25 (DOY)", lambda: T.nat("end_doy", "consensus", 2005, "p25"), 244, 1)
    add("Consensus end 2005, P75 (DOY)", lambda: T.nat("end_doy", "consensus", 2005, "p75"), 272, 1)
    add("Consensus end 2011, median (DOY)", lambda: T.nat("end_doy", "consensus", 2011), 282, 1)
    add("Consensus end 2017, median (DOY)", lambda: T.nat("end_doy", "consensus", 2017), 268, 1)
    add("Consensus end 2023, median (DOY)", lambda: T.nat("end_doy", "consensus", 2023), 283, 1)
    add("Consensus end 2024, median (DOY)", lambda: T.nat("end_doy", "consensus", 2024), 274, 1)
    add("Consensus end, year of the latest median", lambda: T.nat_series("end_doy", "consensus").idxmax(), 2023)
    add("Consensus end, Mann-Kendall p (median series)", lambda: T.trend("end_doy", "consensus", "median", "mk_p"), 0.14, 2)

    def end_iqr(years, col, fn):
        q = T.t6b[(T.t6b.variable == "end_doy")]
        if years is not None:
            q = q[q.year.isin(years)]
        return fn(q[col].astype(float))

    add("Paired differences in season end, all pairs, reference years: lowest P25 (days)", lambda: end_iqr(REF_YEARS, "p25", min), -3)
    add("Paired differences in season end, all pairs, reference years: highest P75 (days)", lambda: end_iqr(REF_YEARS, "p75", max), 1)
    add("Paired differences in season end, all pairs, 20 years: lowest P25 (days)", lambda: end_iqr(None, "p25", min), -5)
    add("Paired differences in season end, all pairs, 20 years: highest P75 (days)", lambda: end_iqr(None, "p75", max), 2)

    # ---- season length (section 3.1.3)
    for year, med, p25, p75 in ((2005, 75.5, 27.5, 108.5), (2011, 118.5, None, None), (2017, 99, None, None),
                                (2024, 113.5, 93.5, 125)):
        add(f"Consensus season length {year}, median (days)", lambda y=year: T.nat("season_length", "consensus", y), med, 1)
        if p25 is not None:
            add(f"Consensus season length {year}, P25 (days)", lambda y=year: T.nat("season_length", "consensus", y, "p25"), p25, 1)
            add(f"Consensus season length {year}, P75 (days)", lambda y=year: T.nat("season_length", "consensus", y, "p75"), p75, 1)
    add("Consensus season length 2023 above 110 days", lambda: float(T.nat("season_length", "consensus", 2023) > 110), 1)
    add("Consensus season length 2024 above 110 days", lambda: float(T.nat("season_length", "consensus", 2024) > 110), 1)

    for m, label, ref_lo, ref_hi, all_lo, all_hi in (("be_sine", "BE-sine", -2, -1, -2, 0),
                                                     ("daily_mean", "hourly average", -4, -3, -4, -2),
                                                     ("midpoint", "min-max average", -7, -4, -8, -3)):
        add(f"Season length, {label} minus hourly sum, median difference, reference years: lowest (days)",
            lambda m=m: T.delta("season_length", m, "hourly_sum", years=REF_YEARS).min(), ref_lo)
        add(f"Season length, {label} minus hourly sum, median difference, reference years: highest (days)",
            lambda m=m: T.delta("season_length", m, "hourly_sum", years=REF_YEARS).max(), ref_hi)
        add(f"Season length, {label} minus hourly sum, median difference, 20 years: lowest (days)",
            lambda m=m: T.delta("season_length", m, "hourly_sum").min(), all_lo)
        add(f"Season length, {label} minus hourly sum, median difference, 20 years: highest (days)",
            lambda m=m: T.delta("season_length", m, "hourly_sum").max(), all_hi)
    add("Season length, min-max average minus hourly sum: lowest P25 over the 20 years (days)",
        lambda: T.delta("season_length", "midpoint", "hourly_sum", "p25").min(), -14)
    add("Season length, min-max average minus hourly sum: year of the lowest P25",
        lambda: T.delta("season_length", "midpoint", "hourly_sum", "p25").idxmin(), 2009)
    add("Onset, min-max average minus hourly sum, median difference, 20 years: lowest (days)",
        lambda: T.delta("start_doy", "midpoint", "hourly_sum").min(), 2)
    add("Onset, min-max average minus hourly sum, median difference, 20 years: highest (days)",
        lambda: T.delta("start_doy", "midpoint", "hourly_sum").max(), 7)
    add("Onset, BE-sine minus hourly sum, median difference, 20 years: lowest (days)",
        lambda: T.delta("start_doy", "be_sine", "hourly_sum").min(), 0)
    add("Onset, BE-sine minus hourly sum, median difference, 20 years: highest (days)",
        lambda: T.delta("start_doy", "be_sine", "hourly_sum").max(), 2)

    # ---- 2005 and 2014: marginal medians against paired differences (section 3.2, after Fig. 3)
    for year, hsm, bem, gap, dmean, between in ((2005, 83, 75, 8, 2.9, 3.3), (2014, 97, 86, 11, 3.0, 2.8)):
        add(f"{year}: national median, hourly sum (days)", lambda y=year: T.nat("season_length", "hourly_sum", y), hsm)
        add(f"{year}: national median, BE-sine (days)", lambda y=year: T.nat("season_length", "be_sine", y), bem)
        add(f"{year}: difference between the two medians (days)",
            lambda y=year: T.nat("season_length", "hourly_sum", y) - T.nat("season_length", "be_sine", y), gap)
        add(f"{year}: difference between the two national means (days)",
            lambda y=year: T.nat("season_length", "hourly_sum", y, "mean") - T.nat("season_length", "be_sine", y, "mean"), dmean, 1)
        add(f"{year}: median paired per-cell difference, hourly sum minus BE-sine (days)",
            lambda y=year: -T.delta("season_length", "be_sine", "hourly_sum").loc[y], 1)
        add(f"{year}: cells between the two medians (%)",
            lambda y=year: T.gap("be_sine", y, "pct_ref_cells_between_medians_strict"), between, 1)
    for m, val in (("midpoint", 63.2), ("daily_mean", 66.2), ("be_sine", 68.2), ("hourly_sum", 71.1)):
        add(f"2005: national mean season length, {m} (days)", lambda m=m: T.nat("season_length", m, 2005, "mean"), val, 1)

    def mean_gap():
        return T.nat_series("season_length", "hourly_sum", "mean") - T.nat_series("season_length", "be_sine", "mean")

    add("Hourly sum minus BE-sine, national means: smallest yearly difference (days)", lambda: mean_gap().min(), 1.0, 1)
    add("Hourly sum minus BE-sine, national means: largest yearly difference (days)", lambda: mean_gap().max(), 3.6, 1)
    add("Onset 2005, min-max average: national median (DOY)", lambda: T.nat("start_doy", "midpoint", 2005), 182)
    others = [m for m in METHODS if m != "midpoint"]
    add("Onset 2005, other methods: earliest median (DOY)", lambda: min(T.nat("start_doy", m, 2005) for m in others), 167)
    add("Onset 2005, other methods: latest median (DOY)", lambda: max(T.nat("start_doy", m, 2005) for m in others), 170)
    add("Onset 2005, min-max average: national mean (DOY)", lambda: T.nat("start_doy", "midpoint", 2005, "mean"), 184)
    add("Onset 2005, other methods: earliest mean (DOY)", lambda: min(T.nat("start_doy", m, 2005, "mean") for m in others), 178)
    add("Onset 2005, other methods: latest mean (DOY)", lambda: max(T.nat("start_doy", m, 2005, "mean") for m in others), 181)

    # ---- distribution (section 3.2)
    add("Shapiro-Wilk p < 0.001 in every year, all four methods",
        lambda: float(all(T.dist("season_length", m, "years_shapiro_p_lt_0.001") == T.dist("season_length", m, "n_years") for m in METHODS)), 1)
    add("Dip test p < 0.001 in every year, all four methods",
        lambda: float(all(T.dist("season_length", m, "years_dip_p_lt_0.001") == T.dist("season_length", m, "n_years") for m in METHODS)), 1)
    add("Skewness of season length, most negative yearly value over the four methods",
        lambda: min(T.dist("season_length", m, "skewness_min") for m in METHODS), -1.7, 1)
    add("Skewness of season length, least negative yearly value over the four methods",
        lambda: max(T.dist("season_length", m, "skewness_max") for m in METHODS), -0.2, 1)

    # ---- trends (section 3.2)
    add("Consensus median series: Sen's slope (days/yr)", lambda: T.trend("season_length", "consensus", "median", "sen_slope"), 1.10, 2)
    add("Consensus median series: 95% CI, lower", lambda: T.trend("season_length", "consensus", "median", "sen_lo95"), -0.26, 2)
    add("Consensus median series: 95% CI, upper", lambda: T.trend("season_length", "consensus", "median", "sen_hi95"), 2.00, 2)
    add("Consensus median series: Kendall tau", lambda: T.trend("season_length", "consensus", "median", "mk_tau"), 0.24, 2)
    add("Consensus median series: p", lambda: T.trend("season_length", "consensus", "median", "mk_p"), 0.15, 2)

    def over_methods(series, col, fn):
        return fn(T.trend("season_length", m, series, col) for m in METHODS)

    add("Median series, four methods: lowest tau", lambda: over_methods("median", "mk_tau", min), 0.19, 2)
    add("Median series, four methods: highest tau", lambda: over_methods("median", "mk_tau", max), 0.25, 2)
    add("Median series, four methods: lowest p", lambda: over_methods("median", "mk_p", min), 0.13, 2)
    add("Median series, four methods: highest p", lambda: over_methods("median", "mk_p", max), 0.25, 2)
    add("Median series, four methods: lowest Sen's slope", lambda: over_methods("median", "sen_slope", min), 1.00, 2)
    add("Median series, four methods: highest Sen's slope", lambda: over_methods("median", "sen_slope", max), 1.11, 2)
    for m, tau, p in (("midpoint", 0.37, 0.025), ("daily_mean", 0.37, 0.025), ("be_sine", 0.37, 0.025), ("hourly_sum", 0.35, 0.035)):
        add(f"Mean series, {m}: tau", lambda m=m: T.trend("season_length", m, "mean", "mk_tau"), tau, 2)
        add(f"Mean series, {m}: p", lambda m=m: T.trend("season_length", m, "mean", "mk_p"), p, 3)
    add("Mean series, four methods: lowest Sen's slope", lambda: over_methods("mean", "sen_slope", min), 0.90, 2)
    add("Mean series, four methods: highest Sen's slope", lambda: over_methods("mean", "sen_slope", max), 0.95, 2)
    add("Consensus mean series: tau", lambda: T.trend("season_length", "consensus", "mean", "mk_tau"), 0.36, 2)
    add("Consensus mean series: p", lambda: T.trend("season_length", "consensus", "mean", "mk_p"), 0.030, 3)
    add("Consensus mean series: Sen's slope (days/yr)", lambda: T.trend("season_length", "consensus", "mean", "sen_slope"), 0.91, 2)
    add("Total EIP days, consensus median: Sen's slope (days/yr)", lambda: T.trend("total_eip_days", "consensus", "median", "sen_slope"), 1.15, 2)
    add("Total EIP days, consensus median: 95% CI, lower", lambda: T.trend("total_eip_days", "consensus", "median", "sen_lo95"), -0.26, 2)
    add("Total EIP days, consensus median: 95% CI, upper", lambda: T.trend("total_eip_days", "consensus", "median", "sen_hi95"), 1.96, 2)
    add("Total EIP days, consensus median: tau", lambda: T.trend("total_eip_days", "consensus", "median", "mk_tau"), 0.23, 2)
    add("Total EIP days, consensus median: p", lambda: T.trend("total_eip_days", "consensus", "median", "mk_p"), 0.16, 2)

    # ---- variance shares
    add("Method share of variance, season length, median series (%)", lambda: T.share("season_length", "median"), 6.0, 1)
    add("Year-to-year share, season length, median series (%)", lambda: 100.0 - T.share("season_length", "median"), 94.0, 1)
    add("V_method, median series (d2)", lambda: T.var_part("season_length", "median", "var_method_noise"), 12.3, 1)
    add("V_year, median series (d2)", lambda: T.var_part("season_length", "median", "var_year_signal"), 192.5, 1)
    add("Method share of variance, season length, mean series (%)", lambda: T.share("season_length", "mean"), 9.0, 1)
    add("V_method, mean series (d2)", lambda: T.var_part("season_length", "mean", "var_method_noise"), 10.0, 1)
    add("V_year, mean series (d2)", lambda: T.var_part("season_length", "mean", "var_year_signal"), 100.8, 1)
    for var, label, med, mean in (("start_doy", "onset", 11.2, 9.9), ("end_doy", "end", 1.0, 1.9),
                                  ("total_eip_days", "total EIP days", 8.5, 8.6)):
        add(f"Method share of variance, {label}, median series (%)", lambda v=var: T.share(v, "median"), med, 1)
        add(f"Method share of variance, {label}, mean series (%)", lambda v=var: T.share(v, "mean"), mean, 1)

    # ---- agreement of suitability classes with the hourly sum
    for m, label, same in (("be_sine", "BE-sine", 90.3), ("daily_mean", "hourly average", 81.2), ("midpoint", "min-max average", 73.4)):
        add(f"Same class as the hourly sum, {label} (%)", lambda m=m: T.agree(m, "pct_same_class"), same, 1)
    add("Spearman rho of season length, BE-sine vs hourly sum", lambda: T.agree("be_sine", "spearman_rho"), 0.993, 3)
    add("BE-sine in a lower class than the hourly sum (%)", lambda: T.agree("be_sine", "pct_class_lower"), 8.5, 1)
    add("BE-sine exactly one class lower (%)", lambda: one_class_lower("be_sine"), 8.2, 1)
    add("BE-sine in a higher class than the hourly sum (%)", lambda: T.agree("be_sine", "pct_class_higher"), 1.2, 1)
    for m, label, n, pct in (("midpoint", "min-max average", 1359, 2.6), ("daily_mean", "hourly average", 682, 1.3),
                             ("be_sine", "BE-sine", 555, 1.0)):
        add(f"Season under the hourly sum, none under the {label} (cell-years)", lambda m=m: lost(m), n)
        add(f"Same, % of the cell-years with a season under the hourly sum ({label})",
            lambda m=m: 100.0 * lost(m) / n_season(), pct, 1)
        add(f"Same, all in class 1 under the hourly sum ({label})", lambda m=m: lost_all_in_class1(m), 1)
    add("Season under BE-sine, none under the hourly sum (cell-years)", lambda: gained("be_sine"), 29)
    add("Season under the hourly average, none under the hourly sum (cell-years)", lambda: gained("daily_mean"), 0)
    add("Season under the min-max average, none under the hourly sum (cell-years)", lambda: gained("midpoint"), 0)
    return C


# ------------------------------------------------------------------ Part B
TABLES = [
    # pack file(s), reference file(s), keys, value columns
    ("T1_trend_MK_Sen_all_methods", ["variable", "method", "series"],
     ["mk_S", "mk_p", "mk_z", "mk_tau", "sen_slope", "sen_lo95", "sen_hi95", "sen_intercept", "acf1"]),
    ("T2_distribution_diagnostics_summary", ["variable", "method"],
     ["n_years", "years_shapiro_p_lt_0.001", "years_shapiro_p_lt_0.05", "years_dip_p_lt_0.05", "years_dip_p_lt_0.001",
      "skewness_median", "skewness_min", "skewness_max", "dip_stat_median", "n_cells_median"]),
    ("T3", ["variable", "series"],
     ["var_method_noise", "var_year_signal", "variance_ratio", "percent_method_uncertainty"]),
    ("T4_class_agreement_pooled", ["method"],
     ["n_cells", "pct_same_class", "pct_within_1_class", "pct_class_higher", "pct_class_lower", "kappa",
      "kappa_linear_w", "spearman_rho", "pearson_r", "median_delta_days", "p25_delta", "p75_delta",
      "mean_abs_delta_days"]),
    ("T4b_class_agreement_by_year", ["method", "year"],
     ["n_cells", "pct_same_class", "pct_within_1_class", "pct_class_higher", "pct_class_lower", "kappa",
      "kappa_linear_w", "spearman_rho", "pearson_r", "median_delta_days", "p25_delta", "p75_delta",
      "mean_abs_delta_days"]),
    ("T5_national_summary_all_methods", ["variable", "method", "year"],
     ["median", "mean", "p25", "p75", "pct_with_season", "count"]),
    ("T6b_pairwise_delta_by_year", ["variable", "year", "method1", "method2"],
     ["median_delta", "p25", "p75", "mean", "sd", "count"]),
    ("T7_suitability_class_counts_by_year", ["year", "method", "class"],
     ["count", "pct", "total_valid", "nonzero_count", "nonzero_pct", "median_season_length"]),
    ("T8_bias_by_reference_class", ["method", "reference_class"],
     ["n_cell_years", "mean_delta_days", "median_delta_days", "p25_delta", "p75_delta", "pct_same_class",
      "pct_class_lower", "pct_class_higher"]),
    ("T9_median_gap_diagnostic", ["method", "year"],
     ["n_cells", "median_method", "median_reference", "delta_medians", "mean_method", "mean_reference",
      "delta_means", "mean_paired_delta", "median_paired_delta", "pct_ref_cells_between_medians_strict",
      "pct_ref_cells_between_medians_inclusive", "pct_method_cells_between_medians_strict"]),
    ("per_cell_classes_ALL_YEARS", ["year", "i", "j"],
     ["midpoint", "daily_mean", "hourly_sum", "be_sine", "consensus", "class_midpoint", "class_daily_mean",
      "class_hourly_sum", "class_be_sine", "class_consensus"]),
]


def _load_for_compare(folder: Path, name: str):
    if name == "T3":
        try:
            return Tables(folder).t3
        except FileNotFoundError:
            return None
    p = Path(folder) / f"{name}.csv"
    return pd.read_csv(p) if p.exists() else None


def compare_table(pack: Path, ref: Path, spec, rtol: float = 1e-6):
    name, keys, cols = spec
    a, b = _load_for_compare(pack, name), _load_for_compare(ref, name)
    if a is None or b is None:
        return {"table": name, "status": "MISSING (" + ("pack" if a is None else "reference") + ")", "rows": 0}
    a, b = a.copy(), b.copy()
    for k in keys:
        a[k], b[k] = a[k].astype(str), b[k].astype(str)
    m = a.merge(b, on=keys, how="outer", suffixes=("", "_ref"), indicator=True)
    only = int((m["_merge"] != "both").sum())
    m = m[m["_merge"] == "both"]
    bad = []
    for c in cols:
        if c not in a.columns or c not in b.columns:
            bad.append(f"{c} (column missing)")
            continue
        x = pd.to_numeric(m[c], errors="coerce").to_numpy(float)
        y = pd.to_numeric(m[c + "_ref"], errors="coerce").to_numpy(float)
        nan_mismatch = np.isnan(x) != np.isnan(y)
        both = ~np.isnan(x) & ~np.isnan(y)
        err = np.abs(x[both] - y[both]) / np.maximum(1.0, np.abs(y[both]))
        e = float(err.max()) if err.size else 0.0
        if e > rtol or nan_mismatch.any():
            bad.append(f"{c} (max rel. diff {e:.2e}, {int((err > rtol).sum())} rows; NaN mismatches {int(nan_mismatch.sum())})")
    if only:
        bad.append(f"{only} rows present on one side only")
    if len(m) == 0:
        bad.append("no common rows")
    return {"table": name, "status": "OK" if not bad else "DIFFERENT: " + "; ".join(bad), "rows": len(m)}


def main(argv=None) -> int:
    _utf8_stdout()
    root = Path(os.environ.get("DDU_ROOT", str(REPO)))
    ap = argparse.ArgumentParser(description="Compare result tables with the paper and with the published run")
    ap.add_argument("--pack", default=str(root / "out" / "12_method_compare" / "win30" / "00_MANUSCRIPT_PACK"),
                    help="folder with T1-T9 and per_cell_classes_ALL_YEARS.csv (step09 pack or results/run_*)")
    ap.add_argument("--reference", default=str(REFERENCE_RUN), help="archived tables of the published run")
    ap.add_argument("--only-manuscript", action="store_true", help="Part A only")
    ap.add_argument("--only-tables", action="store_true", help="Part B only")
    a = ap.parse_args(argv)
    pack, ref = Path(a.pack), Path(a.reference)
    if not (pack / "T5_national_summary_all_methods.csv").exists():
        print(f"tables not found in: {pack}  (run: python run_pipeline.py --from 9 --to 10)")
        return 2

    failures = 0
    if not a.only_tables:
        T = Tables(pack)
        checks = manuscript_checks(T)
        n_ok = 0
        print(f"Part A - numbers printed in the manuscript ({len(checks)} checks; tables: {pack})")
        for desc, fn, expected, dec in checks:
            try:
                value = float(fn())
                got = ds.round_half_up(value, dec)
                ok = abs(got - float(expected)) < 10 ** (-dec) / 2
                detail = f"manuscript {expected}, computed {got} ({value:.6g})"
            except Exception as e:  # a missing year, table or column is a failed check, not a crash
                ok, detail = False, f"manuscript {expected}, not computable ({type(e).__name__}: {e})"
            n_ok += ok
            failures += 0 if ok else 1
            print(f"  [{'OK ' if ok else 'BAD'}] {desc}: {detail}")
        print(f"  Part A: {n_ok} of {len(checks)} reproduced")

    if not a.only_manuscript:
        same = pack.resolve() == ref.resolve()
        print(f"\nPart B - tables compared with the published run ({ref})")
        if same:
            print("  skipped: the tables checked are the archived tables themselves")
        else:
            n_ok = 0
            for spec in TABLES:
                r = compare_table(pack, ref, spec)
                ok = r["status"] == "OK"
                n_ok += ok
                failures += 0 if ok else 1
                print(f"  {r['table']:38s} rows {r['rows']:>6}  {r['status']}")
            print(f"  Part B: {n_ok} of {len(TABLES)} tables identical")

    print("\nRESULT:", "all checks passed" if failures == 0 else f"{failures} check(s) failed")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
