#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_against_manuscript.py - does a regenerated manuscript pack reproduce the paper?

Part A  numbers quoted in the manuscript (season length; text, Tables 2 and 5, abstract) are recomputed
        from the pack written by step09 and compared after rounding half-up to the decimals printed in
        the paper.
Part B  every table of the pack is compared with the tables archived from the published run
        (results/run_20260908_131817_6a83be93/), for all variables present in both
        (tolerance 1e-6 relative).

Usage
  python tools/check_against_manuscript.py                       # pack in <ROOT>/out/.../00_MANUSCRIPT_PACK
  python tools/check_against_manuscript.py --pack <folder>       # any other pack
  python tools/check_against_manuscript.py --only-manuscript     # Part A only

Exit code 0 = everything matches, 1 = at least one difference, 2 = input missing.
"""
from __future__ import annotations

import argparse
import json
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


# ------------------------------------------------------------------ Part A
def manuscript_checks(pack: Path):
    kn = json.loads((pack / "key_numbers.json").read_text(encoding="utf-8"))
    t5 = pd.read_csv(pack / "T5_national_summary_all_methods.csv")
    t7 = pd.read_csv(pack / "T7_suitability_class_counts_by_year.csv")
    t1 = pd.read_csv(pack / "T1_trend_MK_Sen_all_methods.csv")

    def t5v(method, year, col):
        q = t5[(t5.variable == "season_length") & (t5.method == method) & (t5.year == year)]
        return float(q[col].iloc[0])

    def t7pct(year, classes):
        q = t7[(t7.method == "hourly_sum") & (t7.year == year) & (t7["class"].isin(classes))]
        return float(q["pct"].sum())

    def trend(method, series, col):
        q = t1[(t1.variable == "season_length") & (t1.method == method) & (t1.series == series)]
        return float(q[col].iloc[0])

    agree = kn["class_agreement_vs_hourly_sum"]
    nos = kn["no_season_under_method_but_season_under_hourly_sum"]
    nref = kn["n_cell_years_with_season_hourly_sum"]
    gap = kn["median_gap_be_sine_vs_hourly_sum"]
    rng = kn["mean_difference_hourly_sum_minus_be_sine_range"]
    mean05 = kn["national_mean_length_first_year"]
    var = kn["variance_share_method_pct"]

    # (description, computed value, value printed in the manuscript, decimals)
    return [
        ("Table 2: class limit P25 (days)", kn["class_breaks"]["class_breaks_days"][0], 81, 0),
        ("Table 2: class limit P50 (days)", kn["class_breaks"]["class_breaks_days"][1], 108, 0),
        ("Table 2: class limit P75 (days)", kn["class_breaks"]["class_breaks_days"][2], 124, 0),
        ("Table 2: class limit P90 (days)", kn["class_breaks"]["class_breaks_days"][3], 135, 0),
        ("Cell-years with a season under the hourly sum", nref, 53278, 0),
        ("Same class as hourly sum, BE-sine (%)", agree["be_sine"]["pct_same_class"], 90.3, 1),
        ("Same class as hourly sum, hourly average (%)", agree["daily_mean"]["pct_same_class"], 81.2, 1),
        ("Same class as hourly sum, min-max average (%)", agree["midpoint"]["pct_same_class"], 73.4, 1),
        ("Spearman rho, BE-sine vs hourly sum", agree["be_sine"]["spearman_rho"], 0.993, 3),
        ("No season under min-max average, season under hourly sum (cell-years)", nos["midpoint"], 1359, 0),
        ("No season under hourly average, season under hourly sum (cell-years)", nos["daily_mean"], 682, 0),
        ("No season under BE-sine, season under hourly sum (cell-years)", nos["be_sine"], 555, 0),
        ("Same, min-max average, % of cell-years with a season", 100.0 * nos["midpoint"] / nref, 2.6, 1),
        ("Variance share of method choice, median series (%)", var["season_length|median"], 6, 0),
        ("Variance share of method choice, mean series (%)", var["season_length|mean"], 9, 0),
        ("Consensus median season length 2005 (days)", t5v("consensus", 2005, "median"), 75.5, 1),
        ("Consensus median season length 2011 (days)", t5v("consensus", 2011, "median"), 118.5, 1),
        ("Consensus median season length 2017 (days)", t5v("consensus", 2017, "median"), 99, 1),
        ("Consensus median season length 2024 (days)", t5v("consensus", 2024, "median"), 113.5, 1),
        ("Consensus IQR 2005, lower (days)", t5v("consensus", 2005, "p25"), 27.5, 1),
        ("Consensus IQR 2005, upper (days)", t5v("consensus", 2005, "p75"), 108.5, 1),
        ("Consensus IQR 2024, lower (days)", t5v("consensus", 2024, "p25"), 93.5, 1),
        ("Consensus IQR 2024, upper (days)", t5v("consensus", 2024, "p75"), 125, 1),
        ("Median series trend: Mann-Kendall tau", trend("consensus", "median", "mk_tau"), 0.24, 2),
        ("Median series trend: p", trend("consensus", "median", "mk_p"), 0.15, 2),
        ("Median series trend: Sen's slope (days/yr)", trend("consensus", "median", "sen_slope"), 1.10, 2),
        ("Median series trend: 95% CI lower", trend("consensus", "median", "sen_lo95"), -0.26, 2),
        ("Median series trend: 95% CI upper", trend("consensus", "median", "sen_hi95"), 2.00, 2),
        ("Mean series trend: Sen's slope (days/yr)", trend("consensus", "mean", "sen_slope"), 0.9, 1),
        ("Mean series trend significant (p < 0.05)", float(trend("consensus", "mean", "mk_p") < 0.05), 1, 0),
        ("Median series trend not significant (p >= 0.05)", float(trend("consensus", "median", "mk_p") >= 0.05), 1, 0),
        ("Hourly sum minus BE-sine, national means, smallest yearly difference (days)", rng[0], 1.0, 1),
        ("Hourly sum minus BE-sine, national means, largest yearly difference (days)", rng[1], 3.6, 1),
        ("2005 median, hourly sum (days)", gap["2005"]["median_hourly_sum"], 83, 0),
        ("2005 median, BE-sine (days)", gap["2005"]["median_be_sine"], 75, 0),
        ("2014 median, hourly sum (days)", gap["2014"]["median_hourly_sum"], 97, 0),
        ("2014 median, BE-sine (days)", gap["2014"]["median_be_sine"], 86, 0),
        ("2005 national mean, min-max average (days)", mean05["midpoint"], 63.2, 1),
        ("2005 national mean, hourly average (days)", mean05["daily_mean"], 66.2, 1),
        ("2005 national mean, BE-sine (days)", mean05["be_sine"], 68.2, 1),
        ("2005 national mean, hourly sum (days)", mean05["hourly_sum"], 71.1, 1),
        ("Hourly sum, cells without a season 2005 (%)", t7pct(2005, [0]), 15.4, 1),
        ("Hourly sum, cells without a season 2024 (%)", t7pct(2024, [0]), 1.4, 1),
        ("Hourly sum, classes 0-1 in 2005 (%)", t7pct(2005, [0, 1]), 48.5, 1),
        ("Hourly sum, class 4 in 2005 (%)", t7pct(2005, [4]), 8.1, 1),
        ("Hourly sum, class 5 in 2005 (%)", t7pct(2005, [5]), 0.4, 1),
        ("Hourly sum, classes 3-5 in 2024 (%)", t7pct(2024, [3, 4, 5]), 62.1, 1),
    ]


# ------------------------------------------------------------------ Part B
TABLES = [
    # pack file, reference file, keys, value columns, reference filter
    ("T1_trend_MK_Sen_all_methods", "T1_trend_MK_Sen_all_methods", ["variable", "method", "series"],
     ["mk_S", "mk_p", "mk_z", "mk_tau", "sen_slope", "sen_lo95", "sen_hi95", "sen_intercept", "acf1"], None),
    ("T2_distribution_diagnostics_summary", "T2_distribution_diagnostics_summary", ["variable", "method"],
     ["n_years", "years_shapiro_p_lt_0.001", "years_shapiro_p_lt_0.05", "years_dip_p_lt_0.05", "years_dip_p_lt_0.001",
      "skewness_median", "skewness_min", "skewness_max", "dip_stat_median", "n_cells_median"], None),
    ("T3_variance_ratio_national", "T3_variance_ratio_national_and_percell", ["variable", "series"],
     ["var_method_noise", "var_year_signal", "variance_ratio", "percent_method_uncertainty"],
     lambda b: b[b["level"] == "national_series"]),
    ("T4_class_agreement_pooled", "T4_class_agreement_pooled", ["method"],
     ["n_cells", "pct_same_class", "pct_within_1_class", "pct_class_higher", "pct_class_lower", "kappa",
      "kappa_linear_w", "spearman_rho", "pearson_r", "median_delta_days", "p25_delta", "p75_delta",
      "mean_abs_delta_days"], None),
    ("T4b_class_agreement_by_year", "T4b_class_agreement_by_year", ["method", "year"],
     ["n_cells", "pct_same_class", "pct_within_1_class", "pct_class_higher", "pct_class_lower", "kappa",
      "kappa_linear_w", "spearman_rho", "pearson_r", "median_delta_days", "p25_delta", "p75_delta",
      "mean_abs_delta_days"], None),
    ("T5_national_summary_all_methods", "T5_national_summary_all_methods", ["variable", "method", "year"],
     ["median", "mean", "p25", "p75", "pct_with_season", "count"], None),
    ("T6b_pairwise_delta_by_year", "T6b_pairwise_delta_by_year", ["variable", "year", "method1", "method2"],
     ["median_delta", "p25", "p75", "mean", "sd", "count"], None),
    ("T7_suitability_class_counts_by_year", "T7_suitability_class_counts_by_year", ["year", "method", "class"],
     ["count", "pct", "total_valid", "nonzero_count", "nonzero_pct", "median_season_length"], None),
    ("T8_bias_by_reference_class", "T8_bias_by_reference_class", ["method", "reference_class"],
     ["n_cell_years", "mean_delta_days", "median_delta_days", "p25_delta", "p75_delta", "pct_same_class",
      "pct_class_lower", "pct_class_higher"], None),
    ("T9_median_gap_diagnostic", "T9_median_gap_diagnostic", ["method", "year"],
     ["n_cells", "median_method", "median_reference", "delta_medians", "mean_method", "mean_reference",
      "delta_means", "mean_paired_delta", "median_paired_delta", "pct_ref_cells_between_medians_strict",
      "pct_ref_cells_between_medians_inclusive", "pct_method_cells_between_medians_strict"], None),
    ("per_cell_classes_ALL_YEARS", "per_cell_classes_ALL_YEARS", ["year", "i", "j"],
     ["midpoint", "daily_mean", "hourly_sum", "be_sine", "consensus", "class_midpoint", "class_daily_mean",
      "class_hourly_sum", "class_be_sine", "class_consensus"], None),
]


def compare_table(pack: Path, ref: Path, spec, rtol: float = 1e-6):
    mine, theirs, keys, cols, filt = spec
    pa, pb = pack / f"{mine}.csv", ref / f"{theirs}.csv"
    if not pa.exists() or not pb.exists():
        return {"table": mine, "status": "skipped (file missing)", "rows": 0, "worst": ""}
    a, b = pd.read_csv(pa), pd.read_csv(pb)
    if filt is not None:
        b = filt(b)
    for k in keys:
        a[k], b[k] = a[k].astype(str), b[k].astype(str)
    m = a.merge(b, on=keys, how="inner", suffixes=("", "_ref"))
    worst, bad = 0.0, []
    for c in cols:
        if c not in a.columns or c not in b.columns:
            continue
        x = pd.to_numeric(m[c], errors="coerce").to_numpy(float)
        y = pd.to_numeric(m[c + "_ref"], errors="coerce").to_numpy(float)
        nan_mismatch = np.isnan(x) != np.isnan(y)
        both = ~np.isnan(x) & ~np.isnan(y)
        err = np.abs(x[both] - y[both]) / np.maximum(1.0, np.abs(y[both]))
        e = float(err.max()) if err.size else 0.0
        worst = max(worst, e)
        if e > rtol or nan_mismatch.any():
            bad.append(f"{c} (max rel. diff {e:.2e}, NaN mismatches {int(nan_mismatch.sum())})")
    status = "OK" if not bad else "DIFFERENT: " + "; ".join(bad)
    if len(m) == 0:
        status = "no common rows"
    return {"table": mine, "status": status, "rows": len(m), "worst": f"{worst:.1e}"}


def main(argv=None) -> int:
    root = Path(os.environ.get("DDU_ROOT", str(REPO)))
    ap = argparse.ArgumentParser(description="Compare a regenerated manuscript pack with the paper")
    ap.add_argument("--pack", default=str(root / "out" / "12_method_compare" / "win30" / "00_MANUSCRIPT_PACK"))
    ap.add_argument("--reference", default=str(REFERENCE_RUN), help="archived tables of the published run")
    ap.add_argument("--only-manuscript", action="store_true", help="skip the table-by-table comparison")
    a = ap.parse_args(argv)
    pack, ref = Path(a.pack), Path(a.reference)
    if not (pack / "key_numbers.json").exists():
        print(f"pack not found: {pack}  (run: python run_pipeline.py --from 9 --to 10)")
        return 2

    failures = 0
    print(f"Part A - numbers printed in the manuscript (pack: {pack})")
    for desc, value, expected, dec in manuscript_checks(pack):
        got = ds.round_half_up(float(value), dec)
        ok = abs(got - float(expected)) < 10 ** (-dec) / 2
        failures += 0 if ok else 1
        print(f"  [{'OK ' if ok else 'BAD'}] {desc}: manuscript {expected}, computed {got} ({float(value):.6g})")

    if not a.only_manuscript:
        print(f"\nPart B - tables compared with the published run ({ref})")
        for spec in TABLES:
            r = compare_table(pack, ref, spec)
            if r["status"].startswith("DIFFERENT") or r["status"] == "no common rows":
                failures += 1
            print(f"  {r['table']:40s} rows {r['rows']:>6}  {r['status']}")

    print("\nRESULT:", "all checks passed" if failures == 0 else f"{failures} check(s) failed")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
