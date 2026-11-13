#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step09_manuscript_statistics.py - every table and number quoted in the manuscript, from one file.

Input  : <ROOT>/out/08_season_swrap/win30/csv/season_ALL_YEARS_methods.csv   (step05)
Output : <ROOT>/out/12_method_compare/win30/00_MANUSCRIPT_PACK/
           T1_trend_MK_Sen_all_methods.csv        Mann-Kendall + Sen's slope (95 % CI), national median and mean series
           T2_distribution_diagnostics_summary.csv skewness, Shapiro-Wilk, Hartigan's dip (years with p < 0.05 / 0.001)
           T2b_distribution_diagnostics_by_year.csv
           T3_variance_ratio_national.csv         share of variance due to method choice (median and mean series)
           T4_class_agreement_pooled.csv          each method vs hourly sum: % same class, kappa, Spearman, ...
           T4b_class_agreement_by_year.csv
           T5_national_summary_all_methods.csv    national median, mean, P25, P75, % cells with season
           T6b_pairwise_delta_by_year.csv         paired per-cell differences (delta = M1 - M2)
           T7_suitability_class_counts_by_year.csv
           T8_bias_by_reference_class.csv
           T9_median_gap_diagnostic.csv           national medians vs means vs paired differences (bimodality)
           per_cell_classes_ALL_YEARS.csv         season length and class per cell-year and method
           class_breaks.json                      P25/P50/P75/P90 of hourly-sum season length (Table 2)
           key_numbers.json                       the individual numbers quoted in the text
           manuscript_tables/Table3.csv, Table4.csv, Table5.csv, TableS1.csv ... TableS7.csv
           DDU_MANUSCRIPT_TABLES.xlsx             all of the above in one workbook

Definitions follow section 2.4 of the manuscript; the statistical functions are in ddu_stats.py.
The consensus of a grid cell is the median across the four methods; for season start and end the
median is taken over the methods that yield a season in that cell.

Usage
-----
python scripts/step09_manuscript_statistics.py
python scripts/step09_manuscript_statistics.py --season-csv path/to/season_ALL_YEARS_methods.csv --out path/to/folder
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# ------------------------- repo bootstrap -------------------------
_THIS = Path(__file__).resolve()
_REPO = _THIS.parent.parent if _THIS.parent.name.lower() == "scripts" else _THIS.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import ddu_stats as S  # noqa: E402

try:
    os.environ.setdefault("DDU_ROOT", str(_REPO))
    from config import cfg  # noqa: E402
except Exception as _e:  # config needs the boundary shapefile; statistics do not
    cfg = None
    warnings.warn(f"config.py not loaded ({_e}); using built-in defaults")

METHODS: List[str] = list(getattr(cfg, "METHODS", ["midpoint", "daily_mean", "hourly_sum", "be_sine"]))
LABELS: Dict[str, str] = dict(getattr(cfg, "METHOD_LABELS", {
    "midpoint": "Min–max average", "daily_mean": "Hourly average", "hourly_sum": "Hourly sum",
    "be_sine": "BE-sine", "consensus": "Consensus"}))
REF: str = str(getattr(cfg, "REF_METHOD", "hourly_sum"))
PAIRS = [tuple(p) for p in getattr(cfg, "PAIRS", [
    ("be_sine", "hourly_sum"), ("daily_mean", "be_sine"), ("daily_mean", "hourly_sum"),
    ("midpoint", "be_sine"), ("midpoint", "daily_mean"), ("midpoint", "hourly_sum")])]
REF_YEARS: List[int] = list(getattr(cfg, "REFERENCE_YEARS", [2005, 2011, 2017, 2024]))
PCTS = list(getattr(cfg, "CLASS_PERCENTILES", [25, 50, 75, 90]))
CLASS_LABELS: List[str] = list(getattr(cfg, "CLASS_LABELS", S.CLASS_LABELS))
ALL = ["consensus"] + METHODS
TABLE_ORDER = ["midpoint", "daily_mean", "hourly_sum", "be_sine"]   # order of methods in Table 3 / S6

# variable -> column name in season_ALL_YEARS_methods.csv (first match wins)
VAR_COLUMNS = {
    "start_doy": ["season_start_doy", "start_doy", "main_start_doy"],
    "end_doy": ["season_end_doy", "end_doy", "main_end_doy"],
    "season_length": ["season_length", "season_length_days"],
    "total_eip_days": ["total_eip_days", "eip_total_days"],
}
DATE_VARS = ("start_doy", "end_doy")          # undefined (NaN) when a cell has no season
COUNT_VARS = ("season_length", "total_eip_days")   # 0 when a cell has no season


# ------------------------- data -------------------------
def load_wide(season_csv: Path, variables: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
    """Per variable: one row per (year, i, j, lat, lon), one column per method + 'consensus'."""
    head = pd.read_csv(season_csv, nrows=0).columns
    cols = {}
    for v, cands in VAR_COLUMNS.items():
        if variables and v not in variables:
            continue
        c = next((c for c in cands if c in head), None)
        if c is not None:
            cols[v] = c
    keys = ["year", "i", "j", "lat", "lon", "method"]
    df = pd.read_csv(season_csv, usecols=keys + list(cols.values()), low_memory=False)
    df = df[df["method"].isin(METHODS)].copy()
    if "window_days" in head:
        pass
    wide = {}
    for v, c in cols.items():
        w = df.set_index(["year", "i", "j", "lat", "lon", "method"])[c].unstack("method")
        w = w.reindex(columns=METHODS)
        if w.isna().all().all():
            warnings.warn(f"{v}: column {c} is empty - skipped")
            continue
        if v in COUNT_VARS:
            w = w.fillna(0.0)
        w["consensus"] = w[METHODS].median(axis=1, skipna=True)   # date variables: over methods with a season
        wide[v] = w.reset_index().sort_values(["year", "i", "j"]).reset_index(drop=True)
    if not wide:
        raise SystemExit(f"no usable variables in {season_csv}")
    return wide


# ------------------------- tables -------------------------
def national_summary(wide: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for v, w in wide.items():
        for m in ALL:
            for y, g in w.groupby("year"):
                x = g[m].to_numpy(dtype=float)
                x = x[np.isfinite(x)]
                p25, p50, p75 = S.quantiles(x)
                rows.append({"variable": v, "method": m, "year": int(y), "median": p50,
                             "mean": float(x.mean()) if x.size else np.nan, "p25": p25, "p75": p75,
                             "pct_with_season": 100.0 * float(np.mean(x > 0)) if x.size else np.nan,
                             "count": int(x.size)})
    return pd.DataFrame(rows)


def trend_table(t5: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for v in t5["variable"].unique():
        for m in ALL:
            d = t5[(t5["variable"] == v) & (t5["method"] == m)].sort_values("year")
            for series in ("median", "mean"):
                if int(np.isfinite(d[series].to_numpy(dtype=float)).sum()) < 3:
                    warnings.warn(f"T1: {v}/{m}/{series} has fewer than 3 years - trend skipped")
                    continue
                r = S.trend_row(d["year"].to_numpy(), d[series].to_numpy())
                rows.append({"variable": v, "method": m, "series": series, **r})
    return pd.DataFrame(rows)


def variance_table(t5: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for v in t5["variable"].unique():
        for series in ("median", "mean"):
            t = t5[(t5["variable"] == v) & (t5["method"].isin(METHODS))].pivot(index="year", columns="method",
                                                                                 values=series)
            rows.append({"variable": v, "level": "national_series", "series": series, **S.variance_partition(t)})
    return pd.DataFrame(rows)


def distribution_tables(wide: Dict[str, pd.DataFrame]):
    by_year = []
    for v, w in wide.items():
        for m in ALL:
            for y, g in w.groupby("year"):
                by_year.append({"variable": v, "method": m, "year": int(y), **S.distribution_diagnostics(g[m])})
    by = pd.DataFrame(by_year)
    summ = []
    for (v, m), g in by.groupby(["variable", "method"], sort=False):
        summ.append({"variable": v, "method": m, "n_years": int(len(g)),
                     "years_shapiro_p_lt_0.001": int((g["shapiro_p"] < 0.001).sum()),
                     "years_shapiro_p_lt_0.05": int((g["shapiro_p"] < 0.05).sum()),
                     "years_dip_p_lt_0.05": int((g["dip_p"] < 0.05).sum()),
                     "years_dip_p_lt_0.001": int((g["dip_p"] < 0.001).sum()),
                     "skewness_median": float(g["skewness"].median()), "skewness_min": float(g["skewness"].min()),
                     "skewness_max": float(g["skewness"].max()), "dip_stat_median": float(g["dip_stat"].median()),
                     "n_cells_median": float(g["n"].median())})
    return pd.DataFrame(summ), by


def pairwise_by_year(wide: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for v, w in wide.items():
        for y, g in w.groupby("year"):
            for m1, m2 in PAIRS:
                d = (g[m1] - g[m2]).to_numpy(dtype=float)
                d = d[np.isfinite(d)]          # date variables: only cells with a season under both methods
                p25, p50, p75 = S.quantiles(d)
                rows.append({"variable": v, "year": int(y), "method1": m1, "method2": m2, "median_delta": p50,
                             "p25": p25, "p75": p75, "mean": float(d.mean()) if d.size else np.nan,
                             "sd": float(d.std(ddof=1)) if d.size > 1 else np.nan, "count": int(d.size)})
    return pd.DataFrame(rows)


def class_outputs(length: pd.DataFrame):
    breaks = S.class_breaks(length[REF].to_numpy())
    pc = length[["year", "i", "j", "lat", "lon"] + METHODS + ["consensus"]].copy()
    for m in METHODS + ["consensus"]:
        pc[f"class_{m}"] = S.classify(pc[m].to_numpy(), breaks)
    counts = []
    for y, g in pc.groupby("year"):
        for m in ALL:
            c = g[f"class_{m}"]
            n = int(c.notna().sum())
            nz = int((g[m] > 0).sum())
            for k in range(6):
                cnt = int((c == k).sum())
                counts.append({"year": int(y), "method": m, "class": k, "label": CLASS_LABELS[k], "count": cnt,
                               "pct": round(100.0 * cnt / n, 2) if n else np.nan, "total_valid": n,
                               "nonzero_count": nz, "nonzero_pct": round(100.0 * nz / n, 2) if n else np.nan,
                               "median_season_length": float(g[m].median()), "window": 30})
    ref = pc[REF].to_numpy(dtype=float)
    n_season = int(np.sum(ref > 0))
    meta = {"reference_method": REF, "percentiles": PCTS,
            "class_breaks_days": [float(b) for b in breaks], "class_ranges_days": S.class_ranges(breaks),
            "class_labels": CLASS_LABELS,
            "rule": "0 = no season; 1: < P25; 2: [P25,P50); 3: [P50,P75); 4: [P75,P90); 5: >= P90 "
                    "(percentiles of season_length > 0 of the reference method, all years and grid cells)",
            "n_cell_years_total": int(ref.size), "n_cell_years_with_season": n_season,
            "pct_cell_years_no_season": 100.0 * (1 - n_season / ref.size)}
    return breaks, pc, pd.DataFrame(counts), meta


def agreement_tables(length: pd.DataFrame, breaks) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    comp = [m for m in METHODS if m != REF] + ["consensus"]
    pooled, by_year, bias = [], [], []
    for m in comp:
        pooled.append({"variable": "season_length", "method": m, "reference": REF, "year": "ALL",
                       **S.agreement_row(length[m].to_numpy(), length[REF].to_numpy(), breaks)})
        for y, g in length.groupby("year"):
            by_year.append({"variable": "season_length", "method": m, "reference": REF, "year": int(y),
                            **S.agreement_row(g[m].to_numpy(), g[REF].to_numpy(), breaks)})
        b = S.bias_by_reference_class(length[m].to_numpy(), length[REF].to_numpy(), breaks)
        b.insert(0, "reference", REF)
        b.insert(0, "method", m)
        bias.append(b)
    return pd.DataFrame(pooled), pd.DataFrame(by_year), pd.concat(bias, ignore_index=True)


def median_gap_table(length: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for m in [mm for mm in METHODS if mm != REF] + ["consensus"]:
        for y, g in length.groupby("year"):
            a = g[m].to_numpy(dtype=float)
            r = g[REF].to_numpy(dtype=float)
            ma, mr = float(np.median(a)), float(np.median(r))
            lo, hi = min(ma, mr), max(ma, mr)
            rows.append({"variable": "season_length", "year": int(y), "method": m, "reference": REF,
                         "n_cells": int(a.size), "median_method": ma, "median_reference": mr,
                         "delta_medians": ma - mr, "mean_method": float(a.mean()), "mean_reference": float(r.mean()),
                         "delta_means": float(a.mean() - r.mean()), "mean_paired_delta": float(np.mean(a - r)),
                         "median_paired_delta": float(np.median(a - r)),
                         "pct_ref_cells_between_medians_strict": 100.0 * float(np.mean((r > lo) & (r < hi))),
                         "pct_ref_cells_between_medians_inclusive": 100.0 * float(np.mean((r >= lo) & (r <= hi))),
                         "pct_method_cells_between_medians_strict": 100.0 * float(np.mean((a > lo) & (a < hi)))})
    return pd.DataFrame(rows)


# ------------------------- manuscript tables (formatted) -------------------------
def _t5(t5, v, m, y):
    r = t5[(t5["variable"] == v) & (t5["method"] == m) & (t5["year"] == y)]
    return r.iloc[0] if len(r) else None


def table3(t5: pd.DataFrame, years: List[int]) -> pd.DataFrame:
    rows = []
    for y in years:
        for m in TABLE_ORDER:
            ln, st, en = _t5(t5, "season_length", m, y), _t5(t5, "start_doy", m, y), _t5(t5, "end_doy", m, y)
            rows.append({"Year": y, "Method": LABELS[m],
                         "Grid cells with season (%)": S.fmt_fixed(ln["pct_with_season"], 1),
                         "Season start, P50 DOY (date)": S.fmt_doy(st["median"]) if st is not None else "",
                         "Season end, P50 DOY (date)": S.fmt_doy(en["median"]) if en is not None else "",
                         "Season length, P50 (days)": S.fmt_num(ln["median"]),
                         "Season length, mean (days)": S.fmt_fixed(ln["mean"], 1)})
    return pd.DataFrame(rows)


def table_pairs(t6b: pd.DataFrame, variable: str, years: List[int], value_col: str, n_col: Optional[str]):
    rows = []
    d = t6b[t6b["variable"] == variable]
    for y in years:
        for m1, m2 in PAIRS:
            r = d[(d["year"] == y) & (d["method1"] == m1) & (d["method2"] == m2)]
            if not len(r):
                continue
            r = r.iloc[0]
            row = {"Year": y, "M1": LABELS[m1], "M2": LABELS[m2],
                   value_col: S.fmt_p50_iqr(r["median_delta"], r["p25"], r["p75"])}
            if n_col:
                row[n_col] = int(r["count"])
            rows.append(row)
    return pd.DataFrame(rows)


def table5(t7: pd.DataFrame, meta: dict, years: List[int]) -> pd.DataFrame:
    rng = meta["class_ranges_days"]
    heads = ["Class 0 (no season)"] + [f"Class {k} ({rng[k].replace('-', S.EN_DASH).replace('>=', chr(8805))} d)"
                                       for k in range(1, 6)]
    rows = []
    for y in years:
        d = t7[(t7["year"] == y) & (t7["method"] == REF)].set_index("class")
        row = {"Year": y}
        for k in range(6):
            cnt, tot = int(d.loc[k, "count"]), int(d.loc[k, "total_valid"])
            row[heads[k]] = f"{cnt} ({S.fmt_fixed(100.0 * cnt / tot, 1)}%)"
        rows.append(row)
    return pd.DataFrame(rows)


def table_consensus_dates(t5: pd.DataFrame, variable: str, years: List[int], what: str) -> pd.DataFrame:
    rows = []
    for y in years:
        r = _t5(t5, variable, "consensus", y)
        rows.append({"Year": y, f"{what}, P50 DOY (date)": S.fmt_doy(r["median"]),
                     "P25 DOY (date)": S.fmt_doy(r["p25"]), "P75 DOY (date)": S.fmt_doy(r["p75"]),
                     f"Cells with a consensus {what.lower()} (n)": int(r["count"])})
    return pd.DataFrame(rows)


def table_s5(t5: pd.DataFrame, years: List[int]) -> pd.DataFrame:
    rows = []
    for y in years:
        ln, ep = _t5(t5, "season_length", "consensus", y), _t5(t5, "total_eip_days", "consensus", y)
        rows.append({"Year": y,
                     "Season length, P50 (P25, P75)": S.fmt_p50_iqr(ln["median"], ln["p25"], ln["p75"]),
                     "Season length, mean": S.fmt_fixed(ln["mean"], 1),
                     "Total EIP days, P50 (P25, P75)": S.fmt_p50_iqr(ep["median"], ep["p25"], ep["p75"]) if ep is not None else "",
                     "Total EIP days, mean": S.fmt_fixed(ep["mean"], 1) if ep is not None else "",
                     "Cells without a consensus season (%)": S.fmt_fixed(100.0 - ln["pct_with_season"], 1)})
    return pd.DataFrame(rows)


def table_s6(t5: pd.DataFrame, years: List[int]) -> pd.DataFrame:
    rows = []
    for y in years:
        for m in TABLE_ORDER:
            ln, st, en, ep = (_t5(t5, v, m, y) for v in ("season_length", "start_doy", "end_doy", "total_eip_days"))
            rows.append({"Year": y, "Method": LABELS[m],
                         "Grid cells with season (%)": S.fmt_fixed(ln["pct_with_season"], 1),
                         "Start, P50 DOY (date)": S.fmt_doy(st["median"]) if st is not None else "",
                         "End, P50 DOY (date)": S.fmt_doy(en["median"]) if en is not None else "",
                         "Length, P50 (days)": S.fmt_num(ln["median"]),
                         "Length, mean (days)": S.fmt_fixed(ln["mean"], 1),
                         "Total EIP days, P50": S.fmt_num(ep["median"]) if ep is not None else ""})
    return pd.DataFrame(rows)


# ------------------------- key numbers -------------------------
def key_numbers(t1, t3, t4, t8, t9, t6b, t5, meta, length) -> dict:
    def g(df, **kw):
        q = df
        for k, v in kw.items():
            q = q[q[k] == v]
        return q.iloc[0]
    k = {"class_breaks": meta}
    k["class_agreement_vs_hourly_sum"] = {
        m: {"pct_same_class": float(g(t4, method=m)["pct_same_class"]),
            "pct_class_lower": float(g(t4, method=m)["pct_class_lower"]),
            "pct_class_higher": float(g(t4, method=m)["pct_class_higher"]),
            "spearman_rho": float(g(t4, method=m)["spearman_rho"])} for m in t4["method"]}
    k["no_season_under_method_but_season_under_hourly_sum"] = {
        m: int(((length[m] == 0) & (length[REF] > 0)).sum()) for m in METHODS if m != REF}
    k["n_cell_years_with_season_hourly_sum"] = int((length[REF] > 0).sum())
    k["variance_share_method_pct"] = {f"{r.variable}|{r.series}": float(r.percent_method_uncertainty)
                                      for r in t3.itertuples()}
    k["trend"] = {f"{r.variable}|{r.method}|{r.series}": {"tau": r.mk_tau, "p": r.mk_p, "sen": r.sen_slope,
                                                         "sen_lo95": r.sen_lo95, "sen_hi95": r.sen_hi95}
                  for r in t1.itertuples()}
    k["bias_by_class_midpoint"] = {str(r.reference_class): {"median": r.median_delta_days, "p25": r.p25_delta,
                                                            "p75": r.p75_delta, "mean": r.mean_delta_days}
                                   for r in t8[t8["method"] == "midpoint"].itertuples()}
    gap = t9[t9["method"] == "be_sine"].set_index("year")
    k["median_gap_be_sine_vs_hourly_sum"] = {
        str(y): {"median_be_sine": float(gap.loc[y, "median_method"]), "median_hourly_sum": float(gap.loc[y, "median_reference"]),
                 "delta_means": float(gap.loc[y, "delta_means"]), "median_paired_delta": float(gap.loc[y, "median_paired_delta"]),
                 "pct_cells_between_medians": float(gap.loc[y, "pct_ref_cells_between_medians_strict"])}
        for y in gap.index}
    dm = -gap["delta_means"]
    k["mean_difference_hourly_sum_minus_be_sine_range"] = [float(dm.min()), float(dm.max())]
    ln = t5[(t5["variable"] == "season_length") & (t5["year"] == min(t5["year"]))]
    k["national_mean_length_first_year"] = {r.method: float(r.mean) for r in ln.itertuples()}
    return k


# ------------------------- main -------------------------
def main():
    root = Path(getattr(cfg, "ROOT", _REPO)) if cfg is not None else _REPO
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--season-csv", default=str(root / "out" / "08_season_swrap" / "win30" / "csv" /
                                                 "season_ALL_YEARS_methods.csv"))
    ap.add_argument("--out", default=str(root / "out" / "12_method_compare" / "win30" /
                                         str(getattr(cfg, "PACK_DIRNAME", "00_MANUSCRIPT_PACK"))))
    ap.add_argument("--variables", default="start_doy,end_doy,season_length,total_eip_days")
    ap.add_argument("--no-excel", action="store_true")
    a = ap.parse_args()

    season_csv, out = Path(a.season_csv), Path(a.out)
    if not season_csv.exists():
        raise SystemExit(f"missing input: {season_csv} (run step05 first)")
    out.mkdir(parents=True, exist_ok=True)
    (out / "manuscript_tables").mkdir(exist_ok=True)
    print(f"[09] input : {season_csv}\n[09] output: {out}")

    wide = load_wide(season_csv, [v for v in a.variables.split(",") if v])
    if "season_length" not in wide:
        raise SystemExit("season_length is required")
    length = wide["season_length"]
    years = sorted(length["year"].unique().tolist())
    ref_years = [y for y in REF_YEARS if y in years]

    t5 = national_summary(wide)
    t1 = trend_table(t5)
    t3 = variance_table(t5)
    t2, t2b = distribution_tables(wide)
    t6b = pairwise_by_year(wide)
    breaks, pc, t7, meta = class_outputs(length)
    t4, t4b, t8 = agreement_tables(length, breaks)
    t9 = median_gap_table(length)

    files = {"T1_trend_MK_Sen_all_methods": t1, "T2_distribution_diagnostics_summary": t2,
             "T2b_distribution_diagnostics_by_year": t2b, "T3_variance_ratio_national": t3,
             "T4_class_agreement_pooled": t4, "T4b_class_agreement_by_year": t4b,
             "T5_national_summary_all_methods": t5, "T6b_pairwise_delta_by_year": t6b,
             "T7_suitability_class_counts_by_year": t7, "T8_bias_by_reference_class": t8,
             "T9_median_gap_diagnostic": t9, "per_cell_classes_ALL_YEARS": pc}
    for name, df in files.items():
        df.to_csv(out / f"{name}.csv", index=False)
    (out / "class_breaks.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    tabs = {"Table3": table3(t5, ref_years),
            "Table4": table_pairs(t6b, "season_length", ref_years, "Δ season length, P50 (P25, P75), days", None),
            "Table5": table5(t7, meta, ref_years)}
    if "start_doy" in wide:
        tabs["TableS1"] = table_consensus_dates(t5, "start_doy", years, "Start")
        tabs["TableS2"] = table_pairs(t6b, "start_doy", years, "Δ start, P50 (P25, P75)", "n pairs")
    if "end_doy" in wide:
        tabs["TableS3"] = table_consensus_dates(t5, "end_doy", years, "End")
        tabs["TableS4"] = table_pairs(t6b, "end_doy", years, "Δ end, P50 (P25, P75)", "n pairs")
    tabs["TableS5"] = table_s5(t5, years)
    tabs["TableS6"] = table_s6(t5, years)
    tabs["TableS7"] = table_pairs(t6b, "season_length", years, "Δ length, P50 (P25, P75)", "n cells")
    for name, df in tabs.items():
        df.to_csv(out / "manuscript_tables" / f"{name}.csv", index=False, encoding="utf-8-sig")

    kn = key_numbers(t1, t3, t4, t8, t9, t6b, t5, meta, length)
    kn["settings"] = {"season_csv": str(season_csv), "methods": METHODS, "reference_method": REF,
                      "pairs_delta_m1_minus_m2": [list(p) for p in PAIRS], "reference_years": ref_years}
    (out / "key_numbers.json").write_text(json.dumps(kn, indent=2, ensure_ascii=False, default=float),
                                          encoding="utf-8")

    if not a.no_excel:
        xl = out / "DDU_MANUSCRIPT_TABLES.xlsx"
        with pd.ExcelWriter(xl, engine="openpyxl") as w:
            for name, df in tabs.items():
                df.to_excel(w, sheet_name=name, index=False)
            for name, df in files.items():
                if name != "per_cell_classes_ALL_YEARS":
                    df.to_excel(w, sheet_name=name.split("_")[0], index=False)
        print(f"[09] workbook: {xl}")
    print(f"[09] class breaks (days): {meta['class_ranges_days']}")
    print(f"[09] done: {len(files)} tables, {len(tabs)} manuscript tables, key_numbers.json")


if __name__ == "__main__":
    main()
