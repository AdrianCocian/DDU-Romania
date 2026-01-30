#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
step06_DDU_METHOD_COMPARE__ALL.py  (GitHub/publication-ready)

CLEAN ALL-IN-ONE: Compare + Graphs + Maps + Excel (+ Pixel Summary) (+ Combined export)

Runs AFTER:
  step05_ddu_season_wrap.py -> produces:
    <ROOT>/out/08_season_swrap/win{W}/csv/season_ALL_YEARS_methods.csv

Outputs per window W:
  <ROOT>/out/12_method_compare/win{W}/6.compare/

CLEAN output structure:
  00_meta/
    META_{W}D.json
  01_tables/
    national_summary_<var>.csv
    national_summary_<var>_method_<method>.csv
    delta_summary_<var>.csv
    article_pairwise_summary_<var>.csv
    article_pairwise_thresholds_<var>.csv
    pixel_series_<var>.csv                      (optional, --pixel)
    per_year/                                   (needed for maps/plots; can prune)
      {Y}_{var}_MEDIAN.csv
      {Y}_{var}_PAIRWISE.csv
    agg/
      {var}_PAIRWISE_MEDIAN_ALL_YEARS.csv
      {var}_MEDIAN_CONSENSUS_ALL_YEARS.csv
    pixel_summary/                              (optional, --do-pixel-summary)
      {var}_PIXEL_SUMMARY_ALL_YEARS.csv
  02_plots/
    timeseries_<var>.png
    timeseries_permethod_<var>.png
    delta_box_<var>_<pair>.png
    scatter_<var>_<pair>_ALLYEARS.png
    bland_altman_<var>_<pair>_ALLYEARS.png
    hist_allyears_<var>_*.png
    pixel_series_<var>.png                      (optional, --pixel)
  03_maps/
    values/
      map_{Y}_{var}.png                         (CONSENSUS)
    deltas/
      map_{Y}_{var}_DELTA_<pair>.png
      map_{Y}_{var}_DELTAFLAG_<K>_<pair>.png    (optional flag maps)
    permethod/                                  (optional, --permethod-maps)
      <method>/map_{Y}_{var}_{method}.png
  99_excel/
    win{W}_summary.xlsx                         (compact; default ON)
    win{W}_REPORT.xlsx                          (optional, --excel-full)

Optional combined export across processed windows:
  <ROOT>/out/12_method_compare/00_combined/
    ALL_WINDOWS_6_RESULTS.xlsx
    csv/*.csv

Map rules:
  - Values colormap: 'bwr' (blue=low, red=high)
  - EXCEPTION: start_doy uses 'bwr_r' (red=earlier)
  - NaN/NoData: pale yellow #FFF5B1
  - Square pixels: pcolormesh(..., shading="nearest"); aspect='equal'

Δ convention:
  - delta = method1 − method2 in canonical pair order

Canonical pairs:
  midpoint-daily_mean, midpoint-be_sine, midpoint-hourly_sum,
  daily_mean-be_sine, daily_mean-hourly_sum, be_sine-hourly_sum

Repo portability (GitHub-ready)
-------------------------------
- Auto-detect repo ROOT by walking upward until finding config.py or .git.
- Sets DDU_ROOT automatically and fixes sys.path so `from config import cfg` works
  regardless of current working directory.
- No hardcoded Windows paths.

"""

from __future__ import annotations

import os
os.environ.setdefault("MPLBACKEND", "Agg")

import sys
import argparse
import json
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
plt.ioff()

import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1 import make_axes_locatable


# ---------------------------------------------------------------------
# Repo bootstrap (GitHub-safe)
# ---------------------------------------------------------------------
def _preparse_root(argv: List[str]) -> Optional[str]:
    for i, a in enumerate(argv):
        if a == "--root" and i + 1 < len(argv):
            return argv[i + 1]
    for a in argv:
        if a.startswith("--root="):
            return a.split("=", 1)[1]
    return None


def _find_repo_root(start: Path) -> Path:
    cur = start.resolve()
    for _ in range(12):
        if (cur / "config.py").exists() or (cur / ".git").exists():
            return cur
        cur = cur.parent
    return start.resolve()


_root_cli = _preparse_root(sys.argv)
if _root_cli:
    REPO_ROOT = Path(_root_cli).expanduser().resolve()
else:
    this = Path(__file__).resolve()
    guess = this.parent.parent if this.parent.name.lower() == "scripts" else this.parent
    REPO_ROOT = _find_repo_root(guess)

os.environ.setdefault("DDU_ROOT", str(REPO_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------
# Optional config integration
# ---------------------------------------------------------------------
try:
    from config import cfg  # type: ignore[attr-defined]
except Exception:
    cfg = None  # type: ignore


def _cfg_path(attr: str, default: Path) -> Path:
    if cfg is None:
        return default
    v = getattr(cfg, attr, None)
    return Path(v) if v else default


def _cfg_str(attr: str, default: str) -> str:
    if cfg is None:
        return default
    v = getattr(cfg, attr, None)
    return str(v) if v else default


# Prefer config ROOT if present; otherwise REPO_ROOT.
_CFG_ROOT = _cfg_path("ROOT", REPO_ROOT).resolve()

# Defaults
DEFAULT_OUTPUT_ROOT = _cfg_str("DIR_METHOD_COMPARE", str(_CFG_ROOT / "out" / "12_method_compare"))
DEFAULT_OUTLINE_SHP = _cfg_str("SHP_RO", "") or None

DEFAULT_METHODS_CSV_TEMPLATE = str(
    _CFG_ROOT / "out" / "08_season_swrap" / "win{win}" / "csv" / "season_ALL_YEARS_methods.csv"
)

# Keep for “if user did not pass output_root and cfg missing”
_DEFAULT_OUTPUT_ROOT_ORIG = DEFAULT_OUTPUT_ROOT


# ---------------------------------------------------------------------
# Defaults / Aliases
# ---------------------------------------------------------------------
VAR_ALIASES = {
    "start_doy":      ["start_doy", "season_start_doy", "main_start_doy"],
    "end_doy":        ["end_doy", "season_end_doy", "main_end_doy"],
    "season_length":  ["season_length", "season_length_days", "length", "main_length"],
    "total_eip_days": ["total_eip_days", "eip_total_days", "eip_days"],
}

DISPLAY = {
    "start_doy": "Start DOY",
    "end_doy": "End DOY",
    "season_length": "Season length (days)",
    "total_eip_days": "Total EIP days",
}

DEFAULT_METHODS = ["midpoint", "daily_mean", "hourly_sum", "be_sine"]

DEFAULT_PAIRS = [
    ("midpoint", "daily_mean"),
    ("midpoint", "be_sine"),
    ("midpoint", "hourly_sum"),
    ("daily_mean", "be_sine"),
    ("daily_mean", "hourly_sum"),
    ("be_sine", "hourly_sum"),
]

NODATA_HEX = "#FFF5B1"


# ---------------------------------------------------------------------
# Output structure (CLEAN)
# ---------------------------------------------------------------------
def _ensure_out_structure(out_root: Path) -> Dict[str, Path]:
    d00 = out_root / "00_meta"
    d01 = out_root / "01_tables"
    d01_py = d01 / "per_year"
    d01_agg = d01 / "agg"
    d01_pix = d01 / "pixel_summary"
    d02 = out_root / "02_plots"
    d03 = out_root / "03_maps"
    d03_val = d03 / "values"
    d03_del = d03 / "deltas"
    d03_pm = d03 / "permethod"
    d99 = out_root / "99_excel"

    for d in [d00, d01, d01_py, d01_agg, d01_pix, d02, d03, d03_val, d03_del, d03_pm, d99]:
        d.mkdir(parents=True, exist_ok=True)

    return {
        "meta": d00,
        "tables": d01,
        "per_year": d01_py,
        "agg": d01_agg,
        "pixel": d01_pix,
        "plots": d02,
        "maps": d03,
        "maps_values": d03_val,
        "maps_deltas": d03_del,
        "maps_permethod": d03_pm,
        "excel": d99,
    }


# ---------------------------------------------------------------------
# Style helper
# ---------------------------------------------------------------------
def _style_ctx(style: Optional[str]):
    try:
        return plt.style.context(style) if style else plt.style.context("default")
    except Exception:
        return plt.style.context("default")


# ---------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------
def _auto_find_methods_csv(window: int, methods_csv_cli: Optional[str]) -> Optional[Path]:
    # 1) CLI explicit
    if methods_csv_cli:
        try:
            p = Path(methods_csv_cli.format(win=window)) if "{win}" in methods_csv_cli else Path(methods_csv_cli)
            if p.exists():
                return p
        except Exception:
            pass

    # 2) default template
    try:
        pdef = Path(DEFAULT_METHODS_CSV_TEMPLATE.format(win=window))
        if pdef.exists():
            return pdef
    except Exception:
        pass

    # 3) common fallbacks
    script_dir = Path(__file__).resolve().parent
    cands = [
        _CFG_ROOT / "out" / "08_season_swrap" / f"win{window}" / "csv" / "season_ALL_YEARS_methods.csv",
        _CFG_ROOT / "out" / "11_ddu_season_swrap" / f"win{window}" / "csv" / "season_ALL_YEARS_methods.csv",
        Path.cwd() / "out" / "08_season_swrap" / f"win{window}" / "csv" / "season_ALL_YEARS_methods.csv",
        script_dir / ".." / "out" / "08_season_swrap" / f"win{window}" / "csv" / "season_ALL_YEARS_methods.csv",
    ]
    for c in cands:
        c = c.resolve()
        if c.exists():
            return c

    # 4) last resort: rglob
    try:
        for root in [_CFG_ROOT, REPO_ROOT, Path.cwd(), script_dir]:
            root = root.resolve()
            for p in root.rglob("season_ALL_YEARS_methods.csv"):
                if f"win{window}" in str(p).replace("\\", "/"):
                    return p
    except Exception:
        pass

    return None


def _sibling_meta(methods_csv: Path) -> Optional[dict]:
    try:
        return json.loads((methods_csv.parent / "META.json").read_text(encoding="utf-8"))
    except Exception:
        return None


# ---------------------------------------------------------------------
# Parse helpers
# ---------------------------------------------------------------------
def parse_years(s: Optional[str]) -> Optional[List[int]]:
    if not s:
        return None
    s = s.replace("–", "-").replace("—", "-").strip()
    if "-" in s:
        a, b = s.split("-", 1)
        a, b = int(a), int(b)
        return list(range(min(a, b), max(a, b) + 1))
    return [int(x) for x in s.split(",") if x.strip()]


def parse_bbox(s: Optional[str]) -> Optional[Tuple[float, float, float, float]]:
    if not s:
        return None
    vals = [float(x) for x in s.split(",")]
    if len(vals) != 4:
        raise ValueError("--bbox must be N,W,S,E")
    N, W, S, E = vals
    if not (E >= W and N >= S):
        raise ValueError("BBOX order N,W,S,E violated")
    return N, W, S, E


def parse_pair(s: Optional[str]) -> Optional[Tuple[str, str]]:
    if not s:
        return None
    p = [x.strip() for x in s.split(",")]
    if len(p) != 2:
        raise ValueError("--pair method1,method2")
    return p[0], p[1]


# ---------------------------------------------------------------------
# Axis helpers
# ---------------------------------------------------------------------
def _set_year_ticks(ax, years_list) -> None:
    years = []
    for y in years_list:
        try:
            yf = float(y)
            if np.isfinite(yf):
                years.append(int(yf))
        except Exception:
            continue
    if not years:
        return
    y0, y1 = min(years), max(years)
    ticks = list(range(y0, y1 + 1))
    ax.xaxis.set_major_locator(mticker.FixedLocator(ticks))
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%d"))
    ax.set_xlim(y0 - 0.5, y1 + 0.5)
    ax.tick_params(axis="x", labelsize=8)


# ---------------------------------------------------------------------
# Data normalization / filtering
# ---------------------------------------------------------------------
def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    for key, aliases in VAR_ALIASES.items():
        for c in aliases:
            if c in df.columns and c != key:
                df = df.rename(columns={c: key})
                break

    if "year" not in df.columns:
        for c in ("Year", "AN", "an"):
            if c in df.columns:
                df = df.rename(columns={c: "year"})
                break

    for a, b in (("latitude", "lat"), ("longitude", "lon")):
        if a in df.columns and b not in df.columns:
            df = df.rename(columns={a: b})
    return df


def _filter_roi(df: pd.DataFrame, bbox: Optional[Tuple[float, float, float, float]]) -> pd.DataFrame:
    if df.empty:
        return df

    # If an explicit inside/roi column exists (rare), respect it.
    for col in ("inside_mask", "inside", "roi", "in_ro"):
        if col in df.columns:
            mask = df[col].astype(str).str.lower().isin(["1", "true", "yes"])
            df = df[mask].copy()

    if bbox and {"lat", "lon"}.issubset(df.columns):
        N, W, S, E = bbox
        df = df[
            (df["lat"] <= N) & (df["lat"] >= S) &
            (df["lon"] >= W) & (df["lon"] <= E)
        ].copy()

    return df


def _stable_method_order(methods_found: List[str]) -> List[str]:
    want = {m: i for i, m in enumerate(DEFAULT_METHODS)}
    return sorted(methods_found, key=lambda m: want.get(m, 99))


# ---------------------------------------------------------------------
# PHASE A: Compare
# ---------------------------------------------------------------------
def compare_phase(
    in_file: Path,
    out_root: Path,
    years: Optional[List[int]],
    methods_req: List[str],
    variables_req: List[str],
    bbox: Optional[Tuple[float, float, float, float]],
    window: int,
    thresholds_meta: Optional[dict],
) -> None:
    """
    Produces per-year (01_tables/per_year):
      - {Y}_{var}_MEDIAN.csv
      - {Y}_{var}_PAIRWISE.csv

    Produces multi-year (01_tables/agg):
      - {var}_PAIRWISE_MEDIAN_ALL_YEARS.csv
      - {var}_MEDIAN_CONSENSUS_ALL_YEARS.csv

    Writes META in 00_meta/META_{window}D.json
    """
    dirs = _ensure_out_structure(out_root)
    per_year_dir = dirs["per_year"]
    agg_dir = dirs["agg"]
    meta_dir = dirs["meta"]

    df = pd.read_csv(in_file, low_memory=False)
    df = _normalize_columns(df)
    df = _filter_roi(df, bbox)

    if years is not None and "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
        df = df.dropna(subset=["year"]).copy()
        df["year"] = df["year"].astype(int)
        df = df[df["year"].isin(years)].copy()

    if "method" not in df.columns:
        raise ValueError("Input missing 'method' column (expected from season_ALL_YEARS_methods.csv).")

    methods_av = sorted(df["method"].dropna().astype(str).unique().tolist())
    methods = [m for m in methods_req if m in methods_av] if methods_req else methods_av
    if len(methods) < 2:
        raise ValueError(f"Need ≥2 methods. Found: {methods_av}")
    methods = _stable_method_order(methods)

    vars_eff = [v for v in variables_req if v in df.columns]
    if not vars_eff:
        raise ValueError(f"None of requested variables exist in file: {variables_req}")

    years_av = sorted(df["year"].dropna().astype(int).unique().tolist()) if "year" in df.columns else []

    meta = {
        "window_days": window,
        "input": str(in_file.resolve()),
        "years": years_av,
        "methods": methods,
        "variables": vars_eff,
        "bbox": bbox,
        "pairs_canonical_delta": "delta = method1 - method2",
        "canonical_pairs": [f"{a}-{b}" for a, b in DEFAULT_PAIRS],
        "created_utc": pd.Timestamp.utcnow().isoformat(timespec="seconds"),
    }
    if thresholds_meta:
        for k in ("tbase_c", "thresh_dd_window", "thresh_dh_window", "min_gap_days", "use_year_wrap"):
            if k in thresholds_meta:
                meta[k] = thresholds_meta[k]

    (meta_dir / f"META_{window}D.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    # Per-year outputs
    for y in years_av:
        dyy = df[df["year"].astype(int) == y].copy()
        if dyy.empty:
            continue

        idx_cols = [c for c in ["i", "j"] if c in dyy.columns]
        coords = ["lat", "lon"] if {"lat", "lon"}.issubset(dyy.columns) else []
        if not idx_cols and not coords:
            continue

        for v in vars_eff:
            need = idx_cols + coords + ["method", v]
            usecol = [c for c in need if c in dyy.columns]
            dsel = dyy[usecol].dropna(subset=[v]).copy()
            if dsel.empty:
                continue

            piv_index = idx_cols + coords
            piv = dsel.pivot_table(index=piv_index, columns="method", values=v, aggfunc="mean").reset_index()

            cols_m = [m for m in methods if m in piv.columns]
            if cols_m:
                med = piv[cols_m].median(axis=1, skipna=True)
                cnt = piv[cols_m].count(axis=1)

                med_df = piv[piv_index].copy()
                med_df.insert(0, "year", y)
                med_df["variable"] = v
                med_df["median_methods"] = med
                med_df["methods_present"] = cnt.astype(int)
                med_df.to_csv(per_year_dir / f"{y}_{v}_MEDIAN.csv", index=False)

            rows = []
            for (m1, m2) in DEFAULT_PAIRS:
                if (m1 in piv.columns) and (m2 in piv.columns):
                    part = piv[piv_index].copy()
                    part.insert(0, "year", y)
                    part["variable"] = v
                    part["method1"] = m1
                    part["method2"] = m2

                    a = piv[m1]
                    b = piv[m2]
                    if v in ("season_length", "total_eip_days"):
                        a = a.fillna(0)
                        b = b.fillna(0)

                    part["value_method1"] = a
                    part["value_method2"] = b
                    part["delta"] = a - b
                    rows.append(part)

            if rows:
                pd.concat(rows, ignore_index=True).to_csv(per_year_dir / f"{y}_{v}_PAIRWISE.csv", index=False)

    # Multi-year aggregates
    for var_short in ["start_doy", "end_doy", "season_length", "total_eip_days"]:
        if var_short not in vars_eff:
            continue

        pair_files = sorted(per_year_dir.glob(f"*_{var_short}_PAIRWISE.csv"))
        if pair_files:
            big = pd.concat([pd.read_csv(pf) for pf in pair_files], ignore_index=True)
            big = _normalize_columns(big)
            group_cols = [c for c in ["i", "j", "lat", "lon"] if c in big.columns] + ["method1", "method2", "variable"]
            grp = big.groupby(group_cols, dropna=False, as_index=False)["delta"].median()
            grp.to_csv(agg_dir / f"{var_short}_PAIRWISE_MEDIAN_ALL_YEARS.csv", index=False)

        med_files = sorted(per_year_dir.glob(f"*_{var_short}_MEDIAN.csv"))
        if med_files:
            med_big = pd.concat([pd.read_csv(pf) for pf in med_files], ignore_index=True)
            med_big = _normalize_columns(med_big)
            group_cols = [c for c in ["i", "j", "lat", "lon"] if c in med_big.columns] + ["variable"]
            med_grp = med_big.groupby(group_cols, dropna=False, as_index=False)["median_methods"].median()
            med_grp.to_csv(agg_dir / f"{var_short}_MEDIAN_CONSENSUS_ALL_YEARS.csv", index=False)


# ---------------------------------------------------------------------
# Colormaps / norms
# ---------------------------------------------------------------------
def _norm_values_for_var(var: str):
    # Conservative defaults; keep deterministic for publication.
    if var == "season_length":
        return mcolors.Normalize(vmin=0, vmax=175)
    if var == "start_doy":
        return mcolors.Normalize(vmin=75, vmax=225)
    if var == "end_doy":
        return mcolors.Normalize(vmin=150, vmax=300)
    if var == "total_eip_days":
        return mcolors.Normalize(vmin=0, vmax=175)
    return None


def _norm_delta_symmetric(vals: np.ndarray):
    a = np.abs(vals[np.isfinite(vals)])
    if a.size == 0:
        return mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    vmax = np.percentile(a, 98)
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = float(np.nanmax(a)) if a.size else 1.0
    if vmax == 0:
        vmax = 1.0
    return mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)


def _cmap_for_values(var: str):
    base = "bwr_r" if var == "start_doy" else "bwr"
    cm = plt.get_cmap(base).copy()
    cm.set_bad(NODATA_HEX)
    return cm


def _cmap_for_deltas():
    cm = plt.get_cmap("bwr").copy()
    cm.set_bad(NODATA_HEX)
    return cm


# ---------------------------------------------------------------------
# Map helpers
# ---------------------------------------------------------------------
def _grid_pcolormesh(ax, df: pd.DataFrame, vcol: str, cmap, norm, bbox=None):
    need = ["i", "j", "lat", "lon", vcol]
    if any(c not in df.columns for c in need):
        return None

    df2 = df[need].copy()
    for c in need:
        df2[c] = pd.to_numeric(df2[c], errors="coerce")
    df2 = df2.dropna(subset=["i", "j", "lat", "lon"])
    if df2.empty:
        return None

    lat_by_i = df2.groupby("i")["lat"].median().sort_index()
    lon_by_j = df2.groupby("j")["lon"].median().sort_index()

    Zp = df2.pivot_table(index="i", columns="j", values=vcol, aggfunc="mean")
    Zp = Zp.reindex(index=lat_by_i.index, columns=lon_by_j.index)
    Z = np.ma.masked_invalid(Zp.values)

    LAT2D = np.repeat(lat_by_i.values[:, None], len(lon_by_j), axis=1)
    LON2D = np.repeat(lon_by_j.values[None, :], len(lat_by_i), axis=0)

    m = ax.pcolormesh(LON2D, LAT2D, Z, cmap=cmap, norm=norm, shading="nearest")
    if bbox:
        N, W, S, E = bbox
        ax.set_xlim(W, E)
        ax.set_ylim(S, N)
    return m


def _tri_or_scatter(ax, df: pd.DataFrame, vcol: str, cmap, norm, bbox=None, kind="tri"):
    need = ["lat", "lon", vcol]
    if any(c not in df.columns for c in need):
        return None

    x = pd.to_numeric(df["lon"], errors="coerce").values
    y = pd.to_numeric(df["lat"], errors="coerce").values
    z = pd.to_numeric(df[vcol], errors="coerce").values
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[mask], y[mask], z[mask]
    if x.size == 0:
        return None

    if bbox:
        N, W, S, E = bbox
        sel = (y <= N) & (y >= S) & (x >= W) & (x <= E)
        x, y, z = x[sel], y[sel], z[sel]
        if x.size == 0:
            return None

    if kind == "scatter":
        return ax.scatter(x, y, c=z, cmap=cmap, norm=norm, s=8, edgecolor="none")

    try:
        import matplotlib.tri as mtri
        tri = mtri.Triangulation(x, y)
        return ax.tripcolor(tri, z, cmap=cmap, norm=norm, shading="flat")
    except Exception:
        return ax.scatter(x, y, c=z, cmap=cmap, norm=norm, s=8, edgecolor="none")


def _draw_outline(ax, shp_path: Optional[str]):
    if not shp_path:
        return
    try:
        import geopandas as gpd
        gpd.read_file(shp_path).to_crs(epsg=4326).boundary.plot(ax=ax, linewidth=0.7, color="k", zorder=10)
    except Exception as e:
        print(f"[map] Outline skipped: {e}")


# ---------------------------------------------------------------------
# Article tables (pairwise)
# ---------------------------------------------------------------------
def _article_pairwise_tables(
    agg_dir: Path,
    per_year_dir: Path,
    var: str,
    pairs: List[Tuple[str, str]],
    out_tables: Path,
    thr_list=(5, 10),
):
    agg = agg_dir / f"{var}_PAIRWISE_MEDIAN_ALL_YEARS.csv"
    if agg.exists():
        big = pd.read_csv(agg)
    else:
        files = sorted(per_year_dir.glob(f"*_{var}_PAIRWISE.csv"))
        if not files:
            return
        big = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)

    big = _normalize_columns(big)
    if "delta" not in big.columns or "method1" not in big.columns or "method2" not in big.columns:
        return

    big["delta"] = pd.to_numeric(big["delta"], errors="coerce")

    t0 = float(thr_list[0]) if len(thr_list) >= 1 else 5.0
    t1 = float(thr_list[1]) if len(thr_list) >= 2 else 10.0

    rows, rows_thr = [], []
    for m1, m2 in pairs:
        d = big[
            ((big["method1"] == m1) & (big["method2"] == m2)) |
            ((big["method1"] == m2) & (big["method2"] == m1))
        ].copy()
        if d.empty:
            continue

        rev = (d["method1"] == m2) & (d["method2"] == m1)
        if rev.any():
            d.loc[rev, "delta"] = -pd.to_numeric(d.loc[rev, "delta"], errors="coerce")
            d.loc[rev, "method1"] = m1
            d.loc[rev, "method2"] = m2

        s = pd.to_numeric(d["delta"], errors="coerce").to_numpy()
        s = s[np.isfinite(s)]
        if s.size == 0:
            continue

        rows.append({
            "variable": var,
            "pair": f"{m1}-{m2}",
            "n": int(s.size),
            "p25": float(np.nanpercentile(s, 25)),
            "median": float(np.nanmedian(s)),
            "p75": float(np.nanpercentile(s, 75)),
            "mean": float(np.nanmean(s)),
            "sd": float(np.nanstd(s, ddof=1)) if s.size > 1 else 0.0,
            "mad_mean": float(np.nanmean(np.abs(s - np.nanmean(s)))) if s.size else np.nan,
        })

        a = np.abs(s)
        rows_thr.append({
            "variable": var,
            "pair": f"{m1}-{m2}",
            "n": int(a.size),
            f"pct_abs<= {t0:g}": float(np.nanmean(a <= t0) * 100.0),
            f"pct_abs<= {t1:g}": float(np.nanmean(a <= t1) * 100.0),
            "max_abs": float(np.nanmax(a)),
        })

    if rows:
        pd.DataFrame(rows).to_csv(out_tables / f"article_pairwise_summary_{var}.csv", index=False)
    if rows_thr:
        pd.DataFrame(rows_thr).to_csv(out_tables / f"article_pairwise_thresholds_{var}.csv", index=False)


# ---------------------------------------------------------------------
# Scatter / Bland–Altman
# ---------------------------------------------------------------------
def _pair_scatter_all_years(per_year_dir: Path, var: str, pair: Tuple[str, str], out_plots: Path, dpi: int, style: str):
    files = sorted(per_year_dir.glob(f"*_{var}_PAIRWISE.csv"))
    if not files:
        return
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = _normalize_columns(df)

    m1, m2 = pair
    d = df[(df["method1"] == m1) & (df["method2"] == m2)].dropna(subset=["value_method1", "value_method2"])
    if d.empty:
        return

    a = pd.to_numeric(d["value_method1"], errors="coerce").to_numpy()
    b = pd.to_numeric(d["value_method2"], errors="coerce").to_numpy()
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if a.size == 0:
        return

    mn, mx = float(min(a.min(), b.min())), float(max(a.max(), b.max()))
    with _style_ctx(style):
        fig, ax = plt.subplots(figsize=(6.5, 6.5))
        ax.scatter(a, b, s=5, alpha=0.25, edgecolor="none")
        ax.plot([mn, mx], [mn, mx], "k--", lw=1)
        ax.set_xlabel(f"{m1} {DISPLAY.get(var, var)}")
        ax.set_ylabel(f"{m2} {DISPLAY.get(var, var)}")
        ax.set_title(f"{DISPLAY.get(var, var)} — {m1} vs {m2} (all years)")
        fig.tight_layout()
        fig.savefig(out_plots / f"scatter_{var}_{m1}-{m2}_ALLYEARS.png", dpi=dpi)
        plt.close(fig)


def _pair_bland_altman(per_year_dir: Path, var: str, pair: Tuple[str, str], out_plots: Path, dpi: int, style: str):
    files = sorted(per_year_dir.glob(f"*_{var}_PAIRWISE.csv"))
    if not files:
        return
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = _normalize_columns(df)

    m1, m2 = pair
    d = df[(df["method1"] == m1) & (df["method2"] == m2)].dropna(subset=["value_method1", "value_method2"])
    if d.empty:
        return

    a = pd.to_numeric(d["value_method1"], errors="coerce").to_numpy()
    b = pd.to_numeric(d["value_method2"], errors="coerce").to_numpy()
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if a.size == 0:
        return

    meanv = (a + b) / 2.0
    diff = (a - b)  # Δ = m1 − m2
    md = float(np.nanmean(diff))
    sd = float(np.nanstd(diff, ddof=1)) if diff.size > 1 else 0.0

    with _style_ctx(style):
        fig, ax = plt.subplots(figsize=(7.5, 5.2))
        ax.scatter(meanv, diff, s=5, alpha=0.25, edgecolor="none")
        ax.axhline(md, color="k", ls="--", lw=1, label=f"mean Δ (m1−m2) = {md:.2f}")
        ax.axhline(md + 1.96 * sd, color="r", ls="--", lw=1)
        ax.axhline(md - 1.96 * sd, color="r", ls="--", lw=1)
        ax.set_xlabel("Mean of methods")
        ax.set_ylabel(f"Δ {DISPLAY.get(var, var)} (m1−m2)")
        ax.set_title(f"Bland–Altman — {DISPLAY.get(var, var)} ({m1}−{m2})")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_plots / f"bland_altman_{var}_{m1}-{m2}_ALLYEARS.png", dpi=dpi)
        plt.close(fig)


# ---------------------------------------------------------------------
# Histograms (Δ all-years median per pixel)
# ---------------------------------------------------------------------
def _delta_histograms_all_years(agg_dir: Path, var: str, out_plots: Path, style: Optional[str], bins: int = 41):
    agg = agg_dir / f"{var}_PAIRWISE_MEDIAN_ALL_YEARS.csv"
    if not agg.exists():
        return

    df = pd.read_csv(agg)
    df = _normalize_columns(df)
    if "delta" not in df.columns:
        return

    df["delta"] = pd.to_numeric(df["delta"], errors="coerce")
    vals = df["delta"].to_numpy()
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return

    vmax = np.percentile(np.abs(vals), 98)
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = float(np.nanmax(np.abs(vals))) if vals.size else 1.0
    rng = (-float(vmax), float(vmax))

    with _style_ctx(style):
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        ax.hist(vals, bins=bins, range=rng, density=True)
        ax.axvline(0, color="k", lw=1)
        ax.set_title(f"Δ {DISPLAY.get(var, var)} — all-years median per pixel (ALL pairs)")
        ax.set_xlabel(f"Δ {DISPLAY.get(var, var)} (m1−m2)")
        ax.set_ylabel("Density")
        fig.tight_layout()
        fig.savefig(out_plots / f"hist_allyears_{var}_ALLPAIRS.png", dpi=180)
        plt.close(fig)

    for (m1, m2), sub in df.groupby(["method1", "method2"]):
        s = pd.to_numeric(sub["delta"], errors="coerce").to_numpy()
        s = s[np.isfinite(s)]
        if s.size == 0:
            continue
        with _style_ctx(style):
            fig, ax = plt.subplots(figsize=(7.2, 4.6))
            ax.hist(s, bins=bins, range=rng, density=True)
            ax.axvline(0, color="k", lw=1)
            ax.set_title(f"Δ {DISPLAY.get(var, var)} — all-years median per pixel — {m1}−{m2}")
            ax.set_xlabel(f"Δ {DISPLAY.get(var, var)} (m1−m2)")
            ax.set_ylabel("Density")
            fig.tight_layout()
            fig.savefig(out_plots / f"hist_allyears_{var}_{m1}-{m2}.png", dpi=180)
            plt.close(fig)


# ---------------------------------------------------------------------
# Maps
# ---------------------------------------------------------------------
def _map_values_per_year(
    per_year_dir: Path,
    var: str,
    bbox,
    out_maps_values: Path,
    fig_dpi=180,
    outline_shp: Optional[str] = None,
    map_kind="grid",
):
    cmap = _cmap_for_values(var)
    norm = _norm_values_for_var(var)

    for f in sorted(per_year_dir.glob(f"*_{var}_MEDIAN.csv")):
        try:
            year = int(f.name.split("_")[0])
        except Exception:
            continue

        df = pd.read_csv(f)
        df = _normalize_columns(df)

        vcol = "median_methods" if "median_methods" in df.columns else var
        if vcol not in df.columns:
            continue

        with _style_ctx("default"):
            fig, ax = plt.subplots(figsize=(6.8, 6.2))
            if map_kind == "grid":
                m = _grid_pcolormesh(ax, df, vcol, cmap, norm, bbox)
            else:
                m = _tri_or_scatter(ax, df, vcol, cmap, norm, bbox, kind=map_kind)

            if m is None:
                plt.close(fig)
                continue

            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cb = fig.colorbar(m, cax=cax)
            cb.set_label(DISPLAY.get(var, var))

            _draw_outline(ax, outline_shp)

            ax.set_title(f"{year} — {DISPLAY.get(var, var)} (median across methods)")
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            if bbox:
                N, W, S, E = bbox
                ax.set_xlim(W, E)
                ax.set_ylim(S, N)
            ax.set_aspect("equal", adjustable="box")
            fig.tight_layout()
            fig.savefig(out_maps_values / f"map_{year}_{var}.png", dpi=fig_dpi)
            plt.close(fig)


def _map_values_per_year_permethod(
    methods_csv: Path,
    var: str,
    bbox,
    out_maps_permethod: Path,
    fig_dpi=180,
    outline_shp: Optional[str] = None,
    map_kind="grid",
):
    base = pd.read_csv(methods_csv, low_memory=False)
    base = _normalize_columns(base)
    base = _filter_roi(base, bbox)

    if var not in base.columns:
        return
    if "method" not in base.columns or "year" not in base.columns:
        return

    cmap = _cmap_for_values(var)
    norm = _norm_values_for_var(var)

    base["year"] = pd.to_numeric(base["year"], errors="coerce")
    base = base.dropna(subset=["year"]).copy()
    base["year"] = base["year"].astype(int)

    methods = _stable_method_order(base["method"].dropna().astype(str).unique().tolist())

    for mth in methods:
        sub = base[base["method"].astype(str) == str(mth)].copy()
        if sub.empty:
            continue
        for y, year_df in sub.groupby("year"):
            yv = pd.to_numeric(y, errors="coerce")
            if not np.isfinite(yv):
                continue

            with _style_ctx("default"):
                fig, ax = plt.subplots(figsize=(6.8, 6.2))
                if map_kind == "grid":
                    m = _grid_pcolormesh(ax, year_df, var, cmap, norm, bbox)
                else:
                    m = _tri_or_scatter(ax, year_df, var, cmap, norm, bbox, kind=map_kind)
                if m is None:
                    plt.close(fig)
                    continue

                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                cb = fig.colorbar(m, cax=cax)
                cb.set_label(f"{DISPLAY.get(var, var)} — {mth}")

                _draw_outline(ax, outline_shp)

                ax.set_title(f"{int(yv)} — {DISPLAY.get(var, var)} ({mth})")
                ax.set_xlabel("Longitude")
                ax.set_ylabel("Latitude")
                if bbox:
                    N, W, S, E = bbox
                    ax.set_xlim(W, E)
                    ax.set_ylim(S, N)
                ax.set_aspect("equal", adjustable="box")
                fig.tight_layout()

                (out_maps_permethod / mth).mkdir(parents=True, exist_ok=True)
                fig.savefig(out_maps_permethod / mth / f"map_{int(yv)}_{var}_{mth}.png", dpi=fig_dpi)
                plt.close(fig)


def _map_deltas_per_year(
    per_year_dir: Path,
    var: str,
    years: Optional[List[int]],
    bbox,
    out_maps_deltas: Path,
    pair: Optional[Tuple[str, str]] = None,
    fig_dpi=180,
    outline_shp: Optional[str] = None,
    map_kind="grid",
    delta_flag: Optional[float] = None,
):
    files = sorted(per_year_dir.glob(f"*_{var}_PAIRWISE.csv"))
    if not files:
        return

    cmap = _cmap_for_deltas()

    for f in files:
        try:
            y = int(f.name.split("_")[0])
        except Exception:
            continue
        if years and y not in years:
            continue

        df = pd.read_csv(f)
        df = _normalize_columns(df)

        if not {"method1", "method2", "delta"}.issubset(df.columns):
            continue

        if pair:
            m1, m2 = pair
            groups = [(m1, m2, df[(df["method1"] == m1) & (df["method2"] == m2)].copy())]
        else:
            groups = [(mm1, mm2, sub.copy()) for (mm1, mm2), sub in df.groupby(["method1", "method2"])]

        for mm1, mm2, sel in groups:
            sel = _filter_roi(sel, bbox)
            if sel.empty:
                continue

            vals = pd.to_numeric(sel["delta"], errors="coerce").to_numpy()
            norm = _norm_delta_symmetric(vals)

            with _style_ctx("default"):
                fig, ax = plt.subplots(figsize=(7.1, 6))
                ax.set_title(f"Δ {DISPLAY.get(var, var)} ({mm1}−{mm2}) — {y}")
                ax.set_xlabel("Longitude")
                ax.set_ylabel("Latitude")

                if map_kind == "grid":
                    m = _grid_pcolormesh(ax, sel, "delta", cmap, norm, bbox)
                else:
                    m = _tri_or_scatter(ax, sel, "delta", cmap, norm, bbox, kind=map_kind)

                if m is not None:
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("right", size="5%", pad=0.05)
                    cb = fig.colorbar(m, cax=cax)
                    cb.set_label(f"Δ {DISPLAY.get(var, var)}")

                _draw_outline(ax, outline_shp)

                if bbox:
                    N, W, S, E = bbox
                    ax.set_xlim(W, E)
                    ax.set_ylim(S, N)
                ax.set_aspect("equal", adjustable="box")
                fig.tight_layout()

                title_pair = f"{mm1}-{mm2}".replace(" ", "")
                fig.savefig(out_maps_deltas / f"map_{y}_{var}_DELTA_{title_pair}.png", dpi=fig_dpi)
                plt.close(fig)

            if delta_flag is not None:
                sel2 = sel.copy()
                sel2["flag"] = (pd.to_numeric(sel2["delta"], errors="coerce").abs() >= float(delta_flag)).astype(float)
                vnorm = mcolors.Normalize(vmin=0, vmax=1)

                with _style_ctx("default"):
                    fig2, ax2 = plt.subplots(figsize=(7.1, 6))
                    ax2.set_title(f"Disagreement |Δ {DISPLAY.get(var, var)}| ≥ {delta_flag} ({mm1}−{mm2}) — {y}")
                    ax2.set_xlabel("Longitude")
                    ax2.set_ylabel("Latitude")

                    if map_kind == "grid":
                        m2 = _grid_pcolormesh(ax2, sel2, "flag", plt.get_cmap("gray_r"), vnorm, bbox)
                    else:
                        m2 = _tri_or_scatter(ax2, sel2, "flag", plt.get_cmap("gray_r"), vnorm, bbox, kind=map_kind)

                    if m2 is not None:
                        divider2 = make_axes_locatable(ax2)
                        cax2 = divider2.append_axes("right", size="5%", pad=0.05)
                        cb2 = fig2.colorbar(m2, cax=cax2)
                        cb2.set_label("flag (0/1)")

                    _draw_outline(ax2, outline_shp)

                    if bbox:
                        N, W, S, E = bbox
                        ax2.set_xlim(W, E)
                        ax2.set_ylim(S, N)
                    ax2.set_aspect("equal", adjustable="box")
                    fig2.tight_layout()

                    title_pair = f"{mm1}-{mm2}".replace(" ", "")
                    fig2.savefig(out_maps_deltas / f"map_{y}_{var}_DELTAFLAG_{delta_flag}_{title_pair}.png", dpi=fig_dpi)
                    plt.close(fig2)


# ---------------------------------------------------------------------
# Delta summary per year
# ---------------------------------------------------------------------
def _delta_summary_table_from_pairwise_files(
    per_year_dir: Path,
    var: str,
    years: Optional[List[int]],
    bbox,
    out_tables: Path,
) -> None:
    files = sorted(per_year_dir.glob(f"*_{var}_PAIRWISE.csv"))
    if not files:
        return

    df = pd.concat([pd.read_csv(f, low_memory=False) for f in files], ignore_index=True)
    df = _normalize_columns(df)
    df = _filter_roi(df, bbox)

    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
        df = df.dropna(subset=["year"]).copy()
        df["year"] = df["year"].astype(int)

    if years is not None and "year" in df.columns:
        df = df[df["year"].isin(years)].copy()

    if not {"delta", "method1", "method2"}.issubset(df.columns):
        return

    df["delta"] = pd.to_numeric(df["delta"], errors="coerce")
    df = df.dropna(subset=["delta"]).copy()
    if df.empty:
        return

    g = df.groupby(["year", "method1", "method2"], dropna=False)["delta"]
    summ = g.median().reset_index(name="median_delta")
    summ["p25"] = g.quantile(0.25).values
    summ["p75"] = g.quantile(0.75).values
    summ["mean"] = g.mean().values
    summ["sd"] = g.std(ddof=1).values
    summ["count"] = g.count().values

    summ.to_csv(out_tables / f"delta_summary_{var}.csv", index=False)


# ---------------------------------------------------------------------
# PHASE B: Graphs & Maps
# ---------------------------------------------------------------------
def graphs_phase(
    out_root: Path,
    years: Optional[List[int]],
    variables: List[str],
    pair_focus: Optional[Tuple[str, str]],
    fig_dpi: int,
    style: str,
    bbox,
    methods_csv: Optional[Path],
    pixel_latlon: Optional[Tuple[float, float]],
    outline_shp: Optional[str],
    map_kind="grid",
    article_thr="5,10",
    delta_flag=10.0,
    permethod_maps: bool = False,
):
    dirs = _ensure_out_structure(out_root)
    out_tables = dirs["tables"]
    per_year_dir = dirs["per_year"]
    agg_dir = dirs["agg"]
    out_plots = dirs["plots"]
    out_maps_values = dirs["maps_values"]
    out_maps_deltas = dirs["maps_deltas"]
    out_maps_permethod = dirs["maps_permethod"]

    pairs = DEFAULT_PAIRS
    plt.rcParams.update({"figure.dpi": fig_dpi})

    # National series from CONSENSUS
    for var in variables:
        rows = []
        for f in sorted(per_year_dir.glob(f"*_{var}_MEDIAN.csv")):
            try:
                y = int(f.name.split("_")[0])
            except Exception:
                continue
            if years and y not in years:
                continue

            df = pd.read_csv(f)
            df = _normalize_columns(df)
            df = _filter_roi(df, bbox)

            vcol = "median_methods" if "median_methods" in df.columns else var
            if vcol not in df.columns:
                continue

            v = pd.to_numeric(df.get(vcol), errors="coerce")
            rows.append({
                "year": y,
                "median": float(v.median()),
                "p25": float(v.quantile(0.25)),
                "p75": float(v.quantile(0.75)),
                "count": int(v.count()),
            })

        if rows:
            summ = pd.DataFrame(rows).sort_values("year")
            summ.to_csv(out_tables / f"national_summary_{var}.csv", index=False)

            with _style_ctx(style):
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(summ["year"], summ["median"], lw=2, marker="o", label="Median (consensus)")
                ax.fill_between(summ["year"], summ["p25"], summ["p75"], alpha=0.18, label="IQR")

                x = summ["year"].values.astype(float)
                yv = summ["median"].values.astype(float)
                if np.isfinite(x).sum() >= 2 and np.isfinite(yv).sum() >= 2:
                    a, b = np.polyfit(x, yv, 1)
                    ax.plot(x, a * x + b, lw=1.6, ls="--", alpha=0.85, label=f"Trend ≈ {a:.2f}/yr")

                ax.set_title(f"National time series — {DISPLAY.get(var, var)}")
                ax.set_xlabel("Year")
                _set_year_ticks(ax, summ["year"].tolist())
                ax.set_ylabel(DISPLAY.get(var, var))
                ax.legend()
                fig.tight_layout()
                fig.savefig(out_plots / f"timeseries_{var}.png", dpi=fig_dpi)
                plt.close(fig)

    # Per-method national series (tables + plot). Per-method maps optional.
    if methods_csv and methods_csv.exists():
        base = pd.read_csv(methods_csv, low_memory=False)
        base = _normalize_columns(base)
        base = _filter_roi(base, bbox)

        if "year" in base.columns:
            base["year"] = pd.to_numeric(base["year"], errors="coerce")
            base = base.dropna(subset=["year"]).copy()
            base["year"] = base["year"].astype(int)

        if years is not None and "year" in base.columns:
            base = base[base["year"].isin(years)].copy()

        axis_years = sorted(base["year"].unique().tolist()) if "year" in base.columns else []
        methods_list = _stable_method_order(base["method"].dropna().astype(str).unique().tolist()) if "method" in base.columns else []

        for var in variables:
            if var not in base.columns:
                continue

            with _style_ctx(style):
                fig, ax = plt.subplots(figsize=(10, 5))

                for m in methods_list:
                    sub = base[base["method"].astype(str) == str(m)][["year", var]].copy()
                    sub[var] = pd.to_numeric(sub[var], errors="coerce")
                    sub = sub.dropna(subset=[var])
                    if sub.empty:
                        continue

                    g = sub.groupby("year")[var]
                    summ_m = pd.DataFrame({
                        "year": g.median().index.astype(int),
                        "median": g.median().values,
                        "p25": g.quantile(0.25).values,
                        "p75": g.quantile(0.75).values,
                        "count": g.count().values,
                    })

                    safe_m = str(m).replace(" ", "_")
                    summ_m.to_csv(out_tables / f"national_summary_{var}_method_{safe_m}.csv", index=False)
                    ax.plot(summ_m["year"], summ_m["median"], marker="o", lw=1.6, label=m)

                ax.set_xlabel("Year")
                _set_year_ticks(ax, axis_years)
                ax.set_ylabel(DISPLAY.get(var, var))
                ax.legend()
                fig.tight_layout()
                fig.savefig(out_plots / f"timeseries_permethod_{var}.png", dpi=fig_dpi)
                plt.close(fig)

            if permethod_maps:
                _map_values_per_year_permethod(methods_csv, var, bbox, out_maps_permethod, fig_dpi, outline_shp, map_kind)

    # thresholds parsing
    thr_list = tuple(int(x) for x in str(article_thr).split(",") if str(x).strip().isdigit())
    if len(thr_list) == 0:
        thr_list = (5, 10)
    elif len(thr_list) == 1:
        thr_list = (thr_list[0], 10)

    # Pairwise analysis + plots
    for var in variables:
        for (m1, m2) in pairs:
            parts = []
            for f in sorted(per_year_dir.glob(f"*_{var}_PAIRWISE.csv")):
                d = pd.read_csv(f)
                d = _normalize_columns(d)
                if not {"year", "method1", "method2", "delta"}.issubset(d.columns):
                    continue
                sel = d[(d["method1"] == m1) & (d["method2"] == m2)][["year", "delta"]].copy()
                if years is not None:
                    sel["year"] = pd.to_numeric(sel["year"], errors="coerce")
                    sel = sel.dropna(subset=["year"])
                    sel["year"] = sel["year"].astype(int)
                    sel = sel[sel["year"].isin(years)]
                if sel.empty:
                    continue
                parts.append(sel)

            if parts:
                big = pd.concat(parts, ignore_index=True)
                with _style_ctx(style):
                    fig, ax = plt.subplots(figsize=(12, 5))
                    big.boxplot(column="delta", by="year", ax=ax, grid=False, showfliers=False)
                    ax.axhline(0, color="k", lw=1, alpha=0.6)
                    ax.set_title(f"Δ {DISPLAY.get(var, var)} ({m1} − {m2}) per year")
                    ax.set_xlabel("Year")
                    _set_year_ticks(ax, big["year"].tolist())
                    ax.set_ylabel(f"Δ {DISPLAY.get(var, var)} (m1−m2)")
                    plt.suptitle("")
                    fig.tight_layout()
                    fig.savefig(out_plots / f"delta_box_{var}_{m1}-{m2}.png", dpi=fig_dpi)
                    plt.close(fig)

            _pair_scatter_all_years(per_year_dir, var, (m1, m2), out_plots, fig_dpi, style)
            _pair_bland_altman(per_year_dir, var, (m1, m2), out_plots, fig_dpi, style)

        _article_pairwise_tables(agg_dir, per_year_dir, var, pairs, out_tables, thr_list)
        _delta_histograms_all_years(agg_dir, var, out_plots, style)
        _delta_summary_table_from_pairwise_files(per_year_dir, var, years, bbox, out_tables)

    # maps values
    for v in ["season_length", "start_doy", "end_doy", "total_eip_days"]:
        if v in variables:
            _map_values_per_year(per_year_dir, v, bbox, out_maps_values, fig_dpi, outline_shp, map_kind)

    # maps deltas
    for var in variables:
        _map_deltas_per_year(
            per_year_dir, var, years, bbox, out_maps_deltas,
            pair=pair_focus, fig_dpi=fig_dpi,
            outline_shp=outline_shp, map_kind=map_kind,
            delta_flag=delta_flag
        )

    # optional nearest pixel series
    if pixel_latlon is not None:
        plat, plon = pixel_latlon
        for var in variables:
            series = []
            for f in sorted(per_year_dir.glob(f"*_{var}_MEDIAN.csv")):
                try:
                    year = int(f.name.split("_")[0])
                except Exception:
                    continue
                if years and year not in years:
                    continue

                df = pd.read_csv(f)
                df = _normalize_columns(df)
                if not {"lat", "lon"}.issubset(df.columns):
                    continue
                df = _filter_roi(df, bbox)

                df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
                df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
                df["dist2"] = (df["lat"] - plat) ** 2 + (df["lon"] - plon) ** 2
                row = df.sort_values("dist2").head(1)
                if row.empty:
                    continue

                val = pd.to_numeric(row.get("median_methods"), errors="coerce").iloc[0]
                if np.isfinite(val):
                    series.append((year, float(val)))

            if series:
                series.sort(key=lambda x: x[0])
                s_df = pd.DataFrame(series, columns=["year", "value"])
                s_df.to_csv(out_tables / f"pixel_series_{var}.csv", index=False)

                with _style_ctx("default"):
                    fig, ax = plt.subplots(figsize=(8.2, 4.4))
                    ax.plot(s_df["year"], s_df["value"], marker="o")
                    ax.set_title(f"Pixel series (nearest to {plat:.4f},{plon:.4f}) — {DISPLAY.get(var, var)}")
                    ax.set_xlabel("Year")
                    _set_year_ticks(ax, s_df["year"].tolist())
                    ax.set_ylabel(DISPLAY.get(var, var))
                    ax.grid(True, alpha=0.25)
                    fig.tight_layout()
                    fig.savefig(out_plots / f"pixel_series_{var}.png", dpi=fig_dpi)
                    plt.close(fig)


# ---------------------------------------------------------------------
# Excel compact
# ---------------------------------------------------------------------
def excel_compact_summary(out_root: Path, window: int, variables: List[str]) -> Path:
    dirs = _ensure_out_structure(out_root)
    xlsx_path = dirs["excel"] / f"win{window}_summary.xlsx"
    tables_dir = dirs["tables"]
    agg_dir = dirs["agg"]
    meta_dir = dirs["meta"]

    meta_files = sorted(meta_dir.glob(f"META_{window}D.json"))

    with pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as xlw:
        if meta_files:
            meta = json.loads(meta_files[0].read_text(encoding="utf-8"))
            pd.json_normalize(meta).to_excel(xlw, sheet_name="META", index=False)

        files_idx = []
        for f in tables_dir.rglob("*.csv"):
            rel = f.relative_to(out_root).as_posix()
            try:
                sz = f.stat().st_size / 1024 / 1024
            except Exception:
                sz = np.nan
            files_idx.append({"file": rel, "size_MB": round(float(sz), 3)})
        if files_idx:
            pd.DataFrame(files_idx).sort_values("file").to_excel(xlw, sheet_name="FILES_INDEX", index=False)

        for var in variables:
            for stem, sheet in [
                (f"national_summary_{var}.csv", f"nat_{var}"),
                (f"delta_summary_{var}.csv", f"delta_{var}"),
                (f"article_pairwise_summary_{var}.csv", f"pairs_{var}"),
                (f"article_pairwise_thresholds_{var}.csv", f"pairs_thr_{var}"),
            ]:
                f = tables_dir / stem
                if f.exists():
                    pd.read_csv(f).to_excel(xlw, sheet_name=sheet[:31], index=False)

            agg_pairs = agg_dir / f"{var}_PAIRWISE_MEDIAN_ALL_YEARS.csv"
            if agg_pairs.exists():
                d = pd.read_csv(agg_pairs)
                d = _normalize_columns(d)
                if not d.empty and "delta" in d.columns and {"method1", "method2"}.issubset(d.columns):
                    d["delta"] = pd.to_numeric(d["delta"], errors="coerce")
                    grp = d.groupby(["method1", "method2"], dropna=False)["delta"].agg(
                        n="count",
                        mean="mean",
                        sd="std",
                        p25=lambda s: np.nanpercentile(s, 25),
                        median="median",
                        p75=lambda s: np.nanpercentile(s, 75),
                        min="min",
                        max="max",
                    ).reset_index()
                    grp.to_excel(xlw, sheet_name=f"agg_stats_{var}"[:31], index=False)

        for ws in xlw.sheets.values():
            ws.set_zoom(110)

    return xlsx_path


# ---------------------------------------------------------------------
# Full Excel report (optional)
# (kept as in your version, with minimal portability-sensitive edits)
# ---------------------------------------------------------------------
def _safe_rel(p: Path, root: Path) -> str:
    try:
        return p.relative_to(root).as_posix()
    except Exception:
        return p.as_posix()


def _glob_many(root: Path, patterns: List[str]) -> List[Path]:
    out: List[Path] = []
    for pat in patterns:
        out += sorted(root.rglob(pat))
    return out


def _write_chunked_sheet(xlw: pd.ExcelWriter, df: pd.DataFrame, base_name: str, max_rows: int = 1_000_000) -> None:
    if df is None or df.empty:
        return
    base_name = base_name.replace(" ", "_")
    if len(df) <= max_rows:
        df.to_excel(xlw, sheet_name=base_name[:31], index=False)
        return
    if "year" in df.columns:
        for key, chunk in df.groupby("year"):
            if chunk.empty:
                continue
            suffix = f"_{int(key)}" if pd.notna(key) else "_part"
            sheet_name = (base_name + suffix)[:31]
            chunk.to_excel(xlw, sheet_name=sheet_name, index=False)
    else:
        n = len(df)
        nchunks = (n + max_rows - 1) // max_rows
        for k in range(nchunks):
            chunk = df.iloc[k * max_rows: (k + 1) * max_rows]
            if chunk.empty:
                continue
            sheet_name = (base_name + f"_{k+1}")[:31]
            chunk.to_excel(xlw, sheet_name=sheet_name, index=False)


def _write_raw_values_sheets(
    xlw: pd.ExcelWriter,
    out_root: Path,
    variables: List[str],
    methods_csv: Optional[Path],
    bbox,
) -> None:
    if methods_csv is None or not methods_csv.exists():
        print("[Excel-FULL] RAW sheets skipped (methods_csv missing).")
        return

    df = pd.read_csv(methods_csv, low_memory=False)
    df = _normalize_columns(df)
    df = _filter_roi(df, bbox)

    if "year" not in df.columns or "method" not in df.columns:
        print("[Excel-FULL] RAW sheets skipped (year/method columns missing).")
        return

    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)

    methods = _stable_method_order(df["method"].dropna().astype(str).unique().tolist())

    for var in variables:
        if var not in df.columns:
            continue

        cols = [c for c in ["year", "method", "i", "j", "lat", "lon", var] if c in df.columns]
        base = df[cols].copy()
        base[var] = pd.to_numeric(base[var], errors="coerce")
        base = base.dropna(subset=[var])

        cons_parts = []
        per_year_dir = (out_root / "01_tables" / "per_year")
        for cf in sorted(per_year_dir.glob(f"*_{var}_MEDIAN.csv")):
            try:
                cd = pd.read_csv(cf, low_memory=False)
            except Exception:
                continue
            cd = _normalize_columns(cd)
            need = [c for c in ["year", "i", "j", "lat", "lon", "median_methods"] if c in cd.columns]
            if need:
                sub_c = cd[need].copy().rename(columns={"median_methods": "consensus"})
                if not sub_c.empty:
                    cons_parts.append(sub_c)

        if cons_parts:
            cons = pd.concat(cons_parts, ignore_index=True)
            cons["consensus"] = pd.to_numeric(cons["consensus"], errors="coerce")
            cons = cons.dropna(subset=["consensus"])
            _write_chunked_sheet(xlw, cons, base_name=f"RAW_cons_{var}")

        for m in methods:
            sub = base[base["method"].astype(str) == str(m)].copy()
            if sub.empty:
                continue
            sub = sub.drop(columns=["method"], errors="ignore")
            _write_chunked_sheet(xlw, sub, base_name=f"RAW_{var}_{m}")


def excel_full_report(
    out_root: Path,
    window: int,
    variables: List[str],
    embed_images: bool,
    methods_csv: Optional[Path],
    bbox,
) -> Path:
    dirs = _ensure_out_structure(out_root)
    tables_dir = dirs["tables"]
    plots_dir = dirs["plots"]
    maps_dir = dirs["maps"]
    meta_dir = dirs["meta"]
    xlsx_path = dirs["excel"] / f"win{window}_REPORT.xlsx"

    meta_files = list(meta_dir.glob(f"META_{window}D.json"))
    meta_row = {}
    if meta_files:
        try:
            meta_row = json.loads(meta_files[0].read_text(encoding="utf-8"))
        except Exception:
            meta_row = {}

    with pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as xlw:
        readme_rows = [
            {"Section": "Scope", "Details": f"Window={window}d; Variables={', '.join(variables)}; Δ is canonical: m1−m2."},
            {"Section": "Colors", "Details": "Values: bwr (blue=low, red=high), start_doy uses bwr_r; NaN=#FFF5B1; square pixels; equal aspect."},
            {"Section": "RAW sheets", "Details": "RAW_cons_<var> = consensus per pixel & year; RAW_<var>_<method> = per-pixel values for each method."},
        ]
        pd.DataFrame(readme_rows).to_excel(xlw, sheet_name="README", index=False)

        if meta_row:
            pd.json_normalize(meta_row).to_excel(xlw, sheet_name="META", index=False)

        files_idx = []
        for p in _glob_many(out_root, ["01_tables/**/*.csv", "02_plots/**/*.png", "03_maps/**/*.png"]):
            rel = _safe_rel(p, out_root)
            try:
                sz = p.stat().st_size / 1024 / 1024
            except Exception:
                sz = np.nan
            files_idx.append({"file": rel, "size_MB": round(float(sz), 3)})
        if files_idx:
            pd.DataFrame(files_idx).sort_values("file").to_excel(xlw, sheet_name="DATA_INDEX", index=False)

        dict_rows = [
            {"Column": "start_doy", "Meaning": "day-of-year of season start (CONSENSUS); lower=earlier"},
            {"Column": "end_doy", "Meaning": "day-of-year of season end (CONSENSUS); higher=later"},
            {"Column": "season_length", "Meaning": "days between start/end inclusive (CONSENSUS)"},
            {"Column": "total_eip_days", "Meaning": "count of days with EIP-suitable window in the year"},
            {"Column": "delta (m1−m2)", "Meaning": "pairwise difference in canonical order; negative => method2 > method1"},
        ]
        pd.DataFrame(dict_rows).to_excel(xlw, sheet_name="DICTIONARY", index=False)

        _write_raw_values_sheets(xlw, out_root, variables, methods_csv, bbox)

        if embed_images:
            wb = xlw.book
            ws_ts = wb.add_worksheet("IMG_timeseries")
            ws_dlt = wb.add_worksheet("IMG_deltas")
            ws_maps = wb.add_worksheet("IMG_maps_sample")

            row_ts = 0
            for var in variables:
                for p in _glob_many(plots_dir, [f"timeseries_{var}.png", f"timeseries_permethod_{var}.png"]):
                    ws_ts.insert_image(row_ts, 0, str(p))
                    row_ts += 30

            row_d = 0
            for p in _glob_many(plots_dir, ["delta_box_*.png", "scatter_*ALLYEARS.png", "bland_altman_*ALLYEARS.png", "hist_allyears_*.png"]):
                ws_dlt.insert_image(row_d, 0, str(p))
                row_d += 30

            row_m = 0
            map_candidates = _glob_many(maps_dir, ["values/map_*_season_length.png", "deltas/map_*_season_length_DELTA_*.png"])
            for p in map_candidates[:12]:
                ws_maps.insert_image(row_m, 0, str(p))
                row_m += 30

        for ws in xlw.sheets.values():
            ws.set_zoom(110)

    return xlsx_path


# ---------------------------------------------------------------------
# Pixel summary from 6B diffs (optional)
# ---------------------------------------------------------------------
def _rename_latlon_cols(df: pd.DataFrame) -> pd.DataFrame:
    if "latitude" in df.columns and "lat" not in df.columns:
        df = df.rename(columns={"latitude": "lat"})
    if "longitude" in df.columns and "lon" not in df.columns:
        df = df.rename(columns={"longitude": "lon"})
    return df


def _infer_method_alt_from_filename(p: Path) -> Optional[str]:
    name = p.stem.lower()
    if "_delta_" in name and "_vs_hourly_sum" in name:
        mid = name.split("_delta_")[1].split("_vs_hourly_sum")[0]
        return mid if mid else None
    return None


def _ensure_sign_column(d: pd.DataFrame) -> pd.DataFrame:
    if "sign" in d.columns:
        return d
    d = d.copy()
    d["delta"] = pd.to_numeric(d["delta"], errors="coerce")

    def _sgn(x):
        if pd.isna(x):
            return np.nan
        if abs(x) < 1e-9:
            return 0
        return 1 if x > 0 else -1

    d["sign"] = d["delta"].apply(_sgn)
    return d


def _list_diff_files(diffs_dir: Path, years: Optional[List[int]], varname: str) -> List[Path]:
    files: List[Path] = []
    if years is None:
        year_dirs = [d for d in sorted(diffs_dir.glob("*")) if d.is_dir() and d.name.isdigit()]
    else:
        year_dirs = [diffs_dir / str(y) for y in years]

    for ydir in year_dirs:
        if not ydir.exists():
            continue
        files += sorted(ydir.glob(f"{varname}_DELTA*_vs_hourly_sum.csv"))
        f0 = ydir / f"{varname}_DELTA_vs_hourly_sum.csv"
        if f0.exists() and f0 not in files:
            files.append(f0)
    return files


def _resolve_diffs_dir(diffs_cli: Optional[str], output_root_base: Path, window: int) -> Path:
    if diffs_cli:
        p = Path(diffs_cli.format(win=window)) if "{win}" in diffs_cli else Path(diffs_cli)
        return p
    return output_root_base / f"win{window}" / "6B" / "diffs"


def pixel_summary_from_diffs(
    diffs_dir: Path,
    out_root: Path,
    years: Optional[List[int]],
    varlist: List[str],
    stab_med_abs_le: float = 1.0,
    stab_p90_le: float = 3.0,
) -> None:
    if not diffs_dir.exists():
        print(f"[pixel] diffs folder not found: {diffs_dir}")
        return

    dirs = _ensure_out_structure(out_root)
    out_dir = dirs["pixel"]

    for v in varlist:
        paths = _list_diff_files(diffs_dir, years, v)
        if not paths:
            print(f"[pixel:{v}] no diffs found.")
            continue

        frames = []
        for p in paths:
            try:
                d = pd.read_csv(p, low_memory=False)
                d = _rename_latlon_cols(d)

                if "method_alt" not in d.columns:
                    inferred = _infer_method_alt_from_filename(p)
                    if inferred:
                        d["method_alt"] = inferred

                if "variable" not in d.columns:
                    d["variable"] = v

                if "delta" not in d.columns:
                    continue

                d = _ensure_sign_column(d)

                need = {"year", "i", "j", "lat", "lon", "variable", "method_alt", "delta", "sign"}
                miss = need.difference(d.columns)
                if "year" in miss and p.parent.name.isdigit():
                    d["year"] = int(p.parent.name)
                    miss = need.difference(d.columns)

                if miss:
                    print(f"  [pixel warn] {p.name}: missing {sorted(miss)} — skipped.")
                    continue

                keep = ["year", "i", "j", "lat", "lon", "variable", "method_alt", "delta", "sign"]
                d = d[keep].copy()
                for col in ("year", "i", "j", "lat", "lon", "delta", "sign"):
                    d[col] = pd.to_numeric(d[col], errors="coerce")
                d = d[pd.notna(d["delta"])].copy()
                frames.append(d)

            except Exception as e:
                print(f"  [pixel err] {p.name}: {e}")

        if not frames:
            print(f"[pixel:{v}] no valid diffs rows.")
            continue

        big = pd.concat(frames, ignore_index=True)

        def _abs_p90(x):
            x = np.asarray(x, float)
            x = x[np.isfinite(x)]
            if x.size == 0:
                return np.nan
            return float(np.nanpercentile(np.abs(x), 90))

        grp = big.groupby(["i", "j", "lat", "lon", "variable", "method_alt"], as_index=False).agg(
            years_present=("year", lambda s: int(pd.Series(s).nunique())),
            delta_mean=("delta", "mean"),
            delta_median=("delta", "median"),
            delta_p10=("delta", lambda x: float(np.nanpercentile(np.asarray(x, float), 10))),
            delta_p90=("delta", lambda x: float(np.nanpercentile(np.asarray(x, float), 90))),
            delta_abs_p90=("delta", lambda x: _abs_p90(x)),
            delta_abs_max=("delta", lambda x: float(np.nanmax(np.abs(np.asarray(x, float))))),
            n=("delta", "size"),
            n_later=("sign", lambda s: int((pd.Series(s) == 1).sum())),
            n_earlier=("sign", lambda s: int((pd.Series(s) == -1).sum())),
            n_equal=("sign", lambda s: int((pd.Series(s) == 0).sum())),
        )

        grp["pct_later"] = (grp["n_later"] / grp["n"] * 100).round(2)
        grp["pct_earlier"] = (grp["n_earlier"] / grp["n"] * 100).round(2)
        grp["pct_equal"] = (grp["n_equal"] / grp["n"] * 100).round(2)

        grp["stability_flag"] = (grp["delta_median"].abs() <= float(stab_med_abs_le)) & (grp["delta_abs_p90"] <= float(stab_p90_le))

        out_csv = out_dir / f"{v}_PIXEL_SUMMARY_ALL_YEARS.csv"
        grp.to_csv(out_csv, index=False)
        print(f"[pixel:{v}] → {out_csv}  ({len(grp)} pixel×method rows)")


# ---------------------------------------------------------------------
# Combined export (optional)
# ---------------------------------------------------------------------
def export_combined_results(
    output_root_base: Path,
    windows: List[int],
    variables: List[str],
    combined_dir: Optional[Path] = None,
) -> Tuple[Path, Path]:
    combined_dir = combined_dir or (output_root_base / "00_combined")
    combined_dir.mkdir(parents=True, exist_ok=True)
    csv_dir = combined_dir / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)

    meta_rows = []
    nat_cons = []
    nat_methods = []
    delta_year = []
    article_sum = []
    article_thr = []
    agg_stats = []
    pixel_sum = []
    files_index = []

    for W in windows:
        out_root = output_root_base / f"win{W}" / "6.compare"
        if not out_root.exists():
            continue

        meta_path = out_root / "00_meta" / f"META_{W}D.json"
        tables_dir = out_root / "01_tables"
        agg_dir = tables_dir / "agg"
        pix_dir = tables_dir / "pixel_summary"

        if meta_path.exists():
            try:
                m = json.loads(meta_path.read_text(encoding="utf-8"))
                m["window"] = W
                meta_rows.append(m)
                files_index.append({"window": W, "kind": "META", "file": str(meta_path.relative_to(output_root_base))})
            except Exception:
                pass

        for var in variables:
            p_nat = tables_dir / f"national_summary_{var}.csv"
            if p_nat.exists():
                d = pd.read_csv(p_nat)
                d["window"] = W
                d["variable"] = var
                nat_cons.append(d)

            for p in tables_dir.glob(f"national_summary_{var}_method_*.csv"):
                d = pd.read_csv(p)
                meth = p.stem.split(f"national_summary_{var}_method_", 1)[-1]
                d["window"] = W
                d["variable"] = var
                d["method"] = meth
                nat_methods.append(d)

            p_dy = tables_dir / f"delta_summary_{var}.csv"
            if p_dy.exists():
                d = pd.read_csv(p_dy)
                d["window"] = W
                d["variable"] = var
                delta_year.append(d)

            p_as = tables_dir / f"article_pairwise_summary_{var}.csv"
            if p_as.exists():
                d = pd.read_csv(p_as)
                d["window"] = W
                d["variable"] = var
                article_sum.append(d)

            p_at = tables_dir / f"article_pairwise_thresholds_{var}.csv"
            if p_at.exists():
                d = pd.read_csv(p_at)
                d["window"] = W
                d["variable"] = var
                article_thr.append(d)

            p_agg = agg_dir / f"{var}_PAIRWISE_MEDIAN_ALL_YEARS.csv"
            if p_agg.exists():
                d = pd.read_csv(p_agg, low_memory=False)
                d = _normalize_columns(d)
                if "delta" in d.columns and {"method1", "method2"}.issubset(d.columns):
                    d["delta"] = pd.to_numeric(d["delta"], errors="coerce")
                    d = d.dropna(subset=["delta"])
                    if not d.empty:
                        g = d.groupby(["method1", "method2"], dropna=False)["delta"]
                        st = g.agg(n="count", mean="mean", sd="std", median="median", min="min", max="max").reset_index()
                        st["p25"] = g.quantile(0.25).values
                        st["p75"] = g.quantile(0.75).values
                        st["window"] = W
                        st["variable"] = var
                        agg_stats.append(st)

            p_pix = pix_dir / f"{var}_PIXEL_SUMMARY_ALL_YEARS.csv"
            if p_pix.exists():
                d = pd.read_csv(p_pix)
                d["window"] = W
                d["variable"] = var
                pixel_sum.append(d)

    df_meta = pd.json_normalize(meta_rows) if meta_rows else pd.DataFrame()
    df_nat = pd.concat(nat_cons, ignore_index=True) if nat_cons else pd.DataFrame()
    df_nat_m = pd.concat(nat_methods, ignore_index=True) if nat_methods else pd.DataFrame()
    df_dy = pd.concat(delta_year, ignore_index=True) if delta_year else pd.DataFrame()
    df_as = pd.concat(article_sum, ignore_index=True) if article_sum else pd.DataFrame()
    df_at = pd.concat(article_thr, ignore_index=True) if article_thr else pd.DataFrame()
    df_ag = pd.concat(agg_stats, ignore_index=True) if agg_stats else pd.DataFrame()
    df_px = pd.concat(pixel_sum, ignore_index=True) if pixel_sum else pd.DataFrame()
    df_idx = pd.DataFrame(files_index).sort_values(["window", "kind", "file"]) if files_index else pd.DataFrame()

    if not df_meta.empty: df_meta.to_csv(csv_dir / "META_ALL_WINDOWS.csv", index=False)
    if not df_nat.empty: df_nat.to_csv(csv_dir / "NATIONAL_CONSENSUS_ALL_WINDOWS.csv", index=False)
    if not df_nat_m.empty: df_nat_m.to_csv(csv_dir / "NATIONAL_BY_METHOD_ALL_WINDOWS.csv", index=False)
    if not df_dy.empty: df_dy.to_csv(csv_dir / "DELTA_BY_YEAR_ALL_WINDOWS.csv", index=False)
    if not df_as.empty: df_as.to_csv(csv_dir / "PAIRWISE_ARTICLE_SUMMARY_ALL_WINDOWS.csv", index=False)
    if not df_at.empty: df_at.to_csv(csv_dir / "PAIRWISE_THRESHOLDS_ALL_WINDOWS.csv", index=False)
    if not df_ag.empty: df_ag.to_csv(csv_dir / "AGG_PAIRWISE_STATS_ALL_WINDOWS.csv", index=False)
    if not df_px.empty: df_px.to_csv(csv_dir / "PIXEL_SUMMARY_ALL_WINDOWS.csv", index=False)
    if not df_idx.empty: df_idx.to_csv(csv_dir / "FILES_INDEX_ALL_WINDOWS.csv", index=False)

    xlsx_path = combined_dir / "ALL_WINDOWS_6_RESULTS.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as xlw:
        pd.DataFrame([{
            "What": "Combined results across windows",
            "Windows": ",".join(str(w) for w in windows),
            "Variables": ",".join(variables),
        }]).to_excel(xlw, sheet_name="README", index=False)

        if not df_meta.empty: df_meta.to_excel(xlw, sheet_name="META", index=False)
        if not df_idx.empty: df_idx.to_excel(xlw, sheet_name="FILES_INDEX", index=False)

        if not df_nat.empty: df_nat.to_excel(xlw, sheet_name="national_consensus", index=False)
        if not df_nat_m.empty: df_nat_m.to_excel(xlw, sheet_name="national_by_method", index=False)
        if not df_dy.empty: df_dy.to_excel(xlw, sheet_name="delta_by_year", index=False)
        if not df_as.empty: df_as.to_excel(xlw, sheet_name="article_summary", index=False)
        if not df_at.empty: df_at.to_excel(xlw, sheet_name="article_thresholds", index=False)
        if not df_ag.empty: df_ag.to_excel(xlw, sheet_name="agg_pairwise_stats", index=False)
        if not df_px.empty: df_px.to_excel(xlw, sheet_name="pixel_summary", index=False)

        for ws in xlw.sheets.values():
            ws.set_zoom(110)

    return xlsx_path, csv_dir


# ---------------------------------------------------------------------
# Prune helper (optional)
# ---------------------------------------------------------------------
def _prune_per_year(per_year_dir: Path) -> None:
    if not per_year_dir.exists():
        return
    try:
        for p in per_year_dir.glob("*.csv"):
            try:
                p.unlink()
            except Exception:
                pass
        try:
            if not any(per_year_dir.iterdir()):
                per_year_dir.rmdir()
        except Exception:
            pass
    except Exception:
        pass


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    global _CFG_ROOT, DEFAULT_METHODS_CSV_TEMPLATE

    ap = argparse.ArgumentParser(
        description="DDU method compare (CLEAN): compare + graphs + maps + excel (+ pixel summary) (+ combined)"
    )

    ap.add_argument("--root", type=str, default=None,
                    help="Repo/project root (sets DDU_ROOT; used for auto-discovery of step-5 outputs).")

    ap.add_argument("--windows", type=str, default="30", help="Comma list: 30 or 15,30")
    ap.add_argument("--methods-csv", type=str, default=DEFAULT_METHODS_CSV_TEMPLATE,
                    help=r"Path to season_ALL_YEARS_methods.csv; supports {win}")
    ap.add_argument("--output-root", type=str, default=DEFAULT_OUTPUT_ROOT,
                    help="Base output dir; will write into /win{W}/6.compare")

    ap.add_argument("--years", type=str, default=None, help="A-B or comma list")
    ap.add_argument("--methods", type=str, default=",".join(DEFAULT_METHODS))
    ap.add_argument("--variables", type=str, default="start_doy,end_doy,season_length,total_eip_days")

    ap.add_argument("--bbox", type=str, default=None, help="N,W,S,E (e.g., 48.5,19.9,43.5,30.1)")
    ap.add_argument("--pair", type=str, default=None, help="method1,method2 to focus delta maps")
    ap.add_argument("--pixel", type=str, default=None, help="lat,lon for nearest-pixel CONSENSUS time-series")

    ap.add_argument("--style", type=str, default="seaborn-v0_8-whitegrid")
    ap.add_argument("--fig-dpi", type=int, default=180)

    ap.add_argument("--outline-shp", type=str, default=DEFAULT_OUTLINE_SHP,
                    help="EPSG:4326 shapefile/geojson/zip for outline")

    ap.add_argument("--map-kind", type=str, default="grid", choices=["grid", "tri", "scatter"])
    ap.add_argument("--article-thr", type=str, default="5,10", help="Thresholds for |Δ|≤T tables")
    ap.add_argument("--delta-flag", type=float, default=10.0, help="Binary disagreement maps for |Δ|≥value")

    ap.add_argument("--permethod-maps", action="store_true",
                    help="Also write per-method maps into 03_maps/permethod/<method>/...")

    ap.add_argument("--compare-only", action="store_true", help="Only Phase A (compare).")
    ap.add_argument("--graphs-only", action="store_true", help="Assume Phase A done; only graphs/maps/excel.")

    ap.add_argument("--no-excel-compact", action="store_true", help="Do NOT write compact win{W}_summary.xlsx")
    ap.add_argument("--excel-full", action="store_true", help="Write FULL report workbook (can be large).")
    ap.add_argument("--embed-images", action="store_true", help="Embed PNGs in FULL report (only with --excel-full).")

    ap.add_argument("--do-pixel-summary", action="store_true",
                    help="Aggregate 6B diffs vs hourly_sum into per-pixel ALL-YEARS summaries.")
    ap.add_argument("--in-diffs", type=str, default=None,
                    help="6B diffs folder (supports {win}). Default: <output-root>/win{W}/6B/diffs.")
    ap.add_argument("--stab-med-abs-le", type=float, default=1.0)
    ap.add_argument("--stab-p90-le", type=float, default=3.0)

    ap.add_argument("--combined", action="store_true",
                    help="Write combined ALL-windows CSV bundle + one Excel workbook (creates 00_combined).")
    ap.add_argument("--combined-dir", type=str, default=None,
                    help="Where to write combined outputs (default: <output-root>/00_combined)")

    ap.add_argument("--prune-per-year", action="store_true",
                    help="Delete 01_tables/per_year/*.csv at end (keeps final tables/plots/maps).")

    args = ap.parse_args()

    if args.root:
        os.environ["DDU_ROOT"] = args.root
        _CFG_ROOT = Path(args.root).expanduser().resolve()
        DEFAULT_METHODS_CSV_TEMPLATE = str(
            _CFG_ROOT / "out" / "08_season_swrap" / "win{win}" / "csv" / "season_ALL_YEARS_methods.csv"
        )
        if (cfg is None) and (args.output_root == _DEFAULT_OUTPUT_ROOT_ORIG):
            args.output_root = str(_CFG_ROOT / "out" / "12_method_compare")

    years = parse_years(args.years)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    variables = [v.strip() for v in args.variables.split(",") if v.strip()]
    bbox = parse_bbox(args.bbox) if args.bbox else None
    pair_focus = parse_pair(args.pair) if args.pair else None

    pixel = None
    if args.pixel:
        lat, lon = [float(x) for x in args.pixel.split(",")]
        pixel = (lat, lon)

    wins = [int(w) for w in args.windows.replace(" ", "").split(",") if w.strip()]
    processed_wins: List[int] = []

    for W in wins:
        in_csv = _auto_find_methods_csv(W, args.methods_csv)
        if in_csv is None or not in_csv.exists():
            print(f"[WARN] win{W}: season_ALL_YEARS_methods.csv not found — skipping.")
            continue

        processed_wins.append(W)

        out_root = Path(args.output_root) / f"win{W}" / "6.compare"
        _ensure_out_structure(out_root)
        thresholds_meta = _sibling_meta(in_csv)

        print(f"[INFO] Window={W}  in_csv={in_csv}")
        print(f"[INFO] Output dir = {out_root}")

        if not args.graphs_only:
            compare_phase(
                in_file=in_csv,
                out_root=out_root,
                years=years,
                methods_req=methods,
                variables_req=variables,
                bbox=bbox,
                window=W,
                thresholds_meta=thresholds_meta,
            )

        if not args.compare_only:
            graphs_phase(
                out_root=out_root,
                years=years,
                variables=variables,
                pair_focus=pair_focus,
                fig_dpi=args.fig_dpi,
                style=args.style,
                bbox=bbox,
                methods_csv=in_csv,
                pixel_latlon=pixel,
                outline_shp=args.outline_shp,
                map_kind=args.map_kind,
                article_thr=args.article_thr,
                delta_flag=args.delta_flag,
                permethod_maps=bool(args.permethod_maps),
            )

            if args.do_pixel_summary:
                diffs_dir = _resolve_diffs_dir(args.in_diffs, Path(args.output_root), W)
                pixel_summary_from_diffs(
                    diffs_dir=diffs_dir,
                    out_root=out_root,
                    years=years,
                    varlist=variables,
                    stab_med_abs_le=float(args.stab_med_abs_le),
                    stab_p90_le=float(args.stab_p90_le),
                )

            if not args.no_excel_compact:
                p = excel_compact_summary(out_root, W, variables)
                print(f"[Excel] Compact summary: {p}")

            if args.excel_full:
                try:
                    pfull = excel_full_report(
                        out_root=out_root,
                        window=W,
                        variables=variables,
                        embed_images=bool(args.embed_images),
                        methods_csv=in_csv,
                        bbox=bbox,
                    )
                    print(f"[Excel] Full report: {pfull}")
                except Exception as e:
                    print(f"[Excel] Full report skipped due to error: {e}")

        if args.prune_per_year:
            dirs = _ensure_out_structure(out_root)
            _prune_per_year(dirs["per_year"])

    if args.combined and processed_wins:
        comb_dir = Path(args.combined_dir) if args.combined_dir else (Path(args.output_root) / "00_combined")
        try:
            xlsx_path, csv_dir = export_combined_results(
                output_root_base=Path(args.output_root),
                windows=processed_wins,
                variables=variables,
                combined_dir=comb_dir,
            )
            print(f"[COMBINED] Excel: {xlsx_path}")
            print(f"[COMBINED] CSV dir: {csv_dir}")
        except Exception as e:
            print(f"[COMBINED] Skipped due to error: {e}")

    print("DONE.")


if __name__ == "__main__":
    main()
