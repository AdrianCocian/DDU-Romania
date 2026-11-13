#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
step07_suitability_classes.py — relative climatic suitability classes of season length (window 30 days)

Manuscript, section 2.5 and Table 2. The season length of every grid cell and year is assigned to one
of six classes. The class limits are the 25th, 50th, 75th and 90th percentiles of the season length of
the reference method (hourly sum), pooled over all grid cells and years with a season (length > 0):

  0 = no season (season length 0)
  1 = below P25               lowest
  2 = P25 to below P50        below median
  3 = P50 to below P75        above median
  4 = P75 to below P90        high
  5 = P90 or more             highest

Published run (2005–2024, 3-day season rule): P25/P50/P75/P90 = 81/108/124/135 days, i.e. classes
0, 1–80, 81–107, 108–123, 124–134 and >=135 days. The same limits are applied to every method and to the
consensus (per-cell median of the four methods), so that classes are comparable between methods and
years. The classes describe relative climatic suitability for extrinsic development within Romania;
they are not infection-risk categories.

INPUT
  <ROOT>/out/08_season_swrap/win30/csv/season_ALL_YEARS_methods.csv   (written by step05)

OUTPUT   <ROOT>/out/12_method_compare/win30/7_suitability_classes/   (config.STEP7_DIRNAME)
  00_meta/   class_breaks_META_7.json, README.md
  01_tables/ suitability_class_counts_ALL_win30.csv, suitability_summary_ALL_win30.csv
  02_plots/  class shares (stacked area), coverage and national median per method
  03_maps/<method>/map_{YEAR}_season_length_SUITABILITY_{method}_win30.png
  04_gif/    optional (--gif)
  99_excel/  optional (--excel)

Options worth knowing
  --breaks 81,108,124,135   use fixed class limits instead of computing them from the input
                            (e.g. to classify another period with the published limits)
  --no-maps                 tables and metadata only (fast)

Per-cell classes for the manuscript (all methods and the consensus) are written by step09
(00_MANUSCRIPT_PACK/per_cell_classes_ALL_YEARS.csv) with exactly the same rule.

Root resolution: --root > DDU_ROOT > config.ROOT > repository root. No personal paths.
"""

from __future__ import annotations

import os
import re
import json
import sys
import argparse
import warnings
from datetime import datetime, timezone
from pathlib import Path
from contextlib import contextmanager
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd

# headless plotting
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib
matplotlib.use("Agg", force=True)
warnings.filterwarnings("ignore", message=r'.*Creating legend with loc="best".*')
import matplotlib.pyplot as plt
plt.ioff()
import matplotlib.colors as mcolors
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
_this = Path(__file__).resolve()
CODE_ROOT = _find_repo_root(_this.parent.parent if _this.parent.name.lower() == "scripts" else _this.parent)
REPO_ROOT = Path(_root_cli).expanduser().resolve() if _root_cli else CODE_ROOT

if _root_cli:
    os.environ["DDU_ROOT"] = str(REPO_ROOT)          # --root wins over an inherited DDU_ROOT
else:
    os.environ.setdefault("DDU_ROOT", str(REPO_ROOT))
for _p in (str(REPO_ROOT), str(CODE_ROOT)):   # code root ends first on sys.path
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ddu_stats as ds  # noqa: E402  (repository root module)

try:
    from config import cfg  # type: ignore[attr-defined]
except Exception:
    cfg = None  # type: ignore


def _cfg(name: str, default):
    return getattr(cfg, name, default) if cfg is not None else default


# ---------------- constants (from config.py when available) ----------------
WIN = 30
NAN_COLOR = "#FFF5B1"
ROMANIA_BBOX_DEFAULT = "48.5,19.9,43.5,30.1"  # N,W,S,E

METHODS_ALL: List[str] = list(_cfg("METHODS", ["midpoint", "daily_mean", "hourly_sum", "be_sine"]))
METHOD_LABELS: Dict[str, str] = dict(_cfg("METHOD_LABELS", {}))
REF_METHOD: str = str(_cfg("REF_METHOD", "hourly_sum"))
CLASS_PERCENTILES: List[float] = [float(p) for p in _cfg("CLASS_PERCENTILES", list(ds.CLASS_PERCENTILES))]
CLASS_LABELS: List[str] = list(_cfg("CLASS_LABELS", ds.CLASS_LABELS))
CLASS_COLORS: List[str] = list(_cfg("CLASS_COLORS",
                                    ["#e6e6e6", "#4da3ff", "#66c2a5", "#a6d854", "#fdae61", "#d73027"]))
STEP7_DIRNAME: str = str(_cfg("STEP7_DIRNAME", "7_suitability_classes"))

CFG_ROOT = Path(_cfg("ROOT", os.environ.get("DDU_ROOT", str(REPO_ROOT)))).resolve()
DEFAULT_OUTLINE_SHP = str(_cfg("SHP_RO", "") or "")
if not DEFAULT_OUTLINE_SHP:
    _shipped = CFG_ROOT / "boundaries" / "ro_frontiera_poligon.shp"
    if _shipped.exists():
        DEFAULT_OUTLINE_SHP = str(_shipped)


def _lab(m: str) -> str:
    return METHOD_LABELS.get(m, m)


def _class_name(k: int) -> str:
    return CLASS_LABELS[k].split(" ", 1)[1] if " " in CLASS_LABELS[k] else CLASS_LABELS[k]


# ---------------- small helpers ----------------
@contextmanager
def mpl_style(style: Optional[str]):
    if not style:
        yield
        return
    try:
        with plt.style.context(style):
            yield
    except Exception:
        with plt.style.context("default"):
            yield


def parse_years(s: Optional[str]) -> Optional[List[int]]:
    if not s:
        return None
    s = s.replace("–", "-").replace("—", "-").strip()
    if "-" in s:
        a, b = s.split("-", 1)
        a, b = int(a), int(b)
        if a > b:
            a, b = b, a
        return list(range(a, b + 1))
    return [int(x) for x in s.split(",") if x.strip()]


def parse_bbox(s: Optional[str]) -> Optional[Tuple[float, float, float, float]]:
    if not s:
        return None
    vals = [float(x) for x in s.split(",")]
    if len(vals) != 4:
        raise ValueError("--bbox N,W,S,E")
    N, W, S, E = vals
    if not (S <= N and W <= E):
        raise ValueError("BBOX order N,W,S,E violated")
    return N, W, S, E


def parse_breaks(s: Optional[str]) -> Optional[np.ndarray]:
    if not s:
        return None
    vals = np.array([float(x) for x in s.split(",") if x.strip()], dtype=float)
    if vals.size != 4 or np.any(np.diff(vals) <= 0) or vals[0] <= 0:
        raise ValueError("--breaks needs four increasing positive numbers, e.g. 81,108,124,135")
    return vals


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def excel_engine() -> str:
    try:
        import xlsxwriter  # noqa: F401
        return "xlsxwriter"
    except Exception:
        return "openpyxl"


def sanitize_sheet_name(name: str) -> str:
    return re.sub(r'[:\\/*?\[\]]', "_", name)[:31]


def detect_root(user_root: Optional[str] = None) -> Path:
    """--root > DDU_ROOT > config.ROOT > repository root."""
    if user_root:
        return Path(user_root).expanduser().resolve()
    env_root = os.environ.get("DDU_ROOT", "").strip()
    if env_root:
        return Path(env_root).expanduser().resolve()
    if cfg is not None:
        try:
            return Path(getattr(cfg, "ROOT")).resolve()
        except Exception:
            pass
    return REPO_ROOT


def find_season_csv(root: Path) -> Optional[Path]:
    cands = [
        root / f"out/08_season_swrap/win{WIN}/csv/season_ALL_YEARS_methods.csv",
        root / f"out/11_ddu_season_swrap/win{WIN}/csv/season_ALL_YEARS_methods.csv",
    ]
    for p in cands:
        if p.exists():
            return p
    try:
        for p in sorted((root / "out").rglob("season_ALL_YEARS_methods.csv")):
            if f"win{WIN}/csv/" in p.as_posix():
                return p
    except Exception:
        pass
    return None


# ---------------- input ----------------
def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    ren = {"latitude": "lat", "longitude": "lon", "Year": "year"}
    return df.rename(columns={k: v for k, v in ren.items() if k in df.columns and v not in df.columns})


def filter_roi(df: pd.DataFrame, bbox: Optional[Tuple[float, float, float, float]]) -> pd.DataFrame:
    if df.empty:
        return df
    for col in ("inside_mask", "inside", "roi", "in_ro"):
        if col in df.columns:
            df = df[df[col].astype(str).str.lower().isin(["1", "true", "yes"])].copy()
            break
    if bbox and {"lat", "lon"}.issubset(df.columns):
        N, W, S, E = bbox
        df = df[(df["lat"] <= N) & (df["lat"] >= S) & (df["lon"] >= W) & (df["lon"] <= E)].copy()
    return df


def read_season_length_wide(season_csv: Path, bbox, methods_sel: List[str], use_consensus: bool) -> pd.DataFrame:
    """One row per (year, i, j, lat, lon); one season-length column per method (+ 'consensus').

    A missing or empty season length is 0 (no season), exactly as in step09. The consensus is the
    per-cell median of the selected methods (all four in the manuscript)."""
    head = list(pd.read_csv(season_csv, nrows=0).columns)
    want = ["year", "window_days", "method", "i", "j", "lat", "lon", "latitude", "longitude",
            "season_length", "inside_mask", "inside", "roi", "in_ro"]
    df = pd.read_csv(season_csv, usecols=[c for c in want if c in head], low_memory=False)
    df = _normalize_columns(df)
    missing = {"year", "method", "i", "j", "lat", "lon", "season_length"}.difference(df.columns)
    if missing:
        raise ValueError(f"{season_csv.name} is missing columns: {sorted(missing)}")
    if "window_days" in df.columns:
        df = df[pd.to_numeric(df["window_days"], errors="coerce") == WIN]
    for c in ("year", "i", "j", "lat", "lon", "season_length"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["year", "i", "j", "lat", "lon"])
    df["year"] = df["year"].astype(int)
    df["method"] = df["method"].astype(str)
    df = filter_roi(df, bbox)

    present = [m for m in methods_sel if m in set(df["method"].unique())]
    if not present:
        raise SystemExit(f"none of the methods {methods_sel} is present in {season_csv}")
    w = (df[df["method"].isin(present)]
         .set_index(["year", "i", "j", "lat", "lon", "method"])["season_length"]
         .unstack("method").reindex(columns=present).fillna(0.0))
    if use_consensus and len(present) >= 2:
        w["consensus"] = w[present].median(axis=1)
    return w.reset_index().sort_values(["year", "i", "j"]).reset_index(drop=True)


# ---------------- maps ----------------
def _set_year_ticks(ax, years: List[int]) -> None:
    if not years:
        return
    y0, y1 = min(years), max(years)
    ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(list(range(y0, y1 + 1))))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%d"))
    ax.set_xlim(y0 - 0.5, y1 + 0.5)
    ax.tick_params(axis="x", labelsize=8)


def grid_pcolormesh(ax, df: pd.DataFrame, vcol: str, cmap, norm):
    d = df[["i", "j", "lat", "lon", vcol]].dropna(subset=["i", "j", "lat", "lon"])
    if d.empty:
        return None
    lat_by_i = d.groupby("i")["lat"].median().sort_index()
    lon_by_j = d.groupby("j")["lon"].median().sort_index()
    Zp = d.pivot_table(index="i", columns="j", values=vcol, aggfunc="mean")
    Zp = Zp.reindex(index=lat_by_i.index, columns=lon_by_j.index)
    Z = np.ma.masked_invalid(Zp.values.astype(float))
    LAT2D = np.repeat(lat_by_i.values[:, None], len(lon_by_j), axis=1)
    LON2D = np.repeat(lon_by_j.values[None, :], len(lat_by_i), axis=0)
    return ax.pcolormesh(LON2D, LAT2D, Z, cmap=cmap, norm=norm, shading="nearest")


def draw_outline(ax, shp_path: str):
    if not shp_path:
        return
    try:
        import geopandas as gpd  # type: ignore
        gpd.read_file(shp_path).to_crs(epsg=4326).boundary.plot(ax=ax, linewidth=0.7, color="k", zorder=10)
    except Exception as e:
        print(f"[map] outline skipped: {e}")


def files_index(base_dir: Path) -> pd.DataFrame:
    rows = []
    for p in base_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in (".csv", ".png", ".json", ".gif", ".xlsx", ".md"):
            rows.append({"file": p.relative_to(base_dir).as_posix(),
                         "size_MB": round(p.stat().st_size / 1024 / 1024, 3)})
    return pd.DataFrame(rows).sort_values("file") if rows else pd.DataFrame(columns=["file", "size_MB"])


# ---------------- main work ----------------
def process(
    project_root: Path,
    season_csv: Path,
    years: List[int],
    methods_sel: List[str],
    use_consensus: bool,
    bbox: Optional[Tuple[float, float, float, float]],
    breaks_cli: Optional[np.ndarray],
    outline_shp: str,
    dpi: int,
    style: Optional[str],
    map_kind: str,
    make_maps: bool,
    make_gif: bool,
    do_excel: bool,
    embed_images: bool,
    font_scale: float,
) -> Path:
    out_base = project_root / "out" / "12_method_compare" / f"win{WIN}" / STEP7_DIRNAME
    d00 = ensure_dir(out_base / "00_meta")
    d01 = ensure_dir(out_base / "01_tables")
    d02 = ensure_dir(out_base / "02_plots")
    d03 = out_base / "03_maps"
    d04 = out_base / "04_gif"
    d99 = out_base / "99_excel"

    wide = read_season_length_wide(season_csv, bbox, methods_sel, use_consensus)
    methods_here = [c for c in METHODS_ALL + ["consensus"] if c in wide.columns]
    years_in_file = sorted(wide["year"].unique().tolist())

    # ---- class limits: P25/P50/P75/P90 of the reference method, all cells and years in the input ----
    ref_vals = wide[REF_METHOD].to_numpy(dtype=float) if REF_METHOD in wide.columns else None
    if breaks_cli is not None:
        breaks = breaks_cli
        breaks_source = "fixed (--breaks)"
    else:
        if ref_vals is None:
            raise SystemExit(f"reference method '{REF_METHOD}' not in the input: give --breaks or include it in --methods")
        breaks = ds.class_breaks(ref_vals, CLASS_PERCENTILES)
        breaks_source = f"percentiles of {REF_METHOD} season length > 0, years {years_in_file[0]}-{years_in_file[-1]}"
    ranges = ds.class_ranges(breaks)
    print(f"[CLASSES] limits (days) = {[float(b) for b in breaks]}  ->  {ranges}  ({breaks_source})")

    meta = {
        "window_days": WIN,
        "variable": "season_length",
        "input_mode": "season_csv_only",
        "season_csv_used": str(season_csv),
        "class_breaks_reference_method": REF_METHOD,
        "class_breaks_source": breaks_source,
        "class_breaks_years": [int(years_in_file[0]), int(years_in_file[-1])],
        "class_breaks_percentiles_P25_P50_P75_P90": [float(b) for b in breaks],
        "class_breaks_days": ranges,
        "class_breaks_rule": "0 = no season; 1: < P25; 2: [P25,P50); 3: [P50,P75); 4: [P75,P90); 5: >= P90 "
                             "(percentiles of season_length > 0, all years and grid cells)",
        "class_labels": CLASS_LABELS,
        "colors_0to5": CLASS_COLORS,
        "nan_color": NAN_COLOR,
        "bbox": list(bbox) if bbox else None,
        "outline_shp": outline_shp,
        "map_kind": map_kind,
        "methods_selected": methods_sel,
        "consensus_enabled": "consensus" in wide.columns,
        "output_dir": str(out_base),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if ref_vals is not None:
        n_tot = int(np.isfinite(ref_vals).sum())
        n_season = int(np.sum(ref_vals > 0))
        meta.update({
            "n_cell_years_reference_with_season": n_season,
            "n_cell_years_reference_total": n_tot,
            "pct_cell_years_reference_no_season": round(100.0 * (1 - n_season / n_tot), 2) if n_tot else None,
        })
    (d00 / "class_breaks_META_7.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- per year x method: counts, shares, maps ----
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
    cmap = mcolors.ListedColormap(CLASS_COLORS)
    cmap.set_bad(NAN_COLOR)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    plt.rcParams.update({"axes.titlesize": 12 * font_scale, "axes.labelsize": 11 * font_scale,
                         "xtick.labelsize": 9 * font_scale, "ytick.labelsize": 9 * font_scale,
                         "legend.fontsize": 9 * font_scale})

    class_rows: List[Dict] = []
    summary_rows: List[Dict] = []
    gif_frames: Dict[str, List[Path]] = {}
    years_run = [y for y in years if y in set(years_in_file)]
    for y in sorted(set(years) - set(years_run)):
        print(f"[WARN] year={y}: not in {season_csv.name} -> skipped")

    for year in years_run:
        g = wide[wide["year"] == year]
        for method in methods_here:
            vals = g[method].to_numpy(dtype=float)
            cls = ds.classify(vals, breaks)
            total_valid = int(np.isfinite(cls).sum())
            nonzero = int(np.sum(vals > 0))
            nonzero_pct = round(100.0 * nonzero / total_valid, 2) if total_valid else np.nan
            med = float(np.nanmedian(vals)) if total_valid else np.nan
            summary_rows.append({"year": int(year), "method": method, "total_valid": total_valid,
                                 "nonzero_count": nonzero, "nonzero_pct": nonzero_pct,
                                 "median_season_length": med, "window": WIN})
            for k in range(6):
                n = int(np.sum(cls == k))
                class_rows.append({"year": int(year), "method": method, "class": k, "label": CLASS_LABELS[k],
                                   "days": ranges[k], "count": n,
                                   "pct": round(100.0 * n / total_valid, 2) if total_valid else np.nan,
                                   "total_valid": total_valid, "nonzero_count": nonzero, "nonzero_pct": nonzero_pct,
                                   "median_season_length": med, "window": WIN})

            if not make_maps:
                continue
            dmap = g[["i", "j", "lat", "lon"]].copy()
            dmap["suitability_class"] = cls
            mdir = ensure_dir(d03 / method)
            with mpl_style(style):
                fig, ax = plt.subplots(figsize=(9.5, 5.0))
                ax.set_title(f"Relative climatic suitability classes — season length — {_lab(method)} — {year}",
                             fontsize=11 * font_scale)
                ax.set_xlabel("Longitude")
                ax.set_ylabel("Latitude")
                if map_kind == "grid":
                    mapp = grid_pcolormesh(ax, dmap, "suitability_class", cmap, norm)
                else:
                    pts = dmap.dropna(subset=["suitability_class"])
                    mapp = ax.scatter(pts["lon"], pts["lat"], c=pts["suitability_class"], s=9,
                                      cmap=cmap, norm=norm, edgecolor="none")
                draw_outline(ax, outline_shp)
                if bbox:
                    N, Wb, S, E = bbox
                    ax.set_xlim(Wb, E)
                    ax.set_ylim(S, N)
                ax.set_aspect("equal", adjustable="box")
                if mapp is not None:
                    cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.05)
                    cb = fig.colorbar(mapp, cax=cax, ticks=[0, 1, 2, 3, 4, 5])
                    cb.set_label("Suitability class", rotation=270, labelpad=14)
                    cb.ax.set_yticklabels([f"{k}: {_class_name(k)} ({ranges[k]} d)" for k in range(6)])
                fig.tight_layout()
                out_png = mdir / f"map_{year}_season_length_SUITABILITY_{method}_win{WIN}.png"
                fig.savefig(out_png, dpi=dpi)
                plt.close(fig)
            gif_frames.setdefault(method, []).append(out_png)

    if not class_rows:
        print("[WARN] no output: check --years / --methods / --bbox")
        return out_base

    df_counts = pd.DataFrame(class_rows).sort_values(["method", "year", "class"])
    df_summary = pd.DataFrame(summary_rows).sort_values(["method", "year"])
    df_counts.to_csv(d01 / f"suitability_class_counts_ALL_win{WIN}.csv", index=False)
    df_summary.to_csv(d01 / f"suitability_summary_ALL_win{WIN}.csv", index=False)

    # ---- plots ----
    yrs_all = sorted(df_summary["year"].astype(int).unique().tolist())
    for method in methods_here:
        d = df_counts[df_counts["method"] == method]
        if d.empty:
            continue
        dfw = d.pivot_table(index="year", columns="class", values="pct", aggfunc="mean").sort_index()
        dfw = dfw.reindex(columns=list(range(6))).fillna(0.0)
        with mpl_style(style):
            fig, ax = plt.subplots(figsize=(12, 5.2))
            ax.stackplot(dfw.index.values, [dfw[c].values for c in dfw.columns],
                         labels=[f"{k}: {_class_name(k)}" for k in range(6)], colors=CLASS_COLORS, alpha=0.95)
            ax.set_title(f"Suitability classes — share of grid cells (%) — {_lab(method)}")
            ax.set_xlabel("Year")
            ax.set_ylabel("Share of grid cells (%)")
            ax.set_ylim(0, 100)
            _set_year_ticks(ax, dfw.index.astype(int).tolist())
            ax.legend(title="Class", ncols=3, loc="upper center", bbox_to_anchor=(0.5, -0.14), frameon=False)
            fig.tight_layout()
            fig.savefig(d02 / f"stacked_suitability_classes_{method}_win{WIN}.png", dpi=dpi)
            plt.close(fig)

    for col, ylab, fname, ylim in (
        ("nonzero_pct", "Grid cells with a season (%)", "COMPARE_nonzero_coverage", (0, 100)),
        ("median_season_length", "National median season length (days)", "COMPARE_median_season_length", None),
    ):
        with mpl_style(style):
            fig, ax = plt.subplots(figsize=(11, 5.0))
            for method in methods_here:
                d2 = df_summary[df_summary["method"] == method].sort_values("year")
                ax.plot(d2["year"], d2[col], lw=2, marker="o", label=_lab(method),
                        color=dict(_cfg("METHOD_COLORS", {})).get(method))
            ax.set_xlabel("Year")
            ax.set_ylabel(ylab)
            if ylim:
                ax.set_ylim(*ylim)
            _set_year_ticks(ax, yrs_all)
            ax.legend()
            fig.tight_layout()
            fig.savefig(d02 / f"{fname}_win{WIN}.png", dpi=dpi)
            plt.close(fig)

    if make_gif and gif_frames:
        try:
            try:
                import imageio.v2 as imageio
            except Exception:
                import imageio  # type: ignore
            ensure_dir(d04)
            for method, frames in gif_frames.items():
                out_gif = d04 / f"suitability_classes_{method}_win{WIN}.gif"
                imageio.mimsave(out_gif, [imageio.imread(str(p)) for p in frames], duration=0.6)
                print(f"[GIF] {out_gif}")
        except Exception as e:
            print(f"[GIF] skipped: {e}")

    if do_excel:
        ensure_dir(d99)
        xlsx_path = d99 / f"win{WIN}_{STEP7_DIRNAME}.xlsx"
        engine = excel_engine()
        if embed_images and engine != "xlsxwriter":
            print("[EXCEL] xlsxwriter not available -> --embed-images disabled")
            embed_images = False
        try:
            with pd.ExcelWriter(xlsx_path, engine=engine) as xlw:
                pd.DataFrame([{"Section": "Scope",
                               "Details": f"Relative climatic suitability classes of season length, window {WIN} d; "
                                          f"limits {ranges} days ({breaks_source})."}]
                             ).to_excel(xlw, sheet_name="README", index=False)
                pd.json_normalize(meta).to_excel(xlw, sheet_name="META_", index=False)
                files_index(out_base).to_excel(xlw, sheet_name="DATA_INDEX", index=False)
                df_counts.to_excel(xlw, sheet_name="COUNTS_ALL", index=False)
                df_summary.to_excel(xlw, sheet_name="SUMMARY_ALL", index=False)
                for method in methods_here:
                    d = df_counts[df_counts["method"] == method]
                    dfw = d.pivot_table(index="year", columns="class", values="pct", aggfunc="mean").sort_index()
                    dfw.reindex(columns=list(range(6))).reset_index().to_excel(
                        xlw, sheet_name=sanitize_sheet_name(f"PCT_{method}"), index=False)
                if embed_images and engine == "xlsxwriter":
                    ws = xlw.book.add_worksheet("IMG_plots")
                    for r, p in enumerate(sorted(d02.glob("*.png"))[:18]):
                        ws.insert_image(r * 28, 0, str(p))
            print(f"[EXCEL] {xlsx_path}")
        except Exception as e:
            print(f"[EXCEL] skipped: {e}")

    readme = [
        f"# Relative climatic suitability classes (season length, window {WIN} days)",
        "",
        f"Input: {season_csv}",
        "",
        "## Class rule (manuscript, Table 2)",
        "Limits = P25/P50/P75/P90 of the reference (hourly-sum) season length among all cell-years with a",
        "season; the same limits are applied to every method and to the consensus.",
        "",
        f"Limits used: {[float(b) for b in breaks]} ({breaks_source})",
        "",
    ] + [f"- {CLASS_LABELS[k]}: {ranges[k]} days" for k in range(6)] + [
        "",
        "## Tables",
        f"- 01_tables/suitability_class_counts_ALL_win{WIN}.csv (count and % of grid cells per class, year, method)",
        f"- 01_tables/suitability_summary_ALL_win{WIN}.csv (cells with a season, national median)",
        "",
        "## Maps",
        f"- colours 0-5: {CLASS_COLORS}; no data: {NAN_COLOR}; square pixels (pcolormesh, shading='nearest')",
        "",
    ]
    (d00 / "README.md").write_text("\n".join(readme), encoding="utf-8")
    print(f"[DONE] {out_base}")
    return out_base


def main(argv: Optional[List[str]] = None):
    ap = argparse.ArgumentParser(description="step07 - relative climatic suitability classes of season length (win30)")
    ap.add_argument("--root", type=str, default="", help="Project root (parent of out/). Default: DDU_ROOT / repository root")
    ap.add_argument("--season-csv", type=str, default="", help="Explicit path to season_ALL_YEARS_methods.csv")
    ap.add_argument("--years", type=str, default="2005-2024", help="Years to tabulate and map (default 2005-2024)")
    ap.add_argument("--methods", type=str, default=",".join(METHODS_ALL), help="Default: all four methods")
    ap.add_argument("--no-consensus", action="store_true", help="Do not add the consensus (median of the methods)")
    ap.add_argument("--consensus", action="store_true", help=argparse.SUPPRESS)  # kept for old command lines
    ap.add_argument("--breaks", type=str, default="", help="Fixed limits P25,P50,P75,P90 in days, e.g. 81,108,124,135")
    ap.add_argument("--bbox", type=str, default=ROMANIA_BBOX_DEFAULT, help="N,W,S,E. Default: Romania bbox")
    ap.add_argument("--outline-shp", type=str, default=DEFAULT_OUTLINE_SHP, help="Outline shapefile (EPSG:4326)")
    ap.add_argument("--style", type=str, default="seaborn-v0_8-whitegrid")
    ap.add_argument("--map-kind", type=str, default="grid", choices=["grid", "scatter"])
    ap.add_argument("--no-maps", action="store_true", help="Tables and metadata only")
    ap.add_argument("--gif", action="store_true", help="GIF animation per method")
    ap.add_argument("--excel", action="store_true", help="Excel workbook")
    ap.add_argument("--embed-images", action="store_true", help="Embed plots in the Excel workbook (xlsxwriter)")
    ap.add_argument("--fig-dpi", type=int, default=180)
    ap.add_argument("--font-scale", type=float, default=0.9)
    args = ap.parse_args(argv)

    project_root = detect_root(args.root or None)
    season_csv = Path(args.season_csv).expanduser().resolve() if args.season_csv else find_season_csv(project_root)
    if season_csv is None or not season_csv.exists():
        raise SystemExit(f"season_ALL_YEARS_methods.csv not found under {project_root / 'out'} (run step05 or give --season-csv)")
    years = parse_years(args.years) or list(range(2005, 2025))
    methods_sel = [m.strip() for m in args.methods.split(",") if m.strip()] or METHODS_ALL.copy()

    print(f"[INFO] ROOT={project_root}")
    print(f"[INFO] season_csv={season_csv}")
    print(f"[INFO] years={years[0]}..{years[-1]} | methods={methods_sel} | consensus={not args.no_consensus}")

    process(
        project_root=project_root,
        season_csv=season_csv,
        years=years,
        methods_sel=methods_sel,
        use_consensus=not args.no_consensus,
        bbox=parse_bbox(args.bbox) if args.bbox else None,
        breaks_cli=parse_breaks(args.breaks),
        outline_shp=args.outline_shp or "",
        dpi=args.fig_dpi,
        style=args.style,
        map_kind=args.map_kind,
        make_maps=not args.no_maps,
        make_gif=args.gif,
        do_excel=args.excel,
        embed_images=args.embed_images,
        font_scale=args.font_scale,
    )


if __name__ == "__main__":
    main()
