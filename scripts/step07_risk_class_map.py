#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
step07_risk_class_map_win30.py — FINAL (clean) — DDU Risk Classes (season_length) — WIN30 ONLY

Continuare corectă după:
  step05_ddu_season_wrap.py  (win30) -> season_ALL_YEARS_methods.csv (long format, per-pixel per-method)

INPUT (robust):
  A) Dacă există grile exportate (format vechi):
     <ROOT>/out/12_method_compare/win30/per_pixel/grids/{YEAR}_VALUES_season_length.csv
     (sau variante în win30/6A/per_pixel/grids ori win30/6/per_pixel/grids)
  B) Altfel, citește direct:
     <ROOT>/out/08_season_swrap/win30/csv/season_ALL_YEARS_methods.csv

Risk classes FIXE:
  0 = 0 zile (no risk)
  1 = 1–76
  2 = 77–107
  3 = 108–123
  4 = 124–134
  5 = ≥135

OUTPUT:
  <ROOT>/out/12_method_compare/win30/7.risk_classes/
    00_meta/ META_7.json + README.md
    01_tables/ risk_class_counts_ALL_win30.csv + risk_summary_ALL_win30.csv
    02_plots/  *.png
    03_maps/<method>/ map_{YEAR}_season_length_RISKCLASS_{method}_win30.png
    04_gif/ (optional, --gif)
    99_excel/ (optional, --excel)

Map rules:
  - Discrete classes 0..5, low=blue high=red
  - NoData = pale yellow (#FFF8B5)
  - Square pixels: pcolormesh(..., shading="nearest"), aspect='equal'

Repo portability (GitHub-ready)
- Deterministic root resolution: --root > DDU_ROOT > cfg.ROOT > repo root
- No hardcoded personal paths.
"""

from __future__ import annotations

import os
import re
import json
import sys
import argparse
import warnings
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


# ---------------- constants ----------------
NAN_COLOR_DEFAULT = "#FFF8B5"
WIN = 30
DEF_METHODS_ALL = ["midpoint", "daily_mean", "hourly_sum", "be_sine"]
ROMANIA_BBOX_DEFAULT = "48.5,19.9,43.5,30.1"  # N,W,S,E

DEF_LABELS = [
    "0 no risk",
    "1 very low",
    "2 low",
    "3 moderate",
    "4 high",
    "5 very high",
]


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


# ---------------- optional config ----------------
try:
    from config import cfg  # type: ignore[attr-defined]
    CFG_ROOT = Path(getattr(cfg, "ROOT", os.environ.get("DDU_ROOT", str(REPO_ROOT)))).resolve()
    DEFAULT_OUTLINE_SHP = str(getattr(cfg, "SHP_RO", "")) if getattr(cfg, "SHP_RO", None) else ""
    NAN_COLOR = getattr(cfg, "NODATA_COLOR_HEX", NAN_COLOR_DEFAULT)
except Exception:
    cfg = None  # type: ignore
    CFG_ROOT = Path(os.environ.get("DDU_ROOT", str(REPO_ROOT))).resolve()
    DEFAULT_OUTLINE_SHP = ""
    NAN_COLOR = NAN_COLOR_DEFAULT

# fallback to repo-shipped boundary if outline not set
if not DEFAULT_OUTLINE_SHP:
    shipped = CFG_ROOT / "boundaries" / "ro_frontiera_poligon.shp"
    if shipped.exists():
        DEFAULT_OUTLINE_SHP = str(shipped)


# ---------------- plotting helpers ----------------
def _hex(c) -> str:
    return matplotlib.colors.to_hex(c, keep_alpha=False)


def build_colors() -> List[str]:
    # 0 = grey (no risk), 1..5 from blue -> red
    cm = plt.get_cmap("RdBu_r")  # low blue, high red
    samples = np.linspace(0.12, 0.88, 5)
    ramp = [_hex(cm(x)) for x in samples]
    return ["#E6E6E6"] + ramp


DEF_COLORS = build_colors()


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
    bad = r'[:\\/*?\[\]]'
    name = re.sub(bad, "_", name)
    return name[:31]


def apply_nan_cmap(cmap, nan=NAN_COLOR):
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    try:
        cmap = cmap.copy()
    except Exception:
        pass
    try:
        cmap.set_bad(nan)
    except Exception:
        pass
    return cmap


# ---------------- deterministic ROOT resolution ----------------
def detect_root(user_root: Optional[str] = None) -> Path:
    """
    Deterministic project ROOT resolver:
      1) --root
      2) env DDU_ROOT
      3) cfg.ROOT
      4) repo root
    """
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


def find_grids_csv(root: Path, year: int) -> Optional[Path]:
    cands = [
        root / f"out/12_method_compare/win{WIN}/per_pixel/grids/{year}_VALUES_season_length.csv",
        root / f"out/12_method_compare/win{WIN}/6A/per_pixel/grids/{year}_VALUES_season_length.csv",
        root / f"out/12_method_compare/win{WIN}/6/per_pixel/grids/{year}_VALUES_season_length.csv",
    ]
    for p in cands:
        if p.exists():
            return p
    return None


def find_season_csv(root: Path, season_csv_cli: Optional[str] = None) -> Optional[Path]:
    if season_csv_cli:
        p = Path(season_csv_cli).expanduser()
        if p.exists():
            return p.resolve()

    cands = [
        root / f"out/08_season_swrap/win{WIN}/csv/season_ALL_YEARS_methods.csv",
        root / f"out/11_ddu_season_swrap/win{WIN}/csv/season_ALL_YEARS_methods.csv",
    ]
    for p in cands:
        if p.exists():
            return p

    try:
        for p in root.rglob("season_ALL_YEARS_methods.csv"):
            s = str(p).replace("\\", "/")
            if f"win{WIN}/csv/season_ALL_YEARS_methods.csv" in s:
                return p
    except Exception:
        pass

    return None


# ---------------- normalization / ROI ----------------
def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    if "latitude" in df.columns and "lat" not in df.columns:
        df = df.rename(columns={"latitude": "lat"})
    if "longitude" in df.columns and "lon" not in df.columns:
        df = df.rename(columns={"longitude": "lon"})
    if "Year" in df.columns and "year" not in df.columns:
        df = df.rename(columns={"Year": "year"})
    return df


def filter_roi(df: pd.DataFrame, bbox: Optional[Tuple[float, float, float, float]]) -> pd.DataFrame:
    if df.empty:
        return df
    for col in ("inside_mask", "inside", "roi", "in_ro"):
        if col in df.columns:
            m = df[col].astype(str).str.lower().isin(["1", "true", "yes"])
            df = df[m].copy()
            break
    if bbox and {"lat", "lon"}.issubset(df.columns):
        N, W, S, E = bbox
        df = df[(df["lat"] <= N) & (df["lat"] >= S) & (df["lon"] >= W) & (df["lon"] <= E)].copy()
    return df


# ---------------- risk class logic ----------------
def classify_fixed(values: np.ndarray) -> np.ndarray:
    v = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy()
    cls = np.full(v.shape, np.nan)
    cls[(v == 0)] = 0
    cls[(v >= 1) & (v <= 76)] = 1
    cls[(v >= 77) & (v <= 107)] = 2
    cls[(v >= 108) & (v <= 123)] = 3
    cls[(v >= 124) & (v <= 134)] = 4
    cls[(v >= 135)] = 5
    return cls


def _set_year_ticks(ax, years: List[int]) -> None:
    if not years:
        return
    y0, y1 = min(years), max(years)
    ticks = list(range(y0, y1 + 1))
    ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(ticks))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%d"))
    ax.set_xlim(y0 - 0.5, y1 + 0.5)
    ax.tick_params(axis="x", labelsize=8)


# ---------------- map helpers ----------------
def grid_pcolormesh(ax, df: pd.DataFrame, vcol: str, cmap, norm, bbox=None):
    need = ["i", "j", "lat", "lon", vcol]
    if any(c not in df.columns for c in need):
        return None

    d = df[need].copy()
    for c in need:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["i", "j", "lat", "lon"])
    if d.empty:
        return None

    lat_by_i = d.groupby("i")["lat"].median().sort_index()
    lon_by_j = d.groupby("j")["lon"].median().sort_index()
    Zp = d.pivot_table(index="i", columns="j", values=vcol, aggfunc="mean")
    Zp = Zp.reindex(index=lat_by_i.index, columns=lon_by_j.index)

    Z = np.ma.masked_invalid(Zp.values.astype(float))
    LAT2D = np.repeat(lat_by_i.values[:, None], len(lon_by_j), axis=1)
    LON2D = np.repeat(lon_by_j.values[None, :], len(lat_by_i), axis=0)

    m = ax.pcolormesh(LON2D, LAT2D, Z, cmap=cmap, norm=norm, shading="nearest")
    if bbox:
        N, Wb, S, E = bbox
        ax.set_xlim(Wb, E)
        ax.set_ylim(S, N)
    return m


def draw_outline(ax, shp_path: str):
    if not shp_path:
        return
    try:
        import geopandas as gpd  # type: ignore
        gdf = gpd.read_file(shp_path).to_crs(epsg=4326)
        gdf.boundary.plot(ax=ax, linewidth=0.7, color="k", zorder=10)
    except Exception as e:
        print(f"[map] Outline skipped: {e}")


def files_index(base_dir: Path) -> pd.DataFrame:
    rows = []
    for p in base_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in (".csv", ".png", ".json", ".gif", ".xlsx", ".md"):
            try:
                sz = p.stat().st_size / 1024 / 1024
            except Exception:
                sz = np.nan
            rows.append({
                "file": p.relative_to(base_dir).as_posix(),
                "size_MB": round(float(sz), 3) if np.isfinite(sz) else np.nan
            })
    return pd.DataFrame(rows).sort_values("file") if rows else pd.DataFrame(columns=["file", "size_MB"])


def read_season_methods_csv(season_csv: Path) -> pd.DataFrame:
    header = pd.read_csv(season_csv, nrows=0)
    cols = list(header.columns)

    want = [
        "year", "window_days", "method",
        "i", "j", "lat", "lon",
        "season_length",
        "inside_mask", "inside", "roi", "in_ro",
    ]
    usecols = [c for c in want if c in cols]
    if not usecols:
        df = pd.read_csv(season_csv, low_memory=False)
    else:
        df = pd.read_csv(season_csv, usecols=usecols, low_memory=False)

    df = _normalize_columns(df)

    required = {"year", "method", "i", "j", "lat", "lon", "season_length"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"season_ALL_YEARS_methods.csv missing columns: {sorted(missing)}")

    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["season_length"] = pd.to_numeric(df["season_length"], errors="coerce")
    for c in ("i", "j", "lat", "lon"):
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if "window_days" in df.columns:
        df["window_days"] = pd.to_numeric(df["window_days"], errors="coerce")
        df = df[df["window_days"] == WIN].copy()

    df = df.dropna(subset=["year", "method", "i", "j", "lat", "lon"])
    return df


def process_win30(
    project_root: Path,
    years: List[int],
    methods_sel: List[str],
    use_consensus: bool,
    bbox: Optional[Tuple[float, float, float, float]],
    outline_shp: str,
    dpi: int,
    style: Optional[str],
    map_kind: str,
    make_gif: bool,
    do_excel: bool,
    embed_images: bool,
    font_scale: float,
    season_csv: Optional[Path],
):
    out_base = project_root / f"out/12_method_compare/win{WIN}/7.risk_classes"
    d00 = ensure_dir(out_base / "00_meta")
    d01 = ensure_dir(out_base / "01_tables")
    d02 = ensure_dir(out_base / "02_plots")
    d03 = ensure_dir(out_base / "03_maps")
    d04 = ensure_dir(out_base / "04_gif") if make_gif else (out_base / "04_gif")
    d99 = ensure_dir(out_base / "99_excel")

    plt.rcParams.update({
        "axes.titlesize": 12 * font_scale,
        "axes.labelsize": 11 * font_scale,
        "xtick.labelsize": 9 * font_scale,
        "ytick.labelsize": 9 * font_scale,
        "legend.fontsize": 9 * font_scale,
    })

    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
    cmap = matplotlib.colors.ListedColormap(DEF_COLORS)
    cmap = apply_nan_cmap(cmap, nan=NAN_COLOR)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    season_df = None
    if season_csv and season_csv.exists():
        season_df = read_season_methods_csv(season_csv)
        season_df = filter_roi(season_df, bbox)

    meta = {
        "window_days": WIN,
        "variable": "season_length",
        "input_mode": "grids_if_present_else_season_csv",
        "season_csv_used": str(season_csv) if season_csv else "",
        "class_breaks_fixed": ["0", "1-76", "77-107", "108-123", "124-134", ">=135"],
        "class_labels": DEF_LABELS,
        "colors_0to5": DEF_COLORS,
        "nan_color": NAN_COLOR,
        "bbox": bbox,
        "outline_shp": outline_shp,
        "map_kind": map_kind,
        "methods_selected": methods_sel,
        "consensus_enabled": bool(use_consensus),
        "output_dir": str(out_base),
        "created_utc": pd.Timestamp.utcnow().isoformat(timespec="seconds"),
    }
    (d00 / "META_7.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    class_rows: List[Dict] = []
    summary_rows: List[Dict] = []
    gif_frames: Dict[str, List[Path]] = {}

    for year in years:
        grid_path = find_grids_csv(project_root, year)
        df_grid = None

        if grid_path and grid_path.exists():
            df_grid = pd.read_csv(grid_path, low_memory=False)
            df_grid = _normalize_columns(df_grid)
            df_grid = filter_roi(df_grid, bbox)

        if df_grid is None:
            if season_df is None or season_df.empty:
                print(f"[WARN] year={year}: no grids and no season_csv available -> skip")
                continue
            dfy = season_df[season_df["year"].astype(int) == int(year)].copy()
            if dfy.empty:
                print(f"[WARN] year={year}: empty in season_csv -> skip")
                continue
        else:
            dfy = None  # type: ignore

        if df_grid is not None:
            available_methods = [m for m in DEF_METHODS_ALL if m in df_grid.columns]
        else:
            available_methods = sorted(dfy["method"].astype(str).unique().tolist())

        methods_here = [m for m in methods_sel if m in available_methods]

        cons_df = None
        if use_consensus:
            if df_grid is not None:
                cols = [m for m in methods_sel if m in df_grid.columns]
                if len(cols) >= 2:
                    vals = df_grid[cols].apply(pd.to_numeric, errors="coerce").median(axis=1)
                    cons_df = df_grid[["i", "j", "lat", "lon"]].copy()
                    cons_df["season_length"] = pd.to_numeric(vals, errors="coerce")
            else:
                cols = [m for m in methods_sel if m in available_methods]
                if len(cols) >= 2:
                    dtmp = dfy[dfy["method"].isin(cols)][["i", "j", "lat", "lon", "season_length"]].copy()
                    dtmp["season_length"] = pd.to_numeric(dtmp["season_length"], errors="coerce")
                    g = dtmp.groupby(["i", "j", "lat", "lon"])["season_length"]
                    med = g.median()
                    cnt = g.count()
                    cons_df = med[cnt >= 2].reset_index()

        proc_list = methods_here[:]
        if cons_df is not None:
            proc_list.append("consensus")

        for method in proc_list:
            if method == "consensus":
                dvals = cons_df.copy()
            else:
                if df_grid is not None:
                    dvals = df_grid[["i", "j", "lat", "lon"]].copy()
                    dvals["season_length"] = pd.to_numeric(df_grid[method], errors="coerce")
                else:
                    dvals = dfy[dfy["method"].astype(str) == str(method)][["i", "j", "lat", "lon", "season_length"]].copy()
                    dvals["season_length"] = pd.to_numeric(dvals["season_length"], errors="coerce")
                    dvals = dvals.groupby(["i", "j", "lat", "lon"], as_index=False)["season_length"].mean()

            dvals = dvals.dropna(subset=["i", "j", "lat", "lon"])
            if dvals.empty:
                continue

            vals = dvals["season_length"].to_numpy()
            cls = classify_fixed(vals)
            dvals = dvals.copy()
            dvals["risk_class"] = cls

            valid = np.isfinite(vals)
            total_valid = int(valid.sum())
            nonzero = int(np.sum(valid & (vals > 0)))
            nonzero_pct = (100.0 * nonzero / total_valid) if total_valid > 0 else np.nan
            med_season = float(np.nanmedian(vals)) if total_valid > 0 else np.nan

            summary_rows.append({
                "year": int(year),
                "method": method,
                "total_valid": total_valid,
                "nonzero_count": nonzero,
                "nonzero_pct": round(nonzero_pct, 2) if np.isfinite(nonzero_pct) else np.nan,
                "median_season_length": round(med_season, 1) if np.isfinite(med_season) else np.nan,
                "window": WIN,
            })

            for k in range(6):
                n = int(np.nansum(cls == k))
                pct = (100.0 * n / total_valid) if total_valid > 0 else np.nan
                class_rows.append({
                    "year": int(year),
                    "method": method,
                    "class": k,
                    "label": DEF_LABELS[k],
                    "count": n,
                    "pct": round(pct, 2) if np.isfinite(pct) else np.nan,
                    "total_valid": total_valid,
                    "nonzero_count": nonzero,
                    "nonzero_pct": round(nonzero_pct, 2) if np.isfinite(nonzero_pct) else np.nan,
                    "median_season_length": round(med_season, 1) if np.isfinite(med_season) else np.nan,
                    "window": WIN,
                })

            method_map_dir = ensure_dir(d03 / method)
            with mpl_style(style):
                fig, ax = plt.subplots(figsize=(7.8, 6.6))
                ax.set_title(f"Risk classes — season_length — {method} — {year} — win{WIN}", fontsize=12 * font_scale)
                ax.set_xlabel("Longitude", fontsize=11 * font_scale)
                ax.set_ylabel("Latitude", fontsize=11 * font_scale)
                ax.tick_params(labelsize=9 * font_scale)

                mk = "scatter" if map_kind == "tri" else map_kind
                if mk == "grid":
                    mapp = grid_pcolormesh(ax, dvals, "risk_class", cmap, norm, bbox=bbox)
                else:
                    pts = dvals[["lon", "lat", "risk_class"]].dropna()
                    mapp = ax.scatter(pts["lon"], pts["lat"], c=pts["risk_class"], s=9,
                                      cmap=cmap, norm=norm, edgecolor="none")

                draw_outline(ax, outline_shp)
                if bbox:
                    N, Wb, S, E = bbox
                    ax.set_xlim(Wb, E)
                    ax.set_ylim(S, N)
                ax.set_aspect("equal", adjustable="box")

                cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.05)
                cb = fig.colorbar(mapp, cax=cax, ticks=[0, 1, 2, 3, 4, 5])
                cb.set_label("Risk class", fontsize=11 * font_scale, rotation=270, labelpad=14)
                cb.ax.set_yticklabels([f"{k}: {DEF_LABELS[k].split(' ', 1)[1]}" for k in range(6)])
                cb.ax.tick_params(labelsize=9 * font_scale)

                fig.tight_layout()
                out_png = method_map_dir / f"map_{year}_season_length_RISKCLASS_{method}_win{WIN}.png"
                fig.savefig(out_png, dpi=dpi)
                plt.close(fig)

            gif_frames.setdefault(method, []).append(out_png)

    if not class_rows:
        print("[WARN] No outputs produced. Check inputs / bbox / years.")
        return

    df_counts = pd.DataFrame(class_rows).sort_values(["method", "year", "class"])
    df_summary = pd.DataFrame(summary_rows).sort_values(["method", "year"])

    df_counts.to_csv(d01 / f"risk_class_counts_ALL_win{WIN}.csv", index=False)
    df_summary.to_csv(d01 / f"risk_summary_ALL_win{WIN}.csv", index=False)

    methods_out = sorted(df_counts["method"].unique().tolist())

    for method in methods_out:
        d = df_counts[df_counts["method"] == method].copy()
        dfw = d.pivot_table(index="year", columns="class", values="pct", aggfunc="mean").sort_index()
        dfw = dfw.reindex(columns=[0, 1, 2, 3, 4, 5])

        with mpl_style(style):
            fig, ax = plt.subplots(figsize=(12, 5.2))
            ax.stackplot(dfw.index.values, [dfw[c].values for c in dfw.columns],
                         labels=[str(c) for c in dfw.columns], colors=DEF_COLORS, alpha=0.95)
            ax.set_title(f"Risk classes — stacked area (%) — {method} — win{WIN}")
            ax.set_xlabel("Year")
            ax.set_ylabel("Proportion (%)")
            ax.set_ylim(0, 100)
            _set_year_ticks(ax, dfw.index.astype(int).tolist())
            ax.legend(title="Class", ncols=6, loc="upper left")
            fig.tight_layout()
            fig.savefig(d02 / f"stacked_risk_classes_{method}_win{WIN}.png", dpi=dpi)
            plt.close(fig)

        d2 = df_summary[df_summary["method"] == method].drop_duplicates(subset=["year"]).sort_values("year")

        with mpl_style(style):
            fig, ax = plt.subplots(figsize=(10, 4.6))
            ax.plot(d2["year"], d2["nonzero_pct"], lw=2, marker="o")
            ax.set_title(f"% pixels with season_length > 0 — {method} — win{WIN}")
            ax.set_xlabel("Year")
            ax.set_ylabel("Coverage (%)")
            ax.set_ylim(0, 100)
            _set_year_ticks(ax, d2["year"].astype(int).tolist())
            fig.tight_layout()
            fig.savefig(d02 / f"nonzero_coverage_{method}_win{WIN}.png", dpi=dpi)
            plt.close(fig)

        with mpl_style(style):
            fig, ax = plt.subplots(figsize=(10, 4.6))
            ax.plot(d2["year"], d2["median_season_length"], lw=2, marker="o")
            ax.set_title(f"National median season_length (days) — {method} — win{WIN}")
            ax.set_xlabel("Year")
            ax.set_ylabel("Days")
            _set_year_ticks(ax, d2["year"].astype(int).tolist())
            fig.tight_layout()
            fig.savefig(d02 / f"median_season_length_{method}_win{WIN}.png", dpi=dpi)
            plt.close(fig)

    with mpl_style(style):
        fig, ax = plt.subplots(figsize=(11, 5.0))
        for method in methods_out:
            d2 = df_summary[df_summary["method"] == method].sort_values("year")
            ax.plot(d2["year"], d2["nonzero_pct"], lw=2, marker="o", label=method)
        ax.set_title(f"COMPARE — % pixels with season_length > 0 — win{WIN}")
        ax.set_xlabel("Year")
        ax.set_ylabel("Coverage (%)")
        ax.set_ylim(0, 100)
        _set_year_ticks(ax, sorted(df_summary["year"].astype(int).unique().tolist()))
        ax.legend()
        fig.tight_layout()
        fig.savefig(d02 / f"COMPARE_nonzero_coverage_win{WIN}.png", dpi=dpi)
        plt.close(fig)

    with mpl_style(style):
        fig, ax = plt.subplots(figsize=(11, 5.0))
        for method in methods_out:
            d2 = df_summary[df_summary["method"] == method].sort_values("year")
            ax.plot(d2["year"], d2["median_season_length"], lw=2, marker="o", label=method)
        ax.set_title(f"COMPARE — National median season_length (days) — win{WIN}")
        ax.set_xlabel("Year")
        ax.set_ylabel("Days")
        _set_year_ticks(ax, sorted(df_summary["year"].astype(int).unique().tolist()))
        ax.legend()
        fig.tight_layout()
        fig.savefig(d02 / f"COMPARE_median_season_length_win{WIN}.png", dpi=dpi)
        plt.close(fig)

    if make_gif and gif_frames:
        try:
            try:
                import imageio.v2 as imageio
            except Exception:
                import imageio  # type: ignore

            for method, frames in gif_frames.items():
                if not frames:
                    continue
                images = [imageio.imread(str(p)) for p in frames]
                out_gif = d04 / f"risk_classes_{method}_win{WIN}.gif"
                imageio.mimsave(out_gif, images, duration=0.6)
                print(f"[GIF] {out_gif}")
        except Exception as e:
            print(f"[GIF] skipped due to: {e}")

    if do_excel:
        xlsx_path = d99 / f"win{WIN}_7.risk_classes.xlsx"
        engine = excel_engine()
        if embed_images and engine != "xlsxwriter":
            print("[EXCEL] xlsxwriter not available -> disabling --embed-images")
            embed_images = False

        try:
            with pd.ExcelWriter(xlsx_path, engine=engine) as xlw:
                pd.DataFrame([{
                    "Section": "Scope",
                    "Details": f"Fixed risk classes for season_length. Window={WIN}. Methods={', '.join(methods_out)}.",
                }]).to_excel(xlw, sheet_name="README", index=False)

                pd.json_normalize(meta).to_excel(xlw, sheet_name="META_", index=False)
                files_index(out_base).to_excel(xlw, sheet_name="DATA_INDEX", index=False)

                df_counts.to_excel(xlw, sheet_name="COUNTS_ALL", index=False)
                df_summary.to_excel(xlw, sheet_name="SUMMARY_ALL", index=False)

                for method in methods_out:
                    d = df_counts[df_counts["method"] == method]
                    dfw = d.pivot_table(index="year", columns="class", values="pct", aggfunc="mean").sort_index()
                    dfw = dfw.reindex(columns=[0, 1, 2, 3, 4, 5]).reset_index()
                    dfw.to_excel(xlw, sheet_name=sanitize_sheet_name(f"PCT_{method}"), index=False)

                if embed_images and engine == "xlsxwriter":
                    wb = xlw.book
                    ws_plots = wb.add_worksheet("IMG_plots")
                    ws_maps = wb.add_worksheet("IMG_maps_sample")
                    xlw.sheets["IMG_plots"] = ws_plots
                    xlw.sheets["IMG_maps_sample"] = ws_maps

                    plot_files = sorted(d02.glob("*.png"))[:18]
                    row = 0
                    for p in plot_files:
                        ws_plots.insert_image(row, 0, str(p))
                        row += 28

                    row = 0
                    for method in methods_out:
                        mdir = d03 / method
                        mfiles = sorted(mdir.glob(f"map_*_season_length_RISKCLASS_{method}_win{WIN}.png"))
                        for p in mfiles[:4]:
                            ws_maps.insert_image(row, 0, str(p))
                            row += 28

                try:
                    for ws in xlw.sheets.values():
                        ws.set_zoom(110)
                except Exception:
                    pass

            print(f"[EXCEL] {xlsx_path}")
        except Exception as e:
            print(f"[EXCEL] skipped due to: {e}")

    readme_lines = [
        f"# Risk Classes (win{WIN})",
        "",
        "## Inputs",
        "- Uses legacy grids if present:",
        f"  out/12_method_compare/win{WIN}/per_pixel/grids/{{YEAR}}_VALUES_season_length.csv",
        "- Else uses step05 season file:",
        f"  out/08_season_swrap/win{WIN}/csv/season_ALL_YEARS_methods.csv",
        "",
        "## Fixed risk classes",
        "0 = 0 days",
        "1 = 1–76",
        "2 = 77–107",
        "3 = 108–123",
        "4 = 124–134",
        "5 = ≥135",
        "",
        "## Map rules",
        f"- Colors (0..5): {DEF_COLORS}",
        f"- NoData: {NAN_COLOR}",
        "- Square pixels: pcolormesh(..., shading='nearest'); aspect='equal'",
        "",
    ]
    (d00 / "README.md").write_text("\n".join(readme_lines), encoding="utf-8")
    print(f"[DONE] {out_base}")


def main():
    ap = argparse.ArgumentParser(description="step07 — Risk classes (season_length) — WIN30 ONLY (clean outputs)")

    ap.add_argument("--root", type=str, default="", help="Project root (parent of out/). Default: DDU_ROOT/repo root")
    ap.add_argument("--years", type=str, default="2005-2024", help="Default: 2005-2024")
    ap.add_argument("--methods", type=str, default="midpoint,daily_mean,hourly_sum,be_sine", help="Default: all 4")
    ap.add_argument("--consensus", action="store_true", help="Add consensus (median across selected methods, >=2 present)")

    ap.add_argument("--season-csv", type=str, default="", help="Optional explicit path to season_ALL_YEARS_methods.csv")

    ap.add_argument("--bbox", type=str, default=ROMANIA_BBOX_DEFAULT, help="N,W,S,E. Default: Romania bbox")
    ap.add_argument("--outline-shp", type=str, default=DEFAULT_OUTLINE_SHP, help="Outline shapefile (EPSG:4326)")

    ap.add_argument("--style", type=str, default="seaborn-v0_8-whitegrid")
    ap.add_argument("--map-kind", type=str, default="grid", choices=["grid", "scatter", "tri"], help="tri treated as scatter")

    ap.add_argument("--gif", action="store_true", help="Build GIF animations per method (win30)")
    ap.add_argument("--excel", action="store_true", help="Write Excel workbook")
    ap.add_argument("--embed-images", action="store_true", help="Embed a small set of images in Excel (xlsxwriter only)")

    ap.add_argument("--fig-dpi", type=int, default=180, help="Figure DPI")
    ap.add_argument("--font-scale", type=float, default=0.9, help="Global font scaling")

    args = ap.parse_args()

    project_root = detect_root(args.root or None)
    years = parse_years(args.years) or list(range(2005, 2025))
    bbox = parse_bbox(args.bbox) if args.bbox else None

    methods_sel = [m.strip() for m in args.methods.split(",") if m.strip()] or DEF_METHODS_ALL.copy()

    season_csv = Path(args.season_csv).expanduser().resolve() if args.season_csv else find_season_csv(project_root, None)

    print(f"[INFO] PROJECT_ROOT={project_root}")
    print(f"[INFO] window=30 | years={years[0]}..{years[-1]} ({len(years)}) | methods={methods_sel} | consensus={args.consensus}")
    print(f"[INFO] bbox={bbox} | map-kind={args.map_kind} | font-scale={args.font_scale} | dpi={args.fig_dpi}")
    print(f"[INFO] season_csv={season_csv if season_csv else 'NOT FOUND (will rely on grids if present)'}")

    process_win30(
        project_root=project_root,
        years=years,
        methods_sel=methods_sel,
        use_consensus=args.consensus,
        bbox=bbox,
        outline_shp=args.outline_shp or "",
        dpi=args.fig_dpi,
        style=args.style,
        map_kind=args.map_kind,
        make_gif=args.gif,
        do_excel=args.excel,
        embed_images=args.embed_images,
        font_scale=args.font_scale,
        season_csv=season_csv,
    )


if __name__ == "__main__":
    main()
