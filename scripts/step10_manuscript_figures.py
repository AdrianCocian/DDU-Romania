#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step10_manuscript_figures.py - Figures 1-3 of the manuscript at 300 dpi.

Inputs  (written by step09 in <ROOT>/out/12_method_compare/win30/00_MANUSCRIPT_PACK/):
  per_cell_classes_ALL_YEARS.csv     season length and suitability class per cell-year and method
  T5_national_summary_all_methods.csv national median season length per year and method
Boundary: data/boundaries/ro_frontiera_poligon.shp (config.SHP_RO)

Outputs (<pack>/figures/):
  Fig1_season_length_consensus.png/.pdf  season length, median across the four methods, 2005/2011/2017/2024
  Fig2_suitability_classes_hourly_sum.png/.pdf  relative climatic suitability classes (Table 2), hourly sum
  Fig3_season_length_timeseries.png/.pdf  national median season length by method, 2005-2024 (key below the plot)

Figure titles and legends belong in the manuscript text, not in the image files.
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import colors as mcolors  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_THIS = Path(__file__).resolve()
_REPO = _THIS.parent.parent if _THIS.parent.name.lower() == "scripts" else _THIS.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
try:
    os.environ.setdefault("DDU_ROOT", str(_REPO))
    from config import cfg  # noqa: E402
except Exception as _e:
    cfg = None
    warnings.warn(f"config.py not loaded ({_e}); using built-in defaults")

LABELS = dict(getattr(cfg, "METHOD_LABELS", {"midpoint": "Min–max average", "daily_mean": "Hourly average",
                                                "hourly_sum": "Hourly sum", "be_sine": "BE-sine"}))
COLORS = dict(getattr(cfg, "METHOD_COLORS", {"midpoint": "#1f77b4", "daily_mean": "#ff7f0e",
                                                "hourly_sum": "#2ca02c", "be_sine": "#d62728"}))
CLASS_COLORS = list(getattr(cfg, "CLASS_COLORS", ["#e6e6e6", "#4da3ff", "#66c2a5", "#a6d854", "#fdae61", "#d73027"]))
CLASS_LABELS = list(getattr(cfg, "CLASS_LABELS", ["0 no season", "1 lowest", "2 below median", "3 above median",
                                                    "4 high", "5 highest"]))
YEARS = list(getattr(cfg, "REFERENCE_YEARS", [2005, 2011, 2017, 2024]))
BBOX = list(getattr(cfg, "BBOX", [48.5, 19.9, 43.5, 30.1]))          # N, W, S, E
NODATA = "#FFF5B1"
MM = 1 / 25.4


def _grid(df: pd.DataFrame, col: str):
    lat_i = df.groupby("i")["lat"].median().sort_index()
    lon_j = df.groupby("j")["lon"].median().sort_index()
    z = df.pivot_table(index="i", columns="j", values=col, aggfunc="mean").reindex(index=lat_i.index,
                                                                                columns=lon_j.index)
    lon2, lat2 = np.meshgrid(lon_j.values, lat_i.values)
    return lon2, lat2, np.ma.masked_invalid(z.values.astype(float))


def _outline(ax, shp):
    if not shp:
        return
    try:
        import geopandas as gpd  # type: ignore
        gpd.read_file(shp).to_crs(epsg=4326).boundary.plot(ax=ax, linewidth=0.7, color="k", zorder=10)
    except Exception as e:
        print(f"[10] outline skipped: {e}")


def _panel_axes(ax, title):
    n, w, s, e = BBOX
    ax.set_xlim(w, e)
    ax.set_ylim(s, n)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, fontsize=10, loc="left", fontweight="bold")
    ax.set_xlabel("Longitude", fontsize=8)
    ax.set_ylabel("Latitude", fontsize=8)
    ax.tick_params(labelsize=7)


def fig_maps(pc: pd.DataFrame, col: str, cmap, norm, shp, cbar_label, ticks=None, ticklabels=None):
    fig, axes = plt.subplots(2, 2, figsize=(170 * MM, 125 * MM), constrained_layout=True)
    mappable = None
    for ax, y, letter in zip(axes.flat, YEARS, "abcd"):
        d = pc[pc["year"] == y]
        if d.empty:
            warnings.warn(f"year {y} not in the input - empty panel")
            ax.text(0.5, 0.5, f"{y}: no data", transform=ax.transAxes, ha="center", va="center", fontsize=8)
        else:
            lon2, lat2, z = _grid(d, col)
            mappable = ax.pcolormesh(lon2, lat2, z, cmap=cmap, norm=norm, shading="nearest")
        _outline(ax, shp)
        _panel_axes(ax, f"{letter}  {y}")
    if mappable is None:
        mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(mappable, ax=axes, shrink=0.85, ticks=ticks)
    cb.set_label(cbar_label, fontsize=8)
    if ticklabels:
        cb.ax.set_yticklabels(ticklabels)
    cb.ax.tick_params(labelsize=7)
    return fig


def fig3(t5: pd.DataFrame):
    d = t5[(t5["variable"] == "season_length") & (t5["method"].isin(LABELS))]
    order = ["midpoint", "daily_mean", "hourly_sum", "be_sine"]
    with plt.style.context("seaborn-v0_8-whitegrid"):
        fig, ax = plt.subplots(figsize=(170 * MM, 95 * MM))
        for m in order:
            s = d[d["method"] == m].sort_values("year")
            ax.plot(s["year"], s["median"], marker="o", ms=4.5, lw=1.8, color=COLORS[m], label=LABELS[m])
        years = sorted(d["year"].unique())
        ax.set_xticks(years)
        ax.tick_params(axis="x", labelsize=8)
        ax.tick_params(axis="y", labelsize=9)
        ax.set_ylabel("Season length (days)", fontsize=10)
        ax.set_xlim(min(years) - 0.5, max(years) + 0.5)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=4, frameon=False, fontsize=9)
        fig.tight_layout()
    return fig


def main():
    global YEARS
    root = Path(getattr(cfg, "ROOT", _REPO)) if cfg is not None else _REPO
    pack_default = root / "out" / "12_method_compare" / "win30" / str(getattr(cfg, "PACK_DIRNAME", "00_MANUSCRIPT_PACK"))
    ap = argparse.ArgumentParser(description="Figures 1-3 of the manuscript")
    ap.add_argument("--pack", default=str(pack_default), help="folder written by step09")
    ap.add_argument("--out", default=None, help="output folder (default: <pack>/figures)")
    ap.add_argument("--outline-shp", default=str(getattr(cfg, "SHP_RO", "")) if cfg is not None else "")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--years", default=",".join(str(y) for y in YEARS),
                    help="four years for the map panels a-d (default: config.REFERENCE_YEARS)")
    a = ap.parse_args()
    YEARS = [int(y) for y in a.years.split(",") if y.strip()][:4]

    pack = Path(a.pack)
    out = Path(a.out) if a.out else pack / "figures"
    out.mkdir(parents=True, exist_ok=True)
    pc = pd.read_csv(pack / "per_cell_classes_ALL_YEARS.csv")
    t5 = pd.read_csv(pack / "T5_national_summary_all_methods.csv")

    cm1 = plt.get_cmap("bwr").copy()
    cm1.set_bad(NODATA)
    f1 = fig_maps(pc, "consensus", cm1, mcolors.Normalize(vmin=0, vmax=175), a.outline_shp, "Season length (days)")

    cm2 = mcolors.ListedColormap(CLASS_COLORS)
    cm2.set_bad(NODATA)
    labels = [f"{lab.split(' ', 1)[0]}: {lab.split(' ', 1)[1]}" for lab in CLASS_LABELS]
    f2 = fig_maps(pc, "class_hourly_sum", cm2, mcolors.BoundaryNorm(np.arange(-0.5, 6.5, 1.0), 6), a.outline_shp,
                  "Relative climatic suitability class", ticks=list(range(6)), ticklabels=labels)
    f3 = fig3(t5)

    for fig, name in ((f1, "Fig1_season_length_consensus"), (f2, "Fig2_suitability_classes_hourly_sum"),
                      (f3, "Fig3_season_length_timeseries")):
        fig.savefig(out / f"{name}.png", dpi=a.dpi, bbox_inches="tight")
        fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"[10] {out / name}.png")


if __name__ == "__main__":
    main()
