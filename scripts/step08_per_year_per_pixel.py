#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
step08_per_year_per_pixel.py — WIN30 one-shot: YEAR COUNTS (Stage A) + GRAPHS/MAPS (Stage B)
+ EXTRA TS (Stage C) + CENTRAL EXCEL (Stage X) + STATS SUMMARY (Stage S)

Goal (mandatory stats):
- Produce the same statistical outputs as your attached win30_7a_summary.xlsx:
  * Dist_<var>: skewness + Shapiro-Wilk + Hartigan dip test + flags
  * Trend_<var>: Mann–Kendall + Sen slope + Sen intercept (workbook-style)
  * VarRatio_<var>: method-uncertainty vs interannual signal
  * NatSum_<var>: national median + IQR

Scope:
- Window fixed to WIN=30.

Inputs:
1) Step 5:
   <ROOT>/out/08_season_swrap/win30/csv/season_ALL_YEARS_methods.csv
   (fallback: out/11_ddu_season_swrap/win30/csv/season_ALL_YEARS_methods.csv)

2) Step 6 outputs (per-year consensus + pairwise):
   Preferred: <ROOT>/out/12_method_compare/win30/6.compare/
   Fallbacks: <ROOT>/out/12_method_compare/win30/6A/ or .../win30/6/
   Per-year files can live in:
     - <in6>/01_tables/per_year/
     - <in6>/per_year/
     - <in6>/   (legacy)

Outputs:
<ROOT>/out/12_method_compare/win30/8.per_year_per_pixel/
  00_meta/ META_8.json
  yr_counts/ (Stage A)
  graphs/
    tables/ (Stage B tables + stats CSVs)
    plots/  (Stage B+ C plots)
    quickmaps/
      values/
      deltas/
      per_method/<method>/ (optional)
  99_excel/
    win30_8_REPORT.xlsx (+ optional WITH_IMAGES)
    win30_8_summary.xlsx (like win30_7a_summary.xlsx; produced when stats enabled)

Map rules:
- Values cmap: bwr (blue low, red high); start_doy uses bwr_r (red earlier)
- Δ cmap: bwr, symmetric TwoSlopeNorm around 0 (vmax=p98(|Δ|))
- NoData: #FFF5B1
- Square pixels: pcolormesh(..., shading='nearest'); aspect='equal'

Repo portability (GitHub-ready)
-------------------------------
- Auto-detect repo ROOT by walking upward until finding config.py or .git.
- Sets DDU_ROOT automatically and fixes sys.path so `from config import cfg` works
  regardless of current working directory.
- No hardcoded paths.

"""

from __future__ import annotations

import os, sys, json, argparse, warnings, re
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Tuple, List, Dict

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

ROOT_ENV = Path(os.environ.get("DDU_ROOT", str(REPO_ROOT))).resolve()

# Headless
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib
matplotlib.use("Agg", force=True)
warnings.filterwarnings("ignore", message=r'.*Creating legend with loc="best".*')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.ioff()

# optional seaborn
try:
    import seaborn as sns  # type: ignore
    sns.set_style("whitegrid")
    sns.set_context("talk")
except Exception:
    sns = None  # type: ignore

# optional stats deps
try:
    from scipy.stats import skew as _scipy_skew, shapiro as _scipy_shapiro  # type: ignore
except Exception:
    _scipy_skew = None
    _scipy_shapiro = None

try:
    import diptest  # type: ignore
except Exception:
    diptest = None

# optional config
try:
    from config import cfg  # type: ignore[attr-defined]
except Exception:
    cfg = None  # type: ignore

# ---------------- constants ----------------
WIN = 30
NODATA_HEX = "#FFF5B1"

DEFAULT_METHODS = ["midpoint", "daily_mean", "hourly_sum", "be_sine"]

DEFAULT_PAIRS = [
    ("midpoint", "daily_mean"),
    ("midpoint", "be_sine"),
    ("midpoint", "hourly_sum"),
    ("daily_mean", "be_sine"),
    ("daily_mean", "hourly_sum"),
    ("be_sine", "hourly_sum"),
]

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

DEFAULT_LENGTH_BINS = "0,1-30,31-60,61-90,91-120,121-150,151-180,181-365,366+"
DEFAULT_EIP_BINS    = "0,1-30,31-60,61-90,91-120,121-150,151-180,181-240,241+"

# ---------------- small utils ----------------
def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def parse_years(s: Optional[str]) -> Optional[List[int]]:
    if not s:
        return None
    s = s.replace("–","-").replace("—","-").strip()
    if "-" in s:
        a,b = s.split("-",1)
        a,b = int(a), int(b)
        if a>b: a,b=b,a
        return list(range(a,b+1))
    return [int(x) for x in s.split(",") if x.strip()]

def parse_bbox(s: Optional[str]) -> Optional[Tuple[float,float,float,float]]:
    if not s:
        return None
    vals = [float(x) for x in s.split(",")]
    if len(vals)!=4:
        raise ValueError("--bbox must be N,W,S,E")
    N,W,S,E = vals
    if not (S<=N and W<=E):
        raise ValueError("BBOX must be N,W,S,E (S<=N and W<=E).")
    return N,W,S,E

def parse_pair(s: Optional[str]) -> Optional[Tuple[str,str]]:
    if not s:
        return None
    p = [x.strip() for x in s.split(",")]
    if len(p)!=2:
        raise ValueError("--pair m1,m2")
    return p[0], p[1]

def style_ctx(style: str):
    try:
        return plt.style.context(style) if style else plt.style.context("default")
    except Exception:
        return plt.style.context("default")

def apply_nan_cmap(name: str):
    cm = plt.get_cmap(name).copy()
    cm.set_bad(NODATA_HEX)
    return cm

def full_height_colorbar(fig, ax, mappable, label: str):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb = fig.colorbar(mappable, cax=cax)
    cb.set_label(label)
    return cb

def draw_outline(ax, shp_path: Optional[str]):
    if not shp_path:
        return
    try:
        import geopandas as gpd  # type: ignore
        gpd.read_file(shp_path).to_crs(epsg=4326).boundary.plot(ax=ax, linewidth=0.7, color="k", zorder=10)
    except Exception as e:
        print(f"[map] Outline skipped: {e}")

def set_year_ticks(ax, years_list) -> None:
    years=[]
    for y in years_list:
        try:
            yf=float(y)
            if np.isfinite(yf):
                years.append(int(yf))
        except Exception:
            continue
    if not years:
        return
    y0,y1=min(years),max(years)
    ticks=list(range(y0,y1+1))
    ax.set_xlim(y0-0.5, y1+0.5)
    ax.xaxis.set_major_locator(mticker.FixedLocator(ticks))
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%d"))
    ax.tick_params(axis="x", labelsize=8)

# ---------------- normalize/ROI ----------------
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # keep your historical alias but do not override 'method' if present
    if "var_name" in df.columns and "method" not in df.columns:
        df = df.rename(columns={"var_name":"method"})
    if "latitude" in df.columns and "lat" not in df.columns:
        df = df.rename(columns={"latitude":"lat"})
    if "longitude" in df.columns and "lon" not in df.columns:
        df = df.rename(columns={"longitude":"lon"})
    if "Year" in df.columns and "year" not in df.columns:
        df = df.rename(columns={"Year":"year"})
    for k, aliases in VAR_ALIASES.items():
        if k in df.columns:
            continue
        for a in aliases:
            if a in df.columns:
                df = df.rename(columns={a:k})
                break
    return df

def filter_roi(df: pd.DataFrame, bbox: Optional[Tuple[float,float,float,float]]) -> pd.DataFrame:
    if df.empty:
        return df
    for col in ("inside_mask","inside","roi","in_ro","in_roi"):
        if col in df.columns:
            m = df[col].astype(str).str.lower().isin(["1","true","yes"])
            df = df[m].copy()
            break
    if bbox and {"lat","lon"}.issubset(df.columns):
        N,W,S,E = bbox
        df = df[(df["lat"]<=N)&(df["lat"]>=S)&(df["lon"]>=W)&(df["lon"]<=E)].copy()
    return df

# ---------------- DOY mappings ----------------
_BASE = datetime(2001,1,1)
DOY_TO_MONTH = np.array([(_BASE + timedelta(days=i)).month for i in range(365)], dtype=int)
DOY_TO_MMDD  = np.array([(_BASE + timedelta(days=i)).strftime("%m-%d") for i in range(365)], dtype=object)

def doy_to_month(doy: float) -> float:
    if not np.isfinite(doy):
        return np.nan
    d=int(round(float(doy)))
    d=max(1,min(365,d))
    return float(DOY_TO_MONTH[d-1])

def doy_to_mmdd(doy: float) -> Optional[str]:
    if not np.isfinite(doy):
        return None
    d=int(round(float(doy)))
    d=max(1,min(365,d))
    return str(DOY_TO_MMDD[d-1])

# ---------------- bins ----------------
def parse_bins(spec: str) -> List[Tuple[int,int,str]]:
    out=[]
    for tok in [t.strip() for t in spec.split(",") if t.strip()]:
        if "-" in tok:
            a,b = tok.split("-",1)
            out.append((int(a),int(b),f"{int(a)}-{int(b)}"))
        elif tok.endswith("+"):
            lo=int(tok[:-1])
            out.append((lo, int(10**9), f"{lo}+"))
        else:
            v=int(tok)
            out.append((v,v,str(v)))
    return out

def assign_bins(values: np.ndarray, bins: List[Tuple[int,int,str]]) -> np.ndarray:
    lab=np.full(values.shape, None, dtype=object)
    for lo,hi,name in bins:
        m=(values>=lo)&(values<=hi)
        lab[m]=name
    return lab

# ---------------- step-6 discovery ----------------
def list_peryear_files(in6_dir: Path, pattern: str) -> List[Path]:
    cands=[in6_dir/"01_tables"/"per_year", in6_dir/"per_year", in6_dir]
    for base in cands:
        if base.exists() and base.is_dir():
            files=sorted(base.glob(pattern))
            if files:
                return files
    try:
        return sorted(in6_dir.rglob(pattern))
    except Exception:
        return []

def find_in6_dir(out_base: Path, in6_override: Optional[str]) -> Path:
    if in6_override:
        return Path(in6_override)
    cands=[
        out_base/f"win{WIN}"/"6.compare",
        out_base/f"win{WIN}"/"6A",
        out_base/f"win{WIN}"/"6",
    ]
    for c in cands:
        if c.exists():
            return c
    return cands[0]

# ---------------- stats helpers (workbook-matching) ----------------
def _norm_cdf(z: float) -> float:
    import math
    return 0.5*(1.0 + math.erf(z / math.sqrt(2.0)))

def mann_kendall_test(years: np.ndarray, values: np.ndarray) -> dict:
    years=np.asarray(years, dtype=float)
    x=np.asarray(values, dtype=float)
    m=np.isfinite(years)&np.isfinite(x)
    years=years[m]; x=x[m]
    n=int(x.size)
    if n<3:
        return {}
    order=np.argsort(years)
    years=years[order]; x=x[order]

    s=0
    for i in range(n-1):
        di=x[i+1:]-x[i]
        s += int(np.sum(di>0))
        s -= int(np.sum(di<0))

    _,counts=np.unique(x, return_counts=True)
    ties=counts[counts>1]
    var_s = n*(n-1)*(2*n+5)
    if ties.size:
        var_s -= np.sum(ties*(ties-1)*(2*ties+5))
    var_s = var_s/18.0

    if var_s<=0:
        z=0.0
    else:
        if s>0: z=(s-1)/np.sqrt(var_s)
        elif s<0: z=(s+1)/np.sqrt(var_s)
        else: z=0.0

    p=2.0*(1.0 - _norm_cdf(abs(float(z))))
    tau = s/(0.5*n*(n-1))

    slopes=[]
    for i in range(n-1):
        dy=years[i+1:]-years[i]
        dv=x[i+1:]-x[i]
        good = dy!=0
        if np.any(good):
            slopes.extend((dv[good]/dy[good]).tolist())
    sen=float(np.nanmedian(np.asarray(slopes))) if slopes else float("nan")

    # workbook-style intercept
    t=np.arange(n, dtype=float)
    sen_intercept = float(np.nanmedian(x) - sen*np.nanmedian(t)) if np.isfinite(sen) else float("nan")

    alpha=0.05
    if np.isfinite(p) and p<alpha:
        trend="increasing" if z>0 else "decreasing"
    else:
        trend="no trend"

    return {
        "mk_trend": trend,
        "mk_p": float(p),
        "mk_z": float(z),
        "mk_tau": float(tau),
        "sen_slope": float(sen),
        "sen_intercept": float(sen_intercept),
    }

def compute_distribution_diagnostics(values: np.ndarray, year: int, max_n: int=5000) -> dict:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    n = int(v.size)
    out = {"year": int(year), "n": n}

    out["skewness"] = float("nan")
    out["shapiro_stat"] = float("nan")
    out["shapiro_p"] = float("nan")
    out["dip_stat"] = float("nan")
    out["dip_p"] = float("nan")
    out["is_normal"] = False
    out["is_unimodal"] = False

    if n < 3:
        return out

    if _scipy_skew is not None:
        try:
            out["skewness"] = float(_scipy_skew(v, bias=False))
        except Exception:
            pass

    v_test = v
    if n > max_n:
        try:
            rng = np.random.default_rng(42)
            v_test = rng.choice(v, size=max_n, replace=False)
        except Exception:
            v_test = v[:max_n]

    if _scipy_shapiro is not None:
        try:
            stat, p = _scipy_shapiro(v_test)
            out["shapiro_stat"] = float(stat)
            out["shapiro_p"] = float(p)
        except Exception:
            pass

    if diptest is not None:
        try:
            dip, p = diptest.diptest(v_test)
            out["dip_stat"] = float(dip)
            out["dip_p"] = float(p)
        except Exception:
            pass

    if np.isfinite(out["shapiro_p"]):
        out["is_normal"] = bool(out["shapiro_p"] > 0.05)
    if np.isfinite(out["dip_p"]):
        out["is_unimodal"] = bool(out["dip_p"] > 0.05)

    return out

def compute_variance_ratio(method_summaries: Dict[str,pd.DataFrame], value_col: str="median") -> dict:
    if not method_summaries:
        return {}
    combined=pd.DataFrame()
    for meth, df in method_summaries.items():
        if df is None or df.empty or value_col not in df.columns:
            continue
        s=pd.to_numeric(df.set_index("year")[value_col], errors="coerce")
        combined[meth]=s
    if combined.empty:
        return {}
    var_method=float(combined.var(axis=1, ddof=1).mean())
    var_year=float(combined.mean(axis=1).var(ddof=1))
    denom=var_method+var_year
    ratio=(var_method/denom) if denom>0 else float("nan")
    return {
        "var_method_noise": var_method,
        "var_year_signal": var_year,
        "variance_ratio": float(ratio),
        "percent_method_uncertainty": float(ratio*100.0) if np.isfinite(ratio) else float("nan"),
    }

# ---------------- map norms ----------------
def cmap_values(var: str):
    return apply_nan_cmap("bwr_r" if var=="start_doy" else "bwr")

def cmap_deltas():
    return apply_nan_cmap("bwr")

def norm_values_for_var(var: str):
    if var=="season_length":
        return mcolors.Normalize(vmin=0, vmax=175)
    if var=="start_doy":
        return mcolors.Normalize(vmin=75, vmax=225)
    if var=="end_doy":
        return mcolors.Normalize(vmin=150, vmax=300)
    if var=="total_eip_days":
        return mcolors.Normalize(vmin=0, vmax=175)
    return None

def norm_delta_symmetric(vals: np.ndarray):
    a=np.abs(vals[np.isfinite(vals)])
    if a.size==0:
        return mcolors.TwoSlopeNorm(vmin=-1, vcenter=0.0, vmax=1)
    vmax=float(np.percentile(a, 98))
    if not np.isfinite(vmax) or vmax<=0:
        vmax=float(np.nanmax(a)) if a.size else 1.0
    if vmax==0:
        vmax=1.0
    return mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

def grid_pcolormesh(ax, df: pd.DataFrame, vcol: str, cmap, norm, bbox=None):
    need=["i","j","lat","lon",vcol]
    if any(c not in df.columns for c in need):
        return None
    d=df[need].copy()
    for c in need:
        d[c]=pd.to_numeric(d[c], errors="coerce")
    d=d.dropna(subset=["i","j","lat","lon"])
    if d.empty:
        return None
    lat_by_i=d.groupby("i")["lat"].median().sort_index()
    lon_by_j=d.groupby("j")["lon"].median().sort_index()
    Zp=d.pivot_table(index="i", columns="j", values=vcol, aggfunc="mean")
    Zp=Zp.reindex(index=lat_by_i.index, columns=lon_by_j.index)
    Z=np.ma.masked_invalid(Zp.values)
    LAT2D=np.repeat(lat_by_i.values[:,None], len(lon_by_j), axis=1)
    LON2D=np.repeat(lon_by_j.values[None,:], len(lat_by_i), axis=0)
    m=ax.pcolormesh(LON2D, LAT2D, Z, cmap=cmap, norm=norm, shading="nearest")
    if bbox:
        N,W,S,E=bbox
        ax.set_xlim(W,E); ax.set_ylim(S,N)
    ax.set_aspect("equal", adjustable="box")
    return m

def tri_or_scatter(ax, df: pd.DataFrame, vcol: str, cmap, norm, bbox=None, kind="tri"):
    if not {"lat","lon",vcol}.issubset(df.columns):
        return None
    x=pd.to_numeric(df["lon"], errors="coerce").to_numpy()
    y=pd.to_numeric(df["lat"], errors="coerce").to_numpy()
    z=pd.to_numeric(df[vcol], errors="coerce").to_numpy()
    m=np.isfinite(x)&np.isfinite(y)&np.isfinite(z)
    x,y,z=x[m],y[m],z[m]
    if x.size==0:
        return None
    if bbox:
        N,W,S,E=bbox
        sel=(y<=N)&(y>=S)&(x>=W)&(x<=E)
        x,y,z=x[sel],y[sel],z[sel]
        if x.size==0:
            return None
    if kind=="scatter":
        sc=ax.scatter(x,y,c=z,s=9,cmap=cmap,norm=norm,edgecolor="none")
    else:
        try:
            import matplotlib.tri as mtri
            tri=mtri.Triangulation(x,y)
            sc=ax.tripcolor(tri,z,cmap=cmap,norm=norm,shading="flat")
        except Exception:
            sc=ax.scatter(x,y,c=z,s=9,cmap=cmap,norm=norm,edgecolor="none")
    if bbox:
        N,W,S,E=bbox
        ax.set_xlim(W,E); ax.set_ylim(S,N)
    ax.set_aspect("equal", adjustable="box")
    return sc

# ---------------- Stage A ----------------
def stage_a_year_counts(methods_df: pd.DataFrame, out_dir: Path, years: Optional[List[int]], methods: List[str],
                        length_bins_spec: str, eip_bins_spec: str, calendar_dates: bool) -> None:
    ensure_dir(out_dir)
    df=normalize_columns(methods_df.copy())

    req={"year","method","i","j","lat","lon","start_doy","end_doy","season_length","total_eip_days"}
    miss=req - set(df.columns)
    if miss:
        raise ValueError(f"[StageA] Missing columns: {sorted(miss)}")

    df["year"]=pd.to_numeric(df["year"], errors="coerce")
    df=df.dropna(subset=["year"]).copy()
    df["year"]=df["year"].astype(int)
    if years is not None:
        df=df[df["year"].isin(years)].copy()

    df["start_month"]=pd.to_numeric(df["start_doy"], errors="coerce").apply(doy_to_month)
    df["end_month"]=pd.to_numeric(df["end_doy"], errors="coerce").apply(doy_to_month)

    methods_av=sorted(df["method"].dropna().astype(str).unique().tolist())
    use_methods=[m for m in methods if m in methods_av] if methods else methods_av
    if not use_methods:
        raise ValueError(f"[StageA] No methods found. Available: {methods_av}")

    years_av=sorted(df["year"].unique().tolist())
    if not years_av:
        raise ValueError("[StageA] No years after filtering.")

    len_bins=parse_bins(length_bins_spec)
    eip_bins=parse_bins(eip_bins_spec)

    start_doy_rows=[]; end_doy_rows=[]
    start_month_rows=[]; end_month_rows=[]
    len_exact_rows=[]; len_bin_rows=[]
    eip_exact_rows=[]; eip_bin_rows=[]
    year_summary_rows=[]
    start_date_rows=[]; end_date_rows=[]

    for mth in use_methods:
        dmm=df[df["method"].astype(str)==str(mth)].copy()
        if dmm.empty:
            continue
        for y in years_av:
            dyy=dmm[dmm["year"]==y].copy()
            if dyy.empty:
                continue
            n_pixels=int(dyy.shape[0])

            sd=pd.to_numeric(dyy["start_doy"], errors="coerce").dropna()
            if not sd.empty:
                sdi=sd.round().astype(int).clip(1,365)
                vc=sdi.value_counts().sort_index()
                N=int(sdi.shape[0])
                for doy,cnt in vc.items():
                    start_doy_rows.append({"year":y,"method":mth,"doy":int(doy),"count":int(cnt),"pct":round(100.0*cnt/N,4),"N_nonNA":N,"N_pixels":n_pixels})
                if calendar_dates:
                    for doy,cnt in vc.items():
                        mmdd=doy_to_mmdd(float(doy))
                        if mmdd:
                            start_date_rows.append({"year":y,"method":mth,"doy":int(doy),"date_mmdd":mmdd,"date_yyyy_mm_dd":f"{y}-{mmdd}","count":int(cnt),"pct":round(100.0*cnt/N,4),"N_nonNA":N,"N_pixels":n_pixels})

            ed=pd.to_numeric(dyy["end_doy"], errors="coerce").dropna()
            if not ed.empty:
                edi=ed.round().astype(int).clip(1,365)
                vc=edi.value_counts().sort_index()
                N=int(edi.shape[0])
                for doy,cnt in vc.items():
                    end_doy_rows.append({"year":y,"method":mth,"doy":int(doy),"count":int(cnt),"pct":round(100.0*cnt/N,4),"N_nonNA":N,"N_pixels":n_pixels})
                if calendar_dates:
                    for doy,cnt in vc.items():
                        mmdd=doy_to_mmdd(float(doy))
                        if mmdd:
                            end_date_rows.append({"year":y,"method":mth,"doy":int(doy),"date_mmdd":mmdd,"date_yyyy_mm_dd":f"{y}-{mmdd}","count":int(cnt),"pct":round(100.0*cnt/N,4),"N_nonNA":N,"N_pixels":n_pixels})

            sm=pd.to_numeric(dyy["start_month"], errors="coerce").dropna().astype(int)
            if not sm.empty:
                vc=sm.value_counts().sort_index()
                N=int(sm.shape[0])
                for mo,cnt in vc.items():
                    start_month_rows.append({"year":y,"method":mth,"month":int(mo),"count":int(cnt),"pct":round(100.0*cnt/N,4),"N_nonNA":N,"N_pixels":n_pixels})

            em=pd.to_numeric(dyy["end_month"], errors="coerce").dropna().astype(int)
            if not em.empty:
                vc=em.value_counts().sort_index()
                N=int(em.shape[0])
                for mo,cnt in vc.items():
                    end_month_rows.append({"year":y,"method":mth,"month":int(mo),"count":int(cnt),"pct":round(100.0*cnt/N,4),"N_nonNA":N,"N_pixels":n_pixels})

            L=pd.to_numeric(dyy["season_length"], errors="coerce").fillna(0).round().astype(int)
            N=int(L.shape[0])
            if N>0:
                vc=L.value_counts().sort_index()
                for val,cnt in vc.items():
                    len_exact_rows.append({"year":y,"method":mth,"season_length_days":int(val),"count":int(cnt),"pct":round(100.0*cnt/N,4),"N_pixels":n_pixels})
                lab=assign_bins(L.to_numpy(), len_bins)
                tmp=pd.Series(lab).value_counts()
                for blab,cnt in tmp.items():
                    if blab is None: continue
                    len_bin_rows.append({"year":y,"method":mth,"bin_label":str(blab),"count":int(cnt),"pct":round(100.0*cnt/N,4),"N_pixels":n_pixels})

            E=pd.to_numeric(dyy["total_eip_days"], errors="coerce").fillna(0).round().astype(int)
            N=int(E.shape[0])
            if N>0:
                vc=E.value_counts().sort_index()
                for val,cnt in vc.items():
                    eip_exact_rows.append({"year":y,"method":mth,"total_eip_days":int(val),"count":int(cnt),"pct":round(100.0*cnt/N,4),"N_pixels":n_pixels})
                lab=assign_bins(E.to_numpy(), eip_bins)
                tmp=pd.Series(lab).value_counts()
                for blab,cnt in tmp.items():
                    if blab is None: continue
                    eip_bin_rows.append({"year":y,"method":mth,"bin_label":str(blab),"count":int(cnt),"pct":round(100.0*cnt/N,4),"N_pixels":n_pixels})

            sd_all=pd.to_numeric(dyy["start_doy"], errors="coerce").to_numpy()
            ed_all=pd.to_numeric(dyy["end_doy"], errors="coerce").to_numpy()
            L_all=pd.to_numeric(dyy["season_length"], errors="coerce").to_numpy()
            E_all=pd.to_numeric(dyy["total_eip_days"], errors="coerce").to_numpy()

            def _p(x,q):
                x=x[np.isfinite(x)]
                return float(np.nanpercentile(x,q)) if x.size else float("nan")

            n_with=int(np.isfinite(sd_all).sum())
            pct_with=round(100.0*n_with/n_pixels,3) if n_pixels else 0.0

            year_summary_rows.append({
                "year":y,"method":mth,"N_pixels":n_pixels,"N_with_season":n_with,"pct_with_season":pct_with,
                "start_doy_mean": float(np.nanmean(sd_all)) if np.isfinite(sd_all).any() else float("nan"),
                "start_doy_median": float(np.nanmedian(sd_all)) if np.isfinite(sd_all).any() else float("nan"),
                "start_doy_p10": _p(sd_all,10), "start_doy_p90": _p(sd_all,90),
                "end_doy_mean": float(np.nanmean(ed_all)) if np.isfinite(ed_all).any() else float("nan"),
                "end_doy_median": float(np.nanmedian(ed_all)) if np.isfinite(ed_all).any() else float("nan"),
                "end_doy_p10": _p(ed_all,10), "end_doy_p90": _p(ed_all,90),
                "season_length_mean": float(np.nanmean(L_all)) if np.isfinite(L_all).any() else float("nan"),
                "season_length_median": float(np.nanmedian(L_all)) if np.isfinite(L_all).any() else float("nan"),
                "season_length_p10": _p(L_all,10), "season_length_p90": _p(L_all,90),
                "total_eip_days_mean": float(np.nanmean(E_all)) if np.isfinite(E_all).any() else float("nan"),
                "total_eip_days_median": float(np.nanmedian(E_all)) if np.isfinite(E_all).any() else float("nan"),
                "total_eip_days_p10": _p(E_all,10), "total_eip_days_p90": _p(E_all,90),
            })

    def _write(name, rows):
        pd.DataFrame(rows).to_csv(out_dir/name, index=False)

    _write("START_DOY_YEAR_MATRIX.csv", start_doy_rows)
    _write("END_DOY_YEAR_MATRIX.csv", end_doy_rows)
    _write("START_MONTH_YEAR_MATRIX.csv", start_month_rows)
    _write("END_MONTH_YEAR_MATRIX.csv", end_month_rows)
    _write("SEASON_LENGTH_EXACT_YEAR_MATRIX.csv", len_exact_rows)
    _write("SEASON_LENGTH_BINS_YEAR_MATRIX.csv", len_bin_rows)
    _write("TOTAL_EIP_DAYS_EXACT_YEAR_MATRIX.csv", eip_exact_rows)
    _write("TOTAL_EIP_DAYS_BINS_YEAR_MATRIX.csv", eip_bin_rows)
    _write("YEAR_SUMMARY_STATS.csv", year_summary_rows)
    if calendar_dates:
        _write("START_DATE_YEAR_MATRIX.csv", start_date_rows)
        _write("END_DATE_YEAR_MATRIX.csv", end_date_rows)

    print(f"[StageA] Wrote year-count matrices into: {out_dir}")

# ---------------- Stage B ----------------
def find_agg_pairwise(in6_dir: Path, var: str) -> Optional[Path]:
    cands=[
        in6_dir/"agg"/f"{var}_PAIRWISE_MEDIAN_ALL_YEARS.csv",
        in6_dir/"01_tables"/"agg"/f"{var}_PAIRWISE_MEDIAN_ALL_YEARS.csv",
        in6_dir/"01_tables"/f"{var}_PAIRWISE_MEDIAN_ALL_YEARS.csv",
    ]
    for c in cands:
        if c.exists():
            return c
    return None

def stage_b_graphs_maps(
    in6_dir: Path,
    methods_df: pd.DataFrame,
    out_dir: Path,
    years: Optional[List[int]],
    variables: List[str],
    methods: List[str],
    bbox: Optional[Tuple[float,float,float,float]],
    outline_shp: Optional[str],
    style: str,
    fig_dpi: int,
    map_kind: str,
    no_maps: bool,
    pair_focus: Optional[Tuple[str,str]],
    delta_flag: Optional[float],
    no_per_method: bool,
    no_per_method_maps: bool,
    thr_list: Tuple[float,float],
    stats: bool,
    stats_max_n: int,
    scatter_max: int,
) -> None:
    graphs=ensure_dir(out_dir/"graphs")
    tables=ensure_dir(graphs/"tables")
    plots=ensure_dir(graphs/"plots")
    qm=ensure_dir(graphs/"quickmaps")
    maps_values=ensure_dir(qm/"values")
    maps_deltas=ensure_dir(qm/"deltas")
    maps_pm=ensure_dir(qm/"per_method")

    # consensus summaries from per-year MEDIAN files
    for var in variables:
        median_files=list_peryear_files(in6_dir, f"*_{var}_MEDIAN.csv")
        median_by_year={}
        rows=[]
        for f in median_files:
            try: y=int(f.name.split("_")[0])
            except Exception: continue
            median_by_year[y]=f

        for y,f in sorted(median_by_year.items()):
            if years and y not in years:
                continue
            df=filter_roi(normalize_columns(pd.read_csv(f, low_memory=False)), bbox)
            vcol="median_methods" if "median_methods" in df.columns else var
            if vcol not in df.columns:
                continue
            v=pd.to_numeric(df[vcol], errors="coerce").to_numpy()
            v=v[np.isfinite(v)]
            if v.size==0:
                continue
            rows.append({"year":y,"median":float(np.nanmedian(v)),"p25":float(np.nanpercentile(v,25)),"p75":float(np.nanpercentile(v,75)),"count":int(v.size)})

        if rows:
            summ=pd.DataFrame(rows).sort_values("year")
            summ.to_csv(tables/f"national_summary_{var}.csv", index=False)

            if stats:
                dist_rows=[]
                for y in summ["year"].astype(int).tolist():
                    pf=median_by_year.get(int(y))
                    if not pf: continue
                    dfi=filter_roi(normalize_columns(pd.read_csv(pf, low_memory=False)), bbox)
                    vcol="median_methods" if "median_methods" in dfi.columns else var
                    vv=pd.to_numeric(dfi[vcol], errors="coerce").to_numpy()
                    vv=vv[np.isfinite(vv)]
                    dist_rows.append(compute_distribution_diagnostics(vv, int(y), max_n=stats_max_n))
                if dist_rows:
                    pd.DataFrame(dist_rows).sort_values("year").to_csv(tables/f"distribution_stats_{var}.csv", index=False)
                tr=mann_kendall_test(summ["year"].to_numpy(), summ["median"].to_numpy())
                if tr:
                    pd.DataFrame([tr]).to_csv(tables/f"trend_stats_{var}.csv", index=False)

            with style_ctx(style):
                fig,ax=plt.subplots(figsize=(10,5))
                ax.plot(summ["year"], summ["median"], lw=2, marker="o", label="Median (consensus)")
                ax.fill_between(summ["year"], summ["p25"], summ["p75"], alpha=0.18, label="IQR")
                x=summ["year"].to_numpy(float); yv=summ["median"].to_numpy(float)
                m=np.isfinite(x)&np.isfinite(yv)
                if m.sum()>=2:
                    a,b=np.polyfit(x[m], yv[m], 1)
                    ax.plot(x[m], a*x[m]+b, lw=1.6, ls="--", alpha=0.85, label=f"Trend ≈ {a:.2f}/yr")
                ax.set_title(f"National time series — {DISPLAY.get(var,var)} (consensus)")
                ax.set_xlabel("Year"); set_year_ticks(ax, summ["year"].tolist())
                ax.set_ylabel(DISPLAY.get(var,var)); ax.legend()
                fig.tight_layout()
                fig.savefig(plots/f"timeseries_{var}.png", dpi=fig_dpi)
                plt.close(fig)

            # maps
            if not no_maps:
                cmap=cmap_values(var); norm=norm_values_for_var(var)
                for y,f in sorted(median_by_year.items()):
                    if years and y not in years: continue
                    dfm=filter_roi(normalize_columns(pd.read_csv(f, low_memory=False)), bbox)
                    vcol="median_methods" if "median_methods" in dfm.columns else var
                    if vcol not in dfm.columns: continue
                    with style_ctx(style):
                        fig,ax=plt.subplots(figsize=(7.1,6))
                        ax.set_title(f"{DISPLAY.get(var,var)} (consensus) — {y}")
                        ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
                        mapp = grid_pcolormesh(ax, dfm, vcol, cmap, norm, bbox=bbox) if map_kind=="grid" else None
                        if mapp is None:
                            mapp = tri_or_scatter(ax, dfm, vcol, cmap, norm, bbox=bbox, kind=map_kind)
                        if mapp is None:
                            plt.close(fig); continue
                        full_height_colorbar(fig, ax, mapp, DISPLAY.get(var,var))
                        draw_outline(ax, outline_shp)
                        fig.tight_layout()
                        fig.savefig(maps_values/f"map_{y}_{var}.png", dpi=fig_dpi)
                        plt.close(fig)

    # per-method overlays + variance ratio
    if (not no_per_method) and (not methods_df.empty) and {"year","method"}.issubset(methods_df.columns):
        dfm=filter_roi(normalize_columns(methods_df.copy()), bbox)
        if years is not None:
            dfm=dfm[dfm["year"].astype("Int64").isin(years)].copy()
        methods_av=sorted(dfm["method"].dropna().astype(str).unique().tolist())
        use_methods=[m for m in methods if m in methods_av] if methods else methods_av

        for var in variables:
            if var not in dfm.columns:
                continue
            summ_by_method={}
            for mth in use_methods:
                sub=dfm[dfm["method"].astype(str)==str(mth)][["year",var]].copy()
                sub[var]=pd.to_numeric(sub[var], errors="coerce")
                sub=sub.dropna(subset=[var])
                if sub.empty: continue
                g=sub.groupby("year")[var]
                summ=pd.DataFrame({"year":g.median().index.astype(int),
                                   "median":g.median().values,
                                   "p25":g.quantile(0.25).values,
                                   "p75":g.quantile(0.75).values,
                                   "count":g.count().values})
                summ.to_csv(tables/f"national_summary_{var}_method_{mth}.csv", index=False)
                summ_by_method[mth]=summ

            if summ_by_method:
                with style_ctx(style):
                    fig,ax=plt.subplots(figsize=(11,5.2))
                    all_years=set()
                    for mth,summ in summ_by_method.items():
                        ax.plot(summ["year"], summ["median"], lw=2, marker="o", label=str(mth))
                        all_years |= set(summ["year"].astype(int).tolist())
                    ax.set_title(f"National time series — {DISPLAY.get(var,var)} (methods overlay)")
                    ax.set_xlabel("Year"); set_year_ticks(ax, sorted(all_years))
                    ax.set_ylabel(DISPLAY.get(var,var)); ax.legend(title="Method", ncols=2)
                    fig.tight_layout()
                    fig.savefig(plots/f"overlay_timeseries_{var}.png", dpi=fig_dpi)
                    plt.close(fig)

                if stats:
                    vr=compute_variance_ratio(summ_by_method, value_col="median")
                    if vr:
                        vr["variable"]=var
                        pd.DataFrame([vr]).to_csv(tables/f"variance_ratio_{var}.csv", index=False)

            if (not no_maps) and (not no_per_method_maps):
                cmap=cmap_values(var); norm=norm_values_for_var(var)
                for mth in use_methods:
                    sub=dfm[dfm["method"].astype(str)==str(mth)].copy()
                    if sub.empty: continue
                    pm_dir=ensure_dir(maps_pm/mth)
                    for y, dyy in sub.groupby("year"):
                        y=int(y)
                        if years and y not in years: continue
                        dyy=dyy.copy()
                        dyy[var]=pd.to_numeric(dyy[var], errors="coerce")
                        with style_ctx(style):
                            fig,ax=plt.subplots(figsize=(7.1,6))
                            ax.set_title(f"{DISPLAY.get(var,var)} — {mth} — {y}")
                            ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
                            mapp = grid_pcolormesh(ax, dyy, var, cmap, norm, bbox=bbox) if map_kind=="grid" else None
                            if mapp is None:
                                mapp = tri_or_scatter(ax, dyy, var, cmap, norm, bbox=bbox, kind=map_kind)
                            if mapp is None:
                                plt.close(fig); continue
                            full_height_colorbar(fig, ax, mapp, DISPLAY.get(var,var))
                            draw_outline(ax, outline_shp)
                            fig.tight_layout()
                            fig.savefig(pm_dir/f"map_{y}_{var}_{mth}.png", dpi=fig_dpi)
                            plt.close(fig)

    # pairwise deltas: summary, article tables, plots, maps
    for var in variables:
        pair_files=list_peryear_files(in6_dir, f"*_{var}_PAIRWISE.csv")
        if not pair_files:
            continue

        # delta summary per year/pair
        dsum=[]
        for f in pair_files:
            try: y=int(f.name.split("_")[0])
            except Exception: continue
            if years and y not in years: continue
            df=filter_roi(normalize_columns(pd.read_csv(f, low_memory=False)), bbox)
            if not {"method1","method2","delta"}.issubset(df.columns):
                continue
            df["delta"]=pd.to_numeric(df["delta"], errors="coerce")
            for m1,m2 in DEFAULT_PAIRS:
                sub=df[((df["method1"]==m1)&(df["method2"]==m2))|((df["method1"]==m2)&(df["method2"]==m1))].copy()
                if sub.empty: continue
                rev=(sub["method1"]==m2)&(sub["method2"]==m1)
                if rev.any():
                    sub.loc[rev,"delta"]=-pd.to_numeric(sub.loc[rev,"delta"], errors="coerce")
                    sub.loc[rev,"method1"]=m1; sub.loc[rev,"method2"]=m2
                s=pd.to_numeric(sub["delta"], errors="coerce").to_numpy()
                s=s[np.isfinite(s)]
                if s.size==0: continue
                dsum.append({"year":y,"method1":m1,"method2":m2,
                             "median_delta":float(np.nanmedian(s)),
                             "p25":float(np.nanpercentile(s,25)),
                             "p75":float(np.nanpercentile(s,75)),
                             "mean":float(np.nanmean(s)),
                             "sd":float(np.nanstd(s, ddof=1)) if s.size>1 else 0.0,
                             "count":int(s.size)})
        if dsum:
            pd.DataFrame(dsum).sort_values(["method1","method2","year"]).to_csv(tables/f"delta_summary_{var}.csv", index=False)

        # article tables prefer agg if exists
        agg=find_agg_pairwise(in6_dir, var)
        if agg and agg.exists():
            big=pd.read_csv(agg, low_memory=False)
        else:
            parts=[]
            for f in pair_files:
                parts.append(pd.read_csv(f, usecols=lambda c: c in {"method1","method2","delta"}, low_memory=False))
            big=pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

        if not big.empty and {"method1","method2","delta"}.issubset(big.columns):
            big=normalize_columns(big)
            big["delta"]=pd.to_numeric(big["delta"], errors="coerce")
            rows_sum=[]; rows_thr=[]
            t0,t1=float(thr_list[0]), float(thr_list[1])
            for m1,m2 in DEFAULT_PAIRS:
                sub=big[((big["method1"]==m1)&(big["method2"]==m2))|((big["method1"]==m2)&(big["method2"]==m1))].copy()
                if sub.empty: continue
                rev=(sub["method1"]==m2)&(sub["method2"]==m1)
                if rev.any():
                    sub.loc[rev,"delta"]=-pd.to_numeric(sub.loc[rev,"delta"], errors="coerce")
                    sub.loc[rev,"method1"]=m1; sub.loc[rev,"method2"]=m2
                s=pd.to_numeric(sub["delta"], errors="coerce").to_numpy()
                s=s[np.isfinite(s)]
                if s.size==0: continue
                rows_sum.append({"variable":var,"pair":f"{m1}-{m2}","n":int(s.size),
                                 "p25":float(np.nanpercentile(s,25)),
                                 "median":float(np.nanmedian(s)),
                                 "p75":float(np.nanpercentile(s,75)),
                                 "mean":float(np.nanmean(s)),
                                 "sd":float(np.nanstd(s, ddof=1)) if s.size>1 else 0.0,
                                 "mad_mean":float(np.nanmean(np.abs(s-np.nanmean(s)))) if s.size else np.nan})
                a=np.abs(s)
                rows_thr.append({"variable":var,"pair":f"{m1}-{m2}","n":int(a.size),
                                 f"pct_abs<= {t0:g}":float(np.nanmean(a<=t0)*100.0),
                                 f"pct_abs<= {t1:g}":float(np.nanmean(a<=t1)*100.0),
                                 "max_abs":float(np.nanmax(a))})
                # delta hist
                with style_ctx(style):
                    fig,ax=plt.subplots(figsize=(7.2,4.6))
                    ax.hist(s, bins=41, density=False)
                    ax.axvline(0, color="k", lw=1)
                    ax.set_title(f"Δ {DISPLAY.get(var,var)} — all-years median per pixel — {m1}-{m2}")
                    ax.set_xlabel(f"Δ {DISPLAY.get(var,var)} (m1−m2)")
                    ax.set_ylabel("Pixels")
                    fig.tight_layout()
                    fig.savefig(plots/f"delta_hist_{var}_{m1}-{m2}.png", dpi=fig_dpi)
                    plt.close(fig)
            if rows_sum:
                pd.DataFrame(rows_sum).to_csv(tables/f"article_pairwise_summary_{var}.csv", index=False)
            if rows_thr:
                pd.DataFrame(rows_thr).to_csv(tables/f"article_pairwise_thresholds_{var}.csv", index=False)

        # focus pairs
        focus_pairs=[pair_focus] if pair_focus else DEFAULT_PAIRS
        if delta_flag is not None and pair_focus is None:
            print("[WARN] --delta-flag ignored unless --pair is provided.")
        do_flag = (delta_flag is not None and pair_focus is not None)

        for m1,m2 in focus_pairs:
            if m1 is None or m2 is None:
                continue

            box_parts=[]
            xs=[]; ys=[]; n_sc=0

            for f in pair_files:
                try: y=int(f.name.split("_")[0])
                except Exception: continue
                if years and y not in years: continue
                usecols={"year","method1","method2","delta","value_method1","value_method2","i","j","lat","lon"}
                df=pd.read_csv(f, usecols=lambda c: c in usecols, low_memory=False)
                df=filter_roi(normalize_columns(df), bbox)
                if not {"method1","method2","delta"}.issubset(df.columns): continue
                df["delta"]=pd.to_numeric(df["delta"], errors="coerce")
                df["year"]=y

                sub=df[((df["method1"]==m1)&(df["method2"]==m2))|((df["method1"]==m2)&(df["method2"]==m1))].copy()
                if sub.empty: continue
                rev=(sub["method1"]==m2)&(sub["method2"]==m1)
                if rev.any():
                    sub.loc[rev,"delta"]=-pd.to_numeric(sub.loc[rev,"delta"], errors="coerce")

                s=pd.to_numeric(sub["delta"], errors="coerce")
                s=s[np.isfinite(s)]
                if not s.empty:
                    box_parts.append(pd.DataFrame({"year":y,"delta":s.to_numpy()}))

                if {"value_method1","value_method2"}.issubset(sub.columns) and n_sc < scatter_max:
                    v1=pd.to_numeric(sub["value_method1"], errors="coerce").to_numpy()
                    v2=pd.to_numeric(sub["value_method2"], errors="coerce").to_numpy()
                    rr=rev.to_numpy()
                    X=np.where(rr, v2, v1)  # canonical x=m1
                    Y=np.where(rr, v1, v2)  # canonical y=m2
                    mm=np.isfinite(X)&np.isfinite(Y)
                    X,Y=X[mm],Y[mm]
                    if X.size:
                        remain=scatter_max-n_sc
                        if X.size>remain:
                            idx=np.random.default_rng(42).choice(X.size, size=remain, replace=False)
                            X,Y=X[idx],Y[idx]
                        xs.append(X); ys.append(Y); n_sc += X.size

            if box_parts:
                bigb=pd.concat(box_parts, ignore_index=True)
                with style_ctx(style):
                    fig,ax=plt.subplots(figsize=(12,5))
                    bigb.boxplot(column="delta", by="year", ax=ax, grid=False, showfliers=False)
                    plt.suptitle("")
                    ax.axhline(0, color="k", lw=1, alpha=0.6)
                    ax.set_title(f"Δ {DISPLAY.get(var,var)} ({m1}−{m2}) per year")
                    ax.set_xlabel("Year"); set_year_ticks(ax, bigb["year"].tolist())
                    ax.set_ylabel(f"Δ {DISPLAY.get(var,var)} (m1−m2)")
                    fig.tight_layout()
                    fig.savefig(plots/f"delta_box_{var}_{m1}-{m2}.png", dpi=fig_dpi)
                    plt.close(fig)

            if xs and ys:
                X=np.concatenate(xs); Y=np.concatenate(ys)
                if X.size and Y.size:
                    mn=float(min(X.min(), Y.min())); mx=float(max(X.max(), Y.max()))
                    with style_ctx(style):
                        fig,ax=plt.subplots(figsize=(6.5,6.5))
                        ax.scatter(X,Y,s=5,alpha=0.25,edgecolor="none")
                        ax.plot([mn,mx],[mn,mx],"k--",lw=1)
                        ax.set_xlabel(f"{m1} {DISPLAY.get(var,var)}")
                        ax.set_ylabel(f"{m2} {DISPLAY.get(var,var)}")
                        ax.set_title(f"{DISPLAY.get(var,var)} — {m1} vs {m2} (sampled all-years)")
                        fig.tight_layout()
                        fig.savefig(plots/f"scatter_{var}_{m1}-{m2}_ALLYEARS.png", dpi=fig_dpi)
                        plt.close(fig)

                    meanv=(X+Y)/2.0
                    diff=(X-Y)
                    md=float(np.nanmean(diff))
                    sd=float(np.nanstd(diff, ddof=1)) if diff.size>1 else 0.0
                    with style_ctx(style):
                        fig,ax=plt.subplots(figsize=(7.5,5.2))
                        ax.scatter(meanv,diff,s=5,alpha=0.25,edgecolor="none")
                        ax.axhline(md,color="k",ls="--",lw=1,label=f"mean Δ = {md:.2f}")
                        ax.axhline(md+1.96*sd,color="r",ls="--",lw=1)
                        ax.axhline(md-1.96*sd,color="r",ls="--",lw=1)
                        ax.set_xlabel("Mean of methods")
                        ax.set_ylabel(f"Δ {DISPLAY.get(var,var)} (m1−m2)")
                        ax.set_title(f"Bland–Altman — {DISPLAY.get(var,var)} ({m1}−{m2})")
                        ax.legend()
                        fig.tight_layout()
                        fig.savefig(plots/f"bland_altman_{var}_{m1}-{m2}_ALLYEARS.png", dpi=fig_dpi)
                        plt.close(fig)

            # delta maps per year
            if not no_maps:
                cmap=cmap_deltas()
                for f in pair_files:
                    try: y=int(f.name.split("_")[0])
                    except Exception: continue
                    if years and y not in years: continue
                    dfm=filter_roi(normalize_columns(pd.read_csv(f, low_memory=False)), bbox)
                    if not {"method1","method2","delta"}.issubset(dfm.columns): continue
                    dfm["delta"]=pd.to_numeric(dfm["delta"], errors="coerce")
                    sub=dfm[((dfm["method1"]==m1)&(dfm["method2"]==m2))|((dfm["method1"]==m2)&(dfm["method2"]==m1))].copy()
                    if sub.empty: continue
                    rev=(sub["method1"]==m2)&(sub["method2"]==m1)
                    if rev.any():
                        sub.loc[rev,"delta"]=-pd.to_numeric(sub.loc[rev,"delta"], errors="coerce")
                        sub.loc[rev,"method1"]=m1; sub.loc[rev,"method2"]=m2
                    vals=pd.to_numeric(sub["delta"], errors="coerce").to_numpy()
                    norm=norm_delta_symmetric(vals)
                    with style_ctx(style):
                        fig,ax=plt.subplots(figsize=(7.1,6))
                        ax.set_title(f"Δ {DISPLAY.get(var,var)} ({m1}−{m2}) — {y}")
                        ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
                        mapp = grid_pcolormesh(ax, sub, "delta", cmap, norm, bbox=bbox) if map_kind=="grid" else None
                        if mapp is None:
                            mapp = tri_or_scatter(ax, sub, "delta", cmap, norm, bbox=bbox, kind=map_kind)
                        if mapp is None:
                            plt.close(fig); continue
                        full_height_colorbar(fig, ax, mapp, f"Δ {DISPLAY.get(var,var)}")
                        draw_outline(ax, outline_shp)
                        fig.tight_layout()
                        fig.savefig(maps_deltas/f"map_{y}_{var}_DELTA_{m1}-{m2}.png", dpi=fig_dpi)
                        plt.close(fig)

                    if do_flag and (pair_focus == (m1,m2)):
                        thr=float(delta_flag)
                        sub2=sub.copy()
                        sub2["flag"]=(np.abs(pd.to_numeric(sub2["delta"], errors="coerce"))>=thr).astype(float)
                        vnorm=mcolors.Normalize(vmin=0, vmax=1)
                        cmap_flag=apply_nan_cmap("gray_r")
                        with style_ctx(style):
                            fig,ax=plt.subplots(figsize=(7.1,6))
                            ax.set_title(f"|Δ {DISPLAY.get(var,var)}| ≥ {thr} ({m1}−{m2}) — {y}")
                            ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
                            mapp = grid_pcolormesh(ax, sub2, "flag", cmap_flag, vnorm, bbox=bbox) if map_kind=="grid" else None
                            if mapp is None:
                                mapp = tri_or_scatter(ax, sub2, "flag", cmap_flag, vnorm, bbox=bbox, kind=map_kind)
                            if mapp is None:
                                plt.close(fig); continue
                            full_height_colorbar(fig, ax, mapp, "flag (0/1)")
                            draw_outline(ax, outline_shp)
                            fig.tight_layout()
                            fig.savefig(maps_deltas/f"map_{y}_{var}_DELTAFLAG_{int(thr)}_{m1}-{m2}.png", dpi=fig_dpi)
                            plt.close(fig)

    print(f"[StageB] wrote into: {graphs}")

# ---------------- Stage C ----------------
def stage_c_counts_timeseries(out_root: Path, style: str, fig_dpi: int) -> None:
    counts_dir=out_root/"yr_counts"
    p=counts_dir/"YEAR_SUMMARY_STATS.csv"
    if not p.exists():
        return
    df=pd.read_csv(p, low_memory=False)
    if df.empty or not {"year","method"}.issubset(df.columns):
        return
    plots=ensure_dir(out_root/"graphs"/"plots")
    metrics=[
        ("start_doy_median","Start DOY (median)"),
        ("end_doy_median","End DOY (median)"),
        ("season_length_median","Season length (median)"),
        ("total_eip_days_median","Total EIP days (median)"),
    ]
    with style_ctx(style):
        for col,label in metrics:
            if col not in df.columns: continue
            fig,ax=plt.subplots(figsize=(11,5))
            for mth, sub in df.groupby("method"):
                sub=sub.sort_values("year")
                ax.plot(sub["year"], sub[col], lw=2, marker="o", label=str(mth))
            ax.set_title(f"YEAR_SUMMARY_STATS — {label} (by method)")
            ax.set_xlabel("Year"); ax.set_ylabel(label)
            ax.legend(ncols=2)
            fig.tight_layout()
            fig.savefig(plots/f"yr_counts_ts_{col}.png", dpi=fig_dpi)
            plt.close(fig)
    print("[StageC] wrote extra TS plots.")

# ---------------- Excel writers ----------------
def build_files_index(out_root: Path) -> pd.DataFrame:
    rows=[]
    for p in out_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in (".csv",".png",".json",".xlsx"):
            try:
                sz=p.stat().st_size/1024/1024
            except Exception:
                sz=np.nan
            rows.append({"file":p.relative_to(out_root).as_posix(),"size_MB":round(float(sz),4) if np.isfinite(sz) else np.nan})
    return pd.DataFrame(rows).sort_values("file") if rows else pd.DataFrame(columns=["file","size_MB"])

def write_central_excel(out_root: Path, meta: dict, embed_images: bool=False) -> Path:
    excel_dir=ensure_dir(out_root/"99_excel")
    xlsx=excel_dir/("win30_8_REPORT_WITH_IMAGES.xlsx" if embed_images else "win30_8_REPORT.xlsx")
    tables_dir=out_root/"graphs"/"tables"
    counts_dir=out_root/"yr_counts"
    plots_dir=out_root/"graphs"/"plots"
    maps_dir=out_root/"graphs"/"quickmaps"

    files_idx=build_files_index(out_root)

    engine="xlsxwriter"
    try:
        import xlsxwriter  # noqa: F401
    except Exception:
        engine="openpyxl"

    with pd.ExcelWriter(xlsx, engine=engine) as xlw:
        pd.DataFrame([meta]).to_excel(xlw, sheet_name="META", index=False)
        if not files_idx.empty:
            files_idx.to_excel(xlw, sheet_name="FILES_INDEX", index=False)

        stage_a_files=[
            "START_DOY_YEAR_MATRIX.csv","END_DOY_YEAR_MATRIX.csv",
            "START_MONTH_YEAR_MATRIX.csv","END_MONTH_YEAR_MATRIX.csv",
            "SEASON_LENGTH_EXACT_YEAR_MATRIX.csv","SEASON_LENGTH_BINS_YEAR_MATRIX.csv",
            "TOTAL_EIP_DAYS_EXACT_YEAR_MATRIX.csv","TOTAL_EIP_DAYS_BINS_YEAR_MATRIX.csv",
            "YEAR_SUMMARY_STATS.csv","START_DATE_YEAR_MATRIX.csv","END_DATE_YEAR_MATRIX.csv",
        ]
        for fn in stage_a_files:
            p=counts_dir/fn
            if p.exists():
                sheet=("A_"+Path(fn).stem)[:31]
                pd.read_csv(p, low_memory=False).to_excel(xlw, sheet_name=sheet, index=False)

        if tables_dir.exists():
            for p in sorted(tables_dir.glob("*.csv")):
                sheet=("B_"+p.stem)[:31]
                try:
                    pd.read_csv(p, low_memory=False).to_excel(xlw, sheet_name=sheet, index=False)
                except Exception:
                    pass

        if embed_images and engine=="xlsxwriter":
            wb=xlw.book
            ws_ts=wb.add_worksheet("IMG_timeseries")
            ws_dl=wb.add_worksheet("IMG_deltas")
            ws_mp=wb.add_worksheet("IMG_maps_sample")

            r=0
            for p in sorted(plots_dir.glob("timeseries_*.png"))[:12]:
                ws_ts.insert_image(r,0,str(p)); r+=28
            for p in sorted(plots_dir.glob("overlay_timeseries_*.png"))[:8]:
                ws_ts.insert_image(r,0,str(p)); r+=28

            r=0
            for p in sorted(plots_dir.glob("delta_box_*.png"))[:18]:
                ws_dl.insert_image(r,0,str(p)); r+=28
            for p in sorted(plots_dir.glob("delta_hist_*.png"))[:12]:
                ws_dl.insert_image(r,0,str(p)); r+=28

            r=0
            candidates=[]
            candidates += sorted((maps_dir/"values").glob("*.png"))
            candidates += sorted((maps_dir/"deltas").glob("*.png"))
            for p in candidates[:16]:
                ws_mp.insert_image(r,0,str(p)); r+=28

        try:
            for ws in xlw.sheets.values():
                ws.set_zoom(110)
        except Exception:
            pass

    print(f"[Excel] wrote: {xlsx}")
    return xlsx

def write_stats_summary_xlsx(out_root: Path, window: int, variables: List[str]) -> Path:
    """Workbook mirroring win30_7a_summary.xlsx: INFO + Trend/Dist/VarRatio/NatSum per variable."""
    tables_dir = out_root / "graphs" / "tables"
    excel_dir = ensure_dir(out_root / "99_excel")
    xlsx = excel_dir / f"win{window}_8_summary.xlsx"

    engine = "xlsxwriter"
    try:
        import xlsxwriter  # noqa: F401
    except Exception:
        engine = "openpyxl"

    def _read_csv(p: Path) -> pd.DataFrame:
        try:
            return pd.read_csv(p, low_memory=False)
        except Exception:
            return pd.DataFrame()

    created = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

    with pd.ExcelWriter(xlsx, engine=engine) as xlw:
        pd.DataFrame([{"Created": created, "Window": int(window)}]).to_excel(xlw, sheet_name="INFO", index=False)

        for var in variables:
            df_tr = _read_csv(tables_dir / f"trend_stats_{var}.csv")
            if not df_tr.empty:
                keep = [c for c in ["mk_trend", "mk_p", "mk_z", "mk_tau", "sen_slope", "sen_intercept"] if c in df_tr.columns]
                df_tr[keep].to_excel(xlw, sheet_name=f"Trend_{var}"[:31], index=False)

            df_ds = _read_csv(tables_dir / f"distribution_stats_{var}.csv")
            if not df_ds.empty:
                keep = [c for c in ["year", "n", "skewness", "shapiro_stat", "shapiro_p", "dip_stat", "dip_p", "is_normal", "is_unimodal"] if c in df_ds.columns]
                df_ds[keep].to_excel(xlw, sheet_name=f"Dist_{var}"[:31], index=False)

            df_vr = _read_csv(tables_dir / f"variance_ratio_{var}.csv")
            if not df_vr.empty:
                keep = [c for c in ["var_method_noise", "var_year_signal", "variance_ratio", "percent_method_uncertainty"] if c in df_vr.columns]
                df_vr[keep].to_excel(xlw, sheet_name=f"VarRatio_{var}"[:31], index=False)

            df_nat = _read_csv(tables_dir / f"national_summary_{var}.csv")
            if not df_nat.empty:
                keep = [c for c in ["year", "median", "p25", "p75", "count"] if c in df_nat.columns]
                df_nat[keep].to_excel(xlw, sheet_name=f"NatSum_{var}"[:31], index=False)

        try:
            for ws in xlw.sheets.values():
                ws.set_zoom(110)
        except Exception:
            pass

    print(f"[Summary] wrote: {xlsx}")
    return xlsx

# ---------------- path resolution ----------------
def resolve_paths(root: Path, output_root_override: Optional[str], in6_override: Optional[str], out_override: Optional[str],
                  methods_csv_override: Optional[str]) -> Dict[str, Path]:
    if output_root_override:
        out_base=Path(output_root_override)
    else:
        if cfg is not None and getattr(cfg,"DIR_METHOD_COMPARE",None):
            out_base=Path(str(getattr(cfg,"DIR_METHOD_COMPARE")))
        else:
            out_base=root/"out"/"12_method_compare"

    if methods_csv_override:
        methods_csv=Path(methods_csv_override)
    else:
        methods_csv=root/"out"/"08_season_swrap"/f"win{WIN}"/"csv"/"season_ALL_YEARS_methods.csv"
        if not methods_csv.exists():
            alt=root/"out"/"11_ddu_season_swrap"/f"win{WIN}"/"csv"/"season_ALL_YEARS_methods.csv"
            if alt.exists():
                methods_csv=alt

    in6_dir=find_in6_dir(out_base, in6_override)

    if out_override:
        out_dir=Path(out_override)
    else:
        out_dir=out_base/f"win{WIN}"/"8.per_year_per_pixel"

    return {"out_base":out_base,"methods_csv":methods_csv,"in6_dir":in6_dir,"out_dir":out_dir}

# ---------------- MAIN ----------------
def main():
    ap=argparse.ArgumentParser(description="WIN30 8.per_year_per_pixel — ONE RUN")

    ap.add_argument("--root", type=str, default="", help="Repo/project root (parent of out/). Default: auto-detect.")
    ap.add_argument("--years", type=str, default=None, help="2005-2024 or comma list.")
    ap.add_argument("--bbox", type=str, default=None, help="N,W,S,E (e.g., 48.5,19.9,43.5,30.1)")
    ap.add_argument("--methods", type=str, default=",".join(DEFAULT_METHODS))
    ap.add_argument("--variables", type=str, default="season_length,start_doy,end_doy,total_eip_days")

    ap.add_argument("--style", type=str, default="seaborn-v0_8-whitegrid")
    ap.add_argument("--fig-dpi", type=int, default=int(getattr(cfg,"PNG_DPI",180)) if cfg is not None else 180)
    ap.add_argument("--outline-shp", type=str, default=str(getattr(cfg,"SHP_RO","")) if cfg is not None and getattr(cfg,"SHP_RO",None) else "")

    ap.add_argument("--output-root", type=str, default=None, help="Override <ROOT>/out/12_method_compare")
    ap.add_argument("--in6", type=str, default=None, help="Override step-6 folder (default: auto 6.compare then 6A then 6)")
    ap.add_argument("--out", type=str, default=None, help="Override output dir (default: .../win30/8.per_year_per_pixel)")
    ap.add_argument("--methods-csv", type=str, default=None, help="Override season_ALL_YEARS_methods.csv path")

    # Stage toggles
    ap.add_argument("--no-stage-a", action="store_true")
    ap.add_argument("--no-stage-b", action="store_true")
    ap.add_argument("--no-stage-c", action="store_true")
    ap.add_argument("--no-excel", action="store_true")

    # Stage A params
    ap.add_argument("--length-bins", type=str, default=DEFAULT_LENGTH_BINS)
    ap.add_argument("--eip-bins", type=str, default=DEFAULT_EIP_BINS)
    ap.add_argument("--calendar-dates", action="store_true")

    # Stage B params
    ap.add_argument("--no-maps", action="store_true")
    ap.add_argument("--map-kind", type=str, default="grid", choices=["grid","tri","scatter"])
    ap.add_argument("--pair", type=str, default=None)
    ap.add_argument("--delta-flag", type=float, default=None)
    ap.add_argument("--no-per-method", action="store_true")
    ap.add_argument("--no-per-method-maps", action="store_true")
    ap.add_argument("--thr", type=str, default="5,10")
    ap.add_argument("--scatter-max", type=int, default=80000)

    # Stats: default ON (mandatory). Provide --no-stats to disable (not recommended).
    g=ap.add_mutually_exclusive_group()
    g.add_argument("--stats", dest="stats", action="store_true", help="Enable stats (default: ON)")
    g.add_argument("--no-stats", dest="stats", action="store_false", help="Disable stats")
    ap.set_defaults(stats=True)

    ap.add_argument("--stats-max-n", type=int, default=5000)

    # Excel images
    ap.add_argument("--embed-images", action="store_true")

    args=ap.parse_args()

    root = Path(args.root).expanduser().resolve() if args.root else REPO_ROOT

    years=parse_years(args.years)
    bbox=parse_bbox(args.bbox) if args.bbox else None
    methods=[m.strip() for m in args.methods.split(",") if m.strip()]
    variables=[v.strip() for v in args.variables.split(",") if v.strip() in VAR_ALIASES]
    pair_focus=parse_pair(args.pair) if args.pair else None

    thr_vals=[float(x) for x in args.thr.split(",") if x.strip()]
    if len(thr_vals)==0: thr_list=(5.0,10.0)
    elif len(thr_vals)==1: thr_list=(thr_vals[0],10.0)
    else: thr_list=(thr_vals[0],thr_vals[1])

    paths=resolve_paths(root, args.output_root, args.in6, args.out, args.methods_csv)
    methods_csv=paths["methods_csv"]
    in6_dir=paths["in6_dir"]
    out_dir=paths["out_dir"]

    # Mandatory deps for stats (unless --no-stats)
    if args.stats:
        if _scipy_skew is None or _scipy_shapiro is None:
            raise SystemExit("ERROR: SciPy is REQUIRED for stats. Install: pip install scipy")
        if diptest is None:
            raise SystemExit("ERROR: diptest is REQUIRED for Hartigan dip test. Install: pip install diptest")

    # create output structure
    ensure_dir(out_dir)
    ensure_dir(out_dir/"00_meta")
    ensure_dir(out_dir/"yr_counts")
    ensure_dir(out_dir/"graphs"/"tables")
    ensure_dir(out_dir/"graphs"/"plots")
    ensure_dir(out_dir/"graphs"/"quickmaps"/"values")
    ensure_dir(out_dir/"graphs"/"quickmaps"/"deltas")
    ensure_dir(out_dir/"graphs"/"quickmaps"/"per_method")
    ensure_dir(out_dir/"99_excel")

    outline_shp=args.outline_shp.strip() or None

    print(f"[CFG] WIN={WIN}")
    print(f"[CFG] ROOT={root}")
    print(f"[CFG] methods_csv={methods_csv}")
    print(f"[CFG] in6_dir={in6_dir}")
    print(f"[CFG] out_dir={out_dir}")
    print(f"[CFG] years={years if years else 'ALL'} bbox={bbox if bbox else 'None'}")
    print(f"[CFG] methods={methods} variables={variables}")
    print(f"[CFG] stats={'ON' if args.stats else 'OFF'} maps={'ON' if not args.no_maps else 'OFF'}")

    if not methods_csv.exists():
        raise FileNotFoundError(f"Missing methods CSV: {methods_csv}")

    # validate stage B input if enabled
    if not args.no_stage_b:
        if not in6_dir.exists():
            raise FileNotFoundError(f"Missing step-6 directory: {in6_dir}")
        test = list_peryear_files(in6_dir, "*_season_length_MEDIAN.csv") or list_peryear_files(in6_dir, "*_start_doy_MEDIAN.csv")
        if not test:
            raise FileNotFoundError(f"Stage B requires per-year *_MEDIAN.csv from step-6. None found under: {in6_dir}")

    # load step-5 methods csv
    methods_df=filter_roi(normalize_columns(pd.read_csv(methods_csv, low_memory=False)), bbox)
    if "year" in methods_df.columns:
        methods_df["year"]=pd.to_numeric(methods_df["year"], errors="coerce")
        methods_df=methods_df.dropna(subset=["year"]).copy()
        methods_df["year"]=methods_df["year"].astype(int)
    if years is not None and "year" in methods_df.columns:
        methods_df=methods_df[methods_df["year"].isin(years)].copy()

    meta={
        "script":"step08_per_year_per_pixel.py (win30) — FINAL",
        "win":WIN,
        "root":str(root),
        "methods_csv":str(methods_csv),
        "in6_dir":str(in6_dir),
        "out_dir":str(out_dir),
        "years":years,
        "bbox":bbox,
        "methods":methods,
        "variables":variables,
        "pair_focus":list(pair_focus) if pair_focus else None,
        "delta_flag":args.delta_flag,
        "style":args.style,
        "fig_dpi":args.fig_dpi,
        "map_kind":args.map_kind,
        "outline_shp":outline_shp,
        "calendar_dates":bool(args.calendar_dates),
        "thr":list(thr_list),
        "stats":bool(args.stats),
        "stats_max_n":int(args.stats_max_n),
        "scatter_max":int(args.scatter_max),
        "timestamp":datetime.utcnow().isoformat(timespec="seconds")+"Z",
    }
    (out_dir/"00_meta"/"META_8.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # Stage A
    if not args.no_stage_a:
        stage_a_year_counts(
            methods_df=methods_df,
            out_dir=out_dir/"yr_counts",
            years=years,
            methods=methods,
            length_bins_spec=args.length_bins,
            eip_bins_spec=args.eip_bins,
            calendar_dates=bool(args.calendar_dates),
        )

    # Stage B
    if not args.no_stage_b:
        stage_b_graphs_maps(
            in6_dir=in6_dir,
            methods_df=methods_df,
            out_dir=out_dir,
            years=years,
            variables=variables,
            methods=methods,
            bbox=bbox,
            outline_shp=outline_shp,
            style=args.style,
            fig_dpi=int(args.fig_dpi),
            map_kind=args.map_kind,
            no_maps=bool(args.no_maps),
            pair_focus=pair_focus,
            delta_flag=args.delta_flag,
            no_per_method=bool(args.no_per_method),
            no_per_method_maps=bool(args.no_per_method_maps),
            thr_list=(thr_list[0],thr_list[1]),
            stats=bool(args.stats),
            stats_max_n=int(args.stats_max_n),
            scatter_max=int(args.scatter_max),
        )

    # Stage C
    if not args.no_stage_c:
        stage_c_counts_timeseries(out_dir, args.style, int(args.fig_dpi))

    # Stage X
    if not args.no_excel:
        write_central_excel(out_dir, meta=meta, embed_images=bool(args.embed_images))

    # Stage S
    if args.stats and (not args.no_excel):
        write_stats_summary_xlsx(out_dir, window=WIN, variables=variables)

    print("DONE.")

if __name__=="__main__":
    main()
