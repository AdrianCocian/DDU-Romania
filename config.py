# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Tuple, Any

import os
import sys
import json
import logging
import numpy as np


def use_utf8_output() -> None:
    """Write console output as UTF-8 on every platform.

    The steps print symbols and diacritics; when the output of a Windows process is piped or redirected, Python
    encodes it with the ANSI code page, which cannot represent them and stops the step with UnicodeEncodeError."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


use_utf8_output()


def _env_first(*keys: str) -> Optional[str]:
    for k in keys:
        v = os.environ.get(k)
        if v is not None:
            v = str(v).strip()
            if v:
                return v
    return None


def _as_path(p: str | Path) -> Path:
    s = os.path.expandvars(str(p))
    return Path(s).expanduser().resolve(strict=False)


def _norm_mkdir_mode(v: Optional[str]) -> str:
    v = (v or "base").strip().lower()
    if v not in {"none", "base", "all"}:
        raise ValueError(f"[CFG][ERROR] DDU_MKDIR invalid: {v!r} (allowed: none|base|all)")
    return v


def _to_jsonable(x: Any) -> Any:
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(i) for i in x]
    if isinstance(x, dict):
        return {str(k): _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (np.integer, np.floating)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


def _default_shp_candidates(root: Path) -> List[Path]:
    code_root = Path(__file__).resolve().parent   # the boundary shipped with the repository
    cands = []
    for r in (root, code_root):
        cands += [
            r / "data" / "boundaries" / "ro_frontiera_poligon" / "ro_frontiera_poligon.shp",
            r / "data" / "boundaries" / "ro_frontiera_poligon.shp",
            r / "ro_frontiera_poligon.shp",
            r / "ro_frontiera_poligon" / "ro_frontiera_poligon.shp",
        ]
    return list(dict.fromkeys(cands))


def _pick_default_shp(root: Path) -> Path:
    for p in _default_shp_candidates(root):
        if p.exists():
            return p
    return _default_shp_candidates(root)[0]


@dataclass
class CFG:
    YEARS: List[int] = field(default_factory=lambda: list(range(2005, 2025)))
    MONTHS: List[str] = field(default_factory=lambda: [f"{m:02d}" for m in range(1, 13)])

    CDS_DATASET: str = "reanalysis-era5-land"
    CDS_VAR: str = "2m_temperature"
    CDS_FORMAT: str = "netcdf"
    CDS_TIMEOUT: int = 600
    CDS_VERIFY_SSL: bool = True

    BBOX: List[float] = field(default_factory=lambda: [48.5, 19.9, 43.5, 30.1])  # N,W,S,E
    BBOX_ROUND_STEP: float = 0.1
    BBOX_ROUNDED: List[float] = field(init=False)

    DL_GRID: List[float] = field(default_factory=lambda: [0.1, 0.1])  # dlat, dlon (deg)

    ROOT: Path = field(default_factory=lambda: _as_path(_env_first("DDU_ROOT") or Path(__file__).resolve().parent))
    SHP_RO: Optional[Path] = field(default=None)
    SHP_CRS: Optional[str] = None

    TBASE_C: float = 14.0
    WINDOW_DAYS: int = 30
    THRESH_DD_30D: float = 130.0
    THRESH_DH_30D: float = 3120.0
    THRESH_DD_15D: float = 130.0
    THRESH_DH_15D: float = 3120.0

    # Season rule (manuscript section 2.3): a transmission season is a run of suitable days that
    # ends when THREE or more consecutive days are unsuitable; interruptions of one or two unsuitable
    # days stay inside the season ("3-day rule", a grouping convention: it changes the reported season
    # span, not whether a single day meets the 130-DDU/30-day criterion).
    # Implementation (scripts/step05_ddu_season_wrap.py, split_runs): a new run starts when the next
    # suitable day is >= MIN_GAP_DAYS days after the previous suitable day, so at most MIN_GAP_DAYS - 2
    # unsuitable days are bridged.  MIN_GAP_DAYS = 4  <=>  gaps of 1-2 days bridged, >= 3 days split.
    # (The version on 'master' before revision 2 used 10, i.e. up to 8 bridged days.)
    MIN_GAP_DAYS: int = field(default_factory=lambda: int(_env_first("DDU_MIN_GAP") or 4))

    # Methods (pipeline keys are kept for file compatibility; labels are the names used in the paper)
    METHODS: List[str] = field(default_factory=lambda: ["midpoint", "daily_mean", "hourly_sum", "be_sine"])
    METHOD_LABELS: Dict[str, str] = field(default_factory=lambda: {
        "midpoint": "Min–max average",      # (Tmin + Tmax) / 2
        "daily_mean": "Hourly average",          # mean of the 24 hourly temperatures
        "hourly_sum": "Hourly sum",              # reference method
        "be_sine": "BE-sine",                    # Baskerville-Emin sine
        "consensus": "Consensus",                # per-cell median across the four methods
    })
    METHOD_COLORS: Dict[str, str] = field(default_factory=lambda: {
        "midpoint": "#1f77b4", "daily_mean": "#ff7f0e", "hourly_sum": "#2ca02c", "be_sine": "#d62728",
        "consensus": "#000000",
    })
    REF_METHOD: str = "hourly_sum"
    # Pairs for the paired differences, delta = method1 - method2 (order of Tables 4, S2, S4, S7)
    PAIRS: List[Tuple[str, str]] = field(default_factory=lambda: [
        ("be_sine", "hourly_sum"), ("daily_mean", "be_sine"), ("daily_mean", "hourly_sum"),
        ("midpoint", "be_sine"), ("midpoint", "daily_mean"), ("midpoint", "hourly_sum"),
    ])
    REFERENCE_YEARS: List[int] = field(default_factory=lambda: [2005, 2011, 2017, 2024])

    # Relative climatic suitability classes (Table 2): quantile bands of hourly-sum season length
    # among cell-years with a season, pooled over all years (P25, P50, P75, P90); class 0 = no season.
    CLASS_PERCENTILES: List[float] = field(default_factory=lambda: [25, 50, 75, 90])
    CLASS_LABELS: List[str] = field(default_factory=lambda: [
        "0 no season", "1 lowest", "2 below median", "3 above median", "4 high", "5 highest"])
    CLASS_COLORS: List[str] = field(default_factory=lambda: [
        "#e6e6e6", "#4da3ff", "#66c2a5", "#a6d854", "#fdae61", "#d73027"])

    VALID_RANGE_TEMP: Tuple[float, float] = (-60.0, 60.0)
    FILL_PCT_LOW_CONF_THRESH: float = 0.30

    CONVENTIONS: str = "CF-1.10"
    GRID_MAPPING_NAME: str = "latitude_longitude"
    COORD_UNITS: Dict[str, str] = field(default_factory=lambda: {"lat": "degrees_north", "lon": "degrees_east"})
    TIME_CALENDAR: str = "standard"
    CRS_EPSG: int = 4326

    AUTO_DETECT_YEARS: bool = False
    REQUIRE_SHP: bool = True

    USE_YEAR_TAILS: bool = True
    USE_YEAR_WRAP: bool = True
    YEAR_WRAP_START: int = 2006
    VALIDATE_TAIL: bool = True

    CMAP_NAME: str = "RdBu_r"
    NODATA_COLOR_HEX: str = "#FFF8B5"
    IMAGE_INTERPOLATION: str = "nearest"
    PNG_DPI: int = 160

    FILLVALUE_FLOAT: float = -9999.0

    CLIP_ALL_TOUCHED: bool = True
    MASK_ALL_TOUCHED: bool = True
    INSIDE_EXPECTED: int = 2912
    MASK_ACCEPT_NEAREST: bool = True

    PLOT_RANGES: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "eip30_dd": {"vmin": 0, "vmax": 130, "ticks": [0, 30, 60, 90, 120, 130], "cmap": "sequential"},
        "eip30_dh": {"vmin": 0, "vmax": 3120, "ticks": [0, 780, 1560, 2340, 3120], "cmap": "sequential"},
        "total_eip_year": {"vmin": 0, "vmax": 365, "ticks": [0, 50, 100, 150, 200, 300, 365], "cmap": "sequential"},
        "season_length_days": {"vmin": 0, "vmax": 170, "ticks": [0, 30, 60, 90, 120, 150, 170], "cmap": "sequential"},
        "main_start_doy": {"vmin": 60, "vmax": 220, "ticks": [60, 100, 140, 180, 220], "cmap": "sequential"},
        "main_end_doy": {"vmin": 120, "vmax": 300, "ticks": [120, 180, 220, 260, 300], "cmap": "sequential"},
    })

    SKIP_IF_EXISTS: bool = True
    FORCE: bool = False

    MASK_OUTSIDE_IN_YEARLY: bool = True
    MASK_OUTSIDE_IN_CLIM: bool = True

    DIR_DL: Path = field(init=False)
    DIR_STD: Path = field(init=False)
    DIR_CLIP: Path = field(init=False)
    DIR_YEAR: Path = field(init=False)
    DIR_CLIM: Path = field(init=False)

    DIR_FILLED: Path = field(init=False)
    DIR_FILLED_TAILS: Path = field(init=False)

    DIR_REP: Path = field(init=False)
    DIR_MASKS: Path = field(init=False)
    REF_MASK_PATH: Path = field(init=False)

    DIR_SEASON_SWRAP: Path = field(init=False)
    DIR_SEASON_SWRAP_NC: Path = field(init=False)
    DIR_SEASON_SWRAP_CSV: Path = field(init=False)
    DIR_SEASON_SWRAP_PNG: Path = field(init=False)

    DIR_DDU_SEASON: Path = field(init=False, repr=False)
    DIR_DDU_SEASON_WRAP: Path = field(init=False, repr=False)
    DIR_METHOD_COMPARE: Path = field(init=False)

    # Sub-folders of <ROOT>/out/12_method_compare/win<W>/ written by steps 06-10
    STEP6_DIRNAME: str = "6_method_comparison"
    STEP7_DIRNAME: str = "7_suitability_classes"
    STEP8_DIRNAME: str = "8_statistics"
    PACK_DIRNAME: str = "00_MANUSCRIPT_PACK"

    NC_NAME_FMT: str = "season_swrap_{year}_{method}.nc"
    CSV_NAME_FMT: str = "season_swrap_{year}_{method}.csv"
    PNG_NAME_FMT: str = "season_swrap_{year}_{method}.png"

    RUN_TS: str = field(init=False)
    _logger: logging.Logger = field(init=False, repr=False, compare=False)

    MKDIR_MODE: str = field(default_factory=lambda: _norm_mkdir_mode(_env_first("DDU_MKDIR")))
    _created: List[Path] = field(init=False, default_factory=list, repr=False)

    def rounded_bbox(self) -> List[float]:
        N, W, S, E = self.BBOX
        step = float(self.BBOX_ROUND_STEP)

        def q(v: float) -> float:
            return round(round(v / step) * step, 6)

        return [q(N), q(W), q(S), q(E)]

    def master_lat_lon(self):
        N, W, S, E = self.BBOX_ROUNDED
        dlat, dlon = self.DL_GRID
        lat = np.round(np.arange(S, N + 1e-9, dlat), 6)
        lon = np.round(np.arange(W, E + 1e-9, dlon), 6)
        return lat, lon

    def global_attrs(self) -> Dict[str, Any]:
        return {
            "Conventions": self.CONVENTIONS,
            "title": "DDU / EIP processing for Romania",
            "history": f"Created {self.RUN_TS}",
            "source": self.CDS_DATASET,
            "grid_mapping_name": self.GRID_MAPPING_NAME,
            "crs_epsg": int(self.CRS_EPSG),
        }

    def ensure(self, *dirs: Path, verbose: bool = True) -> List[Path]:
        made = []
        for d in dirs:
            p = Path(d)
            p.mkdir(parents=True, exist_ok=True)
            made.append(p)
        if verbose and made:
            print("[CFG] ensured:", ", ".join(str(x) for x in made))
        return made

    @property
    def EIP_WINDOW_DAYS(self) -> int:
        return int(self.WINDOW_DAYS)

    @property
    def DD_THRESH(self) -> float:
        if self.WINDOW_DAYS == 30:
            return float(self.THRESH_DD_30D)
        if self.WINDOW_DAYS == 15:
            return float(self.THRESH_DD_15D)
        raise ValueError(f"WINDOW_DAYS unsupported for DD_THRESH: {self.WINDOW_DAYS}")

    @property
    def DH_THRESH(self) -> float:
        if self.WINDOW_DAYS == 30:
            return float(self.THRESH_DH_30D)
        if self.WINDOW_DAYS == 15:
            return float(self.THRESH_DH_15D)
        raise ValueError(f"WINDOW_DAYS unsupported for DH_THRESH: {self.WINDOW_DAYS}")

    @property
    def DIR_RAW_YEARLY(self) -> Path:
        return self.DIR_YEAR

    @property
    def DIR_REPORTS(self) -> Path:
        return self.DIR_REP

    def __post_init__(self):
        if not logging.getLogger().handlers:
            logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
        self._logger = logging.getLogger("DDU_CFG")

        self.RUN_TS = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.ROOT = _as_path(self.ROOT)

        shp_env = _env_first("DDU_SHP_RO")
        if shp_env:
            self.SHP_RO = _as_path(shp_env)
        elif self.SHP_RO is None:
            self.SHP_RO = _pick_default_shp(self.ROOT)
        else:
            self.SHP_RO = _as_path(self.SHP_RO)

        self.BBOX_ROUNDED = self.rounded_bbox()

        out = self.ROOT / "out"
        self.DIR_DL = out / "01_download"
        self.DIR_STD = out / "02_standardized"
        self.DIR_CLIP = out / "03_clip_ro"
        self.DIR_YEAR = out / "04_yearly_raw"
        self.DIR_CLIM = out / "05_climatology"

        self.DIR_FILLED = out / "06_filled_yearly"
        self.DIR_FILLED_TAILS = out / "06_filled_yearly_tails_29d"

        self.DIR_REP = out / "07_reports"
        self.DIR_MASKS = self.DIR_REP / "_inside_masks"
        self.REF_MASK_PATH = self.DIR_MASKS / "inside_mask_REF.nc"

        self.DIR_SEASON_SWRAP = out / "08_season_swrap"
        self.DIR_SEASON_SWRAP_NC = self.DIR_SEASON_SWRAP / "nc"
        self.DIR_SEASON_SWRAP_CSV = self.DIR_SEASON_SWRAP / "csv"
        self.DIR_SEASON_SWRAP_PNG = self.DIR_SEASON_SWRAP / "png"

        self.DIR_DDU_SEASON = out / "11_ddu_season"
        self.DIR_DDU_SEASON_WRAP = out / "11_ddu_season_wrap"
        self.DIR_METHOD_COMPARE = out / "12_method_compare"

        all_dirs = [
            out,
            self.DIR_DL, self.DIR_STD, self.DIR_CLIP, self.DIR_YEAR, self.DIR_CLIM,
            self.DIR_FILLED, self.DIR_FILLED_TAILS,
            self.DIR_REP, self.DIR_MASKS,
            self.DIR_SEASON_SWRAP, self.DIR_SEASON_SWRAP_NC, self.DIR_SEASON_SWRAP_CSV, self.DIR_SEASON_SWRAP_PNG,
            self.DIR_METHOD_COMPARE,
        ]

        if self.MKDIR_MODE == "all":
            for d in all_dirs:
                d.mkdir(parents=True, exist_ok=True)
                self._created.append(d)
        elif self.MKDIR_MODE == "base":
            out.mkdir(parents=True, exist_ok=True)
            self._created.append(out)

        n, w, s, e = self.BBOX
        if not (n > s and e > w):
            raise ValueError("BBOX must be N,W,S,E with N>S and E>W")
        if not (-90 <= n <= 90 and -90 <= s <= 90 and -180 <= w <= 180 and -180 <= e <= 180):
            raise ValueError("BBOX values out of range")

        if not Path(self.SHP_RO).exists():
            cands = _default_shp_candidates(self.ROOT)
            msg = (
                f"[CFG][ERROR] Romania shapefile not found: {self.SHP_RO}\n"
                + "Tried:\n" + "\n".join([f"  - {p}" for p in cands])
                + "\nFix: place the shapefile in one of the locations above OR set DDU_SHP_RO=<path_to_shp>."
            )
            if self.REQUIRE_SHP:
                raise FileNotFoundError(msg)
            self._logger.warning(msg)

        if self.AUTO_DETECT_YEARS and self.DIR_YEAR.exists():
            cand: List[int] = []
            for p in self.DIR_YEAR.glob("*.nc"):
                stem = p.stem
                if stem.isdigit() and len(stem) == 4:
                    cand.append(int(stem))
            if cand:
                self.YEARS = sorted(set(cand))

        if int(self.MIN_GAP_DAYS) < 2:
            raise ValueError(f"[CFG][ERROR] MIN_GAP_DAYS must be >= 2 (got {self.MIN_GAP_DAYS})")

        print(f"[CFG] ROOT={self.ROOT} | SHP={self.SHP_RO} | YEARS={self.YEARS[0]}-{self.YEARS[-1]} | WIN={self.WINDOW_DAYS}d"
              f" | MIN_GAP_DAYS={self.MIN_GAP_DAYS} (season ends after >= {self.MIN_GAP_DAYS - 1} consecutive"
              f" unsuitable days)")

        if self.MKDIR_MODE in {"base", "all"}:
            try:
                testfile = self.ROOT / ".cfg_write_test"
                with open(testfile, "w", encoding="utf-8") as f:
                    f.write("ok")
                os.remove(testfile)
            except Exception as e:
                raise PermissionError(f"[CFG][ERROR] No write permission in {self.ROOT}: {e}")

    def save_run_meta(self, where: Path | None = None) -> Path:
        meta_dir = (where or (self.ROOT / "out")) / "_run_meta"
        meta_dir.mkdir(parents=True, exist_ok=True)
        meta_path = meta_dir / f"run_{self.RUN_TS}.json"

        payload = asdict(self)
        payload.pop("_logger", None)
        payload["_created_dirs"] = [str(p) for p in getattr(self, "_created", [])]
        payload = _to_jsonable(payload)

        payload.update({
            "_python": sys.version,
            "_platform": sys.platform,
            "_cwd": str(Path.cwd()),
            "_env_overrides": {
                "DDU_ROOT": os.environ.get("DDU_ROOT"),
                "DDU_SHP_RO": os.environ.get("DDU_SHP_RO"),
                "DDU_MKDIR": os.environ.get("DDU_MKDIR"),
                "DDU_MIN_GAP": os.environ.get("DDU_MIN_GAP"),
            },
            "_timestamp": self.RUN_TS,
        })

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        print(f"[CFG] Saved run meta → {meta_path}")
        return meta_path

    def inside_expected(self) -> int:
        return int(self.INSIDE_EXPECTED)


cfg = CFG()
