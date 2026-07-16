#!/usr/bin/env python3
"""
04_script_trial_synchrony_lena_notebooks4_4b_EG.py
By: Elnaz Ghasemi, adapted from Lena Adel's tutorial_dyadic_movesense
Date: June 2026

Purpose
-------
Step 4 of Veronica/Cirkeline TableTask Movesense ECG/HR pipeline.

This script computes dyadic synchrony from the Step 3 paired trial-level 4 Hz HR
files and saves the plot families used in Lena's Notebook 04 and 04b.

Correct pipeline position
-------------------------
Step 1  : metadata/timing/QC master file.
Step 2  : raw ECG -> corrected RR and corrected 4 Hz HR.
Step 2b : pretrial 2-minute QC.
Step 3  : paired trial/window HR files on a common 4 Hz grid.
Step 3a : RR odd-even reliability QC.
Step 4  : THIS SCRIPT. Trial-level dyadic synchrony metrics and plots.

Important boundary
------------------
This script does not rerun ECG processing and does not need the raw
"Table MoveSense Sensor" folder. It uses the already segmented paired trial HR
files from Step 3 for synchrony. For Lena Notebook 04b Poincare/Bland-Altman,
it preferentially uses Step 2 corrected beat-to-beat RR files, matching Lena
more closely than RR reconstructed from interpolated HR.

Inputs expected when run from the top-level TableTask folder
-----------------------------------------------------------
processed_ecg_hr/trial_segments_4hz/paired_trial_hr/*.csv
processed_ecg_hr/trial_segments_4hz/qc/03_trial_segmentation_summary.csv
processed_ecg_hr/rr_reliability_qc/qc/3a_trial_dyad_rr_reliability.csv
qc_outputs/veronica_analysis_units.csv

Main outputs
------------
processed_ecg_hr/trial_synchrony_lena_step4_EG/
  tables/
    04_step4_trial_level_synchrony_metrics_EG.csv/.xlsx
    04_step4_window_level_synchrony_metrics_EG.csv/.xlsx
    04_step4_condition_level_summary_all_windows_EG.csv/.xlsx
    04_step4_condition_level_summary_main_analysis_EG.csv/.xlsx
    04_step4_file_manifest_EG.csv
    04_step4_failures_EG.csv
    04_step4_run_summary_EG.txt
  plots/
    lena_04_windowed_cardiac_synchrony/
    lena_04b_comprehensive_poincare/
    lena_04b_overlaid_ellipses/
    lena_04b_temporal_poincare_patterns/

How to run
----------
cd /Users/e/Desktop/veronica_project/TableTask

python3 04_script_trial_synchrony_lena_notebooks4_4b_EG.py \
  --root . \
  --overwrite

For a fast test that computes all tables but saves only the first 10 trials' Lena-style plots:

python3 04_script_trial_synchrony_lena_notebooks4_4b_EG.py \
  --root . \
  --max-trial-plots 10 \
  --overwrite

Dependencies
------------
Required: pandas, numpy, scipy, matplotlib, openpyxl
Install if needed:
python3 -m pip install pandas numpy scipy matplotlib openpyxl
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# Avoid open-figure accumulation warnings during large batch plot runs.
plt.rcParams["figure.max_open_warning"] = 0

from scipy import signal as sp_signal
from scipy import stats

try:
    from statsmodels.tsa.stattools import grangercausalitytests
    GRANGER_AVAILABLE = True
except Exception:
    GRANGER_AVAILABLE = False

# Optional, matching Lena Notebook 04 CRQA behavior. If PyRQA is unavailable,
# CRQA columns are returned as NaN/unavailable rather than failing Step 4.
try:
    from pyrqa.time_series import TimeSeries
    from pyrqa.settings import Settings
    from pyrqa.analysis_type import Cross
    from pyrqa.neighbourhood import FixedRadius
    from pyrqa.metric import EuclideanMetric
    from pyrqa.computation import RQAComputation
    PYRQA_AVAILABLE = True
except Exception:
    PYRQA_AVAILABLE = False

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

SCRIPT_VERSION = "v14_step4_lena04_04b_lena_exact_logic_no_redundant_plot_folders"
DEFAULT_WINDOW_SEC = 30.0
DEFAULT_WINDOW_OVERLAP = 0.50
DEFAULT_MAX_LAG_SEC = 10.0
DEFAULT_MIN_VALID_SAMPLES = 20
DEFAULT_MIN_WINDOW_SAMPLES = 20
DEFAULT_TARGET_FS = 4.0
EPS = 1e-12


# =============================================================================
# Command-line interface
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 4: compute TableTask dyadic synchrony metrics and save Lena Notebook 04/04b style plots."
    )
    parser.add_argument("--root", type=str, default=".", help="Top-level TableTask folder. Default: current folder.")
    parser.add_argument(
        "--paired-hr-dir",
        type=str,
        default="processed_ecg_hr/trial_segments_4hz/paired_trial_hr",
        help="Folder containing Step 3 paired trial HR CSV files.",
    )
    parser.add_argument(
        "--trial-summary",
        type=str,
        default="processed_ecg_hr/trial_segments_4hz/qc/03_trial_segmentation_summary.csv",
        help="Step 3 trial segmentation summary CSV.",
    )
    parser.add_argument(
        "--reliability-summary",
        type=str,
        default="processed_ecg_hr/rr_reliability_qc/qc/3a_trial_dyad_rr_reliability.csv",
        help="Step 3a dyad-level RR reliability CSV. Optional but strongly recommended.",
    )
    parser.add_argument(
        "--analysis-units",
        type=str,
        default="qc_outputs/veronica_analysis_units.csv",
        help="Step 1 analysis-unit CSV. Optional metadata fallback.",
    )
    parser.add_argument(
        "--rr-dir",
        type=str,
        default="processed_ecg_hr/rpeak_rr_corrected",
        help=(
            "Folder containing Step 2 corrected beat-to-beat RR files. Used for Lena-exact "
            "Notebook 04b Poincare/Bland-Altman plots. If not found, script falls back to "
            "RR-equivalent values from Step 3 HR unless --require-direct-rr-for-poincare is set."
        ),
    )
    parser.add_argument(
        "--require-direct-rr-for-poincare",
        action="store_true",
        help="Fail a trial if Step 2 beat-to-beat RR cannot be found for Poincare/Bland-Altman.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="processed_ecg_hr/trial_synchrony_lena_step4_EG",
        help="Step 4 output folder.",
    )
    parser.add_argument("--window-sec", type=float, default=DEFAULT_WINDOW_SEC, help="Window duration for windowed synchrony, seconds.")
    parser.add_argument("--window-overlap", type=float, default=DEFAULT_WINDOW_OVERLAP, help="Fractional window overlap, e.g. 0.5.")
    parser.add_argument("--max-lag-sec", type=float, default=DEFAULT_MAX_LAG_SEC, help="Maximum lag for cross-correlation, seconds.")
    parser.add_argument("--min-valid-samples", type=int, default=DEFAULT_MIN_VALID_SAMPLES, help="Minimum valid aligned samples per trial.")
    parser.add_argument("--min-window-samples", type=int, default=DEFAULT_MIN_WINDOW_SAMPLES, help="Minimum valid samples per window.")
    parser.add_argument(
        "--max-trial-plots",
        type=int,
        default=-1,
        help=(
            "Maximum number of trials for which to save Lena-style trial plots. "
            "Default -1 saves all trials. Use 0 to skip trial-level Lena plots."
        ),
    )
    parser.add_argument("--skip-group-plots", action="store_true", help="Skip group summary plots.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing Step 4 output folder.")
    return parser.parse_args()


def resolve_path(root: Path, text: str) -> Path:
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()


# =============================================================================
# Basic utilities
# =============================================================================


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    return out


def safe_piece(value: Any) -> str:
    text = "" if value is None or (isinstance(value, float) and np.isnan(value)) else str(value)
    text = text.strip()
    if text.lower() in {"", "nan", "none", "nat"}:
        text = "NA"
    text = re.sub(r"[^A-Za-z0-9_\-]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_") or "NA"


def bool_from_any(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    if isinstance(value, (int, np.integer, float, np.floating)):
        return bool(value)
    return str(value).strip().lower() in {"true", "t", "yes", "y", "1"}


def numeric_or_nan(value: Any) -> float:
    try:
        out = pd.to_numeric(value, errors="coerce")
        return float(out) if pd.notna(out) else np.nan
    except Exception:
        return np.nan


def write_png(fig: plt.Figure, path: Path, dpi: int = 160) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
    finally:
        plt.close(fig)
        plt.close("all")
    return str(path)


def zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = np.full_like(x, np.nan, dtype=float)
    mask = np.isfinite(x)
    if mask.sum() < 2:
        return out
    mu = np.nanmean(x[mask])
    # Match scipy.stats.zscore default used in Lena Notebook 04 (population SD, ddof=0).
    sd = np.nanstd(x[mask], ddof=0)
    if not np.isfinite(sd) or sd <= 0:
        return out
    out[mask] = (x[mask] - mu) / sd
    return out


def interp_nan_linear(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if np.isfinite(x).all():
        return x
    idx = np.arange(len(x))
    mask = np.isfinite(x)
    if mask.sum() < 2:
        return x
    out = x.copy()
    out[~mask] = np.interp(idx[~mask], idx[mask], x[mask])
    return out


def pearson_safe(x: np.ndarray, y: np.ndarray) -> tuple[float, float, int]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    n = int(mask.sum())
    if n < 3:
        return np.nan, np.nan, n
    x2 = x[mask]
    y2 = y[mask]
    if np.nanstd(x2, ddof=1) <= 0 or np.nanstd(y2, ddof=1) <= 0:
        return np.nan, np.nan, n
    r, p = stats.pearsonr(x2, y2)
    return float(r), float(p), n


def spearman_safe(x: np.ndarray, y: np.ndarray) -> tuple[float, float, int]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    n = int(mask.sum())
    if n < 3:
        return np.nan, np.nan, n
    x2 = x[mask]
    y2 = y[mask]
    if np.nanstd(x2, ddof=1) <= 0 or np.nanstd(y2, ddof=1) <= 0:
        return np.nan, np.nan, n
    r, p = stats.spearmanr(x2, y2)
    return float(r), float(p), n


def circular_mean_rad(angles: np.ndarray) -> float:
    angles = np.asarray(angles, dtype=float)
    angles = angles[np.isfinite(angles)]
    if len(angles) == 0:
        return np.nan
    return float(np.angle(np.mean(np.exp(1j * angles))))


def mean_or_nan(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if not np.isfinite(x).any():
        return np.nan
    return float(np.nanmean(x))


def sd_or_nan(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if np.isfinite(x).sum() < 2:
        return np.nan
    return float(np.nanstd(x, ddof=1))


# =============================================================================
# Input readers
# =============================================================================


def read_optional_csv(path: Path, dtype: Optional[dict[str, Any]] = None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, dtype=dtype)
    return clean_column_names(df)


def read_paired_hr_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"recording_folder": str, "dyad_id": str, "sensor_A": str, "sensor_B": str})
    df = clean_column_names(df)
    required = ["TimeRelTrialSec", "HR_A_BPM", "HR_B_BPM"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns {missing}")
    for col in ["TimeUnix", "TimeFromAccelStartSec", "TimeRelTrialSec", "HR_A_BPM", "HR_B_BPM"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["TimeRelTrialSec", "HR_A_BPM", "HR_B_BPM"]).copy()
    df = df.sort_values("TimeRelTrialSec").reset_index(drop=True)
    return df



# =============================================================================
# Lena Notebook 04b RR utilities for Poincare/Bland-Altman
# =============================================================================

RR_VALUE_CANDIDATES = [
    "RR_ms", "RR_interval_ms", "RR_Interval_ms", "RR_corrected_ms", "RR_Corrected_ms",
    "Corrected_RR_ms", "corrected_rr_ms", "RR", "rr_ms", "rr_interval_ms",
    "RRIntervalMs", "rr", "ibi_ms", "IBI_ms", "NN_ms", "NN_interval_ms",
]

RR_TIME_CANDIDATES = [
    # Step 2 corrected RR files use rr_time_unix. Put these exact Unix-time
    # columns first so the fallback does not accidentally choose relative
    # rpeak_time_rel_* columns just because they contain the substring "time".
    "rr_time_unix", "RR_TimeUnix", "RRTimeUnix",
    "rpeak_time_unix_start", "rpeak_time_unix_end",
    "TimeUnix", "time_unix", "UnixTime", "timestamp_unix", "TimestampUnix",
    "Timestamp", "timestamp", "Time", "time", "time_s", "TimeSec", "seconds",
]


def find_first_column(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    cols = {str(c).strip(): c for c in df.columns}
    low = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        if cand in cols:
            return str(cols[cand])
        if cand.lower() in low:
            return str(low[cand.lower()])
    # Fallback: find a plausible RR/time column by substring.
    for c in df.columns:
        cl = str(c).lower()
        if any(cand.lower() in cl for cand in candidates):
            return str(c)
    return None


def build_rr_file_cache(rr_dir: Path) -> list[dict[str, Any]]:
    """Read Step 2 corrected RR CSV files for Lena-exact Notebook 04b plots.

    Lena Notebook 04b loads beat-to-beat RR interval files directly from Notebook 3.
    This cache is the Veronica equivalent: Step 2 corrected RR files. We keep the
    function tolerant to column names because previous Step 2 script versions may
    name RR/time columns slightly differently.
    """
    if not rr_dir.exists():
        return []
    cache: list[dict[str, Any]] = []
    for p in sorted(rr_dir.rglob("*.csv")):
        if p.name.startswith("._"):
            continue
        try:
            df = clean_column_names(pd.read_csv(p))
        except Exception:
            continue
        rr_col = find_first_column(df, RR_VALUE_CANDIDATES)
        if rr_col is None:
            continue
        time_col = find_first_column(df, RR_TIME_CANDIDATES)
        cache.append({"path": p, "name_lower": str(p).lower(), "df": df, "rr_col": rr_col, "time_col": time_col})
    return cache


def metadata_match_score(item: dict[str, Any], meta: dict[str, Any], side: str) -> int:
    text = item["name_lower"]
    score = 0
    # Strong file-name matches. These are intentionally additive.
    for key, weight in [
        ("recording_folder", 5),
        ("dyad_id", 2),
        (f"sensor_{side}", 5),
        (f"participant_{side}", 3),
    ]:
        val = str(meta.get(key, "")).strip()
        if val and val.lower() not in {"nan", "none", "na"} and val.lower() in text:
            score += weight
    # Also check metadata columns inside the RR file if present.
    df = item.get("df", pd.DataFrame())
    for key, weight in [
        ("recording_folder", 5),
        (f"sensor_{side}", 5),
        (f"participant_{side}", 3),
    ]:
        val = str(meta.get(key, "")).strip()
        if not val or val.lower() in {"nan", "none", "na"}:
            continue
        for col in df.columns:
            cl = str(col).lower()
            if any(tok in cl for tok in ["recording", "folder", "sensor", "participant", "subject", "id"]):
                try:
                    vals = set(df[col].dropna().astype(str).str.strip().str.lower().head(50))
                except Exception:
                    vals = set()
                if val.lower() in vals:
                    score += weight
                    break
    return score


def extract_trial_rr_from_item(item: dict[str, Any], meta: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    df = item["df"]
    rr = pd.to_numeric(df[item["rr_col"]], errors="coerce").to_numpy(dtype=float)
    # Step 2 RR is already corrected; only remove impossible/nonfinite values here.
    rr[(~np.isfinite(rr)) | (rr <= 0) | (rr > 3000)] = np.nan
    time_col = item.get("time_col")
    if time_col is not None and time_col in df.columns:
        t = pd.to_numeric(df[time_col], errors="coerce").to_numpy(dtype=float)
    else:
        t = np.arange(len(rr), dtype=float)

    start = numeric_or_nan(meta.get("accel_start_unix", np.nan))
    end = numeric_or_nan(meta.get("accel_end_unix", np.nan))
    mask = np.isfinite(rr)

    if time_col is not None and np.isfinite(t).any() and np.isfinite(start) and np.isfinite(end):
        finite_t = t[np.isfinite(t)]
        med_t = float(np.nanmedian(finite_t)) if len(finite_t) else np.nan
        # Unix time. This is the preferred exact trial extraction.
        if np.isfinite(med_t) and med_t > 1e8:
            mask &= np.isfinite(t) & (t >= start) & (t <= end)
            t_rel = t - start
        else:
            # Relative/recording time but no reliable recording offset is available here.
            # Keep whole RR file rather than inventing an offset.
            t_rel = t - np.nanmin(t[np.isfinite(t)]) if np.isfinite(t).any() else np.arange(len(rr), dtype=float)
    else:
        t_rel = t - np.nanmin(t[np.isfinite(t)]) if np.isfinite(t).any() else np.arange(len(rr), dtype=float)

    rr_out = rr[mask]
    t_out = t_rel[mask] if len(t_rel) == len(rr) else np.arange(len(rr_out), dtype=float)
    order = np.argsort(t_out) if len(t_out) == len(rr_out) else np.arange(len(rr_out))
    return rr_out[order], t_out[order]


def rr_item_has_unix_time(item: dict[str, Any]) -> bool:
    """Return True only if the RR file has a usable Unix-time column.

    This is required for trial-specific Lena 04b Poincare/Bland-Altman extraction
    in Veronica's dataset. Without Unix time, the file can be direct beat-to-beat RR
    but cannot be safely cut to the table-motion trial window.
    """
    df = item.get("df", pd.DataFrame())
    time_col = item.get("time_col")
    if time_col is None or time_col not in df.columns:
        return False
    t = pd.to_numeric(df[time_col], errors="coerce").to_numpy(dtype=float)
    finite_t = t[np.isfinite(t)]
    if len(finite_t) == 0:
        return False
    return bool(np.isfinite(np.nanmedian(finite_t)) and np.nanmedian(finite_t) > 1e8)


def choose_rr_file(rr_cache: list[dict[str, Any]], meta: dict[str, Any], side: str) -> Optional[dict[str, Any]]:
    if not rr_cache:
        return None
    scored = [(metadata_match_score(item, meta, side), item) for item in rr_cache]
    scored = [x for x in scored if x[0] > 0]
    if not scored:
        return None
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1]


def rr_pair_from_hr_equivalent(df: pd.DataFrame) -> dict[str, Any]:
    time_sec = pd.to_numeric(df["TimeRelTrialSec"], errors="coerce").to_numpy(dtype=float)
    hr_a = pd.to_numeric(df["HR_A_BPM"], errors="coerce").to_numpy(dtype=float)
    hr_b = pd.to_numeric(df["HR_B_BPM"], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(time_sec) & np.isfinite(hr_a) & np.isfinite(hr_b) & (hr_a > 0) & (hr_b > 0)
    return {
        "rr_a": 60000.0 / hr_a[mask],
        "rr_b": 60000.0 / hr_b[mask],
        "time_a": time_sec[mask],
        "time_b": time_sec[mask],
        "source": "hr_equivalent_fallback_from_step3_4hz_hr",
        "rr_file_A": "",
        "rr_file_B": "",
    }


def get_lena04b_rr_pair(df: pd.DataFrame, meta: dict[str, Any], rr_cache: list[dict[str, Any]], require_direct: bool = False) -> dict[str, Any]:
    """Return RR arrays for Notebook 04b Poincare/Bland-Altman.

    Exact Lena logic uses beat-to-beat RR interval files, not RR reconstructed from
    interpolated HR. We therefore prefer Step 2 corrected RR files and only fall
    back to HR-equivalent RR if direct RR cannot be located unless strict mode is on.
    """
    item_a = choose_rr_file(rr_cache, meta, "A")
    item_b = choose_rr_file(rr_cache, meta, "B")
    if item_a is not None and item_b is not None:
        has_unix_a = rr_item_has_unix_time(item_a)
        has_unix_b = rr_item_has_unix_time(item_b)

        if require_direct and not (has_unix_a and has_unix_b):
            raise ValueError(
                "Direct Step 2 RR files were found, but at least one file lacks a usable Unix-time column. "
                "Cannot safely extract trial-specific RR for Lena 04b Poincare/Bland-Altman."
            )

        rr_a, time_a = extract_trial_rr_from_item(item_a, meta)
        rr_b, time_b = extract_trial_rr_from_item(item_b, meta)
        if len(rr_a) >= 3 and len(rr_b) >= 3:
            source = (
                "direct_step2_corrected_beat_to_beat_rr_trial_window_unix"
                if (has_unix_a and has_unix_b)
                else "direct_step2_corrected_beat_to_beat_rr_whole_file_no_unix_windowing"
            )
            return {
                "rr_a": rr_a,
                "rr_b": rr_b,
                "time_a": time_a,
                "time_b": time_b,
                "source": source,
                "rr_file_A": str(item_a["path"]),
                "rr_file_B": str(item_b["path"]),
            }
    if require_direct:
        raise ValueError("Direct Step 2 corrected RR files could not be found/extracted for Lena 04b Poincare/Bland-Altman.")
    return rr_pair_from_hr_equivalent(df)


def align_rr_for_dyad(rr_pair: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(rr_pair.get("rr_a", []), dtype=float)
    b = np.asarray(rr_pair.get("rr_b", []), dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    n = min(len(a), len(b))
    if n <= 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    return a[:n], b[:n]

def build_file_manifest(paired_hr_dir: Path, trial_summary: pd.DataFrame) -> pd.DataFrame:
    files = sorted([p for p in paired_hr_dir.glob("*.csv") if p.is_file() and not p.name.startswith("._")])
    rows: list[dict[str, Any]] = []
    summary_lookup = {}
    if not trial_summary.empty and "output_file" in trial_summary.columns:
        for _, row in trial_summary.iterrows():
            base = Path(str(row.get("output_file", ""))).name
            if base:
                summary_lookup[base] = row.to_dict()
    for p in files:
        meta = summary_lookup.get(p.name, {})
        rows.append(
            {
                "paired_hr_file": str(p),
                "paired_hr_filename": p.name,
                "has_step3_summary_match": bool(meta),
                "recording_folder": meta.get("recording_folder", ""),
                "dyad_id": meta.get("dyad_id", ""),
                "candidate_window": meta.get("candidate_window", np.nan),
                "trial": meta.get("trial", np.nan),
                "condition": meta.get("condition", ""),
                "is_practice": meta.get("is_practice", np.nan),
                "is_pilot": meta.get("is_pilot", np.nan),
                "exclude_from_main_analysis": meta.get("exclude_from_main_analysis", np.nan),
            }
        )
    return pd.DataFrame(rows)


def infer_row_metadata(df: pd.DataFrame, file_path: Path, trial_summary: pd.DataFrame, reliability_summary: pd.DataFrame) -> dict[str, Any]:
    # Most metadata are repeated in every paired HR row. Use first row as source.
    first = df.iloc[0].to_dict() if not df.empty else {}

    meta_cols = [
        "recording_folder", "dyad_id", "pair1", "participant_A", "participant_B", "sensor_A", "sensor_B",
        "candidate_window", "trial", "is_practice", "condition", "is_pilot", "exclude_from_main_analysis",
        "exclude_reason", "usable_for_dyadic_ecg", "accel_start_unix", "accel_end_unix", "window_duration_sec",
        "dyad_pretrial_qc_flag", "aggregated_data_missing", "ball_drop_flag", "stopped_flag",
        "talking_laughing_flag", "wrong_way_flag", "sensor_issue_comment_flag", "session_comments",
        "table_motion_mean", "trial_duration_mean", "Coordinated_mean", "Joint_control_mean",
        "Incontrol_leading_mean", "pretrial_status_A", "pretrial_qc_flag_A", "pretrial_qc_reasons_A",
        "pretrial_hr_mean_A", "pretrial_hr_sd_A", "pretrial_hr_min_A", "pretrial_hr_max_A",
        "pretrial_hr_coverage_A", "pretrial_rr_plausible_percent_A", "pretrial_status_B", "pretrial_qc_flag_B",
        "pretrial_qc_reasons_B", "pretrial_hr_mean_B", "pretrial_hr_sd_B", "pretrial_hr_min_B",
        "pretrial_hr_max_B", "pretrial_hr_coverage_B", "pretrial_rr_plausible_percent_B",
    ]
    out = {c: first.get(c, np.nan) for c in meta_cols if c in first}
    out["paired_hr_file"] = str(file_path)
    out["paired_hr_filename"] = file_path.name

    # Attach Step 3 segmentation columns not repeated in paired HR file.
    if not trial_summary.empty:
        match = pd.DataFrame()
        if "output_file" in trial_summary.columns:
            match = trial_summary[trial_summary["output_file"].astype(str).map(lambda x: Path(x).name) == file_path.name]
        if match.empty:
            keys = ["recording_folder", "dyad_id", "candidate_window", "trial", "condition"]
            if all(k in trial_summary.columns and k in out for k in keys):
                tmp = trial_summary.copy()
                mask = np.ones(len(tmp), dtype=bool)
                for k in keys:
                    mask &= tmp[k].astype(str).values == str(out.get(k))
                match = tmp[mask]
        if not match.empty:
            row = match.iloc[0]
            for c in [
                "status", "status_detail", "n_expected_samples", "n_saved_samples", "segment_duration_sec",
                "coverage_A", "coverage_B", "n_missing_A", "n_missing_B",
                "trial_segment_qc_flag", "trial_segment_qc_reasons", "HR_A_mean", "HR_A_sd", "HR_A_min",
                "HR_A_max", "HR_B_mean", "HR_B_sd", "HR_B_min", "HR_B_max",
            ]:
                if c in row.index:
                    out[c] = row[c]

    # Attach Step 3a dyad reliability.
    if not reliability_summary.empty:
        match = pd.DataFrame()
        if "paired_hr_output_file" in reliability_summary.columns:
            match = reliability_summary[reliability_summary["paired_hr_output_file"].astype(str).map(lambda x: Path(x).name) == file_path.name]
        if match.empty:
            keys = ["recording_folder", "dyad_id", "candidate_window", "trial", "condition"]
            if all(k in reliability_summary.columns and k in out for k in keys):
                tmp = reliability_summary.copy()
                mask = np.ones(len(tmp), dtype=bool)
                for k in keys:
                    mask &= tmp[k].astype(str).values == str(out.get(k))
                match = tmp[mask]
        if not match.empty:
            row = match.iloc[0]
            for c in [
                "rr_reliability_A", "rr_reliability_B", "rr_reliability_label_A", "rr_reliability_label_B",
                "dyad_rr_reliability_min", "dyad_rr_reliability_mean", "dyad_rr_reliability_label",
                "n_rr_intervals_A", "n_rr_intervals_B", "n_pairs_A", "n_pairs_B",
            ]:
                if c in row.index:
                    out[c] = row[c]

    # Main analysis flag: inherited from Step 1/3. True means excluded.
    out["main_analysis_eligible"] = not bool_from_any(out.get("exclude_from_main_analysis", True))
    return out


# =============================================================================
# Synchrony metrics
# =============================================================================


def estimate_fs(time_sec: np.ndarray, fallback: float = DEFAULT_TARGET_FS) -> float:
    time_sec = np.asarray(time_sec, dtype=float)
    diffs = np.diff(time_sec[np.isfinite(time_sec)])
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return fallback
    dt = float(np.nanmedian(diffs))
    if not np.isfinite(dt) or dt <= 0:
        return fallback
    fs = 1.0 / dt
    if not np.isfinite(fs) or fs <= 0:
        return fallback
    return fs


def _prep_phase_signal_for_hilbert(x: np.ndarray) -> Optional[np.ndarray]:
    """Prepare one HR signal for Hilbert-phase metrics.

    This matches Lena Notebook 04 more closely than the previous Step 4 version:
    detrend first, then z-score/standardize, then fill any remaining NaNs.
    """
    x = np.asarray(x, dtype=float)
    if int(np.isfinite(x).sum()) < 10:
        return None

    x2 = interp_nan_linear(x)
    if int(np.isfinite(x2).sum()) < 10:
        return None

    # Replace any edge-case non-finite values after interpolation.
    if not np.isfinite(x2).all():
        finite = np.isfinite(x2)
        if finite.sum() < 10:
            return None
        x2[~finite] = np.nanmean(x2[finite])

    if np.nanstd(x2, ddof=1) <= 0:
        return None

    # Lena's Notebook 04 prepares signals with linear detrending before
    # phase-based synchrony. This matters for Hilbert phase estimates.
    try:
        x2 = sp_signal.detrend(x2, type="linear")
    except Exception:
        x2 = x2 - np.nanmean(x2)

    x2 = zscore(x2)
    x2 = interp_nan_linear(x2)

    if not np.isfinite(x2).all() or np.nanstd(x2, ddof=1) <= 0:
        return None
    return x2


def prepare_pair_for_lena_sync(
    x: np.ndarray,
    y: np.ndarray,
    min_samples: int = 10,
    detrend_signal: bool = True,
    standardize: bool = True,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray], int]:
    """Prepare paired HR signals the same way Lena Notebook 04 does before metrics.

    Lena's compute_cardiac_synchrony() first prepares a common paired signal, then
    sends that same cleaned pair to Pearson correlation, cross-correlation,
    coherence, PLI, and envelope correlation. The default Lena preprocessing is:
    trim/equal-length paired samples -> linear detrend -> z-score -> replace NaNs.

    In Veronica's Step 3 paired files, the two participants are already on a common
    4 Hz trial grid, so this function keeps rows where both HR values are finite,
    then applies the Lena detrend + standardize preprocessing.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    n = int(mask.sum())
    if n < min_samples:
        return None, None, n

    x2 = np.asarray(x[mask], dtype=float)
    y2 = np.asarray(y[mask], dtype=float)

    if detrend_signal:
        try:
            x2 = sp_signal.detrend(x2, type="linear")
            y2 = sp_signal.detrend(y2, type="linear")
        except Exception:
            x2 = x2 - np.nanmean(x2)
            y2 = y2 - np.nanmean(y2)

    if standardize:
        x2 = zscore(x2)
        y2 = zscore(y2)

    x2 = np.nan_to_num(x2, nan=0.0, posinf=0.0, neginf=0.0)
    y2 = np.nan_to_num(y2, nan=0.0, posinf=0.0, neginf=0.0)

    if len(x2) < min_samples or len(y2) < min_samples:
        return None, None, n
    if np.nanstd(x2, ddof=1) <= 0 or np.nanstd(y2, ddof=1) <= 0:
        return None, None, n
    return x2, y2, n


def prepare_pair_for_lena04b_leader(
    x: np.ndarray,
    y: np.ndarray,
    min_samples: int = 10,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray], int]:
    """Prepare HR pair for Lena Notebook 04b leader-follower analysis.

    Notebook 04b loads the R-peak-based HR series, truncates the two participants
    to a common length, and z-scores them before leader-follower and Granger
    analyses. Unlike Notebook 04's modular synchrony function, Notebook 04b does
    not explicitly apply linear detrending in that leader-follower cell.

    Veronica's Step 3 files are already aligned to a common 4 Hz trial grid, so
    this keeps finite paired samples and applies z-scoring only.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    n = int(mask.sum())
    if n < min_samples:
        return None, None, n
    x2 = zscore(np.asarray(x[mask], dtype=float))
    y2 = zscore(np.asarray(y[mask], dtype=float))
    x2 = np.nan_to_num(x2, nan=0.0, posinf=0.0, neginf=0.0)
    y2 = np.nan_to_num(y2, nan=0.0, posinf=0.0, neginf=0.0)
    if np.nanstd(x2, ddof=1) <= 0 or np.nanstd(y2, ddof=1) <= 0:
        return None, None, n
    return x2, y2, n


def phase_metrics(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 10:
        return {
            "phase_lag_index": np.nan,
            "phase_locking_index_pli": np.nan,
            "phase_locking_value_plv": np.nan,
            "mean_phase_diff_rad": np.nan,
            "mean_phase_diff_abs_rad": np.nan,
        }

    x2 = _prep_phase_signal_for_hilbert(x[mask])
    y2 = _prep_phase_signal_for_hilbert(y[mask])
    if x2 is None or y2 is None:
        return {
            "phase_lag_index": np.nan,
            "phase_locking_index_pli": np.nan,
            "phase_locking_value_plv": np.nan,
            "mean_phase_diff_rad": np.nan,
            "mean_phase_diff_abs_rad": np.nan,
        }

    try:
        phase_x = np.angle(sp_signal.hilbert(x2))
        phase_y = np.angle(sp_signal.hilbert(y2))
        dphi = np.angle(np.exp(1j * (phase_x - phase_y)))
        plv = np.abs(np.mean(np.exp(1j * dphi)))
        pli = np.abs(np.mean(np.sign(np.sin(dphi))))
        mean_phase = circular_mean_rad(dphi)
        return {
            "phase_lag_index": float(pli),
            "phase_locking_index_pli": float(pli),
            "phase_locking_value_plv": float(plv),
            "mean_phase_diff_rad": float(mean_phase),
            "mean_phase_diff_abs_rad": float(abs(mean_phase)) if np.isfinite(mean_phase) else np.nan,
        }
    except Exception:
        return {
            "phase_lag_index": np.nan,
            "phase_locking_index_pli": np.nan,
            "phase_locking_value_plv": np.nan,
            "mean_phase_diff_rad": np.nan,
            "mean_phase_diff_abs_rad": np.nan,
        }


def coherence_metric(x: np.ndarray, y: np.ndarray, fs: float) -> dict[str, float]:
    """Lena Notebook 04 style mean coherence in 0.04-0.40 Hz band.

    Uses the Lena-prepared pair and Lena's length/nperseg rule: if fewer than
    256 samples are available, return NaN; otherwise nperseg = min(256, n//4).
    """
    x2, y2, n = prepare_pair_for_lena_sync(x, y, min_samples=256)
    if x2 is None or y2 is None:
        return {"mean_coherence_0p04_0p40_hz": np.nan, "max_coherence_0p04_0p40_hz": np.nan, "freq_at_max_coherence_hz": np.nan}
    try:
        nperseg = min(256, len(x2) // 4)
        if nperseg < 16:
            return {"mean_coherence_0p04_0p40_hz": np.nan, "max_coherence_0p04_0p40_hz": np.nan, "freq_at_max_coherence_hz": np.nan}
        f, cxy = sp_signal.coherence(x2, y2, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
        band = (f >= 0.04) & (f <= 0.40)
        if not band.any():
            return {"mean_coherence_0p04_0p40_hz": np.nan, "max_coherence_0p04_0p40_hz": np.nan, "freq_at_max_coherence_hz": np.nan}
        band_c = cxy[band]
        band_f = f[band]
        imax = int(np.nanargmax(band_c))
        return {
            "mean_coherence_0p04_0p40_hz": float(np.nanmean(band_c)),
            "max_coherence_0p04_0p40_hz": float(band_c[imax]),
            "freq_at_max_coherence_hz": float(band_f[imax]),
        }
    except Exception:
        return {"mean_coherence_0p04_0p40_hz": np.nan, "max_coherence_0p04_0p40_hz": np.nan, "freq_at_max_coherence_hz": np.nan}


def envelope_corr_metric(x: np.ndarray, y: np.ndarray) -> float:
    """Lena Notebook 04 style envelope correlation on the Lena-prepared pair."""
    x2, y2, _ = prepare_pair_for_lena_sync(x, y, min_samples=10)
    if x2 is None or y2 is None:
        return np.nan
    try:
        env_x = np.abs(sp_signal.hilbert(x2))
        env_y = np.abs(sp_signal.hilbert(y2))
        if np.nanstd(env_x, ddof=1) <= 0 or np.nanstd(env_y, ddof=1) <= 0:
            return np.nan
        r, _, _ = pearson_safe(env_x, env_y)
        return r
    except Exception:
        return np.nan



def crqa_synchrony_lena04(x: Optional[np.ndarray], y: Optional[np.ndarray], dimension: int = 3, time_delay: int = 1, radius: float = 0.1) -> dict[str, Any]:
    """Optional Lena Notebook 04 CRQA metric.

    Lena Notebook 04 includes CRQA as an optional method requiring PyRQA. This
    implementation follows that logic: when PyRQA is unavailable, or when the
    trial is too short, return NaN/unavailable columns rather than failing.
    """
    out: dict[str, Any] = {
        "crqa_available": bool(PYRQA_AVAILABLE),
        "crqa_RR": np.nan,
        "crqa_DET": np.nan,
        "crqa_L": np.nan,
        "crqa_LMAX": np.nan,
        "crqa_ENT": np.nan,
        "crqa_LAM": np.nan,
        "crqa_TT": np.nan,
        "crqa_error": "" if PYRQA_AVAILABLE else "PyRQA not installed",
    }
    if not PYRQA_AVAILABLE:
        return out
    if x is None or y is None:
        out["crqa_error"] = "signals unavailable"
        return out
    try:
        x_clean = np.asarray(x, dtype=float)
        y_clean = np.asarray(y, dtype=float)
        valid = np.isfinite(x_clean) & np.isfinite(y_clean)
        if int(valid.sum()) < 50:
            out["crqa_error"] = f"too few valid samples for CRQA: {int(valid.sum())}"
            return out
        x_clean = x_clean[valid]
        y_clean = y_clean[valid]
        x_clean = zscore(x_clean)
        y_clean = zscore(y_clean)
        x_clean = np.nan_to_num(x_clean)
        y_clean = np.nan_to_num(y_clean)

        ts_x = TimeSeries(x_clean, embedding_dimension=dimension, time_delay=time_delay)
        ts_y = TimeSeries(y_clean, embedding_dimension=dimension, time_delay=time_delay)
        settings = Settings(
            ts_x,
            ts_y,
            neighbourhood=FixedRadius(radius),
            similarity_measure=EuclideanMetric(),
            theiler_corrector=1,
        )
        settings.analysis_type = Cross()
        result = RQAComputation.create(settings, verbose=False).run()
        out.update({
            "crqa_RR": float(result.recurrence_rate),
            "crqa_DET": float(result.determinism),
            "crqa_L": float(result.average_diagonal_line),
            "crqa_LMAX": float(result.longest_diagonal_line),
            "crqa_ENT": float(result.entropy_diagonal_lines),
            "crqa_LAM": float(result.laminarity),
            "crqa_TT": float(result.average_white_vertical_line),
            "crqa_error": "",
        })
        return out
    except Exception as exc:
        out["crqa_error"] = str(exc)
        return out

def crosscorr_metric(x: np.ndarray, y: np.ndarray, fs: float, max_lag_sec: float) -> dict[str, float | str]:
    """Lena Notebook 04 style cross-correlation with lag.

    Uses correlate(x, y), normalizes by signal energy, searches within +/- max_lag,
    and reports the lag using Lena Notebook 04/04b's interpretation:
    positive lag means A/P1 leads B/P2; negative lag means B/P2 leads A/P1.
    """
    x2, y2, n = prepare_pair_for_lena_sync(x, y, min_samples=10)
    if x2 is None or y2 is None:
        return {"crosscorr_max_abs_r": np.nan, "crosscorr_r_at_max_abs": np.nan, "crosscorr_lag_s": np.nan, "crosscorr_leader": "unavailable", "crosscorr_leader_lena_convention": "unavailable", "crosscorr_leader_scipy_signal_convention": "unavailable"}
    try:
        max_lag = int(max_lag_sec * fs)
        x0 = x2 - np.nanmean(x2)
        y0 = y2 - np.nanmean(y2)
        x0 = np.nan_to_num(x0)
        y0 = np.nan_to_num(y0)
        corr = sp_signal.correlate(x0, y0, mode="full", method="fft")
        lags = np.arange(-len(x0) + 1, len(x0))
        keep = (lags >= -max_lag) & (lags <= max_lag)
        if not keep.any():
            return {"crosscorr_max_abs_r": np.nan, "crosscorr_r_at_max_abs": np.nan, "crosscorr_lag_s": np.nan, "crosscorr_leader": "unavailable", "crosscorr_leader_lena_convention": "unavailable", "crosscorr_leader_scipy_signal_convention": "unavailable"}
        corr = corr[keep]
        lags = lags[keep]
        denom = np.sqrt(np.sum(x0**2) * np.sum(y0**2))
        if not np.isfinite(denom) or denom <= 0:
            return {"crosscorr_max_abs_r": np.nan, "crosscorr_r_at_max_abs": np.nan, "crosscorr_lag_s": np.nan, "crosscorr_leader": "unavailable", "crosscorr_leader_lena_convention": "unavailable", "crosscorr_leader_scipy_signal_convention": "unavailable"}
        corr_norm = corr / denom
        peak_idx = int(np.nanargmax(np.abs(corr_norm)))
        r_at = float(corr_norm[peak_idx])
        lag_s = float(lags[peak_idx] / fs)
        # Two labels are saved intentionally:
        # 1) Lena convention, matching Notebook 04b text: positive lag -> A/P1 leads.
        # 2) SciPy signal convention for correlate(A, B): negative lag usually means A leads B.
        if abs(lag_s) < EPS:
            leader_lena = "zero_lag"
            leader_scipy_signal = "zero_lag"
        elif lag_s > 0:
            leader_lena = "A_leads_B"
            leader_scipy_signal = "B_leads_A"
        else:
            leader_lena = "B_leads_A"
            leader_scipy_signal = "A_leads_B"
        return {
            "crosscorr_max_abs_r": float(abs(r_at)),
            "crosscorr_r_at_max_abs": r_at,
            "crosscorr_lag_s": lag_s,
            "crosscorr_leader": leader_lena,
            "crosscorr_leader_lena_convention": leader_lena,
            "crosscorr_leader_scipy_signal_convention": leader_scipy_signal,
        }
    except Exception:
        return {"crosscorr_max_abs_r": np.nan, "crosscorr_r_at_max_abs": np.nan, "crosscorr_lag_s": np.nan, "crosscorr_leader": "unavailable", "crosscorr_leader_lena_convention": "unavailable", "crosscorr_leader_scipy_signal_convention": "unavailable"}


def lag1_direction_metrics(x: np.ndarray, y: np.ndarray) -> dict[str, float | str]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 4 or len(y) < 4:
        return {"lag1_A_to_B": np.nan, "lag1_B_to_A": np.nan, "lag1_asymmetry_AminusB": np.nan, "lag1_leader": "unavailable"}
    # A->B: A(t) predicts B(t+1). B->A: B(t) predicts A(t+1).
    a_to_b, _, _ = pearson_safe(x[:-1], y[1:])
    b_to_a, _, _ = pearson_safe(y[:-1], x[1:])
    asym = a_to_b - b_to_a if np.isfinite(a_to_b) and np.isfinite(b_to_a) else np.nan
    if not np.isfinite(asym):
        leader = "unavailable"
    elif asym > 0:
        leader = "A_more_predictive"
    elif asym < 0:
        leader = "B_more_predictive"
    else:
        leader = "symmetric"
    return {"lag1_A_to_B": a_to_b, "lag1_B_to_A": b_to_a, "lag1_asymmetry_AminusB": asym, "lag1_leader": leader}


def granger_causality_lena04b(x: np.ndarray, y: np.ndarray, max_lag: int = 5) -> dict[str, Any]:
    """Optional Lena Notebook 04b-style bidirectional Granger test.

    Lena treats Granger as optional and requires statsmodels. This function keeps
    that behavior: if statsmodels is unavailable or the trial is too short for
    the requested lag, it records unavailable rather than failing the whole trial.
    """
    out: dict[str, Any] = {
        "granger_available": bool(GRANGER_AVAILABLE),
        "granger_max_lag": int(max_lag),
        "granger_A_to_B_min_pval": np.nan,
        "granger_B_to_A_min_pval": np.nan,
        "granger_A_to_B_significant": False,
        "granger_B_to_A_significant": False,
        "granger_direction": "unavailable" if not GRANGER_AVAILABLE else "not_run",
        "granger_error": "",
    }
    if not GRANGER_AVAILABLE:
        out["granger_error"] = "statsmodels not installed"
        return out
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = min(len(x), len(y))
    if n <= (max_lag * 3 + 2):
        out["granger_direction"] = "too_short"
        out["granger_error"] = f"Too few samples for max_lag={max_lag}: n={n}"
        return out
    x = np.nan_to_num(x[:n])
    y = np.nan_to_num(y[:n])
    if np.nanstd(x) <= 0 or np.nanstd(y) <= 0:
        out["granger_direction"] = "constant_signal"
        return out
    try:
        data_ab = np.column_stack([y, x])  # A/X -> B/Y, matching Lena's data_xy=[y,x]
        gc_ab = grangercausalitytests(data_ab, max_lag, verbose=False)
        p_ab = [gc_ab[lag][0]["ssr_ftest"][1] for lag in range(1, max_lag + 1)]

        data_ba = np.column_stack([x, y])  # B/Y -> A/X
        gc_ba = grangercausalitytests(data_ba, max_lag, verbose=False)
        p_ba = [gc_ba[lag][0]["ssr_ftest"][1] for lag in range(1, max_lag + 1)]

        sig_ab = any(pv < 0.05 for pv in p_ab if np.isfinite(pv))
        sig_ba = any(pv < 0.05 for pv in p_ba if np.isfinite(pv))
        if sig_ab and not sig_ba:
            direction = "A_to_B"
        elif sig_ba and not sig_ab:
            direction = "B_to_A"
        elif sig_ab and sig_ba:
            direction = "bidirectional"
        else:
            direction = "none"
        out.update({
            "granger_A_to_B_min_pval": float(np.nanmin(p_ab)) if len(p_ab) else np.nan,
            "granger_B_to_A_min_pval": float(np.nanmin(p_ba)) if len(p_ba) else np.nan,
            "granger_A_to_B_significant": bool(sig_ab),
            "granger_B_to_A_significant": bool(sig_ba),
            "granger_direction": direction,
        })
        return out
    except Exception as exc:
        out["granger_direction"] = "error"
        out["granger_error"] = str(exc)
        return out


def leader_follower_lena04b_metrics(x: Optional[np.ndarray], y: Optional[np.ndarray], fs: float, max_lag_sec: float) -> dict[str, Any]:
    """Lena Notebook 04b-style leader-follower summary.

    Implements the Notebook 04b core directionality flow: cross-correlation lag,
    lag-1 directed correlations, Hilbert phase directionality, majority-vote
    overall leader, and optional Granger causality.
    """
    base: dict[str, Any] = {
        "lf_xcorr_peak": np.nan,
        "lf_xcorr_lag_s": np.nan,
        "lf_xcorr_leader_lena_convention": "unavailable",
        "lf_xcorr_leader_scipy_signal_convention": "unavailable",
        "lf_lag1_A_to_B": np.nan,
        "lf_lag1_B_to_A": np.nan,
        "lf_lag1_asymmetry_AminusB": np.nan,
        "lf_mean_phase_diff_rad": np.nan,
        "lf_phase_leader": "unavailable",
        "lf_overall_leader": "unavailable",
        "lf_leader_confidence": np.nan,
    }
    if x is None or y is None:
        base.update(granger_causality_lena04b(np.array([]), np.array([])))
        return base
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = min(len(x), len(y))
    if n < 10:
        base.update(granger_causality_lena04b(x, y))
        return base
    x = np.nan_to_num(x[:n])
    y = np.nan_to_num(y[:n])

    try:
        max_lag = int(max_lag_sec * fs)
        x0 = x - np.nanmean(x)
        y0 = y - np.nanmean(y)
        x0 = np.nan_to_num(x0)
        y0 = np.nan_to_num(y0)
        corr = sp_signal.correlate(x0, y0, mode="full", method="fft")
        lags = np.arange(-len(x0) + 1, len(x0))
        keep = (lags >= -max_lag) & (lags <= max_lag)
        corr = corr[keep]
        lags = lags[keep]
        denom = np.sqrt(np.sum(x0**2) * np.sum(y0**2))
        corr_norm = corr / denom if np.isfinite(denom) and denom > 0 else corr
        peak_idx = int(np.nanargmax(np.abs(corr_norm)))
        base["lf_xcorr_peak"] = float(corr_norm[peak_idx])
        base["lf_xcorr_lag_s"] = float(lags[peak_idx] / fs)
        lag_tmp = base["lf_xcorr_lag_s"]
        if abs(lag_tmp) < EPS:
            base["lf_xcorr_leader_lena_convention"] = "zero_lag"
            base["lf_xcorr_leader_scipy_signal_convention"] = "zero_lag"
        elif lag_tmp > 0:
            base["lf_xcorr_leader_lena_convention"] = "A"
            base["lf_xcorr_leader_scipy_signal_convention"] = "B"
        else:
            base["lf_xcorr_leader_lena_convention"] = "B"
            base["lf_xcorr_leader_scipy_signal_convention"] = "A"
    except Exception:
        pass

    if n > 2:
        try:
            r_ab = np.corrcoef(x[:-1], y[1:])[0, 1]
            r_ba = np.corrcoef(y[:-1], x[1:])[0, 1]
            base["lf_lag1_A_to_B"] = float(r_ab)
            base["lf_lag1_B_to_A"] = float(r_ba)
            base["lf_lag1_asymmetry_AminusB"] = float(r_ab - r_ba)
        except Exception:
            pass

    try:
        phase_x = np.angle(sp_signal.hilbert(x - np.nanmean(x)))
        phase_y = np.angle(sp_signal.hilbert(y - np.nanmean(y)))
        phase_diff = phase_x - phase_y
        mean_phase_diff = np.angle(np.mean(np.exp(1j * phase_diff)))
        base["lf_mean_phase_diff_rad"] = float(mean_phase_diff)
        base["lf_phase_leader"] = "A" if mean_phase_diff > 0 else "B"
    except Exception:
        pass

    evidence: list[str] = []
    lag_s = base.get("lf_xcorr_lag_s", np.nan)
    # Match Lena Notebook 04b's stated interpretation:
    # positive lag -> A/X leads B/Y; negative lag -> B/Y leads A/X.
    if np.isfinite(lag_s):
        if lag_s > 1:
            evidence.append("A")
        elif lag_s < -1:
            evidence.append("B")
    asym = base.get("lf_lag1_asymmetry_AminusB", np.nan)
    if np.isfinite(asym) and abs(asym) > 0.05:
        evidence.append("A" if asym > 0 else "B")
    if base.get("lf_phase_leader") in {"A", "B"}:
        evidence.append(str(base["lf_phase_leader"]))
    if evidence:
        a_votes = evidence.count("A")
        b_votes = evidence.count("B")
        if a_votes > b_votes:
            base["lf_overall_leader"] = "A"
            base["lf_leader_confidence"] = a_votes / len(evidence)
        elif b_votes > a_votes:
            base["lf_overall_leader"] = "B"
            base["lf_leader_confidence"] = b_votes / len(evidence)
        else:
            base["lf_overall_leader"] = "SYMMETRIC"
            base["lf_leader_confidence"] = 0.5
    else:
        base["lf_overall_leader"] = "UNCLEAR"
        base["lf_leader_confidence"] = 0.0

    # The overall leader uses Lena Notebook 04b's stated positive-lag convention.
    base["lf_overall_leader_lena_convention"] = base.get("lf_overall_leader", "unavailable")

    base.update(granger_causality_lena04b(x, y, max_lag=5))
    return base


def bland_altman_metrics(rr_a: np.ndarray, rr_b: np.ndarray) -> dict[str, float]:
    rr_a = np.asarray(rr_a, dtype=float)
    rr_b = np.asarray(rr_b, dtype=float)
    mask = np.isfinite(rr_a) & np.isfinite(rr_b)
    if int(mask.sum()) < 3:
        return {"rr_mean_difference_AminusB_ms": np.nan, "rr_difference_sd_ms": np.nan, "rr_limit_agreement_low_ms": np.nan, "rr_limit_agreement_high_ms": np.nan}
    diff = rr_a[mask] - rr_b[mask]
    md = float(np.nanmean(diff))
    sd = float(np.nanstd(diff, ddof=1))
    return {
        "rr_mean_difference_AminusB_ms": md,
        "rr_difference_sd_ms": sd,
        "rr_limit_agreement_low_ms": md - 1.96 * sd,
        "rr_limit_agreement_high_ms": md + 1.96 * sd,
    }


def poincare_metrics(rr: np.ndarray, prefix: str) -> dict[str, float]:
    rr = np.asarray(rr, dtype=float)
    rr = rr[np.isfinite(rr)]
    if len(rr) < 3:
        return {f"{prefix}_sd1_ms": np.nan, f"{prefix}_sd2_ms": np.nan, f"{prefix}_sd1_sd2_ratio": np.nan, f"{prefix}_poincare_r": np.nan}
    rr_n = rr[:-1]
    rr_np1 = rr[1:]
    diff = rr_np1 - rr_n
    summ = rr_np1 + rr_n
    sd1 = np.sqrt(0.5) * np.nanstd(diff, ddof=1)
    sd2 = np.sqrt(0.5) * np.nanstd(summ, ddof=1)
    r, _, _ = pearson_safe(rr_n, rr_np1)
    return {
        f"{prefix}_sd1_ms": float(sd1),
        f"{prefix}_sd2_ms": float(sd2),
        f"{prefix}_sd1_sd2_ratio": float(sd1 / sd2) if np.isfinite(sd2) and sd2 != 0 else np.nan,
        f"{prefix}_poincare_r": r,
    }


def compute_windowed_metrics(
    time_sec: np.ndarray,
    hr_a: np.ndarray,
    hr_b: np.ndarray,
    fs: float,
    window_sec: float,
    overlap: float,
    min_window_samples: int,
    meta: dict[str, Any],
) -> pd.DataFrame:
    if len(time_sec) == 0:
        return pd.DataFrame()
    duration = float(np.nanmax(time_sec) - np.nanmin(time_sec)) if np.isfinite(time_sec).sum() else 0.0
    if duration <= 0:
        return pd.DataFrame()
    step_sec = window_sec * (1.0 - overlap)
    if step_sec <= 0:
        step_sec = window_sec
    starts = np.arange(0, max(duration - window_sec, 0) + EPS, step_sec)
    if len(starts) == 0:
        starts = np.array([0.0])
    rows: list[dict[str, Any]] = []
    for wi, start in enumerate(starts, start=1):
        end = start + window_sec
        mask = (time_sec >= start) & (time_sec <= end) & np.isfinite(hr_a) & np.isfinite(hr_b)
        n = int(mask.sum())
        if n < min_window_samples:
            continue
        xa = hr_a[mask]
        yb = hr_b[mask]
        xa_clean, yb_clean, _ = prepare_pair_for_lena_sync(xa, yb, min_samples=min_window_samples)
        if xa_clean is None or yb_clean is None:
            r, p, sr, sp = np.nan, np.nan, np.nan, np.nan
        else:
            r, p, _ = pearson_safe(xa_clean, yb_clean)
            sr, sp, _ = spearman_safe(xa_clean, yb_clean)
        ph = phase_metrics(xa, yb)
        cc = crosscorr_metric(xa, yb, fs=fs, max_lag_sec=min(DEFAULT_MAX_LAG_SEC, window_sec / 3.0))
        row = {
            "window_index": wi,
            "window_start_s": float(start),
            "window_end_s": float(end),
            "window_center_s": float((start + end) / 2.0),
            "n_valid_samples_window": n,
            "window_pearson_r": r,
            "window_pearson_p": p,
            "window_spearman_r": sr,
            "window_spearman_p": sp,
            "window_phase_locking_index_pli": ph["phase_locking_index_pli"],
            "window_phase_lag_index": ph["phase_lag_index"],
            "window_phase_locking_value_plv": ph["phase_locking_value_plv"],
            "window_crosscorr_max_abs_r": cc["crosscorr_max_abs_r"],
            "window_crosscorr_lag_s": cc["crosscorr_lag_s"],
            "window_crosscorr_leader": cc["crosscorr_leader"],
            "window_crosscorr_leader_lena_convention": cc.get("crosscorr_leader_lena_convention", cc["crosscorr_leader"]),
            "window_crosscorr_leader_scipy_signal_convention": cc.get("crosscorr_leader_scipy_signal_convention", "unavailable"),
            "window_mean_hr_A_bpm": mean_or_nan(xa),
            "window_mean_hr_B_bpm": mean_or_nan(yb),
            "window_mean_hr_diff_AminusB_bpm": mean_or_nan(xa - yb),
        }
        for key in [
            "recording_folder", "dyad_id", "pair1", "participant_A", "participant_B", "sensor_A", "sensor_B",
            "candidate_window", "trial", "is_practice", "condition", "is_pilot", "exclude_from_main_analysis",
            "exclude_reason", "main_analysis_eligible", "trial_segment_qc_flag", "dyad_pretrial_qc_flag",
            "dyad_rr_reliability_label", "dyad_rr_reliability_min",
        ]:
            if key in meta:
                row[key] = meta[key]
        rows.append(row)
    return pd.DataFrame(rows)


def compute_trial_metrics(df: pd.DataFrame, file_path: Path, meta: dict[str, Any], args: argparse.Namespace, rr_cache: list[dict[str, Any]]) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    time_sec = pd.to_numeric(df["TimeRelTrialSec"], errors="coerce").to_numpy(dtype=float)
    hr_a = pd.to_numeric(df["HR_A_BPM"], errors="coerce").to_numpy(dtype=float)
    hr_b = pd.to_numeric(df["HR_B_BPM"], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(time_sec) & np.isfinite(hr_a) & np.isfinite(hr_b)
    if int(valid.sum()) < args.min_valid_samples:
        raise ValueError(f"Too few valid paired samples: {int(valid.sum())}")

    time_sec = time_sec[valid]
    hr_a = hr_a[valid]
    hr_b = hr_b[valid]
    order = np.argsort(time_sec)
    time_sec = time_sec[order]
    hr_a = hr_a[order]
    hr_b = hr_b[order]

    fs = estimate_fs(time_sec, fallback=DEFAULT_TARGET_FS)
    rr_a_hr_equiv = 60000.0 / hr_a
    rr_b_hr_equiv = 60000.0 / hr_b
    rr_a_hr_equiv[~np.isfinite(rr_a_hr_equiv)] = np.nan
    rr_b_hr_equiv[~np.isfinite(rr_b_hr_equiv)] = np.nan

    rr_pair = get_lena04b_rr_pair(df, meta, rr_cache, require_direct=args.require_direct_rr_for_poincare)
    rr_a_lena = np.asarray(rr_pair.get("rr_a", []), dtype=float)
    rr_b_lena = np.asarray(rr_pair.get("rr_b", []), dtype=float)
    rr_a_dyad, rr_b_dyad = align_rr_for_dyad(rr_pair)

    sync_a, sync_b, n_pair = prepare_pair_for_lena_sync(hr_a, hr_b, min_samples=args.min_valid_samples)
    if sync_a is None or sync_b is None:
        pearson_r, pearson_p = np.nan, np.nan
        spearman_r, spearman_p = np.nan, np.nan
        pearson_z_a_r, pearson_z_a_p = np.nan, np.nan
    else:
        pearson_r, pearson_p, _ = pearson_safe(sync_a, sync_b)
        spearman_r, spearman_p, _ = spearman_safe(sync_a, sync_b)
        # After Lena-style preprocessing, these are already z-scored. Kept as a
        # backward-compatible duplicate of Pearson r for previous Step 4 tables.
        pearson_z_a_r, pearson_z_a_p, _ = pearson_safe(sync_a, sync_b)

    ph = phase_metrics(hr_a, hr_b)
    coh = coherence_metric(hr_a, hr_b, fs=fs)
    cc = crosscorr_metric(hr_a, hr_b, fs=fs, max_lag_sec=args.max_lag_sec)

    # Lena Notebook 04b leader-follower analysis uses z-scored HR without the
    # explicit linear detrending used in Notebook 04's modular synchrony function.
    lf_a, lf_b, _ = prepare_pair_for_lena04b_leader(hr_a, hr_b, min_samples=10)
    lag1 = lag1_direction_metrics(lf_a, lf_b) if lf_a is not None and lf_b is not None else lag1_direction_metrics(np.array([]), np.array([]))
    lf04b = leader_follower_lena04b_metrics(lf_a, lf_b, fs=fs, max_lag_sec=args.max_lag_sec)

    # CRQA belongs to Lena Notebook 04, so it uses the Notebook 04 detrended +
    # z-scored synchrony pair.
    crqa04 = crqa_synchrony_lena04(sync_a, sync_b)
    ba = bland_altman_metrics(rr_a_dyad, rr_b_dyad)
    p_a = poincare_metrics(rr_a_lena, "A_rr")
    p_b = poincare_metrics(rr_b_lena, "B_rr")
    p_dyad = poincare_metrics((rr_a_dyad + rr_b_dyad) / 2.0, "dyad_mean_rr") if len(rr_a_dyad) >= 3 else poincare_metrics(np.array([], dtype=float), "dyad_mean_rr")
    rr_corr, rr_corr_p, _ = pearson_safe(rr_a_dyad, rr_b_dyad)

    window_df = compute_windowed_metrics(
        time_sec=time_sec,
        hr_a=hr_a,
        hr_b=hr_b,
        fs=fs,
        window_sec=args.window_sec,
        overlap=args.window_overlap,
        min_window_samples=args.min_window_samples,
        meta=meta,
    )

    trial = dict(meta)
    trial.update(
        {
            "script_version": SCRIPT_VERSION,
            "sampling_rate_hz_estimated": fs,
            "n_valid_samples": n_pair,
            "duration_s": float(np.nanmax(time_sec) - np.nanmin(time_sec)),
            "mean_hr_A_bpm": mean_or_nan(hr_a),
            "sd_hr_A_bpm": sd_or_nan(hr_a),
            "min_hr_A_bpm": float(np.nanmin(hr_a)),
            "max_hr_A_bpm": float(np.nanmax(hr_a)),
            "mean_hr_B_bpm": mean_or_nan(hr_b),
            "sd_hr_B_bpm": sd_or_nan(hr_b),
            "min_hr_B_bpm": float(np.nanmin(hr_b)),
            "max_hr_B_bpm": float(np.nanmax(hr_b)),
            "mean_hr_dyad_bpm": mean_or_nan((hr_a + hr_b) / 2.0),
            "mean_hr_diff_AminusB_bpm": mean_or_nan(hr_a - hr_b),
            "sd_hr_diff_AminusB_bpm": sd_or_nan(hr_a - hr_b),
            "mean_rr_A_ms_from_hr": mean_or_nan(rr_a_hr_equiv),
            "sd_rr_A_ms_from_hr": sd_or_nan(rr_a_hr_equiv),
            "mean_rr_B_ms_from_hr": mean_or_nan(rr_b_hr_equiv),
            "sd_rr_B_ms_from_hr": sd_or_nan(rr_b_hr_equiv),
            "poincare_bland_altman_rr_source": rr_pair.get("source", ""),
            "poincare_rr_file_A": rr_pair.get("rr_file_A", ""),
            "poincare_rr_file_B": rr_pair.get("rr_file_B", ""),
            "n_rr_A_for_poincare": int(np.isfinite(rr_a_lena).sum()),
            "n_rr_B_for_poincare": int(np.isfinite(rr_b_lena).sum()),
            "n_rr_paired_for_bland_altman": int(min(np.isfinite(rr_a_lena).sum(), np.isfinite(rr_b_lena).sum())),
            "pearson_r": pearson_r,
            "pearson_p": pearson_p,
            "spearman_r": spearman_r,
            "spearman_p": spearman_p,
            "pearson_zscore_r": pearson_z_a_r,
            "pearson_zscore_p": pearson_z_a_p,
            # Backward-compatible old column name retained, but the value uses
            # the RR pair selected for Poincare/Bland-Altman, preferably direct Step 2 RR.
            "rr_corr_from_hr_equivalent": rr_corr,
            "rr_corr_from_hr_equivalent_p": rr_corr_p,
            "rr_corr_for_poincare_selected_rr": rr_corr,
            "rr_corr_for_poincare_selected_rr_p": rr_corr_p,
            "envelope_corr": envelope_corr_metric(hr_a, hr_b),
        }
    )
    trial.update(ph)
    trial.update(coh)
    trial.update(cc)
    trial.update(lag1)
    trial.update(lf04b)
    trial.update(crqa04)
    trial.update(ba)
    trial.update(p_a)
    trial.update(p_b)
    trial.update(p_dyad)

    if window_df.empty:
        trial.update(
            {
                "n_valid_windows": 0,
                "windowed_pearson_mean": np.nan,
                "windowed_pearson_sd": np.nan,
                "windowed_pearson_min": np.nan,
                "windowed_pearson_max": np.nan,
                "windowed_spearman_mean": np.nan,
                "windowed_pli_mean": np.nan,
                "windowed_pli_sd": np.nan,
                "windowed_plv_mean": np.nan,
                "windowed_crosscorr_lag_mean_s": np.nan,
            }
        )
    else:
        trial.update(
            {
                "n_valid_windows": int(len(window_df)),
                "windowed_pearson_mean": mean_or_nan(window_df["window_pearson_r"].to_numpy()),
                "windowed_pearson_sd": sd_or_nan(window_df["window_pearson_r"].to_numpy()),
                "windowed_pearson_min": float(np.nanmin(window_df["window_pearson_r"].to_numpy())) if np.isfinite(window_df["window_pearson_r"]).any() else np.nan,
                "windowed_pearson_max": float(np.nanmax(window_df["window_pearson_r"].to_numpy())) if np.isfinite(window_df["window_pearson_r"]).any() else np.nan,
                "windowed_spearman_mean": mean_or_nan(window_df["window_spearman_r"].to_numpy()),
                "windowed_pli_mean": mean_or_nan(window_df["window_phase_locking_index_pli"].to_numpy()),
                "windowed_pli_sd": sd_or_nan(window_df["window_phase_locking_index_pli"].to_numpy()),
                "windowed_plv_mean": mean_or_nan(window_df["window_phase_locking_value_plv"].to_numpy()),
                "windowed_crosscorr_lag_mean_s": mean_or_nan(window_df["window_crosscorr_lag_s"].to_numpy()),
            }
        )
    return trial, window_df, rr_pair


# =============================================================================
# Plotting: Lena Notebook 04 and 04b style outputs
# =============================================================================


def plot_lena04_windowed(window_df: pd.DataFrame, meta: dict[str, Any], out_dir: Path) -> Optional[str]:
    if window_df.empty:
        return None
    title_id = f"{meta.get('recording_folder','')} | {meta.get('dyad_id','')} | cw{int(meta.get('candidate_window', -1)):02d} | trial {meta.get('trial','')} | {meta.get('condition','')}"
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    x = window_df["window_center_s"].to_numpy(dtype=float)

    y_corr = window_df["window_pearson_r"].to_numpy(dtype=float)
    axes[0].plot(x, y_corr, marker="o", linewidth=1.5)
    axes[0].axhline(0, linestyle="--", linewidth=1)
    if np.isfinite(y_corr).any():
        axes[0].axhline(np.nanmean(y_corr), linestyle=":", linewidth=1.5, label=f"Mean r = {np.nanmean(y_corr):.3f}")
        axes[0].legend(loc="best", fontsize=9)
    axes[0].set_ylabel("Correlation (r)")
    axes[0].set_title("Windowed Cardiac Synchrony", pad=12)
    fig.suptitle(title_id, fontsize=10, y=0.995)
    axes[0].set_ylim(-1.05, 1.05)
    axes[0].grid(True, alpha=0.25)

    y_pli = window_df["window_phase_locking_index_pli"].to_numpy(dtype=float)
    axes[1].plot(x, y_pli, marker="s", linewidth=1.5)
    if np.isfinite(y_pli).any():
        axes[1].axhline(np.nanmean(y_pli), linestyle=":", linewidth=1.5, label=f"Mean PLI = {np.nanmean(y_pli):.3f}")
        axes[1].legend(loc="best", fontsize=9)
    axes[1].set_ylabel("PLI")
    axes[1].set_xlabel("Time (seconds)")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].grid(True, alpha=0.25)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fname = f"{safe_piece(meta.get('recording_folder'))}__{safe_piece(meta.get('dyad_id'))}__cw{int(meta.get('candidate_window', -1)):02d}__trial{int(meta.get('trial', -1)):02d}__{safe_piece(meta.get('condition'))}__lena04_windowed_cardiac_synchrony.png"
    return write_png(fig, out_dir / fname)


def ellipse_from_xy(x: np.ndarray, y: np.ndarray, n_std: float = 1.0) -> Optional[Ellipse]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3:
        return None
    x2 = x[mask]
    y2 = y[mask]
    cov = np.cov(x2, y2)
    if not np.all(np.isfinite(cov)):
        return None
    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, 0)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(vals)
    return Ellipse((np.mean(x2), np.mean(y2)), width=width, height=height, angle=angle, fill=False, linewidth=2)


def add_identity_line(ax, values: np.ndarray) -> None:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return
    lo = float(np.nanmin(values))
    hi = float(np.nanmax(values))
    pad = (hi - lo) * 0.05 if hi > lo else 10
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], linestyle="--", linewidth=1)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)


def plot_comprehensive_poincare(df: pd.DataFrame, meta: dict[str, Any], trial_metrics: dict[str, Any], out_dir: Path, rr_pair: dict[str, Any]) -> Optional[str]:
    rr_a = np.asarray(rr_pair.get("rr_a", []), dtype=float)
    rr_b = np.asarray(rr_pair.get("rr_b", []), dtype=float)
    rr_a = rr_a[np.isfinite(rr_a)]
    rr_b = rr_b[np.isfinite(rr_b)]
    rr_a_dyad, rr_b_dyad = align_rr_for_dyad(rr_pair)
    if len(rr_a) < 4 or len(rr_b) < 4 or len(rr_a_dyad) < 3:
        return None

    fig = plt.figure(figsize=(18, 11))
    gs = fig.add_gridspec(2, 3)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, 0:2])
    ax5 = fig.add_subplot(gs[1, 2])

    # Participant A individual Poincare.
    ax1.scatter(rr_a[:-1], rr_a[1:], s=16, alpha=0.65)
    add_identity_line(ax1, rr_a)
    ax1.set_title(f"Participant A Poincare\nSD1={trial_metrics.get('A_rr_sd1_ms', np.nan):.1f}ms, SD2={trial_metrics.get('A_rr_sd2_ms', np.nan):.1f}ms")
    ax1.set_xlabel("RR(n) [ms]")
    ax1.set_ylabel("RR(n+1) [ms]")
    ax1.grid(True, alpha=0.25)

    # Participant B individual Poincare.
    ax2.scatter(rr_b[:-1], rr_b[1:], s=16, alpha=0.65)
    add_identity_line(ax2, rr_b)
    ax2.set_title(f"Participant B Poincare\nSD1={trial_metrics.get('B_rr_sd1_ms', np.nan):.1f}ms, SD2={trial_metrics.get('B_rr_sd2_ms', np.nan):.1f}ms")
    ax2.set_xlabel("RR(n) [ms]")
    ax2.set_ylabel("RR(n+1) [ms]")
    ax2.grid(True, alpha=0.25)

    # Dyadic coordination scatter.
    ax3.scatter(rr_a_dyad, rr_b_dyad, s=16, alpha=0.65)
    vals = np.r_[rr_a_dyad, rr_b_dyad]
    add_identity_line(ax3, vals)
    r = trial_metrics.get("rr_corr_from_hr_equivalent", np.nan)
    if np.isfinite(r) and len(rr_a_dyad) >= 3:
        try:
            slope, intercept, *_ = stats.linregress(rr_a_dyad, rr_b_dyad)
            xs = np.linspace(np.nanmin(rr_a_dyad), np.nanmax(rr_a_dyad), 100)
            ax3.plot(xs, intercept + slope * xs, linewidth=1.5)
        except Exception:
            pass
    ax3.set_title(f"Dyadic Coordination\nr = {r:.3f}" if np.isfinite(r) else "Dyadic Coordination\nr = NA")
    ax3.set_xlabel("Participant A RR [ms]")
    ax3.set_ylabel("Participant B RR [ms]")
    ax3.grid(True, alpha=0.25)

    # RR time series overlay, matching Lena Notebook 04b beat-index display.
    ax4.plot(np.arange(len(rr_a)), rr_a, linewidth=1.2, label="Participant A")
    ax4.plot(np.arange(len(rr_b)), rr_b, linewidth=1.2, label="Participant B")
    ax4.set_title("RR Interval Time Series")
    ax4.set_xlabel("Beat Number")
    ax4.set_ylabel("RR Interval [ms]")
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.25)

    # Bland-Altman.
    mean_rr = (rr_a_dyad + rr_b_dyad) / 2.0
    diff_rr = rr_a_dyad - rr_b_dyad
    ax5.scatter(mean_rr, diff_rr, s=16, alpha=0.65)
    md = trial_metrics.get("rr_mean_difference_AminusB_ms", np.nan)
    lo = trial_metrics.get("rr_limit_agreement_low_ms", np.nan)
    hi = trial_metrics.get("rr_limit_agreement_high_ms", np.nan)
    if np.isfinite(md):
        ax5.axhline(md, linestyle="-", linewidth=1, label="Mean diff")
    if np.isfinite(lo):
        ax5.axhline(lo, linestyle="--", linewidth=1, label="-1.96 SD")
    if np.isfinite(hi):
        ax5.axhline(hi, linestyle="--", linewidth=1, label="+1.96 SD")
    ax5.set_title("Bland-Altman Plot")
    ax5.set_xlabel("Mean RR [ms]")
    ax5.set_ylabel("Participant A - Participant B [ms]")
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.25)

    fig.suptitle(
        f"Comprehensive Dyadic Poincare Visualization | {meta.get('recording_folder','')} | {meta.get('dyad_id','')} | "
        f"cw{int(meta.get('candidate_window', -1)):02d} | trial {meta.get('trial','')} | {meta.get('condition','')}",
        fontsize=13,
        y=0.985,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fname = f"{safe_piece(meta.get('recording_folder'))}__{safe_piece(meta.get('dyad_id'))}__cw{int(meta.get('candidate_window', -1)):02d}__trial{int(meta.get('trial', -1)):02d}__{safe_piece(meta.get('condition'))}__lena04b_comprehensive_poincare.png"
    return write_png(fig, out_dir / fname)


def plot_overlaid_ellipses(df: pd.DataFrame, meta: dict[str, Any], trial_metrics: dict[str, Any], out_dir: Path, rr_pair: dict[str, Any]) -> Optional[str]:
    rr_a = np.asarray(rr_pair.get("rr_a", []), dtype=float)
    rr_b = np.asarray(rr_pair.get("rr_b", []), dtype=float)
    rr_a = rr_a[np.isfinite(rr_a)]
    rr_b = rr_b[np.isfinite(rr_b)]
    if len(rr_a) < 4 or len(rr_b) < 4:
        return None
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.scatter(rr_a[:-1], rr_a[1:], s=16, alpha=0.55, label="Participant A")
    ax.scatter(rr_b[:-1], rr_b[1:], s=16, alpha=0.55, label="Participant B")
    ell_a = ellipse_from_xy(rr_a[:-1], rr_a[1:])
    if ell_a is not None:
        ax.add_patch(ell_a)
    ell_b = ellipse_from_xy(rr_b[:-1], rr_b[1:])
    if ell_b is not None:
        ax.add_patch(ell_b)
    vals = np.r_[rr_a, rr_b]
    add_identity_line(ax, vals)
    ax.set_title("Dyadic Poincare Plot - A/B Overlay", pad=14)
    ax.set_xlabel("RR(n) [ms]")
    ax.set_ylabel("RR(n+1) [ms]")
    txt = (
        f"A SD1/SD2: {trial_metrics.get('A_rr_sd1_ms', np.nan):.1f}/{trial_metrics.get('A_rr_sd2_ms', np.nan):.1f} ms\n"
        f"B SD1/SD2: {trial_metrics.get('B_rr_sd1_ms', np.nan):.1f}/{trial_metrics.get('B_rr_sd2_ms', np.nan):.1f} ms\n"
        f"Dyadic RR r: {trial_metrics.get('rr_corr_from_hr_equivalent', np.nan):.3f}"
    )
    ax.text(0.02, 0.98, txt, transform=ax.transAxes, va="top", fontsize=9, bbox=dict(boxstyle="round", alpha=0.15))
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fname = f"{safe_piece(meta.get('recording_folder'))}__{safe_piece(meta.get('dyad_id'))}__cw{int(meta.get('candidate_window', -1)):02d}__trial{int(meta.get('trial', -1)):02d}__{safe_piece(meta.get('condition'))}__lena04b_poincare_ab_overlay_with_ellipses.png"
    return write_png(fig, out_dir / fname)


def plot_temporal_poincare(df: pd.DataFrame, window_df: pd.DataFrame, meta: dict[str, Any], out_dir: Path, rr_pair: dict[str, Any], max_panels: int = 12) -> Optional[str]:
    if window_df.empty:
        return None
    rr_a = np.asarray(rr_pair.get("rr_a", []), dtype=float)
    rr_b = np.asarray(rr_pair.get("rr_b", []), dtype=float)
    time_a = np.asarray(rr_pair.get("time_a", np.arange(len(rr_a))), dtype=float)
    time_b = np.asarray(rr_pair.get("time_b", np.arange(len(rr_b))), dtype=float)
    mask_a = np.isfinite(time_a) & np.isfinite(rr_a)
    mask_b = np.isfinite(time_b) & np.isfinite(rr_b)
    time_a, rr_a = time_a[mask_a], rr_a[mask_a]
    time_b, rr_b = time_b[mask_b], rr_b[mask_b]
    if len(rr_a) < 4 or len(rr_b) < 4:
        return None
    wdf = window_df.copy().head(max_panels)
    n_panels = len(wdf)
    if n_panels == 0:
        return None
    ncols = min(3, n_panels)
    nrows = int(math.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.4 * nrows), squeeze=False)
    axes_flat = axes.ravel()
    for ax, (_, row) in zip(axes_flat, wdf.iterrows()):
        start = float(row["window_start_s"])
        end = float(row["window_end_s"])
        mask_a_w = (time_a >= start) & (time_a <= end)
        mask_b_w = (time_b >= start) & (time_b <= end)
        a = rr_a[mask_a_w]
        b = rr_b[mask_b_w]
        if len(a) >= 3:
            ax.scatter(a[:-1], a[1:], s=12, alpha=0.58, label="A")
        if len(b) >= 3:
            ax.scatter(b[:-1], b[1:], s=12, alpha=0.58, label="B")
        vals = np.r_[a, b]
        add_identity_line(ax, vals)
        r = row.get("window_pearson_r", np.nan)
        ax.set_title(f"t={row.get('window_center_s', np.nan):.1f}s, r={r:.2f}, n={int(row.get('n_valid_samples_window', 0))}", fontsize=9)
        ax.set_xlabel("RR(n) [ms]", fontsize=8)
        ax.set_ylabel("RR(n+1) [ms]", fontsize=8)
        ax.grid(True, alpha=0.2)
    for ax in axes_flat[n_panels:]:
        ax.axis("off")
    fig.suptitle(
        f"Temporal Evolution of Dyadic Poincare Patterns\nParticipant A vs Participant B | {meta.get('recording_folder','')} | {meta.get('dyad_id','')} | {meta.get('condition','')}",
        fontsize=13,
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fname = f"{safe_piece(meta.get('recording_folder'))}__{safe_piece(meta.get('dyad_id'))}__cw{int(meta.get('candidate_window', -1)):02d}__trial{int(meta.get('trial', -1)):02d}__{safe_piece(meta.get('condition'))}__lena04b_temporal_poincare_patterns.png"
    return write_png(fig, out_dir / fname)


def plot_temporal_synchrony(window_df: pd.DataFrame, meta: dict[str, Any], out_dir: Path) -> Optional[str]:
    if window_df.empty:
        return None
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = window_df["window_center_s"].to_numpy(dtype=float)
    y = window_df["window_pearson_r"].to_numpy(dtype=float)
    ax.axhspan(0.2, 0.4, alpha=0.08, label="Weak/moderate +")
    ax.axhspan(0.4, 1.0, alpha=0.08, label="Moderate/strong +")
    ax.axhspan(-0.4, -0.2, alpha=0.08, label="Weak/moderate -")
    ax.axhspan(-1.0, -0.4, alpha=0.08, label="Moderate/strong -")
    ax.plot(x, y, marker="o", linewidth=1.5)
    ax.axhline(0, linewidth=1, linestyle="--")
    if np.isfinite(y).any():
        ax.axhline(np.nanmean(y), linewidth=1.5, linestyle=":", label=f"Mean r = {np.nanmean(y):.3f}")
    ax.set_title("Temporal Evolution of Dyadic Synchrony", pad=14)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Dyadic Correlation")
    ax.set_ylim(-1.05, 1.05)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True)
    fig.tight_layout(rect=[0, 0, 0.82, 1])
    fname = f"{safe_piece(meta.get('recording_folder'))}__{safe_piece(meta.get('dyad_id'))}__cw{int(meta.get('candidate_window', -1)):02d}__trial{int(meta.get('trial', -1)):02d}__{safe_piece(meta.get('condition'))}__lena04b_temporal_synchrony.png"
    return write_png(fig, out_dir / fname)


def save_lena_style_plots(df: pd.DataFrame, window_df: pd.DataFrame, meta: dict[str, Any], trial_metrics: dict[str, Any], plot_dirs: dict[str, Path], rr_pair: dict[str, Any]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    funcs = [
        ("lena04_windowed_cardiac_synchrony", lambda: plot_lena04_windowed(window_df, meta, plot_dirs["lena04"])),
        ("lena04b_comprehensive_poincare", lambda: plot_comprehensive_poincare(df, meta, trial_metrics, plot_dirs["comprehensive"], rr_pair)),
        ("lena04b_overlaid_ellipses", lambda: plot_overlaid_ellipses(df, meta, trial_metrics, plot_dirs["ellipses"], rr_pair)),
        ("lena04b_temporal_poincare_patterns", lambda: plot_temporal_poincare(df, window_df, meta, plot_dirs["temporal_poincare"], rr_pair)),
    ]
    for plot_type, func in funcs:
        try:
            path = func()
            rows.append({"paired_hr_filename": str(meta.get("paired_hr_filename", "")), "plot_type": plot_type, "plot_file": path or "", "plot_status": "saved" if path else "skipped_no_data"})
        except Exception as exc:
            rows.append({"paired_hr_filename": str(meta.get("paired_hr_filename", "")), "plot_type": plot_type, "plot_file": "", "plot_status": f"failed: {exc}"})
        finally:
            plt.close("all")
    return rows


# =============================================================================
# Group summary plots and tables
# =============================================================================


def summarize_by_condition(trial_df: pd.DataFrame, main_only: bool) -> pd.DataFrame:
    df = trial_df.copy()
    if main_only and "main_analysis_eligible" in df.columns:
        df = df[df["main_analysis_eligible"].map(bool_from_any)].copy()
    if df.empty:
        return pd.DataFrame()
    group_cols = [c for c in ["condition"] if c in df.columns]
    metrics = [
        "pearson_r", "spearman_r", "windowed_pearson_mean", "phase_locking_index_pli", "phase_locking_value_plv",
        "phase_lag_index", "crosscorr_max_abs_r", "crosscorr_lag_s", "envelope_corr", "mean_coherence_0p04_0p40_hz",
        "dyad_rr_reliability_min", "mean_hr_diff_AminusB_bpm",
    ]
    metrics = [m for m in metrics if m in df.columns]
    rows = []
    for keys, sub in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = {group_cols[i]: keys[i] for i in range(len(group_cols))}
        base["n_trials"] = int(len(sub))
        if "main_analysis_eligible" in sub.columns:
            base["n_main_analysis_eligible"] = int(sub["main_analysis_eligible"].map(bool_from_any).sum())
        for m in metrics:
            vals = pd.to_numeric(sub[m], errors="coerce").to_numpy(dtype=float)
            base[f"{m}_mean"] = mean_or_nan(vals)
            base[f"{m}_sd"] = sd_or_nan(vals)
            base[f"{m}_median"] = float(np.nanmedian(vals)) if np.isfinite(vals).any() else np.nan
            base[f"{m}_n"] = int(np.isfinite(vals).sum())
        rows.append(base)
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def plot_group_metric_boxplots(trial_df: pd.DataFrame, out_dir: Path, main_only: bool = False) -> list[str]:
    df = trial_df.copy()
    if main_only and "main_analysis_eligible" in df.columns:
        df = df[df["main_analysis_eligible"].map(bool_from_any)].copy()
    if df.empty or "condition" not in df.columns:
        return []
    metrics = [
        "pearson_r", "spearman_r", "windowed_pearson_mean", "phase_locking_index_pli", "phase_locking_value_plv",
        "crosscorr_max_abs_r", "envelope_corr", "mean_coherence_0p04_0p40_hz", "dyad_rr_reliability_min",
    ]
    metrics = [m for m in metrics if m in df.columns]
    paths = []
    suffix = "main_analysis" if main_only else "all_windows"
    for metric in metrics:
        tmp = df[["condition", metric]].copy()
        tmp[metric] = pd.to_numeric(tmp[metric], errors="coerce")
        tmp = tmp.dropna(subset=[metric])
        if tmp.empty:
            continue
        conditions = [str(c) for c in sorted(tmp["condition"].dropna().unique())]
        data = [tmp.loc[tmp["condition"].astype(str) == c, metric].to_numpy(dtype=float) for c in conditions]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.boxplot(data, labels=conditions, showmeans=True)
        ax.set_title(f"{metric} by condition ({suffix.replace('_', ' ')})", pad=14)
        ax.set_xlabel("Condition")
        ax.set_ylabel(metric)
        ax.grid(True, axis="y", alpha=0.25)
        fig.autofmt_xdate(rotation=20)
        fig.tight_layout()
        paths.append(write_png(fig, out_dir / f"04_step4_group_{suffix}__{safe_piece(metric)}_by_condition.png"))
    return paths


def plot_qc_counts(trial_df: pd.DataFrame, out_dir: Path) -> list[str]:
    paths = []
    for col in ["trial_segment_qc_flag", "dyad_pretrial_qc_flag", "dyad_rr_reliability_label", "condition"]:
        if col not in trial_df.columns:
            continue
        counts = trial_df[col].astype(str).fillna("NA").value_counts(dropna=False).sort_index()
        if counts.empty:
            continue
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(counts.index.astype(str), counts.values)
        ax.set_title(f"Step 4 included trial counts by {col}", pad=14)
        ax.set_xlabel(col)
        ax.set_ylabel("Number of paired trial windows")
        ax.grid(True, axis="y", alpha=0.25)
        fig.autofmt_xdate(rotation=25)
        fig.tight_layout()
        paths.append(write_png(fig, out_dir / f"04_step4_counts_by_{safe_piece(col)}.png"))
    return paths



def pretty_metric_label(metric: str) -> str:
    labels = {
        "pearson_r": "Pearson r",
        "spearman_r": "Spearman r",
        "windowed_pearson_mean": "Windowed Pearson r",
        "phase_locking_index_pli": "PLI",
        "phase_locking_value_plv": "PLV",
        "crosscorr_max_abs_r": "Max |cross-corr|",
        "envelope_corr": "Envelope corr",
        "mean_coherence_0p04_0p40_hz": "Mean coherence",
        "dyad_rr_reliability_min": "Dyad RR reliability",
    }
    return labels.get(metric, metric.replace("_", " "))


def select_primary_subset(trial_df: pd.DataFrame) -> pd.DataFrame:
    """Return the cleanest display subset for presentation plots.

    This does not delete rows from result tables. It only creates cleaner summary
    figures for discussion with Veronica/Cirkeline.
    """
    df = trial_df.copy()
    if "main_analysis_eligible" in df.columns:
        df = df[df["main_analysis_eligible"].map(bool_from_any)].copy()
    if "condition" in df.columns:
        df = df[df["condition"].astype(str).isin(["FF", "FE"])].copy()
    return df


def mean_ci95(values: np.ndarray) -> tuple[float, float, int]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = len(arr)
    if n == 0:
        return np.nan, np.nan, 0
    mean = float(np.nanmean(arr))
    if n < 2:
        return mean, np.nan, n
    ci = 1.96 * float(np.nanstd(arr, ddof=1)) / math.sqrt(n)
    return mean, ci, n


def plot_presentation_key_metric_means(trial_df: pd.DataFrame, out_dir: Path) -> list[str]:
    df = select_primary_subset(trial_df)
    if df.empty or "condition" not in df.columns:
        return []
    metrics = [
        "pearson_r", "spearman_r", "windowed_pearson_mean",
        "phase_locking_index_pli", "phase_locking_value_plv",
        "crosscorr_max_abs_r", "envelope_corr", "dyad_rr_reliability_min",
    ]
    metrics = [m for m in metrics if m in df.columns]
    if not metrics:
        return []
    conditions = [c for c in ["FF", "FE"] if c in set(df["condition"].astype(str))]
    x = np.arange(len(metrics))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(12, len(metrics) * 1.6), 6.5))
    for j, cond in enumerate(conditions):
        means = []
        cis = []
        ns = []
        for m in metrics:
            mean, ci, n = mean_ci95(pd.to_numeric(df.loc[df["condition"].astype(str) == cond, m], errors="coerce").to_numpy())
            means.append(mean)
            cis.append(0 if not np.isfinite(ci) else ci)
            ns.append(n)
        offset = (j - (len(conditions)-1)/2) * width
        ax.bar(x + offset, means, width=width, yerr=cis, capsize=4, label=f"{cond} (mean ± 95% CI)")
    ax.axhline(0, linewidth=1, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels([pretty_metric_label(m) for m in metrics], rotation=25, ha="right")
    ax.set_ylabel("Metric value")
    ax.set_title("Key synchrony and QC metrics by condition\nMain-analysis eligible FF/FE trials", pad=18)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=max(1, len(conditions)), frameon=True)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout(rect=[0, 0.12, 1, 1])
    path = out_dir / "presentation_01_key_synchrony_metrics_main_analysis_mean_ci.png"
    return [write_png(fig, path, dpi=180)]


def plot_presentation_qc_filter_counts(trial_df: pd.DataFrame, out_dir: Path) -> list[str]:
    if trial_df.empty or "condition" not in trial_df.columns:
        return []
    rows = []
    for cond in ["FF", "FE"]:
        sub = trial_df[trial_df["condition"].astype(str) == cond].copy()
        if sub.empty:
            continue
        main = sub[sub["main_analysis_eligible"].map(bool_from_any)] if "main_analysis_eligible" in sub.columns else sub
        clean = main[main.get("trial_segment_qc_flag", pd.Series(index=main.index, dtype=object)).astype(str).str.lower().eq("clean")] if "trial_segment_qc_flag" in main.columns else main
        reliable = clean[clean.get("dyad_rr_reliability_label", pd.Series(index=clean.index, dtype=object)).astype(str).str.lower().eq("reliable")] if "dyad_rr_reliability_label" in clean.columns else clean
        rows.extend([
            {"condition": cond, "stage": "All computed", "count": len(sub)},
            {"condition": cond, "stage": "Main eligible", "count": len(main)},
            {"condition": cond, "stage": "Main + clean QC", "count": len(clean)},
            {"condition": cond, "stage": "Main + clean + reliable", "count": len(reliable)},
        ])
    if not rows:
        return []
    plot_df = pd.DataFrame(rows)
    stages = ["All computed", "Main eligible", "Main + clean QC", "Main + clean + reliable"]
    conditions = [c for c in ["FF", "FE"] if c in set(plot_df["condition"])]
    x = np.arange(len(stages))
    width = 0.35
    fig, ax = plt.subplots(figsize=(11, 6.5))
    for j, cond in enumerate(conditions):
        vals = [int(plot_df[(plot_df["condition"] == cond) & (plot_df["stage"] == st)]["count"].sum()) for st in stages]
        offset = (j - (len(conditions)-1)/2) * width
        bars = ax.bar(x + offset, vals, width=width, label=cond)
        ax.bar_label(bars, padding=3, fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(stages, rotation=18, ha="right")
    ax.set_ylabel("Number of trial windows")
    ax.set_title("Step 4 trial counts under common QC filters", pad=18)
    ax.legend(title="Condition", loc="upper right", frameon=True)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    return [write_png(fig, out_dir / "presentation_02_qc_filter_counts_by_condition.png", dpi=180)]


def plot_presentation_metric_correlation_heatmap(trial_df: pd.DataFrame, out_dir: Path) -> list[str]:
    df = select_primary_subset(trial_df)
    metrics = [
        "pearson_r", "spearman_r", "windowed_pearson_mean", "phase_locking_index_pli",
        "phase_locking_value_plv", "crosscorr_max_abs_r", "envelope_corr", "dyad_rr_reliability_min",
    ]
    metrics = [m for m in metrics if m in df.columns]
    if len(metrics) < 2:
        return []
    mat = df[metrics].apply(pd.to_numeric, errors="coerce").corr()
    fig, ax = plt.subplots(figsize=(9.5, 8))
    im = ax.imshow(mat.to_numpy(dtype=float), vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_yticks(np.arange(len(metrics)))
    ax.set_xticklabels([pretty_metric_label(m) for m in metrics], rotation=35, ha="right")
    ax.set_yticklabels([pretty_metric_label(m) for m in metrics])
    for i in range(len(metrics)):
        for j in range(len(metrics)):
            val = mat.iloc[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("Agreement among Step 4 synchrony metrics\nMain-analysis eligible FF/FE trials", pad=18)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Correlation between metrics")
    fig.tight_layout()
    return [write_png(fig, out_dir / "presentation_03_metric_correlation_heatmap_main_analysis.png", dpi=180)]


def plot_presentation_windowed_summary(window_df: pd.DataFrame, out_dir: Path) -> list[str]:
    if window_df.empty or "condition" not in window_df.columns or "window_pearson_r" not in window_df.columns:
        return []
    df = window_df.copy()
    if "main_analysis_eligible" in df.columns:
        df = df[df["main_analysis_eligible"].map(bool_from_any)].copy()
    df = df[df["condition"].astype(str).isin(["FF", "FE"])].copy()
    if df.empty:
        return []
    # Average by window index and condition for a clean temporal summary.
    df["window_index"] = pd.to_numeric(df["window_index"], errors="coerce")
    df["window_pearson_r"] = pd.to_numeric(df["window_pearson_r"], errors="coerce")
    summary = df.groupby(["condition", "window_index"], dropna=True)["window_pearson_r"].agg(["mean", "count", "std"]).reset_index()
    summary["se"] = summary["std"] / np.sqrt(summary["count"].clip(lower=1))
    fig, ax = plt.subplots(figsize=(10.5, 6))
    for cond in ["FF", "FE"]:
        sub = summary[summary["condition"].astype(str) == cond].sort_values("window_index")
        if sub.empty:
            continue
        x = sub["window_index"].to_numpy(dtype=float)
        y = sub["mean"].to_numpy(dtype=float)
        se = sub["se"].fillna(0).to_numpy(dtype=float)
        ax.plot(x, y, marker="o", linewidth=1.8, label=cond)
        ax.fill_between(x, y - 1.96*se, y + 1.96*se, alpha=0.12)
    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_xlabel("Window index within trial")
    ax.set_ylabel("Mean windowed Pearson r")
    ax.set_title("Temporal synchrony summary by condition\nMain-analysis eligible FF/FE trials", pad=18)
    ax.legend(title="Condition", frameon=True)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return [write_png(fig, out_dir / "presentation_04_windowed_pearson_by_condition.png", dpi=180)]


def save_presentation_group_figures(trial_df: pd.DataFrame, window_df: pd.DataFrame, out_dir: Path) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    paths.extend(plot_presentation_key_metric_means(trial_df, out_dir))
    paths.extend(plot_presentation_qc_filter_counts(trial_df, out_dir))
    paths.extend(plot_presentation_metric_correlation_heatmap(trial_df, out_dir))
    paths.extend(plot_presentation_windowed_summary(window_df, out_dir))
    return paths

def save_excel(df: pd.DataFrame, path: Path) -> None:
    try:
        df.to_excel(path, index=False)
    except Exception as exc:
        print(f"WARNING: could not write Excel file {path}: {exc}")


# =============================================================================
# Main
# =============================================================================


def main() -> int:
    start_time = time.time()
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    paired_hr_dir = resolve_path(root, args.paired_hr_dir)
    trial_summary_path = resolve_path(root, args.trial_summary)
    reliability_path = resolve_path(root, args.reliability_summary)
    analysis_units_path = resolve_path(root, args.analysis_units)
    rr_dir = resolve_path(root, args.rr_dir)
    out_dir = resolve_path(root, args.out_dir)

    if args.window_sec <= 0:
        raise ValueError("--window-sec must be > 0")
    if not (0 <= args.window_overlap < 1):
        raise ValueError("--window-overlap must be >= 0 and < 1")
    if not paired_hr_dir.exists():
        raise FileNotFoundError(f"Could not find Step 3 paired HR folder: {paired_hr_dir}")

    if out_dir.exists() and args.overwrite:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table_dir = out_dir / "tables"
    plot_root = out_dir / "plots"
    table_dir.mkdir(parents=True, exist_ok=True)
    plot_dirs = {
        "lena04": plot_root / "lena_04_windowed_cardiac_synchrony",
        "comprehensive": plot_root / "lena_04b_comprehensive_poincare",
        "ellipses": plot_root / "lena_04b_overlaid_ellipses",
        "temporal_poincare": plot_root / "lena_04b_temporal_poincare_patterns",
    }
    for p in plot_dirs.values():
        p.mkdir(parents=True, exist_ok=True)

    print(f"Script version: {SCRIPT_VERSION}")
    print(f"Root: {root}")
    print(f"Paired HR folder: {paired_hr_dir}")
    print(f"Output folder: {out_dir}")

    rr_cache = build_rr_file_cache(rr_dir)
    if rr_cache:
        print(f"Step 2 corrected RR files found for Lena 04b Poincare/Bland-Altman: {len(rr_cache)}")
    else:
        msg = f"WARNING: no Step 2 corrected RR files found in {rr_dir}; Poincare/Bland-Altman will use HR-equivalent fallback unless strict mode is on."
        print(msg)
        if args.require_direct_rr_for_poincare:
            raise FileNotFoundError(msg)

    dtype_ids = {"recording_folder": str, "dyad_id": str, "sensor_A": str, "sensor_B": str, "condition": str}
    trial_summary = read_optional_csv(trial_summary_path, dtype=dtype_ids)
    reliability_summary = read_optional_csv(reliability_path, dtype=dtype_ids)
    analysis_units = read_optional_csv(analysis_units_path, dtype=dtype_ids)

    if trial_summary.empty:
        print(f"WARNING: Step 3 summary not found or empty: {trial_summary_path}")
    else:
        print(f"Step 3 summary rows: {len(trial_summary)}")
    if reliability_summary.empty:
        print(f"WARNING: Step 3a reliability summary not found or empty: {reliability_path}")
    else:
        print(f"Step 3a dyad reliability rows: {len(reliability_summary)}")
    if not analysis_units.empty:
        print(f"Step 1 analysis-unit rows: {len(analysis_units)}")

    paired_files = sorted([p for p in paired_hr_dir.glob("*.csv") if p.is_file() and not p.name.startswith("._")])
    print(f"Paired HR files found: {len(paired_files)}")
    if len(paired_files) == 0:
        raise FileNotFoundError(f"No paired trial HR CSV files found in {paired_hr_dir}")

    manifest = build_file_manifest(paired_hr_dir, trial_summary)
    manifest.to_csv(table_dir / "04_step4_file_manifest_EG.csv", index=False)

    trial_rows: list[dict[str, Any]] = []
    window_tables: list[pd.DataFrame] = []
    failure_rows: list[dict[str, Any]] = []
    plot_manifest_rows: list[dict[str, str]] = []

    should_plot_all = args.max_trial_plots < 0
    plot_count = 0

    for i, file_path in enumerate(paired_files, start=1):
        try:
            df = read_paired_hr_file(file_path)
            meta = infer_row_metadata(df, file_path, trial_summary, reliability_summary)
            trial_metrics, window_df, rr_pair = compute_trial_metrics(df, file_path, meta, args, rr_cache)
            trial_rows.append(trial_metrics)
            if not window_df.empty:
                window_tables.append(window_df)

            save_this_plot = should_plot_all or (args.max_trial_plots > 0 and plot_count < args.max_trial_plots)
            if save_this_plot:
                plot_manifest_rows.extend(save_lena_style_plots(df, window_df, meta, trial_metrics, plot_dirs, rr_pair))
                plot_count += 1
        except Exception as exc:
            failure_rows.append({"paired_hr_file": str(file_path), "paired_hr_filename": file_path.name, "error": str(exc)})

        if i % 25 == 0 or i == len(paired_files):
            print(f"  processed {i}/{len(paired_files)} files")

    trial_df = pd.DataFrame(trial_rows)
    window_df_all = pd.concat(window_tables, ignore_index=True) if window_tables else pd.DataFrame()
    failures = pd.DataFrame(failure_rows)
    plot_manifest = pd.DataFrame(plot_manifest_rows)

    # Stable column order: metadata first, metrics second.
    meta_first = [
        "recording_folder", "dyad_id", "pair1", "participant_A", "participant_B", "sensor_A", "sensor_B",
        "candidate_window", "trial", "is_practice", "condition", "is_pilot", "exclude_from_main_analysis",
        "main_analysis_eligible", "exclude_reason", "usable_for_dyadic_ecg", "trial_segment_qc_flag",
        "dyad_pretrial_qc_flag", "pretrial_qc_flag_A", "pretrial_qc_flag_B", "dyad_rr_reliability_label",
        "dyad_rr_reliability_min", "dyad_rr_reliability_mean", "poincare_bland_altman_rr_source", "poincare_rr_file_A", "poincare_rr_file_B", "paired_hr_filename", "paired_hr_file",
    ]
    if not trial_df.empty:
        cols = [c for c in meta_first if c in trial_df.columns] + [c for c in trial_df.columns if c not in meta_first]
        trial_df = trial_df[cols]

    trial_csv = table_dir / "04_step4_trial_level_synchrony_metrics_EG.csv"
    window_csv = table_dir / "04_step4_window_level_synchrony_metrics_EG.csv"
    failures_csv = table_dir / "04_step4_failures_EG.csv"
    plot_manifest_csv = table_dir / "04_step4_plot_manifest_EG.csv"

    trial_df.to_csv(trial_csv, index=False)
    window_df_all.to_csv(window_csv, index=False)
    failures.to_csv(failures_csv, index=False)
    plot_manifest.to_csv(plot_manifest_csv, index=False)
    save_excel(trial_df, table_dir / "04_step4_trial_level_synchrony_metrics_EG.xlsx")
    save_excel(window_df_all, table_dir / "04_step4_window_level_synchrony_metrics_EG.xlsx")

    cond_all = summarize_by_condition(trial_df, main_only=False)
    cond_main = summarize_by_condition(trial_df, main_only=True)
    cond_all.to_csv(table_dir / "04_step4_condition_level_summary_all_windows_EG.csv", index=False)
    cond_main.to_csv(table_dir / "04_step4_condition_level_summary_main_analysis_EG.csv", index=False)
    save_excel(cond_all, table_dir / "04_step4_condition_level_summary_all_windows_EG.xlsx")
    save_excel(cond_main, table_dir / "04_step4_condition_level_summary_main_analysis_EG.xlsx")

    group_plot_paths: list[str] = []

    # Summary text.
    n_main = int(trial_df["main_analysis_eligible"].map(bool_from_any).sum()) if "main_analysis_eligible" in trial_df.columns and not trial_df.empty else 0
    n_practice = int(trial_df["is_practice"].map(bool_from_any).sum()) if "is_practice" in trial_df.columns and not trial_df.empty else 0
    n_pilot = int(trial_df["is_pilot"].map(bool_from_any).sum()) if "is_pilot" in trial_df.columns and not trial_df.empty else 0
    elapsed = time.time() - start_time

    summary = {
        "script_version": SCRIPT_VERSION,
        "root": str(root),
        "paired_hr_dir": str(paired_hr_dir),
        "out_dir": str(out_dir),
        "rr_dir": str(rr_dir),
        "n_step2_rr_files_found_for_poincare": len(rr_cache),
        "require_direct_rr_for_poincare": bool(args.require_direct_rr_for_poincare),
        "n_paired_hr_files_found": len(paired_files),
        "n_successful_trial_metric_rows": int(len(trial_df)),
        "n_failed_files": int(len(failures)),
        "n_window_level_rows": int(len(window_df_all)),
        "n_main_analysis_eligible_rows": n_main,
        "n_practice_rows": n_practice,
        "n_pilot_rows": n_pilot,
        "n_lena_style_trial_plot_sets_requested": "all" if should_plot_all else int(args.max_trial_plots),
        "n_lena_style_trial_plot_sets_attempted": int(plot_count),
        "n_plot_manifest_rows": int(len(plot_manifest)),
        "n_group_plots": int(len(group_plot_paths)),
        "window_sec": args.window_sec,
        "window_overlap": args.window_overlap,
        "max_lag_sec": args.max_lag_sec,
        "elapsed_sec": elapsed,
    }
    with open(table_dir / "04_step4_run_summary_EG.txt", "w", encoding="utf-8") as f:
        f.write("Step 4 TableTask dyadic synchrony run summary\n")
        f.write("=" * 54 + "\n")
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")
        f.write("\nOutput tables\n")
        f.write(f"trial_level: {trial_csv}\n")
        f.write(f"window_level: {window_csv}\n")
        f.write(f"condition_all: {table_dir / '04_step4_condition_level_summary_all_windows_EG.csv'}\n")
        f.write(f"condition_main: {table_dir / '04_step4_condition_level_summary_main_analysis_EG.csv'}\n")
        f.write(f"failures: {failures_csv}\n")
        f.write("\nInterpretation note\n")
        f.write("All Step 3 paired windows are computed when possible. Use main_analysis_eligible and QC/reliability columns for later filtering.\n")
        f.write("Poincare/Bland-Altman prefer Step 2 corrected beat-to-beat RR files, matching Lena Notebook 04b. In strict mode, those RR files must also contain usable Unix time so the script can extract the specific trial window. The table column poincare_bland_altman_rr_source records whether direct trial-window RR or a fallback was used.\n")
        f.write("Lena Notebook 04 synchrony metrics use a common preprocessing path: paired finite samples, linear detrending, z-scoring, then metric calculation. PLI/PLV, Pearson, cross-correlation, coherence, envelope, and optional CRQA metrics are calculated from this Lena-style preprocessing. Lena Notebook 04b leader-follower columns use z-scored HR, without Notebook 04 linear detrending, and include cross-correlation lag, lag-1 directed correlations, phase directionality, majority-vote overall leader, and optional Granger causality when statsmodels is installed. Cross-correlation leader labels follow Lena Notebook 04b's stated convention in crosscorr_leader and crosscorr_leader_lena_convention: positive lag means A leads B; negative lag means B leads A. The script also saves crosscorr_leader_scipy_signal_convention and lf_xcorr_leader_scipy_signal_convention because scipy.signal.correlate(A, B) is often interpreted with the opposite physical lead direction. Use numeric lag_s as the primary directionality result.\n")
        f.write("Lena's Notebook 04/04b displayed figures in notebooks; this script saves them as PNG files.\n")

    with open(table_dir / "04_step4_run_summary_EG.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    readme = out_dir / "README_04_step4_outputs_EG.txt"
    readme.write_text(
        "Step 4 outputs for Veronica TableTask synchrony analysis\n"
        "========================================================\n\n"
        "Primary table:\n"
        "  tables/04_step4_trial_level_synchrony_metrics_EG.csv\n\n"
        "Window table:\n"
        "  tables/04_step4_window_level_synchrony_metrics_EG.csv\n\n"
        "Main condition summaries:\n"
        "  tables/04_step4_condition_level_summary_all_windows_EG.csv\n"
        "  tables/04_step4_condition_level_summary_main_analysis_EG.csv\n\n"
        "Lena-style plot folders:\n"
        "  plots/lena_04_windowed_cardiac_synchrony/\n"
        "  plots/lena_04b_comprehensive_poincare/\n"
        "  plots/lena_04b_overlaid_ellipses/\n"
        "  plots/lena_04b_temporal_poincare_patterns/\n\n"
        "Recommended first checks:\n"
        "  1. Open tables/04_step4_run_summary_EG.txt.\n"
        "  2. Confirm n_successful_trial_metric_rows equals the expected Step 3 paired windows, usually 156.\n"
        "  3. Open tables/04_step4_failures_EG.csv. It should be empty or explain exactly which files failed.\n"
        "  4. Use main_analysis_eligible for primary analysis filtering. Do not drop QC rows silently.\n"
        "  5. Lena-style synchrony metrics use detrended, z-scored HR before metric calculation.\n"
        "  6. Lena 04b leader-follower columns are prefixed with lf_ and granger_.\n",
        encoding="utf-8",
    )

    print("\nDone.")
    print(f"Successful Step 4 computations: {len(trial_df)} / {len(paired_files)}")
    print(f"Failures: {len(failures)}")
    print(f"Main-analysis eligible rows: {n_main}")
    print(f"Output: {out_dir}")
    print(f"Primary table: {trial_csv}")
    return 0 if len(failures) == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
