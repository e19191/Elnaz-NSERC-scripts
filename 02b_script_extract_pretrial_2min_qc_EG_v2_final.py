#!/usr/bin/env python3
"""
02b_script_extract_pretrial_2min_qc_EG_v2_final.py
By: Elnaz Ghasemi
Date: June 2026

Purpose
-------
Step 2b of Veronica/Cirkeline TableTask analysis pipeline.

This script extracts the first 2 minutes of each Movesense sensor recording as
an individual signal-quality reference period. Veronica requested this step to
check whether the ECG/HR signal is already noisy while participants are likely
sitting down before the table task begins.

Important boundary
------------------
This is a QC step only. It does not calculate final synchrony, PLI, or NSTE. It
also does not remove the first 2 minutes from the later trial analysis.

The main question answered by this step is:
  Is the signal already poor before task movement begins?

Interpretation:
  - Poor pretrial signal suggests sensor/contact/recording quality problems.
  - Clean pretrial signal but noisy task trials suggests movement-related noise.

Inputs
------
Expected folder layout, run from top-level TableTask folder:

TableTask/
  02b_script_extract_pretrial_2min_qc_EG_v2_final.py
  Table MoveSense Sensor/
  qc_outputs/
    veronica_analysis_units.csv
  processed_ecg_hr/
    hr_interpolated_4hz/
    rpeak_rr_corrected/
    qc/02_ecg_to_hr_processing_summary.csv

Outputs
-------
processed_ecg_hr/pretrial_2min_qc_v2/
  hr_segments_4hz/
    <recording_folder>__<sensor_id>__pretrial_2min_hr_4hz.csv
  rr_segments/
    <recording_folder>__<sensor_id>__pretrial_2min_rr_corrected.csv
  plots/
    hr_2min/
    ecg_20sec_rpeaks/
    dyad_hr_2min/
    qc_summary/
  qc/
    02b_pretrial_2min_qc_summary.csv
    02b_pretrial_2min_qc_summary.xlsx
    02b_pretrial_2min_dyad_summary.csv
    02b_pretrial_2min_plot_manifest.csv
    02b_pretrial_2min_run_summary.txt

How to run
----------
Recommended:

python3 02b_script_extract_pretrial_2min_qc_EG_v2_final.py \
  --root . \
  --analysis-units qc_outputs/veronica_analysis_units.csv \
  --processed-dir processed_ecg_hr \
  --movesense-folder "Table MoveSense Sensor" \
  --overwrite

Dependencies
------------
Required: pandas, numpy, matplotlib, openpyxl
Optional but recommended for ECG plots: neurokit2

If neurokit2 is unavailable, HR/RR QC still runs, but ECG-cleaned overlay plots
are skipped.
"""

from __future__ import annotations

import argparse
import math
import re
import shutil
import sys
import warnings
from pathlib import Path
from typing import Any, Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import neurokit2 as nk
    HAS_NEUROKIT = True
except Exception:
    nk = None
    HAS_NEUROKIT = False

warnings.filterwarnings("ignore")

LOCAL_TZ = ZoneInfo("America/Vancouver")
DEFAULT_PRETRIAL_DURATION_SEC = 120.0
DEFAULT_ECG_ZOOM_DURATION_SEC = 20.0
DEFAULT_TARGET_FS = 4.0
SCRIPT_VERSION = "v2_duplicate_metadata_column_fix"

RR_MIN_QC_MS = 300.0
RR_MAX_QC_MS = 2000.0
HR_EXTREME_LOW_BPM = 40.0
HR_EXTREME_HIGH_BPM = 200.0


# =============================================================================
# Arguments
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 2b: extract first 2 minutes of each sensor recording for pretrial signal-quality QC."
    )
    parser.add_argument("--root", type=str, default=".", help="Top-level TableTask folder. Default: current folder.")
    parser.add_argument(
        "--analysis-units",
        type=str,
        default="qc_outputs/veronica_analysis_units.csv",
        help="Path to Step 1 veronica_analysis_units.csv or .xlsx.",
    )
    parser.add_argument(
        "--processed-dir",
        type=str,
        default="processed_ecg_hr",
        help="Step 2 output folder. Default: processed_ecg_hr.",
    )
    parser.add_argument(
        "--movesense-folder",
        type=str,
        default="Table MoveSense Sensor",
        help="Raw Movesense folder name inside root. Needed only for ECG zoom plots.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="processed_ecg_hr/pretrial_2min_qc_v2",
        help="Output folder for Step 2b QC.",
    )
    parser.add_argument(
        "--pretrial-duration-sec",
        type=float,
        default=DEFAULT_PRETRIAL_DURATION_SEC,
        help="Pretrial QC duration in seconds from recording start. Default: 120 seconds.",
    )
    parser.add_argument(
        "--ecg-zoom-duration-sec",
        type=float,
        default=DEFAULT_ECG_ZOOM_DURATION_SEC,
        help="ECG zoom plot duration in seconds from recording start. Default: 20 seconds.",
    )
    parser.add_argument(
        "--skip-ecg-plots",
        action="store_true",
        help="Skip raw/cleaned ECG zoom plots. HR and RR QC still run.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing Step 2b output folder before writing new outputs.",
    )
    return parser.parse_args()


def resolve_path(root: Path, text: str) -> Path:
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()


# =============================================================================
# Utilities
# =============================================================================


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    # Defensive cleanup: if an input file somehow contains duplicate column
    # labels, keep the first occurrence. This prevents downstream metadata
    # assignment/reordering from failing or becoming ambiguous.
    df = df.loc[:, ~pd.Index(df.columns).duplicated()]
    return df


def normalize_sensor_id(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if math.isfinite(float(value)) and float(value).is_integer():
            return str(int(value))
        return str(value).strip()
    text = str(value).strip()
    if text.lower() in {"", "nan", "none"}:
        return ""
    if re.fullmatch(r"\d+\.0", text):
        return text.split(".")[0]
    if re.fullmatch(r"[0-9.]+[eE]\+?\d+", text):
        try:
            num = float(text)
            if num.is_integer():
                return str(int(num))
        except Exception:
            pass
    return text


def bool_from_any(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y", "t"}


def unix_to_local_string(unix_seconds: Any) -> str:
    if pd.isna(unix_seconds):
        return ""
    try:
        return str(pd.to_datetime(float(unix_seconds), unit="s", utc=True).tz_convert(str(LOCAL_TZ)))
    except Exception:
        return ""


def safe_filename_piece(value: Any) -> str:
    text = str(value).strip()
    text = re.sub(r"[^A-Za-z0-9_\-]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_") or "NA"


def write_png(fig: plt.Figure, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def safe_std(values: pd.Series | np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 2:
        return np.nan
    return float(np.nanstd(arr, ddof=1))


# =============================================================================
# Input tables and manifest
# =============================================================================


def read_analysis_units(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Could not find analysis-units file: {path}")

    dtype_map = {
        "recording_folder": str,
        "dyad_id": str,
        "pair1": str,
        "sensor_A": str,
        "sensor_B": str,
        "condition": str,
    }
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path, dtype=dtype_map)
    elif path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path, dtype=dtype_map)
    else:
        raise ValueError("analysis-units file must be .csv, .xlsx, or .xls")

    df = clean_column_names(df)
    required = ["recording_folder", "sensor_A", "sensor_B"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"analysis-units file is missing required columns: {missing}")

    for col in ["sensor_A", "sensor_B"]:
        df[col] = df[col].map(normalize_sensor_id)
    df["recording_folder"] = df["recording_folder"].astype(str).str.strip()
    return df


def read_step2_summary(processed_dir: Path) -> pd.DataFrame:
    path = processed_dir / "qc" / "02_ecg_to_hr_processing_summary.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find Step 2 processing summary: {path}\n"
            "Run Step 2 successfully before Step 2b."
        )
    df = pd.read_csv(path, dtype={"recording_folder": str, "sensor_id": str, "role": str, "dyad_id": str, "pair1": str})
    df = clean_column_names(df)
    df["sensor_id"] = df["sensor_id"].map(normalize_sensor_id)
    df["recording_folder"] = df["recording_folder"].astype(str).str.strip()
    return df


def build_sensor_manifest(analysis_units: pd.DataFrame, step2_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in analysis_units.iterrows():
        folder = str(row["recording_folder"]).strip()
        for role, sensor_col, participant_col in [
            ("A", "sensor_A", "participant_A"),
            ("B", "sensor_B", "participant_B"),
        ]:
            sensor_id = normalize_sensor_id(row.get(sensor_col, ""))
            if not sensor_id:
                continue
            rows.append(
                {
                    "recording_folder": folder,
                    "sensor_id": sensor_id,
                    "role": role,
                    "participant_id": row.get(participant_col, np.nan),
                    "dyad_id": row.get("dyad_id", ""),
                    "pair1": row.get("pair1", ""),
                    "is_pilot_session": bool_from_any(row.get("is_pilot", False)),
                }
            )
    manifest = pd.DataFrame(rows).drop_duplicates(subset=["recording_folder", "sensor_id"])
    manifest["sensor_id"] = manifest["sensor_id"].map(normalize_sensor_id)

    # Attach Step 2 summary fields. Step 2 is the source of recording start time.
    summary_cols = [
        "recording_folder", "sensor_id", "status", "first_ecg_unix", "last_ecg_unix",
        "first_ecg_local", "last_ecg_local", "ecg_duration_sec", "artifact_percent",
        "mean_hr_bpm_corrected", "min_hr_bpm_corrected", "max_hr_bpm_corrected",
        "rr_plausible_percent_corrected", "n_extreme_hr_corrected",
        "hr_4hz_file", "rpeak_rr_corrected_file", "sampling_rate_hz",
    ]
    existing = [c for c in summary_cols if c in step2_summary.columns]
    merged = manifest.merge(step2_summary[existing], on=["recording_folder", "sensor_id"], how="left", suffixes=("", "_step2"))
    merged = merged.sort_values(["recording_folder", "sensor_id"]).reset_index(drop=True)
    return merged


# =============================================================================
# File locating
# =============================================================================


def expected_hr_file(processed_dir: Path, recording_folder: str, sensor_id: str) -> Path:
    stem = f"{safe_filename_piece(recording_folder)}__{safe_filename_piece(sensor_id)}"
    return processed_dir / "hr_interpolated_4hz" / f"{stem}__hr_4hz.csv"


def expected_rr_file(processed_dir: Path, recording_folder: str, sensor_id: str) -> Path:
    stem = f"{safe_filename_piece(recording_folder)}__{safe_filename_piece(sensor_id)}"
    return processed_dir / "rpeak_rr_corrected" / f"{stem}__rpeaks_rr_corrected.csv"


def find_ecg_file(movesense_dir: Path, recording_folder: str, sensor_id: str) -> Optional[Path]:
    sensor_dir = movesense_dir / recording_folder / sensor_id
    if not sensor_dir.exists():
        return None
    patterns = [
        f"ECG-{recording_folder}.csv",
        f"{sensor_id}_ECG-{recording_folder}.csv",
        f"*ECG-{recording_folder}.csv",
        "ECG-*.csv",
        "*ECG-*.csv",
    ]
    matches: list[Path] = []
    for pattern in patterns:
        matches.extend(sensor_dir.glob(pattern))
    matches = sorted({p for p in matches if p.is_file() and not p.name.startswith("._")})
    return matches[0] if matches else None


def read_hr_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"recording_folder": str, "sensor_id": str})
    df = clean_column_names(df)
    required = ["TimeUnix", "HR_BPM"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"HR file missing columns {missing}: {path}")
    df["TimeUnix"] = pd.to_numeric(df["TimeUnix"], errors="coerce")
    df["HR_BPM"] = pd.to_numeric(df["HR_BPM"], errors="coerce")
    df = df.dropna(subset=["TimeUnix", "HR_BPM"]).sort_values("TimeUnix").reset_index(drop=True)
    return df


def read_rr_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"recording_folder": str, "sensor_id": str})
    df = clean_column_names(df)
    if "rr_time_unix" not in df.columns or "HR_BPM" not in df.columns or "rr_interval_ms" not in df.columns:
        raise ValueError(f"RR file does not contain expected RR/HR columns: {path}")
    df["rr_time_unix"] = pd.to_numeric(df["rr_time_unix"], errors="coerce")
    df["HR_BPM"] = pd.to_numeric(df["HR_BPM"], errors="coerce")
    df["rr_interval_ms"] = pd.to_numeric(df["rr_interval_ms"], errors="coerce")
    df = df.dropna(subset=["rr_time_unix", "HR_BPM", "rr_interval_ms"]).sort_values("rr_time_unix").reset_index(drop=True)
    return df


# =============================================================================
# Optional ECG loading for 20-second zoom plots
# =============================================================================


def extract_sampling_rate_from_header(header_line: str) -> Optional[float]:
    if not header_line:
        return None
    numbers = re.findall(r"\d+(?:\.\d+)?", header_line)
    if not numbers:
        return None
    try:
        rate = float(numbers[-1])
    except Exception:
        return None
    if 20 <= rate <= 1000:
        return rate
    return None


def estimate_sampling_rate_from_timestamps(timestamps: pd.Series) -> Optional[float]:
    ts = pd.to_numeric(timestamps, errors="coerce").dropna().sort_values()
    if len(ts) < 3:
        return None
    diffs = ts.diff().dropna()
    diffs = diffs[diffs > 0]
    if diffs.empty:
        return None
    median_dt = float(diffs.median())
    if median_dt <= 0:
        return None
    fs = 1.0 / median_dt
    if 20 <= fs <= 1000:
        return fs
    return None


def choose_sampling_rate(header_fs: Optional[float], timestamp_fs: Optional[float]) -> float:
    if header_fs is not None and timestamp_fs is not None:
        if abs(header_fs - timestamp_fs) / header_fs <= 0.10:
            return float(header_fs)
        return float(timestamp_fs)
    if header_fs is not None:
        return float(header_fs)
    if timestamp_fs is not None:
        return float(timestamp_fs)
    raise ValueError("Could not determine ECG sampling rate.")


def load_ecg_data(ecg_file: Path) -> tuple[pd.DataFrame, float]:
    with open(ecg_file, "r", encoding="utf-8", errors="ignore") as f:
        first_line = f.readline().strip()
    header_fs = extract_sampling_rate_from_header(first_line)

    df = pd.read_csv(ecg_file, skiprows=1)
    df = clean_column_names(df)
    if "Timestamp" not in df.columns or "Sample" not in df.columns:
        df = pd.read_csv(ecg_file)
        df = clean_column_names(df)

    missing = [c for c in ["Timestamp", "Sample"] if c not in df.columns]
    if missing:
        raise ValueError(f"ECG file missing columns {missing}: {ecg_file}")

    df = df[["Timestamp", "Sample"]].copy()
    df["Timestamp"] = pd.to_numeric(df["Timestamp"], errors="coerce")
    df["Sample"] = pd.to_numeric(df["Sample"], errors="coerce")
    df = df.dropna(subset=["Timestamp", "Sample"]).reset_index(drop=True)
    timestamp_fs = estimate_sampling_rate_from_timestamps(df["Timestamp"])
    fs = choose_sampling_rate(header_fs, timestamp_fs)
    return df, fs


# =============================================================================
# QC metrics and classification
# =============================================================================


def summarize_hr_segment(hr: pd.DataFrame, target_duration_sec: float) -> dict[str, Any]:
    if hr.empty:
        return {
            "n_hr_samples": 0,
            "hr_duration_sec": 0.0,
            "hr_coverage_percent": 0.0,
            "mean_hr_bpm": np.nan,
            "median_hr_bpm": np.nan,
            "sd_hr_bpm": np.nan,
            "min_hr_bpm": np.nan,
            "max_hr_bpm": np.nan,
            "hr_range_bpm": np.nan,
            "n_extreme_hr_4hz": 0,
            "extreme_hr_4hz_percent": np.nan,
        }
    duration = float(hr["TimeUnix"].max() - hr["TimeUnix"].min()) if len(hr) > 1 else 0.0
    vals = hr["HR_BPM"].to_numpy(dtype=float)
    extreme = (vals < HR_EXTREME_LOW_BPM) | (vals > HR_EXTREME_HIGH_BPM)
    return {
        "n_hr_samples": int(len(hr)),
        "hr_duration_sec": duration,
        "hr_coverage_percent": float(min(100.0, duration / target_duration_sec * 100.0)) if target_duration_sec > 0 else np.nan,
        "mean_hr_bpm": float(np.nanmean(vals)),
        "median_hr_bpm": float(np.nanmedian(vals)),
        "sd_hr_bpm": safe_std(vals),
        "min_hr_bpm": float(np.nanmin(vals)),
        "max_hr_bpm": float(np.nanmax(vals)),
        "hr_range_bpm": float(np.nanmax(vals) - np.nanmin(vals)),
        "n_extreme_hr_4hz": int(np.nansum(extreme)),
        "extreme_hr_4hz_percent": float(np.nanmean(extreme) * 100.0),
    }


def summarize_rr_segment(rr: pd.DataFrame) -> dict[str, Any]:
    if rr.empty:
        return {
            "n_rr_intervals": 0,
            "rr_mean_ms": np.nan,
            "rr_sd_ms": np.nan,
            "rr_min_ms": np.nan,
            "rr_max_ms": np.nan,
            "rr_plausible_percent": np.nan,
            "n_extreme_hr_rr": 0,
            "extreme_hr_rr_percent": np.nan,
        }
    rr_vals = rr["rr_interval_ms"].to_numpy(dtype=float)
    hr_vals = rr["HR_BPM"].to_numpy(dtype=float)
    if "rr_plausible_300_2000_ms" in rr.columns:
        plausible = rr["rr_plausible_300_2000_ms"].map(bool_from_any).to_numpy(dtype=bool)
    else:
        plausible = (rr_vals >= RR_MIN_QC_MS) & (rr_vals <= RR_MAX_QC_MS)
    if "hr_extreme_lt40_gt200" in rr.columns:
        extreme = rr["hr_extreme_lt40_gt200"].map(bool_from_any).to_numpy(dtype=bool)
    else:
        extreme = (hr_vals < HR_EXTREME_LOW_BPM) | (hr_vals > HR_EXTREME_HIGH_BPM)
    return {
        "n_rr_intervals": int(len(rr)),
        "rr_mean_ms": float(np.nanmean(rr_vals)),
        "rr_sd_ms": safe_std(rr_vals),
        "rr_min_ms": float(np.nanmin(rr_vals)),
        "rr_max_ms": float(np.nanmax(rr_vals)),
        "rr_plausible_percent": float(np.nanmean(plausible) * 100.0),
        "n_extreme_hr_rr": int(np.nansum(extreme)),
        "extreme_hr_rr_percent": float(np.nanmean(extreme) * 100.0),
    }


def classify_pretrial_quality(row: dict[str, Any]) -> tuple[str, str]:
    """Classify sensor-level pretrial QC using transparent objective rules.

    The labels are not automatic exclusion decisions. They identify recordings
    that need visual review before final synchrony, PLI, or NSTE.
    """
    reasons: list[str] = []
    poor = False
    caution = False

    duration = row.get("hr_duration_sec", np.nan)
    coverage = row.get("hr_coverage_percent", np.nan)
    rr_plaus = row.get("rr_plausible_percent", np.nan)
    extreme_pct = row.get("extreme_hr_4hz_percent", np.nan)
    hr_min = row.get("min_hr_bpm", np.nan)
    hr_max = row.get("max_hr_bpm", np.nan)
    hr_sd = row.get("sd_hr_bpm", np.nan)
    hr_range = row.get("hr_range_bpm", np.nan)
    n_rr = row.get("n_rr_intervals", 0)

    if pd.isna(duration) or duration < 60:
        poor = True
        reasons.append("less_than_60s_hr_available")
    elif duration < 100 or (not pd.isna(coverage) and coverage < 80):
        caution = True
        reasons.append("short_or_incomplete_pretrial_hr_coverage")

    if n_rr < 30:
        poor = True
        reasons.append("too_few_rr_intervals")

    if not pd.isna(rr_plaus):
        if rr_plaus < 90:
            poor = True
            reasons.append("rr_plausible_percent_below_90")
        elif rr_plaus < 95:
            caution = True
            reasons.append("rr_plausible_percent_90_to_95")

    if not pd.isna(extreme_pct):
        if extreme_pct > 5:
            poor = True
            reasons.append("extreme_hr_percent_above_5")
        elif extreme_pct > 0.5:
            caution = True
            reasons.append("some_extreme_hr_values")

    if not pd.isna(hr_min) and hr_min < 30:
        poor = True
        reasons.append("minimum_hr_below_30")
    elif not pd.isna(hr_min) and hr_min < 40:
        caution = True
        reasons.append("minimum_hr_below_40")

    if not pd.isna(hr_max) and hr_max > 220:
        poor = True
        reasons.append("maximum_hr_above_220")
    elif not pd.isna(hr_max) and hr_max > 200:
        caution = True
        reasons.append("maximum_hr_above_200")

    if not pd.isna(hr_sd) and hr_sd > 30:
        poor = True
        reasons.append("hr_sd_above_30")
    elif not pd.isna(hr_sd) and hr_sd > 20:
        caution = True
        reasons.append("hr_sd_20_to_30")

    if not pd.isna(hr_range) and hr_range > 150:
        poor = True
        reasons.append("hr_range_above_150")
    elif not pd.isna(hr_range) and hr_range > 100:
        caution = True
        reasons.append("hr_range_100_to_150")

    if poor:
        return "poor", "; ".join(reasons)
    if caution:
        return "caution", "; ".join(reasons)
    return "clean", "objective_metrics_within_thresholds"


# =============================================================================
# Plotting
# =============================================================================


def plot_pretrial_hr(hr: pd.DataFrame, title: str, path: Path, duration_sec: float) -> Optional[str]:
    if hr.empty:
        return None
    t = hr["TimeFromRecordingStartSec"].to_numpy(dtype=float)
    y = hr["HR_BPM"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(t, y, linewidth=1.3, alpha=0.85)
    ax.axhspan(HR_EXTREME_LOW_BPM, HR_EXTREME_HIGH_BPM, alpha=0.08, label="Plausible broad HR range 40-200 BPM")
    ax.axhline(np.nanmean(y), linestyle="--", linewidth=1.0, alpha=0.8, label=f"Mean {np.nanmean(y):.1f} BPM")
    ax.set_xlim(0, duration_sec)
    ax.set_xlabel("Seconds from recording start")
    ax.set_ylabel("Corrected HR (BPM)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    return write_png(fig, path)


def plot_pretrial_rr_poincare(rr: pd.DataFrame, title: str, path: Path) -> Optional[str]:
    if rr.empty or len(rr) < 3:
        return None
    rr_vals = rr["rr_interval_ms"].to_numpy(dtype=float)
    rr1 = rr_vals[:-1]
    rr2 = rr_vals[1:]
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(rr1, rr2, s=22, alpha=0.6)
    lo = float(np.nanmin([np.nanmin(rr1), np.nanmin(rr2)]))
    hi = float(np.nanmax([np.nanmax(rr1), np.nanmax(rr2)]))
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1, alpha=0.6)
    ax.set_xlabel("RR(n), ms")
    ax.set_ylabel("RR(n+1), ms")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    return write_png(fig, path)


def plot_ecg_zoom_with_rpeaks(
    ecg_file: Path,
    rr_segment: pd.DataFrame,
    recording_start_unix: float,
    duration_sec: float,
    title: str,
    path: Path,
) -> Optional[str]:
    if not HAS_NEUROKIT:
        return None
    try:
        ecg_df, fs = load_ecg_data(ecg_file)
        end_unix = recording_start_unix + duration_sec
        seg = ecg_df[(ecg_df["Timestamp"] >= recording_start_unix) & (ecg_df["Timestamp"] <= end_unix)].copy()
        if seg.empty or len(seg) < 10:
            return None
        clean = nk.ecg_clean(seg["Sample"].to_numpy(dtype=float), sampling_rate=fs, method="neurokit")
        seg["TimeFromStartSec"] = seg["Timestamp"] - recording_start_unix
        seg["ECG_Clean"] = clean

        # Approximate corrected R-peak times from RR interval endpoints. Use unique end/start times.
        rpeak_times = []
        for col in ["rpeak_time_unix_start", "rpeak_time_unix_end"]:
            if col in rr_segment.columns:
                vals = pd.to_numeric(rr_segment[col], errors="coerce").dropna().tolist()
                rpeak_times.extend(vals)
        rpeak_times = sorted(set([x for x in rpeak_times if recording_start_unix <= x <= end_unix]))

        fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
        axes[0].plot(seg["TimeFromStartSec"], seg["Sample"], linewidth=0.8, alpha=0.8)
        axes[0].set_ylabel("Raw ECG")
        axes[0].set_title(title + " - raw")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(seg["TimeFromStartSec"], seg["ECG_Clean"], linewidth=0.8, alpha=0.8, label="Cleaned ECG")
        if rpeak_times:
            # Mark nearest samples to corrected R-peak times.
            time_arr = seg["Timestamp"].to_numpy(dtype=float)
            clean_arr = seg["ECG_Clean"].to_numpy(dtype=float)
            idxs = [int(np.argmin(np.abs(time_arr - rt))) for rt in rpeak_times]
            axes[1].scatter(seg["TimeFromStartSec"].iloc[idxs], clean_arr[idxs], s=50, marker="x", linewidths=1.8, label="Corrected R-peaks")
        axes[1].set_xlabel("Seconds from recording start")
        axes[1].set_ylabel("Cleaned ECG")
        axes[1].set_title(title + " - cleaned with corrected R-peaks")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(loc="upper right")
        fig.tight_layout()
        return write_png(fig, path)
    except Exception:
        return None


def plot_dyad_pretrial_hr(dyad_df: pd.DataFrame, title: str, path: Path, duration_sec: float) -> Optional[str]:
    if dyad_df.empty:
        return None
    fig, ax = plt.subplots(figsize=(12, 5))
    for sensor_id, sub in dyad_df.groupby("sensor_id"):
        ax.plot(sub["TimeFromRecordingStartSec"], sub["HR_BPM"], linewidth=1.2, alpha=0.85, label=f"sensor {sensor_id}")
    ax.axhspan(HR_EXTREME_LOW_BPM, HR_EXTREME_HIGH_BPM, alpha=0.08)
    ax.set_xlim(0, duration_sec)
    ax.set_xlabel("Seconds from recording start")
    ax.set_ylabel("Corrected HR (BPM)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    return write_png(fig, path)


def plot_qc_flag_counts(summary: pd.DataFrame, path: Path) -> Optional[str]:
    if summary.empty or "pretrial_qc_flag" not in summary.columns:
        return None
    counts = summary["pretrial_qc_flag"].value_counts().reindex(["clean", "caution", "poor"]).dropna()
    fig, ax = plt.subplots(figsize=(6, 4))
    counts.plot(kind="bar", ax=ax)
    ax.set_xlabel("Pretrial QC flag")
    ax.set_ylabel("Number of sensor recordings")
    ax.set_title("Step 2b pretrial QC flag counts")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return write_png(fig, path)


# =============================================================================
# Main processing
# =============================================================================


def make_output_dirs(out_dir: Path, overwrite: bool) -> dict[str, Path]:
    if out_dir.exists() and overwrite:
        shutil.rmtree(out_dir)
    dirs = {
        "root": out_dir,
        "hr_segments": out_dir / "hr_segments_4hz",
        "rr_segments": out_dir / "rr_segments",
        "plots": out_dir / "plots",
        "plot_hr": out_dir / "plots" / "hr_2min",
        "plot_rr": out_dir / "plots" / "rr_poincare_2min",
        "plot_ecg": out_dir / "plots" / "ecg_20sec_rpeaks",
        "plot_dyad": out_dir / "plots" / "dyad_hr_2min",
        "plot_qc": out_dir / "plots" / "qc_summary",
        "qc": out_dir / "qc",
    }
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    return dirs


def process_one_sensor(row: pd.Series, processed_dir: Path, movesense_dir: Path, dirs: dict[str, Path], args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, str]], Optional[pd.DataFrame]]:
    folder = str(row["recording_folder"])
    sensor_id = normalize_sensor_id(row["sensor_id"])
    stem = f"{safe_filename_piece(folder)}__{safe_filename_piece(sensor_id)}"

    summary: dict[str, Any] = {
        "recording_folder": folder,
        "sensor_id": sensor_id,
        "role": row.get("role", ""),
        "participant_id": row.get("participant_id", ""),
        "dyad_id": row.get("dyad_id", ""),
        "pair1": row.get("pair1", ""),
        "is_pilot_session": bool_from_any(row.get("is_pilot_session", False)),
        "step2_status": row.get("status", ""),
        "status": "not_started",
        "error_message": "",
        "recording_start_unix": np.nan,
        "recording_start_local": "",
        "pretrial_start_unix": np.nan,
        "pretrial_end_unix": np.nan,
        "pretrial_start_local": "",
        "pretrial_end_local": "",
        "pretrial_duration_target_sec": float(args.pretrial_duration_sec),
        "hr_segment_file": "",
        "rr_segment_file": "",
        "hr_plot_file": "",
        "rr_poincare_plot_file": "",
        "ecg_zoom_plot_file": "",
        "artifact_percent_whole_recording": row.get("artifact_percent", np.nan),
        "rr_plausible_percent_whole_recording": row.get("rr_plausible_percent_corrected", np.nan),
        "n_extreme_hr_whole_recording": row.get("n_extreme_hr_corrected", np.nan),
    }
    plots: list[dict[str, str]] = []

    try:
        if str(row.get("status", "")).lower() != "success":
            raise RuntimeError("Step 2 status is not success for this sensor.")

        start_unix = pd.to_numeric(row.get("first_ecg_unix", np.nan), errors="coerce")
        if pd.isna(start_unix):
            raise RuntimeError("Missing first_ecg_unix in Step 2 summary.")
        start_unix = float(start_unix)
        end_unix = start_unix + float(args.pretrial_duration_sec)

        summary.update(
            {
                "recording_start_unix": start_unix,
                "recording_start_local": unix_to_local_string(start_unix),
                "pretrial_start_unix": start_unix,
                "pretrial_end_unix": end_unix,
                "pretrial_start_local": unix_to_local_string(start_unix),
                "pretrial_end_local": unix_to_local_string(end_unix),
            }
        )

        hr_path = expected_hr_file(processed_dir, folder, sensor_id)
        rr_path = expected_rr_file(processed_dir, folder, sensor_id)
        if not hr_path.exists():
            raise FileNotFoundError(f"Missing corrected 4 Hz HR file: {hr_path}")
        if not rr_path.exists():
            raise FileNotFoundError(f"Missing corrected RR file: {rr_path}")

        hr = read_hr_file(hr_path)
        rr = read_rr_file(rr_path)
        hr_seg = hr[(hr["TimeUnix"] >= start_unix) & (hr["TimeUnix"] <= end_unix)].copy()
        rr_seg = rr[(rr["rr_time_unix"] >= start_unix) & (rr["rr_time_unix"] <= end_unix)].copy()

        if not hr_seg.empty:
            # Step 2 HR files may already contain metadata columns such as
            # recording_folder, sensor_id, role, dyad_id, or pair1. Overwrite
            # these values instead of using DataFrame.insert(), because insert()
            # fails when the column already exists.
            hr_seg["recording_folder"] = folder
            hr_seg["sensor_id"] = sensor_id
            hr_seg["role"] = row.get("role", "")
            hr_seg["participant_id"] = row.get("participant_id", "")
            hr_seg["dyad_id"] = row.get("dyad_id", "")
            hr_seg["pair1"] = row.get("pair1", "")
            hr_seg["TimeFromRecordingStartSec"] = hr_seg["TimeUnix"] - start_unix
            hr_seg["segment_type"] = "pretrial_2min_qc"

            # Put metadata columns first without assuming whether they existed in
            # the Step 2 file.
            first_cols = [
                "segment_type", "recording_folder", "sensor_id", "role",
                "participant_id", "dyad_id", "pair1",
                "TimeUnix", "TimeFromRecordingStartSec", "HR_BPM",
            ]
            hr_seg = hr_seg[[c for c in first_cols if c in hr_seg.columns] + [c for c in hr_seg.columns if c not in first_cols]]

        if not rr_seg.empty:
            # Step 2 RR files may also already contain metadata columns. Use
            # assignment rather than insert() to avoid duplicate-column errors.
            rr_seg["segment_type"] = "pretrial_2min_qc"
            rr_seg["recording_folder"] = folder
            rr_seg["sensor_id"] = sensor_id
            rr_seg["role"] = row.get("role", "")
            rr_seg["participant_id"] = row.get("participant_id", "")
            rr_seg["dyad_id"] = row.get("dyad_id", "")
            rr_seg["pair1"] = row.get("pair1", "")
            rr_seg["TimeFromRecordingStartSec"] = rr_seg["rr_time_unix"] - start_unix

            first_cols = [
                "segment_type", "recording_folder", "sensor_id", "role",
                "participant_id", "dyad_id", "pair1",
                "rr_time_unix", "TimeFromRecordingStartSec",
                "rr_interval_ms", "HR_BPM",
            ]
            rr_seg = rr_seg[[c for c in first_cols if c in rr_seg.columns] + [c for c in rr_seg.columns if c not in first_cols]]

        hr_out = dirs["hr_segments"] / f"{stem}__pretrial_2min_hr_4hz.csv"
        rr_out = dirs["rr_segments"] / f"{stem}__pretrial_2min_rr_corrected.csv"
        hr_seg.to_csv(hr_out, index=False)
        rr_seg.to_csv(rr_out, index=False)
        summary["hr_segment_file"] = str(hr_out)
        summary["rr_segment_file"] = str(rr_out)

        summary.update(summarize_hr_segment(hr_seg, float(args.pretrial_duration_sec)))
        summary.update(summarize_rr_segment(rr_seg))

        flag, reasons = classify_pretrial_quality(summary)
        summary["pretrial_qc_flag"] = flag
        summary["pretrial_qc_reasons"] = reasons

        hr_plot = plot_pretrial_hr(
            hr_seg,
            title=f"Pretrial first 2 minutes HR QC\n{folder}, sensor {sensor_id}, role {row.get('role', '')}, flag={flag}",
            path=dirs["plot_hr"] / f"{stem}__pretrial_2min_hr.png",
            duration_sec=float(args.pretrial_duration_sec),
        )
        if hr_plot:
            summary["hr_plot_file"] = hr_plot
            plots.append({"recording_folder": folder, "sensor_id": sensor_id, "plot_type": "pretrial_2min_hr", "plot_file": hr_plot})

        rr_plot = plot_pretrial_rr_poincare(
            rr_seg,
            title=f"Pretrial first 2 minutes RR Poincare\n{folder}, sensor {sensor_id}, flag={flag}",
            path=dirs["plot_rr"] / f"{stem}__pretrial_2min_rr_poincare.png",
        )
        if rr_plot:
            summary["rr_poincare_plot_file"] = rr_plot
            plots.append({"recording_folder": folder, "sensor_id": sensor_id, "plot_type": "pretrial_2min_rr_poincare", "plot_file": rr_plot})

        if not args.skip_ecg_plots:
            ecg_file = find_ecg_file(movesense_dir, folder, sensor_id)
            if ecg_file is not None:
                ecg_plot = plot_ecg_zoom_with_rpeaks(
                    ecg_file=ecg_file,
                    rr_segment=rr_seg,
                    recording_start_unix=start_unix,
                    duration_sec=float(args.ecg_zoom_duration_sec),
                    title=f"Pretrial ECG zoom\n{folder}, sensor {sensor_id}",
                    path=dirs["plot_ecg"] / f"{stem}__pretrial_first20sec_ecg_rpeaks.png",
                )
                if ecg_plot:
                    summary["ecg_zoom_plot_file"] = ecg_plot
                    plots.append({"recording_folder": folder, "sensor_id": sensor_id, "plot_type": "pretrial_first20sec_ecg_rpeaks", "plot_file": ecg_plot})
            else:
                summary["ecg_zoom_plot_file"] = "raw_ecg_file_not_found"

        summary["status"] = "success"
        return summary, plots, hr_seg

    except Exception as exc:
        summary["status"] = "failed"
        summary["error_message"] = str(exc)
        return summary, plots, None


def create_dyad_outputs(summary_df: pd.DataFrame, hr_segments: list[pd.DataFrame], dirs: dict[str, Path], duration_sec: float) -> pd.DataFrame:
    if not hr_segments:
        return pd.DataFrame()
    all_hr = pd.concat([x for x in hr_segments if x is not None and not x.empty], ignore_index=True)
    rows: list[dict[str, Any]] = []

    for folder, sub_summary in summary_df.groupby("recording_folder"):
        sub_hr = all_hr[all_hr["recording_folder"] == folder].copy()
        dyad_id = str(sub_summary["dyad_id"].dropna().iloc[0]) if "dyad_id" in sub_summary and len(sub_summary.dropna(subset=["dyad_id"])) else ""
        flags = sorted(set(sub_summary["pretrial_qc_flag"].dropna().astype(str))) if "pretrial_qc_flag" in sub_summary else []
        if "poor" in flags:
            dyad_flag = "one_or_both_poor"
        elif "caution" in flags:
            dyad_flag = "one_or_both_caution"
        elif "clean" in flags and len(flags) == 1:
            dyad_flag = "both_clean"
        else:
            dyad_flag = "incomplete_or_failed"

        plot_path = ""
        if not sub_hr.empty:
            p = plot_dyad_pretrial_hr(
                sub_hr,
                title=f"Dyad pretrial first 2 minutes HR QC\n{folder}, {dyad_id}, flag={dyad_flag}",
                path=dirs["plot_dyad"] / f"{safe_filename_piece(folder)}__pretrial_2min_dyad_hr.png",
                duration_sec=duration_sec,
            )
            plot_path = p or ""

        rows.append(
            {
                "recording_folder": folder,
                "dyad_id": dyad_id,
                "n_sensors_expected": 2,
                "n_sensors_processed_successfully": int((sub_summary["status"] == "success").sum()),
                "sensor_flags": ";".join(flags),
                "dyad_pretrial_qc_flag": dyad_flag,
                "dyad_hr_plot_file": plot_path,
            }
        )
    return pd.DataFrame(rows).sort_values("recording_folder").reset_index(drop=True)


def write_outputs(summary_df: pd.DataFrame, dyad_df: pd.DataFrame, plot_manifest: pd.DataFrame, dirs: dict[str, Path], args: argparse.Namespace) -> None:
    qc_dir = dirs["qc"]
    summary_csv = qc_dir / "02b_pretrial_2min_qc_summary.csv"
    summary_xlsx = qc_dir / "02b_pretrial_2min_qc_summary.xlsx"
    dyad_csv = qc_dir / "02b_pretrial_2min_dyad_summary.csv"
    plot_csv = qc_dir / "02b_pretrial_2min_plot_manifest.csv"
    run_txt = qc_dir / "02b_pretrial_2min_run_summary.txt"

    summary_df.to_csv(summary_csv, index=False)
    dyad_df.to_csv(dyad_csv, index=False)
    plot_manifest.to_csv(plot_csv, index=False)

    try:
        with pd.ExcelWriter(summary_xlsx, engine="openpyxl") as writer:
            summary_df.to_excel(writer, sheet_name="sensor_pretrial_qc", index=False)
            dyad_df.to_excel(writer, sheet_name="dyad_pretrial_qc", index=False)
            plot_manifest.to_excel(writer, sheet_name="plot_manifest", index=False)
            wb = writer.book
            for sheet in wb.sheetnames:
                ws = wb[sheet]
                ws.freeze_panes = "A2"
                ws.auto_filter.ref = ws.dimensions
                for col_cells in ws.columns:
                    letter = col_cells[0].column_letter
                    width = 10
                    for cell in col_cells[:200]:
                        if cell.value is not None:
                            width = max(width, len(str(cell.value)) + 2)
                    ws.column_dimensions[letter].width = min(width, 70)
    except Exception as exc:
        print(f"Warning: could not write Excel workbook: {exc}", file=sys.stderr)

    plot_qc_flag_counts(summary_df, dirs["plot_qc"] / "02b_pretrial_qc_flag_counts.png")

    status_counts = summary_df["status"].value_counts(dropna=False).to_dict() if not summary_df.empty else {}
    flag_counts = summary_df["pretrial_qc_flag"].value_counts(dropna=False).to_dict() if "pretrial_qc_flag" in summary_df else {}
    dyad_flag_counts = dyad_df["dyad_pretrial_qc_flag"].value_counts(dropna=False).to_dict() if "dyad_pretrial_qc_flag" in dyad_df else {}

    with open(run_txt, "w", encoding="utf-8") as f:
        f.write("TableTask Step 2b pretrial 2-minute QC summary\n")
        f.write("================================================\n\n")
        f.write(f"Root: {Path(args.root).expanduser().resolve()}\n")
        f.write(f"Analysis units: {args.analysis_units}\n")
        f.write(f"Processed Step 2 folder: {args.processed_dir}\n")
        f.write(f"Movesense folder: {args.movesense_folder}\n")
        f.write(f"Output folder: {args.out_dir}\n")
        f.write(f"Pretrial duration: {args.pretrial_duration_sec} seconds\n")
        f.write(f"ECG zoom duration: {args.ecg_zoom_duration_sec} seconds\n")
        f.write(f"NeuroKit available for ECG plots: {HAS_NEUROKIT}\n")
        f.write(f"Skip ECG plots: {args.skip_ecg_plots}\n\n")
        f.write(f"Total sensor recordings evaluated: {len(summary_df)}\n")
        f.write(f"Successful sensor pretrial QC rows: {int((summary_df['status'] == 'success').sum()) if not summary_df.empty else 0}\n")
        f.write(f"Failed sensor pretrial QC rows: {int((summary_df['status'] == 'failed').sum()) if not summary_df.empty else 0}\n")
        f.write(f"Saved plot files recorded in manifest: {len(plot_manifest)}\n\n")
        f.write("Status counts:\n")
        for k, v in status_counts.items():
            f.write(f"  {k}: {v}\n")
        f.write("\nSensor pretrial QC flag counts:\n")
        for k, v in flag_counts.items():
            f.write(f"  {k}: {v}\n")
        f.write("\nDyad pretrial QC flag counts:\n")
        for k, v in dyad_flag_counts.items():
            f.write(f"  {k}: {v}\n")
        f.write("\nInterpretation note:\n")
        f.write("  Step 2b flags are QC labels only. They are not automatic exclusion decisions.\n")
        f.write("  Use these outputs to decide whether noisy sensors need trial-level review, exclusion, or sensitivity analysis.\n")

    print(f"Sensor QC summary CSV: {summary_csv}")
    print(f"Sensor QC summary Excel: {summary_xlsx}")
    print(f"Dyad QC summary CSV: {dyad_csv}")
    print(f"Plot manifest: {plot_csv}")
    print(f"Run summary: {run_txt}")


# =============================================================================
# Entrypoint
# =============================================================================


def main() -> None:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    analysis_units_path = resolve_path(root, args.analysis_units)
    processed_dir = resolve_path(root, args.processed_dir)
    movesense_dir = resolve_path(root, args.movesense_folder)
    out_dir = resolve_path(root, args.out_dir)

    if not processed_dir.exists():
        raise FileNotFoundError(f"Processed Step 2 folder not found: {processed_dir}")
    if not movesense_dir.exists() and not args.skip_ecg_plots:
        print(f"Warning: Movesense folder not found, ECG zoom plots will be skipped: {movesense_dir}", file=sys.stderr)

    dirs = make_output_dirs(out_dir, overwrite=args.overwrite)

    print("TableTask Step 2b: first 2-minute pretrial QC")
    print(f"Script version: {SCRIPT_VERSION}")
    print(f"Root: {root}")
    print(f"Analysis units: {analysis_units_path}")
    print(f"Processed Step 2 folder: {processed_dir}")
    print(f"Movesense folder: {movesense_dir}")
    print(f"Output folder: {out_dir}")
    print(f"Pretrial duration: {args.pretrial_duration_sec} seconds")
    print(f"NeuroKit available for ECG zoom plots: {HAS_NEUROKIT}")
    print("Boundary: this is QC only. It does not calculate final synchrony, PLI, or NSTE.\n")

    analysis_units = read_analysis_units(analysis_units_path)
    step2_summary = read_step2_summary(processed_dir)
    manifest = build_sensor_manifest(analysis_units, step2_summary)
    manifest_path = dirs["qc"] / "02b_pretrial_2min_sensor_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    print(f"Sensor recordings to evaluate: {len(manifest)}")
    print(f"Manifest saved to: {manifest_path}\n")

    summary_rows: list[dict[str, Any]] = []
    plot_rows: list[dict[str, str]] = []
    hr_segments: list[pd.DataFrame] = []

    for i, (_, row) in enumerate(manifest.iterrows(), start=1):
        print("=" * 90)
        print(f"[{i}/{len(manifest)}] {row['recording_folder']} / sensor {row['sensor_id']} / role {row.get('role', '')}")
        summary, plots, hr_seg = process_one_sensor(row, processed_dir, movesense_dir, dirs, args)
        summary_rows.append(summary)
        plot_rows.extend(plots)
        if hr_seg is not None and not hr_seg.empty:
            hr_segments.append(hr_seg)
        print(f"  Status: {summary['status']}")
        if summary["status"] == "success":
            print(f"  Pretrial flag: {summary.get('pretrial_qc_flag', '')}")
            print(f"  Reasons: {summary.get('pretrial_qc_reasons', '')}")
        else:
            print(f"  Error: {summary.get('error_message', '')}")

    summary_df = pd.DataFrame(summary_rows)
    plot_manifest = pd.DataFrame(plot_rows)
    dyad_df = create_dyad_outputs(summary_df, hr_segments, dirs, duration_sec=float(args.pretrial_duration_sec))
    # Add dyad plot rows to manifest.
    if not dyad_df.empty and "dyad_hr_plot_file" in dyad_df.columns:
        for _, r in dyad_df.iterrows():
            if isinstance(r.get("dyad_hr_plot_file", ""), str) and r.get("dyad_hr_plot_file", ""):
                plot_manifest = pd.concat(
                    [
                        plot_manifest,
                        pd.DataFrame(
                            [
                                {
                                    "recording_folder": r.get("recording_folder", ""),
                                    "sensor_id": "dyad",
                                    "plot_type": "dyad_pretrial_2min_hr",
                                    "plot_file": r.get("dyad_hr_plot_file", ""),
                                }
                            ]
                        ),
                    ],
                    ignore_index=True,
                )

    write_outputs(summary_df, dyad_df, plot_manifest, dirs, args)

    print("\n" + "=" * 90)
    print("Step 2b pretrial 2-minute QC complete")
    print(f"Successful sensor QC rows: {int((summary_df['status'] == 'success').sum()) if not summary_df.empty else 0}")
    print(f"Failed sensor QC rows: {int((summary_df['status'] == 'failed').sum()) if not summary_df.empty else 0}")
    if "pretrial_qc_flag" in summary_df.columns:
        print("Sensor QC flag counts:")
        for flag, count in summary_df["pretrial_qc_flag"].value_counts(dropna=False).items():
            print(f"  {flag}: {count}")
    print("\nNext: review Step 2b plots and summary before Step 3 trial segmentation.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("\nERROR: Step 2b pretrial QC failed.", file=sys.stderr)
        print(str(exc), file=sys.stderr)
        raise
