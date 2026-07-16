#!/usr/bin/env python3
"""
script_02_process_ecg_to_hr_EG.py
By: Elnaz Ghasemi, adapted from Lena Adel's 03_full_pipeline_ECG_processing.ipynb
Date: June 2026

Purpose
-------
Step 2 of Veronica/Cirkeline TableTask analysis pipeline.

This script adapts Lena's Notebook 3 full ECG processing pipeline for the
TableTask batch dataset. It preserves Lena's core ECG logic as much as possible:

  raw ECG -> NeuroKit2 ECG cleaning -> R-peak detection -> RR intervals
  -> instantaneous HR -> R-peak correction -> HRV metrics -> plots/results

Necessary TableTask changes
---------------------------
1. Batch processing over all unique recording_folder + sensor_id combinations
   listed in qc_outputs/veronica_analysis_units.csv or .xlsx.
2. Full sensor IDs are treated as text, never numeric variables.
3. Absolute Unix timing is preserved because Step 3 will segment HR using
   accel_start_unix and accel_end_unix from Step 1.
4. One processed output set is saved per recording_folder + sensor_id.
5. Plots are saved to disk rather than displayed in a notebook.
6. Optional whole-session dyad preview plots reproduce the spirit of Lena's
   Notebook 3 synchrony section, but they are labelled diagnostic only and are
   not the final FF/FE synchrony analysis.

Important boundary
------------------
This is still Step 2. It does not create final trial-level synchrony, PLI, or
NSTE results. Final synchrony/PLI/NSTE must be calculated later after Step 3
segments HR into table-motion trial windows.

Expected folder layout
----------------------
Run from the top-level TableTask folder:

TableTask/
  script_02_process_ecg_to_hr_EG.py
  Table MoveSense Sensor/
  qc_outputs/
    veronica_analysis_units.csv  or  veronica_analysis_units.xlsx

Main outputs
------------
processed_ecg_hr/
  hr_interpolated_4hz/
    <recording_folder>__<sensor_id>__hr_4hz.csv
  hr_interpolated_4hz_original/
    <recording_folder>__<sensor_id>__hr_4hz_original_uncorrected.csv
  rpeak_rr_corrected/
    <recording_folder>__<sensor_id>__rpeaks_rr_corrected.csv
  rpeak_rr_original/
    <recording_folder>__<sensor_id>__rpeaks_rr_original.csv
  hrv_metrics/
    <recording_folder>__<sensor_id>__hrv_corrected.csv
  plots/
    raw_vs_clean/
    rpeaks/
    hr_time_series/
    hr_before_after_correction/
    poincare/
    hrv_psd/
    qc_summary/
    dyad_session_preview_not_for_final_analysis/
  qc/
    02_sensor_processing_manifest.csv
    02_ecg_to_hr_processing_summary.csv
    02_ecg_to_hr_processing_summary.xlsx
    02_plot_manifest.csv
    02_run_summary.txt

How to run
----------
python3 script_02_process_ecg_to_hr_EG.py

Recommended explicit command:
python3 script_02_process_ecg_to_hr_EG.py \
  --root . \
  --analysis-units qc_outputs/veronica_analysis_units.csv \
  --movesense-folder "Table MoveSense Sensor" \
  --out-dir processed_ecg_hr \
  --make-dyad-preview

Dependency:
python3 -m pip install neurokit2 openpyxl scipy matplotlib pandas numpy
"""

from __future__ import annotations

import argparse
import math
import re
import shutil
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

# Headless plotting so the script works from Terminal and saves every plot.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy import signal as sp_signal
from scipy import stats

try:
    import neurokit2 as nk
except ImportError as exc:
    raise ImportError(
        "neurokit2 is required. Install it with:\n"
        "  python3 -m pip install neurokit2\n"
    ) from exc

warnings.filterwarnings("ignore")

LOCAL_TZ = ZoneInfo("America/Vancouver")
DEFAULT_TARGET_FS = 4.0
DEFAULT_QC_WINDOW_START_SEC = 60.0
DEFAULT_QC_WINDOW_DURATION_SEC = 10.0
RR_MIN_QC_MS = 300.0
RR_MAX_QC_MS = 2000.0
HR_EXTREME_LOW_BPM = 40.0
HR_EXTREME_HIGH_BPM = 200.0


# =============================================================================
# Argument parsing
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 2: process raw TableTask Movesense ECG into HR, HRV, QC plots, and summary files."
    )
    parser.add_argument("--root", type=str, default=".", help="Top-level TableTask folder. Default: current folder.")
    parser.add_argument(
        "--analysis-units",
        type=str,
        default="qc_outputs/veronica_analysis_units.csv",
        help="Path to Step 1 veronica_analysis_units.csv or .xlsx.",
    )
    parser.add_argument(
        "--movesense-folder",
        type=str,
        default="Table MoveSense Sensor",
        help="Raw Movesense folder name inside root.",
    )
    parser.add_argument("--out-dir", type=str, default="processed_ecg_hr", help="Output folder.")
    parser.add_argument("--target-fs", type=float, default=DEFAULT_TARGET_FS, help="Interpolated HR sampling rate in Hz.")
    parser.add_argument(
        "--ecg-method",
        type=str,
        default="neurokit",
        help="NeuroKit2 method for ecg_clean and ecg_peaks. Default preserves Lena: neurokit.",
    )
    parser.add_argument(
        "--qc-window-start-sec",
        type=float,
        default=DEFAULT_QC_WINDOW_START_SEC,
        help="Start time for 10-second ECG QC plots, in seconds from recording start.",
    )
    parser.add_argument(
        "--qc-window-duration-sec",
        type=float,
        default=DEFAULT_QC_WINDOW_DURATION_SEC,
        help="Duration for ECG QC plots in seconds.",
    )
    parser.add_argument(
        "--save-cleaned-ecg",
        action="store_true",
        help="Save full sample-level cleaned ECG files. Off by default because files are large.",
    )
    parser.add_argument(
        "--make-dyad-preview",
        action="store_true",
        help=(
            "Create whole-session dyad HR/correlation preview plots. These reproduce Lena Notebook 3's "
            "session-level synchrony idea but are diagnostic only, not final FF/FE analysis."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output folder if it already exists.",
    )
    return parser.parse_args()


def resolve_path(root: Path, path_text: str) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()


# =============================================================================
# General utilities
# =============================================================================


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def normalize_sensor_id(value: Any) -> str:
    """Convert sensor IDs to stable strings.

    Sensor IDs are labels, not numeric variables. This protects against Excel
    scientific notation and accidental .0 suffixes.
    """
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
            numeric = float(text)
            if numeric.is_integer():
                return str(int(numeric))
        except Exception:
            pass
    return text


def safe_filename_piece(value: Any) -> str:
    text = str(value).strip()
    text = re.sub(r"[^A-Za-z0-9_\-]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_") or "NA"


def unix_to_local_string(unix_seconds: float) -> str:
    if pd.isna(unix_seconds):
        return ""
    return str(pd.to_datetime(float(unix_seconds), unit="s", utc=True).tz_convert(str(LOCAL_TZ)))


def bool_from_any(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y", "t"}


def safe_std(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return np.nan
    return float(np.nanstd(x, ddof=1))


def write_png(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Step 1 analysis-unit input
# =============================================================================


def read_analysis_units(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Could not find Step 1 analysis-units file: {path}")

    dtype_map = {
        "recording_folder": str,
        "dyad_id": str,
        "sensor_A": str,
        "sensor_B": str,
        "condition": str,
        "trial_order": str,
        "pair1": str,
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

    if "condition" in df.columns:
        unknown_n = int((df["condition"].astype(str).str.lower() == "unknown").sum())
        if unknown_n > 0:
            raise ValueError(
                f"Step 1 file has {unknown_n} rows with condition == unknown. "
                "Use the verified final Step 1 output before processing ECG."
            )

    return df


def build_sensor_processing_manifest(analysis_units: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in analysis_units.iterrows():
        folder = str(row["recording_folder"]).strip()
        dyad_id = row.get("dyad_id", "")
        pair1 = row.get("pair1", "")
        is_pilot = bool_from_any(row.get("is_pilot", False))
        for role_label, sensor_col, participant_col in [
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
                    "role": role_label,
                    "participant_id": row.get(participant_col, np.nan),
                    "dyad_id": dyad_id,
                    "pair1": pair1,
                    "is_pilot_session": is_pilot,
                }
            )
    manifest = pd.DataFrame(rows).drop_duplicates(subset=["recording_folder", "sensor_id"])
    manifest = manifest.sort_values(["recording_folder", "sensor_id"]).reset_index(drop=True)
    return manifest


# =============================================================================
# ECG loading and sampling rate
# =============================================================================


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


def choose_sampling_rate(header_fs: Optional[float], timestamp_fs: Optional[float]) -> tuple[float, str, str]:
    if header_fs is not None and timestamp_fs is not None:
        rel_diff = abs(header_fs - timestamp_fs) / header_fs
        if rel_diff <= 0.10:
            return float(header_fs), "header", "Header sampling rate agrees with timestamp-derived estimate."
        return float(timestamp_fs), "timestamp_median_dt", (
            f"Header sampling rate ({header_fs:.3f}) differed from timestamp-derived estimate "
            f"({timestamp_fs:.3f}) by >10%; used timestamp estimate."
        )
    if header_fs is not None:
        return float(header_fs), "header", "Used header sampling rate; timestamp estimate unavailable."
    if timestamp_fs is not None:
        return float(timestamp_fs), "timestamp_median_dt", "Used timestamp-derived sampling rate; header unavailable."
    raise ValueError("Could not determine ECG sampling rate from header or timestamps.")


def load_ecg_data(ecg_file: Path) -> tuple[pd.DataFrame, float, str, str, Optional[float], Optional[float]]:
    """Load Movesense ECG file, preserving Lena's skiprows=1 approach."""
    with open(ecg_file, "r", encoding="utf-8", errors="ignore") as f:
        first_line = f.readline().strip()
    header_fs = extract_sampling_rate_from_header(first_line)

    try:
        df = pd.read_csv(ecg_file, skiprows=1)
        df = clean_column_names(df)
        if "Timestamp" not in df.columns or "Sample" not in df.columns:
            df = pd.read_csv(ecg_file)
            df = clean_column_names(df)
    except Exception as exc:
        raise RuntimeError(f"Could not read ECG CSV: {ecg_file}. Error: {exc}") from exc

    missing = [c for c in ["Timestamp", "Sample"] if c not in df.columns]
    if missing:
        raise ValueError(f"ECG file is missing columns {missing}: {ecg_file}")

    df = df[["Timestamp", "Sample"]].copy()
    df["Timestamp"] = pd.to_numeric(df["Timestamp"], errors="coerce")
    df["Sample"] = pd.to_numeric(df["Sample"], errors="coerce")
    df = df.dropna(subset=["Timestamp", "Sample"]).reset_index(drop=True)
    if df.empty:
        raise ValueError(f"ECG file has no valid Timestamp/Sample rows: {ecg_file}")

    timestamp_fs = estimate_sampling_rate_from_timestamps(df["Timestamp"])
    sampling_rate, source, note = choose_sampling_rate(header_fs, timestamp_fs)
    return df, sampling_rate, source, note, header_fs, timestamp_fs


# =============================================================================
# Lena-preserved ECG processing functions with batch-safe additions
# =============================================================================


def clean_ecg_signal(ecg_df: pd.DataFrame, sampling_rate: float, participant_name: str, method: str = "neurokit") -> np.ndarray:
    """Clean ECG signal using NeuroKit2. Preserves Lena's core logic."""
    print(f"  Cleaning ECG for {participant_name}...")
    raw_ecg = ecg_df["Sample"].values
    cleaned_ecg = nk.ecg_clean(raw_ecg, sampling_rate=sampling_rate, method=method)
    return np.asarray(cleaned_ecg, dtype=float)


def detect_rpeaks(ecg_clean: np.ndarray, sampling_rate: float, participant_name: str, method: str = "neurokit") -> tuple[np.ndarray, Any]:
    """Detect R-peaks in cleaned ECG signal. Preserves Lena's core logic."""
    print(f"  Detecting R-peaks for {participant_name}...")
    _, rpeaks = nk.ecg_peaks(ecg_clean, sampling_rate=sampling_rate, method=method)
    rpeak_indices = np.asarray(rpeaks["ECG_R_Peaks"], dtype=int)
    return rpeak_indices, rpeaks


def fix_rpeaks(rpeaks: np.ndarray, sampling_rate: float, participant_name: str) -> tuple[np.ndarray, dict[str, Any]]:
    """Correct R-peak detection errors using NeuroKit2 signal_fixpeaks.

    This preserves Lena's Kubios correction section and fixes the notebook bug by
    actually calling this function before recomputing corrected RR/HR.
    """
    print(f"  Correcting R-peaks for {participant_name}...")
    info: dict[str, Any] = {
        "original_peaks": int(len(rpeaks)),
        "corrected_peaks": int(len(rpeaks)),
        "ectopic": 0,
        "missed": 0,
        "extra": 0,
        "longshort": 0,
        "total_artifacts": 0,
        "net_change": 0,
        "fixpeaks_status": "not_run",
        "fixpeaks_error": "",
    }

    if len(rpeaks) < 3:
        info["fixpeaks_status"] = "skipped_too_few_peaks"
        return np.asarray(rpeaks, dtype=int), info

    try:
        artifacts, corrected_peaks = nk.signal_fixpeaks(
            rpeaks,
            sampling_rate=sampling_rate,
            iterative=True,
            method="Kubios",
        )
        corrected_peaks = np.asarray(corrected_peaks, dtype=float)
        corrected_peaks = corrected_peaks[np.isfinite(corrected_peaks)]
        corrected_peaks = np.rint(corrected_peaks).astype(int)
        corrected_peaks = np.unique(corrected_peaks[corrected_peaks >= 0])

        n_ectopic = len(artifacts.get("ectopic", []))
        n_missed = len(artifacts.get("missed", []))
        n_extra = len(artifacts.get("extra", []))
        n_longshort = len(artifacts.get("longshort", []))
        total = n_ectopic + n_missed + n_extra + n_longshort

        info.update(
            {
                "corrected_peaks": int(len(corrected_peaks)),
                "ectopic": int(n_ectopic),
                "missed": int(n_missed),
                "extra": int(n_extra),
                "longshort": int(n_longshort),
                "total_artifacts": int(total),
                "net_change": int(len(corrected_peaks) - len(rpeaks)),
                "fixpeaks_status": "success",
            }
        )
        return corrected_peaks, info
    except Exception as exc:
        info["fixpeaks_status"] = "failed_returned_original"
        info["fixpeaks_error"] = str(exc)
        return np.asarray(rpeaks, dtype=int), info


def compute_rr_intervals_from_rpeaks(
    rpeaks: np.ndarray,
    sampling_rate: float,
    first_ecg_unix: float,
    max_index: int,
    label: str,
) -> pd.DataFrame:
    """Compute RR intervals and HR from R-peaks, preserving Lena's math.

    Lena computes rpeak_times = rpeaks / sampling_rate. We preserve that relative
    timing and add absolute Unix time for the TableTask trial segmentation step.
    """
    rpeaks = np.asarray(rpeaks, dtype=int)
    rpeaks = np.unique(rpeaks[(rpeaks >= 0) & (rpeaks < max_index)])
    if len(rpeaks) < 2:
        return pd.DataFrame()

    rpeak_times_rel = rpeaks / sampling_rate
    rpeak_times_unix = first_ecg_unix + rpeak_times_rel

    rr_intervals = np.diff(rpeak_times_rel) * 1000.0
    rr_times_rel = rpeak_times_rel[:-1] + np.diff(rpeak_times_rel) / 2.0
    rr_times_unix = first_ecg_unix + rr_times_rel

    valid = np.isfinite(rr_intervals) & (rr_intervals > 0)
    if not np.any(valid):
        return pd.DataFrame()

    rr_intervals = rr_intervals[valid]
    rr_times_rel = rr_times_rel[valid]
    rr_times_unix = rr_times_unix[valid]
    rpeak_start = rpeaks[:-1][valid]
    rpeak_end = rpeaks[1:][valid]

    heart_rate = 60000.0 / rr_intervals
    plausible = (rr_intervals >= RR_MIN_QC_MS) & (rr_intervals <= RR_MAX_QC_MS)
    extreme_hr = (heart_rate < HR_EXTREME_LOW_BPM) | (heart_rate > HR_EXTREME_HIGH_BPM)

    return pd.DataFrame(
        {
            "interval_number": np.arange(1, len(rr_intervals) + 1),
            "rpeak_index_start": rpeak_start,
            "rpeak_index_end": rpeak_end,
            "rpeak_time_rel_start_sec": rpeak_start / sampling_rate,
            "rpeak_time_rel_end_sec": rpeak_end / sampling_rate,
            "rpeak_time_unix_start": first_ecg_unix + rpeak_start / sampling_rate,
            "rpeak_time_unix_end": first_ecg_unix + rpeak_end / sampling_rate,
            "rr_time_rel_sec": rr_times_rel,
            "rr_time_unix": rr_times_unix,
            "rr_time_local": [unix_to_local_string(x) for x in rr_times_unix],
            "rr_interval_ms": rr_intervals,
            "HR_BPM": heart_rate,
            "rr_plausible_300_2000_ms": plausible,
            "hr_extreme_lt40_gt200": extreme_hr,
            "rpeak_version": label,
        }
    )


def interpolate_hr_to_common_timebase(rr_df: pd.DataFrame, target_fs: float = DEFAULT_TARGET_FS) -> pd.DataFrame:
    """Interpolate heart rate to a regular timebase, preserving Lena's logic."""
    if rr_df.empty or len(rr_df) < 2:
        return pd.DataFrame()

    df = rr_df.sort_values("rr_time_unix").copy()
    times = df["rr_time_unix"].to_numpy(dtype=float)
    hr = df["HR_BPM"].to_numpy(dtype=float)

    unique_times, unique_idx = np.unique(times, return_index=True)
    unique_hr = hr[unique_idx]
    if len(unique_times) < 2:
        return pd.DataFrame()

    time_interp = np.arange(unique_times[0], unique_times[-1], 1.0 / target_fs)
    if len(time_interp) < 2:
        return pd.DataFrame()

    hr_interp = np.interp(time_interp, unique_times, unique_hr)
    return pd.DataFrame(
        {
            "TimeUnix": time_interp,
            "TimeLocal": [unix_to_local_string(x) for x in time_interp],
            "TimeRelSec": time_interp - float(time_interp[0]),
            "HR_BPM": hr_interp,
            "target_fs_hz": float(target_fs),
        }
    )


def compute_hrv_metrics(rpeaks: np.ndarray, sampling_rate: float, participant_name: str) -> tuple[pd.DataFrame, str]:
    """Compute HRV metrics using Lena's NeuroKit2 workflow, with safe fallback."""
    if len(rpeaks) < 5:
        return pd.DataFrame(), "failed_too_few_rpeaks"

    try:
        peaks_dict = {"ECG_R_Peaks": np.asarray(rpeaks, dtype=int)}
        hrv_metrics = nk.hrv(peaks_dict, sampling_rate=sampling_rate, show=False)
        hrv_metrics.insert(0, "hrv_status", "success_neurokit")

        # Lena's fallback idea for LF/HF if NeuroKit returns zeros or NaN.
        needs_freq_fallback = False
        for col in ["HRV_LF", "HRV_HF"]:
            if col not in hrv_metrics.columns:
                needs_freq_fallback = True
            else:
                val = pd.to_numeric(hrv_metrics[col], errors="coerce").iloc[0]
                if not np.isfinite(val) or val <= 0:
                    needs_freq_fallback = True
        if needs_freq_fallback:
            rr_intervals = np.diff(rpeaks) / sampling_rate * 1000.0
            rr_times = rpeaks[:-1] / sampling_rate
            freqs, psd, lf_power, hf_power = compute_rr_psd(rr_intervals, rr_times)
            if np.isfinite(lf_power):
                hrv_metrics.loc[0, "HRV_LF"] = lf_power
            if np.isfinite(hf_power):
                hrv_metrics.loc[0, "HRV_HF"] = hf_power
            if np.isfinite(lf_power) and np.isfinite(hf_power) and hf_power > 0:
                hrv_metrics.loc[0, "HRV_LFHF"] = lf_power / hf_power
            hrv_metrics.loc[0, "hrv_status"] = "success_neurokit_with_welch_frequency_fallback"
        return hrv_metrics, str(hrv_metrics.loc[0, "hrv_status"])
    except Exception as exc:
        out = pd.DataFrame({"hrv_status": ["failed"], "hrv_error": [str(exc)]})
        return out, "failed"


def compute_rr_psd(rr_intervals_ms: np.ndarray, rr_times_sec: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    rr_intervals_ms = np.asarray(rr_intervals_ms, dtype=float)
    rr_times_sec = np.asarray(rr_times_sec, dtype=float)
    valid = np.isfinite(rr_intervals_ms) & np.isfinite(rr_times_sec)
    rr_intervals_ms = rr_intervals_ms[valid]
    rr_times_sec = rr_times_sec[valid]
    if len(rr_intervals_ms) < 8 or len(np.unique(rr_times_sec)) < 2:
        return np.array([]), np.array([]), np.nan, np.nan

    fs_rr = 4.0
    t_regular = np.arange(rr_times_sec[0], rr_times_sec[-1], 1.0 / fs_rr)
    if len(t_regular) < 8:
        return np.array([]), np.array([]), np.nan, np.nan
    rr_regular = np.interp(t_regular, rr_times_sec, rr_intervals_ms)
    rr_detrended = sp_signal.detrend(rr_regular)
    freqs, psd = sp_signal.welch(rr_detrended, fs=fs_rr, nperseg=min(256, len(rr_detrended)))
    lf_mask = (freqs >= 0.04) & (freqs < 0.15)
    hf_mask = (freqs >= 0.15) & (freqs < 0.40)
    lf_power = float(np.trapz(psd[lf_mask], freqs[lf_mask])) if np.any(lf_mask) else np.nan
    hf_power = float(np.trapz(psd[hf_mask], freqs[hf_mask])) if np.any(hf_mask) else np.nan
    return freqs, psd, lf_power, hf_power


# =============================================================================
# Plot functions adapted from Lena, saved as PNG outputs
# =============================================================================


def plot_raw_vs_clean(ecg_df: pd.DataFrame, sampling_rate: float, participant_name: str, path: Path,
                      window_start_sec: float, window_duration_sec: float) -> Optional[str]:
    start_idx = int(window_start_sec * sampling_rate)
    end_idx = start_idx + int(window_duration_sec * sampling_rate)
    if start_idx >= len(ecg_df):
        start_idx = 0
        end_idx = min(len(ecg_df), int(window_duration_sec * sampling_rate))
    end_idx = min(end_idx, len(ecg_df))
    window = ecg_df.iloc[start_idx:end_idx]
    if window.empty:
        return None
    time_axis = np.arange(len(window)) / sampling_rate

    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    axes[0].plot(time_axis, window["Sample"], linewidth=0.8, alpha=0.8)
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title(f"{participant_name} - Raw ECG")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(time_axis, window["ECG_Clean"], linewidth=0.8, alpha=0.8)
    axes[1].set_xlabel("Time (seconds)")
    axes[1].set_ylabel("Amplitude")
    axes[1].set_title(f"{participant_name} - Cleaned ECG")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Raw vs cleaned ECG", fontsize=14, fontweight="bold")
    fig.tight_layout()
    write_png(fig, path)
    return str(path)


def plot_rpeak_detection(cleaned_ecg: np.ndarray, rpeaks: np.ndarray, sampling_rate: float, participant_name: str,
                         path: Path, window_start_sec: float, window_duration_sec: float) -> Optional[str]:
    start_idx = int(window_start_sec * sampling_rate)
    end_idx = start_idx + int(window_duration_sec * sampling_rate)
    if start_idx >= len(cleaned_ecg):
        start_idx = 0
        end_idx = min(len(cleaned_ecg), int(window_duration_sec * sampling_rate))
    end_idx = min(end_idx, len(cleaned_ecg))
    if end_idx <= start_idx:
        return None

    ecg_window = cleaned_ecg[start_idx:end_idx]
    time_axis = np.arange(len(ecg_window)) / sampling_rate
    window_rpeaks = rpeaks[(rpeaks >= start_idx) & (rpeaks < end_idx)] - start_idx

    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(time_axis, ecg_window, linewidth=1, alpha=0.75, label="Cleaned ECG")
    if len(window_rpeaks) > 0:
        ax.scatter(time_axis[window_rpeaks], ecg_window[window_rpeaks], s=80, marker="x", linewidths=2, label="Corrected R-peaks")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Amplitude")
    ax.set_title(f"{participant_name} - R-peak detection ({len(window_rpeaks)} peaks in window)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    write_png(fig, path)
    return str(path)


def plot_hr_from_rpeaks(rr_df: pd.DataFrame, participant_name: str, path: Path, title_suffix: str) -> Optional[str]:
    if rr_df.empty:
        return None
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(rr_df["rr_time_rel_sec"], rr_df["HR_BPM"], linewidth=1.2, alpha=0.75)
    mean_hr = float(rr_df["HR_BPM"].mean())
    ax.axhline(mean_hr, linestyle="--", alpha=0.7, label=f"Mean: {mean_hr:.1f} BPM")
    ax.set_xlabel("Time (seconds from recording start)")
    ax.set_ylabel("Heart rate (BPM)")
    ax.set_title(f"{participant_name} - Instantaneous heart rate {title_suffix}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    write_png(fig, path)
    return str(path)


def plot_hr_before_after_correction(rr_original: pd.DataFrame, rr_corrected: pd.DataFrame, participant_name: str, path: Path) -> Optional[str]:
    if rr_original.empty or rr_corrected.empty:
        return None
    fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

    axes[0].plot(rr_original["rr_time_rel_sec"], rr_original["HR_BPM"], linewidth=1.2, alpha=0.75, label="Original")
    axes[0].axhline(float(rr_original["HR_BPM"].mean()), linestyle="--", alpha=0.6,
                    label=f"Mean: {rr_original['HR_BPM'].mean():.1f} BPM")
    axes[0].set_ylabel("Heart rate (BPM)")
    axes[0].set_title(f"{participant_name} - Before R-peak correction")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(30, 210)

    axes[1].plot(rr_corrected["rr_time_rel_sec"], rr_corrected["HR_BPM"], linewidth=1.2, alpha=0.75, label="Corrected")
    axes[1].axhline(float(rr_corrected["HR_BPM"].mean()), linestyle="--", alpha=0.6,
                    label=f"Mean: {rr_corrected['HR_BPM'].mean():.1f} BPM")
    axes[1].set_xlabel("Time (seconds from recording start)")
    axes[1].set_ylabel("Heart rate (BPM)")
    axes[1].set_title(f"{participant_name} - After R-peak correction")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(30, 210)

    fig.tight_layout()
    write_png(fig, path)
    return str(path)


def plot_poincare(rr_df: pd.DataFrame, participant_name: str, path: Path, title_suffix: str) -> Optional[str]:
    if rr_df.empty or len(rr_df) < 3:
        return None
    rr = rr_df["rr_interval_ms"].to_numpy(dtype=float)
    rr1 = rr[:-1]
    rr2 = rr[1:]
    diff_rr = np.diff(rr)
    sd1 = np.sqrt(np.nanstd(diff_rr, ddof=1) ** 2 * 0.5) if len(diff_rr) > 1 else np.nan
    sd2 = np.sqrt(2 * np.nanstd(rr, ddof=1) ** 2 - 0.5 * np.nanstd(diff_rr, ddof=1) ** 2) if len(diff_rr) > 1 else np.nan

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(rr1, rr2, alpha=0.5, s=20)
    min_rr = float(np.nanmin([np.nanmin(rr1), np.nanmin(rr2)]))
    max_rr = float(np.nanmax([np.nanmax(rr1), np.nanmax(rr2)]))
    ax.plot([min_rr, max_rr], [min_rr, max_rr], linestyle="--", alpha=0.5, linewidth=1)
    ax.set_xlabel("RR(n) - milliseconds")
    ax.set_ylabel("RR(n+1) - milliseconds")
    ax.set_title(f"{participant_name} - Poincare plot {title_suffix}\nSD1={sd1:.1f} ms, SD2={sd2:.1f} ms")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    write_png(fig, path)
    return str(path)


def plot_hrv_psd(rr_df: pd.DataFrame, participant_name: str, path: Path, title_suffix: str) -> Optional[str]:
    if rr_df.empty or len(rr_df) < 8:
        return None
    rr_intervals = rr_df["rr_interval_ms"].to_numpy(dtype=float)
    rr_times = rr_df["rr_time_rel_sec"].to_numpy(dtype=float)
    freqs, psd, lf_power, hf_power = compute_rr_psd(rr_intervals, rr_times)
    if len(freqs) == 0:
        return None

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(freqs, psd, linewidth=2, label="PSD")
    lf_mask = (freqs >= 0.04) & (freqs < 0.15)
    hf_mask = (freqs >= 0.15) & (freqs < 0.40)
    ax.fill_between(freqs[lf_mask], 0, psd[lf_mask], alpha=0.3, label="LF (0.04-0.15 Hz)")
    ax.fill_between(freqs[hf_mask], 0, psd[hf_mask], alpha=0.3, label="HF (0.15-0.4 Hz)")
    ratio = lf_power / hf_power if np.isfinite(hf_power) and hf_power > 0 else np.nan
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power spectral density (ms^2/Hz)")
    ax.set_xlim(0, 0.5)
    ax.set_title(f"{participant_name} - HRV power spectral density {title_suffix}\nLF={lf_power:.1f} ms^2, HF={hf_power:.1f} ms^2, LF/HF={ratio:.2f}")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    write_png(fig, path)
    return str(path)


def plot_summary_status(summary_df: pd.DataFrame, path: Path) -> Optional[str]:
    if summary_df.empty:
        return None
    counts = summary_df["status"].value_counts(dropna=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    counts.plot(kind="bar", ax=ax)
    ax.set_xlabel("Processing status")
    ax.set_ylabel("Number of sensor recordings")
    ax.set_title("Step 2 ECG-to-HR processing status")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    write_png(fig, path)
    return str(path)


def plot_artifact_percent_summary(summary_df: pd.DataFrame, path: Path) -> Optional[str]:
    if summary_df.empty or "artifact_percent" not in summary_df.columns:
        return None
    df = summary_df[summary_df["status"] == "success"].copy()
    if df.empty:
        return None
    df["label"] = df["recording_folder"].astype(str) + "\n" + df["sensor_id"].astype(str)
    df = df.sort_values("artifact_percent", ascending=True)
    fig_height = max(8, 0.28 * len(df))
    fig, ax = plt.subplots(figsize=(12, fig_height))
    ax.barh(df["label"], df["artifact_percent"])
    ax.set_xlabel("Corrected R-peak artifacts (% of original peaks)")
    ax.set_ylabel("Recording / sensor")
    ax.set_title("R-peak correction burden by sensor recording")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    write_png(fig, path)
    return str(path)


def plot_mean_hr_summary(summary_df: pd.DataFrame, path: Path) -> Optional[str]:
    df = summary_df[summary_df["status"] == "success"].copy()
    if df.empty:
        return None
    df["label"] = df["recording_folder"].astype(str) + "\n" + df["sensor_id"].astype(str)
    df = df.sort_values("mean_hr_bpm_corrected", ascending=True)
    fig_height = max(8, 0.28 * len(df))
    fig, ax = plt.subplots(figsize=(12, fig_height))
    ax.barh(df["label"], df["mean_hr_bpm_corrected"])
    ax.set_xlabel("Mean corrected HR (BPM)")
    ax.set_ylabel("Recording / sensor")
    ax.set_title("Mean corrected heart rate by sensor recording")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    write_png(fig, path)
    return str(path)


# =============================================================================
# One-sensor processing
# =============================================================================


def make_output_stem(recording_folder: str, sensor_id: str) -> str:
    return f"{safe_filename_piece(recording_folder)}__{safe_filename_piece(sensor_id)}"


def summarize_rr(rr_df: pd.DataFrame, prefix: str) -> dict[str, Any]:
    if rr_df.empty:
        return {
            f"n_rr_intervals_{prefix}": 0,
            f"rr_mean_ms_{prefix}": np.nan,
            f"rr_sd_ms_{prefix}": np.nan,
            f"rr_min_ms_{prefix}": np.nan,
            f"rr_max_ms_{prefix}": np.nan,
            f"rr_plausible_percent_{prefix}": np.nan,
            f"n_extreme_hr_{prefix}": np.nan,
            f"mean_hr_bpm_{prefix}": np.nan,
            f"sd_hr_bpm_{prefix}": np.nan,
            f"min_hr_bpm_{prefix}": np.nan,
            f"max_hr_bpm_{prefix}": np.nan,
        }
    return {
        f"n_rr_intervals_{prefix}": int(len(rr_df)),
        f"rr_mean_ms_{prefix}": float(rr_df["rr_interval_ms"].mean()),
        f"rr_sd_ms_{prefix}": safe_std(rr_df["rr_interval_ms"].to_numpy()),
        f"rr_min_ms_{prefix}": float(rr_df["rr_interval_ms"].min()),
        f"rr_max_ms_{prefix}": float(rr_df["rr_interval_ms"].max()),
        f"rr_plausible_percent_{prefix}": float(rr_df["rr_plausible_300_2000_ms"].mean() * 100.0),
        f"n_extreme_hr_{prefix}": int(rr_df["hr_extreme_lt40_gt200"].sum()),
        f"mean_hr_bpm_{prefix}": float(rr_df["HR_BPM"].mean()),
        f"sd_hr_bpm_{prefix}": safe_std(rr_df["HR_BPM"].to_numpy()),
        f"min_hr_bpm_{prefix}": float(rr_df["HR_BPM"].min()),
        f"max_hr_bpm_{prefix}": float(rr_df["HR_BPM"].max()),
    }


def process_one_sensor(
    row: pd.Series,
    movesense_dir: Path,
    dirs: dict[str, Path],
    target_fs: float,
    ecg_method: str,
    qc_window_start_sec: float,
    qc_window_duration_sec: float,
    save_cleaned_ecg: bool,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    t0 = time.time()
    recording_folder = str(row["recording_folder"])
    sensor_id = normalize_sensor_id(row["sensor_id"])
    role = str(row.get("role", ""))
    participant_id = row.get("participant_id", "")
    dyad_id = row.get("dyad_id", "")
    pair1 = row.get("pair1", "")
    participant_name = f"{recording_folder} / sensor {sensor_id} / role {role}"
    stem = make_output_stem(recording_folder, sensor_id)

    summary: dict[str, Any] = {
        "recording_folder": recording_folder,
        "sensor_id": sensor_id,
        "role": role,
        "participant_id": participant_id,
        "dyad_id": dyad_id,
        "pair1": pair1,
        "is_pilot_session": bool_from_any(row.get("is_pilot_session", False)),
        "status": "not_started",
        "error_message": "",
        "ecg_file": "",
        "sampling_rate_hz": np.nan,
        "sampling_rate_source": "",
        "sampling_rate_note": "",
        "header_sampling_rate_hz": np.nan,
        "timestamp_sampling_rate_hz": np.nan,
        "n_ecg_samples": np.nan,
        "first_ecg_unix": np.nan,
        "last_ecg_unix": np.nan,
        "first_ecg_local": "",
        "last_ecg_local": "",
        "ecg_duration_sec": np.nan,
        "n_rpeaks_original": np.nan,
        "n_rpeaks_corrected": np.nan,
        "artifact_percent": np.nan,
        "hr_4hz_file": "",
        "hr_4hz_original_file": "",
        "rpeak_rr_corrected_file": "",
        "rpeak_rr_original_file": "",
        "hrv_corrected_file": "",
        "cleaned_ecg_file": "",
        "processing_time_sec": np.nan,
    }
    plot_rows: list[dict[str, str]] = []

    try:
        ecg_file = find_ecg_file(movesense_dir, recording_folder, sensor_id)
        if ecg_file is None:
            raise FileNotFoundError(f"Could not find ECG file for {recording_folder} / {sensor_id}")
        summary["ecg_file"] = str(ecg_file)

        ecg_df, fs, fs_source, fs_note, header_fs, timestamp_fs = load_ecg_data(ecg_file)
        first_unix = float(ecg_df["Timestamp"].iloc[0])
        last_unix = float(ecg_df["Timestamp"].iloc[-1])
        summary.update(
            {
                "sampling_rate_hz": fs,
                "sampling_rate_source": fs_source,
                "sampling_rate_note": fs_note,
                "header_sampling_rate_hz": header_fs if header_fs is not None else np.nan,
                "timestamp_sampling_rate_hz": timestamp_fs if timestamp_fs is not None else np.nan,
                "n_ecg_samples": int(len(ecg_df)),
                "first_ecg_unix": first_unix,
                "last_ecg_unix": last_unix,
                "first_ecg_local": unix_to_local_string(first_unix),
                "last_ecg_local": unix_to_local_string(last_unix),
                "ecg_duration_sec": last_unix - first_unix,
            }
        )

        cleaned = clean_ecg_signal(ecg_df, fs, participant_name, ecg_method)
        ecg_df = ecg_df.copy()
        ecg_df["ECG_Clean"] = cleaned

        rpeaks_original, _ = detect_rpeaks(cleaned, fs, participant_name, ecg_method)
        rpeaks_corrected, correction_info = fix_rpeaks(rpeaks_original, fs, participant_name)
        summary.update(correction_info)
        summary["n_rpeaks_original"] = int(len(rpeaks_original))
        summary["n_rpeaks_corrected"] = int(len(rpeaks_corrected))
        summary["artifact_percent"] = (
            float(correction_info["total_artifacts"] / len(rpeaks_original) * 100.0) if len(rpeaks_original) > 0 else np.nan
        )

        rr_original = compute_rr_intervals_from_rpeaks(rpeaks_original, fs, first_unix, len(ecg_df), "original")
        rr_corrected = compute_rr_intervals_from_rpeaks(rpeaks_corrected, fs, first_unix, len(ecg_df), "corrected")
        if rr_corrected.empty:
            raise RuntimeError("Corrected R-peaks produced fewer than 2 valid RR intervals.")

        # Add identity columns to RR outputs.
        for rr_df in [rr_original, rr_corrected]:
            if not rr_df.empty:
                rr_df.insert(0, "recording_folder", recording_folder)
                rr_df.insert(1, "sensor_id", sensor_id)
                rr_df.insert(2, "role", role)
                rr_df.insert(3, "participant_id", participant_id)
                rr_df.insert(4, "dyad_id", dyad_id)
                rr_df.insert(5, "pair1", pair1)
                rr_df.insert(6, "sampling_rate_hz", fs)

        rr_orig_path = dirs["rr_original"] / f"{stem}__rpeaks_rr_original.csv"
        rr_corr_path = dirs["rr_corrected"] / f"{stem}__rpeaks_rr_corrected.csv"
        rr_original.to_csv(rr_orig_path, index=False)
        rr_corrected.to_csv(rr_corr_path, index=False)
        summary["rpeak_rr_original_file"] = str(rr_orig_path)
        summary["rpeak_rr_corrected_file"] = str(rr_corr_path)

        hr_original = interpolate_hr_to_common_timebase(rr_original, target_fs) if not rr_original.empty else pd.DataFrame()
        hr_corrected = interpolate_hr_to_common_timebase(rr_corrected, target_fs)
        for hr_df, version in [(hr_original, "original"), (hr_corrected, "corrected")]:
            if not hr_df.empty:
                hr_df.insert(0, "recording_folder", recording_folder)
                hr_df.insert(1, "sensor_id", sensor_id)
                hr_df.insert(2, "role", role)
                hr_df.insert(3, "participant_id", participant_id)
                hr_df.insert(4, "dyad_id", dyad_id)
                hr_df.insert(5, "pair1", pair1)
                hr_df.insert(6, "rpeak_version", version)
                hr_df.insert(7, "rpeak_correction_used_for_main_hr", version == "corrected")

        hr_orig_path = dirs["hr_original"] / f"{stem}__hr_4hz_original_uncorrected.csv"
        hr_corr_path = dirs["hr"] / f"{stem}__hr_4hz.csv"
        if not hr_original.empty:
            hr_original.to_csv(hr_orig_path, index=False)
            summary["hr_4hz_original_file"] = str(hr_orig_path)
        hr_corrected.to_csv(hr_corr_path, index=False)
        summary["hr_4hz_file"] = str(hr_corr_path)
        summary["n_hr_4hz_samples_corrected"] = int(len(hr_corrected))

        # HRV metrics from corrected peaks are the main output. Original HRV is saved only if available.
        hrv_corrected, hrv_status = compute_hrv_metrics(rpeaks_corrected, fs, participant_name)
        if not hrv_corrected.empty:
            hrv_corrected.insert(0, "recording_folder", recording_folder)
            hrv_corrected.insert(1, "sensor_id", sensor_id)
            hrv_corrected.insert(2, "role", role)
            hrv_corrected.insert(3, "participant_id", participant_id)
            hrv_corrected.insert(4, "dyad_id", dyad_id)
            hrv_corrected.insert(5, "pair1", pair1)
            hrv_corrected.insert(6, "rpeak_version", "corrected")
            hrv_path = dirs["hrv"] / f"{stem}__hrv_corrected.csv"
            hrv_corrected.to_csv(hrv_path, index=False)
            summary["hrv_corrected_file"] = str(hrv_path)
            summary["hrv_status"] = hrv_status
            # Add common HRV fields to summary when present.
            for col in ["HRV_MeanNN", "HRV_SDNN", "HRV_RMSSD", "HRV_pNN50", "HRV_LF", "HRV_HF", "HRV_LFHF", "HRV_SD1", "HRV_SD2", "HRV_SD1SD2"]:
                if col in hrv_corrected.columns:
                    summary[col] = pd.to_numeric(hrv_corrected[col], errors="coerce").iloc[0]

        summary.update(summarize_rr(rr_original, "original"))
        summary.update(summarize_rr(rr_corrected, "corrected"))

        if save_cleaned_ecg:
            cleaned_path = dirs["cleaned"] / f"{stem}__cleaned_ecg.csv"
            cleaned_df = pd.DataFrame(
                {
                    "recording_folder": recording_folder,
                    "sensor_id": sensor_id,
                    "role": role,
                    "participant_id": participant_id,
                    "dyad_id": dyad_id,
                    "Timestamp": ecg_df["Timestamp"].to_numpy(dtype=float),
                    "TimeLocal": [unix_to_local_string(x) for x in ecg_df["Timestamp"].to_numpy(dtype=float)],
                    "TimeRelSec": np.arange(len(ecg_df), dtype=float) / fs,
                    "Sample": ecg_df["Sample"].to_numpy(dtype=float),
                    "ECG_Clean": cleaned,
                }
            )
            cleaned_df.to_csv(cleaned_path, index=False)
            summary["cleaned_ecg_file"] = str(cleaned_path)

        # Plots adapted from Lena.
        plot_specs = [
            ("raw_vs_clean", plot_raw_vs_clean, dirs["plot_raw_clean"] / f"{stem}__raw_vs_clean.png"),
            ("rpeaks", plot_rpeak_detection, dirs["plot_rpeaks"] / f"{stem}__rpeaks_corrected.png"),
            ("hr_original", plot_hr_from_rpeaks, dirs["plot_hr"] / f"{stem}__hr_original.png"),
            ("hr_corrected", plot_hr_from_rpeaks, dirs["plot_hr"] / f"{stem}__hr_corrected.png"),
            ("hr_before_after_correction", plot_hr_before_after_correction, dirs["plot_before_after"] / f"{stem}__hr_before_after_correction.png"),
            ("poincare_corrected", plot_poincare, dirs["plot_poincare"] / f"{stem}__poincare_corrected.png"),
            ("hrv_psd_corrected", plot_hrv_psd, dirs["plot_psd"] / f"{stem}__hrv_psd_corrected.png"),
        ]
        # Call each plot function with its specific signature.
        p = plot_raw_vs_clean(ecg_df, fs, participant_name, dirs["plot_raw_clean"] / f"{stem}__raw_vs_clean.png", qc_window_start_sec, qc_window_duration_sec)
        if p:
            plot_rows.append({"recording_folder": recording_folder, "sensor_id": sensor_id, "plot_type": "raw_vs_clean", "plot_file": p})
        p = plot_rpeak_detection(cleaned, rpeaks_corrected, fs, participant_name, dirs["plot_rpeaks"] / f"{stem}__rpeaks_corrected.png", qc_window_start_sec, qc_window_duration_sec)
        if p:
            plot_rows.append({"recording_folder": recording_folder, "sensor_id": sensor_id, "plot_type": "rpeaks_corrected", "plot_file": p})
        p = plot_hr_from_rpeaks(rr_original, participant_name, dirs["plot_hr"] / f"{stem}__hr_original.png", "from original R-peaks")
        if p:
            plot_rows.append({"recording_folder": recording_folder, "sensor_id": sensor_id, "plot_type": "hr_original", "plot_file": p})
        p = plot_hr_from_rpeaks(rr_corrected, participant_name, dirs["plot_hr"] / f"{stem}__hr_corrected.png", "from corrected R-peaks")
        if p:
            plot_rows.append({"recording_folder": recording_folder, "sensor_id": sensor_id, "plot_type": "hr_corrected", "plot_file": p})
        p = plot_hr_before_after_correction(rr_original, rr_corrected, participant_name, dirs["plot_before_after"] / f"{stem}__hr_before_after_correction.png")
        if p:
            plot_rows.append({"recording_folder": recording_folder, "sensor_id": sensor_id, "plot_type": "hr_before_after_correction", "plot_file": p})
        p = plot_poincare(rr_corrected, participant_name, dirs["plot_poincare"] / f"{stem}__poincare_corrected.png", "corrected")
        if p:
            plot_rows.append({"recording_folder": recording_folder, "sensor_id": sensor_id, "plot_type": "poincare_corrected", "plot_file": p})
        p = plot_hrv_psd(rr_corrected, participant_name, dirs["plot_psd"] / f"{stem}__hrv_psd_corrected.png", "corrected")
        if p:
            plot_rows.append({"recording_folder": recording_folder, "sensor_id": sensor_id, "plot_type": "hrv_psd_corrected", "plot_file": p})

        summary["status"] = "success"

    except Exception as exc:
        summary["status"] = "failed"
        summary["error_message"] = str(exc)
        print(f"  ERROR for {recording_folder} / {sensor_id}: {exc}")

    summary["processing_time_sec"] = time.time() - t0
    return summary, plot_rows


# =============================================================================
# Optional diagnostic dyad preview, not final analysis
# =============================================================================


def make_dyad_session_previews(analysis_units: pd.DataFrame, summary_df: pd.DataFrame, out_dir: Path, plot_dir: Path, target_fs: float) -> pd.DataFrame:
    """Create whole-session dyad previews similar to Lena Notebook 3 sections 13-14.

    These are diagnostic only. Final FF/FE synchrony is later, after trial segmentation.
    """
    rows: list[dict[str, Any]] = []
    if analysis_units.empty or summary_df.empty:
        return pd.DataFrame()

    successful = summary_df[summary_df["status"] == "success"].copy()
    hr_file_lookup = {
        (str(r["recording_folder"]), str(r["sensor_id"])): Path(str(r["hr_4hz_file"]))
        for _, r in successful.iterrows()
    }

    session_rows = analysis_units.drop_duplicates(subset=["recording_folder", "sensor_A", "sensor_B"]).copy()
    for _, row in session_rows.iterrows():
        folder = str(row["recording_folder"])
        sensor_A = normalize_sensor_id(row.get("sensor_A", ""))
        sensor_B = normalize_sensor_id(row.get("sensor_B", ""))
        dyad_id = str(row.get("dyad_id", ""))
        if (folder, sensor_A) not in hr_file_lookup or (folder, sensor_B) not in hr_file_lookup:
            rows.append({"recording_folder": folder, "dyad_id": dyad_id, "sensor_A": sensor_A, "sensor_B": sensor_B, "status": "missing_hr_file"})
            continue
        try:
            hr_A = pd.read_csv(hr_file_lookup[(folder, sensor_A)])
            hr_B = pd.read_csv(hr_file_lookup[(folder, sensor_B)])
            t0 = max(float(hr_A["TimeUnix"].min()), float(hr_B["TimeUnix"].min()))
            t1 = min(float(hr_A["TimeUnix"].max()), float(hr_B["TimeUnix"].max()))
            if t1 <= t0:
                rows.append({"recording_folder": folder, "dyad_id": dyad_id, "sensor_A": sensor_A, "sensor_B": sensor_B, "status": "no_overlap"})
                continue
            grid = np.arange(t0, t1, 1.0 / target_fs)
            if len(grid) < 20:
                rows.append({"recording_folder": folder, "dyad_id": dyad_id, "sensor_A": sensor_A, "sensor_B": sensor_B, "status": "too_few_overlap_samples"})
                continue
            hA = np.interp(grid, hr_A["TimeUnix"].to_numpy(float), hr_A["HR_BPM"].to_numpy(float))
            hB = np.interp(grid, hr_B["TimeUnix"].to_numpy(float), hr_B["HR_BPM"].to_numpy(float))
            r, p = stats.pearsonr(hA, hB) if len(hA) > 2 else (np.nan, np.nan)

            # Windowed correlation, preserving Lena's logic.
            window_sec = 30.0
            overlap = 0.5
            window_samples = int(window_sec * target_fs)
            step_samples = int(window_samples * (1.0 - overlap))
            corr = []
            corr_t = []
            for start in range(0, len(hA) - window_samples, step_samples):
                end = start + window_samples
                rr, _ = stats.pearsonr(hA[start:end], hB[start:end])
                corr.append(rr)
                corr_t.append((grid[start + window_samples // 2] - grid[0]))
            corr = np.asarray(corr, dtype=float)
            corr_t = np.asarray(corr_t, dtype=float)

            stem = make_output_stem(folder, f"{sensor_A}_{sensor_B}")
            fig, axes = plt.subplots(2, 1, figsize=(15, 9), sharex=False)
            rel_t = grid - grid[0]
            axes[0].plot(rel_t, hA, linewidth=1.0, alpha=0.8, label=f"A sensor {sensor_A}")
            axes[0].plot(rel_t, hB, linewidth=1.0, alpha=0.8, label=f"B sensor {sensor_B}")
            axes[0].set_xlabel("Time from common overlap start (seconds)")
            axes[0].set_ylabel("Heart rate (BPM)")
            axes[0].set_title(f"Diagnostic whole-session HR preview, not final analysis\n{folder}, {dyad_id}, r={r:.3f}, p={p:.3g}")
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            if len(corr) > 0:
                axes[1].plot(corr_t, corr, linewidth=1.5, label="30s windowed Pearson r")
                axes[1].axhline(0, linewidth=0.8, alpha=0.5)
                axes[1].axhline(float(np.nanmean(corr)), linestyle="--", alpha=0.7, label=f"Mean: {np.nanmean(corr):.3f}")
            axes[1].set_xlabel("Time from common overlap start (seconds)")
            axes[1].set_ylabel("Correlation")
            axes[1].set_ylim(-1, 1)
            axes[1].set_title("Diagnostic windowed HR correlation, not final FF/FE result")
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            fig.tight_layout()
            plot_path = plot_dir / f"{stem}__diagnostic_session_hr_correlation.png"
            write_png(fig, plot_path)

            rows.append(
                {
                    "recording_folder": folder,
                    "dyad_id": dyad_id,
                    "sensor_A": sensor_A,
                    "sensor_B": sensor_B,
                    "status": "success_diagnostic_only",
                    "n_overlap_samples": int(len(grid)),
                    "overlap_duration_sec": float(grid[-1] - grid[0]) if len(grid) > 1 else 0.0,
                    "whole_session_hr_correlation": float(r),
                    "whole_session_hr_correlation_p": float(p),
                    "n_windowed_correlations": int(len(corr)),
                    "mean_windowed_correlation": float(np.nanmean(corr)) if len(corr) > 0 else np.nan,
                    "plot_file": str(plot_path),
                    "note": "Diagnostic whole-session preview only. Do not use as final FF/FE synchrony result.",
                }
            )
        except Exception as exc:
            rows.append({"recording_folder": folder, "dyad_id": dyad_id, "sensor_A": sensor_A, "sensor_B": sensor_B, "status": "failed", "error_message": str(exc)})

    preview = pd.DataFrame(rows)
    preview_path = out_dir / "qc" / "02_dyad_session_preview_not_for_final_analysis.csv"
    preview.to_csv(preview_path, index=False)
    return preview


# =============================================================================
# Output writing
# =============================================================================


def make_output_dirs(out_dir: Path, overwrite: bool) -> dict[str, Path]:
    if out_dir.exists() and overwrite:
        shutil.rmtree(out_dir)
    dirs = {
        "root": out_dir,
        "hr": out_dir / "hr_interpolated_4hz",
        "hr_original": out_dir / "hr_interpolated_4hz_original",
        "rr_corrected": out_dir / "rpeak_rr_corrected",
        "rr_original": out_dir / "rpeak_rr_original",
        "hrv": out_dir / "hrv_metrics",
        "cleaned": out_dir / "cleaned_ecg_optional",
        "qc": out_dir / "qc",
        "plots": out_dir / "plots",
        "plot_raw_clean": out_dir / "plots" / "raw_vs_clean",
        "plot_rpeaks": out_dir / "plots" / "rpeaks",
        "plot_hr": out_dir / "plots" / "hr_time_series",
        "plot_before_after": out_dir / "plots" / "hr_before_after_correction",
        "plot_poincare": out_dir / "plots" / "poincare",
        "plot_psd": out_dir / "plots" / "hrv_psd",
        "plot_qc": out_dir / "plots" / "qc_summary",
        "plot_dyad_preview": out_dir / "plots" / "dyad_session_preview_not_for_final_analysis",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def write_summary_outputs(summary_df: pd.DataFrame, plot_manifest: pd.DataFrame, dirs: dict[str, Path], args: argparse.Namespace) -> None:
    qc_dir = dirs["qc"]
    csv_path = qc_dir / "02_ecg_to_hr_processing_summary.csv"
    xlsx_path = qc_dir / "02_ecg_to_hr_processing_summary.xlsx"
    plot_manifest_path = qc_dir / "02_plot_manifest.csv"
    txt_path = qc_dir / "02_run_summary.txt"

    summary_df.to_csv(csv_path, index=False)
    plot_manifest.to_csv(plot_manifest_path, index=False)

    try:
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            summary_df.to_excel(writer, sheet_name="processing_summary", index=False)
            plot_manifest.to_excel(writer, sheet_name="plot_manifest", index=False)
            wb = writer.book
            for sheet_name in wb.sheetnames:
                ws = wb[sheet_name]
                ws.freeze_panes = "A2"
                ws.auto_filter.ref = ws.dimensions
                for col_cells in ws.columns:
                    letter = col_cells[0].column_letter
                    max_len = 0
                    for cell in col_cells[:200]:
                        if cell.value is not None:
                            max_len = max(max_len, len(str(cell.value)))
                    ws.column_dimensions[letter].width = min(max(max_len + 2, 10), 70)
    except Exception as exc:
        print(f"Warning: could not write Excel summary: {exc}", file=sys.stderr)

    n_total = len(summary_df)
    n_success = int((summary_df["status"] == "success").sum()) if not summary_df.empty else 0
    n_failed = int((summary_df["status"] == "failed").sum()) if not summary_df.empty else 0
    n_plots = len(plot_manifest)
    status_counts = summary_df["status"].value_counts(dropna=False).to_dict() if not summary_df.empty else {}

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("TableTask Step 2 ECG-to-HR processing summary\n")
        f.write("=============================================\n\n")
        f.write(f"Root: {Path(args.root).expanduser().resolve()}\n")
        f.write(f"Analysis units: {args.analysis_units}\n")
        f.write(f"Movesense folder: {args.movesense_folder}\n")
        f.write(f"Output folder: {args.out_dir}\n")
        f.write(f"ECG method: {args.ecg_method}\n")
        f.write(f"Target HR sampling rate: {args.target_fs} Hz\n")
        f.write(f"Save cleaned ECG: {args.save_cleaned_ecg}\n")
        f.write(f"Make dyad preview: {args.make_dyad_preview}\n\n")
        f.write(f"Total sensor recordings evaluated: {n_total}\n")
        f.write(f"Successful sensor recordings: {n_success}\n")
        f.write(f"Failed sensor recordings: {n_failed}\n")
        f.write(f"Saved plot files recorded in manifest: {n_plots}\n\n")
        f.write("Status counts:\n")
        for status, count in status_counts.items():
            f.write(f"  {status}: {count}\n")
        f.write("\nMain HR files for Step 3 are in: processed_ecg_hr/hr_interpolated_4hz/\n")
        f.write("These files use corrected R-peaks and preserve absolute TimeUnix.\n")
        f.write("Any dyad-session preview outputs are diagnostic only, not final FF/FE synchrony results.\n")

    # Summary plots.
    plot_summary_status(summary_df, dirs["plot_qc"] / "02_processing_status_counts.png")
    plot_artifact_percent_summary(summary_df, dirs["plot_qc"] / "02_artifact_percent_by_sensor.png")
    plot_mean_hr_summary(summary_df, dirs["plot_qc"] / "02_mean_corrected_hr_by_sensor.png")

    print(f"\nProcessing summary CSV: {csv_path}")
    print(f"Processing summary Excel: {xlsx_path}")
    print(f"Plot manifest: {plot_manifest_path}")
    print(f"Run summary: {txt_path}")


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    analysis_units_path = resolve_path(root, args.analysis_units)
    movesense_dir = resolve_path(root, args.movesense_folder)
    out_dir = resolve_path(root, args.out_dir)

    if not movesense_dir.exists():
        raise FileNotFoundError(f"Movesense folder not found: {movesense_dir}")

    dirs = make_output_dirs(out_dir, overwrite=args.overwrite)

    print("TableTask Step 2: ECG-to-HR processing")
    print(f"Root: {root}")
    print(f"Movesense folder: {movesense_dir}")
    print(f"Analysis units: {analysis_units_path}")
    print(f"Output folder: {out_dir}")
    print(f"ECG method: {args.ecg_method}")
    print(f"Target HR sampling rate: {args.target_fs} Hz")
    print("\nBoundary: this script creates individual sensor HR/HRV/QC outputs.")
    print("Final FF/FE synchrony, PLI, and NSTE come after trial segmentation.\n")

    analysis_units = read_analysis_units(analysis_units_path)
    manifest = build_sensor_processing_manifest(analysis_units)
    manifest_path = dirs["qc"] / "02_sensor_processing_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    print(f"Unique recording_folder + sensor_id recordings to process: {len(manifest)}")
    print(f"Manifest saved to: {manifest_path}\n")

    summary_rows: list[dict[str, Any]] = []
    plot_rows: list[dict[str, str]] = []
    for i, (_, row) in enumerate(manifest.iterrows(), start=1):
        print("=" * 90)
        print(f"[{i}/{len(manifest)}] {row['recording_folder']} / sensor {row['sensor_id']} / role {row['role']}")
        summary, plots = process_one_sensor(
            row=row,
            movesense_dir=movesense_dir,
            dirs=dirs,
            target_fs=args.target_fs,
            ecg_method=args.ecg_method,
            qc_window_start_sec=args.qc_window_start_sec,
            qc_window_duration_sec=args.qc_window_duration_sec,
            save_cleaned_ecg=args.save_cleaned_ecg,
        )
        summary_rows.append(summary)
        plot_rows.extend(plots)
        print(f"  Status: {summary['status']}")
        if summary["status"] == "success":
            print(f"  Original R-peaks: {summary.get('n_rpeaks_original', np.nan)}")
            print(f"  Corrected R-peaks: {summary.get('n_rpeaks_corrected', np.nan)}")
            print(f"  Corrected mean HR: {summary.get('mean_hr_bpm_corrected', np.nan):.2f} BPM")
            print(f"  Artifact percent: {summary.get('artifact_percent', np.nan):.2f}%")
        else:
            print(f"  Error: {summary.get('error_message', '')}")

    summary_df = pd.DataFrame(summary_rows)
    plot_manifest = pd.DataFrame(plot_rows)

    if args.make_dyad_preview:
        print("\nCreating diagnostic whole-session dyad previews, not final analysis...")
        preview_df = make_dyad_session_previews(analysis_units, summary_df, out_dir, dirs["plot_dyad_preview"], args.target_fs)
        print(f"Diagnostic dyad previews created: {int((preview_df.get('status', pd.Series(dtype=str)) == 'success_diagnostic_only').sum()) if not preview_df.empty else 0}")

    write_summary_outputs(summary_df, plot_manifest, dirs, args)

    n_success = int((summary_df["status"] == "success").sum()) if not summary_df.empty else 0
    n_failed = int((summary_df["status"] == "failed").sum()) if not summary_df.empty else 0
    print("\n" + "=" * 90)
    print("Step 2 ECG-to-HR processing complete")
    print(f"Successful sensor recordings: {n_success}")
    print(f"Failed sensor recordings: {n_failed}")
    print(f"Main corrected 4 Hz HR files for Step 3: {dirs['hr']}")
    print(f"Plots: {dirs['plots']}")
    print("\nBefore Step 3, review the summary file and QC plots, especially sensors with high artifact_percent or extreme HR values.")
    if n_failed > 0:
        print("Some files failed. Inspect processed_ecg_hr/qc/02_ecg_to_hr_processing_summary.csv before continuing.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("\nERROR: Step 2 ECG-to-HR processing failed.", file=sys.stderr)
        print(str(exc), file=sys.stderr)
        raise
