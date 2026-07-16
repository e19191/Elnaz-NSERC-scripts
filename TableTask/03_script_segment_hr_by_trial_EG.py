#!/usr/bin/env python3
"""
03_script_segment_hr_by_trial_EG_final.py
By: Elnaz Ghasemi
Date: June 2026

Purpose
-------
Step 3 of Veronica/Cirkeline TableTask Movesense analysis pipeline.

This script takes the corrected 4 Hz heart-rate outputs from Step 2 and cuts
those full-session HR time series into TableTask table-motion windows using the
Step 1 analysis-unit/timing file.

Correct pipeline position
-------------------------
Step 1  : Build metadata/timing/QC master file.
Step 2  : Process raw ECG into corrected 4 Hz HR files and QC plots.
Step 2b : Extract first 2-minute pretrial QC flags.
Step 3  : THIS SCRIPT. Segment corrected HR into paired dyad-window files.
Step 4  : Later. Correlation/PLI synchrony.
Step 5  : Later. NSTE.

Important boundary
------------------
This script does NOT calculate final synchrony, PLI, or NSTE.
It only creates aligned paired HR files and trial-level segmentation QC.

Design decisions
----------------
1. By default, segment all rows where usable_for_dyadic_ecg == True.
   Do not apply exclude_from_main_analysis == False here by default.
   Reason: this is still segmentation/QC preparation. Practice, pilot, and
   flagged rows are useful for documentation and review.

2. Carry Step 2b pretrial QC flags forward into every paired trial file and into
   the segmentation summary. Poor/caution pretrial flags are NOT used as automatic
   exclusions here.

3. Use absolute Unix time. The trial windows come from table accelerometer timing
   in veronica_analysis_units.csv/xlsx, while the HR files come from Movesense ECG.

4. Interpolate both participants to a shared 4 Hz grid inside each table-motion
   window. This is required because sensor recordings can start at slightly
   different absolute times.

Expected folder layout
----------------------
Run from the top-level TableTask folder:

TableTask/
  03_script_segment_hr_by_trial_EG_final.py
  qc_outputs/
    veronica_analysis_units.csv
  processed_ecg_hr/
    hr_interpolated_4hz/
      <recording_folder>__<sensor_id>__hr_4hz.csv
    pretrial_2min_qc_v2/
      qc/
        02b_pretrial_2min_qc_summary.csv
        02b_pretrial_2min_dyad_summary.csv

Main outputs
------------
processed_ecg_hr/trial_segments_4hz/
  paired_trial_hr/
    <recording_folder>__<dyad_id>__cw##_trial##__<condition>__paired_hr_4hz.csv
  qc/
    03_trial_segmentation_summary.csv
    03_trial_segmentation_summary.xlsx
    03_trial_segment_manifest.csv
    03_run_summary.txt

How to run
----------
cd /Users/e/Desktop/veronica_project/TableTask

python3 03_script_segment_hr_by_trial_EG_final.py \
  --root . \
  --analysis-units qc_outputs/veronica_analysis_units.csv \
  --processed-dir processed_ecg_hr \
  --pretrial-qc-dir processed_ecg_hr/pretrial_2min_qc_v2 \
  --overwrite

Optional final-analysis-only mode, not recommended yet unless intentional:
python3 03_script_segment_hr_by_trial_EG_final.py --main-analysis-only --overwrite
"""

from __future__ import annotations

import argparse
import math
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

SCRIPT_VERSION = "v1_step3_trial_segmentation_with_step2b_qc_flags"
TARGET_FS_HZ_DEFAULT = 4.0
MIN_SEGMENT_SAMPLES_DEFAULT = 10

# Trial-level QC thresholds. These are for QC labels only, not automatic exclusions.
MIN_OK_COVERAGE = 0.95
MIN_CAUTION_COVERAGE = 0.80
HR_CAUTION_LOW = 40.0
HR_CAUTION_HIGH = 200.0
HR_POOR_LOW = 30.0
HR_POOR_HIGH = 220.0
EXTREME_HR_CAUTION_FRAC = 0.005
EXTREME_HR_POOR_FRAC = 0.05


# =============================================================================
# Argument parsing
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 3: segment corrected 4 Hz HR into TableTask paired trial windows."
    )
    parser.add_argument("--root", type=str, default=".", help="Top-level TableTask folder. Default: current folder.")
    parser.add_argument(
        "--analysis-units",
        type=str,
        default="qc_outputs/veronica_analysis_units.csv",
        help="Step 1 analysis-unit file: veronica_analysis_units.csv or .xlsx.",
    )
    parser.add_argument(
        "--processed-dir",
        type=str,
        default="processed_ecg_hr",
        help="Step 2 output folder. Default: processed_ecg_hr.",
    )
    parser.add_argument(
        "--pretrial-qc-dir",
        type=str,
        default="processed_ecg_hr/pretrial_2min_qc_v2",
        help="Step 2b output folder. Default: processed_ecg_hr/pretrial_2min_qc_v2.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="processed_ecg_hr/trial_segments_4hz",
        help="Output folder for Step 3. Default: processed_ecg_hr/trial_segments_4hz.",
    )
    parser.add_argument("--target-fs", type=float, default=TARGET_FS_HZ_DEFAULT, help="Common HR grid sampling rate in Hz.")
    parser.add_argument(
        "--min-segment-samples",
        type=int,
        default=MIN_SEGMENT_SAMPLES_DEFAULT,
        help="Minimum number of aligned samples required to save a segment.",
    )
    parser.add_argument(
        "--main-analysis-only",
        action="store_true",
        help=(
            "If set, segment only rows where exclude_from_main_analysis == False. "
            "Default is False because Step 3 is segmentation/QC preparation."
        ),
    )
    parser.add_argument(
        "--include-unusable",
        action="store_true",
        help=(
            "If set, also attempt rows where usable_for_dyadic_ecg != True. "
            "Default is False."
        ),
    )
    parser.add_argument(
        "--exclude-poor-pretrial",
        action="store_true",
        help=(
            "If set, do not segment rows where Step 2b dyad flag contains poor. "
            "Default is False because poor pretrial flags should be carried forward, not automatically excluded."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing Step 3 output folder/files.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        default=True,
        help="Run strict validation checks. Default: True.",
    )
    parser.add_argument(
        "--no-strict",
        action="store_false",
        dest="strict",
        help="Disable strict validation checks.",
    )
    return parser.parse_args()


def resolve_path(root: Path, text: str) -> Path:
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()


# =============================================================================
# Utility functions
# =============================================================================


def normalize_sensor_id(value: Any) -> str:
    """Return sensor IDs as stable text labels, never as numeric variables."""
    if pd.isna(value):
        return ""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if math.isfinite(float(value)) and float(value).is_integer():
            return str(int(value))
        return str(value).strip()
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "nat"}:
        return ""
    if re.fullmatch(r"\d+\.0", text):
        return text[:-2]
    if re.fullmatch(r"[0-9.]+[eE]\+?\d+", text):
        try:
            numeric = float(text)
            if numeric.is_integer():
                return str(int(numeric))
        except Exception:
            pass
    return text


def bool_from_any(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    if isinstance(value, (int, np.integer, float, np.floating)):
        return bool(value)
    text = str(value).strip().lower()
    return text in {"true", "t", "yes", "y", "1"}


def safe_piece(value: Any) -> str:
    text = str(value).strip()
    text = re.sub(r"[^A-Za-z0-9_\-]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_") or "NA"


def numeric_or_nan(value: Any) -> float:
    try:
        out = pd.to_numeric(value, errors="coerce")
        return float(out) if pd.notna(out) else float("nan")
    except Exception:
        return float("nan")


def mean_or_nan(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if np.all(np.isnan(values)):
        return float("nan")
    return float(np.nanmean(values))


def sd_or_nan(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if np.sum(np.isfinite(values)) < 2:
        return float("nan")
    return float(np.nanstd(values, ddof=1))


def min_or_nan(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if np.all(np.isnan(values)):
        return float("nan")
    return float(np.nanmin(values))


def max_or_nan(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if np.all(np.isnan(values)):
        return float("nan")
    return float(np.nanmax(values))


def zscore_within(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    mu = np.nanmean(values) if np.any(np.isfinite(values)) else np.nan
    sd = np.nanstd(values, ddof=1) if np.sum(np.isfinite(values)) >= 2 else np.nan
    if not np.isfinite(sd) or sd == 0:
        return np.full_like(values, np.nan, dtype=float)
    return (values - mu) / sd


# =============================================================================
# Input readers and validation
# =============================================================================


def read_table(path: Path, sensor_cols: Optional[list[str]] = None) -> pd.DataFrame:
    dtype_map = {}
    if sensor_cols:
        dtype_map.update({c: str for c in sensor_cols})
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path, dtype=dtype_map)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, dtype=dtype_map)
    raise ValueError(f"Unsupported file type: {path}")


def read_analysis_units(path: Path) -> pd.DataFrame:
    df = read_table(path, sensor_cols=["recording_folder", "sensor_A", "sensor_B", "dyad_id", "condition"])
    df.columns = [str(c).strip() for c in df.columns]
    for col in ["recording_folder", "dyad_id", "sensor_A", "sensor_B", "condition"]:
        if col in df.columns:
            if col in {"sensor_A", "sensor_B"}:
                df[col] = df[col].map(normalize_sensor_id)
            else:
                df[col] = df[col].astype(str).str.strip()
    return df


def validate_analysis_units(df: pd.DataFrame) -> None:
    required = [
        "recording_folder",
        "dyad_id",
        "sensor_A",
        "sensor_B",
        "candidate_window",
        "trial",
        "is_practice",
        "condition",
        "accel_start_unix",
        "accel_end_unix",
        "usable_for_dyadic_ecg",
        "exclude_from_main_analysis",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError("Analysis-unit file is missing required columns: " + ", ".join(missing))

    if df.empty:
        raise ValueError("Analysis-unit file is empty.")

    if df["condition"].astype(str).str.lower().eq("unknown").any():
        n = int(df["condition"].astype(str).str.lower().eq("unknown").sum())
        raise ValueError(
            f"Analysis-unit file has {n} rows with condition == unknown. "
            "Use the final Step 1 output before segmenting trials."
        )

    for col in ["accel_start_unix", "accel_end_unix"]:
        bad = pd.to_numeric(df[col], errors="coerce").isna().sum()
        if bad:
            raise ValueError(f"Column {col} has {bad} missing/non-numeric values.")

    bad_duration = (pd.to_numeric(df["accel_end_unix"], errors="coerce") <= pd.to_numeric(df["accel_start_unix"], errors="coerce")).sum()
    if bad_duration:
        raise ValueError(f"Found {bad_duration} analysis-unit rows with accel_end_unix <= accel_start_unix.")

    missing_sensors = (df["sensor_A"].map(normalize_sensor_id).eq("") | df["sensor_B"].map(normalize_sensor_id).eq("")).sum()
    if missing_sensors:
        raise ValueError(f"Found {missing_sensors} rows with missing sensor_A or sensor_B.")

    # Known repaired March 3 groups. Only check if timestamp_id exists.
    if "timestamp_id" in df.columns:
        expected = {
            "2026_03_03-22_32_43": [30, 31, 32, 33, 34, 35, 36, 37, 38],
            "2026_03_03-23_45_18": [39, 40, 41, 42, 43, 44, 45, 46, 47],
        }
        for folder, expected_ids in expected.items():
            sub = df[df["recording_folder"].astype(str) == folder].copy()
            if not sub.empty:
                obs = pd.to_numeric(sub.sort_values("candidate_window")["timestamp_id"], errors="coerce").dropna().astype(int).tolist()
                if obs != expected_ids:
                    raise ValueError(
                        f"March 3 timestamp repair check failed for {folder}.\n"
                        f"Observed timestamp_id values: {obs}\n"
                        f"Expected timestamp_id values: {expected_ids}"
                    )


def read_hr_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"recording_folder": str, "sensor_id": str, "role": str, "dyad_id": str})
    df.columns = [str(c).strip() for c in df.columns]
    required = ["TimeUnix", "HR_BPM"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"HR file {path.name} is missing required columns: {missing}")
    df = df.copy()
    df["TimeUnix"] = pd.to_numeric(df["TimeUnix"], errors="coerce")
    df["HR_BPM"] = pd.to_numeric(df["HR_BPM"], errors="coerce")
    df = df.dropna(subset=["TimeUnix", "HR_BPM"]).sort_values("TimeUnix").reset_index(drop=True)
    if df.empty:
        raise ValueError(f"HR file has no valid TimeUnix/HR_BPM rows: {path}")
    return df


def find_hr_file(hr_dir: Path, recording_folder: str, sensor_id: str) -> Optional[Path]:
    exact = hr_dir / f"{recording_folder}__{sensor_id}__hr_4hz.csv"
    if exact.exists():
        return exact
    matches = sorted(hr_dir.glob(f"{recording_folder}__{sensor_id}__*hr_4hz.csv"))
    if matches:
        return matches[0]
    matches = sorted(hr_dir.glob(f"{recording_folder}__{sensor_id}__*.csv"))
    if matches:
        return matches[0]
    return None


def read_step2b_sensor_qc(pretrial_qc_dir: Path) -> pd.DataFrame:
    path = pretrial_qc_dir / "qc" / "02b_pretrial_2min_qc_summary.csv"
    if not path.exists():
        print(f"WARNING: Step 2b sensor QC file not found: {path}")
        return pd.DataFrame()
    df = pd.read_csv(path, dtype={"recording_folder": str, "sensor_id": str})
    df.columns = [str(c).strip() for c in df.columns]
    if "sensor_id" in df.columns:
        df["sensor_id"] = df["sensor_id"].map(normalize_sensor_id)
    return df


def read_step2b_dyad_qc(pretrial_qc_dir: Path) -> pd.DataFrame:
    path = pretrial_qc_dir / "qc" / "02b_pretrial_2min_dyad_summary.csv"
    if not path.exists():
        print(f"WARNING: Step 2b dyad QC file not found: {path}")
        return pd.DataFrame()
    df = pd.read_csv(path, dtype={"recording_folder": str, "dyad_id": str})
    df.columns = [str(c).strip() for c in df.columns]
    return df


def build_sensor_qc_lookup(sensor_qc: pd.DataFrame) -> dict[tuple[str, str], dict[str, Any]]:
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    if sensor_qc.empty:
        return lookup
    required = {"recording_folder", "sensor_id"}
    if not required.issubset(sensor_qc.columns):
        print("WARNING: Step 2b sensor QC file lacks recording_folder/sensor_id. Ignoring Step 2b sensor QC.")
        return lookup
    for _, row in sensor_qc.iterrows():
        key = (str(row["recording_folder"]), normalize_sensor_id(row["sensor_id"]))
        lookup[key] = row.to_dict()
    return lookup


def build_dyad_qc_lookup(dyad_qc: pd.DataFrame) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    if dyad_qc.empty:
        return lookup
    if "recording_folder" not in dyad_qc.columns:
        print("WARNING: Step 2b dyad QC file lacks recording_folder. Ignoring Step 2b dyad QC.")
        return lookup
    for _, row in dyad_qc.iterrows():
        lookup[str(row["recording_folder"])] = row.to_dict()
    return lookup


# =============================================================================
# Segmentation logic
# =============================================================================


def common_grid(start_unix: float, end_unix: float, fs_hz: float) -> np.ndarray:
    grid_start = math.ceil(start_unix * fs_hz) / fs_hz
    grid_end = math.floor(end_unix * fs_hz) / fs_hz
    if grid_end < grid_start:
        return np.array([], dtype=float)
    n = int(round((grid_end - grid_start) * fs_hz)) + 1
    return grid_start + np.arange(n, dtype=float) / fs_hz


def interpolate_to_grid(hr_df: pd.DataFrame, grid: np.ndarray) -> np.ndarray:
    t = hr_df["TimeUnix"].to_numpy(dtype=float)
    h = hr_df["HR_BPM"].to_numpy(dtype=float)
    ok = np.isfinite(t) & np.isfinite(h)
    t = t[ok]
    h = h[ok]
    if len(t) < 2:
        return np.full(len(grid), np.nan, dtype=float)
    order = np.argsort(t)
    t = t[order]
    h = h[order]
    t_unique, idx = np.unique(t, return_index=True)
    h_unique = h[idx]
    if len(t_unique) < 2:
        return np.full(len(grid), np.nan, dtype=float)
    return np.interp(grid, t_unique, h_unique, left=np.nan, right=np.nan)


def classify_trial_segment(hr_a: np.ndarray, hr_b: np.ndarray, min_segment_samples: int) -> tuple[str, str]:
    reasons: list[str] = []
    n = len(hr_a)
    if n < min_segment_samples:
        return "poor", f"too_few_samples_n_{n}"

    flags_poor = False
    flags_caution = False

    for label, hr in [("A", hr_a), ("B", hr_b)]:
        finite = np.isfinite(hr)
        coverage = float(np.mean(finite)) if len(hr) else 0.0
        if coverage < MIN_CAUTION_COVERAGE:
            flags_poor = True
            reasons.append(f"HR_{label}_coverage_below_{MIN_CAUTION_COVERAGE:.2f}")
        elif coverage < MIN_OK_COVERAGE:
            flags_caution = True
            reasons.append(f"HR_{label}_coverage_below_{MIN_OK_COVERAGE:.2f}")

        if np.any(finite):
            vals = hr[finite]
            min_hr = float(np.nanmin(vals))
            max_hr = float(np.nanmax(vals))
            extreme_frac = float(np.mean((vals < HR_CAUTION_LOW) | (vals > HR_CAUTION_HIGH)))
            severe_frac = float(np.mean((vals < HR_POOR_LOW) | (vals > HR_POOR_HIGH)))

            if min_hr < HR_POOR_LOW:
                flags_poor = True
                reasons.append(f"HR_{label}_min_below_{HR_POOR_LOW:g}")
            elif min_hr < HR_CAUTION_LOW:
                flags_caution = True
                reasons.append(f"HR_{label}_min_below_{HR_CAUTION_LOW:g}")

            if max_hr > HR_POOR_HIGH:
                flags_poor = True
                reasons.append(f"HR_{label}_max_above_{HR_POOR_HIGH:g}")
            elif max_hr > HR_CAUTION_HIGH:
                flags_caution = True
                reasons.append(f"HR_{label}_max_above_{HR_CAUTION_HIGH:g}")

            if severe_frac > 0:
                flags_poor = True
                reasons.append(f"HR_{label}_severe_extreme_values_present")
            elif extreme_frac > EXTREME_HR_POOR_FRAC:
                flags_poor = True
                reasons.append(f"HR_{label}_extreme_fraction_above_{EXTREME_HR_POOR_FRAC:g}")
            elif extreme_frac > EXTREME_HR_CAUTION_FRAC:
                flags_caution = True
                reasons.append(f"HR_{label}_extreme_fraction_above_{EXTREME_HR_CAUTION_FRAC:g}")
        else:
            flags_poor = True
            reasons.append(f"HR_{label}_all_missing")

    if flags_poor:
        return "poor", ";".join(reasons)
    if flags_caution:
        return "caution", ";".join(reasons)
    return "clean", "trial_segment_metrics_within_thresholds"


def step2b_fields_for_sensor(sensor_lookup: dict[tuple[str, str], dict[str, Any]], folder: str, sensor_id: str, prefix: str) -> dict[str, Any]:
    row = sensor_lookup.get((folder, sensor_id), {})
    return {
        f"pretrial_status_{prefix}": row.get("status", "not_found" if not sensor_lookup else "missing"),
        f"pretrial_qc_flag_{prefix}": row.get("pretrial_qc_flag", "not_found" if not sensor_lookup else "missing"),
        f"pretrial_qc_reasons_{prefix}": row.get("pretrial_qc_reasons", ""),
        f"pretrial_hr_mean_{prefix}": row.get("mean_hr_bpm", np.nan),
        f"pretrial_hr_sd_{prefix}": row.get("sd_hr_bpm", np.nan),
        f"pretrial_hr_min_{prefix}": row.get("min_hr_bpm", np.nan),
        f"pretrial_hr_max_{prefix}": row.get("max_hr_bpm", np.nan),
        f"pretrial_hr_coverage_{prefix}": row.get("hr_coverage_percent", np.nan),
        f"pretrial_rr_plausible_percent_{prefix}": row.get("rr_plausible_percent", np.nan),
    }


def dyad_pretrial_flag(dyad_lookup: dict[str, dict[str, Any]], folder: str) -> str:
    row = dyad_lookup.get(folder, {})
    if not row:
        return "not_found" if not dyad_lookup else "missing"
    return str(row.get("dyad_pretrial_qc_flag", ""))


def output_filename(row: pd.Series) -> str:
    folder = safe_piece(row["recording_folder"])
    dyad = safe_piece(row.get("dyad_id", "dyad"))
    condition = safe_piece(row.get("condition", "NA"))
    cw = int(pd.to_numeric(row["candidate_window"], errors="coerce"))
    trial = int(pd.to_numeric(row["trial"], errors="coerce"))
    return f"{folder}__{dyad}__cw{cw:02d}__trial{trial:02d}__{condition}__paired_hr_4hz.csv"


def metadata_for_row(row: pd.Series, sensor_qc_lookup: dict[tuple[str, str], dict[str, Any]], dyad_qc_lookup: dict[str, dict[str, Any]]) -> dict[str, Any]:
    folder = str(row["recording_folder"])
    sensor_a = normalize_sensor_id(row["sensor_A"])
    sensor_b = normalize_sensor_id(row["sensor_B"])

    meta = {
        "recording_folder": folder,
        "dyad_id": row.get("dyad_id", ""),
        "pair1": row.get("pair1", ""),
        "participant_A": row.get("participant_A", np.nan),
        "participant_B": row.get("participant_B", np.nan),
        "sensor_A": sensor_a,
        "sensor_B": sensor_b,
        "candidate_window": int(pd.to_numeric(row.get("candidate_window", np.nan), errors="coerce")),
        "trial": int(pd.to_numeric(row.get("trial", np.nan), errors="coerce")),
        "is_practice": bool_from_any(row.get("is_practice", False)),
        "condition": row.get("condition", ""),
        "is_pilot": bool_from_any(row.get("is_pilot", row.get("is_pilot_session", False))),
        "exclude_from_main_analysis": bool_from_any(row.get("exclude_from_main_analysis", False)),
        "exclude_reason": row.get("exclude_reason", ""),
        "usable_for_dyadic_ecg": bool_from_any(row.get("usable_for_dyadic_ecg", False)),
        "accel_start_unix": numeric_or_nan(row.get("accel_start_unix", np.nan)),
        "accel_end_unix": numeric_or_nan(row.get("accel_end_unix", np.nan)),
        "window_duration_sec": numeric_or_nan(row.get("window_duration_sec", np.nan)),
        "dyad_pretrial_qc_flag": dyad_pretrial_flag(dyad_qc_lookup, folder),
    }

    # Carry useful Step 1 QC flags if present.
    for col in [
        "aggregated_data_missing",
        "ball_drop_flag",
        "stopped_flag",
        "talking_laughing_flag",
        "wrong_way_flag",
        "sensor_issue_comment_flag",
        "session_comments",
        "table_motion_mean",
        "trial_duration_mean",
        "Coordinated_mean",
        "Joint_control_mean",
        "Incontrol_leading_mean",
    ]:
        if col in row.index:
            meta[col] = row.get(col, np.nan)

    meta.update(step2b_fields_for_sensor(sensor_qc_lookup, folder, sensor_a, "A"))
    meta.update(step2b_fields_for_sensor(sensor_qc_lookup, folder, sensor_b, "B"))
    return meta


def segment_one_row(
    row: pd.Series,
    hr_cache: dict[tuple[str, str], pd.DataFrame],
    hr_file_lookup: dict[tuple[str, str], Path],
    sensor_qc_lookup: dict[tuple[str, str], dict[str, Any]],
    dyad_qc_lookup: dict[str, dict[str, Any]],
    paired_dir: Path,
    root: Path,
    fs_hz: float,
    min_segment_samples: int,
    overwrite: bool,
) -> tuple[dict[str, Any], Optional[Path]]:
    meta = metadata_for_row(row, sensor_qc_lookup, dyad_qc_lookup)
    folder = meta["recording_folder"]
    sensor_a = meta["sensor_A"]
    sensor_b = meta["sensor_B"]
    key_a = (folder, sensor_a)
    key_b = (folder, sensor_b)

    summary = dict(meta)
    summary.update(
        {
            "status": "not_started",
            "status_detail": "",
            "output_file": "",
            "hr_file_A": str(hr_file_lookup.get(key_a, "")),
            "hr_file_B": str(hr_file_lookup.get(key_b, "")),
            "target_fs_hz": fs_hz,
            "n_expected_samples": 0,
            "n_saved_samples": 0,
            "segment_duration_sec": np.nan,
            "coverage_A": np.nan,
            "coverage_B": np.nan,
            "n_missing_A": np.nan,
            "n_missing_B": np.nan,
            "trial_segment_qc_flag": "not_run",
            "trial_segment_qc_reasons": "",
            "HR_A_mean": np.nan,
            "HR_A_sd": np.nan,
            "HR_A_min": np.nan,
            "HR_A_max": np.nan,
            "HR_B_mean": np.nan,
            "HR_B_sd": np.nan,
            "HR_B_min": np.nan,
            "HR_B_max": np.nan,
        }
    )

    if key_a not in hr_file_lookup:
        summary.update(status="failed_missing_hr_A", status_detail=f"No corrected HR file found for {folder} / {sensor_a}.")
        return summary, None
    if key_b not in hr_file_lookup:
        summary.update(status="failed_missing_hr_B", status_detail=f"No corrected HR file found for {folder} / {sensor_b}.")
        return summary, None

    start = float(meta["accel_start_unix"])
    end = float(meta["accel_end_unix"])
    grid = common_grid(start, end, fs_hz)
    if len(grid) < min_segment_samples:
        summary.update(
            status="failed_too_few_grid_samples",
            status_detail=f"Only {len(grid)} grid samples in this window.",
            n_expected_samples=int(len(grid)),
        )
        return summary, None

    hr_a = interpolate_to_grid(hr_cache[key_a], grid)
    hr_b = interpolate_to_grid(hr_cache[key_b], grid)

    n_missing_a = int(np.isnan(hr_a).sum())
    n_missing_b = int(np.isnan(hr_b).sum())
    coverage_a = float(np.mean(np.isfinite(hr_a))) if len(hr_a) else 0.0
    coverage_b = float(np.mean(np.isfinite(hr_b))) if len(hr_b) else 0.0
    trial_flag, trial_reasons = classify_trial_segment(hr_a, hr_b, min_segment_samples)

    out_name = output_filename(row)
    out_path = paired_dir / out_name
    if out_path.exists() and not overwrite:
        summary.update(
            status="skipped_existing_file",
            status_detail="Output exists. Use --overwrite to replace.",
            output_file=str(out_path.relative_to(root)) if out_path.is_relative_to(root) else str(out_path),
            n_expected_samples=int(len(grid)),
            trial_segment_qc_flag=trial_flag,
            trial_segment_qc_reasons=trial_reasons,
        )
        return summary, out_path

    segment = pd.DataFrame(
        {
            **{k: [v] * len(grid) for k, v in meta.items()},
            "TimeUnix": grid,
            "TimeFromAccelStartSec": grid - start,
            "TimeRelTrialSec": grid - grid[0],
            "HR_A_BPM": hr_a,
            "HR_B_BPM": hr_b,
            "HR_A_z_within_trial": zscore_within(hr_a),
            "HR_B_z_within_trial": zscore_within(hr_b),
            "HR_mean_dyad_BPM": np.nanmean(np.vstack([hr_a, hr_b]), axis=0),
            "HR_diff_A_minus_B_BPM": hr_a - hr_b,
            "target_fs_hz": fs_hz,
        }
    )

    paired_dir.mkdir(parents=True, exist_ok=True)
    segment.to_csv(out_path, index=False)

    summary.update(
        status="success",
        status_detail="Paired trial HR segment saved.",
        output_file=str(out_path.relative_to(root)) if out_path.is_relative_to(root) else str(out_path),
        n_expected_samples=int(len(grid)),
        n_saved_samples=int(len(segment)),
        segment_duration_sec=float(grid[-1] - grid[0]) if len(grid) > 1 else 0.0,
        coverage_A=coverage_a,
        coverage_B=coverage_b,
        n_missing_A=n_missing_a,
        n_missing_B=n_missing_b,
        trial_segment_qc_flag=trial_flag,
        trial_segment_qc_reasons=trial_reasons,
        HR_A_mean=mean_or_nan(hr_a),
        HR_A_sd=sd_or_nan(hr_a),
        HR_A_min=min_or_nan(hr_a),
        HR_A_max=max_or_nan(hr_a),
        HR_B_mean=mean_or_nan(hr_b),
        HR_B_sd=sd_or_nan(hr_b),
        HR_B_min=min_or_nan(hr_b),
        HR_B_max=max_or_nan(hr_b),
    )
    return summary, out_path


# =============================================================================
# Output writers
# =============================================================================


def write_outputs(summary: pd.DataFrame, manifest: pd.DataFrame, qc_dir: Path, args: argparse.Namespace, root: Path, selected_n: int, hr_files_found_n: int, hr_files_needed_n: int) -> None:
    qc_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = qc_dir / "03_trial_segmentation_summary.csv"
    summary_xlsx = qc_dir / "03_trial_segmentation_summary.xlsx"
    manifest_csv = qc_dir / "03_trial_segment_manifest.csv"
    run_txt = qc_dir / "03_run_summary.txt"

    summary.to_csv(summary_csv, index=False)
    manifest.to_csv(manifest_csv, index=False)

    try:
        with pd.ExcelWriter(summary_xlsx, engine="openpyxl") as writer:
            summary.to_excel(writer, sheet_name="trial_segmentation_summary", index=False)
            manifest.to_excel(writer, sheet_name="segment_manifest", index=False)
            wb = writer.book
            for sheet in wb.sheetnames:
                ws = wb[sheet]
                ws.freeze_panes = "A2"
                ws.auto_filter.ref = ws.dimensions
                for col_cells in ws.columns:
                    width = 10
                    letter = col_cells[0].column_letter
                    for cell in col_cells[:200]:
                        if cell.value is not None:
                            width = max(width, len(str(cell.value)) + 2)
                    ws.column_dimensions[letter].width = min(width, 60)
    except Exception as exc:
        print(f"WARNING: Could not write Excel summary: {exc}")

    status_counts = summary["status"].value_counts(dropna=False).to_dict() if not summary.empty else {}
    trial_qc_counts = summary["trial_segment_qc_flag"].value_counts(dropna=False).to_dict() if "trial_segment_qc_flag" in summary else {}
    pretrial_dyad_counts = summary["dyad_pretrial_qc_flag"].value_counts(dropna=False).to_dict() if "dyad_pretrial_qc_flag" in summary else {}
    condition_counts = summary["condition"].value_counts(dropna=False).to_dict() if "condition" in summary else {}
    success_n = int((summary["status"] == "success").sum()) if "status" in summary else 0
    main_success_n = int(((summary["status"] == "success") & (~summary["exclude_from_main_analysis"].map(bool_from_any))).sum()) if "exclude_from_main_analysis" in summary else 0

    with open(run_txt, "w", encoding="utf-8") as f:
        f.write("TableTask Step 3 trial HR segmentation summary\n")
        f.write("==============================================\n\n")
        f.write(f"Script version: {SCRIPT_VERSION}\n")
        f.write(f"Root: {root}\n")
        f.write(f"Analysis units: {args.analysis_units}\n")
        f.write(f"Processed dir: {args.processed_dir}\n")
        f.write(f"Pretrial QC dir: {args.pretrial_qc_dir}\n")
        f.write(f"Output dir: {args.out_dir}\n")
        f.write(f"Target fs: {args.target_fs} Hz\n")
        f.write(f"main_analysis_only: {args.main_analysis_only}\n")
        f.write(f"include_unusable: {args.include_unusable}\n")
        f.write(f"exclude_poor_pretrial: {args.exclude_poor_pretrial}\n\n")

        f.write(f"Rows selected for segmentation: {selected_n}\n")
        f.write(f"Needed HR files: {hr_files_needed_n}\n")
        f.write(f"Found HR files: {hr_files_found_n}\n")
        f.write(f"Successful paired segments: {success_n}\n")
        f.write(f"Successful main-analysis eligible segments: {main_success_n}\n\n")

        f.write("Status counts:\n")
        for k, v in status_counts.items():
            f.write(f"  {k}: {v}\n")
        f.write("\nTrial segment QC flag counts:\n")
        for k, v in trial_qc_counts.items():
            f.write(f"  {k}: {v}\n")
        f.write("\nStep 2b dyad pretrial QC flag counts among segmented rows:\n")
        for k, v in pretrial_dyad_counts.items():
            f.write(f"  {k}: {v}\n")
        f.write("\nCondition counts among segmented rows:\n")
        for k, v in condition_counts.items():
            f.write(f"  {k}: {v}\n")


# =============================================================================
# Main workflow
# =============================================================================


def main() -> None:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    analysis_units_path = resolve_path(root, args.analysis_units)
    processed_dir = resolve_path(root, args.processed_dir)
    pretrial_qc_dir = resolve_path(root, args.pretrial_qc_dir)
    out_dir = resolve_path(root, args.out_dir)
    hr_dir = processed_dir / "hr_interpolated_4hz"
    paired_dir = out_dir / "paired_trial_hr"
    qc_dir = out_dir / "qc"

    print("TableTask Step 3: trial-level paired HR segmentation")
    print(f"Script version: {SCRIPT_VERSION}")
    print(f"Root: {root}")
    print(f"Analysis units: {analysis_units_path}")
    print(f"Processed dir: {processed_dir}")
    print(f"HR dir: {hr_dir}")
    print(f"Pretrial QC dir: {pretrial_qc_dir}")
    print(f"Output dir: {out_dir}")

    if not analysis_units_path.exists():
        raise FileNotFoundError(f"Analysis-unit file not found: {analysis_units_path}")
    if not hr_dir.exists():
        raise FileNotFoundError(f"Corrected HR folder not found: {hr_dir}. Run Step 2 first.")

    if out_dir.exists() and args.overwrite:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paired_dir.mkdir(parents=True, exist_ok=True)
    qc_dir.mkdir(parents=True, exist_ok=True)

    print("\n[1/6] Reading Step 1 analysis-unit file...")
    analysis_units = read_analysis_units(analysis_units_path)
    if args.strict:
        validate_analysis_units(analysis_units)
    print(f"Analysis-unit rows: {len(analysis_units)}")

    print("[2/6] Reading Step 2b pretrial QC summaries...")
    sensor_qc = read_step2b_sensor_qc(pretrial_qc_dir)
    dyad_qc = read_step2b_dyad_qc(pretrial_qc_dir)
    sensor_qc_lookup = build_sensor_qc_lookup(sensor_qc)
    dyad_qc_lookup = build_dyad_qc_lookup(dyad_qc)
    print(f"Step 2b sensor QC rows loaded: {len(sensor_qc)}")
    print(f"Step 2b dyad QC rows loaded: {len(dyad_qc)}")

    print("[3/6] Selecting rows for segmentation...")
    df = analysis_units.copy()
    df["usable_for_dyadic_ecg_bool"] = df["usable_for_dyadic_ecg"].map(bool_from_any)
    df["exclude_from_main_analysis_bool"] = df["exclude_from_main_analysis"].map(bool_from_any)

    selected = df.copy() if args.include_unusable else df[df["usable_for_dyadic_ecg_bool"]].copy()
    if args.main_analysis_only:
        selected = selected[~selected["exclude_from_main_analysis_bool"]].copy()

    if args.exclude_poor_pretrial and dyad_qc_lookup:
        selected = selected[
            selected["recording_folder"].astype(str).map(lambda x: "poor" not in dyad_pretrial_flag(dyad_qc_lookup, x).lower())
        ].copy()

    selected = selected.sort_values(["recording_folder", "candidate_window"]).reset_index(drop=True)
    print(f"Rows selected: {len(selected)}")

    print("[4/6] Locating and loading corrected HR files...")
    needed_pairs: set[tuple[str, str]] = set()
    for _, row in selected.iterrows():
        folder = str(row["recording_folder"])
        needed_pairs.add((folder, normalize_sensor_id(row["sensor_A"])))
        needed_pairs.add((folder, normalize_sensor_id(row["sensor_B"])))

    hr_file_lookup: dict[tuple[str, str], Path] = {}
    hr_cache: dict[tuple[str, str], pd.DataFrame] = {}
    for folder, sensor_id in sorted(needed_pairs):
        path = find_hr_file(hr_dir, folder, sensor_id)
        if path is not None:
            hr_file_lookup[(folder, sensor_id)] = path
            hr_cache[(folder, sensor_id)] = read_hr_file(path)

    print(f"Needed HR files: {len(needed_pairs)}")
    print(f"Found HR files: {len(hr_file_lookup)}")
    if args.strict and len(hr_file_lookup) < len(needed_pairs):
        missing = sorted(needed_pairs - set(hr_file_lookup.keys()))
        print("WARNING: Missing HR files:")
        for folder, sensor_id in missing[:20]:
            print(f"  {folder} / {sensor_id}")
        if len(missing) > 20:
            print(f"  ... plus {len(missing) - 20} more")

    print("[5/6] Segmenting paired trial windows...")
    summary_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []

    for i, (_, row) in enumerate(selected.iterrows(), start=1):
        folder = str(row["recording_folder"])
        cw = int(pd.to_numeric(row["candidate_window"], errors="coerce"))
        trial = int(pd.to_numeric(row["trial"], errors="coerce"))
        condition = str(row.get("condition", ""))
        print(f"  [{i}/{len(selected)}] {folder} / cw={cw} / trial={trial} / {condition}")
        summary, out_path = segment_one_row(
            row=row,
            hr_cache=hr_cache,
            hr_file_lookup=hr_file_lookup,
            sensor_qc_lookup=sensor_qc_lookup,
            dyad_qc_lookup=dyad_qc_lookup,
            paired_dir=paired_dir,
            root=root,
            fs_hz=float(args.target_fs),
            min_segment_samples=int(args.min_segment_samples),
            overwrite=bool(args.overwrite),
        )
        summary_rows.append(summary)
        if out_path is not None and summary.get("status") in {"success", "skipped_existing_file"}:
            manifest_rows.append(
                {
                    "recording_folder": folder,
                    "dyad_id": summary.get("dyad_id", ""),
                    "candidate_window": cw,
                    "trial": trial,
                    "condition": condition,
                    "sensor_A": summary.get("sensor_A", ""),
                    "sensor_B": summary.get("sensor_B", ""),
                    "output_file": summary.get("output_file", str(out_path)),
                    "status": summary.get("status", ""),
                    "trial_segment_qc_flag": summary.get("trial_segment_qc_flag", ""),
                    "dyad_pretrial_qc_flag": summary.get("dyad_pretrial_qc_flag", ""),
                    "exclude_from_main_analysis": summary.get("exclude_from_main_analysis", ""),
                    "exclude_reason": summary.get("exclude_reason", ""),
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    manifest_df = pd.DataFrame(manifest_rows)

    print("[6/6] Writing Step 3 QC outputs...")
    write_outputs(
        summary=summary_df,
        manifest=manifest_df,
        qc_dir=qc_dir,
        args=args,
        root=root,
        selected_n=len(selected),
        hr_files_found_n=len(hr_file_lookup),
        hr_files_needed_n=len(needed_pairs),
    )

    status_counts = summary_df["status"].value_counts(dropna=False).to_dict() if not summary_df.empty else {}
    trial_qc_counts = summary_df["trial_segment_qc_flag"].value_counts(dropna=False).to_dict() if not summary_df.empty else {}
    success_n = int((summary_df["status"] == "success").sum()) if not summary_df.empty else 0

    print("\nStep 3 complete.")
    print(f"Paired trial HR folder: {paired_dir}")
    print(f"QC folder: {qc_dir}")
    print(f"Successful segments: {success_n} / {len(summary_df)}")
    print("\nStatus counts:")
    for k, v in status_counts.items():
        print(f"  {k}: {v}")
    print("\nTrial segment QC flag counts:")
    for k, v in trial_qc_counts.items():
        print(f"  {k}: {v}")
    print("\nUpload these for review:")
    print(f"  {qc_dir / '03_run_summary.txt'}")
    print(f"  {qc_dir / '03_trial_segmentation_summary.csv'}")
    print(f"  {qc_dir / '03_trial_segment_manifest.csv'}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("\nERROR: Step 3 trial segmentation failed.", file=sys.stderr)
        print(str(exc), file=sys.stderr)
        raise
