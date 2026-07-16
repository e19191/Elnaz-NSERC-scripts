#!/usr/bin/env python3
"""
3a_script_rr_odd_even_reliability_QC_EG.py
By: Elnaz Ghasemi, with ChatGPT support
Date: June 2026

Purpose
-------
Step 3a of Veronica/Cirkeline's TableTask Movesense ECG/HR pipeline.

This script adds the objective RR reliability index requested by Veronica after
reviewing the Step 2/Step 2b QC results.

Veronica's requirement
----------------------
"RR samples can be divided into odd and even, in order of recording, and then
we can compute a Pearson correlation between RR-odd and RR-even. This will be
an index of reliability. We can then exclude sensors/sessions based on this
index, and use it to describe the data."

Relationship to Dudarev et al. 2023 Sensors paper
-------------------------------------------------
The paper frames this as a split-half reliability idea for wearable cardiac
measurements. Physiological data are time-sensitive, so odd/even splitting is
useful because the paired measurements are close together in acquisition order.
This script uses that logic to turn the visual Poincare-plot judgment into a
numeric RR reliability index.

Correct pipeline position
-------------------------
Step 1  : Build metadata/timing/QC master file.
Step 2  : Process raw ECG into corrected R-peaks/RR/HR and full-session QC.
Step 2b : Extract first 2-minute pretrial QC.
Step 3  : Segment corrected HR into paired dyad-window files.
Step 3a : THIS SCRIPT. Compute RR odd-even reliability QC.
Step 4  : Later. Trial-level synchrony/PLI.
Step 5  : Later. NSTE.

Why Step 3a comes after Step 3
------------------------------
The final analysis unit is dyad x trial/window, not the full recording. This
script therefore computes reliability at the trial-window level before Step 4.
It also computes full-session and first-2-minute reliability for description and
comparison with earlier QC.

Main outputs
------------
processed_ecg_hr/rr_reliability_qc/
  qc/
    3a_full_session_sensor_rr_reliability.csv/.xlsx
    3a_pretrial_2min_sensor_rr_reliability.csv/.xlsx
    3a_trial_sensor_rr_reliability.csv/.xlsx
    3a_trial_dyad_rr_reliability.csv/.xlsx
    3a_sensor_id_reliability_summary.csv/.xlsx
    3a_session_reliability_summary.csv/.xlsx
    3a_rr_reliability_run_summary.txt
  plots/
    3a_full_session_sensor_reliability_hist.png
    3a_pretrial_sensor_reliability_hist.png
    3a_trial_dyad_min_reliability_hist.png

Important interpretation
------------------------
The reliability index is a QC measure, not final dyadic synchrony.
It asks: "Is this single participant sensor recording internally consistent?"
It does NOT ask: "Are the two people synchronized?"

Thresholds
----------
The script writes numeric Pearson r values. It also gives provisional review
labels using user-editable thresholds:
  reliable    : r >= 0.75
  caution     : 0.50 <= r < 0.75
  low         : r < 0.50
  unavailable : not enough RR pairs or undefined correlation

These are NOT final exclusion criteria. They are review labels to help inspect
the distribution before Veronica/Beck approve a final cutoff.

How to run
----------
cd /Users/e/Desktop/veronica_project/TableTask

python3 3a_script_rr_odd_even_reliability_QC_EG.py \
  --root . \
  --processed-dir processed_ecg_hr \
  --overwrite
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

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


# =============================================================================
# User-adjustable defaults
# =============================================================================

SCRIPT_VERSION = "v1_rr_odd_even_reliability_qc"

RR_MIN_PLAUSIBLE_MS = 300.0
RR_MAX_PLAUSIBLE_MS = 2000.0
MIN_PAIRS_DEFAULT = 10

# These are provisional review labels, not final exclusion criteria.
DEFAULT_RELIABLE_THRESHOLD = 0.75
DEFAULT_CAUTION_THRESHOLD = 0.50

OUTPUT_FOLDER_NAME = "rr_reliability_qc"


# =============================================================================
# Command-line interface
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute RR odd-even reliability QC for TableTask sensor/session/trial data."
    )
    parser.add_argument(
        "--root",
        type=str,
        default=".",
        help="Top-level TableTask folder. Default: current working directory.",
    )
    parser.add_argument(
        "--processed-dir",
        type=str,
        default="processed_ecg_hr",
        help="Processed ECG/HR folder. Default: processed_ecg_hr.",
    )
    parser.add_argument(
        "--analysis-units",
        type=str,
        default="qc_outputs/veronica_analysis_units.csv",
        help="Step 1 analysis-unit CSV/XLSX. Used as fallback metadata. Default: qc_outputs/veronica_analysis_units.csv.",
    )
    parser.add_argument(
        "--step2-summary",
        type=str,
        default="",
        help="Optional explicit Step 2 processing summary CSV. Default: processed_dir/qc/02_ecg_to_hr_processing_summary.csv.",
    )
    parser.add_argument(
        "--pretrial-dir",
        type=str,
        default="processed_ecg_hr/pretrial_2min_qc_v2",
        help="Step 2b pretrial QC folder. Default: processed_ecg_hr/pretrial_2min_qc_v2.",
    )
    parser.add_argument(
        "--trial-dir",
        type=str,
        default="processed_ecg_hr/trial_segments_4hz",
        help="Step 3 trial segmentation folder. Default: processed_ecg_hr/trial_segments_4hz.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Optional output folder. Default: processed_dir/rr_reliability_qc.",
    )
    parser.add_argument(
        "--min-pairs",
        type=int,
        default=MIN_PAIRS_DEFAULT,
        help="Minimum odd-even RR pairs needed to compute Pearson r. Default: 10.",
    )
    parser.add_argument(
        "--reliable-threshold",
        type=float,
        default=DEFAULT_RELIABLE_THRESHOLD,
        help="Provisional threshold for reliable label. Default: 0.75.",
    )
    parser.add_argument(
        "--caution-threshold",
        type=float,
        default=DEFAULT_CAUTION_THRESHOLD,
        help="Provisional lower threshold for caution label. Below this is low. Default: 0.50.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip histogram plot generation.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the existing rr_reliability_qc output folder.",
    )
    return parser.parse_args()


# =============================================================================
# Small utilities
# =============================================================================


def resolve_path(root: Path, path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()



def clean_sensor_id(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"", "nan", "none"}:
        return ""
    if re.fullmatch(r"\d+\.0", text):
        text = text[:-2]
    return text



def bool_from_any(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y", "t"}



def safe_filename_piece(value: Any) -> str:
    text = str(value).strip()
    text = re.sub(r"[^A-Za-z0-9_\-]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_") or "NA"



def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    return out



def infer_role(row: pd.Series, sensor_id: str) -> str:
    if "role" in row and not pd.isna(row["role"]):
        role = str(row["role"]).strip()
        if role:
            return role
    if clean_sensor_id(row.get("sensor_A", "")) == sensor_id:
        return "A"
    if clean_sensor_id(row.get("sensor_B", "")) == sensor_id:
        return "B"
    return ""



def infer_participant(row: pd.Series, sensor_id: str) -> Any:
    if "participant_id" in row and not pd.isna(row["participant_id"]):
        return row["participant_id"]
    if clean_sensor_id(row.get("sensor_A", "")) == sensor_id:
        return row.get("participant_A", np.nan)
    if clean_sensor_id(row.get("sensor_B", "")) == sensor_id:
        return row.get("participant_B", np.nan)
    return np.nan



def parse_rr_filename(path: Path) -> tuple[str, str]:
    """Parse <recording_folder>__<sensor_id>__*.csv."""
    parts = path.stem.split("__")
    if len(parts) < 2:
        return "", ""
    return parts[0], clean_sensor_id(parts[1])


# =============================================================================
# Input readers
# =============================================================================


def read_table(path: Path, dtype: Optional[dict[str, Any]] = None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path, dtype=dtype)
    return pd.read_csv(path, dtype=dtype)



def read_analysis_units(path: Path) -> pd.DataFrame:
    dtype = {
        "recording_folder": str,
        "dyad_id": str,
        "sensor_A": str,
        "sensor_B": str,
        "condition": str,
    }
    df = read_table(path, dtype=dtype)
    if df.empty:
        return df
    for col in ["sensor_A", "sensor_B"]:
        if col in df.columns:
            df[col] = df[col].map(clean_sensor_id)
    return df



def find_step2_summary(root: Path, processed_dir: Path, explicit: str = "") -> Path:
    candidates = []
    if explicit:
        candidates.append(resolve_path(root, explicit))
    candidates.extend(
        [
            processed_dir / "qc" / "02_ecg_to_hr_processing_summary.csv",
            root / "02_ecg_to_hr_processing_summary.csv",
        ]
    )
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Could not find 02_ecg_to_hr_processing_summary.csv. Expected it in processed_ecg_hr/qc/ or root."
    )



def read_step2_summary(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"recording_folder": str, "sensor_id": str, "role": str, "dyad_id": str})
    df = clean_column_names(df)
    if "sensor_id" in df.columns:
        df["sensor_id"] = df["sensor_id"].map(clean_sensor_id)
    return df



def read_pretrial_summary(pretrial_dir: Path) -> pd.DataFrame:
    path = pretrial_dir / "qc" / "02b_pretrial_2min_qc_summary.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, dtype={"recording_folder": str, "sensor_id": str, "role": str, "dyad_id": str})
    df = clean_column_names(df)
    if "sensor_id" in df.columns:
        df["sensor_id"] = df["sensor_id"].map(clean_sensor_id)
    return df



def read_trial_summary(trial_dir: Path) -> pd.DataFrame:
    path = trial_dir / "qc" / "03_trial_segmentation_summary.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find Step 3 summary: {path}\nRun 03_script_segment_hr_by_trial_EG.py first."
        )
    df = pd.read_csv(
        path,
        dtype={
            "recording_folder": str,
            "dyad_id": str,
            "pair1": str,
            "sensor_A": str,
            "sensor_B": str,
            "condition": str,
        },
    )
    df = clean_column_names(df)
    for col in ["sensor_A", "sensor_B"]:
        if col in df.columns:
            df[col] = df[col].map(clean_sensor_id)
    return df



def read_rr_file(path: Path) -> pd.DataFrame:
    """Read one corrected RR file and standardize key columns.

    Expected Step 2 columns are rr_time_unix, rr_interval_ms, and HR_BPM. The
    function includes fallbacks so it can tolerate minor column-name changes.
    """
    df = pd.read_csv(path, dtype={"recording_folder": str, "sensor_id": str, "role": str, "dyad_id": str})
    df = clean_column_names(df)

    time_candidates = ["rr_time_unix", "TimeUnix", "rpeak_time_unix_end", "rpeak_time_unix", "Timestamp"]
    rr_candidates = ["rr_interval_ms", "RR_ms", "RR_Interval_ms", "RR_interval_ms", "RR", "RR_ms_corrected"]
    hr_candidates = ["HR_BPM", "heart_rate_bpm", "HR", "hr_bpm"]

    time_col = next((c for c in time_candidates if c in df.columns), None)
    rr_col = next((c for c in rr_candidates if c in df.columns), None)
    hr_col = next((c for c in hr_candidates if c in df.columns), None)

    if rr_col is None:
        raise ValueError(f"RR file is missing an RR interval column: {path}")

    out = df.copy()
    out["rr_interval_ms"] = pd.to_numeric(out[rr_col], errors="coerce")
    if time_col is not None:
        out["rr_time_unix"] = pd.to_numeric(out[time_col], errors="coerce")
    else:
        out["rr_time_unix"] = np.arange(len(out), dtype=float)

    if hr_col is not None:
        out["HR_BPM"] = pd.to_numeric(out[hr_col], errors="coerce")
    else:
        out["HR_BPM"] = 60000.0 / out["rr_interval_ms"]

    out = out.dropna(subset=["rr_interval_ms", "rr_time_unix"]).copy()
    out = out.sort_values("rr_time_unix").reset_index(drop=True)

    if "sensor_id" in out.columns:
        out["sensor_id"] = out["sensor_id"].map(clean_sensor_id)

    return out


# =============================================================================
# Reliability calculation
# =============================================================================


def pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 2:
        return np.nan
    if np.nanstd(x, ddof=1) == 0 or np.nanstd(y, ddof=1) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])



def odd_even_reliability(rr_values: np.ndarray, min_pairs: int) -> dict[str, Any]:
    rr = np.asarray(rr_values, dtype=float)
    rr = rr[np.isfinite(rr)]
    n_rr = int(len(rr))
    n_pairs = n_rr // 2

    if n_pairs < min_pairs:
        return {
            "rr_odd_even_r": np.nan,
            "rr_odd_even_n_pairs": n_pairs,
            "rr_odd_even_n_rr": n_rr,
            "rr_odd_even_status": "too_few_pairs",
        }

    rr = rr[: n_pairs * 2]
    odd = rr[0::2]
    even = rr[1::2]
    r = pearson_r(odd, even)
    status = "success" if np.isfinite(r) else "undefined_correlation"
    return {
        "rr_odd_even_r": r,
        "rr_odd_even_n_pairs": int(n_pairs),
        "rr_odd_even_n_rr": int(n_rr),
        "rr_odd_even_status": status,
    }



def adjacent_poincare_r(rr_values: np.ndarray, min_pairs: int) -> dict[str, Any]:
    """Also compute classic lag-1 Poincare correlation as a secondary descriptor.

    Veronica requested odd-even reliability. This lag-1 value is included only as
    an additional description of the Poincare structure and is not the primary
    reliability index.
    """
    rr = np.asarray(rr_values, dtype=float)
    rr = rr[np.isfinite(rr)]
    if len(rr) - 1 < min_pairs:
        return {"rr_adjacent_r": np.nan, "rr_adjacent_n_pairs": max(int(len(rr) - 1), 0)}
    return {"rr_adjacent_r": pearson_r(rr[:-1], rr[1:]), "rr_adjacent_n_pairs": int(len(rr) - 1)}



def reliability_label(r: float, reliable_threshold: float, caution_threshold: float) -> str:
    if not np.isfinite(r):
        return "unavailable"
    if r >= reliable_threshold:
        return "reliable"
    if r >= caution_threshold:
        return "caution"
    return "low"



def summarize_rr_quality(rr: pd.DataFrame) -> dict[str, Any]:
    if rr.empty or "rr_interval_ms" not in rr.columns:
        return {
            "n_rr_intervals": 0,
            "rr_mean_ms": np.nan,
            "rr_sd_ms": np.nan,
            "rr_min_ms": np.nan,
            "rr_max_ms": np.nan,
            "rr_plausible_percent": np.nan,
            "hr_mean_bpm_from_rr": np.nan,
            "hr_min_bpm_from_rr": np.nan,
            "hr_max_bpm_from_rr": np.nan,
            "extreme_hr_percent_from_rr": np.nan,
        }

    rr_vals = pd.to_numeric(rr["rr_interval_ms"], errors="coerce").to_numpy(dtype=float)
    rr_vals = rr_vals[np.isfinite(rr_vals)]
    if len(rr_vals) == 0:
        return {
            "n_rr_intervals": 0,
            "rr_mean_ms": np.nan,
            "rr_sd_ms": np.nan,
            "rr_min_ms": np.nan,
            "rr_max_ms": np.nan,
            "rr_plausible_percent": np.nan,
            "hr_mean_bpm_from_rr": np.nan,
            "hr_min_bpm_from_rr": np.nan,
            "hr_max_bpm_from_rr": np.nan,
            "extreme_hr_percent_from_rr": np.nan,
        }

    plausible = (rr_vals >= RR_MIN_PLAUSIBLE_MS) & (rr_vals <= RR_MAX_PLAUSIBLE_MS)
    hr = 60000.0 / rr_vals
    extreme_hr = (hr < 40.0) | (hr > 200.0)
    return {
        "n_rr_intervals": int(len(rr_vals)),
        "rr_mean_ms": float(np.nanmean(rr_vals)),
        "rr_sd_ms": float(np.nanstd(rr_vals, ddof=1)) if len(rr_vals) > 1 else np.nan,
        "rr_min_ms": float(np.nanmin(rr_vals)),
        "rr_max_ms": float(np.nanmax(rr_vals)),
        "rr_plausible_percent": float(np.nanmean(plausible) * 100.0),
        "hr_mean_bpm_from_rr": float(np.nanmean(hr)),
        "hr_min_bpm_from_rr": float(np.nanmin(hr)),
        "hr_max_bpm_from_rr": float(np.nanmax(hr)),
        "extreme_hr_percent_from_rr": float(np.nanmean(extreme_hr) * 100.0),
    }



def compute_rr_reliability_row(
    rr: pd.DataFrame,
    min_pairs: int,
    reliable_threshold: float,
    caution_threshold: float,
) -> dict[str, Any]:
    rr_vals = pd.to_numeric(rr.get("rr_interval_ms", pd.Series(dtype=float)), errors="coerce").to_numpy(dtype=float)

    all_reliability = odd_even_reliability(rr_vals, min_pairs=min_pairs)
    lag1 = adjacent_poincare_r(rr_vals, min_pairs=min_pairs)
    q = summarize_rr_quality(rr)

    plausible_vals = rr_vals[np.isfinite(rr_vals) & (rr_vals >= RR_MIN_PLAUSIBLE_MS) & (rr_vals <= RR_MAX_PLAUSIBLE_MS)]
    plausible_reliability = odd_even_reliability(plausible_vals, min_pairs=min_pairs)

    r = all_reliability["rr_odd_even_r"]
    return {
        **q,
        **all_reliability,
        "rr_odd_even_reliability_label": reliability_label(r, reliable_threshold, caution_threshold),
        "rr_odd_even_r_plausible_only": plausible_reliability["rr_odd_even_r"],
        "rr_odd_even_n_pairs_plausible_only": plausible_reliability["rr_odd_even_n_pairs"],
        "rr_odd_even_status_plausible_only": plausible_reliability["rr_odd_even_status"],
        **lag1,
    }


# =============================================================================
# Metadata helpers
# =============================================================================


def make_metadata_lookup(df: pd.DataFrame, id_col: str = "sensor_id") -> dict[tuple[str, str], dict[str, Any]]:
    if df.empty or "recording_folder" not in df.columns or id_col not in df.columns:
        return {}
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for _, row in df.iterrows():
        folder = str(row.get("recording_folder", ""))
        sensor = clean_sensor_id(row.get(id_col, ""))
        if folder and sensor:
            lookup[(folder, sensor)] = row.to_dict()
    return lookup



def metadata_for_sensor(
    folder: str,
    sensor_id: str,
    step2_lookup: dict[tuple[str, str], dict[str, Any]],
    analysis_units: pd.DataFrame,
) -> dict[str, Any]:
    meta = step2_lookup.get((folder, sensor_id), {}).copy()
    if meta:
        return {
            "recording_folder": folder,
            "sensor_id": sensor_id,
            "role": meta.get("role", ""),
            "participant_id": meta.get("participant_id", np.nan),
            "dyad_id": meta.get("dyad_id", ""),
            "pair1": meta.get("pair1", np.nan),
            "is_pilot_session": bool_from_any(meta.get("is_pilot_session", False)),
        }

    if not analysis_units.empty:
        sub = analysis_units[analysis_units["recording_folder"].astype(str) == folder]
        if not sub.empty:
            first = sub.iloc[0]
            role = infer_role(first, sensor_id)
            participant = infer_participant(first, sensor_id)
            return {
                "recording_folder": folder,
                "sensor_id": sensor_id,
                "role": role,
                "participant_id": participant,
                "dyad_id": first.get("dyad_id", ""),
                "pair1": first.get("pair1", np.nan),
                "is_pilot_session": bool_from_any(first.get("is_pilot", False)),
            }

    return {
        "recording_folder": folder,
        "sensor_id": sensor_id,
        "role": "",
        "participant_id": np.nan,
        "dyad_id": "",
        "pair1": np.nan,
        "is_pilot_session": False,
    }


# =============================================================================
# Level-specific calculations
# =============================================================================


def compute_full_session_reliability(
    rr_dir: Path,
    step2_summary: pd.DataFrame,
    analysis_units: pd.DataFrame,
    min_pairs: int,
    reliable_threshold: float,
    caution_threshold: float,
) -> pd.DataFrame:
    step2_lookup = make_metadata_lookup(step2_summary, "sensor_id")
    files = sorted(rr_dir.glob("*__*__rpeaks_rr_corrected.csv"))
    rows: list[dict[str, Any]] = []

    for path in files:
        folder, sensor_id = parse_rr_filename(path)
        if not folder or not sensor_id:
            continue
        meta = metadata_for_sensor(folder, sensor_id, step2_lookup, analysis_units)
        try:
            rr = read_rr_file(path)
            metrics = compute_rr_reliability_row(rr, min_pairs, reliable_threshold, caution_threshold)
            status = "success"
            error = ""
        except Exception as exc:
            metrics = compute_rr_reliability_row(pd.DataFrame(), min_pairs, reliable_threshold, caution_threshold)
            status = "failed"
            error = str(exc)

        rows.append(
            {
                "level": "full_session_sensor",
                **meta,
                "rr_file": str(path),
                "status": status,
                "error_message": error,
                **metrics,
            }
        )

    return pd.DataFrame(rows)



def compute_pretrial_reliability(
    pretrial_rr_dir: Path,
    pretrial_summary: pd.DataFrame,
    min_pairs: int,
    reliable_threshold: float,
    caution_threshold: float,
) -> pd.DataFrame:
    pretrial_lookup = make_metadata_lookup(pretrial_summary, "sensor_id")
    files = sorted(pretrial_rr_dir.glob("*__*__pretrial_2min_rr_corrected.csv"))
    rows: list[dict[str, Any]] = []

    for path in files:
        folder, sensor_id = parse_rr_filename(path)
        if not folder or not sensor_id:
            continue
        meta = pretrial_lookup.get((folder, sensor_id), {})
        try:
            rr = read_rr_file(path)
            metrics = compute_rr_reliability_row(rr, min_pairs, reliable_threshold, caution_threshold)
            status = "success"
            error = ""
        except Exception as exc:
            metrics = compute_rr_reliability_row(pd.DataFrame(), min_pairs, reliable_threshold, caution_threshold)
            status = "failed"
            error = str(exc)

        rows.append(
            {
                "level": "pretrial_2min_sensor",
                "recording_folder": folder,
                "sensor_id": sensor_id,
                "role": meta.get("role", ""),
                "participant_id": meta.get("participant_id", np.nan),
                "dyad_id": meta.get("dyad_id", ""),
                "pair1": meta.get("pair1", np.nan),
                "is_pilot_session": bool_from_any(meta.get("is_pilot_session", False)),
                "pretrial_qc_flag": meta.get("pretrial_qc_flag", ""),
                "pretrial_qc_reasons": meta.get("pretrial_qc_reasons", ""),
                "rr_file": str(path),
                "status": status,
                "error_message": error,
                **metrics,
            }
        )

    return pd.DataFrame(rows)



def build_rr_file_lookup(rr_dir: Path) -> dict[tuple[str, str], Path]:
    lookup: dict[tuple[str, str], Path] = {}
    for path in sorted(rr_dir.glob("*__*__rpeaks_rr_corrected.csv")):
        folder, sensor = parse_rr_filename(path)
        if folder and sensor:
            lookup[(folder, sensor)] = path
    return lookup



def load_rr_cache(rr_file_lookup: dict[tuple[str, str], Path]) -> dict[tuple[str, str], pd.DataFrame]:
    cache: dict[tuple[str, str], pd.DataFrame] = {}
    for key, path in rr_file_lookup.items():
        cache[key] = read_rr_file(path)
    return cache



def cut_rr_to_window(rr: pd.DataFrame, start_unix: float, end_unix: float) -> pd.DataFrame:
    if rr.empty or "rr_time_unix" not in rr.columns:
        return pd.DataFrame()
    return rr[(rr["rr_time_unix"] >= start_unix) & (rr["rr_time_unix"] <= end_unix)].copy().reset_index(drop=True)



def compute_trial_reliability(
    trial_summary: pd.DataFrame,
    rr_file_lookup: dict[tuple[str, str], Path],
    rr_cache: dict[tuple[str, str], pd.DataFrame],
    min_pairs: int,
    reliable_threshold: float,
    caution_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sensor_rows: list[dict[str, Any]] = []
    dyad_rows: list[dict[str, Any]] = []

    required = ["recording_folder", "sensor_A", "sensor_B", "accel_start_unix", "accel_end_unix", "status"]
    missing = [c for c in required if c not in trial_summary.columns]
    if missing:
        raise ValueError("Step 3 summary missing required columns: " + ", ".join(missing))

    usable_trials = trial_summary[trial_summary["status"].astype(str).str.lower() == "success"].copy()
    usable_trials = usable_trials.sort_values(["recording_folder", "candidate_window"]).reset_index(drop=True)

    for _, row in usable_trials.iterrows():
        folder = str(row["recording_folder"])
        start_unix = float(row["accel_start_unix"])
        end_unix = float(row["accel_end_unix"])
        base = {
            "recording_folder": folder,
            "dyad_id": row.get("dyad_id", ""),
            "pair1": row.get("pair1", np.nan),
            "participant_A": row.get("participant_A", np.nan),
            "participant_B": row.get("participant_B", np.nan),
            "sensor_A": clean_sensor_id(row.get("sensor_A", "")),
            "sensor_B": clean_sensor_id(row.get("sensor_B", "")),
            "candidate_window": int(row.get("candidate_window", -1)),
            "trial": int(row.get("trial", -1)),
            "is_practice": bool_from_any(row.get("is_practice", False)),
            "condition": row.get("condition", ""),
            "is_pilot": bool_from_any(row.get("is_pilot", False)),
            "exclude_from_main_analysis": bool_from_any(row.get("exclude_from_main_analysis", False)),
            "exclude_reason": row.get("exclude_reason", ""),
            "trial_segment_qc_flag": row.get("trial_segment_qc_flag", ""),
            "trial_segment_qc_reasons": row.get("trial_segment_qc_reasons", ""),
            "dyad_pretrial_qc_flag": row.get("dyad_pretrial_qc_flag", ""),
            "pretrial_qc_flag_A": row.get("pretrial_qc_flag_A", ""),
            "pretrial_qc_flag_B": row.get("pretrial_qc_flag_B", ""),
            "accel_start_unix": start_unix,
            "accel_end_unix": end_unix,
            "window_duration_sec": row.get("window_duration_sec", end_unix - start_unix),
            "paired_hr_output_file": row.get("output_file", ""),
        }

        role_results: dict[str, dict[str, Any]] = {}
        for role in ["A", "B"]:
            sensor_id = clean_sensor_id(row.get(f"sensor_{role}", ""))
            key = (folder, sensor_id)
            status = "success"
            error = ""
            rr_path = rr_file_lookup.get(key)
            if rr_path is None:
                rr_seg = pd.DataFrame()
                status = "failed_missing_rr_file"
                error = f"No corrected RR file for {folder} / {sensor_id}"
            else:
                try:
                    rr_seg = cut_rr_to_window(rr_cache[key], start_unix, end_unix)
                except Exception as exc:
                    rr_seg = pd.DataFrame()
                    status = "failed_cut_window"
                    error = str(exc)

            metrics = compute_rr_reliability_row(rr_seg, min_pairs, reliable_threshold, caution_threshold)
            role_row = {
                "level": "trial_window_sensor",
                **base,
                "role": role,
                "sensor_id": sensor_id,
                "participant_id": row.get(f"participant_{role}", np.nan),
                "pretrial_qc_flag_sensor": row.get(f"pretrial_qc_flag_{role}", ""),
                "rr_file": str(rr_path) if rr_path else "",
                "rr_window_status": status,
                "error_message": error,
                **metrics,
            }
            sensor_rows.append(role_row)
            role_results[role] = role_row

        r_A = role_results["A"].get("rr_odd_even_r", np.nan)
        r_B = role_results["B"].get("rr_odd_even_r", np.nan)
        label_A = role_results["A"].get("rr_odd_even_reliability_label", "unavailable")
        label_B = role_results["B"].get("rr_odd_even_reliability_label", "unavailable")
        finite_rs = [x for x in [r_A, r_B] if np.isfinite(x)]
        min_r = min(finite_rs) if finite_rs else np.nan
        mean_r = float(np.mean(finite_rs)) if finite_rs else np.nan

        if "low" in {label_A, label_B}:
            dyad_label = "low"
        elif "unavailable" in {label_A, label_B}:
            dyad_label = "unavailable"
        elif "caution" in {label_A, label_B}:
            dyad_label = "caution"
        else:
            dyad_label = "reliable"

        dyad_rows.append(
            {
                "level": "trial_window_dyad",
                **base,
                "rr_reliability_A": r_A,
                "rr_reliability_B": r_B,
                "rr_reliability_label_A": label_A,
                "rr_reliability_label_B": label_B,
                "dyad_rr_reliability_min": min_r,
                "dyad_rr_reliability_mean": mean_r,
                "dyad_rr_reliability_label": dyad_label,
                "n_rr_intervals_A": role_results["A"].get("n_rr_intervals", 0),
                "n_rr_intervals_B": role_results["B"].get("n_rr_intervals", 0),
                "n_pairs_A": role_results["A"].get("rr_odd_even_n_pairs", 0),
                "n_pairs_B": role_results["B"].get("rr_odd_even_n_pairs", 0),
            }
        )

    return pd.DataFrame(sensor_rows), pd.DataFrame(dyad_rows)


# =============================================================================
# Summaries, outputs, plots
# =============================================================================


def make_output_dirs(output_dir: Path, overwrite: bool) -> dict[str, Path]:
    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)
    dirs = {
        "root": output_dir,
        "qc": output_dir / "qc",
        "plots": output_dir / "plots",
    }
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    return dirs



def write_table(df: pd.DataFrame, csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    xlsx_path = csv_path.with_suffix(".xlsx")
    try:
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name=csv_path.stem[:31], index=False)
            ws = writer.book[csv_path.stem[:31]]
            ws.freeze_panes = "A2"
            ws.auto_filter.ref = ws.dimensions
            for col in ws.columns:
                col_letter = col[0].column_letter
                max_len = 0
                for cell in col[:300]:
                    if cell.value is not None:
                        max_len = max(max_len, len(str(cell.value)))
                ws.column_dimensions[col_letter].width = min(max(max_len + 2, 10), 60)
    except Exception as exc:
        print(f"Warning: could not write Excel file {xlsx_path.name}: {exc}", file=sys.stderr)



def make_sensor_id_summary(full_df: pd.DataFrame, pretrial_df: pd.DataFrame, trial_sensor_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    all_sensors = sorted(
        set(full_df.get("sensor_id", pd.Series(dtype=str)).dropna().astype(str))
        | set(pretrial_df.get("sensor_id", pd.Series(dtype=str)).dropna().astype(str))
        | set(trial_sensor_df.get("sensor_id", pd.Series(dtype=str)).dropna().astype(str))
    )

    def counts_by_label(df: pd.DataFrame, sensor_id: str, prefix: str) -> dict[str, Any]:
        out = {}
        if df.empty or "sensor_id" not in df.columns or "rr_odd_even_reliability_label" not in df.columns:
            return {
                f"{prefix}_n": 0,
                f"{prefix}_reliable": 0,
                f"{prefix}_caution": 0,
                f"{prefix}_low": 0,
                f"{prefix}_unavailable": 0,
                f"{prefix}_mean_r": np.nan,
                f"{prefix}_min_r": np.nan,
            }
        sub = df[df["sensor_id"].astype(str) == sensor_id]
        labels = sub["rr_odd_even_reliability_label"].value_counts().to_dict()
        r = pd.to_numeric(sub.get("rr_odd_even_r", pd.Series(dtype=float)), errors="coerce")
        out[f"{prefix}_n"] = int(len(sub))
        out[f"{prefix}_reliable"] = int(labels.get("reliable", 0))
        out[f"{prefix}_caution"] = int(labels.get("caution", 0))
        out[f"{prefix}_low"] = int(labels.get("low", 0))
        out[f"{prefix}_unavailable"] = int(labels.get("unavailable", 0))
        out[f"{prefix}_mean_r"] = float(r.mean()) if r.notna().any() else np.nan
        out[f"{prefix}_min_r"] = float(r.min()) if r.notna().any() else np.nan
        return out

    for sensor_id in all_sensors:
        row = {"sensor_id": sensor_id}
        row.update(counts_by_label(full_df, sensor_id, "full_session"))
        row.update(counts_by_label(pretrial_df, sensor_id, "pretrial"))
        row.update(counts_by_label(trial_sensor_df, sensor_id, "trial_window"))
        rows.append(row)
    return pd.DataFrame(rows)



def make_session_summary(full_df: pd.DataFrame, pretrial_df: pd.DataFrame, trial_dyad_df: pd.DataFrame) -> pd.DataFrame:
    folders = sorted(
        set(full_df.get("recording_folder", pd.Series(dtype=str)).dropna().astype(str))
        | set(pretrial_df.get("recording_folder", pd.Series(dtype=str)).dropna().astype(str))
        | set(trial_dyad_df.get("recording_folder", pd.Series(dtype=str)).dropna().astype(str))
    )
    rows: list[dict[str, Any]] = []

    def session_stats(df: pd.DataFrame, folder: str, prefix: str, dyad: bool = False) -> dict[str, Any]:
        if df.empty or "recording_folder" not in df.columns:
            return {f"{prefix}_n": 0}
        sub = df[df["recording_folder"].astype(str) == folder]
        if sub.empty:
            return {f"{prefix}_n": 0}
        rcol = "dyad_rr_reliability_min" if dyad and "dyad_rr_reliability_min" in sub.columns else "rr_odd_even_r"
        lcol = "dyad_rr_reliability_label" if dyad and "dyad_rr_reliability_label" in sub.columns else "rr_odd_even_reliability_label"
        r = pd.to_numeric(sub.get(rcol, pd.Series(dtype=float)), errors="coerce")
        labels = sub.get(lcol, pd.Series(dtype=str)).value_counts().to_dict()
        return {
            f"{prefix}_n": int(len(sub)),
            f"{prefix}_reliable": int(labels.get("reliable", 0)),
            f"{prefix}_caution": int(labels.get("caution", 0)),
            f"{prefix}_low": int(labels.get("low", 0)),
            f"{prefix}_unavailable": int(labels.get("unavailable", 0)),
            f"{prefix}_mean_r": float(r.mean()) if r.notna().any() else np.nan,
            f"{prefix}_min_r": float(r.min()) if r.notna().any() else np.nan,
        }

    for folder in folders:
        dyad_id = ""
        for df in [full_df, pretrial_df, trial_dyad_df]:
            if not df.empty and "recording_folder" in df.columns and "dyad_id" in df.columns:
                sub = df[df["recording_folder"].astype(str) == folder]
                if not sub.empty:
                    dyad_id = str(sub.iloc[0].get("dyad_id", ""))
                    if dyad_id:
                        break
        row = {"recording_folder": folder, "dyad_id": dyad_id}
        row.update(session_stats(full_df, folder, "full_session_sensor"))
        row.update(session_stats(pretrial_df, folder, "pretrial_sensor"))
        row.update(session_stats(trial_dyad_df, folder, "trial_dyad", dyad=True))
        rows.append(row)
    return pd.DataFrame(rows)



def plot_hist(values: pd.Series, title: str, path: Path) -> Optional[str]:
    if plt is None:
        return None
    vals = pd.to_numeric(values, errors="coerce").dropna()
    if vals.empty:
        return None
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(vals, bins=np.linspace(-1, 1, 21), edgecolor="black", alpha=0.8)
    ax.axvline(0.50, linestyle="--", linewidth=1.0)
    ax.axvline(0.75, linestyle="--", linewidth=1.0)
    ax.set_xlim(-1, 1)
    ax.set_xlabel("RR odd-even Pearson r")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return str(path)



def write_run_summary(
    path: Path,
    args: argparse.Namespace,
    full_df: pd.DataFrame,
    pretrial_df: pd.DataFrame,
    trial_sensor_df: pd.DataFrame,
    trial_dyad_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    def counts(df: pd.DataFrame, col: str) -> dict[str, int]:
        if df.empty or col not in df.columns:
            return {}
        return {str(k): int(v) for k, v in df[col].value_counts(dropna=False).to_dict().items()}

    with open(path, "w", encoding="utf-8") as f:
        f.write("TableTask Step 3a RR odd-even reliability QC summary\n")
        f.write("==================================================\n\n")
        f.write(f"Script version: {SCRIPT_VERSION}\n")
        f.write(f"Root: {Path(args.root).expanduser().resolve()}\n")
        f.write(f"Processed dir: {args.processed_dir}\n")
        f.write(f"Output dir: {output_dir}\n")
        f.write(f"Minimum odd-even pairs: {args.min_pairs}\n")
        f.write(f"Provisional reliable threshold: r >= {args.reliable_threshold}\n")
        f.write(f"Provisional caution threshold: {args.caution_threshold} <= r < {args.reliable_threshold}\n")
        f.write("Final exclusion thresholds have not been fixed; review with Veronica/Beck before excluding data.\n\n")

        f.write("Rows computed:\n")
        f.write(f"  Full-session sensor rows: {len(full_df)}\n")
        f.write(f"  First-2-minute pretrial sensor rows: {len(pretrial_df)}\n")
        f.write(f"  Trial-window sensor rows: {len(trial_sensor_df)}\n")
        f.write(f"  Trial-window dyad rows: {len(trial_dyad_df)}\n\n")

        f.write("Full-session sensor reliability labels:\n")
        for k, v in counts(full_df, "rr_odd_even_reliability_label").items():
            f.write(f"  {k}: {v}\n")
        f.write("\nPretrial sensor reliability labels:\n")
        for k, v in counts(pretrial_df, "rr_odd_even_reliability_label").items():
            f.write(f"  {k}: {v}\n")
        f.write("\nTrial-window sensor reliability labels:\n")
        for k, v in counts(trial_sensor_df, "rr_odd_even_reliability_label").items():
            f.write(f"  {k}: {v}\n")
        f.write("\nTrial-window dyad reliability labels, based on weaker sensor/min reliability:\n")
        for k, v in counts(trial_dyad_df, "dyad_rr_reliability_label").items():
            f.write(f"  {k}: {v}\n")

        if not trial_dyad_df.empty and "exclude_from_main_analysis" in trial_dyad_df.columns:
            main = trial_dyad_df[~trial_dyad_df["exclude_from_main_analysis"].map(bool_from_any)]
            f.write("\nMain-analysis eligible trial-window dyad reliability labels:\n")
            for k, v in counts(main, "dyad_rr_reliability_label").items():
                f.write(f"  {k}: {v}\n")

        f.write("\nInterpretation reminder:\n")
        f.write("  This index is single-sensor RR reliability, not dyadic synchrony.\n")
        f.write("  For dyad-window QC, use dyad_rr_reliability_min because dyadic analysis is only as trustworthy as the weaker sensor.\n")


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    processed_dir = resolve_path(root, args.processed_dir)
    analysis_units_path = resolve_path(root, args.analysis_units)
    pretrial_dir = resolve_path(root, args.pretrial_dir)
    trial_dir = resolve_path(root, args.trial_dir)
    output_dir = resolve_path(root, args.output_dir) if args.output_dir else processed_dir / OUTPUT_FOLDER_NAME

    if args.reliable_threshold <= args.caution_threshold:
        raise ValueError("--reliable-threshold must be greater than --caution-threshold.")

    rr_dir = processed_dir / "rpeak_rr_corrected"
    pretrial_rr_dir = pretrial_dir / "rr_segments"

    if not processed_dir.exists():
        raise FileNotFoundError(f"Processed folder not found: {processed_dir}")
    if not rr_dir.exists():
        raise FileNotFoundError(f"Corrected RR folder not found: {rr_dir}. Run Step 2 first.")
    if not pretrial_rr_dir.exists():
        raise FileNotFoundError(f"Pretrial RR segment folder not found: {pretrial_rr_dir}. Run Step 2b first.")

    dirs = make_output_dirs(output_dir, args.overwrite)

    print("TableTask Step 3a: RR odd-even reliability QC")
    print(f"Script version: {SCRIPT_VERSION}")
    print(f"Root: {root}")
    print(f"Processed dir: {processed_dir}")
    print(f"RR dir: {rr_dir}")
    print(f"Pretrial RR dir: {pretrial_rr_dir}")
    print(f"Trial dir: {trial_dir}")
    print(f"Output dir: {output_dir}")
    print("\nInterpretation: This is single-sensor RR reliability, not dyadic synchrony.")

    print("\n[1/8] Reading metadata and QC summaries...")
    step2_summary_path = find_step2_summary(root, processed_dir, args.step2_summary)
    step2_summary = read_step2_summary(step2_summary_path)
    pretrial_summary = read_pretrial_summary(pretrial_dir)
    trial_summary = read_trial_summary(trial_dir)
    analysis_units = read_analysis_units(analysis_units_path) if analysis_units_path.exists() else pd.DataFrame()

    print(f"  Step 2 summary rows: {len(step2_summary)}")
    print(f"  Step 2b summary rows: {len(pretrial_summary)}")
    print(f"  Step 3 summary rows: {len(trial_summary)}")
    print(f"  Analysis-unit rows: {len(analysis_units)}")

    print("[2/8] Computing full-session sensor reliability...")
    full_df = compute_full_session_reliability(
        rr_dir=rr_dir,
        step2_summary=step2_summary,
        analysis_units=analysis_units,
        min_pairs=args.min_pairs,
        reliable_threshold=args.reliable_threshold,
        caution_threshold=args.caution_threshold,
    )

    print("[3/8] Computing first-2-minute pretrial sensor reliability...")
    pretrial_df = compute_pretrial_reliability(
        pretrial_rr_dir=pretrial_rr_dir,
        pretrial_summary=pretrial_summary,
        min_pairs=args.min_pairs,
        reliable_threshold=args.reliable_threshold,
        caution_threshold=args.caution_threshold,
    )

    print("[4/8] Loading corrected RR files for trial-window reliability...")
    rr_lookup = build_rr_file_lookup(rr_dir)
    rr_cache = load_rr_cache(rr_lookup)
    print(f"  Corrected RR files loaded: {len(rr_cache)}")

    print("[5/8] Computing trial-window sensor and dyad reliability...")
    trial_sensor_df, trial_dyad_df = compute_trial_reliability(
        trial_summary=trial_summary,
        rr_file_lookup=rr_lookup,
        rr_cache=rr_cache,
        min_pairs=args.min_pairs,
        reliable_threshold=args.reliable_threshold,
        caution_threshold=args.caution_threshold,
    )

    print("[6/8] Computing sensor-ID and session-level summaries...")
    sensor_id_summary = make_sensor_id_summary(full_df, pretrial_df, trial_sensor_df)
    session_summary = make_session_summary(full_df, pretrial_df, trial_dyad_df)

    print("[7/8] Writing CSV/XLSX outputs...")
    write_table(full_df, dirs["qc"] / "3a_full_session_sensor_rr_reliability.csv")
    write_table(pretrial_df, dirs["qc"] / "3a_pretrial_2min_sensor_rr_reliability.csv")
    write_table(trial_sensor_df, dirs["qc"] / "3a_trial_sensor_rr_reliability.csv")
    write_table(trial_dyad_df, dirs["qc"] / "3a_trial_dyad_rr_reliability.csv")
    write_table(sensor_id_summary, dirs["qc"] / "3a_sensor_id_reliability_summary.csv")
    write_table(session_summary, dirs["qc"] / "3a_session_reliability_summary.csv")

    print("[8/8] Writing run summary and optional plots...")
    write_run_summary(
        path=dirs["qc"] / "3a_rr_reliability_run_summary.txt",
        args=args,
        full_df=full_df,
        pretrial_df=pretrial_df,
        trial_sensor_df=trial_sensor_df,
        trial_dyad_df=trial_dyad_df,
        output_dir=output_dir,
    )

    if not args.no_plots:
        plot_hist(full_df.get("rr_odd_even_r", pd.Series(dtype=float)), "Full-session sensor RR odd-even reliability", dirs["plots"] / "3a_full_session_sensor_reliability_hist.png")
        plot_hist(pretrial_df.get("rr_odd_even_r", pd.Series(dtype=float)), "First-2-minute sensor RR odd-even reliability", dirs["plots"] / "3a_pretrial_sensor_reliability_hist.png")
        plot_hist(trial_dyad_df.get("dyad_rr_reliability_min", pd.Series(dtype=float)), "Trial-window dyad minimum RR reliability", dirs["plots"] / "3a_trial_dyad_min_reliability_hist.png")

    print("\nDone.")
    print(f"Output folder: {output_dir}")
    print(f"Run summary: {dirs['qc'] / '3a_rr_reliability_run_summary.txt'}")
    print("\nKey file for Step 4 decisions:")
    print(f"  {dirs['qc'] / '3a_trial_dyad_rr_reliability.csv'}")
    print("\nRecommended next check:")
    print("  Inspect the distribution of dyad_rr_reliability_min before choosing any exclusion cutoff.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("\nERROR: Step 3a RR reliability QC failed.", file=sys.stderr)
        print(str(exc), file=sys.stderr)
        raise
