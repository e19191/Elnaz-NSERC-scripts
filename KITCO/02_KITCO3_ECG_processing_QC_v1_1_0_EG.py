#!/usr/bin/env python3
"""
KITCO3 Script 02: ECG processing and objective QC
Version: 1.1.0

Rate handling corrected: reciprocal median spacing is descriptive, not a
calibrated sampling rate. Default legacy_replay explicitly uses the historical
per-record filter/detector design parameters for compatibility with reviewed peaks.
header_nominal is a separate, review-required alternative, not clock calibration.
No exported sample times are rescaled. Existing results are never overwritten.

Purpose
-------
Process every ECG recording that Script 01 marked as a Script02 candidate, one
recording at a time, without loading the full study into memory.

This script deliberately does NOT:
- choose among nonidentical restart recordings;
- apply final scientific ECG exclusion thresholds;
- crop to the cooking epoch;
- interpolate HR to 4 Hz;
- assemble dyads or compute synchrony.

Those decisions belong to later pipeline stages after ECG QC has been reviewed.

Processing logic
----------------
1. Read the frozen Script 01 sensor manifest and verify Script 01 provenance.
2. Stream-parse each raw ECG CSV robustly, including truncated tail rows.
3. Report timestamp-spacing/count diagnostics and the explicit detector rate policy.
4. Detect discontinuities and process contiguous ECG segments independently.
5. Clean ECG with NeuroKit2's ``neurokit`` method using an explicit 60 Hz
   powerline setting for the Vancouver acquisition context.
6. Detect R-peaks using NeuroKit2 ``ecg_peaks(..., method='neurokit',
   correct_artifacts=False)``.
7. Apply Kubios/Lipponen-Tarvainen peak correction explicitly with
   ``signal_fixpeaks(..., method='Kubios', iterative=True)``.
8. Propagate corrected R-peaks to corrected RR intervals and instantaneous HR.
9. Add the frozen Script 01 VideoAnnotationStart/Stop interval as QC context
   when a single standard cooking interval exists. This changes plots/metadata
   only; it does not crop or exclude ECG in Script 02.
10. Keep annotation-derived cooking-window fields distinct from Script 01 file-level
    video metadata (for example, cooking_interval_duration_s vs video_duration_s).
11. Detect overlapping absolute-time segments after timestamp resets. Very small
    boundary overlaps no larger than the existing technical discontinuity threshold
    are retained as warnings; larger overlaps are blocking time-alignment errors for
    synchrony. Each segment is still processed independently for ECG morphology/QC.
    No timestamp is reconstructed.
12. Save compact beat-level data, recording-level QC, segment-level QC, issues,
    and review plots.

Raw data are read-only. No raw file is modified, renamed, moved, or deleted.

Expected project layout
-----------------------
/Volumes/One Touch/KITCO3_drive/
    scripts/
        02_KITCO3_ECG_processing_QC_v1_0_5_EG.py
    data/
        Participants 01-04/
        ...
    outputs/
        01_reconciliation/
            01_sensor_recordings.csv
            01_run_summary.json
            01_annotation_events.csv
        02_ecg_processing/            # created by this script

Typical full run
----------------
    cd "/Volumes/One Touch/KITCO3_drive"
    python3 scripts/02_KITCO3_ECG_processing_QC_v1_0_5_EG.py

Recommended first test
----------------------
    python3 scripts/02_KITCO3_ECG_processing_QC_v1_0_5_EG.py \
        --max-recordings 8 \
        --output-dir outputs/02_ecg_processing_test

Then inspect the test outputs before running all recordings.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import logging
import math
import os
import platform
import re
import shutil
import sys
import tempfile
import time
import traceback
import warnings
from collections import Counter
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

SCRIPT_VERSION = "1.1.0"
SCRIPT_NAME = "02_KITCO3_ECG_processing_QC_v1_1_0_EG.py"
EXPECTED_SCRIPT01_VERSION = "1.4.1"

SCRIPT_PATH = Path(__file__).resolve()
DEFAULT_PROJECT_ROOT = Path("/Volumes/One Touch/KITCO3_drive")
DEFAULT_RAW_ROOT = Path("data")
DEFAULT_SCRIPT01_MANIFEST = Path("outputs/01_reconciliation/01_sensor_recordings.csv")
DEFAULT_SCRIPT01_SUMMARY = Path("outputs/01_reconciliation/01_run_summary.json")
DEFAULT_SCRIPT01_ANNOTATIONS = Path("outputs/01_reconciliation/01_annotation_events.csv")
DEFAULT_OUTPUT_DIR = Path("outputs/02_ecg_processing_v1_1_0")

REQUIRED_MANIFEST_COLUMNS = {
    "attempt_uid",
    "sensor_full",
    "sensor_short",
    "ECG_path",
    "ECG_file_size_bytes",
    "ECG_schema_status",
    "ECG_time_envelope_ok",
    "ECG_inventory_ok",
    "mapped_session_id",
    "mapped_phase",
    "mapped_condition",
    "mapped_pid",
    "analysis_partner",
    "analysis_role",
    "raw_recording_expectation",
    "Script02_process_candidate",
    "PrimarySynchrony_candidate",
}

# These are technical processing defaults, not scientific exclusion criteria.
DEFAULT_POWERLINE_HZ = 60.0
DEFAULT_GAP_FACTOR = 5.0
DEFAULT_GAP_MIN_SECONDS = 0.10
DEFAULT_MIN_SEGMENT_SECONDS = 10.0
DEFAULT_PLOT_WINDOW_SECONDS = 30.0
DEFAULT_MAX_PLOT_POINTS = 12000


@dataclass
class ParsedECG:
    timestamps: np.ndarray
    samples: np.ndarray
    source_rows: np.ndarray
    header_line_number: int
    total_data_rows: int
    valid_rows: int
    invalid_timestamp_rows: int
    invalid_sample_rows: int
    malformed_rows: int
    blank_rows: int


@dataclass
class SegmentDef:
    segment_id: int
    start: int
    stop: int
    start_timestamp: float
    stop_timestamp: float
    n_samples: int
    duration_s: float
    boundary_reason_before: str


class ProcessingDependencyError(RuntimeError):
    pass


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def resolve_path(project_root: Path, value: Path) -> Path:
    value = Path(value).expanduser()
    if value.is_absolute():
        return value.resolve()
    return (project_root / value).resolve()


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    return text in {"1", "true", "t", "yes", "y"}


def safe_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        if value is None or str(value).strip() == "":
            return default
        return int(float(value))
    except (TypeError, ValueError, OverflowError):
        return default


def safe_float(value: Any, default: float = math.nan) -> float:
    try:
        if value is None or str(value).strip() == "":
            return default
        x = float(value)
        return x if math.isfinite(x) else default
    except (TypeError, ValueError, OverflowError):
        return default


def sanitize_token(value: Any) -> str:
    text = str(value).strip()
    text = re.sub(r"[^A-Za-z0-9_.-]+", "-", text)
    return text.strip("-") or "NA"


def build_recording_uid(row: pd.Series) -> str:
    session = sanitize_token(row.get("mapped_session_id", "NA"))
    phase = safe_int(row.get("mapped_phase"), 0) or 0
    pid = safe_int(row.get("mapped_pid"), 0) or 0
    sensor = sanitize_token(row.get("sensor_short", "NA"))
    attempt = sanitize_token(row.get("attempt_uid", "NA"))
    return f"{session}_P{phase}_PID{pid:03d}_S{sensor}_{attempt}"


def load_video_annotation_context(annotation_path: Path) -> pd.DataFrame:
    """Build one standard video/cooking interval record per acquisition attempt.

    These intervals are QC context only in Script 02. No ECG cropping or scientific
    inclusion/exclusion decision is made from them here.
    """
    columns = [
        "attempt_uid",
        "cooking_interval_status",
        "cooking_start_timestamp",
        "cooking_stop_timestamp",
        "cooking_interval_duration_s",
        "n_cooking_start_events",
        "n_cooking_stop_events",
    ]
    if not annotation_path.exists():
        return pd.DataFrame(columns=columns)

    events = pd.read_csv(annotation_path, keep_default_na=False)
    required = {"attempt_uid", "timestamp", "type"}
    missing = sorted(required - set(events.columns))
    if missing:
        raise RuntimeError(f"Script 01 annotation-event table is missing required columns: {missing}")

    events = events.copy()
    events["timestamp_numeric"] = pd.to_numeric(events["timestamp"], errors="coerce")
    rows: list[dict[str, Any]] = []
    for attempt_uid, g in events.groupby("attempt_uid", sort=False):
        starts = [float(x) for x in g.loc[g["type"] == "VideoAnnotationStart", "timestamp_numeric"].dropna().tolist()]
        stops = [float(x) for x in g.loc[g["type"] == "VideoAnnotationStop", "timestamp_numeric"].dropna().tolist()]
        starts = sorted(x for x in starts if math.isfinite(x))
        stops = sorted(x for x in stops if math.isfinite(x))

        status = "NO_STANDARD_VIDEO_INTERVAL"
        start = math.nan
        stop = math.nan
        duration = math.nan
        if len(starts) == 1 and len(stops) == 1 and stops[0] > starts[0]:
            status = "OK_SINGLE_VIDEO_INTERVAL"
            start, stop = starts[0], stops[0]
            duration = stop - start
        elif len(starts) or len(stops):
            status = "AMBIGUOUS_OR_INVALID_VIDEO_INTERVAL"

        rows.append(
            {
                "attempt_uid": str(attempt_uid),
                "cooking_interval_status": status,
                "cooking_start_timestamp": start,
                "cooking_stop_timestamp": stop,
                "cooking_interval_duration_s": duration,
                "n_cooking_start_events": len(starts),
                "n_cooking_stop_events": len(stops),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def attach_video_annotation_context(candidates: pd.DataFrame, annotation_path: Path) -> pd.DataFrame:
    out = candidates.copy()
    context = load_video_annotation_context(annotation_path)
    if context.empty:
        out["cooking_interval_status"] = (
            "ANNOTATION_EVENT_TABLE_UNAVAILABLE" if not annotation_path.exists() else "NO_STANDARD_VIDEO_INTERVAL"
        )
        out["cooking_start_timestamp"] = math.nan
        out["cooking_stop_timestamp"] = math.nan
        out["cooking_interval_duration_s"] = math.nan
        out["n_cooking_start_events"] = 0
        out["n_cooking_stop_events"] = 0
        return out

    # Annotation-derived cooking context uses dedicated column names so it never
    # collides with Script 01 file-level video metadata such as video_duration_s.
    context_cols = [c for c in context.columns if c != "attempt_uid"]
    stale = [c for c in context_cols if c in out.columns]
    if stale:
        out = out.drop(columns=stale)
    out = out.merge(context, on="attempt_uid", how="left", validate="many_to_one")
    out["cooking_interval_status"] = out["cooking_interval_status"].fillna("NO_STANDARD_VIDEO_INTERVAL")
    for col in ["cooking_start_timestamp", "cooking_stop_timestamp", "cooking_interval_duration_s"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    for col in ["n_cooking_start_events", "n_cooking_stop_events"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype(int)
    return out


def filter_candidates(candidates: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    """Apply explicit development/audit filters. Production defaults remain all candidates."""
    out = candidates.copy()
    if args.session_id:
        wanted = {str(x).strip().upper() for x in args.session_id}
        out = out[out["mapped_session_id"].astype(str).str.upper().isin(wanted)]
    if args.pid:
        wanted = {int(x) for x in args.pid}
        pid_numeric = pd.to_numeric(out["mapped_pid"], errors="coerce")
        out = out[pid_numeric.isin(wanted)]
    if args.phase:
        wanted = {int(x) for x in args.phase}
        phase_numeric = pd.to_numeric(out["mapped_phase"], errors="coerce")
        out = out[phase_numeric.isin(wanted)]
    if args.only_restarts:
        count = pd.to_numeric(out["participant_phase_candidate_count"], errors="coerce").fillna(1)
        out = out[count > 1]
    return out.copy()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True, allow_nan=False)
            f.write("\n")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def atomic_save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.stem + ".", suffix=".npz", dir=str(path.parent))
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return [json_safe(v) for v in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        x = float(value)
        return x if math.isfinite(x) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if value is None or isinstance(value, (str, int)):
        return value
    return str(value)


def load_processing_dependencies():
    try:
        import scipy  # noqa: F401
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - environment dependent
        raise ProcessingDependencyError(
            "Script 02 requires scipy and matplotlib. Install them in the Python "
            "environment used to run the script."
        ) from exc

    try:
        import neurokit2 as nk
    except Exception as exc:  # pragma: no cover - environment dependent
        raise ProcessingDependencyError(
            "Script 02 requires NeuroKit2. Install it with:\n"
            "    python3 -m pip install neurokit2\n"
            "or, in conda:\n"
            "    conda install -c conda-forge neurokit2"
        ) from exc

    return nk, plt


def detect_ecg_header(path: Path, max_lines: int = 30) -> tuple[int, int, int]:
    """Return 1-based header line number and Timestamp/Sample column indices."""
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as f:
        for line_number in range(1, max_lines + 1):
            line = f.readline()
            if line == "":
                break
            try:
                fields = next(csv.reader([line]))
            except csv.Error:
                continue
            normalized = [x.strip().lstrip("\ufeff") for x in fields]
            if "Timestamp" in normalized and "Sample" in normalized:
                return line_number, normalized.index("Timestamp"), normalized.index("Sample")
    raise ValueError("Could not find an ECG header containing Timestamp and Sample.")


def parse_ecg_csv(path: Path) -> ParsedECG:
    """Robustly parse one ECG CSV while retaining only finite Timestamp/Sample rows."""
    header_line, ts_col, sample_col = detect_ecg_header(path)
    max_col = max(ts_col, sample_col)

    timestamps: list[float] = []
    samples: list[float] = []
    source_rows: list[int] = []
    total_data_rows = 0
    invalid_timestamp_rows = 0
    invalid_sample_rows = 0
    malformed_rows = 0
    blank_rows = 0

    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as f:
        for _ in range(header_line):
            f.readline()
        reader = csv.reader(f)
        for row_number, row in enumerate(reader, start=header_line + 1):
            if not row or all(str(x).strip() == "" for x in row):
                blank_rows += 1
                continue
            total_data_rows += 1
            if len(row) <= max_col:
                malformed_rows += 1
                continue
            try:
                ts = float(str(row[ts_col]).strip())
            except (TypeError, ValueError, OverflowError):
                invalid_timestamp_rows += 1
                continue
            try:
                sample = float(str(row[sample_col]).strip())
            except (TypeError, ValueError, OverflowError):
                invalid_sample_rows += 1
                continue
            if not math.isfinite(ts):
                invalid_timestamp_rows += 1
                continue
            if not math.isfinite(sample):
                invalid_sample_rows += 1
                continue
            timestamps.append(ts)
            samples.append(sample)
            source_rows.append(row_number)

    return ParsedECG(
        timestamps=np.asarray(timestamps, dtype=np.float64),
        samples=np.asarray(samples, dtype=np.float64),
        source_rows=np.asarray(source_rows, dtype=np.int64),
        header_line_number=header_line,
        total_data_rows=total_data_rows,
        valid_rows=len(timestamps),
        invalid_timestamp_rows=invalid_timestamp_rows,
        invalid_sample_rows=invalid_sample_rows,
        malformed_rows=malformed_rows,
        blank_rows=blank_rows,
    )


def robust_sampling_metrics(timestamps: np.ndarray) -> dict[str, Any]:
    """Export-axis diagnostics only. The reciprocal median is NOT a rate selection."""
    if timestamps.size < 3:
        raise ValueError("Fewer than 3 valid ECG timestamps; sampling rate cannot be estimated.")

    diffs = np.diff(timestamps)
    finite = np.isfinite(diffs)
    positive = diffs[finite & (diffs > 0)]
    if positive.size < 2:
        raise ValueError("Insufficient positive timestamp differences to estimate sampling rate.")

    median_dt = float(np.median(positive))
    if not math.isfinite(median_dt) or median_dt <= 0:
        raise ValueError("Invalid median ECG sampling interval.")

    fs_est = 1.0 / median_dt
    fs_used = int(round(fs_est))  # legacy diagnostic only; overwritten by explicit policy
    if fs_used < 2:
        raise ValueError(f"Estimated sampling rate is implausibly low ({fs_est:.6g} Hz).")

    mad_dt = float(np.median(np.abs(positive - median_dt)))
    jitter_pct = 100.0 * mad_dt / median_dt if median_dt > 0 else math.nan
    fs_rounding_error_pct = 100.0 * abs(fs_est - fs_used) / fs_est if fs_est > 0 else math.nan

    return {
        "median_dt_s": median_dt,
        "median_dt_ms": median_dt * 1000.0,
        "reciprocal_median_spacing_hz": fs_est,
        "sampling_interval_mad_ms": mad_dt * 1000.0,
        "sampling_interval_jitter_mad_pct": jitter_pct,
        "legacy_median_rounding_difference_pct": fs_rounding_error_pct,
        "n_nonpositive_timestamp_diffs": int(np.sum(finite & (diffs <= 0))),
        "n_duplicate_timestamps": int(np.sum(finite & (diffs == 0))),
        "n_negative_timestamp_diffs": int(np.sum(finite & (diffs < 0))),
        "min_positive_dt_s": float(np.min(positive)),
        "max_positive_dt_s": float(np.max(positive)),
    }


def rate_policy(path: Path, timestamps: np.ndarray, policy: str, legacy_rate=None) -> dict[str, Any]:
    import re
    with path.open(encoding='utf-8-sig', errors='replace') as f:
        header = f.readline().strip()
    match = re.search(r'\d{2}\.\d{2}\.\d{2}\d{2}\.\d{2}\.\d{2}(\d+)\s*$', header)
    nominal = int(match.group(1)) if match else None
    if policy == 'legacy_replay':
        if legacy_rate not in {125,128,200,250,256,500,512}:
            raise ValueError('A recorded per-record detector design rate is required for legacy replay')
        used = int(legacy_rate)
    elif policy == 'header_nominal' and nominal in {125,128,200,250,256,500,512}:
        used = nominal
    else:
        raise ValueError('Recognized header rate required for header_nominal policy')
    dt=np.diff(timestamps)
    continuous=(dt>0)&(dt<=.04)
    count_rate=float(continuous.sum()/dt[continuous].sum()) if continuous.any() else math.nan
    return dict(sampling_rate_hz_used=used, detector_rate_policy=policy,
                sampling_rate_hz_est=None, reciprocal_median_spacing_hz=float(1/np.median(dt[dt>0])),
                sampling_rate_rounding_error_pct=None, header_nominal_rate_hz=nominal,
                continuous_sample_count_per_exported_s=count_rate,
                detector_rate_is_physical_calibration=False,
                detector_changes_require_review=policy!='legacy_replay')


def build_segments(
    timestamps: np.ndarray,
    median_dt_s: float,
    gap_factor: float,
    gap_min_seconds: float,
) -> tuple[list[SegmentDef], dict[str, Any]]:
    if timestamps.size == 0:
        return [], {
            "gap_threshold_s": math.nan,
            "n_gap_events": 0,
            "max_gap_s": math.nan,
            "total_gap_excess_s": 0.0,
        }

    gap_threshold = max(gap_min_seconds, gap_factor * median_dt_s)
    diffs = np.diff(timestamps)
    boundary_indices: list[int] = []
    reasons: dict[int, str] = {}

    gap_values: list[float] = []
    total_gap_excess = 0.0
    for i, dt in enumerate(diffs):
        split_index = i + 1
        if not math.isfinite(float(dt)):
            boundary_indices.append(split_index)
            reasons[split_index] = "NONFINITE_TIMESTAMP_DIFFERENCE"
        elif dt <= 0:
            boundary_indices.append(split_index)
            reasons[split_index] = "NONPOSITIVE_TIMESTAMP_DIFFERENCE"
        elif dt > gap_threshold:
            boundary_indices.append(split_index)
            reasons[split_index] = "LARGE_TIMESTAMP_GAP"
            gap_values.append(float(dt))
            total_gap_excess += max(0.0, float(dt) - median_dt_s)

    boundaries = [0] + sorted(set(boundary_indices)) + [len(timestamps)]
    segments: list[SegmentDef] = []
    for seg_id, (start, stop) in enumerate(zip(boundaries[:-1], boundaries[1:]), start=1):
        if stop <= start:
            continue
        start_ts = float(timestamps[start])
        stop_ts = float(timestamps[stop - 1])
        duration = max(0.0, stop_ts - start_ts)
        segments.append(
            SegmentDef(
                segment_id=seg_id,
                start=start,
                stop=stop,
                start_timestamp=start_ts,
                stop_timestamp=stop_ts,
                n_samples=stop - start,
                duration_s=duration,
                boundary_reason_before="START_OF_RECORDING" if start == 0 else reasons.get(start, "UNKNOWN"),
            )
        )

    return segments, {
        "gap_threshold_s": gap_threshold,
        "n_gap_events": len(gap_values),
        "max_gap_s": max(gap_values) if gap_values else 0.0,
        "total_gap_excess_s": total_gap_excess,
        "n_segments": len(segments),
    }


def interval_union_length(intervals: list[tuple[float, float]]) -> float:
    cleaned = sorted((float(a), float(b)) for a, b in intervals if math.isfinite(a) and math.isfinite(b) and b >= a)
    if not cleaned:
        return 0.0
    total = 0.0
    cur_a, cur_b = cleaned[0]
    for a, b in cleaned[1:]:
        if a <= cur_b:
            cur_b = max(cur_b, b)
        else:
            total += max(0.0, cur_b - cur_a)
            cur_a, cur_b = a, b
    total += max(0.0, cur_b - cur_a)
    return total


def segment_time_metrics(
    segments: list[SegmentDef],
    tolerance_s: float,
    ambiguity_threshold_s: float,
) -> dict[str, Any]:
    intervals = [(s.start_timestamp, s.stop_timestamp) for s in segments]
    duration_sum = float(sum(max(0.0, b - a) for a, b in intervals))
    union_s = interval_union_length(intervals)
    overlap_excess = max(0.0, duration_sum - union_s)

    n_pairs = 0
    max_pair_overlap_s = 0.0
    for i in range(len(intervals)):
        a1, b1 = intervals[i]
        for j in range(i + 1, len(intervals)):
            a2, b2 = intervals[j]
            pair_overlap_s = max(0.0, min(b1, b2) - max(a1, a2))
            if pair_overlap_s > tolerance_s:
                n_pairs += 1
                max_pair_overlap_s = max(max_pair_overlap_s, pair_overlap_s)

    if n_pairs > 0 and max_pair_overlap_s > ambiguity_threshold_s:
        status = "SEGMENT_TIME_OVERLAP_AMBIGUOUS"
    elif n_pairs > 0:
        status = "SEGMENT_TIME_MINOR_OVERLAP"
    elif len(segments) > 1:
        status = "SEGMENTED_NONOVERLAPPING"
    else:
        status = "MONOTONIC_SINGLE_SEGMENT"

    return {
        "timestamp_integrity_status": status,
        "segment_duration_sum_s": duration_sum,
        "segment_time_union_s": union_s,
        "segment_time_overlap_excess_s": overlap_excess,
        "n_overlapping_segment_pairs": n_pairs,
        "max_segment_pair_overlap_s": max_pair_overlap_s,
        "segment_overlap_ambiguity_threshold_s": ambiguity_threshold_s,
    }


def interval_coverage_within_window(intervals: list[tuple[float, float]], start: float, stop: float) -> float:
    if not (math.isfinite(start) and math.isfinite(stop) and stop > start):
        return math.nan
    clipped = [(max(a, start), min(b, stop)) for a, b in intervals if min(b, stop) > max(a, start)]
    return interval_union_length(clipped)


def percentile_or_nan(values: np.ndarray, q: float) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return math.nan
    return float(np.percentile(values, q))


def mean_or_nan(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if values.size else math.nan


def std_or_nan(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.std(values, ddof=1)) if values.size >= 2 else math.nan


def artifact_count(info: dict[str, Any], key: str) -> int:
    value = info.get(key, []) if isinstance(info, dict) else []
    try:
        return len(value)
    except TypeError:
        return 0


def call_kubios_fixpeaks(nk: Any, peaks: np.ndarray, sampling_rate: int) -> tuple[dict[str, Any], np.ndarray]:
    result = nk.signal_fixpeaks(
        peaks,
        sampling_rate=sampling_rate,
        iterative=True,
        method="Kubios",
        show=False,
    )
    if not isinstance(result, tuple) or len(result) != 2:
        raise RuntimeError("Unexpected NeuroKit2 signal_fixpeaks() return value.")
    first, second = result
    # NeuroKit2 currently returns (artifact_info, corrected_peaks). This defensive
    # branch protects against documentation/API ordering inconsistencies.
    if isinstance(first, dict):
        artifacts = first
        corrected = np.asarray(second, dtype=int)
    elif isinstance(second, dict):
        artifacts = second
        corrected = np.asarray(first, dtype=int)
    else:
        raise RuntimeError("Could not identify artifact info in signal_fixpeaks() return value.")
    corrected = corrected[np.isfinite(corrected)] if corrected.dtype.kind == "f" else corrected
    corrected = np.asarray(corrected, dtype=int)
    corrected = np.unique(corrected)
    return artifacts, corrected


def process_one_recording(
    row: pd.Series,
    raw_root: Path,
    output_dir: Path,
    nk: Any,
    plt: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    recording_uid = build_recording_uid(row)
    record_dir = output_dir / "records" / recording_uid
    record_dir.mkdir(parents=True, exist_ok=True)
    result_path = record_dir / "02_result.json"
    beat_npz_path = record_dir / "02_beats.npz"
    plot_path = output_dir / "qc_plots" / f"{recording_uid}.png"

    if args.resume and result_path.exists() and beat_npz_path.exists():
        with result_path.open("r", encoding="utf-8") as f:
            result = json.load(f)
        result["resumed_from_existing_result"] = True
        return result

    start_clock = time.perf_counter()
    issues: list[dict[str, str]] = []
    segment_rows: list[dict[str, Any]] = []

    ecg_rel = str(row.get("ECG_path", "")).strip()
    ecg_path = raw_root / ecg_rel

    base_result: dict[str, Any] = {
        "recording_uid": recording_uid,
        "attempt_uid": str(row.get("attempt_uid", "")),
        "Session_ID": str(row.get("mapped_session_id", "")),
        "PID": safe_int(row.get("mapped_pid")),
        "Phase": safe_int(row.get("mapped_phase")),
        "Condition": str(row.get("mapped_condition", "")),
        "sensor_full": str(row.get("sensor_full", "")),
        "sensor_short": str(row.get("sensor_short", "")),
        "Analysis_Partner": str(row.get("analysis_partner", "")),
        "Analysis_Role": str(row.get("analysis_role", "")),
        "PrimarySynchrony_candidate_from_script01": as_bool(row.get("PrimarySynchrony_candidate")),
        "participant_phase_candidate_count": safe_int(row.get("participant_phase_candidate_count"), 1),
        "script01_video_annotation_status": str(row.get("video_annotation_status", "")),
        "cooking_interval_status": str(row.get("cooking_interval_status", "")),
        "cooking_start_timestamp": safe_float(row.get("cooking_start_timestamp")),
        "cooking_stop_timestamp": safe_float(row.get("cooking_stop_timestamp")),
        "cooking_interval_duration_s": safe_float(row.get("cooking_interval_duration_s")),
        "n_cooking_start_events": safe_int(row.get("n_cooking_start_events"), 0),
        "n_cooking_stop_events": safe_int(row.get("n_cooking_stop_events"), 0),
        "ECG_path": ecg_rel,
        "ECG_absolute_path": str(ecg_path),
        "script01_ECG_schema_status": str(row.get("ECG_schema_status", "")),
        "script01_ECG_time_envelope_ok": as_bool(row.get("ECG_time_envelope_ok")),
        "script01_ECG_file_size_bytes": safe_int(row.get("ECG_file_size_bytes"), 0),
        "processing_status": "NOT_STARTED",
        "generated_utc": utc_now_iso(),
        "resumed_from_existing_result": False,
    }

    if not ecg_path.exists():
        issues.append({"severity": "ERROR", "code": "ECG_FILE_NOT_FOUND", "detail": str(ecg_path)})
        base_result.update(
            {
                "processing_status": "ERROR_FILE_NOT_FOUND",
                "issues": issues,
                "segments": segment_rows,
                "runtime_seconds": time.perf_counter() - start_clock,
                "beat_npz_path": "",
                "qc_plot_path": "",
            }
        )
        atomic_write_json(result_path, json_safe(base_result))
        return base_result

    try:
        parsed = parse_ecg_csv(ecg_path)
    except Exception as exc:
        issues.append({"severity": "ERROR", "code": "ECG_PARSE_FAILED", "detail": f"{type(exc).__name__}: {exc}"})
        base_result.update(
            {
                "processing_status": "ERROR_PARSE_FAILED",
                "issues": issues,
                "segments": segment_rows,
                "runtime_seconds": time.perf_counter() - start_clock,
                "beat_npz_path": "",
                "qc_plot_path": "",
            }
        )
        atomic_write_json(result_path, json_safe(base_result))
        return base_result

    if parsed.valid_rows < 3:
        issues.append({"severity": "ERROR", "code": "TOO_FEW_VALID_ECG_ROWS", "detail": f"valid_rows={parsed.valid_rows}"})
        base_result.update(
            {
                "processing_status": "ERROR_TOO_FEW_VALID_ROWS",
                "header_line_number": parsed.header_line_number,
                "total_data_rows": parsed.total_data_rows,
                "valid_rows": parsed.valid_rows,
                "invalid_timestamp_rows": parsed.invalid_timestamp_rows,
                "invalid_sample_rows": parsed.invalid_sample_rows,
                "malformed_rows": parsed.malformed_rows,
                "blank_rows": parsed.blank_rows,
                "issues": issues,
                "segments": segment_rows,
                "runtime_seconds": time.perf_counter() - start_clock,
                "beat_npz_path": "",
                "qc_plot_path": "",
            }
        )
        atomic_write_json(result_path, json_safe(base_result))
        return base_result

    video_start_ts = safe_float(row.get("cooking_start_timestamp"))
    video_stop_ts = safe_float(row.get("cooking_stop_timestamp"))
    valid_video_interval = (
        math.isfinite(video_start_ts)
        and math.isfinite(video_stop_ts)
        and video_stop_ts > video_start_ts
    )
    if valid_video_interval:
        ecg_first_ts = float(parsed.timestamps[0])
        ecg_last_ts = float(parsed.timestamps[-1])
        video_start_offset_from_ecg_s = video_start_ts - ecg_first_ts
        video_stop_offset_from_ecg_s = video_stop_ts - ecg_first_ts
        video_ecg_overlap_s = max(
            0.0,
            min(ecg_last_ts, video_stop_ts) - max(ecg_first_ts, video_start_ts),
        )
        video_interval_overlaps_ecg = video_ecg_overlap_s > 0
        video_interval_fully_within_ecg = (
            video_start_ts >= ecg_first_ts and video_stop_ts <= ecg_last_ts
        )
    else:
        video_start_offset_from_ecg_s = math.nan
        video_stop_offset_from_ecg_s = math.nan
        video_ecg_overlap_s = math.nan
        video_interval_overlaps_ecg = False
        video_interval_fully_within_ecg = False

    try:
        sm = robust_sampling_metrics(parsed.timestamps)
        historic=json.loads((SCRIPT_PATH.parent/'inputs/historical_detector_rates_185.json').read_text())
        sm.update(rate_policy(ecg_path, parsed.timestamps, args.sampling_rate_policy,
                              legacy_rate=historic.get(recording_uid)))
    except Exception as exc:
        issues.append({"severity": "ERROR", "code": "SAMPLING_RATE_ESTIMATION_FAILED", "detail": f"{type(exc).__name__}: {exc}"})
        base_result.update(
            {
                "processing_status": "ERROR_SAMPLING_RATE",
                "header_line_number": parsed.header_line_number,
                "total_data_rows": parsed.total_data_rows,
                "valid_rows": parsed.valid_rows,
                "invalid_timestamp_rows": parsed.invalid_timestamp_rows,
                "invalid_sample_rows": parsed.invalid_sample_rows,
                "malformed_rows": parsed.malformed_rows,
                "blank_rows": parsed.blank_rows,
                "issues": issues,
                "segments": segment_rows,
                "runtime_seconds": time.perf_counter() - start_clock,
                "beat_npz_path": "",
                "qc_plot_path": "",
            }
        )
        atomic_write_json(result_path, json_safe(base_result))
        return base_result

    segments, gap_metrics = build_segments(
        parsed.timestamps,
        median_dt_s=sm["median_dt_s"],
        gap_factor=args.gap_factor,
        gap_min_seconds=args.gap_min_seconds,
    )
    time_metrics = segment_time_metrics(
        segments,
        tolerance_s=max(2.0 * sm["median_dt_s"], 1e-6),
        ambiguity_threshold_s=max(gap_metrics["gap_threshold_s"], 1e-6),
    )
    timestamp_envelope_s = float(np.max(parsed.timestamps) - np.min(parsed.timestamps))
    row_order_timestamp_delta_s = float(parsed.timestamps[-1] - parsed.timestamps[0])

    raw_intervals = [(seg.start_timestamp, seg.stop_timestamp) for seg in segments]
    if valid_video_interval:
        video_start_offset_from_ecg_s = video_start_ts - float(np.min(parsed.timestamps))
        video_stop_offset_from_ecg_s = video_stop_ts - float(np.min(parsed.timestamps))
        video_ecg_overlap_s = interval_coverage_within_window(raw_intervals, video_start_ts, video_stop_ts)
        video_interval_overlaps_ecg = bool(math.isfinite(video_ecg_overlap_s) and video_ecg_overlap_s > 0)
        video_duration = video_stop_ts - video_start_ts
        video_interval_fully_within_ecg = bool(video_ecg_overlap_s >= max(0.0, video_duration - 2.0 * sm["median_dt_s"]))
    else:
        video_start_offset_from_ecg_s = math.nan
        video_stop_offset_from_ecg_s = math.nan
        video_ecg_overlap_s = math.nan
        video_interval_overlaps_ecg = False
        video_interval_fully_within_ecg = False

    if time_metrics["timestamp_integrity_status"] == "SEGMENT_TIME_OVERLAP_AMBIGUOUS":
        issues.append({
            "severity": "ERROR",
            "code": "TIMESTAMP_SEGMENTS_OVERLAP_AMBIGUOUS",
            "detail": (
                f"overlapping_pairs={time_metrics['n_overlapping_segment_pairs']}; "
                f"max_pair_overlap_s={time_metrics['max_segment_pair_overlap_s']:.6g}; "
                f"overlap_excess_s={time_metrics['segment_time_overlap_excess_s']:.6g}; "
                f"ambiguity_threshold_s={time_metrics['segment_overlap_ambiguity_threshold_s']:.6g}. "
                "Beat morphology can be QC-processed per segment, but absolute time alignment is ambiguous; "
                "do not use this recording for synchrony until timestamp provenance is resolved."
            ),
        })
    elif time_metrics["timestamp_integrity_status"] == "SEGMENT_TIME_MINOR_OVERLAP":
        issues.append({
            "severity": "WARNING",
            "code": "TIMESTAMP_SEGMENTS_MINOR_OVERLAP",
            "detail": (
                f"overlapping_pairs={time_metrics['n_overlapping_segment_pairs']}; "
                f"max_pair_overlap_s={time_metrics['max_segment_pair_overlap_s']:.6g}; "
                f"overlap_excess_s={time_metrics['segment_time_overlap_excess_s']:.6g}; "
                f"ambiguity_threshold_s={time_metrics['segment_overlap_ambiguity_threshold_s']:.6g}. "
                "Small boundary overlap retained as a QC warning; timestamps were not reconstructed."
            ),
        })

    if parsed.invalid_timestamp_rows or parsed.invalid_sample_rows or parsed.malformed_rows:
        issues.append(
            {
                "severity": "WARNING",
                "code": "RAW_ECG_ROWS_SKIPPED",
                "detail": (
                    f"invalid_timestamp_rows={parsed.invalid_timestamp_rows}; "
                    f"invalid_sample_rows={parsed.invalid_sample_rows}; malformed_rows={parsed.malformed_rows}"
                ),
            }
        )
    if sm["n_nonpositive_timestamp_diffs"] > 0:
        issues.append(
            {
                "severity": "WARNING",
                "code": "NONMONOTONIC_OR_DUPLICATE_TIMESTAMPS",
                "detail": (
                    f"nonpositive={sm['n_nonpositive_timestamp_diffs']}; "
                    f"duplicates={sm['n_duplicate_timestamps']}; negative={sm['n_negative_timestamp_diffs']}"
                ),
            }
        )
    if gap_metrics["n_gap_events"] > 0:
        issues.append(
            {
                "severity": "WARNING",
                "code": "ECG_TIMESTAMP_GAPS_PRESENT",
                "detail": (
                    f"n_gap_events={gap_metrics['n_gap_events']}; "
                    f"max_gap_s={gap_metrics['max_gap_s']:.6g}; threshold_s={gap_metrics['gap_threshold_s']:.6g}"
                ),
            }
        )
    if not as_bool(row.get("ECG_time_envelope_ok")):
        issues.append(
            {
                "severity": "WARNING",
                "code": "SCRIPT01_ECG_TAIL_TIMESTAMP_NOT_TRUSTWORTHY",
                "detail": f"Script 01 schema status: {row.get('ECG_schema_status', '')}",
            }
        )
    if safe_int(row.get("participant_phase_candidate_count"), 1) and safe_int(row.get("participant_phase_candidate_count"), 1) > 1:
        issues.append(
            {
                "severity": "INFO",
                "code": "MULTIPLE_RECORDING_CANDIDATES_FOR_PARTICIPANT_PHASE",
                "detail": f"candidate_count={safe_int(row.get('participant_phase_candidate_count'), 1)}; no selection performed",
            }
        )

    fs_used = int(sm["sampling_rate_hz_used"])
    powerline_applied = args.powerline_hz > 0 and args.powerline_hz < (0.49 * fs_used)
    if args.powerline_hz > 0 and not powerline_applied:
        issues.append(
            {
                "severity": "WARNING",
                "code": "POWERLINE_FILTER_SKIPPED_NYQUIST",
                "detail": f"powerline_hz={args.powerline_hz}; sampling_rate_hz_used={fs_used}",
            }
        )

    cleaned_all = np.full(parsed.valid_rows, np.nan, dtype=np.float64)
    original_global: list[int] = []
    corrected_global: list[int] = []
    original_segment_ids: list[int] = []
    corrected_segment_ids: list[int] = []
    neurokit_warning_messages: list[str] = []
    all_artifact_counts = Counter()
    n_removed_original_total = 0
    n_added_corrected_total = 0
    processed_duration_s = 0.0
    n_segments_processed = 0
    n_segments_skipped_short = 0
    n_segments_failed = 0

    for seg in segments:
        seg_signal = parsed.samples[seg.start : seg.stop]
        seg_ts = parsed.timestamps[seg.start : seg.stop]
        min_samples = max(3, int(math.ceil(args.min_segment_seconds * fs_used)))

        seg_row: dict[str, Any] = {
            "recording_uid": recording_uid,
            "segment_id": seg.segment_id,
            "start_valid_index": seg.start,
            "stop_valid_index_exclusive": seg.stop,
            "start_timestamp": seg.start_timestamp,
            "stop_timestamp": seg.stop_timestamp,
            "n_samples": seg.n_samples,
            "duration_s": seg.duration_s,
            "boundary_reason_before": seg.boundary_reason_before,
            "status": "NOT_STARTED",
            "n_rpeaks_original": 0,
            "n_rpeaks_corrected": 0,
            "kubios_ectopic_final": 0,
            "kubios_missed_final": 0,
            "kubios_extra_final": 0,
            "kubios_longshort_final": 0,
            "n_removed_original_peaks": 0,
            "n_added_corrected_peaks": 0,
            "warning_count": 0,
        }

        if seg.n_samples < min_samples or seg.duration_s < args.min_segment_seconds * 0.8:
            seg_row["status"] = "SKIPPED_SHORT_SEGMENT"
            n_segments_skipped_short += 1
            segment_rows.append(seg_row)
            continue

        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                if powerline_applied:
                    cleaned = nk.ecg_clean(
                        seg_signal,
                        sampling_rate=fs_used,
                        method="neurokit",
                        powerline=args.powerline_hz,
                    )
                else:
                    # Do not silently apply NeuroKit2's default 50 Hz powerline filter
                    # when the requested frequency is invalid for the estimated Nyquist.
                    # Use the 0.5 Hz high-pass directly via NeuroKit's signal_filter.
                    cleaned = nk.signal_filter(
                        seg_signal,
                        sampling_rate=fs_used,
                        lowcut=0.5,
                        method="butterworth",
                        order=5,
                    )
                cleaned = np.asarray(cleaned, dtype=np.float64)
                _, peak_info = nk.ecg_peaks(
                    cleaned,
                    sampling_rate=fs_used,
                    method="neurokit",
                    correct_artifacts=False,
                    show=False,
                )
                r_orig = np.asarray(peak_info.get("ECG_R_Peaks", []), dtype=int)
                r_orig = np.unique(r_orig[(r_orig >= 0) & (r_orig < len(cleaned))])

                if r_orig.size >= 4:
                    artifacts, r_corr = call_kubios_fixpeaks(nk, r_orig, fs_used)
                    r_corr = r_corr[(r_corr >= 0) & (r_corr < len(cleaned))]
                else:
                    artifacts = {"ectopic": [], "missed": [], "extra": [], "longshort": []}
                    r_corr = np.asarray([], dtype=int)
                    issues.append(
                        {
                            "severity": "WARNING",
                            "code": "TOO_FEW_RPEAKS_FOR_KUBIOS_SEGMENT",
                            "detail": f"segment={seg.segment_id}; n_rpeaks_original={len(r_orig)}",
                        }
                    )

                for warning_obj in caught:
                    neurokit_warning_messages.append(str(warning_obj.message))

            cleaned_all[seg.start : seg.stop] = cleaned
            orig_set = set(map(int, r_orig.tolist()))
            corr_set = set(map(int, r_corr.tolist()))
            n_removed = len(orig_set - corr_set)
            n_added = len(corr_set - orig_set)

            global_orig = (r_orig + seg.start).astype(int)
            global_corr = (r_corr + seg.start).astype(int)
            original_global.extend(global_orig.tolist())
            corrected_global.extend(global_corr.tolist())
            original_segment_ids.extend([seg.segment_id] * len(global_orig))
            corrected_segment_ids.extend([seg.segment_id] * len(global_corr))

            final_counts = {
                "ectopic": artifact_count(artifacts, "ectopic"),
                "missed": artifact_count(artifacts, "missed"),
                "extra": artifact_count(artifacts, "extra"),
                "longshort": artifact_count(artifacts, "longshort"),
            }
            all_artifact_counts.update(final_counts)
            n_removed_original_total += n_removed
            n_added_corrected_total += n_added
            processed_duration_s += seg.duration_s
            n_segments_processed += 1

            seg_row.update(
                {
                    "status": "PROCESSED" if len(r_corr) >= 2 else "PROCESSED_NO_CORRECTED_RR",
                    "n_rpeaks_original": int(len(r_orig)),
                    "n_rpeaks_corrected": int(len(r_corr)),
                    "kubios_ectopic_final": final_counts["ectopic"],
                    "kubios_missed_final": final_counts["missed"],
                    "kubios_extra_final": final_counts["extra"],
                    "kubios_longshort_final": final_counts["longshort"],
                    "n_removed_original_peaks": n_removed,
                    "n_added_corrected_peaks": n_added,
                    "warning_count": len(caught),
                }
            )
        except Exception as exc:
            n_segments_failed += 1
            seg_row["status"] = "ERROR_PROCESSING_SEGMENT"
            seg_row["error"] = f"{type(exc).__name__}: {exc}"
            issues.append(
                {
                    "severity": "ERROR",
                    "code": "ECG_SEGMENT_PROCESSING_FAILED",
                    "detail": f"segment={seg.segment_id}; {type(exc).__name__}: {exc}",
                }
            )
        segment_rows.append(seg_row)

    original_global_arr = np.asarray(original_global, dtype=np.int64)
    corrected_global_arr = np.asarray(corrected_global, dtype=np.int64)
    original_segment_arr = np.asarray(original_segment_ids, dtype=np.int32)
    corrected_segment_arr = np.asarray(corrected_segment_ids, dtype=np.int32)

    # Sort by original recording index while preserving segment labels.
    if original_global_arr.size:
        order = np.argsort(original_global_arr, kind="stable")
        original_global_arr = original_global_arr[order]
        original_segment_arr = original_segment_arr[order]
    if corrected_global_arr.size:
        order = np.argsort(corrected_global_arr, kind="stable")
        corrected_global_arr = corrected_global_arr[order]
        corrected_segment_arr = corrected_segment_arr[order]

    original_ts = parsed.timestamps[original_global_arr] if original_global_arr.size else np.asarray([], dtype=float)
    corrected_ts = parsed.timestamps[corrected_global_arr] if corrected_global_arr.size else np.asarray([], dtype=float)
    original_source_rows = parsed.source_rows[original_global_arr] if original_global_arr.size else np.asarray([], dtype=np.int64)
    corrected_source_rows = parsed.source_rows[corrected_global_arr] if corrected_global_arr.size else np.asarray([], dtype=np.int64)

    def rr_from_segmented_timestamps(peak_ts: np.ndarray, seg_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        rr = np.full(len(peak_ts), np.nan, dtype=np.float64)
        hr = np.full(len(peak_ts), np.nan, dtype=np.float64)
        if len(peak_ts) == 0:
            return rr, hr
        for seg_id in np.unique(seg_ids):
            idx = np.where(seg_ids == seg_id)[0]
            if len(idx) >= 2:
                intervals = np.diff(peak_ts[idx]) * 1000.0
                rr[idx[1:]] = intervals
                positive = intervals > 0
                local_hr = np.full_like(intervals, np.nan, dtype=np.float64)
                local_hr[positive] = 60000.0 / intervals[positive]
                hr[idx[1:]] = local_hr
        return rr, hr

    original_rr_ms, original_hr_bpm = rr_from_segmented_timestamps(original_ts, original_segment_arr)
    corrected_rr_ms, corrected_hr_bpm = rr_from_segmented_timestamps(corrected_ts, corrected_segment_arr)

    if n_segments_processed == 0:
        issues.append({"severity": "ERROR", "code": "NO_PROCESSABLE_ECG_SEGMENT", "detail": "No segment met processing requirements."})
    if corrected_global_arr.size < 2:
        issues.append({"severity": "ERROR", "code": "NO_CORRECTED_RR_SERIES", "detail": f"n_corrected_rpeaks={len(corrected_global_arr)}"})

    # Basic amplitude metrics in native Movesense sample units. These are descriptive only.
    raw_diffs = np.diff(parsed.samples)
    flatline_exact_fraction = float(np.mean(raw_diffs == 0)) if raw_diffs.size else math.nan

    valid_span_s = timestamp_envelope_s if parsed.valid_rows >= 2 else math.nan
    processed_intervals = [
        (float(seg_row["start_timestamp"]), float(seg_row["stop_timestamp"]))
        for seg_row in segment_rows if seg_row.get("status") in {"PROCESSED", "PROCESSED_NO_CORRECTED_RR"}
    ]
    processed_unique_time_coverage_s = interval_union_length(processed_intervals)
    processed_fraction = (
        processed_unique_time_coverage_s / valid_span_s
        if math.isfinite(valid_span_s) and valid_span_s > 0 else math.nan
    )
    cooking_processed_overlap_s = (
        interval_coverage_within_window(processed_intervals, video_start_ts, video_stop_ts)
        if valid_video_interval else math.nan
    )

    neurokit_warning_unique = sorted(set(neurokit_warning_messages))
    if neurokit_warning_unique:
        issues.append(
            {
                "severity": "WARNING",
                "code": "NEUROKIT_WARNINGS_EMITTED",
                "detail": " | ".join(neurokit_warning_unique[:10]),
            }
        )

    technical_error_count = sum(1 for x in issues if x["severity"] == "ERROR")
    processing_status = "PROCESSED" if technical_error_count == 0 else "PROCESSED_WITH_ERRORS"

    arrays = {
        "original_rpeak_valid_index": original_global_arr,
        "original_rpeak_source_row": original_source_rows.astype(np.int64),
        "original_rpeak_timestamp": original_ts.astype(np.float64),
        "original_rpeak_segment_id": original_segment_arr.astype(np.int32),
        "original_rr_ms": original_rr_ms.astype(np.float64),
        "original_hr_bpm": original_hr_bpm.astype(np.float64),
        "corrected_rpeak_valid_index": corrected_global_arr,
        "corrected_rpeak_source_row": corrected_source_rows.astype(np.int64),
        "corrected_rpeak_timestamp": corrected_ts.astype(np.float64),
        "corrected_rpeak_segment_id": corrected_segment_arr.astype(np.int32),
        "corrected_rr_ms": corrected_rr_ms.astype(np.float64),
        "corrected_hr_bpm": corrected_hr_bpm.astype(np.float64),
    }
    atomic_save_npz(beat_npz_path, **arrays)

    if not args.no_plots:
        try:
            create_qc_plot(
                plt=plt,
                recording_uid=recording_uid,
                parsed=parsed,
                cleaned_all=cleaned_all,
                segments=segments,
                original_global=original_global_arr,
                corrected_global=corrected_global_arr,
                original_rr_ms=original_rr_ms,
                corrected_rr_ms=corrected_rr_ms,
                corrected_hr_bpm=corrected_hr_bpm,
                video_start_timestamp=video_start_ts if valid_video_interval else math.nan,
                video_stop_timestamp=video_stop_ts if valid_video_interval else math.nan,
                plot_window_seconds=args.plot_window_seconds,
                max_plot_points=args.max_plot_points,
                output_path=plot_path,
            )
        except Exception as exc:
            issues.append(
                {
                    "severity": "WARNING",
                    "code": "QC_PLOT_FAILED",
                    "detail": f"{type(exc).__name__}: {exc}",
                }
            )

    result = dict(base_result)
    result.update(
        {
            "processing_status": processing_status,
            "header_line_number": parsed.header_line_number,
            "total_data_rows": parsed.total_data_rows,
            "valid_rows": parsed.valid_rows,
            "invalid_timestamp_rows": parsed.invalid_timestamp_rows,
            "invalid_sample_rows": parsed.invalid_sample_rows,
            "malformed_rows": parsed.malformed_rows,
            "blank_rows": parsed.blank_rows,
            "valid_row_fraction": parsed.valid_rows / parsed.total_data_rows if parsed.total_data_rows else math.nan,
            "valid_first_timestamp": float(parsed.timestamps[0]),
            "valid_last_timestamp": float(parsed.timestamps[-1]),
            "valid_timestamp_span_s": valid_span_s,
            "timestamp_envelope_s": timestamp_envelope_s,
            "row_order_timestamp_delta_s": row_order_timestamp_delta_s,
            **time_metrics,
            "cooking_start_offset_from_ecg_s": video_start_offset_from_ecg_s,
            "cooking_stop_offset_from_ecg_s": video_stop_offset_from_ecg_s,
            "cooking_ecg_overlap_s": video_ecg_overlap_s,
            "cooking_interval_overlaps_ecg": video_interval_overlaps_ecg,
            "cooking_interval_fully_within_ecg": video_interval_fully_within_ecg,
            **sm,
            **gap_metrics,
            "powerline_hz_requested": args.powerline_hz,
            "powerline_filter_applied": powerline_applied,
            "min_segment_seconds": args.min_segment_seconds,
            "n_segments_processed": n_segments_processed,
            "n_segments_skipped_short": n_segments_skipped_short,
            "n_segments_failed": n_segments_failed,
            "processed_duration_s": processed_duration_s,
            "processed_unique_time_coverage_s": processed_unique_time_coverage_s,
            "processed_fraction_of_valid_span": processed_fraction,
            "cooking_processed_overlap_s": cooking_processed_overlap_s,
            "raw_sample_min": float(np.min(parsed.samples)),
            "raw_sample_p01": percentile_or_nan(parsed.samples, 1),
            "raw_sample_p50": percentile_or_nan(parsed.samples, 50),
            "raw_sample_p99": percentile_or_nan(parsed.samples, 99),
            "raw_sample_max": float(np.max(parsed.samples)),
            "raw_sample_iqr": percentile_or_nan(parsed.samples, 75) - percentile_or_nan(parsed.samples, 25),
            "raw_exact_flatline_diff_fraction": flatline_exact_fraction,
            "n_rpeaks_original": int(len(original_global_arr)),
            "n_rpeaks_corrected": int(len(corrected_global_arr)),
            "kubios_ectopic_final": int(all_artifact_counts["ectopic"]),
            "kubios_missed_final": int(all_artifact_counts["missed"]),
            "kubios_extra_final": int(all_artifact_counts["extra"]),
            "kubios_longshort_final": int(all_artifact_counts["longshort"]),
            "n_removed_original_peaks": int(n_removed_original_total),
            "n_added_corrected_peaks": int(n_added_corrected_total),
            "original_peak_change_fraction": (
                n_removed_original_total / len(original_global_arr) if len(original_global_arr) else math.nan
            ),
            "corrected_rr_count": int(np.sum(np.isfinite(corrected_rr_ms))),
            "corrected_rr_ms_mean": mean_or_nan(corrected_rr_ms),
            "corrected_rr_ms_sd": std_or_nan(corrected_rr_ms),
            "corrected_rr_ms_p01": percentile_or_nan(corrected_rr_ms, 1),
            "corrected_rr_ms_p05": percentile_or_nan(corrected_rr_ms, 5),
            "corrected_rr_ms_p50": percentile_or_nan(corrected_rr_ms, 50),
            "corrected_rr_ms_p95": percentile_or_nan(corrected_rr_ms, 95),
            "corrected_rr_ms_p99": percentile_or_nan(corrected_rr_ms, 99),
            "corrected_hr_bpm_mean": mean_or_nan(corrected_hr_bpm),
            "corrected_hr_bpm_sd": std_or_nan(corrected_hr_bpm),
            "corrected_hr_bpm_p01": percentile_or_nan(corrected_hr_bpm, 1),
            "corrected_hr_bpm_p05": percentile_or_nan(corrected_hr_bpm, 5),
            "corrected_hr_bpm_p50": percentile_or_nan(corrected_hr_bpm, 50),
            "corrected_hr_bpm_p95": percentile_or_nan(corrected_hr_bpm, 95),
            "corrected_hr_bpm_p99": percentile_or_nan(corrected_hr_bpm, 99),
            "neurokit_warning_count": len(neurokit_warning_messages),
            "neurokit_warning_messages": " | ".join(neurokit_warning_unique[:10]),
            "issues": issues,
            "segments": segment_rows,
            "runtime_seconds": time.perf_counter() - start_clock,
            "beat_npz_path": str(beat_npz_path.relative_to(output_dir)),
            "qc_plot_path": str(plot_path.relative_to(output_dir)) if plot_path.exists() else "",
        }
    )
    atomic_write_json(result_path, json_safe(result))
    return result


def create_qc_plot(
    plt: Any,
    recording_uid: str,
    parsed: ParsedECG,
    cleaned_all: np.ndarray,
    segments: list[SegmentDef],
    original_global: np.ndarray,
    corrected_global: np.ndarray,
    original_rr_ms: np.ndarray,
    corrected_rr_ms: np.ndarray,
    corrected_hr_bpm: np.ndarray,
    video_start_timestamp: float,
    video_stop_timestamp: float,
    plot_window_seconds: float,
    max_plot_points: int,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = float(np.min(parsed.timestamps))
    t_rel = parsed.timestamps - t0

    raw_for_plot = parsed.samples.astype(float, copy=True)
    clean_for_plot = cleaned_all.astype(float, copy=True)
    if len(segments) > 1:
        for seg in segments[1:]:
            raw_for_plot[seg.start] = np.nan
            clean_for_plot[seg.start] = np.nan

    n = len(t_rel)
    step = max(1, int(math.ceil(n / max_plot_points)))
    idx = np.arange(0, n, step, dtype=int)

    fig, axes = plt.subplots(5, 1, figsize=(13, 16), constrained_layout=True)
    fig.suptitle(recording_uid)

    has_video_interval = (
        math.isfinite(video_start_timestamp)
        and math.isfinite(video_stop_timestamp)
        and video_stop_timestamp > video_start_timestamp
        and video_stop_timestamp >= float(np.min(parsed.timestamps))
        and video_start_timestamp <= float(np.max(parsed.timestamps))
    )
    if has_video_interval:
        video_x0_min = (video_start_timestamp - t0) / 60.0
        video_x1_min = (video_stop_timestamp - t0) / 60.0
    else:
        video_x0_min = math.nan
        video_x1_min = math.nan

    def shade_video_interval(ax: Any) -> None:
        if has_video_interval:
            ax.axvspan(video_x0_min, video_x1_min, alpha=0.12, label="Cooking interval")

    for seg in segments:
        ii=idx[(idx>=seg.start)&(idx<seg.stop)]
        axes[0].plot(t_rel[ii]/60.,raw_for_plot[ii],linewidth=.7,color='#254c65')
    axes[0].set_title("Full recording: raw ECG (downsampled for display only)")
    axes[0].set_xlabel("Minutes from first valid ECG timestamp")
    axes[0].set_ylabel("Native sample units")
    shade_video_interval(axes[0])
    if has_video_interval:
        axes[0].legend(loc="upper right")

    for seg in segments:
        ii=idx[(idx>=seg.start)&(idx<seg.stop)]
        axes[1].plot(t_rel[ii]/60.,clean_for_plot[ii],linewidth=.7,color='#254c65')
    axes[1].set_title("Full recording: cleaned ECG (downsampled for display only)")
    axes[1].set_xlabel("Minutes from first valid ECG timestamp")
    axes[1].set_ylabel("Filtered sample units")
    shade_video_interval(axes[1])

    processed_segments = [s for s in segments if np.any(np.isfinite(cleaned_all[s.start:s.stop]))]
    if processed_segments:
        longest = max(processed_segments, key=lambda s: s.duration_s)
        chosen = longest
        morphology_context = "longest processed segment"
        if has_video_interval:
            video_mid = (video_start_timestamp + video_stop_timestamp) / 2.0
            containing = [
                seg for seg in processed_segments
                if seg.start_timestamp <= video_mid <= seg.stop_timestamp
            ]
            if containing:
                chosen = containing[0]
                center_ts = video_mid
                morphology_context = "standard cooking interval"
            else:
                center_ts = (chosen.start_timestamp + chosen.stop_timestamp) / 2.0
        else:
            center_ts = (chosen.start_timestamp + chosen.stop_timestamp) / 2.0
        half = plot_window_seconds / 2.0
        if chosen.duration_s > plot_window_seconds:
            center_ts = min(
                max(center_ts, chosen.start_timestamp + half),
                chosen.stop_timestamp - half,
            )
        chosen_idx = np.arange(chosen.start, chosen.stop, dtype=int)
        chosen_ts = parsed.timestamps[chosen_idx]
        keep = (chosen_ts >= center_ts - half) & (chosen_ts <= center_ts + half)
        morph_idx = chosen_idx[keep]
        if morph_idx.size:
            x = parsed.timestamps[morph_idx] - center_ts
            axes[2].plot(x, cleaned_all[morph_idx], linewidth=0.9)
            orig_in = original_global[np.isin(original_global, morph_idx)]
            corr_in = corrected_global[np.isin(corrected_global, morph_idx)]
            if len(orig_in):
                axes[2].scatter(parsed.timestamps[orig_in] - center_ts, cleaned_all[orig_in], s=18, marker="x", label="Original R-peaks")
            if len(corr_in):
                axes[2].scatter(parsed.timestamps[corr_in] - center_ts, cleaned_all[corr_in], s=14, marker="o", facecolors="none", label="Corrected R-peaks")
            if len(orig_in) or len(corr_in):
                axes[2].legend(loc="upper right")
            axes[2].set_title(
                f"Morphology / R-peak review: {plot_window_seconds:g}-s window centered in {morphology_context}"
            )
            axes[2].set_xlabel("Seconds from window centre")
            axes[2].set_ylabel("Filtered sample units")
    else:
        axes[2].text(0.5, 0.5, "No processed ECG segment", transform=axes[2].transAxes, ha="center", va="center")
        axes[2].set_title("Morphology / R-peak review")

    if len(corrected_global):
        good_hr = np.isfinite(corrected_hr_bpm)
        axes[3].scatter(
            (parsed.timestamps[corrected_global][good_hr] - t0) / 60.0,
            corrected_hr_bpm[good_hr],
            s=6,
        )
    axes[3].set_title("Corrected instantaneous heart rate from beat-to-beat RR")
    axes[3].set_xlabel("Minutes from first valid ECG timestamp")
    axes[3].set_ylabel("BPM")
    shade_video_interval(axes[3])

    orig_rr = original_rr_ms[np.isfinite(original_rr_ms)]
    corr_rr = corrected_rr_ms[np.isfinite(corrected_rr_ms)]
    joined=np.r_[orig_rr,corr_rr]
    common_bins=np.histogram_bin_edges(joined,bins=50) if len(joined) else 50
    if len(orig_rr):
        axes[4].hist(orig_rr, bins=common_bins, alpha=0.5, label="Original RR")
    if len(corr_rr):
        axes[4].hist(corr_rr, bins=common_bins, alpha=0.5, label="Corrected RR")
    if len(orig_rr) or len(corr_rr):
        axes[4].legend()
    axes[4].set_title("RR interval distribution: original vs corrected")
    axes[4].set_xlabel("RR interval (ms)")
    axes[4].set_ylabel("Count")

    # Shared histogram edges support a valid original/corrected comparison.
    # These are descriptive interval distributions, not independent samples.
    fig.savefig(output_path, dpi=200)
    fig.savefig(output_path.with_suffix('.pdf'))
    fig.savefig(output_path.with_suffix('.svg'))
    plt.close(fig)


def flatten_recording_result(result: dict[str, Any]) -> dict[str, Any]:
    excluded = {"issues", "segments"}
    row: dict[str, Any] = {}
    for key, value in result.items():
        if key in excluded:
            continue
        if isinstance(value, (dict, list, tuple)):
            row[key] = json.dumps(json_safe(value), separators=(",", ":"))
        else:
            row[key] = value
    row["n_error_issues"] = sum(1 for x in result.get("issues", []) if x.get("severity") == "ERROR")
    row["n_warning_issues"] = sum(1 for x in result.get("issues", []) if x.get("severity") == "WARNING")
    row["n_info_issues"] = sum(1 for x in result.get("issues", []) if x.get("severity") == "INFO")
    row["issue_codes"] = ";".join(sorted({x.get("code", "") for x in result.get("issues", []) if x.get("code")}))
    return row


def build_consolidated_outputs(
    results: list[dict[str, Any]],
    manifest_candidates: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Any]:
    qc_rows = [flatten_recording_result(r) for r in results]
    qc_df = pd.DataFrame(qc_rows)
    if not qc_df.empty:
        sort_cols = [c for c in ["Session_ID", "Phase", "PID", "attempt_uid", "sensor_short"] if c in qc_df.columns]
        qc_df = qc_df.sort_values(sort_cols, kind="stable")
    qc_df.to_csv(output_dir / "02_recording_qc.csv", index=False)

    segment_rows: list[dict[str, Any]] = []
    issue_rows: list[dict[str, Any]] = []
    for r in results:
        context = {
            "recording_uid": r.get("recording_uid", ""),
            "Session_ID": r.get("Session_ID", ""),
            "PID": r.get("PID", ""),
            "Phase": r.get("Phase", ""),
            "attempt_uid": r.get("attempt_uid", ""),
            "sensor_short": r.get("sensor_short", ""),
        }
        for seg in r.get("segments", []):
            row = dict(context)
            row.update(seg)
            segment_rows.append(row)
        for issue in r.get("issues", []):
            row = dict(context)
            row.update(issue)
            issue_rows.append(row)

    pd.DataFrame(segment_rows).to_csv(output_dir / "02_segment_qc.csv", index=False)
    pd.DataFrame(issue_rows).to_csv(output_dir / "02_processing_issues.csv", index=False)

    # Participant-phase candidate table: retain every restart alternative; make no selection.
    candidate_table = manifest_candidates.copy()
    qc_join_cols = [
        "recording_uid",
        "processing_status",
        "n_rpeaks_original",
        "n_rpeaks_corrected",
        "original_peak_change_fraction",
        "corrected_rr_count",
        "corrected_hr_bpm_p50",
        "valid_timestamp_span_s",
        "timestamp_integrity_status",
        "n_overlapping_segment_pairs",
        "max_segment_pair_overlap_s",
        "segment_overlap_ambiguity_threshold_s",
        "segment_time_overlap_excess_s",
        "processed_unique_time_coverage_s",
        "cooking_processed_overlap_s",
        "n_gap_events",
        "max_gap_s",
        "n_error_issues",
        "n_warning_issues",
        "issue_codes",
        "beat_npz_path",
        "qc_plot_path",
    ]
    available_qc_cols = [c for c in qc_join_cols if c in qc_df.columns]
    if "recording_uid" in qc_df.columns:
        candidate_table = candidate_table.merge(qc_df[available_qc_cols], on="recording_uid", how="left", validate="one_to_one")
    candidate_table.to_csv(output_dir / "02_participant_phase_candidates.csv", index=False)

    write_aggregate_peak_tables(results, output_dir)

    status_counts = Counter(str(r.get("processing_status", "UNKNOWN")) for r in results)
    issue_severity_counts = Counter()
    issue_code_counts = Counter()
    for r in results:
        for issue in r.get("issues", []):
            issue_severity_counts[str(issue.get("severity", ""))] += 1
            issue_code_counts[str(issue.get("code", ""))] += 1

    return {
        "recording_status_counts": dict(sorted(status_counts.items())),
        "issue_severity_counts": dict(sorted(issue_severity_counts.items())),
        "top_issue_codes": dict(issue_code_counts.most_common(30)),
        "n_recordings": len(results),
        "n_processing_error_recordings": sum(
            1 for r in results if any(x.get("severity") == "ERROR" for x in r.get("issues", []))
        ),
    }


def write_aggregate_peak_tables(results: list[dict[str, Any]], output_dir: Path) -> None:
    corrected_path = output_dir / "02_corrected_beats.csv.gz"
    original_path = output_dir / "02_original_rpeaks.csv.gz"

    corrected_header = [
        "recording_uid", "Session_ID", "PID", "Phase", "Condition", "sensor_short", "attempt_uid",
        "segment_id", "beat_index_recording", "rpeak_valid_index", "rpeak_source_row", "rpeak_timestamp",
        "rr_ms", "instantaneous_hr_bpm",
    ]
    original_header = [
        "recording_uid", "Session_ID", "PID", "Phase", "Condition", "sensor_short", "attempt_uid",
        "segment_id", "peak_index_recording", "rpeak_valid_index", "rpeak_source_row", "rpeak_timestamp",
        "rr_ms", "instantaneous_hr_bpm",
    ]

    with gzip.open(corrected_path, "wt", encoding="utf-8", newline="") as fc, gzip.open(
        original_path, "wt", encoding="utf-8", newline=""
    ) as fo:
        wc = csv.writer(fc)
        wo = csv.writer(fo)
        wc.writerow(corrected_header)
        wo.writerow(original_header)

        for r in results:
            rel_npz = str(r.get("beat_npz_path", ""))
            if not rel_npz:
                continue
            npz_path = output_dir / rel_npz
            if not npz_path.exists():
                continue
            data = np.load(npz_path, allow_pickle=False)
            common = [
                r.get("recording_uid", ""),
                r.get("Session_ID", ""),
                r.get("PID", ""),
                r.get("Phase", ""),
                r.get("Condition", ""),
                r.get("sensor_short", ""),
                r.get("attempt_uid", ""),
            ]

            n_corr = len(data["corrected_rpeak_timestamp"])
            for i in range(n_corr):
                wc.writerow(
                    common
                    + [
                        int(data["corrected_rpeak_segment_id"][i]),
                        i + 1,
                        int(data["corrected_rpeak_valid_index"][i]),
                        int(data["corrected_rpeak_source_row"][i]),
                        repr(float(data["corrected_rpeak_timestamp"][i])),
                        "" if not np.isfinite(data["corrected_rr_ms"][i]) else repr(float(data["corrected_rr_ms"][i])),
                        "" if not np.isfinite(data["corrected_hr_bpm"][i]) else repr(float(data["corrected_hr_bpm"][i])),
                    ]
                )

            n_orig = len(data["original_rpeak_timestamp"])
            for i in range(n_orig):
                wo.writerow(
                    common
                    + [
                        int(data["original_rpeak_segment_id"][i]),
                        i + 1,
                        int(data["original_rpeak_valid_index"][i]),
                        int(data["original_rpeak_source_row"][i]),
                        repr(float(data["original_rpeak_timestamp"][i])),
                        "" if not np.isfinite(data["original_rr_ms"][i]) else repr(float(data["original_rr_ms"][i])),
                        "" if not np.isfinite(data["original_hr_bpm"][i]) else repr(float(data["original_hr_bpm"][i])),
                    ]
                )


def validate_script01_inputs(manifest_path: Path, summary_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Script 01 manifest not found: {manifest_path}")
    if not summary_path.exists():
        raise FileNotFoundError(f"Script 01 summary not found: {summary_path}")

    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)

    script01_version = str(summary.get("script_version", ""))
    if script01_version != EXPECTED_SCRIPT01_VERSION:
        raise RuntimeError(
            f"Expected Script 01 version {EXPECTED_SCRIPT01_VERSION}, but summary reports {script01_version!r}. "
            "Use the frozen v1.4.1 reconciliation outputs."
        )
    if int(summary.get("issues", {}).get("errors", -1)) != 0:
        raise RuntimeError("Script 01 summary contains blocking errors. Script 02 will not proceed.")

    manifest = pd.read_csv(manifest_path, keep_default_na=False)
    missing = sorted(REQUIRED_MANIFEST_COLUMNS - set(manifest.columns))
    if missing:
        raise RuntimeError(f"Script 01 manifest is missing required columns: {missing}")

    candidate_mask = manifest["Script02_process_candidate"].map(as_bool)
    candidates = manifest.loc[candidate_mask].copy()
    if candidates.empty:
        raise RuntimeError("Script 01 manifest contains no Script02_process_candidate rows.")

    # The frozen v1.4.1 full-data audit reported 185 candidates. We warn rather
    # than hard-fail if this changes, because a legitimate future raw-data update
    # may alter the count while retaining the same schema.
    expected_count = safe_int(summary.get("script02", {}).get("candidate_sensor_recordings"))
    if expected_count is not None and len(candidates) != expected_count:
        logging.warning(
            "Manifest candidate count (%d) differs from Script 01 summary (%d).",
            len(candidates), expected_count,
        )

    # Add stable candidate count per participant-phase so restart alternatives remain explicit.
    group_cols = ["mapped_session_id", "mapped_pid", "mapped_phase"]
    candidates["participant_phase_candidate_count"] = candidates.groupby(group_cols)["attempt_uid"].transform("size")
    candidates["recording_uid"] = candidates.apply(build_recording_uid, axis=1)
    if candidates["recording_uid"].duplicated().any():
        dupes = candidates.loc[candidates["recording_uid"].duplicated(keep=False), "recording_uid"].tolist()
        raise RuntimeError(f"Duplicate recording_uid values generated: {dupes[:10]}")

    sort_cols = ["mapped_session_id", "mapped_phase", "mapped_pid", "attempt_uid", "sensor_short"]
    candidates = candidates.sort_values(sort_cols, kind="stable").reset_index(drop=True)
    return candidates, summary


def prepare_output_dir(output_dir: Path, overwrite: bool, resume: bool) -> None:
    if overwrite and resume:
        raise ValueError("--overwrite and --resume cannot be used together.")
    if output_dir.exists() and any(output_dir.iterdir()):
        if overwrite:
            resolved = output_dir.resolve()
            if resolved == Path("/") or len(resolved.parts) < 4:
                raise RuntimeError(f"Refusing to recursively delete suspicious output path: {resolved}")
            shutil.rmtree(output_dir)
        elif not resume:
            raise FileExistsError(
                f"Output directory already exists and is not empty: {output_dir}\n"
                "Use --resume to continue an interrupted run, or --overwrite to deliberately replace Script 02 outputs."
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "records").mkdir(exist_ok=True)
    (output_dir / "qc_plots").mkdir(exist_ok=True)


def write_input_snapshot(candidates: pd.DataFrame, output_dir: Path) -> None:
    snapshot = candidates.copy()
    snapshot.to_csv(output_dir / "02_input_manifest_snapshot.csv", index=False)


def write_run_summary(
    output_dir: Path,
    args: argparse.Namespace,
    manifest_path: Path,
    script01_summary_path: Path,
    script01_annotations_path: Path,
    script01_summary: dict[str, Any],
    candidates: pd.DataFrame,
    aggregate: dict[str, Any],
    runtime_s: float,
    nk_version: str,
) -> None:
    payload = {
        "script": SCRIPT_NAME,
        "script_version": SCRIPT_VERSION,
        "generated_utc": utc_now_iso(),
        "runtime_seconds": runtime_s,
        "paths": {
            "project_root": str(args.project_root_resolved),
            "raw_root": str(args.raw_root_resolved),
            "script01_manifest": str(manifest_path),
            "script01_summary": str(script01_summary_path),
            "script01_annotations": str(script01_annotations_path),
            "output_dir": str(output_dir),
        },
        "provenance": {
            "script_sha256": sha256_file(SCRIPT_PATH),
            "script01_manifest_sha256": sha256_file(manifest_path),
            "script01_summary_sha256": sha256_file(script01_summary_path),
            "script01_annotations_sha256": (
                sha256_file(script01_annotations_path) if script01_annotations_path.exists() else None
            ),
            "script01_version": script01_summary.get("script_version"),
            "script01_script_sha256": script01_summary.get("provenance", {}).get("script_sha256"),
            "runsheet_sha256": script01_summary.get("provenance", {}).get("runsheet_sha256"),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "neurokit2": nk_version,
        },
        "configuration": {
            "sampling_rate_policy": args.sampling_rate_policy,
            "detector_rate_is_physical_calibration": False,
            "powerline_hz": args.powerline_hz,
            "gap_factor": args.gap_factor,
            "gap_min_seconds": args.gap_min_seconds,
            "min_segment_seconds": args.min_segment_seconds,
            "plot_window_seconds": args.plot_window_seconds,
            "max_plot_points": args.max_plot_points,
            "plots_enabled": not args.no_plots,
            "max_recordings": args.max_recordings,
            "resume": args.resume,
            "session_id_filter": args.session_id,
            "pid_filter": args.pid,
            "phase_filter": args.phase,
            "only_restarts": args.only_restarts,
        },
        "input": {
            "candidate_sensor_recordings_this_run": len(candidates),
            "candidate_participant_phases_this_run": int(
                candidates[["mapped_session_id", "mapped_pid", "mapped_phase"]].drop_duplicates().shape[0]
            ),
            "primary_synchrony_candidate_rows_from_script01": int(
                candidates["PrimarySynchrony_candidate"].map(as_bool).sum()
            ),
            "standard_cooking_interval_rows_this_run": int(
                (candidates["cooking_interval_status"] == "OK_SINGLE_VIDEO_INTERVAL").sum()
            ),
        },
        "results": aggregate,
    }
    atomic_write_json(output_dir / "02_run_summary.json", json_safe(payload))

    lines = [
        "KITCO3 Script 02 - ECG Processing and Objective QC",
        f"Version: {SCRIPT_VERSION}",
        f"Generated UTC: {payload['generated_utc']}",
        f"Runtime: {runtime_s:.1f} s",
        "",
        f"Input candidate sensor recordings: {payload['input']['candidate_sensor_recordings_this_run']}",
        f"Input candidate participant-phases: {payload['input']['candidate_participant_phases_this_run']}",
        f"Primary-synchrony candidate rows from Script 01: {payload['input']['primary_synchrony_candidate_rows_from_script01']}",
        f"Rows with one standard cooking interval: {payload['input'].get('standard_cooking_interval_rows_this_run', 0)}",
        "",
        "Recording processing status counts:",
    ]
    for key, value in aggregate.get("recording_status_counts", {}).items():
        lines.append(f"  {key}: {value}")
    lines += [
        "",
        "Issue severity counts:",
    ]
    for key, value in aggregate.get("issue_severity_counts", {}).items():
        lines.append(f"  {key}: {value}")
    lines += [
        "",
        f"Recordings with technical processing ERRORs: {aggregate.get('n_processing_error_recordings', 0)}",
        "",
        "No final scientific ECG exclusion thresholds were applied.",
        "Restart alternatives were preserved; no participant-phase recording was selected here.",
        "Cooking intervals, when uniquely defined by Script 01 annotations, were added as QC context only; ECG was not cropped here.",
    ]
    (output_dir / "02_run_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="KITCO3 Script 02: raw ECG processing and objective QC.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=DEFAULT_PROJECT_ROOT,
        help=f"Project root. Default is inferred as parent of scripts/: {DEFAULT_PROJECT_ROOT}",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=DEFAULT_RAW_ROOT,
        help="Raw participant-folder root, relative to project root unless absolute. Default: data",
    )
    parser.add_argument(
        "--script01-manifest",
        type=Path,
        default=DEFAULT_SCRIPT01_MANIFEST,
        help="Frozen Script 01 sensor manifest. Default: outputs/01_reconciliation/01_sensor_recordings.csv",
    )
    parser.add_argument(
        "--script01-summary",
        type=Path,
        default=DEFAULT_SCRIPT01_SUMMARY,
        help="Frozen Script 01 run summary JSON. Default: outputs/01_reconciliation/01_run_summary.json",
    )
    parser.add_argument(
        "--script01-annotations",
        type=Path,
        default=DEFAULT_SCRIPT01_ANNOTATIONS,
        help=(
            "Frozen Script 01 annotation-event table used only for cooking-interval QC context. "
            "Default: outputs/01_reconciliation/01_annotation_events.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Script 02 output directory. Default: outputs/02_ecg_processing",
    )
    parser.add_argument("--powerline-hz", type=float, default=DEFAULT_POWERLINE_HZ)
    parser.add_argument('--sampling-rate-policy', choices=['legacy_replay','header_nominal'],
                        default='legacy_replay', help='Explicit design rate: recorded per-record setting, or nominal header rate requiring renewed peak compatibility checks. Neither calibrates physical time.')
    parser.add_argument("--gap-factor", type=float, default=DEFAULT_GAP_FACTOR)
    parser.add_argument("--gap-min-seconds", type=float, default=DEFAULT_GAP_MIN_SECONDS)
    parser.add_argument("--min-segment-seconds", type=float, default=DEFAULT_MIN_SEGMENT_SECONDS)
    parser.add_argument("--plot-window-seconds", type=float, default=DEFAULT_PLOT_WINDOW_SECONDS)
    parser.add_argument("--max-plot-points", type=int, default=DEFAULT_MAX_PLOT_POINTS)
    parser.add_argument(
        "--session-id",
        action="append",
        default=[],
        help="Development/audit filter. May be repeated, e.g. --session-id S016.",
    )
    parser.add_argument(
        "--pid",
        action="append",
        type=int,
        default=[],
        help="Development/audit filter for participant ID. May be repeated.",
    )
    parser.add_argument(
        "--phase",
        action="append",
        type=int,
        choices=[1, 2],
        default=[],
        help="Development/audit filter for phase. May be repeated.",
    )
    parser.add_argument(
        "--only-restarts",
        action="store_true",
        help="Development/audit filter: process only participant-phases with >1 Script 02 candidate recording.",
    )
    parser.add_argument(
        "--max-recordings",
        type=int,
        default=None,
        help="Development/testing only: after other filters, process only the first N sorted candidates.",
    )
    parser.add_argument("--no-plots", action="store_true", help="Skip per-recording QC PNG generation.")
    parser.add_argument("--resume", action="store_true", help="Resume from completed per-recording JSON/NPZ outputs.")
    parser.add_argument("--overwrite", action="store_true", help="Delete and recreate the Script 02 output directory.")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate Script 01 inputs and raw ECG paths, then exit without signal processing.",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.resume or args.overwrite:
        print('v1.1.0 requires a new output folder; resume/overwrite disabled to avoid mixing policies.',file=sys.stderr)
        return 2
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s | %(message)s")

    project_root = Path(args.project_root).expanduser().resolve()
    raw_root = resolve_path(project_root, args.raw_root)
    manifest_path = resolve_path(project_root, args.script01_manifest)
    script01_summary_path = resolve_path(project_root, args.script01_summary)
    script01_annotations_path = resolve_path(project_root, args.script01_annotations)
    output_dir = resolve_path(project_root, args.output_dir)
    args.project_root_resolved = project_root
    args.raw_root_resolved = raw_root

    logging.info("Project root: %s", project_root)
    logging.info("Raw root: %s", raw_root)
    logging.info("Script 01 manifest: %s", manifest_path)
    logging.info("Script 01 summary: %s", script01_summary_path)
    logging.info("Script 01 annotations: %s", script01_annotations_path)
    logging.info("Output directory: %s", output_dir)

    try:
        candidates, script01_summary = validate_script01_inputs(manifest_path, script01_summary_path)
    except Exception as exc:
        logging.error("Input validation failed: %s", exc)
        return 2

    try:
        candidates = attach_video_annotation_context(candidates, script01_annotations_path)
    except Exception as exc:
        logging.error("Could not load Script 01 annotation context: %s", exc)
        return 2
    if not script01_annotations_path.exists():
        logging.warning("Script 01 annotation-event table not found; QC plots will not show cooking intervals.")

    candidates = filter_candidates(candidates, args)
    if candidates.empty:
        logging.error("No ECG candidates remain after the requested development/audit filters.")
        return 2

    if args.max_recordings is not None:
        if args.max_recordings < 1:
            logging.error("--max-recordings must be >= 1")
            return 2
        candidates = candidates.head(args.max_recordings).copy()

    logging.info(
        "Validated Script 01 v%s. ECG candidates selected for this run: %d.",
        script01_summary.get("script_version"),
        len(candidates),
    )

    missing_paths = []
    for _, row in candidates.iterrows():
        p = raw_root / str(row["ECG_path"])
        if not p.exists():
            missing_paths.append(str(p))
    if missing_paths:
        logging.error("%d candidate ECG path(s) do not exist. First examples: %s", len(missing_paths), missing_paths[:5])
        return 2

    if args.validate_only:
        logging.info("Validation only: all %d selected ECG paths exist. No signal processing performed.", len(candidates))
        return 0

    try:
        prepare_output_dir(output_dir, overwrite=args.overwrite, resume=args.resume)
    except Exception as exc:
        logging.error("Could not prepare output directory: %s", exc)
        return 2

    write_input_snapshot(candidates, output_dir)

    try:
        nk, plt = load_processing_dependencies()
    except ProcessingDependencyError as exc:
        logging.error("%s", exc)
        return 2

    logging.info("NeuroKit2 version: %s", getattr(nk, "__version__", "unknown"))
    logging.info(
        "Processing method: NeuroKit ECG clean + NeuroKit R-peaks + iterative Kubios correction; powerline %.1f Hz.",
        args.powerline_hz,
    )
    logging.info("No final ECG exclusion thresholds will be applied in Script 02.")

    run_start = time.perf_counter()
    results: list[dict[str, Any]] = []
    n = len(candidates)

    for i, (_, row) in enumerate(candidates.iterrows(), start=1):
        recording_uid = str(row["recording_uid"])
        if i == 1 or i % 10 == 0 or i == n:
            logging.info("ECG processing progress: %d/%d", i, n)
        logging.debug("Processing %s", recording_uid)
        try:
            result = process_one_recording(row, raw_root, output_dir, nk, plt, args)
        except Exception as exc:  # defensive: one recording must not terminate the batch
            logging.exception("Unexpected failure processing %s", recording_uid)
            result = {
                "recording_uid": recording_uid,
                "attempt_uid": str(row.get("attempt_uid", "")),
                "Session_ID": str(row.get("mapped_session_id", "")),
                "PID": safe_int(row.get("mapped_pid")),
                "Phase": safe_int(row.get("mapped_phase")),
                "Condition": str(row.get("mapped_condition", "")),
                "sensor_short": str(row.get("sensor_short", "")),
                "processing_status": "ERROR_UNEXPECTED_EXCEPTION",
                "issues": [
                    {
                        "severity": "ERROR",
                        "code": "UNEXPECTED_RECORDING_EXCEPTION",
                        "detail": f"{type(exc).__name__}: {exc}",
                    }
                ],
                "segments": [],
                "beat_npz_path": "",
                "qc_plot_path": "",
                "runtime_seconds": 0.0,
            }
        results.append(result)

    aggregate = build_consolidated_outputs(results, candidates, output_dir)
    runtime_s = time.perf_counter() - run_start
    write_run_summary(
        output_dir=output_dir,
        args=args,
        manifest_path=manifest_path,
        script01_summary_path=script01_summary_path,
        script01_annotations_path=script01_annotations_path,
        script01_summary=script01_summary,
        candidates=candidates,
        aggregate=aggregate,
        runtime_s=runtime_s,
        nk_version=str(getattr(nk, "__version__", "unknown")),
    )

    logging.info("Script 02 outputs written to: %s", output_dir)
    logging.info("Recording status counts: %s", aggregate["recording_status_counts"])
    logging.info("Recordings with technical processing ERRORs: %d", aggregate["n_processing_error_recordings"])
    logging.info("No final scientific QC exclusions were applied.")

    return 2 if aggregate["n_processing_error_recordings"] > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
