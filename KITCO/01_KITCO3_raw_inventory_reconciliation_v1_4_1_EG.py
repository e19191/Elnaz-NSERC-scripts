#!/usr/bin/env python3
"""
KITCO3 Pipeline, Script 01
Raw dataset inventory and authoritative runsheet reconciliation

Purpose
-------
Create the production-grade, read-only inventory used by all downstream KITCO3
physiology scripts. The script discovers every Data Collector recording attempt,
inventories sensor streams and annotations without loading complete signals, maps
recording attempts to the paper-derived session structure, identifies duplicate copies,
and reconciles each canonical attempt to the final cleaned runsheet.

Authoritative metadata
----------------------
``Runsheet_final_EG.xlsx`` / ``Runsheet_Clean`` is the analysis runsheet derived from
the physical paper runsheets. It is authoritative for PID, sensor, phase, condition,
partner, date, and Session_ID. Raw folder names are retained as audit context only and
never override the runsheet.

Known study conventions encoded here
------------------------------------
- Data Collector timestamped attempt-folder names are UTC; study dates are matched in
  ``America/Vancouver`` local time.
- Top-level ``Participants xx-yy`` batch folders identify the participant set copied
  for a session. Script 01 uses this only as session-location evidence and validates it
  against the authoritative runsheet PID set and study date before accepting a match.
- ``KITCOdate`` versus ``KITCOdate 2`` may represent two back-to-back sessions on the
  same day. The suffix is retained for audit but is not used to infer phase.
- ``(JM)`` and ``(MCL)`` identify the acquisition iPad/session-room stream. They are
  recorded for audit and are not interpreted as participant, phase, or condition labels.
- ``VideoAnnotationStart`` and ``VideoAnnotationStop`` are the intended collaborative
  cooking interval unless a runsheet exception says otherwise. Script 01 inventories
  and validates these events but does not yet define final analysis epochs.
- Known acquisition failures documented in the runsheet (for example sensor did not
  connect, no sensor data, no Movesense, participant no-show) are distinguished from
  unexpected missing raw recordings.

Safety and conservatism
-----------------------
- Raw files are read only. Nothing is renamed, moved, trimmed, deleted, or overwritten.
- Ambiguous session/phase mappings are never guessed.
- Exact duplicate-copy fingerprints are detected before phase mapping and never
  double-counted. Restarts remain separate candidate recordings rather than being
  silently selected or discarded.
- Full-signal CSVs are not loaded into memory. Inventory uses directory metadata and
  only the beginning/end of stream files.
- In ``--scope-mode all`` an unexpected missing recording for a participant-phase that
  should have raw sensor data is a blocking error. Documented acquisition failures are
  retained as explicit nonblocking exceptions.

Primary outputs
---------------
01_run_summary.txt / 01_run_summary.json
    Concise human-readable and machine-readable run summaries.
01_reconciliation_issues.csv
    Review queue containing all blocking errors, warnings, and documented exceptions.
01_session_phase_summary.csv
    One row per expected session-phase (54 rows in the final runsheet).
01_participant_phase_reconciliation.csv
    One row per expected participant-phase (216 rows in the final runsheet).
01_sensor_recordings.csv
    Canonical one-row-per-sensor-recording manifest for Script 02, including explicit
    paths to ECG/Acc/HR-related streams when present.

Detailed audit outputs
----------------------
01_recording_attempts.csv
01_time_blocks.csv                 # diagnostic only, never used to infer phase
01_phase_groups.csv
01_duplicate_attempts.csv
01_annotation_events.csv
01_file_inventory.csv

Current external-drive layout
-----------------------------
/Volumes/One Touch/KITCO3_drive/
    scripts/
        01_script_raw_inventory_and_reconciliation_EG_final.py
    data/
        Participants 01-04/
        Participants 05-08/
        ...
    Runsheet_final_EG.xlsx
    outputs/                  # created by this script

Recommended full-study run
--------------------------
With the script stored in ``KITCO3_drive/scripts/``, all production paths resolve
from the script location. Therefore the recommended command is simply:

cd "/Volumes/One Touch/KITCO3_drive"
python3 scripts/01_script_raw_inventory_and_reconciliation_EG_final.py

Equivalent explicit command:

python3 scripts/01_script_raw_inventory_and_reconciliation_EG_final.py \
  --project-root "/Volumes/One Touch/KITCO3_drive" \
  --raw-root data \
  --runsheet Runsheet_final_EG.xlsx \
  --runsheet-sheet Runsheet_Clean \
  --output-dir outputs/01_reconciliation \
  --scope-mode all

Use ``--allow-unresolved`` only for exploratory inspection. Production runs should
remain strict. Script 01 does not choose a preferred restart. Multiple nonduplicate
recordings remain explicit candidates for downstream ECG processing and QC.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import platform
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

try:
    import pandas as pd
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing required package 'pandas'. Install dependencies once with:\n"
        "  python3 -m pip install pandas openpyxl"
    ) from exc

try:
    import openpyxl  # noqa: F401  # required by pandas for .xlsx runsheets
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing required package 'openpyxl'. Install dependencies once with:\n"
        "  python3 -m pip install pandas openpyxl"
    ) from exc


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

SCRIPT_VERSION = "1.4.1"
DEFAULT_RUNSHEET_SHEET = "Runsheet_Clean"
VALID_PHASES = {1, 2}
VALID_CONDITIONS = {"NA", "NB", "EA", "EB"}
EXPECTED_PARTICIPANT_PHASE_ROWS = 216
EXPECTED_PID_SET = {f"{i:03d}" for i in range(1, 109)}
EXPECTED_SESSION_SET = {f"S{i:03d}" for i in range(1, 28)}
REQUIRED_RUNSHEET_COLUMNS = {
    "PID",
    "Sensor",
    "Phase",
    "Condition",
    "Planned_Partner",
    "Analysis_Partner",
    "Date",
    "Notes",
    "Session_ID",
}

# Data Collector annotation labels observed in KITCO3.
RECORDING_START_TYPES = {"RecordingStart"}
RECORDING_STOP_TYPES = {"RecordingStop"}
VIDEO_START_TYPES = {"VideoAnnotationStart", "VideoStart"}
VIDEO_STOP_TYPES = {"VideoAnnotationStop", "VideoStop"}
PAUSE_TYPES = {"RecordingPause", "Pause"}
RESUME_TYPES = {"RecordingResume", "Resume"}
VIDEO_EXTENSIONS = {".mov", ".mp4", ".m4v", ".avi"}
DEFAULT_STUDY_TIMEZONE = "America/Vancouver"
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_RAW_ROOT = Path("data")
DEFAULT_RUNSHEET_PATH = Path("Runsheet_final_EG.xlsx")
DEFAULT_OUTPUT_DIR = Path("outputs/01_reconciliation")

# A recording attempt folder currently follows this format. The scanner does not
# depend exclusively on the name, but uses it when present for a reliable raw date.
ATTEMPT_FOLDER_RE = re.compile(
    r"^(?P<date>\d{4}_\d{2}_\d{2})-(?P<time>\d{2}_\d{2}_\d{2})$"
)
SESSION_ID_RE = re.compile(r"^S\d{3,}$")
SENSOR_SWITCH_RE = re.compile(
    r"(?:\b(?:chang(?:e|ed|ing)|switch(?:ed|ing)?)\b.*\bsensor\b|"
    r"\bsensor\b.*\b(?:chang(?:e|ed|ing)|switch(?:ed|ing)?)\b)",
    flags=re.IGNORECASE,
)

# Runsheet language used only to classify whether missing raw sensor data are already
# documented. These patterns do NOT make signal-quality decisions.
PARTICIPANT_ABSENCE_RE = re.compile(
    r"\b(?:no[ -]?show|did not show up|didn['’]t show up|absent)\b",
    flags=re.IGNORECASE,
)
EXPLICIT_NO_SENSOR_RE = re.compile(
    r"(?:\bno sensor data\b|\bno sensors?\b|\bno movesense\b|"
    r"\bsensor did not connect\b)",
    flags=re.IGNORECASE,
)
EXACT_DID_NOT_CONNECT_RE = re.compile(r"^\s*did not connect[.!]?\s*$", re.IGNORECASE)
IPAD_LABEL_RE = re.compile(r"\((?P<label>JM|MCL)\)\s*$", re.IGNORECASE)
BACK_TO_BACK_SUFFIX_RE = re.compile(r"(?:^|\s)2\s*$", re.IGNORECASE)
KNOWN_STREAM_TYPES = {
    "ECG",
    "Acc",
    "AccFeatures",
    "Gyro",
    "GyroFeatures",
    "HeartRate",
    "HR",
    "KaasaHR",
    "MotionIntensity",
    "Lactate",
}
PARTICIPANT_BATCH_RE = re.compile(r"^Participants\s+(?P<start>\d+)\s*-\s*(?P<stop>\d+)$", re.IGNORECASE)
MIN_PHASE_SPLIT_GAP_SECONDS = 10 * 60
OUTPUT_FILENAMES = (
    "01_run_summary.txt",
    "01_run_summary.json",
    "01_reconciliation_issues.csv",
    "01_session_phase_summary.csv",
    "01_participant_phase_reconciliation.csv",
    "01_sensor_recordings.csv",
    "01_recording_attempts.csv",
    "01_time_blocks.csv",
    "01_phase_groups.csv",
    "01_duplicate_attempts.csv",
    "01_annotation_events.csv",
    "01_file_inventory.csv",
)


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------


def is_missing(value: Any) -> bool:
    """Return True for None/NaN/empty-string style missing values."""
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    return isinstance(value, str) and value.strip() == ""


def normalize_pid(value: Any) -> Optional[str]:
    """Normalize participant IDs to zero-padded three-character strings."""
    if is_missing(value):
        return None
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    if not text.isdigit():
        raise ValueError(f"Invalid PID value: {value!r}")
    return text.zfill(3)


def normalize_sensor(value: Any) -> Optional[str]:
    """Normalize runsheet sensor IDs as identifier strings, preserving four digits."""
    if is_missing(value):
        return None
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    if not text.isdigit():
        raise ValueError(f"Invalid sensor ID: {value!r}")
    if len(text) > 4:
        # The runsheet is expected to use short sensor IDs. Treat a longer value as
        # a formatting problem rather than silently truncating it.
        raise ValueError(
            f"Runsheet sensor ID {value!r} has more than four digits. "
            "Expected the short sensor code used by KITCO3."
        )
    return text.zfill(4)


def raw_sensor_to_short(full_sensor_id: str) -> str:
    """Convert a numeric Data Collector sensor directory ID to its final 4 digits."""
    text = str(full_sensor_id).strip()
    if not text.isdigit() or len(text) < 4:
        raise ValueError(f"Invalid raw sensor directory name: {full_sensor_id!r}")
    return text[-4:]


def normalize_partner(value: Any) -> Optional[str]:
    """Normalize partner IDs using the same representation as PID."""
    return normalize_pid(value)


def normalize_session_id(value: Any) -> str:
    if is_missing(value):
        raise ValueError("Missing Session_ID")
    text = str(value).strip().upper()
    if not SESSION_ID_RE.fullmatch(text):
        raise ValueError(f"Invalid Session_ID: {value!r}")
    return text


def normalize_phase(value: Any) -> int:
    """Normalize Phase while rejecting non-integer values such as 1.5."""
    if is_missing(value):
        raise ValueError("Missing Phase")
    if isinstance(value, bool):
        raise ValueError(f"Invalid Phase: {value!r}")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid Phase: {value!r}") from exc
    if not math.isfinite(numeric) or not numeric.is_integer():
        raise ValueError(f"Invalid Phase {value!r}; expected integer 1 or 2")
    phase = int(numeric)
    if phase not in VALID_PHASES:
        raise ValueError(f"Invalid Phase {phase}; expected 1 or 2")
    return phase


def normalize_condition(value: Any) -> str:
    if is_missing(value):
        raise ValueError("Missing Condition")
    text = str(value).strip().upper()
    if text not in VALID_CONDITIONS:
        raise ValueError(
            f"Invalid Condition {value!r}; expected one of {sorted(VALID_CONDITIONS)}"
        )
    return text


def normalize_date(value: Any) -> date:
    """Normalize Excel/pandas date values to datetime.date."""
    if is_missing(value):
        raise ValueError("Missing Date")
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"Invalid Date value: {value!r}")
    return parsed.date()


def note_documents_sensor_switch(*values: Any) -> bool:
    """Return True only when source notes explicitly document a sensor change/switch."""
    text = " ".join(str(v) for v in values if not is_missing(v))
    return bool(SENSOR_SWITCH_RE.search(text))


def combined_note_text(*values: Any) -> str:
    """Combine nonmissing source-note fields without changing their content."""
    return " ".join(str(v).strip() for v in values if not is_missing(v)).strip()


def note_documents_participant_absence(*values: Any) -> bool:
    """Detect explicit no-show/absence documentation in source notes."""
    return bool(PARTICIPANT_ABSENCE_RE.search(combined_note_text(*values)))


def note_documents_no_sensor_data(*values: Any) -> bool:
    """Detect explicit statements that a phase has no usable Movesense acquisition."""
    text = combined_note_text(*values)
    if not text:
        return False
    return bool(EXPLICIT_NO_SENSOR_RE.search(text) or EXACT_DID_NOT_CONNECT_RE.fullmatch(text))


def classify_raw_recording_expectation(row: pd.Series) -> str:
    """Classify expected raw-data availability from the authoritative runsheet only."""
    if is_missing(row.get("Sensor")):
        return "NO_SENSOR_ASSIGNED"
    if bool(row.get("Participant_Documented_Absent", False)):
        return "DOCUMENTED_PARTICIPANT_ABSENT"
    if note_documents_no_sensor_data(row.get("Notes"), row.get("Sensor_Note")):
        return "DOCUMENTED_SENSOR_UNAVAILABLE"
    return "EXPECTED_RECORDING"


def parse_allowed_sensors(value: Any) -> frozenset[str]:
    """Parse the internal semicolon-delimited Allowed_Sensors field."""
    if is_missing(value):
        return frozenset()
    return frozenset(x for x in str(value).split(";") if x)


def safe_json(value: Any) -> Any:
    """Convert common pandas/numpy/datetime values to JSON-safe Python values."""
    if is_missing(value):
        return None
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def compact_join(values: Iterable[Any], sep: str = ";") -> str:
    """Stable, duplicate-free join of nonmissing values."""
    cleaned = sorted({str(v) for v in values if not is_missing(v)})
    return sep.join(cleaned)


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    """Return a SHA-256 hash without loading the file into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def extract_raw_path_context(attempt_dir: Path, raw_root: Path) -> dict[str, Any]:
    """Extract folder context without assuming that folder names encode outcomes.

    The participant-batch folder is used later only as acquisition metadata to link a
    raw recording to the corresponding four-person runsheet session.  Nested collector
    folders are preserved because some dates contain structures such as
    ``KITCO FEB 13 2 (JM)/KITCO3FEB13no2a/<attempt>``.
    """
    rel = attempt_dir.resolve().relative_to(raw_root.resolve())
    parts = rel.parts
    participant_batch = parts[0] if len(parts) >= 2 else None
    collection_parts = list(parts[1:-1]) if len(parts) >= 3 else []
    collection_root = collection_parts[0] if collection_parts else None
    collection_folder = collection_parts[-1] if collection_parts else None
    collection_path = "/".join(collection_parts) if collection_parts else None

    ipad_label = None
    for name in reversed(collection_parts):
        match = IPAD_LABEL_RE.search(name)
        if match:
            ipad_label = match.group("label").upper()
            break

    return {
        "participant_batch_folder": participant_batch,
        "collection_root_folder": collection_root,
        "collection_folder": collection_folder,
        "collection_relative_path": collection_path,
        "collection_ipad_label": ipad_label,
        "collection_folder_has_back_to_back_2_suffix": bool(
            collection_root and BACK_TO_BACK_SUFFIX_RE.search(collection_root)
        ),
    }


def make_uid(text: str, prefix: str) -> str:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def relative_posix(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def parse_attempt_folder_datetime(folder_name: str) -> Optional[datetime]:
    match = ATTEMPT_FOLDER_RE.match(folder_name)
    if not match:
        return None
    try:
        return datetime.strptime(folder_name, "%Y_%m_%d-%H_%M_%S").replace(
            tzinfo=timezone.utc
        )
    except ValueError:
        return None


def epoch_to_iso(timestamp: Optional[float]) -> Optional[str]:
    if timestamp is None or not math.isfinite(timestamp):
        return None
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()


def load_timezone(name: str) -> ZoneInfo:
    """Load an IANA timezone and fail clearly if it is unavailable."""
    try:
        return ZoneInfo(name)
    except ZoneInfoNotFoundError as exc:
        raise ValueError(
            f"Unknown study timezone {name!r}. Use an IANA timezone such as "
            f"{DEFAULT_STUDY_TIMEZONE!r}."
        ) from exc


def epoch_to_local_date(timestamp: float, study_tz: ZoneInfo) -> date:
    """Convert an epoch timestamp to the local calendar date of data collection."""
    return datetime.fromtimestamp(float(timestamp), tz=timezone.utc).astimezone(study_tz).date()


# -----------------------------------------------------------------------------
# Runsheet loading and validation
# -----------------------------------------------------------------------------


def load_and_validate_runsheet(
    path: Path, sheet_name: Optional[str]
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """
    Load the authoritative runsheet, normalize identifier types, and enforce study
    design invariants that must be true before raw-data reconciliation can proceed.
    """
    if not path.exists():
        raise FileNotFoundError(f"Runsheet not found: {path}")

    excel = pd.ExcelFile(path)
    selected = sheet_name or DEFAULT_RUNSHEET_SHEET
    if selected not in excel.sheet_names:
        raise ValueError(
            f"Sheet {selected!r} not found. Available sheets: {excel.sheet_names}"
        )

    df = pd.read_excel(
        path, sheet_name=selected, dtype=object, keep_default_na=False
    )
    df.columns = [str(c).strip() for c in df.columns]
    # With keep_default_na=False, completely blank Excel rows are read as empty
    # strings rather than NaN. Remove only rows that are truly blank across all
    # columns; never discard a row because one required field is missing.
    nonblank_mask = ~df.apply(lambda row: all(is_missing(v) for v in row), axis=1)
    df = df.loc[nonblank_mask].copy()

    missing_cols = REQUIRED_RUNSHEET_COLUMNS - set(df.columns)
    if missing_cols:
        raise ValueError(
            "Runsheet is missing required columns: " + ", ".join(sorted(missing_cols))
        )

    # Normalize only after checking the expected schema. This prevents Excel's mixed
    # numeric/text typing from leaking into joins later in the pipeline.
    df["PID"] = df["PID"].map(normalize_pid)
    df["Sensor"] = df["Sensor"].map(normalize_sensor)
    df["Phase"] = df["Phase"].map(normalize_phase)
    df["Condition"] = df["Condition"].map(normalize_condition)
    df["Planned_Partner"] = df["Planned_Partner"].map(normalize_partner)
    df["Analysis_Partner"] = df["Analysis_Partner"].map(normalize_partner)
    df["Date"] = df["Date"].map(normalize_date)
    df["Session_ID"] = df["Session_ID"].map(normalize_session_id)

    # Preserve source notes exactly as text when available.
    for col in ["Notes", "Sensor_Note", "Paper_Source"]:
        if col in df.columns:
            df[col] = df[col].map(lambda x: None if is_missing(x) else str(x).strip())

    # Classify documented acquisition expectations before touching raw data. A
    # participant-level no-show note applies to both phases even when written on only
    # one paper-runsheet row. Phase-specific sensor failures remain phase-specific.
    participant_absent: dict[str, bool] = {}
    for pid, group in df.groupby("PID", sort=True):
        participant_absent[pid] = any(
            note_documents_participant_absence(
                row.get("Notes"), row.get("Sensor_Note")
            )
            for _, row in group.iterrows()
        )
    df["Participant_Documented_Absent"] = df["PID"].map(participant_absent).fillna(False)
    df["Raw_Recording_Expectation"] = df.apply(
        classify_raw_recording_expectation, axis=1
    )
    df["Analysis_Role"] = df["Analysis_Partner"].map(
        lambda x: "ACTIVE_DYAD" if x is not None else "NO_ANALYSIS_PARTNER"
    )

    # Allowed_Sensors is an internal, reproducible representation of documented
    # within-participant sensor switches. Most rows contain only the primary Sensor.
    # If a participant has different sensors across phases, the phase whose source
    # note explicitly documents a sensor change is allowed to reconcile both the old
    # and replacement sensor. This is required for cases such as P086 and P098 and
    # avoids hard-coding participant IDs.
    df["Allowed_Sensors"] = df["Sensor"].map(
        lambda s: s if s is not None else None
    )

    issues: list[dict[str, Any]] = []

    if df["PID"].isna().any():
        bad_rows = (df.index[df["PID"].isna()] + 2).tolist()
        raise ValueError(f"Runsheet contains missing PID values at Excel row(s): {bad_rows}")

    def add_issue(severity: str, code: str, detail: str, **context: Any) -> None:
        row = {"severity": severity, "code": code, "detail": detail}
        row.update(context)
        issues.append(row)

    # This is a project-specific production script, so reject an incomplete or wrong
    # runsheet even if its internal rows happen to look structurally plausible.
    if len(df) != EXPECTED_PARTICIPANT_PHASE_ROWS:
        add_issue(
            "ERROR",
            "RUNSHEET_UNEXPECTED_ROW_COUNT",
            f"Runsheet has {len(df)} participant-phase rows; expected {EXPECTED_PARTICIPANT_PHASE_ROWS}.",
        )
    pid_set = set(df["PID"].dropna())
    if pid_set != EXPECTED_PID_SET:
        add_issue(
            "ERROR",
            "RUNSHEET_UNEXPECTED_PID_SET",
            f"PID set differs from expected 001-108. Missing={sorted(EXPECTED_PID_SET - pid_set)}; "
            f"unexpected={sorted(pid_set - EXPECTED_PID_SET)}.",
        )
    session_set = set(df["Session_ID"].dropna())
    if session_set != EXPECTED_SESSION_SET:
        add_issue(
            "ERROR",
            "RUNSHEET_UNEXPECTED_SESSION_SET",
            f"Session_ID set differs from expected S001-S027. Missing={sorted(EXPECTED_SESSION_SET - session_set)}; "
            f"unexpected={sorted(session_set - EXPECTED_SESSION_SET)}.",
        )

    for pid, group in df.groupby("PID", sort=True):
        assigned = [s for s in group["Sensor"] if s is not None]
        unique_assigned = sorted(set(assigned))
        if len(unique_assigned) <= 1:
            continue

        documented_rows = []
        for idx, row in group.iterrows():
            if note_documents_sensor_switch(
                row.get("Notes"), row.get("Sensor_Note")
            ):
                documented_rows.append(idx)

        if len(documented_rows) != 1:
            add_issue(
                "ERROR",
                "RUNSHEET_SENSOR_CHANGE_NOT_UNIQUELY_DOCUMENTED",
                f"PID {pid} has different sensor assignments across phases "
                f"{unique_assigned}, but exactly one phase must explicitly document "
                "the sensor change/switch in Notes or Sensor_Note.",
                PID=pid,
            )
            continue

        change_idx = documented_rows[0]
        df.at[change_idx, "Allowed_Sensors"] = ";".join(unique_assigned)

    # Participant-level invariant: exactly one row for each of phases 1 and 2.
    for pid, group in df.groupby("PID", sort=True):
        phases = sorted(group["Phase"].tolist())
        if phases != [1, 2]:
            add_issue(
                "ERROR",
                "RUNSHEET_PID_PHASE_STRUCTURE",
                f"PID {pid} has phases {phases}; expected exactly [1, 2].",
                PID=pid,
            )
        if group["Session_ID"].nunique(dropna=False) != 1:
            add_issue(
                "ERROR",
                "RUNSHEET_PID_MULTIPLE_SESSIONS",
                f"PID {pid} occurs in more than one Session_ID.",
                PID=pid,
            )
        if group["Date"].nunique(dropna=False) != 1:
            add_issue(
                "ERROR",
                "RUNSHEET_PID_MULTIPLE_DATES",
                f"PID {pid} occurs on more than one study date.",
                PID=pid,
            )

    # Session-level invariant: four PIDs and eight participant-phase rows.
    for session_id, group in df.groupby("Session_ID", sort=True):
        pids = sorted(group["PID"].unique().tolist())
        if len(group) != 8 or len(pids) != 4:
            add_issue(
                "ERROR",
                "RUNSHEET_SESSION_STRUCTURE",
                f"{session_id} has {len(group)} rows and {len(pids)} unique PIDs; "
                "expected 8 rows and 4 PIDs.",
                Session_ID=session_id,
            )

        if group["Date"].nunique(dropna=False) != 1:
            add_issue(
                "ERROR",
                "RUNSHEET_SESSION_DATE_INCONSISTENT",
                f"{session_id} contains more than one Date.",
                Session_ID=session_id,
            )

        for phase, phase_group in group.groupby("Phase", sort=True):
            if phase_group["Condition"].nunique(dropna=False) != 1:
                add_issue(
                    "ERROR",
                    "RUNSHEET_CONDITION_INCONSISTENT",
                    f"{session_id} Phase {phase} contains multiple conditions.",
                    Session_ID=session_id,
                    Phase=phase,
                )

            sensors = [s for s in phase_group["Sensor"] if s is not None]
            duplicate_sensors = sorted(
                sensor for sensor, n in Counter(sensors).items() if n > 1
            )
            if duplicate_sensors:
                add_issue(
                    "ERROR",
                    "RUNSHEET_DUPLICATE_SENSOR_WITHIN_PHASE",
                    f"{session_id} Phase {phase} has duplicate sensor IDs: "
                    f"{duplicate_sensors}",
                    Session_ID=session_id,
                    Phase=phase,
                )

        # Each session should expose participants to one neutral and one emotional
        # story, with the complementary fruit pairing used by KITCO3: NA<->EB or
        # NB<->EA. This validates metadata structure, not an outcome assumption.
        phase_conditions = {
            int(phase): phase_group["Condition"].iloc[0]
            for phase, phase_group in group.groupby("Phase", sort=True)
            if phase_group["Condition"].nunique(dropna=False) == 1
        }
        if set(phase_conditions) == {1, 2}:
            condition_pair = frozenset(phase_conditions.values())
            valid_pairs = {frozenset(("NA", "EB")), frozenset(("NB", "EA"))}
            if condition_pair not in valid_pairs:
                add_issue(
                    "ERROR",
                    "RUNSHEET_SESSION_CONDITION_SEQUENCE",
                    f"{session_id} has Phase 1/2 conditions {phase_conditions}; "
                    "expected complementary NA/EB or NB/EA conditions.",
                    Session_ID=session_id,
                )

    # Analysis-partner reciprocity is required for dyadic mapping. Missing partners are
    # allowed because the final runsheet intentionally represents no-shows/observers.
    lookup = {
        (row.Session_ID, row.Phase, row.PID): row
        for row in df.itertuples(index=False)
    }
    for row in df.itertuples(index=False):
        partner = row.Analysis_Partner
        if partner is None:
            continue
        if partner == row.PID:
            add_issue(
                "ERROR",
                "RUNSHEET_SELF_PARTNER",
                f"{row.PID} is listed as their own Analysis_Partner.",
                Session_ID=row.Session_ID,
                Phase=row.Phase,
                PID=row.PID,
            )
            continue
        partner_row = lookup.get((row.Session_ID, row.Phase, partner))
        if partner_row is None:
            add_issue(
                "ERROR",
                "RUNSHEET_PARTNER_NOT_IN_SESSION_PHASE",
                f"Partner {partner} for PID {row.PID} is not present in the same "
                "session and phase.",
                Session_ID=row.Session_ID,
                Phase=row.Phase,
                PID=row.PID,
            )
        elif partner_row.Analysis_Partner != row.PID:
            add_issue(
                "ERROR",
                "RUNSHEET_PARTNER_NOT_RECIPROCAL",
                f"{row.PID}->{partner}, but {partner}->{partner_row.Analysis_Partner}.",
                Session_ID=row.Session_ID,
                Phase=row.Phase,
                PID=row.PID,
            )

    hard_errors = [x for x in issues if x["severity"] == "ERROR"]
    if hard_errors:
        preview = "\n".join(f"- {x['code']}: {x['detail']}" for x in hard_errors[:10])
        raise ValueError(
            "Runsheet validation failed. Raw reconciliation was not started.\n" + preview
        )

    return df, issues


@dataclass(frozen=True)
class ExpectedPhase:
    session_id: str
    phase: int
    date: date
    condition: str
    all_sensors: frozenset[str]
    active_sensors: frozenset[str]
    active_pairs: frozenset[frozenset[str]]


def build_expected_phases(runsheet: pd.DataFrame) -> list[ExpectedPhase]:
    """Create one expected sensor/pairing specification per session-phase."""
    output: list[ExpectedPhase] = []

    for (session_id, phase), group in runsheet.groupby(
        ["Session_ID", "Phase"], sort=True
    ):
        allowed_by_pid = {
            row.PID: parse_allowed_sensors(row.Allowed_Sensors)
            for row in group.itertuples(index=False)
        }
        all_sensors = frozenset(
            sensor for sensors in allowed_by_pid.values() for sensor in sensors
        )

        active = group[group["Analysis_Partner"].notna()].copy()
        active_pids = set(active["PID"])
        active_sensors = frozenset(
            sensor
            for pid in active_pids
            for sensor in allowed_by_pid.get(pid, frozenset())
        )

        partner_by_pid = dict(zip(group["PID"], group["Analysis_Partner"]))
        active_pairs: set[frozenset[str]] = set()
        for pid, partner in partner_by_pid.items():
            if partner is None or pid >= partner:
                continue
            if partner_by_pid.get(partner) != pid:
                continue
            for s1 in allowed_by_pid.get(pid, frozenset()):
                for s2 in allowed_by_pid.get(partner, frozenset()):
                    if s1 != s2:
                        active_pairs.add(frozenset((s1, s2)))

        output.append(
            ExpectedPhase(
                session_id=session_id,
                phase=int(phase),
                date=group["Date"].iloc[0],
                condition=group["Condition"].iloc[0],
                all_sensors=all_sensors,
                active_sensors=active_sensors,
                active_pairs=frozenset(active_pairs),
            )
        )

    return output


# -----------------------------------------------------------------------------
# Raw file scanning
# -----------------------------------------------------------------------------


def discover_attempt_dirs(raw_root: Path) -> list[Path]:
    """
    Discover recording-attempt directories without assuming a particular higher-level
    folder naming scheme (e.g. JM, MCL, date, date 2).
    """
    attempts: list[Path] = []

    for dirpath, dirnames, filenames in os.walk(raw_root):
        current = Path(dirpath)
        visible_dirs = [d for d in dirnames if not d.startswith(".")]
        visible_files = [f for f in filenames if not f.startswith(".")]

        annotation_files = [
            f for f in visible_files
            if f.lower().startswith("annotations-") and f.lower().endswith(".csv")
        ]
        video_files = [
            f for f in visible_files if Path(f).suffix.lower() in VIDEO_EXTENSIONS
        ]

        sensor_dirs: list[str] = []
        for dirname in visible_dirs:
            if not dirname.isdigit():
                continue
            sensor_path = current / dirname
            try:
                has_csv = any(
                    p.is_file() and p.suffix.lower() == ".csv" and not p.name.startswith(".")
                    for p in sensor_path.iterdir()
                )
            except OSError:
                has_csv = False
            if has_csv:
                sensor_dirs.append(dirname)

        if annotation_files or sensor_dirs or video_files:
            attempts.append(current)
            # Sensor directories are leaves for our purposes. Prune them to avoid
            # accidentally treating them as recording attempts themselves.
            dirnames[:] = [d for d in dirnames if d not in sensor_dirs]

    return sorted(set(attempts))


def read_tail_lines(path: Path, max_bytes: int = 131072) -> list[str]:
    """Return complete non-empty lines from the tail of a large text file.

    Data Collector files can end with an interrupted partial row.  Reading only the
    final line can therefore produce a numeric-looking but truncated timestamp.  This
    helper reads a bounded tail and drops the first line when the read begins in the
    middle of a row.
    """
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        start = max(0, size - max_bytes)
        handle.seek(start)
        data = handle.read()

    lines = data.splitlines()
    if start > 0 and lines:
        # The first tail fragment may begin mid-row.
        lines = lines[1:]
    return [line.decode("utf-8-sig", errors="replace") for line in lines if line.strip()]


def inspect_stream_csv(path: Path) -> dict[str, Any]:
    """
    Inspect only a few lines at the beginning plus the final nonempty line of a
    potentially large stream CSV. The full signal is never loaded.
    """
    result: dict[str, Any] = {
        "metadata_line": None,
        "header_line": None,
        "columns": None,
        "timestamp_column": None,
        "first_timestamp": None,
        "last_timestamp": None,
        "schema_status": "UNKNOWN",
    }

    try:
        with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
            lines = [handle.readline().rstrip("\r\n") for _ in range(6)]
    except OSError as exc:
        result["schema_status"] = f"READ_ERROR:{type(exc).__name__}"
        return result

    lines = [line for line in lines if line != ""]
    if not lines:
        result["schema_status"] = "EMPTY_FILE"
        return result

    parsed: list[list[str]] = []
    for line in lines:
        try:
            parsed.append(next(csv.reader([line])))
        except (csv.Error, StopIteration):
            parsed.append([])

    header_idx: Optional[int] = None
    for idx, fields in enumerate(parsed):
        normalized = [str(x).strip().lower() for x in fields]
        if "timestamp" in normalized:
            header_idx = idx
            break

    # Fallback for an ordinary CSV whose first line is clearly a multi-column header
    # but uses a nonstandard timestamp label. It remains marked nonstandard below.
    if header_idx is None and parsed and len(parsed[0]) >= 2:
        header_idx = 0

    if header_idx is None:
        result["metadata_line"] = lines[0]
        result["schema_status"] = "MISSING_HEADER"
        return result

    header_fields = [str(x).strip() for x in parsed[header_idx]]
    result["metadata_line"] = (
        " || ".join(lines[:header_idx]) if header_idx > 0 else None
    )
    result["header_line"] = lines[header_idx]
    result["columns"] = "|".join(header_fields) if header_fields else None

    normalized_header = [x.lower() for x in header_fields]
    if "timestamp" not in normalized_header:
        result["schema_status"] = "MISSING_TIMESTAMP_COLUMN"
        return result

    ts_index = normalized_header.index("timestamp")
    result["timestamp_column"] = header_fields[ts_index]
    result["schema_status"] = "OK"

    first_data_idx = header_idx + 1
    if first_data_idx >= len(lines):
        result["schema_status"] = "NO_DATA_ROW_IN_PREFIX"
    else:
        try:
            first_values = next(csv.reader([lines[first_data_idx]]))
            result["first_timestamp"] = float(first_values[ts_index])
        except (csv.Error, StopIteration, IndexError, ValueError, TypeError):
            result["schema_status"] = "BAD_FIRST_TIMESTAMP"

    # Search backwards for the last complete row with a coherent timestamp.  An
    # interrupted final write is common enough that a truncated final row must never
    # define a recording envelope.
    first_ts = result.get("first_timestamp")
    last_ts = None
    try:
        tail_lines = read_tail_lines(path)
    except OSError:
        tail_lines = []
    for tail_line in reversed(tail_lines):
        if tail_line == result["header_line"]:
            continue
        try:
            values = next(csv.reader([tail_line]))
            candidate = float(values[ts_index])
        except (csv.Error, StopIteration, IndexError, ValueError, TypeError):
            continue
        if not math.isfinite(candidate):
            continue
        if first_ts is not None and math.isfinite(float(first_ts)) and candidate < float(first_ts):
            continue
        last_ts = candidate
        break

    result["last_timestamp"] = last_ts
    if last_ts is None and result["schema_status"] == "OK":
        result["schema_status"] = "NO_VALID_LAST_TIMESTAMP"

    return result


def read_annotation_events(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """Read an annotation CSV and preserve every event exactly once."""
    events: list[dict[str, Any]] = []
    problems: list[str] = []

    try:
        if path.stat().st_size == 0:
            return [], ["ANNOTATION_EMPTY_FILE"]
    except OSError:
        pass

    try:
        with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"Timestamp", "Type", "Content"}
            if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
                return [], ["ANNOTATION_BAD_SCHEMA"]

            for row_number, row in enumerate(reader, start=2):
                try:
                    timestamp = float(row["Timestamp"])
                except (TypeError, ValueError):
                    problems.append(f"ANNOTATION_BAD_TIMESTAMP_ROW_{row_number}")
                    continue
                events.append(
                    {
                        "timestamp": timestamp,
                        "type": (row.get("Type") or "").strip(),
                        "content": (row.get("Content") or "").strip() or None,
                        "row_number": row_number,
                    }
                )
    except OSError as exc:
        return [], [f"ANNOTATION_READ_ERROR:{type(exc).__name__}"]

    return events, problems


def summarize_annotations(events: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Summarize annotation timing without deciding the final analysis epoch."""
    by_type: dict[str, list[float]] = defaultdict(list)
    for event in events:
        by_type[event["type"]].append(float(event["timestamp"]))

    recording_starts = sorted(
        ts for label in RECORDING_START_TYPES for ts in by_type.get(label, [])
    )
    recording_stops = sorted(
        ts for label in RECORDING_STOP_TYPES for ts in by_type.get(label, [])
    )
    video_starts = sorted(
        ts for label in VIDEO_START_TYPES for ts in by_type.get(label, [])
    )
    video_stops = sorted(
        ts for label in VIDEO_STOP_TYPES for ts in by_type.get(label, [])
    )
    pause_events = sorted(ts for label in PAUSE_TYPES for ts in by_type.get(label, []))
    resume_events = sorted(ts for label in RESUME_TYPES for ts in by_type.get(label, []))

    recording_single = (
        len(recording_starts) == 1
        and len(recording_stops) == 1
        and recording_starts[0] < recording_stops[0]
    )
    video_single = (
        len(video_starts) == 1
        and len(video_stops) == 1
        and video_starts[0] < video_stops[0]
    )

    if recording_single:
        recording_status = "OK_SINGLE_RECORDING_INTERVAL"
    elif not recording_starts and not recording_stops:
        recording_status = "MISSING_RECORDING_INTERVAL"
    elif len(recording_starts) != len(recording_stops):
        recording_status = "UNPAIRED_RECORDING_EVENTS"
    else:
        recording_status = "MULTIPLE_OR_NONSTANDARD_RECORDING_EVENTS"

    if video_single:
        video_status = "OK_SINGLE_VIDEO_INTERVAL"
    elif not video_starts and not video_stops:
        video_status = "MISSING_VIDEO_INTERVAL"
    elif len(video_starts) != len(video_stops):
        video_status = "UNPAIRED_VIDEO_EVENTS"
    else:
        video_status = "MULTIPLE_OR_NONSTANDARD_VIDEO_EVENTS"

    if not pause_events and not resume_events:
        pause_resume_status = "NO_PAUSE_RESUME_EVENTS"
    elif len(pause_events) == len(resume_events):
        pause_resume_status = "BALANCED_PAUSE_RESUME_COUNTS"
    else:
        pause_resume_status = "UNPAIRED_PAUSE_RESUME_COUNTS"

    video_start = video_starts[0] if video_single else None
    video_stop = video_stops[0] if video_single else None
    recording_start = recording_starts[0] if recording_single else (
        min(recording_starts) if recording_starts else None
    )
    recording_stop = recording_stops[0] if recording_single else (
        max(recording_stops) if recording_stops else None
    )

    n_pause_in_video = 0
    n_resume_in_video = 0
    if video_start is not None and video_stop is not None:
        n_pause_in_video = sum(video_start <= ts <= video_stop for ts in pause_events)
        n_resume_in_video = sum(video_start <= ts <= video_stop for ts in resume_events)

    return {
        "recording_start": recording_start,
        "recording_stop": recording_stop,
        # Envelope is used only for inventory/time-block grouping. Script 03 will
        # define the final cooking epoch after all annotation exceptions are reviewed.
        "video_envelope_start": min(video_starts) if video_starts else None,
        "video_envelope_stop": max(video_stops) if video_stops else None,
        "recording_duration_s": (
            float(recording_stop - recording_start)
            if recording_single else None
        ),
        "video_duration_s": (
            float(video_stop - video_start)
            if video_single else None
        ),
        "n_recording_starts": len(recording_starts),
        "n_recording_stops": len(recording_stops),
        "n_video_starts": len(video_starts),
        "n_video_stops": len(video_stops),
        "n_pause_events": len(pause_events),
        "n_resume_events": len(resume_events),
        "n_pause_events_within_video": n_pause_in_video,
        "n_resume_events_within_video": n_resume_in_video,
        "pause_or_resume_within_video": bool(n_pause_in_video or n_resume_in_video),
        "recording_annotation_status": recording_status,
        "video_annotation_status": video_status,
        "pause_resume_status": pause_resume_status,
        "annotation_types": compact_join(by_type.keys()),
    }


def scan_raw_data(
    raw_root: Path,
    study_tz: ZoneInfo,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    """Return file inventory, annotation events, attempt summary, and scan issues."""
    if not raw_root.exists() or not raw_root.is_dir():
        raise NotADirectoryError(f"Raw root must be an existing directory: {raw_root}")

    attempt_dirs = discover_attempt_dirs(raw_root)
    if not attempt_dirs:
        raise ValueError(f"No recording-attempt directories found under {raw_root}")

    file_rows: list[dict[str, Any]] = []
    annotation_rows: list[dict[str, Any]] = []
    attempt_rows: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []

    def add_issue(severity: str, code: str, detail: str, **context: Any) -> None:
        row = {"severity": severity, "code": code, "detail": detail}
        row.update(context)
        issues.append(row)

    total_attempts = len(attempt_dirs)
    for attempt_index, attempt_dir in enumerate(attempt_dirs, start=1):
        if attempt_index == 1 or attempt_index % 25 == 0 or attempt_index == total_attempts:
            logging.info("Inventory progress: %d/%d recording attempts", attempt_index, total_attempts)

        rel_attempt = relative_posix(attempt_dir, raw_root)
        attempt_uid = make_uid(rel_attempt, "ATT")
        parsed_dt = parse_attempt_folder_datetime(attempt_dir.name)
        folder_local_date = parsed_dt.astimezone(study_tz).date() if parsed_dt else None
        raw_date: Optional[date] = None
        raw_date_source: Optional[str] = None
        path_context = extract_raw_path_context(attempt_dir, raw_root)

        source_path = relative_posix(attempt_dir.parent, raw_root)
        try:
            children = [p for p in attempt_dir.iterdir() if not p.name.startswith(".")]
        except OSError as exc:
            add_issue(
                "ERROR",
                "ATTEMPT_DIRECTORY_READ_ERROR",
                f"Cannot list {rel_attempt}: {type(exc).__name__}: {exc}",
                attempt_uid=attempt_uid,
                raw_path=rel_attempt,
            )
            continue

        annotation_files = sorted(
            p
            for p in children
            if p.is_file()
            and p.name.lower().startswith("annotations-")
            and p.suffix.lower() == ".csv"
        )
        video_files = sorted(
            p for p in children if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
        )
        sensor_dirs = sorted(
            p for p in children if p.is_dir() and p.name.isdigit()
        )
        annotation_total_size_bytes = sum(
            p.stat().st_size for p in annotation_files if p.exists()
        )
        video_total_size_bytes = sum(
            p.stat().st_size for p in video_files if p.exists()
        )

        annotation_summary: dict[str, Any] = {
            "recording_start": None,
            "recording_stop": None,
            "video_envelope_start": None,
            "video_envelope_stop": None,
            "recording_duration_s": None,
            "video_duration_s": None,
            "n_recording_starts": 0,
            "n_recording_stops": 0,
            "n_video_starts": 0,
            "n_video_stops": 0,
            "n_pause_events": 0,
            "n_resume_events": 0,
            "n_pause_events_within_video": 0,
            "n_resume_events_within_video": 0,
            "pause_or_resume_within_video": False,
            "recording_annotation_status": "NO_ANNOTATION_FILE",
            "video_annotation_status": "NO_ANNOTATION_FILE",
            "pause_resume_status": "NO_ANNOTATION_FILE",
            "annotation_types": None,
        }

        if len(annotation_files) > 1:
            add_issue(
                "ERROR",
                "MULTIPLE_ANNOTATION_FILES_IN_ATTEMPT",
                f"{rel_attempt} contains {len(annotation_files)} annotation files.",
                attempt_uid=attempt_uid,
                raw_path=rel_attempt,
            )

        all_events: list[dict[str, Any]] = []
        for annotation_file in annotation_files:
            events, problems = read_annotation_events(annotation_file)
            for problem in problems:
                severity = "WARNING" if problem in {"ANNOTATION_EMPTY_FILE"} else "ERROR"
                add_issue(
                    severity,
                    problem,
                    f"Problem reading {relative_posix(annotation_file, raw_root)}",
                    attempt_uid=attempt_uid,
                    raw_path=rel_attempt,
                )
            for event in events:
                annotation_rows.append(
                    {
                        "attempt_uid": attempt_uid,
                        "attempt_path": rel_attempt,
                        "annotation_file": relative_posix(annotation_file, raw_root),
                        "row_number": event["row_number"],
                        "timestamp": event["timestamp"],
                        "timestamp_utc": epoch_to_iso(event["timestamp"]),
                        "type": event["type"],
                        "content": event["content"],
                    }
                )
                all_events.append(event)

        if all_events:
            annotation_summary = summarize_annotations(all_events)

        if sensor_dirs and not annotation_files:
            add_issue(
                "WARNING",
                "SENSOR_ATTEMPT_HAS_NO_ANNOTATION_FILE",
                f"{rel_attempt} contains sensor data but no annotation CSV.",
                attempt_uid=attempt_uid,
                raw_path=rel_attempt,
            )
        if annotation_files and annotation_summary["recording_annotation_status"] != "OK_SINGLE_RECORDING_INTERVAL":
            add_issue(
                "WARNING",
                "NONSTANDARD_RECORDING_ANNOTATIONS",
                f"{rel_attempt}: {annotation_summary['recording_annotation_status']}",
                attempt_uid=attempt_uid,
                raw_path=rel_attempt,
            )
        if annotation_files and annotation_summary["video_annotation_status"] != "OK_SINGLE_VIDEO_INTERVAL":
            add_issue(
                "WARNING",
                "NONSTANDARD_VIDEO_ANNOTATIONS",
                f"{rel_attempt}: {annotation_summary['video_annotation_status']}",
                attempt_uid=attempt_uid,
                raw_path=rel_attempt,
            )
        if annotation_summary.get("pause_or_resume_within_video"):
            add_issue(
                "WARNING",
                "PAUSE_OR_RESUME_WITHIN_VIDEO_INTERVAL",
                f"{rel_attempt} has pause/resume event(s) inside the annotated cooking interval.",
                attempt_uid=attempt_uid,
                raw_path=rel_attempt,
            )

        # Data Collector folder/epoch timestamps are UTC. The runsheet Date is the
        # Vancouver study date, so match on the local calendar date, not the UTC date.
        anchor_ts = annotation_summary.get("recording_start") or annotation_summary.get(
            "video_envelope_start"
        )
        if anchor_ts is not None:
            raw_date = epoch_to_local_date(float(anchor_ts), study_tz)
            raw_date_source = "ANNOTATION_TIMESTAMP_LOCAL_DATE"
        elif parsed_dt is not None:
            raw_date = folder_local_date
            raw_date_source = "FOLDER_UTC_TIMESTAMP_LOCAL_DATE"

        if anchor_ts is not None and folder_local_date is not None:
            annotation_local_date = epoch_to_local_date(float(anchor_ts), study_tz)
            if annotation_local_date != folder_local_date:
                add_issue(
                    "WARNING",
                    "FOLDER_AND_ANNOTATION_LOCAL_DATE_DISAGREE",
                    f"{rel_attempt}: folder timestamp implies local date "
                    f"{folder_local_date}, annotation implies {annotation_local_date}.",
                    attempt_uid=attempt_uid,
                    raw_path=rel_attempt,
                )

        if annotation_summary["n_pause_events"] != annotation_summary["n_resume_events"]:
            add_issue(
                "WARNING",
                "UNPAIRED_PAUSE_RESUME_EVENTS",
                f"{rel_attempt} has {annotation_summary['n_pause_events']} pause and "
                f"{annotation_summary['n_resume_events']} resume event(s).",
                attempt_uid=attempt_uid,
                raw_path=rel_attempt,
            )

        rs = annotation_summary.get("recording_start")
        re_ = annotation_summary.get("recording_stop")
        vs = annotation_summary.get("video_envelope_start")
        ve = annotation_summary.get("video_envelope_stop")
        if all(v is not None for v in (rs, re_, vs, ve)) and (float(vs) < float(rs) or float(ve) > float(re_)):
            add_issue(
                "WARNING",
                "VIDEO_INTERVAL_OUTSIDE_RECORDING_INTERVAL",
                f"{rel_attempt}: video annotation interval extends outside the sensor "
                "recording interval.",
                attempt_uid=attempt_uid,
                raw_path=rel_attempt,
            )

        sensor_full_ids: list[str] = []
        sensor_short_ids: list[str] = []
        streams_by_sensor: dict[str, set[str]] = defaultdict(set)
        stream_first_timestamps: list[float] = []
        stream_last_timestamps: list[float] = []
        stream_total_size_bytes = 0
        file_rows_start = len(file_rows)

        for sensor_dir in sensor_dirs:
            full_sensor = sensor_dir.name
            try:
                short_sensor = raw_sensor_to_short(full_sensor)
            except ValueError as exc:
                add_issue(
                    "ERROR",
                    "INVALID_RAW_SENSOR_DIR",
                    str(exc),
                    attempt_uid=attempt_uid,
                    raw_path=rel_attempt,
                )
                continue

            sensor_full_ids.append(full_sensor)
            sensor_short_ids.append(short_sensor)

            stream_files = sorted(
                p
                for p in sensor_dir.iterdir()
                if p.is_file() and not p.name.startswith(".") and p.suffix.lower() == ".csv"
            )
            stream_type_counts = Counter(p.name.split("-", 1)[0] for p in stream_files)
            duplicate_stream_types = sorted(
                stream for stream, count in stream_type_counts.items() if count > 1
            )
            if duplicate_stream_types:
                add_issue(
                    "ERROR",
                    "DUPLICATE_STREAM_TYPE_WITHIN_SENSOR_ATTEMPT",
                    f"{relative_posix(sensor_dir, raw_root)} contains multiple files for "
                    f"stream type(s): {duplicate_stream_types}.",
                    attempt_uid=attempt_uid,
                    raw_path=rel_attempt,
                    sensor_short=short_sensor,
                )
            if not stream_files:
                add_issue(
                    "WARNING",
                    "EMPTY_SENSOR_DIRECTORY",
                    f"Sensor directory {relative_posix(sensor_dir, raw_root)} contains no CSV streams.",
                    attempt_uid=attempt_uid,
                    raw_path=rel_attempt,
                    sensor_short=short_sensor,
                )

            for stream_file in stream_files:
                stream_type = stream_file.name.split("-", 1)[0]
                streams_by_sensor[short_sensor].add(stream_type)
                if stream_type not in KNOWN_STREAM_TYPES:
                    add_issue(
                        "INFO",
                        "UNKNOWN_STREAM_TYPE_INVENTORIED",
                        f"{relative_posix(stream_file, raw_root)} has unrecognized stream type {stream_type!r}.",
                        attempt_uid=attempt_uid,
                        raw_path=rel_attempt,
                        sensor_short=short_sensor,
                    )
                inspected = inspect_stream_csv(stream_file)
                try:
                    stream_size_bytes = stream_file.stat().st_size
                except OSError:
                    stream_size_bytes = 0
                stream_total_size_bytes += int(stream_size_bytes)
                first_ts = inspected.get("first_timestamp")
                last_ts = inspected.get("last_timestamp")
                if (
                    first_ts is not None
                    and last_ts is not None
                    and math.isfinite(float(first_ts))
                    and math.isfinite(float(last_ts))
                    and float(last_ts) >= float(first_ts)
                ):
                    stream_first_timestamps.append(float(first_ts))
                    stream_last_timestamps.append(float(last_ts))
                file_rows.append(
                    {
                        "attempt_uid": attempt_uid,
                        "attempt_path": rel_attempt,
                        "source_path": source_path,
                        "raw_date": raw_date.isoformat() if raw_date else None,
                        "sensor_full": full_sensor,
                        "sensor_short": short_sensor,
                        "stream_type": stream_type,
                        "file_path": relative_posix(stream_file, raw_root),
                        "file_size_bytes": stream_size_bytes,
                        "metadata_line": inspected["metadata_line"],
                        "header_line": inspected["header_line"],
                        "columns": inspected["columns"],
                        "timestamp_column": inspected.get("timestamp_column"),
                        "first_timestamp": inspected["first_timestamp"],
                        "first_timestamp_utc": epoch_to_iso(inspected["first_timestamp"]),
                        "last_timestamp": inspected["last_timestamp"],
                        "last_timestamp_utc": epoch_to_iso(inspected["last_timestamp"]),
                        "schema_status": inspected["schema_status"],
                    }
                )
                if inspected["schema_status"] != "OK":
                    primary_stream = stream_type in {"ECG", "Acc"}
                    add_issue(
                        "WARNING" if primary_stream else "INFO",
                        "PRIMARY_STREAM_SCHEMA_NONSTANDARD" if primary_stream else "OPTIONAL_STREAM_SCHEMA_NONSTANDARD",
                        f"{relative_posix(stream_file, raw_root)}: {inspected['schema_status']}",
                        attempt_uid=attempt_uid,
                        raw_path=rel_attempt,
                        sensor_short=short_sensor,
                    )

        stream_interval_start = min(stream_first_timestamps) if stream_first_timestamps else None
        stream_interval_stop = max(stream_last_timestamps) if stream_last_timestamps else None

        # Rare fallback for nonstandard attempt folders with missing annotations.
        if raw_date is None and stream_interval_start is not None:
            raw_date = epoch_to_local_date(stream_interval_start, study_tz)
            raw_date_source = "STREAM_TIMESTAMP_LOCAL_DATE"
        for i in range(file_rows_start, len(file_rows)):
            file_rows[i]["raw_date"] = raw_date.isoformat() if raw_date else None

        # A collision would make short-ID reconciliation unsafe.
        short_to_full: dict[str, set[str]] = defaultdict(set)
        for full, short in zip(sensor_full_ids, sensor_short_ids):
            short_to_full[short].add(full)
        collisions = {k: v for k, v in short_to_full.items() if len(v) > 1}
        if collisions:
            add_issue(
                "ERROR",
                "RAW_SHORT_SENSOR_COLLISION_WITHIN_ATTEMPT",
                f"Multiple full sensor IDs share the same four-digit suffix: {collisions}",
                attempt_uid=attempt_uid,
                raw_path=rel_attempt,
            )

        attempt_rows.append(
            {
                "attempt_uid": attempt_uid,
                "attempt_path": rel_attempt,
                "source_path": source_path,
                **path_context,
                "attempt_folder": attempt_dir.name,
                "folder_datetime_utc": parsed_dt.isoformat() if parsed_dt else None,
                "folder_datetime_local": parsed_dt.astimezone(study_tz).isoformat() if parsed_dt else None,
                "raw_date": raw_date.isoformat() if raw_date else None,
                "raw_date_source": raw_date_source,
                "study_timezone": str(study_tz),
                "n_annotation_files": len(annotation_files),
                "annotation_files": compact_join(
                    relative_posix(p, raw_root) for p in annotation_files
                ),
                "annotation_total_size_bytes": int(annotation_total_size_bytes),
                "n_video_files": len(video_files),
                "video_files": compact_join(
                    relative_posix(p, raw_root) for p in video_files
                ),
                "video_total_size_bytes": int(video_total_size_bytes),
                "n_sensors": len(sensor_short_ids),
                "stream_total_size_bytes": int(stream_total_size_bytes),
                "stream_interval_start": stream_interval_start,
                "stream_interval_start_utc": epoch_to_iso(stream_interval_start),
                "stream_interval_stop": stream_interval_stop,
                "stream_interval_stop_utc": epoch_to_iso(stream_interval_stop),
                "sensor_full_ids": compact_join(sensor_full_ids),
                "sensor_short_ids": compact_join(sensor_short_ids),
                "available_streams": compact_join(
                    stream for streams in streams_by_sensor.values() for stream in streams
                ),
                "streams_by_sensor": json.dumps(
                    {k: sorted(v) for k, v in sorted(streams_by_sensor.items())},
                    sort_keys=True,
                ),
                **annotation_summary,
                "recording_start_utc": epoch_to_iso(annotation_summary["recording_start"]),
                "recording_stop_utc": epoch_to_iso(annotation_summary["recording_stop"]),
                "video_envelope_start_utc": epoch_to_iso(
                    annotation_summary["video_envelope_start"]
                ),
                "video_envelope_stop_utc": epoch_to_iso(
                    annotation_summary["video_envelope_stop"]
                ),
                "is_non_sensor_attempt": len(sensor_short_ids) == 0,
                "is_annotation_only": (
                    len(sensor_short_ids) == 0
                    and len(annotation_files) > 0
                    and len(video_files) == 0
                ),
                "is_video_only": (
                    len(sensor_short_ids) == 0
                    and len(video_files) > 0
                    and len(annotation_files) == 0
                ),
            }
        )

    file_df = pd.DataFrame(file_rows)
    annotation_df = pd.DataFrame(annotation_rows)
    attempt_df = pd.DataFrame(attempt_rows)

    # Detect a global suffix collision as an additional safety check. Different full
    # sensors may legitimately appear across dates, but two full IDs mapping to the same
    # short ID would make runsheet reconciliation intrinsically ambiguous.
    if not file_df.empty:
        collision_table = (
            file_df[["sensor_short", "sensor_full"]]
            .drop_duplicates()
            .groupby("sensor_short")["sensor_full"]
            .nunique()
        )
        for short, count in collision_table[collision_table > 1].items():
            full_ids = sorted(
                file_df.loc[file_df["sensor_short"] == short, "sensor_full"].unique()
            )
            add_issue(
                "ERROR",
                "RAW_SHORT_SENSOR_COLLISION_GLOBAL",
                f"Short sensor {short} maps to multiple full IDs: {full_ids}",
                sensor_short=short,
            )

    return file_df, annotation_df, attempt_df, issues


# -----------------------------------------------------------------------------
# Time-block construction and conservative mapping
# -----------------------------------------------------------------------------


def split_sensor_set(text: Any) -> frozenset[str]:
    if is_missing(text):
        return frozenset()
    return frozenset(x for x in str(text).split(";") if x)


def choose_interval(row: pd.Series) -> tuple[Optional[float], Optional[float], str]:
    """Choose an interval for grouping attempts, not for final epoch analysis."""
    vs = row.get("video_envelope_start")
    ve = row.get("video_envelope_stop")
    if not is_missing(vs) and not is_missing(ve) and float(vs) <= float(ve):
        return float(vs), float(ve), "VIDEO_ENVELOPE"

    rs = row.get("recording_start")
    re_ = row.get("recording_stop")
    if not is_missing(rs) and not is_missing(re_) and float(rs) <= float(re_):
        return float(rs), float(re_), "RECORDING_INTERVAL"

    ss = row.get("stream_interval_start")
    se = row.get("stream_interval_stop")
    if not is_missing(ss) and not is_missing(se) and float(ss) <= float(se):
        return float(ss), float(se), "STREAM_TIMESTAMP_ENVELOPE"

    folder_dt = row.get("folder_datetime_utc")
    if not is_missing(folder_dt):
        try:
            ts = datetime.fromisoformat(str(folder_dt)).timestamp()
            return ts, ts, "FOLDER_TIMESTAMP_POINT"
        except ValueError:
            pass

    return None, None, "NO_INTERVAL"


def build_time_blocks(attempt_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Group recording attempts that overlap in time. Parallel iPads recording the two
    rooms of the same experimental phase should therefore fall into the same block.

    No arbitrary start-time tolerance is used. Attempts are connected only when their
    selected intervals overlap. Non-overlapping restart attempts remain separate and are
    later surfaced rather than silently merged.
    """
    work = attempt_df.copy()
    intervals = work.apply(choose_interval, axis=1)
    work["block_interval_start"] = [x[0] for x in intervals]
    work["block_interval_stop"] = [x[1] for x in intervals]
    work["block_interval_basis"] = [x[2] for x in intervals]

    block_assignments: dict[str, str] = {}
    block_rows: list[dict[str, Any]] = []
    block_counter = 0

    for raw_date, date_group in work.groupby("raw_date", dropna=False, sort=True):
        sortable = date_group[date_group["block_interval_start"].notna()].sort_values(
            ["block_interval_start", "block_interval_stop", "attempt_uid"]
        )

        current_attempts: list[str] = []
        current_start: Optional[float] = None
        current_stop: Optional[float] = None

        def finalize_current() -> None:
            nonlocal block_counter, current_attempts, current_start, current_stop
            if not current_attempts:
                return
            block_counter += 1
            block_id = f"TB{block_counter:04d}"
            for uid in current_attempts:
                block_assignments[uid] = block_id

            members = work[work["attempt_uid"].isin(current_attempts)]
            sensors: set[str] = set()
            pair_strings: list[str] = []
            for sensor_text in members["sensor_short_ids"]:
                sensor_set = split_sensor_set(sensor_text)
                sensors.update(sensor_set)
                if len(sensor_set) == 2:
                    pair_strings.append("+".join(sorted(sensor_set)))

            block_key = make_uid(
                "|".join(sorted(members["attempt_path"].astype(str))), "TBK"
            )
            block_rows.append(
                {
                    "time_block_id": block_id,
                    "time_block_key": block_key,
                    "raw_date": raw_date,
                    "block_start": current_start,
                    "block_start_utc": epoch_to_iso(current_start),
                    "block_stop": current_stop,
                    "block_stop_utc": epoch_to_iso(current_stop),
                    "n_attempts": len(members),
                    "attempt_uids": compact_join(members["attempt_uid"]),
                    "attempt_paths": compact_join(members["attempt_path"]),
                    "source_paths": compact_join(members["source_path"]),
                    "observed_sensors": ";".join(sorted(sensors)),
                    "observed_two_sensor_attempt_pairs": ";".join(sorted(set(pair_strings))),
                    "n_sensor_bearing_attempts": int((members["n_sensors"] > 0).sum()),
                    "n_non_sensor_attempts": int((members["n_sensors"] == 0).sum()),
                    "n_annotation_only_attempts": int(members["is_annotation_only"].sum()),
                    "n_video_only_attempts": int(members["is_video_only"].sum()),
                }
            )
            current_attempts = []
            current_start = None
            current_stop = None

        for row in sortable.itertuples(index=False):
            start = float(row.block_interval_start)
            stop = float(row.block_interval_stop)
            if current_start is None:
                current_start, current_stop = start, stop
                current_attempts = [row.attempt_uid]
            elif start <= float(current_stop):
                # Connected-component interval clustering: extending the current stop
                # also allows A overlaps B and B overlaps C to remain one block.
                current_stop = max(float(current_stop), stop)
                current_attempts.append(row.attempt_uid)
            else:
                finalize_current()
                current_start, current_stop = start, stop
                current_attempts = [row.attempt_uid]

        finalize_current()

        # Attempts with no usable interval are deliberately isolated. They cannot be
        # safely grouped by time and therefore should not inherit another mapping.
        no_interval = date_group[date_group["block_interval_start"].isna()]
        for row in no_interval.itertuples(index=False):
            block_counter += 1
            block_id = f"TB{block_counter:04d}"
            block_assignments[row.attempt_uid] = block_id
            sensors = split_sensor_set(row.sensor_short_ids)
            block_rows.append(
                {
                    "time_block_id": block_id,
                    "time_block_key": make_uid(str(row.attempt_path), "TBK"),
                    "raw_date": raw_date,
                    "block_start": None,
                    "block_start_utc": None,
                    "block_stop": None,
                    "block_stop_utc": None,
                    "n_attempts": 1,
                    "attempt_uids": row.attempt_uid,
                    "attempt_paths": row.attempt_path,
                    "source_paths": row.source_path,
                    "observed_sensors": ";".join(sorted(sensors)),
                    "observed_two_sensor_attempt_pairs": (
                        "+".join(sorted(sensors)) if len(sensors) == 2 else ""
                    ),
                    "n_sensor_bearing_attempts": int(row.n_sensors > 0),
                    "n_non_sensor_attempts": int(row.n_sensors == 0),
                    "n_annotation_only_attempts": int(bool(row.is_annotation_only)),
                    "n_video_only_attempts": int(bool(row.is_video_only)),
                }
            )

    work["time_block_id"] = work["attempt_uid"].map(block_assignments)
    block_df = pd.DataFrame(block_rows)
    return work, block_df



# -----------------------------------------------------------------------------
# Production attempt mapping: batch -> session, duplicate resolution, chronology -> phase
# -----------------------------------------------------------------------------


def parse_participant_batch_pids(value: Any) -> frozenset[str]:
    """Parse a folder such as ``Participants 33-35`` into normalized PID labels."""
    if is_missing(value):
        return frozenset()
    match = PARTICIPANT_BATCH_RE.match(str(value).strip())
    if not match:
        return frozenset()
    start = int(match.group("start"))
    stop = int(match.group("stop"))
    if start < 1 or stop < start or stop > 999:
        return frozenset()
    return frozenset(f"{pid:03d}" for pid in range(start, stop + 1))


def map_attempts_to_sessions(
    attempt_df: pd.DataFrame,
    runsheet: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """
    Map attempts to Session_ID using participant-batch membership plus study date.

    Joey's raw export is organized into top-level ``Participants xx-yy`` batches.
    Those folder names are not treated as study metadata by themselves. A mapping is
    accepted only when the parsed PID set is a subset of exactly one runsheet session
    and the attempt's Vancouver-local raw date agrees with that session's Date.
    """
    out = attempt_df.copy()
    issues: list[dict[str, Any]] = []

    session_specs: dict[str, dict[str, Any]] = {}
    for session_id, group in runsheet.groupby("Session_ID", sort=True):
        session_specs[str(session_id)] = {
            "pids": frozenset(group["PID"]),
            "date": group["Date"].iloc[0],
        }

    mapped_sessions: list[Optional[str]] = []
    mapping_statuses: list[str] = []
    batch_pid_texts: list[Optional[str]] = []
    mapping_evidence: list[str] = []

    for row in out.itertuples(index=False):
        batch_pids = parse_participant_batch_pids(row.participant_batch_folder)
        batch_pid_texts.append(";".join(sorted(batch_pids)) if batch_pids else None)

        raw_date: Optional[date]
        try:
            raw_date = date.fromisoformat(str(row.raw_date)) if not is_missing(row.raw_date) else None
        except ValueError:
            raw_date = None

        pid_candidates = [
            session_id
            for session_id, spec in session_specs.items()
            if batch_pids and batch_pids.issubset(spec["pids"])
        ]
        date_candidates = [
            session_id
            for session_id in pid_candidates
            if raw_date is not None and session_specs[session_id]["date"] == raw_date
        ]

        if len(date_candidates) == 1:
            session_id = date_candidates[0]
            status = "MATCHED_BATCH_PIDSET_AND_DATE"
        elif len(pid_candidates) == 1 and raw_date is None:
            session_id = pid_candidates[0]
            status = "MATCHED_BATCH_PIDSET_DATE_UNAVAILABLE"
            issues.append({
                "severity": "WARNING",
                "code": "ATTEMPT_SESSION_DATE_UNAVAILABLE",
                "detail": f"{row.attempt_path} mapped to {session_id} by participant batch, but no reliable raw date was available.",
                "attempt_uid": row.attempt_uid,
                "raw_path": row.attempt_path,
                "Session_ID": session_id,
            })
        else:
            session_id = None
            if not batch_pids:
                status = "UNPARSEABLE_PARTICIPANT_BATCH"
                detail = f"Cannot parse participant batch folder {row.participant_batch_folder!r} for {row.attempt_path}."
            elif len(pid_candidates) == 0:
                status = "NO_SESSION_MATCH_FOR_BATCH_PIDSET"
                detail = f"Participant batch {sorted(batch_pids)} does not match any final-runsheet session for {row.attempt_path}."
            elif raw_date is not None and len(date_candidates) == 0:
                status = "BATCH_SESSION_DATE_MISMATCH"
                detail = (
                    f"{row.attempt_path} batch PID set matches {pid_candidates}, but raw date {raw_date} "
                    "does not match the final-runsheet date for that session."
                )
            else:
                status = "AMBIGUOUS_BATCH_SESSION"
                detail = f"{row.attempt_path} has ambiguous session candidates {pid_candidates}."
            if int(getattr(row, "n_sensors", 0) or 0) > 0:
                issues.append({
                    "severity": "ERROR",
                    "code": status,
                    "detail": detail,
                    "attempt_uid": row.attempt_uid,
                    "raw_path": row.attempt_path,
                    "raw_date": row.raw_date,
                })

        mapped_sessions.append(session_id)
        mapping_statuses.append(status)
        mapping_evidence.append(json.dumps({
            "participant_batch_pids": sorted(batch_pids),
            "raw_date": raw_date.isoformat() if raw_date else None,
            "pid_candidates": pid_candidates,
            "date_candidates": date_candidates,
        }, sort_keys=True))

    out["participant_batch_pids"] = batch_pid_texts
    out["session_mapping_status"] = mapping_statuses
    out["mapped_session_id"] = mapped_sessions
    out["session_mapping_evidence"] = mapping_evidence
    return out, issues


def build_attempt_fingerprints(
    attempt_df: pd.DataFrame,
    file_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add a deterministic structural fingerprint for detecting copied attempts.

    The fingerprint intentionally excludes directory prefixes so that the same raw
    acquisition copied into two participant-batch folders resolves to the same value.
    It uses stream identities, sizes, first/last coherent timestamps, and attempt-level
    file metadata. It is not a substitute for cryptographic hashing of 85 GB of source
    data; it is a lightweight duplicate-copy detector for inventory reconciliation.
    """
    out = attempt_df.copy()
    descriptors: dict[str, list[tuple[Any, ...]]] = defaultdict(list)
    if not file_df.empty:
        for row in file_df.itertuples(index=False):
            descriptors[str(row.attempt_uid)].append((
                str(row.sensor_full),
                str(row.stream_type),
                Path(str(row.file_path)).name,
                int(row.file_size_bytes) if not is_missing(row.file_size_bytes) else 0,
                None if is_missing(row.first_timestamp) else round(float(row.first_timestamp), 6),
                None if is_missing(row.last_timestamp) else round(float(row.last_timestamp), 6),
                None if is_missing(row.header_line) else str(row.header_line),
            ))

    fingerprints: list[str] = []
    for row in out.itertuples(index=False):
        video_names = sorted(
            Path(x).name for x in str(row.video_files).split(";") if x and x != "nan"
        ) if not is_missing(row.video_files) else []
        annotation_names = sorted(
            Path(x).name for x in str(row.annotation_files).split(";") if x and x != "nan"
        ) if not is_missing(row.annotation_files) else []
        payload = {
            "attempt_folder": row.attempt_folder,
            "sensor_full_ids": None if is_missing(row.sensor_full_ids) else str(row.sensor_full_ids),
            "streams": sorted(descriptors.get(str(row.attempt_uid), [])),
            "annotation_names": annotation_names,
            "annotation_bytes": int(row.annotation_total_size_bytes or 0),
            "video_names": video_names,
            "video_bytes": int(row.video_total_size_bytes or 0),
        }
        fingerprints.append(hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=safe_json, separators=(",", ":")).encode("utf-8")
        ).hexdigest())
    out["attempt_fingerprint"] = fingerprints
    return out


def resolve_duplicate_attempts(
    attempt_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    """
    Resolve exact-copy fingerprints without discarding genuine restarts.

    Cross-session duplicate copies are resolved only when one candidate session has no
    nonduplicate attempts while another has distinct attempts. This is the pattern in
    the verified Feb 13 copy structure. Any less clear cross-session duplicate remains
    blocking rather than being assigned by folder order.
    """
    out = attempt_df.copy()
    out["duplicate_group_id"] = None
    out["duplicate_status"] = "UNIQUE"
    out["canonical_attempt_uid"] = out["attempt_uid"]
    out["excluded_as_duplicate_copy"] = False
    issues: list[dict[str, Any]] = []
    duplicate_rows: list[dict[str, Any]] = []

    duplicated = out.groupby("attempt_fingerprint")["attempt_uid"].transform("size") > 1
    duplicated_fps = set(out.loc[duplicated, "attempt_fingerprint"])
    cross_session_fps = set()
    for fingerprint, group in out.loc[duplicated].groupby("attempt_fingerprint", sort=True):
        sessions = set(group["mapped_session_id"].dropna())
        if len(sessions) > 1:
            cross_session_fps.add(fingerprint)

    non_crossdup_counts: dict[str, int] = {}
    for session_id, group in out[out["mapped_session_id"].notna()].groupby("mapped_session_id"):
        non_crossdup_counts[str(session_id)] = int((~group["attempt_fingerprint"].isin(cross_session_fps)).sum())

    group_counter = 0
    for fingerprint, group in out.loc[duplicated].groupby("attempt_fingerprint", sort=True):
        group_counter += 1
        duplicate_group_id = f"DUP{group_counter:03d}"
        sessions = sorted(set(group["mapped_session_id"].dropna()))
        canonical_session: Optional[str] = None
        resolution = ""

        if len(sessions) == 1:
            canonical_session = sessions[0]
            resolution = "SAME_SESSION_DUPLICATE_COPY"
        elif len(sessions) > 1:
            counts = {s: non_crossdup_counts.get(s, 0) for s in sessions}
            min_count = min(counts.values())
            min_sessions = [s for s, count in counts.items() if count == min_count]
            if min_count == 0 and len(min_sessions) == 1 and any(count > 0 for count in counts.values()):
                canonical_session = min_sessions[0]
                resolution = "CROSS_SESSION_COPY_HOME_HAS_NO_UNIQUE_ATTEMPTS"
            else:
                resolution = "AMBIGUOUS_CROSS_SESSION_DUPLICATE"
        else:
            resolution = "DUPLICATE_WITH_UNRESOLVED_SESSION"

        canonical_uid: Optional[str] = None
        if canonical_session is not None:
            candidates = group[group["mapped_session_id"] == canonical_session].copy()
            # Prefer the copy with richer annotation/video metadata; then stabilize by path.
            candidates["_annotation_ok"] = (candidates["annotation_total_size_bytes"].fillna(0) > 0).astype(int)
            candidates["_video_ok"] = (candidates["n_video_files"].fillna(0) > 0).astype(int)
            candidates = candidates.sort_values(
                ["_annotation_ok", "_video_ok", "stream_total_size_bytes", "attempt_path"],
                ascending=[False, False, False, True],
            )
            canonical_uid = str(candidates.iloc[0]["attempt_uid"])
            for idx in group.index:
                out.at[idx, "duplicate_group_id"] = duplicate_group_id
                out.at[idx, "canonical_attempt_uid"] = canonical_uid
                if str(out.at[idx, "attempt_uid"]) == canonical_uid:
                    out.at[idx, "duplicate_status"] = "CANONICAL_DUPLICATE"
                else:
                    out.at[idx, "duplicate_status"] = "EXCLUDED_DUPLICATE_COPY"
                    out.at[idx, "excluded_as_duplicate_copy"] = True

            issues.append({
                "severity": "WARNING",
                "code": "DUPLICATE_ATTEMPT_COPY_RESOLVED",
                "detail": f"{duplicate_group_id} contains {len(group)} structurally identical attempt copies; canonical copy is {canonical_uid} in {canonical_session} ({resolution}).",
                "Session_ID": canonical_session,
                "attempt_uid": canonical_uid,
            })
        else:
            for idx in group.index:
                out.at[idx, "duplicate_group_id"] = duplicate_group_id
                out.at[idx, "duplicate_status"] = "AMBIGUOUS_DUPLICATE_COPY"
                out.at[idx, "canonical_attempt_uid"] = None
                out.at[idx, "excluded_as_duplicate_copy"] = True
            issues.append({
                "severity": "ERROR",
                "code": "AMBIGUOUS_DUPLICATE_ATTEMPT_COPY",
                "detail": f"{duplicate_group_id} spans session candidates {sessions} and cannot be resolved conservatively. Review source folders before downstream use.",
            })

        for row in group.itertuples(index=False):
            duplicate_rows.append({
                "duplicate_group_id": duplicate_group_id,
                "attempt_fingerprint": fingerprint,
                "attempt_uid": row.attempt_uid,
                "attempt_path": row.attempt_path,
                "participant_batch_folder": row.participant_batch_folder,
                "mapped_session_id": row.mapped_session_id,
                "sensor_short_ids": row.sensor_short_ids,
                "canonical_attempt_uid": canonical_uid,
                "is_canonical_copy": bool(canonical_uid is not None and str(row.attempt_uid) == canonical_uid),
                "resolution": resolution,
            })

    duplicate_df = pd.DataFrame(duplicate_rows)
    return out, duplicate_df, issues


def assign_attempt_phases(
    attempt_df: pd.DataFrame,
    runsheet: pd.DataFrame,
    expected_phases: Sequence[ExpectedPhase],
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    """
    Assign canonical attempts to Phase 1/2 within each validated Session_ID.

    Within a session, the two phase clusters are separated at the largest adjacent
    attempt-start gap. The split is accepted only when that gap is at least 10 minutes
    and the sensors in each resulting cluster are compatible with that phase's final
    runsheet assignments. This accommodates JM/MCL rooms that started several minutes
    apart and does not require their recordings to overlap.
    """
    out = attempt_df.copy()
    out["phase_anchor_start"] = None
    out["phase_anchor_basis"] = None
    out["mapped_phase"] = None
    out["mapped_condition"] = None
    out["phase_group_id"] = None
    out["phase_mapping_status"] = "NOT_PHASE_MAPPED"
    out["phase_split_gap_s"] = None
    out["within_phase_max_start_gap_s"] = None

    issues: list[dict[str, Any]] = []
    phase_rows: list[dict[str, Any]] = []
    expected_lookup = {(x.session_id, x.phase): x for x in expected_phases}

    # Precompute chronological anchors for all attempts.
    for idx, row in out.iterrows():
        start, _stop, basis = choose_interval(row)
        out.at[idx, "phase_anchor_start"] = start
        out.at[idx, "phase_anchor_basis"] = basis

    for session_id, rs_group in runsheet.groupby("Session_ID", sort=True):
        expected_any = bool((rs_group["Raw_Recording_Expectation"] == "EXPECTED_RECORDING").any())
        canonical = out[
            (out["mapped_session_id"] == session_id)
            & (~out["excluded_as_duplicate_copy"].fillna(False))
            & (out["n_sensors"] > 0)
        ].copy()

        if not expected_any:
            if not canonical.empty:
                out.loc[canonical.index, "phase_mapping_status"] = "UNMAPPED_DOCUMENTED_NO_SENSOR_SESSION"
                issues.append({
                    "severity": "INFO",
                    "code": "RAW_ATTEMPTS_IN_DOCUMENTED_NO_SENSOR_SESSION",
                    "detail": f"{session_id} has {len(canonical)} sensor-bearing raw attempts, but all runsheet rows document no expected raw sensor data. They remain inventoried and are excluded from Script 02 by default.",
                    "Session_ID": session_id,
                })
            continue

        if canonical.empty:
            # Participant-level reconciliation determines whether this is truly blocking.
            continue

        timed = canonical[canonical["phase_anchor_start"].notna()].copy()
        if len(timed) != len(canonical):
            missing_uids = canonical.loc[canonical["phase_anchor_start"].isna(), "attempt_uid"].tolist()
            issues.append({
                "severity": "ERROR",
                "code": "ATTEMPT_WITHOUT_CHRONOLOGY_ANCHOR",
                "detail": f"{session_id} has canonical attempts without a usable timing anchor: {missing_uids}.",
                "Session_ID": session_id,
            })
            continue

        timed = timed.sort_values(["phase_anchor_start", "attempt_uid"])
        if len(timed) < 2:
            issues.append({
                "severity": "ERROR",
                "code": "INSUFFICIENT_ATTEMPTS_FOR_TWO_PHASE_SPLIT",
                "detail": f"{session_id} has only {len(timed)} canonical timed attempt(s); two phases cannot be assigned conservatively.",
                "Session_ID": session_id,
            })
            continue

        starts = timed["phase_anchor_start"].astype(float).to_numpy()
        gaps = [float(starts[i + 1] - starts[i]) for i in range(len(starts) - 1)]
        split_index = int(max(range(len(gaps)), key=lambda i: gaps[i]))
        split_gap = gaps[split_index]
        if split_gap < MIN_PHASE_SPLIT_GAP_SECONDS:
            issues.append({
                "severity": "ERROR",
                "code": "PHASE_SPLIT_GAP_TOO_SMALL",
                "detail": f"{session_id} largest adjacent attempt-start gap is {split_gap:.1f}s, below the prespecified {MIN_PHASE_SPLIT_GAP_SECONDS:.0f}s minimum.",
                "Session_ID": session_id,
            })
            continue

        phase_indices = {
            1: timed.iloc[: split_index + 1].index,
            2: timed.iloc[split_index + 1 :].index,
        }

        session_failed = False
        phase_diagnostics: dict[int, dict[str, Any]] = {}
        for phase, indices in phase_indices.items():
            group = out.loc[indices].copy()
            expected = expected_lookup[(session_id, phase)]
            observed_sensors: set[str] = set()
            for value in group["sensor_short_ids"]:
                observed_sensors.update(split_sensor_set(value))

            extra = observed_sensors - set(expected.all_sensors)
            if extra:
                session_failed = True
                issues.append({
                    "severity": "ERROR",
                    "code": "PHASE_CLUSTER_SENSOR_INCOMPATIBLE",
                    "detail": f"{session_id} Phase {phase} chronology cluster contains sensors {sorted(observed_sensors)}; unexpected for runsheet phase: {sorted(extra)}.",
                    "Session_ID": session_id,
                    "Phase": phase,
                })

            phase_starts = sorted(float(x) for x in group["phase_anchor_start"])
            within_gaps = [phase_starts[i + 1] - phase_starts[i] for i in range(len(phase_starts) - 1)]
            max_within = max(within_gaps) if within_gaps else 0.0
            phase_diagnostics[phase] = {
                "indices": indices,
                "group": group,
                "observed_sensors": observed_sensors,
                "max_within": max_within,
            }

        if session_failed:
            continue

        for phase, info in phase_diagnostics.items():
            indices = info["indices"]
            expected = expected_lookup[(session_id, phase)]
            out.loc[indices, "mapped_phase"] = phase
            out.loc[indices, "mapped_condition"] = expected.condition
            out.loc[indices, "phase_group_id"] = f"{session_id}_P{phase}"
            out.loc[indices, "phase_mapping_status"] = "MATCHED_BATCH_SESSION_CHRONOLOGY"
            out.loc[indices, "phase_split_gap_s"] = split_gap
            out.loc[indices, "within_phase_max_start_gap_s"] = info["max_within"]

            group = info["group"]
            phase_rows.append({
                "phase_group_id": f"{session_id}_P{phase}",
                "Session_ID": session_id,
                "Phase": phase,
                "Condition": expected.condition,
                "Date": expected.date.isoformat(),
                "n_attempts": int(len(group)),
                "attempt_uids": compact_join(group["attempt_uid"]),
                "attempt_paths": compact_join(group["attempt_path"]),
                "collection_folders": compact_join(group["collection_folder"]),
                "collection_ipad_labels": compact_join(group["collection_ipad_label"]),
                "observed_sensors": ";".join(sorted(info["observed_sensors"])),
                "expected_all_sensors": ";".join(sorted(expected.all_sensors)),
                "expected_active_sensors": ";".join(sorted(expected.active_sensors)),
                "phase_start_min": min(float(x) for x in group["phase_anchor_start"]),
                "phase_start_max": max(float(x) for x in group["phase_anchor_start"]),
                "phase_start_min_utc": epoch_to_iso(min(float(x) for x in group["phase_anchor_start"])),
                "phase_start_max_utc": epoch_to_iso(max(float(x) for x in group["phase_anchor_start"])),
                "phase_split_gap_s": split_gap,
                "within_phase_max_start_gap_s": info["max_within"],
                "mapping_status": "MATCHED_BATCH_SESSION_CHRONOLOGY",
            })

    # Any canonical sensor-bearing attempt in a session expected to contain raw data
    # must receive a phase mapping before production can proceed.
    expected_sessions = set(
        runsheet.loc[runsheet["Raw_Recording_Expectation"] == "EXPECTED_RECORDING", "Session_ID"]
    )
    unresolved = out[
        (~out["excluded_as_duplicate_copy"].fillna(False))
        & (out["n_sensors"] > 0)
        & (out["mapped_session_id"].isin(expected_sessions))
        & (out["mapped_phase"].isna())
    ]
    for row in unresolved.itertuples(index=False):
        issues.append({
            "severity": "ERROR",
            "code": "CANONICAL_ATTEMPT_PHASE_UNRESOLVED",
            "detail": f"{row.attempt_path} mapped to {row.mapped_session_id} but not to a phase.",
            "Session_ID": row.mapped_session_id,
            "attempt_uid": row.attempt_uid,
            "raw_path": row.attempt_path,
        })

    phase_df = pd.DataFrame(phase_rows)
    if not phase_df.empty:
        phase_df = phase_df.sort_values(["Session_ID", "Phase"]).reset_index(drop=True)
    return out, phase_df, issues


# -----------------------------------------------------------------------------
# Sensor-level canonical manifest and participant-phase reconciliation
# -----------------------------------------------------------------------------


def build_sensor_attempt_table(
    file_df: pd.DataFrame,
    attempt_mapping: pd.DataFrame,
    runsheet: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Build the canonical one-row-per-sensor-recording manifest for Script 02."""
    if file_df.empty:
        return pd.DataFrame(), []

    issues: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    stream_order = [
        "ECG", "Acc", "HeartRate", "HR", "KaasaHR", "MotionIntensity",
        "GyroFeatures", "AccFeatures", "Gyro", "Lactate",
    ]

    for (attempt_uid, sensor_full, sensor_short), group in file_df.groupby(
        ["attempt_uid", "sensor_full", "sensor_short"], sort=True, dropna=False
    ):
        rec: dict[str, Any] = {
            "attempt_uid": attempt_uid,
            "sensor_full": str(sensor_full),
            "sensor_short": str(sensor_short),
            "stream_types": ";".join(sorted(set(group["stream_type"].astype(str)))),
            "n_stream_files": int(group["file_path"].nunique()),
            "stream_paths": ";".join(sorted(set(group["file_path"].astype(str)))),
            "stream_schema_statuses": ";".join(sorted(set(group["schema_status"].astype(str)))),
            "stream_total_size_bytes": int(group["file_size_bytes"].fillna(0).sum()),
        }
        for stream in stream_order:
            g = group[group["stream_type"] == stream]
            rec[f"{stream}_path"] = ";".join(sorted(set(g["file_path"].astype(str)))) if not g.empty else None

        ecg = group[group["stream_type"] == "ECG"]
        if ecg.empty:
            rec.update({
                "ECG_file_size_bytes": 0,
                "ECG_first_timestamp": None,
                "ECG_last_timestamp": None,
                "ECG_schema_status": "MISSING",
                "ECG_time_envelope_ok": False,
                "ECG_inventory_ok": False,
            })
        else:
            ecg_size = int(ecg["file_size_bytes"].fillna(0).sum())
            ecg_statuses = sorted(set(ecg["schema_status"].astype(str)))
            first = float(ecg["first_timestamp"].dropna().min()) if ecg["first_timestamp"].notna().any() else None
            last = float(ecg["last_timestamp"].dropna().max()) if ecg["last_timestamp"].notna().any() else None
            ecg_envelope_ok = bool(first is not None and last is not None and last >= first)
            # Script 01 should not discard an otherwise readable ECG merely because a
            # final line was interrupted. Script 02 will parse the signal row-by-row.
            # A nonempty file with a recognized timestamp schema and valid first
            # timestamp is therefore processable; envelope completeness is reported
            # separately.
            # Inventory usability requires a nonempty ECG whose timestamp column
            # and first timestamp can be recognized. A missing trustworthy tail
            # timestamp is an audit/QC warning, not a reason to discard the signal:
            # Script 02 will stream-parse the file and determine the valid extent.
            ecg_rows_processable = ecg.apply(
                lambda r: (
                    int(r.get("file_size_bytes") or 0) > 0
                    and not is_missing(r.get("timestamp_column"))
                    and not is_missing(r.get("first_timestamp"))
                    and math.isfinite(float(r.get("first_timestamp")))
                ),
                axis=1,
            )
            ecg_ok = bool(ecg_rows_processable.any())
            rec.update({
                "ECG_file_size_bytes": ecg_size,
                "ECG_first_timestamp": first,
                "ECG_last_timestamp": last,
                "ECG_schema_status": ";".join(ecg_statuses),
                "ECG_time_envelope_ok": ecg_envelope_ok,
                "ECG_inventory_ok": ecg_ok,
            })
        records.append(rec)

    sensor_df = pd.DataFrame(records)
    mapping_cols = [
        "attempt_uid", "attempt_path", "source_path", "participant_batch_folder",
        "collection_root_folder", "collection_folder", "collection_ipad_label",
        "collection_folder_has_back_to_back_2_suffix", "raw_date", "n_annotation_files",
        "annotation_files", "annotation_total_size_bytes", "n_video_files", "video_files",
        "video_total_size_bytes", "video_duration_s", "pause_or_resume_within_video",
        "recording_annotation_status", "video_annotation_status", "time_block_id",
        "mapped_session_id", "session_mapping_status", "mapped_phase", "mapped_condition",
        "phase_group_id", "phase_mapping_status", "phase_split_gap_s",
        "within_phase_max_start_gap_s", "attempt_fingerprint", "duplicate_group_id",
        "duplicate_status", "canonical_attempt_uid", "excluded_as_duplicate_copy",
    ]
    mapping_cols = [c for c in mapping_cols if c in attempt_mapping.columns]
    sensor_df = sensor_df.merge(
        attempt_mapping[mapping_cols], on="attempt_uid", how="left", validate="many_to_one"
    )

    # Authoritative sensor -> PID mapping after session and phase are established.
    sensor_lookup: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in runsheet.itertuples(index=False):
        for sensor in parse_allowed_sensors(row.Allowed_Sensors):
            sensor_lookup[(row.Session_ID, int(row.Phase), sensor)].append({
                "PID": row.PID,
                "Analysis_Partner": row.Analysis_Partner,
                "Analysis_Role": row.Analysis_Role,
                "Raw_Recording_Expectation": row.Raw_Recording_Expectation,
                "Expected_Sensor": row.Sensor,
            })

    mapped_pids: list[Optional[str]] = []
    partners: list[Optional[str]] = []
    roles: list[Optional[str]] = []
    expectations: list[Optional[str]] = []
    assignment_statuses: list[str] = []

    for row in sensor_df.itertuples(index=False):
        if bool(row.excluded_as_duplicate_copy):
            candidates: list[dict[str, Any]] = []
            status = "EXCLUDED_DUPLICATE_COPY"
        elif is_missing(row.mapped_session_id) or is_missing(row.mapped_phase):
            candidates = []
            status = "SESSION_OR_PHASE_UNRESOLVED"
        else:
            key = (str(row.mapped_session_id), int(row.mapped_phase), str(row.sensor_short))
            candidates = sensor_lookup.get(key, [])
            status = "MATCHED_UNIQUE_PID" if len(candidates) == 1 else (
                "NO_PID_FOR_SENSOR_IN_SESSION_PHASE" if len(candidates) == 0 else "AMBIGUOUS_PID_FOR_SENSOR_IN_SESSION_PHASE"
            )

        if len(candidates) == 1:
            c = candidates[0]
            mapped_pids.append(c["PID"])
            partners.append(c["Analysis_Partner"])
            roles.append(c["Analysis_Role"])
            expectations.append(c["Raw_Recording_Expectation"])
        else:
            mapped_pids.append(None)
            partners.append(None)
            roles.append(None)
            expectations.append(None)
            if status in {"NO_PID_FOR_SENSOR_IN_SESSION_PHASE", "AMBIGUOUS_PID_FOR_SENSOR_IN_SESSION_PHASE"}:
                issues.append({
                    "severity": "ERROR",
                    "code": status,
                    "detail": f"Attempt {row.attempt_uid}, sensor {row.sensor_short}, {row.mapped_session_id} Phase {row.mapped_phase} cannot be mapped uniquely to a runsheet PID.",
                    "Session_ID": row.mapped_session_id,
                    "Phase": row.mapped_phase,
                    "attempt_uid": row.attempt_uid,
                    "sensor_short": row.sensor_short,
                })
        assignment_statuses.append(status)

    sensor_df["mapped_pid"] = mapped_pids
    sensor_df["analysis_partner"] = partners
    sensor_df["analysis_role"] = roles
    sensor_df["raw_recording_expectation"] = expectations
    sensor_df["sensor_assignment_status"] = assignment_statuses
    sensor_df["Script02_process_candidate"] = (
        (~sensor_df["excluded_as_duplicate_copy"].fillna(False))
        & (sensor_df["sensor_assignment_status"] == "MATCHED_UNIQUE_PID")
        & (sensor_df["raw_recording_expectation"] == "EXPECTED_RECORDING")
        & sensor_df["ECG_inventory_ok"].fillna(False)
    )
    sensor_df["PrimarySynchrony_candidate"] = (
        sensor_df["Script02_process_candidate"]
        & (sensor_df["analysis_role"] == "ACTIVE_DYAD")
    )
    return sensor_df, issues


def reconcile_participant_phases(
    runsheet: pd.DataFrame,
    sensor_attempt_df: pd.DataFrame,
    in_scope_sessions: set[str],
    scope_mode: str,
) -> pd.DataFrame:
    """Create the 216-row expected-vs-observed participant-phase master table."""
    records: list[dict[str, Any]] = []

    for row in runsheet.itertuples(index=False):
        session_in_scope = scope_mode == "all" or row.Session_ID in in_scope_sessions
        if not session_in_scope or sensor_attempt_df.empty:
            matches = pd.DataFrame()
        else:
            matches = sensor_attempt_df[
                (~sensor_attempt_df["excluded_as_duplicate_copy"].fillna(False))
                & (sensor_attempt_df["mapped_session_id"] == row.Session_ID)
                & (sensor_attempt_df["mapped_phase"] == row.Phase)
                & (sensor_attempt_df["mapped_pid"] == row.PID)
            ].copy()

        n_recordings = int(matches["attempt_uid"].nunique()) if not matches.empty else 0
        ecg_ok_matches = matches[matches["ECG_inventory_ok"].fillna(False)] if not matches.empty else matches
        n_ecg_ok = int(ecg_ok_matches["attempt_uid"].nunique()) if not ecg_ok_matches.empty else 0
        raw_expectation = row.Raw_Recording_Expectation

        if not session_in_scope:
            status = "OUT_OF_RAW_INPUT_SCOPE"
        elif row.Sensor is None:
            status = "NO_SENSOR_ASSIGNED_IN_RUNSHEET"
        elif n_recordings == 0 and raw_expectation != "EXPECTED_RECORDING":
            status = "DOCUMENTED_NO_RAW_RECORDING"
        elif n_recordings == 0:
            status = "MISSING_RAW_RECORDING"
        elif raw_expectation != "EXPECTED_RECORDING":
            status = "FOUND_RAW_RECORDING_DESPITE_DOCUMENTED_EXCEPTION"
        elif n_recordings > 1:
            status = "FOUND_MULTIPLE_RAW_RECORDINGS"
        else:
            status = "FOUND_ONE_RAW_RECORDING"

        stream_union: set[str] = set()
        for text in matches.get("stream_types", pd.Series(dtype=object)):
            stream_union.update(x for x in str(text).split(";") if x and x != "nan")

        condition = row.Condition
        records.append({
            "Session_ID": row.Session_ID,
            "PID": row.PID,
            "Phase": row.Phase,
            "Condition": condition,
            "Story_Type": "Neutral" if condition.startswith("N") else "Emotional",
            "Fruit": "Apple" if condition.endswith("A") else "Banana",
            "Planned_Partner": row.Planned_Partner,
            "Analysis_Partner": row.Analysis_Partner,
            "Date": row.Date.isoformat(),
            "Expected_Sensor": row.Sensor,
            "Allowed_Sensors": row.Allowed_Sensors,
            "documented_sensor_switch": len(parse_allowed_sensors(row.Allowed_Sensors)) > 1,
            "Raw_Recording_Expectation": raw_expectation,
            "Analysis_Role": row.Analysis_Role,
            "Participant_Documented_Absent": bool(row.Participant_Documented_Absent),
            "Notes": row.Notes,
            "Sensor_Note": getattr(row, "Sensor_Note", None),
            "Paper_Source": getattr(row, "Paper_Source", None),
            "reconciliation_status": status,
            "n_raw_recordings": n_recordings,
            "n_ecg_inventory_ok_recordings": n_ecg_ok,
            "raw_attempt_uids": compact_join(matches.get("attempt_uid", [])),
            "raw_attempt_paths": compact_join(matches.get("attempt_path", [])),
            "raw_source_paths": compact_join(matches.get("source_path", [])),
            "raw_sensor_full_ids": compact_join(matches.get("sensor_full", [])),
            "available_streams": ";".join(sorted(stream_union)) if stream_union else None,
            "ECG_present": "ECG" in stream_union,
            "ECG_inventory_ok": bool(n_ecg_ok > 0),
            "Acc_present": "Acc" in stream_union,
            "HeartRate_present": "HeartRate" in stream_union,
            "HR_present": "HR" in stream_union,
            "KaasaHR_present": "KaasaHR" in stream_union,
            "MotionIntensity_present": "MotionIntensity" in stream_union,
            "GyroFeatures_present": "GyroFeatures" in stream_union,
            "annotation_files": compact_join(matches.get("annotation_files", [])),
            "raw_video_files": compact_join(matches.get("video_files", [])),
            "recording_annotation_statuses": compact_join(matches.get("recording_annotation_status", [])),
            "video_annotation_statuses": compact_join(matches.get("video_annotation_status", [])),
            "Script02_ECG_candidate": bool(raw_expectation == "EXPECTED_RECORDING" and n_ecg_ok > 0),
            "PrimarySynchrony_candidate": bool(row.Analysis_Role == "ACTIVE_DYAD" and raw_expectation == "EXPECTED_RECORDING" and n_ecg_ok > 0),
        })

    return pd.DataFrame(records)


def build_issue_table(
    base_issues: list[dict[str, Any]],
    attempt_mapping: pd.DataFrame,
    reconciliation: pd.DataFrame,
) -> pd.DataFrame:
    """Build a compact, severity-aware review queue for the production gate."""
    issues = list(base_issues)

    def add_issue(severity: str, code: str, detail: str, **context: Any) -> None:
        record = {"severity": severity, "code": code, "detail": detail}
        record.update(context)
        issues.append(record)

    for row in reconciliation.itertuples(index=False):
        if row.reconciliation_status == "NO_SENSOR_ASSIGNED_IN_RUNSHEET":
            add_issue(
                "INFO", "NO_SENSOR_ASSIGNED_IN_FINAL_RUNSHEET",
                f"{row.Session_ID} PID {row.PID} Phase {row.Phase} has no sensor assignment in the final runsheet.",
                Session_ID=row.Session_ID, PID=row.PID, Phase=row.Phase,
            )
        elif row.reconciliation_status == "DOCUMENTED_NO_RAW_RECORDING":
            add_issue(
                "INFO", "DOCUMENTED_RAW_ABSENCE_CONFIRMED",
                f"{row.Session_ID} PID {row.PID} Phase {row.Phase} has no mapped raw recording, consistent with {row.Raw_Recording_Expectation}.",
                Session_ID=row.Session_ID, PID=row.PID, Phase=row.Phase,
            )
        elif row.reconciliation_status == "FOUND_RAW_RECORDING_DESPITE_DOCUMENTED_EXCEPTION":
            add_issue(
                "WARNING", "RAW_RECORDING_PRESENT_DESPITE_DOCUMENTED_EXCEPTION",
                f"{row.Session_ID} PID {row.PID} Phase {row.Phase} has raw data although the runsheet classifies it as {row.Raw_Recording_Expectation}; it is excluded from Script 02 by default.",
                Session_ID=row.Session_ID, PID=row.PID, Phase=row.Phase,
            )
        elif row.reconciliation_status == "MISSING_RAW_RECORDING":
            severity = "ERROR" if row.Analysis_Role == "ACTIVE_DYAD" else "WARNING"
            code = "ACTIVE_EXPECTED_RAW_RECORDING_NOT_FOUND" if severity == "ERROR" else "NONANALYSIS_EXPECTED_RAW_RECORDING_NOT_FOUND"
            add_issue(
                severity, code,
                f"{row.Session_ID} PID {row.PID} Phase {row.Phase}, expected sensor {row.Expected_Sensor}, has no mapped raw recording.",
                Session_ID=row.Session_ID, PID=row.PID, Phase=row.Phase,
            )
        elif row.reconciliation_status == "FOUND_MULTIPLE_RAW_RECORDINGS":
            add_issue(
                "WARNING", "MULTIPLE_NONIDENTICAL_RAW_RECORDINGS_FOR_PARTICIPANT_PHASE",
                f"{row.Session_ID} PID {row.PID} Phase {row.Phase} has {row.n_raw_recordings} nonduplicate recordings. All remain Script 02 candidates; do not select one until ECG/task QC.",
                Session_ID=row.Session_ID, PID=row.PID, Phase=row.Phase,
            )

        if (
            row.Raw_Recording_Expectation == "EXPECTED_RECORDING"
            and row.Analysis_Role == "ACTIVE_DYAD"
            and row.reconciliation_status.startswith("FOUND")
            and not bool(row.ECG_inventory_ok)
        ):
            add_issue(
                "ERROR", "ACTIVE_EXPECTED_ECG_NOT_INVENTORY_USABLE",
                f"{row.Session_ID} PID {row.PID} Phase {row.Phase} has raw data but no nonempty ECG with recognized timestamp schema/envelope.",
                Session_ID=row.Session_ID, PID=row.PID, Phase=row.Phase,
            )

    # Annotation quality does not block raw ECG processing, but surface phase-level
    # warnings where an active expected participant has no attempt with a standard video interval.
    for row in reconciliation.itertuples(index=False):
        if row.Analysis_Role != "ACTIVE_DYAD" or row.Raw_Recording_Expectation != "EXPECTED_RECORDING":
            continue
        statuses = set(str(row.video_annotation_statuses).split(";")) if not is_missing(row.video_annotation_statuses) else set()
        if row.reconciliation_status.startswith("FOUND") and "OK_SINGLE_VIDEO_INTERVAL" not in statuses:
            add_issue(
                "WARNING", "ACTIVE_PARTICIPANT_PHASE_WITHOUT_STANDARD_VIDEO_ANNOTATION",
                f"{row.Session_ID} PID {row.PID} Phase {row.Phase} has ECG data but no mapped attempt with a standard VideoAnnotationStart/Stop interval. Resolve during epoch segmentation using runsheet/source review.",
                Session_ID=row.Session_ID, PID=row.PID, Phase=row.Phase,
            )

    columns = [
        "severity", "code", "detail", "Session_ID", "PID", "Phase", "attempt_uid",
        "raw_date", "raw_path", "sensor_short", "raw_expectation",
    ]
    if not issues:
        return pd.DataFrame(columns=columns)
    frame = pd.DataFrame(issues)
    for col in columns:
        if col not in frame.columns:
            frame[col] = None
    severity_rank = {"ERROR": 0, "WARNING": 1, "INFO": 2}
    frame["_severity_rank"] = frame["severity"].map(severity_rank).fillna(9)
    frame = frame.sort_values(
        ["_severity_rank", "code", "Session_ID", "PID", "Phase", "attempt_uid"],
        na_position="last",
    ).drop(columns="_severity_rank").reset_index(drop=True)
    return frame


def build_session_phase_summary(
    runsheet: pd.DataFrame,
    expected_phases: Sequence[ExpectedPhase],
    attempt_mapping: pd.DataFrame,
    sensor_df: pd.DataFrame,
    reconciliation: pd.DataFrame,
) -> pd.DataFrame:
    """Create a concise 54-row session-phase audit table."""
    rows: list[dict[str, Any]] = []
    for expected in expected_phases:
        rs = runsheet[(runsheet["Session_ID"] == expected.session_id) & (runsheet["Phase"] == expected.phase)]
        rec = reconciliation[(reconciliation["Session_ID"] == expected.session_id) & (reconciliation["Phase"] == expected.phase)]
        attempts = attempt_mapping[
            (~attempt_mapping["excluded_as_duplicate_copy"].fillna(False))
            & (attempt_mapping["mapped_session_id"] == expected.session_id)
            & (attempt_mapping["mapped_phase"] == expected.phase)
        ]
        sensors = sensor_df[
            (~sensor_df["excluded_as_duplicate_copy"].fillna(False))
            & (sensor_df["mapped_session_id"] == expected.session_id)
            & (sensor_df["mapped_phase"] == expected.phase)
        ] if not sensor_df.empty else pd.DataFrame()

        pair_labels = sorted("+".join(sorted(pair)) for pair in expected.active_pairs)
        observed_sensors = sorted(set(sensors["sensor_short"].astype(str))) if not sensors.empty else []
        split_gaps = attempts["phase_split_gap_s"].dropna().astype(float) if not attempts.empty else pd.Series(dtype=float)
        rows.append({
            "Session_ID": expected.session_id,
            "Phase": expected.phase,
            "Date": expected.date.isoformat(),
            "Condition": expected.condition,
            "Story_Type": "Neutral" if expected.condition.startswith("N") else "Emotional",
            "Fruit": "Apple" if expected.condition.endswith("A") else "Banana",
            "n_participant_rows": int(len(rs)),
            "n_active_dyad_participants": int(rs["Analysis_Partner"].notna().sum()),
            "expected_all_sensors": ";".join(sorted(expected.all_sensors)),
            "expected_active_sensors": ";".join(sorted(expected.active_sensors)),
            "expected_active_sensor_pairs": ";".join(pair_labels),
            "n_canonical_attempts": int(len(attempts)),
            "canonical_attempt_uids": compact_join(attempts.get("attempt_uid", [])),
            "canonical_attempt_paths": compact_join(attempts.get("attempt_path", [])),
            "collection_ipad_labels": compact_join(attempts.get("collection_ipad_label", [])),
            "observed_sensors": ";".join(observed_sensors),
            "phase_mapping_statuses": compact_join(attempts.get("phase_mapping_status", [])),
            "phase_split_gap_s": float(split_gaps.iloc[0]) if not split_gaps.empty else None,
            "n_script02_candidate_participant_phases": int(rec.get("Script02_ECG_candidate", pd.Series(dtype=bool)).sum()),
            "n_primary_synchrony_candidate_participants": int(rec.get("PrimarySynchrony_candidate", pd.Series(dtype=bool)).sum()),
            "n_active_expected_missing_raw": int(((rec.get("Analysis_Role", pd.Series(dtype=object)) == "ACTIVE_DYAD") & (rec.get("Raw_Recording_Expectation", pd.Series(dtype=object)) == "EXPECTED_RECORDING") & (rec.get("reconciliation_status", pd.Series(dtype=object)) == "MISSING_RAW_RECORDING")).sum()),
            "n_active_expected_without_inventory_ok_ecg": int(((rec.get("Analysis_Role", pd.Series(dtype=object)) == "ACTIVE_DYAD") & (rec.get("Raw_Recording_Expectation", pd.Series(dtype=object)) == "EXPECTED_RECORDING") & (~rec.get("ECG_inventory_ok", pd.Series(dtype=bool)).fillna(False))).sum()),
            "n_multiple_recording_participant_phases": int((rec.get("reconciliation_status", pd.Series(dtype=object)) == "FOUND_MULTIPLE_RAW_RECORDINGS").sum()),
            "video_annotation_statuses": compact_join(attempts.get("video_annotation_status", [])),
        })
    return pd.DataFrame(rows).sort_values(["Session_ID", "Phase"]).reset_index(drop=True)


# -----------------------------------------------------------------------------
# Output and main entry point
# -----------------------------------------------------------------------------


def write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, encoding="utf-8", na_rep="")


def make_summary(
    runsheet: pd.DataFrame,
    study_timezone: str,
    scope_mode: str,
    project_root: Path,
    raw_root: Path,
    runsheet_path: Path,
    file_df: pd.DataFrame,
    annotation_df: pd.DataFrame,
    attempt_df: pd.DataFrame,
    time_block_df: pd.DataFrame,
    duplicate_df: pd.DataFrame,
    phase_df: pd.DataFrame,
    sensor_df: pd.DataFrame,
    reconciliation: pd.DataFrame,
    issues: pd.DataFrame,
    runtime_seconds: float,
) -> dict[str, Any]:
    script_path = Path(__file__).resolve()
    return {
        "script": script_path.name,
        "script_version": SCRIPT_VERSION,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": round(float(runtime_seconds), 3),
        "scope_mode": scope_mode,
        "study_timezone": study_timezone,
        "environment": {
            "python": platform.python_version(), "platform": platform.platform(),
            "pandas": pd.__version__, "openpyxl": getattr(openpyxl, "__version__", None),
        },
        "provenance": {
            "script_sha256": sha256_file(script_path) if script_path.exists() else None,
            "runsheet_sha256": sha256_file(runsheet_path),
        },
        "paths": {
            "project_root": project_root.as_posix(), "raw_root": raw_root.as_posix(),
            "runsheet": runsheet_path.as_posix(),
        },
        "runsheet": {
            "participant_phase_rows": int(len(runsheet)),
            "unique_pids": int(runsheet["PID"].nunique()),
            "unique_sessions": int(runsheet["Session_ID"].nunique()),
            "expected_recording_rows": int((runsheet["Raw_Recording_Expectation"] == "EXPECTED_RECORDING").sum()),
            "active_expected_rows": int(((runsheet["Raw_Recording_Expectation"] == "EXPECTED_RECORDING") & (runsheet["Analysis_Role"] == "ACTIVE_DYAD")).sum()),
            "documented_absent_rows": int((runsheet["Raw_Recording_Expectation"] == "DOCUMENTED_PARTICIPANT_ABSENT").sum()),
            "documented_sensor_unavailable_rows": int((runsheet["Raw_Recording_Expectation"] == "DOCUMENTED_SENSOR_UNAVAILABLE").sum()),
            "no_sensor_assigned_rows": int((runsheet["Raw_Recording_Expectation"] == "NO_SENSOR_ASSIGNED").sum()),
        },
        "raw_inventory": {
            "stream_files": int(len(file_df)),
            "stream_bytes": int(file_df["file_size_bytes"].fillna(0).sum()) if not file_df.empty else 0,
            "annotation_events": int(len(annotation_df)),
            "recording_attempts": int(len(attempt_df)),
            "sensor_bearing_attempts": int((attempt_df["n_sensors"] > 0).sum()),
            "video_files": int(attempt_df["n_video_files"].sum()),
            "video_bytes": int(attempt_df["video_total_size_bytes"].fillna(0).sum()),
            "time_blocks_diagnostic_only": int(len(time_block_df)),
        },
        "mapping": {
            "session_mapping_counts": {str(k): int(v) for k, v in attempt_df["session_mapping_status"].value_counts(dropna=False).items()},
            "phase_mapping_counts": {str(k): int(v) for k, v in attempt_df["phase_mapping_status"].value_counts(dropna=False).items()},
            "mapped_phase_groups": int(len(phase_df)),
            "duplicate_groups": int(duplicate_df["duplicate_group_id"].nunique()) if not duplicate_df.empty else 0,
            "excluded_duplicate_copies": int(attempt_df["excluded_as_duplicate_copy"].fillna(False).sum()),
        },
        "participant_phase_reconciliation": {str(k): int(v) for k, v in reconciliation["reconciliation_status"].value_counts(dropna=False).items()},
        "script02": {
            "candidate_sensor_recordings": int(sensor_df["Script02_process_candidate"].sum()) if not sensor_df.empty else 0,
            "candidate_participant_phases": int(reconciliation["Script02_ECG_candidate"].sum()),
            "primary_synchrony_candidate_participant_phases": int(reconciliation["PrimarySynchrony_candidate"].sum()),
        },
        "issues": {
            "total": int(len(issues)),
            "errors": int((issues["severity"] == "ERROR").sum()) if not issues.empty else 0,
            "warnings": int((issues["severity"] == "WARNING").sum()) if not issues.empty else 0,
            "info": int((issues["severity"] == "INFO").sum()) if not issues.empty else 0,
        },
    }


def format_bytes(n_bytes: int) -> str:
    value = float(n_bytes)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(value) < 1024.0 or unit == "TB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} TB"


def write_summary_text(summary: dict[str, Any], path: Path) -> None:
    inv = summary["raw_inventory"]
    mapping = summary["mapping"]
    issues = summary["issues"]
    lines = [
        "KITCO3 Script 01 - Raw Inventory and Reconciliation",
        f"Version: {summary['script_version']}",
        f"Generated UTC: {summary['generated_utc']}",
        f"Runtime: {summary['runtime_seconds']:.1f} s",
        f"Scope mode: {summary['scope_mode']}",
        "",
        f"Runsheet: {summary['runsheet']['participant_phase_rows']} participant-phase rows, {summary['runsheet']['unique_pids']} PIDs, {summary['runsheet']['unique_sessions']} sessions",
        f"Raw attempts: {inv['recording_attempts']} ({inv['sensor_bearing_attempts']} sensor-bearing)",
        f"Stream files: {inv['stream_files']} ({format_bytes(inv['stream_bytes'])})",
        f"Video files: {inv['video_files']} ({format_bytes(inv['video_bytes'])})",
        f"Diagnostic overlap blocks: {inv['time_blocks_diagnostic_only']}",
        f"Mapped observed session-phases: {mapping['mapped_phase_groups']}",
        f"Duplicate groups: {mapping['duplicate_groups']} ({mapping['excluded_duplicate_copies']} copies excluded)",
        "",
        "Session mapping counts:",
    ]
    for key, value in sorted(mapping["session_mapping_counts"].items()):
        lines.append(f"  {key}: {value}")
    lines.append("Phase mapping counts:")
    for key, value in sorted(mapping["phase_mapping_counts"].items()):
        lines.append(f"  {key}: {value}")
    lines.extend(["", "Participant-phase reconciliation counts:"])
    for key, value in sorted(summary["participant_phase_reconciliation"].items()):
        lines.append(f"  {key}: {value}")
    lines.extend([
        "",
        f"Script 02 candidate sensor recordings: {summary['script02']['candidate_sensor_recordings']}",
        f"Script 02 candidate participant-phases: {summary['script02']['candidate_participant_phases']}",
        f"Primary synchrony candidate participant-phases: {summary['script02']['primary_synchrony_candidate_participant_phases']}",
        "",
        f"Issues: {issues['errors']} ERROR, {issues['warnings']} WARNING, {issues['info']} INFO",
        "",
        "Production gate: PASS" if issues["errors"] == 0 else "Production gate: FAIL - resolve ERROR rows before Script 02",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="KITCO3 raw Movesense inventory and final-runsheet reconciliation.")
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--runsheet", type=Path, default=DEFAULT_RUNSHEET_PATH)
    parser.add_argument("--runsheet-sheet", default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--study-timezone", default=DEFAULT_STUDY_TIMEZONE)
    parser.add_argument("--scope-mode", choices=["auto", "all"], default="all")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite prior Script 01 outputs after they have been preserved/reviewed.")
    parser.add_argument("--allow-unresolved", action="store_true", help="Exploratory only: return exit status 0 despite blocking errors.")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    run_started = time.perf_counter()
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s | %(message)s")

    project_root = args.project_root.expanduser().resolve()
    def resolve_from_project(path: Path) -> Path:
        path = path.expanduser()
        return path.resolve() if path.is_absolute() else (project_root / path).resolve()

    raw_root = resolve_from_project(args.raw_root)
    runsheet_path = resolve_from_project(args.runsheet)
    output_dir = resolve_from_project(args.output_dir)
    study_tz = load_timezone(args.study_timezone)

    logging.info("Project root: %s", project_root)
    logging.info("Raw root: %s", raw_root)
    logging.info("Final runsheet: %s", runsheet_path)
    logging.info("Output directory: %s", output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    existing_outputs = [output_dir / name for name in OUTPUT_FILENAMES if (output_dir / name).exists()]
    if existing_outputs and not args.overwrite:
        logging.error("Refusing to overwrite %d existing Script 01 output file(s). Use --overwrite only after preserving/reviewing the prior audit.", len(existing_outputs))
        return 3
    if args.overwrite:
        for path in existing_outputs:
            path.unlink()

    logging.info("Loading and validating final runsheet...")
    runsheet, runsheet_issues = load_and_validate_runsheet(runsheet_path, args.runsheet_sheet)
    expected_phases = build_expected_phases(runsheet)
    logging.info("Runsheet validated: %d participant-phase rows, %d PIDs, %d sessions.", len(runsheet), runsheet["PID"].nunique(), runsheet["Session_ID"].nunique())

    logging.info("Scanning raw Movesense data without loading full signals...")
    file_df, annotation_df, attempt_df, scan_issues = scan_raw_data(raw_root, study_tz)
    logging.info("Found %d recording attempts, %d stream files, and %d annotation events.", len(attempt_df), len(file_df), len(annotation_df))

    # Retain overlap blocks only as a diagnostic description of the raw acquisition.
    # They are deliberately NOT used for session or phase inference.
    attempt_with_blocks, time_block_df = build_time_blocks(attempt_df)

    logging.info("Mapping recording attempts to sessions using participant batches + runsheet date...")
    attempt_mapped, session_issues = map_attempts_to_sessions(attempt_with_blocks, runsheet)
    attempt_mapped = build_attempt_fingerprints(attempt_mapped, file_df)

    logging.info("Detecting copied duplicate recording attempts...")
    attempt_mapped, duplicate_df, duplicate_issues = resolve_duplicate_attempts(attempt_mapped)

    logging.info("Assigning Phase 1/2 by within-session acquisition chronology and sensor validation...")
    attempt_mapped, phase_df, phase_issues = assign_attempt_phases(attempt_mapped, runsheet, expected_phases)

    logging.info("Building canonical sensor-recording manifest...")
    sensor_df, sensor_issues = build_sensor_attempt_table(file_df, attempt_mapped, runsheet)
    in_scope_sessions = set(attempt_mapped["mapped_session_id"].dropna())
    reconciliation = reconcile_participant_phases(runsheet, sensor_df, in_scope_sessions, args.scope_mode)
    session_phase_summary = build_session_phase_summary(runsheet, expected_phases, attempt_mapped, sensor_df, reconciliation)

    issues = build_issue_table(
        runsheet_issues + scan_issues + session_issues + duplicate_issues + phase_issues + sensor_issues,
        attempt_mapped,
        reconciliation,
    )

    write_csv(file_df, output_dir / "01_file_inventory.csv")
    write_csv(annotation_df, output_dir / "01_annotation_events.csv")
    write_csv(attempt_mapped, output_dir / "01_recording_attempts.csv")
    write_csv(time_block_df, output_dir / "01_time_blocks.csv")
    write_csv(phase_df, output_dir / "01_phase_groups.csv")
    write_csv(duplicate_df, output_dir / "01_duplicate_attempts.csv")
    write_csv(sensor_df, output_dir / "01_sensor_recordings.csv")
    write_csv(session_phase_summary, output_dir / "01_session_phase_summary.csv")
    write_csv(reconciliation, output_dir / "01_participant_phase_reconciliation.csv")
    write_csv(issues, output_dir / "01_reconciliation_issues.csv")

    runtime_seconds = time.perf_counter() - run_started
    summary = make_summary(
        runsheet, args.study_timezone, args.scope_mode, project_root, raw_root, runsheet_path,
        file_df, annotation_df, attempt_mapped, time_block_df, duplicate_df, phase_df,
        sensor_df, reconciliation, issues, runtime_seconds,
    )
    with (output_dir / "01_run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True, default=safe_json)
        handle.write("\n")
    write_summary_text(summary, output_dir / "01_run_summary.txt")

    logging.info("Script 01 outputs written to: %s", output_dir)
    logging.info("Session mapping counts: %s", summary["mapping"]["session_mapping_counts"])
    logging.info("Phase mapping counts: %s", summary["mapping"]["phase_mapping_counts"])
    logging.info("Participant-phase reconciliation: %s", summary["participant_phase_reconciliation"])

    blocking_errors = issues[issues["severity"] == "ERROR"] if not issues.empty else issues
    if not blocking_errors.empty and not args.allow_unresolved:
        logging.error("%d blocking reconciliation/inventory error(s) remain. Review 01_reconciliation_issues.csv. Downstream analysis should not proceed until they are resolved.", len(blocking_errors))
        return 2

    logging.info("Script 01 completed successfully. Production gate PASS.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
