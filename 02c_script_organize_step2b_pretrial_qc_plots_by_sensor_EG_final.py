#!/usr/bin/env python3
"""
02c_script_organize_step2b_pretrial_qc_plots_by_sensor_EG_final.py
By: Elnaz Ghasemi
Date: June 2026

Purpose
-------
Organize Step 2b pretrial 2-minute QC plots by sensor recording, so all plots
for one recording_folder + sensor_id are copied into one folder.

This script is only for organization/presentation. It does not change the
original Step 2b outputs and does not delete any original plots.

Expected input after Step 2b v2
-------------------------------
TableTask/
  processed_ecg_hr/
    pretrial_2min_qc_v2/
      plots/
        hr_2min/
        rr_poincare_2min/
        ecg_20sec_rpeaks/
        dyad_hr_2min/
      qc/
        02b_pretrial_2min_plot_manifest.csv
        02b_pretrial_2min_qc_summary.csv
        02b_pretrial_2min_dyad_summary.csv

Outputs
-------
processed_ecg_hr/pretrial_2min_qc_v2/plots/_organized_by_sensor/
  <recording_folder>__<sensor_id>/
    00_sensor_qc_label.txt
    01_pretrial_2min_hr.png
    02_pretrial_2min_rr_poincare.png
    03_pretrial_first20sec_ecg_rpeaks.png

processed_ecg_hr/pretrial_2min_qc_v2/plots/_organized_by_dyad_preview/
  <recording_folder>/
    01_dyad_pretrial_2min_hr.png
    00_dyad_qc_label.txt

How to run
----------
From the top-level TableTask folder:

python3 02c_script_organize_step2b_pretrial_qc_plots_by_sensor_EG_final.py \
  --root . \
  --pretrial-dir processed_ecg_hr/pretrial_2min_qc_v2 \
  --overwrite
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

SCRIPT_VERSION = "v1_step2b_pretrial_qc_plot_organizer"


# =============================================================================
# Argument handling
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Organize Step 2b pretrial QC plots by sensor recording."
    )
    parser.add_argument(
        "--root",
        type=str,
        default=".",
        help="Top-level TableTask folder. Default: current folder.",
    )
    parser.add_argument(
        "--pretrial-dir",
        type=str,
        default="processed_ecg_hr/pretrial_2min_qc_v2",
        help="Step 2b output folder. Default: processed_ecg_hr/pretrial_2min_qc_v2.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing organized folders before recreating them.",
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


def safe_piece(value: Any) -> str:
    text = "" if value is None else str(value).strip()
    if text.lower() in {"", "nan", "none"}:
        text = "NA"
    text = re.sub(r"[^A-Za-z0-9_\-]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_") or "NA"


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def resolve_plot_file(pretrial_dir: Path, plot_file_value: Any) -> Path | None:
    if plot_file_value is None or pd.isna(plot_file_value):
        return None
    text = str(plot_file_value).strip()
    if not text:
        return None
    path = Path(text).expanduser()
    if path.is_absolute():
        return path
    # Step 2b normally stores absolute paths, but this handles relative paths.
    candidate_1 = pretrial_dir / text
    candidate_2 = pretrial_dir.parent.parent / text
    if candidate_1.exists():
        return candidate_1.resolve()
    if candidate_2.exists():
        return candidate_2.resolve()
    return path.resolve()


def standardized_sensor_plot_name(plot_type: str, original_name: str) -> str:
    pt = str(plot_type).strip().lower()
    if pt == "pretrial_2min_hr":
        return "01_pretrial_2min_hr.png"
    if pt == "pretrial_2min_rr_poincare":
        return "02_pretrial_2min_rr_poincare.png"
    if pt == "pretrial_first20sec_ecg_rpeaks":
        return "03_pretrial_first20sec_ecg_rpeaks.png"
    return f"99_{safe_piece(pt)}__{safe_piece(Path(original_name).stem)}.png"


def copy_unique(src: Path, dest: Path) -> bool:
    if not src.exists():
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        # If the exact destination already exists, overwrite intentionally. These
        # are standardized plot names, so one plot type should map to one file.
        dest.unlink()
    shutil.copy2(src, dest)
    return True


def write_sensor_label_file(sensor_dir: Path, row: pd.Series | None) -> None:
    sensor_dir.mkdir(parents=True, exist_ok=True)
    label_file = sensor_dir / "00_sensor_qc_label.txt"
    if row is None:
        label_file.write_text("QC summary row not found for this sensor.\n")
        return

    fields = [
        "recording_folder",
        "sensor_id",
        "role",
        "participant_id",
        "dyad_id",
        "pair1",
        "pretrial_qc_flag",
        "pretrial_qc_reasons",
        "hr_available_duration_sec",
        "hr_coverage_prop",
        "hr_mean_bpm",
        "hr_min_bpm",
        "hr_max_bpm",
        "hr_sd_bpm",
        "hr_range_bpm",
        "hr_extreme_prop",
        "rr_n_intervals",
        "rr_plausible_prop",
        "status",
        "error_message",
    ]
    lines = ["Step 2b pretrial 2-minute sensor QC label", ""]
    for field in fields:
        if field in row.index:
            lines.append(f"{field}: {row.get(field, '')}")
    label_file.write_text("\n".join(lines) + "\n")


def write_dyad_label_file(dyad_dir: Path, row: pd.Series | None) -> None:
    dyad_dir.mkdir(parents=True, exist_ok=True)
    label_file = dyad_dir / "00_dyad_qc_label.txt"
    if row is None:
        label_file.write_text("Dyad QC summary row not found for this session.\n")
        return

    fields = [
        "recording_folder",
        "sensor_A",
        "sensor_B",
        "flag_A",
        "flag_B",
        "dyad_pretrial_qc_flag",
        "dyad_pretrial_qc_reasons",
        "n_common_timepoints",
        "common_duration_sec",
    ]
    lines = ["Step 2b pretrial 2-minute dyad QC label", ""]
    for field in fields:
        if field in row.index:
            lines.append(f"{field}: {row.get(field, '')}")
    label_file.write_text("\n".join(lines) + "\n")


def build_manifest_from_folders(pretrial_dir: Path) -> pd.DataFrame:
    """Fallback if 02b_pretrial_2min_plot_manifest.csv is missing."""
    rows: list[dict[str, str]] = []
    plot_root = pretrial_dir / "plots"
    folder_map = {
        "hr_2min": "pretrial_2min_hr",
        "rr_poincare_2min": "pretrial_2min_rr_poincare",
        "ecg_20sec_rpeaks": "pretrial_first20sec_ecg_rpeaks",
    }

    for folder_name, plot_type in folder_map.items():
        folder = plot_root / folder_name
        if not folder.exists():
            continue
        for file in folder.glob("*.png"):
            parts = file.stem.split("__")
            if len(parts) < 2:
                continue
            rows.append(
                {
                    "recording_folder": parts[0],
                    "sensor_id": parts[1],
                    "plot_type": plot_type,
                    "plot_file": str(file),
                }
            )

    dyad_folder = plot_root / "dyad_hr_2min"
    if dyad_folder.exists():
        for file in dyad_folder.glob("*.png"):
            parts = file.stem.split("__")
            if len(parts) < 1:
                continue
            rows.append(
                {
                    "recording_folder": parts[0],
                    "sensor_id": "dyad",
                    "plot_type": "dyad_pretrial_2min_hr",
                    "plot_file": str(file),
                }
            )

    return pd.DataFrame(rows)


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    pretrial_dir = resolve_path(root, args.pretrial_dir)

    plots_dir = pretrial_dir / "plots"
    qc_dir = pretrial_dir / "qc"
    manifest_path = qc_dir / "02b_pretrial_2min_plot_manifest.csv"
    summary_path = qc_dir / "02b_pretrial_2min_qc_summary.csv"
    dyad_summary_path = qc_dir / "02b_pretrial_2min_dyad_summary.csv"

    out_sensor = plots_dir / "_organized_by_sensor"
    out_dyad = plots_dir / "_organized_by_dyad_preview"

    print(f"Script version: {SCRIPT_VERSION}")
    print(f"Root: {root}")
    print(f"Step 2b folder: {pretrial_dir}")

    if not pretrial_dir.exists():
        raise FileNotFoundError(f"Step 2b folder not found: {pretrial_dir}")
    if not plots_dir.exists():
        raise FileNotFoundError(f"Step 2b plots folder not found: {plots_dir}")

    if args.overwrite:
        for folder in [out_sensor, out_dyad]:
            if folder.exists():
                shutil.rmtree(folder)

    out_sensor.mkdir(parents=True, exist_ok=True)
    out_dyad.mkdir(parents=True, exist_ok=True)

    if manifest_path.exists():
        manifest = pd.read_csv(manifest_path)
        print(f"Using plot manifest: {manifest_path}")
    else:
        manifest = build_manifest_from_folders(pretrial_dir)
        print("WARNING: plot manifest not found. Built fallback manifest by scanning plot folders.")

    if manifest.empty:
        raise RuntimeError("No Step 2b plot files found to organize.")

    summary_df = read_csv_if_exists(summary_path)
    dyad_df = read_csv_if_exists(dyad_summary_path)

    # Build lookup tables for labels.
    sensor_summary_lookup: dict[tuple[str, str], pd.Series] = {}
    if not summary_df.empty and {"recording_folder", "sensor_id"}.issubset(summary_df.columns):
        for _, row in summary_df.iterrows():
            key = (str(row.get("recording_folder", "")), str(row.get("sensor_id", "")))
            sensor_summary_lookup[key] = row

    dyad_summary_lookup: dict[str, pd.Series] = {}
    if not dyad_df.empty and "recording_folder" in dyad_df.columns:
        for _, row in dyad_df.iterrows():
            dyad_summary_lookup[str(row.get("recording_folder", ""))] = row

    n_copied = 0
    n_skipped_missing = 0
    sensor_dirs_created: set[str] = set()
    dyad_dirs_created: set[str] = set()

    required_cols = {"recording_folder", "sensor_id", "plot_type", "plot_file"}
    missing_cols = required_cols - set(manifest.columns)
    if missing_cols:
        raise ValueError(f"Plot manifest is missing required columns: {sorted(missing_cols)}")

    for _, row in manifest.iterrows():
        recording_folder = str(row.get("recording_folder", "")).strip()
        sensor_id = str(row.get("sensor_id", "")).strip()
        plot_type = str(row.get("plot_type", "")).strip()
        src = resolve_plot_file(pretrial_dir, row.get("plot_file", ""))

        if src is None or not src.exists():
            print(f"SKIPPED missing plot file: {row.get('plot_file', '')}")
            n_skipped_missing += 1
            continue

        if sensor_id.lower() == "dyad" or plot_type.lower().startswith("dyad"):
            dyad_dir = out_dyad / safe_piece(recording_folder)
            dyad_dir.mkdir(parents=True, exist_ok=True)
            dyad_dirs_created.add(str(dyad_dir))
            dest = dyad_dir / "01_dyad_pretrial_2min_hr.png"
            if copy_unique(src, dest):
                n_copied += 1
            write_dyad_label_file(dyad_dir, dyad_summary_lookup.get(recording_folder))
            continue

        sensor_dir = out_sensor / f"{safe_piece(recording_folder)}__{safe_piece(sensor_id)}"
        sensor_dir.mkdir(parents=True, exist_ok=True)
        sensor_dirs_created.add(str(sensor_dir))

        dest_name = standardized_sensor_plot_name(plot_type, src.name)
        dest = sensor_dir / dest_name
        if copy_unique(src, dest):
            n_copied += 1

        write_sensor_label_file(
            sensor_dir,
            sensor_summary_lookup.get((recording_folder, sensor_id)),
        )

    # Write simple run summary for the organizer itself.
    organizer_summary = qc_dir / "02c_organized_step2b_plots_summary.txt"
    with open(organizer_summary, "w") as f:
        f.write("Step 2b pretrial QC plot organizer summary\n")
        f.write("========================================\n")
        f.write(f"Script version: {SCRIPT_VERSION}\n")
        f.write(f"Step 2b folder: {pretrial_dir}\n")
        f.write(f"Manifest rows read: {len(manifest)}\n")
        f.write(f"Copied plot files: {n_copied}\n")
        f.write(f"Skipped missing plot files: {n_skipped_missing}\n")
        f.write(f"Sensor folders created/found: {len(sensor_dirs_created)}\n")
        f.write(f"Dyad preview folders created/found: {len(dyad_dirs_created)}\n")
        f.write(f"Sensor-organized folder: {out_sensor}\n")
        f.write(f"Dyad-preview-organized folder: {out_dyad}\n")

    print("Done.")
    print(f"Copied plot files: {n_copied}")
    print(f"Skipped missing plot files: {n_skipped_missing}")
    print(f"Sensor folders created/found: {len(sensor_dirs_created)}")
    print(f"Dyad preview folders created/found: {len(dyad_dirs_created)}")
    print(f"Sensor-organized folder: {out_sensor}")
    print(f"Dyad-preview-organized folder: {out_dyad}")
    print(f"Organizer summary: {organizer_summary}")


if __name__ == "__main__":
    main()
