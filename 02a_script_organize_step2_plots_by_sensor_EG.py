#!/usr/bin/env python3
"""
Organize TableTask Step 2 ECG-to-HR plots by sensor recording.
By: Elnaz Ghasemi
Date: June 2026

Run from the top-level TableTask folder, for example:
  python3 02b_script_organize_step2_plots_by_sensor_EG.py

This script COPIES plot PNGs into new organized folders. It does not delete or move
any original Step 2 output files.

Expected input:
  processed_ecg_hr/plots/
    raw_vs_clean/
    rpeaks/
    hr_before_after_correction/
    hr_time_series/
    poincare/
    hrv_psd/
    dyad_session_preview_not_for_final_analysis/

Outputs:
  processed_ecg_hr/plots/_organized_by_sensor/
  processed_ecg_hr/plots/_organized_by_dyad_preview/
  processed_ecg_hr/plots/_organized_plot_copy_log.csv
"""

from pathlib import Path
import argparse
import csv
import re
import shutil
import sys


def clean_text(text: str) -> str:
    text = str(text).strip()
    text = re.sub(r"[^A-Za-z0-9_\-]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_") or "NA"


def plot_label(folder_name: str, filename: str) -> str:
    stem = Path(filename).stem.lower()

    if folder_name == "raw_vs_clean":
        return "01_raw_vs_clean.png"
    if folder_name == "rpeaks":
        return "02_rpeaks_corrected.png"
    if folder_name == "hr_before_after_correction":
        return "03_hr_before_after_correction.png"
    if folder_name == "hr_time_series":
        if "original" in stem:
            return "04_hr_original.png"
        if "corrected" in stem:
            return "05_hr_corrected.png"
        return "04_hr_time_series.png"
    if folder_name == "poincare":
        return "06_poincare_corrected.png"
    if folder_name == "hrv_psd":
        return "07_hrv_psd_corrected.png"
    return clean_text(filename)


def main() -> int:
    parser = argparse.ArgumentParser(description="Organize Step 2 plots by sensor recording.")
    parser.add_argument(
        "--plots-root",
        default="processed_ecg_hr/plots",
        help="Path to Step 2 plots folder. Default: processed_ecg_hr/plots",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite copied files if they already exist.",
    )
    args = parser.parse_args()

    plots_root = Path(args.plots_root).expanduser().resolve()
    if not plots_root.exists():
        print(f"ERROR: plots folder not found: {plots_root}", file=sys.stderr)
        print("Run this script from the top-level TableTask folder, or pass --plots-root.", file=sys.stderr)
        return 1

    sensor_plot_folders = [
        "raw_vs_clean",
        "rpeaks",
        "hr_before_after_correction",
        "hr_time_series",
        "poincare",
        "hrv_psd",
    ]

    out_sensor = plots_root / "_organized_by_sensor"
    out_dyad = plots_root / "_organized_by_dyad_preview"
    out_sensor.mkdir(parents=True, exist_ok=True)
    out_dyad.mkdir(parents=True, exist_ok=True)

    log_rows = []
    copied = 0
    skipped = 0

    for folder_name in sensor_plot_folders:
        folder = plots_root / folder_name
        if not folder.exists():
            print(f"WARNING: missing folder: {folder}")
            log_rows.append({
                "status": "missing_folder",
                "source_file": str(folder),
                "destination_file": "",
                "note": "Input plot folder missing",
            })
            continue

        for file in sorted(folder.glob("*.png")):
            parts = file.stem.split("__")
            if len(parts) < 3:
                skipped += 1
                log_rows.append({
                    "status": "skipped",
                    "source_file": str(file),
                    "destination_file": "",
                    "note": "Unexpected filename pattern",
                })
                continue

            recording_folder = parts[0]
            sensor_id = parts[1]

            # Sensor IDs should be plain numeric strings. Combined dyad IDs are handled separately.
            if not sensor_id.isdigit():
                skipped += 1
                log_rows.append({
                    "status": "skipped",
                    "source_file": str(file),
                    "destination_file": "",
                    "note": "Not a single sensor-level plot",
                })
                continue

            sensor_dir = out_sensor / f"{recording_folder}__{sensor_id}"
            sensor_dir.mkdir(parents=True, exist_ok=True)
            destination = sensor_dir / plot_label(folder_name, file.name)

            if destination.exists() and not args.overwrite:
                skipped += 1
                log_rows.append({
                    "status": "skipped_exists",
                    "source_file": str(file),
                    "destination_file": str(destination),
                    "note": "Destination already exists; rerun with --overwrite to replace",
                })
                continue

            shutil.copy2(file, destination)
            copied += 1
            log_rows.append({
                "status": "copied",
                "source_file": str(file),
                "destination_file": str(destination),
                "note": "",
            })

    dyad_folder = plots_root / "dyad_session_preview_not_for_final_analysis"
    if dyad_folder.exists():
        for file in sorted(dyad_folder.glob("*.png")):
            parts = file.stem.split("__")
            if len(parts) < 1:
                skipped += 1
                log_rows.append({
                    "status": "skipped",
                    "source_file": str(file),
                    "destination_file": "",
                    "note": "Unexpected dyad preview filename pattern",
                })
                continue

            recording_folder = parts[0]
            dyad_dir = out_dyad / recording_folder
            dyad_dir.mkdir(parents=True, exist_ok=True)
            destination = dyad_dir / "diagnostic_dyad_session_preview_not_final_analysis.png"

            if destination.exists() and not args.overwrite:
                skipped += 1
                log_rows.append({
                    "status": "skipped_exists",
                    "source_file": str(file),
                    "destination_file": str(destination),
                    "note": "Destination already exists; rerun with --overwrite to replace",
                })
                continue

            shutil.copy2(file, destination)
            copied += 1
            log_rows.append({
                "status": "copied",
                "source_file": str(file),
                "destination_file": str(destination),
                "note": "dyad_preview",
            })
    else:
        print(f"WARNING: missing dyad preview folder: {dyad_folder}")

    log_path = plots_root / "_organized_plot_copy_log.csv"
    with log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["status", "source_file", "destination_file", "note"])
        writer.writeheader()
        writer.writerows(log_rows)

    sensor_dirs = sorted([p for p in out_sensor.iterdir() if p.is_dir()])
    dyad_dirs = sorted([p for p in out_dyad.iterdir() if p.is_dir()])

    print("Done.")
    print(f"Copied files: {copied}")
    print(f"Skipped files: {skipped}")
    print(f"Sensor folders created/found: {len(sensor_dirs)}")
    print(f"Dyad preview folders created/found: {len(dyad_dirs)}")
    print(f"Sensor-organized folder: {out_sensor}")
    print(f"Dyad-preview-organized folder: {out_dyad}")
    print(f"Copy log: {log_path}")

    # Expected if all Step 2 plots exist: 36 sensor folders and 18 dyad-preview folders.
    if len(sensor_dirs) != 36:
        print("WARNING: expected 36 sensor folders. Check the copy log and filenames.")
    if len(dyad_dirs) != 18:
        print("WARNING: expected 18 dyad-preview folders. Check the copy log and filenames.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
