#!/usr/bin/env python3
"""
05a_script_tabletask_nste_plots_v4_final_EG.py

Visualization companion for 05_script_tabletask_nste_v4_final_EG.py.
It adapts the laboratory NSTE_figs.py plot family to TableTask without treating
practice as a physiological baseline by default.

Input
-----
processed_ecg_hr/trial_nste_final_EG/tables/05_nste_all_window_tau_rows_EG.csv

Outputs
-------
processed_ecg_hr/trial_nste_final_EG/plots/
  NSTE_asymmetry/*.png
  STE_asymmetry/*.png
processed_ecg_hr/trial_nste_final_EG/tables/
  05_nste_plot_manifest_EG.csv
  05_nste_optimal_tau_descriptive_EG.csv/.xlsx

Plot logic
----------
- Uses the laboratory's primary 30-second window display.
- Plots one line per tau across candidate windows/trials in chronological order.
- Positive asymmetry means B->A exceeds A->B; negative means A->B exceeds B->A.
- Significant points use the direction-specific raw outer-permutation p-value
  with Bonferroni correction across displayed tau values, matching NSTE_figs.py.
- Raw asymmetry is the default. Optional --practice-correct subtracts the dyad's
  practice mean per tau, but this is explicitly labelled practice-corrected and
  should not be called baseline-corrected.
- Optimal tau summaries are descriptive and are computed separately for FF and FE.
"""

from __future__ import annotations

import argparse
import math
import shutil
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_VERSION = "v4_tabletask_nste_plots_final_EG"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create TableTask STE/NSTE asymmetry plots.")
    p.add_argument("--root", default=".")
    p.add_argument("--input", default="processed_ecg_hr/trial_nste_final_EG/tables/05_nste_all_window_tau_rows_EG.csv")
    p.add_argument("--out-dir", default="processed_ecg_hr/trial_nste_final_EG")
    p.add_argument("--window-sec", type=float, default=30.0)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--practice-correct", action="store_true")
    p.add_argument("--overwrite-plots", action="store_true")
    return p.parse_args()


def resolve(root: Path, text: str) -> Path:
    p = Path(text).expanduser()
    return (p if p.is_absolute() else root / p).resolve()


def safe_piece(x: Any) -> str:
    import re
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(x)).strip("_") or "NA"


def save_excel(df: pd.DataFrame, path: Path) -> None:
    try:
        df.to_excel(path, index=False)
    except Exception as exc:
        print(f"WARNING: could not save {path}: {exc}")


def prepare_timeline(sub: pd.DataFrame, measure: str, practice_correct: bool) -> pd.DataFrame:
    out = sub.copy()
    yx = f"{measure}_YX"
    xy = f"{measure}_XY"
    out["asymmetry_raw"] = pd.to_numeric(out[yx], errors="coerce") - pd.to_numeric(out[xy], errors="coerce")
    out["asymmetry_plot"] = out["asymmetry_raw"]
    if practice_correct and "is_practice" in out.columns:
        practice = out[out["is_practice"].astype(str).str.lower().isin(["true", "1", "yes"])]
        means = practice.groupby("tau")["asymmetry_raw"].mean()
        out["asymmetry_plot"] = out.apply(lambda r: r["asymmetry_raw"] - means.get(r["tau"], 0.0), axis=1)

    # Concatenate trial timelines without pretending there was continuous physiology between trials.
    out["candidate_window_num"] = pd.to_numeric(out.get("candidate_window", np.nan), errors="coerce")
    out["trial_num"] = pd.to_numeric(out.get("trial", np.nan), errors="coerce")
    ordered_keys = (
        out[["candidate_window_num", "trial_num"]]
        .drop_duplicates()
        .sort_values(["candidate_window_num", "trial_num"], na_position="last")
        .reset_index(drop=True)
    )
    offsets = {}
    current = 0.0
    gap = 5.0
    for _, row in ordered_keys.iterrows():
        cw = row["candidate_window_num"]
        tr = row["trial_num"]
        mask = (out["candidate_window_num"] == cw) & (out["trial_num"] == tr)
        local = pd.to_numeric(out.loc[mask, "elapsed_time"], errors="coerce")
        if local.notna().any():
            local_min = float(local.min())
            local_max = float(local.max())
            offsets[(cw, tr)] = current - local_min
            current += (local_max - local_min) + gap
    out["plot_time_s"] = out.apply(
        lambda r: float(r["elapsed_time"]) + offsets.get((r["candidate_window_num"], r["trial_num"]), 0.0), axis=1
    )
    return out


def plot_one(sub: pd.DataFrame, measure: str, out_dir: Path, alpha: float, practice_correct: bool) -> str:
    df = prepare_timeline(sub, measure, practice_correct)
    taus = sorted(pd.to_numeric(df["tau"], errors="coerce").dropna().unique())
    fig, ax = plt.subplots(figsize=(15, 6))
    styles = ["-", "--", "-.", ":"]
    for i, tau in enumerate(taus):
        tsub = df[pd.to_numeric(df["tau"], errors="coerce") == tau].sort_values("plot_time_s")
        ax.plot(tsub["plot_time_s"], tsub["asymmetry_plot"], linewidth=1.0, alpha=0.65,
                linestyle=styles[i % len(styles)], label=f"tau={int(tau)}")
        # Bonferroni correction across displayed tau values, matching the
        # laboratory plot family. STE plots use STE permutation p-values; NSTE
        # plots use NSTE-specific surrogate p-values generated by the v4
        # computation script rather than incorrectly reusing STE p-values.
        alpha_tau = alpha / max(1, len(taus))
        if measure == "NSTE":
            p_pos = pd.to_numeric(tsub.get("NSTE_p_YX", np.nan), errors="coerce")
            p_neg = pd.to_numeric(tsub.get("NSTE_p_XY", np.nan), errors="coerce")
        else:
            p_pos = pd.to_numeric(tsub.get("p_YX", np.nan), errors="coerce")
            p_neg = pd.to_numeric(tsub.get("p_XY", np.nan), errors="coerce")
        y = pd.to_numeric(tsub["asymmetry_plot"], errors="coerce")
        sig = ((y > 0) & (p_pos < alpha_tau)) | ((y < 0) & (p_neg < alpha_tau))
        ax.scatter(tsub.loc[sig, "plot_time_s"], y[sig], s=18, facecolors="none", edgecolors="black", linewidths=0.7)

    ax.axhline(0, linestyle="--", linewidth=1, color="black")
    ax.set_xlabel("Concatenated within-trial time (s; gaps separate candidate windows)")
    correction = "practice-corrected" if practice_correct else "raw"
    ax.set_ylabel(f"{measure} asymmetry, B→A minus A→B ({correction})")
    dyad = str(df["dyad_id"].iloc[0]) if "dyad_id" in df.columns else "NA"
    ax.set_title(f"{dyad}: {measure} directional asymmetry ({correction}, 30-s windows)")
    ax.grid(True, alpha=0.2)

    # Trial boundaries and condition labels.
    boundaries = (
        df.groupby(["candidate_window_num", "trial_num", "condition"], dropna=False)["plot_time_s"]
          .agg(["min", "max"]).reset_index().sort_values("min")
    )
    for _, r in boundaries.iterrows():
        ax.axvline(r["min"], linestyle=":", linewidth=0.7, color="black", alpha=0.5)
        ax.text((r["min"] + r["max"]) / 2, ax.get_ylim()[1], f"cw{int(r['candidate_window_num']):02d} {r['condition']}",
                rotation=90, va="top", ha="center", fontsize=7)
    ax.legend(title="Tau", ncol=4, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.13))
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{safe_piece(dyad)}__{measure}_asymmetry_{'practice_corrected' if practice_correct else 'raw'}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def optimal_tau_descriptive(df: pd.DataFrame, window_sec: float, practice_correct: bool) -> pd.DataFrame:
    use = df[np.isclose(pd.to_numeric(df["win_size_sec"], errors="coerce"), window_sec)].copy()
    rows = []
    for dyad, sub in use.groupby("dyad_id", dropna=False):
        for measure in ["STE", "NSTE"]:
            yx, xy = f"{measure}_YX", f"{measure}_XY"
            sub2 = sub.copy()
            sub2["total_flow"] = pd.to_numeric(sub2[yx], errors="coerce") + pd.to_numeric(sub2[xy], errors="coerce")
            if practice_correct and "is_practice" in sub2.columns:
                practice = sub2[sub2["is_practice"].astype(str).str.lower().isin(["true", "1", "yes"])]
                base = practice.groupby("tau")["total_flow"].mean()
                sub2["total_flow_used"] = sub2.apply(lambda r: r["total_flow"] - base.get(r["tau"], 0.0), axis=1)
            else:
                sub2["total_flow_used"] = sub2["total_flow"]
            for cond in ["FF", "FE"]:
                c = sub2[sub2["condition"].astype(str).eq(cond)]
                means = c.groupby("tau")["total_flow_used"].mean().replace([np.inf, -np.inf], np.nan).dropna()
                if means.empty:
                    tau = np.nan
                    value = np.nan
                else:
                    tau = float(means.idxmax())
                    value = float(means.max())
                rows.append({
                    "script_version": SCRIPT_VERSION,
                    "dyad_id": dyad,
                    "measure": measure,
                    "condition": cond,
                    "window_sec": window_sec,
                    "practice_corrected": practice_correct,
                    "optimal_tau_samples_descriptive": tau,
                    "optimal_tau_seconds_descriptive": tau / 4.0 if np.isfinite(tau) else np.nan,
                    "max_mean_total_flow_descriptive": value,
                    "interpretation": "Descriptive parameter summary; not an inferentially selected primary tau.",
                })
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    inp = resolve(root, args.input)
    out = resolve(root, args.out_dir)
    if not inp.exists():
        raise FileNotFoundError(inp)
    plot_root = out / "plots"
    if plot_root.exists() and args.overwrite_plots:
        shutil.rmtree(plot_root)
    nste_dir = plot_root / "NSTE_asymmetry"
    ste_dir = plot_root / "STE_asymmetry"
    table_dir = out / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(inp, dtype={"recording_folder": str, "dyad_id": str, "condition": str})
    use = df[np.isclose(pd.to_numeric(df["win_size_sec"], errors="coerce"), args.window_sec)].copy()
    if use.empty:
        raise ValueError(f"No rows for window size {args.window_sec}s")
    manifest = []
    for dyad, sub in use.groupby("dyad_id", dropna=False):
        for measure, folder in [("NSTE", nste_dir), ("STE", ste_dir)]:
            try:
                path = plot_one(sub, measure, folder, args.alpha, args.practice_correct)
                manifest.append({"dyad_id": dyad, "measure": measure, "status": "saved", "plot_file": path, "error": ""})
            except Exception as exc:
                manifest.append({"dyad_id": dyad, "measure": measure, "status": "failed", "plot_file": "", "error": str(exc)})
    manifest_df = pd.DataFrame(manifest)
    manifest_df.to_csv(table_dir / "05_nste_plot_manifest_EG.csv", index=False)
    opt = optimal_tau_descriptive(df, args.window_sec, args.practice_correct)
    opt.to_csv(table_dir / "05_nste_optimal_tau_descriptive_EG.csv", index=False)
    save_excel(opt, table_dir / "05_nste_optimal_tau_descriptive_EG.xlsx")
    print("Done.")
    print(f"Plots saved: {(manifest_df['status'] == 'saved').sum()} / {len(manifest_df)}")
    print(f"Plot failures: {(manifest_df['status'] == 'failed').sum()}")
    return 0 if not (manifest_df["status"] == "failed").any() else 2


if __name__ == "__main__":
    raise SystemExit(main())
