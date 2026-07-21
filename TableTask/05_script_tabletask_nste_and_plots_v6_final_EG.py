#!/usr/bin/env python3
"""
05_script_tabletask_nste_and_plots_v6_final_EG.py

TableTask Step 5: Symbolic Transfer Entropy (STE) and Normalized STE (NSTE)
for all Step 3 paired 4-Hz HR trial files.

Core estimator logic is adapted from the laboratory NSTE implementation:
  delayRecons.py, F_Int2Symb.py, F_Integer2prob.py, F_EstimateProb.py,
  f_predictiontime.py, f_nste.py, calculate_STE.py, and nste_main_CC.py.

Required input
--------------
processed_ecg_hr/trial_segments_4hz/paired_trial_hr/*.csv
  Required columns: TimeRelTrialSec, HR_A_BPM, HR_B_BPM

Optional metadata/QC inputs
---------------------------
processed_ecg_hr/trial_segments_4hz/qc/03_trial_segmentation_summary.csv
processed_ecg_hr/rr_reliability_qc/qc/3a_trial_dyad_rr_reliability.csv
qc_outputs/veronica_analysis_units.csv

Outputs
-------
processed_ecg_hr/trial_nste_final_EG/
  results_by_trial/*.csv
  tables/05_nste_all_window_tau_rows_EG.csv/.xlsx
  tables/05_nste_file_manifest_EG.csv
  tables/05_nste_failures_EG.csv
  tables/05_nste_run_summary_EG.txt/.json
  plots/NSTE_asymmetry/*.png
  plots/STE_asymmetry/*.png
  tables/05_nste_plot_manifest_EG.csv
  tables/05_nste_optimal_tau_descriptive_EG.csv/.xlsx
  README_05_NSTE_outputs_EG.txt

Important design decisions
--------------------------
1. Every Step 3 paired file is analyzed; no eligibility/QC filtering is applied.
2. Step 3 already provides aligned, uniformly sampled 4-Hz HR. No second
   interpolation or downsampling is performed.
3. The laboratory temporal parameter range (tau 5..49 samples at 15 Hz) is
   converted to unique 4-Hz lags spanning approximately the same durations:
   tau 1..13 samples (0.25..3.25 s).
4. A single 30-s primary window with 80% overlap is used. The 30-s duration is inherited from the laboratory plotting workflow and selected as the TableTask primary scale to balance temporal resolution, estimator stability, and computational cost. It remains a study-level choice to confirm with Veronica.
5. Direction mapping for TableTask is explicit:
      STE_YX / NSTE_YX = B -> A (Y=participant B, X=participant A)
      STE_XY / NSTE_XY = A -> B (X=participant A, Y=participant B)
6. NSTE retains the laboratory internal 20 source-shuffle bias correction.
7. Outer permutation testing follows calculate_STE.py: Y is permuted once per
   outer permutation, that same permutation is reused across tau values, and the
   full STE/NSTE routine is recalculated in both directions.
8. Permutation means, population SDs, Z scores, and one-sided empirical p-values
   follow the original laboratory implementation for STE. NSTE-specific null
   statistics are additionally retained for statistically valid NSTE plotting.
9. Direction-specific and asymmetry-specific Monte Carlo permutation p-values use
   the finite-sample +1 correction. Plot markers use the direct two-sided asymmetry
   test with Bonferroni correction across displayed tau values.
10. TableTask-specific safeguard: windows are created only within contiguous valid
    4-Hz paired-HR runs so missing-data gaps are not compressed or bridged.
11. No arbitrary effective-symbol cutoff is imposed. Every mathematically valid
    window/tau combination is calculated, and n_effective_symbols plus a descriptive
    small-sample flag are saved for review.

This script estimates STE/NSTE, writes auditable outputs, and generates the associated dyad-level plots in the same run. Final scientific choices (primary window, tau, surrogate null, and
condition-level repeated-measures model) must be specified by the study team.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_VERSION = "v6_tabletask_nste_30s_400perm_asymmetry_test_checkpointed_EG"
DEFAULT_FS = 4.0
DEFAULT_DIM = 3
DEFAULT_TAU = tuple(range(1, 14))
DEFAULT_WINDOW_SECONDS = (30,)
DEFAULT_OVERLAP = 0.80
DEFAULT_MAX_PREDICTION_LAG_SEC = 3.25
DEFAULT_INTERNAL_SHUFFLES = 20
DEFAULT_OUTER_SURROGATES = 400
ADVISORY_SMALL_EFFECTIVE_SYMBOLS = 30
EPS = 1e-12


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TableTask STE/NSTE analysis adapted from the laboratory NSTE workflow.")
    p.add_argument("--root", default=".")
    p.add_argument("--paired-hr-dir", default="processed_ecg_hr/trial_segments_4hz/paired_trial_hr")
    p.add_argument("--trial-summary", default="processed_ecg_hr/trial_segments_4hz/qc/03_trial_segmentation_summary.csv")
    p.add_argument("--reliability-summary", default="processed_ecg_hr/rr_reliability_qc/qc/3a_trial_dyad_rr_reliability.csv")
    p.add_argument("--analysis-units", default="qc_outputs/veronica_analysis_units.csv")
    p.add_argument("--out-dir", default="processed_ecg_hr/trial_nste_final_EG")
    p.add_argument("--sampling-rate", type=float, default=DEFAULT_FS)
    p.add_argument("--embedding-dim", type=int, default=DEFAULT_DIM)
    p.add_argument("--tau-min", type=int, default=1)
    p.add_argument("--tau-max", type=int, default=13)
    p.add_argument("--tau-step", type=int, default=1)
    p.add_argument("--window-seconds", nargs="+", type=float, default=list(DEFAULT_WINDOW_SECONDS))
    p.add_argument("--window-overlap", type=float, default=DEFAULT_OVERLAP)
    p.add_argument("--max-prediction-lag-sec", type=float, default=DEFAULT_MAX_PREDICTION_LAG_SEC)
    p.add_argument("--internal-shuffles", type=int, default=DEFAULT_INTERNAL_SHUFFLES)
    p.add_argument("--n-surrogates", type=int, default=DEFAULT_OUTER_SURROGATES)
    p.add_argument("--random-seed", type=int, default=448)
    p.add_argument("--workers", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    p.add_argument("--max-files", type=int, default=-1, help="Test limit; -1 processes every file.")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--alpha", type=float, default=0.05, help="Family-wise alpha for Bonferroni display across tau values.")
    p.add_argument("--practice-correct", action="store_true", help="Subtract each dyad's practice mean asymmetry per tau for plots only.")
    p.add_argument("--skip-plots", action="store_true", help="Compute tables without generating plots.")
    return p.parse_args()


def resolve(root: Path, text: str) -> Path:
    p = Path(text).expanduser()
    return (p if p.is_absolute() else root / p).resolve()


def safe_piece(value: Any) -> str:
    import re
    s = str(value).strip()
    if s.lower() in {"", "nan", "none", "nat"}:
        s = "NA"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s).strip("_") or "NA"


def bool_from_any(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if pd.isna(v):
        return False
    return str(v).strip().lower() in {"1", "true", "t", "yes", "y"}


def stable_seed(base: int, *parts: Any) -> int:
    text = "|".join(map(str, (base,) + parts)).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(text, digest_size=8).digest(), "little") % (2**32 - 1)


def read_optional_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(
            path,
            dtype={"recording_folder": str, "dyad_id": str, "sensor_A": str, "sensor_B": str},
        )
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def delay_reconstruct(data: np.ndarray, lag: int, dim: int) -> np.ndarray:
    data = np.asarray(data, dtype=float)
    n, ch = data.shape
    rows = n - lag * (dim - 1)
    if lag < 1 or dim < 2 or rows <= 0:
        raise ValueError(f"Insufficient samples for embedding: n={n}, lag={lag}, dim={dim}")
    out = np.empty((rows, dim, ch), dtype=float)
    for c in range(ch):
        for j in range(dim):
            start = j * lag
            end = n - lag * (dim - 1 - j)
            out[:, j, c] = data[start:end, c]
    return out


def ordinal_symbols(E: np.ndarray) -> np.ndarray:
    """Exact rank-symbol convention used by the laboratory F_Int2Symb.py."""
    E = np.asarray(E, dtype=float)
    sorted_idx = np.argsort(E, axis=1)
    n, dim = E.shape
    ranks = np.zeros((n, dim), dtype=np.int64)
    rank_values = np.arange(1, dim + 1, dtype=np.int64)
    for i in range(n):
        ranks[i, sorted_idx[i]] = rank_values
    powers = (10 ** np.arange(dim - 1, -1, -1)).astype(np.int64)
    return ranks @ powers


def probabilities_for_triplets(i1: np.ndarray, i2: np.ndarray, i3: np.ndarray) -> tuple[np.ndarray, ...]:
    """Equivalent to the laboratory f_Integer2prob + F_EstimateProb logic."""
    i1 = np.asarray(i1, dtype=np.int64)
    i2 = np.asarray(i2, dtype=np.int64)
    i3 = np.asarray(i3, dtype=np.int64)
    if not (len(i1) == len(i2) == len(i3)) or len(i1) == 0:
        raise ValueError("Symbol arrays must be non-empty and equal length")

    def symbol_len(a: np.ndarray) -> int:
        return int(np.max(np.ceil(np.log10(a + 0.1)).astype(int)))

    L = max(symbol_len(i1), symbol_len(i2), symbol_len(i3))
    b = 10 ** L
    int1 = i1 * (b**2) + i2 * b + i3
    int2 = i2 * b + i3
    int3 = i1 * b + i2
    int4 = i2
    _, inverse1, counts1 = np.unique(int1, return_inverse=True, return_counts=True)
    _, inverse2, counts2 = np.unique(int2, return_inverse=True, return_counts=True)
    _, inverse3, counts3 = np.unique(int3, return_inverse=True, return_counts=True)
    _, inverse4, counts4 = np.unique(int4, return_inverse=True, return_counts=True)
    n = len(int1)
    p1_all = counts1[inverse1] / n
    p2_all = counts2[inverse2] / n
    p3_all = counts3[inverse3] / n
    p4_all = counts4[inverse4] / n
    _, first_idx = np.unique(int1, return_index=True)
    return p1_all[first_idx], p2_all[first_idx], p3_all[first_idx], p4_all[first_idx]


def ste_from_symbols(future_target: np.ndarray, past_target: np.ndarray, past_source: np.ndarray) -> tuple[float, float]:
    p1, p2, p3, p4 = probabilities_for_triplets(future_target, past_target, past_source)
    with np.errstate(divide="ignore", invalid="ignore"):
        ste = np.sum(p1 * (np.log2(p1 * p4) - np.log2(p2 * p3)))
        h = -np.sum(p3 * (np.log2(p3) - np.log2(p4)))
    return float(ste), float(h)


def symbolic_pair(data: np.ndarray, dim: int, tau: int) -> np.ndarray:
    embedded = delay_reconstruct(data, tau, dim)
    symbols = np.empty((embedded.shape[0], 2), dtype=np.int64)
    symbols[:, 0] = ordinal_symbols(embedded[:, :, 0])
    symbols[:, 1] = ordinal_symbols(embedded[:, :, 1])
    return symbols


def raw_ste_pair(data: np.ndarray, dim: int, tau: int, delta: int) -> tuple[np.ndarray, np.ndarray, int]:
    symbols = symbolic_pair(data, dim, tau)
    if delta < 1 or len(symbols) <= delta:
        raise ValueError(f"Insufficient symbols for delta={delta}: {len(symbols)}")
    future = symbols[delta:, :]
    past = symbols[:-delta, :]
    n_eff = len(future)
    ste_yx, h_yx = ste_from_symbols(future[:, 0], past[:, 0], past[:, 1])
    ste_xy, h_xy = ste_from_symbols(future[:, 1], past[:, 1], past[:, 0])
    return np.array([ste_yx, ste_xy]), np.array([h_yx, h_xy]), n_eff


def prediction_delta(x: np.ndarray, y: np.ndarray, maxlag: int) -> int:
    """Laboratory f_predictiontime logic with bounds safe for short windows."""
    n = len(x)
    maxlag = int(max(1, min(maxlag, n - 1)))
    res = np.correlate(x - np.mean(x), y - np.mean(y), mode="full")
    mid = n - 1
    lags = np.arange(-maxlag, maxlag + 1)
    values = res[mid - maxlag:mid + maxlag + 1]
    lag_at_max = int(lags[int(np.argmax(values))])
    return 1 if lag_at_max <= 1 else lag_at_max


def make_surrogate(source: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Original laboratory outer-null operation: unrestricted random permutation."""
    return rng.permutation(source)


def nste_pair(
    data: np.ndarray,
    dim: int,
    tau: int,
    delta: int,
    internal_shuffles: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Laboratory NSTE definition with configurable deterministic RNG."""
    ste, H, n_eff = raw_ste_pair(data, dim, tau, delta)
    bias = np.zeros(2, dtype=float)
    if internal_shuffles > 0:
        # Y -> X: shuffle source Y only.
        vals = []
        for _ in range(internal_shuffles):
            yp = rng.permutation(data[:, 1])
            vals.append(raw_ste_pair(np.column_stack([data[:, 0], yp]), dim, tau, delta)[0][0])
        bias[0] = float(np.mean(vals))
        # X -> Y: shuffle source X only.
        vals = []
        for _ in range(internal_shuffles):
            xp = rng.permutation(data[:, 0])
            vals.append(raw_ste_pair(np.column_stack([xp, data[:, 1]]), dim, tau, delta)[0][1])
        bias[1] = float(np.mean(vals))
    nste = np.divide(ste - bias, H, out=np.zeros(2, dtype=float), where=np.abs(H) > EPS)
    return ste, nste, bias, n_eff


def calculate_window_all_tau(
    data: np.ndarray,
    dim: int,
    tau_values: list[int],
    delta: int,
    internal_shuffles: int,
    n_surrogates: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Calculate observed and surrogate STE/NSTE for one window across all tau.

    The outer null follows the laboratory implementation: Y is randomly permuted
    once per surrogate and that same permutation is reused across all tau values.
    The full STE/NSTE routine is recalculated. In addition to direction-specific
    null statistics, this function directly tests directional asymmetry
    (B->A minus A->B) using the corresponding surrogate asymmetry distribution.
    Monte Carlo p-values use the finite-sample +1 correction.
    """
    T = len(tau_values)
    STE = np.full((T, 2), np.nan)
    NSTE = np.full((T, 2), np.nan)
    BIAS = np.full((T, 2), np.nan)
    N_EFF = np.zeros(T, dtype=int)

    for i, tau in enumerate(tau_values):
        ste, nste, bias, n_eff = nste_pair(data, dim, tau, delta, internal_shuffles, rng)
        STE[i] = ste
        NSTE[i] = nste
        BIAS[i] = bias
        N_EFF[i] = n_eff

    nan22 = np.full((T, 2), np.nan)
    nan1 = np.full(T, np.nan)
    if n_surrogates <= 0:
        return {
            "STE": STE, "NSTE": NSTE, "BIAS": BIAS, "N_EFF": N_EFF,
            "STE_MEAN": nan22.copy(), "STE_SD": nan22.copy(),
            "STE_Z": nan22.copy(), "STE_P": nan22.copy(),
            "NSTE_MEAN": nan22.copy(), "NSTE_SD": nan22.copy(),
            "NSTE_Z": nan22.copy(), "NSTE_P": nan22.copy(),
            "STE_ASYM_NULL_MEAN": nan1.copy(), "STE_ASYM_NULL_SD": nan1.copy(),
            "STE_ASYM_Z": nan1.copy(), "STE_ASYM_P_TWO": nan1.copy(),
            "NSTE_ASYM_NULL_MEAN": nan1.copy(), "NSTE_ASYM_NULL_SD": nan1.copy(),
            "NSTE_ASYM_Z": nan1.copy(), "NSTE_ASYM_P_TWO": nan1.copy(),
        }

    perm_ste = np.empty((n_surrogates, T, 2), dtype=float)
    perm_nste = np.empty((n_surrogates, T, 2), dtype=float)
    for p in range(n_surrogates):
        yp = rng.permutation(data[:, 1])
        pdata = np.column_stack([data[:, 0], yp])
        for i, tau in enumerate(tau_values):
            ste_p, nste_p, _, _ = nste_pair(pdata, dim, tau, delta, internal_shuffles, rng)
            perm_ste[p, i] = ste_p
            perm_nste[p, i] = nste_p

    def summarize_directional(observed: np.ndarray, null: np.ndarray):
        mean = np.mean(null, axis=0)
        sd = np.std(null, axis=0, ddof=0)
        z = np.divide(observed - mean, sd, out=np.zeros_like(observed), where=sd > EPS)
        exceed = np.sum(null >= observed[None, :, :], axis=0)
        pval = (exceed + 1.0) / (n_surrogates + 1.0)
        return mean, sd, z, pval

    def summarize_asymmetry(observed_asym: np.ndarray, null_asym: np.ndarray):
        mean = np.mean(null_asym, axis=0)
        sd = np.std(null_asym, axis=0, ddof=0)
        z = np.divide(observed_asym - mean, sd, out=np.zeros_like(observed_asym), where=sd > EPS)
        observed_distance = np.abs(observed_asym - mean)
        null_distance = np.abs(null_asym - mean[None, :])
        exceed = np.sum(null_distance >= observed_distance[None, :], axis=0)
        p_two = (exceed + 1.0) / (n_surrogates + 1.0)
        return mean, sd, z, p_two

    ste_mean, ste_sd, ste_z, ste_p = summarize_directional(STE, perm_ste)
    nste_mean, nste_sd, nste_z, nste_p = summarize_directional(NSTE, perm_nste)

    ste_asym = STE[:, 0] - STE[:, 1]
    nste_asym = NSTE[:, 0] - NSTE[:, 1]
    perm_ste_asym = perm_ste[:, :, 0] - perm_ste[:, :, 1]
    perm_nste_asym = perm_nste[:, :, 0] - perm_nste[:, :, 1]
    ste_a_mean, ste_a_sd, ste_a_z, ste_a_p = summarize_asymmetry(ste_asym, perm_ste_asym)
    nste_a_mean, nste_a_sd, nste_a_z, nste_a_p = summarize_asymmetry(nste_asym, perm_nste_asym)

    return {
        "STE": STE, "NSTE": NSTE, "BIAS": BIAS, "N_EFF": N_EFF,
        "STE_MEAN": ste_mean, "STE_SD": ste_sd, "STE_Z": ste_z, "STE_P": ste_p,
        "NSTE_MEAN": nste_mean, "NSTE_SD": nste_sd, "NSTE_Z": nste_z, "NSTE_P": nste_p,
        "STE_ASYM_NULL_MEAN": ste_a_mean, "STE_ASYM_NULL_SD": ste_a_sd,
        "STE_ASYM_Z": ste_a_z, "STE_ASYM_P_TWO": ste_a_p,
        "NSTE_ASYM_NULL_MEAN": nste_a_mean, "NSTE_ASYM_NULL_SD": nste_a_sd,
        "NSTE_ASYM_Z": nste_a_z, "NSTE_ASYM_P_TWO": nste_a_p,
    }

def contiguous_valid_runs(time_s: np.ndarray, x: np.ndarray, y: np.ndarray, fs: float, tol: float = 0.02) -> list[np.ndarray]:
    valid = np.isfinite(time_s) & np.isfinite(x) & np.isfinite(y)
    idx = np.flatnonzero(valid)
    if len(idx) == 0:
        return []
    expected = 1.0 / fs
    breaks = [0]
    for k in range(1, len(idx)):
        nonadjacent = idx[k] != idx[k - 1] + 1
        dt_bad = abs((time_s[idx[k]] - time_s[idx[k - 1]]) - expected) > tol
        if nonadjacent or dt_bad:
            breaks.append(k)
    breaks.append(len(idx))
    return [idx[breaks[i]:breaks[i + 1]] for i in range(len(breaks) - 1) if breaks[i + 1] > breaks[i]]


def windows_from_run(run_idx: np.ndarray, window_n: int, step_n: int) -> list[np.ndarray]:
    if len(run_idx) < window_n:
        return []
    starts = range(0, len(run_idx) - window_n + 1, step_n)
    return [run_idx[s:s + window_n] for s in starts]


def extract_metadata(df: pd.DataFrame, file_path: Path, trial_summary: pd.DataFrame, reliability: pd.DataFrame, analysis_units: pd.DataFrame) -> dict[str, Any]:
    first = df.iloc[0].to_dict() if not df.empty else {}
    meta = {k: first.get(k, np.nan) for k in [
        "recording_folder", "dyad_id", "pair1", "participant_A", "participant_B", "sensor_A", "sensor_B",
        "candidate_window", "trial", "condition", "is_practice", "is_pilot", "exclude_from_main_analysis",
        "exclude_reason", "usable_for_dyadic_ecg", "accel_start_unix", "accel_end_unix", "window_duration_sec",
        "dyad_pretrial_qc_flag"
    ] if k in first}
    meta["paired_hr_filename"] = file_path.name
    meta["paired_hr_file"] = str(file_path)

    def attach(source: pd.DataFrame, file_cols: list[str], fields: list[str]) -> None:
        nonlocal meta
        if source.empty:
            return
        match = pd.DataFrame()
        for fc in file_cols:
            if fc in source.columns:
                match = source[source[fc].astype(str).map(lambda z: Path(z).name).eq(file_path.name)]
                if not match.empty:
                    break
        if match.empty:
            keys = [k for k in ["recording_folder", "dyad_id", "candidate_window", "trial", "condition"] if k in source.columns and k in meta]
            if keys:
                m = np.ones(len(source), dtype=bool)
                for k in keys:
                    m &= source[k].astype(str).to_numpy() == str(meta.get(k))
                match = source[m]
        if not match.empty:
            row = match.iloc[0]
            for f in fields:
                if f in row.index:
                    meta[f] = row[f]

    attach(trial_summary, ["output_file", "paired_hr_output_file"], [
        "recording_folder", "dyad_id", "candidate_window", "trial", "condition", "is_practice", "is_pilot",
        "exclude_from_main_analysis", "exclude_reason", "trial_segment_qc_flag", "trial_segment_qc_reasons",
        "coverage_A", "coverage_B", "n_saved_samples", "segment_duration_sec"
    ])
    attach(reliability, ["paired_hr_output_file", "output_file"], [
        "rr_reliability_A", "rr_reliability_B", "rr_reliability_label_A", "rr_reliability_label_B",
        "dyad_rr_reliability_min", "dyad_rr_reliability_mean", "dyad_rr_reliability_label"
    ])
    attach(analysis_units, [], [
        "is_practice", "is_pilot", "exclude_from_main_analysis", "exclude_reason", "usable_for_dyadic_ecg"
    ])
    meta["main_analysis_eligible"] = not bool_from_any(meta.get("exclude_from_main_analysis", True))
    return meta


def process_one(payload: dict[str, Any]) -> dict[str, Any]:
    path = Path(payload["file"])
    args = payload["args"]
    trial_summary = pd.DataFrame(payload["trial_summary"])
    reliability = pd.DataFrame(payload["reliability"])
    analysis_units = pd.DataFrame(payload["analysis_units"])
    try:
        df = pd.read_csv(path, dtype={"recording_folder": str, "dyad_id": str, "sensor_A": str, "sensor_B": str})
        required = ["TimeRelTrialSec", "HR_A_BPM", "HR_B_BPM"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        for c in required:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.sort_values("TimeRelTrialSec").reset_index(drop=True)
        meta = extract_metadata(df, path, trial_summary, reliability, analysis_units)
        t = df["TimeRelTrialSec"].to_numpy(float)
        xa = df["HR_A_BPM"].to_numpy(float)
        yb = df["HR_B_BPM"].to_numpy(float)
        runs = contiguous_valid_runs(t, xa, yb, args["fs"])
        if not runs:
            raise ValueError("No contiguous valid paired HR samples")

        # Match nste_main_CC.py: standardize the complete analysis segment before
        # creating overlapping windows. For TableTask, each contiguous valid run
        # is a segment because windows must not bridge missing/time gaps.
        xa_z = np.full_like(xa, np.nan, dtype=float)
        yb_z = np.full_like(yb, np.nan, dtype=float)
        usable_runs = []
        for run in runs:
            sx = np.std(xa[run], ddof=0)
            sy = np.std(yb[run], ddof=0)
            if not np.isfinite(sx) or not np.isfinite(sy) or sx <= 0 or sy <= 0:
                continue
            xa_z[run] = (xa[run] - np.mean(xa[run])) / sx
            yb_z[run] = (yb[run] - np.mean(yb[run])) / sy
            usable_runs.append(run)
        runs = usable_runs
        if not runs:
            raise ValueError("No non-constant contiguous paired HR segment")

        records: list[dict[str, Any]] = []
        tau_values = args["tau_values"]
        maxlag = max(1, int(round(args["max_prediction_lag_sec"] * args["fs"])))
        for ws in args["window_seconds"]:
            wn = int(round(ws * args["fs"]))
            stepn = max(1, int(round(wn * (1.0 - args["overlap"]))))
            global_window_index = 0
            for run_number, run in enumerate(runs, start=1):
                for widx in windows_from_run(run, wn, stepn):
                    global_window_index += 1
                    x = xa_z[widx].astype(float)
                    y = yb_z[widx].astype(float)
                    if not np.isfinite(x).all() or not np.isfinite(y).all():
                        continue
                    data = np.column_stack([x, y])
                    delta = prediction_delta(x, y, maxlag)
                    center_i = widx[len(widx) // 2]
                    valid_taus = [
                        tau for tau in tau_values
                        if len(x) - tau * (args["dim"] - 1) - delta > 0
                    ]
                    if not valid_taus:
                        continue
                    rng = np.random.default_rng(
                        stable_seed(args["seed"], path.name, ws, run_number, global_window_index)
                    )
                    calc = calculate_window_all_tau(
                        data=data, dim=args["dim"], tau_values=valid_taus, delta=delta,
                        internal_shuffles=args["internal_shuffles"],
                        n_surrogates=args["n_surrogates"], rng=rng,
                    )
                    for ti, tau in enumerate(valid_taus):
                        ste = calc["STE"][ti]
                        nste = calc["NSTE"][ti]
                        bias = calc["BIAS"][ti]
                        n_eff = int(calc["N_EFF"][ti])
                        pm, psd, z, pv = (calc["STE_MEAN"][ti], calc["STE_SD"][ti], calc["STE_Z"][ti], calc["STE_P"][ti])
                        npm, npsd, nz, npv = (calc["NSTE_MEAN"][ti], calc["NSTE_SD"][ti], calc["NSTE_Z"][ti], calc["NSTE_P"][ti])
                        ste_asym_null_mean = float(calc["STE_ASYM_NULL_MEAN"][ti])
                        ste_asym_null_sd = float(calc["STE_ASYM_NULL_SD"][ti])
                        ste_asym_z = float(calc["STE_ASYM_Z"][ti])
                        ste_asym_p_two = float(calc["STE_ASYM_P_TWO"][ti])
                        nste_asym_null_mean = float(calc["NSTE_ASYM_NULL_MEAN"][ti])
                        nste_asym_null_sd = float(calc["NSTE_ASYM_NULL_SD"][ti])
                        nste_asym_z = float(calc["NSTE_ASYM_Z"][ti])
                        nste_asym_p_two = float(calc["NSTE_ASYM_P_TWO"][ti])
                        rec = dict(meta)
                        rec.update({
                            "script_version": SCRIPT_VERSION,
                            "file": path.name,
                            "segment_id": int(meta.get("candidate_window", 0)) if pd.notna(meta.get("candidate_window", np.nan)) else 0,
                            "event": str(meta.get("condition", "")),
                            "d_col": "HR_A_BPM",
                            "r_col": "HR_B_BPM",
                            "win_size_sec": float(ws),
                            "win_step_sec": float(stepn / args["fs"]),
                            "window_index": global_window_index,
                            "finite_run_index": run_number,
                            "window_start_s": float(t[widx[0]]),
                            "window_end_s": float(t[widx[-1]]),
                            "time": float(t[center_i]),
                            "elapsed_time": float(t[center_i]),
                            "tau": int(tau),
                            "tau_seconds": float(tau / args["fs"]),
                            "embedding_dim": int(args["dim"]),
                            "prediction_delta_samples": int(delta),
                            "prediction_delta_seconds": float(delta / args["fs"]),
                            "n_window_samples": int(len(x)),
                            "n_effective_symbols": n_eff,
                            "small_effective_symbol_count_flag": bool(n_eff < ADVISORY_SMALL_EFFECTIVE_SYMBOLS),
                            "small_effective_symbol_advisory_threshold": int(ADVISORY_SMALL_EFFECTIVE_SYMBOLS),
                            "STE_YX": float(ste[0]), "STE_XY": float(ste[1]),
                            "NSTE_YX": float(nste[0]), "NSTE_XY": float(nste[1]),
                            "NSTE_bias_YX": float(bias[0]), "NSTE_bias_XY": float(bias[1]),
                            "perm_mean_YX": float(pm[0]), "perm_mean_XY": float(pm[1]),
                            "perm_std_YX": float(psd[0]), "perm_std_XY": float(psd[1]),
                            "Z_YX": float(z[0]), "Z_XY": float(z[1]),
                            "p_YX": float(pv[0]), "p_XY": float(pv[1]),
                            "NSTE_perm_mean_YX": float(npm[0]), "NSTE_perm_mean_XY": float(npm[1]),
                            "NSTE_perm_std_YX": float(npsd[0]), "NSTE_perm_std_XY": float(npsd[1]),
                            "NSTE_Z_YX": float(nz[0]), "NSTE_Z_XY": float(nz[1]),
                            "NSTE_p_YX": float(npv[0]), "NSTE_p_XY": float(npv[1]),
                            "STE_asym_null_mean": ste_asym_null_mean,
                            "STE_asym_null_sd": ste_asym_null_sd,
                            "STE_asym_Z": ste_asym_z,
                            "STE_asym_p_two": ste_asym_p_two,
                            "NSTE_asym_null_mean": nste_asym_null_mean,
                            "NSTE_asym_null_sd": nste_asym_null_sd,
                            "NSTE_asym_Z": nste_asym_z,
                            "NSTE_asym_p_two": nste_asym_p_two,
                            "asym": float(ste[0] - ste[1]),
                            "NSTE_asym": float(nste[0] - nste[1]),
                            "STE_B_to_A": float(ste[0]), "STE_A_to_B": float(ste[1]),
                            "NSTE_B_to_A": float(nste[0]), "NSTE_A_to_B": float(nste[1]),
                            "STE_asym_BminusA": float(ste[0] - ste[1]),
                            "NSTE_asym_BminusA": float(nste[0] - nste[1]),
                            "surrogate_method": "unrestricted_Y_permutation",
                            "n_surrogates": int(args["n_surrogates"]),
                            "internal_shuffles": int(args["internal_shuffles"]),
                        })
                        records.append(rec)
        if not records:
            raise ValueError("No valid window/tau combinations after length and quality checks")
        out = pd.DataFrame(records)
        return {"status": "success", "file": str(path), "records": out.to_dict("records"), "metadata": meta}
    except Exception as exc:
        return {"status": "failed", "file": str(path), "error": str(exc), "records": []}


def save_excel(df: pd.DataFrame, path: Path) -> None:
    try:
        df.to_excel(path, index=False)
    except Exception as exc:
        print(f"WARNING: Excel output failed for {path}: {exc}")



# ----------------------------- Plotting helpers -----------------------------
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
    alpha_tau = alpha / max(1, len(taus))
    p_col = "NSTE_asym_p_two" if measure == "NSTE" else "STE_asym_p_two"

    for i, tau in enumerate(taus):
        tsub = df[pd.to_numeric(df["tau"], errors="coerce") == tau].sort_values("plot_time_s")
        ax.plot(
            tsub["plot_time_s"], tsub["asymmetry_plot"], linewidth=1.0, alpha=0.65,
            linestyle=styles[i % len(styles)], label=f"tau={int(tau)}",
        )
        y = pd.to_numeric(tsub["asymmetry_plot"], errors="coerce")
        p_asym = pd.to_numeric(tsub.get(p_col, np.nan), errors="coerce")
        sig = p_asym < alpha_tau
        ax.scatter(
            tsub.loc[sig, "plot_time_s"], y[sig], s=18,
            facecolors="none", edgecolors="black", linewidths=0.7,
        )

    ax.axhline(0, linestyle="--", linewidth=1, color="black")
    ax.set_xlabel("Concatenated within-trial time (s; gaps separate candidate windows)")
    correction = "practice-corrected" if practice_correct else "raw"
    ax.set_ylabel(f"{measure} asymmetry, B→A minus A→B ({correction})")
    dyad = str(df["dyad_id"].iloc[0]) if "dyad_id" in df.columns else "NA"
    window_label = float(pd.to_numeric(df["win_size_sec"], errors="coerce").dropna().iloc[0])
    ax.set_title(f"{dyad}: {measure} directional asymmetry ({correction}, {window_label:g}-s windows)")
    ax.grid(True, alpha=0.2)

    boundaries = (
        df.groupby(["candidate_window_num", "trial_num", "condition"], dropna=False)["plot_time_s"]
          .agg(["min", "max"]).reset_index().sort_values("min")
    )
    for _, r in boundaries.iterrows():
        ax.axvline(r["min"], linestyle=":", linewidth=0.7, color="black", alpha=0.5)
        cw = r["candidate_window_num"]
        cw_label = f"cw{int(cw):02d}" if pd.notna(cw) else "cwNA"
        ax.text(
            (r["min"] + r["max"]) / 2, ax.get_ylim()[1], f"{cw_label} {r['condition']}",
            rotation=90, va="top", ha="center", fontsize=7,
        )
    ax.legend(title="Tau", ncol=4, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.13))
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "practice_corrected" if practice_correct else "raw"
    path = out_dir / f"{safe_piece(dyad)}__{measure}_asymmetry_{suffix}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return str(path)

def optimal_tau_descriptive(df: pd.DataFrame, window_sec: float, practice_correct: bool, sampling_rate: float) -> pd.DataFrame:
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
                    "optimal_tau_seconds_descriptive": tau / sampling_rate if np.isfinite(tau) else np.nan,
                    "max_mean_total_flow_descriptive": value,
                    "interpretation": "Descriptive parameter summary; not an inferentially selected primary tau.",
                })
    return pd.DataFrame(rows)



def generate_plots(all_df: pd.DataFrame, out_dir: Path, window_sec: float, alpha: float, practice_correct: bool, overwrite: bool, sampling_rate: float) -> pd.DataFrame:
    """Generate one NSTE and one STE asymmetry plot per dyad from the computed table."""
    plot_root = out_dir / "plots"
    if plot_root.exists() and overwrite:
        shutil.rmtree(plot_root)
    nste_dir = plot_root / "NSTE_asymmetry"
    ste_dir = plot_root / "STE_asymmetry"
    table_dir = out_dir / "tables"
    use = all_df[np.isclose(pd.to_numeric(all_df["win_size_sec"], errors="coerce"), window_sec)].copy()
    manifest_rows = []
    if use.empty:
        return pd.DataFrame(columns=["dyad_id", "measure", "status", "plot_file", "error"])
    for dyad, sub in use.groupby("dyad_id", dropna=False):
        for measure, folder in [("NSTE", nste_dir), ("STE", ste_dir)]:
            try:
                path = plot_one(sub, measure, folder, alpha, practice_correct)
                manifest_rows.append({"dyad_id": dyad, "measure": measure, "status": "saved", "plot_file": path, "error": ""})
            except Exception as exc:
                manifest_rows.append({"dyad_id": dyad, "measure": measure, "status": "failed", "plot_file": "", "error": str(exc)})
    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(table_dir / "05_nste_plot_manifest_EG.csv", index=False)
    opt = optimal_tau_descriptive(all_df, window_sec, practice_correct, sampling_rate)
    opt.to_csv(table_dir / "05_nste_optimal_tau_descriptive_EG.csv", index=False)
    save_excel(opt, table_dir / "05_nste_optimal_tau_descriptive_EG.xlsx")
    return manifest

def main() -> int:
    started = time.time()
    ns = parse_args()
    if ns.embedding_dim < 2:
        raise ValueError("--embedding-dim must be >=2")
    if not (0 <= ns.window_overlap < 1):
        raise ValueError("--window-overlap must be in [0,1)")
    if ns.tau_min < 1 or ns.tau_max < ns.tau_min or ns.tau_step < 1:
        raise ValueError("Invalid tau range")
    if ns.n_surrogates < 0 or ns.internal_shuffles < 0:
        raise ValueError("Shuffle/surrogate counts must be >=0")

    root = Path(ns.root).expanduser().resolve()
    paired_dir = resolve(root, ns.paired_hr_dir)
    out_dir = resolve(root, ns.out_dir)
    if not paired_dir.exists():
        raise FileNotFoundError(paired_dir)
    if out_dir.exists() and ns.overwrite:
        shutil.rmtree(out_dir)
    table_dir = out_dir / "tables"
    trial_dir = out_dir / "results_by_trial"
    table_dir.mkdir(parents=True, exist_ok=True)
    trial_dir.mkdir(parents=True, exist_ok=True)

    trial_summary = read_optional_csv(resolve(root, ns.trial_summary))
    reliability = read_optional_csv(resolve(root, ns.reliability_summary))
    analysis_units = read_optional_csv(resolve(root, ns.analysis_units))
    files = sorted(p for p in paired_dir.glob("*.csv") if p.is_file() and not p.name.startswith("._"))
    if ns.max_files >= 0:
        files = files[:ns.max_files]
    if not files:
        raise FileNotFoundError(f"No paired HR CSV files found in {paired_dir}")

    args = {
        "fs": float(ns.sampling_rate),
        "dim": int(ns.embedding_dim),
        "tau_values": list(range(ns.tau_min, ns.tau_max + 1, ns.tau_step)),
        "window_seconds": [float(x) for x in ns.window_seconds],
        "overlap": float(ns.window_overlap),
        "max_prediction_lag_sec": float(ns.max_prediction_lag_sec),
        "internal_shuffles": int(ns.internal_shuffles),
        "n_surrogates": int(ns.n_surrogates),
        "seed": int(ns.random_seed),
    }
    print(f"Script version: {SCRIPT_VERSION}")
    print(f"Paired HR files found: {len(files)}")
    print(f"Tau samples: {args['tau_values']}")
    print(f"Windows (s): {args['window_seconds']} | overlap={args['overlap']:.0%}")
    print(f"Surrogates: {args['n_surrogates']} (permutation; original laboratory outer null) | workers={ns.workers}")

    payload_base = {
        "args": args,
        "trial_summary": trial_summary.to_dict("records"),
        "reliability": reliability.to_dict("records"),
        "analysis_units": analysis_units.to_dict("records"),
    }
    all_parts: list[pd.DataFrame] = []
    manifest_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []

    def record_result(res: dict[str, Any]) -> None:
        file_name = Path(res["file"]).name
        if res["status"] == "success":
            df_one = pd.DataFrame(res["records"])
            out_name = f"{Path(file_name).stem}__NSTE.csv"
            out_path = trial_dir / out_name
            df_one.to_csv(out_path, index=False)
            all_parts.append(df_one)
            manifest_rows.append({
                "paired_hr_file": res["file"], "paired_hr_filename": file_name,
                "status": "success", "n_rows": len(df_one),
                "result_file": str(out_path), "error": "",
            })
        else:
            error = res.get("error", "")
            failure_rows.append({
                "paired_hr_file": res["file"],
                "paired_hr_filename": file_name,
                "error": error,
            })
            manifest_rows.append({
                "paired_hr_file": res["file"], "paired_hr_filename": file_name,
                "status": "failed", "n_rows": 0, "result_file": "", "error": error,
            })

        # Checkpoint manifests after every completed input file.
        pd.DataFrame(manifest_rows).sort_values("paired_hr_filename").to_csv(
            table_dir / "05_nste_file_manifest_partial_EG.csv", index=False
        )
        pd.DataFrame(
            failure_rows, columns=["paired_hr_file", "paired_hr_filename", "error"]
        ).to_csv(table_dir / "05_nste_failures_partial_EG.csv", index=False)

    if ns.workers == 1:
        for i, f in enumerate(files, 1):
            record_result(process_one(dict(payload_base, file=str(f))))
            print(f"  processed {i}/{len(files)} files")
    else:
        with ProcessPoolExecutor(max_workers=ns.workers) as ex:
            futs = {ex.submit(process_one, dict(payload_base, file=str(f))): f for f in files}
            done = 0
            for fut in as_completed(futs):
                record_result(fut.result())
                done += 1
                print(f"  processed {done}/{len(files)} files")

    all_df = pd.concat(all_parts, ignore_index=True) if all_parts else pd.DataFrame()
    manifest = pd.DataFrame(manifest_rows).sort_values("paired_hr_filename").reset_index(drop=True)
    failures = pd.DataFrame(failure_rows, columns=["paired_hr_file", "paired_hr_filename", "error"])
    all_csv = table_dir / "05_nste_all_window_tau_rows_EG.csv"
    all_df.to_csv(all_csv, index=False)
    save_excel(all_df, table_dir / "05_nste_all_window_tau_rows_EG.xlsx")
    manifest.to_csv(table_dir / "05_nste_file_manifest_EG.csv", index=False)
    failures.to_csv(table_dir / "05_nste_failures_EG.csv", index=False)

    plot_manifest = pd.DataFrame()
    if not ns.skip_plots and not all_df.empty:
        primary_window = float(args["window_seconds"][0])
        print(f"Generating STE/NSTE asymmetry plots for {primary_window:g}-s windows...")
        plot_manifest = generate_plots(
            all_df=all_df, out_dir=out_dir, window_sec=primary_window,
            alpha=float(ns.alpha), practice_correct=bool(ns.practice_correct),
            overwrite=bool(ns.overwrite), sampling_rate=float(ns.sampling_rate),
        )
        n_saved = int((plot_manifest["status"] == "saved").sum()) if not plot_manifest.empty else 0
        n_plot_failed = int((plot_manifest["status"] == "failed").sum()) if not plot_manifest.empty else 0
        print(f"Plots saved: {n_saved}; plot failures: {n_plot_failed}")

    elapsed = time.time() - started
    n_eligible_files = 0
    if not all_df.empty and "main_analysis_eligible" in all_df.columns:
        n_eligible_files = int(all_df.loc[all_df["main_analysis_eligible"].map(bool_from_any), "paired_hr_filename"].nunique())
    summary = {
        "script_version": SCRIPT_VERSION,
        "root": str(root),
        "paired_hr_dir": str(paired_dir),
        "out_dir": str(out_dir),
        "n_paired_hr_files_found": len(files),
        "n_successful_files": int((manifest["status"] == "success").sum()) if not manifest.empty else 0,
        "n_failed_files": len(failures),
        "n_output_rows": len(all_df),
        "n_main_analysis_eligible_files_labelled": n_eligible_files,
        "sampling_rate_hz": ns.sampling_rate,
        "embedding_dim": ns.embedding_dim,
        "tau_samples": args["tau_values"],
        "tau_seconds": [x / ns.sampling_rate for x in args["tau_values"]],
        "window_seconds": args["window_seconds"],
        "window_overlap": ns.window_overlap,
        "max_prediction_lag_sec": ns.max_prediction_lag_sec,
        "internal_shuffles": ns.internal_shuffles,
        "n_surrogates": ns.n_surrogates,
        "surrogate_method": "unrestricted_Y_permutation",
        "small_effective_symbol_advisory_threshold": ADVISORY_SMALL_EFFECTIVE_SYMBOLS,
        "random_seed": ns.random_seed,
        "workers": ns.workers,
        "plots_generated": bool(not ns.skip_plots),
        "n_plots_saved": int((plot_manifest["status"] == "saved").sum()) if not plot_manifest.empty else 0,
        "n_plot_failures": int((plot_manifest["status"] == "failed").sum()) if not plot_manifest.empty else 0,
        "practice_corrected_plots": bool(ns.practice_correct),
        "asymmetry_test": "two-sided Monte Carlo test against surrogate asymmetry distribution",
        "monte_carlo_p_value_correction": "(exceedances + 1) / (B + 1)",
        "elapsed_sec": elapsed,
    }
    with open(table_dir / "05_nste_run_summary_EG.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(table_dir / "05_nste_run_summary_EG.txt", "w", encoding="utf-8") as f:
        f.write("Step 5 TableTask NSTE run summary\n" + "=" * 42 + "\n")
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")
        f.write("\nDirection convention\n")
        f.write("YX = B -> A; XY = A -> B. Positive asym/NSTE_asym means B->A exceeds A->B.\n")
        f.write("All available Step 3 paired files are analyzed; QC/eligibility labels are retained but not used to filter.\n")
    (out_dir / "README_05_NSTE_outputs_EG.txt").write_text(
        "TableTask Step 5 STE/NSTE outputs\n"
        "================================\n\n"
        "Primary table: tables/05_nste_all_window_tau_rows_EG.csv\n"
        "Per-trial outputs: results_by_trial/*.csv\n"
        "Audit files: tables/05_nste_file_manifest_EG.csv and 05_nste_failures_EG.csv\n\n"
        "Directions: YX = participant B -> A; XY = participant A -> B.\n"
        "All Step 3 paired files are computed. Eligibility and QC fields are labels only.\n"
        "Direction-specific and direct asymmetry-specific Monte Carlo permutation p-values use the finite-sample +1 correction. Plot markers use the two-sided asymmetry p-value with Bonferroni correction across displayed tau values.\n"
        "No arbitrary effective-symbol cutoff is imposed; all mathematically valid estimates are retained and small counts are flagged descriptively.\n"
        "TableTask safeguard: windows never bridge missing-data or irregular-time gaps.\n",
        encoding="utf-8",
    )
    print("\nDone.")
    print(f"Successful files: {summary['n_successful_files']} / {len(files)}")
    print(f"Failures: {len(failures)}")
    print(f"Output rows: {len(all_df)}")
    print(f"Primary table: {all_csv}")
    plot_failures = int((plot_manifest["status"] == "failed").sum()) if not plot_manifest.empty else 0
    return 0 if len(failures) == 0 and plot_failures == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
