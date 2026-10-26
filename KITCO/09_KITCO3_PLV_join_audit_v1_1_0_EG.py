#!/usr/bin/env python3
"""KITCO3 Script09 v1.1.0: audit the existing Scripts 04-07 outputs.
Default project root: /Volumes/One Touch/KITCO3_drive
Run this file with python3; no v0.5.0 package and no Script08 results required.
Preserves the 1-second-step PLV estimator and Script07 additive three-term model.
"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import sys
import numpy as np
import pandas as pd
VERSION = "1.1.0"

def require(ok, message):
    if not bool(ok):
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def save(df, path):
    df.to_csv(path, index=False, float_format="%.17g")


def integer(series, name, allow_blank=False):
    s = series.astype(str).str.strip()
    if allow_blank:
        s = s.replace("", np.nan)
    n = pd.to_numeric(s, errors="raise")
    require((n.dropna() == np.floor(n.dropna())).all(), f"Noninteger {name}")
    require(allow_blank or n.notna().all(), f"Missing {name}")
    return n.astype("Int64" if allow_blank else "int64")


def boolean(series, name):
    lookup = {"true": True, "false": False}
    s = series.astype(str).str.lower().map(lookup)
    require(s.notna().all(), f"Invalid boolean in {name}")
    return s.astype(bool)


def union_seconds(frame):
    total, end = 0.0, -np.inf
    for a, b in sorted(zip(frame.core_start_s, frame.core_stop_s)):
        require(b > a, "Nonpositive core length")
        total += max(0.0, b - max(a, end))
        end = max(end, b)
    return total



def verify_source(directory, name):
    require(directory.is_dir(), f"Expected existing pipeline folder: {directory}")
    status = json.loads((directory / name).read_text())
    manifest = status.get("output_sha256", {})
    require(isinstance(manifest, dict) and len(manifest) > 0, f"No source manifest: {directory / name}")
    for relative, expected in manifest.items():
        f = (directory / relative).resolve()
        require(f.is_relative_to(directory), "Unsafe source manifest path")
        require(f.is_file() and sha(f) == expected, f"Source content changed or missing: {f}")
    return status, len(manifest)


def historical_status_hash(mapping, name, actual):
    values = [v for k, v in mapping.items() if Path(k).name == name]
    require(values == [sha(actual)], "Source lineage mismatch: " + name)


def main(args):
    root = args.project_root.expanduser().resolve()
    require(root.is_dir(), f"Project folder not accessible: {root}")
    book = (args.runsheet or root / "Runsheet_final_EG.xlsx").expanduser().resolve()
    defaults = {"04": "04_dyadic_signals_v1_0_0", "05": "05_PLV_feasibility_v1_0_0", "06": "06_timing_method_review_v1_0_0", "07": "07_cardiac_specification_v1_0_0"}
    dirs = {k: (getattr(args, "script" + k + "_dir") or root / "outputs" / v).expanduser().resolve() for k, v in defaults.items()}
    d4, d5, d6, d7 = [dirs[k] for k in ["04", "05", "06", "07"]]
    out = (args.output_dir or root / "outputs/09_PLV_condition_audit_v1_1_0").expanduser().resolve()
    require(not out.exists(), f"Output folder already exists: {out}. Choose a new --output-dir.")
    for directory in [root, out]:
        if str(directory).startswith("/Volumes/"):
            require(Path('/Volumes', directory.parts[2]).is_mount(), "External drive is not mounted")
    require(all(not (out == d or out in d.parents or d in out.parents) for d in dirs.values()), "Output cannot nest inside a source folder")
    sources = [(dirs[k], k + "_RUN_STATUS.json") for k in defaults]
    statuses, verified = {}, {}
    for k, (directory, name) in zip(defaults, sources):
        statuses[k], verified[k] = verify_source(directory, name)
    s4, s5, s6, s7 = [statuses[k] for k in ["04", "05", "06", "07"]]
    expected = ["CONDITIONAL_SIGNAL_EXPORT_COMPLETE", "CONDITIONAL_PLV_FEASIBILITY_COMPLETE", "TIMING_METHOD_AUDIT_COMPLETE_WITH_OPEN_DECISIONS", "CONDITIONAL_CARDIAC_SPECIFICATION_AND_COHORT_FROZEN"]
    require([statuses[k]["status"] for k in defaults] == expected, "Unexpected source completion status")
    require(s5["source_status_sha256"] == sha(d4 / "04_RUN_STATUS.json"), "Script 04/05 lineage mismatch")
    for k in ["04", "05"]:
        historical_status_hash(s6["input_sha256"], k + "_RUN_STATUS.json", dirs[k] / (k + "_RUN_STATUS.json"))
    for k in ["04", "05", "06"]:
        historical_status_hash(s7["input_status_sha256"], k + "_RUN_STATUS.json", dirs[k] / (k + "_RUN_STATUS.json"))
    spec = json.loads((d7 / "07_ANALYSIS_SPECIFICATION.json").read_text())
    require(sha(d7 / "07_ANALYSIS_SPECIFICATION.json") == "0e2262d5da526cfb6afa7ff262f548721e3e6c2cd5fc785a2226cdb3a97cefb4", "Frozen Script07 specification differs; review before proceeding")
    require(spec["primary_method"]["step_s"] == 1 and spec["primary_method"]["core_s"] == 200 and spec["primary_method"]["context_each_side_s"] == 40, "Wrong native estimator")
    require(spec["counts"]["scored_windows"] == 10434, "Wrong frozen window count")
    require(s4["physical_clock_verified"] is False and s5["physical_clock_verified"] is False, "Timing state differs")
    print("Verified existing Script04-07 manifests and frozen method.", flush=True)
    # 'NA' means neutral recipe A. It MUST NOT become a pandas missing value.
    r = pd.read_excel(book, sheet_name="Runsheet_Clean", dtype=str, keep_default_na=False)
    needed = {"Session_ID", "PID", "Phase", "Analysis_Partner", "Condition", "Planned_Partner"}
    require(needed.issubset(r.columns), f"Runsheet missing columns: {needed - set(r.columns)}")
    for c in needed:
        r[c] = r[c].str.strip()
    r["PID"] = integer(r.PID, "PID")
    r["Phase"] = integer(r.Phase, "Phase")
    r["Analysis_Partner"] = integer(r.Analysis_Partner, "Analysis_Partner", True)
    require(not r.duplicated(["Session_ID", "Phase", "PID"]).any(), "Duplicate participant-phase keys")
    require(len(r) == 216 and r.PID.nunique() == 108 and r.Session_ID.nunique() == 27,
            "Runsheet inventory differs from verified 216 rows / 108 PIDs / 27 sessions")
    require(set(r.Condition) == {"NA", "NB", "EA", "EB"}, "Unknown/missing condition code")
    require(r.groupby("PID").Session_ID.nunique().eq(1).all(), "Participants cross sessions; independence assumption fails")
    require(r.groupby("PID").Phase.apply(lambda x: sorted(x) == [1, 2]).all(), "Incorrect PID phase structure")
    require(r.groupby("Session_ID").PID.nunique().eq(4).all(), "Not four scheduled participants per session")
    require(r.groupby(["Session_ID", "Phase"]).Condition.nunique().eq(1).all(), "Within-phase condition conflict")
    sequences = {}
    for sid, g in r.groupby("Session_ID", sort=True):
        codes = g.groupby("Phase").Condition.first().to_dict()
        require(set(codes) == {1, 2} and set(codes.values()) in ({"NA", "EB"}, {"NB", "EA"}),
                f"Invalid complementary sequence: {sid}: {codes}")
        ep = next(k for k, v in codes.items() if v.startswith("E"))
        sequences[sid] = f"{codes[ep]}_P{ep}"
    indexed = r.set_index(["Session_ID", "Phase", "PID"], verify_integrity=True)
    pairs = []
    for row in r.itertuples(index=False):
        if pd.isna(row.Analysis_Partner):
            continue
        partner = int(row.Analysis_Partner)
        key = (row.Session_ID, row.Phase, partner)
        require(partner != row.PID and key in indexed.index, f"Invalid partner for PID {row.PID}")
        other = indexed.loc[key]
        require(pd.notna(other.Analysis_Partner) and int(other.Analysis_Partner) == row.PID,
                f"Nonreciprocal actual partners: {key}")
        require(other.Condition == row.Condition, f"Partner condition mismatch: {key}")
        if row.PID < partner:
            pairs.append({"pair_id": f"{row.Session_ID}_P{row.Phase}_{row.PID:03d}_{partner:03d}",
                          "Session_ID": row.Session_ID, "Phase": row.Phase,
                          "PID_1": row.PID, "PID_2": partner, "Condition": row.Condition,
                          "story": "Emotional" if row.Condition[0] == "E" else "Neutral",
                          "recipe": row.Condition[1], "sequence_cell": sequences[row.Session_ID]})
    meta = pd.DataFrame(pairs).sort_values(["Session_ID", "Phase", "PID_1"]).reset_index(drop=True)
    require(len(meta) == 94 and not meta.pair_id.duplicated().any(), "Expected 94 unique reciprocal active pair phases")

    keys = ["pair_id", "Session_ID", "Phase", "PID_1", "PID_2"]
    reg = pd.read_csv(d7 / "07_pair_phase_register_71.csv", keep_default_na=False, na_values=[""])
    for col in ["conditional_eligible", "primary_condition_comparison_member"]:
        reg[col] = boolean(reg[col], col)
    require(len(reg) == 71 and reg.pair_id.is_unique, "Expected 71 candidate pair phases")
    joined = reg.merge(meta, on=keys, how="left", validate="one_to_one", suffixes=("_recorded", ""), indicator=True)
    require(joined._merge.eq("both").all(), "Script 07 pairs disagree with runsheet actual partners")
    require(joined.Condition_recorded.eq(joined.Condition).all(), "Runsheet/Script 07 condition mismatch")
    require(joined.Story.eq(joined.story).all(), "Story coding mismatch")
    require(joined.Recipe.eq(joined.recipe.map({"A": "Apple", "B": "Banana"})).all(), "Recipe coding mismatch")
    reg = joined.drop(columns=["_merge", "Condition_recorded", "story", "recipe", "sequence_cell"])
    # Reconcile unchanged earlier results against the frozen Script 07 register.
    base = pd.read_csv(d4 / "04_pair_signal_register_71.csv", keep_default_na=False)
    require(set(base.pair_id) == set(reg.pair_id), "Script 04/07 pair membership mismatch")
    base_keys = base[keys].merge(reg[keys], on=keys, how="outer", indicator=True, validate="one_to_one")
    require(base_keys._merge.eq("both").all(), "Script 04/07 identity mismatch")
    windows = pd.read_csv(d5 / "05_working_window_diagnostics.csv", keep_default_na=False)
    segments = pd.read_csv(d4 / "04_paired_interpolation_segments.csv", keep_default_na=False)
    require(not windows.duplicated(["pair_id", "paired_segment_id", "core_start_s"]).any(), "Duplicate windows")
    require(set(windows.pair_id).issubset(set(reg.pair_id)), "Unknown window pair")
    require(np.isfinite(windows.PLV).all() and windows.PLV.between(0, 1).all(), "Invalid PLV")
    require(windows.status.eq("CONDITIONAL_DIAGNOSTIC_SCORE").all(), "Unexpected score status")
    require(windows.core_duration_s.eq(200).all() and windows.context_each_side_s.eq(40).all(), "Unexpected windows")
    support = segments.set_index(["pair_id", "paired_segment_id"], verify_integrity=True)
    require(len(windows) == s7["scored_windows"] == 10434, "Frozen window count mismatch")
    # union_seconds expects core_stop_s; all times remain relative to cooking onset.
    windows["core_stop_s"] = windows.core_start_s + 200
    for row in reg.itertuples(index=False):
        w = windows[windows.pair_id == row.pair_id]
        require(len(w) == row.scored_windows, "Window count differs for " + row.pair_id)
        require(row.shared_cooking_s > 0, "Invalid cooking duration")
        require(bool(len(w)) == row.conditional_eligible, "Eligibility differs from score support")
        if len(w):
            require(abs(w.PLV.mean() - row.mean_local_PLV) < 2e-9, "Pair mean differs for " + row.pair_id)
        else:
            require(pd.isna(row.mean_local_PLV), "Unscored phase has a score")
        covered = union_seconds(w)
        require(abs(covered - row.scored_core_union_s) < 1e-6, "Coverage union mismatch")
        require(abs(covered / row.shared_cooking_s - row.scored_core_coverage_fraction) < 1e-8, "Coverage ratio mismatch")
        for sid, g in w.groupby("paired_segment_id"):
            require((row.pair_id, sid) in support.index, "Unknown continuous support segment")
            z = support.loc[(row.pair_id, sid)]
            t = g.core_start_s.to_numpy(float)
            require(np.allclose(t, np.round(t)) and (t >= 40).all(), "Invalid 1-second window anchors")
            require((t - 40 >= z.support_start_s - 1e-7).all() and (t + 240 <= z.support_stop_s + 1e-7).all(), "Window crosses support gap")
            require((t - 40 >= z.first_grid_s - 1e-7).all() and (t + 239.75 <= z.last_grid_s + 1e-7).all(), "Window exceeds grid support")
    cohort = pd.read_csv(d7 / "07_conditional_cardiac_dataset.csv", keep_default_na=False, na_values=[""])
    require(set(cohort.pair_id) == set(reg.loc[reg.conditional_eligible, "pair_id"]), "Frozen cohort mismatch")
    source = reg.set_index("pair_id")
    for col in ["mean_local_PLV", "scored_windows", "scored_core_union_s"]:
        require(np.allclose(cohort[col], source.loc[cohort.pair_id, col]), "Frozen cohort value mismatch")
    payload = reg.drop(columns=["Condition", "Story", "Recipe"])
    pairs = meta.merge(payload, on=keys, how="left", validate="one_to_one", indicator=True)
    pairs["raw_selected"] = pairs._merge.eq("both")
    pairs["scored"] = pairs.mean_local_PLV.notna()
    pairs["scored_windows"] = pairs.scored_windows.fillna(0).astype(int)
    pairs["disposition"] = np.where(~pairs.raw_selected, "NO_SELECTED_SOURCE_IN_SCRIPT04", np.where(pairs.scored, "SCORED", pairs.inclusion_reason.fillna("")))
    pairs = pairs.drop(columns="_merge")
    session_rows, contrasts = [], []
    for (sid, story), g in pairs.groupby(["Session_ID", "story"], sort=True):
        q = g[g.scored]
        session_rows.append({"Session_ID": sid, "story": story, "Phase": int(g.Phase.iloc[0]),
                             "recipe": g.recipe.iloc[0], "sequence_cell": g.sequence_cell.iloc[0],
                             "active_pair_phases": len(g), "raw_selected_pair_phases": int(g.raw_selected.sum()),
                             "scored_pair_phases": len(q), "mean_pair_phase_PLV": q.mean_local_PLV.mean(),
                             "mean_scored_coverage": q.scored_core_coverage_fraction.mean(),
                             "supported_windows": int(q.scored_windows.sum()),
                             "scored_PIDs": ";".join(map(str, sorted(set(q.PID_1) | set(q.PID_2))))})
    sessions = pd.DataFrame(session_rows)
    for sid, g in sessions.groupby("Session_ID", sort=True):
        e, n = g[g.story == "Emotional"].iloc[0], g[g.story == "Neutral"].iloc[0]
        if not (e.scored_pair_phases and n.scored_pair_phases):
            continue
        contrasts.append({"Session_ID": sid, "emotional_pair_phases": int(e.scored_pair_phases),
                          "neutral_pair_phases": int(n.scored_pair_phases),
                          "delta_banana_proportion": float(e.recipe == "B") - float(n.recipe == "B"),
                          "delta_phase2_proportion": float(e.Phase == 2) - float(n.Phase == 2),
                          "emotional_scored_PIDs": e.scored_PIDs, "neutral_scored_PIDs": n.scored_PIDs,
                          "same_scored_participant_set": e.scored_PIDs == n.scored_PIDs})
    design = pd.DataFrame(contrasts)
    frozen_design = pd.read_csv(d7 / "07_primary_session_design.csv")
    require(list(design.Session_ID) == list(frozen_design.Session_ID), "Matched-session membership changed")
    for col in ["delta_banana_proportion", "delta_phase2_proportion", "emotional_pair_phases", "neutral_pair_phases"]:
        require(np.allclose(design[col], frozen_design[col]), "Frozen design mismatch: " + col)
    X = np.column_stack([np.ones(len(design)), design.delta_banana_proportion, design.delta_phase2_proportion])
    rank = int(np.linalg.matrix_rank(X))
    require(rank == 3 and len(X) > 3, "Frozen additive recipe/order model is not identifiable")
    hat = np.sum((X @ np.linalg.inv(X.T @ X)) * X, axis=1)
    require((hat < 1 - 1e-10).all(), "Unit leverage prevents HC3")
    design["leverage"] = hat
    require(np.allclose(hat, frozen_design.design_leverage), "Frozen leverage mismatch")
    require(all(np.linalg.matrix_rank(np.delete(X, i, axis=0)) == 3 for i in range(len(X))), "Leave-one-session-out rank failure")
    matched = set(design.Session_ID)
    scored = pairs[pairs.scored]
    primary = scored[scored.Session_ID.isin(matched)]
    expected_primary = set(reg.loc[reg.primary_condition_comparison_member, "pair_id"])
    require(set(primary.pair_id) == expected_primary, "Primary pair membership mismatch")
    counts = {"active_pair_phases": len(pairs), "selected_candidate_pair_phases": len(reg),
              "scored_windows": len(windows), "scored_pair_phases": len(scored),
              "sessions_with_scores": scored.Session_ID.nunique(), "matched_sessions": len(design),
              "primary_comparison_pair_phases": len(primary),
              "scored_pair_phases_descriptive_only": len(scored) - len(primary)}
    require((len(scored), len(design), len(primary)) == (48, 16, 42), "Frozen sample count differs")
    summary = []
    for cols in [["story"], ["recipe"], ["Phase"], ["Condition"], ["story", "recipe", "Phase"]]:
        for key, g in pairs.groupby(cols, sort=True):
            key = key if isinstance(key, tuple) else (key,)
            q = g[g.scored]
            summary.append({"grouping": "+".join(cols), "group": "|".join(map(str, key)),
                            "active_pair_phases": len(g), "selected_candidate_pair_phases": int(g.raw_selected.sum()),
                            "scored_pair_phases": len(q), "scored_fraction_of_active": len(q) / len(g),
                            "sessions_with_scores": q.Session_ID.nunique(), "supported_windows": int(q.scored_windows.sum()),
                            "coverage_median_scored": q.scored_core_coverage_fraction.median(),
                            "coverage_min_scored": q.scored_core_coverage_fraction.min(),
                            "coverage_max_scored": q.scored_core_coverage_fraction.max()})
    # Complete validation precedes any output directory or scientific output.
    out.mkdir(parents=True, exist_ok=False)
    snapshot = out / "source_script07"
    snapshot.mkdir()
    for name in list(s7["output_sha256"]) + ["07_RUN_STATUS.json"]:
        shutil.copyfile(d7 / name, snapshot / name)
    save(pairs, out / "09_pair_phase_joined_94.csv")
    save(sessions, out / "09_session_condition_summary_54.csv")
    save(design, out / "09_matched_session_design.csv")
    save(pd.DataFrame(summary), out / "09_availability_coverage_by_design.csv")
    save(pairs[~pairs.scored], out / "09_unscored_active_pair_phases.csv")
    save(pairs.groupby(["Condition", "disposition"]).size().rename("pair_phases").reset_index(), out / "09_dispositions.csv")
    save(windows.merge(meta, on="pair_id", validate="many_to_one"), out / "09_window_traceability_10434.csv")
    write_json(out / "09_design_audit.json", {"rank": rank, "columns": 3, "matched_sessions": len(design),
               "residual_df": len(design) - 3, "max_leverage": float(hat.max()), "analysis_plan_changed": False,
               "model": spec["primary_statistics"]["model"], "step_s": spec["primary_method"]["step_s"]})
    write_json(out / "09_flow_counts.json", counts)
    (out / "09_AUDIT_REPORT.txt").write_text("KITCO3 native-pipeline audit: PASS\n\n" + json.dumps(counts, indent=2) +
        "\n\nSources: existing Scripts 04, 05, 06 and frozen 07. Script 08 result values were not used.\n"
        "The separate v0.5.0/125-window package is not an input.\n"
        "Native method: 200 s cores, 40 s margins, 4 Hz, 1 s window steps. Windows are dependent.\n"
        "Preserved model: D_session ~ intercept + delta_banana_proportion + delta_phase2_proportion.\n"
        "Equal weight per session; HC3/t reference with S-3 df. No recipe-by-order term added.\n"
        "No new QC, PLV recalculation, exclusions or condition test. Physical offset/drift remains unmeasured.\n"
        "Source absence is not proof that original recording data never existed.\n", encoding="utf-8")
    write_json(out / "09_provenance.json", {"version": VERSION, "script_sha256": sha(Path(__file__)),
               "runsheet_sha256": sha(book), "source_status_sha256": {str(d / n): sha(d / n) for d, n in sources},
               "source_output_hashes_verified": verified, "native_sources": True,
               "separate_v0_5_package_used": False, "script08_results_used": False,
               "input_paths": {"project_root": str(root), "runsheet": str(book), "script04": str(d4), "script05": str(d5), "script06": str(d6), "script07": str(d7)}})
    # Recheck before completion to catch source edits during this run.
    for d, n in sources:
        verify_source(d, n)
    for name, expected in s7["output_sha256"].items():
        require(sha(snapshot / name) == expected, "Source snapshot differs")
    manifest = {str(f.relative_to(out)): sha(f) for f in sorted(out.rglob("*")) if f.is_file()}
    write_json(out / "09_RUN_STATUS.json", {"status": "PASS_NATIVE_PIPELINE_AUDIT", "version": VERSION,
               "counts": counts, "output_sha256": manifest, "plan_changed": False})
    print("PASS: native pipeline audit\n" + json.dumps(counts, indent=2) + "\nSaved: " + str(out))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-root", type=Path, default=Path("/Volumes/One Touch/KITCO3_drive"))
    p.add_argument("--runsheet", type=Path)
    for k in ["04", "05", "06", "07"]:
        p.add_argument("--script" + k + "-dir", type=Path)
    p.add_argument("--output-dir", type=Path)
    a = p.parse_args()
    try:
        main(a)
    except Exception as exc:
        print("SCRIPT09 STOPPED: " + str(exc) + "\nNo completion status was published. Preserve existing files.", file=sys.stderr)
        sys.exit(1)
