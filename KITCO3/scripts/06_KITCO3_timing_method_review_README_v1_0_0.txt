# KITCO3 Script06 v1.0.0: timing and method review

## Purpose and scope
The immediate scientific step is to review feasibility and available timing
evidence before choosing/fixing the primary analysis. This script audits existing
outputs and creates review tables, an evidence inventory and a DRAFT analysis
specification. It does not establish physical synchronization merely from an
exported timestamp, a detected motion peak, file names or a second time column.
It does not recompute ECG, RR intervals or PLV, change QC, select the most
favourable specification, freeze inclusion or test a condition effect.

## Run on Emma's Mac
Save the script in /Volumes/One Touch/KITCO3_drive/scripts and run:
python3 "/Volumes/One Touch/KITCO3_drive/scripts/06_KITCO3_timing_method_review_v1_0_0_EG.py"

It reads existing outputs/04_dyadic_signals_v1_0_0 and
outputs/05_PLV_feasibility_v1_0_0 on the external project drive.
It writes a NEW folder on INTERNAL storage:
~/KITCO3_outputs/06_timing_method_review_v1_0_0
No existing folder is overwritten. Use --output-dir with a fresh path to rerun.
It uses only numpy and pandas in addition to the Python standard library.
The original Script05 .py file is not needed: sealed result files are the inputs.

Optional flags: --project-root, --script03-dir, --script04-dir, --script05-dir, --script02-dir,
--raw-data-root, --output-dir, --plotting-dir PATH, --evidence-root PATH.
The last two options can be repeated to include additional folders. The default
plotting checks cover the external v1.1.0 output and the previously suggested
internal local_run1 output. Supply your actual successful --plotting-dir if
you used a different name. Plot completion is reported separately and does
not determine physiological inclusion or invalidate already verified 04/05 data.

## What is checked
All Script04/05 output hashes, their lineage link, all 71 working summaries,
per-phase window counts, mean PLVs and union coverage are checked. Condition
code NA is read as a literal Neutral-Apple code, not a missing value. Story,
recipe and participant/session metadata are used only for coverage/design
description. No emotional-versus-neutral PLV difference is computed.
The existing Script03 raw timestamp audit is reused when a unique matching
copy is available. Its iPad labels and exported-clock diagnostics are preserved
as prior evidence, not promoted to physical calibration. No full ECG timing
audit is repeated. Existing same-core sensitivity scores are aggregated per pair phase; windows
are not treated as independent N. Duration comparisons use common pair phases
but different core lengths and potentially different periods. They are labelled
accordingly and cannot establish the best window. Synthetic independent-noise
PLV distributions are method diagnostics, not a matched human null test.

## Timing evidence inventory and limitations
The data folder (plus any --evidence-root) is traversed read-only. Relevant file
names, video/document availability, CSV/header metadata and first-level ZIP
members are inventoried. At most 64 KiB of each text file is read; no complete
accelerometer waveform, video or manual document review is performed. The
inventory does not assume timestamp units or calculate a physical rate from
an unexplained numeric column. Sensor/host-named time fields are only review
candidates. Files in the same acquisition directory are associated as candidate
evidence, not proof of common events or a shared clock. Original Script02
record metadata supply candidate acquisition-directory associations when present.
Nested archives are explicitly listed as NOT INSPECTED. Encrypted/unreadable
files and inaccessible directories are reported as incomplete inventory, never
silently interpreted as no evidence. If motion/video files were stored elsewhere,
add their folders with --evidence-root; not found does not mean not collected.
No events are matched or offsets/drifts fitted in this release. That requires
identifiable shared events, justified matching, clock-domain mappings and an
uncertainty assessment. Do not align by maximizing ECG/PLV similarity.

## Outputs and interpretation
06_REVIEW_REPORT.md is the main readable result.
06_plotting_completion.csv records checked plotting directories and file hashes.
06_timing_file_inventory.csv, 06_timing_inventory_errors.csv and
06_recording_timing_context.csv describe timing evidence scope and gaps.
06_timing_evidence_review_template.csv gives one unresolved row per pair phase
for documenting independent evidence; this script does not consume or validate
manual entries and does not change alignment based on them.
06_pair_phase_review_71.csv and 06_coverage_by_design.csv describe availability.
06_method_support.csv, 06_matched_core_sensitivity_summary.csv,
06_duration_common_pair_comparisons.csv, 06_synthetic_duration_summary.csv and
06_synthetic_context_summary.csv review existing estimator diagnostics.
06_design_dependence.json records session/participant reuse, not a fitted model.
06_DRAFT_ANALYSIS_SPEC.json and 06_decisions_to_resolve.csv list scientific
decisions that cannot be made by a file inventory or estimator identity test.
06_RUN_STATUS.json is written last and seals outputs. AUDIT_COMPLETE means
the report was generated; it does not mean timing verified or analysis finalized.

## What remains after the script
Review the evidence inventory for actual calibration/shared events and assess
residual offset/drift uncertainty. If independent evidence cannot resolve it,
explicitly justify the exported-time assumption and prespecify sensitivity.
Choose the estimand (mean local versus whole-interaction phase consistency),
signal, band, core/context, coverage and aggregation rules for the research
question. Finalize an autocorrelation-preserving reference approach and
participant/session-dependent condition model only after verifying the original
assignment, counterbalancing and exchangeability. Then freeze inclusion and
the analysis plan before running the condition effect. Prior exploratory work
has been viewed, so this is not a preregistration or proof of outcome blindness.
200 s remains a working candidate; no universal number of cycles guarantees
reliable phase. Do not choose a method because it produces a larger condition
effect. All existing sources remain unchanged.
