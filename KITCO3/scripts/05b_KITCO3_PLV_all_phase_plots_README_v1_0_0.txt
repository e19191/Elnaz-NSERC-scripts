# KITCO3 all-phase PLV figures, Script05b v1.0.0

These are publication-format descriptive figures, not final inferential results.
All 71 pair phases are retained in identity order. All 10,434 currently scored
windows are displayed. The 23 phases without a working score retain a status
figure. No high/low example selection, smoothing, new QC or condition test occurs.

## Panels in every pair-phase figure
A: Both interpolated IBI series, with gaps preserved. Values are in milliseconds.
B: Sealed Script05 window PLV at each 200-second core midpoint; y-axis is 0–1.
Dashed mean is the arithmetic mean of local PLVs. It is not a null threshold.
C: Paired sampled support and the union of scored cores. Coverage is not the
sum of overlapping window durations. Blank space does not mean PLV equals zero.
D/E: Filtered IBI within EVERY scored core, using the original context-specific
processing. Each heatmap column is a separate core. Rows are seconds within it.
F: Wrapped partner-one minus partner-two phase within every scored core, with
a circular colormap from -pi to pi. Consistency within a column determines PLV;
zero phase difference is not required for high PLV. Adjacent columns overlap.

Context phase cannot accurately be represented by filtering the whole interaction
once. This script imports the unchanged Script05 implementation, repeats its
context transforms and checks every reconstructed PLV against the sealed scores
to an absolute tolerance of 1e-8. It DOES NOT rerun QC or synthetic simulations.
No contexts cross an unsupported segment. No new clock correction is fitted.
Heatmaps retain all 800 core samples per column, without temporal downsampling.
Raster rendering limits visible pixel detail; PDFs retain vector labels and lines.

## Processing and scales
4 Hz midpoint-interpolated IBI; 200 s cores; 40 s context per side; 1 s start step;
0.04–0.15 Hz order-3 Butterworth SOS forward/backward filtering; linear detrend
and population-SD standardization of each 280 s context; FFT Hilbert phase.
Filtered units refer to standardized pre-filter context, not ECG amplitude.
Both filtered heatmaps use one symmetric global color limit across all phases,
chosen as the maximum absolute filtered value. No color clipping is applied.
IBI y-ranges adapt per phase for readability; PLV and phase scales are fixed.
Overview durations share a common scale. Overview pages use one point per phase,
with unavailable values explicitly labelled. No window-based SEM/CI is supplied.

## Interpretation limits
All results assume exported time represents common physical time. Sensor offset
and drift remain unverified. PLV is band-limited IBI phase consistency, not proof
of interpersonal causation, common cardiac oscillation or a condition effect.
No arbitrary strong/moderate synchrony cutoffs are drawn. These are method
diagnostics from the working specification, not a final included cohort.

## Outputs
pairs/: 71 individual PDF and PNG figures (PNG 300 dpi).
overview/: three identity-ordered PDF/PNG summary pages, covering all 71 phases.
05b_PLV_all_phases_part_01/02/03.pdf: readable 24/24/23-page pair-phase books.
05b_PLV_overview.pdf: combined three-page overview.
05b_figure_index.csv: one row per phase, status and figure path.
05b_RUN_STATUS.json: completion written LAST, input/output hashes, environment,
number of windows reproduced, and maximum PLV discrepancy.

## Reproduce on Emma's Mac
Save this script next to the ORIGINAL unchanged Script05 v1.0.0 in:
/Volumes/One Touch/KITCO3_drive/scripts
Run:
python3 "/Volumes/One Touch/KITCO3_drive/scripts/05b_KITCO3_PLV_all_phase_plots_v1_0_0_EG.py"

Defaults read outputs/04_dyadic_signals_v1_0_0 and
outputs/05_PLV_feasibility_v1_0_0 and write a NEW
outputs/05b_PLV_all_phase_plots_v1_0_0. If it already exists, choose a fresh
--output-dir. Optional --project-root, --script04-dir, --script05-dir,
--processing-script and --output-dir take explicit paths.
Requires Python >=3.10, numpy, pandas, scipy, matplotlib. Inputs are read-only.
Do not edit Script05 to satisfy a checksum. Use the exact original script.

Suggested caption: Conditional phase consistency of partners' LF-filtered IBI
signals during cooking. PLV was calculated within overlapping 200 s cores with
40 s real context on each side. Every supported scoring core is represented;
window centers are shown on the cooking-time axis. White regions indicate no
scored core. Estimates assume common exported time; physical clock offset/drift
has not been independently verified. Overlapping windows are not independent.
