# KITCO3 publication figures v1.1.0

This is a visualization update, not a new synchrony estimator or final analysis.
All 71 pair phases remain. All 10,434 currently scored windows are selectable.
The current working window remains 200 s with 40 s context per side, 1 s start
step, and LF 0.04-0.15 Hz. Final scientific specification is not frozen.

## What changed
Heatmaps are removed from these main figures. They were mathematically valid
context displays, but repeated overlapping windows create diagonal structures
that are poor stand-alone evidence of interpersonal phase locking. Prior output
files remain unchanged. No claim is made that heatmaps are generally invalid.
Every ECG-derived IBI is labelled RR interval (IBI); RR is displayed in ms.
One three-page PDF per pair phase integrates the RR/PLV page with a raw/cleaned
ECG page for each partner. Three booklets collect all pair PDFs. Separate PNGs
are written at 300 dpi. The overview includes all phases and missing estimates.

## Main page
Paired RR intervals are plotted with original gaps. Both a 0-1 PLV view and a
shared enlarged view show exactly the same sealed scores; no curve smoothing.
The enlarged limit is the global maximum across all scored windows rounded up
to 0.05, with a minimum limit of 0.25. No score is clipped. This display choice
does not choose an analysis window or inclusion rule. Mean line is descriptive,
not a null or significance threshold. Coverage is union duration, not a sum of
overlapping windows. A schematic explains core, context and midpoint.
PLV is the magnitude of the mean unit phase-difference vector per 200 s core.
The pair-phase summary is mean local PLV, not pooled whole-interaction PLV.

## Window viewers
Open index.html locally, then choose a pair. Each pair has a self-contained
HTML viewer with a selector and slider for EVERY scored window. Initial window
is first by segment/time, never ranked by PLV. The viewer does not run the
estimator: it displays validated Python results. Full 280 s context-filtered
RR signals and a wrapped phase-difference trace are linked to the selected PLV.
Grey shading identifies the 40 s contexts; only the central 200 s enters PLV.
The circular histogram uses 18 bins of 20 degrees; radial values are fractions
of the 800 core samples. Its radial scale adapts to the selected distribution,
is numerically labelled, and must not be used for cross-window visual effect
size comparisons. Compare the labelled PLV instead. Histograms are not
significance tests and samples are autocorrelated. No pooled phase histogram
is used as a substitute for mean local PLV. Each full-precision PLV is checked
against sealed Script05 to tolerance 1e-8. Float32 display phases reproduce PLV
within 1e-7. Nothing is temporally downsampled in context diagnostics.
Viewers work offline in a current browser supporting DecompressionStream
(current Chrome recommended). No external scripts, network, or server needed.
Use static PDFs for sharing scientific figures; HTML is for detailed inspection.

## Raw versus cleaned ECG
All 142 original ECG CSV hashes must match the frozen source inventory embedded
in the script. Read-only source paths come from original Script02 metadata.
Full-recording cleaning is reconstructed, per recorded processing segment,
using original design rate, powerline policy and NeuroKit2 0.2.13. Script02 did
not save cleaned arrays, so this is a documented reconstruction, not a claim of
byte-identical recovery of original cleaned samples. No peak detection/correction
is rerun. Final Script03 peak source indices, values and timestamps are checked.
Green circles show trusted peaks; purple crosses untrusted retained peaks.
Withheld spans are grey. The full-cooking overview uses per-bin min/max source
samples for display only. Three fixed 10 s sections centred at 25%, 50%, 75%
of cooking show all raw samples. They are not selected for clean morphology or
high PLV. Raw/cleaned plots share amplitude limits within a row, but amplitude
limits may vary between rows. Units are exported units, not assumed microvolts.
These pages are QC displays, not independent human adjudication or new QC rules.

## Limitations
No sensor-clock offset/drift is calibrated, fitted or corrected. RR phase is an
operational band-limited measure, not proof of one physiological oscillator.
No arbitrary low/moderate/high synchrony thresholds, condition p-values, window
SEM/CI, surrogate tests or final cohort freeze are added. Positive PLV alone
does not establish interpersonal coupling. Reference and statistical analyses
remain separate work. Scientific results are unchanged by this plotting update.

## Run on Emma's Mac
Save this script and original 05_KITCO3_PLV_feasibility_v1_0_0_EG.py in:
/Volumes/One Touch/KITCO3_drive/scripts
Run:
python3 "/Volumes/One Touch/KITCO3_drive/scripts/05b_KITCO3_PLV_publication_figures_v1_1_0_EG.py"

Defaults read outputs/02_ecg_processing_v1_1_0, outputs/03_ecg_QC_v2_1_0,
outputs/04_dyadic_signals_v1_0_0, outputs/05_PLV_feasibility_v1_0_0 and data.
Output is NEW outputs/05b_PLV_publication_figures_v1_1_0. No old outputs deleted.
Options: --project-root, --script02-dir, --script03-dir, --script04-dir,
--script05-dir, --raw-data-root, --processing-script, --output-dir.
Requires numpy, pandas, scipy, matplotlib and neurokit2==0.2.13.
The pinned NeuroKit version is only needed for reconstructed ECG cleaning.
There is no browser or JavaScript dependency to install. All HTML is standalone.
Completion status is written last. Input and output hashes record provenance.

## Output organisation
pairs/PAIR.pdf: RR/PLV summary + partner 1 ECG + partner 2 ECG (three pages).
figures/PAIR_RR_PLV.png and PAIR_partner1/2_ECG.png: 300 dpi pages.
05b_pairs_part_01/02/03.pdf: 72/72/69 pages, preserving identity order.
05b_overview.pdf: three pages covering all 71 phases.
viewers/PAIR.html and index.html: interactive access to every scored window.
05b_figure_index.csv, 05b_RUN_STATUS.json and this README: documentation/audit.

References:
Aydore et al. 2013, doi:10.1016/j.neuroimage.2013.02.008 (PLV properties/bias).
Matsuki et al. 2023, doi:10.1038/s41598-023-30405-5 (Hilbert phase limitations).
These references do not uniquely validate the present band or window length.
