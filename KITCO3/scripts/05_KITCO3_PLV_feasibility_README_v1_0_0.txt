# KITCO3 Script 5 v1.0.0

## Purpose and limits

The next step after Script04 is to evaluate an operational phase-locking measure
on the available paired IBI signals. This script calculates descriptive PLV,
checks phase behavior, quantifies window support, and evaluates numerical and
timing sensitivity. It does not test emotional versus neutral conditions,
establish a coupling mechanism, calibrate clocks, or freeze final inclusion.

This release preserves the working method from the earlier v0.5/v0.6 audits.
It adapts it to the current, verified Script04 exports instead of repeating ECG
detection, waveform decisions or dyadic alignment. It is not a new preregistered
analysis: earlier exploratory outcomes have been viewed. Condition/story/recipe
columns are not loaded into estimation or used to select method settings here.

## Save and run on Emma's Mac

Save the single Python file here. It does not need a separate support package:

```text
/Volumes/One Touch/KITCO3_drive/scripts/05_KITCO3_PLV_feasibility_v1_0_0_EG.py
```

Use the same Python environment as Scripts 2-4. Required packages are numpy,
pandas, scipy and matplotlib. It uses no NeuroKit functions because R-peak
detection and reviewed IBI construction are already complete.

Keep the Mac powered and run:

```bash
python3 "/Volumes/One Touch/KITCO3_drive/scripts/05_KITCO3_PLV_feasibility_v1_0_0_EG.py" \
  --project-root "/Volumes/One Touch/KITCO3_drive" \
  --script04-dir "/Volumes/One Touch/KITCO3_drive/outputs/04_dyadic_signals_v1_0_0" \
  --output-dir "/Volumes/One Touch/KITCO3_drive/outputs/05_PLV_feasibility_v1_0_0" \
  --timing-basis exported_time_unverified
```

The output folder must not already exist. No overwrite or resume is provided;
preserve any failed run and diagnose its error before choosing another path.
The timing argument records an unresolved assumption; it does not verify it.
The original Script02, Script03 and Script04 files are not modified.

Optional self-tests, without loading data or creating an output folder:

```bash
python3 "/Volumes/One Touch/KITCO3_drive/scripts/05_KITCO3_PLV_feasibility_v1_0_0_EG.py" --self-test
```

Successful data-run status:

```text
CONDITIONAL_PLV_FEASIBILITY_COMPLETE
```

## Working specification

| Component | Decision and rationale |
|---|---|
| Signal | Reviewed midpoint IBI, in seconds, resampled at 4 Hz by Script04. Reuse accepted QC rather than redetecting peaks. |
| Phase | Analytic phase after 0.04-0.15 Hz bandpass filtering. This is band-limited IBI phase, not phase of individual heartbeats. |
| Physiological interpretation | LF is a conventional HRV band, not a uniquely validated interpersonal oscillator or selective sympathetic signal. |
| Scoring core | 200 s, inherited as a working choice from previous numerical checks. It contains 8 cycles at 0.04 Hz; that arithmetic is not proof of optimality. |
| Real context | 40 s each side, making a full 280 s supported context. Context/padding sensitivity is evaluated, not assumed absent. |
| Placement | Starts at cooking 40 + k seconds, k integer, throughout usable cooking. One-second steps reduce arbitrary coarse-grid origin effects. |
| Filtering | Linear detrend, divide by population SD, Butterworth SOS order argument 3, forward-backward, default odd padding; then FFT Hilbert at context length. Bandpass one-way order is 6; forward-backward squares magnitude response. |
| Local PLV | Magnitude of the mean complex unit phase difference within the scoring core. Anti-phase and any constant phase difference can yield high PLV. |
| Pair-phase summary | Arithmetic mean of equal-length local PLVs. This differs from pooled whole-interaction PLV. |
| Coverage | Union of scoring intervals, not number of overlapping windows multiplied by 200 s. |
| Gaps and boundaries | Every context stays within one paired trusted segment. No extrapolation, gap filling, stitching, or extension outside cooking. |
| Unequal durations | Report coverage and support; do not delete usable signal merely to force equal total condition durations. Final missingness/estimand handling remains part of the statistical plan. |
| Exclusions | No new spectral-power/envelope cutoff. Truly degenerate contexts cannot define phase and are explicitly logged. Other diagnostics do not automatically exclude observations. |

## Sensitivity and numerical validation

The script evaluates cores of 180, 250 and 300 s; 80 s context; order arguments
2 and 4; and narrower bands 0.04-0.10 and 0.06-0.15 Hz. Narrower-band outcomes
are different definitions, not substitutes selected by a preferred result.
Same-duration variant comparisons use identical supported core starts within
each pair phase. Changed-duration outputs are not presented as matched estimates
of the same temporal estimand.

Timing scenarios are fixed, illustrative shifts of partner 2 by +/-0.25, 1, 2
and 5 s, and drift of +/-100 ppm relative to shared cooking onset. They are not
measured error ranges, confidence bounds or corrections. No scenario is fitted
to maximize PLV. Sensitivity results cannot prove physical synchronization.
These scenarios PCHIP-interpolate the existing 4 Hz partner-2 series, rather
than reprocessing raw ECG timestamps or recovering sensor clock information.

The default 500 synthetic independent LF-noise pairs use fixed seed 20261016
and the same source realizations across durations. Cores of 100, 180, 200, 300
and 500 s illustrate finite-duration PLV behavior. Margins of 20, 40, 80 and
120 s are compared with long-context estimates for the same central samples.
Fixed-lag and known frequency-mismatch controls check numerical recovery;
two-component mixtures illustrate limits of a latent-oscillator interpretation.
These simulated models are not human physiological ground truth. No simulated
noise percentile is used as an observed-data significance threshold.

Participant-window diagnostics report LF power fraction and spectral entropy,
dominant LF bin, weak analytic envelope, and negative instantaneous-frequency
fraction. The envelope diagnostic uses 10% of context RMS as a descriptive
reference, not an exclusion threshold. Analytic phase need not be monotone.

## Output files

| File | Meaning |
|---|---|
| `05_working_window_diagnostics.csv` | Working-spec local PLV, source paired segment, location and participant phase/spectrum diagnostics. Overlapping windows are not independent cases. |
| `05_pair_phase_method_summaries.csv` | All 71 pair phases for each specification; mean local PLV, window count, union coverage and evaluability. No-support rows remain. |
| `05_matched_core_sensitivity.csv` | Within-pair changes for same-duration variants on identical core starts. No hypothesis tests. |
| `05_synthetic_duration_bias.csv` | Independent simulated PLV across durations and edge differences. |
| `05_synthetic_context_errors.csv` | PLV/phase discrepancy from long-context same-core reference. |
| `05_synthetic_known_phase.csv` | Fixed-lag, frequency-mismatch and two-component controls. |
| `05_METHODS.md` | Embedded detailed method, interpretation and reference documentation. |
| `figures/` | Support/bias overview and a deterministic first eligible phase-estimation example; PNG 300 dpi, PDF and SVG. |
| `05_RUN_STATUS.json` | Written last: exact configuration, software versions, script/input/output hashes, runtime and remaining decisions. |

The archive's 4 Hz points and overlapping windows must never be treated as the
inferential sample size. No p values, pair-level surrogate tests or condition
effects are calculated. Finite-duration bias and individual spectral content
need consideration in the eventual reference and condition-comparison design.
Dependence from sessions, participants and changing partners, recipe, order and
missing coverage also remains to be accounted for.

## Reproducibility and references

All declared upstream hashes are checked; input hashes are rechecked at the
end. Input paths are checked to stay inside the specified source directory.
Scientific CSVs use stable row ordering and explicit precision. Cross-platform
gzip byte identity is not assumed. Unexpected errors are not silently treated
as excluded windows. Completion is published only after all stages succeed.

Primary references:

- Matsuki et al. (2023): https://doi.org/10.1038/s41598-023-30405-5
- Aydore et al. (2013): https://doi.org/10.1016/j.neuroimage.2013.02.008
- Quigley et al. (2024): https://doi.org/10.1111/psyp.14604
- SciPy: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html
- SciPy: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sosfiltfilt.html
- SciPy: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.hilbert.html

Version history: v1.0.0 is the first Script05 release against Script04 v1.0.0
paired exports. It preserves the operational definition from the prior v0.5/
v0.6 method audits, while explicitly separating descriptive feasibility from
confirmatory analysis and unresolved physical-clock calibration.

## Release verification on the supplied Script04 exports

The full data run used 500 synthetic replicates and all 71 pair phases. It
completed with 48 pair phases with some working-spec support, 10,434 scored
windows, 5.662222 hours of union scoring-core coverage, and zero degenerate
working contexts. These are conditional support counts, not final inclusion.
All 10,434 window locations matched the earlier v0.6 dense audit. The maximum
PLV discrepancy was 0.000000006, consistent with using rounded Script04 IBI
exports rather than re-interpolating unrounded source IBI in the earlier audit.
Known fixed-lag controls recovered PLV >=0.999856; maximum frequency-mismatch
control error was 0.004774. Built-in tests check anti-phase PLV, deduplicated
coverage, context indexing, rejection of extrapolation and degeneracy.

Testing environment: Python 3.12.14, NumPy 2.3.5, pandas 2.2.3, SciPy 1.17.0,
matplotlib 3.10.8. The script records the actual environment on each new run.
It does not use the SciPy median_filter function that caused the earlier
Script03 short-segment validator issue. This test does not establish that every
library/OS combination is identical, physical clock accuracy, or physiological
ground truth. Figures were inspected for readability and wrap representation.
