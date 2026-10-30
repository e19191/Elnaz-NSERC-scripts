# Run Script07 on your Mac

Save `07_KITCO3_cardiac_specification_freeze_v1_0_0_EG.py` in:

`/Volumes/One Touch/KITCO3_drive/scripts/`

Run:

```bash
python3 "/Volumes/One Touch/KITCO3_drive/scripts/07_KITCO3_cardiac_specification_freeze_v1_0_0_EG.py"
```

The script defaults to your existing inputs:

- Script04: `/Volumes/One Touch/KITCO3_drive/outputs/04_dyadic_signals_v1_0_0`
- Script05: `/Volumes/One Touch/KITCO3_drive/outputs/05_PLV_feasibility_v1_0_0`
- Script06: `/Users/elnaz/KITCO3_outputs/06_timing_method_review_v1_0_0`

New output, on your internal drive:

`/Users/elnaz/KITCO3_outputs/07_cardiac_specification_v1_0_0`

It is standalone. No archive extraction, original Script05 Python file, NeuroKit,
R or statsmodels installation is required for this specification stage. It uses
numpy and pandas already present in the environment that ran Script06.

If you intentionally need another run, preserve the previous output and choose
a new folder with `--output-dir`. Do not rerun Scripts02-06 for this step.

## Validation before release

Tested on the actual uploaded Script04/05/06 output files, not synthetic study
results. All 62 Script04, 13 Script05, and 18 Script06 listed output hashes passed.
71 candidates, 48 scored pair phases, and 10,434 windows were reconciled.
The planned session design has 16 rows, rank 3, residual df 13, and maximum
leverage 0.225807. It uses 42 of the scored pair phases for within-session
comparisons and retains the other six descriptively. No condition contrast or
coefficient was computed in validation.

Additional checks passed for literal `NA` condition preservation, union coverage,
independence of design from outcome values, rejection of a rank-deficient design,
rejection of a changed source file, and refusal to overwrite outputs. Local test
environment: Python 3.12.14, numpy 2.3.5, pandas 2.2.3. Your Mac environment differs;
the script records its actual runtime versions. Byte-identical source verification
uses your existing sealed files and does not require matching Python versions.

A plan lock is a documented operational choice. It does not certify a best
physiological estimator or resolve unmeasured clock offsets. All eventual
interpretations must retain the conditional timing limitation.

---

# KITCO3 Script07: conditional cardiac specification, v1.0.0

## What has been frozen

The analysis specification and deterministic observed-data inclusion rules are
locked before computing an emotional-neutral contrast in this release. This
freezes a conditional operational analysis, not physical sensor synchronization.
Earlier exploratory outputs were viewed. Do not describe this as preregistered,
fully outcome-blind, an optimized estimator, or independently human-adjudicated QC.

No movement files are read. No ECG detection, cleaning, peak edits, waveform QC,
interpolation or PLV calculation is repeated. The sealed Script04/05/06 outputs
are hash-verified first. Script07 validates window support, recomputes arithmetic
means and deduplicated core coverage from existing window PLVs, checks design
coding and nesting, and packages the specification. It fits no outcome model.

## Timing decision

Use the existing exported timestamp axis as an explicit unverified common-time
assumption. No empirical offset/drift correction is justified by this release.
The lack of calibration remains a substantive limitation even after a plan lock.
A hash match or common acquisition folder does not establish physical synchrony.
A later timing correction requires a new version, not silently replacing this lock.

## Cardiac outcome and timescale

Signal: ECG-derived RR interval (IBI), seconds internally and milliseconds for
display, existing 4 Hz PCHIP grid within trusted runs. Band: 0.04-0.15 Hz.
Each score uses 200 s plus 40 s of real context on each side. Starts advance by
1 s across every supported cooking section, not just the beginning. The retained
filter and Hilbert operations are specified exactly in the JSON.

The pair-phase outcome is mean local-window PLV. It asks about local consistency
of phase relationships during retained cooking periods. It is not whole-cooking
PLV, simultaneous heartbeats, percentage synchronization, direction or causation.
Eight cycles of the lower band edge occur in a 200 s core; that rationale and the
existing diagnostics support retaining a tested candidate, not claiming optimality.
A broad operational band can contain multiple oscillations and weak-envelope
periods. The existing diagnostics are reported without new result-driven cutoffs.

## Inclusion and unequal durations

Include a pair phase if at least one existing valid full 280 s context supports a
finite core PLV. Apply no additional coverage-percentage cutoff, imputation,
first-three-minute rule, or movement-based exclusion. This explicitly targets
available usable periods, not unbiased recovery of an entire interaction.
Different cooking lengths are retained. Each pair phase contributes one mean,
not one case per second or overlapping window. Longer phases do not receive
additional weight in the subsequent session-condition mean. Coverage, temporal
support and missingness remain reported limitations, not cured by equal windows.

Verified counts: 71 candidates,
48 eligible cardiac pair phases,
10434 dependent scored windows. The full register keeps all
candidates, including those without a usable core.

## Primary statistical design

First average pair-phase outcomes within session and story condition. Then form
one emotional-minus-neutral difference per session with both conditions observed.
Regress those session differences on an intercept, the between-condition difference
in Banana proportion, and the between-condition difference in phase-2 proportion.
Each session has equal weight. The intercept is the adjusted condition contrast,
under an additive recipe/order adjustment model. This averages over observed
partner configurations; it does not identify a separate causal partner-change effect.

There are 16 eligible sessions, drawing
on 42 eligible pair phases. The other
6 scored pair phases
remain descriptive because their sessions lack an observed within-session
condition contrast. They have not failed ECG QC.

Within-session aggregation keeps shared participants and partner changes inside
one inferential unit. PID nesting across sessions is checked. Independent sessions
are assumed; if recruitment or household links violate that, revise the plan.
Recipe and order enter separately; full rank, leverage and leave-one-session-out
rank are checked without calculating any outcome contrast.

Planned inference: OLS, HC3 covariance, t reference with S-3 residual degrees of
freedom, two-sided 95% CI and alpha .05. This is small-sample approximate inference,
not a valid-label-shuffling claim or exact randomized test. The target is an
observed-data adjusted association. The original assignment unit/randomization
has not been independently verified. No causal or missing-data-unbiased claim.

## Reference and sensitivity decisions

The primary null is adjusted emotional-minus-neutral difference = 0. It is not
PLV = 0. Positive finite-window PLV can occur for unrelated autocorrelated signals.
The previous synthetic-noise distributions must not be used as human reference
cutoffs. This plan supports comparison of measured cardiac phase consistency,
not a test that a dyad is coupled above chance. That separate question would need
a justified cardiac reference design and a new specification.

Report all existing duration, context, order, band and clock-perturbation
sensitivities, plus leave-one-session-out coefficient stability. Apply the same
aggregation and mean model when identifiable. Report membership changes; duration
variants do not necessarily cover the same periods. Same-duration matched-core
tables are retained for estimator comparisons. Clock scenarios are illustrative,
not measured clock-error bounds. Also assess continuous coverage imbalance as an
exploratory extra covariate when estimable. No sensitivity determines the primary
method or inclusion by giving a stronger condition result. One primary hypothesis;
secondary estimates describe robustness, not independent confirmatory discoveries.

## Outputs and next step

- 07_ANALYSIS_SPECIFICATION.json: machine-readable decisions and next-stage contract.
- 07_pair_phase_register_71.csv: every candidate and explicit inclusion reason.
- 07_conditional_cardiac_dataset.csv: one existing PLV mean per eligible pair phase.
- 07_session_availability.csv: complete and one-condition-only observed sessions.
- 07_primary_session_design.csv: model covariates and leverage, no outcome contrast.
- 07_availability_by_design.csv: missingness/coverage counts, no condition effect.
- 07_all_specification_summaries.csv and 07_matched_core_sensitivity.csv: sealed
  earlier method outputs packaged for the next stage.
- 07_RUN_STATUS.json: written last after success, with output and input-status hashes.

Next stage computes the locked condition association and prespecified sensitivities.
No condition coefficient, CI or p-value is calculated here. A successful technical
lock does not establish an unconditional final scientific result.

Dependencies: Python >=3.10, numpy, pandas. No extra package installation is normally
needed in the environment that ran Script06. Default output is on the internal
Mac drive to avoid the earlier external-drive write failure. Inputs remain on the
external drive. Existing output folders are never overwritten; failed folders
must be preserved and a new output path used after the underlying issue is fixed.

## References

Aydore et al. 2013: https://doi.org/10.1016/j.neuroimage.2013.02.008
Matsuki et al. 2023: https://doi.org/10.1038/s41598-023-30405-5
HC3 definition: https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLSResults.HC3_se.html
Robust inference options: https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.RegressionResults.get_robustcov_results.html
These sources inform interpretation and inference; they do not validate this
study's specific clock assumptions, physiological band or 200 s window.
