# KITCO3 Script08 v1.0.0

## Save and run

Save `08_KITCO3_cardiac_condition_analysis_v1_0_0_EG.py` in:

`/Volumes/One Touch/KITCO3_drive/scripts/`

Run this exact command:

```bash
python3 "/Volumes/One Touch/KITCO3_drive/scripts/08_KITCO3_cardiac_condition_analysis_v1_0_0_EG.py" \
  --output-dir "/Volumes/One Touch/KITCO3_drive/outputs/08_cardiac_condition_analysis_v1_0_0"
```

**All new result tables, reports, figures, input snapshots and plotting cache go
inside that output folder on the external drive.** The script never defaults to
an output folder in `/Users/elnaz`. It verifies that the external volume is mounted
before creating an output folder under `/Volumes`.

The default input is your already completed Script07 folder:

`/Users/elnaz/KITCO3_outputs/07_cardiac_specification_v1_0_0`

This is an input location only. Script08 does not modify or write results into it.
A complete copy of the nine verified Script07 output files and its completion
status is saved inside the new drive output's `source_snapshot` folder, together
with the executed Script08 source. Thus the new result package retains its input
snapshot without moving or deleting the original Script07 folder.

If you have moved Script07, add `--script07-dir "the actual new folder"` to the
command. Do not point to a partial run or a folder containing a mixture of releases.

Use the same Python environment that ran Scripts05-07. Required packages are
numpy, pandas, scipy and matplotlib, which were used earlier. **Statsmodels is
not a runtime requirement.** No archive extraction, R, NeuroKit, acquisition app,
raw ECG access, accelerometer data or video review is required to run this step.

## What Script08 does

1. Verifies all nine Script07 output hashes, release identity and frozen method.
2. Confirms the 71-candidate register, 48 eligible scored phases, 10,434 dependent
   windows and 16-session primary design, preserving the literal `NA` condition.
3. Calculates session means and emotional-minus-neutral differences. Each
   session-condition mean equally weights its observed pair-phase PLV means.
4. Fits the frozen three-parameter model across the 16 complete sessions:

   `D_PLV = intercept + beta_recipe * D_Banana + beta_order * D_Phase2 + error`

   `D` denotes emotional minus neutral. The intercept is the primary adjusted
   association. The covariates are between-condition differences in the proportion
   of Banana recipes and phase-2 observations. All sessions receive equal weight.
5. Computes HC3 heteroskedasticity-robust SEs and an approximate Student t reference
   with 13 residual df, two-sided p-value and 95% CI, exactly as specified in Script07.
6. Assesses every existing method and clock-scenario family; reports changed pair
   and session membership. No sensitivity selects the primary method by its result.
7. Separately compares common pair cohorts and identically matched scoring cores.
   For paired changes, uncertainty is calculated from the session-level change
   outcome, not by subtracting two CI endpoints.
8. Performs leave-one-session-out fits and the prespecified exploratory additional
   coverage-imbalance adjustment, if identifiable. No diagnostic excludes a session.
9. Exports six figure sets, statistical tables, methods/results documentation and
   a completion manifest. It checks the source files again before publishing success.

There are 42 eligible pair phases in the 16 primary sessions. The other six scored
pair phases remain in the descriptive/coverage data because their sessions have
only one observed condition. They are not new ECG exclusions. Windows and 4 Hz
samples are never used as independent statistical observations. Participant
nesting is checked across all 71 candidate phases before treating sessions as the
independent units. Independence between sessions is still a study assumption.

## Scope of the conclusion

The outcome remains mean local 200 s PLV of ECG-derived RR intervals (IBIs) in
the 0.04-0.15 Hz band, using 40 s context on either side. This is the locked
operational outcome, not a newly established optimal window or a whole-cooking
PLV. Unequal usable durations are retained; their coverage is reported.

The analysis estimates an adjusted observed-data condition association under an
additive recipe/order model. It does not establish causal effects, isolated partner
change effects, representativeness of missing periods, or absolute interpersonal
coupling above chance. Original randomization/exchangeability has not been verified.
The HC3/t inference is approximate, especially with 16 sessions; it is not an
exact randomization test. An interval containing zero is not proof of no association.

**Physical sensor clock offset and drift remain unverified.** All results are
conditional on treating exported timestamps as a common time axis. Existing timing
perturbations are illustrative scenarios, not measured uncertainty bounds or repairs.
No movement files are read and no physiological alignment is optimized.

Earlier exploratory outcomes were inspected. This follows the Script07 freeze;
it is not preregistration or a fully outcome-blind analysis. One primary test is
specified. Secondary intervals are pointwise robustness descriptions without
multiplicity adjustment and should not be promoted to independent discoveries.
No condition-label shuffling or synthetic-noise cutoff is used.

## Outputs

Start with these in the drive output folder:

- `08_START_HERE.html`: links to the report, primary result and figures.
- `08_RESULTS_AND_METHODS.md`: full primary result, decisions and limitations;
  readable as plain text in TextEdit or any Markdown viewer.
- `08_primary_result.json`: primary estimate, interval, p-value and key counts.
- `08_primary_coefficients.csv`: primary intercept and nuisance coefficients.
  Only the intercept is the primary hypothesis; nuisance p-values are not extra
  confirmatory tests.
- `08_primary_sessions.csv`: the 16 raw session contrasts and model covariates.
- `08_primary_covariance_HC3.csv`: complete coefficient covariance matrix.
- `08_native_family_sensitivity.csv`: 18 alternative families on their own support.
- `08_common_pair_cohort_sensitivity.csv`: same pair identities for each comparison;
  time periods can still differ.
- `08_matched_core_sensitivity.csv`: 15 same-duration families on matched cores,
  including paired changes in the adjusted condition association.
- `08_sensitivity_membership.csv` and `08_sensitivity_sessions.csv`: auditable
  family-specific membership and underlying session contrasts.
- `08_leave_one_session_out.csv` and `08_session_diagnostics.csv`: influence,
  leverage, residuals and conventional Cook distances, without automatic exclusions.
- `08_coverage_adjusted_coefficients.csv`, `08_coverage_adjustment_status.json`,
  `08_pair_phase_coverage.csv`, `08_availability_by_design.csv`: availability and
  the exploratory coverage diagnostic. Coverage adjustment cannot repair
  missing-not-at-random selection and might condition on a consequence of condition.
- `source_snapshot/`: verified Script07 inputs, its status, and executed Script08.
- `08_RUN_STATUS.json`: written **last**, only after successful completion.

`figures/` contains six sets, each as PNG at 300 dpi, vector PDF and editable SVG:

1. `08_primary_condition_comparison`: paired observed session means, raw session
   contrasts, and the recipe/order-adjusted primary association with 95% CI.
2. `08_method_sensitivity`: family-specific adjusted associations with CIs and
   session counts, labelled as secondary sensitivities.
3. `08_clock_scenarios`: illustrative timing perturbations with the same model.
4. `08_matched_core_changes`: paired changes in the condition association.
5. `08_model_diagnostics`: residual/fitted, Q-Q, leverage and leave-one-session-out
   diagnostics. No automatic normality test changes the model.
6. `08_availability_and_coverage`: observed counts and descriptive coverage/PLV
   relationship. The pair-phase scatter is not an independent-observation test.

The source CSV tables contain the plot data. Figures distinguish observed raw
means/differences from model-adjusted estimates. No significance stars or arbitrary
synchrony-strength categories are added. `runtime_cache/` holds only the plotting
cache and is not included in the scientific output hash manifest.

## Validation before release

The script was run end-to-end using your actual uploaded Script07 files. All
nine source hashes and all exported scientific output hashes passed.

**135 fitted models were independently compared with statsmodels 0.15.0 using
`OLS(...).fit(cov_type="HC3", use_t=True)`.** Coefficients, full covariance matrices,
SEs, two-sided p-values and confidence intervals agreed, with maximum absolute
discrepancy below `8e-16` in the local validation environment. This covers the primary,
coverage, all native variants, common-pair comparisons, matched-core changes, and
leave-one-session-out fits. Statsmodels was used for validation only; the delivered
script uses numpy/scipy with explicit matrix operations.

Every fit also verifies its coefficients and HC3 covariance using a separate QR
calculation. Startup self-tests verify the HC3 delete-one coefficient-change identity,
row-order invariance, a known t quantile, and rank-deficiency rejection. Further tests
confirmed that changed inputs are rejected and existing output folders are not
modified. All six vector PDFs were rendered and visually checked for readability.

Local validation used Python 3.12.14, numpy 2.3.5, pandas 2.2.3, scipy 1.17.0,
matplotlib 3.10.8. Your Mac versions differ and are recorded automatically. Expect
numerical agreement to floating-point precision; PDF timestamps/font caches need
not be byte-identical across systems. Library agreement verifies implementation,
not the scientific truth of the clock, missingness, or model assumptions.

## Completion and recovery

Expected successful status:

`CONDITIONAL_CARDIAC_CONDITION_ANALYSIS_COMPLETE`

The output folder must not exist before a new run. If you intentionally need a
second run, preserve the first and choose a new folder under your drive's
`outputs` directory using `--output-dir`. Do not delete inputs or rerun ECG QC.

If an external-drive I/O error occurs, preserve the incomplete folder and resolve
access to the drive before starting a new output folder. An unfinished file or
report without `08_RUN_STATUS.json` is not a completed run. No home-directory
fallback or automatic overwrite is performed.

Optional numerical self-test only, with no output files:

```bash
python3 "/Volumes/One Touch/KITCO3_drive/scripts/08_KITCO3_cardiac_condition_analysis_v1_0_0_EG.py" --self-test-only
```

After running, send `08_RUN_STATUS.json`, `08_primary_result.json`,
`08_native_family_sensitivity.csv`, and `08_RESULTS_AND_METHODS.md` for review.

## Version and references

Release 1.0.0 implements the exact Script07 v1.0.0 session-contrast plan. No post-result
change to the primary model, inclusion rule, frequency band or scoring window was made.

- https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLSResults.HC3_se.html
- https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.RegressionResults.get_robustcov_results.html
- https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.t.html
- https://doi.org/10.1016/j.neuroimage.2013.02.008
- https://doi.org/10.1038/s41598-023-30405-5
