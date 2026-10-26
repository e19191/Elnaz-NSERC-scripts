#!/usr/bin/env python3
"""KITCO3 Script07 v1.0.0: conditional cardiac specification and cohort lock.

Requires Python >=3.10, numpy and pandas. No movement input, ECG redetection,
PLV recalculation, condition-effect estimation, downloads, or overwrite.

Run on Emma's Mac:
python3 "/Volumes/One Touch/KITCO3_drive/scripts/07_KITCO3_cardiac_specification_freeze_v1_0_0_EG.py"

Default output: ~/KITCO3_outputs/07_cardiac_specification_v1_0_0
A completed run freezes a CONDITIONAL analysis specification, not physical
synchronization. Earlier exploratory outputs have been inspected. This is not
preregistration. Read the generated 07_DECISIONS_AND_METHODS.md before reporting.

Primary target: adjusted emotional-minus-neutral difference in SESSION means of
pair-phase mean local RR PLV. Each session with both observed conditions gives
one contrast; participants and changing partners remain inside that session.
The script builds the design but does not calculate condition contrasts.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import numpy as np
import pandas as pd

VERSION = '1.0.0'
FAMILY = 'working_200s_40s'
DEFAULT_ROOT = Path('/Volumes/One Touch/KITCO3_drive')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def read_csv(path):
    # NA is the actual Neutral/Apple condition code, not missing data.
    return pd.read_csv(path, keep_default_na=False)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def atomic_text(path, content):
    tmp = path.with_name(path.name + '.writing')
    with tmp.open('x', encoding='utf-8', newline='') as f:
        f.write(content)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def json_out(path, obj):
    atomic_text(path, json.dumps(obj, indent=2, allow_nan=False) + '\n')


def csv_out(path, df):
    atomic_text(path, df.to_csv(index=False, float_format='%.12g', na_rep=''))


def verify_release(root, status_name):
    root = Path(root).resolve()
    status = read_json(root / status_name)
    manifest = status.get('output_sha256')
    require(isinstance(manifest, dict) and len(manifest) > 0,
            f'Missing output manifest in {root / status_name}')
    for name, expected in manifest.items():
        p = (root / name).resolve()
        require(root in p.parents, f'Unsafe manifest path: {name}')
        require(p.is_file() and digest(p) == expected,
                f'Source hash mismatch or missing file: {p}')
    return status, len(manifest)


def union_seconds(intervals):
    if not intervals:
        return 0.0
    ordered = sorted(intervals)
    left, right = ordered[0]
    total = 0.0
    for a, b in ordered[1:]:
        if a > right:
            total += right - left
            left, right = a, b
        else:
            right = max(right, b)
    return total + right - left


def session_design(cohort):
    """Only membership and covariates are used. No outcome contrast calculated."""
    scored = cohort.loc[cohort.conditional_eligible].copy()
    rows = []
    for sid, g in scored.groupby('Session_ID', sort=True):
        e = g.loc[g.Story.eq('Emotional')]
        n = g.loc[g.Story.eq('Neutral')]
        both = len(e) > 0 and len(n) > 0
        rows.append({
            'Session_ID': sid,
            'emotional_pair_phases': len(e),
            'neutral_pair_phases': len(n),
            'primary_contrast_eligible': both,
            'reason': 'BOTH_CONDITIONS_OBSERVED' if both else 'ONE_CONDITION_OBSERVED_ONLY',
            'delta_banana_proportion': float(e.Recipe.eq('Banana').mean() - n.Recipe.eq('Banana').mean()) if both else '',
            'delta_phase2_proportion': float(e.Phase.eq(2).mean() - n.Phase.eq(2).mean()) if both else '',
        })
    sd = pd.DataFrame(rows)
    eligible = sd.loc[sd.primary_contrast_eligible].copy()
    require(len(eligible) > 3, 'Too few complete sessions for the specified three-parameter model.')
    x = np.column_stack([np.ones(len(eligible)),
                         eligible.delta_banana_proportion.to_numpy(float),
                         eligible.delta_phase2_proportion.to_numpy(float)])
    rank = int(np.linalg.matrix_rank(x))
    require(rank == 3, 'Recipe/order-adjusted session contrast is not identifiable. No freeze published.')
    hat = np.einsum('ij,ji->i', x, np.linalg.solve(x.T @ x, x.T))
    require(np.all(hat < 1 - 1e-10), 'Unit-leverage session prevents planned HC3 inference.')
    loo = []
    for i, sid in enumerate(eligible.Session_ID):
        loo.append({'Session_ID_omitted': sid,
                    'design_rank': int(np.linalg.matrix_rank(np.delete(x, i, axis=0)))})
    require(all(z['design_rank'] == 3 for z in loo),
            'A leave-one-session-out design loses rank; the specified sensitivity is not supported.')
    eligible['design_leverage'] = hat
    return sd, eligible, {'rows': len(eligible), 'columns': 3, 'rank': rank,
                          'residual_df': len(eligible) - 3,
                          'max_leverage': float(hat.max()),
                          'condition_number': float(np.linalg.cond(x)),
                          'leave_one_session_out': loo}


def freeze(args):
    d4, d5, d6 = args.script04_dir, args.script05_dir, args.script06_dir
    out = args.output_dir
    require(not out.exists(), f'Output folder exists: {out}. Preserve it and choose a NEW --output-dir.')
    s4, n4 = verify_release(d4, '04_RUN_STATUS.json')
    s5, n5 = verify_release(d5, '05_RUN_STATUS.json')
    s6, n6 = verify_release(d6, '06_RUN_STATUS.json')
    require(s4.get('status') == 'CONDITIONAL_SIGNAL_EXPORT_COMPLETE', 'Unexpected Script04 status.')
    require(s5.get('status') == 'CONDITIONAL_PLV_FEASIBILITY_COMPLETE', 'Unexpected Script05 status.')
    require(s6.get('status') == 'TIMING_METHOD_AUDIT_COMPLETE_WITH_OPEN_DECISIONS', 'Unexpected Script06 status.')
    require(s4.get('grid_hz') == 4 and s4.get('timing_basis') == 'exported_time_unverified', 'Unsupported grid/timing basis.')
    require(s4.get('physical_clock_verified') is False and s5.get('physical_clock_verified') is False,
            'Different timing status requires a new analysis specification.')
    require(s5.get('condition_effect_computed') is False and s6.get('condition_effect_computed') is False,
            'Condition-effect results already present: cannot claim a before-comparison freeze.')
    require(s5.get('source_status_sha256') == digest(d4 / '04_RUN_STATUS.json'), 'Script04/05 lineage mismatch.')
    recorded_inputs = s6.get('input_sha256', {})
    for name, directory in [('04_RUN_STATUS.json', d4), ('05_RUN_STATUS.json', d5)]:
        prior = [h for p, h in recorded_inputs.items() if Path(p).name == name]
        require(len(prior) == 1 and prior[0] == digest(directory / name), 'Script06 lineage mismatch: ' + name)
    expected_config = {'grid_hz': 4, 'window_step_s': 1, 'window_anchor_s': 40,
                       'working_core_s': 200, 'working_context_each_side_s': 40,
                       'working_band_hz': [0.04, 0.15], 'filter_order_argument': 3}
    require(all(s5['configuration'].get(k) == v for k, v in expected_config.items()), 'Script05 configuration differs.')
    reg = read_csv(d4 / '04_pair_signal_register_71.csv').sort_values('pair_id').reset_index(drop=True)
    summaries = read_csv(d5 / '05_pair_phase_method_summaries.csv')
    working = summaries.loc[summaries.family.eq(FAMILY)].copy()
    audit = read_csv(d6 / '06_pair_phase_review_71.csv')
    windows = read_csv(d5 / '05_working_window_diagnostics.csv')
    segments = read_csv(d4 / '04_paired_interpolation_segments.csv')
    require(len(reg) == 71 and reg.pair_id.is_unique, 'Expected 71 unique candidate pair phases.')
    require(len(working) == 71 and working.pair_id.is_unique and set(working.pair_id) == set(reg.pair_id), 'Working family membership mismatch.')
    require(len(audit) == 71 and audit.pair_id.is_unique and set(audit.pair_id) == set(reg.pair_id), 'Script06 membership mismatch.')
    require(set(windows.pair_id) <= set(reg.pair_id), 'Unknown pair in window table.')
    require(not windows.duplicated(['pair_id', 'paired_segment_id', 'core_start_s']).any(), 'Duplicate working windows.')
    require(not segments.duplicated(['pair_id', 'paired_segment_id']).any(), 'Duplicate paired segment IDs.')
    mapping = {'NA': ('Neutral', 'Apple'), 'NB': ('Neutral', 'Banana'),
               'EA': ('Emotional', 'Apple'), 'EB': ('Emotional', 'Banana')}
    for r in reg.itertuples():
        require(r.Condition in mapping and (r.Story, r.Recipe) == mapping[r.Condition], 'Condition coding mismatch: ' + r.pair_id)
        require(r.Phase in [1, 2] and r.PID_1 != r.PID_2, 'Invalid phase or partner IDs.')
    # Physical participant IDs are globally unique in this study. Do not silently
    # reinterpret reused IDs as different people merely because Session_ID differs.
    people = pd.concat([reg[['Session_ID', 'PID_1']].rename(columns={'PID_1': 'PID'}),
                        reg[['Session_ID', 'PID_2']].rename(columns={'PID_2': 'PID'})])
    require(people.groupby('PID').Session_ID.nunique().max() == 1,
            'A PID appears in multiple sessions. Session-level independence requires reassessment.')
    working = working.set_index('pair_id')
    audit = audit.set_index('pair_id')
    support = segments.set_index(['pair_id', 'paired_segment_id'])
    rows = []
    for r in reg.itertuples():
        g = windows.loc[windows.pair_id.eq(r.pair_id)].copy()
        plv = pd.to_numeric(g.PLV, errors='coerce').to_numpy(float)
        # No diagnostic thresholds or newly selected windows are introduced.
        require(np.all(np.isfinite(plv) & (plv >= 0) & (plv <= 1)), 'Invalid/degenerate PLV requires reassessment: ' + r.pair_id)
        require(g.status.eq('CONDITIONAL_DIAGNOSTIC_SCORE').all(), 'Unexpected window status.')
        intervals = []
        for w in g.itertuples():
            require(w.core_duration_s == 200 and w.context_each_side_s == 40, 'Window parameter mismatch.')
            require((r.pair_id, w.paired_segment_id) in support.index, 'Missing paired support segment.')
            t = float(w.core_start_s)
            require(abs(t - round(t)) < 1e-8 and t >= 40, 'Unexpected core anchor.')
            q = support.loc[(r.pair_id, w.paired_segment_id)]
            # Context is sampled on [t-40, t+240) at 4 Hz, last sample t+239.75.
            require(t - 40 >= q.support_start_s - 1e-7 and t + 240 <= q.support_stop_s + 1e-7,
                    'Context outside continuous trusted support.')
            require(t - 40 >= q.first_grid_s - 1e-7 and t + 239.75 <= q.last_grid_s + 1e-7,
                    'Context outside paired grid support.')
            intervals.append((t, t + 200))
        n = len(g)
        mean = float(plv.mean()) if n else np.nan
        covered = union_seconds(intervals)
        ref = working.loc[r.pair_id]
        a = audit.loc[r.pair_id]
        require(n == int(ref.windows) == int(a.windows), 'Window count mismatch: ' + r.pair_id)
        require(abs(covered - float(ref.core_union_s)) < 1e-6, 'Core union mismatch: ' + r.pair_id)
        if n:
            require(abs(mean - float(ref.mean_local_PLV)) < 2e-9 and abs(mean - float(a.mean_local_PLV)) < 2e-9,
                    'Pair mean mismatch: ' + r.pair_id)
        require(r.shared_cooking_s > 0 and covered <= r.shared_cooking_s + 1e-6, 'Invalid coverage.')
        row = r._asdict()
        row.pop('Index', None)
        row.pop('final_synchrony_inclusion', None)
        row.update(conditional_eligible=bool(n),
                   inclusion_reason='AT_LEAST_ONE_COMPLETE_VALIDATED_CONTEXT' if n else
                    ('NO_PAIRED_RR_GRID' if int(r.common_grid_points) == 0 else 'NO_COMPLETE_280S_CONTEXT'),
                   scored_windows=n, mean_local_PLV=mean, scored_core_union_s=covered,
                   scored_core_coverage_fraction=covered / r.shared_cooking_s)
        rows.append(row)
    cohort = pd.DataFrame(rows)
    sessions, design, diagnostics = session_design(cohort)
    primary_sessions = set(design.Session_ID)
    cohort['primary_condition_comparison_member'] = cohort.conditional_eligible & cohort.Session_ID.isin(primary_sessions)
    cohort['timing_assumption'] = 'EXPORTED_TIME_AS_COMMON_AXIS; PHYSICAL_OFFSET_DRIFT_UNVERIFIED'
    counts = {'candidate_pair_phases': len(cohort),
              'conditional_eligible_pair_phases': int(cohort.conditional_eligible.sum()),
              'scored_windows': int(cohort.scored_windows.sum()),
              'sessions_with_scores': len(sessions),
              'primary_sessions_with_both_conditions': len(design),
              'primary_comparison_pair_phases': int(cohort.primary_condition_comparison_member.sum()),
              'scored_pair_phases_without_within_session_contrast': int((cohort.conditional_eligible & ~cohort.primary_condition_comparison_member).sum())}
    spec = make_spec(counts, diagnostics)
    # No output directory is created until all source, arithmetic and design checks pass.
    out.mkdir(parents=True, exist_ok=False)
    csv_out(out / '07_pair_phase_register_71.csv', cohort)
    csv_out(out / '07_conditional_cardiac_dataset.csv', cohort.loc[cohort.conditional_eligible])
    csv_out(out / '07_session_availability.csv', sessions)
    csv_out(out / '07_primary_session_design.csv', design)
    csv_out(out / '07_all_specification_summaries.csv', summaries)
    csv_out(out / '07_matched_core_sensitivity.csv', read_csv(d5 / '05_matched_core_sensitivity.csv'))
    coverage = cohort.groupby(['Condition', 'Story', 'Recipe', 'Phase'], dropna=False).agg(
        candidates=('pair_id', 'size'), eligible=('conditional_eligible', 'sum'),
        primary_comparison=('primary_condition_comparison_member', 'sum'),
        scored_core_union_s=('scored_core_union_s', 'sum')).reset_index()
    csv_out(out / '07_availability_by_design.csv', coverage)
    json_out(out / '07_ANALYSIS_SPECIFICATION.json', spec)
    atomic_text(out / '07_DECISIONS_AND_METHODS.md', methods(counts))
    payload_hashes = {p.name: digest(p) for p in sorted(out.iterdir()) if p.is_file()}
    status = {'script': Path(__file__).name, 'version': VERSION, 'script_sha256': digest(Path(__file__)),
              'generated_utc': datetime.now(timezone.utc).isoformat(),
              'status': 'CONDITIONAL_CARDIAC_SPECIFICATION_AND_COHORT_FROZEN',
              **counts, 'analysis_plan_frozen': True, 'conditional_cohort_frozen': True,
              'unconditional_timing_validity_established': False, 'physical_clock_verified': False,
              'movement_data_read': False, 'new_ECG_QC_applied': False, 'PLV_recomputed': False,
              'condition_effect_computed': False, 'preregistered': False,
              'verified_source_output_files': {'04': n4, '05': n5, '06': n6},
              'input_status_sha256': {str(d / n): digest(d / n) for d, n in
                    [(d4, '04_RUN_STATUS.json'), (d5, '05_RUN_STATUS.json'), (d6, '06_RUN_STATUS.json')]},
              'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__},
              'output_sha256': payload_hashes}
    json_out(out / '07_RUN_STATUS.json', status)
    print(json.dumps({k: v for k, v in status.items() if k not in ['output_sha256', 'input_status_sha256']}, indent=2))
    print('Read: ' + str(out / '07_DECISIONS_AND_METHODS.md'))


def make_spec(counts, design):
    return {
        'version': VERSION, 'status': 'FROZEN_CONDITIONAL_EXPLORATORY_ANALYSIS_PLAN',
        'research_question': 'Is emotional versus neutral story condition associated with cardiac phase consistency during collaborative cooking?',
        'scope': 'Cardiac ECG-derived RR intervals only; movement and mediation/gaze analyses are outside this release.',
        'history': 'Earlier exploratory phase summaries and method diagnostics were viewed. No condition effect is computed by Scripts05-07. Not a preregistration.',
        'timing': {'basis': 'exported_time_unverified', 'physical_clock_verified': False,
                   'correction_applied': False, 'optimized_alignment': False,
                   'assumption': 'Exported timestamps are treated as a common time axis. Physical offset and drift remain unmeasured. All results remain conditional on this assumption.'},
        'signal': {'name': 'ECG-derived RR interval (IBI)', 'internal_unit': 'seconds',
                   'display_unit': 'milliseconds', 'grid_hz': 4,
                   'interpolation': 'Existing Script04 PCHIP at RR midpoints within trusted runs, without extrapolation or gap bridging'},
        'primary_method': {'family': FAMILY, 'band_hz': [0.04, 0.15], 'core_s': 200,
            'context_each_side_s': 40, 'continuous_context_s': 280, 'step_s': 1, 'anchor_s': 40,
            'filter': 'Linear detrend and population-SD standardization per context; scipy Butterworth bandpass order argument 3 in SOS form; sosfiltfilt default odd padding; FFT Hilbert across full context',
            'sampled_context_interval': '[core_start-40, core_start+240), 1120 samples at 4 Hz',
            'sampled_scoring_interval': '[core_start, core_start+200), 800 samples at 4 Hz',
            'PLV': 'abs(mean(exp(1j*(phase_partner1-phase_partner2)))) on each core',
            'pair_phase_summary': 'Arithmetic mean of all eligible equal-length local-window PLVs',
            'meaning': 'Local consistency of the band-limited RR phase relationship, not zero-lag agreement, directionality or whole-interaction PLV',
            'rationale': 'Retain the existing tested specification, rather than optimize a condition effect. 200 s contains eight cycles at 0.04 Hz. This is not proof of an optimal or universally adequate window; context/band/duration sensitivities remain required.'},
        'eligibility': {'rule': 'At least one existing validated complete 280 s context yielding a finite 200 s core PLV',
            'coverage_fraction_cutoff': None,
            'justification': 'Target is available technically usable cooking periods, not an asserted representative whole interaction. No new arbitrary percentage threshold or first-three-minute restriction.',
            'weights': 'Equal local windows within pair phase; equal pair phases within each observed session-condition; equal sessions in primary model',
            'missingness': 'No imputation. Report all 71 candidate phases, reasons, durations and availability by condition/recipe/order. Missingness can bias observed-data associations.'},
        'primary_statistics': {
            'analysis_unit': 'Session with at least one eligible pair phase in each condition',
            'session_outcome': 'D_s = mean(pair-phase PLV for emotional) - mean(pair-phase PLV for neutral)',
            'covariates': {'delta_banana_proportion': 'Proportion Banana in emotional observations minus proportion Banana in neutral observations',
                           'delta_phase2_proportion': 'Proportion phase 2 in emotional observations minus proportion phase 2 in neutral observations'},
            'model': 'D_s = beta0 + beta_recipe*delta_banana_proportion + beta_order*delta_phase2_proportion + error_s',
            'target': 'beta0: adjusted average within-session emotional-minus-neutral difference at equal recipe and phase composition, assuming the additive adjustment model',
            'estimator': 'Unweighted OLS; one row per session',
            'inference': 'HC3 heteroskedasticity-robust covariance; two-sided t reference with S-3 residual degrees of freedom; 95% CI; alpha 0.05',
            'inference_limit': 'Small-sample approximate inference, not an exact randomization test. Report estimates, uncertainty, leverage and influence; no guarantee of adequate power.',
            'dependence': 'Aggregate all same-session partner and participant overlap before inference. PID nesting checked across all 71 candidates. Independent sessions remain a study assumption.',
            'recipe_and_order': 'Separate adjustment terms; check full rank before freezing. Do not add recipe-by-order interaction automatically: in the observed counterbalancing patterns its contrast can be identically zero.',
            'causal_status': 'Conditional association; original random assignment and exchangeability have not been independently verified. No causal effect claim.',
            'sessions_without_both_conditions': 'Retained in descriptive cardiac dataset, not imputed or used as within-session contrasts',
            'computed_here': False, 'design_checks': design},
        'reference_hypothesis': {
            'primary': 'Adjusted emotional-minus-neutral difference beta0 = 0, not PLV = 0',
            'surrogate_p_values': 'Not part of this condition-association primary test. No absolute above-chance interpersonal coupling claim is authorized.',
            'synthetic_noise': 'Existing independent-noise simulations are estimator diagnostics only, not a matched human null or cutoff',
            'PLV_bias': 'Equal core length standardizes nominal duration, but spectra, autocorrelation and nonstationarity can still differ. Interpret as measured phase consistency, not isolated interpersonal coupling.'},
        'sensitivity_plan': {
            'method_families': ['duration_180s', 'duration_250s', 'duration_300s', 'context_80s', 'order_2', 'order_4', 'band_004_010', 'band_006_015'],
            'clock_families': 'All existing +/-0.25, 1, 2, 5 s shift and +/-100 ppm drift scenarios',
            'rules': 'Apply the same session aggregation and model per family only if full rank and residual df permit. Report changed pair/session membership. Use existing matched-core summaries for same-duration estimator comparisons; duration variants can cover different periods. Do not select primary settings by estimates or p-values.',
            'clock_limit': 'Illustrative perturbations only; not calibrated uncertainty bounds or repairs',
            'influence': 'Leave one primary session out, refit the three-parameter mean model and report coefficient stability',
            'coverage': 'Describe coverage by condition; additionally add the emotional-minus-neutral session mean coverage-fraction difference as a single covariate if identifiable. Treat this as an exploratory selection diagnostic, not a correction for missing-not-at-random data.',
            'multiplicity': 'One primary test. Secondary sensitivities are robustness descriptions, not additional confirmatory discoveries; emphasize estimates and CIs, no smallest-p-value selection.'},
        'counts': counts,
        'references': ['https://doi.org/10.1016/j.neuroimage.2013.02.008',
                       'https://doi.org/10.1038/s41598-023-30405-5',
                       'https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLSResults.HC3_se.html',
                       'https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.RegressionResults.get_robustcov_results.html'],
        'next_stage': 'Estimate the locked session contrasts and adjusted association, run prespecified sensitivities, and report uncertainty and missingness. No repeated ECG processing is required.'}


def methods(c):
    return f'''# KITCO3 Script07: conditional cardiac specification, v{VERSION}

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

Verified counts: {c['candidate_pair_phases']} candidates,
{c['conditional_eligible_pair_phases']} eligible cardiac pair phases,
{c['scored_windows']} dependent scored windows. The full register keeps all
candidates, including those without a usable core.

## Primary statistical design

First average pair-phase outcomes within session and story condition. Then form
one emotional-minus-neutral difference per session with both conditions observed.
Regress those session differences on an intercept, the between-condition difference
in Banana proportion, and the between-condition difference in phase-2 proportion.
Each session has equal weight. The intercept is the adjusted condition contrast,
under an additive recipe/order adjustment model. This averages over observed
partner configurations; it does not identify a separate causal partner-change effect.

There are {c['primary_sessions_with_both_conditions']} eligible sessions, drawing
on {c['primary_comparison_pair_phases']} eligible pair phases. The other
{c['scored_pair_phases_without_within_session_contrast']} scored pair phases
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
'''


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--project-root', type=Path, default=DEFAULT_ROOT)
    p.add_argument('--script04-dir', type=Path)
    p.add_argument('--script05-dir', type=Path)
    p.add_argument('--script06-dir', type=Path, default=Path.home() / 'KITCO3_outputs/06_timing_method_review_v1_0_0')
    p.add_argument('--output-dir', type=Path, default=Path.home() / 'KITCO3_outputs/07_cardiac_specification_v1_0_0')
    args = p.parse_args()
    args.script04_dir = (args.script04_dir or args.project_root / 'outputs/04_dyadic_signals_v1_0_0').expanduser().resolve()
    args.script05_dir = (args.script05_dir or args.project_root / 'outputs/05_PLV_feasibility_v1_0_0').expanduser().resolve()
    args.script06_dir = args.script06_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    try:
        freeze(args)
    except Exception as exc:
        print(f'SCRIPT07 STOPPED: {exc}\nNo completion status is published for a failed run. Preserve existing files.', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
