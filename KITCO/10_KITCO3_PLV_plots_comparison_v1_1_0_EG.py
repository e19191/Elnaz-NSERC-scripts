#!/usr/bin/env python3
"""KITCO3 Script10 v1.1.0: frozen conditional cardiac condition analysis.

Run: python3 "/Volumes/One Touch/KITCO3_drive/scripts/10_KITCO3_PLV_plots_comparison_v1_1_0_EG.py"
Default INPUT: /Volumes/One Touch/KITCO3_drive/outputs/09_PLV_condition_audit_v1_1_0
Default OUTPUT: /Volumes/One Touch/KITCO3_drive/outputs/10_PLV_condition_comparison_v1_1_0
Requires numpy, pandas, scipy, matplotlib. No downloads or new ECG/PLV processing.
All result tables, figures, reports and plotting cache are written inside OUTPUT.
Never overwrites an existing output folder. A failed run publishes no completion.

Statistical contract: one emotional-minus-neutral contrast per observed complete
session; OLS intercept adjusted for recipe and phase-order composition; HC3
covariance and Student t reference with S-p df. Approximate small-sample inference.
No window-level inference, label permutations, causal claim, or absolute coupling
significance. Physical sensor clock synchronization remains unverified.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import html
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import numpy as np
import pandas as pd
import scipy
from scipy import stats

VERSION = '1.1.0'
PRIMARY = 'working_200s_40s'
SCRIPT07_SHA256 = '6e4f45ccfad6b577c3c1c56c07acac1982614089bde87c2a5ce073ee302dcc96'
MODEL_COLS = ['Intercept', 'delta_banana_proportion', 'delta_phase2_proportion']
NEUTRAL, EMOTIONAL = '#0072B2', '#D55E00'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def loadcsv(path):
    return pd.read_csv(path, keep_default_na=False)


def jload(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def clean_json(obj):
    if isinstance(obj, dict):
        return {str(k): clean_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, np.ndarray)):
        return [clean_json(v) for v in obj]
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, Path):
        return str(obj)
    return obj


def textout(path, content):
    tmp = path.with_name(path.name + '.writing')
    with tmp.open('x', encoding='utf-8', newline='') as f:
        f.write(content)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def jout(path, obj):
    textout(path, json.dumps(clean_json(obj), indent=2, allow_nan=False) + '\n')


def cout(path, df):
    textout(path, df.to_csv(index=False, float_format='%.12g', na_rep=''))


def boolean(s):
    vals = s.astype(str).str.lower()
    require(vals.isin(['true', 'false']).all(), 'Invalid boolean column: ' + str(s.name))
    return vals.eq('true')


def numeric(s, allow_blank=False):
    a = pd.to_numeric(s, errors='coerce')
    if allow_blank:
        require((a.notna() | s.astype(str).eq('')).all(), 'Malformed numeric value in ' + str(s.name))
    else:
        require(np.isfinite(a.to_numpy(float)).all(), 'Missing/nonfinite numeric value in ' + str(s.name))
    return a


def fit_hc3(x, y, names):
    """OLS + HC3, no extra residual-df multiplier. Recomputed by QR for checking."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    require(x.ndim == 2 and y.shape == (len(x),) and len(names) == x.shape[1], 'Model dimensions differ.')
    n, p = x.shape
    require(np.isfinite(x).all() and np.isfinite(y).all(), 'Nonfinite model data.')
    require(n > p and np.linalg.matrix_rank(x) == p, 'Insufficient residual df or rank-deficient design.')
    require(np.linalg.cond(x) < 1e10, 'Numerically ill-conditioned design; no unstable estimate reported.')
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    bread = np.linalg.solve(x.T @ x, np.eye(p))
    h = np.einsum('ij,ji->i', x, bread @ x.T)
    require(np.all(h < 1 - 1e-10), 'HC3 is undefined for unit-leverage observations.')
    fitted = x @ beta
    residual = y - fitted
    z = x * (residual / (1 - h))[:, None]
    covariance = bread @ (z.T @ z) @ bread
    covariance = (covariance + covariance.T) / 2
    se = np.sqrt(np.maximum(np.diag(covariance), 0))
    require(np.isfinite(se).all() and np.all(se > 1e-14), 'Degenerate residual variance; t inference is undefined.')
    df = n - p
    tvalue = beta / se
    pvalue = 2 * stats.t.sf(np.abs(tvalue), df)
    critical = stats.t.ppf(.975, df)
    # Independent QR form of the coefficients and HC3 covariance.
    q, r = np.linalg.qr(x, mode='reduced')
    bq = np.linalg.solve(r, q.T @ y)
    hq = np.sum(q*q, axis=1)
    influence = np.linalg.solve(r, q.T) * ((y - x @ bq) / (1 - hq))[None, :]
    cq = influence @ influence.T
    require(np.allclose(beta, bq, rtol=1e-9, atol=1e-11) and
            np.allclose(covariance, cq, rtol=1e-8, atol=1e-13), 'Independent QR validation failed.')
    table = pd.DataFrame({'term': names, 'estimate': beta, 'HC3_SE': se,
                          't_value': tvalue, 'df': df, 'two_sided_p': pvalue,
                          'CI95_low': beta-critical*se, 'CI95_high': beta+critical*se})
    mse = float(residual @ residual / df)
    cook = residual**2 / (p*mse) * h / (1-h)**2
    return {'coefficients': table, 'covariance': covariance, 'beta': beta,
            'fitted': fitted, 'residual': residual, 'leverage': h,
            'cooks_distance': cook, 'standardized_residual': residual / np.sqrt(mse*(1-h)),
            'n': n, 'p': p, 'df': df, 'condition_number': float(np.linalg.cond(x)),
            'mse': mse}


def self_tests():
    rng = np.random.default_rng(20261016)
    x = np.column_stack([np.ones(24), rng.normal(size=(24, 2))])
    y = x @ np.array([.04, -.02, .01]) + rng.normal(size=24)*.05
    m = fit_hc3(x, y, MODEL_COLS)
    # HC3 equals the sum of outer products of uncentered delete-one coefficient
    # changes. This is not the centered jackknife covariance formula.
    changes = []
    for i in range(len(x)):
        b = np.linalg.lstsq(np.delete(x,i,0), np.delete(y,i), rcond=None)[0]
        changes.append(b - m['beta'])
    changes = np.array(changes)
    require(np.allclose(m['covariance'], changes.T@changes, rtol=1e-9, atol=1e-12), 'Delete-one HC3 self-test failed.')
    perm = rng.permutation(len(x))
    other = fit_hc3(x[perm], y[perm], MODEL_COLS)
    require(np.allclose(m['beta'], other['beta'], atol=1e-12), 'Row-order self-test failed.')
    require(abs(stats.t.ppf(.975,13)-2.16036865646)<1e-9, 'Student t quantile check failed.')
    try:
        fit_hc3(np.ones((8,3)), np.arange(8), MODEL_COLS)
    except ValueError:
        pass
    else:
        raise ValueError('Rank-deficiency self-test failed.')
    return {'status': 'PASS', 'checks': ['QR coefficient/covariance agreement',
            'delete-one identity for HC3', 'row-order invariance', 't(13) quantile', 'rank-deficiency rejection']}


def aggregate_sessions(pairdata, value_col='mean_local_PLV'):
    rows = []
    for sid, g in pairdata.groupby('Session_ID', sort=True):
        e, n = g.loc[g.Story.eq('Emotional')], g.loc[g.Story.eq('Neutral')]
        if len(e) == 0 or len(n) == 0:
            continue
        row = {'Session_ID': sid, 'emotional_pair_phases': len(e), 'neutral_pair_phases': len(n),
               'emotional_mean_PLV': float(e[value_col].mean()),
               'neutral_mean_PLV': float(n[value_col].mean()),
               'D_emotional_minus_neutral': float(e[value_col].mean()-n[value_col].mean()),
               'delta_banana_proportion': float(e.Recipe.eq('Banana').mean()-n.Recipe.eq('Banana').mean()),
               'delta_phase2_proportion': float(e.Phase.eq(2).mean()-n.Phase.eq(2).mean()),
               'emotional_pair_ids': '|'.join(sorted(e.pair_id)),
               'neutral_pair_ids': '|'.join(sorted(n.pair_id))}
        if 'coverage_fraction' in pairdata:
            row['emotional_mean_coverage'] = float(e.coverage_fraction.mean())
            row['neutral_mean_coverage'] = float(n.coverage_fraction.mean())
            row['delta_coverage_fraction'] = float(e.coverage_fraction.mean()-n.coverage_fraction.mean())
        rows.append(row)
    return pd.DataFrame(rows)


def fit_sessions(sessions, extra_coverage=False):
    require(len(sessions)>0, 'No sessions have both observed conditions.')
    cols = MODEL_COLS + (['delta_coverage_fraction'] if extra_coverage else [])
    x = np.column_stack([np.ones(len(sessions))] + [sessions[c].to_numpy(float) for c in cols[1:]])
    return fit_hc3(x, sessions.D_emotional_minus_neutral.to_numpy(float), cols)


def intercept_row(model, label, cohort, sessions):
    z = model['coefficients'].iloc[0]
    return {'analysis': label, 'status': 'ESTIMATED', 'pair_phases_with_scores': len(cohort),
            'pair_phases_in_complete_sessions': int(sessions.emotional_pair_phases.sum()+sessions.neutral_pair_phases.sum()),
            'sessions': len(sessions), 'df': model['df'], 'estimate': float(z.estimate),
            'HC3_SE': float(z.HC3_SE), 'CI95_low': float(z.CI95_low), 'CI95_high': float(z.CI95_high),
            'note': 'Secondary interval: pointwise, approximate, not multiplicity adjusted'}


def read_inputs(root):
    require(root.is_dir(), f'Script07 folder not found: {root}. Use --audit-dir to point to the completed Script09 output.')
    status = jload(root/'07_RUN_STATUS.json')
    require(status.get('status')=='CONDITIONAL_CARDIAC_SPECIFICATION_AND_COHORT_FROZEN', 'Script07 did not publish the required completion status.')
    require(status.get('script_sha256')==SCRIPT07_SHA256 and status.get('version')=='1.0.0', 'Unsupported Script07 release; do not silently change the frozen specification.')
    require(status.get('condition_effect_computed') is False and status.get('physical_clock_verified') is False, 'Unexpected upstream analysis/timing state.')
    manifest = status.get('output_sha256',{})
    expected = {'07_ANALYSIS_SPECIFICATION.json', '07_DECISIONS_AND_METHODS.md', '07_all_specification_summaries.csv',
                '07_availability_by_design.csv', '07_conditional_cardiac_dataset.csv', '07_matched_core_sensitivity.csv',
                '07_pair_phase_register_71.csv', '07_primary_session_design.csv', '07_session_availability.csv'}
    require(set(manifest)==expected, 'Unexpected Script07 output manifest.')
    for name, h in manifest.items():
        require((root/name).is_file() and sha(root/name)==h, 'Source file changed or missing: '+name)
    spec = jload(root/'07_ANALYSIS_SPECIFICATION.json')
    require(spec['primary_method']['family']==PRIMARY and spec['primary_method']['core_s']==200 and
            spec['primary_method']['band_hz']==[.04,.15] and spec['primary_method']['context_each_side_s']==40,
            'Unsupported frozen PLV specification.')
    require('HC3' in spec['primary_statistics']['inference'] and spec['timing']['basis']=='exported_time_unverified', 'Unsupported frozen inference/timing specification.')
    reg = loadcsv(root/'07_pair_phase_register_71.csv')
    cohort = loadcsv(root/'07_conditional_cardiac_dataset.csv')
    design = loadcsv(root/'07_primary_session_design.csv')
    for d in [reg, cohort]:
        require(d.pair_id.is_unique, 'Duplicate pair phase.')
        d['conditional_eligible'] = boolean(d.conditional_eligible)
        d['primary_condition_comparison_member'] = boolean(d.primary_condition_comparison_member)
        for col in ['Phase', 'PID_1', 'PID_2', 'shared_cooking_s', 'scored_windows', 'scored_core_union_s', 'scored_core_coverage_fraction']:
            d[col] = numeric(d[col])
        d['mean_local_PLV'] = numeric(d.mean_local_PLV,allow_blank=True)
        coding={'NA':('Neutral','Apple'),'NB':('Neutral','Banana'),'EA':('Emotional','Apple'),'EB':('Emotional','Banana')}
        require(all(r.Condition in coding and (r.Story,r.Recipe)==coding[r.Condition] for r in d.itertuples()), 'Condition coding mismatch; NA must remain a literal condition.')
    require(len(reg)==71 and len(cohort)==48 and cohort.conditional_eligible.all(), 'Frozen membership differs.')
    require(set(cohort.pair_id)==set(reg.loc[reg.conditional_eligible,'pair_id']), 'Cohort and register mismatch.')
    pd.testing.assert_frame_equal(cohort.reset_index(drop=True),reg.loc[reg.conditional_eligible].reset_index(drop=True))
    require(np.isfinite(cohort.mean_local_PLV).all() and cohort.mean_local_PLV.between(0,1).all(), 'Invalid primary PLV values.')
    require(cohort.scored_windows.gt(0).all() and cohort.scored_core_coverage_fraction.between(0,1).all(), 'Invalid support/coverage.')
    require(int(cohort.scored_windows.sum())==10434, 'Unexpected frozen window count.')
    people=pd.concat([reg[['Session_ID','PID_1']].rename(columns={'PID_1':'PID'}),reg[['Session_ID','PID_2']].rename(columns={'PID_2':'PID'})])
    require(people.groupby('PID').Session_ID.nunique().max()==1, 'Participant crosses sessions; the independence plan requires revision.')
    cohort['coverage_fraction']=cohort.scored_core_coverage_fraction
    sessions=aggregate_sessions(cohort)
    require(len(sessions)==16 and set(sessions.Session_ID)==set(design.Session_ID), 'Primary session membership differs from frozen design.')
    for col in ['delta_banana_proportion','delta_phase2_proportion','emotional_pair_phases','neutral_pair_phases']:
        joined=sessions[['Session_ID',col]].merge(design[['Session_ID',col]],on='Session_ID',validate='one_to_one',suffixes=('_now','_frozen'))
        require(np.allclose(joined[col+'_now'].to_numpy(float),joined[col+'_frozen'].to_numpy(float),atol=1e-10), 'Frozen session design mismatch: '+col)
    member=cohort.Session_ID.isin(sessions.Session_ID)
    require(np.array_equal(member.to_numpy(),cohort.primary_condition_comparison_member.to_numpy()) and member.sum()==42, 'Frozen primary membership flag differs.')
    families=loadcsv(root/'07_all_specification_summaries.csv')
    matched=loadcsv(root/'07_matched_core_sensitivity.csv')
    require(len(families)==1349 and not families.duplicated(['pair_id','family']).any(), 'Method-family cardinality/uniqueness mismatch.')
    require(set(families.pair_id)==set(reg.pair_id), 'Unknown sensitivity pair IDs.')
    for col in ['windows','core_union_s','shared_cooking_s']:
        families[col]=numeric(families[col])
    families['mean_local_PLV']=numeric(families.mean_local_PLV,allow_blank=True)
    require(np.isfinite(families.loc[families.windows.gt(0),'mean_local_PLV']).all(), 'Missing finite sensitivity scores.')
    require(families.loc[families.windows.gt(0),'mean_local_PLV'].between(0,1).all(), 'Sensitivity PLV out of range.')
    require(not matched.duplicated(['pair_id','family']).any() and set(matched.pair_id)<=set(reg.pair_id), 'Invalid matched-core membership.')
    return status,spec,reg,cohort,sessions,families,matched


def sensitivity_analysis(reg,cohort,families,matched,primary_sessions):
    metadata=reg[['pair_id','Session_ID','Phase','PID_1','PID_2','Condition','Story','Recipe']]
    native=[]; common=[]; matched_rows=[]; memberships=[]; all_sessions=[]
    primary_pair_ids=set(cohort.pair_id)
    for family,g in families.groupby('family',sort=True):
        if family==PRIMARY:
            continue
        use=g.loc[g.windows.gt(0)].merge(metadata,on='pair_id',validate='one_to_one')
        require(use.shared_cooking_s.gt(0).all(), 'Nonpositive sensitivity cooking duration.')
        use['coverage_fraction']=use.core_union_s/use.shared_cooking_s
        require(use.coverage_fraction.between(-1e-9,1+1e-9).all(), 'Invalid sensitivity union coverage.')
        ss=aggregate_sessions(use)
        if len(ss):
            q=ss.copy();q.insert(0,'analysis',family);q.insert(1,'scope','NATIVE_FAMILY_SUPPORT');all_sessions.append(q)
        ss_ids=set(ss.Session_ID) if len(ss) else set()
        for rr in use.itertuples():
            memberships.append({'family':family,'pair_id':rr.pair_id,'Session_ID':rr.Session_ID,
                'in_primary_pair_cohort':rr.pair_id in primary_pair_ids,
                'in_primary_complete_sessions':rr.Session_ID in set(primary_sessions.Session_ID),
                'in_variant_complete_sessions':rr.Session_ID in ss_ids})
        try:
            model=fit_sessions(ss)
            z=intercept_row(model,family,use,ss)
            z['same_pair_ids_as_primary']=set(use.pair_id)==primary_pair_ids
            z['same_complete_sessions_as_primary']=ss_ids==set(primary_sessions.Session_ID)
            z['scope']='NATIVE_FAMILY_SUPPORT';native.append(z)
        except ValueError as exc:
            native.append({'analysis':family,'status':'NOT_ESTIMABLE','pair_phases_with_scores':len(use),'sessions':len(ss),'note':str(exc),'scope':'NATIVE_FAMILY_SUPPORT'})
        # Same pair identities isolate cohort changes, but duration variants can
        # still use different time periods. Both results share exactly one X.
        ids=primary_pair_ids & set(use.pair_id)
        b=cohort.loc[cohort.pair_id.isin(ids)].copy();v=use.loc[use.pair_id.isin(ids)].copy()
        bs,vs=aggregate_sessions(b),aggregate_sessions(v)
        try:
            z=compare_models(bs,vs,family,'COMMON_PAIR_COHORT_DIFFERENT_PERIODS_POSSIBLE')
            z['common_pair_phases']=len(ids);common.append(z)
        except ValueError as exc:
            common.append({'analysis':family,'status':'NOT_ESTIMABLE','common_pair_phases':len(ids),'note':str(exc)})
    for family,g in matched.groupby('family',sort=True):
        g=g.copy();g['matched_windows']=numeric(g.matched_windows)
        g=g.loc[g.matched_windows.gt(0)]
        if len(g)==0:
            continue
        for col in ['baseline_mean_on_matched_cores','variant_mean_on_matched_cores','signed_change_in_pair_mean']:
            g[col]=numeric(g[col])
        require(g.baseline_mean_on_matched_cores.between(0,1).all() and g.variant_mean_on_matched_cores.between(0,1).all(),'Invalid matched means.')
        require(np.allclose(g.variant_mean_on_matched_cores-g.baseline_mean_on_matched_cores,g.signed_change_in_pair_mean,atol=3e-9),'Matched signed-change arithmetic mismatch.')
        data=g.merge(metadata,on='pair_id',validate='one_to_one')
        bs=aggregate_sessions(data,'baseline_mean_on_matched_cores')
        vs=aggregate_sessions(data,'variant_mean_on_matched_cores')
        try:
            z=compare_models(bs,vs,family,'SAME_PAIR_PHASES_AND_MATCHED_SCORING_CORES')
            z['common_pair_phases']=len(data);z['matched_windows']=int(data.matched_windows.sum());matched_rows.append(z)
        except ValueError as exc:
            matched_rows.append({'analysis':family,'status':'NOT_ESTIMABLE','common_pair_phases':len(data),'note':str(exc)})
    return pd.DataFrame(native),pd.DataFrame(common),pd.DataFrame(matched_rows),pd.DataFrame(memberships),pd.concat(all_sessions,ignore_index=True) if all_sessions else pd.DataFrame()


def compare_models(bs,vs,family,scope):
    require(len(bs)>0 and len(vs)>0,'No common complete session.')
    bs,vs=bs.sort_values('Session_ID').reset_index(drop=True),vs.sort_values('Session_ID').reset_index(drop=True)
    require(bs.Session_ID.equals(vs.Session_ID),'Common-cohort session mismatch.')
    require(np.allclose(bs[MODEL_COLS[1:]].to_numpy(float),vs[MODEL_COLS[1:]].to_numpy(float),atol=1e-12),'Paired models have different designs.')
    bm,vm=fit_sessions(bs),fit_sessions(vs)
    diff=vs.copy();diff['D_emotional_minus_neutral']=vs.D_emotional_minus_neutral-bs.D_emotional_minus_neutral
    # Estimate uncertainty of the paired change directly, never subtract CI ends.
    try:
        dm=fit_sessions(diff);q=dm['coefficients'].iloc[0]
        delta_se,lo,hi=float(q.HC3_SE),float(q.CI95_low),float(q.CI95_high)
        delta_status='ESTIMATED'
    except ValueError as exc:
        if 'Degenerate residual variance' not in str(exc):
            raise
        delta_se,lo,hi=np.nan,np.nan,np.nan;delta_status='DEGENERATE_CHANGE_VARIANCE_NO_CI'
    return {'analysis':family,'status':'ESTIMATED','scope':scope,'sessions':len(bs),'df':bm['df'],
            'baseline_estimate':bm['beta'][0],'variant_estimate':vm['beta'][0],
            'change_in_adjusted_contrast':vm['beta'][0]-bm['beta'][0],
            'change_HC3_SE':delta_se,'change_CI95_low':lo,'change_CI95_high':hi,
            'change_interval_status':delta_status,
            'note':'Pointwise secondary interval; no independent-discovery claim'}


def save_figure(fig,stem):
    for ext in ['png','pdf','svg']:
        p=stem.with_suffix('.'+ext);tmp=p.with_name(p.stem+'.writing.'+ext)
        fig.savefig(tmp,format=ext,dpi=300,facecolor='white',bbox_inches='tight')
        with tmp.open('rb') as f:
            f.seek(max(0,tmp.stat().st_size-1024));tail=f.read()
        require((ext!='pdf' or b'%%EOF' in tail) and (ext!='png' or tail.endswith(b'IEND\xaeB`\x82')) and
                (ext!='svg' or b'</svg>' in tail),'Incomplete figure: '+str(tmp))
        with tmp.open('rb+') as f:
            f.flush();os.fsync(f.fileno())
        os.replace(tmp,p)


def family_label(name):
    labels = {'band_004_010': 'RR band: 0.04-0.10 Hz',
              'band_006_015': 'RR band: 0.06-0.15 Hz',
              'context_80s': 'Context: 80 s per side',
              'duration_180s': 'Core: 180 s', 'duration_250s': 'Core: 250 s',
              'duration_300s': 'Core: 300 s',
              'order_2': 'Butterworth order argument: 2',
              'order_4': 'Butterworth order argument: 4'}
    if name in labels:
        return labels[name]
    if name.startswith('clock_shift_'):
        value = name.removeprefix('clock_shift_').removesuffix('s').replace('minus','-').replace('p','.')
        return f'Partner 2 query shift: {float(value):+g} s'
    if name.startswith('clock_drift_'):
        value = name.removeprefix('clock_drift_').removesuffix('ppm').replace('minus','-')
        return f'Partner 2 query drift: {float(value):+g} ppm'
    return name


def check_final_figure_files(folder):
    # Some filesystems can leave a stale temporary writer copy. It is never a
    # scientific output. Remove only a known plot temporary file whose final
    # counterpart exists and has a complete format trailer.
    for p in folder.glob('*.writing.*'):
        ext = p.suffix.lower()
        require(ext in ['.png', '.pdf', '.svg'], 'Unknown temporary plot: '+str(p))
        final = p.with_name(p.name.replace('.writing.', '.'))
        require(final.is_file(), 'Incomplete plot export: '+str(p))
        with final.open('rb') as f:
            f.seek(max(0, final.stat().st_size-1024));tail=f.read()
        require((ext!='.png' or tail.endswith(b'IEND\xaeB`\x82')) and
                (ext!='.pdf' or b'%%EOF' in tail) and
                (ext!='.svg' or b'</svg>' in tail), 'Incomplete final plot: '+str(final))
        p.unlink()
    require(not list(folder.glob('*.writing.*')), 'Unfinished plot files remain.')


def figures(out,cohort,sessions,model,native,matched,loo,diagnostic,reg):
    # Prevent matplotlib cache writes to the user's home directory.
    os.environ['MPLCONFIGDIR']=str(out/'runtime_cache'/'matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12,
        'axes.labelsize':10,'axes.spines.top':False,'axes.spines.right':False,
        'pdf.fonttype':42,'svg.fonttype':'none','savefig.dpi':300})
    dest=out/'figures';dest.mkdir()
    rng=np.random.default_rng(20261026)
    fig, ax=plt.subplots(figsize=(7,5))
    groups=[cohort.loc[cohort.Story.eq(story),'mean_local_PLV'].to_numpy(float) for story in ['Neutral','Emotional']]
    boxes=ax.boxplot(groups,positions=[0,1],widths=.4,whis=(0,100),showfliers=False,patch_artist=True)
    desc=[]
    for i,(story,values,color) in enumerate(zip(['Neutral','Emotional'],groups,[NEUTRAL,EMOTIONAL])):
        boxes['boxes'][i].set_facecolor(color);boxes['boxes'][i].set_alpha(.2)
        ax.scatter(i+rng.uniform(-.13,.13,len(values)),values,color=color,s=30,alpha=.8)
        ax.scatter([i],[values.mean()],marker='D',s=65,color='black',zorder=4)
        desc.append({'Story':story,'pair_phases':len(values),'sessions':cohort.loc[cohort.Story.eq(story),'Session_ID'].nunique(),'mean':values.mean(),'median':np.median(values),'min':values.min(),'max':values.max(),'q25':np.quantile(values,.25),'q75':np.quantile(values,.75)})
    ax.set(xticks=[0,1],xticklabels=[f"{z['Story']}\n{z['pair_phases']} pair phases; {z['sessions']} sessions" for z in desc],ylabel='Pair-phase mean local PLV',ylim=(0,1),title='All scored pair phases: emotional versus neutral')
    fig.text(.02,.01,'Points: pair phases. Black diamonds: arithmetic means. Boxes: median/IQR; whiskers: min/max.\nThese are descriptive distributions, not confidence intervals. Within-session observations are dependent.',fontsize=8)
    fig.tight_layout(rect=(0,.1,1,1));save_figure(fig,dest/'10_all_pair_phase_distribution');plt.close(fig)
    cout(out/'10_all_pair_phase_descriptive_summary.csv',pd.DataFrame(desc))
    caption='Conditional on exported timestamps; physical synchronization unverified. Intervals are approximate, pointwise HC3/t 95% CIs.'
    q=model['coefficients'].iloc[0]
    fig,ax=plt.subplots(1,3,figsize=(14,5),gridspec_kw={'width_ratios':[1,1.3,1]})
    for r in sessions.itertuples():
        ax[0].plot([0,1],[r.neutral_mean_PLV,r.emotional_mean_PLV],color='#A5A5A5',lw=1,zorder=1)
    ax[0].scatter(np.zeros(len(sessions)),sessions.neutral_mean_PLV,color=NEUTRAL,s=28,label='Neutral',zorder=2)
    ax[0].scatter(np.ones(len(sessions)),sessions.emotional_mean_PLV,color=EMOTIONAL,s=28,label='Emotional',zorder=2)
    ax[0].set(xticks=[0,1],xticklabels=['Neutral','Emotional'],ylabel='Session mean of pair-phase local PLV',
              ylim=(0,max(.4,float(sessions[['neutral_mean_PLV','emotional_mean_PLV']].to_numpy().max())*1.1)),xlim=(-.25,1.25),title='A  Observed session means')
    yy=np.arange(len(sessions))
    ax[1].axvline(0,color='#555555',lw=1)
    ax[1].scatter(sessions.D_emotional_minus_neutral,yy,color='#333333',s=30)
    ax[1].set(yticks=yy,yticklabels=sessions.Session_ID,xlabel='Emotional minus neutral PLV',title='B  Unadjusted session contrasts')
    ax[1].invert_yaxis();ax[1].grid(axis='x',alpha=.18)
    ax[2].axvline(0,color='#555555',lw=1)
    ax[2].errorbar(q.estimate,0,xerr=[[q.estimate-q.CI95_low],[q.CI95_high-q.estimate]],fmt='o',color=NEUTRAL,capsize=5,ms=7)
    ax[2].set(yticks=[],ylim=(-1,1),xlabel='Adjusted emotional minus neutral PLV',title='C  Primary adjusted association')
    ax[2].text(.5,.92,f"{len(sessions)} sessions; residual df = {model['df']}\np = {q.two_sided_p:.4g}",transform=ax[2].transAxes,ha='center',va='top',bbox={'facecolor':'white','edgecolor':'none','pad':4})
    fig.suptitle('Cardiac phase consistency during collaborative cooking',fontsize=15)
    fig.text(.02,.015,caption+'\nLines in A link the same session. B shows raw differences; C adjusts for recipe and phase order. No window is an independent case.',fontsize=8)
    fig.tight_layout(rect=(0,.11,1,.93));save_figure(fig,dest/'10_primary_condition_comparison');plt.close(fig)
    for clock,stem,title in [(False,'10_method_sensitivity','Method sensitivity: adjusted condition associations'),(True,'10_clock_scenarios','Illustrative timing scenarios: adjusted condition associations')]:
        d=native.loc[native.analysis.str.startswith('clock_').eq(clock)].copy()
        d=d.loc[d.status.eq('ESTIMATED')].sort_values('analysis')
        if len(d)==0:continue
        fig,ax=plt.subplots(figsize=(11,max(5,.46*len(d)+2)))
        y=np.arange(len(d));ax.axvline(0,color='#555555',lw=1);ax.axvline(q.estimate,color=NEUTRAL,ls='--',lw=1,label='Primary estimate')
        ax.errorbar(d.estimate,y,xerr=np.array([d.estimate-d.CI95_low,d.CI95_high-d.estimate]),fmt='o',capsize=3,color='#333333')
        ax.set(yticks=y,yticklabels=[f'{family_label(r.analysis)}  (S={int(r.sessions)})' for r in d.itertuples()],xlabel='Adjusted emotional minus neutral PLV',title=title)
        ax.invert_yaxis();ax.grid(axis='x',alpha=.18);ax.legend(loc='best',frameon=False,fontsize=9)
        fig.text(.02,.015,caption+'\nFamily-specific support can change pair/session membership. Clock scenarios are not measured uncertainty bounds.' if clock else caption+'\nFamily-specific support can change pair/session membership and time periods. Secondary intervals are not multiplicity adjusted.',fontsize=8)
        fig.tight_layout(rect=(0,.09,1,1));save_figure(fig,dest/stem);plt.close(fig)
    d=matched.loc[matched.status.eq('ESTIMATED')].copy()
    if len(d):
        d=d.sort_values('analysis');fig,ax=plt.subplots(figsize=(11,max(6,.4*len(d)+2)))
        ax.axvline(0,color='#555555',lw=1);y=np.arange(len(d))
        ax.errorbar(d.change_in_adjusted_contrast,y,xerr=np.array([d.change_in_adjusted_contrast-d.change_CI95_low,d.change_CI95_high-d.change_in_adjusted_contrast]),fmt='o',capsize=3,color='#333333')
        ax.set(yticks=y,yticklabels=[f'{family_label(r.analysis)}  (S={int(r.sessions)})' for r in d.itertuples()],xlabel='Variant minus baseline adjusted condition contrast',title='Matched scoring cores: change in the condition association')
        ax.invert_yaxis();ax.grid(axis='x',alpha=.18)
        fig.text(.02,.015,caption+'\nBaseline and variant share pair phases and scoring cores within each row. CI is fitted to paired changes, not obtained by subtracting CI endpoints.',fontsize=8)
        fig.tight_layout(rect=(0,.09,1,1));save_figure(fig,dest/'10_matched_core_changes');plt.close(fig)
    fig,ax=plt.subplots(2,2,figsize=(11,8))
    ax[0,0].scatter(diagnostic.fitted,diagnostic.residual,color='#333333');ax[0,0].axhline(0,color='#888888',lw=1)
    ax[0,0].set(xlabel='Fitted session contrast',ylabel='Residual',title='A  Residual versus fitted')
    (theory,ordered),(slope,intercept,_) = stats.probplot(diagnostic.standardized_residual,dist='norm')
    ax[0,1].scatter(theory,ordered,color='#333333');ax[0,1].plot(theory,intercept+slope*theory,color=NEUTRAL)
    ax[0,1].set(xlabel='Normal theoretical quantile',ylabel='Standardized residual',title='B  Residual Q-Q diagnostic')
    ax[1,0].scatter(diagnostic.leverage,diagnostic.standardized_residual,color='#333333')
    for r in diagnostic.nlargest(3,'cooks_distance').itertuples():ax[1,0].annotate(r.Session_ID,(r.leverage,r.standardized_residual),xytext=(4,4),textcoords='offset points',fontsize=8)
    ax[1,0].set(xlabel='Design leverage',ylabel='Standardized residual',title='C  Leverage and residual')
    yy=np.arange(len(loo));ax[1,1].axvline(q.estimate,color=NEUTRAL,ls='--',label='Primary estimate');ax[1,1].axvline(0,color='#888888',lw=1)
    ax[1,1].scatter(loo.estimate,yy,color='#333333',s=22)
    ax[1,1].set(yticks=yy,yticklabels=loo.session_omitted,xlabel='Adjusted contrast after omitting session',title='D  Leave-one-session-out estimates')
    ax[1,1].tick_params(axis='y',labelsize=8);ax[1,1].invert_yaxis();ax[1,1].legend(frameon=False,fontsize=8)
    fig.text(.02,.012,'Diagnostics do not trigger automatic exclusions. Largest three conventional Cook distances are labelled in C.\nHC3 uncertainty allows heteroskedasticity; t inference remains approximate. All results are conditional on the timestamp assumption.',fontsize=8)
    fig.tight_layout(rect=(0,.07,1,1));save_figure(fig,dest/'10_model_diagnostics');plt.close(fig)
    fig,ax=plt.subplots(1,2,figsize=(12,5))
    codes=['NA','NB','EA','EB'];xs=np.arange(4)
    n=[int(reg.Condition.eq(c).sum()) for c in codes];k=[int(cohort.Condition.eq(c).sum()) for c in codes]
    ax[0].bar(xs-.17,n,width=.34,color='#B7B7B7',label='Candidate phases');ax[0].bar(xs+.17,k,width=.34,color=NEUTRAL,label='Scored phases')
    ax[0].set(xticks=xs,xticklabels=['Neutral\nApple','Neutral\nBanana','Emotional\nApple','Emotional\nBanana'],ylabel='Pair phases',title='A  Availability, not a condition-effect test')
    ax[0].legend(frameon=False,fontsize=9)
    for story,color in [('Neutral',NEUTRAL),('Emotional',EMOTIONAL)]:
        g=cohort.loc[cohort.Story.eq(story)];ax[1].scatter(g.coverage_fraction,g.mean_local_PLV,color=color,label=story,alpha=.8,s=30)
    ax[1].set(xlabel='Scored-core union / shared cooking duration',ylabel='Pair-phase mean local PLV',xlim=(0,1),ylim=(0,max(.4,cohort.mean_local_PLV.max()*1.1)),title='B  Coverage and observed phase consistency')
    ax[1].legend(frameon=False,fontsize=9)
    fig.text(.02,.015,'Panel B is descriptive: pair phases share participants/sessions. Coverage does not measure representativeness of missing periods.\nNo statistical test treats these points, windows or 4 Hz samples as independent cases.',fontsize=8)
    fig.tight_layout(rect=(0,.1,1,1));save_figure(fig,dest/'10_availability_and_coverage');plt.close(fig)
    return matplotlib.__version__


def report_text(result,primary,native,matched,coverage,loo,counts):
    q=primary.iloc[0]
    return f'''# KITCO3 cardiac condition analysis v{VERSION}

## Primary result

Adjusted emotional-minus-neutral difference in session mean local RR PLV:
**{q.estimate:.6f}**, HC3 SE {q.HC3_SE:.6f}, approximate 95% CI
[{q.CI95_low:.6f}, {q.CI95_high:.6f}], t({int(q.df)}) = {q.t_value:.4f},
two-sided p = {q.two_sided_p:.6g}.

The interval {'includes' if q.CI95_low <= 0 <= q.CI95_high else 'does not include'} zero.
Report the estimate and interval, not a claim that a nonsignificant result proves
no association. This is an adjusted observed-data association, not a causal effect
or evidence of interpersonal coupling above chance. Physical sensor clock offset
and drift remain unverified. The entire analysis is conditional on treating the
exported timestamps as a common time axis.

## Cases and weighting

71 candidate pair phases; 48 have scores. The primary comparison uses 42 pair
phases in 16 sessions with both conditions observed. Six other scored pair phases
remain descriptive. No ECG QC is repeated and no movement data are read.

Within each pair phase, the existing outcome is the mean of all eligible local
200 s PLV cores with 40 s context on each side, 1 s spacing and a 0.04-0.15 Hz
band on the 4 Hz RR interval (IBI) grid. The 10,434 heavily overlapping windows
are not inferential N. No whole-interaction PLV or independent heart-sample N is
implied. Unequal amounts of usable cooking time are retained and reported.

Each session-condition mean equally weights its observed pair-phase means.
Each complete session then contributes one emotional-minus-neutral difference.
The model is D_s = beta0 + beta_recipe*delta_Banana_s + beta_order*delta_Phase2_s.
The two covariates are emotional-minus-neutral differences in recipe and phase
composition. Sessions have equal model weight. The primary coefficient is beta0.
This assumes additive recipe/order adjustment; it does not isolate partner-change
causality or account for every possible interaction or omitted confounder.

## Inference and checks

OLS coefficients, HC3 sandwich covariance with leverage correction, and a t
reference with S-p = 13 df follow Script07 exactly. These CIs and p-values are
small-sample approximations, not exact randomization inference. PID nesting was
checked across the candidate phases; independence between sessions remains a
study assumption. No coefficient or p-value is used to change inclusion.
Nuisance-term p-values in the coefficient table are not additional primary tests.

The implementation checks OLS/HC3 through a separate QR calculation, and startup
self-tests verify the delete-one coefficient-change identity for HC3. The fitted
covariance matrix, design, residuals, leverage and conventional Cook distances
are exported. Cook distances are descriptive diagnostics, not robust tests or
exclusion criteria. No automatic normality or influence screening changes the model.

## Sensitivities and missingness

Native-family models: {int(native.status.eq('ESTIMATED').sum())} estimated,
{int(native.status.ne('ESTIMATED').sum())} not estimable. All variants use the same
aggregation/model rule, with their own supported pair phases and complete sessions.
Membership changes are explicit in the membership and session tables. Duration
variants can change the time periods and the operational estimand.

Common-pair-cohort contrasts additionally hold pair identities fixed. For
same-duration variants the matched-core analysis holds pair identities and scoring
cores fixed. Its change CI is fitted directly to the paired change in session
outcomes, preserving dependence between baseline and variant estimates. It is not
the difference between two confidence-interval endpoints. Matched-core rows:
{len(matched)}. Timing shifts/drifts are illustrative scenarios, not measured clock
uncertainty bounds or fitted corrections.

Leave-one-session-out models report influence without changing the primary
cohort. Coverage adjustment is {coverage['status']}: {coverage.get('note','')}.
This exploratory extra covariate cannot repair missing-not-at-random data and
could condition on a consequence of condition. Missing periods are not imputed.
The observed-data target does not represent the full interaction without further
assumptions about missingness. Coverage and eligibility counts are exported.

There is one primary test. Secondary intervals are pointwise, unadjusted robustness
descriptions, not independent discoveries. No significance stars, lowest-p-value
selection, condition-label permutations or synthetic-human reference tests are used.
Existing positive PLV is not itself evidence of coupling; individual spectra,
autocorrelation, common task effects and phase-definition limitations remain.

## Reproducibility and output locations

All nine Script07 output hashes were verified before analysis and checked again
before completion. All verified Script07 files, its completion status, and the
executed Script10 source are copied under `source_snapshot/` for reproducibility.
Input source paths are recorded, but no new scientific
outputs are written to the input directory or the user's home directory.
All tables, figures, reports and plotting cache reside inside the selected output
folder. A completion manifest is written last. A failed run is not complete.

`10_primary_coefficients.csv` and `10_primary_result.json` hold the primary result.
`10_primary_sessions.csv` gives the 16 observed session contrasts and covariates.
`10_primary_covariance_HC3.csv` gives the covariance matrix.
`10_native_family_sensitivity.csv`, `10_common_pair_cohort_sensitivity.csv` and
`10_matched_core_sensitivity.csv` distinguish support changes from paired estimator
changes. `10_sensitivity_membership.csv` records family-specific membership.
`10_sensitivity_sessions.csv` gives all native-family session contrasts.
`10_leave_one_session_out.csv`, `10_session_diagnostics.csv` and coverage files
record influence/selection diagnostics. Figures are PNG 300 dpi, vector PDF and SVG.
Captions and plot-data tables distinguish raw session differences from adjusted
model estimates; no overlapping window is drawn as an independent case.

Earlier exploratory outcomes were inspected. Script07 froze this analysis before
this condition comparison; this is not preregistration. A later method revision
requires a new version and explicit documentation rather than replacing this run.

## References

https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLSResults.HC3_se.html
https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.RegressionResults.get_robustcov_results.html
https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.t.html
https://doi.org/10.1016/j.neuroimage.2013.02.008
https://doi.org/10.1038/s41598-023-30405-5
These sources do not validate this study's specific 200 s choice, clock assumption
or physiological interpretation. The operational choices are recorded in Script07.
'''


def run(args):
    root,out=args.script07_dir,args.output_dir
    require(not out.exists(),f'Output folder exists: {out}. Preserve it and choose a NEW --output-dir.')
    require(not (out==root or root in out.parents or out in root.parents),'Input and output folders must be separate, not nested.')
    # Never create a missing external volume under /Volumes after disconnection.
    if str(out).startswith('/Volumes/'):
        volume=Path('/Volumes')/out.parts[2]
        require(volume.is_mount(),f'External volume is not mounted: {volume}. Reconnect it before running.')
    tests=self_tests();print('Statistical self-tests passed.',flush=True)
    status,spec,reg,cohort,sessions,families,matched=read_inputs(root)
    print('Verified all Script07 files and the frozen 16-session design.',flush=True)
    model=fit_sessions(sessions)
    primary=model['coefficients'].copy()
    primary['role']=['PRIMARY_CONDITION_CONTRAST','NUISANCE_RECIPE_ADJUSTMENT','NUISANCE_ORDER_ADJUSTMENT']
    primary['inference']='HC3; two-sided t; approximate; conditional timing'
    primary_result=primary.iloc[0].to_dict()
    counts={'candidate_pair_phases':len(reg),'scored_pair_phases':len(cohort),'primary_sessions':len(sessions),
            'primary_pair_phases':int(sessions.emotional_pair_phases.sum()+sessions.neutral_pair_phases.sum()),
            'dependent_windows':int(cohort.scored_windows.sum())}
    native,common,mc,membership,sens_sessions=sensitivity_analysis(reg,cohort,families,matched,sessions)
    print('All method, timing-scenario and matched-core sensitivity models assessed.',flush=True)
    loo=[]
    for sid in sessions.Session_ID:
        subset=sessions.loc[sessions.Session_ID.ne(sid)]
        fitted=fit_sessions(subset);q=fitted['coefficients'].iloc[0]
        loo.append({'session_omitted':sid,'sessions_remaining':len(subset),'estimate':q.estimate,
                    'HC3_SE':q.HC3_SE,'CI95_low':q.CI95_low,'CI95_high':q.CI95_high,
                    'change_from_primary':q.estimate-model['beta'][0]})
    loo=pd.DataFrame(loo)
    try:
        cm=fit_sessions(sessions,extra_coverage=True)
        coverage={'status':'ESTIMATED','note':'Exploratory additional continuous coverage-imbalance covariate; not a missing-data correction.'}
        cover_coeff=cm['coefficients'].drop(columns=['two_sided_p']).copy()
    except ValueError as exc:
        coverage={'status':'NOT_ESTIMABLE','note':str(exc)};cover_coeff=pd.DataFrame(columns=['term','estimate','HC3_SE','CI95_low','CI95_high'])
    diagnostic=sessions[['Session_ID']].copy()
    for col,key in [('fitted','fitted'),('residual','residual'),('leverage','leverage'),('standardized_residual','standardized_residual'),('cooks_distance','cooks_distance')]:
        diagnostic[col]=model[key]
    cv=pd.DataFrame(model['covariance'],index=MODEL_COLS,columns=MODEL_COLS).rename_axis('term').reset_index()
    paircov=cohort[['pair_id','Session_ID','Condition','Story','Recipe','Phase','shared_cooking_s','scored_core_union_s','coverage_fraction','mean_local_PLV','primary_condition_comparison_member']]
    # All analysis checks above precede creating the new output folder.
    out.mkdir(parents=True,exist_ok=False)
    (out/'source_snapshot').mkdir()
    for name in list(status['output_sha256'])+['07_RUN_STATUS.json']:
        shutil.copyfile(root/name,out/'source_snapshot'/name)
    shutil.copyfile(Path(__file__),out/'source_snapshot'/Path(__file__).name)
    for name,data in [('10_primary_coefficients.csv',primary),('10_primary_covariance_HC3.csv',cv),
        ('10_primary_sessions.csv',sessions),('10_session_diagnostics.csv',diagnostic),
        ('10_native_family_sensitivity.csv',native),('10_common_pair_cohort_sensitivity.csv',common),
        ('10_matched_core_sensitivity.csv',mc),('10_sensitivity_membership.csv',membership),
        ('10_sensitivity_sessions.csv',sens_sessions),('10_leave_one_session_out.csv',loo),
        ('10_coverage_adjusted_coefficients.csv',cover_coeff),('10_pair_phase_coverage.csv',paircov)]:
        cout(out/name,data)
    shutil.copyfile(root/'07_availability_by_design.csv',out/'10_availability_by_design.csv')
    jout(out/'10_coverage_adjustment_status.json',coverage)
    result={'status':'CONDITIONAL_ADJUSTED_ASSOCIATION_ESTIMATED','primary':primary_result,**counts,
            'physical_clock_verified':False,'causal_effect_established':False,'above_chance_coupling_tested':False,
            'design_rank':model['p'],'residual_df':model['df'],'max_leverage':float(model['leverage'].max()),
            'condition_number':model['condition_number'],'self_tests':tests,
            'sensitivity_family_models_estimated':int(native.status.eq('ESTIMATED').sum()),
            'sensitivity_family_models_not_estimable':int(native.status.ne('ESTIMATED').sum())}
    jout(out/'10_primary_result.json',result)
    textout(out/'10_RESULTS_AND_METHODS.md',report_text(result,primary,native,mc,coverage,loo,counts))
    mversion=figures(out,cohort,sessions,model,native,mc,loo,diagnostic,reg)
    links=''.join(f'<li><a href="{html.escape(str(p.relative_to(out)))}">{html.escape(p.stem)}</a></li>' for p in sorted((out/'figures').glob('*.pdf')))
    textout(out/'10_START_HERE.html','<!doctype html><html lang="en"><meta charset="utf-8"><title>KITCO3 cardiac results</title><style>body{font:17px system-ui;max-width:900px;margin:40px auto;padding:20px;line-height:1.6}h1{font-size:28px}a{color:#0072B2}</style><h1>KITCO3 cardiac condition analysis</h1><p>Conditional on the unverified exported-time alignment assumption. One adjusted condition comparison across 16 sessions.</p><p><a href="10_RESULTS_AND_METHODS.md">Full results and methods</a> | <a href="10_primary_coefficients.csv">Primary coefficient table</a> | <a href="10_primary_result.json">Primary result and provenance</a></p><h2>Figures</h2><ul>'+links+'</ul><p>Secondary intervals describe sensitivity; they are not additional confirmatory discoveries. No movement analysis or absolute above-chance coupling test was performed.</p></html>')
    # Detect input edits during a run before publishing a valid completion.
    require(sha(root/'07_RUN_STATUS.json')==sha(out/'source_snapshot/07_RUN_STATUS.json'),'Input status changed during analysis.')
    for name,h in status['output_sha256'].items():
        require(sha(root/name)==h,'Input changed during analysis: '+name)
    check_final_figure_files(out/'figures')
    payload={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file() and 'runtime_cache' not in p.relative_to(out).parts}
    completion={'script':Path(__file__).name,'version':VERSION,'script_sha256':sha(Path(__file__)),
        'status':'CONDITIONAL_CARDIAC_CONDITION_ANALYSIS_COMPLETE',
        'generated_utc':datetime.now(timezone.utc).isoformat(),'input_dir':str(root),'output_dir':str(out),
        'source_script07_status_sha256':sha(root/'07_RUN_STATUS.json'),'source_output_sha256':status['output_sha256'],
        **counts,'physical_clock_verified':False,'movement_data_read':False,'ECG_QC_repeated':False,
        'PLV_recomputed':False,'condition_effect_computed':True,'preregistered':False,
        'inference':'Frozen session-contrast OLS; HC3 covariance; t reference df=13; approximate',
        'primary_estimate':primary_result['estimate'],'primary_CI95':[primary_result['CI95_low'],primary_result['CI95_high']],
        'primary_two_sided_p':primary_result['two_sided_p'],
        'self_tests':tests,'environment':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'matplotlib':mversion},
        'output_sha256':payload,'cache_policy':'Plotting cache is under output/runtime_cache; cache is not a scientific output and is not sealed.'}
    jout(out/'10_RUN_STATUS.json',completion)
    print(json.dumps(clean_json({k:v for k,v in completion.items() if k not in ['source_output_sha256','output_sha256']}),indent=2))
    print('Read: '+str(out/'10_RESULTS_AND_METHODS.md'))


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--project-root',type=Path,default=Path('/Volumes/One Touch/KITCO3_drive'))
    p.add_argument('--audit-dir',type=Path,help='PASS output from Script09 v1.1.0')
    p.add_argument('--output-dir',type=Path)
    p.add_argument('--self-test-only',action='store_true',help='Run numerical self-tests; no files are written.')
    args=p.parse_args()
    if args.self_test_only:
        print(json.dumps(self_tests(),indent=2));return 0
    if args.self_test_only:
        return 0
    audit=(args.audit_dir or args.project_root/'outputs/09_PLV_condition_audit_v1_1_0').expanduser().resolve()
    completion=jload(audit/'09_RUN_STATUS.json')
    require(completion.get('status')=='PASS_NATIVE_PIPELINE_AUDIT' and completion.get('version')=='1.1.0','Script09 native audit did not pass')
    for relative, expected in completion['output_sha256'].items():
        f=(audit/relative).resolve()
        require(f.is_relative_to(audit) and f.is_file() and sha(f)==expected,'Script09 output changed: '+relative)
    args.script07_dir=audit/'source_script07'
    args.output_dir=(args.output_dir or args.project_root/'outputs/10_PLV_condition_comparison_v1_1_0').expanduser().resolve()
    try:
        run(args)
    except Exception as exc:
        print(f'SCRIPT10 STOPPED: {exc}\nNo completion status is published for a failed run. Preserve existing files; do not overwrite outputs.',file=sys.stderr)
        return 1
    return 0


if __name__=='__main__':
    raise SystemExit(main())
