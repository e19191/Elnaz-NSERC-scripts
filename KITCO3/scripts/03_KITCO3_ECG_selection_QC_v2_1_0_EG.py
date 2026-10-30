#!/usr/bin/env python3
"""KITCO3 Script 03 v2.0.0: technical selection plus documented ECG QC.

Replays the reviewed, source-specific decisions. It does not automate visual
judgment or certify physical synchronization. See README and QC_STEP_MAP.md.
Requires the accompanying versioned support/ and inputs/ directories.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
import pandas as pd

VERSION = '2.1.0'
ROOT = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_assets(root=ROOT):
    seal = json.loads((root/'ASSET_SHA256.json').read_text())
    for relative, expected in seal.items():
        p = (root/relative).resolve()
        require(p.is_relative_to(root.resolve()), 'Unsafe asset path')
        require(p.is_file() and sha(p) == expected, f'Asset missing or changed: {relative}')
    return len(seal)


def read_selection(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    keys = [(r['Session_ID'], int(r['PID']), int(r['Phase'])) for r in rows]
    require(len(keys) == len(set(keys)), 'Duplicate participant-phase selection')
    require(len(rows) == 173, 'This reviewed release requires the original 173-row selection')
    return rows


def validate_selection(actual, frozen):
    a = {(r['Session_ID'], int(r['PID']), int(r['Phase'])):r for r in actual}
    b = {(r['Session_ID'], int(r['PID']), int(r['Phase'])):r for r in frozen}
    require(set(a) == set(b), 'Selection participant-phase set changed; new review required')
    # Compare scientific/selection content, ignoring only machine locations.
    ignored = {'script02_dir', 'selected_beat_npz_path'}
    numeric = {'cooking_start_timestamp', 'cooking_stop_timestamp',
               'selected_processed_window_coverage_s'}
    for k in a:
        require(set(a[k]) == set(b[k]), f'Selection schema differs at {k}')
        for field in set(b[k])-ignored:
            x, y = a[k][field], b[k][field]
            if field in numeric and x and y:
                require(abs(float(x)-float(y)) <= 1e-6, f'Selection changed: {k} {field}')
            else:
                require(x == y, f'Selection changed: {k} {field}')


def validate_review_scope(selection, qcroot):
    rec = pd.read_csv(qcroot/'inputs/recordings_142.csv', keep_default_na=False)
    source = pd.read_csv(ROOT/'inputs/source_acquisition_manifest_185.csv', keep_default_na=False)
    sheet = pd.read_excel(ROOT/'inputs/Runsheet_final_EG.xlsx', sheet_name='Runsheet_Clean', keep_default_na=False)
    require(len(rec)==142 and rec.recording_uid.is_unique, 'Invalid reviewed recording set')
    require(len(sheet)==216 and not sheet.duplicated(['Session_ID','PID','Phase']).any(), 'Invalid runsheet')
    require(set(sheet.Condition)=={'NA','NB','EA','EB'}, 'Condition codes changed, including literal NA')
    sm = {(r['Session_ID'], int(r['PID']), int(r['Phase'])):r for r in selection}
    rs = sheet.set_index(['Session_ID','PID','Phase'])
    manifest = source.set_index('recording_uid')
    for r in rec.itertuples(index=False):
        k=(r.Session_ID, int(r.PID), int(r.Phase));s=sm[k];m=manifest.loc[r.recording_uid]
        require(s['selected_recording_uid']==r.recording_uid, f'Review/selection identity mismatch: {k}')
        require(s['selected_attempt_uid']==r.attempt_uid, f'Attempt mismatch: {k}')
        require(s['selection_status']=='SELECTED_FOR_DYAD_WINDOW_QC', f'Unexpected selection status: {k}')
        for field in ['cooking_start_timestamp','cooking_stop_timestamp']:
            require(abs(float(s[field])-getattr(r,field))<1e-6, f'Cooking interval mismatch: {k}')
        q=rs.loc[k]
        require(s['Condition']==q.Condition, f'Condition mismatch: {k}')
        require(int(s['Analysis_Partner'])==int(q.Analysis_Partner), f'Actual partner mismatch: {k}')
        require(int(m.sensor_short)==int(q.Sensor), f'Sensor mismatch: {k}')
    for pair,g in rec.groupby('pair_id'):
        require(len(g)==2 and g.attempt_uid.nunique()==1, f'Pair source mismatch: {pair}')
        require(g.cooking_start_timestamp.nunique()==g.cooking_stop_timestamp.nunique()==1, f'Pair interval mismatch: {pair}')
        a,b=list(g.itertuples(index=False))
        qa=rs.loc[(a.Session_ID,a.PID,a.Phase)];qb=rs.loc[(b.Session_ID,b.PID,b.Phase)]
        require(int(qa.Analysis_Partner)==b.PID and int(qb.Analysis_Partner)==a.PID, f'Nonreciprocal actual partners: {pair}')
        require(qa.Condition==qb.Condition, f'Pair condition mismatch: {pair}')
        ma,mb=manifest.loc[a.recording_uid],manifest.loc[b.recording_uid]
        require(ma.attempt_path==mb.attempt_path and ma.collection_ipad_label==mb.collection_ipad_label, f'Pair acquisition mismatch: {pair}')
    return rec,sheet


def run_python(script, *args):
    # Historical reviewed modules use assertions, so never disable them.
    env=dict(os.environ);env.pop('PYTHONOPTIMIZE',None)
    subprocess.run([sys.executable,str(script),*map(str,args)],check=True,env=env)


def verify_detector_compatibility(script02, rec, qcroot, out):
    rows=[]
    for r in rec.itertuples(index=False):
        p=script02/'records'/r.recording_uid/'02_beats.npz'
        require(p.is_file(),f'Missing reviewed recording in Script02 output: {r.recording_uid}')
        with np.load(p,allow_pickle=False) as a, np.load(qcroot/'inputs/original_peaks'/f'{r.recording_uid}.npz') as b:
            require('original_rpeak_valid_index' in a.files,'Original detector indices required for compatibility')
            x=a['original_rpeak_valid_index'];y=b['original_peak_index']
            if not np.array_equal(x,y):
                rows.append(dict(recording_uid=r.recording_uid,previous_count=len(y),new_count=len(x),
                    shared_count=len(np.intersect1d(x,y)),status='REVIEW_REQUIRED_DETECTOR_CHANGED'))
    if rows:
        pd.DataFrame(rows).to_csv(out/'03_REVIEW_REQUIRED_CHANGED_DETECTIONS.csv',index=False)
        raise ValueError('Original detections changed. Prior edits are NOT applied; inspect the review-required queue.')


def build_handoff(out, selection, rec, sheet):
    historical=out/'historical_qc/results'
    data=out/'dataset';data.mkdir()
    recordings=pd.read_csv(historical/'recording_qc_142.csv',keep_default_na=False)
    pairs=pd.read_csv(historical/'pair_qc_71.csv',keep_default_na=False)
    # Window-specific historical fields intentionally excluded from primary QC.
    keep=['pair_id','Session_ID','Phase','PID_1','PID_2','recording_uid_1','recording_uid_2',
          'duration_s','joint_trusted_ibi_s','joint_trusted_ibi_fraction','longest_joint_ibi_span_s']
    clean=pairs[keep].copy()
    clean['ECG_support_status']=np.where(clean.joint_trusted_ibi_s>0,
        'SOME_JOINT_REVIEWED_IBI_SUPPORT','NO_JOINT_REVIEWED_IBI_SUPPORT')
    clean['physical_clock_status']='UNVERIFIED_SENSOR_TO_EXPORT_MAPPING'
    clean['final_synchrony_inclusion']='NOT_DECIDED_BY_SCRIPT03'
    clean.to_csv(data/'03_pair_ECG_QC_71.csv',index=False,float_format='%.9f')
    recordings['ECG_support_status']=np.where(recordings.trusted_ibi_s>0,
        'SOME_REVIEWED_IBI_SUPPORT','NO_REVIEWED_COOKING_IBI_SUPPORT')
    recordings['physical_clock_status']='UNVERIFIED_SENSOR_TO_EXPORT_MAPPING'
    recordings.to_csv(data/'03_recording_ECG_QC_142.csv',index=False,float_format='%.9f')
    for name in ['reviewed_peak_times.csv.gz','reviewed_nonbridging_ibi.csv.gz',
                 'participant_usable_intervals.csv','pair_joint_usable_intervals.csv',
                 'new_peak_edits.csv','sample_anchored_withheld_regions.csv']:
        shutil.copy2(historical/name,data/('03_'+name))
    sm={(r['Session_ID'],int(r['PID']),int(r['Phase'])):r for r in selection}
    qm=recordings.set_index('recording_uid');rows=[]
    for r in sheet.itertuples(index=False):
        s=sm.get((r.Session_ID,int(r.PID),int(r.Phase)))
        uid=s['selected_recording_uid'] if s else ''
        q=qm.loc[uid] if uid in qm.index else None
        rows.append(dict(Session_ID=r.Session_ID,PID=int(r.PID),Phase=int(r.Phase),
            Condition=r.Condition,Analysis_Partner=r.Analysis_Partner,Sensor=r.Sensor,
            Notes=r.Notes,selection_status=s['selection_status'] if s else 'NOT_IN_SCRIPT02_CANDIDATE_SELECTION',
            selected_recording_uid=uid,
            ECG_review_status=q.review_status if q is not None else 'NOT_REVIEWED_IN_142_RECORD_PAIRED_SCOPE',
            trusted_IBI_s=float(q.trusted_ibi_s) if q is not None else np.nan,
            physical_clock_status='UNVERIFIED_SENSOR_TO_EXPORT_MAPPING',
            final_synchrony_inclusion='NOT_DECIDED_BY_SCRIPT03'))
    pd.DataFrame(rows).to_csv(data/'03_participant_phase_QC_216.csv',index=False,float_format='%.9f')
    require(len(rows)==216 and len(clean)==71 and len(recordings)==142,'Handoff row counts failed')
    require(int((recordings.trusted_ibi_s==0).sum())==11,'Unexpected no-support recording count')
    require(int((clean.joint_trusted_ibi_s>0).sum())==60,'Unexpected joint support count')
    hashes={p.name:sha(p) for p in sorted(data.iterdir())}
    handoff=dict(script_version=VERSION,ECG_review_decisions_replayed=True,
        review_scope='AI-assisted targeted waveform review, not independent human adjudication',
        participant_phases=216,selection_rows=173,reviewed_recordings=142,paired_phases=71,
        pairs_with_some_joint_IBI_support=60,records_without_retained_cooking_IBI=11,
        physical_alignment_verified=False,final_synchrony_dataset_frozen=False,
        window_rules_used_for_primary_ECG_exclusion=False,
        historical_window_diagnostics='historical_qc/results/paired_window_qc_217.csv: 200s core,40s context,100s step; NOT current synchrony inclusion',
        timestamp_status='Export axes preserved; no offset fitting, rescaling or clock repair',
        sampling_rate_status='128 header label; 125 legacy detector/display parameter; physical sample-time mapping unresolved',
        remaining_steps=['Script04 timing verification or explicit qualified assumption',
                         'Finalize synchrony signal, estimator, windows and coverage',
                         'Freeze analysis inclusion and statistical plan'],
        output_sha256=hashes)
    (out/'03_STEP04_HANDOFF.json').write_text(json.dumps(handoff,indent=2)+'\n')
    return handoff


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root',type=Path,default=Path('/Volumes/One Touch/KITCO3_drive'))
    sources=p.add_mutually_exclusive_group()
    sources.add_argument('--sources-root',type=Path,help='Optional ZIP replay instead of the original data directory.')
    sources.add_argument('--raw-data-root',type=Path,help='Original raw data root. Default: project-root/data.')
    p.add_argument('--output-dir',type=Path,help='NEW directory. Default: project-root/outputs/03_ecg_QC_v2_1_0.')
    p.add_argument('--script02-dir',type=Path,help='Optional complete real Script02 v1.0.5 production output. Reruns original selection and checks it against reviewed identities.')
    p.add_argument('--regenerate-review-plots',action='store_true',help='Regenerate full overviews, flagged epochs and accepted-edit verification plots. This is NOT fresh visual adjudication.')
    a=p.parse_args(argv)
    project=a.project_root.expanduser().resolve()
    rawdir=(a.raw_data_root or project/'data').expanduser().resolve() if not a.sources_root else None
    if rawdir:
        require(rawdir.is_dir(),f'Raw data directory missing: {rawdir}')
        os.environ['KITCO3_REVIEW_RAW_DATA_ROOT']=str(rawdir)
    else:
        os.environ.pop('KITCO3_REVIEW_RAW_DATA_ROOT',None)
    source_root=rawdir if rawdir else a.sources_root.resolve()
    require(not sys.flags.optimize,'Do not run this pipeline with python -O')
    assets=verify_assets();qcroot=ROOT/'support/ecg_review'
    frozen=read_selection(ROOT/'inputs/selection_173.csv')
    rec,sheet=validate_review_scope(frozen,qcroot)
    out=(a.output_dir or project/'outputs/03_ecg_QC_v2_1_0').resolve();out.mkdir(parents=True,exist_ok=False)
    selection=frozen;selection_mode='VERIFIED_SAVED_SELECTION_REPLAY'
    if a.script02_dir:
        verify_detector_compatibility(a.script02_dir.resolve(),rec,qcroot,out)
        run_python(ROOT/'support/03_KITCO3_ECG_selection_QC_v1_0_1_EG.py',
                   '--script02-dir',a.script02_dir.resolve(),'--output-dir',out/'selection')
        selection=read_selection(out/'selection/03_participant_phase_selection.csv')
        validate_selection(selection,frozen)
        selection_mode='ORIGINAL_SELECTION_RERUN_AND_MATCHED_TO_REVIEWED_SNAPSHOT'
    else:
        (out/'selection').mkdir()
        shutil.copy2(ROOT/'inputs/selection_173.csv',out/'selection/03_participant_phase_selection.csv')
    # Exact historical rebuild includes nonbridging RR validation and a second replay.
    run_python(qcroot/'scripts/run_qc.py','--sources-root',source_root,'--run-dir',out/'historical_qc')
    results=out/'historical_qc/results'
    baseline=json.loads((qcroot/'results/OUTPUT_SHA256.json').read_text())
    require(all(sha(results/n)==h for n,h in baseline.items()),'QC no longer matches reviewed baseline')
    # The baseline seal itself must also match; these are 11 exact result files.
    require(sha(results/'OUTPUT_SHA256.json')==sha(qcroot/'results/OUTPUT_SHA256.json'),'QC output seal differs')
    timing=out/'timing_diagnostics';(timing/'inputs').mkdir(parents=True)
    shutil.copy2(qcroot/'inputs/recordings_142.csv',timing/'inputs/recordings_142.csv')
    shutil.copy2(ROOT/'inputs/source_acquisition_manifest_185.csv',timing/'inputs/source_acquisition_manifest_185.csv')
    run_python(ROOT/'support/01_raw_timing_audit.py','--root',timing,'--raw-root',source_root)
    if a.regenerate_review_plots:
        plot=qcroot/'scripts/01_plot_review.py';work=out/'historical_qc/work'
        common=['--inputs',qcroot/'inputs','--work',work]
        run_python(plot,*common,'--mode','overview','--output',out/'regenerated_review/overview')
        run_python(plot,*common,'--mode','events','--epochs',qcroot/'decisions/review_epochs/review_epochs.csv','--output',out/'regenerated_review/flagged')
        run_python(plot,*common,'--mode','events','--epochs',qcroot/'decisions/review_epochs/final_correction_epochs.csv',
                   '--decisions',qcroot/'decisions/waveform_decisions.json','--output',out/'regenerated_review/accepted_edits')
    handoff=build_handoff(out,selection,rec,sheet)
    from presentation_figures import make_qc_figures
    make_qc_figures(out/'dataset',out/'figures')
    status=dict(script=Path(__file__).name,version=VERSION,status='ECG_QC_REPLAY_COMPLETE',
        assets_verified=assets,selection_mode=selection_mode,historical_result_files_identical=11,
        current_visual_review_performed=False,physical_alignment_verified=False,
        final_synchrony_dataset_frozen=False,PLV_computed=False,
        reviewer='Existing documented Codex AI-assisted decisions, replayed without alteration',
        handoff_sha256=sha(out/'03_STEP04_HANDOFF.json'))
    (out/'03_RUN_STATUS.json').write_text(json.dumps(status,indent=2)+'\n')
    print(json.dumps(status,indent=2),flush=True)
    return 0


if __name__=='__main__':
    try:
        raise SystemExit(main())
    except (ValueError,OSError,KeyError,AssertionError,subprocess.CalledProcessError) as error:
        print(f'Script03 v2 STOPPED: {error}. No completion status is published for a failed run.',file=sys.stderr)
        raise SystemExit(2)
