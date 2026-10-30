#!/usr/bin/env python3
"""Recover only combined exports from KITCO3 Script02 v1.1.0's saved records.

Requires Python 3.9+ and numpy, already used by Script02. No raw ECG, detection,
cleaning, peak correction or plotting is run. Original tables/records/plots stay
unchanged; existing aggregate exports are copied into a timestamped backup.
Outputs are staged and checked before installation. A recovery summary is
written last. An interrupted installation can be rerun if no summary exists.
This is specific to Emma's verified complete 185-recording KITCO3 cohort.
"""
import argparse
import ast
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
import numpy as np

VERSION = '1.0.0'
COHORT_SHA256 = '983ebe46df2bead32ca0eebd1a4ec6f5d48970a27afb05d5a98ed85f1c8fe152'
SCRIPT02 = '02_KITCO3_ECG_processing_QC_v1_1_0_EG.py'
TABLES = ['02_recording_qc.csv', '02_participant_phase_candidates.csv',
          '02_segment_qc.csv', '02_processing_issues.csv', '02_input_manifest_snapshot.csv']
COMMON = ['recording_uid', 'Session_ID', 'PID', 'Phase', 'Condition', 'sensor_short', 'attempt_uid']

def require(test, message):
    if not test:
        raise ValueError(message)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def rows(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))

def checked_path(base, relative):
    p = (base/relative).resolve()
    require(p.is_relative_to(base), 'Path outside output directory: '+relative)
    require(p.is_file(), 'Missing file: '+str(p))
    return p

def nullable(value):
    return '' if not np.isfinite(value) else repr(float(value))

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output-dir', type=Path, required=True, help='Existing interrupted Script02 output directory')
    ap.add_argument('--processing-script', type=Path, required=True, help='Exact Script02 v1.1.0 file used for the run')
    a = ap.parse_args()
    base = a.output_dir.expanduser().resolve()
    script = a.processing_script.expanduser().resolve()
    require(base.is_dir(), 'Output directory does not exist')
    require(not (base/'02_run_summary.json').exists(),
            'A run summary already exists. Refusing to overwrite a completed/recovered run.')
    require(script.is_file(), 'Processing script not found')
    constants = {}
    for node in ast.parse(script.read_text()).body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    constants[target.id] = node.value.value
    require(constants.get('SCRIPT_VERSION') == '1.1.0' and constants.get('SCRIPT_NAME') == SCRIPT02,
            'Requires the original Script02 v1.1.0 file, not an older/revised version')
    input_hashes = {name: sha(checked_path(base, name)) for name in TABLES}
    qc, candidates, segments, issues, snapshot = [rows(base/n) for n in TABLES]
    uids = [r['recording_uid'] for r in qc]
    require(len(uids) == len(set(uids)) == 185, 'Expected 185 unique recording results')
    cohort = hashlib.sha256(('\n'.join(sorted(uids))+'\n').encode()).hexdigest()
    require(cohort == COHORT_SHA256, 'Recording cohort differs from the audited 185-recording run')
    for label, tab in [('candidates', candidates), ('snapshot', snapshot)]:
        ids = [r['recording_uid'] for r in tab]
        require(len(ids) == 185 and set(ids) == set(uids), label+' IDs differ from recording QC')
    actual_dirs = {p.name for p in (base/'records').iterdir() if p.is_dir() and not p.name.startswith('.')}
    require(actual_dirs == set(uids), 'Missing or unexpected recording folders')
    require(all(r['recording_uid'] in set(uids) for r in segments+issues), 'Orphan segment/issue')
    errors = Counter(r['recording_uid'] for r in issues if r['severity'] == 'ERROR')
    segment_counts = Counter()
    for s in segments:
        segment_counts[s['recording_uid']] += int(s['n_rpeaks_corrected'])
    results = []
    source_hashes = {}
    for q in qc:
        uid = q['recording_uid']
        jp = checked_path(base, 'records/'+uid+'/02_result.json')
        result = json.loads(jp.read_text())
        require(result['recording_uid'] == uid, 'JSON recording identity mismatch')
        for k in COMMON + ['processing_status', 'n_rpeaks_original', 'n_rpeaks_corrected']:
            require(str(result[k]) == q[k], uid+': JSON/table mismatch in '+k)
        require(errors[uid] == int(q['n_error_issues']), uid+': error counts disagree')
        require(segment_counts[uid] == int(q['n_rpeaks_corrected']), uid+': segment counts disagree')
        require(result.get('detector_rate_policy') == 'legacy_replay', 'Not a legacy-replay run')
        bp = checked_path(base, result['beat_npz_path'])
        require(bp == (base/'records'/uid/'02_beats.npz').resolve(), 'Unexpected beat-file path')
        source_hashes[str(jp.relative_to(base))] = sha(jp)
        source_hashes[str(bp.relative_to(base))] = sha(bp)
        results.append(result)
    # Match original Script02 output order, including integer PID/Phase ordering.
    results.sort(key=lambda r: (r['Session_ID'], int(r['Phase']), int(r['PID']), r['attempt_uid'], r['sensor_short']))
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    stage = Path(tempfile.mkdtemp(prefix='02_export_recovery_stage_', dir=base))
    totals = Counter()
    try:
        handles = {t:gzip.open(stage/('02_original_rpeaks.csv.gz' if t=='original' else '02_corrected_beats.csv.gz'),
                               'wt', encoding='utf-8', newline='') for t in ['original','corrected']}
        try:
            writers = {t:csv.writer(f) for t,f in handles.items()}
            for t,w in writers.items():
                w.writerow(COMMON + ['segment_id', 'peak_index_recording' if t=='original' else 'beat_index_recording',
                                     'rpeak_valid_index', 'rpeak_source_row', 'rpeak_timestamp', 'rr_ms', 'instantaneous_hr_bpm'])
            for number,r in enumerate(results,1):
                uid = r['recording_uid']
                # Materialize each compressed member once, avoiding the original O(beats^2) decompression.
                with np.load(base/r['beat_npz_path'], allow_pickle=False) as f:
                    arrays = {k:f[k] for k in f.files}
                for t in ['original','corrected']:
                    keys = [t+'_rpeak_'+k for k in ['timestamp','segment_id','valid_index','source_row']]+[t+'_rr_ms',t+'_hr_bpm']
                    ts,sid,indices,source,rr,hr = [arrays[k] for k in keys]
                    n = len(ts)
                    require(all(v.ndim==1 and len(v)==n for v in [ts,sid,indices,source,rr,hr]), uid+': invalid array lengths')
                    require(n == int(r['n_rpeaks_'+t]), uid+': peak count mismatch')
                    require(np.isfinite(ts).all() and (sid>0).all(), uid+': invalid timestamps/segment IDs')
                    require(all(np.issubdtype(v.dtype,np.integer) for v in [sid,indices,source]), uid+': noninteger indices')
                    expected = np.full(n, np.nan)
                    if n>1:
                        same = sid[1:]==sid[:-1]
                        require((np.diff(ts)[same]>0).all(), uid+': nonincreasing peaks within segment')
                        expected[1:] = np.where(same, np.diff(ts)*1000, np.nan)
                    require(np.allclose(rr, expected, rtol=1e-12, atol=1e-7, equal_nan=True), uid+': RR mismatch/gap bridging')
                    require(np.allclose(hr, 60000/rr, rtol=1e-12, atol=1e-7, equal_nan=True), uid+': HR mismatch')
                    w=writers[t]
                    for i in range(n):
                        w.writerow([r.get(k,'') for k in COMMON]+[int(sid[i]),i+1,int(indices[i]),int(source[i]),
                                    repr(float(ts[i])),nullable(rr[i]),nullable(hr[i])])
                    totals[t] += n
                if number==1 or number%10==0 or number==len(results):
                    print(f'Exported {number}/{len(results)} recordings', flush=True)
        finally:
            for f in handles.values():
                f.close()
        output_hashes = {}
        for t,name in [('original','02_original_rpeaks.csv.gz'),('corrected','02_corrected_beats.csv.gz')]:
            # Reading through EOF verifies gzip CRC/trailer and the exact number of serialized rows.
            with gzip.open(stage/name,'rt',encoding='utf-8',newline='') as f:
                reader=csv.reader(f); next(reader)
                count=sum(1 for _ in reader)
            require(count==totals[t], 'Export row-count mismatch: '+name)
            output_hashes[name]=sha(stage/name)
        # Detect changes during recovery before installing output.
        for name,h in {**input_hashes,**source_hashes}.items():
            require(sha(base/name)==h,'Input changed during recovery: '+name)
        backup=base/('02_export_backup_'+stamp);backup.mkdir()
        for name in output_hashes:
            if (base/name).exists():
                shutil.copy2(base/name,backup/name)
            os.replace(stage/name,base/name)
        phases={(r['Session_ID'],int(r['PID']),int(r['Phase'])) for r in results}
        status=dict(Counter(r['processing_status'] for r in results))
        summary={
            'script':SCRIPT02,'script_version':'1.1.0',
            'summary_type':'RECOVERED_FROM_SAVED_RECORD_RESULTS_NOT_ORIGINAL_RUN_SUMMARY',
            'generated_utc':datetime.now(timezone.utc).isoformat(),
            'runtime_seconds':None,
            'configuration':{'sampling_rate_policy':'legacy_replay',
                'detector_rate_is_physical_calibration':False,
                'only_restarts':False,'max_recordings':None,'pid_filter':[], 'phase_filter':[], 'session_id_filter':[],
                'filter_fields_basis':'Inferred full cohort from exact audited 185-UID set and matching manifest/candidate tables'},
            'input':{'candidate_sensor_recordings_this_run':len(results),
                     'candidate_participant_phases_this_run':len(phases)},
            'results':{'n_recordings':len(results),'recording_status_counts':status,
                       'n_processing_error_recordings':sum(int(q['n_error_issues'])>0 for q in qc),
                       'original_peak_rows':totals['original'],'corrected_beat_rows':totals['corrected']},
            'recovery':{'script':Path(__file__).name,'version':VERSION,'script_sha256':sha(Path(__file__)),
                        'supplied_processing_script_sha256':sha(script),
                        'supplied_processing_script_hash_proves_original_execution':False,
                        'original_run_summary_available':False,'original_run_completion_time':None,
                        'original_full_runtime_and_environment':None,
                        'ECG_processing_rerun':False,'plots_regenerated':False,
                        'plots_inspected_by_recovery':False,
                        'source_result_generated_utc_min':min(r['generated_utc'] for r in results),
                        'source_result_generated_utc_max':max(r['generated_utc'] for r in results),
                        'input_table_sha256':input_hashes,'record_source_sha256':source_hashes,
                        'output_sha256':output_hashes,'backup_directory':str(backup),
                        'environment':{'python':platform.python_version(),'numpy':np.__version__},
                        'limitations':['Reconstructs exports, not original execution provenance.',
                                      'No new ECG quality review or physical timing validation.']}}
        text=f"KITCO3 Script02 export recovery {VERSION}\nRecordings: {len(results)}\nOriginal peak rows: {totals['original']}\nCorrected beat rows: {totals['corrected']}\nStatus: EXPORT_RECOVERY_COMPLETE\nECG detection and plots were not rerun.\nOriginal run completion time/environment are unknown.\nSee JSON for validation, hashes and recovery provenance.\n"
        (stage/'02_run_summary.txt').write_text(text,encoding='utf-8')
        (stage/'02_run_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        require(not (base/'02_run_summary.json').exists(),'Another process published a summary during recovery')
        if (base/'02_run_summary.txt').exists():
            shutil.copy2(base/'02_run_summary.txt',backup/'02_run_summary.txt')
        os.replace(stage/'02_run_summary.txt',base/'02_run_summary.txt')
        os.replace(stage/'02_run_summary.json',base/'02_run_summary.json')
        print(text,flush=True)
        print('Existing incomplete exports backed up to: '+str(backup),flush=True)
    finally:
        shutil.rmtree(stage,ignore_errors=True)

if __name__=='__main__':
    try:
        main()
    except (ValueError,OSError,KeyError,EOFError) as error:
        print('RECOVERY STOPPED: '+str(error),file=sys.stderr)
        sys.exit(2)
