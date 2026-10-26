#!/usr/bin/env python3
"""Continue the specific Script03 v2.1.0 median/compression compatibility failure.

Version 1.0.0. Keeps original installed scripts and scientific QC decisions intact.
Requires the verified runtime package and the already prepared 142-record work
cache, selection, and generated historical results. It does not rerun source
preparation or ECG detection. It reruns validation, timing audit and requested
plots, then publishes the normal Script03 handoff/status with continuation
provenance. Do not run simultaneously with another Script03 process.

Compatibility changes apply only to a documented local copy of support code:
1. Explicit 11-point edge-padded median replaces SciPy median_filter, with
   unchanged residual thresholds and strict positive/finite IBI assertions.
2. If (and only if) decompressed CSV bytes equal the sealed reference, use
   reference gzip bytes. Thus all existing scientific-content and byte-exact
   replay checks remain enabled. Different data still stop the continuation.

Existing compressed bytes and failed verification are backed up. Original
result CSV payloads, raw caches, original installed code and QC decisions stay
unchanged. The backup's JSON records versions, source hashes and modifications.
"""
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import numpy as np
import pandas as pd
import scipy

VERSION = '1.0.0'

def require(condition, message):
    if not condition:
        raise ValueError(message)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()

def write_json(path, payload):
    temporary=path.with_suffix(path.suffix+'.staged')
    temporary.write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n')
    os.replace(temporary,path)

def explicit_median(values, size=11, mode='nearest'):
    require(size==11 and mode=='nearest','Unsupported median specification')
    v=np.array(values,dtype=np.float64,copy=True)
    require(v.ndim==1 and np.isfinite(v).all() and (v>0).all(),
            'Trusted IBI must be finite and positive')
    if not len(v):
        return v
    return np.median(np.lib.stride_tricks.sliding_window_view(
        np.pad(v,(5,5),mode='edge'),11),axis=1)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root',type=Path,default=Path('/Volumes/One Touch/KITCO3_drive'))
    parser.add_argument('--regenerate-review-plots',action='store_true')
    a=parser.parse_args()
    require(not sys.flags.optimize,'Do not use Python -O')
    project=a.project_root.expanduser().resolve()
    package=project/'scripts/KITCO3_pipeline_02_03_04_v1.0.0'
    out=project/'outputs/03_ecg_QC_v2_1_0'
    script02=project/'outputs/02_ecg_processing_v1_1_0'
    raw=project/'data'
    historical=out/'historical_qc';results=historical/'results';work=historical/'work'
    require(out.is_dir() and work.is_dir() and results.is_dir(),'Expected prepared failed Script03 output is missing')
    require(not (out/'03_RUN_STATUS.json').exists(),'Script03 already has a completion status; do not rerun continuation')
    require(raw.is_dir(),'Original data directory not found')
    sys.path.insert(0,str(package))
    path=package/'03_KITCO3_ECG_selection_QC_v2_1_0_EG.py'
    spec=importlib.util.spec_from_file_location('kitco3_original_script03',path)
    three=importlib.util.module_from_spec(spec);spec.loader.exec_module(three)
    require(three.VERSION=='2.1.0','Expected Script03 v2.1.0')
    assets=three.verify_assets()
    original=package/'support/ecg_review'
    baseline=json.loads((original/'results/OUTPUT_SHA256.json').read_text())
    current=json.loads((results/'OUTPUT_SHA256.json').read_text())
    require(set(current)==set(baseline),'Generated result manifest has different files')
    require({p.name for p in results.iterdir()}==set(baseline)|{'OUTPUT_SHA256.json'},'Unexpected/missing historical result files')
    require(all(sha(results/n)==h for n,h in current.items()),'Generated result manifest does not match its files')
    normalization=[]
    for name,expected in baseline.items():
        require(sha(original/'results'/name)==expected,'Reference hash mismatch: '+name)
        source=results/name;reference=original/'results'/name
        if name.endswith('.csv.gz'):
            payload=gzip.decompress(source.read_bytes())
            require(payload==gzip.decompress(reference.read_bytes()),'CSV data differ from reference: '+name)
            normalization.append({'file':name,'before_sha256':sha(source),
                                  'reference_sha256':expected,
                                  'identical_decompressed_csv_sha256':hashlib.sha256(payload).hexdigest()})
        else:
            require(sha(source)==expected,'Scientific result differs from reference: '+name)
    frozen=three.read_selection(package/'inputs/selection_173.csv')
    selection=three.read_selection(out/'selection/03_participant_phase_selection.csv')
    three.validate_selection(selection,frozen)
    rec,sheet=three.validate_review_scope(selection,original)
    # Repeat compatibility check only, never peak detection or selection.
    three.verify_detector_compatibility(script02,rec,original,out)
    for uid in rec.recording_uid:
        require((work/'raw_cache'/f'{uid}.npz').is_file(),'Missing prepared source cache: '+uid)
    # Do not skip residual QC: independently require the same thresholds using
    # a deterministic median on the existing published data before any mutation.
    ibi=pd.read_csv(results/'reviewed_nonbridging_ibi.csv.gz')
    for r in rec.itertuples(index=False):
        duration=r.cooking_stop_timestamp-r.cooking_start_timestamp
        good=ibi.loc[ibi.recording_uid.eq(r.recording_uid)&ibi.trusted&
                     ibi.midpoint_cooking_s.ge(0)&ibi.midpoint_cooking_s.lt(duration)]
        for _,s in good.groupby('ibi_segment_id'):
            v=s.ibi_s.to_numpy();median=explicit_median(v);ratio=v/median
            require(not ((ratio<.55)|(ratio>1.7)|(v<.3)|(v>2.0)).any(),
                    'Actual residual IBI cue: '+r.recording_uid+'; fresh waveform review required')
    continuation=out/'continuation_v1_0_0'
    require(not continuation.exists(),'Continuation folder already exists; preserve it and report its status before retrying')
    continuation.mkdir()
    backup=continuation/'backup';backup.mkdir()
    qcroot=continuation/'support_ecg_review'
    shutil.copytree(original,qcroot,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    validator=qcroot/'scripts/03_validate.py'
    builder=qcroot/'scripts/02_build_reviewed_dataset.py'
    # Strict source guards: patch only the verified release implementation.
    old='from scipy.ndimage import median_filter'
    text=validator.read_text();require(text.count(old)==1,'Unexpected validator source')
    function='''def median_filter(values, size=11, mode="nearest"):
    assert size == 11 and mode == "nearest"
    v = np.array(values, dtype=np.float64, copy=True)
    assert v.ndim == 1 and np.isfinite(v).all() and (v > 0).all(), "Invalid trusted IBI"
    if not len(v): return v
    return np.median(np.lib.stride_tricks.sliding_window_view(np.pad(v, (5, 5), mode="edge"), 11), axis=1)
'''
    validator.write_text(text.replace(old,function))
    old='    payload=gzip.compress(text,mtime=0)'
    text=builder.read_text();require(text.count(old)==1,'Unexpected builder source')
    replacement='''    reference = Path(__file__).resolve().parents[1]/"results"/path.name
    assert reference.is_file(), "Sealed reference gzip required"
    payload = reference.read_bytes()
    assert gzip.decompress(payload) == text, "Rebuilt CSV differs from sealed reference"
'''.rstrip()
    builder.write_text(text.replace(old,replacement))
    audit={'continuation_script':Path(__file__).name,'continuation_version':VERSION,
           'script_sha256':sha(Path(__file__)),'started_utc':datetime.now(timezone.utc).isoformat(),
           'status':'STARTED','source_assets_verified':assets,
           'original_script03_sha256':sha(path),
           'environment':{'python':platform.python_version(),'numpy':np.__version__,
                          'pandas':pd.__version__,'scipy':scipy.__version__},
           'gzip_normalization':normalization,
           'support_changes':{
               '03_validate.py':{'before_sha256':sha(original/'scripts/03_validate.py'),'after_sha256':sha(validator)},
               '02_build_reviewed_dataset.py':{'before_sha256':sha(original/'scripts/02_build_reviewed_dataset.py'),'after_sha256':sha(builder)}},
           'QC_thresholds_unchanged':True,'scientific_CSV_payloads_unchanged':True,
           'original_installed_scripts_unchanged':True,'source_preparation_rerun':False,
           'ECG_detection_rerun':False,'fresh_visual_adjudication_performed':False}
    write_json(continuation/'CONTINUATION_AUDIT.json',audit)
    # Preserve pre-normalization bytes and failed validation directory.
    for name in ['OUTPUT_SHA256.json']+[x['file'] for x in normalization]:
        shutil.copy2(results/name,backup/name)
        staged=results/(name+'.continuation_staged')
        shutil.copyfile(original/'results'/name,staged)
        os.replace(staged,results/name)
    if (historical/'verification').exists():
        shutil.move(str(historical/'verification'),str(backup/'failed_verification'))
    os.environ['KITCO3_REVIEW_RAW_DATA_ROOT']=str(raw)
    print('Verified identical data; continuing full validation without re-preparing sources.',flush=True)
    three.run_python(validator,'--release',qcroot,'--work',work,'--results',results,
                     '--output',historical/'verification')
    require(all(sha(results/n)==h for n,h in baseline.items()),'Historical data changed during validation')
    require(sha(results/'OUTPUT_SHA256.json')==sha(original/'results/OUTPUT_SHA256.json'),
            'Historical output seal differs after validation')
    # Continue the original Script03 stages from the point after run_qc.py.
    timing=out/'timing_diagnostics'
    if timing.exists():shutil.move(str(timing),str(backup/'prior_timing_diagnostics'))
    (timing/'inputs').mkdir(parents=True)
    shutil.copy2(original/'inputs/recordings_142.csv',timing/'inputs/recordings_142.csv')
    shutil.copy2(package/'inputs/source_acquisition_manifest_185.csv',timing/'inputs/source_acquisition_manifest_185.csv')
    three.run_python(package/'support/01_raw_timing_audit.py','--root',timing,'--raw-root',raw)
    if a.regenerate_review_plots:
        review=out/'regenerated_review'
        if review.exists():shutil.move(str(review),str(backup/'prior_regenerated_review'))
        plot=original/'scripts/01_plot_review.py';common=['--inputs',original/'inputs','--work',work]
        three.run_python(plot,*common,'--mode','overview','--output',review/'overview')
        three.run_python(plot,*common,'--mode','events','--epochs',original/'decisions/review_epochs/review_epochs.csv',
                         '--output',review/'flagged')
        three.run_python(plot,*common,'--mode','events','--epochs',original/'decisions/review_epochs/final_correction_epochs.csv',
                         '--decisions',original/'decisions/waveform_decisions.json','--output',review/'accepted_edits')
    require(not (out/'dataset').exists(),'Unexpected existing handoff dataset; stop before replacing it')
    three.build_handoff(out,selection,rec,sheet)
    from presentation_figures import make_qc_figures
    make_qc_figures(out/'dataset',out/'figures')
    audit.update(status='CONTINUATION_COMPLETE',completed_utc=datetime.now(timezone.utc).isoformat(),
                 validation_sha256=sha(historical/'verification/validation.json'))
    write_json(continuation/'CONTINUATION_AUDIT.json',audit)
    status={'script':path.name,'version':three.VERSION,'status':'ECG_QC_REPLAY_COMPLETE',
            'assets_verified':assets,'selection_mode':'ORIGINAL_SELECTION_RERUN_AND_MATCHED_TO_REVIEWED_SNAPSHOT',
            'historical_result_files_identical':11,'current_visual_review_performed':False,
            'physical_alignment_verified':False,'final_synchrony_dataset_frozen':False,'PLV_computed':False,
            'reviewer':'Existing documented Codex AI-assisted decisions, replayed without alteration',
            'handoff_sha256':sha(out/'03_STEP04_HANDOFF.json'),
            'compatibility_continuation':{
                'script':Path(__file__).name,'version':VERSION,
                'audit_file':'continuation_v1_0_0/CONTINUATION_AUDIT.json',
                'audit_sha256':sha(continuation/'CONTINUATION_AUDIT.json'),
                'median_implementation':'Explicit edge-padded 11-point median; same QC thresholds',
                'gzip_policy':'Reference compressed bytes only after exact decompressed CSV equality'}}
    write_json(out/'03_RUN_STATUS.json',status)
    print(json.dumps(status,indent=2),flush=True)

if __name__=='__main__':
    try:
        main()
    except (ValueError,OSError,KeyError,AssertionError,subprocess.CalledProcessError) as error:
        print('CONTINUATION STOPPED: '+str(error)+'. Preserve existing files; do not bypass validation.',file=sys.stderr)
        sys.exit(2)
