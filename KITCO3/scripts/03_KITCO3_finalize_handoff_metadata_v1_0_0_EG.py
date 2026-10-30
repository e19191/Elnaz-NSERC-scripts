#!/usr/bin/env python3
"""Remove macOS sidecars from the Script03 scientific-file hash manifest only.

Version 1.0.0. This does not remove files, change datasets, rerun QC or alter
inclusion. It verifies all nine scientific file hashes and the existing status
chain, backs up both JSON files, removes only ._ entries corresponding to known
scientific files, updates the status handoff hash, and records an audit.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

VERSION='1.0.0'
EXPECTED={
    '03_new_peak_edits.csv','03_pair_ECG_QC_71.csv','03_pair_joint_usable_intervals.csv',
    '03_participant_phase_QC_216.csv','03_participant_usable_intervals.csv',
    '03_recording_ECG_QC_142.csv','03_reviewed_nonbridging_ibi.csv.gz',
    '03_reviewed_peak_times.csv.gz','03_sample_anchored_withheld_regions.csv'}

def require(ok,message):
    if not ok:raise ValueError(message)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_json(path,value):
    staged=path.with_suffix(path.suffix+'.staged')
    staged.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
    os.replace(staged,path)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--script03-dir',type=Path,
        default=Path('/Volumes/One Touch/KITCO3_drive/outputs/03_ecg_QC_v2_1_0'))
    a=p.parse_args();root=a.script03_dir.expanduser().resolve()
    hp=root/'03_STEP04_HANDOFF.json';sp=root/'03_RUN_STATUS.json'
    handoff=json.loads(hp.read_text());status=json.loads(sp.read_text())
    require(status['status']=='ECG_QC_REPLAY_COMPLETE','Script03 must be complete')
    require(sha(hp)==status['handoff_sha256'],'Existing handoff/status hash mismatch')
    hashes=handoff['output_sha256']
    require(EXPECTED.issubset(hashes),'Missing scientific files from manifest')
    extra=set(hashes)-EXPECTED
    require(all(n.startswith('._') and n[2:] in EXPECTED for n in extra),
            'Unexpected manifest entries: '+str(sorted(extra)))
    for name in EXPECTED:
        require(sha(root/'dataset'/name)==hashes[name],'Scientific file changed: '+name)
    if not extra:
        print('Handoff already contains only the nine scientific files. No changes made.')
        return
    dest=root/'handoff_metadata_fix_v1_0_0'
    require(not dest.exists(),'Fix folder already exists; preserve it and report its status before retrying')
    dest.mkdir();backup=dest/'backup';backup.mkdir()
    shutil.copy2(hp,backup/hp.name);shutil.copy2(sp,backup/sp.name)
    audit={'script':Path(__file__).name,'version':VERSION,'script_sha256':sha(Path(__file__)),
           'generated_utc':datetime.now(timezone.utc).isoformat(),
           'previous_handoff_sha256':sha(hp),'previous_status_sha256':sha(sp),
           'removed_manifest_entries':{n:hashes[n] for n in sorted(extra)},
           'scientific_dataset_hashes':{n:hashes[n] for n in sorted(EXPECTED)},
           'scientific_files_changed':False,'files_deleted':False,'QC_rerun':False}
    handoff['output_sha256']={n:hashes[n] for n in sorted(EXPECTED)}
    write_json(hp,handoff)
    audit['new_handoff_sha256']=sha(hp)
    write_json(dest/'HANDOFF_METADATA_FIX_AUDIT.json',audit)
    status['handoff_sha256']=sha(hp)
    status['handoff_metadata_fix']={
        'script':Path(__file__).name,'version':VERSION,
        'audit_file':'handoff_metadata_fix_v1_0_0/HANDOFF_METADATA_FIX_AUDIT.json',
        'audit_sha256':sha(dest/'HANDOFF_METADATA_FIX_AUDIT.json')}
    write_json(sp,status)
    require(sha(hp)==json.loads(sp.read_text())['handoff_sha256'],'Updated hash chain failed')
    require(all(sha(root/'dataset'/n)==hashes[n] for n in EXPECTED),'Dataset changed during fix')
    print('HANDOFF_METADATA_FINALIZED: nine scientific file hashes verified; '+
          str(len(extra))+' metadata entries removed from manifest. No data files changed or deleted.')

if __name__=='__main__':
    try:main()
    except (ValueError,OSError,KeyError) as error:
        print('HANDOFF FIX STOPPED: '+str(error),file=sys.stderr)
        sys.exit(2)
