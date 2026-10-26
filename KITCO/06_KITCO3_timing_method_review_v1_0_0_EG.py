#!/usr/bin/env python3
"""KITCO3 Script06 v1.0.0: timing-evidence inventory and PLV method review.

Read-only audit of existing Script04/05 results. No ECG reprocessing, PLV
re-estimation, clock correction, condition-effect test, or inclusion freeze.
Defaults write to the Mac's INTERNAL home folder because an external-drive
write error interrupted the previous plotting run. See generated README.
"""
from __future__ import annotations
import argparse,csv,hashlib,io,json,os,platform,re,sys,zipfile
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import pandas as pd

VERSION='1.0.0'

def require(ok,message):
    if not ok:raise ValueError(message)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def atomic_text(path,text):
    temp=path.with_name(path.name+'.writing')
    with temp.open('w',encoding='utf-8',newline='') as f:
        f.write(text);f.flush();os.fsync(f.fileno())
    temp.replace(path)

def csvout(path,frame):
    if not isinstance(frame,pd.DataFrame):frame=pd.DataFrame(frame)
    atomic_text(path,frame.to_csv(index=False,float_format='%.10g'))

def clean_json(obj):
    if isinstance(obj,dict):return {str(k):clean_json(v) for k,v in obj.items()}
    if isinstance(obj,(list,tuple)):return [clean_json(v) for v in obj]
    if isinstance(obj,(np.integer,)):return int(obj)
    if isinstance(obj,(float,np.floating)):return float(obj) if np.isfinite(obj) else None
    if isinstance(obj,(np.bool_,)):return bool(obj)
    return obj

def jsonout(path,obj):atomic_text(path,json.dumps(clean_json(obj),indent=2,allow_nan=False)+'\n')

def verified_release(root,name,expected,tracked):
    path=root/name;require(path.is_file(),'Missing completion status: '+str(path))
    status=json.loads(path.read_text());require(status.get('status')==expected,'Incomplete or unsupported source: '+str(path))
    tracked[str(path)]=sha(path)
    require(isinstance(status.get('output_sha256'),dict) and status['output_sha256'],'Missing source hashes')
    for rel,digest in status['output_sha256'].items():
        f=(root/rel).resolve();require(f.is_relative_to(root),'Unsafe source manifest path')
        require(f.is_file() and sha(f)==digest,'Changed, missing or unreadable source: '+str(f));tracked[str(f)]=digest
    return status

def union_seconds(intervals):
    result=[]
    for a,b in sorted(intervals):
        if result and a<=result[-1][1]+1e-8:result[-1][1]=max(result[-1][1],b)
        else:result.append([float(a),float(b)])
    return sum(b-a for a,b in result)

def quantiles(frame,group,value):
    rows=[]
    for key,g in frame.groupby(group,sort=True):
        x=pd.to_numeric(g[value],errors='raise').to_numpy(float);require(np.isfinite(x).all(),'Nonfinite diagnostics')
        rows.append({group:key,'rows':len(x),'median':np.median(x),'p05':np.quantile(x,.05),'p95':np.quantile(x,.95),'maximum':x.max()})
    return pd.DataFrame(rows)

def plotting_checks(paths,current05):
    rows=[]
    expected={'ALL_PHASE_PUBLICATION_FORMAT_PLOTS_COMPLETE','ALL_PHASE_DESCRIPTIVE_PLOTS_COMPLETE'}
    for folder in sorted(set(p.resolve() for p in paths)):
        if not folder.exists():
            rows.append({'directory':str(folder),'status':'DIRECTORY_NOT_FOUND','checked_output_files':0,'issues':''});continue
        path=folder/'05b_RUN_STATUS.json'
        if not path.is_file():
            rows.append({'directory':str(folder),'status':'NO_COMPLETION_STATUS','checked_output_files':0,'issues':'Do not infer completion from PNGs or the last pair filename.'});continue
        checked=0;issues=[]
        try:
            s=json.loads(path.read_text());require(s.get('status') in expected,'Unrecognised plotting status')
            source=s.get('input_sha256',{});matches=[h for n,h in source.items() if Path(n).name=='05_RUN_STATUS.json']
            require(len(matches)==1 and matches[0]==current05,'Plotting source Script05 does not match current results')
            require(s.get('paired_phases')==71,'Plotting pair count differs')
            require(isinstance(s.get('output_sha256'),dict) and s['output_sha256'],'Missing plotting manifest')
            for rel,h in s['output_sha256'].items():
                f=(folder/rel).resolve();require(f.is_relative_to(folder),'Unsafe plotting manifest path')
                require(f.is_file() and sha(f)==h,'Changed, missing or unreadable plot output: '+rel);checked+=1
            state='COMPLETION_AND_FILE_HASHES_VERIFIED'
        except Exception as e:state='NOT_VERIFIED';issues.append(str(e))
        rows.append({'directory':str(folder),'status':state,'checked_output_files':checked,'issues':' | '.join(issues)})
    return pd.DataFrame(rows)

def classify(name):
    lower=name.lower();base=Path(lower).name
    if re.search(r'(^|[^a-z])(acc|accelerometer|acceleration)([^a-z]|$)',base):return 'ACCELEROMETER_CANDIDATE'
    if any(x in base for x in ['gyro','magnetometer','inertial']):return 'OTHER_MOTION_CANDIDATE'
    if any(x in base for x in ['sync','stomp','clap','calibrat','clock','offset','drift']):return 'TIMING_DOCUMENT_OR_EVENT_CANDIDATE'
    if any(x in base for x in ['annotat','event','log','metadata','device','sensor','session']):return 'METADATA_OR_EVENT_CANDIDATE'
    if base.startswith('ecg') or 'electrocard' in base:return 'ECG_EXPORT_NOT_INDEPENDENT_TIMING'
    if Path(base).suffix.lower() in ['.mp4','.mov','.m4v','.avi']:return 'VIDEO_CANDIDATE_REQUIRES_EVENT_REVIEW'
    return 'OTHER_POTENTIAL_EVIDENCE'

def header_details(payload,name):
    text=payload.decode('utf-8-sig',errors='replace');lines=text.splitlines()[:24];header=[]
    if Path(name).suffix.lower() in ['.csv','.tsv','.txt','.log']:
        for line in lines:
            cells=next(csv.reader([line],delimiter='\t' if Path(name).suffix.lower()=='.tsv' else ','))
            if any(re.search(r'time|stamp|clock',c,re.I) for c in cells) and len(cells)>=2:
                header=cells;break
    elif Path(name).suffix.lower()=='.json':
        try:
            obj=json.loads(text)
            if isinstance(obj,dict):header=list(obj)
        except json.JSONDecodeError:pass
    time_fields=[c for c in header if re.search(r'time|stamp|clock|counter|tick',c,re.I)]
    sensor=any(re.search(r'sensor|device|hardware|tick',c,re.I) for c in time_fields)
    host=any(re.search(r'host|phone|ipad|tablet|receipt|receive|unix|utc',c,re.I) for c in time_fields)
    return {'header_fields':json.dumps(header),'time_or_counter_fields':json.dumps(time_fields),
            'separate_sensor_and_host_named_fields_candidate':bool(sensor and host),
            'header_note':'Field names and file presence are clues only; units, origins and mappings are unverified.'}

def timing_inventory(roots,associations):
    rows=[];errors=[];seen=set();nested=0
    allowed={'.csv','.tsv','.json','.txt','.log','.yaml','.yml','.pdf','.xlsx','.xls','.docx','.mp4','.mov','.m4v','.avi','.zip'}
    def association(path):
        # Path containment links files to an attempt/sensor as inventory candidates.
        return ';'.join(sorted({pair for parent,pair in associations if path.is_relative_to(parent)}))
    def inspect(source,member,size,opener,path):
        kind=classify(member or str(source));row={'source_file':str(source),'archive_member':member,'size_bytes':size,
            'kind':kind,'candidate_pair_ids':association(path),'inspection':'FILE_METADATA_ONLY','header_fields':'[]',
            'time_or_counter_fields':'[]','separate_sensor_and_host_named_fields_candidate':False,'header_note':''}
        if Path(member or str(source)).suffix.lower() in ['.csv','.tsv','.json','.txt','.log']:
            try:
                with opener() as f:payload=f.read(65536)
                row.update(header_details(payload,member or source.name));row['inspection']='FIRST_64_KIB_ONLY'
                if size>65536:row['header_note']+=' File exceeds sampled prefix; remaining content was not inspected.'
                row['sampled_prefix_sha256']=hashlib.sha256(payload).hexdigest()
            except Exception as e:row['inspection']='READ_ERROR';errors.append({'path':str(source),'member':member,'error':str(e)})
        rows.append(row)
    for root in roots:
        if not root.is_dir():errors.append({'path':str(root),'member':'','error':'Evidence directory unavailable; not scanned'});continue
        def onerror(e):errors.append({'path':str(e.filename),'member':'','error':str(e)})
        for folder,dirs,files in os.walk(root,onerror=onerror,followlinks=False):
            dirs[:]=[d for d in dirs if not d.startswith('.') and not (Path(folder)/d).is_symlink()]
            for name in sorted(files):
                if name.startswith('.') or Path(name).suffix.lower() not in allowed:continue
                p=(Path(folder)/name).resolve()
                if p in seen:continue
                seen.add(p)
                try:
                    if not p.is_relative_to(root):continue
                    size=p.stat().st_size
                    if p.suffix.lower()!='.zip':inspect(p,'',size,lambda p=p:p.open('rb'),p);continue
                    with zipfile.ZipFile(p) as z:
                        for item in z.infolist():
                            if item.is_dir() or Path(item.filename).name.startswith('.') or '__MACOSX/' in item.filename:continue
                            ext=Path(item.filename).suffix.lower()
                            if ext not in allowed:continue
                            if ext=='.zip':
                                nested+=1;rows.append({'source_file':str(p),'archive_member':item.filename,'size_bytes':item.file_size,
                                    'kind':'NESTED_ARCHIVE_NOT_INSPECTED','candidate_pair_ids':'','inspection':'CENTRAL_DIRECTORY_ONLY',
                                    'header_fields':'[]','time_or_counter_fields':'[]','separate_sensor_and_host_named_fields_candidate':False,
                                    'header_note':'Contents require a separate archive review. Do not treat absence in this inventory as absence of data.'})
                            else:inspect(p,item.filename,item.file_size,lambda item=item,z=z:z.open(item),p)
                except Exception as e:errors.append({'path':str(p),'member':'','error':str(e)})
    columns=['source_file','archive_member','size_bytes','kind','candidate_pair_ids','inspection','header_fields','time_or_counter_fields',
             'separate_sensor_and_host_named_fields_candidate','header_note','sampled_prefix_sha256']
    return pd.DataFrame(rows,columns=columns),pd.DataFrame(errors,columns=['path','member','error']),nested

def duration_comparisons(summaries):
    baseline=summaries.loc[summaries.family.eq('working_200s_40s')].set_index('pair_id');rows=[]
    for family in ['duration_180s','duration_250s','duration_300s']:
        other=summaries.loc[summaries.family.eq(family)].set_index('pair_id')
        common=baseline.index[(baseline.windows>0)&(other.reindex(baseline.index).windows>0)]
        a=baseline.loc[common,'mean_local_PLV'];b=other.loc[common,'mean_local_PLV'];delta=b-a
        corr=a.rank().corr(b.rank()) if len(common)>2 and a.nunique()>1 and b.nunique()>1 else np.nan
        rows.append({'family':family,'common_pair_phases':len(common),'median_signed_difference':delta.median(),
            'median_absolute_difference':abs(delta).median(),'maximum_absolute_difference':abs(delta).max(),
            'descriptive_spearman_rank_correlation':corr,
            'comparison_scope':'COMMON_PAIRS_BUT_DIFFERENT_CORE_LENGTHS_AND_POTENTIALLY_DIFFERENT_SUPPORTED_PERIODS',
            'interpretation':'Descriptive sensitivity only; not a matched-period estimator comparison or optimal-window test.'})
    return pd.DataFrame(rows)

README='''# KITCO3 Script06 v1.0.0: timing and method review

## Purpose and scope
The immediate scientific step is to review feasibility and available timing
evidence before choosing/fixing the primary analysis. This script audits existing
outputs and creates review tables, an evidence inventory and a DRAFT analysis
specification. It does not establish physical synchronization merely from an
exported timestamp, a detected motion peak, file names or a second time column.
It does not recompute ECG, RR intervals or PLV, change QC, select the most
favourable specification, freeze inclusion or test a condition effect.

## Run on Emma's Mac
Save the script in /Volumes/One Touch/KITCO3_drive/scripts and run:
python3 "/Volumes/One Touch/KITCO3_drive/scripts/06_KITCO3_timing_method_review_v1_0_0_EG.py"

It reads existing outputs/04_dyadic_signals_v1_0_0 and
outputs/05_PLV_feasibility_v1_0_0 on the external project drive.
It writes a NEW folder on INTERNAL storage:
~/KITCO3_outputs/06_timing_method_review_v1_0_0
No existing folder is overwritten. Use --output-dir with a fresh path to rerun.
It uses only numpy and pandas in addition to the Python standard library.
The original Script05 .py file is not needed: sealed result files are the inputs.

Optional flags: --project-root, --script03-dir, --script04-dir, --script05-dir, --script02-dir,
--raw-data-root, --output-dir, --plotting-dir PATH, --evidence-root PATH.
The last two options can be repeated to include additional folders. The default
plotting checks cover the external v1.1.0 output and the previously suggested
internal local_run1 output. Supply your actual successful --plotting-dir if
you used a different name. Plot completion is reported separately and does
not determine physiological inclusion or invalidate already verified 04/05 data.

## What is checked
All Script04/05 output hashes, their lineage link, all 71 working summaries,
per-phase window counts, mean PLVs and union coverage are checked. Condition
code NA is read as a literal Neutral-Apple code, not a missing value. Story,
recipe and participant/session metadata are used only for coverage/design
description. No emotional-versus-neutral PLV difference is computed.
The existing Script03 raw timestamp audit is reused when a unique matching
copy is available. Its iPad labels and exported-clock diagnostics are preserved
as prior evidence, not promoted to physical calibration. No full ECG timing
audit is repeated. Existing same-core sensitivity scores are aggregated per pair phase; windows
are not treated as independent N. Duration comparisons use common pair phases
but different core lengths and potentially different periods. They are labelled
accordingly and cannot establish the best window. Synthetic independent-noise
PLV distributions are method diagnostics, not a matched human null test.

## Timing evidence inventory and limitations
The data folder (plus any --evidence-root) is traversed read-only. Relevant file
names, video/document availability, CSV/header metadata and first-level ZIP
members are inventoried. At most 64 KiB of each text file is read; no complete
accelerometer waveform, video or manual document review is performed. The
inventory does not assume timestamp units or calculate a physical rate from
an unexplained numeric column. Sensor/host-named time fields are only review
candidates. Files in the same acquisition directory are associated as candidate
evidence, not proof of common events or a shared clock. Original Script02
record metadata supply candidate acquisition-directory associations when present.
Nested archives are explicitly listed as NOT INSPECTED. Encrypted/unreadable
files and inaccessible directories are reported as incomplete inventory, never
silently interpreted as no evidence. If motion/video files were stored elsewhere,
add their folders with --evidence-root; not found does not mean not collected.
No events are matched or offsets/drifts fitted in this release. That requires
identifiable shared events, justified matching, clock-domain mappings and an
uncertainty assessment. Do not align by maximizing ECG/PLV similarity.

## Outputs and interpretation
06_REVIEW_REPORT.md is the main readable result.
06_plotting_completion.csv records checked plotting directories and file hashes.
06_timing_file_inventory.csv, 06_timing_inventory_errors.csv and
06_recording_timing_context.csv describe timing evidence scope and gaps.
06_timing_evidence_review_template.csv gives one unresolved row per pair phase
for documenting independent evidence; this script does not consume or validate
manual entries and does not change alignment based on them.
06_pair_phase_review_71.csv and 06_coverage_by_design.csv describe availability.
06_method_support.csv, 06_matched_core_sensitivity_summary.csv,
06_duration_common_pair_comparisons.csv, 06_synthetic_duration_summary.csv and
06_synthetic_context_summary.csv review existing estimator diagnostics.
06_design_dependence.json records session/participant reuse, not a fitted model.
06_DRAFT_ANALYSIS_SPEC.json and 06_decisions_to_resolve.csv list scientific
decisions that cannot be made by a file inventory or estimator identity test.
06_RUN_STATUS.json is written last and seals outputs. AUDIT_COMPLETE means
the report was generated; it does not mean timing verified or analysis finalized.

## What remains after the script
Review the evidence inventory for actual calibration/shared events and assess
residual offset/drift uncertainty. If independent evidence cannot resolve it,
explicitly justify the exported-time assumption and prespecify sensitivity.
Choose the estimand (mean local versus whole-interaction phase consistency),
signal, band, core/context, coverage and aggregation rules for the research
question. Finalize an autocorrelation-preserving reference approach and
participant/session-dependent condition model only after verifying the original
assignment, counterbalancing and exchangeability. Then freeze inclusion and
the analysis plan before running the condition effect. Prior exploratory work
has been viewed, so this is not a preregistration or proof of outcome blindness.
200 s remains a working candidate; no universal number of cycles guarantees
reliable phase. Do not choose a method because it produces a larger condition
effect. All existing sources remain unchanged.
'''

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root',type=Path,default=Path('/Volumes/One Touch/KITCO3_drive'))
    for name in ['script03-dir','script04-dir','script05-dir','script02-dir','raw-data-root','output-dir']:p.add_argument('--'+name,type=Path)
    p.add_argument('--plotting-dir',type=Path,action='append');p.add_argument('--evidence-root',type=Path,action='append',default=[])
    a=p.parse_args();root=a.project_root.expanduser().resolve()
    src4=(a.script04_dir or root/'outputs/04_dyadic_signals_v1_0_0').resolve()
    src5=(a.script05_dir or root/'outputs/05_PLV_feasibility_v1_0_0').resolve()
    src2=(a.script02_dir or root/'outputs/02_ecg_processing_v1_1_0').resolve()
    src3=(a.script03_dir or root/'outputs/03_ecg_QC_v2_1_0').resolve()
    raw=(a.raw_data_root or root/'data').resolve()
    out=(a.output_dir or Path.home()/'KITCO3_outputs/06_timing_method_review_v1_0_0').expanduser().resolve()
    require(not out.exists(),'Output folder exists. Preserve it and choose a NEW --output-dir.')
    require(not out.is_relative_to(raw),'Output must be outside the raw-data evidence folder')
    tracked={};s4=verified_release(src4,'04_RUN_STATUS.json','CONDITIONAL_SIGNAL_EXPORT_COMPLETE',tracked)
    s5=verified_release(src5,'05_RUN_STATUS.json','CONDITIONAL_PLV_FEASIBILITY_COMPLETE',tracked)
    require(s4.get('script_version')=='1.0.0' and s4.get('grid_hz')==4 and s5.get('version')=='1.0.0','This audit expects Script04/05 v1.0.0 and a 4 Hz grid')
    require(s5['source_status_sha256']==sha(src4/'04_RUN_STATUS.json'),'Script05 used different Script04 inputs')
    require(s4.get('timing_basis')=='exported_time_unverified' and s4.get('physical_clock_verified') is False,'Timing basis changed; review new evidence separately')
    require(s5.get('physical_clock_verified') is False and s5.get('condition_effect_computed') is False,'Unexpected upstream scientific status')
    reg=pd.read_csv(src4/'04_pair_signal_register_71.csv',keep_default_na=False)
    sums=pd.read_csv(src5/'05_pair_phase_method_summaries.csv',keep_default_na=False)
    w=pd.read_csv(src5/'05_working_window_diagnostics.csv',keep_default_na=False)
    require(len(reg)==71 and reg.pair_id.is_unique,'Expected 71 unique pair phases')
    require(not sums.duplicated(['pair_id','family']).any(),'Duplicate method summary')
    require(set(sums.pair_id)==set(reg.pair_id),'Method-summary cohort differs')
    require(not w.duplicated(['pair_id','paired_segment_id','core_start_s']).any(),'Duplicate scored windows')
    require(set(w.pair_id).issubset(set(reg.pair_id)),'Orphan windows')
    require(w.status.eq('CONDITIONAL_DIAGNOSTIC_SCORE').all(),'Degenerate/unscored windows need an adapted review')
    require(w.PLV.between(0,1).all() and w.core_duration_s.eq(200).all(),'Invalid working scores')
    sums['mean_local_PLV']=pd.to_numeric(sums.mean_local_PLV.replace('',np.nan),errors='raise')
    base=sums.loc[sums.family.eq('working_200s_40s')].set_index('pair_id')
    require(len(base)==71 and set(base.index)==set(reg.pair_id),'Working register incomplete')
    require(base.core_s.eq(200).all() and base.context_each_side_s.eq(40).all() and
            base.band_low_hz.eq(.04).all() and base.band_high_hz.eq(.15).all() and
            base.filter_order_argument.eq(3).all(),'Working estimator differs from the documented specification')
    require(w.context_each_side_s.eq(40).all(),'Working context differs')
    require(int(s5['working_spec_scored_windows'])==len(w) and s4.get('paired_phases')==71 and s5.get('paired_phases')==71,'Source status counts differ')
    for r in reg.itertuples(index=False):
        q=w.loc[w.pair_id.eq(r.pair_id)];v=base.loc[r.pair_id]
        require(len(q)==v.windows,'Working window count mismatch')
        if len(q):require(abs(q.PLV.mean()-v.mean_local_PLV)<1e-8,'Working mean PLV mismatch')
        require(abs(union_seconds([(x,x+200) for x in q.core_start_s])-v.core_union_s)<1e-7,'Working core union mismatch')
    print('Verified Script04/05 and all 71 working summaries.',flush=True)
    out.mkdir(parents=True);atomic_text(out/'06_README.md',README)
    plots=a.plotting_dir or [root/'outputs/05b_PLV_publication_figures_v1_1_0',Path.home()/'KITCO3_outputs/05b_PLV_publication_figures_v1_1_0_local_run1']
    plot_status=plotting_checks([x.expanduser() for x in plots],sha(src5/'05_RUN_STATUS.json'));csvout(out/'06_plotting_completion.csv',plot_status)
    recording_rows=[];associations=[]
    for r in reg.itertuples(index=False):
        for j in [1,2]:
            uid=getattr(r,f'recording_uid_{j}');path=src2/'records'/uid/'02_result.json'
            row={'pair_id':r.pair_id,'recording_uid':uid,'metadata_status':'NOT_AVAILABLE','source_attempt_path':'','filter_design_rate_hz':'','header_nominal_rate_hz':'','physical_clock_calibrated_by_this_script':False}
            if path.is_file():
                try:
                    data=json.loads(path.read_text());require(data['recording_uid']==uid,'Recording identity mismatch');tracked[str(path)]=sha(path)
                    source=(raw/data['ECG_path']).resolve();require(source.is_relative_to(raw),'Unsafe source path')
                    row.update(metadata_status='READ',source_attempt_path=str(source.parent.parent),filter_design_rate_hz=data.get('sampling_rate_hz_used',''),header_nominal_rate_hz=data.get('header_nominal_rate_hz',''))
                    associations.append((source.parent.parent,r.pair_id))
                except Exception as e:row['metadata_status']='READ_ERROR: '+str(e)
            recording_rows.append(row)
    context=pd.DataFrame(recording_rows);prior_status='NOT_FOUND_IN_SCRIPT03_DIRECTORY'
    previous=list(src3.rglob('raw_timestamp_audit_142.csv')) if src3.is_dir() else []
    if previous:
        digests={sha(f) for f in previous}
        if len(digests)==1:
            file=previous[0];old=pd.read_csv(file,keep_default_na=False)
            require('recording_uid' in old and old.recording_uid.is_unique,'Prior timing audit has ambiguous recording identities')
            require(set(old.recording_uid)==set(context.recording_uid),'Prior timing audit does not cover the current 142 recordings')
            fields=[c for c in old.columns if c not in context.columns or c=='recording_uid']
            context=context.merge(old[fields],on='recording_uid',validate='one_to_one');tracked[str(file)]=sha(file)
            csvout(out/'06_existing_raw_timestamp_audit.csv',old);prior_status='REUSED_UNCHANGED_NOT_PHYSICAL_CALIBRATION'
        else:prior_status='MULTIPLE_NONIDENTICAL_COPIES_NOT_SELECTED'
    csvout(out/'06_recording_timing_context.csv',context)
    print('Inventorying available timing/motion/event file headers. No clock fitting.',flush=True)
    roots=list(dict.fromkeys([raw,*[x.expanduser().resolve() for x in a.evidence_root]]))
    for directory in roots:require(not out.is_relative_to(directory),'Output folder must be outside every evidence root')
    inv,errors,nested=timing_inventory(roots,associations)
    csvout(out/'06_timing_file_inventory.csv',inv);csvout(out/'06_timing_inventory_errors.csv',errors)
    pairs=reg.merge(base[['windows','mean_local_PLV','core_union_s']],left_on='pair_id',right_index=True,validate='one_to_one')
    pairs['working_score_available']=pairs.windows>0
    pairs['working_evaluability']=np.where(pairs.working_score_available,'SCORED',np.where(pairs.common_grid_points>0,'INSUFFICIENT_CONTINUOUS_CONTEXT','NO_PAIRED_RR_GRID'))
    pairs['scored_core_coverage_fraction']=pairs.core_union_s/pairs.shared_cooking_s
    csvout(out/'06_pair_phase_review_71.csv',pairs)
    grouping=[c for c in ['Condition','Story','Recipe','Phase'] if c in pairs.columns]
    coverage=pairs.groupby(grouping,dropna=False).agg(candidate_pair_phases=('pair_id','size'),working_scored_phases=('working_score_available','sum'),scored_core_union_s=('core_union_s','sum')).reset_index()
    coverage['not_scored_phases']=coverage.candidate_pair_phases-coverage.working_scored_phases;csvout(out/'06_coverage_by_design.csv',coverage)
    support=sums.groupby('family').agg(candidate_pair_phases=('pair_id','size'),scored_pair_phases=('windows',lambda x:int((x>0).sum())),scored_windows=('windows','sum'),core_union_s=('core_union_s','sum'),core_s=('core_s','first'),context_each_side_s=('context_each_side_s','first'),band_low_hz=('band_low_hz','first'),band_high_hz=('band_high_hz','first')).reset_index()
    require(support.candidate_pair_phases.eq(71).all(),'A sensitivity family lacks pair-phase rows')
    csvout(out/'06_method_support.csv',support)
    matched=pd.read_csv(src5/'05_matched_core_sensitivity.csv')
    require(not matched.duplicated(['pair_id','family']).any(),'Duplicate matched sensitivity rows')
    require(set(matched.pair_id).issubset(set(reg.pair_id)),'Orphan matched sensitivity rows')
    require(matched.matched_windows.gt(0).all(),'Matched sensitivity contains no-support rows')
    matched['absolute_change_in_pair_mean']=abs(matched.signed_change_in_pair_mean)
    sensitivity=quantiles(matched,'family','absolute_change_in_pair_mean').rename(columns={'rows':'pair_phases','median':'median_absolute_pair_mean_change','p05':'p05_absolute_pair_mean_change','p95':'p95_absolute_pair_mean_change','maximum':'maximum_absolute_pair_mean_change'})
    sensitivity['scope']='WITHIN_PAIR_PHASE_SAME_MATCHED_SCORING_CORES; NO P_VALUES'
    csvout(out/'06_matched_core_sensitivity_summary.csv',sensitivity)
    duration=duration_comparisons(sums);csvout(out/'06_duration_common_pair_comparisons.csv',duration)
    syn=pd.read_csv(src5/'05_synthetic_duration_bias.csv');synsum=quantiles(syn,'core_duration_s','independent_noise_PLV');csvout(out/'06_synthetic_duration_summary.csv',synsum)
    edge=pd.read_csv(src5/'05_synthetic_context_errors.csv');edgesum=quantiles(edge,'context_each_side_s','absolute_PLV_difference');csvout(out/'06_synthetic_context_summary.csv',edgesum)
    scored=pairs.loc[pairs.working_score_available];members=pd.concat([scored[['Session_ID','pair_id','PID_1']].rename(columns={'PID_1':'PID'}),scored[['Session_ID','pair_id','PID_2']].rename(columns={'PID_2':'PID'})])
    counts=members.groupby(['Session_ID','PID']).pair_id.nunique()
    both=scored.groupby('Session_ID').Story.agg(lambda s:set(s)) if 'Story' in scored else pd.Series(dtype=object)
    design={'candidate_pair_phases':71,'working_scored_pair_phases':len(scored),'sessions_with_working_scores':scored.Session_ID.nunique(),
        'participant_memberships_in_scored_phases':len(members),'unique_session_participant_identities':len(counts),'session_participants_in_multiple_scored_phases':int((counts>1).sum()),
        'sessions_with_both_emotional_and_neutral_scored_phases':int(sum({'Emotional','Neutral'}.issubset(v) for v in both)),
        'independence_note':'Neither windows nor 4 Hz samples are independent N. Pair phases also share sessions/participants.',
        'assignment_unit_verified':False,'model_fitted':False,'permutation_exchangeability_verified':False}
    jsonout(out/'06_design_dependence.json',design)
    decisions=[
        ('TIMING','Independent mapping or documented conditional assumption','Identify clock domains and shared events; assess offset/drift uncertainty. Existing shift scenarios are not measured error bounds.'),
        ('ESTIMAND','Mean local versus whole-interaction consistency','Tie the timescale and pair-phase summary to the research question; the two summaries answer different questions.'),
        ('SIGNAL_AND_BAND','RR interval (IBI) and 0.04-0.15 Hz remain working choices','Document physiological interpretation and Hilbert-phase limitations. No automatic band optimisation.'),
        ('CORE_AND_CONTEXT','200 s core + 40 s per side remains a working candidate','Review support and synthetic/observed sensitivities; different duration summaries are not matched-period comparisons.'),
        ('COVERAGE','Eligibility and minimum phase coverage not frozen','Define continuous support and coverage requirements before condition-effect testing; do not infer them from plotting availability.'),
        ('REFERENCE','Autocorrelation-preserving reference not finalized','Define the null hypothesis and pair-phase aggregation; independent-noise simulations are not a matched human significance test.'),
        ('DESIGN','Original assignment/counterbalancing/exchangeability not verified','Confirm assignment unit, story exposure, recipe and phase order; account for shared participants and sessions; do not assume permutation validity.'),
        ('MISSINGNESS','Availability must be described without result-driven inclusion','Review coverage by design; no imputed PLV or arbitrary new exclusions are applied.'),
        ('FREEZE','Final inclusion and statistical specification not frozen','Record decisions and sensitivities before comparing emotional and neutral effects. Earlier exploratory outputs have been viewed.')]
    csvout(out/'06_decisions_to_resolve.csv',pd.DataFrame(decisions,columns=['decision','current_status','needed']))
    template=pairs[['pair_id','recording_uid_1','recording_uid_2']].copy()
    for c in ['independent_evidence_file','event_identity','sensor_to_export_mapping','offset_estimate_s','drift_estimate_ppm','uncertainty_basis','reviewer','decision']:template[c]=''
    template['review_status']='UNRESOLVED_FROM_VERIFIED_OUTPUTS';csvout(out/'06_timing_evidence_review_template.csv',template)
    draft={'status':'DRAFT_NOT_FROZEN','research_question':'Does emotional versus neutral story condition affect partners cardiac phase consistency during collaborative cooking?',
           'current_working_specification':{'signal':'RR interval (IBI)','sample_grid_hz':4,'band_hz':[.04,.15],'core_s':200,'context_each_side_s':40,'step_s':1,'summary':'arithmetic mean of local window PLV; not pooled whole-interaction PLV'},
           'scientific_decisions':[{'decision':x,'status':y,'needed':z} for x,y,z in decisions],
           'physical_clock_verified':False,'new_exclusions_applied':False,'condition_effect_computed':False,'final_synchrony_dataset_frozen':False}
    jsonout(out/'06_DRAFT_ANALYSIS_SPEC.json',draft)
    motion=int(inv.kind.eq('ACCELEROMETER_CANDIDATE').sum());dual=int(inv.separate_sensor_and_host_named_fields_candidate.sum())
    plot_summary='; '.join(f'{Path(v.directory).name}: {v.status}' for v in plot_status.itertuples(index=False))
    support_text='\n'.join(f'| {x.family} | {x.scored_pair_phases} | {x.scored_windows} |' for x in support.itertuples(index=False))
    synthetic_text='\n'.join(f'| {int(x.core_duration_s)} | {x.median:.4f} | {x.p95:.4f} |' for x in synsum.itertuples(index=False))
    report=f'''# KITCO3 timing and method review v{VERSION}

Verified existing Script04/05 files and their lineage. This report is a review
checkpoint, not physical clock verification, a final analysis specification or
a condition comparison.

## Availability
71 candidate pair phases; {len(scored)} have working PLV scores; {len(w)}
overlapping windows; {base.core_union_s.sum()/3600:.4f} hours of scored-core union.
{int((pairs.working_evaluability=='NO_PAIRED_RR_GRID').sum())} phases have no paired
RR grid; {int((pairs.working_evaluability=='INSUFFICIENT_CONTINUOUS_CONTEXT').sum())}
have some grid but insufficient continuous support for the working context.
These are current evaluability descriptions, not final scientific exclusions.

## Plot export completion
{plot_summary}
Plotting completion is an administrative check, not an ECG inclusion criterion.
An incomplete export does not itself invalidate verified Script04/05 data.

## Timing evidence scope
Prior Script03 timestamp audit: {prior_status}.
Scanned roots: {', '.join(str(x) for x in roots)}.
Inventoried {len(inv)} files/archive members, including {motion} accelerometer
filename candidates and {dual} sensor/host-named time-field candidates.
There were {len(errors)} inventory errors and {nested} uninspected nested archives.
Review the exact inventory and error table. Missing files in this scope must not
be described as never collected. Header sampling is not motion-event matching
or manual review of videos, documents, clock logs or calibration reports.
Finding a motion signal, a shared event name or similar exported timestamps
does not independently establish physical clock offset/drift. No timing
calibration or correction was applied. The physical-time assumption remains
UNRESOLVED_FROM_VERIFIED_OUTPUTS.

## Working-method support
| Specification | Pair phases with scores | Dependent windows |
|---|---:|---:|
{support_text}

Same-duration variants use the existing within-pair matched-core sensitivity
table. Clock changes are scenarios, not measured error bounds. Different bands
define different operational signals. Duration comparisons in the separate
table share pair identities, but not equal cores or guaranteed identical periods.
They do not establish an optimal window and are not paired hypothesis tests.

## Synthetic independent-noise PLV
| Core seconds | Median PLV | 95th percentile |
|---|---:|---:|
{synthetic_text}

These are simulation distributions under one synthetic model, not confidence
intervals or a matched reference for the observed human pair-phase means.
Positive PLV alone is not a significance test. Duration changes its distribution.

## Design and next decision
{design['sessions_with_working_scores']} sessions have working scores;
{design['session_participants_in_multiple_scored_phases']} session-participant
identities occur in multiple scored phases. See design_dependence.json and
coverage_by_design.csv. No condition-effect difference or p-value is calculated.
The original assignment unit and valid exchangeability remain unverified.

Next: inspect the identified independent timing evidence and document either
its mapping/uncertainty or the unresolved exported-time assumption. Resolve the
estimand, band, core/context, phase coverage, reference and statistical design
in 06_decisions_to_resolve.csv. Freeze the plan and inclusion before condition
comparisons. No optimal specification or physical calibration is claimed here.
'''
    atomic_text(out/'06_REVIEW_REPORT.md',report)
    for n,h in tracked.items():require(sha(Path(n))==h,'Critical input changed during review: '+n)
    result={'script':Path(__file__).name,'version':VERSION,'script_sha256':sha(Path(__file__)),'status':'TIMING_METHOD_AUDIT_COMPLETE_WITH_OPEN_DECISIONS',
        'generated_utc':datetime.now(timezone.utc).isoformat(),'paired_phases':71,'working_scored_pair_phases':len(scored),'working_scored_windows':len(w),
        'timing_inventory_scope':'FILE_METADATA_AND_FIRST_64_KIB_HEADERS; NO_EVENT_MATCHING','prior_timestamp_audit':prior_status,'timing_inventory_errors':len(errors),'uninspected_nested_archives':nested,
        'physical_clock_verified':False,'new_QC_exclusions_applied':False,'PLV_recomputed':False,'condition_effect_computed':False,'final_synchrony_dataset_frozen':False,
        'environment':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__},'input_sha256':tracked,
        'output_sha256':{str(f.relative_to(out)):sha(f) for f in sorted(out.rglob('*')) if f.is_file() and '.writing' not in f.name}}
    jsonout(out/'06_RUN_STATUS.json',result)
    print(json.dumps(clean_json({k:v for k,v in result.items() if k not in ['input_sha256','output_sha256']}),indent=2))
    print('Read: '+str(out/'06_REVIEW_REPORT.md'))

if __name__=='__main__':
    try:main()
    except Exception as e:
        print('AUDIT STOPPED: '+str(e)+'. No completion status is published for a failed run.',file=sys.stderr);sys.exit(1)
