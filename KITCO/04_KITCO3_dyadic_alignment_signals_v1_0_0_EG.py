#!/usr/bin/env python3
"""Script04 v1.0.0: verified Script03 handoff -> common-time IBI signal segments.

Does not estimate sensor clock offsets, select a PLV band/window, or compute
condition effects. Output is explicitly conditional on exported time mapping.
"""
import argparse,gzip,hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from presentation_figures import plot_pair_ibi

VERSION='1.0.0'


def require(ok,msg):
    if not ok:raise ValueError(msg)


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def trusted_runs(g):
    runs=[]
    for sid,r in g.loc[g.trusted].groupby('ibi_segment_id',sort=True):
        t=r.midpoint_cooking_s.to_numpy(float);v=r.ibi_s.to_numpy(float)
        require(sid>0 and np.isfinite(v).all() and (v>0).all(),'Invalid trusted IBI run')
        require(np.isfinite(t).all() and (np.diff(t)>0).all(),'Run time not strictly increasing')
        if len(t)>=2:runs.append((int(sid),t,v))
    return runs


def align_pair(g1,g2,duration,fs=4):
    rows=[];segment_rows=[];segment=0;used=set()
    for sid1,t1,v1 in trusted_runs(g1):
        for sid2,t2,v2 in trusted_runs(g2):
            lo=max(0.,t1[0],t2[0]);hi=min(duration,t1[-1],t2[-1])
            if hi<=lo:continue
            first=int(np.ceil(lo*fs));last=int(np.floor(hi*fs))
            grid=np.arange(first,last+1,dtype=np.int64)/fs
            grid=grid[(grid>=lo)&(grid<=hi)&(grid<duration)]
            if len(grid)==0:continue
            require(not any(float(t) in used for t in grid),'Ambiguous overlapping trusted support')
            used.update(map(float,grid));segment+=1
            a=PchipInterpolator(t1,v1,extrapolate=False)(grid)
            b=PchipInterpolator(t2,v2,extrapolate=False)(grid)
            require(np.isfinite(a).all() and np.isfinite(b).all(),'Interpolation exceeded trusted support')
            require((a>0).all() and (b>0).all(),'Nonpositive interpolated IBI')
            for t,x,y in zip(grid,a,b):rows.append(dict(paired_segment_id=segment,
                source_ibi_segment_1=sid1,source_ibi_segment_2=sid2,cooking_s=t,
                ibi_1_s=x,ibi_2_s=y))
            segment_rows.append(dict(paired_segment_id=segment,source_ibi_segment_1=sid1,
                source_ibi_segment_2=sid2,support_start_s=lo,support_stop_s=hi,
                first_grid_s=grid[0],last_grid_s=grid[-1],grid_points=len(grid)))
    return rows,segment_rows


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root',type=Path,default=Path('/Volumes/One Touch/KITCO3_drive'))
    p.add_argument('--script03-dir',type=Path)
    p.add_argument('--output-dir',type=Path)
    p.add_argument('--timing-basis',required=True,choices=['exported_time_unverified'],
        help='Explicitly record the existing timing assumption; this does not validate it.')
    p.add_argument('--plot-all-pairs',action='store_true',help='Export all supported pairs as PNG/PDF/SVG; default one preselected identity example.')
    a=p.parse_args(argv);root=a.project_root.expanduser().resolve()
    src=(a.script03_dir or root/'outputs/03_ecg_QC_v2_1_0').resolve()
    out=(a.output_dir or root/'outputs/04_dyadic_signals_v1_0_0').resolve()
    status=json.loads((src/'03_RUN_STATUS.json').read_text())
    require(status['status']=='ECG_QC_REPLAY_COMPLETE','Script03 incomplete')
    require(status['version'] in {'2.0.0','2.1.0'},'Unsupported Script03 version')
    require(sha(src/'03_STEP04_HANDOFF.json')==status['handoff_sha256'],'Handoff hash mismatch')
    h=json.loads((src/'03_STEP04_HANDOFF.json').read_text());data=src/'dataset'
    for n,expected in h['output_sha256'].items():
        path=(data/n).resolve()
        require(path.is_relative_to(data.resolve()),'Unsafe handoff path')
        require(sha(path)==expected,f'Script03 dataset modified: {n}')
    ibi=pd.read_csv(data/'03_reviewed_nonbridging_ibi.csv.gz')
    require(ibi.trusted.dtype==bool,'Trusted flag must be boolean')
    require(ibi.loc[~ibi.trusted,'ibi_s'].isna().all(),'Invalid IBI must be missing')
    pair=pd.read_csv(data/'03_pair_ECG_QC_71.csv',keep_default_na=False)
    phase_metadata=pd.read_csv(data/'03_participant_phase_QC_216.csv',keep_default_na=False).set_index(['Session_ID','PID','Phase'])
    require(len(pair)==71 and pair.pair_id.is_unique,'Invalid pair register')
    out.mkdir(parents=True,exist_ok=False);(out/'signals').mkdir()
    registers=[];segments=[];hashes={};plotted=False
    for r in pair.sort_values('pair_id').itertuples(index=False):
        meta1=phase_metadata.loc[(r.Session_ID,int(r.PID_1),int(r.Phase))]
        meta2=phase_metadata.loc[(r.Session_ID,int(r.PID_2),int(r.Phase))]
        condition=meta1.Condition
        require(condition==meta2.Condition and condition in {'NA','NB','EA','EB'},'Invalid pair condition')
        one=ibi.loc[ibi.recording_uid.eq(r.recording_uid_1)]
        two=ibi.loc[ibi.recording_uid.eq(r.recording_uid_2)]
        rows,spans=align_pair(one,two,float(r.duration_s))
        for s in spans:segments.append(dict(pair_id=r.pair_id,**s))
        if rows:
            frame=pd.DataFrame(rows).sort_values(['cooking_s','paired_segment_id'])
            require(not frame.cooking_s.duplicated().any(),'Duplicated common time')
            frame.insert(0,'pair_id',r.pair_id)
            frame['recording_uid_1']=r.recording_uid_1;frame['recording_uid_2']=r.recording_uid_2
            # Unix grid derived from the already checked shared annotation onset.
            onset=float(np.median(one.midpoint_unix_s-one.midpoint_cooking_s))
            for g in [one,two]:
                require(np.max(np.abs(g.midpoint_unix_s-g.midpoint_cooking_s-onset))<2e-6,'Shared onset mismatch')
            frame['unix_s']=onset+frame.cooking_s
            file=out/'signals'/f'{r.pair_id}_IBI_4Hz.csv.gz'
            payload=frame.to_csv(index=False,float_format='%.9f').encode()
            file.write_bytes(gzip.compress(payload,mtime=0));hashes[str(file.relative_to(out))]=sha(file)
            # Deterministic example: first supported pair by identifier, never chosen for high synchrony.
            if a.plot_all_pairs or not plotted:
                plot_pair_ibi(r.pair_id,frame,out/'figures');plotted=True
        registers.append(dict(pair_id=r.pair_id,Session_ID=r.Session_ID,Phase=r.Phase,
            PID_1=r.PID_1,PID_2=r.PID_2,Condition=condition,
            Story='Emotional' if condition.startswith('E') else 'Neutral',
            Recipe='Apple' if condition.endswith('A') else 'Banana',
            recording_uid_1=r.recording_uid_1,recording_uid_2=r.recording_uid_2,
            shared_cooking_s=r.duration_s,joint_RR_endpoint_coverage_s=r.joint_trusted_ibi_s,
            paired_interpolation_segments=len(spans),common_grid_points=len(rows),
            signal_status='CONDITIONAL_COMMON_TIME_IBI' if rows else 'NO_COMMON_MIDPOINT_GRID_SUPPORT',
            physical_clock_status='UNVERIFIED_SENSOR_TO_EXPORT_MAPPING',
            final_synchrony_inclusion='NOT_DECIDED_BY_SCRIPT04'))
    pd.DataFrame(registers).to_csv(out/'04_pair_signal_register_71.csv',index=False,float_format='%.9f')
    pd.DataFrame(segments).to_csv(out/'04_paired_interpolation_segments.csv',index=False,float_format='%.9f')
    for n in ['04_pair_signal_register_71.csv','04_paired_interpolation_segments.csv']:hashes[n]=sha(out/n)
    report=dict(script_version=VERSION,status='CONDITIONAL_SIGNAL_EXPORT_COMPLETE',
        timing_basis=a.timing_basis,physical_clock_verified=False,grid_hz=4,
        interpolation='PCHIP at IBI midpoints, within a single trusted run per partner, no extrapolation',
        paired_phases=len(pair),phases_with_grid=sum(r['common_grid_points']>0 for r in registers),
        total_grid_points=sum(r['common_grid_points'] for r in registers),
        source_handoff_sha256=sha(src/'03_STEP04_HANDOFF.json'),
        output_sha256=hashes,PLV_computed=False,final_synchrony_dataset_frozen=False,
        notes=['No LF band or window chosen here','4 Hz points are not independent samples',
               'Endpoint RR coverage differs from IBI midpoint interpolation support',
               'No time shift or drift fitted to physiology; no stomps calibrated in this script'])
    (out/'04_RUN_STATUS.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='output_sha256'},indent=2))


if __name__=='__main__':
    main()
