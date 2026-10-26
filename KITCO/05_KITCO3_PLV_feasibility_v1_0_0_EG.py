#!/usr/bin/env python3
"""KITCO3 Script05 v1.0.0: conditional LF-IBI PLV feasibility and diagnostics.

Run after verified Script04 exports. No raw ECG processing or new beat edits.
This is NOT a confirmatory condition test or final inclusion freeze. It computes
descriptive working PLV and numerical sensitivities; it cannot calibrate clocks,
prove one physiological oscillator, or distinguish interpersonal coupling from
common inputs. No condition labels are used in estimation or method selection.
Earlier exploratory results have been viewed; this is not preregistration.

Working estimand: mean local phase-consistency magnitude across equal 200 s
cores starting at cooking 40 s + integer seconds. Every core has 40 s real IBI
context on each side. Detrend/standardize each complete context, order-3
Butterworth SOS bandpass .04-.15 Hz forward/backward, FFT Hilbert, score core.
All available supported cooking sections are considered. Gaps are not bridged.
Near-zero amplitude and nonmonotone phase are reported, not used for arbitrary
exclusions. Candidate evaluability is not final scientific inclusion.

The same prior v0.5/v0.6 operational method is adapted to Script04's regular
4 Hz exports. Rounding in those exports is retained. Positive time-shift and
drift scenarios query partner2 at t*(1+ppm*1e-6)+shift, within the SAME paired
run. They are scenario checks, never fitted corrections or measured error bounds.

Dependencies: numpy, pandas, scipy, matplotlib. Python >=3.9.
Default project: /Volumes/One Touch/KITCO3_drive
Outputs: outputs/05_PLV_feasibility_v1_0_0 (must be NEW).
Detailed decisions and limitations are written to 05_METHODS.md on each run.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
import platform
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
import scipy
from scipy import signal
from scipy.interpolate import PchipInterpolator
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

VERSION='1.0.0'
FS=4
SEED=20261016

def require(ok,message):
    if not ok:raise ValueError(message)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def write_json(path,data):
    path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def write_csv(path,data):
    if not isinstance(data,pd.DataFrame):data=pd.DataFrame(data)
    data.to_csv(path,index=False,float_format='%.10f',lineterminator='\n')

def analytic(x,band=(.04,.15),order=3):
    x=np.asarray(x,dtype=float)
    require(np.isfinite(x).all(),'Nonfinite IBI cannot enter a phase context')
    y=signal.detrend(x,axis=-1,type='linear')
    sd=np.std(y,axis=-1,keepdims=True)
    require(np.all(sd>np.finfo(float).eps),'Degenerate context has undefined phase')
    sos=signal.butter(order,band,fs=FS,btype='bandpass',output='sos')
    return signal.hilbert(signal.sosfiltfilt(sos,y/sd,axis=-1),axis=-1)

def unit(z):
    amplitude=np.abs(z)
    require(np.isfinite(amplitude).all() and (amplitude>np.finfo(float).tiny).all(),
            'Zero/nonfinite analytic amplitude has undefined phase')
    return z/amplitude

def plv(z1,z2):
    return np.abs(np.mean(unit(z1)*np.conj(unit(z2)),axis=-1))

def merged_coverage(intervals):
    spans=[]
    for lo,hi in sorted(intervals):
        if spans and lo<=spans[-1][1]+1e-8:spans[-1][1]=max(hi,spans[-1][1])
        else:spans.append([float(lo),float(hi)])
    return sum(hi-lo for lo,hi in spans)

def family(name,core=200,margin=40,band=(.04,.15),order=3,shift=0.,ppm=0.):
    return dict(name=name,core=core,margin=margin,band=band,order=order,shift=shift,ppm=ppm)

def specifications():
    f=[family('working_200s_40s'),family('duration_180s',core=180),
       family('duration_250s',core=250),family('duration_300s',core=300),
       family('context_80s',margin=80),family('order_2',order=2),family('order_4',order=4),
       family('band_004_010',band=(.04,.10)),family('band_006_015',band=(.06,.15))]
    for shift in [-5.,-2.,-1.,-.25,.25,1.,2.,5.]:
        f.append(family('clock_shift_'+str(shift).replace('-','minus').replace('.','p')+'s',shift=shift))
    for ppm in [-100.,100.]:f.append(family('clock_drift_'+str(int(ppm)).replace('-','minus')+'ppm',ppm=ppm))
    return f

def window_starts(seg,duration,spec):
    m,c=spec['margin'],spec['core'];scale=1+spec['ppm']*1e-6;shift=spec['shift']
    # Require complete continuous-time context support, as in v0.5/v0.6,
    # AND every sampled context point to be inside the Script04 regular grid.
    lo=max(float(seg.support_start_s),float(seg.first_grid_s),0.)
    hi=min(float(seg.support_stop_s),float(seg.last_grid_s)+1/FS,float(duration))
    start_min=max(lo+m,(lo-shift)/scale+m)
    start_max=min(hi-c-m,(hi-shift)/scale-c-m)
    # Global cooking anchor is unchanged across segments. No origin optimization.
    first=40+int(np.ceil(start_min-40-1e-8))
    last=40+int(np.floor(start_max-40+1e-8))
    if last<first:return np.array([],dtype=float)
    return np.arange(first,last+1,dtype=float)

def contexts(g,starts,spec):
    m,c=spec['margin'],spec['core'];n=int((c+2*m)*FS)
    t=starts[:,None]-m+np.arange(n)[None,:]/FS
    base=g.cooking_s.to_numpy();ix=np.rint((t-base[0])*FS).astype(int)
    require(ix.min()>=0 and ix.max()<len(g),'Context extends outside regular grid')
    require(np.allclose(base[ix],t,rtol=0,atol=1e-8),'Context grid does not match source grid')
    x=g.ibi_1_s.to_numpy()[ix]
    if spec['shift']==0 and spec['ppm']==0:y=g.ibi_2_s.to_numpy()[ix]
    else:
        at=t*(1+spec['ppm']*1e-6)+spec['shift']
        require(at.min()>=base[0]-1e-8 and at.max()<=base[-1]+1e-8,'Clock scenario needs extrapolation')
        y=PchipInterpolator(base,g.ibi_2_s.to_numpy(),extrapolate=False)(at)
    return x,y

def diagnostics(x,z,core):
    zc=z[:,core];env=np.abs(zc);rms=np.sqrt(np.mean(np.abs(z)**2,axis=-1))
    freq=np.diff(np.unwrap(np.angle(zc),axis=-1),axis=-1)*FS/(2*np.pi)
    f,p=signal.periodogram(x[:,core],fs=FS,window='hann',detrend='linear',axis=-1)
    lf=(f>=.04)&(f<.15);total=(f>=.04)&(f<=.40)
    power=p[:,lf];den=power.sum(axis=-1);allpower=p[:,total].sum(axis=-1)
    require((den>0).all() and (allpower>0).all(),'Degenerate spectral diagnostic')
    w=power/den[:,None]
    entropy=-np.sum(w*np.log(w+np.finfo(float).tiny),axis=-1)/np.log(w.shape[-1])
    return dict(low_envelope_fraction=np.mean(env<.1*rms[:,None],axis=-1),
                negative_instant_frequency_fraction=np.mean(freq<0,axis=-1),
                lf_fraction_of_004_040_power=den/allpower,
                lf_normalized_spectral_entropy=entropy,
                lf_peak_hz=f[lf][np.argmax(power,axis=-1)])

def load_inputs(src):
    status=json.loads((src/'04_RUN_STATUS.json').read_text())
    require(status['status']=='CONDITIONAL_SIGNAL_EXPORT_COMPLETE','Unsupported/incomplete Script04 export')
    require(status['script_version']=='1.0.0' and status['grid_hz']==4,'Expected Script04 v1.0.0, 4 Hz')
    require(status['timing_basis']=='exported_time_unverified','This release requires the documented export-time assumption')
    require(not status['physical_clock_verified'],'Unexpected timing claim; review new timing evidence separately')
    require(not status['PLV_computed'],'Unexpected upstream synchrony output')
    for n,h in status['output_sha256'].items():
        path=(src/n).resolve();require(path.is_relative_to(src),'Unsafe input path')
        require(path.is_file() and sha(path)==h,'Script04 file changed: '+n)
    # Do not load condition/story/recipe columns into the estimation data.
    reg=pd.read_csv(src/'04_pair_signal_register_71.csv',keep_default_na=False,
        usecols=['pair_id','recording_uid_1','recording_uid_2','shared_cooking_s','common_grid_points','paired_interpolation_segments'])
    seg=pd.read_csv(src/'04_paired_interpolation_segments.csv')
    require(len(reg)==71 and reg.pair_id.is_unique,'Expected 71 unique pair phases')
    require(not seg.duplicated(['pair_id','paired_segment_id']).any(),'Duplicate interpolation segment')
    require(set(seg.pair_id).issubset(reg.pair_id),'Orphan interpolation segment')
    data={};count=0
    for r in reg.itertuples(index=False):
        if not r.common_grid_points:
            require(not seg.pair_id.eq(r.pair_id).any(),'Zero-grid register has segments');continue
        name=f'signals/{r.pair_id}_IBI_4Hz.csv.gz'
        require(name in status['output_sha256'],'Signal not sealed in Script04 status')
        d=pd.read_csv(src/name,keep_default_na=False)
        require(len(d)==r.common_grid_points and not d.cooking_s.duplicated().any(),'Signal point count/duplicates')
        require(d.pair_id.eq(r.pair_id).all(),'Signal pair identity mismatch')
        require(np.isfinite(d[['cooking_s','ibi_1_s','ibi_2_s']]).all().all(),'Nonfinite paired data')
        require((d[['ibi_1_s','ibi_2_s']]>0).all().all(),'Nonpositive IBI')
        require(((d.cooking_s>=0)&(d.cooking_s<r.shared_cooking_s)).all(),'Grid outside cooking')
        s=seg.loc[seg.pair_id.eq(r.pair_id)].sort_values('paired_segment_id')
        require(len(s)==r.paired_interpolation_segments,'Segment/register mismatch')
        spans=[]
        for sr in s.itertuples(index=False):
            g=d.loc[d.paired_segment_id.eq(sr.paired_segment_id)].sort_values('cooking_s')
            require(len(g)==sr.grid_points and len(g)>0,'Segment point count mismatch')
            require(np.allclose(np.diff(g.cooking_s),.25,rtol=0,atol=1e-8),'Nonuniform/gapped within-run grid')
            require(abs(g.cooking_s.iloc[0]-sr.first_grid_s)<1e-8 and abs(g.cooking_s.iloc[-1]-sr.last_grid_s)<1e-8,'Segment bounds mismatch')
            require(sr.support_start_s<=sr.first_grid_s+1e-8 and sr.support_stop_s>=sr.last_grid_s-1e-8,'Grid beyond paired support')
            for j in [1,2]:
                require(g[f'recording_uid_{j}'].eq(getattr(r,f'recording_uid_{j}')).all(),'Partner recording mismatch')
                require(g[f'source_ibi_segment_{j}'].eq(getattr(sr,f'source_ibi_segment_{j}')).all(),'Source segment mismatch')
            spans.append((sr.first_grid_s,sr.last_grid_s))
        require(all(spans[i][0]>spans[i-1][1] for i in range(1,len(spans))),'Overlapping/unsorted paired runs')
        data[r.pair_id]=d;count+=len(d)
    require(count==status['total_grid_points'],'Total grid point mismatch')
    return status,reg,seg,data

def score_data(reg,segments,data,out,batch):
    specs=specifications();window_rows=[];summaries=[];sensitivity=[];example=None
    for position,r in enumerate(reg.sort_values('pair_id').itertuples(index=False),1):
        baseline={};variants={s['name']:[] for s in specs}
        for sr in segments.loc[segments.pair_id.eq(r.pair_id)].itertuples(index=False):
            g=data[r.pair_id].loc[lambda d:d.paired_segment_id.eq(sr.paired_segment_id)].sort_values('cooking_s')
            for spec in specs:
                starts=window_starts(sr,r.shared_cooking_s,spec)
                for begin in range(0,len(starts),batch):
                    st=starts[begin:begin+batch];x,y=contexts(g,st,spec)
                    # Skip only truly degenerate numerical contexts, and report them.
                    valid=(np.std(signal.detrend(x,axis=-1),axis=-1)>np.finfo(float).eps)&(np.std(signal.detrend(y,axis=-1),axis=-1)>np.finfo(float).eps)
                    for start in st[~valid]:
                        if spec['name']==specs[0]['name']:
                            window_rows.append(dict(pair_id=r.pair_id,paired_segment_id=int(sr.paired_segment_id),core_start_s=start,core_duration_s=spec['core'],status='DEGENERATE_CONTEXT',PLV=np.nan))
                    st=st[valid];x=x[valid];y=y[valid]
                    if not len(st):continue
                    z1,z2=analytic(x,spec['band'],spec['order']),analytic(y,spec['band'],spec['order'])
                    core=slice(int(spec['margin']*FS),int((spec['margin']+spec['core'])*FS))
                    values=plv(z1[:,core],z2[:,core])
                    for start,v in zip(st,values):
                        key=(int(sr.paired_segment_id),float(start));variants[spec['name']].append((key,float(v)))
                    if spec['name']==specs[0]['name']:
                        ds=[diagnostics(x,z1,core),diagnostics(y,z2,core)]
                        for j,(start,v) in enumerate(zip(st,values)):
                            key=(int(sr.paired_segment_id),float(start));baseline[key]=float(v)
                            row=dict(pair_id=r.pair_id,paired_segment_id=key[0],core_start_s=float(start),core_duration_s=200,context_each_side_s=40,status='CONDITIONAL_DIAGNOSTIC_SCORE',PLV=float(v))
                            for k,d in enumerate(ds,1):
                                for name,array in d.items():row[f'partner{k}_{name}']=float(array[j])
                            window_rows.append(row)
                        if example is None:
                            example=(r.pair_id,float(st[0]),x[0],y[0],z1[0],z2[0])
        for spec in specs:
            vals=variants[spec['name']];n=len(vals)
            summaries.append(dict(pair_id=r.pair_id,family=spec['name'],core_s=spec['core'],context_each_side_s=spec['margin'],band_low_hz=spec['band'][0],band_high_hz=spec['band'][1],filter_order_argument=spec['order'],shift_partner2_s=spec['shift'],drift_partner2_ppm=spec['ppm'],windows=n,mean_local_PLV=float(np.mean([v for _,v in vals])) if n else np.nan,core_union_s=merged_coverage([(key[1],key[1]+spec['core']) for key,v in vals]),shared_cooking_s=r.shared_cooking_s,evaluability='SOME_WORKING_SPEC_SUPPORT' if n else 'NO_WORKING_SPEC_SUPPORT',final_inclusion='NOT_FROZEN_BY_SCRIPT05'))
            if spec['core']==200 and spec['name']!=specs[0]['name']:
                matched=[(baseline[k],v) for k,v in vals if k in baseline]
                if matched:
                    a,b=np.array(matched).T
                    sensitivity.append(dict(pair_id=r.pair_id,family=spec['name'],matched_windows=len(a),baseline_mean_on_matched_cores=float(a.mean()),variant_mean_on_matched_cores=float(b.mean()),signed_change_in_pair_mean=float(b.mean()-a.mean()),mean_absolute_window_change=float(abs(b-a).mean()),max_absolute_window_change=float(abs(b-a).max())))
        if position==1 or position%10==0 or position==len(reg):print(f'Assessed {position}/{len(reg)} pair phases',flush=True)
    windows=pd.DataFrame(window_rows);summaries=pd.DataFrame(summaries);sensitivity=pd.DataFrame(sensitivity)
    write_csv(out/'05_working_window_diagnostics.csv',windows)
    write_csv(out/'05_pair_phase_method_summaries.csv',summaries)
    write_csv(out/'05_matched_core_sensitivity.csv',sensitivity)
    return windows,summaries,sensitivity,example

def synthetic(out,nrep):
    rng=np.random.default_rng(SEED);t=np.arange(-1024,1024,1/FS);center=len(t)//2
    durations=[100,180,200,300,500];rows=[];edges=[]
    sos=signal.butter(3,[.04,.15],fs=FS,btype='bandpass',output='sos')
    for begin in range(0,nrep,50):
        n=min(50,nrep-begin)
        x=signal.sosfiltfilt(sos,rng.normal(size=(n,len(t))),axis=-1)
        y=signal.sosfiltfilt(sos,rng.normal(size=(n,len(t))),axis=-1)
        zx,zy=analytic(x),analytic(y)
        for duration in durations:
            a=center-duration*FS//2;b=a+duration*FS;m=40*FS
            v=plv(analytic(x[:,a-m:b+m])[:,m:-m],analytic(y[:,a-m:b+m])[:,m:-m]);reference=plv(zx[:,a:b],zy[:,a:b])
            for j in range(n):rows.append(dict(replicate=begin+j,core_duration_s=duration,independent_noise_PLV=float(v[j]),absolute_long_context_PLV_difference=float(abs(v[j]-reference[j]))))
        a=center-400;b=center+400
        for margin in [20,40,80,120]:
            m=margin*FS;z1=analytic(x[:,a-m:b+m])[:,m:-m];z2=analytic(y[:,a-m:b+m])[:,m:-m]
            v=plv(z1,z2);ref=plv(zx[:,a:b],zy[:,a:b]);phase=np.maximum(np.quantile(abs(np.angle(unit(z1)*np.conj(unit(zx[:,a:b])))),.95,axis=-1),np.quantile(abs(np.angle(unit(z2)*np.conj(unit(zy[:,a:b])))),.95,axis=-1))
            for j in range(n):edges.append(dict(replicate=begin+j,context_each_side_s=margin,absolute_PLV_difference=float(abs(v[j]-ref[j])),max_partner_phase_p95_error_rad=float(phase[j])))
    known=[]
    for f in [.04,.045,.08,.145,.15]:
        for lag in [0,np.pi/3,np.pi]:
            for scenario in ['fixed_lag','frequency_mismatch','two_components']:
                theta=2*np.pi*f*t;other=2*np.pi*(f+.003 if scenario=='frequency_mismatch' else f)*t+lag
                x=.8+.035*np.cos(theta)+.008*np.cos(2*np.pi*.25*t)+.00001*t
                y=.9+.03*np.cos(other)+.008*np.cos(2*np.pi*.27*t)+.00001*t
                if scenario=='two_components':x=x+.03*np.cos(2*np.pi*.11*t+.7);y=y+.03*np.cos(2*np.pi*.065*t+1.9)
                a=center-400;b=center+400;m=160
                got=float(plv(analytic(x[a-m:b+m])[m:-m],analytic(y[a-m:b+m])[m:-m]));truth=float(abs(np.mean(np.exp(1j*(theta[a:b]-other[a:b])))))
                known.append(dict(scenario=scenario,frequency_hz=f,phase_offset_rad=lag,target_component_PLV=truth,estimated_signal_PLV=got,absolute_difference=abs(got-truth)))
    d=pd.DataFrame(rows);e=pd.DataFrame(edges);k=pd.DataFrame(known)
    require(k.loc[k.scenario.eq('fixed_lag'),'estimated_signal_PLV'].min()>.995,'Known fixed-lag calibration failed')
    require(k.loc[k.scenario.eq('frequency_mismatch'),'absolute_difference'].max()<.02,'Known frequency-mismatch calibration failed')
    write_csv(out/'05_synthetic_duration_bias.csv',d);write_csv(out/'05_synthetic_context_errors.csv',e);write_csv(out/'05_synthetic_known_phase.csv',k)
    return d,e,k

def save_figure(fig,out,name):
    fig.savefig(out/(name+'.png'),dpi=300,bbox_inches='tight')
    fig.savefig(out/(name+'.pdf'),bbox_inches='tight')
    fig.savefig(out/(name+'.svg'),bbox_inches='tight');plt.close(fig)

def figures(out,summaries,sensitivity,duration,edges,example):
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
    dest=out/'figures';dest.mkdir()
    fig,axes=plt.subplots(1,2,figsize=(11,4.5));order=['working_200s_40s','duration_180s','duration_250s','duration_300s','context_80s']
    counts=[int((summaries.loc[summaries.family.eq(n),'windows']>0).sum()) for n in order]
    axes[0].bar(range(len(order)),counts,color='#286b8a');axes[0].set_xticks(range(len(order)),['200/40','180/40','250/40','300/40','200/80']);axes[0].set_ylim(0,71);axes[0].set_ylabel('Pair phases with at least one context');axes[0].set_xlabel('Core / context each side (s)')
    for i,n in enumerate(counts):axes[0].text(i,n+.8,str(n),ha='center',va='bottom')
    durations=sorted(duration.core_duration_s.unique());q=[duration.loc[duration.core_duration_s.eq(v),'independent_noise_PLV'].quantile([.05,.5,.95]).to_numpy() for v in durations];q=np.array(q)
    axes[1].plot(durations,q[:,1],'o-',color='#286b8a');axes[1].fill_between(durations,q[:,0],q[:,2],alpha=.2,color='#286b8a');axes[1].set_xlabel('Scoring duration (s)');axes[1].set_ylabel('Independent simulated signal PLV');axes[1].set_ylim(0,1)
    fig.suptitle('Window support and finite-duration reference behavior');fig.text(.5,.005,'Descriptive method diagnostics. Overlapping windows are not independent observations. Shading: simulation 5th-95th percentiles.',ha='center',fontsize=9);fig.tight_layout(rect=[0,.04,1,.95]);save_figure(fig,dest,'05_feasibility_overview')
    if example is not None:
        pair,start,x,y,z1,z2=example;t=np.arange(len(x))/FS+start-40;core=(t>=start)&(t<start+200)
        fig,axes=plt.subplots(4,1,figsize=(11,10),sharex=True)
        axes[0].plot(t,x,label='Partner 1',color='#286b8a');axes[0].plot(t,y,label='Partner 2',color='#bc693b');axes[0].set_ylabel('IBI (s)');axes[0].legend(loc='upper right')
        for z,color in [(z1,'#286b8a'),(z2,'#bc693b')]:
            axes[1].plot(t,z.real,color=color);axes[2].plot(t,np.abs(z),color=color)
        axes[1].set_ylabel('Filtered IBI\n(standardized)');axes[2].set_ylabel('Analytic envelope')
        axes[3].scatter(t[core],np.angle(unit(z1[core])*np.conj(unit(z2[core]))),s=3,color='#444444');axes[3].set_ylim(-np.pi,np.pi);axes[3].set_ylabel('Wrapped phase\ndifference (rad)');axes[3].set_xlabel('Time from shared cooking onset (s)')
        for ax in axes:ax.axvspan(start,start+200,color='#286b8a',alpha=.07)
        fig.suptitle(pair+': first eligible context by identity/time');fig.text(.5,.005,'Illustration selected without PLV ranking or condition labels. Export-time alignment remains physically unverified.',ha='center',fontsize=9);fig.tight_layout(rect=[0,.025,1,.96]);save_figure(fig,dest,'05_phase_estimation_example')

METHODS='''# KITCO3 Script05 v1.0.0: decisions and interpretation

Research question: does meaningful emotional versus neutral factual storytelling
alter partners' cardiac coordination during collaborative cooking? This stage
defines and assesses an operational statistic; it does not test that contrast.

## Working signal and estimator

Use reviewed IBI (seconds between accepted ECG R peaks), midpoint-timed and
already PCHIP-resampled by Script04 at 4 Hz within trusted runs. The working
band is 0.04-0.15 Hz, the conventional LF HRV band. It is not a uniquely
validated interpersonal oscillator or a selective sympathetic measure. Narrower
band variants are different operational outcomes, not interchangeable truths.

For each full context: linearly detrend, divide by population SD, apply an
order-3 Butterworth bandpass using SOS forward-backward filtering with SciPy's
default odd padding, then compute FFT Hilbert at the context length. The
bandpass has one-way order 6 and the forward-backward magnitude response is
squared. Real context does not mean padding is absent. Score only the central
200 seconds: abs(mean(exp(i*(phase1-phase2)))). Values range 0-1. Fixed nonzero
phase differences, including anti-phase, can yield PLV near one. This measures
consistency, not zero-lag agreement, direction or causal influence.

## Windows, coverage and dependencies

Starts are 40 + k seconds from common cooking onset, k integer, throughout ALL
supported sections. A complete 40 s of real context is required on EACH side.
Continuous-time support and sampled-grid support are both checked. No gaps are
bridged, no signal is extrapolated, and no windows span disconnected runs.
Source-run boundaries from Script04 are retained, even if adjacent runs could
otherwise look continuous. No arbitrary first-three-minute limit is imposed.

Mean local PLV is the arithmetic mean of available equal-length window PLVs
within one pair phase. It is not pooled whole-interaction PLV. Overlapping
windows heavily reuse data; their count is not inferential N. Core coverage is
the union of window intervals, not window count times duration. Unequal cooking
and retained durations are reported, not equalized by deleting usable data.
Duration variants change the estimand. Other variants use matched core starts
within each pair phase, so their comparisons do not mix support changes with
estimator changes. Rows with no support remain in the pair-phase register.

## Numerical and phase diagnostics

Synthetic independent band-limited noise pairs use the same independent source
realizations across duration/context variants (seed 20261016, 500 replicates by
default). These show finite-duration PLV behavior and edge differences versus
long-context estimates of the same central samples. Percentile ranges are
simulation distributions, not confidence intervals for the experiment.
Fixed-lag and frequency-mismatch controls check the numerical estimator. The
two-component control demonstrates that an analytic mixture phase need not
equal either latent component phase. These models do not certify physiology.

Each participant-window reports descriptive LF spectral fraction/entropy and
peak, analytic-envelope weakness (below 10% of full-context RMS), and negative
instantaneous frequency fraction. The 10% number is a descriptive diagnostic,
not a validated exclusion rule. No spectral or envelope threshold is used to
drop data. Degenerate contexts fail phase definition and are explicitly logged.

The timing scenarios query partner2 at t*(1+ppm*1e-6)+shift. The chosen shifts
of +/-0.25, 1, 2 and 5 seconds and drift of +/-100 ppm are illustrative stress
scenarios, NOT measured clock errors, plausible bounds or data-derived repairs.
Every scenario retains within-run support. No shift is optimized to maximize
synchrony. Small changes under these scenarios do not prove physical timing.

## Status and remaining decisions

All PLV outputs are CONDITIONAL DESCRIPTIVE diagnostics. 'Some working-spec
support' is not final inclusion. No p values, surrogate significance, independent
sample-size correction, emotional-neutral contrast or inclusion freeze is
performed. 4 Hz samples and windows are autocorrelated. PLV alone cannot prove
interpersonal coupling; common task input and individual spectra can contribute.
Use suitable whole-pair reference designs and account for session/participant
dependence, partner switches, recipe, order and missing coverage when finalizing
the inferential plan. Physical clock mapping remains unverified. If available
event evidence can assess it, that assessment remains necessary; this program
does not use or calibrate accelerometer stomps or video.

Earlier exploratory outputs were viewed. This script does not retrospectively
make the project outcome-blind or preregistered. No condition labels are loaded
for method selection in this stage. The 200/40/1 s choice is inherited from
prior v0.5/v0.6 numerical audits, not a universal rule or proven optimum.

## Outputs and reproducibility

05_working_window_diagnostics.csv: supported working windows, descriptive PLV
and individual phase/spectrum diagnostics; numerically degenerate contexts are
explicitly marked. Never use these windows as independent statistical cases.
05_pair_phase_method_summaries.csv: one row per 71 pair phases per specification,
with mean local PLV, window count and union coverage; no-support rows retained.
05_matched_core_sensitivity.csv: paired changes only on identically placed cores
for same-duration variants; descriptive, not hypothesis tests.
05_synthetic_*.csv: reproducible model calibration outputs.
figures/: PNG 300 dpi, PDF and SVG; deterministic first eligible example.
05_RUN_STATUS.json: versions, input/output hashes, parameters and limitations.
Status is published last. A failed run has no completion status. Keep failed
outputs and choose a new output path after resolving the problem. No overwrite
or resume is supported, to prevent mixing configurations. Original files stay
unchanged. Reproduction means same data/parameters, not necessarily identical
compressed bytes across Python/zlib versions. Gzip inputs are CRC-checked.

## Primary methodological references

Matsuki et al. (2023), Hilbert phase interpretation:
https://doi.org/10.1038/s41598-023-30405-5
Aydore et al. (2013), PLV properties and sample-size dependence:
https://doi.org/10.1016/j.neuroimage.2013.02.008
Quigley et al. (2024), HR/HRV measurement and interpretation:
https://doi.org/10.1111/psyp.14604
SciPy implementation documentation:
https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html
https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sosfiltfilt.html
https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.hilbert.html
'''

def self_test():
    t=np.arange(800)/FS;z=np.exp(2j*np.pi*.08*t)
    require(abs(float(plv(z,z*np.exp(1j*np.pi)))-1)<1e-12,'Anti-phase control failed')
    require(abs(merged_coverage([(0,200),(1,201),(300,500)])-401)<1e-9,'Coverage union failed')
    v=np.arange(1120)/FS+.8;g=pd.DataFrame({'cooking_s':np.arange(1120)/FS,'ibi_1_s':v,'ibi_2_s':v})
    x,y=contexts(g,np.array([40.]),family('test'));require(np.array_equal(x[0],v),'Context indexing failed')
    try:contexts(g,np.array([41.]),family('test'))
    except ValueError:pass
    else:raise ValueError('Extrapolation guard failed')
    try:analytic(np.ones(1120))
    except ValueError:pass
    else:raise ValueError('Degenerate context accepted')
    print('Self-tests passed.',flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root',type=Path,default=Path('/Volumes/One Touch/KITCO3_drive'))
    p.add_argument('--script04-dir',type=Path)
    p.add_argument('--output-dir',type=Path)
    p.add_argument('--timing-basis',choices=['exported_time_unverified'],help='Required acknowledgement for a data run, not physical clock verification')
    p.add_argument('--synthetic-replicates',type=int,default=500)
    p.add_argument('--batch-size',type=int,default=64)
    p.add_argument('--self-test',action='store_true')
    a=p.parse_args()
    if a.self_test:self_test();return
    require(a.timing_basis=='exported_time_unverified','Pass --timing-basis exported_time_unverified to record the conditional assumption')
    require(a.synthetic_replicates>=100,'Use at least 100 synthetic replicates; release default is 500')
    require(1<=a.batch_size<=256,'Batch size must be 1-256')
    project=a.project_root.expanduser().resolve();src=(a.script04_dir or project/'outputs/04_dyadic_signals_v1_0_0').resolve();out=(a.output_dir or project/'outputs/05_PLV_feasibility_v1_0_0').resolve()
    require(not out.exists(),'Output directory exists; preserve it and choose a new path')
    begin=time.monotonic();upstream,reg,seg,data=load_inputs(src)
    source_hashes={n:sha(src/n) for n in ['04_RUN_STATUS.json',*upstream['output_sha256']]}
    out.mkdir(parents=True);(out/'05_METHODS.md').write_text(METHODS,encoding='utf-8')
    self_test();print('Running synthetic estimator checks...',flush=True)
    duration,edges,known=synthetic(out,a.synthetic_replicates)
    windows,summaries,sensitivity,example=score_data(reg,seg,data,out,a.batch_size)
    figures(out,summaries,sensitivity,duration,edges,example)
    baseline=summaries.loc[summaries.family.eq('working_200s_40s')]
    require(len(baseline)==71,'Working pair-phase register incomplete')
    for n,h in source_hashes.items():require(sha(src/n)==h,'Input changed during run: '+n)
    files={str(f.relative_to(out)):sha(f) for f in sorted(out.rglob('*')) if f.is_file() and not f.name.startswith('._') and f.name!='.DS_Store'}
    result={'script':Path(__file__).name,'version':VERSION,'status':'CONDITIONAL_PLV_FEASIBILITY_COMPLETE',
        'generated_utc':datetime.now(timezone.utc).isoformat(),'runtime_seconds':time.monotonic()-begin,
        'script_sha256':sha(Path(__file__)),'environment':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'matplotlib':matplotlib.__version__},
        'configuration':{'grid_hz':4,'window_step_s':1,'window_anchor_s':40,'working_core_s':200,'working_context_each_side_s':40,'working_band_hz':[.04,.15],'filter_order_argument':3,'synthetic_seed':SEED,'synthetic_replicates':a.synthetic_replicates,'specifications':specifications()},
        'source_script04_dir':str(src),'source_status_sha256':source_hashes['04_RUN_STATUS.json'],'source_sha256':source_hashes,
        'paired_phases':71,'phases_with_script04_grid':len(data),'paired_segments':len(seg),
        'working_spec_phases_with_scores':int((baseline.windows>0).sum()),'working_spec_scored_windows':int(baseline.windows.sum()),
        'working_spec_core_union_hours':float(baseline.core_union_s.sum()/3600),
        'degenerate_working_contexts':int(windows.status.eq('DEGENERATE_CONTEXT').sum()) if len(windows) else 0,
        'physical_clock_verified':False,'condition_effect_computed':False,'final_synchrony_dataset_frozen':False,
        'PLV_computed':True,'PLV_status':'CONDITIONAL_DESCRIPTIVE_METHOD_DIAGNOSTICS',
        'inclusion_selected_by_condition_results':False,'new_QC_exclusions_applied':False,
        'output_sha256':files,
        'remaining_steps':['Assess available independent timing evidence or explicitly justify unresolved timing assumptions.',
                           'Review numerical/phase diagnostics and specify the operational primary outcome and sensitivity plan.',
                           'Freeze inclusion and reference/statistical design accounting for dependence, recipe, order and missingness.',
                           'Only then perform condition comparisons; do not use windows or 4 Hz points as independent N.']}
    write_json(out/'05_RUN_STATUS.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['source_sha256','output_sha256','configuration']},indent=2),flush=True)

if __name__=='__main__':
    try:main()
    except (ValueError,OSError,KeyError,EOFError) as error:
        print('SCRIPT05 STOPPED: '+str(error)+'. No completion status is published for a failed run.',file=sys.stderr)
        sys.exit(2)
