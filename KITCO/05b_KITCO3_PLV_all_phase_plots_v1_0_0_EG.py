#!/usr/bin/env python3
"""KITCO3 05b v1.0.0. All-phase descriptive PLV figures from sealed 04/05 outputs.

Requires the unchanged 05_KITCO3_PLV_feasibility_v1_0_0_EG.py beside this file
or supplied with --processing-script. No QC decisions, condition comparisons,
alignment corrections or new inclusion rules are made. See generated README.
"""
from pathlib import Path
from contextlib import contextmanager
import argparse
import importlib.util
import hashlib
import json
import platform
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

VERSION='1.0.0'
PROCESSING_SHA='acb6f6aecc1cae79cb8e8fec707d03e99ab80b8d3303cf755f6c24654628e4c0'
BLUE='#0072B2'
ORANGE='#D55E00'

def require(ok,message):
    if not ok: raise ValueError(message)

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def write_json(p,obj):
    tmp=p.with_name(p.stem+'.writing'+p.suffix)
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');tmp.replace(p)

def save_figure(fig,path,**kwargs):
    # Publish only closed, complete files, including on external drives.
    tmp=path.with_name(path.stem+'.writing'+path.suffix)
    fig.savefig(tmp,**kwargs);tmp.replace(path)

@contextmanager
def pdf_book(path):
    tmp=path.with_name(path.stem+'.writing'+path.suffix)
    with PdfPages(tmp) as book:yield book
    tmp.replace(path)

def load_processor(path):
    require(path.is_file(),'Missing original Script05: '+str(path))
    require(sha(path)==PROCESSING_SHA,'Expected unchanged Script05 v1.0.0; do not substitute another estimator')
    spec=importlib.util.spec_from_file_location('kitco3_original_05',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

def merged(spans):
    result=[]
    for a,b in sorted(spans):
        if result and a<=result[-1][1]+1e-8: result[-1][1]=max(result[-1][1],b)
        else: result.append([float(a),float(b)])
    return result

def runs(frame):
    """Split on source segment and every absent 1-second core start."""
    for _,g in frame.sort_values(['paired_segment_id','core_start_s']).groupby('paired_segment_id'):
        for ix in np.split(np.arange(len(g)),np.flatnonzero(np.diff(g.core_start_s)>1.00000001)+1):
            if len(ix): yield g.iloc[ix]

def calculate_contexts(proc,reg,segments,data,windows):
    """Reproduce every scored context, checking its PLV against sealed Script05."""
    result={};max_error=0.;n=0
    spec=proc.family('working_200s_40s');core=slice(160,960)
    for r in reg.itertuples(index=False):
        w=windows.loc[windows.pair_id.eq(r.pair_id)].sort_values(['paired_segment_id','core_start_s'])
        result[r.pair_id]=[]
        if r.pair_id not in data:
            require(len(w)==0,'Scores exist without paired signal');continue
        for sr in segments.loc[segments.pair_id.eq(r.pair_id)].itertuples(index=False):
            g=data[r.pair_id].loc[lambda x:x.paired_segment_id.eq(sr.paired_segment_id)].sort_values('cooking_s')
            sw=w.loc[w.paired_segment_id.eq(sr.paired_segment_id)]
            starts=proc.window_starts(sr,r.shared_cooking_s,spec)
            require(np.array_equal(starts,sw.core_start_s.to_numpy()),'Scored window support differs from Script05')
            if not len(sw):continue
            all_a=[];all_b=[];all_d=[]
            for i in range(0,len(sw),64):
                q=sw.iloc[i:i+64]
                x,y=proc.contexts(g,q.core_start_s.to_numpy(),spec)
                a,b=proc.analytic(x),proc.analytic(y)
                a,b=a[:,core],b[:,core]
                observed=proc.plv(a,b)
                error=float(np.max(np.abs(observed-q.PLV.to_numpy())))
                max_error=max(max_error,error)
                require(error<=1e-8,'Recomputed context does not reproduce sealed PLV')
                all_a.append(a.real.astype('float32'));all_b.append(b.real.astype('float32'))
                all_d.append(np.angle(a*np.conj(b)).astype('float32'));n+=len(q)
            result[r.pair_id].append((sw,np.concatenate(all_a),np.concatenate(all_b),np.concatenate(all_d)))
        print(f'Validated contexts: {r.pair_id} ({len(w)} windows)',flush=True)
    require(n==len(windows),'Not all scored windows reproduced')
    return result,max_error

def draw_map(ax,blocks,field,lim,cmap):
    # Each column is one complete scored core. No averaging or selection.
    # Rasterization reduces PDF size while labels and lines remain vector.
    for w,*arrays in blocks:
        for part in runs(w):
            ix=w.index.get_indexer(part.index)
            centers=(part.core_start_s.to_numpy()+100)/60
            x=np.r_[centers-.5/60,centers[-1]+.5/60]
            y=np.linspace(0,200,801)
            ax.pcolormesh(x,y,arrays[field][ix].T,cmap=cmap,vmin=lim[0],vmax=lim[1],
                          shading='flat',rasterized=True)
    ax.set_ylim(0,200)
    ax.set_ylabel('Time within\n200 s core (s)')
    return ScalarMappable(norm=Normalize(*lim),cmap=cmap)

def phase_figure(r,summary,signal_data,segments,windows,blocks,amp_lim):
    fig=plt.figure(figsize=(11.7,13.2))
    gs=fig.add_gridspec(6,2,width_ratios=[1,.022],height_ratios=[1.1,1.15,.36,1.25,1.25,1.25],
                       left=.095,right=.92,bottom=.16,top=.91,hspace=.48,wspace=.07)
    axes=[fig.add_subplot(gs[i,0]) for i in range(6)]
    colors=[BLUE,ORANGE]
    pids=r.pair_id.split('_')[-2:]
    fig.suptitle(f'{r.pair_id}  |  Cardiac phase consistency during cooking',x=.095,ha='left',y=.975,fontsize=15,fontweight='bold')
    fig.text(.095,.944,'LF IBI phase: 0.04–0.15 Hz; 200 s scoring cores; 40 s real context on each side',fontsize=10)
    for a in axes:
        a.set_xlim(0,max(r.shared_cooking_s/60,.1))
        a.spines[['top','right']].set_visible(False)
        a.tick_params(labelsize=9)
    if signal_data is not None:
        for _,g in signal_data.groupby('paired_segment_id',sort=True):
            for j in [1,2]:
                axes[0].plot(g.cooking_s/60,g[f'ibi_{j}_s']*1000,color=colors[j-1],lw=.65,
                             label=f'PID {pids[j-1]}' if _==signal_data.paired_segment_id.iloc[0] else None)
        axes[0].legend(frameon=False,ncol=2,fontsize=9,loc='upper right')
    else: axes[0].text(.5,.5,'No jointly supported IBI grid',transform=axes[0].transAxes,ha='center')
    axes[0].set_ylabel('IBI (ms)')
    axes[0].set_title('A  Paired IBI signals on the exported time axis',loc='left')
    for part in runs(windows):
        axes[1].plot((part.core_start_s+100)/60,part.PLV,color=BLUE,lw=1,marker='.' if len(part)==1 else None)
    axes[1].set_ylim(0,1);axes[1].set_yticks([0,.5,1]);axes[1].set_ylabel('PLV')
    axes[1].set_title('B  PLV at each scoring-core midpoint',loc='left')
    if len(windows):
        axes[1].axhline(summary.mean_local_PLV,color='#444444',ls='--',lw=.85)
        axes[1].text(.99,.93,f'Mean local PLV = {summary.mean_local_PLV:.3f}\n{len(windows)} overlapping windows',
                          transform=axes[1].transAxes,ha='right',va='top',fontsize=9,
                          bbox=dict(facecolor='white',edgecolor='none',alpha=.85))
    else:
        reason='No jointly supported IBI grid' if signal_data is None else 'No continuous segment supports a 200 s core plus 40 s context per side'
        axes[1].text(.5,.52,reason,ha='center',va='center',wrap=True,fontsize=10,transform=axes[1].transAxes)
    for s in segments.itertuples(index=False):
        axes[2].broken_barh([(s.first_grid_s/60,(s.last_grid_s-s.first_grid_s+.25)/60)],(.62,.23),facecolors='#999999')
    spans=merged([(a,a+200) for a in windows.core_start_s])
    for a,b in spans: axes[2].broken_barh([(a/60,(b-a)/60)],(.15,.23),facecolors=BLUE)
    axes[2].set_ylim(0,1);axes[2].set_yticks([.265,.735],['Scored core','Paired grid'],fontsize=8)
    axes[2].spines[['left','bottom']].set_visible(False)
    axes[2].set_title('C  Available paired signal and union of scored cores',loc='left')
    for k,label in enumerate(['D  Partner 1 filtered IBI in every scored core','E  Partner 2 filtered IBI in every scored core','F  Phase difference in every scored core (partner 1 − partner 2)']):
        a=axes[k+3];a.set_title(label,loc='left')
        lim=(-amp_lim,amp_lim) if k<2 else (-np.pi,np.pi)
        cmap='RdBu_r' if k<2 else 'twilight_shifted'
        sm=draw_map(a,blocks,k,lim,cmap)
        cb=fig.colorbar(sm,cax=fig.add_subplot(gs[k+3,1]))
        cb.ax.tick_params(labelsize=8)
        if k<2:cb.set_label('Context-standardized units',fontsize=8)
        else:cb.set_ticks([-np.pi,0,np.pi]);cb.set_ticklabels(['−π','0','π']);cb.set_label('Radians',fontsize=8)
        if not len(windows):a.text(.5,.5,'No eligible scoring cores',transform=a.transAxes,ha='center')
    for i,a in enumerate(axes):
        a.set_xlabel('Cooking time (min)' if i==2 else ('Scoring-core midpoint in cooking (min)' if i==5 else ''),fontsize=9)
    fig.text(.095,.064,'Heatmap columns are separate 200 s cores, not a single continuous filtered trace. White areas have no scored window.\n'
             'PLV uses the full Hilbert phase of each context; colors show only the 200 s scoring core. No smoothing or window selection.\n'
             'Overlapping windows are dependent. Dashed line: arithmetic mean of local PLVs, not a significance threshold.',fontsize=8.2,linespacing=1.5)
    fig.text(.095,.022,'Descriptive, conditional estimates. Physical sensor-clock offset/drift remains unverified. Final inclusion is not frozen.',fontsize=8,color='#444444')
    return fig

def overview(reg,summaries,start,stop):
    rs=reg.iloc[start:stop];fig,axes=plt.subplots(1,2,figsize=(11.7,9),sharey=True,gridspec_kw={'width_ratios':[1,1.1]})
    fig.subplots_adjust(left=.22,right=.96,top=.87,bottom=.20,wspace=.16)
    y=np.arange(len(rs));labels=[]
    for i,r in enumerate(rs.itertuples(index=False)):
        s=summaries.loc[r.pair_id];labels.append(r.pair_id)
        if s.windows:
            axes[0].plot(s.mean_local_PLV,i,'o',color=BLUE,ms=5)
        else:axes[0].text(.025,i,'Unavailable',va='center',fontsize=8,color='#666666')
        axes[1].barh(i,r.shared_cooking_s/60,height=.58,color='#DDDDDD')
        axes[1].barh(i,s.core_union_s/60,height=.32,color=BLUE)
    axes[0].set_xlim(0,1);axes[0].set_yticks(y,labels,fontsize=8);axes[0].invert_yaxis()
    axes[0].set_xlabel('Mean local PLV');axes[1].set_xlabel('Duration (min)')
    axes[0].set_title('A  One summary per pair phase',loc='left');axes[1].set_title('B  Shared cooking and scored-core coverage',loc='left')
    for a in axes:a.spines[['top','right']].set_visible(False);a.grid(axis='x',alpha=.15)
    axes[1].set_xlim(0,reg.shared_cooking_s.max()/60*1.03)
    fig.suptitle(f'All pair phases: {start+1}–{stop} of {len(reg)}',x=.22,ha='left',fontsize=15,fontweight='bold')
    fig.text(.22,.915,'Identity order; no ranking or condition comparison',fontsize=10)
    fig.text(.22,.055,'Blue duration: union of scored cores. Grey: shared cooking period. No error bars are inferred from overlapping windows.\n'
             'Conditional LF-IBI phase consistency; unverified physical clock mapping; final inclusion not frozen.',fontsize=8.5,linespacing=1.5)
    return fig

README='''# KITCO3 all-phase PLV figures, Script05b v1.0.0

These are publication-format descriptive figures, not final inferential results.
All 71 pair phases are retained in identity order. All 10,434 currently scored
windows are displayed. The 23 phases without a working score retain a status
figure. No high/low example selection, smoothing, new QC or condition test occurs.

## Panels in every pair-phase figure
A: Both interpolated IBI series, with gaps preserved. Values are in milliseconds.
B: Sealed Script05 window PLV at each 200-second core midpoint; y-axis is 0–1.
Dashed mean is the arithmetic mean of local PLVs. It is not a null threshold.
C: Paired sampled support and the union of scored cores. Coverage is not the
sum of overlapping window durations. Blank space does not mean PLV equals zero.
D/E: Filtered IBI within EVERY scored core, using the original context-specific
processing. Each heatmap column is a separate core. Rows are seconds within it.
F: Wrapped partner-one minus partner-two phase within every scored core, with
a circular colormap from -pi to pi. Consistency within a column determines PLV;
zero phase difference is not required for high PLV. Adjacent columns overlap.

Context phase cannot accurately be represented by filtering the whole interaction
once. This script imports the unchanged Script05 implementation, repeats its
context transforms and checks every reconstructed PLV against the sealed scores
to an absolute tolerance of 1e-8. It DOES NOT rerun QC or synthetic simulations.
No contexts cross an unsupported segment. No new clock correction is fitted.
Heatmaps retain all 800 core samples per column, without temporal downsampling.
Raster rendering limits visible pixel detail; PDFs retain vector labels and lines.

## Processing and scales
4 Hz midpoint-interpolated IBI; 200 s cores; 40 s context per side; 1 s start step;
0.04–0.15 Hz order-3 Butterworth SOS forward/backward filtering; linear detrend
and population-SD standardization of each 280 s context; FFT Hilbert phase.
Filtered units refer to standardized pre-filter context, not ECG amplitude.
Both filtered heatmaps use one symmetric global color limit across all phases,
chosen as the maximum absolute filtered value. No color clipping is applied.
IBI y-ranges adapt per phase for readability; PLV and phase scales are fixed.
Overview durations share a common scale. Overview pages use one point per phase,
with unavailable values explicitly labelled. No window-based SEM/CI is supplied.

## Interpretation limits
All results assume exported time represents common physical time. Sensor offset
and drift remain unverified. PLV is band-limited IBI phase consistency, not proof
of interpersonal causation, common cardiac oscillation or a condition effect.
No arbitrary strong/moderate synchrony cutoffs are drawn. These are method
diagnostics from the working specification, not a final included cohort.

## Outputs
pairs/: 71 individual PDF and PNG figures (PNG 300 dpi).
overview/: three identity-ordered PDF/PNG summary pages, covering all 71 phases.
05b_PLV_all_phases_part_01/02/03.pdf: readable 24/24/23-page pair-phase books.
05b_PLV_overview.pdf: combined three-page overview.
05b_figure_index.csv: one row per phase, status and figure path.
05b_RUN_STATUS.json: completion written LAST, input/output hashes, environment,
number of windows reproduced, and maximum PLV discrepancy.

## Reproduce on Emma's Mac
Save this script next to the ORIGINAL unchanged Script05 v1.0.0 in:
/Volumes/One Touch/KITCO3_drive/scripts
Run:
python3 "/Volumes/One Touch/KITCO3_drive/scripts/05b_KITCO3_PLV_all_phase_plots_v1_0_0_EG.py"

Defaults read outputs/04_dyadic_signals_v1_0_0 and
outputs/05_PLV_feasibility_v1_0_0 and write a NEW
outputs/05b_PLV_all_phase_plots_v1_0_0. If it already exists, choose a fresh
--output-dir. Optional --project-root, --script04-dir, --script05-dir,
--processing-script and --output-dir take explicit paths.
Requires Python >=3.10, numpy, pandas, scipy, matplotlib. Inputs are read-only.
Do not edit Script05 to satisfy a checksum. Use the exact original script.

Suggested caption: Conditional phase consistency of partners' LF-filtered IBI
signals during cooking. PLV was calculated within overlapping 200 s cores with
40 s real context on each side. Every supported scoring core is represented;
window centers are shown on the cooking-time axis. White regions indicate no
scored core. Estimates assume common exported time; physical clock offset/drift
has not been independently verified. Overlapping windows are not independent.
'''

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root',type=Path,default=Path('/Volumes/One Touch/KITCO3_drive'))
    for key in ['script04-dir','script05-dir','processing-script','output-dir']:p.add_argument('--'+key,type=Path)
    a=p.parse_args();root=a.project_root.expanduser().resolve()
    src4=(a.script04_dir or root/'outputs/04_dyadic_signals_v1_0_0').resolve()
    src5=(a.script05_dir or root/'outputs/05_PLV_feasibility_v1_0_0').resolve()
    out=(a.output_dir or root/'outputs/05b_PLV_all_phase_plots_v1_0_0').resolve()
    require(not out.exists(),'Output already exists. Preserve it and choose a new --output-dir.')
    processor=(a.processing_script or Path(__file__).with_name('05_KITCO3_PLV_feasibility_v1_0_0_EG.py')).resolve()
    proc=load_processor(processor)
    state,reg,segments,data=proc.load_inputs(src4)
    s5=json.loads((src5/'05_RUN_STATUS.json').read_text())
    require(s5['status']=='CONDITIONAL_PLV_FEASIBILITY_COMPLETE','Script05 is not complete')
    require(s5['script_sha256']==PROCESSING_SHA,'Script05 estimator mismatch')
    require(s5['source_status_sha256']==sha(src4/'04_RUN_STATUS.json'),'Script05 used a different Script04 input')
    for n,h in s5['output_sha256'].items():
        path=(src5/n).resolve();require(path.is_relative_to(src5),'Unsafe source path')
        require(path.is_file() and sha(path)==h,'Script05 output changed: '+n)
    tracked={str(src4/'04_RUN_STATUS.json'):sha(src4/'04_RUN_STATUS.json'),str(src5/'05_RUN_STATUS.json'):sha(src5/'05_RUN_STATUS.json'),str(processor):sha(processor)}
    for src,seals in [(src4,state['output_sha256']),(src5,s5['output_sha256'])]:
        for n,h in seals.items():tracked[str(src/n)]=h
    windows=pd.read_csv(src5/'05_working_window_diagnostics.csv')
    require(windows.status.eq('CONDITIONAL_DIAGNOSTIC_SCORE').all(),'Unscored or degenerate contexts require separate review')
    require(not windows.duplicated(['pair_id','paired_segment_id','core_start_s']).any(),'Duplicate scored window')
    require(windows.PLV.between(0,1).all(),'PLV outside 0–1')
    summaries=pd.read_csv(src5/'05_pair_phase_method_summaries.csv')
    summaries=summaries.loc[summaries.family.eq('working_200s_40s')].set_index('pair_id')
    require(set(summaries.index)==set(reg.pair_id) and summaries.index.is_unique,'Incomplete summary register')
    require(set(windows.pair_id).issubset(set(reg.pair_id)),'Orphan windows')
    for r in reg.itertuples(index=False):
        w=windows.loc[windows.pair_id.eq(r.pair_id)];s=summaries.loc[r.pair_id]
        require(len(w)==s.windows,'Window count differs from sealed summary')
        if len(w):
            require(abs(w.PLV.mean()-s.mean_local_PLV)<1e-8,'Mean PLV differs from summary')
            coverage=sum(b-a for a,b in merged([(x,x+200) for x in w.core_start_s]))
            require(abs(coverage-s.core_union_s)<1e-7,'Core coverage mismatch')
    reg=reg.sort_values('pair_id').reset_index(drop=True)
    contexts,error=calculate_contexts(proc,reg,segments,data,windows)
    amp_lim=max(float(np.max(np.abs(array))) for blocks in contexts.values() for _,x,y,d in blocks for array in [x,y])
    amp_lim=float(np.ceil(amp_lim*10)/10)
    out.mkdir(parents=True);(out/'pairs').mkdir();(out/'overview').mkdir()
    (out/'05b_README.md').write_text(README)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':10,
                         'pdf.fonttype':42,'ps.fonttype':42,'savefig.facecolor':'white'})
    index=[]
    for first in range(0,len(reg),24):
        with pdf_book(out/f'05b_PLV_all_phases_part_{first//24+1:02d}.pdf') as book:
            for r in reg.iloc[first:first+24].itertuples(index=False):
                w=windows.loc[windows.pair_id.eq(r.pair_id)]
                fig=phase_figure(r,summaries.loc[r.pair_id],data.get(r.pair_id),segments.loc[segments.pair_id.eq(r.pair_id)],w,contexts[r.pair_id],amp_lim)
                name=f'pairs/{r.pair_id}_PLV'
                save_figure(fig,out/(name+'.png'),dpi=300);save_figure(fig,out/(name+'.pdf'),dpi=300);book.savefig(fig,dpi=300);plt.close(fig)
                status='SCORED' if len(w) else ('NO_PAIRED_GRID' if r.pair_id not in data else 'INSUFFICIENT_CONTINUOUS_CONTEXT')
                index.append(dict(pair_id=r.pair_id,status=status,windows=len(w),mean_local_PLV=summaries.loc[r.pair_id].mean_local_PLV,
                                  shared_cooking_s=r.shared_cooking_s,core_union_s=summaries.loc[r.pair_id].core_union_s,pdf=name+'.pdf',png=name+'.png'))
                print(f'Figures {len(index)}/{len(reg)}: {r.pair_id}',flush=True)
    with pdf_book(out/'05b_PLV_overview.pdf') as book:
        for start in range(0,len(reg),24):
            fig=overview(reg,summaries,start,min(start+24,len(reg)))
            stem=out/'overview'/f'05b_PLV_overview_{start//24+1:02d}'
            save_figure(fig,stem.with_suffix('.pdf'));save_figure(fig,stem.with_suffix('.png'),dpi=300);book.savefig(fig);plt.close(fig)
    pd.DataFrame(index).to_csv(out/'05b_figure_index.csv',index=False)
    for n,h in tracked.items():require(sha(n)==h,'Input changed during plotting: '+n)
    status=dict(script=Path(__file__).name,plotting_script_sha256=sha(__file__),version=VERSION,status='ALL_PHASE_DESCRIPTIVE_PLOTS_COMPLETE',
                paired_phases=len(reg),scored_phases=sum(x['windows']>0 for x in index),scored_windows_reproduced=len(windows),
                maximum_absolute_PLV_reproduction_error=error,global_filtered_color_limit=amp_lim,
                physical_clock_verified=False,final_synchrony_dataset_frozen=False,condition_effect_computed=False,
                input_sha256=tracked,environment=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,matplotlib=matplotlib.__version__),
                output_sha256={str(f.relative_to(out)):sha(f) for f in sorted(out.rglob('*')) if f.is_file() and '.writing.' not in f.name})
    write_json(out/'05b_RUN_STATUS.json',status)
    print(json.dumps({k:v for k,v in status.items() if k not in ['input_sha256','output_sha256']},indent=2))

if __name__=='__main__':
    try:main()
    except Exception as e:
        print('PLOTTING STOPPED: '+str(e)+'; no completion status is written for a failed run.',file=sys.stderr);sys.exit(1)
