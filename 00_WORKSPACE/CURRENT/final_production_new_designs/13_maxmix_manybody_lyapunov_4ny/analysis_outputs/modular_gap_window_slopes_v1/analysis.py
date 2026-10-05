# %% markdown
# Rates from trajectory-resolved modular-gap slopes
# Fit $g_s(T)=2\widehat\Delta_s T+b_s$, with a free intercept, independently for each trajectory and in each inclusive window $T/N_y\in[1,2],[2,3],[3,4]$.
# The raw modular gap is $g_s(T)=\min_j|\log[(1-\nu_{j,s})/\nu_{j,s}]|$. Fit this instantaneous minimum; no eigenmode tracking is assumed.
# Geometry: Nx=20, hard walls x=5,15; Ny=20,24,30,36,44,56,60; 100 independent Born trajectories per size; maximally mixed active-slab initialization; alpha1=1, alpha2=30, overcomplete Wannier range nshell=1. Canonical acquisition entry point: classA_U1FGTN_gpu.run_markov_circuit. No new simulation.
# Fit by ordinary least squares with equal weights for saved times within a window. Average trajectory rates after fitting. SEMs are across trajectories, not time points. Negative finite-window slopes from fluctuating trajectories are retained without clipping.
# Resample whole trajectories for 5000 bootstrap replicates, jointly across windows at each size. Paired differences preserve correlations between windows. Agreement within uncertainty is not proof of asymptotic convergence.
# %%
import os
CPU_RANGE=(40,43)
selected=list(range(CPU_RANGE[0],CPU_RANGE[1]+1))
assert set(selected).issubset(os.sched_getaffinity(0))
os.sched_setaffinity(0,selected)
for k in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[k]='1'
print('Allocated CPUs:',selected)
# %% markdown
# Configuration and verified inputs
# Use the previously verified all-time gap cache. Its provenance binds all 140 original raw shards to completion receipts. Also verify and compare the prior many-body first-level gap slopes, accounting explicitly for the factor of two.
# %%
from pathlib import Path
import hashlib,json,shutil,subprocess
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from IPython.display import display,Image
OUT=Path.cwd();BUNDLE=OUT.parents[1];SOURCE=OUT.parent/'gap_closure_time_and_size_v1'
PRIOR=OUT.parent/'hard_4ny_dynamical_critical_v2_ungated/trajectory_window_slopes.csv'
SIZES=[20,24,30,36,44,56,60];SAMPLES=100;WINDOWS=[(1,2),(2,3),(3,4)]
BOOTSTRAPS=5000;SEED=2026092804
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
manifest=json.loads((SOURCE/'completion_manifest.json').read_text());assert manifest['status']=='complete'
inputs=[]
for name in ['trajectory_gap_timeseries.npz','input_provenance.json','diagnostics.json']:
    p=SOURCE/name;r=manifest['files'][name]
    assert p.stat().st_size==r['bytes'] and sha(p)==r['sha256']
    inputs.append(dict(path=str(p),**r))
provenance=json.loads((SOURCE/'input_provenance.json').read_text())
assert provenance['configuration']['Ny_values']==SIZES and provenance['configuration']['samples_per_size']==SAMPLES
prior=pd.read_csv(PRIOR);inputs.append(dict(path=str(PRIOR),bytes=PRIOR.stat().st_size,sha256=sha(PRIOR)))
metadata=dict(Nx=20,Ny_values=SIZES,samples_per_size=SAMPLES,windows=WINDOWS,
 estimator='half the OLS slope of each trajectory raw modular gap, free intercept; then average trajectories',
 uncertainty='trajectory SEM; paired whole-trajectory bootstrap for differences and fit intervals',
 bootstrap_replicates=BOOTSTRAPS,seed=SEED,canonical_dynamics_entry_point='classA_U1FGTN_gpu.run_markov_circuit',
 simulations=0)
display(pd.Series(metadata))
data={};rows=[];summaries=[];rates={};intercepts={};boot={};raw_means=[]
rng=np.random.default_rng(SEED);max_prior_error=0.;max_mean_linearity_error=0.
with np.load(SOURCE/'trajectory_gap_timeseries.npz') as z:
    assert np.array_equal(z['Ny_values'],SIZES) and np.array_equal(z['sample_ids'],np.arange(SAMPLES))
    for ny in tqdm(SIZES,desc='Fit individual modular-gap slopes',unit='size'):
        times=z[f'times_Ny{ny:03d}'];raw=z[f'gaps_Ny{ny:03d}']*(2*times[None,:])
        assert raw.shape==(SAMPLES,len(times)) and np.isfinite(raw).all()
        data[ny]=(times,raw)
        weights=rng.multinomial(SAMPLES,np.full(SAMPLES,1/SAMPLES),size=BOOTSTRAPS)/SAMPLES
        for T,m,se in zip(times,raw.mean(0),raw.std(0,ddof=1)/np.sqrt(SAMPLES)):
            raw_means.append(dict(Ny=ny,T=int(T),aspect=float(T/ny),mean_raw_gap=float(m),sem=float(se)))
        for wi,(start,stop) in enumerate(WINDOWS):
            mask=(times>=start*ny)&(times<=stop*ny);t=times[mask].astype(float);y=raw[:,mask]
            assert t[0]==start*ny and t[-1]==stop*ny and len(t)>=3
            centered=t-t.mean()
            slope=y@centered/(centered@centered);rate=slope/2;b=y.mean(1)-slope*t.mean()
            predicted=b[:,None]+slope[:,None]*t
            residual=y-predicted;ss=(residual**2).sum(1);sst=((y-y.mean(1)[:,None])**2).sum(1)
            r2=np.divide(ss,sst,out=np.full_like(ss,np.nan),where=sst>0);r2=1-r2
            assert np.isfinite(rate).all() and np.isfinite(b).all()
            synthetic=2*.03*t-1.7
            assert abs(synthetic@centered/(centered@centered)/2-.03)<1e-12
            mean_rate=y.mean(0)@centered/(centered@centered)/2
            max_mean_linearity_error=max(max_mean_linearity_error,abs(float(mean_rate-rate.mean())))
            prior_window=f'W{wi+1}_'+('Ny_to_2Ny' if wi==0 else '2Ny_to_3Ny' if wi==1 else '3Ny_to_4Ny')
            check=prior[(prior.Ny==ny)&(prior.window==prior_window)].sort_values('sample_index')
            assert check.sample_index.tolist()==list(range(SAMPLES))
            error=float(np.max(abs(rate-check.gap1.to_numpy()/2)))
            max_prior_error=max(max_prior_error,error);assert error<1e-10
            rates[ny,wi]=rate;intercepts[ny,wi]=b;boot[ny,wi]=weights@rate
            for sid in range(SAMPLES):
                rows.append(dict(Ny=ny,sample_id=sid,window=wi+1,start=start,stop=stop,
                 points=len(t),rate=float(rate[sid]),raw_slope=float(slope[sid]),intercept=float(b[sid]),
                 r_squared=float(r2[sid]),residual_rms=float(np.sqrt(ss[sid]/len(t)))))
            lo,hi=np.quantile(boot[ny,wi],[.025,.975])
            summaries.append(dict(Ny=ny,window=wi+1,start=start,stop=stop,samples=SAMPLES,points=len(t),
             mean_rate=float(rate.mean()),sem=float(rate.std(ddof=1)/np.sqrt(SAMPLES)),
             mean_intercept=float(b.mean()),intercept_sem=float(b.std(ddof=1)/np.sqrt(SAMPLES)),
             scaled_mean=float(ny*rate.mean()),scaled_sem=float(ny*rate.std(ddof=1)/np.sqrt(SAMPLES)),
             mean_ci_low=float(lo),mean_ci_high=float(hi),negative_sample_slopes=int((rate<0).sum()),
             median_sample_r_squared=float(np.nanmedian(r2))))
samples=pd.DataFrame(rows);summary=pd.DataFrame(summaries);raw_summary=pd.DataFrame(raw_means)
samples.to_csv(OUT/'trajectory_window_fits.csv',index=False)
summary.to_csv(OUT/'window_rate_summary.csv',index=False)
raw_summary.to_csv(OUT/'raw_modular_gap_summary.csv',index=False)
np.savez_compressed(OUT/'window_fit_arrays.npz',
 **{f'rates_Ny{ny:03d}':np.stack([rates[ny,w] for w in range(3)],axis=1) for ny in SIZES},
 **{f'intercepts_Ny{ny:03d}':np.stack([intercepts[ny,w] for w in range(3)],axis=1) for ny in SIZES},
 **{f'bootstrap_Ny{ny:03d}':np.stack([boot[ny,w] for w in range(3)],axis=1) for ny in SIZES},
 sample_ids=np.arange(SAMPLES),Ny_values=SIZES,windows=WINDOWS)
display(summary[['Ny','window','mean_rate','sem','mean_intercept','negative_sample_slopes']])
# %% markdown
# Window stability and finite-size fits
# Compare rates using paired trajectory differences. Also fit the mean rates to d0+a/Ny, with a free intercept, both with all sizes and omitting the smallest two. These are finite-window rate estimates.
# %%
changes=[];fitrows=[]
for ny in SIZES:
    for first,last in [(0,1),(1,2)]:
        d=rates[ny,last]-rates[ny,first];bd=boot[ny,last]-boot[ny,first]
        lo,hi=np.quantile(bd,[.025,.975])
        changes.append(dict(Ny=ny,earlier_window=first+1,later_window=last+1,
         difference=float(d.mean()),paired_sem=float(d.std(ddof=1)/np.sqrt(SAMPLES)),
         ci_low=float(lo),ci_high=float(hi),contains_zero=bool(lo<=0<=hi),
         fractional_change=float(rates[ny,last].mean()/rates[ny,first].mean()-1)))
changes=pd.DataFrame(changes);changes.to_csv(OUT/'paired_window_changes.csv',index=False)
fit_objects={}
for wi in range(3):
    for minimum in [20,30]:
        s=summary[(summary.window==wi+1)&(summary.Ny>=minimum)].sort_values('Ny')
        x=1/s.Ny.to_numpy();y=s.mean_rate.to_numpy();se=s['sem'].to_numpy()
        A=np.column_stack([np.ones(len(x)),x]);C=np.linalg.inv(A.T@(A/se[:,None]**2))
        op=C@(A.T/se[None,:]**2);beta=op@y
        bs=np.column_stack([boot[int(ny),wi] for ny in s.Ny])@op.T
        lo,hi=np.quantile(bs,[.025,.975],axis=0)
        fitrows.append(dict(window=wi+1,Ny_min=minimum,intercept=float(beta[0]),
         intercept_ci_low=float(lo[0]),intercept_ci_high=float(hi[0]),slope=float(beta[1]),
         slope_ci_low=float(lo[1]),slope_ci_high=float(hi[1]),
         chi_squared=float(np.sum(((y-A@beta)/se)**2)),dof=len(x)-2))
        fit_objects[wi,minimum]=(beta,bs)
size_fits=pd.DataFrame(fitrows);size_fits.to_csv(OUT/'finite_size_fits.csv',index=False)
display(changes);display(size_fits)
# %% markdown
# Main figure
# Each rate is half the trajectory-resolved free-intercept slope. Error bars are one SEM over the 100 trajectories; the right panel multiplies by circumference. Lines connect data and are not size fits.
# %%
import matplotlib as mpl
import matplotlib.pyplot as plt
shutil.copytree(SOURCE/'latex_support',OUT/'latex_support',dirs_exist_ok=True)
os.environ['TEXINPUTS']=str(OUT/'latex_support')+os.pathsep+os.environ.get('TEXINPUTS','')
mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['CMU Sans Serif'],'font.size':8,
 'axes.labelsize':9,'axes.titlesize':9,'text.usetex':True,'text.latex.preamble':r'\usepackage{amsmath,amssymb}',
 'xtick.direction':'in','ytick.direction':'in','legend.frameon':False,'legend.fontsize':8})
styles=[('#d62728','^',':'),('#2ca02c','s','--'),('#1565c0','o','-')]
fig,axes=plt.subplots(1,2,figsize=(7.05,3.5))
for wi,((start,stop),(color,marker,ls)) in enumerate(zip(WINDOWS,styles)):
    s=summary[summary.window==wi+1].sort_values('Ny')
    label=r'$'+str(start)+r'\le T/N_y\le '+str(stop)+'$'
    for ax,value,error in [(axes[0],'mean_rate','sem'),(axes[1],'scaled_mean','scaled_sem')]:
        ax.errorbar(s.Ny,s[value],yerr=s[error],color=color,marker=marker,ls=ls,mfc='white',
                    ms=3.5,capsize=2,lw=1,label=label)
axes[0].set(xlabel=r'$N_y$',ylabel=r'$\overline{\widehat{\Delta}}$',ylim=(0,None),title='Rate from the raw-gap slope')
axes[1].set(xlabel=r'$N_y$',ylabel=r'$N_y\,\overline{\widehat{\Delta}}$',ylim=(0,None),title='Rescaled rate')
axes[0].legend(loc='upper right');axes[1].legend(loc='lower right')
for ax,label in zip(axes,['(a)','(b)']):
    ax.tick_params(top=True,right=True);ax.text(-.17,1.04,label,transform=ax.transAxes)
fig.suptitle(r'$g_s(T)=2\widehat{\Delta}_sT+b_s$; free intercept; $N_x=20$; hard walls'+'\n'+
 r'Maximally mixed initial active slab; 100 Born trajectories per size; error bars: one SEM',fontsize=9)
fig.tight_layout(pad=1)
fig.savefig(OUT/'gap_rates_from_window_slopes.pdf');plt.close(fig)
subprocess.run(['pdftoppm','-r','300','-singlefile','-png',str(OUT/'gap_rates_from_window_slopes.pdf'),str(OUT/'gap_rates_from_window_slopes')],check=True)
display(Image(filename=str(OUT/'gap_rates_from_window_slopes.png'),width=1200))
# %% markdown
# Raw-gap fit diagnostics for three representative sizes
# Solid gray data and one-SEM bands show the trajectory-averaged raw modular gap. Colored segments show the average fitted lines. Because OLS is linear, this equals fitting the mean curve on the identical time grid, but uncertainty still comes from individual trajectory fits.
# %%
fig,axes=plt.subplots(1,3,figsize=(7.05,2.8),sharey=True)
for ax,ny in zip(axes,[20,36,60]):
    s=raw_summary[raw_summary.Ny==ny]
    ax.fill_between(s.aspect,s.mean_raw_gap-s['sem'],s.mean_raw_gap+s['sem'],color='.5',alpha=.16,lw=0)
    ax.plot(s.aspect,s.mean_raw_gap,color='.3',lw=1)
    for wi,((start,stop),(color,marker,ls)) in enumerate(zip(WINDOWS,styles)):
        x=np.linspace(start,stop,80)
        ax.plot(x,2*rates[ny,wi].mean()*ny*x+intercepts[ny,wi].mean(),color=color,ls=ls,lw=1.5,
         label=r'$['+str(start)+','+str(stop)+']$')
    ax.set(xlabel=r'$T/N_y$',title=r'$N_y='+str(ny)+'$',xlim=(.9,4.05))
    ax.tick_params(top=True,right=True)
axes[0].set_ylabel(r'$\overline{g(T)}$')
axes[0].legend(loc='upper left',fontsize=7,title=r'Fit $T/N_y$')
for ax,label in zip(axes,['(a)','(b)','(c)']):ax.text(-.18,1.04,label,transform=ax.transAxes)
fig.tight_layout()
fig.savefig(OUT/'raw_modular_gap_window_fits.pdf');plt.close(fig)
subprocess.run(['pdftoppm','-r','300','-singlefile','-png',str(OUT/'raw_modular_gap_window_fits.pdf'),str(OUT/'raw_modular_gap_window_fits')],check=True)
display(Image(filename=str(OUT/'raw_modular_gap_window_fits.png'),width=1200))
# %% markdown
# Diagnostics
# No trajectory is removed based on fitted sign or R-squared. The exact same prior many-body data give the same estimator after converting the squared-singular-value slope convention by a factor of two. The endpoint estimator and the free-intercept slope estimator differ by construction; their numerical difference is not a regression.
# %%
diagnostics=dict(trajectories=700,trajectory_window_fits=len(samples),
 prior_manybody_half_gap_max_error=max_prior_error,mean_curve_slope_linearity_error=max_mean_linearity_error,
 negative_window_slopes=int((samples.rate<0).sum()),dropped_trajectories=0,
 last_two_windows_zero_in_paired_95ci_by_size=changes[changes.earlier_window==2][['Ny','contains_zero']].to_dict('records'),
 source_cache_diagnostics=json.loads((SOURCE/'diagnostics.json').read_text()),
 interpretation='Window-to-window rate stability tested with paired trajectories; statistical consistency is not proof of asymptotic convergence.')
(OUT/'diagnostics.json').write_text(json.dumps(diagnostics,indent=2)+'\n')
(OUT/'input_provenance.json').write_text(json.dumps(dict(configuration=metadata,inputs=inputs,
 original_acquisition_provenance=provenance),indent=2)+'\n')
display(pd.Series(diagnostics))
print('Complete: 2100 trajectory fits, paired window changes, size fits and two figures.')
