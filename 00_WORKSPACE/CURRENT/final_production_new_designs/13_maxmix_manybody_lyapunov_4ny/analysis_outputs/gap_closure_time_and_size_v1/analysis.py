# %% markdown
# Purification gap: time convergence and finite-size extrapolation
# Analyze 100 Born trajectories per size, Nx=20, Ny=20,24,30,36,44,56,60; hard walls at x=5,15; maximally mixed active slab, alpha1=1, alpha2=30, overcomplete Wannier range nshell=1. No new dynamics.
#
# For each trajectory, compute $\Delta_s(T)=\min_j|\log[(1-\nu_{j,s})/\nu_{j,s}]|/(2T)$ before averaging. This is the finite-time Lyapunov gap. Exact-cap modes retain infinite costs.
# At fixed $u=T/N_y=2,3,4$, fit $\overline{\Delta}=d_0(u)+a(u)/N_y$, with an unconstrained intercept. Also test Ny cutoffs 20,24,30 and a quadratic correction $b/N_y^2$. These are finite-aspect-ratio extrapolations, not established infinite-time limits.
# Time-curve bands and data bars: one SEM. Fit intervals: 5000 whole-trajectory bootstrap resamples, paired across times within a size, with original inverse-SEM-squared weights held fixed. Intervals are conditional on the fit model.
# %%
import os
CPU_RANGE=(40,43)
selected=list(range(CPU_RANGE[0],CPU_RANGE[1]+1))
assert set(selected).issubset(os.sched_getaffinity(0))
os.sched_setaffinity(0,selected)
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[key]='1'
print('Allocated CPUs:',selected)
# %% markdown
# Configuration and validation
# Validate every raw shard against its receipt and pinned acquisition identity. Reconstruct the gap at every positive saved time and compare against soft-mode costs and the two leading many-body levels.
# %%
from pathlib import Path
import json,hashlib,sys,subprocess,shutil
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from IPython.display import display,Image
OUT=Path.cwd();BUNDLE=OUT.parents[1];BASE=BUNDLE.parent
sys.path.insert(0,str(BUNDLE))
import analyze_campaign as campaign
SIZES=list(campaign.NY_VALUES);SAMPLES=100;ASPECT_RATIOS=[2,3,4]
NY_MIN_VALUES=[20,24,30];BOOTSTRAPS=5000;BOOTSTRAP_SEED=2026092803
PLOT_MIN_ASPECT=.5
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
metadata=dict(Nx=20,Ny_values=SIZES,samples_per_size=SAMPLES,endpoint='4Ny',
 fit_aspect_ratios=ASPECT_RATIOS,minimum_size_sensitivities=NY_MIN_VALUES,
 gap='min(abs(log((1-nu)/nu)))/(2T)',independent_unit='Born trajectory',
 canonical_dynamics_entry_point='classA_U1FGTN_gpu.run_markov_circuit',
 sampling_revision=campaign.REVISION,configuration_hash=campaign.CONFIGURATION_HASH,
 source_hashes=campaign.SOURCE_HASHES,bootstrap_replicates=BOOTSTRAPS,bootstrap_seed=BOOTSTRAP_SEED,simulations=0)
display(pd.Series(metadata))
paths=sorted((campaign.DATA_ROOT/'results').rglob('*.npz'));assert len(paths)==140
inputs=[];pieces={ny:[] for ny in SIZES};times_by_size={}
max_soft_error=0.;max_manybody_error=0.;max_cap_fraction=0.;raw_max_by_size={ny:0. for ny in SIZES}
for p in tqdm(paths,desc='Verify all saved times',unit='shard'):
    rp=p.with_name(p.stem+'.complete.json');r=json.loads(rp.read_text())
    campaign.validate_completion(r,p)
    assert p.stat().st_size==r['result_bytes'] and sha(p)==r['result_sha256']
    inputs.append(dict(path=str(p),bytes=r['result_bytes'],sha256=r['result_sha256'],receipt_sha256=sha(rp),sample_ids=r['sample_indices']))
    with np.load(p) as z:
        ny=int(z['Ny']);ids=z['sample_indices']
        assert int(z['Nx'])==20 and np.array_equal(ids,r['sample_indices'])
        assert str(z['configuration_hash'])==campaign.CONFIGURATION_HASH
        all_times=z['spectrum_cycles'];assert np.array_equal(all_times,campaign.expected_spectrum_cycles(ny))
        keep=all_times>0;times=all_times[keep];nu=z['occupations'][:,keep,:];caps=z['cap_mask'][:,keep,:]
        assert nu.shape==(5,len(times),22*ny) and np.isfinite(nu).all()
        assert nu.min()>=-campaign.CAP_TOLERANCE and nu.max()<=1+campaign.CAP_TOLERANCE
        assert np.array_equal(caps,(nu<=campaign.CAP_TOLERANCE)|(nu>=1-campaign.CAP_TOLERANCE))
        costs=np.full_like(nu,np.inf);v=nu[~caps];assert np.all((v>0)&(v<1))
        costs[~caps]=np.abs(np.log1p(-v)-np.log(v));raw=costs.min(axis=-1)
        assert np.isfinite(raw).all()
        gap=raw/(2*times[None,:]);soft=z['soft_mode_flip_costs'][:,keep,:].min(axis=-1)
        levels=z['leading_log_sigma2'][:,keep,:2]
        e1=float(np.max(abs(raw-soft)));e2=float(np.max(abs(raw-(levels[:,:,0]-levels[:,:,1]))))
        assert e1<1e-12 and e2<5e-10
        max_soft_error=max(max_soft_error,e1);max_manybody_error=max(max_manybody_error,e2)
        max_cap_fraction=max(max_cap_fraction,float(caps.mean(axis=-1).max()))
        raw_max_by_size[ny]=max(raw_max_by_size[ny],float(raw.max()))
        if ny in times_by_size:assert np.array_equal(times_by_size[ny],times)
        times_by_size[ny]=times;pieces[ny].append((ids.copy(),gap))
gaps={};cache={}
for ny in SIZES:
    ids=np.concatenate([v[0] for v in pieces[ny]]);order=np.argsort(ids)
    assert np.array_equal(ids[order],np.arange(SAMPLES))
    gaps[ny]=np.concatenate([v[1] for v in pieces[ny]],axis=0)[order]
    cache[f'times_Ny{ny:03d}']=times_by_size[ny];cache[f'gaps_Ny{ny:03d}']=gaps[ny]
np.savez_compressed(OUT/'trajectory_gap_timeseries.npz',**cache,Ny_values=SIZES,sample_ids=np.arange(SAMPLES))
(OUT/'input_provenance.json').write_text(json.dumps(dict(configuration=metadata,inputs=inputs,
 source_analysis_script=dict(path=str(Path(campaign.__file__)),sha256=sha(Path(campaign.__file__)))),indent=2)+'\n')
# %% markdown
# Time-dependent statistics
# Resample complete trajectories, retaining temporal correlations. Paired differences compare the same trajectory at two observation times.
# %%
rng=np.random.default_rng(BOOTSTRAP_SEED)
time_rows=[];endpoint_rows=[];changes=[];bootmeans={};estimates={}
for ny in SIZES:
    times=times_by_size[ny];g=gaps[ny];m=g.mean(0);sem=g.std(0,ddof=1)/np.sqrt(SAMPLES)
    for j,T in enumerate(times):
        time_rows.append(dict(Ny=ny,T=int(T),aspect=float(T/ny),samples=SAMPLES,
         mean_gap=float(m[j]),sem=float(sem[j]),scaled_mean=float(ny*m[j]),scaled_sem=float(ny*sem[j])))
    counts=rng.multinomial(SAMPLES,np.full(SAMPLES,1/SAMPLES),size=BOOTSTRAPS)
    for u in ASPECT_RATIOS:
        j=int(np.flatnonzero(times==u*ny)[0]);values=g[:,j];estimates[ny,u]=values
        bootmeans[ny,u]=counts@values/SAMPLES
        endpoint_rows.append(dict(Ny=ny,aspect=u,T=u*ny,samples=SAMPLES,mean_gap=float(m[j]),sem=float(sem[j])))
    for start in [2,3]:
        a=estimates[ny,start];b=estimates[ny,4];d=b-a
        lo,hi=np.quantile(bootmeans[ny,4]/bootmeans[ny,start]-1,[.025,.975])
        changes.append(dict(Ny=ny,from_aspect=start,to_aspect=4,mean_difference=float(d.mean()),
         paired_sem=float(d.std(ddof=1)/np.sqrt(SAMPLES)),fractional_increase=float(b.mean()/a.mean()-1),
         fractional_increase_ci_low=float(lo),fractional_increase_ci_high=float(hi)))
time_summary=pd.DataFrame(time_rows);endpoints=pd.DataFrame(endpoint_rows);time_changes=pd.DataFrame(changes)
time_summary.to_csv(OUT/'time_summary.csv',index=False)
endpoints.to_csv(OUT/'endpoint_summary.csv',index=False)
time_changes.to_csv(OUT/'paired_time_changes.csv',index=False)
previous=pd.read_csv(BUNDLE/'analysis_outputs/endpoint_gap_definitions_v1/endpoint_gap_definitions_summary.csv')
endpoint_error=float(np.max(abs(endpoints[endpoints.aspect==4].sort_values('Ny').mean_gap.to_numpy()-
 previous.mean_lyapunov_gap_Delta_lambda.to_numpy())))
assert endpoint_error<1e-13
display(time_changes)
# %% markdown
# Free-intercept fits
# Report all nine linear fits and three quadratic fits. No positivity constraint, and no selection based on whether an interval contains zero.
# %%
fit_rows=[];fit_objects={};boot_cache={}
for u in ASPECT_RATIOS:
    for cutoff,degree in [(20,1),(24,1),(30,1),(20,2)]:
        sizes=[ny for ny in SIZES if ny>=cutoff]
        t=endpoints[(endpoints.aspect==u)&(endpoints.Ny>=cutoff)].sort_values('Ny')
        x=1/t.Ny.to_numpy();y=t.mean_gap.to_numpy();sigma=t['sem'].to_numpy()
        design=np.column_stack([x**p for p in range(degree+1)])
        cov=np.linalg.inv(design.T@(design/sigma[:,None]**2))
        operator=cov@(design.T/sigma[None,:]**2)
        coefficients=operator@y;bootstrap=np.column_stack([bootmeans[ny,u] for ny in sizes])@operator.T
        # Check the same weighted solver on a known synthetic polynomial.
        truth=np.array([.007,1.2]+([.3] if degree==2 else []))
        assert np.allclose(operator@(design@truth),truth,atol=1e-10)
        ci=np.quantile(bootstrap,[.025,.975],axis=0)
        chi=float(np.sum(((y-design@coefficients)/sigma)**2))
        key=f'u{u}_Nymin{cutoff}_degree{degree}'
        fit_objects[key]=dict(coefficients=coefficients,bootstrap=bootstrap,x=x,degree=degree);boot_cache[key]=bootstrap
        fit_rows.append(dict(aspect=u,Ny_min=cutoff,degree=degree,number_of_sizes=len(sizes),
         intercept=float(coefficients[0]),intercept_sem=float(np.sqrt(cov[0,0])),
         intercept_ci_low=float(ci[0,0]),intercept_ci_high=float(ci[1,0]),slope=float(coefficients[1]),
         slope_ci_low=float(ci[0,1]),slope_ci_high=float(ci[1,1]),
         quadratic=float(coefficients[2]) if degree==2 else None,
         chi_squared=chi,dof=len(sizes)-degree-1,contains_zero=bool(ci[0,0]<=0<=ci[1,0])))
fits=pd.DataFrame(fit_rows);fits.to_csv(OUT/'intercept_fit_summary.csv',index=False)
np.savez_compressed(OUT/'fit_bootstrap_coefficients.npz',**boot_cache,seed=BOOTSTRAP_SEED)
display(fits[['aspect','Ny_min','degree','intercept','intercept_ci_low','intercept_ci_high','chi_squared','dof']])
# %% markdown
# Main figure
# Left: rescaled finite-time gap, with one-SEM bands. Right: all-size free-intercept fits; gray background denotes sizes beyond those measured. Colored ribbons and intercept error bars are pointwise 95% bootstrap intervals. Dashed extensions are extrapolations.
# %%
import matplotlib as mpl
import matplotlib.pyplot as plt
reference=BASE/'06_domain_wall_flattened_ground_state_reference/analysis_outputs/disordered_exact_wall_half_filling_gap_Nx020_Ny_scan_W2_9_v1/latex_support'
shutil.copytree(reference,OUT/'latex_support',dirs_exist_ok=True)
os.environ['TEXINPUTS']=str(OUT/'latex_support')+os.pathsep+os.environ.get('TEXINPUTS','')
mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['CMU Sans Serif'],'font.size':8,
 'axes.labelsize':9,'axes.titlesize':9,'text.usetex':True,'text.latex.preamble':r'\usepackage{amsmath,amssymb}',
 'xtick.direction':'in','ytick.direction':'in','legend.frameon':False,'legend.fontsize':7})
size_colors=['#d62728','#2ca02c','#1565c0','#9467bd','#e377c2','#d28a00','#008b8b']
markers=['^','s','o','D','v','P','X'];styles=[':','--','-','-.',':','--','-']
aspect_styles={2:('#d62728','^',':'),3:('#2ca02c','s','--'),4:('#1565c0','o','-')}
fig,axes=plt.subplots(1,2,figsize=(7.05,3.55));a,b=axes
for ny,color,marker,style in zip(SIZES,size_colors,markers,styles):
    t=time_summary[(time_summary.Ny==ny)&(time_summary.aspect>=PLOT_MIN_ASPECT)]
    a.fill_between(t.aspect,t.scaled_mean-t.scaled_sem,t.scaled_mean+t.scaled_sem,color=color,alpha=.09,lw=0)
    a.plot(t.aspect,t.scaled_mean,color=color,marker=marker,ls=style,lw=1,ms=3,mfc='white',
     markevery=max(1,len(t)//6),label=r'$N_y='+str(ny)+'$')
a.set(xlabel=r'$T/N_y$',ylabel=r'$N_y\,\overline{\Delta_{\rm pur}(T)}$',
 xlim=(PLOT_MIN_ASPECT,4.05),title='Time-convergence check')
a.legend(ncol=2,loc='lower right',columnspacing=.8,handlelength=1.5)
b.axvspan(0,1/max(SIZES),color='.94',zorder=0);b.axhline(0,color='.5',lw=.8)
dense=np.linspace(0,1/min(SIZES),250)
for u in ASPECT_RATIOS:
    color,marker,style=aspect_styles[u];t=endpoints[endpoints.aspect==u].sort_values('Ny')
    b.errorbar(1/t.Ny,t.mean_gap,yerr=t['sem'],fmt=marker,color=color,mfc='white',ms=3.5,
     capsize=2,lw=.8,label=r'$T='+str(u)+r'N_y$')
    obj=fit_objects[f'u{u}_Nymin20_degree1'];A=np.column_stack([np.ones(len(dense)),dense])
    y=A@obj['coefficients'];lo,hi=np.quantile(obj['bootstrap']@A.T,[.025,.975],axis=0)
    b.fill_between(dense,lo,hi,color=color,alpha=.10,lw=0);measured=dense>=1/max(SIZES)
    b.plot(dense[measured],y[measured],color=color,ls=style,lw=1)
    b.plot(dense[~measured],y[~measured],color=color,ls='--',lw=.8)
    r=fits[(fits.aspect==u)&(fits.Ny_min==20)&(fits.degree==1)].iloc[0]
    b.errorbar([0],[r.intercept],yerr=[[r.intercept-r.intercept_ci_low],[r.intercept_ci_high-r.intercept]],
     fmt=marker,color=color,mfc='white',ms=4,capsize=2,lw=.8)
b.set(xlabel=r'$1/N_y$',ylabel=r'$\overline{\Delta_{\rm pur}(T)}$',xlim=(-.0015,.052),
 ylim=(-.008,.065),title=r'Size fits at fixed $T/N_y$')
b.text(.04,.96,r'$\overline{\Delta_{\rm pur}}=d_0+a/N_y$'+'\n'+r'$d_0$ fitted freely',
 transform=b.transAxes,va='top',fontsize=8);b.legend(loc='center right',bbox_to_anchor=(.98,.25))
for ax,label in zip(axes,['(a)','(b)']):
    ax.tick_params(top=True,right=True);ax.text(-.19,1.05,label,transform=ax.transAxes,fontsize=9)
fig.suptitle(r'$N_x=20$; hard domain walls; maximally mixed initial active slab'+'\n'+
 r'$100$ Born trajectories per size; finite-time Lyapunov gap',fontsize=9)
fig.tight_layout(pad=1)
fig.savefig(OUT/'gap_time_convergence_and_size_extrapolation.pdf');plt.close(fig)
subprocess.run(['pdftoppm','-r','300','-singlefile','-png',
 str(OUT/'gap_time_convergence_and_size_extrapolation.pdf'),str(OUT/'gap_time_convergence_and_size_extrapolation')],check=True)
display(Image(filename=str(OUT/'gap_time_convergence_and_size_extrapolation.png'),width=1200))
# %% markdown
# Intercept sensitivity
# Changes under size cuts and the subleading correction reveal extrapolation sensitivity. These fits do not establish the time-asymptotic gap.
# %%
fig,ax=plt.subplots(figsize=(5.6,3.6));ax.axhline(0,color='.3',ls='--',lw=1)
settings=[(20,1),(24,1),(30,1),(20,2)]
for u,offset in zip(ASPECT_RATIOS,[-.13,0,.13]):
    color,marker,style=aspect_styles[u]
    r=pd.DataFrame([fits[(fits.aspect==u)&(fits.Ny_min==cutoff)&(fits.degree==degree)].iloc[0] for cutoff,degree in settings])
    ax.errorbar(np.arange(4)+offset,r.intercept,
     yerr=[r.intercept-r.intercept_ci_low,r.intercept_ci_high-r.intercept],
     fmt=marker,ls=style,color=color,mfc='white',capsize=3,ms=4,lw=1,label=r'$T='+str(u)+r'N_y$')
ax.set(xticks=range(4),xticklabels=[r'$N_y\ge20$',r'$N_y\ge24$',r'$N_y\ge30$',
 r'$N_y\ge20$'+'\n'+r'with $b/N_y^2$'],ylabel=r'Extrapolated intercept $d_0$',
 title='Free intercept: size and fit-model sensitivity')
ax.tick_params(top=True,right=True);ax.legend(ncol=3,loc='upper left')
fig.text(.5,.02,r'Error bars: 95\% whole-trajectory bootstrap intervals',ha='center',fontsize=8)
fig.tight_layout(rect=(0,.06,1,1))
fig.savefig(OUT/'gap_intercept_sensitivity.pdf');plt.close(fig)
subprocess.run(['pdftoppm','-r','300','-singlefile','-png',
 str(OUT/'gap_intercept_sensitivity.pdf'),str(OUT/'gap_intercept_sensitivity')],check=True)
display(Image(filename=str(OUT/'gap_intercept_sensitivity.png'),width=1000))
# %% markdown
# Diagnostics and interpretation
# The rate still changes with time. A negative extrapolated intercept is not a physical negative gap; it indicates finite-size model sensitivity or sampling effects. Longer-time convergence remains unresolved.
# %%
diagnostics=dict(trajectories=700,shards=140,direct_soft_mode_max_error=max_soft_error,
 direct_manybody_gap_max_error=max_manybody_error,prior_endpoint_max_error=endpoint_error,
 finite_minima_at_all_saved_positive_times=True,maximum_selected_raw_gap_by_size=raw_max_by_size,
 cap_cost_threshold=float(np.log((1-campaign.CAP_TOLERANCE)/campaign.CAP_TOLERANCE)),
 max_fraction_of_capped_modes=max_cap_fraction,bootstrap_unit='whole trajectory, paired across times',
 intercept_constraint='none',long_time_limit_established=False,
 prior_manifest_note='Historical aggregate inventory digest discrepancy documented in original analysis; each raw NPZ independently checked against its identity-validated receipt.',
 interpretation='Finite-time gaps decrease with circumference. Negative intercepts in all-size linear fits are model-sensitive, not physical negative gaps.')
(OUT/'diagnostics.json').write_text(json.dumps(diagnostics,indent=2)+'\n')
display(pd.Series(diagnostics))
print('Complete: 700 trajectories, all saved times, 12 fits, two figures.')
