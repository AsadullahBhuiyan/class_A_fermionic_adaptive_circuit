from pathlib import Path
import json,hashlib,shutil
import nbformat
from nbclient import NotebookClient
OUT=Path(__file__).resolve().parent
BASE=OUT.parents[2]
style=BASE/'09_pure_tangent_replay_acquisition/analysis_outputs/pure_half_system_energy_size_scan_L099_v1/latex_support'
shutil.copytree(style,OUT/'latex_support',dirs_exist_ok=True)
nb=nbformat.v4.new_notebook(metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'}})
def md(s):nb.cells.append(nbformat.v4.new_markdown_cell(s))
def code(s):nb.cells.append(nbformat.v4.new_code_cell(s))
md(r'''# Disorder dependence of equilibrium central-charge fits
Full von Neumann entropy of the same 900 equilibrium ground states used for the adjacent-gap-ratio figure: $N_x=20,N_y=32$, 100 realizations per $W^2=1,2,3,4,6,9,12,16,25$, plus one clean ground state.
The canonical CPU OW parent has hard domain walls at $x=5,15$, $\alpha_1=1$, $\alpha_2=30$, and $n_{\rm shell}=1$.
$$h_{\rm dis}=h_{\rm flat}-\operatorname{diag}(m_{x_i}v_i),\quad v_i\overset{\rm iid}{\sim}\mathcal N(0,W^2),\quad m_x=1\text{ on }x=5,6,14,15.$$
The two orbital potentials are independent; disorder extends along all $y$. No reflattening. Occupy the globally lowest 640 energies.

For each pure state and periodic strip origin, compute the full entropy (natural logs):
$$S_s(A_y,y_0)=-\sum_j[\nu_j\log\nu_j+(1-\nu_j)\log(1-\nu_j)],\quad
\bar S(A_y)=\frac1{100}\sum_s\frac1{32}\sum_{y_0=0}^{31}S_s(A_y,y_0).$$
Use **all eigenvalues**, with no spectral window or renormalization. Evaluate entropy before averaging states or cuts.

Fit unweighted, free-intercept ordinary least squares:
$$\bar S(A_y)=s_0+\frac{c_{\rm fit}}3\log\!\left[\frac{N_y}{\pi}\sin\frac{\pi A_y}{N_y}\right],\quad5\le A_y\le16.$$
Also fit $2\le A_y\le16$ and $8\le A_y\le16$. $c_{\rm fit}$ is the total-strip finite-size coefficient in the repository's $3\times$slope convention, not per wall or a demonstrated asymptotic central charge. This fixed-size two-parameter fit differs from the report's joint multi-size fit.
Sampling errors use the 100 independent realization profiles, retaining correlations across origins and widths. Coefficient errors are one SEM; bootstrap intervals are 95% whole-realization percentiles. The 3200 cuts are not independent samples.''')
code('''import os
CPU_RANGE=(8,23)
selected=list(range(CPU_RANGE[0],CPU_RANGE[1]+1))
assert set(selected).issubset(os.sched_getaffinity(0))
os.sched_setaffinity(0,selected)
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[key]='1'
print('Allocated CPUs:',selected)''')
md('## Configuration and verified inputs\nRun run_analysis.py to reconstruct profiles from saved potentials. Editing fits below does not repeat diagonalizations.')
code(r'''from pathlib import Path
import json,hashlib,subprocess
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from IPython.display import display,Image
OUT=Path.cwd()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
receipt=json.loads((OUT/'acquisition_complete.json').read_text())
assert receipt['status']=='complete' and receipt['realizations']==900
assert sha(OUT/'entropy_profiles.npz')==receipt['cache_sha256']
provenance=json.loads((OUT/'input_provenance.json').read_text())
assert receipt['identity']==provenance['identity']
with np.load(OUT/'entropy_profiles.npz') as z:
    all_entropy=z['entropy_by_origin'];clean_by_origin=z['clean_entropy_by_origin']
    variances=z['variances'];widths=z['widths'];origins=z['origins'];sample_ids=z['sample_ids']
assert all_entropy.shape==(9,100,16,32)
assert np.isfinite(all_entropy).all() and np.all(all_entropy>=0)
assert np.array_equal(origins,np.arange(32)) and np.array_equal(sample_ids,np.arange(100))
FIT_START=5
FIT_STARTS=sorted(set([2,FIT_START,8]))
BOOTSTRAPS=2000
BOOTSTRAP_SEED=2026092704
profiles=all_entropy.mean(-1);clean=clean_by_origin.mean(-1)
means=profiles.mean(1);sems=profiles.std(1,ddof=1)/10
x=np.log(32/np.pi*np.sin(np.pi*widths/32))
display(pd.Series(provenance['configuration']))
print('Maximum clean origin dependence:',np.ptp(clean_by_origin,axis=-1).max())
print('Saved half-cut spectrum agreement:',receipt['diagnostic_maxima']['saved_spectral_max_error'])
''')
md('## Fits and uncertainty\nThe linear fit propagates the covariance across widths exactly through realization-level coefficients. Bootstrap each whole profile, never individual origins or widths.')
code(r'''rng=np.random.default_rng(BOOTSTRAP_SEED)
rows=[];sample_rows=[];fit_curves={};bootstrap_values={}
for idx,v in enumerate(tqdm(np.r_[0,variances],desc='Fit disorder ensembles',unit='ensemble')):
    a=clean[None,:] if v==0 else profiles[idx-1]
    n=len(a);mean=a.mean(0)
    weights=None if n==1 else rng.multinomial(n,np.full(n,1/n),size=BOOTSTRAPS)/n
    boot_profiles=None if weights is None else weights@a
    for start in FIT_STARTS:
        mask=widths>=start
        X=np.column_stack([np.ones(mask.sum()),x[mask]]);op=np.linalg.pinv(X)
        coeff=a[:,mask]@op.T;coef=coeff.mean(0);pred=coef[0]+coef[1]*x
        residual=mean[mask]-pred[mask]
        r2=1-np.sum(residual**2)/np.sum((mean[mask]-mean[mask].mean())**2)
        c=3*coef[1];c_se=0. if n==1 else 3*coeff[:,1].std(ddof=1)/np.sqrt(n)
        if n==1:c_lo=c_hi=c;r2_lo=r2_hi=r2
        else:
            bc=boot_profiles[:,mask]@op.T
            br2=1-np.sum((boot_profiles[:,mask]-bc@X.T)**2,axis=1)/np.sum((boot_profiles[:,mask]-boot_profiles[:,mask].mean(1,keepdims=True))**2,axis=1)
            c_lo,c_hi=np.quantile(3*bc[:,1],[.025,.975]);r2_lo,r2_hi=np.quantile(br2,[.025,.975])
            bootstrap_values[f'c_W2_{v}_start_{start}']=3*bc[:,1]
            bootstrap_values[f'R2_W2_{v}_start_{start}']=br2
        rows.append(dict(variance=int(v),samples=n,origins=32,fit_Ay_min=int(start),fit_Ay_max=16,
                         intercept=coef[0],slope=coef[1],c_fit=c,c_sampling_sem=c_se,
                         c_bootstrap_ci_low=c_lo,c_bootstrap_ci_high=c_hi,
                         R_squared=r2,R_squared_ci_low=r2_lo,R_squared_ci_high=r2_hi,
                         residual_rms=np.sqrt(np.mean(residual**2)),half_entropy=mean[-1],
                         half_entropy_sem=0. if n==1 else a[:,-1].std(ddof=1)/np.sqrt(n)))
        fit_curves[(int(v),start)]=pred
        for sid,co in enumerate(coeff):
            sample_rows.append(dict(variance=int(v),sample_id=sid,fit_Ay_min=int(start),intercept=co[0],c_fit=3*co[1]))
fits=pd.DataFrame(rows);primary=fits[fits.fit_Ay_min==FIT_START].reset_index(drop=True)
fits.to_csv(OUT/'fit_window_sensitivity.csv',index=False)
primary.to_csv(OUT/'central_charge_fits.csv',index=False)
pd.DataFrame(sample_rows).to_csv(OUT/'realization_fits.csv',index=False)
np.savez_compressed(OUT/'bootstrap_fits.npz',**bootstrap_values,seed=BOOTSTRAP_SEED)
profile_rows=[]
for v,mean,sem in zip(np.r_[0,variances],np.vstack([clean,means]),np.vstack([np.zeros(16),sems])):
    profile_rows.extend(dict(variance=int(v),Ay=int(a),log_chord=xx,entropy=m,entropy_sem=s) for a,xx,m,s in zip(widths,x,mean,sem))
pd.DataFrame(profile_rows).to_csv(OUT/'mean_entropy_profiles.csv',index=False)
np.savez_compressed(OUT/'fit_statistics.npz',variances=variances,widths=widths,
                    realization_origin_averaged_entropy=profiles,mean=means,sem=sems,clean=clean,
                    cross_width_mean_covariance=np.array([np.cov(a,rowvar=False)/100 for a in profiles]))
display(primary[['variance','c_fit','c_sampling_sem','R_squared','residual_rms','half_entropy']])
''')
md('## Coefficient and fit quality versus disorder\nCoefficient bars are one SEM over realizations. Clean points have no sampling uncertainty. Smaller $1-R^2$ indicates a closer log-chord fit. Fit-range effects are distinct from sampling error.')
code(r'''import matplotlib as mpl
import matplotlib.pyplot as plt
os.environ['TEXINPUTS']=str(OUT/'latex_support')+os.pathsep+os.environ.get('TEXINPUTS','')
mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['CMU Sans Serif'],'font.size':9,
 'axes.labelsize':10,'axes.titlesize':10,'text.usetex':True,
 'text.latex.preamble':r'\usepackage{amsmath,amssymb}',
 'xtick.direction':'in','ytick.direction':'in','legend.frameon':False,'legend.fontsize':8,'pdf.fonttype':42})
fig,axes=plt.subplots(1,2,figsize=(8.4,3.4))
for start,color,fmt in [(FIT_START,'#1565c0','o-'),(8,'#cc8800','s--')]:
    table=fits[fits.fit_Ay_min==start]
    label=r'Fit: $'+rf'{start}\le A_y\le16$'
    axes[0].errorbar(table.variance,table.c_fit,yerr=table.c_sampling_sem,
                    fmt=fmt,color=color,mfc='white',ms=4,lw=1,capsize=2,label=label)
    axes[1].plot(table.variance,1-table.R_squared,fmt,color=color,mfc='white',ms=4,lw=1,label=label)
axes[0].set_ylabel(r'$c_{\mathrm{fit}}=3\,\mathrm{slope}$')
axes[1].set(yscale='log',ylabel=r'$1-R^2$')
for ax,letter in zip(axes,'ab'):
    ax.set_xlabel(r'Disorder variance $W^2$');ax.tick_params(top=True,right=True)
    ax.text(-.15,1.04,f'({letter})',transform=ax.transAxes);ax.legend(loc='best')
fig.suptitle(r'$20\times32$; disorder at $x=5,6,14,15$; $100$ realizations $\times32$ cut origins',fontsize=10)
fig.tight_layout(pad=1.2)
fig.savefig(OUT/'central_charge_vs_disorder.pdf');plt.close(fig)
subprocess.run(['pdftoppm','-r','300','-singlefile','-png',str(OUT/'central_charge_vs_disorder.pdf'),str(OUT/'central_charge_vs_disorder')],check=True)
display(Image(filename=str(OUT/'central_charge_vs_disorder.png'),width=1100))
''')
md('## Entropy scaling at every disorder strength\nShow all widths. Gray shading marks the fit range. Subtracting the measured half-system entropy is only for display; the fitted intercept remains free. Gray dotted curves show the clean profile. Bars are one SEM of realization profiles after subtraction.')
code(r'''fig,axes=plt.subplots(3,3,figsize=(9.5,8.4),sharex=True)
xx=np.log(np.sin(np.pi*widths/32))
colors=['#d62728','#228833','#1565c0','#aa4499','#cc8800','#009999','#bb5566','#4477aa','#66aa55']
for i,(ax,v,color,letter) in enumerate(zip(axes.flat,variances,colors,'abcdefghi')):
    shifted=profiles[i]-profiles[i,:,-1,None]
    sem=shifted.std(0,ddof=1)/10;mean=shifted.mean(0)
    ax.axvspan(xx[widths==FIT_START][0],0,color='.92',zorder=0)
    ax.plot(xx,clean-clean[-1],color='.5',ls=':',lw=1.1,label='Clean')
    ax.errorbar(xx,mean,yerr=sem,fmt='o',ms=3,color=color,mfc='white',capsize=1.5,label='Disordered')
    ax.plot(xx,fit_curves[(int(v),FIT_START)]-means[i,-1],'k--',lw=1,label='Fit')
    row=primary[primary.variance==v].iloc[0]
    ax.set_title(rf'$W^2={v:g}$')
    ax.text(.04,.96,rf'$c_{{\mathrm{{fit}}}}={row.c_fit:.3f}\pm{row.c_sampling_sem:.3f}$'+'\n'+rf'$R^2={row.R_squared:.6f}$',
            transform=ax.transAxes,va='top',fontsize=9)
    ax.text(-.18,1.04,f'({letter})',transform=ax.transAxes)
    ax.tick_params(top=True,right=True);ax.set_xlim(xx[0]-.08,.08)
axes[0,0].legend(loc='lower right',fontsize=8)
for ax in axes[:,0]:ax.set_ylabel(r'$\bar S(A_y)-\bar S(N_y/2)$')
for ax in axes[-1]:ax.set_xlabel(r'$\log[\sin(\pi A_y/N_y)]$')
fig.suptitle(r'Full entropy; $N_x=20$, $N_y=32$; $100$ realizations each; all $32$ origins'+'\n'+
             r'Fit: $'+rf'{FIT_START}\le A_y\le16$; disorder at $x=5,6,14,15$',fontsize=11)
fig.tight_layout(pad=1.2)
fig.savefig(OUT/'entropy_central_charge_fits.pdf');plt.close(fig)
subprocess.run(['pdftoppm','-r','300','-singlefile','-png',str(OUT/'entropy_central_charge_fits.pdf'),str(OUT/'entropy_central_charge_fits')],check=True)
display(Image(filename=str(OUT/'entropy_central_charge_fits.png'),width=1100))
''')
md('## Diagnostics\nFinite spectra, physical bounds, Hermiticity before symmetrization, trace identities, projector purity, complement symmetry, and saved half-cut spectra are checked. Direct full-matrix entropy checks include wrapping cuts at clean and strongest disorder. The exterior contribution is included exactly.')
code(r'''checks=receipt['diagnostic_maxima']
assert checks['saved_spectral_max_error']<1e-8 and checks['trace_max_error']<1e-8
assert checks['half_complement_entropy_max_error']<1e-7 and max(receipt['full_space_checks'])<1e-7
assert np.ptp(clean_by_origin,axis=-1).max()<1e-7
diagnostics={'configuration':provenance['configuration'],'acquisition':receipt,
             'primary_fit_start':FIT_START,'fits':fits.to_dict('records'),
             'clean_origin_max_spread':float(np.ptp(clean_by_origin,axis=-1).max()),
             'bootstrap':{'seed':BOOTSTRAP_SEED,'replicates':BOOTSTRAPS,'unit':'whole realization'},
             'uncertainty':'realization SEM; excludes finite-size/model/fit-range error',
             'scope':'effective total-strip coefficient at Nx20 Ny32; no asymptotic central-charge claim'}
(OUT/'diagnostics.json').write_text(json.dumps(diagnostics,indent=2)+'\n')
display(pd.Series(checks))
display(fits[['variance','fit_Ay_min','c_fit','c_sampling_sem','R_squared']])
print('Complete: 900 disordered states and clean reference; 32 origins; 16 widths.')
''')
path=OUT/'disordered_equilibrium_central_charge.ipynb'
nbformat.write(nb,path)
NotebookClient(nb,timeout=180,kernel_name='python3',resources={'metadata':{'path':str(OUT)}}).execute()
nbformat.write(nb,path)
(OUT/'README.md').write_text('''# Equilibrium disorder: central-charge fits
Same 900 saved realizations as the nine-panel gap-ratio comparison.
Nx20 Ny32, iid orbital-resolved Gaussian potential on x=5,6,14,15.
W^2=1,2,3,4,6,9,12,16,25, 100 samples each; one clean reference.
Reconstruct the occupied projector from the saved canonical CPU Hamiltonian
and exact saved potentials. Fill globally lowest 640 energies.
Full von Neumann entropy, no spectral window. Average 32 periodic origins
within each realization, then realizations. Save all widths Ay1..16.
Fit S=s0+(c_fit/3)log[(Ny/pi)sin(pi Ay/Ny)] over Ay5..16;
also Ay2..16 and Ay8..16. Unweighted free-intercept OLS.
c_fit is a finite-size total-strip coefficient, not per wall.
Coefficient errors: one SEM across realization profiles, preserving all
cross-width correlations. Bootstrap whole realizations, not cut origins.
Sampling errors exclude finite-size, model, and fit-range effects.
Run run_analysis.py then build_notebook.py. Acquisition resumes using
verified per-realization receipts. Executed notebook has editable plots.
Saved: raw origin entropies, provenance, fits, bootstrap, diagnostics,
vector PDF and 300-dpi PNG figures, completion checksums.
''')
files={}
for f in OUT.rglob('*'):
    if f.is_file() and f.name!='completion_manifest.json' and 'latex_support' not in f.parts:
        files[str(f.relative_to(OUT))]={'bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()}
(OUT/'completion_manifest.json').write_text(json.dumps({'status':'complete','disordered_states':900,'files':files},indent=2)+'\n')
print(OUT)
