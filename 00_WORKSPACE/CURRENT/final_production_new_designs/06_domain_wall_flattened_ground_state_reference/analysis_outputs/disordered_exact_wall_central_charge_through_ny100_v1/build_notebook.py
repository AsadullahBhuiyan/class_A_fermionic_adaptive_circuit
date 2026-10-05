from pathlib import Path
import json,hashlib,shutil,argparse
import nbformat
from nbclient import NotebookClient
OUT=Path(__file__).resolve().parent;BASE=OUT.parents[2]
shutil.copytree(BASE/'09_pure_tangent_replay_acquisition/analysis_outputs/pure_half_system_energy_size_scan_L099_v1/latex_support',OUT/'latex_support',dirs_exist_ok=True)
nb=nbformat.v4.new_notebook(metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'}})
def md(s):nb.cells.append(nbformat.v4.new_markdown_cell(s))
def code(s):nb.cells.append(nbformat.v4.new_code_cell(s))
md(r'''# Equilibrium disorder: size dependence of entropy fits
Compare $N_x=20$ and $N_y=32,36,40,44,48,80$ with the same hard-wall OW parent, $\alpha_1=1,\alpha_2=30,n_{\rm shell}=1$:
$$h_{\rm dis}=h_{\rm flat}-\operatorname{diag}(m_{x_i}v_i),\qquad
v_i\overset{\rm iid}{\sim}\mathcal N(0,W^2),\quad m_x=1\text{ only at }x=5,15.$$
Potentials are independent across $y$ and both orbitals; no sample demeaning and no reflattening. Occupy the globally lowest $20N_y$ energies.
Use 100 independent realizations for each nonzero $W^2=1,2,3,4,6,9,12,16,25$, plus one clean reference per size. All states are calculated with disorder restricted to the two wall columns. This notebook includes only fully completed sizes; the final target is all six listed sizes.

For every width $A_y=1,\ldots,N_y/2$, include all $x$ and both orbitals, and average **full von Neumann entropy** over all $N_y$ periodic origins within each state, then average states:
$$\bar S(A_y)=\frac1{100}\sum_s\frac1{N_y}\sum_{y_0}[-\operatorname{tr}(C_A\log C_A+(\mathbf1-C_A)\log(\mathbf1-C_A))].$$
No entanglement-energy window is used. Fit unweighted, free-intercept OLS:
$$\bar S(A_y)=s_0+\frac{c_{\rm fit}}3\log\left[\frac{N_y}{\pi}\sin\frac{\pi A_y}{N_y}\right].$$
The primary fit extends the previous range to $5\le A_y\le N_y/2$. Also retain fixed starts 2 and 8, and fixed relative starts $\lceil5N_y/32\rceil$ and $\lceil N_y/4\rceil$. This distinguishes changes in the accessible fitting interval from size dependence.

$c_{\rm fit}$ is the finite-size total-strip coefficient, not per wall. Sampling errors are one SEM of independent realization profiles; bootstrap whole realizations to retain origin and width correlations. Clean states have no disorder sampling uncertainty. We do not infer an asymptotic central charge or impose an extrapolation law from this short size range.''')
code('''import os
CPU_RANGE=(40,55)
selected=list(range(CPU_RANGE[0],CPU_RANGE[1]+1))
assert set(selected).issubset(os.sched_getaffinity(0))
os.sched_setaffinity(0,selected)
for k in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[k]='1'
print('Allocated CPUs:',selected)''')
md('## Configuration and verified caches\nChanging fit windows or figures below does not repeat ground-state calculations. The source large-size run creates resumable per-realization NPZ files with completion receipts. Read completed wall-only sizes from this campaign; incomplete sizes are not plotted.')
code(r'''from pathlib import Path
import json,hashlib,subprocess
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from IPython.display import display,Image
OUT=Path.cwd()
SIZES=[ny for ny in [32,36,40,44,48,80] if (OUT/f'Ny{ny:03d}'/'acquisition_complete.json').exists()]
assert SIZES
VARIANCES=[1,2,3,4,6,9,12,16,25]
BOOTSTRAPS=2000
BOOTSTRAP_SEED=2026092706
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
profiles={};cleans={};inputs=[];acquisition={}
for ny in SIZES:
    folder=OUT/f'Ny{ny:03d}'
    receipt=json.loads((folder/'acquisition_complete.json').read_text())
    f=folder/'entropy_profiles.npz'
    assert receipt['status']=='complete' and receipt['realizations']==900 and receipt['cache_sha256']==sha(f)
    with np.load(f) as z:
        a=z['entropy_by_origin'];clean=z['clean_entropy_by_origin']
        assert a.shape==(9,100,ny//2,ny) and clean.shape==(ny//2,ny)
        assert np.array_equal(z['origins'],np.arange(ny)) and np.array_equal(z['sample_ids'],np.arange(100))
        assert np.array_equal(z['variances'],VARIANCES)
        assert np.isfinite(a).all() and np.isfinite(clean).all()
        profiles[ny]=a.mean(-1);cleans[ny]=clean.mean(-1)
    inputs.append(dict(Ny=ny,path=str(f),bytes=f.stat().st_size,sha256=sha(f)))
    acquisition[ny]=receipt
(OUT/'analysis_inputs.json').write_text(json.dumps(inputs,indent=2)+'\n')
display(pd.Series(json.loads((OUT/'input_provenance.json').read_text())['configuration']))
print('Verified wall-only profiles for sizes:',SIZES)
''')
md('## Fit coefficients, fit quality, and uncertainty\nEvaluate both absolute and fractional fitting ranges. Coefficient SEMs propagate all cross-width correlations through realization-level fits. Bootstrap confidence intervals keep every width of a realization together.')
code(r'''rng=np.random.default_rng(BOOTSTRAP_SEED)
rows=[];profile_rows=[];sample_rows=[];curves={};boot_cache={}
for ny in tqdm(SIZES,desc='Size-dependent fits',unit='size'):
    widths=np.arange(1,ny//2+1);x=np.log(ny/np.pi*np.sin(np.pi*widths/ny))
    windows={'fixed5':5,'fixed8':8,'fixed2':2,'fraction5_32':int(np.ceil(5*ny/32)),'quarter':int(np.ceil(ny/4))}
    for i,v in enumerate([0]+VARIANCES):
        a=cleans[ny][None,:] if v==0 else profiles[ny][i-1]
        mean=a.mean(0);n=len(a);sem=np.zeros(len(widths)) if n==1 else a.std(0,ddof=1)/np.sqrt(n)
        weights=None if n==1 else rng.multinomial(n,np.full(n,1/n),size=BOOTSTRAPS)/n
        ba=None if weights is None else weights@a
        profile_rows.extend(dict(Ny=ny,variance=v,Ay=int(w),entropy=m,entropy_sem=s) for w,m,s in zip(widths,mean,sem))
        for label,start in windows.items():
            mask=widths>=start;X=np.c_[np.ones(mask.sum()),x[mask]];op=np.linalg.pinv(X)
            co=a[:,mask]@op.T;coef=co.mean(0);pred=coef[0]+coef[1]*x
            r2=1-np.sum((mean[mask]-pred[mask])**2)/np.sum((mean[mask]-mean[mask].mean())**2)
            c=3*coef[1];err=0 if n==1 else 3*co[:,1].std(ddof=1)/np.sqrt(n)
            if n==1:clo=chi=c;rlo=rhi=r2
            else:
                bc=ba[:,mask]@op.T
                br=1-np.sum((ba[:,mask]-bc@X.T)**2,axis=1)/np.sum((ba[:,mask]-ba[:,mask].mean(1,keepdims=True))**2,axis=1)
                clo,chi=np.quantile(3*bc[:,1],[.025,.975]);rlo,rhi=np.quantile(br,[.025,.975])
                boot_cache[f'c_Ny{ny}_W2_{v}_{label}']=3*bc[:,1]
                # Independent covariance propagation cross-check.
                cov=np.cov(a[:,mask],rowvar=False)/n
                assert np.isclose(err**2,(3*op[1])@cov@(3*op[1]),atol=1e-12)
            rows.append(dict(Ny=ny,variance=v,samples=n,origins=ny,fit_window=label,Ay_min=start,Ay_max=ny//2,
                             c_fit=c,c_sampling_sem=err,c_ci_low=clo,c_ci_high=chi,
                             R_squared=r2,R_squared_ci_low=rlo,R_squared_ci_high=rhi,
                             slope=coef[1],intercept=coef[0],residual_rms=float(np.sqrt(np.mean((mean[mask]-pred[mask])**2))),
                             half_entropy=mean[-1],half_entropy_sem=sem[-1]))
            curves[(ny,v,label)]=pred
            sample_rows.extend(dict(Ny=ny,variance=v,sample_id=s,fit_window=label,c_fit=3*cc[1],intercept=cc[0]) for s,cc in enumerate(co))
fits=pd.DataFrame(rows);primary=fits[fits.fit_window=='fixed5']
fits.to_csv(OUT/'all_fit_windows.csv',index=False)
primary.to_csv(OUT/'central_charge_size_fits.csv',index=False)
pd.DataFrame(profile_rows).to_csv(OUT/'mean_entropy_profiles.csv',index=False)
pd.DataFrame(sample_rows).to_csv(OUT/'realization_fits.csv',index=False)
np.savez_compressed(OUT/'bootstrap_coefficients.npz',**boot_cache,seed=BOOTSTRAP_SEED)
trends=[]
for window in (['fixed5','quarter','fraction5_32','fixed8'] if 80 in SIZES else []):
    for v in VARIANCES:
        t=fits[(fits.fit_window==window)&(fits.variance==v)].set_index('Ny')
        delta=t.loc[80,'c_fit']-t.loc[32,'c_fit']
        err=np.hypot(t.loc[80,'c_sampling_sem'],t.loc[32,'c_sampling_sem'])
        bd=boot_cache[f'c_Ny80_W2_{v}_{window}']-boot_cache[f'c_Ny32_W2_{v}_{window}']
        lo,hi=np.quantile(bd,[.025,.975])
        trends.append(dict(variance=v,fit_window=window,delta_c_80_minus_32=delta,delta_sampling_sem=err,
                           delta_ci_low=lo,delta_ci_high=hi))
pd.DataFrame(trends).to_csv(OUT/'size_trends.csv',index=False)
display(primary.pivot(index='variance',columns='Ny',values='c_fit'))
display(pd.DataFrame(trends))
''')
code(r'''import matplotlib as mpl
import matplotlib.pyplot as plt
os.environ['TEXINPUTS']=str(OUT/'latex_support')+os.pathsep+os.environ.get('TEXINPUTS','')
mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['CMU Sans Serif'],'font.size':9,
 'axes.labelsize':10,'axes.titlesize':10,'text.usetex':True,'text.latex.preamble':r'\usepackage{amsmath,amssymb}',
 'xtick.direction':'in','ytick.direction':'in','legend.frameon':False,'legend.fontsize':8,'pdf.fonttype':42})
''')
md('## Effective central charge versus disorder\nEach panel uses one fitting window, with one curve per system size. Error bars are one SEM across independent disorder realizations. The horizontal coordinate is the disorder variance W^2. Here c_eff is the same 3-times-slope coefficient stored as c_fit in the tables.')
code(r'''colors=['#d62728','#228833','#1565c0','#aa4499','#cc8800','#009999','#663399','#999933','#333333'];markers=['^','s','o','D','v','P','X','<','>']
styles=[':', '--', '-', '-.', (0,(4,1,1,1)), ':','--','-.','-']
fig,axes=plt.subplots(1,2,figsize=(9,4.1),sharex=True,sharey=True)
for ax,window,title,letter in zip(axes,['fixed5','fixed8'],
        [r'Fitting window: $5\le A_y\le N_y/2$',r'Fitting window: $8\le A_y\le N_y/2$'],'ab'):
    for ny,color,marker,ls in zip(SIZES,colors,markers,styles):
        t=fits[(fits.Ny==ny)&(fits.fit_window==window)].sort_values('variance')
        ax.errorbar(t.variance,t.c_fit,yerr=t.c_sampling_sem,marker=marker,ls=ls,
                    color=color,mfc='white',ms=4,capsize=2,lw=1,label=rf'$N_y={ny}$')
    ax.set_title(title)
    ax.set_xlabel(r'Disorder variance $W^2$')
    ax.tick_params(top=True,right=True)
    ax.text(-.15,1.04,f'({letter})',transform=ax.transAxes)
    ax.legend(ncol=3,loc='lower right',fontsize=7)
axes[0].set_ylabel(r'$c_{\mathrm{eff}}$')
fig.suptitle(r'$N_x=20$; disorder at $x=5,15$; full entropy'+'\n'+
             r'$100$ realizations per ensemble; all cut origins; error bars: one SEM',fontsize=10)
fig.tight_layout(pad=1.2)
fig.savefig(OUT/'c_effective_vs_disorder_fit_windows.pdf');plt.close(fig)
subprocess.run(['pdftoppm','-r','300','-singlefile','-png',str(OUT/'c_effective_vs_disorder_fit_windows.pdf'),str(OUT/'c_effective_vs_disorder_fit_windows')],check=True)
display(Image(filename=str(OUT/'c_effective_vs_disorder_fit_windows.png'),width=1100))
''')
md('## Diagnostics and limits\nCheck exact block reduction against full-matrix ground states at clean and strongest disorder for each new size, including periodic wrapping cuts. State-level checks cover Hermiticity, finite physical occupations, traces, purity, and complement symmetry. Changes across sizes describe this finite strip; the sampling intervals exclude fit-range and finite-size systematic effects.')
code(r'''for ny in SIZES:
    r=acquisition[ny]
    assert r['diagnostic_maxima']['purity_error']<1e-10
    assert r['diagnostic_maxima']['trace_error']<1e-8
    assert r['diagnostic_maxima']['complement_error']<1e-7
    assert max(r['full_space_checks'])<1e-7
diagnostics={'acquisition':acquisition,'bootstrap':{'replicates':BOOTSTRAPS,'seed':BOOTSTRAP_SEED,'unit':'whole realization'},
             'independent_sizes':True,'primary_window':'5..Ny/2','controlled_relative_window':'ceil(Ny/4)..Ny/2',
             'fits':fits.to_dict('records'),'trends':trends,'disorder_x_columns':[5,15]}
(OUT/'diagnostics.json').write_text(json.dumps(diagnostics,indent=2)+'\n')
display(primary[['Ny','variance','c_fit','c_sampling_sem','R_squared']])
print('Complete sizes:',SIZES,'; disordered states:',900*len(SIZES))
''')
parser=argparse.ArgumentParser();parser.add_argument('--write-only',action='store_true');args=parser.parse_args()
path=OUT/'disordered_equilibrium_central_charge_size_scan.ipynb';nbformat.write(nb,path)
if args.write_only:
 print('Notebook generated; execution deferred until acquisition finishes.');raise SystemExit
NotebookClient(nb,timeout=300,kernel_name='python3',resources={'metadata':{'path':str(OUT)}}).execute();nbformat.write(nb,path)
files={}
for f in OUT.iterdir():
 if f.is_file() and f.suffix in ['.csv','.npz','.pdf','.png','.ipynb','.py','.json'] and f.name!='completion_manifest.json':
  files[f.name]=dict(bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest())
completed=[ny for ny in [32,36,40,44,48,80] if (OUT/f'Ny{ny:03d}'/'acquisition_complete.json').exists()]
(OUT/'completion_manifest.json').write_text(json.dumps(dict(status='complete' if len(completed)==6 else 'partial_size_snapshot',sizes=completed,files=files),indent=2)+'\n')
print('NOTEBOOK AND FIGURES COMPLETE:',OUT)
