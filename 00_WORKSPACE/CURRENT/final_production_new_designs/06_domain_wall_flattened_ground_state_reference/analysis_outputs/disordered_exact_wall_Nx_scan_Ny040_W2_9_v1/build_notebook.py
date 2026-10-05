from pathlib import Path
import nbformat,json,hashlib,shutil,argparse
from nbclient import NotebookClient
OUT=Path(__file__).resolve().parent;BASE=OUT.parents[2]
shutil.copytree(BASE/'09_pure_tangent_replay_acquisition/analysis_outputs/pure_half_system_energy_size_scan_L099_v1/latex_support',OUT/'latex_support',dirs_exist_ok=True)
nb=nbformat.v4.new_notebook(metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'}})
def md(s):nb.cells.append(nbformat.v4.new_markdown_cell(s))
def code(s):nb.cells.append(nbformat.v4.new_code_cell(s))
md(r'''# Transverse-size dependence of disordered ground-state entropy
Fix $N_y=40$, $W^2=9$, scan $N_x=12,16,20,24,32,40$. Hard walls at $x=N_x/4,3N_x/4$ have independent zero-mean Gaussian diagonal potentials at each $y$ and orbital; zero elsewhere.
$$h_{\rm dis}=h_{\rm flat}-\operatorname{diag}(m_{x_i}v_i),\qquad v_i\overset{\rm iid}{\sim}\mathcal N(0,9).$$
Canonical CPU OW parent, $\alpha_1=1,\alpha_2=30,n_{\rm shell}=1$; occupy globally lowest $N_xN_y$ levels. No reflattening. Boundary separation and bulk width grow together.

100 pure ground states per size. For each state compute full entropy in nats for all $A_y=1,\ldots,20$, all 40 periodic cut origins, all $x$ and both orbitals. Average entropy over origins within each realization, then disorder. Do not average correlation matrices; no spectral window.
$$\bar S(A_y)=s_0+\frac{c_{\rm eff}}3\log\left[\frac{N_y}{\pi}\sin\frac{\pi A_y}{N_y}\right].$$
Unweighted free-intercept OLS on $5\le A_y\le20$ and $8\le A_y\le20$. This is a finite-size total-strip coefficient, not per wall. Error bars use independent realization profiles, preserving correlations between widths. No extrapolation is imposed.

Reuse the completed $N_x=20$ ensemble with original provenance and seeds. Each new size uses independent size-indexed seeds. Only completed sizes appear in interim plots.''')
code('''import os
CPU_RANGE=(40,55)
selected=list(range(CPU_RANGE[0],CPU_RANGE[1]+1))
assert set(selected).issubset(os.sched_getaffinity(0))
os.sched_setaffinity(0,selected)
for k in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[k]='1'
print('Allocated CPUs:',selected)''')
md('## Editable configuration and inputs\nChecksums and configuration identities are required before using each cache.')
code(r'''from pathlib import Path
import json,hashlib,subprocess
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from IPython.display import display,Image
OUT=Path.cwd();TARGET=[12,16,20,24,32,40];NY=40;STARTS=[5,8]
BOOTSTRAPS=2000;BOOTSTRAP_SEED=2026092711
config=json.loads((OUT/'input_provenance.json').read_text())
SIZES=[nx for nx in TARGET if (OUT/f'Nx{nx:03d}/acquisition_complete.json').exists()]
assert SIZES
profiles={};receipts={};inputs=[]
for nx in tqdm(SIZES,desc='Verify entropy profiles',unit='size'):
    folder=OUT/f'Nx{nx:03d}';r=json.loads((folder/'acquisition_complete.json').read_text());p=folder/'entropy_profiles.npz'
    assert r['identity']==config['identity'] and r['Nx']==nx and r['Ny']==NY and r['variance']==9 and r['samples']==100
    assert r['bytes']==p.stat().st_size and r['cache_sha256']==hashlib.sha256(p.read_bytes()).hexdigest()
    with np.load(p) as z:
        a=z['entropy_by_origin'];assert a.shape==(100,20,40) and np.isfinite(a).all()
        assert np.array_equal(z['sample_ids'],np.arange(100))
        assert np.array_equal(z['widths'],np.arange(1,21)) and np.array_equal(z['origins'],np.arange(40))
        profiles[nx]=a.mean(-1)
    receipts[nx]=r;inputs.append(dict(Nx=nx,path=str(p),bytes=r['bytes'],sha256=r['cache_sha256']))
(OUT/'analysis_inputs.json').write_text(json.dumps(inputs,indent=2)+'\n')
display(pd.Series(config['configuration']))
print('Complete:',SIZES,'Pending:',sorted(set(TARGET)-set(SIZES)))''')
md('## Fits and uncertainty\nFit mean entropy and check against mean realization coefficients. Bootstrap whole profiles; error bars are one realization SEM.')
code(r'''rng=np.random.default_rng(BOOTSTRAP_SEED)
widths=np.arange(1,21);x=np.log(NY/np.pi*np.sin(np.pi*widths/NY))
rows=[];sample_rows=[];profile_rows=[];boot={};predictions={}
for nx in tqdm(SIZES,desc='Entropy fits',unit='size'):
    a=profiles[nx];mean=a.mean(0);sem=a.std(0,ddof=1)/10
    ba=(rng.multinomial(100,np.full(100,.01),size=BOOTSTRAPS)/100)@a
    profile_rows.extend(dict(Nx=nx,Ay=int(w),entropy=m,entropy_sem=e) for w,m,e in zip(widths,mean,sem))
    for start in STARTS:
        mask=widths>=start;X=np.c_[np.ones(mask.sum()),x[mask]];op=np.linalg.pinv(X)
        co=op@mean[mask];individual=a[:,mask]@op.T
        assert np.allclose(co,individual.mean(0),atol=1e-12)
        c=3*co[1];error=3*individual[:,1].std(ddof=1)/10
        assert np.isclose(error**2,(3*op[1])@(np.cov(a[:,mask],rowvar=False)/100)@(3*op[1]),atol=1e-12)
        pred=co[0]+co[1]*x;predictions[nx,start]=pred
        r2=1-np.sum((mean[mask]-pred[mask])**2)/np.sum((mean[mask]-mean[mask].mean())**2)
        bc=3*(ba[:,mask]@op.T)[:,1];lo,hi=np.quantile(bc,[.025,.975]);boot[f'Nx{nx}_Aymin{start}']=bc
        rows.append(dict(Nx=nx,Ny=NY,W2=9,samples=100,origins=40,wall_left=nx//4,wall_right=3*nx//4,
          wall_separation=nx//2,Ay_min=start,Ay_max=20,c_effective=c,c_sampling_sem=error,c_ci_low=lo,c_ci_high=hi,
          R_squared=r2,slope=co[1],intercept=co[0]))
        sample_rows.extend(dict(Nx=nx,sample_id=s,Ay_min=start,c_effective=3*v[1],intercept=v[0]) for s,v in enumerate(individual))
fits=pd.DataFrame(rows);fits.to_csv(OUT/'central_charge_vs_Nx.csv',index=False)
pd.DataFrame(sample_rows).to_csv(OUT/'realization_fits.csv',index=False)
pd.DataFrame(profile_rows).to_csv(OUT/'mean_entropy_profiles.csv',index=False)
np.savez_compressed(OUT/'bootstrap_coefficients.npz',**boot,seed=BOOTSTRAP_SEED)
prior=pd.read_csv(OUT.parent/'disordered_exact_wall_central_charge_through_ny100_v1/all_fit_windows.csv')
for start in STARTS:
    old=prior[(prior.Ny==40)&(prior.variance==9)&(prior.fit_window==f'fixed{start}')].iloc[0]
    new=fits[(fits.Nx==20)&(fits.Ay_min==start)].iloc[0]
    assert np.isclose(new.c_effective,old.c_fit,atol=1e-12,rtol=0)
    assert np.isclose(new.c_sampling_sem,old.c_sampling_sem,atol=1e-12,rtol=0)
endpoint_rows=[]
if 12 in SIZES and 40 in SIZES:
    for start in STARTS:
        t=fits[fits.Ay_min==start].set_index('Nx')
        delta=float(t.loc[40,'c_effective']-t.loc[12,'c_effective'])
        se=float(np.hypot(t.loc[40,'c_sampling_sem'],t.loc[12,'c_sampling_sem']))
        bd=boot[f'Nx40_Aymin{start}']-boot[f'Nx12_Aymin{start}']
        lo,hi=np.quantile(bd,[.025,.975])
        endpoint_rows.append(dict(Ay_min=start,Ay_max=20,delta_c_Nx40_minus_Nx12=delta,sampling_sem=se,bootstrap_ci_low=float(lo),bootstrap_ci_high=float(hi)))
pd.DataFrame(endpoint_rows).to_csv(OUT/'endpoint_comparison.csv',index=False)
display(fits[['Nx','Ay_min','c_effective','c_sampling_sem','R_squared']])''')
md('## Effective central charge versus transverse size\nThe dotted line is the ideal independent-edge prediction; it is not a new clean-state fit. Nx changes wall separation as Nx/2.')
code(r'''import matplotlib as mpl
import matplotlib.pyplot as plt
os.environ['TEXINPUTS']=str(OUT/'latex_support')+os.pathsep+os.environ.get('TEXINPUTS','')
mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['CMU Sans Serif'],'font.size':9,
 'axes.labelsize':10,'axes.titlesize':10,'text.usetex':True,'text.latex.preamble':r'\usepackage{amsmath,amssymb}',
 'xtick.direction':'in','ytick.direction':'in','legend.frameon':False,'legend.fontsize':8,'pdf.fonttype':42})
fig,ax=plt.subplots(figsize=(5.2,3.7))
for start,color,marker,ls in zip(STARTS,['#1565c0','#228833'],['o','s'],['-','--']):
    t=fits[fits.Ay_min==start].sort_values('Nx')
    ax.errorbar(t.Nx,t.c_effective,yerr=t.c_sampling_sem,color=color,marker=marker,ls=ls,mfc='white',ms=4,capsize=3,lw=1,
       label=r'Fitting window: $'+rf'{start}\le A_y\le20$')
ax.axhline(1,color='gray',ls=':',lw=1,label=r'Independent-edge prediction: $c_{\rm eff}=1$')
ax.set(xlabel=r'Transverse size $N_x$',ylabel=r'$c_{\rm eff}$',xticks=TARGET)
ax.tick_params(top=True,right=True);ax.legend(loc='best')
fig.suptitle(r'$N_y=40$; $W^2=9$; disorder only at $x=N_x/4,3N_x/4$'+'\n'+
 r'$100$ realizations; all cut origins; full entropy; error bars: one SEM',fontsize=9)
fig.tight_layout(pad=1)
fig.savefig(OUT/'c_effective_vs_Nx.pdf');plt.close(fig)
subprocess.run(['pdftoppm','-r','300','-singlefile','-png',str(OUT/'c_effective_vs_Nx.pdf'),str(OUT/'c_effective_vs_Nx')],check=True)
display(Image(filename=str(OUT/'c_effective_vs_Nx.png'),width=900))''')
md('## Entropy-fit diagnostics\nAll widths are shown; gray marks 5..20 and a dotted line marks Ay=8. Half-system entropy is subtracted only for display.')
code(r'''fig,axes=plt.subplots(2,3,figsize=(9,5.8),sharex=True,sharey=True)
for ax,nx,letter in zip(axes.flat,TARGET,'abcdef'):
    ax.text(-.15,1.04,f'({letter})',transform=ax.transAxes);ax.set_title(rf'$N_x={nx}$');ax.tick_params(top=True,right=True)
    ax.axvspan(x[4],x[-1],color='.92');ax.axvline(x[7],color='.6',ls=':',lw=.8)
    if nx not in profiles:
        ax.text(.5,.5,'Pending',ha='center',va='center',transform=ax.transAxes);continue
    a=profiles[nx];mean=a.mean(0);sem=a.std(0,ddof=1)/10
    ax.errorbar(x,mean-mean[-1],yerr=sem,fmt='o',color='black',mfc='white',ms=3,capsize=2)
    for start,color,ls in zip(STARTS,['#1565c0','#228833'],['-','--']):
        ax.plot(x,predictions[nx,start]-mean[-1],color=color,ls=ls,lw=1,label=r'$'+rf'{start}\le A_y\le20$')
    t=fits[fits.Nx==nx]
    ax.text(.04,.95,'\n'.join(rf'$R^2_{{{int(r.Ay_min)}}}={r.R_squared:.6f}$' for r in t.itertuples()),ha='left',va='top',transform=ax.transAxes,fontsize=8)
for ax in axes[:,0]:ax.set_ylabel(r'$\bar S(A_y)-\bar S(20)$')
for ax in axes[-1]:ax.set_xlabel(r'$\log[(N_y/\pi)\sin(\pi A_y/N_y)]$')
for ax,nx in zip(axes.flat,TARGET):
    if nx in profiles:ax.legend(loc='lower right');break
fig.suptitle(r'Wall-only disorder; $N_y=40$; $W^2=9$; full entropy',fontsize=11)
fig.tight_layout(pad=1)
fig.savefig(OUT/'entropy_fit_diagnostics.pdf');plt.close(fig)
subprocess.run(['pdftoppm','-r','300','-singlefile','-png',str(OUT/'entropy_fit_diagnostics.pdf'),str(OUT/'entropy_fit_diagnostics')],check=True)
display(Image(filename=str(OUT/'entropy_fit_diagnostics.png'),width=1100))''')
md('## Numerical diagnostics and scope\nErrors exclude finite-size and fit-window systematics. Each new size receives direct full-space checks including a wrapping cut.')
code(r'''for nx,r in receipts.items():
    d=r['diagnostic_maxima']
    assert d['purity_error']<1e-10 and d['trace_error']<1e-8 and d['complement_error']<1e-7
    if nx!=20:assert max(r['full_space_checks'])<1e-7
parent_checks=json.loads((OUT/'active_parent_rows_validation.json').read_text())
assert sorted(r['Nx'] for r in parent_checks['checks'])==TARGET
assert max(r['max_absolute_error'] for r in parent_checks['checks'])<1e-10
(OUT/'diagnostics.json').write_text(json.dumps({'parent_checks':parent_checks,'completed_sizes':SIZES,'requested_sizes':TARGET,
 'fits':rows,'endpoint_comparison':endpoint_rows,'acquisition':receipts,'Nx20_reference_reproduced':True,
 'bootstrap':{'replicates':BOOTSTRAPS,'seed':BOOTSTRAP_SEED,'unit':'whole realization'},
 'interpretation':'finite-size coefficient; Nx changes wall separation and bulk width; no extrapolation'},indent=2)+'\n')
print('Complete:',SIZES,'Pending:',sorted(set(TARGET)-set(SIZES)))''')
parser=argparse.ArgumentParser();parser.add_argument('--write-only',action='store_true');args=parser.parse_args()
p=OUT/'central_charge_vs_Nx.ipynb';nbformat.write(nb,p)
if args.write_only:raise SystemExit
NotebookClient(nb,timeout=300,kernel_name='python3',resources={'metadata':{'path':str(OUT)}}).execute();nbformat.write(nb,p)
sizes=[nx for nx in [12,16,20,24,32,40] if (OUT/f'Nx{nx:03d}/acquisition_complete.json').exists()]
files={f.name:dict(bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for f in OUT.iterdir() if f.is_file() and f.suffix in ['.py','.ipynb','.csv','.npz','.pdf','.png','.json'] and f.name!='completion_manifest.json'}
(OUT/'completion_manifest.json').write_text(json.dumps(dict(status='complete' if len(sizes)==6 else 'partial_size_snapshot',sizes=sizes,files=files),indent=2)+'\n')
print('FIGURES COMPLETE:',OUT)
