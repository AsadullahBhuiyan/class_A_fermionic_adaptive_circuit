from pathlib import Path
import nbformat,json,hashlib,shutil
from nbclient import NotebookClient
OUT=Path(__file__).resolve().parent
SOURCE=OUT.parent/'disordered_exact_wall_Nx_scan_Ny040_W2_9_v1'
shutil.copytree(SOURCE/'latex_support',OUT/'latex_support',dirs_exist_ok=True)
nb=nbformat.v4.new_notebook(metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'}})
def md(s):nb.cells.append(nbformat.v4.new_markdown_cell(s))
def code(s):nb.cells.append(nbformat.v4.new_code_cell(s))
md(r'''# Minimum absolute Hamiltonian eigenvalue in square systems
Set $N_x=N_y=N=12,16,20,24,32,40$ and $W^2=9$. Disorder is iid zero-mean Gaussian on the diagonal entries of the two wall columns $x=N/4,3N/4$, independent in $y$ and orbital, and zero elsewhere. The canonical CPU overcomplete Wannier (OW) parent uses $\alpha_1=1,\alpha_2=30,n_{\rm shell}=1$, trial orbitals X and hard-wall truncation. No reflattening or rescaling.

For each of 100 independent Hamiltonians per size, retain the original energy zero and compute
$$g_s(N)=\min_i |E_{i,s}|,\qquad\bar g(N)=\frac1{100}\sum_s g_s(N).$$
Error bars are one SEM across Hamiltonians. The minimum is taken separately for each realization before averaging; all eigenvalues, including exact zeros if present, are retained. Also save the fixed-half-filling excitation gap and midgap chemical potential as distinct diagnostics.

Five new square sizes require eigenvalues only; no circuit dynamics or entropy calculation. Reuse the original 40-by-40 ensemble after checksum and protocol validation. The new sizes have independent size-indexed seeds. This scan increases both wall separation and edge circumference; it does not isolate either geometric effect by itself.''')
code('''import os
CPU_RANGE=(40,43)
selected=list(range(CPU_RANGE[0],CPU_RANGE[1]+1))
assert set(selected).issubset(os.sched_getaffinity(0))
os.sched_setaffinity(0,selected)
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[key]='1'
print('Allocated CPUs:',selected)''')
md('## Configuration and verified spectra\nRead all 600 saved spectra, validate sample identity, byte count and SHA-256, and cross-check the saved half-filling gap. The 40-by-40 point uses the original reused sample paths.')
code(r'''from pathlib import Path
import json,hashlib,subprocess
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from IPython.display import display,Image
OUT=Path.cwd();SIZES=[12,16,20,24,32,40];W2=9;SAMPLES=100
BOOTSTRAPS=5000;BOOTSTRAP_SEED=2026092803
ZERO_DIAGNOSTIC_TOL=1e-10
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
source_provenance=json.loads((OUT/'input_provenance.json').read_text())
assert source_provenance['configuration']['Nx_equals_Ny'] and source_provenance['configuration']['variance']==W2
rows=[];inputs=[];spectra={};max_saved_gap_error=0.;receipts={}
for nx in tqdm(SIZES,desc='Validate square Hamiltonian spectra',unit='size'):
    folder=OUT/f'N{nx:03d}';rp=folder/'acquisition_complete.json';acq=json.loads(rp.read_text())
    assert acq['status']=='complete' and acq['identity']==source_provenance['identity'] and acq['N']==nx and acq['samples']==100
    p=folder/'hamiltonian_spectra.npz'
    assert p.stat().st_size==acq['bytes'] and sha(p)==acq['sha256']
    with np.load(p) as z:
        assert np.array_equal(z['sample_ids'],np.arange(100)) and int(z['Nx'])==nx and int(z['Ny'])==nx and int(z['variance'])==W2
        es=z['hamiltonian_energies']
    assert es.shape==(100,2*nx*nx) and np.isrealobj(es) and np.isfinite(es).all() and np.all(np.diff(es,axis=1)>=-1e-12)
    receipts[nx]=acq;spectra[nx]=es
    inputs.append(dict(N=nx,path=str(p),bytes=acq['bytes'],sha256=acq['sha256'],receipt_sha256=sha(rp)))
    for sid,e in enumerate(es):
        if nx==40:
            f=Path(acq['reused_inputs'][sid]['path'])
            identity=acq['source_provenance']['identity']
        else:
            f=folder/f'realizations/sample_{sid:03d}.npz';identity=source_provenance['identity']
        r=json.loads(f.with_suffix('.json').read_text())
        assert r['identity']==identity and r['sample_id']==sid and r['Nx']==nx and r['Ny']==nx and r['variance']==W2
        assert r['filename']==f.name and r['bytes']==f.stat().st_size and r['sha256']==sha(f)
        with np.load(f) as z:
            assert np.array_equal(e,z['hamiltonian_energies'])
            assert np.array_equal(z['seed_components'],r['seed_components'])
            potential=z['diagonal_potential']
        assert np.all(potential[~np.isin(np.arange(2*nx*nx)//2%nx,[nx//4,3*nx//4])]==0)
        n=nx*nx;j=int(np.argmin(np.abs(e)));gap=float(e[n]-e[n-1])
        err=abs(gap-r['diagnostics']['half_filling_gap']);max_saved_gap_error=max(max_saved_gap_error,err);assert err<1e-12
        rows.append(dict(N=nx,Nx=nx,Ny=nx,W2=W2,sample_id=sid,min_abs_eigenvalue=float(abs(e[j])),
            nearest_signed_eigenvalue=float(e[j]),nearest_level_index=j,
            E_highest_occupied=float(e[n-1]),E_lowest_unoccupied=float(e[n]),
            half_filling_gap=gap,midgap_chemical_potential=float((e[n]+e[n-1])/2),
            negative_levels=int(np.count_nonzero(e<0)),near_zero_levels=int(np.count_nonzero(np.abs(e)<=ZERO_DIAGNOSTIC_TOL))))
samples=pd.DataFrame(rows)
samples.to_csv(OUT/'sample_gaps.csv',index=False)
np.savez_compressed(OUT/'hamiltonian_eigenvalues.npz',**{f'energies_N{nx:03d}':spectra[nx] for nx in SIZES},sizes=SIZES,W2=W2,sample_ids=np.arange(SAMPLES))
metadata=dict(geometry='Nx=Ny=N',sizes=SIZES,W2=W2,samples_per_size=SAMPLES,disorder_support='x=N/4,3N/4; independent y and orbital',
 estimator='arithmetic disorder mean of per-realization min(abs(E)); original energy zero; no rescaling',
 uncertainty='one SEM across independent Hamiltonians',new_hamiltonians=500,reused_hamiltonians=100)
(OUT/'analysis_inputs.json').write_text(json.dumps(dict(configuration=metadata,inputs=inputs),indent=2)+'\n')
display(pd.Series(metadata))''')
md('## Gap statistics\nSave the mean, SEM, median, quantiles and sample range. Bootstrap complete Hamiltonians for a 95% interval on the mean and on the Nx40 minus Nx12 difference.')
code(r'''rng=np.random.default_rng(BOOTSTRAP_SEED)
summary_rows=[];boot=[];gaps=[]
for nx in SIZES:
    t=samples[samples.Nx==nx].sort_values('sample_id')
    g=t.min_abs_eigenvalue.to_numpy();assert len(g)==SAMPLES and np.all(g>=0)
    b=g[rng.integers(0,SAMPLES,size=(BOOTSTRAPS,SAMPLES))].mean(1)
    lo,hi=np.quantile(b,[.025,.975]);q25,q75=np.quantile(g,[.25,.75])
    summary_rows.append(dict(N=nx,Nx=nx,Ny=nx,W2=W2,samples=SAMPLES,wall_separation=nx//2,
       mean_min_abs_eigenvalue=float(g.mean()),scaled_mean_N_times_gap=float(nx*g.mean()),sem=float(g.std(ddof=1)/np.sqrt(SAMPLES)),
       median=float(np.median(g)),q25=float(q25),q75=float(q75),minimum=float(g.min()),maximum=float(g.max()),
       mean_ci_low=float(lo),mean_ci_high=float(hi),mean_half_filling_gap=float(t.half_filling_gap.mean()),
       mean_midgap_chemical_potential=float(t.midgap_chemical_potential.mean())))
    boot.append(b);gaps.append(g)
summary=pd.DataFrame(summary_rows)
summary.to_csv(OUT/'gap_summary.csv',index=False)
np.savez_compressed(OUT/'gap_statistics.npz',sizes=SIZES,sample_ids=np.arange(SAMPLES),min_abs_eigenvalues=gaps,bootstrap_means=boot,bootstrap_seed=BOOTSTRAP_SEED)
delta=summary.iloc[-1].mean_min_abs_eigenvalue-summary.iloc[0].mean_min_abs_eigenvalue
delta_sem=float(np.hypot(summary.iloc[-1]['sem'],summary.iloc[0]['sem']))
lo,hi=np.quantile(boot[-1]-boot[0],[.025,.975])
endpoint=dict(delta_mean_Nx40_minus_Nx12=float(delta),sem=delta_sem,bootstrap_ci_low=float(lo),bootstrap_ci_high=float(hi))
(OUT/'endpoint_comparison.json').write_text(json.dumps(endpoint,indent=2)+'\n')
display(summary[['Nx','mean_min_abs_eigenvalue','sem','median','mean_half_filling_gap']])
display(pd.Series(endpoint))
previous=OUT.parent/'disordered_exact_wall_hamiltonian_gap_Nx_Ny040_W2_9_v1'
pm=json.loads((previous/'completion_manifest.json').read_text())
pf=previous/'gap_summary.csv';assert sha(pf)==pm['files'][pf.name]['sha256']
fixed=pd.read_csv(pf)
comparison=summary[['Nx','Ny','mean_min_abs_eigenvalue','sem']].merge(
 fixed[['Nx','mean_min_abs_eigenvalue','sem']],on='Nx',suffixes=('_square','_fixedNy40'),validate='one_to_one')
comparison.to_csv(OUT/'geometry_comparison.csv',index=False)
assert np.isclose(comparison.iloc[-1].mean_min_abs_eigenvalue_square,comparison.iloc[-1].mean_min_abs_eigenvalue_fixedNy40,rtol=0,atol=1e-15)
''')
md('## Minimum absolute eigenvalue versus square size\nOne point per size: arithmetic mean of the 100 per-Hamiltonian minima, with one-SEM error bars. Plot uses the saved Hamiltonian energy units.')
code(r'''import matplotlib as mpl
import matplotlib.pyplot as plt
os.environ['TEXINPUTS']=str(OUT/'latex_support')+os.pathsep+os.environ.get('TEXINPUTS','')
mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['CMU Sans Serif'],'font.size':9,
 'axes.labelsize':10,'axes.titlesize':10,'text.usetex':True,'text.latex.preamble':r'\usepackage{amsmath,amssymb}',
 'xtick.direction':'in','ytick.direction':'in','legend.frameon':False,'legend.fontsize':8,'pdf.fonttype':42})
fig,ax=plt.subplots(figsize=(5.2,3.7))
ax.errorbar(summary.Nx,summary.mean_min_abs_eigenvalue,yerr=summary['sem'],fmt='o-',color='#1565c0',
 mfc='white',ms=4,capsize=3,lw=1,label='Disorder mean')
ax.set(xlabel=r'Square size $N=N_x=N_y$',ylabel=r'$\overline{\min_i|E_i|}$',xticks=SIZES,ylim=(0,None))
ax.tick_params(top=True,right=True)
fig.suptitle(r'$N_x=N_y=N$; $W^2=9$; disorder at $x=N/4,3N/4$'+'\n'+
 r'$100$ realizations per size; original energy zero; error bars: one SEM',fontsize=9)
fig.tight_layout(pad=1)
fig.savefig(OUT/'minimum_absolute_eigenvalue_vs_square_size.pdf');plt.close(fig)
subprocess.run(['pdftoppm','-r','300','-singlefile','-png',str(OUT/'minimum_absolute_eigenvalue_vs_square_size.pdf'),str(OUT/'minimum_absolute_eigenvalue_vs_square_size')],check=True)
display(Image(filename=str(OUT/'minimum_absolute_eigenvalue_vs_square_size.png'),width=900))''')
md('## Diagnostics\nAll levels are retained. No near-zero censoring, eigenvalue clipping, spectrum centering, or matrix averaging. The half-filling gap is cross-checked against the original acquisition receipts.')
code(r'''diagnostics=dict(acquisition=receipts,configuration=metadata,total_hamiltonians=len(samples),levels_by_size={str(nx):2*nx*nx for nx in SIZES},
 exact_zero_minima=int((samples.min_abs_eigenvalue==0).sum()),
 near_zero_minima=int((samples.min_abs_eigenvalue<=ZERO_DIAGNOSTIC_TOL).sum()),zero_diagnostic_tolerance=ZERO_DIAGNOSTIC_TOL,
 max_saved_half_filling_gap_error=max_saved_gap_error,endpoint_comparison=endpoint,
 checks={'input_checksums':True,'unique_sample_ids':True,'finite_real_sorted_eigenvalues':True,'wall_only_disorder_support':True},
 bootstrap={'replicates':BOOTSTRAPS,'seed':BOOTSTRAP_SEED,'unit':'whole Hamiltonian'},
 distinction='min(abs(E)) uses original E=0; fixed-N excitation gap is E[N]-E[N-1] with zero-based indexing')
(OUT/'diagnostics.json').write_text(json.dumps(diagnostics,indent=2)+'\n')
display(pd.Series(diagnostics))
print('Complete: all 600 saved Hamiltonian spectra validated.')''')
p=OUT/'hamiltonian_gap_square_sizes.ipynb';nbformat.write(nb,p)
NotebookClient(nb,timeout=300,kernel_name='python3',resources={'metadata':{'path':str(OUT)}}).execute();nbformat.write(nb,p)
files={p.name:dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in OUT.iterdir() if p.is_file() and p.name not in ['completion_manifest.json','run.log','exit_code.txt']}
(OUT/'completion_manifest.json').write_text(json.dumps(dict(status='complete',hamiltonians=600,sizes=[12,16,20,24,32,40],files=files),indent=2)+'\n')
print('GAP ANALYSIS COMPLETE:',OUT)
