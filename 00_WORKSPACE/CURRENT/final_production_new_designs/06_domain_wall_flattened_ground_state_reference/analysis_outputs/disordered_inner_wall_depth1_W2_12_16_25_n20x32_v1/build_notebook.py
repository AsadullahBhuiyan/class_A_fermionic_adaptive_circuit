from pathlib import Path
import nbformat,shutil,json,hashlib
from nbclient import NotebookClient
OUT=Path(__file__).resolve().parent
BASE=OUT.parents[2]
style=BASE/'09_pure_tangent_replay_acquisition/analysis_outputs/entanglement_energy_comparison_n20x32_L099_v1/latex_support'
shutil.copytree(style,OUT/'latex_support',dirs_exist_ok=True)
nb=nbformat.v4.new_notebook(metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'}})
def md(s):nb.cells.append(nbformat.v4.new_markdown_cell(s))
def code(s):nb.cells.append(nbformat.v4.new_code_cell(s))
md(r'''# Gaussian diagonal disorder on the walls and one column into the topological slab
For the existing $20\times32$ hard-wall parent Hamiltonian, add
$$\hat H_{\rm dis}=\hat H_{\rm flat}-\sum_{x,y,\mu}m_xv_{x,y,\mu}\hat c^\dagger_{x,y,\mu}\hat c_{x,y,\mu},$$
where $v_{x,y,\mu}$ are independent real Gaussian variables with population mean zero and variance $W^2=12,16,25$. The mask $m_x=1$ only on $x=5,6,14,15$, and is zero elsewhere. Potentials are independent across the sites and orbitals on these four columns. These three strengths use independent seeds extending the preceding inward-column scan. A realization's spatial sample mean is not artificially subtracted. Each strength has 100 independent, deterministic-seeded realizations, independent also across strengths. Fill the lowest 640 levels globally for each realization; no additional spectral flattening is applied after adding the potential.

The subsystem has all x, both orbitals, and $y=0,\ldots,15$ (640 modes); use the fixed cut $y_0=0$. Diagonalize its centered covariance separately for every ground state and retain $|\lambda|\le0.99$.
$$\varepsilon=\log[(1-\lambda)/(1+\lambda)],\qquad |\varepsilon|\le\log199.$$
The main figure shows raw pooled histogram counts. A second figure shows unit-area densities including the clean reference. Both use identical, zero-centered bins. These are static pure ground states, with no circuit dynamics. The active/exterior block reduction is exact for the hard-wall Hamiltonian and was checked against a full-space calculation for the first realization at every disorder strength.''')
code('''import os
CPU_RANGE=(8,15) # Editable inclusive allocation.
selected=list(range(CPU_RANGE[0],CPU_RANGE[1]+1))
assert set(selected).issubset(os.sched_getaffinity(0))
os.sched_setaffinity(0,selected)
for k in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[k]=str(len(selected))
print('Allocated CPUs:',selected)''')
md('## Acquisition and configuration\nThe companion runner constructs the parent with the canonical CPU class, then solves each static ground state. Existing realization receipts allow a completed run to be reused.')
code(r'''from pathlib import Path
import sys,json,hashlib,subprocess
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from IPython.display import display,Image
OUT=Path.cwd()
RUN_ACQUISITION=False # Set True to generate/revalidate realization products.
if RUN_ACQUISITION:
    subprocess.run([sys.executable,str(OUT/'run_analysis.py')],check=True)
complete=json.loads((OUT/'acquisition_complete.json').read_text())
assert complete['status']=='complete' and complete['realizations']==300
config=complete['configuration']
display(pd.Series(config))
VARIANCES=config['variances']
L=config['L'];E=float(np.log1p(L)-np.log1p(-L))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
''')
md('## Validate and collect the spectra\nEach realization is verified against its checksum and configuration identity. Save the unmodified centered spectra and the finite-window energies with realization labels.')
code(r'''spectra={};energy={};cache={};rows=[];sample_rows=[];receipts=[]
disorder_mask=np.isin(np.arange(1280)//2%20,config['disorder_x_columns'])
assert disorder_mask.sum()==256
cache['disorder_mask']=disorder_mask
for vi,variance in enumerate(VARIANCES):
    arrays=[];potentials=[];eigensystems=[]
    for s in tqdm(range(100),desc=f'Validate variance {variance:g}',unit='state'):
        f=OUT/'realizations'/f'variance_{vi:02}_sample_{s:03}.npz'
        d=json.loads(f.with_suffix('.json').read_text())
        assert d['identity']==complete['identity'] and d['sample_id']==s and d['variance']==variance
        assert d['bytes']==f.stat().st_size and d['sha256']==sha(f)
        receipts.append(d)
        with np.load(f) as z:
            assert int(z['sample_id'])==s and float(z['variance'])==variance
            arrays.append(z['centered_eigenvalues'])
            potentials.append(z['diagonal_potential'])
            eigensystems.append(z['hamiltonian_energies'])
    lam=np.array(arrays);v=np.array(potentials)
    assert lam.shape==(100,640) and v.shape==(100,1280)
    assert np.isfinite(v).all() and np.all(v[:,~disorder_mask]==0)
    assert np.all(v[:,disorder_mask]!=0)
    assert np.all(v[:,0::2][:,disorder_mask[0::2]]!=v[:,1::2][:,disorder_mask[0::2]])
    assert np.isfinite(lam).all() and np.max(np.abs(lam))<=1+1e-8
    mask=np.abs(lam)<=L
    sample_id,mode_index=np.nonzero(mask)
    eps=np.log1p(-lam[mask])-np.log1p(lam[mask])
    assert np.isfinite(eps).all() and np.max(np.abs(eps))<=E
    assert np.allclose(-np.tanh(eps/2),lam[mask],rtol=0,atol=5e-16)
    spectra[variance]=lam;energy[variance]=eps
    cache[f'centered_spectra_{vi}']=lam;cache[f'energy_{vi}']=eps
    cache[f'sample_id_{vi}']=sample_id;cache[f'mode_index_{vi}']=mode_index
    cache[f'diagonal_potential_{vi}']=v;cache[f'hamiltonian_energies_{vi}']=np.array(eigensystems)
    n=np.bincount(sample_id,minlength=100)
    m1=np.bincount(sample_id,weights=eps,minlength=100)
    m2=np.bincount(sample_id,weights=eps**2,minlength=100)
    N=len(eps)
    leave=(m2.sum()-m2)/(N-n)-((m1.sum()-m1)/(N-n))**2
    se=float(np.sqrt(99/100*np.sum((leave-leave.mean())**2)))
    rows.append({'disorder_variance':variance,'disorder_std':np.sqrt(variance),'samples':100,
                 'retained_modes':N,'mean_retained_per_state':float(n.mean()),
                 'energy_mean':float(eps.mean()),'energy_variance':float(eps.var()),
                 'variance_jackknife_se':se,'realized_potential_mean_on_mask':float(v[:,disorder_mask].mean()),
                 'realized_potential_variance_on_mask':float(v[:,disorder_mask].var())})
    for s in range(100):
        sample_rows.append({'disorder_variance':variance,'sample_id':s,'retained_modes':int(n[s]),
                            'energy_sum':float(m1[s]),'energy_square_sum':float(m2[s])})
with np.load(OUT/'clean_reference.npz') as z:
    clean=z['centered_eigenvalues'];m=np.abs(clean)<=L
    clean_energy=np.log1p(-clean[m])-np.log1p(clean[m])
np.savez_compressed(OUT/'spectra_and_energies.npz',**cache,variances=VARIANCES,L=L,clean_energy=clean_energy)
summary=pd.DataFrame(rows)
summary.to_csv(OUT/'summary.csv',index=False)
pd.DataFrame(sample_rows).to_csv(OUT/'sample_moments.csv',index=False)
display(summary)''')
md('## Raw-count entanglement-energy histograms\nPool 100 states per disorder variance. The vertical axis is the number of retained modes in each energy bin, with no normalization. Edit BINS without recomputing ground states.')
code(r'''import matplotlib as mpl
import matplotlib.pyplot as plt
BINS=101
assert BINS%2==1
edges=np.linspace(-E,E,BINS+1)
counts={};densities={};hist_rows=[]
for variance in [0.]+VARIANCES:
    a=clean_energy if variance==0 else energy[variance]
    n,_=np.histogram(a,edges)
    rho=n/(len(a)*np.diff(edges))
    assert n.sum()==len(a) and np.isclose(np.dot(rho,np.diff(edges)),1,rtol=0,atol=1e-14)
    counts[variance]=n;densities[variance]=rho
    for j in range(BINS):
        hist_rows.append({'disorder_variance':variance,'left':edges[j],'right':edges[j+1],
                         'count':int(n[j]),'density':float(rho[j])})
pd.DataFrame(hist_rows).to_csv(OUT/'histograms.csv',index=False)
os.environ['TEXINPUTS']=str(OUT/'latex_support')+os.pathsep+os.environ.get('TEXINPUTS','')
mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['CMU Sans Serif'],
 'font.size':8,'axes.labelsize':8,'axes.titlesize':8,'xtick.labelsize':8,'ytick.labelsize':8,
 'text.usetex':True,'pdf.fonttype':42,'xtick.direction':'in','ytick.direction':'in'})
fig,axes=plt.subplots(1,3,figsize=(8.4,3.0),squeeze=False,sharex=True,sharey=True)
colors=['#d62728','#228833','#1565c0','#aa4499','#cc8800','#009999']
for ax,variance,color,letter in zip(axes.flat,VARIANCES,colors,['(a)','(b)','(c)','(d)','(e)','(f)']):
    ax.bar(edges[:-1],counts[variance],width=np.diff(edges),align='edge',
           color=color,alpha=.65,edgecolor=color,linewidth=.3)
    ax.set(title=rf'$W^2={variance:g}$',xlabel=r'Entanglement energy $\varepsilon$',
           xlim=(-E,E),xticks=[-4,0,4])
    ax.tick_params(top=True,right=True)
    ax.text(-.18,1.04,letter,transform=ax.transAxes,fontweight='bold')
for ax in axes[:,0]:ax.set_ylabel('Raw count per bin')
fig.suptitle(r'Disorder only at $x=5,6,14,15$; $20\times32$; $100$ ground states per variance; $A_y=16$; $y_0=0$; $L=0.99$',fontsize=9)
fig.tight_layout(pad=.8)
fig.savefig(OUT/'disordered_entanglement_energy_counts.pdf')
plt.close(fig)
subprocess.run(['pdftoppm','-r','300','-singlefile','-png',str(OUT/'disordered_entanglement_energy_counts.pdf'),str(OUT/'disordered_entanglement_energy_counts')],check=True)
display(Image(filename=str(OUT/'disordered_entanglement_energy_counts.png'),width=1050))''')
md('## Normalized comparison with the clean ground state\nEach panel is normalized separately to unit area within the selected window. The clean panel contains one state; the disordered panels each contain 100.')
code(r'''fig,axes=plt.subplots(2,2,figsize=(8.4,5.6),sharex=True,sharey=True)

for ax,variance,color,letter in zip(axes.flat,[0.]+VARIANCES,['#555555']+colors,['(a)','(b)','(c)','(d)','(e)','(f)','(g)']):
    ax.bar(edges[:-1],densities[variance],width=np.diff(edges),align='edge',
           color=color,alpha=.65,edgecolor=color,linewidth=.3)
    title='Clean; 1 ground state' if variance==0 else rf'$W^2={variance:g}$; 100 ground states'
    ax.set(title=title,xlim=(-E,E),xticks=[-4,0,4])
    ax.tick_params(top=True,right=True)
    ax.text(-.16,1.04,letter,transform=ax.transAxes,fontweight='bold')
for ax in axes[-1]:ax.set_xlabel(r'Entanglement energy $\varepsilon$')
for ax in axes[:,0]:ax.set_ylabel(r'$\rho_{0.99}(\varepsilon)$')
fig.suptitle(r'Disorder only at $x=5,6,14,15$; $20\times32$; hard domain walls; $A_y=16$; $y_0=0$; $L=0.99$',fontsize=9)
fig.tight_layout(pad=.8)
fig.savefig(OUT/'clean_disordered_entanglement_energy_density.pdf')
plt.close(fig)
subprocess.run(['pdftoppm','-r','300','-singlefile','-png',str(OUT/'clean_disordered_entanglement_energy_density.pdf'),str(OUT/'clean_disordered_entanglement_energy_density')],check=True)
display(Image(filename=str(OUT/'clean_disordered_entanglement_energy_density.png'),width=1000))''')
md('## Diagnostics\nAll spectra are finite and within physical bounds; eigensystem orthogonality/residuals and global half filling were checked for every realization. The finite-window histogram conserves counts.')
code(r'''all_diag=[d['diagnostics'] for d in receipts]
diagnostics={'configuration':config,'bins':BINS,'realizations':len(receipts),
 'clean_reference_checks':json.loads((OUT/'input_provenance.json').read_text()),
 'max_eigen_residual':max(d['eigen_residual'] for d in all_diag),
 'max_orthogonality_error':max(d['orthogonality'] for d in all_diag),
 'minimum_half_filling_gap':min(d['half_filling_gap'] for d in all_diag),
 'raw_lambda_min':min(d['raw_lambda_min'] for d in all_diag),
 'raw_lambda_max':max(d['raw_lambda_max'] for d in all_diag),
 'full_space_crosschecks':[d for d in all_diag if 'full_space_spectrum_crosscheck' in d],
 'summary':rows,'all_count_totals_verified':True,'all_density_integrals_verified':True}
(OUT/'diagnostics.json').write_text(json.dumps(diagnostics,indent=2)+'\n')
display(pd.Series({k:v for k,v in diagnostics.items() if isinstance(v,(int,float,bool))}))
print('Complete: 300 ground states; raw-count and normalized-density figures exported.')''')
path=OUT/'disordered_inner_wall_depth1_entanglement_energies.ipynb'
nbformat.write(nb,path)
NotebookClient(nb,timeout=120,kernel_name='python3',resources={'metadata':{'path':str(OUT)}}).execute()
nbformat.write(nb,path)
(OUT/'README.md').write_text('''# Walls plus one inward column: iid Gaussian disorder scan

300 static half-filled pure ground states of the matched hard-wall flattened parent:
100 each at disorder variance 12, 16, 25. Independent Gaussian diagonal potentials
have population mean zero; each x,y,orbital on x=5,6,14,15 has an
independent potential, and all other entries are exactly zero.
Each strength has independent Gaussian draws on the four selected columns. No spatial
demeaning or post-disorder reflattening is applied. Nx=20, Ny=32, x walls=5,15.
The half-system cut has Ay=16, y0=0. Keep |lambda|<=0.99 and use 101 bins.

Run run_analysis.py to acquire/resume; run build_notebook.py for the executed
analysis notebook, raw histogram and clean/disordered normalized comparison.
The clean Hamiltonian, every random potential, spectrum, seed, configuration,
source hashes and realization checksums are saved. Ground-state eigenvectors
are computed transiently and can be reconstructed from the saved Hamiltonian
and potentials; they are not archived as dense frames.

No circuit dynamics or purification data enter this experiment.
''')
files=[f for f in OUT.rglob('*') if f.is_file() and f.name!='completion_manifest.json' and '__pycache__' not in str(f)]
manifest={'status':'complete','realizations':300,'files':{str(f.relative_to(OUT)):{'bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()} for f in files}}
(OUT/'completion_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(OUT)
