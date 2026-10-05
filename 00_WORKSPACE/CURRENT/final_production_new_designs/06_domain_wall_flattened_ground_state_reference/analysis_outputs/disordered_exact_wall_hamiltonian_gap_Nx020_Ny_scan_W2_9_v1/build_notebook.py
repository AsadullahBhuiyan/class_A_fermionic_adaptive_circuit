
from pathlib import Path
import nbformat,json,hashlib,shutil
from nbclient import NotebookClient
OUT=Path(__file__).resolve().parent
REFERENCE=OUT.parent/'disordered_exact_wall_hamiltonian_gap_Nx_Ny040_W2_9_v1'
shutil.copytree(REFERENCE/'latex_support',OUT/'latex_support',dirs_exist_ok=True)
old=nbformat.read(REFERENCE/'hamiltonian_gap_vs_Nx.ipynb',as_version=4)
nb=nbformat.v4.new_notebook(metadata=old.metadata)
def md(s):nb.cells.append(nbformat.v4.new_markdown_cell(s))
def code(s):nb.cells.append(nbformat.v4.new_code_cell(s))
md(r'''# Minimum absolute Hamiltonian eigenvalue versus longitudinal size
Fix $N_x=20$ and vary $N_y=20,32,36,40,44,48,80$, with $W^2=9$ and 100 independent disorder realizations per size. Disorder is an iid zero-mean Gaussian diagonal potential, independent for every $y$ and orbital, confined to $x=5,15$. The canonical CPU overcomplete Wannier (OW) parent uses $\alpha_1=1,\alpha_2=30,n_{\rm shell}=1$, trial orbitals X and hard-wall truncation.

For each Hamiltonian, compute $g_s=\min_i|E_{i,s}|$ at the original energy zero, then plot $\bar g=\sum_s g_s/100$ with $\mathrm{SEM}=\mathrm{std}(g_s;\mathrm{ddof}=1)/\sqrt{100}$. No centering, rescaling, reflattening, clipping or cut-origin averaging. This observable differs from the half-filled particle-hole excitation gap, which is saved separately as a diagnostic. All spectra are reused; no circuit simulation or diagonalization is needed.''')
code(old.cells[1].source)
md('## Configuration, provenance and validation\nValidate original completion receipts, checksums, geometry, disorder support and all 700 sample identities.')
code(r'''from pathlib import Path
import json,hashlib,subprocess
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from IPython.display import display,Image
OUT=Path.cwd()
SOURCE=OUT.parent/'disordered_exact_wall_central_charge_through_ny100_v1'
SQUARE=OUT.parent/'disordered_exact_wall_square_gap_W2_9_v1'
NX=20;SIZES=[20,32,36,40,44,48,80];W2=9;SAMPLES=100
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
provenances={str(root):json.loads((root/'input_provenance.json').read_text()) for root in [SOURCE,SQUARE]}
cfg=provenances[str(SOURCE)]['configuration']
assert cfg['Nx']==NX and cfg['disorder_x_columns']==[5,15] and cfg['samples']==SAMPLES
for key in ['alpha_1','alpha_2','nshell']:
    assert cfg[key]==provenances[str(SQUARE)]['configuration'][key]
rows=[];inputs=[];spectra={};max_gap_error=0.
for ny in tqdm(SIZES,desc='Validate saved spectra',unit='size'):
    root=SQUARE if ny==20 else SOURCE
    folder=root/(f'N{ny:03d}' if ny==20 else f'Ny{ny:03d}')
    identity=provenances[str(root)]['identity']
    receipt=json.loads((folder/'acquisition_complete.json').read_text())
    assert receipt['status']=='complete' and receipt['identity']==identity
    if ny==20:
        assert receipt['N']==ny and receipt['samples']==SAMPLES
        aggregate=folder/'hamiltonian_spectra.npz'
        expected_hash=receipt['sha256']
    else:
        assert receipt['Ny']==ny and receipt['realizations']==SAMPLES*len(cfg['variances'])
        aggregate=folder/'entropy_profiles.npz'
        expected_hash=receipt['cache_sha256']
    assert aggregate.stat().st_size==receipt['bytes'] and sha(aggregate)==expected_hash
    energies=[];seen=[]
    for sid in range(SAMPLES):
        name=f'sample_{sid:03d}.npz' if ny==20 else f'W2_09_sample_{sid:03d}.npz'
        p=folder/'realizations'/name
        r=json.loads(p.with_suffix('.json').read_text())
        assert r['identity']==identity and r['Ny']==ny and r['sample_id']==sid and r['variance']==W2
        assert r.get('Nx',NX)==NX and r['filename']==name
        assert p.stat().st_size==r['bytes'] and sha(p)==r['sha256']
        expected_seed=[provenances[str(root)]['configuration']['seed'],ny]
        if ny!=20:expected_seed.append(cfg['variances'].index(W2))
        expected_seed.append(sid)
        assert r['seed_components']==expected_seed
        with np.load(p) as z:
            assert int(z['Ny'])==ny and int(z['sample_id'])==sid and int(z['variance'])==W2
            if 'Nx' in z:assert int(z['Nx'])==NX
            assert np.array_equal(z['seed_components'],expected_seed)
            e=z['hamiltonian_energies'];v=z['diagonal_potential']
        assert e.shape==(2*NX*ny,) and np.isrealobj(e) and np.isfinite(e).all()
        assert np.all(np.diff(e)>=-1e-12)
        mask=np.isin(np.arange(2*NX*ny)//2%NX,[5,15])
        assert v.shape==e.shape and np.isfinite(v).all() and np.all(v[~mask]==0)
        n=NX*ny;j=int(np.argmin(abs(e)));gap=float(e[n]-e[n-1])
        error=abs(gap-r['diagnostics']['half_filling_gap'])
        assert error<1e-12
        max_gap_error=max(max_gap_error,error)
        rows.append(dict(Nx=NX,Ny=ny,W2=W2,sample_id=sid,min_abs_eigenvalue=float(abs(e[j])),
            nearest_signed_eigenvalue=float(e[j]),half_filling_gap=gap,
            midgap_chemical_potential=float((e[n]+e[n-1])/2)))
        inputs.append(dict(Nx=NX,Ny=ny,sample_id=sid,path=str(p),identity=identity,
                           bytes=r['bytes'],sha256=r['sha256']))
        energies.append(e);seen.append(sid)
    assert seen==list(range(SAMPLES))
    spectra[ny]=np.array(energies)
samples=pd.DataFrame(rows)
samples.to_csv(OUT/'sample_gaps.csv',index=False)
np.savez_compressed(OUT/'hamiltonian_eigenvalues.npz',
 **{f'energies_Ny{ny:03d}':spectra[ny] for ny in SIZES},
 Nx=NX,Ny_values=SIZES,W2=W2,sample_ids=np.arange(SAMPLES))
metadata=dict(Nx=NX,Ny_values=SIZES,W2=W2,samples_per_size=SAMPLES,disorder_x_columns=[5,15],
 disorder='iid Gaussian, mean zero; independent y and orbital',
 estimator='mean of per-Hamiltonian min(abs(E)); original zero',uncertainty='one SEM',
 new_diagonalizations=0,canonical_parent_sources=provenances[str(SOURCE)]['sources'])
(OUT/'input_provenance.json').write_text(json.dumps(dict(configuration=metadata,
 source_provenances=provenances,inputs=inputs),indent=2)+'\n')
display(pd.Series(metadata))''')
md('## Disorder statistics\nThe rescaled column $N_y\\bar g$ is a diagnostic for inverse-length behavior; no scaling exponent is fitted.')
code(r'''summary_rows=[]
for ny in SIZES:
    t=samples[samples.Ny==ny]
    g=t.min_abs_eigenvalue.to_numpy()
    assert len(g)==SAMPLES and np.isfinite(g).all() and np.all(g>=0)
    mean=float(g.mean());sem=float(g.std(ddof=1)/np.sqrt(SAMPLES))
    summary_rows.append(dict(Nx=NX,Ny=ny,W2=W2,samples=SAMPLES,
      mean_min_abs_eigenvalue=mean,sem=sem,Ny_times_mean=ny*mean,Ny_times_sem=ny*sem,
      median=float(np.median(g)),minimum=float(g.min()),maximum=float(g.max()),
      mean_half_filling_gap=float(t.half_filling_gap.mean())))
summary=pd.DataFrame(summary_rows)
summary.to_csv(OUT/'gap_summary.csv',index=False)
np.savez_compressed(OUT/'gap_statistics.npz',Nx=NX,Ny_values=SIZES,W2=W2,
 sample_ids=np.arange(SAMPLES),
 min_abs_eigenvalues=np.array([samples[samples.Ny==ny].min_abs_eigenvalue.to_numpy() for ny in SIZES]))
prior=pd.read_csv(OUT.parent/'disordered_exact_wall_hamiltonian_gap_Nx_Ny040_W2_9_v1/gap_summary.csv')
anchor_error=abs(float(summary.loc[summary.Ny==40,'mean_min_abs_eigenvalue'].iloc[0])-
                 float(prior.loc[prior.Nx==20,'mean_min_abs_eigenvalue'].iloc[0]))
assert anchor_error<1e-14
display(summary)''')
md('## Minimum absolute eigenvalue versus Ny\nError bars are one SEM across independent disorder realizations. The dashed inverse-length guide passes through the Ny40 mean and is not a fit.')
code(r'''import matplotlib as mpl
import matplotlib.pyplot as plt
os.environ['TEXINPUTS']=str(OUT/'latex_support')+os.pathsep+os.environ.get('TEXINPUTS','')
mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['CMU Sans Serif'],'font.size':9,
 'axes.labelsize':10,'axes.titlesize':10,'text.usetex':True,'text.latex.preamble':r'\usepackage{amsmath,amssymb}',
 'xtick.direction':'in','ytick.direction':'in','legend.frameon':False,'legend.fontsize':8,'pdf.fonttype':42})
fig,ax=plt.subplots(figsize=(5.2,3.7))
ax.errorbar(summary.Ny,summary.mean_min_abs_eigenvalue,yerr=summary['sem'],fmt='o-',color='#1565c0',
 mfc='white',ms=4,capsize=3,lw=1,label='Disorder mean')
coefficient=40*float(summary.loc[summary.Ny==40,'mean_min_abs_eigenvalue'].iloc[0])
grid=np.linspace(min(SIZES),max(SIZES),300)
ax.plot(grid,coefficient/grid,'--',color='gray',lw=1,label=r'$1/N_y$ guide (anchored at $N_y=40$)')
ax.set(xlabel=r'Longitudinal size $N_y$',ylabel=r'$\overline{\min_i|E_i|}$',ylim=(0,None))
ax.tick_params(top=True,right=True)
ax.legend(loc='upper right')
fig.suptitle(r'$N_x=20$; $W^2=9$; disorder only at $x=5,15$'+'\n'+
 r'$100$ realizations per size; original energy zero; error bars: one SEM',fontsize=9)
fig.tight_layout(pad=1)
fig.savefig(OUT/'minimum_absolute_eigenvalue_vs_Ny.pdf');plt.close(fig)
subprocess.run(['pdftoppm','-r','300','-singlefile','-png',str(OUT/'minimum_absolute_eigenvalue_vs_Ny.pdf'),str(OUT/'minimum_absolute_eigenvalue_vs_Ny')],check=True)
display(Image(filename=str(OUT/'minimum_absolute_eigenvalue_vs_Ny.png'),width=900))''')
md('## Numerical diagnostics')
code(r'''diagnostics=dict(total_hamiltonians=len(samples),samples_per_size=samples.groupby('Ny').size().to_dict(),
 max_saved_half_filling_gap_error=max_gap_error,prior_Nx20_Ny40_mean_error=anchor_error,
 exact_zero_minima=int((samples.min_abs_eigenvalue==0).sum()),
 near_zero_minima=int((samples.min_abs_eigenvalue<=1e-10).sum()),
 checks=dict(completion_receipts=True,input_bytes_and_checksums=True,sample_ids_and_seeds=True,
 finite_real_sorted_spectra=True,wall_only_potential=True),
 scaling_guide=dict(coefficient=coefficient,anchor_Ny=40,fitted=False))
(OUT/'diagnostics.json').write_text(json.dumps(diagnostics,indent=2)+'\n')
display(pd.Series(diagnostics))
print('Complete: all 700 saved Hamiltonian spectra validated.')''')
p=OUT/'hamiltonian_gap_vs_Ny.ipynb'
nbformat.write(nb,p)
NotebookClient(nb,timeout=300,kernel_name='python3',resources={'metadata':{'path':str(OUT)}}).execute()
nbformat.write(nb,p)
files={p.name:dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
 for p in OUT.iterdir() if p.is_file() and p.name not in ['completion_manifest.json','run.log','exit_code.txt']}
(OUT/'completion_manifest.json').write_text(json.dumps(dict(status='complete',hamiltonians=700,Nx=20,
 sizes=[20,32,36,40,44,48,80],files=files),indent=2)+'\n')
print('COMPLETE:',OUT)
