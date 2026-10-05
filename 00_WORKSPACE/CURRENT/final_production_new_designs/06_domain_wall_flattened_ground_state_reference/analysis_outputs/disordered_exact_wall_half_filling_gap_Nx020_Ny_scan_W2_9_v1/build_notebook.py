
from pathlib import Path
import nbformat,json,hashlib,shutil
from nbclient import NotebookClient
OUT=Path(__file__).resolve().parent
SRC=OUT.parent/'disordered_exact_wall_hamiltonian_gap_Nx020_Ny_scan_W2_9_v1'
shutil.copytree(SRC/'latex_support',OUT/'latex_support',dirs_exist_ok=True)
old=nbformat.read(SRC/'hamiltonian_gap_vs_Ny.ipynb',as_version=4)
nb=nbformat.v4.new_notebook(metadata=old.metadata)
def md(s):nb.cells.append(nbformat.v4.new_markdown_cell(s))
def code(s):nb.cells.append(nbformat.v4.new_code_cell(s))
md(r'''# Half-filling excitation gap and purification-gap audit
For the disordered Hamiltonian, fix $N_x=20$, $W^2=9$, and disorder at $x=5,15$ only. At each $N_y=20,32,36,40,44,48,80$, average 100 realization-resolved gaps
$$\Delta_{H,s}=E_{N_xN_y+1,s}-E_{N_xN_y,s}.$$
Energies are sorted ascending with one-based indices. This is the lowest fixed-particle-number excitation energy for the half-filled free-fermion ground state. It is invariant under a common energy shift.

The purification comparison uses bundle 13: maximally mixed active slab, $N_x=20$, 100 Born trajectories per size, no added static disorder, and
$$\epsilon_j(T)=\log[(1-\nu_j(T))/\nu_j(T)],\qquad g_{\epsilon,s}(T)=\min_j|\epsilon_{j,s}(T)|,\qquad
\Delta_{{\rm pur},s}(T)=g_{\epsilon,s}(T)/(2T).$$
This is the unrestricted Fock-space modular gap divided by $2T$. It equals the gap between the two largest many-body log squared singular values divided by $2T$, checked below. This is distinct from a fixed-charge particle-hole gap. The initial active-slab state includes all charge sectors. Both analyses take a minimum or level difference within each realization before averaging; error bars are ordinary one SEM. No circuit or diagonalization is rerun.''')
code(old.cells[1].source)
md('## Load and validate the saved Hamiltonian spectra')
code(r'''from pathlib import Path
import json,hashlib,sys,subprocess
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from IPython.display import display,Image
OUT=Path.cwd();SRC=OUT.parent/'disordered_exact_wall_hamiltonian_gap_Nx020_Ny_scan_W2_9_v1'
BASE=OUT.parents[2]
PUR=BASE/'13_maxmix_manybody_lyapunov_4ny'
NX=20;W2=9;SAMPLES=100;SIZES=[20,32,36,40,44,48,80]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
manifest=json.loads((SRC/'completion_manifest.json').read_text())
assert manifest['status']=='complete' and manifest['hamiltonians']==700
inputs=[]
for name in ['hamiltonian_eigenvalues.npz','sample_gaps.csv','input_provenance.json']:
    p=SRC/name;r=manifest['files'][name]
    assert p.stat().st_size==r['bytes'] and sha(p)==r['sha256']
    inputs.append(dict(path=str(p),**r))
prior=pd.read_csv(SRC/'sample_gaps.csv');rows=[];summary=[]
with np.load(SRC/'hamiltonian_eigenvalues.npz') as z:
    assert int(z['Nx'])==NX and int(z['W2'])==W2
    assert np.array_equal(z['Ny_values'],SIZES) and np.array_equal(z['sample_ids'],np.arange(SAMPLES))
    for ny in tqdm(SIZES,desc='Half-filling excitation gaps',unit='size'):
        e=z[f'energies_Ny{ny:03d}'];n=NX*ny
        assert e.shape==(SAMPLES,2*n) and np.isfinite(e).all() and np.all(np.diff(e,axis=1)>=-1e-12)
        g=e[:,n]-e[:,n-1]
        assert np.all(g>0)
        reference=prior[prior.Ny==ny].sort_values('sample_id').half_filling_gap.to_numpy()
        assert np.max(abs(g-reference))<1e-14
        for sid,value in enumerate(g):
            rows.append(dict(Nx=NX,Ny=ny,W2=W2,sample_id=sid,half_filling_gap=float(value)))
        mean=float(g.mean());sem=float(g.std(ddof=1)/np.sqrt(SAMPLES))
        summary.append(dict(Nx=NX,Ny=ny,W2=W2,samples=SAMPLES,mean_gap=mean,sem=sem,
             Ny_mean_gap=ny*mean,Ny_sem=ny*sem,minimum=float(g.min()),median=float(np.median(g))))
samples=pd.DataFrame(rows);summary=pd.DataFrame(summary)
samples.to_csv(OUT/'half_filling_gap_samples.csv',index=False)
summary.to_csv(OUT/'half_filling_gap_summary.csv',index=False)
display(summary)''')
md('## Fixed-charge Hamiltonian gap\nThe right panel multiplies by Ny to test inverse-length behavior visually. The dashed guide is anchored at Ny40, not fitted.')
code(r'''import matplotlib as mpl
import matplotlib.pyplot as plt
os.environ['TEXINPUTS']=str(OUT/'latex_support')+os.pathsep+os.environ.get('TEXINPUTS','')
mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['CMU Sans Serif'],'font.size':9,
 'axes.labelsize':10,'text.usetex':True,'text.latex.preamble':r'\usepackage{amsmath,amssymb}',
 'xtick.direction':'in','ytick.direction':'in','legend.frameon':False,'legend.fontsize':8})
fig,axes=plt.subplots(1,2,figsize=(7.05,3.25))
coefficient=40*float(summary.loc[summary.Ny==40,'mean_gap'].iloc[0])
grid=np.linspace(20,80,300)
axes[0].plot(grid,coefficient/grid,'--',color='gray',lw=1,label=r'$1/N_y$ guide')
for ax,col,err in [(axes[0],'mean_gap','sem'),(axes[1],'Ny_mean_gap','Ny_sem')]:
    ax.errorbar(summary.Ny,summary[col],yerr=summary[err],fmt='o-',color='#1565c0',
                mfc='white',ms=4,capsize=2,lw=1)
    ax.set(xlabel=r'$N_y$',ylim=(0,None))
    ax.tick_params(top=True,right=True)
axes[0].set_ylabel(r'$\overline{\Delta_H}$')
axes[1].set_ylabel(r'$N_y\,\overline{\Delta_H}$')
axes[1].axhline(coefficient,color='gray',ls='--',lw=1)
axes[0].legend()
for ax,label in zip(axes,['(a)','(b)']):ax.text(-.18,1.03,label,transform=ax.transAxes)
fig.suptitle(r'$N_x=20$; $W^2=9$; disorder only at $x=5,15$'+'\n'+
 r'$\Delta_{H,s}=E_{N_xN_y+1,s}-E_{N_xN_y,s}$; 100 realizations; error bars: one SEM',fontsize=9)
fig.tight_layout()
fig.savefig(OUT/'half_filling_gap_vs_Ny.pdf');plt.close(fig)
subprocess.run(['pdftoppm','-r','300','-singlefile','-png',str(OUT/'half_filling_gap_vs_Ny.pdf'),str(OUT/'half_filling_gap_vs_Ny')],check=True)
display(Image(filename=str(OUT/'half_filling_gap_vs_Ny.png'),width=1100))''')
md('## Purification spectra: recheck the estimator and its time dependence\nRead all 140 raw shards, validate their pinned source/configuration identities and checksums, and recompute the gap directly from occupations. Cross-check the stored soft-mode costs and top two many-body levels. Caps map to infinite costs. Audit T/Ny=1,2,3,4 plus fixed T=40. No asymptotic convergence is assumed.')
code(r'''sys.path.insert(0,str(PUR))
import analyze_campaign as campaign
paths=sorted((campaign.DATA_ROOT/'results').rglob('*.npz'))
assert len(paths)==140
purrows=[];max_soft_error=0.;max_manybody_error=0.
for p in tqdm(paths,desc='Verify purification shards',unit='shard'):
    r=json.loads(p.with_name(p.stem+'.complete.json').read_text())
    campaign.validate_completion(r,p)
    assert p.stat().st_size==r['result_bytes'] and sha(p)==r['result_sha256']
    inputs.append(dict(path=str(p),bytes=r['result_bytes'],sha256=r['result_sha256']))
    with np.load(p) as z:
        ny=int(z['Ny']);times=z['spectrum_cycles'];ids=z['sample_indices']
        assert int(z['Nx'])==NX and np.array_equal(ids,r['sample_indices'])
        assert str(z['configuration_hash'])==campaign.CONFIGURATION_HASH
        assert int(z['transfer_mode_count'])==22*ny
        for label,T in [('1Ny',ny),('2Ny',2*ny),('3Ny',3*ny),('4Ny',4*ny),('fixed40',40)]:
            ix=int(np.searchsorted(times,T));assert times[ix]==T
            nu=z['occupations'][:,ix,:];caps=z['cap_mask'][:,ix,:]
            assert np.isfinite(nu).all()
            assert np.array_equal(caps,(nu<=campaign.CAP_TOLERANCE)|(nu>=1-campaign.CAP_TOLERANCE))
            costs=np.full_like(nu,np.inf);v=nu[~caps]
            assert np.all((v>0)&(v<1))
            costs[~caps]=abs(np.log1p(-v)-np.log(v))
            raw=costs.min(axis=1)
            assert np.isfinite(raw).all()
            soft=z['soft_mode_flip_costs'][:,ix,:].min(axis=1)
            levels=z['leading_log_sigma2'][:,ix,:2]
            e1=float(np.max(abs(raw-soft)))
            e2=float(np.max(abs(raw-(levels[:,0]-levels[:,1]))))
            max_soft_error=max(max_soft_error,e1);max_manybody_error=max(max_manybody_error,e2)
            assert e1<1e-12 and e2<5e-10
            for j,sid in enumerate(ids):
                purrows.append(dict(Ny=ny,sample_id=int(sid),checkpoint=label,T=T,
                    raw_modular_gap=float(raw[j]),purification_gap=float(raw[j]/(2*T))))
pur=pd.DataFrame(purrows);pur.to_csv(OUT/'purification_gap_samples.csv',index=False)
pur_summary=[]
for (ny,label),t in pur.groupby(['Ny','checkpoint']):
    assert sorted(t.sample_id.tolist())==list(range(SAMPLES))
    item=dict(Ny=int(ny),checkpoint=label,T=int(t.T.iloc[0]) if False else int(t['T'].iloc[0]),samples=len(t))
    for name in ['raw_modular_gap','purification_gap']:
        v=t[name].to_numpy()
        item['mean_'+name]=float(v.mean());item['sem_'+name]=float(v.std(ddof=1)/np.sqrt(SAMPLES))
    pur_summary.append(item)
pur_summary=pd.DataFrame(pur_summary)
pur_summary.to_csv(OUT/'purification_gap_summary.csv',index=False)
reference=pd.read_csv(PUR/'analysis_outputs/endpoint_gap_definitions_v1/endpoint_gap_definitions_summary.csv')
endpoint=pur_summary[pur_summary.checkpoint=='4Ny'].sort_values('Ny')
endpoint_error=float(np.max(abs(endpoint.mean_purification_gap.to_numpy()-reference.mean_lyapunov_gap_Delta_lambda.to_numpy())))
assert endpoint_error<1e-13
change=[]
for ny in sorted(pur.Ny.unique()):
    early=pur[(pur.Ny==ny)&(pur.checkpoint=='2Ny')].sort_values('sample_id').purification_gap.to_numpy()
    late=pur[(pur.Ny==ny)&(pur.checkpoint=='4Ny')].sort_values('sample_id').purification_gap.to_numpy()
    diff=late-early
    change.append(dict(Ny=int(ny),mean_2Ny=float(early.mean()),mean_4Ny=float(late.mean()),
       fractional_increase=float(late.mean()/early.mean()-1),
       paired_difference=float(diff.mean()),paired_sem=float(diff.std(ddof=1)/np.sqrt(SAMPLES))))
change=pd.DataFrame(change);change.to_csv(OUT/'purification_time_dependence.csv',index=False)
display(pur_summary);display(change)''')
md('## Purification gap at several aspect ratios\nThe endpoint T=4Ny is a finite-time rate. Curves at different T/Ny test time dependence; a stable size exponent alone is not proof of long-time convergence.')
code(r'''fig,ax=plt.subplots(figsize=(5.2,3.7))
for label,color,marker,style in [('2Ny','#d62728','^',':'),('3Ny','#2ca02c','s','--'),('4Ny','#1565c0','o','-')]:
    t=pur_summary[pur_summary.checkpoint==label].sort_values('Ny')
    ax.errorbar(t.Ny,t.mean_purification_gap,yerr=t.sem_purification_gap,color=color,marker=marker,ls=style,
                mfc='white',ms=4,capsize=2,lw=1,label=r'$T='+label[0]+r'N_y$')
ax.set(xlabel=r'$N_y$',ylabel=r'$\overline{\Delta_{\mathrm{pur}}(T)}$',ylim=(0,None))
ax.tick_params(top=True,right=True);ax.legend()
fig.suptitle(r'Purification: $N_x=20$; 100 Born trajectories per size'+'\n'+
 r'$\Delta_{\mathrm{pur}}(T)=\min_j|\log[(1-\nu_j)/\nu_j]|/(2T)$; error bars: one SEM',fontsize=9)
fig.tight_layout()
fig.savefig(OUT/'purification_gap_time_audit.pdf');plt.close(fig)
subprocess.run(['pdftoppm','-r','300','-singlefile','-png',str(OUT/'purification_gap_time_audit.pdf'),str(OUT/'purification_gap_time_audit')],check=True)
display(Image(filename=str(OUT/'purification_gap_time_audit.png'),width=900))''')
md('## Diagnostics and interpretation\nFor a Gaussian state over unrestricted occupations, the modular ground configuration fills every negative modular energy. The cheapest excitation flips one occupation, costing min|epsilon|. Fixing the particle number instead requires a particle-hole pair. Thus the original purification gap is mathematically appropriate to its ensemble. It is not the static disordered-parent gap, and finite-time estimates should not be labeled asymptotic without a convergence study.')
code(r'''diagnostics=dict(hamiltonians=700,purification_trajectories=700,purification_shards=140,
 maximum_direct_soft_mode_error=max_soft_error,maximum_direct_manybody_gap_error=max_manybody_error,
 maximum_prior_endpoint_error=endpoint_error,cap_tolerance=campaign.CAP_TOLERANCE,
 equilibrium_gap='fixed half-filling particle-hole excitation',
 purification_gap='unrestricted Fock-space modular gap/(2T); finite-time Lyapunov gap',
 known_prior_inventory_note='Historical aggregate manifest digest discrepancy already documented by bundle 13; each raw NPZ verified here against its identity-checked completion receipt.')
(OUT/'diagnostics.json').write_text(json.dumps(diagnostics,indent=2)+'\n')
(OUT/'input_provenance.json').write_text(json.dumps(dict(inputs=inputs,source_analysis=str(SRC),
 canonical_purification_entry_point='classA_U1FGTN_gpu.run_markov_circuit',
 purification_config_hash=campaign.CONFIGURATION_HASH,purification_source_hashes=campaign.SOURCE_HASHES,
 static_configuration=dict(Nx=NX,W2=W2,Ny_values=SIZES,disorder_x_columns=[5,15],samples=100)),indent=2)+'\n')
display(pd.Series(diagnostics))''')
p=OUT/'half_filling_and_purification_gaps.ipynb'
nbformat.write(nb,p)
NotebookClient(nb,timeout=600,kernel_name='python3',resources={'metadata':{'path':str(OUT)}}).execute()
nbformat.write(nb,p)
files={p.name:dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
 for p in OUT.iterdir() if p.is_file() and p.name not in ['completion_manifest.json','run.log','exit_code.txt']}
(OUT/'completion_manifest.json').write_text(json.dumps(dict(status='complete',files=files),indent=2)+'\n')
print('COMPLETE',OUT)
