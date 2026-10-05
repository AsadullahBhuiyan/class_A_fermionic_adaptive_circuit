from pathlib import Path
import nbformat,json,hashlib,shutil
from nbclient import NotebookClient
OUT=Path(__file__).resolve().parent
BASE=OUT.parents[2]
shutil.copytree(OUT.parent/'disordered_inner_wall_depth1_strength_scan_n20x32_v1/latex_support',OUT/'latex_support',dirs_exist_ok=True)
nb=nbformat.v4.new_notebook(metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'}})
def md(s):nb.cells.append(nbformat.v4.new_markdown_cell(s))
def code(s):nb.cells.append(nbformat.v4.new_code_cell(s))
md(r'''# Sample-count convergence of equilibrium entanglement-level statistics
Fix $N_x=20,N_y=32$, hard domain walls, $\alpha_1=1,\alpha_2=30,n_{\rm shell}=1$, half filling, disorder variance $W^2=9$, and independent Gaussian diagonal potentials only at $x=5,6,14,15$. The subsystem is all $x$, both orbitals, $y=0,\ldots,15$; retain $|\lambda|\le0.99$. There is no reflattening after disorder.
The first 100 states reproduce the preceding strength scan, with 900 new independent states added. Compare nested prefixes of 100, 250, 500, and 1000 realizations, plus the disjoint new-900 set and ten independent 100-state blocks. Physical size, energy window, disorder law, and estimator are held fixed. Increasing samples estimates the same ensemble more precisely; it is not a thermodynamic or disorder-strength limit.

Within each realization sort $\varepsilon=\log[(1-\lambda)/(1+\lambda)]$ and form
$$R_n=\frac{\min(\delta_n,\delta_{n+1})}{\max(\delta_n,\delta_{n+1})},\qquad
\delta_n=\varepsilon_{n+1}-\varepsilon_n.$$
Only then pool ratios. Mean errors use a delete-whole-realization jackknife.
The folded GUE approximation is
$$P_{\rm GUE}^{(3)}(R)=\frac{81\sqrt3}{2\pi}\frac{(R+R^2)^2}{(1+R+R^2)^4},\qquad 0\le R\le1.$$
Its mean is $2\sqrt3/\pi-1/2\simeq0.60266$; the large-matrix GUE mean is approximately $0.5996$. These are distinguished in the figures and diagnostics. Reference: [Atas et al., PRL 110, 084101](https://arxiv.org/abs/1212.5611).
The maximum CDF separation from this approximation is computed from unbinned ratios, checking both sides of every empirical CDF jump. Its 95% percentile intervals resample whole realizations (500 bootstrap replicates), not individual overlapping ratios. No independent-level p-values are used. Full retained spectra are pooled within realizations without symmetry/wall-sector separation; no universality or many-body-chaos claim is assumed.''')
code('''import os
CPU_RANGE=(8,15)
cpus=list(range(CPU_RANGE[0],CPU_RANGE[1]+1))
assert set(cpus).issubset(os.sched_getaffinity(0))
os.sched_setaffinity(0,cpus)
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[key]=str(len(cpus))
print('Allocated CPUs:',cpus)''')
md('## Configuration and verified spectra\nThe companion runner uses the canonical CPU class for the parent Hamiltonian. Load all 1000 receipts, verify every checksum and ID, and check the first 100 states against the previous scan.')
code(r'''from pathlib import Path
import json,hashlib,subprocess,sys
import numpy as np
import pandas as pd
from scipy.integrate import quad,cumulative_simpson
from tqdm.auto import tqdm
from IPython.display import display,Image
OUT=Path.cwd();BASE=OUT.parents[2]
RUN_ACQUISITION=False
if RUN_ACQUISITION:subprocess.run([sys.executable,str(OUT/'run_analysis.py')],check=True)
PREFIXES=[100,250,500,1000]
L=.99
BOOTSTRAPS=500
BOOTSTRAP_SEED=2026092705
complete=json.loads((OUT/'acquisition_complete.json').read_text())
assert complete['status']=='complete' and complete['realizations']==1000
config=complete['configuration']
display(pd.Series(config))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
PREV=OUT.parent/'disordered_inner_wall_depth1_strength_scan_n20x32_v1'
old_manifest=json.loads((PREV/'completion_manifest.json').read_text())
oldpath=PREV/'spectra_and_energies.npz'
assert sha(oldpath)==old_manifest['files'][oldpath.name]['sha256']
with np.load(oldpath) as z:old_spectra=z['centered_spectra_5'];old_potential=z['diagonal_potential_5']
spectra=[];potentials=[];receipts=[]
mask=np.isin(np.arange(1280)//2%20,[5,6,14,15])
for sid in tqdm(range(1000),desc='Verify equilibrium realizations',unit='state'):
    p=OUT/'realizations'/f'variance_00_sample_{sid:03}.npz'
    d=json.loads(p.with_suffix('.json').read_text())
    assert d['identity']==complete['identity'] and d['sample_id']==sid and d['variance']==9
    assert d['filename']==p.name and d['bytes']==p.stat().st_size and d['sha256']==sha(p)
    with np.load(p) as z:
        assert int(z['sample_id'])==sid and float(z['variance'])==9
        assert np.array_equal(z['seed_components'],[2026092703,5,sid])
        lam=z['centered_eigenvalues'];v=z['diagonal_potential']
        assert lam.shape==(640,) and v.shape==(1280,)
        assert np.isfinite(lam).all() and np.max(abs(lam))<=1+1e-8
        assert np.isfinite(v).all() and np.all(v[~mask]==0) and np.count_nonzero(v)==256
        spectra.append(lam);potentials.append(v)
    receipts.append(d)
spectra=np.array(spectra);potentials=np.array(potentials)
prefix_error=float(np.max(abs(spectra[:100]-old_spectra)))
assert prefix_error<1e-10 and np.array_equal(potentials[:100],old_potential)
SMALL=BASE/'09_pure_tangent_replay_acquisition/analysis_outputs/half_system_centered_spectrum_n20_sizes_hard_alpha1_v1'
nepath=SMALL/'centered_spectra_Ny032.npz'
expected=json.loads((SMALL/'overlay_diagnostics.json').read_text())['sizes']['32']['spectra_sha256']
assert sha(nepath)==expected
with np.load(nepath) as z:
    assert np.array_equal(z['sample_ids'],np.arange(100))
    assert np.array_equal(z['subsystem_indices'],np.arange(640))
    ne_spectra=z['eigenvalues']
np.savez_compressed(OUT/'spectra_and_potentials.npz',centered_spectra=spectra,
 diagonal_potential=potentials,sample_ids=np.arange(1000),disorder_mask=mask,L=L,variance=9.)
provenance={'configuration':config,'acquisition_identity':complete['identity'],
 'prefix_check':{'previous_path':str(oldpath),'previous_sha256':sha(oldpath),'max_spectral_error':prefix_error,
 'identical_first100_potentials':True},
 'nonequilibrium_reference':{'path':str(nepath),'sha256':sha(nepath),'samples':100},
 'reference_paper':'https://arxiv.org/abs/1212.5611'}
(OUT/'analysis_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
print('Verified 1000 unique IDs, first 100 identical potentials, and all source checksums.')''')
md('## Per-realization statistics and convergence\nCompare nested sample prefixes and independent blocks. Nested-prefix estimates share data and are not independent measurements. The nonequilibrium reference remains the existing 100 trajectories.')
code(r'''def extract(a):
    ratios=[];ids=[];energies=[];eids=[];rows=[]
    for sid,lam in enumerate(a):
        lam=lam[abs(lam)<=L]
        e=np.sort(np.log1p(-lam)-np.log1p(lam))
        assert np.isfinite(e).all() and len(e)>=3
        d=np.diff(e);den=np.maximum(d[:-1],d[1:])
        assert np.all(d>=0) and np.all(den>1e-12)
        r=np.minimum(d[:-1],d[1:])/den
        assert np.all((r>=0)&(r<=1))
        ratios.extend(r);ids.extend([sid]*len(r));energies.extend(e);eids.extend([sid]*len(e))
        rows.append({'sample_id':sid,'levels':len(e),'ratio_count':len(r),
            'ratio_sum':float(r.sum()),'sample_mean_r':float(r.mean()),'minimum_gap':float(d.min())})
    return np.array(ratios),np.array(ids),np.array(energies),np.array(eids),pd.DataFrame(rows)
r,ids,energies,eids,sample_table=extract(spectra)
nr,nids,ne,neids,ne_table=extract(ne_spectra)
sample_table.to_csv(OUT/'sample_statistics.csv',index=False)
np.savez_compressed(OUT/'level_statistics.npz',ratios=r,ratio_sample_ids=ids,
 energies=energies,energy_sample_ids=eids,nonequilibrium_ratios=nr,nonequilibrium_sample_ids=nids,
 nonequilibrium_energies=ne,nonequilibrium_energy_ids=neids)
def gue_pdf(t):return 81*np.sqrt(3)/(2*np.pi)*(t+t*t)**2/(1+t+t*t)**4
gue_mean=2*np.sqrt(3)/np.pi-.5
assert abs(quad(gue_pdf,0,1,epsabs=1e-12)[0]-1)<1e-12
assert abs(quad(lambda t:t*gue_pdf(t),0,1,epsabs=1e-12)[0]-gue_mean)<1e-12
grid=np.linspace(0,1,131073)
gue_cdf_grid=cumulative_simpson(gue_pdf(grid),x=grid,initial=0)
assert max(abs(np.interp(t,grid,gue_cdf_grid)-quad(gue_pdf,0,t,epsabs=1e-12)[0])
           for t in np.linspace(0,1,21))<1e-9
def distance(sorted_r,weights):
    total=weights.sum();cum=np.cumsum(weights)/total
    target=np.interp(sorted_r,grid,gue_cdf_grid)
    return float(max(np.max(abs(cum-target)),np.max(abs(cum-weights/total-target))))
rng=np.random.default_rng(BOOTSTRAP_SEED)
def describe(rr,ii,start,end,label,bootstrap=True):
    keep=(ii>=start)&(ii<end);a=rr[keep];sid=ii[keep]-start;S=end-start
    order=np.argsort(a,kind='stable');ar=a[order];ss=sid[order]
    count=np.bincount(sid,minlength=S);sums=np.bincount(sid,weights=a,minlength=S)
    leave=(sums.sum()-sums)/(count.sum()-count)
    se=float(np.sqrt((S-1)/S*np.sum((leave-leave.mean())**2)))
    D=distance(ar,np.ones(len(ar)))
    row={'group':label,'sample_start':start,'sample_stop':end,'samples':S,'ratio_count':len(a),
      'mean_r':float(a.mean()),'mean_r_jackknife_se':se,'cdf_max_to_gue_approximation':D}
    vals=[]
    if bootstrap:
        for b in range(BOOTSTRAPS):
            mult=rng.multinomial(S,np.full(S,1/S))
            vals.append(distance(ar,mult[ss]))
        lo,hi=np.quantile(vals,[.025,.975])
        row.update(cdf_max_ci_low=float(lo),cdf_max_ci_high=float(hi))
    return row,np.array(vals)
rows=[];boots={}
for S in tqdm(PREFIXES,desc='Sample-count convergence',unit='prefix'):
    row,b=describe(r,ids,0,S,f'first_{S}');rows.append(row);boots[f'first_{S}']=b
summary=pd.DataFrame(rows)
holdout,boots['new_900']=describe(r,ids,100,1000,'new_900')
ne_row,boots['nonequilibrium_100']=describe(nr,nids,0,100,'nonequilibrium_100')
blocks=[describe(r,ids,k*100,(k+1)*100,f'block_{k}',bootstrap=False)[0] for k in range(10)]
summary.to_csv(OUT/'sample_count_convergence.csv',index=False)
pd.DataFrame([holdout,ne_row]).to_csv(OUT/'holdout_and_nonequilibrium.csv',index=False)
pd.DataFrame(blocks).to_csv(OUT/'independent_100_sample_blocks.csv',index=False)
np.savez_compressed(OUT/'distance_bootstrap.npz',**boots,seed=BOOTSTRAP_SEED,replicates=BOOTSTRAPS)
display(summary)
display(pd.DataFrame([holdout,ne_row]))
print('Independent 100-state block means:',[x['mean_r'] for x in blocks])''')
md('## Distribution and sample-count convergence\nTop: normalized bar histograms for 100 and 1000 realizations, with the same bins and GUE approximation. Bottom left: pooled mean with one-SE realization-jackknife errors; bottom right: distance from the GUE approximate CDF with 95% whole-realization bootstrap intervals. The latter measures mismatch, not a p-value.')
code(r'''import matplotlib as mpl
import matplotlib.pyplot as plt
os.environ['TEXINPUTS']=str(OUT/'latex_support')+os.pathsep+os.environ.get('TEXINPUTS','')
mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['CMU Sans Serif'],
 'font.size':10,'axes.labelsize':11,'axes.titlesize':11,'xtick.labelsize':10,'ytick.labelsize':10,
 'text.usetex':True,'text.latex.preamble':r'\usepackage{amsmath,amssymb}',
 'pdf.fonttype':42,'xtick.direction':'in','ytick.direction':'in','legend.fontsize':9,'legend.frameon':False})
R_BINS=50
edges=np.linspace(0,1,R_BINS+1)
histrows=[]
for S in PREFIXES:
    a=r[ids<S];counts,_=np.histogram(a,edges);rho=counts/(len(a)*np.diff(edges))
    assert counts.sum()==len(a) and np.isclose(np.dot(rho,np.diff(edges)),1)
    histrows.extend({'samples':S,'left':edges[j],'right':edges[j+1],
                     'count':int(counts[j]),'density':float(rho[j])} for j in range(R_BINS))
pd.DataFrame(histrows).to_csv(OUT/'ratio_histograms.csv',index=False)
fig,axes=plt.subplots(2,2,figsize=(8.4,6.5))
for ax,S,color in zip(axes[0],[100,1000],['#228833','#1565c0']):
    a=r[ids<S];counts,_=np.histogram(a,edges);rho=counts/(len(a)*np.diff(edges));nz=counts>0
    ax.bar(edges[:-1][nz],rho[nz],width=np.diff(edges)[nz],align='edge',
           color=color,alpha=.6,edgecolor=color,linewidth=.3,label='Equilibrium')
    t=np.linspace(.0001,1,1000)
    ax.plot(t,gue_pdf(t),'--',color='black',lw=1.2,label='GUE approximation')
    ax.set(xlim=(0,1),yscale='log',ylim=(.01,30),xlabel=r'Adjacent-gap ratio $R$',
           ylabel=r'$P(R)$',title=rf'$S={S}$ realizations')
    ax.legend(loc='upper left')
ax=axes[1,0]
ax.errorbar(summary.samples,summary.mean_r,yerr=summary.mean_r_jackknife_se,
            fmt='o-',color='#1565c0',mfc='white',capsize=3,label='Equilibrium')
ax.axhline(.5996,color='black',ls='--',lw=1,label=r'GUE, $\langle R\rangle\simeq0.5996$')
ax.axhline(ne_row['mean_r'],color='#777777',ls=':',lw=1,label='Nonequilibrium (100)')
ax.axhspan(ne_row['mean_r']-ne_row['mean_r_jackknife_se'],ne_row['mean_r']+ne_row['mean_r_jackknife_se'],
           color='gray',alpha=.1)
ax.set(xlabel=r'Realizations $S$',ylabel=r'Mean ratio $\langle R\rangle$',ylim=(.56,.96),
       xticks=PREFIXES)
ax.legend(loc='center right')
ax=axes[1,1]
ax.vlines(summary.samples,summary.cdf_max_ci_low,summary.cdf_max_ci_high,color='#1565c0',lw=1.2)
ax.plot(summary.samples,summary.cdf_max_to_gue_approximation,'o-',color='#1565c0',mfc='white')
ax.axhline(0,color='gray',ls=':',lw=.8)
ax.set(xlabel=r'Realizations $S$',ylabel=r'$D_{\max}$ to GUE approximation',ylim=(0,.85),xticks=PREFIXES)
for ax,letter in zip(axes.flat,'abcd'):
    ax.tick_params(top=True,right=True)
    ax.text(-.16,1.04,f'({letter})',transform=ax.transAxes,fontweight='bold')
fig.suptitle(r'Fixed $20\times32$ ensemble; $W^2=9$; disorder on $x=5,6,14,15$; $L=0.99$',fontsize=11)
fig.tight_layout(pad=1)
fig.savefig(OUT/'sample_count_convergence.pdf');plt.close(fig)
subprocess.run(['pdftoppm','-r','300','-singlefile','-png',str(OUT/'sample_count_convergence.pdf'),str(OUT/'sample_count_convergence')],check=True)
display(Image(filename=str(OUT/'sample_count_convergence.png'),width=1100))''')
md('## Cumulative distributions\nCompare the original 100, the expanded 1000, the new 900 alone, the fixed nonequilibrium reference, and the GUE approximation. Curves use raw ratios.')
code(r'''fig,ax=plt.subplots(figsize=(7.05,3.7))
cdfrows=[]
for a,label,color,ls in [
    (r[ids<100],'Equilibrium: first 100','#228833','--'),
    (r,'Equilibrium: all 1000','#1565c0','-'),
    (r[ids>=100],'Equilibrium: new 900','#aa4499',':'),
    (nr,'Nonequilibrium: 100','#777777','-.')]:
    a=np.sort(a);xx=np.r_[0,a,1];yy=np.r_[0,np.arange(1,len(a)+1)/len(a),1]
    ax.step(xx,yy,where='post',color=color,ls=ls,lw=1.2,label=label)
    cdfrows.extend({'ensemble':label,'R':x,'cdf':y} for x,y in zip(xx,yy))
ax.plot(grid,gue_cdf_grid,color='black',ls='--',lw=1.3,label='GUE approximation')
pd.DataFrame(cdfrows).to_csv(OUT/'empirical_cdfs.csv',index=False)
pd.DataFrame({'R':grid[::16],'pdf':gue_pdf(grid[::16]),'cdf':gue_cdf_grid[::16]}).to_csv(OUT/'gue_reference.csv',index=False)
ax.set(xlim=(0,1),ylim=(0,1),xlabel=r'Adjacent-gap ratio $R$',ylabel=r'Cumulative probability $F(R)$',
       title=r'Fixed $20\times32$; $W^2=9$; walls + one inward column')
ax.tick_params(top=True,right=True);ax.legend(loc='upper left')
fig.tight_layout(pad=1)
fig.savefig(OUT/'cumulative_distributions.pdf');plt.close(fig)
subprocess.run(['pdftoppm','-r','300','-singlefile','-png',str(OUT/'cumulative_distributions.pdf'),str(OUT/'cumulative_distributions')],check=True)
display(Image(filename=str(OUT/'cumulative_distributions.png'),width=1000))''')
md('## Diagnostics\nCheck input identity, spectral bounds, ground-state eigensystems, prefix reproduction, GUE normalization, and histogram conservation. The new-900 estimate is independent of the original 100. No additional nonequilibrium trajectories were generated.')
code(r'''ds=[d['diagnostics'] for d in receipts]
diagnostics={'configuration':config,'analysis':{'L':L,'prefixes':PREFIXES,'ratio_bins':R_BINS,
 'bootstrap_replicates':BOOTSTRAPS,'bootstrap_seed':BOOTSTRAP_SEED,
 'bootstrap_unit':'whole realization','distance_intervals':'95% percentile',
 'mean_errors':'one-SE delete-realization jackknife','prefixes_are_nested':True},
 'sample_count_convergence':rows,'new_900':holdout,'nonequilibrium_reference':ne_row,
 'independent_blocks':blocks,'first100_max_spectral_error':prefix_error,
 'gue_reference':{'approximation_mean':float(gue_mean),'large_matrix_mean':.5996,
 'source':'https://arxiv.org/abs/1212.5611','cdf_interpolation_check_tolerance':1e-9},
 'max_eigen_residual':max(d['eigen_residual'] for d in ds),
 'max_orthogonality':max(d['orthogonality'] for d in ds),
 'full_matrix_checks':[d for d in ds if 'full_space_spectrum_crosscheck' in d],
 'raw_lambda_min':float(spectra.min()),'raw_lambda_max':float(spectra.max()),
 'checks':{'all_input_checksums':True,'1000_unique_sample_ids':True,
 'first100_potentials_identical':True,'histogram_counts_and_integrals':True,'finite_ratios':True},
 'scope':'sample-count convergence at fixed physical parameters; full retained single-particle entanglement spectrum, no sector separation; no new nonequilibrium trajectories'}
(OUT/'diagnostics.json').write_text(json.dumps(diagnostics,indent=2)+'\n')
display(summary)
display(pd.DataFrame([holdout,ne_row]))
print('All checks passed; 1000 equilibrium states, including 900 new independent realizations.')''')
path=OUT/'sample_count_convergence.ipynb'
nbformat.write(nb,path)
NotebookClient(nb,timeout=180,kernel_name='python3',resources={'metadata':{'path':str(OUT)}}).execute()
nbformat.write(nb,path)
(OUT/'README.md').write_text('''# Equilibrium sample-count convergence: W^2=9, walls + one inward column

Nx20 Ny32, hard walls x5,15, disorder mask x5,6,14,15, independent Gaussian
potential for every selected orbital. Fixed 640 particles, Ay16, y0=0, L=.99.
1000 realizations; first 100 reproduce the preceding six-strength scan.
Seeds [2026092703,5,sample_id] preserve that prefix exactly.

Ratios are computed within each realization before pooling. Compare nested
prefixes 100,250,500,1000, a disjoint new-900 set, and ten independent 100-state
blocks. Mean uncertainties use a whole-realization jackknife. CDF distance
intervals resample whole realizations (500 replicates, 95% percentile).
No independent-level p-values or additional nonequilibrium simulations.

GUE density/CDF reference: folded Atas et al. 3x3 approximation (mean .60266).
The mean plot separately labels the large-matrix GUE value .5996.
Neither increasing sample count nor this finite geometry establishes a
thermodynamic/sector-resolved universality classification.

Reproduce acquisition with run_analysis.py; execute analysis via build_notebook.py.
Saved outputs include potentials, spectra, energies, sample-resolved ratios,
histograms, prefix/block statistics, bootstrap distances, PDF/PNG figures,
executed notebook, full input provenance, receipts and checksums.
''')
files=[p for p in OUT.rglob('*') if p.is_file() and p.name!='completion_manifest.json']
manifest={'status':'complete','realizations':1000,'new_realizations':900,
 'files':{str(p.relative_to(OUT)):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in files}}
(OUT/'completion_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(OUT)
