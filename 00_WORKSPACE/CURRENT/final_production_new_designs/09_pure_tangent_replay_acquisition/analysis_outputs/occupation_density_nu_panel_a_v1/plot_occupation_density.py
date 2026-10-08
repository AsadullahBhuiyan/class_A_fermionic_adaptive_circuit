"""Standalone occupation-coordinate version of manuscript Figure 7(a)."""
import os
CPU_RANGE=(40,41)
os.sched_setaffinity(0,range(CPU_RANGE[0],CPU_RANGE[1]+1))
from pathlib import Path
import sys,json,hashlib,subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path(__file__).resolve().parent
REPO=next(p for p in OUT.parents if (p/'PROJECT_ADMIN/REPO_POLICY.md').exists())
BUNDLE=REPO/'00_WORKSPACE/CURRENT/manuscript_Overleaf/figures/new_figure'
sys.path.insert(0,str(BUNDLE/'sources'))
import manuscript_typography as typography
from manuscript_palette import ALPHA_COLORS
DATA=BUNDLE/'data/entanglement_spectrum'
STEM='occupation_density_nu'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
p=DATA/'pooled_half_strip_spectra.npz'
provenance=json.loads((DATA/'pooled_half_strip_provenance.json').read_text())
assert sha(p)==provenance['compact_input_sha256']
with np.load(p,allow_pickle=False) as z:
    lam=z['centered_occupations'].copy()
    np.testing.assert_array_equal(z['alpha_1'],[1,3])
    np.testing.assert_array_equal(z['sample_ids'],np.arange(100))
    np.testing.assert_array_equal(z['origins'],np.arange(32))
    for key,value in dict(Nx=20,Ny=32,Ay=16,cycle=64).items():assert z[key].item()==value
assert lam.shape==(2,100,32,640) and np.isfinite(lam).all()
assert lam.min()>=-1-1e-8 and lam.max()<=1+1e-8
# Preserve original bin membership exactly; transform bin widths with the Jacobian.
le=np.linspace(-1,1,101)
counts=np.array([np.histogram(np.clip(v.ravel(),-1,1),le)[0] for v in lam])
np.testing.assert_array_equal(counts.sum(1),[2048000,2048000])
nu_edges=(le+1)/2
rho=counts/(counts.sum(1)[:,None]*np.diff(nu_edges))
lambda_rho=counts/(counts.sum(1)[:,None]*np.diff(le))
np.testing.assert_allclose(rho,2*lambda_rho,rtol=1e-13,atol=1e-14)
np.testing.assert_allclose((rho*np.diff(nu_edges)).sum(1),1,rtol=0,atol=1e-14)
original=np.loadtxt(DATA/'occupation_histogram.csv',delimiter=',',skiprows=1)
np.testing.assert_array_equal(counts.T,original[:,2:4])
# Standalone single-column artifact: use the shared style at its actual output width.
typography.ROOT=OUT
(OUT/'data').mkdir(exist_ok=True)
typography.inclusion_width=lambda stem:3.375 if stem==STEM else (_ for _ in ()).throw(ValueError(stem))
typography.configure_style({'axes.linewidth':.8,'xtick.direction':'in','ytick.direction':'in'})
fig,ax=plt.subplots(figsize=(3.375,2.15))
for density,alpha,style in zip(rho,[1,3],['-',':']):
    ax.stairs(density,nu_edges,baseline=None,color=ALPHA_COLORS[alpha],linestyle=style,linewidth=1,label=rf'$\alpha_1={alpha}$')
positive=rho[rho>0]
ax.set(xlabel=r'Occupation $\nu$',ylabel=r'Probability density $\rho(\nu)$',xlim=(-.01,1.01),yscale='log',ylim=(positive.min()/2,positive.max()*1.5))
ax.set_xticks([0,.25,.5,.75,1])
ax.text(.5,.96,r'$N_y=32$, $A_y=16$',transform=ax.transAxes,ha='center',va='top')
ax.legend(loc='upper center',bbox_to_anchor=(.5,.82),ncol=2,frameon=False,handlelength=1.6,columnspacing=1)
ax.tick_params(which='both',top=True,right=True)
typography.prepare_figure(fig,STEM)
fig.tight_layout(pad=.75)
typography.record_typography(fig,STEM)
fig.savefig(OUT/(STEM+'.pdf'))
plt.close(fig)
subprocess.run(['pdftoppm','-r','300','-singlefile','-png',str(OUT/(STEM+'.pdf')),str(OUT/STEM)],check=True)
font_validation=typography.verify_typography(STEM)
np.savetxt(OUT/'occupation_histogram_nu.csv',np.column_stack([nu_edges[:-1],nu_edges[1:],counts.T,rho.T]),delimiter=',',header='nu_left,nu_right,alpha1_1_count,alpha1_3_count,alpha1_1_density,alpha1_3_density',comments='')
validation=dict(source=str(p),source_sha256=sha(p),source_renderer=str(BUNDLE/'sources/plot_entanglement_spectrum.py'),Nx=20,Ny=32,Ay=16,cycles=64,samples_per_alpha=100,origins_per_trajectory=32,alpha1=[1,3],initialization='random pure half-filled',independent_sampling_unit='trajectory',normalization='unit area separately per alpha; all modes including pure endpoints; pool equal-sized cuts and trajectories',transformation='nu=(lambda+1)/2; rho_nu=2*rho_lambda',bins=100,counts=counts.sum(1).tolist(),integrals=(rho*np.diff(nu_edges)).sum(1).tolist(),original_histogram_counts_match=True,raw_extrema=[float(lam.min()),float(lam.max())],empty_bins='zero; no pseudocounts on logarithmic axis',typography=font_validation)
(OUT/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
files={str(f.relative_to(OUT)):dict(bytes=f.stat().st_size,sha256=sha(f)) for f in OUT.rglob('*') if f.is_file() and f.name!='completion_manifest.json'}
(OUT/'completion_manifest.json').write_text(json.dumps(dict(status='complete',files=files),indent=2)+'\n')
print(json.dumps(validation,indent=2))
