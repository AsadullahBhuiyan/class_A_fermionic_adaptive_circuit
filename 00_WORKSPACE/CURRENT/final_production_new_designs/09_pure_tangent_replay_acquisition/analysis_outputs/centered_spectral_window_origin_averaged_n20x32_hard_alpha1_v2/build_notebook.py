from pathlib import Path
import shutil,nbformat
OUT=Path(__file__).resolve().parent
shim=OUT.parent/'stochastic_equilibrium_spectral_comparison_v1/latex_support/type1ec.sty'
(OUT/'latex_support').mkdir(exist_ok=True);shutil.copy2(shim,OUT/'latex_support/type1ec.sty')
nb=nbformat.v4.new_notebook();cells=[]
def md(s):cells.append(nbformat.v4.new_markdown_cell(s))
def code(s):cells.append(nbformat.v4.new_code_cell(s))
md(r'''# Origin-averaged central spectral weight versus subsystem width
Fixed $N_x=20$, $N_y=32$, hard walls, $\alpha_1=1$: all 100 saved pure-state endpoints at cycle 64. Vary the contiguous subsystem width $A_y=1,\ldots,16$, retaining every x and both orbitals averaging all 32 periodic origins $y_0=0,\ldots,31$. Compute the integrated centered spectral density in $[-L,L]$ directly from eigenvalue counts, without histogram binning or new circuit simulation.''')
code('''import os
if 'AVAILABLE_CPUS' not in globals():
    AVAILABLE_CPUS=sorted(os.sched_getaffinity(0))
CPU_RANGE=(8,15) # editable inclusive range
selected=list(range(CPU_RANGE[0],CPU_RANGE[1]+1))
assert set(selected).issubset(AVAILABLE_CPUS)
os.sched_setaffinity(0,selected)
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[key]=str(len(selected))
from threadpoolctl import threadpool_limits
limits=threadpool_limits(len(selected))
print('Allocated CPUs:',selected)''')
md(r'''## Estimators
The occupation correlation matrix is $G_A=F_AF_A^\dagger$ and its centered version is $Q_A=2G_A-\mathbf{1}_A$, with eigenvalues $\lambda=2\nu-1$. (Older caches call $Q_A$ the centered covariance `G`.) Rows use $i=2N_xy+2x+\mu$ and only the active columns of each saved frame are retained. Diagonalize each trajectory separately before any average.

At each origin $y_0$, define $n_s(L,A_y;y_0)=\sum_j\mathbf{1}_{|\lambda_{s,j}|\le L}$ and $m_s(A_y;y_0)=\sum_j\mathbf{1}_{|\lambda_{s,j}|<1-\tau}$, where $\tau=10^{-8}$. The requested equal-weight normalization and averaging gives
$$I_L^{\rm mixed}(A_y)=\frac{1}{100}\sum_{s=1}^{100}\left[\frac{1}{32}\sum_{y_0=0}^{31}\frac{n_s(L,A_y;y_0)}{m_s(A_y;y_0)}\right].$$
Normalize every trajectory-origin spectrum separately, average the 32 origins within each trajectory, then average the 100 trajectories equally. Earlier v1 products pooled samples at each origin before averaging origins; they are preserved separately. Alternative pooling orders are exported for comparison. Also retain the raw origin-and-trajectory mean count $\overline n_L=\sum_{s,y_0}n_s/(100\times32)$ and the integral normalized over all modes, $I_L^{\rm all}=\overline n_L/(40A_y)$. The normalized integrals and the raw count need not share a scaling law.

Test the proposed form $a+b\log d(A_y)$, with $d=(32/\pi)\sin(\pi A_y/32)$ and lattice spacing one. Fits are descriptive ordinary least squares. Uncertainty propagates the full covariance across widths from the 100 independent trajectories; normalized-integral uncertainties are ordinary SEMs of the 100 trajectory-level origin averages. Eigenvalues and widths are not independent samples. The default fit range is $8\le A_y\le16$, with sensitivity fits starting at 2 and 4. All cuts wrap periodically and every origin is evaluated explicitly. There is no inferred reflection to widths greater than half the system. At half width, complementary origins are redundant, but are included with the same weight as other origins. Origin averaging is performed before estimating trajectory uncertainty; there are 100 independent sampling units, not 3200.''')
md('## Configuration and input validation\nChange L in the statistics cell without recomputing spectra. Input completion receipts, sizes, hashes, identities and sample IDs are verified even on cache reuse.')
code('''from pathlib import Path
import sys,json,subprocess
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from IPython.display import display,Image,Markdown
OUT=Path.cwd()
if not (OUT/'window_analysis.py').exists():
    root=next(p for p in (OUT,*OUT.parents) if (p/'PROJECT_ADMIN/REPO_POLICY.md').exists())
    OUT=root/'00_WORKSPACE/CURRENT/final_production_new_designs/09_pure_tangent_replay_acquisition/analysis_outputs/centered_spectral_window_origin_averaged_n20x32_hard_alpha1_v2'
sys.path.insert(0,str(OUT))
import window_analysis as analysis
print(json.dumps(dict(Nx=20,Ny=32,alpha_1=1,alpha_2=30,nshell=1,construction='hard',samples=100,cycle=64,
                     origins_y=list(range(32)),widths=list(range(1,17)),initialization='pure; exterior product frame',sequence='raster_y',
                     canonical_dynamics_entry_point='classA_U1FGTN_gpu.run_markov_circuit (saved inputs only)',
                     input=str(analysis.DATA),output=str(OUT)),indent=2))
spectra,spectral_diagnostics=analysis.compute_spectra()
print('All-origin spectra validated:',sum(v.size for v in spectra.values()),'eigenvalues across all widths.')''')
md('## Window integral and log-chord fits\nThis cell uses the cached eigenvalues. No histogram integration or clipping is needed for this interior window.')
code('''L=0.5 # editable without re-diagonalization
MIXED_TOL=1e-8
FIT_MIN_WIDTH=8
summary,results=analysis.analyze(spectra,L=L,mixed_tol=MIXED_TOL,fit_min=FIT_MIN_WIDTH)
display(summary[['Ay','mean_mixed_count','mode_count','mode_count_sem','mixed_normalized_integral','mixed_normalized_integral_sem','all_modes_integral']])
print(json.dumps({k:v[str(FIT_MIN_WIDTH)] for k,v in results['fits'].items()},indent=2))''')
code('''os.environ['TEXINPUTS']=str(OUT/'latex_support')+os.pathsep+os.environ.get('TEXINPUTS','')
mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['CMU Sans Serif'],'font.size':8,
 'axes.labelsize':8,'axes.titlesize':8,'xtick.labelsize':8,'ytick.labelsize':8,'text.usetex':True,
 'pdf.fonttype':42,'xtick.direction':'in','ytick.direction':'in'})
def export(fig,name):
    for ax in fig.axes:ax.tick_params(top=True,right=True)
    fig.tight_layout(pad=.8)
    fig.savefig(OUT/(name+'.pdf'));plt.close(fig)
    subprocess.run(['pdftoppm','-r','300','-singlefile','-png',str(OUT/(name+'.pdf')),str(OUT/name)],check=True)
    display(Image(filename=str(OUT/(name+'.png')),width=950))''')
md('## Integrated density and number of modes\nStraightness against log chord length tests the proposed form. Dashed lines are descriptive fits over the specified width range; points include every width. Error bars are one SEM of the 100 trajectory-level origin averages in both panels.')
code(r'''fig,axes=plt.subplots(1,2,figsize=(7.05,2.9))
for ax,key,title,ylabel,letter in zip(axes,
 ['mixed_normalized_integral','mode_count'],
 ['Integral of the normalized mixed-mode density','Mean number of modes in the window'],
 [r'$I_L^{\rm mixed}=\int_{-L}^{L}\rho_{\rm mixed}(\lambda)\,d\lambda$',r'$\overline{n_L}$'],['(a)','(b)']):
    ax.errorbar(summary.log_chord,summary[key],yerr=summary[key+'_sem'],color='#1565c0',marker='o',ls='-',lw=.7,ms=3,capsize=2,label='32 origins; 100 trajectories')
    fit=results['fits'][key][str(FIT_MIN_WIDTH)]
    grid=np.linspace(summary.loc[summary.Ay>=FIT_MIN_WIDTH,'log_chord'].min(),summary.log_chord.max(),200)
    ax.plot(grid,fit['intercept']+fit['slope']*grid,color='black',ls='--',lw=1,label=r'$a+b\log d$ fit')
    ax.set(xlabel=r'$\log d(A_y)$, $d=(32/\pi)\sin(\pi A_y/32)$',ylabel=ylabel,title=title)
    ax.text(-.15,1.08,letter,transform=ax.transAxes,fontweight='bold')
    ax.legend(frameon=False,fontsize=8,loc='best')
fig.suptitle(r'$20\times32$, $L=%.2f$, origin average; fit $%d\leq A_y\leq16$'%(L,FIT_MIN_WIDTH),fontsize=9)
export(fig,'spectral_window_vs_log_chord')''')
md('## Subsystem-width dependence\nThe left panel normalizes each mixed-mode spectrum before averaging origins within trajectories and then trajectories. The right panel retains all modes in the density normalization, including those near the pure-state endpoints. Both use the same numerator.')
code(r'''fig,axes=plt.subplots(1,2,figsize=(7.05,2.7))
for ax,key,ylabel,letter in zip(axes,['mixed_normalized_integral','all_modes_integral'],
 [r'$I_L^{\rm mixed}$',r'$I_L^{\rm all}=\overline{n_L}/(40A_y)$'],['(a)','(b)']):
    ax.errorbar(summary.Ay,summary[key],yerr=summary[key+'_sem'],color='#1565c0',marker='o',ls='-',ms=3,capsize=2,lw=.8)
    ax.set(xlabel=r'Subsystem width $A_y$',ylabel=ylabel,xticks=[1,4,8,12,16])
    ax.text(-.15,1.04,letter,transform=ax.transAxes,fontweight='bold')
export(fig,'spectral_window_vs_width')''')
md('''Caption: 100 independent trajectories, initialized as pure states with a prepared exterior product frame; hard-wall production protocol at cycle 64. The full system is 20 by 32. Each subsystem includes every x and both orbitals, with width 1 through 16 and all 32 periodic origins. Spectra are computed per trajectory and origin. Each spectrum is normalized separately; the 32 origins are averaged within each trajectory, then the 100 trajectories are averaged equally. Error bars are ordinary SEMs of the 100 trajectory-level origin averages. Dashed fits use widths 8–16 by default and propagate correlations between widths. A narrow sharp spectral window can produce level-crossing features, so a straight-line fit alone does not establish an asymptotic scaling law.''')
md('## Numerical diagnostics\nRaw spectra are preserved. Hermiticity is checked before numerical symmetrization. The origin-zero half-system spectrum is compared with the previously saved cache. Fixed-origin results are preserved in the separate earlier output directory. Statistical and fit outputs are saved separately so the window can be edited.')
code('''print(json.dumps({k:v for k,v in spectral_diagnostics.items() if k not in ('inputs','identity')},indent=2))
print(json.dumps(results,indent=2))
if (OUT/'cross_checks.json').exists():print((OUT/'cross_checks.json').read_text())
print('Completed: spectra NPZ, per-sample counts CSV, integrals CSV, covariance NPZ, diagnostics JSON, and two PDF/PNG figures.')''')
nb.cells=cells;nb.metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3'}}
nbformat.write(nb,OUT/'spectral_window_vs_subsystem.ipynb')
