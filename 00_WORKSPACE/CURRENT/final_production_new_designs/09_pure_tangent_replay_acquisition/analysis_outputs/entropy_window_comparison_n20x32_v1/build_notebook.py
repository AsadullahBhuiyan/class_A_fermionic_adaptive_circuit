from pathlib import Path
import nbformat,shutil
OUT=Path(__file__).resolve().parent
source=OUT.parent/'mean_mode_count_window_scan_n20x32_v1'
(OUT/'latex_support').mkdir(exist_ok=True);shutil.copy2(source/'latex_support/type1ec.sty',OUT/'latex_support/type1ec.sty')
nb=nbformat.v4.new_notebook();cells=[]
def md(s):cells.append(nbformat.v4.new_markdown_cell(s))
def code(s):cells.append(nbformat.v4.new_code_cell(s))
md('# Full versus window-restricted entropy\nMatched analysis of all 100 saved 20 by 32 hard-wall pure endpoints at cycle 64; average all 32 origins within each trajectory, then trajectories. Compare full entropy with the contribution from eigenvalues inside a centered spectral window.')
code('''import os
if 'AVAILABLE_CPUS' not in globals():AVAILABLE_CPUS=sorted(os.sched_getaffinity(0))
CPU_RANGE=(8,9)
selected=list(range(CPU_RANGE[0],CPU_RANGE[1]+1));assert set(selected).issubset(AVAILABLE_CPUS)
os.sched_setaffinity(0,selected)
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[key]=str(len(selected))
from threadpoolctl import threadpool_limits
limits=threadpool_limits(len(selected))
print('Allocated CPUs:',selected)''')
md(r'''## Estimator and interpretation
For occupation eigenvalue $\nu=(1+\lambda)/2$, the entropy contribution is $h(\nu)=-\nu\ln\nu-(1-\nu)\ln(1-\nu)$. Compare
$$S_L(A_y)=\sum_{|\lambda_j|\le L}h((1+\lambda_j)/2)$$
with the full von Neumann entropy $S_1$. Origins are averaged inside each trajectory before the ensemble average. There is no renormalization after dropping modes. The original raw spectra are unchanged; numerical excursions within $10^{-8}$ are clipped only when evaluating entropy.

Fits use $a+b\ln[(N_y/\pi)\sin(\pi A_y/N_y)]$, $N_y=32$ and widths 5–16. These are the same unanchored two-parameter, fixed-size fits used for the preceding mode-count comparison, **not** the technical report's joint multi-size anchored fit. Coefficient uncertainties propagate correlations across widths by fitting each trajectory-level origin average and computing the SEM. R-squared describes the mean curve and does not establish universal scaling.

Removing modes changes the observable; a cutoff-dependent coefficient is not automatically the central-charge coefficient of the full entropy. In particular, $h(\nu)$ already vanishes at the pure endpoints, while the counting weight is one per included mode. Sources: [Peschel–Eisler](https://arxiv.org/abs/0906.1663) for free-fermion entropy; [Calabrese–Cardy](https://arxiv.org/abs/hep-th/0405152) for interval entropy scaling.''')
md('## Analysis and metadata\nThe helper checks the spectrum-cache hash and records its production provenance. It cross-checks the full half-system entropy against the earlier independent entropy analysis.')
code('''from pathlib import Path
import sys,json,subprocess
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from IPython.display import display,Image
OUT=Path.cwd();sys.path.insert(0,str(OUT))
from analyze_entropy_windows import analyze
fits,diagnostics=analyze()
print(json.dumps({k:v for k,v in diagnostics.items() if k not in ('input_provenance','fits')},indent=2))
display(fits)
data=dict(np.load(OUT/'entropy_window_statistics.npz'))''')
md('## Residuals of the matched fits\nThe full and truncated entropies each get their own intercept and slope. Residuals are shown over the fitted widths. Connected points guide the eye; widths and windows are correlated.')
code(r'''os.environ['TEXINPUTS']=str(OUT/'latex_support')+os.pathsep+os.environ.get('TEXINPUTS','')
mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['CMU Sans Serif'],'font.size':8,'text.usetex':True,
                     'xtick.direction':'in','ytick.direction':'in'})
fig,ax=plt.subplots(figsize=(3.375,2.7))
for L,color,marker,style in [(.9,'#c62828','^',':'),(.99,'#2e7d32','s','--'),(1.,'#1565c0','o','-')]:
    k=int(np.flatnonzero(np.isclose(data['windows'],L,rtol=0,atol=1e-14))[0])
    ax.plot(data['widths'][data['widths']>=5],data['residuals'][k],color=color,marker=marker,ls=style,ms=3,lw=.8,
            label='Full entropy' if L==1 else r'$L=%.2f$'%L)
ax.axhline(0,color='gray',ls='--',lw=.6)
ax.set(xlabel=r'Subsystem width $A_y$',ylabel='Entropy fit residual (nats)',title=r'$20\times32$; fit $5\leq A_y\leq16$')
ax.legend(frameon=False,fontsize=8);ax.tick_params(top=True,right=True);fig.tight_layout(pad=.8)
fig.savefig(OUT/'entropy_fit_residuals.pdf');plt.close(fig)
subprocess.run(['pdftoppm','-r','300','-singlefile','-png',str(OUT/'entropy_fit_residuals.pdf'),str(OUT/'entropy_fit_residuals')],check=True)
display(Image(filename=str(OUT/'entropy_fit_residuals.png'),width=550))''')
md('Caption: 100 independent trajectories, pure initialization with exterior product preparation, Nx=20, Ny=32, hard construction, alpha1=1, alpha2=30, nshell=1, cycle 64. Sum entropy contributions inside each spectral window before averaging 32 origins within a trajectory, then trajectories. Fits use widths 5–16. Coefficient errors in the table are trajectory SEMs retaining cross-width covariance; the residual plot shows deviations of the ensemble means, not independent samples.')
md('## Diagnostics and result')
code('''full=fits.iloc[-1];central=fits[np.isclose(fits.L,.99)].iloc[0]
summary=dict(full_entropy_slope=full.slope,window_099_slope=central.slope,
 full_entropy_R_squared=full.R_squared,window_099_R_squared=central.R_squared,
 relative_slope_change=central.slope/full.slope-1,
 half_entropy_removed_fraction=1-central.half_entropy_retained_fraction,
 full_entropy_prior_difference=diagnostics['full_entropy_previous_result_difference'])
print(json.dumps(summary,indent=2))
(OUT/'comparison_summary.json').write_text(json.dumps(summary,indent=2)+'\\n')
(OUT/'README.md').write_text('# Full versus window-restricted entropy\\n\\nMatched fixed Ny=32, widths 5–16, 100 trajectories, 32 origins. No renormalization after filtering modes.\\n\\n'+fits.to_string(index=False)+'\\n\\nA small improvement in the L=0.99 residual is accompanied by a change in slope. This is not a reproduction or revision of the technical report joint multi-size anchored central-charge fit.\\n')
print('Completed: entropy table, sample statistics, input provenance, diagnostics, PDF/PNG residual figure.')''')
nb.cells=cells;nb.metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}}
nbformat.write(nb,OUT/'entropy_window_comparison.ipynb')
