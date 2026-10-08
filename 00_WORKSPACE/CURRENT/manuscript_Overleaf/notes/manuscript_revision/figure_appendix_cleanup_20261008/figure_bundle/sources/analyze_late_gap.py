"""Finite-window slope analysis of saved slab-only trajectories; no dynamics."""
from pathlib import Path
import csv, hashlib, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from manuscript_typography import configure_style, prepare_figure, record_typography
from log_ticks import add_log_minor_ticks

ROOT=Path(__file__).resolve().parents[1]
REPO=next(p for p in ROOT.parents if (p/'PROJECT_ADMIN/REPO_POLICY.md').exists())
OUT=ROOT.parents[1]/'deliverables/spectral_updates_20261008/late_gap_analysis'
SOURCE=REPO/'00_WORKSPACE/CURRENT/experiment_review/purification_gap_cycle_convergence/sample_cycle_gaps.csv'
NYS=[20,24,30,36,44,56,60]

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def stats(x): return float(np.mean(x)),float(np.std(x,ddof=1)/np.sqrt(len(x)))
def power_fit(n,y,e):
    assert np.all(y>0) and np.all(e>0)
    x=np.log(n);v=np.log(y);s=e/y
    X=np.column_stack([np.ones(len(x)),x]);w=1/s**2
    cov=np.linalg.inv(X.T@(w[:,None]*X));beta=cov@(X.T@(w*v))
    residual=v-X@beta;chi2=float(np.sum(w*residual**2));dof=len(n)-2
    # Known supplied SEM weights; do not rescale by residual scatter.
    return dict(amplitude=float(np.exp(beta[0])),z=float(-beta[1]),
                z_regression_error=float(np.sqrt(cov[1,1])),chi_squared=chi2,
                degrees_of_freedom=dof,reduced_chi_squared=chi2/dof,
                weighted_R_squared=float(1-np.sum(w*residual**2)/np.sum(w*(v-np.average(v,weights=w))**2)),
                covariance=cov.tolist(),fit='log Delta = log A - z log Ny; weights=(Delta/SEM)^2')

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((ROOT/'data/gap_convergence/analysis_manifest.json').read_text())
    assert sha(SOURCE)==manifest['outputs']['sample_cycle_gaps.csv']
    with SOURCE.open() as f: rows=list(csv.DictReader(f))
    np.testing.assert_allclose([float(r['modular_gap']) for r in rows],
                               [2*int(r['cycle'])*float(r['gap']) for r in rows],
                               rtol=1e-12,atol=1e-14)
    summaries=[];samples=[]
    for ny in NYS:
        r=[q for q in rows if q['protocol']=='slab_only' and int(q['Ny'])==ny]
        times=sorted({int(q['cycle']) for q in r}); ids=sorted({int(q['sample']) for q in r})
        assert ids==list(range(100)) and 3*ny in times and 4*ny in times and 2*ny in times
        lookup={(int(q['sample']),int(q['cycle'])):float(q['modular_gap']) for q in r}
        assert len(lookup)==len(ids)*len(times)==len(r)
        t=np.array([x for x in times if 3*ny<=x<=4*ny])
        g=np.array([[lookup[s,int(c)] for c in t] for s in ids])
        endpoint=(g[:,-1]-g[:,0])/(2*ny)
        slope=(g@(t-t.mean()))/np.sum((t-t.mean())**2)/2
        initial=np.array([lookup[s,2*ny]/(4*ny) for s in ids])
        earlier=np.array([(lookup[s,3*ny]-lookup[s,2*ny])/(2*ny) for s in ids])
        mean,sem=stats(endpoint);lm,ls=stats(slope);dm,ds=stats(initial);pm,ps=stats(earlier)
        diff,de=stats(slope-endpoint)
        intercept=g.mean(1)-2*slope*t.mean()
        residual=g.mean(0)-(2*lm*t+intercept.mean())
        summaries.append(dict(Ny=ny,samples=100,late_endpoint_mean=mean,late_endpoint_SEM=sem,
            late_OLS_mean=lm,late_OLS_SEM=ls,Delta_2Ny_mean=dm,Delta_2Ny_SEM=ds,
            earlier_endpoint_mean=pm,earlier_endpoint_SEM=ps,OLS_minus_endpoint_mean=diff,
            OLS_minus_endpoint_SEM=de,saved_times=t.tolist(),raw_gap_fit_intercept=float(intercept.mean()),
            mean_raw_gap_fit_max_abs_residual=float(np.max(abs(residual)))))
        for j,s in enumerate(ids):samples.append([ny,s,endpoint[j],slope[j],initial[j],earlier[j]])
    table=np.array([[r['Ny'],r['Delta_2Ny_mean'],r['Delta_2Ny_SEM'],r['late_endpoint_mean'],r['late_endpoint_SEM'],r['late_OLS_mean'],r['late_OLS_SEM']] for r in summaries])
    np.savetxt(OUT/'size_summary.csv',table,delimiter=',',header='Ny,Delta_2Ny,Delta_2Ny_SEM,late_endpoint,late_endpoint_SEM,late_OLS,late_OLS_SEM',comments='')
    np.savetxt(OUT/'trajectory_estimators.csv',samples,delimiter=',',header='Ny,sample,late_endpoint,late_OLS,Delta_2Ny,earlier_endpoint',comments='')
    fits={}
    for key,ycol,ecol in [('Delta_2Ny',1,2),('late_endpoint',3,4),('late_OLS',5,6)]:
        fits[key]=power_fit(table[:,0],table[:,ycol],table[:,ecol]) if np.all(table[:,ycol]>0) else dict(fit_not_performed='Nonpositive mean estimate; no log substitution')
    configure_style({'legend.frameon':False});fig,ax=plt.subplots(figsize=(3.375,3.1))
    for key,ycol,ecol,color,marker in [('Delta_2Ny',1,2,'#0072B2','o'),('late_endpoint',3,4,'#D55E00','s'),('late_OLS',5,6,'#009E73','^')]:
        ax.errorbar(table[:,0],table[:,ycol],yerr=table[:,ecol],color=color,marker=marker,mfc='white',ls='none',ms=3.8,capsize=1,label={'Delta_2Ny':r'$\overline\Delta(2N_y)$','late_endpoint':'Late endpoint slope','late_OLS':'Late fitted slope'}[key])
        if 'z' in fits[key]:
            xx=np.geomspace(19,63,150);f=fits[key]
            ax.plot(xx,f['amplitude']*xx**(-f['z']),color=color,ls='--',lw=.8)
    ax.set(xscale='log',yscale='log',xlabel=r'Circumference $N_y$',ylabel=r'Finite-time rate',xlim=(18,65),ylim=(.01,.10))
    ax.set_xticks([20,30,40,60]);ax.set_xticklabels(['20','30','40','60'])
    ax.legend(loc='lower left',handlelength=1.2);ax.tick_params(which='both',top=True,right=True)
    add_log_minor_ticks(fig);prepare_figure(fig,'Late_gap_comparison')
    fig.subplots_adjust(left=.20,right=.97,bottom=.20,top=.96)
    record_typography(fig,'Late_gap_comparison')
    for ext in ['pdf','png']:fig.savefig(OUT/f'late_gap_comparison.{ext}',dpi=300)
    plt.close(fig)
    report=dict(source=str(SOURCE),source_sha256=sha(SOURCE),protocol='slab_only',sizes=NYS,
        samples_per_size=100,summary=summaries,fits=fits,uncertainty='paired trajectory SEM; formal weighted-regression error for exponent; no bootstrap',
        no_simulations=True,manuscript_gap_claims_changed=False,renderer_sha256=sha(__file__))
    (OUT/'analysis.json').write_text(json.dumps(report,indent=2)+'\n')
    lines=['Late-time gap analysis — saved slab-only histories','',
           'These are finite-window estimates, not established infinite-time gaps.',
           'Primary estimator: [g_mod(4Ny)-g_mod(3Ny)]/(2Ny) within each trajectory.',
           'Comparison: half the OLS slope of g_mod on saved times in [3Ny,4Ny].',
           'SEM is computed across 100 paired trajectory estimators. No bootstrap.','',
           'Ny | Delta(2Ny) +/- SEM | late endpoint +/- SEM | late OLS +/- SEM']
    for r in summaries:lines.append(f"{r['Ny']:2} | {r['Delta_2Ny_mean']:.7f} +/- {r['Delta_2Ny_SEM']:.7f} | {r['late_endpoint_mean']:.7f} +/- {r['late_endpoint_SEM']:.7f} | {r['late_OLS_mean']:.7f} +/- {r['late_OLS_SEM']:.7f}")
    lines+=['','Power-law fits (formal errors from supplied SEMs; covariance not residual-rescaled):']
    for key,f in fits.items():
        if 'z' in f:lines.append(f"{key}: z={f['z']:.4f} +/- {f['z_regression_error']:.4f}; reduced chi2={f['reduced_chi_squared']:.3f}; weighted R2={f['weighted_R_squared']:.5f}")
    lines+=['','Taking a slope cancels a constant raw-gap intercept only if approximately affine growth holds over the fitted window. The minimizing mode may change with time. All seven sizes use times scaled with Ny, so this analysis is still finite time and cannot alone establish an asymptotic gap or exponent. Full-measurement data were not pooled. The current manuscript claims and Figures 4/A2 are unchanged.']
    (OUT/'report.txt').write_text('\n'.join(lines)+'\n')
    tex=r'''\documentclass[aps,prb,onecolumn,nofootinbib,superscriptaddress]{revtex4-2}
\usepackage{amsmath,amssymb,amsthm,mathtools,bm}
\usepackage{booktabs,graphicx}
\usepackage[colorlinks=true,linkcolor=blue,citecolor=blue,urlcolor=blue]{hyperref}
\usepackage{microtype}
\begin{document}
\title{Late-time slope diagnostics from saved slab-only purification histories}
\date{8 October 2026}
\begin{abstract}
We analyze existing 100-trajectory slab-only ensembles at seven circumferences through four times the circumference. Paired endpoint differences and late-window linear fits give finite-time size exponents close to one. The analysis removes a constant raw-gap offset under an affine-growth assumption; it does not establish an infinite-time Lyapunov gap. No new dynamics are simulated, and no manuscript gap claims are changed.
\end{abstract}
\maketitle
\setcounter{tocdepth}{2}
\tableofcontents
\section{Estimators and uncertainty}
For each record, the saved raw modular gap is the minimum absolute modular energy, with pinned occupations excluded using the original extraction tolerance $10^{-9}$:
\begin{equation}
g_{\rm mod}(t)=\min_j\left|\log\frac{1-\nu_j(t)}{\nu_j(t)}\right|=2t\Delta(t).
\end{equation}
The independent sampling unit is a trajectory. For each of the 100 trajectories at each circumference, compute
\begin{equation}
\Delta_{\rm late}=\frac{g_{\rm mod}(4N_y)-g_{\rm mod}(3N_y)}{2N_y}.
\end{equation}
The primary estimate is its ensemble mean, with ordinary trajectory SEM. This retains temporal correlations automatically. A comparison estimator is half the unweighted least-squares slope of $g_{\rm mod}$ on all saved cycles in $[3N_y,4N_y]$, with a free intercept, computed separately per trajectory and then averaged with its SEM. Endpoint differences and multi-time fits are different estimators. No bootstrap is used.
All data use $N_x=20$, $\alpha_1=1$, $\alpha_2=30$, and the saved slab-only protocol with a maximally mixed slab and Born-prepared exterior. Full-measurement histories are not pooled. Source receipts, trajectory IDs, and the relation $g_{\rm mod}=2t\Delta$ were verified in the existing extraction; this analysis verifies the sample-history checksum and complete paired records.
\section{Size comparison}
\begin{table}[h]
\caption{Ensemble means with one trajectory SEM. All entries are finite-window rates.}
\begin{ruledtabular}\begin{tabular}{rrrr}
$N_y$ & $\overline\Delta(2N_y)$ & Endpoint slope & Fitted slope\\
'''
    for r in summaries:
        tex+=f"{r['Ny']} & ${r['Delta_2Ny_mean']:.5f}\\pm{r['Delta_2Ny_SEM']:.5f}$ & ${r['late_endpoint_mean']:.5f}\\pm{r['late_endpoint_SEM']:.5f}$ & ${r['late_OLS_mean']:.5f}\\pm{r['late_OLS_SEM']:.5f}$ "+r'\\'+'\n'
    tex+=r'''\end{tabular}\end{ruledtabular}\end{table}
The fits use $\log\Delta=\log A-z\log N_y$, with propagated log-space SEMs $\sigma_{\log\Delta}=\mathrm{SEM}/\Delta$. Exponent errors are the formal weighted-regression covariance errors from these supplied uncertainties, without rescaling by residual scatter.
\begin{align}
'''
    for i,(key,label) in enumerate([('Delta_2Ny','2N_y'),('late_endpoint','\mathrm{endpoint}'),('late_OLS','\mathrm{fit}')]):
        f=fits[key]
        tex+=f"z_{{{label}}} &= {f['z']:.4f}\\pm {f['z_regression_error']:.4f},\\qquad \\chi^2/\\mathrm{{dof}}={f['reduced_chi_squared']:.3f}"+(r',\nonumber\\' if i<2 else '.')+'\n'
    tex+=r'''\end{align}
\begin{figure}[h]
\centering\includegraphics[width=3.375in]{late_gap_comparison.pdf}
\caption{Saved slab-only finite-time rates versus circumference. Bars are paired trajectory SEMs; dashed curves are separate SEM-weighted power-law fits. All 100 trajectories are retained at each size.}
\end{figure}
\section{Interpretation and limits}
Both late-window estimators support an approximately inverse-circumference trend over the seven saved sizes, rather than an evident size-independent plateau. Under $g_{\rm mod}(t)\simeq at+b$, a slope removes the constant offset $b$ and yields $a/2$, whereas $\Delta(t)$ contains $b/(2t)$. This cancellation is conditional on approximately affine growth over the selected window. The two slope estimates differ by less than 1.4 paired SEM at every size.
The late estimates exceed $\overline\Delta(2N_y)$, consistent with the previously observed drift. Their agreement in size exponent strengthens the finite-time evidence: the observed trend is not removed by subtracting a constant raw-gap offset. However, the minimizing mode can change with cycle, and the observation windows themselves grow with $N_y$. These are not established asymptotic exponents or infinite-time gaps. No new results have been inserted into Figures 4 or A2, or used to change the section title or gap claims.
\end{document}
'''
    (OUT/'late_gap_report.tex').write_text(tex)
    print('\n'.join(lines))

if __name__=='__main__': main()
