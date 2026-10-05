"""Overlay the two existing plateau fits; omit closing-power comparison curves."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fit-root',type=Path,required=True)
    root=parser.parse_args().fit_root.resolve()
    manifest=json.loads((root/'manifest.json').read_text())
    for name in ('fits.json','input_diagnostics.csv'):
        assert sha(root/name)==manifest['output_sha256'][name]
    fits=json.loads((root/'fits.json').read_text())['results']
    with (root/'input_diagnostics.csv').open() as handle:
        rows=list(csv.DictReader(handle))
    out=root/('combined_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    out.mkdir(exist_ok=False)
    plt.rcParams.update({'font.family':'CMU Sans Serif','font.size':8,
                         'xtick.direction':'in','ytick.direction':'in'})
    fig,ax=plt.subplots(figsize=(3.375,2.8))
    handles=[]
    annotations={}
    for alpha,color,marker,style in ((1,'#2468ad','o','-'),(3,'#c0392b','^',':')):
        selected=sorted((r for r in rows if int(r['alpha_1'])==alpha),key=lambda r:int(r['Ny']))
        assert [int(r['Ny']) for r in selected]==[20,40,60,80,100]
        ax.plot([1/int(r['Ny']) for r in selected],[float(r['gap']) for r in selected],
                ls='none',marker=marker,mfc='white',color=color,ms=4,zorder=3)
        result=fits[str(alpha)]
        x=np.linspace(0,.05,200)
        ax.plot(x,np.polynomial.polynomial.polyval(x,result['plateau']['coefficients']),
                color=color,ls='-',lw=1)
        ax.plot(0,result['plateau']['limiting_gap'],marker='s',color=color,ms=3.5)
        limit, correction = result['plateau']['coefficients']
        observed=np.array([float(r['gap']) for r in selected])
        predicted=limit+correction/np.array([int(r['Ny']) for r in selected])
        sse=float(np.sum((observed-predicted)**2))
        sst=float(np.sum((observed-observed.mean())**2))
        if sst==0:
            raise ValueError('R-squared is undefined for constant observations')
        r_squared=1-sse/sst
        np.testing.assert_allclose(sse,len(observed)*result['plateau']['rmse']**2,rtol=1e-9)
        annotations[str(alpha)]=dict(limiting_gap=limit,a=correction,r_squared=r_squared,
                                    residual_sum_squares=sse,total_sum_squares=sst,n_points=5)
        ax.text(.07, .34 if alpha == 1 else .79,
                rf'$\Delta_\infty={limit:.5f}$'+'\n'+rf'$R^2={r_squared:.6f}$',
                color=color, transform=ax.transAxes, va='top', fontsize=8)
        handles.append(Line2D([],[],color=color,ls='none',marker=marker,mfc='white',ms=4,
                               label=rf'$\alpha_1={alpha}$'))
    handles.append(Line2D([],[],color='.3',ls='-',lw=1,
                          label=r'Fit: $\Delta_\infty+a/N_y$'))
    ax.set(xlabel=r'$1/N_y$',ylabel=r'$\Delta_C$ (cycle$^{-1}$)',
           xlim=(-.001,.052),ylim=(1.65,3.4))
    ax.tick_params(top=True,right=True)
    ax.legend(handles=handles,frameon=False,loc='center right')
    fig.tight_layout(pad=.7)
    for ext in ('png','pdf'):
        fig.savefig(out/f'combined_limiting_gap_fits.{ext}',dpi=300)
    plt.close(fig)
    (out/'fit_annotations.json').write_text(json.dumps(annotations,indent=2)+'\n')
    (out/'caption.txt').write_text(
        'Fixed Nx=20 covariance-channel gap for alpha_1=1 (blue circles) and 3 (red triangles), '
        'Ny=20,40,60,80,100. Marker-only alpha legend entries denote computed data; '
        'the separate solid-line legend entry denotes the fits. Colored solid lines '
        'are the unchanged unweighted raw-gap least-squares fits '
        'Delta_C=Delta_infinity+a/Ny over all five sizes; squares at inverse length zero '
        'mark extrapolated limits 1.7808491365 and 3.1386957330 cycle^-1. Annotations give '
        'these intercepts rounded to five decimal places; correction coefficients are '
        'not labeled. R-squared is 1-SSE/SST evaluated '
        'on the five raw gaps at the fitted sizes (not adjusted R-squared), shown to '
        'six decimal places; it measures in-sample fit, not certainty of extrapolation. Closing power-law '
        'comparison curves omitted. No inset. No statistical error bars or confidence bands shown. '
        'Hard-wall support truncation, inclusive walls x=5,15, all slabs active, alpha_2=30, '
        'nshell=1, X orbitals, periodic boundaries, zero twist, raster-y Ap/Am/Bp/Bm, '
        'complex128, perfect correction and measurement dephasing. No sampled trajectories, '
        'initial state or evolution horizon: direct channel spectra. Fixed-width '
        'extrapolation only, not a proof of a full many-body or 2D thermodynamic gap.\n')
    (out/'manifest.json').write_text(json.dumps(dict(source_sha256=sha(__file__),
        input_sha256={str(root/name):sha(root/name) for name in ('fits.json','input_diagnostics.csv')},
        output_sha256={p.name:sha(p) for p in out.iterdir() if p.is_file()}),indent=2)+'\n')
    print(out)
    print(json.dumps(annotations,indent=2))


if __name__=='__main__':
    main()

