"""Dimensionless covariance multiplier gap g_C=1-rho(A)^2, with fresh fits."""
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
    parser.add_argument('--root',type=Path,required=True)
    root=parser.parse_args().root.resolve()
    source=root/'analysis/gaps.csv'
    manifest=json.loads((source.parent/'manifest.json').read_text())
    assert sha(source)==manifest['output_sha256']['gaps.csv']
    with source.open() as handle:
        rows=list(csv.DictReader(handle))
    out=root/'multiplier_gap_fits'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out.mkdir(parents=True,exist_ok=False)
    plt.rcParams.update({'font.family':'CMU Sans Serif','font.size':8,
                         'xtick.direction':'in','ytick.direction':'in'})
    fig,ax=plt.subplots(figsize=(3.375,2.8))
    fits,converted,handles={},{},[]
    for alpha,color,marker in ((1,'#2468ad','o'),(3,'#c0392b','^')):
        selected=sorted((r for r in rows if float(r['alpha_1'])==alpha),key=lambda r:int(r['Ny']))
        n=np.array([int(r['Ny']) for r in selected])
        np.testing.assert_array_equal(n,[20,40,60,80,100])
        assert all(int(r['Nx'])==20 and r['status']=='resolved_positive' for r in selected)
        rho=np.array([float(r['rho']) for r in selected])
        decay=np.array([float(r['gap']) for r in selected])
        g=1-rho**2
        np.testing.assert_allclose(g,-np.expm1(-decay),atol=1e-14,rtol=1e-14)
        assert np.all((g>=0)&(g<=1))
        # Refit transformed points; do not transform the previous intercept only.
        coefficients=np.polynomial.polynomial.polyfit(1/n,g,1)
        limit,correction=map(float,coefficients)
        predicted=np.polynomial.polynomial.polyval(1/n,coefficients)
        sse=float(np.sum((g-predicted)**2));sst=float(np.sum((g-g.mean())**2))
        r2=1-sse/sst
        windows={str(cut):float(np.polynomial.polynomial.polyfit(1/n[n>=cut],g[n>=cut],1)[0])
                 for cut in (20,40,60)}
        quadratic=float(np.polynomial.polynomial.polyfit(1/n,g,2)[0])
        fits[str(alpha)]=dict(g_infinity=limit,a=correction,r_squared=r2,rmse=float(np.sqrt(sse/5)),
            residual_sum_squares=sse,total_sum_squares=sst,n_points=5,
            fit_window_Ny=[20,40,60,80,100],window_intercepts=windows,
            quadratic_intercept=quadratic,
            sensitivity_range=[min(*windows.values(),quadratic),max(*windows.values(),quadratic)])
        converted[str(alpha)]=[dict(alpha_1=alpha,Nx=20,Ny=int(size),rho_A=float(r),
                                   decay_rate=float(d),g_C=float(value))
                               for size,r,d,value in zip(n,rho,decay,g)]
        ax.plot(1/n,g,ls='none',marker=marker,mfc='white',color=color,ms=4,zorder=3)
        x=np.linspace(0,.05,200)
        ax.plot(x,np.polynomial.polynomial.polyval(x,coefficients),color=color,lw=1)
        ax.plot(0,limit,marker='s',color=color,ms=3.5)
        ax.text(.07,.34 if alpha==1 else .79,
                rf'$g_\infty={limit:.5f}$'+'\n'+rf'$R^2={r2:.6f}$',
                color=color,transform=ax.transAxes,va='top',fontsize=8)
        handles.append(Line2D([],[],color=color,ls='none',marker=marker,mfc='white',ms=4,
                              label=rf'$\alpha_1={alpha}$'))
    handles.append(Line2D([],[],color='.3',ls='-',lw=1,label=r'Fit: $g_\infty+a/N_y$'))
    ax.set(xlabel=r'$1/N_y$',ylabel=r'$g_C=1-\rho(A)^2$',xlim=(-.001,.052),ylim=(.82,.98))
    ax.tick_params(top=True,right=True)
    ax.legend(handles=handles,frameon=False,loc='center right')
    fig.tight_layout(pad=.7)
    for ext in ('png','pdf'):
        fig.savefig(out/f'multiplier_gap_fits.{ext}',dpi=300)
    plt.close(fig)
    with (out/'data.csv').open('w') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(converted['1'][0]))
        writer.writeheader()
        for entries in converted.values():writer.writerows(entries)
    (out/'fits.json').write_text(json.dumps(dict(definition='g_C=1-rho(A)^2=1-exp(-decay_rate)',
        fit_model='g_infinity+a/Ny',objective='unweighted least squares on g_C',
        confidence_intervals=None,results=fits),indent=2)+'\n')
    (out/'caption.txt').write_text(
        'Dimensionless covariance-channel multiplier gap g_C=1-rho(A)^2 at fixed Nx=20, '
        'alpha_1=1 (blue circles),3 (red triangles), Ny=20,40,60,80,100. Points are '
        'converted directly from the saved one-cycle spectra. Lines are NEW unweighted '
        'fits g_C=g_infinity+a/Ny to all five transformed points, not exponentiated '
        'decay-rate fit lines. Squares at inverse length zero mark extrapolated limits. '
        'R-squared=1-SSE/SST on the transformed points; it is not an extrapolation guarantee. '
        'No numerical a labels, inset, closing-power curves, or statistical error bars. '
        'Hard-wall support truncation, all slabs active, inclusive walls x=5,15, alpha_2=30, '
        'nshell=1, X orbitals, periodic, zero twist, raster-y Ap/Am/Bp/Bm, complex128, '
        'perfect correction and measurement dephasing. No trajectories, initial state or '
        'cycle horizon. Fixed-width covariance-sector result, not a full many-body gap.\n')
    (out/'manifest.json').write_text(json.dumps(dict(source_sha256=sha(__file__),
        input_sha256={str(source):sha(source)},figure_inches=[3.375,2.8],
        output_sha256={p.name:sha(p) for p in out.iterdir() if p.is_file()}),indent=2)+'\n')
    print(json.dumps(dict(output=str(out),fits=fits),indent=2))


if __name__=='__main__':
    main()
