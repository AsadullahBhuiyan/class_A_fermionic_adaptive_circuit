"""Fixed-Nx limiting-gap fits for alpha_1=1,3; deterministic model diagnostics."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_scan import folder, verified, sha, atomic_json


def plateau_fit(n, y, degree=1):
    """Unweighted least squares in the raw gap, not logarithms or centered data."""
    n, y = np.asarray(n, float), np.asarray(y, float)
    if len(n)<=degree or np.any(n<=0):
        raise ValueError('Insufficient points or nonpositive size')
    coefficients = np.polynomial.polynomial.polyfit(1/n, y, degree)
    predicted = np.polynomial.polynomial.polyval(1/n, coefficients)
    return dict(coefficients=coefficients.tolist(), limiting_gap=float(coefficients[0]),
                sizes=n.tolist(), residuals=(y-predicted).tolist(),
                rmse=float(np.sqrt(np.mean((y-predicted)**2))))


def closing_fit(n, y):
    """b*(N/20)^(-z), b>=0, 0<=z<=4, same raw-gap loss as plateau fit."""
    n, y = np.asarray(n, float), np.asarray(y, float)
    def objective(z):
        basis=(n/20)**(-z)
        amplitude=max(0.,float(basis@y/(basis@basis)))
        return float(np.sum((amplitude*basis-y)**2)),amplitude
    opt=minimize_scalar(lambda z:objective(z)[0],bounds=(0,4),method='bounded',
                        options={'xatol':1e-12})
    if not opt.success:
        raise RuntimeError('Closing-power fit failed')
    z=min((0.,float(opt.x),4.),key=lambda z:objective(z)[0])
    loss,b=objective(z)
    return dict(amplitude_at_N20=b,exponent=z,rmse=float(np.sqrt(loss/len(n))),
                sizes=n.tolist(),residuals=(y-b*(n/20)**(-z)).tolist(),
                exponent_bounds=[0,4],at_bound=z in (0.,4.))


def analyze_curve(n,y):
    n,y=np.asarray(n,float),np.asarray(y,float)
    linear=plateau_fit(n,y)
    windows={str(cut):plateau_fit(n[n>=cut],y[n>=cut]) for cut in (20,40,60)}
    quadratic=plateau_fit(n,y,degree=2)
    closing=closing_fit(n,y)
    train_plateau=plateau_fit(n[:-1],y[:-1])
    train_closing=closing_fit(n[:-1],y[:-1])
    plateau_prediction=float(np.polynomial.polynomial.polyval(1/n[-1],train_plateau['coefficients']))
    closing_prediction=float(train_closing['amplitude_at_N20']*(n[-1]/20)**(-train_closing['exponent']))
    estimates=[v['limiting_gap'] for v in windows.values()]+[quadratic['limiting_gap']]
    return dict(plateau=linear,window_fits=windows,quadratic=quadratic,closing_power=closing,
        sensitivity_range=[min(estimates),max(estimates)],
        closing_to_plateau_rmse_ratio=closing['rmse']/linear['rmse'] if linear['rmse'] else None,
        holdout=dict(training_sizes=n[:-1].tolist(),heldout_size=float(n[-1]),observed=float(y[-1]),
            plateau_prediction=plateau_prediction,closing_prediction=closing_prediction,
            plateau_error=plateau_prediction-float(y[-1]),closing_error=closing_prediction-float(y[-1])))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args()
    root=args.root.resolve()
    n=np.array([20,40,60,80,100.])
    curves, results, inputs, raw_rows = {},{}, {},[]
    for index,alpha in ((0,1),(20,3)):
        gaps,conditions=[],[]
        for size in n.astype(int):
            if not verified(root,index,int(size)):
                raise RuntimeError(f'Unverified input: alpha={alpha}, Ny={size}')
            path=folder(root,index,int(size))
            receipt=json.loads((path/'completion.json').read_text())
            cfg=receipt['config']; d=receipt['diagnostics']
            assert cfg['Nx']==20 and cfg['Ny']==size and cfg['alpha_1']==alpha
            with np.load(path/'spectrum.npz',allow_pickle=False) as data:
                radius=float(np.max(abs(data['eigenvalues'])))
                left=data['dominant_left_eigenvector'];right=data['dominant_eigenvector']
                overlap=float(abs(np.vdot(left,right))/(np.linalg.norm(left)*np.linalg.norm(right)))
            gap=float(-2*np.log(radius))
            np.testing.assert_allclose(gap,d['covariance_gap_raw'],atol=1e-12)
            if d['gap_status']!='resolved_positive':
                raise RuntimeError('Unresolved input must be addressed before fitting')
            condition=1/overlap if overlap else float('inf')
            gaps.append(gap);conditions.append(condition)
            raw_rows.append(dict(alpha_1=alpha,Nx=20,Ny=int(size),gap=gap,
                eigenvalue_condition_number=condition,dominant_residual=d['dominant_residual']))
            for name in ('spectrum.npz','completion.json'):
                inputs[str(path/name)]=sha(path/name)
        curves[alpha]=np.array(gaps)
        fit=analyze_curve(n,gaps)
        fit['maximum_eigenvalue_condition_number']=max(conditions)
        results[str(alpha)]=fit
    out=root/'limiting_gap_fits'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out.mkdir(parents=True,exist_ok=False)
    plt.rcParams.update({'font.family':'CMU Sans Serif','font.size':8,
                         'xtick.direction':'in','ytick.direction':'in'})
    fig,axes=plt.subplots(1,2,figsize=(7.05,2.9))
    for ax,alpha,color,marker,panel in zip(axes,(1,3),('#2468ad','#c0392b'),('o','^'),('a','b')):
        y=curves[alpha]; r=results[str(alpha)]
        ax.plot(1/n,y,ls='none',marker=marker,mfc='white',color=color,ms=4,label='computed gap')
        x=np.linspace(0,.051,300)
        ax.plot(x,np.polynomial.polynomial.polyval(x,r['plateau']['coefficients']),
                color=color,lw=1,label=r'$\Delta_\infty+a/N_y$')
        # Closing model shown through x=0 where its mathematical limit is zero.
        # Axis zoom clips the far-offscreen portion, rather than changing its limit.
        yclose=r['closing_power']['amplitude_at_N20']*(20*x)**r['closing_power']['exponent']
        ax.plot(x,yclose,color='.4',ls='--',lw=.9,label=r'$b(N_y/20)^{-z}$')
        ax.plot(0,r['plateau']['limiting_gap'],marker='s',ms=4,color=color)
        ax.axhspan(*r['sensitivity_range'],color=color,alpha=.12)
        ax.set(xlabel=r'$1/N_y$',xlim=(-.001,.052),
               ylim=(r['sensitivity_range'][0]-.015,float(max(y))+.015))
        ax.tick_params(top=True,right=True)
        ax.text(-.12,1.03,f'({panel})',transform=ax.transAxes)
        ax.text(.04,.96,rf'$\alpha_1={alpha}$',transform=ax.transAxes,va='top')
        ax.legend(frameon=False,loc='lower right',fontsize=7)
    axes[0].set_ylabel(r'$\Delta_C$ (cycle$^{-1}$)')
    fig.tight_layout(pad=.8)
    for ext in ('png','pdf'):
        fig.savefig(out/f'limiting_gap_fits.{ext}',dpi=300)
    plt.close(fig)
    summary=[]
    for alpha in (1,3):
        r=results[str(alpha)]
        summary.append(dict(alpha_1=alpha,limiting_gap=r['plateau']['limiting_gap'],
            correction_a=r['plateau']['coefficients'][1],
            sensitivity_low=r['sensitivity_range'][0],sensitivity_high=r['sensitivity_range'][1],
            plateau_rmse=r['plateau']['rmse'],closing_rmse=r['closing_power']['rmse'],
            closing_exponent=r['closing_power']['exponent'],
            plateau_holdout_error=r['holdout']['plateau_error'],closing_holdout_error=r['holdout']['closing_error'],
            max_eigenvalue_condition_number=r['maximum_eigenvalue_condition_number']))
    for filename,rows in (('fit_summary.csv',summary),('input_diagnostics.csv',raw_rows)):
        with (out/filename).open('w') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
            writer.writeheader();writer.writerows(rows)
    atomic_json(out/'fits.json',dict(Nx=20,Ny=n.tolist(),results=results,
        objective='unweighted squared residuals in raw gap',confidence_intervals=None,
        interpretation='Fixed-width empirical extrapolation, not a proof of the limit or a many-body gap.'))
    (out/'caption.txt').write_text(
        'Fixed-width Nx=20 hard-wall covariance-channel gap, alpha_1=1 (a), 3 (b), '
        'Ny=20,40,60,80,100; alpha_2=30, nshell=1, inclusive walls x=5,15, all slabs '
        'active, X orbitals, periodic boundaries, zero twist, raster-y Ap/Am/Bp/Bm, '
        'complex128, perfect correction and measurement dephasing. Points are full '
        'block-spectrum results, not trajectories or relaxation fits; no initial state or '
        'time horizon applies. Solid lines: Delta_infinity+a/Ny, fitted to all five sizes; '
        'dashed: closing power b*(Ny/20)^(-z), 0<=z<=4, fitted to the same raw-gap loss. '
        'Squares mark extrapolated intercepts. Shading spans the linear intercepts for '
        'Ny>=20,40,60 and the all-size quadratic 1/Ny correction; it is model/window '
        'sensitivity, NOT a statistical confidence interval. Lines toward 1/Ny=0 are '
        'extrapolations. Vertical limits zoom the finite gaps and clip the closing curve '
        'near its zero limit. Holdout diagnostics fit Ny<=80 and predict Ny=100. '
        'Five sizes cannot exclude an eventual crossover or other gap-closing forms.\n')
    report=['Limiting-gap fit: fixed Nx=20, only alpha_1=1 and 3.',
        'No new simulation; all ten input pairs verified. Raw data and earlier results unchanged.',
        'Primary fit: Delta(Ny)=Delta_infinity+a/Ny; ordinary unweighted least squares.',
        'Comparator: b*(Ny/20)^(-z), 0<=z<=4, also two parameters and raw-gap least squares.',
        'Sensitivity: fit windows Ny>=20,40,60 and an all-size quadratic inverse-length correction.',
        'No trajectory noise model: no standard-error bars, confidence intervals, p-values or model probabilities.',
        'The sensitivity range is descriptive, not an error bound. Small residuals do not prove an asymptotic law.']
    for row in summary:
        report.append(json.dumps(row))
    report.extend(['Conclusion: both datasets favor a positive fixed-width limiting covariance gap within the tested models and sizes.',
                   'This is not evidence of a full many-body gap or a simultaneous Nx,Ny thermodynamic limit.'])
    (out/'report.txt').write_text('\n'.join(report)+'\n')
    atomic_json(out/'manifest.json',dict(source_sha256=sha(__file__),input_sha256=inputs,
        output_sha256={p.name:sha(p) for p in out.iterdir() if p.is_file()}))
    print(json.dumps(dict(output=str(out),summary=summary),indent=2))


if __name__=='__main__':
    main()
