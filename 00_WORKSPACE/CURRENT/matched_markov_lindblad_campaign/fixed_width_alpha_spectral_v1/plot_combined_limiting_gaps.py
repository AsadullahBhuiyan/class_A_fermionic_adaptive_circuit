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
    sweep_root=root.parent.parent/'analysis'
    sweep_manifest=json.loads((sweep_root/'manifest.json').read_text())
    sweep_csv=sweep_root/'gaps.csv'
    assert sha(sweep_csv)==sweep_manifest['output_sha256']['gaps.csv']
    with sweep_csv.open() as handle:
        sweep_rows=list(csv.DictReader(handle))
    assert len(sweep_rows)==105
    assert all(int(r['Nx'])==20 and r['status']=='resolved_positive' for r in sweep_rows)
    for row in rows:
        match=[r for r in sweep_rows if float(r['alpha_1'])==float(row['alpha_1'])
               and int(r['Ny'])==int(row['Ny'])]
        assert len(match)==1
        np.testing.assert_allclose(float(match[0]['gap']),float(row['gap']),atol=1e-12)
    out=root/('combined_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    out.mkdir(exist_ok=False)
    plt.rcParams.update({'font.family':'CMU Sans Serif','font.size':10,
                         'xtick.direction':'in','ytick.direction':'in'})
    fig,ax=plt.subplots(figsize=(7.05,4.4))
    handles=[]
    annotations={}
    for alpha,color,marker,style in ((1,'#2468ad','o','-'),(3,'#c0392b','^',':')):
        selected=sorted((r for r in rows if int(r['alpha_1'])==alpha),key=lambda r:int(r['Ny']))
        assert [int(r['Ny']) for r in selected]==[20,40,60,80,100]
        ax.plot([1/int(r['Ny']) for r in selected],[float(r['gap']) for r in selected],
                ls='none',marker=marker,mfc='white',color=color,ms=5,zorder=3)
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
        ax.text(.065, .27 if alpha == 1 else .81,
                rf'$\Delta_\infty={limit:.5f}$'+'\n'+rf'$R^2={r_squared:.6f}$',
                color=color, transform=ax.transAxes, va='top', fontsize=10)
        handles.append(Line2D([],[],color=color,ls='none',marker=marker,mfc='white',ms=4,
                               label=rf'$\alpha_1={alpha}$'))
    handles.append(Line2D([],[],color='.3',ls='-',lw=1,
                          label=r'Fit: $\Delta_\infty+a/N_y$'))
    ax.set(xlabel=r'$1/N_y$',ylabel=r'$\Delta_C$ (cycle$^{-1}$)',
           xlim=(-.001,.052),ylim=(1.65,3.4))
    ax.tick_params(top=True,right=True)
    ax.legend(handles=handles,frameon=False,loc='center left',bbox_to_anchor=(.05,.52))
    fig.tight_layout(pad=.7)
    inset=ax.inset_axes([.56,.27,.41,.49])
    sizes=(20,40,60,80,100)
    colors=('#c0392b','#23934c','#2468ad','#8e44ad','#d17b0f')
    markers=('^','s','o','D','v')
    styles=(':','--','-','-.',(0,(3,1,1,1)))
    for size,color,marker,style in zip(sizes,colors,markers,styles):
        selected=sorted((r for r in sweep_rows if int(r['Ny'])==size),key=lambda r:float(r['alpha_1']))
        np.testing.assert_allclose([float(r['alpha_1']) for r in selected],np.linspace(1,3,21),atol=1e-12)
        inset.plot([float(r['alpha_1']) for r in selected],[float(r['gap']) for r in selected],
                   color=color,ls=style,marker=marker,mfc='white',ms=2.5,lw=.8,label=str(size))
    inset.set(xlim=(.95,3.05),ylim=(1.4,3.4),xticks=[1,1.5,2,2.5,3],yticks=[1.5,2,2.5,3])
    inset.set_xlabel(r'$\alpha_1$',fontsize=8,labelpad=1)
    inset.set_ylabel(r'$\Delta_C$',fontsize=8,labelpad=2)
    inset.tick_params(top=True,right=True,labelsize=8,length=3,pad=2)
    inset.legend(title=r'$N_y$ ($N_x=20$)',frameon=False,loc='upper left',
                 ncol=2,fontsize=8,title_fontsize=8,handlelength=1.5,
                 columnspacing=.7,labelspacing=.25,borderaxespad=.4)
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
        'comparison curves omitted. Inset: the full 105-point gap-versus-alpha sweep '
        'for alpha_1=1.0,1.1,...,3.0 and the same five Ny values at Nx=20. Inset lines '
        'connect computed points as guides to the eye, not fits. No statistical error '
        'bars or confidence bands shown. '
        'Hard-wall support truncation, inclusive walls x=5,15, all slabs active, alpha_2=30, '
        'nshell=1, X orbitals, periodic boundaries, zero twist, raster-y Ap/Am/Bp/Bm, '
        'complex128, perfect correction and measurement dephasing. No sampled trajectories, '
        'initial state or evolution horizon: direct channel spectra. Fixed-width '
        'extrapolation only, not a proof of a full many-body or 2D thermodynamic gap.\n')
    (out/'manifest.json').write_text(json.dumps(dict(source_sha256=sha(__file__),
        input_sha256={**{str(root/name):sha(root/name) for name in ('fits.json','input_diagnostics.csv')},
                      str(sweep_csv):sha(sweep_csv)},
        output_sha256={p.name:sha(p) for p in out.iterdir() if p.is_file()}),indent=2)+'\n')
    print(out)
    print(json.dumps(annotations,indent=2))


if __name__=='__main__':
    main()
