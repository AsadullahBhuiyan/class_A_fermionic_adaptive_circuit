"""Finite-time covariance relaxation rates versus the constant spectral gaps."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from run_dynamics import config, sources, verified_complete, sha, atomic_json, SIZES, FLOOR


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args()
    root=args.root.resolve()
    hashes=sources()
    curves=[]
    rows=[]
    inputs={}
    for alpha in (1,3):
        for size in SIZES:
            folder=root/f'alpha{alpha}_Ny{size:03d}'
            if not verified_complete(folder,config(alpha,size),hashes):
                raise RuntimeError(f'Incomplete or invalid result: {folder}')
            with np.load(folder/'dynamics.npz',allow_pickle=False) as data:
                cycle=data['cycle'].copy()
                increment=data['successive_covariance_rms'].copy()
                gap=float(data['spectral_covariance_gap'])
                expected=data['local_log_decay_rate'].copy()
            rates=np.full(cycle.shape,np.nan,dtype=float)
            valid=(increment[1:-1]>FLOOR)&(increment[2:]>FLOOR)
            ids=np.flatnonzero(valid)+2
            rates[ids]=-np.log(increment[ids]/increment[ids-1])
            np.testing.assert_allclose(rates,expected,rtol=1e-13,atol=1e-13,equal_nan=True)
            assert np.isfinite(rates).sum()>0
            curves.append((alpha,size,cycle,rates,gap))
            for t,rate in zip(cycle,rates):
                rows.append(dict(alpha_1=alpha,Nx=20,Ny=size,cycle=int(t),
                    finite_time_rate=float(rate) if np.isfinite(rate) else None,
                    spectral_covariance_gap=gap,resolved=bool(np.isfinite(rate))))
            for name in ('dynamics.npz','completion.json'):
                inputs[str(folder/name)]=sha(folder/name)
    out=root/'analysis_channel_rates'
    out.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'CMU Sans Serif','font.size':8,
                         'xtick.direction':'in','ytick.direction':'in'})
    colors=plt.get_cmap('viridis')(np.linspace(.03,.85,len(SIZES)))
    markers=['o','s','^','v','D','P','X','>']
    for xmax,name in [(20,'channel_rates_resolved'),(60,'channel_rates_full60')]:
        fig,axes=plt.subplots(1,2,figsize=(7.05,3.25))
        for ax,alpha in zip(axes,(1,3)):
            for a,size,cycle,rates,gap in curves:
                if a!=alpha:
                    continue
                i=SIZES.index(size)
                ax.axhline(gap,color=colors[i],ls='--',lw=.7,alpha=.85,zorder=1)
                ax.plot(cycle,rates,color=colors[i],marker=markers[i],mfc='white',
                        ms=3,lw=.9,label=str(size),zorder=2)
            ax.set(xlabel='cycle',xlim=(1,xmax))
            ax.set_ylim((1.7,2.9) if alpha==1 else (2.7,3.9))
            ax.tick_params(top=True,right=True)
            ax.text(-.13,1.03,'(a)' if alpha==1 else '(b)',transform=ax.transAxes)
            ax.text(.97,.97,rf'$\alpha_1={alpha}$',ha='right',va='top',transform=ax.transAxes)
            ax.legend(title=r'$N_y$',frameon=False,ncol=4,fontsize=7,loc='upper center',
                      bbox_to_anchor=(.5,.95),columnspacing=.7,handlelength=1.2)
        axes[0].set_ylabel(r'Relaxation rate (cycle$^{-1}$)')
        handles=[Line2D([],[],color='black',marker='o',mfc='white',ms=3,lw=.9,label=r'Finite-time $\Delta_{\rm eff}(n)$'),
                 Line2D([],[],color='black',ls='--',lw=.8,label=r'Exact spectral $\Delta_C$ (constant)')]
        fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,0),ncol=2,frameon=False)
        fig.tight_layout(pad=.8,rect=(0,.08,1,1))
        for ext in ('pdf','png'):
            fig.savefig(out/f'{name}.{ext}',dpi=300)
        plt.close(fig)
    with (out/'rates.csv').open('w') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    caption=(r'Hard-wall exact Born-averaged raster-y channel, Nx=20, Ny=20,24,28,32,36,40,44,50, '
        r'alpha_1=1/3, alpha_2=30, nshell=1, inclusive walls x=5,...,15, all slabs active, '
        r'periodic boundaries, zero twist, X orbitals, perfect correction and number dephasing, complex128. '
        r'Full-system maximally mixed C0=I/2; all 60 cycles completed. '
        r'Solid markers: Delta_eff(n)=-log(||C_n-C_(n-1)||_F/||C_(n-1)-C_(n-2)||_F), n>=2. '
        r'Rates require both successive RMS increments above 1e-13; omitted values are unavailable, not zero. '
        r'Dashed horizontal lines: the exact covariance-channel spectral rate Delta_C=-2 log rho(A). '
        r'This spectral gap is time independent; the solid curve is an initial-state-dependent finite-time '
        r'decay diagnostic, not a new gap and not the occupation gap. Nonnormal transients and mode '
        r'overlap can prevent equality before roundoff. Resolved view shows cycles 1--20; full view 1--60. '
        r'No fits, sampled trajectories, uncertainty bars or many-body-gap claim.'+'\n')
    (out/'caption.txt').write_text(caption)
    atomic_json(out/'manifest.json',dict(input_sha256=inputs,source_sha256=sha(__file__),
        output_sha256={f.name:sha(f) for f in out.iterdir() if f.is_file() and f.name!='manifest.json'}))
    print('Verified 16 cases; saved',out)


if __name__=='__main__':
    main()
