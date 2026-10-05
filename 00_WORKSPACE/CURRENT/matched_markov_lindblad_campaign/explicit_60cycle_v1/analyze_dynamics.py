"""Plot observed 60-cycle relaxation against the earlier spectral rates (no fits)."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from run_dynamics import config, sources, verified_complete, sha, atomic_json, SIZES, FLOOR


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--allow-partial',action='store_true')
    args=p.parse_args()
    root=args.root.resolve()
    hashes=sources()
    records=[]
    inputs={}
    for alpha in (1,3):
        for ny in SIZES:
            folder=root/f'alpha{alpha}_Ny{ny:03d}'
            if not verified_complete(folder,config(alpha,ny),hashes):
                if args.allow_partial:
                    continue
                raise RuntimeError(f'Missing or invalid case: {folder}')
            receipt=json.loads((folder/'completion.json').read_text())
            with np.load(folder/'dynamics.npz',allow_pickle=False) as saved:
                data={k:saved[k] for k in saved.files if k!='config_json'}
            inputs[str(folder/'completion.json')]=sha(folder/'completion.json')
            inputs[str(folder/'dynamics.npz')]=sha(folder/'dynamics.npz')
            records.append((alpha,ny,data,receipt))
    if not records:
        raise RuntimeError('No verified completed cases')
    out=root/('analysis' if len(records)==16 else 'analysis_partial')
    out.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'CMU Sans Serif','font.size':8,
                         'xtick.direction':'in','ytick.direction':'in'})
    fig,axes=plt.subplots(1,2,figsize=(7.05,2.9),sharey=True)
    colors=plt.get_cmap('viridis')(np.linspace(.05,.9,len(SIZES)))
    markers=['o','s','^','v','D','P','X','>']
    rows=[]
    for axis,alpha in zip(axes,(1,3)):
        for a,ny,d,receipt in records:
            if a!=alpha:
                continue
            i=SIZES.index(ny)
            cycle=d['cycle']
            inc=d['successive_covariance_rms']
            valid=d['above_numerical_floor']
            normalized=inc/inc[1]
            gap=float(d['spectral_covariance_gap'])
            axis.semilogy(cycle,np.where(valid,normalized,np.nan),color=colors[i],
                marker=markers[i],mfc='white',ms=3,markevery=2,lw=.8,label=str(ny))
            reference=np.exp(-gap*(cycle-1))
            reference[cycle<1]=np.nan
            axis.semilogy(cycle,np.where(reference>1e-14,reference,np.nan),
                          color=colors[i],ls=':',lw=.6,alpha=.65)
            below=np.flatnonzero((cycle>=1)&(inc<=FLOOR))
            rows.append(dict(alpha_1=a,Nx=20,Ny=ny,cycles=60,spectral_gap=gap,
                final_increment_rms=float(inc[-1]),first_cycle_below_1e_13=int(below[0]) if len(below) else None,
                final_charge=float(d['global_charge'][-1]),elapsed_seconds=receipt['diagnostics']['elapsed_seconds'],
                final_min_occupation=float(d['endpoint_occupations'][0]),
                final_max_occupation=float(d['endpoint_occupations'][-1])))
        axis.set(xlabel='cycle',xlim=(0,60),ylim=(1e-14,2))
        axis.tick_params(top=True,right=True)
        axis.text(-.13,1.03,'(a)' if alpha==1 else '(b)',transform=axis.transAxes)
        axis.text(.97,.97,rf'$\alpha_1={alpha}$',transform=axis.transAxes,ha='right',va='top')
        axis.legend(title=r'$N_y$',frameon=False,ncol=2,loc='upper right',bbox_to_anchor=(1,.88),fontsize=7)
    axes[0].set_ylabel(r'$\|C_n-C_{n-1}\|_F/\|C_1-C_0\|_F$')
    fig.tight_layout(pad=.8)
    for ext in ('pdf','png'):
        fig.savefig(out/f'relaxation_vs_spectral.{ext}',dpi=300)
    plt.close(fig)
    # Full 60-cycle increments are retained in a separate numerical-floor diagnostic.
    fig,axes=plt.subplots(1,2,figsize=(7.05,2.9),sharey=True)
    for axis,alpha in zip(axes,(1,3)):
        for a,ny,d,receipt in records:
            if a==alpha:
                inc=d['successive_covariance_rms']
                axis.semilogy(d['cycle'],np.where(inc>0,inc,np.nan),color=colors[SIZES.index(ny)],lw=.8,label=str(ny))
        axis.axhline(FLOOR,color='gray',ls='--',lw=.7)
        axis.set(xlabel='cycle',xlim=(0,60))
        axis.tick_params(top=True,right=True)
        axis.text(-.13,1.03,'(a)' if alpha==1 else '(b)',transform=axis.transAxes)
        axis.text(.97,.97,rf'$\alpha_1={alpha}$',transform=axis.transAxes,ha='right',va='top')
    axes[0].set_ylabel(r'$\|C_n-C_{n-1}\|_F/\sqrt{2N_xN_y}$')
    fig.tight_layout(pad=.8)
    for ext in ('pdf','png'):
        fig.savefig(out/f'full60_increment_diagnostic.{ext}',dpi=300)
    plt.close(fig)
    with (out/'summary.csv').open('w') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    atomic_json(out/'summary.json',dict(complete=len(records),total=16,rows=rows,input_sha256=inputs,
        fit=None,initialization='C0=I/2 on full system',cycles=60))
    caption=('Exact Born-averaged canonical raster-y channel, not sampled trajectories; '
        'full-system maximally mixed C0=I/2; Nx=20, Ny=20,24,28,32,36,40,44,50; '
        'alpha_1=1/3, alpha_2=30; nshell=1; inclusive hard-wall slab x=5,...,15, '
        'all slabs active, X orbitals, periodic boundaries, zero twist, perfect correction '
        'and measurement dephasing, complex128, exactly 60 cycles. '
        'Solid curves: normalized successive covariance changes. Dotted: exp[-Delta_C(n-1)], '
        'normalized at cycle one; these indicate spectral slopes, not fitted amplitudes or bounds. '
        'The increment obeys D_(n+1)=A D_n A^dagger; its observed rate need not equal the '
        'slowest rate if that mode is not excited, and nonnormality can create transients. '
        'Solid curves are censored where RMS increments are <=1e-13. The companion diagnostic '
        'shows uncensored positive increments through cycle 60 with that threshold dashed; '
        'exact zeros are omitted on log axes but remain in NPZ. No decay fit, sampling errors, '
        'Gaussian-state assumption, or thermodynamic/full-many-body gap inference.\n')
    (out/'caption.txt').write_text(caption)
    atomic_json(out/'manifest.json',dict(source_sha256=sha(__file__),
        output_sha256={f.name:sha(f) for f in out.iterdir() if f.is_file() and f.name!='manifest.json'}))
    print(json.dumps(dict(complete=len(records),total=16,output=str(out),rows=rows),indent=2))


if __name__=='__main__':
    main()
