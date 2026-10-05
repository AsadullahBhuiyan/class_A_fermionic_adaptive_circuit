"""Verified square 2Ny-cycle comparison, with square-geometry spectral references."""
import argparse
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from run_square import SIZES, config, verified, case_dir, sha, atomic_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args()
    root=args.root.resolve()
    rows=[]
    records=[]
    inputs={}
    for alpha in (1,3):
        for size in SIZES:
            if not verified(root,alpha,size):
                raise RuntimeError(f'Incomplete square case alpha={alpha}, L={size}')
            folder=case_dir(root,alpha,size)
            receipt=json.loads((folder/'completion.json').read_text())
            with np.load(folder/'dynamics.npz',allow_pickle=False) as data:
                cycle=data['cycle'].copy()
                increment=data['successive_covariance_rms'].copy()
                charge=float(data['global_charge'][-1])
            spec=json.loads((folder/'spectral/completion.json').read_text())['diagnostics']
            rows.append(dict(alpha_1=alpha,L=size,cycles=2*size,wall_left=size//4,wall_right=3*size//4,
                rho=spec['spectral_radius'],covariance_gap=spec['covariance_gap_raw'],
                spectral_status=spec['gap_status'],final_increment_rms=float(increment[-1]),
                final_charge=charge,elapsed_dynamics_seconds=receipt['diagnostics']['elapsed_seconds']))
            records.append((alpha,size,cycle,increment,spec))
            for relative in ['completion.json','dynamics.npz','spectral/completion.json','spectral/spectrum.npz']:
                path=folder/relative
                inputs[str(path)]=sha(path)
    out=root/'analysis'
    out.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'CMU Sans Serif','font.size':8,'xtick.direction':'in','ytick.direction':'in'})
    colors=plt.get_cmap('viridis')(np.linspace(.05,.9,len(SIZES)))
    markers=['o','s','^','v','D','P','X','>']
    fig,axes=plt.subplots(1,2,figsize=(7.05,2.9),sharey=True)
    for axis,alpha in zip(axes,(1,3)):
        for a,size,cycle,inc,spec in records:
            if a!=alpha:
                continue
            i=SIZES.index(size)
            axis.semilogy(cycle,np.where(inc>1e-13,inc/inc[1],np.nan),color=colors[i],
                marker=markers[i],ms=3,markevery=2,mfc='white',lw=.8,label=str(size))
            if spec['gap_status']=='resolved_positive':
                reference=np.exp(-spec['covariance_gap_raw']*(cycle-1))
                axis.semilogy(cycle,np.where((cycle>0)&(reference>1e-14),reference,np.nan),
                    color=colors[i],ls=':',lw=.6,alpha=.65)
        axis.set(xlabel='cycle',xlim=(0,100),ylim=(1e-14,2))
        axis.text(-.13,1.03,'(a)' if alpha==1 else '(b)',transform=axis.transAxes)
        axis.text(.97,.97,rf'$\alpha_1={alpha}$',transform=axis.transAxes,ha='right',va='top')
        axis.tick_params(top=True,right=True)
        axis.legend(title=r'$N_x=N_y$',frameon=False,ncol=2,loc='upper right',bbox_to_anchor=(1,.88),fontsize=7)
    axes[0].set_ylabel(r'$\|C_n-C_{n-1}\|_F/\|C_1-C_0\|_F$')
    fig.tight_layout(pad=.8)
    for ext in ('pdf','png'):
        fig.savefig(out/f'square_relaxation.{ext}',dpi=300)
    plt.close(fig)
    fig,axis=plt.subplots(figsize=(3.375,2.7))
    for a,color,marker,style in [(1,'#2468ad','o','-'),(3,'#c0392b','^',':')]:
        selected=[r for r in rows if r['alpha_1']==a]
        axis.plot([r['L'] for r in selected],
            [r['covariance_gap'] if r['spectral_status']=='resolved_positive' else np.nan for r in selected],
            color=color,marker=marker,ls=style,mfc='white',ms=4,lw=.8,label=rf'$\alpha_1={a}$')
        for row in selected:
            if row['spectral_status']!='resolved_positive':
                axis.plot(row['L'],0,'x',color=color)
    axis.set(xlabel=r'$N_x=N_y$',ylabel=r'$\Delta_C$ (cycle$^{-1}$)')
    axis.tick_params(top=True,right=True)
    axis.legend(frameon=False)
    axis.set_ylim(bottom=0)
    fig.tight_layout(pad=.7)
    for ext in ('pdf','png'):
        fig.savefig(out/f'square_gap.{ext}',dpi=300)
    plt.close(fig)
    with (out/'summary.csv').open('w') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    atomic_json(out/'summary.json',dict(complete=16,total=16,rows=rows,input_sha256=inputs,fit=None))
    (out/'caption.txt').write_text(
        'Square hard-wall exact Born-averaged channel, Nx=Ny=20,24,28,32,36,40,44,50; '
        'alpha_1=1/3, alpha_2=30, nshell=1, inclusive walls floor(L/4),floor(3L/4), '
        'all slabs active, periodic boundaries, zero twist, X orbitals, perfect correction '
        'with measurement dephasing, complex128. Full-system maximally mixed C0=I/2. '
        'Exactly 2Ny cycles, no samples, no fit or sampling errors. '
        'Solid curves: normalized successive covariance increments, censored when RMS <=1e-13; '
        'all cycles remain in data. Dotted curves: same-geometry spectral slope guides, '
        'not fitted amplitudes or pointwise bounds. Local initial-state overlaps and nonnormal '
        'transients can differ from the slowest asymptotic rate. Gap-plot lines guide the eye; '
        'unresolved modes are crosses at zero, not assigned positive rates. '
        'No thermodynamic or many-body gap inference.\n')
    atomic_json(out/'manifest.json',dict(source_sha256=sha(__file__),
        output_sha256={f.name:sha(f) for f in out.iterdir() if f.is_file() and f.name!='manifest.json'}))
    print('[analysis complete]',out,flush=True)


if __name__=='__main__':
    main()
