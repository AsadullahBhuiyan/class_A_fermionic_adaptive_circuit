"""Snapshot verified spectral gaps, independently of explicit-dynamics completion."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from run_square import PROJECT, SIZES, spectral, spectral_config, sources, case_dir, sha, atomic_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--square-root',type=Path,required=True)
    args=p.parse_args()
    root=args.square_root.resolve()
    fixed=PROJECT/'spectral_gap_v1/results/20260929T191448Z'
    rows=[]
    inputs={}
    for geometry in ('fixed','square'):
        for a in (1,3):
            for n in SIZES:
                folder=fixed/f'alpha{a}_Ny{n:03d}' if geometry=='fixed' else case_dir(root,a,n)/'spectral'
                cfg=spectral.config(a,n) if geometry=='fixed' else spectral_config(a,n)
                hashes=spectral.sources() if geometry=='fixed' else sources()
                if not spectral.verified_complete(folder,cfg,hashes):
                    if geometry=='fixed' or (folder/'completion.json').exists():
                        raise RuntimeError(f'Invalid/missing expected spectrum: {folder}')
                    continue
                receipt=json.loads((folder/'completion.json').read_text())
                d=receipt['diagnostics']
                with np.load(folder/'spectrum.npz',allow_pickle=False) as data:
                    radius=float(max(abs(data['eigenvalues'])))
                np.testing.assert_allclose(radius,d['spectral_radius'],rtol=1e-14)
                gap=-2*np.log(radius) if radius else None
                if gap is not None:
                    np.testing.assert_allclose(gap,d['covariance_gap_raw'],rtol=1e-14)
                rows.append(dict(geometry=geometry,alpha_1=a,Nx=cfg['Nx'],Ny=n,
                    gap=gap,rho=radius,status=d['gap_status']))
                for filename in ('spectrum.npz','completion.json'):
                    inputs[str(folder/filename)]=sha(folder/filename)
    out=root/'spectral_snapshots'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out.mkdir(parents=True,exist_ok=False)
    plt.rcParams.update({'font.family':'CMU Sans Serif','font.size':8,
                         'xtick.direction':'in','ytick.direction':'in'})
    fig,axes=plt.subplots(1,2,figsize=(7.05,2.9),sharey=True)
    for ax,geometry in zip(axes,('fixed','square')):
        present=[]
        for a,color,marker,style in [(1,'#2468ad','o','-'),(3,'#c0392b','^',':')]:
            selected=[r for r in rows if r['geometry']==geometry and r['alpha_1']==a]
            present.extend(r['Ny'] for r in selected)
            ax.plot([r['Ny'] for r in selected],
                [r['gap'] if r['status']=='resolved_positive' else np.nan for r in selected],
                color=color,marker=marker,ls=style,mfc='white',ms=4,lw=.9,label=rf'$\alpha_1={a}$')
            for r in selected:
                if r['status']!='resolved_positive':
                    ax.plot(r['Ny'],0,'x',color=color)
        ax.set(xlabel=r'$N_y$',xlim=(18.5,51.5),ylim=(0,3.5))
        ax.set_xticks(SIZES)
        ax.tick_params(top=True,right=True)
        ax.text(-.12,1.03,'(a)' if geometry=='fixed' else '(b)',transform=ax.transAxes)
        ax.text(.03,.78,r'$N_x=20$' if geometry=='fixed' else r'$N_x=N_y$',transform=ax.transAxes)
        ax.legend(frameon=False,loc='lower left')
        if geometry=='square' and len(present)<16:
            ax.text(.97,.45,'Larger sizes\nstill running',ha='right',va='center',
                    transform=ax.transAxes,color='.4')
    axes[0].set_ylabel(r'Exact $\Delta_C=-2\log\rho(A)$ (cycle$^{-1}$)')
    fig.tight_layout(pad=.8)
    for ext in ('png','pdf'):
        fig.savefig(out/f'exact_gap_vs_size.{ext}',dpi=300)
    plt.close(fig)
    with (out/'gaps.csv').open('w') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    (out/'caption.txt').write_text('Exact covariance-channel spectral gaps from complete complex128 '
        'one-cycle spectra, not fits to evolving covariances. Fixed Nx=20 with inclusive walls [5,15] '
        'versus Nx=Ny with walls [floor(Nx/4),floor(3Nx/4)]. Both hard-wall, all slabs active, '
        'alpha_2=30, nshell=1, X orbitals, periodic, zero twist, raster-y Ap/Am/Bp/Bm, '
        'perfect correction and measurement dephasing. No trajectories, initial state, time horizon, '
        'fit or sampling errors enter the spectral calculation. Lines guide the eye only; '
        'no extrapolation across pending sizes. Unresolved modes are marked at zero.\n')
    atomic_json(out/'manifest.json',dict(input_sha256=inputs,source_sha256=sha(__file__),
        fixed_count=sum(r['geometry']=='fixed' for r in rows),square_count=sum(r['geometry']=='square' for r in rows),
        output_sha256={f.name:sha(f) for f in out.iterdir() if f.is_file()}))
    print(out)


if __name__=='__main__':
    main()
