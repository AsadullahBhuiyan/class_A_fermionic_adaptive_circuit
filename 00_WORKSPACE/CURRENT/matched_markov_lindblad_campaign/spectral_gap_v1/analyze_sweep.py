"""Verify all sixteen spectral cases and build the table, plot, and note inputs."""
import argparse
import csv
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from run_sweep import HERE, SIZES, config, sources, verified_complete, sha, atomic_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args()
    root=args.root.resolve()
    rows=[]
    input_hashes={}
    for alpha in (1,3):
        for ny in SIZES:
            folder=root/f'alpha{alpha}_Ny{ny:03d}'
            if not verified_complete(folder,config(alpha,ny),sources()):
                raise RuntimeError(f'Missing or invalid result: {folder}')
            receipt=json.loads((folder/'completion.json').read_text())
            d=receipt['diagnostics']
            with np.load(folder/'spectrum.npz',allow_pickle=False) as data:
                eigenvalues=data['eigenvalues']
                np.testing.assert_allclose(max(abs(eigenvalues)),d['spectral_radius'],atol=0,rtol=1e-14)
            rows.append(dict(alpha_1=alpha,Nx=20,Ny=ny,dimension=40*ny,
                spectral_radius=d['spectral_radius'],covariance_gap=d['covariance_gap_raw'],
                covariance_multiplier_gap=d['covariance_multiplier_gap'],gap_status=d['gap_status'],
                dominant_residual=d['dominant_residual'],independent_dominant_residual=d['independent_dominant_residual'],
                product_action_error=d['independent_product_action_error'],
                dominant_left_right_overlap=d['dominant_left_right_overlap'],elapsed_seconds=d['elapsed_seconds']))
            input_hashes[str(folder/'completion.json')]=sha(folder/'completion.json')
            input_hashes[str(folder/'spectrum.npz')]=sha(folder/'spectrum.npz')
    out=root/'analysis'
    out.mkdir(exist_ok=True)
    with (out/'gap_table.csv').open('w') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    table=['| alpha_1 | Ny | rho(A) | Delta_C (per cycle) | status |',
           '|---:|---:|---:|---:|:---|']
    for row in rows:
        table.append(f"| {row['alpha_1']} | {row['Ny']} | {row['spectral_radius']:.10g} | {row['covariance_gap']:.10g} | {row['gap_status']} |")
    (out/'gap_table.md').write_text('\n'.join(table)+'\n')
    plt.rcParams.update({'font.family':'CMU Sans Serif','font.size':8,
                         'xtick.direction':'in','ytick.direction':'in'})
    fig,ax=plt.subplots(figsize=(3.375,2.7))
    for alpha,color,marker,ls in [(1,'#2468ad','o','-'),(3,'#c0392b','^',':')]:
        selected=[r for r in rows if r['alpha_1']==alpha]
        values=[r['covariance_gap'] if r['gap_status']=='resolved_positive' else np.nan for r in selected]
        ax.plot([r['Ny'] for r in selected],values,color=color,marker=marker,
                mfc='white',ms=4,lw=.85,ls=ls,label=rf'$\alpha_1={alpha}$')
        for r in selected:
            if r['gap_status']!='resolved_positive':
                ax.plot(r['Ny'],0,'x',color=color)
    ax.set(xlabel=r'$N_y$',ylabel=r'$\Delta_C$ (cycle$^{-1}$)')
    ax.set_xticks(SIZES)
    ax.tick_params(top=True,right=True)
    ax.legend(frameon=False)
    ax.set_ylim(bottom=0)
    fig.tight_layout(pad=.7)
    for ext in ('pdf','png'):
        fig.savefig(out/f'covariance_gap_vs_Ny.{ext}',dpi=300)
    plt.close(fig)
    caption=('Two-point relaxation rate per fixed raster-y cycle, hard walls, Nx=20, '
        'alpha_2=30, nshell=1, inclusive slab x=5,...,15, all slabs active, periodic boundaries, '
        'zero twist, X trial orbitals, perfect correction, measurement dephasing, complex128. '
        'Each point is -2 log(max |eig(A)|) from the complete one-cycle ordered-product spectrum. '
        'Blue circles: alpha_1=1; red triangles: alpha_1=3. Lines guide the eye. '
        'No trajectories, time evolution, initialization, fits, or sampling error bars. '
        'Unresolved unit-modulus cases, if any, are marked by crosses at zero, not assigned a positive gap.\n')
    (out/'figure_caption.txt').write_text(caption)
    ranges={str(a):[min(r['covariance_gap'] for r in rows if r['alpha_1']==a),
                    max(r['covariance_gap'] for r in rows if r['alpha_1']==a)] for a in (1,3)}
    all_resolved=all(r['gap_status']=='resolved_positive' for r in rows)
    maxres=max(r['independent_dominant_residual'] for r in rows)
    if all_resolved:
        sentence=(r'Across these eight circumferences, we find $\Delta_C\in['
            +f"{ranges['1'][0]:.6f},{ranges['1'][1]:.6f}"
            +r']$ for $\alpha_1=1$ and $\Delta_C\in['
            +f"{ranges['3'][0]:.6f},{ranges['3'][1]:.6f}"
            +r']$ for $\alpha_1=3$, in inverse cycles. '
            +r'All 16 cases have resolved positive covariance rates; this finite-size observation is not a proof of a thermodynamic or many-body gap. ')
    else:
        sentence='Some cases have unresolved unit-modulus modes and are not assigned a positive relaxation gap. '
    exponent=int(np.floor(np.log10(maxres))) if maxres else -300
    sentence+=(r'The largest independent dominant-eigenpair residual is $'
               +f'{maxres/10.**exponent:.2f}'+r'\times10^{'+str(exponent)+r'}$.')
    (out/'results_summary.tex').write_text(sentence+'\n')
    shutil.copy2(HERE/'gap_note.tex',out/'gap_note.tex')
    summary=dict(cases=rows,gap_ranges=ranges,all_resolved=all_resolved,
                 explicit_dynamics=False,fit=None,input_sha256=input_hashes)
    atomic_json(out/'summary.json',summary)
    products=[p for p in out.iterdir() if p.suffix in ('.csv','.md','.png','.pdf','.json','.txt','.tex') and p.name!='manifest.json']
    atomic_json(out/'manifest.json',dict(source_sha256={str(p):sha(p) for p in (Path(__file__),HERE/'gap_note.tex')},
                                        output_sha256={p.name:sha(p) for p in products}))
    print('\n'.join(table))
    print('Analysis:',out)


if __name__=='__main__':
    main()
