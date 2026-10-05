"""Endpoint ky spectra of only the active block in the frozen-exterior runs."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from campaign_schema import sha256_file
from observables import exact_y_twirl, ky_spectrum_and_x_weights, translation_residual

HERE=Path(__file__).resolve().parent
RUN=HERE/'results/alpha3_frozen_exterior_v1_20260914T231338Z'
OUT=HERE/'analysis_outputs/alpha3_frozen_exterior_active_slab_ky'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'CMU Sans Serif','font.size':9,
                         'xtick.direction':'in','ytick.direction':'in'})
    fig,axes=plt.subplots(1,2,figsize=(7.05,3.0),sharey=True)
    products={};summary=[]
    for i,(family,title) in enumerate([('markov_channel','Quantum channel'),('lindblad','Lindblad + dephasing')]):
        root=RUN/(family+'_hard')
        receipt=json.loads((root/'completion.json').read_text())
        config=receipt['config'];path=root/receipt['result_filename']
        assert receipt['status']=='complete'
        assert path.stat().st_size==receipt['result_bytes']
        assert sha256_file(path)==receipt['result_sha256']
        assert config['meas_slab_only'] and config['alpha_1']==3 and config['cycles']==128
        with np.load(path,allow_pickle=False) as z:
            active=z['active_G_final'];indices=z['active_indices'];full=z['G_final'][0]
            expected=np.array([2*(y*20+x)+mu for y in range(64) for x in range(5,16) for mu in (0,1)])
            np.testing.assert_array_equal(indices,expected)
            np.testing.assert_array_equal(active,full[np.ix_(indices,indices)])
            step_distance=float(z['successive_state_distance'][0,-1])
        assert active.shape==(1408,1408)
        twirled=exact_y_twirl(active,11,64)
        ky,occupations,weights=ky_spectrum_and_x_weights(twirled,11,64)
        assert occupations.min()>=-1e-10 and occupations.max()<=1+1e-10
        trace_error=float(abs(occupations.sum()-np.trace(active).real))
        assert trace_error<1e-8
        order=np.argsort(ky);ky=ky[order];occupations=occupations[order];weights=weights[order]
        ax=axes[i]
        ax.scatter(np.repeat(ky/np.pi,22),occupations.ravel(),s=8,
                   marker='o' if i==0 else '^',facecolors='none',
                   edgecolors='#2468ad' if i==0 else '#b3312d',linewidths=.5)
        ax.axhline(.5,color='.5',ls='--',lw=.8)
        ax.set(xlabel=r'$k_y/\pi$',title=title,xlim=(-1.03,1.03),ylim=(-.035,1.035))
        ax.set_xticks([-1,-.5,0,.5,1]);ax.tick_params(top=True,right=True)
        ax.text(-.12,1.06,f'({chr(97+i)})',transform=ax.transAxes)
        products[family+'_occupations']=occupations
        products[family+'_x_weights']=weights
        row=dict(family=family,source=str(path),sha256=receipt['result_sha256'],
            physical_Nx=20,physical_Ny=64,slab_x=[5,15],modes_per_ky=22,
            alpha_1=3,alpha_2=30,nshell=1,endpoint=128,
            minimum_distance_to_half=float(abs(occupations-.5).min()),
            occupation_min=float(occupations.min()),occupation_max=float(occupations.max()),
            k0_central_occupations=occupations[np.argmin(abs(ky)),10:12].tolist(),
            count_within_0p01_of_half=int(np.count_nonzero(abs(occupations-.5)<.01)),
            raw_translation_residual=translation_residual(active,11,64),
            twirled_translation_residual=translation_residual(twirled,11,64),
            trace_error=trace_error,last_step_distance=step_distance,
            estimator='Spectrum of ky diagonal blocks of active_G_final (exact y twirl); no temporal averaging')
        summary.append(row)
    axes[0].set_ylabel(r'Occupation $\nu_a(k_y)$')
    fig.suptitle(r'Active slab only: $x=5,\ldots,15$, $\alpha_1=3$, endpoint 128',fontsize=10)
    fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(OUT/f'active_slab_ky_occupations.{ext}',dpi=300,bbox_inches='tight')
    np.savez_compressed(OUT/'active_slab_ky_occupations.npz',ky=ky,physical_x=np.arange(5,16),**products)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (OUT/'caption.txt').write_text(
        'Active-slab endpoint occupation spectra from the hard-wall frozen-exterior runs. '
        'Physical lattice Nx=20, Ny=64; retain x=5,...,15 and both orbitals (22 modes per ky). '
        'Alpha_1=3, alpha_2=30, nshell=1, perfect correction, complex128, maximally mixed '
        'active initialization; exterior is a separately sampled frozen product state and is '
        'excluded completely. Left: exact outcome-averaged channel after 128 cycles, one seeded '
        'random site schedule. Right: deterministic Lindblad with unit gain/loss/number-dephasing '
        'rates, dt=0.05, t=128. Fourier-transform along y and diagonalize the ky-diagonal blocks; '
        'equivalently apply the exact y-translation twirl before diagonalizing. In the channel '
        'this discards inter-momentum coherences, so it is not the raw untwirled natural spectrum. '
        'No temporal averaging, trajectory sampling error bars, fits, or band interpolation. '
        'The half-occupation reference is shown dashed. These are alpha_1=3 controls, not '
        'alpha_1=1 topological-slab results. Source checksums and translation residuals are in summary.json.\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
