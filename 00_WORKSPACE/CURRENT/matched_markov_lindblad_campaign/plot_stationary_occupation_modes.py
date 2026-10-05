"""Momentum-resolved transverse profiles of occupations nearest half filling."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from observables import ky_blocks_from_twirled, translation_residual

HERE = Path(__file__).resolve().parent
ROOT = HERE / 'results/raster_y_channel_endpoints_v1_20260915T025739Z'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    products, summary, inputs = {}, {}, {}
    for alpha in (1, 3):
        folder = ROOT / f'alpha{alpha}_hard'
        receipt = folder / 'completion.json'
        record = json.loads(receipt.read_text())
        source = folder / record['result_filename']
        assert record['status'] == 'complete'
        assert source.stat().st_size == record['result_bytes']
        assert sha(source) == record['result_sha256']
        cfg = record['config']
        assert (cfg['Nx'], cfg['Ny'], cfg['alpha_1'], cfg['cycles']) == (20,64,alpha,128)
        assert cfg['wall'] == 'hard' and cfg['site_schedule'] == 'raster_y'
        inputs.update({str(p):sha(p) for p in (receipt, source)})
        with np.load(source, allow_pickle=False) as data:
            assert json.loads(data['config_json'].item()) == cfg
            final = data['G_final'][0]
            twirl = data['G_final_twirl'][0]
            late_change = float(np.max(data['successive_state_distance'].ravel()[-10:]))
            assert late_change < 1e-12
            ky, blocks = ky_blocks_from_twirled(twirl,20,64)
            values, vectors = np.linalg.eigh(blocks)
            np.testing.assert_allclose(values,data['twirled_ky_occupations'],atol=1e-12)
            raw_translation_residual = translation_residual(final,20,64)
        order = np.argsort(ky)
        ky, blocks, values, vectors = ky[order], blocks[order], values[order], vectors[order]
        residual = float(np.max(np.linalg.norm(blocks@vectors-vectors*values[:,None,:],axis=1)))
        assert residual < 1e-12
        np.testing.assert_allclose(vectors.conj().transpose(0,2,1)@vectors,
                                   np.broadcast_to(np.eye(40),(64,40,40)),atol=1e-12)
        weights = np.sum(abs(vectors.reshape(64,20,2,40))**2,axis=2)
        np.testing.assert_allclose(weights.sum(axis=1),1,atol=1e-12)
        distance = abs(values-.5)
        # Include any numerical tie at the second-closest distance, avoiding
        # an arbitrary basis choice at a degenerate selection boundary.
        threshold = np.sort(distance,axis=1)[:,1]
        selected = distance <= threshold[:,None]+1e-10
        counts = selected.sum(axis=1)
        profile = np.einsum('kxm,km->kx',weights,selected)/counts[:,None]
        np.testing.assert_allclose(profile.sum(axis=1),1,atol=1e-12)
        key = f'alpha{alpha}'
        products.update({f'{key}_occupations':values,f'{key}_eigenvectors':vectors,
                         f'{key}_all_x_weights':weights,f'{key}_selection_mask':selected,
                         f'{key}_profile':profile})
        summary[key] = dict(config=cfg,last_ten_cycles_max_change=late_change,
            original_translation_residual=raw_translation_residual,
            eigenpair_residual=residual,min_distance_to_half=float(distance.min()),
            selected_modes_per_ky=counts.tolist(),
            selected_occupation_min=float(values[selected].min()),
            selected_occupation_max=float(values[selected].max()),
            mean_weight_on_wall_sites=float(profile[:,[5,15]].sum(axis=1).mean()))
    out = HERE / 'analysis_outputs/stationary_occupation_modes' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out.mkdir(parents=True,exist_ok=False)
    plt.rcParams.update({'font.family':'CMU Sans Serif','font.size':9,
                         'xtick.direction':'in','ytick.direction':'in'})
    fig,axes = plt.subplots(1,2,figsize=(7.05,3.35),sharex=True,sharey=True,layout='constrained')
    vmax = max(products[f'alpha{a}_profile'].max() for a in (1,3))
    step = 2/64
    for ax,alpha,letter in zip(axes,(1,3),('a','b')):
        im = ax.imshow(products[f'alpha{alpha}_profile'],origin='lower',aspect='auto',
            extent=(-.5,19.5,ky[0]/np.pi-step/2,ky[-1]/np.pi+step/2),
            cmap='magma',vmin=0,vmax=vmax,interpolation='nearest')
        for wall in (5,15):
            ax.axvline(wall,color='cyan',ls='--',lw=.7)
        ax.set(xlabel=r'$x$',title=rf'$\alpha_1={alpha}$',xticks=[0,5,10,15,19],
               yticks=[-1,-.5,0,.5,1])
        ax.tick_params(top=True,right=True)
        ax.text(-.06,1.04,f'({letter})',transform=ax.transAxes,ha='right',va='bottom')
    axes[0].set_ylabel(r'$k_y/\pi$')
    fig.colorbar(im,ax=axes,fraction=.045,pad=.025,label=r'Mean mode weight $w(x;k_y)$')
    for extension in ('png','pdf'):
        fig.savefig(out/f'occupation_modes_alpha1_alpha3.{extension}',dpi=300)
    plt.close(fig)
    np.savez_compressed(out/'occupation_eigenvectors.npz',ky=ky,x=np.arange(20),**products)
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (out/'caption.txt').write_text(
        'Nx=20, Ny=64 hard-wall stationary-endpoint occupation eigenvectors: '
        '(a) alpha_1=1 and (b) alpha_1=3. At each momentum diagonalize the exact '
        'y-translation twirl of the saved cycle-128 correlation matrix. Choose '
        'the two occupations closest to 1/2 (including all ties within 1e-10 of '
        'the second distance); color is their mean orbital-summed weight at x, '
        'normalized to sum_x w=1. Same linear color scale. This is a subspace '
        'profile, not one eigenvector, a tracked band, or a relaxation mode of A. '
        'All 40 occupation eigenvectors at all 64 momenta for each alpha are saved. '
        'Cyan dashed lines indicate inclusive wall sites x=5,15; all slabs active. '
        'Full real-space plane-wave weights would be w(x;ky)/Ny. '
        'The untwirled endpoints are stationary at cycle boundaries (last ten '
        'normalized increments below 1e-12); the twirl is a postprocessing operation '
        'and need not itself be a fixed point of the ordered channel. '
        'Alpha_2=30, nshell=1, X trial orbitals, periodic boundaries, zero twist, '
        'raster-y Ap/Am/Bp/Bm, complex128, maximally mixed initialization, '
        'perfect correction and measurement dephasing. Exact outcome-averaged '
        'evolution; no trajectory sampling, temporal average, fit, or error bars.\n')
    (out/'manifest.json').write_text(json.dumps(dict(
        source_sha256={str(p):sha(p) for p in (Path(__file__),HERE/'observables.py')},
        input_sha256=inputs,output_sha256={p.name:sha(p) for p in out.iterdir() if p.is_file()}),
        indent=2)+'\n')
    print(json.dumps(dict(output=str(out),diagnostics={k:{n:v for n,v in row.items()
        if n not in ('config','selected_modes_per_ky')} for k,row in summary.items()},
        mode_counts={k:sorted(set(row['selected_modes_per_ky'])) for k,row in summary.items()}),indent=2))


if __name__=='__main__':
    main()
