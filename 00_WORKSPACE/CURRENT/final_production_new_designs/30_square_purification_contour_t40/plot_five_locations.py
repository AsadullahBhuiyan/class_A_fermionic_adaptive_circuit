"""Offline, verified five-location purification contours; no dynamics rerun."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from io_utils import sha
from run_campaign import default_config, identity, load_pair

ROOT = Path(__file__).resolve().parent
CONFIG = default_config()
DATA = ROOT / 'data' / CONFIG['sampling_revision']
OUT = ROOT / 'analysis_outputs/five_location_purification'
DRIVE_FILES = {
    30: {'result.npz': '1baCShQHzzaPL4nYL8ToqzbTnGmYSn1Wt',
         'result.json': '1jKUMYnwyVZFK0Bgl3t9pqTRvPMmJbHBV'},
    40: {'result.npz': '1Yo40GAW68dC0-U65dRyka1zVEV7gvbsz',
         'result.json': '10EhkvSynyzXlEBJyG_2NFb5OPZe8H7tz'}}


def five_locations(L, walls):
    left, right = map(int, walls)
    # Boundary columns belong to the topological region in the canonical engine.
    return [('Left wall', left), ('Right wall', right),
            ('Left trivial bulk', left//2), ('Right trivial bulk', (right+L)//2),
            ('Topological center', L//2)]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'CMU Sans Serif', 'font.size': 9,
                         'mathtext.fontset': 'cm', 'xtick.direction': 'in',
                         'ytick.direction': 'in', 'legend.frameon': False})
    fig, axes = plt.subplots(2, 1, figsize=(3.375, 7.0), layout='constrained')
    colors = ['#d94738', '#1675bd', '#249b57', '#8951a1', '#222222']
    markers = ['^', 'o', 's', 'D', 'x']
    lines = [':', '-', '--', '-.', '-']
    rows, profiles, summaries, files = [], [], [], []
    eps = CONFIG['entropy_eps']
    h = lambda n: -n*np.log(n) - (1-n)*np.log1p(-n)
    floor_low, floor_high = 2*min(h(eps), h(1-eps)), 2*max(h(eps), h(1-eps))
    for ax, L, letter in zip(axes, (30, 40), 'ab'):
        p = load_pair(DATA, L, 'result', identity(CONFIG, L))
        if p is None:
            raise ValueError(f'Invalid or incomplete result pair for L={L}')
        np.testing.assert_array_equal(p['cycles'], np.arange(41))
        np.testing.assert_array_equal(p['sample_indices'], [0])
        assert int(p['Nx']) == int(p['Ny']) == L
        assert json.loads(str(p['configuration_json'])) == CONFIG
        np.testing.assert_array_equal(p['walls'], [L//2-L//4, L//2+L//4])
        contour, nu = p['entropy_contour'], p['occupation_spectrum']
        assert contour.dtype == np.float64 and contour.shape == (41,L,L)
        assert np.isfinite(contour).all() and np.all(contour >= 0)
        assert np.min(nu) >= -CONFIG['occupation_tolerance']
        assert np.max(nu) <= 1+CONFIG['occupation_tolerance']
        np.testing.assert_allclose(contour.sum((1,2)), p['total_entropy'], rtol=1e-11, atol=1e-10)
        np.testing.assert_allclose(h(np.clip(nu, eps, 1-eps)).sum(1), p['total_entropy'], rtol=1e-11, atol=1e-10)
        np.testing.assert_allclose(nu.sum(1), p['global_charge'], atol=1e-9)
        np.testing.assert_allclose(contour[0], 2*np.log(2), atol=1e-12)
        # Stored contour axes are (cycle,x,y): observer transposes the (y,x) array.
        profile = contour.mean(axis=2)
        np.testing.assert_allclose(profile.sum(1)*L, p['total_entropy'], atol=1e-9)
        locations = five_locations(L, p['walls'])
        result_locations = []
        for index, ((label, x), color, marker, ls) in enumerate(zip(locations, colors, markers, lines)):
            values = profile[:, x]
            ax.plot(p['cycles'], values, color=color, marker=marker, ls=ls,
                    label=rf'{label} ($x={x}$)', lw=1.1, ms=3.1, mfc='white', mew=.8,
                    markevery=(index % 3, 3))
            for t, value in zip(p['cycles'], values):
                rows.append(dict(L=L, sample=0, location=label, x=x, cycle=int(t),
                                 y_averaged_entropy_nats_per_cell=float(value)))
            result_locations.append(dict(location=label, x=x, at_cycle20=float(values[20]),
                                         at_cycle40=float(values[40])))
        for t in range(41):
            for x in range(L):
                profiles.append(dict(L=L, cycle=t, x=x, entropy_nats_per_cell=profile[t,x]))
        ax.axhline(floor_low, color='.6', lw=.7, ls='--', zorder=0)
        ax.text(.98, .12, 'entropy regulator floor', color='.4', fontsize=8,
                ha='right', transform=ax.transAxes)
        ax.text(.96, .94, rf'${L}\times{L},\ S=1$', ha='right', va='top', transform=ax.transAxes)
        ax.set(xlabel='cycle', ylabel=r'$\langle s(x,y,t)\rangle_y$ (nats/cell)',
               xlim=(0,40), yscale='log', ylim=(2e-11,2))
        ax.tick_params(top=True, right=True)
        ax.legend(loc='lower left', bbox_to_anchor=(-.02,1.035), ncol=1,
                  fontsize=8, handlelength=3, labelspacing=.2, borderaxespad=0)
        ax.text(-.22, 1.035, f'({letter})', transform=ax.transAxes)
        for name, file_id in DRIVE_FILES[L].items():
            path = DATA / f'L{L:03d}_sample000' / name
            files.append(dict(path=str(path.relative_to(ROOT)), bytes=path.stat().st_size,
                              sha256=sha(path), drive_url=f'https://drive.google.com/file/d/{file_id}/view'))
        summaries.append(dict(L=L, samples=1, cycles=40, walls=p['walls'].tolist(),
                              locations=result_locations, final_total_entropy=float(p['total_entropy'][-1]),
                              max_closure_error=float(p['entropy_closure_error'].max()),
                              elapsed_seconds=float(p['elapsed_seconds'])))
    for ext in ('pdf', 'png'):
        fig.savefig(OUT/f'five_location_purification.{ext}', dpi=300)
    plt.close(fig)
    for name, table in [('five_location_curves.csv', rows), ('all_x_profiles.csv', profiles)]:
        with (OUT/name).open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(table[0]))
            writer.writeheader(); writer.writerows(table)
    summary = dict(config=CONFIG, results=summaries, sources=files,
                   estimator='Arithmetic mean over all y of full-system entropy contour at fixed x, '
                   'two orbitals summed per cell. One trajectory per size; no ensemble error bars.',
                   contour_axis_order=['cycle','x','y'], regulator_floor_range=[floor_low,floor_high],
                   boundary_convention='x_L and x_R are the inclusive topological-slab endpoint columns.',
                   periodic_caveat='The two displayed trivial-bulk portions connect across the periodic x seam.',
                   script_sha256=sha(Path(__file__)),
                   products={p.name:sha(p) for p in OUT.iterdir() if p.suffix in ('.pdf','.png','.csv')})
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (DATA/'IMPORT_MANIFEST.json').write_text(json.dumps(dict(date='2026-09-29',
        drive_folder='https://drive.google.com/drive/folders/18U3fG_WT4AB75tFY1r8C1i1MSttnWUMT',
        files=files, verified_sizes=[30,40], cycles=[0,40], samples_per_size=1), indent=2)+'\n')
    print(json.dumps(summaries,indent=2))


if __name__ == '__main__':
    main()
