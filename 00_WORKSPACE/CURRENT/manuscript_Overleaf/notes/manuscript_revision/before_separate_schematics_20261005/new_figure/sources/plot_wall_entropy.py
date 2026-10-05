#!/usr/bin/env python3
"""Reproduce Figure 9 from bundled anchored curves and original covariance SEMs."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import numpy as np
import wall_entropy_support as base
from manuscript_typography import configure_style as manuscript_style, prepare_figure, record_typography

BUNDLE = Path(__file__).resolve().parent.parent
DATA = BUNDLE / 'data/wall_entropy'
STEM = 'Figure_09_wall_entropy'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(output):
    output.mkdir(parents=True, exist_ok=True)
    provenance = json.loads((DATA/'input_provenance.json').read_text())
    for entry in provenance['sources']:
        path = DATA/Path(entry['path']).name
        if path.suffix == '.csv':
            assert sha256(path) == entry['sha256']
    archived = json.loads((DATA/'original_figure_manifest.json').read_text())
    with (DATA/'lane_B_wall_two_cell_anchored_curves.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    with (DATA/'lane_B_wall_two_cell_joint_fits.csv').open(newline='') as stream:
        fit_rows = {row['wall']: row for row in csv.DictReader(stream)}
    fits = {}
    recovered = {}
    for wall, x_cells in (('left', '5,6'), ('right', '14,15')):
        by_size = {}
        num = den = 0.0
        for ny in base.NY_VALUES:
            selected = sorted((r for r in rows if r['wall'] == wall and int(r['Ny']) == ny),
                              key=lambda r: int(r['Ay']))
            ay = np.array([int(r['Ay']) for r in selected])
            np.testing.assert_array_equal(ay, np.arange(1, ny//2 + 1))
            assert all(int(r['Nx']) == 20 and int(r['samples']) == 100 and
                       r['x_cells'] == x_cells and int(r['anchor_Ay']) == ny//2 for r in selected)
            xx = np.array([float(r['delta_log_sine_chord']) for r in selected])
            mean = np.array([float(r['anchored_entropy_mean']) for r in selected])
            sem = np.array([float(r['anchored_entropy_trajectory_SEM']) for r in selected])
            in_fit = np.array([r['in_fit_window'] == 'True' for r in selected])
            np.testing.assert_array_equal(in_fit, ay >= 8)
            assert np.isfinite(mean).all() and np.isfinite(sem).all() and (sem >= 0).all()
            assert xx[-1] == mean[-1] == sem[-1] == 0
            by_size[ny] = dict(ay_all=ay, x_all=xx, mean_all=mean, sem_all=sem, x_fit=xx[in_fit])
            num += float(xx[in_fit] @ mean[in_fit]) / in_fit.sum()
            den += float(xx[in_fit] @ xx[in_fit]) / in_fit.sum()
        row = fit_rows[wall]
        assert row['x_cells'] == x_cells and int(row['Ay_fit_min']) == 8
        summary = dict(slope=float(row['slope']), slope_covariance_sem=float(row['slope_covariance_SEM']),
                       R0_squared=float(row['R0_squared']))
        for key, value in summary.items():
            assert value == archived['fits'][wall][key]
        np.testing.assert_allclose(num/den, summary['slope'], rtol=0, atol=2e-15)
        recovered[wall] = num/den
        fits[wall] = by_size, summary
    base.FIGURE_DIR = output
    base.make_figure(fits, output/STEM, {
        'left': ('left wall', r'$x=5,6$', r'm_{\rm L}', r'\overline{S}^{\rm wall}_{\rm L}'),
        'right': ('right wall', r'$x=14,15$', r'm_{\rm R}', r'\overline{S}^{\rm wall}_{\rm R}'),
    }, wall_window_cells={'left': np.array([5,6]), 'right': np.array([14,15])},
       inset_right_wall_at_cell_edge=True, plot_min_ay=2)
    checks = dict(input_csv_hashes_verified=True, display_min_Ay=2,
                  wall_windows={'left': [5,6], 'right': [14,15]},
                  fit_window='8 <= Ay <= floor(Ny/2)', fits_unchanged=True,
                  slopes_independently_recovered=recovered,
                  covariance_SEMs='preserved from archived trajectory covariance analysis; not recomputed from averaged CSV',
                  renderer_sha256=sha256(__file__), helper_sha256=sha256(base.__file__),
                  outputs={f'{STEM}.{ext}': sha256(output/f'{STEM}.{ext}') for ext in ('pdf','png')})
    target=output/'data/wall_entropy'; target.mkdir(parents=True, exist_ok=True)
    (target/'validation.json').write_text(json.dumps(checks, indent=2)+'\n')
    print(json.dumps(checks, indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=BUNDLE)
    main(parser.parse_args().output_dir)
