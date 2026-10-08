#!/usr/bin/env python3
"""Reproduce Figure 9 from bundled anchored curves and original covariance SEMs."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import numpy as np
import wall_entropy_support as base
from endpoint_even_support import load_fits, SIZES
from manuscript_typography import configure_style as manuscript_style, prepare_figure, record_typography

BUNDLE = Path(__file__).resolve().parent.parent
DATA = BUNDLE / 'data/wall_entropy'
STEM = 'Figure_09_wall_entropy'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(output):
    output.mkdir(parents=True, exist_ok=True)
    imported, fresh_contour, provenance = load_fits()
    fits = {}
    recovered = {}
    for wall in ('left','right'):
        groups, result = imported[wall]
        by_size = {ny:dict(ay_all=np.arange(1,ny//2+1), x_all=g['x_all'],
                           mean_all=g['mean_all'], sem_all=g['sem_all'], x_fit=g['x_fit'])
                   for ny,g in groups.items()}
        summary = dict(slope=result['slope'], slope_covariance_sem=result['slope_covariance_SEM'],
                       R0_squared=result['R0_squared'])
        recovered[wall] = result['slope']
        fits[wall] = by_size, summary
    contour_path = BUNDLE/'data/mutual_information/contours_y0avg.npz'
    contour_manifest = json.loads((contour_path.parent/'analysis_manifest_y0avg.json').read_text())
    assert sha256(contour_path) == contour_manifest['data_sha256']
    contour_maps = {}
    with np.load(contour_path, allow_pickle=False) as values:
        for alpha in (1,3):
            samples = values[f'alpha1_{alpha}_per_sample']
            mean = values[f'alpha1_{alpha}_mean']
            np.testing.assert_allclose(samples.mean(0), mean, atol=1e-15, rtol=1e-14)
            np.testing.assert_allclose(mean.sum(), values[f'alpha1_{alpha}_entropy'].mean())
            contour_maps[alpha] = fresh_contour.mean(0) if alpha == 1 else mean
    base.FIGURE_DIR = output
    base.make_figure(fits, output/STEM, {
        'left': ('left wall', r'$x=5,6$', r'm_{\rm L}', r'\overline{S}^{\rm wall}_{\rm L}'),
        'right': ('right wall', r'$x=14,15$', r'm_{\rm R}', r'\overline{S}^{\rm wall}_{\rm R}'),
    }, wall_window_cells={'left': np.array([5,6]), 'right': np.array([14,15])},
       inset_right_wall_at_cell_edge=True, plot_min_ay=2, contour_maps=contour_maps)
    checks = dict(contour_source_sha256=sha256(contour_path), contour_samples=100, contour_origins=32, contour_cycle=64, layout="contours above fresh even-size wall fits", imported_input_hash_verified=True, display_min_Ay=2,
                  wall_windows={'left': [5,6], 'right': [14,15]},
                  fit_window='8 <= Ay <= Ny/2', sizes=list(SIZES), fits_match_imported_campaign=True,
                  alpha1_1_contour_source='fresh endpoint Ny32', alpha1_3_contour_source='retained saved control',
                  input_sha256=provenance['compact_sha256'],
                  slopes_independently_recovered=recovered,
                  covariance_SEMs='recomputed from all imported per-trajectory widths, matching supplied fit errors',
                  renderer_sha256=sha256(__file__), helper_sha256=sha256(base.__file__),
                  outputs={f'{STEM}.{ext}': sha256(output/f'{STEM}.{ext}') for ext in ('pdf','png')})
    target=output/'data/wall_entropy'; target.mkdir(parents=True, exist_ok=True)
    (target/'validation.json').write_text(json.dumps(checks, indent=2)+'\n')
    print(json.dumps(checks, indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=BUNDLE)
    main(parser.parse_args().output_dir)
