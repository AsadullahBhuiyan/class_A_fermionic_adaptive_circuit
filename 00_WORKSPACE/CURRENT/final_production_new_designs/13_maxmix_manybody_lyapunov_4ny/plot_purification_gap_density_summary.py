#!/usr/bin/env python3
"""Compose existing verified purification panels and Ny=30 endpoint density."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from plot_purification_gap_summary import ROOT, OUT as SOURCE, make_figure

OUT = ROOT / 'analysis_outputs/purification_contour_gap_density_4x1_v1_single_column'
DENSITY_ROOT = ROOT.parents[1] / 'experiment_review/purification_slow_mode_profiles/hard_wall_mean_heatmaps'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(name):
    with (SOURCE/name).open() as stream:
        return [{k:(v if k=='window' else int(v) if k in ('Ny','x','cycle','samples','bundle') else float(v)) for k,v in row.items()}
                for row in csv.DictReader(stream)]


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    manifest = json.loads((DENSITY_ROOT/'manifest.json').read_text())
    expected = {item['file']:item['sha256'] for item in manifest['outputs']}
    means = {}
    inputs = []
    for ny in [20,30,40]:
        path = DENSITY_ROOT/f'hard_Ny{ny:03d}_mean_density.npz'
        assert sha(path)==expected[path.name], f'Checksum mismatch: {path}'
        with np.load(path,allow_pickle=False) as data:
            assert int(data['Ny'])==ny and int(data['cycles'])==4*ny and int(data['samples'])==100
            np.testing.assert_array_equal(sorted(data['sample_indices']),np.arange(100))
            mean = data['mean_density']
            assert mean.shape==(ny,20)
            np.testing.assert_allclose(mean,data['sample_densities'].mean(axis=0),atol=1e-15)
            np.testing.assert_allclose(mean.sum(),1,atol=1e-12)
            means[ny] = mean
        inputs.append(dict(path=str(path),sha256=sha(path)))
    # Preserve the original common scale, even though only Ny=30 is shown.
    vmax = max(float(m.max()) for m in means.values())
    summary = json.loads((SOURCE/'analysis_summary.json').read_text())
    fit = summary['weighted_log_space_power_law_fit']
    make_figure(read_rows('total_entropy_curves.csv'), read_rows('spatial_entropy_curves.csv'),
                read_rows('gap_summary.csv'), fit, density=means[30], density_vmax=vmax, output=OUT)
    for name in ['total_entropy_curves.csv','spatial_entropy_curves.csv','gap_summary.csv','analysis_summary.json']:
        inputs.append(dict(path=str(SOURCE/name),sha256=sha(SOURCE/name)))
    record = dict(figure_size_inches=[3.375,6.8], panel_d_Ny=30,
                  panel_d_estimator=manifest['estimator'], color_scale=[0,vmax],
                  inputs=inputs, fit=fit, script_sha256=sha(Path(__file__)),
                  plotting_script_sha256=sha(ROOT/'plot_purification_gap_summary.py'),
                  note='Panels a-c reuse existing verified tables unchanged; no new fitting or simulations.')
    (OUT/'analysis_summary.json').write_text(json.dumps(record,indent=2)+'\n')
    print(OUT)


if __name__=='__main__': main()
