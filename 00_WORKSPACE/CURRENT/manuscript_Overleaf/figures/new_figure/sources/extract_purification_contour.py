#!/usr/bin/env python3
"""Copy and validate saved cycle-60 entropy contours; never simulate dynamics."""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/purification'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    provenance = json.loads((DATA/'occupation_source_provenance.json').read_text())
    inputs = [s for s in provenance['inputs'] if '/alpha1_1/' in s['path']]
    assert len(inputs) == 20
    contours = np.empty((100, 20, 30))
    entropy = np.empty(100)
    seen = set()
    for item in inputs:
        assert digest(item['path']) == item['sha256']
        with np.load(item['path'], allow_pickle=False) as z:
            assert int(z['alpha_1']) == 1 and int(z['cycles'][-1]) == 60
            ids = z['sample_indices']
            assert not seen.intersection(ids.tolist())
            seen.update(ids.tolist())
            contours[ids] = z['entropy_contour'][:, -1]
            entropy[ids] = z['total_entropy'][:, -1]
    assert seen == set(range(100))
    assert np.isfinite(contours).all() and contours.min() >= 0 and entropy.min() > 0
    np.testing.assert_allclose(contours.sum((1,2)), entropy, atol=1e-14, rtol=1e-13)
    normalized = contours/entropy[:,None,None]
    np.testing.assert_allclose(normalized.sum((1,2)), 1, atol=1e-12)
    mean = contours.mean(0)
    with (DATA/'total_entropy_curves.csv').open() as f:
        rows = list(csv.DictReader(f))
    endpoint = next(r for r in rows if int(r['alpha_1']) == 1 and int(r['cycle']) == 60)
    np.testing.assert_allclose(mean.sum(), 30*float(endpoint['mean']), atol=1e-13)
    np.testing.assert_allclose(entropy.std(ddof=1)/10, 30*float(endpoint['sem']), atol=1e-13)
    output = DATA/'Ny030_entropy_contour_cycle60.npz'
    np.savez_compressed(output, sample_ids=np.arange(100), raw_contours=contours,
        entropy_per_trajectory=entropy, mean_raw_contour=mean,
        raw_trajectory_SEM=contours.std(0,ddof=1)/10,
        normalized_contours=normalized, mean_normalized_contour=normalized.mean(0),
        normalized_trajectory_SEM=normalized.std(0,ddof=1)/10)
    receipt = dict(status='passed', alpha_1=1, Nx=20, Ny=30, cycle=60, samples=100,
        source_observable='Full-system entropy contour from saved observer, summed over both orbitals',
        initialization='Full system maximally mixed', protocol='full measurement',
        estimator='Compute within each trajectory, then arithmetic mean; no spatial average or smoothing',
        normalization_in_manuscript=False, normalized_preview='Divide each trajectory by its own total entropy, then average',
        entropy_log_base='natural', entropy_log_cutoff=1e-12,
        entropy_closure_max_error=float(abs(contours.sum((1,2))-entropy).max()),
        mean_total_entropy=float(mean.sum()), raw_color_limits=[0,.0075],
        normalized_color_limits=[0,.02], inputs=inputs,
        output_sha256=digest(output), script_sha256=digest(__file__), new_simulations=False)
    (DATA/'entropy_contour_provenance.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print('Validated 100 saved contours; mean total entropy:', mean.sum())


if __name__ == '__main__':
    main()
