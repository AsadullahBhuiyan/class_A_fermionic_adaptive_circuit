#!/usr/bin/env python3
"""Extract fixed-cut spectra from saved campaign-09 frames; no dynamics run.

Ordinary figure reproduction needs only occupation_inputs.npz. This optional
extraction step requires the original campaign and verifies its completion receipts.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.linalg import eigvalsh
from threadpoolctl import threadpool_limits
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/entanglement_spectrum'
CAMPAIGN = Path('00_WORKSPACE/CURRENT/final_production_new_designs/09_pure_tangent_replay_acquisition')
REVISION = 'pure_tangent_replay_nx20_ny24-32_hard-soft_a1-1-3_s100_v1'
CONFIG = 'd1a7c212d8f2c06b0f774af14272a2c71b9d0482240f0c4c43be75a9cd494be4'


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', type=Path)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    repo = args.repo_root or next(p for p in ROOT.parents if (p / CAMPAIGN).is_dir())
    indices = np.arange(640)  # 2*Nx*y + 2*x + orbital; y=0,...,15; all x.
    spectra = np.empty((2, 100, 640))
    records, checks = [], []
    source_hashes = None
    with threadpool_limits(limits=args.threads), tqdm(total=200, desc='Saved-frame spectra', unit='trajectory') as progress:
        for ai, alpha in enumerate((1, 3)):
            directory = repo / CAMPAIGN / 'gpu_data' / REVISION / f'hard/Ny032/alpha1_{alpha}'
            receipts = sorted(directory.glob('*.complete.json'))
            assert len(receipts) == 4
            seen = []
            for receipt_path in receipts:
                receipt = json.loads(receipt_path.read_text())
                for key, expected in dict(Nx=20, Ny=32, cycles=64, construction='hard', alpha_1=alpha,
                                          alpha_2=30, nshell=1, status='complete',
                                          configuration_sha256=CONFIG, sampling_revision=REVISION,
                                          canonical_entry_point='classA_U1FGTN_gpu.run_markov_circuit').items():
                    assert receipt[key] == expected, (receipt_path, key)
                if source_hashes is None:
                    source_hashes = receipt['source_hashes']
                assert source_hashes == receipt['source_hashes']
                path = directory / receipt['result_filename']
                assert path.stat().st_size == receipt['result_bytes']
                assert digest(path) == receipt['result_sha256']
                with np.load(path, allow_pickle=False) as data:
                    for key, expected in dict(Nx=20, Ny=32, cycles_total=64, construction='hard',
                                              alpha_1=alpha, alpha_2=30, nshell=1, sequence='raster_y',
                                              dtype='complex128', configuration_sha256=CONFIG,
                                              prepared_initial_state='post_exterior_product_frame').items():
                        assert data[key].item() == expected, (path, key)
                    ids = data['case_sample_indices'].tolist()
                    assert ids == receipt['case_sample_indices']
                    assert data['global_sample_indices'].tolist() == receipt['global_sample_indices']
                    assert json.loads(data['source_hashes_json'].item()) == source_hashes
                    frames, ranks = data['final_frame'], data['final_ranks']
                assert frames.shape[:2] == (25, 1280) and frames.dtype == np.complex128
                assert ranks.shape == (25,) and len(ids) == 25
                seen.extend(ids)
                records.append(dict(alpha_1=alpha, result=str(path.relative_to(repo)),
                                    receipt=str(receipt_path.relative_to(repo)),
                                    receipt_sha256=digest(receipt_path), completion=receipt))
                for row, sample in enumerate(ids):
                    rank = int(ranks[row])
                    assert 0 < rank <= frames.shape[2]
                    frame = frames[row, :, :rank]
                    assert np.isfinite(frame).all()
                    gram = float(np.max(np.abs(frame.conj().T @ frame - np.eye(rank))))
                    assert gram < 1e-8
                    restricted = frame[indices]
                    occupation = restricted @ restricted.conj().T
                    centered = 2 * occupation - np.eye(640)
                    herm = float(np.max(np.abs(centered - centered.conj().T)))
                    assert herm < 1e-12
                    values = eigvalsh((centered + centered.conj().T) / 2, driver='evr')
                    assert values.min() >= -1-1e-8 and values.max() <= 1+1e-8
                    trace_error = float(abs(values.sum() - np.trace(centered)))
                    assert trace_error < 1e-9
                    if sample == 0:
                        np.testing.assert_allclose(values, 2*eigvalsh(occupation, driver='evd')-1,
                                                   rtol=0, atol=1e-10)
                    spectra[ai, sample] = values
                    checks.append(dict(alpha_1=alpha, sample_id=sample, rank=rank, gram_residual=gram,
                                       hermiticity_residual=herm, trace_error=trace_error))
                    progress.update()
                del frames
            assert sorted(seen) == list(range(100))
    # Independent agreement with the existing alpha_1=1, fixed-origin spectrum.
    reference = repo / CAMPAIGN / 'analysis_outputs/half_system_centered_spectrum_n20x32_hard_alpha1_v1/centered_spectra.npz'
    with np.load(reference, allow_pickle=False) as data:
        np.testing.assert_array_equal(data['sample_ids'], np.arange(100))
        np.testing.assert_array_equal(data['subsystem_indices'], indices)
        cache_error = float(np.max(np.abs(spectra[0] - data['eigenvalues'])))
        assert cache_error < 1e-10
    output = DATA / 'occupation_inputs.npz'
    np.savez_compressed(output, centered_occupations=spectra, alpha_1=np.array([1, 3]),
                        sample_ids=np.arange(100), subsystem_indices=indices,
                        Nx=20, Ny=32, Ay=16, cycle=64, origin=0)
    provenance = dict(schema='full_occupation_comparison_v1', no_dynamics_rerun=True,
                      extraction_source_sha256=digest(Path(__file__)),
                      compact_input_sha256=digest(output), source_files=records,
                      reference_cache=str(reference.relative_to(repo)), reference_sha256=digest(reference),
                      reference_max_error=cache_error, sample_checks=checks,
                      protocol=dict(Nx=20, Ny=32, Ay=16, cycle=64, origin=0, alpha_1=[1,3], alpha_2=30,
                                    samples_per_alpha=100, modes_per_sample=640, walls=[5,15], nshell=1,
                                    construction='hard/support-terminated', measurements='slab-only',
                                    initialization='pure; product exterior', sequence='raster_y',
                                    perfect_correction=True, postselect=False),
                      presentation=dict(variable='lambda=2*occupation-1', full_range=[-1,1], bins=100,
                                        normalization='unit-area probability density separately per alpha',
                                        pooling='100 trajectories, one fixed origin y0=0 per trajectory',
                                        endpoint_roundoff='clip only excursions below tolerance 1e-8'))
    (DATA / 'occupation_provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    print(json.dumps(dict(shape=list(spectra.shape), reference_max_error=cache_error,
                          minimum=float(spectra.min()), maximum=float(spectra.max()), output=str(output)), indent=2))


if __name__ == '__main__':
    main()
