from pathlib import Path
import importlib.util
import json

import numpy as np
import pytest
from scipy.linalg import eigvals
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
FILE = ROOT / '00_WORKSPACE/CURRENT/matched_markov_lindblad_campaign/square_large_spectral_v1/run_large.py'
spec = importlib.util.spec_from_file_location('large_square_spectral', FILE)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_extended_contract():
    assert runner.SIZES == (60, 80, 100)
    for a in (1, 3):
        for n in runner.SIZES:
            cfg = runner.config(a, n)
            assert cfg['Nx'] == cfg['Ny'] == n
            assert cfg['walls'] == [n//4, 3*n//4]
            assert cfg['all_slabs_active'] and cfg['dw_truncation']
            assert cfg['alpha_1'] == a and cfg['alpha_2'] == 30
            assert cfg['sequence'] == 'raster_y' and cfg['dtype'] == 'complex128'
            assert not cfg['explicit_dynamics']


@pytest.mark.parametrize('alpha', [1, 3])
def test_full_spectrum_and_action_equivalence(alpha):
    model = runner.spectral.make_model(runner.config(alpha, 4))
    indices, blocks, checks = runner.construct_blocks(model, progress=False)
    full, _ = runner.spectral.construct_product(model, progress=False)
    combined = np.zeros_like(full)
    for idx, block in zip(indices, blocks):
        combined[np.ix_(idx,idx)] = block
    np.testing.assert_allclose(combined, full, atol=1e-14)
    np.testing.assert_allclose(runner.block_action(indices, blocks, np.eye(len(full), dtype=complex)),
                               runner.spectral.independent_action(model, np.eye(len(full))), atol=1e-13)
    actual, dominant, diagnostics = runner.solve_blocks(model, indices, blocks)
    expected = eigvals(full)
    distances = abs(actual[:,None] - expected[None,:])
    i,j = linear_sum_assignment(distances)
    assert distances[i,j].max() < 1e-10
    assert abs(dominant['radius'] - max(abs(expected))) < 1e-12
    assert max(d['dominant_residual'] for d in diagnostics) < 1e-12
    assert checks['inter_sector_support_max'] == 0
    # The exterior spans the periodic seam; it must not be split into two slabs.
    assert sum(map(len, indices)) == len(full)


def test_reject_cross_sector_support():
    model = runner.spectral.make_model(runner.config(1, 4))
    original = model._get_ow_local_support_data
    def leaky(x,y):
        payload = {key: value.copy() for key,value in original(x,y).items()}
        if x == 0:
            # Canonical compact support already excludes the other sector.
            # Inject an additional index there, with tiny but nonzero weight.
            payload['idx'] = np.r_[payload['idx'], 2]
            for name in runner.spectral.ORDER:
                payload[name] = np.r_[payload[name], 1e-15 if name == 'Ap' else 0.]
        return payload
    model._get_ow_local_support_data = leaky
    with pytest.raises(ValueError, match='inter-sector'):
        runner.construct_blocks(model, progress=False)


def test_publication_and_resume(tmp_path):
    runner.run_case(tmp_path, 1, 4)
    assert runner.verified(tmp_path, 1, 4)
    receipt = runner.folder(tmp_path, 1, 4) / 'completion.json'
    before = receipt.read_bytes()
    runner.run_case(tmp_path, 1, 4)
    assert receipt.read_bytes() == before
    data = json.loads(before)
    assert data['diagnostics']['independent_dominant_residual'] < 1e-10
    archive = receipt.parent/'spectrum.npz'
    with archive.open('ab') as handle:
        handle.write(b'corrupt')
    assert not runner.verified(tmp_path, 1, 4)
