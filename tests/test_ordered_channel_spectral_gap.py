from pathlib import Path
import importlib.util
import json

import numpy as np
import pytest
from scipy.linalg import eigvals
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
FILE = ROOT / "00_WORKSPACE/CURRENT/matched_markov_lindblad_campaign/spectral_gap_v1/run_sweep.py"
spec = importlib.util.spec_from_file_location("spectral_gap_runner", FILE)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


@pytest.fixture(scope="module")
def small():
    cfg = runner.config(1, 20)
    cfg.update(Nx=4, Ny=2, walls=[1,2])
    return runner.make_model(cfg)


def test_grid_and_contract():
    cases = [runner.config(a,n) for a in (1,3) for n in runner.SIZES]
    assert len(cases) == 16
    assert len({(c['alpha_1'],c['Ny']) for c in cases}) == 16
    assert all(c['channel_order'] == ['Ap','Am','Bp','Bm'] and not c['explicit_dynamics'] for c in cases)


def test_product_order_normalization_and_independent_action(small):
    a, checks = runner.construct_product(small, progress=False)
    expected = np.eye(len(a), dtype=complex)
    for x in range(small.Nx):
        for y in range(small.Ny):
            for name in runner.ORDER:
                w = getattr(small,'WF_'+name)[:,x,y]
                w = w/np.linalg.norm(w)
                q = np.eye(len(a))-np.outer(w,w.conj())
                expected = q @ expected
    np.testing.assert_allclose(a, expected, atol=1e-13)
    np.testing.assert_allclose(a, runner.independent_action(small, np.eye(len(a))), atol=1e-13)
    assert checks['maximum_mode_norm_error'] < 1e-12
    assert runner.raster_word(small) == [(x,y) for x in range(4) for y in range(2)]


def test_covariance_pair_product_spectrum(small):
    a, _ = runner.construct_product(small, progress=False)
    aa = eigvals(a)
    expected = (aa[:,None]*aa.conj()[None,:]).ravel()
    superoperator = np.kron(a.conj(),a)
    actual = eigvals(superoperator)
    cost = abs(actual[:,None]-expected[None,:])
    i,j = linear_sum_assignment(cost)
    assert cost[i,j].max() < 1e-11
    assert abs(max(abs(actual))-max(abs(aa))**2) < 1e-12
    rng = np.random.default_rng(9)
    x = rng.normal(size=a.shape)+1j*rng.normal(size=a.shape)
    np.testing.assert_allclose(superoperator @ x.ravel(order='F'),
                               (a @ x @ a.conj().T).ravel(order='F'), atol=1e-13)


def test_unresolved_modes_not_filtered():
    assert runner.classify_radius(1.) == 'unresolved_unit_modulus'
    assert runner.classify_radius(1.+1e-12) == 'unresolved_unit_modulus'
    assert runner.classify_radius(.8) == 'resolved_positive'
    with pytest.raises(ValueError):
        runner.classify_radius(1.001)


def test_checksum_resume_partial_and_changed_identity(tmp_path):
    cfg=runner.config(1,20)
    sources={'engine':'abc'}
    assert not runner.verified_complete(tmp_path,cfg,sources)
    result=tmp_path/'spectrum.npz'
    np.savez_compressed(result,eigenvalues=np.zeros(800,dtype=complex),config_json=np.asarray(json.dumps(cfg)))
    assert not runner.verified_complete(tmp_path,cfg,sources)
    receipt=dict(status='complete',config=cfg,sources=sources,result_filename=result.name,
                 result_bytes=result.stat().st_size,result_sha256=runner.sha(result))
    runner.atomic_json(tmp_path/'completion.json',receipt)
    assert runner.verified_complete(tmp_path,cfg,sources)
    assert type(runner.verified_complete(tmp_path,cfg,sources)) is bool
    json.dumps({'complete':sum([runner.verified_complete(tmp_path,cfg,sources)])})
    assert not runner.verified_complete(tmp_path,runner.config(3,20),sources)
    assert not runner.verified_complete(tmp_path,cfg,{'engine':'def'})
    with result.open('ab') as handle:
        handle.write(b'corrupt')
    assert not runner.verified_complete(tmp_path,cfg,sources)
