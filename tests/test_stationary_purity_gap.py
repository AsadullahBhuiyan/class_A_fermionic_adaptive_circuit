"""Regression tests for actual versus translation-twirled stationary spectra."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
PROJECT = REPO/'00_WORKSPACE/CURRENT/matched_markov_lindblad_campaign'
spec = importlib.util.spec_from_file_location('stationary_purity', PROJECT/'steady_purity_gap_v1/run_purity.py')
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)
sys.path.insert(0, str(PROJECT))
from observables import ky_blocks_from_twirled


def test_grid_and_reference_sources():
    for a in pilot.ALPHAS:
        for n in pilot.SIZES:
            c = pilot.config(a, n)
            assert (c['Nx'], c['Ny'], c['cycles']) == (20, n, 61)
            assert c['alpha_1'] == a and c['alpha_2'] == 30
            assert c['walls'] == [5,15] and c['dw_truncation']
            assert c['all_slabs_active'] and c['perfect_correction'] and c['dephasing']
            assert c['sequence'] == 'raster_y' and c['dtype'] == 'complex128'
            assert pilot.reference.verified(pilot.REFERENCE, 0 if a == 1 else 20, n)


def test_spectra_against_dense_and_explicit_twirl():
    nx, ny, walls = 4, 5, [1,2]
    n = 2*nx*ny
    x = (np.arange(n)//2) % nx
    sectors = [(x>=1)&(x<=2), (x<1)|(x>2)]
    rng = np.random.default_rng(10)
    c = np.zeros((n,n), dtype=np.complex128)
    for mask in sectors:
        idx = np.flatnonzero(mask)
        raw = rng.normal(size=(len(idx),len(idx)))+1j*rng.normal(size=(len(idx),len(idx)))
        q,_ = np.linalg.qr(raw)
        c[np.ix_(idx,idx)] = (q*np.linspace(.05,.95,len(idx)))@q.conj().T
    saved = c.copy()
    result = pilot.occupation_spectra(c,nx,ny,walls)
    np.testing.assert_allclose(result['full'], np.linalg.eigvalsh(c), atol=1e-13)
    _, blocks = ky_blocks_from_twirled(c,nx,ny)
    np.testing.assert_allclose(result['ky'], np.linalg.eigvalsh(blocks), atol=1e-13)
    np.testing.assert_array_equal(c,saved)
    # Twirling must not silently replace the actual spectrum.
    assert not np.allclose(result['full'],np.sort(result['ky'].ravel()))
    c[0,2] = c[2,0] = .01
    with pytest.raises(ValueError,match='block decomposition'):
        pilot.occupation_spectra(c,nx,ny,walls)


def test_canonical_smoke_observer_and_dense_projector_sweep():
    cfg = pilot.config(1,4)
    cfg.update(Nx=4, Ny=4, walls=[1,2], cycles=4, observation_cycles=[0,1,3,4])
    model = pilot.spectral.make_model(cfg)
    arrays, diag = pilot.collect(model,cfg,progress=False)
    np.testing.assert_array_equal(arrays['cycles'], np.arange(5))
    assert arrays['occupations_full'].shape == (4,32)
    assert arrays['occupations_ky_twirl'].shape == (4,4,8)
    assert arrays['purity_gap_full'][0] == 0
    assert arrays['purity_gap_twirl'][0] == 0
    n = 32
    c = np.eye(n,dtype=np.complex128)*.5
    for _ in range(4):
        for x,y in pilot.spectral.raster_word(model):
            for name in pilot.spectral.ORDER:
                w = getattr(model,'WF_'+name)[:,x,y]
                w = w/np.linalg.norm(w)
                p = np.outer(w,w.conj())
                q = np.eye(n)-p
                c = q@c@q + (p if name.endswith('m') else 0)
    np.testing.assert_allclose(arrays['C_final'], c, atol=2e-13)
    np.testing.assert_allclose(arrays['global_charge'][-1], np.trace(c).real,atol=1e-12)
    assert diag['converged'] == (max(arrays['successive_frobenius_change'][1:])<pilot.TOL)


def test_completion_missing_and_corrupt(tmp_path):
    assert not pilot.verified(tmp_path,1,20)
    folder = pilot.case_dir(tmp_path,1,20)
    folder.mkdir()
    (folder/'completion.json').write_text('{}')
    assert not pilot.verified(tmp_path,1,20)
