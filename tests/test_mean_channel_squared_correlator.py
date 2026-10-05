"""Checks of the Figure-12 mean-state analogue, without running dynamics."""
from pathlib import Path
import importlib.util

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]


def load(relative):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


analysis = load("00_WORKSPACE/CURRENT/matched_markov_lindblad_campaign/plot_mean_channel_squared_correlator.py")
legacy = load("00_WORKSPACE/CURRENT/final_production_new_designs/14_hard_wall_xresolved_correlator_scaling/endpoint_correlator.py")


def frame(nx=2, ny=4):
    rng = np.random.default_rng(1515)
    n = 2*nx*ny
    raw = rng.normal(size=(n, n//2)) + 1j*rng.normal(size=(n, n//2))
    return np.linalg.qr(raw)[0]


def test_explicit_indices_and_legacy_observer():
    nx, ny = 2, 4
    f = frame(nx, ny)
    c = f @ f.conj().T
    actual = analysis.square_correlator(c, nx, ny)
    expected = np.zeros_like(actual)
    for x in range(nx):
        for r in range(ny//2+1):
            for y in range(ny):
                for mu in range(2):
                    for nu in range(2):
                        i = mu + 2*x + 2*nx*y
                        j = nu + 2*x + 2*nx*((y+r) % ny)
                        expected[x, r] += abs(c[i, j])**2/(2*ny)
    np.testing.assert_allclose(actual, expected, atol=1e-15)
    observed = legacy.x_resolved_square_correlator_from_frame(
        torch.tensor(f[None]), torch.tensor([f.shape[1]]), nx=nx, ny=ny).numpy()[0]
    np.testing.assert_allclose(actual, observed, atol=1e-15)


def test_normalization_and_no_long_range_contact():
    actual = analysis.square_correlator(.5*np.eye(16), 2, 4)
    np.testing.assert_array_equal(actual[:, 0], [.25, .25])
    np.testing.assert_array_equal(actual[:, 1:], 0)


def test_origin_translation_and_hermitian_transpose():
    f = frame()
    c = f @ f.conj().T
    ids = np.roll(np.arange(16).reshape(4, 2, 2), 1, axis=0).ravel()
    expected = analysis.square_correlator(c, 2, 4)
    np.testing.assert_allclose(analysis.square_correlator(c[np.ix_(ids, ids)], 2, 4), expected)
    np.testing.assert_allclose(analysis.square_correlator(c.T, 2, 4), expected)


def test_mean_before_square_is_distinct():
    plus = np.array([1, 0, 1, 0])/np.sqrt(2)
    minus = np.array([1, 0, -1, 0])/np.sqrt(2)
    cp, cm = np.outer(plus, plus), np.outer(minus, minus)
    mean_square = (analysis.square_correlator(cp, 1, 2)
                   + analysis.square_correlator(cm, 1, 2))/2
    square_mean = analysis.square_correlator((cp+cm)/2, 1, 2)
    assert mean_square[0, 1] > .1
    assert square_mean[0, 1] == 0
    assert np.all(square_mean <= mean_square + 1e-15)


def test_both_figure_variants_render(tmp_path, monkeypatch):
    monkeypatch.setattr(analysis, "OUT", tmp_path)
    curves = {a: np.tile(np.exp(-a*np.arange(33)), (20, 1)) for a in (1, 3)}
    for uncut in (False, True):
        analysis.plot(curves, uncut=uncut)
    for ext in ("pdf", "png"):
        paths = list(tmp_path.glob(f"*.{ext}"))
        assert len(paths) == 2
        assert all(p.stat().st_size > 1000 for p in paths)
