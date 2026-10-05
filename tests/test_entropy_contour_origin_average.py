"""Small dense checks for the matched-alpha origin-averaged contour figure."""
import importlib.util
from pathlib import Path

import numpy as np


PATH = (Path(__file__).resolve().parents[1] /
        '00_WORKSPACE/CURRENT/experiment_review/entropy_contour_alpha_comparison/make_contour_comparison.py')
spec = importlib.util.spec_from_file_location('contour_alpha_comparison_test', PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def random_frame():
    rng = np.random.default_rng(41)
    return np.linalg.qr(rng.normal(size=(24, 10)) + 1j*rng.normal(size=(24, 10)))[0]


def test_periodic_wrapping_cut_and_closure():
    frame = random_frame()
    projector = frame @ frame.conj().T
    indices = np.array([20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6, 7])
    nu, vectors = np.linalg.eigh(projector[np.ix_(indices, indices)])
    reference = ((abs(vectors)**2) @ module.entropy_weights(nu)).reshape(3, 2, 2).sum(axis=2)
    values, entropy, _ = module.contour(frame, 10, 2, 6, y0=5, check_svd=True)
    np.testing.assert_allclose(values, reference, atol=1e-12)
    np.testing.assert_allclose(values.sum(), entropy, atol=1e-12)


def test_origin_average_invariant_under_periodic_translation():
    frame = random_frame()
    translated = np.roll(frame.reshape(6, 4, 10), 2, axis=0).reshape(24, 10)
    first, entropy, diag = module.origin_averaged_contour(frame, 10, 2, 6)
    second, shifted_entropy, _ = module.origin_averaged_contour(translated, 10, 2, 6)
    assert diag['origin_count'] == 6
    np.testing.assert_allclose(first, second, atol=1e-12)
    np.testing.assert_allclose(entropy, shifted_entropy, atol=1e-12)


def test_origin_mean_equals_explicit_cut_mean():
    frame = random_frame()
    values, entropy, _ = module.origin_averaged_contour(frame, 10, 2, 6)
    cuts = [module.contour(frame, 10, 2, 6, y0=y0) for y0 in range(6)]
    np.testing.assert_allclose(values, np.mean([x[0] for x in cuts], axis=0), atol=1e-12)
    np.testing.assert_allclose(entropy, np.mean([x[1] for x in cuts]), atol=1e-12)


def test_product_state_has_exact_zero_contour():
    frame = np.eye(24, dtype=complex)[:, ::2]
    values, entropy, _ = module.origin_averaged_contour(frame, 12, 2, 6)
    np.testing.assert_array_equal(values, np.zeros((3, 2)))
    assert entropy == 0
