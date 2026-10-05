"""Ground-state half-system spectra: dense check, cutoff, normalization."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.linalg import eigh
from threadpoolctl import threadpool_limits

REPO = Path(__file__).resolve().parents[1]
ANALYSIS = REPO / "00_WORKSPACE/CURRENT/experiment_review/domain_wall_correlator_scaling_analysis"
sys.path.insert(0, str(ANALYSIS))
import plot_flattened_half_system_spectra as analysis


def test_filter_and_histogram_normalization():
    raw = np.array([-1e-15, 0, 1e-13, 0.1, 0.5, 0.9, 1 - 1e-13, 1, 1 + 1e-15])
    mask, nu, energy = analysis.resolved_spectrum(raw)
    assert mask.sum() == 3
    np.testing.assert_allclose(energy, [np.log(9), 0, -np.log(9)])
    counts, density = analysis.normalized_histogram(nu, np.linspace(0, 1, 11))
    assert counts.sum() == 3
    np.testing.assert_allclose(density.sum() * 0.1, 1)
    assert json.loads(json.dumps({"ranks": np.array([4, 4])},
                                 default=analysis.json_default))["ranks"] == [4, 4]
    with pytest.raises(FloatingPointError):
        analysis.resolved_spectrum(np.array([-1e-6, 0.5]))


def test_momentum_ground_state_matches_dense_canonical_parent(monkeypatch):
    # Alpha=3 has a gap, so a zero-twist half-filled ground state is unambiguous.
    monkeypatch.setattr(analysis.reference, "OCCUPATION_TWIST", 0.0)
    with threadpool_limits(limits=2):
        model = analysis.build_model(8, 8, 3.0)
        assert model.DW_loc == [2, 6] and model.dw_truncation
        h = np.zeros((128, 128), dtype=np.complex128)
        for name, sign in (("WF_Ap", 1), ("WF_Bp", 1), ("WF_Am", -1), ("WF_Bm", -1)):
            w = getattr(model, name).reshape(128, -1)
            h += sign * (w @ w.conj().T)
        _, vectors = eigh(h)
        occupied = vectors[:, :64]
        dense = occupied @ occupied.conj().T
        delta, diagnostics = analysis.reference.regulated_flattened_momentum_projector(model)
        hybrid = analysis.restricted_projector(delta, range(4))
        np.testing.assert_allclose(hybrid, dense[:64, :64], atol=2e-13, rtol=2e-12)
        np.testing.assert_allclose(eigh(hybrid, eigvals_only=True),
                                   eigh(dense[:64, :64], eigvals_only=True), atol=2e-13)
        assert diagnostics["half_filling_rank"] == 64
        assert diagnostics["projector_idempotency_max_abs"] < 1e-12
