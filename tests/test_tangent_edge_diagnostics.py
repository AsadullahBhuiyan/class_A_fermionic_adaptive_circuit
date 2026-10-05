from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fgtn.classA_U1FGTN import classA_U1FGTN
from fgtn.diagnostics.edge_modes import (
    analytic_domain_wall_bloch_hamiltonian,
    build_physical_edge_frame,
    project_edge_frame,
)
from fgtn.diagnostics.tangent import TangentChannelRecorder


def _domain_wall_model(nx: int = 8, ny: int = 8) -> classA_U1FGTN:
    return classA_U1FGTN(
        nx,
        ny,
        DW=True,
        nshell=1,
        alpha_1=1,
        alpha_2=30,
        trial_orbitals="X",
        dw_truncation=True,
    )


def test_bloch_hamiltonian_matches_real_space_action() -> None:
    model = _domain_wall_model(nx=6, ny=6)
    ky_index = 1
    ky = 2.0 * np.pi * ky_index / model.Ny
    block = analytic_domain_wall_bloch_hamiltonian(model, ky)
    rng = np.random.default_rng(9)
    vector_x = rng.normal(size=2 * model.Nx) + 1j * rng.normal(size=2 * model.Nx)
    phase = np.exp(1j * ky * np.arange(model.Ny)) / np.sqrt(model.Ny)
    vector_full = np.kron(phase, vector_x)
    expected = np.kron(phase, block @ vector_x)
    actual = model._domain_wall_hamiltonian(periodic=True) @ vector_full
    np.testing.assert_allclose(actual, expected, atol=1e-11, rtol=1e-11)


def test_physical_edge_frame_is_wall_resolved_and_projectable() -> None:
    model = _domain_wall_model()
    edge = build_physical_edge_frame(model, wall="left", interface_width=2)
    np.testing.assert_allclose(edge.frame.conj().T @ edge.frame, np.eye(2), atol=1e-11)
    assert np.all(edge.target_wall_weight > 0.9)
    assert np.all(edge.target_wall_weight > edge.opposite_wall_weight)
    assert edge.momentum_indices.tolist() == [0, 1]

    active = model.active_top_layer_indices(meas_slab_only=True)
    projected = project_edge_frame(edge, active)
    np.testing.assert_allclose(projected.frame.conj().T @ projected.frame, np.eye(2), atol=1e-11)
    assert np.all(projected.retained_norm > 0.9)
    inactive = np.setdiff1d(np.arange(edge.frame.shape[0]), active)
    np.testing.assert_allclose(projected.frame[inactive], 0.0, atol=1e-14)


def test_wall_resolution_is_invariant_to_bulk_candidate_budget() -> None:
    model = _domain_wall_model()
    reference = build_physical_edge_frame(model, candidate_count=2)
    enlarged = build_physical_edge_frame(model, candidate_count=8)
    np.testing.assert_array_equal(enlarged.momentum_indices, reference.momentum_indices)
    np.testing.assert_allclose(enlarged.energies, reference.energies, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(
        enlarged.target_wall_weight, reference.target_wall_weight, atol=1e-12, rtol=1e-12
    )
    overlap = np.abs(reference.frame.conj().T @ enlarged.frame)
    np.testing.assert_allclose(overlap, np.eye(2), atol=1e-11, rtol=1e-11)


def test_tangent_recorder_reconstructs_stabilized_restricted_core() -> None:
    model = _domain_wall_model()
    edge = build_physical_edge_frame(model, wall="left", interface_width=2)
    active = model.active_top_layer_indices(meas_slab_only=True)
    edge = project_edge_frame(edge, active)
    recorder = TangentChannelRecorder(
        nx=model.Nx,
        ny=model.Ny,
        samples=1,
        observation_cycles=2,
        edge_frame=edge,
        active_indices=active,
    )
    factors = (
        np.asarray([[2.0, 0.25], [0.0, 0.5]], dtype=np.complex128),
        np.asarray([[0.75, -0.1j], [0.0, 1.5]], dtype=np.complex128),
    )
    product = np.eye(2, dtype=np.complex128)
    for elapsed, factor in enumerate(factors, start=1):
        product = factor @ product
        recorder(
            cycle=3 + elapsed,
            lyapunov_cycle=elapsed,
            lyapunov_frame=edge.frame[None],
            lyapunov_qr_r=factor[None],
            lyapunov_log_diag=np.zeros((1, 2)),
            lyapunov_cycle_null_mask=np.zeros((1, 2), dtype=bool),
            lyapunov_null_counts=np.zeros((1,), dtype=np.int64),
            lyapunov_active_mask=np.ones((1,), dtype=bool),
            lyapunov_min_branch_probability=np.ones((1,)),
            lyapunov_min_abs_born_denominator=2.0 * np.ones((1,)),
            lyapunov_invalid_branch_count=np.zeros((1,), dtype=np.int64),
            batch_start=0,
            batch_count=1,
        )

    recorder.assert_complete()
    reconstructed = (
        recorder.core_hat[0, -1] * np.exp(recorder.core_log_scale[0, -1])
    )
    np.testing.assert_allclose(reconstructed, product, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(recorder.phase_displacement(), 0.0, atol=1e-12)
    assert recorder.core_rank[0, -1] == 2
    assert np.isfinite(recorder.mean_survival_exponent[0, -1])
