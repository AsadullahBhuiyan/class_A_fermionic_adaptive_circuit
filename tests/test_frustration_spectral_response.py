from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fgtn.classA_U1FGTN import classA_U1FGTN
from fgtn.diagnostics import (
    ChoiSpectrumRecorder,
    LyapunovSpectrumRecorder,
    RegionMasks,
    TrajectoryActivityRecorder,
    analyze_paired_response,
    exact_choi_spectrum,
    local_charge_map,
    localize_vector_batch,
    unit_cell_reset,
    validate_covariance,
)


def _two_by_two_regions() -> RegionMasks:
    active = np.ones((2, 2), dtype=bool)
    left = np.zeros_like(active)
    left[0] = True
    interior = active & ~left
    right = np.zeros_like(active)
    masks = np.stack((active, left, interior, left, right), axis=0)
    return RegionMasks(
        names=("all", "interface", "interior", "left_wall", "right_wall"),
        masks=masks,
        active=active,
        interface_width=1,
        wall_x=(0,),
    )


def test_exact_choi_spectrum_matches_diagonal_reference_and_endpoint_counts():
    a_values = np.asarray([-1.0, -0.6, 0.2, 0.8, 1.0])
    cycle = 2
    result = exact_choi_spectrum(np.diag(a_values), cycle=cycle, n_eigenstates=3)
    finite_expected = np.sort(
        (np.log1p(-a_values[1:4]) - np.log1p(a_values[1:4])) / (2.0 * cycle)
    )

    assert np.isneginf(result["spectrum"][0])
    assert np.isposinf(result["spectrum"][-1])
    np.testing.assert_allclose(result["finite_spectrum"], finite_expected, atol=1e-14)
    np.testing.assert_allclose(result["gap"], np.min(np.abs(finite_expected)), atol=1e-14)
    assert result["particle_zero_count"] == 1
    assert result["particle_pole_count"] == 1
    assert result["finite_eigenstate_count"] == 3
    np.testing.assert_allclose(result["near_gap_residuals"], 0.0, atol=1e-14)
    np.testing.assert_allclose(np.abs(result["eigenstates"][:, 0]), np.eye(5)[:, 2], atol=1e-14)

    recorder = ChoiSpectrumRecorder(samples=1, cycles=(cycle,), dimension=5, n_eigenstates=3)
    recorder(cycle=cycle, sigma_ll=np.diag(a_values)[None, ...], batch_start=0, batch_count=1)
    assert not np.any(np.isinf(recorder.spectrum))
    np.testing.assert_allclose(recorder.spectrum[0, 0, :3], finite_expected, atol=1e-14)
    assert np.all(np.isnan(recorder.spectrum[0, 0, 3:]))


def test_choi_recorder_retains_latest_finite_mode_when_endpoints_saturate():
    recorder = ChoiSpectrumRecorder(samples=1, cycles=(1, 2), dimension=2, n_eigenstates=1)
    recorder(cycle=1, sigma_ll=np.diag([-0.4, 0.3])[None, ...], batch_start=0, batch_count=1)
    first_vector = recorder.final_eigenvectors.copy()
    recorder(cycle=2, sigma_ll=np.diag([-1.0, 1.0])[None, ...], batch_start=0, batch_count=1)

    assert np.isnan(recorder.gap[0, 1])
    assert recorder.finite_count[0, 1] == 0
    assert recorder.final_eigenvector_cycle[0] == 1
    np.testing.assert_allclose(recorder.final_eigenvectors, first_vector)


def test_lyapunov_recorder_and_localization_match_dense_references():
    recorder = LyapunovSpectrumRecorder(samples=2, cycles=2, nvec=3, vector_dimension=4)
    recorder(
        cycle=1,
        spectra=np.asarray([[-1.0, 0.2, 0.7], [-0.5, -0.1, 0.9]]),
        batch_start=0,
        batch_count=2,
    )
    final_vectors = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=np.complex128,
    )
    recorder(
        cycle=2,
        spectra=np.asarray([[-0.8, 0.05, 0.6], [-0.4, -0.2, 0.3]]),
        batch_start=0,
        batch_count=2,
        lyapunov_min_abs_vector=final_vectors,
        lyapunov_min_abs_value=np.asarray([0.05, -0.2]),
        lyapunov_min_abs_index=np.asarray([1, 1]),
        lyapunov_null_counts=np.asarray([0, 1]),
    )
    np.testing.assert_allclose(recorder.gap, [[0.2, 0.05], [0.1, 0.2]])
    np.testing.assert_allclose(recorder.final_vector, final_vectors)
    np.testing.assert_array_equal(recorder.final_index, [1, 1])
    np.testing.assert_array_equal(recorder.null_count, [0, 1])

    localization = localize_vector_batch(
        final_vectors,
        nx=2,
        ny=2,
        regions=_two_by_two_regions(),
        active_indices=np.arange(4),
    )
    np.testing.assert_allclose(localization["ipr"], 1.0)
    np.testing.assert_allclose(localization["region_weight"][:, 0, 0], 1.0)
    np.testing.assert_allclose(localization["region_weight"][0, 0, 1], 1.0)
    np.testing.assert_allclose(localization["region_weight"][1, 0, 2], 1.0)


def test_unit_cell_resets_preserve_covariance_bounds_and_normalize_perturbation():
    nx, ny = 2, 3
    covariance = np.zeros((2 * nx * ny, 2 * nx * ny), dtype=np.complex128)
    plus = unit_cell_reset(covariance, nx=nx, ny=ny, x=1, y=2, occupied=True)
    minus = unit_cell_reset(covariance, nx=nx, ny=ny, x=1, y=2, occupied=False)
    plus_validation = validate_covariance(plus)
    minus_validation = validate_covariance(minus)
    assert plus_validation["hermiticity_residual"] == 0.0
    assert minus_validation["spectral_bound_violation"] == 0.0

    perturbation = 0.5 * (
        local_charge_map(plus, nx=nx, ny=ny) - local_charge_map(minus, nx=nx, ny=ny)
    )
    np.testing.assert_allclose(np.sum(perturbation), 1.0, atol=1e-14)
    np.testing.assert_allclose(perturbation[1, 2], 1.0, atol=1e-14)
    np.testing.assert_allclose(np.count_nonzero(np.abs(perturbation) > 1e-14), 1)


def test_paired_response_extracts_periodic_drift_width_and_wall_weight():
    nx, ny = 2, 8
    active = np.ones((nx, ny), dtype=bool)
    wall = np.zeros_like(active)
    wall[0] = True
    regions = RegionMasks(
        names=("all", "interface"),
        masks=np.stack((active, wall), axis=0),
        active=active,
        interface_width=1,
        wall_x=(0,),
    )
    y0 = 1
    delta = np.zeros((3, 4, nx, ny), dtype=np.float64)
    for time in range(delta.shape[1]):
        delta[:, time, 0, (y0 + time) % ny] = 1.0

    result = analyze_paired_response(delta, regions=regions, y0=y0, fit_max_cycle=3)
    np.testing.assert_allclose(
        result["signed_first_moment"][:, :3, 0],
        np.tile([0.0, 1.0, 2.0], (delta.shape[0], 1)),
    )
    np.testing.assert_allclose(result["response_width"], 0.0, atol=1e-14)
    np.testing.assert_allclose(result["wall_weight"], 1.0, atol=1e-14)
    np.testing.assert_allclose(result["velocity_per_sample"], 1.0, atol=1e-14)
    np.testing.assert_allclose(result["velocity_mean"], 1.0, atol=1e-14)


def _uniform_model() -> classA_U1FGTN:
    model = classA_U1FGTN(2, 2, DW=False, nshell=1, alpha_1=1, alpha_2=1, trial_orbitals="X")
    model.construct_OW_projectors(nshell=1, DW=False, trial_orbitals="X", dw_truncation=False)
    return model


def test_paired_live_runs_use_identical_random_schedules():
    model = _uniform_model()
    covariance = np.zeros((8, 8), dtype=np.complex128)
    plus = unit_cell_reset(covariance, nx=2, ny=2, x=0, y=0, occupied=True)
    minus = unit_cell_reset(covariance, nx=2, ny=2, x=0, y=0, occupied=False)
    plus_recorder = TrajectoryActivityRecorder.from_model(model, cycles=2, samples=1)
    minus_recorder = TrajectoryActivityRecorder.from_model(model, cycles=2, samples=1)
    common = dict(
        G_history=False,
        progress=False,
        cycles=2,
        samples=1,
        save=False,
        perfect_correction=True,
        sequence="random",
        random_seed=999,
    )
    model.run_markov_circuit(G_init=plus, trajectory_weight_observer=plus_recorder, **common)
    model.run_markov_circuit(G_init=minus, trajectory_weight_observer=minus_recorder, **common)
    plus_recorder.assert_complete()
    minus_recorder.assert_complete()
    np.testing.assert_array_equal(plus_recorder.visit_order, minus_recorder.visit_order)
