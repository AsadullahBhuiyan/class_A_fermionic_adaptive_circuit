from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "00_WORKSPACE" / "CURRENT" / "mean_channel_lindblad_cpu_campaign"
sys.path.insert(0, str(CAMPAIGN))

from mean_channel_lindblad_cpu import (  # noqa: E402
    MeanChannelLindbladCPU,
    run_dephasing_control,
    run_gain_loss_case,
    validate_selected_observable_schema,
)
from run_campaign import expand_cases  # noqa: E402


def _config() -> dict:
    return json.loads((CAMPAIGN / "campaign_config.json").read_text(encoding="utf-8"))


def _case(nshell: int | None) -> dict:
    config = _config()
    config["physical_time_fraction"] = 1.0
    config["observation_time_fractions"] = [0.0, 0.5, 1.0]
    config["finite_channel_p"] = [1.0, 0.5, 0.25, 0.125]
    return {
        "case_id": f"test-shell-{nshell}",
        "campaign": "L1_TEST",
        "kind": "gain_loss",
        "model": {
            "Nx": 4,
            "Ny": 4,
            "alpha_top": 1.0,
            "alpha_triv": 3.0,
            "domain_wall": True,
            "dw_truncation": True,
            "wall_rule": "legacy",
            "nshell": nshell,
            "n_a": 0.5,
            "dtype": "complex128",
        },
        "run": {
            "init_mode": "maxmix",
            "physical_burn_in_time": 0.0,
            "physical_time_fraction": 1.0,
            "observation_time_fractions": [0.0, 0.5, 1.0],
            "finite_channel_p": [1.0, 0.5, 0.25, 0.125],
            "response": {"enabled": False},
            "save_covariance_history": False,
        },
    }


def test_production_matrix_matches_hybrid_static_and_response_grid() -> None:
    cases = expand_cases(_config(), nx=20, smoke=False)
    main = [case for case in cases if case["campaign"] == "L1_MAIN"]
    controls = [case for case in cases if case["campaign"] == "L1_CONTROL"]
    dephasing = [case for case in cases if case["campaign"] == "L1_DEPHASING_CONTROL"]
    assert len(main) == 39
    assert len(controls) == 21
    assert len(dephasing) == 6
    assert len(cases) == 66
    for alpha in (1.0, 1.5, 1.75, 1.875, 2.125, 2.25, 2.5, 3.0):
        selected = [
            case for case in main
            if case["model"]["Ny"] == 64 and case["model"]["alpha_top"] == alpha
            and case["model"]["dw_truncation"] is True
        ]
        assert {case["model"]["nshell"] for case in selected} == {1, 2, None}
    endpoint = [case for case in main if case["model"]["alpha_top"] == 1.0]
    for ny in (32, 48, 64):
        for dw_truncation in (False, True):
            selected = [
                case for case in endpoint
                if case["model"]["Ny"] == ny
                and case["model"]["dw_truncation"] is dw_truncation
            ]
            assert {case["model"]["nshell"] for case in selected} == {1, 2, None}
            assert all(case["run"]["response"]["enabled"] for case in selected)
    assert not any(case["model"]["Ny"] == 24 for case in main)
    assert sum(case["run"]["response"]["enabled"] for case in main) == 18
    uniform = [case for case in controls if not case["model"]["domain_wall"]]
    assert len(uniform) == 6
    assert all(case["model"]["dw_truncation"] is False for case in uniform)
    assert all(case["run"]["response"]["enabled"] for case in uniform)


@pytest.mark.parametrize("nshell", [1, 2])
def test_finite_shell_products_are_integrated_only(nshell: int) -> None:
    arrays, metadata = run_gain_loss_case(_case(nshell))
    validate_selected_observable_schema(nshell, arrays)
    forbidden = ("ky", "momentum", "branch")
    assert not any(any(token in key.lower() for token in forbidden) for key in arrays)
    assert "occupation_histogram_counts" in arrays
    assert "wall_midgap_x_profile" in arrays
    assert metadata["permanent_covariance_bytes"] == 0
    assert not any("covariance" in key.lower() for key in arrays)


def test_untruncated_product_is_the_only_momentum_resolved_arm() -> None:
    arrays, metadata = run_gain_loss_case(_case(None))
    validate_selected_observable_schema(None, arrays)
    for key in (
        "ky",
        "occupation_spectrum_ky",
        "wall_branch_occupations_ky",
        "wall_branch_weights_ky",
    ):
        assert key in arrays
    assert metadata["model"]["nshell"] is None
    assert metadata["permanent_covariance_bytes"] == 0


def test_finite_channel_is_physical_and_converges_at_fixed_time() -> None:
    arrays, metadata = run_gain_loss_case(_case(1))
    errors = arrays["finite_channel_relative_error_to_continuous"]
    assert np.all(np.diff(errors) < 0.0)
    order = np.polyfit(np.log(arrays["finite_channel_p"]), np.log(errors), 1)[0]
    assert 1.7 < order < 2.3
    assert np.max(arrays["finite_channel_physicality_violation"]) < 1e-12
    assert metadata["solve"]["stationary_relative_residual_max"] < 1e-10
    assert metadata["diagnostics"]["occupation_min"] >= -1e-12
    assert metadata["diagnostics"]["occupation_max"] <= 1.0 + 1e-12


def test_empty_filled_and_maxmix_share_the_stationary_solution() -> None:
    solver = MeanChannelLindbladCPU(
        nx=4,
        ny=4,
        alpha_top=1.0,
        alpha_triv=3.0,
        domain_wall=True,
        dw_truncation=True,
        wall_rule="legacy",
        nshell=1,
        n_a=0.5,
    )
    solution, _ = solver.stationary_solution()
    final = [
        solver.evolve_continuous(solution, times=[200.0], init_mode=mode)[0]
        for mode in ("empty", "filled", "maxmix")
    ]
    for block in final:
        assert np.linalg.norm(block - solution.covariance_blocks) / np.sqrt(block.size) < 2e-7


def test_small_system_frame_operators_match_legacy_implementation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MY_CPU_COUNT", "1")
    path = ROOT / "scripts" / "legacy" / "CI_Lindblad_DW.py"
    spec = importlib.util.spec_from_file_location("legacy_ci_lindblad_dw_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    before = os.sched_getaffinity(0) if hasattr(os, "sched_getaffinity") else None
    spec.loader.exec_module(module)
    if before is not None:
        os.sched_setaffinity(0, before)
    legacy = module.CI_Lindblad_DW(
        4, 4, nshell=1, DW=True, n_a=0.5, alpha_1=3.0, alpha_2=1.0
    )
    legacy.construct_OW_functions()
    legacy_blocks = legacy.partial_fourier_ky_superoperator_no_decoh(
        n_a=0.5, norm="ortho", include_lifted=False
    )
    solver = MeanChannelLindbladCPU(
        nx=4,
        ny=4,
        alpha_top=1.0,
        alpha_triv=3.0,
        domain_wall=True,
        dw_truncation=False,
        wall_rule="legacy",
        nshell=1,
        n_a=0.5,
    )
    v_minus, v_plus, _ = solver.build_frame_operator_blocks()
    assert np.max(np.abs(v_minus - legacy_blocks["single_particle_V_minus_blocks"])) < 1e-12
    assert np.max(np.abs(v_plus - legacy_blocks["single_particle_V_plus_blocks"])) < 1e-12


def test_domain_wall_truncation_masks_cross_sector_support_and_normalizes() -> None:
    solver = MeanChannelLindbladCPU(
        nx=8, ny=6, alpha_top=1.0, alpha_triv=30.0, domain_wall=True,
        dw_truncation=True, wall_rule="canonical", nshell=None, n_a=0.5,
    )
    frame = solver._frame(solver._flattened_hamiltonian(), band_sign=-1, trial_sign=1)
    x0, x1 = solver.wall_locations
    topological = np.zeros(solver.nx, dtype=bool)
    topological[x0 : x1 + 1] = True
    for center_x in range(solver.nx):
        forbidden = ~topological if topological[center_x] else topological
        assert np.max(np.abs(frame[forbidden, :, :, center_x, :])) < 1e-14
    norms = np.sum(np.abs(frame) ** 2, axis=(0, 1, 2))
    assert np.max(np.abs(norms - 1.0)) < 1e-12


def test_domain_wall_truncated_frames_match_canonical_cpu_engine() -> None:
    fgtn_path = ROOT / "src" / "fgtn"
    sys.path.insert(0, str(fgtn_path))
    try:
        from classA_U1FGTN import classA_U1FGTN
    finally:
        sys.path.pop(0)
    solver = MeanChannelLindbladCPU(
        nx=8, ny=6, alpha_top=1.0, alpha_triv=30.0, domain_wall=True,
        dw_truncation=True, wall_rule="canonical", nshell=2, n_a=0.5,
    )
    legacy = classA_U1FGTN(
        8, 6, DW=True, nshell=2, alpha_1=1.0, alpha_2=30.0,
        dw_truncation=True,
    )
    # Align the wall convention before testing the shared OW construction.  The
    # canonical engine's historical default slab is a separately retained control.
    legacy.alpha_profile = solver.alpha_profile.astype(np.complex128)
    legacy.alpha = legacy.alpha_profile
    legacy.DW_loc = list(solver.wall_locations)
    legacy.construct_OW_projectors(nshell=2, DW=True, dw_truncation=True)
    hamiltonian = solver._flattened_hamiltonian()
    for band_sign, trial_sign, attribute in (
        (-1, +1, "WF_Am"), (-1, -1, "WF_Bm"),
        (+1, +1, "WF_Ap"), (+1, -1, "WF_Bp"),
    ):
        expected = getattr(legacy, attribute).reshape(
            2, solver.nx, solver.ny, solver.nx, solver.ny, order="F"
        ).transpose(1, 2, 0, 3, 4)
        actual = solver._frame(
            hamiltonian, band_sign=band_sign, trial_sign=trial_sign
        )
        assert np.max(np.abs(actual - expected)) < 1e-12


def test_uniform_system_rejects_domain_wall_truncation() -> None:
    with pytest.raises(ValueError, match="requires domain_wall=True"):
        MeanChannelLindbladCPU(
            nx=4, ny=4, alpha_top=1.0, alpha_triv=1.0,
            domain_wall=False, dw_truncation=True, nshell=1,
        )


@pytest.mark.parametrize("alpha", [1.0, 30.0])
def test_uniform_stationary_state_has_zero_density_phase_response(alpha: float) -> None:
    solver = MeanChannelLindbladCPU(
        nx=4, ny=8, alpha_top=alpha, alpha_triv=alpha, domain_wall=False,
        dw_truncation=False, wall_rule="canonical", nshell=None, n_a=0.5,
    )
    solution, _ = solver.stationary_solution()
    arrays, metadata = solver.density_phase_response(
        solution, epsilon=1e-3, epsilon_multipliers=[0.5, 1.0, 2.0],
        time_step=0.25, time_fraction=0.5, fit_time_min=0.5,
        fit_time_max_fraction=0.375, wall_window_columns=1,
        finite_channel_p=[1.0, 0.5], source_y=0,
    )
    assert metadata["response_kick_physicality_violation"] < 1e-12
    assert np.max(np.abs(arrays["response_density_ty"])) < 1e-12
    assert np.array_equal(arrays["response_velocity"], np.zeros(2))
    assert np.array_equal(arrays["response_mean_directionality"], np.zeros(2))


def test_density_phase_response_is_physical_translational_and_second_order() -> None:
    solver = MeanChannelLindbladCPU(
        nx=4, ny=8, alpha_top=1.0, alpha_triv=3.0, domain_wall=True,
        dw_truncation=True, wall_rule="legacy", nshell=None, n_a=0.5,
    )
    solution, _ = solver.stationary_solution()
    kwargs = dict(
        epsilon=1e-3, epsilon_multipliers=[0.5, 1.0, 2.0], time_step=0.25,
        time_fraction=0.5, fit_time_min=0.5, fit_time_max_fraction=0.375,
        wall_window_columns=1, finite_channel_p=[1.0, 0.5, 0.25, 0.125],
    )
    arrays0, metadata = solver.density_phase_response(solution, source_y=0, **kwargs)
    arrays1, _ = solver.density_phase_response(solution, source_y=1, **kwargs)
    assert metadata["response_covariance_materialized"] is False
    assert metadata["response_kick_physicality_violation"] < 1e-12
    assert np.max(arrays0["response_epsilon_relative_error"]) < 1e-5
    assert np.allclose(
        arrays1["response_density_ty"],
        np.roll(arrays0["response_density_ty"], 1, axis=2),
        atol=1e-12,
    )
    errors = arrays0["response_finite_channel_relative_error"]
    assert np.all(np.diff(errors) < 0.0)
    order = np.polyfit(np.log(arrays0["response_finite_channel_p"]), np.log(errors), 1)[0]
    assert 1.7 < order < 2.3
    assert arrays0["response_density_ty"].shape == (2, 17, 8)
    assert np.all(np.isfinite(arrays0["response_mean_directionality"]))


def test_low_rank_response_matches_brute_force_real_space_evolution() -> None:
    solver = MeanChannelLindbladCPU(
        nx=4, ny=6, alpha_top=1.0, alpha_triv=3.0, domain_wall=True,
        dw_truncation=True, wall_rule="legacy", nshell=None, n_a=0.5,
    )
    solution, _ = solver.stationary_solution()
    epsilon, comparison_time = 1e-3, 1.0
    arrays, _ = solver.density_phase_response(
        solution, epsilon=epsilon, epsilon_multipliers=[1.0], time_step=0.25,
        time_fraction=0.5, fit_time_min=0.5, fit_time_max_fraction=0.375,
        wall_window_columns=1, finite_channel_p=[1.0, 0.5], source_y=0,
    )
    dimension = solver.ny * solver.d

    def real_matrix(blocks: np.ndarray) -> np.ndarray:
        matrix = np.empty((dimension, dimension), dtype=np.complex128)
        for column in range(dimension):
            vector = np.zeros((solver.ny, solver.d), dtype=np.complex128)
            vector.reshape(-1)[column] = 1.0
            vector_k = np.fft.ifft(vector, axis=0, norm="ortho")
            result_k = (blocks @ vector_k[..., None])[..., 0]
            matrix[:, column] = np.fft.fft(result_k, axis=0, norm="ortho").reshape(-1)
        return matrix

    covariance = real_matrix(solution.covariance_blocks)
    vectors = solution.damping_eigenvectors
    propagator_blocks = (
        vectors * np.exp(-solution.damping_eigenvalues * comparison_time)[:, None, :]
    ) @ np.swapaxes(vectors.conj(), -2, -1)
    propagator = real_matrix(propagator_blocks)
    phase_mask = np.zeros(dimension)
    source_x = solver.wall_locations[0]
    for orbital in (0, 1):
        phase_mask[2 * source_x + orbital] = 1.0
    plus = np.diag(np.exp(-1j * epsilon * phase_mask))
    minus = np.diag(np.exp(+1j * epsilon * phase_mask))
    response = propagator @ (
        plus @ covariance @ plus.conj().T - minus @ covariance @ minus.conj().T
    ) @ propagator.conj().T / (2.0 * epsilon)
    density = np.real(np.diag(response)).reshape(solver.ny, solver.nx, 2).sum(axis=2).T
    columns = sorted({(source_x + offset) % solver.nx for offset in (-1, 0, 1)})
    expected = density[columns].sum(axis=0)
    time_index = int(np.flatnonzero(np.isclose(arrays["response_times"], comparison_time))[0])
    assert np.max(np.abs(expected - arrays["response_density_ty"][0, time_index])) < 1e-12


def test_dephasing_controls_never_emit_momentum_or_covariance_products() -> None:
    config = _config()
    cases = expand_cases(config, nx=20, smoke=False)
    case = next(
        case
        for case in cases
        if case["campaign"] == "L1_DEPHASING_CONTROL" and case["model"]["nshell"] is None
    )
    case = json.loads(json.dumps(case))
    case["model"]["Nx"], case["model"]["Ny"] = 3, 4
    case["run"]["dt"] = 0.1
    case["run"]["physical_time_fraction"] = 0.5
    case["run"]["observation_time_fractions"] = [0.0, 0.5]
    arrays, metadata = run_dephasing_control(case)
    forbidden = ("ky", "momentum", "branch", "covariance")
    assert not any(any(token in key.lower() for token in forbidden) for key in arrays)
    assert metadata["permanent_covariance_bytes"] == 0


def test_colab_case_expansion_has_no_retired_m1_family() -> None:
    shared = (
        ROOT
        / "00_WORKSPACE"
        / "CURRENT"
        / "final_production_ready_figure_scripts"
        / "_shared_src"
        / "campaign_cases.py"
    ).read_text(encoding="utf-8")
    assert "M1_FIXED_SCHEDULE_MEAN" not in shared
    assert "mean_descendant" not in shared
