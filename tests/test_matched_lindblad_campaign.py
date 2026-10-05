from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


REPO = Path(__file__).resolve().parents[1]
CAMPAIGN = REPO / "00_WORKSPACE" / "CURRENT" / "matched_markov_lindblad_campaign"
if str(CAMPAIGN) not in sys.path:
    sys.path.insert(0, str(CAMPAIGN))

from lindblad_adapter import run_lindblad_case
from lindblad_response import local_density_kick_response, local_kick_q_sector
from matched_model import build_model
from src.fgtn.diagnostics.mean_lindblad import (
    PerfectCorrectionLindblad,
    dense_to_y_momentum,
    extract_q_sector,
    integrate_rk4,
)


def _case(*, dephasing: bool, response: bool = False) -> dict:
    return {
        "schema": "test",
        "case_id": f"small-deph{int(dephasing)}",
        "campaign_roles": ["smoke" if response else "unit"],
        "model": {
            "Nx": 4,
            "Ny": 6,
            "domain_wall": True,
            "wall_locations": [1, 2],
            "alpha_run_in": 1.0,
            "alpha_run_out": 30.0,
            "trial_orbitals": "X",
            "nshell": 1,
            "dw_truncation": True,
        },
        "dynamics": {
            "family": "lindblad",
            "dephasing": bool(dephasing),
            "perfect_correction": True,
            "init_mode": "maxmix",
            "cycles": 12,
            "physical_time": 12.0,
            "sample_ids": [0],
            "sample_seeds": [17],
        },
    }


def _config(*, response: bool = False) -> dict:
    return {
        "dynamics": {"lindblad": {"dt": 0.05}},
        "analysis": {"physicality_tolerance": 1e-8},
        "response": {
            "enabled": bool(response),
            "lindblad_output_dt": 0.25,
            "lindblad_source_y": [0],
            "epsilon": 1e-3,
            "epsilon_multipliers": [0.5, 1.0, 2.0],
            "wall_window_columns": 1,
            "fit_time_min": 0.5,
            "lindblad_q_batch_size": 2,
        },
    }


@pytest.fixture(scope="module")
def small_generator() -> PerfectCorrectionLindblad:
    model = build_model(_case(dephasing=False))
    assert model.DW_loc == [1, 2]
    for name in ("WF_Ap", "WF_Am", "WF_Bp", "WF_Bm"):
        norms = np.sum(np.abs(getattr(model, name)) ** 2, axis=0)
        assert np.allclose(norms, 1.0, atol=2e-13)
    result = PerfectCorrectionLindblad.from_canonical_model(model)
    assert result.y_translation_residual < 1e-12
    return result


def _nontrivial_baseline(
    generator: PerfectCorrectionLindblad, *, dephasing: bool
) -> np.ndarray:
    ny, dimension = generator.ny, generator.block_dimension
    blocks = np.zeros((ny, dimension, dimension), dtype=np.complex128)
    blocks[:, np.arange(dimension), np.arange(dimension)] = 0.5
    return generator.integrate_q_sector(
        blocks,
        q_index=0,
        dt=0.01,
        observation_times=[0.4],
        include_number_dephasing=dephasing,
    ).states[-1]


def test_q_batch_action_matches_scalar_sectors(
    small_generator: PerfectCorrectionLindblad,
) -> None:
    generator = small_generator
    rng = np.random.default_rng(11)
    values = rng.normal(
        size=(3, 2, generator.ny, generator.block_dimension, generator.block_dimension)
    ) + 1j * rng.normal(
        size=(3, 2, generator.ny, generator.block_dimension, generator.block_dimension)
    )
    q_values = np.asarray([0, 2, 5])
    for dephasing in (False, True):
        batched = generator.q_sectors_homogeneous_rhs(
            values,
            q_indices=q_values,
            include_number_dephasing=dephasing,
        )
        scalar = np.stack(
            [
                generator.q_sector_homogeneous_rhs(
                    values[index],
                    q_index=int(q_index),
                    include_number_dephasing=dephasing,
                )
                for index, q_index in enumerate(q_values)
            ]
        )
        assert np.allclose(batched, scalar, rtol=2e-13, atol=2e-13)


def test_local_kick_q_sector_equals_dense_commutator(
    small_generator: PerfectCorrectionLindblad,
) -> None:
    generator = small_generator
    baseline = _nontrivial_baseline(generator, dephasing=True)
    dense = generator.q_sector_to_dense(baseline, q_index=0)
    source_x, source_y, epsilon = 1, 2, 1e-3
    projector = np.zeros_like(dense)
    for orbital in (0, 1):
        index = orbital + 2 * source_x + 2 * generator.nx * source_y
        projector[index, index] = 1.0
    kicked = 1j * np.sinc(epsilon / np.pi) * (
        projector @ dense - dense @ projector
    )
    momentum = dense_to_y_momentum(kicked, nx=generator.nx, ny=generator.ny)
    for q_index in range(generator.ny):
        expected = extract_q_sector(momentum, q_index)
        observed = local_kick_q_sector(
            baseline,
            q_index=q_index,
            source_x=[source_x],
            source_y=[source_y],
            nx=generator.nx,
            epsilon=epsilon,
        )[0]
        assert np.allclose(observed, expected, rtol=5e-13, atol=5e-13)


@pytest.mark.parametrize("dephasing", [False, True])
def test_q_response_matches_dense_4x6(
    small_generator: PerfectCorrectionLindblad, dephasing: bool
) -> None:
    generator = small_generator
    baseline = _nontrivial_baseline(generator, dephasing=dephasing)
    times = np.asarray([0.0, 0.05, 0.10])
    response = local_density_kick_response(
        generator,
        baseline,
        walls=[1],
        source_ys=[0],
        epsilon=1e-3,
        times=times,
        integration_dt=0.005,
        include_number_dephasing=dephasing,
        wall_window_columns=1,
        fit_time_min=0.0,
        fit_time_max=0.1,
        algorithm="q_sector_rk4",
        q_batch_size=2,
    )

    dense = generator.q_sector_to_dense(baseline, q_index=0)
    projector = np.zeros_like(dense)
    for orbital in (0, 1):
        index = orbital + 2
        projector[index, index] = 1.0
    initial = 1j * np.sinc(1e-3 / np.pi) * (
        projector @ dense - dense @ projector
    )
    evolved = integrate_rk4(
        lambda value: generator.dense_homogeneous_rhs(
            value, include_number_dephasing=dephasing
        ),
        initial,
        dt=0.005,
        observation_times=times,
        hermitize=True,
    ).states
    dense_density = np.real(np.diagonal(evolved, axis1=-2, axis2=-1)).reshape(
        times.size, generator.ny, generator.nx, 2
    ).sum(axis=-1).transpose(0, 2, 1)
    dense_density[0] = 0.0
    columns = [0, 1, 2]
    expected_profile = np.sum(dense_density[:, columns, :], axis=1)
    observed_profile = response.arrays["response_density_source_wall_time_y"][0, 0]
    assert np.allclose(observed_profile, expected_profile, rtol=2e-9, atol=2e-11)


def test_rank_four_no_dephasing_equals_q_rk4(
    small_generator: PerfectCorrectionLindblad,
) -> None:
    generator = small_generator
    baseline = _nontrivial_baseline(generator, dephasing=False)
    kwargs = dict(
        walls=[1, 2],
        source_ys=[0],
        epsilon=1e-3,
        times=[0.0, 0.1, 0.2],
        integration_dt=0.005,
        include_number_dephasing=False,
        wall_window_columns=1,
        fit_time_min=0.0,
        fit_time_max=0.2,
    )
    exact = local_density_kick_response(
        generator, baseline, algorithm="rank_four_exact", **kwargs
    )
    numerical = local_density_kick_response(
        generator, baseline, algorithm="q_sector_rk4", q_batch_size=2, **kwargs
    )
    assert np.allclose(
        exact.arrays["response_density_source_wall_time_y"],
        numerical.arrays["response_density_source_wall_time_y"],
        rtol=2e-8,
        atol=3e-11,
    )


@pytest.mark.parametrize("dephasing", [False, True])
def test_lindblad_adapter_small_contract(dephasing: bool) -> None:
    case = _case(dephasing=dephasing)
    result = run_lindblad_case(case, _config(response=False))
    assert np.array_equal(result.arrays["cycle"], np.arange(13))
    assert result.arrays["translation_residual"].shape == (1, 13)
    assert result.arrays["successive_state_distance"].shape == (1, 13)
    assert result.arrays["spectral_checkpoint_cycle"].tolist() == [0, 3, 6, 9, 12]
    for name in (
        "occupation_min",
        "occupation_max",
        "half_occupation_gap",
        "gaussian_entropy_proxy_per_circumference",
        "gaussian_charge_variance_proxy",
    ):
        assert result.arrays[name].shape == (1, 5)
        assert np.all(np.isfinite(result.arrays[name]))
    assert result.arrays["G_final"].shape == (1, 48, 48)
    assert result.arrays["G_late_cycle_average"].shape == (1, 48, 48)
    assert np.allclose(
        result.arrays["G_final"],
        result.arrays["G_final"].conj().swapaxes(-1, -2),
    )
    assert result.metadata["sample_seeds"] == [17]
    assert result.metadata["response_enabled"] is False


def test_lindblad_adapter_response_smoke() -> None:
    result = run_lindblad_case(
        _case(dephasing=True, response=True), _config(response=True)
    )
    assert result.metadata["response_enabled"] is True
    assert result.metadata["response"]["response_algorithm"] == "q_sector_rk4"
    assert result.metadata["response"]["response_q_batch_size"] == 2
    assert result.arrays["response_density_source_wall_time_y"].shape == (
        1,
        1,
        2,
        13,
        6,
    )
    assert result.arrays["response_velocity_source_wall"].shape == (1, 1, 2)
