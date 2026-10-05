from __future__ import annotations

import numpy as np
import pytest

from fgtn.classA_U1FGTN import classA_U1FGTN
from fgtn.diagnostics.mean_lindblad import (
    FAMILY_NAMES,
    PerfectCorrectionLindblad,
    dense_to_y_momentum,
    integrate_rk4,
    linear_gain_rhs,
    linear_loss_rhs,
    number_dephasing_rhs,
    y_momentum_to_dense,
)


def _random_matrix(rng: np.random.Generator, dimension: int) -> np.ndarray:
    return rng.normal(size=(dimension, dimension)) + 1j * rng.normal(
        size=(dimension, dimension)
    )


def _random_hermitian(rng: np.random.Generator, dimension: int) -> np.ndarray:
    raw = _random_matrix(rng, dimension)
    return 0.5 * (raw + raw.conj().T)


@pytest.fixture(scope="module")
def canonical_pair() -> tuple[classA_U1FGTN, PerfectCorrectionLindblad]:
    model = classA_U1FGTN(
        Nx=4,
        Ny=4,
        DW=True,
        nshell=1,
        alpha_1=1,
        alpha_2=30,
        trial_orbitals="X",
        dw_truncation=True,
    )
    engine = PerfectCorrectionLindblad.from_canonical_model(model)
    return model, engine


def test_single_mode_decomposition_gives_exact_reset_increment() -> None:
    rng = np.random.default_rng(12)
    mode = rng.normal(size=5) + 1j * rng.normal(size=5)
    mode = mode / np.linalg.norm(mode)
    projector = np.outer(mode, mode.conj())
    complement = np.eye(5) - projector
    correlation = _random_hermitian(rng, 5)

    empty_increment = linear_loss_rhs(correlation, mode) + number_dephasing_rhs(
        correlation, mode
    )
    filled_increment = linear_gain_rhs(correlation, mode) + number_dephasing_rhs(
        correlation, mode
    )

    np.testing.assert_allclose(
        correlation + empty_increment,
        complement @ correlation @ complement,
        rtol=2e-13,
        atol=2e-13,
    )
    np.testing.assert_allclose(
        correlation + filled_increment,
        complement @ correlation @ complement + projector,
        rtol=2e-13,
        atol=2e-13,
    )


@pytest.mark.parametrize("include_number_dephasing", [False, True])
def test_dense_engine_matches_explicit_mode_sum(
    canonical_pair: tuple[classA_U1FGTN, PerfectCorrectionLindblad],
    include_number_dephasing: bool,
) -> None:
    _model, engine = canonical_pair
    rng = np.random.default_rng(21)
    correlation = _random_hermitian(rng, engine.dimension)
    expected = np.zeros_like(correlation)
    for family in ("A_minus", "B_minus"):
        for mode in engine.frames[family].reshape(engine.dimension, -1).T:
            expected += linear_gain_rhs(correlation, mode)
            if include_number_dephasing:
                expected += number_dephasing_rhs(correlation, mode)
    for family in ("A_plus", "B_plus"):
        for mode in engine.frames[family].reshape(engine.dimension, -1).T:
            expected += linear_loss_rhs(correlation, mode)
            if include_number_dephasing:
                expected += number_dephasing_rhs(correlation, mode)

    actual = engine.dense_rhs(
        correlation,
        include_number_dephasing=include_number_dephasing,
    )
    np.testing.assert_allclose(actual, expected, rtol=5e-12, atol=5e-12)
    np.testing.assert_allclose(actual, actual.conj().T, rtol=0.0, atol=2e-12)


def test_dephasing_toggle_changes_the_dense_generator(
    canonical_pair: tuple[classA_U1FGTN, PerfectCorrectionLindblad],
) -> None:
    _model, engine = canonical_pair
    rng = np.random.default_rng(22)
    correlation = _random_hermitian(rng, engine.dimension)
    without = engine.dense_rhs(
        correlation, include_number_dephasing=False
    )
    with_dephasing = engine.dense_rhs(
        correlation, include_number_dephasing=True
    )
    assert np.linalg.norm(with_dephasing - without) > 1e-8


def test_engine_consumes_canonical_projectors_and_preserves_translation(
    canonical_pair: tuple[classA_U1FGTN, PerfectCorrectionLindblad],
) -> None:
    model, engine = canonical_pair
    for family, attribute in {
        "A_minus": "WF_Am",
        "B_minus": "WF_Bm",
        "A_plus": "WF_Ap",
        "B_plus": "WF_Bp",
    }.items():
        np.testing.assert_array_equal(engine.frames[family], getattr(model, attribute))
        assert engine.frame_norm_error[family] < 1e-12
    assert engine.y_translation_residual < 1e-12

    rng = np.random.default_rng(23)
    blocks = np.stack(
        [_random_hermitian(rng, engine.block_dimension) for _ in range(engine.ny)]
    )
    dense = engine.q_sector_to_dense(blocks, q_index=0)
    derivative = engine.dense_rhs(
        dense, include_number_dephasing=True
    )
    momentum_derivative = dense_to_y_momentum(
        derivative, nx=engine.nx, ny=engine.ny
    )
    off_diagonal = momentum_derivative.copy()
    for k in range(engine.ny):
        off_diagonal[k, :, k, :] = 0.0
    assert np.linalg.norm(off_diagonal) < 2e-10


def test_y_momentum_transform_round_trip() -> None:
    rng = np.random.default_rng(31)
    nx, ny = 3, 5
    dimension = 2 * nx * ny
    matrix = _random_matrix(rng, dimension)
    transformed = dense_to_y_momentum(matrix, nx=nx, ny=ny)
    reconstructed = y_momentum_to_dense(transformed, nx=nx, ny=ny)
    np.testing.assert_allclose(reconstructed, matrix, rtol=2e-13, atol=2e-13)


@pytest.mark.parametrize("include_number_dephasing", [False, True])
@pytest.mark.parametrize("q_index", [0, 1])
def test_q_sector_action_matches_dense_reference(
    canonical_pair: tuple[classA_U1FGTN, PerfectCorrectionLindblad],
    include_number_dephasing: bool,
    q_index: int,
) -> None:
    _model, engine = canonical_pair
    rng = np.random.default_rng(40 + 2 * q_index + int(include_number_dephasing))
    if q_index == 0:
        sector = np.stack(
            [
                _random_hermitian(rng, engine.block_dimension)
                for _ in range(engine.ny)
            ]
        )
        dense = engine.q_sector_to_dense(sector, q_index=0)
        dense_action = engine.dense_rhs(
            dense,
            include_number_dephasing=include_number_dephasing,
        )
        sector_action = engine.q_sector_rhs(
            sector,
            q_index=0,
            include_number_dephasing=include_number_dephasing,
        )
    else:
        sector = np.stack(
            [
                _random_matrix(rng, engine.block_dimension)
                for _ in range(engine.ny)
            ]
        )
        dense = engine.q_sector_to_dense(sector, q_index=q_index)
        dense_action = engine.dense_homogeneous_rhs(
            dense,
            include_number_dephasing=include_number_dephasing,
        )
        sector_action = engine.q_sector_homogeneous_rhs(
            sector,
            q_index=q_index,
            include_number_dephasing=include_number_dephasing,
        )

    dense_sector_action = engine.dense_q_sector(
        dense_action, q_index=q_index
    )
    np.testing.assert_allclose(
        sector_action,
        dense_sector_action,
        rtol=2e-11,
        atol=2e-11,
    )


def test_rk4_has_fourth_order_dt_convergence() -> None:
    mode = np.array([1.0, 0.0], dtype=np.complex128)
    initial = np.array([[0.6, 0.2], [0.2, 0.4]], dtype=np.complex128)
    time = 1.0
    attenuation = np.diag([np.exp(-0.5 * time), 1.0])
    exact = attenuation @ initial @ attenuation

    coarse = integrate_rk4(
        lambda matrix: linear_loss_rhs(matrix, mode),
        initial,
        dt=0.2,
        observation_times=[0.0, time],
        hermitize=True,
    ).states[-1]
    fine = integrate_rk4(
        lambda matrix: linear_loss_rhs(matrix, mode),
        initial,
        dt=0.1,
        observation_times=[0.0, time],
        hermitize=True,
    ).states[-1]
    coarse_error = np.linalg.norm(coarse - exact)
    fine_error = np.linalg.norm(fine - exact)
    assert fine_error < coarse_error / 12.0


def test_public_family_contract_is_complete() -> None:
    assert FAMILY_NAMES == ("A_minus", "B_minus", "A_plus", "B_plus")
