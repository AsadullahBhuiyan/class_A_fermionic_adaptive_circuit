from __future__ import annotations

import itertools
import math
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SHARED = (
    ROOT
    / "00_WORKSPACE/CURRENT/final_production_ready_figure_scripts/_shared_src"
)
sys.path.insert(0, str(SHARED))

from g5_manybody_spectrum_analysis import (  # noqa: E402
    G5_SECTORS,
    fit_casimir,
    fit_tower_means,
    lowest_charge_resolved_levels,
    spectrum_factors,
)


def _binary_products(occupations: np.ndarray) -> np.ndarray:
    values = []
    for bits in itertools.product((0, 1), repeat=len(occupations)):
        value = 1.0
        for bit, nu in zip(bits, occupations):
            value *= nu if bit else 1.0 - nu
        values.append(value)
    return np.sort(np.asarray(values))


def _diagonal_fock_singular_values(one_particle_values: np.ndarray) -> np.ndarray:
    values = []
    for bits in itertools.product((0, 1), repeat=len(one_particle_values)):
        value = 1.0
        for bit, singular_value in zip(bits, one_particle_values):
            if bit:
                value *= singular_value
        values.append(value)
    return np.sort(np.asarray(values))


def _fermion_operators(modes: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
    dimension = 1 << modes
    annihilation, creation = [], []
    for mode in range(modes):
        operator = np.zeros((dimension, dimension), dtype=np.complex128)
        for state in range(dimension):
            if (state >> mode) & 1:
                target = state & ~(1 << mode)
                parity = (state & ((1 << mode) - 1)).bit_count()
                operator[target, state] = -1.0 if parity % 2 else 1.0
        annihilation.append(operator)
        creation.append(operator.conj().T)
    return annihilation, creation


def _correlation_from_density(
    density: np.ndarray, annihilation: list[np.ndarray], creation: list[np.ndarray]
) -> np.ndarray:
    modes = len(annihilation)
    return np.asarray(
        [
            [np.trace(density @ creation[j] @ annihilation[i]).real for j in range(modes)]
            for i in range(modes)
        ]
    )


def test_invertible_gaussian_fock_spectrum_matches_factorization() -> None:
    one_particle = np.asarray([0.31, 0.8, 1.7, 4.2])
    direct_singular = _diagonal_fock_singular_values(one_particle)
    z_k = float(np.sum(direct_singular**2))
    occupations = one_particle**2 / (1.0 + one_particle**2)
    factors = spectrum_factors(occupations)
    reconstructed_density = _binary_products(occupations)
    np.testing.assert_allclose(
        reconstructed_density, np.sort(direct_singular**2 / z_k), rtol=2e-14, atol=2e-14
    )
    np.testing.assert_allclose(
        np.sqrt(z_k * reconstructed_density), direct_singular, rtol=2e-14, atol=2e-14
    )
    np.testing.assert_allclose(
        factors["j"], 2.0 * one_particle / (1.0 + one_particle**2), rtol=2e-14
    )
    np.testing.assert_allclose(
        factors["density_gap"], 2.0 * factors["amplitude_gap"], rtol=2e-14
    )
    np.testing.assert_allclose(np.sum(reconstructed_density), 1.0, atol=2e-14)
    np.testing.assert_allclose(
        math.log(z_k), np.sum(np.log1p(one_particle**2)), atol=2e-14
    )


def test_exact_gain_loss_caps_and_orientation() -> None:
    annihilation, creation = _fermion_operators(2)
    for word, expected in (
        (creation[0], np.asarray([0.5, 1.0])),
        (annihilation[0], np.asarray([0.0, 0.5])),
        (creation[0] @ annihilation[1], np.asarray([0.0, 1.0])),
    ):
        density = word @ word.conj().T
        density /= np.trace(density)
        correlation = _correlation_from_density(density, annihilation, creation)
        occupations = np.linalg.eigvalsh(correlation)
        factors = spectrum_factors(occupations)
        direct_eigenvalues = np.sort(np.linalg.eigvalsh(density))
        np.testing.assert_allclose(_binary_products(occupations), direct_eigenvalues, atol=2e-14)
        assert np.any(np.abs(factors["cap_orientation"]) == 1)
        np.testing.assert_allclose(np.sort(occupations), expected, atol=2e-14)


def test_rank_deficient_projector_and_mandatory_caps() -> None:
    occupations = np.asarray([0.0, 0.2, 0.8, 1.0])
    factors = spectrum_factors(occupations)
    np.testing.assert_array_equal(factors["cap_orientation"], [-1, 0, 0, 1])
    np.testing.assert_array_equal(factors["dominant_occupation"], [0, 0, 1, 1])
    assert np.isinf(factors["amplitude_gap"][[0, 3]]).all()
    np.testing.assert_allclose(factors["mu_minus"] + factors["mu_plus"], 1.0)


def test_charge_sector_heap_matches_bruteforce() -> None:
    gaps = np.asarray([0.1, 0.3, 0.8, 1.2, 1.7, 2.4])
    charges = np.asarray([1, -1, 1, -1, 1, -1])
    levels, counts, _ = lowest_charge_resolved_levels(
        gaps, charges, levels_per_sector=4
    )
    brute = {int(q): [] for q in G5_SECTORS}
    for bits in itertools.product((0, 1), repeat=gaps.size):
        bits_array = np.asarray(bits)
        charge = int(np.sum(bits_array * charges))
        if charge in brute:
            brute[charge].append(float(np.sum(bits_array * gaps)))
    for index, sector in enumerate(G5_SECTORS):
        expected = np.sort(brute[int(sector)])[:4]
        np.testing.assert_allclose(levels[index, : expected.size], expected, atol=1e-14)
        assert counts[index] == min(4, len(expected))


def test_synthetic_conformal_tower_recovers_velocity_and_level() -> None:
    sizes = np.asarray([20.0, 30.0, 40.0, 50.0, 60.0])
    velocity, level = 1.75, 1.0
    neutral = 2.0 * math.pi * velocity / sizes
    charged = neutral / (2.0 * level)
    fit = fit_tower_means(sizes, neutral, charged, charged)
    np.testing.assert_allclose(fit["v"], velocity, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(fit["k"], level, rtol=1e-13, atol=1e-13)
    assert fit["neutral_relative_rms"] < 1e-13
    assert fit["charged_relative_rms"] < 1e-13

    with pytest.raises(ValueError, match="positive finite-size points"):
        fit_tower_means(sizes, neutral, charged, np.full_like(charged, np.inf))


def test_synthetic_casimir_fit_obeys_one_wall_two_wall_normalization() -> None:
    sizes = np.asarray([20.0, 30.0, 40.0, 50.0, 60.0])
    velocity, central_charge, background = 1.4, 1.0, 0.23
    per_wall = background - math.pi * central_charge * velocity / (12.0 * sizes**2)
    one_wall = fit_casimir(sizes, per_wall, velocity=velocity, wall_count=1)
    two_wall = fit_casimir(sizes, 2.0 * per_wall, velocity=velocity, wall_count=2)
    np.testing.assert_allclose(one_wall["c_eff"], central_charge, atol=2e-12)
    np.testing.assert_allclose(two_wall["c_eff"], central_charge, atol=2e-12)
    np.testing.assert_allclose(one_wall["slope"], two_wall["slope"], atol=2e-14)


def test_g5_source_is_descendant_only_and_never_requests_choi() -> None:
    source = (SHARED / "g5_manybody_spectrum_analysis.py").read_text(encoding="utf-8")
    assert '"track_choi": False' in source
    assert '"campaign") != "S1_T1_B2_MASTER"' in source
    assert "R1_RECORD_SPECTRUM" not in source
    assert "itertools.product" not in source
