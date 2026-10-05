from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

EXPERIMENT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT.parents[2]
sys.path[:0] = [str(EXPERIMENT), str(REPO_ROOT / "src")]

from fgtn.occupied_frame import OccupiedFrameState
from reference_probe import (
    anisotropy_from_matching_time,
    gaussian_mutual_information,
    gaussian_subsystem_entropy,
    insert_reference_cell,
    insert_reference_mode,
    matching_time,
)


def product_frame(occupations: list[int]) -> OccupiedFrameState:
    dimension = len(occupations)
    occupied = np.flatnonzero(occupations)
    return OccupiedFrameState(
        np.eye(dimension, dtype=np.complex128)[:, occupied],
        representation="physical_frame",
        physical_dimension=dimension,
    )


def test_reference_mode_is_maximally_entangled_for_both_measurement_outcomes():
    for occupation in (0, 1):
        state = product_frame([occupation])
        event = insert_reference_mode(
            state, support_idx=[0], orbital_local=[1.0], rng=np.random.default_rng(3)
        )
        assert event["outcome_occupied"] is bool(occupation)
        correlation = state.physical_frame @ state.physical_frame.conj().T
        np.testing.assert_allclose(np.diag(correlation), [0.5, 0.5], atol=1e-12)
        np.testing.assert_allclose(np.linalg.eigvalsh(correlation), [0.0, 1.0], atol=1e-12)
        np.testing.assert_allclose(
            gaussian_subsystem_entropy(state, [event["reference_row"]]), np.log(2.0), atol=1e-12
        )


def test_two_independent_reference_cells_begin_with_zero_mutual_information():
    state = product_frame([0, 1, 1, 0])
    first = insert_reference_cell(
        state, nx=2, x=0, y=0, system_dimension=4, rng=np.random.default_rng(1)
    )
    second = insert_reference_cell(
        state, nx=2, x=1, y=0, system_dimension=4, rng=np.random.default_rng(2)
    )
    mutual_information, entropy_one, entropy_two, entropy_union = gaussian_mutual_information(
        state, first["reference_rows"], second["reference_rows"]
    )
    np.testing.assert_allclose(mutual_information, 0.0, atol=1e-12)
    np.testing.assert_allclose(entropy_one, 2.0 * np.log(2.0), atol=1e-12)
    np.testing.assert_allclose(entropy_two, 2.0 * np.log(2.0), atol=1e-12)
    np.testing.assert_allclose(entropy_union, 4.0 * np.log(2.0), atol=1e-12)


def test_matching_and_anisotropy_interpolation():
    time_star = matching_time([2, 3, 4], [0.8, 0.5, 0.2], 0.35)
    np.testing.assert_allclose(time_star, 3.5)
    expected = np.log1p(np.sqrt(2.0)) * 8.0 / (np.pi * 3.5)
    np.testing.assert_allclose(anisotropy_from_matching_time(8, time_star), expected)


def test_upward_noise_excursion_does_not_create_a_late_crossing():
    assert np.isnan(matching_time([1, 2, 3], [0.1, 0.5, 0.05], 0.3))
