from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

EXPERIMENT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT.parents[2]
sys.path[:0] = [str(EXPERIMENT), str(REPO_ROOT / "src")]

from fgtn.diagnostics import compute_completion_from_constraints
from flag import analyze_constraint_flag


def labels(count: int) -> dict[str, np.ndarray]:
    return {
        "channel": np.asarray(["Ap"] * count),
        "center_x": np.arange(count),
        "center_y": np.zeros(count, dtype=int),
        "wall_distance": np.arange(count),
    }


def test_compatible_synthetic_flag_has_no_breakdown_and_monotone_cost():
    vectors = np.eye(4, dtype=complex)
    targets = np.asarray([1, 1, 0, 0])
    scan = analyze_constraint_flag(
        vectors,
        targets,
        labels=labels(4),
        target_rank=2,
        ordering=np.arange(4),
        checkpoint_indices=np.arange(1, 5),
    )
    assert scan.first_breakdown is None
    costs = np.asarray([row["f_star"] for row in scan.rows])
    assert np.all(np.diff(costs) >= -1e-12)
    assert costs[-1] < 1e-12


def test_known_nonorthogonal_breakdown_letter():
    vectors = np.column_stack((np.asarray([1, 0, 0, 0]), np.asarray([1, 1, 0, 0]) / np.sqrt(2)))
    scan = analyze_constraint_flag(
        vectors,
        np.asarray([1, 0]),
        labels=labels(2),
        target_rank=2,
        ordering=[0, 1],
        checkpoint_indices=[1, 2],
    )
    assert scan.first_breakdown is not None
    assert scan.first_breakdown["prefix"] == 2
    assert scan.first_breakdown["reason"] == "nonorthogonality"


def test_final_metrics_match_existing_solver_and_are_order_invariant():
    rng = np.random.default_rng(9)
    vectors = rng.normal(size=(6, 8)) + 1j * rng.normal(size=(6, 8))
    targets = np.asarray([0, 1] * 4)
    reference = compute_completion_from_constraints(vectors, targets, target_rank=3)
    finals = []
    for order in (np.arange(8), np.asarray([7, 0, 4, 3, 1, 6, 2, 5])):
        scan = analyze_constraint_flag(
            vectors,
            targets,
            labels=labels(8),
            target_rank=3,
            ordering=order,
            checkpoint_indices=[4, 8],
        )
        finals.append(scan.rows[-1])
    for final in finals:
        np.testing.assert_allclose(final["f_star"], reference.f_star, atol=1e-11)
        np.testing.assert_allclose(final["sigma_max"], reference.sigma_max, atol=1e-11)
        np.testing.assert_allclose(final["overlap_phi"], reference.overlap_phi, atol=1e-11)
    np.testing.assert_allclose(finals[0]["f_star"], finals[1]["f_star"], atol=1e-12)
