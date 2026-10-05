from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


SHARED = (
    Path(__file__).resolve().parents[1]
    / "00_WORKSPACE"
    / "CURRENT"
    / "final_production_ready_figure_scripts"
    / "_shared_src"
)
sys.path.insert(0, str(SHARED))

from fused_chirality_observables import _wall_retention_time_series  # noqa: E402


def test_h1_wall_retention_preserves_modular_time_axis() -> None:
    time_points = 161
    probability = np.arange(
        2 * time_points * 10 * 20 * 4, dtype=np.float64
    ).reshape(2, time_points, 10, 20, 4)
    wall_mask = np.zeros(20, dtype=bool)
    wall_mask[[19, 0, 1]] = True

    actual = _wall_retention_time_series(
        probability,
        sample=1,
        packet_index=3,
        wall_mask=wall_mask,
    )
    expected = probability[1, :, :, :, 3][:, :, wall_mask].sum(axis=(1, 2))

    assert actual.shape == (time_points,)
    np.testing.assert_array_equal(actual, expected)


def test_h1_wall_retention_rejects_wrong_mask_geometry() -> None:
    probability = np.zeros((1, 9, 4, 8, 4), dtype=np.float64)
    with np.testing.assert_raises_regex(ValueError, "wall mask must match the x axis"):
        _wall_retention_time_series(
            probability,
            sample=0,
            packet_index=0,
            wall_mask=np.ones(3, dtype=bool),
        )
