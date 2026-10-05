from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")


SHARED = (
    Path(__file__).resolve().parents[1]
    / "00_WORKSPACE"
    / "CURRENT"
    / "final_production_ready_figure_scripts"
    / "_shared_src"
)
sys.path.insert(0, str(SHARED))

from selected_observables import SelectedCovarianceObserver  # noqa: E402


def test_successive_covariance_frobenius_is_dimension_normalized(
    tmp_path: Path,
) -> None:
    observer = SelectedCovarianceObserver(
        nx=1,
        ny=1,
        samples=2,
        physical_cycles=2,
        observation_cycles=[],
        compute_correlator=False,
    )
    initial = torch.zeros((2, 2, 2), dtype=torch.complex128)
    cycle_1 = initial.clone()
    cycle_1[0] = torch.eye(2, dtype=torch.complex128)
    cycle_1[1] = 2.0 * torch.eye(2, dtype=torch.complex128)
    cycle_2 = cycle_1 + 3.0 * torch.eye(2, dtype=torch.complex128)[None]

    observer(cycle=0, G=initial, batch_start=0, batch_count=2)
    observer(cycle=1, G=cycle_1, batch_start=0, batch_count=2)
    observer(cycle=2, G=cycle_2, batch_start=0, batch_count=2)

    expected = np.asarray(
        [
            [np.sqrt(2.0) / 2.0, 3.0 * np.sqrt(2.0) / 2.0],
            [2.0 * np.sqrt(2.0) / 2.0, 3.0 * np.sqrt(2.0) / 2.0],
        ]
    )
    np.testing.assert_allclose(
        observer.successive_covariance_frobenius_per_dimension, expected
    )
    assert observer._previous_covariance is None

    path = tmp_path / "selected.npz"
    result = observer.save(path, config={})
    assert result["convergence_normalization_dimension"] == 2
    with np.load(path, allow_pickle=False) as data:
        np.testing.assert_array_equal(data["convergence_cycles"], [1, 2])
        np.testing.assert_allclose(
            data["successive_covariance_frobenius_per_dimension"], expected
        )


def test_successive_covariance_diagnostic_requires_contiguous_cycles() -> None:
    observer = SelectedCovarianceObserver(
        nx=1,
        ny=1,
        samples=1,
        physical_cycles=2,
        observation_cycles=[],
        compute_correlator=False,
    )
    covariance = torch.zeros((1, 2, 2), dtype=torch.complex128)
    observer(cycle=0, G=covariance, batch_start=0, batch_count=1)
    with pytest.raises(RuntimeError, match="cycle-contiguous"):
        observer(cycle=2, G=covariance, batch_start=0, batch_count=1)
