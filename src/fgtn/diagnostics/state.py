from __future__ import annotations

from typing import Any

import numpy as np


class StateObservableRecorder:
    """Stream compact equilibration checks from the active covariance block."""

    def __init__(self, *, samples: int, cycles: int, active_indices: np.ndarray) -> None:
        self.samples = int(samples)
        self.cycles = int(cycles)
        self.active_indices = np.asarray(active_indices, dtype=np.int64).reshape(-1)
        shape = (self.samples, self.cycles + 1)
        self.total_charge = np.full(shape, np.nan, dtype=np.float64)
        self.charge_variance = np.full(shape, np.nan, dtype=np.float64)
        self.entropy = np.full(shape, np.nan, dtype=np.float64)
        self.purity_defect = np.full(shape, np.nan, dtype=np.float64)
        self.successive_delta = np.full(shape, np.nan, dtype=np.float64)
        self._previous: list[np.ndarray | None] = [None] * self.samples

    def __call__(
        self,
        *,
        cycle: int,
        G: np.ndarray,
        batch_start: int,
        batch_count: int,
        **_: Any,
    ) -> None:
        if int(batch_count) != 1:
            raise ValueError("The CPU state observer expects serial trajectories.")
        sample = int(batch_start)
        matrix = np.asarray(G, dtype=np.complex128)
        if matrix.ndim == 3:
            matrix = matrix[0]
        active = matrix[np.ix_(self.active_indices, self.active_indices)]
        active = 0.5 * (active + active.conj().T)
        dimension = active.shape[0]
        C = 0.5 * (active + np.eye(dimension, dtype=np.complex128))
        eigenvalues = np.clip(np.linalg.eigvalsh(C).real, 0.0, 1.0)
        interior = (eigenvalues > 1e-15) & (eigenvalues < 1.0 - 1e-15)
        entropy = -np.sum(
            eigenvalues[interior] * np.log(eigenvalues[interior])
            + (1.0 - eigenvalues[interior]) * np.log1p(-eigenvalues[interior])
        )
        self.total_charge[sample, int(cycle)] = float(np.trace(C).real)
        self.charge_variance[sample, int(cycle)] = float(np.trace(C - C @ C).real)
        self.entropy[sample, int(cycle)] = float(entropy)
        self.purity_defect[sample, int(cycle)] = float(
            np.linalg.norm(active @ active - np.eye(dimension), ord="fro") / max(1, dimension)
        )
        previous = self._previous[sample]
        if previous is not None:
            self.successive_delta[sample, int(cycle)] = float(
                np.linalg.norm(active - previous, ord="fro") / np.sqrt(max(1, dimension))
            )
        self._previous[sample] = active.copy()

    def payload(self) -> dict[str, np.ndarray]:
        return {
            "state_total_charge": self.total_charge,
            "state_charge_variance": self.charge_variance,
            "state_entropy": self.entropy,
            "state_purity_defect": self.purity_defect,
            "state_successive_delta": self.successive_delta,
            "state_active_indices": self.active_indices,
        }
