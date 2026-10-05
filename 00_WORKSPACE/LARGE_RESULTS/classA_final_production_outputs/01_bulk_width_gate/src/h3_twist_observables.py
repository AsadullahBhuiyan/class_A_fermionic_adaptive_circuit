from __future__ import annotations

from typing import Any

import numpy as np


H3_SCHEMA = "h3_frozen_record_entanglement_flow_v1"


def half_torus_indices(nx: int, ny: int, *, y0: int = 0) -> np.ndarray:
    return np.asarray(
        [
            orbital + 2 * x + 2 * nx * ((y0 + yrel) % ny)
            for yrel in range(ny // 2)
            for x in range(nx)
            for orbital in (0, 1)
        ],
        dtype=np.int64,
    )


def wall_locations(nx: int) -> tuple[int, int]:
    half = int(nx) // 2
    width = max(1, int(nx) // 4)
    return max(0, half - width), min(int(nx), half + width + 1) - 1


class BranchWeightObserver:
    def __init__(self, *, samples: int, cycles: int) -> None:
        self.samples, self.cycles = int(samples), int(cycles)
        self.log_probability_per_cycle: Any = None
        self._minimum_probability: Any = None

    def _materialize_cpu(self) -> None:
        if hasattr(self.log_probability_per_cycle, "detach"):
            self.log_probability_per_cycle = (
                self.log_probability_per_cycle.detach().cpu().numpy()
            )
            self._minimum_probability = (
                self._minimum_probability.detach().cpu().numpy()
            )

    def __call__(
        self,
        *,
        cycle: int,
        sample_indices: Any,
        channel_labels: tuple[str, ...],
        conditional_log_probability: Any,
        realized_probability: Any,
        **_: Any,
    ) -> None:
        if hasattr(conditional_log_probability, "detach"):
            import torch

            device = conditional_log_probability.device
            if self.log_probability_per_cycle is None:
                self.log_probability_per_cycle = torch.zeros(
                    (self.samples, self.cycles),
                    dtype=torch.float64,
                    device=device,
                )
                self._minimum_probability = torch.ones(
                    (self.samples,), dtype=torch.float64, device=device
                )
            indices = sample_indices.to(dtype=torch.long, device=device)
            logs = conditional_log_probability[:, : len(channel_labels)]
            probabilities = realized_probability[:, : len(channel_labels)]
            self.log_probability_per_cycle[indices, int(cycle) - 1] += torch.sum(
                logs, dim=1
            )
            self._minimum_probability[indices] = torch.minimum(
                self._minimum_probability[indices],
                torch.min(probabilities, dim=1).values,
            )
            return

        if self.log_probability_per_cycle is None:
            self.log_probability_per_cycle = np.zeros(
                (self.samples, self.cycles), dtype=np.float64
            )
            self._minimum_probability = np.ones(
                (self.samples,), dtype=np.float64
            )
        indices = np.asarray(sample_indices, dtype=np.int64)
        logs = np.asarray(conditional_log_probability)[:, : len(channel_labels)]
        probabilities = np.asarray(realized_probability)[:, : len(channel_labels)]
        self.log_probability_per_cycle[indices, int(cycle) - 1] += np.sum(
            logs, axis=1
        )
        self._minimum_probability[indices] = np.minimum(
            self._minimum_probability[indices], np.min(probabilities, axis=1)
        )

    @property
    def total_log_probability(self) -> np.ndarray:
        self._materialize_cpu()
        return np.sum(self.log_probability_per_cycle, axis=1)

    @property
    def minimum_probability(self) -> np.ndarray:
        self._materialize_cpu()
        return self._minimum_probability


class FinalEntanglementObserver:
    def __init__(
        self,
        *,
        samples: int,
        nx: int,
        ny: int,
        final_cycle: int,
        tracked_modes: int,
        wall_half_width: int,
    ) -> None:
        self.samples, self.nx, self.ny = int(samples), int(nx), int(ny)
        self.final_cycle = int(final_cycle)
        self.tracked_modes = int(tracked_modes)
        self.wall_half_width = int(wall_half_width)
        self.indices = half_torus_indices(nx, ny)
        self.values: np.ndarray | None = None
        self.vectors: np.ndarray | None = None
        self.wall_weights: np.ndarray | None = None
        self.reference_gap: np.ndarray | None = None

    def __call__(self, *, cycle: int, G: Any, batch_start: int, batch_count: int, **_: Any) -> None:
        if int(cycle) != self.final_cycle:
            return
        import torch

        idx = torch.as_tensor(self.indices, dtype=torch.long, device=G.device)
        restricted_g = G.index_select(1, idx).index_select(2, idx)
        eye = torch.eye(idx.numel(), dtype=G.dtype, device=G.device)
        C = 0.5 * (restricted_g + eye[None])
        C = 0.5 * (C + C.conj().transpose(-2, -1))
        values, vectors = torch.linalg.eigh(C)
        order = torch.argsort(torch.abs(values - 0.5), dim=1)[:, : self.tracked_modes]
        selected_values = torch.gather(values.real, 1, order)
        selected_vectors = torch.gather(
            vectors, 2, order[:, None, :].expand(-1, vectors.shape[1], -1)
        )
        mode_x = (idx // 2) % self.nx
        walls = wall_locations(self.nx)
        weights = []
        for center in walls:
            mask = torch.as_tensor(
                [
                    min((int(x) - center) % self.nx, (center - int(x)) % self.nx)
                    <= self.wall_half_width
                    for x in mode_x.detach().cpu().tolist()
                ],
                dtype=torch.bool,
                device=G.device,
            )
            weights.append(selected_vectors[:, mask].abs().square().sum(dim=1).real)
        start, stop = int(batch_start), int(batch_start) + int(batch_count)
        if self.values is None:
            dim = int(idx.numel())
            self.values = np.full((self.samples, self.tracked_modes), np.nan, dtype=np.float64)
            self.vectors = np.empty((self.samples, dim, self.tracked_modes), dtype=np.complex128)
            self.wall_weights = np.full((self.samples, self.tracked_modes, 2), np.nan, dtype=np.float64)
            self.reference_gap = np.full((self.samples,), np.nan, dtype=np.float64)
        self.values[start:stop] = selected_values.detach().cpu().numpy()
        self.vectors[start:stop] = selected_vectors.detach().cpu().numpy()
        self.wall_weights[start:stop] = torch.stack(weights, dim=-1).detach().cpu().numpy()
        self.reference_gap[start:stop] = torch.min(torch.abs(values.real - 0.5), dim=1).values.detach().cpu().numpy()

    def payload(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if any(value is None for value in (self.values, self.vectors, self.wall_weights, self.reference_gap)):
            raise RuntimeError("final H3 entanglement observer was not emitted")
        return self.values, self.vectors, self.wall_weights, self.reference_gap


def initial_order(values: np.ndarray) -> np.ndarray:
    eps = np.log(np.clip(1.0 - values, 1e-14, 1.0) / np.clip(values, 1e-14, 1.0))
    return np.argsort(eps, axis=1)


def track_step(
    previous_vectors: np.ndarray,
    current_values: np.ndarray,
    current_vectors: np.ndarray,
    current_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from scipy.optimize import linear_sum_assignment

    samples, _, modes = previous_vectors.shape
    values = np.empty_like(current_values)
    vectors = np.empty_like(current_vectors)
    weights = np.empty_like(current_weights)
    overlap_matrices = np.empty((samples, modes, modes), dtype=np.complex128)
    assignments = np.empty((samples, modes), dtype=np.int16)
    for sample in range(samples):
        overlap = previous_vectors[sample].conj().T @ current_vectors[sample]
        rows, cols = linear_sum_assignment(-np.abs(overlap) ** 2)
        assignment = np.empty((modes,), dtype=np.int64)
        assignment[rows] = cols
        tracked_vectors = current_vectors[sample][:, assignment]
        diagonal = np.einsum("ik,ik->k", previous_vectors[sample].conj(), tracked_vectors)
        phase = np.ones_like(diagonal)
        nonzero = np.abs(diagonal) > 1e-14
        phase[nonzero] = diagonal[nonzero].conj() / np.abs(diagonal[nonzero])
        tracked_vectors *= phase[None, :]
        values[sample] = current_values[sample, assignment]
        vectors[sample] = tracked_vectors
        weights[sample] = current_weights[sample][assignment]
        overlap_matrices[sample] = overlap
        assignments[sample] = assignment.astype(np.int16)
    return values, vectors, weights, overlap_matrices, assignments


def crossing_summary(
    entanglement_energies: np.ndarray, wall_weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    samples, points, modes = entanglement_energies.shape
    counts = np.zeros((samples, 2), dtype=np.int16)
    ambiguous = np.zeros((samples,), dtype=np.int16)
    for sample in range(samples):
        for point in range(points - 1):
            left, right = entanglement_energies[sample, point], entanglement_energies[sample, point + 1]
            crossing = ((left < 0) & (right >= 0)) | ((left > 0) & (right <= 0))
            for mode in np.flatnonzero(crossing):
                weight = 0.5 * (
                    wall_weights[sample, point, mode] + wall_weights[sample, point + 1, mode]
                )
                if np.isclose(weight[0], weight[1], rtol=0.0, atol=1e-3):
                    ambiguous[sample] += 1
                    continue
                wall = int(np.argmax(weight))
                counts[sample, wall] += 1 if right[mode] > left[mode] else -1
    return counts, ambiguous
