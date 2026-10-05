"""Checkpoint-only ambient log-Gram eigensystem extraction."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch


N_MODES = 16
BLOCK_MIXED = -1
BLOCK_OCCUPIED = 0
BLOCK_EMPTY = 1


def checkpoint_cycles(ny: int) -> list[int]:
    values = np.rint(np.linspace(int(ny), 2 * int(ny), 6)).astype(np.int64)
    result = values.tolist()
    if len(result) != 6 or result != sorted(set(result)):
        raise ValueError(f"invalid six-checkpoint schedule for Ny={ny}: {result}")
    return result


def _phase_fix(vectors: np.ndarray) -> np.ndarray:
    fixed = np.asarray(vectors, dtype=np.complex128).copy()
    for column in range(fixed.shape[1]):
        vector = fixed[:, column]
        pivot = int(np.argmax(np.abs(vector)))
        amplitude = vector[pivot]
        if abs(amplitude) > 0.0:
            fixed[:, column] *= np.exp(-1j * np.angle(amplitude))
        if fixed[pivot, column].real < 0.0:
            fixed[:, column] *= -1.0
    return fixed


def _degenerate_groups(values: np.ndarray, labels: np.ndarray, tolerance: float):
    count = len(values)
    unseen = set(range(count))
    groups = []
    while unseen:
        seed = min(unseen)
        group = {seed}
        changed = True
        while changed:
            changed = False
            for candidate in tuple(unseen - group):
                if labels[candidate] != labels[seed]:
                    continue
                if any(
                    abs(values[candidate] - values[index])
                    <= tolerance * (1.0 + max(abs(values[candidate]), abs(values[index])))
                    for index in group
                ):
                    group.add(candidate)
                    changed = True
        unseen.difference_update(group)
        groups.append(sorted(group))
    return groups


@dataclass
class EigenbasisTransport:
    degeneracy_tolerance: float = 64.0 * np.finfo(np.float64).eps
    previous_values: np.ndarray | None = None
    previous_vectors: np.ndarray | None = None
    previous_labels: np.ndarray | None = None
    previous_clusters: np.ndarray | None = None
    next_cluster_id: int = 0

    def align(self, values, vectors, labels):
        values = np.asarray(values, dtype=np.float64)
        vectors = np.asarray(vectors, dtype=np.complex128).copy()
        labels = np.asarray(labels, dtype=np.int8)
        clusters = np.full(values.shape, -1, dtype=np.int32)
        coordinate = np.linspace(-1.0, 1.0, vectors.shape[0], dtype=np.float64)

        for group in _degenerate_groups(
            values, labels, float(self.degeneracy_tolerance)
        ):
            current = vectors[:, group]
            cluster_id = self.next_cluster_id
            prior_indices = []
            transported = False
            if self.previous_vectors is not None:
                candidates = np.flatnonzero(self.previous_labels == labels[group[0]])
                if candidates.size >= len(group):
                    overlap = np.abs(
                        self.previous_vectors[:, candidates].conj().T @ current
                    )
                    score = np.sum(overlap**2, axis=1)
                    chosen = np.argsort(score)[-len(group) :]
                    prior_indices = candidates[chosen].tolist()
            if len(group) > 1:
                if len(prior_indices) == len(group):
                    previous = self.previous_vectors[:, prior_indices]
                    u, _, vh = np.linalg.svd(
                        current.conj().T @ previous, full_matrices=False
                    )
                    current = current @ (u @ vh)
                    transported = True
                    prior_cluster_ids = self.previous_clusters[prior_indices]
                    if np.all(prior_cluster_ids == prior_cluster_ids[0]):
                        cluster_id = int(prior_cluster_ids[0])
                else:
                    tie_breaker = current.conj().T @ (coordinate[:, None] * current)
                    _, rotation = np.linalg.eigh(
                        0.5 * (tie_breaker + tie_breaker.conj().T)
                    )
                    current = current @ rotation
            elif len(prior_indices) == 1:
                overlap = np.vdot(
                    self.previous_vectors[:, prior_indices[0]], current[:, 0]
                )
                if abs(overlap) > 0.0:
                    current[:, 0] *= np.exp(-1j * np.angle(overlap))
                cluster_id = int(self.previous_clusters[prior_indices[0]])
            if len(group) == 1 or not transported:
                current = _phase_fix(current)
            vectors[:, group] = current
            clusters[group] = cluster_id
            if cluster_id == self.next_cluster_id:
                self.next_cluster_id += 1

        self.previous_values = values.copy()
        self.previous_vectors = vectors.copy()
        self.previous_labels = labels.copy()
        self.previous_clusters = clusters.copy()
        return vectors, clusters


def _select_modes(candidates, n_modes=N_MODES):
    if len(candidates) < int(n_modes):
        raise RuntimeError(
            f"checkpoint has only {len(candidates)} finite ambient modes; "
            f"{int(n_modes)} are required"
        )
    order = sorted(
        range(len(candidates)),
        key=lambda index: (abs(candidates[index][0]), candidates[index][0], candidates[index][1]),
    )
    selected = [candidates[index] for index in order[: int(n_modes)]]
    boundary_gap = (
        abs(candidates[order[int(n_modes)]][0]) - abs(selected[-1][0])
        if len(order) > int(n_modes)
        else np.inf
    )
    values = np.asarray([row[0] for row in selected], dtype=np.float64)
    labels = np.asarray([row[1] for row in selected], dtype=np.int8)
    vectors = np.stack([row[2] for row in selected], axis=1).astype(np.complex128)
    residuals = np.asarray([row[3] for row in selected], dtype=np.float64)
    return values, vectors, labels, float(boundary_gap), residuals


@dataclass
class LogGramObserver:
    checkpoints: tuple[int, ...]
    active_indices: np.ndarray
    samples: int
    arm: str
    n_modes: int = N_MODES
    singular_tolerance: float = 1e-12
    degeneracy_tolerance: float = 64.0 * np.finfo(np.float64).eps
    transports: list[EigenbasisTransport] = field(init=False)
    rows: dict[int, dict[int, dict]] = field(default_factory=dict, init=False)

    def __post_init__(self):
        self.checkpoints = tuple(int(value) for value in self.checkpoints)
        self.active_indices = np.asarray(self.active_indices, dtype=np.int64)
        self.transports = [
            EigenbasisTransport(degeneracy_tolerance=float(self.degeneracy_tolerance))
            for _ in range(int(self.samples))
        ]
        if self.arm not in ("maxmix", "pure"):
            raise ValueError("arm must be 'maxmix' or 'pure'")

    def _store(self, sample_id, cycle, selected, ranks, nulls):
        values, vectors, labels, gap, residuals = selected
        vectors, clusters = self.transports[int(sample_id)].align(
            values, vectors, labels
        )
        residuals = np.asarray(residuals, dtype=np.float64).copy()
        for cluster_id in np.unique(clusters):
            members = np.flatnonzero(clusters == cluster_id)
            if members.size > 1:
                # Procrustes rotates only roundoff-degenerate modes.  The
                # cluster spread is a conservative post-transport residual.
                residuals[members] = np.maximum(
                    residuals[members], np.ptp(values[members])
                )
        gram_error = np.linalg.norm(
            vectors.conj().T @ vectors - np.eye(self.n_modes), ord="fro"
        )
        self.rows.setdefault(int(sample_id), {})[int(cycle)] = {
            "values": values,
            "vectors": vectors,
            "labels": labels,
            "clusters": clusters,
            "finite_ranks": np.asarray(ranks, dtype=np.int32),
            "null_counts": np.asarray(nulls, dtype=np.int32),
            "boundary_gap": float(gap),
            "residuals": residuals,
            "gram_error": float(gram_error),
        }

    def maxmix_callback(self, *, cycle, G, batch_start, batch_count, **_):
        if int(cycle) not in self.checkpoints:
            return
        active = torch.as_tensor(self.active_indices, dtype=torch.long, device=G.device)
        for offset in range(int(batch_count)):
            covariance = G[offset].index_select(0, active).index_select(1, active)
            covariance = 0.5 * (covariance + covariance.mH)
            occupations, vectors = torch.linalg.eigh(covariance)
            gram_values = 1.0 - occupations.to(torch.float64) ** 2
            gram_tolerance = float(self.singular_tolerance) ** 2
            finite = torch.isfinite(gram_values) & (gram_values > gram_tolerance)
            finite_indices = torch.nonzero(finite, as_tuple=False).flatten()
            rank = int(finite_indices.numel())
            if rank < int(self.n_modes):
                raise RuntimeError(
                    f"checkpoint has only {rank} finite ambient modes; "
                    f"{int(self.n_modes)} are required"
                )
            finite_h = torch.log(gram_values.index_select(0, finite_indices))
            order = torch.argsort(torch.abs(finite_h), stable=True)
            selected_indices = finite_indices.index_select(
                0, order[: int(self.n_modes)]
            )
            selected_values = finite_h.index_select(
                0, order[: int(self.n_modes)]
            )
            boundary_gap = (
                float(
                    (
                        torch.abs(finite_h[order[int(self.n_modes)]])
                        - torch.abs(selected_values[-1])
                    ).item()
                )
                if rank > int(self.n_modes)
                else np.inf
            )
            selected_vectors = vectors.index_select(1, selected_indices)
            selected_mu = gram_values.index_select(0, selected_indices)
            gram_vectors = (
                selected_vectors
                - covariance @ (covariance @ selected_vectors)
            )
            gram_norm = max(
                float(torch.amax(torch.abs(gram_values[finite])).item()),
                gram_tolerance,
            )
            residuals = (
                torch.linalg.vector_norm(
                    gram_vectors
                    - selected_vectors * selected_mu.to(covariance.dtype).unsqueeze(0),
                    dim=0,
                )
                / gram_norm
            )
            selected = (
                selected_values.detach().cpu().numpy().astype(np.float64),
                selected_vectors.detach().cpu().numpy().astype(np.complex128),
                np.full((self.n_modes,), BLOCK_MIXED, dtype=np.int8),
                boundary_gap,
                residuals.detach().cpu().numpy().astype(np.float64),
            )
            self._store(
                int(batch_start) + offset,
                int(cycle),
                selected,
                [rank],
                [len(gram_values) - rank],
            )

    def pure_callback(
        self,
        *,
        cycle,
        batch_start,
        lyapunov_frame,
        lyapunov_block_sizes,
        lyapunov_block_core_hat,
        lyapunov_block_core_log_scale,
        **_,
    ):
        if int(cycle) not in self.checkpoints:
            return
        if int(lyapunov_frame.shape[0]) != 1:
            raise RuntimeError("pure ambient observer requires batch_size=1")
        active = torch.as_tensor(
            self.active_indices, dtype=torch.long, device=lyapunov_frame.device
        )
        block_data = []
        scalar_candidates = []
        finite_ranks = []
        null_counts = []
        start = 0
        log_tolerance = float(np.log(self.singular_tolerance))
        for block_index, block_size in enumerate(lyapunov_block_sizes):
            stop = start + int(block_size)
            frame = lyapunov_frame[0, :, start:stop].index_select(0, active)
            core = lyapunov_block_core_hat[block_index][0]
            scale = float(lyapunov_block_core_log_scale[block_index][0].item())
            left, singular_values, right_h = torch.linalg.svd(
                core, full_matrices=False
            )
            positive = singular_values > 0.0
            log_sigma = torch.full_like(singular_values, -torch.inf)
            log_sigma[positive] = torch.log(singular_values[positive]) + scale
            finite = torch.isfinite(log_sigma) & (log_sigma > log_tolerance)
            finite_indices = torch.nonzero(finite, as_tuple=False).flatten()
            finite_rank = int(finite_indices.numel())
            finite_ranks.append(finite_rank)
            null_counts.append(int(block_size) - finite_rank)
            h_values = 2.0 * log_sigma.index_select(0, finite_indices)
            for index, h_value in zip(
                finite_indices.detach().cpu().tolist(),
                h_values.detach().cpu().tolist(),
            ):
                scalar_candidates.append((float(h_value), block_index, int(index)))
            block_data.append(
                {
                    "frame": frame,
                    "core": core,
                    "left": left,
                    "singular_values": singular_values,
                    "right_h": right_h,
                }
            )
            start = stop
        if len(scalar_candidates) < int(self.n_modes):
            raise RuntimeError(
                f"checkpoint has only {len(scalar_candidates)} finite ambient modes; "
                f"{int(self.n_modes)} are required"
            )
        order = sorted(
            range(len(scalar_candidates)),
            key=lambda index: (
                abs(scalar_candidates[index][0]),
                scalar_candidates[index][0],
                scalar_candidates[index][1],
            ),
        )
        chosen = [scalar_candidates[index] for index in order[: int(self.n_modes)]]
        boundary_gap = (
            abs(scalar_candidates[order[int(self.n_modes)]][0])
            - abs(chosen[-1][0])
            if len(order) > int(self.n_modes)
            else np.inf
        )
        selected_values = np.asarray([row[0] for row in chosen], dtype=np.float64)
        selected_labels = np.asarray(
            [
                BLOCK_OCCUPIED if row[1] == 0 else BLOCK_EMPTY
                for row in chosen
            ],
            dtype=np.int8,
        )
        selected_vectors = np.empty(
            (len(self.active_indices), self.n_modes), dtype=np.complex128
        )
        selected_residuals = np.empty((self.n_modes,), dtype=np.float64)
        for block_index in range(len(block_data)):
            output_positions = [
                position
                for position, row in enumerate(chosen)
                if row[1] == block_index
            ]
            if not output_positions:
                continue
            singular_indices = torch.as_tensor(
                [chosen[position][2] for position in output_positions],
                dtype=torch.long,
                device=lyapunov_frame.device,
            )
            data = block_data[block_index]
            left_vectors = data["left"].index_select(1, singular_indices)
            ambient_vectors = data["frame"] @ left_vectors
            ambient_vectors /= torch.linalg.vector_norm(
                ambient_vectors, dim=0
            ).clamp_min(torch.finfo(torch.float64).tiny)
            selected_vectors[:, output_positions] = (
                ambient_vectors.detach().cpu().numpy()
            )
            selected_s = data["singular_values"].index_select(
                0, singular_indices
            )
            gram_left = data["core"] @ (data["core"].mH @ left_vectors)
            gram_residual = torch.linalg.vector_norm(
                gram_left
                - left_vectors * (selected_s * selected_s).to(data["core"].dtype).unsqueeze(0),
                dim=0,
            )
            gram_norm = torch.clamp(
                data["singular_values"][0] ** 2,
                min=torch.finfo(torch.float64).tiny,
            )
            selected_residuals[output_positions] = (
                (gram_residual / gram_norm).detach().cpu().numpy()
            )
        selected = (
            selected_values,
            selected_vectors,
            selected_labels,
            float(boundary_gap),
            selected_residuals,
        )
        self._store(
            int(batch_start), int(cycle), selected, finite_ranks, null_counts
        )

    def arrays(self, global_sample_ids):
        sample_ids = [int(value) for value in global_sample_ids]
        shape = (len(sample_ids), len(self.checkpoints))
        values = np.empty(shape + (self.n_modes,), dtype=np.float64)
        vectors = np.empty(
            shape + (len(self.active_indices), self.n_modes), dtype=np.complex128
        )
        labels = np.empty(shape + (self.n_modes,), dtype=np.int8)
        clusters = np.empty(shape + (self.n_modes,), dtype=np.int32)
        residuals = np.empty(shape + (self.n_modes,), dtype=np.float64)
        finite_ranks = np.empty(shape + ((1,) if self.arm == "maxmix" else (2,)), dtype=np.int32)
        null_counts = np.empty_like(finite_ranks)
        boundary_gaps = np.empty(shape, dtype=np.float64)
        gram_errors = np.empty(shape, dtype=np.float64)
        for sample_offset, sample_id in enumerate(sample_ids):
            if set(self.rows.get(sample_id, {})) != set(self.checkpoints):
                raise RuntimeError(
                    f"sample {sample_id} is missing checkpoint eigensystems"
                )
            for checkpoint_offset, cycle in enumerate(self.checkpoints):
                row = self.rows[sample_id][cycle]
                target = (sample_offset, checkpoint_offset)
                values[target] = row["values"]
                vectors[target] = row["vectors"]
                labels[target] = row["labels"]
                clusters[target] = row["clusters"]
                residuals[target] = row["residuals"]
                finite_ranks[target] = row["finite_ranks"]
                null_counts[target] = row["null_counts"]
                boundary_gaps[target] = row["boundary_gap"]
                gram_errors[target] = row["gram_error"]
        return {
            "log_gram_eigenvalues": values,
            "log_gram_eigenvectors": vectors,
            "mode_block_labels": labels,
            "degenerate_cluster_labels": clusters,
            "eigenpair_residuals": residuals,
            "finite_ranks": finite_ranks,
            "null_counts": null_counts,
            "boundary_spectral_gaps": boundary_gaps,
            "eigenvector_gram_errors": gram_errors,
        }
