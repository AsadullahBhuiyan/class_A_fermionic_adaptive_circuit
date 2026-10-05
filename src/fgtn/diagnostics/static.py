from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .common import CHANNELS, CHANNEL_TARGETS, active_cell_coordinates


@dataclass
class StaticCompletionResult:
    dimension: int
    target_rank: int
    filled_rank: int
    empty_rank: int
    singular_values_filled: np.ndarray
    singular_values_empty: np.ndarray
    principal_cosines: np.ndarray
    sigma_max: float
    overlap_phi: float
    rank_bounds_satisfied: bool
    exact_completion_exists: bool
    completion_constraint_error: float
    f_star: float
    f_star_formula: float
    f_star_per_constraint: float
    operator_eigenvalues: np.ndarray
    residuals: np.ndarray
    residual_map: np.ndarray
    target_occupancies: np.ndarray
    weights: np.ndarray
    channel_indices: np.ndarray
    center_x: np.ndarray
    center_y: np.ndarray
    optimal_projector: np.ndarray
    completion_projector: np.ndarray | None

    def payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "dimension": np.asarray(self.dimension, dtype=np.int64),
            "target_rank": np.asarray(self.target_rank, dtype=np.int64),
            "filled_rank": np.asarray(self.filled_rank, dtype=np.int64),
            "empty_rank": np.asarray(self.empty_rank, dtype=np.int64),
            "singular_values_filled": self.singular_values_filled,
            "singular_values_empty": self.singular_values_empty,
            "principal_cosines": self.principal_cosines,
            "sigma_max": np.asarray(self.sigma_max),
            "overlap_phi": np.asarray(self.overlap_phi),
            "rank_bounds_satisfied": np.asarray(self.rank_bounds_satisfied),
            "exact_completion_exists": np.asarray(self.exact_completion_exists),
            "completion_constraint_error": np.asarray(self.completion_constraint_error),
            "f_star": np.asarray(self.f_star),
            "f_star_formula": np.asarray(self.f_star_formula),
            "f_star_per_constraint": np.asarray(self.f_star_per_constraint),
            "operator_eigenvalues": self.operator_eigenvalues,
            "residuals": self.residuals,
            "residual_map": self.residual_map,
            "target_occupancies": self.target_occupancies,
            "weights": self.weights,
            "channel_indices": self.channel_indices,
            "center_x": self.center_x,
            "center_y": self.center_y,
            "optimal_projector": self.optimal_projector,
        }
        if self.completion_projector is not None:
            payload["completion_projector"] = self.completion_projector
        return payload

    def summary(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension,
            "target_rank": self.target_rank,
            "filled_rank": self.filled_rank,
            "empty_rank": self.empty_rank,
            "sigma_max": self.sigma_max,
            "overlap_phi": self.overlap_phi,
            "rank_bounds_satisfied": self.rank_bounds_satisfied,
            "exact_completion_exists": self.exact_completion_exists,
            "completion_constraint_error": self.completion_constraint_error,
            "f_star": self.f_star,
            "f_star_formula": self.f_star_formula,
            "f_star_per_constraint": self.f_star_per_constraint,
        }


def _normalize_columns(W: np.ndarray) -> np.ndarray:
    W = np.asarray(W, dtype=np.complex128)
    if W.ndim != 2:
        raise ValueError("Constraint vectors must form a two-dimensional matrix.")
    norms = np.linalg.norm(W, axis=0)
    if np.any(~np.isfinite(norms)) or np.any(norms <= 1e-14):
        raise ValueError("Every constraint vector must have finite nonzero norm.")
    return W / norms[None, :]


def _span_basis(W: np.ndarray, rtol: float) -> tuple[np.ndarray, np.ndarray]:
    if W.shape[1] == 0:
        return np.empty((W.shape[0], 0), dtype=np.complex128), np.empty((0,), dtype=np.float64)
    U, singular_values, _ = np.linalg.svd(W, full_matrices=False)
    threshold = float(rtol) * float(singular_values[0]) if singular_values.size else 0.0
    rank = int(np.count_nonzero(singular_values > threshold))
    return U[:, :rank], singular_values


def _completion_projector(Q_f: np.ndarray, Q_e: np.ndarray, target_rank: int) -> np.ndarray:
    dimension = int(Q_f.shape[0])
    extra = int(target_rank) - int(Q_f.shape[1])
    if extra < 0:
        raise ValueError("Filled span exceeds the target rank.")
    occupied = Q_f
    if extra:
        joined = np.concatenate((Q_f, Q_e), axis=1)
        if joined.shape[1] == 0:
            complement = np.eye(dimension, dtype=np.complex128)
        else:
            _, singular_values, vh = np.linalg.svd(joined.conj().T, full_matrices=True)
            threshold = 1e-12 * float(singular_values[0]) if singular_values.size else 0.0
            joined_rank = int(np.count_nonzero(singular_values > threshold))
            complement = vh[joined_rank:, :].conj().T
        if complement.shape[1] < extra:
            raise ValueError("Insufficient unconstrained complement for the requested rank.")
        occupied = np.concatenate((Q_f, complement[:, :extra]), axis=1)
    Q_occ, _ = np.linalg.qr(occupied, mode="reduced")
    return Q_occ @ Q_occ.conj().T


def compute_completion_from_constraints(
    vectors: np.ndarray,
    target_occupancies: np.ndarray,
    *,
    target_rank: int,
    weights: np.ndarray | None = None,
    rtol: float = 1e-10,
    channel_indices: np.ndarray | None = None,
    center_x: np.ndarray | None = None,
    center_y: np.ndarray | None = None,
    map_shape: tuple[int, int, int] | None = None,
) -> StaticCompletionResult:
    if not np.isfinite(rtol) or rtol <= 0.0:
        raise ValueError("rtol must be positive and finite.")
    W = _normalize_columns(vectors)
    dimension, n_constraints = W.shape
    targets = np.asarray(target_occupancies, dtype=np.int8).reshape(-1)
    if targets.size != n_constraints or np.any((targets != 0) & (targets != 1)):
        raise ValueError("target_occupancies must contain one binary value per constraint.")
    target_rank = int(target_rank)
    if target_rank < 0 or target_rank > dimension:
        raise ValueError("target_rank must lie between zero and the Hilbert-space dimension.")
    gamma = np.ones((n_constraints,), dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64).reshape(-1)
    if gamma.size != n_constraints or np.any(~np.isfinite(gamma)) or np.any(gamma < 0.0):
        raise ValueError("weights must be finite, nonnegative, and match the constraints.")

    W_f = W[:, targets == 1]
    W_e = W[:, targets == 0]
    Q_f, singular_f = _span_basis(W_f, rtol)
    Q_e, singular_e = _span_basis(W_e, rtol)
    principal = (
        np.linalg.svd(Q_f.conj().T @ Q_e, compute_uv=False)
        if Q_f.shape[1] and Q_e.shape[1]
        else np.empty((0,), dtype=np.float64)
    )
    principal = np.clip(np.real(principal), 0.0, 1.0)
    sigma_max = float(principal[0]) if principal.size else 0.0
    overlap_phi = float(np.sum(principal * principal))
    rank_bounds = bool(Q_f.shape[1] <= target_rank <= dimension - Q_e.shape[1])
    exact = bool(rank_bounds and sigma_max <= float(rtol))

    completion = _completion_projector(Q_f, Q_e, target_rank) if exact else None
    completion_error = np.nan
    if completion is not None:
        filled_error = np.linalg.norm((np.eye(dimension) - completion) @ Q_f, ord="fro")
        empty_error = np.linalg.norm(completion @ Q_e, ord="fro")
        rank_error = abs(float(np.trace(completion).real) - target_rank)
        completion_error = float(max(filled_error, empty_error, rank_error))

    signs = 1.0 - 2.0 * targets.astype(np.float64)
    operator = (W * (gamma * signs)[None, :]) @ W.conj().T
    operator = 0.5 * (operator + operator.conj().T)
    eigenvalues, eigenvectors = np.linalg.eigh(operator)
    occupied = eigenvectors[:, :target_rank]
    optimal = occupied @ occupied.conj().T
    occupancies = np.real(np.einsum("ia,ij,ja->a", W.conj(), optimal, W, optimize=True))
    occupancies = np.clip(occupancies, 0.0, 1.0)
    residuals = np.where(targets == 1, 1.0 - occupancies, occupancies)
    f_star = float(np.dot(gamma, residuals))
    f_formula = float(np.dot(gamma, targets) + np.sum(eigenvalues[:target_rank]))

    cidx = np.full((n_constraints,), -1, dtype=np.int64) if channel_indices is None else np.asarray(channel_indices, dtype=np.int64)
    xpos = np.full((n_constraints,), -1, dtype=np.int64) if center_x is None else np.asarray(center_x, dtype=np.int64)
    ypos = np.full((n_constraints,), -1, dtype=np.int64) if center_y is None else np.asarray(center_y, dtype=np.int64)
    if cidx.size != n_constraints or xpos.size != n_constraints or ypos.size != n_constraints:
        raise ValueError("Constraint labels must match the number of vectors.")
    residual_map = np.empty((0,), dtype=np.float64)
    if map_shape is not None:
        residual_map = np.full(map_shape, np.nan, dtype=np.float64)
        for value, x, y, channel in zip(residuals, xpos, ypos, cidx):
            residual_map[int(x), int(y), int(channel)] = float(value)

    return StaticCompletionResult(
        dimension=dimension,
        target_rank=target_rank,
        filled_rank=int(Q_f.shape[1]),
        empty_rank=int(Q_e.shape[1]),
        singular_values_filled=singular_f,
        singular_values_empty=singular_e,
        principal_cosines=principal,
        sigma_max=sigma_max,
        overlap_phi=overlap_phi,
        rank_bounds_satisfied=rank_bounds,
        exact_completion_exists=exact,
        completion_constraint_error=float(completion_error),
        f_star=f_star,
        f_star_formula=f_formula,
        f_star_per_constraint=f_star / float(np.sum(gamma)) if np.sum(gamma) > 0 else np.nan,
        operator_eigenvalues=eigenvalues,
        residuals=residuals,
        residual_map=residual_map,
        target_occupancies=targets,
        weights=gamma,
        channel_indices=cidx,
        center_x=xpos,
        center_y=ypos,
        optimal_projector=optimal,
        completion_projector=completion,
    )


def compute_static_completion(
    model: Any,
    *,
    meas_slab_only: bool = True,
    weights: np.ndarray | None = None,
    rtol: float = 1e-10,
) -> StaticCompletionResult:
    have_frames = all(hasattr(model, name) for name in ("WF_Ap", "WF_Am", "WF_Bp", "WF_Bm"))
    if not have_frames:
        model.construct_OW_projectors(
            nshell=model.nshell,
            DW=model.DW,
            trial_orbitals=model.trial_orbitals,
            dw_truncation=model.dw_truncation,
        )
    active_indices = np.asarray(model.active_top_layer_indices(meas_slab_only=meas_slab_only), dtype=np.int64)
    centers = active_cell_coordinates(model, meas_slab_only=meas_slab_only)
    columns: list[np.ndarray] = []
    targets: list[int] = []
    channel_indices: list[int] = []
    center_x: list[int] = []
    center_y: list[int] = []
    for x, y in centers:
        for channel_index, channel in enumerate(CHANNELS):
            vector = np.asarray(getattr(model, f"WF_{channel}")[:, x, y], dtype=np.complex128)
            columns.append(vector[active_indices])
            targets.append(CHANNEL_TARGETS[channel])
            channel_indices.append(channel_index)
            center_x.append(x)
            center_y.append(y)
    vectors = np.column_stack(columns)
    dimension = int(active_indices.size)
    if dimension % 2:
        raise ValueError("The active single-particle dimension must be even at half filling.")
    return compute_completion_from_constraints(
        vectors,
        np.asarray(targets, dtype=np.int8),
        target_rank=dimension // 2,
        weights=weights,
        rtol=rtol,
        channel_indices=np.asarray(channel_indices, dtype=np.int64),
        center_x=np.asarray(center_x, dtype=np.int64),
        center_y=np.asarray(center_y, dtype=np.int64),
        map_shape=(int(model.Nx), int(model.Ny), len(CHANNELS)),
    )
