from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class FlagScan:
    """Prefix data for a pair of filled/empty constraint flags."""

    rows: tuple[dict[str, Any], ...]
    first_breakdown: dict[str, Any] | None


def _normalized_columns(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.complex128)
    if vectors.ndim != 2:
        raise ValueError("vectors must be a D by M matrix")
    norms = np.linalg.norm(vectors, axis=0)
    if np.any(~np.isfinite(norms)) or np.any(norms <= 1e-14):
        raise ValueError("constraint vectors must be finite and nonzero")
    return vectors / norms[None, :]


def _append_basis(Q: np.ndarray, vector: np.ndarray, rtol: float) -> tuple[np.ndarray, bool]:
    residual = vector - Q @ (Q.conj().T @ vector) if Q.shape[1] else vector.copy()
    # Reorthogonalize once; this is much more stable for long, redundant frames.
    if Q.shape[1]:
        residual -= Q @ (Q.conj().T @ residual)
    norm = float(np.linalg.norm(residual))
    if norm <= rtol:
        return Q, False
    return np.column_stack((Q, residual / norm)), True


def _reason_map(
    checkpoint_indices: Sequence[int] | Mapping[int, Sequence[str]], n_constraints: int
) -> dict[int, tuple[str, ...]]:
    if isinstance(checkpoint_indices, Mapping):
        result = {
            int(index): tuple(str(reason) for reason in reasons)
            for index, reasons in checkpoint_indices.items()
        }
    else:
        result = {int(index): ("requested",) for index in checkpoint_indices}
    result[n_constraints] = tuple(sorted(set(result.get(n_constraints, ()) + ("final",))))
    if any(index < 1 or index > n_constraints for index in result):
        raise ValueError("checkpoint indices must lie in 1..M")
    return result


def analyze_constraint_flag(
    vectors: np.ndarray,
    target_occupancies: np.ndarray,
    *,
    labels: Mapping[str, Sequence[Any]],
    target_rank: int,
    ordering: Sequence[int],
    checkpoint_indices: Sequence[int] | Mapping[int, Sequence[str]],
    rtol: float = 1e-10,
) -> FlagScan:
    """Scan an ordered constraint word without changing the canonical engine.

    Ranks and exact feasibility are updated at every OW letter.  Principal-angle
    and Ky--Fan quantities are evaluated only at the supplied checkpoints.  At a
    checkpoint m, ``delta_f_star`` is the literal F*(m)-F*(m-1), not a coarse
    difference between checkpoints.
    """

    from fgtn.diagnostics import compute_completion_from_constraints

    W = _normalized_columns(vectors)
    targets = np.asarray(target_occupancies, dtype=np.int8).reshape(-1)
    order = np.asarray(ordering, dtype=np.int64).reshape(-1)
    dimension, count = W.shape
    if targets.size != count or order.size != count or set(order.tolist()) != set(range(count)):
        raise ValueError("targets and ordering must each describe all constraints exactly once")
    if not 0 <= int(target_rank) <= dimension:
        raise ValueError("invalid target rank")
    if np.any((targets != 0) & (targets != 1)):
        raise ValueError("targets must be binary")
    for key, values in labels.items():
        if len(values) != count:
            raise ValueError(f"label {key!r} does not match the constraints")

    checkpoints = _reason_map(checkpoint_indices, count)
    Q_f = np.empty((dimension, 0), dtype=np.complex128)
    Q_e = np.empty((dimension, 0), dtype=np.complex128)
    rows: list[dict[str, Any]] = []
    first: dict[str, Any] | None = None

    for position, source_index in enumerate(order, start=1):
        vector = W[:, source_index]
        target = int(targets[source_index])
        cross_norm = float(np.linalg.norm((Q_e if target else Q_f).conj().T @ vector))
        if target:
            Q_f, _ = _append_basis(Q_f, vector, rtol)
        else:
            Q_e, _ = _append_basis(Q_e, vector, rtol)
        rank_ok = Q_f.shape[1] <= target_rank <= dimension - Q_e.shape[1]
        incompatible = cross_norm > rtol
        exact = bool(rank_ok and not incompatible and first is None)
        breakdown_reason = ""
        if first is None and (incompatible or not rank_ok):
            breakdown_reason = "nonorthogonality" if incompatible else "rank_bound"
            first = {
                "prefix": position,
                "source_index": int(source_index),
                "reason": breakdown_reason,
                "cross_norm": cross_norm,
                "filled_rank": int(Q_f.shape[1]),
                "empty_rank": int(Q_e.shape[1]),
                **{key: np.asarray(value)[source_index].item() for key, value in labels.items()},
            }

        row: dict[str, Any] = {
            "prefix": position,
            "source_index": int(source_index),
            "constraint_fraction": position / count,
            "target": target,
            "filled_rank": int(Q_f.shape[1]),
            "empty_rank": int(Q_e.shape[1]),
            "rank_bounds_satisfied": bool(rank_ok),
            "new_cross_norm": cross_norm,
            "exact_feasible_so_far": exact,
            "breakdown_here": bool(breakdown_reason),
            "breakdown_reason": breakdown_reason,
            "is_checkpoint": position in checkpoints,
            "checkpoint_reason": "+".join(checkpoints.get(position, ())),
            "sigma_max": np.nan,
            "overlap_phi": np.nan,
            "f_star": np.nan,
            "f_star_per_constraint": np.nan,
            "delta_f_star": np.nan,
            "principal_cosines": "",
        }
        row.update({key: np.asarray(value)[source_index].item() for key, value in labels.items()})

        if position in checkpoints:
            prefix_indices = order[:position]
            result = compute_completion_from_constraints(
                W[:, prefix_indices], targets[prefix_indices], target_rank=target_rank, rtol=rtol
            )
            previous_cost = 0.0
            if position > 1:
                previous_indices = order[: position - 1]
                previous_cost = compute_completion_from_constraints(
                    W[:, previous_indices],
                    targets[previous_indices],
                    target_rank=target_rank,
                    rtol=rtol,
                ).f_star
            row.update(
                sigma_max=result.sigma_max,
                overlap_phi=result.overlap_phi,
                f_star=result.f_star,
                f_star_per_constraint=result.f_star_per_constraint,
                delta_f_star=result.f_star - previous_cost,
                principal_cosines=";".join(f"{value:.17g}" for value in result.principal_cosines),
                exact_feasible_so_far=bool(result.exact_completion_exists),
            )
        rows.append(row)
    return FlagScan(rows=tuple(rows), first_breakdown=first)


def cell_order(
    center_x: np.ndarray,
    center_y: np.ndarray,
    channel: np.ndarray,
    *,
    mode: str,
    wall_x: tuple[int, int] = (3, 13),
    random_site_order: Sequence[tuple[int, int]] | None = None,
) -> np.ndarray:
    """Return a site-major four-letter OW ordering."""

    x = np.asarray(center_x, dtype=int)
    y = np.asarray(center_y, dtype=int)
    c = np.asarray(channel, dtype=int)
    sites = sorted(set(zip(x.tolist(), y.tolist())))
    if mode == "interior_to_wall":
        sites.sort(key=lambda xy: (-min(abs(xy[0] - wall_x[0]), abs(xy[0] - wall_x[1])), xy[0], xy[1]))
    elif mode == "raster_y":
        sites.sort(key=lambda xy: (xy[0], xy[1]))
    elif mode == "random":
        if random_site_order is None:
            raise ValueError("random mode requires random_site_order")
        filtered = [tuple(map(int, site)) for site in random_site_order if tuple(map(int, site)) in set(sites)]
        if len(filtered) != len(sites) or len(set(filtered)) != len(sites):
            raise ValueError("random site word does not cover the static center set")
        sites = filtered
    else:
        raise ValueError(f"unknown ordering mode {mode!r}")
    lookup = {(int(xi), int(yi), int(ci)): i for i, (xi, yi, ci) in enumerate(zip(x, y, c))}
    return np.asarray([lookup[(xi, yi, ci)] for xi, yi in sites for ci in range(4)], dtype=np.int64)


def checkpoint_map(
    order: Sequence[int],
    center_x: np.ndarray,
    *,
    ordering_mode: str,
    wall_x: tuple[int, int] = (3, 13),
) -> dict[int, tuple[str, ...]]:
    """Sparse expensive checkpoints: columns/shells, deciles, and final blocks."""

    order = np.asarray(order, dtype=int)
    x = np.asarray(center_x, dtype=int)[order]
    count = order.size
    reasons: dict[int, set[str]] = {}
    for fraction in np.linspace(0.1, 1.0, 10):
        index = min(count, max(1, int(round(fraction * count))))
        index -= index % 4
        index = max(4, index)
        reasons.setdefault(index, set()).add("decile")
    if ordering_mode in ("raster_y", "interior_to_wall"):
        for index in range(4, count + 1, 4):
            if index == count or (index < count and x[index - 1] != x[index]):
                reasons.setdefault(index, set()).add("column")
    if ordering_mode == "interior_to_wall":
        distances = np.minimum(np.abs(x - wall_x[0]), np.abs(x - wall_x[1]))
        for index in range(4, count + 1, 4):
            if index == count or (index < count and distances[index - 1] != distances[index]):
                reasons.setdefault(index, set()).add("shell")
    # The terminal prefix completes all four OW channel families.
    reasons.setdefault(count, set()).add("channel_blocks_complete")
    return {index: tuple(sorted(values)) for index, values in reasons.items()}
