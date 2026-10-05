#!/usr/bin/env python3
"""Generate preregistered controls and sensitivities for the wall pump."""

from __future__ import annotations

import os

for _name in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_name, "1")

import argparse
import contextlib
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import io
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time
from typing import Any, Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment
from tqdm.auto import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_ROOT.parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fgtn.classA_U1FGTN import classA_U1FGTN  # noqa: E402
import run_wall_diabatic_spectral_pump_s100 as core  # noqa: E402


EXACT_SOURCE_SCHEMA = "wall_diabatic_exact_control_source_v1"
EXACT_SOURCE_COMPLETION_SCHEMA = "wall_diabatic_exact_control_source_completion_v1"
DEFAULT_OUTPUT = core.DEFAULT_OUTPUT
CONTROL_NX, CONTROL_NY = 20, 40
CONTROL_WALLS = ("soft", "hard")
CONTROL_SOURCE_KINDS = ("topological", "trivial", "conjugated")
EXACT_DEGENERACY_TOLERANCE = 1e-9
SENSITIVITY_VARIANTS = {
    "M128": {"grid_intervals": 128, "edge_block_rank": 2, "wall_window": 2, "twist_gauge": "uniform"},
    "M512": {"grid_intervals": 512, "edge_block_rank": 2, "wall_window": 2, "twist_gauge": "uniform"},
    "seam_M256": {"grid_intervals": 256, "edge_block_rank": 2, "wall_window": 2, "twist_gauge": "seam"},
    "radius3_M256": {"grid_intervals": 256, "edge_block_rank": 2, "wall_window": 3, "twist_gauge": "uniform"},
    "rank4_M256": {"grid_intervals": 256, "edge_block_rank": 4, "wall_window": 2, "twist_gauge": "uniform"},
}
HISTORICAL_FLIPS = {
    "hard": {3, 6, 31, 85, 86, 87},
    "soft": {4, 8, 34, 59, 69, 80, 98},
}


def _frame_projector(frame: np.ndarray, dimension: int, rank: int) -> np.ndarray:
    values = np.asarray(frame, dtype=np.complex128).reshape(dimension, rank, order="F")
    return values @ values.conj().T


def _exact_hamiltonian(
    *, wall: str, source_kind: str, phi: float = 0.0,
) -> np.ndarray:
    """Build the authoritative exact OW Hamiltonian at the requested flux.

    This is deliberately not obtained by twisting ``I - 2 P(0)``.  The exact
    equilibrium control rebuilds the OW functions at every flux, matching the
    legacy flattened-Hamiltonian benchmark.  The conjugated control is the
    time-reversed family ``H_top(-phi)^*``.
    """
    if source_kind not in CONTROL_SOURCE_KINDS:
        raise ValueError(f"unknown exact source kind {source_kind!r}")
    if source_kind == "conjugated":
        return np.asarray(
            _exact_hamiltonian(wall=wall, source_kind="topological", phi=-float(phi)).conj(),
            dtype=np.complex128,
        )
    alpha_top = 30.0 if source_kind == "trivial" else 1.0
    truncation = wall == "hard"
    with contextlib.redirect_stdout(io.StringIO()):
        model = classA_U1FGTN(
            Nx=CONTROL_NX, Ny=CONTROL_NY, DW=True, nshell=1,
            filling_frac=0.5, alpha_1=alpha_top, alpha_2=30.0,
            trial_orbitals="X", dw_truncation=truncation,
            twist_y=float(phi), dw_interval=(5, 15),
        )
        model.construct_OW_projectors(
            nshell=1, DW=True, trial_orbitals="X",
            dw_truncation=truncation, twist_y=float(phi),
        )
    dimension, rank = 2 * CONTROL_NX * CONTROL_NY, CONTROL_NX * CONTROL_NY
    h = (
        _frame_projector(model.WF_Ap, dimension, rank)
        + _frame_projector(model.WF_Bp, dimension, rank)
        - _frame_projector(model.WF_Am, dimension, rank)
        - _frame_projector(model.WF_Bm, dimension, rank)
    )
    return np.asarray(0.5 * (h + h.conj().T), dtype=np.complex128)


def _y_blocks(matrix: np.ndarray) -> tuple[np.ndarray, float]:
    """Transform a translation-invariant real-space matrix to fixed-k blocks."""
    orbitals = 2 * CONTROL_NX
    shaped = np.asarray(matrix, dtype=np.complex128).reshape(
        CONTROL_NY, orbitals, CONTROL_NY, orbitals
    )
    transformed = np.fft.ifft(np.fft.fft(shaped, axis=0), axis=2)
    blocks = np.stack([transformed[k, :, k, :] for k in range(CONTROL_NY)])
    off_diagonal = transformed.copy()
    for k in range(CONTROL_NY):
        off_diagonal[k, :, k, :] = 0.0
    return blocks, float(np.max(np.abs(off_diagonal)))


def _reconstruct_from_blocks(blocks: np.ndarray) -> np.ndarray:
    orbitals = 2 * CONTROL_NX
    transformed = np.zeros(
        (CONTROL_NY, orbitals, CONTROL_NY, orbitals), dtype=np.complex128
    )
    for k in range(CONTROL_NY):
        transformed[k, :, k, :] = blocks[k]
    shaped = np.fft.ifft(np.fft.fft(transformed, axis=2), axis=0)
    return shaped.reshape(2 * CONTROL_NX * CONTROL_NY, 2 * CONTROL_NX * CONTROL_NY)


def _exact_diagonalize(
    phi: float, wall: str, source_kind: str, twist_gauge: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Return the exact uniform-gauge Hamiltonian and its fixed-k eigensystems.

    A seam-gauge control is evaluated in the same common uniform gauge.  We
    explicitly construct the seam representative and transform it back before
    accepting the eigensystem, so the comparison tests gauge covariance rather
    than comparing unrelated eigenvector conventions.
    """
    hamiltonian = _exact_hamiltonian(
        wall=wall, source_kind=source_kind, phi=float(phi)
    )
    gauge_error = 0.0
    if twist_gauge == "seam":
        y = np.repeat(np.arange(CONTROL_NY), 2 * CONTROL_NX)
        basis = np.exp(1j * float(phi) * y / CONTROL_NY)
        seam = basis.conj()[:, None] * hamiltonian * basis[None, :]
        restored = basis[:, None] * seam * basis.conj()[None, :]
        gauge_error = float(np.max(np.abs(restored - hamiltonian)))
        if gauge_error > 1e-12:
            raise RuntimeError(f"exact seam/uniform gauge restoration failed: {gauge_error:.3e}")
    blocks, off_diagonal = _y_blocks(hamiltonian)
    eigenvalues, eigenvectors = np.linalg.eigh(blocks)
    return (
        hamiltonian,
        np.asarray(eigenvalues, dtype=np.float64),
        np.asarray(eigenvectors, dtype=np.complex128),
        off_diagonal,
        gauge_error,
    )


def _lowest_occupations(eigenvalues: np.ndarray) -> np.ndarray:
    occupied = np.zeros_like(eigenvalues, dtype=bool)
    order = np.argsort(eigenvalues, axis=None)[: CONTROL_NX * CONTROL_NY]
    occupied[np.unravel_index(order, eigenvalues.shape)] = True
    return occupied


def _continue_exact_basis(
    previous: np.ndarray, eigenvalues: np.ndarray, eigenvectors: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """Continue every eigenvector within its conserved momentum block."""
    tracked = np.empty_like(eigenvectors)
    assigned_minimum = 1.0
    principal_minimum = 1.0
    for momentum in range(CONTROL_NY):
        overlap = np.abs(previous[momentum].conj().T @ eigenvectors[momentum]) ** 2
        rows, columns = linear_sum_assignment(-overlap)
        permutation = columns[np.argsort(rows)]
        values = eigenvalues[momentum, permutation]
        vectors = np.asarray(eigenvectors[momentum][:, permutation], dtype=np.complex128)
        assigned_minimum = min(
            assigned_minimum,
            float(np.sqrt(overlap[np.arange(values.size), permutation]).min()),
        )
        unused = set(range(values.size))
        while unused:
            seed = min(unused)
            cluster = sorted(
                index for index in unused
                if abs(values[index] - values[seed]) < EXACT_DEGENERACY_TOLERANCE
            )
            unused -= set(cluster)
            indices = np.asarray(cluster, dtype=np.int64)
            small_overlap = (
                previous[momentum][:, indices].conj().T @ vectors[:, indices]
            )
            left, singular, right_h = np.linalg.svd(
                small_overlap, full_matrices=False
            )
            vectors[:, indices] = vectors[:, indices] @ (
                right_h.conj().T @ left.conj().T
            )
            principal_minimum = min(principal_minimum, float(singular.min()))
        tracked[momentum] = vectors
    return tracked, assigned_minimum, principal_minimum


def _block_frame(eigenvectors: np.ndarray, occupied: np.ndarray) -> np.ndarray:
    """Reconstruct an orthonormal real-space frame from fixed-k modes."""
    phases = np.exp(
        2j * np.pi * np.outer(np.arange(CONTROL_NY), np.arange(CONTROL_NY))
        / CONTROL_NY
    ) / np.sqrt(CONTROL_NY)
    columns: list[np.ndarray] = []
    for momentum in range(CONTROL_NY):
        block = eigenvectors[momentum][:, occupied[momentum]]
        spatial = phases[:, momentum, None, None] * block[None, :, :]
        columns.append(spatial.reshape(2 * CONTROL_NX * CONTROL_NY, block.shape[1]))
    frame = np.column_stack(columns)
    if frame.shape != (1600, 800):
        raise RuntimeError(f"exact block frame shape changed: {frame.shape}")
    return np.asarray(frame, dtype=np.complex128)


def _block_density_x(eigenvectors: np.ndarray, occupied: np.ndarray) -> np.ndarray:
    density = np.zeros(2 * CONTROL_NX, dtype=np.float64)
    for momentum in range(CONTROL_NY):
        density += np.sum(
            np.abs(eigenvectors[momentum][:, occupied[momentum]]) ** 2, axis=1
        )
    return density.reshape(CONTROL_NX, 2).sum(axis=1)


def _block_projector_residual(eigenvectors: np.ndarray, occupied: np.ndarray) -> float:
    residual = 0.0
    for momentum in range(CONTROL_NY):
        frame = eigenvectors[momentum][:, occupied[momentum]]
        if frame.shape[1]:
            residual = max(
                residual,
                float(np.max(np.abs(frame.conj().T @ frame - np.eye(frame.shape[1])))),
            )
    return residual


def _edge_diagnostics(
    eigenvalues: np.ndarray, eigenvectors: np.ndarray,
    previous_cluster: np.ndarray | None, wall_mask: np.ndarray,
    b_diagonal: np.ndarray,
) -> tuple[float, float, np.ndarray, np.ndarray, float, np.ndarray]:
    order = np.argsort(eigenvalues, axis=None)
    lower_flat, upper_flat = order[CONTROL_NX * CONTROL_NY - 1 : CONTROL_NX * CONTROL_NY + 1]
    lower = np.unravel_index(lower_flat, eigenvalues.shape)
    upper = np.unravel_index(upper_flat, eigenvalues.shape)
    sorted_values = eigenvalues.ravel()[order]
    internal = float(sorted_values[800] - sorted_values[799])
    external = float(min(sorted_values[799] - sorted_values[798], sorted_values[801] - sorted_values[800]))

    phases = np.exp(
        2j * np.pi * np.outer(np.arange(CONTROL_NY), np.arange(CONTROL_NY))
        / CONTROL_NY
    ) / np.sqrt(CONTROL_NY)
    vectors: list[np.ndarray] = []
    for momentum, band in (lower, upper):
        block = eigenvectors[momentum, :, band]
        vectors.append(
            (phases[:, momentum, None] * block[None, :]).reshape(1600)
        )
    cluster = np.column_stack(vectors)
    b_values, wall_modes = core._wall_modes(
        cluster, b_diagonal, previous_cluster, None
    )
    wall_weights = np.real(np.sum(np.abs(wall_modes[wall_mask, :]) ** 2, axis=0))
    link = 1.0 if previous_cluster is None else float(
        np.min(np.linalg.svd(previous_cluster.conj().T @ cluster, compute_uv=False))
    )
    return internal, external, b_values, wall_weights, link, cluster


def _source_paths(output_root: Path, source_kind: str, wall: str) -> tuple[Path, Path]:
    result = Path(output_root) / "control_sources" / source_kind / wall / "source.npz"
    return result, result.with_suffix(".completion.json")


def _source_identity(source_kind: str, wall: str, hashes: dict[str, str]) -> dict[str, Any]:
    return {
        "schema": EXACT_SOURCE_COMPLETION_SCHEMA,
        "source_kind": source_kind, "wall": wall,
        "Nx": CONTROL_NX, "Ny": CONTROL_NY, "rank": CONTROL_NX * CONTROL_NY,
        "construction": (
            "complex_conjugate_of_topological_ground_projector"
            if source_kind == "conjugated"
            else "canonical_exact_flattened_ground_projector"
        ),
        "nshell": 1, "alpha_1": 30.0 if source_kind == "trivial" else 1.0,
        "alpha_2": 30.0, "dw_truncation": wall == "hard",
        "source_hashes": hashes,
    }


def _verify_source(
    output_root: Path, source_kind: str, wall: str, hashes: dict[str, str]
) -> core.EndpointRef | None:
    result, completion_path = _source_paths(output_root, source_kind, wall)
    if not result.is_file() or not completion_path.is_file():
        return None
    try:
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        for key, value in _source_identity(source_kind, wall, hashes).items():
            if completion.get(key) != value:
                return None
        record = completion["result"]
        if record != {
            "name": result.name, "bytes": result.stat().st_size,
            "sha256": core.sha256_path(result),
        }:
            return None
        with np.load(result, allow_pickle=False) as saved:
            if str(np.asarray(saved["schema"]).item()) != EXACT_SOURCE_SCHEMA:
                return None
            frame = np.asarray(saved["frame"])
            if frame.dtype != np.complex128 or frame.shape != (1600, 800):
                return None
            if json.loads(str(np.asarray(saved["metadata_json"]).item())) != _source_identity(
                source_kind, wall, hashes
            ):
                return None
        return core.EndpointRef(
            str(result), str(completion_path), None, record["sha256"], record["bytes"],
            core.sha256_path(completion_path), "exact-control-v1", completion["schema"],
        )
    except Exception:
        return None


def ensure_exact_sources(output_root: Path, hashes: dict[str, str]) -> dict[tuple[str, str], core.EndpointRef]:
    refs: dict[tuple[str, str], core.EndpointRef] = {}
    for wall in CONTROL_WALLS:
        topological_frame: np.ndarray | None = None
        for source_kind in CONTROL_SOURCE_KINDS:
            existing = _verify_source(output_root, source_kind, wall, hashes)
            if existing is not None:
                refs[(source_kind, wall)] = existing
                if source_kind == "topological":
                    topological_frame = core.load_endpoint(existing, CONTROL_NX, CONTROL_NY)
                continue
            if source_kind == "conjugated":
                if topological_frame is None:
                    topological_frame = core.load_endpoint(
                        refs[("topological", wall)], CONTROL_NX, CONTROL_NY
                    )
                frame = topological_frame.conj()
            else:
                hamiltonian = _exact_hamiltonian(
                    wall=wall, source_kind=source_kind, phi=0.0
                )
                _, vectors = np.linalg.eigh(hamiltonian)
                frame = np.asarray(vectors[:, : CONTROL_NX * CONTROL_NY], dtype=np.complex128)
                if source_kind == "topological":
                    topological_frame = frame
            identity = _source_identity(source_kind, wall, hashes)
            result, completion_path = _source_paths(output_root, source_kind, wall)
            payload = {
                "schema": np.asarray(EXACT_SOURCE_SCHEMA), "frame": frame,
                "rank": np.asarray(CONTROL_NX * CONTROL_NY, dtype=np.int64),
                "metadata_json": np.asarray(core.canonical_json(identity)),
            }
            core._atomic_npz(result, payload)
            record = {
                "name": result.name, "bytes": result.stat().st_size,
                "sha256": core.sha256_path(result),
            }
            core._atomic_json(completion_path, {**identity, "result": record})
            ref = _verify_source(output_root, source_kind, wall, hashes)
            if ref is None:
                raise RuntimeError(f"new exact source did not verify: {source_kind}/{wall}")
            refs[(source_kind, wall)] = ref
    return refs


def _exact_direction(
    task: dict[str, Any], config: dict[str, Any], sigma: int, queue: Any,
) -> dict[str, Any]:
    intervals = int(task["grid_intervals"])
    phi = core.flux_grid(intervals, float(config["flux"]["regulator"]), sigma)
    source_kind = str(task["source_kind"])
    wall = str(task["wall"])
    twist_gauge = str(task["twist_gauge"])
    rank = CONTROL_NX * CONTROL_NY
    mode_x, y, _ = core._coordinates(CONTROL_NX, CONTROL_NY)
    b_diagonal = np.where(mode_x < CONTROL_NX // 2, -1.0, 1.0)
    radius = int(task.get("wall_window", 2))
    wall_centers = (CONTROL_NX // 4, 3 * CONTROL_NX // 4)
    wall_mask = (
        (core._periodic_distance(mode_x, wall_centers[0], CONTROL_NX) <= radius)
        | (core._periodic_distance(mode_x, wall_centers[1], CONTROL_NX) <= radius)
    )

    left = np.empty(phi.size)
    right = np.empty(phi.size)
    density_x = np.empty((phi.size, CONTROL_NX))
    instant_left = np.empty(phi.size)
    instant_right = np.empty(phi.size)
    principal = np.ones(phi.size)
    assigned = np.ones(phi.size)
    projector_residual = np.empty(phi.size)
    charge_residual = np.empty(phi.size)
    internal = np.empty(phi.size)
    external = np.empty(phi.size)
    b_values = np.empty((phi.size, 2))
    wall_weights = np.empty((phi.size, 2))
    links = np.ones(phi.size)
    cache: list[tuple[np.ndarray, np.ndarray]] = []
    occupied: np.ndarray | None = None
    previous: np.ndarray | None = None
    previous_cluster: np.ndarray | None = None
    start_frame: np.ndarray | None = None
    endpoint_frame: np.ndarray | None = None
    h_start: np.ndarray | None = None
    h_end: np.ndarray | None = None
    maximum_block_off_diagonal = 0.0
    maximum_seam_restoration_error = 0.0

    for point, value in enumerate(phi):
        hamiltonian, eigenvalues, raw_vectors, off_diagonal, gauge_error = _exact_diagonalize(
            float(value), wall, source_kind, twist_gauge
        )
        maximum_block_off_diagonal = max(maximum_block_off_diagonal, off_diagonal)
        maximum_seam_restoration_error = max(maximum_seam_restoration_error, gauge_error)
        h_start = hamiltonian if point == 0 else h_start
        h_end = hamiltonian
        if occupied is None:
            occupied = _lowest_occupations(eigenvalues)
            tracked = np.array(raw_vectors, copy=True)
        else:
            assert previous is not None
            tracked, assigned[point], principal[point] = _continue_exact_basis(
                previous, eigenvalues, raw_vectors
            )
        cache.append((eigenvalues, raw_vectors))
        current_occupied = _lowest_occupations(eigenvalues)
        density_x[point] = _block_density_x(tracked, occupied)
        instantaneous_density = _block_density_x(raw_vectors, current_occupied)
        left[point] = float(density_x[point, : CONTROL_NX // 2].sum())
        right[point] = float(density_x[point, CONTROL_NX // 2 :].sum())
        instant_left[point] = float(
            instantaneous_density[: CONTROL_NX // 2].sum()
        )
        instant_right[point] = float(
            instantaneous_density[CONTROL_NX // 2 :].sum()
        )
        projector_residual[point] = _block_projector_residual(tracked, occupied)
        charge_residual[point] = abs(left[point] + right[point] - rank)
        (
            internal[point], external[point], b_values[point], wall_weights[point],
            links[point], previous_cluster,
        ) = _edge_diagnostics(
            eigenvalues, raw_vectors, previous_cluster, wall_mask, b_diagonal
        )
        previous = tracked
        if point == 0:
            start_frame = _block_frame(tracked, occupied)
        if point == phi.size - 1:
            endpoint_frame = _block_frame(tracked, occupied)
        if queue is not None:
            queue.put(1)

    assert occupied is not None and previous is not None
    assert start_frame is not None and endpoint_frame is not None
    assert h_start is not None and h_end is not None
    links[:-1] = np.minimum(links[:-1], links[1:])

    # A genuine undo follows the cached eigensystems backward from the
    # transported endpoint.  No Hamiltonian reconstruction or branch reset is
    # allowed on this leg.
    returned = np.array(previous, copy=True)
    if queue is not None:
        queue.put(1)
    for eigenvalues, raw_vectors in cache[-2::-1]:
        returned, _, _ = _continue_exact_basis(returned, eigenvalues, raw_vectors)
        if queue is not None:
            queue.put(1)
    returned_frame = _block_frame(returned, occupied)
    returned_projector = returned_frame @ returned_frame.conj().T
    start_projector = start_frame @ start_frame.conj().T
    undo_error = float(np.max(np.abs(returned_projector - start_projector)))

    gauge = np.exp(1j * sigma * 2.0 * np.pi * y / CONTROL_NY)
    gauged_start = gauge[:, None] * h_start * gauge.conj()[None, :]
    parent_error = float(np.max(np.abs(h_end - gauged_start)))
    endpoint = core._defect_diagnostics(
        endpoint_frame, start_projector, sigma, mode_x, y, CONTROL_NX, CONTROL_NY
    )
    delta_left = left - left[0]
    delta_right = right - right[0]
    instant_delta_left = instant_left - instant_left[0]
    instant_delta_right = instant_right - instant_right[0]
    minimum = int(np.argmin(internal))
    active = np.zeros(phi.size, dtype=bool)
    active[minimum] = source_kind != "trivial"
    point_valid = np.zeros(phi.size, dtype=bool)
    point_valid[minimum] = source_kind != "trivial"
    return {
        "phi": phi,
        "N_left": left,
        "N_right": right,
        "delta_N_left": delta_left,
        "delta_N_right": delta_right,
        "density_x": density_x,
        # For the exact control the momentum-resolved overlap continuation is
        # the ordinary spectral-flow construction.  It is intentionally kept
        # distinct from independently refilling the lowest energies.
        "ordinary_delta_N_left": delta_left,
        "ordinary_delta_N_right": delta_right,
        "instantaneous_delta_N_left": instant_delta_left,
        "instantaneous_delta_N_right": instant_delta_right,
        "principal_overlap": principal,
        "selected_weight_floor": assigned * assigned,
        "ordinary_principal_overlap": principal,
        "ordinary_selected_weight_floor": assigned * assigned,
        "projector_residual": projector_residual,
        "total_charge_residual": charge_residual,
        "edge_internal_gap": internal,
        "edge_external_gap": external,
        "edge_B_eigenvalues": b_values,
        "edge_combined_wall_weight": wall_weights,
        "edge_link_min_singular": links,
        "edge_active_mask": active,
        "edge_point_valid": point_valid,
        "resolved": source_kind != "trivial",
        "reason": "" if source_kind != "trivial" else "matched_trivial_control",
        "minimum": minimum,
        "start": minimum if source_kind != "trivial" else -1,
        "end": minimum if source_kind != "trivial" else -1,
        "entering_labels": np.asarray([-1], dtype=np.int8),
        "parent_error": parent_error,
        "undo_error": undo_error,
        "endpoint": endpoint,
        "maximum_block_off_diagonal": maximum_block_off_diagonal,
        "maximum_seam_restoration_error": maximum_seam_restoration_error,
    }


def compute_exact_control(
    task: dict[str, Any], config: dict[str, Any], *, queue: Any = None,
) -> dict[str, Any]:
    """Compute one two-direction exact equilibrium control pair."""
    if task.get("stage") != "control":
        raise ValueError("compute_exact_control accepts only exact control tasks")
    if int(task["Nx"]) != CONTROL_NX or int(task["Ny"]) != CONTROL_NY:
        raise ValueError("exact control geometry changed")
    source_kind = str(task["source_kind"])
    _, zero_values, zero_vectors, _, _ = _exact_diagonalize(
        0.0, str(task["wall"]), source_kind, "uniform"
    )
    zero_occupied = _lowest_occupations(zero_values)
    source_frame = _block_frame(zero_vectors, zero_occupied)
    rank = source_frame.shape[1]
    mode_x, _, _ = core._coordinates(CONTROL_NX, CONTROL_NY)
    source_left, source_right, source_density_x = core._frame_observables(
        source_frame, mode_x, CONTROL_NX
    )
    gram = float(np.max(np.abs(source_frame.conj().T @ source_frame - np.eye(rank))))
    source_projector = source_frame @ source_frame.conj().T
    input_projector_residual = float(
        np.max(np.abs(source_projector @ source_projector - source_projector))
    )
    c_by_y0 = core.real_space_chern_by_y0(
        source_frame, CONTROL_NX, CONTROL_NY, float(config["chern_control"]["radius"])
    )
    payloads = [
        _exact_direction(task, config, int(config["flux"]["directions"][direction]), queue)
        for direction in core.DIRECTIONS
    ]
    stack = lambda key: np.stack([np.asarray(row[key]) for row in payloads])
    n_left, n_right = stack("N_left"), stack("N_right")
    delta_left, delta_right = stack("delta_N_left"), stack("delta_N_right")
    ordinary_left = stack("ordinary_delta_N_left")
    ordinary_right = stack("ordinary_delta_N_right")
    instant_left = stack("instantaneous_delta_N_left")
    instant_right = stack("instantaneous_delta_N_right")
    arrays: dict[str, Any] = {
        "schema": np.asarray(core.RESULT_SCHEMA),
        "directions": np.asarray(core.DIRECTIONS),
        "sigma": np.asarray([1, -1], dtype=np.int8),
        "phi": stack("phi"),
        "N_left": n_left,
        "N_right": n_right,
        "N_total": n_left + n_right,
        "delta_N_left": delta_left,
        "delta_N_right": delta_right,
        "delta_N_total": delta_left + delta_right,
        "q_x": 0.5 * (delta_right - delta_left),
        "density_x": stack("density_x"),
        "ordinary_delta_N_left": ordinary_left,
        "ordinary_delta_N_right": ordinary_right,
        "ordinary_delta_N_total": ordinary_left + ordinary_right,
        "ordinary_q_x": 0.5 * (ordinary_right - ordinary_left),
        "instantaneous_delta_N_left": instant_left,
        "instantaneous_delta_N_right": instant_right,
        "instantaneous_delta_N_total": instant_left + instant_right,
        "instantaneous_q_x": 0.5 * (instant_right - instant_left),
        "source_N_left": np.asarray(source_left),
        "source_N_right": np.asarray(source_right),
        "source_density_x": source_density_x,
        "source_real_space_chern_by_y0": c_by_y0,
        "source_real_space_chern_mean": np.asarray(float(np.mean(c_by_y0))),
        "source_real_space_chern_std": np.asarray(float(np.std(c_by_y0, ddof=1))),
        "source_real_space_chern_xref": np.asarray(CONTROL_NX // 2, dtype=np.int64),
        "source_real_space_chern_radius": np.asarray(float(config["chern_control"]["radius"])),
        "edge_internal_gap": stack("edge_internal_gap"),
        "edge_external_gap": stack("edge_external_gap"),
        "edge_B_eigenvalues": stack("edge_B_eigenvalues"),
        "edge_combined_wall_weight": stack("edge_combined_wall_weight"),
        "edge_link_min_singular": stack("edge_link_min_singular"),
        "edge_active_mask": stack("edge_active_mask").astype(bool),
        "edge_point_valid": stack("edge_point_valid").astype(bool),
        "edge_minimum_gap_index": np.asarray([row["minimum"] for row in payloads], dtype=np.int64),
        "edge_active_start_index": np.asarray([row["start"] for row in payloads], dtype=np.int64),
        "edge_active_end_index": np.asarray([row["end"] for row in payloads], dtype=np.int64),
        "edge_entering_wall_label": np.asarray([-1, -1], dtype=np.int8),
        "edge_entering_wall_labels": np.full((2, 1), -1, dtype=np.int8),
        "resolved": np.asarray([row["resolved"] for row in payloads], dtype=bool),
        "unresolved_reason": np.asarray([row["reason"] for row in payloads]),
        "principal_overlap": stack("principal_overlap"),
        "selected_weight_floor": stack("selected_weight_floor"),
        "ordinary_principal_overlap": stack("ordinary_principal_overlap"),
        "ordinary_selected_weight_floor": stack("ordinary_selected_weight_floor"),
        "endpoint_defect_eigenvalues": np.stack([row["endpoint"]["eigenvalues"] for row in payloads]),
        "endpoint_particle_density_x": np.stack([row["endpoint"]["particle_density_x"] for row in payloads]),
        "endpoint_hole_density_x": np.stack([row["endpoint"]["hole_density_x"] for row in payloads]),
        "endpoint_leading_positive_eigenvalue": np.asarray([row["endpoint"]["leading_positive_eigenvalue"] for row in payloads]),
        "endpoint_leading_negative_eigenvalue": np.asarray([row["endpoint"]["leading_negative_eigenvalue"] for row in payloads]),
        "endpoint_leading_particle_mode_density_x": np.stack([row["endpoint"]["leading_particle_mode_density_x"] for row in payloads]),
        "endpoint_leading_hole_mode_density_x": np.stack([row["endpoint"]["leading_hole_mode_density_x"] for row in payloads]),
        "endpoint_leading_particle_wall_weights": np.stack([row["endpoint"]["leading_particle_wall_weights"] for row in payloads]),
        "endpoint_leading_hole_wall_weights": np.stack([row["endpoint"]["leading_hole_wall_weights"] for row in payloads]),
        "endpoint_positive_defect_count_above_0p9": np.asarray([row["endpoint"]["positive_count_above_0p9"] for row in payloads], dtype=np.int8),
        "endpoint_negative_defect_count_below_minus_0p9": np.asarray([row["endpoint"]["negative_count_below_minus_0p9"] for row in payloads], dtype=np.int8),
        "endpoint_maximum_remaining_abs_defect_eigenvalue": np.asarray([row["endpoint"]["maximum_remaining_abs_eigenvalue"] for row in payloads]),
        "multicut_positions": payloads[0]["endpoint"]["multicut_positions"],
        "multicut_q_x": np.stack([row["endpoint"]["multicut_q_x"] for row in payloads]),
        "center_of_charge_displacement": np.asarray([row["endpoint"]["center_displacement"] for row in payloads]),
        "total_charge_residual": stack("total_charge_residual"),
        "projector_residual": stack("projector_residual"),
        "input_frame_gram_residual": np.asarray(gram),
        "input_projector_residual": np.asarray(input_projector_residual),
        "large_gauge_parent_error": np.asarray([row["parent_error"] for row in payloads]),
        "continuation_undo_error": np.asarray([row["undo_error"] for row in payloads]),
        "rank": np.asarray(rank, dtype=np.int64),
        "exact_block_off_diagonal_maximum": np.asarray(max(row["maximum_block_off_diagonal"] for row in payloads)),
        "exact_seam_restoration_error": np.asarray(max(row["maximum_seam_restoration_error"] for row in payloads)),
    }
    near_unit = np.abs(arrays["q_x"][:, -1]) > 0.9
    single_pair = (
        (arrays["endpoint_positive_defect_count_above_0p9"] == 1)
        & (arrays["endpoint_negative_defect_count_below_minus_0p9"] == 1)
        & (arrays["endpoint_maximum_remaining_abs_defect_eigenvalue"] < 1e-6)
    )
    particle_wall = np.asarray(arrays["endpoint_leading_particle_wall_weights"])
    hole_wall = np.asarray(arrays["endpoint_leading_hole_wall_weights"])
    localized_opposite = (
        (np.max(particle_wall, axis=1) > 0.8)
        & (np.max(hole_wall, axis=1) > 0.8)
        & (np.argmax(particle_wall, axis=1) != np.argmax(hole_wall, axis=1))
    )
    arrays["endpoint_near_unit_event"] = near_unit
    arrays["endpoint_single_defect_pair"] = single_pair
    arrays["endpoint_defect_modes_opposite_wall_localized"] = localized_opposite
    arrays["endpoint_near_unit_defect_validation_pass"] = (~near_unit) | (
        single_pair & localized_opposite
    )
    core.validate_result_arrays(arrays, task, config)
    return arrays


def _exact_worker(payload: tuple[Any, ...]) -> dict[str, Any]:
    task, _source, ref, config, output_text, config_hash, hashes, queue = payload
    started = time.perf_counter()
    output_root = Path(output_text)
    log_path = output_root / "logs/tasks" / f"{task['task_id']}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        # The source pair is a provenance dependency.  Reopen and hash it even
        # though the exact path itself is rebuilt directly at every flux.
        core.load_endpoint(ref, CONTROL_NX, CONTROL_NY)
        with core.threadpool_limits(limits=int(config["execution"]["blas_threads"])):
            with log_path.open("a", encoding="utf-8") as log, \
                    contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                arrays = compute_exact_control(task, config, queue=queue)
        core.publish_pair(
            output_root, task, arrays, config_hash, hashes, ref,
            time.perf_counter() - started,
        )
        core.failure_path(output_root, task).unlink(missing_ok=True)
        return {"ok": True, "task_id": task["task_id"]}
    except BaseException as exc:
        core._record_failure(output_root, task, exc)
        return {
            "ok": False,
            "task_id": task["task_id"],
            "error": f"{type(exc).__name__}: {exc}",
        }


def exact_control_tasks() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for wall in CONTROL_WALLS:
        for intervals in (128, 256, 512):
            rows.append(_control_task(
                wall, "topological", f"topological_M{intervals}", intervals,
                "uniform", "primary" if intervals == 256 else "mesh_sensitivity",
            ))
        rows.append(_control_task(
            wall, "topological", "topological_seam_M256", 256,
            "seam", "gauge_seam",
        ))
        rows.append(_control_task(wall, "trivial", "trivial_M256", 256, "uniform", "trivial"))
        rows.append(_control_task(
            wall, "conjugated", "conjugated_M256", 256, "uniform", "conjugated"
        ))
    if len(rows) != 12 or len({row["task_id"] for row in rows}) != 12:
        raise RuntimeError("exact control table must contain 12 unique endpoint pairs")
    return rows


def _control_task(
    wall: str, source_kind: str, variant: str, intervals: int,
    twist_gauge: str, control_kind: str,
) -> dict[str, Any]:
    return {
        "stage": "control", "task_id": f"control_{variant}_{wall}",
        "cell": f"exact_{source_kind}_N20x40", "protocol": f"exact_{source_kind}",
        "size": "N20x40", "Nx": 20, "Ny": 40, "wall": wall, "sample_id": 0,
        "grid_intervals": intervals, "edge_block_rank": 2, "wall_window": 2,
        "twist_gauge": twist_gauge, "variant": variant,
        "result_collection": "controls", "is_primary": False,
        "control_kind": control_kind, "source_kind": source_kind,
        "control_pair_id": f"exact_{wall}_M{intervals}",
    }


def sensitivity_tasks(config: dict[str, Any]) -> list[dict[str, Any]]:
    base_ids = set(range(0, 100, 4))
    rows: list[dict[str, Any]] = []
    for primary in core.tasks(config, include_bridge=False):
        selected = int(primary["sample_id"]) in base_ids
        if primary["cell"] == "nsh1_N20x24":
            selected = selected or int(primary["sample_id"]) in HISTORICAL_FLIPS[primary["wall"]]
        if not selected:
            continue
        for variant, settings in SENSITIVITY_VARIANTS.items():
            rows.append({
                **primary, **settings,
                "stage": "sensitivity",
                "task_id": f"sensitivity_{variant}_{primary['cell']}_{primary['wall']}_sample_{primary['sample_id']:03d}",
                "variant": variant, "result_collection": "sensitivity",
                "is_primary": False, "control_kind": f"sensitivity_{variant}",
                "primary_task_id": primary["task_id"],
            })
    if len(rows) != 2050 or len({row["task_id"] for row in rows}) != 2050:
        raise RuntimeError(f"sensitivity table must contain 2,050 unique pairs, found {len(rows)}")
    return rows


def _filters(
    rows: Iterable[dict[str, Any]], cells: set[str] | None, walls: set[str] | None,
    sample_ids: set[int] | None, variants: set[str] | None,
) -> list[dict[str, Any]]:
    return [row for row in rows if (
        (cells is None or row["cell"] in cells)
        and (walls is None or row["wall"] in walls)
        and (sample_ids is None or int(row["sample_id"]) in sample_ids)
        and (variants is None or row["variant"] in variants)
    )]


def resolve_sensitivity_refs(
    rows: Iterable[dict[str, Any]], sources: dict[str, dict[str, Any]],
    new_endpoint_root: Path,
) -> tuple[dict[str, core.EndpointRef], dict[str, str]]:
    """Verify each immutable endpoint once, then share it across variants.

    Every preregistered endpoint normally appears in five sensitivity tasks.
    Reopening the same compressed five-sample shard for every variant can
    transiently allocate hundreds of megabytes repeatedly during report-only
    discovery.  The endpoint dependency is identical across those variants,
    so one checksum/identity verification per (cell, wall, sample) is both
    authoritative and substantially cheaper.
    """
    refs: dict[str, core.EndpointRef] = {}
    missing: dict[str, str] = {}
    cache: dict[tuple[str, str, int], core.EndpointRef | str] = {}
    for task in rows:
        key = (str(task["cell"]), str(task["wall"]), int(task["sample_id"]))
        if key not in cache:
            try:
                cache[key] = core.endpoint_ref(
                    sources[key[0]], key[1], key[2], Path(new_endpoint_root)
                )
            except Exception as exc:
                cache[key] = f"{type(exc).__name__}: {exc}"
        cached = cache[key]
        if isinstance(cached, core.EndpointRef):
            refs[task["task_id"]] = cached
        else:
            missing[task["task_id"]] = cached
    return refs, missing


def _run(
    rows: list[dict[str, Any]], refs: dict[str, core.EndpointRef], config: dict[str, Any],
    output_root: Path, workers: int, resume: bool, hashes: dict[str, str],
) -> None:
    config_hash = core.scientific_config_hash(config)
    verified: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    for task in rows:
        ok, _, _ = core.verify_pair(
            output_root, task, config, config_hash, hashes, refs[task["task_id"]]
        )
        (verified if ok else pending).append(task)
    run_rows = pending if resume else rows
    print(f"[resume] verified={len(verified)}/{len(rows)} run={len(run_rows)}", flush=True)
    if not run_rows:
        return
    if all(task.get("stage") == "control" for task in run_rows):
        worker = _exact_worker
    elif all(task.get("stage") == "sensitivity" for task in run_rows):
        worker = core._worker
    else:
        raise RuntimeError("control and sensitivity tasks cannot share one worker pool")
    context = mp.get_context("spawn")
    with context.Manager() as manager:
        queue = manager.Queue()
        total_points = sum(4 * (int(task["grid_intervals"]) + 1) for task in rows)
        verified_points = sum(
            4 * (int(task["grid_intervals"]) + 1) for task in verified
        ) if resume else 0
        with tqdm(total=len(rows), initial=(len(verified) if resume else 0), desc="control/sensitivity pairs") as task_bar, \
                tqdm(total=total_points, initial=verified_points, desc="continued flux points") as point_bar, \
                ProcessPoolExecutor(max_workers=min(workers, len(run_rows)), mp_context=context) as pool:
            futures = {
                pool.submit(worker, (
                    task, {}, refs[task["task_id"]], config, str(output_root),
                    config_hash, hashes, queue,
                )) for task in run_rows
            }
            failures: list[dict[str, Any]] = []
            while futures:
                done, futures = wait(futures, timeout=0.25, return_when=FIRST_COMPLETED)
                while True:
                    try:
                        point_bar.update(int(queue.get_nowait()))
                    except Exception:
                        break
                for future in done:
                    result = future.result()
                    task_bar.update(1)
                    if not result["ok"]:
                        failures.append(result)
                        task_bar.write(f"[failure] {result['task_id']}: {result['error']}")
            if failures:
                raise RuntimeError(f"{len(failures)} control/sensitivity tasks failed")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=("report-controls", "controls", "report-sensitivity", "sensitivity")
    )
    parser.add_argument("--config", type=Path, default=core.DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--new-endpoint-root", type=Path, default=core.DEFAULT_NEW_ENDPOINT_ROOT)
    parser.add_argument("--workers", type=int, default=28)
    parser.add_argument("--cells", nargs="*")
    parser.add_argument("--walls", nargs="*", choices=CONTROL_WALLS)
    parser.add_argument("--sample-ids", nargs="*", type=int)
    parser.add_argument("--variants", nargs="*")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = core.load_config(args.config.resolve())
    core.validate_config(config)
    output_root = args.output_root.resolve()
    hashes = {
        **core.source_hashes(),
        "control_runner": core.sha256_path(Path(__file__).resolve()),
        "cpu_engine": core.sha256_path(REPO_ROOT / "src/fgtn/classA_U1FGTN.py"),
    }
    if "controls" in args.stage:
        rows = exact_control_tasks()
        print("[controls] exact Nx=20 Ny=40: topological M128/256/512, seam, trivial, conjugated")
        if args.stage == "report-controls":
            print(f"[tasks] {len(rows)} result pairs; source construction deferred")
            return 0
        source_refs = ensure_exact_sources(output_root, hashes)
        refs = {
            task["task_id"]: source_refs[(task["source_kind"], task["wall"])] for task in rows
        }
    else:
        rows = sensitivity_tasks(config)
        rows = _filters(
            rows,
            None if args.cells is None else set(args.cells),
            None if args.walls is None else set(args.walls),
            None if args.sample_ids is None else set(args.sample_ids),
            None if args.variants is None else set(args.variants),
        )
        sources = core.source_rows(config)
        refs, missing = resolve_sensitivity_refs(
            rows, sources, args.new_endpoint_root.resolve()
        )
        print(f"[sensitivity] selected={len(rows)} sources={len(refs)} missing={len(missing)}")
        if args.stage == "report-sensitivity":
            return 0
        if missing:
            raise RuntimeError(f"selected sensitivity table has {len(missing)} missing sources")
    output_root.mkdir(parents=True, exist_ok=True)
    _run(rows, refs, config, output_root, args.workers, args.resume, hashes)
    print("[complete] requested control/sensitivity stage finished", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
