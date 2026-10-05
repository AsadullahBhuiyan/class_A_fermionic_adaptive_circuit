"""Torch/CUDA eigensolver backend for the frozen-record spectral-pump core.

The scientific continuation and result construction remain in
``spectral_cpu_reference``.  This module replaces only the dense Hermitian
eigensolver and the endpoint-defect diagonalization.  Eigenvectors return to
NumPy because the continuation is intrinsically sequential and the reference
implementation is the acceptance oracle.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def require_a100() -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this bundle")
    name = torch.cuda.get_device_name(0)
    properties = torch.cuda.get_device_properties(0)
    total = int(properties.total_memory)
    if "A100" not in name.upper() or total < 38 * 1024**3:
        raise RuntimeError(
            f"a 40-GB-class A100 is required; found {name} with "
            f"{total / 1024**3:.1f} GiB"
        )
    torch.set_default_dtype(torch.float64)
    return {"name": name, "total_memory_bytes": total}


def install(core: Any, device: str = "cuda:0") -> None:
    """Install the CUDA complex128 eigensolvers into a core module."""

    dev = torch.device(device)
    dense_cache: dict[str, Any] = {}

    def parent_eigensystem(
        h0: np.ndarray,
        dy: np.ndarray,
        phi: float,
        ny: int,
        y: np.ndarray,
        twist_gauge: str,
        subset: tuple[int, int] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        # Constructing the phase on CUDA avoids moving another dense complex
        # matrix after h0.  The full spectrum is used even for the small edge
        # window because torch.linalg.eigh has no subset-by-index interface.
        if dense_cache.get("h0_object") is not h0:
            dense_cache.clear()
            dense_cache["h0_object"] = h0
            dense_cache["h"] = torch.as_tensor(
                h0, dtype=torch.complex128, device=dev
            )
            dense_cache["dy"] = torch.as_tensor(
                dy, dtype=torch.float64, device=dev
            )
            dense_cache["y"] = torch.as_tensor(
                y, dtype=torch.float64, device=dev
            )
        h = dense_cache["h"]
        delta_y = dense_cache["dy"]
        matrix = h * torch.exp(1j * float(phi) * delta_y / int(ny))
        matrix = 0.5 * (matrix + matrix.mH)
        if twist_gauge == "seam":
            yy = dense_cache["y"]
            basis = torch.exp(1j * float(phi) * yy / int(ny))
            matrix = basis.conj()[:, None] * matrix * basis[None, :]
            matrix = 0.5 * (matrix + matrix.mH)
        elif twist_gauge != "uniform":
            raise ValueError("twist_gauge must be uniform or seam")
        values, vectors = torch.linalg.eigh(matrix)
        if twist_gauge == "seam":
            yy = dense_cache["y"]
            basis = torch.exp(1j * float(phi) * yy / int(ny))
            vectors = basis[:, None] * vectors
        if subset is not None:
            lower, upper = map(int, subset)
            values = values[lower : upper + 1]
            vectors = vectors[:, lower : upper + 1]
        return values.cpu().numpy(), vectors.cpu().numpy()

    def ordinary_select(
        previous: np.ndarray,
        eigenvalues: np.ndarray,
        eigenvectors: np.ndarray,
        rank: int,
    ) -> tuple[np.ndarray, float, float]:
        prev = torch.as_tensor(previous, dtype=torch.complex128, device=dev)
        vec = torch.as_tensor(eigenvectors, dtype=torch.complex128, device=dev)
        weights = torch.sum(torch.abs(prev.mH @ vec) ** 2, dim=0).real
        # Eigenvalues already arrive in ascending order, so a stable descending
        # weight sort reproduces np.lexsort((eigenvalues, -weights)).
        order = torch.argsort(weights, descending=True, stable=True)
        selected = vec[:, order[: int(rank)]]
        singular = torch.linalg.svdvals(prev.mH @ selected)
        return (
            selected.cpu().numpy(),
            float(torch.min(singular).item()),
            float(torch.min(weights[order[: int(rank)]]).item()),
        )

    def spectator_select(
        previous: np.ndarray,
        eigenvalues: np.ndarray,
        eigenvectors: np.ndarray,
        rank: int,
        edge_indices: tuple[int, int],
    ) -> np.ndarray:
        keep = np.ones(eigenvalues.size, dtype=bool)
        keep[list(edge_indices)] = False
        candidates_np = eigenvectors[:, keep]
        prev = torch.as_tensor(previous, dtype=torch.complex128, device=dev)
        candidates = torch.as_tensor(candidates_np, dtype=torch.complex128, device=dev)
        weights = torch.sum(torch.abs(prev.mH @ candidates) ** 2, dim=0).real
        order = torch.argsort(weights, descending=True, stable=True)
        return candidates[:, order[: int(rank)]].cpu().numpy()

    def defect_diagnostics(
        endpoint_frame: np.ndarray,
        projector0: np.ndarray,
        sigma: int,
        mode_x: np.ndarray,
        y: np.ndarray,
        nx: int,
        ny: int,
    ) -> dict[str, Any]:
        # This is one more dense diagonalization per direction; keep it on the
        # GPU, then use the reference NumPy reduction on its eigenpairs.
        frame = torch.as_tensor(endpoint_frame, dtype=torch.complex128, device=dev)
        p0 = torch.as_tensor(projector0, dtype=torch.complex128, device=dev)
        yy = torch.as_tensor(y, dtype=torch.float64, device=dev)
        gauge = torch.exp(1j * int(sigma) * 2.0 * np.pi * yy / int(ny))
        pend = frame @ frame.mH
        unwrapped = gauge.conj()[:, None] * pend * gauge[None, :]
        defect = 0.5 * (unwrapped - p0 + (unwrapped - p0).mH)
        values_t, vectors_t = torch.linalg.eigh(defect)
        values = values_t.cpu().numpy()
        vectors = vectors_t.cpu().numpy()
        defect_np = defect.cpu().numpy()
        positive, negative = values > 0.0, values < 0.0
        particle_mode = np.sum(
            np.abs(vectors[:, positive]) ** 2 * values[positive][None, :], axis=1
        )
        hole_mode = np.sum(
            np.abs(vectors[:, negative]) ** 2 * (-values[negative])[None, :], axis=1
        )
        particle_x = np.bincount(mode_x, weights=particle_mode, minlength=nx)
        hole_x = np.bincount(mode_x, weights=hole_mode, minlength=nx)
        leading_positive, leading_negative = int(np.argmax(values)), int(np.argmin(values))
        leading_particle_x = np.bincount(
            mode_x, weights=np.abs(vectors[:, leading_positive]) ** 2, minlength=nx
        )
        leading_hole_x = np.bincount(
            mode_x, weights=np.abs(vectors[:, leading_negative]) ** 2, minlength=nx
        )
        wall_centers = (nx // 4, 3 * nx // 4)
        wall_masks_x = np.stack([
            core._periodic_distance(np.arange(nx), center, nx) <= 2
            for center in wall_centers
        ])
        particle_wall_weights = wall_masks_x @ leading_particle_x
        hole_wall_weights = wall_masks_x @ leading_hole_x
        remaining = np.delete(values, (leading_negative, leading_positive))
        _, _, defect_density_x = core._projector_observables(defect_np, mode_x, nx)
        cuts = np.arange(1, nx, dtype=np.int64)
        multicut = np.asarray([
            0.5 * (defect_density_x[cut:].sum() - defect_density_x[:cut].sum())
            for cut in cuts
        ])
        centered_x = np.arange(nx, dtype=np.float64) - 0.5 * (nx - 1)
        return {
            "eigenvalues": values.astype(np.float64),
            "particle_density_x": particle_x.astype(np.float64),
            "hole_density_x": hole_x.astype(np.float64),
            "leading_positive_eigenvalue": float(values[leading_positive]),
            "leading_negative_eigenvalue": float(values[leading_negative]),
            "leading_particle_mode_density_x": leading_particle_x.astype(np.float64),
            "leading_hole_mode_density_x": leading_hole_x.astype(np.float64),
            "leading_particle_wall_weights": particle_wall_weights.astype(np.float64),
            "leading_hole_wall_weights": hole_wall_weights.astype(np.float64),
            "positive_count_above_0p9": int(np.count_nonzero(values > 0.9)),
            "negative_count_below_minus_0p9": int(np.count_nonzero(values < -0.9)),
            "maximum_remaining_abs_eigenvalue": float(
                np.max(np.abs(remaining), initial=0.0)
            ),
            "multicut_positions": cuts,
            "multicut_q_x": multicut,
            "center_displacement": float(centered_x @ defect_density_x),
            "defect": defect_np,
        }

    core._parent_eigensystem = parent_eigensystem
    core._defect_diagnostics = defect_diagnostics
    core._ordinary_select = ordinary_select
    core._spectator_select = spectator_select


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
