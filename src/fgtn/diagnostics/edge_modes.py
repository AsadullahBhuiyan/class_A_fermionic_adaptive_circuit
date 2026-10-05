from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


def _cell_indices(nx: int, x_values: Iterable[int], ny: int | None = None) -> np.ndarray:
    nx = int(nx)
    ys = (0,) if ny is None else range(int(ny))
    return np.asarray(
        [mu + 2 * int(x) + 2 * nx * int(y) for y in ys for x in x_values for mu in (0, 1)],
        dtype=np.int64,
    )


def _periodic_x_mask(nx: int, wall_x: int, width: int) -> np.ndarray:
    x = np.arange(int(nx), dtype=np.int64)
    distance = np.minimum((x - int(wall_x)) % int(nx), (int(wall_x) - x) % int(nx))
    return distance < int(width)


def _fix_column_phases(frame: np.ndarray) -> np.ndarray:
    frame = np.asarray(frame, dtype=np.complex128).copy()
    for column in range(frame.shape[1]):
        pivot = int(np.argmax(np.abs(frame[:, column])))
        value = frame[pivot, column]
        if abs(value) > 0.0:
            frame[:, column] *= np.exp(-1j * np.angle(value))
    return frame


def _symmetric_orthonormalize(frame: np.ndarray, *, rank_tol: float = 1e-12) -> np.ndarray:
    frame = np.asarray(frame, dtype=np.complex128)
    gram = 0.5 * (frame.conj().T @ frame + (frame.conj().T @ frame).conj().T)
    values, vectors = np.linalg.eigh(gram)
    if values.size == 0 or float(np.min(values)) <= float(rank_tol):
        raise ValueError(
            "The requested edge-frame columns become linearly dependent after projection; "
            f"Gram eigenvalues={values.tolist()}."
        )
    inverse_sqrt = (vectors * (1.0 / np.sqrt(values))[None, :]) @ vectors.conj().T
    return frame @ inverse_sqrt


def analytic_domain_wall_bloch_hamiltonian(
    model: Any,
    ky: float,
    *,
    periodic_x: bool = True,
) -> np.ndarray:
    """Return the exact Dirac/Chern domain-wall Hamiltonian at fixed ``k_y``.

    The returned basis is ``mu + 2*x``.  Its embedding into the full top-layer
    basis is ``mu + 2*x + 2*Nx*y``, matching :class:`classA_U1FGTN`.
    Translation invariance along ``y`` is checked rather than assumed silently.
    """

    nx, ny = int(model.Nx), int(model.Ny)
    if not hasattr(model, "alpha_profile"):
        raise ValueError("The model has no alpha_profile from which to build a domain wall.")
    alpha = np.asarray(model.alpha_profile, dtype=np.complex128)
    if alpha.shape != (nx, ny):
        raise ValueError(f"Expected alpha_profile shape {(nx, ny)}, got {alpha.shape}.")
    alpha_x = np.mean(alpha, axis=1)
    translation_error = float(np.max(np.abs(alpha - alpha_x[:, None])))
    if translation_error > 1e-12:
        raise ValueError(
            "A k_y-resolved edge frame requires y-translation invariance; "
            f"max alpha variation is {translation_error:.3e}."
        )

    sx = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    sy = np.asarray([[0.0, -1j], [1j, 0.0]], dtype=np.complex128)
    sz = np.asarray([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
    hop_x = -0.5 * sz - 0.5j * sx
    onsite_ky = np.sin(float(ky)) * sy - np.cos(float(ky)) * sz

    hamiltonian = np.zeros((2 * nx, 2 * nx), dtype=np.complex128)
    for x in range(nx):
        block = alpha_x[x] * sz + onsite_ky
        sl = slice(2 * x, 2 * x + 2)
        hamiltonian[sl, sl] += block
        xp = x + 1
        if xp < nx or periodic_x:
            xp %= nx
            slp = slice(2 * xp, 2 * xp + 2)
            hamiltonian[sl, slp] += hop_x
            hamiltonian[slp, sl] += hop_x.conj().T
    return 0.5 * (hamiltonian + hamiltonian.conj().T)


def _wall_resolve_degenerate_subspaces(
    energies: np.ndarray,
    vectors: np.ndarray,
    discriminator: np.ndarray,
    *,
    degeneracy_tol: float,
) -> tuple[np.ndarray, np.ndarray]:
    energies = np.asarray(energies, dtype=np.float64).copy()
    vectors = np.asarray(vectors, dtype=np.complex128).copy()
    start = 0
    while start < energies.size:
        stop = start + 1
        while stop < energies.size and abs(energies[stop] - energies[start]) <= degeneracy_tol:
            stop += 1
        if stop - start > 1:
            subspace = vectors[:, start:stop]
            projected = subspace.conj().T @ (discriminator[:, None] * subspace)
            projected = 0.5 * (projected + projected.conj().T)
            _, rotation = np.linalg.eigh(projected)
            vectors[:, start:stop] = subspace @ rotation
        start = stop
    return energies, vectors


def _select_wall_state(
    energies: np.ndarray,
    vectors: np.ndarray,
    target_mask: np.ndarray,
    opposite_mask: np.ndarray,
    *,
    candidate_count: int,
) -> dict[str, Any]:
    requested_count = min(int(candidate_count), energies.size)
    # The wall-resolved low-energy sector is a two-state doublet, one state per
    # periodic domain wall. ``candidate_count`` is retained as a compatibility
    # guard, but bulk states must never enter the localization rotation.
    count = min(2, requested_count)
    candidates = np.argsort(np.abs(energies), kind="stable")[:count]
    # A finite cylinder weakly hybridizes the two nominal wall states at their
    # crossing. The resulting symmetric/antisymmetric eigenvectors each have
    # half their weight on either wall even though their two-dimensional span
    # is exponentially close to the physical left/right edge subspace. Resolve
    # that span with the wall discriminator before choosing a state. Away from
    # the crossing this rotation is numerically the identity.
    subspace = vectors[:, candidates]
    discriminator = target_mask.astype(np.float64) - opposite_mask.astype(np.float64)
    projected = subspace.conj().T @ (discriminator[:, None] * subspace)
    projected = 0.5 * (projected + projected.conj().T)
    _, rotation = np.linalg.eigh(projected)
    localized = subspace @ rotation
    localized_energy = np.real(
        np.sum(np.abs(rotation) ** 2 * energies[candidates, None], axis=0)
    )
    target_weight = np.sum(np.abs(localized[target_mask]) ** 2, axis=0)
    opposite_weight = np.sum(np.abs(localized[opposite_mask]) ** 2, axis=0)

    def score(index: int) -> tuple[float, float, float]:
        return (
            float(target_weight[index] - opposite_weight[index]),
            float(target_weight[index]),
            -float(abs(localized_energy[index])),
        )

    selected = max(range(count), key=score)
    return {
        "index": int(candidates[selected]),
        "energy": float(localized_energy[selected]),
        "vector": localized[:, selected].copy(),
        "target_weight": float(target_weight[selected]),
        "opposite_weight": float(opposite_weight[selected]),
        "interface_weight": float(target_weight[selected] + opposite_weight[selected]),
    }


def _crossing_pair(momentum: np.ndarray, energy: np.ndarray, wall_weight: np.ndarray, *, zero_tol: float) -> tuple[int, int]:
    momentum = np.asarray(momentum, dtype=np.float64)
    energy = np.asarray(energy, dtype=np.float64)
    wall_weight = np.asarray(wall_weight, dtype=np.float64)
    if momentum.shape != energy.shape or energy.shape != wall_weight.shape:
        raise ValueError("momentum, energy, and wall_weight must have identical shapes.")
    order = np.argsort(momentum, kind="stable")
    best: tuple[tuple[float, ...], tuple[int, int]] | None = None
    for left, right in zip(order[:-1], order[1:]):
        e0, e1 = float(energy[left]), float(energy[right])
        crosses = e0 * e1 <= 0.0 or min(abs(e0), abs(e1)) <= float(zero_tol)
        midpoint = 0.5 * float(momentum[left] + momentum[right])
        cost = (
            0.0 if crosses else 1.0,
            -min(float(wall_weight[left]), float(wall_weight[right])),
            abs(midpoint),
            0.0 if midpoint >= 0.0 else 1.0,
            abs(e0) + abs(e1),
            float(momentum[left]),
        )
        if best is None or cost < best[0]:
            best = (cost, (int(left), int(right)))
    if best is None:
        raise ValueError("At least two momenta are required to construct a two-mode edge frame.")
    return best[1]


@dataclass(frozen=True)
class PhysicalEdgeFrame:
    """Two physical Bloch edge modes and the diagnostics used to select them."""

    frame: np.ndarray
    momentum: np.ndarray
    momentum_indices: np.ndarray
    energies: np.ndarray
    target_wall_weight: np.ndarray
    opposite_wall_weight: np.ndarray
    interface_weight: np.ndarray
    retained_norm: np.ndarray
    wall: str
    wall_x: tuple[int, int]
    interface_width: int
    source: str
    scan_momentum: np.ndarray
    scan_energy_left: np.ndarray
    scan_energy_right: np.ndarray
    scan_wall_weight_left: np.ndarray
    scan_wall_weight_right: np.ndarray
    active_indices: np.ndarray

    def metadata(self) -> dict[str, Any]:
        gram = self.frame.conj().T @ self.frame
        delta_k = float(np.angle(np.exp(1j * (self.momentum[1] - self.momentum[0]))))
        return {
            "source": self.source,
            "wall": self.wall,
            "wall_x": list(self.wall_x),
            "interface_width": int(self.interface_width),
            "momentum": self.momentum.tolist(),
            "momentum_indices": self.momentum_indices.tolist(),
            "delta_momentum": delta_k,
            "energies": self.energies.tolist(),
            "target_wall_weight": self.target_wall_weight.tolist(),
            "opposite_wall_weight": self.opposite_wall_weight.tolist(),
            "interface_weight": self.interface_weight.tolist(),
            "retained_norm": self.retained_norm.tolist(),
            "orthonormality_residual": float(np.linalg.norm(gram - np.eye(2), ord="fro")),
            "active_dimension": int(self.active_indices.size),
            "full_dimension": int(self.frame.shape[0]),
        }

    def payload(self, *, prefix: str = "edge_") -> dict[str, np.ndarray]:
        return {
            f"{prefix}initial_frame": np.asarray(self.frame, dtype=np.complex128),
            f"{prefix}momentum": np.asarray(self.momentum, dtype=np.float64),
            f"{prefix}momentum_indices": np.asarray(self.momentum_indices, dtype=np.int64),
            f"{prefix}energies": np.asarray(self.energies, dtype=np.float64),
            f"{prefix}target_wall_weight": np.asarray(self.target_wall_weight, dtype=np.float64),
            f"{prefix}opposite_wall_weight": np.asarray(self.opposite_wall_weight, dtype=np.float64),
            f"{prefix}interface_weight": np.asarray(self.interface_weight, dtype=np.float64),
            f"{prefix}retained_norm": np.asarray(self.retained_norm, dtype=np.float64),
            f"{prefix}wall_x": np.asarray(self.wall_x, dtype=np.int64),
            f"{prefix}active_indices": np.asarray(self.active_indices, dtype=np.int64),
            f"{prefix}scan_momentum": np.asarray(self.scan_momentum, dtype=np.float64),
            f"{prefix}scan_energy_left": np.asarray(self.scan_energy_left, dtype=np.float64),
            f"{prefix}scan_energy_right": np.asarray(self.scan_energy_right, dtype=np.float64),
            f"{prefix}scan_wall_weight_left": np.asarray(self.scan_wall_weight_left, dtype=np.float64),
            f"{prefix}scan_wall_weight_right": np.asarray(self.scan_wall_weight_right, dtype=np.float64),
        }


def project_edge_frame(
    edge: PhysicalEdgeFrame,
    active_indices: np.ndarray,
    *,
    rank_tol: float = 1e-12,
) -> PhysicalEdgeFrame:
    """Project a physical edge frame into a canonical active measurement basis."""

    active = np.asarray(active_indices, dtype=np.int64).reshape(-1)
    if active.size == 0 or np.any(active < 0) or np.any(active >= edge.frame.shape[0]):
        raise ValueError("active_indices must be a nonempty subset of the full edge-frame basis.")
    if np.unique(active).size != active.size:
        raise ValueError("active_indices must be unique.")
    projected = np.zeros_like(edge.frame)
    projected[active] = edge.frame[active]
    retained = np.linalg.norm(projected, axis=0)
    projected = _symmetric_orthonormalize(projected, rank_tol=rank_tol)
    projected = _fix_column_phases(projected)
    return PhysicalEdgeFrame(
        frame=projected,
        momentum=edge.momentum.copy(),
        momentum_indices=edge.momentum_indices.copy(),
        energies=edge.energies.copy(),
        target_wall_weight=edge.target_wall_weight.copy(),
        opposite_wall_weight=edge.opposite_wall_weight.copy(),
        interface_weight=edge.interface_weight.copy(),
        retained_norm=retained,
        wall=edge.wall,
        wall_x=edge.wall_x,
        interface_width=edge.interface_width,
        source=edge.source,
        scan_momentum=edge.scan_momentum.copy(),
        scan_energy_left=edge.scan_energy_left.copy(),
        scan_energy_right=edge.scan_energy_right.copy(),
        scan_wall_weight_left=edge.scan_wall_weight_left.copy(),
        scan_wall_weight_right=edge.scan_wall_weight_right.copy(),
        active_indices=active,
    )


def build_physical_edge_frame(
    model: Any,
    *,
    wall: str = "left",
    interface_width: int = 2,
    momentum_indices: tuple[int, int] | None = None,
    candidate_count: int = 2,
    degeneracy_tol: float = 1e-9,
    crossing_zero_tol: float = 1e-8,
    minimum_wall_weight: float = 0.05,
    active_indices: np.ndarray | None = None,
) -> PhysicalEdgeFrame:
    """Construct two adjacent physical edge-band modes on one domain wall.

    The edge bands are obtained from the analytic fixed-``k_y`` blocks of the
    exact real-space Chern-insulator Hamiltonian.  Exact or near-exact wall
    degeneracies are resolved by diagonalizing a left-minus-right wall
    discriminator inside each degenerate energy subspace.  No Lyapunov data are
    used in selecting the frame.
    """

    wall = str(wall).strip().lower()
    if wall not in ("left", "right"):
        raise ValueError("wall must be 'left' or 'right'.")
    if not bool(getattr(model, "DW", False)) or not hasattr(model, "DW_loc"):
        raise ValueError("Physical domain-wall edge modes require a model with DW=True.")
    nx, ny = int(model.Nx), int(model.Ny)
    if ny < 2:
        raise ValueError("At least two y momenta are required for a two-mode edge frame.")
    if int(interface_width) <= 0:
        raise ValueError("interface_width must be positive.")
    if int(candidate_count) < 2:
        raise ValueError("candidate_count must retain at least the two counter-propagating wall states.")
    wall_x = tuple(sorted(int(value) % nx for value in model.DW_loc))
    if len(wall_x) != 2 or wall_x[0] == wall_x[1]:
        raise ValueError(f"Expected two distinct domain-wall columns, got {wall_x}.")

    left_x_mask = _periodic_x_mask(nx, wall_x[0], int(interface_width))
    right_x_mask = _periodic_x_mask(nx, wall_x[1], int(interface_width))
    left_mask = np.repeat(left_x_mask, 2)
    right_mask = np.repeat(right_x_mask, 2)
    discriminator = left_mask.astype(np.float64) - right_mask.astype(np.float64)
    ky_fft = 2.0 * np.pi * np.fft.fftfreq(ny)
    ky_wrapped = np.angle(np.exp(1j * ky_fft))

    selected: dict[str, list[dict[str, Any]]] = {"left": [], "right": []}
    for ky in ky_fft:
        hamiltonian = analytic_domain_wall_bloch_hamiltonian(model, float(ky))
        energies, vectors = np.linalg.eigh(hamiltonian)
        energies, vectors = _wall_resolve_degenerate_subspaces(
            energies,
            vectors,
            discriminator,
            degeneracy_tol=float(degeneracy_tol),
        )
        selected["left"].append(
            _select_wall_state(
                energies,
                vectors,
                left_mask,
                right_mask,
                candidate_count=candidate_count,
            )
        )
        selected["right"].append(
            _select_wall_state(
                energies,
                vectors,
                right_mask,
                left_mask,
                candidate_count=candidate_count,
            )
        )

    scan_energy = {
        key: np.asarray([entry["energy"] for entry in values], dtype=np.float64)
        for key, values in selected.items()
    }
    scan_weight = {
        key: np.asarray([entry["target_weight"] for entry in values], dtype=np.float64)
        for key, values in selected.items()
    }
    if momentum_indices is None:
        pair = _crossing_pair(
            ky_wrapped, scan_energy[wall], scan_weight[wall], zero_tol=float(crossing_zero_tol)
        )
    else:
        if len(momentum_indices) != 2:
            raise ValueError("momentum_indices must contain exactly two indices.")
        pair = tuple(int(index) % ny for index in momentum_indices)
        separation = abs(float(np.angle(np.exp(1j * (ky_fft[pair[1]] - ky_fft[pair[0]])))))
        if not np.isclose(separation, 2.0 * np.pi / ny, atol=1e-10, rtol=1e-10):
            raise ValueError("The two momentum indices must be adjacent on the periodic momentum grid.")

    block_modes = np.column_stack([selected[wall][index]["vector"] for index in pair])
    full_frame = np.empty((2 * nx * ny, 2), dtype=np.complex128)
    for column, index in enumerate(pair):
        phase_y = np.exp(1j * float(ky_fft[index]) * np.arange(ny)) / np.sqrt(ny)
        full_frame[:, column] = np.kron(phase_y, block_modes[:, column])
    full_frame = _fix_column_phases(_symmetric_orthonormalize(full_frame))

    target_weight = np.asarray([selected[wall][index]["target_weight"] for index in pair])
    opposite_weight = np.asarray([selected[wall][index]["opposite_weight"] for index in pair])
    interface_weight = np.asarray([selected[wall][index]["interface_weight"] for index in pair])
    if np.any(target_weight < float(minimum_wall_weight)):
        raise ValueError(
            "Selected states are not sufficiently localized on the requested wall: "
            f"weights={target_weight.tolist()}, threshold={minimum_wall_weight}."
        )

    edge = PhysicalEdgeFrame(
        frame=full_frame,
        momentum=np.asarray([ky_wrapped[index] for index in pair], dtype=np.float64),
        momentum_indices=np.asarray(pair, dtype=np.int64),
        energies=np.asarray([selected[wall][index]["energy"] for index in pair]),
        target_wall_weight=target_weight,
        opposite_wall_weight=opposite_weight,
        interface_weight=interface_weight,
        retained_norm=np.ones((2,), dtype=np.float64),
        wall=wall,
        wall_x=wall_x,
        interface_width=int(interface_width),
        source="analytic_exact_domain_wall_bloch_hamiltonian_v1",
        scan_momentum=ky_wrapped,
        scan_energy_left=scan_energy["left"],
        scan_energy_right=scan_energy["right"],
        scan_wall_weight_left=scan_weight["left"],
        scan_wall_weight_right=scan_weight["right"],
        active_indices=np.arange(2 * nx * ny, dtype=np.int64),
    )
    return edge if active_indices is None else project_edge_frame(edge, active_indices)

