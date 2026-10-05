"""Small equilibrium flux-threading model for the wall-crossing notebook.

The module deliberately uses the canonical CPU overcomplete-Wannier (OW)
constructor.  It contains no monitored dynamics and writes no production data.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import contextlib
import io
from pathlib import Path
import sys
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment
from tqdm.auto import tqdm


HERE = Path(__file__).resolve()
REPO_ROOT = next(parent for parent in HERE.parents if (parent / "PROJECT_ADMIN").is_dir())
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from fgtn.classA_U1FGTN import classA_U1FGTN  # noqa: E402


@dataclass(frozen=True)
class ExplorerConfig:
    """Resolved parameters for one deterministic equilibrium demonstration."""

    nx: int = 20
    ny: int = 24
    nshell: int = 1
    wall: str = "soft"
    alpha_top: float = 1.0
    alpha_trivial: float = 30.0
    trial_orbitals: str = "X"
    intervals: int = 64
    regulator: float = 1.0e-7
    direction: str = "ccw"
    local_half_width: float = 0.08
    local_points: int = 49
    degeneracy_tolerance: float = 1.0e-9

    def validate(self) -> None:
        if self.nx < 8 or self.nx % 2:
            raise ValueError("nx must be an even integer at least 8")
        if self.ny < 4:
            raise ValueError("ny must be at least 4")
        if self.nshell != 1:
            raise ValueError("this teaching notebook is intentionally locked to nshell=1")
        if self.wall not in {"soft", "hard"}:
            raise ValueError("wall must be soft or hard")
        if self.direction not in {"ccw", "cw"}:
            raise ValueError("direction must be ccw or cw")
        if self.intervals < 8:
            raise ValueError("intervals must be at least 8")
        if self.local_points < 5 or self.local_points % 2 == 0:
            raise ValueError("local_points must be odd and at least 5")
        if not np.isfinite(self.regulator) or self.regulator <= 0:
            raise ValueError("regulator must be finite and positive")

    @property
    def dimension(self) -> int:
        return 2 * self.nx * self.ny

    @property
    def rank(self) -> int:
        return self.nx * self.ny

    @property
    def sigma(self) -> int:
        return 1 if self.direction == "ccw" else -1

    @property
    def wall_interval(self) -> tuple[int, int]:
        return self.nx // 4, 3 * self.nx // 4

    def metadata(self) -> dict[str, Any]:
        result = asdict(self)
        result.update({
            "dimension": self.dimension,
            "rank": self.rank,
            "sigma": self.sigma,
            "wall_interval": self.wall_interval,
            "translation_invariant_direction": "y",
            "canonical_constructor": "fgtn.classA_U1FGTN.classA_U1FGTN.construct_OW_projectors",
        })
        return result


def _frame_projector(frame: np.ndarray, dimension: int, rank: int) -> np.ndarray:
    values = np.asarray(frame, dtype=np.complex128).reshape(dimension, rank, order="F")
    return values @ values.conj().T


def exact_ow_hamiltonian(config: ExplorerConfig, phi: float) -> np.ndarray:
    """Rebuild the exact single-particle OW Hamiltonian at absolute flux phi."""
    config.validate()
    with contextlib.redirect_stdout(io.StringIO()):
        model = classA_U1FGTN(
            Nx=config.nx,
            Ny=config.ny,
            DW=True,
            nshell=config.nshell,
            filling_frac=0.5,
            alpha_1=config.alpha_top,
            alpha_2=config.alpha_trivial,
            trial_orbitals=config.trial_orbitals,
            dw_truncation=config.wall == "hard",
            twist_y=float(phi),
            dw_interval=config.wall_interval,
        )
        model.construct_OW_projectors(
            nshell=config.nshell,
            DW=True,
            trial_orbitals=config.trial_orbitals,
            dw_truncation=config.wall == "hard",
            twist_y=float(phi),
        )
    projectors = (
        _frame_projector(model.WF_Ap, config.dimension, config.rank)
        + _frame_projector(model.WF_Bp, config.dimension, config.rank)
        - _frame_projector(model.WF_Am, config.dimension, config.rank)
        - _frame_projector(model.WF_Bm, config.dimension, config.rank)
    )
    return np.asarray(0.5 * (projectors + projectors.conj().T), dtype=np.complex128)


def _y_blocks(matrix: np.ndarray, config: ExplorerConfig) -> tuple[np.ndarray, float]:
    """Fourier transform the translation-invariant y direction."""
    orbitals = 2 * config.nx
    shaped = np.asarray(matrix, dtype=np.complex128).reshape(
        config.ny, orbitals, config.ny, orbitals
    )
    transformed = np.fft.ifft(np.fft.fft(shaped, axis=0), axis=2)
    blocks = np.stack([transformed[k, :, k, :] for k in range(config.ny)])
    off_diagonal = np.array(transformed, copy=True)
    for momentum in range(config.ny):
        off_diagonal[momentum, :, momentum, :] = 0.0
    return blocks, float(np.max(np.abs(off_diagonal)))


def diagonalize(config: ExplorerConfig, phi: float) -> tuple[np.ndarray, np.ndarray, float]:
    hamiltonian = exact_ow_hamiltonian(config, phi)
    blocks, off_diagonal = _y_blocks(hamiltonian, config)
    eigenvalues, eigenvectors = np.linalg.eigh(blocks)
    return (
        np.asarray(eigenvalues, dtype=np.float64),
        np.asarray(eigenvectors, dtype=np.complex128),
        off_diagonal,
    )


def _lowest_occupations(eigenvalues: np.ndarray, rank: int) -> np.ndarray:
    occupied = np.zeros_like(eigenvalues, dtype=bool)
    order = np.argsort(eigenvalues, axis=None)[:rank]
    occupied[np.unravel_index(order, eigenvalues.shape)] = True
    return occupied


def _continue_basis(
    previous: np.ndarray,
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    degeneracy_tolerance: float,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Match each fixed-momentum eigenbasis to its previous-flux basis."""
    tracked = np.empty_like(eigenvectors)
    tracked_values = np.empty_like(eigenvalues)
    assigned_minimum = 1.0
    principal_minimum = 1.0
    for momentum in range(eigenvalues.shape[0]):
        overlap = np.abs(previous[momentum].conj().T @ eigenvectors[momentum]) ** 2
        rows, columns = linear_sum_assignment(-overlap)
        permutation = columns[np.argsort(rows)]
        values = np.asarray(eigenvalues[momentum, permutation], dtype=np.float64)
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
                if abs(values[index] - values[seed]) < degeneracy_tolerance
            )
            unused -= set(cluster)
            indices = np.asarray(cluster, dtype=np.int64)
            small_overlap = previous[momentum][:, indices].conj().T @ vectors[:, indices]
            left, singular, right_h = np.linalg.svd(small_overlap, full_matrices=False)
            vectors[:, indices] = vectors[:, indices] @ (right_h.conj().T @ left.conj().T)
            principal_minimum = min(principal_minimum, float(singular.min()))
        tracked[momentum] = vectors
        tracked_values[momentum] = values
    return tracked, tracked_values, assigned_minimum, principal_minimum


def _density_x(eigenvectors: np.ndarray, occupied: np.ndarray, config: ExplorerConfig) -> np.ndarray:
    density = np.zeros(2 * config.nx, dtype=np.float64)
    for momentum in range(config.ny):
        density += np.sum(
            np.abs(eigenvectors[momentum][:, occupied[momentum]]) ** 2,
            axis=1,
        )
    return density.reshape(config.nx, 2).sum(axis=1)


def _block_mode(
    eigenvectors: np.ndarray,
    momentum: int,
    band: int,
    config: ExplorerConfig,
) -> np.ndarray:
    phase = np.exp(2j * np.pi * np.arange(config.ny) * momentum / config.ny)
    phase /= np.sqrt(config.ny)
    block = eigenvectors[momentum, :, band]
    return np.asarray((phase[:, None] * block[None, :]).reshape(config.dimension))


def _edge_pair(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    config: ExplorerConfig,
) -> dict[str, np.ndarray | float]:
    order = np.argsort(eigenvalues, axis=None)
    lower_flat, upper_flat = order[config.rank - 1 : config.rank + 1]
    lower = np.unravel_index(lower_flat, eigenvalues.shape)
    upper = np.unravel_index(upper_flat, eigenvalues.shape)
    sorted_values = eigenvalues.ravel()[order]
    energy_modes = np.column_stack([
        _block_mode(eigenvectors, *lower, config),
        _block_mode(eigenvectors, *upper, config),
    ])
    mode_x = np.tile(np.repeat(np.arange(config.nx), 2), config.ny)
    b_diagonal = np.where(mode_x < config.nx // 2, -1.0, 1.0)
    energy_polarization = np.real(
        np.sum(energy_modes.conj() * b_diagonal[:, None] * energy_modes, axis=0)
    )
    b_matrix = energy_modes.conj().T @ (b_diagonal[:, None] * energy_modes)
    wall_polarization, rotation = np.linalg.eigh(0.5 * (b_matrix + b_matrix.conj().T))
    wall_modes = energy_modes @ rotation
    density = lambda modes: np.sum(
        np.abs(modes.reshape(config.ny, config.nx, 2, -1)) ** 2,
        axis=(0, 2),
    ).T
    external_gap = min(
        sorted_values[config.rank - 1] - sorted_values[config.rank - 2],
        sorted_values[config.rank + 1] - sorted_values[config.rank],
    )
    return {
        "indices": np.asarray([lower, upper], dtype=np.int64),
        "energy": np.asarray([eigenvalues[lower], eigenvalues[upper]]),
        "internal_gap": float(eigenvalues[upper] - eigenvalues[lower]),
        "external_gap": float(external_gap),
        "energy_polarization": energy_polarization,
        "wall_polarization": wall_polarization,
        "energy_density_x": density(energy_modes),
        "wall_density_x": density(wall_modes),
    }


def _edge_continued_weights(
    raw_vectors: np.ndarray,
    tracked: np.ndarray,
    fixed_occupied: np.ndarray,
    indices: np.ndarray,
) -> np.ndarray:
    weights = []
    for momentum, band in indices:
        frame = tracked[int(momentum)][:, fixed_occupied[int(momentum)]]
        raw = raw_vectors[int(momentum), :, int(band)]
        weights.append(float(np.sum(np.abs(frame.conj().T @ raw) ** 2)))
    return np.asarray(weights)


def scan_local_crossing(config: ExplorerConfig, *, progress: bool = True) -> dict[str, Any]:
    """Resolve the two energy-ordered modes around the finite-size crossing."""
    config.validate()
    phi = np.linspace(-config.local_half_width, config.local_half_width, config.local_points)
    rows: list[dict[str, Any]] = []
    iterator = tqdm(phi, desc="local avoided crossing", unit="flux", disable=not progress)
    maximum_off_diagonal = 0.0
    for value in iterator:
        eigenvalues, eigenvectors, off_diagonal = diagonalize(config, float(value))
        maximum_off_diagonal = max(maximum_off_diagonal, off_diagonal)
        pair = _edge_pair(eigenvalues, eigenvectors, config)
        order = np.argsort(eigenvalues, axis=None)
        near = order[config.rank - 6 : config.rank + 6]
        rows.append({
            **pair,
            "near_energy": eigenvalues[np.unravel_index(near, eigenvalues.shape)],
        })
    stack = lambda key: np.stack([np.asarray(row[key]) for row in rows])
    return {
        "config": config.metadata(),
        "phi": phi,
        "edge_energy": stack("energy"),
        "edge_internal_gap": stack("internal_gap"),
        "edge_external_gap": stack("external_gap"),
        "edge_energy_polarization": stack("energy_polarization"),
        "wall_polarization": stack("wall_polarization"),
        "energy_density_x": stack("energy_density_x"),
        "wall_density_x": stack("wall_density_x"),
        "near_energy": stack("near_energy"),
        "maximum_momentum_off_diagonal": maximum_off_diagonal,
    }


def scan_flux_pump(config: ExplorerConfig, *, progress: bool = True) -> dict[str, Any]:
    """Compare energy refilling with previous-overlap spectral continuation."""
    config.validate()
    phi = (
        -config.sigma * config.regulator
        + config.sigma * 2.0 * np.pi * np.arange(config.intervals + 1) / config.intervals
    )
    fixed_occupied: np.ndarray | None = None
    previous: np.ndarray | None = None
    continued_density: list[np.ndarray] = []
    instantaneous_density: list[np.ndarray] = []
    edge_energy: list[np.ndarray] = []
    edge_polarization: list[np.ndarray] = []
    edge_continued_weight: list[np.ndarray] = []
    edge_gap: list[float] = []
    assigned: list[float] = []
    principal: list[float] = []
    maximum_off_diagonal = 0.0

    iterator = tqdm(phi, desc="one-flux spectral continuation", unit="flux", disable=not progress)
    for point, value in enumerate(iterator):
        eigenvalues, raw_vectors, off_diagonal = diagonalize(config, float(value))
        maximum_off_diagonal = max(maximum_off_diagonal, off_diagonal)
        if point == 0:
            fixed_occupied = _lowest_occupations(eigenvalues, config.rank)
            tracked = np.asarray(raw_vectors, dtype=np.complex128)
            assigned.append(1.0)
            principal.append(1.0)
        else:
            assert previous is not None
            tracked, _, assigned_minimum, principal_minimum = _continue_basis(
                previous,
                eigenvalues,
                raw_vectors,
                config.degeneracy_tolerance,
            )
            assigned.append(assigned_minimum)
            principal.append(principal_minimum)
        assert fixed_occupied is not None
        instantaneous_occupied = _lowest_occupations(eigenvalues, config.rank)
        pair = _edge_pair(eigenvalues, raw_vectors, config)
        continued_density.append(_density_x(tracked, fixed_occupied, config))
        instantaneous_density.append(_density_x(raw_vectors, instantaneous_occupied, config))
        edge_energy.append(np.asarray(pair["energy"]))
        edge_polarization.append(np.asarray(pair["energy_polarization"]))
        edge_continued_weight.append(
            _edge_continued_weights(
                raw_vectors,
                tracked,
                fixed_occupied,
                np.asarray(pair["indices"]),
            )
        )
        edge_gap.append(float(pair["internal_gap"]))
        previous = tracked

    continued_density_array = np.stack(continued_density)
    instantaneous_density_array = np.stack(instantaneous_density)

    def regional(density: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        left = density[:, : config.nx // 2].sum(axis=1)
        right = density[:, config.nx // 2 :].sum(axis=1)
        delta_left = left - left[0]
        delta_right = right - right[0]
        q_x = 0.5 * (delta_right - delta_left)
        return delta_left, delta_right, q_x

    continued_left, continued_right, continued_q_x = regional(continued_density_array)
    instant_left, instant_right, instant_q_x = regional(instantaneous_density_array)
    return {
        "config": config.metadata(),
        "phi": phi,
        "threaded_flux": np.abs(phi - phi[0]),
        "continued_density_x": continued_density_array,
        "instantaneous_density_x": instantaneous_density_array,
        "continued_delta_N_left": continued_left,
        "continued_delta_N_right": continued_right,
        "continued_q_x": continued_q_x,
        "instantaneous_delta_N_left": instant_left,
        "instantaneous_delta_N_right": instant_right,
        "instantaneous_q_x": instant_q_x,
        "edge_energy": np.stack(edge_energy),
        "edge_energy_polarization": np.stack(edge_polarization),
        "edge_continued_occupation_weight": np.stack(edge_continued_weight),
        "edge_instantaneous_occupation_weight": np.tile([1.0, 0.0], (phi.size, 1)),
        "edge_internal_gap": np.asarray(edge_gap),
        "assigned_overlap_minimum": np.asarray(assigned),
        "principal_overlap_minimum": np.asarray(principal),
        "maximum_momentum_off_diagonal": maximum_off_diagonal,
        "maximum_charge_residual": float(
            max(
                np.max(np.abs(continued_left + continued_right)),
                np.max(np.abs(instant_left + instant_right)),
            )
        ),
    }


def validate_results(local: dict[str, Any], pump: dict[str, Any]) -> dict[str, Any]:
    """Return the compact diagnostics printed at the end of the notebook."""
    config = pump["config"]
    expected_local = int(config["local_points"])
    expected_pump = int(config["intervals"]) + 1
    if np.asarray(local["phi"]).size != expected_local:
        raise RuntimeError("local scan length changed")
    if np.asarray(pump["phi"]).size != expected_pump:
        raise RuntimeError("pump scan length changed")
    diagnostics = {
        "local_points": expected_local,
        "pump_points": expected_pump,
        "minimum_edge_gap": float(np.min(local["edge_internal_gap"])),
        "maximum_translation_block_residual": float(
            max(local["maximum_momentum_off_diagonal"], pump["maximum_momentum_off_diagonal"])
        ),
        "minimum_assigned_overlap": float(np.min(pump["assigned_overlap_minimum"])),
        "minimum_principal_overlap": float(np.min(pump["principal_overlap_minimum"])),
        "maximum_charge_residual": float(pump["maximum_charge_residual"]),
        "continued_endpoint_q_x": float(np.asarray(pump["continued_q_x"])[-1]),
        "instantaneous_endpoint_q_x": float(np.asarray(pump["instantaneous_q_x"])[-1]),
    }
    if diagnostics["maximum_translation_block_residual"] > 1.0e-10:
        raise RuntimeError("Hamiltonian is not translation invariant in y")
    if diagnostics["maximum_charge_residual"] > 1.0e-9:
        raise RuntimeError("regional charge does not close")
    return diagnostics
