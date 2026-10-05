from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


CHANNELS = ("Ap", "Am", "Bp", "Bm")
CHANNEL_TARGETS = {"Ap": 0, "Am": 1, "Bp": 0, "Bm": 1}
CANONICAL_CPU_ENTRY_POINT = "classA_U1FGTN.run_markov_circuit"


@dataclass(frozen=True)
class RegionMasks:
    names: tuple[str, ...]
    masks: np.ndarray
    active: np.ndarray
    interface_width: int
    wall_x: tuple[int, ...]

    def mask(self, name: str) -> np.ndarray:
        return self.masks[self.names.index(str(name))]

    def payload(self) -> dict[str, Any]:
        return {
            "region_names": np.asarray(self.names),
            "region_masks": self.masks.astype(bool, copy=False),
            "active_cell_mask": self.active.astype(bool, copy=False),
            "interface_width": np.asarray(self.interface_width, dtype=np.int64),
            "wall_x": np.asarray(self.wall_x, dtype=np.int64),
        }


def default_interface_width(nshell: float | int | None) -> int:
    if nshell is None:
        return 1
    return max(1, int(np.ceil(float(nshell))) + 1)


def active_cell_coordinates(model: Any, *, meas_slab_only: bool = True) -> list[tuple[int, int]]:
    if bool(model._meas_slab_only_effective(meas_slab_only)):
        x_min, x_max = sorted(int(value) for value in model.DW_loc)
        xs = range(x_min, x_max + 1)
    else:
        xs = range(int(model.Nx))
    return [(x, y) for y in range(int(model.Ny)) for x in xs]


def active_site_ids(model: Any, *, meas_slab_only: bool = True) -> np.ndarray:
    return np.asarray(
        [x + int(model.Nx) * y for x, y in active_cell_coordinates(model, meas_slab_only=meas_slab_only)],
        dtype=np.int64,
    )


def build_region_masks(
    model: Any,
    *,
    interface_width: int | None = None,
    meas_slab_only: bool = True,
) -> RegionMasks:
    nx, ny = int(model.Nx), int(model.Ny)
    width = default_interface_width(getattr(model, "nshell", None)) if interface_width is None else int(interface_width)
    if width <= 0:
        raise ValueError("interface_width must be positive.")

    active = np.zeros((nx, ny), dtype=bool)
    for x, y in active_cell_coordinates(model, meas_slab_only=meas_slab_only):
        active[x, y] = True

    interface = np.zeros_like(active)
    left = np.zeros_like(active)
    right = np.zeros_like(active)
    wall_x: tuple[int, ...] = ()
    if bool(getattr(model, "DW", False)) and hasattr(model, "DW_loc") and len(model.DW_loc) == 2:
        x_left, x_right = sorted(int(value) for value in model.DW_loc)
        wall_x = (x_left, x_right)
        for x in range(nx):
            if not np.any(active[x]):
                continue
            if abs(x - x_left) < width:
                left[x, :] = active[x, :]
            if abs(x - x_right) < width:
                right[x, :] = active[x, :]
        interface = left | right

    interior = active & ~interface
    masks = np.stack((active, interface, interior, left, right), axis=0)
    return RegionMasks(
        names=("all", "interface", "interior", "left_wall", "right_wall"),
        masks=masks,
        active=active,
        interface_width=width,
        wall_x=wall_x,
    )


def local_charge_map(G: np.ndarray, *, nx: int, ny: int) -> np.ndarray:
    G = np.asarray(G, dtype=np.complex128)
    nlayer = 2 * int(nx) * int(ny)
    if G.shape != (nlayer, nlayer):
        raise ValueError(f"Expected covariance shape {(nlayer, nlayer)}, got {G.shape}.")
    occupancy = 0.5 * (1.0 + np.real(np.diag(G)))
    return occupancy.reshape((2, int(nx), int(ny)), order="F").sum(axis=0)


def vector_cell_weight(
    vector: np.ndarray,
    *,
    nx: int,
    ny: int,
    active_indices: np.ndarray | None = None,
) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.complex128).reshape(-1)
    nlayer = 2 * int(nx) * int(ny)
    if vector.size == nlayer:
        full = vector
    elif active_indices is not None and vector.size == int(np.asarray(active_indices).size):
        full = np.zeros((nlayer,), dtype=np.complex128)
        full[np.asarray(active_indices, dtype=np.int64)] = vector
    else:
        raise ValueError(
            f"Vector length {vector.size} is neither full dimension {nlayer} nor the active dimension."
        )
    weights = np.abs(full) ** 2
    cell = weights.reshape((2, int(nx), int(ny)), order="F").sum(axis=0)
    total = float(np.sum(cell))
    return cell / total if total > 0.0 else cell


def mode_localization(
    vector: np.ndarray,
    *,
    nx: int,
    ny: int,
    regions: RegionMasks,
    active_indices: np.ndarray | None = None,
) -> dict[str, Any]:
    cell = vector_cell_weight(vector, nx=nx, ny=ny, active_indices=active_indices)
    flat = cell.reshape(-1)
    result: dict[str, Any] = {
        "cell_weight": cell,
        "x_profile": np.sum(cell, axis=1),
        "ipr": float(np.sum(flat * flat)),
    }
    for name, mask in zip(regions.names, regions.masks):
        result[f"{name}_weight"] = float(np.sum(cell[mask]))
    return result
