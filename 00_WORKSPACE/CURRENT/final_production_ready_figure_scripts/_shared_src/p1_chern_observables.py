from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


P1_CHERN_SCHEMA = "p1_periodic_trijunction_chern_v1"
CENTER_SAMPLING_SCHEMA = "fresh_per_sample_cycle_shared_across_shells_v1"


def _atomic_npz(path: Path | str, **arrays: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def _sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def center_seed(*, root_seed: int, size: int, sample_id: int, cycle: int) -> int:
    """Counter-derived center seed deliberately independent of ``n_shell``."""

    key = (
        f"{int(root_seed)}:{int(size)}:{int(sample_id)}:{int(cycle)}:"
        f"{CENTER_SAMPLING_SCHEMA}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "little")


def sample_trijunction_centers(
    *, root_seed: int, size: int, sample_id: int, cycle: int, count: int = 10
) -> np.ndarray:
    """Draw distinct periodic-lattice centers reproducibly for one sample-cycle."""

    size = int(size)
    count = int(count)
    if size <= 0 or not 0 < count <= size * size:
        raise ValueError("center count must lie in 1..L^2")
    generator = np.random.default_rng(
        center_seed(
            root_seed=root_seed, size=size, sample_id=sample_id, cycle=cycle
        )
    )
    flat = generator.choice(size * size, size=count, replace=False)
    return np.column_stack((flat % size, flat // size)).astype(np.int64, copy=False)


def build_periodic_chern_partition_indices(
    *,
    nx: int,
    ny: int,
    xref: int,
    yref: int,
    radius: float,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    """Build the three radius-``R`` wedges using periodic minimum-image offsets."""

    nx, ny = int(nx), int(ny)
    xref, yref = int(xref), int(yref)
    radius = float(radius)
    if nx <= 0 or ny <= 0:
        raise ValueError("lattice dimensions must be positive")
    if not 0 <= xref < nx or not 0 <= yref < ny:
        raise ValueError("trijunction center lies outside the lattice")
    if not 0.0 < radius < 0.5 * min(nx, ny):
        raise ValueError("periodic disk radius must lie in (0, min(nx,ny)/2)")

    x = np.arange(nx, dtype=np.int64)
    y = np.arange(ny, dtype=np.int64)
    dx = (x - xref + nx // 2) % nx - nx // 2
    dy = (y - yref + ny // 2) % ny - ny // 2
    dx_grid, dy_grid = np.meshgrid(dx, dy, indexing="ij")
    inside = dx_grid * dx_grid + dy_grid * dy_grid <= radius * radius
    theta = np.mod(np.arctan2(dy_grid, dx_grid), 2.0 * np.pi)
    boundaries = (0.0, 2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0, 2.0 * np.pi)
    masks = (
        inside & (theta >= boundaries[0]) & (theta < boundaries[1]),
        inside & (theta >= boundaries[1]) & (theta < boundaries[2]),
        inside & (theta >= boundaries[2]) & (theta < boundaries[3]),
    )

    def indices(mask: np.ndarray) -> torch.Tensor:
        xs, ys = np.nonzero(mask)
        first = 2 * xs + 2 * nx * ys
        values = np.sort(np.concatenate((first, first + 1))).astype(np.int64)
        return torch.as_tensor(values, dtype=torch.long, device=device)

    a, b, c = (indices(mask) for mask in masks)
    if min(int(a.numel()), int(b.numel()), int(c.numel())) == 0:
        raise ValueError("periodic Chern partition contains an empty wedge")
    return {
        "nx": nx,
        "ny": ny,
        "xref": xref,
        "yref": yref,
        "radius": radius,
        "inside_mask": inside,
        "A": a,
        "B": b,
        "C": c,
    }


def real_space_chern_from_frame(
    frame: torch.Tensor, partitions: dict[str, Any]
) -> torch.Tensor:
    """Evaluate the repository's +1-oriented disk Chern estimator without forming C."""

    if frame.ndim != 3:
        raise ValueError("occupied frame must have shape (sample, mode, rank)")
    projector_frame = frame.conj()
    a = partitions["A"].to(frame.device)
    b = partitions["B"].to(frame.device)
    c = partitions["C"].to(frame.device)
    w_a = projector_frame.index_select(1, a)
    w_b = projector_frame.index_select(1, b)
    w_c = projector_frame.index_select(1, c)
    p_ca = w_c @ w_a.mH
    p_ab = w_a @ w_b.mH
    p_bc = w_b @ w_c.mH
    p_ac = w_a @ w_c.mH
    p_cb = w_c @ w_b.mH
    p_ba = w_b @ w_a.mH
    first = torch.diagonal(p_ca @ p_ab @ p_bc, dim1=-2, dim2=-1).sum(-1)
    second = torch.diagonal(p_ac @ p_cb @ p_ba, dim1=-2, dim2=-1).sum(-1)
    return (12.0 * math.pi * 1j * (first - second)).real.to(torch.float64)


def real_space_chern_from_centered_covariance(
    centered: torch.Tensor, partitions: dict[str, Any]
) -> torch.Tensor:
    """Dense reference for validation of the occupied-frame implementation."""

    if centered.ndim != 3 or centered.shape[-1] != centered.shape[-2]:
        raise ValueError("centered covariance must have shape (sample, mode, mode)")
    identity = torch.eye(
        centered.shape[-1], dtype=centered.dtype, device=centered.device
    )
    projector = (0.5 * (centered + identity[None])).conj()
    a = partitions["A"].to(centered.device)
    b = partitions["B"].to(centered.device)
    c = partitions["C"].to(centered.device)

    def block(rows: torch.Tensor, columns: torch.Tensor) -> torch.Tensor:
        return projector.index_select(1, rows).index_select(2, columns)

    first = torch.diagonal(
        block(c, a) @ block(a, b) @ block(b, c), dim1=-2, dim2=-1
    ).sum(-1)
    second = torch.diagonal(
        block(a, c) @ block(c, b) @ block(b, a), dim1=-2, dim2=-1
    ).sum(-1)
    return (12.0 * math.pi * 1j * (first - second)).real.to(torch.float64)


class P1ChernObserver:
    """Retain only ten-center real-space Chern values at cycles ``0..L``."""

    def __init__(
        self,
        *,
        size: int,
        physical_cycles: int,
        global_sample_ids: Iterable[int],
        root_seed: int,
        center_count: int = 10,
        radius_fraction: float = 0.4,
    ) -> None:
        self.size = int(size)
        self.physical_cycles = int(physical_cycles)
        self.global_sample_ids = np.asarray(
            list(global_sample_ids), dtype=np.int64
        )
        self.root_seed = int(root_seed)
        self.center_count = int(center_count)
        self.radius_fraction = float(radius_fraction)
        self.radius = self.radius_fraction * self.size
        if self.physical_cycles != self.size:
            raise ValueError("P1 physical duration must equal L")
        samples = int(self.global_sample_ids.size)
        shape = (samples, self.physical_cycles + 1, self.center_count)
        self.center_x = np.full(shape, -1, dtype=np.int64)
        self.center_y = np.full(shape, -1, dtype=np.int64)
        self.chern_by_center = np.full(shape, np.nan, dtype=np.float64)
        self.seen = np.zeros((samples, self.physical_cycles + 1), dtype=np.bool_)
        self.observer_seconds = np.zeros(
            (samples, self.physical_cycles + 1), dtype=np.float64
        )
        self.peak_working_bytes = 0

    def __call__(
        self,
        *,
        cycle: int,
        state: Any | None = None,
        G: Any | None = None,
        batch_start: int,
        batch_count: int,
        **_: Any,
    ) -> None:
        cycle = int(cycle)
        start, stop = int(batch_start), int(batch_start) + int(batch_count)
        if not 0 <= cycle <= self.physical_cycles:
            raise IndexError("observer cycle lies outside 0..L")
        if stop > len(self.global_sample_ids):
            raise IndexError("observer batch lies outside the declared shard")
        if np.any(self.seen[start:stop, cycle]):
            raise RuntimeError("duplicate P1 Chern observation")

        if state is not None and hasattr(state, "frame"):
            frame = state.frame.detach()
            centered = None
        elif G is not None:
            frame = None
            centered = torch.as_tensor(G)
        else:
            raise TypeError("P1 observer requires an occupied frame or centered covariance")

        for local in range(start, stop):
            began = time.perf_counter()
            centers = sample_trijunction_centers(
                root_seed=self.root_seed,
                size=self.size,
                sample_id=int(self.global_sample_ids[local]),
                cycle=cycle,
                count=self.center_count,
            )
            self.center_x[local, cycle] = centers[:, 0]
            self.center_y[local, cycle] = centers[:, 1]
            batch_local = local - start
            for center_index, (xref, yref) in enumerate(centers):
                partitions = build_periodic_chern_partition_indices(
                    nx=self.size,
                    ny=self.size,
                    xref=int(xref),
                    yref=int(yref),
                    radius=self.radius,
                    device=(frame.device if frame is not None else centered.device),
                )
                if frame is not None:
                    value = real_space_chern_from_frame(
                        frame[batch_local : batch_local + 1], partitions
                    )
                else:
                    value = real_space_chern_from_centered_covariance(
                        centered[batch_local : batch_local + 1], partitions
                    )
                self.chern_by_center[local, cycle, center_index] = float(
                    value.detach().cpu()[0]
                )
                element_bytes = 16  # complex128 is locked by the P1 production contract.
                wedge = max(
                    int(partitions[key].numel()) for key in ("A", "B", "C")
                )
                self.peak_working_bytes = max(
                    self.peak_working_bytes, 6 * wedge * wedge * element_bytes
                )
            self.observer_seconds[local, cycle] = time.perf_counter() - began
            self.seen[local, cycle] = True

    @property
    def chern_center_mean(self) -> np.ndarray:
        return np.mean(self.chern_by_center, axis=-1)

    def validate(self) -> dict[str, Any]:
        if not self.seen.all():
            raise RuntimeError("P1 Chern observer is missing one or more sample-cycles")
        if not np.isfinite(self.chern_by_center).all():
            raise FloatingPointError("P1 Chern values contain non-finite entries")
        for sample in range(self.center_x.shape[0]):
            for cycle in range(self.center_x.shape[1]):
                flat = (
                    self.center_x[sample, cycle]
                    + self.size * self.center_y[sample, cycle]
                )
                if len(np.unique(flat)) != self.center_count:
                    raise RuntimeError("trijunction centers are not distinct")
        return {
            "schema": P1_CHERN_SCHEMA,
            "samples": int(self.center_x.shape[0]),
            "cycles": self.physical_cycles + 1,
            "center_count": self.center_count,
            "radius": self.radius,
            "radius_fraction": self.radius_fraction,
            "observer_seconds": float(self.observer_seconds.sum()),
            "peak_working_bytes_estimate": int(self.peak_working_bytes),
        }

    def save(self, path: Path | str, *, config: dict[str, Any]) -> dict[str, Any]:
        diagnostics = self.validate()
        path = Path(path)
        _atomic_npz(
            path,
            schema=np.asarray(P1_CHERN_SCHEMA),
            center_sampling_schema=np.asarray(CENTER_SAMPLING_SCHEMA),
            config_json=np.asarray(json.dumps(config, sort_keys=True)),
            cycles=np.arange(self.physical_cycles + 1, dtype=np.int64),
            global_sample_ids=self.global_sample_ids,
            center_x=self.center_x,
            center_y=self.center_y,
            chern_by_center=self.chern_by_center,
            chern_center_mean=self.chern_center_mean,
            observer_seconds=self.observer_seconds,
        )
        return {
            **diagnostics,
            "path": str(path),
            "sha256": _sha256_file(path),
            "bytes": path.stat().st_size,
            "arrays": [
                "cycles",
                "global_sample_ids",
                "center_x",
                "center_y",
                "chern_by_center",
                "chern_center_mean",
                "observer_seconds",
            ],
        }
