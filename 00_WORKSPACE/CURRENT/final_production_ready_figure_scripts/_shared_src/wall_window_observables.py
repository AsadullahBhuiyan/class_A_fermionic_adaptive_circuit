"""Frame-native raw observables for the hard/soft programmable-wall campaign."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


SCHEMA = "wall_cft_periodic_windows_v1"


def checkpoint_cycles(ny: int) -> list[int]:
    """Six nearest-integer sixths in the second Ny-cycle time window."""

    ny = int(ny)
    if ny <= 0:
        raise ValueError("Ny must be positive")
    return [ny + (j * ny + 3) // 6 for j in range(1, 7)]


def periodic_window_indices(
    *, nx: int, ny: int, y0: int, ay: int, device: torch.device | str = "cpu"
) -> torch.Tensor:
    """Mode indices for [0,Nx) x [y0,y0+Ay), ordered by dy,x,orbital."""

    nx, ny, y0, ay = int(nx), int(ny), int(y0), int(ay)
    if nx <= 0 or ny <= 0 or not 0 <= y0 < ny or not 0 <= ay <= ny:
        raise ValueError("invalid periodic window geometry")
    values = [
        mu + 2 * x + 2 * nx * ((y0 + dy) % ny)
        for dy in range(ay)
        for x in range(nx)
        for mu in range(2)
    ]
    return torch.as_tensor(values, dtype=torch.long, device=device)


def window_natural_data_from_frame(
    frame: torch.Tensor,
    *,
    rank: int,
    indices: torch.Tensor,
    nx: int,
    ay: int,
    entropy_epsilon: float = 1e-15,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Natural occupations and cell-resolved entropy/charge-variance contours."""

    if frame.ndim != 2:
        raise ValueError("frame must have shape (mode, capacity)")
    rank, nx, ay = int(rank), int(nx), int(ay)
    mode_count = 2 * nx * ay
    if indices.numel() != mode_count or not 0 <= rank <= frame.shape[1]:
        raise ValueError("frame rank or window indices are inconsistent")
    real_dtype = frame.real.dtype
    if mode_count == 0:
        empty = torch.empty((0,), dtype=real_dtype, device=frame.device)
        contour = torch.zeros((nx, 0), dtype=real_dtype, device=frame.device)
        return empty, contour, contour.clone()
    if rank == 0:
        return (
            torch.zeros((mode_count,), dtype=real_dtype, device=frame.device),
            torch.zeros((nx, ay), dtype=real_dtype, device=frame.device),
            torch.zeros((nx, ay), dtype=real_dtype, device=frame.device),
        )

    rows = frame.index_select(0, indices.to(frame.device))[:, :rank]
    vectors, singular, _ = torch.linalg.svd(rows, full_matrices=False)
    occupations = singular.square().real.clamp(0.0, 1.0)
    weights = vectors.abs().square()
    clipped = occupations.clamp(float(entropy_epsilon), 1.0 - float(entropy_epsilon))
    entropy_weight = -(clipped * clipped.log() + (1.0 - clipped) * (1.0 - clipped).log())
    variance_weight = occupations * (1.0 - occupations)
    entropy_orbital = weights @ entropy_weight
    variance_orbital = weights @ variance_weight

    spectrum = torch.zeros((mode_count,), dtype=real_dtype, device=frame.device)
    spectrum[-occupations.numel() :] = torch.sort(occupations).values
    entropy = entropy_orbital.reshape(ay, nx, 2).sum(-1).transpose(0, 1)
    variance = variance_orbital.reshape(ay, nx, 2).sum(-1).transpose(0, 1)
    return spectrum, entropy, variance


def window_natural_data_from_correlation(
    correlation: torch.Tensor, *, indices: torch.Tensor, nx: int, ay: int,
    entropy_epsilon: float = 1e-15,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Dense reference implementation used by validation tests."""

    idx = indices.to(correlation.device)
    sub = correlation.index_select(0, idx).index_select(1, idx)
    occupations, vectors = torch.linalg.eigh(0.5 * (sub + sub.mH))
    occupations = occupations.real.clamp(0.0, 1.0)
    if occupations.numel() == 0:
        contour = torch.zeros((nx, 0), dtype=correlation.real.dtype, device=correlation.device)
        return occupations, contour, contour.clone()
    clipped = occupations.clamp(float(entropy_epsilon), 1.0 - float(entropy_epsilon))
    entropy_weight = -(clipped * clipped.log() + (1.0 - clipped) * (1.0 - clipped).log())
    variance_weight = occupations * (1.0 - occupations)
    weights = vectors.abs().square()
    entropy = (weights @ entropy_weight).reshape(ay, nx, 2).sum(-1).transpose(0, 1)
    variance = (weights @ variance_weight).reshape(ay, nx, 2).sum(-1).transpose(0, 1)
    return occupations, entropy, variance


def x_resolved_square_correlator_from_frame(
    frame: torch.Tensor, *, rank: int, nx: int, ny: int
) -> torch.Tensor:
    """Return C_sq(x,r) for r=1..Ny//2 without forming the full correlation matrix."""

    rank, nx, ny = int(rank), int(nx), int(ny)
    result = torch.zeros((nx, ny // 2), dtype=frame.real.dtype, device=frame.device)
    if rank == 0:
        return result
    active = frame[:, :rank]
    for x in range(nx):
        left = torch.as_tensor(
            [mu + 2 * x + 2 * nx * y for y in range(ny) for mu in range(2)],
            dtype=torch.long, device=frame.device,
        )
        left_rows = active.index_select(0, left).reshape(ny, 2, rank)
        for offset, ry in enumerate(range(1, ny // 2 + 1)):
            right = torch.as_tensor(
                [nu + 2 * x + 2 * nx * ((y + ry) % ny) for y in range(ny) for nu in range(2)],
                dtype=torch.long, device=frame.device,
            )
            right_rows = active.index_select(0, right).reshape(ny, 2, rank)
            overlaps = torch.einsum("ymk,ynk->ymn", left_rows, right_rows.conj())
            result[x, offset] = overlaps.abs().square().sum() / float(2 * ny)
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


class WallWindowObserver:
    """Checkpoint-only observer retaining the approved raw wall-window products."""

    def __init__(
        self, *, nx: int, ny: int, checkpoints: Iterable[int],
        global_sample_ids: Iterable[int],
    ) -> None:
        self.nx, self.ny = int(nx), int(ny)
        self.checkpoints = np.asarray(list(checkpoints), dtype=np.int64)
        self.sample_ids = np.asarray(list(global_sample_ids), dtype=np.int64)
        if self.checkpoints.tolist() != checkpoint_cycles(self.ny):
            raise ValueError("checkpoint schedule differs from the locked six-point schedule")
        shape = (len(self.sample_ids), len(self.checkpoints))
        self.seen = np.zeros(shape, dtype=np.bool_)
        self.seconds = np.zeros(shape, dtype=np.float64)
        self.square_correlator = np.empty((*shape, self.nx, self.ny // 2), dtype=np.float64)
        self.spectrum: dict[int, np.ndarray] = {}
        self.entropy_contour: dict[int, np.ndarray] = {}
        self.charge_variance_contour: dict[int, np.ndarray] = {}
        for ay in range(self.ny // 2 + 1):
            modes = 2 * self.nx * ay
            self.spectrum[ay] = np.empty((*shape, self.ny, modes), dtype=np.float64)
            self.entropy_contour[ay] = np.empty((*shape, self.ny, self.nx, ay), dtype=np.float64)
            self.charge_variance_contour[ay] = np.empty((*shape, self.ny, self.nx, ay), dtype=np.float64)
        self.peak_working_bytes = 0

    def __call__(self, *, cycle: int, state: Any, batch_start: int, batch_count: int, **_: Any) -> None:
        matches = np.flatnonzero(self.checkpoints == int(cycle))
        if not len(matches):
            return
        checkpoint = int(matches[0])
        start, stop = int(batch_start), int(batch_start) + int(batch_count)
        if stop > len(self.sample_ids) or np.any(self.seen[start:stop, checkpoint]):
            raise RuntimeError("invalid or duplicate wall-window observer batch")
        if not hasattr(state, "frame") or not hasattr(state, "ranks"):
            raise TypeError("wall-window observer requires an occupied-frame state")
        for local in range(start, stop):
            began = time.perf_counter()
            batch_local = local - start
            frame = state.frame[batch_local].detach()
            rank = int(state.ranks[batch_local].item())
            self.square_correlator[local, checkpoint] = (
                x_resolved_square_correlator_from_frame(
                    frame, rank=rank, nx=self.nx, ny=self.ny
                ).cpu().numpy()
            )
            for ay in range(self.ny // 2 + 1):
                for y0 in range(self.ny):
                    indices = periodic_window_indices(
                        nx=self.nx, ny=self.ny, y0=y0, ay=ay, device=frame.device
                    )
                    spectrum, entropy, variance = window_natural_data_from_frame(
                        frame, rank=rank, indices=indices, nx=self.nx, ay=ay
                    )
                    self.spectrum[ay][local, checkpoint, y0] = spectrum.cpu().numpy()
                    self.entropy_contour[ay][local, checkpoint, y0] = entropy.cpu().numpy()
                    self.charge_variance_contour[ay][local, checkpoint, y0] = variance.cpu().numpy()
                    modes = int(indices.numel())
                    self.peak_working_bytes = max(
                        self.peak_working_bytes, 16 * modes * rank + 24 * modes * max(1, min(modes, rank))
                    )
            self.seconds[local, checkpoint] = time.perf_counter() - began
            self.seen[local, checkpoint] = True

    def validate(self) -> dict[str, Any]:
        if not self.seen.all():
            raise RuntimeError("one or more sample-checkpoint products are missing")
        arrays = [self.square_correlator, *self.spectrum.values(), *self.entropy_contour.values(), *self.charge_variance_contour.values()]
        if any(not np.isfinite(array).all() for array in arrays):
            raise FloatingPointError("wall-window product contains non-finite values")
        return {
            "schema": SCHEMA,
            "samples": len(self.sample_ids),
            "checkpoints": self.checkpoints.tolist(),
            "window_count": self.ny * (self.ny // 2 + 1),
            "observer_seconds": float(self.seconds.sum()),
            "peak_working_bytes_estimate": int(self.peak_working_bytes),
        }

    def save(self, directory: Path | str, *, config: dict[str, Any]) -> dict[str, Any]:
        diagnostics = self.validate()
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        files: list[dict[str, Any]] = []
        common = directory / "common.npz"
        _atomic_npz(
            common,
            schema=np.asarray(SCHEMA), config_json=np.asarray(json.dumps(config, sort_keys=True)),
            checkpoints=self.checkpoints, global_sample_ids=self.sample_ids,
            x=np.arange(self.nx), ry=np.arange(1, self.ny // 2 + 1),
            y0=np.arange(self.ny), ay=np.arange(self.ny // 2 + 1),
            wall_x_inclusive=np.asarray(
                [max(0, self.nx // 2 - max(1, self.nx // 4)),
                 min(self.nx, self.nx // 2 + max(1, self.nx // 4) + 1) - 1],
                dtype=np.int64,
            ),
            square_correlator=self.square_correlator, observer_seconds=self.seconds,
        )
        files.append({"path": common.name, "sha256": _sha256(common), "bytes": common.stat().st_size})
        for ay in range(self.ny // 2 + 1):
            path = directory / f"Ay_{ay:03d}.npz"
            _atomic_npz(
                path, schema=np.asarray(SCHEMA), ay=np.asarray(ay),
                occupation_spectrum=self.spectrum[ay],
                entropy_contour=self.entropy_contour[ay],
                charge_variance_contour=self.charge_variance_contour[ay],
            )
            files.append({"path": path.name, "sha256": _sha256(path), "bytes": path.stat().st_size})
        return {**diagnostics, "path": str(directory), "files": files, "bytes": sum(row["bytes"] for row in files)}
