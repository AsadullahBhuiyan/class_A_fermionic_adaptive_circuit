"""Periodic three-sector Chern estimator; RNG is separate from circuit RNG."""
from __future__ import annotations

import time
import numpy as np
import torch


def center_choices(root_seed, nx, ny, sample_ids, cycle, count=10):
    if not 1 <= count <= ny:
        raise ValueError("center count must lie in [1, Ny]")
    return np.stack([
        np.random.default_rng(np.random.SeedSequence(
            [int(root_seed), 2301, nx, ny, int(sample), int(cycle)]
        )).choice(ny, size=count, replace=False) for sample in sample_ids
    ])


def sector_table(nx, ny, radius):
    """All integer y centers, minimum-image coordinates; both orbitals per cell.

    x + Nx*y is the canonical unit-cell ordering. Disk membership includes R.
    Sector boundaries are [0,2pi/3), [2pi/3,4pi/3), [4pi/3,2pi).
    """
    if not 0 < radius < min(nx, ny) / 2:
        raise ValueError("disk must be smaller than the periodic half-length")
    x = np.tile(np.arange(nx), ny)
    y = np.repeat(np.arange(ny), nx)
    tables = [[], [], []]
    x0 = nx / 2
    for y0 in range(ny):
        dx = (x - x0 + nx / 2) % nx - nx / 2
        dy = (y - y0 + ny / 2) % ny - ny / 2
        angle = np.mod(np.arctan2(dy, dx), 2 * np.pi)
        inside = dx * dx + dy * dy <= radius * radius
        for sector in range(3):
            sites = np.flatnonzero(inside & (angle >= sector * 2*np.pi/3)
                                   & (angle < (sector + 1) * 2*np.pi/3))
            tables[sector].append((2 * sites[:, None] + [0, 1]).ravel())
    return tuple(np.stack(rows) for rows in tables)


@torch.no_grad()
def batched_chern(frame, ranks, centers, tables, chunk_size=10):
    """Contract Gamma=(V V^dagger)^T, without a full covariance.

    Only selected sector rows are gathered. Matmuls have leading dimensions
    (trajectory, center). Rank padding is masked, never treated as occupied.
    """
    if frame.dtype != torch.complex128 or frame.ndim != 3 or chunk_size < 1:
        raise ValueError("expected a batched complex128 frame and positive chunk")
    device = frame.device
    ranks = torch.as_tensor(ranks, device=device, dtype=torch.long)
    centers = torch.as_tensor(centers, device=device, dtype=torch.long)
    if ranks.shape != (frame.shape[0],) or centers.shape[0] != frame.shape[0]:
        raise ValueError("inconsistent trajectory dimensions")
    if bool(((ranks < 0) | (ranks > frame.shape[-1])).any()):
        raise ValueError("invalid occupied-frame ranks")
    tables = tuple(torch.as_tensor(t, device=device, dtype=torch.long) for t in tables)
    columns = torch.arange(frame.shape[-1], device=device)
    mask = columns[None, None, None, :] < ranks[:, None, None, None]
    batch_index = torch.arange(frame.shape[0], device=device)[:, None, None]
    values = []
    for start in range(0, centers.shape[1], chunk_size):
        chosen = centers[:, start:start + chunk_size]
        a, b, c = [torch.where(mask, frame[batch_index, table[chosen], :].conj(), 0)
                   for table in tables]
        ca, ab, bc = c @ a.mH, a @ b.mH, b @ c.mH
        # The reverse trace is the complex conjugate by Hermiticity.
        forward = ((ca @ ab) * bc.transpose(-2, -1)).sum(dim=(-2, -1))
        values.append(-24 * np.pi * forward.imag)
    return torch.cat(values, dim=1)


def ensemble_statistics(values):
    """Input (trajectory, cycle), after averaging centers within trajectories."""
    values = np.asarray(values)
    if values.ndim != 2 or not len(values):
        raise ValueError("expected nonempty trajectory-by-cycle array")
    sem = (values.std(axis=0, ddof=1) / np.sqrt(len(values))
           if len(values) > 1 else np.full(values.shape[1], np.nan))
    return values.mean(axis=0), sem


class RandomCenterObserver:
    def __init__(self, nx, ny, cycles, sample_ids, root_seed, radius=4.,
                 count=10, chunk_size=10):
        self.nx, self.ny = nx, ny
        self.sample_ids = np.asarray(sample_ids, dtype=np.int64)
        self.root_seed, self.count, self.chunk_size = root_seed, count, chunk_size
        self.tables = sector_table(nx, ny, radius)
        self.device_tables = {}
        shape = (len(sample_ids), cycles + 1, count)
        self.centers = np.empty(shape, dtype=np.int64)
        self.chern = np.empty(shape, dtype=np.float64)
        self.charge = np.empty(shape[:2], dtype=np.int64)
        self.seen = np.zeros(shape[:2], dtype=bool)
        self.cycle_times = {}

    def capture(self, *, cycle, state, batch_start, batch_count, **kwargs):
        frame = torch.as_tensor(state.frame)
        ranks = torch.as_tensor(state.ranks if hasattr(state, 'ranks') else state.rank,
                                device=frame.device)
        if frame.ndim == 2:  # canonical CPU single-sample observer, for tests
            frame, ranks = frame[None], ranks.reshape(1)
        stop = batch_start + batch_count
        selected = slice(batch_start, stop)
        if frame.shape[0] != batch_count or self.seen[selected, cycle].any():
            raise ValueError("unexpected or repeated native observation")
        centers = center_choices(self.root_seed, self.nx, self.ny,
                                 self.sample_ids[selected], cycle, self.count)
        if frame.device not in self.device_tables:
            self.device_tables[frame.device] = tuple(
                torch.as_tensor(t, device=frame.device) for t in self.tables)
        values = batched_chern(frame, ranks, centers,
                               self.device_tables[frame.device], self.chunk_size)
        if not bool(torch.isfinite(values).all()):
            raise ValueError("nonfinite Chern observation")
        self.centers[selected, cycle] = centers
        self.chern[selected, cycle] = values.cpu().numpy()
        self.charge[selected, cycle] = ranks.cpu().numpy()
        self.seen[selected, cycle] = True
        # CPU readback above synchronizes these contractions; synchronize all
        # work as well so the calibration interval includes observer overhead.
        if frame.is_cuda:
            torch.cuda.synchronize(frame.device)
        self.cycle_times[cycle] = time.perf_counter()

    def arrays(self):
        if not self.seen.all():
            raise ValueError("missing cycle observations")
        return dict(cycles=np.arange(self.chern.shape[1], dtype=np.int64),
                    sample_ids=self.sample_ids, centers_y=self.centers,
                    real_space_chern=self.chern,
                    center_average=self.chern.mean(axis=2), global_charge=self.charge)
