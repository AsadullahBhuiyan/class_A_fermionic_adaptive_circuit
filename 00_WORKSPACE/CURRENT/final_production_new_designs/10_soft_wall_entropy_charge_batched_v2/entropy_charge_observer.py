"""Batched endpoint entropy and charge observables for soft-wall trajectories.

Dynamics callbacks record only rank-based charge.  Expensive restricted-region
spectra are evaluated after the final occupied frame is durable.  At fixed
``Ay`` the implementation batches independent ``(trajectory, y0)`` matrices,
forms ``C_A = R R^dagger``, and diagonalizes the Hermitian Gram matrices.  The
same occupations feed von Neumann, Renyi-2, Renyi-3, charge expectation, and
intrinsic quantum charge variance.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any

import numpy as np
import torch


OBSERVER_SCHEMA = "soft_wall_entropy_charge_endpoint_observer_v2"
ENTROPY_KEYS = ("entropy_von_neumann", "entropy_renyi2", "entropy_renyi3")
SCALAR_KEYS = (*ENTROPY_KEYS, "charge_mean", "charge_variance")
CONTOUR_KEYS = (
    "contour_von_neumann",
    "contour_renyi2",
    "contour_renyi3",
    "contour_charge_variance",
)


def periodic_window_indices(
    *,
    nx: int,
    ny: int,
    y0_values: Iterable[int],
    ay: int,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Return orbital indices for periodic full-x strips.

    Rows are ordered by relative ``dy``, then ``x``, then the two orbitals.
    """

    nx, ny, ay = int(nx), int(ny), int(ay)
    y0 = torch.as_tensor(list(y0_values), dtype=torch.long, device=device)
    if nx <= 0 or ny <= 0:
        raise ValueError("nx and ny must be positive")
    if not 0 <= ay <= ny // 2:
        raise ValueError(f"ay must lie in 0..ny//2; got ay={ay}, ny={ny}")
    if y0.ndim != 1 or bool(torch.any((y0 < 0) | (y0 >= ny))):
        raise ValueError("periodic strip origins must lie in 0..ny-1")
    if ay == 0:
        return torch.empty((int(y0.numel()), 0), dtype=torch.long, device=device)
    dy = torch.arange(ay, dtype=torch.long, device=device)
    x = torch.arange(nx, dtype=torch.long, device=device)
    orbital = torch.arange(2, dtype=torch.long, device=device)
    y = (y0[:, None] + dy[None, :]) % ny
    indices = (
        2 * nx * y[:, :, None, None]
        + 2 * x[None, None, :, None]
        + orbital[None, None, None, :]
    )
    return indices.reshape(int(y0.numel()), 2 * nx * ay).contiguous()


def _entropy_weights(occupation: torch.Tensor) -> dict[str, torch.Tensor]:
    occupation = occupation.clamp(0.0, 1.0)
    complement = 1.0 - occupation
    von_neumann = -torch.xlogy(occupation, occupation) - torch.xlogy(
        complement, complement
    )
    renyi2 = -torch.log(occupation.square() + complement.square())
    renyi3 = -0.5 * torch.log(occupation.pow(3) + complement.pow(3))
    return {
        "entropy_von_neumann": von_neumann,
        "entropy_renyi2": renyi2,
        "entropy_renyi3": renyi3,
    }


def _validate_frame(frame: torch.Tensor, *, nx: int, ny: int | None = None) -> None:
    if frame.ndim != 3 or not torch.is_complex(frame):
        raise ValueError("frame must have complex shape (samples,modes,capacity)")
    if frame.dtype != torch.complex128:
        raise TypeError(f"frame must be complex128, got {frame.dtype}")
    if ny is not None and int(frame.shape[1]) != 2 * int(nx) * int(ny):
        raise ValueError("frame physical dimension does not match nx and ny")
    if not torch.isfinite(frame).all():
        raise FloatingPointError("occupied frame contains nonfinite values")


def validate_padded_frame_ranks(
    frame: torch.Tensor,
    ranks: torch.Tensor | np.ndarray,
    *,
    zero_tolerance: float = 1.0e-12,
) -> None:
    """Validate variable-rank frames without allocating a second full frame."""

    rank_tensor = torch.as_tensor(ranks, dtype=torch.int64, device=frame.device)
    if rank_tensor.shape != (int(frame.shape[0]),):
        raise ValueError("rank vector does not match the frame sample axis")
    if bool(torch.any((rank_tensor < 0) | (rank_tensor > int(frame.shape[2])))):
        raise ValueError("one or more ranks lie outside frame capacity")
    for sample, rank in enumerate(rank_tensor.detach().cpu().tolist()):
        if rank < int(frame.shape[2]):
            maximum = float(frame[sample, :, rank:].abs().amax().detach().cpu())
            if maximum > float(zero_tolerance):
                raise FloatingPointError(
                    f"sample {sample} has nonzero padded frame columns: {maximum:.3e}"
                )


def _pair_rows(
    frame: torch.Tensor,
    *,
    sample_indices: torch.Tensor,
    origins: torch.Tensor,
    nx: int,
    ny: int,
    ay: int,
) -> torch.Tensor:
    indices = periodic_window_indices(
        nx=nx,
        ny=ny,
        y0_values=origins.detach().cpu().tolist(),
        ay=ay,
        device=frame.device,
    )
    return frame[sample_indices[:, None], indices]


def _gram_occupations(
    rows: torch.Tensor,
    *,
    return_vectors: bool,
    occupation_tolerance: float,
) -> tuple[torch.Tensor, torch.Tensor | None, float, float]:
    gram = rows @ rows.mH
    gram = 0.5 * (gram + gram.mH)
    if return_vectors:
        raw, vectors = torch.linalg.eigh(gram)
    else:
        raw = torch.linalg.eigvalsh(gram)
        vectors = None
    minimum = float(raw.amin().detach().cpu())
    maximum = float(raw.amax().detach().cpu())
    tolerance = float(occupation_tolerance)
    if minimum < -tolerance or maximum > 1.0 + tolerance:
        raise FloatingPointError(
            "restricted occupation outside [0,1] tolerance: "
            f"min={minimum:.6e}, max={maximum:.6e}"
        )
    return raw.clamp(0.0, 1.0), vectors, minimum, maximum


def _scalars_from_occupations(occupation: torch.Tensor) -> dict[str, torch.Tensor]:
    weights = _entropy_weights(occupation)
    variance = occupation * (1.0 - occupation)
    return {
        **{key: value.sum(-1).to(torch.float64) for key, value in weights.items()},
        "charge_mean": occupation.sum(-1).to(torch.float64),
        "charge_variance": variance.sum(-1).to(torch.float64),
    }


def _contours_from_spectrum(
    vectors: torch.Tensor,
    occupation: torch.Tensor,
    *,
    nx: int,
    ay: int,
) -> dict[str, torch.Tensor]:
    entropy = _entropy_weights(occupation)
    weights = {
        "contour_von_neumann": entropy["entropy_von_neumann"],
        "contour_renyi2": entropy["entropy_renyi2"],
        "contour_renyi3": entropy["entropy_renyi3"],
        "contour_charge_variance": occupation * (1.0 - occupation),
    }
    probabilities = vectors.abs().square()
    result: dict[str, torch.Tensor] = {}
    for key, weight in weights.items():
        orbital = probabilities @ weight.unsqueeze(-1)
        result[key] = (
            orbital.squeeze(-1)
            .reshape(int(vectors.shape[0]), ay, nx, 2)
            .sum(-1)
            .transpose(-2, -1)
            .contiguous()
            .to(torch.float64)
        )
    return result


def frame_window_observables(
    frame: torch.Tensor,
    *,
    indices: torch.Tensor,
    nx: int,
    ay: int,
    return_contours: bool,
    occupation_tolerance: float = 1.0e-8,
) -> dict[str, torch.Tensor]:
    """Dense-reference-friendly Cartesian sample/origin evaluator."""

    nx, ay = int(nx), int(ay)
    _validate_frame(frame, nx=nx)
    if indices.ndim != 2 or int(indices.shape[1]) != 2 * nx * ay:
        raise ValueError("window-index shape does not match nx and ay")
    if indices.dtype != torch.long:
        raise TypeError("window indices must use torch.long")
    samples, origins = int(frame.shape[0]), int(indices.shape[0])
    shape = (samples, origins)
    if ay == 0:
        zeros = torch.zeros(shape, dtype=torch.float64, device=frame.device)
        result = {key: zeros.clone() for key in SCALAR_KEYS}
        result.update(
            occupation_min=torch.zeros((), dtype=torch.float64, device=frame.device),
            occupation_max=torch.zeros((), dtype=torch.float64, device=frame.device),
        )
        if return_contours:
            empty = torch.zeros(
                (samples, origins, nx, 0), dtype=torch.float64, device=frame.device
            )
            result.update({key: empty.clone() for key in CONTOUR_KEYS})
        return result

    idx = indices.to(frame.device)
    rows = frame.index_select(1, idx.reshape(-1)).reshape(
        samples * origins, 2 * nx * ay, int(frame.shape[2])
    )
    occupation, vectors, minimum, maximum = _gram_occupations(
        rows,
        return_vectors=return_contours,
        occupation_tolerance=occupation_tolerance,
    )
    result = {
        key: value.reshape(shape) for key, value in _scalars_from_occupations(occupation).items()
    }
    result["occupation_min"] = torch.as_tensor(
        minimum, dtype=torch.float64, device=frame.device
    )
    result["occupation_max"] = torch.as_tensor(
        maximum, dtype=torch.float64, device=frame.device
    )
    if vectors is not None:
        result.update(
            {
                key: value.reshape(samples, origins, nx, ay)
                for key, value in _contours_from_spectrum(
                    vectors, occupation, nx=nx, ay=ay
                ).items()
            }
        )
    return result


def frame_y0_averaged_width(
    frame: torch.Tensor,
    *,
    nx: int,
    ny: int,
    ay: int,
    matrix_batch_size: int,
    occupation_tolerance: float = 1.0e-8,
    pair_progress: Callable[[int], None] | None = None,
) -> dict[str, Any]:
    """Compute one exact all-origin width in flattened matrix batches."""

    nx, ny, ay = int(nx), int(ny), int(ay)
    matrix_batch_size = int(matrix_batch_size)
    _validate_frame(frame, nx=nx, ny=ny)
    if not 0 <= ay <= ny // 2:
        raise ValueError("ay lies outside 0..ny//2")
    if matrix_batch_size <= 0:
        raise ValueError("matrix_batch_size must be positive")
    samples = int(frame.shape[0])
    if ay == 0:
        return {
            **{key: np.zeros(samples, dtype=np.float64) for key in SCALAR_KEYS},
            "occupation_min": 0.0,
            "occupation_max": 0.0,
            "pairs": 0,
        }

    sample_ids = torch.arange(samples, device=frame.device).repeat_interleave(ny)
    origins = torch.arange(ny, device=frame.device).repeat(samples)
    totals = {
        key: torch.zeros(samples, dtype=torch.float64, device=frame.device)
        for key in SCALAR_KEYS
    }
    minimum, maximum = np.inf, -np.inf
    with torch.inference_mode():
        for start in range(0, samples * ny, matrix_batch_size):
            stop = min(samples * ny, start + matrix_batch_size)
            block_samples = sample_ids[start:stop]
            rows = _pair_rows(
                frame,
                sample_indices=block_samples,
                origins=origins[start:stop],
                nx=nx,
                ny=ny,
                ay=ay,
            )
            occupation, _, block_minimum, block_maximum = _gram_occupations(
                rows,
                return_vectors=False,
                occupation_tolerance=occupation_tolerance,
            )
            scalars = _scalars_from_occupations(occupation)
            for key in SCALAR_KEYS:
                totals[key].index_add_(0, block_samples, scalars[key])
            minimum = min(minimum, block_minimum)
            maximum = max(maximum, block_maximum)
            if pair_progress is not None:
                pair_progress(stop - start)
    return {
        **{
            key: (value / float(ny)).detach().cpu().numpy()
            for key, value in totals.items()
        },
        "occupation_min": float(minimum),
        "occupation_max": float(maximum),
        "pairs": samples * ny,
    }


def frame_fixed_half_strip_contours(
    frame: torch.Tensor,
    *,
    nx: int,
    ny: int,
    matrix_batch_size: int,
    occupation_tolerance: float = 1.0e-8,
) -> dict[str, Any]:
    """Return fixed ``y0=0, Ay=ny//2`` trajectory-resolved contours."""

    _validate_frame(frame, nx=nx, ny=ny)
    ay = int(ny) // 2
    samples = int(frame.shape[0])
    output = {
        key: np.full((samples, int(nx), ay), np.nan, dtype=np.float64)
        for key in CONTOUR_KEYS
    }
    fixed_scalars = {
        key: np.full(samples, np.nan, dtype=np.float64)
        for key in (*ENTROPY_KEYS, "charge_variance")
    }
    minimum, maximum = np.inf, -np.inf
    indices = periodic_window_indices(nx=nx, ny=ny, y0_values=[0], ay=ay, device=frame.device)
    with torch.inference_mode():
        for start in range(0, samples, int(matrix_batch_size)):
            stop = min(samples, start + int(matrix_batch_size))
            block = frame_window_observables(
                frame[start:stop],
                indices=indices,
                nx=nx,
                ay=ay,
                return_contours=True,
                occupation_tolerance=occupation_tolerance,
            )
            for key in CONTOUR_KEYS:
                output[key][start:stop] = block[key][:, 0].detach().cpu().numpy()
            for key in fixed_scalars:
                fixed_scalars[key][start:stop] = block[key][:, 0].detach().cpu().numpy()
            minimum = min(minimum, float(block["occupation_min"].detach().cpu()))
            maximum = max(maximum, float(block["occupation_max"].detach().cpu()))
    return {
        **output,
        **{f"fixed_scalar__{key}": value for key, value in fixed_scalars.items()},
        "occupation_min": minimum,
        "occupation_max": maximum,
    }


def frame_y0_averaged_observables(
    frame: torch.Tensor,
    *,
    nx: int,
    ny: int,
    ay_values: Iterable[int],
    matrix_batch_size: int = 64,
    occupation_tolerance: float = 1.0e-8,
) -> dict[str, Any]:
    """Convenience evaluator used by tests and small analyses."""

    ay_values = tuple(int(value) for value in ay_values)
    curves = {
        key: np.zeros((int(frame.shape[0]), len(ay_values)), dtype=np.float64)
        for key in SCALAR_KEYS
    }
    minimum, maximum = np.inf, -np.inf
    for width_index, ay in enumerate(ay_values):
        block = frame_y0_averaged_width(
            frame,
            nx=nx,
            ny=ny,
            ay=ay,
            matrix_batch_size=matrix_batch_size,
            occupation_tolerance=occupation_tolerance,
        )
        for key in SCALAR_KEYS:
            curves[key][:, width_index] = block[key]
        minimum = min(minimum, block["occupation_min"])
        maximum = max(maximum, block["occupation_max"])
    if not np.isfinite(minimum):
        minimum = maximum = 0.0
    return {**curves, "occupation_min": minimum, "occupation_max": maximum}


class SoftWallEntropyChargeObserver:
    """Checkpointable charge history plus separately evaluated endpoint arrays."""

    def __init__(
        self,
        *,
        nx: int,
        ny: int,
        physical_cycles: int,
        sample_ids: Iterable[int],
        occupation_tolerance: float = 1.0e-8,
    ) -> None:
        self.nx, self.ny = int(nx), int(ny)
        self.physical_cycles = int(physical_cycles)
        self.sample_ids = np.asarray(list(sample_ids), dtype=np.int64)
        self.samples = int(self.sample_ids.size)
        self.occupation_tolerance = float(occupation_tolerance)
        if self.physical_cycles != 2 * self.ny:
            raise ValueError("locked campaign requires physical_cycles=2*ny")
        if self.samples <= 0 or len(np.unique(self.sample_ids)) != self.samples:
            raise ValueError("sample_ids must be nonempty and unique")
        self.cycles = np.arange(self.physical_cycles + 1, dtype=np.int64)
        self.ay_values = np.arange(self.ny // 2 + 1, dtype=np.int64)
        self.seen_cycles = np.zeros(self.physical_cycles + 1, dtype=np.bool_)
        self.global_charge = np.full((self.samples, self.physical_cycles + 1), -1, dtype=np.int64)
        shape = (self.samples, len(self.ay_values))
        self.endpoint = {
            key: np.full(shape, np.nan, dtype=np.float64) for key in SCALAR_KEYS
        }
        half = self.ny // 2
        self.fixed_contours = {
            key: np.full((self.samples, self.nx, half), np.nan, dtype=np.float64)
            for key in CONTOUR_KEYS
        }
        self.fixed_scalars = {
            key: np.full(self.samples, np.nan, dtype=np.float64)
            for key in (*ENTROPY_KEYS, "charge_variance")
        }
        self.endpoint_width_seen = np.zeros(len(self.ay_values), dtype=np.bool_)
        self.fixed_contours_seen = False
        self.minimum_occupation = np.inf
        self.maximum_occupation = -np.inf
        self.matrix_batch_size_by_ay = np.zeros(len(self.ay_values), dtype=np.int64)
        self.endpoint_seconds_by_ay = np.zeros(len(self.ay_values), dtype=np.float64)

    def __call__(self, *, cycle: int, state: Any, batch_start: int = 0, batch_count: int | None = None, **_: Any) -> None:
        cycle = int(cycle)
        batch_count = self.samples if batch_count is None else int(batch_count)
        if not 0 <= cycle <= self.physical_cycles:
            raise IndexError("cycle is outside the physical range")
        if int(batch_start) != 0 or batch_count != self.samples:
            raise ValueError("observer requires the complete execution batch")
        if self.seen_cycles[cycle]:
            raise RuntimeError(f"duplicate observation at global cycle {cycle}")
        if not hasattr(state, "frame") or not hasattr(state, "ranks"):
            raise TypeError("observer requires native occupied-frame state")
        ranks = state.ranks.detach().cpu().numpy().astype(np.int64, copy=False)
        if ranks.shape != (self.samples,) or np.any(ranks < 0) or np.any(ranks > 2 * self.nx * self.ny):
            raise ValueError("invalid occupied-frame ranks")
        self.global_charge[:, cycle] = ranks
        self.seen_cycles[cycle] = True

    def record_endpoint_width(
        self,
        frame: torch.Tensor,
        *,
        ay: int,
        matrix_batch_size: int,
        elapsed_seconds: float,
        pair_progress: Callable[[int], None] | None = None,
    ) -> None:
        ay = int(ay)
        width_index = int(np.where(self.ay_values == ay)[0][0])
        if self.endpoint_width_seen[width_index]:
            raise RuntimeError(f"endpoint width Ay={ay} is already complete")
        block = frame_y0_averaged_width(
            frame,
            nx=self.nx,
            ny=self.ny,
            ay=ay,
            matrix_batch_size=matrix_batch_size,
            occupation_tolerance=self.occupation_tolerance,
            pair_progress=pair_progress,
        )
        for key in SCALAR_KEYS:
            self.endpoint[key][:, width_index] = block[key]
        if ay == self.ny // 2:
            contours = frame_fixed_half_strip_contours(
                frame,
                nx=self.nx,
                ny=self.ny,
                matrix_batch_size=matrix_batch_size,
                occupation_tolerance=self.occupation_tolerance,
            )
            for key in CONTOUR_KEYS:
                self.fixed_contours[key][:] = contours[key]
            for key in self.fixed_scalars:
                self.fixed_scalars[key][:] = contours[f"fixed_scalar__{key}"]
            self.fixed_contours_seen = True
            self.minimum_occupation = min(self.minimum_occupation, contours["occupation_min"])
            self.maximum_occupation = max(self.maximum_occupation, contours["occupation_max"])
        self.minimum_occupation = min(self.minimum_occupation, block["occupation_min"])
        self.maximum_occupation = max(self.maximum_occupation, block["occupation_max"])
        self.matrix_batch_size_by_ay[width_index] = int(matrix_batch_size)
        self.endpoint_seconds_by_ay[width_index] = float(elapsed_seconds)
        self.endpoint_width_seen[width_index] = True

    def _identity_payload(self) -> dict[str, np.ndarray]:
        return {
            "observer_schema": np.asarray(OBSERVER_SCHEMA),
            "nx": np.asarray(self.nx, dtype=np.int64),
            "ny": np.asarray(self.ny, dtype=np.int64),
            "physical_cycles": np.asarray(self.physical_cycles, dtype=np.int64),
            "sample_ids": self.sample_ids.copy(),
            "cycles": self.cycles.copy(),
            "ay_values": self.ay_values.copy(),
        }

    def checkpoint_payload(self) -> dict[str, np.ndarray]:
        payload = {
            **self._identity_payload(),
            "seen_cycles": self.seen_cycles.copy(),
            "global_charge": self.global_charge.copy(),
            "endpoint_width_seen": self.endpoint_width_seen.copy(),
            "fixed_contours_seen": np.asarray(self.fixed_contours_seen, dtype=np.bool_),
            "minimum_occupation": np.asarray(self.minimum_occupation, dtype=np.float64),
            "maximum_occupation": np.asarray(self.maximum_occupation, dtype=np.float64),
            "matrix_batch_size_by_ay": self.matrix_batch_size_by_ay.copy(),
            "endpoint_seconds_by_ay": self.endpoint_seconds_by_ay.copy(),
        }
        payload.update({f"endpoint__{key}": value.copy() for key, value in self.endpoint.items()})
        payload.update({f"fixed__{key}": value.copy() for key, value in self.fixed_contours.items()})
        payload.update(
            {f"fixed_scalar__{key}": value.copy() for key, value in self.fixed_scalars.items()}
        )
        return payload

    state_dict = checkpoint_payload

    def restore_checkpoint(self, payload: Mapping[str, Any]) -> None:
        expected = self._identity_payload()
        for key, value in expected.items():
            if key not in payload or not np.array_equal(np.asarray(payload[key]), value):
                raise ValueError(f"observer checkpoint identity mismatch for {key}")
        template = self.checkpoint_payload()
        missing = sorted(set(template).difference(payload))
        if missing:
            raise ValueError(f"observer checkpoint missing keys: {missing}")
        for key, value in template.items():
            actual = np.asarray(payload[key])
            if actual.shape != value.shape or actual.dtype != value.dtype:
                raise ValueError(f"observer checkpoint array mismatch for {key}")
        self.seen_cycles[:] = payload["seen_cycles"]
        self.global_charge[:] = payload["global_charge"]
        self.endpoint_width_seen[:] = payload["endpoint_width_seen"]
        self.fixed_contours_seen = bool(np.asarray(payload["fixed_contours_seen"]).item())
        self.minimum_occupation = float(np.asarray(payload["minimum_occupation"]).item())
        self.maximum_occupation = float(np.asarray(payload["maximum_occupation"]).item())
        self.matrix_batch_size_by_ay[:] = payload["matrix_batch_size_by_ay"]
        self.endpoint_seconds_by_ay[:] = payload["endpoint_seconds_by_ay"]
        for key in SCALAR_KEYS:
            self.endpoint[key][:] = payload[f"endpoint__{key}"]
        for key in CONTOUR_KEYS:
            self.fixed_contours[key][:] = payload[f"fixed__{key}"]
        for key in self.fixed_scalars:
            self.fixed_scalars[key][:] = payload[f"fixed_scalar__{key}"]

    load_state_dict = restore_checkpoint

    def validate(self, *, require_dynamics: bool, require_endpoint: bool) -> dict[str, Any]:
        seen = np.flatnonzero(self.seen_cycles)
        if seen.size and not np.array_equal(seen, np.arange(int(seen[-1]) + 1)):
            raise RuntimeError("observed cycles are not a contiguous prefix")
        if np.any(self.global_charge[:, self.seen_cycles] < 0):
            raise FloatingPointError("observed global charge is negative")
        if require_dynamics and not bool(np.all(self.seen_cycles)):
            raise RuntimeError("one or more global-charge cycles are missing")
        if self.endpoint_width_seen[0]:
            for key in SCALAR_KEYS:
                if not np.all(self.endpoint[key][:, 0] == 0.0):
                    raise FloatingPointError(f"Ay=0 {key} must be exactly zero")
        completed = self.endpoint_width_seen
        for key in SCALAR_KEYS:
            if completed.any() and not np.isfinite(self.endpoint[key][:, completed]).all():
                raise FloatingPointError(f"endpoint {key} contains nonfinite values")
            if completed.any() and np.min(self.endpoint[key][:, completed]) < -1.0e-9:
                raise FloatingPointError(f"endpoint {key} is negative below tolerance")
        if self.fixed_contours_seen:
            for key in CONTOUR_KEYS:
                value = self.fixed_contours[key]
                if not np.isfinite(value).all() or np.min(value) < -1.0e-9:
                    raise FloatingPointError(f"fixed contour {key} is invalid")
            for key, value in self.fixed_scalars.items():
                if not np.isfinite(value).all() or np.min(value) < -1.0e-9:
                    raise FloatingPointError(f"fixed scalar {key} is invalid")
        if require_endpoint:
            if not bool(np.all(self.endpoint_width_seen)):
                raise RuntimeError("one or more endpoint widths are missing")
            if not self.fixed_contours_seen:
                raise RuntimeError("fixed half-strip contours are missing")
        return {
            "schema": OBSERVER_SCHEMA,
            "cycles_seen": int(self.seen_cycles.sum()),
            "widths_seen": int(self.endpoint_width_seen.sum()),
            "minimum_occupation": float(self.minimum_occupation),
            "maximum_occupation": float(self.maximum_occupation),
            "covariance_materializations": 0,
        }

    def result_payload(self, sample_slice: slice) -> dict[str, np.ndarray]:
        self.validate(require_dynamics=True, require_endpoint=True)
        indices = np.arange(self.samples)[sample_slice]
        if indices.size == 0 or not np.array_equal(indices, np.arange(indices[0], indices[-1] + 1)):
            raise ValueError("result shard must select one contiguous nonempty range")
        payload = {
            **self._identity_payload(),
            "sample_ids": self.sample_ids[indices].copy(),
            "global_charge": self.global_charge[indices].copy(),
            "half_filling_offset": (self.global_charge[indices] - self.nx * self.ny).copy(),
            "endpoint_cycle": np.asarray(self.physical_cycles, dtype=np.int64),
            "origin_average_count": np.asarray(self.ny, dtype=np.int64),
            "fixed_contour_y0": np.asarray(0, dtype=np.int64),
            "fixed_contour_ay": np.asarray(self.ny // 2, dtype=np.int64),
            "minimum_occupation": np.asarray(self.minimum_occupation, dtype=np.float64),
            "maximum_occupation": np.asarray(self.maximum_occupation, dtype=np.float64),
            "matrix_batch_size_by_ay": self.matrix_batch_size_by_ay.copy(),
            "endpoint_seconds_by_ay": self.endpoint_seconds_by_ay.copy(),
        }
        payload.update({f"endpoint__{key}": value[indices].copy() for key, value in self.endpoint.items()})
        payload.update({f"fixed__{key}": value[indices].copy() for key, value in self.fixed_contours.items()})
        payload.update(
            {
                f"fixed_scalar__{key}": value[indices].copy()
                for key, value in self.fixed_scalars.items()
            }
        )
        return payload


__all__ = [
    "CONTOUR_KEYS",
    "ENTROPY_KEYS",
    "SoftWallEntropyChargeObserver",
    "OBSERVER_SCHEMA",
    "SCALAR_KEYS",
    "frame_fixed_half_strip_contours",
    "frame_window_observables",
    "frame_y0_averaged_observables",
    "frame_y0_averaged_width",
    "periodic_window_indices",
    "validate_padded_frame_ranks",
]
