"""Frame-native entropy and charge observables for the hard-wall campaign.

The public observer stores *trajectory-resolved* quantities after averaging over
all periodic strip origins.  It never constructs a covariance matrix.  Every
restricted occupied-frame SVD is shared by the entropy, charge, variance, and
(when requested) contour estimators for that window.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

import numpy as np
import torch


OBSERVER_SCHEMA = "hard_wall_entropy_charge_observer_v1"


def periodic_window_indices(
    *,
    nx: int,
    ny: int,
    y0_values: Iterable[int],
    ay: int,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Return indices for periodic ``[0,nx) x [y0,y0+ay)`` strips.

    The returned shape is ``(len(y0_values), 2*nx*ay)``.  Within each row the
    ordering is relative y displacement, x coordinate, then orbital.
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


def _binary_entropy(probability: torch.Tensor) -> torch.Tensor:
    probability = probability.clamp(0.0, 1.0)
    complement = 1.0 - probability
    return -torch.xlogy(probability, probability) - torch.xlogy(
        complement, complement
    )


def frame_window_observables(
    frame: torch.Tensor,
    *,
    indices: torch.Tensor,
    nx: int,
    ay: int,
    return_contours: bool,
    occupation_tolerance: float = 1.0e-8,
) -> dict[str, torch.Tensor]:
    """Evaluate one or more strip placements from padded occupied frames.

    Parameters
    ----------
    frame:
        Complex tensor ``(samples, physical_modes, capacity)``.  Inactive
        capacity columns must be zero, as guaranteed by
        ``BatchedOccupiedFrameState``.
    indices:
        Integer tensor ``(origins, 2*nx*ay)`` from
        :func:`periodic_window_indices`.

    Returns
    -------
    Scalars have shape ``(samples, origins)``.  Requested cell contours have
    shape ``(samples, origins, nx, ay)``.
    """

    nx, ay = int(nx), int(ay)
    if frame.ndim != 3 or not torch.is_complex(frame):
        raise ValueError("frame must be a complex tensor of shape (samples,modes,capacity)")
    if indices.ndim != 2 or int(indices.shape[1]) != 2 * nx * ay:
        raise ValueError("window-index shape does not match nx and ay")
    if indices.dtype != torch.long:
        raise TypeError("window indices must use torch.long")
    if not torch.isfinite(frame).all():
        raise FloatingPointError("occupied frame contains nonfinite values")

    samples = int(frame.shape[0])
    origins = int(indices.shape[0])
    real_dtype = frame.real.dtype
    scalar_shape = (samples, origins)
    if ay == 0:
        result = {
            "entropy": torch.zeros(scalar_shape, dtype=real_dtype, device=frame.device),
            "charge_mean": torch.zeros(scalar_shape, dtype=real_dtype, device=frame.device),
            "charge_variance": torch.zeros(
                scalar_shape, dtype=real_dtype, device=frame.device
            ),
            "occupation_min": torch.zeros((), dtype=real_dtype, device=frame.device),
            "occupation_max": torch.zeros((), dtype=real_dtype, device=frame.device),
        }
        if return_contours:
            empty = torch.zeros(
                (samples, origins, nx, 0), dtype=real_dtype, device=frame.device
            )
            result["entropy_contour"] = empty
            result["charge_variance_contour"] = empty.clone()
        return result

    idx = indices.to(frame.device)
    rows = frame.index_select(1, idx.reshape(-1)).reshape(
        samples * origins, 2 * nx * ay, int(frame.shape[2])
    )
    if return_contours:
        vectors, singular, _ = torch.linalg.svd(rows, full_matrices=False)
    else:
        singular = torch.linalg.svdvals(rows)
        vectors = None
    raw_occupation = singular.square().real
    occupation_min = raw_occupation.amin()
    occupation_max = raw_occupation.amax()
    tolerance = float(occupation_tolerance)
    if float(occupation_min.detach().cpu()) < -tolerance or float(
        occupation_max.detach().cpu()
    ) > 1.0 + tolerance:
        raise FloatingPointError(
            "restricted-frame occupation lies outside [0,1] tolerance: "
            f"min={float(occupation_min):.6e}, max={float(occupation_max):.6e}"
        )
    occupation = raw_occupation.clamp(0.0, 1.0)
    entropy_weight = _binary_entropy(occupation)
    variance_weight = occupation * (1.0 - occupation)
    result = {
        "entropy": entropy_weight.sum(-1).reshape(scalar_shape).to(torch.float64),
        "charge_mean": occupation.sum(-1).reshape(scalar_shape).to(torch.float64),
        "charge_variance": variance_weight.sum(-1)
        .reshape(scalar_shape)
        .to(torch.float64),
        "occupation_min": occupation_min.to(torch.float64),
        "occupation_max": occupation_max.to(torch.float64),
    }
    if vectors is not None:
        entropy_orbital = vectors.abs().square() @ entropy_weight.unsqueeze(-1)
        variance_orbital = vectors.abs().square() @ variance_weight.unsqueeze(-1)
        contour_shape = (samples, origins, ay, nx, 2)
        entropy_contour = (
            entropy_orbital.squeeze(-1)
            .reshape(contour_shape)
            .sum(-1)
            .transpose(-2, -1)
            .contiguous()
        )
        variance_contour = (
            variance_orbital.squeeze(-1)
            .reshape(contour_shape)
            .sum(-1)
            .transpose(-2, -1)
            .contiguous()
        )
        result["entropy_contour"] = entropy_contour.to(torch.float64)
        result["charge_variance_contour"] = variance_contour.to(torch.float64)
    return result


def frame_y0_averaged_observables(
    frame: torch.Tensor,
    *,
    nx: int,
    ny: int,
    ay_values: Iterable[int],
    contour_ays: Iterable[int] = (),
    sample_chunk: int = 20,
    y0_chunk: int = 4,
    occupation_tolerance: float = 1.0e-8,
) -> dict[str, Any]:
    """Average strip observables over all origins inside each trajectory.

    This function deliberately never mixes trajectories.  Returned curve arrays
    have shape ``(samples, len(ay_values))``; each requested contour has shape
    ``(samples, nx, ay)`` and is likewise an origin average.  The production
    defaults expose 80 restricted matrices per batched SVD (20 trajectories by
    four origins), a conservative workload for a 40-GB A100.
    """

    nx, ny = int(nx), int(ny)
    sample_chunk, y0_chunk = int(sample_chunk), int(y0_chunk)
    ay_values = tuple(int(value) for value in ay_values)
    contour_ays = frozenset(int(value) for value in contour_ays)
    if frame.ndim != 3 or int(frame.shape[1]) != 2 * nx * ny:
        raise ValueError("frame physical dimension does not match nx and ny")
    if sample_chunk <= 0 or y0_chunk <= 0:
        raise ValueError("sample_chunk and y0_chunk must be positive")
    if len(set(ay_values)) != len(ay_values):
        raise ValueError("ay_values must not contain duplicates")
    if any(value < 0 or value > ny // 2 for value in ay_values):
        raise ValueError("all ay_values must lie in 0..ny//2")
    if not contour_ays.issubset(ay_values):
        raise ValueError("contour_ays must be a subset of ay_values")

    samples = int(frame.shape[0])
    width_count = len(ay_values)
    curves = {
        name: np.zeros((samples, width_count), dtype=np.float64)
        for name in ("entropy", "charge_mean", "charge_variance")
    }
    contours = {
        name: {
            ay: np.zeros((samples, nx, ay), dtype=np.float64)
            for ay in sorted(contour_ays)
        }
        for name in ("entropy", "charge_variance")
    }
    min_occupation = np.inf
    max_occupation = -np.inf

    with torch.inference_mode():
        for width_index, ay in enumerate(ay_values):
            if ay == 0:
                continue
            for sample_start in range(0, samples, sample_chunk):
                sample_stop = min(samples, sample_start + sample_chunk)
                frame_part = frame[sample_start:sample_stop]
                scalar_totals = {
                    name: torch.zeros(
                        sample_stop - sample_start,
                        dtype=torch.float64,
                        device=frame.device,
                    )
                    for name in curves
                }
                contour_totals = None
                if ay in contour_ays:
                    contour_totals = {
                        name: torch.zeros(
                            (sample_stop - sample_start, nx, ay),
                            dtype=torch.float64,
                            device=frame.device,
                        )
                        for name in contours
                    }
                for y0_start in range(0, ny, y0_chunk):
                    y0_stop = min(ny, y0_start + y0_chunk)
                    indices = periodic_window_indices(
                        nx=nx,
                        ny=ny,
                        y0_values=range(y0_start, y0_stop),
                        ay=ay,
                        device=frame.device,
                    )
                    block = frame_window_observables(
                        frame_part,
                        indices=indices,
                        nx=nx,
                        ay=ay,
                        return_contours=ay in contour_ays,
                        occupation_tolerance=occupation_tolerance,
                    )
                    for name in curves:
                        scalar_totals[name] += block[name].sum(dim=1)
                    if contour_totals is not None:
                        contour_totals["entropy"] += block["entropy_contour"].sum(dim=1)
                        contour_totals["charge_variance"] += block[
                            "charge_variance_contour"
                        ].sum(dim=1)
                    min_occupation = min(
                        min_occupation, float(block["occupation_min"].detach().cpu())
                    )
                    max_occupation = max(
                        max_occupation, float(block["occupation_max"].detach().cpu())
                    )
                for name in curves:
                    curves[name][sample_start:sample_stop, width_index] = (
                        scalar_totals[name] / float(ny)
                    ).detach().cpu().numpy()
                if contour_totals is not None:
                    for name in contours:
                        contours[name][ay][sample_start:sample_stop] = (
                            contour_totals[name] / float(ny)
                        ).detach().cpu().numpy()

    if not np.isfinite(min_occupation):
        min_occupation = 0.0
        max_occupation = 0.0
    return {
        **curves,
        "entropy_contours": contours["entropy"],
        "charge_variance_contours": contours["charge_variance"],
        "occupation_min": float(min_occupation),
        "occupation_max": float(max_occupation),
    }


def _scalar_from_mapping(payload: Mapping[str, Any], key: str) -> Any:
    value = np.asarray(payload[key])
    return value.item() if value.ndim == 0 else value


class HardWallEntropyChargeObserver:
    """Collect checkpointable compact products for one execution batch.

    ``detailed=True`` is valid only for ``ny=40``.  The callable follows the
    canonical native-cycle-observer interface.  The runner must pass global
    physical cycle numbers when a trajectory is split into five-cycle engine
    calls.
    """

    def __init__(
        self,
        *,
        nx: int,
        ny: int,
        physical_cycles: int,
        sample_ids: Iterable[int] | None = None,
        samples: int | None = None,
        detailed: bool,
        sample_chunk: int = 20,
        y0_chunk: int = 4,
        occupation_tolerance: float = 1.0e-8,
    ) -> None:
        self.nx, self.ny = int(nx), int(ny)
        self.physical_cycles = int(physical_cycles)
        self.detailed = bool(detailed)
        if sample_ids is None:
            if samples is None:
                raise ValueError("provide sample_ids or samples")
            sample_ids = range(int(samples))
        self.sample_ids = np.asarray(list(sample_ids), dtype=np.int64)
        if samples is not None and int(samples) != len(self.sample_ids):
            raise ValueError("samples disagrees with sample_ids")
        self.samples = int(len(self.sample_ids))
        self.sample_chunk, self.y0_chunk = int(sample_chunk), int(y0_chunk)
        self.occupation_tolerance = float(occupation_tolerance)
        if self.nx <= 0 or self.ny <= 0 or self.samples <= 0:
            raise ValueError("nx, ny, and sample count must be positive")
        if self.physical_cycles != 2 * self.ny:
            raise ValueError("the locked campaign requires physical_cycles=2*ny")
        if self.detailed and self.ny != 40:
            raise ValueError("cycle-resolved strip products are locked to ny=40")
        if len(np.unique(self.sample_ids)) != self.samples:
            raise ValueError("sample_ids must be unique")

        self.cycles = np.arange(self.physical_cycles + 1, dtype=np.int64)
        self.ay_values = np.arange(self.ny // 2 + 1, dtype=np.int64)
        self.seen_cycles = np.zeros(self.physical_cycles + 1, dtype=np.bool_)
        self.global_charge = np.full(
            (self.samples, self.physical_cycles + 1), -1, dtype=np.int64
        )
        width_shape = (self.samples, len(self.ay_values))
        self.endpoint_entropy = np.full(width_shape, np.nan, dtype=np.float64)
        self.endpoint_charge_mean = np.full(width_shape, np.nan, dtype=np.float64)
        self.endpoint_charge_variance = np.full(width_shape, np.nan, dtype=np.float64)
        self.endpoint_seen = False
        self.min_occupation = np.inf
        self.max_occupation = -np.inf

        if self.detailed:
            self.curve_cycles = np.arange(10, self.physical_cycles + 1, dtype=np.int64)
            detailed_shape = (self.samples, len(self.curve_cycles), len(self.ay_values))
            self.entropy_curves = np.full(detailed_shape, np.nan, dtype=np.float64)
            self.charge_mean_curves = np.full(detailed_shape, np.nan, dtype=np.float64)
            self.charge_variance_curves = np.full(
                detailed_shape, np.nan, dtype=np.float64
            )
            self.curve_seen = np.zeros(len(self.curve_cycles), dtype=np.bool_)
            half = self.ny // 2
            contour_shape = (
                self.samples,
                self.physical_cycles + 1,
                self.nx,
                half,
            )
            self.half_strip_entropy_contour = np.full(
                contour_shape, np.nan, dtype=np.float64
            )
            self.half_strip_charge_variance_contour = np.full(
                contour_shape, np.nan, dtype=np.float64
            )
            final_shape = (self.samples, len(self.ay_values), self.nx, half)
            self.final_entropy_contour = np.zeros(final_shape, dtype=np.float64)
            self.final_charge_variance_contour = np.zeros(
                final_shape, dtype=np.float64
            )
            self.final_contour_seen = False

    def __call__(
        self,
        *,
        cycle: int,
        state: Any,
        batch_index: int = 0,
        batch_start: int = 0,
        batch_count: int | None = None,
        **_: Any,
    ) -> None:
        """Record one global physical cycle from a native frame state."""

        del batch_index
        cycle = int(cycle)
        batch_start = int(batch_start)
        batch_count = self.samples if batch_count is None else int(batch_count)
        if not 0 <= cycle <= self.physical_cycles:
            raise IndexError(f"cycle {cycle} is outside 0..{self.physical_cycles}")
        if batch_start != 0 or batch_count != self.samples:
            raise ValueError("observer requires the complete execution batch")
        if self.seen_cycles[cycle]:
            raise RuntimeError(f"duplicate observation at global cycle {cycle}")
        if not hasattr(state, "frame") or not hasattr(state, "ranks"):
            raise TypeError("observer requires a native occupied-frame state")
        frame = state.frame.detach()
        ranks = state.ranks.detach().cpu().numpy().astype(np.int64, copy=False)
        if tuple(frame.shape[:2]) != (self.samples, 2 * self.nx * self.ny):
            raise ValueError(f"unexpected occupied-frame shape {tuple(frame.shape)}")
        if ranks.shape != (self.samples,) or np.any(ranks < 0) or np.any(
            ranks > int(frame.shape[2])
        ):
            raise ValueError("invalid occupied-frame ranks")
        self.global_charge[:, cycle] = ranks

        compute_all_widths = cycle == self.physical_cycles or (
            self.detailed and cycle >= 10
        )
        compute_half_contour = self.detailed
        if compute_all_widths:
            contour_ays: Iterable[int]
            if self.detailed and cycle == self.physical_cycles:
                contour_ays = self.ay_values
            elif self.detailed:
                contour_ays = [self.ny // 2]
            else:
                contour_ays = []
            observed = frame_y0_averaged_observables(
                frame,
                nx=self.nx,
                ny=self.ny,
                ay_values=self.ay_values,
                contour_ays=contour_ays,
                sample_chunk=self.sample_chunk,
                y0_chunk=self.y0_chunk,
                occupation_tolerance=self.occupation_tolerance,
            )
            if self.detailed:
                curve_index = cycle - 10
                self.entropy_curves[:, curve_index] = observed["entropy"]
                self.charge_mean_curves[:, curve_index] = observed["charge_mean"]
                self.charge_variance_curves[:, curve_index] = observed[
                    "charge_variance"
                ]
                self.curve_seen[curve_index] = True
            if cycle == self.physical_cycles:
                self.endpoint_entropy[:] = observed["entropy"]
                self.endpoint_charge_mean[:] = observed["charge_mean"]
                self.endpoint_charge_variance[:] = observed["charge_variance"]
                self.endpoint_seen = True
                if self.detailed:
                    half = self.ny // 2
                    for width_index, ay in enumerate(self.ay_values):
                        ay = int(ay)
                        if ay:
                            self.final_entropy_contour[:, width_index, :, :ay] = observed[
                                "entropy_contours"
                            ][ay]
                            self.final_charge_variance_contour[
                                :, width_index, :, :ay
                            ] = observed["charge_variance_contours"][ay]
                    self.final_contour_seen = True
            self.min_occupation = min(self.min_occupation, observed["occupation_min"])
            self.max_occupation = max(self.max_occupation, observed["occupation_max"])

        if compute_half_contour:
            half = self.ny // 2
            if compute_all_widths:
                half_entropy = observed["entropy_contours"][half]
                half_variance = observed["charge_variance_contours"][half]
            else:
                half_observed = frame_y0_averaged_observables(
                    frame,
                    nx=self.nx,
                    ny=self.ny,
                    ay_values=[half],
                    contour_ays=[half],
                    sample_chunk=self.sample_chunk,
                    y0_chunk=self.y0_chunk,
                    occupation_tolerance=self.occupation_tolerance,
                )
                half_entropy = half_observed["entropy_contours"][half]
                half_variance = half_observed["charge_variance_contours"][half]
                self.min_occupation = min(
                    self.min_occupation, half_observed["occupation_min"]
                )
                self.max_occupation = max(
                    self.max_occupation, half_observed["occupation_max"]
                )
            self.half_strip_entropy_contour[:, cycle] = half_entropy
            self.half_strip_charge_variance_contour[:, cycle] = half_variance

        self.seen_cycles[cycle] = True

    def _identity_payload(self) -> dict[str, np.ndarray]:
        return {
            "observer_schema": np.asarray(OBSERVER_SCHEMA),
            "nx": np.asarray(self.nx, dtype=np.int64),
            "ny": np.asarray(self.ny, dtype=np.int64),
            "physical_cycles": np.asarray(self.physical_cycles, dtype=np.int64),
            "detailed": np.asarray(self.detailed, dtype=np.bool_),
            "sample_ids": self.sample_ids.copy(),
            "cycles": self.cycles.copy(),
            "ay_values": self.ay_values.copy(),
        }

    def checkpoint_payload(self) -> dict[str, np.ndarray]:
        """Return an NPZ-safe mapping containing all partial accumulators."""

        payload = {
            **self._identity_payload(),
            "seen_cycles": self.seen_cycles.copy(),
            "global_charge": self.global_charge.copy(),
            "endpoint_seen": np.asarray(self.endpoint_seen, dtype=np.bool_),
            "endpoint_entropy": self.endpoint_entropy.copy(),
            "endpoint_charge_mean": self.endpoint_charge_mean.copy(),
            "endpoint_charge_variance": self.endpoint_charge_variance.copy(),
            "min_occupation": np.asarray(self.min_occupation, dtype=np.float64),
            "max_occupation": np.asarray(self.max_occupation, dtype=np.float64),
        }
        if self.detailed:
            payload.update(
                {
                    "curve_cycles": self.curve_cycles.copy(),
                    "curve_seen": self.curve_seen.copy(),
                    "entropy_curves": self.entropy_curves.copy(),
                    "charge_mean_curves": self.charge_mean_curves.copy(),
                    "charge_variance_curves": self.charge_variance_curves.copy(),
                    "half_strip_entropy_contour": self.half_strip_entropy_contour.copy(),
                    "half_strip_charge_variance_contour": self.half_strip_charge_variance_contour.copy(),
                    "final_contour_seen": np.asarray(
                        self.final_contour_seen, dtype=np.bool_
                    ),
                    "final_entropy_contour": self.final_entropy_contour.copy(),
                    "final_charge_variance_contour": self.final_charge_variance_contour.copy(),
                }
            )
        return payload

    state_dict = checkpoint_payload

    def restore_checkpoint(self, payload: Mapping[str, Any]) -> None:
        """Restore arrays from an ``np.load(..., allow_pickle=False)`` mapping."""

        expected = self._identity_payload()
        for key, value in expected.items():
            if key not in payload or not np.array_equal(np.asarray(payload[key]), value):
                raise ValueError(f"observer checkpoint identity mismatch for {key}")
        required = self.checkpoint_payload()
        missing = sorted(set(required).difference(payload))
        if missing:
            raise ValueError(f"observer checkpoint is missing keys: {missing}")
        for key, expected_value in required.items():
            actual = np.asarray(payload[key])
            if actual.shape != expected_value.shape or actual.dtype != expected_value.dtype:
                raise ValueError(f"observer checkpoint array mismatch for {key}")

        self.seen_cycles[:] = payload["seen_cycles"]
        self.global_charge[:] = payload["global_charge"]
        self.endpoint_seen = bool(_scalar_from_mapping(payload, "endpoint_seen"))
        self.endpoint_entropy[:] = payload["endpoint_entropy"]
        self.endpoint_charge_mean[:] = payload["endpoint_charge_mean"]
        self.endpoint_charge_variance[:] = payload["endpoint_charge_variance"]
        self.min_occupation = float(_scalar_from_mapping(payload, "min_occupation"))
        self.max_occupation = float(_scalar_from_mapping(payload, "max_occupation"))
        if self.detailed:
            self.curve_seen[:] = payload["curve_seen"]
            self.entropy_curves[:] = payload["entropy_curves"]
            self.charge_mean_curves[:] = payload["charge_mean_curves"]
            self.charge_variance_curves[:] = payload["charge_variance_curves"]
            self.half_strip_entropy_contour[:] = payload[
                "half_strip_entropy_contour"
            ]
            self.half_strip_charge_variance_contour[:] = payload[
                "half_strip_charge_variance_contour"
            ]
            self.final_contour_seen = bool(
                _scalar_from_mapping(payload, "final_contour_seen")
            )
            self.final_entropy_contour[:] = payload["final_entropy_contour"]
            self.final_charge_variance_contour[:] = payload[
                "final_charge_variance_contour"
            ]

    load_state_dict = restore_checkpoint

    def validate(self, *, final: bool = True) -> dict[str, Any]:
        """Validate partial checkpoint state or a complete production result."""

        seen_indices = np.flatnonzero(self.seen_cycles)
        if seen_indices.size and not np.array_equal(
            seen_indices, np.arange(int(seen_indices[-1]) + 1)
        ):
            raise RuntimeError("observed cycles are not one contiguous prefix from zero")
        maximum_seen = int(seen_indices[-1]) if seen_indices.size else -1
        final_cycle_seen = maximum_seen == self.physical_cycles
        if bool(self.endpoint_seen) != final_cycle_seen:
            raise RuntimeError(
                "endpoint-seen flag must be true if and only if the final cycle is seen"
            )
        if self.detailed:
            expected_curve_seen = self.curve_cycles <= maximum_seen
            if not np.array_equal(self.curve_seen, expected_curve_seen):
                raise RuntimeError(
                    "detailed curve-seen flags do not match cycles 10..maximum_seen"
                )
            if bool(self.final_contour_seen) != final_cycle_seen:
                raise RuntimeError(
                    "final-contour flag must be true if and only if the final cycle is seen"
                )
        if np.any(self.global_charge[:, self.seen_cycles] < 0):
            raise FloatingPointError("observed global charge is negative")
        if np.any(self.global_charge[:, self.seen_cycles] > 2 * self.nx * self.ny):
            raise FloatingPointError("observed global charge exceeds the mode count")
        if final:
            if not bool(np.all(self.seen_cycles)):
                raise RuntimeError("one or more global-charge cycles are missing")
            if not self.endpoint_seen:
                raise RuntimeError("endpoint strip curves are missing")
            arrays = [
                self.endpoint_entropy,
                self.endpoint_charge_mean,
                self.endpoint_charge_variance,
            ]
            if self.detailed:
                if not bool(np.all(self.curve_seen)) or not self.final_contour_seen:
                    raise RuntimeError("one or more detailed Ny=40 products are missing")
                arrays.extend(
                    [
                        self.entropy_curves,
                        self.charge_mean_curves,
                        self.charge_variance_curves,
                        self.half_strip_entropy_contour,
                        self.half_strip_charge_variance_contour,
                        self.final_entropy_contour,
                        self.final_charge_variance_contour,
                    ]
                )
            if any(not np.isfinite(value).all() for value in arrays):
                raise FloatingPointError("observer product contains nonfinite values")
            if any(np.min(value) < -1.0e-9 for value in arrays):
                raise FloatingPointError("observer product is negative below tolerance")
            if not np.all(self.endpoint_entropy[:, 0] == 0.0):
                raise FloatingPointError("Ay=0 entropy must be exactly zero")
            if not np.all(self.endpoint_charge_variance[:, 0] == 0.0):
                raise FloatingPointError("Ay=0 charge variance must be exactly zero")
        return {
            "schema": OBSERVER_SCHEMA,
            "samples": self.samples,
            "cycles_seen": int(self.seen_cycles.sum()),
            "endpoint_seen": bool(self.endpoint_seen),
            "detailed": bool(self.detailed),
            "minimum_occupation": float(self.min_occupation),
            "maximum_occupation": float(self.max_occupation),
            "covariance_materializations": 0,
        }

    def result_payload(self, sample_slice: slice | tuple[int, int] | None = None) -> dict[str, np.ndarray]:
        """Return a compact result mapping, optionally for one five-sample shard."""

        self.validate(final=True)
        if sample_slice is None:
            selected = slice(None)
        elif isinstance(sample_slice, tuple):
            if len(sample_slice) != 2:
                raise ValueError("sample_slice tuple must be (start, stop)")
            selected = slice(int(sample_slice[0]), int(sample_slice[1]))
        elif isinstance(sample_slice, slice):
            selected = sample_slice
        else:
            raise TypeError("sample_slice must be a slice, (start,stop), or None")
        indices = np.arange(self.samples)[selected]
        if indices.ndim != 1 or indices.size == 0:
            raise ValueError("sample_slice selects no trajectories")
        if not np.array_equal(indices, np.arange(indices[0], indices[-1] + 1)):
            raise ValueError("result shards must select one contiguous sample range")

        payload = {
            "observer_schema": np.asarray(OBSERVER_SCHEMA),
            "nx": np.asarray(self.nx, dtype=np.int64),
            "ny": np.asarray(self.ny, dtype=np.int64),
            "physical_cycles": np.asarray(self.physical_cycles, dtype=np.int64),
            "detailed": np.asarray(self.detailed, dtype=np.bool_),
            "sample_ids": self.sample_ids[indices].copy(),
            "cycles": self.cycles.copy(),
            "global_charge": self.global_charge[indices].copy(),
            "half_filling_offset": (
                self.global_charge[indices] - self.nx * self.ny
            ).copy(),
            "ay_values": self.ay_values.copy(),
            "endpoint_cycle": np.asarray(self.physical_cycles, dtype=np.int64),
            "endpoint_entropy": self.endpoint_entropy[indices].copy(),
            "endpoint_charge_mean": self.endpoint_charge_mean[indices].copy(),
            "endpoint_charge_variance": self.endpoint_charge_variance[indices].copy(),
            "origin_average_count": np.asarray(self.ny, dtype=np.int64),
        }
        if self.detailed:
            half = self.ny // 2
            valid = np.zeros((len(self.ay_values), half), dtype=np.bool_)
            for width_index, ay in enumerate(self.ay_values):
                valid[width_index, : int(ay)] = True
            payload.update(
                {
                    "curve_cycles": self.curve_cycles.copy(),
                    "entropy_curves": self.entropy_curves[indices].copy(),
                    "charge_mean_curves": self.charge_mean_curves[indices].copy(),
                    "charge_variance_curves": self.charge_variance_curves[
                        indices
                    ].copy(),
                    "half_strip_ay": np.asarray(half, dtype=np.int64),
                    "half_contour_cycles": self.cycles.copy(),
                    "half_strip_entropy_contour": self.half_strip_entropy_contour[
                        indices
                    ].copy(),
                    "half_strip_charge_variance_contour": self.half_strip_charge_variance_contour[
                        indices
                    ].copy(),
                    "final_contour_valid": valid,
                    "final_entropy_contour": self.final_entropy_contour[
                        indices
                    ].copy(),
                    "final_charge_variance_contour": self.final_charge_variance_contour[
                        indices
                    ].copy(),
                }
            )
        return payload


__all__ = [
    "HardWallEntropyChargeObserver",
    "OBSERVER_SCHEMA",
    "frame_window_observables",
    "frame_y0_averaged_observables",
    "periodic_window_indices",
]
