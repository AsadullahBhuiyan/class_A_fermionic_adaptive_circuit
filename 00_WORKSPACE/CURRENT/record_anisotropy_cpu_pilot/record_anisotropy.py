from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


CHANNELS = ("Ap", "Am", "Bp", "Bm")
EXPECTED_OCCUPIED = {"Ap": False, "Am": True, "Bp": False, "Bm": True}
OBSERVABLES = ("surprisal", "mismatch_fraction")


class WallRecordObserver:
    """Retain every Born measurement event on both domain walls.

    The canonical CPU engine calls this observer once per visited lattice site.
    Arrays use the order ``sample, cycle, wall, y, channel``.  Cycles from the
    engine are one-indexed and are converted to zero-indexed array positions.
    """

    def __init__(
        self,
        *,
        nx: int,
        ny: int,
        cycles: int,
        samples: int,
        wall_x: Sequence[int],
    ) -> None:
        self.nx = int(nx)
        self.ny = int(ny)
        self.cycles = int(cycles)
        self.samples = int(samples)
        self.wall_x = tuple(int(x) for x in wall_x)
        if len(self.wall_x) != 2 or len(set(self.wall_x)) != 2:
            raise ValueError("wall_x must contain two distinct wall positions")
        if any(x < 0 or x >= self.nx for x in self.wall_x):
            raise ValueError("wall position lies outside the lattice")
        shape = (self.samples, self.cycles, 2, self.ny, len(CHANNELS))
        self.log_realized_probability = np.full(shape, np.nan, dtype=np.float64)
        self.occupation_probability = np.full(shape, np.nan, dtype=np.float64)
        self.outcome_occupied = np.zeros(shape, dtype=np.bool_)
        self.mismatch = np.zeros(shape, dtype=np.bool_)
        self._seen = np.zeros(shape[:-1], dtype=np.bool_)
        self._wall_lookup = {x: index for index, x in enumerate(self.wall_x)}

    def __call__(
        self,
        *,
        cycle: int,
        site_id: int,
        sample_index: int,
        branch_events: Sequence[Mapping[str, Any]],
        **_: Any,
    ) -> None:
        x = int(site_id) % self.nx
        wall_index = self._wall_lookup.get(x)
        if wall_index is None:
            return
        y = int(site_id) // self.nx
        cycle_index = int(cycle) - 1
        sample_index = int(sample_index)
        if not (0 <= y < self.ny):
            raise IndexError(f"decoded y={y} outside 0..{self.ny - 1}")
        if not (0 <= cycle_index < self.cycles):
            raise IndexError(f"cycle={cycle} outside 1..{self.cycles}")
        if not (0 <= sample_index < self.samples):
            raise IndexError(f"sample={sample_index} outside 0..{self.samples - 1}")
        destination = (sample_index, cycle_index, wall_index, y)
        if self._seen[destination]:
            raise RuntimeError(f"duplicate wall-site record at {destination}")

        measurements = {
            str(event["channel"]): event
            for event in branch_events
            if str(event.get("kind")) == "measurement"
        }
        if set(measurements) != set(CHANNELS):
            raise RuntimeError(f"expected measurement channels {CHANNELS}, got {tuple(measurements)}")
        for channel_index, channel in enumerate(CHANNELS):
            event = measurements[channel]
            outcome = bool(event["outcome_occupied"])
            self.log_realized_probability[destination + (channel_index,)] = float(
                event["log_weight"]
            )
            self.occupation_probability[destination + (channel_index,)] = float(
                event["probability"]
            )
            self.outcome_occupied[destination + (channel_index,)] = outcome
            self.mismatch[destination + (channel_index,)] = (
                outcome != EXPECTED_OCCUPIED[channel]
            )
        self._seen[destination] = True

    def assert_complete(self) -> None:
        if not np.all(self._seen):
            missing = np.argwhere(~self._seen)
            raise RuntimeError(
                f"wall record is incomplete: {missing.shape[0]} missing sites; first={missing[0].tolist()}"
            )
        if not np.all(np.isfinite(self.log_realized_probability)):
            raise RuntimeError("wall record contains a non-finite realized log probability")
        if not np.all(np.isfinite(self.occupation_probability)):
            raise RuntimeError("wall record contains a non-finite occupation probability")

    def derived_fields(self) -> dict[str, np.ndarray]:
        self.assert_complete()
        return {
            "surprisal": -np.sum(self.log_realized_probability, axis=-1),
            "mismatch_fraction": np.mean(self.mismatch, axis=-1, dtype=np.float64),
        }

    def payload(self) -> dict[str, np.ndarray]:
        self.assert_complete()
        fields = self.derived_fields()
        return {
            "schema": np.asarray("wall_born_record_anisotropy_v1"),
            "wall_x": np.asarray(self.wall_x, dtype=np.int64),
            "channels": np.asarray(CHANNELS),
            "log_realized_probability": self.log_realized_probability,
            "occupation_probability": self.occupation_probability,
            "outcome_occupied": self.outcome_occupied,
            "mismatch": self.mismatch,
            **fields,
        }


@dataclass(frozen=True)
class CorrelationEstimate:
    spatial_by_trajectory: np.ndarray
    temporal_by_trajectory: np.ndarray
    spatial_mean: np.ndarray
    temporal_mean: np.ndarray
    mean_by_trajectory_wall: np.ndarray
    spatial_product_by_trajectory_wall: np.ndarray
    temporal_product_by_trajectory_wall: np.ndarray


def connected_correlations(
    field: np.ndarray,
    *,
    burn_in: int,
    max_temporal_lag: int | None = None,
) -> CorrelationEstimate:
    """Connected periodic-space and stationary-time correlations.

    The connected subtraction uses the ensemble spacetime mean separately on
    each wall.  This avoids the negative long-distance bias caused by forcing
    every finite trajectory to have exactly zero mean.  The raw first and second
    moments are retained so a bootstrap resample can repeat that subtraction.
    The two walls are averaged inside a trajectory and never counted as
    independent Monte Carlo samples.
    """

    field = np.asarray(field, dtype=np.float64)
    if field.ndim != 4:
        raise ValueError("field must have shape (samples, cycles, walls, y)")
    samples, cycles, walls, circumference = field.shape
    burn_in = int(burn_in)
    if samples < 1 or walls != 2 or circumference < 2:
        raise ValueError("field requires samples>=1, exactly two walls, and y>=2")
    if not (0 <= burn_in < cycles - 1):
        raise ValueError("burn_in must leave at least two recorded cycles")
    stationary = field[:, burn_in:, :, :]
    if not np.all(np.isfinite(stationary)):
        raise ValueError("stationary field contains non-finite values")
    first_moment = np.mean(stationary, axis=(1, 3))
    ensemble_wall_mean = np.mean(first_moment, axis=0)

    max_r = circumference // 2
    spatial_product = np.empty((samples, walls, max_r + 1), dtype=np.float64)
    for separation in range(max_r + 1):
        spatial_product[:, :, separation] = np.mean(
            stationary * np.roll(stationary, shift=-separation, axis=3), axis=(1, 3)
        )

    stationary_cycles = stationary.shape[1]
    if max_temporal_lag is None:
        max_temporal_lag = stationary_cycles // 2
    max_temporal_lag = min(int(max_temporal_lag), stationary_cycles - 1)
    if max_temporal_lag < 1:
        raise ValueError("max_temporal_lag must be positive")
    temporal_product = np.empty((samples, walls, max_temporal_lag + 1), dtype=np.float64)
    temporal_product[:, :, 0] = np.mean(stationary * stationary, axis=(1, 3))
    for lag in range(1, max_temporal_lag + 1):
        temporal_product[:, :, lag] = np.mean(
            stationary[:, :-lag] * stationary[:, lag:], axis=(1, 3)
        )
    spatial = np.mean(spatial_product - ensemble_wall_mean[None, :, None] ** 2, axis=1)
    temporal = np.mean(temporal_product - ensemble_wall_mean[None, :, None] ** 2, axis=1)
    return CorrelationEstimate(
        spatial_by_trajectory=spatial,
        temporal_by_trajectory=temporal,
        spatial_mean=np.mean(spatial, axis=0),
        temporal_mean=np.mean(temporal, axis=0),
        mean_by_trajectory_wall=first_moment,
        spatial_product_by_trajectory_wall=spatial_product,
        temporal_product_by_trajectory_wall=temporal_product,
    )


def match_time(
    temporal_correlation: np.ndarray,
    spatial_target: float,
    *,
    minimum_noncontact_lag: int = 1,
) -> float:
    """Return the first resolved non-contact match ``C(0,t)=C(L/2,0)``.

    Lag zero contains the local variance/contact term and is not a continuum
    time-separated correlator.  A target already below the first admissible lag
    is therefore unresolved, rather than being interpolated to a fictitious
    sub-cycle time.
    """

    temporal = np.asarray(temporal_correlation, dtype=np.float64).reshape(-1)
    target = float(spatial_target)
    minimum_lag = int(minimum_noncontact_lag)
    if (
        temporal.size < minimum_lag + 2
        or minimum_lag < 1
        or not np.all(np.isfinite(temporal))
        or not np.isfinite(target)
    ):
        return float("nan")
    if target <= 0.0 or temporal[minimum_lag] <= target:
        return float("nan")
    difference = temporal - target
    exact = np.flatnonzero(difference[minimum_lag:] == 0.0)
    if exact.size:
        return float(minimum_lag + exact[0])
    for left in range(minimum_lag, temporal.size - 1):
        if difference[left] > 0.0 and difference[left + 1] < 0.0:
            fraction = difference[left] / (difference[left] - difference[left + 1])
            return float(left + fraction)
    return float("nan")


def alpha_from_match(circumference: int, matched_time: float) -> float:
    matched_time = float(matched_time)
    if not np.isfinite(matched_time) or matched_time <= 0.0:
        return float("nan")
    return float(np.arcsinh(1.0) * int(circumference) / (np.pi * matched_time))


def bootstrap_match(
    estimate: CorrelationEstimate,
    *,
    circumference: int,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    """Bootstrap complete trajectories, keeping the two walls paired."""

    spatial_product = estimate.spatial_product_by_trajectory_wall
    temporal_product = estimate.temporal_product_by_trajectory_wall
    first_moment = estimate.mean_by_trajectory_wall
    samples = spatial_product.shape[0]
    target_index = int(circumference) // 2
    rng = np.random.default_rng(int(seed))
    times = np.full(int(draws), np.nan, dtype=np.float64)
    alphas = np.full(int(draws), np.nan, dtype=np.float64)
    for draw in range(int(draws)):
        indices = rng.integers(0, samples, size=samples)
        wall_mean = np.mean(first_moment[indices], axis=0)
        spatial = np.mean(
            np.mean(spatial_product[indices], axis=0) - wall_mean[:, None] ** 2,
            axis=0,
        )
        temporal = np.mean(
            np.mean(temporal_product[indices], axis=0) - wall_mean[:, None] ** 2,
            axis=0,
        )
        target = float(spatial[target_index])
        times[draw] = match_time(temporal, target)
        alphas[draw] = alpha_from_match(circumference, times[draw])
    finite = np.isfinite(alphas)
    result: dict[str, Any] = {
        "bootstrap_draws": int(draws),
        "bootstrap_resolved": int(np.sum(finite)),
        "bootstrap_resolved_fraction": float(np.mean(finite)),
        "t_star_bootstrap": times,
        "alpha_bootstrap": alphas,
    }
    if np.any(finite):
        result.update(
            alpha_ci_low=float(np.nanpercentile(alphas, 2.5)),
            alpha_ci_high=float(np.nanpercentile(alphas, 97.5)),
            t_star_ci_low=float(np.nanpercentile(times, 2.5)),
            t_star_ci_high=float(np.nanpercentile(times, 97.5)),
        )
    else:
        result.update(
            alpha_ci_low=float("nan"),
            alpha_ci_high=float("nan"),
            t_star_ci_low=float("nan"),
            t_star_ci_high=float("nan"),
        )
    return result


def stationarity_diagnostic(field: np.ndarray, *, burn_in: int) -> dict[str, float]:
    field = np.asarray(field, dtype=np.float64)[:, int(burn_in) :]
    midpoint = field.shape[1] // 2
    first = field[:, :midpoint]
    second = field[:, midpoint:]
    mean_first = float(np.mean(first))
    mean_second = float(np.mean(second))
    variance_first = float(np.var(first))
    variance_second = float(np.var(second))
    pooled_scale = float(np.sqrt(0.5 * (variance_first + variance_second)))
    return {
        "mean_first_half": mean_first,
        "mean_second_half": mean_second,
        "mean_shift_in_pooled_sigma": (
            float((mean_second - mean_first) / pooled_scale) if pooled_scale > 0.0 else float("nan")
        ),
        "variance_first_half": variance_first,
        "variance_second_half": variance_second,
        "variance_ratio_second_over_first": (
            float(variance_second / variance_first) if variance_first > 0.0 else float("nan")
        ),
    }
