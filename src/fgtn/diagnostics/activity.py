from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from joblib import Parallel, delayed, parallel_backend
import numpy as np
from threadpoolctl import threadpool_limits

from .common import CHANNELS, CHANNEL_TARGETS, RegionMasks, active_site_ids


@dataclass
class ActivityAnalysisResult:
    region_names: tuple[str, ...]
    channel_names: tuple[str, ...]
    activity_names: tuple[str, ...]
    burn_in: int
    s_grid: np.ndarray
    window_lengths: np.ndarray
    counts_by_cycle: np.ndarray
    attempts_by_cycle: np.ndarray
    rates_by_cycle: np.ndarray
    spatial_profile_x: np.ndarray
    success_probability: np.ndarray
    cumulants_per_cycle: np.ndarray
    theta: np.ndarray
    theta_ci_low: np.ndarray
    theta_ci_high: np.ndarray
    effective_sample_fraction: np.ndarray
    reliable: np.ndarray
    waiting_survival: np.ndarray
    waiting_interval_count: np.ndarray
    stationarity_slope: np.ndarray

    def payload(self) -> dict[str, Any]:
        return {
            "region_names": np.asarray(self.region_names),
            "channel_names": np.asarray(self.channel_names),
            "activity_names": np.asarray(self.activity_names),
            "burn_in": np.asarray(self.burn_in, dtype=np.int64),
            "s_grid": self.s_grid,
            "window_lengths": self.window_lengths,
            "counts_by_cycle": self.counts_by_cycle,
            "attempts_by_cycle": self.attempts_by_cycle,
            "rates_by_cycle": self.rates_by_cycle,
            "spatial_profile_x": self.spatial_profile_x,
            "success_probability": self.success_probability,
            "cumulants_per_cycle": self.cumulants_per_cycle,
            "theta": self.theta,
            "theta_ci_low": self.theta_ci_low,
            "theta_ci_high": self.theta_ci_high,
            "effective_sample_fraction": self.effective_sample_fraction,
            "reliable": self.reliable,
            "waiting_survival": self.waiting_survival,
            "waiting_interval_count": self.waiting_interval_count,
            "stationarity_slope": self.stationarity_slope,
        }


class TrajectoryActivityRecorder:
    """Parse canonical CPU branch events into defect and transfer records."""

    def __init__(
        self,
        *,
        nx: int,
        ny: int,
        cycles: int,
        samples: int,
        site_ids: Iterable[int],
    ) -> None:
        self.nx = int(nx)
        self.ny = int(ny)
        self.cycles = int(cycles)
        self.samples = int(samples)
        self.site_ids = np.asarray(list(site_ids), dtype=np.int64)
        if self.cycles <= 0 or self.samples <= 0 or self.site_ids.size == 0:
            raise ValueError("cycles, samples, and the active site set must be nonempty.")
        if np.unique(self.site_ids).size != self.site_ids.size:
            raise ValueError("site_ids must be unique.")
        self._site_to_offset = {int(site): idx for idx, site in enumerate(self.site_ids)}
        shape = (self.samples, self.cycles, self.site_ids.size, len(CHANNELS))
        self.defect = np.zeros(shape, dtype=np.uint8)
        self.transfer = np.zeros(shape, dtype=np.int8)
        self.success_probability = np.full(shape, np.nan, dtype=np.float64)
        self.valid = np.zeros(shape, dtype=bool)
        self.forced_site = np.zeros(shape[:-1], dtype=bool)
        self.visit_order = np.full(shape[:-1], -1, dtype=np.int64)
        self._visit_count = np.zeros((self.samples, self.cycles), dtype=np.int64)

    @classmethod
    def from_model(cls, model: Any, *, cycles: int, samples: int, meas_slab_only: bool = True):
        return cls(
            nx=int(model.Nx),
            ny=int(model.Ny),
            cycles=int(cycles),
            samples=int(samples),
            site_ids=active_site_ids(model, meas_slab_only=meas_slab_only),
        )

    def __call__(
        self,
        *,
        cycle: int,
        site_id: int,
        sample_index: int,
        branch_events: Iterable[dict[str, Any]],
        forced_postselect: bool,
        **_: Any,
    ) -> None:
        sample = int(sample_index)
        cycle_index = int(cycle) - 1
        if not (0 <= sample < self.samples) or not (0 <= cycle_index < self.cycles):
            raise IndexError("Observed activity event lies outside the configured sample/cycle range.")
        try:
            site_offset = self._site_to_offset[int(site_id)]
        except KeyError as exc:
            raise KeyError(f"Observed inactive site_id={site_id}.") from exc
        if self.visit_order[sample, cycle_index, site_offset] >= 0:
            raise RuntimeError("The same site was observed more than once in one cycle.")
        self.visit_order[sample, cycle_index, site_offset] = self._visit_count[sample, cycle_index]
        self._visit_count[sample, cycle_index] += 1

        if bool(forced_postselect):
            self.forced_site[sample, cycle_index, site_offset] = True
            return

        measurements: dict[str, dict[str, Any]] = {}
        corrections: dict[str, dict[str, Any]] = {}
        for event in branch_events:
            channel = str(event.get("channel"))
            kind = str(event.get("kind"))
            if channel not in CHANNELS:
                raise ValueError(f"Unexpected activity channel {channel!r}.")
            target = measurements if kind == "measurement" else corrections if kind == "correction" else None
            if target is None:
                raise ValueError(f"Unexpected branch event kind {kind!r}.")
            if channel in target:
                raise RuntimeError(f"Duplicate {kind} event for channel {channel}.")
            target[channel] = dict(event)
        if set(measurements) != set(CHANNELS):
            raise RuntimeError(f"Expected one measurement per OW channel; observed {sorted(measurements)}.")

        for channel_index, channel in enumerate(CHANNELS):
            measurement = measurements[channel]
            expected = int(CHANNEL_TARGETS[channel])
            outcome = int(bool(measurement["outcome_occupied"]))
            p_occ = float(measurement["probability"])
            if not np.isfinite(p_occ) or p_occ < -1e-12 or p_occ > 1.0 + 1e-12:
                raise FloatingPointError(f"Invalid Born probability {p_occ} for channel {channel}.")
            p_occ = float(np.clip(p_occ, 0.0, 1.0))
            defect = int(outcome != expected)
            target_after = outcome
            if channel in corrections:
                correction = corrections[channel]
                correction_expected = int(bool(correction["expected_occupied"]))
                if correction_expected != expected:
                    raise RuntimeError(f"Correction target mismatch for channel {channel}.")
                target_after = int(bool(correction["target_occupied"]))
            elif defect:
                raise RuntimeError(f"Missing correction event for wrong outcome in channel {channel}.")
            transfer = int(target_after - outcome)
            if transfer not in (-1, 0, 1):
                raise RuntimeError("Transfer activity left its signed binary range.")
            self.defect[sample, cycle_index, site_offset, channel_index] = defect
            self.transfer[sample, cycle_index, site_offset, channel_index] = transfer
            self.success_probability[sample, cycle_index, site_offset, channel_index] = (
                p_occ if expected else 1.0 - p_occ
            )
            self.valid[sample, cycle_index, site_offset, channel_index] = True

    def assert_complete(self, *, allow_forced: bool = False) -> None:
        visited = self.visit_order >= 0
        if not np.all(visited):
            raise RuntimeError(f"Activity recording missed {int(np.count_nonzero(~visited))} site visits.")
        expected = np.broadcast_to(~self.forced_site[..., None], self.valid.shape)
        if allow_forced:
            if np.any(self.valid & ~expected):
                raise RuntimeError("Forced postselection sites unexpectedly contain sampled activity.")
        elif not np.all(self.valid):
            raise RuntimeError("Activity records are incomplete or contain forced postselection sites.")

    def payload(self) -> dict[str, Any]:
        return {
            "nx": np.asarray(self.nx, dtype=np.int64),
            "ny": np.asarray(self.ny, dtype=np.int64),
            "cycles": np.asarray(self.cycles, dtype=np.int64),
            "samples": np.asarray(self.samples, dtype=np.int64),
            "channel_names": np.asarray(CHANNELS),
            "site_ids": self.site_ids,
            "site_x": self.site_ids % self.nx,
            "site_y": self.site_ids // self.nx,
            "defect_X": self.defect,
            "transfer_Y": self.transfer,
            "success_probability": self.success_probability,
            "valid": self.valid,
            "forced_site": self.forced_site,
            "visit_order": self.visit_order,
        }


def empirical_scgf(
    counts: np.ndarray,
    *,
    observation_time: int,
    s_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    counts = np.asarray(counts, dtype=np.float64).reshape(-1)
    s_grid = np.asarray(s_grid, dtype=np.float64).reshape(-1)
    observation_time = int(observation_time)
    if counts.size == 0 or observation_time <= 0:
        raise ValueError("SCGF estimation needs samples and a positive observation time.")
    log_weights = -s_grid[:, None] * counts[None, :]
    maxima = np.max(log_weights, axis=1)
    shifted = np.exp(log_weights - maxima[:, None])
    sums = np.sum(shifted, axis=1)
    theta = (maxima + np.log(sums / counts.size)) / float(observation_time)
    ess = (sums * sums) / np.sum(shifted * shifted, axis=1)
    return theta, ess / float(counts.size)


def bootstrap_scgf(
    counts: np.ndarray,
    *,
    observation_time: int,
    s_grid: np.ndarray,
    bootstrap_samples: int = 1000,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    counts = np.asarray(counts, dtype=np.float64).reshape(-1)
    bootstrap_samples = int(bootstrap_samples)
    if bootstrap_samples <= 0:
        nan = np.full(np.asarray(s_grid).shape, np.nan, dtype=np.float64)
        return nan, nan.copy()
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, counts.size, size=(bootstrap_samples, counts.size))
    sampled = counts[indices]
    estimates = np.empty((bootstrap_samples, np.asarray(s_grid).size), dtype=np.float64)
    for s_index, field in enumerate(np.asarray(s_grid, dtype=np.float64)):
        values = -float(field) * sampled
        maxima = np.max(values, axis=1)
        estimates[:, s_index] = (
            maxima + np.log(np.mean(np.exp(values - maxima[:, None]), axis=1))
        ) / float(observation_time)
    return np.quantile(estimates, 0.025, axis=0), np.quantile(estimates, 0.975, axis=0)


def _fcs_cell(task: tuple[Any, ...]) -> tuple[Any, ...]:
    (
        kind_index,
        window_index,
        region_index,
        counts,
        length,
        s_grid,
        bootstrap_samples,
        seed,
    ) = task
    theta, ess = empirical_scgf(counts, observation_time=length, s_grid=s_grid)
    low, high = bootstrap_scgf(
        counts,
        observation_time=length,
        s_grid=s_grid,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )
    centered = counts - np.mean(counts)
    cumulants = np.asarray(
        (
            np.mean(counts) / length,
            np.mean(centered ** 2) / length,
            np.mean(centered ** 3) / length,
        ),
        dtype=np.float64,
    )
    return kind_index, window_index, region_index, theta, ess, low, high, cumulants


def _waiting_cell(task: tuple[Any, ...]) -> tuple[Any, ...]:
    kind_index, region_index, channel_index, event, valid, sites, channels, max_wait = task
    values, count = _waiting_survival(
        event,
        valid,
        site_selection=sites,
        channel_selection=channels,
        max_wait=max_wait,
    )
    return kind_index, region_index, channel_index, values, count


def _waiting_survival(
    event: np.ndarray,
    valid: np.ndarray,
    *,
    site_selection: np.ndarray,
    channel_selection: np.ndarray,
    max_wait: int,
) -> tuple[np.ndarray, int]:
    intervals: list[int] = []
    for sample in range(event.shape[0]):
        for site in np.flatnonzero(site_selection):
            for channel in np.flatnonzero(channel_selection):
                stream_valid = valid[sample, :, site, channel]
                times = np.flatnonzero(event[sample, :, site, channel] & stream_valid)
                if times.size >= 2:
                    intervals.extend(np.diff(times).astype(int).tolist())
    if not intervals:
        return np.full((max_wait,), np.nan, dtype=np.float64), 0
    values = np.asarray(intervals, dtype=np.int64)
    tau = np.arange(1, max_wait + 1, dtype=np.int64)
    return np.mean(values[:, None] >= tau[None, :], axis=0), int(values.size)


def analyze_activity(
    recorder: TrajectoryActivityRecorder,
    *,
    regions: RegionMasks,
    burn_in: int,
    s_grid: np.ndarray | None = None,
    window_fractions: tuple[float, ...] = (0.25, 0.5, 1.0),
    bootstrap_samples: int = 1000,
    bootstrap_seed: int = 0,
    reliability_threshold: float = 0.1,
    parallel_jobs: int = 1,
) -> ActivityAnalysisResult:
    burn_in = int(burn_in)
    if burn_in < 0 or burn_in >= recorder.cycles:
        raise ValueError("burn_in must leave at least one observed cycle.")
    s_grid = np.linspace(-0.2, 0.2, 41) if s_grid is None else np.asarray(s_grid, dtype=np.float64)
    observation_cycles = recorder.cycles - burn_in
    window_lengths = np.asarray(
        [max(1, min(observation_cycles, int(round(observation_cycles * fraction)))) for fraction in window_fractions],
        dtype=np.int64,
    )
    channel_names = CHANNELS + ("all",)
    activity_names = ("defect_X", "transfer_abs_Y")
    n_regions = len(regions.names)
    n_channels = len(channel_names)
    site_x = recorder.site_ids % recorder.nx
    site_y = recorder.site_ids // recorder.nx
    site_region = np.empty((n_regions, recorder.site_ids.size), dtype=bool)
    for region_index, mask in enumerate(regions.masks):
        site_region[region_index] = mask[site_x, site_y]

    event_kinds = (recorder.defect.astype(bool), np.abs(recorder.transfer).astype(bool))
    counts = np.zeros(
        (len(activity_names), recorder.samples, recorder.cycles, n_regions, n_channels), dtype=np.int64
    )
    attempts = np.zeros((recorder.samples, recorder.cycles, n_regions, n_channels), dtype=np.int64)
    for region_index in range(n_regions):
        selected = site_region[region_index]
        for channel_index in range(len(CHANNELS)):
            attempts[:, :, region_index, channel_index] = np.sum(
                recorder.valid[:, :, selected, channel_index], axis=2
            )
            for kind_index, event in enumerate(event_kinds):
                counts[kind_index, :, :, region_index, channel_index] = np.sum(
                    event[:, :, selected, channel_index] & recorder.valid[:, :, selected, channel_index], axis=2
                )
        attempts[:, :, region_index, -1] = np.sum(recorder.valid[:, :, selected, :], axis=(2, 3))
        for kind_index, event in enumerate(event_kinds):
            counts[kind_index, :, :, region_index, -1] = np.sum(
                event[:, :, selected, :] & recorder.valid[:, :, selected, :], axis=(2, 3)
            )
    rates = np.divide(
        counts,
        attempts[None, ...],
        out=np.full(counts.shape, np.nan, dtype=np.float64),
        where=attempts[None, ...] > 0,
    )

    spatial_profile = np.full((len(activity_names), recorder.nx, n_channels), np.nan, dtype=np.float64)
    obs = slice(burn_in, None)
    for x in range(recorder.nx):
        sites = site_x == x
        for channel_index in range(len(CHANNELS)):
            denom = np.sum(recorder.valid[:, obs, sites, channel_index])
            if denom:
                for kind_index, event in enumerate(event_kinds):
                    spatial_profile[kind_index, x, channel_index] = np.sum(
                        event[:, obs, sites, channel_index] & recorder.valid[:, obs, sites, channel_index]
                    ) / denom
        denom_all = np.sum(recorder.valid[:, obs, sites, :])
        if denom_all:
            for kind_index, event in enumerate(event_kinds):
                spatial_profile[kind_index, x, -1] = np.sum(
                    event[:, obs, sites, :] & recorder.valid[:, obs, sites, :]
                ) / denom_all

    success = np.full((n_regions, n_channels), np.nan, dtype=np.float64)
    for region_index in range(n_regions):
        selected = site_region[region_index]
        for channel_index in range(len(CHANNELS)):
            values = recorder.success_probability[:, obs, selected, channel_index]
            if np.any(np.isfinite(values)):
                success[region_index, channel_index] = float(np.nanmean(values))
        values = recorder.success_probability[:, obs, selected, :]
        if np.any(np.isfinite(values)):
            success[region_index, -1] = float(np.nanmean(values))

    shape_fcs = (len(activity_names), len(window_lengths), n_regions, s_grid.size)
    theta = np.full(shape_fcs, np.nan, dtype=np.float64)
    ci_low = np.full(shape_fcs, np.nan, dtype=np.float64)
    ci_high = np.full(shape_fcs, np.nan, dtype=np.float64)
    ess = np.full(shape_fcs, np.nan, dtype=np.float64)
    cumulants = np.full((len(activity_names), len(window_lengths), n_regions, 3), np.nan, dtype=np.float64)
    fcs_tasks: list[tuple[Any, ...]] = []
    for kind_index in range(len(activity_names)):
        for window_index, length in enumerate(window_lengths):
            cycle_slice = slice(burn_in, burn_in + int(length))
            for region_index in range(n_regions):
                K = np.sum(counts[kind_index, :, cycle_slice, region_index, -1], axis=1).astype(np.float64)
                fcs_tasks.append(
                    (
                        kind_index,
                        window_index,
                        region_index,
                        K,
                        int(length),
                        s_grid,
                        int(bootstrap_samples),
                        int(bootstrap_seed + 1000 * kind_index + 100 * window_index + region_index),
                    )
                )
    jobs = max(1, min(int(parallel_jobs), len(fcs_tasks)))
    if jobs > 1:
        with parallel_backend("loky", n_jobs=jobs, inner_max_num_threads=1):
            with threadpool_limits(limits=1):
                fcs_results = Parallel(n_jobs=jobs)(delayed(_fcs_cell)(task) for task in fcs_tasks)
    else:
        fcs_results = [_fcs_cell(task) for task in fcs_tasks]
    for kind_index, window_index, region_index, values, weights, low, high, cell_cumulants in fcs_results:
        theta[kind_index, window_index, region_index] = values
        ess[kind_index, window_index, region_index] = weights
        ci_low[kind_index, window_index, region_index] = low
        ci_high[kind_index, window_index, region_index] = high
        cumulants[kind_index, window_index, region_index] = cell_cumulants

    survival = np.full(
        (len(activity_names), n_regions, n_channels, observation_cycles), np.nan, dtype=np.float64
    )
    interval_count = np.zeros((len(activity_names), n_regions, n_channels), dtype=np.int64)
    valid_obs = recorder.valid[:, obs]
    waiting_tasks: list[tuple[Any, ...]] = []
    for kind_index, event in enumerate(event_kinds):
        event_obs = event[:, obs]
        for region_index in range(n_regions):
            selected = site_region[region_index]
            for channel_index in range(n_channels):
                channels = np.ones((len(CHANNELS),), dtype=bool) if channel_index == len(CHANNELS) else (
                    np.arange(len(CHANNELS)) == channel_index
                )
                waiting_tasks.append(
                    (
                        kind_index,
                        region_index,
                        channel_index,
                        event_obs,
                        valid_obs,
                        selected,
                        channels,
                        observation_cycles,
                    )
                )
    waiting_jobs = max(1, min(int(parallel_jobs), len(waiting_tasks)))
    if waiting_jobs > 1:
        waiting_results = Parallel(n_jobs=waiting_jobs, backend="threading")(
            delayed(_waiting_cell)(task) for task in waiting_tasks
        )
    else:
        waiting_results = [_waiting_cell(task) for task in waiting_tasks]
    for kind_index, region_index, channel_index, values, count in waiting_results:
        survival[kind_index, region_index, channel_index] = values
        interval_count[kind_index, region_index, channel_index] = count

    stationarity = np.full((len(activity_names), n_regions), np.nan, dtype=np.float64)
    time = np.arange(observation_cycles, dtype=np.float64)
    for kind_index in range(len(activity_names)):
        for region_index in range(n_regions):
            block = rates[kind_index, :, obs, region_index, -1]
            finite_count = np.sum(np.isfinite(block), axis=0)
            series = np.divide(
                np.nansum(block, axis=0),
                finite_count,
                out=np.full((observation_cycles,), np.nan, dtype=np.float64),
                where=finite_count > 0,
            )
            finite = np.isfinite(series)
            if np.count_nonzero(finite) >= 2:
                stationarity[kind_index, region_index] = np.polyfit(time[finite], series[finite], 1)[0]

    return ActivityAnalysisResult(
        region_names=regions.names,
        channel_names=channel_names,
        activity_names=activity_names,
        burn_in=burn_in,
        s_grid=s_grid,
        window_lengths=window_lengths,
        counts_by_cycle=counts,
        attempts_by_cycle=attempts,
        rates_by_cycle=rates,
        spatial_profile_x=spatial_profile,
        success_probability=success,
        cumulants_per_cycle=cumulants,
        theta=theta,
        theta_ci_low=ci_low,
        theta_ci_high=ci_high,
        effective_sample_fraction=ess,
        reliable=ess >= float(reliability_threshold),
        waiting_survival=survival,
        waiting_interval_count=interval_count,
        stationarity_slope=stationarity,
    )


CLICK_TYPES = ("A_loss", "A_gain", "B_loss", "B_gain")
CLICK_SIGNS = np.asarray((-1, 1, -1, 1), dtype=np.int8)
MOTIF_LABELS = tuple(
    "none"
    if code == 0
    else "+".join(name for bit, name in enumerate(CLICK_TYPES) if code & (1 << bit))
    for code in range(16)
)


@dataclass
class ClickSequenceAnalysisResult:
    """Exploratory spatial and temporal statistics of four-bit click motifs."""

    region_names: tuple[str, ...]
    motif_labels: tuple[str, ...]
    click_type_names: tuple[str, ...]
    burn_in: int
    permutation_count: int
    bootstrap_count: int
    minimum_support: int
    analysis_seed: int
    motif_counts: np.ndarray
    motif_frequency: np.ndarray
    temporal_pair_counts: np.ndarray
    temporal_pair_null_frequency: np.ndarray
    temporal_pair_log2_enrichment: np.ndarray
    temporal_pair_ci_low: np.ndarray
    temporal_pair_ci_high: np.ndarray
    spatial_pair_counts: np.ndarray
    spatial_pair_null_frequency: np.ndarray
    spatial_pair_log2_enrichment: np.ndarray
    spatial_pair_ci_low: np.ndarray
    spatial_pair_ci_high: np.ndarray
    temporal_triplet_counts: np.ndarray
    temporal_triplet_null_frequency: np.ndarray
    temporal_triplet_log2_enrichment: np.ndarray
    temporal_triplet_ci_low: np.ndarray
    temporal_triplet_ci_high: np.ndarray
    spatial_triplet_counts: np.ndarray
    spatial_triplet_null_frequency: np.ndarray
    spatial_triplet_log2_enrichment: np.ndarray
    spatial_triplet_ci_low: np.ndarray
    spatial_triplet_ci_high: np.ndarray

    def payload(self) -> dict[str, Any]:
        return {
            "region_names": np.asarray(self.region_names),
            "motif_labels": np.asarray(self.motif_labels),
            "click_type_names": np.asarray(self.click_type_names),
            "burn_in": np.asarray(self.burn_in, dtype=np.int64),
            "permutation_count": np.asarray(self.permutation_count, dtype=np.int64),
            "bootstrap_count": np.asarray(self.bootstrap_count, dtype=np.int64),
            "minimum_support": np.asarray(self.minimum_support, dtype=np.int64),
            "analysis_seed": np.asarray(self.analysis_seed, dtype=np.int64),
            "motif_counts": self.motif_counts,
            "motif_frequency": self.motif_frequency,
            "temporal_pair_counts": self.temporal_pair_counts,
            "temporal_pair_null_frequency": self.temporal_pair_null_frequency,
            "temporal_pair_log2_enrichment": self.temporal_pair_log2_enrichment,
            "temporal_pair_ci_low": self.temporal_pair_ci_low,
            "temporal_pair_ci_high": self.temporal_pair_ci_high,
            "spatial_pair_counts": self.spatial_pair_counts,
            "spatial_pair_null_frequency": self.spatial_pair_null_frequency,
            "spatial_pair_log2_enrichment": self.spatial_pair_log2_enrichment,
            "spatial_pair_ci_low": self.spatial_pair_ci_low,
            "spatial_pair_ci_high": self.spatial_pair_ci_high,
            "temporal_triplet_counts": self.temporal_triplet_counts,
            "temporal_triplet_null_frequency": self.temporal_triplet_null_frequency,
            "temporal_triplet_log2_enrichment": self.temporal_triplet_log2_enrichment,
            "temporal_triplet_ci_low": self.temporal_triplet_ci_low,
            "temporal_triplet_ci_high": self.temporal_triplet_ci_high,
            "spatial_triplet_counts": self.spatial_triplet_counts,
            "spatial_triplet_null_frequency": self.spatial_triplet_null_frequency,
            "spatial_triplet_log2_enrichment": self.spatial_triplet_log2_enrichment,
            "spatial_triplet_ci_low": self.spatial_triplet_ci_low,
            "spatial_triplet_ci_high": self.spatial_triplet_ci_high,
        }


def encode_click_motifs(
    transfer: np.ndarray,
    *,
    channel_names: Iterable[str] = CHANNELS,
    validate: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode Ap/Am/Bp/Bm signed transfers as four click bits and a uint8 motif."""
    values = np.asarray(transfer)
    names = tuple(str(name) for name in channel_names)
    if values.ndim < 1 or values.shape[-1] != len(names):
        raise ValueError("transfer must have one trailing axis matching channel_names.")
    if len(names) != len(CHANNELS) or set(names) != set(CHANNELS):
        raise ValueError(f"channel_names must be a permutation of {CHANNELS}.")
    if not np.issubdtype(values.dtype, np.integer):
        if not np.all(np.isfinite(values)) or not np.all(values == np.rint(values)):
            raise ValueError("transfer values must be finite integers.")
    ordered = np.stack([values[..., names.index(channel)] for channel in CHANNELS], axis=-1).astype(
        np.int8,
        copy=False,
    )
    if validate:
        valid = (ordered == 0) | (ordered == CLICK_SIGNS)
        if not np.all(valid):
            bad = np.argwhere(~valid)[0]
            channel = CHANNELS[int(bad[-1])]
            raise ValueError(
                f"Invalid signed transfer for {channel}: expected 0 or "
                f"{int(CLICK_SIGNS[int(bad[-1])])}."
            )
    bits = ordered != 0
    weights = np.asarray((1, 2, 4, 8), dtype=np.uint8)
    motif = np.sum(bits.astype(np.uint8) * weights, axis=-1, dtype=np.uint8)
    return bits, motif


def _site_region_labels(recorder: TrajectoryActivityRecorder, regions: RegionMasks) -> np.ndarray:
    x = recorder.site_ids % recorder.nx
    y = recorder.site_ids // recorder.nx
    labels = np.full((recorder.site_ids.size,), "all", dtype="<U12")
    for name in ("interior", "interface", "left_wall", "right_wall"):
        if name in regions.names:
            mask = regions.mask(name)
            labels[mask[x, y]] = name
    return labels


def _motif_grid(
    motif: np.ndarray,
    recorder: TrajectoryActivityRecorder,
) -> np.ndarray:
    grid = np.full(
        (recorder.samples, motif.shape[1], recorder.nx, recorder.ny),
        255,
        dtype=np.uint8,
    )
    x = recorder.site_ids % recorder.nx
    y = recorder.site_ids // recorder.nx
    grid[:, :, x, y] = motif
    return grid


def _site_region_selection(
    recorder: TrajectoryActivityRecorder,
    regions: RegionMasks,
) -> np.ndarray:
    x = recorder.site_ids % recorder.nx
    y = recorder.site_ids // recorder.nx
    return np.stack([mask[x, y] for mask in regions.masks], axis=0)


def _regularized_frequency(counts: np.ndarray) -> np.ndarray:
    values = np.asarray(counts, dtype=np.float64)
    states = int(values.shape[-1])
    return (values + 0.5) / (np.sum(values, axis=-1, keepdims=True) + 0.5 * states)


def _temporal_word_counts(
    motif: np.ndarray,
    site_region: np.ndarray,
    *,
    order: int,
) -> np.ndarray:
    motif = np.asarray(motif, dtype=np.uint8)
    order = int(order)
    if motif.ndim != 3 or order not in (2, 3):
        raise ValueError("Temporal words require sample×time×site motifs and order 2 or 3.")
    counts = np.zeros((motif.shape[0], site_region.shape[0], 16**order), dtype=np.int64)
    if motif.shape[1] < order:
        return counts
    length = motif.shape[1] - order + 1
    for region_index, selected in enumerate(site_region):
        if not np.any(selected):
            continue
        code = np.zeros((motif.shape[0], length, int(np.count_nonzero(selected))), dtype=np.int64)
        for offset in range(order):
            code = 16 * code + motif[:, offset : offset + length, selected]
        for sample in range(motif.shape[0]):
            counts[sample, region_index] = np.bincount(
                code[sample].reshape(-1),
                minlength=16**order,
            )
    return counts


def _spatial_word_counts(
    grid: np.ndarray,
    regions: RegionMasks,
    *,
    order: int,
) -> np.ndarray:
    grid = np.asarray(grid, dtype=np.uint8)
    order = int(order)
    if grid.ndim != 4 or order not in (2, 3):
        raise ValueError("Spatial words require sample×time×x×y motifs and order 2 or 3.")
    if grid.shape[-1] < order:
        raise ValueError("The periodic y ring is shorter than the requested spatial word.")
    counts = np.zeros((grid.shape[0], len(regions.names), 16**order), dtype=np.int64)
    for region_index, mask in enumerate(regions.masks):
        row_any = np.any(mask, axis=1)
        row_all = np.all(mask, axis=1)
        if not np.array_equal(row_any, row_all):
            raise ValueError("Spatial word regions must contain complete periodic y rings.")
        x_selection = np.flatnonzero(row_all)
        if x_selection.size == 0:
            continue
        block = grid[:, :, x_selection, :]
        if np.any(block == 255):
            raise ValueError("A selected spatial ring contains an unrecorded unit cell.")
        code = np.zeros(block.shape, dtype=np.int64)
        for offset in range(order):
            code = 16 * code + np.roll(block, -offset, axis=-1)
        for sample in range(grid.shape[0]):
            counts[sample, region_index] = np.bincount(
                code[sample].reshape(-1),
                minlength=16**order,
            )
    return counts


def _permute_temporal_motifs(motif: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Independently permute post-burn-in cycles for every sample and active cell."""
    indices = np.argsort(rng.random(motif.shape), axis=1)
    return np.take_along_axis(motif, indices, axis=1)


def _permute_spatial_grid(grid: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Independently permute y positions on every active sample/cycle/x ring."""
    shuffled = np.array(grid, copy=True)
    active_x = np.flatnonzero(np.all(np.any(grid != 255, axis=(0, 1)), axis=1))
    for x in active_x:
        block = grid[:, :, x, :]
        if np.any(block == 255):
            raise ValueError("Active x columns must contain complete periodic y rings.")
        indices = np.argsort(rng.random(block.shape), axis=-1)
        shuffled[:, :, x, :] = np.take_along_axis(block, indices, axis=-1)
    return shuffled


def _bootstrap_enrichment_interval(
    sample_counts: np.ndarray,
    null_replicates: np.ndarray,
    *,
    bootstrap_count: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    samples = int(sample_counts.shape[0])
    draws = np.empty(
        (int(bootstrap_count), sample_counts.shape[1], sample_counts.shape[2]),
        dtype=np.float32,
    )
    for draw in range(int(bootstrap_count)):
        selection = rng.integers(0, samples, size=samples)
        observed = _regularized_frequency(np.sum(sample_counts[selection], axis=0))
        null = null_replicates[int(rng.integers(0, null_replicates.shape[0]))]
        draws[draw] = np.log2(observed / null)
    return (
        np.quantile(draws, 0.025, axis=0).astype(np.float64),
        np.quantile(draws, 0.975, axis=0).astype(np.float64),
    )


def _word_statistics(
    sample_counts: np.ndarray,
    null_replicates: np.ndarray,
    *,
    bootstrap_count: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    aggregate = np.sum(sample_counts, axis=0)
    observed = _regularized_frequency(aggregate)
    null_mean = np.mean(null_replicates, axis=0, dtype=np.float64)
    enrichment = np.log2(observed / null_mean)
    low, high = _bootstrap_enrichment_interval(
        sample_counts,
        null_replicates,
        bootstrap_count=bootstrap_count,
        rng=rng,
    )
    return aggregate, null_mean, enrichment, low, high


def analyze_click_sequences(
    recorder: TrajectoryActivityRecorder,
    *,
    regions: RegionMasks,
    burn_in: int,
    permutations: int = 1000,
    bootstrap_samples: int = 1000,
    minimum_support: int = 20,
    seed: int = 0,
) -> ClickSequenceAnalysisResult:
    """Analyze four-bit unit-cell motifs with temporal and periodic-y shuffle nulls."""
    recorder.assert_complete()
    burn_in = int(burn_in)
    permutations = int(permutations)
    bootstrap_samples = int(bootstrap_samples)
    minimum_support = int(minimum_support)
    if burn_in < 0 or burn_in >= recorder.cycles - 1:
        raise ValueError("burn_in must leave at least two observed cycles.")
    if permutations <= 0 or bootstrap_samples <= 0:
        raise ValueError("permutations and bootstrap_samples must be positive.")
    if minimum_support < 1:
        raise ValueError("minimum_support must be positive.")

    _, motif_all = encode_click_motifs(recorder.transfer, channel_names=CHANNELS, validate=True)
    motif = motif_all[:, burn_in:, :]
    grid = _motif_grid(motif, recorder)
    site_region = _site_region_selection(recorder, regions)

    motif_by_sample = np.zeros((recorder.samples, len(regions.names), 16), dtype=np.int64)
    for region_index, selected in enumerate(site_region):
        for sample in range(recorder.samples):
            motif_by_sample[sample, region_index] = np.bincount(
                motif[sample, :, selected].reshape(-1),
                minlength=16,
            )
    motif_counts = np.sum(motif_by_sample, axis=0)
    motif_denominator = np.sum(motif_counts, axis=-1, keepdims=True)
    motif_frequency = np.divide(
        motif_counts,
        motif_denominator,
        out=np.full(motif_counts.shape, np.nan, dtype=np.float64),
        where=motif_denominator > 0,
    )

    temporal_pair_samples = _temporal_word_counts(motif, site_region, order=2)
    temporal_triplet_samples = _temporal_word_counts(motif, site_region, order=3)
    spatial_pair_samples = _spatial_word_counts(grid, regions, order=2)
    spatial_triplet_samples = _spatial_word_counts(grid, regions, order=3)

    seed_sequence = np.random.SeedSequence(int(seed))
    temporal_seed, spatial_seed, temporal_boot_seed, spatial_boot_seed = seed_sequence.spawn(4)
    temporal_rng = np.random.default_rng(temporal_seed)
    spatial_rng = np.random.default_rng(spatial_seed)
    temporal_pair_null = np.empty(
        (permutations, len(regions.names), 16**2),
        dtype=np.float32,
    )
    temporal_triplet_null = np.empty(
        (permutations, len(regions.names), 16**3),
        dtype=np.float32,
    )
    for index in range(permutations):
        shuffled = _permute_temporal_motifs(motif, temporal_rng)
        temporal_pair_null[index] = _regularized_frequency(
            np.sum(_temporal_word_counts(shuffled, site_region, order=2), axis=0)
        )
        temporal_triplet_null[index] = _regularized_frequency(
            np.sum(_temporal_word_counts(shuffled, site_region, order=3), axis=0)
        )

    temporal_boot_rng = np.random.default_rng(temporal_boot_seed)
    temporal_pair = _word_statistics(
        temporal_pair_samples,
        temporal_pair_null,
        bootstrap_count=bootstrap_samples,
        rng=temporal_boot_rng,
    )
    temporal_triplet = _word_statistics(
        temporal_triplet_samples,
        temporal_triplet_null,
        bootstrap_count=bootstrap_samples,
        rng=temporal_boot_rng,
    )
    del temporal_pair_null, temporal_triplet_null

    spatial_pair_null = np.empty(
        (permutations, len(regions.names), 16**2),
        dtype=np.float32,
    )
    spatial_triplet_null = np.empty(
        (permutations, len(regions.names), 16**3),
        dtype=np.float32,
    )
    for index in range(permutations):
        shuffled = _permute_spatial_grid(grid, spatial_rng)
        spatial_pair_null[index] = _regularized_frequency(
            np.sum(_spatial_word_counts(shuffled, regions, order=2), axis=0)
        )
        spatial_triplet_null[index] = _regularized_frequency(
            np.sum(_spatial_word_counts(shuffled, regions, order=3), axis=0)
        )

    spatial_boot_rng = np.random.default_rng(spatial_boot_seed)
    spatial_pair = _word_statistics(
        spatial_pair_samples,
        spatial_pair_null,
        bootstrap_count=bootstrap_samples,
        rng=spatial_boot_rng,
    )
    spatial_triplet = _word_statistics(
        spatial_triplet_samples,
        spatial_triplet_null,
        bootstrap_count=bootstrap_samples,
        rng=spatial_boot_rng,
    )

    return ClickSequenceAnalysisResult(
        region_names=regions.names,
        motif_labels=MOTIF_LABELS,
        click_type_names=CLICK_TYPES,
        burn_in=burn_in,
        permutation_count=permutations,
        bootstrap_count=bootstrap_samples,
        minimum_support=minimum_support,
        analysis_seed=int(seed),
        motif_counts=motif_counts,
        motif_frequency=motif_frequency,
        temporal_pair_counts=temporal_pair[0],
        temporal_pair_null_frequency=temporal_pair[1],
        temporal_pair_log2_enrichment=temporal_pair[2],
        temporal_pair_ci_low=temporal_pair[3],
        temporal_pair_ci_high=temporal_pair[4],
        spatial_pair_counts=spatial_pair[0],
        spatial_pair_null_frequency=spatial_pair[1],
        spatial_pair_log2_enrichment=spatial_pair[2],
        spatial_pair_ci_low=spatial_pair[3],
        spatial_pair_ci_high=spatial_pair[4],
        temporal_triplet_counts=temporal_triplet[0],
        temporal_triplet_null_frequency=temporal_triplet[1],
        temporal_triplet_log2_enrichment=temporal_triplet[2],
        temporal_triplet_ci_low=temporal_triplet[3],
        temporal_triplet_ci_high=temporal_triplet[4],
        spatial_triplet_counts=spatial_triplet[0],
        spatial_triplet_null_frequency=spatial_triplet[1],
        spatial_triplet_log2_enrichment=spatial_triplet[2],
        spatial_triplet_ci_low=spatial_triplet[3],
        spatial_triplet_ci_high=spatial_triplet[4],
    )


def _decode_word(code: int, order: int) -> tuple[int, ...]:
    values = [0] * int(order)
    remainder = int(code)
    for index in range(int(order) - 1, -1, -1):
        values[index] = remainder % 16
        remainder //= 16
    return tuple(values)


def click_sequence_candidate_rows(
    result: ClickSequenceAnalysisResult,
) -> list[dict[str, Any]]:
    """Return supported pair/triplet words ranked by absolute enrichment."""
    rows: list[dict[str, Any]] = []
    families = (
        ("temporal_pair", 2, result.temporal_pair_counts, result.temporal_pair_null_frequency, result.temporal_pair_log2_enrichment, result.temporal_pair_ci_low, result.temporal_pair_ci_high),
        ("spatial_pair", 2, result.spatial_pair_counts, result.spatial_pair_null_frequency, result.spatial_pair_log2_enrichment, result.spatial_pair_ci_low, result.spatial_pair_ci_high),
        ("temporal_triplet", 3, result.temporal_triplet_counts, result.temporal_triplet_null_frequency, result.temporal_triplet_log2_enrichment, result.temporal_triplet_ci_low, result.temporal_triplet_ci_high),
        ("spatial_triplet", 3, result.spatial_triplet_counts, result.spatial_triplet_null_frequency, result.spatial_triplet_log2_enrichment, result.spatial_triplet_ci_low, result.spatial_triplet_ci_high),
    )
    for family, order, counts, null, enrichment, low, high in families:
        for region_index, region in enumerate(result.region_names):
            for word_code in np.flatnonzero(counts[region_index] >= result.minimum_support):
                motifs = _decode_word(int(word_code), order)
                rows.append(
                    {
                        "family": family,
                        "region": region,
                        "word_code": int(word_code),
                        "motif_codes": "-".join(str(value) for value in motifs),
                        "motif_labels": " -> ".join(result.motif_labels[value] for value in motifs),
                        "count": int(counts[region_index, word_code]),
                        "null_frequency": float(null[region_index, word_code]),
                        "log2_enrichment": float(enrichment[region_index, word_code]),
                        "ci_low": float(low[region_index, word_code]),
                        "ci_high": float(high[region_index, word_code]),
                        "candidate": bool(low[region_index, word_code] > 0.0 or high[region_index, word_code] < 0.0),
                    }
                )
    rows.sort(
        key=lambda row: (
            not bool(row["candidate"]),
            -abs(float(row["log2_enrichment"])),
            -int(row["count"]),
        )
    )
    return rows


def activity_record_frames(
    recorder: TrajectoryActivityRecorder,
    *,
    regions: RegionMasks,
    schedule: str,
    trial_pauli: str = "X",
):
    """Build tidy event and unit-cell motif tables; pandas is imported lazily."""
    import pandas as pd

    recorder.assert_complete()
    bits, motif = encode_click_motifs(recorder.transfer, channel_names=CHANNELS)
    samples, cycles, sites, channels = recorder.transfer.shape
    shape = (samples, cycles, sites, channels)
    sample_grid, cycle_grid, site_grid, channel_grid = np.indices(shape, sparse=False)
    visit = np.broadcast_to(recorder.visit_order[..., None], shape)
    global_site_step = cycle_grid * sites + visit
    global_channel_step = 4 * global_site_step + channel_grid
    site_x = recorder.site_ids % recorder.nx
    site_y = recorder.site_ids // recorder.nx
    channel_array = np.asarray(CHANNELS)
    trial_array = np.asarray(("A", "A", "B", "B"))
    eigenvalue_array = np.asarray((1, 1, -1, -1), dtype=np.int8)
    target_array = np.asarray([CHANNEL_TARGETS[channel] for channel in CHANNELS], dtype=np.int8)
    transfer = recorder.transfer.reshape(-1)
    click_label = np.where(transfer < 0, "loss", np.where(transfer > 0, "gain", "none"))
    region_label = _site_region_labels(recorder, regions)
    wall_x = tuple(int(value) for value in regions.wall_x)
    left_wall = wall_x[0] if wall_x else -1
    right_wall = wall_x[-1] if wall_x else -1
    flattened_sites = site_grid.reshape(-1)
    flattened_channels = channel_grid.reshape(-1)

    events = pd.DataFrame(
        {
            "schedule": str(schedule),
            "sample": sample_grid.reshape(-1).astype(np.int16),
            "cycle": (cycle_grid.reshape(-1) + 1).astype(np.int16),
            "visit_order": visit.reshape(-1).astype(np.int32),
            "global_site_step": global_site_step.reshape(-1).astype(np.int32),
            "global_channel_step": global_channel_step.reshape(-1).astype(np.int32),
            "x": site_x[flattened_sites].astype(np.int16),
            "y": site_y[flattened_sites].astype(np.int16),
            "region": region_label[flattened_sites],
            "distance_left_wall": np.abs(site_x[flattened_sites] - left_wall).astype(np.int16),
            "distance_right_wall": np.abs(site_x[flattened_sites] - right_wall).astype(np.int16),
            "channel": channel_array[flattened_channels],
            "trial_orbital": trial_array[flattened_channels],
            "trial_pauli": str(trial_pauli).upper(),
            "trial_pauli_eigenvalue": eigenvalue_array[flattened_channels],
            "band": np.where(np.char.endswith(channel_array[flattened_channels], "p"), "upper", "lower"),
            "target_occupied": target_array[flattened_channels],
            "defect": recorder.defect.reshape(-1).astype(bool),
            "transfer": transfer.astype(np.int8),
            "click": click_label,
            "success_probability": recorder.success_probability.reshape(-1),
            "valid": recorder.valid.reshape(-1),
        }
    )

    motif_shape = (samples, cycles, sites)
    motif_sample, motif_cycle, motif_site = np.indices(motif_shape, sparse=False)
    motif_visit = recorder.visit_order
    motif_global = motif_cycle * sites + motif_visit
    flattened_motif_sites = motif_site.reshape(-1)
    motifs = pd.DataFrame(
        {
            "schedule": str(schedule),
            "sample": motif_sample.reshape(-1).astype(np.int16),
            "cycle": (motif_cycle.reshape(-1) + 1).astype(np.int16),
            "visit_order": motif_visit.reshape(-1).astype(np.int32),
            "global_site_step": motif_global.reshape(-1).astype(np.int32),
            "x": site_x[flattened_motif_sites].astype(np.int16),
            "y": site_y[flattened_motif_sites].astype(np.int16),
            "region": region_label[flattened_motif_sites],
            "motif_code": motif.reshape(-1).astype(np.uint8),
            "motif_label": np.asarray(MOTIF_LABELS)[motif.reshape(-1)],
            "A_loss": bits[..., 0].reshape(-1),
            "A_gain": bits[..., 1].reshape(-1),
            "B_loss": bits[..., 2].reshape(-1),
            "B_gain": bits[..., 3].reshape(-1),
        }
    )
    return events, motifs
