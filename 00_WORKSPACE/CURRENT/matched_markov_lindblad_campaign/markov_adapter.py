"""Production adapter for the exact discrete perfect-correction mean channel."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from campaign_schema import case_requires_response, late_cycle_bounds
from matched_model import build_model
from observables import y_translate


@dataclass
class MarkovAdapterResult:
    arrays: dict[str, np.ndarray]
    metadata: dict[str, Any]


CHANNEL_ORDER = ("Ap", "Am", "Bp", "Bm")
TARGET_OCCUPIED = {"Ap": False, "Am": True, "Bp": False, "Bm": True}


def _spectral_checkpoint_cycles(ny: int) -> np.ndarray:
    return np.asarray([0, ny // 2, ny, (3 * ny) // 2, 2 * ny], dtype=np.int64)


def _binary_entropy(values: np.ndarray) -> float:
    occupations = np.clip(np.asarray(values, dtype=float), 0.0, 1.0)
    interior = (occupations > 0.0) & (occupations < 1.0)
    terms = np.zeros_like(occupations)
    terms[interior] = -(
        occupations[interior] * np.log(occupations[interior])
        + (1.0 - occupations[interior])
        * np.log(1.0 - occupations[interior])
    )
    return float(np.sum(terms))


def _raw_to_physical(raw: np.ndarray, output: np.ndarray | None = None) -> np.ndarray:
    """Convert ``Q=2G-1`` without materializing a dense identity matrix."""

    source = np.asarray(raw, dtype=np.complex128)
    physical = np.empty_like(source) if output is None else output
    np.multiply(source, 0.5, out=physical)
    diagonal = np.diag_indices_from(physical)
    physical[diagonal] += 0.5
    return physical


def _apply_site_word(model: Any, vectors: np.ndarray, site_ids: np.ndarray) -> None:
    """Apply the exact homogeneous product of local ``Q=1-P`` maps in place."""

    if model.nshell is None:
        raise ValueError("the discrete Markov production arm is finite-shell only")
    for site_id in np.asarray(site_ids, dtype=np.int64):
        x = int(site_id) % int(model.Nx)
        y = int(site_id) // int(model.Nx)
        payload = model._get_ow_local_support_data(x, y)
        support = payload["idx"]
        for channel in CHANNEL_ORDER:
            chi = payload[channel]
            restricted = vectors[support]
            overlap = chi.conj() @ restricted
            vectors[support] = restricted - chi[:, None] * overlap[None, :]


def _leading_contraction_cocycle(
    model: Any,
    schedule_words: np.ndarray,
    *,
    seed: int,
    n_vectors: int = 8,
) -> np.ndarray:
    """Estimate leading covariance multipliers of the realized schedule cocycle."""

    dimension = 2 * int(model.Nx) * int(model.Ny)
    count = min(int(n_vectors), dimension)
    rng = np.random.default_rng(int(seed) ^ 0x5A17C0C1)
    frame = rng.standard_normal((dimension, count)) + 1j * rng.standard_normal(
        (dimension, count)
    )
    frame, _ = np.linalg.qr(frame, mode="reduced")
    log_scales = np.zeros(count, dtype=float)
    for word in np.asarray(schedule_words, dtype=np.int64):
        _apply_site_word(model, frame, word)
        frame, triangular = np.linalg.qr(frame, mode="reduced")
        diagonal = np.maximum(np.abs(np.diag(triangular)), 1e-300)
        log_scales += np.log(diagonal)
    exponents = log_scales / max(int(schedule_words.shape[0]), 1)
    covariance_multipliers = np.exp(2.0 * exponents)
    return np.sort(covariance_multipliers)[::-1]


def _periodic_displacements(ny: int) -> np.ndarray:
    values = np.arange(int(ny), dtype=float)
    return ((values + ny / 2.0) % ny) - ny / 2.0


def _fit_velocity(
    times: np.ndarray,
    centers: np.ndarray,
    norms: np.ndarray,
    *,
    fit_min: float,
    fit_max: float,
) -> tuple[float, float]:
    valid = (
        (times >= float(fit_min))
        & (times <= float(fit_max))
        & np.isfinite(centers)
        & (norms > max(1e-8 * float(np.max(norms)), 1e-15))
    )
    if np.count_nonzero(valid) < 2:
        return math.nan, math.nan
    x = times[valid]
    y = centers[valid]
    slope, intercept = np.polyfit(x, y, 1)
    fitted = slope * x + intercept
    residual = float(np.sum((y - fitted) ** 2))
    total = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - residual / total if total > 0.0 else math.nan
    return float(slope), float(r2)


def _channel_response(
    model: Any,
    correlation: np.ndarray,
    schedule_words: np.ndarray,
    config: dict[str, Any],
) -> dict[str, np.ndarray]:
    """Propagate every wall/source-y density kick in a rank-four representation."""

    nx, ny = int(model.Nx), int(model.Ny)
    dimension = 2 * nx * ny
    horizon = ny // 2
    if schedule_words.shape[0] < horizon:
        raise ValueError("response schedule segment is shorter than Ny/2")
    walls = [int(value) for value in model.DW_loc]
    source_basis: list[int] = []
    source_records: list[tuple[int, int]] = []
    for wall_index, wall in enumerate(walls):
        for source_y in range(ny):
            source_records.append((wall_index, source_y))
            source_basis.extend(
                [mu + 2 * wall + 2 * nx * source_y for mu in (0, 1)]
            )
    source_basis_array = np.asarray(source_basis, dtype=np.int64)
    source_count = len(source_records)
    column_count = source_basis_array.size
    left = np.zeros((dimension, column_count), dtype=np.complex128)
    left[source_basis_array, np.arange(column_count)] = 1.0
    right = np.asarray(correlation[:, source_basis_array], dtype=np.complex128).copy()
    propagated = np.concatenate([left, right], axis=1)

    times = np.arange(horizon + 1, dtype=float)
    raw = np.zeros((ny, 2, horizon + 1, ny), dtype=float)
    norms = np.zeros((ny, 2, horizon + 1), dtype=float)
    retention = np.zeros_like(norms)
    centers = np.full_like(norms, np.nan)
    displacement = _periodic_displacements(ny)
    window_radius = int(config["response"]["wall_window_columns"])

    def observe(time_index: int) -> None:
        u = propagated[:, :column_count].reshape(dimension, source_count, 2)
        v = propagated[:, column_count:].reshape(dimension, source_count, 2)
        diagonal = -2.0 * np.imag(np.sum(u * v.conj(), axis=2)).T
        density = diagonal.reshape(source_count, ny, nx, 2).sum(axis=3)
        for source_index, (wall_index, source_y) in enumerate(source_records):
            columns = sorted(
                {(walls[wall_index] + delta) % nx for delta in range(-window_radius, window_radius + 1)}
            )
            density_at_source = density[source_index]
            profile = np.sum(density_at_source[:, columns], axis=1)
            raw[source_y, wall_index, time_index] = profile
            full_norm = float(np.sum(np.abs(density[source_index])))
            wall_norm = float(np.sum(np.abs(density_at_source[:, columns])))
            norms[source_y, wall_index, time_index] = full_norm
            retention[source_y, wall_index, time_index] = (
                wall_norm / full_norm if full_norm > 0.0 else math.nan
            )
            shifted = np.roll(profile, -source_y)
            positive = np.maximum(shifted, 0.0)
            denominator = float(np.sum(positive))
            if denominator > 0.0:
                centers[source_y, wall_index, time_index] = float(
                    np.sum(displacement * positive) / denominator
                )

    observe(0)
    for time_index, word in enumerate(schedule_words[:horizon], start=1):
        _apply_site_word(model, propagated, word)
        observe(time_index)

    shifted_profiles = np.empty_like(raw)
    for source_y in range(ny):
        shifted_profiles[source_y] = np.roll(raw[source_y], -source_y, axis=-1)
    mean_source = np.mean(shifted_profiles, axis=0)

    velocity = np.full((ny, 2), np.nan)
    velocity_r2 = np.full((ny, 2), np.nan)
    for source_y in range(ny):
        for wall_index in range(2):
            velocity[source_y, wall_index], velocity_r2[source_y, wall_index] = _fit_velocity(
                times,
                centers[source_y, wall_index],
                norms[source_y, wall_index],
                fit_min=float(config["response"]["fit_time_min"]),
                fit_max=0.375 * ny,
            )

    epsilon = float(config["response"]["epsilon"])
    multipliers = np.asarray(config["response"]["epsilon_multipliers"], dtype=float)
    sinc = lambda value: float(np.sinc(float(value) / np.pi))
    baseline = sinc(epsilon)
    epsilon_error = np.asarray(
        [abs(sinc(epsilon * value) - baseline) / max(abs(baseline), 1e-300) for value in multipliers]
    )
    scale = baseline
    return {
        "response_times": times,
        "response_source_y": np.arange(ny, dtype=np.int64),
        "response_density_source_wall_time_y": scale * raw,
        "response_density_ty_mean_source": scale * mean_source,
        "response_norm_time": abs(scale) * norms,
        "response_wall_retention_time": retention,
        "response_positive_center_time": centers,
        "response_velocity_source_wall": velocity,
        "response_velocity_r2_source_wall": velocity_r2,
        "response_epsilon_relative_error": epsilon_error,
    }


def run_markov_channel_case(
    case: dict[str, Any], config: dict[str, Any]
) -> MarkovAdapterResult:
    dynamics = case["dynamics"]
    if dynamics["family"] != "markov_channel":
        raise ValueError("run_markov_channel_case received a non-channel case")
    if not dynamics["perfect_correction"] or not dynamics["dephasing"]:
        raise ValueError("production channel cases require perfect correction and dephasing")
    if dynamics["site_schedule"] != "random":
        raise ValueError("production channel cases require a fresh random permutation each cycle")

    model = build_model(case)
    nx, ny = int(model.Nx), int(model.Ny)
    dimension = 2 * nx * ny
    cycles = int(dynamics["cycles"])
    if cycles != 2 * ny:
        raise ValueError("channel cycles must equal 2*Ny")
    if list(model.DW_loc) != list(case["model"]["wall_locations"]):
        raise ValueError("resolved channel wall locations do not match the case")
    late_start, late_end = late_cycle_bounds(case)
    checkpoint_cycle = _spectral_checkpoint_cycles(ny)
    checkpoint_lookup = {int(value): index for index, value in enumerate(checkpoint_cycle)}
    sample_seeds = [int(value) for value in dynamics["sample_seeds"]]
    sample_count = len(sample_seeds)
    site_count = nx * ny

    translation_all = np.empty((sample_count, cycles + 1), dtype=float)
    successive_all = np.empty_like(translation_all)
    occupation_min_all = np.empty((sample_count, checkpoint_cycle.size), dtype=float)
    occupation_max_all = np.empty_like(occupation_min_all)
    half_gap_all = np.empty_like(occupation_min_all)
    entropy_all = np.empty_like(occupation_min_all)
    charge_variance_all = np.empty_like(occupation_min_all)
    final_all = np.empty((sample_count, dimension, dimension), dtype=np.complex128)
    late_all = np.empty_like(final_all)
    schedule_all = np.empty((sample_count, cycles, site_count), dtype=np.int64)
    relaxation_all = []
    response_records: list[dict[str, np.ndarray]] = []
    response_enabled = case_requires_response(case, config)

    for sample_index, seed in enumerate(sample_seeds):
        previous_raw: np.ndarray | None = None
        late_sum = np.zeros((dimension, dimension), dtype=np.complex128)
        late_count = 0
        observed_cycle: list[int] = []

        def observer(**payload: Any) -> None:
            nonlocal previous_raw, late_count
            cycle = int(payload["cycle"])
            raw = np.asarray(payload["G"], dtype=np.complex128)
            observed_cycle.append(cycle)
            raw_norm_sq = float(np.vdot(raw, raw).real)
            trace_raw = float(np.trace(raw).real)
            correlation_norm = 0.5 * math.sqrt(
                max(raw_norm_sq + dimension + 2.0 * trace_raw, 0.0)
            )
            shifted = y_translate(raw, nx, ny)
            translation_all[sample_index, cycle] = (
                0.5 * float(np.linalg.norm(shifted - raw))
                / max(correlation_norm, 1e-300)
            )
            successive_all[sample_index, cycle] = (
                0.0
                if previous_raw is None
                else 0.5
                * float(np.linalg.norm(raw - previous_raw))
                / math.sqrt(dimension)
            )
            previous_raw = raw.copy()
            if cycle > 0:
                ordered = np.asarray(payload["ordered_site_ids"], dtype=np.int64)
                if ordered.shape != (site_count,) or np.unique(ordered).size != site_count:
                    raise ValueError("random schedule must visit every site exactly once")
                schedule_all[sample_index, cycle - 1] = ordered
            if late_start <= cycle <= late_end:
                late_sum[:] += raw
                late_count += 1
            checkpoint_index = checkpoint_lookup.get(cycle)
            if checkpoint_index is not None:
                correlation = _raw_to_physical(raw)
                occupations = np.linalg.eigvalsh(
                    0.5 * (correlation + correlation.conj().T)
                ).real
                occupation_min_all[sample_index, checkpoint_index] = occupations[0]
                occupation_max_all[sample_index, checkpoint_index] = occupations[-1]
                half_gap_all[sample_index, checkpoint_index] = float(
                    np.min(np.abs(occupations - 0.5))
                )
                entropy_all[sample_index, checkpoint_index] = _binary_entropy(occupations) / ny
                charge_variance_all[sample_index, checkpoint_index] = float(
                    np.sum(occupations * (1.0 - occupations))
                )

        result = model.run_markov_channel(
            G_history=False,
            progress=False,
            cycles=cycles,
            init_mode=dynamics["init_mode"],
            save=False,
            n_a=float(config["model"]["n_a_metadata"]),
            sequence=dynamics["site_schedule"],
            decoh=True,
            perfect_correction=True,
            schedule_seed=seed,
            cycle_observer=observer,
        )
        if observed_cycle != list(range(cycles + 1)):
            raise ValueError("channel observer did not receive the complete cycle coordinate")
        expected_late_count = late_end - late_start + 1
        if late_count != expected_late_count:
            raise ValueError("late-cycle streaming average is incomplete")
        raw_final = np.asarray(result["G_final"], dtype=np.complex128)
        _raw_to_physical(raw_final, final_all[sample_index])
        np.divide(late_sum, float(late_count), out=late_sum)
        _raw_to_physical(late_sum, late_all[sample_index])
        relaxation_all.append(
            _leading_contraction_cocycle(model, schedule_all[sample_index], seed=seed)
        )
        if response_enabled:
            response_words = schedule_all[sample_index, ny : ny + ny // 2]
            response_records.append(
                _channel_response(model, late_all[sample_index], response_words, config)
            )

    arrays: dict[str, np.ndarray] = {
        "cycle": np.arange(cycles + 1, dtype=np.int64),
        "translation_residual": translation_all,
        "successive_state_distance": successive_all,
        "spectral_checkpoint_cycle": checkpoint_cycle,
        "occupation_min": occupation_min_all,
        "occupation_max": occupation_max_all,
        "half_occupation_gap": half_gap_all,
        "gaussian_entropy_proxy_per_circumference": entropy_all,
        "gaussian_charge_variance_proxy": charge_variance_all,
        "G_final": final_all,
        "G_late_cycle_average": late_all,
        "schedule_site_ids": schedule_all,
        "leading_relaxation_spectrum": np.asarray(relaxation_all),
    }
    if response_enabled:
        arrays["response_times"] = response_records[0]["response_times"]
        arrays["response_source_y"] = response_records[0]["response_source_y"]
        for name in (
            "response_density_source_wall_time_y",
            "response_density_ty_mean_source",
            "response_norm_time",
            "response_wall_retention_time",
            "response_positive_center_time",
            "response_velocity_source_wall",
            "response_velocity_r2_source_wall",
            "response_epsilon_relative_error",
        ):
            arrays[name] = np.stack([record[name] for record in response_records])

    projector_hashes = {
        name: model._checkpoint_array_signature(getattr(model, name))
        for name in ("WF_Ap", "WF_Am", "WF_Bp", "WF_Bm")
    }
    metadata = {
        "canonical_dynamics_entry_point": "classA_U1FGTN.run_markov_channel",
        "adapter_entry_point": "markov_adapter.run_markov_channel_case",
        "raw_engine_convention": "Q=2*G-identity",
        "saved_correlation_convention": "G=(Q+identity)/2",
        "perfect_correction": True,
        "dephasing": True,
        "channel_order": list(CHANNEL_ORDER),
        "site_schedule": "random fresh permutation per cycle",
        "sample_seeds": sample_seeds,
        "wall_locations": [int(value) for value in model.DW_loc],
        "dw_interval_inclusive": True,
        "late_cycle_window_inclusive": [late_start, late_end],
        "spectral_checkpoint_cycles": checkpoint_cycle.tolist(),
        "dense_history_saved": False,
        "response_enabled": response_enabled,
        "response_schedule_segment": [ny + 1, ny + ny // 2] if response_enabled else None,
        "relaxation_estimator": "eight-vector QR homogeneous schedule cocycle; covariance multipliers per cycle",
        "projector_hashes": projector_hashes,
    }
    return MarkovAdapterResult(arrays=arrays, metadata=metadata)
