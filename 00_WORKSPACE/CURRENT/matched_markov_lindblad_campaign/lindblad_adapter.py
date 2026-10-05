"""Production adapter for the matched infinitesimal perfect-correction dynamics."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh

from campaign_schema import (
    case_requires_response,
    late_cycle_bounds,
    spectral_checkpoint_cycles,
)
from lindblad_response import local_density_kick_response
from matched_model import build_model
from src.fgtn.diagnostics.mean_lindblad import PerfectCorrectionLindblad


@dataclass
class LindbladAdapterResult:
    arrays: dict[str, np.ndarray]
    metadata: dict[str, Any]


def _binary_entropy(occupations: np.ndarray) -> float:
    values = np.clip(np.asarray(occupations, dtype=np.float64), 0.0, 1.0)
    interior = (values > 0.0) & (values < 1.0)
    terms = np.zeros_like(values)
    terms[interior] = -(
        values[interior] * np.log(values[interior])
        + (1.0 - values[interior]) * np.log(1.0 - values[interior])
    )
    return float(np.sum(terms))


def _spectral_scalars(blocks: np.ndarray, *, ny: int) -> tuple[float, ...]:
    occupations = np.linalg.eigvalsh(blocks).real
    return (
        float(np.min(occupations)),
        float(np.max(occupations)),
        float(np.min(np.abs(occupations - 0.5))),
        _binary_entropy(occupations) / float(ny),
        float(np.sum(occupations * (1.0 - occupations))),
    )


def _leading_relaxation_spectrum(
    generator: PerfectCorrectionLindblad,
    *,
    include_number_dephasing: bool,
    count: int = 16,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Leading homogeneous generator eigenvalues in the translation-invariant q=0 sector."""

    if not include_number_dephasing:
        operators = generator.momentum_frame_operators
        damping = 0.5 * (
            operators["A_minus"]
            + operators["B_minus"]
            + operators["A_plus"]
            + operators["B_plus"]
        )
        damping_eigenvalues = np.linalg.eigvalsh(damping).real
        rates = (
            damping_eigenvalues[:, :, None]
            + damping_eigenvalues[:, None, :]
        ).reshape(-1)
        selected_count = min(max(int(count), 1), rates.size)
        selected = np.partition(rates, selected_count - 1)[:selected_count]
        values = -np.sort(selected)
        return values, {
            "method": "exact q=0 damping-rate pair sums",
            "q_sector": 0,
            "requested_count": int(count),
        }

    dimension = (
        int(generator.ny)
        * int(generator.block_dimension)
        * int(generator.block_dimension)
    )
    selected_count = min(max(int(count), 1), max(dimension - 2, 1))

    def matvec(vector: np.ndarray) -> np.ndarray:
        blocks = np.asarray(vector, dtype=np.complex128).reshape(
            generator.ny, generator.block_dimension, generator.block_dimension
        )
        return generator.q_sector_homogeneous_rhs(
            blocks,
            q_index=0,
            include_number_dephasing=True,
        ).reshape(-1)

    operator = LinearOperator(
        (dimension, dimension), dtype=np.complex128, matvec=matvec, rmatvec=matvec
    )
    values = eigsh(
        operator,
        k=selected_count,
        which="LA",
        return_eigenvectors=False,
        tol=2e-8,
        maxiter=600,
    ).real
    values = np.sort(values)[::-1]
    if float(np.max(values, initial=-np.inf)) > 5e-9:
        raise FloatingPointError("homogeneous Lindblad generator has a positive mode")
    return values, {
        "method": "matrix-free Hermitian Lanczos on the q=0 covariance sector",
        "q_sector": 0,
        "requested_count": int(count),
        "tolerance": 2e-8,
        "maxiter": 600,
    }


def run_lindblad_case(
    case: dict[str, Any], config: dict[str, Any]
) -> LindbladAdapterResult:
    """Run one deterministic continuous case through time ``2*Ny``."""

    started = time.perf_counter()
    dynamics = case["dynamics"]
    if dynamics["family"] != "lindblad":
        raise ValueError("run_lindblad_case received a non-Lindblad case")
    if not bool(dynamics["perfect_correction"]):
        raise ValueError("the continuous campaign is perfect-correction only")
    if dynamics["init_mode"] != "maxmix":
        raise ValueError("the continuous campaign requires maxmix initialization")
    if len(dynamics["sample_ids"]) != 1:
        raise ValueError("a deterministic Lindblad case must have one declared sample")

    model = build_model(case)
    generator = PerfectCorrectionLindblad.from_canonical_model(model)
    nx, ny = int(model.Nx), int(model.Ny)
    block_dimension = 2 * nx
    full_dimension = block_dimension * ny
    cycles = int(dynamics["cycles"])
    if cycles != 2 * ny or not math.isclose(
        float(dynamics["physical_time"]), float(cycles), rel_tol=0.0, abs_tol=1e-13
    ):
        raise ValueError("continuous physical time and cycles must both equal 2*Ny")
    include_number_dephasing = bool(dynamics["dephasing"])
    dt = float(config["dynamics"]["lindblad"]["dt"])
    steps_per_cycle = int(round(1.0 / dt))
    if steps_per_cycle < 1 or not math.isclose(
        steps_per_cycle * dt, 1.0, rel_tol=1e-12, abs_tol=1e-13
    ):
        raise ValueError("the configured Lindblad dt must divide one cycle exactly")

    checkpoint_cycle = np.asarray(spectral_checkpoint_cycles(case), dtype=np.int64)
    checkpoint_lookup = {
        int(cycle): index for index, cycle in enumerate(checkpoint_cycle.tolist())
    }
    checkpoint_count = checkpoint_cycle.size
    occupation_min = np.empty((1, checkpoint_count), dtype=np.float64)
    occupation_max = np.empty_like(occupation_min)
    half_gap = np.empty_like(occupation_min)
    entropy = np.empty_like(occupation_min)
    variance_proxy = np.empty_like(occupation_min)
    translation = np.zeros((1, cycles + 1), dtype=np.float64)
    successive = np.zeros_like(translation)

    late_start, late_end = late_cycle_bounds(case)
    cycle_coordinate = np.arange(cycles + 1, dtype=np.float64)
    canonical_run = model.run_lindblad_evolution(
        cycles=float(cycles),
        dt=dt,
        init_mode=dynamics["init_mode"],
        include_number_dephasing=include_number_dephasing,
        observation_times=cycle_coordinate,
        representation="q0",
    )
    observed_times = np.asarray(canonical_run["times"], dtype=np.float64)
    canonical_run_config = dict(canonical_run["run_config"])
    if not np.array_equal(observed_times, cycle_coordinate):
        raise RuntimeError("canonical Lindblad entry point returned the wrong cycle grid")
    history = np.asarray(
        canonical_run["correlation_history"], dtype=np.complex128
    )
    expected_history_shape = (
        cycles + 1,
        ny,
        block_dimension,
        block_dimension,
    )
    if history.shape != expected_history_shape:
        raise RuntimeError(
            "canonical Lindblad q0 history has shape "
            f"{history.shape}, expected {expected_history_shape}"
        )
    cycle_differences = history[1:] - history[:-1]
    successive[0, 1:] = np.sqrt(
        np.sum(np.abs(cycle_differences) ** 2, axis=(-3, -2, -1))
    ) / math.sqrt(ny * block_dimension)
    for cycle, checkpoint_index in checkpoint_lookup.items():
        values = _spectral_scalars(history[cycle], ny=ny)
        occupation_min[0, checkpoint_index] = values[0]
        occupation_max[0, checkpoint_index] = values[1]
        half_gap[0, checkpoint_index] = values[2]
        entropy[0, checkpoint_index] = values[3]
        variance_proxy[0, checkpoint_index] = values[4]
    expected_late_count = late_end - late_start + 1
    late_history = history[late_start : late_end + 1]
    if late_history.shape[0] != expected_late_count:
        raise RuntimeError("continuous late-cycle average is incomplete")
    blocks = history[-1].copy()
    late_blocks = np.mean(late_history, axis=0)
    del late_history, history, canonical_run
    final_dense = generator.q_sector_to_dense(blocks, q_index=0)
    late_dense = generator.q_sector_to_dense(late_blocks, q_index=0)
    final_dense = 0.5 * (final_dense + final_dense.conj().T)
    late_dense = 0.5 * (late_dense + late_dense.conj().T)
    if final_dense.shape != (full_dimension, full_dimension):
        raise RuntimeError("q=0 reconstruction returned the wrong dense dimension")

    tolerance = float(config["analysis"]["physicality_tolerance"])
    minimum = float(np.min(occupation_min))
    maximum = float(np.max(occupation_max))
    violation = max(0.0, -minimum, maximum - 1.0)
    if violation > tolerance:
        raise FloatingPointError(
            f"RK4 correlation spectrum violates [0,1] by {violation:.3e}"
        )
    stationarity = float(
        np.linalg.norm(
            generator.q_sector_rhs(
                late_blocks,
                q_index=0,
                include_number_dephasing=include_number_dephasing,
            )
        )
        / math.sqrt(ny * block_dimension)
    )
    leading, relaxation_metadata = _leading_relaxation_spectrum(
        generator,
        include_number_dephasing=include_number_dephasing,
    )

    arrays: dict[str, np.ndarray] = {
        "cycle": np.arange(cycles + 1, dtype=np.int64),
        "translation_residual": translation,
        "successive_state_distance": successive,
        "spectral_checkpoint_cycle": checkpoint_cycle,
        "occupation_min": occupation_min,
        "occupation_max": occupation_max,
        "half_occupation_gap": half_gap,
        "gaussian_entropy_proxy_per_circumference": entropy,
        "gaussian_charge_variance_proxy": variance_proxy,
        "G_final": final_dense[None],
        "G_late_cycle_average": late_dense[None],
        "leading_relaxation_spectrum": leading[None],
    }

    response_enabled = case_requires_response(case, config)
    response_metadata: dict[str, Any] = {}
    if response_enabled:
        response_config = config["response"]
        output_dt = float(response_config["lindblad_output_dt"])
        horizon = 0.5 * ny
        output_count = int(round(horizon / output_dt))
        if not math.isclose(
            output_count * output_dt, horizon, rel_tol=1e-12, abs_tol=1e-13
        ):
            raise ValueError("Lindblad response output grid does not end at Ny/2")
        response = local_density_kick_response(
            generator,
            late_blocks,
            walls=case["model"]["wall_locations"],
            source_ys=response_config["lindblad_source_y"],
            epsilon=float(response_config["epsilon"]),
            times=np.arange(output_count + 1, dtype=np.float64) * output_dt,
            integration_dt=dt,
            include_number_dephasing=include_number_dephasing,
            wall_window_columns=int(response_config["wall_window_columns"]),
            fit_time_min=float(response_config["fit_time_min"]),
            fit_time_max=0.375 * ny,
            algorithm="auto",
            q_batch_size=int(response_config.get("lindblad_q_batch_size", 1)),
            epsilon_multipliers=response_config["epsilon_multipliers"],
        )
        arrays["response_times"] = response.arrays["response_times"]
        arrays["response_source_y"] = response.arrays["response_source_y"]
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
            arrays[name] = response.arrays[name][None]
        response_metadata = response.metadata

    projector_hashes = {
        name: model._checkpoint_array_signature(getattr(model, name))
        for name in ("WF_Ap", "WF_Am", "WF_Bp", "WF_Bm")
    }
    metadata = {
        "canonical_dynamics_entry_point": "classA_U1FGTN.run_lindblad_evolution",
        "adapter_entry_point": "lindblad_adapter.run_lindblad_case",
        "correlation_convention": "G_ij=Tr(rho c_i^dagger c_j)",
        "equation": (
            "dG/dt=V_minus-1/2{V_minus+V_plus,G}"
            + (
                "-1/2 sum_a[P_a,[P_a,G]]"
                if include_number_dephasing
                else ""
            )
        ),
        "perfect_correction_coefficients": {
            "gain": 1.0,
            "loss": 1.0,
            "number_dephasing": 1.0 if include_number_dephasing else 0.0,
        },
        "include_number_dephasing": include_number_dephasing,
        "two_point_closure": "exact",
        "many_body_gaussianity": (
            "not preserved by number-dephasing jumps"
            if include_number_dephasing
            else "preserved"
        ),
        "integrator": "fixed-step RK4",
        "dt": dt,
        "steps_per_cycle": steps_per_cycle,
        "physical_time": float(cycles),
        "representation": "exact y-translation-invariant q=0 sector",
        "canonical_run_config": canonical_run_config,
        "translation_residual": 0.0,
        "projector_translation_residual": float(generator.y_translation_residual),
        "wall_locations": [int(value) for value in model.DW_loc],
        "sample_seeds": [int(value) for value in dynamics["sample_seeds"]],
        "dw_interval_inclusive": True,
        "late_cycle_window_inclusive": [late_start, late_end],
        "spectral_checkpoint_cycles": checkpoint_cycle.tolist(),
        "stationarity_residual_late": stationarity,
        "physicality_violation_checkpoints": violation,
        "gaussian_charge_variance_proxy_note": (
            "tr[G(1-G)]; not the physical many-body charge variance when "
            "number dephasing is enabled"
        ),
        "response_enabled": response_enabled,
        "response": response_metadata,
        "relaxation_estimator": relaxation_metadata,
        "projector_hashes": projector_hashes,
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    return LindbladAdapterResult(arrays=arrays, metadata=metadata)


__all__ = ["LindbladAdapterResult", "run_lindblad_case"]
