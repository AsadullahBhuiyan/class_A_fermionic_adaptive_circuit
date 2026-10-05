#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fgtn.classA_U1FGTN import classA_U1FGTN
from fgtn.diagnostics import (
    CANONICAL_CPU_ENTRY_POINT,
    ChoiSpectrumRecorder,
    LyapunovSpectrumRecorder,
    SingleTrajectoryChargeRecorder,
    StateObservableRecorder,
    TrajectoryActivityRecorder,
    activity_record_frames,
    analyze_click_sequences,
    click_sequence_candidate_rows,
    analyze_activity,
    analyze_paired_response,
    build_region_masks,
    compute_static_completion,
    localize_vector_batch,
    unit_cell_reset,
    validate_covariance,
)
from fgtn.diagnostics.cpu_parallel import (
    ParallelPolicy,
    run_parallel_tasks,
)
from fgtn.diagnostics.io import (
    save_npz_atomic,
    write_csv_atomic,
    write_json_atomic,
    write_parquet_atomic,
)


FIGURE_WIDTH = 3.375
RNG_STREAMS = ("initialization", "exterior", "schedule", "dynamics")
SCGF_RELIABILITY_THRESHOLD = 0.1
CHOI_ENDPOINT_TOLERANCE = 1e-12
CHOI_SPECTRAL_TOLERANCE = 1e-10
CHOI_SINGULAR_TOLERANCE = 1e-10
COVARIANCE_TOLERANCE = 1e-10
RESPONSE_NORM_FRACTION_CUTOFF = 0.2


def configure_plotting() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["CMU Sans Serif", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "mathtext.fontset": "cm",
            "figure.dpi": 150,
            "savefig.dpi": 300,
        }
    )


def save_figure(fig: plt.Figure, directory: Path, stem: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(directory / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(directory / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def parse_nshell(value: str) -> float | int | None:
    if str(value).strip().lower() in ("none", "full"):
        return None
    number = float(value)
    return int(number) if number.is_integer() else number


def nshell_tag(value: float | int | None) -> str:
    return "none" if value is None else str(value).replace(".", "p")


def geometry_values(label: str) -> tuple[str, ...]:
    return ("uniform", "dw") if label == "both" else (label,)


def model_for(*, nx: int, ny: int, geometry: str, nshell: float | int | None) -> classA_U1FGTN:
    is_dw = geometry == "dw"
    legacy_half_width = max(1, int(nx) // 3)
    legacy_dw_interval = (
        max(0, int(nx) // 2 - legacy_half_width),
        min(int(nx), int(nx) // 2 + legacy_half_width + 1) - 1,
    )
    model = classA_U1FGTN(
        nx,
        ny,
        DW=is_dw,
        nshell=nshell,
        alpha_1=1,
        alpha_2=30,
        trial_orbitals="X",
        dw_truncation=is_dw,
        # Preserve this completed precursor's recorded one-third-width slab;
        # new implicit CPU defaults follow the GPU-production Nx//4 rule.
        dw_interval=legacy_dw_interval if is_dw else None,
    )
    model.construct_OW_projectors(
        nshell=nshell,
        DW=is_dw,
        trial_orbitals="X",
        dw_truncation=is_dw,
    )
    return model


def git_metadata() -> dict[str, Any]:
    def run(*command: str) -> str:
        result = subprocess.run(command, cwd=REPO_ROOT, text=True, capture_output=True, check=False)
        return result.stdout.strip()

    return {
        "commit": run("git", "rev-parse", "HEAD"),
        "dirty": bool(run("git", "status", "--short")),
    }


def manifest(
    config: dict[str, Any],
    *,
    sample_seeds: list[int | None] | None = None,
    active_indices: np.ndarray | None = None,
) -> dict[str, Any]:
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "canonical_dynamics_entry_point": CANONICAL_CPU_ENTRY_POINT,
        "config": config,
        "sample_seeds": sample_seeds,
        "active_top_layer_indices": (
            [] if active_indices is None else np.asarray(active_indices, dtype=np.int64)
        ),
        "git": git_metadata(),
    }


def case_directory(root: Path, diagnostic: str, *, nx: int, ny: int, geometry: str, nshell: Any, suffix: str = "") -> Path:
    name = f"N{nx}x{ny}_{geometry}_dwtrunc{int(geometry == 'dw')}_nsh{nshell_tag(nshell)}"
    if suffix:
        name += f"_{suffix}"
    path = root / diagnostic / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def dimensions(args: argparse.Namespace, diagnostic: str) -> tuple[int, tuple[int, ...]]:
    if args.smoke:
        return (args.nx or 4), tuple(args.ny or (6,))
    if diagnostic == "spectral":
        return (args.nx or 8), tuple(args.ny or (8, 12, 16))
    return (args.nx or 12), tuple(args.ny or (16, 24, 32))


def nshell_values(args: argparse.Namespace, diagnostic: str) -> tuple[float | int | None, ...]:
    if args.nshell is not None:
        values = tuple(parse_nshell(value) for value in args.nshell)
    else:
        values = (1,)
    if diagnostic != "static" and any(value is None for value in values):
        raise ValueError("nshell=None is reserved for static diagnostics in the CPU-first suite.")
    return values


def parse_auto_positive(value: str) -> int | None:
    if str(value).strip().lower() == "auto":
        return None
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("Expected a positive integer or 'auto'.")
    return parsed


def parallel_policy(args: argparse.Namespace) -> ParallelPolicy:
    return ParallelPolicy(
        cpu_budget=int(args.cpu_budget),
        workers=args.workers,
        threads_per_worker=args.threads_per_worker,
        enabled=not bool(args.no_parallel),
        memory_fraction=float(args.memory_fraction),
    )


def estimate_worker_bytes(nx: int, ny: int, *, matrix_factor: float) -> int:
    dimension = 2 * int(nx) * int(ny)
    return int(max(1.0, float(matrix_factor)) * dimension * dimension * np.dtype(np.complex128).itemsize)


def compact_parallel_execution(metadata: dict[str, Any]) -> dict[str, Any]:
    """Keep run-level resource decisions in tables without repeating task logs."""
    return {key: value for key, value in metadata.items() if key != "task_telemetry"}


def finite_mean(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(np.mean(finite)) if finite.size else np.nan


def latest_finite_by_row(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(values, dtype=np.float64)
    latest = np.full((array.shape[0],), np.nan, dtype=np.float64)
    cycles = np.full((array.shape[0],), -1, dtype=np.int64)
    for sample, row in enumerate(array):
        indices = np.flatnonzero(np.isfinite(row))
        if indices.size:
            cycles[sample] = int(indices[-1]) + 1
            latest[sample] = row[indices[-1]]
    return latest, cycles


def sample_root_seeds(root_seed: int, samples: int) -> list[int]:
    return [
        int(sequence.generate_state(1, dtype=np.uint64)[0])
        for sequence in np.random.SeedSequence(int(root_seed)).spawn(int(samples))
    ]


def response_seed_records(root_seed: int, samples: int, wall_count: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for sample, sample_sequence in enumerate(np.random.SeedSequence(int(root_seed)).spawn(int(samples))):
        streams = sample_sequence.spawn(1 + int(wall_count))
        records.append(
            {
                "sample": sample,
                "equilibration_seed": int(streams[0].generate_state(1, dtype=np.uint64)[0]),
                "response_seeds": [
                    int(stream.generate_state(1, dtype=np.uint64)[0]) for stream in streams[1:]
                ],
            }
        )
    return records


def _case_key(task: dict[str, Any]) -> tuple[Any, ...]:
    return (
        task.get("nshell"),
        str(task["geometry"]),
        int(task["ny"]),
        str(task.get("protocol", "")),
        str(task.get("sequence", "")),
    )


def _worker_model(task: dict[str, Any]) -> classA_U1FGTN:
    source = task.get("model")
    if source is not None:
        return source._spawn_for_parallel()
    return model_for(
        nx=int(task["nx"]),
        ny=int(task["ny"]),
        geometry=str(task["geometry"]),
        nshell=task["nshell"],
    )


def _static_case_worker(task: dict[str, Any]) -> dict[str, Any]:
    model = _worker_model(task)
    active_indices = model.active_top_layer_indices(meas_slab_only=True)
    result = compute_static_completion(model, rtol=float(task["svd_rtol"]))
    return {
        "task": task,
        "active_indices": active_indices,
        "payload": result.payload(),
        "metrics": result.summary(),
    }


_ACTIVITY_FIELD_MAP = {
    "defect_X": "defect",
    "transfer_Y": "transfer",
    "success_probability": "success_probability",
    "valid": "valid",
    "forced_site": "forced_site",
    "visit_order": "visit_order",
}
_STATE_FIELD_MAP = {
    "state_total_charge": "total_charge",
    "state_charge_variance": "charge_variance",
    "state_entropy": "entropy",
    "state_purity_defect": "purity_defect",
    "state_successive_delta": "successive_delta",
}


def _activity_sample_worker(task: dict[str, Any]) -> dict[str, Any]:
    model = _worker_model(task)
    active_indices = model.active_top_layer_indices(meas_slab_only=True)
    recorder = TrajectoryActivityRecorder.from_model(model, cycles=int(task["cycles"]), samples=1)
    state = StateObservableRecorder(samples=1, cycles=int(task["cycles"]), active_indices=active_indices)
    protocol = str(task["protocol"])
    result = model.run_markov_circuit(
        G_history=False,
        progress=False,
        cycles=int(task["cycles"]),
        samples=1,
        init_mode=str(task["init_mode"]),
        save=False,
        perfect_correction=protocol == "perfect_correction",
        p_gain=None if protocol == "perfect_correction" else 0.5,
        p_loss=None if protocol == "perfect_correction" else 0.5,
        sequence=str(task["sequence"]),
        meas_slab_only=True,
        random_seed=int(task["worker_seed"]),
        cycle_observer=state,
        trajectory_weight_observer=recorder,
        parallelize_samples=False,
    )
    recorder.assert_complete()
    return {
        "task": task,
        "activity": recorder.payload(),
        "state": state.payload(),
        "active_indices": active_indices,
        "dynamics_sample_seed": result.get("sample_seeds", [None])[0],
        "rng_streams": result.get("rng_streams", RNG_STREAMS),
        "seed_derivation": result.get("seed_derivation"),
    }


def _merge_activity_results(
    model: classA_U1FGTN,
    results: list[dict[str, Any]],
    *,
    samples: int,
    cycles: int,
) -> tuple[TrajectoryActivityRecorder, StateObservableRecorder, list[int | None]]:
    active_indices = model.active_top_layer_indices(meas_slab_only=True)
    recorder = TrajectoryActivityRecorder.from_model(model, cycles=cycles, samples=samples)
    state = StateObservableRecorder(samples=samples, cycles=cycles, active_indices=active_indices)
    dynamics_seeds: list[int | None] = [None] * samples
    for result in sorted(results, key=lambda item: int(item["task"]["sample"])):
        sample = int(result["task"]["sample"])
        np.testing.assert_array_equal(result["active_indices"], active_indices)
        np.testing.assert_array_equal(result["activity"]["site_ids"], recorder.site_ids)
        for payload_name, attribute_name in _ACTIVITY_FIELD_MAP.items():
            getattr(recorder, attribute_name)[sample] = np.asarray(result["activity"][payload_name])[0]
        for payload_name, attribute_name in _STATE_FIELD_MAP.items():
            getattr(state, attribute_name)[sample] = np.asarray(result["state"][payload_name])[0]
        dynamics_seeds[sample] = result["dynamics_sample_seed"]
    recorder.assert_complete()
    return recorder, state, dynamics_seeds


def _spectral_sample_worker(task: dict[str, Any]) -> dict[str, Any]:
    nx, ny = int(task["nx"]), int(task["ny"])
    model = _worker_model(task)
    active_indices = model.active_top_layer_indices(meas_slab_only=True)
    dimension = int(active_indices.size)
    nvec = min(dimension, int(task["nvec"]))
    cycles = int(task["cycles"])
    protocol = str(task["protocol"])
    lyapunov = LyapunovSpectrumRecorder(
        samples=1,
        cycles=cycles,
        nvec=nvec,
        vector_dimension=dimension,
    )
    state = StateObservableRecorder(samples=1, cycles=cycles, active_indices=active_indices)
    lyapunov_result = model.run_markov_circuit(
        G_history=False,
        progress=False,
        cycles=cycles,
        samples=1,
        init_mode=str(task["init_mode"]),
        save=False,
        postselect=protocol == "postselect",
        perfect_correction=protocol == "perfect_correction",
        sequence=str(task["sequence"]),
        meas_slab_only=True,
        random_seed=int(task["worker_seed"]),
        cycle_observer=state,
        lyapunov_observer=lyapunov,
        lyapunov_nvec=nvec,
        parallelize_samples=False,
    )

    observer_cycles = tuple(range(1, cycles + 1))
    choi_model = model_for(nx=nx, ny=ny, geometry=str(task["geometry"]), nshell=task["nshell"])
    choi = ChoiSpectrumRecorder(
        samples=1,
        cycles=observer_cycles,
        dimension=dimension,
        n_eigenstates=3,
        endpoint_tol=CHOI_ENDPOINT_TOLERANCE,
        spectral_tol=CHOI_SPECTRAL_TOLERANCE,
    )
    choi_result = choi_model.run_markov_circuit(
        G_history=False,
        progress=False,
        cycles=cycles,
        samples=1,
        init_mode=str(task["init_mode"]),
        save=False,
        postselect=protocol == "postselect",
        perfect_correction=protocol == "perfect_correction",
        sequence=str(task["sequence"]),
        meas_slab_only=True,
        random_seed=int(task["worker_seed"]),
        track_choi=True,
        choi_observer=choi,
        choi_observer_cycles=observer_cycles,
        choi_singular_tol=CHOI_SINGULAR_TOLERANCE,
        parallelize_samples=False,
    )
    return {
        "task": task,
        "active_indices": active_indices,
        "lyapunov": lyapunov.payload(),
        "choi": choi.payload(),
        "state": state.payload(),
        "lyapunov_sample_seed": lyapunov_result.get("sample_seeds", [None])[0],
        "choi_sample_seed": choi_result.get("sample_seeds", [None])[0],
        "rng_streams": lyapunov_result.get("rng_streams", RNG_STREAMS),
        "seed_derivation": lyapunov_result.get("seed_derivation"),
        "choi_failure_records": choi.failure_records,
    }


def _merge_spectral_results(
    results: list[dict[str, Any]],
    *,
    samples: int,
    cycles: int,
    dimension: int,
    nvec: int,
) -> tuple[
    LyapunovSpectrumRecorder,
    ChoiSpectrumRecorder,
    StateObservableRecorder,
    list[int | None],
    list[int | None],
]:
    active_indices = np.asarray(results[0]["active_indices"], dtype=np.int64)
    lyapunov = LyapunovSpectrumRecorder(
        samples=samples,
        cycles=cycles,
        nvec=nvec,
        vector_dimension=dimension,
    )
    choi = ChoiSpectrumRecorder(
        samples=samples,
        cycles=tuple(range(1, cycles + 1)),
        dimension=dimension,
        n_eigenstates=3,
        endpoint_tol=CHOI_ENDPOINT_TOLERANCE,
        spectral_tol=CHOI_SPECTRAL_TOLERANCE,
    )
    state = StateObservableRecorder(samples=samples, cycles=cycles, active_indices=active_indices)
    lyapunov_seeds: list[int | None] = [None] * samples
    choi_seeds: list[int | None] = [None] * samples
    for result in sorted(results, key=lambda item: int(item["task"]["sample"])):
        sample = int(result["task"]["sample"])
        np.testing.assert_array_equal(result["active_indices"], active_indices)
        lyapunov.spectrum[sample] = result["lyapunov"]["lyapunov_spectrum"][0]
        lyapunov.gap[sample] = result["lyapunov"]["lyapunov_gap"][0]
        lyapunov.final_vector[sample] = result["lyapunov"]["lyapunov_final_vector"][0]
        lyapunov.final_value[sample] = result["lyapunov"]["lyapunov_final_value"][0]
        lyapunov.final_index[sample] = result["lyapunov"]["lyapunov_final_index"][0]
        lyapunov.null_count[sample] = result["lyapunov"]["lyapunov_null_count"][0]
        choi.gap[sample] = result["choi"]["choi_gap"][0]
        choi.spectrum[sample] = result["choi"]["choi_spectrum"][0]
        choi.near_gap_exponents[sample] = result["choi"]["choi_near_gap_exponents"][0]
        choi.near_gap_a_eigenvalues[sample] = result["choi"]["choi_near_gap_a_eigenvalues"][0]
        choi.near_gap_residuals[sample] = result["choi"]["choi_near_gap_residuals"][0]
        choi.finite_count[sample] = result["choi"]["choi_finite_count"][0]
        choi.zero_count[sample] = result["choi"]["choi_zero_count"][0]
        choi.pole_count[sample] = result["choi"]["choi_pole_count"][0]
        choi.active[sample] = result["choi"]["choi_active"][0]
        choi.final_eigenvectors[sample] = result["choi"]["choi_final_eigenvectors"][0]
        choi.final_eigenvector_cycle[sample] = result["choi"]["choi_final_eigenvector_cycle"][0]
        choi.active_indices = active_indices.copy()
        for payload_name, attribute_name in _STATE_FIELD_MAP.items():
            getattr(state, attribute_name)[sample] = np.asarray(result["state"][payload_name])[0]
        for failure in result["choi_failure_records"]:
            choi.failure_records.append({**dict(failure), "suite_sample_index": sample})
        lyapunov_seeds[sample] = result["lyapunov_sample_seed"]
        choi_seeds[sample] = result["choi_sample_seed"]
    return lyapunov, choi, state, lyapunov_seeds, choi_seeds


def _response_analysis_worker(task: dict[str, Any]) -> dict[str, Any]:
    return analyze_paired_response(
        np.asarray(task["delta"], dtype=np.float64),
        regions=task["regions"],
        y0=int(task["y0"]),
        wall_x=tuple(int(value) for value in task["walls"]),
        norm_fraction_cutoff=float(task["norm_fraction_cutoff"]),
    )


def _response_sample_worker(task: dict[str, Any]) -> dict[str, Any]:
    nx, ny = int(task["nx"]), int(task["ny"])
    model = _worker_model(task)
    active_indices = model.active_top_layer_indices(meas_slab_only=True)
    walls = tuple(int(value) for value in task["walls"])
    y0 = int(task["y0"])
    response_cycles = int(task["response_cycles"])
    equilibrium_result = model.run_markov_circuit(
        G_history=False,
        progress=False,
        cycles=int(task["equilibration_cycles"]),
        samples=1,
        init_mode=str(task["init_mode"]),
        save=False,
        perfect_correction=True,
        sequence=str(task["sequence"]),
        meas_slab_only=True,
        random_seed=int(task["equilibration_seed"]),
        parallelize_samples=False,
    )
    equilibrium = equilibrium_result["G_final"][0]
    delta = np.full((len(walls), response_cycles + 1, nx, ny), np.nan, dtype=np.float64)
    covariance_residuals: list[dict[str, float]] = []
    schedule_identical: list[bool] = []
    dynamics_response_seeds: list[int | None] = []
    for wall_index, wall in enumerate(walls):
        plus = unit_cell_reset(equilibrium, nx=nx, ny=ny, x=wall, y=y0, occupied=True)
        minus = unit_cell_reset(equilibrium, nx=nx, ny=ny, x=wall, y=y0, occupied=False)
        covariance_residuals.extend(
            (
                validate_covariance(plus, tolerance=COVARIANCE_TOLERANCE),
                validate_covariance(minus, tolerance=COVARIANCE_TOLERANCE),
            )
        )
        plus_charge = SingleTrajectoryChargeRecorder(samples=1, cycles=response_cycles, nx=nx, ny=ny)
        minus_charge = SingleTrajectoryChargeRecorder(samples=1, cycles=response_cycles, nx=nx, ny=ny)
        plus_activity = TrajectoryActivityRecorder.from_model(model, cycles=response_cycles, samples=1)
        minus_activity = TrajectoryActivityRecorder.from_model(model, cycles=response_cycles, samples=1)
        common = dict(
            G_history=False,
            progress=False,
            cycles=response_cycles,
            samples=1,
            save=False,
            perfect_correction=True,
            sequence=str(task["sequence"]),
            meas_slab_only=True,
            random_seed=int(task["response_seeds"][wall_index]),
            parallelize_samples=False,
        )
        plus_result = model.run_markov_circuit(
            G_init=plus,
            cycle_observer=plus_charge,
            trajectory_weight_observer=plus_activity,
            **common,
        )
        minus_result = model.run_markov_circuit(
            G_init=minus,
            cycle_observer=minus_charge,
            trajectory_weight_observer=minus_activity,
            **common,
        )
        plus_activity.assert_complete()
        minus_activity.assert_complete()
        schedule_identical.append(bool(np.array_equal(plus_activity.visit_order, minus_activity.visit_order)))
        if plus_result.get("sample_seeds") != minus_result.get("sample_seeds"):
            raise RuntimeError("Paired response runs derived different canonical sample seeds.")
        dynamics_response_seeds.append(plus_result.get("sample_seeds", [None])[0])
        delta[wall_index] = 0.5 * (plus_charge.charge[0] - minus_charge.charge[0])
    return {
        "task": task,
        "active_indices": active_indices,
        "delta": delta,
        "equilibration_dynamics_seed": equilibrium_result.get("sample_seeds", [None])[0],
        "response_dynamics_seeds": dynamics_response_seeds,
        "covariance_residuals": covariance_residuals,
        "schedule_identical": schedule_identical,
    }


def _analyze_run_worker(task: dict[str, Any]) -> list[dict[str, str]]:
    run_directory = Path(task["run_directory"])
    configure_plotting()
    generated: list[dict[str, str]] = []
    if (run_directory / "static_completion.npz").exists():
        with np.load(run_directory / "static_completion.npz") as loaded:
            plot_static(dict(loaded), run_directory)
        generated.append({"type": "static", "directory": str(run_directory)})
    if (run_directory / "activity_analysis.npz").exists():
        with np.load(run_directory / "activity_analysis.npz") as loaded:
            plot_activity(dict(loaded), run_directory)
        generated.append({"type": "activity", "directory": str(run_directory)})
    if (run_directory / "spectral_diagnostics.npz").exists():
        with np.load(run_directory / "spectral_diagnostics.npz") as loaded:
            plot_spectral(dict(loaded), run_directory)
        generated.append({"type": "spectral", "directory": str(run_directory)})
    if (run_directory / "response_analysis.npz").exists():
        with np.load(run_directory / "response_analysis.npz") as loaded:
            plot_response(dict(loaded), run_directory)
        generated.append({"type": "response", "directory": str(run_directory)})
    return generated


def plot_static(data: dict[str, np.ndarray], output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(2 * FIGURE_WIDTH, 2.45))
    principal = np.asarray(data["principal_cosines"])
    axes[0].plot(np.arange(principal.size), principal, marker="o", ms=2, lw=0.8)
    axes[0].set(xlabel="principal-angle index", ylabel=r"$\cos\vartheta_j$", title="Controller-span overlap")
    residual = np.asarray(data["residual_map"])
    finite_count = np.sum(np.isfinite(residual), axis=(1, 2))
    profile = np.divide(
        np.nansum(residual, axis=(1, 2)),
        finite_count,
        out=np.full((residual.shape[0],), np.nan, dtype=np.float64),
        where=finite_count > 0,
    )
    axes[1].plot(np.arange(profile.size), profile, marker="o", ms=2, lw=0.8, color="#b24745")
    axes[1].set(xlabel=r"$x$", ylabel=r"$\overline{f_\star(x)}$", title="Optimal residual profile")
    save_figure(fig, output / "figures", "static_completion")


def run_static(args: argparse.Namespace) -> list[dict[str, Any]]:
    nx, ny_values = dimensions(args, "static")
    tasks = []
    for nshell in nshell_values(args, "static"):
        for geometry in geometry_values(args.geometry):
            for ny in ny_values:
                tasks.append(
                    {
                        "nx": nx,
                        "ny": ny,
                        "geometry": geometry,
                        "nshell": nshell,
                        "svd_rtol": args.svd_rtol,
                        "model": model_for(nx=nx, ny=ny, geometry=geometry, nshell=nshell),
                    }
                )
    results, parallel = run_parallel_tasks(
        _static_case_worker,
        tasks,
        policy=parallel_policy(args),
        single_thread_tasks=False,
        estimated_bytes_per_worker=max(
            estimate_worker_bytes(nx, ny, matrix_factor=10.0) for ny in ny_values
        ),
        quiet=True,
    )
    summaries: list[dict[str, Any]] = []
    for item in sorted(results, key=lambda result: _case_key(result["task"])):
        task = item["task"]
        output = case_directory(
            args.output_dir,
            "static",
            nx=task["nx"],
            ny=task["ny"],
            geometry=task["geometry"],
            nshell=task["nshell"],
        )
        config = {
            "diagnostic": "static",
            "Nx": task["nx"],
            "Ny": task["ny"],
            "geometry": task["geometry"],
            "DW": task["geometry"] == "dw",
            "dw_truncation": task["geometry"] == "dw",
            "meas_slab_only": True,
            "alpha_1": 1,
            "alpha_2": 30,
            "nshell": task["nshell"],
            "svd_rtol": task["svd_rtol"],
            "parallel_execution": compact_parallel_execution(parallel),
        }
        summary = {**config, **item["metrics"]}
        save_npz_atomic(output / "static_completion.npz", **item["payload"])
        write_json_atomic(
            output / "run_summary.json",
            {
                **manifest(config, active_indices=item["active_indices"]),
                "parallel_execution": parallel,
                "metrics": item["metrics"],
            },
        )
        write_csv_atomic(output / "scalar_metrics.csv", [summary])
        plot_static(item["payload"], output)
        summaries.append(summary)
    return summaries


def plot_activity(data: dict[str, np.ndarray], output: Path) -> None:
    activity_names = [str(value) for value in data["activity_names"]]
    region_names = [str(value) for value in data["region_names"]]
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, 2.45))
    for kind, label in enumerate(activity_names):
        ax.plot(data["spatial_profile_x"][kind, :, -1], marker="o", ms=2, lw=0.9, label=label)
    ax.set(xlabel=r"$x$", ylabel="event probability", title="Post-burn-in activity profile")
    ax.legend(frameon=False)
    save_figure(fig, output / "figures", "activity_profile")

    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, 2.45))
    final_window = data["theta"].shape[1] - 1
    for region in ("all", "interface", "interior"):
        index = region_names.index(region)
        values = data["theta"][0, final_window, index]
        if np.any(np.isfinite(values)):
            ax.plot(data["s_grid"], values, lw=0.9, label=region)
    ax.axhline(0.0, color="black", lw=0.5)
    ax.set(xlabel=r"counting field $s$", ylabel=r"$\theta_X(s,T)$", title="Empirical defect SCGF")
    ax.legend(frameon=False)
    save_figure(fig, output / "figures", "activity_scgf")


def run_activity(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.protocol not in (None, "perfect_correction", "imperfect"):
        raise ValueError("Activity mode supports perfect_correction or imperfect feedback.")
    protocol = args.protocol or "perfect_correction"
    nx, ny_values = dimensions(args, "activity")
    summaries: list[dict[str, Any]] = []
    geometries = geometry_values(args.geometry)
    for nshell in nshell_values(args, "activity"):
        for ny in ny_values:
            samples = args.samples or (2 if args.smoke else 64)
            cycles = args.cycles or (2 if args.smoke else 4 * ny)
            burn_in = args.burn_in if args.burn_in is not None else (1 if args.smoke else 2 * ny)
            sequence = args.sequence or "raster_y"
            worker_seeds = sample_root_seeds(args.seed, samples)
            models = {
                geometry: model_for(nx=nx, ny=ny, geometry=geometry, nshell=nshell)
                for geometry in geometries
            }
            tasks = [
                {
                    "sample": sample,
                    "nx": nx,
                    "ny": ny,
                    "geometry": geometry,
                    "nshell": nshell,
                    "cycles": cycles,
                    "protocol": protocol,
                    "sequence": sequence,
                    "init_mode": args.init_mode,
                    "worker_seed": worker_seeds[sample],
                    "model": models[geometry],
                }
                for geometry in geometries
                for sample in range(samples)
            ]
            all_results, execution = run_parallel_tasks(
                _activity_sample_worker,
                tasks,
                policy=parallel_policy(args),
                single_thread_tasks=True,
                estimated_bytes_per_worker=estimate_worker_bytes(nx, ny, matrix_factor=12.0),
                quiet=not args.progress,
            )
            for geometry in geometries:
                results = [
                    result for result in all_results if result["task"]["geometry"] == geometry
                ]
                model = models[geometry]
                active_indices = model.active_top_layer_indices(meas_slab_only=True)
                regions = build_region_masks(model, interface_width=args.interface_width)
                recorder, state, dynamics_seeds = _merge_activity_results(
                    model,
                    results,
                    samples=samples,
                    cycles=cycles,
                )
                analysis_jobs = 1 if args.no_parallel else max(
                    1,
                    min(execution["effective_cpu_budget"], 30),
                )
                analysis = analyze_activity(
                    recorder,
                    regions=regions,
                    burn_in=burn_in,
                    bootstrap_samples=args.bootstrap_samples,
                    bootstrap_seed=args.seed,
                    reliability_threshold=SCGF_RELIABILITY_THRESHOLD,
                    parallel_jobs=analysis_jobs,
                )
                output = case_directory(
                    args.output_dir,
                    "activity",
                    nx=nx,
                    ny=ny,
                    geometry=geometry,
                    nshell=nshell,
                    suffix=protocol,
                )
                save_npz_atomic(
                    output / "activity_raw.npz",
                    **recorder.payload(),
                    **state.payload(),
                    **regions.payload(),
                )
                analysis_payload = analysis.payload()
                save_npz_atomic(output / "activity_analysis.npz", **analysis_payload)

                sequence_analysis = None
                sequence_rows: list[dict[str, Any]] = []
                if args.click_sequences:
                    sequence_seed = int(args.seed) + 1_000_003
                    sequence_analysis = analyze_click_sequences(
                        recorder,
                        regions=regions,
                        burn_in=burn_in,
                        permutations=args.sequence_permutations,
                        bootstrap_samples=args.sequence_bootstraps,
                        minimum_support=args.sequence_min_support,
                        seed=sequence_seed,
                    )
                    save_npz_atomic(
                        output / "click_sequence_analysis.npz",
                        **sequence_analysis.payload(),
                    )
                    event_frame, motif_frame = activity_record_frames(
                        recorder,
                        regions=regions,
                        schedule=sequence,
                        trial_pauli="X",
                    )
                    write_parquet_atomic(output / "click_events.parquet", event_frame)
                    write_parquet_atomic(output / "unit_cell_motifs.parquet", motif_frame)
                    sequence_rows = [
                        {"schedule": sequence, **row}
                        for row in click_sequence_candidate_rows(sequence_analysis)
                    ]
                    write_csv_atomic(output / "click_sequence_candidates.csv", sequence_rows)
                seed_derivation = (
                    "SeedSequence(root).spawn(samples) creates suite worker seeds shared by "
                    "geometry index; each canonical single-sample run applies "
                    "SeedSequence(worker_seed).spawn(1), then SeedSequence(sample_seed).spawn(4)"
                )
                config = {
                    "diagnostic": "activity",
                    "Nx": nx,
                    "Ny": ny,
                    "geometry": geometry,
                    "DW": geometry == "dw",
                    "dw_truncation": geometry == "dw",
                    "meas_slab_only": True,
                    "alpha_1": 1,
                    "alpha_2": 30,
                    "nshell": nshell,
                    "samples": samples,
                    "cycles": cycles,
                    "burn_in": burn_in,
                    "protocol": protocol,
                    "sequence": sequence,
                    "random_seed": args.seed,
                    "rng_streams": RNG_STREAMS,
                    "seed_derivation": seed_derivation,
                    "bootstrap_samples": args.bootstrap_samples,
                    "scgf_s_grid": analysis.s_grid,
                    "scgf_window_lengths": analysis.window_lengths,
                    "scgf_window_fractions": [0.25, 0.5, 1.0],
                    "effective_sample_fraction_threshold": SCGF_RELIABILITY_THRESHOLD,
                    "interface_width": regions.interface_width,
                    "wall_x": list(regions.wall_x),
                    "parallel_execution": compact_parallel_execution(execution),
                    "analysis_parallel_jobs": analysis_jobs,
                    "trial_orbitals": "X",
                    "click_sequences": bool(args.click_sequences),
                    "sequence_permutations": int(args.sequence_permutations),
                    "sequence_bootstraps": int(args.sequence_bootstraps),
                    "sequence_min_support": int(args.sequence_min_support),
                    "sequence_analysis_seed": (
                        int(args.seed) + 1_000_003 if args.click_sequences else None
                    ),
                }
                rows = []
                for kind_index, kind in enumerate(analysis.activity_names):
                    for region_index, region in enumerate(analysis.region_names):
                        rates = analysis.rates_by_cycle[kind_index, :, burn_in:, region_index, -1]
                        rows.append(
                            {
                                **config,
                                "activity": kind,
                                "region": region,
                                "mean_rate": float(np.nanmean(rates)) if np.any(np.isfinite(rates)) else np.nan,
                                "stationarity_slope": analysis.stationarity_slope[kind_index, region_index],
                                "final_window_mean_per_cycle": analysis.cumulants_per_cycle[kind_index, -1, region_index, 0],
                                "final_window_variance_per_cycle": analysis.cumulants_per_cycle[kind_index, -1, region_index, 1],
                            }
                        )
                metrics = {
                    "total_defects": int(np.sum(recorder.defect)),
                    "total_abs_transfer": int(np.sum(np.abs(recorder.transfer))),
                    "perfect_correction_identity": bool(np.array_equal(recorder.defect, np.abs(recorder.transfer)))
                    if protocol == "perfect_correction"
                    else None,
                    "sequence_supported_words": len(sequence_rows),
                    "sequence_candidate_words": int(sum(bool(row["candidate"]) for row in sequence_rows)),
                }
                write_json_atomic(
                    output / "run_summary.json",
                    {
                        **manifest(config, sample_seeds=dynamics_seeds, active_indices=active_indices),
                        "suite_worker_seeds": worker_seeds,
                        "parallel_execution": execution,
                        "metrics": metrics,
                    },
                )
                write_csv_atomic(output / "scalar_metrics.csv", rows)
                plot_activity(analysis_payload, output)
                summaries.append({**config, **metrics, "output": str(output)})
    return summaries


def plot_spectral(data: dict[str, np.ndarray], output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(2 * FIGURE_WIDTH, 2.45))
    lyapunov_gap = np.asarray(data["lyapunov_gap"])
    choi_gap = np.asarray(data["choi_gap"])
    axes[0].plot(np.arange(1, lyapunov_gap.shape[1] + 1), np.nanmean(lyapunov_gap, axis=0), lw=0.9)
    axes[0].set(xlabel="cycle", ylabel=r"$\min_j|\lambda_j|$", title="Lyapunov gap proxy")
    choi_count = np.sum(np.isfinite(choi_gap), axis=0)
    choi_mean = np.divide(
        np.nansum(choi_gap, axis=0),
        choi_count,
        out=np.full(choi_count.shape, np.nan, dtype=np.float64),
        where=choi_count > 0,
    )
    axes[1].plot(np.asarray(data["choi_cycles"]), choi_mean, lw=0.9, color="#b24745")
    axes[1].set(xlabel="cycle", ylabel=r"$\Delta_{\rm Choi}$", title="Regularized-Choi proxy")
    save_figure(fig, output / "figures", "spectral_gaps")


def run_spectral(args: argparse.Namespace) -> list[dict[str, Any]]:
    protocols = ("perfect_correction", "postselect") if args.protocol in (None, "both") else (args.protocol,)
    if any(protocol not in ("perfect_correction", "postselect") for protocol in protocols):
        raise ValueError("Spectral mode supports perfect_correction, postselect, or both.")
    nx, ny_values = dimensions(args, "spectral")
    summaries: list[dict[str, Any]] = []
    geometries = geometry_values(args.geometry)
    for nshell in nshell_values(args, "spectral"):
        for ny in ny_values:
            cycles = args.cycles or (2 if args.smoke else 2 * ny)
            requested_samples = args.samples or (2 if args.smoke else 8)
            sequence = args.sequence or "raster_y"
            for protocol in protocols:
                samples = 1 if protocol == "postselect" else requested_samples
                worker_seeds = sample_root_seeds(args.seed, samples)
                contexts: dict[str, dict[str, Any]] = {}
                tasks: list[dict[str, Any]] = []
                for geometry in geometries:
                    model = model_for(nx=nx, ny=ny, geometry=geometry, nshell=nshell)
                    active_indices = model.active_top_layer_indices(meas_slab_only=True)
                    dimension = int(active_indices.size)
                    nvec = dimension if args.lyapunov_nvec is None else min(
                        dimension,
                        int(args.lyapunov_nvec),
                    )
                    contexts[geometry] = {
                        "model": model,
                        "active_indices": active_indices,
                        "dimension": dimension,
                        "nvec": nvec,
                        "regions": build_region_masks(model, interface_width=args.interface_width),
                    }
                    tasks.extend(
                        {
                            "sample": sample,
                            "nx": nx,
                            "ny": ny,
                            "geometry": geometry,
                            "nshell": nshell,
                            "cycles": cycles,
                            "protocol": protocol,
                            "sequence": sequence,
                            "init_mode": args.init_mode,
                            "worker_seed": worker_seeds[sample],
                            "nvec": nvec,
                            "model": model,
                        }
                        for sample in range(samples)
                    )
                all_results, execution = run_parallel_tasks(
                    _spectral_sample_worker,
                    tasks,
                    policy=parallel_policy(args),
                    single_thread_tasks=True,
                    estimated_bytes_per_worker=estimate_worker_bytes(nx, ny, matrix_factor=40.0),
                    quiet=not args.progress,
                )
                for geometry in geometries:
                    context = contexts[geometry]
                    active_indices = context["active_indices"]
                    dimension = context["dimension"]
                    nvec = context["nvec"]
                    regions = context["regions"]
                    results = [
                        result for result in all_results if result["task"]["geometry"] == geometry
                    ]
                    lyapunov, choi, state, lyapunov_seeds, choi_seeds = _merge_spectral_results(
                        results,
                        samples=samples,
                        cycles=cycles,
                        dimension=dimension,
                        nvec=nvec,
                    )
                    lyapunov_localization = localize_vector_batch(
                        lyapunov.final_vector,
                        nx=nx,
                        ny=ny,
                        regions=regions,
                        active_indices=active_indices,
                    )
                    choi_localization = localize_vector_batch(
                        choi.final_eigenvectors,
                        nx=nx,
                        ny=ny,
                        regions=regions,
                        active_indices=active_indices,
                    )
                    output = case_directory(
                        args.output_dir,
                        "spectral",
                        nx=nx,
                        ny=ny,
                        geometry=geometry,
                        nshell=nshell,
                        suffix=protocol,
                    )
                    payload = {
                        **lyapunov.payload(),
                        **choi.payload(),
                        **state.payload(),
                        **regions.payload(),
                        "lyapunov_mode_cell_weight": lyapunov_localization["cell_weight"],
                        "lyapunov_mode_x_profile": lyapunov_localization["x_profile"],
                        "lyapunov_mode_ipr": lyapunov_localization["ipr"],
                        "lyapunov_mode_region_weight": lyapunov_localization["region_weight"],
                        "choi_mode_cell_weight": choi_localization["cell_weight"],
                        "choi_mode_x_profile": choi_localization["x_profile"],
                        "choi_mode_ipr": choi_localization["ipr"],
                        "choi_mode_region_weight": choi_localization["region_weight"],
                    }
                    save_npz_atomic(output / "spectral_diagnostics.npz", **payload)
                    seed_derivation = (
                        "SeedSequence(root).spawn(samples) creates suite worker seeds shared by "
                        "geometry index; each canonical single-sample run derives its canonical seed"
                    )
                    config = {
                        "diagnostic": "spectral",
                        "Nx": nx,
                        "Ny": ny,
                        "geometry": geometry,
                        "DW": geometry == "dw",
                        "dw_truncation": geometry == "dw",
                        "meas_slab_only": True,
                        "alpha_1": 1,
                        "alpha_2": 30,
                        "nshell": nshell,
                        "samples": samples,
                        "cycles": cycles,
                        "protocol": protocol,
                        "sequence": sequence,
                        "lyapunov_nvec": nvec,
                        "random_seed": args.seed,
                        "rng_streams": RNG_STREAMS,
                        "seed_derivation": seed_derivation,
                        "choi_endpoint_tolerance": CHOI_ENDPOINT_TOLERANCE,
                        "choi_spectral_tolerance": CHOI_SPECTRAL_TOLERANCE,
                        "choi_singular_tolerance": CHOI_SINGULAR_TOLERANCE,
                        "parallel_execution": compact_parallel_execution(execution),
                    }
                    latest_choi_gap, latest_choi_cycle = latest_finite_by_row(choi.gap)
                    metrics = {
                        "lyapunov_gap_final_mean": float(np.nanmean(lyapunov.gap[:, -1])),
                        "choi_gap_final_mean": finite_mean(choi.gap[:, -1]),
                        "choi_gap_latest_finite_mean": finite_mean(latest_choi_gap),
                        "choi_latest_finite_cycle_min": int(np.min(latest_choi_cycle)),
                        "choi_latest_finite_cycle_max": int(np.max(latest_choi_cycle)),
                        "choi_final_saturated_fraction": float(np.mean(choi.finite_count[:, -1] == 0)),
                        "lyapunov_interface_weight_mean": float(
                            np.nanmean(lyapunov_localization["region_weight"][:, 0, 1])
                        ),
                        "choi_interface_weight_mean": float(
                            np.nanmean(choi_localization["region_weight"][:, 0, 1])
                        ),
                    }
                    write_json_atomic(
                        output / "run_summary.json",
                        {
                            **manifest(config, sample_seeds=lyapunov_seeds, active_indices=active_indices),
                            "suite_worker_seeds": worker_seeds,
                            "choi_sample_seeds": choi_seeds,
                            "parallel_execution": execution,
                            "metrics": metrics,
                            "choi_failure_records": choi.failure_records,
                        },
                    )
                    write_csv_atomic(output / "scalar_metrics.csv", [{**config, **metrics}])
                    plot_spectral(payload, output)
                    summaries.append({**config, **metrics, "output": str(output)})
    return summaries


def plot_response(data: dict[str, np.ndarray], output: Path) -> None:
    moment = np.asarray(data["signed_first_moment"])
    walls = np.asarray(data["wall_x"])
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, 2.45))
    for injection in range(moment.shape[1]):
        observed = min(injection, moment.shape[3] - 1)
        ax.plot(
            np.nanmean(moment[:, injection, :, observed], axis=0),
            lw=0.9,
            label=f"inject x={int(walls[injection])}",
        )
    ax.axhline(0.0, color="black", lw=0.5)
    ax.set(xlabel="response cycle", ylabel="signed first moment", title="Wall charge response")
    ax.legend(frameon=False)
    save_figure(fig, output / "figures", "paired_wall_response")


def run_response(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.protocol not in (None, "perfect_correction"):
        raise ValueError("Response mode is defined only for perfect correction.")
    nx, ny_values = dimensions(args, "response")
    summaries: list[dict[str, Any]] = []
    response_sequences = (args.sequence,) if args.sequence is not None else ("random", "raster_y")
    geometries = geometry_values(args.geometry)
    for nshell in nshell_values(args, "response"):
        for ny in ny_values:
            samples = args.samples or (1 if args.smoke else 64)
            equilibration_cycles = args.equilibration_cycles or (1 if args.smoke else 2 * ny)
            response_cycles = args.response_cycles or (2 if args.smoke else ny // 2)
            for sequence in response_sequences:
                contexts: dict[str, dict[str, Any]] = {}
                tasks: list[dict[str, Any]] = []
                for geometry in geometries:
                    model = model_for(nx=nx, ny=ny, geometry=geometry, nshell=nshell)
                    active_indices = model.active_top_layer_indices(meas_slab_only=True)
                    regions = build_region_masks(model, interface_width=args.interface_width)
                    walls = regions.wall_x if regions.wall_x else (nx // 2,)
                    y0 = ny // 2
                    seed_records = response_seed_records(args.seed, samples, len(walls))
                    contexts[geometry] = {
                        "model": model,
                        "active_indices": active_indices,
                        "regions": regions,
                        "walls": walls,
                        "y0": y0,
                        "seed_records": seed_records,
                    }
                    tasks.extend(
                        {
                            **record,
                            "nx": nx,
                            "ny": ny,
                            "geometry": geometry,
                            "nshell": nshell,
                            "walls": walls,
                            "y0": y0,
                            "equilibration_cycles": equilibration_cycles,
                            "response_cycles": response_cycles,
                            "sequence": sequence,
                            "init_mode": args.init_mode,
                            "model": model,
                        }
                        for record in seed_records
                    )
                all_results, execution = run_parallel_tasks(
                    _response_sample_worker,
                    tasks,
                    policy=parallel_policy(args),
                    single_thread_tasks=True,
                    estimated_bytes_per_worker=estimate_worker_bytes(nx, ny, matrix_factor=18.0),
                    quiet=not args.progress,
                )
                for geometry in geometries:
                    context = contexts[geometry]
                    active_indices = context["active_indices"]
                    regions = context["regions"]
                    walls = context["walls"]
                    y0 = context["y0"]
                    seed_records = context["seed_records"]
                    results = [
                        result for result in all_results if result["task"]["geometry"] == geometry
                    ]
                    delta = np.full(
                        (samples, len(walls), response_cycles + 1, nx, ny),
                        np.nan,
                        dtype=np.float64,
                    )
                    enriched_seed_records: list[dict[str, Any]] = []
                    covariance_residuals: list[dict[str, float]] = []
                    for result in sorted(results, key=lambda item: int(item["task"]["sample"])):
                        sample = int(result["task"]["sample"])
                        np.testing.assert_array_equal(result["active_indices"], active_indices)
                        if not all(result["schedule_identical"]):
                            raise RuntimeError("A paired response worker consumed different plus/minus schedules.")
                        delta[sample] = result["delta"]
                        covariance_residuals.extend(result["covariance_residuals"])
                        enriched_seed_records.append(
                            {
                                **seed_records[sample],
                                "equilibration_dynamics_seed": result["equilibration_dynamics_seed"],
                                "response_dynamics_seeds": result["response_dynamics_seeds"],
                                "paired_schedule_identical": result["schedule_identical"],
                            }
                        )

                    analysis_tasks = [
                        {
                            "delta": delta[:, injection],
                            "regions": regions,
                            "y0": y0,
                            "walls": walls,
                            "norm_fraction_cutoff": RESPONSE_NORM_FRACTION_CUTOFF,
                        }
                        for injection in range(len(walls))
                    ]
                    analyses, analysis_execution = run_parallel_tasks(
                        _response_analysis_worker,
                        analysis_tasks,
                        policy=parallel_policy(args),
                        single_thread_tasks=True,
                        estimated_bytes_per_worker=int(delta[:, 0].nbytes * 2),
                        quiet=True,
                    )
                    analysis_payload: dict[str, Any] = {
                        "wall_x": np.asarray(walls, dtype=np.int64),
                        "injection_wall_x": np.asarray(walls, dtype=np.int64),
                    }
                    for key in (
                        "wall_profiles",
                        "response_norm",
                        "signed_first_moment",
                        "absolute_center",
                        "response_width",
                        "wall_weight",
                        "velocity_per_sample",
                        "velocity_fit_point_count",
                        "velocity_relaxed_fit",
                    ):
                        analysis_payload[key] = np.stack([analysis[key] for analysis in analyses], axis=1)
                    for key in ("velocity_mean", "velocity_sem"):
                        analysis_payload[key] = np.stack([analysis[key] for analysis in analyses], axis=0)
                    analysis_payload["wall_masks"] = analyses[0]["wall_masks"]
                    analysis_payload["periodic_displacement"] = analyses[0]["periodic_displacement"]
                    analysis_payload["y0"] = analyses[0]["y0"]
                    analysis_payload["fit_max_cycle"] = analyses[0]["fit_max_cycle"]
                    output = case_directory(
                        args.output_dir,
                        "response",
                        nx=nx,
                        ny=ny,
                        geometry=geometry,
                        nshell=nshell,
                        suffix=sequence,
                    )
                    save_npz_atomic(output / "response_raw.npz", delta_charge=delta, **regions.payload())
                    save_npz_atomic(output / "response_analysis.npz", **analysis_payload)
                    velocity_mean = analysis_payload["velocity_mean"]
                    diagonal_velocity = np.asarray(
                        [
                            velocity_mean[index, min(index, velocity_mean.shape[1] - 1)]
                            for index in range(len(walls))
                        ]
                    )
                    config = {
                        "diagnostic": "response",
                        "Nx": nx,
                        "Ny": ny,
                        "geometry": geometry,
                        "DW": geometry == "dw",
                        "dw_truncation": geometry == "dw",
                        "alpha_1": 1,
                        "alpha_2": 30,
                        "nshell": nshell,
                        "meas_slab_only": True,
                        "samples": samples,
                        "equilibration_cycles": equilibration_cycles,
                        "response_cycles": response_cycles,
                        "sequence": sequence,
                        "ordering_role": "primary" if sequence == "random" else "ordering_bias_control",
                        "random_seed": args.seed,
                        "y0": y0,
                        "rng_streams": RNG_STREAMS,
                        "seed_derivation": (
                            "SeedSequence(root).spawn(samples), then equilibration and per-wall "
                            "response worker streams shared by geometry index; each canonical run "
                            "derives one sample stream"
                        ),
                        "interface_width": regions.interface_width,
                        "covariance_tolerance": COVARIANCE_TOLERANCE,
                        "response_norm_fraction_cutoff": RESPONSE_NORM_FRACTION_CUTOFF,
                        "parallel_execution": compact_parallel_execution(execution),
                        "analysis_parallel_execution": compact_parallel_execution(analysis_execution),
                    }
                    metrics = {
                        "diagonal_velocity": diagonal_velocity,
                        "opposite_wall_drift": bool(diagonal_velocity[0] * diagonal_velocity[1] < 0.0)
                        if diagonal_velocity.size == 2 and np.all(np.isfinite(diagonal_velocity))
                        else None,
                        "initial_delta_charge": np.sum(delta[:, :, 0], axis=(2, 3)),
                        "paired_schedule_identical": bool(
                            all(
                                all(record["paired_schedule_identical"])
                                for record in enriched_seed_records
                            )
                        ),
                        "max_covariance_hermiticity_residual": float(
                            max(item["hermiticity_residual"] for item in covariance_residuals)
                        ),
                        "max_covariance_spectral_bound_violation": float(
                            max(item["spectral_bound_violation"] for item in covariance_residuals)
                        ),
                    }
                    rows = []
                    for injection, injection_x in enumerate(walls):
                        for observed, observed_x in enumerate(walls):
                            rows.append(
                                {
                                    **config,
                                    "injection_wall_x": injection_x,
                                    "observed_wall_x": observed_x,
                                    "velocity_mean": analysis_payload["velocity_mean"][injection, observed],
                                    "velocity_sem": analysis_payload["velocity_sem"][injection, observed],
                                }
                            )
                    write_json_atomic(
                        output / "run_summary.json",
                        {
                            **manifest(config, active_indices=active_indices),
                            "seed_records": enriched_seed_records,
                            "parallel_execution": execution,
                            "analysis_parallel_execution": analysis_execution,
                            "metrics": metrics,
                        },
                    )
                    write_csv_atomic(output / "scalar_metrics.csv", rows)
                    plot_response(analysis_payload, output)
                    summaries.append(
                        {
                            **config,
                            "opposite_wall_drift": metrics["opposite_wall_drift"],
                            "output": str(output),
                        }
                    )
    return summaries


def analyze_existing(
    path: Path,
    *,
    policy: ParallelPolicy | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    run_directories = sorted({file.parent for file in path.rglob("*.npz")})
    if not run_directories:
        return [], {
            "backend": "serial",
            "enabled": False,
            "task_count": 0,
            "workers": 0,
            "completed_tasks": 0,
        }
    results, execution = run_parallel_tasks(
        _analyze_run_worker,
        [{"run_directory": str(run_directory)} for run_directory in run_directories],
        policy=policy or ParallelPolicy(enabled=False),
        single_thread_tasks=True,
        estimated_bytes_per_worker=0,
        quiet=True,
    )
    generated = [item for group in results for item in group]
    return generated, execution


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CPU domain-wall frustration diagnostics")
    parser.add_argument("mode", choices=("static", "activity", "spectral", "response", "analyze", "all"))
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "topological_frustration_diagnostics" / "results")
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--geometry", choices=("uniform", "dw", "both"), default="both")
    parser.add_argument("--nx", type=int)
    parser.add_argument("--ny", type=int, nargs="+")
    parser.add_argument("--nshell", nargs="+")
    parser.add_argument("--samples", type=int)
    parser.add_argument("--cycles", type=int)
    parser.add_argument("--burn-in", type=int)
    parser.add_argument("--equilibration-cycles", type=int)
    parser.add_argument("--response-cycles", type=int)
    parser.add_argument("--interface-width", type=int)
    parser.add_argument("--lyapunov-nvec", type=int)
    parser.add_argument("--protocol", choices=("perfect_correction", "imperfect", "postselect", "both"))
    parser.add_argument("--sequence", choices=("raster_y", "raster_x", "random", "dw_symmetric_random"))
    parser.add_argument("--init-mode", choices=("default", "maxmix"), default="maxmix")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--svd-rtol", type=float, default=1e-10)
    parser.add_argument("--click-sequences", action="store_true")
    parser.add_argument("--sequence-permutations", type=int, default=1000)
    parser.add_argument("--sequence-bootstraps", type=int, default=1000)
    parser.add_argument("--sequence-min-support", type=int, default=20)
    parser.add_argument("--cpu-budget", type=int, default=80)
    parser.add_argument("--workers", type=parse_auto_positive, default=None)
    parser.add_argument("--threads-per-worker", type=parse_auto_positive, default=None)
    parser.add_argument("--memory-fraction", type=float, default=0.7)
    parser.add_argument("--no-parallel", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--progress", action="store_true")
    return parser


def main() -> int:
    configure_plotting()
    parser = build_parser()
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "analyze":
        if args.input_dir is None:
            parser.error("analyze mode requires --input-dir")
        summaries, analysis_execution = analyze_existing(
            args.input_dir.resolve(),
            policy=parallel_policy(args),
        )
    elif args.mode == "static":
        summaries = run_static(args)
    elif args.mode == "activity":
        summaries = run_activity(args)
    elif args.mode == "spectral":
        summaries = run_spectral(args)
    elif args.mode == "response":
        summaries = run_response(args)
    else:
        original_protocol = args.protocol
        summaries = run_static(args)
        args.protocol = "perfect_correction" if original_protocol is None else original_protocol
        summaries.extend(run_activity(args))
        args.protocol = "both" if original_protocol is None else original_protocol
        summaries.extend(run_spectral(args))
        args.protocol = "perfect_correction"
        summaries.extend(run_response(args))
    campaign_payload: Any = summaries
    if args.mode == "analyze":
        campaign_payload = {"runs": summaries, "parallel_execution": analysis_execution}
    write_json_atomic(args.output_dir / f"{args.mode}_campaign_summary.json", campaign_payload)
    print(json.dumps({"mode": args.mode, "runs": len(summaries), "output_dir": str(args.output_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
