#!/usr/bin/env python3
"""One-record, resumable frozen-flux pilot using the canonical CPU engine."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from typing import Any

import numpy as np
from threadpoolctl import threadpool_limits
from tqdm.auto import tqdm


def repository_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (
            (candidate / "src/fgtn/classA_U1FGTN.py").is_file()
            and (candidate / "PROJECT_ADMIN/REPO_POLICY.md").is_file()
        ):
            return candidate
    raise RuntimeError("could not locate repository root containing src/fgtn")


REPOSITORY_ROOT = repository_root()
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
sys.path.insert(0, str(REPOSITORY_ROOT / "00_WORKSPACE/CURRENT/final_production_ready_figure_scripts/_shared_src"))

from fgtn.classA_U1FGTN import classA_U1FGTN  # noqa: E402
from h3_twist_observables import (  # noqa: E402
    crossing_summary,
    half_torus_indices,
    initial_order,
    track_step,
)


SCHEMA = "h3_cpu_one_record_flux_pilot_v2_production_geometry"
CANONICAL_ENTRY_POINT = "classA_U1FGTN.run_markov_circuit"


def production_wall_locations(nx: int) -> tuple[int, int]:
    """Return the frozen H3 GPU-production domain-wall centers."""
    half = int(nx) // 2
    width = max(1, int(nx) // 4)
    return max(0, half - width), min(int(nx), half + width + 1) - 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("cpu_flux_pilot_N20x24"))
    parser.add_argument("--nx", type=int, default=20)
    parser.add_argument("--ny", type=int, default=40)
    parser.add_argument("--grid-points", type=int, default=33)
    parser.add_argument(
        "--cycles",
        type=int,
        help="Physical cycles; defaults to 2*Ny.",
    )
    parser.add_argument("--seed", type=int, default=2026081703)
    parser.add_argument("--tracked-modes", type=int, default=24)
    parser.add_argument("--wall-half-width", type=int, default=2)
    parser.add_argument("--cpu-start", type=int)
    parser.add_argument("--cpu-stop", type=int)
    parser.add_argument("--minimum-free-gb", type=float, default=1.0)
    parser.add_argument(
        "--protocol",
        choices=(
            "explicit_interface",
            "matched_trivial",
            "support_terminated",
            "support_terminated_matched_trivial",
        ),
        default="explicit_interface",
    )
    return parser.parse_args()


def configure_cpu_range(start: int | None, stop: int | None) -> list[int]:
    if (start is None) != (stop is None):
        raise ValueError("--cpu-start and --cpu-stop must be supplied together")
    if start is not None:
        if start < 0 or stop is None or stop <= start:
            raise ValueError("CPU range must satisfy 0 <= --cpu-start < --cpu-stop")
        requested = set(range(start, stop))
        available = set(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else requested
        selected = sorted(requested & available)
        if not selected:
            raise ValueError(
                f"requested CPU range [{start}, {stop}) does not overlap available CPUs {sorted(available)}"
            )
        if hasattr(os, "sched_setaffinity"):
            os.sched_setaffinity(0, selected)
    elif hasattr(os, "sched_getaffinity"):
        selected = sorted(os.sched_getaffinity(0))
    else:
        selected = list(range(os.cpu_count() or 1))
    threads = max(1, len(selected))
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = str(threads)
    return selected


def atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npy", delete=False) as handle:
        temporary = Path(handle.name)
        np.save(handle, array, allow_pickle=False)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def atomic_gzip_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=6) as handle:
        json.dump(payload, handle, separators=(",", ":"))
    os.replace(temporary, path)


class RecordObserver:
    def __init__(self, *, capture: bool) -> None:
        self.capture = bool(capture)
        self.entries: list[dict[str, Any]] = []
        self.total_log_probability = 0.0
        self.minimum_selected_probability = 1.0

    @staticmethod
    def _selected_probability(event: dict[str, Any]) -> float:
        probability = float(event["probability"])
        if event["kind"] == "measurement":
            return probability if bool(event["outcome_occupied"]) else 1.0 - probability
        expected = bool(event["expected_occupied"])
        target = bool(event["target_occupied"])
        if bool(event.get("perfect_correction")):
            return 1.0 if target == expected else 0.0
        occurred = target if expected else not target
        return probability if occurred else 1.0 - probability

    def __call__(
        self,
        *,
        cycle: int,
        site_id: int,
        branch_log_weight: float,
        branch_events: Any,
        **_: Any,
    ) -> None:
        events = [dict(event) for event in branch_events]
        self.total_log_probability += float(branch_log_weight)
        if events:
            self.minimum_selected_probability = min(
                self.minimum_selected_probability,
                *(self._selected_probability(event) for event in events),
            )
        if self.capture:
            self.entries.append(
                {"cycle": int(cycle), "site_id": int(site_id), "branch_events": events}
            )


class InitialStateObserver:
    def __init__(self) -> None:
        self.value: np.ndarray | None = None

    def __call__(self, *, cycle: int, G: np.ndarray, **_: Any) -> None:
        if int(cycle) == 0 and self.value is None:
            self.value = np.array(G, dtype=np.complex128, copy=True)


def model_config(
    protocol: str,
    twist: float,
    *,
    nx: int,
    ny: int,
    wall_centers: tuple[int, int],
) -> dict[str, Any]:
    matched = protocol in (
        "matched_trivial",
        "support_terminated_matched_trivial",
    )
    support_terminated = protocol.startswith("support_terminated")
    config = {
        "Nx": int(nx),
        "Ny": int(ny),
        "DW": support_terminated or not matched,
        "nshell": 1,
        "filling_frac": 0.5,
        "alpha_1": 30.0 if matched else 1.0,
        "alpha_2": 30.0,
        "trial_orbitals": "X",
        "dw_truncation": support_terminated,
        "twist_y": float(twist),
    }
    if config["DW"]:
        config["dw_interval"] = tuple(int(value) for value in wall_centers)
    return config


def make_model(
    protocol: str,
    twist: float,
    *,
    nx: int,
    ny: int,
    wall_centers: tuple[int, int],
) -> classA_U1FGTN:
    config = model_config(
        protocol,
        twist,
        nx=nx,
        ny=ny,
        wall_centers=wall_centers,
    )
    model = classA_U1FGTN(**config)
    model.construct_OW_projectors(
        nshell=1,
        DW=bool(config["DW"]),
        trial_orbitals="X",
        dw_truncation=bool(config["dw_truncation"]),
        twist_y=float(twist),
    )
    return model


def run_kwargs(
    cycles: int, seed: int, *, meas_slab_only: bool
) -> dict[str, Any]:
    return {
        "G_history": False,
        "progress": True,
        "cycles": int(cycles),
        "samples": 1,
        "parallelize_samples": False,
        "init_mode": "default",
        "save": False,
        "sequence": "random",
        "meas_slab_only": bool(meas_slab_only),
        "random_seed": int(seed),
        "perfect_correction": True,
        "physical_covariance_update": "rank1",
    }


def entanglement_eigensystem(
    G: np.ndarray,
    *,
    nx: int,
    ny: int,
    tracked_modes: int,
    wall_half_width: int,
    wall_centers: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    indices = half_torus_indices(nx, ny)
    restricted = G[np.ix_(indices, indices)]
    C = 0.5 * (restricted + np.eye(indices.size, dtype=np.complex128))
    C = 0.5 * (C + C.conj().T)
    all_values, all_vectors = np.linalg.eigh(C)
    order = np.argsort(np.abs(all_values.real - 0.5))[:tracked_modes]
    values = all_values.real[order]
    vectors = all_vectors[:, order]
    mode_x = (indices // 2) % nx
    weights = []
    for center in wall_centers:
        distances = np.minimum((mode_x - center) % nx, (center - mode_x) % nx)
        weights.append(np.sum(np.abs(vectors[distances <= wall_half_width]) ** 2, axis=0))
    return values, vectors, np.stack(weights, axis=-1), float(np.min(np.abs(all_values.real - 0.5)))


def load_record(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return list(json.load(handle))


def state_arrays(
    points: int, tracked_modes: int, *, nx: int, ny: int
) -> dict[str, np.ndarray]:
    dim = nx * (ny // 2) * 2
    return {
        "completed_points": np.asarray(0, dtype=np.int64),
        "phi": np.linspace(0.0, 2.0 * math.pi, points),
        "occupation_values": np.full((points, tracked_modes), np.nan),
        "entanglement_energies": np.full((points, tracked_modes), np.nan),
        "wall_weights": np.full((points, tracked_modes, 2), np.nan),
        "branch_log_probability": np.full((points,), np.nan),
        "minimum_event_probability": np.full((points,), np.nan),
        "reference_gap": np.full((points,), np.nan),
        "point_elapsed_seconds": np.full((points,), np.nan),
        "successive_covariance_frobenius_per_dimension": np.full((points,), np.nan),
        "overlap_matrices": np.full((points - 1, tracked_modes, tracked_modes), np.nan + 0j),
        "tracking_assignments": np.full((points - 1, tracked_modes), -1, dtype=np.int16),
        "first_vectors": np.empty((dim, tracked_modes), dtype=np.complex128),
        "last_vectors": np.empty((dim, tracked_modes), dtype=np.complex128),
        "phi_zero_replay_error": np.asarray(np.nan),
        "phi_zero_cycle_zero_error": np.asarray(np.nan),
    }


def main() -> None:
    args = parse_args()
    if args.nx <= 0 or args.ny <= 0 or args.ny % 2:
        raise ValueError("--nx and --ny must be positive, and --ny must be even")
    cycles = 2 * args.ny if args.cycles is None else int(args.cycles)
    meas_slab_only = args.protocol.startswith("support_terminated")
    wall_centers = production_wall_locations(args.nx)
    if args.grid_points < 3:
        raise ValueError("--grid-points must be at least 3 for a closed circle")
    if cycles <= 0 or args.tracked_modes <= 0:
        raise ValueError("--cycles and --tracked-modes must be positive")
    cpus = configure_cpu_range(args.cpu_start, args.cpu_stop)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(output).free / 1024**3
    if free_gb < args.minimum_free_gb:
        raise RuntimeError(
            f"only {free_gb:.2f} GiB free at {output}; require {args.minimum_free_gb:.2f} GiB"
        )

    record_path = output / "trajectory_record.json.gz"
    parent_path = output / "parent_state.npz"
    checkpoint_path = output / "flux_checkpoint.npz"
    last_covariance_path = output / "resume_last_covariance.npy"
    manifest_path = output / "manifest.json"
    print(
        json.dumps(
            {
                "schema": SCHEMA,
                "canonical_entry_point": CANONICAL_ENTRY_POINT,
                "geometry": f"{args.nx}x{args.ny}",
                "sample_count": 1,
                "cycles": cycles,
                "grid_points": args.grid_points,
                "protocol": args.protocol,
                "meas_slab_only": meas_slab_only,
                "wall_locations": list(wall_centers),
                "domain_wall_rule": "frozen_h3_gpu_production_max(1, Nx // 4)",
                "cpus": cpus,
                "output": str(output),
                "free_GiB": round(free_gb, 2),
            },
            indent=2,
        )
    )

    run_start = time.perf_counter()
    thread_count = max(1, len(cpus))
    with threadpool_limits(limits=thread_count):
        if not record_path.exists() or not parent_path.exists():
            print("[parent] generating one twist-zero trajectory and compact branch record")
            recorder = RecordObserver(capture=True)
            initial = InitialStateObserver()
            parent_started = time.perf_counter()
            parent_result = make_model(
                args.protocol,
                0.0,
                nx=args.nx,
                ny=args.ny,
                wall_centers=wall_centers,
            ).run_markov_circuit(
                trajectory_weight_observer=recorder,
                cycle_observer=initial,
                **run_kwargs(cycles, args.seed, meas_slab_only=meas_slab_only),
            )
            if initial.value is None:
                raise RuntimeError("canonical CPU engine did not emit its cycle-zero state")
            parent_final = np.asarray(parent_result["G_final"][0], dtype=np.complex128)
            atomic_gzip_json(record_path, recorder.entries)
            atomic_npz(
                parent_path,
                G_init=initial.value,
                G_final=parent_final,
                Nx=np.asarray(args.nx, dtype=np.int64),
                Ny=np.asarray(args.ny, dtype=np.int64),
                wall_locations=np.asarray(wall_centers, dtype=np.int64),
                meas_slab_only=np.asarray(meas_slab_only),
                protocol=np.asarray(args.protocol),
                cycles=np.asarray(cycles, dtype=np.int64),
                seed=np.asarray(args.seed, dtype=np.int64),
                total_log_probability=np.asarray(recorder.total_log_probability),
                minimum_selected_probability=np.asarray(recorder.minimum_selected_probability),
                parent_elapsed_seconds=np.asarray(time.perf_counter() - parent_started),
            )
            print(
                f"[parent complete] sites={len(recorder.entries)}, "
                f"elapsed={(time.perf_counter() - parent_started) / 60.0:.2f} min, "
                f"record={record_path.stat().st_size / 1024**2:.2f} MiB"
            )
        else:
            print("[resume] loading existing parent covariance and trajectory record")

        record = load_record(record_path)
        with np.load(parent_path) as parent:
            if (
                str(parent["protocol"]) != args.protocol
                or int(parent["cycles"]) != cycles
                or int(parent["seed"]) != args.seed
                or int(parent.get("Nx", args.nx)) != args.nx
                or int(parent.get("Ny", args.ny)) != args.ny
                or "wall_locations" not in parent.files
                or not np.array_equal(
                    np.asarray(parent["wall_locations"], dtype=np.int64),
                    np.asarray(wall_centers, dtype=np.int64),
                )
                or bool(parent.get("meas_slab_only", meas_slab_only))
                != meas_slab_only
            ):
                raise ValueError(
                    "saved parent uses different protocol/cycles/seed/wall geometry; "
                    "choose a new output directory"
                )
            G_init = np.array(parent["G_init"], copy=True)
            parent_final = np.array(parent["G_final"], copy=True)

        if checkpoint_path.exists():
            with np.load(checkpoint_path) as saved:
                state = {name: np.array(saved[name], copy=True) for name in saved.files}
            expected_phi = np.linspace(0.0, 2.0 * math.pi, args.grid_points)
            if not np.array_equal(state["phi"], expected_phi):
                raise ValueError("checkpoint flux grid differs; choose a new output directory")
            if state["occupation_values"].shape != (args.grid_points, args.tracked_modes):
                raise ValueError(
                    "checkpoint tracked-mode count differs; choose a new output directory"
                )
            completed = int(state["completed_points"])
            previous_covariance = np.load(last_covariance_path, allow_pickle=False) if completed else None
            print(f"[resume] {completed}/{args.grid_points} flux points already complete")
        else:
            state = state_arrays(
                args.grid_points,
                args.tracked_modes,
                nx=args.nx,
                ny=args.ny,
            )
            completed = 0
            previous_covariance = None

        phi = state["phi"]
        point_bar = tqdm(
            range(completed, args.grid_points),
            initial=completed,
            total=args.grid_points,
            desc="CPU frozen-flux points",
            unit="point",
        )
        for point_index in point_bar:
            point_started = time.perf_counter()
            twist = float(phi[point_index])
            print(
                f"[point {point_index + 1}/{args.grid_points}] phi={twist:.9f}; "
                "constructing twisted CPU projectors"
            )
            replay_observer = RecordObserver(capture=False)
            replay_initial = InitialStateObserver()
            mode_y = np.repeat(np.arange(args.ny), 2 * args.nx)
            initial_phase = np.exp(1j * twist * mode_y / float(args.ny))
            twisted_G_init = (
                initial_phase[:, None]
                * G_init
                * initial_phase.conj()[None, :]
            )
            result = make_model(
                args.protocol,
                twist,
                nx=args.nx,
                ny=args.ny,
                wall_centers=wall_centers,
            ).run_markov_circuit(
                G_init=twisted_G_init,
                trajectory_replay=record,
                trajectory_weight_observer=replay_observer,
                cycle_observer=replay_initial,
                trajectory_replay_probability_tol=1e-14,
                **run_kwargs(cycles, args.seed, meas_slab_only=meas_slab_only),
            )
            if replay_initial.value is None:
                raise RuntimeError("replay did not emit its cycle-zero state")
            final_covariance = np.asarray(result["G_final"][0], dtype=np.complex128)
            values, vectors, weights, reference_gap = entanglement_eigensystem(
                final_covariance,
                nx=args.nx,
                ny=args.ny,
                tracked_modes=args.tracked_modes,
                wall_half_width=args.wall_half_width,
                wall_centers=wall_centers,
            )
            if point_index == 0:
                order = initial_order(values[None])[0]
                values, vectors, weights = values[order], vectors[:, order], weights[order]
                state["first_vectors"] = vectors
                state["phi_zero_replay_error"] = np.asarray(
                    np.linalg.norm(final_covariance - parent_final) / final_covariance.shape[0]
                )
                state["phi_zero_cycle_zero_error"] = np.asarray(
                    np.linalg.norm(replay_initial.value - G_init)
                    / replay_initial.value.shape[0]
                )
            else:
                tracked = track_step(
                    state["last_vectors"][None],
                    values[None],
                    vectors[None],
                    weights[None],
                )
                values = tracked[0][0]
                vectors = tracked[1][0]
                weights = tracked[2][0]
                state["overlap_matrices"][point_index - 1] = tracked[3][0]
                state["tracking_assignments"][point_index - 1] = tracked[4][0]
            if previous_covariance is not None:
                state["successive_covariance_frobenius_per_dimension"][point_index] = (
                    np.linalg.norm(final_covariance - previous_covariance) / final_covariance.shape[0]
                )
            state["last_vectors"] = vectors
            state["occupation_values"][point_index] = values
            state["entanglement_energies"][point_index] = np.log(
                np.clip(1.0 - values, 1e-14, 1.0) / np.clip(values, 1e-14, 1.0)
            )
            state["wall_weights"][point_index] = weights
            state["branch_log_probability"][point_index] = replay_observer.total_log_probability
            state["minimum_event_probability"][point_index] = replay_observer.minimum_selected_probability
            state["reference_gap"][point_index] = reference_gap
            state["point_elapsed_seconds"][point_index] = time.perf_counter() - point_started
            state["completed_points"] = np.asarray(point_index + 1, dtype=np.int64)
            atomic_npy(last_covariance_path, final_covariance)
            atomic_npz(checkpoint_path, **state)
            previous_covariance = final_covariance
            mean_seconds = float(np.nanmean(state["point_elapsed_seconds"][: point_index + 1]))
            eta_seconds = mean_seconds * (args.grid_points - point_index - 1)
            point_bar.set_postfix(
                point_min=f"{state['point_elapsed_seconds'][point_index] / 60.0:.1f}",
                eta_h=f"{eta_seconds / 3600.0:.2f}",
            )
            print(
                f"[point complete] elapsed={state['point_elapsed_seconds'][point_index] / 60.0:.2f} min; "
                f"rolling ETA={eta_seconds / 3600.0:.2f} h; checkpoint={checkpoint_path}"
            )

    if int(state["completed_points"]) == args.grid_points:
        crossings, ambiguous = crossing_summary(
            state["entanglement_energies"][None], state["wall_weights"][None]
        )
        subsystem_y = np.repeat(np.arange(args.ny // 2), 2 * args.nx)
        large_gauge = np.exp(2j * math.pi * subsystem_y / float(args.ny))
        closure_vectors = large_gauge.conj()[:, None] * state["last_vectors"]
        closure_projector_error = float(
            np.linalg.norm(
                closure_vectors @ closure_vectors.conj().T
                - state["first_vectors"] @ state["first_vectors"].conj().T
            )
            / state["first_vectors"].shape[0]
        )
        last_covariance = np.load(last_covariance_path, allow_pickle=False)
        full_y = np.repeat(np.arange(args.ny), 2 * args.nx)
        full_gauge = np.exp(2j * math.pi * full_y / float(args.ny))
        closed_covariance = full_gauge.conj()[:, None] * last_covariance * full_gauge[None, :]
        covariance_closure = float(
            np.linalg.norm(closed_covariance - parent_final) / parent_final.shape[0]
        )
        manifest = {
            "schema": SCHEMA,
            "status": "complete",
            "canonical_dynamics_entry_point": CANONICAL_ENTRY_POINT,
            "geometry": {"Nx": args.nx, "Ny": args.ny},
            "wall_locations": list(wall_centers),
            "domain_wall_rule": "frozen_h3_gpu_production_max(1, Nx // 4)",
            "protocol": args.protocol,
            "model": model_config(
                args.protocol,
                0.0,
                nx=args.nx,
                ny=args.ny,
                wall_centers=wall_centers,
            ),
            "correction": {
                "supersedes_schema": "h3_cpu_one_record_flux_pilot_v1",
                "reason": "match the frozen H3 GPU wall interval and use it for both dynamics and wall assignment",
                "superseded_result_preserved": "results/N20x24_explicit_interface_grid9",
            },
            "run": {
                "samples": 1,
                "cycles": cycles,
                "sequence": "random",
                "perfect_correction": True,
                "meas_slab_only": meas_slab_only,
                "grid_points": args.grid_points,
                "phi": phi.tolist(),
                "seed": args.seed,
                "tracked_modes": args.tracked_modes,
            },
            "cpu_allocation": cpus,
            "record_sites": len(record),
            "saved_covariance_history": False,
            "phi_zero_cycle_zero_frobenius_per_dimension": float(
                state["phi_zero_cycle_zero_error"]
            ),
            "phi_zero_replay_frobenius_per_dimension": float(state["phi_zero_replay_error"]),
            "successive_covariance_frobenius_per_dimension": [
                None if not np.isfinite(value) else float(value)
                for value in state["successive_covariance_frobenius_per_dimension"]
            ],
            "covariance_closure_frobenius_per_dimension": covariance_closure,
            "tracked_subspace_closure_frobenius_per_dimension": closure_projector_error,
            "wall_crossing_counts": crossings[0].tolist(),
            "ambiguous_crossings": int(ambiguous[0]),
            "minimum_replayed_event_probability": float(np.nanmin(state["minimum_event_probability"])),
            "point_elapsed_seconds": state["point_elapsed_seconds"].tolist(),
            "total_elapsed_seconds_this_invocation": time.perf_counter() - run_start,
            "output_bytes": sum(path.stat().st_size for path in output.iterdir() if path.is_file()),
            "artifacts": {
                "trajectory_record": record_path.name,
                "parent_state": parent_path.name,
                "checkpoint_and_results": checkpoint_path.name,
                "resume_last_covariance": last_covariance_path.name,
            },
        }
        atomic_json(manifest_path, manifest)
        print(json.dumps(manifest, indent=2, sort_keys=True))
        print(f"[complete] manifest={manifest_path}")


if __name__ == "__main__":
    main()
