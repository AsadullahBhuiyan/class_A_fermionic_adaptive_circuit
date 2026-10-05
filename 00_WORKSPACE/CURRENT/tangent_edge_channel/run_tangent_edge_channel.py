#!/usr/bin/env python3
"""CPU reference campaign for the physical two-mode tangent edge channel.

The physical trajectory is burned in and observed in one uninterrupted call to
``classA_U1FGTN.run_markov_circuit``.  The observer follows two predetermined
Bloch edge modes through the fixed realized measurement record; it does not
differentiate the Born draw itself.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import zlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

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
    TangentChannelRecorder,
    build_physical_edge_frame,
    project_edge_frame,
)
from fgtn.diagnostics.cpu_parallel import ParallelPolicy, run_parallel_tasks
from fgtn.diagnostics.io import save_npz_atomic, write_csv_atomic, write_json_atomic


FIGURE_WIDTH = 3.375
STATIC_RECORDER_KEYS = {
    "tangent_target_orbital_mask",
    "tangent_opposite_orbital_mask",
    "tangent_interface_orbital_mask",
    "tangent_delta_momentum",
    "tangent_initial_interference_harmonic",
}


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
            "savefig.dpi": 300,
        }
    )


def _save_figure(fig: plt.Figure, directory: Path, stem: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(directory / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(directory / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def _git_metadata() -> dict[str, Any]:
    def run(*command: str) -> str:
        result = subprocess.run(
            command, cwd=REPO_ROOT, text=True, capture_output=True, check=False
        )
        return result.stdout.strip()

    return {
        "commit": run("git", "rev-parse", "HEAD"),
        "dirty": bool(run("git", "status", "--short")),
    }


def _nshell_tag(value: float | int | None) -> str:
    return "none" if value is None else str(value).replace(".", "p")


def _normalize_nshell(value: Any) -> float | int | None:
    if value is None or str(value).strip().lower() in ("none", "full"):
        return None
    number = float(value)
    return int(number) if number.is_integer() else number


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict) or not isinstance(config.get("cases"), list):
        raise ValueError("The config must be an object with a 'cases' list.")
    return config


def _case_model(
    case: dict[str, Any], *, nx: int, ny: int, nshell: float | int | None
) -> classA_U1FGTN:
    model = classA_U1FGTN(
        nx,
        ny,
        DW=bool(case["DW"]),
        nshell=nshell,
        alpha_1=float(case["alpha_1"]),
        alpha_2=float(case["alpha_2"]),
        trial_orbitals=str(case.get("trial_orbitals", "X")),
        dw_truncation=bool(case["dw_truncation"]),
    )
    model.construct_OW_projectors(
        nshell=nshell,
        DW=bool(case["DW"]),
        trial_orbitals=str(case.get("trial_orbitals", "X")),
        dw_truncation=bool(case["dw_truncation"]),
    )
    return model


def _reference_edge_frame(
    *,
    nx: int,
    ny: int,
    active_indices: np.ndarray,
    wall: str,
    interface_width: int,
) -> Any:
    reference = classA_U1FGTN(
        nx,
        ny,
        DW=True,
        nshell=None,
        alpha_1=1,
        alpha_2=30,
        trial_orbitals="X",
        dw_truncation=False,
    )
    edge = build_physical_edge_frame(
        reference,
        wall=wall,
        interface_width=interface_width,
    )
    full = np.arange(2 * nx * ny, dtype=np.int64)
    if np.array_equal(np.asarray(active_indices, dtype=np.int64), full):
        return edge
    return project_edge_frame(edge, active_indices)


def _protocol_kwargs(protocol: str) -> dict[str, Any]:
    protocol = str(protocol).strip().lower()
    if protocol == "born_perfect_correction":
        return {"postselect": False, "perfect_correction": True}
    if protocol == "postselect_validation":
        return {"postselect": True, "perfect_correction": False}
    raise ValueError(f"Unknown protocol {protocol!r}.")


def _single_sample_worker(task: dict[str, Any]) -> dict[str, Any]:
    case = dict(task["case"])
    model = _case_model(
        case,
        nx=int(task["nx"]),
        ny=int(task["ny"]),
        nshell=task["nshell"],
    )
    edge = task["edge_frame"]
    active_indices = np.asarray(task["active_indices"], dtype=np.int64)
    recorder = TangentChannelRecorder(
        nx=int(task["nx"]),
        ny=int(task["ny"]),
        samples=1,
        observation_cycles=int(task["observation_cycles"]),
        edge_frame=edge,
        active_indices=active_indices,
        rank_tol=float(task["singular_tol"]),
    )
    total_cycles = int(task["burn_cycles"]) + int(task["observation_cycles"])
    model.run_markov_circuit(
        G_history=False,
        progress=False,
        cycles=total_cycles,
        samples=1,
        parallelize_samples=False,
        init_mode="maxmix",
        save=False,
        save_init=False,
        sequence=str(case["sequence"]),
        meas_slab_only=bool(case["meas_slab_only"]),
        random_seed=int(task["seed"]),
        physical_covariance_update="rank1",
        lyapunov_frame_observer=recorder,
        lyapunov_initial_frame=edge.frame,
        lyapunov_start_cycle=int(task["burn_cycles"]) + 1,
        lyapunov_track_restricted_core=True,
        lyapunov_track_record_fisher=True,
        lyapunov_singular_tol=float(task["singular_tol"]),
        lyapunov_failure_mode=str(task["failure_mode"]),
        **_protocol_kwargs(str(case["protocol"])),
    )
    recorder.assert_complete(allow_censored=True)
    payload = recorder.payload()
    static_payload = {key: payload.pop(key) for key in STATIC_RECORDER_KEYS}
    row = recorder.summary_rows()[0]
    return {
        "payload": payload,
        "static_payload": static_payload,
        "summary": row,
        "failure_records": recorder.failure_records,
    }


def _sample_seeds(root_seed: int, run_key: str, samples: int) -> list[int]:
    stable_key = int(zlib.crc32(run_key.encode("utf-8")))
    sequence = np.random.SeedSequence([int(root_seed), stable_key])
    return [
        int(child.generate_state(1, dtype=np.uint64)[0])
        for child in sequence.spawn(int(samples))
    ]


def _merge_sample_payloads(results: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    if not results:
        raise ValueError("No sample results were returned.")
    keys = tuple(results[0]["payload"])
    merged: dict[str, np.ndarray] = {}
    for key in keys:
        arrays = [np.asarray(result["payload"][key]) for result in results]
        if any(array.ndim == 0 or array.shape[0] != 1 for array in arrays):
            raise ValueError(f"Dynamic payload {key!r} has no singleton sample axis.")
        merged[key] = np.concatenate(arrays, axis=0)
    for key, value in results[0]["static_payload"].items():
        reference = np.asarray(value)
        for result in results[1:]:
            np.testing.assert_allclose(
                np.asarray(result["static_payload"][key]), reference, equal_nan=True
            )
        merged[key] = reference
    return merged


def _finite_mean_sem(values: np.ndarray, *, axis: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    array = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(array)
    count = np.sum(valid, axis=axis)
    total = np.sum(np.where(valid, array, 0.0), axis=axis)
    mean = np.divide(total, count, out=np.full_like(total, np.nan, dtype=np.float64), where=count > 0)
    centered = np.where(valid, array - np.expand_dims(mean, axis=axis), 0.0)
    variance = np.divide(
        np.sum(centered**2, axis=axis),
        count - 1,
        out=np.full_like(total, np.nan, dtype=np.float64),
        where=count > 1,
    )
    sem = np.sqrt(variance / np.maximum(count, 1))
    sem = np.where(count == 1, 0.0, sem)
    return mean, sem, count


def _cycle_rows(payload: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    cycles = np.arange(1, payload["tangent_observed"].shape[1] + 1, dtype=np.int64)
    metrics = {
        "mean_survival_exponent": payload["tangent_mean_survival_exponent"],
        "isotropy_defect": payload["tangent_isotropy_defect"],
        "target_wall_retention": np.nanmean(
            payload["tangent_target_wall_retention"], axis=-1
        ),
        "opposite_wall_leakage": np.nanmean(
            payload["tangent_opposite_wall_leakage"], axis=-1
        ),
        "interface_retention": np.nanmean(
            payload["tangent_interface_retention"], axis=-1
        ),
        "interference_coherence": payload["tangent_interference_coherence"],
        "phase_displacement": payload["tangent_phase_displacement"],
        "record_fisher_max_log_density": payload[
            "tangent_record_fisher_max_log_density"
        ],
    }
    statistics = {
        key: _finite_mean_sem(np.asarray(values), axis=0)
        for key, values in metrics.items()
    }
    rows: list[dict[str, Any]] = []
    for time_index, cycle in enumerate(cycles):
        active = payload["tangent_active"][:, time_index]
        active_ranks = payload["tangent_core_rank"][:, time_index][active]
        row: dict[str, Any] = {
            "lyapunov_cycle": int(cycle),
            "active_fraction": float(np.mean(active)),
            "rank_loss_fraction_among_active": float(
                np.mean(active_ranks < 2) if active_ranks.size else np.nan
            ),
        }
        for key, (mean, sem, count) in statistics.items():
            row[f"{key}_mean"] = float(mean[time_index])
            row[f"{key}_sem"] = float(sem[time_index])
            row[f"{key}_count"] = int(count[time_index])
        rows.append(row)
    return rows


def _aggregate_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = (
        "mean_survival_exponent_final",
        "survival_exponent_per_measurement_channel_final",
        "isotropy_defect_final",
        "target_wall_retention_final",
        "opposite_wall_leakage_final",
        "wall_velocity",
        "record_fisher_max_log_density_final",
    )
    summary: dict[str, Any] = {"samples": len(rows)}
    for metric in metrics:
        values = np.asarray([row.get(metric, np.nan) for row in rows], dtype=np.float64)
        mean, sem, count = _finite_mean_sem(values, axis=0)
        summary[f"{metric}_mean"] = float(mean)
        summary[f"{metric}_sem"] = float(sem)
        summary[f"{metric}_finite_count"] = int(count)
    summary["censored_fraction"] = float(
        np.mean([not bool(row["active_final"]) for row in rows])
    )
    active_rows = [row for row in rows if bool(row["active_final"])]
    summary["rank_loss_fraction_final_among_active"] = float(
        np.mean([int(row["core_rank_final"]) < 2 for row in active_rows])
        if active_rows
        else np.nan
    )
    summary["infinite_record_fisher_fraction_final"] = float(
        np.mean(
            [
                np.isposinf(float(row["record_fisher_max_log_density_final"]))
                for row in rows
            ]
        )
    )
    return summary


def _plot_case(payload: dict[str, np.ndarray], directory: Path) -> None:
    configure_plotting()
    cycles = np.arange(1, payload["tangent_observed"].shape[1] + 1)

    fig, axes = plt.subplots(2, 1, figsize=(FIGURE_WIDTH, 3.7), sharex=True)
    for axis, key, ylabel in (
        (axes[0], "tangent_mean_survival_exponent", r"$\bar\lambda_{\rm edge}(t)$"),
        (axes[1], "tangent_isotropy_defect", "isotropy defect"),
    ):
        mean, sem, _ = _finite_mean_sem(payload[key], axis=0)
        axis.plot(cycles, mean, color="C0", lw=1.2)
        axis.fill_between(cycles, mean - sem, mean + sem, color="C0", alpha=0.2, lw=0)
        axis.set_ylabel(ylabel)
    axes[-1].set_xlabel("observed cycle")
    _save_figure(fig, directory, "channel_strength")

    fig, axes = plt.subplots(2, 1, figsize=(FIGURE_WIDTH, 3.7), sharex=True)
    for key, label, color in (
        ("tangent_target_wall_retention", "target wall", "C0"),
        ("tangent_opposite_wall_leakage", "opposite wall", "C3"),
    ):
        data = np.nanmean(payload[key], axis=-1)
        mean, sem, _ = _finite_mean_sem(data, axis=0)
        axes[0].plot(cycles, mean, color=color, label=label, lw=1.2)
        axes[0].fill_between(cycles, mean - sem, mean + sem, color=color, alpha=0.18, lw=0)
    axes[0].set_ylabel("normalized weight")
    axes[0].legend(frameon=False)
    mean, sem, _ = _finite_mean_sem(payload["tangent_phase_displacement"], axis=0)
    axes[1].plot(cycles, mean, color="C2", lw=1.2)
    axes[1].fill_between(cycles, mean - sem, mean + sem, color="C2", alpha=0.2, lw=0)
    axes[1].set_ylabel("wall displacement")
    axes[1].set_xlabel("observed cycle")
    _save_figure(fig, directory, "wall_transport")


def _run_case(
    spec: dict[str, Any],
    *,
    output_root: Path,
    policy: ParallelPolicy,
    root_seed: int,
    singular_tol: float,
    failure_mode: str,
) -> dict[str, Any]:
    case = dict(spec["case"])
    nx, ny = int(spec["nx"]), int(spec["ny"])
    nshell = spec["nshell"]
    model = _case_model(case, nx=nx, ny=ny, nshell=nshell)
    active_indices = model.active_top_layer_indices(
        meas_slab_only=bool(case["meas_slab_only"])
    )
    edge = _reference_edge_frame(
        nx=nx,
        ny=ny,
        active_indices=active_indices,
        wall=str(spec["wall"]),
        interface_width=int(spec["interface_width"]),
    )
    run_key = (
        f"{case['name']}|N{nx}x{ny}|nsh={nshell}|"
        f"burn={spec['burn_cycles']}|observe={spec['observation_cycles']}"
    )
    seeds = _sample_seeds(root_seed, run_key, int(spec["samples"]))
    tasks = [
        {
            **spec,
            "seed": seed,
            "active_indices": active_indices,
            "edge_frame": edge,
            "singular_tol": singular_tol,
            "failure_mode": failure_mode,
        }
        for seed in seeds
    ]
    dimension = 2 * nx * ny
    results, parallel = run_parallel_tasks(
        _single_sample_worker,
        tasks,
        policy=policy,
        single_thread_tasks=True,
        estimated_bytes_per_worker=int(20 * dimension * dimension * np.dtype(np.complex128).itemsize),
        quiet=True,
    )
    payload = _merge_sample_payloads(results)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    channels_per_cycle = 4 * (active_indices.size // 2)
    for sample, (seed, result) in enumerate(zip(seeds, results)):
        row = dict(result["summary"])
        row.update(
            {
                "sample_index": sample,
                "seed": seed,
                "case": str(case["name"]),
                "nx": nx,
                "ny": ny,
                "nshell": nshell,
                "burn_cycles": int(spec["burn_cycles"]),
                "observation_cycles": int(spec["observation_cycles"]),
                "measurement_channels_per_cycle": channels_per_cycle,
                "survival_exponent_per_measurement_channel_final": (
                    float(row["mean_survival_exponent_final"]) / channels_per_cycle
                ),
            }
        )
        rows.append(row)
        for record in result["failure_records"]:
            failures.append({**dict(record), "sample_index": sample, "seed": seed})

    run_directory = (
        output_root
        / f"N{nx}x{ny}"
        / f"{case['name']}_nsh{_nshell_tag(nshell)}"
    )
    run_directory.mkdir(parents=True, exist_ok=True)
    save_npz_atomic(
        run_directory / "tangent_diagnostics.npz",
        **payload,
        **edge.payload(),
    )
    write_csv_atomic(run_directory / "sample_metrics.csv", rows)
    cycle_rows = _cycle_rows(payload)
    write_csv_atomic(run_directory / "cycle_metrics.csv", cycle_rows)
    aggregate = _aggregate_summary(rows)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "canonical_dynamics_entry_point": CANONICAL_CPU_ENTRY_POINT,
        "cocycle_definition": "fixed-realized-record state tangent; Born draws are not differentiated",
        "physical_trajectory_continuity": "burn-in and observation use one run_markov_circuit call",
        "tangent_first_physical_cycle": int(spec["burn_cycles"]) + 1,
        "initial_physical_state": "maximally mixed covariance",
        "case": case,
        "nx": nx,
        "ny": ny,
        "nshell": nshell,
        "samples": int(spec["samples"]),
        "burn_cycles": int(spec["burn_cycles"]),
        "observation_cycles": int(spec["observation_cycles"]),
        "sample_seeds": seeds,
        "active_top_layer_indices": active_indices,
        "active_cell_count": int(active_indices.size // 2),
        "measurement_channels_per_cycle": channels_per_cycle,
        "measurement_schedule_group": str(case["measurement_schedule_group"]),
        "edge_frame": edge.metadata(),
        "singular_tol": singular_tol,
        "failure_mode": failure_mode,
        "failure_record_count": len(failures),
        "failure_records": failures,
        "aggregate": aggregate,
        "parallel_execution": parallel,
        "git": _git_metadata(),
        "outputs": {
            "arrays": "tangent_diagnostics.npz",
            "samples": "sample_metrics.csv",
            "cycles": "cycle_metrics.csv",
            "figures": [
                "figures/channel_strength.png",
                "figures/channel_strength.pdf",
                "figures/wall_transport.png",
                "figures/wall_transport.pdf",
            ],
        },
    }
    write_json_atomic(run_directory / "run_summary.json", manifest)
    _plot_case(payload, run_directory / "figures")
    return {
        "case": str(case["name"]),
        "nx": nx,
        "ny": ny,
        "nshell": nshell,
        "output_directory": str(run_directory),
        "measurement_schedule_group": str(case["measurement_schedule_group"]),
        **aggregate,
    }


def _iter_specs(config: dict[str, Any], args: argparse.Namespace) -> Iterable[dict[str, Any]]:
    default_nx = config.get("smoke_nx", 8) if args.smoke else config["nx"]
    nx = int(args.nx if args.nx is not None else default_nx)
    selected = None if args.cases is None else set(args.cases)
    smoke_cases = set(config.get("smoke_cases", ()))
    for raw_case in config["cases"]:
        case = dict(raw_case)
        if not bool(case.get("enabled", True)):
            continue
        if selected is not None and str(case["name"]) not in selected:
            continue
        if args.smoke and selected is None and smoke_cases and str(case["name"]) not in smoke_cases:
            continue
        ny_values = (
            [int(args.ny)]
            if args.ny is not None
            else [int(value) for value in case.get("ny_values", config["ny_values"])]
        )
        nshell_values = [
            _normalize_nshell(value)
            for value in case.get("nshell_values", config["nshell_values"])
        ]
        if args.smoke:
            ny_values = [int(args.ny if args.ny is not None else config.get("smoke_ny", 8))]
            nshell_values = [1]
        for ny in ny_values:
            for nshell in nshell_values:
                burn = (
                    int(args.burn_cycles)
                    if args.burn_cycles is not None
                    else int(np.ceil(float(config["burn_in_cycles_per_ny"]) * ny))
                )
                observe = (
                    int(args.observation_cycles)
                    if args.observation_cycles is not None
                    else int(np.ceil(float(config["observation_cycles_per_ny"]) * ny))
                )
                samples = int(args.samples if args.samples is not None else case["samples"])
                if args.smoke:
                    burn = int(args.burn_cycles if args.burn_cycles is not None else 1)
                    observe = int(args.observation_cycles if args.observation_cycles is not None else 2)
                    samples = int(args.samples if args.samples is not None else 1)
                yield {
                    "case": case,
                    "nx": nx,
                    "ny": ny,
                    "nshell": nshell,
                    "samples": samples,
                    "burn_cycles": burn,
                    "observation_cycles": observe,
                    "wall": str(config.get("wall", "left")),
                    "interface_width": int(config.get("interface_width", 2)),
                }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("reference_config.json"),
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--nx", type=int)
    parser.add_argument("--ny", type=int)
    parser.add_argument("--samples", type=int)
    parser.add_argument("--burn-cycles", type=int)
    parser.add_argument("--observation-cycles", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--cpu-budget", type=int, default=80)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--threads-per-worker", type=int)
    parser.add_argument("--memory-fraction", type=float, default=0.7)
    parser.add_argument("--no-parallel", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = _load_config(args.config.resolve())
    if args.output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_root = Path(__file__).with_name("results") / f"campaign_{stamp}"
    else:
        output_root = args.output_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    specs = list(_iter_specs(config, args))
    if not specs:
        raise ValueError("No enabled run specifications remain after CLI filtering.")
    policy = ParallelPolicy(
        cpu_budget=int(args.cpu_budget),
        workers=args.workers,
        threads_per_worker=args.threads_per_worker,
        enabled=not bool(args.no_parallel),
        memory_fraction=float(args.memory_fraction),
    )
    root_seed = int(config["root_seed"] if args.seed is None else args.seed)
    summaries = [
        _run_case(
            spec,
            output_root=output_root,
            policy=policy,
            root_seed=root_seed,
            singular_tol=float(config.get("singular_tol", 1e-12)),
            failure_mode=str(config.get("failure_mode", "censor")),
        )
        for spec in specs
    ]
    write_csv_atomic(output_root / "campaign_summary.csv", summaries)
    write_json_atomic(
        output_root / "campaign_manifest.json",
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "canonical_dynamics_entry_point": CANONICAL_CPU_ENTRY_POINT,
            "config_path": str(args.config.resolve()),
            "config": config,
            "cli": vars(args),
            "root_seed": root_seed,
            "runs": summaries,
            "git": _git_metadata(),
        },
    )
    print(f"Completed {len(summaries)} tangent-edge run(s): {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
