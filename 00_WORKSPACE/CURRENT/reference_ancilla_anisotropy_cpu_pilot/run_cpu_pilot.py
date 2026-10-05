#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EXPERIMENT = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT.parents[2]
sys.path[:0] = [str(REPO_ROOT / "src"), str(EXPERIMENT)]

from fgtn.classA_U1FGTN import classA_U1FGTN
from reference_probe import (
    ReferencePairObserver,
    anisotropy_from_matching_time,
    matching_time,
)


CANONICAL_ENTRY = "classA_U1FGTN.run_markov_circuit"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[6, 8])
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--equilibration-multiplier", type=int, default=4)
    parser.add_argument("--follow-multiplier", type=int, default=2)
    parser.add_argument("--max-separation-multiplier", type=float, default=0.75)
    parser.add_argument("--bootstrap-draws", type=int, default=1000)
    parser.add_argument("--root-seed", type=int, default=20260827)
    parser.add_argument("--nx", type=int, default=20)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n")
    os.replace(temporary, path)


def save_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_metadata() -> dict[str, Any]:
    def call(*args: str) -> str:
        return subprocess.run(
            args, cwd=REPO_ROOT, capture_output=True, text=True, check=False
        ).stdout.strip()

    return {
        "commit": call("git", "rev-parse", "HEAD"),
        "dirty": bool(call("git", "status", "--short")),
    }


def build_model(nx: int, ny: int) -> classA_U1FGTN:
    model = classA_U1FGTN(
        nx,
        ny,
        DW=True,
        nshell=1,
        filling_frac=0.5,
        alpha_1=1.0,
        alpha_2=30.0,
        trial_orbitals="X",
        dw_truncation=True,
    )
    model.construct_OW_projectors(
        nshell=1, DW=True, trial_orbitals="X", dw_truncation=True
    )
    return model


def sample_seeds(root_seed: int, ny: int, samples: int) -> list[tuple[int, int, int]]:
    size_sequence = np.random.SeedSequence([int(root_seed), int(ny)])
    result = []
    for child in size_sequence.spawn(int(samples)):
        engine, probe, position = child.generate_state(3, dtype=np.uint64)
        result.append((int(engine), int(probe), int(position)))
    return result


def run_configuration(
    *,
    nx: int,
    ny: int,
    samples: int,
    tau1: int,
    delta_tau: int,
    follow_cycles: int,
    root_seed: int,
) -> dict[str, Any]:
    shape = (samples, follow_cycles + 1)
    mutual_information = np.full(shape, np.nan, dtype=np.float64)
    entropy_r1 = np.full(shape, np.nan, dtype=np.float64)
    entropy_r2 = np.full(shape, np.nan, dtype=np.float64)
    entropy_r12 = np.full(shape, np.nan, dtype=np.float64)
    absolute_cycles = np.full(shape, -1, dtype=np.int64)
    insertion_probability = np.full((samples, 2, 2), np.nan, dtype=np.float64)
    insertion_selected_probability = np.full((samples, 2, 2), np.nan, dtype=np.float64)
    insertion_outcome = np.zeros((samples, 2, 2), dtype=np.bool_)
    sites = np.full((samples, 2, 2), -1, dtype=np.int64)
    wall_x = np.full(samples, -1, dtype=np.int64)
    y1_values = np.full(samples, -1, dtype=np.int64)
    engine_seeds = np.full(samples, 0, dtype=np.uint64)
    probe_seeds = np.full(samples, 0, dtype=np.uint64)
    tau2 = int(tau1) + int(delta_tau)
    final_cycle = tau2 + int(follow_cycles)

    for sample_index, (engine_seed, probe_seed, position_seed) in enumerate(
        sample_seeds(root_seed, ny, samples)
    ):
        model = build_model(nx, ny)
        walls = tuple(int(value) for value in model.DW_loc)
        x = walls[sample_index % len(walls)]
        y1 = int(np.random.default_rng(position_seed).integers(0, ny))
        y2 = y1 if delta_tau > 0 else (y1 + ny // 2) % ny
        first_site = (x, y1)
        second_site = (x, y2)
        observer = ReferencePairObserver(
            nx=nx,
            ny=ny,
            tau1=tau1,
            tau2=tau2,
            follow_cycles=follow_cycles,
            first_site=first_site,
            second_site=second_site,
            rng=np.random.default_rng(probe_seed),
        )
        result = model.run_markov_circuit(
            cycles=final_cycle,
            samples=1,
            sequence="random",
            perfect_correction=True,
            G_history=False,
            save=False,
            progress=False,
            random_seed=engine_seed,
            state_representation="physical_frame",
            return_native_state=True,
            native_cycle_observer=observer,
            meas_slab_only=True,
            parallelize_samples=False,
            init_mode="default",
        )
        payload = observer.payload()
        mutual_information[sample_index] = payload["mutual_information"]
        entropy_r1[sample_index] = payload["entropy_r1"]
        entropy_r2[sample_index] = payload["entropy_r2"]
        entropy_r12[sample_index] = payload["entropy_r12"]
        absolute_cycles[sample_index] = payload["cycles"]
        for reference_index, reference in enumerate(
            (payload["reference_one"], payload["reference_two"])
        ):
            sites[sample_index, reference_index] = (reference["x"], reference["y"])
            for orbital_index, event in enumerate(reference["events"]):
                insertion_probability[sample_index, reference_index, orbital_index] = event[
                    "probability_occupied"
                ]
                insertion_selected_probability[
                    sample_index, reference_index, orbital_index
                ] = event["selected_probability"]
                insertion_outcome[sample_index, reference_index, orbital_index] = event[
                    "outcome_occupied"
                ]
        final = result["native_final"]
        if int(final["physical_dimension"]) != 2 * nx * ny + 4:
            raise RuntimeError("final state does not contain exactly four reference modes")
        if float(final["gram_residual"]) > 1e-8:
            raise RuntimeError("augmented occupied frame lost orthonormality")
        wall_x[sample_index] = x
        y1_values[sample_index] = y1
        engine_seeds[sample_index] = np.uint64(engine_seed)
        probe_seeds[sample_index] = np.uint64(probe_seed)

    if not np.all(np.isfinite(mutual_information)) or np.any(absolute_cycles < 0):
        raise RuntimeError("configuration output is incomplete")
    return {
        "schema": np.asarray("gaussian_reference_pair_anisotropy_v1"),
        "nx": np.asarray(nx),
        "ny": np.asarray(ny),
        "samples": np.asarray(samples),
        "tau1": np.asarray(tau1),
        "tau2": np.asarray(tau2),
        "delta_tau": np.asarray(delta_tau),
        "follow_cycles": np.asarray(follow_cycles),
        "absolute_cycles": absolute_cycles,
        "mutual_information": mutual_information,
        "entropy_r1": entropy_r1,
        "entropy_r2": entropy_r2,
        "entropy_r12": entropy_r12,
        "insertion_probability": insertion_probability,
        "insertion_selected_probability": insertion_selected_probability,
        "insertion_outcome": insertion_outcome,
        "sites": sites,
        "wall_x": wall_x,
        "y1": y1_values,
        "engine_seed": engine_seeds,
        "probe_seed": probe_seeds,
        "canonical_dynamics_entry_point": np.asarray(CANONICAL_ENTRY),
    }


def plateau_rows(payload: dict[str, Any], window: int) -> np.ndarray:
    values = np.asarray(payload["mutual_information"], dtype=np.float64)
    return np.mean(values[:, -int(window) :], axis=1)


def analyze_size(
    *,
    ny: int,
    payloads: dict[int, dict[str, Any]],
    bootstrap_draws: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    window = int(ny)
    spatial_rows = plateau_rows(payloads[0], window)
    separations = np.asarray(sorted(value for value in payloads if value > 0), dtype=float)
    temporal_rows = np.stack(
        [plateau_rows(payloads[int(separation)], window) for separation in separations],
        axis=1,
    )
    spatial_mean = float(np.mean(spatial_rows))
    temporal_mean = np.mean(temporal_rows, axis=0)
    time_star = matching_time(separations, temporal_mean, spatial_mean)
    alpha = anisotropy_from_matching_time(ny, time_star)

    rng = np.random.default_rng(int(bootstrap_seed))
    samples = spatial_rows.size
    boot_time = np.full(int(bootstrap_draws), np.nan, dtype=np.float64)
    boot_alpha = np.full(int(bootstrap_draws), np.nan, dtype=np.float64)
    for draw in range(int(bootstrap_draws)):
        indices = rng.integers(0, samples, size=samples)
        boot_time[draw] = matching_time(
            separations,
            np.mean(temporal_rows[indices], axis=0),
            float(np.mean(spatial_rows[indices])),
        )
        boot_alpha[draw] = anisotropy_from_matching_time(ny, boot_time[draw])
    finite = np.isfinite(boot_alpha)

    def coefficient_of_variation(values: np.ndarray) -> float | None:
        mean = float(np.mean(values))
        if mean <= 0.0 or values.size < 2:
            return None
        return float(np.std(values, ddof=1) / mean)

    spatial_cv = coefficient_of_variation(spatial_rows)
    temporal_cv = [coefficient_of_variation(temporal_rows[:, index]) for index in range(temporal_rows.shape[1])]
    relative_sem_target = 0.20
    samples_for_target = lambda cv: None if cv is None else int(np.ceil((cv / relative_sem_target) ** 2))

    plateau_drift = {}
    for separation, payload in payloads.items():
        values = np.asarray(payload["mutual_information"], dtype=np.float64)
        previous = np.mean(values[:, -2 * window : -window], axis=1)
        final = np.mean(values[:, -window:], axis=1)
        difference = final - previous
        plateau_drift[str(separation)] = {
            "previous_window_mean": float(np.mean(previous)),
            "final_window_mean": float(np.mean(final)),
            "paired_mean_change": float(np.mean(difference)),
            "paired_change_standard_error": (
                float(np.std(difference, ddof=1) / np.sqrt(samples)) if samples > 1 else None
            ),
        }

    return {
        "ny": int(ny),
        "plateau_window_cycles": window,
        "spatial_by_sample": spatial_rows,
        "spatial_mean": spatial_mean,
        "temporal_separations": separations,
        "temporal_by_sample": temporal_rows,
        "temporal_mean": temporal_mean,
        "time_star": None if not np.isfinite(time_star) else float(time_star),
        "alpha": None if not np.isfinite(alpha) else float(alpha),
        "bootstrap_resolved_fraction": float(np.mean(finite)),
        "time_star_ci_low": float(np.nanpercentile(boot_time, 2.5)) if np.any(finite) else None,
        "time_star_ci_high": float(np.nanpercentile(boot_time, 97.5)) if np.any(finite) else None,
        "alpha_ci_low": float(np.nanpercentile(boot_alpha, 2.5)) if np.any(finite) else None,
        "alpha_ci_high": float(np.nanpercentile(boot_alpha, 97.5)) if np.any(finite) else None,
        "bootstrap_time_star": boot_time,
        "bootstrap_alpha": boot_alpha,
        "sampling_diagnostics": {
            "relative_sem_target": relative_sem_target,
            "spatial_coefficient_of_variation": spatial_cv,
            "temporal_coefficient_of_variation": temporal_cv,
            "spatial_samples_estimated_for_target": samples_for_target(spatial_cv),
            "temporal_samples_estimated_for_target": [samples_for_target(value) for value in temporal_cv],
        },
        "plateau_drift": plateau_drift,
    }


def configure_plotting() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["CMU Sans Serif", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "legend.fontsize": 6.5,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "lines.linewidth": 1.1,
            "savefig.dpi": 300,
        }
    )


def make_figure(
    *, output: Path, size_payloads: dict[int, dict[int, dict[str, Any]]], analyses: list[dict[str, Any]]
) -> None:
    configure_plotting()
    figure, axes = plt.subplots(2, len(analyses), figsize=(7.05, 4.8), constrained_layout=True)
    if len(analyses) == 1:
        axes = axes[:, None]
    for column, analysis in enumerate(analyses):
        ny = int(analysis["ny"])
        payloads = size_payloads[ny]
        separations_to_show = [0]
        if analysis["time_star"] is None:
            separations_to_show += [int(value) for value in analysis["temporal_separations"][:2]]
        else:
            ordered = sorted(
                analysis["temporal_separations"], key=lambda value: abs(value - analysis["time_star"])
            )
            separations_to_show += [int(value) for value in ordered[:2]]
        for separation in separations_to_show:
            values = np.asarray(payloads[separation]["mutual_information"], dtype=float)
            x = np.arange(values.shape[1]) / ny
            mean = np.mean(values, axis=0)
            error = np.std(values, axis=0, ddof=1) / np.sqrt(values.shape[0])
            label = "space, $L/2$" if separation == 0 else rf"time, $\delta\tau={separation}$"
            axes[0, column].plot(x, mean, label=label)
            axes[0, column].fill_between(x, mean - error, mean + error, alpha=0.18)
        axes[0, column].axvspan(1.0, 2.0, color="0.85", alpha=0.4, label="plateau window")
        axes[0, column].set_title(f"L={ny}: post-insertion evolution")
        axes[0, column].set_xlabel(r"follow time$/L$")
        axes[0, column].set_ylabel(r"$I(R_1:R_2)$ [nats]")
        axes[0, column].legend(frameon=False)

        separations = np.asarray(analysis["temporal_separations"], dtype=float)
        temporal_rows = np.asarray(analysis["temporal_by_sample"], dtype=float)
        mean = np.mean(temporal_rows, axis=0)
        error = np.std(temporal_rows, axis=0, ddof=1) / np.sqrt(temporal_rows.shape[0])
        axes[1, column].errorbar(
            separations / ny, mean, yerr=error, fmt="o-", color="#2166ac", capsize=2, label="temporal"
        )
        spatial_rows = np.asarray(analysis["spatial_by_sample"], dtype=float)
        spatial_mean = float(np.mean(spatial_rows))
        spatial_error = float(np.std(spatial_rows, ddof=1) / np.sqrt(spatial_rows.size))
        axes[1, column].axhline(spatial_mean, color="#b2182b", ls="--", label="spatial $L/2$")
        axes[1, column].fill_between(
            [0.0, max(separations / ny)],
            spatial_mean - spatial_error,
            spatial_mean + spatial_error,
            color="#b2182b",
            alpha=0.12,
        )
        if analysis["time_star"] is not None:
            axes[1, column].axvline(analysis["time_star"] / ny, color="black", ls=":", label=r"$t_*$")
        axes[1, column].set_xlabel(r"$\delta\tau/L$")
        axes[1, column].set_ylabel(r"plateau $I(R_1:R_2)$ [nats]")
        axes[1, column].legend(frameon=False)
    for label, axis in zip("abcd", axes.flat):
        axis.text(-0.14, 1.04, f"({label})", transform=axis.transAxes, fontweight="bold")
        axis.tick_params(direction="in", top=True, right=True)
    figure.savefig(output / "reference_anisotropy_pilot.pdf")
    figure.savefig(output / "reference_anisotropy_pilot.png")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    sizes = sorted(set(int(value) for value in args.sizes))
    if any(value < 4 or value % 2 for value in sizes):
        raise ValueError("sizes must be even integers >=4")
    if args.samples < 2:
        raise ValueError("at least two trajectories are required")
    if args.follow_multiplier < 2:
        raise ValueError("follow_multiplier must be at least 2 to define two L-cycle plateau windows")
    output = args.output or EXPERIMENT / "outputs" / datetime.now().strftime("pilot_%Y%m%d_%H%M%S")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    config = {
        "experiment": "gaussian_reference_ancilla_anisotropy_cpu_pilot",
        "sizes_ny": sizes,
        "nx": int(args.nx),
        "samples_per_configuration": int(args.samples),
        "equilibration_multiplier": int(args.equilibration_multiplier),
        "follow_multiplier": int(args.follow_multiplier),
        "max_separation_multiplier": float(args.max_separation_multiplier),
        "bootstrap_draws": int(args.bootstrap_draws),
        "root_seed": int(args.root_seed),
        "probe": "two canonical cell modes Bell-paired with two spectator reference fermions",
        "model": {
            "DW": True,
            "nshell": 1,
            "filling_frac": 0.5,
            "alpha_1": 1.0,
            "alpha_2": 30.0,
            "trial_orbitals": "X",
            "dw_truncation": True,
        },
        "run": {
            "sequence": "random",
            "perfect_correction": True,
            "meas_slab_only": True,
            "init_mode": "default",
            "state_representation": "physical_frame",
            "G_history": False,
            "save": False,
        },
        "canonical_dynamics_entry_point": CANONICAL_ENTRY,
        "created_utc": utc_now(),
        "git": git_metadata(),
        "source_sha256": {
            "canonical_engine": sha256(REPO_ROOT / "src/fgtn/classA_U1FGTN.py"),
            "occupied_frame": sha256(REPO_ROOT / "src/fgtn/occupied_frame.py"),
            "probe": sha256(EXPERIMENT / "reference_probe.py"),
            "runner": sha256(Path(__file__)),
        },
    }
    write_json(output / "manifest.json", config)
    print(f"output={output}", flush=True)

    size_payloads: dict[int, dict[int, dict[str, Any]]] = {}
    analyses: list[dict[str, Any]] = []
    raw_index: list[dict[str, Any]] = []
    started = time.perf_counter()
    for size_index, ny in enumerate(sizes):
        tau1 = int(args.equilibration_multiplier) * ny
        follow_cycles = int(args.follow_multiplier) * ny
        max_separation = max(2, int(np.ceil(args.max_separation_multiplier * ny)))
        separations = [0] + list(range(1, max_separation + 1))
        size_payloads[ny] = {}
        for config_index, separation in enumerate(separations):
            config_started = time.perf_counter()
            label = "space" if separation == 0 else f"time_dt{separation}"
            print(
                f"[L={ny} {config_index + 1}/{len(separations)}] {label}, "
                f"samples={args.samples}, tau1={tau1}, follow={follow_cycles}",
                flush=True,
            )
            payload = run_configuration(
                nx=args.nx,
                ny=ny,
                samples=args.samples,
                tau1=tau1,
                delta_tau=separation,
                follow_cycles=follow_cycles,
                root_seed=args.root_seed,
            )
            size_payloads[ny][separation] = payload
            raw_path = output / f"raw_Nx{args.nx}_Ny{ny}_{label}.npz"
            save_npz(raw_path, **payload)
            row = {
                "ny": ny,
                "separation": separation,
                "kind": "spatial" if separation == 0 else "temporal",
                "raw_file": raw_path.name,
                "raw_sha256": sha256(raw_path),
                "wall_time_seconds": time.perf_counter() - config_started,
            }
            raw_index.append(row)
            write_json(output / "raw_index.json", raw_index)
            print(f"{label} complete in {row['wall_time_seconds']:.1f}s", flush=True)

        analysis = analyze_size(
            ny=ny,
            payloads=size_payloads[ny],
            bootstrap_draws=args.bootstrap_draws,
            bootstrap_seed=args.root_seed + 1000 * (size_index + 1),
        )
        bootstrap_path = output / f"bootstrap_Ny{ny}.npz"
        save_npz(
            bootstrap_path,
            time_star=analysis.pop("bootstrap_time_star"),
            alpha=analysis.pop("bootstrap_alpha"),
        )
        analysis["bootstrap_file"] = bootstrap_path.name
        analysis["bootstrap_sha256"] = sha256(bootstrap_path)
        analyses.append(analysis)
        write_json(output / "results.json", {"config": config, "sizes": analyses, "raw_index": raw_index})

    total_time = time.perf_counter() - started
    make_figure(output=output, size_payloads=size_payloads, analyses=analyses)
    write_json(
        output / "results.json",
        {"config": config, "sizes": analyses, "raw_index": raw_index, "total_wall_time_seconds": total_time},
    )
    lines = [
        "# Gaussian reference-ancilla anisotropy CPU pilot",
        "",
        f"Canonical dynamics: `{CANONICAL_ENTRY}`.",
        "",
        "| L | I_space | t* | alpha | bootstrap resolved | alpha 95% interval |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for analysis in analyses:
        fmt = lambda value: "unresolved" if value is None else f"{value:.4g}"
        interval = (
            "unresolved"
            if analysis["alpha_ci_low"] is None
            else f"[{analysis['alpha_ci_low']:.4g}, {analysis['alpha_ci_high']:.4g}]"
        )
        lines.append(
            f"| {analysis['ny']} | {analysis['spatial_mean']:.4g} | {fmt(analysis['time_star'])} | "
            f"{fmt(analysis['alpha'])} | {analysis['bootstrap_resolved_fraction']:.3f} | {interval} |"
        )
    lines.extend(
        [
            "",
            f"Plateau values average the final `L` cycles after following both references for `{args.follow_multiplier}L` cycles.",
            "Bootstrap resampling uses matched complete-trajectory indices across spatial and temporal configurations.",
            "These small sizes and four trajectories are a feasibility pilot, not a production uncertainty estimate.",
            "",
            f"Total wall time: {total_time:.1f} s.",
        ]
    )
    (output / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    (output / "SUCCESS").write_text(utc_now() + "\n")
    print(f"complete in {total_time:.1f}s; summary={output / 'SUMMARY.md'}", flush=True)


if __name__ == "__main__":
    main()
