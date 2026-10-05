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
from record_anisotropy import (
    OBSERVABLES,
    WallRecordObserver,
    alpha_from_match,
    bootstrap_match,
    connected_correlations,
    match_time,
    stationarity_diagnostic,
)


CANONICAL_ENTRY = "classA_U1FGTN.run_markov_circuit"


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


def git_metadata() -> dict[str, Any]:
    def call(*args: str) -> str:
        return subprocess.run(
            args, cwd=REPO_ROOT, capture_output=True, text=True, check=False
        ).stdout.strip()

    return {
        "commit": call("git", "rev-parse", "HEAD"),
        "dirty": bool(call("git", "status", "--short")),
    }


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[6, 8, 10])
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--burn-multiplier", type=int, default=2)
    parser.add_argument("--record-multiplier", type=int, default=6)
    parser.add_argument("--max-lag-multiplier", type=float, default=3.0)
    parser.add_argument("--bootstrap-draws", type=int, default=1000)
    parser.add_argument("--root-seed", type=int, default=20260826)
    parser.add_argument("--nx", type=int, default=20)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


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
        dw_truncation=False,
    )
    model.construct_OW_projectors(
        nshell=1, DW=True, trial_orbitals="X", dw_truncation=False
    )
    return model


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
            "figure.dpi": 150,
            "savefig.dpi": 300,
        }
    )


def make_figure(results: list[dict[str, Any]], output: Path) -> None:
    configure_plotting()
    figure, axes = plt.subplots(2, 2, figsize=(7.05, 4.5), constrained_layout=True)
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(results)))
    for row, observable in enumerate(OBSERVABLES):
        for color, result in zip(colors, results):
            item = result["observables"][observable]
            temporal = np.asarray(item["temporal_mean"], dtype=float)
            scale = temporal[0]
            lag = np.arange(temporal.size) / result["ny"]
            axes[row, 0].plot(lag, temporal / scale, color=color, label=f"L={result['ny']}")
            axes[row, 0].axhline(item["spatial_target"] / scale, color=color, ls="--", alpha=0.7)
            if item["t_star"] is not None:
                axes[row, 0].plot(item["t_star"] / result["ny"], item["spatial_target"] / scale, "o", color=color, ms=3)

        resolved = [result for result in results if result["observables"][observable]["alpha"] is not None]
        if resolved:
            sizes = np.asarray([result["ny"] for result in resolved], dtype=float)
            alpha = np.asarray([result["observables"][observable]["alpha"] for result in resolved])
            low = np.asarray([result["observables"][observable]["alpha_ci_low"] for result in resolved])
            high = np.asarray([result["observables"][observable]["alpha_ci_high"] for result in resolved])
            finite_error = np.isfinite(low) & np.isfinite(high)
            axes[row, 1].plot(1.0 / sizes, alpha, "o-", color="black", ms=3)
            if np.any(finite_error):
                axes[row, 1].errorbar(
                    1.0 / sizes[finite_error],
                    alpha[finite_error],
                    yerr=np.vstack((alpha[finite_error] - low[finite_error], high[finite_error] - alpha[finite_error])),
                    fmt="none",
                    color="black",
                    capsize=2,
                )
        else:
            axes[row, 1].text(
                0.5,
                0.5,
                "unresolved\n(no non-contact crossing)",
                ha="center",
                va="center",
                transform=axes[row, 1].transAxes,
                color="0.35",
            )
        axes[row, 0].set_ylabel(f"{observable.replace('_', ' ')}\n$C(0,t)/C(0,0)$")
        axes[row, 1].set_ylabel(r"$\alpha(L)$")
        axes[row, 1].set_xlabel(r"$1/L$")
        axes[row, 0].set_xlabel(r"$t/L$")
        axes[row, 0].axhline(0.0, color="0.75", lw=0.6)
    axes[0, 0].legend(frameon=False, ncol=max(1, len(results)))
    axes[0, 0].set_title("solid: temporal; dashed: spatial target")
    axes[0, 1].set_title(r"$\alpha=L\,\mathrm{asinh}(1)/(\pi t_*)$")
    for label, axis in zip("abcd", axes.flat):
        axis.text(-0.12, 1.04, f"({label})", transform=axis.transAxes, fontweight="bold")
    figure.savefig(output / "anisotropy_pilot.png")
    figure.savefig(output / "anisotropy_pilot.pdf")
    plt.close(figure)


def nullable(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def main() -> None:
    args = parse_args()
    sizes = sorted(set(int(value) for value in args.sizes))
    if any(value < 4 or value % 2 for value in sizes):
        raise ValueError("pilot sizes must be even integers >=4 so L/2 is a lattice separation")
    if args.samples < 2:
        raise ValueError("at least two trajectories are required")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = args.output or EXPERIMENT / "outputs" / f"pilot_{stamp}"
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)

    config = {
        "experiment": "record_anisotropy_cpu_pilot",
        "purpose": "Born-record spacetime anisotropy; not prepared-state velocity",
        "nx": int(args.nx),
        "sizes_ny": sizes,
        "samples_per_size": int(args.samples),
        "burn_multiplier": int(args.burn_multiplier),
        "record_multiplier": int(args.record_multiplier),
        "max_lag_multiplier": float(args.max_lag_multiplier),
        "bootstrap_draws": int(args.bootstrap_draws),
        "root_seed": int(args.root_seed),
        "model": {
            "DW": True,
            "nshell": 1,
            "filling_frac": 0.5,
            "alpha_1": 1.0,
            "alpha_2": 30.0,
            "trial_orbitals": "X",
            "dw_truncation": False,
        },
        "run": {
            "sequence": "random",
            "perfect_correction": True,
            "meas_slab_only": False,
            "init_mode": "default",
            "state_representation": "physical_frame",
            "G_history": False,
            "save": False,
            "parallelize_samples": False,
        },
        "canonical_dynamics_entry_point": CANONICAL_ENTRY,
        "created_utc": utc_now(),
        "git": git_metadata(),
        "source_sha256": {
            "canonical_engine": sha256(REPO_ROOT / "src/fgtn/classA_U1FGTN.py"),
            "observer_analysis": sha256(EXPERIMENT / "record_anisotropy.py"),
            "runner": sha256(Path(__file__)),
        },
    }
    write_json(output / "manifest.json", config)
    print(f"output={output}", flush=True)

    seed_sequences = np.random.SeedSequence(args.root_seed).spawn(len(sizes))
    results: list[dict[str, Any]] = []
    started = time.perf_counter()
    for size_index, (ny, seed_sequence) in enumerate(zip(sizes, seed_sequences)):
        case_started = time.perf_counter()
        seed = int(seed_sequence.generate_state(1, dtype=np.uint64)[0])
        burn_in = int(args.burn_multiplier) * ny
        record_cycles = int(args.record_multiplier) * ny
        cycles = burn_in + record_cycles
        model = build_model(args.nx, ny)
        wall_x = tuple(int(value) for value in model.DW_loc)
        observer = WallRecordObserver(
            nx=args.nx,
            ny=ny,
            cycles=cycles,
            samples=args.samples,
            wall_x=wall_x,
        )
        print(
            f"[{size_index + 1}/{len(sizes)}] Nx={args.nx} Ny={ny} samples={args.samples} "
            f"burn={burn_in} record={record_cycles} walls={wall_x} seed={seed}",
            flush=True,
        )
        model.run_markov_circuit(
            cycles=cycles,
            samples=args.samples,
            sequence="random",
            perfect_correction=True,
            G_history=False,
            save=False,
            progress=False,
            random_seed=seed,
            state_representation="physical_frame",
            trajectory_weight_observer=observer,
            meas_slab_only=False,
            parallelize_samples=False,
            init_mode="default",
        )
        observer.assert_complete()
        payload = observer.payload()
        raw_path = output / f"raw_Nx{args.nx}_Ny{ny}.npz"
        save_npz(
            raw_path,
            **payload,
            nx=np.asarray(args.nx),
            ny=np.asarray(ny),
            cycles=np.asarray(cycles),
            burn_in=np.asarray(burn_in),
            samples=np.asarray(args.samples),
            random_seed=np.asarray(seed, dtype=np.uint64),
            canonical_dynamics_entry_point=np.asarray(CANONICAL_ENTRY),
        )

        case: dict[str, Any] = {
            "nx": int(args.nx),
            "ny": ny,
            "wall_x": list(wall_x),
            "cycles": cycles,
            "burn_in": burn_in,
            "record_cycles": record_cycles,
            "samples": int(args.samples),
            "random_seed": seed,
            "raw_file": raw_path.name,
            "raw_sha256": sha256(raw_path),
            "observables": {},
        }
        max_lag = min(int(round(args.max_lag_multiplier * ny)), record_cycles - 1)
        for observable_index, observable in enumerate(OBSERVABLES):
            field = np.asarray(payload[observable], dtype=np.float64)
            estimate = connected_correlations(
                field, burn_in=burn_in, max_temporal_lag=max_lag
            )
            spatial_target = float(estimate.spatial_mean[ny // 2])
            t_star = match_time(estimate.temporal_mean, spatial_target)
            alpha = alpha_from_match(ny, t_star)
            bootstrap = bootstrap_match(
                estimate,
                circumference=ny,
                draws=args.bootstrap_draws,
                seed=args.root_seed + 10000 * (size_index + 1) + observable_index,
            )
            correlation_path = output / f"correlations_{observable}_Ny{ny}.npz"
            save_npz(
                correlation_path,
                spatial_by_trajectory=estimate.spatial_by_trajectory,
                temporal_by_trajectory=estimate.temporal_by_trajectory,
                spatial_mean=estimate.spatial_mean,
                temporal_mean=estimate.temporal_mean,
                mean_by_trajectory_wall=estimate.mean_by_trajectory_wall,
                spatial_product_by_trajectory_wall=estimate.spatial_product_by_trajectory_wall,
                temporal_product_by_trajectory_wall=estimate.temporal_product_by_trajectory_wall,
                t_star_bootstrap=bootstrap.pop("t_star_bootstrap"),
                alpha_bootstrap=bootstrap.pop("alpha_bootstrap"),
            )
            case["observables"][observable] = {
                "spatial_target": spatial_target,
                "temporal_variance": float(estimate.temporal_mean[0]),
                "t_star": nullable(t_star),
                "alpha": nullable(alpha),
                "cycles_per_circumference": nullable(1.0 / alpha),
                "spatial_mean": estimate.spatial_mean,
                "temporal_mean": estimate.temporal_mean,
                "correlation_file": correlation_path.name,
                "correlation_sha256": sha256(correlation_path),
                "stationarity": stationarity_diagnostic(field, burn_in=burn_in),
                **{key: nullable(value) if isinstance(value, float) else value for key, value in bootstrap.items()},
            }
        case["wall_time_seconds"] = time.perf_counter() - case_started
        results.append(case)
        write_json(output / "results.json", {"config": config, "sizes": results})
        print(f"Ny={ny} complete in {case['wall_time_seconds']:.1f}s", flush=True)

    make_figure(results, output)
    total_time = time.perf_counter() - started
    write_json(
        output / "results.json",
        {"config": config, "sizes": results, "total_wall_time_seconds": total_time},
    )
    lines = [
        "# Record anisotropy CPU pilot results",
        "",
        f"Canonical dynamics: `{CANONICAL_ENTRY}`.",
        "",
        "This estimates the Born-record anisotropy, not the prepared-state velocity.",
        "",
        "| observable | L | t* | alpha | cycles/L=1/alpha | bootstrap resolved |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for case in results:
        for observable in OBSERVABLES:
            item = case["observables"][observable]
            fmt = lambda value: "unresolved" if value is None else f"{value:.4g}"
            lines.append(
                f"| {observable} | {case['ny']} | {fmt(item['t_star'])} | {fmt(item['alpha'])} | "
                f"{fmt(item['cycles_per_circumference'])} | {item['bootstrap_resolved_fraction']:.3f} |"
            )
    lines.extend(
        [
            "",
            "Matching rule: `C(L/2,0)=C(0,t*)`, with linear interpolation of the first raw temporal crossing.",
            "Bootstrap resampling is by complete trajectory; the two walls remain paired.",
            "An unresolved entry means the positive spatial target had no crossing in the preregistered lag window.",
            "",
            f"Total wall time: {total_time:.1f} s.",
        ]
    )
    (output / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    (output / "SUCCESS").write_text(utc_now() + "\n")
    print(f"complete in {total_time:.1f}s; summary={output / 'SUMMARY.md'}", flush=True)


if __name__ == "__main__":
    main()
