#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import tempfile
import os
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from twist_torus import analyze_frame_surface

PROTOCOLS = ("fresh_full_record", "fresh_outcomes")
LABELS = {
    "target": "exact target",
    "fresh_full_record": "fresh schedule + outcomes",
    "fresh_outcomes": "fixed schedule + fresh outcomes",
}


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, suffix=".json.tmp")
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def save_npz_atomic(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, suffix=".npz")
    os.close(descriptor)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def trajectory_path(root: Path, protocol: str, ix: int, iy: int) -> Path:
    return root / "raw/trajectories" / protocol / f"tx_{ix:02d}_ty_{iy:02d}/trajectory.npz"


def target_path(root: Path, ix: int, iy: int) -> Path:
    return root / "raw/target" / f"tx_{ix:02d}_ty_{iy:02d}.npz"


def pack_ragged(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flat = []
    offsets = [0]
    for index in np.ndindex(values.shape):
        item = np.asarray(values[index], dtype=np.float64).reshape(-1)
        flat.append(item)
        offsets.append(offsets[-1] + item.size)
    return (
        np.concatenate(flat) if flat else np.empty(0, dtype=np.float64),
        np.asarray(offsets, dtype=np.int64),
    )


def finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def surface_summary(result: dict[str, Any]) -> dict[str, Any]:
    minimum = np.minimum(result["minimum_singular_x"], result["minimum_singular_y"])
    finite_log = np.concatenate(
        [
            result["log_abs_determinant_x"][np.isfinite(result["log_abs_determinant_x"])],
            result["log_abs_determinant_y"][np.isfinite(result["log_abs_determinant_y"])],
        ]
    )
    return {
        "classification": result["classification"],
        "chern": finite_or_none(result["chern"]),
        "constant_rank": bool(result["constant_rank"]),
        "minimum_rank": int(np.min(result["ranks"])),
        "maximum_rank": int(np.max(result["ranks"])),
        "unique_ranks": [int(value) for value in np.unique(result["ranks"])],
        "valid_link_fraction": float(result["valid_link_fraction"]),
        "minimum_link_singular_value": float(np.min(minimum)),
        "minimum_finite_log_abs_determinant": (
            float(np.min(finite_log)) if finite_log.size else None
        ),
    }


def analyze(root: Path) -> dict[str, Any]:
    config = json.loads((root / "campaign_config.v1.json").read_text())
    g = config["geometry"]
    gx, gy = int(g["twist_grid_x"]), int(g["twist_grid_y"])
    cycles = int(g["cycles"])
    nx, ny = int(g["Nx"]), int(g["Ny"])
    tolerance = float(config["acceptance"]["link_singular_value_hard"])
    integer_tolerance = float(config["acceptance"]["chern_integer_hard"])

    frames: dict[str, np.ndarray] = {
        name: np.empty((gx, gy), dtype=object) for name in ("target", *PROTOCOLS)
    }
    target_marker = np.empty((gx, gy), dtype=np.float64)
    cycle_arrays = {
        protocol: {
            key: np.empty((gx, gy, cycles + 1), dtype=np.float64)
            for key in (
                "real_space_chern",
                "half_entropy",
                "charge",
                "rank",
                "gram_residual",
                "cumulative_log_weight",
            )
        }
        for protocol in PROTOCOLS
    }
    wall_ns = {protocol: np.empty((gx, gy), dtype=np.int64) for protocol in PROTOCOLS}
    schedule_hash = {protocol: np.empty((gx, gy), dtype="U64") for protocol in PROTOCOLS}
    outcome_digest = {protocol: np.empty((gx, gy), dtype="U64") for protocol in PROTOCOLS}

    for ix in range(gx):
        for iy in range(gy):
            with np.load(target_path(root, ix, iy)) as data:
                frames["target"][ix, iy] = np.array(data["frame"], copy=True)
                target_marker[ix, iy] = float(data["real_space_chern"])
            for protocol in PROTOCOLS:
                path = trajectory_path(root, protocol, ix, iy)
                with np.load(path) as data:
                    frames[protocol][ix, iy] = np.array(data["final_frame"], copy=True)
                    for key in cycle_arrays[protocol]:
                        cycle_arrays[protocol][key][ix, iy] = data[key]
                summary = json.loads(path.with_name("summary.json").read_text())
                wall_ns[protocol][ix, iy] = int(summary["wall_ns"])
                schedule_hash[protocol][ix, iy] = summary["schedule_content_sha256"]
                outcome_digest[protocol][ix, iy] = summary["outcome_digest"]

    results = {
        name: analyze_frame_surface(
            frame_surface,
            nx=nx,
            ny=ny,
            singular_value_tolerance=tolerance,
            expected_chern=1.0,
            integer_tolerance=integer_tolerance,
        )
        for name, frame_surface in frames.items()
    }

    terminal = config["terminal_window"]
    terminal_slice = slice(int(terminal["start_cycle"]), int(terminal["stop_cycle_inclusive"]) + 1)
    marker_tolerance = float(config["acceptance"]["real_space_marker_hard"])
    topology = {}
    for protocol in PROTOCOLS:
        final_error = np.abs(cycle_arrays[protocol]["real_space_chern"][:, :, -1] - target_marker)
        terminal_error = np.abs(
            np.mean(cycle_arrays[protocol]["real_space_chern"][:, :, terminal_slice], axis=2)
            - target_marker
        )
        topology[protocol] = {
            "passed": bool(np.all(final_error <= marker_tolerance) and np.all(terminal_error <= marker_tolerance)),
            "maximum_final_marker_error": float(np.max(final_error)),
            "maximum_terminal_mean_marker_error": float(np.max(terminal_error)),
            "final_pass_fraction": float(np.mean(final_error <= marker_tolerance)),
            "terminal_pass_fraction": float(np.mean(terminal_error <= marker_tolerance)),
        }

    preliminary_support = bool(
        all(results[protocol]["classification"] == "defined_C1" for protocol in PROTOCOLS)
        and all(topology[protocol]["passed"] for protocol in PROTOCOLS)
    )
    summary = {
        "status": "complete",
        "geometry": {
            "Nx": nx,
            "Ny": ny,
            "twist_grid_x": gx,
            "twist_grid_y": gy,
            "cycles": cycles,
        },
        "surface_count": 1,
        "twist_points_are_independent_samples": False,
        "surfaces": {name: surface_summary(result) for name, result in results.items()},
        "real_space_topology": topology,
        "preliminary_fresh_record_support": preliminary_support,
    }
    write_json_atomic(root / "processed/analysis_summary.json", summary)

    aggregate: dict[str, Any] = {
        "target_real_space_chern": target_marker,
        "twist_x": 2 * np.pi * np.arange(gx) / gx,
        "twist_y": 2 * np.pi * np.arange(gy) / gy,
        "cycles": np.arange(cycles + 1),
    }
    for protocol in PROTOCOLS:
        for key, value in cycle_arrays[protocol].items():
            aggregate[f"{protocol}_{key}"] = value
        aggregate[f"{protocol}_wall_ns"] = wall_ns[protocol]
        aggregate[f"{protocol}_schedule_hash"] = schedule_hash[protocol]
        aggregate[f"{protocol}_outcome_digest"] = outcome_digest[protocol]
    for name, result in results.items():
        for key in (
            "ranks",
            "minimum_singular_x",
            "minimum_singular_y",
            "log_abs_determinant_x",
            "log_abs_determinant_y",
            "valid_x",
            "valid_y",
            "plaquette_phase",
        ):
            aggregate[f"{name}_{key}"] = result[key]
        for axis in ("x", "y"):
            values, offsets = pack_ragged(result[f"singular_values_{axis}"])
            aggregate[f"{name}_singular_values_{axis}"] = values
            aggregate[f"{name}_singular_offsets_{axis}"] = offsets
    save_npz_atomic(root / "processed/twist_torus_products.npz", **aggregate)

    table_path = root / "processed/twist_points.csv"
    with table_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "protocol",
                "ix",
                "iy",
                "twist_x",
                "twist_y",
                "final_rank",
                "final_real_space_chern",
                "half_entropy",
                "minimum_neighbor_singular_value",
                "wall_seconds",
            ),
        )
        writer.writeheader()
        for protocol in PROTOCOLS:
            minimum = np.minimum(
                results[protocol]["minimum_singular_x"],
                results[protocol]["minimum_singular_y"],
            )
            for ix, iy in np.ndindex(gx, gy):
                writer.writerow(
                    {
                        "protocol": protocol,
                        "ix": ix,
                        "iy": iy,
                        "twist_x": 2 * np.pi * ix / gx,
                        "twist_y": 2 * np.pi * iy / gy,
                        "final_rank": int(cycle_arrays[protocol]["rank"][ix, iy, -1]),
                        "final_real_space_chern": cycle_arrays[protocol]["real_space_chern"][ix, iy, -1],
                        "half_entropy": cycle_arrays[protocol]["half_entropy"][ix, iy, -1],
                        "minimum_neighbor_singular_value": minimum[ix, iy],
                        "wall_seconds": wall_ns[protocol][ix, iy] / 1e9,
                    }
                )

    make_figures(root, results, cycle_arrays, target_marker)
    write_report(root, summary)
    return summary


def make_figures(
    root: Path,
    results: dict[str, dict[str, Any]],
    cycle_arrays: dict[str, dict[str, np.ndarray]],
    target_marker: np.ndarray,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["CMU Sans Serif", "DejaVu Sans"],
            "font.size": 8,
            "axes.linewidth": 0.7,
            "xtick.direction": "in",
            "ytick.direction": "in",
        }
    )
    figure_dir = root / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    def save(fig: Any, name: str) -> None:
        fig.savefig(figure_dir / f"{name}.pdf", bbox_inches="tight")
        fig.savefig(figure_dir / f"{name}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(7.05, 3.0), constrained_layout=True)
    for ax, protocol in zip(axes, PROTOCOLS):
        image = ax.imshow(cycle_arrays[protocol]["rank"][:, :, -1].T, origin="lower", aspect="equal")
        ax.set(title=LABELS[protocol], xlabel=r"$\theta_x$ index", ylabel=r"$\theta_y$ index")
        fig.colorbar(image, ax=ax, label="final rank")
    save(fig, "final_rank_maps")

    fig, axes = plt.subplots(1, 3, figsize=(7.05, 2.5), constrained_layout=True)
    marker_maps = [target_marker] + [cycle_arrays[p]["real_space_chern"][:, :, -1] for p in PROTOCOLS]
    for ax, name, values in zip(axes, ("target", *PROTOCOLS), marker_maps):
        image = ax.imshow(values.T, origin="lower", vmin=0.94, vmax=1.01, cmap="viridis")
        ax.set(title=LABELS[name], xlabel=r"$\theta_x$ index", ylabel=r"$\theta_y$ index")
        fig.colorbar(image, ax=ax, label="real-space marker")
    save(fig, "final_real_space_chern_maps")

    fig, axes = plt.subplots(1, 2, figsize=(7.05, 3.0), constrained_layout=True)
    for ax, protocol in zip(axes, PROTOCOLS):
        image = ax.imshow(cycle_arrays[protocol]["half_entropy"][:, :, -1].T, origin="lower", cmap="magma")
        ax.set(title=LABELS[protocol], xlabel=r"$\theta_x$ index", ylabel=r"$\theta_y$ index")
        fig.colorbar(image, ax=ax, label="half-system entropy")
    save(fig, "final_entropy_maps")

    fig, axes = plt.subplots(2, 2, figsize=(7.05, 5.5), constrained_layout=True)
    for column, protocol in enumerate(PROTOCOLS):
        minimum = np.minimum(results[protocol]["minimum_singular_x"], results[protocol]["minimum_singular_y"])
        logdet = np.minimum(results[protocol]["log_abs_determinant_x"], results[protocol]["log_abs_determinant_y"])
        image = axes[0, column].imshow(np.log10(np.maximum(minimum.T, 1e-300)), origin="lower", cmap="cividis")
        axes[0, column].set(title=LABELS[protocol], ylabel=r"$\theta_y$ index")
        fig.colorbar(image, ax=axes[0, column], label=r"$\log_{10}s_{\min}$")
        image = axes[1, column].imshow(logdet.T, origin="lower", cmap="plasma")
        axes[1, column].set(xlabel=r"$\theta_x$ index", ylabel=r"$\theta_y$ index")
        fig.colorbar(image, ax=axes[1, column], label=r"$\min\log|\det M|$")
    save(fig, "link_conditioning_maps")

    fig, axes = plt.subplots(1, 3, figsize=(7.05, 2.5), constrained_layout=True)
    for ax, name in zip(axes, ("target", *PROTOCOLS)):
        values = results[name]["plaquette_phase"]
        image = ax.imshow(values.T, origin="lower", cmap="coolwarm", vmin=-np.pi, vmax=np.pi)
        ax.set(title=LABELS[name], xlabel=r"$\theta_x$ index", ylabel=r"$\theta_y$ index")
        fig.colorbar(image, ax=ax, label="oriented plaquette phase")
    save(fig, "plaquette_phase_maps")


def write_report(root: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Fresh-record twist-torus validation",
        "",
        "This campaign used a 16×16 physical torus, a 16×16 boundary-twist mesh, and one independently sampled occupied-frame trajectory per twist point.",
        "",
        "No full measurement record was replayed. The `fresh_full_record` surface refreshed both the site order and outcomes, while `fresh_outcomes` held only the site schedule fixed.",
        "",
        "## Result",
        "",
        f"- Exact target: `{summary['surfaces']['target']['classification']}`, C = {summary['surfaces']['target']['chern']}",
    ]
    for protocol in PROTOCOLS:
        surface = summary["surfaces"][protocol]
        topology = summary["real_space_topology"][protocol]
        lines.extend(
            [
                f"- {LABELS[protocol]}: `{surface['classification']}`, C = {surface['chern']}, ranks {surface['unique_ranks']}, valid-link fraction {surface['valid_link_fraction']:.6f}",
                f"  Real-space topology gate: {topology['passed']} (maximum final error {topology['maximum_final_marker_error']:.6g}).",
            ]
        )
    lines.extend(
        [
            "",
            f"Preliminary support for stitching fresh records: **{summary['preliminary_fresh_record_support']}**.",
            "",
            "An undefined result is not rounded or repaired. Rank-changing neighboring Slater determinants live in different particle-number sectors, and singular overlaps do not define a stable Berry link.",
            "",
            "The 256 twist points form one surface, not 256 independent statistical replicates.",
        ]
    )
    (root / "reports/validation_note.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-root", required=True, type=Path)
    args = parser.parse_args()
    summary = analyze(args.campaign_root.resolve())
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
