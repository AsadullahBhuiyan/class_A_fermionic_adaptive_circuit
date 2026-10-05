#!/usr/bin/env python3
"""Build the focused legacy modular-propagation chirality figure."""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


REPOSITORY = Path(__file__).resolve().parents[1]
MODULAR_SOURCE = REPOSITORY / (
    "00_WORKSPACE/COLAB/colab_small_system_testing/gpu_data/"
    "dynamic_modular_hamiltonians/runs/N16x40_nsh1_perfect_correction/"
    "modular_hamiltonians.npz"
)
DISPLACEMENT_SOURCE = REPOSITORY / (
    "00_WORKSPACE/COLAB/colab_small_system_testing/analysis_outputs/"
    "dynamic_modular_charge_spreading/sample_y0_averaged_dy_com/runs/"
    "N16x40_nsh1_perfect_correction/sample_y0_averaged_dy_com.npz"
)
FIGURES = Path(__file__).resolve().parent / "figures"
STEM = "modular_charge_chirality_cycle50_nsh1"
LEGACY_ROOT = REPOSITORY / "00_WORKSPACE/COLAB/colab_small_system_testing"
SNAPSHOT_SOURCE = LEGACY_ROOT / (
    "gpu_data/pure_state_covariance_snapshots/runs/"
    "N16x40_nsh1_perfect_correction/run_3301a140f933/batch_00000_snapshots.npy"
)


def _displacement_at_y(job):
    """Use the original analysis helpers, changing only packet coordinates."""
    from threadpoolctl import threadpool_limits

    sys.path.insert(0, str(LEGACY_ROOT / "src"))
    from dynamic_modular_charge_spreading import (
        restrict_covariance, modular_hamiltonian_from_restricted_covariance,
        eigensystem_from_h_mod,
    )
    from run_sample_y0_averaged_dy_com import evolve_packet_dy_com

    sample, y0, packet_y, times, eps, radius = job
    packets = [dict(label=f"x{x}_y{packet_y}", x_dw=x, y_rel=packet_y)
               for x in (5, 11)]
    with threadpool_limits(limits=1):
        shard = np.load(SNAPSHOT_SOURCE, mmap_mode="r")
        sub = restrict_covariance(shard[sample, 3], nx=16, ny=40, y0=y0)
        mod = modular_hamiltonian_from_restricted_covariance(sub, eps=eps)
        eig = eigensystem_from_h_mod(mod["h_mod"])
        result = evolve_packet_dy_com(
            h_vals=eig["h_vals"], h_vecs=eig["h_vecs"], nx=16, ny=40,
            packets=packets, times=times, x_window_radius=radius,
        )
    if np.max(result["charge_drift"]) > 1e-6:
        raise RuntimeError("Packet propagation failed charge conservation")
    return result


def _style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "stix",
        "font.size": 8.0,
        "axes.labelsize": 8.0,
        "axes.titlesize": 8.0,
        "legend.fontsize": 6.2,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "savefig.transparent": False,
    })


def _site_index(nx: int, x: int, y: int, orbital: int) -> int:
    return int(orbital) + 2 * int(x) + 2 * int(nx) * int(y)


def _packet_density(
    h_vals: np.ndarray,
    h_vecs: np.ndarray,
    *,
    nx: int,
    ny_sub: int,
    x: int,
    y: int,
    times: tuple[float, ...],
) -> np.ndarray:
    """Evolve one two-orbital packet and return cell-resolved densities."""
    occupied = np.asarray([_site_index(nx, x, y, orbital) for orbital in (0, 1)])
    coefficients = h_vecs[occupied, :].conj().T
    result = []
    for time_value in times:
        phase = np.exp(-1j * float(time_value) * h_vals)
        amplitudes = (h_vecs * phase[None, :]) @ coefficients
        site_density = np.sum(np.abs(amplitudes) ** 2, axis=1).real
        cell_density = site_density.reshape(ny_sub, nx, 2).sum(axis=-1).T
        if abs(float(cell_density.sum()) - 2.0) > 1e-10:
            raise RuntimeError("single-packet modular evolution failed to conserve charge")
        result.append(cell_density)
    return np.asarray(result, dtype=np.float64)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet-y", type=int, default=None,
                        help="Move both packets to this cut-relative y; preserve original outputs.")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.packet_y is not None and not 0 <= args.packet_y < 20:
        parser.error("--packet-y must be in 0..19")
    if args.workers < 1:
        parser.error("--workers must be positive")
    stem = STEM
    displacement_source = DISPLACEMENT_SOURCE
    _style()
    FIGURES.mkdir(parents=True, exist_ok=True)

    with np.load(MODULAR_SOURCE, allow_pickle=True) as source:
        snapshot_cycles = np.asarray(source["snapshot_cycles"], dtype=int)
        cycle_matches = np.flatnonzero(snapshot_cycles == 50)
        if cycle_matches.size != 1:
            raise RuntimeError(f"expected one cycle-50 modular generator, found {cycle_matches.size}")
        cycle_index = int(cycle_matches[0])
        h_vals = np.asarray(source["avg_h_vals"][cycle_index], dtype=np.float64)
        h_vecs = np.asarray(source["avg_h_vecs"][cycle_index], dtype=np.complex128)
        modular_metadata = json.loads(str(source["metadata_json"]))

    with np.load(DISPLACEMENT_SOURCE, allow_pickle=True) as source:
        times = np.asarray(source["times"], dtype=float)
        raw = np.asarray(source["dy_com_curves"], dtype=float)
        packet_labels = [str(value) for value in source["packet_labels"]]
        sample_indices = np.asarray(source["sample_indices"], dtype=int)
        y0_values = np.asarray(source["y0_values"], dtype=int)
        charge_drift = np.asarray(source["charge_drift"], dtype=float)
        displacement_metadata = json.loads(str(source["metadata_json"]))

    if raw.shape[:4] != (1, 10, 40, 4):
        raise RuntimeError(f"unexpected displacement shape: {raw.shape}")
    if packet_labels != ["x5_y0", "x5_y19", "x11_y0", "x11_y19"]:
        raise RuntimeError(f"unexpected packet labels: {packet_labels}")

    if args.packet_y is not None:
        from tqdm import tqdm

        stem = f"{STEM}_y{args.packet_y}_original_pipeline"
        displacement_source = FIGURES / f"{stem}_data.npz"
        jobs = [(int(s), int(y0), args.packet_y, times,
                 displacement_metadata["modular_eps"],
                 displacement_metadata["x_window_radius"])
                for s in sample_indices for y0 in y0_values]
        raw = np.empty((1, len(sample_indices), len(y0_values), 2, len(times)))
        charge_drift = np.empty(raw.shape[:-1])
        q_window = np.empty_like(raw)
        print("Recomputing displacements with the original legacy helpers", flush=True)
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for i, result in enumerate(tqdm(pool.map(_displacement_at_y, jobs),
                                            total=len(jobs), unit="cut")):
                s, y = divmod(i, len(y0_values))
                raw[0, s, y] = result["dy_com"]
                charge_drift[0, s, y] = result["charge_drift"]
                q_window[0, s, y] = result["Q_window"]
        packet_labels = [f"x{x}_y{args.packet_y}" for x in (5, 11)]
        displacement_metadata = dict(
            displacement_metadata, packet_y_rel=[args.packet_y, args.packet_y],
            source_shard_path=str(SNAPSHOT_SOURCE),
            com_support="radius_2_wall_window", com_center="initial_packet_position",
            com_distance_definition="y_com(t)-y_com(0)",
            implementation="original run_sample_y0_averaged_dy_com.evolve_packet_dy_com",
        )
        np.savez_compressed(
            displacement_source, times=times, dy_com_curves=raw,
            Q_window_curves=q_window, charge_drift=charge_drift,
            sample_indices=sample_indices, y0_values=y0_values,
            packet_labels=packet_labels, metadata_json=json.dumps(displacement_metadata),
        )

    # The translated cut origins are correlated reductions within a trajectory.
    # Average those first, then estimate sampling uncertainty over ten trajectories.
    trajectory_curves = raw[0].mean(axis=1)
    mean = trajectory_curves.mean(axis=0)
    sem = trajectory_curves.std(axis=0, ddof=1) / np.sqrt(trajectory_curves.shape[0])

    snapshot_times = (0.0, 0.5, 1.0) if args.packet_y is not None else (0.0, 0.5, 1.0, 1.5)
    snapshot_indices = [int(np.argmin(np.abs(times - value))) for value in snapshot_times]
    if any(abs(times[index] - value) > 1e-12 for index, value in zip(snapshot_indices, snapshot_times)):
        raise RuntimeError("requested modular-time snapshot is absent")

    # Keep the diagonally opposed initial packets only.  The old spatial file
    # stored their sum with the other two corners, so reconstruct the two
    # packet densities directly from the saved averaged modular generator.
    packet_specs = (
        {"index": 0, "x": 5, "y": 0, "color": "#D55E00", "label": r"$(5,0)$"},
        {"index": 3, "x": 11, "y": 19, "color": "#0072B2", "label": r"$(11,19)$"},
    )
    if args.packet_y is not None:
        packet_specs = tuple(
            dict(index=i, x=x, y=args.packet_y, color=color,
                 label=rf"$({x},{args.packet_y})$")
            for i, (x, color) in enumerate(((5, "#D55E00"), (11, "#0072B2")))
        )
    packet_densities = np.asarray(
        [
            _packet_density(
                h_vals,
                h_vecs,
                nx=16,
                ny_sub=20,
                x=int(spec["x"]),
                y=int(spec["y"]),
                times=snapshot_times,
            )
            for spec in packet_specs
        ]
    )

    figure = plt.figure(figsize=(7.05, 2.65))
    grid = figure.add_gridspec(1, 2, width_ratios=(0.92, 1.55), wspace=0.34)
    density_axis = figure.add_subplot(grid[0, 0])
    curve_axis = figure.add_subplot(grid[0, 1])

    x_cells, y_cells = np.meshgrid(np.arange(16), np.arange(20), indexing="ij")
    snapshot_colors = ("#332288", "#E69F00", "#009E73", "#CC79A7")
    for packet_index, spec in enumerate(packet_specs):
        for time_index, (time_value, face_color) in enumerate(zip(snapshot_times, snapshot_colors)):
            values = np.maximum(packet_densities[packet_index, time_index], 0.0)
            mask = values > 1e-4
            marker_area = 105.0 * np.sqrt(values[mask] / 2.0)
            density_axis.scatter(
                x_cells[mask],
                y_cells[mask],
                s=marker_area,
                facecolors=face_color,
                edgecolors=face_color,
                marker="o",
                alpha=1.0,
                linewidths=0.25,
                zorder=5 - time_index,
            )
    for wall in (5, 11):
        density_axis.axvline(wall, color="0.35", linestyle=":", linewidth=0.85, zorder=1)
    density_axis.set_xlim(-0.5, 15.5)
    density_axis.set_ylim(-0.5, 19.5)
    density_axis.set_aspect("equal")
    density_axis.set_xticks((0, 5, 11, 15))
    density_axis.set_yticks((0, 5, 10, 15, 19))
    density_axis.set_xlabel(r"$x$")
    density_axis.set_ylabel(r"$y-y_0$")
    time_handles = [
        Line2D(
            [0], [0], marker="o", linestyle="none", markersize=4.0,
            markerfacecolor=color, markeredgecolor=color, label=rf"${value:g}$",
        )
        for value, color in zip(snapshot_times, snapshot_colors)
    ]
    density_axis.legend(
        handles=time_handles,
        title=r"$t_{\rm mod}$",
        frameon=False,
        loc="center left",
        borderaxespad=0.25,
        handlelength=0.8,
        handletextpad=0.25,
        labelspacing=0.25,
    )
    density_axis.text(-0.18, 1.04, "(a)", transform=density_axis.transAxes, fontweight="bold")

    for spec, linestyle in zip(packet_specs, ("-", "--")):
        packet = int(spec["index"])
        color = str(spec["color"])
        curve_axis.plot(
            times,
            mean[packet],
            color=color,
            linestyle=linestyle,
            linewidth=1.15,
            label=str(spec["label"]),
        )
        curve_axis.fill_between(
            times,
            mean[packet] - sem[packet],
            mean[packet] + sem[packet],
            color=color,
            alpha=0.12,
            linewidth=0,
        )
    curve_axis.axhline(0.0, color="0.35", linewidth=0.7)
    curve_axis.set_xlim(0.0, 32.0)
    curve_axis.set_ylim(-18.5, 18.5)
    curve_axis.set_xticks((0, 8, 16, 24, 32))
    curve_axis.set_xlabel(r"modular time $t_{\rm mod}$")
    curve_axis.set_ylabel(r"$\langle\Delta y\rangle$")
    curve_axis.legend(
        frameon=False, ncol=2, loc="upper right", columnspacing=0.9,
        handlelength=2.4,
    )
    curve_axis.text(-0.15, 1.04, "(b)", transform=curve_axis.transAxes, fontweight="bold")

    figure.subplots_adjust(left=0.065, right=0.985, bottom=0.19, top=0.91)
    pdf = FIGURES / f"{stem}.pdf"
    png = FIGURES / f"{stem}.png"
    figure.savefig(pdf, bbox_inches="tight")
    figure.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(figure)

    summary = {
        "schema": "technical_report_modular_chirality_figure_v5",
        "modular_generator_source": str(MODULAR_SOURCE.relative_to(REPOSITORY)),
        "displacement_source": str(displacement_source.relative_to(REPOSITORY)),
        "modular_generator_source_metadata": modular_metadata,
        "displacement_source_metadata": displacement_metadata,
        "sample_count": int(sample_indices.size),
        "translated_origins_per_trajectory": int(y0_values.size),
        "averaging_order": "mean over 40 translated origins within each trajectory, then mean and ddof=1 SEM over 10 trajectories",
        "packet_labels": packet_labels,
        "displayed_packet_labels": [packet_labels[int(spec["index"])] for spec in packet_specs],
        "final_displacement_mean": mean[:, -1].tolist(),
        "final_displacement_sem": sem[:, -1].tolist(),
        "maximum_charge_drift": float(np.max(np.abs(charge_drift))),
        "spatial_snapshot_modular_times": list(snapshot_times),
        "packet_positions": [[int(spec["x"]), int(spec["y"])] for spec in packet_specs],
        "spatial_density_cutoff": 1e-4,
        "spatial_marker_alpha": 1.0,
        "spatial_marker_edge": "same as marker fill",
        "outputs": {"pdf": str(pdf), "png": str(png)},
    }
    summary_path = FIGURES / f"{stem}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
