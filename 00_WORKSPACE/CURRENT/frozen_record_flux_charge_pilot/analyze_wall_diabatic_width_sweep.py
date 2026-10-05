#!/usr/bin/env python3
"""Analyze verified raw CW/CCW wall-diabatic spectral-pump paths.

One durable result pair belongs to one monitored endpoint and stores both flux
directions on axis zero.  This script explodes that axis only in memory.  It
never substitutes a direction-odd average for the requested sample-wise raw
charge transfer.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from wall_diabatized_io import discover_verified_paths, first_array, first_scalar, sha256_path


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "results" / "N20_24_28_32x24_wall_diabatic_spectral_pump_s100_v1"
)
FIGURE_WIDTH = 7.05
COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9")
LINESTYLES = {"ccw": "-", "cw": "--"}


def _plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["CMU Sans Serif", "Computer Modern Sans Serif", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "pdf.fonttype": 42,
            "savefig.dpi": 300,
        }
    )


def _text(value: Any) -> str:
    raw = np.asarray(value).item()
    return raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)


def _direction_names(arrays: dict[str, np.ndarray], count: int) -> list[str]:
    names = first_array(arrays, ("directions", "direction"), required=False)
    if names is None:
        if count != 2:
            raise ValueError("direction labels are missing")
        return ["ccw", "cw"]
    result = [_text(value).lower() for value in np.asarray(names)]
    if len(result) != count or sorted(result) != ["ccw", "cw"]:
        raise ValueError(f"expected exactly CW and CCW directions, got {result!r}")
    return result


def _directional(array: np.ndarray | None, index: int, count: int) -> np.ndarray | None:
    if array is None:
        return None
    array = np.asarray(array)
    if array.ndim > 0 and array.shape[0] == count:
        return np.array(array[index], copy=True)
    return np.array(array, copy=True)


def _minimum(array: np.ndarray | None) -> float:
    return float(np.nanmin(array)) if array is not None and np.asarray(array).size else float("nan")


def _maximum_abs(array: np.ndarray | None) -> float:
    return float(np.nanmax(np.abs(array))) if array is not None and np.asarray(array).size else float("nan")


def _endpoint(array: np.ndarray | None) -> float:
    return float(np.asarray(array).reshape(-1)[-1]) if array is not None and np.asarray(array).size else float("nan")


def _crossing_value(array: np.ndarray | None, index: int) -> np.ndarray | None:
    if array is None:
        return None
    values = np.asarray(array)
    if not values.size or index < 0 or index >= values.shape[0]:
        return None
    return np.asarray(values[index])


def _cell(metadata: dict[str, Any]) -> str:
    if metadata.get("cell"):
        return str(metadata["cell"])
    nx, ny = int(metadata["Nx"]), int(metadata["Ny"])
    protocol = str(metadata.get("protocol", "primary"))
    return f"N{nx}x{ny}_{protocol}"


def load_path_rows(output_root: Path, *, allow_invalid: bool = False) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    verified, invalid = discover_verified_paths(output_root, allow_invalid=allow_invalid)
    rows: list[dict[str, Any]] = []
    for item in verified:
        arrays, metadata = item.arrays, item.metadata
        phi_all = first_array(arrays, ("phi",))
        direction_count = 1 if phi_all.ndim == 1 else phi_all.shape[0]
        names = [str(metadata["direction"])] if direction_count == 1 else _direction_names(arrays, direction_count)
        sigmas = first_array(arrays, ("sigma", "sigmas"), required=False)
        optional = {
            "edge_internal_gap": first_array(arrays, ("edge_internal_gap", "internal_gap"), required=False),
            "edge_external_gap": first_array(arrays, ("edge_external_gap", "external_gap"), required=False),
            "edge_link_min_singular": first_array(arrays, ("edge_link_min_singular", "link_min_singular"), required=False),
            "wall_character_margin": first_array(arrays, ("wall_character_margin", "edge_wall_margin"), required=False),
            "total_charge_residual": first_array(arrays, ("total_charge_residual", "charge_residual"), required=False),
            "projector_residual": first_array(arrays, ("projector_residual", "idempotency_residual"), required=False),
            "multicut_q_x": first_array(arrays, ("multicut_q_x", "q_x_by_cut"), required=False),
            "center_displacement": first_array(arrays, ("center_of_charge_displacement", "center_displacement"), required=False),
            "defect_eigenvalues": first_array(arrays, ("endpoint_defect_eigenvalues", "defect_eigenvalues"), required=False),
            "particle_density_x": first_array(arrays, ("endpoint_particle_density_x", "particle_density_x"), required=False),
            "hole_density_x": first_array(arrays, ("endpoint_hole_density_x", "hole_density_x"), required=False),
            "resolved": first_array(arrays, ("resolved",), required=False),
            "unresolved_reason": first_array(arrays, ("unresolved_reason",), required=False),
            "ordinary_q_x": first_array(arrays, ("ordinary_q_x",), required=False),
            "instantaneous_q_x": first_array(arrays, ("instantaneous_q_x",), required=False),
            "source_chern_mean": first_array(arrays, ("source_real_space_chern_mean",), required=False),
            "source_chern_std": first_array(arrays, ("source_real_space_chern_std",), required=False),
            "edge_minimum_gap_index": first_array(arrays, ("edge_minimum_gap_index",), required=False),
            "edge_B_eigenvalues": first_array(arrays, ("edge_B_eigenvalues",), required=False),
            "edge_combined_wall_weight": first_array(arrays, ("edge_combined_wall_weight",), required=False),
        }
        q_all = first_array(arrays, ("q_x", "raw_q_x"))
        dl_all = first_array(arrays, ("delta_N_left", "delta_n_left"))
        dr_all = first_array(arrays, ("delta_N_right", "delta_n_right"))
        dt_all = first_array(arrays, ("delta_N_total", "delta_n_total"))
        density_all = first_array(arrays, ("density_x", "delta_density_x"), required=False)
        for index, direction in enumerate(names):
            phi = _directional(phi_all, index, direction_count)
            q_x = _directional(q_all, index, direction_count)
            delta_left = _directional(dl_all, index, direction_count)
            delta_right = _directional(dr_all, index, direction_count)
            delta_total = _directional(dt_all, index, direction_count)
            defect = _directional(optional["defect_eigenvalues"], index, direction_count)
            multicut = _directional(optional["multicut_q_x"], index, direction_count)
            particle_density = _directional(optional["particle_density_x"], index, direction_count)
            hole_density = _directional(optional["hole_density_x"], index, direction_count)
            sigma = int(np.asarray(sigmas)[index]) if sigmas is not None else {"ccw": 1, "cw": -1}[direction]
            endpoint_defect_particle = float(np.nanmax(defect)) if defect is not None else float("nan")
            endpoint_defect_hole = float(np.nanmin(defect)) if defect is not None else float("nan")
            endpoint_multicut = np.asarray(multicut)[-1] if multicut is not None and np.asarray(multicut).ndim > 1 else multicut
            resolved_value = _directional(optional["resolved"], index, direction_count)
            reason_value = _directional(optional["unresolved_reason"], index, direction_count)
            ordinary_q_x = _directional(optional["ordinary_q_x"], index, direction_count)
            instantaneous_q_x = _directional(optional["instantaneous_q_x"], index, direction_count)
            internal_path = _directional(optional["edge_internal_gap"], index, direction_count)
            external_path = _directional(optional["edge_external_gap"], index, direction_count)
            link_path = _directional(optional["edge_link_min_singular"], index, direction_count)
            crossing_index_value = _directional(optional["edge_minimum_gap_index"], index, direction_count)
            if crossing_index_value is not None:
                crossing_index = int(np.asarray(crossing_index_value).item())
            elif internal_path is not None and np.asarray(internal_path).size:
                crossing_index = int(np.nanargmin(internal_path))
            else:
                crossing_index = -1
            b_path = _directional(optional["edge_B_eigenvalues"], index, direction_count)
            wall_weight_path = _directional(optional["edge_combined_wall_weight"], index, direction_count)
            crossing_b = _crossing_value(b_path, crossing_index)
            crossing_wall_weight = _crossing_value(wall_weight_path, crossing_index)
            crossing_internal = _crossing_value(internal_path, crossing_index)
            crossing_external = _crossing_value(external_path, crossing_index)
            crossing_link = _crossing_value(link_path, crossing_index)
            crossing_b_left = float(np.min(crossing_b)) if crossing_b is not None else float("nan")
            crossing_b_right = float(np.max(crossing_b)) if crossing_b is not None else float("nan")
            row = {
                **metadata,
                "cell": _cell(metadata),
                "protocol": str(metadata.get("protocol", "primary")),
                "direction": direction,
                "sigma": sigma,
                "source_backend": str(metadata.get("source_backend", "unknown")),
                "result_path": str(item.result_path),
                "result_sha256": str(item.completion["result"]["sha256"]),
                "phi": phi,
                "q_x": q_x,
                "delta_N_left": delta_left,
                "delta_N_right": delta_right,
                "delta_N_total": delta_total,
                "density_x": _directional(density_all, index, direction_count),
                "endpoint_q_x": float(q_x[-1]),
                "correctly_signed_endpoint_q_x": float(sigma * q_x[-1]),
                "endpoint_delta_N_left": float(delta_left[-1]),
                "endpoint_delta_N_right": float(delta_right[-1]),
                "endpoint_delta_N_total": float(delta_total[-1]),
                "endpoint_ordinary_q_x": _endpoint(ordinary_q_x),
                "endpoint_instantaneous_q_x": _endpoint(instantaneous_q_x),
                "resolved": bool(np.asarray(resolved_value).item()) if resolved_value is not None else False,
                "unresolved_reason": _text(reason_value) if reason_value is not None else "missing_resolution_diagnostic",
                "source_real_space_chern_mean": first_scalar(arrays, ("source_real_space_chern_mean",), required=False),
                "source_real_space_chern_std": first_scalar(arrays, ("source_real_space_chern_std",), required=False),
                "edge_crossing_index": crossing_index,
                "crossing_edge_internal_gap": float(crossing_internal) if crossing_internal is not None else float("nan"),
                "crossing_edge_external_gap": float(crossing_external) if crossing_external is not None else float("nan"),
                "crossing_edge_link_singular": float(crossing_link) if crossing_link is not None else float("nan"),
                "crossing_left_B_eigenvalue": crossing_b_left,
                "crossing_right_B_eigenvalue": crossing_b_right,
                "crossing_minimum_wall_weight": float(np.min(crossing_wall_weight)) if crossing_wall_weight is not None else float("nan"),
                "crossing_wall_character_margin": (
                    float(min(-crossing_b_left, crossing_b_right)) if crossing_b is not None else float("nan")
                ),
                "minimum_edge_internal_gap": _minimum(internal_path),
                "minimum_edge_external_gap": _minimum(external_path),
                "minimum_edge_link_singular": _minimum(link_path),
                "minimum_wall_character_margin": _minimum(_directional(optional["wall_character_margin"], index, direction_count)),
                "maximum_total_charge_residual": _maximum_abs(_directional(optional["total_charge_residual"], index, direction_count)),
                "maximum_projector_residual": _maximum_abs(_directional(optional["projector_residual"], index, direction_count)),
                "endpoint_defect_particle": endpoint_defect_particle,
                "endpoint_defect_hole": endpoint_defect_hole,
                "endpoint_multicut_spread": (
                    float(np.nanmax(endpoint_multicut) - np.nanmin(endpoint_multicut))
                    if endpoint_multicut is not None and np.asarray(endpoint_multicut).size else float("nan")
                ),
                "center_displacement": _endpoint(_directional(optional["center_displacement"], index, direction_count)),
                "particle_center_x": (
                    float(np.dot(np.arange(particle_density.size), particle_density) / np.sum(particle_density))
                    if particle_density is not None and np.sum(particle_density) > 0 else float("nan")
                ),
                "hole_center_x": (
                    float(np.dot(np.arange(hole_density.size), hole_density) / np.sum(hole_density))
                    if hole_density is not None and np.sum(hole_density) > 0 else float("nan")
                ),
            }
            rows.append(row)
    return rows, invalid


def _is_primary(row: dict[str, Any]) -> bool:
    if "is_primary" in row:
        return bool(row["is_primary"])
    return row.get("protocol", "primary") in {"primary", "width_sweep", "wall_diabatic_primary"}


def _write_csv(rows: Iterable[dict[str, Any]], path: Path, fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _save(figure: plt.Figure, root: Path, stem: str) -> dict[str, str]:
    root.mkdir(parents=True, exist_ok=True)
    pdf, png = root / f"{stem}.pdf", root / f"{stem}.png"
    figure.savefig(pdf, bbox_inches="tight")
    figure.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return {"pdf": str(pdf), "png": str(png)}


def _mean_sd(selected: list[dict[str, Any]], key: str) -> tuple[np.ndarray, np.ndarray]:
    values = np.stack([np.asarray(row[key], dtype=float) for row in selected])
    ddof = 1 if values.shape[0] > 1 else 0
    return np.mean(values, axis=0), np.std(values, axis=0, ddof=ddof)


def _bootstrap_mean_ci(values: np.ndarray, seed: int, draws: int = 10_000) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if values.size == 1:
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(seed)
    chunk = 1000
    means: list[np.ndarray] = []
    for start in range(0, draws, chunk):
        count = min(chunk, draws - start)
        indices = rng.integers(0, values.size, size=(count, values.size))
        means.append(np.mean(values[indices], axis=1))
    distribution = np.concatenate(means)
    return float(np.quantile(distribution, 0.025)), float(np.quantile(distribution, 0.975))


def plot_raw_paths(rows: list[dict[str, Any]], root: Path) -> dict[str, str]:
    cells = sorted({row["cell"] for row in rows})
    figure, axes = plt.subplots(2, 2, figsize=(FIGURE_WIDTH, 4.5), sharex=True, sharey=True)
    for row_index, wall in enumerate(("soft", "hard")):
        for column, direction in enumerate(("ccw", "cw")):
            axis = axes[row_index, column]
            for cell_index, cell in enumerate(cells):
                selected = [row for row in rows if row["cell"] == cell and row["wall"] == wall and row["direction"] == direction]
                if not selected:
                    continue
                progress = np.abs((selected[0]["phi"] - selected[0]["phi"][0]) / (2 * np.pi))
                mean, sd = _mean_sd(selected, "q_x")
                color = COLORS[cell_index % len(COLORS)]
                axis.plot(progress, mean, LINESTYLES[direction], color=color, lw=1.15, label=cell)
                axis.fill_between(progress, mean - sd, mean + sd, color=color, alpha=0.14, linewidth=0)
            axis.axhline(0, color="0.35", ls=":", lw=0.7)
            axis.axhline(1, color="0.65", ls="--", lw=0.6)
            axis.axhline(-1, color="0.65", ls="--", lw=0.6)
            axis.set_title(f"{wall} wall, {direction.upper()}")
            if row_index == 1:
                axis.set_xlabel(r"flux path $|\phi-\phi_0|/(2\pi)$")
            if column == 0:
                axis.set_ylabel(r"raw $q_x$")
            if row_index == 0 and column == 0:
                axis.legend(frameon=False, fontsize=6)
    for label, axis in zip("abcdefghijklmnopqrstuvwxyz", axes.ravel()):
        axis.text(-0.17, 1.04, f"({label})", transform=axis.transAxes, fontweight="bold")
    figure.tight_layout()
    return _save(figure, root, "wall_diabatic_raw_qx_paths")


def plot_endpoint_histograms(rows: list[dict[str, Any]], root: Path) -> dict[str, str]:
    cells = sorted({row["cell"] for row in rows})
    figure, axes = plt.subplots(2, 2, figsize=(FIGURE_WIDTH, 4.5), sharex=True, sharey=True)
    bins = np.linspace(-1.25, 1.25, 31)
    for row_index, wall in enumerate(("soft", "hard")):
        for column, direction in enumerate(("ccw", "cw")):
            axis = axes[row_index, column]
            for index, cell in enumerate(cells):
                values = [row["endpoint_q_x"] for row in rows if row["cell"] == cell and row["wall"] == wall and row["direction"] == direction]
                if values:
                    axis.hist(values, bins=bins, histtype="step", lw=1.2, color=COLORS[index % len(COLORS)], label=cell)
            axis.axvline(0, color="0.35", ls=":", lw=0.7)
            axis.axvline(1 if direction == "ccw" else -1, color="0.65", ls="--", lw=0.7)
            axis.set_title(f"{wall} wall, {direction.upper()}")
            if row_index == 1:
                axis.set_xlabel(r"endpoint raw $q_x$")
            if column == 0:
                axis.set_ylabel("paths")
    if cells:
        axes[0, 0].legend(frameon=False, fontsize=6)
    for label, axis in zip("abcd", axes.ravel()):
        axis.text(-0.16, 1.04, f"({label})", transform=axis.transAxes, fontweight="bold")
    figure.tight_layout()
    return _save(figure, root, "wall_diabatic_endpoint_histograms")


def plot_diagnostics(rows: list[dict[str, Any]], root: Path) -> dict[str, str]:
    figure, axes = plt.subplots(1, 3, figsize=(FIGURE_WIDTH, 2.45))
    specs = (
        ("crossing_edge_external_gap", r"crossing external edge--bulk gap"),
        ("crossing_edge_link_singular", r"crossing polar-link singular value"),
        ("endpoint_multicut_spread", r"endpoint multi-cut spread"),
    )
    for axis, (key, label) in zip(axes, specs):
        for index, cell in enumerate(sorted({row["cell"] for row in rows})):
            selected = [row for row in rows if row["cell"] == cell and np.isfinite(row[key])]
            if selected:
                axis.scatter([row[key] for row in selected], [abs(row["endpoint_q_x"]) for row in selected], s=9, alpha=0.55, color=COLORS[index % len(COLORS)], label=cell)
        axis.set_xlabel(label)
        axis.set_ylabel(r"$|q_x(2\pi)|$")
        axis.axhline(1, color="0.65", ls="--", lw=0.7)
    if rows:
        axes[0].legend(frameon=False, fontsize=6)
    for label, axis in zip("abc", axes):
        axis.text(-0.20, 1.04, f"({label})", transform=axis.transAxes, fontweight="bold")
    figure.tight_layout()
    return _save(figure, root, "wall_diabatic_numerical_diagnostics")


def _statistics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups = sorted({(row["cell"], row["wall"], row["direction"]) for row in rows})
    result: list[dict[str, Any]] = []
    for cell, wall, direction in groups:
        selected = [row for row in rows if (row["cell"], row["wall"], row["direction"]) == (cell, wall, direction)]
        values = np.asarray([row["endpoint_q_x"] for row in selected])
        signed = np.asarray([row["correctly_signed_endpoint_q_x"] for row in selected])
        seed = int.from_bytes(hashlib.sha256(f"{cell}|{wall}|{direction}".encode()).digest()[:8], "little")
        ci_low, ci_high = _bootstrap_mean_ci(values, seed)
        resolved = np.asarray([bool(row["resolved"]) for row in selected])
        result.append(
            {
                "cell": cell,
                "wall": wall,
                "direction": direction,
                "path_count": int(values.size),
                "endpoint_mean": float(np.mean(values)),
                "endpoint_sd": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
                "endpoint_mean_bootstrap95_low": ci_low,
                "endpoint_mean_bootstrap95_high": ci_high,
                "abs_qx_gt_0p5_fraction": float(np.mean(np.abs(values) > 0.5)),
                "abs_qx_gt_0p9_fraction": float(np.mean(np.abs(values) > 0.9)),
                "abs_qx_0p95_to_1p05_fraction": float(np.mean((np.abs(values) >= 0.95) & (np.abs(values) <= 1.05))),
                "correct_sign_qx_gt_0p5_fraction": float(np.mean(signed > 0.5)),
                "correct_sign_qx_gt_0p9_fraction": float(np.mean(signed > 0.9)),
                "correct_sign_qx_0p95_to_1p05_fraction": float(np.mean((signed >= 0.95) & (signed <= 1.05))),
                "resolved_fraction": float(np.mean(resolved)),
                "unresolved_count": int(np.count_nonzero(~resolved)),
                "abs_quantization_error_median": float(np.median(np.abs(np.abs(values) - 1.0))),
            }
        )
    return result


def _edge_gap_fits(rows: list[dict[str, Any]], *, bootstrap_draws: int = 2_000) -> list[dict[str, Any]]:
    """Fit the edge splitting against physical wall separation W=Nx/2.

    The diagnostic used by the method is the internal edge gap at the selected
    minimum-gap crossing, not an unrelated global minimum elsewhere on the
    flux path.  Bootstrap resampling is independent within every width.
    """
    fits: list[dict[str, Any]] = []
    groups = sorted({(row["protocol"], row["wall"], row["direction"]) for row in rows})
    for protocol, wall, direction in groups:
        selected = [
            row for row in rows
            if (row["protocol"], row["wall"], row["direction"]) == (protocol, wall, direction)
            and np.isfinite(row["crossing_edge_internal_gap"])
            and row["crossing_edge_internal_gap"] > 0
        ]
        x_values = sorted({int(row["Nx"]) for row in selected})
        wall_separations = np.asarray(x_values, dtype=float) / 2.0
        values_by_width = {
            nx: np.asarray([
                row["crossing_edge_internal_gap"] for row in selected if int(row["Nx"]) == nx
            ], dtype=float)
            for nx in x_values
        }
        medians = np.asarray([
            np.median(values_by_width[nx])
            for nx in x_values
        ])
        record: dict[str, Any] = {
            "protocol": protocol,
            "wall": wall,
            "direction": direction,
            "Nx": x_values,
            "wall_separation_sites": wall_separations.tolist(),
            "median_delta_edge": medians.tolist(),
            "status": "pending" if len(x_values) < 3 or np.any(medians <= 0) else "fit",
        }
        if record["status"] == "fit":
            slope, intercept = np.polyfit(wall_separations, np.log(medians), 1)
            predicted = intercept + slope * wall_separations
            observed = np.log(medians)
            denominator = float(np.sum((observed - np.mean(observed)) ** 2))
            r_squared = 1.0 - float(np.sum((observed - predicted) ** 2)) / denominator if denominator > 0 else float("nan")
            seed = int.from_bytes(
                hashlib.sha256(f"edge-fit|{protocol}|{wall}|{direction}".encode()).digest()[:8],
                "little",
            )
            rng = np.random.default_rng(seed)
            slopes = np.empty(bootstrap_draws, dtype=float)
            for draw in range(bootstrap_draws):
                draw_medians = np.asarray([
                    np.median(values[rng.integers(0, values.size, size=values.size)])
                    for values in (values_by_width[nx] for nx in x_values)
                ])
                slopes[draw] = np.polyfit(wall_separations, np.log(draw_medians), 1)[0]
            negative = slopes[slopes < 0]
            decay_lengths = -1.0 / negative if negative.size else np.asarray([], dtype=float)
            record.update(
                {
                    "log_slope": float(slope),
                    "log_intercept": float(intercept),
                    "decay_length_sites": float(-1.0 / slope) if slope < 0 else float("nan"),
                    "log_slope_bootstrap95_low": float(np.quantile(slopes, 0.025)),
                    "log_slope_bootstrap95_high": float(np.quantile(slopes, 0.975)),
                    "decay_length_sites_bootstrap95_low": (
                        float(np.quantile(decay_lengths, 0.025)) if decay_lengths.size else float("nan")
                    ),
                    "decay_length_sites_bootstrap95_high": (
                        float(np.quantile(decay_lengths, 0.975)) if decay_lengths.size else float("nan")
                    ),
                    "bootstrap_draws": bootstrap_draws,
                    "bootstrap_negative_slope_fraction": float(np.mean(slopes < 0)),
                    "r_squared": r_squared,
                    "decreasing_with_width": bool(slope < 0),
                }
            )
        fits.append(record)
    return fits


def _empirical_ks(first: np.ndarray, second: np.ndarray) -> float:
    first, second = np.sort(np.asarray(first, dtype=float)), np.sort(np.asarray(second, dtype=float))
    grid = np.unique(np.concatenate((first, second)))
    return float(np.max(np.abs(np.searchsorted(first, grid, side="right") / first.size - np.searchsorted(second, grid, side="right") / second.size)))


def _backend_fixed_effect(rows: list[dict[str, Any]], *, bootstrap_draws: int = 2_000) -> dict[str, Any]:
    """Estimate the independent-ensemble GPU shift with stratum fixed effects."""
    if not rows:
        return {"status": "pending", "reason": "no bridge comparison rows"}
    strata = sorted({str(row["stratum"]) for row in rows})
    stratum_index = {value: index for index, value in enumerate(strata)}

    def coefficient(selected: list[dict[str, Any]]) -> tuple[float, float]:
        y = np.asarray([row["q_x"] for row in selected], dtype=float)
        columns = [np.ones(y.size), np.asarray([row["gpu"] for row in selected], dtype=float)]
        for stratum in strata[1:]:
            columns.append(np.asarray([row["stratum"] == stratum for row in selected], dtype=float))
        matrix = np.column_stack(columns)
        beta, _, rank, _ = np.linalg.lstsq(matrix, y, rcond=None)
        if rank < matrix.shape[1]:
            return float(beta[1]), float("nan")
        residual = y - matrix @ beta
        dof = max(1, y.size - matrix.shape[1])
        covariance = (float(residual @ residual) / dof) * np.linalg.pinv(matrix.T @ matrix)
        return float(beta[1]), float(np.sqrt(max(0.0, covariance[1, 1])))

    estimate, standard_error = coefficient(rows)
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row["stratum"]), int(row["gpu"])), []).append(row)
    seed = int.from_bytes(hashlib.sha256(b"wall-diabatic-backend-factor-v1").digest()[:8], "little")
    rng = np.random.default_rng(seed)
    boot = np.empty(bootstrap_draws, dtype=float)
    for draw in range(bootstrap_draws):
        sampled: list[dict[str, Any]] = []
        for values in groups.values():
            sampled.extend(values[index] for index in rng.integers(0, len(values), size=len(values)))
        boot[draw] = coefficient(sampled)[0]
    return {
        "status": "complete",
        "model": "raw endpoint q_x ~ GPU indicator + cell:wall:direction fixed effects",
        "gpu_coefficient": estimate,
        "conventional_standard_error": standard_error,
        "stratified_bootstrap95_low": float(np.quantile(boot, 0.025)),
        "stratified_bootstrap95_high": float(np.quantile(boot, 0.975)),
        "bootstrap_draws": bootstrap_draws,
        "observation_count": len(rows),
        "stratum_count": len(stratum_index),
        "ensemble_relation": "independent and unpaired",
        "pooled_into_primary_estimates": False,
    }


def _backend_bridge(primary: list[dict[str, Any]], all_rows: list[dict[str, Any]]) -> dict[str, Any]:
    bridge = [
        row for row in all_rows
        if not _is_primary(row) and str(row.get("source_backend", "")).lower() == "gpu"
        and str(row.get("result_collection", "")) == "bridge_pump"
    ]
    if not bridge:
        return {"status": "pending", "reason": "no verified GPU bridge paths", "comparisons": []}
    comparisons: list[dict[str, Any]] = []
    regression_rows: list[dict[str, Any]] = []
    for bridge_cell, wall, direction in sorted({(row["cell"], row["wall"], row["direction"]) for row in bridge}):
        primary_cell = bridge_cell.removeprefix("bridge_")
        gpu = np.asarray([
            row["endpoint_q_x"] for row in bridge
            if (row["cell"], row["wall"], row["direction"]) == (bridge_cell, wall, direction)
        ])
        sample_ids = {
            int(row["sample_id"]) for row in bridge
            if (row["cell"], row["wall"], row["direction"]) == (bridge_cell, wall, direction)
        }
        cpu = np.asarray([
            row["endpoint_q_x"] for row in primary
            if (row["cell"], row["wall"], row["direction"]) == (primary_cell, wall, direction)
            and int(row["sample_id"]) in sample_ids
        ])
        if not cpu.size or not gpu.size:
            continue
        comparisons.append(
            {
                "bridge_cell": bridge_cell, "primary_cell": primary_cell,
                "wall": wall, "direction": direction,
                "cpu_count": int(cpu.size), "gpu_count": int(gpu.size),
                "cpu_mean": float(np.mean(cpu)), "gpu_mean": float(np.mean(gpu)),
                "absolute_mean_difference": float(abs(np.mean(cpu) - np.mean(gpu))),
                "empirical_ks_distance": _empirical_ks(cpu, gpu),
                "pooled": False,
            }
        )
        stratum = f"{primary_cell}|{wall}|{direction}"
        regression_rows.extend({"q_x": float(value), "gpu": 0, "stratum": stratum} for value in cpu)
        regression_rows.extend({"q_x": float(value), "gpu": 1, "stratum": stratum} for value in gpu)
    return {
        "status": "complete" if comparisons else "pending",
        "reason": "" if comparisons else "bridge paths lack matched primary cells",
        "comparisons": comparisons,
        "backend_factor_sensitivity": _backend_fixed_effect(regression_rows),
        "ensembles_are_independent_and_unpaired": True,
        "bridge_is_never_pooled_into_primary": True,
    }


def _sensitivity_status(primary: list[dict[str, Any]], all_rows: list[dict[str, Any]]) -> dict[str, Any]:
    expected = ("M128", "M512", "seam_M256", "radius3_M256", "rank4_M256")
    variants = {
        name: [row for row in all_rows if str(row.get("variant", "")) == name]
        for name in expected
    }
    primary_index = {
        (row["cell"], row["wall"], int(row["sample_id"]), row["direction"]): row
        for row in primary
    }
    summaries: list[dict[str, Any]] = []
    for name in expected:
        rows = variants[name]
        differences = []
        for row in rows:
            base = primary_index.get((row["cell"], row["wall"], int(row["sample_id"]), row["direction"]))
            if base is not None:
                differences.append(abs(row["endpoint_q_x"] - base["endpoint_q_x"]))
        summaries.append(
            {
                "variant": name,
                "expected_pairs": 410,
                "verified_pairs": len({row["task_id"] for row in rows}),
                "verified_directional_paths": len(rows),
                "matched_primary_paths": len(differences),
                "maximum_endpoint_change": max(differences, default=float("nan")),
                "status": (
                    "complete" if len({row["task_id"] for row in rows}) == 410 and len(differences) == 820
                    else "pending"
                ),
                "grid_stability_gate_1e-4": (
                    bool(max(differences) <= 1e-4) if differences and name in {"M128", "M512"} else None
                ),
            }
        )
    return {
        "status": "complete" if all(row["status"] == "complete" for row in summaries) else "pending",
        "variants": summaries,
    }


def analyze(
    output_root: Path,
    *,
    expected_pairs: int | None = None,
    expected_base_pairs: int | None = None,
    allow_partial: bool = False,
) -> dict[str, Any]:
    rows, invalid = load_path_rows(output_root, allow_invalid=allow_partial)
    primary = [row for row in rows if _is_primary(row)]
    pair_count = len({row["task_id"] for row in rows})
    primary_pair_count = len({row["task_id"] for row in primary})
    bridge_pair_count = len({row["task_id"] for row in rows if str(row.get("result_collection", "")) == "bridge_pump"})
    if expected_pairs is not None and primary_pair_count != expected_pairs and not allow_partial:
        raise RuntimeError(f"expected {expected_pairs} verified primary pairs, found {primary_pair_count}")
    base_pair_count = primary_pair_count + bridge_pair_count
    if expected_base_pairs is not None and base_pair_count != expected_base_pairs and not allow_partial:
        raise RuntimeError(
            f"expected {expected_base_pairs} verified base pairs, found {base_pair_count} "
            f"({primary_pair_count} primary + {bridge_pair_count} bridge)"
        )
    if not primary:
        raise RuntimeError("no primary wall-diabetic paths are available")
    analysis_root = Path(output_root) / "analysis"
    figures_root = analysis_root / "figures"
    scalar_fields = [
        "task_id", "cell", "protocol", "Nx", "Ny", "wall", "sample_id", "direction", "sigma",
        "is_primary", "source_backend", "result_collection", "variant", "control_kind",
        "grid_intervals", "edge_block_rank", "wall_window", "endpoint_q_x",
        "correctly_signed_endpoint_q_x",
        "endpoint_delta_N_left", "endpoint_delta_N_right", "endpoint_delta_N_total",
        "edge_crossing_index", "crossing_edge_internal_gap", "crossing_edge_external_gap",
        "crossing_edge_link_singular", "crossing_left_B_eigenvalue", "crossing_right_B_eigenvalue",
        "crossing_minimum_wall_weight", "crossing_wall_character_margin",
        "minimum_edge_internal_gap", "minimum_edge_external_gap", "minimum_edge_link_singular",
        "minimum_wall_character_margin", "maximum_total_charge_residual", "maximum_projector_residual",
        "endpoint_ordinary_q_x", "endpoint_instantaneous_q_x", "resolved", "unresolved_reason",
        "source_real_space_chern_mean", "source_real_space_chern_std",
        "endpoint_defect_particle", "endpoint_defect_hole", "endpoint_multicut_spread",
        "center_displacement", "particle_center_x", "hole_center_x", "result_path", "result_sha256",
    ]
    sample_csv = analysis_root / "samplewise_raw_qx.csv"
    _write_csv(primary, sample_csv, scalar_fields)
    all_paths_csv = analysis_root / "all_verified_raw_qx.csv"
    _write_csv(rows, all_paths_csv, scalar_fields)
    statistics = _statistics(primary)
    statistics_csv = analysis_root / "raw_directional_statistics.csv"
    _write_csv(statistics, statistics_csv, list(statistics[0]))

    curves: list[dict[str, Any]] = []
    for cell, wall, direction in sorted({(row["cell"], row["wall"], row["direction"]) for row in primary}):
        selected = [row for row in primary if (row["cell"], row["wall"], row["direction"]) == (cell, wall, direction)]
        mean_qx, sd_qx = _mean_sd(selected, "q_x")
        mean_left, sd_left = _mean_sd(selected, "delta_N_left")
        mean_right, sd_right = _mean_sd(selected, "delta_N_right")
        phi = selected[0]["phi"]
        for index in range(phi.size):
            curves.append(
                {
                    "cell": cell, "wall": wall, "direction": direction, "index": index,
                    "phi": float(phi[index]), "q_x_mean": float(mean_qx[index]), "q_x_sd": float(sd_qx[index]),
                    "delta_N_left_mean": float(mean_left[index]), "delta_N_left_sd": float(sd_left[index]),
                    "delta_N_right_mean": float(mean_right[index]), "delta_N_right_sd": float(sd_right[index]),
                }
            )
    curve_csv = analysis_root / "raw_directional_curves.csv"
    _write_csv(curves, curve_csv, list(curves[0]))

    _plot_style()
    figures = {
        "raw_paths": plot_raw_paths(primary, figures_root),
        "endpoint_histograms": plot_endpoint_histograms(primary, figures_root),
        "diagnostics": plot_diagnostics(primary, figures_root),
    }
    edge_gap_fits = _edge_gap_fits(primary)
    backend_bridge = _backend_bridge(primary, rows)
    sensitivity = _sensitivity_status(primary, rows)
    unresolved_reasons: dict[str, int] = {}
    for row in primary:
        if not row["resolved"]:
            reason = str(row["unresolved_reason"])
            unresolved_reasons[reason] = unresolved_reasons.get(reason, 0) + 1
    summary = {
        "schema": "wall_diabatic_width_sweep_analysis_v1",
        "output_root": str(Path(output_root).resolve()),
        "verified_pair_count": pair_count,
        "verified_primary_pair_count": primary_pair_count,
        "verified_optional_bridge_pair_count": bridge_pair_count,
        "verified_base_pair_count": base_pair_count,
        "raw_directional_path_count": len(rows),
        "primary_raw_directional_path_count": len(primary),
        "invalid_pair_count": len(invalid),
        "invalid_pairs": invalid,
        "expected_primary_pairs": expected_pairs,
        "expected_base_pairs": expected_base_pairs,
        "expected_gpu_bridge_pairs": 150,
        "gpu_route_expected_base_pairs": 1750,
        "cpu_fallback_route_expected_base_pairs": 1600,
        "partial_analysis": bool(allow_partial),
        "statistics": statistics,
        "resolved_primary_fraction": float(np.mean([row["resolved"] for row in primary])),
        "unresolved_reason_counts": unresolved_reasons,
        "edge_gap_exponential_fits": edge_gap_fits,
        "backend_bridge_sensitivity": backend_bridge,
        "sensitivity_status": sensitivity,
        "samplewise_csv": str(sample_csv),
        "all_verified_paths_csv": str(all_paths_csv),
        "curve_csv": str(curve_csv),
        "statistics_csv": str(statistics_csv),
        "figures": figures,
        "analysis_source_sha256": sha256_path(Path(__file__).resolve()),
        "verified_config_hashes": sorted({str(row.get("config_hash", "")) for row in rows}),
        "verified_source_hash_sets": sorted({json.dumps(row.get("source_hashes", {}), sort_keys=True) for row in rows}),
    }
    summary_path = analysis_root / "analysis_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--expected-pairs", type=int)
    parser.add_argument("--expected-base-pairs", type=int)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    analyze(
        args.output_root,
        expected_pairs=args.expected_pairs,
        expected_base_pairs=args.expected_base_pairs,
        allow_partial=args.allow_partial,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
