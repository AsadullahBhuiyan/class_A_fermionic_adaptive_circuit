"""Trajectory-first modular-packet reduction for the S100 CPU campaign."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ANALYSIS_SCHEMA = "modular_handedness_packet_product_v1"


def half_window_indices(nx: int, ny: int, y0: int) -> np.ndarray:
    """Legacy basis order: ``y_rel, x, orbital`` for a translated half cylinder."""
    nx, ny, y0 = int(nx), int(ny), int(y0)
    return np.asarray(
        [
            orbital + 2 * x + 2 * nx * ((y0 + y_rel) % ny)
            for y_rel in range(ny // 2)
            for x in range(nx)
            for orbital in range(2)
        ],
        dtype=np.int64,
    )


def restricted_centered_covariance_from_frame(
    frame: np.ndarray, *, nx: int, ny: int, y0: int
) -> np.ndarray:
    """Return legacy ``G_A=2 F_A F_A^dagger-I`` without forming full covariance."""
    frame = np.asarray(frame, dtype=np.complex128)
    expected_rows = 2 * int(nx) * int(ny)
    if frame.ndim != 2 or frame.shape[0] != expected_rows:
        raise ValueError(f"Frame shape {frame.shape} is incompatible with {(nx, ny)}.")
    rows = frame[half_window_indices(nx, ny, y0)]
    correlation = rows @ rows.conj().T
    correlation = 0.5 * (correlation + correlation.conj().T)
    return 2.0 * correlation - np.eye(correlation.shape[0], dtype=np.complex128)


def restrict_legacy_full_covariance(
    full_centered_covariance: np.ndarray, *, nx: int, ny: int, y0: int
) -> np.ndarray:
    """Reference implementation used to regression-test frame reconstruction."""
    idx = half_window_indices(nx, ny, y0)
    full = np.asarray(full_centered_covariance, dtype=np.complex128)
    return full[np.ix_(idx, idx)]


def packet_specs(nx: int, ny: int, wall_x: Iterable[int] | None = None) -> list[dict[str, int | str]]:
    ny_sub = int(ny) // 2
    walls = list(wall_x) if wall_x is not None else [int(nx) // 4, 3 * int(nx) // 4]
    return [
        {"label": f"x{x}_s{s}", "x": int(x), "source": int(s), "y_rel": 0 if s == 0 else ny_sub - 1}
        for x in walls
        for s in (0, 1)
    ]


def _site_index(nx: int, x: int, y_rel: int, orbital: int) -> int:
    return int(orbital) + 2 * int(x) + 2 * int(nx) * int(y_rel)


def _periodic_x_mask(nx: int, center: int, radius: int) -> np.ndarray:
    x = np.arange(int(nx))
    distance = np.minimum(np.abs(x - int(center)), int(nx) - np.abs(x - int(center)))
    return distance <= int(radius)


def _linear_fit(times: np.ndarray, values: np.ndarray, window: tuple[float, float]) -> tuple[float, float, float, int]:
    mask = (times >= float(window[0]) - 1e-12) & (times <= float(window[1]) + 1e-12)
    x = np.asarray(times[mask], dtype=np.float64)
    y = np.asarray(values[mask], dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < 3:
        return np.nan, np.nan, np.nan, int(len(x))
    design = np.column_stack((x, np.ones_like(x)))
    slope, intercept = np.linalg.lstsq(design, y, rcond=None)[0]
    residual = y - (slope * x + intercept)
    denom = np.sum((y - y.mean()) ** 2)
    r2 = 1.0 - float(np.sum(residual**2) / denom) if denom > 0 else float(np.allclose(residual, 0.0))
    return float(slope), float(intercept), float(r2), int(len(x))


def _evolve_packets(
    *,
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    eps: float,
    nx: int,
    ny: int,
    specs: list[dict[str, Any]],
    times: np.ndarray,
    radii: list[int],
    time_chunk: int,
) -> dict[str, np.ndarray]:
    clipped = np.clip(np.asarray(eigenvalues).real, -1.0 + float(eps), 1.0 - float(eps))
    modular_spectrum = -2.0 * np.arctanh(clipped)
    vectors = np.asarray(eigenvectors, dtype=np.complex128)
    occupied = np.asarray(
        [_site_index(nx, int(spec["x"]), int(spec["y_rel"]), orbital) for spec in specs for orbital in (0, 1)],
        dtype=np.int64,
    )
    coefficients = vectors[occupied, :].conj().T
    p_count, t_count, ny_sub = len(specs), len(times), int(ny) // 2
    density = np.zeros((len(radii), p_count, t_count, ny_sub), dtype=np.float64)
    total_norm = np.empty((p_count, t_count), dtype=np.float64)
    masks = [[_periodic_x_mask(nx, int(spec["x"]), r) for spec in specs] for r in radii]
    for start in range(0, t_count, int(time_chunk)):
        stop = min(t_count, start + int(time_chunk))
        phase = np.exp(-1j * times[start:stop, None] * modular_spectrum[None, :])
        amplitudes = np.einsum("ik,tk,ka->tia", vectors, phase, coefficients, optimize=True)
        probability = np.abs(amplitudes) ** 2
        for packet_index in range(p_count):
            cells = probability[:, :, 2 * packet_index : 2 * packet_index + 2].sum(axis=2)
            cells = cells.reshape(stop - start, ny_sub, int(nx), 2).sum(axis=3).real
            total_norm[packet_index, start:stop] = cells.sum(axis=(1, 2))
            for radius_index in range(len(radii)):
                density[radius_index, packet_index, start:stop] = cells[:, :, masks[radius_index][packet_index]].sum(axis=2)
    retention = density.sum(axis=3) / 2.0
    y_coordinate = np.arange(ny_sub, dtype=np.float64)
    denominator = np.maximum(density.sum(axis=3), 1e-300)
    center = np.einsum("rpty,y->rpt", density, y_coordinate, optimize=True) / denominator
    displacement = center - center[:, :, [0]]
    return {
        "longitudinal_density": density,
        "retention": retention,
        "displacement": displacement,
        "norm_drift": np.max(np.abs(total_norm - total_norm[:, [0]]), axis=1),
    }


def analyze_frame(frame: np.ndarray, config: dict[str, Any], *, cut_progress=None) -> dict[str, np.ndarray]:
    """Reduce one trajectory without pooling it with any other trajectory."""
    geometry = config["geometry"]
    analysis = config["modular_analysis"]
    nx, ny = int(geometry["Nx"]), int(geometry["Ny"])
    eps_values = [float(value) for value in analysis["spectral_cutoffs"]]
    radii = [int(value) for value in analysis["wall_window_radii"]]
    windows = [tuple(map(float, value)) for value in analysis["fit_windows"]]
    times = np.arange(
        int(round((float(analysis["time_stop"]) - float(analysis["time_start"])) / float(analysis["time_step"]))) + 1,
        dtype=np.float64,
    ) * float(analysis["time_step"]) + float(analysis["time_start"])
    y0_values = np.arange(ny, dtype=np.int64)
    specs = packet_specs(nx, ny, analysis["wall_x"])
    shape_path = (len(eps_values), len(radii), len(specs), len(times))
    displacement_sum = np.zeros(shape_path, dtype=np.float64)
    density_sum = np.zeros(shape_path + (ny // 2,), dtype=np.float64)
    retention_sum = np.zeros(shape_path, dtype=np.float64)
    cut_slopes = np.empty((len(eps_values), len(radii), len(windows), ny, len(specs)), dtype=np.float64)
    cut_intercepts = np.empty_like(cut_slopes)
    cut_r2 = np.empty_like(cut_slopes)
    cut_fit_points = np.empty(cut_slopes.shape, dtype=np.int16)
    norm_drift = np.empty((len(eps_values), ny, len(specs)), dtype=np.float64)
    clip_counts = np.empty((len(eps_values), ny, 2), dtype=np.int32)
    covariance_hermiticity = np.empty(ny, dtype=np.float64)

    for y0_index, y0 in enumerate(y0_values):
        restricted = restricted_centered_covariance_from_frame(frame, nx=nx, ny=ny, y0=int(y0))
        covariance_hermiticity[y0_index] = float(np.max(np.abs(restricted - restricted.conj().T)))
        restricted = 0.5 * (restricted + restricted.conj().T)
        eigenvalues, eigenvectors = np.linalg.eigh(restricted)
        for eps_index, eps in enumerate(eps_values):
            clip_counts[eps_index, y0_index] = (
                int(np.count_nonzero(eigenvalues < -1.0 + eps)),
                int(np.count_nonzero(eigenvalues > 1.0 - eps)),
            )
            evolved = _evolve_packets(
                eigenvalues=eigenvalues,
                eigenvectors=eigenvectors,
                eps=eps,
                nx=nx,
                ny=ny,
                specs=specs,
                times=times,
                radii=radii,
                time_chunk=int(analysis["time_chunk"]),
            )
            displacement_sum[eps_index] += evolved["displacement"]
            density_sum[eps_index] += evolved["longitudinal_density"]
            retention_sum[eps_index] += evolved["retention"]
            norm_drift[eps_index, y0_index] = evolved["norm_drift"]
            for radius_index in range(len(radii)):
                for window_index, window in enumerate(windows):
                    for packet_index in range(len(specs)):
                        values = evolved["displacement"][radius_index, packet_index]
                        fit = _linear_fit(times, values, window)
                        cut_slopes[eps_index, radius_index, window_index, y0_index, packet_index] = fit[0]
                        cut_intercepts[eps_index, radius_index, window_index, y0_index, packet_index] = fit[1]
                        cut_r2[eps_index, radius_index, window_index, y0_index, packet_index] = fit[2]
                        cut_fit_points[eps_index, radius_index, window_index, y0_index, packet_index] = fit[3]
        if cut_progress is not None:
            cut_progress(1)

    displacement_mean = displacement_sum / float(ny)
    density_mean = density_sum / float(ny)
    retention_mean = retention_sum / float(ny)
    trajectory_slopes = np.empty((len(eps_values), len(radii), len(windows), len(specs)), dtype=np.float64)
    trajectory_intercepts = np.empty_like(trajectory_slopes)
    trajectory_r2 = np.empty_like(trajectory_slopes)
    for e in range(len(eps_values)):
        for r in range(len(radii)):
            for w, window in enumerate(windows):
                for p in range(len(specs)):
                    fit = _linear_fit(times, displacement_mean[e, r, p], window)
                    trajectory_slopes[e, r, w, p] = fit[0]
                    trajectory_intercepts[e, r, w, p] = fit[1]
                    trajectory_r2[e, r, w, p] = fit[2]

    endpoint_orientation = np.asarray([1.0 if int(spec["source"]) == 0 else -1.0 for spec in specs])
    oriented_displacement = displacement_mean * endpoint_orientation[None, None, :, None]
    # H_s remains coordinate locked: compare x=15 and x=5 at fixed source endpoint.
    handedness = 0.5 * (trajectory_slopes[..., 2:4] - trajectory_slopes[..., 0:2])
    return {
        "schema": np.asarray(ANALYSIS_SCHEMA),
        "times": times,
        "y0_values": y0_values,
        "spectral_cutoffs": np.asarray(eps_values),
        "wall_window_radii": np.asarray(radii, dtype=np.int64),
        "fit_windows": np.asarray(windows),
        "packet_labels": np.asarray([spec["label"] for spec in specs]),
        "packet_x": np.asarray([spec["x"] for spec in specs], dtype=np.int64),
        "packet_source": np.asarray([spec["source"] for spec in specs], dtype=np.int64),
        "packet_y_rel": np.asarray([spec["y_rel"] for spec in specs], dtype=np.int64),
        "endpoint_orientation": endpoint_orientation,
        "displacement_mean": displacement_mean,
        "oriented_displacement_mean": oriented_displacement,
        "longitudinal_density_mean": density_mean,
        "retention_mean": retention_mean,
        "trajectory_velocity": trajectory_slopes,
        "trajectory_intercept": trajectory_intercepts,
        "trajectory_velocity_r2": trajectory_r2,
        "handedness_by_source": handedness,
        "cut_velocity": cut_slopes,
        "cut_intercept": cut_intercepts,
        "cut_velocity_r2": cut_r2,
        "cut_fit_points": cut_fit_points,
        "norm_drift_by_cut": norm_drift,
        "clip_counts": clip_counts,
        "covariance_hermiticity_error": covariance_hermiticity,
    }


def validate_analysis_arrays(arrays: dict[str, np.ndarray], config: dict[str, Any]) -> None:
    a = config["modular_analysis"]
    g = config["geometry"]
    expected_time = int(round((a["time_stop"] - a["time_start"]) / a["time_step"])) + 1
    expected = (len(a["spectral_cutoffs"]), len(a["wall_window_radii"]), 4, expected_time)
    if str(np.asarray(arrays["schema"]).item()) != ANALYSIS_SCHEMA:
        raise ValueError("Analysis schema mismatch.")
    if np.asarray(arrays["displacement_mean"]).shape != expected:
        raise ValueError("Analysis displacement shape mismatch.")
    if np.asarray(arrays["longitudinal_density_mean"]).shape != expected + (int(g["Ny"]) // 2,):
        raise ValueError("Analysis longitudinal-density shape mismatch.")
    for key in ("displacement_mean", "longitudinal_density_mean", "trajectory_velocity", "retention_mean"):
        if not np.all(np.isfinite(arrays[key])):
            raise FloatingPointError(f"Non-finite analysis field: {key}.")


def _bootstrap_mean_ci(values: np.ndarray, draws: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim < 1 or len(values) == 0:
        raise ValueError("Bootstrap needs at least one trajectory.")
    rng = np.random.default_rng(int(seed))
    flattened = values.reshape(len(values), -1)
    boot = np.empty((int(draws), flattened.shape[1]), dtype=np.float32)
    for start in range(0, int(draws), 128):
        stop = min(int(draws), start + 128)
        weights = rng.multinomial(len(values), np.full(len(values), 1.0 / len(values)), size=stop - start)
        boot[start:stop] = (weights @ flattened / float(len(values))).astype(np.float32)
    low, high = np.quantile(boot, [0.025, 0.975], axis=0)
    shape = values.shape[1:]
    return values.mean(axis=0), low.reshape(shape), high.reshape(shape)


def aggregate_and_plot(
    products: dict[str, list[Path]], config: dict[str, Any], output_dir: Path
) -> dict[str, Any]:
    """Aggregate only complete trajectory products and create the paper figure."""
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    a = config["modular_analysis"]
    eps_values = np.asarray(a["spectral_cutoffs"], dtype=float)
    radii = np.asarray(a["wall_window_radii"], dtype=int)
    windows = np.asarray(a["fit_windows"], dtype=float)
    primary_eps = int(np.argmin(np.abs(eps_values - float(a["primary_spectral_cutoff"]))))
    primary_radius = int(np.flatnonzero(radii == int(a["primary_wall_window_radius"]))[0])
    primary_window = int(np.flatnonzero(np.all(np.isclose(windows, a["primary_fit_window"]), axis=1))[0])
    draws, seed = int(a["bootstrap_draws"]), int(a["bootstrap_seed"])
    loaded: dict[str, dict[str, np.ndarray]] = {}
    summary: dict[str, Any] = {
        "schema": "modular_handedness_aggregate_v1",
        "bootstrap_draws": draws,
        "bootstrap_seed": seed,
        "primary_spectral_cutoff": float(eps_values[primary_eps]),
        "primary_wall_window_radius": int(radii[primary_radius]),
        "primary_fit_window": windows[primary_window].tolist(),
        "constructions": {},
    }
    rows: list[dict[str, Any]] = []
    for construction, paths in products.items():
        records = []
        for path in sorted(paths):
            with np.load(path, allow_pickle=False) as data:
                records.append({key: np.asarray(data[key]) for key in data.files if key != "metadata_json"})
        if not records:
            raise ValueError(f"No analysis products for {construction}.")
        times = records[0]["times"]
        paths_array = np.stack([row["displacement_mean"][primary_eps, primary_radius] for row in records])
        velocity = np.stack([row["trajectory_velocity"] for row in records])
        handedness = np.stack([row["handedness_by_source"] for row in records])
        mean_path, path_low, path_high = _bootstrap_mean_ci(paths_array, draws, seed + (0 if construction == "hard" else 1))
        loaded[construction] = {
            "times": times,
            "paths": paths_array,
            "path_mean": mean_path,
            "path_low": path_low,
            "path_high": path_high,
            "velocity": velocity,
            "handedness": handedness,
            "labels": records[0]["packet_labels"],
        }
        c_summary: dict[str, Any] = {"trajectories": len(records), "primary": {}}
        for packet in range(4):
            values = velocity[:, primary_eps, primary_radius, primary_window, packet]
            mean, low, high = _bootstrap_mean_ci(values, draws, seed + 100 + packet + (0 if construction == "hard" else 10))
            sign_fraction = float(np.mean(values > 0.0))
            label = str(records[0]["packet_labels"][packet])
            c_summary["primary"][label] = {
                "velocity_mean": float(mean), "ci_low": float(low), "ci_high": float(high),
                "positive_sign_fraction": sign_fraction,
                "mean_r2": float(np.mean([row["trajectory_velocity_r2"][primary_eps, primary_radius, primary_window, packet] for row in records])),
            }
            rows.append({"construction": construction, "quantity": "velocity", "source": label, **c_summary["primary"][label]})
        c_summary["handedness"] = {}
        for source in range(2):
            values = handedness[:, primary_eps, primary_radius, primary_window, source]
            mean, low, high = _bootstrap_mean_ci(values, draws, seed + 200 + source + (0 if construction == "hard" else 10))
            item = {"mean": float(mean), "ci_low": float(low), "ci_high": float(high), "positive_sign_fraction": float(np.mean(values > 0))}
            c_summary["handedness"][f"source_{source}"] = item
            rows.append({"construction": construction, "quantity": "H", "source": f"source_{source}", **item})
        summary["constructions"][construction] = c_summary

    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["CMU Sans Serif", "DejaVu Sans"],
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 6.5,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.8,
    })
    fig, axes = plt.subplots(2, 2, figsize=(7.05, 5.25), constrained_layout=True)
    colors = ["#0072B2", "#56B4E9", "#D55E00", "#E69F00"]
    for ax, construction, letter in ((axes[0, 0], "hard", "a"), (axes[0, 1], "soft", "b")):
        data = loaded[construction]
        for packet, color in enumerate(colors):
            label = str(data["labels"][packet])
            ax.plot(data["times"], data["path_mean"][packet], color=color, lw=1.2, label=label)
            ax.fill_between(data["times"], data["path_low"][packet], data["path_high"][packet], color=color, alpha=0.17, lw=0)
        ax.axhline(0, color="0.25", lw=0.7, ls="--")
        ax.set(xlabel=r"modular time $t_{\rm mod}$", ylabel=r"$\langle\Delta y\rangle$", title=f"{construction.capitalize()} wall")
        ax.legend(ncol=2, frameon=False)
        ax.text(-0.15, 1.04, f"({letter})", transform=ax.transAxes, fontsize=9)
    ax = axes[1, 0]
    x = np.arange(4)
    for ci, construction in enumerate(("hard", "soft")):
        values = loaded[construction]["velocity"][:, primary_eps, primary_radius, primary_window]
        means, lows, highs = [], [], []
        for packet in range(4):
            mean, low, high = _bootstrap_mean_ci(values[:, packet], draws, seed + 300 + ci * 10 + packet)
            means.append(float(mean)); lows.append(float(low)); highs.append(float(high))
        offset = -0.11 if ci == 0 else 0.11
        ax.errorbar(x + offset, means, yerr=[np.maximum(0.0, np.asarray(means) - lows), np.maximum(0.0, np.asarray(highs) - means)], fmt="o", ms=4,
                    capsize=2, label=construction, color=("#0072B2" if ci == 0 else "#D55E00"))
    ax.axhline(0, color="0.25", lw=0.7, ls="--")
    ax.set_xticks(x, [str(value) for value in loaded["hard"]["labels"]], rotation=20)
    ax.set_ylabel(r"raw $+y$ velocity")
    ax.legend(frameon=False)
    ax.text(-0.15, 1.04, "(c)", transform=ax.transAxes, fontsize=9)
    ax = axes[1, 1]
    window_labels = [f"{lo:g}--{hi:g}" for lo, hi in windows]
    eps_styles = ["o-", "s--", "^:"]
    for ci, construction in enumerate(("hard", "soft")):
        color = "#0072B2" if ci == 0 else "#D55E00"
        for ei, eps in enumerate(eps_values):
            values = loaded[construction]["handedness"][:, ei, primary_radius, :, 0]
            means, lows, highs = [], [], []
            for wi in range(len(windows)):
                mean, low, high = _bootstrap_mean_ci(values[:, wi], draws, seed + 400 + ci * 50 + ei * 10 + wi)
                means.append(float(mean)); lows.append(float(low)); highs.append(float(high))
            offset = (-0.07 if ci == 0 else 0.07) + (ei - 1) * 0.018
            ax.errorbar(np.arange(len(windows)) + offset, means,
                        yerr=[np.maximum(0.0, np.asarray(means) - lows), np.maximum(0.0, np.asarray(highs) - means)], fmt=eps_styles[ei],
                        lw=0.9, ms=3, capsize=1.5, color=color,
                        label=rf"{construction}, $\epsilon={eps:.0e}$")
    ax.axhline(0, color="0.25", lw=0.7, ls="--")
    ax.set_xticks(np.arange(len(windows)), window_labels)
    ax.set(xlabel="fit window", ylabel=r"$H_{s=0}=(v_{15,s}-v_{5,s})/2$")
    ax.legend(frameon=False, ncol=2, loc="best")
    ax.text(-0.15, 1.04, "(d)", transform=ax.transAxes, fontsize=9)
    for ax in axes.flat:
        ax.tick_params(direction="in", top=True, right=True)
    figure_base = output_dir / "modular_handedness_s100"
    fig.savefig(figure_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(figure_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    with (output_dir / "trajectory_bootstrap_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader(); writer.writerows(rows)
    (output_dir / "analysis_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary
