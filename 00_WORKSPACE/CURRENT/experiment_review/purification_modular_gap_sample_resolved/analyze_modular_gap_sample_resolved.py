#!/usr/bin/env python3
"""Build sample-resolved modular-spectrum and Lyapunov-gap figures.

The primary inputs are the checksum-pinned hard-v2 and soft-v3 maximally
mixed purification campaigns at Nx=20, Ny=20,30,40, and T=4 Ny.  The older
hard-wall T=2 Ny campaign is used only as a visibly separate depth-limited
finite-size comparison.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[4]
ANALYSIS_ROOT = Path(__file__).resolve().parent
FIGURE_ROOT = ANALYSIS_ROOT / "figures"
TABLE_ROOT = ANALYSIS_ROOT / "tables"

BUNDLE_07 = (
    REPO_ROOT
    / "00_WORKSPACE/CURRENT/final_production_new_designs/07_maxmix_hard_soft_purification"
)
BUNDLE_04 = (
    REPO_ROOT
    / "00_WORKSPACE/CURRENT/final_production_new_designs/04_maxmix_manybody_lyapunov_pilot"
)
OLD_SLOPES = (
    BUNDLE_04
    / "analysis_outputs/maxmix_manybody_lyapunov_nx20_ny20to40_s100_gpu_v2"
    / "trajectory_slopes_and_record_rates.csv"
)

NY_VALUES = (20, 30, 40)
SAMPLES = 100
CAP_TOLERANCE = 1.0e-9
WINDOWS = {
    "W1_Ny_to_2Ny": (1, 2),
    "W2_2Ny_to_3Ny": (2, 3),
    "W3_3Ny_to_4Ny": (3, 4),
}

COLORS = {20: "#d62728", 30: "#2ca02c", 40: "#1f77b4"}
MARKERS = {20: "^", 30: "s", 40: "o"}
LINESTYLES = {20: ":", 30: "--", 40: "-"}


@dataclass(frozen=True)
class CampaignSpec:
    construction: str
    revision: str
    root: Path
    completion_schema: str
    result_schema: str


SPECS = {
    "hard": CampaignSpec(
        construction="hard",
        revision="maxmix_hard_soft_purification_nx20_ny20-40_s100_4ny_raster_v2",
        root=(
            BUNDLE_07
            / "gpu_data/maxmix_hard_soft_purification_nx20_ny20-40_s100_4ny_raster_v2/hard"
        ),
        completion_schema="maxmix_purification_completion_v2",
        result_schema="maxmix_purification_result_v2",
    ),
    "soft": CampaignSpec(
        construction="soft",
        revision="maxmix_hard_soft_purification_nx20_ny20-40_s100_4ny_raster_v3",
        root=(
            BUNDLE_07
            / "gpu_data/maxmix_hard_soft_purification_nx20_ny20-40_s100_4ny_raster_v3/soft"
        ),
        completion_schema="maxmix_purification_completion_v3",
        result_schema="maxmix_purification_result_v3",
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sem(values: np.ndarray, axis: int | None = None) -> np.ndarray | float:
    values = np.asarray(values, dtype=np.float64)
    count = values.shape[axis] if axis is not None else values.size
    if count < 2:
        if axis is None:
            return math.nan
        shape = list(values.shape)
        del shape[axis]
        return np.full(shape, np.nan)
    return np.std(values, axis=axis, ddof=1) / math.sqrt(count)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def modular_flip_costs(occupations: np.ndarray) -> np.ndarray:
    """Return |log[(1-nu)/nu]|, retaining exact caps as infinite costs."""

    nu = np.asarray(occupations, dtype=np.float64)
    if not np.all(np.isfinite(nu)):
        raise FloatingPointError("occupation spectrum contains nonfinite values")
    residual = max(
        0.0,
        float(-nu.min(initial=0.0)),
        float(nu.max(initial=1.0) - 1.0),
    )
    if residual > CAP_TOLERANCE:
        raise FloatingPointError(f"occupation bound residual {residual:.3e} exceeds tolerance")
    caps = (nu <= CAP_TOLERANCE) | (nu >= 1.0 - CAP_TOLERANCE)
    safe = np.clip(nu, CAP_TOLERANCE, 1.0 - CAP_TOLERANCE)
    costs = np.abs(np.log(safe) - np.log1p(-safe))
    costs[caps] = np.inf
    return costs


def _window_slope(cycles: np.ndarray, d1: np.ndarray, start: int, stop: int) -> float:
    selected = (cycles >= start) & (cycles <= stop)
    if not np.all(np.isfinite(d1[selected])):
        return math.nan
    x = np.asarray(cycles[selected], dtype=np.float64)
    y = np.asarray(d1[selected], dtype=np.float64)
    if x.size < 3:
        return math.nan
    return float(np.polyfit(x, y, 1)[0])


def _validate_completion(
    *, spec: CampaignSpec, ny: int, result_path: Path, completion_path: Path
) -> dict[str, Any]:
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    expected = {
        "schema": spec.completion_schema,
        "sampling_revision": spec.revision,
        "construction": spec.construction,
        "Nx": 20,
        "Ny": ny,
        "cycles": 4 * ny,
        "dtype": "complex128",
        "canonical_dynamics_entry_point": "classA_U1FGTN_gpu.run_markov_circuit",
    }
    for key, value in expected.items():
        if completion.get(key) != value:
            raise RuntimeError(f"{completion_path}: {key} mismatch")
    if completion.get("result_filename") != result_path.name:
        raise RuntimeError(f"{completion_path}: result filename mismatch")
    if int(completion.get("result_bytes", -1)) != result_path.stat().st_size:
        raise RuntimeError(f"{result_path}: byte-count mismatch")
    if completion.get("result_sha256") != sha256_file(result_path):
        raise RuntimeError(f"{result_path}: SHA-256 mismatch")
    return completion


def load_primary_campaigns() -> tuple[
    dict[tuple[str, int], dict[str, np.ndarray]],
    list[dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    datasets: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    slope_rows: list[dict[str, Any]] = []
    provenance: dict[str, dict[str, Any]] = {}

    for construction, spec in SPECS.items():
        common_config_hash: str | None = None
        common_source_hashes: dict[str, str] | None = None
        campaign_files = 0
        campaign_bytes = 0
        for ny in NY_VALUES:
            result_paths = sorted((spec.root / f"Ny{ny:03d}").glob("*.npz"))
            if len(result_paths) != 20:
                raise RuntimeError(f"{construction}, Ny={ny}: expected 20 shards")
            sample_ids: list[int] = []
            d1_blocks: list[np.ndarray] = []
            cycles_reference: np.ndarray | None = None
            for result_path in result_paths:
                completion_path = result_path.with_suffix(".complete.json")
                if not completion_path.is_file():
                    raise RuntimeError(f"missing completion file: {completion_path}")
                completion = _validate_completion(
                    spec=spec,
                    ny=ny,
                    result_path=result_path,
                    completion_path=completion_path,
                )
                if common_config_hash is None:
                    common_config_hash = str(completion["configuration_hash"])
                    common_source_hashes = dict(completion["source_hashes"])
                if completion["configuration_hash"] != common_config_hash:
                    raise RuntimeError(f"{completion_path}: configuration identity drift")
                if completion["source_hashes"] != common_source_hashes:
                    raise RuntimeError(f"{completion_path}: source identity drift")
                with np.load(result_path, allow_pickle=False) as data:
                    if str(data["result_schema"].item()) != spec.result_schema:
                        raise RuntimeError(f"{result_path}: result schema mismatch")
                    cycles = np.asarray(data["cycles"], dtype=np.int64)
                    ids = np.asarray(data["sample_indices"], dtype=np.int64)
                    occupations = np.asarray(data["occupation_spectrum"], dtype=np.float64)
                expected_cycles = np.arange(4 * ny + 1, dtype=np.int64)
                if not np.array_equal(cycles, expected_cycles):
                    raise RuntimeError(f"{result_path}: cycle grid mismatch")
                if occupations.shape != (5, 4 * ny + 1, 40 * ny):
                    raise RuntimeError(f"{result_path}: occupation shape mismatch")
                if ids.shape != (5,) or ids.tolist() != completion["sample_indices"]:
                    raise RuntimeError(f"{result_path}: sample identity mismatch")
                costs = modular_flip_costs(occupations)
                d1 = np.min(costs, axis=-1)
                if not np.allclose(d1[:, 0], 0.0, rtol=0.0, atol=2.0e-12):
                    raise RuntimeError(f"{result_path}: cycle-zero modular gap is not zero")
                sample_ids.extend(ids.tolist())
                d1_blocks.append(d1)
                cycles_reference = cycles
                campaign_files += 2
                campaign_bytes += result_path.stat().st_size + completion_path.stat().st_size

            order = np.argsort(sample_ids)
            ids_array = np.asarray(sample_ids, dtype=np.int64)[order]
            if not np.array_equal(ids_array, np.arange(SAMPLES)):
                raise RuntimeError(f"{construction}, Ny={ny}: sample IDs are not exactly 0..99")
            d1_all = np.concatenate(d1_blocks, axis=0)[order]
            assert cycles_reference is not None
            rates = np.full_like(d1_all, np.nan)
            rates[:, 1:] = d1_all[:, 1:] / cycles_reference[None, 1:]
            datasets[(construction, ny)] = {
                "sample_ids": ids_array,
                "cycles": cycles_reference,
                "d1": d1_all,
                "finite_time_rate": rates,
            }
            for position, sample_id in enumerate(ids_array):
                row: dict[str, Any] = {
                    "construction": construction,
                    "Nx": 20,
                    "Ny": ny,
                    "sample_index": int(sample_id),
                }
                for window, multiples in WINDOWS.items():
                    start, stop = (multiples[0] * ny, multiples[1] * ny)
                    value = _window_slope(cycles_reference, d1_all[position], start, stop)
                    row[f"{window}_gap_slope"] = value
                    row[f"{window}_resolved"] = bool(math.isfinite(value))
                slope_rows.append(row)

        provenance[construction] = {
            "sampling_revision": spec.revision,
            "root": str(spec.root.relative_to(REPO_ROOT)),
            "verified_files": campaign_files,
            "verified_bytes": campaign_bytes,
            "configuration_hash": common_config_hash,
            "source_hashes": common_source_hashes,
        }
    return datasets, slope_rows, provenance


def load_representative_occupations(
    *, construction: str, ny: int, sample_id: int
) -> np.ndarray:
    spec = SPECS[construction]
    for result_path in sorted((spec.root / f"Ny{ny:03d}").glob("*.npz")):
        with np.load(result_path, allow_pickle=False) as data:
            ids = np.asarray(data["sample_indices"], dtype=np.int64)
            positions = np.flatnonzero(ids == sample_id)
            if positions.size:
                return np.asarray(data["occupation_spectrum"][positions[0]], dtype=np.float64)
    raise RuntimeError(f"sample {sample_id} not found for {construction}, Ny={ny}")


def representative_sample(
    slope_rows: list[dict[str, Any]], construction: str
) -> int:
    rows = [
        row
        for row in slope_rows
        if row["construction"] == construction and row["Ny"] == 40
    ]
    key = "W3_3Ny_to_4Ny_gap_slope"
    resolved = [row for row in rows if math.isfinite(float(row[key]))]
    values = np.asarray([float(row[key]) for row in resolved])
    median = float(np.median(values))
    chosen = min(resolved, key=lambda row: abs(float(row[key]) - median))
    return int(chosen["sample_index"])


def cycle_summary_rows(
    datasets: dict[tuple[str, int], dict[str, np.ndarray]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for construction in SPECS:
        for ny in NY_VALUES:
            data = datasets[(construction, ny)]
            rates = data["finite_time_rate"]
            for position, cycle in enumerate(data["cycles"]):
                finite = np.isfinite(rates[:, position])
                values = rates[finite, position]
                rows.append(
                    {
                        "construction": construction,
                        "Nx": 20,
                        "Ny": ny,
                        "cycle": int(cycle),
                        "normalized_cycle": float(cycle / ny),
                        "resolved_count": int(finite.sum()),
                        "resolved_fraction": float(finite.mean()),
                        "finite_time_gap_rate_mean": (
                            float(values.mean()) if values.size else math.nan
                        ),
                        "finite_time_gap_rate_sem": (
                            float(sem(values)) if values.size > 1 else math.nan
                        ),
                    }
                )
    return rows


def slope_summary_rows(slope_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for construction in SPECS:
        for ny in NY_VALUES:
            selected = [
                row
                for row in slope_rows
                if row["construction"] == construction and row["Ny"] == ny
            ]
            for window in WINDOWS:
                key = f"{window}_gap_slope"
                values = np.asarray(
                    [float(row[key]) for row in selected if math.isfinite(float(row[key]))]
                )
                rows.append(
                    {
                        "dataset": "primary_4Ny",
                        "construction": construction,
                        "Nx": 20,
                        "Ny": ny,
                        "depth": "4Ny",
                        "window": window,
                        "resolved_count": int(values.size),
                        "resolved_fraction": float(values.size / SAMPLES),
                        "gap_slope_mean": float(values.mean()) if values.size else math.nan,
                        "gap_slope_sem": float(sem(values)) if values.size > 1 else math.nan,
                        "Ny_times_gap_mean": float(ny * values.mean()) if values.size else math.nan,
                        "Ny_times_gap_sem": float(ny * sem(values)) if values.size > 1 else math.nan,
                    }
                )
    with OLD_SLOPES.open(newline="", encoding="utf-8") as handle:
        old_rows = list(csv.DictReader(handle))
    for ny in (20, 22, 24, 26, 28, 30, 36, 40):
        values = np.asarray(
            [float(row["late_gap_1"]) for row in old_rows if int(row["Ny"]) == ny],
            dtype=np.float64,
        )
        if values.size != SAMPLES:
            raise RuntimeError(f"older 2Ny Ny={ny}: expected 100 trajectories")
        rows.append(
            {
                "dataset": "independent_hard_2Ny_depth_limited",
                "construction": "hard",
                "Nx": 20,
                "Ny": ny,
                "depth": "2Ny",
                "window": "late_3Ny_over_2_to_2Ny",
                "resolved_count": int(values.size),
                "resolved_fraction": 1.0,
                "gap_slope_mean": float(values.mean()),
                "gap_slope_sem": float(sem(values)),
                "Ny_times_gap_mean": float(ny * values.mean()),
                "Ny_times_gap_sem": float(ny * sem(values)),
            }
        )
    return rows


def configure_plotting() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["CMU Sans Serif", "DejaVu Sans"],
            "mathtext.fontset": "cm",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "axes.linewidth": 0.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "legend.frameon": False,
            "legend.fontsize": 6.8,
            "savefig.bbox": None,
            "pdf.fonttype": 42,
        }
    )


def panel_label(axis: Any, label: str, x: float = -0.15) -> None:
    axis.text(x, 1.04, label, transform=axis.transAxes, fontweight="bold", va="bottom")


def save_figure(fig: Any, stem: str) -> list[Path]:
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    pdf = FIGURE_ROOT / f"{stem}.pdf"
    png = FIGURE_ROOT / f"{stem}.png"
    fig.savefig(pdf, bbox_inches=None)
    fig.savefig(png, dpi=300, bbox_inches=None)
    plt.close(fig)
    return [pdf, png]


def plot_representative_occupations(
    slope_rows: list[dict[str, Any]],
) -> tuple[list[Path], dict[str, int]]:
    configure_plotting()
    representatives = {
        construction: representative_sample(slope_rows, construction)
        for construction in SPECS
    }
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.55), constrained_layout=True)
    images = []
    for axis, construction, letter in zip(axes, ("hard", "soft"), ("(a)", "(b)")):
        sample_id = representatives[construction]
        occupations = load_representative_occupations(
            construction=construction, ny=40, sample_id=sample_id
        )
        sorted_occupations = np.sort(occupations, axis=1)
        image = axis.imshow(
            sorted_occupations.T,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            extent=(0.0, 4.0, 0.0, 1.0),
            cmap="RdBu_r",
            vmin=0.0,
            vmax=1.0,
            rasterized=True,
        )
        images.append(image)
        axis.set(
            xlabel=r"cycle number $t/N_y$",
            ylabel="ordered mode fraction" if construction == "hard" else "",
            title=rf"{construction} wall, $N_y=40$, sample {sample_id}",
        )
        panel_label(axis, letter)
    colorbar = fig.colorbar(images[0], ax=axes, pad=0.02, fraction=0.035)
    colorbar.set_label(r"occupation $\nu_a^\xi(t)$")
    return save_figure(fig, "purification_modular_occupation_spectrum"), representatives


def plot_every_sample_gap(
    datasets: dict[tuple[str, int], dict[str, np.ndarray]]
) -> list[Path]:
    configure_plotting()
    cmap = mpl.colormaps["viridis"].copy()
    cmap.set_bad("0.82")
    fig, axes = plt.subplots(3, 2, figsize=(7.05, 6.0), constrained_layout=True)
    last_image = None
    letters = iter("abcdef")
    for row, ny in enumerate(NY_VALUES):
        for column, construction in enumerate(("hard", "soft")):
            axis = axes[row, column]
            rates = datasets[(construction, ny)]["finite_time_rate"][:, 1:]
            last_image = axis.imshow(
                np.ma.masked_invalid(rates),
                origin="lower",
                aspect="auto",
                interpolation="nearest",
                extent=(1.0 / ny, 4.0, -0.5, 99.5),
                cmap=cmap,
                vmin=0.0,
                vmax=0.18,
                rasterized=True,
            )
            for boundary in (2.0, 3.0):
                axis.axvline(boundary, color="white", linestyle="--", linewidth=0.55, alpha=0.8)
            axis.set(
                xlabel=r"cycle number $t/N_y$" if row == 2 else "",
                ylabel="trajectory index" if column == 0 else "",
                title=rf"{construction}, $N_y={ny}$",
            )
            panel_label(axis, f"({next(letters)})", x=-0.12)
    assert last_image is not None
    colorbar = fig.colorbar(last_image, ax=axes, pad=0.012, fraction=0.022)
    colorbar.set_label(r"finite-time gap $d_1^\xi(t)/t$")
    colorbar.ax.text(
        0.5,
        1.015,
        "gray: unresolved",
        ha="center",
        va="bottom",
        transform=colorbar.ax.transAxes,
        fontsize=6,
    )
    return save_figure(fig, "purification_single_particle_gap_every_sample")


def _mean_sem_with_resolution(rates: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    means = np.full(rates.shape[1], np.nan)
    errors = np.full(rates.shape[1], np.nan)
    fractions = np.mean(np.isfinite(rates), axis=0)
    for position in range(rates.shape[1]):
        values = rates[np.isfinite(rates[:, position]), position]
        if values.size:
            means[position] = values.mean()
        if values.size > 1:
            errors[position] = sem(values)
    return means, errors, fractions


def _origin_fit(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.dot(x, y) / np.dot(x, x))


def plot_gap_summary(
    datasets: dict[tuple[str, int], dict[str, np.ndarray]],
    slope_summary: list[dict[str, Any]],
) -> list[Path]:
    configure_plotting()
    fig, axes = plt.subplots(2, 2, figsize=(7.05, 4.8), constrained_layout=True)

    for column, construction in enumerate(("hard", "soft")):
        axis = axes[0, column]
        for ny in NY_VALUES:
            data = datasets[(construction, ny)]
            x = data["cycles"][1:] / ny
            means, errors, fractions = _mean_sem_with_resolution(
                data["finite_time_rate"][:, 1:]
            )
            reliable = fractions >= 0.95
            axis.plot(
                x[reliable],
                means[reliable],
                color=COLORS[ny],
                marker=MARKERS[ny],
                markevery=max(1, ny // 4),
                markersize=3,
                linestyle=LINESTYLES[ny],
                linewidth=1.0,
                label=rf"$N_y={ny}$",
            )
            axis.fill_between(
                x[reliable],
                means[reliable] - errors[reliable],
                means[reliable] + errors[reliable],
                color=COLORS[ny],
                alpha=0.12,
                linewidth=0,
            )
            if np.any(~reliable):
                axis.plot(
                    x[~reliable],
                    means[~reliable],
                    color=COLORS[ny],
                    linestyle=":",
                    linewidth=0.75,
                    alpha=0.45,
                )
        for boundary in (2.0, 3.0):
            axis.axvline(boundary, color="0.55", linestyle="--", linewidth=0.6)
        axis.set(
            xlabel=r"cycle number $t/N_y$",
            ylabel=r"$\overline{d_1(t)/t}$" if column == 0 else "",
            title=f"{construction}-wall finite-time gap",
            xlim=(0.0, 4.02),
            ylim=(0.0, None),
        )
        axis.legend(ncol=3)
        panel_label(axis, "(a)" if column == 0 else "(b)")

    axis = axes[1, 0]
    series = [
        (
            "independent_hard_2Ny_depth_limited",
            "late_3Ny_over_2_to_2Ny",
            "#7f7f7f",
            "D",
            ":",
            r"hard, $T=2N_y$ late",
        ),
        (
            "primary_4Ny",
            "W2_2Ny_to_3Ny",
            "#2ca02c",
            "s",
            "--",
            r"hard, $T=4N_y$, $W_2$",
        ),
        (
            "primary_4Ny",
            "W3_3Ny_to_4Ny",
            "#1f77b4",
            "o",
            "-",
            r"hard, $T=4N_y$, $W_3$",
        ),
    ]
    for dataset, window, color, marker, linestyle, label in series:
        rows = [
            row
            for row in slope_summary
            if row["dataset"] == dataset
            and row["construction"] == "hard"
            and row["window"] == window
        ]
        rows.sort(key=lambda row: row["Ny"])
        x = 1.0 / np.asarray([row["Ny"] for row in rows], dtype=np.float64)
        y = np.asarray([row["gap_slope_mean"] for row in rows], dtype=np.float64)
        yerr = np.asarray([row["gap_slope_sem"] for row in rows], dtype=np.float64)
        coefficient = _origin_fit(x, y)
        axis.errorbar(
            x,
            y,
            yerr=yerr,
            color=color,
            marker=marker,
            markerfacecolor="white",
            linestyle="none",
            capsize=2,
            markersize=4,
            label=label + rf", $A={coefficient:.2f}$",
        )
        dense = np.linspace(0.0, x.max() * 1.04, 100)
        axis.plot(dense, coefficient * dense, color=color, linestyle=linestyle, linewidth=0.9)
    axis.set(
        xlabel=r"$1/N_y$",
        ylabel=r"$\overline{\Delta}_{\rm sp}$",
        title=r"hard-wall closure: $\Delta_{\rm sp}\simeq A/N_y$",
        xlim=(0.0, 0.053),
        ylim=(0.0, None),
    )
    axis.legend()
    panel_label(axis, "(c)")

    axis = axes[1, 1]
    for ny in NY_VALUES:
        data = datasets[("soft", ny)]
        fractions = np.mean(np.isfinite(data["finite_time_rate"]), axis=0)
        axis.plot(
            data["cycles"] / ny,
            fractions,
            color=COLORS[ny],
            marker=MARKERS[ny],
            markevery=max(1, ny // 4),
            markersize=3,
            linestyle=LINESTYLES[ny],
            linewidth=1.0,
            label=rf"$N_y={ny}$",
        )
    axis.axhline(0.95, color="0.35", linestyle="--", linewidth=0.7, label="95% resolved")
    for boundary in (2.0, 3.0):
        axis.axvline(boundary, color="0.65", linestyle="--", linewidth=0.55)
    axis.set(
        xlabel=r"cycle number $t/N_y$",
        ylabel="resolved trajectory fraction",
        title="soft-wall precision boundary",
        xlim=(0.0, 4.02),
        ylim=(-0.02, 1.03),
    )
    axis.legend(ncol=2)
    panel_label(axis, "(d)")
    return save_figure(fig, "purification_single_particle_gap_summary")


def save_derived_npz(datasets: dict[tuple[str, int], dict[str, np.ndarray]]) -> Path:
    payload: dict[str, np.ndarray] = {}
    for (construction, ny), data in datasets.items():
        prefix = f"{construction}_Ny{ny:03d}"
        for key in ("sample_ids", "cycles", "d1", "finite_time_rate"):
            payload[f"{prefix}_{key}"] = data[key]
    path = ANALYSIS_ROOT / "sample_resolved_modular_gap_dynamics.npz"
    np.savez_compressed(path, **payload)
    return path


def main() -> int:
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    TABLE_ROOT.mkdir(parents=True, exist_ok=True)
    datasets, trajectory_rows, provenance = load_primary_campaigns()
    cycle_rows = cycle_summary_rows(datasets)
    slope_rows = slope_summary_rows(trajectory_rows)

    trajectory_csv = TABLE_ROOT / "trajectory_modular_gap_slopes.csv"
    cycle_csv = TABLE_ROOT / "cycle_modular_gap_summary.csv"
    size_csv = TABLE_ROOT / "modular_gap_size_summary.csv"
    write_csv(trajectory_csv, trajectory_rows)
    write_csv(cycle_csv, cycle_rows)
    write_csv(size_csv, slope_rows)
    derived_npz = save_derived_npz(datasets)

    outputs: list[Path] = [trajectory_csv, cycle_csv, size_csv, derived_npz]
    occupation_outputs, representatives = plot_representative_occupations(trajectory_rows)
    outputs.extend(occupation_outputs)
    outputs.extend(plot_every_sample_gap(datasets))
    outputs.extend(plot_gap_summary(datasets, slope_rows))

    inventory_rows = [
        {
            "dataset": "primary hard v2 / soft v3",
            "Nx": 20,
            "Ny_values": "20;30;40",
            "constructions": "hard;soft",
            "samples_per_point": 100,
            "depth": "4Ny",
            "occupation_spectrum": "every cycle",
            "status": "complete; checksum verified; primary",
        },
        {
            "dataset": "maxmix many-body Lyapunov GPU v2",
            "Nx": 20,
            "Ny_values": "20;22;24;26;28;30;36;40",
            "constructions": "hard",
            "samples_per_point": 100,
            "depth": "2Ny",
            "occupation_spectrum": "stride 4 plus fit boundaries",
            "status": "complete; depth-limited comparison only",
        },
        {
            "dataset": "legacy purification dynamics maxmix",
            "Nx": 20,
            "Ny_values": "30;40;50",
            "constructions": "hard",
            "samples_per_point": 100,
            "depth": "2Ny",
            "occupation_spectrum": "not saved",
            "status": "complete; entropy/charge only; unusable for modular gap",
        },
        {
            "dataset": "transverse-width purification convergence",
            "Nx": "20;24;28",
            "Ny_values": 20,
            "constructions": "hard",
            "samples_per_point": 50,
            "depth": 100,
            "occupation_spectrum": "only extrema saved",
            "status": "complete; transverse-width diagnostic only",
        },
        {
            "dataset": "Nx16 hard/soft CPU 4Ny pilot",
            "Nx": 16,
            "Ny_values": "20;22;24;26;28;30",
            "constructions": "hard;soft",
            "samples_per_point": "2 total completed of 1200 intended",
            "depth": "4Ny",
            "occupation_spectrum": "planned stride 4",
            "status": "incomplete numerical failure; excluded",
        },
    ]
    inventory_csv = TABLE_ROOT / "purification_size_inventory.csv"
    write_csv(inventory_csv, inventory_rows)
    outputs.append(inventory_csv)

    output_records = []
    for path in sorted(outputs):
        output_records.append(
            {
                "path": str(path.relative_to(ANALYSIS_ROOT)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    manifest = {
        "analysis_schema": "purification_modular_gap_sample_resolved_v1",
        "primary_contract": {
            "Nx": 20,
            "Ny": list(NY_VALUES),
            "samples_per_construction_size": SAMPLES,
            "constructions": ["hard", "soft"],
            "cycles": "0..4Ny inclusive",
            "initialization": "maximally mixed",
            "uncertainty": "ordinary trajectory standard error; no bootstrap",
            "cap_tolerance": CAP_TOLERANCE,
        },
        "estimator": {
            "single_particle_modular_energy": "epsilon_a=log[(1-nu_a)/nu_a]",
            "instantaneous_flip_cost": "d1(t)=min_a |epsilon_a(t)|",
            "finite_time_gap": "d1(t)/t",
            "trajectory_gap": "OLS slope of d1(t) against t within each window",
            "squared_singular_value_convention": "Delta_sp=lambda_0-lambda_1",
            "finite_size_model": "Delta_sp=A/Ny through the origin",
        },
        "representative_selection": {
            "rule": "Ny40 resolved trajectory closest to the median W3 single-particle gap slope, separately by construction",
            "sample_indices": representatives,
        },
        "input_provenance": provenance,
        "independent_2Ny_comparison": {
            "path": str(OLD_SLOPES.relative_to(REPO_ROOT)),
            "sha256": sha256_file(OLD_SLOPES),
            "pooled_with_primary": False,
            "reason": "all eight sizes failed at least one temporal-convergence gate at T=2Ny",
        },
        "outputs": output_records,
    }
    manifest_path = ANALYSIS_ROOT / "analysis_manifest.json"
    write_json(manifest_path, manifest)
    print(json.dumps({"representatives": representatives, "outputs": output_records}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
