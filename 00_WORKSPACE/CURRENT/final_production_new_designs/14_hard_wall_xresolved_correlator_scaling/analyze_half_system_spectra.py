#!/usr/bin/env python3
"""Pool the saved half-system occupations and their single-particle modular energies."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, LogFormatterSciNotation, NullLocator, MaxNLocator
import numpy as np


BUNDLE = Path(__file__).resolve().parent
REVISION = (
    "hard_wall_xresolved_nx20_ny40-50-60_a1-1_nsh1_s100_2ny_raster_"
    "endpoint_frame_halfcov_occupations_v2_30gib_batched"
)
DEFAULT_DATA = BUNDLE / "gpu_data" / REVISION
DEFAULT_OUTPUT = BUNDLE / "analysis_outputs" / "half_system_spectra_ny40_50_60_roundoff_filtered"
OCCUPATION_MARGIN = 1e-12
SIZES = (40, 50, 60)
STYLES = {
    40: dict(color="#c43c39", linestyle=":"),
    50: dict(color="#29904d", linestyle="--"),
    60: dict(color="#2674ba", linestyle="-"),
    "pooled": dict(color="#242424", linestyle="-"),
}


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def modular_energies(occupations: np.ndarray) -> np.ndarray:
    """No new clipping: exact saved 0/1 values retain +infinity/-infinity."""
    if not np.isfinite(occupations).all() or np.any(
        (occupations < 0) | (occupations > 1)
    ):
        raise ValueError("Invalid saved occupation spectrum")
    with np.errstate(divide="ignore"):
        return np.log1p(-occupations) - np.log(occupations)


def normalized_histogram(values: np.ndarray, edges: np.ndarray):
    counts, _ = np.histogram(values, bins=edges)
    if counts.sum() != values.size:
        raise ValueError("Histogram edges lost input values")
    density = counts / (values.size * np.diff(edges))
    np.testing.assert_allclose(np.sum(density * np.diff(edges)), 1.0, atol=1e-14)
    return counts, density


def load_spectra(root: Path):
    manifest = json.loads((root / "DOWNLOAD_MANIFEST.json").read_text())
    inventory = {entry["relative_path"]: entry for entry in manifest["files"]}
    occupations, diagnostics = {}, {}
    for ny in SIZES:
        by_sample = {}
        minima, maxima = [], []
        paths = sorted((root / "results" / f"Ny{ny:03d}").glob("*.npz"))
        if len(paths) != 20:
            raise ValueError(f"Ny={ny}: expected 20 result shards")
        for path in paths:
            receipt_path = path.with_suffix(".complete.json")
            receipt = json.loads(receipt_path.read_text())
            for file_path in (path, receipt_path):
                entry = inventory[file_path.relative_to(root).as_posix()]
                if file_path.stat().st_size != entry["bytes"] or digest(file_path) != entry["sha256"]:
                    raise ValueError(f"Import checksum mismatch: {file_path}")
            if receipt["result_sha256"] != inventory[path.relative_to(root).as_posix()]["sha256"]:
                raise ValueError(f"Result/receipt hash mismatch: {path}")
            if receipt["status"] != "complete" or receipt["sampling_revision"] != REVISION:
                raise ValueError(f"Incomplete or incompatible shard: {path}")
            with np.load(path, allow_pickle=False) as data:
                expected = {
                    "Nx": 20, "Ny": ny, "alpha_1": 1.0, "alpha_2": 30.0,
                    "nshell": 1, "sequence": "raster_y", "dtype": "complex128",
                    "dw_truncation": True, "meas_slab_only": True,
                    "perfect_correction": True, "sampling_revision": REVISION,
                    "configuration_sha256": manifest["configuration_sha256"],
                }
                for key, value in expected.items():
                    if data[key].item() != value:
                        raise ValueError(f"Scientific contract mismatch: {path.name}: {key}")
                np.testing.assert_array_equal(data["half_system_region_bounds"], [0, 20, 0, ny // 2])
                np.testing.assert_array_equal(data["cycles"], [2 * ny])
                np.testing.assert_array_equal(data["normalized_cycles"], [2.0])
                indices = data["global_sample_indices"]
                np.testing.assert_array_equal(indices, receipt["global_sample_indices"])
                values = data["half_system_occupation_spectrum"]
                if values.shape != (5, 1, 20 * ny) or values.dtype != np.float64:
                    raise ValueError(f"Unexpected occupation array: {path}")
                for sample, row in zip(indices, values[:, 0, :]):
                    if int(sample) in by_sample:
                        raise ValueError(f"Duplicate Ny={ny}, sample={sample}")
                    by_sample[int(sample)] = row.copy()
                minima.append(float(data["half_system_raw_occupation_minimum"]))
                maxima.append(float(data["half_system_raw_occupation_maximum"]))
        if set(by_sample) != set(range(100)):
            raise ValueError(f"Ny={ny}: sample coverage is not exactly 0..99")
        occupations[ny] = np.stack([by_sample[index] for index in range(100)])
        diagnostics[ny] = {"raw_minimum": min(minima), "raw_maximum": max(maxima)}
        print(f"[verified] Ny={ny}: 100 samples, {occupations[ny].size:,} occupations", flush=True)
    return occupations, diagnostics, manifest


def style_axes(axis):
    axis.set_yscale("log")
    axis.tick_params(which="both", direction="in", top=True, right=True)
    axis.yaxis.set_major_locator(LogLocator(base=10, numticks=5))
    axis.yaxis.set_minor_locator(NullLocator())
    axis.xaxis.set_major_locator(MaxNLocator(5))


def draw_histograms(axis, histograms, edges):
    # Zero-count bins stay empty on a log axis; no pseudocount is added.
    for key in ("pooled", *SIZES):
        _, density = histograms[key]
        style = STYLES[key]
        axis.stairs(
            np.where(density > 0, density, np.nan), edges, baseline=None,
            label="All sizes" if key == "pooled" else rf"$N_y={key}$",
            linewidth=1.7 if key == "pooled" else 0.9,
            alpha=0.8 if key == "pooled" else 1.0, **style,
        )
    style_axes(axis)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    occupations, diagnostics, manifest = load_spectra(args.data_root)
    energies = {ny: modular_energies(values) for ny, values in occupations.items()}
    all_occupations = {ny: values.ravel() for ny, values in occupations.items()}
    all_occupations["pooled"] = np.concatenate(list(all_occupations.values()))
    all_energies = {ny: values.ravel() for ny, values in energies.items()}
    all_energies["pooled"] = np.concatenate(list(all_energies.values()))
    finite = {key: values[np.isfinite(values)] for key, values in all_energies.items()}
    retained_masks = {
        key: (values > OCCUPATION_MARGIN) & (values < 1 - OCCUPATION_MARGIN)
        for key, values in all_occupations.items()
    }
    retained = {key: all_energies[key][mask] for key, mask in retained_masks.items()}
    energy_limit = float(np.log1p(-OCCUPATION_MARGIN) - np.log(OCCUPATION_MARGIN))
    occupation_edges = np.linspace(0.0, 1.0, 81)
    # Partial edge bins respect the filter support; densities use actual bin widths.
    modular_edges = np.r_[-energy_limit, np.arange(np.ceil(-energy_limit), np.floor(energy_limit) + 1), energy_limit]
    occupation_hist = {key: normalized_histogram(values, occupation_edges) for key, values in all_occupations.items()}
    modular_hist = {key: normalized_histogram(values, modular_edges) for key, values in retained.items()}
    summary = {
        "sampling_revision": REVISION,
        "source_root": str(args.data_root.resolve()),
        "download_manifest_sha256": digest(args.data_root / "DOWNLOAD_MANIFEST.json"),
        "configuration_sha256": manifest["configuration_sha256"],
        "subsystem": "A=[0,20)x[0,Ny//2), both physical orbitals",
        "samples_per_size": 100,
        "cycles_by_Ny": {str(ny): 2 * ny for ny in SIZES},
        "single_particle_modular_energy": "epsilon=log(1-nu)-log(nu)",
        "ensemble_order": "Transform each sample spectrum separately, then pool levels",
        "pooling": "Every eigenvalue has equal weight; size contributions scale with subsystem dimension",
        "occupation_normalization": "bin_count / (all_saved_levels * bin_width), area=1",
        "modular_normalization": "bin_count / (retained_levels * bin_width), retained-conditional area=1",
        "new_clipping": False,
        "modular_occupation_filter": "1e-12 < nu < 1-1e-12",
        "modular_occupation_margin": OCCUPATION_MARGIN,
        "modular_energy_limit": energy_limit,
        "near_endpoint_note": "Saved occupations were clipped into [0,1] by the acquisition observer; exact endpoint and far-tail energies do not determine physically resolved finite energies.",
        "groups": {},
    }
    npz = {"occupation_edges": occupation_edges, "modular_edges": modular_edges}
    for key, values in all_occupations.items():
        energy = all_energies[key]
        nu_count, nu_density = occupation_hist[key]
        en_count, en_density = modular_hist[key]
        record = {
            "levels": int(values.size),
            "finite_modular_levels": int(np.isfinite(energy).sum()),
            "saved_nu_zero_positive_infinity": int(np.isposinf(energy).sum()),
            "saved_nu_one_negative_infinity": int(np.isneginf(energy).sum()),
            "finite_fraction": float(np.isfinite(energy).mean()),
            "finite_modular_range": [float(finite[key].min()), float(finite[key].max())],
            "retained_modular_levels": int(retained[key].size),
            "excluded_modular_levels": int(values.size - retained[key].size),
            "retained_modular_fraction": float(retained[key].size / values.size),
            "retained_modular_range": [float(retained[key].min()), float(retained[key].max())],
            "occupation_histogram_integral": float(np.sum(nu_density * np.diff(occupation_edges))),
            "retained_modular_histogram_integral": float(np.sum(en_density * np.diff(modular_edges))),
            "levels_at_distance_at_most_1e_minus12_from_endpoints": int(np.count_nonzero((values <= 1e-12) | (values >= 1 - 1e-12))),
        }
        if key != "pooled":
            record.update(diagnostics[key])
            npz[f"sample_indices_Ny{key}"] = np.arange(100)
            npz[f"occupations_Ny{key}"] = occupations[key]
            npz[f"modular_energies_by_occupation_Ny{key}"] = energies[key]
            npz[f"modular_retained_mask_Ny{key}"] = retained_masks[key].reshape(occupations[key].shape)
            npz[f"retained_modular_energies_Ny{key}"] = retained[key]
        summary["groups"][str(key)] = record
        npz[f"occupation_counts_{key}"] = nu_count
        npz[f"occupation_density_{key}"] = nu_density
        npz[f"modular_counts_{key}"] = en_count
        npz[f"modular_density_{key}"] = en_density
        for label, edges, counts, density in (
            ("occupations", occupation_edges, nu_count, nu_density),
            ("modular", modular_edges, en_count, en_density),
        ):
            np.savetxt(output / f"histogram_{label}_{key}.csv", np.column_stack([edges[:-1], edges[1:], counts, density]),
                       delimiter=",", header="bin_left,bin_right,count,probability_density", comments="")

    # Verify the nonlinear transform and infinity convention on analytic examples.
    test = modular_energies(np.asarray([0.0, 0.25, 0.5, 0.75, 1.0]))
    np.testing.assert_allclose(test[1:-1], [np.log(3), 0, -np.log(3)])
    assert np.isposinf(test[0]) and np.isneginf(test[-1])
    for ny in SIZES:
        mask = np.isfinite(energies[ny])
        recovered = 1 / (1 + np.exp(energies[ny][mask]))
        np.testing.assert_allclose(recovered, occupations[ny][mask], rtol=3e-14, atol=3e-16)

    np.savez_compressed(output / "half_system_occupation_and_modular_spectra.npz", **npz)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["CMU Sans Serif"],
        "mathtext.fontset": "cm", "font.size": 9, "axes.labelsize": 10,
        "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.linewidth": 0.8, "pdf.fonttype": 42, "savefig.dpi": 300,
    })
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 3.25))
    draw_histograms(axes[0], occupation_hist, occupation_edges)
    axes[0].set(xlim=(0, 1), xlabel=r"occupation $\nu$", ylabel=r"probability density $p(\nu)$", title="Occupation spectrum")
    axes[0].legend(frameon=False, loc="upper center", ncol=2, columnspacing=1.0)
    draw_histograms(axes[1], modular_hist, modular_edges)
    axes[1].set(xlim=(modular_edges[0], modular_edges[-1]), xlabel=r"modular energy $\epsilon$",
                ylabel=r"probability density $p(\epsilon)$", title="Filtered modular spectrum")
    axes[1].text(0.5, 0.96, r"$10^{-12}<\nu<1-10^{-12}$",
                 transform=axes[1].transAxes, va="top", ha="center", fontsize=8)
    axes[1].yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
    axes[1].yaxis.set_major_formatter(LogFormatterSciNotation(base=10, minor_thresholds=(np.inf, np.inf)))
    for letter, axis in zip("ab", axes):
        axis.text(-0.19, 1.06, f"({letter})", transform=axis.transAxes, fontsize=10)
    fig.suptitle(r"$N_x=20$, $A_y=N_y/2$; $S=100$ per size, $T=2N_y$", y=0.99, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95), w_pad=1.3)
    for extension in ("pdf", "png"):
        fig.savefig(output / f"half_system_spectra_log_density.{extension}")
    plt.close(fig)

    # A central view uses exactly the same normalized histogram, with no renormalization.
    fig, axis = plt.subplots(figsize=(3.375, 2.75))
    draw_histograms(axis, modular_hist, modular_edges)
    axis.set(xlim=(-12, 12), xlabel=r"modular energy $\epsilon$",
             ylabel=r"probability density $p(\epsilon)$", title="Central modular spectrum")
    central = (modular_edges[:-1] >= -12) & (modular_edges[1:] <= 12)
    densities = np.concatenate([density[central] for _, density in modular_hist.values()])
    axis.set_ylim(densities[densities > 0].min() * 0.65, densities.max() * 2.1)
    axis.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
    axis.yaxis.set_major_formatter(LogFormatterSciNotation(base=10, minor_thresholds=(np.inf, np.inf)))
    axis.set_xticks([-10, -5, 0, 5, 10])
    axis.legend(frameon=False, ncol=2, loc="upper center", columnspacing=1.2)
    fig.tight_layout()
    for extension in ("pdf", "png"):
        fig.savefig(output / f"half_system_modular_central_log_density.{extension}")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(3.375, 3.0))
    draw_histograms(axis, modular_hist, modular_edges)
    axis.set(xlim=(-energy_limit, energy_limit), xlabel=r"modular energy $\epsilon$",
             ylabel=r"probability density $p(\epsilon)$", title="Filtered modular spectrum")
    all_densities = np.concatenate([density for _, density in modular_hist.values()])
    axis.set_ylim(all_densities[all_densities > 0].min() * 0.75, all_densities.max() * 2.0)
    axis.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
    axis.yaxis.set_major_formatter(LogFormatterSciNotation(base=10, minor_thresholds=(np.inf, np.inf)))
    axis.set_xticks([-20, -10, 0, 10, 20])
    axis.legend(frameon=False, ncol=2, loc="upper center", columnspacing=1.0)
    axis.text(0.5, 0.025, r"$10^{-12}<\nu<1-10^{-12}$",
              transform=axis.transAxes, ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    for extension in ("pdf", "png"):
        fig.savefig(output / f"half_system_modular_filtered_log_density.{extension}")
    plt.close(fig)
    print(json.dumps(summary["groups"]["pooled"], indent=2), flush=True)
    print(f"[complete] {output}", flush=True)


if __name__ == "__main__":
    main()
