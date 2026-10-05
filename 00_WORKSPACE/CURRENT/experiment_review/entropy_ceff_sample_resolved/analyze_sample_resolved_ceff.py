#!/usr/bin/env python3
"""Extract a central-charge slope per trajectory before ensemble averaging."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SOURCE_ROOT = (
    ROOT
    / "00_WORKSPACE/COLAB/colab_charge_fluctuations/gpu_data"
    / "streaming_covariance_observables/campaigns"
    / "N20_selected_nsh1_dwtrunc1_init-default_S100_cycles-2Ny/runs"
)
DATA_DIR = HERE / "data"
FIGURE_DIR = HERE / "figures"
CSV_PATH = DATA_DIR / "sample_resolved_ceff_summary.csv"
NPZ_PATH = DATA_DIR / "sample_resolved_ceff_arrays.npz"
PDF_PATH = FIGURE_DIR / "sample_resolved_ceff_dynamics_and_endpoints.pdf"
PNG_PATH = FIGURE_DIR / "sample_resolved_ceff_dynamics_and_endpoints.png"
MANIFEST_PATH = HERE / "analysis_manifest.json"

NX = 20
NY_VALUES = (30, 40, 50)
NSHELL = 1
SAMPLES = 100
FIT_AY_MIN = 8
PLOT_CYCLE_MIN = 10

STYLES = {
    30: {"color": "#D92725", "marker": "^", "linestyle": ":"},
    40: {"color": "#2CA02C", "marker": "s", "linestyle": "--"},
    50: {"color": "#1F77B4", "marker": "o", "linestyle": "-"},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_entropy(ny: int) -> tuple[Path, np.ndarray, np.ndarray, np.ndarray]:
    source = (
        SOURCE_ROOT
        / f"N20x{ny}_nsh1_init-default_perfect_correction"
        / "entropy_y0avg_vs_ay.npz"
    )
    if not source.is_file():
        raise FileNotFoundError(source)
    with np.load(source, allow_pickle=False) as payload:
        required = {
            "cycle_labels",
            "sample_indices",
            "config_json",
            "entropy_y0avg_vs_ay",
            "ay_values",
        }
        missing = required.difference(payload.files)
        if missing:
            raise ValueError(f"{source}: missing keys {sorted(missing)}")
        cycles = np.asarray(payload["cycle_labels"], dtype=np.int64)
        sample_indices = np.asarray(payload["sample_indices"], dtype=np.int64)
        entropy = np.asarray(payload["entropy_y0avg_vs_ay"], dtype=np.float64)
        ay = np.asarray(payload["ay_values"], dtype=np.int64)
        config = json.loads(str(payload["config_json"].item()))

    contract = {
        "Nx": NX,
        "Ny": ny,
        "nshell": NSHELL,
        "samples_actual": SAMPLES,
        "cycles": 2 * ny,
        "perfect_correction": True,
        "postselect": False,
        "sequence": "raster_y",
        "dtype": "complex128",
        "alpha_1": 1.0,
        "alpha_2": 30.0,
        "dw_truncation": True,
        "init_mode": "default",
    }
    mismatches = {
        key: {"expected": expected, "actual": config.get(key)}
        for key, expected in contract.items()
        if config.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"{source}: scientific-contract mismatch: {mismatches}")
    if entropy.shape != (SAMPLES, 2 * ny, ny // 2 + 1):
        raise ValueError(f"{source}: unexpected entropy shape {entropy.shape}")
    if not np.array_equal(sample_indices, np.arange(SAMPLES)):
        raise ValueError(f"{source}: sample indices are not 0,...,{SAMPLES - 1}")
    if not np.array_equal(cycles, np.arange(1, 2 * ny + 1)):
        raise ValueError(f"{source}: cycles are not 1,...,{2 * ny}")
    if not np.array_equal(ay, np.arange(ny // 2 + 1)):
        raise ValueError(f"{source}: unexpected Ay grid")
    if not np.isfinite(entropy).all():
        raise FloatingPointError(f"{source}: nonfinite entropy")
    return source, cycles, ay, entropy


def trajectory_ceff(entropy: np.ndarray, ay: np.ndarray, ny: int) -> np.ndarray:
    """Return c_eff for every sample and cycle using one locked fit window."""
    selected = (ay >= FIT_AY_MIN) & (ay <= ny // 2)
    if np.count_nonzero(selected) < 2:
        raise ValueError(f"Ny={ny}: too few fit points")
    x = np.log((ny / np.pi) * np.sin(np.pi * ay[selected] / ny))
    centered = x - x.mean()
    denominator = float(centered @ centered)
    return 3.0 * np.einsum("sta,a->st", entropy[:, :, selected], centered) / denominator


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.family": "sans-serif",
            "font.sans-serif": ["CMU Sans Serif", "DejaVu Sans"],
            "mathtext.fontset": "cm",
            "font.size": 8.0,
            "axes.labelsize": 8.0,
            "axes.titlesize": 8.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 7.0,
            "axes.linewidth": 0.8,
            "lines.markersize": 3.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 3.2,
            "ytick.major.size": 3.2,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def make_figure(results: dict[int, dict[str, np.ndarray]]) -> None:
    configure_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.75))

    ax = axes[0]
    for ny in NY_VALUES:
        result = results[ny]
        keep = result["cycles"] >= PLOT_CYCLE_MIN
        x = result["cycles"][keep] / ny
        style = STYLES[ny]
        ax.fill_between(
            x,
            result["mean"][keep] - result["sem"][keep],
            result["mean"][keep] + result["sem"][keep],
            color=style["color"],
            alpha=0.13,
            linewidth=0,
        )
        ax.plot(
            x,
            result["mean"][keep],
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            markerfacecolor="white",
            markeredgewidth=0.8,
            markevery=max(1, ny // 10),
            linewidth=1.0,
            label=rf"$N_y={ny}$",
        )
    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlim(0.18, 2.02)
    ax.set_ylim(0.90, 2.55)
    ax.set_xlabel(r"normalized cycle $t/N_y$")
    ax.set_ylabel(r"$langle c_{1,\xi}(t)\rangle_\xi$")
    ax.legend(loc="upper right", handletextpad=0.35)
    ax.text(-0.13, 1.035, "(a)", transform=ax.transAxes, ha="left", va="bottom")

    ax = axes[1]
    endpoints = [results[ny]["ceff"][:, -1] for ny in NY_VALUES]
    violin = ax.violinplot(
        endpoints,
        positions=np.arange(len(NY_VALUES)),
        widths=0.72,
        showmeans=False,
        showmedians=True,
        showextrema=False,
        bw_method=0.35,
    )
    for body, ny in zip(violin["bodies"], NY_VALUES):
        body.set_facecolor(STYLES[ny]["color"])
        body.set_edgecolor(STYLES[ny]["color"])
        body.set_alpha(0.24)
    violin["cmedians"].set_color("0.25")
    violin["cmedians"].set_linewidth(0.8)
    for position, ny in enumerate(NY_VALUES):
        result = results[ny]
        mean = result["mean"][-1]
        sem = result["sem"][-1]
        style = STYLES[ny]
        ax.errorbar(
            position,
            mean,
            yerr=sem,
            color=style["color"],
            marker=style["marker"],
            markerfacecolor="white",
            markeredgewidth=0.9,
            capsize=2.0,
            linewidth=1.0,
            zorder=3,
        )
    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xticks(np.arange(len(NY_VALUES)), [str(ny) for ny in NY_VALUES])
    ax.set_xlabel(r"$N_y$")
    ax.set_ylabel(r"endpoint $c_{1,\xi}(2N_y)$")
    ax.text(-0.13, 1.035, "(b)", transform=ax.transAxes, ha="left", va="bottom")

    fig.suptitle(
        r"$N_x=20$, hard wall, $S=100$; bands/error bars: sample-wise SEM",
        fontsize=8.0,
        y=0.985,
    )
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.18, top=0.86, wspace=0.28)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PDF_PATH)
    fig.savefig(PNG_PATH, dpi=300)
    plt.close(fig)


def main() -> None:
    results: dict[int, dict[str, np.ndarray]] = {}
    rows: list[dict[str, float | int | str]] = []
    sources: list[dict[str, str | int]] = []
    npz_payload: dict[str, np.ndarray] = {}
    equality_max = 0.0

    for ny in NY_VALUES:
        source, cycles, ay, entropy = load_entropy(ny)
        ceff = trajectory_ceff(entropy, ay, ny)
        mean = ceff.mean(axis=0)
        standard_deviation = ceff.std(axis=0, ddof=1)
        sem = standard_deviation / np.sqrt(SAMPLES)

        mean_curve_ceff = trajectory_ceff(entropy.mean(axis=0, keepdims=True), ay, ny)[0]
        difference = np.abs(mean - mean_curve_ceff)
        equality_max = max(equality_max, float(difference.max()))
        if float(difference.max()) > 1.0e-11:
            raise RuntimeError("linear estimator-order identity failed")

        results[ny] = {
            "cycles": cycles,
            "ceff": ceff,
            "mean": mean,
            "standard_deviation": standard_deviation,
            "sem": sem,
        }
        prefix = f"Ny{ny:03d}"
        npz_payload[f"cycles_{prefix}"] = cycles
        npz_payload[f"sample_ids_{prefix}"] = np.arange(SAMPLES, dtype=np.int64)
        npz_payload[f"ceff_by_sample_{prefix}"] = ceff
        npz_payload[f"standard_error_of_mean_{prefix}"] = sem

        for index, cycle in enumerate(cycles):
            rows.append(
                {
                    "Nx": NX,
                    "Ny": ny,
                    "samples": SAMPLES,
                    "cycle": int(cycle),
                    "normalized_cycle": float(cycle / ny),
                    "Ay_fit_min": FIT_AY_MIN,
                    "Ay_fit_max": ny // 2,
                    "mean_sample_resolved_ceff": float(mean[index]),
                    "sample_standard_deviation": float(standard_deviation[index]),
                    "standard_error_of_mean": float(sem[index]),
                    "sample_median": float(np.median(ceff[:, index])),
                    "sample_q025": float(np.percentile(ceff[:, index], 2.5)),
                    "sample_q975": float(np.percentile(ceff[:, index], 97.5)),
                    "fit_of_mean_curve_ceff": float(mean_curve_ceff[index]),
                    "estimator_order_abs_difference": float(difference[index]),
                    "source": str(source.relative_to(ROOT)),
                }
            )
        sources.append(
            {
                "path": str(source.relative_to(ROOT)),
                "bytes": source.stat().st_size,
                "sha256": sha256(source),
            }
        )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(NPZ_PATH, **npz_payload)
    make_figure(results)

    outputs = {}
    for label, path in {
        "summary_csv": CSV_PATH,
        "sample_arrays_npz": NPZ_PATH,
        "figure_pdf": PDF_PATH,
        "figure_png": PNG_PATH,
    }.items():
        outputs[label] = {
            "path": str(path.relative_to(ROOT)),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
    manifest = {
        "schema": "sample_resolved_ceff_analysis_v1",
        "scientific_contract": {
            "Nx": NX,
            "Ny": list(NY_VALUES),
            "nshell": NSHELL,
            "samples_per_size": SAMPLES,
            "cycles": "1..2Ny",
            "perfect_correction": True,
            "sequence": "raster_y",
            "dtype": "complex128",
            "hard_wall": True,
            "initial_state": "default random pure state",
        },
        "estimator": {
            "independent_unit": "one complete trajectory",
            "within_trajectory_origin_reduction": "mean over all periodic y0",
            "fit_window": "Ay=8..Ny/2 inclusive",
            "resolved_quantity": "c_{1,xi}(t)=3 times the per-trajectory log-chord slope",
            "reported_mean": "arithmetic mean over 100 trajectory-resolved slopes",
            "uncertainty": "sample-wise standard error SD(c_{1,xi})/sqrt(100)",
            "maximum_estimator_order_abs_difference": equality_max,
            "linearity_note": "the mean point estimate equals the fit of the mean curve for a fixed linear OLS slope",
        },
        "sources": sources,
        "outputs": outputs,
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(f"[saved] {PDF_PATH}")
    print(f"[saved] {PNG_PATH}")
    print(f"[saved] {CSV_PATH}")
    print(f"[saved] {NPZ_PATH}")
    print(f"[check] estimator-order maximum absolute difference = {equality_max:.3e}")
    for ny in NY_VALUES:
        result = results[ny]
        endpoint = result["ceff"][:, -1]
        print(
            f"Ny={ny}: <c_xi(2Ny)>={result['mean'][-1]:.6f} "
            f"+/- {result['sem'][-1]:.6f} SEM, "
            f"sample SD={endpoint.std(ddof=1):.6f}, median={np.median(endpoint):.6f}"
        )


if __name__ == "__main__":
    main()
