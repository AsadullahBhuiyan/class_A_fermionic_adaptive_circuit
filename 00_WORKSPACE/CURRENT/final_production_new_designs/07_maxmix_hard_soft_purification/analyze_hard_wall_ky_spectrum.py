#!/usr/bin/env python3
"""Translation-twirled, sample-averaged hard-wall endpoint spectrum."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import analyze_completed_campaign as campaign

BUNDLE_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = BUNDLE_ROOT / "analysis_outputs" / "hard_wall_ky_spectrum_v1"
NX, NY = 20, 40
ACTIVE_X = tuple(range(5, 16))
ACTIVE_MODES_PER_Y = 2 * len(ACTIVE_X)
EXPECTED_SAMPLES = 100


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def active_indices() -> np.ndarray:
    return np.asarray(
        [mu + 2*x + 2*NX*y for y in range(NY) for x in ACTIVE_X for mu in (0, 1)],
        dtype=np.int64,
    )


def configure_plotting() -> None:
    available = {font.name for font in mpl.font_manager.fontManager.ttflist}
    sans = "CMU Sans Serif" if "CMU Sans Serif" in available else "DejaVu Sans"
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": [sans],
        "mathtext.fontset": "cm", "font.size": 8.0, "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.linewidth": 0.8,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def translation_twirl_kernel(centered_covariance: np.ndarray) -> np.ndarray:
    expected = (NY * ACTIVE_MODES_PER_Y,) * 2
    centered_covariance = np.asarray(centered_covariance, dtype=np.complex128)
    if centered_covariance.shape != expected:
        raise ValueError(f"expected active covariance shape {expected}")
    blocks = centered_covariance.reshape(NY, ACTIVE_MODES_PER_Y, NY, ACTIVE_MODES_PER_Y)
    y = np.arange(NY, dtype=np.int64)
    kernel = np.empty((NY, ACTIVE_MODES_PER_Y, ACTIVE_MODES_PER_Y), dtype=np.complex128)
    for displacement in range(NY):
        kernel[displacement] = blocks[y, :, (y + displacement) % NY, :].mean(axis=0)
    return kernel


def momentum_blocks(kernel: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    momentum_indices = np.arange(-NY // 2, NY // 2, dtype=np.int64)
    ky = 2.0 * np.pi * momentum_indices / NY
    phases = np.exp(1j * np.outer(ky, np.arange(NY, dtype=np.float64)))
    raw = np.einsum("kd,dab->kab", phases, kernel, optimize=True)
    residual = np.max(np.abs(raw - np.swapaxes(raw.conj(), -1, -2)), axis=(1, 2))
    centered = 0.5 * (raw + np.swapaxes(raw.conj(), -1, -2))
    identity = np.eye(ACTIVE_MODES_PER_Y, dtype=np.complex128)[None, :, :]
    occupations = np.linalg.eigvalsh(0.5 * (identity + centered))
    return ky, centered, occupations, residual


def load_sample_averaged_active_covariance(*, verify_hashes: bool) -> tuple[np.ndarray, dict[str, Any]]:
    spec = next(item for item in campaign.CAMPAIGNS if item.construction == "hard")
    inventory = campaign._inventory_campaigns(campaign.DEFAULT_REMOTE_INVENTORY)
    record = inventory.get(spec.key)
    if record is None:
        raise RuntimeError("hard-wall campaign is absent from the remote inventory")
    rows = [row for row in record["records"] if "/Ny040/" in f"/{row['relative_result_path']}"]
    if len(rows) != 20:
        raise RuntimeError(f"expected 20 Ny=40 shards, found {len(rows)}")

    active = active_indices()
    exterior = np.setdiff1d(np.arange(2 * NX * NY, dtype=np.int64), active)
    accumulator = np.zeros((NY * ACTIVE_MODES_PER_Y,) * 2, dtype=np.complex128)
    sample_indices: list[int] = []
    maximum_coupling = 0.0
    maximum_hermiticity = 0.0

    for remote in rows:
        result_path = spec.data_root / remote["relative_result_path"]
        completion_path = spec.data_root / remote["relative_completion_path"]
        for item, bytes_key, hash_key in (
            (result_path, "result_bytes", "result_sha256"),
            (completion_path, "completion_bytes", "completion_sha256"),
        ):
            if not item.is_file() or item.stat().st_size != int(remote[bytes_key]):
                raise RuntimeError(f"missing or wrong-sized campaign file: {item}")
            if verify_hashes and campaign.sha256_file(item) != remote[hash_key]:
                raise RuntimeError(f"checksum mismatch: {item}")
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        campaign._validate_completion(
            completion, spec, result_path,
            expected_result_bytes=int(remote["result_bytes"]),
            expected_result_sha256=remote["result_sha256"],
        )
        if int(completion["Ny"]) != NY:
            raise RuntimeError(f"unexpected Ny in {completion_path.name}")
        with np.load(result_path, allow_pickle=False) as data:
            campaign._validate_product(data, spec=spec, completion=completion, result_path=result_path)
            full = np.asarray(data["G_final"], dtype=np.complex128)
            reduced = np.take(np.take(full, active, axis=1), active, axis=2)
            coupling = np.take(np.take(full, active, axis=1), exterior, axis=2)
            maximum_coupling = max(maximum_coupling, float(np.max(np.abs(coupling), initial=0.0)))
            maximum_hermiticity = max(
                maximum_hermiticity,
                float(np.max(np.abs(reduced - np.swapaxes(reduced.conj(), -1, -2)), initial=0.0)),
            )
            accumulator += reduced.sum(axis=0, dtype=np.complex128)
            sample_indices.extend(int(value) for value in data["sample_indices"])

    if sorted(sample_indices) != list(range(EXPECTED_SAMPLES)):
        raise RuntimeError("Ny=40 sample coverage is not exactly 0,...,99")
    averaged = accumulator / EXPECTED_SAMPLES
    averaged = 0.5 * (averaged + averaged.conj().T)
    diagnostics = {
        "campaign": spec.revision,
        "configuration_hash": spec.configuration_hash,
        "source_hashes": spec.source_hashes,
        "remote_inventory": str(campaign.DEFAULT_REMOTE_INVENTORY),
        "remote_inventory_sha256": campaign.sha256_file(campaign.DEFAULT_REMOTE_INVENTORY),
        "verify_hashes": verify_hashes,
        "verified_result_completion_pairs": len(rows),
        "verified_trajectories": len(sample_indices),
        "maximum_active_exterior_coupling": maximum_coupling,
        "maximum_active_hermiticity_residual": maximum_hermiticity,
    }
    return averaged, diagnostics


def make_figure(ky: np.ndarray, occupations: np.ndarray) -> None:
    configure_plotting()
    figure, axis = plt.subplots(figsize=(3.45, 2.8), constrained_layout=True)
    x = np.r_[ky / np.pi, 1.0]
    closed = np.vstack((occupations, occupations[0]))
    colors = mpl.colormaps["coolwarm"](np.linspace(0.06, 0.94, ACTIVE_MODES_PER_Y))
    for band in range(ACTIVE_MODES_PER_Y):
        axis.plot(x, closed[:, band], color=colors[band], linewidth=0.9)
        axis.scatter(ky / np.pi, occupations[:, band], color=colors[band], s=5.0, linewidths=0.0, zorder=3)
    axis.axhline(0.5, color="black", linestyle=(0, (3, 2)), linewidth=0.75)
    axis.set(xlim=(-1.0, 1.0), ylim=(-0.02, 1.02), xlabel=r"$k_y/\pi$", ylabel=r"occupation $\nu_n(k_y)$")
    axis.set_xticks((-1.0, -0.5, 0.0, 0.5, 1.0))
    axis.text(0.03, 0.04, rf"hard wall, $N_y={NY}$, $S={EXPECTED_SAMPLES}$", transform=axis.transAxes, fontsize=7.2)
    figure.savefig(OUTPUT_ROOT / "hard_wall_Ny40_twirl_annealed_ky_spectrum.pdf")
    figure.savefig(OUTPUT_ROOT / "hard_wall_Ny40_twirl_annealed_ky_spectrum.png", dpi=300)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-hashes", action="store_true")
    args = parser.parse_args()
    averaged_g, provenance = load_sample_averaged_active_covariance(verify_hashes=not args.skip_hashes)
    kernel = translation_twirl_kernel(averaged_g)
    ky, centered_blocks, occupations, block_hermiticity = momentum_blocks(kernel)

    bound = max(0.0, float(-occupations.min(initial=0.0)), float(occupations.max(initial=1.0) - 1.0))
    if bound > 1.0e-9:
        raise RuntimeError(f"annealed momentum occupations leave [0,1] by {bound:.3e}")
    occupations = np.clip(occupations, 0.0, 1.0)
    trace_real = float(np.trace(0.5 * (np.eye(averaged_g.shape[0]) + averaged_g)).real)
    trace_momentum = float(np.sum(occupations))
    trace_error = abs(trace_real - trace_momentum)
    if trace_error > 1.0e-8:
        raise RuntimeError(f"real/momentum trace mismatch: {trace_error:.3e}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT_ROOT / "hard_wall_Ny40_twirl_annealed_ky_spectrum.npz",
        schema=np.asarray("hard_wall_twirl_annealed_ky_spectrum_v1"),
        Nx=np.asarray(NX), Ny=np.asarray(NY), active_x=np.asarray(ACTIVE_X),
        samples=np.asarray(EXPECTED_SAMPLES), ky=ky, ky_over_pi=ky / np.pi,
        centered_covariance_blocks=centered_blocks, occupations=occupations,
        translation_kernel=kernel,
    )
    rows = [
        {"Ny": NY, "samples": EXPECTED_SAMPLES,
         "momentum_index": int(round(ky_value * NY / (2.0 * np.pi))),
         "ky": float(ky_value), "ky_over_pi": float(ky_value / np.pi),
         "band_rank": band, "occupation": float(occupations[position, band])}
        for position, ky_value in enumerate(ky)
        for band in range(ACTIVE_MODES_PER_Y)
    ]
    write_csv(OUTPUT_ROOT / "hard_wall_Ny40_twirl_annealed_ky_spectrum.csv", rows)
    summary = {
        "schema": "hard_wall_twirl_annealed_ky_spectrum_v1",
        "construction": "hard", "Nx": NX, "Ny": NY,
        "active_x": list(ACTIVE_X), "active_modes_per_y": ACTIVE_MODES_PER_Y,
        "samples": EXPECTED_SAMPLES,
        "operation_order": [
            "trace out decoupled pure exterior by taking the active principal block",
            "average the active centered covariance over 100 trajectories",
            "twirl the averaged covariance over all 40 y translations",
            "Fourier transform relative-y blocks and diagonalize C(ky)=(I+G(ky))/2",
        ],
        "interpretation": "annealed spectrum of the translation-twirled sample-mean covariance; not the mean of sample-wise occupation spectra",
        "occupation_min": float(occupations.min()),
        "occupation_max": float(occupations.max()),
        "minimum_distance_to_half": float(np.min(np.abs(occupations - 0.5))),
        "maximum_momentum_block_hermiticity_residual": float(block_hermiticity.max(initial=0.0)),
        "occupation_bound_residual": bound,
        "real_momentum_trace_closure_error": trace_error,
        "provenance": provenance,
    }
    (OUTPUT_ROOT / "analysis_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    make_figure(ky, occupations)
    print(json.dumps({"output_root": str(OUTPUT_ROOT), **summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
