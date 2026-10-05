#!/usr/bin/env python3
"""Pool sample-wise Ny=40 hard-wall endpoint occupations into a histogram."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import analyze_completed_campaign as campaign


BUNDLE_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = BUNDLE_ROOT / "analysis_outputs" / "hard_wall_Ny40_pooled_occupations_v1"
NX, NY = 20, 40
ACTIVE_X = tuple(range(5, 16))
ACTIVE_MODES = 22 * NY
EXPECTED_SAMPLES = 100
BIN_COUNT = 100


def active_indices() -> np.ndarray:
    return np.asarray(
        [mu + 2 * x + 2 * NX * y for y in range(NY) for x in ACTIVE_X for mu in (0, 1)],
        dtype=np.int64,
    )


def configure_plotting() -> None:
    available = {font.name for font in mpl.font_manager.fontManager.ttflist}
    sans = "CMU Sans Serif" if "CMU Sans Serif" in available else "DejaVu Sans"
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [sans],
            "mathtext.fontset": "cm",
            "font.size": 8.0,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "axes.linewidth": 0.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_pooled_occupations(*, verify_hashes: bool) -> tuple[np.ndarray, dict[str, object]]:
    spec = next(item for item in campaign.CAMPAIGNS if item.construction == "hard")
    inventory = campaign._inventory_campaigns(campaign.DEFAULT_REMOTE_INVENTORY)
    record = inventory.get(spec.key)
    if record is None:
        raise RuntimeError("hard-wall campaign is absent from the verified inventory")
    rows = [
        row
        for row in record["records"]
        if "/Ny040/" in f"/{row['relative_result_path']}"
    ]
    if len(rows) != 20:
        raise RuntimeError(f"expected 20 Ny=40 shards, found {len(rows)}")

    active = active_indices()
    sample_indices: list[int] = []
    spectra: list[np.ndarray] = []
    maximum_bound_residual = 0.0
    maximum_active_exterior_coupling = 0.0
    exterior = np.setdiff1d(np.arange(2 * NX * NY, dtype=np.int64), active)

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
            completion,
            spec,
            result_path,
            expected_result_bytes=int(remote["result_bytes"]),
            expected_result_sha256=remote["result_sha256"],
        )
        with np.load(result_path, allow_pickle=False) as data:
            campaign._validate_product(
                data, spec=spec, completion=completion, result_path=result_path
            )
            full = np.asarray(data["G_final"], dtype=np.complex128)
            reduced = np.take(np.take(full, active, axis=1), active, axis=2)
            coupling = np.take(np.take(full, active, axis=1), exterior, axis=2)
            maximum_active_exterior_coupling = max(
                maximum_active_exterior_coupling,
                float(np.max(np.abs(coupling), initial=0.0)),
            )
            identity = np.eye(ACTIVE_MODES, dtype=np.complex128)[None, :, :]
            occupations = np.linalg.eigvalsh(0.5 * (identity + reduced))
            bound = max(
                0.0,
                float(-occupations.min(initial=0.0)),
                float(occupations.max(initial=1.0) - 1.0),
            )
            maximum_bound_residual = max(maximum_bound_residual, bound)
            if bound > 1.0e-9:
                raise RuntimeError(f"{result_path.name}: occupation bound residual {bound:.3e}")
            spectra.append(np.clip(occupations, 0.0, 1.0))
            sample_indices.extend(int(value) for value in data["sample_indices"])

    if sorted(sample_indices) != list(range(EXPECTED_SAMPLES)):
        raise RuntimeError("Ny=40 sample coverage is not exactly 0,...,99")
    pooled = np.concatenate(spectra, axis=0).reshape(-1)
    if pooled.size != EXPECTED_SAMPLES * ACTIVE_MODES:
        raise RuntimeError("pooled occupation count mismatch")
    provenance = {
        "campaign": spec.revision,
        "configuration_hash": spec.configuration_hash,
        "source_hashes": spec.source_hashes,
        "remote_inventory": str(campaign.DEFAULT_REMOTE_INVENTORY),
        "remote_inventory_sha256": campaign.sha256_file(campaign.DEFAULT_REMOTE_INVENTORY),
        "verify_hashes": verify_hashes,
        "verified_result_completion_pairs": len(rows),
        "verified_trajectories": len(sample_indices),
        "maximum_occupation_bound_residual_before_clipping": maximum_bound_residual,
        "maximum_active_exterior_coupling": maximum_active_exterior_coupling,
    }
    return pooled, provenance


def make_figure(edges: np.ndarray, density: np.ndarray) -> None:
    configure_plotting()
    figure, axis = plt.subplots(figsize=(3.45, 2.8), constrained_layout=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    axis.bar(
        centers,
        density,
        width=np.diff(edges),
        align="center",
        color="#2878b5",
        edgecolor="white",
        linewidth=0.25,
    )
    axis.set_yscale("log")
    axis.set_xlim(0.0, 1.0)
    axis.set_xlabel(r"endpoint occupation $\nu$")
    axis.set_ylabel("normalized density")
    axis.text(
        0.5,
        0.94,
        rf"hard wall, $N_y={NY}$, $S={EXPECTED_SAMPLES}$",
        transform=axis.transAxes,
        ha="center",
        va="top",
        fontsize=7.2,
    )
    figure.savefig(OUTPUT_ROOT / "hard_wall_Ny40_pooled_occupation_histogram.pdf")
    figure.savefig(OUTPUT_ROOT / "hard_wall_Ny40_pooled_occupation_histogram.png", dpi=300)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-hashes", action="store_true")
    args = parser.parse_args()
    pooled, provenance = load_pooled_occupations(verify_hashes=not args.skip_hashes)
    counts, edges = np.histogram(pooled, bins=BIN_COUNT, range=(0.0, 1.0))
    mass = counts / counts.sum()
    density = mass / np.diff(edges)
    if not np.isclose(np.sum(density * np.diff(edges)), 1.0, rtol=0.0, atol=1.0e-14):
        raise RuntimeError("histogram normalization failure")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT_ROOT / "hard_wall_Ny40_pooled_occupation_histogram.npz",
        schema=np.asarray("hard_wall_Ny40_pooled_occupation_histogram_v1"),
        pooled_occupations=pooled,
        bin_edges=edges,
        counts=counts,
        probability_mass=mass,
        probability_density=density,
    )
    rows = [
        {
            "bin_index": index,
            "left_edge": float(edges[index]),
            "right_edge": float(edges[index + 1]),
            "center": float(0.5 * (edges[index] + edges[index + 1])),
            "count": int(counts[index]),
            "probability_mass": float(mass[index]),
            "probability_density": float(density[index]),
        }
        for index in range(BIN_COUNT)
    ]
    with (OUTPUT_ROOT / "hard_wall_Ny40_pooled_occupation_histogram.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "schema": "hard_wall_Ny40_pooled_occupation_histogram_v1",
        "construction": "hard",
        "Nx": NX,
        "Ny": NY,
        "active_x": list(ACTIVE_X),
        "samples": EXPECTED_SAMPLES,
        "occupations_per_sample": ACTIVE_MODES,
        "pooled_occupation_count": int(pooled.size),
        "histogram_bins": BIN_COUNT,
        "histogram_range": [0.0, 1.0],
        "histogram_normalization": "unit integral; density_i=count_i/(total_count*bin_width)",
        "sample_averaging": "none; sample-wise active-slab spectra pooled after diagonalization",
        "mean_occupation": float(pooled.mean()),
        "minimum_distance_to_half": float(np.min(np.abs(pooled - 0.5))),
        "fraction_below_0p01": float(np.mean(pooled < 0.01)),
        "fraction_above_0p99": float(np.mean(pooled > 0.99)),
        "provenance": provenance,
    }
    (OUTPUT_ROOT / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    make_figure(edges, density)
    print(json.dumps({"output_root": str(OUTPUT_ROOT), **summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
