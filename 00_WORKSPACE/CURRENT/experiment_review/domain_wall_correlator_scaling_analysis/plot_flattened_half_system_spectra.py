#!/usr/bin/env python3
"""Half-y occupation spectra of the hard-wall OW flattened-parent ground state.

Uses the canonical CPU OW constructor and the existing regulated, half-filled
momentum-projector benchmark. No stochastic dynamics or sample averaging.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
import json
from pathlib import Path
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[4]
CANONICAL = ROOT / "src/fgtn/classA_U1FGTN.py"
sys.path.insert(0, str(CANONICAL.parent))
from classA_U1FGTN import classA_U1FGTN

import plot_hard_wall_flattened_benchmark as reference
from run_flattened_ground_state_large_ny import restricted_projector
from plot_typical_hard_wall_x_slices import configure_matplotlib

OUTPUT = Path(__file__).resolve().parent / "outputs/flattened_half_system_nx80_ny80"
MARGIN = 1.0e-12


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Unsupported JSON type: {type(value)}")


def build_model(nx: int, ny: int, alpha: float) -> classA_U1FGTN:
    if Path(inspect.getfile(classA_U1FGTN)).resolve() != CANONICAL.resolve():
        raise RuntimeError("The canonical CPU source must be imported")
    model = classA_U1FGTN(
        Nx=nx, Ny=ny, DW=True, nshell=1, filling_frac=0.5,
        alpha_1=alpha, alpha_2=30.0, trial_orbitals="X", dw_truncation=True,
    )
    model.construct_OW_projectors(
        nshell=1, DW=True, trial_orbitals="X", dw_truncation=True,
    )
    # These intermediate Bloch projectors are no longer needed; retain the OWs.
    del model.Pplus, model.Pminus
    return model


def resolved_spectrum(raw: np.ndarray, margin: float = MARGIN):
    raw = np.asarray(raw, dtype=np.float64)
    if not np.all(np.isfinite(raw)) or raw.min() < -margin or raw.max() > 1 + margin:
        raise FloatingPointError("Occupation eigenvalues violate the numerical bounds")
    mask = (raw > margin) & (raw < 1.0 - margin)
    occupations = raw[mask]
    energies = np.log1p(-occupations) - np.log(occupations)
    return mask, occupations, energies


def normalized_histogram(values: np.ndarray, edges: np.ndarray):
    counts, _ = np.histogram(values, bins=edges)
    if counts.sum() != values.size or values.size == 0:
        raise ValueError("All retained eigenvalues must lie within the histogram")
    density = counts / (values.size * np.diff(edges))
    np.testing.assert_allclose(np.sum(density * np.diff(edges)), 1.0, atol=1e-14)
    return counts, density


def compute(nx: int, ny: int, alpha: float, output: Path):
    start = time.monotonic()
    print(f"[alpha1={alpha:g}] constructing canonical OW modes for {nx} x {ny}", flush=True)
    model = build_model(nx, ny, alpha)
    walls = list(model.DW_loc)
    print(f"[alpha1={alpha:g}] diagonalizing the y-momentum parent blocks", flush=True)
    delta, diagnostics = reference.regulated_flattened_momentum_projector(model)
    del model
    gc.collect()
    print(f"[alpha1={alpha:g}] diagonalizing {nx * ny} half-system occupations", flush=True)
    covariance = restricted_projector(delta, range(ny // 2))
    hermiticity = float(np.max(np.abs(covariance - covariance.conj().T)))
    trace = float(np.trace(covariance).real)
    occupations = eigh(
        covariance, eigvals_only=True, overwrite_a=True,
        check_finite=False, driver="evd",
    )
    del covariance, delta
    mask, retained, energies = resolved_spectrum(occupations)
    diagnostics.update({
        "alpha_1": alpha, "Nx": nx, "Ny": ny, "wall_locations": walls,
        "half_system_modes": int(occupations.size), "subsystem_charge": trace,
        "subsystem_hermiticity_max_abs": hermiticity,
        "occupation_min_raw": float(occupations.min()),
        "occupation_max_raw": float(occupations.max()),
        "occupation_sum_trace_residual": float(abs(occupations.sum() - trace)),
        "roundoff_margin": MARGIN, "retained_modes": int(mask.sum()),
        "excluded_modes": int((~mask).sum()),
        "excluded_near_zero": int((occupations <= MARGIN).sum()),
        "excluded_near_one": int((occupations >= 1.0 - MARGIN).sum()),
        "retained_modular_min": float(energies.min()),
        "retained_modular_max": float(energies.max()),
        "elapsed_seconds": time.monotonic() - start,
    })
    if hermiticity > 1e-12 or diagnostics["occupation_sum_trace_residual"] > 1e-8:
        raise FloatingPointError("Half-system projector checks failed")
    np.savez_compressed(
        output / f"alpha1_{alpha:g}_spectrum.npz", alpha_1=alpha, Nx=nx, Ny=ny,
        occupation_eigenvalues_raw=occupations, resolved_mask=mask,
        occupation_eigenvalues_resolved=retained, modular_energies_resolved=energies,
        subsystem_y_values=np.arange(ny // 2), subsystem_x_values=np.arange(nx),
        wall_locations=walls, occupation_twist=reference.OCCUPATION_TWIST,
        roundoff_margin=MARGIN,
    )
    (output / f"alpha1_{alpha:g}_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, default=json_default) + "\n", encoding="utf-8"
    )
    print(f"[alpha1={alpha:g}] {mask.sum()}/{occupations.size} resolved modes; "
          f"{diagnostics['elapsed_seconds']:.1f} seconds", flush=True)
    return retained, energies, diagnostics


def plot(records, output: Path, nx: int, ny: int):
    configure_matplotlib()
    plt.rcParams.update({"legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8})
    occupation_edges = np.linspace(0, 1, 51)
    modular_edges = np.linspace(-28, 28, 57)
    figure, axes = plt.subplots(1, 2, figsize=(7.05, 3.0), layout="constrained")
    histograms = {"occupation_bin_edges": occupation_edges, "modular_bin_edges": modular_edges}
    for (occupation, energy, diagnostics), color, style in zip(
        records, ("#0072B2", "#D55E00"), ("-", "--")
    ):
        alpha = diagnostics["alpha_1"]
        for ax, name, values, edges in zip(
            axes, ("occupation", "modular"), (occupation, energy),
            (occupation_edges, modular_edges),
        ):
            counts, density = normalized_histogram(values, edges)
            histograms[f"alpha1_{alpha:g}_{name}_counts"] = counts
            histograms[f"alpha1_{alpha:g}_{name}_density"] = density
            # Empty bins have zero density, not a fabricated positive floor.
            ax.stairs(np.where(counts > 0, density, np.nan), edges,
                      color=color, linestyle=style, linewidth=1.25,
                      label=rf"$\alpha_1={alpha:g}$")
    for i, ax in enumerate(axes):
        ax.set_yscale("log")
        ax.set_ylabel("Normalized density")
        ax.legend(loc="upper center", ncol=2, frameon=False)
        ax.text(-0.16, 1.03, f"({chr(97 + i)})", transform=ax.transAxes, fontsize=10)
    axes[0].set(xlabel=r"Occupation $\nu$", xlim=(0, 1), title="Half-system occupation spectrum")
    axes[1].set(xlabel=r"Entanglement energy $\epsilon=\log[(1-\nu)/\nu]$", xlim=(-28, 28),
                title="Half-system entanglement spectrum")
    axes[0].set_xticks((0, 0.25, 0.5, 0.75, 1))
    axes[1].set_xticks((-20, 0, 20))
    figure.suptitle(rf"${nx}\times{ny}$, hard walls, $n_{{\rm shell}}=1$, $A_y=N_y/2$", fontsize=10)
    for suffix in ("png", "pdf"):
        figure.savefig(output / f"half_system_spectra_log_density.{suffix}", dpi=300)
    plt.close(figure)
    np.savez_compressed(output / "histograms.npz", **histograms)
    # Standalone version of the occupation comparison explicitly requested.
    figure, ax = plt.subplots(figsize=(3.375, 3.0), layout="constrained")
    for (_, _, diagnostics), color, style in zip(
        records, ("#0072B2", "#D55E00"), ("-", "--")
    ):
        alpha = diagnostics["alpha_1"]
        counts = histograms[f"alpha1_{alpha:g}_occupation_counts"]
        density = histograms[f"alpha1_{alpha:g}_occupation_density"]
        ax.stairs(np.where(counts > 0, density, np.nan), occupation_edges,
                  color=color, linestyle=style, linewidth=1.25,
                  label=rf"$\alpha_1={alpha:g}$")
    ax.set(yscale="log", xlim=(0, 1), xlabel=r"Occupation $\nu$",
           ylabel="Normalized density",
           title=rf"${nx}\times{ny}$, hard walls, $n_{{\rm shell}}=1$")
    ax.set_xticks((0, 0.25, 0.5, 0.75, 1))
    ax.legend(loc="upper center", frameon=False)
    for suffix in ("png", "pdf"):
        figure.savefig(output / f"half_system_occupation_log_density.{suffix}", dpi=300)
    plt.close(figure)

    # Histogram transformed eigenvalues, not relabeled occupation-density bins.
    figure, ax = plt.subplots(figsize=(3.375, 3.0), layout="constrained")
    peak = 0.0
    for (_, energies, diagnostics), color, style in zip(
        records, ("#0072B2", "#D55E00"), ("-", "--")
    ):
        alpha = diagnostics["alpha_1"]
        counts, density = normalized_histogram(energies, modular_edges)
        peak = max(peak, float(density.max()))
        ax.stairs(np.where(counts > 0, density, np.nan), modular_edges,
                  color=color, linestyle=style, linewidth=1.25,
                  label=rf"$\alpha_1={alpha:g}$")
    ax.set(yscale="log", xlim=(-28, 28),
           xlabel=r"Entanglement energy $\epsilon$",
           ylabel="Normalized density",
           title=rf"${nx}\times{ny}$, hard walls, $n_{{\rm shell}}=1$")
    ax.set_ylim(top=peak * 2.5)
    ax.set_xticks((-20, -10, 0, 10, 20))
    ax.legend(loc="upper center", frameon=False, ncol=2)
    for suffix in ("png", "pdf"):
        figure.savefig(output / f"half_system_entanglement_energy_log_density.{suffix}", dpi=300)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nx", type=int, default=80)
    parser.add_argument("--ny", type=int, default=80)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--replot", action="store_true")
    args = parser.parse_args()
    if args.nx % 4 or args.ny % 2 or min(args.nx, args.ny) < 4:
        parser.error("Require Nx divisible by 4 and even Ny, both at least 4")
    args.output.mkdir(parents=True, exist_ok=True)
    sources = (Path(__file__), CANONICAL, Path(reference.__file__))
    with threadpool_limits(limits=args.threads):
        if args.replot:
            records = []
            for alpha in (1.0, 3.0):
                with np.load(args.output / f"alpha1_{alpha:g}_spectrum.npz") as data:
                    if int(data["Nx"]) != args.nx or int(data["Ny"]) != args.ny:
                        raise ValueError("Saved spectrum geometry does not match request")
                    _, occupation, energy = resolved_spectrum(data["occupation_eigenvalues_raw"])
                diagnostics = json.loads((args.output / f"alpha1_{alpha:g}_diagnostics.json").read_text())
                records.append((occupation, energy, diagnostics))
        else:
            records = [compute(args.nx, args.ny, alpha, args.output) for alpha in (1.0, 3.0)]
        plot(records, args.output, args.nx, args.ny)
    summary = {
        "kind": "deterministic_half_filled_OW_flattened_parent_ground_state",
        "Nx": args.nx, "Ny": args.ny, "alpha_1_values": [1, 3], "alpha_2": 30,
        "nshell": 1, "DW": True, "dw_truncation": True, "trial_orbitals": "X",
        "subsystem": "[0,Nx) x [0,Ny//2), both physical orbitals",
        "dtype": "complex128", "occupation_twist": reference.OCCUPATION_TWIST,
        "histogram_normalization": "unit integral separately for each retained spectrum",
        "roundoff_margin": MARGIN, "cases": [r[2] for r in records],
        "sources_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
    }
    if not args.replot:
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2, default=json_default) + "\n")
    print(f"[done] {args.output}", flush=True)


if __name__ == "__main__":
    main()
