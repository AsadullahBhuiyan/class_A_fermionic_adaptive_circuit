#!/usr/bin/env python3
"""Build the hard-wall OW-flattened correlator benchmark at two shell depths."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh

from plot_typical_hard_wall_x_slices import OUTPUT_DIR, REPO_ROOT, configure_matplotlib


REFERENCE_ROOT = (
    REPO_ROOT
    / "00_WORKSPACE/CURRENT/final_production_new_designs"
    / "06_domain_wall_flattened_ground_state_reference"
)
for source_path in (REFERENCE_ROOT / "src", REFERENCE_ROOT):
    if str(source_path) not in sys.path:
        sys.path.insert(0, str(source_path))

from run_flattened_ground_state_large_ny import (  # noqa: E402
    Case,
    _build_model,
)


STEM = "hard_wall_flattened_benchmark_nshell1_vs_dense"
NX = 20
NY = 32
ALPHA_1 = 1.0
ALPHA_2 = 30.0
OCCUPATION_TWIST = 1.0e-7
X_SLICES = (5, 6, 7, 9, 11, 13, 14, 15)
X_LABELS = {
    5: r"$x_L=5$",
    6: r"$x_L+1=6$",
    7: r"$x=7$",
    9: r"$x=9$",
    11: r"$x=11$",
    13: r"$x=13$",
    14: r"$x_R-1=14$",
    15: r"$x_R=15$",
}
MARKERS = ("o", "s", "^", "D", "v", "P", "X", "h")
LINESTYLES = ("-", "--", ":", "-.", "-.", ":", "--", "-")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def regulated_flattened_momentum_projector(model: object) -> tuple[np.ndarray, dict[str, object]]:
    """Fill the OW flattened parent on the B0-regulated momentum grid.

    The tiny uniform shift selects one side of the finite-volume wall-mode crossing.
    The correlation transform retains the periodic untwisted Bloch basis, matching
    the B0 occupation-twist convention.
    """

    ny = int(model.Ny)
    nx = int(model.Nx)
    block_dimension = 2 * nx
    base_momenta = 2.0 * np.pi * np.arange(ny, dtype=np.float64) / float(ny)
    occupied_momenta = base_momenta + OCCUPATION_TWIST / float(ny)
    y_values = np.arange(ny, dtype=np.float64)
    fourier = np.exp(-1j * occupied_momenta[:, None] * y_values[None, :]) / np.sqrt(
        float(ny)
    )

    hamiltonian_k = np.zeros(
        (ny, block_dimension, block_dimension), dtype=np.complex128
    )
    for name, sign in (
        ("WF_Ap", 1.0),
        ("WF_Bp", 1.0),
        ("WF_Am", -1.0),
        ("WF_Bm", -1.0),
    ):
        frame = np.asarray(getattr(model, name), dtype=np.complex128).reshape(
            ny, block_dimension, nx, ny
        )
        representative = np.einsum(
            "ky,yar->kar", fourier, frame[:, :, :, 0], optimize=True
        )
        hamiltonian_k += sign * float(ny) * np.einsum(
            "kar,kbr->kab", representative, representative.conj(), optimize=True
        )
    hermiticity = float(
        np.max(np.abs(hamiltonian_k - hamiltonian_k.conj().transpose(0, 2, 1)))
    )
    hamiltonian_k = 0.5 * (
        hamiltonian_k + hamiltonian_k.conj().transpose(0, 2, 1)
    )

    energies = np.empty((ny, block_dimension), dtype=np.float64)
    eigenvectors = np.empty(
        (ny, block_dimension, block_dimension), dtype=np.complex128
    )
    for momentum in range(ny):
        energies[momentum], eigenvectors[momentum] = eigh(
            hamiltonian_k[momentum], check_finite=False, driver="evd"
        )
    total_rank = ny * block_dimension // 2
    order = np.argsort(energies.ravel(), kind="stable")
    occupied = np.zeros(ny * block_dimension, dtype=bool)
    occupied[order[:total_rank]] = True
    occupied = occupied.reshape(ny, block_dimension)
    rank_by_momentum = occupied.sum(axis=1)
    if not np.array_equal(rank_by_momentum, np.full(ny, nx, dtype=np.int64)):
        raise RuntimeError(
            "Regulated hard-wall parent does not have Nx occupied modes at each momentum"
        )

    projector_k = np.empty_like(hamiltonian_k)
    for momentum in range(ny):
        vectors = eigenvectors[momentum][:, occupied[momentum]]
        projector_k[momentum] = vectors @ vectors.conj().T
    projector_delta = np.fft.ifft(projector_k, axis=0)
    sorted_energies = energies.ravel()[order]
    diagnostics: dict[str, object] = {
        "occupation_twist": OCCUPATION_TWIST,
        "half_filling_rank": total_rank,
        "half_filling_gap": float(
            sorted_energies[total_rank] - sorted_energies[total_rank - 1]
        ),
        "minimum_absolute_energy": float(np.min(np.abs(energies))),
        "occupied_rank_by_momentum": rank_by_momentum,
        "hamiltonian_block_hermiticity_max_abs": hermiticity,
        "projector_idempotency_max_abs": float(
            max(np.max(np.abs(block @ block - block)) for block in projector_k)
        ),
        "projector_block_hermiticity_max_abs": float(
            max(
                np.max(
                    np.abs(
                        projector_delta[delta]
                        - projector_delta[-delta % ny].conj().T
                    )
                )
                for delta in range(ny)
            )
        ),
    }
    return projector_delta, diagnostics


def benchmark(nshell: int | None, label: str, shell_index: int) -> dict[str, object]:
    case = Case(
        wall_index=0,
        nshell_index=shell_index,
        ny_index=0,
        alpha_index=20,
        wall="hard",
        nshell_label=label,
        nshell=nshell,
        ny=NY,
        alpha_1=ALPHA_1,
    )
    model = _build_model(case)
    if tuple(int(value) for value in model.DW_loc) != (5, 15):
        raise RuntimeError(f"Unexpected hard-wall locations: {model.DW_loc}")
    if not bool(model.dw_truncation):
        raise RuntimeError("Flattened benchmark did not enable hard DW truncation")
    projector_delta, diagnostics = regulated_flattened_momentum_projector(model)
    del model

    independent = np.empty((NX, NY // 2 + 1), dtype=np.float64)
    for x in range(NX):
        orbital_slice = slice(2 * x, 2 * x + 2)
        for separation in range(NY // 2 + 1):
            block = projector_delta[-separation % NY][orbital_slice, orbital_slice]
            independent[x, separation] = 0.5 * np.square(np.abs(block)).sum()

    full = np.concatenate((independent[:, 1:], independent[:, -2::-1]), axis=1)
    if full.shape != (NX, NY):
        raise RuntimeError(f"Unexpected reconstructed correlator shape: {full.shape}")
    reflected = full[:, NY // 2 : NY - 1][:, ::-1]
    if not np.array_equal(full[:, : NY // 2 - 1], reflected):
        raise RuntimeError("Periodic-reflection reconstruction failed")

    fit_r = np.arange(2, 9, dtype=np.float64)
    fit_chord = (NY / np.pi) * np.sin(np.pi * fit_r / NY)
    wall_beta = {}
    for x in (5, 15):
        slope, _ = np.polyfit(np.log(fit_chord), np.log(independent[x, 2:9]), 1)
        wall_beta[str(x)] = float(-slope)
    return {
        "nshell": nshell,
        "label": label,
        "independent": independent,
        "full": full,
        "diagnostics": diagnostics,
        "wall_beta_log_chord_r2_8": wall_beta,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    records = [benchmark(1, "1", 0), benchmark(None, "inf", 2)]
    separations = np.arange(1, NY + 1, dtype=np.int64)

    configure_matplotlib()
    figure, axes = plt.subplots(1, 2, figsize=(6.9, 3.5), sharex=True, sharey=True)
    colors = plt.get_cmap("viridis")(np.linspace(0.04, 0.96, len(X_SLICES)))
    handles = []
    for panel, (axis, record) in enumerate(zip(axes, records)):
        full = np.asarray(record["full"], dtype=np.float64)
        for index, x in enumerate(X_SLICES):
            is_wall = x in (5, 15)
            (line,) = axis.plot(
                separations,
                full[x],
                color=colors[index],
                marker=MARKERS[index],
                linestyle=LINESTYLES[index],
                linewidth=1.35 if is_wall else 0.95,
                markersize=3.4 if is_wall else 2.7,
                markerfacecolor=colors[index] if is_wall else "white",
                markeredgecolor=colors[index],
                markeredgewidth=0.7,
                label=X_LABELS[x],
            )
            if panel == 0:
                handles.append(line)
        axis.axvspan(16, 32.5, color="#777777", alpha=0.055, linewidth=0, zorder=-10)
        axis.axvline(16, color="#777777", linestyle=(0, (2, 2)), linewidth=0.7, zorder=-9)
        axis.set_yscale("log")
        axis.set_xlim(0.5, 32.5)
        axis.set_ylim(1e-19, 5e-1)
        axis.set_xticks([1, 8, 16, 24, 32])
        axis.set_xlabel(r"separation $r_y$")
        shell_title = r"1" if record["label"] == "1" else r"\infty"
        axis.set_title(rf"({chr(97 + panel)}) $n_{{\rm shell}}={shell_title}$", loc="left")
        axis.text(
            0.55,
            0.05,
            r"periodic reflection",
            transform=axis.transAxes,
            fontsize=6.2,
            color="#555555",
        )
    axes[0].set_ylabel(r"squared correlator $G_x(r_y)$")
    figure.legend(
        handles=handles,
        labels=[X_LABELS[x] for x in X_SLICES],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        frameon=False,
        ncol=4,
        columnspacing=0.8,
        handlelength=2.0,
        handletextpad=0.35,
        labelspacing=0.25,
    )
    figure.text(
        0.5,
        0.83,
        r"hard/support-truncated OW parent: $20\times32$, $\alpha_1=1$, $\alpha_2=30$, $\phi=10^{-7}$",
        ha="center",
        va="bottom",
        fontsize=7.0,
        color="#333333",
    )
    figure.subplots_adjust(left=0.095, right=0.99, bottom=0.14, top=0.73, wspace=0.08)
    metadata = {
        "Title": "Hard-wall OW-flattened ground-state correlator benchmark",
        "Author": "class_A_fermionic_adaptive_circuit analysis",
        "Subject": "Nx=20, Ny=32, alpha1=1, alpha2=30; nshell=1 versus dense",
    }
    figure.savefig(OUTPUT_DIR / f"{STEM}.pdf", metadata=metadata)
    figure.savefig(OUTPUT_DIR / f"{STEM}.png", dpi=300, metadata=metadata)
    plt.close(figure)

    with (OUTPUT_DIR / f"{STEM}_source_data.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["nshell", "x", "position", "r_y", "squared_correlator", "data_origin"],
        )
        writer.writeheader()
        for record in records:
            full = np.asarray(record["full"], dtype=np.float64)
            for x in X_SLICES:
                for index, separation in enumerate(separations):
                    origin = (
                        "direct independent separation"
                        if separation <= NY // 2
                        else "periodic Hermitian reflection"
                        if separation < NY
                        else "r_y=0 periodic contact recurrence"
                    )
                    writer.writerow(
                        {
                            "nshell": record["label"],
                            "x": x,
                            "position": X_LABELS[x].replace("$", ""),
                            "r_y": int(separation),
                            "squared_correlator": f"{full[x, index]:.17g}",
                            "data_origin": origin,
                        }
                    )

    np.savez_compressed(
        OUTPUT_DIR / f"{STEM}.npz",
        schema=np.asarray("hard_wall_ow_flattened_correlator_benchmark_v1"),
        Nx=np.asarray(NX, dtype=np.int64),
        Ny=np.asarray(NY, dtype=np.int64),
        alpha_1=np.asarray(ALPHA_1),
        alpha_2=np.asarray(ALPHA_2),
        occupation_twist=np.asarray(OCCUPATION_TWIST),
        wall=np.asarray("hard/support-truncated"),
        dw_location=np.asarray([5, 15], dtype=np.int64),
        nshell_labels=np.asarray(["1", "inf"]),
        x_values=np.arange(NX, dtype=np.int64),
        ry_independent=np.arange(NY // 2 + 1, dtype=np.int64),
        ry_full=separations,
        correlator_independent=np.stack([record["independent"] for record in records]),
        correlator_full=np.stack([record["full"] for record in records]),
    )

    source_files = (
        Path(__file__).resolve(),
        REFERENCE_ROOT / "run_flattened_ground_state_large_ny.py",
        REFERENCE_ROOT / "src/classA_U1FGTN.py",
        REFERENCE_ROOT / "src/occupied_frame.py",
    )
    summary = {
        "schema": "hard_wall_ow_flattened_correlator_benchmark_summary_v1",
        "contract": {
            "Nx": NX,
            "Ny": NY,
            "alpha_1": ALPHA_1,
            "alpha_2": ALPHA_2,
            "wall": "hard/support-truncated",
            "DW": True,
            "dw_truncation": True,
            "nshell_values": [1, None],
            "trial_orbital": "X",
            "dtype": "complex128",
            "occupation_twist": OCCUPATION_TWIST,
            "occupation_twist_role": "select one side of the finite-volume wall-mode crossing",
            "state": "exact half-filled ground state of the OW flattened parent",
            "flattened_parent": "sum_R(P_A+ + P_B+ - P_A- - P_B-)",
            "dw_location": [5, 15],
        },
        "diagnostics": {
            str(record["label"]): {
                "half_filling_gap": float(record["diagnostics"]["half_filling_gap"]),
                "minimum_absolute_energy": float(record["diagnostics"]["minimum_absolute_energy"]),
                "wall_beta_log_chord_r2_8": record["wall_beta_log_chord_r2_8"],
                "occupation_twist": float(record["diagnostics"]["occupation_twist"]),
                "hamiltonian_block_hermiticity_max_abs": float(
                    record["diagnostics"]["hamiltonian_block_hermiticity_max_abs"]
                ),
                "projector_idempotency_max_abs": float(
                    record["diagnostics"]["projector_idempotency_max_abs"]
                ),
            }
            for record in records
        },
        "source_hashes": {
            str(path.relative_to(REPO_ROOT)): sha256(path) for path in source_files
        },
        "separation_display": {
            "range": [1, 32],
            "independent_range": [0, 16],
            "reflected_range": [17, 31],
            "r_y_32": "periodic recurrence of r_y=0 contact term",
        },
        "outputs": [
            f"{STEM}.pdf",
            f"{STEM}.png",
            f"{STEM}.npz",
            f"{STEM}_source_data.csv",
        ],
    }
    (OUTPUT_DIR / f"{STEM}_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary["diagnostics"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
