#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from mean_channel_lindblad_cpu import validate_selected_observable_schema


HERE = Path(__file__).resolve().parent


def load_cases(run_root: Path) -> list[dict[str, Any]]:
    manifest = json.loads((run_root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise RuntimeError("campaign manifest is not complete")
    rows = []
    for receipt in manifest["cases"]:
        metadata_path = run_root / "cases" / receipt["metadata"]
        arrays_path = run_root / "cases" / receipt["observables"]
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        with np.load(arrays_path, allow_pickle=False) as payload:
            arrays = {key: payload[key] for key in payload.files}
        if metadata["campaign"] != "L1_DEPHASING_CONTROL":
            validate_selected_observable_schema(metadata["model"]["nshell"], arrays)
        rows.append({"metadata": metadata, "arrays": arrays})
    return rows


def configure_plotting() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.family": "sans-serif",
            "font.sans-serif": ["CMU Sans Serif", "DejaVu Sans"],
            "mathtext.fontset": "cm",
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 6.5,
            "xtick.direction": "in",
            "ytick.direction": "in",
        }
    )


def save(fig: Any, output: Path, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(output / f"{stem}.pdf")
    fig.savefig(output / f"{stem}.png", dpi=300)
    plt.close(fig)


def analyze(run_root: Path, output: Path) -> dict[str, Any]:
    configure_plotting()
    output.mkdir(parents=True, exist_ok=True)
    cases = load_cases(run_root)
    scalar_rows = []
    for row in cases:
        metadata = row["metadata"]
        model = metadata["model"]
        diagnostics = metadata.get("diagnostics", {})
        arrays = row["arrays"]
        p = arrays.get("finite_channel_p", np.asarray([]))
        error = arrays.get("finite_channel_relative_error_to_continuous", np.asarray([]))
        convergence_order = (
            float(np.polyfit(np.log(p), np.log(error), 1)[0])
            if p.size >= 2 and np.all(error > 0)
            else None
        )
        scalar_rows.append(
            {
                "case_id": metadata["case_id"],
                "campaign": metadata["campaign"],
                "Nx": model["Nx"],
                "Ny": model["Ny"],
                "alpha_top": model["alpha_top"],
                "alpha_triv": model["alpha_triv"],
                "nshell": "None" if model["nshell"] is None else model["nshell"],
                "n_a": model["n_a"],
                "wall_rule": model["wall_rule"],
                "domain_wall": model["domain_wall"],
                "dw_truncation": model["dw_truncation"],
                "init_mode": metadata["run"]["init_mode"],
                "half_occupation_gap": diagnostics.get("stationary_half_occupation_gap"),
                "bulk_half_occupation_gap": diagnostics.get("bulk_half_occupation_gap"),
                "entropy_per_circumference": diagnostics.get("stationary_entropy_per_circumference"),
                "physicality_violation": diagnostics.get("physicality_violation", max(0.0, -diagnostics.get("occupation_min", 0.0), diagnostics.get("occupation_max", 1.0) - 1.0)),
                "channel_continuum_order": convergence_order,
                "momentum_resolved_output": model["nshell"] is None and "ky" in arrays,
                "response_enabled": bool(metadata.get("response", {}).get("response_enabled", False)),
                "response_velocity_lower": (
                    float(arrays["response_velocity"][0]) if "response_velocity" in arrays else None
                ),
                "response_velocity_upper": (
                    float(arrays["response_velocity"][1]) if "response_velocity" in arrays else None
                ),
                "response_directionality_lower": (
                    float(arrays["response_mean_directionality"][0])
                    if "response_mean_directionality" in arrays else None
                ),
                "response_directionality_upper": (
                    float(arrays["response_mean_directionality"][1])
                    if "response_mean_directionality" in arrays else None
                ),
            }
        )
    summary = {
        "schema": "mean_channel_lindblad_cpu_analysis_v2_hybrid_response",
        "completed_cases": len(cases),
        "trajectory_samples": 0,
        "schedule_samples": 0,
        "permanent_covariance_bytes": 0,
        "rows": scalar_rows,
    }
    (output / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output / "analysis_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(scalar_rows[0]))
        writer.writeheader()
        writer.writerows(scalar_rows)

    main = [row for row in cases if row["metadata"]["campaign"] == "L1_MAIN"]
    if main:
        fig, ax = plt.subplots(figsize=(3.375, 2.45))
        styles = {1: ("#D92725", "^", ":"), 2: ("#2CA02C", "s", "--"), None: ("#1F77B4", "o", "-")}
        max_ny = max(row["metadata"]["model"]["Ny"] for row in main)
        for shell, (color, marker, linestyle) in styles.items():
            selected = [
                row for row in main
                if row["metadata"]["model"]["Ny"] == max_ny
                and row["metadata"]["model"]["nshell"] == shell
                and row["metadata"]["model"]["dw_truncation"] is True
            ]
            selected.sort(key=lambda row: row["metadata"]["model"]["alpha_top"])
            ax.semilogy(
                [row["metadata"]["model"]["alpha_top"] for row in selected],
                [row["metadata"]["diagnostics"]["stationary_half_occupation_gap"] for row in selected],
                color=color, marker=marker, linestyle=linestyle,
                label=rf"$n_{{\rm shell}}={shell}$" if shell is not None else r"$n_{\rm shell}=\mathrm{None}$",
            )
        ax.axvline(2.0, color="black", linestyle="--", linewidth=0.8)
        ax.set(xlabel=r"$\alpha_{\rm in}$", ylabel=r"$\min|\nu-1/2|$")
        ax.legend(frameon=False)
        save(fig, output, "integrated_half_gap_shell_comparison")

        # The static scan is an appendix diagnostic, not an additional finite-size
        # claim.  Everything in this panel comes from the canonical Ny=64 arm.
        fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.8), sharex=True)
        for shell, (color, marker, linestyle) in styles.items():
            selected = [
                row for row in main
                if row["metadata"]["model"]["Ny"] == max_ny
                and row["metadata"]["model"]["nshell"] == shell
                and row["metadata"]["model"]["dw_truncation"] is True
            ]
            selected.sort(key=lambda row: row["metadata"]["model"]["alpha_top"])
            alpha = np.asarray([row["metadata"]["model"]["alpha_top"] for row in selected])
            label = rf"$n_{{\rm shell}}={shell}$" if shell is not None else r"$n_{\rm shell}=\mathrm{None}$"
            half_gap = np.asarray([
                row["metadata"]["diagnostics"]["stationary_half_occupation_gap"]
                for row in selected
            ])
            bulk_gap = np.asarray([
                row["metadata"]["diagnostics"]["bulk_half_occupation_gap"]
                for row in selected
            ])
            entropy_density = np.asarray([
                row["metadata"]["diagnostics"]["stationary_entropy_per_circumference"]
                for row in selected
            ])
            localization = []
            relaxation = []
            for row in selected:
                profile = np.asarray(row["arrays"]["wall_midgap_x_profile"], dtype=float)
                walls = row["metadata"]["diagnostics"]["wall_locations"]
                wall_columns = sorted({
                    (int(wall) + delta) % profile.size
                    for wall in walls for delta in (-1, 0, 1)
                })
                localization.append(float(np.sum(profile[wall_columns]) / np.sum(profile)))
                rates = np.asarray(row["arrays"]["leading_integrated_decay_rates"], dtype=float)
                positive = rates[rates > 1e-12]
                relaxation.append(float(positive[0]) if positive.size else 0.0)
            axes[0, 0].semilogy(alpha, half_gap, color=color, marker=marker,
                                linestyle=linestyle, label=label)
            axes[0, 0].semilogy(alpha, bulk_gap, color=color, linestyle=linestyle,
                                linewidth=.8, alpha=.45)
            axes[0, 1].plot(alpha, entropy_density, color=color, marker=marker,
                            linestyle=linestyle)
            axes[1, 0].plot(alpha, localization, color=color, marker=marker,
                            linestyle=linestyle)
            axes[1, 1].semilogy(alpha, relaxation, color=color, marker=marker,
                               linestyle=linestyle)
        for ax in axes.flat:
            ax.axvline(2.0, color="black", linestyle="--", linewidth=.7)
            ax.set_xlabel(r"$\alpha_{\rm in}$")
        axes[0, 0].set_ylabel(r"occupation gaps")
        axes[0, 0].legend(frameon=False)
        axes[0, 1].set_ylabel(r"$S_{\rm G}/N_y$")
        axes[1, 0].set_ylabel("wall localization weight")
        axes[1, 1].set_ylabel("leading relaxation rate")
        save(fig, output, "appendix_static_alpha_scan")

        untruncated = [
            row for row in main
            if row["metadata"]["model"]["nshell"] is None
            and row["metadata"]["model"]["dw_truncation"] is True
        ]
        representative = min(
            untruncated,
            key=lambda row: (abs(row["metadata"]["model"]["alpha_top"] - 1.0), -row["metadata"]["model"]["Ny"]),
        )
        arrays = representative["arrays"]
        order = np.argsort(arrays["ky"])
        fig, ax = plt.subplots(figsize=(3.375, 2.45))
        ax.plot(arrays["ky"][order], arrays["occupation_spectrum_ky"][order], ".", color="0.65", ms=1.0)
        for branch, color in zip(arrays["wall_branch_occupations_ky"], ("#D92725", "#1F77B4")):
            ax.plot(arrays["ky"][order], branch[order], color=color, linewidth=1.1)
        ax.axhline(0.5, color="black", linestyle="--", linewidth=0.7)
        ax.set(xlabel=r"$k_y$", ylabel=r"occupation $\nu$")
        save(fig, output, "untruncated_ky_occupation_spectrum")

        response_cases = [
            row for row in main
            if row["metadata"].get("response", {}).get("response_enabled", False)
        ]
        canonical_largest = max(
            (
                row for row in response_cases
                if row["metadata"]["model"]["nshell"] is None
                and row["metadata"]["model"]["dw_truncation"] is True
            ),
            key=lambda row: row["metadata"]["model"]["Ny"],
        )
        arrays = canonical_largest["arrays"]
        fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.45), sharex=True, sharey=True)
        extent = [0, canonical_largest["metadata"]["model"]["Ny"] - 1,
                  arrays["response_times"][-1], arrays["response_times"][0]]
        vmax = float(np.max(np.abs(arrays["response_density_ty"])))
        for wall, ax in enumerate(axes):
            image = ax.imshow(arrays["response_density_ty"][wall], aspect="auto", cmap="RdBu_r",
                              vmin=-vmax, vmax=vmax, extent=extent)
            ax.set(xlabel=r"displacement $y-y_0$", title=f"wall {wall + 1}")
        axes[0].set_ylabel("response time")
        fig.colorbar(image, ax=axes, label=r"$\chi_w(y,t)$")
        save(fig, output, "directional_density_response_heatmaps")

        fig, ax = plt.subplots(figsize=(3.375, 2.45))
        for dw_truncation, linestyle in ((True, "-"), (False, "--")):
            selected = [
                row for row in response_cases
                if row["metadata"]["model"]["nshell"] is None
                and row["metadata"]["model"]["dw_truncation"] is dw_truncation
            ]
            selected.sort(key=lambda row: row["metadata"]["model"]["Ny"])
            inv_ny = [1.0 / row["metadata"]["model"]["Ny"] for row in selected]
            velocity = np.asarray([row["arrays"]["response_velocity"] for row in selected])
            for wall, color in enumerate(("#D92725", "#1F77B4")):
                ax.plot(inv_ny, velocity[:, wall], color=color, marker=("^", "o")[wall],
                        linestyle=linestyle,
                        label=f"wall {wall + 1}, dwtrunc={int(dw_truncation)}")
        ax.axhline(0.0, color="black", linewidth=0.7)
        ax.set(xlabel=r"$1/N_y$", ylabel="positive-lobe velocity")
        ax.legend(frameon=False, ncol=2)
        save(fig, output, "response_velocity_size_check")

        fig, ax = plt.subplots(figsize=(3.375, 2.45))
        order = np.argsort(arrays["response_finite_channel_p"])
        p = arrays["response_finite_channel_p"][order]
        error = arrays["response_finite_channel_relative_error"][order]
        ax.loglog(p, error, color="#1F77B4", marker="o", label="response error")
        ax.loglog(p, error[0] * (p / p[0]) ** 2, color="black", linestyle="--", label=r"$p^2$")
        ax.set(xlabel=r"finite-channel step $p$", ylabel="response relative error")
        ax.legend(frameon=False)
        save(fig, output, "response_channel_to_continuous")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    output = args.output or args.run_root / "analysis"
    result = analyze(args.run_root, output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
