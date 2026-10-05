#!/usr/bin/env python3
"""Regenerate estimator-annotated figures used by the campaign working note.

The production archive intentionally retained momentum-resolved arrays only for
``nshell=None``.  This script reconstructs the published-convention correlation
matrix ``G = C_arch.T`` from the pinned campaign solver and makes the
shell-one/shell-two/full-frame comparison requested
for the paper summary.  It also annotates the archived exact response with the
center estimator and fit window; it does not modify the production results.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
CAMPAIGN = REPO / "00_WORKSPACE" / "CURRENT" / "mean_channel_lindblad_cpu_campaign"
RUN_ROOT = (
    CAMPAIGN
    / "results"
    / "mean_channel_lindblad_cpu_v2_hybrid_response_production_df092e76037c014f"
)
CANONICAL_CASE = (
    RUN_ROOT
    / "cases"
    / "L1_MAIN_N20x64_wall_ain-1p0_nsh-none_dwtrunc-1_na-0p5_init-maxmix.npz"
)
SHELL_ONE_METADATA = (
    RUN_ROOT
    / "cases"
    / "L1_MAIN_N20x64_wall_ain-1p0_nsh-1_dwtrunc-1_na-0p5_init-maxmix.json"
)
SHELL_TWO_METADATA = (
    RUN_ROOT
    / "cases"
    / "L1_MAIN_N20x64_wall_ain-1p0_nsh-2_dwtrunc-1_na-0p5_init-maxmix.json"
)
TEX_SOURCE = HERE / "mean_channel_lindblad_cpu_v2_hybrid_response_working_note.tex"
sys.path.insert(0, str(CAMPAIGN))

from mean_channel_lindblad_cpu import MeanChannelLindbladCPU  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_production_manifest() -> dict[str, str]:
    """Verify and return all immutable case-file hashes from the run manifest."""

    manifest = json.loads((RUN_ROOT / "manifest.json").read_text(encoding="utf-8"))
    observed: dict[str, str] = {}
    for row in manifest["cases"]:
        for path_key, hash_key in (
            ("metadata", "metadata_sha256"),
            ("observables", "observables_sha256"),
        ):
            relative = Path("cases") / row[path_key]
            actual = sha256_file(RUN_ROOT / relative)
            if actual != row[hash_key]:
                raise RuntimeError(f"production checksum mismatch: {relative}")
            observed[str(relative)] = actual
    if len(observed) != 132:
        raise RuntimeError(f"expected 132 production case files, found {len(observed)}")
    return observed


def wall_branches(solver: MeanChannelLindbladCPU, covariance: np.ndarray):
    occupations, eigenvectors = np.linalg.eigh(covariance)
    x_weights = np.sum(
        np.abs(eigenvectors.reshape(solver.ny, solver.nx, 2, solver.d)) ** 2,
        axis=2,
    )
    branches, weights, slopes, fit_masks = [], [], [], []
    assert solver.wall_locations is not None
    for wall in solver.wall_locations:
        columns = sorted({(wall + delta) % solver.nx for delta in (-1, 0, 1)})
        wall_weight = np.sum(x_weights[:, columns, :], axis=1)
        eligible = wall_weight >= 0.25
        cost = np.where(
            eligible,
            np.abs(occupations - 0.5) + 0.02 * (1.0 - wall_weight),
            np.inf,
        )
        mode = np.argmin(cost, axis=1)
        missing = ~np.any(eligible, axis=1)
        mode[missing] = np.argmax(wall_weight[missing], axis=1)
        branch = occupations[np.arange(solver.ny), mode]
        branch_weight = wall_weight[np.arange(solver.ny), mode]
        cutoff = max(2.5 * 2.0 * np.pi / solver.ny, 0.22)
        fit = np.abs(2.0 * np.pi * np.fft.fftfreq(solver.ny)) <= cutoff
        slope, _ = np.polyfit(
            2.0 * np.pi * np.fft.fftfreq(solver.ny)[fit], branch[fit], 1
        )
        branches.append(branch)
        weights.append(branch_weight)
        slopes.append(float(slope))
        fit_masks.append(fit)
    return (
        occupations,
        np.asarray(branches),
        np.asarray(weights),
        np.asarray(slopes),
        np.asarray(fit_masks),
        x_weights,
    )


def assert_close(actual: np.ndarray | float, expected: np.ndarray | float, label: str) -> None:
    if not np.allclose(actual, expected, rtol=1e-6, atol=1e-14):
        raise RuntimeError(f"{label} changed: observed {actual!r}, expected {expected!r}")


def verify_notation_contract() -> None:
    """Reject legacy mathematical notation outside the explicit archive bridge."""

    source = TEX_SOURCE.read_text(encoding="utf-8")
    forbidden = {
        r"\Css": "legacy stationary-covariance macro",
        r"\delta C": "legacy covariance response",
        r"C_{\rm ss}": "legacy stationary-covariance symbol",
        r"n_a": "legacy ancillary-density symbol",
        r"\eta": "legacy A/B family index",
        r"\widetilde w": "legacy lowercase OW state",
        r"V_\pm": "legacy frame symbol",
    }
    for token, label in forbidden.items():
        if token in source:
            raise RuntimeError(f"notation audit failed ({label}): {token}")
    nonarchive_c = [
        match.group(0)
        for match in re.finditer(r"C_\{[^}]+\}", source)
        if match.group(0) != r"C_{\rm arch}"
    ]
    if nonarchive_c:
        raise RuntimeError(f"notation audit found non-archive C symbols: {nonarchive_c}")
    if re.search(r"(?<!\\hat )c_[ijk]", source):
        raise RuntimeError("notation audit found an unhatted Fock operator")


def validate_published_convention(
    solver: MeanChannelLindbladCPU, solution: object
) -> None:
    """Check the G=C_arch^T equations and density-response equivalence."""

    correlation = np.swapaxes(solution.covariance_blocks, -1, -2)
    damping = np.swapaxes(solution.damping_blocks, -1, -2)
    source = solver.n_a * np.swapaxes(solution.v_minus_blocks, -1, -2)
    residual = damping @ correlation + correlation @ damping - source
    residual_max = float(np.max(np.linalg.norm(residual, axis=(-2, -1))))
    if residual_max > 5e-13:
        raise RuntimeError(
            f"published-convention Sylvester residual too large: {residual_max:.3e}"
        )
    assert_close(
        np.linalg.eigvalsh(correlation),
        np.linalg.eigvalsh(solution.covariance_blocks),
        "G/C_arch natural occupations",
    )

    with np.load(CANONICAL_CASE, allow_pickle=False) as archive:
        times = archive["response_times"]
        archived_response = archive["response_density_ty"]
        source_y = int(archive["response_source_y"])
    time_index = int(np.argmin(np.abs(times - 1.25)))
    response_time = float(times[time_index])
    vectors = solution.damping_eigenvectors
    propagator_arch = (
        vectors * np.exp(-solution.damping_eigenvalues * response_time)[:, None, :]
    ) @ np.swapaxes(vectors.conj(), -2, -1)
    response_arch, _, _ = solver._response_profiles_for_propagator(
        solution,
        propagator_arch,
        epsilon=1e-3,
        source_y=source_y,
        wall_window_columns=1,
    )

    propagator_g = np.swapaxes(propagator_arch, -1, -2)
    sinc = np.sin(1e-3) / 1e-3
    response_g = []
    for wall in solver.wall_locations:
        source_real = np.zeros((solver.ny, solver.d, 2), dtype=np.complex128)
        for orbital in (0, 1):
            source_real[source_y, 2 * int(wall) + orbital, orbital] = 1.0
        source_k = np.fft.fft(source_real, axis=0, norm="ortho")
        evolved_source_k = propagator_g @ source_k
        evolved_correlation_k = propagator_g @ (correlation @ source_k)
        evolved_source = np.fft.ifft(evolved_source_k, axis=0, norm="ortho")
        evolved_correlation = np.fft.ifft(
            evolved_correlation_k, axis=0, norm="ortho"
        )
        diagonal = 2.0 * sinc * np.imag(
            np.sum(evolved_correlation * evolved_source.conj(), axis=2)
        )
        density_xy = diagonal.reshape(solver.ny, solver.nx, 2).sum(axis=2).T
        columns = sorted({(int(wall) + offset) % solver.nx for offset in (-1, 0, 1)})
        response_g.append(density_xy[columns].sum(axis=0))
    response_g = np.asarray(response_g)
    assert_close(response_g, response_arch, "G/C_arch density response")
    assert_close(
        response_g,
        archived_response[:, time_index],
        "transformed response versus production archive",
    )


def make_response_figure() -> None:
    """Plot the exact wall response together with the center estimator and fit window."""

    with np.load(CANONICAL_CASE, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}

    times = arrays["response_times"]
    response = arrays["response_density_ty"]
    centers = arrays["response_positive_center_time"]
    norms = arrays["response_norm_time"]
    retention = arrays["response_wall_retention_time"]
    velocities = arrays["response_velocity"]
    intercepts = arrays["response_velocity_intercept"]
    r2 = arrays["response_velocity_r2"]
    ny = response.shape[-1]
    fit_min, fit_max = 2.0, 0.375 * ny
    fit_time = (times >= fit_min) & (times <= fit_max)
    active = np.stack(
        [
            fit_time & (norms[wall] > max(float(norms[wall].max()) * 1e-8, 1e-15))
            for wall in range(response.shape[0])
        ]
    )

    assert_close(velocities, np.asarray([0.077454632, -0.077454632]), "response velocity")
    assert_close(r2, np.asarray([0.99752815, 0.99752815]), "response fit R^2")
    if min(float(retention[wall, active[wall]].min()) for wall in range(2)) <= 0.999:
        raise RuntimeError("active-window wall retention fell below 0.999")

    p = arrays["finite_channel_p"]
    state_error = arrays["finite_channel_relative_error_to_continuous"]
    response_p = arrays["response_finite_channel_p"]
    response_error = arrays["response_finite_channel_relative_error"]
    assert_close(np.polyfit(np.log(p), np.log(state_error), 1)[0], 1.9969716994, "state order")
    assert_close(
        np.polyfit(np.log(response_p), np.log(response_error), 1)[0],
        2.0008756089,
        "response order",
    )

    displacement = (np.arange(ny) + ny // 2) % ny - ny // 2
    displacement_order = np.argsort(displacement)
    sorted_displacement = displacement[displacement_order]
    vmax = float(np.max(np.abs(response)))
    figure, axes = plt.subplots(1, 2, figsize=(7.0, 2.55), sharex=True, sharey=True)
    for wall, axis in enumerate(axes):
        image = axis.imshow(
            response[wall][:, displacement_order],
            aspect="auto",
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
            extent=[sorted_displacement[0], sorted_displacement[-1], times[-1], times[0]],
        )
        axis.axhline(fit_min, color="black", linestyle="--", linewidth=1.4)
        axis.axhline(fit_min, color="white", linestyle="--", linewidth=0.7)
        axis.axhline(fit_max, color="black", linestyle="--", linewidth=1.4)
        axis.axhline(fit_max, color="white", linestyle="--", linewidth=0.7)
        axis.plot(
            centers[wall, active[wall]],
            times[active[wall]],
            color="black",
            linewidth=2.0,
            marker="o",
            markersize=2.5,
        )
        axis.plot(
            centers[wall, active[wall]],
            times[active[wall]],
            color="white",
            linewidth=0.8,
            marker="o",
            markersize=1.4,
        )
        fit_values = velocities[wall] * times[active[wall]] + intercepts[wall]
        axis.plot(fit_values, times[active[wall]], color="#F6C431", linewidth=1.0)
        axis.set(
            xlabel=r"periodic displacement $\widetilde r$",
            title=rf"source and probe at wall {wall + 1}",
        )
        axis.text(
            0.03,
            0.96,
            rf"$v={velocities[wall]:+.4f}$" + "\n" + rf"$R^2={r2[wall]:.4f}$",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=7,
            color="black",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.80, "pad": 1.5},
        )
    axes[0].set_ylabel("response time")
    figure.colorbar(image, ax=axes, label=r"$\chi_a^{\rm wall}(\widetilde r,t)$")
    output = HERE / "figures" / "directional_density_response_estimators"
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    production_hashes = verify_production_manifest()
    verify_notation_contract()
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["CMU Sans Serif", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    spectrum_series: list[dict[str, object]] = []
    for nshell, title in zip(
        (1, 2, None),
        (
            r"finite shell $n_{\rm shell}=1$",
            r"finite shell $n_{\rm shell}=2$",
            r"full OW frame $n_{\rm shell}=\mathrm{None}$",
        ),
    ):
        solver = MeanChannelLindbladCPU(
            nx=20,
            ny=64,
            alpha_top=1.0,
            alpha_triv=30.0,
            domain_wall=True,
            dw_truncation=True,
            wall_rule="canonical",
            nshell=nshell,
            n_a=0.5,
        )
        solution, _ = solver.stationary_solution()
        published_correlation = np.swapaxes(solution.covariance_blocks, -1, -2)
        occupations, branches, weights, slopes, fit_masks, x_weights = wall_branches(
            solver, published_correlation
        )
        metadata_path = {
            1: SHELL_ONE_METADATA,
            2: SHELL_TWO_METADATA,
            None: CANONICAL_CASE.with_suffix(".json"),
        }[nshell]
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        diagnostics = metadata["diagnostics"]
        archived_slopes = {
            1: np.asarray([0.6599144996, -0.6599144996]),
            2: np.asarray([0.8235982845, -0.8235982845]),
            None: np.asarray(
                [entry["slope_dnu_dky"] for entry in diagnostics["branch_slopes"]]
            ),
        }[nshell]
        assert_close(slopes, archived_slopes, f"shell {nshell} slopes")
        assert_close(
            float(np.min(np.abs(occupations - 0.5))),
            diagnostics["stationary_half_occupation_gap"],
            f"shell {nshell} half-occupation gap",
        )
        union_columns = sorted(
            {
                (wall + delta) % solver.nx
                for wall in solver.wall_locations
                for delta in (-1, 0, 1)
            }
        )
        union_weight = np.sum(x_weights[:, union_columns, :], axis=1)
        bulk_gap = float(np.min(np.abs(occupations[union_weight < 0.25] - 0.5)))
        assert_close(
            bulk_gap,
            diagnostics["bulk_half_occupation_gap"],
            f"shell {nshell} bulk gap",
        )
        clipped = np.clip(occupations, 0.0, 1.0)
        entropy_terms = np.zeros_like(clipped)
        interior = (clipped > 0.0) & (clipped < 1.0)
        entropy_terms[interior] = -(
            clipped[interior] * np.log(clipped[interior])
            + (1.0 - clipped[interior]) * np.log(1.0 - clipped[interior])
        )
        assert_close(
            float(np.sum(entropy_terms) / solver.ny),
            diagnostics["stationary_entropy_per_circumference"],
            f"shell {nshell} entropy per circumference",
        )
        if nshell is None:
            validate_published_convention(solver, solution)
            dmin = float(np.min(solution.damping_eigenvalues))
            rates = (
                solution.damping_eigenvalues[:, :, None]
                + solution.damping_eigenvalues[:, None, :]
            )
            assert_close(dmin, metadata["solve"]["damping_eigenvalue_min"], "canonical d_min")
            assert_close(float(np.min(rates[rates > 1e-12])), 0.5591312933, "canonical gamma_min")
        spectrum_series.append(
            {
                "title": title,
                "ky": np.asarray(solution.ky),
                "occupations": np.asarray(occupations),
                "slopes": np.asarray(slopes),
            }
        )

    figure, axis = plt.subplots(figsize=(7.0, 2.8))
    shell_one, shell_two, full_frame = spectrum_series
    for row, color, marker, open_marker, zorder in (
        (shell_one, "#1F77B4", "o", False, 2),
        (shell_two, "#2CA02C", "s", True, 3),
        (full_frame, "#FF7F0E", "d", True, 4),
    ):
        ky = np.asarray(row["ky"])
        occupations = np.asarray(row["occupations"])
        order = np.argsort(ky)
        x_points = np.repeat(ky[order], occupations.shape[1])
        y_points = occupations[order].reshape(-1)
        axis.scatter(
            x_points,
            y_points,
            s=7.0 if open_marker else 5.0,
            marker=marker,
            facecolors="none" if open_marker else color,
            edgecolors=color,
            linewidths=0.50 if open_marker else 0.0,
            alpha=0.88 if open_marker else 0.62,
            label=str(row["title"]),
            rasterized=True,
            zorder=zorder,
        )
    axis.axhline(0.5, color="0.25", ls="--", lw=0.65, zorder=1)
    axis.set(xlabel=r"$k_y$", ylabel=r"stationary occupation $\nu$", ylim=(-0.025, 1.025))
    axis.legend(loc="upper center", ncol=3, frameon=False, handletextpad=0.35, columnspacing=1.1)
    figure.tight_layout(pad=0.45)
    output = HERE / "figures" / "overlaid_finite_shell_ky_occupation_spectrum"
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)
    make_response_figure()
    if verify_production_manifest() != production_hashes:
        raise RuntimeError("production hashes changed during summary figure generation")


if __name__ == "__main__":
    main()
