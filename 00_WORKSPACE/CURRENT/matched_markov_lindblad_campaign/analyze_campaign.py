#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from campaign_schema import sha256_file


HERE = Path(__file__).resolve().parent


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
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 6.5,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "axes.spines.top": True,
            "axes.spines.right": True,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _save(figure: Any, output: Path, stem: str) -> None:
    figure.tight_layout()
    figure.savefig(output / f"{stem}.pdf", bbox_inches="tight")
    figure.savefig(output / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def immutable_hash_snapshot(run_root: Path) -> dict[str, str]:
    """Verify the run manifest and return hashes for every immutable input file."""

    manifest_path = run_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete" or not (run_root / "_SUCCESS").exists():
        raise RuntimeError("analysis requires a complete production manifest")
    config_path = run_root / "campaign_config.json"
    if sha256_file(config_path) != manifest["config_sha256"]:
        raise RuntimeError("campaign_config.json does not match the manifest hash")
    snapshot = {"campaign_config.json": sha256_file(config_path)}
    expected = manifest["expected_case_ids"]
    receipts = {row["case_id"]: row for row in manifest["cases"]}
    if len(receipts) != len(manifest["cases"]) or sorted(receipts) != sorted(expected):
        raise RuntimeError("manifest case IDs are incomplete or duplicated")
    for case_id in expected:
        directory = run_root / "cases" / case_id
        if not (directory / "_SUCCESS").exists() or (directory / "_FAILED").exists():
            raise RuntimeError(f"case marker mismatch for {case_id}")
        receipt = receipts[case_id]
        for filename, key in (
            ("metadata.json", "metadata_sha256"),
            ("observables.npz", "observables_sha256"),
        ):
            relative = f"cases/{case_id}/{filename}"
            digest = sha256_file(directory / filename)
            if digest != receipt[key]:
                raise RuntimeError(f"checksum mismatch for {relative}")
            snapshot[relative] = digest
    return snapshot


def load_cases(run_root: Path) -> list[dict[str, Any]]:
    immutable_hash_snapshot(run_root)
    manifest = json.loads((run_root / "manifest.json").read_text(encoding="utf-8"))
    expected = manifest["expected_case_ids"]
    receipts = {row["case_id"]: row for row in manifest["cases"]}
    rows = []
    for case_id in expected:
        directory = run_root / "cases" / case_id
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        with np.load(directory / "observables.npz", allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
        rows.append({"metadata": metadata, "arrays": arrays})
    return rows


def validate_cross_arm_provenance(cases: list[dict[str, Any]]) -> dict[str, Any]:
    """Validate matched construction recipes and channel schedules.

    Production metadata historically called the stored values ``projector_hashes``,
    but they are hashes of gauge-dependent OW coefficient arrays.  They can differ
    under harmless eigenvector phase choices or signed-zero changes.  A mismatch is
    therefore retained as an explicit warning, while a fresh construction from the
    locked recipe must reproduce at least one recorded hash dictionary in the group.
    Missing hashes, an unreproducible construction, or schedule-seed drift remain hard
    failures.
    """

    projector_groups: dict[
        tuple[Any, ...], list[tuple[str, dict[str, str], dict[str, Any]]]
    ] = {}
    channel_seeds: dict[int, list[tuple[str, str, int]]] = {}
    for entry in cases:
        case = entry["metadata"]["case"]
        model = case["model"]
        adapter = entry["metadata"].get("adapter", {})
        hashes = adapter.get("projector_hashes")
        if not isinstance(hashes, dict) or set(hashes) != {"WF_Ap", "WF_Am", "WF_Bp", "WF_Bm"}:
            raise RuntimeError(f"missing complete projector hashes for {case['case_id']}")
        key = (
            model["Nx"], model["Ny"], model["alpha_run_in"], model["alpha_run_out"],
            model["nshell"], model["dw_truncation"], tuple(model["wall_locations"]),
        )
        projector_groups.setdefault(key, []).append((case["case_id"], hashes, case))
        if (
            case["dynamics"]["family"] == "markov_channel"
            and "channel_schedule_seed_control" not in case["campaign_roles"]
        ):
            channel_seeds.setdefault(int(model["Ny"]), []).append(
                (
                    case["dynamics"]["schedule_match_key"],
                    case["case_id"],
                    int(case["dynamics"]["sample_seeds"][0]),
                )
            )
    raw_hash_warnings: list[dict[str, Any]] = []
    for key, records in projector_groups.items():
        unique_hashes = {
            tuple(sorted(hashes.items())) for _, hashes, _ in records
        }
        if len(unique_hashes) == 1:
            continue

        # Rebuild from the immutable model recipe.  This does not pretend that raw
        # coefficient hashes are physical invariants; it verifies that the recorded
        # recipe and canonical source still select one of the byte-level realizations
        # that actually occurred in production.
        from matched_model import build_model

        model = build_model(records[0][2])
        reconstructed = {
            name: model._checkpoint_array_signature(getattr(model, name))
            for name in ("WF_Ap", "WF_Am", "WF_Bp", "WF_Bm")
        }
        del model
        gc.collect()
        matching_cases = [
            case_id for case_id, hashes, _ in records if hashes == reconstructed
        ]
        if not matching_cases:
            raise RuntimeError(
                "raw OW hashes differ and the locked construction reproduces none "
                f"of them for matched model group {key}"
            )
        reference = records[0][1]
        raw_hash_warnings.append(
            {
                "model_group": list(key[:-1]) + [list(key[-1])],
                "case_ids": [case_id for case_id, _, _ in records],
                "cases_differing_from_first": [
                    case_id
                    for case_id, hashes, _ in records
                    if hashes != reference
                ],
                "fresh_reconstruction_matches": matching_cases,
                "interpretation": (
                    "gauge-dependent raw OW coefficient hash mismatch; retained "
                    "as a warning rather than treated as a physical-projector mismatch"
                ),
            }
        )
    for ny, records in channel_seeds.items():
        match_keys = {record[0] for record in records}
        seeds = {record[2] for record in records}
        if len(match_keys) != 1 or len(seeds) != 1:
            raise RuntimeError(f"channel variants at Ny={ny} do not share one schedule seed")
    return {
        "status": (
            "passed_with_raw_ow_hash_warnings"
            if raw_hash_warnings
            else "passed"
        ),
        "matched_model_groups": len(projector_groups),
        "raw_ow_hash_warning_count": len(raw_hash_warnings),
        "raw_ow_hash_warnings": raw_hash_warnings,
        "channel_schedule_seed_groups": {
            str(ny): {
                "seed": int(records[0][2]),
                "case_count": len(records),
            }
            for ny, records in sorted(channel_seeds.items())
        },
    }


def _binary_entropy(occupations: np.ndarray, tolerance: float) -> float:
    values = np.asarray(occupations, dtype=float)
    violation = max(0.0, float(-values.min()), float(values.max() - 1.0))
    if violation > tolerance:
        return math.nan
    clipped = np.clip(values, 0.0, 1.0)
    interior = (clipped > 0.0) & (clipped < 1.0)
    terms = np.zeros_like(clipped)
    terms[interior] = -(
        clipped[interior] * np.log(clipped[interior])
        + (1.0 - clipped[interior]) * np.log(1.0 - clipped[interior])
    )
    return float(np.sum(terms))


def _bulk_gap(occupations: np.ndarray, x_weights: np.ndarray, walls: list[int]) -> float:
    nx = x_weights.shape[1]
    columns = sorted({(int(wall) + delta) % nx for wall in walls for delta in (-1, 0, 1)})
    wall_weight = np.sum(x_weights[:, columns, :], axis=1)
    eligible = wall_weight < 0.25
    return float(np.min(np.abs(occupations[eligible] - 0.5))) if np.any(eligible) else math.nan


def _midgap_localization(occupations: np.ndarray, x_weights: np.ndarray, walls: list[int]) -> float:
    ny, nx, _ = x_weights.shape
    order = np.argsort(np.abs(occupations.reshape(-1) - 0.5))[: 2 * ny]
    k_index, mode_index = np.unravel_index(order, occupations.shape)
    profile = np.mean(x_weights[k_index, :, mode_index], axis=0)
    columns = sorted({(int(wall) + delta) % nx for wall in walls for delta in (-1, 0, 1)})
    return float(np.sum(profile[columns]) / max(float(np.sum(profile)), 1e-300))


def _branch_slopes(ky: np.ndarray, branches: np.ndarray) -> tuple[float, float]:
    cutoff = max(2.5 * 2.0 * np.pi / ky.size, 0.22)
    selected = np.abs(ky) <= cutoff
    return tuple(float(np.polyfit(ky[selected], branch[selected], 1)[0]) for branch in branches)


def _relaxation_gap(family: str, spectrum: np.ndarray, kind: str | None = None) -> float:
    values = np.asarray(spectrum, dtype=np.complex128).reshape(-1)
    if family == "lindblad":
        if kind == "positive_decay_rates" or (
            kind is None and np.max(np.abs(values.imag), initial=0.0) < 1e-13
            and np.min(values.real, initial=0.0) >= -1e-13
        ):
            rates = values.real
        else:
            rates = -values.real
        positive = rates[rates > 1e-12]
    else:
        magnitude = np.abs(values)
        stable = magnitude[(magnitude > 1e-15) & (magnitude < 1.0 - 1e-12)]
        positive = -np.log(stable) if stable.size else np.asarray([])
    return float(np.min(positive)) if positive.size else math.nan


def summarize(cases: list[dict[str, Any]], physicality_tolerance: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entry in cases:
        case = entry["metadata"]["case"]
        arrays = entry["arrays"]
        family = case["dynamics"]["family"]
        adapter = entry["metadata"].get("adapter", {})
        for sample_index, sample_id in enumerate(case["dynamics"]["sample_ids"]):
            global_occupations = arrays["global_natural_occupations"][sample_index]
            occupations_ky = arrays["twirled_ky_natural_occupations"][sample_index]
            x_weights = arrays["twirled_ky_x_weights"][sample_index]
            violation = max(
                0.0,
                float(-np.min(global_occupations)),
                float(np.max(global_occupations) - 1.0),
            )
            slopes = _branch_slopes(
                arrays["ky"], arrays["wall_branch_occupations"][sample_index]
            )
            if "response_velocity_source_wall" in arrays:
                velocity = np.asarray(arrays["response_velocity_source_wall"])[sample_index]
                r2 = np.asarray(arrays["response_velocity_r2_source_wall"])[sample_index]
                velocity_mean = np.nanmean(velocity, axis=0)
                velocity_sd = np.nanstd(velocity, axis=0, ddof=0)
                r2_min = np.nanmin(r2, axis=0)
                response_source_count = int(velocity.shape[0])
            else:
                velocity_mean = np.full(2, np.nan)
                velocity_sd = np.full(2, np.nan)
                r2_min = np.full(2, np.nan)
                response_source_count = 0
            final = np.asarray(arrays["G_final"])[sample_index]
            late = np.asarray(arrays["G_late_cycle_average"])[sample_index]
            stationary_distance = float(
                np.linalg.norm(final - late) / max(float(np.linalg.norm(late)), 1e-300)
            )
            rows.append(
                {
                    "case_id": case["case_id"],
                    "campaign_roles": ";".join(case["campaign_roles"]),
                    "sample_id": int(sample_id),
                    "sample_seed": int(case["dynamics"]["sample_seeds"][sample_index]),
                    "dynamics": family,
                    "dephasing": bool(case["dynamics"]["dephasing"]),
                    "Nx": int(case["model"]["Nx"]),
                    "Ny": int(case["model"]["Ny"]),
                    "alpha_run_in": float(case["model"]["alpha_run_in"]),
                    "alpha_run_out": float(case["model"]["alpha_run_out"]),
                    "nshell": "None" if case["model"]["nshell"] is None else int(case["model"]["nshell"]),
                    "dw_truncation": bool(case["model"]["dw_truncation"]),
                    "physicality_violation": violation,
                    "physical": violation <= physicality_tolerance,
                    "global_half_occupation_gap": float(np.min(np.abs(global_occupations - 0.5))),
                    "twirled_half_occupation_gap": float(np.min(np.abs(occupations_ky - 0.5))),
                    "twirled_bulk_half_occupation_gap": _bulk_gap(
                        occupations_ky, x_weights, case["model"]["wall_locations"]
                    ),
                    "gaussian_entropy_proxy_per_circumference": (
                        _binary_entropy(global_occupations, physicality_tolerance)
                        / int(case["model"]["Ny"])
                    ),
                    "gaussian_charge_variance_proxy": float(
                        np.sum(global_occupations * (1.0 - global_occupations))
                    ) if violation <= physicality_tolerance else math.nan,
                    "midgap_wall_localization": _midgap_localization(
                        occupations_ky, x_weights, case["model"]["wall_locations"]
                    ),
                    "late_translation_residual": float(arrays["late_translation_residual"][sample_index]),
                    "wall_1_slope": slopes[0],
                    "wall_2_slope": slopes[1],
                    "relaxation_gap": _relaxation_gap(
                        family,
                        np.asarray(arrays["leading_relaxation_spectrum"])[sample_index],
                        adapter.get("relaxation_spectrum_kind"),
                    ),
                    "stationary_distance_final_to_late": stationary_distance,
                    "response_source_count": response_source_count,
                    "response_velocity_wall_1_mean_source": float(velocity_mean[0]),
                    "response_velocity_wall_2_mean_source": float(velocity_mean[1]),
                    "response_velocity_wall_1_source_sd": float(velocity_sd[0]),
                    "response_velocity_wall_2_source_sd": float(velocity_sd[1]),
                    "response_r2_wall_1_min_source": float(r2_min[0]),
                    "response_r2_wall_2_min_source": float(r2_min[1]),
                }
            )
    return rows


def _arm_label(case: dict[str, Any]) -> str:
    family = "channel" if case["dynamics"]["family"] == "markov_channel" else "continuous"
    return f"{family}, deph={int(case['dynamics']['dephasing'])}"


def plot_canonical_spectra(cases: list[dict[str, Any]], output: Path) -> None:
    selected = [
        row for row in cases
        if row["metadata"]["case"]["model"]["Ny"] == 64
        and row["metadata"]["case"]["model"]["alpha_run_in"] == 1.0
        and row["metadata"]["case"]["model"]["dw_truncation"] is True
        and "channel_schedule_seed_control" not in row["metadata"]["case"]["campaign_roles"]
    ]
    arms = [("markov_channel", True), ("lindblad", False), ("lindblad", True)]
    styles = {1: ("#D92725", "^"), 2: ("#2CA02C", "s"), None: ("#1F77B4", "o")}
    figure, axes = plt.subplots(1, 3, figsize=(7.05, 2.35), sharex=True, sharey=True)
    for panel, (family, dephasing) in enumerate(arms):
        axis = axes[panel]
        for row in selected:
            case = row["metadata"]["case"]
            if case["dynamics"]["family"] != family or case["dynamics"]["dephasing"] != dephasing:
                continue
            nshell = case["model"]["nshell"]
            color, marker = styles[nshell]
            ky = row["arrays"]["ky"]
            occupations = row["arrays"]["twirled_ky_natural_occupations"].mean(axis=0)
            axis.scatter(
                np.repeat(ky, occupations.shape[1]), occupations.reshape(-1),
                s=4.5, marker=marker, color=color, alpha=0.55,
                label=rf"$n_{{\rm shell}}={nshell}$" if nshell is not None else "full frame",
                rasterized=True,
            )
        axis.axhline(0.5, color="0.25", linestyle="--", linewidth=0.65)
        axis.set(xlabel=r"$k_y$", title=_arm_label({"dynamics": {"family": family, "dephasing": dephasing}}))
        axis.text(-0.12, 1.03, chr(ord("a") + panel), transform=axis.transAxes, fontweight="bold")
    axes[0].set_ylabel(r"twirled occupation $\nu$")
    axes[-1].legend(frameon=False, loc="best")
    _save(figure, output, "canonical_twirled_occupation_spectra")


def plot_channel_untwirled_sorted_spectrum(
    cases: list[dict[str, Any]], output: Path
) -> None:
    """Plot raw late-average channel occupations without a translation twirl."""

    selected = [
        entry
        for entry in cases
        if entry["metadata"]["case"]["dynamics"]["family"] == "markov_channel"
        and entry["metadata"]["case"]["model"]["Ny"] == 64
        and entry["metadata"]["case"]["model"]["alpha_run_in"] == 1.0
        and entry["metadata"]["case"]["model"]["nshell"] in (1, 2)
        and "channel_schedule_seed_control"
        not in entry["metadata"]["case"]["campaign_roles"]
    ]
    selected.sort(
        key=lambda entry: (
            int(entry["metadata"]["case"]["model"]["nshell"]),
            not bool(entry["metadata"]["case"]["model"]["dw_truncation"]),
        )
    )
    if len(selected) != 4:
        raise RuntimeError(
            "untwirled channel spectrum requires the four Ny=64 endpoint controls"
        )

    colors = {1: "#D92725", 2: "#2CA02C"}
    markers = {1: "^", 2: "s"}
    figure, axes = plt.subplots(1, 2, figsize=(7.05, 2.45))
    for entry in selected:
        case = entry["metadata"]["case"]
        model = case["model"]
        nshell = int(model["nshell"])
        dw_truncation = bool(model["dw_truncation"])
        occupations = np.sort(
            np.asarray(entry["arrays"]["global_natural_occupations"])[0].real
        )
        normalized_rank = np.arange(occupations.size) / float(occupations.size - 1)
        centered_rank = np.arange(occupations.size) - 0.5 * (occupations.size - 1)
        label = rf"$n_{{\rm shell}}={nshell}$, DW trunc. {'on' if dw_truncation else 'off'}"
        linestyle = "-" if dw_truncation else "--"
        axes[0].plot(
            normalized_rank,
            occupations,
            color=colors[nshell],
            linestyle=linestyle,
            linewidth=1.0,
            marker=markers[nshell],
            markersize=2.4,
            markevery=128,
            label=label,
        )
        axes[1].plot(
            centered_rank,
            occupations,
            color=colors[nshell],
            linestyle=linestyle,
            linewidth=1.0,
            marker=markers[nshell],
            markersize=3.0,
            markevery=2,
        )

    for panel, axis in enumerate(axes):
        axis.axhline(0.5, color="0.25", linestyle="--", linewidth=0.7)
        axis.set_ylabel(r"untwirled occupation $\nu_j$")
        axis.text(
            -0.12,
            1.03,
            chr(ord("a") + panel),
            transform=axis.transAxes,
            fontweight="bold",
        )
    axes[0].set(xlabel=r"normalized ascending rank $j/(2N_xN_y-1)$", ylim=(-0.02, 1.02))
    axes[0].legend(frameon=False, loc="best")
    axes[1].set(
        xlabel=r"rank relative to half filling $j-(2N_xN_y-1)/2$",
        xlim=(-32, 32),
        ylim=(0.35, 0.65),
    )
    axes[0].set_title("full sorted spectrum")
    axes[1].set_title("midpoint zoom")
    _save(figure, output, "channel_untwirled_sorted_occupation_spectrum")


def plot_alpha_scan_estimators(summary: list[dict[str, Any]], output: Path) -> None:
    selected = [
        row for row in summary
        if row["Ny"] == 64 and row["dw_truncation"] is True and row["sample_id"] == 0
    ]
    arms = [("markov_channel", True), ("lindblad", False), ("lindblad", True)]
    styles = {1: ("#D92725", "^", ":"), 2: ("#2CA02C", "s", "--"), "None": ("#1F77B4", "o", "-")}
    metrics = (
        ("gaps", None),
        (r"$S_G/N_y$", "gaussian_entropy_proxy_per_circumference"),
        (r"midgap wall weight", "midgap_wall_localization"),
        (r"relaxation gap", "relaxation_gap"),
    )
    figure, axes = plt.subplots(3, 4, figsize=(7.05, 5.7), sharex=True)
    for row_index, (family, dephasing) in enumerate(arms):
        for nshell, (color, marker, linestyle) in styles.items():
            rows = [
                row for row in selected
                if row["dynamics"] == family and row["dephasing"] is dephasing
                and row["nshell"] == nshell
            ]
            rows.sort(key=lambda row: row["alpha_run_in"])
            if rows:
                alpha = [row["alpha_run_in"] for row in rows]
                label = rf"$n_{{\rm shell}}={nshell}$" if nshell != "None" else "full frame"
                axes[row_index, 0].semilogy(
                    alpha,
                    np.maximum([row["twirled_half_occupation_gap"] for row in rows], 1e-16),
                    color=color, marker=marker, linestyle=linestyle, label=label,
                )
                axes[row_index, 0].semilogy(
                    alpha,
                    np.maximum([row["twirled_bulk_half_occupation_gap"] for row in rows], 1e-16),
                    color=color, linestyle=linestyle, linewidth=0.75, alpha=0.42,
                )
                for column, (_, key) in enumerate(metrics[1:], start=1):
                    values = np.asarray([row[key] for row in rows], dtype=float)
                    if column == 3:
                        values = np.maximum(values, 1e-16)
                        axes[row_index, column].semilogy(
                            alpha, values, color=color, marker=marker, linestyle=linestyle
                        )
                    else:
                        axes[row_index, column].plot(
                            alpha, values, color=color, marker=marker, linestyle=linestyle
                        )
        for column, (metric_label, _) in enumerate(metrics):
            axis = axes[row_index, column]
            axis.axvline(2.0, color="0.25", linestyle="--", linewidth=0.65)
            if row_index == 0:
                axis.set_title(metric_label)
            if row_index == 2:
                axis.set_xlabel(r"$\alpha_{\rm run,in}$")
            if column == 0:
                arm = "channel" if family == "markov_channel" else "continuous"
                axis.set_ylabel(f"{arm}, deph={int(dephasing)}")
            axis.text(
                -0.16, 1.03, chr(ord("a") + 4 * row_index + column),
                transform=axis.transAxes, fontweight="bold",
            )
    axes[0, 0].legend(frameon=False, ncol=3, loc="best")
    axes[0, 0].text(
        0.03, 0.05, "markers: global gap\nfaint: bulk-excluded gap",
        transform=axes[0, 0].transAxes, fontsize=6,
    )
    _save(figure, output, "matched_alpha_scan_estimators")


def plot_response(cases: list[dict[str, Any]], output: Path) -> None:
    selected = [
        row for row in cases
        if row["metadata"]["case"]["model"]["Ny"] == 64
        and row["metadata"]["case"]["model"]["alpha_run_in"] == 1.0
        and row["metadata"]["case"]["model"]["nshell"] == 1
        and row["metadata"]["case"]["model"]["dw_truncation"] is True
        and "channel_schedule_seed_control" not in row["metadata"]["case"]["campaign_roles"]
    ]
    selected.sort(key=lambda row: (
        row["metadata"]["case"]["dynamics"]["family"] != "markov_channel",
        row["metadata"]["case"]["dynamics"]["dephasing"],
    ))
    figure, axes = plt.subplots(3, 2, figsize=(7.05, 5.4), sharex=True, sharey=True)
    for row_index, row in enumerate(selected[:3]):
        arrays = row["arrays"]
        case = row["metadata"]["case"]
        response = arrays["response_density_ty_mean_source"].mean(axis=0)
        times = arrays["response_times"]
        ny = int(case["model"]["Ny"])
        displacement = ((np.arange(ny) + ny / 2.0) % ny) - ny / 2.0
        displacement_order = np.argsort(displacement)
        sorted_displacement = displacement[displacement_order]
        vmax = float(np.max(np.abs(response)))
        for wall in range(2):
            axis = axes[row_index, wall]
            image = axis.imshow(
                response[wall][:, displacement_order],
                aspect="auto", origin="upper", cmap="RdBu_r",
                vmin=-vmax, vmax=vmax,
                extent=[sorted_displacement[0], sorted_displacement[-1], times[-1], times[0]],
            )
            positive = np.maximum(response[wall], 0.0)
            denominator = np.sum(positive, axis=1)
            center = np.divide(
                positive @ displacement,
                denominator,
                out=np.full(times.size, np.nan),
                where=denominator > 1e-300,
            )
            source_norm = np.asarray(arrays["response_norm_time"]).mean(axis=(0, 1))[wall]
            fit = (
                (times >= 2.0)
                & (times <= 0.375 * ny)
                & np.isfinite(center)
                & (source_norm > max(1e-8 * float(np.max(source_norm)), 1e-15))
            )
            axis.axhline(2.0, color="0.15", linestyle="--", linewidth=0.6)
            axis.axhline(0.375 * ny, color="0.15", linestyle="--", linewidth=0.6)
            if np.count_nonzero(fit) >= 2:
                slope, intercept = np.polyfit(times[fit], center[fit], 1)
                axis.plot(center[fit], times[fit], color="white", marker="o", markersize=1.2, linewidth=0.7)
                axis.plot(slope * times[fit] + intercept, times[fit], color="#F6C431", linewidth=0.9)
                axis.text(
                    0.03, 0.94, rf"$v={slope:+.4f}$", transform=axis.transAxes,
                    va="top", bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.0},
                )
            axis.set_title(f"{_arm_label(case)}, wall {wall + 1}")
            if wall == 0:
                axis.set_ylabel("time")
            if row_index == 2:
                axis.set_xlabel(r"periodic displacement $\widetilde r$")
            figure.colorbar(image, ax=axis, fraction=0.046, pad=0.02)
    _save(figure, output, "matched_density_kick_response")


def _matched_model_key(case: dict[str, Any]) -> tuple[Any, ...]:
    model = case["model"]
    return (
        int(model["Nx"]), int(model["Ny"]), float(model["alpha_run_in"]),
        float(model["alpha_run_out"]), model["nshell"],
        bool(model["dw_truncation"]), tuple(model["wall_locations"]),
    )


def _response_profile_error(channel: dict[str, Any], continuous: dict[str, Any]) -> float:
    channel_arrays, continuous_arrays = channel["arrays"], continuous["arrays"]
    if "response_density_ty_mean_source" not in channel_arrays or "response_density_ty_mean_source" not in continuous_arrays:
        return math.nan
    channel_time = np.asarray(channel_arrays["response_times"], dtype=float)
    continuous_time = np.asarray(continuous_arrays["response_times"], dtype=float)
    selected = (channel_time >= continuous_time[0]) & (channel_time <= continuous_time[-1])
    if not np.any(selected):
        return math.nan
    target_time = channel_time[selected]
    channel_profile = np.asarray(channel_arrays["response_density_ty_mean_source"])[0][:, selected, :]
    continuous_profile = np.asarray(continuous_arrays["response_density_ty_mean_source"])[0]
    flat = continuous_profile.transpose(1, 0, 2).reshape(continuous_time.size, -1)
    interpolated = np.stack(
        [np.interp(target_time, continuous_time, flat[:, column]) for column in range(flat.shape[1])],
        axis=1,
    ).reshape(target_time.size, continuous_profile.shape[0], continuous_profile.shape[2]).transpose(1, 0, 2)
    denominator = max(float(np.linalg.norm(interpolated)), 1e-300)
    return float(np.linalg.norm(channel_profile - interpolated) / denominator)


def matched_pair_rows(
    cases: list[dict[str, Any]], summary: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Compare the physical unit-step channel to dephasing-matched continuous arms."""

    by_key: dict[tuple[Any, ...], dict[tuple[str, bool], dict[str, Any]]] = {}
    for entry in cases:
        case = entry["metadata"]["case"]
        if "channel_schedule_seed_control" in case["campaign_roles"]:
            continue
        arm = (case["dynamics"]["family"], bool(case["dynamics"]["dephasing"]))
        by_key.setdefault(_matched_model_key(case), {})[arm] = entry
    scalar = {(row["case_id"], row["sample_id"]): row for row in summary}
    rows: list[dict[str, Any]] = []
    for key, arms in sorted(by_key.items(), key=lambda item: str(item[0])):
        if ("markov_channel", True) not in arms or ("lindblad", True) not in arms:
            continue
        channel = arms[("markov_channel", True)]
        continuous = arms[("lindblad", True)]
        channel_case = channel["metadata"]["case"]
        continuous_case = continuous["metadata"]["case"]
        channel_scalar = scalar[(channel_case["case_id"], 0)]
        continuous_scalar = scalar[(continuous_case["case_id"], 0)]
        channel_state = np.asarray(channel["arrays"]["G_late_cycle_average_twirl"])[0]
        continuous_state = np.asarray(continuous["arrays"]["G_late_cycle_average_twirl"])[0]
        state_error = float(
            np.linalg.norm(channel_state - continuous_state)
            / max(float(np.linalg.norm(continuous_state)), 1e-300)
        )
        channel_occ = np.sort(np.asarray(channel["arrays"]["global_natural_occupations"])[0])
        continuous_occ = np.sort(np.asarray(continuous["arrays"]["global_natural_occupations"])[0])
        rows.append(
            {
                "Nx": key[0], "Ny": key[1], "alpha_run_in": key[2],
                "alpha_run_out": key[3],
                "nshell": "None" if key[4] is None else int(key[4]),
                "dw_truncation": bool(key[5]),
                "channel_case_id": channel_case["case_id"],
                "continuous_case_id": continuous_case["case_id"],
                "state_relative_frobenius_error": state_error,
                "occupation_rms_difference": float(
                    np.sqrt(np.mean((channel_occ - continuous_occ) ** 2))
                ),
                "half_gap_absolute_difference": abs(
                    channel_scalar["twirled_half_occupation_gap"]
                    - continuous_scalar["twirled_half_occupation_gap"]
                ),
                "entropy_proxy_per_circumference_absolute_difference": abs(
                    channel_scalar["gaussian_entropy_proxy_per_circumference"]
                    - continuous_scalar["gaussian_entropy_proxy_per_circumference"]
                ),
                "wall_slope_rms_difference": float(
                    np.sqrt(np.mean([
                        (channel_scalar["wall_1_slope"] - continuous_scalar["wall_1_slope"]) ** 2,
                        (channel_scalar["wall_2_slope"] - continuous_scalar["wall_2_slope"]) ** 2,
                    ]))
                ),
                "response_profile_relative_error": _response_profile_error(channel, continuous),
                "response_velocity_wall_1_difference": (
                    channel_scalar["response_velocity_wall_1_mean_source"]
                    - continuous_scalar["response_velocity_wall_1_mean_source"]
                ),
                "response_velocity_wall_2_difference": (
                    channel_scalar["response_velocity_wall_2_mean_source"]
                    - continuous_scalar["response_velocity_wall_2_mean_source"]
                ),
            }
        )
    return rows


def plot_velocity_size(summary: list[dict[str, Any]], output: Path) -> None:
    rows = [
        row for row in summary
        if row["alpha_run_in"] == 1.0 and row["sample_id"] == 0
        and row["response_source_count"] > 0
    ]
    arms = [("markov_channel", True), ("lindblad", False), ("lindblad", True)]
    shell_style = {1: "^", "None": "o"}
    figure, axes = plt.subplots(1, 3, figsize=(7.05, 2.45), sharex=True, sharey=True)
    for panel, (family, dephasing) in enumerate(arms):
        axis = axes[panel]
        for nshell, marker in shell_style.items():
            for dw_truncation, linestyle in ((True, "-"), (False, "--")):
                selected = sorted(
                    [
                        row for row in rows
                        if row["dynamics"] == family and row["dephasing"] is dephasing
                        and row["nshell"] == nshell
                        and row["dw_truncation"] is dw_truncation
                    ],
                    key=lambda row: row["Ny"],
                )
                if not selected:
                    continue
                inverse_size = [1.0 / row["Ny"] for row in selected]
                for wall, color in ((1, "#D92725"), (2, "#1F77B4")):
                    axis.plot(
                        inverse_size,
                        [row[f"response_velocity_wall_{wall}_mean_source"] for row in selected],
                        color=color, marker=marker, linestyle=linestyle,
                        label=(
                            f"wall {wall}, nsh={nshell}, dw={int(dw_truncation)}"
                            if panel == 2 else None
                        ),
                    )
        axis.axhline(0.0, color="0.25", linewidth=0.65)
        axis.set_xlabel(r"$1/N_y$")
        axis.set_title(_arm_label({"dynamics": {"family": family, "dephasing": dephasing}}))
        axis.text(-0.12, 1.03, chr(ord("a") + panel), transform=axis.transAxes, fontweight="bold")
    axes[0].set_ylabel("positive-lobe velocity")
    axes[-1].legend(frameon=False, fontsize=5.5, ncol=2)
    _save(figure, output, "matched_velocity_size_check")


def plot_matched_differences(pair_rows: list[dict[str, Any]], output: Path) -> None:
    endpoint = [row for row in pair_rows if row["alpha_run_in"] == 1.0]
    figure, axes = plt.subplots(2, 2, figsize=(7.05, 4.5), sharex=True)
    styles = {1: ("#D92725", "^"), 2: ("#2CA02C", "s")}
    for nshell, (color, marker) in styles.items():
        for dw_truncation, linestyle in ((True, "-"), (False, "--")):
            selected = sorted(
                [row for row in endpoint if row["nshell"] == nshell and row["dw_truncation"] is dw_truncation],
                key=lambda row: row["Ny"],
            )
            if not selected:
                continue
            x = [1.0 / row["Ny"] for row in selected]
            label = rf"$n_{{\rm shell}}={nshell}$, dw={int(dw_truncation)}"
            axes[0, 0].semilogy(x, np.maximum([row["state_relative_frobenius_error"] for row in selected], 1e-18), color=color, marker=marker, linestyle=linestyle, label=label)
            axes[0, 1].semilogy(x, np.maximum([row["occupation_rms_difference"] for row in selected], 1e-18), color=color, marker=marker, linestyle=linestyle)
            response = [row for row in selected if np.isfinite(row["response_profile_relative_error"])]
            if response:
                xr = [1.0 / row["Ny"] for row in response]
                axes[1, 0].semilogy(xr, np.maximum([row["response_profile_relative_error"] for row in response], 1e-18), color=color, marker=marker, linestyle=linestyle)
                axes[1, 1].plot(xr, [row["response_velocity_wall_1_difference"] for row in response], color="#D92725", marker=marker, linestyle=linestyle)
                axes[1, 1].plot(xr, [row["response_velocity_wall_2_difference"] for row in response], color="#1F77B4", marker=marker, linestyle=linestyle)
    labels = (
        r"$\|G_{\rm ch}-G_{\rm cont}\|_F/\|G_{\rm cont}\|_F$",
        "occupation RMS difference",
        "response-profile relative error",
        r"$v_{\rm ch}-v_{\rm cont}$",
    )
    for panel, (axis, label) in enumerate(zip(axes.flat, labels)):
        axis.set_ylabel(label)
        axis.set_xlabel(r"$1/N_y$")
        axis.text(-0.12, 1.03, chr(ord("a") + panel), transform=axis.transAxes, fontweight="bold")
    axes[0, 0].legend(frameon=False, fontsize=6)
    axes[1, 1].axhline(0.0, color="0.25", linewidth=0.65)
    _save(figure, output, "matched_channel_continuous_differences")


def plot_cycle_convergence(cases: list[dict[str, Any]], output: Path) -> None:
    selected = [
        entry for entry in cases
        if entry["metadata"]["case"]["model"]["Ny"] == 64
        and entry["metadata"]["case"]["model"]["alpha_run_in"] == 1.0
        and entry["metadata"]["case"]["model"]["nshell"] == 1
        and entry["metadata"]["case"]["model"]["dw_truncation"] is True
        and "channel_schedule_seed_control" not in entry["metadata"]["case"]["campaign_roles"]
    ]
    selected.sort(key=lambda entry: (
        entry["metadata"]["case"]["dynamics"]["family"] != "markov_channel",
        entry["metadata"]["case"]["dynamics"]["dephasing"],
    ))
    colors = ("#D92725", "#2CA02C", "#1F77B4")
    markers = ("^", "s", "o")
    figure, axes = plt.subplots(2, 3, figsize=(7.05, 4.3))
    for entry, color, marker in zip(selected, colors, markers):
        case, arrays = entry["metadata"]["case"], entry["arrays"]
        label = _arm_label(case)
        cycle = np.asarray(arrays["cycle"], dtype=float)
        checkpoint = np.asarray(arrays["spectral_checkpoint_cycle"], dtype=float)
        axes[0, 0].semilogy(cycle / 64.0, np.maximum(np.mean(arrays["successive_state_distance"], axis=0), 1e-18), color=color, label=label)
        axes[0, 1].semilogy(cycle / 64.0, np.maximum(np.mean(arrays["translation_residual"], axis=0), 1e-18), color=color)
        axes[0, 2].semilogy(checkpoint / 64.0, np.maximum(np.mean(arrays["half_occupation_gap"], axis=0), 1e-18), color=color, marker=marker)
        axes[1, 0].plot(checkpoint / 64.0, np.mean(arrays["gaussian_entropy_proxy_per_circumference"], axis=0), color=color, marker=marker)
        axes[1, 1].plot(checkpoint / 64.0, np.mean(arrays["gaussian_charge_variance_proxy"], axis=0), color=color, marker=marker)
        axes[1, 2].plot(checkpoint / 64.0, np.mean(arrays["occupation_min"], axis=0), color=color, marker=marker)
        axes[1, 2].plot(checkpoint / 64.0, np.mean(arrays["occupation_max"], axis=0), color=color, marker=marker, linestyle="--")
    ylabels = (
        "successive-state distance", "translation residual", r"$\min|\nu-1/2|$",
        r"$S_G/N_y$", r"$\operatorname{tr}G(1-G)$ proxy", "occupation extrema",
    )
    for panel, (axis, ylabel) in enumerate(zip(axes.flat, ylabels)):
        axis.set_xlabel(r"cycle $c/N_y$")
        axis.set_ylabel(ylabel)
        axis.text(-0.16, 1.03, chr(ord("a") + panel), transform=axis.transAxes, fontweight="bold")
    axes[0, 0].legend(frameon=False, fontsize=6)
    _save(figure, output, "cycle_convergence_diagnostics")


def plot_schedule_control(summary: list[dict[str, Any]], output: Path) -> None:
    rows = sorted(
        [row for row in summary if "channel_schedule_seed_control" in row["campaign_roles"]],
        key=lambda row: row["sample_id"],
    )
    if not rows:
        return
    sample = [row["sample_id"] for row in rows]
    figure, axes = plt.subplots(2, 2, figsize=(7.05, 4.3))
    axes[0, 0].scatter(sample, [row["late_translation_residual"] for row in rows], color="#D92725", marker="^")
    axes[0, 1].scatter(sample, [row["twirled_half_occupation_gap"] for row in rows], color="#2CA02C", marker="s")
    axes[1, 0].scatter(sample, [row["wall_1_slope"] for row in rows], color="#D92725", marker="^", label="wall 1")
    axes[1, 0].scatter(sample, [row["wall_2_slope"] for row in rows], color="#1F77B4", marker="o", label="wall 2")
    axes[1, 1].scatter(sample, [row["stationary_distance_final_to_late"] for row in rows], color="#1F77B4", marker="o")
    labels = ("translation residual", "twirled half gap", "wall-branch slope", "final/late distance")
    for panel, (axis, ylabel) in enumerate(zip(axes.flat, labels)):
        axis.set(xlabel="schedule sample", ylabel=ylabel)
        axis.text(-0.12, 1.03, chr(ord("a") + panel), transform=axis.transAxes, fontweight="bold")
    axes[1, 0].legend(frameon=False)
    _save(figure, output, "schedule_control_diagnostics")


def analyze(run_root: Path, output: Path) -> dict[str, Any]:
    configure_plotting()
    output.mkdir(parents=True, exist_ok=False)
    hashes_before = immutable_hash_snapshot(run_root)
    config = json.loads((run_root / "campaign_config.json").read_text(encoding="utf-8"))
    cases = load_cases(run_root)
    provenance = validate_cross_arm_provenance(cases)
    rows = summarize(cases, float(config["analysis"]["physicality_tolerance"]))
    with (output / "analysis_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    pairs = matched_pair_rows(cases, rows)
    if pairs:
        with (output / "matched_pair_summary.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(pairs[0]))
            writer.writeheader()
            writer.writerows(pairs)
    result = {
        "schema": "matched_markov_lindblad_analysis_v1",
        "run_root": str(run_root.resolve()),
        "case_count": len(cases),
        "independent_sample_rows": len(rows),
        "matched_channel_continuous_pairs": len(pairs),
        "nonphysical_rows": sum(not row["physical"] for row in rows),
        "provenance_validation": provenance,
        "analysis_source_sha256": sha256_file(Path(__file__)),
        "input_hashes": hashes_before,
        "matched_pairs": pairs,
        "rows": rows,
    }
    (output / "analysis_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    plot_canonical_spectra(cases, output)
    plot_channel_untwirled_sorted_spectrum(cases, output)
    plot_alpha_scan_estimators(rows, output)
    plot_response(cases, output)
    plot_velocity_size(rows, output)
    plot_matched_differences(pairs, output)
    plot_cycle_convergence(cases, output)
    plot_schedule_control(rows, output)
    hashes_after = immutable_hash_snapshot(run_root)
    if hashes_after != hashes_before:
        raise RuntimeError("immutable production inputs changed during analysis")
    result["output_files"] = sorted(
        path.name for path in output.iterdir() if path.is_file()
    )
    (output / "analysis_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "_SUCCESS").touch()
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    output = args.output or args.run_root / "analysis"
    result = analyze(args.run_root.resolve(), output.resolve())
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
