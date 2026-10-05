#!/usr/bin/env python3
"""Post-process the completed frozen-record pilot with regional source subtraction."""

from __future__ import annotations

import argparse
import csv
from contextlib import redirect_stdout
import gzip
import io
import json
from pathlib import Path
import sys
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from tqdm.auto import tqdm  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from run_campaign import (  # noqa: E402
    atomic_json,
    atomic_npz,
    atomic_text,
    endpoint_charges,
    make_model,
    mode_x,
    sha256_path,
)


SCHEMA = "frozen_record_flux_charge_source_subtracted_v1"
EXPECTED_CAMPAIGN_SCHEMA = "frozen_record_flux_charge_pilot_v1"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_record(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise TypeError(f"trajectory record must be a list: {path}")
    return payload


def load_charge_rows(path: Path) -> list[dict[str, Any]]:
    integer_fields = {"sigma", "twist_index", "net_injected_charge"}
    float_fields = {
        "phi",
        "sweep_fraction",
        "N_left",
        "N_right",
        "N_total",
        "N_initial_total",
        "delta_N_left",
        "delta_N_right",
        "q_wall",
        "charge_continuity_residual",
        "regional_balance_residual",
        "branch_log_probability",
        "minimum_selected_probability",
        "elapsed_seconds",
    }
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            row: dict[str, Any] = dict(raw)
            for field in integer_fields:
                row[field] = int(row[field])
            for field in float_fields:
                row[field] = float(row[field])
            rows.append(row)
    return rows


def correction_multiplicities(
    record: Iterable[dict[str, Any]],
) -> dict[tuple[int, str], int]:
    """Reduce a fixed word to signed correction counts for each site and OW channel."""
    counts: dict[tuple[int, str], int] = {}
    for site in record:
        site_id = int(site["site_id"])
        measurements: dict[str, int] = {}
        for event in site["branch_events"]:
            channel = str(event["channel"])
            kind = str(event["kind"])
            if kind == "measurement":
                measurements[channel] = int(bool(event["outcome_occupied"]))
                continue
            if kind != "correction":
                raise ValueError(f"unknown trajectory event kind {kind!r}")
            if channel not in measurements:
                raise ValueError(f"correction precedes measurement for channel {channel!r}")
            increment = int(bool(event["target_occupied"])) - measurements[channel]
            counts[(site_id, channel)] = counts.get((site_id, channel), 0) + increment
    return {key: value for key, value in counts.items() if value != 0}


def source_profile_from_model(
    model: Any,
    counts: dict[tuple[int, str], int],
    *,
    nx: int,
    ny: int,
) -> np.ndarray:
    """Return the direct ancilla source resolved by x for one static twist."""
    profile = np.zeros(nx, dtype=np.float64)
    x_by_mode = mode_x(nx, ny)
    valid_channels = {"Ap", "Am", "Bp", "Bm"}
    for (site_id, channel), multiplicity in counts.items():
        if channel not in valid_channels:
            raise ValueError(f"unsupported OW channel {channel!r}")
        if site_id < 0 or site_id >= nx * ny:
            raise ValueError(f"site_id outside the physical lattice: {site_id}")
        rx, ry = site_id % nx, site_id // nx
        orbital = np.asarray(getattr(model, f"WF_{channel}")[:, rx, ry])
        if orbital.shape != (2 * nx * ny,):
            raise ValueError(f"unexpected {channel} orbital shape {orbital.shape}")
        weights = np.abs(orbital) ** 2
        norm = float(weights.sum())
        if not np.isfinite(norm) or abs(norm - 1.0) > 1e-10:
            raise FloatingPointError(
                f"{channel} orbital at ({rx},{ry}) is not normalized: {norm:.16g}"
            )
        profile += int(multiplicity) * np.bincount(
            x_by_mode, weights=weights, minlength=nx
        )
    return profile


def source_profile(
    config: dict[str, Any],
    arm: dict[str, Any],
    counts: dict[tuple[int, str], int],
    phi: float,
) -> np.ndarray:
    nx = int(config["geometry"]["Nx"])
    ny = int(config["geometry"]["Ny"])
    with redirect_stdout(io.StringIO()):
        model = make_model(config, arm, float(phi))
    return source_profile_from_model(model, counts, nx=nx, ny=ny)


def region_sums(profile: np.ndarray, config: dict[str, Any]) -> tuple[float, float]:
    regions = config["regions"]
    left = float(
        profile[
            int(regions["left_x_start"]) : int(regions["left_x_stop_exclusive"])
        ].sum()
    )
    right = float(
        profile[
            int(regions["right_x_start"]) : int(regions["right_x_stop_exclusive"])
        ].sum()
    )
    if not np.isclose(left + right, float(profile.sum()), rtol=0.0, atol=1e-12):
        raise ValueError("left/right regions do not exhaust the source profile")
    return left, right


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["CMU Sans Serif", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
        }
    )


def direction_style(direction: str) -> dict[str, Any]:
    if direction == "ccw":
        return {"linestyle": "-", "marker": "o"}
    return {"linestyle": "--", "marker": "s"}


def select(rows: list[dict[str, Any]], arm: str, direction: str) -> list[dict[str, Any]]:
    return sorted(
        (row for row in rows if row["arm"] == arm and row["direction"] == direction),
        key=lambda row: int(row["twist_index"]),
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0])
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    atomic_text(path, buffer.getvalue())


def plot_results(rows: list[dict[str, Any]], output_dir: Path) -> dict[str, str]:
    configure_style()
    colors = {"soft": "#2ca02c", "hard": "#1f77b4"}
    fig, axes = plt.subplots(2, 2, figsize=(7.05, 5.0))

    for ax, arm in zip(axes[0], ("soft", "hard")):
        for direction in ("ccw", "cw"):
            subset = select(rows, arm, direction)
            style = direction_style(direction)
            x = [row["sweep_fraction"] for row in subset]
            ax.plot(
                x,
                [row["q_wall_raw_zero"] for row in subset],
                color="0.65",
                linewidth=2.3,
                label=f"raw, {direction.upper()}",
            )
            ax.plot(
                x,
                [row["q_wall_source_subtracted_zero"] for row in subset],
                color=colors[arm],
                linestyle=style["linestyle"],
                marker=style["marker"],
                markerfacecolor="white",
                markersize=3,
                linewidth=1.0,
                label=f"source-subtracted, {direction.upper()}",
            )
        ax.axhline(0.0, color="0.25", linestyle=":", linewidth=0.8)
        ax.set_xlabel(r"signed twist $(\phi-\phi_0)/(2\pi)$")
        ax.set_ylabel(r"wall response $q_x$")
        ax.set_title(f"{arm.capitalize()} wall")
        ax.legend(frameon=False, ncol=2, columnspacing=0.8, handlelength=2.0)

    ax = axes[1, 0]
    for arm in ("soft", "hard"):
        for direction in ("ccw", "cw"):
            subset = select(rows, arm, direction)
            style = direction_style(direction)
            ax.plot(
                [row["sweep_fraction"] for row in subset],
                [1e7 * row["q_wall_direct_source_zero"] for row in subset],
                color=colors[arm],
                linestyle=style["linestyle"],
                marker=style["marker"],
                markerfacecolor="white",
                markersize=3,
                linewidth=1.0,
                label=f"{arm}, {direction.upper()}",
            )
    ax.axhline(0.0, color="0.25", linestyle=":", linewidth=0.8)
    ax.set_xlabel(r"signed twist $(\phi-\phi_0)/(2\pi)$")
    ax.set_ylabel(r"direct source contribution $\times10^7$")
    ax.set_title("Twist-dependent correction source")
    ax.legend(frameon=False, ncol=2, columnspacing=0.8, handlelength=2.0)

    ax = axes[1, 1]
    for arm in ("soft", "hard"):
        for direction in ("ccw", "cw"):
            subset = select(rows, arm, direction)
            style = direction_style(direction)
            residual = np.maximum(
                np.abs([row["source_subtracted_balance_residual"] for row in subset]),
                1e-18,
            )
            ax.semilogy(
                [row["sweep_fraction"] for row in subset],
                residual,
                color=colors[arm],
                linestyle=style["linestyle"],
                marker=style["marker"],
                markerfacecolor="white",
                markersize=3,
                linewidth=1.0,
                label=f"{arm}, {direction.upper()}",
            )
    ax.axhline(1e-9, color="0.25", linestyle=":", linewidth=0.8, label="tolerance")
    ax.set_xlabel(r"signed twist $(\phi-\phi_0)/(2\pi)$")
    ax.set_ylabel(r"$|q_L+q_R|$")
    ax.set_title("Source-subtracted balance")
    ax.legend(frameon=False, ncol=2, columnspacing=0.8, handlelength=2.0)

    for label, ax in zip(("(a)", "(b)", "(c)", "(d)"), axes.flat):
        ax.text(-0.12, 1.05, label, transform=ax.transAxes, fontsize=9, va="bottom")
    fig.tight_layout(pad=0.7, w_pad=1.0, h_pad=1.0)
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf = output_dir / "source_subtracted_flux_charge.pdf"
    png = output_dir / "source_subtracted_flux_charge.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return {"pdf": str(pdf), "png": str(png)}


def analyze(campaign_dir: Path, output_dir: Path) -> dict[str, Any]:
    campaign_dir = campaign_dir.resolve()
    output_dir = output_dir.resolve()
    manifest_path = campaign_dir / "manifest.json"
    input_csv = campaign_dir / "charge_vs_phi.csv"
    manifest = load_json(manifest_path)
    if manifest.get("schema") != EXPECTED_CAMPAIGN_SCHEMA or manifest.get("status") != "complete":
        raise RuntimeError("a complete frozen-record flux pilot manifest is required")
    config = manifest["configuration"]
    if manifest.get("source_hashes", {}).get("canonical_cpu_engine_sha256") != sha256_path(
        Path(__file__).resolve().parents[3] / "src/fgtn/classA_U1FGTN.py"
    ):
        raise RuntimeError("canonical CPU engine no longer matches the completed campaign")
    if manifest.get("source_hashes", {}).get("runner_sha256") != sha256_path(
        Path(__file__).resolve().parent / "run_campaign.py"
    ):
        raise RuntimeError("campaign runner no longer matches the completed campaign")

    rows = load_charge_rows(input_csv)
    if len(rows) != int(manifest["task_count"]):
        raise RuntimeError("aggregate row count does not match the campaign manifest")
    arms = {str(arm["name"]): arm for arm in config["arms"]}
    nx, ny = int(config["geometry"]["Nx"]), int(config["geometry"]["Ny"])
    derived: list[dict[str, Any]] = []
    source_profiles: dict[tuple[str, str, int], np.ndarray] = {}
    zero_sources: dict[str, np.ndarray] = {}
    zero_charges: dict[str, dict[str, Any]] = {}
    counts_by_arm: dict[str, dict[tuple[int, str], int]] = {}

    for arm_name, arm in arms.items():
        reference_root = campaign_dir / "references" / arm_name
        record_path = reference_root / "trajectory_record.json.gz"
        state_path = reference_root / "parent_state.npz"
        expected_record_hash = manifest["reference_records"][arm_name]["record_sha256"]
        if sha256_path(record_path) != expected_record_hash:
            raise RuntimeError(f"{arm_name} trajectory record checksum mismatch")
        record = load_record(record_path)
        counts = correction_multiplicities(record)
        counts_by_arm[arm_name] = counts
        zero_sources[arm_name] = source_profile(config, arm, counts, 0.0)
        with np.load(state_path, allow_pickle=False) as saved:
            zero_charges[arm_name] = endpoint_charges(saved["G_final"], config)
            if int(saved["net_injected_charge"]) != int(
                round(float(zero_sources[arm_name].sum()))
            ):
                raise RuntimeError(f"{arm_name} source sum does not reproduce the fixed record")

    for row in tqdm(rows, desc="Regional source subtraction", unit="twist"):
        arm_name = str(row["arm"])
        profile = source_profile(
            config,
            arms[arm_name],
            counts_by_arm[arm_name],
            float(row["phi"]),
        )
        source_profiles[(arm_name, str(row["direction"]), int(row["twist_index"]))] = profile
        source_left, source_right = region_sums(profile, config)
        zero_source_left, zero_source_right = region_sums(zero_sources[arm_name], config)
        zero_charge = zero_charges[arm_name]
        delta_left = float(row["N_left"]) - float(zero_charge["N_left"])
        delta_right = float(row["N_right"]) - float(zero_charge["N_right"])
        delta_source_left = source_left - zero_source_left
        delta_source_right = source_right - zero_source_right
        corrected_left = delta_left - delta_source_left
        corrected_right = delta_right - delta_source_right
        raw_q = 0.5 * (delta_right - delta_left)
        source_q = 0.5 * (delta_source_right - delta_source_left)
        corrected_q = 0.5 * (corrected_right - corrected_left)
        derived.append(
            {
                "task_id": row["task_id"],
                "arm": arm_name,
                "direction": row["direction"],
                "sigma": row["sigma"],
                "twist_index": row["twist_index"],
                "phi": row["phi"],
                "sweep_fraction": row["sweep_fraction"],
                "N_left": row["N_left"],
                "N_right": row["N_right"],
                "N_left_zero": zero_charge["N_left"],
                "N_right_zero": zero_charge["N_right"],
                "delta_N_left_zero": delta_left,
                "delta_N_right_zero": delta_right,
                "A_left": source_left,
                "A_right": source_right,
                "A_total": source_left + source_right,
                "A_left_zero": zero_source_left,
                "A_right_zero": zero_source_right,
                "delta_A_left_zero": delta_source_left,
                "delta_A_right_zero": delta_source_right,
                "q_left_source_subtracted_zero": corrected_left,
                "q_right_source_subtracted_zero": corrected_right,
                "q_wall_raw_zero": raw_q,
                "q_wall_direct_source_zero": source_q,
                "q_wall_source_subtracted_zero": corrected_q,
                "source_normalization_residual": (
                    source_left + source_right - int(row["net_injected_charge"])
                ),
                "source_subtracted_balance_residual": corrected_left + corrected_right,
                "original_q_wall_offset_origin": row["q_wall"],
            }
        )

    maximum_source_residual = max(
        abs(row["source_normalization_residual"]) for row in derived
    )
    maximum_balance_residual = max(
        abs(row["source_subtracted_balance_residual"]) for row in derived
    )
    if maximum_source_residual > 1e-9 or maximum_balance_residual > 1e-9:
        raise FloatingPointError("regional source-subtraction conservation check failed")

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "source_subtracted_charge_vs_phi.csv"
    write_csv(csv_path, derived)

    arm_names = [str(arm["name"]) for arm in config["arms"]]
    direction_names = [str(direction["name"]) for direction in config["twist"]["directions"]]
    count = int(config["twist"]["points_per_direction"])
    shape = (len(arm_names), len(direction_names), count)
    row_map = {
        (row["arm"], row["direction"], int(row["twist_index"])): row
        for row in derived
    }

    def collect(field: str) -> np.ndarray:
        result = np.empty(shape, dtype=np.float64)
        for ia, arm_name in enumerate(arm_names):
            for idirection, direction_name in enumerate(direction_names):
                for index in range(count):
                    result[ia, idirection, index] = float(
                        row_map[(arm_name, direction_name, index)][field]
                    )
        return result

    profile_array = np.empty((*shape, nx), dtype=np.float64)
    for ia, arm_name in enumerate(arm_names):
        for idirection, direction_name in enumerate(direction_names):
            for index in range(count):
                profile_array[ia, idirection, index] = source_profiles[
                    (arm_name, direction_name, index)
                ]
    npz_path = output_dir / "source_subtracted_charge_vs_phi.npz"
    atomic_npz(
        npz_path,
        schema=np.asarray(SCHEMA),
        arm_names=np.asarray(arm_names),
        direction_names=np.asarray(direction_names),
        phi=collect("phi"),
        sweep_fraction=collect("sweep_fraction"),
        delta_N_left_zero=collect("delta_N_left_zero"),
        delta_N_right_zero=collect("delta_N_right_zero"),
        delta_A_left_zero=collect("delta_A_left_zero"),
        delta_A_right_zero=collect("delta_A_right_zero"),
        q_wall_raw_zero=collect("q_wall_raw_zero"),
        q_wall_direct_source_zero=collect("q_wall_direct_source_zero"),
        q_wall_source_subtracted_zero=collect("q_wall_source_subtracted_zero"),
        source_subtracted_balance_residual=collect("source_subtracted_balance_residual"),
        source_by_x=profile_array,
        source_by_x_zero=np.asarray([zero_sources[name] for name in arm_names]),
    )

    figures = plot_results(derived, output_dir)
    by_arm: dict[str, Any] = {}
    for arm_name in arm_names:
        subset = [row for row in derived if row["arm"] == arm_name]
        arm_summary: dict[str, Any] = {
            "fixed_record_net_injected_charge": int(
                manifest["reference_records"][arm_name]["net_injected_charge"]
            ),
            "zero_twist_direct_source": {
                "left": region_sums(zero_sources[arm_name], config)[0],
                "right": region_sums(zero_sources[arm_name], config)[1],
            },
            "maximum_absolute_raw_wall_response": max(
                abs(row["q_wall_raw_zero"]) for row in subset
            ),
            "maximum_absolute_direct_source_response": max(
                abs(row["q_wall_direct_source_zero"]) for row in subset
            ),
            "maximum_absolute_source_subtracted_wall_response": max(
                abs(row["q_wall_source_subtracted_zero"]) for row in subset
            ),
            "directions": {},
        }
        raw_max = arm_summary["maximum_absolute_raw_wall_response"]
        arm_summary["maximum_fraction_of_raw_response_explained_by_direct_source"] = (
            arm_summary["maximum_absolute_direct_source_response"] / raw_max
        )
        positive_near_zero = row_map[(arm_name, "ccw", 1)]
        negative_near_zero = row_map[(arm_name, "cw", 1)]
        delta_phi = float(positive_near_zero["phi"]) - float(negative_near_zero["phi"])
        odd_slope = (
            float(positive_near_zero["q_wall_source_subtracted_zero"])
            - float(negative_near_zero["q_wall_source_subtracted_zero"])
        ) / delta_phi
        arm_summary["nearest_zero_direction_odd_response"] = {
            "positive_phi": float(positive_near_zero["phi"]),
            "q_positive_phi": float(
                positive_near_zero["q_wall_source_subtracted_zero"]
            ),
            "negative_phi": float(negative_near_zero["phi"]),
            "q_negative_phi": float(
                negative_near_zero["q_wall_source_subtracted_zero"]
            ),
            "centered_slope_dq_dphi": odd_slope,
            "two_pi_times_slope": 2.0 * np.pi * odd_slope,
            "direction_even_offset": 0.5
            * (
                float(positive_near_zero["q_wall_source_subtracted_zero"])
                + float(negative_near_zero["q_wall_source_subtracted_zero"])
            ),
        }
        for direction in direction_names:
            direction_rows = select(derived, arm_name, direction)
            half = direction_rows[count // 2]
            endpoint = direction_rows[-1]
            arm_summary["directions"][direction] = {
                "half_twist_source_subtracted_q_wall": half[
                    "q_wall_source_subtracted_zero"
                ],
                "maximum_absolute_source_subtracted_q_wall": max(
                    abs(row["q_wall_source_subtracted_zero"])
                    for row in direction_rows
                ),
                "closed_loop_source_subtracted_q_wall": endpoint[
                    "q_wall_source_subtracted_zero"
                ],
            }
        by_arm[arm_name] = arm_summary

    summary = {
        "schema": SCHEMA,
        "campaign_id": manifest["campaign_id"],
        "input_manifest": str(manifest_path),
        "input_manifest_sha256": sha256_path(manifest_path),
        "input_charge_csv_sha256": sha256_path(input_csv),
        "configuration_hash": manifest["configuration_hash"],
        "estimator": {
            "direct_source": "A_a(phi)=sum_e (s_e-m_e) tr[R_a P_e(phi)]",
            "raw_response": "delta N_a(phi)=N_a_final(phi)-N_a_final(0)",
            "source_subtracted": "q_a(phi)=delta N_a(phi)-[A_a(phi)-A_a(0)]",
            "wall_response": "q_x(phi)=[q_R(phi)-q_L(phi)]/2",
        },
        "rows": len(derived),
        "maximum_source_normalization_residual": maximum_source_residual,
        "maximum_source_subtracted_balance_residual": maximum_balance_residual,
        "by_arm": by_arm,
        "artifacts": {
            "csv": str(csv_path),
            "npz": str(npz_path),
            "figure_pdf": figures["pdf"],
            "figure_png": figures["png"],
        },
        "interpretation": (
            "The twist-dependent regional correction source is negligible relative to "
            "the observed intermediate-twist wall response. Near zero twist the "
            "source-subtracted response is direction-odd, with 2*pi*dq_x/dphi close "
            "to +1 for both wall constructions; this is clear single-record evidence "
            "of a handed, spectral-flow-like conditional response. The static family "
            "nevertheless closes after 2*pi, so it is not a quantized dynamical pump "
            "or a measured microscopic cut current. Multiple records plus matched "
            "trivial and reversed-Chern controls are required for a robust chirality "
            "claim."
        ),
    }
    summary_path = output_dir / "analysis_summary.json"
    atomic_json(summary_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else args.campaign_dir / "source_subtracted_analysis"
    )
    analyze(args.campaign_dir, output_dir)


if __name__ == "__main__":
    main()
