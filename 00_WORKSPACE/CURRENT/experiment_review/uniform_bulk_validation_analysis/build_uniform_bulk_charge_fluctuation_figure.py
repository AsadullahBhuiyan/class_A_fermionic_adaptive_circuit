#!/usr/bin/env python3
"""Build the S100 global-charge fluctuation figure for the uniform bulk campaign."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import build_uniform_bulk_chern_L12_L32 as base


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = base.OUTPUT_DIR
FIGURE_STEM = "uniform_bulk_global_charge_fluctuations_s100"
PDF_PATH = OUTPUT_DIR / f"{FIGURE_STEM}.pdf"
PNG_PATH = OUTPUT_DIR / f"{FIGURE_STEM}.png"
SOURCE_DATA_PATH = OUTPUT_DIR / f"{FIGURE_STEM}_source_data.csv"
SUMMARY_PATH = OUTPUT_DIR / f"{FIGURE_STEM}_summary.json"

LATE_TIME_START = 21
LATE_TIME_STOP = 40
FIXED_SIZE = 32
HISTOGRAM_SHELL = "1"
HISTOGRAM_CYCLE = 40

SHELL_STYLES = {
    "1": {
        "color": "#D55E4A",
        "marker": "^",
        "linestyle": (0, (1.0, 1.6)),
    },
    "2": {
        "color": "#3A9D5D",
        "marker": "s",
        "linestyle": (0, (5.0, 2.0)),
    },
    "dense": {
        "color": "#2878B5",
        "marker": "o",
        "linestyle": "-",
    },
}


def _shell_key(value: object) -> str:
    return "dense" if value is None else str(int(value))


def _mean_sem(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean and sample-to-sample SEM along the trajectory axis."""

    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or values.shape[0] != base.SAMPLE_COUNT:
        raise ValueError(f"expected shape ({base.SAMPLE_COUNT}, checkpoints), got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("nonfinite charge statistic")
    return (
        values.mean(axis=0),
        values.std(axis=0, ddof=1) / np.sqrt(float(base.SAMPLE_COUNT)),
    )


def _load_charge_cases() -> tuple[
    dict[tuple[int, str], np.ndarray], list[dict[str, object]]
]:
    """Load and checksum-verify every completed L=12--32 charge trajectory."""

    cases: dict[tuple[int, str], np.ndarray] = {}
    verification: list[dict[str, object]] = []

    for revision, specification in base.CAMPAIGNS.items():
        root = Path(specification["root"])
        sizes = tuple(int(value) for value in specification["sizes"])
        config_sha256 = str(specification["config_sha256"])
        if not (root / "results").is_dir():
            raise FileNotFoundError(root / "results")

        expected_source_identity: str | None = None
        for size in sizes:
            for shell in base.SHELLS:
                directory = root / "results" / f"L{size:02d}" / f"nsh-{shell}"
                receipts = sorted(directory.glob("*.complete.json"))
                if not receipts:
                    raise FileNotFoundError(f"no completion records under {directory}")

                parts: list[tuple[np.ndarray, np.ndarray]] = []
                for receipt_path in receipts:
                    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                    required_identity = {
                        "status": "complete",
                        "sampling_revision": revision,
                        "size": size,
                        "config_sha256": config_sha256,
                        "cycles_saved": base.SAVED_CYCLES.size,
                    }
                    for key, expected in required_identity.items():
                        if receipt.get(key) != expected:
                            raise ValueError(
                                f"{receipt_path}: completion identity mismatch for {key}"
                            )
                    if _shell_key(receipt.get("nshell")) != shell:
                        raise ValueError(f"{receipt_path}: shell identity mismatch")

                    source_identity = json.dumps(
                        receipt.get("source_hashes"), sort_keys=True
                    )
                    if expected_source_identity is None:
                        expected_source_identity = source_identity
                    elif source_identity != expected_source_identity:
                        raise ValueError(f"{revision}: inconsistent source hashes")

                    result_path = receipt_path.with_name(
                        str(receipt["result_filename"])
                    )
                    fingerprint = base._fingerprint(result_path)
                    if fingerprint != {
                        "bytes": int(receipt["result_bytes"]),
                        "sha256": str(receipt["result_sha256"]),
                    }:
                        raise ValueError(f"{result_path}: result checksum mismatch")

                    with np.load(result_path, allow_pickle=False) as archive:
                        required_arrays = {
                            "cycles",
                            "global_charge",
                            "particle_number",
                            "half_filling_offset",
                            "global_sample_indices",
                        }
                        missing = required_arrays.difference(archive.files)
                        if missing:
                            raise ValueError(
                                f"{result_path}: missing arrays {sorted(missing)}"
                            )
                        cycles = np.array(archive["cycles"], dtype=np.int64, copy=True)
                        charge = np.array(
                            archive["global_charge"], dtype=np.float64, copy=True
                        )
                        particle_number = np.array(
                            archive["particle_number"], dtype=np.int64, copy=True
                        )
                        offset = np.array(
                            archive["half_filling_offset"], dtype=np.int64, copy=True
                        )
                        indices = np.array(
                            archive["global_sample_indices"], dtype=np.int64, copy=True
                        )

                    count = int(receipt["samples_saved"])
                    expected_shape = (count, base.SAVED_CYCLES.size)
                    if not np.array_equal(cycles, base.SAVED_CYCLES):
                        raise ValueError(f"{result_path}: cycles are not exactly 0..40")
                    if charge.shape != expected_shape or not np.isfinite(charge).all():
                        raise ValueError(f"{result_path}: invalid global-charge array")
                    if particle_number.shape != expected_shape or offset.shape != expected_shape:
                        raise ValueError(f"{result_path}: invalid integer-charge arrays")
                    if indices.shape != (count,):
                        raise ValueError(f"{result_path}: invalid sample-index array")
                    if not np.allclose(charge, particle_number, rtol=0.0, atol=1.0e-8):
                        raise ValueError(f"{result_path}: charge/rank identity failed")
                    if not np.array_equal(offset, particle_number - size * size):
                        raise ValueError(f"{result_path}: half-filling offset mismatch")

                    parts.append((indices, offset))
                    verification.append(
                        {
                            "campaign": revision,
                            "completion": str(receipt_path.relative_to(root)),
                            "result": str(result_path.relative_to(root)),
                            "samples": count,
                            **fingerprint,
                        }
                    )

                parts.sort(key=lambda item: int(item[0][0]))
                indices = np.concatenate([item[0] for item in parts])
                offset = np.concatenate([item[1] for item in parts], axis=0)
                if not np.array_equal(indices, np.arange(base.SAMPLE_COUNT)):
                    raise ValueError(f"L={size}, shell={shell}: samples are not 0..99")
                if offset.shape != (base.SAMPLE_COUNT, base.SAVED_CYCLES.size):
                    raise ValueError(f"L={size}, shell={shell}: incomplete S100 case")
                if not np.all(offset[:, 0] == 0):
                    raise ValueError(f"L={size}, shell={shell}: initial state is not half filled")
                cases[(size, shell)] = offset

    expected_cases = {
        (size, shell) for size in base.SIZES for shell in base.SHELLS
    }
    if set(cases) != expected_cases:
        raise ValueError(f"case-grid mismatch: {sorted(cases)}")
    if len(verification) != 33:
        raise ValueError(f"expected 33 result/completion pairs, found {len(verification)}")
    total_trajectories = sum(values.shape[0] for values in cases.values())
    if total_trajectories != 1_800:
        raise ValueError(f"expected 1,800 trajectories, found {total_trajectories}")
    return cases, verification


def _summarize(
    cases: dict[tuple[int, str], np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cycle_mask = (base.SAVED_CYCLES >= LATE_TIME_START) & (
        base.SAVED_CYCLES <= LATE_TIME_STOP
    )
    if int(cycle_mask.sum()) != 20:
        raise ValueError("late-time window must contain cycles 21..40")

    size_rows: list[dict[str, object]] = []
    for shell in base.SHELLS:
        for size in base.SIZES:
            offset = cases[(size, shell)]
            trajectory_late_time = (
                100.0 * np.abs(offset[:, cycle_mask]) / float(size * size)
            ).mean(axis=1)
            mean, sem = _mean_sem(trajectory_late_time)
            size_rows.append(
                {
                    "panel": "a",
                    "metric": "late_time_mean_absolute_relative_half_filling_deviation_percent",
                    "L": size,
                    "n_shell": shell,
                    "cycle_start": LATE_TIME_START,
                    "cycle_stop": LATE_TIME_STOP,
                    "trajectories": base.SAMPLE_COUNT,
                    "mean": float(mean[0]),
                    "sem": float(sem[0]),
                }
            )

    cycle_rows: list[dict[str, object]] = []
    for shell in base.SHELLS:
        offset = cases[(FIXED_SIZE, shell)]
        absolute_filling_deviation = np.abs(offset) / float(2 * FIXED_SIZE**2)
        mean, sem = _mean_sem(absolute_filling_deviation)
        for index, cycle in enumerate(base.SAVED_CYCLES):
            cycle_rows.append(
                {
                    "panel": "b",
                    "metric": "mean_absolute_filling_fraction_deviation",
                    "L": FIXED_SIZE,
                    "n_shell": shell,
                    "cycle": int(cycle),
                    "trajectories": base.SAMPLE_COUNT,
                    "mean": float(mean[index]),
                    "sem": float(sem[index]),
                }
            )

    final_offsets = cases[(FIXED_SIZE, HISTOGRAM_SHELL)][:, HISTOGRAM_CYCLE]
    values, counts = np.unique(final_offsets, return_counts=True)
    histogram = pd.DataFrame(
        {
            "panel": "c",
            "metric": "integer_global_charge_offset_histogram",
            "L": FIXED_SIZE,
            "n_shell": HISTOGRAM_SHELL,
            "cycle": HISTOGRAM_CYCLE,
            "trajectories": base.SAMPLE_COUNT,
            "delta_Q": values.astype(np.int64),
            "count": counts.astype(np.int64),
        }
    )
    if int(histogram["count"].sum()) != base.SAMPLE_COUNT:
        raise ValueError("histogram does not contain exactly 100 trajectories")

    return pd.DataFrame(size_rows), pd.DataFrame(cycle_rows), histogram


def _legend_handles() -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            color=SHELL_STYLES[shell]["color"],
            linestyle=SHELL_STYLES[shell]["linestyle"],
            marker=SHELL_STYLES[shell]["marker"],
            markerfacecolor="white",
            markeredgewidth=0.8,
            linewidth=1.25,
            markersize=4.0,
            label=base.SHELL_LABELS[shell],
        )
        for shell in base.SHELLS
    ]


def _panel_label(axis: plt.Axes, letter: str) -> None:
    axis.text(
        -0.16,
        1.025,
        f"({letter})",
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
    )


def _draw(
    size_summary: pd.DataFrame,
    cycle_summary: pd.DataFrame,
    histogram: pd.DataFrame,
) -> None:
    base._configure_style()
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(3.375, 6.35),
        constrained_layout=False,
    )
    ax_size, ax_cycle, ax_histogram = axes

    for shell in base.SHELLS:
        style = SHELL_STYLES[shell]
        group = size_summary.loc[size_summary["n_shell"] == shell].sort_values("L")
        x = group["L"].to_numpy(dtype=np.float64)
        y = group["mean"].to_numpy(dtype=np.float64)
        yerr = group["sem"].to_numpy(dtype=np.float64)
        ax_size.errorbar(
            x,
            y,
            yerr=yerr,
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            markerfacecolor="white",
            markeredgewidth=0.8,
            linewidth=1.1,
            markersize=3.8,
            elinewidth=0.7,
            capsize=1.8,
            zorder=3,
        )

        group = cycle_summary.loc[
            cycle_summary["n_shell"] == shell
        ].sort_values("cycle")
        x = group["cycle"].to_numpy(dtype=np.float64)
        y = group["mean"].to_numpy(dtype=np.float64)
        sem = group["sem"].to_numpy(dtype=np.float64)
        ax_cycle.fill_between(
            x,
            np.maximum(y - sem, 0.0),
            y + sem,
            color=style["color"],
            alpha=0.10,
            linewidth=0.0,
            zorder=1,
        )
        ax_cycle.plot(
            x,
            y,
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            markevery=np.arange(base.SAVED_CYCLES.size),
            markerfacecolor="white",
            markeredgewidth=0.55,
            linewidth=0.9,
            markersize=2.1,
            zorder=3,
        )

    ax_size.set_xticks(base.SIZES)
    ax_size.set_xlim(10.7, 33.3)
    ax_size.set_ylim(bottom=-0.012)
    ax_size.set_xlabel(r"system size $L$")
    ax_size.set_ylabel(
        r"$100\,\langle|\Delta Q|\rangle/L^2$ (\%)"
    )
    ax_size.legend(
        handles=_legend_handles(),
        loc="upper right",
        ncol=1,
        frameon=False,
        handlelength=2.35,
        handletextpad=0.4,
        labelspacing=0.16,
        borderaxespad=0.25,
    )

    ax_cycle.set_xlim(-0.8, 40.8)
    ax_cycle.set_xticks((0, 10, 20, 30, 40))
    ax_cycle.set_ylim(bottom=-2.0e-5)
    ax_cycle.ticklabel_format(axis="y", style="sci", scilimits=(-3, -3))
    ax_cycle.set_xlabel("cycle")
    ax_cycle.set_ylabel(r"$\langle|f-1/2|\rangle$")
    ax_cycle.text(
        0.98,
        0.94,
        r"$L=32$",
        transform=ax_cycle.transAxes,
        ha="right",
        va="top",
        fontsize=8,
    )

    delta_q = histogram["delta_Q"].to_numpy(dtype=np.int64)
    counts = histogram["count"].to_numpy(dtype=np.int64)
    ax_histogram.bar(
        delta_q,
        counts,
        width=0.82,
        color=SHELL_STYLES[HISTOGRAM_SHELL]["color"],
        edgecolor="black",
        linewidth=0.65,
        alpha=0.82,
        zorder=2,
    )
    ax_histogram.axvline(0.0, color="black", linestyle="--", linewidth=0.75)
    ax_histogram.set_xticks(
        np.arange(int(delta_q.min()), int(delta_q.max()) + 1, dtype=np.int64)
    )
    ax_histogram.set_ylim(0.0, 1.16 * float(counts.max()))
    ax_histogram.set_xlabel(r"global charge offset $\Delta Q=Q-L^2$")
    ax_histogram.set_ylabel("trajectories")
    ax_histogram.text(
        0.98,
        0.94,
        r"$L=32$, $n_{\rm shell}=1$, cycle 40",
        transform=ax_histogram.transAxes,
        ha="right",
        va="top",
        fontsize=8,
    )

    for axis, letter in zip(axes, "abc"):
        _panel_label(axis, letter)
        for spine in axis.spines.values():
            spine.set_linewidth(0.8)

    fig.subplots_adjust(
        left=0.205,
        right=0.985,
        bottom=0.075,
        top=0.975,
        hspace=0.40,
    )
    fig.savefig(PDF_PATH)
    plt.close(fig)
    subprocess.run(
        [
            "pdftoppm",
            "-png",
            "-r",
            "300",
            "-singlefile",
            str(PDF_PATH),
            str(PNG_PATH.with_suffix("")),
        ],
        check=True,
    )


def main() -> int:
    cases, verification = _load_charge_cases()
    size_summary, cycle_summary, histogram = _summarize(cases)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    source_data = pd.concat(
        [size_summary, cycle_summary, histogram], ignore_index=True, sort=False
    )
    source_data.to_csv(SOURCE_DATA_PATH, index=False, float_format="%.17g")
    _draw(size_summary, cycle_summary, histogram)

    payload = {
        "schema": "uniform_bulk_global_charge_fluctuation_figure_v1",
        "campaigns": {
            revision: {
                "root": str(Path(specification["root"]).relative_to(base.REPOSITORY_ROOT)),
                "sizes": list(specification["sizes"]),
                "config_sha256": specification["config_sha256"],
            }
            for revision, specification in base.CAMPAIGNS.items()
        },
        "protocol": {
            "geometry": "uniform square lattice; Nx=Ny=L; DW=False",
            "initialization": "random pure at exact half filling",
            "perfect_correction": True,
            "measurement_order": "random serial",
            "dtype": "complex128",
            "sizes": list(base.SIZES),
            "shells": list(base.SHELLS),
            "cycles": list(map(int, base.SAVED_CYCLES)),
            "trajectories_per_case": base.SAMPLE_COUNT,
        },
        "definitions": {
            "delta_Q": "Q - L^2",
            "filling_fraction": "Q / (2 L^2)",
            "panel_a_trajectory_value": "mean over cycles 21..40 of 100 abs(delta_Q) / L^2 percent",
            "panel_b_trajectory_value": "abs(filling_fraction - 1/2) at each cycle",
            "panel_c_value": "integer delta_Q at cycle 40",
        },
        "uncertainty": {
            "independent_unit": "trajectory",
            "statistic": "sample standard deviation with ddof=1 divided by sqrt(100)",
            "panel_a_order": "average cycles 21..40 within each trajectory, then mean and SEM across trajectories",
            "panel_b_order": "mean and SEM across trajectories separately at each cycle",
        },
        "fixed_case": {
            "L": FIXED_SIZE,
            "histogram_n_shell": HISTOGRAM_SHELL,
            "histogram_cycle": HISTOGRAM_CYCLE,
        },
        "verification": {
            "result_completion_pairs": len(verification),
            "trajectories": sum(values.shape[0] for values in cases.values()),
            "charge_rank_identity_verified": True,
            "half_filling_offset_identity_verified": True,
            "initial_half_filling_verified": True,
            "inputs": verification,
        },
        "panel_a": size_summary.to_dict(orient="records"),
        "panel_c": histogram.to_dict(orient="records"),
        "outputs": {
            str(PDF_PATH.relative_to(PROJECT_DIR)): base._fingerprint(PDF_PATH),
            str(PNG_PATH.relative_to(PROJECT_DIR)): base._fingerprint(PNG_PATH),
            str(SOURCE_DATA_PATH.relative_to(PROJECT_DIR)): base._fingerprint(
                SOURCE_DATA_PATH
            ),
        },
    }
    SUMMARY_PATH.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "figure_pdf": str(PDF_PATH),
                "figure_png": str(PNG_PATH),
                "source_data": str(SOURCE_DATA_PATH),
                "summary": str(SUMMARY_PATH),
                "verified_pairs": len(verification),
                "verified_trajectories": sum(
                    values.shape[0] for values in cases.values()
                ),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
