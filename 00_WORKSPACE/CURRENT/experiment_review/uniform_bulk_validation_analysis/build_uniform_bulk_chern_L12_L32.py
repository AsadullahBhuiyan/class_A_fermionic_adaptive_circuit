#!/usr/bin/env python3
"""Plot every-cycle trajectory-averaged bulk Chern data for L=12,...,32."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


PROJECT_DIR = Path(__file__).resolve().parent


def _repository_root() -> Path:
    for candidate in (PROJECT_DIR, *PROJECT_DIR.parents):
        if (candidate / "PROJECT_ADMIN" / "REPO_POLICY.md").is_file():
            return candidate
    raise RuntimeError("could not locate repository root")


REPOSITORY_ROOT = _repository_root()
RESULTS_PARENT = (
    REPOSITORY_ROOT
    / "00_WORKSPACE"
    / "LARGE_RESULTS"
    / "classA_final_production_outputs"
)
SMALL_REVISION = "uniform_perfect_correction_40cycle_s100_maxl24_batched_v1"
LARGE_REVISION = "uniform_perfect_correction_40cycle_s100_l28_l40_batched_v2"
CAMPAIGNS = {
    SMALL_REVISION: {
        "root": RESULTS_PARENT / SMALL_REVISION,
        "sizes": (12, 16, 20, 24),
        "config_sha256": "5f8ced37fc3bfa5ada24c7f3ff744ae14773cf347d98260551730130c6713b11",
    },
    LARGE_REVISION: {
        "root": RESULTS_PARENT / LARGE_REVISION,
        "sizes": (28, 32),
        "config_sha256": "5d76508a76275cc95ad499ff28f894700e67725dece500fca0ea2b02f79f13c8",
    },
}

SIZES = (12, 16, 20, 24, 28, 32)
SHELLS = ("1", "2", "dense")
SHELL_LABELS = {
    "1": r"$n_{\rm shell}=1$",
    "2": r"$n_{\rm shell}=2$",
    "dense": r"$n_{\rm shell}=\infty$",
}
SAVED_CYCLES = np.arange(41, dtype=np.int64)
SAMPLE_COUNT = 100
BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_SEED = 2026090701

OUTPUT_DIR = PROJECT_DIR / "outputs" / "uniform_perfect_correction_40cycle_s100_L12-L32"
FIGURE_STEM = "fig02_uniform_bulk_chern_L12_L32_s100"
PDF_PATH = OUTPUT_DIR / f"{FIGURE_STEM}.pdf"
PNG_PATH = OUTPUT_DIR / f"{FIGURE_STEM}.png"
SOURCE_DATA_PATH = OUTPUT_DIR / f"{FIGURE_STEM}_source_data.csv"
SUMMARY_PATH = OUTPUT_DIR / f"{FIGURE_STEM}_summary.json"

SIZE_COLORS = {
    12: "#0072B2",
    16: "#E69F00",
    20: "#009E73",
    24: "#D55E00",
    28: "#CC79A7",
    32: "#6B4C9A",
}
SIZE_MARKERS = {12: "o", 16: "s", 20: "^", 24: "D", 28: "v", 32: "P"}
SIZE_LINESTYLES: dict[int, Any] = {
    12: ":",
    16: "--",
    20: "-.",
    24: "-",
    28: (0, (5, 1.3)),
    32: (0, (3, 1, 1, 1)),
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fingerprint(path: Path) -> dict[str, int | str]:
    return {"bytes": path.stat().st_size, "sha256": _sha256_file(path)}


def _shell_from_receipt(value: object) -> str:
    return "dense" if value is None else str(int(value))


def _load_cases() -> tuple[dict[tuple[int, str], np.ndarray], list[dict[str, object]]]:
    cases: dict[tuple[int, str], np.ndarray] = {}
    verification: list[dict[str, object]] = []

    for revision, specification in CAMPAIGNS.items():
        root = Path(specification["root"])
        sizes = tuple(int(value) for value in specification["sizes"])
        config_sha256 = str(specification["config_sha256"])
        if not (root / "results").is_dir():
            raise FileNotFoundError(root / "results")

        expected_source_identity: str | None = None
        for size in sizes:
            for shell in SHELLS:
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
                        "cycles_saved": SAVED_CYCLES.size,
                    }
                    for key, expected in required_identity.items():
                        if receipt.get(key) != expected:
                            raise ValueError(
                                f"{receipt_path}: completion identity mismatch for {key}"
                            )
                    if _shell_from_receipt(receipt.get("nshell")) != shell:
                        raise ValueError(f"{receipt_path}: shell identity mismatch")

                    source_identity = json.dumps(
                        receipt.get("source_hashes"), sort_keys=True
                    )
                    if expected_source_identity is None:
                        expected_source_identity = source_identity
                    elif source_identity != expected_source_identity:
                        raise ValueError(f"{revision}: inconsistent source hashes")

                    result_path = receipt_path.with_name(str(receipt["result_filename"]))
                    fingerprint = _fingerprint(result_path)
                    if fingerprint != {
                        "bytes": int(receipt["result_bytes"]),
                        "sha256": str(receipt["result_sha256"]),
                    }:
                        raise ValueError(f"{result_path}: result checksum mismatch")

                    with np.load(result_path, allow_pickle=False) as archive:
                        cycles = np.array(archive["cycles"], dtype=np.int64, copy=True)
                        chern = np.array(
                            archive["real_space_chern"], dtype=np.float64, copy=True
                        )
                        charge = np.array(
                            archive["global_charge"], dtype=np.float64, copy=True
                        )
                        particle_number = np.array(
                            archive["particle_number"], dtype=np.int64, copy=True
                        )
                        indices = np.array(
                            archive["global_sample_indices"], dtype=np.int64, copy=True
                        )

                    count = int(receipt["samples_saved"])
                    if not np.array_equal(cycles, SAVED_CYCLES):
                        raise ValueError(f"{result_path}: cycles are not exactly 0..40")
                    if chern.shape != (count, SAVED_CYCLES.size):
                        raise ValueError(f"{result_path}: invalid Chern shape {chern.shape}")
                    if not np.isfinite(chern).all():
                        raise ValueError(f"{result_path}: nonfinite Chern data")
                    if charge.shape != chern.shape or particle_number.shape != chern.shape:
                        raise ValueError(f"{result_path}: invalid charge shape")
                    if not np.allclose(charge, particle_number, rtol=0.0, atol=1.0e-8):
                        raise ValueError(f"{result_path}: charge/rank identity failed")
                    if indices.shape != (count,):
                        raise ValueError(f"{result_path}: invalid sample-index shape")

                    parts.append((indices, chern))
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
                chern = np.concatenate([item[1] for item in parts], axis=0)
                if not np.array_equal(indices, np.arange(SAMPLE_COUNT)):
                    raise ValueError(f"L={size}, shell={shell}: samples are not 0..99")
                if chern.shape != (SAMPLE_COUNT, SAVED_CYCLES.size):
                    raise ValueError(f"L={size}, shell={shell}: incomplete S100 case")
                cases[(size, shell)] = chern

    expected_cases = {(size, shell) for size in SIZES for shell in SHELLS}
    if set(cases) != expected_cases:
        raise ValueError(f"case-grid mismatch: {sorted(cases)}")
    return cases, verification


def _bootstrap_summary(values: np.ndarray, label: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    digest = hashlib.sha256(f"{BOOTSTRAP_SEED}:{label}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    weights = rng.multinomial(
        SAMPLE_COUNT,
        np.full(SAMPLE_COUNT, 1.0 / SAMPLE_COUNT),
        size=BOOTSTRAP_REPLICATES,
    )
    replicates = (weights @ values) / SAMPLE_COUNT
    low, high = np.quantile(replicates, [0.025, 0.975], axis=0)
    return values.mean(axis=0), low, high


def _summarize(cases: dict[tuple[int, str], np.ndarray]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for shell in SHELLS:
        for size in SIZES:
            mean, low, high = _bootstrap_summary(
                cases[(size, shell)], f"chern:L{size}:shell={shell}"
            )
            for index, cycle in enumerate(SAVED_CYCLES):
                rows.append(
                    {
                        "L": size,
                        "n_shell": shell,
                        "cycle": int(cycle),
                        "trajectories": SAMPLE_COUNT,
                        "mean_real_space_chern": float(mean[index]),
                        "ci95_low": float(low[index]),
                        "ci95_high": float(high[index]),
                    }
                )
    return pd.DataFrame(rows)


def _configure_style() -> None:
    os.environ["TEXINPUTS"] = (
        str(PROJECT_DIR / "latex_support")
        + os.pathsep
        + os.environ.get("TEXINPUTS", "")
    )
    mpl.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.family": "sans-serif",
            "font.sans-serif": ["CMU Sans Serif"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "text.usetex": True,
            "text.latex.preamble": r"\usepackage{amsmath}\renewcommand{\familydefault}{\sfdefault}",
            "axes.linewidth": 0.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 3.2,
            "ytick.major.size": 3.2,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _draw(summary: pd.DataFrame) -> None:
    _configure_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(7.05, 2.55), sharex=True, sharey=True)

    y_low = float(summary["ci95_low"].min())
    y_high = float(summary["ci95_high"].max())
    padding = 0.04 * (y_high - y_low)
    marker_cycles = np.arange(0, 41, 5)

    for axis, shell, letter in zip(axes, SHELLS, "abc"):
        for size in SIZES:
            group = summary.loc[
                (summary["n_shell"] == shell) & (summary["L"] == size)
            ].sort_values("cycle")
            x = group["cycle"].to_numpy(dtype=np.float64)
            mean = group["mean_real_space_chern"].to_numpy(dtype=np.float64)
            low = group["ci95_low"].to_numpy(dtype=np.float64)
            high = group["ci95_high"].to_numpy(dtype=np.float64)
            color = SIZE_COLORS[size]
            axis.fill_between(x, low, high, color=color, alpha=0.075, linewidth=0)
            axis.plot(
                x,
                mean,
                color=color,
                linestyle=SIZE_LINESTYLES[size],
                linewidth=1.0,
                marker=SIZE_MARKERS[size],
                markevery=marker_cycles,
                markersize=2.7,
                markerfacecolor="white",
                markeredgewidth=0.6,
                zorder=3,
            )
        axis.axhline(1.0, color="black", linestyle="--", linewidth=0.75, zorder=1)
        axis.set_xlim(-0.8, 40.8)
        axis.set_ylim(y_low - padding, y_high + padding)
        axis.set_xticks((0, 10, 20, 30, 40))
        axis.set_xlabel("cycle")
        axis.set_title(SHELL_LABELS[shell], pad=3)
        axis.text(
            -0.16,
            1.035,
            f"({letter})",
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=9,
        )
    axes[0].set_ylabel(r"trajectory mean $\overline{C_G}$")

    handles = [
        Line2D(
            [0],
            [0],
            color=SIZE_COLORS[size],
            linestyle=SIZE_LINESTYLES[size],
            marker=SIZE_MARKERS[size],
            markerfacecolor="white",
            linewidth=1.0,
            markersize=3.2,
            label=rf"$L={size}$",
        )
        for size in SIZES
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=6,
        frameon=False,
        handlelength=1.8,
        columnspacing=0.95,
    )
    fig.subplots_adjust(left=0.083, right=0.995, bottom=0.19, top=0.82, wspace=0.08)
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
    cases, verification = _load_cases()
    summary = _summarize(cases)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SOURCE_DATA_PATH, index=False, float_format="%.17g")
    _draw(summary)

    final_rows = summary.loc[summary["cycle"] == 40].to_dict(orient="records")
    caption = (
        r"Trajectory-averaged real-space Chern number for uniform perfect-correction "
        r"dynamics without domain walls. Panels separate $n_{\rm shell}=1$, 2, and "
        r"the dense OW construction; colors, markers, and line styles denote "
        r"$L=12,16,20,24,28,32$. Each curve averages $S=100$ independent Born "
        r"trajectories initialized as random pure half-filled states and evolved for "
        r"40 random-serial perfect-correction cycles in complex128. The nonlinear "
        r"$C_G$ estimator is evaluated trajectory by trajectory before averaging. "
        r"Shaded bands are deterministic 95\% whole-trajectory bootstrap intervals "
        r"from 20,000 resamples; the dashed line marks $C_G=1$."
    )
    payload = {
        "schema": "uniform_bulk_chern_L12_L32_summary_v1",
        "campaigns": list(CAMPAIGNS),
        "sizes": list(SIZES),
        "shells": list(SHELLS),
        "cycles": list(map(int, SAVED_CYCLES)),
        "trajectories_per_case": SAMPLE_COUNT,
        "total_trajectories": len(SIZES) * len(SHELLS) * SAMPLE_COUNT,
        "estimator_order": "trajectory-resolved C_G, then trajectory average",
        "uncertainty": {
            "unit": "whole trajectory",
            "interval": "equal-tailed 95 percent bootstrap",
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
        },
        "verification": {
            "result_completion_pairs": len(verification),
            "all_result_sizes_and_sha256_verified": True,
            "all_selected_cases_are_S100": True,
            "cycles_are_exactly_0_through_40": True,
            "charge_rank_identity_verified": True,
            "inputs": verification,
        },
        "final_cycle": final_rows,
        "caption": caption,
        "outputs": {
            str(PDF_PATH.relative_to(PROJECT_DIR)): _fingerprint(PDF_PATH),
            str(PNG_PATH.relative_to(PROJECT_DIR)): _fingerprint(PNG_PATH),
            str(SOURCE_DATA_PATH.relative_to(PROJECT_DIR)): _fingerprint(
                SOURCE_DATA_PATH
            ),
        },
    }
    SUMMARY_PATH.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"figure": str(PNG_PATH), "summary": str(SUMMARY_PATH)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
