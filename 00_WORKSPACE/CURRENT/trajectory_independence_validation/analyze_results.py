#!/usr/bin/env python3
"""Analyze and document one trajectory-independence campaign."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def save_npz_atomic(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def write_csv_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fields = sorted({key for row in rows for key in row}) if rows else []
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if fields:
            writer.writeheader()
            writer.writerows(rows)
    temporary.replace(path)


def _read_record(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def projector_distance(left: np.ndarray, right: np.ndarray) -> float:
    c_left = left @ left.conj().T
    c_right = right @ right.conj().T
    denominator = np.sqrt(max(left.shape[1] + right.shape[1], 1))
    return float(np.linalg.norm(c_left - c_right, ord="fro") / denominator)


def target_defects(target: np.ndarray, frame: np.ndarray) -> tuple[float, float, float]:
    common = float(np.linalg.norm(target.conj().T @ frame, ord="fro") ** 2)
    hole = max(float(target.shape[1]) - common, 0.0)
    excess = max(float(frame.shape[1]) - common, 0.0)
    distance = np.sqrt(max(hole + excess, 0.0) / max(target.shape[1] + frame.shape[1], 1))
    return float(distance), float(hole), float(excess)


def _measurement_word(record: dict[str, Any]) -> tuple[list[tuple[int, int, str]], np.ndarray]:
    identities = []
    outcomes = []
    for entry in record["entries"]:
        for event in entry["branch_events"]:
            if event.get("kind") != "measurement":
                continue
            identities.append((int(entry["cycle"]), int(entry["site_id"]), str(event["channel"])))
            outcomes.append(bool(event["outcome_occupied"]))
    return identities, np.asarray(outcomes, dtype=bool)


def _bootstrap_mean(values: np.ndarray, repetitions: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    samples = values.shape[0]
    indices = rng.integers(0, samples, size=(repetitions, samples))
    means = np.mean(values[indices], axis=1)
    return np.percentile(means, 2.5, axis=0), np.percentile(means, 97.5, axis=0)


def _bootstrap_pairwise_mean(distance: np.ndarray, repetitions: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    cycles, samples, _ = distance.shape
    draws = np.empty((repetitions, cycles), dtype=np.float64)
    upper = np.triu_indices(samples, 1)
    for draw in range(repetitions):
        selected = rng.integers(0, samples, size=samples)
        draws[draw] = np.mean(distance[:, selected[:, None], selected[None, :]][:, upper[0], upper[1]], axis=1)
    return np.percentile(draws, 2.5, axis=0), np.percentile(draws, 97.5, axis=0)


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["CMU Sans Serif", "DejaVu Sans"],
            "font.size": 8,
            "axes.linewidth": 0.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "legend.frameon": False,
        }
    )


def _save_figure(fig: plt.Figure, root: Path, name: str, dpi: int) -> None:
    (root / "figures").mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(root / f"figures/{name}.pdf", bbox_inches="tight")
    fig.savefig(root / f"figures/{name}.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def analyze(root: Path) -> dict[str, Any]:
    config = json.loads((root / "campaign_config.v1.json").read_text())
    g = config["geometry"]
    samples, cycles = int(g["samples"]), int(g["cycles"])
    cycle_coordinate = np.arange(cycles + 1)
    payloads = []
    handles = []
    summaries = []
    records = []
    for sample in range(samples):
        directory = root / f"raw/trajectories/sample_{sample:02d}"
        handle = np.load(directory / "cycle_data.npz")
        handles.append(handle)
        payloads.append(handle)
        summaries.append(json.loads((directory / "summary.json").read_text()))
        records.append(_read_record(directory / "record.json.gz"))

    declared = ("real_space_chern", "charge", "rank", "half_entropy", "gram_residual", "local_occupations", "cumulative_log_weight")
    arrays = {key: np.stack([np.asarray(payload[key]) for payload in payloads]) for key in declared}
    with np.load(root / "prepared/target_state.npz") as target_payload:
        target = np.array(target_payload["frame"], copy=True)
        target_chern = float(target_payload["real_space_chern"])

    pairwise = np.zeros((cycles + 1, samples, samples), dtype=np.float64)
    target_distance = np.empty((samples, cycles + 1), dtype=np.float64)
    hole_weight = np.empty_like(target_distance)
    excess_weight = np.empty_like(target_distance)
    minimum_principal_cosine = np.full((cycles + 1, samples, samples), np.nan)
    for cycle in range(cycles + 1):
        frames = [np.asarray(payload[f"frame_cycle_{cycle:03d}"]) for payload in payloads]
        projectors = [frame @ frame.conj().T for frame in frames]
        for sample, frame in enumerate(frames):
            target_distance[sample, cycle], hole_weight[sample, cycle], excess_weight[sample, cycle] = target_defects(target, frame)
        for left in range(samples):
            for right in range(left + 1, samples):
                denominator = np.sqrt(max(frames[left].shape[1] + frames[right].shape[1], 1))
                value = float(np.linalg.norm(projectors[left] - projectors[right], ord="fro") / denominator)
                pairwise[cycle, left, right] = pairwise[cycle, right, left] = value
                if frames[left].shape[1] == frames[right].shape[1]:
                    singular = np.linalg.svd(frames[left].conj().T @ frames[right], compute_uv=False)
                    minimum_principal_cosine[cycle, left, right] = minimum_principal_cosine[cycle, right, left] = float(np.min(singular))

    identities, first_word = _measurement_word(records[0])
    outcome_words = [first_word]
    for record in records[1:]:
        other_identities, word = _measurement_word(record)
        if other_identities != identities:
            raise RuntimeError("Outcome records do not share identical event identities.")
        outcome_words.append(word)
    outcome_hamming = np.zeros((samples, samples), dtype=np.float64)
    for left in range(samples):
        for right in range(left + 1, samples):
            value = float(np.mean(outcome_words[left] != outcome_words[right]))
            outcome_hamming[left, right] = outcome_hamming[right, left] = value

    upper = np.triu_indices(samples, 1)
    pairwise_mean = np.mean(pairwise[:, upper[0], upper[1]], axis=1)
    pairwise_maximum = np.max(pairwise[:, upper[0], upper[1]], axis=1)
    local_spread = np.max(arrays["local_occupations"], axis=0) - np.min(arrays["local_occupations"], axis=0)
    local_spread_maximum = np.max(local_spread, axis=1)

    rng = np.random.default_rng(int(config["analysis"]["bootstrap_seed"]))
    repetitions = int(config["analysis"]["bootstrap_repetitions"])
    chern_low, chern_high = _bootstrap_mean(arrays["real_space_chern"], repetitions, rng)
    entropy_low, entropy_high = _bootstrap_mean(arrays["half_entropy"], repetitions, rng)
    pair_low, pair_high = _bootstrap_pairwise_mean(pairwise, repetitions, rng)

    acceptance = config["acceptance"]
    ranks_final = arrays["rank"][:, -1]
    exact_gate = bool(
        np.all(ranks_final == ranks_final[0])
        and pairwise_maximum[-1] < float(acceptance["exact_projector_distance_hard"])
        and local_spread_maximum[-1] < float(acceptance["exact_local_occupation_hard"])
    )
    terminal = config["terminal_window"]
    start, stop = int(terminal["start_cycle"]), int(terminal["stop_cycle_inclusive"]) + 1
    final_error = np.abs(arrays["real_space_chern"][:, -1] - target_chern)
    window_error = np.abs(np.mean(arrays["real_space_chern"][:, start:stop], axis=1) - target_chern)
    initial_error = np.abs(arrays["real_space_chern"][:, 0] - target_chern)
    final_sd = float(np.std(arrays["real_space_chern"][:, -1], ddof=1))
    topology_gate = bool(
        np.all(final_error <= float(acceptance["chern_per_trajectory_hard"]))
        and np.all(window_error <= float(acceptance["chern_terminal_window_hard"]))
        and final_sd <= float(acceptance["chern_final_standard_deviation_hard"])
        and np.all(final_error < initial_error)
    )
    if exact_gate and topology_gate:
        classification = "exact_and_topological"
    elif topology_gate:
        classification = "topology_only_independent"
    elif exact_gate:
        classification = "state_independent_not_target_topological"
    else:
        classification = "trajectory_dependent_at_32_cycles"

    rows = []
    for sample in range(samples):
        rows.append(
            {
                "sample": sample,
                "outcome_seed": summaries[sample]["outcome_seed"],
                "outcome_digest": summaries[sample]["outcome_digest"],
                "final_rank": int(arrays["rank"][sample, -1]),
                "final_charge": float(arrays["charge"][sample, -1]),
                "final_chern": float(arrays["real_space_chern"][sample, -1]),
                "final_chern_error": float(final_error[sample]),
                "terminal_window_chern_error": float(window_error[sample]),
                "final_half_entropy": float(arrays["half_entropy"][sample, -1]),
                "final_target_distance": float(target_distance[sample, -1]),
                "final_hole_weight": float(hole_weight[sample, -1]),
                "final_excess_weight": float(excess_weight[sample, -1]),
                "maximum_gram_residual": float(np.max(arrays["gram_residual"][sample])),
            }
        )
    write_csv_atomic(root / "processed/tables/per_trajectory.csv", rows)
    save_npz_atomic(
        root / "processed/cycle_resolved_analysis.npz",
        cycles=cycle_coordinate,
        pairwise_projector_distance=pairwise,
        pairwise_projector_distance_mean=pairwise_mean,
        pairwise_projector_distance_maximum=pairwise_maximum,
        pairwise_projector_distance_bootstrap_low=pair_low,
        pairwise_projector_distance_bootstrap_high=pair_high,
        minimum_principal_cosine=minimum_principal_cosine,
        outcome_hamming=outcome_hamming,
        target_distance=target_distance,
        hole_weight=hole_weight,
        excess_weight=excess_weight,
        local_occupation_spread_maximum=local_spread_maximum,
        chern=arrays["real_space_chern"],
        chern_bootstrap_low=chern_low,
        chern_bootstrap_high=chern_high,
        charge=arrays["charge"],
        rank=arrays["rank"],
        half_entropy=arrays["half_entropy"],
        half_entropy_bootstrap_low=entropy_low,
        half_entropy_bootstrap_high=entropy_high,
    )

    summary = {
        "classification": classification,
        "exact_state_gate_passed": exact_gate,
        "topology_gate_passed": topology_gate,
        "target_chern_finite_size": target_chern,
        "final_chern_mean": float(np.mean(arrays["real_space_chern"][:, -1])),
        "final_chern_standard_deviation": final_sd,
        "maximum_final_chern_error": float(np.max(final_error)),
        "maximum_terminal_window_chern_error": float(np.max(window_error)),
        "final_rank_values": ranks_final.tolist(),
        "maximum_final_pairwise_projector_distance": float(pairwise_maximum[-1]),
        "mean_final_pairwise_projector_distance": float(pairwise_mean[-1]),
        "maximum_final_local_occupation_difference": float(local_spread_maximum[-1]),
        "mean_outcome_hamming_fraction": float(np.mean(outcome_hamming[upper])),
        "maximum_gram_residual": float(np.max(arrays["gram_residual"])),
        "sample_count": samples,
        "independent_unit": "one Born-outcome trajectory conditional on the fixed initial state and site schedule",
        "pairwise_distances_are_independent_samples": False,
        "bootstrap_repetitions": repetitions,
    }
    write_json_atomic(root / "processed/analysis_summary.json", summary)

    _style()
    width, dpi = float(config["analysis"]["figure_width_inches"]), int(config["analysis"]["dpi"])
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, samples))

    fig, ax = plt.subplots(figsize=(width, 3.2))
    for sample in range(samples):
        ax.plot(cycle_coordinate, arrays["real_space_chern"][sample], color=colors[sample], lw=0.8, alpha=0.7)
    mean_chern = np.mean(arrays["real_space_chern"], axis=0)
    ax.plot(cycle_coordinate, mean_chern, color="black", lw=1.6, label="trajectory mean")
    ax.fill_between(cycle_coordinate, chern_low, chern_high, color="black", alpha=0.15, label="bootstrap 95% interval")
    ax.axhline(target_chern, color="0.4", ls="--", lw=1.0, label="finite-size target")
    ax.set(xlabel="cycle", ylabel="real-space Chern marker")
    ax.legend(ncol=3, loc="lower right")
    _save_figure(fig, root, "chern_trajectories", dpi)

    fig, axes = plt.subplots(1, 2, figsize=(width, 3.0))
    for sample in range(samples):
        axes[0].plot(cycle_coordinate, arrays["rank"][sample], color=colors[sample], lw=0.8)
        axes[1].plot(cycle_coordinate, arrays["charge"][sample], color=colors[sample], lw=0.8)
    axes[0].set(xlabel="cycle", ylabel="frame rank")
    axes[1].set(xlabel="cycle", ylabel="physical charge")
    _save_figure(fig, root, "rank_and_charge", dpi)

    fig, ax = plt.subplots(figsize=(width, 3.2))
    ax.plot(cycle_coordinate, pairwise_mean, color="#2166ac", marker="o", ms=2.2, lw=1.2, label="mean pair distance")
    ax.fill_between(cycle_coordinate, pair_low, pair_high, color="#2166ac", alpha=0.18, label="trajectory-bootstrap 95% interval")
    ax.plot(cycle_coordinate, pairwise_maximum, color="#b2182b", ls="--", lw=1.2, label="maximum pair distance")
    ax.plot(cycle_coordinate, np.mean(target_distance, axis=0), color="black", ls=":", lw=1.2, label="mean distance to target")
    ax.set(xlabel="cycle", ylabel="normalized projector distance")
    ax.legend(ncol=2)
    _save_figure(fig, root, "projector_distance_convergence", dpi)

    fig, axes = plt.subplots(1, 2, figsize=(width, 3.1))
    im0 = axes[0].imshow(pairwise[-1], origin="lower", cmap="magma")
    axes[0].set(title="final projector distance", xlabel="trajectory", ylabel="trajectory")
    fig.colorbar(im0, ax=axes[0], fraction=0.046)
    im1 = axes[1].imshow(outcome_hamming, origin="lower", cmap="cividis", vmin=0.0, vmax=1.0)
    axes[1].set(title="outcome Hamming fraction", xlabel="trajectory", ylabel="trajectory")
    fig.colorbar(im1, ax=axes[1], fraction=0.046)
    _save_figure(fig, root, "pairwise_heatmaps", dpi)

    fig, axes = plt.subplots(1, 2, figsize=(width, 3.0))
    axes[0].plot(cycle_coordinate, np.mean(hole_weight, axis=0), color="#b2182b", lw=1.2, label="hole weight")
    axes[0].plot(cycle_coordinate, np.mean(excess_weight, axis=0), color="#2166ac", ls="--", lw=1.2, label="excess weight")
    axes[0].set(xlabel="cycle", ylabel="mean target defect weight")
    axes[0].legend()
    mean_entropy = np.mean(arrays["half_entropy"], axis=0)
    axes[1].plot(cycle_coordinate, mean_entropy, color="black", lw=1.2)
    axes[1].fill_between(cycle_coordinate, entropy_low, entropy_high, color="black", alpha=0.15)
    axes[1].set(xlabel="cycle", ylabel="half-system entropy")
    _save_figure(fig, root, "target_defects_and_entropy", dpi)

    for handle in handles:
        handle.close()
    return summary


def render_report(root: Path) -> Path:
    summary_path = root / "processed/analysis_summary.json"
    if not summary_path.exists():
        analyze(root)
    summary = json.loads(summary_path.read_text())
    audit = json.loads((root / "raw/audit/summary.json").read_text())
    config = json.loads((root / "campaign_config.v1.json").read_text())
    g = config["geometry"]
    report = root / "reports/trajectory_independence_validation.tex"
    report.parent.mkdir(parents=True, exist_ok=True)
    classification_tex = str(summary["classification"]).replace("_", r"\_")
    text = rf"""\documentclass[reprint,aps,prresearch]{{revtex4-2}}
\usepackage{{amsmath,amssymb,graphicx,booktabs}}
\begin{{document}}
\title{{Outcome-Trajectory Independence of Occupied-Frame Chern-State Preparation}}
\author{{Numerical validation campaign}}
\date{{\today}}
\maketitle
\section{{Protocol}}
Ten independent Born-outcome trajectories share one random-pure initial state and one frozen random site schedule. The controller uses $N_x=N_y={g['Nx']}$, {g['cycles']} cycles, $n_{{\rm shell}}=1$, domain wall off, $\alpha=1$, perfect correction, and no postselection. The independent sampling unit is one complete outcome trajectory conditional on the fixed initial state and schedule. Pairwise distances are not treated as independent samples.

Evolution is performed directly on the occupied frame. Physical projectors are reconstructed only after the trajectories finish, for diagnostic comparison. A covariance replay of trajectory zero gives maximum relative error {audit['maximum_relative_error']:.3e} and maximum element error {audit['maximum_element_error']:.3e}.

\section{{Classification}}
The campaign classification is \texttt{{{classification_tex}}}. The exact-state gate is {str(summary['exact_state_gate_passed']).lower()}, and the topology gate is {str(summary['topology_gate_passed']).lower()}. The finite-size target marker is {summary['target_chern_finite_size']:.9f}; the final ensemble mean is {summary['final_chern_mean']:.9f} with trajectory standard deviation {summary['final_chern_standard_deviation']:.3e}. The maximum final pairwise projector distance is {summary['maximum_final_pairwise_projector_distance']:.3e}, and the final ranks are {summary['final_rank_values']}.

\begin{{figure}}[t]
\includegraphics[width=\linewidth]{{../figures/chern_trajectories.pdf}}
\caption{{Individual and ensemble real-space Chern-marker convergence for $S=10$ outcome trajectories. The band is a whole-trajectory bootstrap 95\% interval with 10,000 resamples.}}
\end{{figure}}

\begin{{figure}}[t]
\includegraphics[width=\linewidth]{{../figures/projector_distance_convergence.pdf}}
\caption{{Mean and maximum pairwise occupied-projector distance. The 45 pairs are dependent; uncertainty is obtained by resampling the ten trajectories.}}
\end{{figure}}

\begin{{figure}}[t]
\includegraphics[width=\linewidth]{{../figures/pairwise_heatmaps.pdf}}
\caption{{Final microscopic state separation compared with measurement-record separation.}}
\end{{figure}}

\section{{Interpretation}}
Exact final-state independence would permit independently realized records to represent the same state at different boundary twists. Topology-only independence establishes a common Chern phase but does not make unrelated trajectories into one smooth Berry bundle. This experiment is conditional on one frozen site schedule and does not test schedule-to-schedule variation.
\end{{document}}
"""
    report.write_text(text)
    if shutil.which("pdflatex"):
        for _ in range(2):
            subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", report.name],
                cwd=report.parent,
                text=True,
                capture_output=True,
                check=False,
            )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("analyze", "report"))
    parser.add_argument("--campaign-root", required=True, type=Path)
    args = parser.parse_args()
    if args.mode == "analyze":
        summary = analyze(args.campaign_root)
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(render_report(args.campaign_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
