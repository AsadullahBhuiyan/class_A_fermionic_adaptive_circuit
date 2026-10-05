#!/usr/bin/env python3
"""Build the immutable data snapshot and figures for the flux-threading note.

The wall-diabatic production queue may still be active.  On its first run this
script freezes the set of already-published completion JSON files, verifies
the byte count and SHA-256 of every referenced NPZ, and records the inventory
in ``docs/flux_threading_note_assets/snapshot_manifest.json``.  Later runs use
that manifest and therefore cannot silently absorb newly completed tasks.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT = Path(__file__).resolve().parent
REPOSITORY = PROJECT.parents[2]
RESULT_ROOT = PROJECT / "results" / "N20_24_28_32x24_wall_diabatic_spectral_pump_s100_v1"
ASSET_ROOT = PROJECT / "docs" / "flux_threading_note_assets"
FIGURE_ROOT = ASSET_ROOT / "figures"
MANIFEST = ASSET_ROOT / "snapshot_manifest.json"
OLD_CAMPAIGNS = {
    "nsh1_N20x24": PROJECT / "results" / "N20x24_state_projector_pump_s100_v1" / "analysis" / "state_projector_pump_endpoints.csv",
    "dense_N20x24": PROJECT / "results" / "N20x24_state_projector_pump_s100_dense_v1" / "analysis" / "state_projector_pump_endpoints.csv",
    "nsh1_N24x24": PROJECT / "results" / "N24x24_state_projector_pump_s100_v1" / "analysis" / "state_projector_pump_endpoints.csv",
}

WIDTH = 7.05
COLORS = {20: "#0072B2", 24: "#D55E00", 28: "#009E73", 32: "#CC79A7"}
MARKERS = {20: "o", 24: "s", 28: "^", 32: "D"}


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPOSITORY.resolve()))


def result_for_completion(completion: Path, payload: dict[str, Any]) -> Path:
    return completion.with_name(str(payload["result"]["name"]))


def verify_pair(completion: Path) -> dict[str, Any]:
    payload = json.loads(completion.read_text(encoding="utf-8"))
    result = result_for_completion(completion, payload)
    if not result.is_file():
        raise RuntimeError(f"missing result for {completion}: {result}")
    expected = payload["result"]
    actual_bytes = result.stat().st_size
    if actual_bytes != int(expected["bytes"]):
        raise RuntimeError(f"byte-count mismatch for {result}")
    actual_sha = sha256_path(result)
    if actual_sha != str(expected["sha256"]):
        raise RuntimeError(f"SHA-256 mismatch for {result}")
    return {
        "completion": relative(completion),
        "completion_sha256": sha256_path(completion),
        "result": relative(result),
        "result_bytes": actual_bytes,
        "result_sha256": actual_sha,
        "task_id": payload.get("task_id", completion.stem),
    }


def freeze_or_load_manifest() -> dict[str, Any]:
    ASSET_ROOT.mkdir(parents=True, exist_ok=True)
    if MANIFEST.is_file():
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    else:
        pump_completions = sorted((RESULT_ROOT / "pump").rglob("sample_*.completion.json"))
        control_completions = sorted((RESULT_ROOT / "controls").rglob("sample_*.completion.json"))
        old_sources = []
        for cell, path in OLD_CAMPAIGNS.items():
            if not path.is_file():
                raise RuntimeError(f"missing legacy comparison table: {path}")
            old_sources.append(
                {
                    "cell": cell,
                    "path": relative(path),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_path(path),
                }
            )
        manifest = {
            "schema": "flux_threading_note_snapshot_manifest_v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "campaign_complete_at_snapshot": False,
            "production_campaign": "N20_24_28_32x24_wall_diabatic_spectral_pump_s100_v1",
            "pump_pairs": [verify_pair(path) for path in pump_completions],
            "control_pairs": [verify_pair(path) for path in control_completions],
            "legacy_endpoint_tables": old_sources,
        }
        MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    for item in manifest["pump_pairs"] + manifest["control_pairs"]:
        completion = REPOSITORY / item["completion"]
        result = REPOSITORY / item["result"]
        if sha256_path(completion) != item["completion_sha256"]:
            raise RuntimeError(f"frozen completion changed: {completion}")
        if result.stat().st_size != int(item["result_bytes"]) or sha256_path(result) != item["result_sha256"]:
            raise RuntimeError(f"frozen result changed: {result}")
    for item in manifest["legacy_endpoint_tables"]:
        path = REPOSITORY / item["path"]
        if path.stat().st_size != int(item["bytes"]) or sha256_path(path) != item["sha256"]:
            raise RuntimeError(f"frozen legacy table changed: {path}")
    return manifest


def scalar_text(value: np.ndarray) -> str:
    raw = np.asarray(value).item()
    return raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)


def load_pump_rows(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in manifest["pump_pairs"]:
        completion = json.loads((REPOSITORY / item["completion"]).read_text(encoding="utf-8"))
        with np.load(REPOSITORY / item["result"], allow_pickle=False) as data:
            directions = [scalar_text(value) for value in data["directions"]]
            for direction_index, direction in enumerate(directions):
                sigma = int(data["sigma"][direction_index])
                endpoint = float(data["q_x"][direction_index, -1])
                resolved = bool(data["resolved"][direction_index])
                gap_index = int(data["edge_minimum_gap_index"][direction_index])
                rows.append(
                    {
                        "task_id": str(completion["task_id"]),
                        "protocol": str(completion["protocol"]),
                        "Nx": int(completion["Nx"]),
                        "Ny": int(completion["Ny"]),
                        "wall": str(completion["wall"]),
                        "sample_id": int(completion["sample_id"]),
                        "direction": direction,
                        "sigma": sigma,
                        "phi": np.array(data["phi"][direction_index], dtype=float),
                        "q_x": np.array(data["q_x"][direction_index], dtype=float),
                        "delta_N_left": np.array(data["delta_N_left"][direction_index], dtype=float),
                        "delta_N_right": np.array(data["delta_N_right"][direction_index], dtype=float),
                        "endpoint_q_x": endpoint,
                        "signed_endpoint_q_x": sigma * endpoint,
                        "resolved": resolved,
                        "unresolved_reason": scalar_text(data["unresolved_reason"][direction_index]),
                        "source_chern": float(data["source_real_space_chern_mean"]),
                        "crossing_internal_gap": float(data["edge_internal_gap"][direction_index, gap_index]),
                        "crossing_external_gap": float(data["edge_external_gap"][direction_index, gap_index]),
                        "crossing_link": float(data["edge_link_min_singular"][direction_index, gap_index]),
                        "crossing_wall_weight": float(np.min(data["edge_combined_wall_weight"][direction_index, gap_index])),
                        "crossing_wall_polarization": float(
                            min(
                                -np.min(data["edge_B_eigenvalues"][direction_index, gap_index]),
                                np.max(data["edge_B_eigenvalues"][direction_index, gap_index]),
                            )
                        ),
                        "endpoint_instantaneous_q_x": float(data["instantaneous_q_x"][direction_index, -1]),
                        "charge_residual": float(np.max(np.abs(data["total_charge_residual"][direction_index]))),
                        "projector_residual": float(np.max(np.abs(data["projector_residual"][direction_index]))),
                        "undo_error": float(data["continuation_undo_error"][direction_index]),
                    }
                )
    return rows


def load_controls(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in manifest["control_pairs"]:
        completion = json.loads((REPOSITORY / item["completion"]).read_text(encoding="utf-8"))
        with np.load(REPOSITORY / item["result"], allow_pickle=False) as data:
            directions = [scalar_text(value) for value in data["directions"]]
            for index, direction in enumerate(directions):
                rows.append(
                    {
                        "variant": str(completion.get("variant", (REPOSITORY / item["result"]).parts[-5])),
                        "source_kind": str(completion.get("source_kind", completion.get("protocol", "unknown"))),
                        "wall": str(completion["wall"]),
                        "direction": direction,
                        "phi": np.array(data["phi"][index], dtype=float),
                        "q_x": np.array(data["q_x"][index], dtype=float),
                        "instantaneous_q_x": np.array(data["instantaneous_q_x"][index], dtype=float),
                        "endpoint_q_x": float(data["q_x"][index, -1]),
                        "endpoint_instantaneous_q_x": float(data["instantaneous_q_x"][index, -1]),
                        "source_chern": float(data["source_real_space_chern_mean"]),
                        "charge_residual": float(np.max(np.abs(data["total_charge_residual"][index]))),
                        "projector_residual": float(np.max(np.abs(data["projector_residual"][index]))),
                        "undo_error": float(data["continuation_undo_error"][index]),
                    }
                )
    return rows


def load_old_rows(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in manifest["legacy_endpoint_tables"]:
        with (REPOSITORY / item["path"]).open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                for direction, column, sigma in (("ccw", "ccw_q_x", 1), ("cw", "cw_q_x", -1)):
                    rows.append(
                        {
                            "cell": item["cell"],
                            "wall": row["wall"],
                            "sample_id": int(row["sample_id"]),
                            "direction": direction,
                            "sigma": sigma,
                            "endpoint_q_x": float(row[column]),
                            "signed_endpoint_q_x": sigma * float(row[column]),
                        }
                    )
    return rows


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["CMU Sans Serif", "Computer Modern Sans Serif", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "legend.fontsize": 6.5,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
        }
    )


def save(fig: plt.Figure, stem: str) -> None:
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_ROOT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIGURE_ROOT / f"{stem}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def panel_labels(axes: Iterable[plt.Axes]) -> None:
    for label, axis in zip("abcdefghijklmnopqrstuvwxyz", axes):
        axis.text(-0.16, 1.04, f"({label})", transform=axis.transAxes, fontweight="bold", va="bottom")


def figure_exact_controls(controls: list[dict[str, Any]]) -> None:
    topological = [
        row for row in controls
        if "topological_M256" in row["variant"] and "seam" not in row["variant"] and row["wall"] == "soft"
    ]
    if not topological:
        # Completion metadata from older controls may not include ``variant``.
        topological = [row for row in controls if row["source_chern"] > 0.9 and row["wall"] == "soft" and row["phi"].size == 257]
        topological = topological[:2]
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.55))
    ax = axes[0]
    direction_style = {"ccw": ("-", "o"), "cw": ("--", "s")}
    for row in topological:
        ls, marker = direction_style[row["direction"]]
        x = np.abs(row["phi"] - row["phi"][0]) / (2 * np.pi)
        ax.plot(x, row["q_x"], ls, color="#0072B2", lw=1.25, marker=marker, markevery=32,
                ms=3, label=f"continued {row['direction'].upper()}")
        ax.plot(x, row["instantaneous_q_x"], ls, color="#D55E00", lw=0.9, alpha=0.8,
                label=f"instantaneous {row['direction'].upper()}")
    ax.axhline(0, color="0.35", ls=":", lw=0.7)
    ax.axhline(1, color="0.65", ls="--", lw=0.6)
    ax.axhline(-1, color="0.65", ls="--", lw=0.6)
    ax.set_xlabel(r"threaded flux $|\phi-\phi_0|/(2\pi)$")
    ax.set_ylabel(r"wall transfer $q_x$")
    ax.set_title(r"exact topological slab, soft wall")
    ax.legend(frameon=False, ncol=2, columnspacing=0.8, handlelength=2.2)

    ax = axes[1]
    groups = [
        ("topological", 1, "#0072B2"),
        ("conjugated", -1, "#D55E00"),
        ("trivial", 0, "#009E73"),
    ]
    x_positions, labels = [], []
    for group_index, (kind, _expected, color) in enumerate(groups):
        selected = [row for row in controls if abs(row["source_chern"] - _expected) < 0.1 and row["phi"].size == 257]
        for wall_index, wall in enumerate(("soft", "hard")):
            for direction_index, direction in enumerate(("ccw", "cw")):
                candidates = [row for row in selected if row["wall"] == wall and row["direction"] == direction]
                if not candidates:
                    continue
                x = group_index * 3.0 + wall_index * 1.15 + (direction_index - 0.5) * 0.24
                marker = "o" if direction == "ccw" else "s"
                ax.scatter(x, candidates[0]["endpoint_q_x"], color=color, marker=marker, s=24,
                           facecolors=color if direction == "ccw" else "none", zorder=3)
            x_positions.append(group_index * 3.0 + wall_index * 1.15)
            labels.append(f"{kind[:4]}.\n{wall}")
    ax.axhline(0, color="0.35", ls=":", lw=0.7)
    ax.axhline(1, color="0.65", ls="--", lw=0.6)
    ax.axhline(-1, color="0.65", ls="--", lw=0.6)
    ax.set_xticks(x_positions, labels)
    ax.set_ylabel(r"endpoint $q_x(2\pi)$")
    ax.set_title("topological, conjugated, and trivial controls")
    ax.scatter([], [], color="0.2", marker="o", s=24, label="CCW")
    ax.scatter([], [], edgecolor="0.2", facecolor="none", marker="s", s=24, label="CW")
    ax.legend(frameon=False, loc="lower left")
    panel_labels(axes)
    fig.tight_layout()
    save(fig, "exact_flux_threading_controls")


def complete_widths(rows: list[dict[str, Any]]) -> dict[tuple[str, int], int]:
    counts: dict[tuple[str, int], set[int]] = defaultdict(set)
    for row in rows:
        if row["protocol"] == "nsh1":
            counts[(row["wall"], row["Nx"])].add(row["sample_id"])
    return {key: len(value) for key, value in counts.items()}


def figure_mean_paths(rows: list[dict[str, Any]]) -> None:
    counts = complete_widths(rows)
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 4.65), sharex=True, sharey=True)
    for row_index, wall in enumerate(("soft", "hard")):
        for column, direction in enumerate(("ccw", "cw")):
            ax = axes[row_index, column]
            for nx in (20, 24, 28, 32):
                selected = [
                    row for row in rows
                    if row["protocol"] == "nsh1" and row["Nx"] == nx
                    and row["wall"] == wall and row["direction"] == direction
                ]
                if not selected:
                    continue
                values = np.stack([row["q_x"] for row in selected])
                mean = np.mean(values, axis=0)
                sd = np.std(values, axis=0, ddof=1) if len(selected) > 1 else np.zeros_like(mean)
                x = np.abs(selected[0]["phi"] - selected[0]["phi"][0]) / (2 * np.pi)
                label = rf"$N_x={nx}$, $S={counts[(wall, nx)]}$"
                ax.plot(x, mean, color=COLORS[nx], lw=1.25, marker=MARKERS[nx], markevery=32,
                        ms=2.7, label=label)
                ax.fill_between(x, mean - sd, mean + sd, color=COLORS[nx], alpha=0.14, linewidth=0)
            ax.axhline(0, color="0.35", ls=":", lw=0.7)
            ax.axhline(1 if direction == "ccw" else -1, color="0.55", ls="--", lw=0.65)
            ax.set_title(f"{wall} wall, {direction.upper()}")
            if row_index == 1:
                ax.set_xlabel(r"threaded flux $|\phi-\phi_0|/(2\pi)$")
            if column == 0:
                ax.set_ylabel(r"raw $q_x$")
            ax.legend(frameon=False, loc="upper left" if direction == "ccw" else "lower left")
    panel_labels(axes.ravel())
    fig.tight_layout()
    save(fig, "wall_diabatic_mean_qx_paths_snapshot")


def figure_endpoint_histograms(rows: list[dict[str, Any]]) -> None:
    counts = complete_widths(rows)
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 4.55), sharex=True, sharey=True)
    bins = np.linspace(-1.15, 1.15, 47)
    for row_index, wall in enumerate(("soft", "hard")):
        for column, direction in enumerate(("ccw", "cw")):
            ax = axes[row_index, column]
            for nx in (20, 24, 28, 32):
                selected = [
                    row["endpoint_q_x"] for row in rows
                    if row["protocol"] == "nsh1" and row["Nx"] == nx
                    and row["wall"] == wall and row["direction"] == direction
                ]
                if selected:
                    ax.hist(selected, bins=bins, histtype="step", density=False, lw=1.2,
                            color=COLORS[nx], label=rf"$N_x={nx}$, $S={counts[(wall, nx)]}$")
            ax.axvline(0, color="0.35", ls=":", lw=0.7)
            ax.axvline(1 if direction == "ccw" else -1, color="0.55", ls="--", lw=0.65)
            ax.set_title(f"{wall} wall, {direction.upper()}")
            if row_index == 1:
                ax.set_xlabel(r"endpoint raw $q_x$")
            if column == 0:
                ax.set_ylabel("trajectories")
            ax.legend(frameon=False)
    panel_labels(axes.ravel())
    fig.tight_layout()
    save(fig, "wall_diabatic_endpoint_histograms_snapshot")


def figure_method_and_width(rows: list[dict[str, Any]], old_rows: list[dict[str, Any]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.75))
    ax = axes[0]
    comparisons = [
        ("nsh1_N20x24", 20, "soft"),
        ("nsh1_N20x24", 20, "hard"),
        ("nsh1_N24x24", 24, "soft"),
        ("nsh1_N24x24", 24, "hard"),
        ("dense_N20x24", 20, "soft"),
        ("dense_N20x24", 20, "hard"),
    ]
    labels, old_fraction, new_fraction = [], [], []
    for cell, nx, wall in comparisons:
        protocol = cell.split("_", 1)[0]
        old = [row["signed_endpoint_q_x"] for row in old_rows if row["cell"] == cell and row["wall"] == wall]
        new = [
            row["signed_endpoint_q_x"] for row in rows
            if row["protocol"] == protocol and row["Nx"] == nx and row["wall"] == wall
        ]
        labels.append(f"{protocol}\n$N_x={nx}$\n{wall}")
        old_fraction.append(np.mean(np.asarray(old) > 0.5))
        new_fraction.append(np.mean(np.asarray(new) > 0.5))
    x = np.arange(len(labels))
    ax.bar(x - 0.18, old_fraction, width=0.36, color="0.72", edgecolor="0.25", label="ordinary overlap")
    ax.bar(x + 0.18, new_fraction, width=0.36, color="#0072B2", edgecolor="#005580", label="wall diabatic")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel(r"fraction with $\sigma q_x>0.5$")
    ax.legend(frameon=False, loc="lower right")
    ax.set_title("same endpoint ensembles")

    ax = axes[1]
    for wall, marker, ls in (("soft", "o", "-"), ("hard", "s", "--")):
        x_values, medians, q25, q75 = [], [], [], []
        for nx in (20, 24, 28, 32):
            selected = np.asarray([
                abs(row["signed_endpoint_q_x"] - 1.0) for row in rows
                if row["protocol"] == "nsh1" and row["Nx"] == nx and row["wall"] == wall
            ])
            if selected.size:
                x_values.append(nx / 2)
                medians.append(np.median(selected))
                q25.append(np.quantile(selected, 0.25))
                q75.append(np.quantile(selected, 0.75))
        ax.plot(x_values, medians, ls, color="#D55E00" if wall == "hard" else "#009E73",
                marker=marker, ms=4, lw=1.2, label=wall)
        ax.fill_between(x_values, q25, q75, color="#D55E00" if wall == "hard" else "#009E73", alpha=0.14)
    ax.set_yscale("log")
    ax.set_xlabel(r"wall separation $W=N_x/2$")
    ax.set_ylabel(r"$|\sigma q_x-1|$")
    ax.set_title(r"median and interquartile range, $n_{\rm shell}=1$")
    ax.legend(frameon=False)
    panel_labels(axes)
    fig.tight_layout()
    save(fig, "ordinary_overlap_vs_wall_diabatic_snapshot")


def summarize(rows: list[dict[str, Any]], controls: list[dict[str, Any]], manifest: dict[str, Any]) -> dict[str, Any]:
    groups: list[dict[str, Any]] = []
    keys = sorted({(row["protocol"], row["Nx"], row["wall"], row["direction"]) for row in rows})
    for protocol, nx, wall, direction in keys:
        selected = [
            row for row in rows
            if (row["protocol"], row["Nx"], row["wall"], row["direction"]) == (protocol, nx, wall, direction)
        ]
        endpoint = np.asarray([row["endpoint_q_x"] for row in selected])
        signed = np.asarray([row["signed_endpoint_q_x"] for row in selected])
        resolved = np.asarray([row["resolved"] for row in selected])
        groups.append(
            {
                "protocol": protocol,
                "Nx": nx,
                "Ny": int(selected[0]["Ny"]),
                "wall": wall,
                "direction": direction,
                "samples": len(selected),
                "endpoint_mean": float(np.mean(endpoint)),
                "endpoint_sd": float(np.std(endpoint, ddof=1)) if len(selected) > 1 else 0.0,
                "resolved_fraction": float(np.mean(resolved)),
                "correct_sign_gt_0p5_fraction": float(np.mean(signed > 0.5)),
                "correct_sign_gt_0p9_fraction": float(np.mean(signed > 0.9)),
                "near_unit_fraction": float(np.mean((signed >= 0.95) & (signed <= 1.05))),
                "median_abs_quantization_error": float(np.median(np.abs(signed - 1.0))),
                "median_crossing_internal_gap": float(np.median([row["crossing_internal_gap"] for row in selected])),
                "median_crossing_external_gap": float(np.median([row["crossing_external_gap"] for row in selected])),
            }
        )
    unresolved: dict[str, int] = defaultdict(int)
    for row in rows:
        if not row["resolved"]:
            unresolved[row["unresolved_reason"]] += 1
    return {
        "schema": "flux_threading_note_snapshot_summary_v1",
        "created_utc": manifest["created_utc"],
        "campaign_complete_at_snapshot": manifest["campaign_complete_at_snapshot"],
        "verified_pump_pairs": len(manifest["pump_pairs"]),
        "verified_control_pairs": len(manifest["control_pairs"]),
        "directional_pump_paths": len(rows),
        "cells": groups,
        "unresolved_directional_reason_counts": dict(sorted(unresolved.items())),
        "maximum_charge_residual": float(max(row["charge_residual"] for row in rows)),
        "maximum_projector_residual": float(max(row["projector_residual"] for row in rows)),
        "maximum_undo_error": float(max(row["undo_error"] for row in rows)),
        "control_endpoints": [
            {
                key: row[key]
                for key in ("variant", "source_kind", "wall", "direction", "endpoint_q_x", "endpoint_instantaneous_q_x", "source_chern")
            }
            for row in controls
        ],
    }


def main() -> int:
    manifest = freeze_or_load_manifest()
    pump_rows = load_pump_rows(manifest)
    controls = load_controls(manifest)
    old_rows = load_old_rows(manifest)
    style()
    figure_exact_controls(controls)
    figure_mean_paths(pump_rows)
    figure_endpoint_histograms(pump_rows)
    figure_method_and_width(pump_rows, old_rows)

    scalar_fields = [
        "task_id", "protocol", "Nx", "Ny", "wall", "sample_id", "direction", "sigma",
        "endpoint_q_x", "signed_endpoint_q_x", "resolved", "unresolved_reason", "source_chern",
        "crossing_internal_gap", "crossing_external_gap", "crossing_link", "crossing_wall_weight",
        "crossing_wall_polarization", "endpoint_instantaneous_q_x", "charge_residual",
        "projector_residual", "undo_error",
    ]
    write_csv(ASSET_ROOT / "samplewise_raw_qx.csv", pump_rows, scalar_fields)
    summary = summarize(pump_rows, controls, manifest)
    write_csv(ASSET_ROOT / "cell_summary.csv", summary["cells"], list(summary["cells"][0]))
    (ASSET_ROOT / "snapshot_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
