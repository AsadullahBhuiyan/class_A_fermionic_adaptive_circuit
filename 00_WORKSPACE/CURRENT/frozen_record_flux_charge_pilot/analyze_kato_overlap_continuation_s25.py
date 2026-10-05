#!/usr/bin/env python3
"""Analyze raw CW/CCW Kato continuation and compare with saved static grids."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import run_kato_overlap_continuation_s25 as campaign  # noqa: E402
import run_state_projector_pump_s100 as static20  # noqa: E402
import run_state_projector_pump_variants as static_variants  # noqa: E402


DEFAULT_OUTPUT = campaign.DEFAULT_OUTPUT
FIGURE_WIDTH = 7.05
COLORS = {"N20x24": "#0072B2", "N24x24": "#D55E00"}
LINESTYLES = {"ccw": "-", "cw": "--"}
MARKERS = {"ccw": "o", "cw": "s"}


def _configure_plotting() -> None:
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
            "axes.linewidth": 0.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "pdf.fonttype": 42,
        }
    )


def _save_figure(figure: plt.Figure, root: Path, stem: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    figure.savefig(root / f"{stem}.pdf", bbox_inches="tight")
    figure.savefig(root / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def _bootstrap_mean(values: np.ndarray, rng: np.random.Generator, draws: int) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    indices = rng.integers(0, values.size, size=(draws, values.size))
    means = np.mean(values[indices], axis=1)
    return float(np.mean(values)), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _legacy_contexts(config: dict[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    contexts: dict[tuple[str, int], dict[str, Any]] = {}
    n20_config = static20.load_config(PROJECT_ROOT / config["sources"]["N20x24"]["config"])
    static20.validate_config(n20_config)
    contexts[("N20x24", 64)] = {
        "module": static20,
        "config": n20_config,
        "root": PROJECT_ROOT / config["sources"]["N20x24"]["legacy_grid64_root"],
        "config_hash": static20.scientific_config_hash(n20_config),
        "hashes": static20.source_hashes(),
    }
    n20_128_config = static_variants.load_config(
        PROJECT_ROOT / config["sources"]["N20x24"]["legacy_grid128_config"]
    )
    static_variants.validate_config(n20_128_config)
    contexts[("N20x24", 128)] = {
        "module": static_variants,
        "config": n20_128_config,
        "root": PROJECT_ROOT / config["sources"]["N20x24"]["legacy_grid128_root"],
        "config_hash": static_variants.scientific_config_hash(n20_128_config),
        "hashes": static_variants.source_hashes(),
    }
    n24_config = static_variants.load_config(
        PROJECT_ROOT / config["sources"]["N24x24"]["config"]
    )
    static_variants.validate_config(n24_config)
    contexts[("N24x24", 64)] = {
        "module": static_variants,
        "config": n24_config,
        "root": PROJECT_ROOT / config["sources"]["N24x24"]["legacy_grid64_root"],
        "config_hash": static_variants.scientific_config_hash(n24_config),
        "hashes": static_variants.source_hashes(),
    }
    return contexts


def _verified_legacy_endpoint(
    context: dict[str, Any], wall: str, direction: str, sample_id: int,
) -> float:
    module = context["module"]
    config = context["config"]
    root = context["root"]
    burnin = next(
        row for row in module.burnin_tasks(config)
        if row["wall"] == wall and int(row["sample_id"]) == sample_id
    )
    if module is static20:
        burnin_status = module.verify_pair(
            root, burnin, context["config_hash"], context["hashes"], config
        )
        verify = module.verify_pair
    else:
        endpoint_context = module._endpoint_context(config, root)
        burnin_status = module._verify_own_pair(
            endpoint_context["root"], burnin, endpoint_context["config_hash"],
            endpoint_context["source_hashes"], endpoint_context["config"],
        )
        verify = module._verify_own_pair
    if not burnin_status[0] or burnin_status[2] is None:
        raise RuntimeError(f"legacy burn-in is not verified: {burnin['task_id']}: {burnin_status[1]}")
    burnin_sha = str(burnin_status[2]["result"]["sha256"])
    pump = next(
        row for row in module.pump_tasks(config)
        if row["wall"] == wall and row["direction"] == direction
        and int(row["sample_id"]) == sample_id
    )
    status = verify(
        root, pump, context["config_hash"], context["hashes"], config,
        burnin_sha256=burnin_sha,
    )
    if not status[0]:
        raise RuntimeError(f"legacy path is not verified: {pump['task_id']}: {status[1]}")
    path, _ = module.result_paths(root, pump)
    with np.load(path, allow_pickle=False) as saved:
        return float(np.asarray(saved["continued_q_x"])[-1])


def load_rows(config: dict[str, Any], output_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    status = campaign.inventory(config, output_root)
    missing = [task_id for task_id, row in status["paths"].items() if not row[0]]
    if missing:
        raise RuntimeError(f"analysis requires 200 verified paths; missing/invalid={len(missing)}")
    legacy = _legacy_contexts(config)
    rows: list[dict[str, Any]] = []
    for task in campaign.tasks(config):
        path, _ = campaign.result_paths(output_root, task)
        with np.load(path, allow_pickle=False) as saved:
            row: dict[str, Any] = {
                **task,
                "path": str(path),
                "phi": np.array(saved["phi"], copy=True),
                "q_x": np.array(saved["q_x"], copy=True),
                "delta_N_left": np.array(saved["delta_N_left"], copy=True),
                "delta_N_right": np.array(saved["delta_N_right"], copy=True),
                "density_x": np.array(saved["density_x"], copy=True),
                "endpoint_q_x": float(saved["q_x"][-1]),
                "minimum_delta_sel": float(saved["minimum_delta_sel_over_path"]),
                "minimum_overlap_margin": float(saved["minimum_selected_weight_margin_over_path"]),
                "maximum_adaptive_error": float(np.max(saved["adaptive_error"])),
                "accepted_steps": int(saved["accepted_steps_total"]),
                "rejected_steps": int(saved["rejected_steps_total"]),
                "maximum_charge_residual": float(saved["maximum_charge_residual"]),
                "maximum_projector_mismatch": float(saved["maximum_direct_projector_mismatch"]),
            }
        row["legacy64_q_x"] = _verified_legacy_endpoint(
            legacy[(task["size"], 64)], task["wall"], task["direction"], task["sample_id"]
        )
        row["legacy128_q_x"] = (
            _verified_legacy_endpoint(
                legacy[("N20x24", 128)], task["wall"], task["direction"], task["sample_id"]
            )
            if task["size"] == "N20x24" else np.nan
        )
        row["event_kato"] = abs(row["endpoint_q_x"]) > 0.5
        row["event_legacy64"] = abs(row["legacy64_q_x"]) > 0.5
        row["event_legacy128"] = (
            abs(row["legacy128_q_x"]) > 0.5 if np.isfinite(row["legacy128_q_x"]) else None
        )
        row["classification_changed_64"] = row["event_kato"] != row["event_legacy64"]
        row["classification_changed_128"] = (
            row["event_kato"] != row["event_legacy128"]
            if row["event_legacy128"] is not None else None
        )
        rows.append(row)
    return rows, status


def _write_sample_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "task_id", "size", "Nx", "Ny", "wall", "direction", "sigma", "sample_id",
        "endpoint_q_x", "legacy64_q_x", "legacy128_q_x", "event_kato",
        "event_legacy64", "event_legacy128", "classification_changed_64",
        "classification_changed_128", "minimum_delta_sel", "minimum_overlap_margin",
        "maximum_adaptive_error", "accepted_steps", "rejected_steps",
        "maximum_charge_residual", "maximum_projector_mismatch",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})


def plot_paths(rows: list[dict[str, Any]], root: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(FIGURE_WIDTH, 4.6), sharex=True, sharey=True)
    for column, size in enumerate(("N20x24", "N24x24")):
        for row_index, wall in enumerate(("soft", "hard")):
            axis = axes[row_index, column]
            for direction in ("ccw", "cw"):
                selected = [row for row in rows if row["size"] == size and row["wall"] == wall and row["direction"] == direction]
                phi = np.abs(selected[0]["phi"] - selected[0]["phi"][0])
                values = np.stack([row["q_x"] for row in selected])
                mean, sd = np.mean(values, axis=0), np.std(values, axis=0, ddof=1)
                label = "CCW" if direction == "ccw" else "CW"
                axis.plot(phi / np.pi, mean, LINESTYLES[direction], color=COLORS[size], label=label)
                axis.fill_between(phi / np.pi, mean - sd, mean + sd, color=COLORS[size], alpha=0.17)
            axis.axhline(0.0, color="0.35", linestyle=":", linewidth=0.8)
            axis.axhline(1.0, color="0.55", linestyle="--", linewidth=0.7)
            axis.axhline(-1.0, color="0.55", linestyle="--", linewidth=0.7)
            axis.set_title(f"{size.replace('N', '').replace('x', r'$\\times$')}, {wall} wall")
            if column == 0:
                axis.set_ylabel(r"raw $q_x$")
            if row_index == 1:
                axis.set_xlabel(r"$|\phi-\phi_0|/\pi$")
            if row_index == 0 and column == 0:
                axis.legend(frameon=False, ncol=2)
    for label, axis in zip("abcd", axes.ravel()):
        axis.text(-0.17, 1.04, f"({label})", transform=axis.transAxes, fontweight="bold")
    figure.tight_layout()
    _save_figure(figure, root, "kato_raw_qx_paths")


def plot_endpoint_histograms(rows: list[dict[str, Any]], root: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(FIGURE_WIDTH, 4.6), sharex=True, sharey=True)
    bins = np.linspace(-1.2, 1.2, 25)
    for row_index, wall in enumerate(("soft", "hard")):
        for column, direction in enumerate(("ccw", "cw")):
            axis = axes[row_index, column]
            for size in ("N20x24", "N24x24"):
                values = [row["endpoint_q_x"] for row in rows if row["size"] == size and row["wall"] == wall and row["direction"] == direction]
                axis.hist(values, bins=bins, histtype="step", linewidth=1.4, color=COLORS[size], label=size.replace("N", ""))
            axis.axvline(0.0, color="0.35", linestyle=":", linewidth=0.8)
            axis.set_title(f"{wall} wall, {direction.upper()}")
            if row_index == 1:
                axis.set_xlabel(r"endpoint raw $q_x$")
            if column == 0:
                axis.set_ylabel("paths")
            if row_index == 0 and column == 0:
                axis.legend(frameon=False, title=r"$N_x\times N_y$")
    for label, axis in zip("abcd", axes.ravel()):
        axis.text(-0.17, 1.04, f"({label})", transform=axis.transAxes, fontweight="bold")
    figure.tight_layout()
    _save_figure(figure, root, "kato_endpoint_histograms")


def plot_legacy_comparison(rows: list[dict[str, Any]], root: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(FIGURE_WIDTH, 4.8), sharex=True, sharey=True)
    for row_index, wall in enumerate(("soft", "hard")):
        for column, size in enumerate(("N20x24", "N24x24")):
            axis = axes[row_index, column]
            for direction in ("ccw", "cw"):
                selected = [row for row in rows if row["size"] == size and row["wall"] == wall and row["direction"] == direction]
                axis.scatter(
                    [row["legacy64_q_x"] for row in selected],
                    [row["endpoint_q_x"] for row in selected],
                    s=15, facecolors="none", edgecolors=COLORS[size],
                    marker=MARKERS[direction], label=direction.upper(), alpha=0.85,
                )
            axis.plot([-1.2, 1.2], [-1.2, 1.2], color="0.4", linestyle="--", linewidth=0.8)
            axis.axhline(0.0, color="0.65", linewidth=0.6)
            axis.axvline(0.0, color="0.65", linewidth=0.6)
            axis.set_title(f"{size.replace('N', '')}, {wall} wall")
            if row_index == 1:
                axis.set_xlabel(r"64-grid static $q_x$")
            if column == 0:
                axis.set_ylabel(r"adaptive Kato $q_x$")
            if row_index == 0 and column == 0:
                axis.legend(frameon=False)
    for label, axis in zip("abcd", axes.ravel()):
        axis.text(-0.17, 1.04, f"({label})", transform=axis.transAxes, fontweight="bold")
    figure.tight_layout()
    _save_figure(figure, root, "kato_vs_legacy64")


def plot_change_diagnostics(rows: list[dict[str, Any]], root: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(FIGURE_WIDTH, 2.45))
    x_fields = ("minimum_delta_sel", "minimum_overlap_margin", "accepted_steps")
    labels = (r"minimum $\delta_{\rm sel}$", "minimum overlap margin", "accepted RK4 steps")
    for axis, field, label in zip(axes, x_fields, labels):
        for changed, marker, face in ((False, "o", "none"), (True, "x", None)):
            selected = [row for row in rows if row["classification_changed_64"] == changed]
            kwargs = {"marker": marker, "s": 16, "alpha": 0.75, "label": "changed" if changed else "unchanged"}
            if face is not None:
                kwargs.update({"facecolors": face, "edgecolors": "#0072B2"})
            else:
                kwargs.update({"color": "#D55E00"})
            axis.scatter([row[field] for row in selected], [abs(row["endpoint_q_x"]) for row in selected], **kwargs)
        axis.set_xlabel(label)
        axis.set_ylabel(r"endpoint $|q_x|$")
        if field in {"minimum_delta_sel", "minimum_overlap_margin"}:
            positive = [row[field] for row in rows if row[field] > 0.0]
            if positive:
                axis.set_xscale("log")
    axes[0].legend(frameon=False)
    for label, axis in zip("abc", axes):
        axis.text(-0.22, 1.04, f"({label})", transform=axis.transAxes, fontweight="bold")
    figure.tight_layout()
    _save_figure(figure, root, "kato_classification_diagnostics")


def summarize(rows: list[dict[str, Any]], config: dict[str, Any], status: dict[str, Any]) -> dict[str, Any]:
    rng = np.random.default_rng(int(config["analysis"]["bootstrap_seed"]))
    draws = int(config["analysis"]["bootstrap_draws"])
    groups = []
    for size in ("N20x24", "N24x24"):
        for wall in ("soft", "hard"):
            for direction in ("ccw", "cw"):
                selected = [row for row in rows if row["size"] == size and row["wall"] == wall and row["direction"] == direction]
                values = np.asarray([row["endpoint_q_x"] for row in selected])
                mean, low, high = _bootstrap_mean(values, rng, draws)
                groups.append(
                    {
                        "size": size, "wall": wall, "direction": direction, "n": len(selected),
                        "mean_endpoint_q_x": mean, "bootstrap_ci95": [low, high],
                        "standard_deviation": float(np.std(values, ddof=1)),
                        "fraction_abs_qx_gt_0p5": float(np.mean(np.abs(values) > 0.5)),
                        "fraction_abs_qx_gt_0p75": float(np.mean(np.abs(values) > 0.75)),
                        "legacy64_classification_changes": int(sum(row["classification_changed_64"] for row in selected)),
                        "legacy128_classification_changes": (
                            int(sum(row["classification_changed_128"] for row in selected))
                            if size == "N20x24" else None
                        ),
                    }
                )
    return {
        "schema": "kato_overlap_continuation_analysis_v1",
        "campaign_id": config["campaign_id"],
        "config_hash": status["config_hash"],
        "verified_paths": len(rows),
        "groups": groups,
        "maximum_charge_residual": float(max(row["maximum_charge_residual"] for row in rows)),
        "maximum_projector_mismatch": float(max(row["maximum_projector_mismatch"] for row in rows)),
        "minimum_delta_sel": float(min(row["minimum_delta_sel"] for row in rows)),
        "minimum_overlap_margin": float(min(row["minimum_overlap_margin"] for row in rows)),
        "quantization_is_acceptance_gate": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=campaign.DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    config = campaign.load_config(args.config.resolve())
    campaign.validate_config(config)
    output_root = args.output_root.resolve()
    rows, status = load_rows(config, output_root)
    analysis_root = output_root / "analysis"
    figure_root = analysis_root / "figures"
    _configure_plotting()
    _write_sample_csv(rows, analysis_root / "sample_resolved_raw_qx.csv")
    _write_sample_csv(
        [row for row in rows if row["classification_changed_64"]],
        analysis_root / "classification_changes_vs_legacy64.csv",
    )
    _write_sample_csv(
        [row for row in rows if row["classification_changed_128"]],
        analysis_root / "classification_changes_vs_legacy128.csv",
    )
    plot_paths(rows, figure_root)
    plot_endpoint_histograms(rows, figure_root)
    plot_legacy_comparison(rows, figure_root)
    plot_change_diagnostics(rows, figure_root)
    summary = summarize(rows, config, status)
    campaign._atomic_json(analysis_root / "analysis_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
