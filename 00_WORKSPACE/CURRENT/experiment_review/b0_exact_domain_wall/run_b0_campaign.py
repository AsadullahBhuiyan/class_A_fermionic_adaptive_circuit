#!/usr/bin/env python3
"""Run the complete deterministic B0 campaign with atomic, resumable stages."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import os
import shutil
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

from b0lib import (  # noqa: E402
    FGTN_SRC,
    PACKAGE_DIR,
    REPO_ROOT,
    atomic_csv,
    atomic_json,
    environment_metadata,
    geometry_key,
    git_metadata,
    load_locked_config,
    make_model,
    preflight_checks,
    run_geometry,
    run_twist_validation,
    sha256_file,
    utc_now,
)


LOG = logging.getLogger("b0_campaign")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-id", help="New timestamped campaign identifier")
    parser.add_argument("--resume", help="Resume an existing campaign identifier")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--max-workers", type=int, default=int(os.environ.get("MAX_WORKERS", "1")))
    parser.add_argument("--cpu-list", default=os.environ.get("CPU_LIST", ""))
    parser.add_argument(
        "--select-idle-cpus",
        action="store_true",
        help="Print a balanced comma-separated idle physical-core CPU list and exit",
    )
    parser.add_argument("--limit", type=int, default=24, help=argparse.SUPPRESS)
    parser.add_argument("--idle-threshold", type=float, default=20.0, help=argparse.SUPPRESS)
    return parser.parse_args()


def select_idle_cpus(limit: int, idle_threshold: float) -> str:
    import psutil

    proc = subprocess.run(
        ["lscpu", "-p=CPU,CORE,SOCKET,NODE"], text=True, capture_output=True, check=True
    )
    siblings: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for line in proc.stdout.splitlines():
        if not line or line.startswith("#"):
            continue
        cpu, core, socket, node = (int(token) for token in line.split(",")[:4])
        siblings.setdefault((socket, core), []).append((cpu, node))
    usage = psutil.cpu_percent(interval=0.6, percpu=True)
    by_node: dict[int, list[int]] = {}
    for members in siblings.values():
        if all(usage[cpu] < idle_threshold for cpu, _ in members):
            cpu, node = min(members)
            by_node.setdefault(node, []).append(cpu)
    for values in by_node.values():
        values.sort()
    chosen: list[int] = []
    nodes = sorted(by_node)
    while len(chosen) < limit and any(by_node[node] for node in nodes):
        for node in nodes:
            if by_node[node] and len(chosen) < limit:
                chosen.append(by_node[node].pop(0))
    return ",".join(str(cpu) for cpu in chosen)


def setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(processName)s %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    root.handlers[:] = [stream, file_handler]


def new_manifest(
    campaign_id: str,
    campaign_dir: Path,
    config_path: Path,
    config_hash: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    sources = [
        config_path,
        PACKAGE_DIR / "b0lib.py",
        PACKAGE_DIR / "run_b0_campaign.py",
        PACKAGE_DIR / "launch_b0_tmux.sh",
        FGTN_SRC / "classA_U1FGTN.py",
    ]
    legacy_paths = [REPO_ROOT / value for value in config_legacy_paths(config_path)]
    return {
        "schema_version": 2,
        "campaign": "B0_exact_domain_wall",
        "campaign_id": campaign_id,
        "campaign_dir": str(campaign_dir.relative_to(REPO_ROOT)),
        "created_utc": utc_now(),
        "updated_utc": utc_now(),
        "status": "running",
        "dtype": "complex128",
        "canonical_hamiltonian_source": "classA_U1FGTN._domain_wall_hamiltonian",
        "supersedes_campaign_id": "20260816_184251",
        "correction_classification": ["protocol_or_estimator_mismatch", "finite_size_or_convergence_effect"],
        "config_path": str(config_path.relative_to(REPO_ROOT)),
        "config_sha256": config_hash,
        "source_hashes": {
            str(path.relative_to(REPO_ROOT)): sha256_file(path) for path in sources if path.exists()
        },
        "legacy_reference_hashes": {
            str(path.relative_to(REPO_ROOT)): sha256_file(path) for path in legacy_paths if path.exists()
        },
        "git": git_metadata(),
        "environment": environment_metadata(),
        "allocation": {
            "cpu_list": args.cpu_list,
            "max_workers": args.max_workers,
            "blas_threads": int(os.environ.get("BLAS_THREADS", "1")),
        },
        "stages": {},
    }


def config_legacy_paths(config_path: Path) -> list[str]:
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    return [str(value) for value in config.get("legacy_modular_paths", [])]


def update_manifest(path: Path, manifest: dict[str, Any]) -> None:
    manifest["updated_utc"] = utc_now()
    atomic_json(path, manifest)


def stage_start(manifest: dict[str, Any], path: Path, stage: str) -> None:
    manifest["stages"][stage] = {"status": "running", "started_utc": utc_now()}
    update_manifest(path, manifest)


def stage_done(
    manifest: dict[str, Any], path: Path, stage: str, details: dict[str, Any] | None = None
) -> None:
    item = manifest["stages"].setdefault(stage, {})
    item.update({"status": "complete", "completed_utc": utc_now()})
    if details:
        item.update(details)
    update_manifest(path, manifest)


def geometry_paths(campaign_dir: Path, construction: str, nx: int, ny: int) -> tuple[Path, Path]:
    key = geometry_key(construction, nx, ny)
    return campaign_dir / "raw" / f"{key}.npz", campaign_dir / "status" / f"{key}.json"


def valid_complete_geometry(npz_path: Path, summary_path: Path) -> dict[str, Any] | None:
    if not npz_path.is_file() or not summary_path.is_file():
        return None
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("npz_sha256") != sha256_file(npz_path):
            return None
        with np.load(npz_path, allow_pickle=False) as data:
            if int(data["schema_version"]) != 1:
                return None
        return summary
    except Exception:
        return None


def run_geometry_tasks(
    tasks: list[tuple[str, int, int]],
    config: dict[str, Any],
    campaign_dir: Path,
    workers: int,
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    pending = []
    for construction, nx, ny in tasks:
        npz_path, summary_path = geometry_paths(campaign_dir, construction, nx, ny)
        existing = valid_complete_geometry(npz_path, summary_path)
        if existing is not None:
            LOG.info("resume: skipping complete geometry %s", existing["key"])
            summaries.append(existing)
        else:
            pending.append((construction, nx, ny, npz_path, summary_path))
    if not pending:
        return summaries
    LOG.info("running %d geometry jobs with max_workers=%d", len(pending), workers)
    with concurrent.futures.ProcessPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {
            pool.submit(run_geometry, construction, nx, ny, config, npz_path, summary_path): (
                construction,
                nx,
                ny,
            )
            for construction, nx, ny, npz_path, summary_path in pending
        }
        try:
            for future in concurrent.futures.as_completed(futures):
                task = futures[future]
                summary = future.result()
                LOG.info("completed %s", summary["key"])
                summaries.append(summary)
        except BaseException:
            for future in futures:
                future.cancel()
            raise
    return summaries


def single_geometry_calibration(summary: dict[str, Any], config: dict[str, Any]) -> dict[str, bool]:
    """Evaluate every B0 gate available on one Ny=48 geometry."""
    edges = summary["edge_rows"]
    v0, v1 = float(edges[0]["velocity"]), float(edges[1]["velocity"])
    relative_v = abs(abs(v0) - abs(v1)) / max(0.5 * (abs(v0) + abs(v1)), 1e-300)
    entropy_rows = summary["entropy_rows"]
    full_rows = [row for row in entropy_rows if row["quantity"] == "full_strip"]
    full_q1 = next((row for row in full_rows if row["q"] == 1), None)
    wall_q1 = [
        row
        for row in entropy_rows
        if row["quantity"] == "physical_wall_contour" and row["q"] == 1
    ]
    factor_errors = (
        [abs(2 * float(row["slope"]) / float(full_q1["slope"]) - 1) for row in wall_q1]
        if full_q1 is not None
        else []
    )
    corr = [
        row
        for row in summary["correlator_rows"]
        if row["curve"] in {"wall_0", "wall_1"}
        and row["model"] == "chord_power"
        and row["endpoint_shift"] == 0
    ]
    response = summary["response_rows"]
    rv = [float(row["physical_velocity"]) for row in response]
    response_mismatch = (
        [
            abs(abs(rv[0]) - abs(v0)) / max(abs(v0), 1e-300),
            abs(abs(rv[1]) - abs(v1)) / max(abs(v1), 1e-300),
        ]
        if len(rv) == 2
        else [np.inf]
    )
    flows = summary["flow_rows"]
    checks = {
        "fixed_rank": bool(summary["pass_fixed_rank"]),
        "numerics": bool(summary["pass_numerics"]),
        "mass": float(summary["edge"]["width_ratio"]) <= float(config["width_ratio_max"]),
        "spectral": bool(v0 * v1 < 0 and relative_v <= config["edge_velocity_relative_tolerance"]),
        "localization": all(
            np.isfinite(float(row["localization_length"]))
            and float(row["localization_length"]) > 0
            for row in edges
        ),
        "entropy": bool(full_rows)
        and all(
            abs(float(row["c_estimate"]) - 1.0) <= config["entropy_c_absolute_tolerance"]
            for row in full_rows
        ),
        "contour": len(factor_errors) == 2
        and max(factor_errors) <= config["entropy_wall_factor_relative_tolerance"],
        "correlator": len(corr) == 2
        and all(
            abs(float(row["parameter_1"]) - 2.0)
            <= config["correlator_beta_absolute_tolerance"]
            for row in corr
        ),
        "physical_response": len(rv) == 2
        and rv[0] * rv[1] < 0
        and np.sign(rv[0]) == np.sign(v0)
        and np.sign(rv[1]) == np.sign(v1)
        and max(response_mismatch) <= config["edge_velocity_relative_tolerance"]
        and summary["response_charge_drift"] <= config["response_charge_tolerance"]
        and summary["response_epsilon_max_relative_error"]
        <= config["response_epsilon_relative_tolerance"],
        "modular": bool(summary["modular_diagnostics"]["pass"]),
        "twist": len(flows) == 2
        and min(float(row["min_overlap_singular_value"]) for row in flows)
        > config["overlap_singular_tolerance"]
        and [row["absolute_flow"] for row in flows] == [1, 1]
        and sum(row["signed_flow"] for row in flows) == 0,
    }
    checks["pass"] = bool(all(checks.values()))
    return checks


def choose_width(summaries: list[dict[str, Any]], config: dict[str, Any]) -> tuple[int, bool, list[dict[str, Any]]]:
    lookup = {(row["construction"], int(row["nx"])): row for row in summaries}
    widths = [int(value) for value in config["nx_scan"]]
    per_width = {
        nx: {
            construction: single_geometry_calibration(lookup[(construction, nx)], config)
            for construction in config["constructions"]
        }
        for nx in widths
    }
    selection_rows = []
    accepted = None
    for index, nx in enumerate(widths):
        ratios = {
            construction: float(lookup[(construction, nx)]["edge"]["width_ratio"])
            for construction in config["constructions"]
        }
        mass_pass = all(value <= float(config["width_ratio_max"]) for value in ratios.values())
        calibration_pass = all(
            per_width[nx][construction]["pass"] for construction in config["constructions"]
        )
        next_nx = widths[index + 1] if index + 1 < len(widths) else None
        next_calibration_pass = bool(
            next_nx is not None
            and all(
                per_width[next_nx][construction]["pass"]
                for construction in config["constructions"]
            )
        )
        passed = bool(mass_pass and calibration_pass and next_calibration_pass)
        selection_rows.append(
            {
                "nx": nx,
                **{f"ratio_{key}": value for key, value in ratios.items()},
                "threshold": float(config["width_ratio_max"]),
                "mass_pass_both": mass_pass,
                **{
                    f"calibration_pass_{construction}": per_width[nx][construction]["pass"]
                    for construction in config["constructions"]
                },
                "single_width_full_calibration_pass": calibration_pass,
                "next_nx": "" if next_nx is None else next_nx,
                "next_width_full_calibration_pass": next_calibration_pass,
                "pass_both": passed,
            }
        )
        if accepted is None and passed:
            accepted = nx
    gate_pass = accepted is not None
    if accepted is None:
        accepted = max(widths)
    return accepted, gate_pass, selection_rows


def flatten_rows(summaries: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    out = []
    for summary in summaries:
        prefix = {
            "geometry_key": summary["key"],
            "construction": summary["construction"],
            "nx": summary["nx"],
            "ny": summary["ny"],
        }
        for row in summary.get(key, []):
            out.append({**prefix, **row})
    return out


def configure_plot_style(config: dict[str, Any]) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["CMU Sans Serif", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "legend.fontsize": 7,
            "figure.dpi": int(config["figure_dpi"]),
            "savefig.dpi": int(config["figure_dpi"]),
        }
    )


def save_figure(fig: plt.Figure, figures_dir: Path, stem: str) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(figures_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(figures_dir / f"{stem}.png", bbox_inches="tight")
    plt.close(fig)


def build_figures(
    summaries: list[dict[str, Any]],
    selected_nx: int,
    selection_rows: list[dict[str, Any]],
    campaign_dir: Path,
    config: dict[str, Any],
) -> None:
    configure_plot_style(config)
    width = float(config["figure_width_inches"])
    figures_dir = campaign_dir / "figures"
    fig, ax = plt.subplots(figsize=(width, 2.45))
    for construction in config["constructions"]:
        values = [row[f"ratio_{construction}"] for row in selection_rows]
        ax.semilogy(config["nx_scan"], values, "o-", label=construction.replace("_", " "))
    ax.axhline(float(config["width_ratio_max"]), color="black", ls="--", lw=0.8, label="10% gate")
    ax.axvline(selected_nx, color="0.5", ls=":", lw=0.8)
    ax.set(xlabel=r"$N_x$", ylabel=r"$m/(2\pi |v|/N_y)$", title=r"B0 transverse-width gate ($N_y=48$)")
    ax.legend(frameon=False)
    save_figure(fig, figures_dir, "width_convergence")

    fig, axes = plt.subplots(
        len(config["constructions"]), 2,
        figsize=(2 * width, 2.2 * len(config["constructions"])),
        squeeze=False,
    )
    for row_index, construction in enumerate(config["constructions"]):
        scan = sorted(
            [
                item
                for item in summaries
                if item["construction"] == construction
                and int(item["ny"]) == int(config["nx_scan_ny"])
                and int(item["nx"]) in config["nx_scan"]
            ],
            key=lambda item: int(item["nx"]),
        )
        nxs = [int(item["nx"]) for item in scan]
        beta0, beta1, mv0, mv1 = [], [], [], []
        for item in scan:
            corr = [
                row
                for row in item["correlator_rows"]
                if row["curve"] in {"wall_0", "wall_1"}
                and row["model"] == "chord_power"
                and row["endpoint_shift"] == 0
            ]
            corr.sort(key=lambda row: row["curve"])
            primary = sorted(
                [row for row in item["modular_rows"] if row["is_primary"]],
                key=lambda row: int(row["wall_index"]),
            )
            beta0.append(float(corr[0]["parameter_1"]))
            beta1.append(float(corr[1]["parameter_1"]))
            mv0.append(float(primary[0]["modular_velocity"]))
            mv1.append(float(primary[1]["modular_velocity"]))
        axes[row_index, 0].plot(nxs, beta0, "o-", label="wall 0")
        axes[row_index, 0].plot(nxs, beta1, "s--", label="wall 1")
        axes[row_index, 0].axhline(2, color="black", ls=":", lw=0.8)
        axes[row_index, 0].set(
            xlabel=r"$N_x$", ylabel=r"$\beta$", title=f"{construction}: wall correlator"
        )
        axes[row_index, 0].legend(frameon=False)
        axes[row_index, 1].plot(nxs, mv0, "o-", label="wall 0")
        axes[row_index, 1].plot(nxs, mv1, "s--", label="wall 1")
        axes[row_index, 1].axhline(0, color="black", ls=":", lw=0.8)
        axes[row_index, 1].set(
            xlabel=r"$N_x$", ylabel=r"$v_{\rm mod}$",
            title=f"{construction}: endpoint handedness",
        )
        axes[row_index, 1].legend(frameon=False)
    save_figure(fig, figures_dir, "local_calibration_convergence")

    for construction in config["constructions"]:
        candidates = [
            row for row in summaries if row["construction"] == construction and int(row["nx"]) == selected_nx
        ]
        candidates.sort(key=lambda row: int(row["ny"]))
        nys, c1, wall0, wall1 = [], [], [], []
        for row in candidates:
            fits = row["entropy_rows"]
            full = next(item for item in fits if item["quantity"] == "full_strip" and item["q"] == 1)
            one = next(item for item in fits if item["quantity"] == "physical_wall_contour" and item["wall_index"] == 0 and item["q"] == 1)
            two = next(item for item in fits if item["quantity"] == "physical_wall_contour" and item["wall_index"] == 1 and item["q"] == 1)
            nys.append(row["ny"])
            c1.append(full["c_estimate"])
            wall0.append(one["slope"])
            wall1.append(two["slope"])
        fig, axes = plt.subplots(1, 2, figsize=(2 * width, 2.45))
        axes[0].plot(nys, c1, "o-")
        axes[0].axhline(1, color="black", ls="--", lw=0.8)
        axes[0].set(xlabel=r"$N_y$", ylabel=r"$c_1$", title="full-strip entropy")
        axes[1].plot(nys, wall0, "o-", label="lower-x wall")
        axes[1].plot(nys, wall1, "s-", label="upper-x wall")
        axes[1].axhline(1 / 6, color="black", ls="--", lw=0.8)
        axes[1].set(xlabel=r"$N_y$", ylabel="contour slope", title="single-physical-wall contours")
        axes[1].legend(frameon=False)
        save_figure(fig, figures_dir, f"entropy_scaling__{construction}")

        representative = min(candidates, key=lambda row: abs(int(row["ny"]) - 48))
        npz_path, _ = geometry_paths(campaign_dir, construction, selected_nx, representative["ny"])
        with np.load(npz_path, allow_pickle=False) as data:
            fig, axes = plt.subplots(1, 2, figsize=(2 * width, 2.45))
            k = data["spectrum_k"]
            axes[0].plot(k, data["spectrum_low_energies"], color="0.25", lw=0.45)
            axes[0].plot(k, data["wall_branch_energies"][:, 0], color="C0", lw=1.1)
            axes[0].plot(k, data["wall_branch_energies"][:, 1], color="C1", lw=1.1)
            axes[0].set(xlabel=r"$k_y$", ylabel=r"$E$", title="low spectrum and wall branches")
            r = data["correlator_r"]
            curves = data["correlator_derived"]
            axes[1].loglog(r, curves[0], "o-", ms=2.5, label="wall 0")
            axes[1].loglog(r, curves[1], "s-", ms=2.5, label="wall 1")
            axes[1].loglog(r, curves[-1], ".-", ms=2, label="full-x mean")
            axes[1].set(xlabel=r"$r_y$", ylabel=r"$C_G$", title="fixed-slice correlators")
            axes[1].legend(frameon=False)
            save_figure(fig, figures_dir, f"spectrum_correlator__{construction}")

            fig, axes = plt.subplots(1, 2, figsize=(2 * width, 2.45))
            response = data["response_chi_aligned_mean"]
            wall = int(data["wall_positions"][0])
            image = axes[0].imshow(
                response[0, :, wall, :], aspect="auto", origin="lower", cmap="RdBu_r"
            )
            axes[0].set(xlabel=r"$y-y_s$", ylabel="time index", title="physical density response")
            fig.colorbar(image, ax=axes[0], fraction=0.046)
            twist_e = data["twist_wall_branch_energies"]
            phi = data["twist_phi"]
            axes[1].plot(phi, twist_e[:, :, 0], color="C0", lw=0.4)
            axes[1].plot(phi, twist_e[:, :, 1], color="C1", lw=0.4)
            axes[1].set(xlabel=r"$\phi_y$", ylabel=r"$E_{\rm wall}$", title="twist spectral flow")
            save_figure(fig, figures_dir, f"response_twist__{construction}")

            primary_index = int(data["modular_primary_width_index"])
            modular_times = data["modular_times"]
            handedness = data["modular_handedness"][primary_index]
            predictions = data["modular_fit_predictions"][primary_index, :, 0]
            retention = data["modular_wall_retention"][primary_index]
            packets = data["modular_packets"][primary_index]
            fig, axes = plt.subplots(2, 2, figsize=(2 * width, 4.5))
            for wall_index in range(2):
                axes[0, 0].plot(modular_times, handedness[wall_index], label=f"wall {wall_index}")
                axes[0, 0].plot(modular_times, predictions[wall_index], "--", lw=0.9)
                axes[0, 1].plot(
                    modular_times,
                    np.min(retention[wall_index], axis=0),
                    label=f"wall {wall_index}",
                )
                contrast = packets[wall_index, 0].sum(axis=1) - packets[wall_index, 1].sum(axis=1)[:, ::-1]
                image = axes[1, wall_index].imshow(
                    contrast,
                    aspect="auto",
                    origin="lower",
                    extent=[0, representative["ny"] // 2 - 1, modular_times[0], modular_times[-1]],
                    cmap="RdBu_r",
                )
                axes[1, wall_index].set(
                    xlabel=r"relative $y$", ylabel=r"modular time",
                    title=f"wall {wall_index}: endpoint contrast",
                )
                fig.colorbar(image, ax=axes[1, wall_index], fraction=0.046)
            axes[0, 0].set(
                xlim=(0, 3), xlabel=r"modular time", ylabel=r"$D_w(t)$",
                title="signed endpoint drift",
            )
            axes[0, 0].legend(frameon=False)
            axes[0, 1].axhline(
                float(config["modular_primary_retention_min"]), color="black", ls=":", lw=0.8
            )
            axes[0, 1].set(
                xlim=(0, 3), xlabel=r"modular time", ylabel="retained charge",
                title="wall-window retention",
            )
            axes[0, 1].legend(frameon=False)
            save_figure(fig, figures_dir, f"modular_endpoint_drift__{construction}")


def latex_escape(value: Any) -> str:
    text = str(value)
    for old, new in (("_", r"\_"), ("%", r"\%"), ("&", r"\&"), ("#", r"\#")):
        text = text.replace(old, new)
    return text


def generate_report(
    summaries: list[dict[str, Any]],
    selected_nx: int,
    width_gate_pass: bool,
    campaign_dir: Path,
    config: dict[str, Any],
    requirements: list[dict[str, Any]],
) -> dict[str, Any]:
    reports = campaign_dir / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    rows = []
    for construction in config["constructions"]:
        summary = next(
            item
            for item in summaries
            if item["construction"] == construction
            and int(item["nx"]) == selected_nx
            and int(item["ny"]) == 48
        )
        entropy = next(
            item for item in summary["entropy_rows"] if item["quantity"] == "full_strip" and item["q"] == 1
        )
        beta = next(
            item
            for item in summary["correlator_rows"]
            if item["curve"] == "wall_0"
            and item["model"] == "chord_power"
            and item["endpoint_shift"] == 0
        )
        modular = sorted(
            [item for item in summary["modular_rows"] if item["is_primary"]],
            key=lambda item: int(item["wall_index"]),
        )
        flows = summary["flow_rows"]
        rows.append(
            f"{latex_escape(construction)} & {summary['edge']['dirac_velocity']:.6g} & "
            f"{summary['edge']['hybridization_mass']:.3g} & {summary['edge']['width_ratio']:.3g} & "
            f"{entropy['c_estimate']:.5g} & {float(beta['parameter_1']):.5g} & "
            f"{float(modular[0]['modular_velocity']):+.4g}, "
            f"{float(modular[1]['modular_velocity']):+.4g} & "
            f"{flows[0]['signed_flow']:+d}, {flows[1]['signed_flow']:+d} \\\\"
        )
    gate_word = "PASS" if width_gate_pass else "FAIL"
    requirement_rows = "\n".join(
        f"{latex_escape(item['requirement'])} & {('PASS' if item['pass'] else 'FAIL')} & "
        f"{latex_escape(item['details'])} \\\\"
        for item in requirements
    )
    tex = rf"""\documentclass[aps,prb,onecolumn,nofootinbib,superscriptaddress]{{revtex4-2}}
\usepackage{{amsmath,amssymb,amsthm,mathtools,bm}}
\usepackage{{booktabs}}
\usepackage{{graphicx}}
\usepackage[colorlinks=true,linkcolor=blue,citecolor=blue,urlcolor=blue]{{hyperref}}
\usepackage{{microtype}}
\setcounter{{tocdepth}}{{2}}
\begin{{document}}
\title{{Campaign B0: Exact Class-A Domain-Wall Calibration}}
\author{{Automated deterministic campaign report}}
\affiliation{{Repository experiment review}}
\date{{{datetime.now().strftime('%B %d, %Y')}}}
\begin{{abstract}}
This report records the fixed-rank CPU benchmark generated by campaign
\texttt{{{latex_escape(campaign_dir.name)}}}.  It calibrates the analysis pipeline and is
not evidence about the adaptive circuit.  The transverse gate status is \textbf{{{gate_word}}};
the declared analysis width is $N_x={selected_nx}$.
\end{{abstract}}
\maketitle
\tableofcontents
\section{{Protocol and provenance}}
Both the coupled spatial-mass interface and the hard-exterior analogue use the canonical
\texttt{{classA\_U1FGTN.\_domain\_wall\_hamiltonian}} source and exactly $N_xN_y$
occupied orbitals.  The locked configuration and complete source hashes are in
\texttt{{manifest.json}}.  The mass criterion
$m/(2\pi|v|/N_y)\leq {config['width_ratio_max']}$ is necessary but not sufficient:
the accepted width and the next larger scanned width must both pass every single-geometry
calibration at $N_y=48$.

\section{{Correction and legacy reconciliation}}
This version supersedes campaign \texttt{{{latex_escape(config['supersedes_campaign_id'])}}}
for interpretation without modifying that run.  Version 1 injected a normalized packet at
the center of the retained half-cylinder and reduced its spreading to an absolute center
of mass.  That observable cancelled counterpropagating modular components.  It also chose
$N_x=12$ from the hybridization mass alone, although the coupled wall correlator and the
modular endpoint diagnostic do not converge until $N_x=20$.  The discrepancy is therefore
classified as an estimator mismatch compounded by finite-width underconvergence, not as a
failure of modular charge spreading.

The corrected construction follows the legacy exact-domain-wall notebook: it reduces a
translated half-cylinder, injects at both entanglement endpoints, and retains the complete
$N(x,y,t)$ evolution.  Unlike the aggregate legacy movies, the four wall/endpoint sources
are evolved separately.  Their symmetry-cancelling signed statistic is
\begin{{equation}}
D_w(t)=\frac12\left[\bar y_{{w,0}}(t)+\bar y_{{w,L-1}}(t)-(L-1)\right],\qquad L=N_y/2.
\end{{equation}}
Its slope is a modular-time handedness diagnostic; its magnitude is not compared with a
physical or spectral velocity.

\section{{Representative results}}
\begin{{table}}[h]
\caption{{Accepted-width representative at $N_y=48$.}}
\begin{{ruledtabular}}
\begin{{tabular}}{{lrrrrrrr}}
construction & $|v|$ & $m$ & width ratio & $c_1$ & $\beta$ & $v_{{\rm mod}}^{{0,1}}$ & wall flows \\
\hline
{chr(10).join(rows)}
\end{{tabular}}
\end{{ruledtabular}}
\end{{table}}

\begin{{figure}}[h]
\includegraphics[width=0.48\textwidth]{{../figures/width_convergence.pdf}}
\caption{{The hybridization-mass condition remains a necessary transverse-width gate.}}
\end{{figure}}

\begin{{figure}}[h]
\includegraphics[width=0.80\textwidth]{{../figures/local_calibration_convergence.pdf}}
\caption{{Local-observable convergence.  The coupled correlator and endpoint modular
handedness exclude $N_x=12,16$ even though their fitted hybridization masses pass.}}
\end{{figure}}

\begin{{figure}}[h]
\includegraphics[width=0.48\textwidth]{{../figures/modular_endpoint_drift__coupled.pdf}}
\includegraphics[width=0.48\textwidth]{{../figures/modular_endpoint_drift__hard_exterior.pdf}}
\caption{{Separate entanglement-endpoint injections, signed drifts, and wall retention for
the accepted-width coupled and hard-exterior constructions.}}
\end{{figure}}

\section{{Entropy accounting}}
The full-width strip has two entanglement boundaries and intersects both physical domain
walls.  Each three-column physical-wall contour window includes both entanglement cuts.
Those window sums are diagnostic contributions to the full entropy, not isolated subsystem
entropies.  The saved products test equality of the contour sum and the entropy, equality
of the two wall slopes, and the factor-of-two relation between a wall-window slope and the
full-strip slope.

\section{{Acceptance ledger}}
\begin{{longtable}}{{p{{0.36\textwidth}}p{{0.10\textwidth}}p{{0.46\textwidth}}}}
\toprule
requirement & status & details \\
\midrule
{requirement_rows}
\bottomrule
\end{{longtable}}
The machine-readable ledger retains the same decisions and their thresholds.  Any failed
item remains explicit rather than changing the estimator or gate.
\end{{document}}
"""
    tex = tex.replace(r"\usepackage{booktabs}", "\\usepackage{booktabs}\n\\usepackage{longtable}")
    tex_path = reports / "b0_exact_domain_wall_report.tex"
    tex_path.write_text(tex, encoding="utf-8")
    compile_proc = subprocess.run(
        ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
        cwd=reports,
        text=True,
        capture_output=True,
        check=False,
    )
    (reports / "latexmk.stdout.log").write_text(compile_proc.stdout, encoding="utf-8")
    (reports / "latexmk.stderr.log").write_text(compile_proc.stderr, encoding="utf-8")
    return {
        "tex": str(tex_path.relative_to(campaign_dir)),
        "pdf": str((reports / "b0_exact_domain_wall_report.pdf").relative_to(campaign_dir)),
        "latexmk_returncode": compile_proc.returncode,
    }


def write_legacy_reconciliation(
    campaign_dir: Path,
    selected_nx: int,
    summaries: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    processed = campaign_dir / "processed"
    old_id = str(config["supersedes_campaign_id"])
    old_ledger_path = PACKAGE_DIR / "results" / old_id / "processed" / "acceptance_ledger.json"
    old_ledger = (
        json.loads(old_ledger_path.read_text(encoding="utf-8"))
        if old_ledger_path.is_file()
        else {}
    )
    legacy_references = []
    for relative in config["legacy_modular_paths"]:
        path = REPO_ROOT / relative
        item: dict[str, Any] = {
            "path": relative,
            "sha256": sha256_file(path) if path.is_file() else None,
        }
        if path.is_file():
            with np.load(path, allow_pickle=True) as data:
                if "metadata_json" in data:
                    item["metadata"] = json.loads(str(data["metadata_json"].item()))
        legacy_references.append(item)
    selected = {
        construction: next(
            item
            for item in summaries
            if item["construction"] == construction
            and int(item["nx"]) == int(selected_nx)
            and int(item["ny"]) == 48
        )
        for construction in config["constructions"]
    }
    current = {}
    for construction, summary in selected.items():
        primary = sorted(
            [row for row in summary["modular_rows"] if row["is_primary"]],
            key=lambda row: int(row["wall_index"]),
        )
        betas = [
            float(row["parameter_1"])
            for row in summary["correlator_rows"]
            if row["curve"] in {"wall_0", "wall_1"}
            and row["model"] == "chord_power"
            and row["endpoint_shift"] == 0
        ]
        current[construction] = {
            "modular_velocities": [float(row["modular_velocity"]) for row in primary],
            "modular_diagnostics": summary["modular_diagnostics"],
            "wall_correlator_betas": betas,
        }
    protocols = [
        {
            "protocol": "legacy_exact_dw",
            "source_y": "0 and L-1",
            "physical_walls": "both simultaneously",
            "x_support": "one and three columns",
            "normalization": "unnormalized aggregate charge",
            "observable": "complete N(x,y,t) and movies",
            "fit": "none",
        },
        {
            "protocol": f"v1_{old_id}",
            "source_y": "L/2",
            "physical_walls": "separate",
            "x_support": "one cell source; three-column analysis",
            "normalization": "unit norm",
            "observable": "absolute COM",
            "fit": "t=2 through 0.75 tmax",
        },
        {
            "protocol": f"v2_{campaign_dir.name}",
            "source_y": "0 and L-1, evolved separately",
            "physical_walls": "separate",
            "x_support": "one and three columns",
            "normalization": "unit norm isolated; legacy aggregate also saved",
            "observable": "D_w endpoint handedness and complete N(x,y,t)",
            "fit": "primary t=0.1 through 2.0 plus declared sensitivities",
        },
    ]
    atomic_csv(processed / "tables" / "modular_protocol_reconciliation.csv", protocols)
    reconciliation = {
        "schema_version": 1,
        "classification": [
            "protocol_or_estimator_mismatch",
            "finite_size_or_convergence_effect",
        ],
        "superseded_campaign_id": old_id,
        "superseded_artifact_preserved": True,
        "old_acceptance_ledger_sha256": (
            sha256_file(old_ledger_path) if old_ledger_path.is_file() else None
        ),
        "old_failed_requirements": [
            row for row in old_ledger.get("requirements", []) if not row.get("pass", False)
        ],
        "legacy_references": legacy_references,
        "corrected_selected_nx": int(selected_nx),
        "corrected_results": current,
        "protocols": protocols,
    }
    atomic_json(processed / "legacy_reconciliation.json", reconciliation)
    return reconciliation


def aggregate(
    summaries: list[dict[str, Any]],
    selection_rows: list[dict[str, Any]],
    selected_nx: int,
    width_gate_pass: bool,
    campaign_dir: Path,
    config: dict[str, Any],
    twist_validations: list[dict[str, Any]],
) -> dict[str, Any]:
    tables = campaign_dir / "processed" / "tables"
    atomic_csv(tables / "width_selection.csv", selection_rows)
    atomic_csv(tables / "edge_parameters.csv", flatten_rows(summaries, "edge_rows"))
    atomic_csv(tables / "entropy_fits.csv", flatten_rows(summaries, "entropy_rows"))
    atomic_csv(tables / "correlator_fits.csv", flatten_rows(summaries, "correlator_rows"))
    atomic_csv(tables / "response_velocities.csv", flatten_rows(summaries, "response_rows"))
    atomic_csv(tables / "modular_velocities.csv", flatten_rows(summaries, "modular_rows"))
    atomic_csv(tables / "twist_flow_counts.csv", flatten_rows(summaries, "flow_rows"))
    validation_rows = []
    for validation in twist_validations:
        validation_rows.extend(
            {"construction": validation["construction"], **row} for row in validation["rows"]
        )
    atomic_csv(tables / "twist_validation.csv", validation_rows)
    representative = {
        construction: next(
            item
            for item in summaries
            if item["construction"] == construction
            and int(item["nx"]) == selected_nx
            and int(item["ny"]) == 48
        )
        for construction in config["constructions"]
    }
    requirements: list[dict[str, Any]] = []

    def record(name: str, passed: bool, details: str) -> None:
        requirements.append({"requirement": name, "pass": bool(passed), "details": details})

    record("small-system formula and gauge preflight", True, "completed before production stages")
    record(
        "fixed-rank half filling",
        all(item["pass_fixed_rank"] for item in summaries),
        "every geometry must contain exactly Nx Ny occupied orbitals",
    )
    record(
        "Hermiticity, projector, and contour numerics",
        all(item["pass_numerics"] for item in summaries),
        f"absolute tolerance {config['numerical_tolerance']}",
    )
    record(
        "consecutive full-calibration transverse-width gate",
        width_gate_pass,
        f"selected Nx={selected_nx}; mass threshold {config['width_ratio_max']}; "
        "selected and next larger widths must pass every single-geometry gate",
    )
    for construction, summary in representative.items():
        edges = summary["edge_rows"]
        v0, v1 = float(edges[0]["velocity"]), float(edges[1]["velocity"])
        relative_v = abs(abs(v0) - abs(v1)) / max(0.5 * (abs(v0) + abs(v1)), 1e-300)
        record(
            f"{construction}: opposite, compatible spectral velocities",
            v0 * v1 < 0 and relative_v <= config["edge_velocity_relative_tolerance"],
            f"v0={v0:.6g}, v1={v1:.6g}, relative mismatch={relative_v:.3g}",
        )
        xis = [float(edges[0]["localization_length"]), float(edges[1]["localization_length"])]
        record(
            f"{construction}: finite transverse localization lengths",
            all(np.isfinite(value) and value > 0 for value in xis),
            f"xi0={xis[0]:.6g}, xi1={xis[1]:.6g}",
        )
        entropy_rows = summary["entropy_rows"]
        full_rows = [row for row in entropy_rows if row["quantity"] == "full_strip"]
        entropy_pass = bool(full_rows) and all(
            abs(float(row["c_estimate"]) - 1.0) <= config["entropy_c_absolute_tolerance"]
            for row in full_rows
        )
        record(
            f"{construction}: Renyi q=1,2,3 full-strip c=1",
            entropy_pass,
            ", ".join(f"q={row['q']}: c={float(row['c_estimate']):.5g}" for row in full_rows),
        )
        full_q1 = next(row for row in full_rows if row["q"] == 1)
        wall_q1 = [
            row
            for row in entropy_rows
            if row["quantity"] == "physical_wall_contour" and row["q"] == 1
        ]
        factor_errors = [
            abs(2 * float(row["slope"]) / float(full_q1["slope"]) - 1) for row in wall_q1
        ]
        record(
            f"{construction}: single-physical-wall contour factor of two",
            len(factor_errors) == 2
            and max(factor_errors) <= config["entropy_wall_factor_relative_tolerance"],
            f"relative errors={factor_errors}; each window includes both entanglement cuts",
        )
        corr = [
            row
            for row in summary["correlator_rows"]
            if row["curve"] in {"wall_0", "wall_1"}
            and row["model"] == "chord_power"
            and row["endpoint_shift"] == 0
        ]
        betas = [float(row["parameter_1"]) for row in corr]
        record(
            f"{construction}: wall squared-correlator exponent two",
            len(betas) == 2
            and all(abs(beta - 2) <= config["correlator_beta_absolute_tolerance"] for beta in betas),
            f"betas={betas}; all four candidate models remain in correlator_fits.csv",
        )
        response = summary["response_rows"]
        rv = [float(row["physical_velocity"]) for row in response]
        response_mismatch = [
            abs(abs(rv[0]) - abs(v0)) / max(abs(v0), 1e-300),
            abs(abs(rv[1]) - abs(v1)) / max(abs(v1), 1e-300),
        ]
        response_sign = (
            len(rv) == 2
            and rv[0] * rv[1] < 0
            and np.sign(rv[0]) == np.sign(v0)
            and np.sign(rv[1]) == np.sign(v1)
            and max(response_mismatch) <= config["edge_velocity_relative_tolerance"]
        )
        record(
            f"{construction}: physical response chirality",
            response_sign,
            f"response velocities={rv}; spectral velocities={[v0, v1]}; "
            f"relative magnitude errors={response_mismatch}",
        )
        record(
            f"{construction}: response linearity and charge conservation",
            summary["response_charge_drift"] <= config["response_charge_tolerance"]
            and summary["response_epsilon_max_relative_error"]
            <= config["response_epsilon_relative_tolerance"],
            f"charge drift={summary['response_charge_drift']:.3g}; "
            f"epsilon relative error={summary['response_epsilon_max_relative_error']:.3g}",
        )
        modular = sorted(
            [row for row in summary["modular_rows"] if row["is_primary"]],
            key=lambda row: int(row["wall_index"]),
        )
        mv = [float(row["modular_velocity"]) for row in modular]
        modular_diagnostics = summary["modular_diagnostics"]
        record(
            f"{construction}: stable opposite endpoint modular velocities",
            bool(modular_diagnostics["pass"]),
            f"primary velocities={mv}; sign stable={modular_diagnostics['sign_stable']}; "
            f"minimum retention={modular_diagnostics['primary_minimum_wall_retention']:.4g}; "
            f"norm drift={modular_diagnostics['max_norm_drift']:.3g}",
        )
        flows = summary["flow_rows"]
        min_overlap = min(float(row["min_overlap_singular_value"]) for row in flows)
        record(
            f"{construction}: nonsingular overlap tracking",
            min_overlap > config["overlap_singular_tolerance"],
            f"minimum singular value={min_overlap:.6g}",
        )
        record(
            f"{construction}: unit wall flow and zero net flow",
            [row["absolute_flow"] for row in flows] == [1, 1]
            and sum(row["signed_flow"] for row in flows) == 0,
            f"signed flows={[row['signed_flow'] for row in flows]}",
        )
    validation_pass = all(
        validation["pass_gauge"]
        and all(row["absolute_flow"] == 1 for row in validation["rows"])
        for validation in twist_validations
    )
    record(
        "33/65/129-point, both-sign, seam-relocation twist validation",
        validation_pass,
        "all validation rows require unit absolute wall flow and seam/uniform spectral agreement",
    )
    reconciliation = write_legacy_reconciliation(
        campaign_dir, selected_nx, summaries, config
    )
    build_figures(summaries, selected_nx, selection_rows, campaign_dir, config)
    report = generate_report(
        summaries, selected_nx, width_gate_pass, campaign_dir, config, requirements
    )
    ledger = {
        "selected_nx": selected_nx,
        "width_gate_pass": width_gate_pass,
        "width_threshold": config["width_ratio_max"],
        "geometry_count": len(summaries),
        "all_fixed_rank": all(item["pass_fixed_rank"] for item in summaries),
        "all_numerics": all(item["pass_numerics"] for item in summaries),
        "requirements": requirements,
        "all_requirements_pass": all(item["pass"] for item in requirements),
        "report": report,
        "legacy_reconciliation": "processed/legacy_reconciliation.json",
        "correction_classification": reconciliation["classification"],
    }
    atomic_json(campaign_dir / "processed" / "acceptance_ledger.json", ledger)
    return ledger


def main() -> int:
    args = parse_args()
    if args.select_idle_cpus:
        print(select_idle_cpus(args.limit, args.idle_threshold))
        return 0
    if args.resume and args.campaign_id:
        raise SystemExit("Use either --resume or --campaign-id, not both")
    campaign_id = args.resume or args.campaign_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    campaign_dir = PACKAGE_DIR / "results" / campaign_id
    if args.resume:
        resume_manifest_path = campaign_dir / "manifest.json"
        if not resume_manifest_path.is_file():
            raise SystemExit(f"Cannot resume: missing {resume_manifest_path}")
        resume_manifest = json.loads(resume_manifest_path.read_text(encoding="utf-8"))
        recorded_config = REPO_ROOT / str(resume_manifest["config_path"])
        config, config_path, config_hash = load_locked_config(recorded_config)
    else:
        config, config_path, config_hash = load_locked_config()
    for subdir in ("raw", "processed/tables", "figures", "logs", "reports", "status"):
        (campaign_dir / subdir).mkdir(parents=True, exist_ok=True)
    setup_logging(campaign_dir / "logs" / "campaign.log")
    manifest_path = campaign_dir / "manifest.json"
    if args.resume:
        if not manifest_path.is_file():
            raise SystemExit(f"Cannot resume: missing {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("config_sha256") != config_hash:
            raise SystemExit("Cannot resume with a changed locked configuration")
        manifest["status"] = "running"
        manifest.setdefault("resume_events", []).append({"utc": utc_now(), "pid": os.getpid()})
        manifest["allocation"] = {
            "cpu_list": args.cpu_list,
            "max_workers": args.max_workers,
            "blas_threads": int(os.environ.get("BLAS_THREADS", "1")),
        }
    else:
        if manifest_path.exists():
            raise SystemExit(f"Campaign already exists: {campaign_id}; use --resume")
        manifest = new_manifest(campaign_id, campaign_dir, config_path, config_hash, args)
        shutil.copy2(config_path, campaign_dir / config_path.name)
    update_manifest(manifest_path, manifest)
    LOG.info("campaign_id=%s output=%s", campaign_id, campaign_dir)
    try:
        if manifest.get("stages", {}).get("preflight", {}).get("status") != "complete":
            stage_start(manifest, manifest_path, "preflight")
            checks, arrays = preflight_checks(config)
            from b0lib import atomic_npz

            atomic_npz(campaign_dir / "raw" / "preflight_arrays.npz", **arrays)
            atomic_json(campaign_dir / "status" / "preflight.json", checks)
            if not checks["pass"]:
                raise RuntimeError("B0 preflight failed; inspect status/preflight.json")
            stage_done(manifest, manifest_path, "preflight", {"checks": checks})
        if args.preflight_only:
            manifest["status"] = "preflight_complete"
            update_manifest(manifest_path, manifest)
            LOG.info("preflight-only campaign complete")
            return 0

        scan_tasks = [
            (construction, int(nx), int(config["nx_scan_ny"]))
            for construction in config["constructions"]
            for nx in config["nx_scan"]
        ]
        stage_start(manifest, manifest_path, "nx_scan")
        scan_summaries = run_geometry_tasks(scan_tasks, config, campaign_dir, args.max_workers)
        selected_nx, width_gate_pass, selection_rows = choose_width(scan_summaries, config)
        stage_done(
            manifest,
            manifest_path,
            "nx_scan",
            {"selected_nx": selected_nx, "width_gate_pass": width_gate_pass},
        )

        ny_tasks = [
            (construction, selected_nx, int(ny))
            for construction in config["constructions"]
            for ny in config["ny_sequence"]
        ]
        stage_start(manifest, manifest_path, "ny_sequence")
        ny_summaries = run_geometry_tasks(ny_tasks, config, campaign_dir, args.max_workers)
        stage_done(manifest, manifest_path, "ny_sequence")
        summary_map = {item["key"]: item for item in scan_summaries + ny_summaries}
        summaries = list(summary_map.values())

        stage_start(manifest, manifest_path, "twist_validation")
        twist_summaries = []
        for construction in config["constructions"]:
            stem = f"twist_validation__{construction}__Nx{selected_nx:03d}__Ny048"
            npz_path = campaign_dir / "raw" / f"{stem}.npz"
            summary_path = campaign_dir / "status" / f"{stem}.json"
            existing = valid_complete_geometry(npz_path, summary_path)
            if existing is None:
                model = make_model(selected_nx, 48, config)
                existing = run_twist_validation(model, construction, config, npz_path, summary_path)
            twist_summaries.append(existing)
        stage_done(manifest, manifest_path, "twist_validation", {"summaries": twist_summaries})

        stage_start(manifest, manifest_path, "aggregate")
        ledger = aggregate(
            summaries,
            selection_rows,
            selected_nx,
            width_gate_pass,
            campaign_dir,
            config,
            twist_summaries,
        )
        stage_done(manifest, manifest_path, "aggregate", {"ledger": ledger})
        manifest["status"] = "complete" if width_gate_pass else "complete_width_gate_failed"
        manifest["completed_utc"] = utc_now()
        update_manifest(manifest_path, manifest)
        LOG.info("campaign finished with status=%s", manifest["status"])
        return 0
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["failure"] = {
            "utc": utc_now(),
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        update_manifest(manifest_path, manifest)
        LOG.exception("campaign failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
