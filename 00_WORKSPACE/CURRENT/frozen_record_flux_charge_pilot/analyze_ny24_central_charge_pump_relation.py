#!/usr/bin/env python3
"""Pair Ny=24 single-trajectory endpoint c_eff fits with quantized pump sectors."""

from __future__ import annotations

import os

for _name in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_name, "1")

import csv
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mannwhitneyu, pearsonr, spearmanr
from threadpoolctl import threadpool_limits
from tqdm.auto import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import run_state_projector_pump_s100 as campaign  # noqa: E402
from state_projector_endpoint_cft import endpoint_cft_payload  # noqa: E402


CONFIG_PATH = PROJECT_ROOT / "campaign_config.state_projector_pump_n20x24_s100_v1.json"
INPUT_ROOT = PROJECT_ROOT / "results" / "N20x24_state_projector_pump_s100_v1"
OUTPUT_ROOT = INPUT_ROOT / "analysis" / "endpoint_cft_relation_v1"
WALLS = ("soft", "hard")
DIRECTIONS = ("ccw", "cw")
CFT_CONTRACT = {
    "estimator": "full-x strip von Neumann entropy from occupied-frame singular values",
    "widths": "Ay=0,...,Ny//2",
    "origins": "all y0=0,...,Ny-1 averaged within each trajectory",
    "fit_model": "S(Ay)=b+(c_eff/3)*log[(Ny/pi)*sin(pi*Ay/Ny)]",
    "fit_ay_min": 8,
    "fit_ay_max": "Ny//2",
    "logarithm": "natural",
    "y0_chunk": 4,
    "occupation_tolerance": 1e-8,
    "full_covariance_materialized": False,
}


def _fit_worker(payload: tuple[str, int, str]) -> dict[str, Any]:
    wall, sample_id, path_text = payload
    with threadpool_limits(limits=1):
        with np.load(path_text, allow_pickle=False) as saved:
            frame = np.array(saved["frame"], dtype=np.complex128, copy=True)
        fit = endpoint_cft_payload(frame, nx=20, ny=24, contract=CFT_CONTRACT)
    return {
        "wall": wall,
        "sample_id": int(sample_id),
        **{key: np.asarray(value) for key, value in fit.items()},
    }


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["CMU Sans Serif", "Computer Modern Sans Serif", "DejaVu Sans"],
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
            "savefig.dpi": 300,
        }
    )


def _bootstrap_relation(
    c_eff: np.ndarray,
    q_odd: np.ndarray,
    event: np.ndarray,
    *,
    draws: int,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, len(c_eff), size=(int(draws), len(c_eff)))
    correlation = np.full(draws, np.nan)
    contrast = np.full(draws, np.nan)
    for draw, index in enumerate(indices):
        fit_draw, q_draw, event_draw = c_eff[index], q_odd[index], event[index]
        if np.std(fit_draw) > 0 and np.std(q_draw) > 0:
            correlation[draw] = np.corrcoef(fit_draw, q_draw)[0, 1]
        if np.any(event_draw) and np.any(~event_draw):
            contrast[draw] = fit_draw[event_draw].mean() - fit_draw[~event_draw].mean()
    correlation = correlation[np.isfinite(correlation)]
    contrast = contrast[np.isfinite(contrast)]
    return {
        "pearson_bootstrap95_low": float(np.quantile(correlation, 0.025)),
        "pearson_bootstrap95_high": float(np.quantile(correlation, 0.975)),
        "event_minus_closure_bootstrap95_low": float(np.quantile(contrast, 0.025)),
        "event_minus_closure_bootstrap95_high": float(np.quantile(contrast, 0.975)),
    }


def analyze(*, workers: int = 20) -> dict[str, Any]:
    config = campaign.load_config(CONFIG_PATH)
    campaign.validate_config(config)
    status = campaign.inventory(config, INPUT_ROOT)
    if not all(row[0] for row in status["burnins"].values()):
        raise RuntimeError("Ny=24 analysis requires all 200 verified endpoint frames")
    if not all(row[0] for row in status["pumps"].values()):
        raise RuntimeError("Ny=24 analysis requires all 400 verified pump paths")

    inputs = []
    for task in campaign.burnin_tasks(config):
        result_path, _ = campaign.result_paths(INPUT_ROOT, task)
        completion = status["burnins"][task["task_id"]][2]
        inputs.append((task["wall"], int(task["sample_id"]), str(result_path)))
        if completion["result"]["sha256"] != campaign.sha256_path(result_path):
            raise RuntimeError(f"endpoint frame changed during analysis: {result_path}")

    context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=int(workers), mp_context=context) as pool:
        fits = list(
            tqdm(
                pool.map(_fit_worker, inputs),
                total=len(inputs),
                desc="Ny=24 endpoint c_eff",
                unit="trajectory",
            )
        )
    fits.sort(key=lambda row: (WALLS.index(row["wall"]), row["sample_id"]))

    samples, widths = 100, 13
    entropy = np.empty((2, samples, widths), dtype=np.float64)
    c_eff = np.empty((2, samples), dtype=np.float64)
    c_eff_stderr = np.empty_like(c_eff)
    fit_r2 = np.empty_like(c_eff)
    fit_rss = np.empty_like(c_eff)
    for row in fits:
        wi, sample = WALLS.index(row["wall"]), int(row["sample_id"])
        entropy[wi, sample] = row["endpoint_entropy_y0_averaged"]
        c_eff[wi, sample] = row["endpoint_c_eff"]
        c_eff_stderr[wi, sample] = row["endpoint_c_eff_stderr"]
        fit_r2[wi, sample] = row["endpoint_entropy_fit_r2"]
        fit_rss[wi, sample] = row["endpoint_entropy_fit_rss"]

    q_endpoint = np.empty((2, 2, samples), dtype=np.float64)
    for task in campaign.pump_tasks(config):
        wi = WALLS.index(task["wall"])
        di = DIRECTIONS.index(task["direction"])
        sample = int(task["sample_id"])
        result_path, _ = campaign.result_paths(INPUT_ROOT, task)
        with np.load(result_path, allow_pickle=False) as saved:
            q_endpoint[wi, di, sample] = float(saved["continued_q_x"][-1])
    q_odd = 0.5 * (q_endpoint[:, 0] - q_endpoint[:, 1])
    threshold = float(config["acceptance"]["pump_event_threshold"])
    event = np.abs(q_odd) > threshold

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    table_path = OUTPUT_ROOT / "ny24_samplewise_central_charge_and_pump.csv"
    table_rows = []
    with table_path.open("w", encoding="utf-8", newline="") as handle:
        fields = (
            "wall", "sample_id", "endpoint_c_eff", "endpoint_c_eff_fit_stderr",
            "endpoint_entropy_fit_r2", "endpoint_entropy_fit_rss", "ccw_q_x",
            "cw_q_x", "direction_odd_q_x", "pump_event",
        )
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for wi, wall in enumerate(WALLS):
            for sample in range(samples):
                row = {
                    "wall": wall,
                    "sample_id": sample,
                    "endpoint_c_eff": float(c_eff[wi, sample]),
                    "endpoint_c_eff_fit_stderr": float(c_eff_stderr[wi, sample]),
                    "endpoint_entropy_fit_r2": float(fit_r2[wi, sample]),
                    "endpoint_entropy_fit_rss": float(fit_rss[wi, sample]),
                    "ccw_q_x": float(q_endpoint[wi, 0, sample]),
                    "cw_q_x": float(q_endpoint[wi, 1, sample]),
                    "direction_odd_q_x": float(q_odd[wi, sample]),
                    "pump_event": int(event[wi, sample]),
                }
                writer.writerow(row)
                table_rows.append(row)

    aggregate_path = OUTPUT_ROOT / "ny24_central_charge_pump_aggregate.npz"
    np.savez_compressed(
        aggregate_path,
        walls=np.asarray(WALLS),
        sample_ids=np.arange(samples),
        ay_values=np.arange(widths),
        endpoint_entropy_y0_averaged=entropy,
        endpoint_c_eff=c_eff,
        endpoint_c_eff_fit_stderr=c_eff_stderr,
        endpoint_entropy_fit_r2=fit_r2,
        endpoint_entropy_fit_rss=fit_rss,
        q_endpoint=q_endpoint,
        direction_odd_q_x=q_odd,
        pump_event=event,
    )

    statistics = []
    for wi, wall in enumerate(WALLS):
        pearson = pearsonr(c_eff[wi], q_odd[wi])
        spearman = spearmanr(c_eff[wi], q_odd[wi])
        event_values = c_eff[wi, event[wi]]
        closure_values = c_eff[wi, ~event[wi]]
        mann_whitney = mannwhitneyu(event_values, closure_values, alternative="two-sided")
        pooled_std = np.sqrt(
            (
                (len(event_values) - 1) * event_values.var(ddof=1)
                + (len(closure_values) - 1) * closure_values.var(ddof=1)
            )
            / (len(event_values) + len(closure_values) - 2)
        )
        row = {
            "wall": wall,
            "samples": samples,
            "pump_events": int(event[wi].sum()),
            "closures": int((~event[wi]).sum()),
            "c_eff_mean": float(c_eff[wi].mean()),
            "c_eff_std": float(c_eff[wi].std(ddof=1)),
            "c_eff_min": float(c_eff[wi].min()),
            "c_eff_max": float(c_eff[wi].max()),
            "minimum_fit_r2": float(fit_r2[wi].min()),
            "pearson_c_eff_vs_q_odd": float(pearson.statistic),
            "pearson_pvalue": float(pearson.pvalue),
            "spearman_c_eff_vs_q_odd": float(spearman.statistic),
            "spearman_pvalue": float(spearman.pvalue),
            "pump_event_c_eff_mean": float(event_values.mean()),
            "pump_event_c_eff_std": float(event_values.std(ddof=1)),
            "pump_event_c_eff_median": float(np.median(event_values)),
            "closure_c_eff_mean": float(closure_values.mean()),
            "closure_c_eff_std": float(closure_values.std(ddof=1)),
            "closure_c_eff_median": float(np.median(closure_values)),
            "event_minus_closure_c_eff": float(
                event_values.mean() - closure_values.mean()
            ),
            "event_minus_closure_cohen_d": float(
                (event_values.mean() - closure_values.mean()) / pooled_std
            ),
            "mann_whitney_pvalue": float(mann_whitney.pvalue),
            "event_over_closure_auc": float(
                mann_whitney.statistic / (len(event_values) * len(closure_values))
            ),
            **_bootstrap_relation(
                c_eff[wi], q_odd[wi], event[wi], draws=10000, seed=2026090424 + wi
            ),
        }
        statistics.append(row)

    _style()
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.7), sharex=True, sharey=True)
    for wi, (wall, ax) in enumerate(zip(WALLS, axes)):
        ax.scatter(
            c_eff[wi, ~event[wi]], q_odd[wi, ~event[wi]],
            s=14, color="0.55", alpha=0.8, label="closure",
        )
        ax.scatter(
            c_eff[wi, event[wi]], q_odd[wi, event[wi]],
            s=14, color="#1f77b4", alpha=0.8, label="pump event",
        )
        ax.axhline(0, color="0.65", linestyle=":", linewidth=0.7)
        ax.axhline(threshold, color="0.35", linestyle="--", linewidth=0.7)
        ax.set_xlabel(r"single-trajectory endpoint $c_{\rm eff}$")
        ax.set_title(
            f"{wall.capitalize()} wall, "
            rf"$r={statistics[wi]['pearson_c_eff_vs_q_odd']:.2f}$"
        )
        ax.text(-0.13, 1.04, f"({chr(97 + wi)})", transform=ax.transAxes, fontsize=9)
    axes[0].set_ylabel(r"endpoint direction-odd $q_x$")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    figure_pdf = OUTPUT_ROOT / "ny24_central_charge_vs_pump_endpoint.pdf"
    figure_png = OUTPUT_ROOT / "ny24_central_charge_vs_pump_endpoint.png"
    fig.savefig(figure_pdf, bbox_inches="tight")
    fig.savefig(figure_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "schema": "ny24_samplewise_endpoint_cft_pump_relation_v1",
        "campaign_id": config["campaign_id"],
        "config_hash": status["config_hash"],
        "source_hashes": status["source_hashes"],
        "endpoint_cft_contract": CFT_CONTRACT,
        "input_endpoint_pairs": 200,
        "input_pump_pairs": 400,
        "independent_sampling_unit": "wall-specific monitored trajectory",
        "statistics": statistics,
        "samplewise_csv": str(table_path),
        "aggregate_npz": str(aggregate_path),
        "figures": {"pdf": str(figure_pdf), "png": str(figure_png)},
        "endpoint_cft_source_sha256": campaign.sha256_path(
            PROJECT_ROOT / "state_projector_endpoint_cft.py"
        ),
        "analysis_source_sha256": campaign.sha256_path(Path(__file__).resolve()),
    }
    summary_path = OUTPUT_ROOT / "analysis_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


if __name__ == "__main__":
    workers = int(os.environ.get("NY24_CFT_WORKERS", "20"))
    analyze(workers=workers)
