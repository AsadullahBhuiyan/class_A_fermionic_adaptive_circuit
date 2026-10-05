#!/usr/bin/env python3
"""Extract verified sample-resolved endpoint charge for the Ny=24 pump data."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mannwhitneyu, pearsonr, spearmanr


PROJECT_ROOT = Path(__file__).resolve().parent
BASE_ROOT = PROJECT_ROOT / "results" / "N20x24_state_projector_pump_s100_v1"
DENSE_ROOT = PROJECT_ROOT / "results" / "N20x24_state_projector_pump_s100_dense_v1"
BASE_CFT = (
    BASE_ROOT / "analysis" / "endpoint_cft_relation_v1"
    / "ny24_samplewise_central_charge_and_pump.csv"
)
BASE_OUTPUT = BASE_ROOT / "analysis" / "endpoint_cft_charge_relation_v1"
DENSE_OUTPUT = DENSE_ROOT / "analysis" / "endpoint_charge_snapshot_v1"
HALF_FILLING = 20 * 24
WALLS = ("soft", "hard")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_join() -> dict[tuple[str, int], dict[str, str]]:
    with BASE_CFT.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    lookup = {(row["wall"], int(row["sample_id"])): row for row in rows}
    if len(rows) != 200 or len(lookup) != 200:
        raise RuntimeError("central-charge/pump table is not the expected 200 unique trajectories")
    return lookup


def _endpoint_rows(root: Path, *, require_complete: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for wall in WALLS:
        completion_paths = sorted((root / "burnins" / wall).glob("sample_*.completion.json"))
        for completion_path in completion_paths:
            completion = json.loads(completion_path.read_text(encoding="utf-8"))
            sample_id = int(completion["sample_id"])
            result_path = completion_path.with_suffix("").with_suffix(".npz")
            result = completion["result"]
            if result.get("name") != result_path.name:
                raise RuntimeError(f"result filename mismatch: {completion_path}")
            if int(result.get("bytes", -1)) != result_path.stat().st_size:
                raise RuntimeError(f"result byte-count mismatch: {result_path}")
            if result.get("sha256") != _sha256(result_path):
                raise RuntimeError(f"result checksum mismatch: {result_path}")
            with np.load(result_path, allow_pickle=False) as saved:
                rank = int(np.asarray(saved["rank"]).item())
                initial = float(np.asarray(saved["initial_total_charge"]).item())
                final = float(np.asarray(saved["final_total_charge"]).item())
                left = float(np.asarray(saved["N_left"]).item())
                right = float(np.asarray(saved["N_right"]).item())
                injected = int(np.asarray(saved["net_injected_charge"]).item())
                event_count = int(np.asarray(saved["feedback_event_count"]).item())
                continuity = float(np.asarray(saved["charge_continuity_residual"]).item())
            if abs(final - rank) > 1e-9 or abs(left + right - rank) > 1e-9:
                raise RuntimeError(f"rank/charge disagreement: {result_path}")
            if abs(final - initial - injected) > 1e-9 or abs(continuity) > 1e-9:
                raise RuntimeError(f"feedback charge-continuity disagreement: {result_path}")
            offset = rank - HALF_FILLING
            rows.append(
                {
                    "wall": wall,
                    "sample_id": sample_id,
                    "seed": int(completion["seed"]),
                    "total_charge": rank,
                    "half_filling_charge": HALF_FILLING,
                    "half_filling_offset": offset,
                    "absolute_half_filling_offset": abs(offset),
                    "half_filling_deviation_percent": 100.0 * abs(offset) / HALF_FILLING,
                    "initial_total_charge": initial,
                    "final_total_charge": final,
                    "N_left": left,
                    "N_right": right,
                    "net_injected_charge": injected,
                    "feedback_event_count": event_count,
                    "charge_continuity_residual": continuity,
                    "result_sha256": str(result["sha256"]),
                }
            )
    rows.sort(key=lambda row: (WALLS.index(str(row["wall"])), int(row["sample_id"])))
    if require_complete:
        keys = {(str(row["wall"]), int(row["sample_id"])) for row in rows}
        expected = {(wall, sample_id) for wall in WALLS for sample_id in range(100)}
        if keys != expected:
            raise RuntimeError(f"completed Ny=24 ensemble has {len(keys)}/200 endpoint pairs")
    return rows


def _correlation(x: np.ndarray, y: np.ndarray) -> float | None:
    if x.size < 2 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _bootstrap_mean_difference(
    first: np.ndarray, second: np.ndarray, *, seed: int, draws: int = 20000
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    lhs = np.asarray(first, dtype=np.float64)
    rhs = np.asarray(second, dtype=np.float64)
    lhs_draw = lhs[rng.integers(0, lhs.size, size=(draws, lhs.size))].mean(axis=1)
    rhs_draw = rhs[rng.integers(0, rhs.size, size=(draws, rhs.size))].mean(axis=1)
    difference = lhs_draw - rhs_draw
    return (
        float(lhs.mean() - rhs.mean()),
        float(np.quantile(difference, 0.025)),
        float(np.quantile(difference, 0.975)),
    )


def _statistics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for wall in WALLS:
        selected = [row for row in rows if row["wall"] == wall]
        charge = np.asarray([row["total_charge"] for row in selected], dtype=np.float64)
        offset = charge - HALF_FILLING
        entry: dict[str, Any] = {
            "wall": wall,
            "trajectories": len(selected),
            "mean_total_charge": float(charge.mean()),
            "sample_std_total_charge": float(charge.std(ddof=1)) if len(charge) > 1 else None,
            "minimum_total_charge": int(charge.min()),
            "maximum_total_charge": int(charge.max()),
            "mean_signed_half_filling_offset": float(offset.mean()),
            "mean_absolute_half_filling_offset": float(np.abs(offset).mean()),
            "exact_half_filling_fraction": float(np.mean(offset == 0)),
        }
        if "endpoint_c_eff" in selected[0]:
            c_eff = np.asarray([row["endpoint_c_eff"] for row in selected], dtype=np.float64)
            q_odd = np.asarray([row["direction_odd_q_x"] for row in selected], dtype=np.float64)
            event = np.asarray([row["pump_event"] for row in selected], dtype=np.float64)
            absolute_offset = np.abs(offset)
            pumped_offset = absolute_offset[event == 1]
            closing_offset = absolute_offset[event == 0]
            contrast, contrast_low, contrast_high = _bootstrap_mean_difference(
                pumped_offset,
                closing_offset,
                seed=2026090501 + WALLS.index(wall),
            )
            mann_whitney = mannwhitneyu(
                pumped_offset, closing_offset, alternative="two-sided"
            )
            entry.update(
                {
                    "pearson_total_charge_vs_c_eff": _correlation(charge, c_eff),
                    "pearson_total_charge_vs_direction_odd_q_x": _correlation(charge, q_odd),
                    "mean_charge_pump_event": float(charge[event == 1].mean()),
                    "mean_charge_closing_event": float(charge[event == 0].mean()),
                    "pearson_signed_offset_vs_abs_direction_odd_q_x": _correlation(
                        offset, np.abs(q_odd)
                    ),
                    "pearson_absolute_offset_vs_abs_direction_odd_q_x": _correlation(
                        absolute_offset, np.abs(q_odd)
                    ),
                    "spearman_absolute_offset_vs_abs_direction_odd_q_x": float(
                        spearmanr(absolute_offset, np.abs(q_odd)).statistic
                    ),
                    "point_biserial_absolute_offset_vs_pump_event": _correlation(
                        absolute_offset, event
                    ),
                    "mean_absolute_offset_pump_event": float(pumped_offset.mean()),
                    "mean_absolute_offset_closing_event": float(closing_offset.mean()),
                    "mean_absolute_offset_pump_minus_closing": contrast,
                    "mean_absolute_offset_pump_minus_closing_bootstrap95_low": contrast_low,
                    "mean_absolute_offset_pump_minus_closing_bootstrap95_high": contrast_high,
                    "absolute_offset_mann_whitney_u": float(mann_whitney.statistic),
                    "absolute_offset_mann_whitney_p": float(mann_whitney.pvalue),
                }
            )
        output.append(entry)
    return output


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot_charge_relation(rows: list[dict[str, Any]], output_root: Path) -> dict[str, str]:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["CMU Sans Serif", "Computer Modern Sans Serif", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.75), sharey=True, constrained_layout=True)
    for index, (axis, wall) in enumerate(zip(axes, WALLS)):
        selected = [row for row in rows if row["wall"] == wall]
        offset = np.asarray([row["half_filling_offset"] for row in selected], dtype=float)
        response = np.abs(
            np.asarray([row["direction_odd_q_x"] for row in selected], dtype=float)
        )
        axis.scatter(
            offset,
            response,
            s=18,
            facecolors="none",
            edgecolors="#1675b9" if wall == "soft" else "#c4473a",
            linewidths=0.8,
            alpha=0.8,
        )
        axis.axhline(0.5, color="0.25", linestyle="--", linewidth=0.9)
        axis.set_xlabel(r"Final charge deviation $\Delta N=N_{\rm tot}-480$")
        axis.set_title(f"{wall.capitalize()} wall, $S=100$")
        axis.text(-0.12, 1.04, f"({chr(ord('a') + index)})", transform=axis.transAxes)
    axes[0].set_ylabel(r"Pump-response magnitude $|q_x^{\rm odd}|$")
    output_root.mkdir(parents=True, exist_ok=True)
    pdf = output_root / "ny24_charge_deviation_vs_pump_response.pdf"
    png = output_root / "ny24_charge_deviation_vs_pump_response.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return {"pdf": str(pdf), "png": str(png)}


def main() -> int:
    joined = _read_join()
    base_rows = _endpoint_rows(BASE_ROOT, require_complete=True)
    for row in base_rows:
        source = joined[(str(row["wall"]), int(row["sample_id"]))]
        row.update(
            {
                "endpoint_c_eff": float(source["endpoint_c_eff"]),
                "endpoint_c_eff_fit_stderr": float(source["endpoint_c_eff_fit_stderr"]),
                "endpoint_entropy_fit_r2": float(source["endpoint_entropy_fit_r2"]),
                "ccw_q_x": float(source["ccw_q_x"]),
                "cw_q_x": float(source["cw_q_x"]),
                "direction_odd_q_x": float(source["direction_odd_q_x"]),
                "pump_event": int(source["pump_event"]),
            }
        )
    base_csv = BASE_OUTPUT / "ny24_samplewise_central_charge_pump_and_total_charge.csv"
    _write_csv(base_csv, base_rows)
    base_summary = {
        "schema": "ny24_samplewise_endpoint_charge_relation_v1",
        "created_unix": time.time(),
        "geometry": {"Nx": 20, "Ny": 24},
        "nshell": 1,
        "half_filling_charge": HALF_FILLING,
        "charge_definition": "N_total = Tr(C) = rank(F)",
        "samplewise_csv": str(base_csv),
        "figure": _plot_charge_relation(base_rows, BASE_OUTPUT),
        "statistics": _statistics(base_rows),
    }
    BASE_OUTPUT.mkdir(parents=True, exist_ok=True)
    (BASE_OUTPUT / "analysis_summary.json").write_text(
        json.dumps(base_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    dense_rows = _endpoint_rows(DENSE_ROOT, require_complete=False)
    dense_csv = DENSE_OUTPUT / "ny24_dense_samplewise_total_charge.csv"
    if dense_rows:
        _write_csv(dense_csv, dense_rows)
    dense_summary = {
        "schema": "ny24_dense_samplewise_endpoint_charge_snapshot_v1",
        "created_unix": time.time(),
        "geometry": {"Nx": 20, "Ny": 24},
        "nshell": None,
        "snapshot_is_complete": len(dense_rows) == 200,
        "verified_endpoint_pairs": len(dense_rows),
        "half_filling_charge": HALF_FILLING,
        "charge_definition": "N_total = Tr(C) = rank(F)",
        "samplewise_csv": str(dense_csv) if dense_rows else None,
        "statistics": _statistics(dense_rows) if dense_rows else [],
    }
    DENSE_OUTPUT.mkdir(parents=True, exist_ok=True)
    (DENSE_OUTPUT / "analysis_summary.json").write_text(
        json.dumps(dense_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"completed_nshell1": base_summary, "dense_snapshot": dense_summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
