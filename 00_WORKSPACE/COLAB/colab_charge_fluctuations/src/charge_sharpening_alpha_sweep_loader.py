"""Load and summarize CPU purification charge-sharpening alpha-sweep campaigns."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


BUNDLE_NAME = "colab_charge_fluctuations"
CAMPAIGN_NAME = "purification_charge_sharpening_alpha_sweep"
PROTOCOLS = ("perfect_correction", "postselection")
SHARPENING_THRESHOLD = 1e-2


def metadata_value(run: dict[str, Any], key: str, default: Any = None) -> Any:
    entry = run.get("entry", {})
    if key in entry:
        return entry[key]
    summary = run.get("summary", {})
    if key in summary:
        return summary[key]
    return default


def load_json(path: Path | str) -> Any:
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def resolve_bundle_root(start: Path | str | None = None) -> Path:
    start_path = Path.cwd() if start is None else Path(start)
    for base in [start_path, *start_path.parents]:
        candidates = (base, base / BUNDLE_NAME)
        for candidate in candidates:
            if (candidate / "cpu_data" / CAMPAIGN_NAME).exists() and (candidate / "src").exists():
                return candidate.resolve()
    raise FileNotFoundError(f"Could not locate {BUNDLE_NAME}/cpu_data/{CAMPAIGN_NAME}.")


def _campaign_root(data_root: Path, protocol: str, campaign_id: str) -> Path:
    root = data_root / protocol / "campaigns" / campaign_id
    if not root.exists():
        raise FileNotFoundError(root)
    return root


def _resolve_campaign_id(data_root: Path, campaign_id: str | None, require_complete: bool) -> str:
    if campaign_id is not None:
        return str(campaign_id)
    common_ids: set[str] | None = None
    for protocol in PROTOCOLS:
        campaigns_root = data_root / protocol / "campaigns"
        protocol_ids = {path.name for path in campaigns_root.iterdir() if path.is_dir()}
        common_ids = protocol_ids if common_ids is None else common_ids & protocol_ids
    candidates = sorted(candidate for candidate in (common_ids or ()) if "smoke" not in candidate.lower())
    if require_complete:
        candidates = [
            candidate
            for candidate in candidates
            if all(
                bool(load_json(_campaign_root(data_root, protocol, candidate) / "campaign_manifest.json").get("complete"))
                for protocol in PROTOCOLS
            )
        ]
    if not candidates:
        qualifier = " complete" if require_complete else ""
        raise FileNotFoundError(f"No common{qualifier} {CAMPAIGN_NAME} campaign exists for {PROTOCOLS}.")
    return max(
        candidates,
        key=lambda candidate: max(
            (_campaign_root(data_root, protocol, candidate) / "campaign_manifest.json").stat().st_mtime
            for protocol in PROTOCOLS
        ),
    )


def _load_run(run_dir: Path, entry: dict[str, Any], require_complete: bool) -> dict[str, Any]:
    observables_path = run_dir / "trajectory_observables.npz"
    summary_path = run_dir / "run_summary.json"
    metrics_path = run_dir / "scalar_metrics.csv"
    missing = [path.name for path in (observables_path, summary_path, metrics_path) if not path.exists()]
    if missing:
        if require_complete:
            raise FileNotFoundError(f"{run_dir}: missing {missing}")
        return {"entry": entry, "run_dir": run_dir, "missing": missing, "complete": False}

    summary = load_json(summary_path)
    with np.load(observables_path, allow_pickle=False) as payload:
        arrays = {key: payload[key] for key in payload.files}
    metrics = pd.read_csv(metrics_path)
    samples = int(summary["samples_actual"])
    cycles_count = int(summary["cycles"])
    expected_shape = (samples, cycles_count)
    cycles = arrays["cycles"].astype(np.int64, copy=False)
    sample_indices = arrays["sample_indices"].astype(np.int64, copy=False)
    entropy = arrays["total_entropy"].astype(np.float64, copy=False)
    variance = arrays["total_charge_variance"].astype(np.float64, copy=False)
    if cycles.shape != (cycles_count,) or not np.array_equal(cycles, np.arange(1, cycles_count + 1)):
        raise ValueError(f"{run_dir}: unexpected cycle labels {cycles}.")
    if sample_indices.shape != (samples,):
        raise ValueError(f"{run_dir}: unexpected sample_indices shape {sample_indices.shape}.")
    if entropy.shape != expected_shape or variance.shape != expected_shape:
        raise ValueError(
            f"{run_dir}: expected observable shape {expected_shape}, got entropy={entropy.shape}, variance={variance.shape}."
        )
    if not np.all(np.isfinite(entropy)) or not np.all(np.isfinite(variance)):
        raise FloatingPointError(f"{run_dir}: non-finite scalar observables.")
    if len(metrics) != samples * cycles_count:
        raise ValueError(f"{run_dir}: expected {samples * cycles_count} metric rows, got {len(metrics)}.")
    return {
        "entry": entry,
        "run_dir": run_dir,
        "summary": summary,
        "metrics": metrics,
        "cycles": cycles,
        "sample_indices": sample_indices,
        "seeds": arrays["seeds"].astype(np.uint32, copy=False),
        "total_entropy": entropy,
        "total_charge_variance": variance,
        "complete": True,
    }


def load_charge_sharpening_campaign(
    start: Path | str | None = None,
    *,
    campaign_id: str | None = None,
    require_complete: bool = True,
) -> dict[str, Any]:
    bundle_root = resolve_bundle_root(start)
    data_root = bundle_root / "cpu_data" / CAMPAIGN_NAME
    manifests: dict[str, Any] = {}
    runs: dict[str, dict[str, Any]] = {}
    resolved_campaign_id = _resolve_campaign_id(data_root, campaign_id, require_complete)

    for protocol in PROTOCOLS:
        root = _campaign_root(data_root, protocol, resolved_campaign_id)
        manifest = load_json(root / "campaign_manifest.json")
        manifests[protocol] = manifest
        for entry in manifest.get("results", []):
            config_id = str(entry["config_id"])
            run_dir = data_root / str(entry["run_dir_relative"])
            runs[config_id] = _load_run(run_dir, entry, require_complete=require_complete)

    payload = {
        "bundle_root": bundle_root,
        "data_root": data_root,
        "campaign_id": resolved_campaign_id,
        "manifests": manifests,
        "runs": runs,
    }
    payload["inventory_df"] = build_inventory_table(payload)
    payload["trajectory_df"] = build_trajectory_table(payload)
    payload["steady_state_df"] = build_steady_state_table(payload)
    payload["critical_diagnostics_df"] = build_critical_diagnostics_table(payload)
    return payload


def build_inventory_table(payload: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for config_id, run in payload["runs"].items():
        entry = run["entry"]
        rows.append(
            {
                "config_id": config_id,
                "protocol": entry["protocol"],
                "Nx": int(entry["Nx"]),
                "Ny": int(entry["Ny"]),
                "alpha_topological_region": float(entry["alpha_topological_region"]),
                "dw_truncation": bool(metadata_value(run, "dw_truncation", True)),
                "observable_region": str(metadata_value(run, "observable_region", "domain_wall_slab")),
                "cycles": int(entry["cycles"]),
                "samples_expected": int(entry["samples_expected"]),
                "samples_completed": int(entry["samples_completed"]),
                "complete": bool(run["complete"]),
                "run_dir": str(run["run_dir"]),
            }
        )
    return pd.DataFrame(rows).sort_values(["protocol", "Ny", "alpha_topological_region"]).reset_index(drop=True)


def build_trajectory_table(payload: dict[str, Any]) -> pd.DataFrame:
    frames = []
    for config_id, run in payload["runs"].items():
        if not run["complete"]:
            continue
        summary = run["summary"]
        samples = len(run["sample_indices"])
        cycles = len(run["cycles"])
        dw_truncation = bool(metadata_value(run, "dw_truncation", True))
        observable_region = str(metadata_value(run, "observable_region", "domain_wall_slab"))
        frames.append(
            pd.DataFrame(
                {
                    "config_id": np.repeat(config_id, samples * cycles),
                    "protocol": np.repeat(summary["protocol"], samples * cycles),
                    "Nx": np.repeat(int(summary["Nx"]), samples * cycles),
                    "Ny": np.repeat(int(summary["Ny"]), samples * cycles),
                    "alpha_topological_region": np.repeat(
                        float(summary["alpha_topological_region"]), samples * cycles
                    ),
                    "dw_truncation": np.repeat(dw_truncation, samples * cycles),
                    "observable_region": np.repeat(observable_region, samples * cycles),
                    "sample_index": np.repeat(run["sample_indices"], cycles),
                    "seed": np.repeat(run["seeds"], cycles),
                    "cycle": np.tile(run["cycles"], samples),
                    "total_entropy": run["total_entropy"].reshape(-1),
                    "total_charge_variance": run["total_charge_variance"].reshape(-1),
                }
            )
        )
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values(
        ["protocol", "Ny", "alpha_topological_region", "sample_index", "cycle"]
    ).reset_index(drop=True)


def build_steady_state_table(payload: dict[str, Any]) -> pd.DataFrame:
    trajectory_df = payload.get("trajectory_df")
    if trajectory_df is None:
        trajectory_df = build_trajectory_table(payload)
    if trajectory_df.empty:
        return pd.DataFrame()
    final = trajectory_df.loc[
        trajectory_df["cycle"]
        == trajectory_df.groupby("config_id")["cycle"].transform("max")
    ].copy()
    grouped = final.groupby(
        ["protocol", "Nx", "Ny", "alpha_topological_region", "dw_truncation", "observable_region"],
        sort=True,
    )
    rows = []
    for keys, group in grouped:
        protocol, nx, ny, alpha, dw_truncation, observable_region = keys
        rows.append(
            {
                "protocol": protocol,
                "Nx": int(nx),
                "Ny": int(ny),
                "alpha_topological_region": float(alpha),
                "dw_truncation": bool(dw_truncation),
                "observable_region": str(observable_region),
                "samples": len(group),
                "steady_state_cycle": int(group["cycle"].iloc[0]),
                "total_entropy_mean": float(group["total_entropy"].mean()),
                "total_entropy_std": float(group["total_entropy"].std(ddof=1)) if len(group) > 1 else 0.0,
                "total_charge_variance_mean": float(group["total_charge_variance"].mean()),
                "total_charge_variance_std": (
                    float(group["total_charge_variance"].std(ddof=1)) if len(group) > 1 else 0.0
                ),
                "sharpening_fraction": float(np.mean(group["total_charge_variance"] < SHARPENING_THRESHOLD)),
            }
        )
    return pd.DataFrame(rows)


def build_critical_diagnostics_table(payload: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for config_id, run in payload["runs"].items():
        if not run["complete"]:
            continue
        diagnostics = run["summary"].get("projector_diagnostics") or {}
        if diagnostics.get("critical_point_alpha_equals_2"):
            rows.append(
                {
                    "config_id": config_id,
                    "protocol": run["summary"]["protocol"],
                    "Ny": int(run["summary"]["Ny"]),
                    "dw_truncation": bool(metadata_value(run, "dw_truncation", True)),
                    "observable_region": str(metadata_value(run, "observable_region", "domain_wall_slab")),
                    **diagnostics,
                }
            )
    return pd.DataFrame(rows)
