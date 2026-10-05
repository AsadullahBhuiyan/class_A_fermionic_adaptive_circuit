from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


BUNDLE_NAME = "colab_charge_fluctuations"
CAMPAIGN_NAME = "purification_dynamics_maxmix"
REQUIRED_ARRAY_FILES = {
    "total_entropy": "total_entropy.npz",
    "total_charge_variance": "total_charge_variance.npz",
    "local_charge_cell_mean": "local_charge_cell_mean.npz",
    "local_charge_cell_variance": "local_charge_cell_variance.npz",
}


def load_json(path: Path | str) -> Any:
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def resolve_bundle_root(start: Path | str | None = None) -> Path:
    start_path = Path.cwd() if start is None else Path(start)
    candidates: list[Path] = []
    for base in [start_path, *start_path.parents]:
        candidates.extend([base, base / BUNDLE_NAME])

    seen: set[Path] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except FileNotFoundError:
            resolved = candidate.absolute()
        if resolved in seen:
            continue
        seen.add(resolved)
        if (resolved / "gpu_data" / CAMPAIGN_NAME).exists() and (resolved / "src").exists():
            return resolved
    raise FileNotFoundError(f"Could not locate {BUNDLE_NAME} with src/ and gpu_data/{CAMPAIGN_NAME}.")


def _load_npz_eager(path: Path | str) -> dict[str, Any]:
    with np.load(Path(path), allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _required_path(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _gpu_data_root(bundle_root: Path) -> Path:
    return bundle_root / "gpu_data"


def load_purification_campaign(start: Path | str | None = None) -> dict[str, Any]:
    bundle_root = resolve_bundle_root(start)
    gpu_data_root = _gpu_data_root(bundle_root)
    index_path = _required_path(gpu_data_root / "index.json")
    index_payload = load_json(index_path)

    campaigns = index_payload.get("campaigns", [])
    campaign_record = next(
        (
            record
            for record in campaigns
            if record.get("kind") == CAMPAIGN_NAME or record.get("name") == CAMPAIGN_NAME
        ),
        None,
    )
    if campaign_record is None:
        raise KeyError(f"Campaign {CAMPAIGN_NAME!r} not registered in {index_path}.")

    pointer_rel = campaign_record.get("manifest")
    if pointer_rel is None:
        raise KeyError(f"Campaign {CAMPAIGN_NAME!r} is missing its latest manifest pointer.")
    latest_campaign_path = _required_path(gpu_data_root / pointer_rel)
    latest_campaign = load_json(latest_campaign_path)

    manifest_rel = latest_campaign.get("campaign_manifest_path_relative") or latest_campaign.get("manifest")
    if manifest_rel is None:
        raise KeyError(f"{latest_campaign_path} is missing the campaign manifest path.")
    campaign_manifest_path = _required_path(gpu_data_root / manifest_rel)
    campaign_manifest = load_json(campaign_manifest_path)

    if campaign_manifest.get("campaign_name") != CAMPAIGN_NAME:
        raise ValueError(
            f"Expected campaign_name {CAMPAIGN_NAME!r}, got {campaign_manifest.get('campaign_name')!r}."
        )

    return {
        "bundle_root": bundle_root,
        "gpu_data_root": gpu_data_root,
        "index_path": index_path,
        "index": index_payload,
        "campaign_record": campaign_record,
        "latest_campaign_path": latest_campaign_path,
        "latest_campaign": latest_campaign,
        "campaign_manifest_path": campaign_manifest_path,
        "campaign_manifest": campaign_manifest,
    }


def load_purification_run(
    run_dir: Path | str,
    *,
    config_id: str | None = None,
    result_entry: dict[str, Any] | None = None,
) -> dict[str, Any]:
    run_dir = _required_path(Path(run_dir))
    summary_path = _required_path(run_dir / "run_summary.json")
    metrics_path = _required_path(run_dir / "scalar_metrics.csv")

    summary = load_json(summary_path)
    metrics_df = pd.read_csv(metrics_path)
    if metrics_df.empty:
        raise ValueError(f"{metrics_path} is empty.")

    inferred_config_id = str(metrics_df["config_id"].iloc[0])
    config_id = inferred_config_id if config_id is None else str(config_id)
    if inferred_config_id != config_id:
        raise ValueError(f"Expected config_id {config_id!r}, got {inferred_config_id!r} in {metrics_path}.")

    arrays: dict[str, dict[str, Any]] = {}
    file_paths: dict[str, Path] = {
        "run_summary": summary_path,
        "scalar_metrics": metrics_path,
    }
    for key, filename in REQUIRED_ARRAY_FILES.items():
        array_path = _required_path(run_dir / filename)
        arrays[key] = _load_npz_eager(array_path)
        file_paths[key] = array_path

    nx = int(metrics_df["Nx"].iloc[0])
    ny = int(metrics_df["Ny"].iloc[0])
    protocol = str(metrics_df["protocol"].iloc[0])
    samples_actual = int(summary["samples_actual"])
    cycles = int(summary["cycles"])

    if metrics_df["sample_index"].nunique() != samples_actual:
        raise ValueError(f"{config_id}: metrics sample count does not match run summary.")
    if metrics_df["cycle_label"].nunique() != cycles:
        raise ValueError(f"{config_id}: metrics cycle count does not match run summary.")
    if len(metrics_df) != samples_actual * cycles:
        raise ValueError(f"{config_id}: expected {samples_actual * cycles} metric rows, got {len(metrics_df)}.")

    total_entropy = arrays["total_entropy"]["total_entropy"]
    total_charge_variance = arrays["total_charge_variance"]["total_charge_variance"]
    local_charge_cell_mean = arrays["local_charge_cell_mean"]["local_charge_cell_mean"]
    local_charge_cell_variance = arrays["local_charge_cell_variance"]["local_charge_cell_variance"]

    if total_entropy.shape != (samples_actual, cycles):
        raise ValueError(f"{config_id}: unexpected total_entropy shape {total_entropy.shape}.")
    if total_charge_variance.shape != (samples_actual, cycles):
        raise ValueError(f"{config_id}: unexpected total_charge_variance shape {total_charge_variance.shape}.")
    if local_charge_cell_mean.shape != (samples_actual, cycles, nx, ny):
        raise ValueError(f"{config_id}: unexpected local_charge_cell_mean shape {local_charge_cell_mean.shape}.")
    if local_charge_cell_variance.shape != (samples_actual, cycles, nx, ny):
        raise ValueError(
            f"{config_id}: unexpected local_charge_cell_variance shape {local_charge_cell_variance.shape}."
        )

    payload = {
        "config_id": config_id,
        "geometry_key": str(metrics_df["geometry_key"].iloc[0]),
        "protocol": protocol,
        "Nx": nx,
        "Ny": ny,
        "summary": summary,
        "metrics_df": metrics_df,
        "arrays": arrays,
        "paths": file_paths,
    }
    if result_entry is not None:
        payload["result_entry"] = result_entry
    return payload


def load_all_purification_runs(start: Path | str | None = None) -> dict[str, Any]:
    campaign = load_purification_campaign(start)
    bundle_root = campaign["bundle_root"]
    gpu_data_root = campaign["gpu_data_root"]
    manifest = campaign["campaign_manifest"]

    runs: dict[str, dict[str, Any]] = {}
    for result_entry in manifest.get("results", []):
        config_id = str(result_entry["config_id"])
        run_dir = _required_path(gpu_data_root / str(result_entry["run_dir_relative"]))
        runs[config_id] = load_purification_run(
            run_dir,
            config_id=config_id,
            result_entry=result_entry,
        )

    return {
        "bundle_root": bundle_root,
        "campaign": campaign,
        "runs": runs,
    }


def build_run_inventory_table(payload: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for config_id, run_payload in payload["runs"].items():
        summary = run_payload["summary"]
        result_entry = run_payload.get("result_entry", {})
        rows.append(
            {
                "config_id": config_id,
                "Nx": int(run_payload["Nx"]),
                "Ny": int(run_payload["Ny"]),
                "protocol": str(run_payload["protocol"]),
                "samples_actual": int(summary["samples_actual"]),
                "cycles": int(summary["cycles"]),
                "scalar_metrics_bytes": result_entry.get("scalar_metrics_bytes"),
                "total_entropy_bytes": result_entry.get("total_entropy_bytes"),
                "total_charge_variance_bytes": result_entry.get("total_charge_variance_bytes"),
                "local_charge_cell_mean_bytes": result_entry.get("local_charge_cell_mean_bytes"),
                "local_charge_cell_variance_bytes": result_entry.get("local_charge_cell_variance_bytes"),
                "saved_total_bytes": result_entry.get("saved_total_bytes", summary.get("saved_total_bytes")),
            }
        )
    return pd.DataFrame(rows).sort_values(["Ny", "protocol"]).reset_index(drop=True)


__all__ = [
    "build_run_inventory_table",
    "load_all_purification_runs",
    "load_purification_campaign",
    "load_purification_run",
    "resolve_bundle_root",
]
