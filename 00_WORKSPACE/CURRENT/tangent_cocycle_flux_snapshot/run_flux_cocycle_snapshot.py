#!/usr/bin/env python3
"""Save a scale-separated full tangent cocycle around a boundary-flux loop."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fgtn.classA_U1FGTN import classA_U1FGTN
from fgtn.diagnostics.io import save_npz_atomic, write_json_atomic

CANONICAL_ENTRY_POINT = "classA_U1FGTN.run_markov_circuit"
DEFAULT_CONFIG = Path(__file__).with_name("reference_config.json")


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _canonical_json(payload: Any) -> str:
    return json.dumps(_json_ready(payload), sort_keys=True, separators=(",", ":"))


def _record_checksum(record: list[dict[str, Any]]) -> str:
    return hashlib.sha256(_canonical_json(record).encode("utf-8")).hexdigest()


def _git_metadata() -> dict[str, Any]:
    def run(*command: str) -> str:
        result = subprocess.run(
            command, cwd=REPO_ROOT, text=True, capture_output=True, check=False
        )
        return result.stdout.strip()

    return {
        "commit": run("git", "rev-parse", "HEAD"),
        "dirty": bool(run("git", "status", "--short")),
    }


class TrajectoryRecorder:
    def __init__(self) -> None:
        self.entries: list[dict[str, Any]] = []

    def __call__(
        self,
        *,
        cycle: int,
        site_id: int,
        branch_events: Any,
        branch_log_weight: float,
        measurement_log_weight: float,
        correction_log_weight: float,
        cumulative_log_weight: float,
        forced_postselect: bool,
        **_: Any,
    ) -> None:
        self.entries.append(
            {
                "cycle": int(cycle),
                "site_id": int(site_id),
                "branch_events": [dict(event) for event in branch_events],
                "branch_log_weight": float(branch_log_weight),
                "measurement_log_weight": float(measurement_log_weight),
                "correction_log_weight": float(correction_log_weight),
                "cumulative_log_weight": float(cumulative_log_weight),
                "forced_postselect": bool(forced_postselect),
            }
        )


class ReplayAudit:
    def __init__(self) -> None:
        self.site_count = 0
        self.event_count = 0
        self.cumulative_log_weight = 0.0
        self.min_selected_probability = np.inf

    def __call__(
        self,
        *,
        branch_events: Any,
        cumulative_log_weight: float,
        **_: Any,
    ) -> None:
        self.site_count += 1
        self.cumulative_log_weight = float(cumulative_log_weight)
        for event in branch_events:
            self.event_count += 1
            selected = float(np.exp(float(event["log_weight"])))
            self.min_selected_probability = min(
                self.min_selected_probability, selected
            )

    def payload(self) -> dict[str, Any]:
        return {
            "replayed_site_count": int(self.site_count),
            "replayed_event_count": int(self.event_count),
            "cumulative_log_path_weight": float(self.cumulative_log_weight),
            "min_forced_branch_probability": (
                None
                if not np.isfinite(self.min_selected_probability)
                else float(self.min_selected_probability)
            ),
        }


class FinalCocycleRecorder:
    def __init__(self, observation_cycles: int) -> None:
        self.observation_cycles = int(observation_cycles)
        self.payload: dict[str, Any] | None = None

    def __call__(
        self,
        *,
        lyapunov_cycle: int,
        lyapunov_frame: Any,
        lyapunov_core_hat: Any,
        lyapunov_core_log_scale: Any,
        lyapunov_log_diag: Any,
        lyapunov_null_counts: Any,
        lyapunov_active_mask: Any,
        lyapunov_min_branch_probability: Any,
        lyapunov_min_abs_born_denominator: Any,
        lyapunov_invalid_branch_count: Any,
        lyapunov_failure_records: Any = (),
        **_: Any,
    ) -> None:
        if int(lyapunov_cycle) != self.observation_cycles:
            return
        self.payload = {
            "Q": np.array(lyapunov_frame[0], dtype=np.complex128, copy=True),
            "core_hat": np.array(
                lyapunov_core_hat[0], dtype=np.complex128, copy=True
            ),
            "core_log_scale": float(lyapunov_core_log_scale[0]),
            "log_qr_diagonal": np.array(
                lyapunov_log_diag[0], dtype=np.float64, copy=True
            ),
            "qr_null_count": int(lyapunov_null_counts[0]),
            "active": bool(lyapunov_active_mask[0]),
            "tangent_min_branch_probability": float(
                lyapunov_min_branch_probability[0]
            ),
            "tangent_min_abs_born_denominator": float(
                lyapunov_min_abs_born_denominator[0]
            ),
            "tangent_invalid_branch_count": int(
                lyapunov_invalid_branch_count[0]
            ),
            "tangent_failure_records": [
                dict(record) for record in lyapunov_failure_records
            ],
        }


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    required = {
        "nx",
        "ny",
        "nshell",
        "burn_cycles",
        "observation_cycles",
        "twist_count",
        "root_seed",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Configuration is missing required keys: {missing}.")
    if int(config["twist_count"]) < 2:
        raise ValueError("twist_count must be at least two to include 0 and 2pi.")
    return config


def _model(config: dict[str, Any], twist: float) -> classA_U1FGTN:
    model = classA_U1FGTN(
        int(config["nx"]),
        int(config["ny"]),
        DW=bool(config["DW"]),
        nshell=config["nshell"],
        alpha_1=float(config["alpha_1"]),
        alpha_2=float(config["alpha_2"]),
        trial_orbitals=str(config["trial_orbitals"]),
        dw_truncation=bool(config["dw_truncation"]),
        twist_y=float(twist),
    )
    model.construct_OW_projectors(
        nshell=config["nshell"],
        DW=bool(config["DW"]),
        trial_orbitals=str(config["trial_orbitals"]),
        dw_truncation=bool(config["dw_truncation"]),
        twist_y=float(twist),
    )
    return model


def _run_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "G_history": False,
        "progress": True,
        "cycles": int(config["burn_cycles"]) + int(config["observation_cycles"]),
        "samples": 1,
        "parallelize_samples": False,
        "init_mode": str(config["init_mode"]),
        "save": False,
        "save_init": False,
        "sequence": str(config["sequence"]),
        "meas_slab_only": bool(config["meas_slab_only"]),
        "random_seed": int(config["root_seed"]),
        "physical_covariance_update": str(config["physical_covariance_update"]),
        "postselect": False,
        "perfect_correction": bool(config["perfect_correction"]),
    }


def _save_reference(
    run_dir: Path, config: dict[str, Any]
) -> tuple[list[dict[str, Any]], np.ndarray, str]:
    recorder = TrajectoryRecorder()
    model = _model(config, 0.0)
    result = model.run_markov_circuit(
        trajectory_weight_observer=recorder,
        **_run_kwargs(config),
    )
    record = recorder.entries
    checksum = _record_checksum(record)
    final_g = np.asarray(result["G_final"][0], dtype=np.complex128)
    metadata = {
        "canonical_dynamics_entry_point": CANONICAL_ENTRY_POINT,
        "record_checksum_sha256": checksum,
        "site_count": len(record),
        "total_cycles": int(config["burn_cycles"]) + int(config["observation_cycles"]),
        "config": config,
    }
    save_npz_atomic(
        run_dir / "reference_record.npz",
        record_json=np.asarray(_canonical_json(record)),
        reference_final_G=final_g,
        metadata_json=np.asarray(_canonical_json(metadata)),
    )
    return record, final_g, checksum


def _load_reference(
    run_dir: Path,
) -> tuple[list[dict[str, Any]], np.ndarray, str]:
    with np.load(run_dir / "reference_record.npz", allow_pickle=False) as data:
        record = json.loads(str(data["record_json"].item()))
        final_g = np.array(data["reference_final_G"], copy=True)
        metadata = json.loads(str(data["metadata_json"].item()))
    checksum = _record_checksum(record)
    if checksum != metadata["record_checksum_sha256"]:
        raise ValueError("Saved reference-record checksum does not match its contents.")
    return record, final_g, checksum


def _twist_filename(index: int) -> str:
    return f"twist_{int(index):03d}.npz"


def _run_twist(
    *,
    run_dir: Path,
    config: dict[str, Any],
    twist_index: int,
    twist: float,
    record: list[dict[str, Any]],
    record_checksum: str,
    reference_final_g: np.ndarray,
) -> dict[str, Any]:
    model = _model(config, twist)
    dimension = 2 * int(config["nx"]) * int(config["ny"])
    cocycle = FinalCocycleRecorder(int(config["observation_cycles"]))
    audit = ReplayAudit()
    started = time.perf_counter()
    result = model.run_markov_circuit(
        trajectory_replay=record,
        trajectory_replay_probability_tol=float(config["replay_probability_tol"]),
        trajectory_weight_observer=audit,
        lyapunov_frame_observer=cocycle,
        lyapunov_initial_frame=np.eye(dimension, dtype=np.complex128),
        lyapunov_start_cycle=int(config["burn_cycles"]) + 1,
        lyapunov_full_space=True,
        lyapunov_track_restricted_core=True,
        lyapunov_track_record_fisher=False,
        lyapunov_singular_tol=float(config["singular_tol"]),
        lyapunov_failure_mode="raise",
        **_run_kwargs(config),
    )
    if cocycle.payload is None:
        raise RuntimeError("The final cocycle callback was not emitted.")

    payload = cocycle.payload
    q = payload.pop("Q")
    core_hat = payload.pop("core_hat")
    core_log_scale = float(payload.pop("core_log_scale"))
    product_hat = q @ core_hat
    singular_values_hat = np.linalg.svd(core_hat, compute_uv=False)
    threshold = float(config["singular_tol"]) * max(
        1.0, float(singular_values_hat[0]) if singular_values_hat.size else 1.0
    )
    valid = np.isfinite(singular_values_hat) & (singular_values_hat > threshold)
    rank = int(np.count_nonzero(valid))
    log_singular_values = np.full(singular_values_hat.shape, -np.inf, dtype=np.float64)
    log_singular_values[valid] = core_log_scale + np.log(singular_values_hat[valid])
    condition_number = (
        np.inf
        if rank == 0
        else float(singular_values_hat[0] / singular_values_hat[rank - 1])
    )
    final_g = np.asarray(result["G_final"][0], dtype=np.complex128)
    phi0_replay_error = (
        float(np.linalg.norm(final_g - reference_final_g, ord="fro"))
        if int(twist_index) == 0
        else np.nan
    )
    metadata = {
        "canonical_dynamics_entry_point": CANONICAL_ENTRY_POINT,
        "twist_index": int(twist_index),
        "twist": float(twist),
        "record_checksum_sha256": record_checksum,
        "root_seed": int(config["root_seed"]),
        "config": config,
        "git": _git_metadata(),
        "elapsed_seconds": float(time.perf_counter() - started),
        "dimension": dimension,
        "rank": rank,
        "condition_number_on_retained_rank": condition_number,
        "phi0_reference_final_G_frobenius_error": phi0_replay_error,
        **audit.payload(),
        **payload,
    }
    save_npz_atomic(
        run_dir / _twist_filename(twist_index),
        Q=q,
        core_hat=core_hat,
        core_log_scale=np.asarray(core_log_scale),
        product_hat=product_hat,
        final_G=final_g,
        singular_values_hat=singular_values_hat,
        log_singular_values=log_singular_values,
        numerical_rank=np.asarray(rank, dtype=np.int64),
        metadata_json=np.asarray(_canonical_json(metadata)),
    )
    return metadata


def _orbital_gauge(nx: int, ny: int) -> np.ndarray:
    y = np.repeat(np.arange(ny, dtype=np.float64), 2 * nx)
    return np.exp(2j * np.pi * y / float(ny))


def _write_closure_diagnostic(run_dir: Path, config: dict[str, Any]) -> None:
    last_index = int(config["twist_count"]) - 1
    first_path = run_dir / _twist_filename(0)
    last_path = run_dir / _twist_filename(last_index)
    if not first_path.exists() or not last_path.exists():
        return
    with np.load(first_path, allow_pickle=False) as first, np.load(
        last_path, allow_pickle=False
    ) as last:
        first_product = np.array(first["product_hat"], copy=True)
        last_product = np.array(last["product_hat"], copy=True)
        first_scale = float(first["core_log_scale"])
        last_scale = float(last["core_log_scale"])
        first_g = np.array(first["final_G"], copy=True)
        last_g = np.array(last["final_G"], copy=True)
    gauge = _orbital_gauge(int(config["nx"]), int(config["ny"]))
    gauged_product = gauge[:, None] * first_product * gauge.conj()[None, :]
    gauged_g = gauge[:, None] * first_g * gauge.conj()[None, :]
    product_den = max(float(np.linalg.norm(last_product, ord="fro")), 1e-300)
    g_den = max(float(np.linalg.norm(last_g, ord="fro")), 1e-300)
    write_json_atomic(
        run_dir / "closure_diagnostic.json",
        {
            "gauge_convention": "D[y]=exp(+2pi*i*y/Ny); expected X(2pi)=D X(0) D^dagger",
            "normalized_product_relative_frobenius_error": float(
                np.linalg.norm(last_product - gauged_product, ord="fro") / product_den
            ),
            "core_log_scale_difference": float(last_scale - first_scale),
            "final_G_relative_frobenius_error": float(
                np.linalg.norm(last_g - gauged_g, ord="fro") / g_den
            ),
        },
    )


def _new_run_dir(output_root: Path, config: dict[str, Any]) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stem = (
        f"N{int(config['nx'])}x{int(config['ny'])}_B{int(config['burn_cycles'])}"
        f"_C{int(config['observation_cycles'])}_P{int(config['twist_count'])}_{stamp}"
    )
    run_dir = output_root / stem
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--output-root", type=Path, default=Path(__file__).with_name("outputs")
    )
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Override to N=4x4, one burn cycle, one observed cycle, and three twists.",
    )
    args = parser.parse_args()

    config = _load_config(args.config.resolve())
    if args.smoke:
        config.update(
            {
                "nx": 4,
                "ny": 4,
                "burn_cycles": 1,
                "observation_cycles": 1,
                "twist_count": 3,
                "sequence": "raster_y",
            }
        )
    run_dir = (
        args.resume.resolve()
        if args.resume is not None
        else _new_run_dir(args.output_root.resolve(), config)
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "config.json"
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as handle:
            saved_config = json.load(handle)
        if saved_config != config:
            raise ValueError("Resume configuration differs from the saved run configuration.")
    else:
        write_json_atomic(config_path, config)

    manifest_path = run_dir / "manifest.json"
    manifest = {
        "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "canonical_dynamics_entry_point": CANONICAL_ENTRY_POINT,
        "config": config,
        "git": _git_metadata(),
        "completed_twist_indices": [],
        "failures": [],
    }
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        manifest["status"] = "running"
    write_json_atomic(manifest_path, manifest)

    if (run_dir / "reference_record.npz").exists():
        record, reference_final_g, checksum = _load_reference(run_dir)
    else:
        record, reference_final_g, checksum = _save_reference(run_dir, config)
    manifest["reference_record_checksum_sha256"] = checksum
    manifest["reference_site_count"] = len(record)
    write_json_atomic(manifest_path, manifest)

    twists = np.linspace(0.0, 2.0 * np.pi, int(config["twist_count"]))
    for index, twist in enumerate(twists):
        output_path = run_dir / _twist_filename(index)
        if output_path.exists():
            if index not in manifest["completed_twist_indices"]:
                manifest["completed_twist_indices"].append(index)
            continue
        try:
            metadata = _run_twist(
                run_dir=run_dir,
                config=config,
                twist_index=index,
                twist=float(twist),
                record=record,
                record_checksum=checksum,
                reference_final_g=reference_final_g,
            )
        except Exception as exc:
            manifest["status"] = "failed"
            manifest["failures"].append(
                {
                    "twist_index": int(index),
                    "twist": float(twist),
                    "type": type(exc).__name__,
                    "message": str(exc),
                }
            )
            write_json_atomic(manifest_path, manifest)
            raise
        manifest["completed_twist_indices"].append(index)
        manifest.setdefault("twist_summaries", []).append(metadata)
        write_json_atomic(manifest_path, manifest)

    _write_closure_diagnostic(run_dir, config)
    manifest["completed_twist_indices"] = sorted(
        set(int(index) for index in manifest["completed_twist_indices"])
    )
    manifest["status"] = "complete"
    manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
    write_json_atomic(manifest_path, manifest)
    print(f"Completed flux cocycle snapshot: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
