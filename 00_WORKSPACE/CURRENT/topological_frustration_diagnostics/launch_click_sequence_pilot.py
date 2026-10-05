#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = (
    REPO_ROOT
    / "topological_frustration_diagnostics"
    / "notebooks"
    / "analyze_local_gain_loss_click_sequences.ipynb"
)
CLI = REPO_ROOT / "topological_frustration_diagnostics" / "run_cpu.py"
DEFAULT_PARENT = (
    REPO_ROOT
    / "topological_frustration_diagnostics"
    / "results"
    / "local_gain_loss_clicks"
)
SCHEDULES = ("raster_y", "random")
CANONICAL_ENTRY_POINT = "classA_U1FGTN.run_markov_circuit"


class PilotValidationError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def wait_for_tmux_session(session: str, *, poll_seconds: int) -> None:
    """Defer without interrupting a resource-conflicting tmux campaign."""
    session = str(session).strip()
    if not session:
        return
    while subprocess.run(
        ["tmux", "has-session", "-t", session],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0:
        print(f"{utc_now()} waiting for tmux session {session!r}", flush=True)
        time.sleep(int(poll_seconds))
    print(f"{utc_now()} tmux gate {session!r} cleared", flush=True)


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    temporary.replace(path)


def stage_run_directory(stage: Path) -> Path:
    matches = sorted(stage.rglob("activity_raw.npz"))
    if len(matches) != 1:
        raise PilotValidationError(
            f"Expected exactly one activity case under {stage}; found {len(matches)}."
        )
    return matches[0].parent


def validate_stage(
    stage: Path,
    *,
    schedule: str,
    samples: int,
    cycles: int,
    burn_in: int,
) -> dict[str, Any]:
    run = stage_run_directory(stage)
    required = (
        "activity_raw.npz",
        "activity_analysis.npz",
        "click_sequence_analysis.npz",
        "click_events.parquet",
        "unit_cell_motifs.parquet",
        "click_sequence_candidates.csv",
        "run_summary.json",
        "scalar_metrics.csv",
    )
    missing = [name for name in required if not (run / name).is_file()]
    if missing:
        raise PilotValidationError(f"Missing outputs in {run}: {missing}.")

    summary = json.loads((run / "run_summary.json").read_text())
    config = summary["config"]
    expected_nx = 4 if samples == 2 and cycles == 2 else 20
    half = expected_nx // 2
    slab_half_width = max(1, expected_nx // 3)
    expected_wall_x = [max(0, half - slab_half_width), min(expected_nx, half + slab_half_width + 1) - 1]
    expected_config = {
        "Nx": expected_nx,
        "Ny": 6 if samples == 2 and cycles == 2 else 24,
        "samples": samples,
        "cycles": cycles,
        "burn_in": burn_in,
        "geometry": "dw",
        "DW": True,
        "dw_truncation": True,
        "meas_slab_only": True,
        "wall_x": expected_wall_x,
        "alpha_1": 1,
        "alpha_2": 30,
        "nshell": 1,
        "protocol": "perfect_correction",
        "sequence": schedule,
        "trial_orbitals": "X",
        "click_sequences": True,
    }
    for key, expected in expected_config.items():
        if config.get(key) != expected:
            raise PilotValidationError(
                f"Config mismatch for {key}: expected {expected!r}, got {config.get(key)!r}."
            )
    git_state = summary.get("git", {})
    if not isinstance(git_state.get("commit"), str) or not isinstance(
        git_state.get("dirty"), bool
    ):
        raise PilotValidationError("Git commit/dirty state is incomplete.")
    parallel = summary.get("parallel_execution", {})
    expected_parallel = {
        "enabled": True,
        "requested_cpu_budget": samples,
        "workers": samples,
        "threads_per_worker": 1,
    }
    for key, expected in expected_parallel.items():
        if parallel.get(key) != expected:
            raise PilotValidationError(
                f"CPU policy mismatch for {key}: expected {expected!r}, "
                f"got {parallel.get(key)!r}."
            )
    if int(parallel.get("affinity_cpus", 0)) < samples:
        raise PilotValidationError("CPU affinity is smaller than the worker count.")
    if summary.get("canonical_dynamics_entry_point") != CANONICAL_ENTRY_POINT:
        raise PilotValidationError("Canonical CPU entry point was not recorded.")

    with np.load(run / "activity_raw.npz") as raw:
        transfer = raw["transfer_Y"]
        defect = raw["defect_X"]
        valid = raw["valid"]
        if not np.array_equal(raw["wall_x"], np.asarray(expected_wall_x)):
            raise PilotValidationError("Raw-data wall positions do not match the run metadata.")
        probability = raw["success_probability"][valid]
        if transfer.shape[:2] != (samples, cycles):
            raise PilotValidationError(f"Unexpected raw shape {transfer.shape}.")
        if not np.all(valid):
            raise PilotValidationError("The activity record contains missing channel visits.")
        if not np.array_equal(defect, np.abs(transfer)):
            raise PilotValidationError("Perfect-correction identity X=|Y| failed.")
        allowed = np.asarray((-1, 1, -1, 1), dtype=np.int8)
        if not np.all((transfer == 0) | (transfer == allowed)):
            raise PilotValidationError("A channel contains a forbidden transfer direction.")
        if probability.size and (
            not np.all(np.isfinite(probability))
            or np.min(probability) < 0.0
            or np.max(probability) > 1.0
        ):
            raise PilotValidationError("Invalid Born success probabilities.")

    events = pd.read_parquet(run / "click_events.parquet")
    motifs = pd.read_parquet(run / "unit_cell_motifs.parquet")
    expected_events = int(np.prod(transfer.shape))
    expected_motifs = int(np.prod(transfer.shape[:-1]))
    if len(events) != expected_events or len(motifs) != expected_motifs:
        raise PilotValidationError(
            f"Tidy row mismatch: events={len(events)}/{expected_events}, "
            f"motifs={len(motifs)}/{expected_motifs}."
        )
    if set(events["trial_orbital"]) != {"A", "B"}:
        raise PilotValidationError("A/B Pauli-eigenvector labels are incomplete.")
    if set(events.loc[events["trial_orbital"] == "A", "trial_pauli_eigenvalue"]) != {1}:
        raise PilotValidationError("Trial orbital A is not labeled as the +1 Pauli eigenvector.")
    if set(events.loc[events["trial_orbital"] == "B", "trial_pauli_eigenvalue"]) != {-1}:
        raise PilotValidationError("Trial orbital B is not labeled as the -1 Pauli eigenvector.")

    with np.load(run / "click_sequence_analysis.npz") as analysis:
        if int(analysis["burn_in"]) != burn_in:
            raise PilotValidationError("Sequence-analysis burn-in mismatch.")
        if analysis["temporal_pair_counts"].shape[-1] != 16**2:
            raise PilotValidationError("Temporal pair state space is not 16^2.")
        if analysis["spatial_triplet_counts"].shape[-1] != 16**3:
            raise PilotValidationError("Spatial triplet state space is not 16^3.")

    return {
        "schedule": schedule,
        "run_directory": str(run),
        "sample_seeds": summary.get("sample_seeds"),
        "suite_worker_seeds": summary.get("suite_worker_seeds"),
        "events": len(events),
        "motifs": len(motifs),
        "metrics": summary.get("metrics", {}),
    }


def command_for(
    args: argparse.Namespace,
    *,
    schedule: str,
    stage: Path,
    samples: int,
    cycles: int,
    burn_in: int,
    nx: int,
    ny: int,
) -> list[str]:
    return [
        sys.executable,
        str(CLI),
        "activity",
        "--output-dir",
        str(stage),
        "--geometry",
        "dw",
        "--nx",
        str(nx),
        "--ny",
        str(ny),
        "--nshell",
        "1",
        "--samples",
        str(samples),
        "--cycles",
        str(cycles),
        "--burn-in",
        str(burn_in),
        "--protocol",
        "perfect_correction",
        "--sequence",
        schedule,
        "--init-mode",
        "maxmix",
        "--seed",
        str(args.seed),
        "--bootstrap-samples",
        str(args.activity_bootstraps),
        "--cpu-budget",
        str(args.cpu_budget),
        "--workers",
        str(args.workers),
        "--threads-per-worker",
        "1",
        "--memory-fraction",
        str(args.memory_fraction),
        "--click-sequences",
        "--sequence-permutations",
        str(args.sequence_permutations),
        "--sequence-bootstraps",
        str(args.sequence_bootstraps),
        "--sequence-min-support",
        str(args.sequence_min_support),
    ]


def run_stage(command: list[str], *, stage: Path) -> dict[str, Any]:
    stage.mkdir(parents=True, exist_ok=False)
    log_path = stage / "stage.log"
    marker_failed = stage / "_FAILED"
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    environment["OMP_NUM_THREADS"] = "1"
    environment["OPENBLAS_NUM_THREADS"] = "1"
    environment["MKL_NUM_THREADS"] = "1"
    environment["NUMEXPR_NUM_THREADS"] = "1"
    environment.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-local-click-pilot")
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"command={json.dumps(command)}\n")
        log.flush()
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if completed.returncode != 0:
        marker_failed.write_text(f"{utc_now()} return_code={completed.returncode}\n")
        raise RuntimeError(f"Stage failed with return code {completed.returncode}: {stage}")
    return {"command": command, "log": str(log_path), "return_code": 0}


def execute_notebook(root: Path) -> dict[str, Any]:
    """Execute an output copy only after both schedules validate successfully."""
    output_name = "analyze_local_gain_loss_click_sequences.executed.ipynb"
    output_path = root / output_name
    log_path = root / "notebook.log"
    command = [
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        str(NOTEBOOK),
        "--output-dir",
        str(root),
        "--output",
        output_name,
        "--ExecutePreprocessor.timeout=900",
    ]
    environment = os.environ.copy()
    environment["CLICK_CAMPAIGN_ROOT"] = str(root)
    environment.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-local-click-pilot")
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"command={json.dumps(command)}\n")
        log.flush()
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if completed.returncode != 0 or not output_path.is_file():
        raise RuntimeError(
            f"Notebook execution failed with return code {completed.returncode}; see {log_path}."
        )
    return {
        "command": command,
        "log": str(log_path),
        "output_notebook": str(output_path),
        "return_code": 0,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sequential raster/random local click pilot")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--cpu-budget", type=int, default=10)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--memory-fraction", type=float, default=0.7)
    parser.add_argument("--activity-bootstraps", type=int, default=1000)
    parser.add_argument("--sequence-permutations", type=int, default=1000)
    parser.add_argument("--sequence-bootstraps", type=int, default=1000)
    parser.add_argument("--sequence-min-support", type=int, default=20)
    parser.add_argument("--wait-for-tmux-session")
    parser.add_argument("--wait-poll-seconds", type=int, default=60)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.workers <= 0 or args.cpu_budget <= 0:
        raise ValueError("workers and cpu-budget must be positive.")
    if args.workers > args.cpu_budget:
        raise ValueError("workers cannot exceed cpu-budget for single-thread trajectories.")
    if args.smoke:
        nx, ny, samples, cycles, burn_in = 4, 6, 2, 2, 0
        args.workers = min(args.workers, samples)
        args.cpu_budget = min(args.cpu_budget, samples)
    else:
        nx, ny, samples, cycles, burn_in = 20, 24, 10, 48, 24

    if args.output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        label = "smoke" if args.smoke else "production"
        root = DEFAULT_PARENT / f"{stamp}_N{nx}x{ny}_S{samples}_C{cycles}_{label}"
    else:
        root = args.output_dir.resolve()
    if args.wait_poll_seconds <= 0:
        raise ValueError("wait-poll-seconds must be positive.")
    if args.wait_for_tmux_session:
        wait_for_tmux_session(
            args.wait_for_tmux_session,
            poll_seconds=args.wait_poll_seconds,
        )
    root.mkdir(parents=True, exist_ok=False)

    slab_half_width = max(1, nx // 3)
    wall_x = [max(0, nx // 2 - slab_half_width), min(nx, nx // 2 + slab_half_width + 1) - 1]
    cpu_affinity = sorted(int(cpu) for cpu in os.sched_getaffinity(0))

    manifest = {
        "created_utc": utc_now(),
        "canonical_dynamics_entry_point": CANONICAL_ENTRY_POINT,
        "schedules": list(SCHEDULES),
        "execution_order": "serial schedules; parallel independent trajectories within a schedule",
        "Nx": nx,
        "Ny": ny,
        "samples_per_schedule": samples,
        "cycles": cycles,
        "burn_in": burn_in,
        "DW": True,
        "dw_truncation": True,
        "meas_slab_only": True,
        "wall_x": wall_x,
        "alpha_1": 1,
        "alpha_2": 30,
        "nshell": 1,
        "trial_orbitals": "X",
        "trial_orbital_semantics": {
            "A": "+1 eigenvector of the selected Pauli X",
            "B": "-1 eigenvector of the selected Pauli X",
        },
        "protocol": "perfect_correction",
        "init_mode": "maxmix",
        "seed": args.seed,
        "cpu_budget": args.cpu_budget,
        "workers": args.workers,
        "threads_per_worker": 1,
        "cpu_affinity": cpu_affinity,
        "sequence_permutations": args.sequence_permutations,
        "sequence_bootstraps": args.sequence_bootstraps,
        "sequence_min_support": args.sequence_min_support,
        "wait_for_tmux_session": args.wait_for_tmux_session,
        "wait_poll_seconds": args.wait_poll_seconds,
        "smoke": bool(args.smoke),
    }
    write_json_atomic(root / "campaign_manifest.json", manifest)

    stage_records: list[dict[str, Any]] = []
    notebook_record: dict[str, Any] | None = None
    try:
        for schedule in SCHEDULES:
            stage = root / schedule
            command = command_for(
                args,
                schedule=schedule,
                stage=stage,
                samples=samples,
                cycles=cycles,
                burn_in=burn_in,
                nx=nx,
                ny=ny,
            )
            execution = run_stage(command, stage=stage)
            validation = validate_stage(
                stage,
                schedule=schedule,
                samples=samples,
                cycles=cycles,
                burn_in=burn_in,
            )
            (stage / "_SUCCESS").write_text(utc_now() + "\n")
            stage_records.append({**execution, **validation})
        if stage_records[0]["sample_seeds"] != stage_records[1]["sample_seeds"]:
            raise PilotValidationError("Raster and random schedules did not share sample seeds.")
        if stage_records[0]["suite_worker_seeds"] != stage_records[1]["suite_worker_seeds"]:
            raise PilotValidationError("Raster and random schedules did not share suite worker seeds.")
        notebook_record = execute_notebook(root)
    except Exception as exc:
        (root / "_FAILED").write_text(f"{utc_now()} {type(exc).__name__}: {exc}\n")
        write_json_atomic(
            root / "campaign_status.json",
            {"status": "failed", "stages": stage_records, "notebook": notebook_record},
        )
        raise

    write_json_atomic(
        root / "campaign_status.json",
        {
            "status": "complete",
            "completed_utc": utc_now(),
            "stages": stage_records,
            "notebook": notebook_record,
        },
    )
    (root / "campaign_complete.txt").write_text(utc_now() + "\n")
    print(json.dumps({"status": "complete", "output_dir": str(root)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
