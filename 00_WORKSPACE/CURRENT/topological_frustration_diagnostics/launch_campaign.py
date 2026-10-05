#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import psutil


REPO_ROOT = Path(__file__).resolve().parents[1]
CLI = REPO_ROOT / "topological_frustration_diagnostics" / "run_cpu.py"
DEFAULT_RESULTS = REPO_ROOT / "topological_frustration_diagnostics" / "results" / "campaigns"
ACTIVITY_REGIONS = ("all", "interface", "interior")
MIN_FREE_BYTES = 100 * 1024**3


class ValidationError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    temporary.replace(path)


def repository_identity() -> dict[str, Any]:
    def git(*arguments: str) -> str:
        result = subprocess.run(
            ["git", *arguments],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        return result.stdout

    status = git("status", "--short")
    diff = git("diff", "--binary")
    identity = hashlib.sha256((status + "\0" + diff).encode("utf-8")).hexdigest()
    return {
        "commit": git("rev-parse", "HEAD").strip(),
        "dirty": bool(status.strip()),
        "worktree_identity_sha256": identity,
        "status": status.splitlines(),
    }


def machine_snapshot() -> dict[str, Any]:
    try:
        affinity = sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        affinity = list(range(os.cpu_count() or 1))
    memory = psutil.virtual_memory()
    disk = shutil.disk_usage(REPO_ROOT)
    try:
        load = list(os.getloadavg())
    except (AttributeError, OSError):
        load = [0.0, 0.0, 0.0]
    return {
        "hostname": socket.gethostname(),
        "affinity_cpus": affinity,
        "affinity_cpu_count": len(affinity),
        "load_average": load,
        "memory_available_bytes": int(memory.available),
        "memory_total_bytes": int(memory.total),
        "disk_free_bytes": int(disk.free),
        "disk_total_bytes": int(disk.total),
    }


def process_tree_usage(process: psutil.Process) -> tuple[int, float]:
    processes = [process]
    try:
        processes.extend(process.children(recursive=True))
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    rss = 0
    cpu_seconds = 0.0
    for item in processes:
        try:
            rss += int(item.memory_info().rss)
            cpu = item.cpu_times()
            cpu_seconds += float(cpu.user + cpu.system)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return rss, cpu_seconds


def common_parallel_arguments(args: argparse.Namespace) -> list[str]:
    values = [
        "--cpu-budget",
        str(args.cpu_budget),
        "--workers",
        str(args.workers),
        "--threads-per-worker",
        str(args.threads_per_worker),
        "--memory-fraction",
        str(args.memory_fraction),
    ]
    if args.no_parallel:
        values.append("--no-parallel")
    return values


def cli_command(
    args: argparse.Namespace,
    mode: str,
    stage_directory: Path,
    *arguments: str,
) -> list[str]:
    return [
        sys.executable,
        str(CLI),
        mode,
        "--output-dir",
        str(stage_directory),
        *arguments,
        *common_parallel_arguments(args),
    ]


def run_command(command: list[str], *, log_path: Path) -> dict[str, Any]:
    environment = os.environ.copy()
    environment.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-frustration-campaign")
    environment["OMP_NUM_THREADS"] = "1"
    environment["OPENBLAS_NUM_THREADS"] = "1"
    environment["MKL_NUM_THREADS"] = "1"
    environment["NUMEXPR_NUM_THREADS"] = "1"
    wall_start = time.perf_counter()
    peak_rss = 0
    final_cpu_seconds = 0.0
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"command={json.dumps(command)}\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        monitored = psutil.Process(process.pid)
        while process.poll() is None:
            rss, cpu_seconds = process_tree_usage(monitored)
            peak_rss = max(peak_rss, rss)
            final_cpu_seconds = max(final_cpu_seconds, cpu_seconds)
            time.sleep(0.2)
        rss, cpu_seconds = process_tree_usage(monitored)
        peak_rss = max(peak_rss, rss)
        final_cpu_seconds = max(final_cpu_seconds, cpu_seconds)
        return_code = int(process.returncode)
    wall_seconds = float(time.perf_counter() - wall_start)
    return {
        "return_code": return_code,
        "wall_seconds": wall_seconds,
        "peak_rss_bytes": peak_rss,
        "process_tree_cpu_seconds": final_cpu_seconds,
        "effective_process_tree_cores": (
            final_cpu_seconds / wall_seconds if wall_seconds > 0.0 else 0.0
        ),
    }


def require_common_outputs(directory: Path) -> list[Path]:
    summaries = sorted(directory.rglob("run_summary.json"))
    if not summaries:
        raise ValidationError(f"No run summaries found under {directory}.")
    for summary in summaries:
        run_directory = summary.parent
        if not (run_directory / "scalar_metrics.csv").is_file():
            raise ValidationError(f"Missing scalar_metrics.csv in {run_directory}.")
        if not list((run_directory / "figures").glob("*.png")):
            raise ValidationError(f"Missing PNG figure in {run_directory}.")
        if not list((run_directory / "figures").glob("*.pdf")):
            raise ValidationError(f"Missing PDF figure in {run_directory}.")
    return summaries


def validate_smoke(directory: Path) -> dict[str, Any]:
    summaries = require_common_outputs(directory)
    expected = {
        "static_completion.npz": 2,
        "activity_raw.npz": 2,
        "activity_analysis.npz": 2,
        "spectral_diagnostics.npz": 4,
        "response_raw.npz": 4,
        "response_analysis.npz": 4,
    }
    observed = {name: len(list(directory.rglob(name))) for name in expected}
    if len(summaries) != 12 or observed != expected:
        raise ValidationError(
            f"Smoke schema mismatch: summaries={len(summaries)}, files={observed}."
        )
    return {"run_count": len(summaries), "file_counts": observed}


def validate_spectral(directory: Path, *, expected_cases: int = 2) -> dict[str, Any]:
    summaries = require_common_outputs(directory)
    if len(summaries) != expected_cases:
        raise ValidationError(
            f"Expected {expected_cases} spectral cases, found {len(summaries)}."
        )
    checked = 0
    for summary_path in summaries:
        summary = json.loads(summary_path.read_text())
        if summary.get("choi_failure_records"):
            raise ValidationError(f"Choi failures recorded in {summary_path.parent}.")
        data_path = summary_path.parent / "spectral_diagnostics.npz"
        with np.load(data_path) as data:
            if not np.all(np.isfinite(data["lyapunov_spectrum"])):
                raise ValidationError(f"Non-finite Lyapunov spectrum in {data_path}.")
            active = data["choi_active"]
            finite_count = data["choi_finite_count"]
            zero_count = data["choi_zero_count"]
            pole_count = data["choi_pole_count"]
            dimension = int(data["choi_spectrum"].shape[-1])
            if np.any(finite_count[active] < 0):
                raise ValidationError(f"Missing active Choi counts in {data_path}.")
            if not np.all(
                finite_count[active] + zero_count[active] + pole_count[active] == dimension
            ):
                raise ValidationError(f"Choi endpoint bookkeeping failed in {data_path}.")
            expected_finite_gap = active & (finite_count > 0)
            if not np.array_equal(np.isfinite(data["choi_gap"]), expected_finite_gap):
                raise ValidationError(f"Choi gap/count consistency failed in {data_path}.")
            if np.any(np.sum(expected_finite_gap, axis=1) == 0):
                raise ValidationError(f"No finite Choi sector was observed in {data_path}.")
            residual = data["choi_near_gap_residuals"]
            finite_mode = np.isfinite(data["choi_near_gap_exponents"])
            if not np.all(np.isfinite(residual[finite_mode])):
                raise ValidationError(f"Non-finite Choi residual in {data_path}.")
            for key in ("lyapunov_mode_region_weight", "choi_mode_region_weight"):
                values = data[key]
                finite = values[np.isfinite(values)]
                if finite.size and (np.min(finite) < -1e-10 or np.max(finite) > 1.0 + 1e-10):
                    raise ValidationError(f"Localization weights leave [0,1] in {data_path}:{key}.")
        checked += 1
    return {"checked_cases": checked}


def validate_static(directory: Path) -> dict[str, Any]:
    summaries = require_common_outputs(directory)
    if len(summaries) != 6:
        raise ValidationError(f"Expected six static cases, found {len(summaries)}.")
    for path in directory.rglob("static_completion.npz"):
        with np.load(path) as data:
            for key in ("principal_cosines", "overlap_phi", "f_star"):
                values = np.asarray(data[key])
                if not np.all(np.isfinite(values)):
                    raise ValidationError(f"Non-finite static output {key} in {path}.")
            residual = np.asarray(data["residual_map"])
            if not np.any(np.isfinite(residual)) or np.any(np.isinf(residual)):
                raise ValidationError(f"Invalid active residual map in {path}.")
            np.testing.assert_allclose(data["f_star"], data["f_star_formula"], atol=1e-10)
            if bool(np.asarray(data["exact_completion_exists"]).item()):
                if float(data["completion_constraint_error"]) > 1e-8:
                    raise ValidationError(f"Completion validation failed for {path}.")
    return {"checked_cases": len(summaries)}


def validate_activity(
    directory: Path,
    *,
    require_stationarity: bool,
) -> dict[str, Any]:
    summaries = require_common_outputs(directory)
    stationarity_failures: list[dict[str, Any]] = []
    reliability_failures: list[dict[str, Any]] = []
    for raw_path in directory.rglob("activity_raw.npz"):
        summary = json.loads((raw_path.parent / "run_summary.json").read_text())
        with np.load(raw_path) as raw:
            valid = raw["valid"]
            probabilities = raw["success_probability"][valid]
            if probabilities.size and (
                np.min(probabilities) < -1e-12 or np.max(probabilities) > 1.0 + 1e-12
            ):
                raise ValidationError(f"Invalid Born probabilities in {raw_path}.")
            if summary["config"]["protocol"] == "perfect_correction":
                if not np.array_equal(raw["defect_X"], np.abs(raw["transfer_Y"])):
                    raise ValidationError(f"X != |Y| under perfect correction in {raw_path}.")

        analysis_path = raw_path.parent / "activity_analysis.npz"
        with np.load(analysis_path) as data:
            region_names = [str(value) for value in data["region_names"]]
            s_grid = data["s_grid"]
            zero_index = int(np.argmin(np.abs(s_grid)))
            theta_zero = data["theta"][..., zero_index]
            if not np.allclose(theta_zero[np.isfinite(theta_zero)], 0.0, atol=1e-13):
                raise ValidationError(f"theta(0,T) != 0 in {analysis_path}.")
            burn_in = int(data["burn_in"])
            cycles = int(data["rates_by_cycle"].shape[2])
            observation_cycles = cycles - burn_in
            central = np.abs(s_grid) <= 0.05 + 1e-14
            for region in ACTIVITY_REGIONS:
                region_index = region_names.index(region)
                attempts = data["attempts_by_cycle"][:, burn_in:, region_index, -1]
                if int(np.sum(attempts)) == 0:
                    continue
                reliability = data["effective_sample_fraction"][:, :, region_index, central]
                if np.any(reliability < 0.1 - 1e-12):
                    reliability_failures.append(
                        {"run": str(raw_path.parent), "region": region, "minimum": float(np.min(reliability))}
                    )
                for kind_index, kind in enumerate(data["activity_names"]):
                    counts = data["counts_by_cycle"][kind_index, :, burn_in:, region_index, -1]
                    if int(np.sum(counts)) == 0:
                        continue
                    rates = data["rates_by_cycle"][kind_index, :, burn_in:, region_index, -1]
                    mean_rate = float(np.nanmean(rates))
                    slope = float(data["stationarity_slope"][kind_index, region_index])
                    drift = abs(slope) * observation_cycles
                    threshold = max(0.1 * mean_rate, 1e-3)
                    if not np.isfinite(drift) or drift > threshold:
                        stationarity_failures.append(
                            {
                                "run": str(raw_path.parent),
                                "activity": str(kind),
                                "region": region,
                                "drift": drift,
                                "threshold": threshold,
                            }
                        )
    if reliability_failures:
        raise ValidationError(f"Central SCGF reliability failed: {reliability_failures}")
    if require_stationarity and stationarity_failures:
        raise ValidationError(f"Activity stationarity failed: {stationarity_failures}")
    return {
        "checked_cases": len(summaries),
        "stationarity_pass": not stationarity_failures,
        "stationarity_failures": stationarity_failures,
    }


def validate_response(directory: Path) -> dict[str, Any]:
    summaries = require_common_outputs(directory)
    fit_eligible = 0
    fit_success = 0
    for raw_path in directory.rglob("response_raw.npz"):
        summary = json.loads((raw_path.parent / "run_summary.json").read_text())
        metrics = summary["metrics"]
        if not metrics.get("paired_schedule_identical"):
            raise ValidationError(f"Paired schedules differ in {raw_path.parent}.")
        if float(metrics["max_covariance_hermiticity_residual"]) > 1e-10:
            raise ValidationError(f"Covariance Hermiticity failed in {raw_path.parent}.")
        if float(metrics["max_covariance_spectral_bound_violation"]) > 1e-10:
            raise ValidationError(f"Covariance bounds failed in {raw_path.parent}.")
        with np.load(raw_path) as raw:
            initial = np.sum(raw["delta_charge"][:, :, 0], axis=(2, 3))
            if not np.allclose(initial, 1.0, atol=1e-10):
                raise ValidationError(f"Initial paired response is not unit normalized in {raw_path}.")
        with np.load(raw_path.parent / "response_analysis.npz") as data:
            profiles = data["wall_profiles"]
            if not np.all(np.isfinite(profiles)):
                raise ValidationError(f"Non-finite response profiles in {raw_path.parent}.")
            eligible = data["response_norm"][:, :, 0, :] > 1e-14
            fit_count = data["velocity_fit_point_count"]
            fit_eligible += int(np.count_nonzero(eligible))
            fit_success += int(np.count_nonzero((fit_count >= 2) & eligible))
    fit_fraction = fit_success / fit_eligible if fit_eligible else 1.0
    if fit_fraction < 0.9:
        raise ValidationError(
            f"Only {fit_fraction:.3f} of eligible response trajectories have usable velocity fits."
        )
    return {
        "checked_cases": len(summaries),
        "fit_eligible": fit_eligible,
        "fit_success": fit_success,
        "fit_fraction": fit_fraction,
    }


def parallel_utilization(directory: Path) -> dict[str, Any]:
    records = []
    for path in directory.rglob("run_summary.json"):
        summary = json.loads(path.read_text())
        execution = summary.get("parallel_execution", {})
        if execution:
            records.append(
                {
                    "run": str(path.parent),
                    "wall_seconds": float(execution.get("wall_seconds", 0.0)),
                    "effective_utilized_cores": float(execution.get("effective_utilized_cores", 0.0)),
                    "workers": int(execution.get("workers", 1)),
                    "task_count": int(execution.get("task_count", 1)),
                }
            )
    failures = []
    for record in records:
        if record["wall_seconds"] <= 60.0:
            continue
        available = max(1, min(record["workers"], record["task_count"]))
        if record["effective_utilized_cores"] < 0.5 * available:
            failures.append(record)
    if failures:
        raise ValidationError(f"Sustained CPU utilization gate failed: {failures}")
    return {"records": records, "long_stage_failures": failures}


def validate_benchmark(directory: Path) -> dict[str, Any]:
    activity = validate_activity(directory / "activity", require_stationarity=False)
    response = validate_response(directory / "response")
    utilization = parallel_utilization(directory)
    return {"activity": activity, "response": response, "utilization": utilization}


def build_campaign_index(campaign_root: Path, output: Path) -> dict[str, Any]:
    run_rows: list[dict[str, Any]] = []
    scalar_rows: list[dict[str, Any]] = []
    for summary_path in sorted(campaign_root.rglob("run_summary.json")):
        summary = json.loads(summary_path.read_text())
        config = summary.get("config", {})
        run_rows.append(
            {
                "stage": summary_path.relative_to(campaign_root).parts[0],
                "run_directory": str(summary_path.parent),
                **config,
                **summary.get("metrics", {}),
            }
        )
        scalar_path = summary_path.parent / "scalar_metrics.csv"
        if scalar_path.is_file():
            with scalar_path.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    scalar_rows.append(
                        {
                            "stage": summary_path.relative_to(campaign_root).parts[0],
                            "run_directory": str(summary_path.parent),
                            **row,
                        }
                    )

    def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
        fields: list[str] = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)

    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "campaign_runs.csv", run_rows)
    write_csv(output / "campaign_scalars.csv", scalar_rows)
    payload = {
        "created_utc": utc_now(),
        "run_count": len(run_rows),
        "scalar_row_count": len(scalar_rows),
        "existing_spectral_references": [
            str(REPO_ROOT.parent / "COLAB" / "colab_lyapunov"),
            str(REPO_ROOT.parent / "LARGE_RESULTS" / "choi_covariance_cpu"),
            str(REPO_ROOT.parent / "COLAB" / "colab_regularized_choi_transfer_matrix"),
        ],
        "runs": run_rows,
    }
    write_json_atomic(output / "campaign_index.json", payload)
    return payload


def run_stage(
    args: argparse.Namespace,
    campaign_root: Path,
    name: str,
    command_builder: Callable[[Path], list[str]],
    validator: Callable[[Path], dict[str, Any]] | None,
) -> dict[str, Any]:
    stage_directory = campaign_root / name
    if stage_directory.exists():
        raise RuntimeError(f"Refusing to overwrite existing stage directory {stage_directory}.")
    stage_directory.mkdir(parents=True)
    snapshot = machine_snapshot()
    if snapshot["disk_free_bytes"] < MIN_FREE_BYTES:
        raise RuntimeError("Fewer than 100 GiB are free; production launch is refused.")
    command = command_builder(stage_directory)
    status = {
        "name": name,
        "status": "running",
        "started_utc": utc_now(),
        "command": command,
        "repository": repository_identity(),
        "machine_at_start": snapshot,
    }
    write_json_atomic(stage_directory / "stage_status.json", status)
    if args.dry_run:
        status.update({"status": "dry_run", "finished_utc": utc_now()})
        write_json_atomic(stage_directory / "stage_status.json", status)
        return status
    execution = run_command(command, log_path=stage_directory / "stage.log")
    status["execution"] = execution
    if execution["return_code"] != 0:
        status.update({"status": "failed", "finished_utc": utc_now()})
        write_json_atomic(stage_directory / "stage_status.json", status)
        (stage_directory / "_FAILED").write_text("command failed\n")
        raise RuntimeError(f"Stage {name} failed; see {stage_directory / 'stage.log'}.")
    try:
        validation = {} if validator is None else validator(stage_directory)
    except Exception as exc:
        status.update(
            {
                "status": "failed_validation",
                "finished_utc": utc_now(),
                "validation_error": repr(exc),
            }
        )
        write_json_atomic(stage_directory / "stage_status.json", status)
        (stage_directory / "_FAILED").write_text("validation failed\n")
        raise
    status.update(
        {
            "status": "complete",
            "finished_utc": utc_now(),
            "validation": validation,
            "machine_at_end": machine_snapshot(),
        }
    )
    write_json_atomic(stage_directory / "stage_status.json", status)
    (stage_directory / "_SUCCESS").write_text(status["finished_utc"] + "\n")
    return status


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sequential CPU-parallel frustration campaign")
    parser.add_argument("--campaign-root", type=Path)
    parser.add_argument("--cpu-budget", type=int, default=80)
    parser.add_argument("--workers", default="auto")
    parser.add_argument("--threads-per-worker", default="auto")
    parser.add_argument("--memory-fraction", type=float, default=0.7)
    parser.add_argument("--no-parallel", action="store_true")
    parser.add_argument("--stop-after")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    campaign_root = (
        args.campaign_root.resolve()
        if args.campaign_root is not None
        else (DEFAULT_RESULTS / f"{timestamp}_new_first").resolve()
    )
    if campaign_root.exists() and any(campaign_root.iterdir()):
        raise RuntimeError(f"Refusing nonempty campaign root {campaign_root}.")
    campaign_root.mkdir(parents=True, exist_ok=True)
    campaign_config = {
        "created_utc": utc_now(),
        "campaign_root": str(campaign_root),
        "cpu_budget": args.cpu_budget,
        "workers": args.workers,
        "threads_per_worker": args.threads_per_worker,
        "memory_fraction": args.memory_fraction,
        "parallel_enabled": not args.no_parallel,
        "defaults": {"alpha_1": 1, "alpha_2": 30, "nshell": 1},
        "repository": repository_identity(),
        "machine": machine_snapshot(),
    }
    write_json_atomic(campaign_root / "campaign_config.json", campaign_config)
    stages: list[dict[str, Any]] = []

    def launch(
        name: str,
        builder: Callable[[Path], list[str]],
        validator: Callable[[Path], dict[str, Any]] | None,
    ) -> bool:
        stages.append(run_stage(args, campaign_root, name, builder, validator))
        write_json_atomic(campaign_root / "campaign_status.json", stages)
        return bool(args.stop_after == name)

    test_files = sorted(str(path) for path in (REPO_ROOT / "tests").glob("test_frustration_*.py"))
    if launch(
        "00_preflight_tests",
        lambda _: [sys.executable, "-m", "pytest", "-q", *test_files],
        None,
    ):
        return 0
    if launch(
        "00_preflight_smoke",
        lambda path: cli_command(args, "all", path, "--smoke", "--bootstrap-samples", "100"),
        validate_smoke,
    ):
        return 0
    if launch(
        "01_spectral_crosscheck",
        lambda path: cli_command(
            args,
            "spectral",
            path,
            "--nx",
            "8",
            "--ny",
            "8",
            "--samples",
            "2",
            "--cycles",
            "16",
            "--geometry",
            "both",
            "--protocol",
            "perfect_correction",
            "--seed",
            "50008",
        ),
        validate_spectral,
    ):
        return 0
    if launch(
        "02_static",
        lambda path: cli_command(
            args,
            "static",
            path,
            "--nx",
            "12",
            "--ny",
            "16",
            "24",
            "32",
            "--geometry",
            "both",
            "--nshell",
            "1",
        ),
        validate_static,
    ):
        return 0

    if launch(
        "02_spectral",
        lambda path: cli_command(
            args,
            "spectral",
            path,
            "--nx",
            "8",
            "--ny",
            "8",
            "12",
            "16",
            "--geometry",
            "both",
            "--samples",
            "8",
            "--protocol",
            "perfect_correction",
            "--seed",
            "51008",
        ),
        lambda path: validate_spectral(path, expected_cases=6),
    ):
        return 0

    benchmark_root = campaign_root / "03_parallel_benchmark"
    if benchmark_root.exists():
        raise RuntimeError(f"Refusing existing benchmark directory {benchmark_root}.")
    benchmark_root.mkdir()
    benchmark_status = []
    benchmark_status.append(
        run_stage(
            args,
            benchmark_root,
            "activity",
            lambda path: cli_command(
                args,
                "activity",
                path,
                "--nx",
                "12",
                "--ny",
                "16",
                "--geometry",
                "both",
                "--samples",
                "16",
                "--cycles",
                "16",
                "--burn-in",
                "8",
                "--bootstrap-samples",
                "100",
                "--protocol",
                "perfect_correction",
                "--seed",
                "52016",
            ),
            lambda path: validate_activity(path, require_stationarity=False),
        )
    )
    benchmark_status.append(
        run_stage(
            args,
            benchmark_root,
            "response",
            lambda path: cli_command(
                args,
                "response",
                path,
                "--nx",
                "12",
                "--ny",
                "16",
                "--geometry",
                "both",
                "--samples",
                "8",
                "--equilibration-cycles",
                "8",
                "--response-cycles",
                "4",
                "--sequence",
                "random",
                "--seed",
                "53016",
            ),
            validate_response,
        )
    )
    benchmark_validation = (
        {"dry_run": True} if args.dry_run else validate_benchmark(benchmark_root)
    )
    write_json_atomic(benchmark_root / "benchmark_status.json", benchmark_status)
    write_json_atomic(benchmark_root / "benchmark_validation.json", benchmark_validation)
    (benchmark_root / "_SUCCESS").write_text(utc_now() + "\n")
    stages.append(
        {
            "name": "03_parallel_benchmark",
            "status": "complete",
            "validation": benchmark_validation,
        }
    )
    write_json_atomic(campaign_root / "campaign_status.json", stages)
    if args.stop_after == "03_parallel_benchmark":
        return 0

    for protocol, seed in (("perfect_correction", 54016), ("imperfect", 55016)):
        name = f"04_activity_pilot_{protocol}"
        pilot_status = run_stage(
            args,
            campaign_root,
            name,
            lambda path, protocol=protocol, seed=seed: cli_command(
                args,
                "activity",
                path,
                "--nx",
                "12",
                "--ny",
                "16",
                "--geometry",
                "both",
                "--samples",
                "16",
                "--cycles",
                "64",
                "--burn-in",
                "32",
                "--bootstrap-samples",
                "200",
                "--protocol",
                protocol,
                "--sequence",
                "raster_y",
                "--seed",
                str(seed),
            ),
            lambda path: validate_activity(path, require_stationarity=False),
        )
        stages.append(pilot_status)
        stationarity_pass = bool(
            args.dry_run or pilot_status.get("validation", {}).get("stationarity_pass")
        )
        if not stationarity_pass:
            extension_name = name + "_extended"
            extension = run_stage(
                args,
                campaign_root,
                extension_name,
                lambda path, protocol=protocol, seed=seed: cli_command(
                    args,
                    "activity",
                    path,
                    "--nx",
                    "12",
                    "--ny",
                    "16",
                    "--geometry",
                    "both",
                    "--samples",
                    "16",
                    "--cycles",
                    "96",
                    "--burn-in",
                    "64",
                    "--bootstrap-samples",
                    "200",
                    "--protocol",
                    protocol,
                    "--sequence",
                    "raster_y",
                    "--seed",
                    str(seed),
                ),
                lambda path: validate_activity(path, require_stationarity=True),
            )
            stages.append(extension)
        write_json_atomic(campaign_root / "campaign_status.json", stages)
        if args.stop_after in (name, name + "_extended"):
            return 0

    for ny in (16, 24, 32):
        for protocol, seed_base in (("perfect_correction", 60000), ("imperfect", 61000)):
            name = f"05_activity_Ny{ny}_{protocol}"
            if launch(
                name,
                lambda path, ny=ny, protocol=protocol, seed_base=seed_base: cli_command(
                    args,
                    "activity",
                    path,
                    "--nx",
                    "12",
                    "--ny",
                    str(ny),
                    "--geometry",
                    "both",
                    "--samples",
                    "64",
                    "--cycles",
                    str(4 * ny),
                    "--burn-in",
                    str(2 * ny),
                    "--bootstrap-samples",
                    "1000",
                    "--protocol",
                    protocol,
                    "--sequence",
                    "raster_y",
                    "--seed",
                    str(seed_base + ny),
                ),
                lambda path: validate_activity(path, require_stationarity=True),
            ):
                return 0

    for sequence, seed in (("random", 56016), ("raster_y", 57016)):
        name = f"06_response_pilot_{sequence}"
        if launch(
            name,
            lambda path, sequence=sequence, seed=seed: cli_command(
                args,
                "response",
                path,
                "--nx",
                "12",
                "--ny",
                "16",
                "--geometry",
                "both",
                "--samples",
                "8",
                "--equilibration-cycles",
                "32",
                "--response-cycles",
                "8",
                "--sequence",
                sequence,
                "--seed",
                str(seed),
            ),
            validate_response,
        ):
            return 0

    for ny in (16, 24, 32):
        for sequence, seed_base in (("random", 70000), ("raster_y", 71000)):
            name = f"07_response_Ny{ny}_{sequence}"
            if launch(
                name,
                lambda path, ny=ny, sequence=sequence, seed_base=seed_base: cli_command(
                    args,
                    "response",
                    path,
                    "--nx",
                    "12",
                    "--ny",
                    str(ny),
                    "--geometry",
                    "both",
                    "--samples",
                    "64",
                    "--equilibration-cycles",
                    str(2 * ny),
                    "--response-cycles",
                    str(ny // 2),
                    "--sequence",
                    sequence,
                    "--seed",
                    str(seed_base + ny),
                ),
                validate_response,
            ):
                return 0

    final_name = "08_final_analysis"
    final_status = run_stage(
        args,
        campaign_root,
        final_name,
        lambda path: cli_command(
            args,
            "analyze",
            path,
            "--input-dir",
            str(campaign_root),
        ),
        None,
    )
    if args.dry_run:
        final_index = {"run_count": 0, "scalar_row_count": 0}
    else:
        final_index = build_campaign_index(campaign_root, campaign_root / final_name)
    final_status["campaign_index"] = {
        "run_count": final_index["run_count"],
        "scalar_row_count": final_index["scalar_row_count"],
    }
    write_json_atomic(campaign_root / final_name / "stage_status.json", final_status)
    stages.append(final_status)
    write_json_atomic(campaign_root / "campaign_status.json", stages)
    write_json_atomic(
        campaign_root / "campaign_complete.json",
        {
            "finished_utc": utc_now(),
            "campaign_root": str(campaign_root),
            "stage_count": len(stages),
            "campaign_index": final_status["campaign_index"],
        },
    )
    print(json.dumps({"status": "complete", "campaign_root": str(campaign_root)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
