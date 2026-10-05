#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

try:
    from threadpoolctl import threadpool_limits
except Exception:  # pragma: no cover
    threadpool_limits = None


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from mean_channel_lindblad_cpu import run_case, validate_selected_observable_schema


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def save_npz_atomic(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        np.savez_compressed(handle, **arrays)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def parse_cpu_list(spec: str | None) -> list[int] | None:
    if spec is None or not str(spec).strip():
        return None
    cpus: set[int] = set()
    for token in str(spec).split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            lo, hi = (int(value) for value in token.split("-", 1))
            if hi < lo:
                raise ValueError(f"invalid CPU range {token!r}")
            cpus.update(range(lo, hi + 1))
        else:
            cpus.add(int(token))
    available = set(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else set(range(os.cpu_count() or 1))
    selected = sorted(cpus.intersection(available))
    if not selected:
        raise ValueError("--cpu-list selects no available CPUs")
    return selected


def configure_resources(cpu_list: list[int] | None, blas_threads: int) -> None:
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(max(1, int(blas_threads)))
    if cpu_list and hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, set(cpu_list))


def _tag(value: Any) -> str:
    if value is None:
        return "none"
    return str(value).replace(".", "p")


def _base_case(
    *,
    campaign: str,
    nx: int,
    ny: int,
    alpha_top: float,
    alpha_triv: float,
    nshell: int | None,
    n_a: float,
    wall_rule: str,
    domain_wall: bool,
    dw_truncation: bool,
    init_mode: str,
    config: dict[str, Any],
    label: str,
    response_enabled: bool = False,
) -> dict[str, Any]:
    if dw_truncation and not domain_wall:
        raise ValueError("dw_truncation=True requires domain_wall=True")
    case_id = (
        f"{campaign}_N{nx}x{ny}_{label}_ain-{_tag(alpha_top)}_nsh-{_tag(nshell)}"
        f"_dwtrunc-{int(bool(dw_truncation))}_na-{_tag(n_a)}_init-{init_mode}"
    )
    return {
        "case_id": case_id,
        "campaign": campaign,
        "kind": "gain_loss",
        "model": {
            "Nx": int(nx), "Ny": int(ny), "alpha_top": float(alpha_top),
            "alpha_triv": float(alpha_triv), "domain_wall": bool(domain_wall),
            "dw_truncation": bool(dw_truncation), "wall_rule": str(wall_rule),
            "nshell": nshell, "n_a": float(n_a),
            "dtype": "complex128",
        },
        "run": {
            "init_mode": str(init_mode),
            "physical_burn_in_time": 0.0,
            "physical_time_fraction": float(config["physical_time_fraction"]),
            "observation_time_fractions": list(config["observation_time_fractions"]),
            "finite_channel_p": list(config["finite_channel_p"]),
            "response": {
                **config["response"],
                "enabled": bool(response_enabled),
            },
            "save_covariance_history": False,
        },
    }


def expand_cases(config: dict[str, Any], *, nx: int, smoke: bool = False) -> list[dict[str, Any]]:
    if smoke:
        static_ny, response_ny = 6, [6]
        alpha_values, p_values = [1.0, 3.0], [1.0, 0.5, 0.25]
        nx = 4
        config = dict(config)
        config["physical_time_fraction"] = 1.0
        config["observation_time_fractions"] = [0.0, 0.5, 1.0]
        config["finite_channel_p"] = p_values
    else:
        static_ny = int(config["static_scan_Ny"])
        response_ny = [int(value) for value in config["response_Ny"]]
        alpha_values = config["alpha_in"]
    cases = []
    endpoint_alpha = float(config["response"]["alpha_in"])
    canonical_dw = bool(config["canonical_dw_truncation"])
    main_keys: set[tuple[int, float, int | None, bool]] = set()
    for alpha in alpha_values:
        for nshell in config["nshell"]:
            main_keys.add((static_ny, float(alpha), nshell, canonical_dw))
    for ny in response_ny:
        for nshell in config["nshell"]:
            for dw_truncation in config["endpoint_dw_truncation"]:
                main_keys.add((ny, endpoint_alpha, nshell, bool(dw_truncation)))
    for ny, alpha, nshell, dw_truncation in sorted(
        main_keys,
        key=lambda row: (row[0], row[1], -1 if row[2] is None else row[2], not row[3]),
    ):
        cases.append(
            _base_case(
                campaign="L1_MAIN", nx=nx, ny=ny, alpha_top=alpha,
                alpha_triv=config["alpha_out"], nshell=nshell,
                n_a=config["n_a"], wall_rule=config["wall_rule"],
                domain_wall=True, dw_truncation=dw_truncation,
                init_mode="maxmix", config=config, label="wall",
                response_enabled=(
                    math.isclose(alpha, endpoint_alpha)
                    and ny in response_ny
                    and dw_truncation in {bool(value) for value in config["endpoint_dw_truncation"]}
                )
            )
        )
    if smoke:
        return cases
    controls = config["controls"]
    ny = int(controls["representative_Ny"])
    alpha = float(controls["representative_alpha"])
    for nshell in config["nshell"]:
        cases.append(
            _base_case(
                campaign="L1_CONTROL", nx=nx, ny=ny, alpha_top=alpha,
                alpha_triv=config["alpha_out"], nshell=nshell, n_a=config["n_a"],
                wall_rule="legacy", domain_wall=True, init_mode="maxmix",
                dw_truncation=canonical_dw,
                config=config, label="legacy-wall",
            )
        )
        for n_a in controls["ancilla_fillings"]:
            cases.append(
                _base_case(
                    campaign="L1_CONTROL", nx=nx, ny=ny, alpha_top=alpha,
                    alpha_triv=config["alpha_out"], nshell=nshell, n_a=n_a,
                    wall_rule="canonical", domain_wall=True, init_mode="maxmix",
                    dw_truncation=canonical_dw,
                    config=config, label="ancilla",
                )
            )
        for phase in controls["uniform_phases"]:
            uniform_alpha = 1.0 if phase == "topological" else 30.0
            cases.append(
                _base_case(
                    campaign="L1_CONTROL", nx=nx, ny=ny, alpha_top=uniform_alpha,
                    alpha_triv=uniform_alpha, nshell=nshell, n_a=config["n_a"],
                    wall_rule="canonical", domain_wall=False, init_mode="maxmix",
                    dw_truncation=False, config=config, label=f"uniform-{phase}",
                    response_enabled=True,
                )
            )
        for init_mode in controls["initial_modes"]:
            cases.append(
                _base_case(
                    campaign="L1_CONTROL", nx=nx, ny=ny, alpha_top=alpha,
                    alpha_triv=config["alpha_out"], nshell=nshell, n_a=config["n_a"],
                    wall_rule="canonical", domain_wall=True, init_mode=init_mode,
                    dw_truncation=canonical_dw,
                    config=config, label="initialization",
                )
            )
    dephasing = config["dephasing_controls"]
    dx, dy = (int(value) for value in dephasing["geometry"])
    for alpha in dephasing["alpha_in"]:
        for nshell in dephasing["nshell"]:
            cases.append(
                {
                    "case_id": f"L1_DEPH_N{dx}x{dy}_ain-{_tag(alpha)}_nsh-{_tag(nshell)}",
                    "campaign": "L1_DEPHASING_CONTROL",
                    "kind": "dephasing_control",
                    "model": {
                        "Nx": dx, "Ny": dy, "alpha_top": float(alpha),
                        "alpha_triv": float(config["alpha_out"]), "domain_wall": True,
                        "dw_truncation": canonical_dw, "wall_rule": "canonical", "nshell": nshell,
                        "n_a": float(config["n_a"]), "dtype": "complex128",
                    },
                    "run": {
                        "init_mode": "maxmix", "dt": float(dephasing["dt"]),
                        "physical_burn_in_time": 0.0,
                        "physical_time_fraction": float(dephasing["physical_time_fraction"]),
                        "observation_time_fractions": list(dephasing["observation_time_fractions"]),
                        "save_covariance_history": False,
                    },
                }
            )
    ids = [case["case_id"] for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("case expansion produced duplicate IDs")
    return cases


def accepted_width(default: int, path: Path | None) -> int:
    if path is None:
        return int(default)
    payload = json.loads(path.read_text(encoding="utf-8"))
    value = payload.get("accepted_Nx", payload.get("accepted_width"))
    if value is None:
        raise ValueError("accepted-width JSON lacks accepted_Nx")
    if payload.get("status") not in (None, "accepted", "passed"):
        raise ValueError("accepted-width JSON is not accepted")
    return int(value)


def _worker(payload: dict[str, Any]) -> dict[str, Any]:
    configure_resources(payload["cpu_list"], payload["blas_threads"])
    case = payload["case"]
    started = time.perf_counter()
    if threadpool_limits is None:
        arrays, metadata = run_case(case)
    else:
        with threadpool_limits(limits=int(payload["blas_threads"])):
            arrays, metadata = run_case(case)
    validate_selected_observable_schema(case["model"]["nshell"], arrays) if case["kind"] == "gain_loss" else None
    output = Path(payload["case_root"])
    npz_path = output / f"{case['case_id']}.npz"
    json_path = output / f"{case['case_id']}.json"
    save_npz_atomic(npz_path, arrays)
    metadata["elapsed_seconds"] = float(time.perf_counter() - started)
    metadata["selected_observable_sha256"] = sha256_file(npz_path)
    metadata["selected_observable_bytes"] = npz_path.stat().st_size
    metadata["array_names"] = sorted(arrays)
    write_json_atomic(json_path, metadata)
    return {
        "case_id": case["case_id"], "status": "complete",
        "metadata": json_path.name, "observables": npz_path.name,
        "metadata_sha256": sha256_file(json_path), "observables_sha256": sha256_file(npz_path),
        "elapsed_seconds": metadata["elapsed_seconds"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CPU production campaign for exact mean channels and Lindblad limits")
    parser.add_argument("--config", type=Path, default=HERE / "campaign_config.json")
    parser.add_argument("--output-root", type=Path, default=HERE / "results")
    parser.add_argument("--accepted-width-json", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--blas-threads", type=int, default=1)
    parser.add_argument("--cpu-list", default="")
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    nx = accepted_width(config["default_Nx"], args.accepted_width_json)
    cases = expand_cases(config, nx=nx, smoke=args.smoke)
    if args.case_id:
        wanted = set(args.case_id)
        cases = [case for case in cases if case["case_id"] in wanted]
        missing = wanted.difference(case["case_id"] for case in cases)
        if missing:
            raise KeyError(f"unknown case IDs: {sorted(missing)}")
    if args.list_cases:
        print("\n".join(case["case_id"] for case in cases))
        return 0
    mode = "smoke" if args.smoke else "production"
    run_id = hashlib.sha256(
        json.dumps({"config": config, "Nx": nx, "mode": mode}, sort_keys=True).encode()
    ).hexdigest()[:16]
    run_root = args.output_root / f"{config['campaign']}_{mode}_{run_id}"
    case_root = run_root / "cases"
    case_root.mkdir(parents=True, exist_ok=True)
    manifest_path = run_root / "manifest.json"
    completed: dict[str, Any] = {}
    if args.resume and manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        completed = {row["case_id"]: row for row in old.get("cases", []) if row.get("status") == "complete"}
    pending = [case for case in cases if case["case_id"] not in completed]
    cpu_list = parse_cpu_list(args.cpu_list)
    manifest = {
        "schema": "mean_channel_lindblad_cpu_campaign_v2_hybrid_response",
        "status": "running",
        "mode": mode,
        "run_id": run_id,
        "Nx": nx,
        "trajectory_samples": 0,
        "schedule_samples": 0,
        "saved_covariance_history": False,
        "permanent_covariance_bytes": 0,
        "config": config,
        "config_sha256": sha256_file(args.config),
        "solver_sha256": sha256_file(HERE / "mean_channel_lindblad_cpu.py"),
        "cpu_list": cpu_list,
        "workers": int(args.workers),
        "blas_threads": int(args.blas_threads),
        "cases": list(completed.values()),
        "started_unix": time.time(),
    }
    write_json_atomic(manifest_path, manifest)
    payloads = [
        {"case": case, "case_root": str(case_root), "cpu_list": cpu_list, "blas_threads": int(args.blas_threads)}
        for case in pending
    ]
    if int(args.workers) <= 1:
        for payload in payloads:
            row = _worker(payload)
            completed[row["case_id"]] = row
            manifest["cases"] = [completed[key] for key in sorted(completed)]
            write_json_atomic(manifest_path, manifest)
            print(f"[complete] {row['case_id']}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=int(args.workers)) as pool:
            futures = {pool.submit(_worker, payload): payload["case"]["case_id"] for payload in payloads}
            for future in as_completed(futures):
                row = future.result()
                completed[row["case_id"]] = row
                manifest["cases"] = [completed[key] for key in sorted(completed)]
                write_json_atomic(manifest_path, manifest)
                print(f"[complete] {row['case_id']}", flush=True)
    manifest.update({"status": "complete", "completed_unix": time.time()})
    manifest["cases"] = [completed[key] for key in sorted(completed)]
    write_json_atomic(manifest_path, manifest)
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
