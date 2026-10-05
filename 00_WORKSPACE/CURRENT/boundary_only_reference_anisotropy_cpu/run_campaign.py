#!/usr/bin/env python3
"""Run the resumable boundary-only reference-anisotropy CPU campaign."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import multiprocessing
import os
import pickle
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_name] = "1"

import numpy as np
from threadpoolctl import threadpool_limits
from tqdm.auto import tqdm


PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parents[2]
sys.path[:0] = [str(PACKAGE_ROOT), str(REPO_ROOT / "src")]

from fgtn.classA_U1FGTN import classA_U1FGTN
from reference_probe import ReferencePairObserver
from scientific import (
    CANONICAL_ENTRY_POINT,
    anisotropy,
    configuration_hash,
    ordered_crossings,
    qwz_negative_band_frame,
    stable_seed,
    validate_config,
    wall_measurement_site_ids,
)


BASE_SCHEMA = "boundary_reference_base_checkpoint_v1"
BRANCH_SCHEMA = "boundary_reference_branch_v1"
COMPLETION_SCHEMA = "boundary_reference_completion_v1"

_MODEL: classA_U1FGTN | None = None
_GROUND_FRAME: np.ndarray | None = None
_GROUND_DIAGNOSTICS: dict[str, float] | None = None
_NY: int | None = None
_CONFIG: dict[str, Any] | None = None
_CONFIG_HASH: str | None = None
_OUTPUT_ROOT: Path | None = None
_CPU_IDS: list[int] = []


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_hashes(config_path: Path) -> dict[str, str]:
    paths = (
        Path(__file__).resolve(),
        PACKAGE_ROOT / "scientific.py",
        PACKAGE_ROOT / "reference_probe.py",
        PACKAGE_ROOT / "analyze_campaign.py",
        config_path.resolve(),
        REPO_ROOT / "src/fgtn/classA_U1FGTN.py",
        REPO_ROOT / "src/fgtn/occupied_frame.py",
    )
    return {str(path.relative_to(REPO_ROOT)): sha256_file(path) for path in paths}


def save_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def save_checkpoint_npz_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    raw = pickle.dumps(dict(payload), protocol=pickle.HIGHEST_PROTOCOL)
    save_npz_atomic(
        path,
        schema=np.asarray(BASE_SCHEMA),
        checkpoint_pickle=np.frombuffer(raw, dtype=np.uint8),
    )


def load_checkpoint_npz(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        if str(data["schema"]) != BASE_SCHEMA:
            raise ValueError("base checkpoint schema mismatch")
        raw = np.asarray(data["checkpoint_pickle"], dtype=np.uint8).tobytes()
    payload = pickle.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("base checkpoint payload is not a mapping")
    return payload


def save_npz_atomic(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".npz", dir=path.parent)
    os.close(descriptor)
    try:
        np.savez_compressed(temporary, **arrays)
        with open(temporary, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def publish_completion(path: Path, *, task_id: str, config_hash: str) -> Path:
    completion = path.with_suffix(path.suffix + ".complete.json")
    save_json_atomic(
        completion,
        {
            "schema": COMPLETION_SCHEMA,
            "task_id": task_id,
            "configuration_sha256": config_hash,
            "result_file": path.name,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "completed_at": utc_now(),
        },
    )
    return completion


def verify_completion(path: Path, *, task_id: str, config_hash: str) -> tuple[bool, str]:
    completion = path.with_suffix(path.suffix + ".complete.json")
    if not path.is_file() or not completion.is_file():
        return False, "missing result/completion pair"
    try:
        payload = json.loads(completion.read_text(encoding="utf-8"))
        if payload.get("schema") != COMPLETION_SCHEMA:
            return False, "completion schema mismatch"
        if payload.get("task_id") != task_id:
            return False, "task identity mismatch"
        if payload.get("configuration_sha256") != config_hash:
            return False, "configuration mismatch"
        if payload.get("result_file") != path.name:
            return False, "result filename mismatch"
        if int(payload.get("bytes", -1)) != path.stat().st_size:
            return False, "byte-count mismatch"
        if payload.get("sha256") != sha256_file(path):
            return False, "checksum mismatch"
        return True, "verified"
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        return False, f"verification error: {exc}"


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    validate_config(config)
    return config


def build_model_and_ground(config: Mapping[str, Any], ny: int) -> tuple[classA_U1FGTN, np.ndarray, dict[str, float]]:
    geometry, controller = config["geometry"], config["controller"]
    nx = int(geometry["Nx"])
    with threadpool_limits(limits=1):
        model = classA_U1FGTN(
            nx,
            int(ny),
            DW=True,
            nshell=int(controller["nshell"]),
            filling_frac=0.5,
            alpha_1=float(controller["alpha_1"]),
            alpha_2=float(controller["alpha_2"]),
            trial_orbitals=str(controller["trial_orbitals"]),
            dw_truncation=True,
            dw_interval=tuple(int(value) for value in geometry["domain_wall_interval"]),
            twist_y=float(geometry["controller_twist_y"]),
        )
        model.construct_OW_projectors(
            nshell=int(controller["nshell"]),
            DW=True,
            trial_orbitals=str(controller["trial_orbitals"]),
            dw_truncation=True,
            twist_y=float(geometry["controller_twist_y"]),
        )
        frame, diagnostics = qwz_negative_band_frame(
            nx, int(ny), mass=float(config["initial_state"]["mass"])
        )
    walls = tuple(int(value) for value in geometry["domain_wall_interval"])
    if tuple(int(value) for value in model.DW_loc) != walls:
        raise RuntimeError(f"unexpected controller wall locations: {model.DW_loc}")
    validation = config["validation"]
    if diagnostics["gram_residual"] > float(validation["frame_gram_residual_max"]):
        raise FloatingPointError("uniform ground frame failed Gram validation")
    if max(diagnostics["translation_x_residual"], diagnostics["translation_y_residual"]) > float(
        validation["translation_residual_max"]
    ):
        raise FloatingPointError("uniform ground state is not translation invariant")
    if abs(diagnostics["particle_number"] - nx * int(ny)) > 1e-8:
        raise FloatingPointError("uniform ground state is not exactly half filled")
    return model, frame, diagnostics


def prepare_size(config: dict[str, Any], output_root: Path, ny: int) -> None:
    global _MODEL, _GROUND_FRAME, _GROUND_DIAGNOSTICS, _NY, _CONFIG, _CONFIG_HASH, _OUTPUT_ROOT
    _MODEL, _GROUND_FRAME, _GROUND_DIAGNOSTICS = build_model_and_ground(config, ny)
    _NY, _CONFIG = int(ny), config
    _CONFIG_HASH = configuration_hash(config)
    _OUTPUT_ROOT = output_root


def require_context() -> tuple[classA_U1FGTN, np.ndarray, dict[str, Any], str, Path, int]:
    if any(value is None for value in (_MODEL, _GROUND_FRAME, _CONFIG, _CONFIG_HASH, _OUTPUT_ROOT, _NY)):
        raise RuntimeError("worker scientific context was not prepared")
    return _MODEL, _GROUND_FRAME, _CONFIG, _CONFIG_HASH, _OUTPUT_ROOT, int(_NY)


def base_task_id(ny: int, sample: int, multiplier: int) -> str:
    return f"base_Ny{ny:03d}_sample{sample:04d}_eq{multiplier:02d}Ny"


def base_path(root: Path, ny: int, sample: int, multiplier: int) -> Path:
    return root / "checkpoints" / f"Ny{ny:03d}" / f"sample{sample:04d}_eq{multiplier:02d}Ny.npz"


def branch_task_id(ny: int, sample: int, multiplier: int, follow: int, separation: int) -> str:
    label = "space" if separation == 0 else f"time{separation:03d}"
    wall = sample % 2
    return f"branch_Ny{ny:03d}_sample{sample:04d}_wall{wall}_eq{multiplier:02d}_follow{follow:02d}_{label}"


def branch_path(root: Path, ny: int, sample: int, multiplier: int, follow: int, separation: int) -> Path:
    label = "space" if separation == 0 else f"time_dt{separation:03d}"
    return (
        root
        / "branches"
        / f"Ny{ny:03d}"
        / f"eq{multiplier:02d}_follow{follow:02d}"
        / label
        / f"sample{sample:04d}.npz"
    )


def verified_base(root: Path, ny: int, sample: int, multiplier: int, config_hash: str) -> bool:
    path = base_path(root, ny, sample, multiplier)
    valid, _ = verify_completion(
        path, task_id=base_task_id(ny, sample, multiplier), config_hash=config_hash
    )
    if not valid:
        return False
    try:
        payload = load_checkpoint_npz(path)
        checkpoint = payload["checkpoint"]
        return bool(
            payload.get("schema") == BASE_SCHEMA
            and int(payload["Ny"]) == ny
            and int(payload["sample"]) == sample
            and int(payload["burn_in_multiplier"]) == multiplier
            and int(checkpoint["completed_cycles"]) == multiplier * ny - 1
        )
    except (OSError, ValueError, KeyError, pickle.UnpicklingError):
        return False


def reset_post_rngs(checkpoint: dict[str, Any], seed: int) -> dict[str, Any]:
    state = copy.deepcopy(checkpoint)
    children = np.random.SeedSequence(int(seed)).spawn(2)
    state["rng_states"]["schedule"] = copy.deepcopy(
        np.random.default_rng(children[0]).bit_generator.state
    )
    state["rng_states"]["dynamics"] = copy.deepcopy(
        np.random.default_rng(children[1]).bit_generator.state
    )
    return state


def generate_base(task: Mapping[str, int]) -> dict[str, Any]:
    model_template, ground, config, config_hash, root, ny = require_context()
    sample = int(task["sample"])
    multipliers = sorted(set(int(value) for value in task["multipliers"]))
    missing = [value for value in multipliers if not verified_base(root, ny, sample, value, config_hash)]
    if not missing:
        return {"sample": sample, "resumed": True, "generated": 0}
    lower = [value for value in multipliers if value < max(missing) and verified_base(root, ny, sample, value, config_hash)]
    checkpoint_state = None
    if lower:
        previous = max(lower)
        checkpoint_state = load_checkpoint_npz(
            base_path(root, ny, sample, previous)
        )["checkpoint"]
    model = copy.deepcopy(model_template)
    engine_seed = stable_seed(int(config["root_seed"]), ny, sample, "engine")
    site_ids = wall_measurement_site_ids(
        int(config["geometry"]["Nx"]), ny, config["controller"]["measurement_x"]
    )
    targets = {value * ny - 1: value for value in missing}
    generated = 0

    def capture(*, cycle: int, state: dict[str, Any], **_: Any) -> None:
        nonlocal generated
        if int(cycle) not in targets:
            return
        multiplier = targets[int(cycle)]
        path = base_path(root, ny, sample, multiplier)
        payload = {
            "schema": BASE_SCHEMA,
            "configuration_sha256": config_hash,
            "Ny": ny,
            "sample": sample,
            "burn_in_multiplier": multiplier,
            "engine_seed": engine_seed,
            "initial_state": "translation_invariant_qwz_mass_1_ground_state",
            "checkpoint": state,
        }
        save_checkpoint_npz_atomic(path, payload)
        publish_completion(
            path,
            task_id=base_task_id(ny, sample, multiplier),
            config_hash=config_hash,
        )
        generated += 1

    model.run_markov_circuit(
        cycles=max(missing) * ny - 1,
        samples=1,
        sequence=str(config["controller"]["sequence"]),
        perfect_correction=True,
        postselect=False,
        G_history=False,
        save=False,
        progress=False,
        random_seed=engine_seed,
        state_representation="physical_frame",
        return_native_state=True,
        parallelize_samples=False,
        init_mode="default",
        frame_init=None if checkpoint_state is not None else ground,
        checkpoint_state=checkpoint_state,
        checkpoint_observer=capture,
        measurement_site_ids=site_ids,
        meas_slab_only=False,
        require_no_covariance_materialization=True,
    )
    if any(not verified_base(root, ny, sample, value, config_hash) for value in missing):
        raise RuntimeError(f"base checkpoint publication failed for Ny={ny}, sample={sample}")
    return {"sample": sample, "resumed": False, "generated": generated}


def validate_branch(path: Path, *, task_id: str, config_hash: str, follow_cycles: int) -> tuple[bool, str]:
    valid, reason = verify_completion(path, task_id=task_id, config_hash=config_hash)
    if not valid:
        return valid, reason
    try:
        with np.load(path, allow_pickle=False) as data:
            coordinate = np.asarray(data["relative_cycles"], dtype=np.int64)
            if not np.array_equal(coordinate, np.arange(follow_cycles + 1)):
                return False, "relative-cycle coordinate mismatch"
            for name in ("mutual_information", "entropy_r1", "entropy_r2", "entropy_r12"):
                values = np.asarray(data[name], dtype=np.float64)
                if values.shape != coordinate.shape or not np.all(np.isfinite(values)):
                    return False, f"invalid {name}"
            if str(data["schema"]) != BRANCH_SCHEMA:
                return False, "branch schema mismatch"
        return True, "verified"
    except (OSError, ValueError, KeyError) as exc:
        return False, f"branch verification error: {exc}"


def generate_branch(task: Mapping[str, int]) -> dict[str, Any]:
    model_template, _, config, config_hash, root, ny = require_context()
    sample = int(task["sample"])
    multiplier = int(task["multiplier"])
    follow = int(task["follow"])
    separation = int(task["separation"])
    follow_cycles = follow * ny
    task_id = branch_task_id(ny, sample, multiplier, follow, separation)
    output = branch_path(root, ny, sample, multiplier, follow, separation)
    valid, _ = validate_branch(
        output,
        task_id=task_id,
        config_hash=config_hash,
        follow_cycles=follow_cycles,
    )
    if valid:
        return {"task_id": task_id, "resumed": True}
    base = base_path(root, ny, sample, multiplier)
    if not verified_base(root, ny, sample, multiplier, config_hash):
        raise RuntimeError(f"missing verified base checkpoint: {base}")
    checkpoint = load_checkpoint_npz(base)["checkpoint"]
    post_seed = stable_seed(int(config["root_seed"]), ny, sample, "post")
    checkpoint = reset_post_rngs(checkpoint, post_seed)
    wall_index = sample % 2
    wall_x = int(config["controller"]["measurement_x"][wall_index])
    position_seed = stable_seed(int(config["root_seed"]), ny, sample, "position")
    y1 = int(np.random.default_rng(position_seed).integers(0, ny))
    y2 = (y1 + ny // 2) % ny if separation == 0 else y1
    tau1 = multiplier * ny
    tau2 = tau1 + separation
    probe_seed = stable_seed(int(config["root_seed"]), ny, sample, "probe")
    observer = ReferencePairObserver(
        nx=int(config["geometry"]["Nx"]),
        ny=ny,
        tau1=tau1,
        tau2=tau2,
        follow_cycles=follow_cycles,
        first_site=(wall_x, y1),
        second_site=(wall_x, y2),
        rng=np.random.default_rng(probe_seed),
    )
    model = copy.deepcopy(model_template)
    site_ids = wall_measurement_site_ids(
        int(config["geometry"]["Nx"]), ny, config["controller"]["measurement_x"]
    )
    started = time.perf_counter()
    result = model.run_markov_circuit(
        cycles=tau2 + follow_cycles,
        samples=1,
        sequence=str(config["controller"]["sequence"]),
        perfect_correction=True,
        postselect=False,
        G_history=False,
        save=False,
        progress=False,
        random_seed=int(checkpoint["random_seed"]),
        state_representation="physical_frame",
        return_native_state=True,
        parallelize_samples=False,
        init_mode="default",
        checkpoint_state=checkpoint,
        native_cycle_observer=observer,
        measurement_site_ids=site_ids,
        meas_slab_only=False,
        require_no_covariance_materialization=True,
    )
    payload = observer.payload()
    final = result["native_final"]
    gram = float(final["gram_residual"])
    if gram > float(config["validation"]["engine_gram_residual_max"]):
        raise FloatingPointError(f"final augmented-frame Gram residual={gram:.3e}")
    insertion_probability = np.empty((2, 2), dtype=np.float64)
    insertion_outcome = np.empty((2, 2), dtype=np.bool_)
    insertion_entropy = np.empty(2, dtype=np.float64)
    for reference_index, reference in enumerate(
        (payload["reference_one"], payload["reference_two"])
    ):
        insertion_entropy[reference_index] = float(reference["insertion_entropy"])
        for orbital_index, event in enumerate(reference["events"]):
            insertion_probability[reference_index, orbital_index] = float(
                event["probability_occupied"]
            )
            insertion_outcome[reference_index, orbital_index] = bool(
                event["outcome_occupied"]
            )
    absolute_cycles = np.asarray(payload["cycles"], dtype=np.int64)
    save_npz_atomic(
        output,
        schema=np.asarray(BRANCH_SCHEMA),
        configuration_sha256=np.asarray(config_hash),
        canonical_dynamics_entry_point=np.asarray(CANONICAL_ENTRY_POINT),
        Ny=np.asarray(ny),
        sample=np.asarray(sample),
        wall_index=np.asarray(wall_index),
        wall_x=np.asarray(wall_x),
        y_origin=np.asarray(y1),
        spatial_branch=np.asarray(separation == 0),
        delta_tau=np.asarray(separation),
        burn_in_multiplier=np.asarray(multiplier),
        follow_multiplier=np.asarray(follow),
        relative_cycles=absolute_cycles - tau2,
        absolute_cycles=absolute_cycles,
        mutual_information=np.asarray(payload["mutual_information"], dtype=np.float64),
        entropy_r1=np.asarray(payload["entropy_r1"], dtype=np.float64),
        entropy_r2=np.asarray(payload["entropy_r2"], dtype=np.float64),
        entropy_r12=np.asarray(payload["entropy_r12"], dtype=np.float64),
        insertion_probability=insertion_probability,
        insertion_outcome=insertion_outcome,
        insertion_entropy=insertion_entropy,
        post_seed=np.asarray(post_seed, dtype=np.uint64),
        probe_seed=np.asarray(probe_seed, dtype=np.uint64),
        position_seed=np.asarray(position_seed, dtype=np.uint64),
        final_gram_residual=np.asarray(gram),
        elapsed_seconds=np.asarray(time.perf_counter() - started),
    )
    publish_completion(output, task_id=task_id, config_hash=config_hash)
    valid, reason = validate_branch(
        output,
        task_id=task_id,
        config_hash=config_hash,
        follow_cycles=follow_cycles,
    )
    if not valid:
        raise RuntimeError(f"branch publication failed: {reason}")
    return {"task_id": task_id, "resumed": False}


def parse_cpu_list(text: str) -> list[int]:
    values: list[int] = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start, stop = (int(value) for value in token.split("-", 1))
            values.extend(range(start, stop + 1))
        else:
            values.append(int(token))
    if len(values) != len(set(values)):
        raise ValueError("CPU list contains duplicates")
    return values


def worker_initializer(cpu_ids: Sequence[int]) -> None:
    identity = multiprocessing.current_process()._identity
    ordinal = identity[-1] - 1 if identity else 0
    cpu = int(cpu_ids[ordinal % len(cpu_ids)])
    os.sched_setaffinity(0, {cpu})


def run_tasks(
    tasks: Sequence[Mapping[str, int]],
    function: Any,
    *,
    workers: int,
    cpu_ids: Sequence[int],
    description: str,
) -> None:
    if not tasks:
        return
    count = min(int(workers), len(tasks))
    if count <= 0 or len(cpu_ids) < count:
        raise ValueError("insufficient CPUs for requested worker count")
    progress = tqdm(total=len(tasks), desc=description, unit="task", dynamic_ncols=True)
    if count == 1:
        worker_initializer(cpu_ids[:1])
        for task in tasks:
            function(task)
            progress.update(1)
        progress.close()
        return
    with ProcessPoolExecutor(
        max_workers=count,
        mp_context=multiprocessing.get_context("fork"),
        initializer=worker_initializer,
        initargs=(list(cpu_ids[:count]),),
    ) as pool:
        futures = [pool.submit(function, task) for task in tasks]
        for future in as_completed(futures):
            future.result()
            progress.update(1)
    progress.close()


def ensure_bases(
    config: dict[str, Any],
    root: Path,
    ny: int,
    samples: int,
    multipliers: Sequence[int],
    workers: int,
    cpu_ids: Sequence[int],
) -> None:
    prepare_size(config, root, ny)
    tasks = [{"sample": sample, "multipliers": list(multipliers)} for sample in range(samples)]
    run_tasks(
        tasks,
        generate_base,
        workers=workers,
        cpu_ids=cpu_ids,
        description=f"base Ny={ny}",
    )


def ensure_branches(
    config: dict[str, Any],
    root: Path,
    ny: int,
    samples: int,
    multiplier: int,
    follow: int,
    separations: Sequence[int],
    workers: int,
    cpu_ids: Sequence[int],
) -> None:
    prepare_size(config, root, ny)
    tasks = [
        {
            "sample": sample,
            "multiplier": int(multiplier),
            "follow": int(follow),
            "separation": int(separation),
        }
        for sample in range(samples)
        for separation in sorted(set(int(value) for value in separations))
    ]
    run_tasks(
        tasks,
        generate_branch,
        workers=workers,
        cpu_ids=cpu_ids,
        description=f"branches Ny={ny} S={samples}",
    )


def plateau_values(
    root: Path,
    config: Mapping[str, Any],
    ny: int,
    samples: int,
    multiplier: int,
    follow: int,
    separation: int,
    *,
    window_from_end: int = 1,
) -> np.ndarray:
    output = np.empty(samples, dtype=np.float64)
    config_hash = configuration_hash(config)
    follow_cycles = follow * ny
    stop = follow_cycles + 1 - (window_from_end - 1) * ny
    start = stop - ny
    if start < 0:
        raise ValueError("requested plateau window precedes branch start")
    for sample in range(samples):
        path = branch_path(root, ny, sample, multiplier, follow, separation)
        task_id = branch_task_id(ny, sample, multiplier, follow, separation)
        valid, reason = validate_branch(
            path,
            task_id=task_id,
            config_hash=config_hash,
            follow_cycles=follow_cycles,
        )
        if not valid:
            raise RuntimeError(f"invalid branch {task_id}: {reason}")
        with np.load(path, allow_pickle=False) as data:
            output[sample] = float(np.mean(np.asarray(data["mutual_information"])[start:stop]))
    return output


def grid_values(
    root: Path,
    config: Mapping[str, Any],
    ny: int,
    samples: int,
    multiplier: int,
    follow: int,
    separations: Sequence[int],
    *,
    window_from_end: int = 1,
) -> dict[int, np.ndarray]:
    return {
        int(separation): plateau_values(
            root,
            config,
            ny,
            samples,
            multiplier,
            follow,
            int(separation),
            window_from_end=window_from_end,
        )
        for separation in sorted(set(int(value) for value in separations))
    }


def bootstrap_grid(
    values: Mapping[int, np.ndarray], ny: int, *, draws: int, seed: int
) -> dict[str, Any]:
    separations = np.asarray(sorted(value for value in values if value > 0), dtype=float)
    sample_count = len(values[0])
    means = np.asarray([np.mean(values[int(value)]) for value in separations])
    crossings = ordered_crossings(separations, means, float(np.mean(values[0])))
    point = crossings[0] if len(crossings) == 1 else None
    rng = np.random.default_rng(seed)
    boot_t = np.full(draws, np.nan, dtype=np.float64)
    for draw in range(draws):
        indices = rng.integers(0, sample_count, size=sample_count)
        temporal = np.asarray([np.mean(values[int(value)][indices]) for value in separations])
        candidate = ordered_crossings(separations, temporal, float(np.mean(values[0][indices])))
        if len(candidate) == 1:
            boot_t[draw] = candidate[0]["time_star"]
    boot_alpha = np.asarray([anisotropy(ny, value) for value in boot_t])
    finite = np.isfinite(boot_alpha)
    alpha_point = None if point is None else anisotropy(ny, point["time_star"])
    return {
        "samples": sample_count,
        "spatial_mean": float(np.mean(values[0])),
        "temporal_separations": separations.astype(int).tolist(),
        "temporal_means": means.tolist(),
        "crossing_count": len(crossings),
        "bracket": point,
        "time_star": None if point is None else float(point["time_star"]),
        "alpha": alpha_point,
        "bootstrap_resolved_fraction": float(np.mean(finite)),
        "alpha_ci95": (
            None
            if not np.any(finite)
            else [float(value) for value in np.quantile(boot_alpha[finite], [0.025, 0.975])]
        ),
        "bootstrap_alpha": boot_alpha,
    }


def paired_setting_gate(
    smaller: Mapping[int, np.ndarray],
    larger: Mapping[int, np.ndarray],
    ny: int,
    *,
    tolerance: float,
    minimum_resolved: float,
    seed: int,
    draws: int = 4000,
) -> dict[str, Any]:
    first = bootstrap_grid(smaller, ny, draws=draws, seed=seed)
    second = bootstrap_grid(larger, ny, draws=draws, seed=seed + 1)
    common = sorted(set(smaller) & set(larger))
    rng = np.random.default_rng(seed + 2)
    alpha_shift = np.full(draws, np.nan, dtype=np.float64)
    point_checks: dict[str, Any] = {}
    for separation in common:
        delta = larger[separation] - smaller[separation]
        boot = np.empty(draws, dtype=np.float64)
        for draw in range(draws):
            indices = rng.integers(0, delta.size, size=delta.size)
            boot[draw] = float(np.mean(delta[indices]))
        low, high = np.quantile(boot, [0.025, 0.975])
        reference = max(abs(float(np.mean(larger[separation]))), 1e-15)
        point_checks[str(separation)] = {
            "mean_shift": float(np.mean(delta)),
            "relative_shift": float(abs(np.mean(delta)) / reference),
            "ci95": [float(low), float(high)],
            "passed": bool(abs(np.mean(delta)) / reference <= tolerance and low <= 0.0 <= high),
        }
    alpha_rng = np.random.default_rng(seed + 3)
    sample_count = len(smaller[0])
    separations = np.asarray(sorted(value for value in common if value > 0), dtype=float)
    for draw in range(draws):
        indices = alpha_rng.integers(0, sample_count, size=sample_count)
        candidates = []
        for grid in (smaller, larger):
            temporal = np.asarray(
                [np.mean(grid[int(value)][indices]) for value in separations]
            )
            crossing = ordered_crossings(
                separations, temporal, float(np.mean(grid[0][indices]))
            )
            candidates.append(
                float("nan")
                if len(crossing) != 1
                else anisotropy(ny, crossing[0]["time_star"])
            )
        if np.all(np.isfinite(candidates)):
            alpha_shift[draw] = candidates[1] - candidates[0]
    shift_values = alpha_shift[np.isfinite(alpha_shift)]
    shift_ci = [float("nan"), float("nan")]
    if shift_values.size:
        shift_ci = [float(value) for value in np.quantile(shift_values, [0.025, 0.975])]
    point_relative = float("inf")
    if first["alpha"] is not None and second["alpha"] is not None:
        point_relative = abs(float(second["alpha"]) - float(first["alpha"])) / max(
            abs(float(second["alpha"])), 1e-15
        )
    bracket_points: list[int] = [0]
    if second["bracket"] is not None:
        bracket_points.extend(
            [int(second["bracket"]["lower"]), int(second["bracket"]["upper"])]
        )
    points_pass = all(point_checks[str(value)]["passed"] for value in set(bracket_points))
    passed = bool(
        first["bootstrap_resolved_fraction"] >= minimum_resolved
        and second["bootstrap_resolved_fraction"] >= minimum_resolved
        and point_relative <= tolerance
        and shift_values.size > 0
        and shift_ci[0] <= 0.0 <= shift_ci[1]
        and points_pass
    )
    for result in (first, second):
        result.pop("bootstrap_alpha", None)
    return {
        "passed": passed,
        "smaller": first,
        "larger": second,
        "alpha_relative_shift": point_relative,
        "alpha_shift_ci95": shift_ci,
        "point_checks": point_checks,
    }


def wall_gate(
    values: Mapping[int, np.ndarray],
    ny: int,
    *,
    tolerance: float,
    seed: int,
    draws: int,
) -> dict[str, Any]:
    wall_results = []
    for wall in (0, 1):
        indices = np.arange(wall, len(values[0]), 2)
        selected = {key: array[indices] for key, array in values.items()}
        wall_results.append(bootstrap_grid(selected, ny, draws=draws, seed=seed + wall))
    if any(result["alpha"] is None for result in wall_results):
        for result in wall_results:
            result.pop("bootstrap_alpha", None)
        return {"passed": False, "reason": "one wall lacks a unique crossing", "walls": wall_results}
    alpha0, alpha1 = (float(result["alpha"]) for result in wall_results)
    relative = abs(alpha1 - alpha0) / max(abs(0.5 * (alpha0 + alpha1)), 1e-15)
    boot0 = np.asarray(wall_results[0].pop("bootstrap_alpha"))
    boot1 = np.asarray(wall_results[1].pop("bootstrap_alpha"))
    size = min(boot0.size, boot1.size)
    difference = boot1[:size] - boot0[:size]
    finite = difference[np.isfinite(difference)]
    ci = [float("nan"), float("nan")]
    if finite.size:
        ci = [float(value) for value in np.quantile(finite, [0.025, 0.975])]
    return {
        "passed": bool(relative <= tolerance and finite.size and ci[0] <= 0.0 <= ci[1]),
        "relative_difference": relative,
        "difference_ci95": ci,
        "walls": wall_results,
        "comparison": "wall-stratified whole-trajectory bootstrap",
    }


def coarse_separations(config: Mapping[str, Any], ny: int) -> list[int]:
    fractions = config["production"]["coarse_fractional_separations"]
    values = {1, 2, 3, 4}
    values.update(max(1, int(round(float(value) * ny))) for value in fractions)
    return sorted(value for value in values if value <= ny)


def stage_audit(config: dict[str, Any], root: Path, workers: int, cpu_ids: Sequence[int]) -> dict[str, Any]:
    audit = config["audit"]
    ny = int(audit["Ny"])
    screen = int(audit["screen_samples"])
    confirm = int(audit["confirmation_samples"])
    multipliers = [int(value) for value in audit["burn_in_multipliers"]]
    separations = [0] + [int(value) for value in audit["coarse_temporal_separations"]]
    follow = int(audit["initial_follow_multiplier"])
    ensure_bases(config, root, ny, screen, multipliers, workers, cpu_ids)
    for multiplier in multipliers:
        ensure_branches(config, root, ny, screen, multiplier, follow, separations, workers, cpu_ids)
    refinement: set[int] = set()
    for multiplier in multipliers:
        initial = bootstrap_grid(
            grid_values(root, config, ny, screen, multiplier, follow, separations),
            ny,
            draws=1000,
            seed=stable_seed(int(config["root_seed"]), ny, 8000 + multiplier, "bootstrap"),
        )
        if initial["bracket"] is not None:
            refinement.update(
                range(int(initial["bracket"]["lower"]) + 1, int(initial["bracket"]["upper"]))
            )
    if refinement:
        for multiplier in multipliers:
            ensure_branches(
                config, root, ny, screen, multiplier, follow, sorted(refinement), workers, cpu_ids
            )
        separations = sorted(set(separations) | refinement)
    screen_gates: dict[str, Any] = {}
    candidate: tuple[int, int] | None = None
    for smaller, larger in zip(multipliers[:-1], multipliers[1:]):
        left = grid_values(root, config, ny, screen, smaller, follow, separations)
        right = grid_values(root, config, ny, screen, larger, follow, separations)
        gate = paired_setting_gate(
            left,
            right,
            ny,
            tolerance=float(audit["setting_relative_tolerance"]),
            minimum_resolved=float(audit["minimum_bootstrap_resolved_fraction"]),
            seed=stable_seed(int(config["root_seed"]), ny, smaller * 100 + larger, "bootstrap"),
        )
        screen_gates[f"{smaller}_vs_{larger}"] = gate
        if candidate is None and gate["passed"]:
            candidate = (smaller, larger)
    if candidate is None:
        candidate = (multipliers[-2], multipliers[-1])
    ensure_bases(config, root, ny, confirm, candidate, workers, cpu_ids)
    for multiplier in candidate:
        ensure_branches(config, root, ny, confirm, multiplier, follow, separations, workers, cpu_ids)
    confirmation_refinement: set[int] = set()
    for multiplier in candidate:
        initial = bootstrap_grid(
            grid_values(root, config, ny, confirm, multiplier, follow, separations),
            ny,
            draws=1000,
            seed=stable_seed(int(config["root_seed"]), ny, 8500 + multiplier, "bootstrap"),
        )
        if initial["bracket"] is not None:
            confirmation_refinement.update(
                range(int(initial["bracket"]["lower"]) + 1, int(initial["bracket"]["upper"]))
            )
    if confirmation_refinement:
        for multiplier in candidate:
            ensure_branches(
                config,
                root,
                ny,
                confirm,
                multiplier,
                follow,
                sorted(confirmation_refinement),
                workers,
                cpu_ids,
            )
        separations = sorted(set(separations) | confirmation_refinement)
    left = grid_values(root, config, ny, confirm, candidate[0], follow, separations)
    right = grid_values(root, config, ny, confirm, candidate[1], follow, separations)
    confirmation = paired_setting_gate(
        left,
        right,
        ny,
        tolerance=float(audit["setting_relative_tolerance"]),
        minimum_resolved=float(audit["minimum_bootstrap_resolved_fraction"]),
        seed=stable_seed(int(config["root_seed"]), ny, 9000, "bootstrap"),
        draws=10000,
    )
    if not confirmation["passed"]:
        decision = {
            "status": "unresolved",
            "reason": "burn-in did not stabilize through 16 Ny",
            "candidate_pair": list(candidate),
            "screen_gates": screen_gates,
            "confirmation_gate": confirmation,
        }
        save_json_atomic(root / "analysis" / "audit_decision.json", decision)
        return decision
    selected = candidate[1]
    selected_grid = grid_values(root, config, ny, confirm, selected, follow, separations)
    crossing = bootstrap_grid(
        selected_grid,
        ny,
        draws=10000,
        seed=stable_seed(int(config["root_seed"]), ny, 9100, "bootstrap"),
    )
    crossing.pop("bootstrap_alpha", None)
    bracket = crossing["bracket"]
    if bracket is None:
        decision = {"status": "unresolved", "reason": "audit has no unique crossing", "grid": crossing}
        save_json_atomic(root / "analysis" / "audit_decision.json", decision)
        return decision
    relevant = sorted({0, int(bracket["lower"]), int(bracket["upper"])})
    previous = grid_values(
        root, config, ny, confirm, selected, follow, relevant, window_from_end=2
    )
    current = grid_values(root, config, ny, confirm, selected, follow, relevant)
    follow_gate = paired_setting_gate(
        previous,
        current,
        ny,
        tolerance=float(audit["setting_relative_tolerance"]),
        minimum_resolved=float(audit["minimum_bootstrap_resolved_fraction"]),
        seed=stable_seed(int(config["root_seed"]), ny, 9200, "bootstrap"),
        draws=10000,
    )
    if not follow_gate["passed"]:
        follow = int(audit["extended_follow_multiplier"])
        ensure_branches(config, root, ny, confirm, selected, follow, separations, workers, cpu_ids)
        selected_grid = grid_values(root, config, ny, confirm, selected, follow, separations)
        crossing = bootstrap_grid(
            selected_grid,
            ny,
            draws=10000,
            seed=stable_seed(int(config["root_seed"]), ny, 9300, "bootstrap"),
        )
        crossing.pop("bootstrap_alpha", None)
        bracket = crossing["bracket"]
        if bracket is None:
            decision = {"status": "unresolved", "reason": "extended follow has no unique crossing"}
            save_json_atomic(root / "analysis" / "audit_decision.json", decision)
            return decision
        relevant = sorted({0, int(bracket["lower"]), int(bracket["upper"])})
        previous = grid_values(
            root, config, ny, confirm, selected, follow, relevant, window_from_end=2
        )
        current = grid_values(root, config, ny, confirm, selected, follow, relevant)
        follow_gate = paired_setting_gate(
            previous,
            current,
            ny,
            tolerance=float(audit["setting_relative_tolerance"]),
            minimum_resolved=float(audit["minimum_bootstrap_resolved_fraction"]),
            seed=stable_seed(int(config["root_seed"]), ny, 9400, "bootstrap"),
            draws=10000,
        )
    status = "passed" if follow_gate["passed"] else "unresolved"
    decision = {
        "status": status,
        "reason": None if status == "passed" else "reference plateau did not stabilize through 5 Ny",
        "selected_burn_in_multiplier": selected,
        "selected_follow_multiplier": follow,
        "candidate_pair": list(candidate),
        "screen_gates": screen_gates,
        "confirmation_gate": confirmation,
        "follow_gate": follow_gate,
        "audit_grid": crossing,
    }
    save_json_atomic(root / "analysis" / "audit_decision.json", decision)
    return decision


def production_size(
    config: dict[str, Any], root: Path, ny: int, workers: int, cpu_ids: Sequence[int]
) -> dict[str, Any]:
    decision_path = root / "analysis" / "audit_decision.json"
    if not decision_path.exists():
        raise RuntimeError("production requires a completed audit")
    audit = json.loads(decision_path.read_text(encoding="utf-8"))
    if audit.get("status") != "passed":
        raise RuntimeError(f"production blocked by audit: {audit.get('reason')}")
    multiplier = int(audit["selected_burn_in_multiplier"])
    follow = int(audit["selected_follow_multiplier"])
    production = config["production"]
    bracket_samples = int(production["bracket_samples"])
    coarse = [0] + coarse_separations(config, ny)
    ensure_bases(config, root, ny, bracket_samples, [multiplier], workers, cpu_ids)
    ensure_branches(config, root, ny, bracket_samples, multiplier, follow, coarse, workers, cpu_ids)
    values = grid_values(root, config, ny, bracket_samples, multiplier, follow, coarse)
    coarse_result = bootstrap_grid(
        values,
        ny,
        draws=4000,
        seed=stable_seed(int(config["root_seed"]), ny, 10000, "bootstrap"),
    )
    coarse_result.pop("bootstrap_alpha", None)
    bracket = coarse_result["bracket"]
    if bracket is None:
        exhaustive = list(range(1, ny + 1))
        ensure_branches(
            config, root, ny, bracket_samples, multiplier, follow, exhaustive, workers, cpu_ids
        )
        coarse = [0] + exhaustive
        values = grid_values(root, config, ny, bracket_samples, multiplier, follow, coarse)
        coarse_result = bootstrap_grid(
            values,
            ny,
            draws=4000,
            seed=stable_seed(int(config["root_seed"]), ny, 10001, "bootstrap"),
        )
        coarse_result.pop("bootstrap_alpha", None)
        bracket = coarse_result["bracket"]
    if bracket is None:
        result = {"status": "unresolved", "reason": "no unique ordered crossing through delta_tau=Ny", "Ny": ny}
        save_json_atomic(root / "analysis" / f"Ny{ny:03d}_result.json", result)
        return result
    if int(bracket["upper"]) - int(bracket["lower"]) > 1:
        refinement = list(range(int(bracket["lower"]) + 1, int(bracket["upper"])))
        ensure_branches(
            config,
            root,
            ny,
            bracket_samples,
            multiplier,
            follow,
            refinement,
            workers,
            cpu_ids,
        )
        coarse = sorted(set(coarse) | set(refinement))
        values = grid_values(root, config, ny, bracket_samples, multiplier, follow, coarse)
        coarse_result = bootstrap_grid(
            values,
            ny,
            draws=4000,
            seed=stable_seed(int(config["root_seed"]), ny, 10002, "bootstrap"),
        )
        coarse_result.pop("bootstrap_alpha", None)
        bracket = coarse_result["bracket"]
        if bracket is None or int(bracket["upper"]) - int(bracket["lower"]) > 1:
            result = {
                "status": "unresolved",
                "reason": "crossing could not be reduced to adjacent integer cycles",
                "Ny": ny,
            }
            save_json_atomic(root / "analysis" / f"Ny{ny:03d}_result.json", result)
            return result
    lower, upper = int(bracket["lower"]), int(bracket["upper"])
    fine = sorted({0, max(1, lower - 1), lower, upper, min(ny, upper + 1)})
    history: list[dict[str, Any]] = []
    final_result: dict[str, Any] | None = None
    for samples in production["sample_targets"]:
        samples = int(samples)
        ensure_bases(config, root, ny, samples, [multiplier], workers, cpu_ids)
        ensure_branches(config, root, ny, samples, multiplier, follow, fine, workers, cpu_ids)
        grid = grid_values(root, config, ny, samples, multiplier, follow, fine)
        estimate = bootstrap_grid(
            grid,
            ny,
            draws=int(production["bootstrap_draws"]),
            seed=stable_seed(int(config["root_seed"]), ny, 11000 + samples, "bootstrap"),
        )
        estimate.pop("bootstrap_alpha", None)
        walls = wall_gate(
            grid,
            ny,
            tolerance=float(production["wall_relative_tolerance"]),
            seed=stable_seed(int(config["root_seed"]), ny, 12000 + samples, "bootstrap"),
            draws=int(production["bootstrap_draws"]),
        )
        ci = estimate["alpha_ci95"]
        relative_half_width = float("inf")
        if estimate["alpha"] is not None and ci is not None:
            relative_half_width = (float(ci[1]) - float(ci[0])) / (
                2.0 * abs(float(estimate["alpha"]))
            )
        passed = bool(
            estimate["crossing_count"] == 1
            and estimate["bootstrap_resolved_fraction"]
            >= float(production["minimum_bootstrap_resolved_fraction"])
            and relative_half_width <= float(production["relative_ci_half_width_target"])
            and walls["passed"]
        )
        row = {
            "samples": samples,
            "passed": passed,
            "relative_ci_half_width": relative_half_width,
            "estimate": estimate,
            "wall_gate": walls,
        }
        history.append(row)
        if passed:
            final_result = row
            break
    if final_result is None:
        final_result = history[-1]
    result = {
        "status": "calibrated" if final_result["passed"] else "unresolved",
        "reason": None if final_result["passed"] else "precision or wall-agreement gate failed at S=400",
        "Ny": ny,
        "burn_in_multiplier": multiplier,
        "follow_multiplier": follow,
        "fine_separations": fine,
        "coarse_grid": coarse_result,
        "sample_history": history,
        "final": final_result,
    }
    save_json_atomic(root / "analysis" / f"Ny{ny:03d}_result.json", result)
    return result


def write_inventory(config: Mapping[str, Any], root: Path) -> None:
    config_hash = configuration_hash(config)
    rows = []
    for ny in config["geometry"]["Ny_values"]:
        base_valid = base_invalid = 0
        branch_valid = branch_invalid = 0
        for completion in (root / "checkpoints" / f"Ny{int(ny):03d}").glob("*.complete.json"):
            try:
                payload = json.loads(completion.read_text(encoding="utf-8"))
                result = completion.with_name(str(payload["result_file"]))
                valid, _ = verify_completion(
                    result,
                    task_id=str(payload["task_id"]),
                    config_hash=config_hash,
                )
            except (OSError, KeyError, ValueError, json.JSONDecodeError):
                valid = False
            base_valid += int(valid)
            base_invalid += int(not valid)
        for completion in (root / "branches" / f"Ny{int(ny):03d}").glob("**/*.complete.json"):
            try:
                payload = json.loads(completion.read_text(encoding="utf-8"))
                result = completion.with_name(str(payload["result_file"]))
                valid, _ = verify_completion(
                    result,
                    task_id=str(payload["task_id"]),
                    config_hash=config_hash,
                )
            except (OSError, KeyError, ValueError, json.JSONDecodeError):
                valid = False
            branch_valid += int(valid)
            branch_invalid += int(not valid)
        rows.append(
            {
                "Ny": int(ny),
                "verified_base_pairs": base_valid,
                "invalid_base_pairs": base_invalid,
                "verified_branch_pairs": branch_valid,
                "invalid_branch_pairs": branch_invalid,
            }
        )
    save_json_atomic(
        root / "inventory.json",
        {
            "schema": "boundary_reference_inventory_v1",
            "configuration_sha256": config_hash,
            "updated_at": utc_now(),
            "sizes": rows,
        },
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("preflight", "audit", "production", "report"))
    parser.add_argument("--config", type=Path, default=PACKAGE_ROOT / "campaign_config.json")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--sizes", default="")
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--cpu-list", default="0-19")
    args = parser.parse_args(argv)
    config_path = args.config.resolve()
    config = load_config(config_path)
    root = (
        args.output_root
        if args.output_root is not None
        else PACKAGE_ROOT / "outputs" / str(config["revision"])
    ).resolve()
    root.mkdir(parents=True, exist_ok=True)
    immutable_config = root / "campaign_config.json"
    if immutable_config.exists() and immutable_config.read_bytes() != config_path.read_bytes():
        raise RuntimeError("output-root configuration differs from requested configuration")
    if not immutable_config.exists():
        immutable_config.write_bytes(config_path.read_bytes())
    manifest_path = root / "campaign_manifest.json"
    hashes = source_hashes(config_path)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("configuration_sha256") != configuration_hash(config):
            raise RuntimeError("campaign manifest configuration mismatch")
        if manifest.get("source_sha256") != hashes:
            raise RuntimeError("campaign sources changed after output collection creation")
    else:
        save_json_atomic(
            manifest_path,
            {
                "schema": "boundary_reference_anisotropy_manifest_v1",
                "revision": config["revision"],
                "configuration_sha256": configuration_hash(config),
                "source_sha256": hashes,
                "canonical_dynamics_entry_point": CANONICAL_ENTRY_POINT,
                "created_at": utc_now(),
            },
        )
    cpu_ids = parse_cpu_list(args.cpu_list)
    if len(cpu_ids) < args.workers:
        raise ValueError("CPU list is shorter than --workers")
    print(f"[campaign] revision={config['revision']}", flush=True)
    print(f"[campaign] canonical={CANONICAL_ENTRY_POINT}", flush=True)
    print(f"[campaign] output={root}", flush=True)
    print(
        "[campaign] uniform QWZ mass=1 ground state; no initial domain wall; "
        "controller walls x=4,12; exact-wall measurements only",
        flush=True,
    )
    print(f"[campaign] workers={args.workers} cpus={cpu_ids[:args.workers]}", flush=True)
    if args.stage == "preflight":
        for ny in (4, 16):
            nx = 4 if ny == 4 else 16
            _, diagnostics = qwz_negative_band_frame(nx, ny, mass=1.0)
            print(f"[preflight] Nx={nx} Ny={ny} {json.dumps(diagnostics, sort_keys=True)}")
        write_inventory(config, root)
        return 0
    if args.stage == "report":
        write_inventory(config, root)
        print((root / "inventory.json").read_text(encoding="utf-8"))
        return 0
    if args.stage == "audit":
        result = stage_audit(config, root, args.workers, cpu_ids)
        write_inventory(config, root)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["status"] == "passed" else 2
    sizes = (
        [int(value) for value in args.sizes.split(",") if value.strip()]
        if args.sizes
        else [int(value) for value in config["geometry"]["Ny_values"]]
    )
    allowed = set(int(value) for value in config["geometry"]["Ny_values"])
    if not sizes or any(value not in allowed for value in sizes):
        raise ValueError(f"--sizes must be a nonempty subset of {sorted(allowed)}")
    results = []
    for ny in sizes:
        print(f"[production] starting Ny={ny}", flush=True)
        results.append(production_size(config, root, ny, args.workers, cpu_ids))
    write_inventory(config, root)
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
