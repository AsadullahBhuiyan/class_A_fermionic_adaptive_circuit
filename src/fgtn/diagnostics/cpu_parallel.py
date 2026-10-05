from __future__ import annotations

import contextlib
import io
import math
import os
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Iterable, TypeVar

import joblib
import psutil
from joblib import Parallel, delayed, parallel_backend
from threadpoolctl import threadpool_limits


T = TypeVar("T")
R = TypeVar("R")


@dataclass(frozen=True)
class ParallelPolicy:
    cpu_budget: int = 80
    workers: int | None = None
    threads_per_worker: int | None = None
    enabled: bool = True
    memory_fraction: float = 0.7
    minimum_dynamic_budget: int = 16


@dataclass(frozen=True)
class ParallelDecision:
    backend: str
    enabled: bool
    affinity_cpus: int
    load_1m: float
    requested_cpu_budget: int
    effective_cpu_budget: int
    task_count: int
    workers: int
    threads_per_worker: int
    available_memory_bytes: int
    estimated_bytes_per_worker: int
    memory_worker_cap: int


def _affinity_count() -> int:
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 1)


def resolve_parallel_decision(
    task_count: int,
    *,
    policy: ParallelPolicy,
    single_thread_tasks: bool,
    estimated_bytes_per_worker: int = 0,
) -> ParallelDecision:
    task_count = int(task_count)
    if task_count <= 0:
        raise ValueError("task_count must be positive.")
    affinity = _affinity_count()
    try:
        load_1m = float(os.getloadavg()[0])
    except (AttributeError, OSError):
        load_1m = 0.0
    requested_budget = int(policy.cpu_budget)
    if requested_budget <= 0:
        raise ValueError("cpu_budget must be positive.")
    if not 0.0 < float(policy.memory_fraction) <= 1.0:
        raise ValueError("memory_fraction must lie in (0, 1].")

    spare = affinity - int(math.ceil(max(0.0, load_1m)))
    floor = min(affinity, max(1, int(policy.minimum_dynamic_budget)))
    dynamic_budget = min(requested_budget, affinity, max(floor, spare))
    available_memory = int(psutil.virtual_memory().available)
    estimate = max(0, int(estimated_bytes_per_worker))
    memory_cap = task_count
    if estimate:
        memory_cap = max(1, int((available_memory * float(policy.memory_fraction)) // estimate))

    if not policy.enabled:
        workers = 1
        threads = 1
    else:
        requested_workers = task_count if policy.workers is None else int(policy.workers)
        if requested_workers <= 0:
            raise ValueError("workers must be positive or auto.")
        workers = max(1, min(task_count, dynamic_budget, memory_cap, requested_workers))
        if policy.threads_per_worker is not None:
            threads = int(policy.threads_per_worker)
            if threads <= 0:
                raise ValueError("threads_per_worker must be positive or auto.")
            workers = max(1, min(workers, dynamic_budget // min(dynamic_budget, threads)))
        elif single_thread_tasks:
            threads = 1
        else:
            threads = max(1, dynamic_budget // workers)
        threads = max(1, min(threads, dynamic_budget))

    return ParallelDecision(
        backend="loky" if policy.enabled and workers > 1 else "serial",
        enabled=bool(policy.enabled and workers > 1),
        affinity_cpus=affinity,
        load_1m=load_1m,
        requested_cpu_budget=requested_budget,
        effective_cpu_budget=dynamic_budget,
        task_count=task_count,
        workers=workers,
        threads_per_worker=threads,
        available_memory_bytes=available_memory,
        estimated_bytes_per_worker=estimate,
        memory_worker_cap=memory_cap,
    )


def _execute_task(
    function: Callable[[T], R],
    task: T,
    task_index: int,
    threads_per_worker: int,
    quiet: bool,
) -> dict[str, Any]:
    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    captured = io.StringIO()
    output_context = contextlib.redirect_stdout(captured) if quiet else contextlib.nullcontext()
    error_context = contextlib.redirect_stderr(captured) if quiet else contextlib.nullcontext()
    try:
        with threadpool_limits(limits=int(threads_per_worker)):
            with output_context, error_context:
                value = function(task)
    except Exception as exc:
        worker_output = captured.getvalue().strip()
        suffix = f"\nWorker output:\n{worker_output[-4000:]}" if worker_output else ""
        raise RuntimeError(f"Parallel task {task_index} failed.{suffix}") from exc
    return {
        "task_index": int(task_index),
        "value": value,
        "telemetry": {
            "task_index": int(task_index),
            "pid": int(os.getpid()),
            "cpu_seconds": float(time.process_time() - cpu_start),
            "wall_seconds": float(time.perf_counter() - wall_start),
            "threads_per_worker": int(threads_per_worker),
            "worker_output_tail": captured.getvalue().strip()[-1000:] if quiet else "",
        },
    }


def run_parallel_tasks(
    function: Callable[[T], R],
    tasks: Iterable[T],
    *,
    policy: ParallelPolicy,
    single_thread_tasks: bool = True,
    estimated_bytes_per_worker: int = 0,
    quiet: bool = True,
) -> tuple[list[R], dict[str, Any]]:
    task_list = list(tasks)
    decision = resolve_parallel_decision(
        len(task_list),
        policy=policy,
        single_thread_tasks=single_thread_tasks,
        estimated_bytes_per_worker=estimated_bytes_per_worker,
    )
    wall_start = time.perf_counter()
    if decision.enabled:
        with parallel_backend(
            "loky",
            n_jobs=decision.workers,
            inner_max_num_threads=decision.threads_per_worker,
        ):
            envelopes = Parallel(n_jobs=decision.workers)(
                delayed(_execute_task)(
                    function,
                    task,
                    index,
                    decision.threads_per_worker,
                    quiet,
                )
                for index, task in enumerate(task_list)
            )
    else:
        envelopes = [
            _execute_task(function, task, index, decision.threads_per_worker, quiet)
            for index, task in enumerate(task_list)
        ]
    wall_seconds = float(time.perf_counter() - wall_start)
    envelopes.sort(key=lambda item: item["task_index"])
    telemetry = [item["telemetry"] for item in envelopes]
    cpu_seconds = float(sum(float(item["cpu_seconds"]) for item in telemetry))
    metadata = {
        **asdict(decision),
        "joblib_version": joblib.__version__,
        "wall_seconds": wall_seconds,
        "worker_cpu_seconds": cpu_seconds,
        "effective_utilized_cores": cpu_seconds / wall_seconds if wall_seconds > 0.0 else 0.0,
        "worker_pids": sorted({int(item["pid"]) for item in telemetry}),
        "completed_tasks": len(envelopes),
        "task_telemetry": telemetry,
    }
    return [item["value"] for item in envelopes], metadata

