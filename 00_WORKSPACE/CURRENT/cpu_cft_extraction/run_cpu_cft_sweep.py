#!/usr/bin/env python3
"""CPU sweep for trajectory free energy and Lyapunov spectra.

This script intentionally calls the canonical CPU entry point
``classA_U1FGTN.run_markov_circuit(...)``.  It does not duplicate the Markov
cycle loop.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

try:
    from threadpoolctl import threadpool_limits
except Exception:  # pragma: no cover - optional dependency
    threadpool_limits = None

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from fgtn.classA_U1FGTN import classA_U1FGTN
from cft_analysis import (
    additive_fock_gap,
    bootstrap_stat,
    choi_finite_rapidities,
    fit_campaign,
    min_abs_gap,
    write_csv_dicts,
    write_json,
)


def repo_commit() -> str | None:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        return out
    except Exception:
        return None


def campaign_id(config: dict) -> str:
    payload = json.dumps(config, sort_keys=True).encode("utf-8")
    return time.strftime("%Y%m%d_%H%M%S_") + hashlib.sha1(payload).hexdigest()[:10]


def parse_cpu_list(spec: str | None) -> list[int] | None:
    """Parse ``0-3,8,10-11`` style CPU lists."""
    if spec is None or str(spec).strip() == "":
        return None
    cpus: set[int] = set()
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo_s, hi_s = part.split("-", 1)
            lo = int(lo_s)
            hi = int(hi_s)
            if hi < lo:
                raise ValueError(f"Bad CPU range {part!r}")
            cpus.update(range(lo, hi + 1))
        else:
            cpus.add(int(part))
    if not cpus:
        raise ValueError("--cpu-list did not contain any CPUs")
    return sorted(cpus)


def available_cpu_list() -> list[int]:
    try:
        return sorted(os.sched_getaffinity(0))
    except Exception:
        return list(range(os.cpu_count() or 1))


def configure_process_resources(blas_threads: int, cpu_list: list[int] | None = None) -> None:
    """Pin BLAS thread counts and optionally constrain process CPU affinity."""
    blas_threads = max(1, int(blas_threads))
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = str(blas_threads)
    if cpu_list:
        try:
            os.sched_setaffinity(0, set(int(cpu) for cpu in cpu_list))
        except Exception:
            pass


@dataclass
class WeightRecorder:
    site_rows: list[dict[str, object]] = field(default_factory=list)

    def __call__(self, **payload):
        self.site_rows.append(
            {
                "sample_index": int(payload["sample_index"]),
                "cycle": int(payload["cycle"]),
                "site_id": int(payload["site_id"]),
                "branch_log_weight": float(payload["branch_log_weight"]),
                "measurement_log_weight": float(payload["measurement_log_weight"]),
                "correction_log_weight": float(payload["correction_log_weight"]),
                "cumulative_log_weight": float(payload["cumulative_log_weight"]),
                "minus_log_p": float(-payload["cumulative_log_weight"]),
                "forced_postselect": int(bool(payload["forced_postselect"])),
            }
        )

    def cycle_rows(self) -> list[dict[str, object]]:
        latest: dict[tuple[int, int], dict[str, object]] = {}
        counts: dict[tuple[int, int], int] = {}
        for row in self.site_rows:
            key = (int(row["sample_index"]), int(row["cycle"]))
            latest[key] = row
            counts[key] = counts.get(key, 0) + 1
        out = []
        for key in sorted(latest):
            row = dict(latest[key])
            row["sites_seen"] = counts[key]
            out.append(row)
        return out


@dataclass
class TangentRecorder:
    spectra_rows: list[dict[str, object]] = field(default_factory=list)
    final_payloads: list[dict[str, object]] = field(default_factory=list)

    def __call__(self, **payload):
        spectra = np.asarray(payload["spectra"], dtype=np.float64)
        if spectra.ndim == 1:
            spectra = spectra[None, :]
        for offset in range(spectra.shape[0]):
            self.spectra_rows.append(
                {
                    "sample_index": int(payload["batch_start"]) + offset,
                    "cycle": int(payload["cycle"]),
                    "spectra": spectra[offset].copy(),
                }
            )
        if "lyapunov_null_counts" in payload:
            self.final_payloads.append(
                {
                    "cycle": int(payload["cycle"]),
                    "batch_start": int(payload["batch_start"]),
                    "lyapunov_null_counts": np.asarray(payload["lyapunov_null_counts"], dtype=np.int64).copy(),
                    "lyapunov_min_abs_value": np.asarray(payload["lyapunov_min_abs_value"], dtype=np.float64).copy(),
                    "lyapunov_min_abs_index": np.asarray(payload["lyapunov_min_abs_index"], dtype=np.int64).copy(),
                }
            )

    def arrays(self):
        samples = sorted({int(row["sample_index"]) for row in self.spectra_rows})
        cycles = sorted({int(row["cycle"]) for row in self.spectra_rows})
        if not samples or not cycles:
            return {}, np.empty((0, 0, 0), dtype=np.float64)
        n_vec = int(np.asarray(self.spectra_rows[0]["spectra"]).size)
        sample_to_i = {sample: idx for idx, sample in enumerate(samples)}
        cycle_to_i = {cycle: idx for idx, cycle in enumerate(cycles)}
        arr = np.full((len(samples), len(cycles), n_vec), np.nan, dtype=np.float64)
        for row in self.spectra_rows:
            arr[sample_to_i[int(row["sample_index"])], cycle_to_i[int(row["cycle"])], :] = row["spectra"]
        meta = {"samples": samples, "cycles": cycles}
        if self.final_payloads:
            null_counts = []
            min_abs = []
            min_idx = []
            for payload in self.final_payloads:
                null_counts.extend(payload["lyapunov_null_counts"].tolist())
                min_abs.extend(payload["lyapunov_min_abs_value"].tolist())
                min_idx.extend(payload["lyapunov_min_abs_index"].tolist())
            meta["final_null_counts"] = null_counts
            meta["final_min_abs_value"] = min_abs
            meta["final_min_abs_index"] = min_idx
        return meta, arr


@dataclass
class ChoiRecorder:
    endpoint_tol: float
    rapidity_tol: float
    rows: list[dict[str, object]] = field(default_factory=list)

    def __call__(self, **payload):
        sigma_ll = np.asarray(payload["sigma_ll"], dtype=np.complex128)
        if sigma_ll.ndim == 2:
            sigma_ll = sigma_ll[None, :, :]
        active = np.asarray(payload["choi_active_mask"], dtype=bool).reshape(-1)
        for offset in range(sigma_ll.shape[0]):
            herm = 0.5 * (sigma_ll[offset] + sigma_ll[offset].conj().T)
            eig = np.linalg.eigvalsh(herm).real
            info = choi_finite_rapidities(eig, int(payload["cycle"]), endpoint_tol=self.endpoint_tol)
            gap = min_abs_gap(info["rapidities"], floor=self.rapidity_tol)
            self.rows.append(
                {
                    "sample_index": int(payload["batch_start"]) + offset,
                    "cycle": int(payload["cycle"]),
                    "eigenvalues": eig,
                    "rapidities": info["rapidities"],
                    "choi_gap": gap,
                    "endpoint_plus": int(info["endpoint_plus"]),
                    "endpoint_minus": int(info["endpoint_minus"]),
                    "finite_count": int(info["finite_count"]),
                    "active": int(bool(active[offset])) if offset < active.size else 1,
                    "min_abs_d": float(payload["min_abs_d"]),
                }
            )
        return None

    def arrays(self):
        samples = sorted({int(row["sample_index"]) for row in self.rows})
        cycles = sorted({int(row["cycle"]) for row in self.rows})
        if not samples or not cycles:
            return {}, {}
        dim = max(int(np.asarray(row["eigenvalues"]).size) for row in self.rows)
        sample_to_i = {sample: idx for idx, sample in enumerate(samples)}
        cycle_to_i = {cycle: idx for idx, cycle in enumerate(cycles)}
        eig = np.full((len(samples), len(cycles), dim), np.nan, dtype=np.float64)
        rapid = np.full_like(eig, np.nan)
        gap = np.full((len(samples), len(cycles)), np.nan, dtype=np.float64)
        endpoint_plus = np.zeros((len(samples), len(cycles)), dtype=np.int64)
        endpoint_minus = np.zeros_like(endpoint_plus)
        active = np.zeros_like(endpoint_plus)
        for row in self.rows:
            i = sample_to_i[int(row["sample_index"])]
            j = cycle_to_i[int(row["cycle"])]
            vals = np.asarray(row["eigenvalues"], dtype=np.float64)
            raps = np.asarray(row["rapidities"], dtype=np.float64)
            eig[i, j, : vals.size] = vals
            rapid[i, j, : raps.size] = raps
            gap[i, j] = float(row["choi_gap"])
            endpoint_plus[i, j] = int(row["endpoint_plus"])
            endpoint_minus[i, j] = int(row["endpoint_minus"])
            active[i, j] = int(row["active"])
        meta = {"samples": samples, "cycles": cycles}
        return meta, {
            "eigenvalues": eig,
            "rapidities": rapid,
            "choi_gap": gap,
            "endpoint_plus": endpoint_plus,
            "endpoint_minus": endpoint_minus,
            "active": active,
        }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Use the small smoke grid.")
    parser.add_argument("--output-root", default=str(Path(__file__).resolve().parent / "outputs"))
    parser.add_argument("--nx", type=int, default=None)
    parser.add_argument("--ny", type=int, nargs="*", default=None)
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--cycles-factor", type=float, default=2.0)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--endpoint-tol", type=float, default=1e-8)
    parser.add_argument("--rapidity-tol", type=float, default=1e-12)
    parser.add_argument("--choi-signal-tol", type=float, default=1e-10)
    parser.add_argument("--fock-rank", type=int, default=1, help="Number of finite one-body gaps to add for each Fock-sector estimator.")
    parser.add_argument(
        "--sample-workers",
        type=int,
        default=None,
        help="Parallel sample workers across all requested (Ny, sample) tasks. Defaults to 2 for smoke and 28 otherwise, capped by the task count.",
    )
    parser.add_argument("--blas-threads", type=int, default=1, help="BLAS/OpenMP threads per sample worker.")
    parser.add_argument("--cpu-list", default=None, help="Optional CPU affinity list, e.g. '0-27' or '0-27,56-83'.")
    parser.add_argument("--benchmark-workers", type=int, nargs="*", default=None, help="Run a core-count timing benchmark for the listed worker counts and exit.")
    parser.add_argument("--benchmark-samples", type=int, default=4)
    parser.add_argument("--benchmark-ny", type=int, default=20)
    parser.add_argument("--benchmark-cycles", type=int, default=4)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--postselect", action="store_true", help="Refused for c_eff extraction; kept as an explicit guard.")
    return parser.parse_args()


def write_weight_csv(path: Path, rows: list[dict[str, object]]) -> None:
    write_csv_dicts(
        path,
        rows,
        [
            "sample_index",
            "cycle",
            "site_id",
            "sites_seen",
            "branch_log_weight",
            "measurement_log_weight",
            "correction_log_weight",
            "cumulative_log_weight",
            "minus_log_p",
            "forced_postselect",
        ],
    )


def sample_seed(base_seed: int, nx: int, ny: int, sample_index: int) -> int:
    payload = f"{int(base_seed)}:{int(nx)}:{int(ny)}:{int(sample_index)}".encode("utf-8")
    return int.from_bytes(hashlib.sha1(payload).digest()[:4], "little")


def remap_weight_rows(rows: list[dict[str, object]], sample_index: int) -> list[dict[str, object]]:
    out = []
    for row in rows:
        copied = dict(row)
        copied["sample_index"] = int(sample_index)
        out.append(copied)
    return out


def _run_sample_worker(payload: dict[str, object]) -> dict[str, object]:
    configure_process_resources(int(payload["blas_threads"]), payload.get("cpu_list"))
    started = time.perf_counter()
    sample_index = int(payload["sample_index"])
    nx = int(payload["nx"])
    ny = int(payload["ny"])
    cycles = int(payload["cycles"])
    seed = int(payload["seed"])
    np.random.seed(seed)

    model = classA_U1FGTN(
        Nx=nx,
        Ny=ny,
        DW=True,
        nshell=1,
        alpha_1=1.0,
        alpha_2=30.0,
        dw_truncation=True,
    )
    weight_rec = WeightRecorder()
    tangent_rec = TangentRecorder()
    choi_rec = ChoiRecorder(
        endpoint_tol=float(payload["endpoint_tol"]),
        rapidity_tol=float(payload["rapidity_tol"]),
    )
    choi_cycles = list(range(1, cycles + 1))
    run_kwargs = dict(
        G_history=False,
        progress=False,
        cycles=cycles,
        samples=1,
        save=False,
        n_a=0.5,
        sequence="raster_y",
        meas_slab_only=True,
        parallelize_samples=False,
        perfect_correction=True,
        postselect=False,
        postselect_probability=0.0,
        track_choi=True,
        choi_observer=choi_rec,
        choi_observer_cycles=choi_cycles,
        choi_singular_tol=float(payload["choi_signal_tol"]),
        choi_failure_mode="censor",
        trajectory_weight_observer=weight_rec,
        lyapunov_observer=tangent_rec,
    )
    if threadpool_limits is None:
        model.run_markov_circuit(**run_kwargs)
    else:
        with threadpool_limits(limits=int(payload["blas_threads"])):
            model.run_markov_circuit(**run_kwargs)

    cycle_weight_rows = remap_weight_rows(weight_rec.cycle_rows(), sample_index)
    tangent_meta, tangent_arr = tangent_rec.arrays()
    choi_meta, choi_arrays = choi_rec.arrays()
    return {
        "sample_index": sample_index,
        "nx": nx,
        "ny": ny,
        "cycles": cycles,
        "seed": seed,
        "elapsed_s": float(time.perf_counter() - started),
        "weight_rows": cycle_weight_rows,
        "tangent_cycles": tangent_meta.get("cycles", []),
        "tangent_spectra": tangent_arr[0].copy() if tangent_arr.size else np.empty((0, 0), dtype=np.float64),
        "tangent_null_count": (
            int(tangent_meta.get("final_null_counts", [0])[0])
            if tangent_meta.get("final_null_counts")
            else 0
        ),
        "tangent_min_abs_value": (
            float(tangent_meta.get("final_min_abs_value", [np.nan])[0])
            if tangent_meta.get("final_min_abs_value")
            else np.nan
        ),
        "tangent_min_abs_index": (
            int(tangent_meta.get("final_min_abs_index", [-1])[0])
            if tangent_meta.get("final_min_abs_index")
            else -1
        ),
        "choi_cycles": choi_meta.get("cycles", []),
        "choi_arrays": {
            key: value[0].copy()
            for key, value in choi_arrays.items()
        } if choi_arrays else {},
    }


def run_samples(payloads: list[dict[str, object]], workers: int, progress: bool) -> list[dict[str, object]]:
    if workers <= 1 or len(payloads) <= 1:
        out = []
        for idx, payload in enumerate(payloads, start=1):
            out.append(_run_sample_worker(payload))
            if progress:
                print(f"[sample] {idx}/{len(payloads)} complete", flush=True)
        return sorted(out, key=lambda row: (int(row["ny"]), int(row["sample_index"])))

    rows: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=int(workers)) as pool:
        futures = [pool.submit(_run_sample_worker, payload) for payload in payloads]
        for done_idx, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            rows.append(row)
            if progress:
                print(f"[sample] {done_idx}/{len(payloads)} complete", flush=True)
    if len(rows) != len(payloads):
        raise RuntimeError("Sample-parallel execution did not return every sample.")
    return sorted(rows, key=lambda row: (int(row["ny"]), int(row["sample_index"])))


def merge_tangent(sample_rows: list[dict[str, object]]) -> tuple[dict[str, object], np.ndarray]:
    samples = [int(row["sample_index"]) for row in sample_rows]
    cycles = sorted({int(cyc) for row in sample_rows for cyc in row["tangent_cycles"]})
    n_vec = max((np.asarray(row["tangent_spectra"]).shape[-1] for row in sample_rows), default=0)
    arr = np.full((len(samples), len(cycles), n_vec), np.nan, dtype=np.float64)
    cycle_to_i = {cycle: idx for idx, cycle in enumerate(cycles)}
    for s_i, row in enumerate(sample_rows):
        spectra = np.asarray(row["tangent_spectra"], dtype=np.float64)
        row_cycles = [int(cyc) for cyc in row["tangent_cycles"]]
        for c_i, cycle in enumerate(row_cycles):
            if spectra.size:
                arr[s_i, cycle_to_i[cycle], : spectra.shape[1]] = spectra[c_i]
    meta = {
        "samples": samples,
        "cycles": cycles,
        "final_null_counts": [int(row["tangent_null_count"]) for row in sample_rows],
        "final_min_abs_value": [float(row["tangent_min_abs_value"]) for row in sample_rows],
        "final_min_abs_index": [int(row["tangent_min_abs_index"]) for row in sample_rows],
    }
    return meta, arr


def merge_choi(sample_rows: list[dict[str, object]]) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    samples = [int(row["sample_index"]) for row in sample_rows]
    cycles = sorted({int(cyc) for row in sample_rows for cyc in row["choi_cycles"]})
    if not samples or not cycles:
        return {"samples": samples, "cycles": cycles}, {}
    dim = max(
        (
            np.asarray(row["choi_arrays"].get("eigenvalues", np.empty((0, 0)))).shape[-1]
            for row in sample_rows
            if row["choi_arrays"]
        ),
        default=0,
    )
    if dim == 0:
        return {"samples": samples, "cycles": cycles}, {}
    cycle_to_i = {cycle: idx for idx, cycle in enumerate(cycles)}
    eig = np.full((len(samples), len(cycles), dim), np.nan, dtype=np.float64)
    rapid = np.full_like(eig, np.nan)
    gap = np.full((len(samples), len(cycles)), np.nan, dtype=np.float64)
    endpoint_plus = np.zeros((len(samples), len(cycles)), dtype=np.int64)
    endpoint_minus = np.zeros_like(endpoint_plus)
    active = np.zeros_like(endpoint_plus)
    for s_i, row in enumerate(sample_rows):
        arrays = row["choi_arrays"]
        row_cycles = [int(cyc) for cyc in row["choi_cycles"]]
        for c_i, cycle in enumerate(row_cycles):
            j = cycle_to_i[cycle]
            if "eigenvalues" in arrays:
                vals = np.asarray(arrays["eigenvalues"][c_i], dtype=np.float64)
                eig[s_i, j, : vals.size] = vals
            if "rapidities" in arrays:
                vals = np.asarray(arrays["rapidities"][c_i], dtype=np.float64)
                rapid[s_i, j, : vals.size] = vals
            if "choi_gap" in arrays:
                gap[s_i, j] = float(np.asarray(arrays["choi_gap"])[c_i])
            if "endpoint_plus" in arrays:
                endpoint_plus[s_i, j] = int(np.asarray(arrays["endpoint_plus"])[c_i])
            if "endpoint_minus" in arrays:
                endpoint_minus[s_i, j] = int(np.asarray(arrays["endpoint_minus"])[c_i])
            if "active" in arrays:
                active[s_i, j] = int(np.asarray(arrays["active"])[c_i])
    return {"samples": samples, "cycles": cycles}, {
        "eigenvalues": eig,
        "rapidities": rapid,
        "choi_gap": gap,
        "endpoint_plus": endpoint_plus,
        "endpoint_minus": endpoint_minus,
        "active": active,
    }


def build_payloads(args, nx: int, ny: int, samples: int, cycles: int, cpu_list: list[int] | None) -> list[dict[str, object]]:
    return [
        {
            "sample_index": sample_idx,
            "nx": nx,
            "ny": ny,
            "cycles": cycles,
            "seed": sample_seed(args.seed, nx, ny, sample_idx),
            "endpoint_tol": float(args.endpoint_tol),
            "rapidity_tol": float(args.rapidity_tol),
            "choi_signal_tol": float(args.choi_signal_tol),
            "blas_threads": int(args.blas_threads),
            "cpu_list": cpu_list,
        }
        for sample_idx in range(samples)
    ]


def run_size(
    args,
    outdir: Path,
    ny: int,
    samples: int,
    nx: int,
    cycles_override: int | None = None,
    sample_rows: list[dict[str, object]] | None = None,
    workers_override: int | None = None,
) -> dict[str, object]:
    cycles = int(cycles_override) if cycles_override is not None else int(round(args.cycles_factor * ny))
    workers = int(workers_override) if workers_override is not None else max(1, min(int(args.sample_workers), int(samples)))
    cpu_list = parse_cpu_list(args.cpu_list)
    if sample_rows is None:
        payloads = build_payloads(args, nx, ny, samples, cycles, cpu_list)
        sample_rows = run_samples(payloads, workers=workers, progress=bool(args.progress))
    else:
        sample_rows = sorted(sample_rows, key=lambda row: int(row["sample_index"]))

    size_dir = outdir / f"N{nx}x{ny}"
    size_dir.mkdir(parents=True, exist_ok=True)
    cycle_weight_rows = [row for sample in sample_rows for row in sample["weight_rows"]]
    cycle_weight_rows.sort(key=lambda row: (int(row["sample_index"]), int(row["cycle"])))
    write_weight_csv(size_dir / "trajectory_weights.csv", cycle_weight_rows)

    tangent_meta, tangent_arr = merge_tangent(sample_rows)
    tangent_meta["sample_elapsed_s"] = [float(row["elapsed_s"]) for row in sample_rows]
    np.savez_compressed(
        size_dir / "tangent_lyapunov.npz",
        spectra=tangent_arr,
        metadata=np.asarray(json.dumps(tangent_meta, sort_keys=True)),
    )
    choi_meta, choi_arrays = merge_choi(sample_rows)
    np.savez_compressed(
        size_dir / "choi_rapidity.npz",
        metadata=np.asarray(json.dumps(choi_meta, sort_keys=True)),
        **choi_arrays,
    )

    final_weights = [row for row in cycle_weight_rows if int(row["cycle"]) == cycles]
    Y = [float(row["minus_log_p"]) for row in final_weights]
    f0_stats = bootstrap_stat([y / (args.alpha * ny * cycles) for y in Y], stat="mean", seed=args.seed + ny)

    tangent_final = tangent_arr[:, -1, :] if tangent_arr.size else np.empty((0, 0))
    tangent_one_body_gaps = [min_abs_gap(row, floor=args.rapidity_tol) for row in tangent_final]
    tangent_fock_gaps = [additive_fock_gap(row, rank=args.fock_rank, floor=args.rapidity_tol) for row in tangent_final]
    tangent_one_body_stats = bootstrap_stat(tangent_one_body_gaps, stat="median", seed=args.seed + 10_000 + ny)
    tangent_fock_stats = bootstrap_stat(tangent_fock_gaps, stat="median", seed=args.seed + 11_000 + ny)

    choi_one_body_gap_arr = choi_arrays["choi_gap"][:, -1] if choi_arrays else np.empty((0,), dtype=np.float64)
    choi_fock_gap_arr = (
        np.asarray(
            [
                additive_fock_gap(row, rank=args.fock_rank, floor=args.rapidity_tol)
                for row in choi_arrays["rapidities"][:, -1, :]
            ],
            dtype=np.float64,
        )
        if choi_arrays
        else np.empty((0,), dtype=np.float64)
    )
    choi_one_body_stats = bootstrap_stat(choi_one_body_gap_arr, stat="median", seed=args.seed + 20_000 + ny)
    choi_fock_stats = bootstrap_stat(choi_fock_gap_arr, stat="median", seed=args.seed + 21_000 + ny)
    endpoint_total = (
        choi_arrays["endpoint_plus"][:, -1] + choi_arrays["endpoint_minus"][:, -1]
        if choi_arrays
        else np.empty((0,), dtype=np.int64)
    )
    active_final = choi_arrays["active"][:, -1] if choi_arrays else np.empty((0,), dtype=np.int64)
    tangent_null = np.asarray(tangent_meta.get("final_null_counts", []), dtype=np.float64)

    row = {
        "Nx": nx,
        "Ny": ny,
        "L": ny,
        "cycles": cycles,
        "samples": samples,
        "alpha": float(args.alpha),
        "valid_weight_samples": int(f0_stats["n"]),
        "f0": f0_stats["value"],
        "f0_se": f0_stats["se"],
        "fock_rank": int(args.fock_rank),
        "tangent_one_body_gap": tangent_one_body_stats["value"],
        "tangent_one_body_gap_se": tangent_one_body_stats["se"],
        "tangent_fock_gap": tangent_fock_stats["value"],
        "tangent_fock_gap_se": tangent_fock_stats["se"],
        "x_tangent_fock_size": float(ny * tangent_fock_stats["value"] / (2.0 * np.pi * args.alpha)),
        "choi_one_body_gap": choi_one_body_stats["value"],
        "choi_one_body_gap_se": choi_one_body_stats["se"],
        "choi_fock_gap": choi_fock_stats["value"],
        "choi_fock_gap_se": choi_fock_stats["se"],
        "x_choi_fock_size": float(ny * choi_fock_stats["value"] / (2.0 * np.pi * args.alpha)),
        "choi_censored_count": int(np.count_nonzero(active_final == 0)),
        "choi_endpoint_mean": float(np.mean(endpoint_total)) if endpoint_total.size else np.nan,
        "tangent_null_mean": float(np.mean(tangent_null)) if tangent_null.size else np.nan,
        "sample_workers": int(workers),
        "blas_threads": int(args.blas_threads),
        "sample_elapsed_s_mean": float(np.mean([row["elapsed_s"] for row in sample_rows])),
        "sample_elapsed_s_max": float(np.max([row["elapsed_s"] for row in sample_rows])),
    }
    write_json(size_dir / "scalar_metrics.json", row)
    return row


def memory_snapshot() -> dict[str, object]:
    try:
        import psutil

        mem = psutil.virtual_memory()
        return {
            "available_bytes": int(mem.available),
            "used_bytes": int(mem.used),
            "total_bytes": int(mem.total),
            "percent": float(mem.percent),
        }
    except Exception:
        return {}


def run_worker_benchmark(args, nx: int) -> Path:
    output_root = Path(args.output_root).resolve()
    bench_id = time.strftime("benchmark_%Y%m%d_%H%M%S")
    bench_dir = output_root / bench_id
    bench_dir.mkdir(parents=True, exist_ok=False)
    cpu_list = parse_cpu_list(args.cpu_list)
    if cpu_list:
        configure_process_resources(args.blas_threads, cpu_list)

    rows = []
    worker_counts = args.benchmark_workers or [8, 14, 20, 28, 40, 56]
    for workers_req in worker_counts:
        workers = max(1, min(int(workers_req), int(args.benchmark_samples)))
        payloads = [
            {
                "sample_index": sample_idx,
                "nx": int(nx),
                "ny": int(args.benchmark_ny),
                "cycles": int(args.benchmark_cycles),
                "seed": sample_seed(args.seed, int(nx), int(args.benchmark_ny), sample_idx),
                "endpoint_tol": float(args.endpoint_tol),
                "rapidity_tol": float(args.rapidity_tol),
                "choi_signal_tol": float(args.choi_signal_tol),
                "blas_threads": int(args.blas_threads),
                "cpu_list": cpu_list,
            }
            for sample_idx in range(int(args.benchmark_samples))
        ]
        mem_before = memory_snapshot()
        started = time.perf_counter()
        sample_rows = run_samples(payloads, workers=workers, progress=bool(args.progress))
        elapsed_s = float(time.perf_counter() - started)
        mem_after = memory_snapshot()
        rows.append(
            {
                "requested_workers": int(workers_req),
                "effective_workers": int(workers),
                "Nx": int(nx),
                "Ny": int(args.benchmark_ny),
                "cycles": int(args.benchmark_cycles),
                "samples": int(args.benchmark_samples),
                "blas_threads": int(args.blas_threads),
                "elapsed_s": elapsed_s,
                "samples_per_hour": float(3600.0 * len(sample_rows) / elapsed_s) if elapsed_s > 0 else np.nan,
                "sample_elapsed_s_mean": float(np.mean([row["elapsed_s"] for row in sample_rows])),
                "sample_elapsed_s_max": float(np.max([row["elapsed_s"] for row in sample_rows])),
                "mem_available_before_bytes": mem_before.get("available_bytes", ""),
                "mem_available_after_bytes": mem_after.get("available_bytes", ""),
                "mem_percent_after": mem_after.get("percent", ""),
            }
        )
    fieldnames = [
        "requested_workers",
        "effective_workers",
        "Nx",
        "Ny",
        "cycles",
        "samples",
        "blas_threads",
        "elapsed_s",
        "samples_per_hour",
        "sample_elapsed_s_mean",
        "sample_elapsed_s_max",
        "mem_available_before_bytes",
        "mem_available_after_bytes",
        "mem_percent_after",
    ]
    write_csv_dicts(bench_dir / "benchmark_summary.csv", rows, fieldnames)
    best_rate = max((float(row["samples_per_hour"]) for row in rows if np.isfinite(float(row["samples_per_hour"]))), default=np.nan)
    viable = [
        row for row in rows
        if np.isfinite(float(row["samples_per_hour"])) and float(row["samples_per_hour"]) >= 0.95 * best_rate
    ]
    recommended = min(viable, key=lambda row: int(row["effective_workers"])) if viable else None
    payload = {
        "benchmark_kind": "sample_worker_core_tuning",
        "canonical_dynamics_entry_point": "classA_U1FGTN.run_markov_circuit",
        "host": os.uname().nodename,
        "available_cpus": available_cpu_list(),
        "cpu_list": cpu_list,
        "repo_commit": repo_commit(),
        "rows": rows,
        "selection_rule": "smallest worker count within 5 percent of best samples_per_hour",
        "recommended": recommended,
    }
    write_json(bench_dir / "benchmark_summary.json", payload)
    print(f"[done] wrote benchmark report to {bench_dir}")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return bench_dir


def run_campaign_sizes(args, outdir: Path, ny_values: list[int], samples: int, nx: int) -> list[dict[str, object]]:
    """Run all requested size/sample tasks through one global worker pool."""
    cpu_list = parse_cpu_list(args.cpu_list)
    payloads: list[dict[str, object]] = []
    cycles_by_ny: dict[int, int] = {}
    for ny in ny_values:
        cycles = int(round(args.cycles_factor * int(ny)))
        cycles_by_ny[int(ny)] = cycles
        payloads.extend(build_payloads(args, nx, int(ny), int(samples), cycles, cpu_list))

    workers = max(1, min(int(args.sample_workers), len(payloads)))
    sample_rows = run_samples(payloads, workers=workers, progress=bool(args.progress))
    rows = []
    for ny in ny_values:
        grouped = [row for row in sample_rows if int(row["ny"]) == int(ny)]
        if len(grouped) != int(samples):
            raise RuntimeError(f"Expected {samples} samples for Ny={ny}, got {len(grouped)}")
        rows.append(
            run_size(
                args,
                outdir,
                int(ny),
                int(samples),
                int(nx),
                cycles_override=cycles_by_ny[int(ny)],
                sample_rows=grouped,
                workers_override=workers,
            )
        )
    return rows


def main():
    args = parse_args()
    if args.smoke:
        nx = 8 if args.nx is None else args.nx
        ny_values = [8, 10] if args.ny is None else args.ny
        samples = 2 if args.samples is None else args.samples
    else:
        nx = 20 if args.nx is None else args.nx
        ny_values = [20, 30, 40] if args.ny is None else args.ny
        samples = 10 if args.samples is None else args.samples
    if samples <= 0:
        raise ValueError("samples must be positive")
    if args.alpha <= 0.0 or not np.isfinite(args.alpha):
        raise ValueError("--alpha must be positive and finite")
    if args.fock_rank <= 0:
        raise ValueError("--fock-rank must be positive")
    if args.blas_threads <= 0:
        raise ValueError("--blas-threads must be positive")
    if args.sample_workers is None:
        args.sample_workers = min(samples, 2 if args.smoke else 28)
    if args.sample_workers <= 0:
        raise ValueError("--sample-workers must be positive")
    if args.cpu_list is not None:
        configure_process_resources(args.blas_threads, parse_cpu_list(args.cpu_list))
    if args.postselect:
        raise ValueError("Postselected or forced-control trajectories are excluded from the c_eff extraction.")
    if args.benchmark_workers is not None:
        if args.benchmark_samples <= 0:
            raise ValueError("--benchmark-samples must be positive")
        if args.benchmark_ny <= 0:
            raise ValueError("--benchmark-ny must be positive")
        if args.benchmark_cycles <= 0:
            raise ValueError("--benchmark-cycles must be positive")
        run_worker_benchmark(args, nx=nx)
        return

    config = {
        "mode": "smoke" if args.smoke else "production",
        "Nx": nx,
        "Ny": [int(v) for v in ny_values],
        "samples": samples,
        "cycles_factor": float(args.cycles_factor),
        "alpha": float(args.alpha),
        "fock_rank": int(args.fock_rank),
        "seed": int(args.seed),
        "endpoint_tol": float(args.endpoint_tol),
        "rapidity_tol": float(args.rapidity_tol),
        "choi_signal_tol": float(args.choi_signal_tol),
        "sample_workers": int(args.sample_workers),
        "blas_threads": int(args.blas_threads),
        "cpu_list": parse_cpu_list(args.cpu_list),
        "available_cpus_at_launch": available_cpu_list(),
        "sample_seed_rule": "sha1(base_seed:Nx:Ny:sample_index) first uint32",
        "canonical_dynamics_entry_point": "classA_U1FGTN.run_markov_circuit",
        "postselect": False,
        "postselect_probability": 0.0,
        "perfect_correction": True,
        "track_choi": True,
        "lyapunov_route": "CPU tangent QR cocycle",
        "choi_route": "CPU regularized Choi covariance finite sector",
        "repo_commit": repo_commit(),
    }
    outdir = Path(args.output_root).resolve() / campaign_id(config)
    outdir.mkdir(parents=True, exist_ok=False)
    write_json(outdir / "manifest.json", config)

    rows = run_campaign_sizes(args, outdir, [int(ny) for ny in ny_values], int(samples), int(nx))

    fieldnames = [
        "Nx",
        "Ny",
        "L",
        "cycles",
        "samples",
        "alpha",
        "valid_weight_samples",
        "f0",
        "f0_se",
        "fock_rank",
        "tangent_one_body_gap",
        "tangent_one_body_gap_se",
        "tangent_fock_gap",
        "tangent_fock_gap_se",
        "x_tangent_fock_size",
        "choi_one_body_gap",
        "choi_one_body_gap_se",
        "choi_fock_gap",
        "choi_fock_gap_se",
        "x_choi_fock_size",
        "choi_censored_count",
        "choi_endpoint_mean",
        "tangent_null_mean",
        "sample_workers",
        "blas_threads",
        "sample_elapsed_s_mean",
        "sample_elapsed_s_max",
    ]
    write_csv_dicts(outdir / "scalars_by_size.csv", rows, fieldnames)
    fit_summary = fit_campaign(rows, alpha=args.alpha)
    write_json(outdir / "fit_summary.json", fit_summary)
    print(f"[done] wrote CPU CFT campaign to {outdir}")
    print(json.dumps(fit_summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
