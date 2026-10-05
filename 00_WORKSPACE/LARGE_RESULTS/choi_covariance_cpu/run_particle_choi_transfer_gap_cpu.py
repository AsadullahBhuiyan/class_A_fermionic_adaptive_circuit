#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import contextlib
import csv
import json
import os
import platform
import socket
import sys
import time
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
TMP_ROOT = REPO_ROOT / ".tmp"
TMP_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(TMP_ROOT / "mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(TMP_ROOT / "cache"))

THREAD_ENV_KEYS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _preparse_blas_threads(argv: list[str]) -> int | None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--blas-threads", type=int, default=None)
    args, _ = parser.parse_known_args(argv)
    if args.blas_threads is None:
        return None
    if int(args.blas_threads) <= 0:
        raise SystemExit("--blas-threads must be a positive integer.")
    return int(args.blas_threads)


_PREPARSED_BLAS_THREADS = _preparse_blas_threads(sys.argv[1:])
if _PREPARSED_BLAS_THREADS is not None:
    for _key in THREAD_ENV_KEYS:
        os.environ[_key] = str(_PREPARSED_BLAS_THREADS)

import numpy as np
from tqdm.auto import tqdm

try:
    from threadpoolctl import threadpool_info, threadpool_limits
except Exception:  # pragma: no cover - optional dependency fallback
    threadpool_info = None
    threadpool_limits = None

SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fgtn.classA_U1FGTN import classA_U1FGTN  # noqa: E402


HELPER_VERSION = "choi_covariance_cpu_v6_openbc_transfer_matrix_by_cycle"
CANONICAL_ENTRY_POINT = "classA_U1FGTN.run_markov_circuit"
OBSERVABLE = "complex_particle_choi_transfer_gap"
CHOI_FORMULA_VERSION = "regularized_resolvent_rank_one_v2"
PARTICLE_TRANSFER_DEFINITION = "T_p=-(I+Sigma_LL)^(-1) Sigma_LR on finite sectors"
FULL_PARTICLE_TRANSFER_DEFINITION = "T_p=-(I+Sigma_LL)^(-1) Sigma_LR"
GAP_DEFINITION = "min_j abs(log(s_j(T_p))/cycle)"
GAP_EXTRACTION = "exact_eigh_Sigma_LL_cpu"
FINAL_SPECTRUM_EXTRACTION = "exact_eigh_Sigma_LL"
ROOTED_SINGULAR_VALUE_DEFINITION = "rooted_singular_value=exp(log_singular_value/cycle)=s^(1/C)"
N_EIGENSTATES = 3
ENDPOINT_TOL = 1e-12
CHOI_SPECTRAL_TOL = 1e-10
CHOI_ENTRY_ABS_TOL = 1e-10
CHOI_SINGULAR_TOL = 1e-10
CHOI_FAILURE_MODE = "censor"
PHYSICAL_HEALTH_TOL = 1e-8
TRANSFER_MATRIX_MAX_ABS_TOL = 1e12


def write_json_atomic(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    tmp_path.replace(path)


def save_npz_atomic(path: Path, **arrays: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("wb") as fh:
        np.savez_compressed(fh, **arrays)
    tmp_path.replace(path)


def save_csv_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with tmp_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp_path.replace(path)


def log(message: str, *, quiet: bool = False) -> None:
    if not quiet:
        print(message, flush=True)


def parse_cpu_list(spec: str | None) -> list[int] | None:
    if spec is None:
        return None
    cpus: list[int] = []
    for raw_part in str(spec).split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, stop_s = part.split("-", 1)
            start = int(start_s)
            stop = int(stop_s)
            if stop < start:
                raise ValueError(f"Invalid CPU range {part!r}: stop is smaller than start.")
            cpus.extend(range(start, stop + 1))
        else:
            cpus.append(int(part))
    if not cpus:
        raise ValueError("--cpu-list did not contain any CPUs.")
    unique = sorted(set(cpus))
    if unique[0] < 0:
        raise ValueError("CPU indices must be non-negative.")
    return unique


def effective_cpu_affinity() -> list[int] | None:
    if hasattr(os, "sched_getaffinity"):
        return sorted(int(cpu) for cpu in os.sched_getaffinity(0))
    return None


def apply_cpu_affinity(cpu_list: str | None, *, quiet: bool = False) -> list[int] | None:
    requested = parse_cpu_list(cpu_list)
    if requested is not None:
        if not hasattr(os, "sched_setaffinity"):
            log("[warn] CPU affinity is not supported on this platform; ignoring --cpu-list.", quiet=quiet)
        else:
            os.sched_setaffinity(0, requested)
    return effective_cpu_affinity()


def normalize_blas_threads(value: int | None) -> int | None:
    if value is None:
        return None
    value = int(value)
    if value <= 0:
        raise ValueError("--blas-threads must be a positive integer.")
    return value


def apply_thread_env(blas_threads: int | None) -> None:
    if blas_threads is None:
        return
    for key in THREAD_ENV_KEYS:
        os.environ[key] = str(int(blas_threads))


def json_safe_threadpool_info() -> list[dict[str, Any]]:
    if threadpool_info is None:
        return []
    info = threadpool_info()
    try:
        json.dumps(info, sort_keys=True)
        return info
    except TypeError:
        return json.loads(json.dumps(info, default=str, sort_keys=True))


def runtime_metadata(
    *,
    cpu_list_requested: str | None,
    cpu_affinity_effective: list[int] | None,
    blas_threads_requested: int | None,
    progress_enabled: bool,
    quiet: bool,
) -> dict[str, Any]:
    return {
        "cpu_list_requested": cpu_list_requested,
        "cpu_affinity_effective": cpu_affinity_effective,
        "blas_threads_requested": blas_threads_requested,
        "thread_env": {key: os.environ.get(key) for key in THREAD_ENV_KEYS},
        "threadpool_info": json_safe_threadpool_info(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": sys.version,
        "progress_enabled": bool(progress_enabled),
        "quiet": bool(quiet),
    }


def _hermitize(mat: np.ndarray) -> np.ndarray:
    return 0.5 * (mat + mat.conj().T)


def _exponents_from_a_values(
    a_values: np.ndarray,
    *,
    cycle: int,
    endpoint_tol: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if int(cycle) <= 0:
        raise ValueError("Particle-transfer exponents require a positive completed cycle.")
    values = np.clip(np.real(a_values), -1.0, 1.0)
    particle_poles = values <= (-1.0 + float(endpoint_tol))
    particle_zeros = values >= (1.0 - float(endpoint_tol))
    finite = ~(particle_poles | particle_zeros)
    exponents = np.empty_like(values, dtype=np.float64)
    exponents[particle_poles] = np.inf
    exponents[particle_zeros] = -np.inf
    exponents[finite] = (
        np.log1p(-values[finite]) - np.log1p(values[finite])
    ) / (2.0 * float(cycle))
    return exponents, finite, particle_zeros, particle_poles


def _rooted_singular_values_from_log_spectrum(log_spectrum: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore", invalid="ignore"):
        return np.exp(np.asarray(log_spectrum, dtype=np.float64))


def _nan_transfer_matrix(nlayer: int) -> np.ndarray:
    return np.full((int(nlayer), int(nlayer)), np.nan + 1j * np.nan, dtype=np.complex128)


def _exact_particle_single(
    sigma_ll: np.ndarray,
    *,
    cycle: int,
    endpoint_tol: float,
    choi_spectral_tol: float,
    n_eigenstates: int,
    include_all_finite_eigenstates: bool = False,
) -> dict[str, Any]:
    a_herm = _hermitize(np.asarray(sigma_ll, dtype=np.complex128))
    a_values, a_vectors = np.linalg.eigh(a_herm)
    raw_min = float(np.real(a_values[0]))
    raw_max = float(np.real(a_values[-1]))
    if raw_min < -1.0 - float(choi_spectral_tol) or raw_max > 1.0 + float(choi_spectral_tol):
        raise FloatingPointError(
            "Sigma_LL violates the pure-Choi spectral interval: "
            f"eigenvalue range=[{raw_min:.6e}, {raw_max:.6e}], "
            f"allowed tolerance={float(choi_spectral_tol):.6e}."
        )
    clipped = np.clip(np.real(a_values), -1.0, 1.0)
    exponents, finite, particle_zeros, particle_poles = _exponents_from_a_values(
        clipped, cycle=int(cycle), endpoint_tol=float(endpoint_tol)
    )
    finite_count = int(np.count_nonzero(finite))
    score = np.where(finite, np.abs(exponents), np.inf)
    all_finite_selected = np.argsort(score, kind="mergesort")[:finite_count]
    n_select = min(finite_count, int(n_eigenstates))
    selected = all_finite_selected[:n_select]
    spectrum_order = np.argsort(exponents, kind="mergesort")
    spectrum = exponents[spectrum_order]
    inverse_order = np.empty_like(spectrum_order)
    inverse_order[spectrum_order] = np.arange(spectrum_order.size)
    near_gap_exponents = np.full((int(n_eigenstates),), np.nan, dtype=np.float64)
    near_gap_a_eigenvalues = np.full((int(n_eigenstates),), np.nan, dtype=np.float64)
    selected_indices = np.full((int(n_eigenstates),), -1, dtype=np.int64)
    residuals = np.full((int(n_eigenstates),), np.nan, dtype=np.float64)
    eigenstates = np.full(
        (a_vectors.shape[0], int(n_eigenstates)),
        np.nan + 1j * np.nan,
        dtype=np.complex128,
    )
    if n_select:
        selected_vectors = a_vectors[:, selected]
        near_gap_exponents[:n_select] = exponents[selected]
        near_gap_a_eigenvalues[:n_select] = clipped[selected]
        selected_indices[:n_select] = inverse_order[selected]
        eigenstates[:, :n_select] = selected_vectors
        residuals[:n_select] = np.linalg.norm(
            a_herm @ selected_vectors - selected_vectors * clipped[selected][None, :],
            axis=0,
        )
    result = {
        "spectrum": spectrum,
        "gap": float(score[selected[0]]) if n_select else np.nan,
        "near_gap_exponents": near_gap_exponents,
        "near_gap_a_eigenvalues": near_gap_a_eigenvalues,
        "near_gap_residuals": residuals,
        "eigenstates": eigenstates,
        "selected_indices": selected_indices,
        "finite_eigenstate_count": finite_count,
        "near_gap_valid_count": n_select,
        "particle_zero_count": int(np.count_nonzero(particle_zeros)),
        "particle_pole_count": int(np.count_nonzero(particle_poles)),
        "a_eigenvalue_min": raw_min,
        "a_eigenvalue_max": raw_max,
    }
    if include_all_finite_eigenstates:
        finite_eigenstates = np.full(
            (a_vectors.shape[0], a_vectors.shape[0]),
            np.nan + 1j * np.nan,
            dtype=np.complex128,
        )
        finite_eigenstate_exponents = np.full((a_vectors.shape[0],), np.nan, dtype=np.float64)
        finite_eigenstate_indices = np.full((a_vectors.shape[0],), -1, dtype=np.int64)
        finite_eigenstate_a_eigenvalues = np.full((a_vectors.shape[0],), np.nan, dtype=np.float64)
        if finite_count:
            finite_eigenstates[:, :finite_count] = a_vectors[:, all_finite_selected]
            finite_eigenstate_exponents[:finite_count] = exponents[all_finite_selected]
            finite_eigenstate_indices[:finite_count] = inverse_order[all_finite_selected]
            finite_eigenstate_a_eigenvalues[:finite_count] = clipped[all_finite_selected]
        result.update(
            {
                "finite_eigenstates": finite_eigenstates,
                "finite_eigenstate_exponents": finite_eigenstate_exponents,
                "finite_eigenstate_indices": finite_eigenstate_indices,
                "finite_eigenstate_a_eigenvalues": finite_eigenstate_a_eigenvalues,
                "finite_eigenstate_valid_count": finite_count,
            }
        )
    return result


class CpuParticleChoiGapObserver:
    def __init__(
        self,
        *,
        samples_expected: int,
        cycles: Iterable[int],
        final_cycle: int,
        nlayer: int,
        n_eigenstates: int = N_EIGENSTATES,
        endpoint_tol: float = ENDPOINT_TOL,
        choi_spectral_tol: float = CHOI_SPECTRAL_TOL,
        censor_unstable: bool = True,
        choi_entry_abs_tol: float = CHOI_ENTRY_ABS_TOL,
        validate_choi: bool = False,
        choi_hermiticity_tol: float = 1e-10,
        choi_involution_tol: float = 1e-8,
        active_top_layer_indices: np.ndarray | None = None,
        full_nlayer: int | None = None,
        save_transfer_matrices: bool = False,
        transfer_matrix_max_abs: float = TRANSFER_MATRIX_MAX_ABS_TOL,
        transfer_matrix_config_id: str = "",
    ) -> None:
        self.samples_expected = int(samples_expected)
        self.cycles = [int(cycle) for cycle in cycles]
        self.final_cycle = int(final_cycle)
        if self.final_cycle not in self.cycles:
            raise ValueError("final_cycle must be included in cycles.")
        self.nlayer = int(nlayer)
        self.n_eigenstates = int(n_eigenstates)
        self.endpoint_tol = float(endpoint_tol)
        self.choi_spectral_tol = float(choi_spectral_tol)
        self.censor_unstable = bool(censor_unstable)
        self.choi_entry_abs_tol = float(choi_entry_abs_tol)
        self.validate_choi = bool(validate_choi)
        self.choi_hermiticity_tol = float(choi_hermiticity_tol)
        self.choi_involution_tol = float(choi_involution_tol)
        self.save_transfer_matrices = bool(save_transfer_matrices)
        self.transfer_matrix_max_abs_tol = float(transfer_matrix_max_abs)
        self.transfer_matrix_config_id = str(transfer_matrix_config_id)
        self._cycle_to_index = {cycle: idx for idx, cycle in enumerate(self.cycles)}

        shape = (self.samples_expected, len(self.cycles))
        near_shape = shape + (self.n_eigenstates,)
        self.gap = np.full(shape, np.nan, dtype=np.float64)
        self.spectrum_by_cycle = np.full(shape + (self.nlayer,), np.nan, dtype=np.float64)
        self.near_gap_exponents = np.full(near_shape, np.nan, dtype=np.float64)
        self.near_gap_a_eigenvalues = np.full(near_shape, np.nan, dtype=np.float64)
        self.near_gap_residuals = np.full(near_shape, np.nan, dtype=np.float64)
        self.solver_iterations = np.zeros(shape, dtype=np.int64)
        self.used_exact_fallback = np.ones(shape, dtype=np.bool_)
        self.endpoint_counts_evaluated = np.zeros(shape, dtype=np.bool_)
        self.particle_zero_count = np.full(shape, -1, dtype=np.int64)
        self.particle_pole_count = np.full(shape, -1, dtype=np.int64)
        self.finite_exponent_count = np.full(shape, -1, dtype=np.int64)
        self.near_gap_valid_count = np.full(shape, -1, dtype=np.int64)
        self.no_finite_exponents = np.zeros(shape, dtype=np.bool_)
        self.min_abs_d = np.full(shape, np.nan, dtype=np.float64)
        self.choi_active_at_observation = np.zeros(shape, dtype=np.bool_)
        self.stable_at_cycle = np.zeros(shape, dtype=np.bool_)
        self.hermiticity_residual = np.full(shape, np.nan, dtype=np.float64)
        self.involution_residual = np.full(shape, np.nan, dtype=np.float64)
        self.max_abs_choi_entry = np.full(shape, np.nan, dtype=np.float64)
        self.first_failure_cycle = np.full((self.samples_expected,), -1, dtype=np.int64)
        self.first_failure_reason = np.full((self.samples_expected,), "", dtype="<U96")
        self.final_spectrum = np.full((self.samples_expected, self.nlayer), np.nan, dtype=np.float64)
        self.final_eigenstates_near_gap = np.full(
            (self.samples_expected, self.nlayer, self.n_eigenstates),
            np.nan + 1j * np.nan,
            dtype=np.complex128,
        )
        self.final_eigenstate_exponents = np.full(
            (self.samples_expected, self.n_eigenstates), np.nan, dtype=np.float64
        )
        self.final_eigenstate_indices = np.full(
            (self.samples_expected, self.n_eigenstates), -1, dtype=np.int64
        )
        self.final_eigenstate_a_eigenvalues = np.full(
            (self.samples_expected, self.n_eigenstates), np.nan, dtype=np.float64
        )
        self.final_finite_eigenstates = np.full(
            (self.samples_expected, self.nlayer, self.nlayer),
            np.nan + 1j * np.nan,
            dtype=np.complex128,
        )
        self.final_finite_eigenstate_exponents = np.full(
            (self.samples_expected, self.nlayer), np.nan, dtype=np.float64
        )
        self.final_finite_eigenstate_indices = np.full(
            (self.samples_expected, self.nlayer), -1, dtype=np.int64
        )
        self.final_finite_eigenstate_a_eigenvalues = np.full(
            (self.samples_expected, self.nlayer), np.nan, dtype=np.float64
        )
        self.final_finite_eigenstate_valid_count = np.zeros((self.samples_expected,), dtype=np.int64)
        self.a_eigenvalue_min_final = np.full((self.samples_expected,), np.nan, dtype=np.float64)
        self.a_eigenvalue_max_final = np.full((self.samples_expected,), np.nan, dtype=np.float64)
        self.exact_vs_iterative_gap_error_final = np.full((self.samples_expected,), 0.0, dtype=np.float64)
        if self.save_transfer_matrices:
            self.transfer_matrix_by_cycle = np.full(
                shape + (self.nlayer, self.nlayer),
                np.nan + 1j * np.nan,
                dtype=np.complex128,
            )
            self.transfer_matrix_valid = np.zeros(shape, dtype=np.bool_)
            self.transfer_matrix_failure_reason = np.full(shape, "", dtype="<U160")
            self.transfer_matrix_max_abs = np.full(shape, np.nan, dtype=np.float64)
        else:
            self.transfer_matrix_by_cycle = None
            self.transfer_matrix_valid = None
            self.transfer_matrix_failure_reason = None
            self.transfer_matrix_max_abs = None
        self.denominator_context: list[dict[str, Any]] = []
        self.failure_details: list[dict[str, Any]] = []
        self._seen_engine_failure_records: set[str] = set()
        self.active_top_layer_indices = (
            None if active_top_layer_indices is None else np.asarray(active_top_layer_indices, dtype=np.int64).copy()
        )
        self.full_nlayer = None if full_nlayer is None else int(full_nlayer)

    def _record_failure(self, sample_idx: int, cycle: int, reason: str, detail: dict[str, Any]) -> None:
        if self.first_failure_cycle[int(sample_idx)] < 0:
            self.first_failure_cycle[int(sample_idx)] = int(cycle)
            self.first_failure_reason[int(sample_idx)] = str(reason)
        payload = dict(detail)
        payload.update({"sample_index": int(sample_idx), "cycle_observed": int(cycle), "reason": str(reason)})
        self.failure_details.append(payload)

    def _clear_selected(self, sample_idx: int, cidx: int) -> None:
        self.gap[sample_idx, cidx] = np.nan
        self.near_gap_exponents[sample_idx, cidx] = np.nan
        self.near_gap_a_eigenvalues[sample_idx, cidx] = np.nan
        self.near_gap_residuals[sample_idx, cidx] = np.nan
        self.solver_iterations[sample_idx, cidx] = -1
        self.used_exact_fallback[sample_idx, cidx] = False
        self.endpoint_counts_evaluated[sample_idx, cidx] = False
        self.particle_zero_count[sample_idx, cidx] = -1
        self.particle_pole_count[sample_idx, cidx] = -1
        self.finite_exponent_count[sample_idx, cidx] = -1
        self.near_gap_valid_count[sample_idx, cidx] = -1
        self.no_finite_exponents[sample_idx, cidx] = False

    def _record_transfer_matrix_failure(self, sample_idx: int, cidx: int, cycle: int, reason: str) -> None:
        if self.transfer_matrix_by_cycle is None:
            return
        self.transfer_matrix_by_cycle[sample_idx, cidx] = _nan_transfer_matrix(self.nlayer)
        self.transfer_matrix_valid[sample_idx, cidx] = False
        self.transfer_matrix_failure_reason[sample_idx, cidx] = str(reason)[:159]
        prefix = f"[transfer-matrix] {self.transfer_matrix_config_id}" if self.transfer_matrix_config_id else "[transfer-matrix]"
        print(
            f"{prefix} sample={int(sample_idx)} cycle={int(cycle)} invalid; saved NaN matrix. reason={reason}",
            file=sys.stderr,
            flush=True,
        )

    def _store_transfer_matrix(
        self,
        sample_idx: int,
        cidx: int,
        *,
        cycle: int,
        sigma_ll: np.ndarray | None,
        sigma_lr: np.ndarray | None,
        reason_if_unavailable: str = "",
    ) -> None:
        if self.transfer_matrix_by_cycle is None:
            return
        if reason_if_unavailable:
            self._record_transfer_matrix_failure(sample_idx, cidx, cycle, reason_if_unavailable)
            return
        try:
            if sigma_ll is None or sigma_lr is None:
                raise FloatingPointError("missing_choi_blocks")
            ll = np.asarray(sigma_ll, dtype=np.complex128)
            lr = np.asarray(sigma_lr, dtype=np.complex128)
            if ll.shape != (self.nlayer, self.nlayer) or lr.shape != (self.nlayer, self.nlayer):
                raise FloatingPointError(f"shape_mismatch_ll{ll.shape}_lr{lr.shape}")
            if not (np.all(np.isfinite(ll)) and np.all(np.isfinite(lr))):
                raise FloatingPointError("nonfinite_transfer_input")
            lhs = np.eye(self.nlayer, dtype=np.complex128) + ll
            transfer = -np.linalg.solve(lhs, lr)
            if not np.all(np.isfinite(transfer)):
                raise FloatingPointError("nonfinite_transfer_matrix")
            max_abs = float(np.max(np.abs(transfer))) if transfer.size else 0.0
            self.transfer_matrix_max_abs[sample_idx, cidx] = max_abs
            if max_abs > self.transfer_matrix_max_abs_tol:
                raise FloatingPointError(f"transfer_matrix_out_of_bounds:{max_abs:.6e}")
            self.transfer_matrix_by_cycle[sample_idx, cidx] = transfer
            self.transfer_matrix_valid[sample_idx, cidx] = True
            self.transfer_matrix_failure_reason[sample_idx, cidx] = ""
        except (FloatingPointError, RuntimeError, np.linalg.LinAlgError, ValueError) as exc:
            self._record_transfer_matrix_failure(sample_idx, cidx, cycle, f"{type(exc).__name__}:{exc}")

    def _store_exact(self, sample_idx: int, cidx: int, result: dict[str, Any]) -> None:
        self.gap[sample_idx, cidx] = float(result["gap"])
        self.spectrum_by_cycle[sample_idx, cidx] = result["spectrum"]
        self.near_gap_exponents[sample_idx, cidx] = result["near_gap_exponents"]
        self.near_gap_a_eigenvalues[sample_idx, cidx] = result["near_gap_a_eigenvalues"]
        self.near_gap_residuals[sample_idx, cidx] = result["near_gap_residuals"]
        self.solver_iterations[sample_idx, cidx] = 0
        self.used_exact_fallback[sample_idx, cidx] = True
        self.endpoint_counts_evaluated[sample_idx, cidx] = True
        self.particle_zero_count[sample_idx, cidx] = int(result["particle_zero_count"])
        self.particle_pole_count[sample_idx, cidx] = int(result["particle_pole_count"])
        self.finite_exponent_count[sample_idx, cidx] = int(result["finite_eigenstate_count"])
        self.near_gap_valid_count[sample_idx, cidx] = int(result["near_gap_valid_count"])
        self.no_finite_exponents[sample_idx, cidx] = int(result["finite_eigenstate_count"]) == 0

    def __call__(
        self,
        *,
        cycle: int,
        sigma_ll: np.ndarray,
        sigma_lr: np.ndarray,
        sigma_rr: np.ndarray,
        batch_index: int,
        batch_start: int,
        batch_count: int,
        min_abs_d: float,
        min_abs_d_context: dict[str, Any] | None,
        choi_active_mask: np.ndarray | None = None,
        choi_failure_records: tuple[dict[str, Any], ...] = (),
        active_top_layer_indices: np.ndarray | None = None,
        full_nlayer: int | None = None,
    ) -> dict[str, Any] | None:
        if int(cycle) not in self._cycle_to_index:
            raise ValueError(f"Observed unconfigured Choi cycle {cycle}.")
        cidx = self._cycle_to_index[int(cycle)]
        start = int(batch_start)
        stop = start + int(batch_count)
        if stop > self.samples_expected or int(sigma_ll.shape[0]) != int(batch_count):
            raise ValueError("Observed Choi batch does not match the configured sample slice.")
        if int(sigma_ll.shape[-1]) != self.nlayer:
            raise ValueError(f"Expected Choi block dimension {self.nlayer}, got {int(sigma_ll.shape[-1])}.")
        if active_top_layer_indices is not None:
            basis = np.asarray(active_top_layer_indices, dtype=np.int64).reshape(-1)
            if self.active_top_layer_indices is None:
                self.active_top_layer_indices = basis.copy()
            elif not np.array_equal(self.active_top_layer_indices, basis):
                raise ValueError("Active Choi basis changed during observation.")
        if full_nlayer is not None:
            if self.full_nlayer is None:
                self.full_nlayer = int(full_nlayer)
            elif self.full_nlayer != int(full_nlayer):
                raise ValueError("Full top-layer dimension changed during observation.")

        active = (
            np.ones((int(batch_count),), dtype=bool)
            if choi_active_mask is None
            else np.asarray(choi_active_mask, dtype=bool).reshape(-1)
        )
        self.choi_active_at_observation[start:stop, cidx] = active
        self.min_abs_d[start:stop, cidx] = float(min_abs_d)
        if min_abs_d_context is not None:
            self.denominator_context.append(
                {
                    "cycle_observed": int(cycle),
                    "batch_index": int(batch_index),
                    "batch_start": int(batch_start),
                    "min_abs_d": float(min_abs_d),
                    "context": dict(min_abs_d_context),
                }
            )

        for record in choi_failure_records:
            if record.get("stage") == "observer":
                continue
            key = json.dumps(record, sort_keys=True)
            if key in self._seen_engine_failure_records:
                continue
            self._seen_engine_failure_records.add(key)
            local_idx = int(record["sample_offset"])
            if record.get("stage") == "denominator_regularized":
                payload = dict(record)
                payload.update(
                    {
                        "sample_index": int(start + local_idx),
                        "cycle_observed": int(record.get("cycle", cycle)),
                        "reason": "denominator_regularized",
                    }
                )
                self.failure_details.append(payload)
                continue
            self._record_failure(start + local_idx, int(record.get("cycle", cycle)), "denominator", dict(record))

        deactivate: list[int] = []
        returned_records: list[dict[str, Any]] = []
        for local_idx in range(int(batch_count)):
            sample_idx = start + local_idx
            if not bool(active[local_idx]):
                self._store_transfer_matrix(
                    sample_idx,
                    cidx,
                    cycle=int(cycle),
                    sigma_ll=None,
                    sigma_lr=None,
                    reason_if_unavailable="choi_inactive",
                )
                if self.first_failure_cycle[sample_idx] < 0:
                    self._record_failure(sample_idx, int(cycle), "engine_censored", {"stage": "engine"})
                continue
            blocks = (sigma_ll[local_idx], sigma_lr[local_idx], sigma_rr[local_idx])
            self._store_transfer_matrix(
                sample_idx,
                cidx,
                cycle=int(cycle),
                sigma_ll=blocks[0],
                sigma_lr=blocks[1],
            )
            reason = ""
            if not all(np.all(np.isfinite(block)) for block in blocks):
                reason = "nonfinite_choi_block"
            else:
                max_entry = max(float(np.max(np.abs(block))) for block in blocks)
                self.max_abs_choi_entry[sample_idx, cidx] = max_entry
                if max_entry > 1.0 + self.choi_entry_abs_tol:
                    reason = "choi_entry_out_of_bounds"
            if not reason and self.validate_choi:
                sigma = np.block([[blocks[0], blocks[1]], [blocks[1].conj().T, blocks[2]]])
                identity = np.eye(2 * self.nlayer, dtype=np.complex128)
                herm = float(np.linalg.norm(sigma - sigma.conj().T, ord="fro"))
                invol = float(np.linalg.norm(sigma @ sigma - identity, ord="fro"))
                self.hermiticity_residual[sample_idx, cidx] = herm
                self.involution_residual[sample_idx, cidx] = invol
                if herm > self.choi_hermiticity_tol:
                    reason = "choi_hermiticity_residual"
                elif invol > self.choi_involution_tol:
                    reason = "choi_involution_residual"
            if reason:
                if not self.censor_unstable:
                    raise FloatingPointError(f"Unstable Choi trajectory at cycle {cycle}: {reason}.")
                record = {"stage": "observer", "cycle": int(cycle), "sample_offset": int(local_idx), "reason": reason}
                deactivate.append(local_idx)
                returned_records.append(record)
                self._record_failure(sample_idx, int(cycle), reason, record)
                continue
            try:
                exact = _exact_particle_single(
                    sigma_ll[local_idx],
                    cycle=int(cycle),
                    endpoint_tol=self.endpoint_tol,
                    choi_spectral_tol=self.choi_spectral_tol,
                    n_eigenstates=self.n_eigenstates,
                    include_all_finite_eigenstates=int(cycle) == self.final_cycle,
                )
                self._store_exact(sample_idx, cidx, exact)
                if int(cycle) == self.final_cycle:
                    self.final_spectrum[sample_idx] = exact["spectrum"]
                    self.final_eigenstates_near_gap[sample_idx] = exact["eigenstates"]
                    self.final_eigenstate_exponents[sample_idx] = exact["near_gap_exponents"]
                    self.final_eigenstate_indices[sample_idx] = exact["selected_indices"]
                    self.final_eigenstate_a_eigenvalues[sample_idx] = exact["near_gap_a_eigenvalues"]
                    self.final_finite_eigenstates[sample_idx] = exact["finite_eigenstates"]
                    self.final_finite_eigenstate_exponents[sample_idx] = exact["finite_eigenstate_exponents"]
                    self.final_finite_eigenstate_indices[sample_idx] = exact["finite_eigenstate_indices"]
                    self.final_finite_eigenstate_a_eigenvalues[sample_idx] = exact[
                        "finite_eigenstate_a_eigenvalues"
                    ]
                    self.final_finite_eigenstate_valid_count[sample_idx] = int(
                        exact["finite_eigenstate_valid_count"]
                    )
                    self.a_eigenvalue_min_final[sample_idx] = float(exact["a_eigenvalue_min"])
                    self.a_eigenvalue_max_final[sample_idx] = float(exact["a_eigenvalue_max"])
                    self.exact_vs_iterative_gap_error_final[sample_idx] = 0.0
                self.stable_at_cycle[sample_idx, cidx] = True
            except (FloatingPointError, RuntimeError, np.linalg.LinAlgError) as exc:
                if not self.censor_unstable:
                    raise
                self._clear_selected(sample_idx, cidx)
                record = {
                    "stage": "observer",
                    "cycle": int(cycle),
                    "sample_offset": int(local_idx),
                    "reason": "spectral_or_solver_failure",
                    "message": str(exc),
                }
                deactivate.append(local_idx)
                returned_records.append(record)
                self._record_failure(sample_idx, int(cycle), "spectral_or_solver_failure", record)
        if deactivate:
            return {"deactivate_sample_offsets": deactivate, "failure_records": returned_records}
        return None

    def assert_complete(self) -> None:
        stable = self.stable_at_cycle
        finite_stable = stable & (self.finite_exponent_count > 0)
        if np.isnan(self.gap[finite_stable]).any():
            raise AssertionError("Accepted particle Choi gap values with finite modes contain NaNs.")
        slots = np.arange(self.n_eigenstates)[None, None, :]
        valid_slots = stable[:, :, None] & (slots < self.near_gap_valid_count[:, :, None])
        if np.isnan(self.near_gap_exponents[valid_slots]).any():
            raise AssertionError("Accepted particle Choi near-gap finite slots contain NaNs.")
        final_stable = stable[:, self._cycle_to_index[self.final_cycle]]
        if np.any(final_stable & ~self.endpoint_counts_evaluated[:, self._cycle_to_index[self.final_cycle]]):
            raise AssertionError("Stable final-cycle endpoint multiplicities were not evaluated exactly.")
        if np.any(
            final_stable
            & (
                self.final_finite_eigenstate_valid_count
                != self.finite_exponent_count[:, self._cycle_to_index[self.final_cycle]]
            )
        ):
            raise AssertionError("Final finite eigenstate counts do not match the final finite spectrum counts.")

    def gap_payload(self) -> dict[str, np.ndarray]:
        payload = {
            "cycles": np.asarray(self.cycles, dtype=np.int64),
            "gap": self.gap,
            "near_gap_exponents": self.near_gap_exponents,
            "near_gap_a_eigenvalues": self.near_gap_a_eigenvalues,
            "near_gap_residuals": self.near_gap_residuals,
            "solver_iterations": self.solver_iterations,
            "used_exact_fallback": self.used_exact_fallback,
            "endpoint_counts_evaluated": self.endpoint_counts_evaluated,
            "particle_zero_count": self.particle_zero_count,
            "particle_pole_count": self.particle_pole_count,
            "finite_exponent_count": self.finite_exponent_count,
            "near_gap_valid_count": self.near_gap_valid_count,
            "no_finite_exponents": self.no_finite_exponents,
            "stable_at_cycle": self.stable_at_cycle,
            "stable_fraction": self.stable_at_cycle.mean(axis=0),
            "first_failure_cycle": self.first_failure_cycle,
            "first_failure_reason": self.first_failure_reason,
        }
        if self.active_top_layer_indices is not None:
            payload["active_top_layer_indices"] = self.active_top_layer_indices
        return payload

    def final_spectrum_payload(self) -> dict[str, np.ndarray]:
        final_idx = self._cycle_to_index[self.final_cycle]
        payload = {
            "final_cycle": np.asarray(self.final_cycle, dtype=np.int64),
            "final_spectrum": self.final_spectrum,
            "final_rooted_singular_values": _rooted_singular_values_from_log_spectrum(self.final_spectrum),
            "particle_zero_count_final": self.particle_zero_count[:, final_idx],
            "particle_pole_count_final": self.particle_pole_count[:, final_idx],
            "finite_exponent_count_final": self.finite_exponent_count[:, final_idx],
            "near_gap_valid_count_final": self.near_gap_valid_count[:, final_idx],
            "no_finite_exponents_final": self.no_finite_exponents[:, final_idx],
            "final_cycle_survivor": self.stable_at_cycle[:, final_idx],
        }
        if self.active_top_layer_indices is not None:
            payload["active_top_layer_indices"] = self.active_top_layer_indices
        return payload

    def spectrum_by_cycle_payload(self) -> dict[str, np.ndarray]:
        payload = {
            "cycles": np.asarray(self.cycles, dtype=np.int64),
            "spectrum": self.spectrum_by_cycle,
            "rooted_singular_values": _rooted_singular_values_from_log_spectrum(self.spectrum_by_cycle),
            "stable_at_cycle": self.stable_at_cycle,
            "stable_fraction": self.stable_at_cycle.mean(axis=0),
            "particle_zero_count": self.particle_zero_count,
            "particle_pole_count": self.particle_pole_count,
            "finite_exponent_count": self.finite_exponent_count,
            "near_gap_valid_count": self.near_gap_valid_count,
            "no_finite_exponents": self.no_finite_exponents,
            "first_failure_cycle": self.first_failure_cycle,
            "first_failure_reason": self.first_failure_reason,
        }
        if self.active_top_layer_indices is not None:
            payload["active_top_layer_indices"] = self.active_top_layer_indices
        return payload

    def transfer_matrix_payload(self) -> dict[str, np.ndarray]:
        if self.transfer_matrix_by_cycle is None:
            raise RuntimeError("Transfer matrix saving was not enabled for this observer.")
        payload = {
            "cycles": np.asarray(self.cycles, dtype=np.int64),
            "transfer_matrix": self.transfer_matrix_by_cycle,
            "transfer_matrix_valid": self.transfer_matrix_valid,
            "transfer_matrix_failure_reason": self.transfer_matrix_failure_reason,
            "transfer_matrix_max_abs": self.transfer_matrix_max_abs,
        }
        if self.active_top_layer_indices is not None:
            payload["active_top_layer_indices"] = self.active_top_layer_indices
        if self.full_nlayer is not None:
            payload["full_nlayer"] = np.asarray(self.full_nlayer, dtype=np.int64)
        return payload

    def final_eigenstates_payload(self) -> dict[str, np.ndarray]:
        payload = {
            "final_cycle": np.asarray(self.final_cycle, dtype=np.int64),
            "eigenstates_near_gap": self.final_eigenstates_near_gap,
            "eigenstate_exponents": self.final_eigenstate_exponents,
            "eigenstate_indices": self.final_eigenstate_indices,
            "eigenstate_a_eigenvalues": self.final_eigenstate_a_eigenvalues,
            "near_gap_valid_count_final": self.near_gap_valid_count[:, self._cycle_to_index[self.final_cycle]],
            "finite_exponent_count_final": self.finite_exponent_count[:, self._cycle_to_index[self.final_cycle]],
            "final_cycle_survivor": self.stable_at_cycle[:, self._cycle_to_index[self.final_cycle]],
        }
        if self.active_top_layer_indices is not None:
            payload["active_top_layer_indices"] = self.active_top_layer_indices
        return payload

    def final_finite_eigenstates_payload(self) -> dict[str, np.ndarray]:
        payload = {
            "final_cycle": np.asarray(self.final_cycle, dtype=np.int64),
            "finite_eigenstates": self.final_finite_eigenstates,
            "finite_eigenstate_exponents": self.final_finite_eigenstate_exponents,
            "finite_eigenstate_indices": self.final_finite_eigenstate_indices,
            "finite_eigenstate_a_eigenvalues": self.final_finite_eigenstate_a_eigenvalues,
            "finite_eigenstate_valid_count_final": self.final_finite_eigenstate_valid_count,
            "final_cycle_survivor": self.stable_at_cycle[:, self._cycle_to_index[self.final_cycle]],
        }
        if self.active_top_layer_indices is not None:
            payload["active_top_layer_indices"] = self.active_top_layer_indices
        return payload

    def diagnostics_payload(self) -> dict[str, np.ndarray]:
        payload = {
            "cycles": np.asarray(self.cycles, dtype=np.int64),
            "min_abs_d": self.min_abs_d,
            "a_eigenvalue_min_final": self.a_eigenvalue_min_final,
            "a_eigenvalue_max_final": self.a_eigenvalue_max_final,
            "exact_vs_iterative_gap_error_final": self.exact_vs_iterative_gap_error_final,
            "used_exact_fallback": self.used_exact_fallback,
            "solver_iterations": self.solver_iterations,
            "finite_exponent_count": self.finite_exponent_count,
            "near_gap_valid_count": self.near_gap_valid_count,
            "no_finite_exponents": self.no_finite_exponents,
            "particle_zero_count": self.particle_zero_count,
            "particle_pole_count": self.particle_pole_count,
            "hermiticity_residual": self.hermiticity_residual,
            "involution_residual": self.involution_residual,
            "max_abs_choi_entry": self.max_abs_choi_entry,
            "choi_active_at_observation": self.choi_active_at_observation,
            "stable_at_cycle": self.stable_at_cycle,
            "stable_fraction": self.stable_at_cycle.mean(axis=0),
            "first_failure_cycle": self.first_failure_cycle,
            "first_failure_reason": self.first_failure_reason,
        }
        if self.active_top_layer_indices is not None:
            payload["active_top_layer_indices"] = self.active_top_layer_indices
        if self.full_nlayer is not None:
            payload["full_nlayer"] = np.asarray(self.full_nlayer, dtype=np.int64)
        return payload


def config_id(cfg: dict[str, Any]) -> str:
    alpha_label = f"{float(cfg['alpha_1']):g}".replace(".", "p")
    alpha2_label = f"{float(cfg['alpha_2']):g}".replace(".", "p")
    boundary_label = "_openBC" if bool(cfg.get("open_boundary", False)) else ""
    return (
        f"N{cfg['Nx']}x{cfg['Ny']}_DW1{boundary_label}_dwtrunc1_slab"
        f"_a1-{alpha_label}_a2-{alpha2_label}_nsh{cfg['nshell']}_perfect_correction"
    )


def expected_paths(out_dir: Path) -> dict[str, Path]:
    return {
        "gap": out_dir / "particle_choi_transfer_gap_vs_cycle.npz",
        "spectrum_vs_cycle": out_dir / "particle_choi_transfer_spectrum_vs_cycle.npz",
        "spectrum": out_dir / "particle_choi_transfer_final_spectrum.npz",
        "eigenstates": out_dir / "particle_choi_transfer_final_eigenstates_near_gap.npz",
        "finite_eigenstates": out_dir / "particle_choi_transfer_final_finite_eigenstates.npz",
        "transfer_matrix_vs_cycle": out_dir / "particle_choi_transfer_matrix_vs_cycle.npz",
        "diagnostics": out_dir / "particle_choi_transfer_diagnostics.npz",
        "summary": out_dir / "run_summary.json",
        "latest": out_dir / "latest_run.json",
    }


def completed_row(
    cfg: dict[str, Any],
    out_dir: Path,
    overwrite: bool,
    *,
    quiet: bool = False,
) -> dict[str, Any] | None:
    latest = out_dir / "latest_run.json"
    if overwrite or not latest.exists():
        return None
    payload = json.loads(latest.read_text(encoding="utf-8"))
    if payload.get("status") != "completed" or payload.get("config_id") != config_id(cfg):
        return None
    summary_path = out_dir / "run_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    if summary.get("helper_version") == HELPER_VERSION:
        log(f"[skip] completed {config_id(cfg)}", quiet=quiet)
        return payload.get("run_index_row")
    log(f"[rerun] stale output found for {config_id(cfg)}", quiet=quiet)
    return None


def physical_health_summary(result: dict[str, Any], *, tol: float = PHYSICAL_HEALTH_TOL) -> dict[str, Any]:
    G_final = np.asarray(result["G_final"], dtype=np.complex128)
    if G_final.ndim != 3:
        raise FloatingPointError(f"Expected G_final with shape (samples,N,N), got {G_final.shape}.")
    if not np.all(np.isfinite(G_final)):
        raise FloatingPointError("Physical covariance contains non-finite entries.")
    herm = np.linalg.norm(G_final - np.swapaxes(G_final.conj(), -2, -1), axis=(-2, -1))
    eig_min = []
    eig_max = []
    for sample in G_final:
        vals = np.linalg.eigvalsh(_hermitize(sample))
        eig_min.append(float(vals[0]))
        eig_max.append(float(vals[-1]))
    eig_min_arr = np.asarray(eig_min, dtype=np.float64)
    eig_max_arr = np.asarray(eig_max, dtype=np.float64)
    max_herm = float(np.max(herm)) if herm.size else 0.0
    min_eval = float(np.min(eig_min_arr)) if eig_min_arr.size else np.nan
    max_eval = float(np.max(eig_max_arr)) if eig_max_arr.size else np.nan
    if max_herm > float(tol):
        raise FloatingPointError(f"Physical covariance hermiticity residual {max_herm:.6e} exceeds {tol:.6e}.")
    if min_eval < -1.0 - float(tol) or max_eval > 1.0 + float(tol):
        raise FloatingPointError(
            f"Physical covariance spectrum [{min_eval:.6e}, {max_eval:.6e}] exceeds [-1,1] by tol {tol:.6e}."
        )
    return {
        "physical_covariance_max_hermiticity_residual": max_herm,
        "physical_covariance_min_eigenvalue": min_eval,
        "physical_covariance_max_eigenvalue": max_eval,
    }


def run_engine_once(
    cfg: dict[str, Any],
    *,
    solver: str,
    progress: bool,
    quiet: bool,
    force_failure: bool = False,
) -> tuple[dict[str, Any], CpuParticleChoiGapObserver, dict[str, Any], float]:
    if force_failure:
        raise RuntimeError("Forced rank-1 failure for dense fallback smoke test.")
    stdout_context = (
        open(os.devnull, "w", encoding="utf-8") if quiet else contextlib.nullcontext()
    )
    with stdout_context as stdout_target:
        redirect_context = (
            contextlib.redirect_stdout(stdout_target) if quiet else contextlib.nullcontext()
        )
        with redirect_context:
            model = classA_U1FGTN(
                cfg["Nx"],
                cfg["Ny"],
                DW=cfg["DW"],
                nshell=cfg["nshell"],
                alpha_1=cfg["alpha_1"],
                alpha_2=cfg["alpha_2"],
                dw_truncation=cfg["dw_truncation"],
            )
            active = model.active_top_layer_indices(meas_slab_only=cfg["meas_slab_only"])
            n_active = int(np.asarray(active).size)
            if bool(cfg.get("open_boundary", False)):
                G_init = model.G_CI_domain_wall(periodic=False)
                initial_covariance = "G_CI_domain_wall"
                domain_wall_boundary_condition = "open"
                domain_wall_periodic = False
            else:
                G_init = None
                initial_covariance = "random_complex_fermion_covariance"
                domain_wall_boundary_condition = "periodic_default"
                domain_wall_periodic = True
            observer = CpuParticleChoiGapObserver(
                samples_expected=cfg["samples"],
                cycles=range(1, cfg["cycles"] + 1),
                final_cycle=cfg["cycles"],
                nlayer=n_active,
                n_eigenstates=N_EIGENSTATES,
                endpoint_tol=ENDPOINT_TOL,
                choi_spectral_tol=CHOI_SPECTRAL_TOL,
                censor_unstable=True,
                choi_entry_abs_tol=CHOI_ENTRY_ABS_TOL,
                active_top_layer_indices=np.asarray(active, dtype=np.int64),
                full_nlayer=model.Ntot // 2,
                save_transfer_matrices=bool(cfg.get("save_transfer_matrices", False)),
                transfer_matrix_max_abs=float(cfg.get("transfer_matrix_max_abs", TRANSFER_MATRIX_MAX_ABS_TOL)),
                transfer_matrix_config_id=config_id(cfg),
            )
            started = time.perf_counter()
            result = model.run_markov_circuit(
                G_history=False,
                progress=progress,
                cycles=cfg["cycles"],
                samples=cfg["samples"],
                postselect=False,
                perfect_correction=True,
                save=False,
                n_a=0.5,
                sequence="raster_y",
                meas_slab_only=cfg["meas_slab_only"],
                track_choi=True,
                choi_observer=observer,
                choi_observer_cycles=list(range(1, cfg["cycles"] + 1)),
                choi_singular_tol=CHOI_SINGULAR_TOL,
                choi_failure_mode=CHOI_FAILURE_MODE,
                physical_covariance_update=solver,
                init_mode="default",
                G_init=G_init,
            )
            result["initial_covariance"] = initial_covariance
            result["domain_wall_boundary_condition"] = domain_wall_boundary_condition
            result["domain_wall_periodic"] = bool(domain_wall_periodic)
            result["domain_wall_periodic_flag"] = f"periodic={domain_wall_periodic}"
            result["dw_locations"] = [int(x) for x in getattr(model, "DW_loc", [])]
            result["dw_slab_half_width_rule"] = str(
                getattr(model, "DW_slab_half_width_rule", "")
            )
            result["dw_slab_half_width"] = (
                int(getattr(model, "DW_slab_half_width"))
                if hasattr(model, "DW_slab_half_width")
                else None
            )
            result["dw_slab_width_sites"] = (
                int(getattr(model, "DW_slab_width_sites"))
                if hasattr(model, "DW_slab_width_sites")
                else None
            )
    elapsed_s = float(time.perf_counter() - started)
    observer.assert_complete()
    health = physical_health_summary(result)
    return result, observer, health, elapsed_s


def _shift_sample_detail(detail: dict[str, Any], sample_index: int) -> dict[str, Any]:
    shifted = dict(detail)
    shifted["sample_index"] = int(sample_index)
    if "sample_offset" in shifted:
        shifted["sample_offset"] = int(sample_index)
    if "batch_start" in shifted:
        shifted["batch_start"] = int(sample_index)
    context = shifted.get("context")
    if isinstance(context, dict):
        context = dict(context)
        context["sample_index"] = int(sample_index)
        context["batch_start"] = int(sample_index)
        shifted["context"] = context
    return shifted


def merge_sample_observers(
    observers: list[CpuParticleChoiGapObserver],
    *,
    samples_expected: int,
) -> CpuParticleChoiGapObserver:
    if len(observers) != int(samples_expected):
        raise ValueError(f"Expected {samples_expected} sample observers, got {len(observers)}.")
    first = observers[0]
    merged = CpuParticleChoiGapObserver(
        samples_expected=samples_expected,
        cycles=first.cycles,
        final_cycle=first.final_cycle,
        nlayer=first.nlayer,
        n_eigenstates=first.n_eigenstates,
        endpoint_tol=first.endpoint_tol,
        choi_spectral_tol=first.choi_spectral_tol,
        censor_unstable=first.censor_unstable,
        choi_entry_abs_tol=first.choi_entry_abs_tol,
        validate_choi=first.validate_choi,
        choi_hermiticity_tol=first.choi_hermiticity_tol,
        choi_involution_tol=first.choi_involution_tol,
        active_top_layer_indices=first.active_top_layer_indices,
        full_nlayer=first.full_nlayer,
        save_transfer_matrices=first.save_transfer_matrices,
        transfer_matrix_max_abs=first.transfer_matrix_max_abs_tol,
        transfer_matrix_config_id=first.transfer_matrix_config_id,
    )
    row_arrays = (
        "gap",
        "spectrum_by_cycle",
        "near_gap_exponents",
        "near_gap_a_eigenvalues",
        "near_gap_residuals",
        "solver_iterations",
        "used_exact_fallback",
        "endpoint_counts_evaluated",
        "particle_zero_count",
        "particle_pole_count",
        "finite_exponent_count",
        "near_gap_valid_count",
        "no_finite_exponents",
        "min_abs_d",
        "choi_active_at_observation",
        "stable_at_cycle",
        "hermiticity_residual",
        "involution_residual",
        "max_abs_choi_entry",
        "first_failure_cycle",
        "first_failure_reason",
        "final_spectrum",
        "final_eigenstates_near_gap",
        "final_eigenstate_exponents",
        "final_eigenstate_indices",
        "final_eigenstate_a_eigenvalues",
        "final_finite_eigenstates",
        "final_finite_eigenstate_exponents",
        "final_finite_eigenstate_indices",
        "final_finite_eigenstate_a_eigenvalues",
        "final_finite_eigenstate_valid_count",
        "a_eigenvalue_min_final",
        "a_eigenvalue_max_final",
        "exact_vs_iterative_gap_error_final",
    )
    if first.save_transfer_matrices:
        row_arrays = row_arrays + (
            "transfer_matrix_by_cycle",
            "transfer_matrix_valid",
            "transfer_matrix_failure_reason",
            "transfer_matrix_max_abs",
        )
    for sample_idx, observer in enumerate(observers):
        if observer.nlayer != merged.nlayer or observer.cycles != merged.cycles:
            raise ValueError("Cannot merge Choi observers with incompatible basis/cycle layouts.")
        for attr in row_arrays:
            getattr(merged, attr)[sample_idx] = getattr(observer, attr)[0]
        merged.denominator_context.extend(
            _shift_sample_detail(detail, sample_idx) for detail in observer.denominator_context
        )
        merged.failure_details.extend(
            _shift_sample_detail(detail, sample_idx) for detail in observer.failure_details
        )
    return merged


def _shift_choi_diagnostic(diag: dict[str, Any], sample_index: int) -> dict[str, Any]:
    shifted = dict(diag)
    shifted["batch_index"] = int(sample_index)
    shifted["batch_start"] = int(sample_index)
    records = shifted.get("choi_failure_records", [])
    shifted["choi_failure_records"] = [
        _shift_sample_detail(dict(record), sample_index) for record in records
    ]
    context = shifted.get("min_abs_d_context")
    if isinstance(context, dict):
        context = dict(context)
        context["sample_index"] = int(sample_index)
        context["batch_start"] = int(sample_index)
        shifted["min_abs_d_context"] = context
    return shifted


def merge_sample_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        raise ValueError("Cannot merge an empty result list.")
    first = dict(results[0])
    g_final = np.concatenate([np.asarray(result["G_final"], dtype=np.complex128) for result in results], axis=0)
    first["G_final"] = g_final
    first["G_final_avg"] = np.mean(g_final, axis=0)
    first["samples"] = int(g_final.shape[0])
    diagnostics = []
    for sample_idx, result in enumerate(results):
        diagnostics.extend(
            _shift_choi_diagnostic(dict(diag), sample_idx)
            for diag in result.get("choi_diagnostics", [])
        )
    first["choi_diagnostics"] = diagnostics
    return first


def _run_sample_worker(payload: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(payload["cfg"])
    cfg["samples"] = 1
    solver = str(payload["solver"])
    sample_index = int(payload["sample_index"])
    blas_threads = payload.get("blas_threads")
    apply_thread_env(blas_threads)
    limit_context = (
        threadpool_limits(blas_threads)
        if threadpool_limits is not None and blas_threads is not None
        else contextlib.nullcontext()
    )
    with limit_context:
        result, observer, health, elapsed_s = run_engine_once(
            cfg,
            solver=solver,
            progress=False,
            quiet=True,
            force_failure=bool(payload.get("force_failure", False)),
        )
    return {
        "sample_index": sample_index,
        "result": result,
        "observer": observer,
        "health": health,
        "elapsed_s": float(elapsed_s),
    }


def run_engine_samples_parallel(
    cfg: dict[str, Any],
    *,
    solver: str,
    sample_workers: int,
    blas_threads: int | None,
    progress: bool,
    quiet: bool,
    force_failure: bool = False,
) -> tuple[dict[str, Any], CpuParticleChoiGapObserver, dict[str, Any], float, list[float]]:
    samples = int(cfg["samples"])
    workers = max(1, min(int(sample_workers), samples))
    if workers <= 1 or samples <= 1:
        result, observer, health, elapsed_s = run_engine_once(
            cfg,
            solver=solver,
            progress=progress,
            quiet=quiet,
            force_failure=force_failure,
        )
        return result, observer, health, elapsed_s, [elapsed_s]

    started = time.perf_counter()
    payloads = [
        {
            "cfg": cfg,
            "solver": solver,
            "sample_index": sample_idx,
            "blas_threads": blas_threads,
            "force_failure": bool(force_failure),
        }
        for sample_idx in range(samples)
    ]
    rows: list[dict[str, Any] | None] = [None] * samples
    progress_iter = None
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_run_sample_worker, payload) for payload in payloads]
        iterator = as_completed(futures)
        if progress and not quiet:
            iterator = tqdm(iterator, total=len(futures), desc=f"{solver} samples", unit="sample", leave=False)
        for future in iterator:
            row = future.result()
            rows[int(row["sample_index"])] = row
    del progress_iter
    if any(row is None for row in rows):
        raise RuntimeError("Sample-parallel execution did not return every sample.")
    typed_rows = [row for row in rows if row is not None]
    results = [row["result"] for row in typed_rows]
    observers = [row["observer"] for row in typed_rows]
    merged_result = merge_sample_results(results)
    merged_observer = merge_sample_observers(observers, samples_expected=samples)
    health = physical_health_summary(merged_result)
    elapsed_s = float(time.perf_counter() - started)
    individual_elapsed = [float(row["elapsed_s"]) for row in typed_rows]
    return merged_result, merged_observer, health, elapsed_s, individual_elapsed


def run_one_config(
    cfg: dict[str, Any],
    *,
    campaign_root: Path,
    overwrite: bool,
    progress: bool,
    force_dense_fallback_smoke: bool,
    runtime_meta: dict[str, Any],
    quiet: bool,
    sample_workers: int,
    blas_threads: int | None,
) -> dict[str, Any]:
    out_dir = campaign_root / "runs" / config_id(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    old_row = completed_row(cfg, out_dir, overwrite=overwrite, quiet=quiet)
    if old_row is not None:
        return old_row

    cfg_id = config_id(cfg)
    log(
        (
            f"[config] start {cfg_id}: Ny={cfg['Ny']}, alpha_1={cfg['alpha_1']:g}, "
            f"cycles={cfg['cycles']}, samples={cfg['samples']}, output={out_dir}"
        ),
        quiet=quiet,
    )
    rank1_failure_message = ""
    rank1_failed = False
    result = None
    observer = None
    health: dict[str, Any] = {}
    elapsed_s = 0.0
    sample_elapsed_s: list[float] = []
    solver_used = "rank1"
    try:
        log(
            f"[solver] {cfg_id}: attempting rank1 with sample_workers={sample_workers}",
            quiet=quiet,
        )
        result, observer, health, elapsed_s, sample_elapsed_s = run_engine_samples_parallel(
            cfg,
            solver="rank1",
            sample_workers=sample_workers,
            blas_threads=blas_threads,
            progress=progress,
            quiet=quiet,
            force_failure=force_dense_fallback_smoke,
        )
        log(f"[solver] {cfg_id}: rank1 completed", quiet=quiet)
    except Exception as exc:
        rank1_failed = True
        rank1_failure_message = f"{type(exc).__name__}: {exc}"
        log(f"[fallback] {cfg_id}: rank1 failed; retrying dense. {rank1_failure_message}", quiet=quiet)
        try:
            log(
                f"[solver] {cfg_id}: attempting dense with sample_workers={sample_workers}",
                quiet=quiet,
            )
            result, observer, health, elapsed_s, sample_elapsed_s = run_engine_samples_parallel(
                cfg,
                solver="dense",
                sample_workers=sample_workers,
                blas_threads=blas_threads,
                progress=progress,
                quiet=quiet,
                force_failure=False,
            )
            solver_used = "dense"
            log(f"[solver] {cfg_id}: dense completed", quiet=quiet)
        except Exception as dense_exc:
            failure_payload = {
                "status": "failed",
                "config_id": cfg_id,
                "configuration": cfg,
                "canonical_dynamics_entry_point": CANONICAL_ENTRY_POINT,
                "physical_covariance_update_requested": "rank1",
                "physical_covariance_update_used": "failed",
                "rank1_failed_before_dense_fallback": True,
                "rank1_failure_message": rank1_failure_message,
                "dense_failure_message": f"{type(dense_exc).__name__}: {dense_exc}",
                "helper_version": HELPER_VERSION,
                "sample_workers": int(sample_workers),
                "sample_parallelized": bool(int(sample_workers) > 1 and int(cfg["samples"]) > 1),
                **runtime_meta,
            }
            write_json_atomic(out_dir / "latest_run.json", failure_payload)
            write_json_atomic(out_dir / "run_summary.json", failure_payload)
            log(
                f"[failed] {cfg_id}: dense failed after rank1 failure. output={out_dir}",
                quiet=quiet,
            )
            row = {
                "config_id": cfg_id,
                "Nx": cfg["Nx"],
                "Ny": cfg["Ny"],
                "cycles": cfg["cycles"],
                "alpha_1": cfg["alpha_1"],
                "alpha_2": cfg["alpha_2"],
                "DW": cfg["DW"],
                "dw_truncation": cfg["dw_truncation"],
                "samples": cfg["samples"],
                "status": "failed",
                "physical_covariance_update_used": "failed",
                "sample_workers": int(sample_workers),
                "output_dir": str(out_dir),
            }
            return row

    assert result is not None and observer is not None
    effective_slab = bool(cfg["DW"] and cfg["dw_truncation"] and cfg["meas_slab_only"])
    n_active = int(np.asarray(result["active_top_layer_indices"]).size)
    dw_locations = [int(x) for x in result.get("dw_locations", [])]
    if dw_locations:
        dw_slab_x_min = int(min(dw_locations))
        dw_slab_x_max = int(max(dw_locations))
        dw_slab_width_sites = int(dw_slab_x_max - dw_slab_x_min + 1)
    else:
        dw_slab_x_min = None
        dw_slab_x_max = None
        dw_slab_width_sites = None
    metadata = {
        **cfg,
        "config_id": cfg_id,
        "actual_samples": cfg["samples"],
        "Nlayer_active": n_active,
        "canonical_dynamics_entry_point": CANONICAL_ENTRY_POINT,
        "observable": OBSERVABLE,
        "helper_version": HELPER_VERSION,
        "choi_formula_version": CHOI_FORMULA_VERSION,
        "particle_transfer_definition": PARTICLE_TRANSFER_DEFINITION,
        "transfer_object": "T_p T_p^dagger",
        "final_eigenstate_index_convention": "indices_into_sorted_final_spectrum",
        "final_finite_eigenstates_saved": True,
        "finite_eigenstate_order": "increasing_abs_log_singular_exponent",
        "finite_eigenstate_padding": "NaN eigenvectors/exponents/a-eigenvalues and -1 spectrum indices",
        "gap_definition": GAP_DEFINITION,
        "gap_extraction": GAP_EXTRACTION,
        "final_spectrum_extraction": FINAL_SPECTRUM_EXTRACTION,
        "final_cycle_gap_override": "mandatory_exact_eigh_Sigma_LL",
        "spectrum_by_cycle_saved": True,
        "rooted_singular_values_saved": True,
        "transfer_matrix_saved_by_cycle": bool(cfg.get("save_transfer_matrices", False)),
        "transfer_matrix_definition": FULL_PARTICLE_TRANSFER_DEFINITION,
        "transfer_matrix_failure_policy": "nan_continue",
        "transfer_matrix_max_abs_tol": float(cfg.get("transfer_matrix_max_abs", TRANSFER_MATRIX_MAX_ABS_TOL)),
        "rooted_singular_value_definition": ROOTED_SINGULAR_VALUE_DEFINITION,
        "rooted_singular_value_endpoint_policy": "-inf log entries map to 0, +inf log entries map to inf, NaN stays NaN",
        "spectrum_by_cycle_extraction": "exact_eigh_Sigma_LL",
        "observer_censoring_policy": "do_not_censor_few_finite_modes_v3",
        "partial_finite_mode_policy": (
            "cycles with fewer than n_eigenstates finite particle-transfer exponents are accepted; "
            "near-gap slots are padded with NaN/-1 and counted by near_gap_valid_count"
        ),
        "cycles_rule": "cycles = 2 * Ny",
        "choi_basis": "reduced_topological_slab" if effective_slab else "full_top_layer",
        "meas_slab_only_requested": bool(cfg["meas_slab_only"]),
        "meas_slab_only_effective": effective_slab,
        "initial_covariance": result.get("initial_covariance", "random_complex_fermion_covariance"),
        "domain_wall_boundary_condition": result.get("domain_wall_boundary_condition", "periodic_default"),
        "domain_wall_periodic": bool(result.get("domain_wall_periodic", True)),
        "domain_wall_periodic_flag": result.get("domain_wall_periodic_flag", "periodic=True"),
        "domain_wall_locations": dw_locations,
        "dw_slab_half_width_rule": result.get("dw_slab_half_width_rule", ""),
        "dw_slab_half_width": result.get("dw_slab_half_width", None),
        "dw_slab_x_min": dw_slab_x_min,
        "dw_slab_x_max": dw_slab_x_max,
        "dw_slab_width_sites": result.get("dw_slab_width_sites", dw_slab_width_sites),
        "dw_slab_width_sites_from_locations": dw_slab_width_sites,
        "active_top_layer_indices": np.asarray(result["active_top_layer_indices"], dtype=np.int64).tolist(),
        "elapsed_s": elapsed_s,
        "solver": {
            "n_eigenstates": N_EIGENSTATES,
            "method": "exact_eigh_cpu",
            "exact_fallback_count": int(observer.used_exact_fallback.sum()),
        },
        "endpoint_counts": "evaluated exactly on every accepted CPU observation cycle",
        "choi_failure_mode": CHOI_FAILURE_MODE,
        "choi_singular_tol": CHOI_SINGULAR_TOL,
        "choi_entry_abs_tol": CHOI_ENTRY_ABS_TOL,
        "censoring_rule": (
            "exact Choi observable is NaN at and after first numerical-instability failure; "
            "few-finite-mode endpoint saturation is not censored; physical G evolution continues"
        ),
        "stable_fraction_by_cycle": observer.stable_at_cycle.mean(axis=0).tolist(),
        "stable_samples_final": int(observer.stable_at_cycle[:, -1].sum()),
        "first_failure_cycle": observer.first_failure_cycle.tolist(),
        "first_failure_reason": observer.first_failure_reason.tolist(),
        "observer_failure_details": observer.failure_details,
        "choi_diagnostics": result["choi_diagnostics"],
        "full_choi_or_transfer_matrices_saved": bool(cfg.get("save_transfer_matrices", False)),
        "transfer_matrix_saved_by_cycle": bool(cfg.get("save_transfer_matrices", False)),
        "transfer_matrix_definition": FULL_PARTICLE_TRANSFER_DEFINITION,
        "transfer_matrix_failure_policy": "nan_continue",
        "transfer_matrix_max_abs_tol": float(cfg.get("transfer_matrix_max_abs", TRANSFER_MATRIX_MAX_ABS_TOL)),
        "physical_covariance_update_requested": "rank1",
        "physical_covariance_update_used": solver_used,
        "rank1_failed_before_dense_fallback": bool(rank1_failed),
        "rank1_failure_message": rank1_failure_message,
        "dense_solver_regularization_policy": "exact solve first, _solve_regularized Tikhonov fallback",
        "sample_workers": int(sample_workers),
        "sample_parallelized": bool(int(sample_workers) > 1 and int(cfg["samples"]) > 1),
        "per_sample_elapsed_s": sample_elapsed_s,
        **runtime_meta,
        **health,
    }
    paths = expected_paths(out_dir)
    if not overwrite and any(path.exists() for path in paths.values()):
        raise FileExistsError(f"Output already exists for {config_id(cfg)}. Use --overwrite to replace it.")
    metadata_json = np.asarray(json.dumps(metadata, sort_keys=True))
    save_npz_atomic(paths["gap"], **observer.gap_payload(), metadata_json=metadata_json)
    save_npz_atomic(paths["spectrum_vs_cycle"], **observer.spectrum_by_cycle_payload(), metadata_json=metadata_json)
    save_npz_atomic(paths["spectrum"], **observer.final_spectrum_payload(), metadata_json=metadata_json)
    save_npz_atomic(paths["eigenstates"], **observer.final_eigenstates_payload(), metadata_json=metadata_json)
    save_npz_atomic(
        paths["finite_eigenstates"],
        **observer.final_finite_eigenstates_payload(),
        metadata_json=metadata_json,
    )
    if bool(cfg.get("save_transfer_matrices", False)):
        save_npz_atomic(
            paths["transfer_matrix_vs_cycle"],
            **observer.transfer_matrix_payload(),
            metadata_json=metadata_json,
        )
    save_npz_atomic(paths["diagnostics"], **observer.diagnostics_payload(), metadata_json=metadata_json)
    write_json_atomic(paths["summary"], metadata)
    row = {
        "config_id": cfg_id,
        "Nx": cfg["Nx"],
        "Ny": cfg["Ny"],
        "cycles": cfg["cycles"],
        "alpha_1": cfg["alpha_1"],
        "alpha_2": cfg["alpha_2"],
        "DW": cfg["DW"],
        "dw_truncation": cfg["dw_truncation"],
        "domain_wall_locations": dw_locations,
        "dw_slab_half_width_rule": result.get("dw_slab_half_width_rule", ""),
        "dw_slab_half_width": result.get("dw_slab_half_width", None),
        "dw_slab_width_sites": result.get("dw_slab_width_sites", dw_slab_width_sites),
        "samples": cfg["samples"],
        "Nlayer_active": n_active,
        "stable_samples_final": int(observer.stable_at_cycle[:, -1].sum()),
        "elapsed_s": elapsed_s,
        "status": "completed",
        "physical_covariance_update_used": solver_used,
        "rank1_failed_before_dense_fallback": bool(rank1_failed),
        "transfer_matrix_saved_by_cycle": bool(cfg.get("save_transfer_matrices", False)),
        "sample_workers": int(sample_workers),
        "output_dir": str(out_dir),
    }
    write_json_atomic(paths["latest"], {"status": "completed", "config_id": cfg_id, "run_index_row": row})
    log(
        (
            f"[production] {cfg_id} completed in {elapsed_s:.1f}s; "
            f"solver={solver_used}; stable_final={int(observer.stable_at_cycle[:, -1].sum())}/{cfg['samples']}; "
            f"output={out_dir}"
        ),
        quiet=quiet,
    )
    return row


def nx_values_from_args(args: argparse.Namespace) -> list[int]:
    if args.smoke:
        return [12]
    if args.nx_values is not None:
        return [int(v) for v in args.nx_values]
    return [int(args.nx)]


def campaign_id_from_args(args: argparse.Namespace) -> str:
    nx_values = nx_values_from_args(args)
    ny_values = [12] if args.smoke else args.ny_values
    alpha_values = [args.alpha_top_values[0]] if args.smoke else args.alpha_top_values
    samples = 1 if args.smoke else int(args.samples)
    nx_label = "-".join(str(int(v)) for v in nx_values)
    ny_label = "-".join(str(int(v)) for v in ny_values)
    alpha_label = "-".join(f"{float(v):g}".replace(".", "p") for v in alpha_values)
    boundary = "_openBC" if bool(args.open_boundary) else ""
    transfer = "_transfer" if bool(args.save_transfer_matrices) else ""
    suffix = "_smoke" if args.smoke else ""
    return f"Nx{nx_label}_Ny{ny_label}_DW1{boundary}_dwtrunc1{transfer}_a1-{alpha_label}_S{samples}_cycles2Ny{suffix}"


def build_configs(args: argparse.Namespace) -> list[dict[str, Any]]:
    nx_values = nx_values_from_args(args)
    ny_values = [12] if args.smoke else [int(v) for v in args.ny_values]
    alpha_values = [float(args.alpha_top_values[0])] if args.smoke else [float(v) for v in args.alpha_top_values]
    samples = 1 if args.smoke else int(args.samples)
    configs = []
    for nx in nx_values:
        for ny in ny_values:
            for alpha_1 in alpha_values:
                cycles = int(args.cycles_override) if args.cycles_override is not None else int(args.cycles_factor * ny)
                if args.smoke:
                    cycles = int(args.cycles_override) if args.cycles_override is not None else 2
                cfg = {
                    "Nx": int(nx),
                    "Ny": int(ny),
                    "cycles": int(cycles),
                    "samples": int(samples),
                    "nshell": 1,
                    "alpha_1": float(alpha_1),
                    "alpha_2": 30.0,
                    "protocol": "perfect_correction",
                    "DW": True,
                    "dw_truncation": True,
                    "meas_slab_only": True,
                    "open_boundary": bool(args.open_boundary),
                    "initial_covariance": "G_CI_domain_wall" if bool(args.open_boundary) else "random_complex_fermion_covariance",
                    "domain_wall_periodic": not bool(args.open_boundary),
                    "save_transfer_matrices": bool(args.save_transfer_matrices),
                    "transfer_matrix_max_abs": float(args.transfer_matrix_max_abs),
                }
                configs.append(cfg)
    return configs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parent / "cpu_data")
    parser.add_argument("--nx", type=int, default=12)
    parser.add_argument("--nx-values", type=int, nargs="+", default=None)
    parser.add_argument("--ny-values", type=int, nargs="+", default=[12, 16, 20])
    parser.add_argument("--alpha-top-values", type=float, nargs="+", default=[1.0, 3.0])
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--cycles-factor", type=int, default=2)
    parser.add_argument("--cycles-override", type=int, default=None)
    parser.add_argument("--campaign-id", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--cpu-list", type=str, default=None)
    parser.add_argument("--blas-threads", type=int, default=_PREPARSED_BLAS_THREADS)
    parser.add_argument("--sample-workers", type=int, default=1)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force-dense-fallback-smoke", action="store_true")
    parser.add_argument("--open-boundary", action="store_true")
    parser.add_argument("--save-transfer-matrices", action="store_true")
    parser.add_argument("--transfer-matrix-max-abs", type=float, default=TRANSFER_MATRIX_MAX_ABS_TOL)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.blas_threads = normalize_blas_threads(args.blas_threads)
    args.sample_workers = int(args.sample_workers)
    if args.sample_workers <= 0:
        raise ValueError("--sample-workers must be a positive integer.")
    args.transfer_matrix_max_abs = float(args.transfer_matrix_max_abs)
    if args.transfer_matrix_max_abs <= 0:
        raise ValueError("--transfer-matrix-max-abs must be positive.")
    apply_thread_env(args.blas_threads)
    cpu_affinity = apply_cpu_affinity(args.cpu_list, quiet=bool(args.quiet))
    progress_enabled = not bool(args.no_progress)
    runtime_meta = runtime_metadata(
        cpu_list_requested=args.cpu_list,
        cpu_affinity_effective=cpu_affinity,
        blas_threads_requested=args.blas_threads,
        progress_enabled=progress_enabled,
        quiet=bool(args.quiet),
    )
    configs = build_configs(args)
    campaign_id = args.campaign_id or campaign_id_from_args(args)
    campaign_root = (
        Path(args.output_root)
        / "complex_particle_choi_transfer"
        / "campaigns"
        / campaign_id
    )
    campaign_root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    log(
        (
            f"[campaign] start {campaign_id}: configs={len(configs)}, output={campaign_root}, "
            f"cpu_affinity={runtime_meta['cpu_affinity_effective']}, "
            f"blas_threads={runtime_meta['blas_threads_requested']}, "
            f"sample_workers={args.sample_workers}, progress={progress_enabled}"
        ),
        quiet=bool(args.quiet),
    )
    progress_iter = (
        tqdm(configs, desc="CPU Choi configs", unit="config")
        if progress_enabled and not bool(args.quiet)
        else configs
    )
    limit_context = (
        threadpool_limits(args.blas_threads)
        if threadpool_limits is not None and args.blas_threads is not None
        else contextlib.nullcontext()
    )
    rows = []
    with limit_context:
        for cfg in progress_iter:
            rows.append(
                run_one_config(
                    cfg,
                    campaign_root=campaign_root,
                    overwrite=bool(args.overwrite),
                    progress=progress_enabled,
                    force_dense_fallback_smoke=bool(args.force_dense_fallback_smoke),
                    runtime_meta=runtime_meta,
                    quiet=bool(args.quiet),
                    sample_workers=int(args.sample_workers),
                    blas_threads=args.blas_threads,
                )
            )
    runtime_meta = runtime_metadata(
        cpu_list_requested=args.cpu_list,
        cpu_affinity_effective=effective_cpu_affinity(),
        blas_threads_requested=args.blas_threads,
        progress_enabled=progress_enabled,
        quiet=bool(args.quiet),
    )
    rows = sorted(rows, key=lambda row: (int(row["Nx"]), int(row["Ny"]), float(row["alpha_1"])))
    save_csv_atomic(campaign_root / "run_index.csv", rows)
    manifest = {
        "status": "completed" if all(row.get("status") == "completed" for row in rows) else "partial",
        "campaign_name": "CPU complex particle Choi-transfer gap",
        "campaign_id": campaign_id,
        "canonical_dynamics_entry_point": CANONICAL_ENTRY_POINT,
        "observable": OBSERVABLE,
        "helper_version": HELPER_VERSION,
        "Nx_values": nx_values_from_args(args),
        "Ny_values": [int(v) for v in ([12] if args.smoke else args.ny_values)],
        "alpha_top_values": [float(v) for v in ([args.alpha_top_values[0]] if args.smoke else args.alpha_top_values)],
        "alpha_2": 30.0,
        "cycles_rule": "cycles = cycles_override if provided else cycles_factor * Ny",
        "cycles_override": None if args.cycles_override is None else int(args.cycles_override),
        "cycles_factor": int(args.cycles_factor),
        "samples": 1 if args.smoke else int(args.samples),
        "open_boundary": bool(args.open_boundary),
        "initial_covariance": "G_CI_domain_wall" if bool(args.open_boundary) else "random_complex_fermion_covariance",
        "domain_wall_boundary_condition": "open" if bool(args.open_boundary) else "periodic_default",
        "domain_wall_periodic": not bool(args.open_boundary),
        "physical_covariance_update_requested": "rank1",
        "final_finite_eigenstates_saved": True,
        "finite_eigenstate_order": "increasing_abs_log_singular_exponent",
        "rooted_singular_values_saved": True,
        "transfer_matrix_saved_by_cycle": bool(args.save_transfer_matrices),
        "transfer_matrix_definition": FULL_PARTICLE_TRANSFER_DEFINITION,
        "transfer_matrix_failure_policy": "nan_continue",
        "transfer_matrix_max_abs_tol": float(args.transfer_matrix_max_abs),
        "rooted_singular_value_definition": ROOTED_SINGULAR_VALUE_DEFINITION,
        "rooted_singular_value_endpoint_policy": "-inf log entries map to 0, +inf log entries map to inf, NaN stays NaN",
        "dense_fallback_policy": "rerun failed rank1 config with dense_regularized_v1",
        "dense_solver_regularization_policy": "exact solve first, _solve_regularized Tikhonov fallback",
        "observer_censoring_policy": "do_not_censor_few_finite_modes_v3",
        "partial_finite_mode_policy": (
            "few-finite-mode endpoint saturation is saved with finite_exponent_count "
            "and near_gap_valid_count instead of censoring the sample"
        ),
        "sample_workers": int(args.sample_workers),
        "sample_parallelized": bool(int(args.sample_workers) > 1 and (1 if args.smoke else int(args.samples)) > 1),
        "configs": configs,
        "results": rows,
        "elapsed_s": float(time.perf_counter() - started),
        "campaign_root": str(campaign_root),
        "run_index": str(campaign_root / "run_index.csv"),
        **runtime_meta,
    }
    write_json_atomic(campaign_root / "campaign_manifest.json", manifest)
    write_json_atomic(campaign_root / "latest_campaign.json", manifest)
    log(
        (
            f"[campaign] complete {campaign_id}: status={manifest['status']}, "
            f"elapsed={manifest['elapsed_s']:.1f}s, run_index={campaign_root / 'run_index.csv'}"
        ),
        quiet=bool(args.quiet),
    )
    if not bool(args.quiet):
        print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
