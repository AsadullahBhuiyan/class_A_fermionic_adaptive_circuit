#!/usr/bin/env python3
"""Small-system physical covariance update survival check."""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import MethodType

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
os.chdir(ROOT)
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/class_A_fermionic_adaptive_circuit_cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/class_A_fermionic_adaptive_circuit_matplotlib")

from fgtn.classA_U1FGTN import classA_U1FGTN


BLOWUP_THRESHOLD = 1e12


@dataclass
class PhysicalRecord:
    cycle: int
    max_abs_g_entry: float
    hermiticity_residual: float
    involution_residual: float
    eval_min: float
    eval_max: float
    nonfinite: bool


class PhysicalStateHealth:
    def __init__(self, history, blowup_threshold: float = BLOWUP_THRESHOLD):
        hist = np.asarray(history, dtype=np.complex128)
        if hist.ndim != 3 or hist.shape[-1] != hist.shape[-2]:
            raise ValueError(f"Expected one-sample G history with shape (T,N,N), got {hist.shape}.")
        self.history = hist
        self.blowup_threshold = float(blowup_threshold)
        self.records = [self._record(cycle, g) for cycle, g in enumerate(hist)]
        self.final_g = hist[-1].copy()

    def _record(self, cycle: int, g: np.ndarray) -> PhysicalRecord:
        nonfinite = not np.all(np.isfinite(g))
        if nonfinite:
            return PhysicalRecord(
                cycle=int(cycle),
                max_abs_g_entry=float("inf"),
                hermiticity_residual=float("inf"),
                involution_residual=float("inf"),
                eval_min=float("nan"),
                eval_max=float("nan"),
                nonfinite=True,
            )
        g_herm = 0.5 * (g + g.conj().T)
        evals = np.linalg.eigvalsh(g_herm)
        ident = np.eye(g.shape[0], dtype=np.complex128)
        return PhysicalRecord(
            cycle=int(cycle),
            max_abs_g_entry=float(np.max(np.abs(g))),
            hermiticity_residual=float(np.linalg.norm(g - g.conj().T, ord="fro")),
            involution_residual=float(np.linalg.norm(g @ g - ident, ord="fro")),
            eval_min=float(evals[0]),
            eval_max=float(evals[-1]),
            nonfinite=False,
        )

    @property
    def first_bad_cycle(self) -> int | None:
        for record in self.records[1:]:
            if record.nonfinite or record.max_abs_g_entry > self.blowup_threshold:
                return record.cycle
        return None

    @property
    def completed_50_cycles(self) -> bool:
        return len(self.records) >= 51 and self.records[-1].cycle == 50 and self.first_bad_cycle is None

    def summary(self, protocol: str, solver: str, elapsed_s: float) -> dict[str, object]:
        final = self.records[-1]
        return {
            "protocol": protocol,
            "solver": solver,
            "completed_50_cycles": self.completed_50_cycles,
            "first_bad_cycle": self.first_bad_cycle,
            "final_max_abs_G_entry": final.max_abs_g_entry,
            "max_G_entry_over_run": max(record.max_abs_g_entry for record in self.records[1:]),
            "final_G_hermiticity": final.hermiticity_residual,
            "final_G_involution": final.involution_residual,
            "final_eval_min": final.eval_min,
            "final_eval_max": final.eval_max,
            "elapsed_s": elapsed_s,
        }


def _rank1_resolvent_action(g_block: np.ndarray, chi: np.ndarray, rhs: np.ndarray, sign: float):
    """Apply (I + sign * G chi chi^dagger)^(-1) rhs."""
    v = g_block @ chi
    chi_rhs = chi.conj() @ rhs
    denom = 1.0 + float(sign) * (chi.conj() @ v)
    if abs(denom) < 1e-14:
        p = np.outer(chi, chi.conj())
        solve_mat = np.eye(g_block.shape[0], dtype=np.complex128) + float(sign) * g_block @ p
        eps_scale = np.linalg.norm(solve_mat, ord=np.inf)
        if not np.isfinite(eps_scale) or eps_scale < 1.0:
            eps_scale = 1.0
        try:
            return np.linalg.solve(solve_mat + 1e-9 * eps_scale * np.eye(g_block.shape[0]), rhs)
        except np.linalg.LinAlgError:
            return np.linalg.pinv(solve_mat) @ rhs
    return rhs - (float(sign) / denom) * v[:, None] * chi_rhs[None, :]


def _chi_from_projector(p: np.ndarray) -> np.ndarray:
    evals, evecs = np.linalg.eigh(0.5 * (p + p.conj().T))
    pos = int(np.argmax(evals))
    val = float(max(evals[pos], 0.0))
    return np.sqrt(val) * evecs[:, pos]


def dense_measure_only_top_layer(self, G, P, particle=True, symmetrize=True, chi=None):
    del chi
    G = np.asarray(G, dtype=np.complex128)
    P = np.asarray(P, dtype=np.complex128)
    nlayer = self.Ntot // 2
    gtt = G[:nlayer, :nlayer]
    eye = np.eye(nlayer, dtype=np.complex128)
    q = eye - P
    if particle:
        h11, h21, h22 = -P, q, P
    else:
        h11, h21, h22 = P, q, -P
    k = gtt @ h11 - eye
    l = gtt @ h21.conj().T
    gp = h22 - h21 @ self._solve_regularized(k, l, eps=1e-9)
    if symmetrize:
        gp = 0.5 * (gp + gp.conj().T)
    return gp


def dense_measure_only_top_layer_local(self, G, support_idx, comp_idx, chi_local, particle=True):
    G = np.asarray(G, dtype=np.complex128)
    support_idx = np.asarray(support_idx, dtype=np.int64).reshape(-1)
    comp_idx = np.asarray(comp_idx, dtype=np.int64).reshape(-1)
    g_ss, g_sr = self._local_support_block(G, support_idx, comp_idx)
    p = self._projector_from_vector(chi_local)
    eye = np.eye(support_idx.size, dtype=np.complex128)
    q = eye - p
    if particle:
        solve_mat = eye + g_ss @ p
        delta_sign = -1.0
        support_sign = 1.0
    else:
        solve_mat = eye - g_ss @ p
        delta_sign = 1.0
        support_sign = -1.0
    eps_scale = np.linalg.norm(solve_mat, ord=np.inf)
    if not np.isfinite(eps_scale) or eps_scale < 1.0:
        eps_scale = 1.0
    if comp_idx.size == 0:
        y = np.empty((support_idx.size, 0), dtype=np.complex128)
        py = y
        g_sr_new = y
    else:
        y = self._solve_regularized(solve_mat, g_sr, eps=1e-9 * eps_scale)
        py = p @ y
        g_sr_new = q @ y
    z = self._solve_regularized(solve_mat, g_ss @ q, eps=1e-9 * eps_scale)
    g_ss_new = support_sign * p + q @ z
    g_ss_new = 0.5 * (g_ss_new + g_ss_new.conj().T)
    gnew = np.array(G, copy=True)
    gnew[np.ix_(support_idx, support_idx)] = g_ss_new
    if comp_idx.size > 0:
        delta_rr = delta_sign * (g_sr.conj().T @ py)
        delta_rr = 0.5 * (delta_rr + delta_rr.conj().T)
        gnew[np.ix_(comp_idx, comp_idx)] += delta_rr
        gnew[np.ix_(support_idx, comp_idx)] = g_sr_new
        gnew[np.ix_(comp_idx, support_idx)] = g_sr_new.conj().T
    return 0.5 * (gnew + gnew.conj().T)


def sherman_measure_only_top_layer(self, G, P, particle=True, symmetrize=True, chi=None):
    G = np.asarray(G, dtype=np.complex128)
    P = np.asarray(P, dtype=np.complex128)
    nlayer = self.Ntot // 2
    gtt = G[:nlayer, :nlayer]
    chi = _chi_from_projector(P)
    q = np.eye(nlayer, dtype=np.complex128) - P
    sign = 1.0 if bool(particle) else -1.0
    v = gtt @ chi
    denom = 1.0 + sign * (chi.conj() @ v)
    gp = sign * P + q @ gtt @ q - (sign / denom) * np.outer(q @ v, chi.conj() @ gtt @ q)
    if symmetrize:
        gp = 0.5 * (gp + gp.conj().T)
    return gp


def sherman_measure_only_top_layer_local(self, G, support_idx, comp_idx, chi_local, particle=True):
    G = np.asarray(G, dtype=np.complex128)
    support_idx = np.asarray(support_idx, dtype=np.int64).reshape(-1)
    comp_idx = np.asarray(comp_idx, dtype=np.int64).reshape(-1)
    chi = np.asarray(chi_local, dtype=np.complex128).reshape(-1)
    g_ss, g_sr = self._local_support_block(G, support_idx, comp_idx)
    p = self._projector_from_vector(chi)
    q = np.eye(support_idx.size, dtype=np.complex128) - p
    sign = 1.0 if bool(particle) else -1.0
    v = g_ss @ chi
    denom = 1.0 + sign * (chi.conj() @ v)

    gnew = np.array(G, copy=True)
    if comp_idx.size == 0:
        y = np.empty((support_idx.size, 0), dtype=np.complex128)
        py = y
        g_sr_new = y
    else:
        y = g_sr - (sign / denom) * np.outer(v, chi.conj() @ g_sr)
        py = p @ y
        g_sr_new = q @ y

    z = g_ss @ q - (sign / denom) * np.outer(v, chi.conj() @ g_ss @ q)
    g_ss_new = sign * p + q @ z
    g_ss_new = 0.5 * (g_ss_new + g_ss_new.conj().T)
    gnew[np.ix_(support_idx, support_idx)] = g_ss_new

    if comp_idx.size > 0:
        delta_rr = -sign * (g_sr.conj().T @ py)
        delta_rr = 0.5 * (delta_rr + delta_rr.conj().T)
        gnew[np.ix_(comp_idx, comp_idx)] += delta_rr
        gnew[np.ix_(support_idx, comp_idx)] = g_sr_new
        gnew[np.ix_(comp_idx, support_idx)] = g_sr_new.conj().T
    return 0.5 * (gnew + gnew.conj().T)


def make_model() -> classA_U1FGTN:
    return classA_U1FGTN(
        Nx=4,
        Ny=8,
        DW=True,
        nshell=1,
        alpha_1=1.0,
        alpha_2=30.0,
        dw_truncation=True,
    )


def run_case(protocol: str, solver: str) -> tuple[dict[str, object], PhysicalStateHealth]:
    np.random.seed(12345)
    model = make_model()
    g_init = model.random_complex_fermion_covariance(
        N=model.Ntot // 2,
        rng=np.random.default_rng(20240601),
    )
    if solver == "dense":
        model.measure_only_top_layer = MethodType(dense_measure_only_top_layer, model)
        model._measure_only_top_layer_local = MethodType(dense_measure_only_top_layer_local, model)
    elif solver == "sherman":
        model.measure_only_top_layer = MethodType(sherman_measure_only_top_layer, model)
        model._measure_only_top_layer_local = MethodType(sherman_measure_only_top_layer_local, model)
    elif solver != "rank1":
        raise ValueError(f"unknown solver mode: {solver}")

    run_kwargs = {
        "G_history": True,
        "progress": False,
        "cycles": 50,
        "samples": 1,
        "save": False,
        "save_init": True,
        "G_init": g_init,
        "n_a": 0.5,
        "sequence": "raster_y",
        "meas_slab_only": True,
        "parallelize_samples": False,
        "track_choi": False,
    }
    if protocol == "postselect":
        run_kwargs["postselect"] = True
    elif protocol == "perfect_correction":
        run_kwargs["perfect_correction"] = True
    else:
        raise ValueError(f"unknown protocol: {protocol}")

    started = time.perf_counter()
    result = model.run_markov_circuit(**run_kwargs)
    elapsed_s = float(time.perf_counter() - started)
    history = np.asarray(result["G_hist"])[0]
    health = PhysicalStateHealth(history)
    return health.summary(protocol, solver, elapsed_s), health


def format_value(value: object) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        if np.isfinite(value):
            return f"{value:.3e}"
        return str(value)
    return str(value)


def print_table(rows: list[dict[str, object]], columns: list[str]) -> None:
    rendered = [[format_value(row.get(col)) for col in columns] for row in rows]
    widths = [
        max(len(col), *(len(row[idx]) for row in rendered))
        for idx, col in enumerate(columns)
    ]
    print("  ".join(col.ljust(widths[idx]) for idx, col in enumerate(columns)))
    print("  ".join("-" * widths[idx] for idx in range(len(columns))))
    for row in rendered:
        print("  ".join(row[idx].ljust(widths[idx]) for idx in range(len(columns))))


def comparison_rows(observers: dict[tuple[str, str], PhysicalStateHealth]) -> list[dict[str, object]]:
    rows = []
    for protocol in ("postselect", "perfect_correction"):
        if (protocol, "rank1") not in observers:
            continue
        rank1 = observers[(protocol, "rank1")]
        for solver in ("dense", "sherman"):
            if (protocol, solver) not in observers:
                continue
            other = observers[(protocol, solver)]
            diff = other.final_g - rank1.final_g
            rows.append(
                {
                    "protocol": protocol,
                    "solver_vs_rank1": solver,
                    "G_final_fro_diff": float(np.linalg.norm(diff, ord="fro")),
                    "G_final_max_diff": float(np.max(np.abs(diff))),
                    "same_survival_outcome": other.completed_50_cycles == rank1.completed_50_cycles,
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        choices=("all", "postselect", "perfect_correction"),
        default="all",
        help="Restrict the smoke test to one protocol.",
    )
    args = parser.parse_args()

    protocols = (
        ("postselect", "perfect_correction")
        if args.protocol == "all"
        else (args.protocol,)
    )
    summaries: list[dict[str, object]] = []
    observers: dict[tuple[str, str], PhysicalStateHealth] = {}
    for protocol in protocols:
        for solver in ("rank1", "dense", "sherman"):
            print(f"[run] protocol={protocol}, solver={solver}")
            summary, observer = run_case(protocol, solver)
            summaries.append(summary)
            observers[(protocol, solver)] = observer

    print("\nPhysical covariance survival summary")
    print_table(
        summaries,
        [
            "protocol",
            "solver",
            "completed_50_cycles",
            "first_bad_cycle",
            "final_max_abs_G_entry",
            "max_G_entry_over_run",
            "final_G_hermiticity",
            "final_G_involution",
            "final_eval_min",
            "final_eval_max",
            "elapsed_s",
        ],
    )

    print("\nReference modes vs production rank-1 final-state comparison")
    print_table(
        comparison_rows(observers),
        [
            "protocol",
            "solver_vs_rank1",
            "G_final_fro_diff",
            "G_final_max_diff",
            "same_survival_outcome",
        ],
    )


if __name__ == "__main__":
    main()
