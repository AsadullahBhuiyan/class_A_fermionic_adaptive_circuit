#!/usr/bin/env python3
"""Small-system GPU physical covariance update survival check."""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import MethodType

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
os.chdir(ROOT)
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/class_A_fermionic_adaptive_circuit_cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/class_A_fermionic_adaptive_circuit_matplotlib")

from fgtn.classA_U1FGTN_gpu import classA_U1FGTN_gpu


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
            return PhysicalRecord(cycle, float("inf"), float("inf"), float("inf"), float("nan"), float("nan"), True)
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

    def summary(self, protocol: str, solver: str, elapsed_s: float, device: str) -> dict[str, object]:
        final = self.records[-1]
        return {
            "protocol": protocol,
            "solver": solver,
            "device": device,
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


def deterministic_g_init(nlayer: int) -> np.ndarray:
    rng = np.random.default_rng(20240601)
    mat = rng.standard_normal((nlayer, nlayer)) + 1j * rng.standard_normal((nlayer, nlayer))
    herm = 0.5 * (mat + mat.conj().T)
    _, vecs = np.linalg.eigh(herm)
    diag = np.concatenate((np.ones(nlayer // 2), -np.ones(nlayer - nlayer // 2)))
    return vecs.conj().T @ np.diag(diag).astype(np.complex128) @ vecs


def dense_measure_only_top_layer_batched(self, G, P, particle=True, chi=None):
    del chi
    eye = self._eye_top if P.ndim == 2 else self._eye_top.unsqueeze(0)
    q = eye - P
    if particle:
        h11, h21, h22 = -P, q, P
    else:
        h11, h21, h22 = P, q, -P
    a = torch.matmul(G, h11) - eye
    b = torch.matmul(G, h21.conj().transpose(-2, -1))
    x = self._solve_regularized_batched(a, b, eps=1e-9)
    g_upd = h22 - torch.matmul(h21, x)
    return 0.5 * (g_upd + g_upd.conj().transpose(-2, -1))


def dense_measure_only_top_layer_local_batched(self, G, support_idx, comp_idx, chi_local, particle=True):
    g_ss, g_sr = self._local_support_block(G, support_idx, comp_idx)
    p = self._projector_from_vector(chi_local)
    q = self._eye_of_size(support_idx.numel()) - p
    if particle:
        solve_mat = self._eye_of_size(support_idx.numel()).unsqueeze(0) + torch.matmul(g_ss, p.unsqueeze(0))
        delta_sign = -1.0
        support_sign = 1.0
    else:
        solve_mat = self._eye_of_size(support_idx.numel()).unsqueeze(0) - torch.matmul(g_ss, p.unsqueeze(0))
        delta_sign = 1.0
        support_sign = -1.0
    y = self._solve_regularized_batched(solve_mat, g_sr, eps=1e-9)
    py = torch.matmul(p.unsqueeze(0), y)
    g_sr_new = torch.matmul(q.unsqueeze(0), y)
    z = self._solve_regularized_batched(solve_mat, torch.matmul(g_ss, q.unsqueeze(0)), eps=1e-9)
    g_ss_new = support_sign * p.unsqueeze(0) + torch.matmul(q.unsqueeze(0), z)
    g_ss_new = 0.5 * (g_ss_new + g_ss_new.conj().transpose(-2, -1))
    G[:, support_idx[:, None], support_idx[None, :]] = g_ss_new
    if comp_idx.numel() > 0:
        delta_rr = delta_sign * torch.matmul(g_sr.conj().transpose(-2, -1), py)
        delta_rr = 0.5 * (delta_rr + delta_rr.conj().transpose(-2, -1))
        G[:, comp_idx[:, None], comp_idx[None, :]] += delta_rr
        G[:, support_idx[:, None], comp_idx[None, :]] = g_sr_new
        G[:, comp_idx[:, None], support_idx[None, :]] = g_sr_new.conj().transpose(-2, -1)
    return G


def _chi_from_projector_torch(P):
    p_herm = 0.5 * (P + P.conj().transpose(-2, -1))
    evals, evecs = torch.linalg.eigh(p_herm)
    pos = torch.argmax(evals, dim=-1)
    if P.ndim == 2:
        return torch.sqrt(torch.clamp(evals[pos], min=0.0)).to(P.dtype) * evecs[:, int(pos.item())]
    vals = torch.clamp(torch.gather(evals, -1, pos.unsqueeze(-1)).squeeze(-1), min=0.0)
    gather_idx = pos[:, None, None].expand(-1, evecs.shape[-2], 1)
    return torch.sqrt(vals).to(P.dtype)[:, None] * torch.gather(evecs, -1, gather_idx).squeeze(-1)


def sherman_measure_only_top_layer_batched(self, G, P, particle=True, chi=None):
    eye = self._eye_top if P.ndim == 2 else self._eye_top.unsqueeze(0)
    q = eye - P
    chi_vec = _chi_from_projector_torch(P) if chi is None else chi.to(dtype=self.dtype, device=self.device)
    if chi_vec.ndim == 1:
        chi_vec = chi_vec.unsqueeze(0).expand(G.shape[0], -1)
    sign_value = 1.0 if bool(particle) else -1.0
    sign = torch.full((G.shape[0],), sign_value, dtype=self.real_dtype, device=self.device).to(self.dtype)
    v = torch.matmul(G, chi_vec.unsqueeze(-1)).squeeze(-1)
    denom = 1.0 + sign * torch.sum(chi_vec.conj() * v, dim=-1)
    qv = torch.matmul(q, v.unsqueeze(-1)).squeeze(-1)
    chi_gq = torch.matmul(chi_vec.conj().unsqueeze(1), torch.matmul(G, q)).squeeze(1)
    g_upd = sign_value * P + torch.matmul(q, torch.matmul(G, q)) - (
        sign / denom
    )[:, None, None] * qv.unsqueeze(-1) * chi_gq.unsqueeze(-2)
    return 0.5 * (g_upd + g_upd.conj().transpose(-2, -1))


def sherman_measure_only_top_layer_local_batched(self, G, support_idx, comp_idx, chi_local, particle=True):
    g_ss, g_sr = self._local_support_block(G, support_idx, comp_idx)
    chi = chi_local.to(dtype=self.dtype, device=self.device)
    p = self._projector_from_vector(chi)
    q = self._eye_of_size(support_idx.numel()) - p
    sign_value = 1.0 if bool(particle) else -1.0
    sign = torch.full((G.shape[0],), sign_value, dtype=self.real_dtype, device=self.device).to(self.dtype)
    v = torch.matmul(g_ss, chi.unsqueeze(-1)).squeeze(-1)
    denom = 1.0 + sign * torch.sum(chi.conj().unsqueeze(0) * v, dim=-1)
    if comp_idx.numel() == 0:
        y = g_sr
        py = y
        g_sr_new = y
    else:
        chi_gsr = torch.matmul(chi.conj().view(1, 1, -1), g_sr).squeeze(1)
        y = g_sr - (sign / denom)[:, None, None] * v.unsqueeze(-1) * chi_gsr.unsqueeze(-2)
        py = torch.matmul(p.unsqueeze(0), y)
        g_sr_new = torch.matmul(q.unsqueeze(0), y)
    g_ss_q = torch.matmul(g_ss, q.unsqueeze(0))
    chi_gss_q = torch.matmul(chi.conj().view(1, 1, -1), g_ss_q).squeeze(1)
    z = g_ss_q - (sign / denom)[:, None, None] * v.unsqueeze(-1) * chi_gss_q.unsqueeze(-2)
    g_ss_new = sign_value * p.unsqueeze(0) + torch.matmul(q.unsqueeze(0), z)
    g_ss_new = 0.5 * (g_ss_new + g_ss_new.conj().transpose(-2, -1))
    G[:, support_idx[:, None], support_idx[None, :]] = g_ss_new
    if comp_idx.numel() > 0:
        delta_rr = -sign_value * torch.matmul(g_sr.conj().transpose(-2, -1), py)
        delta_rr = 0.5 * (delta_rr + delta_rr.conj().transpose(-2, -1))
        G[:, comp_idx[:, None], comp_idx[None, :]] += delta_rr
        G[:, support_idx[:, None], comp_idx[None, :]] = g_sr_new
        G[:, comp_idx[:, None], support_idx[None, :]] = g_sr_new.conj().transpose(-2, -1)
    return G


def make_model(device: str) -> classA_U1FGTN_gpu:
    return classA_U1FGTN_gpu(
        4,
        8,
        DW=True,
        nshell=1,
        alpha_1=1.0,
        alpha_2=30.0,
        dw_truncation=True,
        device=device,
        dtype="complex128",
        backend="local",
    )


def run_case(protocol: str, solver: str, device: str) -> tuple[dict[str, object], PhysicalStateHealth]:
    np.random.seed(12345)
    torch.manual_seed(12345)
    model = make_model(device)
    g_init = deterministic_g_init(model.Nlayer)
    if solver == "dense":
        model._measure_only_top_layer_batched = MethodType(dense_measure_only_top_layer_batched, model)
        model._measure_only_top_layer_local_batched = MethodType(dense_measure_only_top_layer_local_batched, model)
    elif solver == "sherman":
        model._measure_only_top_layer_batched = MethodType(sherman_measure_only_top_layer_batched, model)
        model._measure_only_top_layer_local_batched = MethodType(sherman_measure_only_top_layer_local_batched, model)
    elif solver != "rank1":
        raise ValueError(f"unknown solver mode: {solver}")

    kwargs = {
        "G_history": True,
        "progress": False,
        "cycles": 50,
        "samples": 1,
        "save": False,
        "return_data": True,
        "save_init": True,
        "G_init": g_init,
        "n_a": 0.5,
        "sequence": "raster_y",
        "meas_slab_only": True,
        "track_choi": False,
        "batch_size": 1,
    }
    if protocol == "postselect":
        kwargs["postselect"] = True
    elif protocol == "perfect_correction":
        kwargs["perfect_correction"] = True
    else:
        raise ValueError(f"unknown protocol: {protocol}")

    started = time.perf_counter()
    result = model.run_markov_circuit(**kwargs)
    elapsed_s = float(time.perf_counter() - started)
    health = PhysicalStateHealth(np.asarray(result["G_hist"])[0])
    return health.summary(protocol, solver, elapsed_s, device), health


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
    widths = [max(len(col), *(len(row[idx]) for row in rendered)) for idx, col in enumerate(columns)]
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
    parser.add_argument("--protocol", choices=("all", "postselect", "perfect_correction"), default="all")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    protocols = ("postselect", "perfect_correction") if args.protocol == "all" else (args.protocol,)
    summaries = []
    observers = {}
    for protocol in protocols:
        for solver in ("rank1", "dense", "sherman"):
            print(f"[run] protocol={protocol}, solver={solver}, device={args.device}")
            summary, observer = run_case(protocol, solver, args.device)
            summaries.append(summary)
            observers[(protocol, solver)] = observer

    print("\nGPU physical covariance survival summary")
    print_table(
        summaries,
        [
            "protocol",
            "solver",
            "device",
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
        ["protocol", "solver_vs_rank1", "G_final_fro_diff", "G_final_max_diff", "same_survival_outcome"],
    )


if __name__ == "__main__":
    main()
