#!/usr/bin/env python3
"""Small-system Choi covariance survival check.

This compares the production rank-1 Choi resolvent action against a dense
linear solve of the same resolvent on an identical small CPU setup.
"""

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
class CycleRecord:
    cycle: int
    active: bool
    max_abs_choi_entry: float
    hermiticity_residual: float
    involution_residual: float
    min_abs_d: float
    denominator_regularized_count: int
    nonfinite: bool


class ChoiHealthObserver:
    def __init__(self, cycles: list[int], blowup_threshold: float = BLOWUP_THRESHOLD):
        self.cycles = tuple(int(cycle) for cycle in cycles)
        self.blowup_threshold = float(blowup_threshold)
        self.records: list[CycleRecord] = []
        self.final_blocks: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

    def __call__(
        self,
        *,
        cycle,
        sigma_ll,
        sigma_lr,
        sigma_rr,
        min_abs_d,
        choi_active_mask=None,
        choi_failure_records=(),
        **_,
    ):
        ll = np.asarray(sigma_ll)
        lr = np.asarray(sigma_lr)
        rr = np.asarray(sigma_rr)
        active_mask = (
            np.ones((ll.shape[0],), dtype=bool)
            if choi_active_mask is None
            else np.asarray(choi_active_mask, dtype=bool).reshape(-1)
        )
        local_idx = 0
        blocks = (ll[local_idx], lr[local_idx], rr[local_idx])
        nonfinite = not all(np.all(np.isfinite(block)) for block in blocks)
        max_entry = (
            float(max(np.max(np.abs(block)) for block in blocks))
            if not nonfinite
            else float("inf")
        )

        sigma = np.block(
            [
                [blocks[0], blocks[1]],
                [blocks[1].conj().T, blocks[2]],
            ]
        )
        if nonfinite:
            herm = float("inf")
            invol = float("inf")
        else:
            ident = np.eye(sigma.shape[0], dtype=np.complex128)
            herm = float(np.linalg.norm(sigma - sigma.conj().T, ord="fro"))
            invol = float(np.linalg.norm(sigma @ sigma - ident, ord="fro"))

        regularized_count = sum(
            1 for record in choi_failure_records
            if dict(record).get("stage") == "denominator_regularized"
        )
        self.records.append(
            CycleRecord(
                cycle=int(cycle),
                active=bool(active_mask[local_idx]),
                max_abs_choi_entry=max_entry,
                hermiticity_residual=herm,
                involution_residual=invol,
                min_abs_d=float(min_abs_d),
                denominator_regularized_count=int(regularized_count),
                nonfinite=bool(nonfinite),
            )
        )
        self.final_blocks = tuple(block.copy() for block in blocks)
        bad = nonfinite or max_entry > self.blowup_threshold
        if bad and bool(active_mask[local_idx]):
            return {
                "deactivate_sample_offsets": [0],
                "failure_records": [
                    {
                        "cycle": int(cycle),
                        "sample_offset": 0,
                        "reason": "nonfinite_choi_block" if nonfinite else "choi_entry_blowup",
                        "max_abs_choi_entry": max_entry,
                    }
                ],
            }
        return None

    @property
    def first_bad_cycle(self) -> int | None:
        for record in self.records:
            if record.nonfinite or record.max_abs_choi_entry > self.blowup_threshold:
                return record.cycle
        return None

    @property
    def completed_50_cycles(self) -> bool:
        return bool(self.records) and self.records[-1].cycle == 50 and self.first_bad_cycle is None

    def final_summary(self, protocol: str, solver: str, elapsed_s: float) -> dict[str, object]:
        final = self.records[-1]
        return {
            "protocol": protocol,
            "solver": solver,
            "completed_50_cycles": self.completed_50_cycles,
            "first_bad_cycle": self.first_bad_cycle,
            "final_active": final.active,
            "final_max_abs_entry": final.max_abs_choi_entry,
            "max_entry_over_run": max(record.max_abs_choi_entry for record in self.records),
            "final_hermiticity": final.hermiticity_residual,
            "final_involution": final.involution_residual,
            "min_abs_d": min(record.min_abs_d for record in self.records),
            "denominator_regularized_count": final.denominator_regularized_count,
            "elapsed_s": elapsed_s,
        }


def dense_choi_resolvent_action(self, rhs, v, chi, d, eta1, regularized, eps=1e-9):
    del d
    count, dim, _ = rhs.shape
    out = np.empty_like(rhs)
    eye = np.eye(dim, dtype=np.complex128)
    eta1 = np.asarray(eta1, dtype=np.float64).reshape(-1)
    regularized = np.asarray(regularized, dtype=bool).reshape(-1)
    for idx in range(count):
        projector_right = v[idx, :, None] * chi[idx].conj()[None, :]
        solve_mat = eye - eta1[idx] * projector_right
        if regularized[idx]:
            row_abs_sum = np.sum(np.abs(solve_mat), axis=-1)
            eps_scale = max(float(np.max(row_abs_sum)), 1.0)
            solve_mat = solve_mat + float(eps) * eps_scale * eye
        try:
            out[idx] = np.linalg.solve(solve_mat, rhs[idx])
        except np.linalg.LinAlgError:
            out[idx] = np.linalg.pinv(solve_mat) @ rhs[idx]
    return out


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


def run_case(protocol: str, solver: str) -> tuple[dict[str, object], ChoiHealthObserver]:
    np.random.seed(12345)
    model = make_model()
    if solver == "dense":
        model._choi_resolvent_action = MethodType(dense_choi_resolvent_action, model)
    elif solver != "rank1":
        raise ValueError(f"unknown solver mode: {solver}")

    observer = ChoiHealthObserver(cycles=list(range(1, 51)))
    run_kwargs = {
        "G_history": False,
        "progress": False,
        "cycles": 50,
        "samples": 1,
        "save": False,
        "n_a": 0.5,
        "sequence": "raster_y",
        "meas_slab_only": True,
        "parallelize_samples": False,
        "track_choi": True,
        "choi_observer": observer,
        "choi_observer_cycles": list(range(1, 51)),
        "choi_singular_tol": 1e-10,
        "choi_failure_mode": "censor",
    }
    if protocol == "postselect":
        run_kwargs["postselect"] = True
    elif protocol == "perfect_correction":
        run_kwargs["perfect_correction"] = True
    else:
        raise ValueError(f"unknown protocol: {protocol}")

    started = time.perf_counter()
    model.run_markov_circuit(**run_kwargs)
    elapsed_s = float(time.perf_counter() - started)
    return observer.final_summary(protocol, solver, elapsed_s), observer


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


def comparison_rows(observers: dict[tuple[str, str], ChoiHealthObserver]) -> list[dict[str, object]]:
    rows = []
    for protocol in ("postselect", "perfect_correction"):
        rank1 = observers[(protocol, "rank1")]
        dense = observers[(protocol, "dense")]
        if rank1.final_blocks is None or dense.final_blocks is None:
            diffs = (float("nan"), float("nan"), float("nan"))
        else:
            diffs = tuple(
                float(np.linalg.norm(d_block - r_block, ord="fro"))
                for d_block, r_block in zip(dense.final_blocks, rank1.final_blocks)
            )
        rows.append(
            {
                "protocol": protocol,
                "dense_rank1_LL_diff": diffs[0],
                "dense_rank1_LR_diff": diffs[1],
                "dense_rank1_RR_diff": diffs[2],
                "same_survival_outcome": dense.completed_50_cycles == rank1.completed_50_cycles,
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
    observers: dict[tuple[str, str], ChoiHealthObserver] = {}
    for protocol in protocols:
        for solver in ("rank1", "dense"):
            print(f"[run] protocol={protocol}, solver={solver}")
            summary, observer = run_case(protocol, solver)
            summaries.append(summary)
            observers[(protocol, solver)] = observer

    print("\nSurvival summary")
    print_table(
        summaries,
        [
            "protocol",
            "solver",
            "completed_50_cycles",
            "first_bad_cycle",
            "final_active",
            "final_max_abs_entry",
            "max_entry_over_run",
            "final_hermiticity",
            "final_involution",
            "min_abs_d",
            "denominator_regularized_count",
            "elapsed_s",
        ],
    )

    if args.protocol == "all":
        print("\nDense vs rank-1 final-block comparison")
        print_table(
            comparison_rows(observers),
            [
                "protocol",
                "dense_rank1_LL_diff",
                "dense_rank1_LR_diff",
                "dense_rank1_RR_diff",
                "same_survival_outcome",
            ],
        )


if __name__ == "__main__":
    main()
