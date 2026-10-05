#!/usr/bin/env python3
"""Run and exactly replay one event-resolved hard-wall max-mix trajectory."""

from __future__ import annotations

import argparse
import copy
import gzip
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from threadpoolctl import threadpool_limits

from common import NX, spectral_snapshot


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from fgtn.classA_U1FGTN import classA_U1FGTN  # noqa: E402


NY = 20
DEFAULT_CYCLES = 4
DEFAULT_SEED = 2026091704
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parent
    / "analysis_outputs/kraus_boundary_contours_v1/event_replay"
)
SCHEMA = "kraus_boundary_event_replay_v1"


def make_model() -> classA_U1FGTN:
    model = classA_U1FGTN(
        Nx=NX,
        Ny=NY,
        DW=True,
        nshell=1,
        filling_frac=0.5,
        alpha_1=1.0,
        alpha_2=30.0,
        trial_orbitals="X",
        dw_truncation=True,
    )
    model.construct_OW_projectors(
        nshell=1,
        DW=True,
        trial_orbitals="X",
        dw_truncation=True,
    )
    return model


def binary_entropy(probability: float) -> float:
    p = float(np.clip(probability, 0.0, 1.0))
    terms = []
    if p > 0.0:
        terms.append(-p * np.log(p))
    if p < 1.0:
        terms.append(-(1.0 - p) * np.log1p(-p))
    return float(sum(terms))


class EventCollector:
    def __init__(self, model: classA_U1FGTN, cycles: int):
        self.model = model
        self.cycles = int(cycles)
        shape = (self.cycles + 1, NX)
        self.realized_center = np.zeros(shape, dtype=np.float64)
        self.realized_support = np.zeros(shape, dtype=np.float64)
        self.predictable_center = np.zeros(shape, dtype=np.float64)
        self.predictable_support = np.zeros(shape, dtype=np.float64)
        self.event_count_center = np.zeros(shape, dtype=np.int64)
        self.covariances: dict[int, np.ndarray] = {}
        self.records: list[dict[str, Any]] = []
        self._support_cache: dict[tuple[int, str], np.ndarray] = {}

    def cycle_observer(self, *, cycle: int, G: np.ndarray, **_: Any) -> None:
        array = np.asarray(G, dtype=np.complex128)
        if array.ndim == 3:
            if array.shape[0] != 1:
                raise ValueError("event replay expects a single trajectory")
            array = array[0]
        self.covariances[int(cycle)] = array.copy()

    def support_profile(self, site_id: int, channel: str) -> np.ndarray:
        key = (int(site_id), str(channel))
        cached = self._support_cache.get(key)
        if cached is not None:
            return cached
        x = int(site_id) % NX
        y = int(site_id) // NX
        payload = self.model._get_ow_local_support_data(x, y)
        indices = np.asarray(payload["idx"], dtype=np.int64)
        orbital = np.asarray(payload[str(channel)], dtype=np.complex128)
        weights = np.abs(orbital) ** 2
        if weights.ndim != 1 or indices.shape != weights.shape:
            raise RuntimeError("unexpected OW support payload")
        profile = np.bincount(
            (indices // 2) % NX, weights=weights, minlength=NX
        ).astype(np.float64)
        closure = float(profile.sum())
        if not np.isclose(closure, 1.0, atol=2.0e-10, rtol=0.0):
            raise FloatingPointError(f"OW support profile closes to {closure}")
        profile /= closure
        self._support_cache[key] = profile
        return profile

    def trajectory_weight_observer(self, **payload: Any) -> None:
        cycle = int(payload["cycle"])
        site_id = int(payload["site_id"])
        x = site_id % NX
        branch_events = tuple(copy.deepcopy(payload["branch_events"]))
        event_log_sum = float(sum(float(event["log_weight"]) for event in branch_events))
        branch_log = float(payload["branch_log_weight"])
        if not np.isclose(event_log_sum, branch_log, atol=2.0e-12, rtol=0.0):
            raise FloatingPointError("branch-event log weights do not close")
        self.realized_center[cycle, x] += branch_log
        predictable_site = 0.0
        for event in branch_events:
            if str(event["kind"]) != "measurement":
                continue
            channel = str(event["channel"])
            log_weight = float(event["log_weight"])
            probability = float(event["probability"])
            entropy = binary_entropy(probability)
            support = self.support_profile(site_id, channel)
            self.realized_support[cycle] += support * log_weight
            self.predictable_support[cycle] += support * entropy
            predictable_site += entropy
            self.event_count_center[cycle, x] += 1
        self.predictable_center[cycle, x] += predictable_site
        self.records.append(
            {
                "cycle": cycle,
                "site_id": site_id,
                "branch_events": list(branch_events),
            }
        )


def execute(
    *, cycles: int, seed: int, trajectory_replay: list[dict[str, Any]] | None
) -> tuple[EventCollector, float]:
    model = make_model()
    collector = EventCollector(model, cycles)
    started = time.perf_counter()
    with threadpool_limits(limits=8):
        model.run_markov_circuit(
            G_history=False,
            progress=True,
            cycles=int(cycles),
            postselect=False,
            perfect_correction=True,
            samples=1,
            parallelize_samples=False,
            init_mode="maxmix",
            save=False,
            n_a=0.5,
            sequence="raster_y",
            meas_slab_only=True,
            random_seed=int(seed),
            state_representation="covariance",
            cycle_observer=collector.cycle_observer,
            trajectory_weight_observer=collector.trajectory_weight_observer,
            trajectory_replay=trajectory_replay,
        )
    return collector, time.perf_counter() - started


def cumulative(values: np.ndarray) -> np.ndarray:
    result = np.zeros_like(values)
    result[1:] = np.cumsum(values[1:], axis=0)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycles", type=int, default=DEFAULT_CYCLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.cycles <= 0:
        raise ValueError("cycles must be positive")
    args.output.mkdir(parents=True, exist_ok=True)

    first, first_seconds = execute(
        cycles=args.cycles, seed=args.seed, trajectory_replay=None
    )
    replay, replay_seconds = execute(
        cycles=args.cycles,
        seed=args.seed,
        trajectory_replay=first.records,
    )
    expected_cycles = set(range(args.cycles + 1))
    if set(first.covariances) != expected_cycles or set(replay.covariances) != expected_cycles:
        raise RuntimeError("cycle observer did not capture the complete covariance history")

    covariance_error = max(
        float(np.max(np.abs(first.covariances[cycle] - replay.covariances[cycle])))
        for cycle in expected_cycles
    )
    contour_errors = {
        name: float(np.max(np.abs(getattr(first, name) - getattr(replay, name))))
        for name in (
            "realized_center",
            "realized_support",
            "predictable_center",
            "predictable_support",
        )
    }
    if covariance_error > 2.0e-10 or max(contour_errors.values()) > 2.0e-12:
        raise FloatingPointError("forced replay failed exact contour/covariance agreement")

    spectrum_x = np.empty((args.cycles + 1, NX), dtype=np.float64)
    spectral_total = np.empty(args.cycles + 1, dtype=np.float64)
    soft_costs = np.empty((args.cycles + 1, 16), dtype=np.float64)
    soft_profiles = np.empty((args.cycles + 1, 16, NX), dtype=np.float64)
    for cycle in range(args.cycles + 1):
        snapshot = spectral_snapshot(first.covariances[cycle], nx=NX, ny=NY)
        spectrum_x[cycle] = snapshot["spectral_x"]
        spectral_total[cycle] = snapshot["spectral_total"]
        soft_costs[cycle] = snapshot["soft_costs"]
        soft_profiles[cycle] = snapshot["soft_profiles"]

    omega_center = cumulative(first.realized_center)
    omega_support = cumulative(first.realized_support)
    ell0_center_no_constant = omega_center + spectrum_x
    ell0_support_no_constant = omega_support + spectrum_x
    global_omega = omega_center.sum(axis=1)
    closure_center = float(
        np.max(np.abs(ell0_center_no_constant.sum(axis=1) - (global_omega + spectral_total)))
    )
    closure_support = float(
        np.max(np.abs(ell0_support_no_constant.sum(axis=1) - (global_omega + spectral_total)))
    )
    if max(closure_center, closure_support) > 2.0e-10:
        raise FloatingPointError("leading-level contour failed its exact sum rule")

    with gzip.open(args.output / "trajectory_replay.json.gz", "wt", encoding="utf-8") as handle:
        json.dump(first.records, handle, separators=(",", ":"))
    np.savez_compressed(
        args.output / "event_replay_contours.npz",
        schema=np.asarray(SCHEMA),
        cycles=np.arange(args.cycles + 1, dtype=np.int64),
        realized_center=first.realized_center,
        realized_support=first.realized_support,
        predictable_center=first.predictable_center,
        predictable_support=first.predictable_support,
        event_count_center=first.event_count_center,
        cumulative_omega_center=omega_center,
        cumulative_omega_support=omega_support,
        spectral_x=spectrum_x,
        spectral_total=spectral_total,
        ell0_center_no_constant=ell0_center_no_constant,
        ell0_support_no_constant=ell0_support_no_constant,
        soft_costs=soft_costs,
        soft_profiles=soft_profiles,
    )
    summary = {
        "schema": SCHEMA,
        "scientific_contract": {
            "Nx": NX,
            "Ny": NY,
            "cycles": int(args.cycles),
            "samples": 1,
            "seed": int(args.seed),
            "wall": "hard/support-truncated",
            "nshell": 1,
            "alpha_1": 1.0,
            "alpha_2": 30.0,
            "initialization": "maxmix",
            "sequence": "raster_y",
            "perfect_correction": True,
            "canonical_dynamics_entry_point": "classA_U1FGTN.run_markov_circuit",
        },
        "record_sites": len(first.records),
        "measurement_events": int(first.event_count_center.sum()),
        "first_run_seconds": first_seconds,
        "forced_replay_seconds": replay_seconds,
        "maximum_covariance_replay_error": covariance_error,
        "maximum_contour_replay_errors": contour_errors,
        "center_contour_sum_rule_error": closure_center,
        "support_contour_sum_rule_error": closure_support,
        "final_total_log_probability": float(global_omega[-1]),
        "final_spectral_correction": float(spectral_total[-1]),
        "final_ell0_minus_log_initial_dimension": float(
            global_omega[-1] + spectral_total[-1]
        ),
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"[done] {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
