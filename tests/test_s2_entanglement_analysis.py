from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SHARED = ROOT / "00_WORKSPACE/CURRENT/final_production_ready_figure_scripts/_shared_src"
sys.path.insert(0, str(SHARED))

from s2_entanglement_analysis import (  # noqa: E402
    ArchiveInput,
    analyze_case_arrays,
    chord_coordinate,
    fit_entropy_curve,
    select_case_archives,
)


def test_exact_log_chord_and_constant_models_are_distinguished() -> None:
    ny = 24
    ay = np.arange(ny // 2 + 1)
    log_curve = np.zeros_like(ay, dtype=float)
    log_curve[1:] = 1.7 + chord_coordinate(ny, ay[1:]) / 3.0
    log_fit = fit_entropy_curve(log_curve, ny)
    assert log_fit["c_eff"] == pytest.approx(1.0, abs=1e-12)
    assert log_fit["delta_aicc"] > 0

    constant = np.full_like(ay, 2.25, dtype=float)
    area_fit = fit_entropy_curve(constant, ny)
    assert area_fit["c_eff"] == pytest.approx(0.0, abs=1e-12)
    assert area_fit["delta_aicc"] < 0


def test_checkpoint_analysis_uses_whole_trajectories_and_requires_all_checkpoints() -> None:
    ny = 24
    ay = np.arange(ny // 2 + 1)
    curve = np.zeros_like(ay, dtype=float)
    curve[1:] = 0.9 + chord_coordinate(ny, ay[1:]) / 3.0
    checkpoints = {
        ny: np.stack([curve] * 10),
        3 * ny // 2: np.stack([curve] * 10),
        2 * ny: np.stack([curve] * 10),
    }
    per, summary = analyze_case_arrays(
        checkpoints, ny=ny, bootstrap_resamples=50, bootstrap_seed=7
    )
    assert len(per) == 30
    assert len(summary) == 3
    assert all(row["actual_samples"] == 10 for row in summary)
    assert all(row["c_eff"] == pytest.approx(1.0) for row in summary)

    del checkpoints[ny]
    with pytest.raises(ValueError, match="missing entropy checkpoints"):
        analyze_case_arrays(checkpoints, ny=ny)


def _archive(case: dict, shard: int, priority: int = 0) -> ArchiveInput:
    return ArchiveInput(
        path=Path(f"p{priority}-shard-{shard}.tar.gz"),
        sha256=f"{priority}{shard}".ljust(64, "0"),
        priority=priority,
        manifest={
            "status": "complete_local",
            "shard_index": shard,
            "global_sample_indices": list(range(5 * shard, 5 * shard + 5)),
            "run_config": {"case": case},
        },
    )


@pytest.mark.parametrize("shards", [5, 6])
def test_archive_merger_accepts_at_least_twenty_five_samples(shards: int) -> None:
    case = {
        "case_id": "S2_test",
        "model": {"Nx": 20, "Ny": 24},
        "run": {"samples": 25, "cycles": 48},
    }
    legacy = {**case, "run": {**case["run"], "samples": 25}}
    selected = select_case_archives([_archive(legacy, shard) for shard in range(shards)], case)
    assert len(selected) == shards


def test_archive_merger_rejects_incomplete_and_prefers_higher_priority() -> None:
    case = {
        "case_id": "S2_test",
        "model": {"Nx": 20, "Ny": 24},
        "run": {"samples": 25, "cycles": 48},
    }
    with pytest.raises(RuntimeError, match="non-consecutive"):
        select_case_archives([_archive(case, 0), _archive(case, 2)], case)
    selected = select_case_archives(
        [_archive(case, 0, 1), *[_archive(case, shard, 0) for shard in range(5)]],
        case,
    )
    assert selected[0].priority == 0
