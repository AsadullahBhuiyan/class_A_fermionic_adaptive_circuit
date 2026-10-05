from __future__ import annotations

import json
import hashlib
import io
import sys
import tarfile
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SHARED = ROOT / "00_WORKSPACE/CURRENT/final_production_new_designs/_shared_src"
BUNDLE = ROOT / "00_WORKSPACE/CURRENT/final_production_new_designs/02_wall_cft_windows"
for path in (SHARED, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from wall_window_observables import (  # noqa: E402
    WallWindowObserver,
    checkpoint_cycles,
    periodic_window_indices,
    window_natural_data_from_correlation,
    window_natural_data_from_frame,
    x_resolved_square_correlator_from_frame,
)
from wall_window_runner import expand_cases, load_config  # noqa: E402
from wall_window_loader import build_case_index, iter_case_window_height  # noqa: E402


def _random_frame(dimension: int, rank: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(9182)
    raw = torch.complex(
        torch.randn((dimension, rank), dtype=torch.float64, generator=generator),
        torch.randn((dimension, rank), dtype=torch.float64, generator=generator),
    )
    return torch.linalg.qr(raw, mode="reduced").Q


def test_exact_case_matrix_and_locked_hyperparameters():
    cases = expand_cases(load_config(BUNDLE))
    assert len(cases) == 10
    assert {case["model"]["Ny"] for case in cases} == {20, 30, 40, 50, 60}
    assert {case["protocol"] for case in cases} == {"hard", "soft"}
    assert all(case["model"]["Nx"] == 20 for case in cases)
    assert all(case["model"]["alpha_1"] == 1.0 for case in cases)
    assert all(case["model"]["alpha_2"] == 30.0 for case in cases)
    assert all(case["run"]["samples"] == 25 for case in cases)
    assert all(case["run"]["cycles"] == 2 * case["model"]["Ny"] for case in cases)
    for case in cases:
        hard = case["protocol"] == "hard"
        assert case["model"]["dw_truncation"] is hard
        assert case["run"]["meas_slab_only"] is hard


def test_locked_checkpoint_schedule():
    assert checkpoint_cycles(20) == [23, 27, 30, 33, 37, 40]
    assert checkpoint_cycles(30) == [35, 40, 45, 50, 55, 60]
    assert checkpoint_cycles(40) == [47, 53, 60, 67, 73, 80]
    assert checkpoint_cycles(50) == [58, 67, 75, 83, 92, 100]
    assert checkpoint_cycles(60) == [70, 80, 90, 100, 110, 120]


def test_periodic_windows_wrap_and_preserve_relative_order():
    idx = periodic_window_indices(nx=2, ny=5, y0=4, ay=2)
    expected = [
        16, 17, 18, 19,
        0, 1, 2, 3,
    ]
    assert idx.tolist() == expected
    assert len(set(idx.tolist())) == len(expected)


def test_frame_natural_data_and_contours_match_dense_reference():
    nx, ny, ay, y0 = 2, 4, 2, 3
    frame = _random_frame(2 * nx * ny, 7)
    idx = periodic_window_indices(nx=nx, ny=ny, y0=y0, ay=ay)
    actual = window_natural_data_from_frame(frame, rank=7, indices=idx, nx=nx, ay=ay)
    reference = window_natural_data_from_correlation(
        frame @ frame.mH, indices=idx, nx=nx, ay=ay
    )
    for left, right in zip(actual, reference):
        torch.testing.assert_close(left, right, atol=2e-12, rtol=2e-12)
    nu, entropy, variance = actual
    entropy_total = (-(nu.clamp(1e-15, 1 - 1e-15) * nu.clamp(1e-15, 1 - 1e-15).log()
                       + (1 - nu.clamp(1e-15, 1 - 1e-15)) * (1 - nu.clamp(1e-15, 1 - 1e-15)).log())).sum()
    torch.testing.assert_close(entropy.sum(), entropy_total)
    torch.testing.assert_close(variance.sum(), (nu * (1 - nu)).sum())


def test_x_resolved_square_correlator_matches_dense_definition():
    nx, ny, rank = 3, 6, 8
    frame = _random_frame(2 * nx * ny, rank)
    correlation = frame @ frame.mH
    actual = x_resolved_square_correlator_from_frame(frame, rank=rank, nx=nx, ny=ny)
    expected = torch.empty_like(actual)
    for x in range(nx):
        for r in range(1, ny // 2 + 1):
            total = 0.0
            for y in range(ny):
                for mu in range(2):
                    for nu in range(2):
                        i = mu + 2 * x + 2 * nx * y
                        j = nu + 2 * x + 2 * nx * ((y + r) % ny)
                        total += correlation[i, j].abs().square()
            expected[x, r - 1] = total / (2 * ny)
    torch.testing.assert_close(actual, expected, atol=1e-13, rtol=1e-13)


def test_complete_observer_schema_and_retired_products_absent(tmp_path: Path):
    nx, ny, samples = 2, 6, 2
    frame = _random_frame(2 * nx * ny, 11).repeat(samples, 1, 1)
    state = type("FrameState", (), {
        "frame": frame,
        "ranks": torch.full((samples,), 11, dtype=torch.long),
    })()
    observer = WallWindowObserver(
        nx=nx, ny=ny, checkpoints=checkpoint_cycles(ny), global_sample_ids=[5, 6]
    )
    for cycle in checkpoint_cycles(ny):
        observer(cycle=cycle, state=state, batch_start=0, batch_count=samples)
    receipt = observer.save(tmp_path / "wall_windows", config={"test": True})
    assert receipt["samples"] == samples
    with np.load(tmp_path / "wall_windows/common.npz", allow_pickle=False) as common:
        assert common["square_correlator"].shape == (2, 6, 2, 3)
        assert set(common.files).isdisjoint({"covariance", "eigenvectors", "fits", "entropy"})
    with np.load(tmp_path / "wall_windows/Ay_003.npz", allow_pickle=False) as data:
        assert data["occupation_spectrum"].shape == (2, 6, 6, 12)
        assert data["entropy_contour"].shape == (2, 6, 6, 2, 3)
        assert data["charge_variance_contour"].shape == (2, 6, 6, 2, 3)
        assert set(data.files) == {
            "schema", "ay", "occupation_spectrum", "entropy_contour",
            "charge_variance_contour",
        }


def test_exact_five_shard_completion_index_and_lazy_ay_loading(tmp_path: Path):
    archives = []
    for shard in range(5):
        sample_ids = list(range(5 * shard, 5 * shard + 5))
        manifest = {
            "case_id": "WALL_TEST", "shard_index": shard,
            "global_sample_indices": sample_ids, "run_config_hash": f"hash-{shard}",
        }
        common_buffer = io.BytesIO()
        np.savez_compressed(
            common_buffer, global_sample_ids=np.asarray(sample_ids),
            square_correlator=np.zeros((5, 6, 2, 3)),
        )
        ay_buffer = io.BytesIO()
        np.savez_compressed(ay_buffer, occupation_spectrum=np.full((5, 1), shard))
        archive = tmp_path / f"shard-{shard}.tar.gz"
        with tarfile.open(archive, "w:gz") as handle:
            for name, payload in (
                ("manifest.json", json.dumps(manifest).encode()),
                ("shards/shard_000/wall_windows/common.npz", common_buffer.getvalue()),
                ("shards/shard_000/wall_windows/Ay_001.npz", ay_buffer.getvalue()),
                ("src/source_manifest.json", b"{}"),
            ):
                info = tarfile.TarInfo(name)
                info.size = len(payload)
                handle.addfile(info, io.BytesIO(payload))
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        archive.with_suffix(archive.suffix + ".receipt.json").write_text(
            json.dumps({"archive": archive.name, "archive_sha256": digest})
        )
        archives.append(archive)
    index = build_case_index(tmp_path, case_id="WALL_TEST")
    assert index["complete"] is True
    assert index["sample_ids"] == list(range(25))
    loaded = list(iter_case_window_height(archives, ay=1))
    assert len(loaded) == 5
    assert [int(row["occupation_spectrum"][0, 0]) for row in loaded] == list(range(5))
