from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[1]
PROJECT = (
    REPO
    / "00_WORKSPACE/CURRENT/final_production_new_designs"
    / "06_domain_wall_flattened_ground_state_reference"
)
RUNNER_PATH = PROJECT / "run_flattened_ground_state_reference.py"
LARGE_RUNNER_PATH = PROJECT / "run_flattened_ground_state_large_ny.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
REFERENCE = _load(RUNNER_PATH, "tested_domain_wall_flattened_reference")
LARGE_REFERENCE = _load(
    LARGE_RUNNER_PATH, "tested_domain_wall_flattened_large_ny_reference"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_locked_campaign_and_log_chord_fit() -> None:
    assert REFERENCE.NX == 20
    assert REFERENCE.NY_VALUES == (20, 24, 28)
    assert REFERENCE.ALPHA_VALUES == (
        3.0, 2.75, 2.5, 2.3, 2.2, 2.15, 2.1, 2.075, 2.05, 2.025,
        2.0, 1.975, 1.95, 1.925, 1.9, 1.85, 1.8, 1.7, 1.5, 1.25, 1.0,
    )
    assert REFERENCE.WALLS == ("hard", "soft")
    assert REFERENCE.NSHELL == 1
    assert REFERENCE.ALPHA_2 == 30.0
    ay = np.arange(1, 11, dtype=np.int64)
    x = np.log((20.0 / np.pi) * np.sin(np.pi * ay / 20.0))
    target_c = 1.25
    fit = REFERENCE.fit_central_charge(ay, 2.0 + (target_c / 3.0) * x, 20)
    assert abs(fit["c_fit"] - target_c) < 1.0e-12
    assert fit["c_fit_se"] < 1.0e-12
    assert abs(fit["r2"] - 1.0) < 1.0e-12


def test_sources_are_synchronized_and_completed_products_verify() -> None:
    assert _sha256(PROJECT / "src/classA_U1FGTN_gpu.py") == _sha256(
        REPO / "src/fgtn/classA_U1FGTN_gpu.py"
    )
    assert _sha256(PROJECT / "src/occupied_frame_gpu.py") == _sha256(
        REPO / "src/fgtn/occupied_frame_gpu.py"
    )
    verified, reason = REFERENCE.verified_reference(
        PROJECT / "results", REFERENCE.source_hashes()
    )
    assert verified, reason


def test_large_ny_campaign_contract_and_cpu_sources() -> None:
    assert LARGE_REFERENCE.NX == 20
    assert LARGE_REFERENCE.NY_VALUES == (40, 50, 60)
    assert LARGE_REFERENCE.NSHELL_LABELS == ("1", "2", "inf")
    assert LARGE_REFERENCE.NSHELL_VALUES == (1, 2, None)
    assert LARGE_REFERENCE.ALPHA_VALUES == REFERENCE.ALPHA_VALUES
    expanded = LARGE_REFERENCE.cases()
    assert len(expanded) == 378
    assert len({case.case_id for case in expanded}) == 378
    assert {case.width for case in expanded if case.ny == 40} == {10}
    assert {case.width for case in expanded if case.ny == 50} == {12}
    assert {case.width for case in expanded if case.ny == 60} == {15}
    assert _sha256(PROJECT / "src/classA_U1FGTN.py") == _sha256(
        REPO / "src/fgtn/classA_U1FGTN.py"
    )
    assert _sha256(PROJECT / "src/occupied_frame.py") == _sha256(
        REPO / "src/fgtn/occupied_frame.py"
    )


def test_cft_reference_and_momentum_blocks_match_dense_small_reference() -> None:
    assert abs(
        LARGE_REFERENCE.cft_mutual_information_reference(40, 10)
        - np.log(2.0) / 3.0
    ) < 1.0e-15
    case = LARGE_REFERENCE.Case(
        wall_index=0,
        nshell_index=0,
        ny_index=0,
        alpha_index=0,
        wall="hard",
        nshell_label="1",
        nshell=1,
        ny=20,
        alpha_1=1.5,
    )
    record = LARGE_REFERENCE.compute_case(case, threads_per_worker=2)
    with np.load(
        PROJECT
        / "results/flattened_ground_state_reference/flattened_ground_state_reference.npz"
    ) as completed:
        alpha_index = int(
            np.flatnonzero(np.isclose(completed["alpha_1_values"], 1.5))[0]
        )
        expected_mi = completed["mutual_information_y0avg"][0, 0, alpha_index]
        expected_c = completed["c_fit"][0, 0, alpha_index]
    assert abs(record["mutual_information_y0avg"] - expected_mi) < 5.0e-11
    assert abs(record["c_fit"] - expected_c) < 5.0e-11
    assert record["ow_y_translation_covariance_max_abs"] < 2.0e-10
    assert record["projector_idempotency_max_abs"] < 2.0e-10


def test_large_ny_completed_aggregate_verifies() -> None:
    output_root = PROJECT / "results"
    verified, reason = LARGE_REFERENCE.verified_aggregate(
        output_root, LARGE_REFERENCE.source_hashes()
    )
    assert verified, reason
    aggregate = output_root / LARGE_REFERENCE.RESULT_DIRNAME
    with np.load(aggregate / "flattened_ground_state_large_ny.npz") as completed:
        assert completed["mutual_information_y0avg"].shape == (2, 3, 3, 21)
        assert completed["c_fit"].shape == (2, 3, 3, 21)
        np.testing.assert_array_equal(completed["width_values"], [10, 12, 15])
        np.testing.assert_allclose(
            completed["mutual_information_cft_c1_by_Ny"][[0, 2]],
            np.log(2.0) / 3.0,
            atol=1.0e-15,
        )
        maximum_index = np.unravel_index(
            np.argmax(completed["mutual_information_y0avg"]),
            completed["mutual_information_y0avg"].shape,
        )
        minimum_gap_index = np.unravel_index(
            np.argmin(completed["half_filling_gap"]),
            completed["half_filling_gap"].shape,
        )
        assert maximum_index == (0, 2, 0, 11)
        assert completed["alpha_1_values"][maximum_index[-1]] == 1.975
        assert completed["half_filling_gap"][maximum_index] > 0.17
        assert minimum_gap_index == (0, 2, 0, 20)
        assert completed["alpha_1_values"][minimum_gap_index[-1]] == 1.0
        rank_mismatch = (
            completed["negative_energy_count"] != completed["half_filling_rank"]
        )
        np.testing.assert_array_equal(
            np.argwhere(rank_mismatch), np.asarray([minimum_gap_index])
        )
    for filename in (
        "flattened_ground_state_large_ny.pdf",
        "flattened_ground_state_large_ny.png",
        "mutual_information_geometry.pdf",
        "mutual_information_geometry.png",
    ):
        assert (aggregate / filename).stat().st_size > 0


def test_critical_grid_uses_symmetric_midpoint_prescription() -> None:
    case = LARGE_REFERENCE.Case(
        wall_index=1,
        nshell_index=2,
        ny_index=0,
        alpha_index=10,
        wall="soft",
        nshell_label="inf",
        nshell=None,
        ny=20,
        alpha_1=2.0,
    )
    model = LARGE_REFERENCE._build_model(case)
    midpoint = np.asarray(model.Pplus[0, 0, 10, 0])
    np.testing.assert_allclose(midpoint, 0.5 * np.eye(2), atol=1.0e-15)
    assert np.max(np.abs(midpoint @ midpoint - midpoint)) == 0.25


def test_real_cpu_case_is_half_filled_and_translation_invariant() -> None:
    record = REFERENCE.compute_case(
        ny=20, alpha_1=1.5, wall="hard", device=torch.device("cpu")
    )
    assert record["half_filling_rank"] == 400
    assert record["Ay_values"].tolist() == list(range(1, 11))
    assert record["entropy_profile_y0avg"].shape == (10,)
    assert np.isfinite(record["mutual_information_y0avg"])
    assert record["mutual_information_y0avg"] >= -1.0e-9
    assert np.isfinite(record["c_fit"])
    assert np.isfinite(record["c_fit_se"])
    assert record["projector_y_translation_max_abs"] < 2.0e-10
    assert record["entropy_profile_y0_shift_max_abs"] < 2.0e-10
    assert record["mutual_information_y0_spread"] < 2.0e-10
    assert record["hamiltonian_hermiticity_max_abs"] < 1.0e-12
    assert record["projector_idempotency_max_abs"] < 2.0e-10
    assert record["projector_rank_residual"] < 2.0e-10
