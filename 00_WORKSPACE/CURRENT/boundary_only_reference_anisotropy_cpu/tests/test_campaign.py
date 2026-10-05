from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parents[2]
sys.path[:0] = [str(PACKAGE_ROOT), str(REPO_ROOT / "src")]

from analyze_campaign import finite_size_fit
from fgtn.classA_U1FGTN import classA_U1FGTN
from fgtn.occupied_frame import OccupiedFrameState
from reference_probe import gaussian_subsystem_entropy, insert_reference_cell
from reference_probe import ReferencePairObserver
from run_campaign import publish_completion, verify_completion
from scientific import (
    anisotropy,
    ordered_crossings,
    qwz_negative_band_frame,
    validate_config,
    wall_measurement_site_ids,
)


def load_config() -> dict:
    return json.loads((PACKAGE_ROOT / "campaign_config.json").read_text(encoding="utf-8"))


def test_locked_configuration_and_wall_schedule() -> None:
    config = load_config()
    validate_config(config)
    ids = wall_measurement_site_ids(16, 4, [4, 12])
    np.testing.assert_array_equal(ids, [4, 12, 20, 28, 36, 44, 52, 60])


def test_uniform_qwz_ground_matches_direct_canonical_projector() -> None:
    frame, diagnostics = qwz_negative_band_frame(4, 4, mass=1.0)
    assert frame.shape == (32, 16)
    assert diagnostics["particle_number"] == 16.0
    assert diagnostics["single_particle_gap"] > 1.9
    assert diagnostics["gram_residual"] < 1e-12
    assert diagnostics["translation_x_residual"] < 1e-12
    assert diagnostics["translation_y_residual"] < 1e-12
    model = classA_U1FGTN(4, 4, DW=False, nshell=1, alpha_1=1.0, alpha_2=30.0)
    canonical_covariance = model.G_CI(alpha=1.0)
    canonical_projector = 0.5 * (canonical_covariance.conj() + np.eye(32))
    np.testing.assert_allclose(frame @ frame.conj().T, canonical_projector, atol=1e-12)


def test_reference_cell_has_two_log_two_entropy() -> None:
    occupied = np.asarray([1, 3], dtype=np.int64)
    frame = np.eye(4, dtype=np.complex128)[:, occupied]
    state = OccupiedFrameState(
        frame, representation="physical_frame", physical_dimension=4
    )
    reference = insert_reference_cell(
        state,
        nx=2,
        x=0,
        y=0,
        system_dimension=4,
        rng=np.random.default_rng(10),
    )
    np.testing.assert_allclose(
        gaussian_subsystem_entropy(state, reference["reference_rows"]),
        2.0 * np.log(2.0),
        atol=1e-12,
    )


def test_unique_ordered_crossing_and_known_anisotropy() -> None:
    crossing = ordered_crossings([2, 3, 4], [0.8, 0.5, 0.2], 0.35)
    assert len(crossing) == 1
    np.testing.assert_allclose(crossing[0]["time_star"], 3.5)
    expected = np.arcsinh(1.0) * 8.0 / (np.pi * 3.5)
    np.testing.assert_allclose(anisotropy(8, crossing[0]["time_star"]), expected)
    assert ordered_crossings([1, 2, 3], [0.1, 0.5, 0.05], 0.3) == []


def test_completion_pair_detects_corruption(tmp_path: Path) -> None:
    result = tmp_path / "result.bin"
    result.write_bytes(b"scientific payload")
    publish_completion(result, task_id="task", config_hash="config")
    assert verify_completion(result, task_id="task", config_hash="config")[0]
    result.write_bytes(b"corrupt")
    valid, reason = verify_completion(result, task_id="task", config_hash="config")
    assert not valid
    assert "mismatch" in reason


def test_finite_size_fit_recovers_synthetic_alpha() -> None:
    nys = np.asarray([16, 20, 24, 28, 32], dtype=float)
    expected = 0.83
    values = expected + 2.5 / nys**2
    bootstrap = np.tile(values[:, None], (1, 100))
    fit = finite_size_fit(nys, values, bootstrap)
    np.testing.assert_allclose(fit["alpha_infinity"], expected, atol=1e-12)
    np.testing.assert_allclose(
        fit["omit_Ny16"]["alpha_infinity"], expected, atol=1e-12
    )
    np.testing.assert_allclose(
        fit["quartic_sensitivity"]["alpha_infinity"], expected, atol=1e-12
    )


def test_canonical_checkpoint_resume_and_live_reference_insertion() -> None:
    model = classA_U1FGTN(
        4,
        4,
        DW=True,
        nshell=1,
        filling_frac=0.5,
        alpha_1=1.0,
        alpha_2=30.0,
        trial_orbitals="X",
        dw_truncation=True,
        dw_interval=(1, 3),
        twist_y=1e-7,
    )
    model.construct_OW_projectors(
        nshell=1,
        DW=True,
        trial_orbitals="X",
        dw_truncation=True,
        twist_y=1e-7,
    )
    frame, _ = qwz_negative_band_frame(4, 4)
    site_ids = wall_measurement_site_ids(4, 4, [1, 3])
    captured: dict[int, dict] = {}

    def checkpoint(*, cycle: int, state: dict, **_: object) -> None:
        if cycle in (1, 2, 3):
            captured[cycle] = state

    baseline = model.run_markov_circuit(
        cycles=4,
        samples=1,
        sequence="raster_y",
        perfect_correction=True,
        G_history=False,
        save=False,
        progress=False,
        random_seed=123,
        state_representation="physical_frame",
        return_native_state=True,
        parallelize_samples=False,
        frame_init=frame,
        checkpoint_observer=checkpoint,
        measurement_site_ids=site_ids,
        meas_slab_only=False,
        require_no_covariance_materialization=True,
    )
    baseline_frame = np.asarray(baseline["native_final"]["frame"])
    baseline_projector = baseline_frame @ baseline_frame.conj().T
    for boundary in (1, 2, 3):
        assert captured[boundary]["completed_cycles"] == boundary
        resumed = model.run_markov_circuit(
            cycles=4,
            samples=1,
            sequence="raster_y",
            perfect_correction=True,
            G_history=False,
            save=False,
            progress=False,
            random_seed=captured[boundary]["random_seed"],
            state_representation="physical_frame",
            return_native_state=True,
            parallelize_samples=False,
            init_mode="default",
            checkpoint_state=copy.deepcopy(captured[boundary]),
            measurement_site_ids=site_ids,
            meas_slab_only=False,
            require_no_covariance_materialization=True,
        )
        resumed_frame = np.asarray(resumed["native_final"]["frame"])
        np.testing.assert_allclose(
            resumed_frame @ resumed_frame.conj().T, baseline_projector, atol=1e-12
        )
    observer = ReferencePairObserver(
        nx=4,
        ny=4,
        tau1=2,
        tau2=3,
        follow_cycles=1,
        first_site=(1, 0),
        second_site=(1, 0),
        rng=np.random.default_rng(5),
    )
    result = model.run_markov_circuit(
        cycles=4,
        samples=1,
        sequence="raster_y",
        perfect_correction=True,
        G_history=False,
        save=False,
        progress=False,
        random_seed=captured[1]["random_seed"],
        state_representation="physical_frame",
        return_native_state=True,
        parallelize_samples=False,
        init_mode="default",
        checkpoint_state=copy.deepcopy(captured[1]),
        native_cycle_observer=observer,
        measurement_site_ids=site_ids,
        meas_slab_only=False,
        require_no_covariance_materialization=True,
    )
    observer.assert_complete()
    assert int(result["native_final"]["physical_dimension"]) == 36
    assert float(result["native_final"]["gram_residual"]) < 1e-10
