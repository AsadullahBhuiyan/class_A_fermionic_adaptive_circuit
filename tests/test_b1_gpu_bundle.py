from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SHARED = ROOT / "00_WORKSPACE/CURRENT/final_production_ready_figure_scripts/_shared_src"
if str(SHARED) not in sys.path:
    sys.path.insert(0, str(SHARED))

from b1_analysis import analyze_case
from b1_controller_frame import (
    ControllerFrame,
    ControllerFrameObserver,
    b1_cases,
    construct_controller_frame,
    validate_b1_config,
)
CONFIG = (
    ROOT
    / "00_WORKSPACE/CURRENT/final_production_ready_figure_scripts/prior_designs/06_b1_controller_frame/production_config.json"
)


def _config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def _torch():
    return pytest.importorskip("torch")


def _gpu_model(nx: int = 4, ny: int = 6):
    _torch()
    from src.fgtn.classA_U1FGTN_gpu import classA_U1FGTN_gpu

    return classA_U1FGTN_gpu(
        Nx=nx,
        Ny=ny,
        DW=True,
        nshell=1,
        alpha_1=1.0,
        alpha_2=30.0,
        filling_frac=0.5,
        trial_orbitals="X",
        dw_truncation=False,
        device="cpu",
        dtype="complex128",
        backend="local",
    )


def test_b1_locked_matrix_has_twenty_total_trajectories() -> None:
    config = _config()
    validate_b1_config(config)
    cases = b1_cases(config)
    assert [case["protocol"] for case in cases] == [
        "explicit_interface",
        "explicit_interface_matched_trivial",
    ]
    assert all(case["run"]["samples"] == 10 for case in cases)
    assert all(case["run"]["cycles"] == 80 for case in cases)
    assert sum(case["run"]["samples"] for case in cases) == 20
    assert config["B1"]["charge_sector_weights"] is False
    assert config["B1"]["training_test_split"] is False
    assert config["B1"]["numerical_tolerance"] == 1e-10
    assert config["B1"]["charge_integer_tolerance"] == 1e-8
    assert config["output_bundle"] == "06_b1_controller_frame_frame_native_v2"


def test_gpu_controller_frame_matches_cpu_constraint_completion() -> None:
    from src.fgtn.classA_U1FGTN import classA_U1FGTN
    from src.fgtn.diagnostics.static import compute_static_completion

    gpu = _gpu_model()
    frame = construct_controller_frame(gpu, degeneracy_tolerance=1e-10)
    cpu = classA_U1FGTN(
        4,
        6,
        DW=True,
        nshell=1,
        alpha_1=1.0,
        alpha_2=30.0,
        trial_orbitals="X",
        dw_truncation=False,
    )
    cpu.construct_OW_projectors(
        nshell=1, DW=True, trial_orbitals="X", dw_truncation=False
    )
    result = compute_static_completion(cpu, meas_slab_only=False, rtol=1e-10)
    np.testing.assert_allclose(
        frame.static_payload["operator_eigenvalues"], result.operator_eigenvalues, atol=2e-11
    )
    # The two implementations enumerate constraints in different raster
    # orders.  Compare the labeled (x, y, channel) maps, not flat array order.
    np.testing.assert_allclose(
        frame.static_payload["residual_map_half_filling"], result.residual_map, atol=2e-11
    )
    assert abs(float(frame.static_payload["f_star_half_filling"]) - result.f_star) < 2e-10
    assert cpu.Nlayer == gpu.Nlayer == int(frame.vectors.shape[0])


def test_degenerate_ground_manifold_has_zero_distance_and_excess() -> None:
    torch = _torch()
    dtype = torch.complex128
    values = torch.tensor([-1.0, 0.0, 0.0, 1.0], dtype=torch.float64)
    vectors = torch.eye(4, dtype=dtype)
    operator = torch.diag(values.to(dtype))
    below = torch.diag(torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=dtype))
    cluster = vectors[:, 1:3]
    prefix = torch.cat((torch.zeros(1, dtype=torch.float64), torch.cumsum(values, dim=0)))
    frame = ControllerFrame(
        vectors=vectors,
        targets=torch.zeros(4, dtype=torch.int64),
        operator=operator,
        eigenvalues=values,
        eigenvectors=vectors,
        active_indices=torch.arange(4),
        center_x=np.arange(4),
        center_y=np.zeros(4, dtype=np.int64),
        channel_indices=np.zeros(4, dtype=np.int64),
        target_rank=2,
        cluster=(1, 3),
        target_sum=0.0,
        eigenvalue_prefix=prefix,
        below_projector=below,
        cluster_vectors=cluster,
        cluster_selected=1,
        static_payload={},
    )
    direction = torch.tensor([0.0, 1.0, 1.0j, 0.0], dtype=dtype)
    direction = direction / torch.linalg.vector_norm(direction)
    occupation = vectors[:, :1] @ vectors[:, :1].conj().T + direction[:, None] @ direction.conj()[None]
    covariance = 2.0 * occupation - torch.eye(4, dtype=dtype)
    observer = ControllerFrameObserver(frame=frame, samples=1, cycles=0)
    observer(cycle=0, G=covariance[None], batch_start=0, batch_count=1)
    diagnostics = observer.validate(tolerance=1e-11)
    assert diagnostics["minimum_ky_fan_excess"] >= -1e-12
    np.testing.assert_allclose(observer.ky_fan_excess, 0.0, atol=1e-12)
    np.testing.assert_allclose(observer.manifold_distance, 0.0, atol=1e-12)


def test_charge_roundoff_has_a_separate_tolerance_and_is_saved_before_rejection(
    tmp_path: Path,
) -> None:
    observer = ControllerFrameObserver(
        frame=construct_controller_frame(_gpu_model(2, 4)), samples=1, cycles=0
    )
    observer.total_charge[:] = 8.0 + 4.0870418160920963e-10
    observer.charge_integer_residual[:] = 4.0870418160920963e-10
    observer.controller_cost[:] = 0.0
    observer.ky_fan_excess[:] = 0.0
    observer.manifold_distance[:] = 0.0
    observer.purity_defect[:] = 0.0

    diagnostics = observer.validate(
        tolerance=1e-10, charge_integer_tolerance=1e-8
    )
    assert diagnostics["maximum_charge_integer_residual"] == pytest.approx(
        4.0870418160920963e-10
    )
    assert diagnostics["numerical_tolerance"] == 1e-10
    assert diagnostics["charge_integer_tolerance"] == 1e-8

    rejected_path = tmp_path / "controller_observables.npz"
    with pytest.raises(RuntimeError, match="trajectory charge is not integer"):
        observer.save(rejected_path, tolerance=1e-10)
    assert rejected_path.is_file()
    with np.load(rejected_path, allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["cycle"], np.asarray([0]))
        np.testing.assert_allclose(
            saved["charge_integer_residual"], 4.0870418160920963e-10
        )


def test_b1_observer_is_diagnostic_only() -> None:
    torch = _torch()
    torch.manual_seed(3801)
    baseline = _gpu_model(2, 4).run_markov_circuit(
        G_history=False,
        progress=False,
        cycles=2,
        samples=2,
        save=False,
        return_data=True,
        sequence="random",
        perfect_correction=True,
        meas_slab_only=False,
        batch_size=2,
    )
    model = _gpu_model(2, 4)
    frame = construct_controller_frame(model)
    observer = ControllerFrameObserver(frame=frame, samples=2, cycles=2)
    torch.manual_seed(3801)
    observed = model.run_markov_circuit(
        G_history=False,
        progress=False,
        cycles=2,
        samples=2,
        save=False,
        return_data=True,
        sequence="random",
        perfect_correction=True,
        meas_slab_only=False,
        batch_size=2,
        native_cycle_observer=observer,
        require_no_covariance_materialization=False,
    )
    np.testing.assert_allclose(observed["G_final"], baseline["G_final"], atol=0.0, rtol=0.0)
    assert observed["covariance_materialization_count"] == 1
    assert observed["covariance_materializations"][0]["reason"] == "legacy_final_return_or_save"
    observer.validate(tolerance=2e-10)


def test_b1_frozen_record_cpu_device_cuda_device_parity() -> None:
    torch = _torch()
    if not torch.cuda.is_available():
        pytest.skip("CUDA device parity requires a GPU")
    from record_observables import OrderedBornRecordWriter
    from src.fgtn.classA_U1FGTN_gpu import classA_U1FGTN_gpu

    def model(device: str):
        return classA_U1FGTN_gpu(
            Nx=2,
            Ny=4,
            DW=True,
            nshell=1,
            alpha_1=1.0,
            alpha_2=30.0,
            device=device,
            dtype="complex128",
            backend="local",
        )

    cpu_model = model("cpu")
    coords = cpu_model._sequence_helper("random", meas_slab_only=False)["coords_for_len"]
    sites = [int(x + cpu_model.Nx * y) for x, y in coords]
    record = OrderedBornRecordWriter(
        samples=2, cycles=2, sites_per_cycle=len(sites), expected_site_ids=sites
    )
    torch.manual_seed(2241)
    initial = cpu_model._prepare_initial_batch(batch_size=2, init_mode="default")
    cpu_model.run_markov_circuit(
        G_history=False,
        progress=False,
        cycles=2,
        samples=2,
        save=False,
        return_data=False,
        G_init=initial,
        sequence="random",
        perfect_correction=True,
        meas_slab_only=False,
        batch_size=2,
        record_observer=record,
    )
    schedule, outcomes = record.site_ids, record.outcomes

    def replay(target_model, initial_state):
        frame = construct_controller_frame(target_model)
        observer = ControllerFrameObserver(frame=frame, samples=2, cycles=2)
        result = target_model.run_markov_circuit(
            G_history=False,
            progress=False,
            cycles=2,
            samples=2,
            save=False,
            return_data=True,
            G_init=initial_state,
            sequence="random",
            perfect_correction=True,
            meas_slab_only=False,
            batch_size=2,
            frozen_schedule=schedule,
            frozen_outcomes=outcomes,
            cycle_observer=observer,
        )
        return result, observer

    cpu_result, cpu_observer = replay(model("cpu"), initial)
    gpu_result, gpu_observer = replay(model("cuda:0"), initial.to("cuda:0"))
    np.testing.assert_allclose(cpu_result["G_final"], gpu_result["G_final"], atol=2e-10, rtol=2e-10)
    for name in (
        "total_charge",
        "controller_cost",
        "ky_fan_excess",
        "manifold_distance",
        "purity_defect",
    ):
        np.testing.assert_allclose(
            getattr(cpu_observer, name), getattr(gpu_observer, name), atol=2e-10, rtol=2e-10
        )


def test_static_archive_payload_contains_no_dense_operator_or_projector() -> None:
    frame = construct_controller_frame(_gpu_model(2, 4))
    dimension = int(frame.static_payload["dimension"])
    forbidden = []
    for key, value in frame.static_payload.items():
        array = np.asarray(value)
        if array.shape == (dimension, dimension):
            forbidden.append(key)
    assert forbidden == []
    assert int(frame.static_payload["dense_arrays_archived"]) == 0


def test_analysis_classification_requires_both_negative_changes(tmp_path: Path) -> None:
    samples, ny = 100, 4
    time = np.arange(2 * ny + 1, dtype=np.float64)
    decreasing = 2.0 - 0.1 * time[None] + np.zeros((samples, 1))
    flat = np.ones((samples, time.size), dtype=np.float64)
    path = tmp_path / "merged.npz"
    np.savez_compressed(
        path,
        case_id=np.asarray("synthetic"),
        ky_fan_excess=decreasing,
        half_filled_manifold_distance=flat,
    )
    summary, _ = analyze_case(path, ny=ny, draws=200, seed=19)
    assert summary["classification"].startswith("no evidence")
