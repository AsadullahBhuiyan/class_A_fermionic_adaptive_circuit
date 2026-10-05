"""Regression tests for the standalone ambient log-Gram campaign."""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "00_WORKSPACE/CURRENT/final_production_new_designs/07_log_gram_alpha_scan"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(BUNDLE / "src"))

from fgtn.classA_U1FGTN_gpu import classA_U1FGTN_gpu
from fgtn.occupied_frame_gpu import BatchedOccupiedFrameState
from log_gram_observer import EigenbasisTransport, LogGramObserver, checkpoint_cycles
from log_gram_runner import (
    _existing_valid,
    expand_cases,
    load_config,
    main as runner_main,
    production_queue,
)


def _model(*, dw=False, truncation=False):
    pytest.importorskip("torch")
    return classA_U1FGTN_gpu(
        Nx=4,
        Ny=2,
        DW=dw,
        nshell=1,
        dw_truncation=truncation,
        device="cpu",
        dtype="complex128",
        backend="local",
    )


def _project_and_reset(model, covariance, tangent_states, chi, outcome, target):
    torch = pytest.importorskip("torch")
    projector = chi[:, None] * chi.conj()[None, :]
    offsets = torch.asarray([0], dtype=torch.long)
    for state in tangent_states:
        model._lyapunov_apply_dense_channel(
            state, covariance, offsets, projector, particle=bool(outcome)
        )
    covariance = model._measure_only_top_layer_batched(
        covariance.clone(), projector, particle=bool(outcome), chi=chi
    )
    for state in tangent_states:
        model._lyapunov_apply_dense_reset(state, offsets, projector)
    complement = torch.eye(
        model.Nlayer, dtype=model.dtype
    ) - projector
    eta = 1.0 if bool(target) else -1.0
    covariance = (
        complement.unsqueeze(0) @ covariance @ complement.unsqueeze(0)
        + eta * projector.unsqueeze(0)
    )
    return 0.5 * (covariance + covariance.mH)


def test_campaign_case_and_queue_contract():
    config = load_config(BUNDLE)
    assert len(expand_cases(config)) == 220
    assert len(expand_cases(config, pilot=True)) == 12
    queue = production_queue(config)
    assert len(queue) == 1100
    assert queue[0]["sample_start"] == 0
    assert queue[-1]["sample_stop"] == 25
    assert checkpoint_cycles(20) == [20, 24, 28, 32, 36, 40]
    assert checkpoint_cycles(30) == [30, 36, 42, 48, 54, 60]


def test_bundle_declares_no_choi_or_state_outputs():
    config = json.loads((BUNDLE / "production_config.json").read_text())
    assert config["contract"]["choi_tracked"] is False
    runner = (BUNDLE / "src/log_gram_runner.py").read_text()
    assert '"track_choi": False' in runner
    assert '"G_history": False' in runner
    assert "require_no_covariance_materialization=True" in runner


def test_maxmix_identity_matches_dense_half_jacobian_on_fixed_record():
    torch = pytest.importorskip("torch")
    model = _model()
    dimension = model.Nlayer
    covariance = torch.zeros((1, dimension, dimension), dtype=model.dtype)
    dense = model._init_lyapunov_state(
        batch_count=1,
        n_vec=dimension,
        initial_frame=torch.eye(dimension, dtype=model.dtype),
    )
    generator = torch.Generator().manual_seed(74)
    for step, (outcome, target) in enumerate(
        ((False, True), (True, False), (True, True), (False, False))
    ):
        raw = torch.complex(
            torch.randn(dimension, generator=generator),
            torch.randn(dimension, generator=generator),
        ).to(model.dtype)
        chi = raw / torch.linalg.vector_norm(raw)
        covariance = _project_and_reset(
            model, covariance, (dense,), chi, outcome, target
        )
    product = dense["frame"][0]
    explicit = product @ product.mH
    identity = torch.eye(dimension, dtype=model.dtype)
    inferred = identity - covariance[0] @ covariance[0]
    torch.testing.assert_close(explicit, inferred, rtol=2e-10, atol=2e-10)


def test_pure_occupied_empty_blocks_reconstruct_full_ambient_gram():
    torch = pytest.importorskip("torch")
    model = _model()
    dimension = model.Nlayer
    rank = dimension // 2
    generator = torch.Generator().manual_seed(91)
    raw = torch.complex(
        torch.randn((dimension, rank), generator=generator),
        torch.randn((dimension, rank), generator=generator),
    ).to(model.dtype)
    occupied, _ = torch.linalg.qr(raw, mode="reduced")
    physical = BatchedOccupiedFrameState.from_frame(occupied)
    covariance = physical.centered_covariance(reason="unit_test")
    full = model._init_lyapunov_state(
        batch_count=1,
        n_vec=dimension,
        initial_frame=torch.eye(dimension, dtype=model.dtype),
    )
    blocks = model._init_pure_occupied_empty_lyapunov_state(
        physical, basis_idx=None
    )
    generator = torch.Generator().manual_seed(92)
    for cycle, (outcome, target) in enumerate(
        ((False, True), (True, False), (False, False), (True, True)), start=1
    ):
        raw = torch.complex(
            torch.randn(dimension, generator=generator),
            torch.randn(dimension, generator=generator),
        ).to(model.dtype)
        chi = raw / torch.linalg.vector_norm(raw)
        covariance = _project_and_reset(
            model, covariance, (full, blocks), chi, outcome, target
        )
        model._lyapunov_end_cycle(blocks, cycle)

    full_product = full["frame"][0]
    reconstructed = torch.zeros(
        (dimension, dimension), dtype=model.dtype
    )
    start = 0
    for block_index, block_size in enumerate(blocks["block_sizes"]):
        stop = start + int(block_size)
        image = (
            blocks["frame"][0, :, start:stop]
            @ blocks["block_core_hat"][block_index][0]
            * torch.exp(blocks["block_core_log_scale"][block_index][0])
        )
        reconstructed += image @ image.mH
        start = stop
    torch.testing.assert_close(
        reconstructed,
        full_product @ full_product.mH,
        rtol=5e-10,
        atol=5e-10,
    )


def test_hard_exterior_record_replay_and_callback_filtering():
    torch = pytest.importorskip("torch")
    model = _model(dw=True, truncation=True)
    exterior_records = []
    checkpoints_seen = []
    common = dict(
        G_history=False,
        progress=False,
        cycles=2,
        samples=1,
        save=False,
        return_data=False,
        sequence="raster_y",
        meas_slab_only=True,
        perfect_correction=True,
        state_representation="physical_frame",
        batch_size=1,
        lyapunov_basis_mode="pure_occupied_empty",
        lyapunov_frame_observer=lambda **row: checkpoints_seen.append(row["cycle"]),
        lyapunov_observer_cycles=[2],
        require_no_covariance_materialization=True,
    )
    torch.manual_seed(123)
    model.run_markov_circuit(
        **common,
        exterior_outcome_observer=lambda **row: exterior_records.append(
            row["outcome_occupied"].detach().clone()
        ),
    )
    assert checkpoints_seen == [2]
    assert len(exterior_records) == 1

    replay_records = []
    torch.manual_seed(999)
    model.run_markov_circuit(
        **common,
        frozen_exterior_outcomes=exterior_records[0],
        exterior_outcome_observer=lambda **row: replay_records.append(
            row["outcome_occupied"].detach().clone()
        ),
    )
    torch.testing.assert_close(replay_records[0], exterior_records[0])



def test_scale_aware_pure_svd_retains_physical_unit_mode_and_matches_dense_gram():
    torch = pytest.importorskip("torch")
    dimension = 20
    frame = torch.eye(dimension, dtype=torch.complex128).unsqueeze(0)
    first_singular = torch.tensor(
        [1.0, *([0.5] * 8), np.exp(-30.0)], dtype=torch.float64
    )
    second_singular = torch.linspace(0.3, 0.9, 10, dtype=torch.float64)
    first_core = torch.diag(first_singular).to(torch.complex128).unsqueeze(0)
    second_core = torch.diag(second_singular).to(torch.complex128).unsqueeze(0)
    observer = LogGramObserver(
        checkpoints=(1,),
        active_indices=np.arange(dimension),
        samples=1,
        arm="pure",
    )
    observer.pure_callback(
        cycle=1,
        batch_start=0,
        lyapunov_frame=frame,
        lyapunov_block_sizes=(10, 10),
        lyapunov_block_core_hat=(first_core, second_core),
        lyapunov_block_core_log_scale=(
            torch.tensor([30.0], dtype=torch.float64),
            torch.zeros(1, dtype=torch.float64),
        ),
    )
    arrays = observer.arrays([0])
    values = arrays["log_gram_eigenvalues"][0, 0]
    vectors = arrays["log_gram_eigenvectors"][0, 0]
    assert abs(values[0]) < 1e-12

    full_sigma = np.concatenate(
        (first_singular.numpy() * np.exp(30.0), second_singular.numpy())
    )
    full_h = 2.0 * np.log(full_sigma)
    expected = full_h[np.argsort(np.abs(full_h), kind="stable")[:16]]
    np.testing.assert_allclose(values, expected, rtol=0.0, atol=2e-12)
    dense_gram = np.diag(full_sigma**2)
    for index in range(16):
        vector = vectors[:, index]
        residual = np.linalg.norm(
            dense_gram @ vector - np.exp(values[index]) * vector
        ) / max(np.linalg.norm(dense_gram, ord=2), 1.0)
        assert residual < 1e-12


def test_degenerate_procrustes_transport_preserves_cluster_identity():
    transport = EigenbasisTransport()
    values = np.asarray([0.0, 0.0])
    labels = np.asarray([0, 0], dtype=np.int8)
    initial = np.eye(4, 2, dtype=np.complex128)
    aligned_initial, clusters_initial = transport.align(values, initial, labels)
    rotation = np.asarray(
        [[1.0, 1.0], [-1.0, 1.0]], dtype=np.complex128
    ) / np.sqrt(2.0)
    rotated = initial @ rotation
    aligned_rotated, clusters_rotated = transport.align(values, rotated, labels)
    np.testing.assert_allclose(
        aligned_rotated, aligned_initial, rtol=0.0, atol=1e-12
    )
    np.testing.assert_array_equal(clusters_rotated, clusters_initial)


def test_resume_identity_rejects_pilot_production_collision_and_orphans(tmp_path):
    data_path = tmp_path / "shard_00.npz"
    manifest_path = tmp_path / "shard_00.manifest.json"
    np.savez_compressed(
        data_path,
        log_gram_eigenvalues=np.zeros((2, 6, 16), dtype=np.float64),
        log_gram_eigenvectors=np.zeros((2, 6, 20, 16), dtype=np.complex128),
    )
    checksum = hashlib.sha256(data_path.read_bytes()).hexdigest()
    identity = {
        "bundle": "test",
        "campaign_revision": "revision",
        "sample_count": 2,
    }
    manifest_path.write_text(
        json.dumps({**identity, "output_sha256": checksum}) + "\n"
    )
    assert _existing_valid(
        data_path, manifest_path, expected_identity=identity
    )
    with pytest.raises(RuntimeError, match="identity mismatch"):
        _existing_valid(
            data_path,
            manifest_path,
            expected_identity={**identity, "sample_count": 5},
        )
    manifest_path.unlink()
    with pytest.raises(RuntimeError, match="orphan immutable"):
        _existing_valid(data_path, manifest_path, expected_identity=identity)


def test_new_runner_options_preserve_old_positional_parameter_order():
    parameters = list(inspect.signature(classA_U1FGTN_gpu.run_markov_circuit).parameters)
    assert parameters.index("native_cycle_observer") == parameters.index("cycle_observer") + 1
    assert parameters.index("site_observer") == parameters.index("native_cycle_observer") + 1
    old_tail = parameters.index("frame_reorthonormalize_interval")
    for name in (
        "cycle_observer_cycles",
        "native_cycle_observer_cycles",
        "exterior_outcome_observer",
        "frozen_exterior_outcomes",
        "lyapunov_observer_cycles",
        "lyapunov_basis_mode",
    ):
        assert parameters.index(name) > old_tail


def test_production_cli_rejects_out_of_range_shard_ids():
    with pytest.raises(SystemExit):
        runner_main(
            [
                "production",
                "--shard-index",
                "5",
                "--allow-cpu",
            ]
        )
