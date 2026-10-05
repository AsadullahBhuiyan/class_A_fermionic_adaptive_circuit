"""Focused regression tests for the fixed-record tangent cocycle.

These tests intentionally exercise the canonical dynamics classes rather than a
second implementation of the Markov-cycle loop.  At the primitive level the
one-particle factor ``L`` is tested through the covariance differential

    delta G' = L delta G L^dagger.

The discrete Born outcome is held fixed in every finite difference.  In
particular, these are branchwise derivatives and never derivatives through the
outcome sampler.
"""

from __future__ import annotations

import inspect
from pathlib import Path
import sys

import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fgtn.classA_U1FGTN import classA_U1FGTN
from fgtn.classA_U1FGTN_gpu import classA_U1FGTN_gpu


def _normalized(vector):
    vector = np.asarray(vector, dtype=np.complex128)
    return vector / np.linalg.norm(vector)


def _mixed_covariance(dimension: int, seed: int = 1234, radius: float = 0.45):
    """Return a deterministic Hermitian covariance strictly inside [-I,I]."""
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(dimension, dimension)) + 1j * rng.normal(
        size=(dimension, dimension)
    )
    hermitian = 0.5 * (raw + raw.conj().T)
    spectral_radius = np.max(np.abs(np.linalg.eigvalsh(hermitian)))
    return radius * hermitian / spectral_radius


def _hermitian_direction(dimension: int, seed: int = 5678):
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(dimension, dimension)) + 1j * rng.normal(
        size=(dimension, dimension)
    )
    direction = 0.5 * (raw + raw.conj().T)
    return direction / np.linalg.norm(direction, ord="fro")


def _central_difference(map_fn, covariance, direction, epsilon=2.0e-6):
    return (
        map_fn(covariance + epsilon * direction)
        - map_fn(covariance - epsilon * direction)
    ) / (2.0 * epsilon)


def _cpu_state(model, initial_frame, **kwargs):
    initial_frame = np.asarray(initial_frame, dtype=np.complex128)
    return model._init_lyapunov_state(
        batch_count=1,
        n_vec=initial_frame.shape[-1],
        initial_frame=initial_frame,
        **kwargs,
    )


def _gpu_state(model, initial_frame, **kwargs):
    """Construct a GPU state while an in-progress API is being synchronized.

    The desired public primitive accepts ``initial_frame``.  Assigning the same
    frame after initialization keeps the CPU/GPU factor test useful while older
    deployable copies are being synchronized.
    """
    torch = pytest.importorskip("torch")
    frame = torch.as_tensor(initial_frame, dtype=model.dtype, device=model.device)
    parameters = inspect.signature(model._init_lyapunov_state).parameters
    if "initial_frame" in parameters:
        state = model._init_lyapunov_state(
            batch_count=1,
            n_vec=frame.shape[-1],
            initial_frame=frame,
            **kwargs,
        )
    else:
        state = model._init_lyapunov_state(
            batch_count=1,
            n_vec=frame.shape[-1],
        )
        state["frame"][0] = frame
    return state


@pytest.fixture()
def cpu_model():
    # Nlayer=4 is large enough for nontrivial support/complement tests.
    return classA_U1FGTN(Nx=1, Ny=2, DW=False, nshell=None)


@pytest.fixture()
def gpu_model():
    pytest.importorskip("torch")
    return classA_U1FGTN_gpu(
        Nx=1,
        Ny=2,
        DW=False,
        nshell=None,
        device="cpu",
        dtype="complex128",
        backend="dense",
    )


@pytest.mark.parametrize("particle", [False, True])
def test_rank_one_measurement_factor_matches_covariance_finite_difference(
    cpu_model, particle
):
    n = cpu_model.Ntot // 2
    covariance = _mixed_covariance(n)
    direction = _hermitian_direction(n)
    chi = _normalized([1.0, 0.4j, -0.3, 0.2 - 0.1j])
    projector = np.outer(chi, chi.conj())

    state = _cpu_state(cpu_model, np.eye(n, dtype=np.complex128))
    cpu_model._lyapunov_apply_dense_channel(
        state, covariance, np.asarray([0]), projector, particle=particle
    )
    factor = state["frame"][0]

    numerical = _central_difference(
        lambda trial: cpu_model.measure_only_top_layer(
            trial, projector, particle=particle, chi=chi
        ),
        covariance,
        direction,
    )
    predicted = factor @ direction @ factor.conj().T

    np.testing.assert_allclose(numerical, predicted, rtol=2.0e-6, atol=2.0e-8)


@pytest.mark.parametrize("particle", [False, True])
@pytest.mark.parametrize("target_occupied", [False, True])
def test_measurement_plus_reset_factor_matches_finite_difference(
    cpu_model, particle, target_occupied
):
    n = cpu_model.Ntot // 2
    covariance = _mixed_covariance(n, seed=10)
    direction = _hermitian_direction(n, seed=11)
    chi = _normalized([0.2 + 0.1j, 1.0, -0.25j, 0.35])
    projector = np.outer(chi, chi.conj())
    complement = np.eye(n, dtype=np.complex128) - projector
    reset_covariance = 1.0 if target_occupied else -1.0

    state = _cpu_state(cpu_model, np.eye(n, dtype=np.complex128))
    cpu_model._lyapunov_apply_dense_channel(
        state, covariance, np.asarray([0]), projector, particle=particle
    )
    factor_after_measurement = state["frame"][0].copy()
    cpu_model._lyapunov_apply_dense_reset(state, np.asarray([0]), projector)
    factor_after_reset = state["frame"][0]

    def fixed_branch_map(trial):
        measured = cpu_model.measure_only_top_layer(
            trial, projector, particle=particle, chi=chi
        )
        return (
            complement @ measured @ complement
            + reset_covariance * projector
        )

    numerical = _central_difference(fixed_branch_map, covariance, direction)
    predicted = factor_after_reset @ direction @ factor_after_reset.conj().T

    # A projective measurement already puts the tangent image in Q, so the
    # subsequent gain/loss reset derivative is exactly redundant.
    np.testing.assert_allclose(
        factor_after_reset, factor_after_measurement, rtol=2.0e-12, atol=2.0e-12
    )
    np.testing.assert_allclose(numerical, predicted, rtol=2.0e-6, atol=2.0e-8)


@pytest.mark.parametrize("particle", [False, True])
def test_dense_and_finite_support_measurement_factors_agree(cpu_model, particle):
    n = cpu_model.Ntot // 2
    covariance = _mixed_covariance(n, seed=20)
    initial_frame, _ = np.linalg.qr(
        _hermitian_direction(n, seed=21)[:, :2], mode="reduced"
    )
    support = np.asarray([0, 2], dtype=np.int64)
    complement = np.asarray([1, 3], dtype=np.int64)
    chi_local = _normalized([1.0, 0.3 + 0.4j])
    chi = np.zeros((n,), dtype=np.complex128)
    chi[support] = chi_local
    projector = np.outer(chi, chi.conj())

    dense_state = _cpu_state(cpu_model, initial_frame)
    local_state = _cpu_state(cpu_model, initial_frame)
    cpu_model._lyapunov_apply_dense_channel(
        dense_state, covariance, np.asarray([0]), projector, particle=particle
    )
    cpu_model._lyapunov_apply_local_channel(
        local_state,
        covariance,
        np.asarray([0]),
        support,
        complement,
        chi_local,
        particle=particle,
    )

    dense_covariance = cpu_model.measure_only_top_layer(
        covariance, projector, particle=particle, chi=chi
    )
    local_covariance = cpu_model._measure_only_top_layer_local(
        covariance,
        support,
        complement,
        chi_local,
        particle=particle,
    )

    np.testing.assert_allclose(
        local_state["frame"], dense_state["frame"], rtol=2.0e-12, atol=2.0e-12
    )
    np.testing.assert_allclose(
        local_covariance, dense_covariance, rtol=2.0e-12, atol=2.0e-12
    )

    before_reset = local_state["frame"].copy()
    cpu_model._lyapunov_apply_local_reset(
        local_state, np.asarray([0]), support, complement, chi_local
    )
    np.testing.assert_allclose(
        local_state["frame"], before_reset, rtol=2.0e-12, atol=2.0e-12
    )


@pytest.mark.parametrize("particle", [False, True])
def test_impossible_fixed_branch_raises_or_is_censored(cpu_model, particle):
    n = cpu_model.Ntot // 2
    chi = _normalized([1.0, 0.5j, -0.2, 0.1])
    projector = np.outer(chi, chi.conj())
    identity = np.eye(n, dtype=np.complex128)
    # particle=True is impossible when <G>_chi=-1; particle=False is
    # impossible when <G>_chi=+1.
    covariance = identity - 2.0 * projector if particle else 2.0 * projector - identity

    raising = _cpu_state(
        cpu_model,
        identity,
        singular_tol=1.0e-12,
        failure_mode="raise",
    )
    with pytest.raises(FloatingPointError, match="zero or non-finite Born probability"):
        cpu_model._lyapunov_apply_dense_channel(
            raising, covariance, np.asarray([0]), projector, particle=particle
        )

    censoring = _cpu_state(
        cpu_model,
        identity,
        singular_tol=1.0e-12,
        failure_mode="censor",
    )
    cpu_model._lyapunov_apply_dense_channel(
        censoring, covariance, np.asarray([0]), projector, particle=particle
    )
    assert not bool(censoring["active"][0])
    assert int(censoring["invalid_branch_count"][0]) == 1
    assert censoring["min_branch_probability"][0] <= 1.0e-14
    assert len(censoring["failure_records"]) == 1
    np.testing.assert_array_equal(censoring["frame"][0], 0.0)


def test_qr_gauge_reconstructs_pre_qr_frame_and_restricted_core(cpu_model):
    n = cpu_model.Ntot // 2
    initial_frame = np.eye(n, 2, dtype=np.complex128)
    state = _cpu_state(
        cpu_model, initial_frame, track_restricted_core=True
    )
    raw = np.asarray(
        [
            [1.0 + 0.3j, -0.2j],
            [0.2 - 0.4j, 0.8 + 0.1j],
            [-0.3, 0.1 + 0.5j],
            [0.4j, -0.2 + 0.2j],
        ],
        dtype=np.complex128,
    )
    state["frame"][0] = raw

    cpu_model._lyapunov_end_cycle(state, cycle=1)
    q_factor = state["frame"][0]
    r_factor = state["last_r"][0]
    diagonal = np.diag(r_factor)

    np.testing.assert_allclose(q_factor.conj().T @ q_factor, np.eye(2), atol=2.0e-14)
    np.testing.assert_allclose(q_factor @ r_factor, raw, rtol=2.0e-14, atol=2.0e-14)
    np.testing.assert_allclose(diagonal.imag, 0.0, atol=2.0e-14)
    assert np.all(diagonal.real >= -2.0e-14)

    restricted_core = (
        np.exp(state["core_log_scale"][0]) * state["core_hat"][0]
    )
    np.testing.assert_allclose(
        q_factor @ restricted_core, raw, rtol=2.0e-14, atol=2.0e-14
    )


def test_custom_two_mode_core_is_invariant_under_u2_frame_rotation(cpu_model):
    n = cpu_model.Ntot // 2
    rng = np.random.default_rng(40)
    seed_frame = rng.normal(size=(n, 2)) + 1j * rng.normal(size=(n, 2))
    initial_frame, _ = np.linalg.qr(seed_frame, mode="reduced")
    rotation = np.asarray([[1.0, 1.0j], [1.0j, 1.0]]) / np.sqrt(2.0)

    state = _cpu_state(
        cpu_model, initial_frame, track_restricted_core=True
    )
    rotated_state = _cpu_state(
        cpu_model, initial_frame @ rotation, track_restricted_core=True
    )

    covariances = [_mixed_covariance(n, seed=41), _mixed_covariance(n, seed=42)]
    chis = [
        _normalized([1.0, 0.1j, -0.2, 0.3]),
        _normalized([0.2, 1.0, 0.4j, -0.1]),
    ]
    outcomes = [True, False]

    for cycle, (covariance, chi, particle) in enumerate(
        zip(covariances, chis, outcomes), start=1
    ):
        projector = np.outer(chi, chi.conj())
        for current in (state, rotated_state):
            cpu_model._lyapunov_apply_dense_channel(
                current,
                covariance,
                np.asarray([0]),
                projector,
                particle=particle,
            )
            cpu_model._lyapunov_end_cycle(current, cycle=cycle)

    def reconstructed_image(current):
        core = np.exp(current["core_log_scale"][0]) * current["core_hat"][0]
        return current["frame"][0] @ core

    image = reconstructed_image(state)
    rotated_image = reconstructed_image(rotated_state)
    np.testing.assert_allclose(
        rotated_image, image @ rotation, rtol=3.0e-13, atol=3.0e-13
    )
    np.testing.assert_allclose(
        np.linalg.svd(rotated_image, compute_uv=False),
        np.linalg.svd(image, compute_uv=False),
        rtol=3.0e-13,
        atol=3.0e-13,
    )


@pytest.mark.parametrize("local", [False, True])
@pytest.mark.parametrize("particle", [False, True])
def test_cpu_gpu_measurement_factor_parity(cpu_model, gpu_model, local, particle):
    torch = pytest.importorskip("torch")
    n = cpu_model.Ntot // 2
    covariance = _mixed_covariance(n, seed=50)
    seed_frame = np.asarray(
        [[1.0, 0.0], [0.0, 1.0], [0.3j, 0.2], [0.1, -0.4j]],
        dtype=np.complex128,
    )
    initial_frame, _ = np.linalg.qr(seed_frame, mode="reduced")
    support = np.asarray([0, 2], dtype=np.int64)
    complement = np.asarray([1, 3], dtype=np.int64)
    chi_local = _normalized([1.0, 0.2 + 0.3j])
    chi = np.zeros((n,), dtype=np.complex128)
    chi[support] = chi_local
    projector = np.outer(chi, chi.conj())

    cpu_state = _cpu_state(cpu_model, initial_frame)
    gpu_state = _gpu_state(gpu_model, initial_frame)
    offsets_np = np.asarray([0], dtype=np.int64)
    offsets_gpu = torch.as_tensor([0], dtype=torch.long, device=gpu_model.device)
    covariance_gpu = torch.as_tensor(
        covariance[None, ...], dtype=gpu_model.dtype, device=gpu_model.device
    )

    if local:
        support_gpu = torch.as_tensor(support, dtype=torch.long, device=gpu_model.device)
        complement_gpu = torch.as_tensor(
            complement, dtype=torch.long, device=gpu_model.device
        )
        chi_local_gpu = torch.as_tensor(
            chi_local, dtype=gpu_model.dtype, device=gpu_model.device
        )
        cpu_model._lyapunov_apply_local_channel(
            cpu_state,
            covariance,
            offsets_np,
            support,
            complement,
            chi_local,
            particle=particle,
        )
        gpu_model._lyapunov_apply_local_channel(
            gpu_state,
            covariance_gpu,
            offsets_gpu,
            support_gpu,
            complement_gpu,
            chi_local_gpu,
            particle=particle,
        )
    else:
        projector_gpu = torch.as_tensor(
            projector, dtype=gpu_model.dtype, device=gpu_model.device
        )
        cpu_model._lyapunov_apply_dense_channel(
            cpu_state, covariance, offsets_np, projector, particle=particle
        )
        gpu_model._lyapunov_apply_dense_channel(
            gpu_state,
            covariance_gpu,
            offsets_gpu,
            projector_gpu,
            particle=particle,
        )

    gpu_frame = gpu_state["frame"][0].detach().cpu().numpy()
    np.testing.assert_allclose(
        gpu_frame, cpu_state["frame"][0], rtol=3.0e-11, atol=3.0e-12
    )


@pytest.mark.parametrize("local", [False, True])
def test_gpu_reset_factor_is_explicit_and_redundant(gpu_model, local):
    torch = pytest.importorskip("torch")
    n = gpu_model.Nlayer
    covariance = _mixed_covariance(n, seed=60)
    initial_frame = np.eye(n, 2, dtype=np.complex128)
    support = np.asarray([0, 2], dtype=np.int64)
    complement = np.asarray([1, 3], dtype=np.int64)
    chi_local = _normalized([1.0, -0.2j])
    chi = np.zeros((n,), dtype=np.complex128)
    chi[support] = chi_local
    projector = np.outer(chi, chi.conj())

    state = _gpu_state(gpu_model, initial_frame)
    offsets = torch.as_tensor([0], dtype=torch.long, device=gpu_model.device)
    covariance_gpu = torch.as_tensor(
        covariance[None, ...], dtype=gpu_model.dtype, device=gpu_model.device
    )

    if local:
        assert hasattr(gpu_model, "_lyapunov_apply_local_reset"), (
            "The canonical GPU engine must expose the reset differential even "
            "though it is redundant after the present rank-one measurement."
        )
        support_gpu = torch.as_tensor(support, dtype=torch.long, device=gpu_model.device)
        complement_gpu = torch.as_tensor(
            complement, dtype=torch.long, device=gpu_model.device
        )
        chi_local_gpu = torch.as_tensor(
            chi_local, dtype=gpu_model.dtype, device=gpu_model.device
        )
        gpu_model._lyapunov_apply_local_channel(
            state,
            covariance_gpu,
            offsets,
            support_gpu,
            complement_gpu,
            chi_local_gpu,
            particle=True,
        )
        before = state["frame"].clone()
        reset_parameters = inspect.signature(
            gpu_model._lyapunov_apply_local_reset
        ).parameters
        if "comp_idx" in reset_parameters:
            gpu_model._lyapunov_apply_local_reset(
                state, offsets, support_gpu, complement_gpu, chi_local_gpu
            )
        else:
            gpu_model._lyapunov_apply_local_reset(
                state, offsets, support_gpu, chi_local_gpu
            )
    else:
        assert hasattr(gpu_model, "_lyapunov_apply_dense_reset"), (
            "The canonical GPU engine must expose the reset differential even "
            "though it is redundant after the present rank-one measurement."
        )
        projector_gpu = torch.as_tensor(
            projector, dtype=gpu_model.dtype, device=gpu_model.device
        )
        gpu_model._lyapunov_apply_dense_channel(
            state, covariance_gpu, offsets, projector_gpu, particle=True
        )
        before = state["frame"].clone()
        gpu_model._lyapunov_apply_dense_reset(state, offsets, projector_gpu)

    torch.testing.assert_close(state["frame"], before, rtol=2.0e-12, atol=2.0e-12)


def test_gpu_class_declares_explicit_reset_differentials():
    # This structural check remains active on CPU-only test hosts, where the
    # numerical GPU parity tests are necessarily skipped.
    assert callable(getattr(classA_U1FGTN_gpu, "_lyapunov_apply_local_reset", None))
    assert callable(getattr(classA_U1FGTN_gpu, "_lyapunov_apply_dense_reset", None))


def test_gpu_dense_schedule_routes_custom_gain_and_loss_to_every_channel(
    monkeypatch,
):
    # This routing test deliberately uses a skeletal instance, so it runs even
    # when PyTorch is absent from the local CPU test environment.
    gpu_model = object.__new__(classA_U1FGTN_gpu)
    gpu_model.backend = "dense"
    dummy_spinor = np.asarray([1.0], dtype=np.complex128)
    monkeypatch.setattr(
        gpu_model, "_site_spinors", lambda site_id: (dummy_spinor,) * 4
    )
    monkeypatch.setattr(gpu_model, "_site_uses_local_mode", lambda site_id: False)
    calls = []

    def fake_channel(covariance, chi, expected_occupied, **kwargs):
        calls.append(
            {
                "expected_occupied": bool(expected_occupied),
                "p_gain": kwargs.get("p_gain"),
                "p_loss": kwargs.get("p_loss"),
            }
        )
        if kwargs.get("return_occ_prob", False):
            probability = np.full((covariance.shape[0],), 0.5, dtype=np.float64)
            return covariance, probability
        return covariance

    monkeypatch.setattr(gpu_model, "_apply_channel_shared_site", fake_channel)
    covariance = np.zeros((1, 1, 1), dtype=np.complex128)
    gpu_model._markov_meas_feedback_shared_site(
        covariance,
        site_id=0,
        n_a=0.5,
        p_gain=0.23,
        p_loss=0.71,
    )

    assert [call["expected_occupied"] for call in calls] == [False, True, False, True]
    assert all(call["p_gain"] == pytest.approx(0.23) for call in calls)
    assert all(call["p_loss"] == pytest.approx(0.71) for call in calls)


def test_native_start_cycle_propagates_and_reports_only_elapsed_cycles():
    parameters = inspect.signature(classA_U1FGTN.run_markov_circuit).parameters
    required = {
        "lyapunov_frame_observer",
        "lyapunov_initial_frame",
        "lyapunov_start_cycle",
    }
    assert required.issubset(parameters), (
        "The canonical CPU engine is missing the native custom-frame/start-cycle API: "
        f"{sorted(required.difference(parameters))}"
    )

    model = classA_U1FGTN(Nx=1, Ny=1, DW=False, nshell=0)
    frame_events = []
    cycle_states = {}

    def cycle_observer(**payload):
        cycle_states[int(payload["cycle"])] = np.array(payload["G"], copy=True)

    def frame_observer(**payload):
        event = dict(payload)
        for key in ("G", "lyapunov_frame", "frame", "Q"):
            if key in event:
                event[key] = np.array(event[key], copy=True)
        frame_events.append(event)

    initial_frame = np.asarray([[1.0], [0.0]], dtype=np.complex128)
    model.run_markov_circuit(
        G_history=False,
        progress=False,
        cycles=3,
        samples=1,
        parallelize_samples=False,
        init_mode="maxmix",
        save=False,
        save_init=False,
        perfect_correction=True,
        sequence="raster_y",
        meas_slab_only=False,
        random_seed=2468,
        cycle_observer=cycle_observer,
        lyapunov_frame_observer=frame_observer,
        lyapunov_nvec=1,
        lyapunov_initial_frame=initial_frame,
        lyapunov_start_cycle=2,
    )

    assert [int(event["cycle"]) for event in frame_events] == [2, 3]
    elapsed = [
        int(
            event.get(
                "lyapunov_cycle",
                event.get("elapsed_cycle", event.get("lyapunov_elapsed_cycle", -1)),
            )
        )
        for event in frame_events
    ]
    assert elapsed == [1, 2]
    for event in frame_events:
        np.testing.assert_allclose(event["G"], cycle_states[int(event["cycle"])])

