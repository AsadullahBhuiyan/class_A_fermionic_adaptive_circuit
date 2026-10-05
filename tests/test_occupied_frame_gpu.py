from __future__ import annotations

import pytest
import numpy as np

torch = pytest.importorskip("torch")

from src.fgtn.classA_U1FGTN_gpu import classA_U1FGTN_gpu
from src.fgtn.occupied_frame_gpu import BatchedOccupiedFrameState


def _orbital(batch, dimension, seed):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    real = torch.randn((batch, dimension), dtype=torch.float64, generator=generator)
    imag = torch.randn((batch, dimension), dtype=torch.float64, generator=generator)
    value = torch.complex(real, imag)
    return value / torch.linalg.vector_norm(value, dim=1, keepdim=True)


def test_gpu_padded_gain_loss_and_divergent_ranks_match_projectors():
    state = BatchedOccupiedFrameState.random_pure(
        3, 10, 5, device="cpu", dtype=torch.complex128
    )
    orbital = _orbital(3, 10, 71)
    correlation = state.physical_correlation().clone()
    probability = state.occupation_probability(orbital)
    selected_loss = torch.tensor([True, False, False])
    selected_gain = torch.tensor([False, True, False])
    state.loss(orbital, selected_loss)
    state.gain(orbital, selected_gain)
    assert state.ranks.tolist() == [4, 6, 5]
    expected_loss = correlation[0] - torch.outer(
        correlation[0] @ orbital[0], (correlation[0] @ orbital[0]).conj()
    ) / probability[0]
    residual = orbital[1] - correlation[1] @ orbital[1]
    expected_gain = correlation[1] + torch.outer(residual, residual.conj()) / (
        1.0 - probability[1]
    )
    observed = state.physical_correlation()
    torch.testing.assert_close(observed[0], expected_loss, atol=2e-11, rtol=2e-11)
    torch.testing.assert_close(observed[1], expected_gain, atol=2e-11, rtol=2e-11)
    assert float(state.gram_residual().max()) < 1e-10


def test_gpu_runner_auto_frame_has_complete_native_cycle_axis():
    model = classA_U1FGTN_gpu(
        2, 2, DW=False, nshell=1, device="cpu", dtype="complex128"
    )
    observed = []
    result = model.run_markov_circuit(
        cycles=2,
        samples=3,
        batch_size=3,
        init_mode="default",
        G_history=False,
        save=False,
        progress=False,
        return_native_state=True,
        require_no_covariance_materialization=True,
        native_cycle_observer=lambda **payload: observed.append(payload["cycle"]),
    )
    assert observed == [0, 1, 2]
    assert result["state_representation_resolved"] == "physical_frame"
    assert result["covariance_materialization_count"] == 0
    assert result["native_final"]["ranks"].shape == (3,)


def test_gpu_frozen_record_frame_and_covariance_agree_every_cycle_and_channel():
    model = classA_U1FGTN_gpu(
        2, 2, DW=False, nshell=1, device="cpu", dtype="complex128"
    )
    samples, cycles = 2, 3
    initial_state = BatchedOccupiedFrameState.random_pure(
        samples, model.Nlayer, model.Nlayer // 2,
        device="cpu", dtype=torch.complex128,
        generator=torch.Generator(device="cpu").manual_seed(112),
    )
    initial = initial_state.centered_covariance(reason="test_fixture")
    sequence = model._sequence_helper("raster_y", meas_slab_only=False)
    site_ids = torch.as_tensor(
        [int(x + model.Nx * y) for x, y in sequence["coords_for_len"]],
        dtype=torch.long,
    )
    schedule = site_ids.view(1, 1, -1).expand(samples, cycles, -1).clone()
    outcomes = torch.zeros(
        (samples, cycles, site_ids.numel(), 4), dtype=torch.bool
    )

    histories = {}
    records = {}

    def run(representation, *, replay):
        torch.manual_seed(991)
        history, record = [], []

        def observe(*, state, **_):
            if hasattr(state, "centered_covariance"):
                value = state.centered_covariance(reason="paired_test_observation")
            else:
                value = state
            history.append(value.detach().clone())

        def record_observer(**payload):
            if not replay:
                sample_indices = payload["sample_indices"].to(torch.long).cpu()
                outcomes[
                    sample_indices,
                    int(payload["cycle"]) - 1,
                    int(payload["update_index"]),
                    : payload["outcome_occupied"].shape[1],
                ] = payload["outcome_occupied"].detach().cpu()
            record.append(
                (
                    int(payload["cycle"]),
                    int(payload["update_index"]),
                    payload["occupation_probability"].detach().clone(),
                    payload["outcome_occupied"].detach().clone(),
                )
            )

        result = model.run_markov_circuit(
            cycles=cycles,
            samples=samples,
            batch_size=samples,
            init_mode="default",
            G_init=initial,
            G_history=False,
            save=False,
            progress=False,
            return_data=False,
            sequence="raster_y",
            perfect_correction=True,
                frozen_schedule=schedule if replay else None,
                frozen_outcomes=outcomes if replay else None,
            state_representation=representation,
            native_cycle_observer=observe,
            record_observer=record_observer,
        )
        histories[representation] = history
        records[representation] = record
        return result

    frame_result = run("physical_frame", replay=False)
    covariance_result = run("covariance", replay=True)
    assert len(histories["physical_frame"]) == cycles + 1
    assert len(records["physical_frame"]) == cycles * site_ids.numel()
    for left, right in zip(
        histories["physical_frame"], histories["covariance"], strict=True
    ):
        torch.testing.assert_close(left, right, atol=2e-9, rtol=2e-9)
    for left, right in zip(
        records["physical_frame"], records["covariance"], strict=True
    ):
        assert left[:2] == right[:2]
        torch.testing.assert_close(left[2], right[2], atol=2e-10, rtol=2e-10)
        torch.testing.assert_close(left[3], right[3])
    assert frame_result["state_representation_resolved"] == "physical_frame"
    assert covariance_result["explicit_covariance_override"] is True


def test_gpu_native_observer_materialization_guard_and_mean_resolution():
    model = classA_U1FGTN_gpu(
        2, 2, DW=False, nshell=1, device="cpu", dtype="complex128"
    )
    with pytest.raises(RuntimeError, match="native_cycle_observer"):
        model.run_markov_circuit(
            cycles=1,
            samples=1,
            batch_size=1,
            G_history=False,
            save=False,
            progress=False,
            return_data=False,
            require_no_covariance_materialization=True,
            native_cycle_observer=lambda *, state, **_: state.centered_covariance(
                reason="forbidden_test_materialization"
            ),
        )
    seen = []
    result = model.run_markov_circuit(
        cycles=1,
        samples=2,
        batch_size=2,
        init_mode="maxmix",
        mean_replacement=True,
        G_history=False,
        save=False,
        progress=False,
        return_data=False,
        native_cycle_observer=lambda **payload: seen.append(payload["cycle"]),
    )
    assert seen == [0, 1]
    assert result["state_representation_resolved"] == "covariance"
    assert result["state_representation_resolution_reason"] == "mean_replacement_is_mixed"
