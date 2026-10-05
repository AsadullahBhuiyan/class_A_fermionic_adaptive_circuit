from __future__ import annotations

import numpy as np
import pytest

from src.fgtn.classA_U1FGTN import classA_U1FGTN
from src.fgtn.occupied_frame import OccupiedFrameState, UpdateTimingCollector


def _random_frame(dimension: int, rank: int, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal((dimension, rank)) + 1j * rng.standard_normal(
        (dimension, rank)
    )
    frame, _ = np.linalg.qr(raw, mode="reduced")
    return frame


def _random_orbital(dimension: int, seed: int = 11) -> np.ndarray:
    rng = np.random.default_rng(seed)
    orbital = rng.standard_normal(dimension) + 1j * rng.standard_normal(dimension)
    return orbital / np.linalg.norm(orbital)


def _state(frame: np.ndarray, *, timing: str = "off") -> OccupiedFrameState:
    return OccupiedFrameState(
        frame,
        representation="physical_frame",
        physical_dimension=frame.shape[0],
        timing=UpdateTimingCollector(timing),
    )


def test_gain_and_loss_match_rank_one_projector_identities():
    frame = _random_frame(12, 5)
    orbital = _random_orbital(12)
    correlation = frame @ frame.conj().T
    support = np.arange(12)

    gained = _state(frame)
    gain = gained.gain_local(support, orbital)
    residual = (np.eye(12) - correlation) @ orbital
    expected_gain = correlation + np.outer(residual, residual.conj()) / np.vdot(
        residual, residual
    )
    np.testing.assert_allclose(gain.probability, np.vdot(residual, residual).real)
    np.testing.assert_allclose(gained.physical_correlation(), expected_gain, atol=2e-13)
    assert gained.rank == 6
    assert gained.gram_residual() < 2e-13

    lost = _state(frame)
    loss = lost.loss_local(support, orbital)
    occupied = correlation @ orbital
    expected_loss = correlation - np.outer(occupied, occupied.conj()) / np.vdot(
        occupied, occupied
    )
    np.testing.assert_allclose(loss.probability, np.vdot(occupied, occupied).real)
    np.testing.assert_allclose(lost.physical_correlation(), expected_loss, atol=2e-13)
    assert lost.rank == 4
    assert lost.gram_residual() < 2e-13


def test_pauli_blocked_and_empty_loss_are_literal_zero_branches():
    frame = np.eye(6, dtype=np.complex128)[:, :3]
    occupied = frame[:, 1]
    empty = np.eye(6, dtype=np.complex128)[:, 5]
    with pytest.raises(FloatingPointError, match="Pauli-blocked gain"):
        _state(frame).gain_local(np.arange(6), occupied)
    with pytest.raises(FloatingPointError, match="Empty-orbital loss"):
        _state(frame).loss_local(np.arange(6), empty)


def test_occupied_and_empty_projectors_match_closed_covariance_updates():
    frame = _random_frame(10, 4, seed=19)
    orbital = _random_orbital(10, seed=23)
    correlation = frame @ frame.conj().T
    support = np.arange(10)
    occupied_probability = float(np.real(orbital.conj() @ correlation @ orbital))
    empty_probability = 1.0 - occupied_probability

    occupied_state = _state(frame)
    observed = occupied_state.project_occupied_local(support, orbital)
    expected = (
        correlation
        - np.outer(correlation @ orbital, (correlation @ orbital).conj())
        / occupied_probability
        + np.outer(orbital, orbital.conj())
    )
    np.testing.assert_allclose(observed, occupied_probability, atol=2e-13)
    np.testing.assert_allclose(occupied_state.physical_correlation(), expected, atol=3e-13)
    assert occupied_state.rank == frame.shape[1]

    empty_state = _state(frame)
    observed = empty_state.project_empty_local(support, orbital)
    residual = (np.eye(10) - correlation) @ orbital
    expected = (
        correlation
        + np.outer(residual, residual.conj()) / empty_probability
        - np.outer(orbital, orbital.conj())
    )
    np.testing.assert_allclose(observed, empty_probability, atol=2e-13)
    np.testing.assert_allclose(empty_state.physical_correlation(), expected, atol=3e-13)
    assert empty_state.rank == frame.shape[1]


def test_simplified_perfect_correction_matches_unsimplified_words():
    frame = _random_frame(10, 5, seed=29)
    orbital = _random_orbital(10, seed=31)
    support = np.arange(10)

    bare_loss = _state(frame)
    bare_loss.loss_local(support, orbital)
    decomposed_loss = _state(frame)
    decomposed_loss.project_occupied_local(support, orbital)
    deterministic_loss = decomposed_loss.loss_local(support, orbital)
    np.testing.assert_allclose(deterministic_loss.probability, 1.0, atol=2e-13)
    np.testing.assert_allclose(
        bare_loss.physical_correlation(),
        decomposed_loss.physical_correlation(),
        atol=4e-13,
    )

    bare_gain = _state(frame)
    bare_gain.gain_local(support, orbital)
    decomposed_gain = _state(frame)
    decomposed_gain.project_empty_local(support, orbital)
    deterministic_gain = decomposed_gain.gain_local(support, orbital)
    np.testing.assert_allclose(deterministic_gain.probability, 1.0, atol=2e-13)
    np.testing.assert_allclose(
        bare_gain.physical_correlation(),
        decomposed_gain.physical_correlation(),
        atol=4e-13,
    )


def test_repeated_nonorthogonal_word_preserves_reduced_orthonormal_frame():
    frame = _random_frame(14, 7, seed=37)
    state = _state(frame, timing="detailed")
    support = np.arange(14)
    for seed in range(40, 48):
        orbital = _random_orbital(14, seed=seed)
        if seed % 2:
            state.project_occupied_local(support, orbital)
        else:
            state.project_empty_local(support, orbital)
    assert state.rank == 7
    assert state.gram_residual() < 2e-12
    timing = state.timing.snapshot()
    assert timing["counts"]["gain_count"] == 8
    assert timing["counts"]["loss_count"] == 8
    assert timing["total_ns"]["loss_householder_apply"] > 0


def test_householder_deletion_is_stable_for_nearly_aligned_last_coefficient():
    frame = np.eye(8, dtype=np.complex128)[:, :4]
    orbital = frame[:, -1] + 1e-13 * np.eye(8, dtype=np.complex128)[:, 6]
    orbital /= np.linalg.norm(orbital)
    state = _state(frame)
    result = state.loss_local(np.arange(8), orbital)
    np.testing.assert_allclose(result.probability, 1.0, atol=2e-13)
    np.testing.assert_allclose(
        state.physical_correlation(),
        np.diag([1, 1, 1, 0, 0, 0, 0, 0]),
        atol=3e-13,
    )
    assert state.gram_residual() < 2e-13


def test_pure_and_maxmix_initialization_and_frame_native_observables():
    frame = _random_frame(12, 6, seed=43)
    correlation = frame @ frame.conj().T
    centered = 2.0 * correlation - np.eye(12)
    pure = OccupiedFrameState.from_centered_covariance(
        centered, representation="physical_frame"
    )
    np.testing.assert_allclose(pure.centered_covariance(), centered, atol=3e-13)
    rows = np.arange(6)
    np.testing.assert_allclose(
        pure.regional_charge(rows), np.trace(correlation[np.ix_(rows, rows)]).real
    )
    left = np.asarray([0, 2, 7])
    right = np.asarray([1, 2, 9])
    np.testing.assert_allclose(
        pure.selected_correlators(left, right),
        correlation[left, right],
        atol=2e-13,
    )
    restricted_evals = np.linalg.eigvalsh(correlation[np.ix_(rows, rows)])
    expected_entropy = OccupiedFrameState._binary_entropy(restricted_evals)
    np.testing.assert_allclose(pure.regional_entropy(rows), expected_entropy, atol=2e-13)

    maxmix = OccupiedFrameState.maximally_mixed(12)
    np.testing.assert_allclose(maxmix.physical_correlation(), np.eye(12) / 2.0)
    np.testing.assert_allclose(maxmix.physical_entropy(), 12.0 * np.log(2.0))
    assert maxmix.rank == 12
    assert maxmix.gram_residual() < 2e-13


class _Record:
    def __init__(self):
        self.entries: list[dict] = []
        self.cumulative: list[float] = []

    def __call__(self, **payload):
        self.entries.append(
            {
                "cycle": int(payload["cycle"]),
                "site_id": int(payload["site_id"]),
                "branch_events": tuple(
                    dict(event) for event in payload["branch_events"]
                ),
            }
        )
        self.cumulative.append(float(payload["cumulative_log_weight"]))


def _scheduled_frame_run(seed: int, initial: np.ndarray, schedule: np.ndarray):
    model = classA_U1FGTN(
        Nx=2,
        Ny=2,
        DW=False,
        nshell=1,
        alpha_1=1,
        alpha_2=1,
        trial_orbitals="X",
    )
    record = _Record()
    result = model.run_markov_circuit(
        cycles=int(schedule.shape[0]),
        samples=1,
        init_mode="default",
        G_init=initial,
        sequence="random",
        perfect_correction=True,
        random_seed=seed,
        G_history=False,
        save=False,
        progress=False,
        state_representation="physical_frame",
        return_native_state=True,
        site_schedule_replay=schedule,
        trajectory_weight_observer=record,
    )
    return result, record


def test_site_schedule_replay_fixes_order_but_leaves_outcomes_live():
    initializer = classA_U1FGTN(
        Nx=2,
        Ny=2,
        DW=False,
        nshell=1,
        alpha_1=1,
        alpha_2=1,
        trial_orbitals="X",
    )
    initial = initializer.random_complex_fermion_covariance(
        N=8, rng=np.random.default_rng(104)
    )
    schedule = np.asarray([[3, 1, 0, 2], [2, 0, 3, 1]], dtype=np.int64)

    first, first_record = _scheduled_frame_run(501, initial, schedule)
    repeat, repeat_record = _scheduled_frame_run(501, initial, schedule)
    changed, changed_record = _scheduled_frame_run(502, initial, schedule)

    expected_sites = schedule.reshape(-1).tolist()
    assert [entry["site_id"] for entry in first_record.entries] == expected_sites
    assert [entry["site_id"] for entry in changed_record.entries] == expected_sites
    assert first_record.entries == repeat_record.entries
    np.testing.assert_allclose(
        first["native_final"]["frame"], repeat["native_final"]["frame"], atol=0.0
    )
    first_outcomes = [
        event["outcome_occupied"]
        for entry in first_record.entries
        for event in entry["branch_events"]
        if event["kind"] == "measurement"
    ]
    changed_outcomes = [
        event["outcome_occupied"]
        for entry in changed_record.entries
        for event in entry["branch_events"]
        if event["kind"] == "measurement"
    ]
    assert first_outcomes != changed_outcomes
    assert first["site_schedule_replay"] is True
    assert first["site_schedule_replay_shape"] == [2, 4]
    assert first["site_schedule_replay_sha256"] == initializer._checkpoint_array_signature(
        schedule
    )


def test_site_schedule_replay_validation_and_conflicts():
    model = classA_U1FGTN(
        Nx=2,
        Ny=2,
        DW=False,
        nshell=1,
        alpha_1=1,
        alpha_2=1,
        trial_orbitals="X",
    )
    common = dict(
        cycles=1,
        init_mode="default",
        sequence="random",
        random_seed=77,
        G_history=False,
        save=False,
        progress=False,
    )
    with pytest.raises(ValueError, match="exactly once"):
        model.run_markov_circuit(
            **common,
            site_schedule_replay=np.asarray([[0, 0, 2, 3]], dtype=np.int64),
        )
    with pytest.raises(ValueError, match="integer site IDs"):
        model.run_markov_circuit(
            **common,
            site_schedule_replay=np.asarray([[0.0, 1.0, 2.0, 3.0]]),
        )
    with pytest.raises(ValueError, match="one row per cycle"):
        model.run_markov_circuit(
            **common,
            site_schedule_replay=np.asarray(
                [[0, 1, 2, 3], [0, 1, 2, 3]], dtype=np.int64
            ),
        )
    with pytest.raises(ValueError, match="cannot be combined"):
        model.run_markov_circuit(
            **common,
            site_schedule_replay=np.asarray([[0, 1, 2, 3]], dtype=np.int64),
            trajectory_replay=[
                {"cycle": 1, "site_id": site, "branch_events": []}
                for site in range(4)
            ],
        )
    with pytest.raises(ValueError, match="exactly one serial sample"):
        model.run_markov_circuit(
            **common,
            samples=2,
            site_schedule_replay=np.asarray([[0, 1, 2, 3]], dtype=np.int64),
        )


@pytest.mark.parametrize(
    ("init_mode", "frame_representation"),
    (("default", "physical_frame"), ("maxmix", "purification_frame")),
)
def test_canonical_one_cycle_record_replay_matches_covariance(
    init_mode, frame_representation
):
    kwargs = {
        "Nx": 2,
        "Ny": 2,
        "DW": False,
        "nshell": 1,
        "alpha_1": 1,
        "alpha_2": 1,
        "trial_orbitals": "X",
    }
    covariance_model = classA_U1FGTN(**kwargs)
    initial: list[np.ndarray] = []
    final: list[np.ndarray] = []
    record = _Record()

    def covariance_cycle(**payload):
        target = initial if int(payload["cycle"]) == 0 else final
        target.append(np.array(payload["G"], copy=True))

    covariance_model.run_markov_circuit(
        cycles=1,
        samples=1,
        init_mode=init_mode,
        sequence="random",
        perfect_correction=True,
        random_seed=991,
        G_history=False,
        save=False,
        progress=False,
        cycle_observer=covariance_cycle,
        trajectory_weight_observer=record,
    )

    frame_model = classA_U1FGTN(**kwargs)
    frame_cycles: list[np.ndarray] = []
    replay = _Record()
    result = frame_model.run_markov_circuit(
        cycles=1,
        samples=1,
        init_mode=init_mode,
        G_init=initial[0],
        sequence="random",
        perfect_correction=True,
        random_seed=992,
        G_history=False,
        save=False,
        progress=False,
        state_representation=frame_representation,
        return_native_state=True,
        trajectory_replay=record.entries,
        trajectory_weight_observer=replay,
        native_cycle_observer=lambda **payload: frame_cycles.append(
            payload["state"].centered_covariance()
        ),
        timing_level="detailed",
    )

    np.testing.assert_allclose(frame_cycles[0], initial[0], atol=5e-13)
    np.testing.assert_allclose(frame_cycles[-1], final[-1], atol=2e-11)
    np.testing.assert_allclose(replay.cumulative, record.cumulative, atol=2e-11)
    assert result["state_representation"] == frame_representation
    assert result["native_final"]["gram_residual"] < 1e-10
    assert result["timing"]["total_ns"]["trajectory_total"] > 0
    counts = result["timing"]["counts"]
    initial_rank = 4 if frame_representation == "physical_frame" else 8
    assert result["native_final"]["rank"] == (
        initial_rank + counts["gain_count"] - counts["loss_count"]
    )


def test_physical_frame_rejects_maximally_mixed_initial_state():
    with pytest.raises(ValueError, match="requires a pure initial covariance"):
        OccupiedFrameState.from_centered_covariance(
            np.zeros((6, 6), dtype=np.complex128),
            representation="physical_frame",
        )


def test_auto_dispatch_and_materialization_guard():
    model = classA_U1FGTN(
        Nx=2, Ny=2, DW=False, nshell=1, alpha_1=1, alpha_2=1
    )
    cycles = []
    result = model.run_markov_circuit(
        cycles=1,
        samples=2,
        init_mode="default",
        random_seed=1201,
        G_history=False,
        save=False,
        progress=False,
        return_native_state=True,
        require_no_covariance_materialization=True,
        native_cycle_observer=lambda **payload: cycles.append(
            (payload["batch_start"], payload["cycle"], payload["state"].rank)
        ),
    )
    assert result["state_representation_requested"] == "auto"
    assert result["state_representation_resolved"] == "physical_frame"
    assert result["covariance_materialization_count"] == 0
    assert len(result["native_final"]) == 2
    assert [(sample, cycle) for sample, cycle, _ in cycles] == [
        (0, 0), (0, 1), (1, 0), (1, 1)
    ]

    with pytest.raises(RuntimeError, match="forbids the requested"):
        model.run_markov_circuit(
            cycles=1,
            samples=1,
            init_mode="default",
            random_seed=1202,
            G_history=False,
            save=False,
            progress=False,
            cycle_observer=lambda **payload: None,
            require_no_covariance_materialization=True,
        )


def test_auto_dispatch_keeps_maxmix_on_covariance_and_rejects_heterogeneous_batch():
    model = classA_U1FGTN(
        Nx=2, Ny=2, DW=False, nshell=1, alpha_1=1, alpha_2=1
    )
    result = model.run_markov_circuit(
        cycles=1,
        samples=1,
        init_mode="maxmix",
        random_seed=1203,
        G_history=False,
        save=False,
        progress=False,
    )
    assert result["state_representation_resolved"] == "covariance"

    pure = model.random_complex_fermion_covariance(
        8, rng=np.random.default_rng(1204)
    )
    mixed = np.zeros_like(pure)
    with pytest.raises(ValueError, match="heterogeneous pure/mixed batch"):
        model.run_markov_circuit(
            cycles=1,
            samples=2,
            G_init=np.stack((pure, mixed)),
            G_history=False,
            save=False,
            progress=False,
        )


def test_frame_checkpoint_resume_and_postselection_are_native():
    kwargs = dict(Nx=2, Ny=2, DW=False, nshell=1, alpha_1=1, alpha_2=1)
    checkpoints = []
    model = classA_U1FGTN(**kwargs)
    model.run_markov_circuit(
        cycles=1,
        samples=1,
        init_mode="default",
        random_seed=1205,
        G_history=False,
        save=False,
        progress=False,
        return_native_state=True,
        checkpoint_observer=lambda **payload: checkpoints.append(payload["state"]),
    )
    checkpoint = checkpoints[-1]
    assert checkpoint["version"] == 2
    assert checkpoint["native_state"]["representation"] == "physical_frame"

    resumed = classA_U1FGTN(**kwargs).run_markov_circuit(
        cycles=2,
        samples=1,
        init_mode="default",
        random_seed=1205,
        G_history=False,
        save=False,
        progress=False,
        return_native_state=True,
        checkpoint_state=checkpoint,
    )
    assert resumed["native_final"]["gram_residual"] < 1e-9

    postselected = classA_U1FGTN(**kwargs).run_markov_circuit(
        cycles=1,
        samples=1,
        init_mode="default",
        random_seed=1206,
        postselect=True,
        G_history=False,
        save=False,
        progress=False,
        return_native_state=True,
    )
    assert postselected["state_representation_resolved"] == "physical_frame"


def test_native_frame_timed_path_does_not_materialize_covariance(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("timed native frame path reconstructed a covariance")

    monkeypatch.setattr(OccupiedFrameState, "centered_covariance", forbidden)
    events: list[tuple[int, int, str]] = []
    model = classA_U1FGTN(
        Nx=2,
        Ny=2,
        DW=False,
        nshell=1,
        alpha_1=1,
        alpha_2=1,
        trial_orbitals="X",
    )
    result = model.run_markov_circuit(
        cycles=1,
        samples=1,
        init_mode="maxmix",
        sequence="random",
        perfect_correction=True,
        random_seed=818,
        G_history=False,
        save=False,
        progress=False,
        state_representation="purification_frame",
        return_native_state=True,
        native_event_observer=lambda **payload: events.append(
            (payload["cycle"], payload["site_id"], payload["channel"])
        ),
        timing_level="coarse",
    )
    assert len(events) == 2 * 2 * 4
    assert result["native_final"]["representation"] == "purification_frame"


def test_maxmix_doubled_frame_matches_dense_choi_replay():
    kwargs = {
        "Nx": 2,
        "Ny": 2,
        "DW": False,
        "nshell": 1,
        "alpha_1": 1,
        "alpha_2": 1,
        "trial_orbitals": "X",
    }
    record = _Record()
    initial: list[np.ndarray] = []
    dense_choi: list[np.ndarray] = []

    def cycle_observer(**payload):
        if int(payload["cycle"]) == 0:
            initial.append(np.array(payload["G"], copy=True))

    def choi_observer(**payload):
        identity = np.eye(payload["sigma_ll"].shape[-1], dtype=np.complex128)
        dense_choi.append(
            np.block(
                [
                    [
                        0.5 * (payload["sigma_ll"][0] + identity),
                        0.5 * payload["sigma_lr"][0],
                    ],
                    [
                        0.5 * payload["sigma_lr"][0].conj().T,
                        0.5 * (payload["sigma_rr"][0] + identity),
                    ],
                ]
            )
        )

    classA_U1FGTN(**kwargs).run_markov_circuit(
        cycles=1,
        samples=1,
        init_mode="maxmix",
        sequence="random",
        perfect_correction=True,
        random_seed=1441,
        G_history=False,
        save=False,
        progress=False,
        cycle_observer=cycle_observer,
        trajectory_weight_observer=record,
        track_choi=True,
        choi_observer=choi_observer,
        choi_observer_cycles=[1],
    )
    frame_projector: list[np.ndarray] = []
    classA_U1FGTN(**kwargs).run_markov_circuit(
        cycles=1,
        samples=1,
        init_mode="maxmix",
        G_init=initial[0],
        sequence="random",
        perfect_correction=True,
        random_seed=1442,
        G_history=False,
        save=False,
        progress=False,
        trajectory_replay=record.entries,
        state_representation="purification_frame",
        return_native_state=True,
        native_cycle_observer=lambda **payload: (
            frame_projector.append(payload["state"].doubled_projector())
            if int(payload["cycle"]) == 1
            else None
        ),
    )
    assert len(dense_choi) == len(frame_projector) == 1
    np.testing.assert_allclose(frame_projector[0], dense_choi[0], atol=3e-11)
