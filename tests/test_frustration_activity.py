from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fgtn.classA_U1FGTN import classA_U1FGTN
from fgtn.diagnostics import (
    RegionMasks,
    TrajectoryActivityRecorder,
    analyze_activity,
    bootstrap_scgf,
    empirical_scgf,
)


def _single_cell_regions() -> RegionMasks:
    active = np.ones((1, 1), dtype=bool)
    return RegionMasks(
        names=("all",),
        masks=active[None, ...],
        active=active,
        interface_width=1,
        wall_x=(),
    )


def test_activity_event_parser_targets_probabilities_and_signed_transfer():
    recorder = TrajectoryActivityRecorder(nx=1, ny=1, cycles=1, samples=1, site_ids=[0])
    events = [
        {"channel": "Ap", "kind": "measurement", "probability": 0.75, "outcome_occupied": True},
        {"channel": "Am", "kind": "measurement", "probability": 0.20, "outcome_occupied": False},
        {"channel": "Bp", "kind": "measurement", "probability": 0.10, "outcome_occupied": False},
        {"channel": "Bm", "kind": "measurement", "probability": 0.80, "outcome_occupied": True},
        {"channel": "Ap", "kind": "correction", "expected_occupied": False, "target_occupied": False},
        {"channel": "Am", "kind": "correction", "expected_occupied": True, "target_occupied": True},
    ]
    recorder(
        cycle=1,
        site_id=0,
        sample_index=0,
        branch_events=events,
        forced_postselect=False,
    )
    recorder.assert_complete()

    np.testing.assert_array_equal(recorder.defect[0, 0, 0], [1, 1, 0, 0])
    np.testing.assert_array_equal(recorder.transfer[0, 0, 0], [-1, 1, 0, 0])
    np.testing.assert_allclose(recorder.success_probability[0, 0, 0], [0.25, 0.20, 0.90, 0.80])
    assert np.all((recorder.success_probability[recorder.valid] >= 0.0))
    assert np.all((recorder.success_probability[recorder.valid] <= 1.0))


def test_forced_postselection_does_not_manufacture_activity():
    recorder = TrajectoryActivityRecorder(nx=1, ny=1, cycles=1, samples=1, site_ids=[0])
    recorder(
        cycle=1,
        site_id=0,
        sample_index=0,
        branch_events=(),
        forced_postselect=True,
    )
    recorder.assert_complete(allow_forced=True)
    assert recorder.forced_site[0, 0, 0]
    assert not np.any(recorder.valid)
    assert not np.any(recorder.defect)
    assert not np.any(recorder.transfer)
    assert np.all(np.isnan(recorder.success_probability))


def test_empirical_scgf_matches_exact_bernoulli_enumeration():
    observation_time = 5
    samples = np.arange(2**observation_time, dtype=np.uint64)
    bits = ((samples[:, None] >> np.arange(observation_time, dtype=np.uint64)) & 1).astype(int)
    counts = np.sum(bits, axis=1)
    fields = np.linspace(-0.4, 0.4, 17)

    theta, effective_fraction = empirical_scgf(
        counts,
        observation_time=observation_time,
        s_grid=fields,
    )
    expected = np.log((1.0 + np.exp(-fields)) / 2.0)
    np.testing.assert_allclose(theta, expected, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(theta[fields == 0.0], 0.0, atol=1e-15)
    assert np.all((effective_fraction > 0.0) & (effective_fraction <= 1.0))

    low_one, high_one = bootstrap_scgf(
        counts,
        observation_time=observation_time,
        s_grid=fields,
        bootstrap_samples=32,
        seed=17,
    )
    low_two, high_two = bootstrap_scgf(
        counts,
        observation_time=observation_time,
        s_grid=fields,
        bootstrap_samples=32,
        seed=17,
    )
    np.testing.assert_array_equal(low_one, low_two)
    np.testing.assert_array_equal(high_one, high_two)


def test_activity_analysis_recovers_bernoulli_cumulants():
    cycles = 4
    samples = 2**cycles
    recorder = TrajectoryActivityRecorder(nx=1, ny=1, cycles=cycles, samples=samples, site_ids=[0])
    labels = np.arange(samples, dtype=np.uint64)
    bits = ((labels[:, None] >> np.arange(cycles, dtype=np.uint64)) & 1).astype(np.uint8)
    recorder.defect[:, :, 0, 0] = bits
    recorder.transfer[:, :, 0, 0] = -bits.astype(np.int8)
    recorder.valid[:] = True
    recorder.success_probability[:] = 0.5

    fields = np.linspace(-0.3, 0.3, 13)
    analysis = analyze_activity(
        recorder,
        regions=_single_cell_regions(),
        burn_in=0,
        s_grid=fields,
        window_fractions=(1.0,),
        bootstrap_samples=16,
        bootstrap_seed=9,
    )
    np.testing.assert_allclose(
        analysis.cumulants_per_cycle[0, 0, 0],
        [0.5, 0.25, 0.0],
        atol=1e-14,
    )
    expected = np.log((1.0 + np.exp(-fields)) / 2.0)
    np.testing.assert_allclose(analysis.theta[0, 0, 0], expected, atol=1e-13)
    assert np.all(analysis.reliable[0, 0, 0])


def test_waiting_time_survival_for_periodic_events():
    recorder = TrajectoryActivityRecorder(nx=1, ny=1, cycles=6, samples=2, site_ids=[0])
    recorder.defect[:, [0, 2, 4], 0, 0] = 1
    recorder.transfer[:, [0, 2, 4], 0, 0] = -1
    recorder.valid[:] = True
    recorder.success_probability[:] = 0.5
    analysis = analyze_activity(
        recorder,
        regions=_single_cell_regions(),
        burn_in=0,
        window_fractions=(1.0,),
        bootstrap_samples=0,
    )

    np.testing.assert_allclose(analysis.waiting_survival[0, 0, 0, :3], [1.0, 1.0, 0.0])
    assert analysis.waiting_interval_count[0, 0, 0] == 4
    np.testing.assert_allclose(analysis.cumulants_per_cycle[0, 0, 0], [0.5, 0.0, 0.0])


def _uniform_model() -> classA_U1FGTN:
    model = classA_U1FGTN(2, 2, DW=False, nshell=1, alpha_1=1, alpha_2=1, trial_orbitals="X")
    model.construct_OW_projectors(nshell=1, DW=False, trial_orbitals="X", dw_truncation=False)
    return model


def _recorded_run(seed: int):
    model = _uniform_model()
    recorder = TrajectoryActivityRecorder.from_model(model, cycles=2, samples=2)
    result = model.run_markov_circuit(
        G_history=False,
        progress=False,
        cycles=2,
        samples=2,
        save=False,
        perfect_correction=True,
        sequence="random",
        random_seed=seed,
        trajectory_weight_observer=recorder,
    )
    recorder.assert_complete()
    return result, recorder


def test_seeded_live_trajectories_are_reproducible_and_change_with_seed():
    first_result, first = _recorded_run(123)
    repeat_result, repeat = _recorded_run(123)
    changed_result, changed = _recorded_run(124)

    assert first_result["sample_seeds"] == repeat_result["sample_seeds"]
    assert first_result["rng_streams"] == ["initialization", "exterior", "schedule", "dynamics"]
    assert "SeedSequence" in first_result["seed_derivation"]
    np.testing.assert_array_equal(first.defect, repeat.defect)
    np.testing.assert_array_equal(first.transfer, repeat.transfer)
    np.testing.assert_array_equal(first.visit_order, repeat.visit_order)
    np.testing.assert_allclose(first.success_probability, repeat.success_probability, equal_nan=True)
    np.testing.assert_allclose(first_result["G_final"], repeat_result["G_final"], atol=0.0, rtol=0.0)
    np.testing.assert_array_equal(first.defect, np.abs(first.transfer))

    same_record = (
        np.array_equal(first.defect, changed.defect)
        and np.array_equal(first.visit_order, changed.visit_order)
        and np.array_equal(first_result["G_final"], changed_result["G_final"])
    )
    assert not same_record


def test_seeded_samples_are_independent_of_serial_or_parallel_dispatch():
    common = dict(
        G_history=False,
        progress=False,
        cycles=1,
        samples=3,
        save=False,
        perfect_correction=True,
        sequence="random",
        random_seed=456,
    )
    serial = _uniform_model().run_markov_circuit(parallelize_samples=False, **common)
    parallel = _uniform_model().run_markov_circuit(
        parallelize_samples=True,
        n_jobs=2,
        backend="threading",
        throttle=False,
        **common,
    )
    assert serial["sample_seeds"] == parallel["sample_seeds"]
    np.testing.assert_allclose(serial["G_final"], parallel["G_final"], rtol=0.0, atol=0.0)


@pytest.mark.parametrize("seed", [True, -1, 1.5])
def test_random_seed_validation(seed):
    with pytest.raises(ValueError, match="random_seed"):
        _uniform_model().run_markov_circuit(
            G_history=False,
            progress=False,
            cycles=1,
            samples=1,
            save=False,
            random_seed=seed,
        )
