from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/fgtn"))

from classA_U1FGTN import classA_U1FGTN


class _Record:
    def __init__(self) -> None:
        self.entries = []

    def __call__(self, *, cycle, site_id, branch_events, **_):
        self.entries.append(
            {
                "cycle": int(cycle),
                "site_id": int(site_id),
                "branch_events": [dict(event) for event in branch_events],
            }
        )


def _small_model(*, twist_x=0.0, twist_y=0.0):
    model = classA_U1FGTN(
        Nx=1,
        Ny=2,
        DW=False,
        nshell=0,
        dw_truncation=False,
        twist_x=twist_x,
        twist_y=twist_y,
    )
    model.construct_OW_projectors(
        nshell=0,
        DW=False,
        dw_truncation=False,
        twist_x=twist_x,
        twist_y=twist_y,
    )
    return model


def _run_kwargs():
    return {
        "G_history": False,
        "progress": False,
        "cycles": 2,
        "samples": 1,
        "parallelize_samples": False,
        "init_mode": "maxmix",
        "save": False,
        "sequence": "raster_y",
        "meas_slab_only": False,
        "random_seed": 31415,
        "perfect_correction": True,
    }


def test_phi_zero_replay_reproduces_reference_trajectory():
    recorder = _Record()
    reference = _small_model()
    expected = reference.run_markov_circuit(
        trajectory_weight_observer=recorder, **_run_kwargs()
    )["G_final"][0]

    replay_recorder = _Record()
    replay = _small_model()
    actual = replay.run_markov_circuit(
        trajectory_replay=recorder.entries,
        trajectory_weight_observer=replay_recorder,
        **_run_kwargs(),
    )["G_final"][0]

    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)
    assert [entry["site_id"] for entry in replay_recorder.entries] == [
        entry["site_id"] for entry in recorder.entries
    ]
    assert [
        event["outcome_occupied"]
        for entry in replay_recorder.entries
        for event in entry["branch_events"]
        if event["kind"] == "measurement"
    ] == [
        event["outcome_occupied"]
        for entry in recorder.entries
        for event in entry["branch_events"]
        if event["kind"] == "measurement"
    ]


def test_compact_replay_can_omit_diagnostic_event_probabilities():
    recorder = _Record()
    expected = _small_model().run_markov_circuit(
        trajectory_weight_observer=recorder, **_run_kwargs()
    )["G_final"][0]
    compact = []
    for entry in recorder.entries:
        compact.append(
            {
                "cycle": entry["cycle"],
                "site_id": entry["site_id"],
                "branch_events": [
                    {
                        key: value
                        for key, value in event.items()
                        if key not in {
                            "probability",
                            "log_weight",
                            "replay_reference_probability",
                            "replay_probability_error",
                            "perfect_correction",
                        }
                    }
                    for event in entry["branch_events"]
                ],
            }
        )

    replay_recorder = _Record()
    actual = _small_model().run_markov_circuit(
        trajectory_replay=compact,
        trajectory_weight_observer=replay_recorder,
        **_run_kwargs(),
    )["G_final"][0]

    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)
    measurement_events = [
        event
        for entry in replay_recorder.entries
        for event in entry["branch_events"]
        if event["kind"] == "measurement"
    ]
    assert measurement_events
    assert all(event.get("replay_reference_probability") is None for event in measurement_events)
    assert all(event.get("replay_probability_error") is None for event in measurement_events)


def test_replay_rejects_sub_tolerance_forced_branch():
    recorder = _Record()
    _small_model().run_markov_circuit(
        trajectory_weight_observer=recorder, **_run_kwargs()
    )
    with pytest.raises(FloatingPointError, match="forced trajectory branch"):
        _small_model().run_markov_circuit(
            trajectory_replay=recorder.entries,
            trajectory_replay_probability_tol=1.0,
            **_run_kwargs(),
        )


def test_twist_shifts_transverse_momentum_grid():
    zero = classA_U1FGTN(Nx=2, Ny=4, DW=False, nshell=0, twist_y=0.0)
    zero.construct_OW_projectors(nshell=0, DW=False, twist_y=0.0)
    closed = classA_U1FGTN(Nx=2, Ny=4, DW=False, nshell=0, twist_y=2 * np.pi)
    closed.construct_OW_projectors(nshell=0, DW=False, twist_y=2 * np.pi)
    np.testing.assert_allclose(
        closed.Pminus, np.roll(zero.Pminus, -1, axis=1), atol=1e-12, rtol=1e-12
    )


def test_twist_x_shifts_longitudinal_momentum_grid():
    zero = classA_U1FGTN(Nx=4, Ny=2, DW=False, nshell=0)
    zero.construct_OW_projectors(nshell=0, DW=False)
    closed = classA_U1FGTN(
        Nx=4, Ny=2, DW=False, nshell=0, twist_x=2 * np.pi
    )
    closed.construct_OW_projectors(
        nshell=0, DW=False, twist_x=2 * np.pi
    )
    np.testing.assert_allclose(
        closed.Pminus, np.roll(zero.Pminus, -1, axis=0), atol=1e-12, rtol=1e-12
    )


def test_twisted_ow_orbitals_obey_large_gauge_closure():
    nx, ny = 3, 4
    zero = classA_U1FGTN(Nx=nx, Ny=ny, DW=False, nshell=1)
    zero.construct_OW_projectors(nshell=1, DW=False)
    closed_x = classA_U1FGTN(
        Nx=nx, Ny=ny, DW=False, nshell=1, twist_x=2 * np.pi
    )
    closed_x.construct_OW_projectors(
        nshell=1, DW=False, twist_x=2 * np.pi
    )
    closed_y = classA_U1FGTN(
        Nx=nx, Ny=ny, DW=False, nshell=1, twist_y=2 * np.pi
    )
    closed_y.construct_OW_projectors(
        nshell=1, DW=False, twist_y=2 * np.pi
    )

    row = np.arange(2 * nx * ny)
    cell = row // 2
    x = cell % nx
    y = cell // nx
    gauge_x = np.exp(2j * np.pi * x / nx)[:, None, None]
    gauge_y = np.exp(2j * np.pi * y / ny)[:, None, None]
    for name in ("WF_Ap", "WF_Am", "WF_Bp", "WF_Bm"):
        np.testing.assert_allclose(
            getattr(closed_x, name), gauge_x * getattr(zero, name), atol=2e-12
        )
        np.testing.assert_allclose(
            getattr(closed_y, name), gauge_y * getattr(zero, name), atol=2e-12
        )


def test_twist_x_is_reported_by_canonical_run():
    result = _small_model(twist_x=0.25, twist_y=-0.5).run_markov_circuit(
        **_run_kwargs()
    )
    assert result["twist_x"] == pytest.approx(0.25)
    assert result["twist_y"] == pytest.approx(-0.5)


@pytest.mark.parametrize("name", ["twist_x", "twist_y"])
def test_nonfinite_twists_are_rejected(name):
    with pytest.raises(ValueError, match=name):
        classA_U1FGTN(Nx=2, Ny=2, DW=False, **{name: np.inf})


def test_full_space_tangent_frame_overrides_slab_reduction():
    model = classA_U1FGTN(
        Nx=8, Ny=1, DW=True, nshell=0, dw_truncation=True
    )
    model.construct_OW_projectors(
        nshell=0, DW=True, dw_truncation=True
    )
    dimension = 16
    observed_shapes = []

    def observer(*, lyapunov_frame, lyapunov_core_hat, **_):
        observed_shapes.append(
            (tuple(lyapunov_frame.shape), tuple(lyapunov_core_hat.shape))
        )

    model.run_markov_circuit(
        G_history=False,
        progress=False,
        cycles=1,
        samples=1,
        parallelize_samples=False,
        init_mode="maxmix",
        save=False,
        sequence="raster_y",
        meas_slab_only=True,
        random_seed=7,
        perfect_correction=True,
        lyapunov_frame_observer=observer,
        lyapunov_initial_frame=np.eye(dimension, dtype=np.complex128),
        lyapunov_full_space=True,
        lyapunov_track_restricted_core=True,
    )
    assert observed_shapes == [((1, dimension, dimension), (1, dimension, dimension))]
