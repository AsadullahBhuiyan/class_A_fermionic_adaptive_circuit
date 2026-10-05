from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fgtn.classA_U1FGTN import classA_U1FGTN


def make_model(nx: int = 4, ny: int = 3) -> classA_U1FGTN:
    model = classA_U1FGTN(
        nx,
        ny,
        DW=False,
        nshell=1,
        alpha_1=1.0,
        alpha_2=30.0,
        trial_orbitals="X",
    )
    model.construct_OW_projectors(
        nshell=1, DW=False, trial_orbitals="X", dw_truncation=False
    )
    return model


class Recorder:
    def __init__(self) -> None:
        self.entries: list[dict] = []

    def __call__(self, **payload) -> None:
        self.entries.append(
            {
                "cycle": int(payload["cycle"]),
                "site_id": int(payload["site_id"]),
                "branch_log_weight": float(payload["branch_log_weight"]),
                "branch_events": [copy.deepcopy(dict(event)) for event in payload["branch_events"]],
            }
        )


def run(model: classA_U1FGTN, recorder: Recorder, *, selected=None, replay=None):
    return model.run_markov_circuit(
        G_history=False,
        progress=False,
        cycles=2,
        perfect_correction=True,
        samples=1,
        init_mode="maxmix",
        save=False,
        sequence="raster_y",
        meas_slab_only=False,
        measurement_site_ids=selected,
        random_seed=12345,
        state_representation="covariance",
        trajectory_weight_observer=recorder,
        trajectory_replay=replay,
    )


def test_all_site_selection_preserves_existing_trajectory() -> None:
    baseline_record = Recorder()
    selected_record = Recorder()
    baseline = run(make_model(), baseline_record)
    selected = run(
        make_model(), selected_record, selected=np.arange(4 * 3, dtype=np.int64)
    )
    np.testing.assert_allclose(
        baseline["G_final"], selected["G_final"], rtol=2.0e-14, atol=2.0e-15
    )
    assert [entry["site_id"] for entry in baseline_record.entries] == [
        entry["site_id"] for entry in selected_record.entries
    ]
    assert [
        [event["outcome_occupied"] for event in entry["branch_events"] if event["kind"] == "measurement"]
        for entry in baseline_record.entries
    ] == [
        [event["outcome_occupied"] for event in entry["branch_events"] if event["kind"] == "measurement"]
        for entry in selected_record.entries
    ]
    np.testing.assert_allclose(
        [entry["branch_log_weight"] for entry in baseline_record.entries],
        [entry["branch_log_weight"] for entry in selected_record.entries],
        rtol=2.0e-14,
        atol=2.0e-15,
    )
    assert selected["measurement_site_ids"] == list(range(12))


def test_subset_filters_raster_and_supports_exact_replay() -> None:
    selected_ids = np.asarray([1, 5, 9], dtype=np.int64)
    first_record = Recorder()
    first = run(make_model(), first_record, selected=selected_ids)
    assert [entry["site_id"] for entry in first_record.entries] == [1, 5, 9, 1, 5, 9]

    replay_record = Recorder()
    replay = run(
        make_model(),
        replay_record,
        selected=selected_ids,
        replay=first_record.entries,
    )
    np.testing.assert_allclose(
        first["G_final"], replay["G_final"], rtol=2.0e-14, atol=2.0e-15
    )
    assert [entry["site_id"] for entry in replay_record.entries] == [1, 5, 9, 1, 5, 9]


@pytest.mark.parametrize(
    "selected",
    [[], [0, 0], [-1], [12], [True], [1.5], [[0, 1]]],
)
def test_invalid_measurement_site_subsets_are_rejected(selected) -> None:
    with pytest.raises(ValueError, match="measurement_site_ids"):
        run(make_model(), Recorder(), selected=selected)


def test_subset_must_survive_other_geometry_filters() -> None:
    model = classA_U1FGTN(
        8,
        3,
        DW=True,
        nshell=1,
        alpha_1=1.0,
        alpha_2=30.0,
        trial_orbitals="X",
        dw_truncation=True,
        dw_interval=(2, 6),
    )
    model.construct_OW_projectors(
        nshell=1, DW=True, trial_orbitals="X", dw_truncation=True
    )
    with pytest.raises(ValueError, match="excluded by the canonical geometry"):
        model.run_markov_circuit(
            G_history=False,
            progress=False,
            cycles=1,
            init_mode="maxmix",
            save=False,
            meas_slab_only=True,
            measurement_site_ids=[0],
            random_seed=1,
        )
