from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fgtn.classA_U1FGTN import classA_U1FGTN


def _model(*, nx: int = 2, ny: int = 3) -> classA_U1FGTN:
    model = classA_U1FGTN(nx, ny, DW=False, nshell=0)
    model.construct_OW_projectors(nshell=0, DW=False)
    return model


def _kwargs() -> dict[str, object]:
    return {
        "G_history": False,
        "progress": False,
        "samples": 1,
        "save": False,
        "sequence": "random",
        "meas_slab_only": False,
        "random_seed": 1729,
        "perfect_correction": True,
        "init_mode": "maxmix",
    }


def test_reverse_raster_is_exact_reverse_after_filtering():
    model = classA_U1FGTN(8, 3, DW=True, nshell=0, dw_truncation=True)
    forward = model._sequence_helper(
        "raster_y", skip_trivial=True
    )["iter_fn"]()
    reverse = model._sequence_helper(
        "reverse_raster_y", skip_trivial=True
    )["iter_fn"]()
    alias = model._sequence_helper(
        "reverse_raster", skip_trivial=True
    )["iter_fn"]()
    assert reverse == list(reversed(forward))
    assert alias == reverse


def test_random_schedule_is_a_fresh_complete_permutation():
    helper = _model()._sequence_helper("random", rng=np.random.default_rng(11))
    expected = set(helper["coords_for_len"])
    first = helper["iter_fn"]()
    second = helper["iter_fn"]()
    assert len(first) == len(expected) == len(set(first))
    assert len(second) == len(expected) == len(set(second))
    assert set(first) == expected == set(second)
    assert first != second


def test_checkpoint_restart_matches_uninterrupted_random_word_exactly():
    expected = _model().run_markov_circuit(cycles=4, **_kwargs())["G_final"][0]
    states: list[dict[str, object]] = []
    _model().run_markov_circuit(
        cycles=2,
        checkpoint_observer=lambda **payload: states.append(payload["state"]),
        **_kwargs(),
    )
    resumed = _model().run_markov_circuit(
        cycles=4,
        checkpoint_state=states[-1],
        **_kwargs(),
    )
    np.testing.assert_array_equal(resumed["G_final"][0], expected)
    assert resumed["checkpoint_state"]["completed_cycles"] == 4


def test_checkpoint_rejects_changed_schedule_or_projectors():
    states: list[dict[str, object]] = []
    _model().run_markov_circuit(
        cycles=1,
        checkpoint_observer=lambda **payload: states.append(payload["state"]),
        **_kwargs(),
    )
    changed = _kwargs()
    changed["sequence"] = "raster_y"
    with pytest.raises(ValueError, match="Checkpoint signature"):
        _model().run_markov_circuit(cycles=2, checkpoint_state=states[-1], **changed)

    changed_model = classA_U1FGTN(2, 3, DW=False, nshell=0, twist_y=0.2)
    changed_model.construct_OW_projectors(nshell=0, DW=False, twist_y=0.2)
    with pytest.raises(ValueError, match="Checkpoint signature"):
        changed_model.run_markov_circuit(
            cycles=2, checkpoint_state=states[-1], **_kwargs()
        )


def test_checkpoint_requires_explicit_seed_and_serial_sample():
    kwargs = _kwargs()
    kwargs["random_seed"] = None
    with pytest.raises(ValueError, match="explicit nonnegative random_seed"):
        _model().run_markov_circuit(
            cycles=1, checkpoint_observer=lambda **_: None, **kwargs
        )

    kwargs = _kwargs()
    kwargs["samples"] = 2
    with pytest.raises(ValueError, match="exactly one serial sample"):
        _model().run_markov_circuit(
            cycles=1, checkpoint_observer=lambda **_: None, **kwargs
        )
