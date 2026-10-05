from __future__ import annotations

import numpy as np
import pytest

from src.fgtn.classA_U1FGTN import classA_U1FGTN
from src.fgtn.occupied_frame import OccupiedFrameState


def _model(*, twist: float = 0.0, hard: bool = False) -> classA_U1FGTN:
    return classA_U1FGTN(
        4,
        4,
        DW=True,
        nshell=1,
        filling_frac=0.5,
        alpha_1=1.0,
        alpha_2=30.0,
        trial_orbitals="X",
        dw_interval=(1, 3),
        dw_truncation=hard,
        twist_y=twist,
    )


def _kwargs(*, cycles: int = 2, hard: bool = False) -> dict:
    return {
        "G_history": False,
        "progress": False,
        "cycles": cycles,
        "perfect_correction": True,
        "samples": 1,
        "parallelize_samples": False,
        "save": False,
        "sequence": "raster_y",
        "meas_slab_only": hard,
        "random_seed": 7123,
        "state_representation": "physical_frame",
        "return_native_state": True,
    }


@pytest.mark.parametrize(
    "schedule,gauge,message",
    [
        ([0.0, 0.1], "uniform", r"cycles \+ 1"),
        ([[0.0, 0.1, 0.2]], "uniform", "one-dimensional"),
        ([0.0, np.nan, 0.2], "uniform", "finite"),
        ([0.0, 0.1, 0.2], "seam", "uniform"),
    ],
)
def test_controller_twist_schedule_validation(schedule, gauge, message):
    with pytest.raises(ValueError, match=message):
        _model().run_markov_circuit(
            controller_twist_schedule=schedule,
            controller_twist_gauge=gauge,
            **_kwargs(),
        )


def test_constant_schedule_has_exact_static_twist_projector_parity():
    twist = 0.271
    static = _model(twist=twist).run_markov_circuit(**_kwargs())
    scheduled = _model(twist=twist).run_markov_circuit(
        controller_twist_schedule=np.full(3, twist),
        controller_twist_gauge="uniform",
        **_kwargs(),
    )
    static_frame = static["native_final"]["frame"]
    scheduled_frame = scheduled["native_final"]["frame"]
    assert np.allclose(
        static_frame @ static_frame.conj().T,
        scheduled_frame @ scheduled_frame.conj().T,
        rtol=0.0,
        atol=1e-13,
    )
    assert scheduled["controller_twist_schedule"] == [twist, twist, twist]


def test_omitting_new_options_matches_their_legacy_defaults_physically():
    legacy_call = _model(twist=0.137).run_markov_circuit(**_kwargs())
    explicit_defaults = _model(twist=0.137).run_markov_circuit(
        controller_twist_schedule=None,
        controller_twist_gauge="uniform",
        frame_init_prepared=False,
        **_kwargs(),
    )
    legacy_frame = legacy_call["native_final"]["frame"]
    explicit_frame = explicit_defaults["native_final"]["frame"]
    assert np.allclose(
        legacy_frame @ legacy_frame.conj().T,
        explicit_frame @ explicit_frame.conj().T,
        rtol=0.0,
        atol=1e-13,
    )
    assert legacy_call["native_final"]["rank"] == explicit_defaults["native_final"]["rank"]


def test_schedule_is_installed_before_each_cycle(monkeypatch):
    model = _model()
    installed: list[float] = []
    original = model.construct_OW_projectors

    def recording_construct(*args, **kwargs):
        installed.append(float(kwargs.get("twist_y", model.twist_y)))
        return original(*args, **kwargs)

    monkeypatch.setattr(model, "construct_OW_projectors", recording_construct)
    seen: list[tuple[int, float]] = []
    model.run_markov_circuit(
        controller_twist_schedule=[0.0, 0.2, 0.4],
        native_cycle_observer=lambda *, cycle, **_: seen.append((cycle, model.twist_y)),
        **_kwargs(),
    )
    assert installed == [0.0, 0.2, 0.4]
    assert seen == [(0, 0.0), (1, 0.2), (2, 0.4)]


def test_prepared_frame_skips_hard_wall_exterior(monkeypatch):
    dimension = 2 * 4 * 4
    frame = OccupiedFrameState.random_pure(
        dimension, dimension // 2, rng=np.random.default_rng(55)
    ).snapshot()
    model = _model(hard=True)

    def forbidden(*args, **kwargs):
        raise AssertionError("exterior preparation was repeated")

    monkeypatch.setattr(model, "_prepare_exterior_product_state", forbidden)
    result = model.run_markov_circuit(
        frame_init=frame,
        frame_init_prepared=True,
        controller_twist_schedule=[0.0, 0.2],
        **_kwargs(cycles=1, hard=True),
    )
    assert result["exterior_preparation"] == "skipped_prepared_frame"


def test_prepared_flag_requires_frame():
    with pytest.raises(ValueError, match="requires frame_init"):
        _model().run_markov_circuit(frame_init_prepared=True, **_kwargs(cycles=1))
