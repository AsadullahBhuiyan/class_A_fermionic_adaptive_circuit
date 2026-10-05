from __future__ import annotations

import numpy as np
import pytest

from fgtn.classA_U1FGTN import classA_U1FGTN
from fgtn.diagnostics.mean_lindblad import PerfectCorrectionLindblad


def _model() -> classA_U1FGTN:
    return classA_U1FGTN(
        Nx=3,
        Ny=4,
        DW=True,
        nshell=1,
        alpha_1=1,
        alpha_2=30,
        trial_orbitals="X",
        dw_truncation=True,
        dw_interval=(1, 2),
    )


@pytest.mark.parametrize("include_dephasing", [False, True])
def test_class_lindblad_q0_entrypoint_matches_validated_engine(include_dephasing: bool):
    model = _model()
    times = np.asarray([0.0, 0.1, 0.2])
    result = model.run_lindblad_evolution(
        cycles=0.2,
        dt=0.05,
        init_mode="maxmix",
        include_number_dephasing=include_dephasing,
        observation_times=times,
        representation="q0",
    )

    engine = PerfectCorrectionLindblad.from_canonical_model(model)
    d = 2 * model.Nx
    initial = np.broadcast_to(
        0.5 * np.eye(d, dtype=np.complex128), (model.Ny, d, d)
    ).copy()
    expected = engine.integrate_q_sector(
        initial,
        q_index=0,
        dt=0.05,
        observation_times=times,
        include_number_dephasing=include_dephasing,
    )
    np.testing.assert_allclose(result["correlation_history"], expected.states)
    np.testing.assert_array_equal(result["times"], times)
    assert result["run_config"]["gain_rate"] == 1.0
    assert result["run_config"]["loss_rate"] == 1.0
    assert result["run_config"]["number_dephasing_rate"] == (
        1.0 if include_dephasing else 0.0
    )
    assert result["run_config"]["DW_loc"] == [1, 2]


def test_class_lindblad_dense_class_convention_conversion():
    model = _model()
    dimension = model.Ntot // 2
    class_covariance = np.zeros((dimension, dimension), dtype=np.complex128)
    result = model.run_lindblad_evolution(
        cycles=0,
        dt=0.05,
        G_init=class_covariance,
        initial_convention="class",
        include_number_dephasing=True,
        observation_times=[0.0],
        representation="dense",
    )
    np.testing.assert_allclose(
        result["correlation_final"],
        0.5 * np.eye(dimension, dtype=np.complex128),
    )


def test_class_lindblad_q0_rejects_translation_breaking_initial_state():
    model = _model()
    dimension = model.Ntot // 2
    initial = 0.5 * np.eye(dimension, dtype=np.complex128)
    initial[0, 0] = 0.75
    with pytest.raises(ValueError, match="translation-invariant"):
        model.run_lindblad_evolution(
            cycles=0,
            dt=0.05,
            G_init=initial,
            initial_convention="correlation",
            observation_times=[0.0],
            representation="q0",
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"cycles": -1}, "cycles"),
        ({"dt": 0}, "dt"),
        ({"representation": "bad"}, "representation"),
        ({"init_mode": "bad"}, "init_mode"),
    ],
)
def test_class_lindblad_entrypoint_validation(kwargs, message):
    with pytest.raises(ValueError, match=message):
        _model().run_lindblad_evolution(
            observation_times=[0.0],
            **kwargs,
        )
