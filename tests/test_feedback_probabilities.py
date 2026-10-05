from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fgtn.classA_U1FGTN import classA_U1FGTN
from fgtn.classA_U1FGTN_gpu import classA_U1FGTN_gpu


def test_feedback_probability_defaults_and_legacy_alias():
    assert classA_U1FGTN._resolve_feedback_probabilities() == (0.5, 0.5, 0.5)
    assert classA_U1FGTN._resolve_feedback_probabilities(n_a=0.2) == (0.2, 0.2, 0.8)
    assert classA_U1FGTN._resolve_feedback_probabilities(n_a=0.2, p_gain=0.9) == (0.2, 0.9, 0.8)
    assert classA_U1FGTN._resolve_feedback_probabilities(n_a=0.2, p_loss=0.1) == (0.2, 0.2, 0.1)


def test_gpu_feedback_probability_resolver_matches_cpu():
    cases = [
        {},
        {"n_a": 0.2},
        {"n_a": 0.2, "p_gain": 0.9},
        {"n_a": 0.2, "p_loss": 0.1},
        {"n_a": 0.2, "p_gain": 0.9, "p_loss": 0.1},
    ]
    for kwargs in cases:
        assert classA_U1FGTN_gpu._resolve_feedback_probabilities(**kwargs) == classA_U1FGTN._resolve_feedback_probabilities(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_a": -0.1},
        {"n_a": 1.1},
        {"p_gain": -0.1},
        {"p_gain": 1.1},
        {"p_loss": -0.1},
        {"p_loss": 1.1},
    ],
)
def test_feedback_probability_validation(kwargs):
    with pytest.raises(ValueError):
        classA_U1FGTN._resolve_feedback_probabilities(**kwargs)
    with pytest.raises(ValueError):
        classA_U1FGTN_gpu._resolve_feedback_probabilities(**kwargs)
