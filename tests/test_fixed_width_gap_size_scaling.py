"""Tests of fixed-time size fits and sampling-error normalization."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


@pytest.fixture
def analysis(monkeypatch):
    bundle = Path(__file__).resolve().parents[1] / '00_WORKSPACE/CURRENT/final_production_new_designs/26_fixed_width_hard_wall_gap_t40'
    monkeypatch.syspath_prepend(str(bundle))
    for name in ('run_campaign', 'endpoint_spectrum', 'analyze_campaign'):
        monkeypatch.delitem(sys.modules, name, raising=False)
    spec = importlib.util.spec_from_file_location('fixed_width_scaling_test', bundle/'analyze_size_scaling.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_exact_power_and_fixed_time_normalization(analysis):
    x = np.arange(20, 49, 4)
    y = .021*(x/32.)**(-1.2)
    sem = np.full(len(x), .003)
    fit = analysis.fit_models(x, y, sem)['power_law']
    np.testing.assert_allclose(list(fit['parameters'].values()), [.021, 1.2], rtol=1e-6)
    raw = analysis.fit_models(x, 80*y, 80*sem)['power_law']
    np.testing.assert_allclose(raw['parameters']['z'], fit['parameters']['z'], rtol=1e-6)
    np.testing.assert_allclose(raw['parameter_sem']['z'], fit['parameter_sem']['z'], rtol=1e-5)
    np.testing.assert_allclose(raw['parameters']['amplitude_at_Ny32'], 80*fit['parameters']['amplitude_at_Ny32'])


def test_absolute_errors_not_residual_scaled(analysis):
    x = np.arange(20, 49, 4)
    y = .02*(x/32.)**(-1.)
    sem = np.full(len(x), .003)
    first = analysis.fit_models(x, y, sem)['power_law']
    doubled = analysis.fit_models(x, y, 2*sem)['power_law']
    assert first['parameter_sem']['z'] > 0.1  # perfect fit does not imply zero uncertainty
    np.testing.assert_allclose(doubled['parameter_sem']['z'], 2*first['parameter_sem']['z'], rtol=1e-6)
    design = np.column_stack([(x/32.)**-1, -.02*(x/32.)**-1*np.log(x/32.)])
    expected = np.linalg.inv((design/sem[:,None]).T@(design/sem[:,None]))
    np.testing.assert_allclose(first['covariance'], expected, rtol=1e-5)


def test_contrast_and_invalid_input(analysis):
    result = analysis.endpoint_contrast([.04, .01], [.004, .003])
    assert result['difference'] == pytest.approx(.03)
    assert result['sem'] == pytest.approx(.005)
    assert result['combined_SEMs'] == pytest.approx(6.)
    assert result['fractional_decrease'] == pytest.approx(.75)
    with pytest.raises(ValueError):
        analysis.fit_models([20, 24, 28], [.03, .02, .01], [.001, 0, .001])
