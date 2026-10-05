"""Saved-time selection and normalization for the Campaign 13 panel-C remake."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


@pytest.fixture
def remake(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    folder = root/'00_WORKSPACE/CURRENT/experiment_review/purification_full_measurement_ny30'
    monkeypatch.syspath_prepend(str(folder))
    for key in ('make_figure', 'analyze_campaign', 'plot_endpoint_lyapunov_gap'):
        monkeypatch.delitem(sys.modules, key, raising=False)
    spec = importlib.util.spec_from_file_location('test_gap_t40_remake', folder/'remake_gap_t40.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fake_spectra():
    nu = np.array([[[.2, .5, .8, 1.], [.1, .4, .9, 1.]],
                   [[.1, .6, .9, 1.], [.2, .3, .8, 1.]]])
    caps = nu == 1
    cost = np.full(nu.shape, np.inf)
    cost[~caps] = abs(np.log1p(-nu[~caps])-np.log(nu[~caps]))
    return dict(spectrum_cycles=np.array([20,40]), spectrum_seen=np.ones((2,2), dtype=bool),
                occupations=nu, cap_mask=caps, soft_mode_flip_costs=cost)


def test_exact_cycle_and_samplewise_minimum(remake):
    raw, gaps, nearest, counts = remake.gaps_at_cycle(fake_spectra(), 40)
    np.testing.assert_allclose(raw, [np.log(.6/.4), np.log(.7/.3)])
    np.testing.assert_allclose(gaps, raw/80)
    np.testing.assert_allclose(nearest, [.4,.3])
    np.testing.assert_array_equal(counts, [3,3])


def test_reject_unsaved_and_incomplete_cycle(remake):
    data = fake_spectra()
    for cycle in (0,30):
        with pytest.raises(ValueError):
            remake.gaps_at_cycle(data, cycle)
    data['spectrum_seen'][1,1] = False
    with pytest.raises(ValueError):
        remake.gaps_at_cycle(data,40)


def test_pure_caps_and_zero_mode(remake):
    raw, gaps, _, _ = remake.gaps_at_cycle(fake_spectra(),20)
    assert raw[0] == 0 and gaps[0] == 0
    assert np.isfinite(gaps).all()


def test_t60_uses_divisor_120(remake):
    data = fake_spectra()
    data['spectrum_cycles'] = np.array([20,60])
    raw, gaps, _, _ = remake.gaps_at_cycle(data,60)
    np.testing.assert_allclose(gaps, raw/120)
    np.testing.assert_allclose(raw, [np.log(.6/.4),np.log(.7/.3)])


def test_selected_sizes_reject_duplicates_and_unknown_values(remake):
    for sizes in ((30,30), (31,), ()):
        with pytest.raises(ValueError):
            remake.extract(cycle=60,sizes=sizes)


def test_fit_normalization_and_uncertainty(remake):
    x = np.array([20,24,30,36,44,56,60])
    mean = 2*x**-1.3
    sem = mean*.06
    f = remake.weighted_power_law(x,mean,sem)
    g = remake.weighted_power_law(x,80*mean,80*sem)
    assert f['exponent'] == pytest.approx(1.3)
    assert g['exponent'] == pytest.approx(f['exponent'])
    assert g['exponent_sem'] == pytest.approx(f['exponent_sem'])
    assert f['exponent_sem'] > 0  # not zeroed by a perfect residual fit
