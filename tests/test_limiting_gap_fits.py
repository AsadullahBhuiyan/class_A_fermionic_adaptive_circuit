from pathlib import Path
import importlib.util
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
HERE=ROOT/'00_WORKSPACE/CURRENT/matched_markov_lindblad_campaign/fixed_width_alpha_spectral_v1'
sys.path.insert(0,str(HERE))
spec=importlib.util.spec_from_file_location('limiting_gap_fits',HERE/'fit_limiting_gap.py')
fit=importlib.util.module_from_spec(spec)
spec.loader.exec_module(fit)


def test_recover_plateau():
    n=np.array([20,40,60,80,100.])
    result=fit.plateau_fit(n,1.7+2.3/n)
    np.testing.assert_allclose(result['coefficients'],[1.7,2.3],atol=1e-12)


def test_recover_closing_power():
    n=np.array([20,40,60,80,100.])
    result=fit.closing_fit(n,2.5*(n/20)**(-.6))
    np.testing.assert_allclose([result['amplitude_at_N20'],result['exponent']],[2.5,.6],atol=1e-7)
    assert result['rmse']<1e-8


def test_holdout_does_not_train_on_final_point():
    n=np.array([20,40,60,80,100.])
    y=1.7+2.3/n
    first=fit.analyze_curve(n,y)
    y[-1]+=.2
    second=fit.analyze_curve(n,y)
    for key in ('plateau_prediction','closing_prediction'):
        assert first['holdout'][key]==second['holdout'][key]
    np.testing.assert_allclose(second['holdout']['plateau_error'],-.2,atol=1e-12)
