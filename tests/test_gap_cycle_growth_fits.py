import importlib.util
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
FOLDER=ROOT/'00_WORKSPACE/CURRENT/experiment_review/purification_gap_cycle_convergence'
sys.path.insert(0,str(FOLDER))
spec=importlib.util.spec_from_file_location('cycle_growth_fit_test',FOLDER/'fit_cycle_growth.py')
F=importlib.util.module_from_spec(spec);spec.loader.exec_module(F)
sys.path.remove(str(FOLDER))


def test_affine_mean_first_and_paired_sampling_sem():
    rng=np.random.default_rng(17);t=np.arange(10,51,2)
    slopes=rng.normal(.2,.02,100);intercepts=rng.normal(-.7,.1,100)
    samples=slopes[:,None]*t+intercepts[:,None]
    row,influence=F.mean_first_fit(t,samples,10,50,'affine')
    np.testing.assert_allclose(row['coefficient'],slopes.mean(),atol=1e-14)
    np.testing.assert_allclose(row['intercept'],intercepts.mean(),atol=1e-14)
    np.testing.assert_allclose(row['coefficient_sem'],slopes.std(ddof=1)/10,atol=1e-14)
    np.testing.assert_allclose(row['intercept_sem'],intercepts.std(ddof=1)/10,atol=1e-14)
    np.testing.assert_allclose(influence[:,0],slopes-slopes.mean(),atol=1e-14)
    assert row['r2_raw']>1-1e-14


def test_correlated_power_amplitude_noise_does_not_create_exponent_noise():
    t=np.arange(4,81,4);amplitudes=np.linspace(1,2,100)
    samples=amplitudes[:,None]*t**1.4
    row,_=F.mean_first_fit(t,samples,20,80,'power')
    np.testing.assert_allclose(row['coefficient'],1.4,atol=1e-14)
    np.testing.assert_allclose(row['rate_exponent'],.4,atol=1e-14)
    np.testing.assert_allclose(row['amplitude'],amplitudes.mean(),atol=1e-13)
    assert row['coefficient_sem']<1e-14


def test_log_of_mean_not_mean_of_log_and_free_intercept():
    t=np.arange(10,81,2);exponents=np.linspace(.8,1.8,100)
    samples=(t[None,:]/10)**exponents[:,None]
    row,_=F.mean_first_fit(t,samples,10,80,'power')
    expected=np.polyfit(np.log(t),np.log(samples.mean(0)),1)[0]
    np.testing.assert_allclose(row['coefficient'],expected,atol=1e-13)
    assert abs(row['coefficient']-exponents.mean())>.01
    affine=np.tile(.2*t-1,(100,1))
    power,_=F.mean_first_fit(t,affine,10,80,'power')
    linear,_=F.mean_first_fit(t,affine,10,80,'affine')
    assert power['coefficient']>1
    np.testing.assert_allclose(linear['coefficient'],.2,atol=1e-14)
    np.testing.assert_allclose(linear['intercept'],-1,atol=1e-14)
