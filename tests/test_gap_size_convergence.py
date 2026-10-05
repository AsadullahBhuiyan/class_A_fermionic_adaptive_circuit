import importlib.util
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
HERE=ROOT/'00_WORKSPACE/CURRENT/experiment_review/purification_gap_cycle_convergence'
def load(name):
    spec=importlib.util.spec_from_file_location('test_size_'+name,HERE/(name+'.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
source=load('analyze')
old=sys.modules.get('analyze');sys.modules['analyze']=source
C=load('compare_sizes')
if old is None:del sys.modules['analyze']
else:sys.modules['analyze']=old


def test_reference_ratio_retains_pairing_and_exact_endpoint():
    scale=np.linspace(.5,1.5,100)
    samples=scale[:,None]*np.array([.4,.7,1.])[None,:]
    ratio,sem=C.relative_curve(samples)
    np.testing.assert_allclose(ratio,[.4,.7,1],atol=1e-14)
    np.testing.assert_allclose(sem,0,atol=1e-14)
    # Mean of per-trajectory ratios is a different estimator.
    varied=np.array([[1.,2.],[4.,8.],[1.,4.]])
    got,_=C.relative_curve(varied)
    assert np.isclose(got[0],6/14)
    assert not np.isclose(got[0],np.mean(varied[:,0]/varied[:,-1]))


def test_stable_threshold_requires_all_later_points():
    times=np.array([4,8,12,16,20])
    ratio=np.array([.8,.95,.85,.94,1.])
    assert C.first_stable_reference_time(times,ratio)==16
    assert C.first_stable_reference_time(times,np.array([.8,.95,1.2,.94,1.]))==16


def test_saved_products_and_size_coverage():
    import json
    m=json.loads((HERE/'size_comparison_v1/analysis_manifest.json').read_text())
    assert len(m['inputs'])==280
    assert m['Ny_values']==[20,24,30,36,44,56,60]
    assert m['fitted_dynamic_exponent'] is None
    assert not m['full_measurement_data_included']
    for row in m['summary']:
        assert 2.7<row['t90_over_Ny']<3.2
        assert row['samples']==100
        assert row['t3Ny_percent_of_reference']>row['t2Ny_percent_of_reference']
        assert row['t2Ny_percent_sem']>0

