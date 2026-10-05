import importlib.util
from pathlib import Path
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
PATH=ROOT/'00_WORKSPACE/CURRENT/experiment_review/purification_gap_cycle_convergence/analyze.py'
spec=importlib.util.spec_from_file_location('cycle_gap_analysis_test',PATH)
A=importlib.util.module_from_spec(spec);spec.loader.exec_module(A)


def test_time_normalization_and_caps():
    cycles=np.array([10,40])
    raw=np.array([[.4,1.6],[.6,2.4]])
    nu=np.zeros((2,2,3));nu[:,:,1]=1/(1+np.exp(raw));nu[:,:,2]=1
    g,d=A.gaps(nu,cycles)
    np.testing.assert_allclose(g,raw,atol=1e-14)
    np.testing.assert_allclose(d,[[.02,.02],[.03,.03]],atol=1e-14)
    assert not np.allclose(A.gaps(nu,cycles)[1],g/cycles)
    with pytest.raises(ValueError):A.gaps(nu,np.array([0,40]))
    with pytest.raises(ValueError):A.gaps(np.ones((2,2,3)),cycles)


def test_gap_is_taken_before_ensemble_average():
    nu=np.array([[[.4,.9]],[[.1,.6]]])
    raw,_=A.gaps(nu,np.array([40]))
    np.testing.assert_allclose(raw,np.log(1.5))
    averaged,_=A.gaps(nu.mean(0,keepdims=True),np.array([40]))
    assert not np.isclose(averaged[0,0],raw.mean())


def test_paired_sem_respects_temporal_covariance():
    x=np.linspace(.01,.1,100)
    values=np.column_stack([x,2*x])
    change=A.paired_change(values,np.array([20,40]),20,40)
    np.testing.assert_allclose(change['change'],x.mean())
    np.testing.assert_allclose(change['change_sem'],x.std(ddof=1)/10)
    np.testing.assert_allclose(change['percent_change'],100)
    np.testing.assert_allclose(change['percent_change_sem'],0,atol=1e-13)
    unpaired=np.sqrt(x.var(ddof=1)+(2*x).var(ddof=1))/10
    assert change['change_sem']<unpaired


def test_saved_coverage_and_normalization():
    import json
    manifest=json.loads((PATH.parent/'analysis_manifest.json').read_text())
    assert len(manifest['inputs'])==320 and not manifest['campaign28_data_used']
    with np.load(PATH.parent/'sample_cycle_gaps.npz',allow_pickle=False) as z:
        for protocol,sizes in [('full_measurement',[30]),('slab_only',[20,24,30,36,44,56,60])]:
            for ny in sizes:
                stem=f'{protocol}_Ny{ny}_'
                t=z[stem+'cycles'];g=z[stem+'raw'];d=z[stem+'rate']
                assert g.shape==d.shape==(100,len(t))
                np.testing.assert_array_equal(d,g/(2*t))
                assert np.isfinite(d).all()

