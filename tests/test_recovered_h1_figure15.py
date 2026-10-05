"""Regressions for the recovered-data Figure 15 adapter."""
import hashlib
import importlib.util
from pathlib import Path
import numpy as np
import pytest

PATH = Path(__file__).resolve().parents[1] / '00_WORKSPACE/CURRENT/experiment_review/h1_recovered_packet_review/build_figure15_comparison.py'
spec = importlib.util.spec_from_file_location('h1_figure15', PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_product_checksum_and_size():
    raw=b'packet product'
    record=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
    module.verify_bytes(raw,record)
    with pytest.raises(ValueError):module.verify_bytes(raw+b'x',record)
    with pytest.raises(ValueError):module.verify_bytes(b'x'*len(raw),record)


def fixture_profiles():
    p=np.zeros((2,2,2,161,20))
    fraction=np.linspace(0,.4,161)
    for e in range(2):
        start,target=(0,19) if e==0 else (19,0)
        p[:,:,e,:,start]=1-fraction
        p[:,:,e,:,target]=fraction
    c=np.repeat((p@np.arange(20))[:,None],40,axis=1)
    return c,p


def test_endpoint_selection_and_initial_subtraction():
    centers,profiles=fixture_profiles()
    dy=module.reduce_profiles(centers,profiles)
    assert dy.shape==(2,2,161)
    np.testing.assert_allclose(dy[...,0],0)
    np.testing.assert_allclose(dy[:,:,-1],[[7.6,-7.6]]*2)


def test_profile_normalization_and_center_must_agree():
    centers,profiles=fixture_profiles()
    with pytest.raises(AssertionError):module.reduce_profiles(centers,2*profiles)
    with pytest.raises(AssertionError):module.reduce_profiles(centers+1,profiles)
    profiles[0,0,0,0,0]=np.nan
    with pytest.raises(ValueError):module.reduce_profiles(centers,profiles)


def test_saved_cohorts_and_reduction():
    with np.load(module.OUT/'plotted_data.npz',allow_pickle=False) as data:
        for alpha,n in [(1,20),(3,25)]:
            np.testing.assert_array_equal(data[f'alpha{alpha}_ids'],np.arange(n))
            profiles=data[f'alpha{alpha}_profiles']
            np.testing.assert_allclose(profiles.sum(-1),1,atol=1e-12)
            centers=profiles@np.arange(20)
            expected=np.stack([centers[:,w,e]-centers[:,w,e,:1] for w,e in module.PACKETS],axis=1)
            np.testing.assert_allclose(data[f'alpha{alpha}_dy'],expected,atol=3e-12)
