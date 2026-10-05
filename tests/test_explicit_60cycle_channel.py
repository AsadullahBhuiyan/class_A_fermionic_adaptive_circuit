import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
FILE=ROOT/'00_WORKSPACE/CURRENT/matched_markov_lindblad_campaign/explicit_60cycle_v1/run_dynamics.py'
spec=importlib.util.spec_from_file_location('explicit_channel60',FILE)
runner=importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_fixed_grid():
    cases=[runner.config(a,n) for a in (1,3) for n in runner.SIZES]
    assert len(cases)==16
    assert all(c['cycles']==60 and c['init_mode']=='maxmix' and c['all_slabs_active']
               and c['sequence']=='raster_y' and c['perfect_correction']
               and c['dephasing'] and c['dtype']=='complex128' for c in cases)


def test_canonical_evolution_matches_independent_dense_sweep_and_observer_is_passive():
    cfg=runner.config(1,20)
    cfg.update(Nx=4,Ny=2,walls=[1,2])
    model=runner.spectral.make_model(cfg)
    actual=runner.collect(model,cycles=3,progress=False)
    n=2*model.Nx*model.Ny
    expected=np.eye(n,dtype=complex)/2
    increments=[]
    for _ in range(3):
        previous=expected.copy()
        for x in range(model.Nx):
            for y in range(model.Ny):
                for name in runner.spectral.ORDER:
                    w=getattr(model,'WF_'+name)[:,x,y]
                    w=w/np.linalg.norm(w)
                    p=np.outer(w,w.conj())
                    q=np.eye(n)-p
                    expected=q@expected@q+(p if name.endswith('m') else 0)
        increments.append(np.linalg.norm(expected-previous)/np.sqrt(n))
    np.testing.assert_allclose(actual['C_final'],expected,atol=1e-13)
    np.testing.assert_allclose(actual['successive_covariance_rms'][1:],increments,atol=1e-13)
    np.testing.assert_array_equal(actual['cycle'],np.arange(4))
    assert actual['global_charge'][0]==n/2
    bare=model.run_markov_channel(cycles=3,init_mode='maxmix',save=False,G_history=False,
         sequence='raster_y',decoh=True,perfect_correction=True,progress=False)
    np.testing.assert_array_equal(actual['C_final'],(bare['G_final']+np.eye(n))/2)


def test_publication_resume_checksum_and_partial_pairs(tmp_path):
    cfg=runner.config(1,20)
    cfg.update(Nx=4,Ny=2,walls=[1,2],cycles=3)
    arrays=runner.collect(runner.spectral.make_model(cfg),cycles=3,progress=False)
    hashes={'test':'testhash'}
    assert not runner.verified_complete(tmp_path,cfg,hashes)
    runner.publish(tmp_path,cfg,hashes,arrays,{}, {})
    assert runner.verified_complete(tmp_path,cfg,hashes) is True
    assert not runner.verified_complete(tmp_path,{**cfg,'cycles':60},hashes)
    assert not runner.verified_complete(tmp_path,cfg,{'test':'changed'})
    # Loss of the completion record makes an intact NPZ incomplete.
    (tmp_path/'completion.json').rename(tmp_path/'saved_completion.json')
    assert not runner.verified_complete(tmp_path,cfg,hashes)
    (tmp_path/'saved_completion.json').rename(tmp_path/'completion.json')
    with (tmp_path/'dynamics.npz').open('ab') as handle:
        handle.write(b'corrupt')
    assert not runner.verified_complete(tmp_path,cfg,hashes)
