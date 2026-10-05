"""Opt-in spectral stabilization and explicit read-only cycle-30 continuation."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT/'00_WORKSPACE/CURRENT/final_production_new_designs/22_hard_wall_full_measurement_clipped'


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


OBS = load(BUNDLE/'purification_observer.py', '_clip22_obs')
previous = sys.modules.get('purification_observer')
sys.modules['purification_observer'] = OBS
RUN = load(BUNDLE/'run_campaign.py', '_clip22_runner')
if previous is None: sys.modules.pop('purification_observer', None)
else: sys.modules['purification_observer'] = previous


def model():
    return RUN.classA_U1FGTN_gpu(Nx=4, Ny=4, DW=True, nshell=1, alpha_1=3,
        alpha_2=30, dw_truncation=True, device='cpu', dtype='complex128', backend='local')


def task(cycles=4):
    return SimpleNamespace(nx=4, ny=4, samples=2, cycles=cycles, alpha_1=3.,
        construction='hard', sample_indices=np.arange(2), seed=713,
        sample_start=0, sample_stop=2, task_id='test_alpha3')


def observer(t):
    return OBS.PurificationObserver(nx=t.nx, ny=t.ny, cycles=t.cycles,
        sample_indices=t.sample_indices, sample_chunk=2, construction='hard', save_contours=False)


class Bar:
    def update(self, n): pass


def test_projection_preserves_interior_and_reports_corrections():
    m=model()
    occupations=np.array([-4e-9, .1, .25, .45, .6, .8, .95, 1+4e-9])
    generator=np.random.default_rng(2)
    q,_=np.linalg.qr(generator.normal(size=(8,8))+1j*generator.normal(size=(8,8)))
    original=q@np.diag(2*occupations-1)@q.conj().T
    g=torch.tensor(original[None],dtype=torch.complex128)
    rng=torch.get_rng_state().clone()
    d=m.clip_covariance_spectrum(g, sample_chunk=1)
    values=(torch.linalg.eigvalsh(g)[0].numpy()+1)/2
    np.testing.assert_allclose(values, np.clip(occupations,0,1), atol=2e-15)
    assert d['clip_mode_count'][0]==2
    assert d['clip_max_correction'][0]==pytest.approx(4e-9,abs=1e-15)
    assert d['clip_covariance_frobenius'][0]==pytest.approx(np.linalg.norm(g.numpy()[0]-original),abs=1e-15)
    assert torch.equal(rng,torch.get_rng_state())
    physical=torch.diag(torch.linspace(-.8,.8,8)).to(torch.complex128)[None]
    old=physical.clone();m.clip_covariance_spectrum(physical)
    assert torch.equal(old,physical)


@pytest.mark.parametrize('kind',['large','nan','nonhermitian'])
def test_projection_rejects_bad_states(kind):
    g=torch.zeros((1,8,8),dtype=torch.complex128)
    if kind=='large': g[0,0,0]=1.01
    if kind=='nan': g[0,0,0]=float('nan')
    if kind=='nonhermitian': g[0,0,1]=.1
    with pytest.raises(FloatingPointError): model().clip_covariance_spectrum(g)


def test_clipped_checkpoint_resume_exact(tmp_path, monkeypatch):
    monkeypatch.setattr(RUN,'SEGMENT_CYCLES',2)
    t=task();cfg={**RUN.expected_config(),'device':'cpu'}
    torch.manual_seed(713);np.random.seed(713)
    a=observer(t)
    full=RUN.run_segment(model(),cfg,t,a,completed_cycle=0,segment_cycles=4,
        G_init=None,continuing=False,progress_bar=Bar())
    rng=RUN.capture_rng();a.validate(completed_cycle=4,final=True)
    torch.manual_seed(713);np.random.seed(713)
    b=observer(t)
    half=RUN.run_segment(model(),cfg,t,b,completed_cycle=0,segment_cycles=2,
        G_init=None,continuing=False,progress_bar=Bar())
    RUN.save_checkpoint(tmp_path/'o',tmp_path/'s',t,completed_cycle=2,elapsed_seconds=1,
        G=half,observer=b,cfg_hash='cfg',hashes={})
    cp,status=RUN.load_checkpoint(tmp_path/'o',t,cfg_hash='cfg',hashes={})
    assert status=='verified'
    c=observer(t);c.restore_checkpoint(cp.observer_payload,completed_cycle=2)
    m=model();RUN.restore_rng(cp.rng_payload)
    resumed=RUN.run_segment(m,cfg,t,c,completed_cycle=2,segment_cycles=2,
        G_init=cp.G,continuing=True,progress_bar=Bar())
    np.testing.assert_array_equal(full,resumed)
    for name in a.ARRAY_NAMES: np.testing.assert_array_equal(getattr(a,name),getattr(c,name))
    for name in rng: np.testing.assert_array_equal(rng[name],RUN.capture_rng()[name])


def test_default_flag_no_extra_eigensolver(monkeypatch):
    m=model();monkeypatch.setattr(m,'clip_covariance_spectrum',lambda *a,**k:pytest.fail('default clipped'))
    m.run_markov_circuit(cycles=1,samples=1,batch_size=1,save=False,G_history=False,
        init_mode='maxmix',state_representation='covariance',perfect_correction=True,
        meas_slab_only=False,progress=False)


def test_clipping_happens_before_observer_and_continuation(monkeypatch):
    m=model();actual=m.clip_covariance_spectrum;events=[]
    def clip(g,**kwargs):
        g[0]=torch.eye(32,dtype=g.dtype)*(1+8.6e-9)
        d=actual(g,**kwargs);events.append('clip');return d
    monkeypatch.setattr(m,'clip_covariance_spectrum',clip)
    def inspect(*,cycle,G,**kwargs):
        if cycle:
            assert events[-1]=='clip'
            assert float(torch.linalg.eigvalsh(G).max())<=1+1e-12
            events.append('observe')
    result=m.run_markov_circuit(cycles=2,samples=1,batch_size=1,save=False,G_history=False,
        init_mode='maxmix',state_representation='covariance',perfect_correction=True,
        meas_slab_only=False,progress=False,covariance_spectral_clip=True,cycle_observer=inspect)
    assert events==['clip','observe','clip','observe']
    assert result['covariance_spectral_clip']


def test_bundle_contract_and_notebook():
    cfg=RUN.expected_config();tasks=RUN.expand_execution_batches(cfg,'hard')
    assert len(tasks)==1 and tasks[0].alpha_1==3 and tasks[0].cycles==60
    assert tasks[0].seed==4223483256 and tasks[0].samples==100
    assert len(RUN.all_result_shards(cfg,'hard'))==20
    assert cfg['unclipped_prefix_cycles']==30 and cfg['covariance_spectral_clip']
    assert not RUN.source_identity_matches({'a':'old'},{'a':'new'})
    nb=json.loads((BUNDLE/'run_hard_wall_full_measurement_clipped.ipynb').read_text())
    for cell in nb['cells']:
        if cell['cell_type']=='code': ast.parse(''.join(cell['source']))
    text=''.join(''.join(c['source']) for c in nb['cells'])
    for word in ['V1_OUTPUT_ROOT','--fork-from-v1','REPORT_ONLY','A100','runtime.unassign()']: assert word in text
    for name,row in json.loads((BUNDLE/'deployment_manifest.json').read_text())['files'].items():
        raw=(BUNDLE/name).read_bytes()
        assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
    assert (BUNDLE/'src/classA_U1FGTN_gpu.py').read_bytes()==(ROOT/'src/fgtn/classA_U1FGTN_gpu.py').read_bytes()


def test_fork_checksum_rejection_and_no_source_write(tmp_path):
    t=task();source=tmp_path/'v1';dest=tmp_path/'v2'
    with pytest.raises(ValueError,match='overwrite'):
        RUN.fork_v1_checkpoint(RUN.expected_config(),source,source,tmp_path/'scratch',t,cfg_hash='x',hashes={})
    npz,receipt=RUN.checkpoint_paths(source,t);receipt.parent.mkdir(parents=True)
    receipt.write_text('{}');npz.write_bytes(b'not a checkpoint')
    with pytest.raises(ValueError,match='identity'):
        RUN.fork_v1_checkpoint(RUN.expected_config(),source,dest,tmp_path/'scratch',t,cfg_hash='x',hashes={})
    assert receipt.read_text()=='{}' and npz.read_bytes()==b'not a checkpoint'
    assert not dest.exists()


def test_verified_fork_keeps_prefix_rng_and_original_bytes(tmp_path, monkeypatch):
    t=task(cycles=40);obs=observer(t)
    for name in obs.COVARIANCE_ARRAY_NAMES:
        getattr(obs,name)[:,:31]=0.
    obs.occupation_spectrum[:,:31]=.5
    obs.total_charge[:,:31]=16.
    obs.total_entropy[:,:31]=32*np.log(2)
    obs.total_charge_variance[:,:31]=8.
    obs.seen[:31]=True
    obs.measurement_log_probability[:,:31]=0.
    obs.cumulative_log_probability[:,:31]=0.
    obs.site_event_count[:,1:31]=16;obs.channel_event_count[:,1:31]=64
    old=obs.checkpoint_payload()
    old={k:v for k,v in old.items() if not k.startswith('observer_clip_')}
    old['observer_schema']=np.asarray('full_measurement_purification_observer_v1')
    rng=RUN.capture_rng()
    G=np.broadcast_to(np.eye(32,dtype=np.complex128)*(1+3.92e-10),(2,32,32)).copy()
    source=tmp_path/'v1';out=tmp_path/'v2';scratch=tmp_path/'scratch'
    path,receipt=RUN.checkpoint_paths(source,t)
    RUN.save_npz(path,dict(checkpoint_schema=np.asarray('full_measurement_purification_checkpoint_v1'),
        completed_cycle=np.asarray(30),elapsed_seconds=np.asarray(10.),sample_indices=t.sample_indices,
        G=G,**rng,**old))
    sources={
        'run_campaign.py':'8f6f5958ad466adbe3e8298c98d1acd906901b12e7e024c4aff0aacd341e3899',
        'purification_observer.py':'16400cc51ea4a074ff4ac64c7ef743fea3b024ba8d942f6ec338f5671f42e472',
        'src/classA_U1FGTN_gpu.py':'53de96bced6839b485afe04fe4aaa15d2e42c249a9cf14f6ca0c931f55409700',
        'src/occupied_frame_gpu.py':'bfc10cefea98ce00184a88b5c375eadfcda8e3dc6954d9f66183d566008951b0'}
    meta=RUN._checkpoint_identity(t,cfg_hash='41088ca86a49cc74a5046848e83101302a9889bafd8a9a8249b278fd20f456a0',hashes=sources)
    meta.update(schema='full_measurement_purification_checkpoint_v1',
        sampling_revision='hard_wall_full_measurement_nx20_ny30_alpha1-3_s100_2ny_v1',completed_cycle=30,
        checkpoint_filename=path.name,checkpoint_bytes=path.stat().st_size,checkpoint_sha256=RUN.sha256_file(path))
    RUN.write_json(receipt,meta)
    before=(path.read_bytes(),receipt.read_bytes())
    monkeypatch.setattr(RUN,'build_model',lambda *args:model())
    cfg={**RUN.expected_config(),'device':'cpu'}
    RUN.fork_v1_checkpoint(cfg,source,out,scratch,t,cfg_hash='cfg',hashes={})
    assert before==(path.read_bytes(),receipt.read_bytes())
    cp,status=RUN.load_checkpoint(out,t,cfg_hash='cfg',hashes={})
    assert status=='verified' and cp.completed_cycle==30
    for name in rng: np.testing.assert_array_equal(cp.rng_payload[name],rng[name])
    np.testing.assert_array_equal(cp.observer_payload['observer_occupation_spectrum'],obs.occupation_spectrum)
    np.testing.assert_allclose(np.linalg.eigvalsh(cp.G),1.,atol=1e-14)
    p=json.loads((out/'fork_provenance.json').read_text())
    assert p['source_checkpoint_sha256']==meta['checkpoint_sha256']
    assert p['handoff_correction']['clip_max_correction'][0]==pytest.approx(1.96e-10,abs=1e-15)
    np.testing.assert_array_equal(cp.observer_payload['observer_clip_max_correction'][:,:31],0.)


def test_new_report_is_read_only(tmp_path):
    out=tmp_path/'v2'
    report=RUN.run_campaign(RUN.expected_config(),construction='hard',output_root=out,
        scratch_root=tmp_path/'scratch',report_only=True,max_new_execution_batches=None,
        fork_from_v1=tmp_path/'v1')
    assert report['pending_shards']==20 and report['execution_batches']==1
    assert not out.exists()
