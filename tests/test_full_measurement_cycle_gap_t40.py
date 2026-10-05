import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
BUNDLE=ROOT/'00_WORKSPACE/CURRENT/final_production_new_designs/28_full_measurement_purification_gap_t40'


def load(name):
    spec=importlib.util.spec_from_file_location('full_cycle_test_'+name,BUNDLE/(name+'.py'))
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    return module


O=load('endpoint_spectrum')
previous=sys.modules.get('endpoint_spectrum')
sys.modules['endpoint_spectrum']=O
R=load('run_campaign')
if previous is None: del sys.modules['endpoint_spectrum']
else: sys.modules['endpoint_spectrum']=previous


def test_contract():
    c=R.default_config(); tasks=R.tasks(c)
    assert sorted({t.Ny for t in tasks})==[30,36,42,48,54,60]
    assert len(tasks)==18 and sum(t.samples for t in tasks)==600
    assert len({t.seed for t in tasks})==18
    assert sum(len(R.slots(t,c)) for t in tasks)==4200
    for ny in c['Ny_values']:
        ids=[i for t in tasks if t.Ny==ny for i in range(t.sample_start,t.sample_start+t.samples)]
        assert sorted(ids)==list(range(100))
    assert all(t.Nx==20 and t.cycles==40 and t.active_modes==40*t.Ny for t in tasks)
    assert c['DW'] and c['dw_truncation'] and c['perfect_correction']
    assert not c['meas_slab_only'] and not c['covariance_spectral_clip'] and not c['postselect']
    assert c['alpha_1']==1 and c['alpha_2']==30 and c['nshell']==1
    assert c['spectrum_cycles']==list(range(6,41)) and c['segment_cycles']==5
    assert c['init_mode']=='maxmix' and c['sequence']=='raster_y' and c['dtype']=='complex128'


def test_full_dynamics_one_cycle_resume_exact(tmp_path):
    torch.set_num_threads(1)
    t=R.Task(Ny=8,Nx=4,samples=2,cycles=40,sample_start=10)
    c=R.default_config(); ident=R.identity(t,c)
    model=R.build_model(t,'cpu')
    np.random.seed(t.seed);torch.manual_seed(t.seed)
    expected,expected_rng=R.run_segment(model,t,None,0,40,None)
    np.random.seed(t.seed);torch.manual_seed(t.seed)
    G=None;rng=None
    for completed in range(40):
        model=R.build_model(t,'cpu')
        G,rng=R.run_segment(model,t,G,completed,1,rng)
        R.save_checkpoint(tmp_path/'out',tmp_path/'scratch',t,ident,G,completed+1,0.,rng)
        checkpoint=R.load_checkpoint(tmp_path/'out',t,ident,segment=1)
        assert checkpoint is not None
        G=checkpoint['G'];rng=checkpoint
        # Observation leaves state/RNG unchanged, and later restore overrides incidental RNG use.
        if completed>=5:
            before=G.copy()
            O.extract_endpoint(G,np.arange(t.active_modes),completed+1,device='cpu',progress=False)
            np.testing.assert_array_equal(G,before)
        np.random.random(7);torch.rand(5)
    np.testing.assert_array_equal(G,expected)
    for key in expected_rng:np.testing.assert_array_equal(rng[key],expected_rng[key])
    assert R.load_checkpoint(tmp_path/'out',t,ident|{'seed':-1}) is None
    _,receipt=R.checkpoint_paths(tmp_path/'out',t);receipt.unlink()
    assert R.load_checkpoint(tmp_path/'out',t,ident) is None


def test_full_basis_and_time_normalization():
    rng=np.random.default_rng(112)
    q,_=np.linalg.qr(rng.normal(size=(8,8))+1j*rng.normal(size=(8,8)))
    nu=np.array([0,.1,.3,.49,.6,.7,.9,1.])
    G=((q*(2*nu-1))@q.conj().T)[None]
    for cycle in (6,20,40):
        p=O.extract_endpoint(G,np.arange(8),cycle,device='cpu',progress=False)
        np.testing.assert_allclose(p['occupation_spectrum_raw'][0],nu,atol=1e-14)
        np.testing.assert_allclose(p['lyapunov_gap'],p['modular_gap']/(2*cycle))
        n=int(p['finite_mode_count'][0]);assert n==6
        idx=p['finite_mode_indices'][0,:n];v=p['finite_mode_vectors'][0,:,:n]
        np.testing.assert_allclose(((G[0]+np.eye(8))/2)@v,v*nu[idx],atol=1e-14)
    pure=O.extract_endpoint(np.diag([-1.,1.]).astype(complex)[None],np.arange(2),6,device='cpu',progress=False)
    assert pure['finite_mode_count'][0]==0 and not pure['finite_gap'][0]
    with pytest.raises(FloatingPointError):
        O.spectral_products(np.array([[-1e-5,.5]]),6)


def tiny():
    return R.Task(Ny=8,Nx=4,samples=10,cycles=7,sample_start=20)


def test_mid_cycle_publication_and_completed_resume(tmp_path,monkeypatch):
    torch.set_num_threads(1)
    t=tiny(); c=R.default_config();ident=R.identity(t,c)
    out,scratch=tmp_path/'out',tmp_path/'scratch'
    publish=R.publish_pair
    def fail(payload,path,receipt,expected,*args,**kwargs):
        if expected.get('cycle')==6 and expected.get('sample_indices')==list(range(25,30)):
            raise OSError('interrupt second spectral shard')
        return publish(payload,path,receipt,expected,*args,**kwargs)
    monkeypatch.setattr(R,'publish_pair',fail)
    with pytest.raises(OSError,match='interrupt'):
        R.run_task(t,c,out,scratch,device='cpu')
    assert int(R.load_checkpoint(out,t,ident)['completed_cycle'])==5
    assert R.result_verified(out,t,0,ident,6)
    assert not R.result_verified(out,t,5,ident,6)
    first,_=R.result_paths(out,t,0,6);first_hash=R.sha(first)
    monkeypatch.setattr(R,'publish_pair',publish)
    original=R.run_segment;calls=[]
    def tracked(*args,**kwargs):
        calls.append(args[3])
        return original(*args,**kwargs)
    monkeypatch.setattr(R,'run_segment',tracked)
    R.run_task(t,c,out,scratch,device='cpu')
    assert calls==[5,6]  # replay from cycle five, skip already verified spectral shards
    assert R.sha(first)==first_hash
    assert all(R.result_verified(out,t,s,ident,cycle) for cycle,s in R.slots(t,c))
    assert not any(p.exists() for p in R.checkpoint_paths(out,t))
    monkeypatch.setattr(R,'build_model',lambda *a:pytest.fail('completed task must skip'))
    R.run_task(t,c,out,scratch,device='cpu')


def test_spectral_failure_keeps_current_state(tmp_path,monkeypatch):
    torch.set_num_threads(1)
    t=R.Task(Ny=8,Nx=4,samples=2,cycles=6)
    c=R.default_config();ident=R.identity(t,c)
    def fail(*a,**k):raise RuntimeError('spectral failure')
    monkeypatch.setattr(R,'extract_endpoint',fail)
    with pytest.raises(RuntimeError,match='spectral failure'):
        R.run_task(t,c,tmp_path/'out',tmp_path/'scratch',device='cpu')
    saved=R.load_checkpoint(tmp_path/'out',t,ident)
    assert int(saved['completed_cycle'])==6
    with pytest.raises(RuntimeError,match='before every'):
        R.cleanup_checkpoint(tmp_path/'out',t,ident)


def test_drive_readback_failure_no_receipt(tmp_path,monkeypatch):
    original=R.shutil.copyfile
    def broken(source,destination):
        original(source,destination)
        with Path(destination).open('ab') as stream:stream.write(b'bad')
    monkeypatch.setattr(R.shutil,'copyfile',broken)
    path=tmp_path/'out/result.npz';receipt=path.with_suffix('.json')
    with pytest.raises(OSError,match='readback'):
        R.publish_pair({'v':np.arange(3)},path,receipt,{},tmp_path/'scratch',True)
    assert not receipt.exists()


def test_corrupt_previous_cycle_replays_from_zero(tmp_path,monkeypatch):
    torch.set_num_threads(1)
    t=R.Task(Ny=8,Nx=4,samples=2,cycles=7)
    c=R.default_config();ident=R.identity(t,c)
    out,scratch=tmp_path/'out',tmp_path/'scratch'
    cleanup=R.cleanup_checkpoint
    monkeypatch.setattr(R,'cleanup_checkpoint',lambda *a:None)
    R.run_task(t,c,out,scratch,device='cpu')
    path,_=R.result_paths(out,t,0,6)
    with path.open('ab') as stream:stream.write(b'bad')
    assert not R.result_verified(out,t,0,ident,6)
    original=R.run_segment;calls=[]
    def track(*args,**kwargs):
        calls.append(args[3]);return original(*args,**kwargs)
    monkeypatch.setattr(R,'run_segment',track)
    monkeypatch.setattr(R,'cleanup_checkpoint',cleanup)
    R.run_task(t,c,out,scratch,device='cpu')
    assert calls==list(range(7))
    assert R.result_verified(out,t,0,ident,6)


def test_notebook_config_and_source_copies():
    import nbformat
    nb=nbformat.read(BUNDLE/'run_full_measurement_purification_gap_t40.ipynb',as_version=4)
    nbformat.validate(nb)
    source='\n'.join(c.source for c in nb.cells)
    assert 'stdout.buffer' not in source and 'decoder.decode(chunk)' in source
    for keyword in ('REPORT_ONLY','MAX_NEW_EXECUTION_BATCHES','RUN_ANALYSIS','/content/','A100'):
        assert keyword in source
    for cell in nb.cells:
        if cell.cell_type=='code':compile(cell.source,'<notebook>','exec')
    assert nb.cells[-1].source=="from google.colab import runtime\nruntime.unassign()\nprint('done')\n"
    assert json.loads((BUNDLE/'campaign_config.json').read_text())==R.default_config()
    for filename in ('classA_U1FGTN_gpu.py','occupied_frame_gpu.py'):
        assert (BUNDLE/'src'/filename).read_bytes()==(ROOT/'src/fgtn'/filename).read_bytes()


def test_report_only(tmp_path,monkeypatch,capsys):
    monkeypatch.setattr(sys,'argv',['runner','--output-root',str(tmp_path/'out'),
                                    '--scratch-root',str(tmp_path/'scratch'),'--report-only'])
    R.main()
    report=json.loads(capsys.readouterr().out)
    assert report['trajectories']==600 and report['pending']==4200 and report['execution_batches']==18
    assert not (tmp_path/'out').exists()


def test_analysis_sem_and_incomplete_rejection(tmp_path,monkeypatch):
    monkeypatch.setitem(sys.modules,'run_campaign',R)
    A=load('analyze_campaign')
    values=np.arange(100)/1000
    summary=A.summarize(values)
    np.testing.assert_allclose(summary['mean'],values.mean())
    np.testing.assert_allclose(summary['sem'],values.std(ddof=1)/10)
    missing=A.summarize([.1,np.inf])
    assert missing['mean'] is None and missing['finite_count']==1
    with pytest.raises(RuntimeError,match='4,200'):
        A.analyze(tmp_path/'absent',tmp_path/'analysis')
    assert not (tmp_path/'analysis').exists()


def test_batched_eigh_matches_single_and_leaves_rng_untouched(monkeypatch):
    rng=np.random.default_rng(80)
    q,_=np.linalg.qr(rng.normal(size=(5,8,8))+1j*rng.normal(size=(5,8,8)))
    nu=np.array([0,.1,.2,.49,.6,.8,.9,1])
    G=(q*(2*nu-1))@q.conj().transpose(0,2,1)
    original=torch.linalg.eigh; shapes=[]
    def observed(matrix):
        shapes.append(tuple(matrix.shape));return original(matrix)
    monkeypatch.setattr(torch.linalg,'eigh',observed)
    reference=O.extract_endpoint(G,np.arange(8),6,'cpu',False,1)
    before=torch.get_rng_state().clone();before_G=G.copy()
    for batch in (2,5):
        got=O.extract_endpoint(G,np.arange(8),6,'cpu',False,batch)
        np.testing.assert_allclose(got['occupation_spectrum_raw'],reference['occupation_spectrum_raw'],atol=1e-14)
        np.testing.assert_allclose(got['lyapunov_gap'],reference['lyapunov_gap'],atol=1e-14)
        for i,n in enumerate(got['finite_mode_count']):
            v=got['finite_mode_vectors'][i,:,:n];w=reference['finite_mode_vectors'][i,:,:n]
            np.testing.assert_allclose(v@v.conj().T,w@w.conj().T,atol=1e-13)
    assert (5,8,8) in shapes and (2,8,8) in shapes
    assert torch.equal(before,torch.get_rng_state())
    np.testing.assert_array_equal(G,before_G)


def test_spectral_batch_tuning_and_oom_fallback(monkeypatch):
    calls=[]
    def fake(G,active,cycle,device='cpu',progress=True,matrix_batch_size=1):
        calls.append(matrix_batch_size)
        if matrix_batch_size==5:raise torch.cuda.OutOfMemoryError('injected')
        return {'ok':True}
    monkeypatch.setattr(O,'extract_endpoint',fake)
    batcher=O.SpectrumBatcher()
    _,meta=batcher.extract(np.zeros((5,2,2)),np.arange(2),6,'cpu')
    assert meta['selected'] in (1,2)
    assert any(r['batch']==5 and r['status']=='oom' for r in meta['candidates'])
    batcher.cache[('cpu',2)]['selected']=5
    result,meta=batcher.extract(np.zeros((5,2,2)),np.arange(2),7,'cpu')
    assert meta['selected']==2 and result['ok']


def test_lossless_fast_compression_and_checkpoint_schedule(tmp_path,monkeypatch):
    payload={'modes':np.eye(8,dtype=complex)[None],'label':np.array('test'),'counts':np.arange(4)}
    path=tmp_path/'fast.npz';R.save_fast_npz(path,payload)
    with np.load(path,allow_pickle=False) as z:
        for key,value in payload.items():np.testing.assert_array_equal(z[key],value)
    t=R.Task(Ny=8,Nx=4,samples=2,cycles=11)
    save=R.save_checkpoint;cycles=[]
    def record(*args,**kwargs):
        cycles.append(args[5]);return save(*args,**kwargs)
    monkeypatch.setattr(R,'save_checkpoint',record)
    R.run_task(t,R.default_config(),tmp_path/'out',tmp_path/'scratch',device='cpu')
    assert cycles==[5,10,11]
