import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
BUNDLE=ROOT/'00_WORKSPACE/CURRENT/final_production_new_designs/25_square_hard_wall_gap_pilot'
sys.path.insert(0,str(BUNDLE))


def load(name):
    spec=importlib.util.spec_from_file_location('square_gap_test_'+name,BUNDLE/(name+'.py'))
    mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
    return mod


R=load('run_campaign'); O=load('endpoint_spectrum')


def test_contract_and_workload():
    c=R.default_config(); tasks=R.tasks(c)
    assert [t.L for t in tasks]==[44,40,36,32,28,24,20]
    assert len(tasks)==7 and sum(t.samples for t in tasks)==70
    assert sum(t.samples//5 for t in tasks)==14 and len({t.seed for t in tasks})==7
    assert all(t.cycles==10 and t.samples==10 for t in tasks)
    assert [t.active_modes for t in reversed(tasks)]==[440,624,840,1088,1368,1680,2024]
    assert c['DW'] and c['dw_truncation'] and c['meas_slab_only'] and c['perfect_correction']
    assert c['alpha_1']==1 and c['alpha_2']==30 and c['nshell']==1
    assert c['init_mode']=='maxmix' and c['sequence']=='raster_y' and c['dtype']=='complex128'
    assert not c['postselect'] and not c['covariance_spectral_clip'] and c['spectrum_cycles']==[10]


def test_spectra_caps_and_factor_twenty():
    nu=np.array([[0.,.2,.49,.8,1.],[0.,0.,1.,1.,1.]])
    p=O.spectral_products(nu,10)
    expected=abs(np.log(.51/.49))
    np.testing.assert_allclose(p['modular_gap'][0],expected)
    np.testing.assert_allclose(p['lyapunov_gap'][0],expected/20)
    assert not p['finite_gap'][1] and np.isinf(p['modular_gap'][1])
    assert np.isposinf(p['modular_energies'][0,0]) and np.isneginf(p['modular_energies'][0,-1])
    q=O.spectral_products(np.array([[-1e-10,.5,1+1e-10]]),10)
    assert q['lyapunov_gap'][0]==0
    with pytest.raises(FloatingPointError):O.spectral_products(np.array([[-1e-6,.5]]),10)


def test_endpoint_against_original_cap_and_cost_convention():
    path=BUNDLE.parent/'13_maxmix_manybody_lyapunov_4ny/lyapunov_observer.py'
    spec=importlib.util.spec_from_file_location('original_gap_observer',path)
    original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
    rng=np.random.default_rng(123)
    u,_=np.linalg.qr(rng.normal(size=(12,12))+1j*rng.normal(size=(12,12)))
    G=(u*np.linspace(-1,1,12))@u.conj().T
    got=O.extract_endpoint(G[None],np.arange(12),10,device='cpu',progress=False)
    nu=np.linalg.eigvalsh((G+np.eye(12))/2)
    _,costs,caps=original.natural_spectrum_factors(nu)
    np.testing.assert_allclose(got['lyapunov_gap'][0],np.min(costs)/20,atol=1e-14)
    np.testing.assert_array_equal(got['cap_mask'][0],caps)


def test_cpu_backed_uninterrupted_vs_checkpoint_resume(tmp_path):
    torch.set_num_threads(1)
    t=R.Task(4,samples=2,cycles=10)
    c=R.default_config();ident=R.identity(t,c)
    model=R.build_model(t,'cpu')
    np.random.seed(t.seed);torch.manual_seed(t.seed)
    expected,expected_rng=R.run_segment(model,t,None,0,10,None)
    model=R.build_model(t,'cpu')
    np.random.seed(t.seed);torch.manual_seed(t.seed)
    prefix,rng=R.run_segment(model,t,None,0,5,None)
    R.save_checkpoint(tmp_path/'out',tmp_path/'scratch',t,ident,prefix,5,0.,rng)
    saved=R.load_checkpoint(tmp_path/'out',t,ident)
    assert saved is not None
    # Perturb RNG and reconstruct model, as happens in a fresh runtime.
    np.random.random(99);torch.rand(55)
    model=R.build_model(t,'cpu')
    final,final_rng=R.run_segment(model,t,saved['G'],5,5,saved)
    np.testing.assert_array_equal(final,expected)
    for key in expected_rng:np.testing.assert_array_equal(final_rng[key],expected_rng[key])
    active=model.active_top_layer_indices(True).numpy()
    a=O.extract_endpoint(expected,active,10,device='cpu',progress=False)
    b=O.extract_endpoint(final,active,10,device='cpu',progress=False)
    for key in a:np.testing.assert_array_equal(a[key],b[key])


def test_checkpoint_identity_partial_and_checksum_rejection(tmp_path):
    t=R.Task(4,samples=2);c=R.default_config();ident=R.identity(t,c)
    out,scratch=tmp_path/'out',tmp_path/'scratch'
    R.save_checkpoint(out,scratch,t,ident,np.zeros((2,32,32),complex),5,0.,R.capture_rng())
    assert R.load_checkpoint(out,t,ident) is not None
    assert R.load_checkpoint(out,t,ident|{'seed':-1}) is None
    path,receipt=R.checkpoint_paths(out,t)
    raw=receipt.read_bytes();receipt.unlink()
    assert R.load_checkpoint(out,t,ident) is None
    receipt.write_bytes(raw)
    with path.open('ab') as f:f.write(b'corrupt')
    assert R.load_checkpoint(out,t,ident) is None


def test_readback_failure_never_publishes_completion(tmp_path,monkeypatch):
    copy=R.shutil.copyfile
    def bad(source,target):
        copy(source,target)
        with Path(target).open('ab') as f:f.write(b'bad')
    monkeypatch.setattr(R.shutil,'copyfile',bad)
    path=tmp_path/'out/result.npz';receipt=path.with_suffix('.json')
    with pytest.raises(OSError,match='readback'):
        R.publish_pair({'a':np.arange(3)},path,receipt,{},tmp_path/'scratch',True)
    assert not receipt.exists() and not path.exists()


def test_endpoint_failure_resume_publication_and_cleanup_windows(tmp_path,monkeypatch):
    torch.set_num_threads(1)
    t=R.Task(4);c=R.default_config();ident=R.identity(t,c)
    out,scratch=tmp_path/'out',tmp_path/'scratch'
    extract=R.extract_endpoint
    def fail(*args,**kwargs):raise RuntimeError('interrupt endpoint')
    monkeypatch.setattr(R,'extract_endpoint',fail)
    with pytest.raises(RuntimeError,match='interrupt endpoint'):
        R.run_task(t,c,out,scratch,device='cpu')
    assert int(R.load_checkpoint(out,t,ident)['completed_cycle'])==10
    monkeypatch.setattr(R,'extract_endpoint',extract)
    monkeypatch.setattr(R,'run_segment',lambda *a,**k:pytest.fail('Dynamics must not repeat after T=10'))
    publish=R.publish_pair
    def fail_second(payload,path,receipt,expected,*args,**kwargs):
        if expected.get('sample_indices')==list(range(5,10)):
            raise OSError('interrupt second result')
        return publish(payload,path,receipt,expected,*args,**kwargs)
    monkeypatch.setattr(R,'publish_pair',fail_second)
    with pytest.raises(OSError,match='interrupt second result'):
        R.run_task(t,c,out,scratch,device='cpu')
    assert R.result_verified(out,t,0,ident) and not R.result_verified(out,t,5,ident)
    first,_=R.result_paths(out,t,0);first_hash=R.sha(first)
    with pytest.raises(RuntimeError,match='before every result'):
        R.cleanup_checkpoint(out,t,ident)
    monkeypatch.setattr(R,'publish_pair',publish)
    cleanup=R.cleanup_checkpoint
    monkeypatch.setattr(R,'cleanup_checkpoint',lambda *a:(_ for _ in ()).throw(RuntimeError('interrupt cleanup')))
    with pytest.raises(RuntimeError,match='interrupt cleanup'):
        R.run_task(t,c,out,scratch,device='cpu')
    assert R.result_verified(out,t,5,ident) and R.load_checkpoint(out,t,ident) is not None
    monkeypatch.setattr(R,'cleanup_checkpoint',cleanup)
    monkeypatch.setattr(R,'build_model',lambda *a:pytest.fail('Completed batch must skip model'))
    R.run_task(t,c,out,scratch,device='cpu')
    assert not any(p.exists() for p in R.checkpoint_paths(out,t))
    assert R.sha(first)==first_hash


def test_notebook_and_source_copies():
    import nbformat
    nb=nbformat.read(BUNDLE/'run_square_hard_wall_gap.ipynb',as_version=4);nbformat.validate(nb)
    source='\n'.join(c.source for c in nb.cells)
    for c in nb.cells:
        if c.cell_type=='code':compile(c.source,'<notebook>','exec')
    assert 'stdout.buffer' not in source and 'decoder.decode(chunk)' in source
    assert 'MAX_NEW_EXECUTION_BATCHES' in source and 'REPORT_ONLY' in source
    assert '/content/' in source and 'RUN_ANALYSIS' in source
    assert nb.cells[-1].source=="from google.colab import runtime\nruntime.unassign()\nprint('done')\n"
    for name in ('classA_U1FGTN_gpu.py','occupied_frame_gpu.py'):
        assert (BUNDLE/'src'/name).read_bytes()==(ROOT/'src/fgtn'/name).read_bytes()


def test_report_only_needs_no_gpu_or_output_directory(tmp_path, monkeypatch, capsys):
    out=tmp_path/'absent';scratch=tmp_path/'scratch'
    monkeypatch.setattr(sys,'argv',['run_campaign.py','--output-root',str(out),
                                    '--scratch-root',str(scratch),'--report-only'])
    monkeypatch.setattr(R.torch.cuda,'is_available',lambda:False)
    R.main()
    report=json.loads(capsys.readouterr().out)
    assert report['trajectories']==70 and report['pending']==14
    assert not out.exists() and not scratch.exists()


def test_analysis_uses_sample_gaps_and_sem_and_requires_complete_data(tmp_path,monkeypatch):
    A=load('analyze_campaign')
    for name in ('default_config','tasks','identity','result_paths','result_verified','sha'):
        monkeypatch.setattr(A,name,getattr(R,name))
    out=tmp_path/'out';dest=tmp_path/'analysis';config=R.default_config()
    with pytest.raises(RuntimeError,match='all 14'):
        A.analyze(out,dest)
    expected=np.linspace(.01,.1,10)
    for task in R.tasks(config):
        # Synthetic endpoint fixtures, not simulated production data.
        nu=np.zeros((10,task.active_modes))
        nu[:,0]=1/(1+np.exp(expected*20))
        products=O.spectral_products(nu,10)
        ident=R.identity(task,config)
        for start in (0,5):
            path,receipt=R.result_paths(out,task,start)
            payload={key:value[start:start+5] for key,value in products.items()}
            payload.update(sample_indices=np.arange(start,start+5),Nx=task.L,Ny=task.L,T=10)
            R.publish_pair(payload,path,receipt,dict(ident,kind='result',sample_indices=list(range(start,start+5))),
                           tmp_path/'scratch',True)
    A.analyze(out,dest)
    manifest=json.loads((dest/'analysis_manifest.json').read_text())
    for row in manifest['summary']:
        np.testing.assert_allclose(row['mean_lyapunov_gap'],expected.mean())
        np.testing.assert_allclose(row['sem_lyapunov_gap'],expected.std(ddof=1)/np.sqrt(10))
        np.testing.assert_allclose(row['mean_modular_gap'],expected.mean()*20)
    assert len(manifest['inputs'])==28 and len(manifest['outputs'])==6
    assert manifest['fitted_exponent'] is None
    path,_=R.result_paths(out,R.tasks(config)[0],0)
    with path.open('ab') as f:f.write(b'corrupt')
    assert not R.result_verified(out,R.tasks(config)[0],0,R.identity(R.tasks(config)[0],config))
    with pytest.raises(RuntimeError,match='all 14'):
        A.analyze(out,dest)
