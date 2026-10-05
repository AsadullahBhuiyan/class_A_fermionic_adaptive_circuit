import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
BUNDLE=ROOT/'00_WORKSPACE/CURRENT/final_production_new_designs/31_hard_wall_alpha_endpoint_ny30'
sys.path.insert(0,str(BUNDLE))


def load(name):
    spec=importlib.util.spec_from_file_location('alpha31_test_'+name,BUNDLE/(name+'.py'))
    mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
    return mod


O=load('endpoint_spectrum')
_previous_observer=sys.modules.get('endpoint_spectrum')
sys.modules['endpoint_spectrum']=O
R=load('run_campaign')
if _previous_observer is None:
    del sys.modules['endpoint_spectrum']
else:
    sys.modules['endpoint_spectrum']=_previous_observer


def test_grid_lanes_contract():
    c=R.default_config(); all_tasks=R.tasks(c)
    assert len(all_tasks)==21 and sum(t.samples for t in all_tasks)==210
    assert len({t.seed for t in all_tasks})==21
    assert len({(t.name,i) for t in all_tasks for i in range(10)})==210
    expected={'A':[1,1.6,1.85,1.975,2.05,2.2,2.6],
              'B':[1.2,1.7,1.9,2,2.1,2.3,2.8],'C':[1.4,1.8,1.95,2.025,2.15,2.4,3]}
    names=[]
    for lane,values in expected.items():
        ts=R.tasks(c,lane)
        assert sorted(t.alpha_1 for t in ts)==values
        assert ts==sorted(ts,key=lambda t:(round(abs(t.alpha_1-2),12),t.alpha_1))
        names.extend(t.name for t in ts)
        assert all(t.seed==next(a.seed for a in all_tasks if a.name==t.name) for t in ts)
    assert len(set(names))==21
    assert all((t.Nx,t.Ny,t.cycles,t.samples,t.active_modes)==(20,30,60,10,660) for t in all_tasks)
    assert c['alpha_2']==30 and c['nshell']==1 and c['meas_slab_only']
    assert c['perfect_correction'] and not c['postselect'] and not c['covariance_spectral_clip']
    assert c['gpu_allocator_limit_bytes']==38*1024**3
    assert c['segment_cycles']==10 and c['spectrum_cycles']==[60]
    paths=[]
    for lane in 'ABC':
        for task in R.tasks(c,lane):
            paths.extend(R.checkpoint_paths(Path('/example'),task))
            for start in (0,5):paths.extend(R.result_paths(Path('/example'),task,start))
    assert len(paths)==len(set(paths))


def test_caps_placeholder_and_finite_100():
    a=np.array([[-1,-.4,.1,.7,1],[-1,-1,1,1,1.]])
    p=O.spectral_products(a,60)
    assert p['gap_value'][0]==abs(np.arctanh(.1))/60
    assert p['gap_raw'][1]==np.inf and p['gap_value'][1]==100
    assert p['gap_is_infinite'].tolist()==[False,True]
    assert p['gap_mode_index'].tolist()==[2,-1]
    q=O.gap_fields(np.array([[100.,120.],[np.inf,-np.inf]]))
    assert q['gap_value'].tolist()==[100,100] and q['gap_is_infinite'].tolist()==[False,True]
    # Distinguishes centered 1e-9 from occupation 1e-9 capping.
    q=O.spectral_products(np.array([[1-1.5e-9,1-0.5e-9]]),60)
    assert q['cap_mask'].tolist()==[[False,True]]
    with pytest.raises(FloatingPointError):O.spectral_products(np.array([[-1.0001,0.]]),60)
    with pytest.raises(ValueError):O.spectral_products(np.array([[np.nan,0.]]),60)


def test_modes_residuals_degeneracies_and_observer_invariance():
    rng=np.random.default_rng(91)
    u,_=np.linalg.qr(rng.normal(size=(6,6))+1j*rng.normal(size=(6,6)))
    values=np.array([[-1,-.6,-.1,.1,.7,1],[-1,-1,-1,1,1,1],[-1,-.5,0,0,.5,1]])
    G=np.array([(u*a)@u.conj().T for a in values],np.complex128)
    before=G.copy(); state=R.capture_rng()
    p=O.extract_endpoint(G,np.arange(6),60,device='cpu',progress=False)
    assert p['gap_tie_count'].tolist()==[2,0,2]
    assert p['gap_mode_index'].tolist()==[2,-1,2]
    for i in (0,2):
        v=p['gap_mode_vector'][i]; j=p['gap_mode_index'][i]
        np.testing.assert_allclose(G[i]@v,p['centered_spectrum_raw'][i,j]*v,atol=1e-13)
        assert abs(v[np.argmax(abs(v))].imag)<1e-12
    assert not p['gap_mode_vector'][1].any() and not p['gap_mode_valid'][1]
    np.testing.assert_array_equal(G,before)
    for k,v in state.items():np.testing.assert_array_equal(v,R.capture_rng()[k])
    bad=dict(p);bad['gap_value']=p['gap_value'].copy();bad['gap_value'][1]=0
    with pytest.raises(AssertionError):O.validate_products(bad,60)


@pytest.mark.parametrize('near_caps_only',[False,True])
def test_unsorted_solver_pairs_are_sorted_together(monkeypatch,near_caps_only):
    rng=np.random.default_rng(807)
    u,_=np.linalg.qr(rng.normal(size=(8,8))+1j*rng.normal(size=(8,8)))
    a=np.array([-1.,-1+3e-15,-.7,-.2,.01,.8,1-3e-15,1.])
    G=np.array([(u*a)@u.conj().T],np.complex128)
    expected=O.extract_endpoint(G,np.arange(8),60,device='cpu',progress=False)
    original=O.torch.linalg.eigh
    permutation=[1,0,2,3,4,5,7,6] if near_caps_only else [7,4,0,6,2,1,5,3]
    def scrambled(block):
        values,vectors=original(block)
        return values[permutation],vectors[:,permutation]
    monkeypatch.setattr(O.torch.linalg,'eigh',scrambled)
    got=O.extract_endpoint(G,np.arange(8),60,device='cpu',progress=False)
    for k in expected:np.testing.assert_allclose(got[k],expected[k],atol=1e-14,rtol=0)
    v=got['gap_mode_vector'][0];j=got['gap_mode_index'][0]
    np.testing.assert_allclose(G[0]@v,got['centered_spectrum_raw'][0,j]*v,atol=1e-13)


def test_only_pinned_historical_sources_are_accepted():
    current=R.identity(R.Task(),R.default_config())['source_hashes']
    old=R.PRE_SORT_SOURCE_HASHES.copy()
    assert R.compatible_sources(old,current)
    assert R.compatible_sources(current,current)
    for key in old:
        assert not R.compatible_sources(old|{key:'unknown'},current)
    assert not R.compatible_sources(old,current|{'src/classA_U1FGTN_gpu.py':'changed-engine'})


def test_notebooks_sources_manifest_and_registration():
    import nbformat
    for lane in 'ABC':
        nb=nbformat.read(BUNDLE/f'run_alpha_endpoint_lane_{lane}.ipynb',as_version=4)
        nbformat.validate(nb)
        source='\n'.join(c.source for c in nb.cells)
        for c in nb.cells:
            if c.cell_type=='code':compile(c.source,'<notebook>','exec')
        assert f"LANE = '{lane}'" in source and '--lane' in source
        assert 'stderr=subprocess.STDOUT' in source and 'decoder.decode(chunk)' in source
        assert 'bufsize=0' in source and 'process.terminate()' in source
        assert 'MAX_NEW_EXECUTION_BATCHES = None' in source and 'REPORT_ONLY' in source
        assert source.count("drive.mount('/content/drive')")==1
        assert nb.cells[-1].source=="from google.colab import runtime\nruntime.unassign()\nprint('done')\n"
    for name in ('classA_U1FGTN_gpu.py','occupied_frame_gpu.py'):
        assert (BUNDLE/'src'/name).read_bytes()==(ROOT/'src/fgtn'/name).read_bytes()
    manifest=json.loads((BUNDLE/'deployment_manifest.json').read_text())
    for name,record in manifest['files'].items():
        assert R.sha(BUNDLE/name)==record['sha256']
        assert (BUNDLE/name).stat().st_size==record['bytes']
    spec=importlib.util.spec_from_file_location('alpha31_layout',BUNDLE.parent/'bundle_layout.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    assert BUNDLE.name in m.validate_bundle_layout(BUNDLE.parent)


def load_analysis():
    previous=sys.modules.get('run_campaign');sys.modules['run_campaign']=R
    try:return load('analyze_campaign')
    finally:
        if previous is None:sys.modules.pop('run_campaign',None)
        else:sys.modules['run_campaign']=previous


def test_analysis_never_averages_placeholder():
    A=load_analysis()
    r=A.summarize(2,[.1,.2,np.inf],[False,False,True])
    assert r['statistic']=='finite_subset' and r['finite_samples']==2
    np.testing.assert_allclose(r['mean'],.15);np.testing.assert_allclose(r['sem'],.05)
    r=A.summarize(2,[100,100],[False,False])
    assert r['mean']==100 and r['statistic']=='full_ensemble'
    r=A.summarize(2,[np.inf]*10,[True]*10)
    assert r['mean'] is None and r['sem'] is None and r['infinite_fraction']==1
    assert A.summarize(2,[.1,np.inf],[False,True])['sem'] is None


@pytest.mark.parametrize('boundary', [10,20,30,40,50])
def test_cpu_backed_uninterrupted_vs_checkpoint_resume(tmp_path, boundary):
    torch.set_num_threads(1)
    t=R.Task(Ny=8,Nx=4,samples=2,cycles=60)
    c=R.default_config();ident=R.identity(t,c)
    model=R.build_model(t,'cpu')
    np.random.seed(t.seed);torch.manual_seed(t.seed)
    expected,expected_rng=R.run_segment(model,t,None,0,60,None)
    model=R.build_model(t,'cpu')
    np.random.seed(t.seed);torch.manual_seed(t.seed)
    prefix,rng=R.run_segment(model,t,None,0,boundary,None)
    # A pre-hotfix checkpoint must survive the endpoint-only code upgrade.
    old_ident=ident|{'source_hashes':R.PRE_SORT_SOURCE_HASHES.copy()}
    R.save_checkpoint(tmp_path/'out',tmp_path/'scratch',t,old_ident,prefix,boundary,0.,rng)
    saved=R.load_checkpoint(tmp_path/'out',t,ident)
    assert saved is not None
    # Perturb RNG and reconstruct model, as happens in a fresh runtime.
    np.random.random(99);torch.rand(55)
    model=R.build_model(t,'cpu')
    final,final_rng=R.run_segment(model,t,saved['G'],boundary,60-boundary,saved)
    np.testing.assert_array_equal(final,expected)
    for key in expected_rng:np.testing.assert_array_equal(final_rng[key],expected_rng[key])
    active=model.active_top_layer_indices(True).numpy()
    a=O.extract_endpoint(expected,active,60,device='cpu',progress=False)
    b=O.extract_endpoint(final,active,60,device='cpu',progress=False)
    for key in a:np.testing.assert_array_equal(a[key],b[key])


def test_checkpoint_identity_partial_and_checksum_rejection(tmp_path):
    t=R.Task(Ny=8,Nx=4,samples=2);c=R.default_config();ident=R.identity(t,c)
    out,scratch=tmp_path/'out',tmp_path/'scratch'
    R.save_checkpoint(out,scratch,t,ident,np.zeros((2,64,64),complex),10,0.,R.capture_rng())
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
    t=R.Task(Ny=8,Nx=4);c=R.default_config();ident=R.identity(t,c)
    out,scratch=tmp_path/'out',tmp_path/'scratch'
    extract=R.extract_endpoint
    saved_cycles=[]
    save=R.save_checkpoint
    def record_checkpoint(*args,**kwargs):
        saved_cycles.append(args[5])
        return save(*args,**kwargs)
    monkeypatch.setattr(R,'save_checkpoint',record_checkpoint)
    def fail(*args,**kwargs):raise RuntimeError('interrupt endpoint')
    monkeypatch.setattr(R,'extract_endpoint',fail)
    with pytest.raises(RuntimeError,match='interrupt endpoint'):
        R.run_task(t,c,out,scratch,device='cpu')
    assert saved_cycles==list(range(10,61,10))
    assert int(R.load_checkpoint(out,t,ident)['completed_cycle'])==60
    monkeypatch.setattr(R,'extract_endpoint',extract)
    monkeypatch.setattr(R,'run_segment',lambda *a,**k:pytest.fail('Dynamics must not repeat after T=60'))
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


def test_report_only_needs_no_gpu_or_output_directory(tmp_path, monkeypatch, capsys):
    out=tmp_path/'absent';scratch=tmp_path/'scratch'
    monkeypatch.setattr(sys,'argv',['run_campaign.py','--lane','A','--output-root',str(out),
                                    '--scratch-root',str(scratch),'--report-only'])
    monkeypatch.setattr(R.torch.cuda,'is_available',lambda:False)
    R.main()
    report=json.loads(capsys.readouterr().out)
    assert report['trajectories']==630 and report['pending']==126
    assert not out.exists() and not scratch.exists()


def test_a100_and_allocator_enforcement_without_gpu(tmp_path,monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(sys,'argv',['run_campaign.py','--lane','C','--output-root',str(tmp_path/'out'),
                                    '--scratch-root',str(tmp_path/'scratch'),'--max-new-execution-batches','0'])
    monkeypatch.setattr(R.torch.cuda,'is_available',lambda:False)
    with pytest.raises(RuntimeError,match='CUDA is unavailable'):R.main()
    monkeypatch.setattr(R.torch.cuda,'is_available',lambda:True)
    props=SimpleNamespace(name='T4',total_memory=40*1024**3)
    monkeypatch.setattr(R.torch.cuda,'get_device_properties',lambda *a:props)
    with pytest.raises(RuntimeError,match='Expected A100'):R.main()
    props.name='NVIDIA A100';props.total_memory=37*1024**3
    with pytest.raises(RuntimeError,match='Expected A100'):R.main()
    props.total_memory=40*1024**3
    fractions=[]
    monkeypatch.setattr(R.torch.cuda,'set_per_process_memory_fraction',fractions.append)
    monkeypatch.setattr(R.shutil,'disk_usage',lambda *a:SimpleNamespace(free=10*1024**3))
    R.main()
    assert fractions==[38/40]


def test_atomic_replace_failure_does_not_publish_receipt(tmp_path,monkeypatch):
    def fail(*a):raise OSError('atomic replace interrupted')
    monkeypatch.setattr(R.os,'replace',fail)
    path=tmp_path/'out/result.npz';receipt=path.with_suffix('.json')
    with pytest.raises(OSError,match='atomic replace'):
        R.publish_pair({'a':np.arange(3)},path,receipt,{},tmp_path/'scratch',True)
    assert not path.exists() and not receipt.exists()


def test_full_analysis_synthetic_pairs_and_corruption(tmp_path):
    A=load_analysis(); config=R.default_config();out=tmp_path/'out'
    with pytest.raises(RuntimeError,match='42 verified'):
        A.analyze(out,tmp_path/'analysis')
    for task in R.tasks(config):
        # Analytic diagonal test spectra, NOT production simulation data.
        a=np.ones((10,660));a[:,:330]=-1
        if task.alpha_1!=3:
            a[:,330]=np.linspace(.01,.1,10)
            if task.alpha_1==2:a[5:,330]=1
        p=O.spectral_products(a,60)
        v=np.zeros((10,660),np.complex128)
        occ=np.zeros(10);rate=np.zeros(10)
        for i in range(10):
            if p['gap_mode_valid'][i]:
                j=p['gap_mode_index'][i];v[i,j]=1
                occ[i]=p['occupation_spectrum_raw'][i,j];rate[i]=p['lyapunov_rates'][i,j]
        p.update(gap_mode_vector=v,gap_mode_occupation=occ,gap_mode_rate=rate,
                 gap_mode_residual=np.zeros(10),eigensolver_residual=np.zeros(10),hermiticity_residual=np.zeros(10))
        active=np.array([2*(y*20+x)+mu for y in range(30) for x in range(5,16) for mu in range(2)])
        for start in (0,5):
            payload={k:val[start:start+5] for k,val in p.items()}
            payload.update(sample_indices=np.arange(start,start+5),Nx=20,Ny=30,T=60,alpha_1=task.alpha_1,
                           active_indices=active,active_coordinates_x_y_orbital=np.column_stack(((active//2)%20,active//40,active%2)))
            path,receipt=R.result_paths(out,task,start)
            old_ident=R.identity(task,config)|{'source_hashes':R.PRE_SORT_SOURCE_HASHES.copy()}
            R.publish_pair(payload,path,receipt,dict(old_ident,kind='result',sample_indices=list(range(start,start+5))),tmp_path/'scratch',True)
    result=A.analyze(out,tmp_path/'analysis')
    assert len(result['inputs'])==84 and len(result['summary'])==21
    rows={r['alpha_1']:r for r in result['summary']}
    assert rows[3]['statistic']=='all_capped' and rows[3]['mean'] is None
    assert rows[2]['statistic']=='finite_subset' and rows[2]['finite_samples']==5
    expected=np.arctanh(np.linspace(.01,.1,10))/60
    np.testing.assert_allclose(rows[1]['mean'],expected.mean())
    np.testing.assert_allclose(rows[1]['sem'],expected.std(ddof=1)/np.sqrt(10))
    assert (tmp_path/'analysis/gap_vs_alpha.pdf').stat().st_size>1000
    path,_=R.result_paths(out,R.tasks(config)[0],0)
    with path.open('ab') as f:f.write(b'corrupt')
    with pytest.raises(RuntimeError,match='42 verified'):A.analyze(out,tmp_path/'analysis2')


def test_offline_libm_roundoff_does_not_relax_caps_or_corruption():
    A=load_analysis()
    p=O.extract_endpoint(np.diag([-1.,-.3,.1,1.])[None].astype(np.complex128),
                         np.arange(4),60,device='cpu',progress=False)
    q={k:v.copy() for k,v in p.items()}
    q['lyapunov_rates'][0,1]=np.nextafter(q['lyapunov_rates'][0,1],np.inf)
    A.validate_offline_products(q,60)
    with pytest.raises(AssertionError):O.validate_products(q,60)
    q['lyapunov_rates'][0,1]+=1e-8
    with pytest.raises(AssertionError):A.validate_offline_products(q,60)
    q={k:v.copy() for k,v in p.items()};q['cap_mask'][0,0]=False
    with pytest.raises(AssertionError):A.validate_offline_products(q,60)
    q={k:v.copy() for k,v in p.items()};q['gap_mode_vector'][0]*=2
    with pytest.raises(AssertionError):A.validate_offline_products(q,60)


def test_add90_disjoint_ids_seeds_and_science():
    old=R.default_config();new=R.extension_config()
    changed={k for k in set(old)|set(new) if old.get(k)!=new.get(k)}
    assert changed=={'sampling_revision','samples_per_case','sample_start','execution_batch_size',
                     'combined_samples_per_case','previous_sampling_revision'}
    ts=R.tasks(new)
    assert len(ts)==21 and sum(t.samples for t in ts)==1890
    assert sum((t.samples+4)//5 for t in ts)==378
    assert len({t.seed for t in ts+R.tasks(old)})==42
    for t in ts:
        assert t.sample_ids().tolist()==list(range(10,100))
        assert 'samples010-099' in t.name
        paths=[R.result_paths(Path('/data'),t,i)[0] for i in range(0,90,5)]
        assert paths[0].name=='samples_010-014.npz' and paths[-1].name=='samples_095-099.npz'
    assert all(len(R.tasks(new,lane))==7 for lane in 'ABC')
    for lane in 'ABC':
        nb=json.loads((BUNDLE/f'run_alpha_endpoint_lane_{lane}.ipynb').read_text())
        source='\n'.join(''.join(c['source']) for c in nb['cells'])
        assert "'sample_start': 10" in source and "'samples_per_case': 90" in source
        assert "'meas_slab_only': True" in source and new['sampling_revision'] in source


def test_add90_offset_checkpoint_and_final_shard_recovery(tmp_path,monkeypatch):
    torch.set_num_threads(1)
    t=R.Task(Nx=4,Ny=8,samples=15,cycles=10,sample_start=10)
    c=R.extension_config();ident=R.identity(t,c)
    out=tmp_path/'out';scratch=tmp_path/'scratch'
    publish=R.publish_pair
    def fail_third(payload,path,receipt,expected,*args,**kwargs):
        if expected.get('kind')=='result' and expected['sample_indices']==list(range(20,25)):
            raise OSError('third shard interruption')
        return publish(payload,path,receipt,expected,*args,**kwargs)
    monkeypatch.setattr(R,'publish_pair',fail_third)
    with pytest.raises(OSError,match='third shard'):R.run_task(t,c,out,scratch,device='cpu')
    saved=R.load_checkpoint(out,t,ident)
    np.testing.assert_array_equal(saved['sample_indices'],np.arange(10,25))
    monkeypatch.setattr(R,'tasks',lambda *a:[t])
    row=R.inventory(c,out)[0]
    assert row['completed_shards']==2 and row['recoverable_cycle']==10
    monkeypatch.setattr(R,'publish_pair',publish)
    monkeypatch.setattr(R,'run_segment',lambda *a,**k:pytest.fail('Must resume final checkpoint without dynamics'))
    R.run_task(t,c,out,scratch,device='cpu')
    assert not any(p.exists() for p in R.checkpoint_paths(out,t))
    for start in (0,5,10):
        assert R.result_verified(out,t,start,ident)
        A=load_analysis();A.verify_offline_pair(out,t,start,ident)
