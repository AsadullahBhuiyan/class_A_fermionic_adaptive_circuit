"""CPU numerical/contract tests; no local GPU-engine dynamics or A100 claims."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
PARENT = REPO/'00_WORKSPACE/CURRENT/final_production_new_designs'
BUNDLE = PARENT/'23_hard_wall_random_center_chern'
sys.path.insert(0, str(BUNDLE))


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


R = load(BUNDLE/'run_campaign.py', 'random_chern_runner_test')
O = load(BUNDLE/'random_center_observer.py', 'random_chern_observer_test')


@pytest.fixture(autouse=True)
def few_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_campaign_and_frozen_tasks():
    c = R.default_config()
    R.validate_config(c)
    tasks = R.task_table(c, {'20': 100, '30': 50, '40': 25})
    assert sum(len(t['sample_ids']) for t in tasks) == 300
    assert sum(len(t['sample_ids'])*(2*t['ny']+1)*10 for t in tasks) == 183000
    assert len({t['seed'] for t in tasks}) == len(tasks)
    assert tasks == R.task_table(c, {'20': 100, '30': 50, '40': 25})
    for ny in c['ny_values']:
        assert [s for t in tasks if t['ny']==ny for s in t['sample_ids']] == list(range(100))
    assert c['perfect_correction'] and not c['postselect'] and c['meas_slab_only']
    assert c['dw_truncation'] and c['dtype']=='complex128' and c['sequence']=='raster_y'
    altered = dict(c, alpha_1=3)
    with pytest.raises(ValueError):
        R.validate_config(altered)


@pytest.mark.parametrize('ny', [20, 30, 40])
def test_centers_all_samples_cycles(ny):
    centers = np.stack([O.center_choices(2026092701, 20, ny, range(100), cycle)
                        for cycle in range(2*ny+1)], axis=1)
    assert centers.shape==(100, 2*ny+1, 10)
    assert (np.diff(np.sort(centers, axis=2), axis=2)>0).all()
    assert centers.min()>=0 and centers.max()<ny
    assert np.array_equal(centers[7, 3], O.center_choices(2026092701,20,ny,[7],3)[0])
    assert not np.array_equal(centers[:, 0], centers[:, 1])


@pytest.mark.parametrize('chunk', [1, 3, 10])
def test_batched_estimator_explicit_covariance_and_rank_padding(chunk):
    nx, ny, radius = 8, 10, 2.
    rng = np.random.default_rng(73)
    raw = rng.normal(size=(2,2*nx*ny,23)) + 1j*rng.normal(size=(2,2*nx*ny,23))
    frame = np.linalg.qr(raw)[0]
    ranks = np.array([17,23])
    # Inactive columns contain deliberately nonzero garbage: must be masked.
    centers = np.array([[0,9,1,4,6], [9,0,1,3,5]])
    tables = O.sector_table(nx,ny,radius)
    values = O.batched_chern(torch.from_numpy(frame), ranks, centers, tables, chunk).numpy()
    for sample in range(2):
        v = frame[sample,:,:ranks[sample]]
        gamma = (v @ v.conj().T).T
        for k,y0 in enumerate(centers[sample]):
            a,b,c = (t[y0] for t in tables)
            first = np.trace(gamma[np.ix_(c,a)] @ gamma[np.ix_(a,b)] @ gamma[np.ix_(b,c)])
            second = np.trace(gamma[np.ix_(a,c)] @ gamma[np.ix_(c,b)] @ gamma[np.ix_(b,a)])
            expected = (12j*np.pi*(first-second)).real
            assert values[sample,k] == pytest.approx(expected, abs=2e-13)
    sites = np.concatenate([t[0] for t in tables])//2
    assert ny-1 in sites//nx  # not a clipped seam disk
    assert len(np.unique(sites)) == 13  # integer disk of radius 2
    # Translation-covariance of sectors, not merely disk area.
    for table in tables:
        shift = (table[0]//(2*nx)+1)%ny*(2*nx)+table[0]%(2*nx)
        assert np.array_equal(np.sort(shift), np.sort(table[1]))


def test_observer_chunk_reproducibility_and_rng_isolation():
    rng = np.random.default_rng(71)
    frame = torch.from_numpy(np.linalg.qr(rng.normal(size=(160,30))+1j*rng.normal(size=(160,30)))[0])[None]
    state = SimpleNamespace(frame=frame, ranks=torch.tensor([30]))
    np.random.seed(18)
    torch.manual_seed(18)
    numpy_before = np.random.get_state()
    torch_before = torch.get_rng_state().clone()
    original = frame.clone()
    obs = []
    for chunk in [1,3,10]:
        o = O.RandomCenterObserver(8,10,2,[11],127,radius=2,count=10,chunk_size=chunk)
        for cycle in range(3):
            o.capture(cycle=cycle,state=state,batch_start=0,batch_count=1)
        obs.append(o.arrays())
    for payload in obs[1:]:
        assert np.array_equal(payload['centers_y'], obs[0]['centers_y'])
        np.testing.assert_allclose(payload['real_space_chern'],obs[0]['real_space_chern'],atol=2e-13)
    assert torch.equal(frame,original)
    assert torch.equal(torch_before,torch.get_rng_state())
    after = np.random.get_state()
    assert numpy_before[0]==after[0] and np.array_equal(numpy_before[1],after[1]) and numpy_before[2:]==after[2:]
    assert (obs[0]['global_charge']==30).all()


def test_trajectory_first_sem():
    raw = np.arange(100*41*10,dtype=float).reshape(100,41,10)
    values = raw.mean(axis=2)
    mean,sem = O.ensemble_statistics(values)
    np.testing.assert_array_equal(mean,values.mean(axis=0))
    np.testing.assert_array_equal(sem,values.std(axis=0,ddof=1)/10)


def small_payload():
    c = dict(R.default_config(), nx=8, ny_values=[10], samples=2)
    task = dict(id='test_batch',ny=10,sample_ids=[0,1],seed=83)
    o = O.RandomCenterObserver(8,10,20,[0,1],c['root_seed'],radius=1.6)
    frame = np.zeros((2,160,3),dtype=np.complex128)
    frame[0,:2,:2]=np.eye(2)
    frame[1,:3,:3]=np.eye(3)
    state=SimpleNamespace(frame=torch.from_numpy(frame),ranks=torch.tensor([2,3]))
    for cycle in range(21):
        o.capture(cycle=cycle,state=state,batch_start=0,batch_count=2)
    arrays=o.arrays()
    arrays.update(final_frame=frame,final_ranks=np.array([2,3],dtype=np.int64))
    return c,task,arrays


def test_publication_roundtrip_and_corrupt_pairs(tmp_path):
    c,task,arrays=small_payload()
    ident=R.identity(c)
    out,scratch=tmp_path/'drive',tmp_path/'scratch'
    R.publish_result(arrays,task,c,ident,out,scratch,4.)
    assert R.verified_result(out,task,ident,c)
    old=(out/'test_batch.npz').read_bytes()
    (out/'test_batch.npz').write_bytes(old[:-20])
    assert R.verified_result(out,task,ident,c) is None
    R.publish_result(arrays,task,c,ident,out,scratch,4.)
    assert R.verified_result(out,task,ident,c)
    (out/'test_batch.json').unlink()
    assert R.verified_result(out,task,ident,c) is None
    R.publish_result(arrays,task,c,ident,out,scratch,4.)
    assert R.verified_result(out,dict(task,seed=task['seed']+1),ident,c) is None
    assert R.verified_result(out,task,dict(ident,config_sha256='bad'),c) is None
    (out/'test_batch.npz').unlink()
    assert R.verified_result(out,task,ident,c) is None


def test_failed_readback_never_publishes_receipt(tmp_path,monkeypatch):
    c,task,arrays=small_payload()
    original=R.digest
    def wrong_readback(path):
        value=original(path)
        if str(path).endswith('.tmp'):
            value['sha256']='bad'
        return value
    monkeypatch.setattr(R,'digest',wrong_readback)
    with pytest.raises(IOError,match='readback'):
        R.publish_result(arrays,task,c,R.identity(c),tmp_path/'drive',tmp_path/'scratch',1.)
    assert not (tmp_path/'drive/test_batch.json').exists()
    assert not (tmp_path/'drive/test_batch.npz').exists()


def test_failed_atomic_replacement_preserves_previous(tmp_path,monkeypatch):
    source,dest=tmp_path/'local',tmp_path/'drive/result'
    source.write_bytes(b'old')
    R.atomic_copy(source,dest)
    source.write_bytes(b'new')
    def fail(*args):
        raise OSError('interrupted rename')
    monkeypatch.setattr(R.os,'replace',fail)
    with pytest.raises(OSError):
        R.atomic_copy(source,dest)
    assert dest.read_bytes()==b'old'
    assert list(dest.parent.glob('*.tmp'))==[]


@pytest.mark.parametrize('field', ['global_charge','final_ranks','final_frame','center_average','centers_y','cycles'])
def test_bad_scientific_arrays_rejected(field):
    c,task,arrays=small_payload()
    altered=copy.deepcopy(arrays)
    altered[field].flat[0] += 1
    if field=='global_charge':
        altered[field][0,-1] += 1
    with pytest.raises(ValueError):
        R.validate_arrays(altered,task,c)


def test_calibration_limits_and_throughput_selection():
    c=R.default_config()
    rows=[dict(batch=25,status='ok',seconds_per_cycle=2,peak_reserved_gib=8,forecast_seconds=300),
          dict(batch=50,status='ok',seconds_per_cycle=3,peak_reserved_gib=20,forecast_seconds=450),
          dict(batch=100,status='ok',seconds_per_cycle=4,peak_reserved_gib=33,forecast_seconds=600)]
    assert R.select_candidate(rows,c)['batch']==50
    rows[1]['forecast_seconds']=2700
    assert R.select_candidate(rows,c)['batch']==25
    rows[0]['peak_reserved_gib']=32
    assert R.select_candidate(rows,c) is None


def test_registration_sources_manifest_and_notebook():
    layout=load(PARENT/'bundle_layout.py','random_chern_layout_test')
    assert BUNDLE.name in layout.validate_bundle_layout(PARENT)
    for name in ['classA_U1FGTN_gpu.py','occupied_frame_gpu.py']:
        assert (BUNDLE/'src'/name).read_bytes()==(REPO/'src/fgtn'/name).read_bytes()
    manifest=json.loads((BUNDLE/'deployment_manifest.json').read_text())
    for name,expected in manifest['files'].items():
        assert R.digest(BUNDLE/name)==expected
    nb=json.loads((BUNDLE/'run_hard_wall_random_center_chern.ipynb').read_text())
    code=[''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code']
    for source in code:
        compile(source,'notebook','exec')
    cfg=next(source for source in code if 'CONFIG = {' in source)
    ns={}
    exec(cfg,ns)
    assert ns['CONFIG']==R.default_config()
    assert sum('drive.mount(' in s for s in code)==1
    assert any('os.read(' in s and 'stderr=subprocess.STDOUT' in s for s in code)
    assert code[-1]=="from google.colab import runtime\nruntime.unassign()\nprint('done')\n"


def test_cpu_canonical_observer_noninvasive_and_interruption_rerun():
    # Local simulations must use the canonical CPU class, not GPU on device=cpu.
    sys.path.insert(0,str(REPO/'src/fgtn'))
    from classA_U1FGTN import classA_U1FGTN
    from threadpoolctl import threadpool_limits
    def run(observer):
        np.random.seed(232345)
        torch.manual_seed(232345)
        model=classA_U1FGTN(Nx=8,Ny=10,DW=True,nshell=1,
                           filling_frac=.5,alpha_1=1,alpha_2=30,
                           dw_truncation=True)
        return model.run_markov_circuit(
            G_history=False,progress=False,cycles=2,samples=1,init_mode='default',
            save=False,postselect=False,perfect_correction=True,sequence='raster_y',
            meas_slab_only=True,state_representation='physical_frame',
            native_cycle_observer=observer,track_choi=False,return_native_state=True,
            require_no_covariance_materialization=True,frame_reorthonormalize_interval=1,
            parallelize_samples=False,random_seed=232345)
    with threadpool_limits(limits=1):
        obs=O.RandomCenterObserver(8,10,2,[0],2026092701,radius=1.6)
        recorded=run(obs.capture)
        rng_after_recorded=np.random.get_state()
        torch_after_recorded=torch.get_rng_state().clone()
        plain=run(None)
        rng_after_plain=np.random.get_state()
        assert np.array_equal(rng_after_recorded[1],rng_after_plain[1])
        assert rng_after_recorded[2:]==rng_after_plain[2:]
        assert torch.equal(torch_after_recorded,torch.get_rng_state())
        np.testing.assert_array_equal(recorded['native_final']['frame'],plain['native_final']['frame'])
        assert np.array_equal(obs.arrays()['cycles'],[0,1,2])
        assert obs.charge[0,-1]==recorded['native_final']['rank']
        def interrupt(**kwargs):
            if kwargs['cycle']==1:
                raise RuntimeError('simulated interruption')
        with pytest.raises(RuntimeError,match='simulated interruption'):
            run(interrupt)
        rerun=run(None)
        np.testing.assert_array_equal(plain['native_final']['frame'],rerun['native_final']['frame'])


def test_calibration_fallback_and_distinct_seeds(monkeypatch):
    config=R.default_config()
    calls=[]
    for method in ['empty_cache','reset_peak_memory_stats']:
        monkeypatch.setattr(R.torch.cuda,method,lambda:None)
    def fake_run(c,task,cycles,calibration=False):
        b=len(task['sample_ids'])
        assert calibration and cycles==6
        calls.append((task['ny'],b,task['seed']))
        return None,dict(seconds_per_cycle=2.,setup_seconds=1.,elapsed_seconds=13.,
                         peak_reserved_gib=33. if b>=25 else 10.)
    monkeypatch.setattr(R,'run_dynamics',fake_run)
    sizes,rows=R.calibrate(config)
    assert sizes=={'20':10,'30':10,'40':10}
    assert len(calls)==15 and len({seed for ny,b,seed in calls})==15
    prod_seeds={t['seed'] for t in R.task_table(config,sizes)}
    assert not prod_seeds.intersection(seed for ny,b,seed in calls)
    plan=dict(config=config,identity=R.identity(config),batch_sizes=sizes,
              tasks=R.task_table(config,sizes),calibration=rows)
    R.validate_plan(plan,config,R.identity(config))
    plan['tasks'][0]['seed']+=1
    with pytest.raises(RuntimeError,match='task/seed'):
        R.validate_plan(plan,config,R.identity(config))


def test_queue_skip_corruption_and_report_only(tmp_path,monkeypatch):
    config,_,arrays=small_payload()
    ident=R.identity(config)
    sizes={'10':5}
    tasks=R.task_table(config,sizes)
    plan=dict(config=config,identity=ident,batch_sizes=sizes,tasks=tasks,
              calibration={'10':[dict(batch=5,status='ok',seconds_per_cycle=1.,
                                     peak_reserved_gib=1.,forecast_seconds=30.)]})
    out,scratch=tmp_path/'output',tmp_path/'scratch'
    out.mkdir()
    configpath=tmp_path/'config.json'
    configpath.write_bytes(R.json_bytes(config))
    (out/'execution_plan.json').write_bytes(R.json_bytes(plan))
    monkeypatch.setattr(R,'validate_config',lambda c:None)  # small synthetic fixture only
    monkeypatch.setattr(R,'require_a100',lambda:None)
    monkeypatch.setattr(R.shutil,'disk_usage',lambda p:SimpleNamespace(free=100*1024**3))
    for method in ['empty_cache','reset_peak_memory_stats']:
        monkeypatch.setattr(R.torch.cuda,method,lambda:None)
    calls=[]
    def fake_run(c,task,cycles):
        calls.append(task['seed'])
        assert cycles==20
        return copy.deepcopy(arrays),dict(peak_reserved_gib=1.)
    monkeypatch.setattr(R,'run_dynamics',fake_run)
    args=['--config',str(configpath),'--output-root',str(out),'--scratch-root',str(scratch)]
    R.main(args+['--report-only'])
    assert calls==[]
    R.main(args)
    assert calls==[tasks[0]['seed']]
    first=R.verified_result(out,tasks[0],ident,config,load=True)
    R.main(args)
    assert len(calls)==1  # no calibration, no GPU work, no rewriting completed pair
    (out/(tasks[0]['id']+'.npz')).write_bytes(b'interrupted/corrupt')
    R.main(args)
    assert calls==[tasks[0]['seed']]*2
    second=R.verified_result(out,tasks[0],ident,config,load=True)
    np.testing.assert_array_equal(first['center_average'],second['center_average'])
    receipt_path=out/(tasks[0]['id']+'.json')
    receipt=json.loads(receipt_path.read_text())
    receipt['elapsed_seconds']=3601
    receipt_path.write_bytes(R.json_bytes(receipt))
    # Complete campaigns may still be inspected without any new work.
    R.main(args)
    assert len(calls)==2


def test_engine_entrypoint_signature_and_native_numpy_export(monkeypatch):
    """Contract stub only: no GPU engine dynamics are run locally."""
    import inspect
    module=load(REPO/'src/fgtn/classA_U1FGTN_gpu.py','random_chern_signature')
    actual=inspect.signature(module.classA_U1FGTN_gpu.run_markov_circuit)
    config=R.default_config()
    task=dict(ny=20,sample_ids=[0],seed=1)
    class Model:
        DW_loc=(5,15)
        dtype=torch.complex128
        def __init__(self,**kwargs):
            assert kwargs['dw_truncation'] and kwargs['device']=='cuda'
        def run_markov_circuit(self,**kwargs):
            actual.bind(self,**kwargs)
            observer=kwargs['native_cycle_observer'].__self__
            frame=np.zeros((1,800,400),dtype=np.complex128)
            frame[0,:400,:]=np.eye(400)
            state=SimpleNamespace(frame=torch.from_numpy(frame),ranks=torch.tensor([400]))
            for cycle in range(kwargs['cycles']+1):
                observer.capture(cycle=cycle,state=state,batch_start=0,batch_count=1)
            return dict(state_representation_resolved='physical_frame',samples=1,
                        covariance_materialization_count=0,exterior_preparation_performed=True,
                        native_final=dict(frame=frame,ranks=np.array([400])))
    monkeypatch.setitem(sys.modules,'classA_U1FGTN_gpu',SimpleNamespace(classA_U1FGTN_gpu=Model))
    monkeypatch.setattr(R.torch.cuda,'synchronize',lambda *args:None)
    monkeypatch.setattr(R.torch.cuda,'max_memory_reserved',lambda:100)
    arrays,metrics=R.run_dynamics(config,task,1)
    assert arrays['final_frame'].dtype==np.complex128
    assert arrays['final_ranks'].tolist()==[400]
    assert arrays['center_average'].shape==(1,2)


@pytest.mark.parametrize('ny',[20,30,40])
def test_production_geometry_radius_four(ny):
    nx=20
    tables=O.sector_table(nx,ny,4.)
    assert sum(t.shape[1] for t in tables)==98  # 49 cells, two orbitals
    rng=np.random.default_rng(67)
    raw=rng.normal(size=(2*nx*ny,31))+1j*rng.normal(size=(2*nx*ny,31))
    v=np.linalg.qr(raw)[0]
    centers=O.center_choices(2026092701,nx,ny,[99],2*ny)
    centers[0,0]=0
    value=O.batched_chern(torch.from_numpy(v)[None],[31],centers,tables).numpy()[0]
    gamma=(v@v.conj().T).T
    for idx,y0 in enumerate(centers[0]):
        a,b,c=(t[y0] for t in tables)
        first=np.trace(gamma[np.ix_(c,a)]@gamma[np.ix_(a,b)]@gamma[np.ix_(b,c)])
        reverse=np.trace(gamma[np.ix_(a,c)]@gamma[np.ix_(c,b)]@gamma[np.ix_(b,a)])
        assert value[idx]==pytest.approx((12j*np.pi*(first-reverse)).real,abs=2e-13)
        cells=np.concatenate([t[y0] for t in tables])//2
        assert cells.min()>=0 and cells.max()<nx*ny
        assert cells.size==np.unique(cells).size*2
        assert (cells%nx).min()==6 and (cells%nx).max()==14


def test_overlong_completed_batch_stops_pending_queue(tmp_path,monkeypatch):
    c=R.default_config()
    ident=R.identity(c)
    sizes={'20':100,'30':100,'40':100}
    plan=dict(config=c,identity=ident,batch_sizes=sizes,tasks=R.task_table(c,sizes),
              calibration={str(ny):[dict(batch=100,status='ok',seconds_per_cycle=1,
                  peak_reserved_gib=10,forecast_seconds=100)] for ny in c['ny_values']})
    out=tmp_path/'output'
    out.mkdir()
    (out/'execution_plan.json').write_bytes(R.json_bytes(plan))
    def inventory(output,task,*args):
        return dict(elapsed_seconds=3601) if task['ny']==20 else None
    monkeypatch.setattr(R,'verified_result',inventory)
    def forbidden():
        raise AssertionError('must not launch more GPU work')
    monkeypatch.setattr(R,'require_a100',forbidden)
    with pytest.raises(RuntimeError,match='exceeded one hour'):
        R.main(['--output-root',str(out)])


def test_no_qualifying_calibration_stops(monkeypatch):
    for method in ['empty_cache','reset_peak_memory_stats']:
        monkeypatch.setattr(R.torch.cuda,method,lambda:None)
    def oom(*args,**kwargs):
        raise torch.cuda.OutOfMemoryError('simulated')
    monkeypatch.setattr(R,'run_dynamics',oom)
    with pytest.raises(RuntimeError,match='production NOT launched'):
        R.calibrate(R.default_config())
