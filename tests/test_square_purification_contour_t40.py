import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
PARENT=ROOT/'00_WORKSPACE/CURRENT/final_production_new_designs'
BUNDLE=PARENT/'30_square_purification_contour_t40'
sys.path.insert(0,str(BUNDLE))
spec=importlib.util.spec_from_file_location('square_contour_t40_runner',BUNDLE/'run_campaign.py')
R=importlib.util.module_from_spec(spec);spec.loader.exec_module(R)
O=sys.modules['contour_observer']
IO=sys.modules['io_utils']
sys.path.remove(str(BUNDLE))


def tiny():
    return R.default_config()|dict(sizes=[4],cycles=4,segment_cycles=2,device='cpu')


def test_contract_notebook_and_sources():
    c=R.default_config();R.validate_config(c)
    assert c['sizes']==[30,40] and c['cycles']==40 and c['samples']==1
    assert c['alpha_1']==1 and c['alpha_2']==30 and c['nshell']==1
    assert not c['meas_slab_only'] and not c['postselect'] and not c['covariance_spectral_clip']
    assert c['init_mode']=='maxmix' and c['dtype']=='complex128' and c['segment_cycles']==5
    assert R.identity(c,30)['seed']!=R.identity(c,40)['seed']
    assert R.identity(c,40)==R.identity(c|{'sizes':[40]},40)
    for name in ('classA_U1FGTN_gpu.py','occupied_frame_gpu.py'):
        assert (BUNDLE/'src'/name).read_bytes()==(ROOT/'src/fgtn'/name).read_bytes()
    nb=json.loads((BUNDLE/'run_square_purification_contour.ipynb').read_text())
    for cell in nb['cells']:
        if cell['cell_type']=='code':compile(''.join(cell['source']),cell['id'],'exec')
    text='\n'.join(''.join(cell['source']) for cell in nb['cells'])
    assert 'sys.stdout.buffer' not in text and 'sys.stdout.write' in text
    assert 'REPORT_ONLY' in text and 'MAX_NEW_TASKS' in text and '/content/' in text
    assert ''.join(nb['cells'][-1]['source'])=="from google.colab import runtime\nruntime.unassign()\nprint('done')\n"
    assert BUNDLE.name in json.loads((PARENT/'bundle_index.json').read_text())['bundles']


def test_dense_observer_and_no_mutation():
    torch.set_num_threads(1)
    L=4; rng=np.random.default_rng(23);n=2*L*L
    v,_=np.linalg.qr(rng.normal(size=(n,n))+1j*rng.normal(size=(n,n)))
    nu=np.linspace(.01,.99,n);G=torch.tensor(((v*(2*nu-1))@v.conj().T)[None])
    original=G.clone();state=R.capture_rng()
    p=O.observe_covariance(G,L,4)
    weights=-nu*np.log(nu)-(1-nu)*np.log1p(-nu)
    reference=(abs(v)**2@weights).reshape(L,L,2).sum(-1).T
    np.testing.assert_allclose(p['entropy_contour'],reference,atol=1e-13)
    np.testing.assert_allclose(p['entropy_contour'].sum(),weights.sum(),atol=1e-13)
    np.testing.assert_allclose(p['occupation_spectrum'],nu,atol=1e-14)
    np.testing.assert_allclose(p['lyapunov_gap'],p['modular_gap']/8)
    assert torch.equal(G,original)
    for key,value in state.items():np.testing.assert_array_equal(R.capture_rng()[key],value)
    p=O.observe_covariance(torch.zeros_like(G),L,0)
    np.testing.assert_allclose(p['entropy_contour'],2*np.log(2))
    assert np.isnan(p['lyapunov_gap'])
    with pytest.raises(FloatingPointError,match='occupation excess'):
        O.observe_covariance(torch.eye(n,dtype=torch.complex128)[None]*1.1,L,1)


def test_exact_segment_resume_and_observer_rng():
    torch.set_num_threads(1);c=tiny();ident=R.identity(c,4)
    model=R.build_model(c,4)
    np.random.seed(ident['seed']);torch.manual_seed(ident['seed'])
    a=O.allocate(4,4)
    expected,expected_rng=R.run_segment(model,c,4,None,0,4,None,a)
    np.random.seed(ident['seed']);torch.manual_seed(ident['seed'])
    b=O.allocate(4,4)
    G,rng=R.run_segment(model,c,4,None,0,2,None,b)
    model=R.build_model(c,4)
    np.random.random(10);torch.rand(7)
    actual,actual_rng=R.run_segment(model,c,4,G,2,2,rng,b)
    np.testing.assert_array_equal(actual,expected)
    for key in a:np.testing.assert_array_equal(a[key],b[key])
    for key in expected_rng:np.testing.assert_array_equal(actual_rng[key],expected_rng[key])


def test_checkpoint_result_resume_and_readback(tmp_path,monkeypatch):
    torch.set_num_threads(1);c=tiny();ident=R.identity(c,4)
    out=tmp_path/'out';scratch=tmp_path/'scratch'
    assert not R.run_task(c,4,out,scratch,max_segments=1)
    p=R.load_pair(out,4,'checkpoint',ident);assert int(p['completed_cycle'])==2
    assert R.load_pair(out,4,'checkpoint',ident|{'seed':0}) is None
    original=R.publish_pair
    def failure(payload,path,receipt,expected,*args,**kwargs):
        if expected['kind']=='result':raise OSError('failed result write')
        return original(payload,path,receipt,expected,*args,**kwargs)
    monkeypatch.setattr(R,'publish_pair',failure)
    with pytest.raises(OSError,match='failed result'):
        R.run_task(c,4,out,scratch)
    assert int(R.load_pair(out,4,'checkpoint',ident)['completed_cycle'])==4
    monkeypatch.setattr(R,'publish_pair',original)
    monkeypatch.setattr(R,'run_segment',lambda *a,**kw:pytest.fail('must skip finished dynamics'))
    assert R.run_task(c,4,out,scratch)
    assert R.load_pair(out,4,'result',ident) is not None
    assert not any(p.exists() for p in R.paths(out,4,'checkpoint'))
    assert R.run_task(c,4,out,scratch)
    path,receipt=R.paths(out,4,'result');before=path.read_bytes()
    path.write_bytes(before+b'bad')
    assert R.load_pair(out,4,'result',ident) is None
    path.write_bytes(before);receipt.unlink()
    assert R.load_pair(out,4,'result',ident) is None


def test_failed_readback_never_publishes_receipt(tmp_path,monkeypatch):
    original=IO.sha
    monkeypatch.setattr(IO,'sha',lambda p:'bad' if str(p).endswith('.tmp') else original(p))
    path=tmp_path/'drive/result.npz';receipt=path.with_suffix('.json')
    with pytest.raises(OSError,match='readback'):
        IO.publish_pair({'x':np.ones(3)},path,receipt,{},tmp_path/'scratch',True)
    assert not receipt.exists()

def test_full_forty_cycle_history_and_resume(tmp_path):
    torch.set_num_threads(1)
    c=R.default_config()|dict(sizes=[4],device='cpu')
    ident=R.identity(c,4)
    out,scratch=tmp_path/'out',tmp_path/'scratch'
    assert not R.run_task(c,4,out,scratch,max_segments=4)
    checkpoint=R.load_pair(out,4,'checkpoint',ident)
    assert int(checkpoint['completed_cycle'])==20
    prefix={key:checkpoint[key][:21].copy() for key in O.allocate(4,40)}
    assert R.run_task(c,4,out,scratch)
    result=R.load_pair(out,4,'result',ident)
    np.testing.assert_array_equal(result['cycles'],np.arange(41))
    assert result['entropy_contour'].shape==(41,4,4)
    assert result['occupation_spectrum'].shape==(41,32)
    for key,values in prefix.items():np.testing.assert_array_equal(result[key][:21],values)
    np.testing.assert_allclose(result['entropy_contour'].sum((1,2)),result['total_entropy'],atol=1e-12)
    assert R.load_pair(out,4,'result',ident|{'config_sha256':'old60cycle'}) is None
