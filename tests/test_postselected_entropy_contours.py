"""Small CPU-only validation of the Colab observer and serialization contract."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / '00_WORKSPACE/CURRENT/final_production_new_designs/19_postselected_hard_soft_n20x40'
sys.path.insert(0, str(BUNDLE))
spec = importlib.util.spec_from_file_location('postselected_contour_test_runner', BUNDLE / 'run_campaign.py')
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)
O = R.PostselectedObserver


@pytest.mark.parametrize('slab', [True, False])
def test_contour_numpy_formula_mapping_and_checkpoint(slab):
    nx, ny = 4, 3
    ids = np.array([mu+2*x+2*nx*y for y in range(ny)
                    for x in range(nx) if not slab or x in (1, 2) for mu in (0, 1)])
    rng = np.random.default_rng(42)
    n = len(ids)
    U, _ = np.linalg.qr(rng.normal(size=(n,n)) + 1j*rng.normal(size=(n,n)))
    nu = np.linspace(.02, .98, n)
    active = (U*(2*nu-1)) @ U.conj().T
    G = np.eye(2*nx*ny, dtype=complex)
    G[np.ix_(ids, ids)] = active
    observer = O(cycles=2, active_indices=torch.tensor(ids), nx=nx, ny=ny)
    tensor = torch.tensor(G[None], dtype=torch.complex128)
    original = tensor.clone()
    state = torch.get_rng_state().clone()
    for cycle in range(3):
        observer.observe(cycle=cycle, G=tensor)
    h = -nu*np.log(nu)-(1-nu)*np.log1p(-nu)
    expected = np.zeros((nx,ny))
    for basis, value in zip(ids, abs(U)**2 @ h):
        expected[(basis//2)%nx, basis//(2*nx)] += value
    np.testing.assert_allclose(observer.entropy_contour, np.broadcast_to(expected,(3,nx,ny)), atol=2e-14)
    np.testing.assert_allclose(observer.entropy_contour_x, observer.entropy_contour.sum(axis=2))
    np.testing.assert_allclose(observer.total_entropy_nats, h.sum())
    assert torch.equal(tensor, original) and torch.equal(state, torch.get_rng_state())
    restored = O(cycles=2, active_indices=torch.tensor(ids), nx=nx, ny=ny)
    restored.restore(observer.checkpoint_payload(1), completed_cycle=1)
    restored.observe(cycle=2, G=tensor)
    for key in O.ARRAY_NAMES:
        np.testing.assert_array_equal(getattr(restored,key), getattr(observer,key))
    restored.validate()


def test_maxmix_caps_and_old_checkpoint_rejection():
    o = O(cycles=1, active_indices=torch.arange(8), nx=2, ny=2)
    o.observe(cycle=0, G=torch.zeros((1,8,8),dtype=torch.complex128))
    np.testing.assert_allclose(o.entropy_contour[0], 2*np.log(2))
    o.observe(cycle=1, G=torch.diag(torch.tensor([1.,-1.]*4,dtype=torch.complex128))[None])
    np.testing.assert_array_equal(o.entropy_contour[1], 0.)
    old = o.checkpoint_payload(1)
    del old['entropy_contour']
    with pytest.raises(KeyError):
        o.restore(old, completed_cycle=1)


@pytest.mark.parametrize('construction', ['hard','soft'])
@pytest.mark.parametrize('alpha', [1.,3.])
def test_small_engine_observer_and_serialized_resume(tmp_path, construction, alpha):
    # Validation only: exercise the GPU implementation's CPU tensor backend.
    cfg = json.loads((BUNDLE/'campaign_config.json').read_text())
    cfg.update(Nx=4, Ny=4, cycles=2, device='cpu', alpha_1=alpha)
    model = R.build_model(cfg, construction)
    idx = model.active_top_layer_indices(meas_slab_only=construction=='hard')
    def observer(): return O(cycles=2, active_indices=idx, nx=4, ny=4)
    class Bar:
        def update(self, *a): pass
        def set_postfix(self, **kw): pass
    def run(o, start, count, G=None):
        return R.run_segment(model,cfg,construction,o,completed_cycle=start,
            segment_cycles=count,G_init=G,progress_bar=Bar())
    np.random.seed(15); torch.manual_seed(15)
    whole = observer()
    final = run(whole,0,2)
    rng_final = R.capture_rng()
    np.random.seed(15); torch.manual_seed(15)
    baseline = model.run_markov_circuit(G_history=False,progress=False,cycles=2,
        postselect=True,postselect_probability=1.,perfect_correction=False,samples=1,
        init_mode='maxmix',save=False,save_init=False,n_a=.5,sequence='raster_y',
        meas_slab_only=construction=='hard',batch_size=1,return_data=True,
        state_representation='covariance',initial_purity_tolerance=.50000001)
    np.testing.assert_array_equal(final,baseline['G_final'])
    for key,value in rng_final.items():
        np.testing.assert_array_equal(value,R.capture_rng()[key])
    np.random.seed(15); torch.manual_seed(15)
    prefix = observer()
    first = run(prefix,0,1)
    R.save_checkpoint(tmp_path/'out',tmp_path/'scratch',construction,completed_cycle=1,
        elapsed_seconds=1,G=first,observer=prefix,cfg_hash='test',hashes={})
    restored = observer()
    cycle, _, G, rng, _ = R.load_checkpoint(tmp_path/'out',construction,cycles=2,
        observer=restored,cfg_hash='test',hashes={})
    R.restore_rng(rng)
    resumed = run(restored,cycle,1,G)
    np.testing.assert_array_equal(final,resumed)
    for key in O.ARRAY_NAMES:
        np.testing.assert_allclose(getattr(whole,key),getattr(restored,key),atol=1e-13)
    for key,value in rng_final.items():
        np.testing.assert_array_equal(value,R.capture_rng()[key])
    restored.validate()
    assert restored.entropy_contour.shape == (3,4,4)
    if construction == 'hard':
        np.testing.assert_array_equal(restored.entropy_contour[:,0,:],0.)


def test_four_task_queue_notebook_manifest_and_sources():
    cfg = json.loads((BUNDLE/'campaign_config.json').read_text())
    tasks = list(R.campaign_tasks(cfg))
    assert [(d,c) for d,c,_ in tasks] == [('alpha1_1','hard'),('alpha1_1','soft'),('alpha1_3','hard'),('alpha1_3','soft')]
    assert [c['alpha_1'] for _,_,c in tasks] == [1,1,3,3]
    assert cfg['sampling_revision'].endswith('_v2')
    nb = json.loads((BUNDLE/'run_postselected_hard_soft_n20x40.ipynb').read_text())
    for cell in nb['cells']:
        if cell['cell_type']=='code': ast.parse(''.join(cell['source']))
    assert ''.join(nb['cells'][-1]['source']) == "from google.colab import runtime\nruntime.unassign()\nprint('done')\n"
    manifest = json.loads((BUNDLE/'deployment_manifest.json').read_text())
    for file,row in manifest['files'].items():
        raw = (BUNDLE/file).read_bytes()
        assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
    for name in ['classA_U1FGTN_gpu.py','occupied_frame_gpu.py']:
        assert (BUNDLE/'src'/name).read_bytes() == (ROOT/'src/fgtn'/name).read_bytes()


def test_old_output_root_rejected(tmp_path):
    with pytest.raises(ValueError,match='v1 results must remain untouched'):
        R.main(['--output-root',str(tmp_path/'old_v1'),'--scratch-root',str(tmp_path/'scratch'),'--report-only'])
    assert not (tmp_path/'old_v1').exists()


def test_main_runs_all_four_with_isolated_paths_and_limit(tmp_path,monkeypatch):
    cfg = json.loads((BUNDLE/'campaign_config.json').read_text())
    launched = []
    monkeypatch.setattr(R,'require_runtime',lambda cfg: None)
    monkeypatch.setattr(R,'execute',lambda cfg,out,scratch,construction,**kw:
        launched.append((cfg['alpha_1'],construction,out,scratch)))
    args = ['--output-root',str(tmp_path/cfg['sampling_revision']),
            '--scratch-root',str(tmp_path/'scratch')]
    assert R.main(args+['--report-only']) == 0 and not launched
    assert R.main(args+['--max-new-constructions','1']) == 0
    assert len(launched)==1
    launched.clear()
    assert R.main(args)==0
    assert [(a,c) for a,c,_,_ in launched]==[(1.,'hard'),(1.,'soft'),(3.,'hard'),(3.,'soft')]
    assert len({(out,c) for _,c,out,_ in launched})==4
