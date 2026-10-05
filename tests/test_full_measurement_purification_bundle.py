"""Bundle 21: full-layer measurements, selective observables, scientific resume."""
import ast
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / '00_WORKSPACE/CURRENT/final_production_new_designs/21_hard_wall_full_measurement_purification'


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


OBS = load(BUNDLE / 'purification_observer.py', '_bundle21_observer_test')
previous = sys.modules.get('purification_observer')
sys.modules['purification_observer'] = OBS
RUN = load(BUNDLE / 'run_campaign.py', '_bundle21_runner_test')
if previous is None:
    sys.modules.pop('purification_observer', None)
else:
    sys.modules['purification_observer'] = previous


def tiny(alpha=1):
    return SimpleNamespace(nx=4, ny=4, samples=2, cycles=4, alpha_1=float(alpha),
        construction='hard', sample_indices=np.arange(2), seed=713,
        sample_start=0, sample_stop=2, task_id=f'test_alpha{alpha}')


def observer(task):
    return OBS.PurificationObserver(nx=task.nx, ny=task.ny, cycles=task.cycles,
        sample_indices=task.sample_indices, sample_chunk=2, construction='hard',
        save_contours=task.alpha_1 == 1)


class Bar:
    def update(self, n): pass


def model(alpha):
    return RUN.classA_U1FGTN_gpu(Nx=4, Ny=4, DW=True, nshell=1, alpha_1=alpha,
        alpha_2=30, dw_truncation=True, device='cpu', dtype='complex128', backend='local')


def seed(task):
    np.random.seed(task.seed)
    torch.manual_seed(task.seed)


def test_contract_and_paths():
    config = RUN.expected_config()
    assert config['Ny_values'] == [30] and config['Nx'] == 20
    assert config['alpha_1_values'] == [1, 3] and config['cycles_multiplier'] == 2
    assert config['constructions']['hard'] == dict(DW=True, dw_truncation=True, meas_slab_only=False)
    assert config['perfect_correction'] and not config['postselect']
    assert config['init_mode'] == 'maxmix' and config['dtype'] == 'complex128'
    assert config['sequence'] == 'raster_y' and config['observer_sample_chunk_by_Ny'] == {'30': 10}
    tasks = RUN.expand_execution_batches(config, 'hard')
    assert [t.alpha_1 for t in tasks] == [1, 3]
    assert all(t.samples == 100 and t.cycles == 60 for t in tasks)
    assert len({t.seed for t in tasks}) == len({t.task_id for t in tasks}) == 2
    assert tasks == RUN.expand_execution_batches(config, 'hard')
    shards = RUN.all_result_shards(config, 'hard')
    assert len(shards) == 40
    assert len({RUN.result_paths(Path('out'), s)[0] for s in shards}) == 40
    for task in tasks:
        np.testing.assert_array_equal(np.concatenate([s.sample_indices for s in RUN.result_shards(task)]), np.arange(100))
    with pytest.raises(ValueError): RUN.validate_config({**config, 'cycles_multiplier': 4})
    with pytest.raises(ValueError): RUN.expand_execution_batches(config, 'soft')


@pytest.mark.parametrize('alpha', [1, 3])
def test_small_engine_exact_resume_and_global_cycle_zero(tmp_path, monkeypatch, alpha):
    monkeypatch.setattr(RUN, 'SEGMENT_CYCLES', 2)
    task = tiny(alpha)
    cfg = {**RUN.expected_config(), 'device': 'cpu'}
    seed(task)
    whole = observer(task)
    final = RUN.run_segment(model(alpha), cfg, task, whole, completed_cycle=0,
        segment_cycles=4, G_init=None, continuing=False, progress_bar=Bar())
    whole.validate(completed_cycle=4, final=True)
    rng = RUN.capture_rng()
    np.testing.assert_allclose(whole.occupation_spectrum[:, 0], .5, atol=0)
    np.testing.assert_allclose(whole.total_entropy[:, 0], 32 * np.log(2), atol=1e-13)
    assert np.all(whole.site_event_count[:, 1:] == 16)
    assert np.all(whole.channel_event_count[:, 1:] == 64)
    assert whole.transfer_mode_count == 32
    if alpha == 1:
        np.testing.assert_allclose(whole.entropy_contour.sum(axis=(2, 3)), whole.total_entropy, atol=1e-12)
    else:
        assert not hasattr(whole, 'entropy_contour')
        assert 'entropy_contour' not in whole.result_payload(slice(None))
    seed(task)
    prefix = observer(task)
    g = RUN.run_segment(model(alpha), cfg, task, prefix, completed_cycle=0,
        segment_cycles=2, G_init=None, continuing=False, progress_bar=Bar())
    RUN.save_checkpoint(tmp_path/'out', tmp_path/'scratch', task, completed_cycle=2,
        elapsed_seconds=1, G=g, observer=prefix, cfg_hash='cfg', hashes={})
    cp, status = RUN.load_checkpoint(tmp_path/'out', task, cfg_hash='cfg', hashes={})
    assert status == 'verified'
    resumed = observer(task)
    resumed.restore_checkpoint(cp.observer_payload, completed_cycle=2)
    resumed_model = model(alpha)
    RUN.restore_rng(cp.rng_payload)
    result = RUN.run_segment(resumed_model, cfg, task, resumed, completed_cycle=2,
        segment_cycles=2, G_init=cp.G, continuing=True, progress_bar=Bar())
    np.testing.assert_array_equal(result, final)
    for k in whole.ARRAY_NAMES:
        np.testing.assert_array_equal(getattr(whole, k), getattr(resumed, k))
    for k, value in rng.items(): np.testing.assert_array_equal(RUN.capture_rng()[k], value)


def test_scalar_and_contour_formulas_and_eigenvalue_only_branch(monkeypatch):
    rng = np.random.default_rng(121)
    u, _ = np.linalg.qr(rng.normal(size=(8, 8)) + 1j*rng.normal(size=(8, 8)))
    nu = np.array([0, .1, .2, .4, .55, .8, .9, 1.])
    c = (u * nu) @ u.conj().T
    g = torch.as_tensor(np.stack([2*c-np.eye(8)]*3), dtype=torch.complex128)
    result = OBS.covariance_observables(g, nx=2, ny=2, sample_chunk=2)
    clipped = np.clip(nu, 1e-12, 1-1e-12)
    h = -clipped*np.log(clipped)-(1-clipped)*np.log1p(-clipped)
    expected = ((abs(u)**2) @ h).reshape(2, 2, 2).sum(axis=-1).T
    np.testing.assert_allclose(result.entropy_contour[0], expected, atol=1e-12)
    np.testing.assert_allclose(result.charge_variance_contour[0], np.diag(c-c@c).real.reshape(2,2,2).sum(-1).T, atol=1e-12)
    monkeypatch.setattr(torch.linalg, 'eigh', lambda *a, **kw: pytest.fail('alpha3 computed eigenvectors'))
    scalar = OBS.covariance_observables(g, nx=2, ny=2, sample_chunk=1, save_contours=False)
    np.testing.assert_allclose(scalar.occupation_spectrum, result.occupation_spectrum, atol=1e-13)
    np.testing.assert_allclose(scalar.total_entropy, result.total_entropy, atol=1e-12)
    assert scalar.entropy_contour is None


def test_minimum_magnitude_mode_and_degeneracy():
    nu = np.array([0., .1, .2, .4, .55, .8, .9, 1.])
    g = np.diag(2*nu-1).astype(np.complex128)[None]
    modes = OBS.extract_slowest_modes(g, nx=2, ny=2, cycles=4, device='cpu', sample_chunk=1)
    assert modes['slow_mode_spectrum_index'][0] == 4
    assert modes['slow_mode_signed_rate'][0] < 0
    assert modes['slow_mode_min_abs_multiplicity'][0] == 1
    np.testing.assert_allclose(modes['slow_mode_signed_rate'][0], np.log(.45/.55)/8)
    np.testing.assert_allclose(modes['slow_mode_density'].sum(), 1)
    assert modes['slow_mode_vector'].dtype == np.complex128
    tied = OBS.extract_slowest_modes(np.zeros_like(g), nx=2, ny=2, cycles=4, device='cpu', sample_chunk=1)
    assert tied['slow_mode_min_abs_multiplicity'][0] == 8
    capped = OBS.extract_slowest_modes(np.eye(8,dtype=np.complex128)[None], nx=2, ny=2, cycles=4, device='cpu', sample_chunk=1)
    assert not capped['slow_mode_resolved'][0] and np.isnan(capped['slow_mode_abs_rate'][0])


@pytest.mark.parametrize('alpha', [1, 3])
def test_final_checkpoint_publication_recovery_and_receipts(tmp_path, monkeypatch, alpha):
    monkeypatch.setattr(RUN, 'SEGMENT_CYCLES', 2)
    task = tiny(alpha)
    cfg = {**RUN.expected_config(), 'device': 'cpu', 'observer_sample_chunk_by_Ny': {'4': 2}}
    obs = observer(task)
    g = RUN.run_segment(model(alpha), cfg, task, obs, completed_cycle=0,
        segment_cycles=4, G_init=None, continuing=False, progress_bar=Bar())
    out, scratch = tmp_path/'out', tmp_path/'scratch'
    RUN.save_checkpoint(out, scratch, task, completed_cycle=4, elapsed_seconds=1,
        G=g, observer=obs, cfg_hash='cfg', hashes={})
    paths = RUN.checkpoint_paths(out, task)
    before = [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
    wrong, reason = RUN.load_checkpoint(out, task, cfg_hash='old-config', hashes={})
    assert wrong is None and 'identity mismatch' in reason
    publish = RUN.publish_result
    monkeypatch.setattr(RUN, 'build_model', lambda *a: None)
    monkeypatch.setattr(RUN, 'run_segment', lambda *a, **kw: pytest.fail('final checkpoint reran dynamics'))
    monkeypatch.setattr(RUN, 'publish_result', lambda *a, **kw: (_ for _ in ()).throw(OSError('test failure')))
    with pytest.raises(OSError, match='test failure'):
        RUN.execute_batch(cfg, out, scratch, task, cfg_hash='cfg', hashes={}, shard_bar=Bar())
    assert [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths] == before
    monkeypatch.setattr(RUN, 'publish_result', publish)
    RUN.execute_batch(cfg, out, scratch, task, cfg_hash='cfg', hashes={}, shard_bar=Bar())
    assert not any(p.exists() for p in paths)
    shard = RUN.result_shards(task)[0]
    assert RUN.verified_complete(out, shard, cfg_hash='cfg', hashes={})[0]
    result, receipt = RUN.result_paths(out, shard)
    original = result.read_bytes()
    result.write_bytes(original + b'corrupt')
    assert not RUN.verified_complete(out, shard, cfg_hash='cfg', hashes={})[0]
    result.write_bytes(original)
    receipt.unlink()
    assert not RUN.verified_complete(out, shard, cfg_hash='cfg', hashes={})[0]


def test_publication_failed_readback(tmp_path, monkeypatch):
    local, dest = tmp_path/'local', tmp_path/'dest'
    local.write_bytes(b'new')
    dest.write_bytes(b'old')
    monkeypatch.setattr(RUN.shutil, 'copyfile', lambda a,b: Path(b).write_bytes(b'bad'))
    with pytest.raises(OSError, match='readback mismatch'): RUN.publish_file(local, dest)
    assert dest.read_bytes() == b'old'


def test_notebook_sources_config_and_registration():
    nb = json.loads((BUNDLE/'run_hard_wall_full_measurement_purification.ipynb').read_text())
    code = '\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code')
    for c in nb['cells']:
        if c['cell_type']=='code': ast.parse(''.join(c['source']))
    assert "runtime.unassign()\nprint('done')" in ''.join(nb['cells'][-1]['source'])
    for text in ('/content/', 'REPORT_ONLY', 'MAX_NEW_EXECUTION_BATCHES', 'A100', 'runner.main(args)'):
        assert text in code
    tree = ast.parse(code)
    node = next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CONFIG' for t in n.targets))
    assert ast.literal_eval(node.value) == RUN.expected_config()
    manifest = json.loads((BUNDLE/'deployment_manifest.json').read_text())
    for name, row in manifest['files'].items():
        raw = (BUNDLE/name).read_bytes()
        assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
    for name in ('classA_U1FGTN_gpu.py','occupied_frame_gpu.py'):
        assert (BUNDLE/'src'/name).read_bytes() == (ROOT/'src/fgtn'/name).read_bytes()
    layout = load(BUNDLE.parent/'bundle_layout.py', '_bundle21_layout_test')
    assert BUNDLE.name in layout.validate_bundle_layout(BUNDLE.parent)


def test_report_only_never_launches_gpu(tmp_path, monkeypatch):
    monkeypatch.setattr(RUN, 'build_model', lambda *a: pytest.fail('report constructed model'))
    report = RUN.run_campaign(RUN.expected_config(), construction='hard', output_root=tmp_path/'o',
        scratch_root=tmp_path/'s', report_only=True, max_new_execution_batches=None)
    assert report['durable_result_shards']==40 and report['pending_shards']==40
    assert report['execution_batches']==2 and report['workload_sample_cycles']==12000


@pytest.mark.parametrize('contours', [False, True])
def test_suspect_eigensolver_recheck_preserves_state_and_rng(monkeypatch, contours):
    g = torch.diag(torch.linspace(-1., 1., 8)).to(torch.complex128)[None]
    original = g.clone()
    rng = torch.get_rng_state().clone()
    reference = OBS.covariance_observables(g, nx=2, ny=2, sample_chunk=1, save_contours=contours)
    method = 'eigh' if contours else 'eigvalsh'
    primary = getattr(torch.linalg, method)
    def suspect(matrix):
        result = primary(matrix)
        values = result[0] if contours else result
        values[..., -1] += 2e-9
        return result
    monkeypatch.setattr(torch.linalg, method, suspect)
    with pytest.warns(RuntimeWarning, match='independent CPU recheck'):
        result = OBS.covariance_observables(g, nx=2, ny=2, sample_chunk=1, save_contours=contours)
    np.testing.assert_allclose(result.occupation_spectrum, reference.occupation_spectrum, atol=1e-14)
    np.testing.assert_allclose(result.total_entropy, reference.total_entropy, atol=1e-13)
    if contours:
        np.testing.assert_allclose(result.entropy_contour, reference.entropy_contour, atol=1e-13)
    assert torch.equal(g, original) and torch.equal(torch.get_rng_state(), rng)


def test_true_violation_is_not_clipped_or_hidden():
    # Old formatting reports max=1.000e+00 and disguises the actual failure.
    g = torch.eye(8, dtype=torch.complex128)[None] * (1 + 4e-9)
    original = g.clone()
    with pytest.raises(OBS.OccupationSpectrumError) as error:
        OBS.covariance_observables(g, nx=2, ny=2, sample_chunk=1, save_contours=False)
    assert error.value.diagnostics['bound_excess'] > 1e-9
    assert '1.00000000' in str(error.value) and 'excess=' in str(error.value)
    np.testing.assert_array_equal(error.value.covariance, g[0])
    assert torch.equal(g, original)


def test_pre_hotfix_checkpoint_and_completion_resume(tmp_path, monkeypatch):
    monkeypatch.setattr(RUN, 'SEGMENT_CYCLES', 2)
    task = tiny(3)
    cfg = {**RUN.expected_config(), 'device': 'cpu'}
    obs = observer(task)
    g = RUN.run_segment(model(3), cfg, task, obs, completed_cycle=0,
        segment_cycles=4, G_init=None, continuing=False, progress_bar=Bar())
    original_hashes = RUN.PRE_HOTFIX_SOURCE_HASHES.copy()
    current_hashes = RUN.source_hashes()
    RUN.save_checkpoint(tmp_path/'out', tmp_path/'scratch', task, completed_cycle=4,
        elapsed_seconds=1, G=g, observer=obs, cfg_hash='cfg', hashes=original_hashes)
    checkpoint, reason = RUN.load_checkpoint(tmp_path/'out', task, cfg_hash='cfg', hashes=current_hashes)
    assert reason == 'verified' and checkpoint.completed_cycle == 4
    shard = RUN.result_shards(task)[0]
    RUN.publish_result(tmp_path/'out', tmp_path/'scratch', shard, G_final=g, observer=obs,
        elapsed_seconds=1, cfg_hash='cfg', hashes=original_hashes, endpoint_modes=None)
    assert RUN.verified_complete(tmp_path/'out', shard, cfg_hash='cfg', hashes=current_hashes)[0]
    assert not RUN.verified_complete(tmp_path/'out', shard, cfg_hash='wrong', hashes=current_hashes)[0]
    for name in original_hashes:
        corrupt = {**original_hashes, name: 'unknown'}
        assert not RUN.source_identity_matches(corrupt, current_hashes)
    assert not RUN.source_identity_matches(original_hashes, {**current_hashes, 'src/classA_U1FGTN_gpu.py': 'new-engine'})


def test_true_failure_publishes_evidence_not_completion(tmp_path, monkeypatch):
    task = tiny(3)
    g = np.eye(32, dtype=np.complex128) * (1 + 4e-9)
    error = OBS.OccupationSpectrumError(dict(nu_min=0., nu_max=1+2e-9,
        bound_excess=2e-9, sample_offset=0, sample_index=0, cycle=3),
        g, np.ones(32)*(1+2e-9))
    monkeypatch.setattr(RUN, 'expand_execution_batches', lambda *a: [task])
    monkeypatch.setattr(RUN, 'require_runtime', lambda *a: None)
    monkeypatch.setattr(RUN, 'require_space', lambda *a: None)
    monkeypatch.setattr(RUN, 'execute_batch', lambda *a, **k: (_ for _ in ()).throw(error))
    out, scratch = tmp_path/'out', tmp_path/'scratch'
    paths = RUN.checkpoint_paths(out, task)
    paths[0].parent.mkdir(parents=True)
    for p in paths: p.write_bytes(b'preserve this checkpoint')
    with pytest.raises(OBS.OccupationSpectrumError):
        RUN.run_campaign(RUN.expected_config(), construction='hard', output_root=out,
            scratch_root=scratch, report_only=False, max_new_execution_batches=None)
    for p in paths: assert p.read_bytes() == b'preserve this checkpoint'
    report = json.loads((out/'diagnostics'/task.task_id/'occupation_failure.json').read_text())
    assert report['cycle'] == 3 and report['bound_excess'] == 2e-9
    with np.load(out/'diagnostics'/task.task_id/'occupation_failure.npz') as data:
        np.testing.assert_array_equal(data['G'], g)
    assert not list(out.rglob('*.complete.json'))
