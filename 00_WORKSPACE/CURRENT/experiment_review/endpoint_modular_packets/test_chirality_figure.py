import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import analyze_chirality_figure as reduction
import analyze_endpoint_packets as base


def test_optional_probability_preserves_existing_outputs(monkeypatch):
    rng = np.random.default_rng(818)
    n, length, nx = 32, 4, 4
    vectors, _ = np.linalg.qr(rng.normal(size=(n,n)) + 1j*rng.normal(size=(n,n)))
    vals = np.linspace(-.99, .99, n)
    kwargs = dict(nx=nx, length=length, walls=(0,2), y_start=2,
                  times=np.array([0., .1, .2]), snapshot_times=np.array([0., .2]))
    old = base.evolve_spectrum(vals, vectors, **kwargs)
    new = base.evolve_spectrum(vals, vectors, **kwargs, return_longitudinal=True)
    assert 'longitudinal_probability' not in old
    for key in old:
        np.testing.assert_array_equal(old[key], new[key])
    p = new['longitudinal_probability']
    np.testing.assert_allclose(p.sum(axis=-1), 1, atol=1e-12, rtol=0)
    np.testing.assert_allclose(p @ (np.arange(length)-2), new['dy_window'], atol=1e-12, rtol=0)
    monkeypatch.setattr(base, 'LENGTH', length)
    monkeypatch.setattr(base, 'Y_START', 2)
    assert reduction.validate_probability(p, new['dy_window']) < 1e-12


def test_normalize_before_averaging():
    # Different retained charges: pooling density would weight trajectories unequally.
    density = np.array([[2.,0.], [0.,.2]])
    p = density/density.sum(axis=1, keepdims=True)
    mean = p.mean(axis=0)
    np.testing.assert_allclose(mean @ np.arange(2), (p @ np.arange(2)).mean())
    assert abs(mean[1] - (density.mean(axis=0)/density.mean(axis=0).sum())[1]) > .4


def test_contrast_sem_uses_paired_walls():
    common = np.arange(100, dtype=float)[:,None]
    dy = np.stack((common+1, common-1), axis=1)
    mean, sem = reduction.mean_sem(reduction.contrast_samples(dy))
    np.testing.assert_array_equal(mean, [1.])
    np.testing.assert_array_equal(sem, [0.])
    assert reduction.mean_sem(dy[:,0])[1][0] > 0


def test_completion_validation(tmp_path):
    cache = tmp_path/'sample.npz'; identity = {'sample':0}
    base.atomic_npz(cache, example=np.arange(5))
    assert not reduction.cache_valid(cache, identity)
    receipt = dict(identity=identity, filename=cache.name, bytes=cache.stat().st_size,
                   sha256=base.digest(cache))
    cache.with_suffix('.json').write_text(json.dumps(receipt))
    assert reduction.cache_valid(cache, identity)
    assert not reduction.cache_valid(cache, {'sample':1})
    original = cache.read_bytes()
    cache.write_bytes(original[:-1]+bytes([original[-1]^1]))
    assert not reduction.cache_valid(cache, identity)
    cache.write_bytes(original)
    receipt['filename']='wrong.npz'
    cache.with_suffix('.json').write_text(json.dumps(receipt))
    assert not reduction.cache_valid(cache, identity)
    cache.with_suffix('.json').write_text('{partial')
    assert not reduction.cache_valid(cache, identity)


def test_worker_failure_does_not_publish_completion(tmp_path, monkeypatch):
    monkeypatch.setattr(base, 'load_frames', lambda _: (np.full((1,4,2), np.nan+0j), [2]))
    cache=tmp_path/'failed.npz'
    with pytest.raises(ValueError, match='Invalid endpoint'):
        reduction.analyze_sample(('unused',0,1,0,str(cache),{'sample':0}))
    assert not cache.exists() and not cache.with_suffix('.json').exists()


def test_reused_cache_does_not_reload_frame(tmp_path, monkeypatch):
    cache=tmp_path/'complete.npz'; identity={'sample':0}
    base.atomic_npz(cache, example=[1])
    cache.with_suffix('.json').write_text(json.dumps(dict(identity=identity,
        filename=cache.name, bytes=cache.stat().st_size, sha256=base.digest(cache))))
    def fail(_):
        raise AssertionError('A valid completed trajectory must not be recomputed')
    monkeypatch.setattr(base, 'load_frames', fail)
    assert reduction.analyze_sample(('unused',0,1,0,str(cache),identity)) == 'reused'


def test_source_checksums_are_required(tmp_path, monkeypatch):
    monkeypatch.setattr(base, 'SOURCE', tmp_path)
    folder=tmp_path/'hard/Ny032/alpha1_1'
    folder.mkdir(parents=True)
    source=folder/'source.npz'
    base.atomic_npz(source, example=[1])
    receipt=dict(result_filename=source.name, result_bytes=source.stat().st_size,
                 result_sha256='bad-checksum')
    source.with_suffix('.complete.json').write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match='Source checksum mismatch'):
        base.collect_jobs(tmp_path/'analysis', 2)
    receipt['result_bytes']+=1
    source.with_suffix('.complete.json').write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match='Invalid source completion'):
        base.collect_jobs(tmp_path/'analysis', 2)
