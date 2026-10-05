"""Independent small-system checks for deterministic bundle 24."""
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.linalg import block_diag, eigh

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT/'00_WORKSPACE/CURRENT/final_production_new_designs/24_flattened_imaginary_time_density'
sys.path.insert(0, str(BUNDLE))


def load(name):
    spec = importlib.util.spec_from_file_location('imaginary_test_'+name, BUNDLE/(name+'.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


science = load('density_correlations')
runner = load('run_campaign')


def random_blocks(ny=3, q=4):
    rng = np.random.default_rng(104)
    blocks = []
    for k in range(ny):
        u, _ = np.linalg.qr(rng.normal(size=(q,q))+1j*rng.normal(size=(q,q)))
        energies = np.linspace(-1, 1, q) + .02*k
        blocks.append((u*energies)@u.conj().T)
    return np.array(blocks)


def lift(blocks):
    ny, q, _ = blocks.shape
    f = np.exp(2j*np.pi*np.arange(ny)[:, None]*np.arange(ny)/ny)/np.sqrt(ny)
    transform = np.kron(f, np.eye(q))
    return transform @ block_diag(*blocks) @ transform.conj().T


def dense_weights(h, selected):
    e, u = eigh(h)
    half = len(e)//2
    overlap = u[selected, :half].conj().T @ u[selected, half:]
    return (e[half:][None,:]-e[:half,None]).ravel(), (abs(overlap)**2).ravel()


def test_momentum_correlations_vs_dense_and_equal_time():
    blocks = random_blocks()
    e, u, occupied, gap = science.eigensystem(blocks)
    gaps, local, column, number = science.transition_weights(e, u, occupied)
    assert gap > 0 and number < 1e-25
    tau = np.array([0, .01, 1, 10, 1000])
    lc = science.evaluate(gaps, local, tau, progress=False)
    cc = science.evaluate(gaps, column, tau, progress=False)
    h = lift(blocks)
    ny, q, _ = blocks.shape
    for x in range(q//2):
        expected = []
        for y in range(ny):
            g, w = dense_weights(h, [y*q+2*x, y*q+2*x+1])
            expected.append(np.exp(-tau[:,None]*g)@w)
        np.testing.assert_allclose(lc[:,x], np.mean(expected, axis=0), atol=2e-14)
        g, w = dense_weights(h, [y*q+2*x+a for y in range(ny) for a in (0,1)])
        np.testing.assert_allclose(cc[:,x], np.exp(-tau[:,None]*g)@w/ny, atol=2e-14)
    static, diagnostics = science.covariance_checks(u, occupied)
    np.testing.assert_allclose(lc[0], static['equal_time_local_variance'], atol=2e-14)
    np.testing.assert_allclose(cc[0], static['equal_time_column_variance_per_Ny'], atol=2e-14)
    np.testing.assert_allclose(static['spatial_connected'][:,1:],
                               -2*static['spatial_legacy_positive'][:,1:], atol=0)
    assert diagnostics['projector_idempotency_max_abs'] < 1e-13
    assert (lc >= 0).all() and (np.diff(lc, axis=0) <= 1e-14).all()
    np.testing.assert_allclose(science.evaluate(gaps, local, tau, chunk=1, progress=False), lc, atol=1e-15)


def test_regulated_parent_blocks_equal_dense_sum_of_translated_columns():
    config = runner.default_config() | dict(Nx=8, Ny=4)
    blocks, error = science.parent_blocks(config)
    nx, ny = config['Nx'], config['Ny']
    model = science.classA_U1FGTN(Nx=nx, Ny=ny, DW=True, nshell=1,
                                 filling_frac=.5, alpha_1=1, alpha_2=30,
                                 trial_orbitals='X', dw_truncation=True)
    model.construct_OW_projectors(nshell=1, DW=True, trial_orbitals='X', dw_truncation=True)
    h = np.zeros((2*nx*ny, 2*nx*ny), complex)
    phase = np.exp(-1j*config['occupation_twist']*np.arange(ny)/ny)
    for name, sign in [('WF_Ap',1), ('WF_Bp',1), ('WF_Am',-1), ('WF_Bm',-1)]:
        base = getattr(model, name).reshape(ny,2*nx,nx,ny)[:,:,:,0]*phase[:,None,None]
        for shift in range(ny):
            translated = np.roll(base, shift, axis=0).reshape(2*nx*ny,nx)
            h += sign*translated@translated.conj().T
    np.testing.assert_allclose(lift(blocks), h, atol=2e-13)
    np.testing.assert_allclose(np.sort(np.linalg.eigvalsh(blocks).ravel()),
                               np.linalg.eigvalsh(h), atol=2e-13)
    assert error < 1e-12


def test_particle_hole_formula_against_exact_fock_space():
    h = random_blocks(ny=1)[0]
    n = len(h)
    annihilation = []
    for i in range(n):
        c = np.zeros((2**n,2**n), complex)
        for state in range(2**n):
            if state & (1<<i):
                c[state^(1<<i),state] = (-1)**((state & ((1<<i)-1)).bit_count())
        annihilation.append(c)
    many = sum(h[i,j]*annihilation[i].conj().T@annihilation[j] for i in range(n) for j in range(n))
    energies, states = eigh(many)
    ground = states[:,0]
    density = sum(annihilation[i].conj().T@annihilation[i] for i in (0,1))
    fluctuation = (density-np.vdot(ground,density@ground)*np.eye(2**n))@ground
    amplitudes = states.conj().T@fluctuation
    tau = np.array([0., .2, 1., 10.])
    exact = np.exp(-tau[:,None]*(energies-energies[0]))@abs(amplitudes)**2
    gaps, weights = dense_weights(h, [0,1])
    np.testing.assert_allclose(np.exp(-tau[:,None]*gaps)@weights, exact, atol=3e-14)


def test_zero_variance_normalization_is_explicit():
    values = np.array([[0., 1e-20, 2.], [0., 1e-21, 1.]])
    result, valid = science.normalized(values, 1e-12)
    np.testing.assert_array_equal(valid, [False, False, True])
    assert np.isnan(result[:,:2]).all()
    np.testing.assert_array_equal(result[:,2], [1., .5])


def test_completion_checksum_identity_and_partial_pair(tmp_path):
    config = runner.default_config()
    ident = runner.identity(config)
    output, scratch = tmp_path/'out', tmp_path/'scratch'
    result = runner.publish(dict(test=np.arange(4)), {}, config, ident, output, scratch)
    assert runner.verified(output, ident)
    assert not runner.verified(output, ident | dict(seed=123))
    receipt = result.with_suffix('.complete.json')
    saved = receipt.read_bytes()
    receipt.unlink()
    assert not runner.verified(output, ident)
    receipt.write_bytes(saved)
    with result.open('ab') as f:
        f.write(b'bad')
    assert not runner.verified(output, ident)


def test_failed_readback_does_not_publish_receipt(tmp_path, monkeypatch):
    config = runner.default_config()
    ident = runner.identity(config)
    original = runner.shutil.copyfile
    def corrupt(source, target):
        original(source, target)
        with Path(target).open('ab') as f:
            f.write(b'bad')
    monkeypatch.setattr(runner.shutil, 'copyfile', corrupt)
    with pytest.raises(OSError, match='Readback mismatch'):
        runner.publish(dict(test=np.arange(4)), {}, config, ident, tmp_path/'out', tmp_path/'scratch')
    result, receipt = runner.paths(tmp_path/'out', ident)
    assert not result.exists() and not receipt.exists()


def test_completion_skips_calculation_and_report_only_writes_nothing(tmp_path, monkeypatch):
    import density_correlations
    import plot_results
    def forbidden(*args, **kwargs):
        raise AssertionError('Completed calculation must not rerun')
    monkeypatch.setattr(density_correlations, 'calculate', forbidden)
    monkeypatch.setattr(plot_results, 'plot_all', lambda result, out: out.mkdir(parents=True, exist_ok=True))
    config = runner.default_config()
    runner.run(config, tmp_path/'absent', tmp_path/'scratch', report_only=True)
    assert not (tmp_path/'absent').exists() and not (tmp_path/'scratch').exists()
    ident = runner.identity(config)
    output, scratch = tmp_path/'out', tmp_path/'scratch'
    result = runner.publish(dict(test=np.arange(4)), {}, config, ident, output, scratch)
    assert runner.run(config, output, scratch) == result


def test_notebook_and_canonical_source_identity():
    import nbformat
    nb = nbformat.read(BUNDLE/'run_flattened_imaginary_time_density.ipynb', as_version=4)
    nbformat.validate(nb)
    source = '\n'.join(c.source for c in nb.cells)
    for c in nb.cells:
        if c.cell_type == 'code':
            compile(c.source, '<notebook>', 'exec')
    assert 'stdout.buffer' not in source and "decoder.decode(chunk)" in source
    assert '/content/' in source and 'REPORT_ONLY' in source and 'CPU_THREADS' in source
    assert nb.cells[-1].source == "from google.colab import runtime\nruntime.unassign()\nprint('done')\n"
    for name in ('classA_U1FGTN.py', 'occupied_frame.py'):
        assert (BUNDLE/'src'/name).read_bytes() == (ROOT/'src/fgtn'/name).read_bytes()
