"""Shell selection, dense control, exact action, saved-reference and resume checks."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.linalg import eigvals

ROOT = Path(__file__).resolve().parents[1]
FILE = ROOT / '00_WORKSPACE/CURRENT/matched_markov_lindblad_campaign/shell_alpha_spectral_v1/run_scan.py'
spec = importlib.util.spec_from_file_location('shell_alpha_gap', FILE)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_contract():
    tasks = runner.tasks()
    assert len(tasks) == len(set(tasks)) == 315
    assert runner.SHELLS == (1, 2, None)
    assert len({runner.folder(Path('/unused'), *t) for t in tasks}) == 315
    for i, ny, shell in tasks:
        cfg = runner.config(i, ny, shell)
        assert cfg['Nx'] == 20 and cfg['walls'] == [5, 15]
        assert cfg['nshell'] == shell and cfg['alpha_2'] == 30
        assert cfg['all_slabs_active'] and cfg['dw_truncation']
        assert cfg['sequence'] == 'raster_y' and cfg['dtype'] == 'complex128'
        assert cfg['perfect_correction'] and cfg['dephasing'] and not cfg['explicit_dynamics']


@pytest.mark.parametrize('shell', [1, 2, None])
def test_full_action_and_resume(tmp_path, shell):
    cfg = runner.config(10, 4, shell)
    model = runner.make_model(cfg)
    assert model.nshell == shell
    full, _ = runner.backend.spectral.construct_product(model, progress=False)
    direct = runner.backend.spectral.independent_action(model, np.eye(len(full), dtype=complex))
    np.testing.assert_allclose(full, direct, atol=1e-13)
    expected = max(abs(eigvals(full)))
    runner.run_case(tmp_path, 10, 4, shell)
    assert runner.verified(tmp_path, 10, 4, shell)
    receipt = runner.folder(tmp_path, 10, 4, shell) / 'completion.json'
    before = receipt.read_bytes()
    d = json.loads(before)['diagnostics']
    np.testing.assert_allclose(d['spectral_radius'], expected, atol=1e-12)
    runner.run_case(tmp_path, 10, 4, shell)
    assert receipt.read_bytes() == before
    receipt.unlink()
    assert not runner.verified(tmp_path, 10, 4, shell)


def test_shell_selection_changes_modes():
    modes = [runner.make_model(runner.config(10, 8, s)).WF_Am for s in runner.SHELLS]
    assert np.linalg.norm(modes[0] - modes[1]) > .1
    assert np.linalg.norm(modes[1] - modes[2]) > .1


def test_saved_alpha2_shell1_reference_and_checksum(tmp_path):
    runner.run_case(tmp_path, 10, 20, 1)
    path = runner.folder(tmp_path, 10, 20, 1)
    actual = json.loads((path / 'completion.json').read_text())
    old = FILE.parent.parent / 'fixed_width_alpha_spectral_v1/results/20261001T220546Z/Ny020_a10_2.0/completion.json'
    expected = json.loads(old.read_text())
    np.testing.assert_allclose(actual['diagnostics']['spectral_radius'],
                               expected['diagnostics']['spectral_radius'], atol=1e-11)
    with (path / 'spectrum.npz').open('ab') as handle:
        handle.write(b'checksum regression')
    assert not runner.verified(tmp_path, 10, 20, 1)
