"""Square-grid contract and completion-only restart regression tests."""
import importlib.util
from pathlib import Path
import json

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT/'00_WORKSPACE/CURRENT/matched_markov_lindblad_campaign/square_steady_purity_gap_v1/run_square.py'
spec = importlib.util.spec_from_file_location('square_purity_test', PATH)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_grid_and_scientific_contract():
    assert len(runner.SIZES)*len(runner.ALPHAS) == 14
    assert runner.SIZES == (20,30,40,50,60,70,80)
    for a in runner.ALPHAS:
        for n in runner.SIZES:
            cfg = runner.config(a,n)
            assert cfg['Nx'] == cfg['Ny'] == n
            assert cfg['walls'] == [n//4,3*n//4]
            old = runner.parent.config(a,n)
            for key in ('cycles','observation_cycles','dtype','dephasing','perfect_correction',
                        'all_slabs_active','alpha_2','nshell','sequence','init_mode'):
                assert cfg[key] == old[key]
    assert runner.config(1,20)['walls'] == [5,15]


def test_case_resume_and_corrupt_output(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(runner.parent.spectral, 'make_model', lambda cfg: object())
    def collect(model, cfg):
        calls.append(cfg)
        return {'C_final':np.eye(3,dtype=np.complex128)}, {'converged':True}
    monkeypatch.setattr(runner.parent, 'collect', collect)
    runner.run_case(tmp_path,1,20)
    assert runner.verified(tmp_path,1,20)
    runner.run_case(tmp_path,1,20)
    assert len(calls) == 1
    result = runner.folder(tmp_path,1,20)/'dynamics.npz'
    result.write_bytes(b'corrupt')
    assert not runner.verified(tmp_path,1,20)
    runner.run_case(tmp_path,1,20)
    assert len(calls) == 2
    receipt = result.parent/'completion.json'
    rec = json.loads(receipt.read_text())
    rec['status'] = 'not_converged'
    receipt.write_text(json.dumps(rec))
    assert not runner.verified(tmp_path,1,20)
