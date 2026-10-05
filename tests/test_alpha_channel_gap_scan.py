from pathlib import Path
import importlib.util
import json
import os
import subprocess
import sys
import numpy as np
import pytest
from scipy.linalg import eigvals

ROOT = Path(__file__).resolve().parents[1]
FILE = ROOT/'00_WORKSPACE/CURRENT/matched_markov_lindblad_campaign/alpha_scan_spectral_v1/run_scan.py'
spec = importlib.util.spec_from_file_location('alpha_channel_gap_scan',FILE)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_grid_and_balanced_work():
    assert runner.ALPHAS == tuple(i/10 for i in range(10,31))
    cases = runner.tasks()
    assert len(cases) == len(set(cases)) == 63
    paths = [runner.folder(Path('/unused'),i,n) for i,n in cases]
    assert len(set(paths)) == 63
    for k in range(3):
        assigned = cases[k::3]
        assert len(assigned) == 21
        assert all(sum(n==size for _,n in assigned)==7 for size in runner.SIZES)
    for i,n in cases:
        cfg = runner.config(i,n)
        assert cfg['alpha_1']==runner.ALPHAS[i] and cfg['alpha_2']==30
        assert cfg['Nx']==cfg['Ny']==n
        assert cfg['walls']==[n//4,3*n//4]
        assert cfg['all_slabs_active'] and cfg['dw_truncation']
        assert cfg['sequence']=='raster_y' and cfg['dtype']=='complex128'
        assert not cfg['explicit_dynamics']


@pytest.mark.parametrize('index',[1,10,19])
def test_noninteger_and_critical_alpha_against_full_matrix(tmp_path,index):
    # Includes exact alpha=2 with canonical normalization; no parameter shift.
    cfg = runner.config(index,4)
    model = runner.backend.spectral.make_model(cfg)
    full,_ = runner.backend.spectral.construct_product(model,progress=False)
    expected = float(-2*np.log(np.max(abs(eigvals(full)))))
    runner.run_case(tmp_path,index,4)
    assert runner.verified(tmp_path,index,4)
    receipt_path = runner.folder(tmp_path,index,4)/'completion.json'
    before = receipt_path.read_bytes()
    receipt = json.loads(before)
    np.testing.assert_allclose(receipt['diagnostics']['covariance_gap_raw'],expected,atol=1e-11)
    assert receipt['config']['alpha_1']==runner.ALPHAS[index]
    runner.run_case(tmp_path,index,4)
    assert receipt_path.read_bytes()==before
    with (receipt_path.parent/'spectrum.npz').open('ab') as handle:
        handle.write(b'bad')
    assert not runner.verified(tmp_path,index,4)


def test_report_under_single_core_affinity(tmp_path):
    cpu = min(os.sched_getaffinity(0))
    result = subprocess.run(['taskset','-c',str(cpu),sys.executable,str(FILE),'report','--root',str(tmp_path)],
                             capture_output=True,text=True,check=True)
    data = json.loads(result.stdout)
    assert data['total']==63 and data['complete']==0
