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
FILE = ROOT/'00_WORKSPACE/CURRENT/matched_markov_lindblad_campaign/fixed_width_alpha_spectral_v1/run_scan.py'
spec = importlib.util.spec_from_file_location('fixed_width_gap_scan',FILE)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_grid_and_one_alpha_per_worker():
    assert runner.SIZES == (20,40,60,80,100)
    assert runner.ALPHAS == tuple(i/10 for i in range(10,31))
    cases = runner.tasks()
    assert len(cases) == len(set(cases)) == 105
    assert len({runner.folder(Path('/unused'),i,n) for i,n in cases})==105
    for index in range(21):
        assert cases[index::21] == [(index,n) for n in runner.SIZES]
    for i,n in cases:
        cfg = runner.config(i,n)
        assert cfg['Nx']==20 and cfg['Ny']==n and cfg['walls']==[5,15]
        assert cfg['alpha_1']==runner.ALPHAS[i] and cfg['alpha_2']==30
        assert cfg['all_slabs_active'] and cfg['dw_truncation']
        assert cfg['sequence']=='raster_y' and cfg['dtype']=='complex128'
        assert not cfg['explicit_dynamics'] and cfg['nshell']==1


@pytest.mark.parametrize('index',[1,10,19])
def test_rectangular_against_full_matrix(tmp_path,index):
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
    assert sum(receipt['diagnostics']['block_dimensions'])==160
    assert receipt['diagnostics']['peak_rss_gib']>0
    with np.load(receipt_path.parent/'spectrum.npz',allow_pickle=False) as data:
        assert data['eigenvalues'].shape==(160,)
        assert data['dominant_eigenvector'].shape==(160,)
    runner.run_case(tmp_path,index,4)
    assert receipt_path.read_bytes()==before
    with (receipt_path.parent/'spectrum.npz').open('ab') as handle:
        handle.write(b'bad')
    assert not runner.verified(tmp_path,index,4)


def test_saved_full_matrix_reference(tmp_path):
    runner.run_case(tmp_path,0,20)
    actual = json.loads((runner.folder(tmp_path,0,20)/'completion.json').read_text())
    old = ROOT/'00_WORKSPACE/CURRENT/matched_markov_lindblad_campaign/spectral_gap_v1/results/20260929T191448Z/alpha1_Ny020/completion.json'
    expected = json.loads(old.read_text())
    np.testing.assert_allclose(actual['diagnostics']['covariance_gap_raw'],
                               expected['diagnostics']['covariance_gap_raw'],atol=1e-11)


def test_report_under_one_core(tmp_path):
    cpu = min(os.sched_getaffinity(0))
    result = subprocess.run(['taskset','-c',str(cpu),sys.executable,str(FILE),'report','--root',str(tmp_path)],
                             capture_output=True,text=True,check=True)
    data = json.loads(result.stdout)
    assert data['total']==105 and data['complete']==0
