import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest

ROOT=Path(__file__).resolve().parents[1]
FILE=ROOT/'00_WORKSPACE/CURRENT/matched_markov_lindblad_campaign/square_2Ny_v1/run_square.py'
spec=importlib.util.spec_from_file_location('square_channel',FILE)
runner=importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_square_contract_and_matching_spectra():
    configs=[runner.config(a,n) for a in (1,3) for n in runner.SIZES]
    assert len(configs)==16
    assert [runner.config(1,n)['cycles'] for n in runner.SIZES]==[40,48,56,64,72,80,88,100]
    for cfg in configs:
        assert cfg['Nx']==cfg['Ny']
        assert cfg['walls']==[cfg['Nx']//4,3*cfg['Nx']//4]
        reference=runner.spectral_config(cfg['alpha_1'],cfg['Ny'])
        for key in ['Nx','Ny','walls','alpha_1','alpha_2','nshell','sequence','channel_order','dw_truncation','all_slabs_active']:
            assert cfg[key]==reference[key]
    assert runner.config(1,50)['walls']==[12,37]


def test_queue_does_not_pass_incomplete_predecessor(tmp_path,monkeypatch):
    assert runner.predecessor_complete(tmp_path) is False
    for a in (1,3):
        (tmp_path/f'worker_alpha{a}.json').write_text(json.dumps(dict(complete=8,failures=[])))
    monkeypatch.setattr(runner.dynamics,'inventory',lambda root:[dict(complete=False)])
    with pytest.raises(RuntimeError,match='verification'):
        runner.predecessor_complete(tmp_path)
    monkeypatch.setattr(runner.dynamics,'inventory',lambda root:[dict(complete=True)]*16)
    assert runner.predecessor_complete(tmp_path) is True
    (tmp_path/'worker_alpha1.json').write_text(json.dumps(dict(complete=7,failures=['bad'])))
    with pytest.raises(RuntimeError,match='failed/incomplete'):
        runner.predecessor_complete(tmp_path)


def test_small_square_case_publishes_and_resumes(tmp_path,monkeypatch):
    cfg=runner.config(1,20)
    cfg.update(Nx=4,Ny=4,walls=[1,3],cycles=8)
    spectral_cfg=runner.spectral_config(1,20)
    spectral_cfg.update(Nx=4,Ny=4,walls=[1,3])
    monkeypatch.setattr(runner,'config',lambda alpha,size:cfg)
    monkeypatch.setattr(runner,'spectral_config',lambda alpha,size:spectral_cfg)
    runner.run_case(tmp_path,1,4)
    assert runner.verified(tmp_path,1,4)
    path=runner.case_dir(tmp_path,1,4)/'dynamics.npz'
    first_hash=runner.sha(path)
    first_mtime=path.stat().st_mtime_ns
    runner.run_case(tmp_path,1,4)
    assert runner.sha(path)==first_hash
    assert path.stat().st_mtime_ns==first_mtime


@pytest.mark.parametrize('mode', ['case', 'worker', 'report'])
def test_cli_under_single_core_inherited_affinity(tmp_path, mode):
    """Real subprocess reproduces the worker's restricted inherited affinity."""
    cpu=min(os.sched_getaffinity(0))
    script='''
import importlib.util, json, os, sys
from pathlib import Path
file, root, cpu, mode=sys.argv[1:]
cpu=int(cpu)
os.sched_setaffinity(0,{cpu})
spec=importlib.util.spec_from_file_location('square_cli_test',file)
runner=importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
def dispatched(*args):
    print(json.dumps({'dispatched':mode,'affinity':sorted(os.sched_getaffinity(0))}))
    return 0
runner.run_case=dispatched
runner.worker=dispatched
sys.argv=[file,mode,'--root',root,'--alpha','1','--size','20','--cpu',str(cpu)]
raise SystemExit(runner.main())
'''
    result=subprocess.run([sys.executable,'-c',script,str(FILE),str(tmp_path),str(cpu),mode],
                          capture_output=True,text=True,timeout=45)
    assert result.returncode==0,result.stdout+result.stderr
    if mode!='report':
        assert json.loads(result.stdout.splitlines()[-1])==dict(dispatched=mode,affinity=[cpu])
