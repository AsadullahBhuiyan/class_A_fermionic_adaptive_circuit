"""Exact contract and algebra tests for the versioned G4/G5 redesign."""

from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
CAMPAIGN=ROOT/"00_WORKSPACE/CURRENT/final_production_new_designs"
G4=CAMPAIGN/"04_maxmix_operator_cft"
G5=CAMPAIGN/"05_pure_tangent_stability"

def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path); module=importlib.util.module_from_spec(spec); sys.modules[name]=module; spec.loader.exec_module(module); return module

def test_redesign_contracts_are_independent_and_save_no_records():
    g4=json.loads((G4/"production_config.json").read_text()); g5=json.loads((G5/"production_config.json").read_text())
    assert g4["root_seed"] != g5["root_seed"]
    assert g4["contract"]["trajectory_record_saved"] is False
    assert g4["contract"]["tangent_tracked"] is False
    assert g5["contract"]["trajectory_record_saved"] is False
    assert g5["contract"]["finite_difference_states"] is False
    assert g5["contract"]["tangent_basis_mode"] == "pure_occupied_empty"
    assert g4["storage"]["production_shard_count"] == g5["storage"]["production_shard_count"] == 50

def test_g4_maxmix_normalization_and_fock_factorization():
    m=load(G4/"src/g4_observables.py","g4_observables_test")
    nu=np.asarray([0.2,0.7,0.5,0.0,1.0]); f=m.spectrum_factors(nu)
    assert np.array_equal(f["flip_charge"],np.asarray([1,-1,1,1,-1],dtype=np.int8))
    assert np.array_equal(f["cap_orientation"],np.asarray([0,0,0,-1,1],dtype=np.int8))
    finite=np.isfinite(f["amplitude_cost"]); np.testing.assert_allclose(f["amplitude_cost"][finite],.5*np.abs(np.log(nu[finite]/(1-nu[finite]))))
    levels,counts,_=m.lowest_charge_resolved_levels(f["amplitude_cost"],f["flip_charge"],sectors=(-1,0,1),levels_per_sector=8)
    assert levels[1,0] == 0.0
    assert np.all(counts>=1)
    norb=6; logp=-3.25; assert np.isclose(norb*np.log(2)+logp,np.log(2**norb*np.exp(logp)))

def test_g4_checkpoint_schedule_contains_required_prefixes():
    m=load(G4/"src/g4_observables.py","g4_schedule_test")
    for ny in (20,30,40,50,60):
        cycles=m.observation_cycles(ny)
        assert cycles==sorted(set(cycles)); assert 0 in cycles and 2*ny in cycles
        assert {ny,3*ny//2,2*ny}.issubset(cycles)
        assert set(range(min(16,ny//2)+1)).issubset(cycles)

def test_g4_maxmix_contours_and_logz_sum_exactly():
    m=load(G4/"src/g4_observables.py","g4_observer_test")
    observer=m.G4Observer(2,2,1,(0,),(),soft_modes=8,levels_per_sector=8)
    observer(cycle=0,G=torch.zeros((1,8,8),dtype=torch.complex128),batch_start=0,batch_count=1)
    arrays=observer.arrays(np.zeros((1,4)))
    np.testing.assert_allclose(arrays["total_entropy"],8*np.log(2))
    np.testing.assert_allclose(arrays["total_charge_variance"],2.0)
    np.testing.assert_allclose(arrays["entropy_contour"].sum(axis=(-2,-1)),arrays["total_entropy"])
    np.testing.assert_allclose(arrays["charge_uncertainty_contour"].sum(axis=(-2,-1)),arrays["total_charge_variance"])
    np.testing.assert_allclose(arrays["log_Z"],8*np.log(2))

def test_g5_physical_pairs_are_formed_before_truncation():
    m=load(G5/"src/g5_tangent_observer.py","g5_tangent_test")
    occupied=np.asarray([-8.0,-0.4,4.0]); empty=np.asarray([-4.1,0.39,8.0])
    rates=m.physical_pair_rates(occupied,empty,elapsed=2)
    indices,selected,boundary=m.select_slowest_pairs(rates,count=3)
    brute=sorted((abs(rates[i,j]),i,j,rates[i,j]) for i in range(3) for j in range(3))[:3]
    assert [(int(i),int(j)) for i,j in indices] == [(i,j) for _,i,j,_ in brute]
    np.testing.assert_allclose(selected,[v for *_,v in brute]); assert boundary>=0
    # The slowest pair is a cancellation of non-near-zero one-leg logs.
    assert tuple(indices[0]) == (0,2)

def test_notebooks_end_with_runtime_disconnect():
    expected="from google.colab import runtime\nruntime.unassign()\nprint('done')\n"
    for root in (G4,G5):
        notebook=json.loads((root/"run_production_bundle.ipynb").read_text())
        assert "".join(notebook["cells"][-1]["source"]) == expected

def test_engine_window_reset_preserves_frame_and_resets_products():
    from src.fgtn.classA_U1FGTN_gpu import classA_U1FGTN_gpu
    model=classA_U1FGTN_gpu(Nx=2,Ny=2,nshell=1,device="cpu",dtype="complex128")
    frame=torch.linalg.qr(torch.randn(8,8,dtype=torch.complex128)).Q.unsqueeze(0)
    state={
        "basis_mode":"pure_occupied_empty","frame":frame.clone(),
        "log_diag":torch.ones((1,8),dtype=torch.float64),"null_counts":torch.ones((1,8),dtype=torch.int64),
        "block_core_hat":(2*torch.eye(4,dtype=torch.complex128)[None],3*torch.eye(4,dtype=torch.complex128)[None]),
        "block_core_log_scale":(torch.ones(1,dtype=torch.float64),torch.ones(1,dtype=torch.float64)),
        "block_core_null_count":(torch.ones(1,dtype=torch.int64),torch.ones(1,dtype=torch.int64)),
    }
    model._reset_lyapunov_accumulator(state)
    torch.testing.assert_close(state["frame"],frame)
    assert not torch.any(state["log_diag"]) and not torch.any(state["null_counts"])
    for core,scale,nulls in zip(state["block_core_hat"],state["block_core_log_scale"],state["block_core_null_count"]):
        torch.testing.assert_close(core,torch.eye(4,dtype=torch.complex128)[None]); assert not torch.any(scale); assert not torch.any(nulls)

def test_g4_g5_preflight_rejects_non_a100_runtimes_by_default():
    for root, runner in ((G4, "g4_runner.py"), (G5, "g5_runner.py")):
        source = (root / "src" / runner).read_text(encoding="utf-8")
        assert "preflight requires an A100 GPU" in source
        assert "not in name.upper()" in source

