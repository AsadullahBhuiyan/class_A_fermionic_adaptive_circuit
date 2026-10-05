from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

EXPERIMENT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT.parents[2]
sys.path[:0] = [str(EXPERIMENT), str(REPO_ROOT / "src")]

from fgtn.classA_U1FGTN import classA_U1FGTN
from reference_probe import ReferencePairObserver


def test_canonical_engine_continues_with_four_spectator_reference_modes():
    model = classA_U1FGTN(
        4, 4, DW=True, nshell=1, alpha_1=1, alpha_2=30, trial_orbitals="X", dw_truncation=True
    )
    model.construct_OW_projectors(
        nshell=1, DW=True, trial_orbitals="X", dw_truncation=True
    )
    wall = int(model.DW_loc[0])
    observer = ReferencePairObserver(
        nx=4,
        ny=4,
        tau1=1,
        tau2=2,
        follow_cycles=2,
        first_site=(wall, 0),
        second_site=(wall, 0),
        rng=np.random.default_rng(9),
    )
    result = model.run_markov_circuit(
        cycles=4,
        samples=1,
        sequence="random",
        perfect_correction=True,
        G_history=False,
        save=False,
        progress=False,
        random_seed=11,
        state_representation="physical_frame",
        return_native_state=True,
        native_cycle_observer=observer,
        meas_slab_only=True,
    )
    observer.assert_complete()
    final = result["native_final"]
    assert final["physical_dimension"] == 2 * 4 * 4 + 4
    assert final["frame"].shape[0] == 2 * 4 * 4 + 4
    assert final["gram_residual"] < 1e-8
    assert np.all(observer.mutual_information >= 0.0)
