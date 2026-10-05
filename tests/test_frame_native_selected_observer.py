from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
SHARED = ROOT / "00_WORKSPACE/CURRENT/final_production_ready_figure_scripts/_shared_src"
STREAMING = ROOT / "00_WORKSPACE/COLAB/colab_charge_fluctuations/src"
for path in (SHARED, STREAMING):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from selected_observables import SelectedCovarianceObserver  # noqa: E402
from h3_twist_observables import FinalEntanglementObserver  # noqa: E402
from src.fgtn.occupied_frame_gpu import BatchedOccupiedFrameState  # noqa: E402
from src.fgtn.classA_U1FGTN_gpu import classA_U1FGTN_gpu  # noqa: E402


def test_selected_observer_frame_and_dense_projector_formulas_agree():
    nx, ny, samples = 4, 4, 2
    dimension = 2 * nx * ny
    state = BatchedOccupiedFrameState.random_pure(
        samples,
        dimension,
        dimension // 2,
        device="cpu",
        dtype=torch.complex128,
        generator=torch.Generator(device="cpu").manual_seed(877),
    )
    centered = 2.0 * (state.frame @ state.frame.mH) - torch.eye(
        dimension, dtype=torch.complex128
    )
    kwargs = dict(
        nx=nx,
        ny=ny,
        samples=samples,
        physical_cycles=1,
        observation_cycles=[0, 1],
        strip_entropy_cycles=[1],
        local_marker_cycles=[1],
        bott_cycles=[1],
    )
    frame_observer = SelectedCovarianceObserver(**kwargs)
    dense_observer = SelectedCovarianceObserver(**kwargs)
    for cycle in (0, 1):
        frame_observer(
            cycle=cycle, state=state, batch_start=0, batch_count=samples
        )
        dense_observer(
            cycle=cycle, G=centered, batch_start=0, batch_count=samples
        )
    frame_observer.validate()
    dense_observer.validate()
    np.testing.assert_allclose(frame_observer.density, dense_observer.density, atol=2e-12)
    np.testing.assert_allclose(
        frame_observer.real_space_chern,
        dense_observer.real_space_chern,
        atol=2e-10,
    )
    np.testing.assert_allclose(
        frame_observer.local_marker[1], dense_observer.local_marker[1], atol=2e-10
    )
    np.testing.assert_allclose(
        frame_observer.strip_entropy[1], dense_observer.strip_entropy[1], atol=2e-9
    )
    np.testing.assert_allclose(
        frame_observer.correlator[1], dense_observer.correlator[1], atol=2e-12
    )
    np.testing.assert_allclose(
        frame_observer.bott_index, dense_observer.bott_index, atol=2e-9
    )
    np.testing.assert_allclose(
        frame_observer.successive_covariance_frobenius_per_dimension,
        dense_observer.successive_covariance_frobenius_per_dimension,
        atol=2e-12,
    )
    assert state.materialization_count == 0


def test_h3_final_entanglement_uses_restricted_frame_rows():
    nx, ny, samples = 4, 4, 2
    dimension = 2 * nx * ny
    state = BatchedOccupiedFrameState.random_pure(
        samples,
        dimension,
        dimension // 2,
        device="cpu",
        dtype=torch.complex128,
        generator=torch.Generator(device="cpu").manual_seed(142),
    )
    centered = 2.0 * (state.frame @ state.frame.mH) - torch.eye(
        dimension, dtype=torch.complex128
    )
    kwargs = dict(
        samples=samples,
        nx=nx,
        ny=ny,
        final_cycle=0,
        tracked_modes=4,
        wall_half_width=1,
    )
    native = FinalEntanglementObserver(**kwargs)
    dense = FinalEntanglementObserver(**kwargs)
    native(cycle=0, state=state, batch_start=0, batch_count=samples)
    dense(cycle=0, G=centered, batch_start=0, batch_count=samples)
    native_values, _, native_weights, native_gap = native.payload()
    dense_values, _, dense_weights, dense_gap = dense.payload()
    np.testing.assert_allclose(native_values, dense_values, atol=2e-12)
    np.testing.assert_allclose(native_weights, dense_weights, atol=2e-10)
    np.testing.assert_allclose(native_gap, dense_gap, atol=2e-12)
    assert state.materialization_count == 0


def test_runner_selected_observer_keeps_complete_cycle_axis_without_materialization():
    nx = ny = 4
    cycles, samples = 1, 2
    observer = SelectedCovarianceObserver(
        nx=nx,
        ny=ny,
        samples=samples,
        physical_cycles=cycles,
        observation_cycles=range(cycles + 1),
        strip_entropy_cycles=range(cycles + 1),
        local_marker_cycles=range(cycles + 1),
        bott_cycles=range(cycles + 1),
    )
    model = classA_U1FGTN_gpu(
        nx, ny, DW=False, nshell=1, device="cpu", dtype="complex128"
    )
    result = model.run_markov_circuit(
        cycles=cycles,
        samples=samples,
        batch_size=samples,
        G_history=False,
        save=False,
        progress=False,
        return_data=False,
        state_representation="auto",
        native_cycle_observer=observer,
        require_no_covariance_materialization=True,
    )
    observer.validate()
    assert observer.cycles == [0, 1]
    assert result["state_representation_resolved"] == "physical_frame"
    assert result["covariance_materialization_count"] == 0
