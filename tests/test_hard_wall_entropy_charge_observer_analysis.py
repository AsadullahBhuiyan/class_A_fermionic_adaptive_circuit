from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


REPO = Path(__file__).resolve().parents[1]
BUNDLE = REPO / "00_WORKSPACE/CURRENT/final_production_new_designs/05_hard_wall_entropy_charge_batched_v2"
sys.path.insert(0, str(BUNDLE))
from entropy_charge_observer import (  # noqa: E402
    HardWallEntropyChargeObserver,
    frame_fixed_half_strip_contours,
    frame_window_observables,
    frame_y0_averaged_width,
    periodic_window_indices,
    validate_padded_frame_ranks,
)


def _analysis():
    spec = importlib.util.spec_from_file_location("_tested_hard_wall_v2_analysis", BUNDLE / "analyze_campaign.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ANALYSIS = _analysis()


def random_frame(samples: int, dimension: int, rank: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(7162)
    raw = torch.complex(
        torch.randn((samples, dimension, rank), dtype=torch.float64, generator=generator),
        torch.randn((samples, dimension, rank), dtype=torch.float64, generator=generator),
    )
    return torch.linalg.qr(raw, mode="reduced").Q.to(torch.complex128)


def dense_reference(frame: torch.Tensor, indices: torch.Tensor, nx: int, ay: int):
    outputs = {key: [] for key in ("entropy_von_neumann", "entropy_renyi2", "entropy_renyi3", "charge_mean", "charge_variance")}
    contours = {key: [] for key in ("contour_von_neumann", "contour_renyi2", "contour_renyi3", "contour_charge_variance")}
    for sample in range(frame.shape[0]):
        correlation = frame[sample] @ frame[sample].mH
        sample_values = {key: [] for key in outputs}
        sample_contours = {key: [] for key in contours}
        for idx in indices:
            ca = correlation.index_select(0, idx).index_select(1, idx)
            nu, vectors = torch.linalg.eigh(0.5 * (ca + ca.mH))
            nu = nu.clamp(0, 1)
            one = 1 - nu
            weights = {
                "entropy_von_neumann": -torch.xlogy(nu, nu) - torch.xlogy(one, one),
                "entropy_renyi2": -torch.log(nu.square() + one.square()),
                "entropy_renyi3": -0.5 * torch.log(nu.pow(3) + one.pow(3)),
                "charge_mean": nu,
                "charge_variance": nu * one,
            }
            for key, value in weights.items():
                sample_values[key].append(value.sum())
            for scalar, contour in (
                ("entropy_von_neumann", "contour_von_neumann"),
                ("entropy_renyi2", "contour_renyi2"),
                ("entropy_renyi3", "contour_renyi3"),
                ("charge_variance", "contour_charge_variance"),
            ):
                value = (vectors.abs().square() @ weights[scalar]).reshape(ay, nx, 2).sum(-1).T
                sample_contours[contour].append(value)
        for key in outputs:
            outputs[key].append(torch.stack(sample_values[key]))
        for key in contours:
            contours[key].append(torch.stack(sample_contours[key]))
    return {
        **{key: torch.stack(value) for key, value in outputs.items()},
        **{key: torch.stack(value) for key, value in contours.items()},
    }


def test_periodic_indices_and_ay_zero() -> None:
    indices = periodic_window_indices(nx=2, ny=5, y0_values=[4, 1], ay=2)
    assert indices[0].tolist() == [16, 17, 18, 19, 0, 1, 2, 3]
    frame = random_frame(2, 20, 7)
    zero = frame_y0_averaged_width(frame, nx=2, ny=5, ay=0, matrix_batch_size=16)
    for key in ("entropy_von_neumann", "entropy_renyi2", "entropy_renyi3", "charge_mean", "charge_variance"):
        assert np.array_equal(zero[key], np.zeros(2))


def test_batched_gram_matches_dense_and_is_batch_invariant() -> None:
    nx, ny, ay = 2, 6, 3
    frame = random_frame(3, 2 * nx * ny, 10)
    indices = periodic_window_indices(nx=nx, ny=ny, y0_values=[0, 5], ay=ay)
    actual = frame_window_observables(frame, indices=indices, nx=nx, ay=ay, return_contours=True)
    expected = dense_reference(frame, indices, nx, ay)
    for key in expected:
        assert torch.allclose(actual[key], expected[key], atol=2e-11, rtol=2e-11), key
    small = frame_y0_averaged_width(frame, nx=nx, ny=ny, ay=ay, matrix_batch_size=2)
    large = frame_y0_averaged_width(frame, nx=nx, ny=ny, ay=ay, matrix_batch_size=128)
    for key in ("entropy_von_neumann", "entropy_renyi2", "entropy_renyi3", "charge_mean", "charge_variance"):
        assert np.allclose(small[key], large[key], atol=2e-11, rtol=2e-11)


def test_fixed_half_contours_close_and_use_y0_zero() -> None:
    nx, ny = 3, 8
    frame = random_frame(2, 2 * nx * ny, 17)
    contours = frame_fixed_half_strip_contours(frame, nx=nx, ny=ny, matrix_batch_size=1)
    scalar = frame_window_observables(
        frame,
        indices=periodic_window_indices(nx=nx, ny=ny, y0_values=[0], ay=ny // 2),
        nx=nx,
        ay=ny // 2,
        return_contours=False,
    )
    mapping = {
        "contour_von_neumann": "entropy_von_neumann",
        "contour_renyi2": "entropy_renyi2",
        "contour_renyi3": "entropy_renyi3",
        "contour_charge_variance": "charge_variance",
    }
    for contour, key in mapping.items():
        assert np.allclose(contours[contour].sum((1, 2)), scalar[key][:, 0].numpy(), atol=2e-11)
        assert np.allclose(
            contours[f"fixed_scalar__{key}"], scalar[key][:, 0].numpy(), atol=2e-11
        )


class State:
    def __init__(self, ranks: list[int]):
        self.ranks = torch.tensor(ranks)
        self.frame = torch.empty(0)


def test_observer_dynamics_callback_never_invokes_eigensolver(monkeypatch) -> None:
    observer = HardWallEntropyChargeObserver(nx=2, ny=4, physical_cycles=8, sample_ids=[0, 1])
    monkeypatch.setattr(torch.linalg, "eigvalsh", lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("endpoint solver called")))
    for cycle in range(9):
        observer(cycle=cycle, state=State([7 + cycle, 8 + cycle]), batch_start=0, batch_count=2)
    observer.validate(require_dynamics=True, require_endpoint=False)
    assert observer.global_charge.shape == (2, 9)


def test_occupation_tolerance_rejects_nonphysical_frame() -> None:
    frame = torch.zeros((1, 8, 4), dtype=torch.complex128)
    frame[0, 0, 0] = 2
    with pytest.raises(FloatingPointError, match="outside"):
        frame_y0_averaged_width(frame, nx=1, ny=4, ay=1, matrix_batch_size=16)


def test_variable_ranks_require_zero_padding() -> None:
    frame = random_frame(2, 16, 6)
    padded = torch.cat((frame, torch.zeros((2, 16, 3), dtype=frame.dtype)), dim=2)
    validate_padded_frame_ranks(padded, np.asarray([6, 6]))
    bad = padded.clone()
    bad[1, 0, 7] = 1e-4
    with pytest.raises(FloatingPointError, match="padded"):
        validate_padded_frame_ranks(bad, np.asarray([6, 6]))


def test_renyi_prefactors_and_paired_bootstrap() -> None:
    ay = np.arange(8, 21)
    x = ANALYSIS.log_chord(ay, 40)
    curves = {
        "c1": np.tile((1 / 3) * x, (100, 1)),
        "c2": np.tile((1 / 4) * x, (100, 1)),
        "c3": np.tile((2 / 9) * x, (100, 1)),
        "k": np.tile((1 / np.pi**2) * x, (100, 1)),
    }
    for label, values in curves.items():
        slope, _ = ANALYSIS.trajectory_slopes(ay, values, 40)
        assert ANALYSIS.PREFACTOR[label] * slope.mean() == pytest.approx(1.0)
