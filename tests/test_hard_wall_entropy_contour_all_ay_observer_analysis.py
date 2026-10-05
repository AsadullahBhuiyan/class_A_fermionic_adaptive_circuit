from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


REPO = Path(__file__).resolve().parents[1]
BUNDLE = (
    REPO
    / "00_WORKSPACE/CURRENT/final_production_new_designs/16_hard_wall_entropy_contour_all_ay"
)
if str(BUNDLE) not in sys.path:
    sys.path.insert(0, str(BUNDLE))

from entropy_contour_observer import (  # noqa: E402
    CONTOUR_KEY,
    HardWallAllAyContourObserver,
    SCALAR_KEYS,
    frame_window_observables,
    frame_y0_averaged_width,
    periodic_window_indices,
    validate_padded_frame_ranks,
)


def _load_analysis():
    name = "tested_hard_wall_entropy_contour_all_ay_analysis"
    spec = importlib.util.spec_from_file_location(name, BUNDLE / "analyze_campaign.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ANALYSIS = _load_analysis()


def random_frame(samples: int, dimension: int, rank: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(2026091416)
    raw = torch.complex(
        torch.randn((samples, dimension, rank), dtype=torch.float64, generator=generator),
        torch.randn((samples, dimension, rank), dtype=torch.float64, generator=generator),
    )
    return torch.linalg.qr(raw, mode="reduced").Q.to(torch.complex128)


def dense_reference(
    frame: torch.Tensor, *, nx: int, ny: int, ay: int
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    curves = {key: np.zeros(frame.shape[0], dtype=np.float64) for key in SCALAR_KEYS}
    contours = np.zeros((frame.shape[0], nx, ay), dtype=np.float64)
    for sample in range(frame.shape[0]):
        correlation = frame[sample] @ frame[sample].mH
        for y0 in range(ny):
            indices = periodic_window_indices(
                nx=nx, ny=ny, y0_values=[y0], ay=ay
            )[0]
            restricted = correlation.index_select(0, indices).index_select(1, indices)
            occupation, vectors = torch.linalg.eigh(0.5 * (restricted + restricted.mH))
            occupation = occupation.clamp(0.0, 1.0)
            complement = 1.0 - occupation
            weights = {
                "entropy_von_neumann": (
                    -torch.xlogy(occupation, occupation)
                    - torch.xlogy(complement, complement)
                ),
                "entropy_renyi2": -torch.log(
                    occupation.square() + complement.square()
                ),
                "entropy_renyi3": -0.5
                * torch.log(occupation.pow(3) + complement.pow(3)),
                "charge_mean": occupation,
                "charge_variance": occupation * complement,
            }
            for key in SCALAR_KEYS:
                curves[key][sample] += float(weights[key].sum()) / ny
            contour = (
                (vectors.abs().square() @ weights["entropy_von_neumann"])
                .reshape(ay, nx, 2)
                .sum(-1)
                .T
            )
            contours[sample] += contour.numpy() / ny
    return curves, contours


def test_periodic_indices_use_relative_y_order_and_wrap() -> None:
    indices = periodic_window_indices(nx=2, ny=5, y0_values=[4, 1], ay=2)
    assert indices[0].tolist() == [16, 17, 18, 19, 0, 1, 2, 3]
    assert indices[1].tolist() == [4, 5, 6, 7, 8, 9, 10, 11]


def test_all_origin_batched_results_match_dense_reference_and_close() -> None:
    nx, ny, ay = 2, 6, 3
    frame = random_frame(samples=3, dimension=2 * nx * ny, rank=10)
    expected_curves, expected_contour = dense_reference(
        frame, nx=nx, ny=ny, ay=ay
    )
    small = frame_y0_averaged_width(
        frame, nx=nx, ny=ny, ay=ay, matrix_batch_size=2
    )
    large = frame_y0_averaged_width(
        frame, nx=nx, ny=ny, ay=ay, matrix_batch_size=128
    )
    for key in SCALAR_KEYS:
        assert np.allclose(small[key], expected_curves[key], atol=2e-11, rtol=2e-11)
        assert np.allclose(small[key], large[key], atol=2e-11, rtol=2e-11)
    assert np.allclose(small[CONTOUR_KEY], expected_contour, atol=2e-11, rtol=2e-11)
    assert np.allclose(small[CONTOUR_KEY], large[CONTOUR_KEY], atol=2e-11, rtol=2e-11)
    assert np.allclose(
        small[CONTOUR_KEY].sum(axis=(1, 2)),
        small["entropy_von_neumann"],
        atol=2e-11,
        rtol=2e-11,
    )


def test_zero_width_is_exact_and_does_not_invoke_eigensolver(monkeypatch) -> None:
    frame = random_frame(samples=2, dimension=20, rank=7)
    monkeypatch.setattr(
        torch.linalg,
        "eigh",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("unexpected eigensolve")),
    )
    result = frame_y0_averaged_width(
        frame, nx=2, ny=5, ay=0, matrix_batch_size=8
    )
    for key in SCALAR_KEYS:
        assert np.array_equal(result[key], np.zeros(2))
    assert result[CONTOUR_KEY].shape == (2, 2, 0)


class State:
    def __init__(self, ranks: list[int]) -> None:
        self.ranks = torch.tensor(ranks, dtype=torch.int64)
        self.frame = torch.empty(0)


def test_dynamics_observer_records_only_charge_and_never_eigensolves(monkeypatch) -> None:
    observer = HardWallAllAyContourObserver(
        nx=2, ny=4, physical_cycles=8, sample_ids=[0, 1]
    )
    fail = lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("endpoint solver called"))
    monkeypatch.setattr(torch.linalg, "eigvalsh", fail)
    monkeypatch.setattr(torch.linalg, "eigh", fail)
    for cycle in range(9):
        observer(
            cycle=cycle,
            state=State([7 + cycle, 8 + cycle]),
            batch_start=0,
            batch_count=2,
        )
    observer.validate(require_dynamics=True, require_endpoint=False)
    assert observer.global_charge.shape == (2, 9)
    assert not observer.endpoint_width_seen.any()


def test_observer_ragged_padding_checkpoint_restore_and_result_contract() -> None:
    nx, ny = 2, 6
    frame = random_frame(samples=2, dimension=2 * nx * ny, rank=10)
    observer = HardWallAllAyContourObserver(
        nx=nx, ny=ny, physical_cycles=2 * ny, sample_ids=[4, 5]
    )
    for cycle in range(2 * ny + 1):
        observer(
            cycle=cycle,
            state=State([10, 10]),
            batch_start=0,
            batch_count=2,
        )
    for ay in range(ny // 2 + 1):
        observer.record_endpoint_width(
            frame, ay=ay, matrix_batch_size=2, elapsed_seconds=float(ay)
        )
    observer.validate(require_dynamics=True, require_endpoint=True)
    assert observer.contours.shape == (2, 4, nx, 3)
    for ay in range(4):
        assert np.all(observer.contours[:, ay, :, ay:] == 0.0)

    restored = HardWallAllAyContourObserver(
        nx=nx, ny=ny, physical_cycles=2 * ny, sample_ids=[4, 5]
    )
    restored.restore_checkpoint(observer.checkpoint_payload())
    payload = restored.result_payload(slice(0, 1))
    assert payload["endpoint__contour_von_neumann_y0avg"].shape == (1, 4, nx, 3)
    assert payload["contour_coordinate"].item() == "relative_dy=(y-y0)_mod_Ny"
    assert payload["origin_average_count"].item() == ny
    assert np.array_equal(payload["valid_dy_count"], np.arange(4))


def test_nonphysical_occupations_and_nonzero_rank_padding_are_rejected() -> None:
    frame = torch.zeros((1, 8, 4), dtype=torch.complex128)
    frame[0, 0, 0] = 2
    with pytest.raises(FloatingPointError, match="outside"):
        frame_y0_averaged_width(frame, nx=1, ny=4, ay=1, matrix_batch_size=8)

    good = random_frame(samples=2, dimension=16, rank=6)
    padded = torch.cat((good, torch.zeros((2, 16, 3), dtype=good.dtype)), dim=2)
    validate_padded_frame_ranks(padded, np.asarray([6, 6]))
    padded[1, 0, 7] = 1.0e-4
    with pytest.raises(FloatingPointError, match="padded"):
        validate_padded_frame_ranks(padded, np.asarray([6, 6]))


def test_mean_curve_fit_uses_full_trajectory_covariance_and_100_samples() -> None:
    ny = 40
    ay = np.arange(ny // 2 + 1)
    selected = ay >= 8
    x = ANALYSIS.log_chord(ay[selected], ny)
    rng = np.random.default_rng(9292)
    sample_slopes = 1.0 / 6.0 + rng.normal(scale=0.01, size=100)
    sample_offsets = rng.normal(scale=0.03, size=100)
    curves = np.zeros((100, len(ay)), dtype=np.float64)
    curves[:, selected] = sample_slopes[:, None] * x + sample_offsets[:, None]
    fit = ANALYSIS.mean_curve_fit(ay, curves, ny)
    assert fit["slope"] == pytest.approx(sample_slopes.mean(), abs=2e-13)
    assert fit["slope_sem"] == pytest.approx(
        sample_slopes.std(ddof=1) / np.sqrt(100), abs=2e-13
    )


def test_wall_integration_closes_to_full_entropy() -> None:
    samples, ny, nx = 100, 30, 20
    half = ny // 2
    contour = np.zeros((samples, half + 1, nx, half), dtype=np.float64)
    for ay in range(half + 1):
        contour[:, ay, :, :ay] = 1.0 / max(1, nx * ay)
    entropy = contour.sum(axis=(2, 3))
    case = {
        ANALYSIS.CONTOUR_KEY: contour,
        ANALYSIS.CURVE_KEYS["S1"]: entropy,
        "ay_values": np.arange(half + 1),
    }
    resolved = ANALYSIS.integrated_curves(case)
    assert np.allclose(resolved["full"], entropy)
    assert np.allclose(
        resolved["full"],
        resolved["left"] + resolved["right"] + resolved["leakage"],
    )

