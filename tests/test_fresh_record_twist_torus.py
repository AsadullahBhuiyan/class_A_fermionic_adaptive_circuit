from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/fgtn"))
sys.path.insert(
    0,
    str(ROOT / "00_WORKSPACE/CURRENT/fresh_record_twist_torus_validation"),
)

from classA_U1FGTN import classA_U1FGTN
from twist_torus import analyze_frame_surface, exact_target_frame, gauge_vector


def _target_frame(nx: int, ny: int, tx: float, ty: float) -> np.ndarray:
    model = classA_U1FGTN(
        nx,
        ny,
        DW=False,
        nshell=None,
        alpha_1=1,
        alpha_2=1,
        twist_x=tx,
        twist_y=ty,
    )
    model.construct_OW_projectors(
        None, DW=False, twist_x=tx, twist_y=ty
    )
    return exact_target_frame(model)


def _surface(nx: int = 2, ny: int = 2, grid: int = 3) -> np.ndarray:
    frames = np.empty((grid, grid), dtype=object)
    for ix in range(grid):
        for iy in range(grid):
            frames[ix, iy] = _target_frame(
                nx, ny, 2 * np.pi * ix / grid, 2 * np.pi * iy / grid
            )
    return frames


def test_exact_target_surface_has_positive_repository_chern_orientation():
    result = analyze_frame_surface(
        _surface(), nx=2, ny=2, singular_value_tolerance=1e-10
    )
    assert result["classification"] == "defined_C1"
    np.testing.assert_allclose(result["chern"], 1.0, atol=1e-12)


def test_fhs_is_invariant_under_occupied_frame_gauge_rotations():
    frames = _surface()
    rng = np.random.default_rng(14)
    transformed = np.empty_like(frames)
    for index in np.ndindex(frames.shape):
        rank = frames[index].shape[1]
        matrix = rng.standard_normal((rank, rank)) + 1j * rng.standard_normal(
            (rank, rank)
        )
        q, _ = np.linalg.qr(matrix)
        transformed[index] = frames[index] @ q
    result = analyze_frame_surface(
        transformed, nx=2, ny=2, singular_value_tolerance=1e-10
    )
    assert result["classification"] == "defined_C1"
    np.testing.assert_allclose(result["chern"], 1.0, atol=1e-12)


def test_rank_mismatch_is_undefined_without_padding():
    frames = _surface()
    frames[1, 1] = frames[1, 1][:, :-1]
    result = analyze_frame_surface(
        frames, nx=2, ny=2, singular_value_tolerance=1e-10
    )
    assert result["classification"] == "undefined_rank_mismatch"
    assert np.isnan(result["chern"])


def test_singular_link_is_undefined_without_pseudoinverse():
    frames = np.empty((2, 2), dtype=object)
    eye = np.eye(4, dtype=np.complex128)
    for index in np.ndindex(frames.shape):
        frames[index] = eye[:, :2].copy()
    frames[1, 0] = eye[:, 2:].copy()
    result = analyze_frame_surface(
        frames, nx=1, ny=2, singular_value_tolerance=1e-10
    )
    assert result["classification"] == "undefined_singular_links"
    assert result["valid_link_fraction"] < 1.0


def test_large_gauge_vectors_match_target_closure():
    zero = _target_frame(3, 2, 0.0, 0.0)
    closed_x = _target_frame(3, 2, 2 * np.pi, 0.0)
    closed_y = _target_frame(3, 2, 0.0, 2 * np.pi)
    for actual, gauge in (
        (closed_x, gauge_vector(3, 2, "x")),
        (closed_y, gauge_vector(3, 2, "y")),
    ):
        overlap = zero.conj().T @ (gauge.conj()[:, None] * actual)
        np.testing.assert_allclose(
            overlap.conj().T @ overlap, np.eye(zero.shape[1]), atol=2e-12
        )
