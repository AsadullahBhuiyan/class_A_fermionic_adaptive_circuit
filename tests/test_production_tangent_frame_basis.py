from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


SHARED = (
    Path(__file__).resolve().parents[1]
    / "00_WORKSPACE"
    / "CURRENT"
    / "final_production_ready_figure_scripts"
    / "_shared_src"
)
sys.path.insert(0, str(SHARED))

from tangent_observables import TangentFrameWriter  # noqa: E402


def _support_terminated_basis(nx: int, ny: int) -> np.ndarray:
    return np.asarray(
        [
            orbital + 2 * x + 2 * nx * y
            for y in range(ny)
            for x in range(5, 16)
            for orbital in (0, 1)
        ],
        dtype=np.int64,
    )


def test_support_terminated_full_frame_is_saved_in_active_basis(
    tmp_path: Path,
) -> None:
    nx = ny = 20
    samples, nvec = 5, 16
    basis = _support_terminated_basis(nx, ny)
    assert basis.shape == (440,)
    full_rows = 2 * nx * ny
    full_frame = np.broadcast_to(
        np.arange(full_rows, dtype=np.float64)[None, :, None],
        (samples, full_rows, nvec),
    ).astype(np.complex128)
    identity_r = np.broadcast_to(
        np.eye(nvec, dtype=np.complex128), (samples, nvec, nvec)
    )

    writer = TangentFrameWriter(
        samples=samples,
        physical_cycles=1,
        nlayer=len(basis),
        nvec=nvec,
        alignment_cycles=0,
        frame_cycles=(1,),
        nx=nx,
        ny=ny,
        basis_indices=basis,
    )
    writer(
        cycle=1,
        spectra=np.zeros((samples, nvec)),
        batch_start=0,
        batch_count=samples,
        lyapunov_qr_r=identity_r,
        lyapunov_frame=full_frame,
        lyapunov_cycle_null_mask=np.zeros((samples, nvec), dtype=bool),
        lyapunov_active_mask=np.ones(samples, dtype=bool),
    )

    saved = writer.frames[1]
    assert saved.shape == (samples, 440, nvec)
    np.testing.assert_array_equal(saved, full_frame[:, basis, :])
    assert writer.frame_x_weight[1].shape == (samples, nvec, nx)
    result = writer.save(tmp_path / "tangent.npz")
    with np.load(result["path"], allow_pickle=False) as data:
        np.testing.assert_array_equal(data["frame_basis_indices"], basis)
        np.testing.assert_array_equal(data["aligned_frame_cycle_0001"], saved)


def test_tangent_writer_rejects_unrecognized_frame_dimension() -> None:
    writer = TangentFrameWriter(
        samples=1,
        physical_cycles=1,
        nlayer=4,
        nvec=2,
        alignment_cycles=0,
        frame_cycles=(1,),
        nx=2,
        ny=2,
        basis_indices=(0, 1, 2, 3),
    )
    with np.testing.assert_raises_regex(ValueError, "tangent frame rows"):
        writer._frame_in_declared_basis(np.zeros((1, 6, 2)))
