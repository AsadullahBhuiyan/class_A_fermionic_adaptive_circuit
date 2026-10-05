from __future__ import annotations

import numpy as np
import pytest

from src.fgtn.classA_U1FGTN import classA_U1FGTN
from src.fgtn.occupied_frame import OccupiedFrameState


def _numpy_reconstructed_gram(state: dict) -> np.ndarray:
    dimension = state["frame"].shape[1]
    result = np.zeros((dimension, dimension), dtype=np.complex128)
    start = 0
    for index, size in enumerate(state["block_sizes"]):
        stop = start + int(size)
        image = (
            state["frame"][0, :, start:stop]
            @ state["block_core_hat"][index][0]
            * np.exp(state["block_core_log_scale"][index][0])
        )
        result += image @ image.conj().T
        start = stop
    return result


def test_cpu_pure_blocks_reconstruct_full_product_and_match_gpu_convention():
    torch = pytest.importorskip("torch")
    from src.fgtn.classA_U1FGTN_gpu import classA_U1FGTN_gpu
    from src.fgtn.occupied_frame_gpu import BatchedOccupiedFrameState

    dimension = 8
    rank = dimension // 2
    rng = np.random.default_rng(9102)
    raw = rng.normal(size=(dimension, rank)) + 1j * rng.normal(
        size=(dimension, rank)
    )
    occupied, _ = np.linalg.qr(raw, mode="reduced")

    cpu_model = classA_U1FGTN(2, 2, DW=False, nshell=0)
    cpu_physical = OccupiedFrameState(
        occupied,
        representation="physical_frame",
        physical_dimension=dimension,
    )
    cpu = cpu_model._init_pure_occupied_empty_lyapunov_state(cpu_physical)

    gpu_model = classA_U1FGTN_gpu(
        Nx=2,
        Ny=2,
        DW=False,
        nshell=0,
        device="cpu",
        dtype="complex128",
        backend="local",
    )
    gpu_physical = BatchedOccupiedFrameState.from_frame(
        torch.as_tensor(occupied, dtype=torch.complex128)
    )
    gpu = gpu_model._init_pure_occupied_empty_lyapunov_state(gpu_physical)

    product = np.eye(dimension, dtype=np.complex128)
    for cycle in range(1, 4):
        raw_map = np.eye(dimension, dtype=np.complex128) + 0.03 * (
            rng.normal(size=(dimension, dimension))
            + 1j * rng.normal(size=(dimension, dimension))
        )
        product = raw_map @ product
        cpu["frame"] = raw_map[None, ...] @ cpu["frame"]
        gpu["frame"] = (
            torch.as_tensor(raw_map, dtype=torch.complex128)[None, ...]
            @ gpu["frame"]
        )
        cpu_model._lyapunov_end_cycle(cpu, cycle)
        gpu_model._lyapunov_end_cycle(gpu, cycle)

    np.testing.assert_allclose(
        _numpy_reconstructed_gram(cpu),
        product @ product.conj().T,
        rtol=2e-11,
        atol=2e-11,
    )
    for index in range(2):
        cpu_s = np.linalg.svd(cpu["block_core_hat"][index][0], compute_uv=False)
        cpu_logs = np.log(cpu_s) + cpu["block_core_log_scale"][index][0]
        gpu_s = torch.linalg.svdvals(gpu["block_core_hat"][index][0])
        gpu_logs = (
            torch.log(gpu_s) + gpu["block_core_log_scale"][index][0]
        ).cpu().numpy()
        np.testing.assert_allclose(
            np.sort(cpu_logs), np.sort(gpu_logs), rtol=2e-11, atol=2e-11
        )


def test_cpu_public_pure_tangent_mode_is_opt_in_and_emits_block_payload():
    rows: list[dict] = []
    model = classA_U1FGTN(2, 2, DW=False, nshell=0)
    result = model.run_markov_circuit(
        G_history=False,
        progress=False,
        cycles=2,
        samples=1,
        parallelize_samples=False,
        init_mode="default",
        save=False,
        sequence="raster_y",
        perfect_correction=True,
        random_seed=17,
        state_representation="physical_frame",
        physical_covariance_update="rank1",
        lyapunov_frame_observer=lambda **row: rows.append(row),
        lyapunov_basis_mode="pure_occupied_empty",
        lyapunov_start_cycle=1,
        lyapunov_full_space=True,
        lyapunov_track_restricted_core=True,
    )
    assert len(rows) == 2
    assert rows[-1]["lyapunov_basis_mode"] == "pure_occupied_empty"
    assert rows[-1]["lyapunov_block_sizes"] == (4, 4)
    assert len(rows[-1]["lyapunov_initial_block_basis"]) == 2
    assert result["lyapunov_basis_mode"] == "pure_occupied_empty"
    assert result["G_final"].dtype == np.complex128


def test_cpu_pure_tangent_rejects_mixed_covariance():
    model = classA_U1FGTN(2, 2, DW=False, nshell=0)
    mixed = np.zeros((8, 8), dtype=np.complex128)
    with pytest.raises(ValueError, match="physical-frame"):
        model.run_markov_circuit(
            G_history=False,
            progress=False,
            cycles=1,
            samples=1,
            save=False,
            G_init=mixed,
            state_representation="covariance",
            lyapunov_frame_observer=lambda **_: None,
            lyapunov_basis_mode="pure_occupied_empty",
        )
