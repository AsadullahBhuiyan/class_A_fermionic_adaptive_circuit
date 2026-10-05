from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch


HELPER_VERSION = "purification_dynamics_observables_gpu_v2"
PROTOCOL_ORDER = ("perfect_correction", "postselect")
TRACE_IMAG_TOL = 1e-8
HERM_TOL = 1e-9
CHARGE_VARIANCE_IMAG_TOL = 1e-9
BORN_PROB_TOL = 1e-8


def write_json_atomic(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    tmp_path.replace(path)


def save_npz_atomic(path: Path, **arrays: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("wb") as fh:
        np.savez_compressed(fh, **arrays)
    tmp_path.replace(path)


def save_dataframe_atomic_csv(df: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp_path, index=False)
    tmp_path.replace(path)


def rel_to_root(path: Path | str | None, root: Path | str) -> str | None:
    if path is None:
        return None
    try:
        return str(Path(path).resolve().relative_to(Path(root).resolve()))
    except Exception:
        return None


def expected_samples_for_config(cfg: dict[str, Any]) -> int:
    postselect_probability = float(
        cfg.get("postselect_probability", 1.0 if bool(cfg.get("postselect", False)) else 0.0)
    )
    return 1 if bool(cfg.get("postselect", False)) or postselect_probability == 1.0 else int(cfg["samples"])


def case_output_dir(runs_root: Path, cfg: dict[str, Any]) -> Path:
    return Path(runs_root) / (
        f"N{int(cfg['Nx'])}x{int(cfg['Ny'])}_nsh{int(cfg['nshell'])}_"
        f"init-{cfg['init_mode']}_{cfg['protocol']}"
    )


def domain_wall_metadata(model: Any) -> dict[str, Any]:
    dw_loc = [int(x) for x in getattr(model, "DW_loc", [])]
    payload: dict[str, Any] = {"dw_loc": dw_loc}
    if len(dw_loc) == 2:
        payload["topological_x_range"] = [dw_loc[0], dw_loc[1]]
        payload["trivial_x_segments"] = [
            [0, max(-1, dw_loc[0] - 1)],
            [min(int(model.Nx), dw_loc[1] + 1), int(model.Nx) - 1],
        ]
    else:
        payload["topological_x_range"] = None
        payload["trivial_x_segments"] = None
    return payload


def geometry_key(nx: int, ny: int) -> str:
    return f"N{int(nx)}x{int(ny)}"


def config_id(nx: int, ny: int, protocol: str) -> str:
    return f"{geometry_key(nx, ny)}_{str(protocol)}"


def _is_eigh_convergence_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "linalg.eigh" in msg and ("failed to converge" in msg or "ill-conditioned" in msg)


def _eigh_with_fallback(occ: torch.Tensor) -> tuple[torch.Tensor, int]:
    try:
        evals = torch.linalg.eigvalsh(occ)
        return evals, 0
    except Exception as exc:
        if not _is_eigh_convergence_error(exc):
            raise
    if occ.ndim == 3 and int(occ.shape[0]) > 1:
        mid = int(occ.shape[0]) // 2
        evals_l, fall_l = _eigh_with_fallback(occ[:mid])
        evals_r, fall_r = _eigh_with_fallback(occ[mid:])
        return torch.cat([evals_l, evals_r], dim=0), fall_l + fall_r
    occ_cpu = occ.detach().cpu()
    try:
        evals_cpu = torch.linalg.eigvalsh(occ_cpu)
    except Exception:
        n = int(occ_cpu.shape[-1])
        jitter = 100.0 * torch.finfo(occ_cpu.real.dtype).eps
        eye = torch.eye(n, dtype=occ_cpu.dtype, device=occ_cpu.device)
        evals_cpu = torch.linalg.eigvalsh(occ_cpu + jitter * eye)
    return evals_cpu.to(occ.device), 1


def entropy_total_batch_torch(
    sub_g: torch.Tensor,
    *,
    eps: float = 1e-12,
    validate: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    if sub_g.ndim != 3:
        raise ValueError(f"Expected sub_G shape (B,M,M), got {tuple(sub_g.shape)}")
    batch_count, nrow, ncol = sub_g.shape
    if nrow != ncol:
        raise ValueError(f"Expected restricted covariance shape (B,M,M), got {tuple(sub_g.shape)}")
    if nrow == 0:
        return torch.zeros((batch_count,), dtype=torch.float64, device=sub_g.device), {
            "max_hermiticity_error": 0.0,
            "min_occupation_eval": 0.0,
            "max_occupation_eval": 0.0,
            "eigh_cpu_fallback_count": 0,
        }
    if validate and not torch.isfinite(sub_g).all():
        raise FloatingPointError("Non-finite restricted covariance entries.")
    herm_error = torch.amax(torch.abs(sub_g - sub_g.conj().transpose(-2, -1))).detach()
    eye = torch.eye(nrow, dtype=sub_g.dtype, device=sub_g.device)
    occ = 0.5 * (sub_g + eye.unsqueeze(0))
    occ = 0.5 * (occ + occ.conj().transpose(-2, -1))
    evals, fallback_count = _eigh_with_fallback(occ)
    evals_real = evals.real
    evals_clamped = torch.clamp(evals_real, float(eps), 1.0 - float(eps))
    weights = -(evals_clamped * torch.log(evals_clamped) + (1.0 - evals_clamped) * torch.log(1.0 - evals_clamped))
    totals = weights.sum(dim=-1).to(torch.float64)
    return totals, {
        "max_hermiticity_error": float(herm_error.detach().cpu()),
        "min_occupation_eval": float(torch.min(evals_real).detach().cpu()),
        "max_occupation_eval": float(torch.max(evals_real).detach().cpu()),
        "eigh_cpu_fallback_count": int(fallback_count),
    }


def build_chern_partition_indices(
    *,
    nx: int,
    ny: int,
    xref: int | None = None,
    yref: int | None = None,
    radius: float | None = None,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    nx = int(nx)
    ny = int(ny)
    xref = nx // 2 if xref is None else int(xref)
    yref = ny // 2 if yref is None else int(yref)
    radius = 0.4 * min(nx, ny) if radius is None else float(radius)
    if radius <= 0:
        raise ValueError(f"radius must be positive; got {radius}")
    inside = np.zeros((nx, ny), dtype=bool)
    a_mask = np.zeros_like(inside)
    b_mask = np.zeros_like(inside)
    c_mask = np.zeros_like(inside)
    rr = radius * radius
    ymax = int(math.floor(radius))
    a2 = 2.0 * np.pi / 3.0
    a4 = 4.0 * np.pi / 3.0
    for dy in range(-ymax, ymax + 1):
        y = yref + dy
        if y < 0 or y >= ny:
            continue
        max_dx = int(math.floor(math.sqrt(rr - dy * dy)))
        x0 = max(0, xref - max_dx)
        x1 = min(nx - 1, xref + max_dx)
        if x0 > x1:
            continue
        inside[x0 : x1 + 1, y] = True
        dxs = np.arange(x0, x1 + 1) - xref
        dys = np.full_like(dxs, dy)
        theta = np.mod(np.arctan2(dys, dxs), 2 * np.pi)
        a_mask[x0 : x1 + 1, y] = (theta >= 0.0) & (theta < a2)
        b_mask[x0 : x1 + 1, y] = (theta >= a2) & (theta < a4)
        c_mask[x0 : x1 + 1, y] = (theta >= a4) & (theta < 2 * np.pi)

    def idx_from_mask(mask: np.ndarray) -> np.ndarray:
        xs, ys = np.nonzero(mask)
        idx0 = 0 + 2 * xs + 2 * nx * ys
        idx1 = 1 + 2 * xs + 2 * nx * ys
        return np.sort(np.concatenate([idx0, idx1])).astype(np.int64, copy=False)

    torch_device = torch.device(device)
    return {
        "nx": nx,
        "ny": ny,
        "xref": xref,
        "yref": yref,
        "radius": radius,
        "inside_mask": inside.astype(bool, copy=True),
        "A": torch.as_tensor(idx_from_mask(a_mask), dtype=torch.long, device=torch_device),
        "B": torch.as_tensor(idx_from_mask(b_mask), dtype=torch.long, device=torch_device),
        "C": torch.as_tensor(idx_from_mask(c_mask), dtype=torch.long, device=torch_device),
    }


def real_space_chern_batch_torch(g_batch: torch.Tensor, partitions: dict[str, Any]) -> torch.Tensor:
    if g_batch.ndim != 3:
        raise ValueError(f"Expected G_batch shape (B,N,N), got {tuple(g_batch.shape)}")
    _, nrow, ncol = g_batch.shape
    if nrow != ncol:
        raise ValueError(f"Expected square covariance batch, got {tuple(g_batch.shape)}")
    eye = torch.eye(nrow, dtype=g_batch.dtype, device=g_batch.device)
    p = torch.conj(0.5 * (g_batch + eye.unsqueeze(0)))
    i_a = partitions["A"].to(g_batch.device)
    i_b = partitions["B"].to(g_batch.device)
    i_c = partitions["C"].to(g_batch.device)

    def gather(rows: torch.Tensor, cols: torch.Tensor) -> torch.Tensor:
        out = torch.index_select(p, 1, rows)
        return torch.index_select(out, 2, cols)

    p_ca = gather(i_c, i_a)
    p_ab = gather(i_a, i_b)
    p_bc = gather(i_b, i_c)
    p_ac = gather(i_a, i_c)
    p_cb = gather(i_c, i_b)
    p_ba = gather(i_b, i_a)
    t1 = torch.diagonal(torch.bmm(torch.bmm(p_ca, p_ab), p_bc), dim1=-2, dim2=-1).sum(dim=-1)
    t2 = torch.diagonal(torch.bmm(torch.bmm(p_ac, p_cb), p_ba), dim1=-2, dim2=-1).sum(dim=-1)
    y = 12.0 * math.pi * 1j * (t1 - t2)
    return y.real.to(torch.float64)


def local_charge_cell_mean_batch_torch(g_batch: torch.Tensor, *, nx: int, ny: int) -> torch.Tensor:
    diag = torch.diagonal(g_batch, dim1=-2, dim2=-1)
    occ = 0.5 * (diag.real.to(torch.float64) + 1.0)
    occ = occ.reshape(int(g_batch.shape[0]), int(ny), int(nx), 2)
    return occ.sum(dim=-1).transpose(1, 2).contiguous()


def local_charge_cell_variance_batch_torch(
    g_batch: torch.Tensor,
    *,
    nx: int,
    ny: int,
) -> tuple[torch.Tensor, float]:
    if g_batch.ndim != 3:
        raise ValueError(f"Expected G_batch shape (B,N,N), got {tuple(g_batch.shape)}")
    batch_count, nrow, ncol = g_batch.shape
    if nrow != ncol:
        raise ValueError(f"Expected square covariance batch, got {tuple(g_batch.shape)}")
    expected = 2 * int(nx) * int(ny)
    if nrow != expected:
        raise ValueError(f"Expected covariance shape (B,{expected},{expected}), got {tuple(g_batch.shape)}")

    eye = torch.eye(nrow, dtype=g_batch.dtype, device=g_batch.device)
    occ = 0.5 * (g_batch + eye.unsqueeze(0))
    occ = 0.5 * (occ + occ.conj().transpose(-2, -1))

    x = torch.arange(int(nx), dtype=torch.long, device=g_batch.device)
    y = torch.arange(int(ny), dtype=torch.long, device=g_batch.device)
    base = 2 * x[:, None] + 2 * int(nx) * y[None, :]
    a_flat = base.reshape(-1)
    b_flat = (base + 1).reshape(-1)

    diag = torch.diagonal(occ, dim1=-2, dim2=-1)
    n_a = diag[:, a_flat].real.to(torch.float64).reshape(batch_count, int(nx), int(ny))
    n_b = diag[:, b_flat].real.to(torch.float64).reshape(batch_count, int(nx), int(ny))
    c_ab = occ[:, a_flat, b_flat].reshape(batch_count, int(nx), int(ny))
    c_ba = occ[:, b_flat, a_flat].reshape(batch_count, int(nx), int(ny))

    n_sum = n_a.to(occ.dtype) + n_b.to(occ.dtype)
    joint = n_a.to(occ.dtype) * n_b.to(occ.dtype) - c_ab * c_ba
    variance_complex = n_sum + 2.0 * joint - n_sum * n_sum
    imag_max_abs = float(torch.max(torch.abs(variance_complex.imag)).detach().cpu())
    variance = variance_complex.real.to(torch.float64)
    return variance, imag_max_abs


def total_charge_variance_batch_torch(g_batch: torch.Tensor) -> tuple[torch.Tensor, float]:
    if g_batch.ndim != 3:
        raise ValueError(f"Expected G_batch shape (B,N,N), got {tuple(g_batch.shape)}")
    _, nrow, ncol = g_batch.shape
    if nrow != ncol:
        raise ValueError(f"Expected square covariance batch, got {tuple(g_batch.shape)}")

    eye = torch.eye(nrow, dtype=g_batch.dtype, device=g_batch.device)
    occ = 0.5 * (g_batch + eye.unsqueeze(0))
    occ = 0.5 * (occ + occ.conj().transpose(-2, -1))

    trace_occ = torch.diagonal(occ, dim1=-2, dim2=-1).sum(dim=-1)
    trace_occ_sq = torch.diagonal(torch.bmm(occ, occ), dim1=-2, dim2=-1).sum(dim=-1)
    variance_complex = trace_occ - trace_occ_sq
    imag_max_abs = float(torch.max(torch.abs(variance_complex.imag)).detach().cpu())
    variance = variance_complex.real.to(torch.float64)
    return variance, imag_max_abs


def run_internal_checks() -> None:
    zero = torch.zeros((2, 2 * 4 * 6, 2 * 4 * 6), dtype=torch.complex128)
    totals, metrics = entropy_total_batch_torch(zero)
    expected_entropy = float((2 * 4 * 6) * math.log(2.0))
    if not np.allclose(totals.detach().cpu().numpy(), expected_entropy, atol=1e-10, rtol=1e-10):
        raise AssertionError("Maxmix total entropy check failed for G=0")
    if metrics["min_occupation_eval"] < -1e-12 or metrics["max_occupation_eval"] > 1.0 + 1e-12:
        raise AssertionError("Unexpected occupation spectrum range for G=0")

    charges = local_charge_cell_mean_batch_torch(zero, nx=4, ny=6).detach().cpu().numpy()
    if not np.allclose(charges, 1.0, atol=1e-12, rtol=1e-12):
        raise AssertionError("Maxmix charge mean should be exactly one per cell")

    variances, imag_max = local_charge_cell_variance_batch_torch(zero, nx=4, ny=6)
    if imag_max > CHARGE_VARIANCE_IMAG_TOL:
        raise AssertionError("Charge variance should be real for G=0")
    if not np.allclose(variances.detach().cpu().numpy(), 0.5, atol=1e-12, rtol=1e-12):
        raise AssertionError("Maxmix charge variance should be exactly one half per cell")
    total_var, imag_max = total_charge_variance_batch_torch(zero)
    if imag_max > CHARGE_VARIANCE_IMAG_TOL:
        raise AssertionError("Total charge variance should be real for G=0")
    if not np.allclose(total_var.detach().cpu().numpy(), 12.0, atol=1e-12, rtol=1e-12):
        raise AssertionError("Maxmix total charge variance should be Nx*Ny/2 for G=0")

    g_test = torch.zeros((1, 2 * 3 * 4, 2 * 3 * 4), dtype=torch.complex128)
    diag_vals = torch.tensor([1.0, -1.0] * (3 * 4), dtype=torch.float64)
    g_test[0] = torch.diag(diag_vals.to(torch.complex128))
    mean_test = local_charge_cell_mean_batch_torch(g_test, nx=3, ny=4).detach().cpu().numpy()
    var_test, imag_max = local_charge_cell_variance_batch_torch(g_test, nx=3, ny=4)
    total_var_test, imag_max_total = total_charge_variance_batch_torch(g_test)
    if imag_max > CHARGE_VARIANCE_IMAG_TOL:
        raise AssertionError("Diagonal charge variance should be real")
    if imag_max_total > CHARGE_VARIANCE_IMAG_TOL:
        raise AssertionError("Diagonal total charge variance should be real")
    if not np.allclose(mean_test, 1.0, atol=1e-12, rtol=1e-12):
        raise AssertionError("Diagonal occupancy test should have unit cell charge mean equal to one")
    if not np.allclose(var_test.detach().cpu().numpy(), 0.0, atol=1e-12, rtol=1e-12):
        raise AssertionError("Diagonal occupancy test should have zero cell charge variance")
    if not np.allclose(total_var_test.detach().cpu().numpy(), 0.0, atol=1e-12, rtol=1e-12):
        raise AssertionError("Diagonal occupancy test should have zero total charge variance")

    partitions = build_chern_partition_indices(nx=3, ny=4, device="cpu")
    cherns = real_space_chern_batch_torch(g_test, partitions).detach().cpu().numpy()
    if not np.allclose(cherns, 0.0, atol=1e-12, rtol=1e-12):
        raise AssertionError("Diagonal test covariance should yield zero real-space Chern in this partition")

    try:
        try:
            from classA_U1FGTN_gpu import classA_U1FGTN_gpu
        except ImportError:
            from fgtn.classA_U1FGTN_gpu import classA_U1FGTN_gpu

        gpu_model = classA_U1FGTN_gpu(
            Nx=3,
            Ny=4,
            DW=False,
            nshell=1,
            device="cpu",
            dtype="complex128",
            backend="local",
        )
        born_maps = gpu_model.cycle_end_born_probabilities_batch(zero)
        for name, value in born_maps.items():
            arr = value.detach().cpu().numpy()
            if arr.shape != (2, 3, 4):
                raise AssertionError(f"Unexpected Born probability shape for {name}: {arr.shape}")
            if not np.allclose(arr, 0.5, atol=1e-12, rtol=1e-12):
                raise AssertionError(f"Maxmix Born probabilities should be exactly one half for {name}")
    except ImportError:
        pass

    try:
        try:
            from classA_U1FGTN import classA_U1FGTN
        except ImportError:
            from fgtn.classA_U1FGTN import classA_U1FGTN

        cpu_model = classA_U1FGTN(Nx=3, Ny=4, DW=False, nshell=1)
        rng = np.random.default_rng(0)
        random_real = rng.standard_normal((2 * 3 * 4, 2 * 3 * 4))
        random_imag = rng.standard_normal((2 * 3 * 4, 2 * 3 * 4))
        g_ref = random_real + 1j * random_imag
        g_ref = 0.5 * (g_ref + g_ref.conj().T)
        g_ref_torch = torch.as_tensor(g_ref, dtype=torch.complex128)
        chern_ref = cpu_model.real_space_chern_number(
            g_ref,
            xref=cpu_model.Nx // 2,
            yref=cpu_model.Ny // 2,
            radius=0.4 * min(cpu_model.Nx, cpu_model.Ny),
        )
        chern_fast = real_space_chern_batch_torch(g_ref_torch.unsqueeze(0), partitions)[0].detach().cpu().item()
        if not np.allclose(chern_fast, chern_ref, atol=1e-8, rtol=1e-8):
            raise AssertionError("Real-space Chern implementation failed the CPU reference check")
    except ImportError:
        pass


class PurificationDynamicsObservables:
    def __init__(
        self,
        *,
        nx: int,
        ny: int,
        cycles: int,
        samples_expected: int,
        protocol: str,
        nshell: int,
        dw_loc: Iterable[int] | None = None,
        entropy_eps: float = 1e-12,
        trace_imag_tol: float = TRACE_IMAG_TOL,
        herm_tol: float = HERM_TOL,
        charge_variance_imag_tol: float = CHARGE_VARIANCE_IMAG_TOL,
    ) -> None:
        self.nx = int(nx)
        self.ny = int(ny)
        self.cycles = int(cycles)
        self.samples_expected = int(samples_expected)
        self.protocol = str(protocol)
        self.nshell = int(nshell)
        self.entropy_eps = float(entropy_eps)
        self.trace_imag_tol = float(trace_imag_tol)
        self.herm_tol = float(herm_tol)
        self.charge_variance_imag_tol = float(charge_variance_imag_tol)
        self.nlayer = 2 * self.nx * self.ny
        self.actual_samples: int | None = None

        scalar_shape = (self.samples_expected, self.cycles)
        map_shape = (self.samples_expected, self.cycles, self.nx, self.ny)
        self.total_entropy = np.full(scalar_shape, np.nan, dtype=np.float64)
        self.real_space_chern = np.full(scalar_shape, np.nan, dtype=np.float64)
        self.total_charge_variance = np.full(scalar_shape, np.nan, dtype=np.float64)
        self.local_charge_cell_mean = np.full(map_shape, np.nan, dtype=np.float64)
        self.local_charge_cell_variance = np.full(map_shape, np.nan, dtype=np.float64)
        self.N_A_lower = np.full(map_shape, np.nan, dtype=np.float64)
        self.N_B_lower = np.full(map_shape, np.nan, dtype=np.float64)
        self.one_minus_N_A_upper = np.full(map_shape, np.nan, dtype=np.float64)
        self.one_minus_N_B_upper = np.full(map_shape, np.nan, dtype=np.float64)

        self.observer_stats = {
            "full_eigh_cpu_fallback_count": 0,
            "full_occupation_eval_min": np.inf,
            "full_occupation_eval_max": -np.inf,
            "full_max_hermiticity_error": 0.0,
            "full_batch_trace_imag_max_abs": 0.0,
            "full_batch_hermitian_max_err": 0.0,
            "charge_variance_imag_max_abs": 0.0,
            "total_entropy_min": np.inf,
            "total_entropy_max": -np.inf,
            "real_space_chern_min": np.inf,
            "real_space_chern_max": -np.inf,
            "total_charge_variance_imag_max_abs": 0.0,
            "total_charge_variance_min": np.inf,
            "total_charge_variance_max": -np.inf,
            "local_charge_mean_min": np.inf,
            "local_charge_mean_max": -np.inf,
            "local_charge_variance_min": np.inf,
            "local_charge_variance_max": -np.inf,
            "N_A_lower_min": np.inf,
            "N_A_lower_max": -np.inf,
            "N_B_lower_min": np.inf,
            "N_B_lower_max": -np.inf,
            "one_minus_N_A_upper_min": np.inf,
            "one_minus_N_A_upper_max": -np.inf,
            "one_minus_N_B_upper_min": np.inf,
            "one_minus_N_B_upper_max": -np.inf,
        }
        self._progress_bar = None
        self._chern_partitions_cache: dict[str, dict[str, Any]] = {}
        self._born_prob_model = None

        dw_loc_list = [int(x) for x in dw_loc] if dw_loc is not None else []
        if len(dw_loc_list) == 2:
            self.xref = int(math.floor((dw_loc_list[0] + dw_loc_list[1]) / 2))
        else:
            self.xref = self.nx // 2
        self.yref = self.ny // 2
        self.radius = 0.4 * min(self.nx, self.ny)

    def _chern_partitions_for_device(self, device: torch.device) -> dict[str, Any]:
        key = str(device)
        if key not in self._chern_partitions_cache:
            self._chern_partitions_cache[key] = build_chern_partition_indices(
                nx=self.nx,
                ny=self.ny,
                xref=self.xref,
                yref=self.yref,
                radius=self.radius,
                device=device,
            )
        return self._chern_partitions_cache[key]

    def _validate_top_layer_batch(self, g_batch: torch.Tensor) -> None:
        if g_batch.ndim != 3:
            raise ValueError(f"Expected G_batch with shape (B,N,N), got {tuple(g_batch.shape)}")
        _, nlayer, nlayer_2 = g_batch.shape
        if nlayer != nlayer_2 or nlayer != self.nlayer:
            raise ValueError(f"Expected covariance shape (B,{self.nlayer},{self.nlayer}), got {tuple(g_batch.shape)}")
        if not torch.isfinite(g_batch).all():
            bad = torch.nonzero(~torch.isfinite(g_batch), as_tuple=False)[0].detach().cpu().tolist()
            raise FloatingPointError(f"Non-finite covariance batch entry encountered at index {bad}")

    def _store_born_probability_maps(self, *, sl: slice, cycle_idx: int, born_maps: dict[str, torch.Tensor]) -> None:
        for name in ("N_A_lower", "N_B_lower", "one_minus_N_A_upper", "one_minus_N_B_upper"):
            tensor = born_maps[name]
            arr = tensor.detach().cpu().numpy().astype(np.float64, copy=False)
            tol = float(BORN_PROB_TOL)
            arr_min = float(np.min(arr))
            arr_max = float(np.max(arr))
            if not np.isfinite(arr).all():
                raise FloatingPointError(f"{name} contains non-finite values")
            if arr_min < -tol or arr_max > 1.0 + tol:
                raise FloatingPointError(
                    f"{name} left the probability interval: [{arr_min:.6e}, {arr_max:.6e}]"
                )
            arr = np.clip(arr, 0.0, 1.0)
            getattr(self, name)[sl, cycle_idx] = arr
            self.observer_stats[f"{name}_min"] = min(float(self.observer_stats[f"{name}_min"]), float(np.min(arr)))
            self.observer_stats[f"{name}_max"] = max(float(self.observer_stats[f"{name}_max"]), float(np.max(arr)))

    def observe(self, *, cycle: int, G: torch.Tensor, batch_index: int, batch_start: int, batch_count: int) -> None:
        del batch_index
        cycle = int(cycle)
        if cycle == 0:
            return
        g_work = G.detach().clone()
        self._validate_top_layer_batch(g_work)
        batch_start = int(batch_start)
        batch_count = int(batch_count)
        sl = slice(batch_start, batch_start + batch_count)
        cycle_idx = cycle - 1

        herm_vals = torch.amax(torch.abs(g_work - g_work.conj().transpose(-2, -1)), dim=(-2, -1)).to(torch.float64)
        herm_max = float(torch.max(herm_vals).detach().cpu())
        self.observer_stats["full_batch_hermitian_max_err"] = max(
            float(self.observer_stats["full_batch_hermitian_max_err"]),
            herm_max,
        )
        if herm_max > self.herm_tol:
            raise FloatingPointError(
                f"Full covariance Hermitian error exceeded tolerance at cycle={cycle}: "
                f"{herm_max:.6e} > {self.herm_tol:.6e}"
            )

        trace_vals = torch.diagonal(g_work, dim1=-2, dim2=-1).sum(dim=-1)
        trace_imag_abs = torch.abs(trace_vals.imag).to(torch.float64)
        trace_imag_max = float(torch.max(trace_imag_abs).detach().cpu())
        self.observer_stats["full_batch_trace_imag_max_abs"] = max(
            float(self.observer_stats["full_batch_trace_imag_max_abs"]),
            trace_imag_max,
        )
        if trace_imag_max > self.trace_imag_tol:
            raise FloatingPointError(
                f"Trace imaginary part exceeded tolerance at cycle={cycle}: "
                f"{trace_imag_max:.6e} > {self.trace_imag_tol:.6e}"
            )

        total_entropy, entropy_metrics = entropy_total_batch_torch(g_work, eps=self.entropy_eps, validate=True)
        total_entropy_np = total_entropy.detach().cpu().numpy().astype(np.float64, copy=False)
        self.total_entropy[sl, cycle_idx] = total_entropy_np
        self.observer_stats["full_eigh_cpu_fallback_count"] += int(entropy_metrics["eigh_cpu_fallback_count"])
        self.observer_stats["full_occupation_eval_min"] = min(
            float(self.observer_stats["full_occupation_eval_min"]),
            float(entropy_metrics["min_occupation_eval"]),
        )
        self.observer_stats["full_occupation_eval_max"] = max(
            float(self.observer_stats["full_occupation_eval_max"]),
            float(entropy_metrics["max_occupation_eval"]),
        )
        self.observer_stats["full_max_hermiticity_error"] = max(
            float(self.observer_stats["full_max_hermiticity_error"]),
            float(entropy_metrics["max_hermiticity_error"]),
        )
        self.observer_stats["total_entropy_min"] = min(
            float(self.observer_stats["total_entropy_min"]),
            float(np.min(total_entropy_np)),
        )
        self.observer_stats["total_entropy_max"] = max(
            float(self.observer_stats["total_entropy_max"]),
            float(np.max(total_entropy_np)),
        )

        partitions = self._chern_partitions_for_device(g_work.device)
        chern_vals = real_space_chern_batch_torch(g_work, partitions)
        chern_np = chern_vals.detach().cpu().numpy().astype(np.float64, copy=False)
        self.real_space_chern[sl, cycle_idx] = chern_np
        self.observer_stats["real_space_chern_min"] = min(
            float(self.observer_stats["real_space_chern_min"]),
            float(np.min(chern_np)),
        )
        self.observer_stats["real_space_chern_max"] = max(
            float(self.observer_stats["real_space_chern_max"]),
            float(np.max(chern_np)),
        )

        total_charge_variance, total_variance_imag_max = total_charge_variance_batch_torch(g_work)
        if total_variance_imag_max > self.charge_variance_imag_tol:
            raise FloatingPointError(
                f"Total charge variance imaginary leakage exceeded tolerance at cycle={cycle}: "
                f"{total_variance_imag_max:.6e} > {self.charge_variance_imag_tol:.6e}"
            )
        total_charge_variance_np = total_charge_variance.detach().cpu().numpy().astype(np.float64, copy=False)
        self.total_charge_variance[sl, cycle_idx] = total_charge_variance_np
        self.observer_stats["total_charge_variance_imag_max_abs"] = max(
            float(self.observer_stats["total_charge_variance_imag_max_abs"]),
            float(total_variance_imag_max),
        )
        self.observer_stats["total_charge_variance_min"] = min(
            float(self.observer_stats["total_charge_variance_min"]),
            float(np.min(total_charge_variance_np)),
        )
        self.observer_stats["total_charge_variance_max"] = max(
            float(self.observer_stats["total_charge_variance_max"]),
            float(np.max(total_charge_variance_np)),
        )

        charge_mean = local_charge_cell_mean_batch_torch(g_work, nx=self.nx, ny=self.ny)
        charge_mean_np = charge_mean.detach().cpu().numpy().astype(np.float64, copy=False)
        self.local_charge_cell_mean[sl, cycle_idx] = charge_mean_np
        self.observer_stats["local_charge_mean_min"] = min(
            float(self.observer_stats["local_charge_mean_min"]),
            float(np.min(charge_mean_np)),
        )
        self.observer_stats["local_charge_mean_max"] = max(
            float(self.observer_stats["local_charge_mean_max"]),
            float(np.max(charge_mean_np)),
        )

        charge_variance, variance_imag_max = local_charge_cell_variance_batch_torch(g_work, nx=self.nx, ny=self.ny)
        if variance_imag_max > self.charge_variance_imag_tol:
            raise FloatingPointError(
                f"Charge variance imaginary leakage exceeded tolerance at cycle={cycle}: "
                f"{variance_imag_max:.6e} > {self.charge_variance_imag_tol:.6e}"
            )
        charge_variance_np = charge_variance.detach().cpu().numpy().astype(np.float64, copy=False)
        self.local_charge_cell_variance[sl, cycle_idx] = charge_variance_np
        self.observer_stats["charge_variance_imag_max_abs"] = max(
            float(self.observer_stats["charge_variance_imag_max_abs"]),
            float(variance_imag_max),
        )
        self.observer_stats["local_charge_variance_min"] = min(
            float(self.observer_stats["local_charge_variance_min"]),
            float(np.min(charge_variance_np)),
        )
        self.observer_stats["local_charge_variance_max"] = max(
            float(self.observer_stats["local_charge_variance_max"]),
            float(np.max(charge_variance_np)),
        )

        if self._born_prob_model is None:
            raise RuntimeError("Born-probability model is not configured. Call make_cycle_observer(model=...) first.")
        born_maps = self._born_prob_model.cycle_end_born_probabilities_batch(g_work)
        self._store_born_probability_maps(sl=sl, cycle_idx=cycle_idx, born_maps=born_maps)

        if self._progress_bar is not None:
            self._progress_bar.update(batch_count)

    def make_cycle_observer(self, model, progress_bar=None):
        self._born_prob_model = model
        self._progress_bar = progress_bar

        def _observer(*, cycle, G, batch_index, batch_start, batch_count):
            self.observe(
                cycle=cycle,
                G=G,
                batch_index=batch_index,
                batch_start=batch_start,
                batch_count=batch_count,
            )

        return _observer

    def finalize(self, actual_samples: int) -> None:
        actual_samples = int(actual_samples)
        if actual_samples <= 0 or actual_samples > self.samples_expected:
            raise ValueError(
                f"actual_samples must satisfy 1 <= actual_samples <= {self.samples_expected}; got {actual_samples}"
            )
        self.actual_samples = actual_samples
        sample_slice = slice(0, actual_samples)
        required_arrays = {
            "total_entropy": self.total_entropy[sample_slice],
            "real_space_chern": self.real_space_chern[sample_slice],
            "total_charge_variance": self.total_charge_variance[sample_slice],
            "local_charge_cell_mean": self.local_charge_cell_mean[sample_slice],
            "local_charge_cell_variance": self.local_charge_cell_variance[sample_slice],
            "N_A_lower": self.N_A_lower[sample_slice],
            "N_B_lower": self.N_B_lower[sample_slice],
            "one_minus_N_A_upper": self.one_minus_N_A_upper[sample_slice],
            "one_minus_N_B_upper": self.one_minus_N_B_upper[sample_slice],
        }
        for name, value in required_arrays.items():
            if np.isnan(np.asarray(value, dtype=np.float64)).any():
                raise RuntimeError(f"{name} contains unfilled NaN entries after finalize().")

    def metrics_dataframe(self) -> pd.DataFrame:
        if self.actual_samples is None:
            raise RuntimeError("Call finalize(actual_samples=...) before exporting data.")
        rows = []
        for sample_index in range(self.actual_samples):
            for cycle_label in range(1, self.cycles + 1):
                cycle_idx = cycle_label - 1
                rows.append(
                    {
                        "config_id": config_id(self.nx, self.ny, self.protocol),
                        "geometry_key": geometry_key(self.nx, self.ny),
                        "protocol": self.protocol,
                        "Nx": self.nx,
                        "Ny": self.ny,
                        "nshell": self.nshell,
                        "sample_index": int(sample_index),
                        "cycle_label": int(cycle_label),
                        "total_entropy": float(self.total_entropy[sample_index, cycle_idx]),
                        "real_space_chern": float(self.real_space_chern[sample_index, cycle_idx]),
                        "total_charge_variance": float(self.total_charge_variance[sample_index, cycle_idx]),
                    }
                )
        return pd.DataFrame(rows)

    def _base_payload(self, *, config: dict[str, Any]) -> dict[str, Any]:
        if self.actual_samples is None:
            raise RuntimeError("Call finalize(actual_samples=...) before exporting data.")
        return {
            "cycle_labels": np.arange(1, self.cycles + 1, dtype=np.int64),
            "sample_indices": np.arange(self.actual_samples, dtype=np.int64),
            "config_json": np.asarray(json.dumps(config, sort_keys=True)),
            "helper_version": np.asarray(HELPER_VERSION),
        }

    def total_entropy_npz_payload(self, *, config: dict[str, Any]) -> dict[str, Any]:
        return {
            **self._base_payload(config=config),
            "total_entropy": self.total_entropy[: self.actual_samples].copy(),
            "formula_total_entropy": np.asarray("-Tr[C log C + (I-C) log(I-C)] with C=(G+I)/2"),
        }

    def local_charge_mean_npz_payload(self, *, config: dict[str, Any]) -> dict[str, Any]:
        return {
            **self._base_payload(config=config),
            "local_charge_cell_mean": self.local_charge_cell_mean[: self.actual_samples].copy(),
            "x_coords": np.arange(self.nx, dtype=np.int64),
            "y_coords": np.arange(self.ny, dtype=np.int64),
            "convention": np.asarray("cell charge mean = C_aa + C_bb with C=(G+I)/2"),
        }

    def total_charge_variance_npz_payload(self, *, config: dict[str, Any]) -> dict[str, Any]:
        return {
            **self._base_payload(config=config),
            "total_charge_variance": self.total_charge_variance[: self.actual_samples].copy(),
            "formula_total_charge_variance": np.asarray("Tr[C - C^2] with C=(G+I)/2"),
        }

    def local_charge_variance_npz_payload(self, *, config: dict[str, Any]) -> dict[str, Any]:
        return {
            **self._base_payload(config=config),
            "local_charge_cell_variance": self.local_charge_cell_variance[: self.actual_samples].copy(),
            "x_coords": np.arange(self.nx, dtype=np.int64),
            "y_coords": np.arange(self.ny, dtype=np.int64),
            "convention": np.asarray(
                "Var(N_r)=(C_aa+C_bb)+2(C_aa C_bb-C_ab C_ba)-(C_aa+C_bb)^2 with C=(G+I)/2"
            ),
        }

    def born_rule_probabilities_npz_payload(self, *, config: dict[str, Any]) -> dict[str, Any]:
        return {
            **self._base_payload(config=config),
            "N_A_lower": self.N_A_lower[: self.actual_samples].copy(),
            "N_B_lower": self.N_B_lower[: self.actual_samples].copy(),
            "one_minus_N_A_upper": self.one_minus_N_A_upper[: self.actual_samples].copy(),
            "one_minus_N_B_upper": self.one_minus_N_B_upper[: self.actual_samples].copy(),
            "x_coords": np.arange(self.nx, dtype=np.int64),
            "y_coords": np.arange(self.ny, dtype=np.int64),
            "formula_N_A_lower": np.asarray("p_occ(Am | G_end_of_cycle)"),
            "formula_N_B_lower": np.asarray("p_occ(Bm | G_end_of_cycle)"),
            "formula_one_minus_N_A_upper": np.asarray("1 - p_occ(Ap | G_end_of_cycle)"),
            "formula_one_minus_N_B_upper": np.asarray("1 - p_occ(Bp | G_end_of_cycle)"),
        }

    def run_summary_metrics(self) -> dict[str, Any]:
        eval_min = self.observer_stats["full_occupation_eval_min"]
        eval_max = self.observer_stats["full_occupation_eval_max"]
        return {
            "observer_stats": {
                "full_eigh_cpu_fallback_count": int(self.observer_stats["full_eigh_cpu_fallback_count"]),
                "full_occupation_eval_min": None if not np.isfinite(eval_min) else float(eval_min),
                "full_occupation_eval_max": None if not np.isfinite(eval_max) else float(eval_max),
                "full_max_hermiticity_error": float(self.observer_stats["full_max_hermiticity_error"]),
                "full_batch_trace_imag_max_abs": float(self.observer_stats["full_batch_trace_imag_max_abs"]),
                "full_batch_hermitian_max_err": float(self.observer_stats["full_batch_hermitian_max_err"]),
                "charge_variance_imag_max_abs": float(self.observer_stats["charge_variance_imag_max_abs"]),
                "total_entropy_min": None
                if not np.isfinite(self.observer_stats["total_entropy_min"])
                else float(self.observer_stats["total_entropy_min"]),
                "total_entropy_max": None
                if not np.isfinite(self.observer_stats["total_entropy_max"])
                else float(self.observer_stats["total_entropy_max"]),
                "real_space_chern_min": None
                if not np.isfinite(self.observer_stats["real_space_chern_min"])
                else float(self.observer_stats["real_space_chern_min"]),
                "real_space_chern_max": None
                if not np.isfinite(self.observer_stats["real_space_chern_max"])
                else float(self.observer_stats["real_space_chern_max"]),
                "total_charge_variance_imag_max_abs": float(
                    self.observer_stats["total_charge_variance_imag_max_abs"]
                ),
                "total_charge_variance_min": None
                if not np.isfinite(self.observer_stats["total_charge_variance_min"])
                else float(self.observer_stats["total_charge_variance_min"]),
                "total_charge_variance_max": None
                if not np.isfinite(self.observer_stats["total_charge_variance_max"])
                else float(self.observer_stats["total_charge_variance_max"]),
                "local_charge_mean_min": None
                if not np.isfinite(self.observer_stats["local_charge_mean_min"])
                else float(self.observer_stats["local_charge_mean_min"]),
                "local_charge_mean_max": None
                if not np.isfinite(self.observer_stats["local_charge_mean_max"])
                else float(self.observer_stats["local_charge_mean_max"]),
                "local_charge_variance_min": None
                if not np.isfinite(self.observer_stats["local_charge_variance_min"])
                else float(self.observer_stats["local_charge_variance_min"]),
                "local_charge_variance_max": None
                if not np.isfinite(self.observer_stats["local_charge_variance_max"])
                else float(self.observer_stats["local_charge_variance_max"]),
                "N_A_lower_min": None
                if not np.isfinite(self.observer_stats["N_A_lower_min"])
                else float(self.observer_stats["N_A_lower_min"]),
                "N_A_lower_max": None
                if not np.isfinite(self.observer_stats["N_A_lower_max"])
                else float(self.observer_stats["N_A_lower_max"]),
                "N_B_lower_min": None
                if not np.isfinite(self.observer_stats["N_B_lower_min"])
                else float(self.observer_stats["N_B_lower_min"]),
                "N_B_lower_max": None
                if not np.isfinite(self.observer_stats["N_B_lower_max"])
                else float(self.observer_stats["N_B_lower_max"]),
                "one_minus_N_A_upper_min": None
                if not np.isfinite(self.observer_stats["one_minus_N_A_upper_min"])
                else float(self.observer_stats["one_minus_N_A_upper_min"]),
                "one_minus_N_A_upper_max": None
                if not np.isfinite(self.observer_stats["one_minus_N_A_upper_max"])
                else float(self.observer_stats["one_minus_N_A_upper_max"]),
                "one_minus_N_B_upper_min": None
                if not np.isfinite(self.observer_stats["one_minus_N_B_upper_min"])
                else float(self.observer_stats["one_minus_N_B_upper_min"]),
                "one_minus_N_B_upper_max": None
                if not np.isfinite(self.observer_stats["one_minus_N_B_upper_max"])
                else float(self.observer_stats["one_minus_N_B_upper_max"]),
            }
        }
