from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


HELPER_VERSION = "lyapunov_observables_gpu_v1"
CANONICAL_ENTRY_POINT = "classA_U1FGTN_gpu.run_markov_circuit"
CHANNEL_ORDER = ("Ap", "Am", "Bp", "Bm")


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


def geometry_key(nx: int, ny: int) -> str:
    return f"N{int(nx)}x{int(ny)}"


def config_id(cfg: dict[str, Any]) -> str:
    alpha2_tag = f"{float(cfg['alpha_2']):g}".replace(".", "p")
    return (
        f"{geometry_key(cfg['Nx'], cfg['Ny'])}_DW{int(bool(cfg['DW']))}"
        f"_dwtrunc{int(bool(cfg['dw_truncation']))}_a2-{alpha2_tag}"
        f"_nsh{int(cfg['nshell'])}_{cfg['protocol']}"
    )


def case_output_dir(runs_root: Path, cfg: dict[str, Any]) -> Path:
    return Path(runs_root) / config_id(cfg)


def domain_wall_metadata(model: Any) -> dict[str, Any]:
    dw_loc = [int(x) for x in getattr(model, "DW_loc", [])]
    payload: dict[str, Any] = {"dw_loc": dw_loc}
    if len(dw_loc) == 2:
        payload["topological_x_range"] = [dw_loc[0], dw_loc[1]]
        payload["trijunction_xref"] = int(math.floor((dw_loc[0] + dw_loc[1]) / 2))
    else:
        payload["topological_x_range"] = None
        payload["trijunction_xref"] = int(model.Nx) // 2
    payload["trijunction_yref"] = int(model.Ny) // 2
    payload["trijunction_radius"] = 0.4 * min(int(model.Nx), int(model.Ny))
    return payload


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
    a2 = 2.0 * math.pi / 3.0
    a4 = 4.0 * math.pi / 3.0
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


def local_charge_cell_batch_torch(g_batch: torch.Tensor, *, nx: int, ny: int) -> torch.Tensor:
    diag = torch.diagonal(g_batch, dim1=-2, dim2=-1)
    occ = 0.5 * (diag.real.to(torch.float64) + 1.0)
    occ = occ.reshape(int(g_batch.shape[0]), int(ny), int(nx), 2)
    return occ.sum(dim=-1).transpose(1, 2).contiguous()


class LyapunovSmokeObserver:
    def __init__(
        self,
        *,
        nx: int,
        ny: int,
        cycles: int,
        samples_expected: int,
        n_vec: int,
        protocol: str,
        chern_metadata: dict[str, Any],
    ) -> None:
        self.nx = int(nx)
        self.ny = int(ny)
        self.cycles = int(cycles)
        self.samples_expected = int(samples_expected)
        self.n_vec = int(n_vec)
        self.protocol = str(protocol)
        self.chern_metadata = dict(chern_metadata)
        self.lyapunov_spectra = np.full((self.samples_expected, self.cycles, self.n_vec), np.nan, dtype=np.float64)
        self.real_space_chern = np.full((self.samples_expected, self.cycles), np.nan, dtype=np.float64)
        self.local_charge_cell = np.full((self.samples_expected, self.cycles, self.nx, self.ny), np.nan, dtype=np.float64)
        self.lyapunov_min_abs_vector = np.full(
            (self.samples_expected, 2 * self.nx * self.ny),
            np.nan + 1j * np.nan,
            dtype=np.complex128,
        )
        self.lyapunov_min_abs_value = np.full((self.samples_expected,), np.nan, dtype=np.float64)
        self.lyapunov_min_abs_index = np.full((self.samples_expected,), -1, dtype=np.int64)
        self.actual_samples: int | None = None
        self._chern_partitions_cache: dict[str, dict[str, Any]] = {}

    def _chern_partitions_for_device(self, device: torch.device) -> dict[str, Any]:
        key = str(device)
        if key not in self._chern_partitions_cache:
            self._chern_partitions_cache[key] = build_chern_partition_indices(
                nx=self.nx,
                ny=self.ny,
                xref=int(self.chern_metadata["trijunction_xref"]),
                yref=int(self.chern_metadata["trijunction_yref"]),
                radius=float(self.chern_metadata["trijunction_radius"]),
                device=device,
            )
        return self._chern_partitions_cache[key]

    def __call__(
        self,
        *,
        cycle: int,
        spectra: torch.Tensor,
        G: torch.Tensor,
        batch_index: int,
        batch_start: int,
        batch_count: int,
        lyapunov_min_abs_vector: torch.Tensor | None = None,
        lyapunov_min_abs_value: torch.Tensor | None = None,
        lyapunov_min_abs_index: torch.Tensor | None = None,
    ) -> None:
        cycle_idx = int(cycle) - 1
        if cycle_idx < 0 or cycle_idx >= self.cycles:
            raise ValueError(f"cycle must be in 1..{self.cycles}; got {cycle}")
        start = int(batch_start)
        stop = start + int(batch_count)
        if stop > self.samples_expected:
            raise ValueError(f"Batch slice {start}:{stop} exceeds expected samples {self.samples_expected}")
        sl = slice(start, stop)
        spectra_np = spectra.detach().cpu().numpy().astype(np.float64, copy=False)
        if spectra_np.shape != (int(batch_count), self.n_vec):
            raise ValueError(f"Unexpected spectra shape {spectra_np.shape}; expected {(int(batch_count), self.n_vec)}")
        self.lyapunov_spectra[sl, cycle_idx, :] = spectra_np

        g_work = G
        if g_work.ndim != 3:
            raise ValueError(f"Expected G shape (B,N,N), got {tuple(g_work.shape)}")
        partitions = self._chern_partitions_for_device(g_work.device)
        chern = real_space_chern_batch_torch(g_work, partitions)
        charge = local_charge_cell_batch_torch(g_work, nx=self.nx, ny=self.ny)
        self.real_space_chern[sl, cycle_idx] = chern.detach().cpu().numpy().astype(np.float64, copy=False)
        self.local_charge_cell[sl, cycle_idx, :, :] = charge.detach().cpu().numpy().astype(np.float64, copy=False)

        has_vector_payload = lyapunov_min_abs_vector is not None
        if has_vector_payload != (lyapunov_min_abs_value is not None) or has_vector_payload != (lyapunov_min_abs_index is not None):
            raise ValueError("Incomplete final Lyapunov min-abs vector payload.")
        if has_vector_payload:
            if cycle_idx != self.cycles - 1:
                raise ValueError("Lyapunov min-abs vectors should only be emitted at the final cycle.")
            vector_np = lyapunov_min_abs_vector.detach().cpu().numpy().astype(np.complex128, copy=False)
            value_np = lyapunov_min_abs_value.detach().cpu().numpy().astype(np.float64, copy=False)
            index_np = lyapunov_min_abs_index.detach().cpu().numpy().astype(np.int64, copy=False)
            if vector_np.shape != (int(batch_count), 2 * self.nx * self.ny):
                raise ValueError(
                    f"Unexpected min-abs vector shape {vector_np.shape}; "
                    f"expected {(int(batch_count), 2 * self.nx * self.ny)}"
                )
            if value_np.shape != (int(batch_count),):
                raise ValueError(f"Unexpected min-abs value shape {value_np.shape}; expected {(int(batch_count),)}")
            if index_np.shape != (int(batch_count),):
                raise ValueError(f"Unexpected min-abs index shape {index_np.shape}; expected {(int(batch_count),)}")
            self.lyapunov_min_abs_vector[sl, :] = vector_np
            self.lyapunov_min_abs_value[sl] = value_np
            self.lyapunov_min_abs_index[sl] = index_np

    def finalize(self, *, actual_samples: int) -> None:
        self.actual_samples = int(actual_samples)
        if self.actual_samples <= 0 or self.actual_samples > self.samples_expected:
            raise ValueError(f"Invalid actual_samples={self.actual_samples}")
        arrays = [
            ("lyapunov_spectra", self.lyapunov_spectra[: self.actual_samples]),
            ("real_space_chern", self.real_space_chern[: self.actual_samples]),
            ("local_charge_cell", self.local_charge_cell[: self.actual_samples]),
            ("lyapunov_min_abs_vector", self.lyapunov_min_abs_vector[: self.actual_samples]),
            ("lyapunov_min_abs_value", self.lyapunov_min_abs_value[: self.actual_samples]),
        ]
        for name, arr in arrays:
            if not np.isfinite(arr).all():
                raise FloatingPointError(f"Non-finite entries remain in {name}.")
        if np.any(self.lyapunov_min_abs_index[: self.actual_samples] < 0):
            raise FloatingPointError("Missing final Lyapunov min-abs vector indices.")

    def metrics_dataframe(self) -> pd.DataFrame:
        actual = self.samples_expected if self.actual_samples is None else self.actual_samples
        rows = []
        spectra = self.lyapunov_spectra[:actual]
        charge = self.local_charge_cell[:actual]
        for sample_idx in range(actual):
            for cycle_idx in range(self.cycles):
                spec = spectra[sample_idx, cycle_idx]
                rows.append(
                    {
                        "sample_index": sample_idx,
                        "cycle": cycle_idx + 1,
                        "protocol": self.protocol,
                        "lyapunov_min": float(np.min(spec)),
                        "lyapunov_max": float(np.max(spec)),
                        "lyapunov_mean": float(np.mean(spec)),
                        "real_space_chern": float(self.real_space_chern[sample_idx, cycle_idx]),
                        "total_charge": float(np.sum(charge[sample_idx, cycle_idx])),
                    }
                )
        return pd.DataFrame(rows)

    def spectra_npz_payload(self, *, config: dict[str, Any]) -> dict[str, Any]:
        actual = self.samples_expected if self.actual_samples is None else self.actual_samples
        return {
            "lyapunov_spectra": self.lyapunov_spectra[:actual].copy(),
            "config_json": np.asarray(json.dumps(config, sort_keys=True)),
            "helper_version": np.asarray(HELPER_VERSION),
            "channel_order": np.asarray(CHANNEL_ORDER),
            "normalization": np.asarray("log singular values divided by cycle index"),
        }

    def min_abs_vector_npz_payload(self, *, config: dict[str, Any]) -> dict[str, Any]:
        actual = self.samples_expected if self.actual_samples is None else self.actual_samples
        return {
            "lyapunov_min_abs_vector": self.lyapunov_min_abs_vector[:actual].copy(),
            "lyapunov_min_abs_value": self.lyapunov_min_abs_value[:actual].copy(),
            "lyapunov_min_abs_index": self.lyapunov_min_abs_index[:actual].copy(),
            "config_json": np.asarray(json.dumps(config, sort_keys=True)),
            "helper_version": np.asarray(HELPER_VERSION),
            "description": np.asarray(
                "Final-cycle QR Lyapunov-frame column whose unsorted finite-time exponent has minimal absolute value."
            ),
            "basis": np.asarray("top-layer complex orbital basis"),
        }

    def chern_npz_payload(self, *, config: dict[str, Any]) -> dict[str, Any]:
        actual = self.samples_expected if self.actual_samples is None else self.actual_samples
        return {
            "real_space_chern": self.real_space_chern[:actual].copy(),
            "config_json": np.asarray(json.dumps(config, sort_keys=True)),
            "chern_metadata_json": np.asarray(json.dumps(self.chern_metadata, sort_keys=True)),
        }

    def charge_npz_payload(self, *, config: dict[str, Any]) -> dict[str, Any]:
        actual = self.samples_expected if self.actual_samples is None else self.actual_samples
        return {
            "local_charge_cell": self.local_charge_cell[:actual].copy(),
            "config_json": np.asarray(json.dumps(config, sort_keys=True)),
            "convention": np.asarray("cell charge = sum_mu (G_mumu + 1) / 2"),
        }


def run_internal_checks() -> None:
    g = torch.zeros((1, 8, 8), dtype=torch.complex128)
    partitions = build_chern_partition_indices(nx=2, ny=2, device=g.device)
    chern = real_space_chern_batch_torch(g, partitions).detach().cpu().numpy()
    if not np.all(np.isfinite(chern)):
        raise AssertionError("Chern check produced non-finite values.")
    charge = local_charge_cell_batch_torch(g, nx=2, ny=2).detach().cpu().numpy()
    if charge.shape != (1, 2, 2):
        raise AssertionError(f"Unexpected charge shape {charge.shape}")
