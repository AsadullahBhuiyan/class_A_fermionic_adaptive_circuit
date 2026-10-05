from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np

from production_runtime import save_npz_atomic, sha256_file


TANGENT_SCHEMA = "quenched_qr_tangent_record_v1"


class TangentFrameWriter:
    """Retain per-cycle QR increments and selected stabilized frames per trajectory."""

    def __init__(
        self,
        *,
        samples: int,
        physical_cycles: int,
        nlayer: int,
        nvec: int,
        alignment_cycles: int,
        frame_cycles: Iterable[int],
        nx: int | None = None,
        ny: int | None = None,
        basis_indices: Iterable[int] | None = None,
    ) -> None:
        self.samples = int(samples)
        self.physical_cycles = int(physical_cycles)
        self.nlayer = int(nlayer)
        self.nvec = int(nvec)
        self.alignment_cycles = int(alignment_cycles)
        self.frame_cycles = {int(value) for value in frame_cycles}
        self.nx = None if nx is None else int(nx)
        self.ny = None if ny is None else int(ny)
        if (self.nx is None) != (self.ny is None):
            raise ValueError("nx and ny must be provided together")
        self.basis_indices = (
            np.arange(self.nlayer, dtype=np.int64)
            if basis_indices is None
            else np.asarray(list(basis_indices), dtype=np.int64)
        )
        if self.basis_indices.shape != (self.nlayer,):
            raise ValueError("basis_indices must label every tangent-frame row")
        if self.nx is not None and (
            np.any(self.basis_indices < 0)
            or np.any(self.basis_indices >= 2 * self.nx * self.ny)
        ):
            raise ValueError("basis_indices lie outside the declared lattice")
        if not (0 <= self.alignment_cycles < self.physical_cycles):
            raise ValueError("alignment_cycles must lie in 0..physical_cycles-1")
        shape = (self.samples, self.physical_cycles, self.nvec)
        self.qr_log_increment = np.full(shape, np.nan, dtype=np.float64)
        self.null_mask = np.ones(shape, dtype=np.bool_)
        self.cumulative_spectrum = np.full(shape, np.nan, dtype=np.float64)
        self.qr_r = np.full(
            (self.samples, self.physical_cycles, self.nvec, self.nvec),
            np.nan + 1j * np.nan,
            dtype=np.complex128,
        )
        self.frames: dict[int, np.ndarray] = {}
        self.frame_x_weight: dict[int, np.ndarray] = {}
        self.active_final = np.zeros((self.samples,), dtype=np.bool_)
        self._device_buffers: dict[str, Any] | None = None
        self._device_frames: dict[int, Any] = {}

    @staticmethod
    def _cpu(value: Any) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return np.asarray(value)

    def _ensure_device_buffers(self, device: Any) -> None:
        if self._device_buffers is not None:
            return
        import torch

        shape = (self.samples, self.physical_cycles, self.nvec)
        self._device_buffers = {
            "qr_log_increment": torch.full(
                shape, torch.nan, dtype=torch.float64, device=device
            ),
            "null_mask": torch.ones(shape, dtype=torch.bool, device=device),
            "cumulative_spectrum": torch.full(
                shape, torch.nan, dtype=torch.float64, device=device
            ),
            "qr_r": torch.full(
                (self.samples, self.physical_cycles, self.nvec, self.nvec),
                complex(float("nan"), float("nan")),
                dtype=torch.complex128,
                device=device,
            ),
            "active_final": torch.zeros(
                (self.samples,), dtype=torch.bool, device=device
            ),
        }

    def _materialize_device_buffers(self) -> None:
        if self._device_buffers is None:
            return
        self.qr_log_increment = (
            self._device_buffers["qr_log_increment"].detach().cpu().numpy()
        )
        self.null_mask = self._device_buffers["null_mask"].detach().cpu().numpy()
        self.cumulative_spectrum = (
            self._device_buffers["cumulative_spectrum"].detach().cpu().numpy()
        )
        self.qr_r = self._device_buffers["qr_r"].detach().cpu().numpy()
        self.active_final = (
            self._device_buffers["active_final"].detach().cpu().numpy()
        )
        for cycle, value in self._device_frames.items():
            frame_cpu = value.detach().cpu().numpy()
            self.frames[cycle] = frame_cpu
            if self.nx is not None:
                row_x = (self.basis_indices // 2) % self.nx
                row_weights = np.abs(frame_cpu) ** 2
                weights = np.zeros(
                    (self.samples, self.nvec, self.nx), dtype=np.float64
                )
                for x in range(self.nx):
                    weights[:, :, x] = row_weights[:, row_x == x, :].sum(axis=1)
                self.frame_x_weight[cycle] = weights
        self._device_buffers = None
        self._device_frames.clear()

    def __call__(
        self,
        *,
        cycle: int,
        spectra: Any,
        batch_start: int,
        batch_count: int,
        lyapunov_qr_r: Any,
        lyapunov_frame: Any,
        lyapunov_cycle_null_mask: Any,
        lyapunov_active_mask: Any,
        **_: Any,
    ) -> None:
        cycle = int(cycle)
        index = cycle - 1
        if not (0 <= index < self.physical_cycles):
            raise IndexError(f"tangent cycle {cycle} outside 1..{self.physical_cycles}")
        start = int(batch_start)
        stop = start + int(batch_count)
        if (
            hasattr(lyapunov_qr_r, "detach")
            and getattr(lyapunov_qr_r.device, "type", None) == "cuda"
        ):
            import torch

            self._ensure_device_buffers(lyapunov_qr_r.device)
            buffers = self._device_buffers
            r = lyapunov_qr_r.to(torch.complex128)
            diag = torch.abs(torch.diagonal(r, dim1=-2, dim2=-1)).to(
                torch.float64
            )
            floor = torch.finfo(torch.float64).tiny
            buffers["qr_log_increment"][start:stop, index] = torch.log(
                torch.clamp(diag, min=floor)
            )
            buffers["null_mask"][start:stop, index] = (
                lyapunov_cycle_null_mask.to(torch.bool)
            )
            buffers["cumulative_spectrum"][start:stop, index] = spectra.to(
                torch.float64
            )
            buffers["qr_r"][start:stop, index] = r
            buffers["active_final"][start:stop] = lyapunov_active_mask.to(
                torch.bool
            )
            if cycle in self.frame_cycles:
                if cycle not in self._device_frames:
                    self._device_frames[cycle] = torch.full(
                        (self.samples, self.nlayer, self.nvec),
                        complex(float("nan"), float("nan")),
                        dtype=torch.complex128,
                        device=lyapunov_frame.device,
                    )
                self._device_frames[cycle][start:stop] = lyapunov_frame.to(
                    torch.complex128
                )
            return
        r = self._cpu(lyapunov_qr_r).astype(np.complex128, copy=False)
        diag = np.abs(np.diagonal(r, axis1=-2, axis2=-1))
        floor = np.finfo(np.float64).tiny
        self.qr_log_increment[start:stop, index] = np.log(np.maximum(diag, floor))
        self.null_mask[start:stop, index] = self._cpu(lyapunov_cycle_null_mask).astype(
            np.bool_, copy=False
        )
        self.cumulative_spectrum[start:stop, index] = self._cpu(spectra).astype(
            np.float64, copy=False
        )
        self.qr_r[start:stop, index] = r
        self.active_final[start:stop] = self._cpu(lyapunov_active_mask).astype(
            np.bool_, copy=False
        )
        if cycle in self.frame_cycles:
            frame_cpu = self._cpu(lyapunov_frame)
            self.frames.setdefault(
                cycle,
                np.full(
                    (self.samples, self.nlayer, self.nvec),
                    np.nan + 1j * np.nan,
                    dtype=np.complex128,
                ),
            )[start:stop] = frame_cpu
            if self.nx is not None:
                row_x = (self.basis_indices // 2) % self.nx
                row_weights = np.abs(frame_cpu) ** 2
                weights = np.zeros(
                    (int(batch_count), self.nvec, self.nx), dtype=np.float64
                )
                for x in range(self.nx):
                    weights[:, :, x] = row_weights[:, row_x == x, :].sum(axis=1)
                self.frame_x_weight.setdefault(
                    cycle,
                    np.full((self.samples, self.nvec, self.nx), np.nan, dtype=np.float64),
                )[start:stop] = weights

    def validate(self) -> dict[str, Any]:
        self._materialize_device_buffers()
        if not np.isfinite(self.qr_log_increment).all():
            raise FloatingPointError("tangent QR log increments are incomplete or non-finite")
        if not np.isfinite(self.qr_r).all():
            raise FloatingPointError("tangent block-R history is incomplete or non-finite")
        if not np.all(self.active_final):
            raise RuntimeError("one or more tangent trajectories were censored")
        accumulation = self.qr_log_increment[:, self.alignment_cycles :, :]
        window = self.physical_cycles - self.alignment_cycles
        reported = np.mean(accumulation, axis=1)
        return {
            "schema": TANGENT_SCHEMA,
            "samples": self.samples,
            "physical_cycles": self.physical_cycles,
            "alignment_cycles": self.alignment_cycles,
            "accumulation_cycles": window,
            "reported_spectrum_shape": list(reported.shape),
            "finite_fraction": float(np.mean(np.isfinite(reported))),
        }

    def save(self, path: Path | str) -> dict[str, Any]:
        diagnostics = self.validate()
        reported = np.mean(
            self.qr_log_increment[:, self.alignment_cycles :, :], axis=1
        )
        payload: dict[str, Any] = {
            "schema": np.asarray(TANGENT_SCHEMA),
            "physical_cycles": np.asarray(self.physical_cycles, dtype=np.int64),
            "alignment_cycles": np.asarray(self.alignment_cycles, dtype=np.int64),
            "qr_log_increment": self.qr_log_increment,
            "qr_null_mask": self.null_mask,
            "qr_r": self.qr_r,
            "cumulative_spectrum": self.cumulative_spectrum,
            "reported_final_window_spectrum": reported,
            "active_final": self.active_final,
        }
        for cycle, frame in self.frames.items():
            payload[f"aligned_frame_cycle_{cycle:04d}"] = frame
        for cycle, weight in self.frame_x_weight.items():
            payload[f"aligned_frame_x_weight_cycle_{cycle:04d}"] = weight
        path = Path(path)
        save_npz_atomic(path, **payload)
        return {
            **diagnostics,
            "path": str(path),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
