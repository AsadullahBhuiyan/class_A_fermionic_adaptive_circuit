from __future__ import annotations

from pathlib import Path
from typing import Any
import hashlib

import numpy as np

from production_runtime import save_npz_atomic, sha256_file


NOISE_SCHEMA = "BPJ_iid_uniform_onsite_U1_digest_record_v2"


class OnsitePhaseNoiseWriter:
    def __init__(self, *, samples: int, cycles: int, nlayer: int, sigma: float) -> None:
        self.samples = int(samples)
        self.cycles = int(cycles)
        self.nlayer = int(nlayer)
        self.sigma = float(sigma)
        shape = (self.samples, self.cycles)
        self.theta_mean = np.full(shape, np.nan, dtype=np.float64)
        self.theta_std = np.full(shape, np.nan, dtype=np.float64)
        self.theta_min = np.full(shape, np.nan, dtype=np.float64)
        self.theta_max = np.full(shape, np.nan, dtype=np.float64)
        self.theta_sha256 = np.zeros(shape + (32,), dtype=np.uint8)
        self.seen = np.zeros(shape, dtype=np.bool_)
        self._theta_device: Any | None = None
        self._seen_device: Any | None = None

    def _materialize_device_record(self) -> None:
        if self._theta_device is None:
            return
        values = self._theta_device.detach().cpu().numpy()
        seen = self._seen_device.detach().cpu().numpy()
        self.theta_mean = np.mean(values, axis=2)
        self.theta_std = np.std(values, axis=2)
        self.theta_min = np.min(values, axis=2)
        self.theta_max = np.max(values, axis=2)
        for sample in range(self.samples):
            for cycle in range(self.cycles):
                digest = hashlib.sha256(
                    np.ascontiguousarray(values[sample, cycle]).view(np.uint8)
                ).digest()
                self.theta_sha256[sample, cycle] = np.frombuffer(
                    digest, dtype=np.uint8
                )
        self.seen = seen
        self._theta_device = None
        self._seen_device = None

    def __call__(
        self,
        *,
        cycle: int,
        theta: Any,
        sigma: float,
        sample_indices: Any,
        **_: Any,
    ) -> None:
        if not np.isclose(float(sigma), self.sigma, rtol=0.0, atol=0.0):
            raise ValueError("noise observer sigma changed inside one immutable run")
        if hasattr(theta, "detach"):
            import torch

            device = theta.device
            if self._theta_device is None:
                self._theta_device = torch.empty(
                    (self.samples, self.cycles, self.nlayer),
                    dtype=torch.float64,
                    device=device,
                )
                self._seen_device = torch.zeros(
                    (self.samples, self.cycles),
                    dtype=torch.bool,
                    device=device,
                )
            indices = sample_indices.to(dtype=torch.long, device=device)
            cycle_index = int(cycle) - 1
            self._theta_device[indices, cycle_index] = theta.to(torch.float64)
            self._seen_device[indices, cycle_index] = True
            return
        indices = np.asarray(sample_indices, dtype=np.int64)
        values = np.asarray(theta, dtype=np.float64)
        cycle_index = int(cycle) - 1
        if np.any(self.seen[indices, cycle_index]):
            raise RuntimeError("duplicate onsite phase-noise observation")
        self.theta_mean[indices, cycle_index] = np.mean(values, axis=1)
        self.theta_std[indices, cycle_index] = np.std(values, axis=1)
        self.theta_min[indices, cycle_index] = np.min(values, axis=1)
        self.theta_max[indices, cycle_index] = np.max(values, axis=1)
        for row, sample in enumerate(indices):
            digest = hashlib.sha256(np.ascontiguousarray(values[row]).view(np.uint8)).digest()
            self.theta_sha256[sample, cycle_index] = np.frombuffer(digest, dtype=np.uint8)
        self.seen[indices, cycle_index] = True

    def save(self, path: Path | str) -> dict[str, Any]:
        self._materialize_device_record()
        if not self.seen.all() or not all(
            np.isfinite(value).all()
            for value in (self.theta_mean, self.theta_std, self.theta_min, self.theta_max)
        ):
            raise FloatingPointError("onsite phase-noise record is incomplete")
        tolerance = 8.0 * np.finfo(np.float64).eps
        if np.any(self.theta_min < -tolerance) or np.any(self.theta_max >= self.sigma + tolerance):
            if self.sigma != 0.0:
                raise ValueError("onsite phase draws fall outside [0,sigma)")
        path = Path(path)
        save_npz_atomic(
            path,
            schema=np.asarray(NOISE_SCHEMA),
            sigma=np.asarray(self.sigma, dtype=np.float64),
            theta_mean=self.theta_mean,
            theta_std=self.theta_std,
            theta_min=self.theta_min,
            theta_max=self.theta_max,
            theta_sha256=self.theta_sha256,
            regeneration_contract=np.asarray("pre_run_rng_state+engine_hash+immutable_config"),
        )
        return {
            "schema": NOISE_SCHEMA,
            "sigma": self.sigma,
            "path": str(path),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
