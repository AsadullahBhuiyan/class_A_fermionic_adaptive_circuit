"""Compact every-cycle entropy and occupation-derived Lyapunov-gap observer."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


OBSERVER_SCHEMA = "postselected_maxmix_entropy_contour_gap_observer_v2"
SPECTRAL_TOLERANCE = 1.0e-9
HERMITICITY_TOLERANCE = 1.0e-9


class PostselectedObserver:
    ARRAY_NAMES = (
        "total_entropy_nats",
        "entropy_contour",
        "entropy_contour_x",
        "entropy_closure_error",
        "active_charge",
        "centered_min_abs",
        "raw_logit_gap",
        "lyapunov_gap",
        "hermiticity_residual",
        "spectral_bound_violation",
        "seen",
    )

    def __init__(self, *, cycles: int, active_indices: torch.Tensor, nx: int, ny: int) -> None:
        self.cycles = int(cycles)
        self.nx, self.ny = int(nx), int(ny)
        self.active_indices = active_indices.to(dtype=torch.long)
        indices = self.active_indices.detach().cpu().numpy()
        if (indices.ndim != 1 or not len(indices) or len(np.unique(indices)) != len(indices)
                or indices.min() < 0 or indices.max() >= 2 * self.nx * self.ny):
            raise ValueError("invalid physical active-basis indices")
        length = self.cycles + 1
        self.total_entropy_nats = np.full(length, np.nan, dtype=np.float64)
        self.entropy_contour = np.full((length, self.nx, self.ny), np.nan)
        self.entropy_contour_x = np.full((length, self.nx), np.nan)
        self.entropy_closure_error = np.full(length, np.nan)
        self.active_charge = np.full(length, np.nan, dtype=np.float64)
        self.centered_min_abs = np.full(length, np.nan, dtype=np.float64)
        self.raw_logit_gap = np.full(length, np.nan, dtype=np.float64)
        self.lyapunov_gap = np.full(length, np.nan, dtype=np.float64)
        self.hermiticity_residual = np.full(length, np.nan, dtype=np.float64)
        self.spectral_bound_violation = np.full(length, np.nan, dtype=np.float64)
        self.seen = np.zeros(length, dtype=np.bool_)

    def _active_matrix(self, G: torch.Tensor) -> torch.Tensor:
        if G.ndim != 3 or int(G.shape[0]) != 1 or G.shape[-1] != G.shape[-2]:
            raise ValueError(f"expected one batched square covariance, got {tuple(G.shape)}")
        idx = self.active_indices.to(device=G.device)
        return G.index_select(-2, idx).index_select(-1, idx)[0]

    def observe(self, *, cycle: int, G: torch.Tensor) -> None:
        cycle = int(cycle)
        if not 0 <= cycle <= self.cycles:
            raise ValueError(f"cycle {cycle} lies outside 0..{self.cycles}")
        if G.dtype != torch.complex128:
            raise ValueError(f"expected complex128 covariance, got {G.dtype}")
        with torch.inference_mode():
            matrix = self._active_matrix(G)
            hermiticity = torch.max(torch.abs(matrix - matrix.mH)).real
            hermiticity_value = float(hermiticity.item())
            if not np.isfinite(hermiticity_value) or hermiticity_value > HERMITICITY_TOLERANCE:
                raise FloatingPointError(
                    f"Hermiticity residual {hermiticity_value:.3e} at cycle {cycle}"
                )
            centered, vectors = torch.linalg.eigh(0.5 * (matrix + matrix.mH))
            if not bool(torch.isfinite(centered).all().item()):
                raise FloatingPointError("non-finite occupation spectrum")
            minimum = float(centered.min().item())
            maximum = float(centered.max().item())
            violation = max(0.0, -1.0 - minimum, maximum - 1.0)
            if violation > SPECTRAL_TOLERANCE:
                raise FloatingPointError(
                    f"centered spectrum left [-1,1] by {violation:.3e} at cycle {cycle}"
                )
            centered = torch.clamp(centered, -1.0, 1.0)
            occupations = 0.5 * (1.0 + centered)
            interior = (occupations > 0.0) & (occupations < 1.0)
            entropy_terms = torch.zeros_like(occupations)
            nu = occupations[interior]
            entropy_terms[interior] = -(nu * torch.log(nu) + (1.0 - nu) * torch.log1p(-nu))
            # diag[h(C)] in the active physical basis; omitted exterior modes
            # are deterministic product modes and have exactly zero entropy.
            orbital = torch.abs(vectors).square() @ entropy_terms
            physical = torch.zeros(2 * self.nx * self.ny, dtype=torch.float64, device=G.device)
            physical[self.active_indices.to(device=G.device)] = orbital
            # Canonical index: mu + 2*x + 2*Nx*y.
            contour = physical.reshape(self.ny, self.nx, 2).sum(dim=-1).T
            self.entropy_contour[cycle] = contour.cpu().numpy()
            self.entropy_contour_x[cycle] = contour.sum(dim=-1).cpu().numpy()
            self.entropy_closure_error[cycle] = abs(float((contour.sum() - entropy_terms.sum()).item()))
            finite = torch.abs(centered) < 1.0
            if bool(torch.any(finite).item()):
                half_logit_gap = float(torch.min(torch.abs(torch.atanh(centered[finite]))).item())
            else:
                half_logit_gap = float("inf")

            self.total_entropy_nats[cycle] = float(entropy_terms.sum().item())
            self.active_charge[cycle] = float(occupations.sum().item())
            self.centered_min_abs[cycle] = float(torch.min(torch.abs(centered)).item())
            self.raw_logit_gap[cycle] = 2.0 * half_logit_gap
            self.lyapunov_gap[cycle] = (
                np.nan if cycle == 0 else half_logit_gap / float(cycle)
            )
            self.hermiticity_residual[cycle] = hermiticity_value
            self.spectral_bound_violation[cycle] = violation
            self.seen[cycle] = True

    def checkpoint_payload(self, completed_cycle: int) -> dict[str, np.ndarray]:
        stop = int(completed_cycle) + 1
        return {
            name: np.array(getattr(self, name)[:stop], copy=True)
            for name in self.ARRAY_NAMES
        }

    def restore(self, payload: dict[str, np.ndarray], *, completed_cycle: int) -> None:
        stop = int(completed_cycle) + 1
        for name in self.ARRAY_NAMES:
            source = np.asarray(payload[name])
            expected = getattr(self, name)[:stop].shape
            if source.shape != expected:
                raise ValueError(f"checkpoint {name} shape {source.shape}, expected {expected}")
            getattr(self, name)[:stop] = source
        if not np.all(self.seen[:stop]):
            raise ValueError("checkpoint observer history is not a complete prefix")

    def validate(self) -> None:
        if not np.all(self.seen):
            raise RuntimeError("observer did not see every cycle")
        if not np.allclose(self.entropy_contour.sum(axis=(1, 2)), self.total_entropy_nats,
                           rtol=1e-10, atol=1e-10):
            raise RuntimeError("entropy contour does not sum to total entropy")
        if not np.allclose(self.entropy_contour.sum(axis=2), self.entropy_contour_x,
                           rtol=1e-10, atol=1e-10):
            raise RuntimeError("x contour does not match the cell contour")
        for name in self.ARRAY_NAMES:
            values = getattr(self, name)
            if name == "lyapunov_gap":
                if not np.isnan(values[0]) or not np.all(np.isfinite(values[1:])):
                    raise RuntimeError("invalid Lyapunov-gap history")
            elif name == "seen":
                continue
            elif not np.all(np.isfinite(values)):
                raise RuntimeError(f"non-finite observer array {name}")

    def result_payload(self) -> dict[str, Any]:
        self.validate()
        return {
            "observer_schema": np.asarray(OBSERVER_SCHEMA),
            "cycles": np.arange(self.cycles + 1, dtype=np.int64),
            **{
                name: np.array(getattr(self, name), copy=True)
                for name in self.ARRAY_NAMES
                if name != "seen"
            },
            "gap_formula": np.asarray("Delta_lambda(t)=min_j|atanh(a_j)|/t"),
            "centered_covariance_convention": np.asarray("G=2C-I"),
            "entropy_contour_convention": np.asarray("s(x,y)=sum_mu,j |U_(mu,x,y),j|^2 h(nu_j); nats; axes=(cycle,x,y)"),
        }
