"""Streaming max-mix state and trajectory-operator observables for redesigned G4."""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch


SCHEMA = "g4_maxmix_operator_cft_observables_v1"


def observation_cycles(ny: int) -> list[int]:
    """Lean union of early, logarithmic, and prefix-stability checkpoints."""
    ny = int(ny)
    early = range(0, min(16, ny // 2) + 1)
    logarithmic = []
    unused = set(range(1, 2 * ny + 1))
    for target in np.geomspace(1, 2 * ny, 16):
        value = min(
            unused,
            key=lambda candidate: (abs(np.log(candidate) - np.log(target)), candidate),
        )
        logarithmic.append(value)
        unused.remove(value)
    if len(set(logarithmic)) != 16:
        raise RuntimeError(f"could not construct 16 unique log checkpoints for Ny={ny}")
    return sorted(set(early) | set(logarithmic) | {ny, 3 * ny // 2, 2 * ny})


def spectrum_factors(occupations: np.ndarray, cap_tolerance: float = 1e-12):
    nu = np.asarray(occupations, dtype=np.float64)
    tolerance = max(float(cap_tolerance), 64 * np.finfo(np.float64).eps)
    if not np.all(np.isfinite(nu)) or np.min(nu) < -tolerance or np.max(nu) > 1 + tolerance:
        raise ValueError("natural occupations must be finite and lie in [0,1]")
    nu = np.clip(nu, 0.0, 1.0)
    dominant = (nu > 0.5).astype(np.int8)
    flip_charge = (1 - 2 * dominant).astype(np.int8)
    cap = np.zeros(nu.shape, dtype=np.int8)
    cap[nu <= tolerance] = -1
    cap[nu >= 1 - tolerance] = 1
    with np.errstate(divide="ignore", invalid="ignore"):
        signed_e = np.log1p(-nu) - np.log(nu)
    signed_e[np.isclose(nu, 0.5, atol=tolerance, rtol=0)] = 0.0
    delta = 0.5 * np.abs(signed_e)
    leading_log_lambda = np.sum(np.log(np.maximum(nu, 1 - nu)), axis=-1)
    return {
        "occupation": nu,
        "signed_one_body_generator": signed_e,
        "amplitude_cost": delta,
        "dominant_occupation": dominant,
        "flip_charge": flip_charge,
        "cap_orientation": cap,
        "leading_log_normalized_eigenvalue": leading_log_lambda,
    }


def lowest_charge_resolved_levels(
    amplitude_costs: np.ndarray,
    flip_charges: np.ndarray,
    sectors: Iterable[int] = range(-3, 4),
    levels_per_sector: int = 128,
    maximum_states: int = 2_000_000,
):
    gaps = np.asarray(amplitude_costs, dtype=np.float64).reshape(-1)
    charges = np.asarray(flip_charges, dtype=np.int8).reshape(-1)
    if gaps.shape != charges.shape:
        raise ValueError("cost and charge arrays differ")
    order = np.argsort(gaps[np.isfinite(gaps)], kind="stable")
    charges = charges[np.isfinite(gaps)][order]
    gaps = gaps[np.isfinite(gaps)][order]
    sectors = np.asarray(tuple(sectors), dtype=np.int8)
    result = np.full((len(sectors), int(levels_per_sector)), np.inf)
    counts = np.zeros(len(sectors), dtype=np.int32)
    lookup = {int(q): i for i, q in enumerate(sectors)}
    heap = [(0.0, (), 0)]
    popped = 0
    while heap and np.any(counts < levels_per_sector) and popped < maximum_states:
        energy, subset, charge = heapq.heappop(heap)
        popped += 1
        sector = lookup.get(int(charge))
        if sector is not None and counts[sector] < levels_per_sector:
            result[sector, counts[sector]] = energy
            counts[sector] += 1
        if not len(gaps):
            continue
        if not subset:
            heapq.heappush(heap, (float(gaps[0]), (0,), int(charges[0])))
            continue
        nxt = subset[-1] + 1
        if nxt < len(gaps):
            heapq.heappush(heap, (energy + gaps[nxt], subset + (nxt,), charge + int(charges[nxt])))
            heapq.heappush(heap, (energy - gaps[subset[-1]] + gaps[nxt], subset[:-1] + (nxt,), charge - int(charges[subset[-1]]) + int(charges[nxt])))
    return result, counts, popped


class CycleProbabilityAccumulator:
    """Retain cycle aggregates only; outcomes and controller words are discarded."""

    def __init__(self, samples: int, cycles: int):
        self.log_probability_per_cycle = np.zeros((samples, cycles), dtype=np.float64)
        self.event_count_per_cycle = np.zeros((samples, cycles), dtype=np.int32)

    def __call__(self, *, cycle, sample_indices, conditional_log_probability, **_):
        sample = np.asarray(sample_indices.detach().cpu(), dtype=np.int64)
        logs = np.asarray(conditional_log_probability.detach().cpu(), dtype=np.float64)
        self.log_probability_per_cycle[sample, int(cycle) - 1] += logs.sum(axis=1)
        self.event_count_per_cycle[sample, int(cycle) - 1] += logs.shape[1]

    def arrays(self):
        if np.any(self.event_count_per_cycle <= 0):
            raise RuntimeError("missing branch-probability events")
        return {
            "log_probability_per_cycle": self.log_probability_per_cycle,
            "cumulative_log_probability": np.cumsum(self.log_probability_per_cycle, axis=1),
            "events_per_cycle": self.event_count_per_cycle,
        }


@dataclass
class G4Observer:
    nx: int
    ny: int
    samples: int
    cycles: tuple[int, ...]
    strip_cycles: tuple[int, ...]
    soft_modes: int = 16
    levels_per_sector: int = 128

    def __post_init__(self):
        self.cycles = tuple(sorted(set(map(int, self.cycles))))
        self.strip_cycles = tuple(sorted(set(map(int, self.strip_cycles))))
        self.nlayer = 2 * self.nx * self.ny
        self.soft_modes = min(int(self.soft_modes), self.nlayer)
        self.rows = {}
        self.previous = {}
        self.convergence = np.full((self.samples, 2 * self.ny), np.nan)
        self.impurity = np.full((self.samples, 2 * self.ny + 1), np.nan)

    @staticmethod
    def _strip_curves(C: torch.Tensor, nx: int, ny: int):
        # Periodic strips, averaged over every translated y origin.
        output = {k: np.zeros((ny // 2 + 1,), dtype=np.float64) for k in ("entropy", "charge_mean", "charge_k2", "charge_k3", "charge_k4")}
        for ay in range(1, ny // 2 + 1):
            totals = {k: 0.0 for k in output}
            for y0 in range(ny):
                ys = [(y0 + offset) % ny for offset in range(ay)]
                idx = [2 * (x + nx * y) + orbital for y in ys for x in range(nx) for orbital in (0, 1)]
                index = torch.as_tensor(idx, device=C.device)
                nu = torch.linalg.eigvalsh(C.index_select(0, index).index_select(1, index)).real.clamp(0, 1)
                clipped = nu.clamp(1e-12, 1 - 1e-12)
                q = nu * (1 - nu)
                totals["entropy"] += float(torch.sum(-clipped * torch.log(clipped) - (1-clipped)*torch.log1p(-clipped)))
                totals["charge_mean"] += float(torch.sum(nu))
                totals["charge_k2"] += float(torch.sum(q))
                totals["charge_k3"] += float(torch.sum(q * (1 - 2*nu)))
                totals["charge_k4"] += float(torch.sum(q * (1 - 6*q)))
            for key in output:
                output[key][ay] = totals[key] / ny
        return output

    def __call__(self, *, cycle, G, batch_start, batch_count, **_):
        start, stop, cycle = int(batch_start), int(batch_start + batch_count), int(cycle)
        work = G.detach()
        if cycle > 0:
            prior = self.previous.get(start)
            if prior is None:
                raise RuntimeError("non-contiguous covariance observations")
            self.convergence[start:stop, cycle - 1] = (torch.linalg.vector_norm((work-prior).reshape(batch_count, -1), dim=1) / self.nlayer).cpu().numpy()
        self.previous[start] = work.clone() if cycle < 2*self.ny else None
        eye = torch.eye(self.nlayer, dtype=work.dtype, device=work.device)
        C = 0.5 * (work + eye[None])
        C = 0.5 * (C + C.mH)
        self.impurity[start:stop, cycle] = (
            torch.diagonal(C, dim1=-2, dim2=-1).real.sum(1)
            - C.abs().square().sum(dim=(-2, -1))
        ).cpu().numpy()
        if cycle not in self.cycles:
            return
        for local in range(batch_count):
            sample = start + local
            c = C[local]
            nu, vectors = torch.linalg.eigh(c)
            nu = nu.real.clamp(0, 1)
            clipped = nu.clamp(1e-12, 1-1e-12)
            h = -clipped*torch.log(clipped) - (1-clipped)*torch.log1p(-clipped)
            q = nu*(1-nu)
            weights = vectors.abs().square()
            entropy_contour = (weights @ h).reshape(self.ny, self.nx, 2).sum(2).T
            charge_contour = (weights @ q).reshape(self.ny, self.nx, 2).sum(2).T
            density = torch.diagonal(c).real.reshape(self.ny, self.nx, 2).sum(2).T
            cell_var = torch.empty((self.nx, self.ny), dtype=torch.float64, device=c.device)
            for x in range(self.nx):
                for y in range(self.ny):
                    idx = torch.as_tensor([2*(x+self.nx*y), 2*(x+self.nx*y)+1], device=c.device)
                    ca = c.index_select(0, idx).index_select(1, idx)
                    cell_var[x, y] = torch.trace(ca - ca@ca).real
            factors = spectrum_factors(nu.cpu().numpy())
            levels, counts, popped = lowest_charge_resolved_levels(factors["amplitude_cost"], factors["flip_charge"], levels_per_sector=self.levels_per_sector)
            soft_order = np.argsort(factors["amplitude_cost"], kind="stable")[:self.soft_modes]
            soft_vectors = vectors[:, torch.as_tensor(soft_order, device=vectors.device)]
            soft_values = nu[torch.as_tensor(soft_order, device=nu.device)]
            spectral_norm = nu.abs().max().clamp_min(torch.finfo(torch.float64).tiny)
            soft_residuals = torch.linalg.vector_norm(
                c @ soft_vectors - soft_vectors * soft_values.to(c.dtype)[None], dim=0
            ) / spectral_norm
            soft_gram_error = torch.linalg.matrix_norm(
                soft_vectors.mH @ soft_vectors
                - torch.eye(self.soft_modes, dtype=c.dtype, device=c.device), ord="fro"
            )
            neutral = levels[3]
            neutral_gap = neutral[1] if len(neutral) > 1 else np.inf
            self.rows[(sample, cycle)] = {
                **factors,
                "total_entropy": float(h.sum()), "total_charge_variance": float(q.sum()),
                "density": density.cpu().numpy(), "local_cell_charge_variance": cell_var.cpu().numpy(),
                "entropy_contour": entropy_contour.cpu().numpy(), "charge_uncertainty_contour": charge_contour.cpu().numpy(),
                "soft_indices": soft_order.astype(np.int32), "soft_eigenvectors": soft_vectors.cpu().numpy(),
                "soft_eigenpair_residuals": soft_residuals.cpu().numpy(),
                "soft_eigenvector_gram_error": float(soft_gram_error),
                "radial_gap": float(np.min(factors["amplitude_cost"])), "neutral_gap": float(neutral_gap),
                "charge_sector_levels": levels, "charge_sector_counts": counts, "heap_states_popped": popped,
                "strip": self._strip_curves(c, self.nx, self.ny) if cycle in self.strip_cycles else None,
            }

    def arrays(self, cumulative_log_probability):
        shape = (self.samples, len(self.cycles)); rows = [self.rows[(s,c)] for s in range(self.samples) for c in self.cycles]
        if len(rows) != self.samples * len(self.cycles): raise RuntimeError("missing checkpoint")
        def stack(key): return np.asarray([[self.rows[(s,c)][key] for c in self.cycles] for s in range(self.samples)])
        payload = {"schema": np.asarray(SCHEMA), "checkpoint_cycles": np.asarray(self.cycles, dtype=np.int32), "convergence": self.convergence, "impurity": self.impurity}
        for key in ("occupation","signed_one_body_generator","amplitude_cost","dominant_occupation","flip_charge","cap_orientation","total_entropy","total_charge_variance","density","local_cell_charge_variance","entropy_contour","charge_uncertainty_contour","soft_indices","soft_eigenvectors","soft_eigenpair_residuals","soft_eigenvector_gram_error","radial_gap","neutral_gap","charge_sector_levels","charge_sector_counts","heap_states_popped","leading_log_normalized_eigenvalue"):
            payload[key] = stack(key)
        for key in ("entropy","charge_mean","charge_k2","charge_k3","charge_k4"):
            payload[f"strip_{key}"] = np.asarray([[self.rows[(s,c)]["strip"][key] for c in self.strip_cycles] for s in range(self.samples)])
        logp = np.column_stack((np.zeros(self.samples), cumulative_log_probability))[:, np.asarray(self.cycles)]
        payload["checkpoint_log_probability"] = logp
        payload["log_Z"] = self.nlayer*np.log(2.0) + logp
        payload["leading_log_sigma"] = 0.5*(payload["log_Z"] + payload["leading_log_normalized_eigenvalue"])
        np.testing.assert_allclose(payload["entropy_contour"].sum(axis=(-2,-1)), payload["total_entropy"], rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(payload["charge_uncertainty_contour"].sum(axis=(-2,-1)), payload["total_charge_variance"], rtol=1e-9, atol=1e-9)
        return payload
