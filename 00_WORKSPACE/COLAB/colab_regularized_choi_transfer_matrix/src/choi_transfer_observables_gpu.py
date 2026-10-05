from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


HELPER_VERSION = "choi_transfer_observables_gpu_v6_resolvent_choi"
CANONICAL_ENTRY_POINT = "classA_U1FGTN_gpu.run_markov_circuit"
OBSERVABLE = "complex_particle_choi_transfer_gap"
CHOI_FORMULA_VERSION = "regularized_resolvent_rank_one_v2"
PARTICLE_TRANSFER_DEFINITION = "T_p=-(I+Sigma_LL)^(-1) Sigma_LR on finite sectors"
GAP_DEFINITION = "min_j abs(log(s_j(T_p))/cycle)"
GAP_EXTRACTION = "restarted_block_davidson_on_Sigma_LL_squared"
FINAL_SPECTRUM_EXTRACTION = "exact_eigh_Sigma_LL"


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


def _hermitize(sigma_ll: torch.Tensor) -> torch.Tensor:
    return 0.5 * (sigma_ll + sigma_ll.mH)


def _exponents_from_a_values(
    a_values: torch.Tensor,
    *,
    cycle: int,
    endpoint_tol: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if int(cycle) <= 0:
        raise ValueError("Particle-transfer exponents require a positive completed cycle.")
    values = torch.clamp(a_values.real, min=-1.0, max=1.0)
    particle_poles = values <= (-1.0 + float(endpoint_tol))
    particle_zeros = values >= (1.0 - float(endpoint_tol))
    finite = ~(particle_poles | particle_zeros)
    exponents = torch.empty_like(values)
    exponents[particle_poles] = float("inf")
    exponents[particle_zeros] = -float("inf")
    if bool(torch.any(finite).item()):
        finite_values = values[finite]
        exponents[finite] = (
            torch.log1p(-finite_values) - torch.log1p(finite_values)
        ) / (2.0 * float(cycle))
    return exponents, finite, particle_zeros, particle_poles


def _select_near_gap(
    a_values: torch.Tensor,
    a_vectors: torch.Tensor,
    *,
    cycle: int,
    endpoint_tol: float,
    n_eigenstates: int,
) -> dict[str, torch.Tensor | int]:
    exponents, finite, particle_zeros, particle_poles = _exponents_from_a_values(
        a_values, cycle=int(cycle), endpoint_tol=float(endpoint_tol)
    )
    finite_count = int(torch.count_nonzero(finite).item())
    if finite_count < int(n_eigenstates):
        raise FloatingPointError(
            f"Only {finite_count} finite T_p T_p^dagger eigenstates are available; "
            f"need {int(n_eigenstates)}."
        )
    score = torch.where(finite, torch.abs(exponents), torch.full_like(exponents, float("inf")))
    selected = torch.topk(score, k=int(n_eigenstates), largest=False).indices
    selected = selected[torch.argsort(score.index_select(0, selected))]
    return {
        "gap": score.index_select(0, selected)[0],
        "near_gap_exponents": exponents.index_select(0, selected),
        "near_gap_a_eigenvalues": a_values.real.index_select(0, selected),
        "eigenstates": a_vectors.index_select(1, selected),
        "selected_indices": selected,
        "finite_eigenstate_count": finite_count,
        "particle_zero_count": int(torch.count_nonzero(particle_zeros).item()),
        "particle_pole_count": int(torch.count_nonzero(particle_poles).item()),
    }


def _exact_particle_single(
    sigma_ll: torch.Tensor,
    *,
    cycle: int,
    endpoint_tol: float,
    choi_spectral_tol: float,
    n_eigenstates: int,
) -> dict[str, torch.Tensor | int]:
    a_herm = _hermitize(sigma_ll)
    a_values, a_vectors = torch.linalg.eigh(a_herm)
    raw_min = a_values[0]
    raw_max = a_values[-1]
    if bool((raw_min < -1.0 - choi_spectral_tol).item()) or bool(
        (raw_max > 1.0 + choi_spectral_tol).item()
    ):
        raise FloatingPointError(
            "Sigma_LL violates the pure-Choi spectral interval: "
            f"eigenvalue range=[{float(raw_min.item()):.6e}, {float(raw_max.item()):.6e}], "
            f"allowed tolerance={choi_spectral_tol:.6e}."
        )
    selected = _select_near_gap(
        torch.clamp(a_values, min=-1.0, max=1.0),
        a_vectors,
        cycle=int(cycle),
        endpoint_tol=float(endpoint_tol),
        n_eigenstates=int(n_eigenstates),
    )
    exponents, _, _, _ = _exponents_from_a_values(
        a_values, cycle=int(cycle), endpoint_tol=float(endpoint_tol)
    )
    spectrum_order = torch.argsort(exponents)
    spectrum = exponents.index_select(0, spectrum_order)
    inverse_order = torch.empty_like(spectrum_order)
    inverse_order[spectrum_order] = torch.arange(
        int(spectrum_order.numel()), dtype=spectrum_order.dtype, device=spectrum_order.device
    )
    selected["selected_indices"] = inverse_order.index_select(0, selected["selected_indices"])
    selected.update(
        {
            "spectrum": spectrum,
            "a_eigenvalue_min": raw_min,
            "a_eigenvalue_max": raw_max,
        }
    )
    return selected


def exact_particle_spectrum_from_choi_blocks(
    sigma_ll: torch.Tensor,
    *,
    cycle: int,
    endpoint_tol: float = 1e-12,
    choi_spectral_tol: float = 1e-10,
    n_eigenstates: int = 3,
) -> dict[str, torch.Tensor]:
    if sigma_ll.ndim != 3 or sigma_ll.shape[-2] != sigma_ll.shape[-1]:
        raise ValueError(f"Expected Sigma_LL with shape (B,N,N), got {tuple(sigma_ll.shape)}.")
    rows = [
        _exact_particle_single(
            sigma_ll[idx],
            cycle=int(cycle),
            endpoint_tol=float(endpoint_tol),
            choi_spectral_tol=float(choi_spectral_tol),
            n_eigenstates=int(n_eigenstates),
        )
        for idx in range(int(sigma_ll.shape[0]))
    ]
    tensor_keys = (
        "spectrum",
        "gap",
        "near_gap_exponents",
        "near_gap_a_eigenvalues",
        "eigenstates",
        "selected_indices",
        "a_eigenvalue_min",
        "a_eigenvalue_max",
    )
    result = {key: torch.stack([row[key] for row in rows]) for key in tensor_keys}
    for key in ("finite_eigenstate_count", "particle_zero_count", "particle_pole_count"):
        result[key] = torch.as_tensor([row[key] for row in rows], dtype=torch.long, device=sigma_ll.device)
    return result


def _orthogonalize(block: torch.Tensor, against: torch.Tensor | None = None) -> torch.Tensor:
    if against is not None and int(against.shape[1]) > 0:
        block = block - against @ (against.mH @ block)
    q, r = torch.linalg.qr(block, mode="reduced")
    keep = torch.abs(torch.diagonal(r)) > (64.0 * torch.finfo(block.real.dtype).eps)
    return q[:, keep]


def _random_block(
    nlayer: int,
    ncols: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    return torch.randn((int(nlayer), int(ncols)), dtype=dtype, device=device, generator=generator)


def _augment_block(
    block: torch.Tensor,
    target_cols: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
) -> torch.Tensor:
    q = _orthogonalize(block)
    trial = 0
    while int(q.shape[1]) < int(target_cols):
        extra = _random_block(
            int(q.shape[0]),
            int(target_cols) - int(q.shape[1]),
            dtype=dtype,
            device=device,
            seed=int(seed) + 997 * trial,
        )
        extra = _orthogonalize(extra, against=q)
        q = torch.cat((q, extra), dim=1)
        trial += 1
        if trial > 8:
            raise RuntimeError("Failed to generate a full-rank Davidson search block.")
    return q[:, : int(target_cols)]


def _resolve_signed_near_gap(
    a_herm: torch.Tensor,
    subspace: torch.Tensor,
    *,
    cycle: int,
    endpoint_tol: float,
    n_eigenstates: int,
) -> dict[str, torch.Tensor | int]:
    projected = _hermitize(subspace.mH @ a_herm @ subspace)
    a_values, coeff = torch.linalg.eigh(projected)
    vectors = subspace @ coeff
    selected = _select_near_gap(
        a_values,
        vectors,
        cycle=int(cycle),
        endpoint_tol=float(endpoint_tol),
        n_eigenstates=int(n_eigenstates),
    )
    chosen_vectors = selected["eigenstates"]
    chosen_a = selected["near_gap_a_eigenvalues"]
    residuals = torch.linalg.vector_norm(
        a_herm @ chosen_vectors - chosen_vectors * chosen_a.unsqueeze(0), dim=0
    )
    selected["near_gap_residuals"] = residuals
    return selected


def matrix_free_near_gap_single(
    sigma_ll: torch.Tensor,
    *,
    cycle: int,
    endpoint_tol: float = 1e-12,
    choi_spectral_tol: float = 1e-10,
    n_eigenstates: int = 3,
    block_size: int = 8,
    max_subspace: int = 32,
    max_iter: int = 80,
    residual_tol: float = 1e-10,
    warm_start: torch.Tensor | None = None,
    seed: int = 0,
    exact_fallback: bool = True,
) -> dict[str, torch.Tensor | int | bool]:
    if int(block_size) < int(n_eigenstates):
        raise ValueError("block_size must be at least n_eigenstates.")
    if int(max_subspace) < int(block_size) + int(n_eigenstates):
        raise ValueError("max_subspace must exceed block_size by at least n_eigenstates.")
    a_herm = _hermitize(sigma_ll)
    nlayer = int(a_herm.shape[0])
    initial = (
        warm_start
        if warm_start is not None
        else _random_block(nlayer, int(block_size), dtype=a_herm.dtype, device=a_herm.device, seed=int(seed))
    )
    basis = _augment_block(
        initial,
        int(block_size),
        dtype=a_herm.dtype,
        device=a_herm.device,
        seed=int(seed) + 103,
    )
    candidate = None
    converged = False
    iterations = 0
    for iterations in range(1, int(max_iter) + 1):
        a2_basis = a_herm @ (a_herm @ basis)
        reduced = _hermitize(basis.mH @ a2_basis)
        theta, coeff = torch.linalg.eigh(reduced)
        keep = min(int(block_size), int(coeff.shape[1]))
        candidate = basis @ coeff[:, :keep]
        residual = a_herm @ (a_herm @ candidate) - candidate * theta[:keep].unsqueeze(0)
        residual_norm = torch.linalg.vector_norm(residual[:, : int(n_eigenstates)], dim=0)
        if bool(torch.all(residual_norm <= float(residual_tol)).item()):
            signed = _resolve_signed_near_gap(
                a_herm,
                candidate,
                cycle=int(cycle),
                endpoint_tol=float(endpoint_tol),
                n_eigenstates=int(n_eigenstates),
            )
            if bool(torch.all(signed["near_gap_residuals"] <= float(residual_tol)).item()):
                converged = True
                break
        expansion = _orthogonalize(residual, against=basis)
        if int(expansion.shape[1]) == 0:
            break
        if int(basis.shape[1]) + int(expansion.shape[1]) > int(max_subspace):
            restart = torch.cat(
                (candidate, expansion[:, : max(0, int(max_subspace) - int(candidate.shape[1]))]), dim=1
            )
            basis = _augment_block(
                restart,
                min(int(max_subspace), int(restart.shape[1])),
                dtype=a_herm.dtype,
                device=a_herm.device,
                seed=int(seed) + 1009 * iterations,
            )
        else:
            basis = torch.cat((basis, expansion), dim=1)
    if converged:
        signed.update(
            {
                "solver_iterations": iterations,
                "used_exact_fallback": False,
                "a_eigenvalue_min": torch.full((), float("nan"), dtype=a_herm.real.dtype, device=a_herm.device),
                "a_eigenvalue_max": torch.full((), float("nan"), dtype=a_herm.real.dtype, device=a_herm.device),
                "particle_zero_count": -1,
                "particle_pole_count": -1,
            }
        )
        return signed
    if not bool(exact_fallback):
        raise FloatingPointError(
            f"Matrix-free near-gap solve did not converge by iteration {iterations} "
            f"at cycle {int(cycle)}."
        )
    exact = _exact_particle_single(
        a_herm,
        cycle=int(cycle),
        endpoint_tol=float(endpoint_tol),
        choi_spectral_tol=float(choi_spectral_tol),
        n_eigenstates=int(n_eigenstates),
    )
    vectors = exact["eigenstates"]
    values = exact["near_gap_a_eigenvalues"]
    exact["near_gap_residuals"] = torch.linalg.vector_norm(
        a_herm @ vectors - vectors * values.unsqueeze(0), dim=0
    )
    exact["solver_iterations"] = iterations
    exact["used_exact_fallback"] = True
    return exact


class ParticleChoiGapObserver:
    """Extract particle gaps until a trajectory's normalized Choi chart is unreliable."""

    def __init__(
        self,
        *,
        samples_expected: int,
        cycles: Iterable[int],
        final_cycle: int,
        nlayer: int,
        n_eigenstates: int = 3,
        near_gap_block_size: int = 8,
        near_gap_max_subspace: int = 32,
        near_gap_max_iter: int = 80,
        near_gap_residual_tol: float = 1e-10,
        endpoint_tol: float = 1e-12,
        choi_spectral_tol: float = 1e-10,
        exact_fallback: bool = True,
        validate_choi: bool = False,
        censor_unstable: bool = False,
        choi_entry_abs_tol: float = 1e-10,
        choi_hermiticity_tol: float = 1e-10,
        choi_involution_tol: float = 1e-8,
        active_top_layer_indices: np.ndarray | None = None,
        full_nlayer: int | None = None,
    ) -> None:
        self.samples_expected = int(samples_expected)
        self.cycles = [int(cycle) for cycle in cycles]
        self.final_cycle = int(final_cycle)
        if self.final_cycle not in self.cycles:
            raise ValueError("final_cycle must be included in cycles.")
        self.nlayer = int(nlayer)
        self.n_eigenstates = int(n_eigenstates)
        self.near_gap_block_size = int(near_gap_block_size)
        self.near_gap_max_subspace = int(near_gap_max_subspace)
        self.near_gap_max_iter = int(near_gap_max_iter)
        self.near_gap_residual_tol = float(near_gap_residual_tol)
        self.endpoint_tol = float(endpoint_tol)
        self.choi_spectral_tol = float(choi_spectral_tol)
        self.exact_fallback = bool(exact_fallback)
        self.validate_choi = bool(validate_choi)
        self.censor_unstable = bool(censor_unstable)
        self.choi_entry_abs_tol = float(choi_entry_abs_tol)
        self.choi_hermiticity_tol = float(choi_hermiticity_tol)
        self.choi_involution_tol = float(choi_involution_tol)
        self._cycle_to_index = {cycle: idx for idx, cycle in enumerate(self.cycles)}
        shape = (self.samples_expected, len(self.cycles))
        near_shape = shape + (self.n_eigenstates,)
        self.gap = np.full(shape, np.nan, dtype=np.float64)
        self.near_gap_exponents = np.full(near_shape, np.nan, dtype=np.float64)
        self.near_gap_a_eigenvalues = np.full(near_shape, np.nan, dtype=np.float64)
        self.near_gap_residuals = np.full(near_shape, np.nan, dtype=np.float64)
        self.solver_iterations = np.full(shape, -1, dtype=np.int64)
        self.used_exact_fallback = np.zeros(shape, dtype=np.bool_)
        self.endpoint_counts_evaluated = np.zeros(shape, dtype=np.bool_)
        self.particle_zero_count = np.full(shape, -1, dtype=np.int64)
        self.particle_pole_count = np.full(shape, -1, dtype=np.int64)
        self.min_abs_d = np.full(shape, np.nan, dtype=np.float64)
        self.choi_active_at_observation = np.zeros(shape, dtype=np.bool_)
        self.stable_at_cycle = np.zeros(shape, dtype=np.bool_)
        self.hermiticity_residual = np.full(shape, np.nan, dtype=np.float64)
        self.involution_residual = np.full(shape, np.nan, dtype=np.float64)
        self.max_abs_choi_entry = np.full(shape, np.nan, dtype=np.float64)
        self.first_failure_cycle = np.full((self.samples_expected,), -1, dtype=np.int64)
        self.first_failure_reason = np.full((self.samples_expected,), "", dtype="<U96")
        self.final_spectrum = np.full((self.samples_expected, self.nlayer), np.nan, dtype=np.float64)
        self.final_eigenstates_near_gap = np.full(
            (self.samples_expected, self.nlayer, self.n_eigenstates), np.nan + 1j * np.nan, dtype=np.complex128
        )
        self.final_eigenstate_exponents = np.full((self.samples_expected, self.n_eigenstates), np.nan, dtype=np.float64)
        self.final_eigenstate_indices = np.full((self.samples_expected, self.n_eigenstates), -1, dtype=np.int64)
        self.final_eigenstate_a_eigenvalues = np.full((self.samples_expected, self.n_eigenstates), np.nan, dtype=np.float64)
        self.a_eigenvalue_min_final = np.full((self.samples_expected,), np.nan, dtype=np.float64)
        self.a_eigenvalue_max_final = np.full((self.samples_expected,), np.nan, dtype=np.float64)
        self.exact_vs_iterative_gap_error_final = np.full((self.samples_expected,), np.nan, dtype=np.float64)
        self._warm_start: dict[int, torch.Tensor] = {}
        self.denominator_context: list[dict[str, Any]] = []
        self.failure_details: list[dict[str, Any]] = []
        self._seen_engine_failure_records: set[str] = set()
        self.active_top_layer_indices = (
            None if active_top_layer_indices is None else np.asarray(active_top_layer_indices, dtype=np.int64).copy()
        )
        self.full_nlayer = None if full_nlayer is None else int(full_nlayer)

    def _record_failure(self, sample_idx: int, cycle: int, reason: str, detail: dict[str, Any]) -> None:
        if self.first_failure_cycle[int(sample_idx)] < 0:
            self.first_failure_cycle[int(sample_idx)] = int(cycle)
            self.first_failure_reason[int(sample_idx)] = str(reason)
        payload = dict(detail)
        payload.update({"sample_index": int(sample_idx), "cycle_observed": int(cycle), "reason": str(reason)})
        self.failure_details.append(payload)

    def _store_selected(self, sample_idx: int, cidx: int, result: dict[str, Any]) -> None:
        self.gap[sample_idx, cidx] = float(result["gap"].item())
        self.near_gap_exponents[sample_idx, cidx] = result["near_gap_exponents"].detach().cpu().numpy()
        self.near_gap_a_eigenvalues[sample_idx, cidx] = result["near_gap_a_eigenvalues"].detach().cpu().numpy()
        self.near_gap_residuals[sample_idx, cidx] = result["near_gap_residuals"].detach().cpu().numpy()
        self.solver_iterations[sample_idx, cidx] = int(result["solver_iterations"])
        self.used_exact_fallback[sample_idx, cidx] = bool(result["used_exact_fallback"])
        self._warm_start[sample_idx] = result["eigenstates"].detach()
        if bool(result["used_exact_fallback"]):
            self.endpoint_counts_evaluated[sample_idx, cidx] = True
            self.particle_zero_count[sample_idx, cidx] = int(result["particle_zero_count"])
            self.particle_pole_count[sample_idx, cidx] = int(result["particle_pole_count"])

    def _clear_selected(self, sample_idx: int, cidx: int) -> None:
        self.gap[sample_idx, cidx] = np.nan
        self.near_gap_exponents[sample_idx, cidx] = np.nan
        self.near_gap_a_eigenvalues[sample_idx, cidx] = np.nan
        self.near_gap_residuals[sample_idx, cidx] = np.nan
        self.solver_iterations[sample_idx, cidx] = -1
        self.used_exact_fallback[sample_idx, cidx] = False
        self.endpoint_counts_evaluated[sample_idx, cidx] = False
        self.particle_zero_count[sample_idx, cidx] = -1
        self.particle_pole_count[sample_idx, cidx] = -1
        self._warm_start.pop(sample_idx, None)

    def __call__(
        self,
        *,
        cycle: int,
        sigma_ll: torch.Tensor,
        sigma_lr: torch.Tensor,
        sigma_rr: torch.Tensor,
        batch_index: int,
        batch_start: int,
        batch_count: int,
        min_abs_d: float,
        min_abs_d_context: dict[str, Any] | None,
        choi_active_mask: torch.Tensor | None = None,
        choi_failure_records: tuple[dict[str, Any], ...] = (),
        active_top_layer_indices: torch.Tensor | None = None,
        full_nlayer: int | None = None,
    ) -> dict[str, Any] | None:
        if int(cycle) not in self._cycle_to_index:
            raise ValueError(f"Observed unconfigured Choi cycle {cycle}.")
        cidx = self._cycle_to_index[int(cycle)]
        start = int(batch_start)
        stop = start + int(batch_count)
        if stop > self.samples_expected or int(sigma_ll.shape[0]) != int(batch_count):
            raise ValueError("Observed Choi batch does not match the configured sample slice.")
        if int(sigma_ll.shape[-1]) != self.nlayer:
            raise ValueError(f"Expected Choi block dimension {self.nlayer}, got {int(sigma_ll.shape[-1])}.")
        if active_top_layer_indices is not None:
            basis = active_top_layer_indices.detach().cpu().numpy().astype(np.int64, copy=False)
            if self.active_top_layer_indices is None:
                self.active_top_layer_indices = basis.copy()
            elif not np.array_equal(self.active_top_layer_indices, basis):
                raise ValueError("Active Choi basis changed during observation.")
        if full_nlayer is not None:
            if self.full_nlayer is None:
                self.full_nlayer = int(full_nlayer)
            elif self.full_nlayer != int(full_nlayer):
                raise ValueError("Full top-layer dimension changed during observation.")
        active = (
            torch.ones((int(batch_count),), dtype=torch.bool, device=sigma_ll.device)
            if choi_active_mask is None
            else choi_active_mask.to(dtype=torch.bool, device=sigma_ll.device)
        )
        self.choi_active_at_observation[start:stop, cidx] = active.detach().cpu().numpy()
        self.min_abs_d[start:stop, cidx] = float(min_abs_d)
        if min_abs_d_context is not None:
            self.denominator_context.append(
                {
                    "cycle_observed": int(cycle), "batch_index": int(batch_index),
                    "batch_start": int(batch_start), "min_abs_d": float(min_abs_d),
                    "context": dict(min_abs_d_context),
                }
            )
        for record in choi_failure_records:
            if record.get("stage") == "observer":
                continue
            key = json.dumps(record, sort_keys=True)
            if key in self._seen_engine_failure_records:
                continue
            self._seen_engine_failure_records.add(key)
            local_idx = int(record["sample_offset"])
            if record.get("stage") == "denominator_regularized":
                payload = dict(record)
                payload.update(
                    {
                        "sample_index": int(start + local_idx),
                        "cycle_observed": int(record.get("cycle", cycle)),
                        "reason": "denominator_regularized",
                    }
                )
                self.failure_details.append(payload)
                continue
            self._record_failure(start + local_idx, int(record.get("cycle", cycle)), "denominator", dict(record))

        deactivate: list[int] = []
        returned_records: list[dict[str, Any]] = []
        for local_idx in range(int(batch_count)):
            sample_idx = start + local_idx
            if not bool(active[local_idx].item()):
                if self.first_failure_cycle[sample_idx] < 0:
                    self._record_failure(sample_idx, int(cycle), "engine_censored", {"stage": "engine"})
                continue
            blocks = (sigma_ll[local_idx], sigma_lr[local_idx], sigma_rr[local_idx])
            if not all(bool(torch.isfinite(block).all().item()) for block in blocks):
                reason = "nonfinite_choi_block"
            else:
                max_entry = max(float(torch.max(torch.abs(block)).item()) for block in blocks)
                self.max_abs_choi_entry[sample_idx, cidx] = max_entry
                reason = "choi_entry_out_of_bounds" if max_entry > 1.0 + self.choi_entry_abs_tol else ""
            if not reason and self.validate_choi:
                sigma = torch.cat(
                    (torch.cat((blocks[0], blocks[1]), dim=-1), torch.cat((blocks[1].mH, blocks[2]), dim=-1)), dim=-2
                )
                identity = torch.eye(2 * self.nlayer, dtype=sigma.dtype, device=sigma.device)
                herm = float(torch.linalg.matrix_norm(sigma - sigma.mH, ord="fro").item())
                invol = float(torch.linalg.matrix_norm(sigma @ sigma - identity, ord="fro").item())
                self.hermiticity_residual[sample_idx, cidx] = herm
                self.involution_residual[sample_idx, cidx] = invol
                if herm > self.choi_hermiticity_tol:
                    reason = "choi_hermiticity_residual"
                elif invol > self.choi_involution_tol:
                    reason = "choi_involution_residual"
            if reason:
                if not self.censor_unstable:
                    raise FloatingPointError(f"Unstable Choi trajectory at cycle {cycle}: {reason}.")
                record = {"stage": "observer", "cycle": int(cycle), "sample_offset": int(local_idx), "reason": reason}
                deactivate.append(local_idx)
                returned_records.append(record)
                self._record_failure(sample_idx, int(cycle), reason, record)
                continue
            try:
                iterative = matrix_free_near_gap_single(
                    sigma_ll[local_idx], cycle=int(cycle), endpoint_tol=self.endpoint_tol,
                    choi_spectral_tol=self.choi_spectral_tol, n_eigenstates=self.n_eigenstates,
                    block_size=self.near_gap_block_size, max_subspace=self.near_gap_max_subspace,
                    max_iter=self.near_gap_max_iter, residual_tol=self.near_gap_residual_tol,
                    warm_start=self._warm_start.get(sample_idx),
                    seed=7919 * sample_idx + 104729 * int(cycle) + 17, exact_fallback=self.exact_fallback,
                )
                self._store_selected(sample_idx, cidx, iterative)
                if int(cycle) == self.final_cycle:
                    final = _exact_particle_single(
                        sigma_ll[local_idx], cycle=int(cycle), endpoint_tol=self.endpoint_tol,
                        choi_spectral_tol=self.choi_spectral_tol, n_eigenstates=self.n_eigenstates,
                    )
                    self.final_spectrum[sample_idx] = final["spectrum"].detach().cpu().numpy()
                    self.final_eigenstates_near_gap[sample_idx] = final["eigenstates"].detach().cpu().numpy()
                    self.final_eigenstate_exponents[sample_idx] = final["near_gap_exponents"].detach().cpu().numpy()
                    self.final_eigenstate_indices[sample_idx] = final["selected_indices"].detach().cpu().numpy()
                    self.final_eigenstate_a_eigenvalues[sample_idx] = final["near_gap_a_eigenvalues"].detach().cpu().numpy()
                    self.a_eigenvalue_min_final[sample_idx] = float(final["a_eigenvalue_min"].item())
                    self.a_eigenvalue_max_final[sample_idx] = float(final["a_eigenvalue_max"].item())
                    self.particle_zero_count[sample_idx, cidx] = int(final["particle_zero_count"])
                    self.particle_pole_count[sample_idx, cidx] = int(final["particle_pole_count"])
                    self.endpoint_counts_evaluated[sample_idx, cidx] = True
                    self.exact_vs_iterative_gap_error_final[sample_idx] = abs(
                        self.gap[sample_idx, cidx] - float(final["gap"].item())
                    )
                    final["near_gap_residuals"] = torch.linalg.vector_norm(
                        _hermitize(sigma_ll[local_idx]) @ final["eigenstates"]
                        - final["eigenstates"] * final["near_gap_a_eigenvalues"].unsqueeze(0), dim=0
                    )
                    final["solver_iterations"] = int(iterative["solver_iterations"])
                    final["used_exact_fallback"] = bool(iterative["used_exact_fallback"])
                    self._store_selected(sample_idx, cidx, final)
                self.stable_at_cycle[sample_idx, cidx] = True
            except (FloatingPointError, RuntimeError) as exc:
                if not self.censor_unstable:
                    raise
                self._clear_selected(sample_idx, cidx)
                record = {
                    "stage": "observer", "cycle": int(cycle), "sample_offset": int(local_idx),
                    "reason": "spectral_or_solver_failure", "message": str(exc),
                }
                deactivate.append(local_idx)
                returned_records.append(record)
                self._record_failure(sample_idx, int(cycle), "spectral_or_solver_failure", record)
        if deactivate:
            return {"deactivate_sample_offsets": deactivate, "failure_records": returned_records}
        return None

    def assert_complete(self) -> None:
        stable = self.stable_at_cycle
        if np.isnan(self.gap[stable]).any() or np.isnan(self.near_gap_exponents[stable]).any():
            raise AssertionError("Accepted particle Choi gap values contain NaNs.")
        if np.any(self.solver_iterations[stable] < 0):
            raise AssertionError("Accepted particle Choi gap values lack solver diagnostics.")
        if not self.censor_unstable and not np.all(stable):
            raise AssertionError("Particle Choi gap observer has missing values in fail-fast mode.")
        final_stable = stable[:, self._cycle_to_index[self.final_cycle]]
        if np.any(final_stable & ~self.endpoint_counts_evaluated[:, self._cycle_to_index[self.final_cycle]]):
            raise AssertionError("Stable final-cycle endpoint multiplicities were not evaluated exactly.")

    def gap_payload(self) -> dict[str, np.ndarray]:
        payload = {
            "cycles": np.asarray(self.cycles, dtype=np.int64), "gap": self.gap,
            "near_gap_exponents": self.near_gap_exponents, "near_gap_a_eigenvalues": self.near_gap_a_eigenvalues,
            "near_gap_residuals": self.near_gap_residuals, "solver_iterations": self.solver_iterations,
            "used_exact_fallback": self.used_exact_fallback, "endpoint_counts_evaluated": self.endpoint_counts_evaluated,
            "particle_zero_count": self.particle_zero_count, "particle_pole_count": self.particle_pole_count,
            "stable_at_cycle": self.stable_at_cycle,
            "stable_fraction": self.stable_at_cycle.mean(axis=0),
            "first_failure_cycle": self.first_failure_cycle, "first_failure_reason": self.first_failure_reason,
        }
        if self.active_top_layer_indices is not None:
            payload["active_top_layer_indices"] = self.active_top_layer_indices
        return payload

    def final_spectrum_payload(self) -> dict[str, np.ndarray]:
        final_idx = self._cycle_to_index[self.final_cycle]
        payload = {
            "final_cycle": np.asarray(self.final_cycle, dtype=np.int64), "final_spectrum": self.final_spectrum,
            "particle_zero_count_final": self.particle_zero_count[:, final_idx],
            "particle_pole_count_final": self.particle_pole_count[:, final_idx],
            "final_cycle_survivor": self.stable_at_cycle[:, final_idx],
        }
        if self.active_top_layer_indices is not None:
            payload["active_top_layer_indices"] = self.active_top_layer_indices
        return payload

    def final_eigenstates_payload(self) -> dict[str, np.ndarray]:
        payload = {
            "final_cycle": np.asarray(self.final_cycle, dtype=np.int64),
            "eigenstates_near_gap": self.final_eigenstates_near_gap,
            "eigenstate_exponents": self.final_eigenstate_exponents,
            "eigenstate_indices": self.final_eigenstate_indices,
            "eigenstate_a_eigenvalues": self.final_eigenstate_a_eigenvalues,
            "final_cycle_survivor": self.stable_at_cycle[:, self._cycle_to_index[self.final_cycle]],
        }
        if self.active_top_layer_indices is not None:
            payload["active_top_layer_indices"] = self.active_top_layer_indices
        return payload

    def diagnostics_payload(self) -> dict[str, np.ndarray]:
        payload = {
            "cycles": np.asarray(self.cycles, dtype=np.int64), "min_abs_d": self.min_abs_d,
            "a_eigenvalue_min_final": self.a_eigenvalue_min_final, "a_eigenvalue_max_final": self.a_eigenvalue_max_final,
            "exact_vs_iterative_gap_error_final": self.exact_vs_iterative_gap_error_final,
            "used_exact_fallback": self.used_exact_fallback, "solver_iterations": self.solver_iterations,
            "hermiticity_residual": self.hermiticity_residual, "involution_residual": self.involution_residual,
            "max_abs_choi_entry": self.max_abs_choi_entry,
            "choi_active_at_observation": self.choi_active_at_observation,
            "stable_at_cycle": self.stable_at_cycle, "stable_fraction": self.stable_at_cycle.mean(axis=0),
            "first_failure_cycle": self.first_failure_cycle, "first_failure_reason": self.first_failure_reason,
        }
        if self.active_top_layer_indices is not None:
            payload["active_top_layer_indices"] = self.active_top_layer_indices
        if self.full_nlayer is not None:
            payload["full_nlayer"] = np.asarray(self.full_nlayer, dtype=np.int64)
        return payload
