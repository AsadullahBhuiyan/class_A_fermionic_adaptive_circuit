"""Online analytic occupied-empty tangent-cocycle products for redesigned G5."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch


SCHEMA = "g5_pure_tangent_stability_v1"


def accumulation_checkpoints(ny: int) -> list[int]:
    return sorted(set([int(ny), *[int(ny + round(f * ny)) for f in (0.25, 0.5, 0.75, 1.0)]]))


def physical_pair_rates(occupied_logs, empty_logs, elapsed: int):
    """All covariance-tangent rates, formed before any low-mode truncation."""
    occupied_logs = np.asarray(occupied_logs, dtype=np.float64)
    empty_logs = np.asarray(empty_logs, dtype=np.float64)
    if elapsed <= 0:
        raise ValueError("elapsed accumulation time must be positive")
    return (occupied_logs[:, None] + empty_logs[None, :]) / float(elapsed)


def select_slowest_pairs(pair_rates: np.ndarray, count: int = 16):
    rates = np.asarray(pair_rates, dtype=np.float64)
    finite = np.isfinite(rates)
    indices = np.argwhere(finite)
    if len(indices) < count:
        raise RuntimeError("fewer than requested finite physical tangent modes")
    values = rates[finite]
    order = np.lexsort((indices[:, 1], indices[:, 0], np.abs(values)))
    chosen = indices[order[:count]]
    boundary = abs(values[order[count]]) - abs(values[order[count-1]]) if len(order) > count else np.inf
    return chosen.astype(np.int32), rates[tuple(chosen.T)], float(boundary)


@dataclass
class PureTangentObserver:
    nx: int
    ny: int
    samples: int
    checkpoints: tuple[int, ...]
    mode_count: int = 16
    singular_tolerance: float = 1e-12

    def __post_init__(self):
        self.checkpoints = tuple(sorted(set(map(int, self.checkpoints))))
        self.per_cycle = {}
        self.per_cycle_null = {}
        self.per_cycle_active = {}
        self.per_cycle_min_probability = {}
        self.per_cycle_min_denominator = {}
        self.rows = {}

    def __call__(
        self, *, cycle, batch_start, spectra, lyapunov_frame, lyapunov_qr_r,
        lyapunov_block_sizes, lyapunov_block_core_hat,
        lyapunov_block_core_log_scale, lyapunov_finite_mask=None,
        lyapunov_null_mask=None, lyapunov_cycle_null_mask=None,
        lyapunov_active_mask=None, lyapunov_min_branch_probability=None,
        lyapunov_min_abs_born_denominator=None, **_
    ):
        if int(lyapunov_frame.shape[0]) != 1:
            raise RuntimeError("pure occupied-empty tangent extraction requires batch_size=1")
        sample, cycle = int(batch_start), int(cycle)
        diagonal = torch.diagonal(lyapunov_qr_r[0])
        self.per_cycle[(sample, cycle)] = np.asarray(
            torch.log(torch.abs(diagonal).clamp_min(torch.finfo(torch.float64).tiny)).cpu(),
            dtype=np.float64,
        )
        self.per_cycle_null[(sample, cycle)] = np.asarray(
            lyapunov_cycle_null_mask[0].detach().cpu(), dtype=np.bool_
        )
        self.per_cycle_active[(sample, cycle)] = bool(
            lyapunov_active_mask[0].detach().cpu()
        )
        self.per_cycle_min_probability[(sample, cycle)] = float(
            lyapunov_min_branch_probability[0].detach().cpu()
        )
        self.per_cycle_min_denominator[(sample, cycle)] = float(
            lyapunov_min_abs_born_denominator[0].detach().cpu()
        )
        if cycle not in self.checkpoints or cycle <= self.ny:
            return
        block = []
        start = 0
        for block_index, block_size in enumerate(lyapunov_block_sizes):
            stop = start + int(block_size)
            frame = lyapunov_frame[0, :, start:stop]
            core = lyapunov_block_core_hat[block_index][0]
            scale = float(lyapunov_block_core_log_scale[block_index][0])
            left, singular, right_h = torch.linalg.svd(core, full_matrices=False)
            logs = torch.full_like(singular, -torch.inf)
            positive = singular > self.singular_tolerance
            logs[positive] = torch.log(singular[positive]) + scale
            output = frame @ left
            output /= torch.linalg.vector_norm(output, dim=0).clamp_min(torch.finfo(torch.float64).tiny)
            residual = torch.linalg.vector_norm(core @ (core.mH @ left) - left * singular.square().to(core.dtype)[None], dim=0)
            residual /= singular[0].square().clamp_min(torch.finfo(torch.float64).tiny)
            block.append({
                "logs": logs.cpu().numpy(),
                "output": output.cpu().numpy(),
                "input": right_h.mH.resolve_conj().cpu().numpy(),
                "residual": residual.cpu().numpy(),
            })
            start = stop
        elapsed = cycle - self.ny
        # The canonical engine resets only its accumulators at cycle Ny while
        # preserving this aligned QR frame, so these are exact final-window logs.
        accumulated = [part["logs"] for part in block]
        pair = physical_pair_rates(accumulated[0], accumulated[1], elapsed)
        pair_indices, selected_rates, boundary = select_slowest_pairs(pair, self.mode_count)
        occupied_indices, empty_indices = pair_indices.T
        occ_out = block[0]["output"][:, occupied_indices]
        emp_out = block[1]["output"][:, empty_indices]
        occ_in = block[0]["input"][:, occupied_indices]
        emp_in = block[1]["input"][:, empty_indices]
        cell_density = (
            np.abs(occ_out)**2 + np.abs(emp_out)**2
        ).reshape(self.ny, self.nx, 2, self.mode_count).sum(axis=2).transpose(1, 0, 2) / 2
        x_density = cell_density.sum(axis=1).T
        physical_gram = (occ_out.conj().T @ occ_out) * (emp_out.conj().T @ emp_out)
        gram_error = np.linalg.norm(physical_gram - np.eye(self.mode_count), ord="fro")
        self.rows[(sample, cycle)] = {
            "elapsed": elapsed, "pair_indices": pair_indices, "pair_rates": selected_rates,
            "pair_selection_boundary": boundary, "all_pair_rate_min_abs": float(np.min(np.abs(pair[np.isfinite(pair)]))),
            "occupied_output_vectors": occ_out, "empty_output_vectors": emp_out,
            "occupied_input_vectors": occ_in, "empty_input_vectors": emp_in,
            "pair_cell_density": cell_density, "pair_x_density": x_density,
            "one_leg_logs_occupied": accumulated[0], "one_leg_logs_empty": accumulated[1],
            "selected_residuals": np.maximum(block[0]["residual"][occupied_indices], block[1]["residual"][empty_indices]),
            "eigenvector_gram_error": float(gram_error),
        }

    def arrays(self):
        final_checkpoints = tuple(c for c in self.checkpoints if c > self.ny)
        def stack(key): return np.asarray([[self.rows[(s,c)][key] for c in final_checkpoints] for s in range(self.samples)])
        payload = {
            "schema": np.asarray(SCHEMA),
            "alignment_stop_cycle": np.asarray(self.ny, dtype=np.int32),
            "accumulation_checkpoint_cycles": np.asarray(final_checkpoints, dtype=np.int32),
            "per_cycle_qr_log_factors": np.asarray([[self.per_cycle[(s,c)] for c in range(1,2*self.ny+1)] for s in range(self.samples)]),
            "per_cycle_null_mask": np.asarray([[self.per_cycle_null[(s,c)] for c in range(1,2*self.ny+1)] for s in range(self.samples)]),
            "per_cycle_active_mask": np.asarray([[self.per_cycle_active[(s,c)] for c in range(1,2*self.ny+1)] for s in range(self.samples)]),
            "per_cycle_min_branch_probability": np.asarray([[self.per_cycle_min_probability[(s,c)] for c in range(1,2*self.ny+1)] for s in range(self.samples)]),
            "per_cycle_min_abs_born_denominator": np.asarray([[self.per_cycle_min_denominator[(s,c)] for c in range(1,2*self.ny+1)] for s in range(self.samples)]),
        }
        for key in ("elapsed","pair_indices","pair_rates","pair_selection_boundary","all_pair_rate_min_abs","occupied_output_vectors","empty_output_vectors","occupied_input_vectors","empty_input_vectors","pair_cell_density","pair_x_density","one_leg_logs_occupied","one_leg_logs_empty","selected_residuals","eigenvector_gram_error"):
            payload[key] = stack(key)
        return payload
