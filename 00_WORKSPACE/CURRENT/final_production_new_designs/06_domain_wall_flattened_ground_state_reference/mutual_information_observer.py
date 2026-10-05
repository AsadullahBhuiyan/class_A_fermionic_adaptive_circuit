"""Fixed-width, translation-averaged bipartite mutual information."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


OBSERVER_SCHEMA = "domain_wall_bipartite_mutual_information_observer_v1"


@dataclass
class EntropyDiagnostics:
    max_restricted_hermiticity_error: float = 0.0
    min_occupation_eigenvalue: float = np.inf
    max_occupation_eigenvalue: float = -np.inf
    eigensolve_count: int = 0

    def update(self, *, hermiticity: float, eigen_min: float, eigen_max: float) -> None:
        self.max_restricted_hermiticity_error = max(
            self.max_restricted_hermiticity_error, float(hermiticity)
        )
        self.min_occupation_eigenvalue = min(
            self.min_occupation_eigenvalue, float(eigen_min)
        )
        self.max_occupation_eigenvalue = max(
            self.max_occupation_eigenvalue, float(eigen_max)
        )
        self.eigensolve_count += 1


def strip_mode_indices(
    *,
    nx: int,
    ny: int,
    width: int,
    y0: int,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Return both-orbital mode indices for a periodic full-x strip."""

    nx = int(nx)
    ny = int(ny)
    width = int(width)
    y0 = int(y0)
    if nx <= 0 or ny <= 0:
        raise ValueError("nx and ny must be positive")
    if not 1 <= width < ny:
        raise ValueError(f"width must be in [1, ny); got width={width}, ny={ny}")
    ys = (torch.arange(width, dtype=torch.long, device=device) + y0) % ny
    xs = torch.arange(nx, dtype=torch.long, device=device)
    orbitals = torch.arange(2, dtype=torch.long, device=device)
    indices = (
        2 * xs[None, :, None]
        + 2 * nx * ys[:, None, None]
        + orbitals[None, None, :]
    )
    return indices.reshape(-1).contiguous()


def opposite_strip_indices(
    *,
    nx: int,
    ny: int,
    width: int,
    y0: int,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build disjoint strips A and B whose starts differ by ny/2."""

    nx = int(nx)
    ny = int(ny)
    width = int(width)
    if ny % 2:
        raise ValueError(f"ny must be even; got {ny}")
    if 2 * width > ny:
        raise ValueError(
            f"opposite strips overlap: 2*width={2 * width} exceeds ny={ny}"
        )
    a = strip_mode_indices(
        nx=nx, ny=ny, width=width, y0=y0, device=device
    )
    b = strip_mode_indices(
        nx=nx, ny=ny, width=width, y0=y0 + ny // 2, device=device
    )
    union = torch.cat((a, b), dim=0)
    if int(torch.unique(union).numel()) != int(union.numel()):
        raise RuntimeError("opposite strip construction produced overlapping modes")
    return a, b, union


def _restricted_covariance(
    centered_covariance: torch.Tensor, indices: torch.Tensor
) -> torch.Tensor:
    if centered_covariance.ndim != 3:
        raise ValueError(
            "centered_covariance must have shape (samples, modes, modes); "
            f"got {tuple(centered_covariance.shape)}"
        )
    if centered_covariance.shape[-1] != centered_covariance.shape[-2]:
        raise ValueError("centered_covariance must be square")
    indices = indices.to(centered_covariance.device)
    return centered_covariance.index_select(1, indices).index_select(2, indices)


def gaussian_entropy_from_centered_covariance(
    centered_covariance: torch.Tensor,
    *,
    eps: float = 1.0e-12,
    hermiticity_tolerance: float = 1.0e-9,
    occupation_tolerance: float = 1.0e-8,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute von Neumann entropy in nats from centered covariance ``G``."""

    if centered_covariance.ndim != 3:
        raise ValueError(
            "centered_covariance must have shape (samples, modes, modes)"
        )
    if centered_covariance.shape[-1] != centered_covariance.shape[-2]:
        raise ValueError("centered_covariance must be square")
    if not torch.isfinite(centered_covariance).all():
        raise FloatingPointError("restricted covariance contains nonfinite values")
    hermiticity = float(
        torch.amax(
            torch.abs(
                centered_covariance
                - centered_covariance.conj().transpose(-2, -1)
            )
        )
        .detach()
        .cpu()
    )
    if hermiticity > float(hermiticity_tolerance):
        raise FloatingPointError(
            "restricted covariance Hermiticity error exceeds tolerance: "
            f"{hermiticity:.6e} > {hermiticity_tolerance:.6e}"
        )
    modes = int(centered_covariance.shape[-1])
    eye = torch.eye(
        modes,
        dtype=centered_covariance.dtype,
        device=centered_covariance.device,
    )
    occupation = 0.5 * (centered_covariance + eye.unsqueeze(0))
    occupation = 0.5 * (occupation + occupation.conj().transpose(-2, -1))
    eigenvalues = torch.linalg.eigvalsh(occupation).real
    eigen_min = float(eigenvalues.min().detach().cpu())
    eigen_max = float(eigenvalues.max().detach().cpu())
    if eigen_min < -float(occupation_tolerance) or eigen_max > 1.0 + float(
        occupation_tolerance
    ):
        raise FloatingPointError(
            "restricted occupation eigenvalue lies outside [0,1] tolerance: "
            f"min={eigen_min:.6e}, max={eigen_max:.6e}"
        )
    probabilities = eigenvalues.clamp(float(eps), 1.0 - float(eps))
    entropy = -(
        probabilities * torch.log(probabilities)
        + (1.0 - probabilities) * torch.log(1.0 - probabilities)
    ).sum(dim=-1)
    return entropy.to(torch.float64), {
        "max_restricted_hermiticity_error": hermiticity,
        "min_occupation_eigenvalue": eigen_min,
        "max_occupation_eigenvalue": eigen_max,
    }


def mutual_information_components_for_translation(
    centered_covariance: torch.Tensor,
    *,
    nx: int,
    ny: int,
    width: int,
    y0: int,
    eps: float = 1.0e-12,
    hermiticity_tolerance: float = 1.0e-9,
    occupation_tolerance: float = 1.0e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, EntropyDiagnostics]:
    """Evaluate ``S_A``, ``S_B``, ``S_AB``, and ``I(A:B)`` for one placement."""

    a, b, union = opposite_strip_indices(
        nx=nx,
        ny=ny,
        width=width,
        y0=y0,
        device=centered_covariance.device,
    )
    diagnostics = EntropyDiagnostics()
    entropies: list[torch.Tensor] = []
    for indices in (a, b, union):
        entropy, stats = gaussian_entropy_from_centered_covariance(
            _restricted_covariance(centered_covariance, indices),
            eps=eps,
            hermiticity_tolerance=hermiticity_tolerance,
            occupation_tolerance=occupation_tolerance,
        )
        diagnostics.update(
            hermiticity=stats["max_restricted_hermiticity_error"],
            eigen_min=stats["min_occupation_eigenvalue"],
            eigen_max=stats["max_occupation_eigenvalue"],
        )
        entropies.append(entropy)
    entropy_a, entropy_b, entropy_union = entropies
    mutual_information = entropy_a + entropy_b - entropy_union
    return entropy_a, entropy_b, entropy_union, mutual_information, diagnostics


class FixedWidthMutualInformationObserver:
    """Collect endpoint mutual information for one durable trajectory shard."""

    def __init__(
        self,
        *,
        nx: int,
        ny: int,
        width: int,
        samples: int,
        expected_cycle: int,
        progress_bar: Any | None = None,
        eps: float = 1.0e-12,
        hermiticity_tolerance: float = 1.0e-9,
        occupation_tolerance: float = 1.0e-8,
        negative_mi_tolerance: float = 1.0e-8,
    ) -> None:
        self.nx = int(nx)
        self.ny = int(ny)
        self.width = int(width)
        self.samples = int(samples)
        self.expected_cycle = int(expected_cycle)
        self.progress_bar = progress_bar
        self.eps = float(eps)
        self.hermiticity_tolerance = float(hermiticity_tolerance)
        self.occupation_tolerance = float(occupation_tolerance)
        self.negative_mi_tolerance = float(negative_mi_tolerance)
        if self.nx <= 0 or self.ny <= 0 or self.samples <= 0:
            raise ValueError("nx, ny, and samples must be positive")
        if self.ny % 4 or self.width != self.ny // 4:
            raise ValueError(
                "production strips require ny divisible by four and width=ny//4"
            )
        if self.expected_cycle <= 0:
            raise ValueError("expected_cycle must be positive")
        self._observed = False
        self._payload: dict[str, np.ndarray] | None = None

    def __call__(
        self,
        *,
        cycle: int,
        G: torch.Tensor,
        batch_index: int,
        batch_start: int,
        batch_count: int,
        **_: Any,
    ) -> None:
        del batch_index
        if self._observed:
            raise RuntimeError("endpoint mutual-information observer called twice")
        if int(cycle) != self.expected_cycle:
            raise ValueError(
                f"expected cycle {self.expected_cycle}, received cycle {cycle}"
            )
        if int(batch_start) != 0 or int(batch_count) != self.samples:
            raise ValueError(
                "observer requires one complete five-trajectory task batch"
            )
        expected_modes = 2 * self.nx * self.ny
        if tuple(G.shape) != (self.samples, expected_modes, expected_modes):
            raise ValueError(
                f"unexpected final covariance shape {tuple(G.shape)}; expected "
                f"({self.samples}, {expected_modes}, {expected_modes})"
            )
        full_hermiticity = float(
            torch.amax(torch.abs(G - G.conj().transpose(-2, -1))).detach().cpu()
        )
        if full_hermiticity > self.hermiticity_tolerance:
            raise FloatingPointError(
                "full covariance Hermiticity error exceeds tolerance: "
                f"{full_hermiticity:.6e} > {self.hermiticity_tolerance:.6e}"
            )

        entropy_a_sum = torch.zeros(
            self.samples, dtype=torch.float64, device=G.device
        )
        entropy_b_sum = torch.zeros_like(entropy_a_sum)
        entropy_union_sum = torch.zeros_like(entropy_a_sum)
        mutual_information_sum = torch.zeros_like(entropy_a_sum)
        diagnostics = EntropyDiagnostics()
        unique_translations = self.ny // 2
        for y0 in range(unique_translations):
            entropy_a, entropy_b, entropy_union, mutual_information, stats = (
                mutual_information_components_for_translation(
                    G,
                    nx=self.nx,
                    ny=self.ny,
                    width=self.width,
                    y0=y0,
                    eps=self.eps,
                    hermiticity_tolerance=self.hermiticity_tolerance,
                    occupation_tolerance=self.occupation_tolerance,
                )
            )
            entropy_a_sum += entropy_a
            entropy_b_sum += entropy_b
            entropy_union_sum += entropy_union
            mutual_information_sum += mutual_information
            diagnostics.update(
                hermiticity=stats.max_restricted_hermiticity_error,
                eigen_min=stats.min_occupation_eigenvalue,
                eigen_max=stats.max_occupation_eigenvalue,
            )
            diagnostics.eigensolve_count += stats.eigensolve_count - 1
            if self.progress_bar is not None:
                self.progress_bar.update(1)

        divisor = float(unique_translations)
        # The omitted translations exchange A and B.  Their individual
        # all-y0 averages are therefore the common mean of the two half-orbits,
        # while I(A:B) and S(A union B) are already duplicated exactly.
        entropy_single_strip_avg = 0.5 * (entropy_a_sum + entropy_b_sum) / divisor
        entropy_a_avg = entropy_single_strip_avg
        entropy_b_avg = entropy_single_strip_avg
        entropy_union_avg = entropy_union_sum / divisor
        mutual_information_avg = mutual_information_sum / divisor
        arrays = {
            "entropy_a_y0avg": entropy_a_avg.detach().cpu().numpy(),
            "entropy_b_y0avg": entropy_b_avg.detach().cpu().numpy(),
            "entropy_union_y0avg": entropy_union_avg.detach().cpu().numpy(),
            "mutual_information_y0avg": mutual_information_avg.detach()
            .cpu()
            .numpy(),
        }
        for key, values in arrays.items():
            if not np.isfinite(values).all():
                raise FloatingPointError(f"{key} contains nonfinite values")
        for key in ("entropy_a_y0avg", "entropy_b_y0avg", "entropy_union_y0avg"):
            if float(np.min(arrays[key])) < -1.0e-10:
                raise FloatingPointError(f"{key} contains a negative entropy")

        negative_mask = (
            arrays["mutual_information_y0avg"] < -self.negative_mi_tolerance
        )
        self._payload = {
            **arrays,
            "endpoint_cycle": np.asarray(self.expected_cycle, dtype=np.int64),
            "width": np.asarray(self.width, dtype=np.int64),
            "nominal_y0_count": np.asarray(self.ny, dtype=np.int64),
            "unique_y0_count": np.asarray(unique_translations, dtype=np.int64),
            "full_covariance_max_hermiticity_error": np.asarray(
                full_hermiticity, dtype=np.float64
            ),
            "restricted_max_hermiticity_error": np.asarray(
                diagnostics.max_restricted_hermiticity_error, dtype=np.float64
            ),
            "restricted_occupation_eigenvalue_min": np.asarray(
                diagnostics.min_occupation_eigenvalue, dtype=np.float64
            ),
            "restricted_occupation_eigenvalue_max": np.asarray(
                diagnostics.max_occupation_eigenvalue, dtype=np.float64
            ),
            "restricted_eigensolve_count": np.asarray(
                diagnostics.eigensolve_count, dtype=np.int64
            ),
            "materially_negative_mi_count": np.asarray(
                int(np.count_nonzero(negative_mask)), dtype=np.int64
            ),
            "mutual_information_min": np.asarray(
                float(np.min(arrays["mutual_information_y0avg"])),
                dtype=np.float64,
            ),
            "entropy_identity_max_abs_residual": np.asarray(
                float(
                    np.max(
                        np.abs(
                            arrays["mutual_information_y0avg"]
                            - (
                                arrays["entropy_a_y0avg"]
                                + arrays["entropy_b_y0avg"]
                                - arrays["entropy_union_y0avg"]
                            )
                        )
                    )
                ),
                dtype=np.float64,
            ),
        }
        self._observed = True

    def payload(self) -> dict[str, np.ndarray]:
        if not self._observed or self._payload is None:
            raise RuntimeError("endpoint mutual-information observation is missing")
        return {key: np.asarray(value).copy() for key, value in self._payload.items()}
