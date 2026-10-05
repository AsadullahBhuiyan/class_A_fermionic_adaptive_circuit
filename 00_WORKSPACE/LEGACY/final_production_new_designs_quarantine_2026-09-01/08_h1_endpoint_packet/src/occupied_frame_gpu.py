"""Torch-native padded occupied frames for pure Gaussian trajectory batches."""

from __future__ import annotations

from dataclasses import dataclass

import torch


GPU_FRAME_ALGORITHM_VERSION = "padded_append_householder_qr_v2"


@dataclass(frozen=True)
class BatchedFrameOperationResult:
    probability: torch.Tensor
    selected: torch.Tensor


class BatchedOccupiedFrameState:
    """A batch of orthonormal occupied frames with trajectory-dependent ranks.

    Active columns occupy the prefix ``frame[b, :, :ranks[b]]``.  Unused padded
    columns are zero and have no physical meaning.
    """

    def __init__(
        self,
        frame: torch.Tensor,
        ranks: torch.Tensor,
        *,
        physical_dimension: int,
        capacity_chunk: int = 8,
        zero_tolerance: float | None = None,
    ) -> None:
        if frame.ndim != 3:
            raise ValueError("frame must have shape (batch, dimension, capacity).")
        self.frame = frame
        self.ranks = torch.as_tensor(
            ranks, dtype=torch.long, device=frame.device
        ).reshape(-1)
        self.physical_dimension = int(physical_dimension)
        if frame.shape[0] != self.ranks.numel() or frame.shape[1] != self.physical_dimension:
            raise ValueError("frame/rank dimensions are inconsistent.")
        if torch.any(self.ranks < 0) or torch.any(self.ranks > frame.shape[2]):
            raise ValueError("ranks must lie inside the padded frame capacity.")
        self.capacity_chunk = max(1, int(capacity_chunk))
        if zero_tolerance is None:
            zero_tolerance = 1e-12 if frame.dtype == torch.complex64 else 1e-14
        self.zero_tolerance = float(zero_tolerance)
        self.min_ranks = self.ranks.clone()
        self.max_ranks = self.ranks.clone()
        self.log_weight = torch.zeros(
            frame.shape[0], dtype=self.real_dtype, device=frame.device
        )
        self.materialization_count = 0
        self.materialization_reasons: list[str] = []
        self._zero_padding()
        if not torch.isfinite(self.frame).all():
            raise FloatingPointError("Occupied frame contains non-finite entries.")
        residual = self.gram_residual()
        hard = 2e-5 if frame.dtype == torch.complex64 else 1e-10
        if torch.any(residual > hard):
            raise ValueError("Active occupied-frame columns must be orthonormal.")

    @property
    def device(self):
        return self.frame.device

    @property
    def dtype(self):
        return self.frame.dtype

    @property
    def real_dtype(self):
        return torch.float32 if self.frame.dtype == torch.complex64 else torch.float64

    @property
    def batch_size(self) -> int:
        return int(self.frame.shape[0])

    @property
    def capacity(self) -> int:
        return int(self.frame.shape[2])

    @classmethod
    def random_pure(
        cls,
        batch_size: int,
        dimension: int,
        rank: int,
        *,
        device,
        dtype,
        generator=None,
        capacity_chunk: int = 8,
    ) -> "BatchedOccupiedFrameState":
        batch_size, dimension, rank = int(batch_size), int(dimension), int(rank)
        if not (0 <= rank <= dimension):
            raise ValueError("rank must lie in 0..dimension.")
        capacity = min(
            dimension,
            max(rank, ((rank + capacity_chunk - 1) // capacity_chunk) * capacity_chunk),
        )
        frame = torch.zeros(
            (batch_size, dimension, capacity), dtype=dtype, device=device
        )
        if rank:
            real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
            raw = torch.complex(
                torch.randn(
                    (batch_size, dimension, rank),
                    dtype=real_dtype,
                    device=device,
                    generator=generator,
                ),
                torch.randn(
                    (batch_size, dimension, rank),
                    dtype=real_dtype,
                    device=device,
                    generator=generator,
                ),
            )
            q, triangular = torch.linalg.qr(raw, mode="reduced")
            diagonal = torch.diagonal(triangular, dim1=-2, dim2=-1)
            phase = torch.where(
                diagonal.abs() > 0,
                diagonal / diagonal.abs().clamp_min(torch.finfo(real_dtype).tiny),
                torch.ones_like(diagonal),
            )
            frame[:, :, :rank] = q * phase.conj().unsqueeze(1)
        ranks = torch.full((batch_size,), rank, dtype=torch.long, device=device)
        return cls(
            frame,
            ranks,
            physical_dimension=dimension,
            capacity_chunk=capacity_chunk,
        )

    @classmethod
    def from_centered_covariance(
        cls,
        centered_covariance: torch.Tensor,
        *,
        purity_tolerance: float = 1e-9,
        capacity_chunk: int = 8,
    ) -> "BatchedOccupiedFrameState":
        centered = torch.as_tensor(centered_covariance)
        if centered.ndim == 2:
            centered = centered.unsqueeze(0)
        if centered.ndim != 3 or centered.shape[-1] != centered.shape[-2]:
            raise ValueError("centered_covariance must have shape (batch,N,N).")
        centered = 0.5 * (centered + centered.mH)
        dimension = int(centered.shape[-1])
        identity = torch.eye(dimension, dtype=centered.dtype, device=centered.device)
        correlation = 0.5 * (centered + identity)
        occupations, eigenvectors = torch.linalg.eigh(correlation)
        defects = torch.minimum(occupations.abs(), (1.0 - occupations).abs()).amax(dim=1)
        if torch.any(defects > float(purity_tolerance)):
            raise ValueError("physical_frame requires a pure initial covariance.")
        occupied = occupations > 0.5
        ranks = occupied.sum(dim=1).to(torch.long)
        max_rank = int(ranks.max().item()) if ranks.numel() else 0
        capacity = min(
            dimension,
            max(
                max_rank,
                ((max_rank + capacity_chunk - 1) // capacity_chunk) * capacity_chunk,
            ),
        )
        frame = torch.zeros(
            (centered.shape[0], dimension, capacity),
            dtype=centered.dtype,
            device=centered.device,
        )
        order = torch.argsort(occupations, dim=1, descending=True)
        gathered = torch.gather(
            eigenvectors,
            2,
            order.unsqueeze(1).expand(-1, dimension, -1),
        )
        if capacity:
            frame = gathered[:, :, :capacity] * (
                torch.arange(capacity, device=centered.device).unsqueeze(0)
                < ranks.unsqueeze(1)
            ).unsqueeze(1)
        return cls(
            frame,
            ranks,
            physical_dimension=dimension,
            capacity_chunk=capacity_chunk,
        )

    @classmethod
    def from_frame(
        cls,
        frame,
        *,
        ranks=None,
        device=None,
        dtype=None,
        capacity_chunk: int = 8,
    ) -> "BatchedOccupiedFrameState":
        tensor = torch.as_tensor(frame, device=device, dtype=dtype)
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 3:
            raise ValueError("frame_init must have shape (N,k) or (batch,N,Rcap).")
        if ranks is None:
            ranks = torch.full(
                (tensor.shape[0],), tensor.shape[2], dtype=torch.long, device=tensor.device
            )
        return cls(
            tensor.clone(),
            ranks,
            physical_dimension=tensor.shape[1],
            capacity_chunk=capacity_chunk,
        )

    def _active_mask(self, capacity: int | None = None) -> torch.Tensor:
        capacity = self.capacity if capacity is None else int(capacity)
        columns = torch.arange(capacity, dtype=torch.long, device=self.device)
        return columns.unsqueeze(0) < self.ranks.unsqueeze(1)

    def _zero_padding(self) -> None:
        if self.capacity:
            self.frame = self.frame * self._active_mask().unsqueeze(1)

    def _ensure_capacity(self, required: int) -> None:
        required = int(required)
        if required <= self.capacity:
            return
        if required > self.physical_dimension:
            raise FloatingPointError("Gain attempted beyond complete occupancy.")
        capacity = min(
            self.physical_dimension,
            ((required + self.capacity_chunk - 1) // self.capacity_chunk)
            * self.capacity_chunk,
        )
        expanded = torch.zeros(
            (self.batch_size, self.physical_dimension, capacity),
            dtype=self.dtype,
            device=self.device,
        )
        expanded[:, :, : self.capacity] = self.frame
        self.frame = expanded

    def clone(self) -> "BatchedOccupiedFrameState":
        copied = BatchedOccupiedFrameState(
            self.frame.clone(),
            self.ranks.clone(),
            physical_dimension=self.physical_dimension,
            capacity_chunk=self.capacity_chunk,
            zero_tolerance=self.zero_tolerance,
        )
        copied.min_ranks = self.min_ranks.clone()
        copied.max_ranks = self.max_ranks.clone()
        copied.log_weight = self.log_weight.clone()
        copied.materialization_count = int(self.materialization_count)
        copied.materialization_reasons = list(self.materialization_reasons)
        return copied

    def snapshot(self, *, cpu: bool = True) -> dict:
        def convert(value):
            value = value.detach().clone()
            return value.cpu().numpy() if cpu else value

        return {
            "representation": "physical_frame",
            "frame": convert(self.frame),
            "ranks": convert(self.ranks),
            "min_ranks": convert(self.min_ranks),
            "max_ranks": convert(self.max_ranks),
            "log_weight": convert(self.log_weight),
            "physical_dimension": self.physical_dimension,
            "capacity": self.capacity,
            "frame_algorithm_version": GPU_FRAME_ALGORITHM_VERSION,
            "gram_residual": convert(self.gram_residual()),
        }

    def occupation_probability(self, orbital: torch.Tensor) -> torch.Tensor:
        orbital = torch.as_tensor(orbital, dtype=self.dtype, device=self.device)
        if orbital.ndim == 1:
            orbital = orbital.unsqueeze(0).expand(self.batch_size, -1)
        coefficients = torch.einsum("bnr,bn->br", self.frame.conj(), orbital)
        coefficients = coefficients * self._active_mask()
        return coefficients.abs().square().sum(dim=1).real.clamp(0.0, 1.0)

    def gain(self, orbital: torch.Tensor, selected=None) -> BatchedFrameOperationResult:
        orbital = torch.as_tensor(orbital, dtype=self.dtype, device=self.device)
        if orbital.ndim == 1:
            orbital = orbital.unsqueeze(0).expand(self.batch_size, -1)
        selected = (
            torch.ones(self.batch_size, dtype=torch.bool, device=self.device)
            if selected is None
            else torch.as_tensor(selected, dtype=torch.bool, device=self.device)
        )
        coefficients = torch.einsum("bnr,bn->br", self.frame.conj(), orbital)
        coefficients = coefficients * self._active_mask()
        residual = orbital - torch.einsum("bnr,br->bn", self.frame, coefficients)
        probability = residual.abs().square().sum(dim=1).real
        invalid = selected & (
            ~torch.isfinite(probability) | (probability <= self.zero_tolerance)
        )
        if torch.any(invalid):
            raise FloatingPointError("Pauli-blocked or non-finite frame gain branch.")
        if torch.any(selected):
            required = int((self.ranks[selected] + 1).max().item())
            self._ensure_capacity(required)
            normalized = residual / probability.clamp_min(self.zero_tolerance).sqrt().unsqueeze(1)
            rows = torch.nonzero(selected, as_tuple=False).flatten()
            self.frame[rows, :, self.ranks[rows]] = normalized.index_select(0, rows)
            self.ranks[rows] += 1
            self.max_ranks = torch.maximum(self.max_ranks, self.ranks)
        return BatchedFrameOperationResult(probability.clamp(0.0, 1.0), selected)

    def loss(self, orbital: torch.Tensor, selected=None) -> BatchedFrameOperationResult:
        orbital = torch.as_tensor(orbital, dtype=self.dtype, device=self.device)
        if orbital.ndim == 1:
            orbital = orbital.unsqueeze(0).expand(self.batch_size, -1)
        selected = (
            torch.ones(self.batch_size, dtype=torch.bool, device=self.device)
            if selected is None
            else torch.as_tensor(selected, dtype=torch.bool, device=self.device)
        )
        coefficients = torch.einsum("bnr,bn->br", self.frame.conj(), orbital)
        coefficients = coefficients * self._active_mask()
        probability = coefficients.abs().square().sum(dim=1).real
        invalid = selected & (
            (self.ranks == 0)
            | ~torch.isfinite(probability)
            | (probability <= self.zero_tolerance)
        )
        if torch.any(invalid):
            raise FloatingPointError("Empty-orbital or non-finite frame loss branch.")
        if torch.any(selected):
            last_index = (self.ranks - 1).clamp_min(0).unsqueeze(1)
            last = coefficients.gather(1, last_index).squeeze(1)
            norm = probability.clamp_min(0.0).sqrt()
            phase = torch.where(
                last.abs() > 0,
                last / last.abs().clamp_min(torch.finfo(self.real_dtype).tiny),
                torch.ones_like(last),
            )
            target = -phase * norm
            reflector = coefficients.clone()
            reflector.scatter_add_(1, last_index, (-target).unsqueeze(1))
            denominator = reflector.abs().square().sum(dim=1).real
            if torch.any(selected & (denominator <= 0.0)):
                raise FloatingPointError("Degenerate complex Householder reflector.")
            safe_denominator = torch.where(
                selected, denominator, torch.ones_like(denominator)
            )
            frame_reflector = torch.einsum("bnr,br->bn", self.frame, reflector)
            rotated = self.frame - (
                2.0 / safe_denominator
            )[:, None, None] * frame_reflector.unsqueeze(2) * reflector.conj().unsqueeze(1)
            self.frame = torch.where(selected[:, None, None], rotated, self.frame)
            rows = torch.nonzero(selected, as_tuple=False).flatten()
            removed_columns = (self.ranks[rows] - 1).clamp_min(0)
            self.frame[rows, :, removed_columns] = 0.0
            self.ranks[rows] -= 1
        self.min_ranks = torch.minimum(self.min_ranks, self.ranks)
        return BatchedFrameOperationResult(probability.clamp(0.0, 1.0), selected)

    def project_occupied(self, orbital: torch.Tensor, selected=None) -> torch.Tensor:
        selected = (
            torch.ones(self.batch_size, dtype=torch.bool, device=self.device)
            if selected is None
            else torch.as_tensor(selected, dtype=torch.bool, device=self.device)
        )
        probability = self.loss(orbital, selected).probability
        deterministic = self.gain(orbital, selected).probability
        if torch.any(selected & ((deterministic - 1.0).abs() > 2e-5)):
            raise FloatingPointError("Occupied projection loss/gain composition failed.")
        return probability

    def project_empty(self, orbital: torch.Tensor, selected=None) -> torch.Tensor:
        selected = (
            torch.ones(self.batch_size, dtype=torch.bool, device=self.device)
            if selected is None
            else torch.as_tensor(selected, dtype=torch.bool, device=self.device)
        )
        probability = self.gain(orbital, selected).probability
        deterministic = self.loss(orbital, selected).probability
        if torch.any(selected & ((deterministic - 1.0).abs() > 2e-5)):
            raise FloatingPointError("Empty projection gain/loss composition failed.")
        return probability

    def apply_row_phases(self, phase: torch.Tensor) -> None:
        phase = torch.as_tensor(phase, dtype=self.dtype, device=self.device)
        if phase.ndim == 1:
            phase = phase.unsqueeze(0)
        self.frame *= phase.unsqueeze(-1)

    def gram_residual(self) -> torch.Tensor:
        gram = torch.matmul(self.frame.mH, self.frame)
        active = self._active_mask()
        target = torch.diag_embed(active.to(self.dtype))
        difference = gram - target
        denominator = self.ranks.clamp_min(1).to(self.real_dtype).sqrt()
        return torch.linalg.matrix_norm(difference, ord="fro", dim=(-2, -1)) / denominator

    def reorthonormalize(self, *, threshold: float | None = None) -> torch.Tensor:
        threshold = (
            2e-5 if self.dtype == torch.complex64 else 1e-11
        ) if threshold is None else float(threshold)
        residual = self.gram_residual()
        selected = residual > threshold
        if torch.any(selected) and self.capacity:
            q, triangular = torch.linalg.qr(self.frame, mode="reduced")
            diagonal = torch.diagonal(triangular, dim1=-2, dim2=-1)
            active = self._active_mask()
            if torch.any(selected[:, None] & active & (diagonal.abs() <= self.zero_tolerance)):
                raise FloatingPointError("Frame QR detected numerical rank loss.")
            phase = torch.where(
                active,
                diagonal / diagonal.abs().clamp_min(torch.finfo(self.real_dtype).tiny),
                torch.ones_like(diagonal),
            )
            reorthonormalized = q * phase.conj().unsqueeze(1)
            reorthonormalized *= active.unsqueeze(1)
            self.frame = torch.where(
                selected[:, None, None], reorthonormalized, self.frame
            )
        return self.gram_residual()

    def physical_correlation(self, *, reason: str = "explicit_native_consumer") -> torch.Tensor:
        self.materialization_count += 1
        self.materialization_reasons.append(str(reason))
        return torch.matmul(self.frame, self.frame.mH)

    def centered_covariance(self, *, reason: str = "explicit_native_consumer") -> torch.Tensor:
        identity = torch.eye(
            self.physical_dimension, dtype=self.dtype, device=self.device
        )
        return 2.0 * self.physical_correlation(reason=reason) - identity

    def native_state_bytes(self) -> int:
        return int(
            self.frame.numel() * self.frame.element_size()
            + self.ranks.numel() * self.ranks.element_size()
        )
