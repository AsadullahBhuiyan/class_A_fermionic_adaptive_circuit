"""Occupied-frame state and timing primitives for Gaussian trajectories.

The evolution methods in this module never materialize a physical covariance.
Covariance and doubled-projector construction are explicit diagnostic methods.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import time
from typing import Any, Iterator

import numpy as np


FRAME_REPRESENTATIONS = ("physical_frame", "purification_frame")
FRAME_ALGORITHM_VERSION = "append_householder_qr_v2"


class UpdateTimingCollector:
    """Accumulate raw nanosecond timings and operation counts.

    ``coarse`` records only scopes explicitly marked coarse. ``detailed`` records
    every scope. Raw values are retained; timer overhead is measured separately
    and is never subtracted from the data.
    """

    LEVELS = ("off", "coarse", "detailed")

    def __init__(self, level: str = "off") -> None:
        level = str(level).strip().lower()
        if level not in self.LEVELS:
            raise ValueError(f"timing_level must be one of {self.LEVELS}; got {level!r}.")
        self.level = level
        self.total_ns: dict[str, int] = {}
        self.counts: dict[str, int] = {}
        self.minimums: dict[str, float] = {}
        self.maximums: dict[str, float] = {}
        self.cycle_total_ns: dict[int, dict[str, int]] = {}
        self.cycle_counts: dict[int, dict[str, int]] = {}
        self.current_cycle: int | None = None
        self.timer_overhead_ns = self._measure_timer_overhead()

    @staticmethod
    def _measure_timer_overhead(repeats: int = 257) -> int:
        values = np.empty(repeats, dtype=np.int64)
        for index in range(repeats):
            start = time.perf_counter_ns()
            stop = time.perf_counter_ns()
            values[index] = stop - start
        return int(np.median(values))

    def enabled(self, *, detailed: bool = False) -> bool:
        if self.level == "off":
            return False
        return self.level == "detailed" or not detailed

    def set_cycle(self, cycle: int | None) -> None:
        self.current_cycle = None if cycle is None else int(cycle)

    def add_time(self, name: str, elapsed_ns: int, *, detailed: bool = False) -> None:
        if not self.enabled(detailed=detailed):
            return
        key = str(name)
        elapsed = int(elapsed_ns)
        self.total_ns[key] = self.total_ns.get(key, 0) + elapsed
        call_key = f"{key}_calls"
        self.counts[call_key] = self.counts.get(call_key, 0) + 1
        if self.current_cycle is not None:
            cycle_map = self.cycle_total_ns.setdefault(self.current_cycle, {})
            cycle_map[key] = cycle_map.get(key, 0) + elapsed
            cycle_counts = self.cycle_counts.setdefault(self.current_cycle, {})
            cycle_counts[call_key] = cycle_counts.get(call_key, 0) + 1

    def increment(self, name: str, amount: int = 1) -> None:
        if self.level == "off":
            return
        key = str(name)
        count = int(amount)
        self.counts[key] = self.counts.get(key, 0) + count
        if self.current_cycle is not None:
            cycle_map = self.cycle_counts.setdefault(self.current_cycle, {})
            cycle_map[key] = cycle_map.get(key, 0) + count

    def observe_min(self, name: str, value: float) -> None:
        if self.level == "off" or not np.isfinite(value):
            return
        key = str(name)
        numeric = float(value)
        self.minimums[key] = min(self.minimums.get(key, numeric), numeric)

    def observe_max(self, name: str, value: float) -> None:
        if self.level == "off" or not np.isfinite(value):
            return
        key = str(name)
        numeric = float(value)
        self.maximums[key] = max(self.maximums.get(key, numeric), numeric)

    @contextmanager
    def measure(self, name: str, *, detailed: bool = False) -> Iterator[None]:
        if not self.enabled(detailed=detailed):
            yield
            return
        start = time.perf_counter_ns()
        try:
            yield
        finally:
            self.add_time(name, time.perf_counter_ns() - start, detailed=detailed)

    def snapshot(self, *, cycle: int | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "level": self.level,
            "timer_overhead_ns": int(self.timer_overhead_ns),
            "total_ns": {key: int(value) for key, value in self.total_ns.items()},
            "counts": {key: int(value) for key, value in self.counts.items()},
            "minimums": {key: float(value) for key, value in self.minimums.items()},
            "maximums": {key: float(value) for key, value in self.maximums.items()},
        }
        if cycle is not None:
            cycle = int(cycle)
            payload["cycle"] = cycle
            payload["cycle_total_ns"] = {
                key: int(value)
                for key, value in self.cycle_total_ns.get(cycle, {}).items()
            }
            payload["cycle_counts"] = {
                key: int(value)
                for key, value in self.cycle_counts.get(cycle, {}).items()
            }
        else:
            payload["per_cycle_total_ns"] = {
                str(cycle_key): {
                    key: int(value) for key, value in values.items()
                }
                for cycle_key, values in self.cycle_total_ns.items()
            }
            payload["per_cycle_counts"] = {
                str(cycle_key): {
                    key: int(value) for key, value in values.items()
                }
                for cycle_key, values in self.cycle_counts.items()
            }
        return payload


@dataclass(frozen=True)
class FrameOperationResult:
    probability: float
    rank_before: int
    rank_after: int


class OccupiedFrameState:
    """Reduced orthonormal occupied frame for a pure or purified Gaussian state."""

    def __init__(
        self,
        frame: np.ndarray,
        *,
        representation: str,
        physical_dimension: int,
        timing: UpdateTimingCollector | None = None,
        zero_tolerance: float = 1e-14,
        orthonormality_tolerance: float = 1e-10,
    ) -> None:
        representation = str(representation).strip().lower()
        if representation not in FRAME_REPRESENTATIONS:
            raise ValueError(
                f"representation must be one of {FRAME_REPRESENTATIONS}; got {representation!r}."
            )
        frame = np.asarray(frame, dtype=np.complex128)
        if frame.ndim != 2:
            raise ValueError("frame must be a two-dimensional complex array.")
        physical_dimension = int(physical_dimension)
        expected_rows = physical_dimension if representation == "physical_frame" else 2 * physical_dimension
        if frame.shape[0] != expected_rows:
            raise ValueError(
                f"{representation} requires {expected_rows} frame rows; got {frame.shape[0]}."
            )
        zero_tolerance = float(zero_tolerance)
        if not np.isfinite(zero_tolerance) or zero_tolerance < 0.0:
            raise ValueError("zero_tolerance must be finite and nonnegative.")
        self.frame = np.array(frame, dtype=np.complex128, copy=True, order="C")
        self.representation = representation
        self.physical_dimension = physical_dimension
        self.zero_tolerance = zero_tolerance
        self.orthonormality_tolerance = float(orthonormality_tolerance)
        if (
            not np.isfinite(self.orthonormality_tolerance)
            or self.orthonormality_tolerance <= 0.0
        ):
            raise ValueError("orthonormality_tolerance must be positive and finite.")
        self.timing = timing if timing is not None else UpdateTimingCollector("off")
        self.log_weight = 0.0
        self.materialization_count = 0
        self.materialization_reasons: list[str] = []
        self.min_rank = int(self.frame.shape[1])
        self.max_rank = int(self.frame.shape[1])
        self.timing.observe_min("frame_rank", self.rank)
        self.timing.observe_max("frame_rank", self.rank)
        self._validate_finite()
        residual = self.gram_residual()
        if residual > self.orthonormality_tolerance:
            raise ValueError(
                "frame columns must be orthonormal; normalized Gram residual is "
                f"{residual:.3e}."
            )

    @classmethod
    def random_pure(
        cls,
        dimension: int,
        rank: int,
        *,
        rng: np.random.Generator | None = None,
        timing: UpdateTimingCollector | None = None,
        zero_tolerance: float = 1e-14,
    ) -> "OccupiedFrameState":
        """Draw a Haar-Stiefel occupied frame without constructing a covariance."""

        dimension = int(dimension)
        rank = int(rank)
        if not (0 <= rank <= dimension):
            raise ValueError("rank must lie in 0..dimension.")
        if rank == 0:
            frame = np.empty((dimension, 0), dtype=np.complex128)
        else:
            rng = np.random.default_rng() if rng is None else rng
            raw = rng.standard_normal((dimension, rank)) + 1j * rng.standard_normal(
                (dimension, rank)
            )
            frame, triangular = np.linalg.qr(raw, mode="reduced")
            diagonal = np.diag(triangular)
            phase = np.ones_like(diagonal)
            nonzero = np.abs(diagonal) > 0.0
            phase[nonzero] = diagonal[nonzero] / np.abs(diagonal[nonzero])
            frame = frame * phase.conj()[None, :]
        return cls(
            frame,
            representation="physical_frame",
            physical_dimension=dimension,
            timing=timing,
            zero_tolerance=zero_tolerance,
        )

    @classmethod
    def from_centered_covariance(
        cls,
        centered_covariance: np.ndarray,
        *,
        representation: str,
        timing: UpdateTimingCollector | None = None,
        purity_tolerance: float = 1e-9,
        zero_tolerance: float = 1e-14,
    ) -> "OccupiedFrameState":
        representation = str(representation).strip().lower()
        centered = np.asarray(centered_covariance, dtype=np.complex128)
        if centered.ndim != 2 or centered.shape[0] != centered.shape[1]:
            raise ValueError("centered_covariance must be a square matrix.")
        centered = 0.5 * (centered + centered.conj().T)
        dimension = int(centered.shape[0])
        correlation = 0.5 * (centered + np.eye(dimension, dtype=np.complex128))
        evals, evecs = np.linalg.eigh(0.5 * (correlation + correlation.conj().T))
        if np.min(evals) < -purity_tolerance or np.max(evals) > 1.0 + purity_tolerance:
            raise ValueError("Initial physical correlation has eigenvalues outside [0, 1].")
        evals = np.clip(np.real(evals), 0.0, 1.0)

        if representation == "physical_frame":
            defect = float(np.max(np.minimum(evals, 1.0 - evals))) if evals.size else 0.0
            if defect > float(purity_tolerance):
                raise ValueError(
                    "physical_frame requires a pure initial covariance; "
                    f"maximum occupation defect is {defect:.3e}."
                )
            occupied = evals > 0.5
            frame = evecs[:, occupied]
        elif representation == "purification_frame":
            sqrt_c = (evecs * np.sqrt(evals)[None, :]) @ evecs.conj().T
            sqrt_h = (evecs * np.sqrt(1.0 - evals)[None, :]) @ evecs.conj().T
            frame = np.vstack((sqrt_h, sqrt_c))
        else:
            raise ValueError(
                f"representation must be one of {FRAME_REPRESENTATIONS}; got {representation!r}."
            )
        return cls(
            frame,
            representation=representation,
            physical_dimension=dimension,
            timing=timing,
            zero_tolerance=zero_tolerance,
        )

    @classmethod
    def maximally_mixed(
        cls,
        dimension: int,
        *,
        timing: UpdateTimingCollector | None = None,
        zero_tolerance: float = 1e-14,
    ) -> "OccupiedFrameState":
        dimension = int(dimension)
        identity = np.eye(dimension, dtype=np.complex128) / np.sqrt(2.0)
        return cls(
            np.vstack((identity, identity)),
            representation="purification_frame",
            physical_dimension=dimension,
            timing=timing,
            zero_tolerance=zero_tolerance,
        )

    @property
    def rank(self) -> int:
        return int(self.frame.shape[1])

    @property
    def physical_frame(self) -> np.ndarray:
        if self.representation == "physical_frame":
            return self.frame
        return self.frame[self.physical_dimension :, :]

    def _validate_finite(self) -> None:
        if not np.all(np.isfinite(self.frame)):
            raise FloatingPointError("Occupied frame contains non-finite entries.")

    def copy(self) -> "OccupiedFrameState":
        copied = OccupiedFrameState(
            self.frame,
            representation=self.representation,
            physical_dimension=self.physical_dimension,
            timing=self.timing,
            zero_tolerance=self.zero_tolerance,
            orthonormality_tolerance=self.orthonormality_tolerance,
        )
        copied.log_weight = float(self.log_weight)
        copied.min_rank = int(self.min_rank)
        copied.max_rank = int(self.max_rank)
        copied.materialization_count = int(self.materialization_count)
        copied.materialization_reasons = list(self.materialization_reasons)
        return copied

    def snapshot(self, *, copy: bool = True) -> dict[str, Any]:
        frame = np.array(self.frame, copy=True) if copy else self.frame
        return {
            "representation": self.representation,
            "frame": frame,
            "physical_dimension": int(self.physical_dimension),
            "rank": int(self.rank),
            "log_weight": float(self.log_weight),
            "min_rank": int(self.min_rank),
            "max_rank": int(self.max_rank),
            "gram_residual": float(self.gram_residual()),
            "frame_algorithm_version": FRAME_ALGORITHM_VERSION,
            "materialization_count": int(self.materialization_count),
            "materialization_reasons": list(self.materialization_reasons),
        }

    def _physical_overlap(self, support_idx: np.ndarray, orbital_local: np.ndarray) -> np.ndarray:
        support_idx = np.asarray(support_idx, dtype=np.int64).reshape(-1)
        orbital_local = np.asarray(orbital_local, dtype=np.complex128).reshape(-1)
        if support_idx.size != orbital_local.size:
            raise ValueError("support_idx and orbital_local must have equal lengths.")
        if np.any(support_idx < 0) or np.any(support_idx >= self.physical_dimension):
            raise IndexError("support_idx lies outside the physical one-particle space.")
        return self.physical_frame[support_idx, :].conj().T @ orbital_local

    def occupation_probability_local(
        self, support_idx: np.ndarray, orbital_local: np.ndarray
    ) -> float:
        with self.timing.measure("occupation_probability", detailed=True):
            coefficients = self._physical_overlap(support_idx, orbital_local)
            probability = float(np.real(np.vdot(coefficients, coefficients)))
        return float(np.clip(probability, 0.0, 1.0))

    def _embedded_residual(
        self,
        support_idx: np.ndarray,
        orbital_local: np.ndarray,
        coefficients: np.ndarray,
    ) -> np.ndarray:
        residual = -(self.frame @ coefficients)
        row_offset = 0 if self.representation == "physical_frame" else self.physical_dimension
        residual[row_offset + np.asarray(support_idx, dtype=np.int64)] += np.asarray(
            orbital_local, dtype=np.complex128
        )
        return residual

    def gain_local(
        self, support_idx: np.ndarray, orbital_local: np.ndarray
    ) -> FrameOperationResult:
        rank_before = self.rank
        self.timing.increment("gain_count")
        with self.timing.measure("gain_overlap", detailed=True):
            coefficients = self._physical_overlap(support_idx, orbital_local)
        with self.timing.measure("gain_residual_projection", detailed=True):
            residual = self._embedded_residual(support_idx, orbital_local, coefficients)
            probability = float(np.real(np.vdot(residual, residual)))
        if not np.isfinite(probability) or probability <= self.zero_tolerance:
            self.timing.increment("zero_branch_detection_count")
            raise FloatingPointError(
                f"Pauli-blocked gain has probability {probability!r} at tolerance "
                f"{self.zero_tolerance:.3e}."
            )
        with self.timing.measure("gain_normalize_append", detailed=True):
            residual /= np.sqrt(probability)
            updated = np.empty(
                (self.frame.shape[0], self.frame.shape[1] + 1), dtype=np.complex128
            )
            updated[:, :-1] = self.frame
            updated[:, -1] = residual
            self.frame = updated
        self.max_rank = max(self.max_rank, self.rank)
        self.timing.observe_min("frame_rank", self.rank)
        self.timing.observe_max("frame_rank", self.rank)
        return FrameOperationResult(float(probability), rank_before, self.rank)

    @staticmethod
    def _householder_to_last(vector: np.ndarray) -> tuple[np.ndarray, float]:
        vector = np.asarray(vector, dtype=np.complex128).reshape(-1)
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            raise ValueError("Cannot build a Householder reflector for the zero vector.")
        last = vector[-1]
        phase = 1.0 + 0.0j if abs(last) == 0.0 else last / abs(last)
        target = -phase * norm
        reflector = vector.copy()
        reflector[-1] -= target
        denominator = float(np.real(np.vdot(reflector, reflector)))
        if not np.isfinite(denominator) or denominator <= 0.0:
            raise FloatingPointError("Degenerate complex Householder reflector.")
        return reflector, 2.0 / denominator

    def loss_local(
        self, support_idx: np.ndarray, orbital_local: np.ndarray
    ) -> FrameOperationResult:
        rank_before = self.rank
        if rank_before == 0:
            self.timing.increment("zero_branch_detection_count")
            raise FloatingPointError("Loss from an empty frame is a zero branch.")
        self.timing.increment("loss_count")
        with self.timing.measure("loss_overlap", detailed=True):
            coefficients = self._physical_overlap(support_idx, orbital_local)
            probability = float(np.real(np.vdot(coefficients, coefficients)))
        if not np.isfinite(probability) or probability <= self.zero_tolerance:
            self.timing.increment("zero_branch_detection_count")
            raise FloatingPointError(
                f"Empty-orbital loss has probability {probability!r} at tolerance "
                f"{self.zero_tolerance:.3e}."
            )
        with self.timing.measure("loss_householder_build", detailed=True):
            reflector, beta = self._householder_to_last(coefficients)
        with self.timing.measure("loss_householder_apply", detailed=True):
            frame_reflector = self.frame @ reflector
            rotated = self.frame - beta * np.outer(frame_reflector, reflector.conj())
        with self.timing.measure("loss_column_delete", detailed=True):
            self.frame = np.array(rotated[:, :-1], copy=True, order="C")
        self.min_rank = min(self.min_rank, self.rank)
        self.timing.observe_min("frame_rank", self.rank)
        self.timing.observe_max("frame_rank", self.rank)
        return FrameOperationResult(float(probability), rank_before, self.rank)

    def project_occupied_local(
        self, support_idx: np.ndarray, orbital_local: np.ndarray
    ) -> float:
        self.timing.increment("occupied_projector_count")
        selected = self.loss_local(support_idx, orbital_local).probability
        deterministic = self.gain_local(support_idx, orbital_local).probability
        if abs(deterministic - 1.0) > 1e-9:
            raise FloatingPointError(
                f"Occupied projector second letter should have probability one; got {deterministic:.16g}."
            )
        return float(selected)

    def project_empty_local(
        self, support_idx: np.ndarray, orbital_local: np.ndarray
    ) -> float:
        self.timing.increment("empty_projector_count")
        selected = self.gain_local(support_idx, orbital_local).probability
        deterministic = self.loss_local(support_idx, orbital_local).probability
        if abs(deterministic - 1.0) > 1e-9:
            raise FloatingPointError(
                f"Empty projector second letter should have probability one; got {deterministic:.16g}."
            )
        return float(selected)

    def gram_residual(self) -> float:
        rank = self.rank
        if rank == 0:
            return 0.0
        gram = self.frame.conj().T @ self.frame
        return float(np.linalg.norm(gram - np.eye(rank), ord="fro") / np.sqrt(rank))

    def reorthonormalize(self, *, force: bool = False) -> float:
        """Restore the Stiefel constraint with a thin QR when drift warrants it."""

        residual = self.gram_residual()
        if self.rank == 0 or (
            not force and residual <= 0.1 * self.orthonormality_tolerance
        ):
            return residual
        with self.timing.measure("frame_reorthonormalization"):
            frame, triangular = np.linalg.qr(self.frame, mode="reduced")
            diagonal = np.diag(triangular)
            if np.any(np.abs(diagonal) <= self.zero_tolerance):
                raise FloatingPointError(
                    "Frame QR detected numerical rank loss during reorthonormalization."
                )
            phase = diagonal / np.abs(diagonal)
            self.frame = np.ascontiguousarray(frame * phase.conj()[None, :])
            self.timing.increment("frame_reorthonormalization_count")
        return self.gram_residual()

    def physical_correlation(self, *, reason: str = "explicit_native_consumer") -> np.ndarray:
        with self.timing.measure("physical_covariance_reconstruction", detailed=True):
            self.materialization_count += 1
            self.materialization_reasons.append(str(reason))
            physical = self.physical_frame
            return physical @ physical.conj().T

    def centered_covariance(self, *, reason: str = "explicit_native_consumer") -> np.ndarray:
        correlation = self.physical_correlation(reason=reason)
        return 2.0 * correlation - np.eye(self.physical_dimension, dtype=np.complex128)

    def doubled_projector(self) -> np.ndarray:
        if self.representation != "purification_frame":
            raise ValueError("doubled_projector is defined only for a purification frame.")
        with self.timing.measure("choi_projector_reconstruction", detailed=True):
            return self.frame @ self.frame.conj().T

    def regional_charge(self, rows: np.ndarray | slice) -> float:
        block = self.physical_frame[rows, :]
        return float(np.real(np.vdot(block, block)))

    def selected_correlators(
        self, left_rows: np.ndarray, right_rows: np.ndarray
    ) -> np.ndarray:
        """Return selected entries of ``C`` directly from physical frame rows."""

        left = np.asarray(left_rows, dtype=np.int64).reshape(-1)
        right = np.asarray(right_rows, dtype=np.int64).reshape(-1)
        if left.size != right.size:
            raise ValueError("left_rows and right_rows must have equal lengths.")
        if (
            np.any(left < 0)
            or np.any(right < 0)
            or np.any(left >= self.physical_dimension)
            or np.any(right >= self.physical_dimension)
        ):
            raise IndexError("Selected correlator rows lie outside the physical space.")
        physical = self.physical_frame
        return np.asarray(
            [
                np.dot(physical[left_index], physical[right_index].conj())
                for left_index, right_index in zip(left, right)
            ],
            dtype=np.complex128,
        )

    @staticmethod
    def _binary_entropy(occupations: np.ndarray, *, tolerance: float = 1e-14) -> float:
        occupations = np.clip(np.real(np.asarray(occupations)), 0.0, 1.0)
        interior = occupations[(occupations > tolerance) & (occupations < 1.0 - tolerance)]
        if interior.size == 0:
            return 0.0
        return float(-np.sum(interior * np.log(interior) + (1.0 - interior) * np.log1p(-interior)))

    def physical_entropy(self) -> float:
        with self.timing.measure("global_entropy_eigh_or_svd", detailed=True):
            singular = np.linalg.svd(self.physical_frame, compute_uv=False)
            return self._binary_entropy(singular * singular)

    def regional_entropy(self, rows: np.ndarray | slice) -> float:
        with self.timing.measure("regional_entropy_eigh_or_svd", detailed=True):
            singular = np.linalg.svd(self.physical_frame[rows, :], compute_uv=False)
            return self._binary_entropy(singular * singular)


__all__ = [
    "FRAME_REPRESENTATIONS",
    "FrameOperationResult",
    "OccupiedFrameState",
    "UpdateTimingCollector",
]
