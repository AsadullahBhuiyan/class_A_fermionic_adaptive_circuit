"""Compact replay-complete record capture for pure occupied-frame trajectories."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import torch


OBSERVER_SCHEMA = "pure_tangent_replay_record_observer_v1"
CHANNEL_LABELS = ("Ap", "Am", "Bp", "Bm")


def _to_numpy(value: Any, *, dtype: np.dtype | None = None) -> np.ndarray:
    array = (
        value.detach().cpu().numpy()
        if torch.is_tensor(value)
        else np.asarray(value)
    )
    return np.asarray(array, dtype=dtype)


def native_frame_arrays(native: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Copy a canonical native occupied-frame snapshot to NumPy."""

    frame = _to_numpy(native["frame"], dtype=np.complex128)
    ranks = _to_numpy(native["ranks"], dtype=np.int64)
    if frame.ndim != 3 or ranks.shape != (frame.shape[0],):
        raise ValueError("native occupied-frame snapshot has invalid shapes")
    if np.any(ranks < 0) or np.any(ranks > frame.shape[2]):
        raise ValueError("native occupied-frame ranks are outside the saved capacity")
    return np.array(frame, copy=True), np.array(ranks, copy=True)


def pack_boolean_record(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pack a sample-major Boolean record and retain its exact logical shape."""

    array = np.asarray(values, dtype=np.bool_)
    if array.ndim < 2:
        raise ValueError("a packed record must have a leading sample dimension")
    flattened = array.reshape(array.shape[0], -1).astype(np.uint8, copy=False)
    packed = np.packbits(flattened, axis=1, bitorder="little")
    return packed, np.asarray(array.shape, dtype=np.int64)


def unpack_boolean_record(packed: np.ndarray, shape: np.ndarray) -> np.ndarray:
    """Invert :func:`pack_boolean_record` without accepting ambiguous padding."""

    logical_shape = tuple(int(value) for value in np.asarray(shape).tolist())
    if len(logical_shape) < 2 or any(value <= 0 for value in logical_shape):
        raise ValueError("packed record shape metadata is invalid")
    raw = np.asarray(packed, dtype=np.uint8)
    expected_bytes = (int(np.prod(logical_shape[1:])) + 7) // 8
    if raw.shape != (logical_shape[0], expected_bytes):
        raise ValueError("packed record byte shape disagrees with logical shape")
    bits = np.unpackbits(raw, axis=1, count=int(np.prod(logical_shape[1:])), bitorder="little")
    return bits.astype(np.bool_, copy=False).reshape(logical_shape)


class ReplayRecordObserver:
    """Capture prepared cycle zero, the ordered Born record, and record weights."""

    def __init__(self, *, samples: int, cycles: int, updates_per_cycle: int) -> None:
        self.samples = int(samples)
        self.cycles = int(cycles)
        self.updates_per_cycle = int(updates_per_cycle)
        if self.samples <= 0 or self.cycles <= 0 or self.updates_per_cycle <= 0:
            raise ValueError("observer dimensions must be positive")
        shape = (self.samples, self.cycles, self.updates_per_cycle)
        self.schedule = np.full(shape, -1, dtype=np.int32)
        self.outcomes = np.zeros(shape + (len(CHANNEL_LABELS),), dtype=np.bool_)
        self.targets = np.zeros_like(self.outcomes)
        self.event_seen = np.zeros(shape, dtype=np.bool_)
        self.measurement_log_probability = np.zeros(
            (self.samples, self.cycles + 1), dtype=np.float64
        )
        self.site_event_count = np.zeros(
            (self.samples, self.cycles + 1), dtype=np.int32
        )
        self.channel_event_count = np.zeros_like(self.site_event_count)
        self.initial_frame: np.ndarray | None = None
        self.initial_ranks: np.ndarray | None = None

    def capture_native_cycle(
        self,
        *,
        cycle: int,
        state: Any,
        batch_start: int,
        batch_count: int,
        **_: Any,
    ) -> None:
        """Copy only the prepared cycle-zero state; later callbacks remain cheap."""

        if int(batch_start) != 0 or int(batch_count) != self.samples:
            raise RuntimeError("record acquisition requires one canonical engine batch per task")
        if int(cycle) != 0:
            return
        if self.initial_frame is not None:
            raise RuntimeError("duplicate prepared cycle-zero frame observation")
        frame, ranks = native_frame_arrays(state.snapshot(cpu=True))
        if frame.shape[0] != self.samples:
            raise ValueError("prepared frame sample dimension is incorrect")
        self.initial_frame, self.initial_ranks = frame, ranks

    def record_event(
        self,
        *,
        cycle: int,
        update_index: int,
        site_ids: torch.Tensor,
        sample_offsets: torch.Tensor,
        channel_labels: tuple[str, ...],
        outcome_occupied: torch.Tensor,
        target_occupied: torch.Tensor,
        conditional_log_probability: torch.Tensor,
        **_: Any,
    ) -> None:
        cycle_index = int(cycle) - 1
        update_index = int(update_index)
        if not 0 <= cycle_index < self.cycles:
            raise ValueError(f"record event has invalid cycle {cycle}")
        if not 0 <= update_index < self.updates_per_cycle:
            raise ValueError(f"record event has invalid update index {update_index}")
        if tuple(channel_labels) != CHANNEL_LABELS:
            raise ValueError(f"unexpected channel ordering {tuple(channel_labels)!r}")

        rows = _to_numpy(sample_offsets, dtype=np.int64).reshape(-1)
        sites = _to_numpy(site_ids, dtype=np.int64).reshape(-1)
        outcomes = _to_numpy(outcome_occupied, dtype=np.bool_)
        targets = _to_numpy(target_occupied, dtype=np.bool_)
        conditional = _to_numpy(conditional_log_probability, dtype=np.float64)
        expected_matrix_shape = (rows.size, len(CHANNEL_LABELS))
        if sites.shape != (rows.size,):
            raise ValueError("record site IDs do not align with sample offsets")
        if outcomes.shape != expected_matrix_shape or targets.shape != expected_matrix_shape:
            raise ValueError("record outcome/target payload has an invalid shape")
        if conditional.shape != expected_matrix_shape or not np.all(np.isfinite(conditional)):
            raise FloatingPointError("record log-probability payload is invalid")
        if np.any(rows < 0) or np.any(rows >= self.samples):
            raise IndexError("record sample offset is outside this batch")
        if np.any(self.event_seen[rows, cycle_index, update_index]):
            raise RuntimeError("duplicate record event for a sample/cycle/update")

        self.schedule[rows, cycle_index, update_index] = sites.astype(np.int32)
        self.outcomes[rows, cycle_index, update_index] = outcomes
        self.targets[rows, cycle_index, update_index] = targets
        self.event_seen[rows, cycle_index, update_index] = True
        self.measurement_log_probability[rows, cycle_index + 1] += conditional.sum(axis=1)
        self.site_event_count[rows, cycle_index + 1] += 1
        self.channel_event_count[rows, cycle_index + 1] += len(CHANNEL_LABELS)

    def result_arrays(self) -> dict[str, np.ndarray]:
        if self.initial_frame is None or self.initial_ranks is None:
            raise RuntimeError("prepared cycle-zero frame was not captured")
        if not np.all(self.event_seen) or np.any(self.schedule < 0):
            missing = int(np.size(self.event_seen) - np.count_nonzero(self.event_seen))
            raise RuntimeError(f"ordered trajectory record is incomplete ({missing} missing events)")
        expected_sites = self.updates_per_cycle
        if np.any(self.site_event_count[:, 1:] != expected_sites):
            raise RuntimeError("site-event counts are incomplete")
        if np.any(self.channel_event_count[:, 1:] != len(CHANNEL_LABELS) * expected_sites):
            raise RuntimeError("channel-event counts are incomplete")
        cumulative = np.cumsum(self.measurement_log_probability, axis=1)
        packed_outcomes, outcome_shape = pack_boolean_record(self.outcomes)
        packed_targets, target_shape = pack_boolean_record(self.targets)
        return {
            "initial_frame": np.array(self.initial_frame, copy=True),
            "initial_ranks": np.array(self.initial_ranks, copy=True),
            "record_schedule": np.array(self.schedule, copy=True),
            "record_outcomes_packed": packed_outcomes,
            "record_outcomes_shape": outcome_shape,
            "record_targets_packed": packed_targets,
            "record_targets_shape": target_shape,
            "measurement_log_probability": np.array(
                self.measurement_log_probability, copy=True
            ),
            "cumulative_log_probability": cumulative,
            "site_event_count": np.array(self.site_event_count, copy=True),
            "channel_event_count": np.array(self.channel_event_count, copy=True),
            "channel_labels": np.asarray(CHANNEL_LABELS),
        }

