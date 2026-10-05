from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np

from h1_io import save_npz_atomic, sha256_file


RECORD_SCHEMA = "ordered_site_channel_born_record_v2_compact_replay"
LEGACY_RECORD_SCHEMA = "ordered_site_channel_born_record_v1"
DIGEST_SCHEMA = "ordered_born_record_digest_v1"


def load_ordered_record(path: Path | str) -> dict[str, np.ndarray]:
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        schema = str(np.asarray(data["schema"]).item())
        if schema not in (RECORD_SCHEMA, LEGACY_RECORD_SCHEMA):
            raise ValueError(f"unsupported ordered-record schema {schema!r}")
        bit_shape = tuple(int(value) for value in np.asarray(data["bit_shape"]).tolist())
        outcomes = np.unpackbits(
            np.asarray(data["outcome_bits_packed"]), axis=-1, count=bit_shape[-1], bitorder="little"
        ).astype(np.bool_, copy=False)
        targets = np.unpackbits(
            np.asarray(data["target_bits_packed"]), axis=-1, count=bit_shape[-1], bitorder="little"
        ).astype(np.bool_, copy=False)
        outcomes = outcomes.reshape(bit_shape)
        targets = targets.reshape(bit_shape)
        payload = {
            "site_ids": np.asarray(data["site_ids"], dtype=np.int64),
            "channel_count": np.asarray(data["channel_count"], dtype=np.uint8),
            "outcomes": outcomes,
            "targets": targets,
            "transfers": targets.astype(np.int8) - outcomes.astype(np.int8),
        }
        if schema == LEGACY_RECORD_SCHEMA:
            payload["conditional_log_probability"] = np.asarray(
                data["conditional_log_probability"], dtype=np.float64
            )
            payload["realized_probability"] = np.asarray(
                data["realized_probability"], dtype=np.float64
            )
        payload["self_information_per_cycle"] = np.asarray(
            data["self_information_per_cycle"], dtype=np.float64
        )
        payload["cumulative_self_information"] = np.asarray(
            data["cumulative_self_information"], dtype=np.float64
        )
        payload["total_log_probability"] = -payload["cumulative_self_information"][:, -1]
        return payload


class OrderedBornRecordWriter:
    """Collect one complete shard without covariance histories.

    Production callers provide ``buffer_device=model.device``.  In that mode every
    event is written directly into preallocated device tensors and the complete record
    crosses to CPU once, after the dynamics finish.  This avoids a CUDA synchronization
    at every site update while retaining the exact archived schema.
    """

    def __init__(
        self,
        *,
        samples: int,
        cycles: int,
        sites_per_cycle: int,
        expected_site_ids: Iterable[int] | None = None,
        buffer_device: Any | None = None,
    ) -> None:
        self.samples = int(samples)
        self.cycles = int(cycles)
        self.sites_per_cycle = int(sites_per_cycle)
        if self.samples <= 0 or self.cycles <= 0 or self.sites_per_cycle <= 0:
            raise ValueError("samples, cycles, and sites_per_cycle must be positive")
        self.expected_site_ids = (
            None
            if expected_site_ids is None
            else np.sort(np.asarray(list(expected_site_ids), dtype=np.int32))
        )
        shape = (self.samples, self.cycles, self.sites_per_cycle)
        event_shape = shape + (4,)
        self._capture_backend = "numpy"
        self._storage_backend = "numpy"
        self._device_buffer_bytes = 0
        if buffer_device is None:
            self.site_ids = np.full(shape, -1, dtype=np.int32)
            self.channel_count = np.zeros(shape, dtype=np.uint8)
            self.outcomes = np.zeros(event_shape, dtype=np.bool_)
            self.targets = np.zeros(event_shape, dtype=np.bool_)
            self.transfers = np.zeros(event_shape, dtype=np.int8)
            self.conditional_log_probability = np.full(
                event_shape, np.nan, dtype=np.float64
            )
            self.realized_probability = np.full(
                event_shape, np.nan, dtype=np.float64
            )
            self.reset_covariance = np.full(
                event_shape, np.nan, dtype=np.float64
            )
        else:
            import torch

            device = torch.device(buffer_device)
            self._capture_backend = "torch_device_buffer"
            self._storage_backend = "torch"
            self.site_ids = torch.full(
                shape, -1, dtype=torch.int32, device=device
            )
            self.channel_count = torch.zeros(
                shape, dtype=torch.uint8, device=device
            )
            self.outcomes = torch.zeros(
                event_shape, dtype=torch.bool, device=device
            )
            self.targets = torch.zeros(
                event_shape, dtype=torch.bool, device=device
            )
            self.transfers = torch.zeros(
                event_shape, dtype=torch.int8, device=device
            )
            self.conditional_log_probability = torch.full(
                event_shape, torch.nan, dtype=torch.float64, device=device
            )
            self.realized_probability = torch.full(
                event_shape, torch.nan, dtype=torch.float64, device=device
            )
            self.reset_covariance = torch.full(
                event_shape, torch.nan, dtype=torch.float64, device=device
            )
            self._device_buffer_bytes = int(
                sum(
                    value.numel() * value.element_size()
                    for value in (
                        self.site_ids,
                        self.channel_count,
                        self.outcomes,
                        self.targets,
                        self.transfers,
                        self.conditional_log_probability,
                        self.realized_probability,
                        self.reset_covariance,
                    )
                )
            )

    @staticmethod
    def _cpu(value: Any) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return np.asarray(value)

    def materialize_cpu(self) -> None:
        """Synchronize and release device record buffers exactly once."""
        if self._storage_backend != "torch":
            return
        for name in (
            "site_ids",
            "channel_count",
            "outcomes",
            "targets",
            "transfers",
            "conditional_log_probability",
            "realized_probability",
            "reset_covariance",
        ):
            value = getattr(self, name)
            setattr(self, name, value.detach().cpu().numpy())
        self._storage_backend = "numpy"

    def __call__(
        self,
        *,
        cycle: int,
        update_index: int,
        site_ids: Any,
        sample_indices: Any,
        channel_labels: tuple[str, ...],
        outcome_occupied: Any,
        target_occupied: Any,
        transfer: Any,
        conditional_log_probability: Any,
        realized_probability: Any,
        reset_covariance: Any,
        **_: Any,
    ) -> None:
        cycle_index = int(cycle) - 1
        update_index = int(update_index)
        if not (0 <= cycle_index < self.cycles):
            raise IndexError(f"record cycle {cycle} outside 1..{self.cycles}")
        if not (0 <= update_index < self.sites_per_cycle):
            raise IndexError(f"record update index {update_index} outside the declared cycle")
        channel_count = len(channel_labels)
        if channel_count not in (2, 4):
            raise ValueError(f"unexpected channel order {channel_labels!r}")

        if self._storage_backend == "torch":
            import torch

            device = self.site_ids.device
            sample_indices = torch.as_tensor(
                sample_indices, dtype=torch.long, device=device
            ).reshape(-1)
            site_ids = torch.as_tensor(
                site_ids, dtype=torch.int32, device=device
            ).reshape(-1)
            index = (sample_indices, cycle_index, update_index)
            self.site_ids[index] = site_ids
            self.channel_count[index] = channel_count
            destination = (
                sample_indices,
                cycle_index,
                update_index,
                slice(0, channel_count),
            )
            assignments = (
                ("outcomes", outcome_occupied, torch.bool),
                ("targets", target_occupied, torch.bool),
                ("transfers", transfer, torch.int8),
                (
                    "conditional_log_probability",
                    conditional_log_probability,
                    torch.float64,
                ),
                ("realized_probability", realized_probability, torch.float64),
                ("reset_covariance", reset_covariance, torch.float64),
            )
            for name, value, dtype in assignments:
                getattr(self, name)[destination] = torch.as_tensor(
                    value, dtype=dtype, device=device
                )[:, :channel_count]
            return

        sample_indices = self._cpu(sample_indices).astype(np.int64, copy=False)
        site_ids = self._cpu(site_ids).astype(np.int32, copy=False)
        if np.any(sample_indices < 0) or np.any(sample_indices >= self.samples):
            raise IndexError("record observer emitted sample indices outside this shard")

        index = (sample_indices, cycle_index, update_index)
        if np.any(self.site_ids[index] >= 0):
            raise RuntimeError("duplicate record event for a sample/cycle/update index")
        self.site_ids[index] = site_ids
        self.channel_count[index] = channel_count
        destination = (sample_indices, cycle_index, update_index, slice(0, channel_count))
        self.outcomes[destination] = self._cpu(outcome_occupied)
        self.targets[destination] = self._cpu(target_occupied)
        self.transfers[destination] = self._cpu(transfer)
        self.conditional_log_probability[destination] = self._cpu(conditional_log_probability)
        self.realized_probability[destination] = self._cpu(realized_probability)
        self.reset_covariance[destination] = self._cpu(reset_covariance)

    def validate(self) -> dict[str, Any]:
        self.materialize_cpu()
        missing = int(np.count_nonzero(self.site_ids < 0))
        if missing:
            raise RuntimeError(f"ordered record is missing {missing} sample/cycle/update entries")
        active = np.arange(4)[None, None, None, :] < self.channel_count[..., None]
        if not np.isfinite(self.conditional_log_probability[active]).all():
            raise FloatingPointError("record contains non-finite conditional log probabilities")
        if not np.isfinite(self.realized_probability[active]).all():
            raise FloatingPointError("record contains non-finite realized probabilities")
        if np.any(self.realized_probability[active] < 0.0) or np.any(
            self.realized_probability[active] > 1.0
        ):
            raise ValueError("record contains probabilities outside [0,1]")
        if not np.array_equal(
            self.transfers[active], self.targets[active].astype(np.int8) - self.outcomes[active].astype(np.int8)
        ):
            raise ValueError("record transfer is inconsistent with target minus outcome")
        if self.expected_site_ids is not None:
            for sample in range(self.samples):
                for cycle in range(self.cycles):
                    if not np.array_equal(np.sort(self.site_ids[sample, cycle]), self.expected_site_ids):
                        raise ValueError(
                            f"sample {sample} cycle {cycle + 1} is not a permutation of active sites"
                        )
        return {
            "schema": RECORD_SCHEMA,
            "samples": self.samples,
            "cycles": self.cycles,
            "sites_per_cycle": self.sites_per_cycle,
            "event_count": int(np.sum(self.channel_count, dtype=np.int64)),
            "minimum_realized_probability": float(np.min(self.realized_probability[active])),
            "maximum_abs_log_probability": float(
                np.max(np.abs(self.conditional_log_probability[active]))
            ),
            "capture_backend": self._capture_backend,
            "device_buffer_bytes": self._device_buffer_bytes,
        }

    def save(self, path: Path | str) -> dict[str, Any]:
        diagnostics = self.validate()
        path = Path(path)
        active = np.arange(4)[None, None, None, :] < self.channel_count[..., None]
        self_information = -np.sum(
            np.where(active, self.conditional_log_probability, 0.0), axis=(2, 3)
        )
        cumulative_self_information = np.cumsum(self_information, axis=1)
        active_transfer = np.where(active, self.transfers, 0)
        signed_transfer_per_cycle = np.sum(active_transfer, axis=(2, 3), dtype=np.int64)
        absolute_transfer_per_cycle = np.sum(
            np.abs(active_transfer), axis=(2, 3), dtype=np.int64
        )
        wrong_outcome_per_cycle = np.sum(
            np.where(active, self.outcomes != self.targets, False), axis=(2, 3), dtype=np.int64
        )
        save_npz_atomic(
            path,
            schema=np.asarray(RECORD_SCHEMA),
            site_ids=self.site_ids,
            channel_count=self.channel_count,
            outcome_bits_packed=np.packbits(self.outcomes, axis=-1, bitorder="little"),
            target_bits_packed=np.packbits(self.targets, axis=-1, bitorder="little"),
            bit_shape=np.asarray(self.outcomes.shape, dtype=np.int64),
            self_information_per_cycle=self_information,
            cumulative_self_information=cumulative_self_information,
            signed_transfer_per_cycle=signed_transfer_per_cycle,
            absolute_transfer_per_cycle=absolute_transfer_per_cycle,
            wrong_outcome_per_cycle=wrong_outcome_per_cycle,
            storage_contract=np.asarray(
                "site_word+bitpacked_outcomes_targets+cycle_log_sums;event_floats_discarded"
            ),
        )
        return {
            **diagnostics,
            "path": str(path),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "event_float_arrays_archived": 0,
        }


class OrderedBornRecordDigestWriter:
    """Stream a reproducibility digest and cycle aggregates without a replay word."""

    def __init__(
        self,
        *,
        samples: int,
        cycles: int,
        sites_per_cycle: int,
        expected_site_ids: Iterable[int],
    ) -> None:
        import hashlib

        self.samples = int(samples)
        self.cycles = int(cycles)
        self.sites_per_cycle = int(sites_per_cycle)
        self.expected_site_ids = np.sort(
            np.asarray(list(expected_site_ids), dtype=np.int32)
        )
        self.next_update = np.zeros((self.samples, self.cycles), dtype=np.int32)
        self.site_ids = np.full(
            (self.samples, self.cycles, self.sites_per_cycle), -1, dtype=np.int32
        )
        self.self_information = np.zeros((self.samples, self.cycles), dtype=np.float64)
        self.signed_transfer = np.zeros((self.samples, self.cycles), dtype=np.int64)
        self._digests = [hashlib.sha256() for _ in range(self.samples)]

    @staticmethod
    def _cpu(value: Any) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return np.asarray(value)

    def __call__(
        self,
        *,
        cycle: int,
        update_index: int,
        site_ids: Any,
        sample_indices: Any,
        outcome_occupied: Any,
        target_occupied: Any,
        conditional_log_probability: Any,
        **_: Any,
    ) -> None:
        cycle_index = int(cycle) - 1
        update_index = int(update_index)
        samples = self._cpu(sample_indices).astype(np.int64, copy=False)
        sites = self._cpu(site_ids).astype(np.int32, copy=False)
        outcomes = self._cpu(outcome_occupied).astype(np.bool_, copy=False)
        targets = self._cpu(target_occupied).astype(np.bool_, copy=False)
        logs = self._cpu(conditional_log_probability).astype(np.float64, copy=False)
        for row, sample in enumerate(samples):
            if self.next_update[sample, cycle_index] != update_index:
                raise RuntimeError("record digest received an out-of-order event")
            self.next_update[sample, cycle_index] += 1
            self.site_ids[sample, cycle_index, update_index] = sites[row]
            self.self_information[sample, cycle_index] -= float(np.sum(logs[row]))
            self.signed_transfer[sample, cycle_index] += int(
                np.sum(targets[row].astype(np.int8) - outcomes[row].astype(np.int8))
            )
            packed = np.packbits(
                np.concatenate((outcomes[row], targets[row])).astype(np.uint8),
                bitorder="little",
            )
            self._digests[sample].update(
                np.asarray([cycle, update_index, sites[row]], dtype=np.int64).tobytes()
            )
            self._digests[sample].update(packed.tobytes())
            self._digests[sample].update(np.ascontiguousarray(logs[row]).view(np.uint8))

    def save(self, path: Path | str) -> dict[str, Any]:
        if not np.all(self.next_update == self.sites_per_cycle):
            raise RuntimeError("record digest is incomplete")
        for sample in range(self.samples):
            for cycle in range(self.cycles):
                if not np.array_equal(
                    np.sort(self.site_ids[sample, cycle]), self.expected_site_ids
                ):
                    raise ValueError("record digest saw an incomplete random permutation")
        digest = np.asarray(
            [np.frombuffer(item.digest(), dtype=np.uint8) for item in self._digests],
            dtype=np.uint8,
        )
        path = Path(path)
        save_npz_atomic(
            path,
            schema=np.asarray(DIGEST_SCHEMA),
            record_sha256=digest,
            self_information_per_cycle=self.self_information,
            cumulative_self_information=np.cumsum(self.self_information, axis=1),
            signed_transfer_per_cycle=self.signed_transfer,
            sites_per_cycle=np.asarray(self.sites_per_cycle, dtype=np.int64),
            storage_contract=np.asarray("cycle_aggregates+record_digest;not_replay_capable"),
        )
        return {
            "schema": DIGEST_SCHEMA,
            "samples": self.samples,
            "cycles": self.cycles,
            "path": str(path),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "replay_capable": False,
        }
