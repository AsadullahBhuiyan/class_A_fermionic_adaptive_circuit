from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


SHARED = (
    Path(__file__).resolve().parents[1]
    / "00_WORKSPACE"
    / "CURRENT"
    / "final_production_ready_figure_scripts"
    / "_shared_src"
)
sys.path.insert(0, str(SHARED))

from record_observables import (  # noqa: E402
    OrderedBornRecordDigestWriter,
    OrderedBornRecordWriter,
    load_ordered_record,
)


def _emit(writer, cycle: int, update: int, site: int) -> None:
    outcome = np.asarray([[False, True, False, True]], dtype=np.bool_)
    target = np.asarray([[True, True, False, False]], dtype=np.bool_)
    probability = np.asarray([[0.4, 0.7, 0.6, 0.3]], dtype=np.float64)
    writer(
        cycle=cycle,
        update_index=update,
        site_ids=np.asarray([site]),
        sample_indices=np.asarray([0]),
        channel_labels=("Ap", "Am", "Bp", "Bm"),
        outcome_occupied=outcome,
        target_occupied=target,
        transfer=target.astype(np.int8) - outcome.astype(np.int8),
        conditional_log_probability=np.log(probability),
        realized_probability=probability,
        reset_covariance=np.zeros_like(probability),
    )


def test_compact_replay_record_omits_event_float_arrays(tmp_path: Path) -> None:
    writer = OrderedBornRecordWriter(
        samples=1, cycles=1, sites_per_cycle=2, expected_site_ids=[0, 1]
    )
    _emit(writer, 1, 0, 1)
    _emit(writer, 1, 1, 0)
    path = tmp_path / "record.npz"
    writer.save(path)
    with np.load(path, allow_pickle=False) as data:
        assert "conditional_log_probability" not in data.files
        assert "realized_probability" not in data.files
        assert "reset_covariance" not in data.files
    loaded = load_ordered_record(path)
    assert loaded["site_ids"].tolist() == [[[1, 0]]]
    np.testing.assert_array_equal(loaded["outcomes"], writer.outcomes)
    np.testing.assert_array_equal(loaded["transfers"], writer.transfers)
    assert np.isfinite(loaded["total_log_probability"]).all()


def test_digest_record_is_small_and_not_replay_capable(tmp_path: Path) -> None:
    writer = OrderedBornRecordDigestWriter(
        samples=1, cycles=1, sites_per_cycle=2, expected_site_ids=[0, 1]
    )
    _emit(writer, 1, 0, 1)
    _emit(writer, 1, 1, 0)
    path = tmp_path / "digest.npz"
    result = writer.save(path)
    assert result["replay_capable"] is False
    with np.load(path, allow_pickle=False) as data:
        assert set(data.files) == {
            "schema",
            "record_sha256",
            "self_information_per_cycle",
            "cumulative_self_information",
            "signed_transfer_per_cycle",
            "sites_per_cycle",
            "storage_contract",
        }
