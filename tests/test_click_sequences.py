from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fgtn.diagnostics import (
    RegionMasks,
    TrajectoryActivityRecorder,
    activity_record_frames,
    analyze_click_sequences,
    encode_click_motifs,
)
from fgtn.diagnostics.activity import (
    CLICK_SIGNS,
    _motif_grid,
    _permute_spatial_grid,
    _permute_temporal_motifs,
)


def _all_regions(nx: int, ny: int) -> RegionMasks:
    active = np.ones((nx, ny), dtype=bool)
    return RegionMasks(
        names=("all",),
        masks=active[None, ...],
        active=active,
        interface_width=1,
        wall_x=(),
    )


def _recorder_from_motifs(motifs: np.ndarray) -> TrajectoryActivityRecorder:
    motifs = np.asarray(motifs, dtype=np.uint8)
    samples, cycles, sites = motifs.shape
    recorder = TrajectoryActivityRecorder(
        nx=1,
        ny=sites,
        cycles=cycles,
        samples=samples,
        site_ids=np.arange(sites),
    )
    bits = ((motifs[..., None] >> np.arange(4, dtype=np.uint8)) & 1).astype(bool)
    recorder.transfer[:] = bits.astype(np.int8) * CLICK_SIGNS
    recorder.defect[:] = bits.astype(np.uint8)
    recorder.valid[:] = True
    recorder.success_probability[:] = 0.5
    recorder.visit_order[:] = np.arange(sites, dtype=np.int64)[None, None, :]
    recorder._visit_count[:] = sites
    return recorder


def test_all_sixteen_motifs_and_simultaneous_clicks_round_trip():
    transfer = np.zeros((16, 1, 1, 4), dtype=np.int8)
    for code in range(16):
        for bit in range(4):
            if code & (1 << bit):
                transfer[code, 0, 0, bit] = CLICK_SIGNS[bit]

    bits, motif = encode_click_motifs(transfer)
    np.testing.assert_array_equal(motif[:, 0, 0], np.arange(16, dtype=np.uint8))
    np.testing.assert_array_equal(bits[15, 0, 0], np.ones((4,), dtype=bool))
    assert int(motif[3, 0, 0]) == 3  # A loss and A gain are both retained.


@pytest.mark.parametrize(
    ("channel", "value"),
    (("Ap", 1), ("Am", -1), ("Bp", 1), ("Bm", -1)),
)
def test_encoder_rejects_wrong_transfer_direction(channel: str, value: int):
    transfer = np.zeros((1, 4), dtype=np.int8)
    transfer[0, ("Ap", "Am", "Bp", "Bm").index(channel)] = value
    with pytest.raises(ValueError, match=channel):
        encode_click_motifs(transfer)


def test_temporal_boundaries_and_periodic_spatial_words():
    motifs = np.zeros((2, 5, 3), dtype=np.uint8)
    motifs[0, :, :] = np.asarray((1, 2, 4), dtype=np.uint8)
    recorder = _recorder_from_motifs(motifs)
    result = analyze_click_sequences(
        recorder,
        regions=_all_regions(1, 3),
        burn_in=2,
        permutations=4,
        bootstrap_samples=5,
        minimum_support=1,
        seed=17,
    )

    # Three post-burn-in cycles produce two temporal pairs per site/sample.
    assert int(np.sum(result.temporal_pair_counts[0])) == 2 * 3 * 2
    assert result.temporal_pair_counts[0, 1 * 16 + 1] == 2
    assert result.temporal_pair_counts[0, 2 * 16 + 2] == 2
    assert result.temporal_pair_counts[0, 4 * 16 + 4] == 2
    assert result.temporal_pair_counts[0, 0] == 6

    # Every periodic y ring contains 1→2, 2→4, and the wraparound 4→1.
    assert int(np.sum(result.spatial_pair_counts[0])) == 3 * 3 * 2
    assert result.spatial_pair_counts[0, 1 * 16 + 2] == 3
    assert result.spatial_pair_counts[0, 2 * 16 + 4] == 3
    assert result.spatial_pair_counts[0, 4 * 16 + 1] == 3
    assert result.spatial_pair_counts[0, 0] == 9


def test_shuffle_nulls_preserve_one_point_motif_counts():
    motifs = np.asarray(
        [
            [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]],
            [[12, 13, 14], [15, 0, 1], [2, 3, 4], [5, 6, 7]],
        ],
        dtype=np.uint8,
    )
    recorder = _recorder_from_motifs(motifs)
    rng = np.random.default_rng(123)
    temporal = _permute_temporal_motifs(motifs, rng)
    for sample in range(motifs.shape[0]):
        for site in range(motifs.shape[2]):
            np.testing.assert_array_equal(
                np.bincount(temporal[sample, :, site], minlength=16),
                np.bincount(motifs[sample, :, site], minlength=16),
            )

    grid = _motif_grid(motifs, recorder)
    spatial = _permute_spatial_grid(grid, np.random.default_rng(456))
    for sample in range(motifs.shape[0]):
        for cycle in range(motifs.shape[1]):
            np.testing.assert_array_equal(
                np.bincount(spatial[sample, cycle, 0], minlength=16),
                np.bincount(grid[sample, cycle, 0], minlength=16),
            )


def test_sequence_analysis_is_seed_deterministic():
    motifs = np.asarray(
        [
            [[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5]],
            [[5, 4, 3], [4, 3, 2], [3, 2, 1], [2, 1, 0]],
        ],
        dtype=np.uint8,
    )
    recorder = _recorder_from_motifs(motifs)
    kwargs = dict(
        regions=_all_regions(1, 3),
        burn_in=1,
        permutations=5,
        bootstrap_samples=6,
        minimum_support=1,
        seed=99,
    )
    first = analyze_click_sequences(recorder, **kwargs)
    second = analyze_click_sequences(recorder, **kwargs)
    for key, value in first.payload().items():
        other = second.payload()[key]
        if np.asarray(value).dtype.kind in "f":
            np.testing.assert_allclose(value, other, rtol=0.0, atol=0.0, equal_nan=True)
        else:
            np.testing.assert_array_equal(value, other)


def test_tidy_frames_record_pauli_eigenvectors_and_time_steps():
    recorder = _recorder_from_motifs(np.asarray([[[15, 0, 1], [2, 4, 8]]], dtype=np.uint8))
    events, motifs = activity_record_frames(
        recorder,
        regions=_all_regions(1, 3),
        schedule="raster_y",
        trial_pauli="X",
    )
    assert set(events["trial_orbital"]) == {"A", "B"}
    assert set(events.loc[events["trial_orbital"] == "A", "trial_pauli_eigenvalue"]) == {1}
    assert set(events.loc[events["trial_orbital"] == "B", "trial_pauli_eigenvalue"]) == {-1}
    assert events["cycle"].min() == 1
    assert events["global_channel_step"].is_unique
    assert set(motifs.loc[motifs["motif_code"] == 15, "motif_label"]) == {
        "A_loss+A_gain+B_loss+B_gain"
    }


def test_tidy_frames_reject_incomplete_activity():
    recorder = TrajectoryActivityRecorder(nx=1, ny=1, cycles=1, samples=1, site_ids=[0])
    with pytest.raises(RuntimeError, match="missed"):
        activity_record_frames(
            recorder,
            regions=_all_regions(1, 1),
            schedule="raster_y",
        )
