from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT))

from record_anisotropy import (
    CHANNELS,
    WallRecordObserver,
    alpha_from_match,
    connected_correlations,
    match_time,
)


def measurement_events() -> tuple[dict[str, object], ...]:
    events = []
    for index, channel in enumerate(CHANNELS):
        p_occ = 0.2 + 0.1 * index
        outcome = index % 2 == 1
        selected = p_occ if outcome else 1.0 - p_occ
        events.append(
            {
                "kind": "measurement",
                "channel": channel,
                "probability": p_occ,
                "outcome_occupied": outcome,
                "log_weight": np.log(selected),
            }
        )
    return tuple(events)


def test_observer_decodes_site_and_keeps_channel_record():
    observer = WallRecordObserver(nx=5, ny=2, cycles=1, samples=1, wall_x=(1, 3))
    for wall_x in (1, 3):
        for y in range(2):
            observer(
                cycle=1,
                site_id=wall_x + 5 * y,
                sample_index=0,
                branch_events=measurement_events(),
            )
    observer.assert_complete()
    payload = observer.payload()
    assert payload["log_realized_probability"].shape == (1, 1, 2, 2, 4)
    np.testing.assert_allclose(payload["mismatch_fraction"], 0.0)


def test_connected_correlations_keep_trajectories_as_outer_axis():
    rng = np.random.default_rng(4)
    field = rng.normal(size=(3, 12, 2, 6))
    estimate = connected_correlations(field, burn_in=2, max_temporal_lag=4)
    assert estimate.spatial_by_trajectory.shape == (3, 4)
    assert estimate.temporal_by_trajectory.shape == (3, 5)
    np.testing.assert_allclose(estimate.spatial_mean, np.mean(estimate.spatial_by_trajectory, axis=0))
    np.testing.assert_allclose(estimate.temporal_mean, np.mean(estimate.temporal_by_trajectory, axis=0))


def test_zabalo_matching_rule_recovers_known_integer_crossing():
    circumference = 12
    t_star_expected = 4.0
    alpha_expected = np.arcsinh(1.0) * circumference / (np.pi * t_star_expected)
    scaling_dimension = 0.7
    lags = np.arange(9, dtype=float)
    temporal = np.empty_like(lags)
    temporal[0] = 5.0
    temporal[1:] = np.sinh(np.pi * alpha_expected * lags[1:] / circumference) ** (
        -2.0 * scaling_dimension
    )
    spatial_target = np.sin(np.pi * 0.5) ** (-2.0 * scaling_dimension)
    matched = match_time(temporal, spatial_target)
    np.testing.assert_allclose(matched, t_star_expected, atol=1e-12)
    np.testing.assert_allclose(alpha_from_match(circumference, matched), alpha_expected, atol=1e-12)


def test_nonpositive_spatial_target_is_explicitly_unresolved():
    assert np.isnan(match_time(np.asarray([1.0, 0.5, 0.1]), -0.01))


def test_contact_to_first_lag_drop_is_not_an_anisotropy_crossing():
    temporal = np.asarray([2.0, 0.01, 0.02, -0.01])
    assert np.isnan(match_time(temporal, 0.1))
