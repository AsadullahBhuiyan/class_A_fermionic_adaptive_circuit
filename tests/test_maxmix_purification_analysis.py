from __future__ import annotations

import importlib.util
import itertools
import math
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    REPO_ROOT
    / "00_WORKSPACE"
    / "CURRENT"
    / "final_production_new_designs"
    / "07_maxmix_hard_soft_purification"
    / "analyze_completed_campaign.py"
)
SPEC = importlib.util.spec_from_file_location("maxmix_purification_analysis", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_lowest_subset_sums_match_exhaustive_products() -> None:
    costs = np.asarray([0.0, 0.2, 0.7, 1.1, 1.8, 2.0])
    exhaustive = sorted(
        sum(cost for cost, chosen in zip(costs, bits) if chosen)
        for bits in itertools.product((False, True), repeat=costs.size)
    )
    actual = MODULE.lowest_subset_sums(costs, 32)
    assert np.allclose(actual, exhaustive[:32], rtol=0.0, atol=1.0e-14)


def test_identity_spectrum_and_exact_caps_have_correct_levels() -> None:
    identity = np.full(8, 0.5)
    levels = MODULE.leading_log_sigma2_levels(
        identity, 8 * math.log(2.0), count=16
    )
    assert np.allclose(levels, 0.0, rtol=0.0, atol=1.0e-14)

    with_caps = np.asarray([0.0, 1.0, 0.25, 0.75])
    levels = MODULE.leading_log_sigma2_levels(
        with_caps, math.log(4.0), count=8
    )
    assert np.isfinite(levels[:4]).all()
    assert np.isneginf(levels[4:]).all()


def test_known_finite_size_coefficients_and_signs() -> None:
    ny = np.asarray([20.0, 30.0, 40.0])
    f_inf = 3.2
    a0 = -1.7
    ai = np.asarray([2.0, 3.0, 4.0, 5.0])
    lambda0 = -ny * (f_inf + a0 / ny**2)
    means = np.empty((ny.size, 5))
    means[:, 0] = lambda0
    for level in range(1, 5):
        gap = ai[level - 1] / ny
        means[:, level] = lambda0 - gap
    fit = MODULE._finite_fit_from_means(ny, means)
    assert math.isclose(fit["primary"]["A0"], a0, rel_tol=0.0, abs_tol=1.0e-10)
    assert math.isclose(
        fit["primary"]["alpha_c_eff"], -6.0 * a0 / math.pi, rel_tol=0.0, abs_tol=1.0e-10
    )
    for level, expected in enumerate(ai, start=1):
        assert math.isclose(
            fit["gaps"][level - 1]["Ai"], expected, rel_tol=0.0, abs_tol=1.0e-10
        )


def test_window_slopes_are_computed_per_trajectory() -> None:
    arrays = {}
    for construction in ("hard", "soft"):
        for ny in MODULE.NY_VALUES:
            cycles = np.arange(4 * ny + 1, dtype=np.float64)
            levels = np.empty((100, cycles.size, 64))
            for sample in range(100):
                lambda0 = -2.0 - sample * 1.0e-3
                levels[sample, :, 0] = lambda0 * cycles
                for level in range(1, 64):
                    levels[sample, :, level] = (lambda0 - 0.1 * level) * cycles
            arrays[(construction, ny)] = {
                "levels": levels,
                "omega": levels[:, :, 0].copy(),
            }
    rows, fitted = MODULE.trajectory_slopes(arrays)
    assert len(rows) == 2 * 3 * 100 * 3
    values = fitted[("hard", 20)]["W3_3Ny_to_4Ny_levels"]
    assert values.shape == (100, 5)
    assert np.allclose(values[:, 0], -2.0 - np.arange(100) * 1.0e-3)
    assert np.allclose(values[:, 0] - values[:, 4], 0.4)


def test_campaign_specs_pin_distinct_historical_identities() -> None:
    hard, soft = MODULE.CAMPAIGNS
    assert hard.construction == "hard" and soft.construction == "soft"
    assert hard.revision.endswith("_v2") and soft.revision.endswith("_v3")
    assert hard.configuration_hash != soft.configuration_hash
    assert hard.source_hashes["purification_observer.py"] == soft.source_hashes["purification_observer.py"]
    assert hard.source_hashes["run_campaign.py"] != soft.source_hashes["run_campaign.py"]


def test_legacy_2ny_comparison_is_checksum_pinned_and_never_pooled() -> None:
    current = []
    for ny in MODULE.NY_VALUES:
        for metric in ("lambda0", "gap1", "gap2", "gap3", "gap4"):
            current.append(
                {
                    "construction": "hard",
                    "Ny": ny,
                    "metric": metric,
                    "earlier_window": "W2_2Ny_to_3Ny",
                    "later_window": "W3_3Ny_to_4Ny",
                    "resolved_fraction": 1.0,
                    "relative_shift": 0.01,
                    "shift_ci_low": -0.02,
                    "shift_ci_high": 0.02,
                    "shift_ci_contains_zero": True,
                    "passes": True,
                }
            )
    rows, summary = MODULE.legacy_depth_comparison(current)
    assert len(rows) == 30
    assert all(row.get("pooled_with_4Ny") is False for row in rows if row["depth_multiple"] == 2)
    assert all(row.get("pooled_with_2Ny") is False for row in rows if row["depth_multiple"] == 4)
    assert summary["lambda0_pass_by_depth"]["2"] == {
        "20": False,
        "30": False,
        "40": True,
    }
    assert summary["lambda0_pass_by_depth"]["4"] == {
        "20": True,
        "30": True,
        "40": True,
    }
