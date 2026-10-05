from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    REPO_ROOT
    / "00_WORKSPACE/CURRENT/final_production_new_designs"
    / "13_maxmix_manybody_lyapunov_4ny/analyze_campaign.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("bundle13_analysis", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def analysis():
    return load_module()


def test_heap_levels_match_exhaustive_fock_products(analysis):
    occupations = np.asarray([0.2, 0.35, 0.6, 0.8])
    factors = []
    for mask in range(1 << occupations.size):
        log_value = 0.0
        for index, occupation in enumerate(occupations):
            log_value += math.log(occupation if mask & (1 << index) else 1.0 - occupation)
        factors.append(log_value)
    expected = np.sort(factors)[::-1]
    actual = analysis.leading_levels(occupations, 0.0, count=expected.size)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2e-15)


def test_exact_caps_remain_infinite(analysis):
    levels = analysis.leading_levels(np.asarray([0.0, 1.0, 0.25]), 0.0, count=8)
    assert np.count_nonzero(np.isfinite(levels)) == 2
    assert np.all(np.isneginf(levels[2:]))


def test_synthetic_finite_size_sign_and_coefficients(analysis):
    ny = np.asarray(analysis.NY_VALUES, dtype=float)
    x = 1.0 / ny**2
    f_inf, a0 = 2.5, -0.7
    leading = -ny * (f_inf + a0 * x)
    fit = analysis.fit_line(x, -leading / ny)
    assert fit["coefficient"] == pytest.approx(a0, abs=1e-11)
    ai = 3.2
    gap_density = ai * x
    gap_fit = analysis.fit_line(x, gap_density, intercept=False)
    assert gap_fit["coefficient"] == pytest.approx(ai, abs=1e-11)
    assert -6 * fit["coefficient"] / math.pi > 0


def test_whole_trajectory_bootstrap_is_deterministic(analysis):
    values = np.linspace(-1, 1, 100)
    first = analysis.bootstrap_mean_ci(values, 2000, np.random.default_rng(analysis.BOOTSTRAP_SEED))
    second = analysis.bootstrap_mean_ci(values, 2000, np.random.default_rng(analysis.BOOTSTRAP_SEED))
    assert first == second


def test_real_campaign_integration(tmp_path, analysis):
    output = tmp_path / "analysis"
    summary = analysis.run_analysis(
        output_root=output,
        bootstrap_count=100,
        verify_hashes=True,
        subset_repeats=3,
        build_document=False,
    )
    assert summary["provenance"]["verified_result_pairs"] == 140
    assert summary["provenance"]["verified_trajectories"] == 700
    assert summary["verification"]["roundoff_spectrum_reconstruction"] is True
    assert np.isfinite(summary["reported_estimates"]["alpha_c_eff"])
    assert len(summary["reported_estimates"]["r_i"]) == 4
    assert len((output / "trajectory_window_slopes.csv").read_text().splitlines()) == 2101
    assert len((output / "purification_times.csv").read_text().splitlines()) == 2101
    payload = json.loads((output / "analysis_summary.json").read_text())
    assert payload["schema"] == analysis.ANALYSIS_SCHEMA
    assert (output / "figure_assets/dynamical_critical_main.png").stat().st_size > 100_000
    assert (output / "figure_assets/dynamical_critical_diagnostics.png").stat().st_size > 100_000


def test_reader_document_is_single_pdf(analysis):
    output = analysis.DEFAULT_OUTPUT_ROOT
    pdfs = list(output.rglob("*.pdf"))
    assert pdfs == [output / "dynamical_critical_analysis.pdf"]
    assert pdfs[0].stat().st_size > 100_000
