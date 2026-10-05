from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest


REPO = Path(__file__).resolve().parents[1]
BUNDLE = REPO / "00_WORKSPACE/CURRENT/final_production_new_designs/04_maxmix_manybody_lyapunov_pilot"
DATA = BUNDLE / "gpu_data/maxmix_manybody_lyapunov_nx20_ny20to40_s100_gpu_v2"


def _load_analysis():
    old_path = list(sys.path)
    old_runner = sys.modules.get("run_campaign")
    old_observer = sys.modules.get("lyapunov_observer")
    try:
        sys.path.insert(0, str(BUNDLE))
        spec = importlib.util.spec_from_file_location("tested_v2_lyapunov_analysis", BUNDLE / "analyze_campaign.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path[:] = old_path
        if old_runner is None:
            sys.modules.pop("run_campaign", None)
        else:
            sys.modules["run_campaign"] = old_runner
        if old_observer is None:
            sys.modules.pop("lyapunov_observer", None)
        else:
            sys.modules["lyapunov_observer"] = old_observer


ANALYSIS = _load_analysis()


def test_known_finite_size_coefficients_have_the_zabalo_signs() -> None:
    ny = np.asarray([20, 22, 24, 26, 28, 30, 36, 40], dtype=float)
    a0 = -2.25
    ai = np.asarray([1.5, 2.5, 3.5, 4.5])
    f0 = 3.0 + a0 / ny**2
    lambda0 = -ny * f0
    levels = [lambda0]
    for coefficient in ai:
        levels.append(lambda0 - coefficient / ny)
    fit = ANALYSIS.fit_finite_size(ny, np.column_stack(levels))
    assert fit["primary"]["A0"] == pytest.approx(a0, abs=1e-10)
    assert fit["primary"]["alpha_c_eff"] == pytest.approx(-6 * a0 / math.pi)
    for expected, row in zip(ai, fit["gaps"]):
        assert row["Ai"] == pytest.approx(expected, abs=1e-10)
        assert row["alpha_x_typ"] == pytest.approx(expected / (2 * math.pi))
        assert row["x_typ_over_c_eff"] == pytest.approx(-expected / (12 * a0))


def test_window_and_paired_bootstrap_use_one_row_per_trajectory() -> None:
    cycles = np.arange(0, 41, 4)
    slopes = np.asarray([-3.0, -3.1, -3.2, -3.3, -3.4])
    levels = 2.0 + cycles[:, None] * slopes
    middle, late = ANALYSIS.trajectory_window_slopes(cycles, levels, ny=20)
    np.testing.assert_allclose(middle, slopes)
    np.testing.assert_allclose(late, slopes)
    trajectories = np.stack([slopes + index * 0.01 for index in range(12)])
    rows = ANALYSIS.paired_convergence(
        trajectories,
        trajectories.copy(),
        bootstrap_count=100,
        rng=np.random.default_rng(4),
        relative_threshold=0.1,
    )
    assert len(rows) == 5
    assert all(row["passes"] for row in rows)
    assert all(row["shift"] == pytest.approx(0.0) for row in rows)


def test_random_subset_sample_convergence_is_deterministic() -> None:
    by_ny = {}
    for ny in (20, 22, 24, 26):
        rows = np.empty((100, 5))
        rows[:, 0] = -(2.0 * ny - 1.0 / ny) + np.linspace(-0.1, 0.1, 100)
        for level in range(1, 5):
            rows[:, level] = rows[:, 0] - level / ny
        by_ny[ny] = {"late_slopes": rows}
    first = ANALYSIS.sample_convergence(
        by_ny, counts=(25, 100), repeats=20, rng=np.random.default_rng(9)
    )
    second = ANALYSIS.sample_convergence(
        by_ny, counts=(25, 100), repeats=20, rng=np.random.default_rng(9)
    )
    assert first == second
    assert {row["selection"] for row in first} == {
        "random_without_replacement",
        "full_ensemble",
    }


@pytest.mark.skipif(not (DATA / "DOWNLOAD_MANIFEST.json").is_file(), reason="downloaded v2 data are not present")
def test_real_v2_campaign_uses_pinned_provenance_and_fails_claim_gates(tmp_path: Path) -> None:
    config = json.loads((DATA / "campaign_config.v2.json").read_text(encoding="utf-8"))
    output = tmp_path / "analysis"
    assert ANALYSIS.main(
        [
            "--config", str(DATA / "campaign_config.v2.json"),
            "--results-root", str(DATA),
            "--output-root", str(output),
            "--bootstrap-count", "24",
        ]
    ) == 0
    summary = json.loads((output / "analysis_summary.json").read_text(encoding="utf-8"))
    assert summary["verified_tasks"] == 160
    assert summary["verified_files"] == 320
    assert summary["verified_trajectories"] == 800
    assert summary["historical_provenance"]["configuration_sha256"] == ANALYSIS.config_hash(config)
    assert summary["historical_provenance"]["historical_manifest_used"] is True
    assert summary["historical_provenance"]["executed_engine_sha256"] != summary["historical_provenance"]["current_canonical_engine_sha256"]
    decisions = summary["acceptance_decisions"]
    assert decisions["spectral_reconstruction_validated"] is True
    assert decisions["boundary_localization_validated"] is True
    assert decisions["T_2Ny_temporally_sufficient"] is False
    assert decisions["alpha_c_eff_reportable"] is False
    assert decisions["precision_cft_claim_supported"] is False
    assert len(list(output.glob("*.csv"))) == 9
    assert (output / "many_body_lyapunov_v2_diagnostic.pdf").is_file()
    assert (output / "many_body_lyapunov_v2_supplementary.png").is_file()
