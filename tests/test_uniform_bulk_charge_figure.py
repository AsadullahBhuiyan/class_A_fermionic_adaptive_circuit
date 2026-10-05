from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_ROOT = (
    REPOSITORY_ROOT
    / "00_WORKSPACE"
    / "CURRENT"
    / "experiment_review"
    / "uniform_bulk_validation_analysis"
)
SCRIPT_PATH = ANALYSIS_ROOT / "build_uniform_bulk_charge_fluctuation_figure.py"


@pytest.fixture(scope="module")
def charge_figure_module():
    module_name = "uniform_bulk_charge_figure_test_module"
    specification = importlib.util.spec_from_file_location(module_name, SCRIPT_PATH)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.path.insert(0, str(ANALYSIS_ROOT))
    try:
        specification.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def test_mean_sem_uses_sample_standard_error(charge_figure_module) -> None:
    values = np.arange(100, dtype=np.float64)[:, None]
    mean, sem = charge_figure_module._mean_sem(values)
    assert mean[0] == pytest.approx(values.mean())
    assert sem[0] == pytest.approx(values.std(ddof=1) / np.sqrt(100.0))


def test_charge_figure_statistics_match_completed_campaigns(
    charge_figure_module,
) -> None:
    cases, verification = charge_figure_module._load_charge_cases()
    assert len(verification) == 33
    assert len(cases) == 18
    assert sum(values.shape[0] for values in cases.values()) == 1_800

    size_summary, cycle_summary, histogram = charge_figure_module._summarize(cases)

    offsets = cases[(32, "1")]
    trajectory_late_time = (
        100.0 * np.abs(offsets[:, 21:41]) / float(32**2)
    ).mean(axis=1)
    row = size_summary.loc[
        (size_summary["L"] == 32) & (size_summary["n_shell"] == "1")
    ].iloc[0]
    assert float(row["mean"]) == pytest.approx(trajectory_late_time.mean())
    assert float(row["sem"]) == pytest.approx(
        trajectory_late_time.std(ddof=1) / np.sqrt(100.0)
    )

    filling_deviation = np.abs(offsets) / float(2 * 32**2)
    cycle_40 = cycle_summary.loc[
        (cycle_summary["n_shell"] == "1") & (cycle_summary["cycle"] == 40)
    ].iloc[0]
    assert float(cycle_40["mean"]) == pytest.approx(
        filling_deviation[:, 40].mean()
    )
    assert float(cycle_40["sem"]) == pytest.approx(
        filling_deviation[:, 40].std(ddof=1) / np.sqrt(100.0)
    )

    observed_histogram = {
        int(row.delta_Q): int(row.count) for row in histogram.itertuples()
    }
    assert observed_histogram == {-3: 2, -2: 9, -1: 19, 0: 34, 1: 13, 2: 17, 3: 5, 4: 1}
    assert sum(observed_histogram.values()) == 100


def test_charge_figure_source_contains_no_bootstrap_contract() -> None:
    source = SCRIPT_PATH.read_text(encoding="utf-8").lower()
    assert "bootstrap" not in source
