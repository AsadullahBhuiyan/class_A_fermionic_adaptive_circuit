from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest


REPO = Path(__file__).resolve().parents[1]
SCRIPT = (
    REPO
    / "00_WORKSPACE/CURRENT/experiment_review/entropy_charge_endpoint_sample_resolved"
    / "make_contour_scaling_figures.py"
)


def _module():
    name = "_tested_endpoint_mean_curve_figures"
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


FIGURES = _module()


class _Analyzer:
    CURVE_KEYS = {"c1": "s1", "c2": "s2", "c3": "s3", "k": "variance"}
    PREFACTOR = {"c1": 3.0, "c2": 4.0, "c3": 4.5, "k": math.pi**2}

    @staticmethod
    def log_chord(ay: np.ndarray, ny: int) -> np.ndarray:
        ay = np.asarray(ay, dtype=np.float64)
        return np.log((ny / np.pi) * np.sin(np.pi * ay / ny))


def _synthetic_cases(samples: int = 24) -> dict[int, dict[str, np.ndarray]]:
    rng = np.random.default_rng(9012)
    cases: dict[int, dict[str, np.ndarray]] = {}
    factors = {"s1": 1 / 3, "s2": 1 / 4, "s3": 2 / 9, "variance": 1 / np.pi**2}
    for size_index, ny in enumerate(FIGURES.NY_VALUES):
        ay = np.arange(ny // 2 + 1, dtype=np.int64)
        x = np.zeros_like(ay, dtype=np.float64)
        x[1:] = _Analyzer.log_chord(ay[1:], ny)
        common = rng.normal(0.0, 0.012, size=samples)
        case: dict[str, np.ndarray] = {
            "ay_values": ay,
            "sample_ids": np.arange(samples, dtype=np.int64),
        }
        for observable_index, (key, target) in enumerate(factors.items()):
            slope = target * (1.0 + 0.004 * size_index + common)
            slope += rng.normal(0.0, 0.0006 * (observable_index + 1), size=samples)
            intercept = 2.0 + 0.2 * observable_index + rng.normal(0.0, 0.03, size=samples)
            curves = slope[:, None] * x[None, :] + intercept[:, None]
            curves[:, 0] = 0.0
            case[key] = curves
        cases[ny] = case
    return cases


def test_mean_curve_fit_uses_full_covariance_and_equals_mean_slope() -> None:
    rng = np.random.default_rng(778)
    x = np.linspace(-1.2, 0.9, 9)
    common = rng.normal(size=(80, 1))
    correlated_noise = 0.03 * common * np.linspace(-0.7, 1.1, x.size)[None, :]
    curves = (0.37 + 0.02 * common) * x + 1.4 + correlated_noise

    result = FIGURES.mean_curve_fit(x, curves)
    design = np.column_stack((x, np.ones_like(x)))
    projection = np.linalg.solve(design.T @ design, design.T)[0]
    sample_slopes = curves @ projection

    assert result["slope"] == pytest.approx(sample_slopes.mean(), abs=2e-13)
    assert result["slope_sem"] == pytest.approx(
        sample_slopes.std(ddof=1) / math.sqrt(sample_slopes.size), abs=2e-14
    )
    diagonal_only = math.sqrt(
        float(np.sum(projection**2 * np.diag(np.cov(curves, rowvar=False, ddof=1))))
        / curves.shape[0]
    )
    assert not np.isclose(result["slope_sem"], diagonal_only, rtol=1e-3)


def test_anchored_mean_fit_handles_even_and_odd_sizes_and_equal_size_weights() -> None:
    cases = _synthetic_cases()
    by_size, summary = FIGURES.anchored_mean_curve_fit(_Analyzer, cases, "c1")

    numerator = 0.0
    denominator = 0.0
    for ny in FIGURES.NY_VALUES:
        item = by_size[ny]
        assert item["anchor_Ay"] == ny // 2
        assert item["mean_all"][-1] == pytest.approx(0.0, abs=2e-14)
        assert item["sem_all"][-1] == pytest.approx(0.0, abs=2e-14)
        x = item["x_fit"]
        y = item["delta_fit_samples"].mean(axis=0)
        weight = 1.0 / x.size
        numerator += weight * float(x @ y)
        denominator += weight * float(x @ x)

    assert summary["slope"] == pytest.approx(numerator / denominator, abs=2e-14)
    assert summary["slope_covariance_SEM"] > 0.0
    assert by_size[35]["anchor_Ay"] == 17
    assert by_size[40]["anchor_Ay"] == 20


def test_entropy_charge_ratio_uses_joint_trajectory_covariance() -> None:
    cases = _synthetic_cases()
    fits, _, _ = FIGURES.mean_curve_results(_Analyzer, cases)
    rows = FIGURES.mean_curve_ratio_rows(_Analyzer, cases, fits)
    row = next(item for item in rows if item["Ny"] == 60 and item["renyi_order"] == 1)

    assert row["mean_curve_coefficient_ratio"] == pytest.approx(
        fits[60]["c1"]["converted"] / fits[60]["k"]["converted"]
    )
    assert row["joint_covariance_SEM"] > 0.0
    assert row["uncertainty_method"] == "delta_method_from_joint_trajectory_curve_covariance"


def test_canonical_manifest_is_mean_first_and_pdf_sizes_are_locked() -> None:
    import json
    import subprocess

    manifest_path = SCRIPT.parent / "contour_scaling_figure_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema"] == "endpoint_contour_scaling_figures_v5_ensemble_mean_first"
    assert manifest["estimator_order"] == "average_100_trajectory_curves_at_fixed_Ny_then_fit"
    assert "representative" not in " ".join(manifest["figure_layouts"]).lower()
    assert manifest["linearity_audit"]["fit_mean_minus_mean_fit_max_abs"] < 3e-13

    expected = {
        "endpoint_entropy_mean_curve_size_collapse_3x1.pdf": (3.375, 7.05),
        "endpoint_charge_variance_mean_curve_size_collapse_1x1.pdf": (3.375, 2.65),
        "endpoint_mean_curve_cq_k_scaling_1x1.pdf": (3.375, 2.65),
    }
    for filename, inches in expected.items():
        completed = subprocess.run(
            ["pdfinfo", str(SCRIPT.parent / "figures" / filename)],
            check=True,
            capture_output=True,
            text=True,
        )
        page_line = next(line for line in completed.stdout.splitlines() if line.startswith("Page size:"))
        width_points, height_points = [float(value) for value in page_line.split()[2:5:2]]
        assert width_points / 72 == pytest.approx(inches[0], abs=0.002)
        assert height_points / 72 == pytest.approx(inches[1], abs=0.002)
