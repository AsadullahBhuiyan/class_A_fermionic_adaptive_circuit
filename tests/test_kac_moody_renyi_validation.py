from __future__ import annotations

import csv
import hashlib
import importlib.util
import itertools
import json
import re
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
PROJECT = (
    ROOT
    / "00_WORKSPACE/CURRENT/Paper Methods/kac_moody_renyi_validation"
)
BUILDER = PROJECT / "build_validation_figures.py"
NOTE = PROJECT / "kac_moody_renyi_validation.tex"
CURATED_KEY = "20_THEORY_AND_METHODS/27_KAC_MOODY_RENYI_VALIDATION"


def _load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def analysis() -> ModuleType:
    assert BUILDER.is_file()
    return _load_module(BUILDER, "kac_moody_renyi_validation_builder")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_note_structure_outputs_and_curated_registration() -> None:
    required_source_files = (
        NOTE,
        PROJECT / "README.md",
        PROJECT / "references.bib",
        BUILDER,
        PROJECT / "analysis_manifest.json",
    )
    for path in required_source_files:
        assert path.is_file(), path

    compiled_candidates = (
        PROJECT / "kac_moody_renyi_validation.pdf",
        PROJECT / "build/kac_moody_renyi_validation.pdf",
    )
    assert any(path.is_file() and path.stat().st_size > 0 for path in compiled_candidates)

    for stem in (
        "figure_01_geometry_pipeline",
        "figure_02_modular_kernels",
        "figure_03_exact_b0_validation",
        "figure_04_stochastic_legacy_validation",
    ):
        for suffix in (".pdf", ".png"):
            path = PROJECT / "figures" / f"{stem}{suffix}"
            assert path.is_file() and path.stat().st_size > 0, path

    for filename in (
        "exact_b0_regression.csv",
        "exact_b0_sensitivity.csv",
        "result21_regression.csv",
        "result21_sensitivity_summary.csv",
        "result21_paired_fits.csv",
        "prior_production_regression.csv",
        "prior_production_sensitivity_summary.csv",
        "scientific_claim_status.json",
        "validation_summary.json",
    ):
        assert (PROJECT / "tables" / filename).is_file(), filename

    source = NOTE.read_text(encoding="utf-8")
    assert (
        r"\documentclass[aps,prb,onecolumn,nofootinbib,superscriptaddress]{revtex4-2}"
        in source
    )
    assert "Kac--Moody Level and R\\'enyi Entanglement" in source
    for section in (
        "Gaussian reduced states and interval entropies",
        "Charge statistics and measurement-record averages",
        "Replica derivation of the interval entropy",
        "The physical current and its Kac--Moody level",
        "Why central charge and current level are distinct",
        "A common susceptibility spectrum",
        "What can be inferred about individual walls?",
    ):
        assert rf"\section{{{section}}}" in source
    for required_text in (
        r"G_{\boldsymbol m,A}",
        r"\label{eq:transpose}",
        r"\label{eq:twist-weight}",
        r"\label{eq:kernel-integrals}",
        r"\label{eq:contour-sum}",
        "annealed replica moment",
        "entropy of the Born mixture",
        "record-to-record charge wandering",
        "Static entropy and charge",
    ):
        assert required_text in " ".join(source.split())
    assert r"\tableofcontents" in source
    # Historical numerical products remain covered above and below, but the
    # pedagogical document must compile without loading any of them.
    assert not re.search(r"\\(?:input|include|includegraphics)\b", source)
    for removed_section in (
        "Numerical validation ladder",
        "Acceptance criteria and failure semantics",
        "Existing evidence and provenance audit",
    ):
        assert removed_section not in source
    assert not re.search(r"\\mathbbm?\{?(?:1|I)\}?", source)

    notes_builder = _load_module(
        ROOT / "scripts/build_notes_view.py", "build_notes_view_for_km_test"
    )
    assert notes_builder.CURATED[CURATED_KEY].resolve() == PROJECT.resolve()


def test_gaussian_entropy_fcs_and_coefficients_against_fock_space(
    analysis: ModuleType,
) -> None:
    eigenvalues = np.asarray([0.17, 0.41, 0.73], dtype=float)
    probabilities = []
    charges = []
    for occupations in itertools.product((0, 1), repeat=eigenvalues.size):
        occupations_array = np.asarray(occupations, dtype=int)
        probability = np.prod(
            np.where(occupations_array, eigenvalues, 1.0 - eigenvalues)
        )
        probabilities.append(float(probability))
        charges.append(int(np.sum(occupations_array)))
    probabilities_array = np.asarray(probabilities)
    charges_array = np.asarray(charges, dtype=float)
    assert np.sum(probabilities_array) == pytest.approx(1.0, abs=1e-15)

    direct_entropy_1 = -np.sum(probabilities_array * np.log(probabilities_array))
    assert analysis.entropy_renyi(eigenvalues, 1) == pytest.approx(
        direct_entropy_1, abs=1e-12
    )
    for order in (2, 3):
        direct = np.log(np.sum(probabilities_array**order)) / (1 - order)
        assert analysis.entropy_renyi(eigenvalues, order) == pytest.approx(
            direct, abs=1e-12
        )

    counting_field = 0.713
    direct_characteristic = np.sum(
        probabilities_array * np.exp(1j * counting_field * charges_array)
    )
    determinant_characteristic = np.prod(
        1 - eigenvalues + np.exp(1j * counting_field) * eigenvalues
    )
    assert determinant_characteristic == pytest.approx(
        direct_characteristic, abs=1e-12
    )

    rng = np.random.default_rng(1729)
    matrix = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    unitary, _ = np.linalg.qr(matrix)
    correlation = (unitary * eigenvalues) @ unitary.conj().T
    identity = np.eye(correlation.shape[0], dtype=np.complex128)
    matrix_characteristic = np.linalg.det(
        identity - correlation + np.exp(1j * counting_field) * correlation
    )
    assert abs(matrix_characteristic - determinant_characteristic) <= 1e-10
    assert abs(matrix_characteristic - direct_characteristic) <= 1e-10

    mean = np.sum(probabilities_array * charges_array)
    centered = charges_array - mean
    direct_cumulants = {
        "kappa_1": mean,
        "kappa_2": np.sum(probabilities_array * centered**2),
        "kappa_3": np.sum(probabilities_array * centered**3),
        "kappa_4": np.sum(probabilities_array * centered**4)
        - 3 * np.sum(probabilities_array * centered**2) ** 2,
    }
    calculated_cumulants = analysis.charge_cumulants(eigenvalues)
    for key, value in direct_cumulants.items():
        assert calculated_cumulants[key] == pytest.approx(value, abs=1e-12)

    susceptibility = correlation @ (identity - correlation)
    trace_cumulants = {
        "kappa_1": np.trace(correlation).real,
        "kappa_2": np.trace(susceptibility).real,
        "kappa_3": np.trace(
            susceptibility @ (identity - 2 * correlation)
        ).real,
        "kappa_4": np.trace(
            susceptibility @ (identity - 6 * susceptibility)
        ).real,
    }
    for key, value in trace_cumulants.items():
        assert abs(calculated_cumulants[key] - value) <= 1e-10

    for order in (1, 2, 3):
        charge_slope = 1 / np.pi**2
        entropy_slope = (1 + 1 / order) / 6
        coefficients = analysis.coefficient_estimates(
            entropy_slope, charge_slope, order
        )
        assert coefficients["c_wall"] == pytest.approx(1.0, abs=1e-14)
        assert coefficients["k_wall"] == pytest.approx(1.0, abs=1e-14)
        assert coefficients["delta_n"] == pytest.approx(0.0, abs=1e-14)
        assert coefficients["normalized_delta"] == pytest.approx(0.0, abs=1e-14)

    for order in (1, 2, 3):
        assert analysis.entropy_renyi(np.asarray([0.0, 1.0]), order) == 0.0
        near_endpoint_entropy = analysis.entropy_renyi(
            np.asarray([1e-15, 1 - 1e-15]), order
        )
        assert np.isfinite(near_endpoint_entropy)
        assert 0 <= near_endpoint_entropy < 1e-9
    with pytest.raises(ValueError, match=r"outside \[0,1\]"):
        analysis.entropy_renyi(np.asarray([1.1]), 2)

    modular_energy = np.linspace(-10, 10, 101)
    raw_kernel = analysis.susceptibility_kernel(modular_energy)
    expected_kernel = 1 / (4 * np.cosh(modular_energy / 2) ** 2)
    np.testing.assert_allclose(raw_kernel, expected_kernel, atol=1e-15, rtol=1e-14)
    assert raw_kernel[50] == pytest.approx(0.25, abs=1e-15)
    assert raw_kernel[50] != pytest.approx(4 * np.log(2) * 0.25, abs=1e-6)


def test_builder_rejects_noncanonical_and_source_overlapping_outputs(
    analysis: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    noncanonical = tmp_path / "analysis-output"
    with pytest.raises(ValueError, match="Refusing noncanonical project_root"):
        analysis.run_analysis(project_root=noncanonical, repo_root=ROOT)
    assert not noncanonical.exists()

    project_relative = PROJECT.relative_to(ROOT).as_posix()
    monkeypatch.setattr(analysis, "SOURCE_ROOTS_RELATIVE", (project_relative,))
    with pytest.raises(ValueError, match="Refusing output/source overlap"):
        analysis.run_analysis(project_root=PROJECT, repo_root=ROOT)


def test_periodic_origin_average_matches_explicit_submatrices(
    analysis: ModuleType,
) -> None:
    nx, ny = 1, 8
    dimension = 2 * nx * ny
    rng = np.random.default_rng(20260901)
    matrix = rng.normal(size=(dimension, dimension)) + 1j * rng.normal(
        size=(dimension, dimension)
    )
    unitary, _ = np.linalg.qr(matrix)
    occupations = np.linspace(0.03, 0.97, dimension)
    correlation = (unitary * occupations) @ unitary.conj().T
    archived_covariance = 2 * correlation - np.eye(dimension)

    lengths, fast_variance, auxiliaries = analysis.periodic_full_x_charge_curve(
        archived_covariance, nx, ny
    )
    explicit = []
    for length in lengths:
        origin_values = []
        for origin in range(ny):
            ys = (origin + np.arange(int(length))) % ny
            indices = (
                ys[:, None] * (2 * nx) + np.arange(2 * nx)[None, :]
            ).reshape(-1)
            restricted = correlation[np.ix_(indices, indices)]
            origin_values.append(
                np.trace(restricted).real - np.square(np.abs(restricted)).sum()
            )
        explicit.append(np.mean(origin_values))
    np.testing.assert_allclose(fast_variance, explicit, atol=1e-12, rtol=0)
    assert auxiliaries["nbar"] == pytest.approx(
        np.trace(correlation).real / ny, abs=1e-12
    )


def test_archived_regressions_and_manifest_integrity(analysis: ModuleType) -> None:
    exact_rows = {
        row["construction"]: row
        for row in _read_csv(PROJECT / "tables/exact_b0_regression.csv")
    }
    assert set(exact_rows) == set(analysis.EXPECTED_B0)
    for construction, expected in analysis.EXPECTED_B0.items():
        row = exact_rows[construction]
        for key, value in expected.items():
            assert float(row[key]) == pytest.approx(value, abs=1e-10)

    result21_rows = {
        row["case_id"]: row
        for row in _read_csv(PROJECT / "tables/result21_regression.csv")
    }
    assert set(result21_rows) == set(analysis.EXPECTED_RESULT21_K)
    for case_id, expected_k in analysis.EXPECTED_RESULT21_K.items():
        row = result21_rows[case_id]
        assert int(row["trajectories"]) == 10
        assert int(row["Ay_fit_min"]) == 8
        assert float(row["k_wall"]) == pytest.approx(expected_k, abs=5e-5)
        for key in ("c_1", "c_2", "c_3"):
            assert np.isfinite(float(row[key]))
        assert np.isfinite(float(row["c_1_archive_compatibility"]))
        assert np.isfinite(float(row["endpoint_exact_minus_archive_c_1"]))
        assert float(row["archive_compatibility_S1_reproduction_error"]) < 1e-10
        assert row["archive_compatibility_S1_reproduction_pass"].lower() in {
            "true",
            "1",
        }

    prior_rows = _read_csv(PROJECT / "tables/prior_production_regression.csv")
    assert len(prior_rows) == 16
    assert sum(int(row["trajectories"]) for row in prior_rows) == 160
    assert sum(
        row["matched_trivial"].lower() in {"true", "1"} for row in prior_rows
    ) == 8
    assert len(_read_csv(PROJECT / "tables/result21_paired_fits.csv")) == 40

    validation = json.loads(
        (PROJECT / "tables/validation_summary.json").read_text(encoding="utf-8")
    )
    assert all(row["pass"] for row in validation["exact_b0"])
    for row in validation["exact_b0"]:
        assert row["archived_entropy_estimator"]
        assert row["canonical_entropy_estimator"]
        assert row["archived_entropy_estimator"] != row["canonical_entropy_estimator"]
        assert (
            row[
                "maximum_endpoint_exact_vs_archive_compatibility_entropy_difference"
            ]
            < 3e-8
        )
    assert all(row["pass"] for row in validation["result21_expected_checks"])
    assert max(
        row["max_abs_error"] for row in validation["result21_formula_checks"]
    ) < 1e-10

    manifest = json.loads(
        (PROJECT / "analysis_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["schema_version"] == 1
    assert manifest["workloads"]["stochastic_renyi_computed"] is True
    assert "whole trajectory" in manifest["conventions"]["sample_unit"]
    assert "do not determine chirality" in " ".join(manifest["limitations"])

    ledger = {
        row["gate"]: row for row in manifest["validation"]["gate_ledger"]
    }
    required_gates = {
        "source_fingerprints",
        "exact_b0_regression",
        "fit_window_sensitivity",
        "result21_origin_formula",
        "result21_purity_and_hermiticity",
        "result21_locked_k_regression",
        "result21_archived_s1_reproduction",
        "stochastic_renyi_reduction",
        "restricted_spectrum_range",
        "paired_whole_trajectory_bootstrap",
        "prior_production_completeness",
        "prior_compact_purity",
        "wall_log_chord_vs_constant",
        "matched_trivial_null",
    }
    assert set(ledger) == required_gates
    assert all(row["required"] is True for row in ledger.values())
    assert all(row["status"] in {"pass", "fail", "not_evaluated"} for row in ledger.values())
    computed_complete = all(
        row["status"] == "pass" for row in ledger.values() if row["required"]
    )
    assert manifest["validation"]["all_required_checks_pass"] is computed_complete
    assert computed_complete, ledger

    claim_table = json.loads(
        (PROJECT / "tables/scientific_claim_status.json").read_text(
            encoding="utf-8"
        )
    )
    claim_status = manifest["scientific_claim_status"]
    assert claim_table == claim_status
    assert claim_status["overall_scientific_acceptance"] == {
        "status": "qualified_not_met",
        "all_prespecified_criteria_met": False,
        "reason": (
            "strict paired minimal-U(1)_1 null and literal trivial-control "
            "model-preference criteria are not both met"
        ),
    }
    assert (
        claim_status["checks"]["trivial_control_coefficient_null"]["status"]
        == "pass"
    )
    trivial_model_check = claim_status["checks"][
        "trivial_control_log_model_preference"
    ]
    assert trivial_model_check["status"] == "not_met_numerical_floor"
    assert trivial_model_check["offending_rows"]
    assert all(
        row["entropy_constant_minus_log_aic"] > 0
        or row["charge_constant_minus_log_aic"] > 0
        for row in trivial_model_check["offending_rows"]
    )

    minimal_claim = claim_status["claims"]["minimal_u1_1"]
    assert minimal_claim["status"] == "blocked_by_paired_null"
    expected_result21_offenders = {
        ("N16x30_nsh1_perfect_correction", 30, 1, 1),
        ("N16x30_nsh2_perfect_correction", 30, 2, 1),
        ("N16x30_nsh2_perfect_correction", 30, 2, 2),
        ("N16x30_nsh2_perfect_correction", 30, 2, 3),
    }
    observed_result21_offenders = {
        (row["case_id"], row["Ny"], row["nshell"], row["renyi_order"])
        for row in minimal_claim["offending_primary_rows"]
        if row["dataset"] == "result21"
    }
    assert expected_result21_offenders <= observed_result21_offenders
    for row in minimal_claim["offending_primary_rows"]:
        assert row["window"] == "primary"
        assert row["zero_excluded"] is True
        assert row["ci_low"] > 0 or row["ci_high"] < 0

    exact_sensitivity = _read_csv(PROJECT / "tables/exact_b0_sensitivity.csv")
    result21_sensitivity = _read_csv(
        PROJECT / "tables/result21_sensitivity_summary.csv"
    )
    prior_sensitivity = _read_csv(
        PROJECT / "tables/prior_production_sensitivity_summary.csv"
    )
    assert len(exact_sensitivity) == 6
    assert len(result21_sensitivity) == 12
    assert len(prior_sensitivity) == 48
    for rows, identifier in (
        (exact_sensitivity, "construction"),
        (result21_sensitivity, "case_id"),
        (prior_sensitivity, "case_id"),
    ):
        grouped: dict[str, set[str]] = {}
        for row in rows:
            grouped.setdefault(row[identifier], set()).add(row["window"])
        assert all(
            windows == {"primary", "drop_largest", "drop_smallest"}
            for windows in grouped.values()
        )

    for key, expected_count in (
        ("result21_bootstrap", 12),
        ("prior_production_bootstrap", 48),
    ):
        bootstrap_rows = manifest["validation"][key]
        assert len(bootstrap_rows) == expected_count
        for row in bootstrap_rows:
            labels = row["covariance_labels"]
            assert row["replicates"] == 20_000
            assert row["confidence"] == pytest.approx(0.95, abs=0)
            assert set(labels) == set(row["mean"]) == set(row["ci_low"]) == set(
                row["ci_high"]
            )
            for label in labels:
                assert row["ci_low"][label] <= row["mean"][label] <= row["ci_high"][label]
            covariance = np.asarray(row["covariance"], dtype=float)
            assert covariance.shape == (len(labels), len(labels))
            np.testing.assert_allclose(covariance, covariance.T, atol=1e-14, rtol=0)
            assert np.min(np.linalg.eigvalsh(covariance)) >= -1e-12

    spectrum_checks = manifest["validation"]["result21_renyi_spectrum_checks"]
    assert len(spectrum_checks) == 4
    assert all(row["outside_tolerance_values"] == 0 for row in spectrum_checks)
    assert all(row["pass"] for row in manifest["validation"]["result21_state_checks"])
    assert all(
        row["pass"] for row in manifest["validation"]["prior_production_state_checks"]
    )

    source_paths = [row["path"] for row in manifest["sources"]]
    assert len(source_paths) == len(set(source_paths))
    for row in manifest["sources"]:
        assert re.fullmatch(r"[0-9a-f]{64}", row["sha256"])
        assert int(row["bytes"]) > 0
    assert any(
        "covariance_convention" in row.get("metadata", {})
        and "sequence" in row["metadata"]
        and any(key.startswith("dtype") for key in row["metadata"])
        for row in manifest["sources"]
    )

    for row in manifest["outputs"]:
        path = PROJECT / row["path"]
        assert path.is_file(), path
        assert path.stat().st_size == int(row["bytes"])
        assert _sha256(path) == row["sha256"]
