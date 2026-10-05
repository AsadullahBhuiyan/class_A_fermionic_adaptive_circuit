from __future__ import annotations

import json
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np

PACKAGE_DIR = Path(__file__).resolve().parents[1]
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

from b0lib import (  # noqa: E402
    atomic_json,
    atomic_npz,
    _modular_source,
    correlation_displacements,
    entropy_observables,
    full_covariance_from_displacements,
    load_locked_config,
    make_model,
    modular_hamiltonian_from_correlation,
    modular_packet,
    occupied_blocks,
    physical_response,
    preflight_checks,
    spectral_observables,
    sha256_file,
    strip_correlation,
    translated_half_window_average,
    twist_observables,
)
from run_b0_campaign import choose_width, valid_complete_geometry  # noqa: E402


class B0CampaignTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config, _, _ = load_locked_config()

    def test_small_system_formula_gauge_and_projector_preflight(self) -> None:
        checks, _ = preflight_checks(self.config)
        self.assertTrue(checks["pass"], checks)

    def test_fixed_rank_for_both_constructions(self) -> None:
        nx, ny = 6, 8
        for construction in self.config["constructions"]:
            model = make_model(nx, ny, self.config)
            _, _, projectors = occupied_blocks(
                model, construction, self.config["occupation_twist"]
            )
            self.assertEqual(int(round(np.trace(projectors, axis1=1, axis2=2).real.sum())), nx * ny)
            cfull = full_covariance_from_displacements(
                correlation_displacements(projectors, self.config["occupation_twist"])
            )
            self.assertLess(np.max(np.abs(cfull @ cfull - cfull)), 1e-10)

    def test_entropy_contour_reconstructs_entropy(self) -> None:
        model = make_model(6, 8, self.config)
        _, _, projectors = occupied_blocks(
            model, "coupled", self.config["occupation_twist"]
        )
        result = entropy_observables(
            correlation_displacements(projectors, self.config["occupation_twist"]),
            model,
            self.config,
        )
        self.assertLess(float(result.payload["contour_sum_max_error"]), 1e-10)
        q = result.payload["entropy_q"]
        spectra = result.payload["entropy_spectra"]
        ay = model.Ny // 2
        for qi, order in enumerate(q):
            values = spectra[ay, : 2 * model.Nx * ay]
            values = np.clip(values, 1e-12, 1 - 1e-12)
            if order == 1:
                reconstructed = -np.sum(values * np.log(values) + (1 - values) * np.log(1 - values))
            else:
                reconstructed = np.sum(np.log(values**order + (1 - values) ** order) / (1 - order))
            self.assertAlmostEqual(reconstructed, result.payload["entropy_values"][qi, ay], places=10)

    def test_translated_half_window_and_modular_convention(self) -> None:
        model = make_model(6, 8, self.config)
        _, _, projectors = occupied_blocks(
            model, "coupled", self.config["occupation_twist"]
        )
        cdisp = correlation_displacements(projectors, self.config["occupation_twist"])
        cfull = full_covariance_from_displacements(cdisp)
        fixed = strip_correlation(cdisp, model.Ny // 2)
        averaged = translated_half_window_average(cfull, model.Nx, model.Ny)
        self.assertLess(float(np.max(np.abs(fixed - averaged))), 1e-10)
        hmod, occupations, vectors = modular_hamiltonian_from_correlation(
            averaged, self.config["modular_eigenvalue_clip"]
        )
        legacy_energies = -2 * np.arctanh(2 * occupations - 1)
        legacy_hmod = (vectors * legacy_energies[None, :]) @ vectors.conj().T
        self.assertLess(
            float(np.max(np.abs(hmod - legacy_hmod))),
            self.config["modular_convention_tolerance"],
        )

    def test_endpoint_sources_and_legacy_aggregate_charge(self) -> None:
        nx, ay = 20, 20
        endpoints = [0, ay - 1]
        wall_only = _modular_source(nx, ay, [6, 14], endpoints, normalize=False)
        three_column = _modular_source(
            nx, ay, [5, 6, 7, 13, 14, 15], endpoints, normalize=False
        )
        self.assertAlmostEqual(float(np.vdot(wall_only, wall_only).real), 8.0)
        self.assertAlmostEqual(float(np.vdot(three_column, three_column).real), 24.0)
        isolated = _modular_source(nx, ay, [5, 6, 7], [0], normalize=True)
        self.assertAlmostEqual(float(np.vdot(isolated, isolated).real), 1.0)
        self.assertEqual(np.count_nonzero(isolated), 6)

    def test_consecutive_full_calibration_width_selection(self) -> None:
        config = dict(self.config)
        config["nx_scan"] = [12, 16, 20, 24]
        summaries = []
        for construction in config["constructions"]:
            for nx in config["nx_scan"]:
                summaries.append(
                    {
                        "construction": construction,
                        "nx": nx,
                        "edge": {"width_ratio": 1e-4},
                        "calibration": {"pass": nx >= 20},
                    }
                )
        with patch(
            "run_b0_campaign.single_geometry_calibration",
            side_effect=lambda summary, _: summary["calibration"],
        ):
            selected, passed, rows = choose_width(summaries, config)
        self.assertTrue(passed)
        self.assertEqual(selected, 20)
        self.assertTrue(next(row for row in rows if row["nx"] == 12)["mass_pass_both"])
        self.assertFalse(next(row for row in rows if row["nx"] == 12)["pass_both"])

    def test_exact_20x40_endpoint_handedness_regression(self) -> None:
        for construction in self.config["constructions"]:
            model = make_model(20, 40, self.config)
            _, _, projectors = occupied_blocks(
                model, construction, self.config["occupation_twist"]
            )
            cdisp = correlation_displacements(projectors, self.config["occupation_twist"])
            payload, rows, diagnostics = modular_packet(
                cdisp, model, construction, self.config
            )
            self.assertTrue(diagnostics["pass"], diagnostics)
            self.assertLess(
                diagnostics["primary_velocities"][0]
                * diagnostics["primary_velocities"][1],
                0,
            )
            self.assertTrue(all(diagnostics["sign_stable"]))
            self.assertLess(diagnostics["max_norm_drift"], 1e-10)
            self.assertEqual(payload["modular_packets"].shape[:3], (2, 2, 2))
            self.assertEqual(len([row for row in rows if row["is_primary"]]), 2)
            if construction == "coupled":
                self.assertEqual(payload["legacy_aggregate_packets"].shape[:2], (2, 801))

    def test_response_linearity_charge_and_opposite_edges(self) -> None:
        model = make_model(6, 8, self.config)
        _, edge_rows, _ = spectral_observables(model, "coupled", self.config)
        self.assertLess(edge_rows[0]["velocity"] * edge_rows[1]["velocity"], 0)
        _, _, projectors = occupied_blocks(model, "coupled", self.config["occupation_twist"])
        payload, _ = physical_response(model, "coupled", projectors, self.config)
        self.assertLess(float(payload["response_charge_drift"]), 1e-10)
        self.assertLess(
            float(np.max(payload["response_epsilon_relative_errors"])),
            self.config["response_epsilon_relative_tolerance"],
        )

    def test_overlap_tracking_rank_and_unit_wall_flow(self) -> None:
        model = make_model(6, 8, self.config)
        for construction in self.config["constructions"]:
            payload, rows = twist_observables(
                model, construction, self.config, points=17, sign=1, save_links=True
            )
            self.assertEqual(int(payload["twist_frame_rank"]), model.Nx)
            self.assertGreater(
                float(np.min(payload["twist_overlap_singular_values"])),
                self.config["overlap_singular_tolerance"],
            )
            self.assertEqual([row["absolute_flow"] for row in rows], [1, 1])
            self.assertEqual(sum(row["signed_flow"] for row in rows), 0)

    def test_atomic_resume_rejects_partial_or_modified_products(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            npz = root / "geometry.npz"
            status = root / "geometry.json"
            atomic_npz(npz, schema_version=np.asarray(1), value=np.arange(3))
            atomic_json(status, {"npz_sha256": sha256_file(npz), "key": "test"})
            self.assertIsNotNone(valid_complete_geometry(npz, status))
            with npz.open("ab") as handle:
                handle.write(b"interruption")
            self.assertIsNone(valid_complete_geometry(npz, status))


if __name__ == "__main__":
    unittest.main()
