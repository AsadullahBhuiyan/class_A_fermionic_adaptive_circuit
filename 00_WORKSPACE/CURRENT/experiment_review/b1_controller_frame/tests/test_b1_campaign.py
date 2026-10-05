from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np


RUNNER = Path(__file__).resolve().parents[1] / "run_b1_campaign.py"
SPEC = importlib.util.spec_from_file_location("b1_campaign", RUNNER)
b1 = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(b1)


class B1CampaignTests(unittest.TestCase):
    def test_locked_matrix(self) -> None:
        config = b1.load_config()
        self.assertEqual(config["schema_version"], 2)
        self.assertEqual(config["nx"], 20)
        self.assertEqual(config["ny_values"], [24, 40])
        self.assertEqual(config["trajectory_splits"], {"train": 4, "test": 4})
        self.assertEqual(set(config["constructions"]), set(b1.CONSTRUCTIONS))

    def test_cpu_snapshot_and_forced_selection(self) -> None:
        snapshot = b1._cpu_snapshot()
        self.assertGreaterEqual(len(snapshot), 4)
        selected = b1.select_idle_cpus(4, allow_busy=True).split(",")
        self.assertEqual(len(selected), 4)
        self.assertEqual(len(set(selected)), 4)

    def test_controls_retain_geometry(self) -> None:
        explicit = b1.model_for("explicit_interface", 4, 6, 1)
        explicit_control = b1.model_for("explicit_wall_off", 4, 6, 1)
        support = b1.model_for("support_terminated", 4, 6, 1)
        support_control = b1.model_for("support_wall_off", 4, 6, 1)
        self.assertEqual(explicit.DW_loc, explicit_control.DW_loc)
        self.assertEqual(support.DW_loc, support_control.DW_loc)
        self.assertEqual(
            len(support.active_top_layer_indices(meas_slab_only=True)),
            len(support_control.active_top_layer_indices(meas_slab_only=True)),
        )
        self.assertTrue(np.allclose(explicit_control.alpha_profile, 30.0))
        self.assertTrue(np.allclose(support_control.alpha_profile, 30.0))

    def test_ky_fan_cost_and_sector_residual(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "small.npz"
            row = b1._static_worker(
                {
                    "label": "explicit_interface",
                    "nx": 4,
                    "ny": 6,
                    "nshell": 1,
                    "svd_rtol": 1e-10,
                    "numerical_tolerance": 1e-10,
                    "output": str(output),
                }
            )
            self.assertLess(row["cost_residual"], 1e-10)
            self.assertLess(row["hermiticity_residual"], 1e-10)
            with np.load(output) as data:
                residual = b1.frame_residual(
                    data["mode_constraint_weights"],
                    data["target_occupancies"],
                    int(data["target_rank"]),
                )
                np.testing.assert_allclose(residual, data["residuals_half_filling"], atol=1e-12)

    def test_closed_gap_reports_minimizer_range(self) -> None:
        weights = np.asarray([[0.8, 0.2], [0.2, 0.8]])
        targets = np.asarray([0, 1])
        eigenvalues = np.asarray([0.0, 0.0])
        lower, upper, cluster = b1.degenerate_residual_bounds(
            weights, targets, eigenvalues, rank=1, tolerance=1e-10
        )
        self.assertEqual(cluster, (0, 2))
        self.assertTrue(np.all(lower <= upper))
        self.assertTrue(np.any(upper > lower))

    def test_train_test_and_geometry_seeds_are_disjoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            tasks = b1.trajectory_tasks(Path(temporary))
        seeds = [task["seed"] for task in tasks]
        self.assertEqual(len(seeds), 64)
        self.assertEqual(len(set(seeds)), len(seeds))

    def test_profile_metrics_do_not_shift_origins(self) -> None:
        static = np.asarray([0.0, 1.0, 0.0, 1.0])
        activity = np.asarray([0.0, 2.0, 0.0, 2.0])
        self.assertAlmostEqual(b1.pearson_overlap(static, activity), 1.0)
        wall = np.asarray([False, True, False, True])
        bulk = ~wall
        bounded, ratio = b1.contrast(activity, wall, bulk)
        self.assertAlmostEqual(bounded, 1.0)
        self.assertTrue(np.isinf(ratio))


if __name__ == "__main__":
    unittest.main()
