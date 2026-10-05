from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parents[2]
for path in (PACKAGE_ROOT, REPO_ROOT / "src"):
    if str(path) in sys.path:
        sys.path.remove(str(path))
    sys.path.insert(0, str(path))

import analyze_nx_wall_purification_convergence as analysis
import run_nx_wall_purification_convergence_cpu as runner
from fgtn.classA_U1FGTN import classA_U1FGTN


def make_model(nx: int, ny: int, *, projectors: bool = False) -> classA_U1FGTN:
    model = classA_U1FGTN(
        nx,
        ny,
        DW=True,
        nshell=runner.NSHELL,
        alpha_1=runner.ALPHA_TOPOLOGICAL,
        alpha_2=runner.ALPHA_TRIVIAL,
        trial_orbitals=runner.TRIAL_ORBITALS,
        dw_truncation=True,
        dw_interval=runner.expected_domain_wall(nx),
    )
    if projectors:
        model.construct_OW_projectors(
            nshell=runner.NSHELL,
            DW=True,
            trial_orbitals=runner.TRIAL_ORBITALS,
            dw_truncation=True,
        )
    return model


@pytest.mark.parametrize(
    ("nx", "walls", "slab_width"),
    ((20, (4, 16), 13), (24, (4, 20), 17), (28, (5, 23), 19)),
)
def test_standard_domain_wall_geometry(nx: int, walls: tuple[int, int], slab_width: int) -> None:
    model = make_model(nx, 20)
    metadata = runner.geometry_metadata(model)
    assert tuple(metadata["domain_wall_locations"]) == walls
    assert metadata["active_slab_width_sites"] == slab_width
    assert metadata["active_mode_count"] == 2 * slab_width * 20
    assert metadata["bulk_rows"] == list(range(walls[0] + 2, walls[1] - 1))


def test_cycle_zero_maxmix_observables_and_contour_sum() -> None:
    model = make_model(20, 4)
    geometry = runner.geometry_metadata(model)
    active = model.active_top_layer_indices(meas_slab_only=True)
    dimension = 2 * model.Nx * model.Ny
    shifted = np.eye(dimension, dtype=np.complex128)
    shifted[np.ix_(active, active)] = 0.0
    observed = runner.compute_cycle_observables(
        shifted,
        active_indices=active,
        nx=model.Nx,
        ny=model.Ny,
        walls=tuple(geometry["wall_rows"]),
        bulk_rows=np.asarray(geometry["bulk_rows"]),
    )
    runner.validate_cycle_zero(observed, geometry)
    assert observed["total_entropy_bits"] == pytest.approx(geometry["active_mode_count"])
    assert observed["total_charge_variance"] == pytest.approx(geometry["active_mode_count"] / 4)
    assert observed["contour_sum_error_bits"] < 1e-10
    assert observed["charge_variance_per_total_mode"] == pytest.approx(
        geometry["active_mode_count"] / 4 / geometry["total_mode_count"]
    )


def test_first_sustained_crossing_ignores_early_false_crossing() -> None:
    cycles = np.arange(6)
    curve = np.asarray([1.0, 0.009, 0.02, 0.008, 0.007, 0.006])
    assert analysis.first_sustained_crossing(cycles, curve, 0.01) == 3
    assert np.isnan(analysis.first_sustained_crossing(cycles, np.full(6, 0.02), 0.01))


def test_synthetic_power_and_bulk_fits() -> None:
    cycles = np.arange(0, 101)
    wall = np.zeros_like(cycles, dtype=np.float64)
    wall[1:] = 1.7 * cycles[1:] ** -1.15
    wall[0] = 2.0
    wall_fit = analysis.fit_wall_models(cycles, wall)
    assert wall_fit["power_alpha"] == pytest.approx(1.15, rel=1e-6)
    assert wall_fit["delta_aicc_exp_minus_power"] > 0

    bulk = np.zeros_like(cycles, dtype=np.float64)
    bulk[1:] = 1.2 * np.exp(-cycles[1:] / 0.8) + 0.08 * cycles[1:] ** -1.15
    bulk[0] = 2.0
    bulk_fit = analysis.fit_bulk_model(cycles, bulk, wall_fit["power_alpha"])
    assert bulk_fit["bulk_tau"] == pytest.approx(0.8, rel=0.05)


def synthetic_geometry(nx: int, *, sample_count: int = 25) -> analysis.GeometryData:
    cycles = np.arange(101)
    wall_curve = np.empty(101)
    wall_curve[0] = 2.0
    wall_curve[1:] = 2.0 * cycles[1:] ** -1.2
    bulk_curve = np.empty(101)
    bulk_curve[0] = 2.0
    bulk_curve[1:] = 1.5 * np.exp(-cycles[1:] / 0.7) + 0.05 * cycles[1:] ** -1.2
    wall = np.repeat(wall_curve[None, :], sample_count, axis=0)
    bulk = np.repeat(bulk_curve[None, :], sample_count, axis=0)
    profile = np.repeat(wall[:, :, None], nx, axis=2)
    return analysis.GeometryData(
        nx=nx,
        ny=20,
        cycles=cycles,
        sample_indices=np.arange(sample_count),
        wall_entropy=wall,
        bulk_entropy=bulk,
        total_entropy_per_total_mode=0.5 * wall,
        charge_variance_per_total_mode=0.125 * wall,
        active_purity_deficit_rms=0.5 * wall,
        active_covariance_frobenius_rms=np.maximum(0.0, 1.0 - 0.5 * wall),
        full_covariance_frobenius_per_total_mode=0.1 * np.maximum(0.0, 1.0 - 0.5 * wall),
        entropy_profile_bits=profile,
        run_directory=Path(f"N{nx}x20"),
    )


def test_bootstrap_reproducibility_and_equivalence_decision() -> None:
    geometries = {nx: synthetic_geometry(nx) for nx in runner.DEFAULT_NX_VALUES}
    first_rng = np.random.default_rng(123)
    second_rng = np.random.default_rng(123)
    first = {
        nx: analysis.bootstrap_geometry(data, bootstrap_count=50, rng=first_rng, epsilon=0.01)
        for nx, data in geometries.items()
    }
    second = {
        nx: analysis.bootstrap_geometry(data, bootstrap_count=50, rng=second_rng, epsilon=0.01)
        for nx, data in geometries.items()
    }
    for nx in geometries:
        np.testing.assert_array_equal(first[nx]["wall_thresholds"], second[nx]["wall_thresholds"])
    decision, _, z_values = analysis.compute_joint_decision(
        geometries, first, epsilon=0.01, equivalence_margin=5.0
    )
    assert decision["classification"] == "no_resolved_material_Nx_dependence"
    assert np.nanmedian(z_values) == pytest.approx(0.0, abs=1e-12)


def test_build_specs_topup_preserves_first_25(tmp_path: Path) -> None:
    args_25 = runner.parse_args(
        ["--output-root", str(tmp_path), "--campaign-id", "topup", "--samples", "25", "--cycles", "4"]
    )
    args_50 = runner.parse_args(
        ["--output-root", str(tmp_path), "--campaign-id", "topup", "--samples", "50", "--cycles", "4"]
    )
    specs_25 = runner.build_specs(args_25, tmp_path)
    specs_50 = runner.build_specs(args_50, tmp_path)
    first_25_keys = {(spec.nx, spec.sample_index): (spec.seed, spec.result_path) for spec in specs_25}
    first_50_keys = {(spec.nx, spec.sample_index): (spec.seed, spec.result_path) for spec in specs_50}
    for key, value in first_25_keys.items():
        assert first_50_keys[key] == value


def trajectory_spec(tmp_path: Path, name: str, cycles: int) -> runner.TrajectorySpec:
    return runner.TrajectorySpec(
        campaign_id="resume-test",
        nx=8,
        ny=4,
        cycles=cycles,
        sample_index=0,
        seed=7319,
        checkpoint_stride=1,
        result_path=str(tmp_path / name / "trajectory_000.npz"),
        restart_path=str(tmp_path / name / "trajectory_000.pkl"),
    )


def test_interrupted_resume_matches_uninterrupted_run(tmp_path: Path) -> None:
    resumed_model = make_model(8, 4, projectors=True)
    runner.run_trajectory(trajectory_spec(tmp_path, "resumed", 1), resumed_model)
    runner.run_trajectory(trajectory_spec(tmp_path, "resumed", 3), resumed_model)

    uninterrupted_model = make_model(8, 4, projectors=True)
    runner.run_trajectory(trajectory_spec(tmp_path, "uninterrupted", 3), uninterrupted_model)
    with np.load(tmp_path / "resumed/trajectory_000.npz", allow_pickle=False) as resumed:
        with np.load(tmp_path / "uninterrupted/trajectory_000.npz", allow_pickle=False) as uninterrupted:
            for key in runner.OBSERVABLE_KEYS:
                np.testing.assert_allclose(resumed[key], uninterrupted[key], rtol=0, atol=1e-12, equal_nan=True)


def write_fake_campaign(results_root: Path, campaign_id: str) -> None:
    campaign_root = results_root / "campaigns" / campaign_id
    result_rows = []
    for nx in runner.DEFAULT_NX_VALUES:
        data = synthetic_geometry(nx)
        run_dir = campaign_root / "runs" / f"N{nx}x20"
        run_dir.mkdir(parents=True)
        np.savez_compressed(
            run_dir / "trajectory_observables.npz",
            cycles=data.cycles,
            sample_indices=data.sample_indices,
            wall_entropy_bits_per_cell=data.wall_entropy,
            bulk_entropy_bits_per_cell=data.bulk_entropy,
            total_entropy_per_total_mode_bits=data.total_entropy_per_total_mode,
            charge_variance_per_total_mode=data.charge_variance_per_total_mode,
            active_purity_deficit_rms=data.active_purity_deficit_rms,
            active_covariance_frobenius_rms=data.active_covariance_frobenius_rms,
            full_covariance_frobenius_per_total_mode=data.full_covariance_frobenius_per_total_mode,
            entropy_profile_bits=data.entropy_profile_bits,
        )
        result_rows.append(
            {
                "config_id": f"N{nx}x20",
                "Nx": nx,
                "Ny": 20,
                "run_directory": str(run_dir),
                "complete": True,
            }
        )
    (campaign_root / "campaign_manifest.json").write_text(
        json.dumps({"status": "complete", "results": result_rows}), encoding="utf-8"
    )


def test_analysis_pipeline_writes_figures_tables_and_decision(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    output_root = tmp_path / "analysis"
    campaign_id = "synthetic"
    write_fake_campaign(results_root, campaign_id)
    status = analysis.main(
        [
            "--campaign-id",
            campaign_id,
            "--results-root",
            str(results_root),
            "--output-root",
            str(output_root),
            "--bootstrap-count",
            "20",
        ]
    )
    assert status == 0
    root = output_root / campaign_id
    assert (root / "analysis_manifest.json").exists()
    assert (root / "conclusion.json").exists()
    assert (root / "tables/fit_summary.csv").exists()
    assert (root / "figures/nx_wall_purification_convergence_main.pdf").exists()
    conclusion = json.loads((root / "conclusion.json").read_text(encoding="utf-8"))
    assert conclusion["classification"] == "no_resolved_material_Nx_dependence"
