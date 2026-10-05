from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (
    REPO_ROOT / "colab_charge_fluctuations" / "src",
    REPO_ROOT / "colab_charge_fluctuations" / "scripts",
    REPO_ROOT / "src",
):
    path_text = str(path)
    if path_text in sys.path:
        sys.path.remove(path_text)
    sys.path.insert(0, path_text)

from fgtn.classA_U1FGTN import classA_U1FGTN
from charge_sharpening_alpha_sweep_loader import load_charge_sharpening_campaign
from run_purification_charge_sharpening_alpha_sweep_cpu import (
    TrajectorySpec,
    build_specs,
    parse_alpha_csv,
    resolve_alpha_values,
    resolve_cpu_pool,
    run_trajectory_block,
    stable_seed,
    trajectory_observables,
)


def test_cycle_observer_runs_at_initialization_and_completed_cycles() -> None:
    model = classA_U1FGTN(2, 2, DW=True, nshell=0, alpha_1=1, alpha_2=30, dw_truncation=True)
    observed = []

    def observer(**payload):
        observed.append((payload["cycle"], payload["G"].shape, payload["batch_count"]))

    model.run_markov_circuit(
        G_history=False,
        progress=False,
        cycles=2,
        perfect_correction=True,
        samples=1,
        parallelize_samples=False,
        init_mode="maxmix",
        save=False,
        cycle_observer=observer,
    )
    assert [item[0] for item in observed] == [0, 1, 2]
    assert all(item[1] == (8, 8) for item in observed)
    assert all(item[2] == 1 for item in observed)


def test_cycle_observer_rejects_internal_sample_parallelism() -> None:
    model = classA_U1FGTN(2, 2, DW=True, nshell=0, alpha_1=1, alpha_2=30, dw_truncation=True)
    with pytest.raises(ValueError, match="cycle_observer requires serial"):
        model.run_markov_circuit(
            G_history=False,
            progress=False,
            cycles=1,
            perfect_correction=True,
            samples=2,
            parallelize_samples=True,
            save=False,
            cycle_observer=lambda **_: None,
        )


def test_scalar_observables_for_maximally_mixed_state() -> None:
    dimension = 6
    entropy, variance = trajectory_observables(np.zeros((dimension, dimension), dtype=np.complex128), np.arange(dimension))
    assert entropy == pytest.approx(dimension * np.log(2.0))
    assert variance == pytest.approx(dimension / 4.0)


def test_stable_seed_is_deterministic_and_sensitive() -> None:
    assert stable_seed("pc", 20, 1.0, 0) == stable_seed("pc", 20, 1.0, 0)
    assert stable_seed("pc", 20, 1.0, 0) != stable_seed("pc", 20, 1.0, 1)


def test_alpha_grid_parsing_and_spec_counts(tmp_path: Path) -> None:
    class Args:
        smoke = False
        protocols = ["perfect_correction", "postselection"]
        alpha_values = None
        alpha_linspace = (1.0, 3.0, 21.0)
        dw_truncation = 1

    args = Args()
    args.resolved_alpha_values = resolve_alpha_values(args)
    assert args.resolved_alpha_values == pytest.approx(tuple(np.linspace(1.0, 3.0, 21)))
    specs = build_specs(args, tmp_path, "fine")
    assert sum(spec.protocol == "perfect_correction" for spec in specs) == 630
    assert sum(spec.protocol == "postselection" for spec in specs) == 63
    assert sorted({spec.alpha_topological_region for spec in specs}) == pytest.approx(list(np.linspace(1.0, 3.0, 21)))


def test_alpha_csv_parser() -> None:
    assert parse_alpha_csv("1, 1.5,2") == (1.0, 1.5, 2.0)


def test_free_cpu_pool_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    class Args:
        cpu_selection = "free"
        cpu_free_samples = 2
        cpu_free_sample_interval = 0.05
        cpu_free_threshold = 20.0
        min_free_workers = 2

    def fake_sample_cpu_usage(available_cpus, *, samples, interval):
        return {
            "samples": samples,
            "interval_seconds": interval,
            "available_cpus": available_cpus,
            "per_cpu_average_percent": {"0": 5.0, "1": 55.0, "2": 10.0},
            "per_cpu_max_percent": {"0": 7.0, "1": 60.0, "2": 12.0},
        }

    monkeypatch.setattr(
        "run_purification_charge_sharpening_alpha_sweep_cpu.sample_cpu_usage",
        fake_sample_cpu_usage,
    )
    cpus, metadata = resolve_cpu_pool(Args(), [0, 1, 2])
    assert cpus == [0, 2]
    assert metadata["cpu_selection"] == "free"
    assert metadata["busy_cpus"] == [1]


def test_trajectory_checkpoint_resume(tmp_path: Path) -> None:
    checkpoint = tmp_path / "trajectory_000.npz"
    spec = TrajectorySpec(
        protocol="postselection",
        nx=2,
        ny=2,
        alpha_topological_region=1.0,
        alpha_trivial_region=30.0,
        dw_truncation=False,
        cycles=1,
        sample_index=0,
        seed=1234,
        checkpoint_path=str(checkpoint),
    )
    first = run_trajectory_block([spec.__dict__])
    second = run_trajectory_block([spec.__dict__])
    assert first["results"][0]["status"] == "completed"
    assert second["results"][0]["status"] == "skipped"
    with np.load(checkpoint, allow_pickle=False) as payload:
        assert payload["total_entropy"].shape == (1,)
        assert payload["total_charge_variance"].shape == (1,)


def _write_fake_protocol(root: Path, protocol: str, campaign_id: str) -> None:
    data_root = root / "cpu_data" / "purification_charge_sharpening_alpha_sweep"
    config_id = f"N20x20_alpha1-1_{protocol}"
    run_dir_relative = Path(protocol) / "campaigns" / campaign_id / "runs" / config_id
    run_dir = data_root / run_dir_relative
    run_dir.mkdir(parents=True)
    samples = 2 if protocol == "perfect_correction" else 1
    cycles = np.arange(1, 3)
    entropy = np.ones((samples, 2))
    variance = np.full((samples, 2), 0.01)
    np.savez_compressed(
        run_dir / "trajectory_observables.npz",
        cycles=cycles,
        sample_indices=np.arange(samples),
        seeds=np.arange(samples, dtype=np.uint32),
        total_entropy=entropy,
        total_charge_variance=variance,
    )
    summary = {
        "config_id": config_id,
        "protocol": protocol,
        "Nx": 20,
        "Ny": 20,
        "cycles": 2,
        "samples_actual": samples,
        "alpha_topological_region": 1.0,
        "dw_truncation": False,
        "observable_region": "full_top_layer",
        "projector_diagnostics": {"critical_point_alpha_equals_2": False},
    }
    (run_dir / "run_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    rows = []
    for sample in range(samples):
        for cycle in cycles:
            rows.append({"sample_index": sample, "cycle": cycle})
    pd.DataFrame(rows).to_csv(run_dir / "scalar_metrics.csv", index=False)
    campaign_root = data_root / protocol / "campaigns" / campaign_id
    manifest = {
        "campaign_id": campaign_id,
        "complete": True,
        "results": [
            {
                "config_id": config_id,
                "protocol": protocol,
                "Nx": 20,
                "Ny": 20,
                "alpha_topological_region": 1.0,
                "dw_truncation": False,
                "observable_region": "full_top_layer",
                "cycles": 2,
                "samples_expected": samples,
                "samples_completed": samples,
                "complete": True,
                "run_dir_relative": str(run_dir_relative),
            }
        ],
    }
    (campaign_root / "campaign_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (data_root / protocol / "latest_campaign.json").write_text(
        json.dumps({"campaign_id": campaign_id}), encoding="utf-8"
    )


def test_loader_validates_and_builds_tables(tmp_path: Path) -> None:
    bundle_root = tmp_path / "colab_charge_fluctuations"
    (bundle_root / "src").mkdir(parents=True)
    campaign_id = "test_campaign"
    _write_fake_protocol(bundle_root, "perfect_correction", campaign_id)
    _write_fake_protocol(bundle_root, "postselection", campaign_id)
    payload = load_charge_sharpening_campaign(bundle_root)
    assert len(payload["inventory_df"]) == 2
    assert len(payload["trajectory_df"]) == 6
    assert len(payload["steady_state_df"]) == 2
    assert payload["inventory_df"]["dw_truncation"].eq(False).all()
    assert payload["trajectory_df"]["observable_region"].eq("full_top_layer").all()
    assert payload["steady_state_df"]["observable_region"].eq("full_top_layer").all()
