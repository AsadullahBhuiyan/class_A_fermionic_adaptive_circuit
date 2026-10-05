from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest
from threadpoolctl import threadpool_limits


REPO_ROOT = Path(__file__).resolve().parents[1]
PILOT_ROOT = REPO_ROOT / "00_WORKSPACE/CURRENT/frozen_record_flux_charge_pilot"
if str(PILOT_ROOT) not in sys.path:
    sys.path.insert(0, str(PILOT_ROOT))

import run_wall_diabatic_spectral_pump_s100 as campaign  # noqa: E402
import run_wall_diabatic_controls_and_sensitivities as controls  # noqa: E402
import benchmark_a100_spectral_eigensolver as spectral_benchmark  # noqa: E402


def _config() -> dict:
    config = campaign.load_config(campaign.DEFAULT_CONFIG)
    campaign.validate_config(config)
    return config


def _source(config: dict, cell: str) -> dict:
    return campaign.source_rows(config)[cell]


def _orthonormal_frame(dimension: int, rank: int, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(dimension, rank)) + 1j * rng.normal(size=(dimension, rank))
    frame, _ = np.linalg.qr(raw)
    return np.asarray(frame, dtype=np.complex128)


def _write_shard(
    root: Path, source: dict, wall: str = "soft", shard_index: int = 0,
    backend: str = "gpu",
) -> Path:
    result = (
        root / source["relative_root"] / wall / f"shard_{shard_index:02d}.npz"
    )
    result.parent.mkdir(parents=True)
    sample_ids = list(range(5 * shard_index, 5 * shard_index + 5))
    dimension, capacity = 2 * source["Nx"] * source["Ny"], 6
    ranks = np.asarray([2, 3, 4, 5, 6], dtype=np.int64)
    frames = np.zeros((5, dimension, capacity), dtype=np.complex128)
    for member, rank in enumerate(ranks):
        frames[member, :, :rank] = _orthonormal_frame(dimension, int(rank), member + 11)
    contract = source["accepted_source_contracts"][backend]
    collection = "bridge" if source["relative_root"].startswith("bridge/") else "endpoints"
    source_cell = source.get("source_cell", source["cell"])
    identity = {
        "schema": campaign.SHARD_COMPLETION_SCHEMA,
        "stage": "endpoint_shard", "collection": collection, "cell": source_cell,
        "protocol": source["protocol"], "Nx": source["Nx"], "Ny": source["Ny"],
        "wall": wall, "shard_index": shard_index, "sample_ids": sample_ids,
        "sampling_revision": contract["sampling_revision"],
        "canonical_entry_point": contract["canonical_entry_point"],
        "execution_backend": contract["execution_backend"], "cycles": 48,
        contract["config_hash_key"]: contract["config_hash"],
        "source_hashes": contract["source_hashes"],
    }
    metadata = {
        **identity,
        "wall_flags": {"dw_truncation": wall == "hard", "meas_slab_only": wall == "hard"},
        "dtype": "complex128", "sequence": "raster_y",
        "perfect_correction": True, "postselect": False,
    }
    np.savez(
        result, schema=np.asarray(campaign.SHARD_RESULT_SCHEMA),
        sample_ids=np.asarray(sample_ids), frames=frames, ranks=ranks,
        metadata_json=np.asarray(campaign.canonical_json(metadata)),
        sampling_revision=np.asarray(contract["sampling_revision"]),
        canonical_entry_point=np.asarray(contract["canonical_entry_point"]),
        execution_backend=np.asarray(contract["execution_backend"]),
        collection=np.asarray(collection), protocol=np.asarray(source["protocol"]),
        wall=np.asarray(wall), Nx=np.asarray(source["Nx"]), Ny=np.asarray(source["Ny"]),
        cycles_total=np.asarray(48), nshell_label=np.asarray(source["protocol"]),
        alpha_1=np.asarray(1.0), alpha_2=np.asarray(30.0),
    )
    digest = campaign.sha256_path(result)
    completion = {
        **identity,
        "result": {"name": result.name, "bytes": result.stat().st_size, "sha256": digest},
    }
    result.with_suffix(".completion.json").write_text(
        json.dumps(completion, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def test_locked_config_expands_exact_primary_matrix() -> None:
    config = _config()
    rows = campaign.tasks(config)
    assert len(rows) == 1750
    assert len({row["task_id"] for row in rows}) == 1750
    assert sum(row["is_primary"] for row in rows) == 1600
    bridge = [row for row in rows if not row["is_primary"]]
    assert len(bridge) == 150
    assert {row["source_backend"] for row in bridge} == {"gpu"}
    assert len({campaign.result_paths(Path("/tmp/out"), row)[0] for row in rows}) == 1750
    assert {
        (row["protocol"], row["Nx"], row["Ny"]) for row in rows
    } == {
        (protocol, nx, 24)
        for protocol in ("nsh1", "dense")
        for nx in (20, 24, 28, 32)
    }
    assert {row["wall"] for row in rows} == {"soft", "hard"}
    assert {row["sample_id"] for row in rows} == set(range(100))


def test_five_sample_shard_is_verified_and_expanded(tmp_path: Path) -> None:
    config = _config()
    source = _source(config, "nsh1_N28x24")
    result = _write_shard(tmp_path, source)
    refs = [campaign.endpoint_ref(source, "soft", sample, tmp_path) for sample in range(5)]
    assert [ref.member_index for ref in refs] == list(range(5))
    assert len({ref.result_sha256 for ref in refs}) == 1
    for sample, ref in enumerate(refs):
        frame = campaign.load_endpoint(ref, 28, 24)
        assert frame.shape == (1344, sample + 2)
        np.testing.assert_allclose(frame.conj().T @ frame, np.eye(sample + 2), atol=1e-12)
    raw = bytearray(result.read_bytes())
    raw[-1] ^= 1
    result.write_bytes(bytes(raw))
    with pytest.raises(RuntimeError, match="dependency changed"):
        campaign.load_endpoint(refs[0], 28, 24)


def test_five_sample_cpu_fallback_contract_is_accepted_without_weakening_identity(
    tmp_path: Path,
) -> None:
    source = _source(_config(), "dense_N24x24")
    _write_shard(tmp_path, source, wall="hard", backend="canonical_cpu")
    reference = campaign.endpoint_ref(source, "hard", 3, tmp_path)
    frame = campaign.load_endpoint(reference, 24, 24)
    assert frame.dtype == np.complex128
    assert frame.shape == (1152, 5)


def test_bridge_lane_rejects_cpu_fallback_endpoints(tmp_path: Path) -> None:
    source = campaign.source_rows(_config())["bridge_nsh1_N20x24"]
    _write_shard(tmp_path, source, backend="canonical_cpu")
    with pytest.raises(RuntimeError, match="differs from the source lane"):
        campaign.endpoint_ref(source, "soft", 0, tmp_path)


def test_endpoint_bundle_publisher_roundtrips_into_core_loader(tmp_path: Path) -> None:
    bundle_root = REPO_ROOT / "00_WORKSPACE/CURRENT/final_production_new_designs/11_wall_pump_width_endpoints"
    name = "wall_pump_width_endpoint_bundle_runner_for_test"
    spec = importlib.util.spec_from_file_location(name, bundle_root / "run_campaign.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    case = module.Case("endpoints", "nsh1", 28, "soft", 100)
    batch = module.ExecutionBatch(case, 0, 0, 10, 12345)
    shard = module.ResultShard(batch, 0, 0, 5)
    dimension, rank = 2 * 28 * 24, 4
    frames = np.stack([_orthonormal_frame(dimension, rank, 100 + index) for index in range(10)])
    ranks = np.full(10, rank, dtype=np.int64)
    observer = module.ChargeObserver(10)
    observer.seen_cycles[:] = True
    observer.global_charge[:] = rank
    config = module.expected_config(bundle_root)
    hashes = module.source_hashes(bundle_root)
    payload = module._result_payload(
        shard=shard, frame=frames, ranks=ranks, gram_residual=np.zeros(10),
        observer=observer, elapsed_seconds=1.0, config=config, hashes=hashes,
        selected_batch_size=10,
        benchmark={"projected_missing_1000_seconds": 1.0},
    )
    module.save_result_shard(
        output_root=tmp_path, scratch_root=tmp_path / "scratch", shard=shard,
        payload=payload, elapsed_seconds=1.0, config=config, hashes=hashes,
        selected_batch_size=10,
    )
    # Exercise the public loader with the unadorned source row from the JSON,
    # not only with ``source_rows``' convenience injection.  Provenance must
    # remain strict in both call paths.
    source = next(row for row in _config()["sources"] if row["cell"] == "nsh1_N28x24")
    ref = campaign.endpoint_ref(source, "soft", 4, tmp_path)
    loaded = campaign.load_endpoint(ref, 28, 24)
    np.testing.assert_allclose(loaded, frames[4], rtol=0.0, atol=0.0)


@pytest.mark.parametrize("field", ["execution_backend", "base_config_sha256", "source_hashes"])
def test_shard_rejects_wrong_backend_config_or_sources(tmp_path: Path, field: str) -> None:
    config = _config()
    source = _source(config, "nsh1_N28x24")
    result = _write_shard(tmp_path, source)
    completion_path = result.with_suffix(".completion.json")
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion[field] = "wrong" if field != "source_hashes" else {"runner": "0" * 64}
    completion_path.write_text(json.dumps(completion), encoding="utf-8")
    with pytest.raises(RuntimeError, match="backend is not accepted|completion .* mismatch"):
        campaign.endpoint_ref(source, "soft", 0, tmp_path)


def test_shard_rejects_self_consistent_wrong_dtype(tmp_path: Path) -> None:
    config = _config()
    source = _source(config, "nsh1_N28x24")
    result = _write_shard(tmp_path, source)
    with np.load(result, allow_pickle=False) as saved:
        payload = {key: np.array(saved[key], copy=True) for key in saved.files}
    metadata = json.loads(str(payload["metadata_json"].item()))
    metadata["dtype"] = "complex64"
    payload["metadata_json"] = np.asarray(campaign.canonical_json(metadata))
    np.savez(result, **payload)
    completion_path = result.with_suffix(".completion.json")
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion["result"] = {
        "name": result.name, "bytes": result.stat().st_size,
        "sha256": campaign.sha256_path(result),
    }
    completion_path.write_text(json.dumps(completion), encoding="utf-8")
    with pytest.raises(RuntimeError, match="scientific metadata mismatch"):
        campaign.endpoint_ref(source, "soft", 0, tmp_path)


def test_all_existing_per_sample_receipts_share_pinned_source_identity() -> None:
    config = _config()
    for source in config["sources"]:
        if source["kind"] != "per_sample":
            continue
        for wall in ("soft", "hard"):
            root = PILOT_ROOT / source["relative_root"] / wall
            completions = sorted(root.glob("sample_*.completion.json"))
            assert len(completions) == 100
            for path in completions:
                payload = json.loads(path.read_text(encoding="utf-8"))
                assert payload["config_hash"] == source["expected_config_hash"]
                assert payload["source_hashes"] == source["expected_source_hashes"]


def test_active_region_requires_every_locked_local_condition() -> None:
    config = _config()
    scan = {
        "internal": np.asarray([0.2, 0.01, 0.2]),
        "external": np.asarray([0.2, 0.3, 0.2]),
        "b_values": np.asarray([[-0.9, 0.9]] * 3),
        "wall_weights": np.asarray([[0.9, 0.9]] * 3),
        "neighbor_links": np.asarray([0.9, 0.9, 0.9]),
    }
    qualified = campaign._qualify_active_region(scan, config)
    assert qualified["resolved"]
    assert np.array_equal(qualified["active"], [False, True, False])
    for key, replacement, reason in (
        ("external", [0.2, 0.09, 0.2], "external_gap_below_0.1"),
        ("b_values", [[-0.9, 0.9], [-0.7, 0.9], [-0.9, 0.9]], "left_wall_character"),
        ("wall_weights", [[0.9, 0.9], [0.7, 0.9], [0.9, 0.9]], "radius2_wall_weight"),
        ("neighbor_links", [0.9, 0.7, 0.9], "neighboring_cluster_overlap"),
    ):
        changed = {name: np.array(value, copy=True) for name, value in scan.items()}
        changed[key] = np.asarray(replacement)
        rejected = campaign._qualify_active_region(changed, config)
        assert not rejected["resolved"]
        assert reason in rejected["reason"]


def test_rank4_qualification_uses_extreme_wall_eigenvalues() -> None:
    config = _config()
    scan = {
        "internal": np.asarray([0.02]),
        "external": np.asarray([0.4]),
        "b_values": np.asarray([[-0.95, -0.1, 0.2, 0.96]]),
        "wall_weights": np.asarray([[0.91, 0.2, 0.3, 0.92]]),
        "neighbor_links": np.asarray([0.99]),
    }
    qualified = campaign._qualify_active_region(scan, config)
    assert qualified["resolved"]
    assert np.array_equal(qualified["active"], [True])


def test_preregistered_control_and_sensitivity_tables_are_exact() -> None:
    config = _config()
    exact = controls.exact_control_tasks()
    sensitivity = controls.sensitivity_tasks(config)
    assert len(exact) == 12
    assert len(sensitivity) == 2050
    assert {row["variant"] for row in sensitivity} == set(controls.SENSITIVITY_VARIANTS)
    assert {
        row["sample_id"] for row in sensitivity
        if row["cell"] == "nsh1_N20x24" and row["wall"] == "hard"
    } == set(range(0, 100, 4)) | controls.HISTORICAL_FLIPS["hard"]
    assert {
        row["sample_id"] for row in sensitivity
        if row["cell"] == "nsh1_N20x24" and row["wall"] == "soft"
    } == set(range(0, 100, 4)) | controls.HISTORICAL_FLIPS["soft"]


def test_exact_control_rebuilds_the_ow_family_at_each_flux_and_conjugates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, float]] = []

    class FakeModel:
        def __init__(self, **kwargs):
            calls.append(("init", float(kwargs["twist_y"])))
            self.WF_Ap, self.WF_Bp, self.WF_Am, self.WF_Bm = range(4)

        def construct_OW_projectors(self, **kwargs):
            calls.append(("construct", float(kwargs["twist_y"])))

    matrices = {
        0: np.asarray([[1.0, 1.0j], [-1.0j, 0.0]]),
        1: np.asarray([[0.0, 0.2], [0.2, 1.0]]),
        2: np.asarray([[0.1, 0.0], [0.0, 0.2]]),
        3: np.asarray([[0.3, -0.1j], [0.1j, 0.1]]),
    }
    monkeypatch.setattr(controls, "classA_U1FGTN", FakeModel)
    monkeypatch.setattr(
        controls, "_frame_projector",
        lambda frame, _dimension, _rank: matrices[int(frame)],
    )
    direct = controls._exact_hamiltonian(
        wall="soft", source_kind="topological", phi=0.375
    )
    conjugated = controls._exact_hamiltonian(
        wall="soft", source_kind="conjugated", phi=0.375
    )
    assert calls == [
        ("init", 0.375), ("construct", 0.375),
        ("init", -0.375), ("construct", -0.375),
    ]
    expected = matrices[0] + matrices[1] - matrices[2] - matrices[3]
    np.testing.assert_allclose(direct, 0.5 * (expected + expected.conj().T))
    np.testing.assert_allclose(conjugated, direct.conj())


def test_exact_block_frame_matches_reconstructed_projector() -> None:
    eigenvectors = np.broadcast_to(
        np.eye(2 * controls.CONTROL_NX, dtype=np.complex128),
        (controls.CONTROL_NY, 2 * controls.CONTROL_NX, 2 * controls.CONTROL_NX),
    ).copy()
    occupied = np.zeros((controls.CONTROL_NY, 2 * controls.CONTROL_NX), dtype=bool)
    occupied[:, : controls.CONTROL_NX] = True
    frame = controls._block_frame(eigenvectors, occupied)
    block_projectors = np.stack([
        eigenvectors[k][:, occupied[k]] @ eigenvectors[k][:, occupied[k]].conj().T
        for k in range(controls.CONTROL_NY)
    ])
    expected = controls._reconstruct_from_blocks(block_projectors)
    np.testing.assert_allclose(frame @ frame.conj().T, expected, atol=3e-14)
    np.testing.assert_allclose(frame.conj().T @ frame, np.eye(800), atol=3e-14)


def test_authoritative_exact_control_reproduces_quantized_block_spectral_flow() -> None:
    if os.environ.get("RUN_WALL_DIABATIC_EXACT_CONTROL_TESTS") != "1":
        pytest.skip("set RUN_WALL_DIABATIC_EXACT_CONTROL_TESTS=1 for the exact OW rebuild test")
    config = _config()
    task = controls._control_task(
        "soft", "topological", "test_M8", 8, "uniform", "primary"
    )
    with threadpool_limits(limits=1):
        arrays = controls.compute_exact_control(task, config)
    np.testing.assert_allclose(
        arrays["q_x"][:, -1], np.asarray([1.0, -1.0]), rtol=0.0, atol=2e-7
    )
    assert np.max(np.abs(arrays["instantaneous_q_x"][:, -1])) < 2e-7
    assert np.max(arrays["continuation_undo_error"]) < 1e-10
    assert float(arrays["source_real_space_chern_mean"]) > 0.99


def test_sensitivity_discovery_verifies_one_endpoint_once_for_all_variants(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    config = _config()
    rows = [
        row for row in controls.sensitivity_tasks(config)
        if row["cell"] == "nsh1_N20x24"
        and row["wall"] == "soft"
        and row["sample_id"] == 0
    ]
    assert len(rows) == 5
    reference = campaign.EndpointRef(
        "frame", "completion", None, "a" * 64, 1, "b" * 64,
        "c" * 64, "schema",
    )
    calls: list[tuple[str, str, int, Path]] = []

    def fake_endpoint_ref(source, wall, sample_id, new_root):
        calls.append((source["cell"], wall, sample_id, new_root))
        return reference

    monkeypatch.setattr(campaign, "endpoint_ref", fake_endpoint_ref)
    refs, missing = controls.resolve_sensitivity_refs(
        rows, campaign.source_rows(config), tmp_path
    )
    assert not missing
    assert len(refs) == 5
    assert list(refs.values()) == [reference] * 5
    assert calls == [("nsh1_N20x24", "soft", 0, tmp_path)]


def test_gpu_port_receipt_requires_accuracy_and_twofold_speedup() -> None:
    base = {
        "schema": spectral_benchmark.SCHEMA, "status": "approved",
        "dtype": "complex128", "maximum_eigenvalue_error": 1e-11,
        "maximum_projector_error": 1e-11, "throughput_speedup_vs_local56": 2.01,
        "cpu_reference": "same-host-56-affinity-cpus", "cpu_threads": 56,
        "available_affinity_cpus": 56,
    }
    assert spectral_benchmark.approved(base)
    assert not spectral_benchmark.approved(
        {key: value for key, value in base.items() if key != "cpu_reference"}
    )
    assert not spectral_benchmark.approved({**base, "available_affinity_cpus": 12})
    assert not spectral_benchmark.approved({**base, "maximum_projector_error": 1.1e-10})
    assert not spectral_benchmark.approved({**base, "throughput_speedup_vs_local56": 1.99})


def test_polar_alignment_and_wall_modes_are_cluster_gauge_invariant() -> None:
    rng = np.random.default_rng(22)
    cluster = _orthonormal_frame(30, 4, 8)
    b = np.linspace(-1.0, 1.0, 30)
    rotation = _orthonormal_frame(4, 4, 9)
    values_a, modes_a = campaign._wall_modes(cluster, b, None, None)
    values_b, modes_b = campaign._wall_modes(cluster @ rotation, b, cluster, modes_a)
    np.testing.assert_allclose(values_b, values_a, atol=2e-14)
    np.testing.assert_allclose(
        modes_b @ modes_b.conj().T, modes_a @ modes_a.conj().T, atol=2e-14
    )


def test_source_chern_matches_existing_legacy_frame_estimator() -> None:
    torch = pytest.importorskip("torch")
    import analyze_ny24_ny28_bulk_chern_pump_relation as legacy

    path = (
        PILOT_ROOT / "results/N20x24_state_projector_pump_s100_v1"
        / "burnins/soft/sample_000.npz"
    )
    with np.load(path, allow_pickle=False) as saved:
        frame = np.array(saved["frame"], dtype=np.complex128, copy=True)
    observed = campaign.real_space_chern_by_y0(frame, 20, 24, 4.0)
    expected = legacy.chern_by_y0(frame, legacy.batched_partitions(24, 4.0))
    np.testing.assert_allclose(observed, expected, rtol=0.0, atol=2e-12)
    assert torch.get_num_threads() >= 1


@pytest.mark.parametrize("edge_rank", [2, 4])
@pytest.mark.parametrize("twist_gauge", ["uniform", "seam"])
def test_small_exact_frame_has_finite_controls_and_true_undo(
    edge_rank: int, twist_gauge: str
) -> None:
    config = _config()
    nx = ny = 8
    dimension, rank = 2 * nx * ny, nx * ny
    frame = np.eye(dimension, rank, dtype=np.complex128)
    task = {
        "Nx": nx, "Ny": ny, "grid_intervals": 4,
        "edge_block_rank": edge_rank, "wall_window": 2,
        "twist_gauge": twist_gauge,
    }
    with threadpool_limits(limits=1):
        arrays = campaign.compute_wall_diabatic_pump(frame, task, config, grid_intervals=4)
    assert arrays["phi"].shape == (2, 5)
    assert arrays["edge_B_eigenvalues"].shape == (2, 5, edge_rank)
    assert arrays["edge_entering_wall_labels"].shape == (2, edge_rank // 2)
    assert np.max(arrays["continuation_undo_error"]) < 1e-12
    assert np.max(np.abs(arrays["q_x"])) < 1e-12
    assert np.all(np.isfinite(arrays["endpoint_defect_eigenvalues"]))


@pytest.mark.parametrize("intervals", [128, 256, 512])
@pytest.mark.parametrize(
    ("wall", "expected"),
    [
        ("soft", np.asarray([0.999978709, -0.999981497])),
        ("hard", np.asarray([0.998936877, -0.999068176])),
    ],
)
def test_real_n20_sample0_is_stable_at_m128_m256_m512(
    intervals: int, wall: str, expected: np.ndarray
) -> None:
    if os.environ.get("RUN_WALL_DIABATIC_REAL_ENDPOINT_TESTS") != "1":
        pytest.skip("set RUN_WALL_DIABATIC_REAL_ENDPOINT_TESTS=1 for the multi-hour endpoint test")
    config = _config()
    task = next(
        row for row in campaign.tasks(config)
        if row["cell"] == "nsh1_N20x24" and row["wall"] == wall and row["sample_id"] == 0
    )
    source = _source(config, task["cell"])
    ref = campaign.endpoint_ref(source, wall, 0, campaign.DEFAULT_NEW_ENDPOINT_ROOT)
    frame = campaign.load_endpoint(ref, 20, 24)
    with threadpool_limits(limits=1):
        arrays = campaign.compute_wall_diabatic_pump(
            frame, task, config, grid_intervals=intervals
        )
    assert np.all(arrays["resolved"])
    np.testing.assert_allclose(arrays["q_x"][:, -1], expected, rtol=0.0, atol=2e-6)
    assert np.max(arrays["continuation_undo_error"]) <= 1e-8


def test_wall_diabatic_documentation_covers_full_campaign_contract() -> None:
    readme = (PILOT_ROOT / "README.md").read_text(encoding="utf-8")
    runbook = (PILOT_ROOT / "WALL_DIABATIC_WIDTH_SWEEP_RUNBOOK.md").read_text(
        encoding="utf-8"
    )
    methods = (
        PILOT_ROOT / "docs/wall_diabatic_spectral_pump_methods.tex"
    ).read_text(encoding="utf-8")
    dictionary = (PILOT_ROOT / "WALL_DIABATIC_DATA_DICTIONARY.md").read_text(
        encoding="utf-8"
    )
    assert "Current recommended calculation: wall-diabatized width sweep" in readme
    normalized_readme = " ".join(readme.split()).lower()
    assert "all 230 five-trajectory shards were imported" in normalized_readme
    assert "rebuilds the ow functions and `h_exact(phi)`" in normalized_readme
    normalized_runbook = " ".join(runbook.split()).lower()
    for required in (
        "Locked data lineage and workload",
        "600 immutable CPU endpoints",
        "1,000 new endpoints",
        "1,600 primary endpoint pairs",
        "2,050 sensitivity pairs",
        "unresolved_reason",
        "No `q_x^odd` replacement is used",
        "Local spectral result layout",
        "Spectral interruption cost",
        "Final acceptance checklist",
        "Recovery guide",
    ):
        assert required.lower() in normalized_runbook
    for required in (
        r"\section{Reproducibility, durability, and present status}",
        "wall_pump_width_endpoints_s100_v1",
        "root seed",
        "No production",
        "corresponding atomic outputs verify",
    ):
        assert required in methods
    for required in (
        "A100 endpoint files",
        "wall_pump_width_endpoint_shard_v1",
        "Rolling checkpoint",
        "CPU-fallback endpoint files",
        "Wall-diabatized spectral pair",
        "wall_diabatic_spectral_pump_result_v1",
        "edge_B_eigenvalues",
        "endpoint_defect_eigenvalues",
        "Control and sensitivity collections",
        "Analysis products",
        "Status vocabulary",
    ):
        assert required in dictionary
