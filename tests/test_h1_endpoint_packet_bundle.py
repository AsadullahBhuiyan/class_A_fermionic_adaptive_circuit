from __future__ import annotations

import hashlib
import io
import json
import sys
import tarfile
from pathlib import Path

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "00_WORKSPACE/CURRENT/final_production_new_designs/08_h1_endpoint_packet"
SRC = BUNDLE / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from h1_packet_observables import (  # noqa: E402
    H1EndpointPacketObserver,
    linear_fit_last_axis,
    localized_source_matrix,
    packet_probabilities_from_eigensystem,
    paired_endpoint_drift,
    translated_half_indices,
    wall_columns,
)
import h1_packet_observables as packet_observables  # noqa: E402
from h1_packet_runner import (  # noqa: E402
    _find_replay_source,
    expand_cases,
    load_config,
    shard_seed,
)
import h1_packet_analysis as packet_analysis  # noqa: E402
import h1_packet_runner as packet_runner  # noqa: E402
from h1_v3_migration import load_allowlist  # noqa: E402


def test_v4_migration_allowlist_pins_twelve_and_reuses_eleven() -> None:
    ledger = load_allowlist(BUNDLE)
    accepted = ledger["accepted_archives"]
    assert len(accepted) == 12
    assert len(ledger["rejected_receipt_only_run_ids"]) == 8
    assert sum(bool(row["reuse_in_v4"]) for row in accepted) == 11
    forced = ledger["qualification_rerun"]
    source = next(
        row for row in accepted
        if row["case_id"] == forced["case_id"]
        and row["shard_index"] == forced["shard_index"]
    )
    assert source["reuse_in_v4"] is False


def _random_frame(dimension: int, rank: int, seed: int = 917) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    raw = torch.complex(
        torch.randn((dimension, rank), generator=generator, dtype=torch.float64),
        torch.randn((dimension, rank), generator=generator, dtype=torch.float64),
    )
    return torch.linalg.qr(raw, mode="reduced").Q


def _v3_numerical_kwargs() -> dict[str, float]:
    return {
        "raw_norm_warning_tolerance": 1e-8,
        "raw_norm_hard_failure_tolerance": 1e-6,
        "post_normalization_norm_tolerance": 1e-12,
        "gram_diagnostic_trigger": 1e-10,
    }


def _small_v3_observer() -> H1EndpointPacketObserver:
    return H1EndpointPacketObserver(
        nx=4,
        ny=4,
        wall_x=[1, 3],
        checkpoints=[1],
        global_sample_ids=[7],
        modular_times=[0.0, 0.1, 0.2],
        spectral_clip_eps=[1e-8],
        source_widths=[1, 3],
        retention_widths=[1, 3],
        primary_epsilon=1e-8,
        primary_source_width=3,
        primary_retention_width=3,
        fixed_time=0.1,
        fit_windows=[[0.0, 0.2]],
        orientation_signs=[-1, 1],
        minimum_primary_retention=0.0,
        **_v3_numerical_kwargs(),
    )


def _scaled_uniform_packet(scales: list[float]):
    def packet(
        *, vectors: torch.Tensor, source_matrix: torch.Tensor,
        modular_times: torch.Tensor, nx: int, ay: int, **_kwargs,
    ) -> torch.Tensor:
        assert len(scales) == len(modular_times)
        probability = torch.ones(
            (len(modular_times), ay, nx, source_matrix.shape[1]),
            dtype=vectors.real.dtype,
            device=vectors.device,
        )
        probability /= float(ay * nx)
        scale = torch.as_tensor(
            scales, dtype=vectors.real.dtype, device=vectors.device
        )
        return probability * scale[:, None, None, None]

    return packet


def test_locked_endpoint_packet_matrix_and_retired_response() -> None:
    config = load_config(BUNDLE)
    cases = expand_cases(config)
    assert len(cases) == 4
    assert {(row["protocol"], row["model"]["alpha_1"]) for row in cases} == {
        (protocol, alpha)
        for protocol in ("hard", "soft") for alpha in (1.0, 3.0)
    }
    assert config["packet_observer"]["wall_orientation_signs"] == [-1, 1]
    assert config["packet_observer"]["source_width_columns"] == [1, 3]
    assert config["packet_observer"]["retention_width_columns"] == [1, 3, 5]
    assert config["packet_observer"]["fixed_time"] == 2.0
    assert "signed_retarded_modular_response" in config["retired_products"]
    assert "signed_retarded_modular_response" not in config["raw_products"]
    for case in cases:
        assert case["run"]["cycles"] == 80
        assert case["run"]["samples"] == 25
        assert case["model"]["Nx"] == 20 and case["model"]["Ny"] == 40
        assert case["model"]["nshell"] == 1
        assert case["model"]["dtype"] == "complex128"
        assert case["model"]["dw_truncation"] is (case["protocol"] == "hard")
        assert case["run"]["meas_slab_only"] is (case["protocol"] == "hard")


def test_exact_benchmark_and_source_manifest_are_pinned() -> None:
    benchmark = json.loads((BUNDLE / "exact_benchmark_provenance.json").read_text())
    assert benchmark["wall_orientation_signs"] == [-1, 1]
    assert benchmark["results"]["topological_coupled"]["wall_delta_at_t2"][0] < 0
    assert benchmark["results"]["topological_coupled"]["wall_delta_at_t2"][1] > 0
    assert abs(benchmark["results"]["trivial_coupled"]["oriented_H_delta_at_t2"]) < 0.05
    manifest = json.loads((SRC / "source_manifest.json").read_text())
    for name, row in manifest["files"].items():
        assert hashlib.sha256((SRC / name).read_bytes()).hexdigest() == row["sha256"]
    assert (SRC / "classA_U1FGTN_gpu.py").read_bytes() == (
        ROOT / "00_WORKSPACE/CURRENT/src/fgtn/classA_U1FGTN_gpu.py"
    ).read_bytes()
    notebook = json.loads((BUNDLE / "run_production_bundle.ipynb").read_text())
    assert "".join(notebook["cells"][-1]["source"]) == (
        "from google.colab import runtime\n"
        "runtime.unassign()\n"
        "print('done')\n"
    )


def test_translated_cut_order_wraps_and_source_support_is_exact() -> None:
    nx, ny = 4, 6
    indices = translated_half_indices(nx=nx, ny=ny, cut_origin=5).tolist()
    expected = [
        mu + 2 * x + 2 * nx * y
        for y in (5, 0, 1) for x in range(nx) for mu in (0, 1)
    ]
    assert indices == expected
    np.testing.assert_array_equal(wall_columns(0, 6, 3), [5, 0, 1])
    sources, source_index = localized_source_matrix(
        nx=6, ay=3, walls=[1, 4], source_widths=[1, 3]
    )
    assert sources.shape == (36, 8)
    torch.testing.assert_close(
        torch.sum(torch.abs(sources) ** 2, dim=0),
        torch.ones(8, dtype=torch.float64),
    )
    assert source_index.tolist() == [
        [width, wall, endpoint]
        for width in range(2) for wall in range(2) for endpoint in range(2)
    ]
    assert torch.count_nonzero(sources[:, 0]).item() == 2
    assert torch.count_nonzero(sources[:, 4]).item() == 6


def test_paired_drift_orientation_and_velocity_fit() -> None:
    times = np.linspace(0.0, 2.0, 41)
    centers = np.empty((2, 2, len(times)))
    centers[0, 0] = 0.0 - 0.8 * times
    centers[0, 1] = 19.0 - 0.8 * times
    centers[1, 0] = 0.0 + 1.1 * times
    centers[1, 1] = 19.0 + 1.1 * times
    drift = paired_endpoint_drift(centers, ay=20)
    slopes, r2, count = linear_fit_last_axis(times, drift, (0.1, 2.0))
    np.testing.assert_allclose(slopes, [-0.8, 1.1], atol=2e-14)
    np.testing.assert_allclose(r2, 1.0)
    assert count == 39
    handed_delta = np.mean(
        (drift[:, -1] - drift[:, 0]) * np.asarray([-1, 1])
    )
    assert handed_delta > 0.0


def test_primary_spectral_cutoff_selection_uses_relative_precision() -> None:
    values = np.asarray([1e-8, 1e-10, 1e-12], dtype=np.float64)
    assert H1EndpointPacketObserver._unique_index(values, 1e-8) == 0
    assert H1EndpointPacketObserver._unique_index(values, 1e-10) == 1
    assert H1EndpointPacketObserver._unique_index(values, 1e-12) == 2


def test_packet_propagation_matches_dense_matrix_exponential() -> None:
    dimension, nx, ay = 12, 3, 2
    vectors = _random_frame(dimension, dimension, seed=88)
    occupations = torch.linspace(0.1, 0.9, dimension, dtype=torch.float64)
    sources, _ = localized_source_matrix(
        nx=nx, ay=ay, walls=[0, 2], source_widths=[1]
    )
    times = torch.tensor([0.0, 0.3, 1.0], dtype=torch.float64)
    actual = packet_probabilities_from_eigensystem(
        occupations=occupations,
        vectors=vectors,
        source_matrix=sources,
        modular_times=times,
        epsilon=1e-10,
        nx=nx,
        ay=ay,
    )
    energies = torch.log((1.0 - occupations) / occupations)
    hamiltonian = vectors @ torch.diag(energies.to(torch.complex128)) @ vectors.mH
    expected = []
    for time in times:
        evolved = torch.matrix_exp(-1j * time * hamiltonian) @ sources
        expected.append(
            evolved.abs().square().reshape(ay, nx, 2, 4).sum(dim=2)
        )
    torch.testing.assert_close(actual, torch.stack(expected), atol=3e-12, rtol=3e-12)
    torch.testing.assert_close(
        actual.sum(dim=(1, 2)), torch.ones((3, 4), dtype=torch.float64),
        atol=3e-12, rtol=3e-12,
    )


def test_packet_kernel_reproduces_exact_domain_wall_endpoint_drift() -> None:
    benchmark_root = (
        ROOT / "00_WORKSPACE/CURRENT/experiment_review/b0_exact_domain_wall"
    )
    sys.path.insert(0, str(benchmark_root))
    try:
        from b0lib import (  # type: ignore
            correlation_displacements,
            load_locked_config,
            make_model,
            modular_packet,
            occupied_blocks,
            strip_correlation,
        )
    finally:
        sys.path.remove(str(benchmark_root))
    config, _, _ = load_locked_config(benchmark_root / "campaign_config.v2.json")
    config = dict(config)
    config["modular_time_max"] = 2.5
    model = make_model(20, 40, config)
    _, _, projectors = occupied_blocks(
        model, "hard_exterior", config["occupation_twist"]
    )
    displacements = correlation_displacements(
        projectors, config["occupation_twist"]
    )
    legacy, _, _ = modular_packet(
        displacements, model, "hard_exterior", config
    )
    correlation = strip_correlation(displacements, 20)
    occupations, vectors = np.linalg.eigh(correlation)
    sources, source_index = localized_source_matrix(
        nx=20, ay=20, walls=model.DW_loc, source_widths=[1, 3]
    )
    probabilities = packet_probabilities_from_eigensystem(
        occupations=torch.as_tensor(occupations),
        vectors=torch.as_tensor(vectors),
        source_matrix=sources,
        modular_times=torch.as_tensor(legacy["modular_times"]),
        epsilon=1e-10,
        nx=20,
        ay=20,
    ).numpy()
    centers = np.empty((2, 2, 2, len(legacy["modular_times"])))
    for source_column, (width_index, wall_index, endpoint_index) in enumerate(source_index):
        columns = wall_columns(
            model.DW_loc[wall_index], 20, [1, 3][width_index]
        )
        profile = probabilities[:, :, columns, source_column].sum(axis=2)
        retained = profile.sum(axis=1)
        centers[width_index, wall_index, endpoint_index] = (
            profile @ np.arange(20)
        ) / retained
    actual = paired_endpoint_drift(centers, ay=20)
    np.testing.assert_allclose(
        actual, legacy["modular_handedness"], atol=2e-12, rtol=2e-12
    )


def test_small_observer_constructs_every_cut_before_reduction(tmp_path: Path) -> None:
    nx, ny = 4, 4
    dimension = 2 * nx * ny
    frame = _random_frame(dimension, dimension // 2).unsqueeze(0)
    state = type("FrameState", (), {
        "frame": frame,
        "ranks": torch.tensor([dimension // 2], dtype=torch.long),
    })()
    times = np.asarray([0.0, 0.1, 0.2])
    observer = H1EndpointPacketObserver(
        nx=nx,
        ny=ny,
        wall_x=[1, 3],
        checkpoints=[1],
        global_sample_ids=[7],
        modular_times=times,
        spectral_clip_eps=[1e-8],
        source_widths=[1, 3],
        retention_widths=[1, 3],
        primary_epsilon=1e-8,
        primary_source_width=3,
        primary_retention_width=3,
        fixed_time=0.1,
        fit_windows=[[0.0, 0.2]],
        orientation_signs=[-1, 1],
        minimum_primary_retention=0.0,
        **_v3_numerical_kwargs(),
    )
    observer(cycle=1, state=state, batch_start=0, batch_count=1)
    assert observer.seen.all()
    assert observer.paired_drift.shape == (1, 1, 4, 1, 2, 2, 2, 3)
    expected = paired_endpoint_drift(observer.primary_endpoint_centers, ay=2)
    np.testing.assert_allclose(
        observer.paired_drift[0, 0, :, 0, 1, 1], expected[0, 0]
    )
    output = tmp_path / "packet"
    product = observer.save(output, config={"test": True})
    assert product["translated_cuts_per_trajectory_checkpoint"] == ny
    assert product["dtype_contract_failure"] is False
    for path in output.glob("*.npz"):
        with np.load(path, allow_pickle=False) as data:
            assert not any("retarded" in key.lower() for key in data.files)

def test_v3_raw_norm_warning_saves_and_normalizes_before_science(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        packet_observables,
        "packet_probabilities_from_eigensystem",
        _scaled_uniform_packet([1.0, 1.0 + 5e-8, 1.0 - 2e-8]),
    )
    dimension = 2 * 4 * 4
    state = type("FrameState", (), {
        "frame": _random_frame(dimension, dimension // 2).unsqueeze(0),
        "ranks": torch.tensor([dimension // 2], dtype=torch.long),
    })()
    observer = _small_v3_observer()
    observer(cycle=1, state=state, batch_start=0, batch_count=1)

    output = tmp_path / "warning"
    product = observer.save(output, config={"test": "warning"})
    assert product["numerical_status"] == "warning"
    assert product["raw_norm_warning"] is True
    assert product["raw_norm_hard_failure"] is False
    assert 1e-8 < product["maximum_raw_packet_norm_error"] < 1e-6
    assert (
        product["maximum_post_normalization_norm_error"]
        <= product["post_normalization_norm_tolerance"]
    )
    assert product["raw_norm_error_argmax"]["error"] == pytest.approx(5e-8)
    np.testing.assert_allclose(
        observer.primary_endpoint_retention,
        0.75,
        rtol=0.0,
        atol=2e-15,
    )

    with np.load(output / "packet_drift.npz", allow_pickle=False) as data:
        assert {
            "raw_packet_total_norm",
            "maximum_raw_packet_norm_error",
            "maximum_raw_packet_norm_drift",
            "maximum_post_normalization_norm_error",
            "raw_norm_error_argmax_json",
            "numerical_status",
        } <= set(data.files)
        assert str(data["numerical_status"].item()) == "warning"
    with np.load(output / "common.npz", allow_pickle=False) as data:
        assert {
            "conditional_eigenvector_gram_residual",
            "conditional_eigenvector_gram_residual_computed",
            "actual_dtype",
            "actual_probability_dtype",
        } <= set(data.files)
        assert np.any(data["conditional_eigenvector_gram_residual_computed"])


def test_v3_dtype_regression_is_a_persisted_hard_failure(tmp_path: Path) -> None:
    dimension = 2 * 4 * 4
    state = type("FrameState", (), {
        "frame": _random_frame(dimension, dimension // 2).unsqueeze(0),
        "ranks": torch.tensor([dimension // 2], dtype=torch.long),
    })()
    observer = _small_v3_observer()
    observer(cycle=1, state=state, batch_start=0, batch_count=1)
    observer.actual_frame_dtype = "torch.complex64"

    output = tmp_path / "dtype_failure"
    product = observer.save(output, config={"test": "dtype_failure"})
    assert product["numerical_status"] == "hard_failure"
    assert product["dtype_contract_failure"] is True
    assert {path.name for path in output.glob("*.npz")} == {
        "common.npz",
        "packet_drift.npz",
        "primary_profiles.npz",
    }
    with np.load(output / "packet_drift.npz", allow_pickle=False) as data:
        assert bool(data["dtype_contract_failure"].item()) is True


def test_v3_raw_norm_hard_ceiling_is_classified_after_raw_files_persist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        packet_observables,
        "packet_probabilities_from_eigensystem",
        _scaled_uniform_packet([1.0, 1.0 + 2e-6, 1.0]),
    )
    dimension = 2 * 4 * 4
    state = type("FrameState", (), {
        "frame": _random_frame(dimension, dimension // 2).unsqueeze(0),
        "ranks": torch.tensor([dimension // 2], dtype=torch.long),
    })()
    observer = _small_v3_observer()
    observer(cycle=1, state=state, batch_start=0, batch_count=1)

    output = tmp_path / "hard"
    product = observer.save(output, config={"test": "hard"})
    assert product["numerical_status"] == "hard_failure"
    assert product["raw_norm_warning"] is True
    assert product["raw_norm_hard_failure"] is True
    assert product["post_normalization_norm_failure"] is False
    assert product["maximum_raw_packet_norm_error"] > 1e-6
    assert {path.name for path in output.glob("*.npz")} == {
        "common.npz",
        "packet_drift.npz",
        "primary_profiles.npz",
    }
    with np.load(output / "packet_drift.npz", allow_pickle=False) as data:
        assert str(data["numerical_status"].item()) == "hard_failure"
        assert bool(data["raw_norm_hard_failure"].item()) is True


def test_production_size_one_cut_packet_normalization_calibration_on_cpu() -> None:
    nx, ny, ay = 20, 40, 20
    subsystem_modes = 2 * nx * ay
    indices = translated_half_indices(nx=nx, ny=ny, cut_origin=17)
    assert indices.shape == (subsystem_modes,)
    assert torch.unique(indices).numel() == subsystem_modes

    occupations = torch.linspace(
        0.01, 0.99, subsystem_modes, dtype=torch.float64
    )
    vectors = torch.eye(subsystem_modes, dtype=torch.complex128)
    sources, _ = localized_source_matrix(
        nx=nx, ay=ay, walls=[5, 15], source_widths=[1]
    )
    times = torch.tensor([0.0, 0.05, 2.0, 8.0], dtype=torch.float64)
    raw = packet_probabilities_from_eigensystem(
        occupations=occupations,
        vectors=vectors,
        source_matrix=sources,
        modular_times=times,
        epsilon=1e-10,
        nx=nx,
        ay=ay,
    )
    assert raw.shape == (len(times), ay, nx, 4)
    raw_norm = raw.sum(dim=(1, 2))
    normalized = raw / raw_norm[:, None, None, :]
    post_error = torch.max(torch.abs(normalized.sum(dim=(1, 2)) - 1.0))
    assert float(post_error) <= 1e-12



def _npz_payload(**arrays: np.ndarray) -> bytes:
    handle = io.BytesIO()
    np.savez_compressed(handle, **arrays)
    return handle.getvalue()


def _add_member(archive: tarfile.TarFile, name: str, payload: bytes) -> None:
    member = tarfile.TarInfo(name)
    member.size = len(payload)
    archive.addfile(member, io.BytesIO(payload))


def test_response_replay_requires_receipt_engine_and_physical_contract(tmp_path: Path) -> None:
    config = load_config(BUNDLE)
    case = expand_cases(config)[1]
    source_case = dict(case)
    source_case["campaign"] = "H1_MODULAR_RESPONSE"
    engine_hash = hashlib.sha256((SRC / "classA_U1FGTN_gpu.py").read_bytes()).hexdigest()
    source_root = (
        tmp_path / config["trajectory_reuse"]["source_collection"]
        / config["trajectory_reuse"]["source_bundle"]
    )
    source_root.mkdir(parents=True)
    site_ids = np.arange(6, dtype=np.int32)[None, None].repeat(5, 0).repeat(80, 1)
    outcomes = np.zeros((5, 80, 6, 4), dtype=np.uint8)
    record = _npz_payload(
        schema=np.asarray("ordered_site_channel_born_record_v2_compact_replay"),
        site_ids=site_ids,
        channel_count=np.ones((5, 80, 6), dtype=np.uint8),
        outcome_bits_packed=np.packbits(outcomes, axis=-1, bitorder="little"),
        target_bits_packed=np.packbits(outcomes, axis=-1, bitorder="little"),
        bit_shape=np.asarray(outcomes.shape),
        cumulative_self_information=np.zeros((5, 80)),
    )
    rng = _npz_payload(
        torch_cpu_rng_state=torch.get_rng_state().numpy(),
        numpy_state_json=np.asarray(json.dumps([
            "MT19937", np.random.get_state()[1].tolist(), 0, 0, 0.0
        ])),
    )
    manifest = {
        "status": "complete_local",
        "bundle": "03_h1_modular_response",
        "sampling_revision": "production_25sample_h1_modular_response_v1",
        "canonical_engine_sha256": engine_hash,
        "root_seed": config["root_seed"],
        "shard_generator_seed": shard_seed(config["root_seed"], case["case_id"], 0),
        "case_id": case["case_id"],
        "shard_index": 0,
        "global_sample_indices": list(range(5)),
        "run_config": {"case": source_case},
    }
    path = source_root / "source.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        _add_member(archive, "manifest.json", json.dumps(manifest).encode())
        _add_member(archive, "shards/shard_000/ordered_born_record.npz", record)
        _add_member(archive, "shards/shard_000/rng_before.npz", rng)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix(path.suffix + ".receipt.json").write_text(json.dumps({
        "archive": path.name, "archive_sha256": digest,
    }))
    source, rejected = _find_replay_source(
        drive_root=tmp_path,
        config=config,
        case=case,
        shard_index=0,
        engine_hash=engine_hash,
        mode="production",
    )
    assert source is not None and not rejected
    np.testing.assert_array_equal(source["record"]["site_ids"], site_ids)
    stale, reasons = _find_replay_source(
        drive_root=tmp_path,
        config=config,
        case=case,
        shard_index=0,
        engine_hash="0" * 64,
        mode="production",
    )
    assert stale is None
    assert any("canonical_engine_sha256" in reason for reason in reasons)


def test_analysis_reduces_cuts_inside_trajectory_and_accepts_synthetic_signal(
    tmp_path: Path, monkeypatch,
) -> None:
    config = load_config(BUNDLE)
    cases = {case["case_id"]: case for case in expand_cases(config)}
    samples, checkpoints, cuts = 25, 6, 2
    neps, nsource, nretention, nfit, nwalls, ntimes = 3, 2, 3, 4, 2, 3
    merged = {}
    for case_id, case in cases.items():
        topological = case["model"]["alpha_1"] == 1.0
        wall_delta = np.empty(
            (samples, checkpoints, cuts, neps, nsource, nretention, nwalls)
        )
        wall_delta[..., 0] = -1.0 if topological else -0.005
        wall_delta[..., 1] = 1.0 if topological else 0.005
        orientation = np.asarray([-1, 1])
        handed = np.mean(wall_delta * orientation, axis=-1)
        velocity = np.repeat(wall_delta[..., None, :], nfit, axis=-2)
        handed_velocity = np.mean(velocity * orientation, axis=-1)
        paired = np.empty(wall_delta.shape + (ntimes,))
        paired[..., 0] = 0.0
        paired[..., 1] = 0.5 * wall_delta
        paired[..., 2] = wall_delta
        merged[case_id] = {
            "case": case,
            "sample_ids": np.arange(samples),
            "checkpoints": np.asarray([40, 48, 56, 64, 72, 80]),
            "cut_origins": np.arange(cuts),
            "modular_times": np.asarray([0.0, 1.0, 2.0]),
            "wall_x": np.asarray([5, 15]),
            "wall_orientation_signs": orientation,
            "source_width_columns": np.asarray([1, 3]),
            "retention_width_columns": np.asarray([1, 3, 5]),
            "spectral_clip_eps": np.asarray([1e-8, 1e-10, 1e-12]),
            "fit_windows": np.asarray([
                [0.1, 2.0], [0.05, 1.5], [0.2, 2.0], [0.1, 2.5]
            ]),
            "primary_epsilon_index": np.asarray(1),
            "primary_source_width_index": np.asarray(1),
            "primary_retention_width_index": np.asarray(1),
            "fixed_time_index": np.asarray(2),
            "source_index": np.zeros((8, 3), dtype=np.int64),
            "spectral_clip_counts": np.zeros(
                (samples, checkpoints, cuts, neps, 2), dtype=np.int32
            ),
            "restricted_correlation_hermiticity_error": np.zeros(
                (samples, checkpoints, cuts)
            ),
            "conditional_eigenvector_gram_residual": np.zeros(
                (samples, checkpoints, cuts)
            ),
            "conditional_eigenvector_gram_residual_computed": np.zeros(
                (samples, checkpoints, cuts), dtype=np.bool_
            ),
            "observer_seconds": np.ones((samples, checkpoints)),
            "paired_endpoint_wall_drift": paired,
            "wall_delta_at_fixed_time": wall_delta,
            "oriented_handed_delta": handed,
            "wall_velocity": velocity,
            "wall_velocity_r2": np.full_like(velocity, 0.95),
            "oriented_handed_velocity": handed_velocity,
            "minimum_wall_retention": np.ones_like(wall_delta),
            "maximum_packet_norm_drift": np.zeros(
                (samples, checkpoints, cuts, neps, nsource, nwalls, 2)
            ),
            "raw_packet_total_norm": np.ones(
                (
                    samples, checkpoints, cuts, neps, nsource,
                    nwalls, 2, ntimes,
                )
            ),
            "maximum_raw_packet_norm_error": np.zeros(
                (samples, checkpoints, cuts, neps, nsource, nwalls, 2)
            ),
            "maximum_raw_packet_norm_drift": np.zeros(
                (samples, checkpoints, cuts, neps, nsource, nwalls, 2)
            ),
            "maximum_post_normalization_norm_error": np.zeros(
                (samples, checkpoints, cuts, neps, nsource, nwalls, 2)
            ),
            "actual_dtype": np.asarray("torch.complex128"),
            "actual_probability_dtype": np.asarray("torch.float64"),
            "primary_endpoint_center": np.zeros(
                (samples, checkpoints, cuts, nwalls, 2, ntimes)
            ),
            "primary_endpoint_retention": np.ones(
                (samples, checkpoints, cuts, nwalls, 2, ntimes)
            ),
            "primary_cut_mean_conditional_profile": np.ones(
                (samples, checkpoints, nwalls, 2, ntimes, 2)
            ) / 2,
            "archives": [],
            "trajectory_provenance": [
                {"mode": "fresh_same_preregistered_seed"} for _ in range(5)
            ],
        }
    monkeypatch.setattr(packet_analysis, "merge_archives", lambda _: merged)
    result = packet_analysis.analyze(
        tmp_path / "archives", tmp_path / "analysis", bundle_root=BUNDLE
    )
    assert result["status"] == "accepted"
    assert all(row["pass"] for row in result["gates"])
    assert result["secondary_velocity_diagnostics"]["hard"]["report_velocity"]
    assert (tmp_path / "analysis/h1_endpoint_analysis_summary.json").is_file()


def test_full_shard_a100_qualification_becomes_real_scientific_archive(
    tmp_path: Path, monkeypatch,
) -> None:
    config = load_config(BUNDLE)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    packet_product = _fake_v3_product(
        scratch / "h1_endpoint_packet", status="pass"
    )
    manifest = {
        "gpu_preflight": {
            "device": "NVIDIA A100-SXM4-40GB",
            "total_bytes": 40 * 1024**3,
        },
        "gpu_peak_reserved_bytes": 2 * 1024**3,
        "products": {
            "h1_endpoint_packet": packet_product,
            "ordered_born_record": {"bytes": 1000},
        },
    }
    monkeypatch.setattr(packet_runner, "_verify_existing", lambda _: None)
    monkeypatch.setattr(
        packet_runner,
        "run_case",
        lambda **_: {
            "status": "preflight_complete",
            "manifest": manifest,
            "scratch": str(scratch),
        },
    )
    archived = []

    def fake_archive(source: Path, target: Path, run_id: str):
        archived.append((source, target, run_id))
        return {"archive": target.name, "archive_sha256": "a" * 64}

    monkeypatch.setattr(packet_runner, "_archive", fake_archive)
    result = packet_runner.a100_preflight(
        bundle_root=BUNDLE, config=config, drive_root=tmp_path
    )
    assert result["safe"] is True
    assert result["scientific_shard_archived"] is True
    assert result["qualification_archive_sha256"] == "a" * 64
    assert archived and archived[0][0] == scratch


def _fake_v3_product(directory: Path, *, status: str) -> dict[str, object]:
    directory.mkdir(parents=True, exist_ok=True)
    hard = status == "hard_failure"
    warning = status in {"warning", "hard_failure"}
    raw_error = 2e-6 if hard else (5e-8 if warning else 0.0)
    arrays = {
        "common.npz": {
            "conditional_eigenvector_gram_residual": np.asarray([0.0]),
            "conditional_eigenvector_gram_residual_computed": np.asarray([False]),
        },
        "packet_drift.npz": {
            "maximum_raw_packet_norm_error": np.asarray(raw_error),
            "maximum_raw_packet_norm_drift": np.asarray(raw_error),
            "maximum_post_normalization_norm_error": np.asarray(2e-16),
            "numerical_status": np.asarray(status),
            "raw_norm_warning": np.asarray(warning),
            "raw_norm_hard_failure": np.asarray(hard),
        },
        "primary_profiles.npz": {
            "primary_endpoint_retention": np.ones((1,), dtype=np.float64),
        },
    }
    file_rows = []
    total_bytes = 0
    for name, payload in arrays.items():
        path = directory / name
        np.savez_compressed(path, **payload)
        total_bytes += path.stat().st_size
        file_rows.append({
            "path": name,
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })
    return {
        "schema": "h1_endpoint_packet_v3",
        "files": file_rows,
        "bytes": total_bytes,
        "numerical_status": status,
        "raw_norm_warning": warning,
        "raw_norm_hard_failure": hard,
        "post_normalization_norm_failure": False,
        "hermiticity_failure": False,
        "dtype_contract_failure": False,
        "maximum_raw_packet_norm_error": raw_error,
        "maximum_raw_packet_norm_drift": raw_error,
        "maximum_post_normalization_norm_error": 2e-16,
        "maximum_hermiticity_error": 0.0,
        "maximum_conditional_eigenvector_gram_residual": 0.0,
        "conditional_eigenvector_gram_diagnostics": 0,
        "raw_norm_warning_tolerance": 1e-8,
        "raw_norm_hard_failure_tolerance": 1e-6,
        "post_normalization_norm_tolerance": 1e-12,
        "gram_diagnostic_trigger": 1e-10,
        "raw_norm_error_argmax": {"time_index": 1, "error": raw_error},
        "actual_dtype": "torch.complex128",
        "actual_probability_dtype": "torch.float64",
    }


def _install_fake_h1_run(
    monkeypatch: pytest.MonkeyPatch, *, numerical_status: str
) -> None:
    class FakeModel:
        def __init__(self, **kwargs):
            self.Nx = int(kwargs["Nx"])
            self.Ny = int(kwargs["Ny"])
            self.DW_loc = [5, 15]
            self.device = torch.device("cpu")

        def _sequence_helper(self, *_args, **_kwargs):
            return {"coords_for_len": [(0, 0)]}

        def run_markov_circuit(self, **_kwargs):
            return {
                "state_representation_resolved": "physical_frame",
                "covariance_materialization_count": 0,
            }

    class FakeObserver:
        def __init__(self, **_kwargs):
            pass

        def save(self, directory: Path, *, config):
            assert config["sampling_revision"] == packet_runner.REVISION
            return _fake_v3_product(directory, status=numerical_status)

    class FakeRecord:
        def __init__(self, **_kwargs):
            pass

        def save(self, path: Path):
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                path,
                schema=np.asarray("synthetic_ordered_record"),
                site_ids=np.zeros((5, 80, 1), dtype=np.int32),
            )
            return {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }

    monkeypatch.setattr(packet_runner, "classA_U1FGTN_gpu", FakeModel)
    monkeypatch.setattr(packet_runner, "H1EndpointPacketObserver", FakeObserver)
    monkeypatch.setattr(packet_runner, "OrderedBornRecordWriter", FakeRecord)
    monkeypatch.setattr(
        packet_runner,
        "_find_replay_source",
        lambda **_kwargs: (None, ["synthetic fresh-run test"]),
    )
    monkeypatch.setattr(
        packet_runner,
        "_require_a100",
        lambda **_kwargs: {
            "device": "NVIDIA A100-SXM4-40GB",
            "free_bytes": 38 * 1024**3,
            "total_bytes": 40 * 1024**3,
            "total_gib": 40.0,
            "free_fraction": 0.95,
            "smoke_override": False,
        },
    )


def test_v3_warning_is_a_canonical_runner_success_with_saved_raw_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config(BUNDLE)
    case = next(
        row for row in expand_cases(config)
        if row["protocol"] == "soft" and row["model"]["alpha_1"] == 1.0
    )
    _install_fake_h1_run(monkeypatch, numerical_status="warning")
    monkeypatch.setattr(
        packet_runner.tempfile, "gettempdir", lambda: str(tmp_path / "scratch")
    )

    result = packet_runner.run_case(
        bundle_root=BUNDLE,
        config=config,
        case=case,
        shard_index=0,
        drive_root=tmp_path / "drive",
        mode="production",
        archive_result=False,
    )
    assert result["status"] == "preflight_complete"
    assert result["manifest"]["status"] == "complete_local"
    assert result["manifest"]["numerical_status"] == "warning"
    scratch = Path(result["scratch"])
    assert (scratch / "shards/shard_000/h1_endpoint_packet/packet_drift.npz").is_file()
    assert (scratch / "manifest.json").is_file()


def test_v3_hard_failure_persists_then_forensically_archives_and_preflight_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config(BUNDLE)
    case = next(
        row for row in expand_cases(config)
        if row["protocol"] == "soft" and row["model"]["alpha_1"] == 1.0
    )
    _install_fake_h1_run(monkeypatch, numerical_status="hard_failure")
    monkeypatch.setattr(
        packet_runner.tempfile, "gettempdir", lambda: str(tmp_path / "scratch")
    )

    unarchived = packet_runner.run_case(
        bundle_root=BUNDLE,
        config=config,
        case=case,
        shard_index=0,
        drive_root=tmp_path / "unarchived_drive",
        mode="production",
        archive_result=False,
    )
    assert unarchived["status"] == "numerical_hard_failure"
    assert unarchived["manifest"]["status"] == "numerical_hard_failure"
    assert unarchived["manifest"]["numerical_status"] == "hard_failure"
    scratch = Path(unarchived["scratch"])
    assert (scratch / "shards/shard_000/h1_endpoint_packet/packet_drift.npz").is_file()
    assert (scratch / "shards/shard_000/ordered_born_record.npz").is_file()
    assert (scratch / "shards/shard_000/rng_before.npz").is_file()
    assert (scratch / "shards/shard_000/rng_after.npz").is_file()
    assert json.loads((scratch / "manifest.json").read_text())["status"] == (
        "numerical_hard_failure"
    )

    archived = packet_runner.run_case(
        bundle_root=BUNDLE,
        config=config,
        case=case,
        shard_index=0,
        drive_root=tmp_path / "ordinary_drive",
        mode="production",
        archive_result=True,
    )
    assert archived["status"] == "numerical_hard_failure_archived"
    forensic_archive = Path(archived["archive_path"])
    assert forensic_archive.parent.name == "_failed_qualifications"
    assert forensic_archive.is_file()
    assert packet_runner._root_manifest_from_archive(forensic_archive)["status"] == (
        "numerical_hard_failure"
    )
    with tarfile.open(forensic_archive, "r:gz") as archive:
        assert any(
            member.name.endswith("h1_endpoint_packet/packet_drift.npz")
            for member in archive.getmembers()
        )

    payload = packet_runner.a100_preflight(
        bundle_root=BUNDLE,
        config=config,
        drive_root=tmp_path / "preflight_drive",
    )
    assert payload["safe"] is False
    assert payload["case_id"] == "H1_N20x40_soft_a1-1"
    assert payload["measured_trajectories"] == 5
    assert payload["measured_cycles"] == 80
    assert payload["numerical_status"] == "hard_failure"
    assert payload["scientific_shard_archived"] is False
    assert payload["failed_qualification_archived"] is True
    assert payload["failure"].startswith("FloatingPointError:")
    failed_archive = Path(payload["failed_qualification_archive"])
    assert failed_archive.parent.name == "_failed_qualifications"
    assert failed_archive.is_file()
    failed_receipt = json.loads(
        failed_archive.with_suffix(
            failed_archive.suffix + ".receipt.json"
        ).read_text()
    )
    assert failed_receipt["archive_sha256"] == (
        payload["failed_qualification_archive_sha256"]
    )
    assert hashlib.sha256(failed_archive.read_bytes()).hexdigest() == (
        payload["failed_qualification_archive_sha256"]
    )
    receipt_payload = json.loads(Path(payload["receipt_path"]).read_text())
    assert receipt_payload["safe"] is False
    assert receipt_payload["case_id"] == "H1_N20x40_soft_a1-1"


@pytest.mark.parametrize(
    "error_type",
    [FloatingPointError, RuntimeError],
    ids=["floating-point", "controlled-runtime"],
)
def test_v3_preflight_catches_controlled_failure_and_writes_unsafe_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error_type,
) -> None:
    config = load_config(BUNDLE)

    def fail_run_case(**_kwargs):
        raise error_type("synthetic qualification failure")

    monkeypatch.setattr(packet_runner, "run_case", fail_run_case)
    monkeypatch.setattr(
        packet_runner.tempfile, "gettempdir", lambda: str(tmp_path / "scratch")
    )
    payload = packet_runner.a100_preflight(
        bundle_root=BUNDLE,
        config=config,
        drive_root=tmp_path / "drive",
    )
    assert payload["safe"] is False
    assert payload["case_id"] == "H1_N20x40_soft_a1-1"
    assert payload["failure"] == (
        f"{error_type.__name__}: synthetic qualification failure"
    )
    assert payload["scientific_shard_archived"] is False
    assert payload["failed_qualification_archived"] is False
    persisted = json.loads(Path(payload["receipt_path"]).read_text())
    assert persisted["safe"] is False
    assert persisted["failure"] == payload["failure"]


def test_v3_safe_preflight_binds_the_current_qualification_archive(
    tmp_path: Path,
) -> None:
    config = load_config(BUNDLE)
    case = next(
        row for row in expand_cases(config)
        if row["case_id"] == "H1_N20x40_soft_a1-1"
    )
    _, archive, run_id, _ = packet_runner._archive_paths(
        bundle_root=BUNDLE,
        config=config,
        case=case,
        shard_index=0,
        drive_root=tmp_path,
        mode="production",
    )
    source_digest = packet_runner.sha256_json(
        packet_runner._source_hashes(BUNDLE / "src")
    )
    scratch = tmp_path / "qualification_scratch"
    scratch.mkdir()
    (scratch / "manifest.json").write_text(json.dumps({
        "bundle": packet_runner.BUNDLE,
        "sampling_revision": packet_runner.REVISION,
        "audit_sha256": packet_runner.AUDIT,
        "bundle_source_hashes_sha256": source_digest,
        "case_id": case["case_id"],
        "shard_index": 0,
        "status": "complete_local",
    }))
    qualification_receipt = packet_runner._archive(scratch, archive, run_id)
    payload = {
        "schema": packet_runner.A100_PREFLIGHT_SCHEMA,
        "bundle": packet_runner.BUNDLE,
        "sampling_revision": packet_runner.REVISION,
        "audit_sha256": packet_runner.AUDIT,
        "canonical_engine_sha256": packet_runner.sha256_file(
            BUNDLE / "src" / "classA_U1FGTN_gpu.py"
        ),
        "bundle_source_hashes_sha256": source_digest,
        "case_id": case["case_id"],
        "safe": True,
        "contract_changed": True,
        "scientific_shard_archived": True,
        "failed_qualification_archived": False,
        "dtype_contract_failure": False,
        "numerical_status": "pass",
        "maximum_raw_packet_norm_error": 0.0,
        "maximum_raw_packet_norm_drift": 0.0,
        "maximum_post_normalization_norm_error": 0.0,
        "qualification_archive": qualification_receipt["archive"],
        "qualification_archive_sha256": qualification_receipt[
            "archive_sha256"
        ],
    }
    preflight = packet_runner._preflight_path(tmp_path, config)
    preflight.parent.mkdir(parents=True, exist_ok=True)
    preflight.write_text(json.dumps(payload))
    assert packet_runner.require_safe_preflight(
        drive_root=tmp_path, config=config
    )["case_id"] == case["case_id"]

    archive.unlink()
    with pytest.raises(RuntimeError, match="qualification_archive_pair"):
        packet_runner.require_safe_preflight(drive_root=tmp_path, config=config)
