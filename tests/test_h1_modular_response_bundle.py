from __future__ import annotations

import hashlib
import io
import json
import sys
import tarfile
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "00_WORKSPACE/CURRENT/final_production_new_designs/03_h1_modular_response"
SRC = BUNDLE / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from h1_modular_analysis import (  # noqa: E402
    extract_ridge,
    finite_window_spectrum,
    merge_archives,
)
from h1_modular_observables import (  # noqa: E402
    H1ModularObserver,
    checkpoint_cycles,
    dense_retarded_reference,
    packet_probability_from_eigensystem,
    reduced_correlation_from_frame,
    retained_upper_half_indices,
    sample_wall_sources,
    wall_x_positions,
)
from h1_modular_runner import expand_cases, load_config  # noqa: E402
from h1_modular_runner import require_safe_preflight  # noqa: E402


def _random_frame(dimension: int, rank: int, seed: int = 917) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    raw = torch.complex(
        torch.randn((dimension, rank), generator=generator, dtype=torch.float64),
        torch.randn((dimension, rank), generator=generator, dtype=torch.float64),
    )
    return torch.linalg.qr(raw, mode="reduced").Q


def test_exact_four_case_matrix_and_locked_hyperparameters() -> None:
    cases = expand_cases(load_config(BUNDLE))
    assert len(cases) == 4
    assert {(case["protocol"], case["model"]["alpha_1"]) for case in cases} == {
        (protocol, alpha) for protocol in ("hard", "soft") for alpha in (1.0, 3.0)
    }
    for case in cases:
        assert case["model"]["Nx"] == 20
        assert case["model"]["Ny"] == 40
        assert case["model"]["alpha_2"] == 30.0
        assert case["model"]["nshell"] == 1
        assert case["model"]["init_mode"] == "default"
        assert case["run"] == {
            "cycles": 80,
            "samples": 25,
            "sequence": "random",
            "perfect_correction": True,
            "postselect": False,
            "postselect_probability": 0.0,
            "n_a": 0.5,
            "meas_slab_only": case["protocol"] == "hard",
        }
        assert case["model"]["dw_truncation"] is (case["protocol"] == "hard")
    assert checkpoint_cycles(40) == [40, 48, 56, 64, 72, 80]


def test_source_manifest_matches_files_and_production_is_preflight_locked(
    tmp_path: Path,
) -> None:
    manifest = json.loads((SRC / "source_manifest.json").read_text())
    for name, row in manifest["files"].items():
        assert hashlib.sha256((SRC / name).read_bytes()).hexdigest() == row["sha256"]
    assert (SRC / "classA_U1FGTN_gpu.py").read_bytes() == (
        ROOT / "src/fgtn/classA_U1FGTN_gpu.py"
    ).read_bytes()
    assert (SRC / "occupied_frame_gpu.py").read_bytes() == (
        ROOT / "src/fgtn/occupied_frame_gpu.py"
    ).read_bytes()
    with np.testing.assert_raises_regex(RuntimeError, "locked until --a100-preflight"):
        require_safe_preflight(drive_root=tmp_path, config=load_config(BUNDLE))


def test_retained_indices_are_exactly_upper_half_and_sources_never_leave_it() -> None:
    nx, ny = 20, 40
    indices = retained_upper_half_indices(nx=nx, ny=ny).tolist()
    expected = [
        mu + 2 * x + 2 * nx * y
        for y in range(20, 40) for x in range(20) for mu in (0, 1)
    ]
    assert indices == expected
    assert not set(indices).intersection({
        mu + 2 * x + 2 * nx * y
        for y in range(20) for x in range(20) for mu in (0, 1)
    })
    sources = sample_wall_sources(
        root_seed=123, nx=nx, ny=ny, sample_id=4, checkpoint=64
    )
    assert set(sources[:, :, 0].ravel()) == {5, 15}
    assert np.all((sources[:, :, 1] >= 20) & (sources[:, :, 1] < 40))


def test_sources_are_deterministic_unique_fresh_and_case_independent() -> None:
    kwargs = dict(root_seed=2026082603, nx=20, ny=40, sample_id=7, source_count=10)
    first = sample_wall_sources(checkpoint=40, **kwargs)
    repeat = sample_wall_sources(checkpoint=40, **kwargs)
    later = sample_wall_sources(checkpoint=48, **kwargs)
    np.testing.assert_array_equal(first, repeat)
    assert not np.array_equal(first, later)
    for wall in range(2):
        assert len(np.unique(first[wall, :, 1])) == 10


def test_frame_reduced_correlation_and_spectrum_match_dense_reference() -> None:
    nx, ny, rank = 4, 10, 37
    frame = _random_frame(2 * nx * ny, rank)
    indices = retained_upper_half_indices(nx=nx, ny=ny)
    correlation, occupations, vectors, error = reduced_correlation_from_frame(
        frame, rank=rank, indices=indices
    )
    expected = frame.index_select(0, indices) @ frame.index_select(0, indices).mH
    torch.testing.assert_close(correlation, expected, atol=2e-13, rtol=2e-13)
    torch.testing.assert_close(
        vectors @ torch.diag(occupations.to(vectors.dtype)) @ vectors.mH,
        expected,
        atol=2e-12,
        rtol=2e-12,
    )
    assert error < 1e-13


def test_analytic_retarded_density_matches_symmetric_finite_kick() -> None:
    dimension = 12
    unitary = _random_frame(dimension, dimension, seed=199)
    occupations = torch.linspace(0.15, 0.85, dimension, dtype=torch.float64)
    correlation = unitary @ torch.diag(occupations.to(torch.complex128)) @ unitary.mH
    source = [2, 3]
    times = np.array([0.0, 0.2, 0.7])
    analytic = dense_retarded_reference(correlation, source, times, 1e-10)
    hvals = torch.log((1.0 - occupations) / occupations)
    hamiltonian = unitary @ torch.diag(hvals.to(torch.complex128)) @ unitary.mH
    projector = torch.zeros((dimension, dimension), dtype=torch.complex128)
    projector[source, source] = 1.0
    kick = 1e-6
    plus_kick = torch.matrix_exp(-1j * kick * projector)
    minus_kick = torch.matrix_exp(+1j * kick * projector)
    finite = []
    for time in times:
        evolution = torch.matrix_exp(-1j * float(time) * hamiltonian)
        plus = evolution @ plus_kick @ correlation @ plus_kick.mH @ evolution.mH
        minus = evolution @ minus_kick @ correlation @ minus_kick.mH @ evolution.mH
        finite.append(torch.real(torch.diag((plus - minus) / (2 * kick))))
    torch.testing.assert_close(
        analytic, torch.stack(finite), atol=3e-9, rtol=3e-8
    )
    assert float(torch.max(torch.abs(analytic.sum(dim=1)))) < 1e-12


def test_spectral_cutoff_regularizes_h_not_the_initial_tangent() -> None:
    dimension = 8
    vectors = _random_frame(dimension, dimension, seed=211)
    occupations = torch.tensor(
        [1e-14, 0.08, 0.2, 0.4, 0.6, 0.8, 0.92, 1 - 1e-14],
        dtype=torch.float64,
    )
    correlation = vectors @ torch.diag(occupations.to(torch.complex128)) @ vectors.mH
    source = [0, 1]
    first = dense_retarded_reference(correlation, source, [0.0], 1e-8)
    second = dense_retarded_reference(correlation, source, [0.0], 1e-12)
    torch.testing.assert_close(first, second, atol=2e-13, rtol=2e-13)


def test_packet_probability_matches_direct_modular_propagation() -> None:
    dimension = 10
    vectors = _random_frame(dimension, dimension, seed=88)
    occupations = torch.linspace(0.1, 0.9, dimension, dtype=torch.float64)
    source = torch.tensor([4, 5])
    times = torch.tensor([0.0, 0.3, 1.0], dtype=torch.float64)
    actual = packet_probability_from_eigensystem(
        occupations=occupations,
        vectors=vectors,
        source_indices=source,
        modular_times=times,
        epsilon=1e-10,
    )
    energies = torch.log((1 - occupations) / occupations)
    hamiltonian = vectors @ torch.diag(energies.to(torch.complex128)) @ vectors.mH
    packet = torch.zeros(dimension, dtype=torch.complex128)
    packet[source] = 1 / np.sqrt(2)
    expected = torch.stack([
        (torch.matrix_exp(-1j * time * hamiltonian) @ packet).abs().square()
        for time in times
    ])
    torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)
    torch.testing.assert_close(
        actual.sum(dim=1), torch.ones(len(times), dtype=torch.float64)
    )


def test_complete_observer_schema_averaging_order_and_retired_absence(tmp_path: Path) -> None:
    nx, ny, samples = 4, 10, 1
    dimension = 2 * nx * ny
    frame = _random_frame(dimension, dimension // 2).unsqueeze(0)
    state = type("FrameState", (), {
        "frame": frame,
        "ranks": torch.tensor([dimension // 2], dtype=torch.long),
    })()
    response_times = np.arange(0.0, 4.0 + 0.025, 0.05)
    packet_times = np.arange(0.0, 8.0 + 0.025, 0.05)
    observer = H1ModularObserver(
        nx=nx,
        ny=ny,
        checkpoints=checkpoint_cycles(ny),
        global_sample_ids=[11],
        root_seed=99,
        source_count=2,
        response_times=response_times,
        packet_times=packet_times,
        spectral_clip_eps=[1e-10],
        wall_half_widths=[1],
        primary_epsilon=1e-10,
        primary_wall_half_width=1,
        fit_window=(0.1, 0.8),
    )
    for cycle in checkpoint_cycles(ny):
        observer(cycle=cycle, state=state, batch_start=0, batch_count=1)
    output = tmp_path / "h1"
    observer.save(output, config={"test": True})
    displacement_axis = np.arange(-(ny // 2 - 1), ny // 2)
    mean_profile = observer.retarded_aligned_profile[0, 0, 0, 0, 0].mean(axis=0)
    mean_dipole = mean_profile @ displacement_axis
    mask = (response_times >= 0.1) & (response_times <= 0.8)
    expected_slope = np.polyfit(response_times[mask], mean_dipole[mask], 1)[0]
    np.testing.assert_allclose(observer.wall_velocity[0, 0, 0, 0, 0], expected_slope)
    forbidden = {"static_susceptibility", "covariance", "eigenvectors", "entropy", "chern", "bott", "tangent"}
    for path in output.glob("*.npz"):
        with np.load(path, allow_pickle=False) as data:
            assert forbidden.isdisjoint(data.files)
    with np.load(output / "retarded_fields.npz", allow_pickle=False) as data:
        assert data["retarded_density_xy"].shape == (1, 6, 2, 2, 81, 5, 4)


def test_fourier_ridge_recovers_synthetic_opposite_propagation() -> None:
    times = np.arange(0.0, 4.0 + 0.025, 0.05)
    nd, nx = 39, 20
    displacement = np.arange(-19, 20)
    velocity = 1.2
    field = np.zeros((len(times), nd, nx))
    for index, time in enumerate(times):
        center = velocity * time
        profile = np.exp(-0.5 * ((displacement - center) / 1.5) ** 2)
        profile -= profile.mean()
        field[index, :, 5] = profile
    omega = np.linspace(0.0, 3.0, 601)
    k, spectrum = finite_window_spectrum(
        field, wall_x=5, wall_half_width=2,
        modular_times=times, eta=0.35, omega_values=omega,
    )
    fit = extract_ridge(
        spectrum, k_values=k, omega_values=omega, sector_sign=1,
        omega_window=(0.2, 2.5), k_max=2.6,
    )
    assert np.isfinite(fit["velocity"])
    assert len(fit["omega_peak"]) == 8


def _npz_bytes(**arrays: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    return buffer.getvalue()


def _tar_member(handle: tarfile.TarFile, name: str, payload: bytes) -> None:
    member = tarfile.TarInfo(name)
    member.size = len(payload)
    handle.addfile(member, io.BytesIO(payload))


def test_merge_requires_five_shards_and_exact_25_sample_axis(tmp_path: Path) -> None:
    config = load_config(BUNDLE)
    source_template = np.zeros((5, 1, 2, 1, 2), dtype=np.int64)
    source_template[:, :, 0, :, 0] = 5
    source_template[:, :, 1, :, 0] = 15
    source_template[..., 1] = 20
    for case in expand_cases(config):
        for shard in range(5):
            sample_ids = np.arange(5 * shard, 5 * shard + 5, dtype=np.int64)
            common = _npz_bytes(
                schema=np.asarray("h1_retarded_modular_response_and_packet_v1"),
                global_sample_ids=sample_ids,
                checkpoints=np.asarray([40]),
                modular_times=np.asarray([0.0, 0.05]),
                packet_modular_times=np.asarray([0.0, 0.05]),
                spectral_clip_eps=np.asarray([1e-10]),
                wall_half_widths=np.asarray([2]),
                primary_epsilon_index=np.asarray(0),
                primary_wall_width_index=np.asarray(0),
                source_xy=source_template,
                restricted_occupation_spectrum=np.full((5, 1, 2), 0.5),
                spectral_clip_counts=np.zeros((5, 1, 1, 2), dtype=np.int32),
                covariance_hermiticity_error=np.zeros((5, 1)),
                observer_seconds=np.zeros((5, 1)),
            )
            summary = _npz_bytes(
                schema=np.asarray("h1_retarded_modular_response_and_packet_v1"),
                retarded_source_mean_aligned_density=np.zeros((5, 1, 2, 2, 1, 2)),
                retarded_source_mean_aligned_profile=np.zeros((5, 1, 1, 1, 2, 2, 1)),
                retarded_wall_velocity=np.zeros((5, 1, 1, 1, 2)),
                retarded_wall_velocity_r2=np.zeros((5, 1, 1, 1, 2)),
                handed_response=np.full((5, 1, 1, 1), float(case["model"]["alpha_1"])),
                retarded_total_charge=np.zeros((5, 1, 1, 2, 1, 2)),
                retarded_transverse_leakage=np.zeros((5, 1, 1, 1, 2, 1, 2)),
            )
            packet = _npz_bytes(
                schema=np.asarray("h1_retarded_modular_response_and_packet_v1"),
                packet_source_mean_profile=np.zeros((5, 1, 2, 2, 1)),
                packet_wall_velocity=np.zeros((5, 1, 1, 1, 2)),
                packet_wall_fit_points=np.ones((5, 1, 1, 1, 2), dtype=np.int16),
                packet_handed_velocity=np.zeros((5, 1, 1, 1)),
                packet_wall_retention=np.ones((5, 1, 1, 1, 2, 1, 2)),
            )
            fields = _npz_bytes(
                schema=np.asarray("h1_retarded_modular_response_and_packet_v1"),
                retarded_density_xy=np.zeros((5, 1, 2, 1, 2, 1, 2)),
            )
            manifest = {
                "bundle": "03_h1_modular_response",
                "sampling_revision": "production_25sample_h1_modular_response_v1",
                "audit_sha256": config["audit_sha256"],
                "shard_index": shard,
                "run_config": {"case": case},
            }
            path = tmp_path / f"{case['case_id']}_shard-{shard}.tar.gz"
            with tarfile.open(path, "w:gz") as archive:
                _tar_member(archive, "manifest.json", json.dumps(manifest).encode())
                _tar_member(archive, "shards/shard_000/h1_modular/common.npz", common)
                _tar_member(archive, "shards/shard_000/h1_modular/retarded_summary.npz", summary)
                _tar_member(archive, "shards/shard_000/h1_modular/packet_drift.npz", packet)
                _tar_member(archive, "shards/shard_000/h1_modular/retarded_fields.npz", fields)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            path.with_suffix(path.suffix + ".receipt.json").write_text(json.dumps({
                "archive": path.name, "archive_sha256": digest,
            }))
    merged = merge_archives(tmp_path)
    assert len(merged) == 4
    for item in merged.values():
        assert item["sample_ids"].tolist() == list(range(25))
        assert item["handed_response"].shape[0] == 25
