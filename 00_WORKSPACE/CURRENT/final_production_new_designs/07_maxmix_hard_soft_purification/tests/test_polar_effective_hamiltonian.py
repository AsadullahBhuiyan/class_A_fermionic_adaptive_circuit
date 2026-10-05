from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


BUNDLE = Path(__file__).resolve().parents[1]
REPO = BUNDLE.parents[3]
MODULE_PATH = BUNDLE / "polar_effective_hamiltonian.py"
SPEC = importlib.util.spec_from_file_location("polar_effective_hamiltonian", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
polar = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = polar
SPEC.loader.exec_module(polar)


def random_unitary(rng: np.random.Generator, size: int) -> np.ndarray:
    raw = rng.normal(size=(size, size)) + 1j * rng.normal(size=(size, size))
    q, r = np.linalg.qr(raw)
    phases = np.diag(r)
    phases = np.where(np.abs(phases) > 0, phases / np.abs(phases), 1.0)
    return q * phases.conj()[None, :]


def test_capped_log_polar_recovers_unsaturated_generator() -> None:
    rng = np.random.default_rng(7)
    vectors = random_unitary(rng, 8)
    levels = np.linspace(-2.5, 2.5, 8)
    generator = (vectors * levels[None, :]) @ vectors.conj().T
    covariance = (vectors * np.tanh(levels)[None, :]) @ vectors.conj().T
    recovered, sign_matrix, values, diagnostics = polar.capped_log_polar(covariance, 10.0)
    assert np.max(np.abs(recovered - generator)) < 2.0e-13
    assert np.max(np.abs(sign_matrix @ sign_matrix - np.eye(8))) < 2.0e-13
    assert np.allclose(values, np.sort(np.tanh(levels)))
    assert diagnostics["capped_mode_count"] == 0


def test_exact_caps_are_finite_and_equal_to_requested_log_cap() -> None:
    covariance = np.diag([-1.0, -0.5, 0.5, 1.0]).astype(np.complex128)
    recovered, _, _, diagnostics = polar.capped_log_polar(covariance, 10.0)
    assert np.all(np.isfinite(recovered))
    assert np.isclose(recovered[0, 0], -10.0)
    assert np.isclose(recovered[-1, -1], 10.0)
    assert diagnostics["capped_mode_count"] == 2


def test_y_twirl_fourier_reconstruction_and_translation_invariance() -> None:
    rng = np.random.default_rng(11)
    ny, block = 8, 3
    raw = rng.normal(size=(ny * block, ny * block)) + 1j * rng.normal(
        size=(ny * block, ny * block)
    )
    matrix = 0.5 * (raw + raw.conj().T)
    disp = polar.y_twirl_displacements(matrix, ny)
    assert polar.displacement_hermiticity_residual(disp) < 1.0e-13
    assert polar.displacement_translation_residual(disp) < 1.0e-13
    assert polar.fourier_reconstruction_residual(disp) < 2.0e-13
    momenta, blocks = polar.momentum_blocks(disp, twist=1.0e-7)
    assert momenta.shape == (ny,)
    assert np.max(np.abs(blocks - blocks.conj().transpose(0, 2, 1))) < 1.0e-13


def test_quenched_log_and_annealed_log_do_not_commute() -> None:
    first = np.diag([0.8, -0.2]).astype(np.complex128)
    rotate = np.asarray([[1.0, 1.0], [-1.0, 1.0]], dtype=np.complex128) / np.sqrt(2.0)
    second = rotate @ np.diag([0.7, -0.6]) @ rotate.conj().T
    a_first, _, _, _ = polar.capped_log_polar(first, 10.0)
    a_second, _, _, _ = polar.capped_log_polar(second, 10.0)
    quenched = 0.5 * (a_first + a_second)
    annealed, _, _, _ = polar.capped_log_polar(0.5 * (first + second), 10.0)
    assert np.max(np.abs(quenched - annealed)) > 1.0e-3


def test_half_filled_projector_and_twist() -> None:
    ny, block = 8, 2
    disp = np.zeros((ny, block, block), dtype=np.complex128)
    disp[0] = np.diag([-0.4, 0.4])
    disp[1] = np.diag([-0.2, 0.2])
    disp[-1] = disp[1].conj().T
    correlation_plus, diagnostics_plus = polar.half_filled_ground_state(
        disp, twist=1.0e-7
    )
    correlation_minus, diagnostics_minus = polar.half_filled_ground_state(
        disp, twist=-1.0e-7
    )
    assert diagnostics_plus["occupied_modes"] == ny
    assert diagnostics_plus["projector_idempotency_residual"] < 1.0e-12
    assert diagnostics_minus["projector_idempotency_residual"] < 1.0e-12
    assert abs(np.trace(correlation_plus).real - ny) < 1.0e-12
    assert abs(np.trace(correlation_minus).real - ny) < 1.0e-12


def test_opposite_wall_modes_are_localized_and_counterpropagating() -> None:
    ny = 16
    momenta = 2.0 * np.pi * np.arange(ny) / ny
    for active_x in (polar.ACTIVE_X, polar.SOFT_ACTIVE_X):
        block = 2 * len(active_x)
        blocks = np.zeros((ny, block, block), dtype=np.complex128)
        left = 2 * active_x.index(5)
        right = 2 * active_x.index(15)
        for ki, momentum in enumerate(momenta):
            diagonal = np.r_[
                np.full(block // 2, -3.0), np.full(block // 2, 3.0)
            ]
            diagonal[left] = np.sin(momentum)
            diagonal[right] = -np.sin(momentum)
            blocks[ki] = np.diag(diagonal)
        displacements = np.fft.ifft(blocks, axis=0)
        bands = polar.band_observables(
            displacements, wall_weight_min=0.5, active_x=active_x
        )
        assert np.min(bands["wall_weight"]) > 0.99
        assert np.prod(bands["velocity"]) < 0.0


def test_entropy_fit_uses_two_wall_cardy_calabrese_factor() -> None:
    ny = 40
    ay = np.arange(1, ny // 2 + 1)
    entropy = (1.0 / 3.0) * polar.log_chord(ny, ay) + 2.5
    fit = polar.fit_entropy_curve(entropy, ny=ny, order=1, ay_min=8)
    assert np.isclose(fit["slope"], 1.0 / 3.0)
    assert np.isclose(fit["c_per_wall"], 1.0)
    assert fit["r_squared"] > 1.0 - 1.0e-13


def test_existing_b0_hard_wall_calibration_is_near_one() -> None:
    table = (
        REPO
        / "00_WORKSPACE"
        / "CURRENT"
        / "experiment_review"
        / "b0_exact_domain_wall"
        / "results"
        / "20260816_183928"
        / "processed"
        / "tables"
        / "entropy_fits.csv"
    )
    with table.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected = [
        row
        for row in rows
        if row["construction"] == "hard_exterior"
        and row["geometry_key"] == "hard_exterior__Nx020__Ny048"
        and row["quantity"] == "full_strip"
    ]
    assert {int(row["q"]) for row in selected} == {1, 2, 3}
    assert all(abs(float(row["c_estimate"]) - 1.0) < 0.02 for row in selected)
    coupled = [
        row
        for row in rows
        if row["construction"] == "coupled"
        and row["geometry_key"] == "coupled__Nx020__Ny048"
        and row["quantity"] == "full_strip"
    ]
    assert {int(row["q"]) for row in coupled} == {1, 2, 3}
    assert all(abs(float(row["c_estimate"]) - 1.0) < 0.02 for row in coupled)


def test_real_hard_wall_sample_has_decoupled_active_slab() -> None:
    result = (
        BUNDLE
        / "gpu_data"
        / polar.EXPECTED_REVISION
        / "hard"
        / "Ny020"
        / "shard_000_samples_000-004.npz"
    )
    with np.load(result, allow_pickle=False) as saved:
        full_g = np.asarray(saved["G_final"][0])
    active_g, diagnostics = polar.extract_active_covariance(full_g, nx=20, ny=20)
    assert active_g.shape == (440, 440)
    assert diagnostics["active_exterior_coupling"] == 0.0
    assert diagnostics["exterior_purity_residual"] == 0.0
    assert diagnostics["exterior_inter_y_coupling"] == 0.0
    assert diagnostics["active_hermiticity_residual"] < 1.0e-12


def test_analysis_notebook_is_editable_complete_and_local_cpu() -> None:
    for filename, construction in (
        ("polar_effective_hamiltonian_analysis.ipynb", "hard"),
        ("soft_polar_effective_hamiltonian_analysis.ipynb", "soft"),
    ):
        notebook_path = BUNDLE / filename
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        cells = notebook["cells"]
        assert cells[0]["cell_type"] == "markdown"
        first_code = next(cell for cell in cells if cell["cell_type"] == "code")
        first_source = "".join(first_code["source"])
        assert "CPU_RANGE" in first_source
        assert "N_WORKERS" in first_source
        assert "OUTPUT_ROOT_OVERRIDE" in first_source
        assert "os.sched_setaffinity" in first_source
        assert "MAX_SAMPLES_PER_NY = None" in first_source
        all_code = "\n".join(
            "".join(cell["source"]) for cell in cells if cell["cell_type"] == "code"
        )
        assert f"construction='{construction}'" in all_code or f"CONSTRUCTION = '{construction}'" in all_code
        theory = "\n".join(
            "".join(cell["source"])
            for cell in cells
            if cell["cell_type"] == "markdown"
        )
        assert "trajectory by trajectory" in theory
        assert "twirl" in theory
        assert "quenched" in theory
        figure_cells = [
            "".join(cell["source"])
            for cell in cells
            if cell["cell_type"] == "code"
            and "fig, axes = plt.subplots" in "".join(cell["source"])
        ]
        assert len(figure_cells) == 5
        assert all(
            "fig.savefig" in source and "dpi=300" in source
            for source in figure_cells
        )
        assert "h_{\\mathrm{eff}}/(4N_y)" in figure_cells[0]
        assert "fit-window sensitivity" in figure_cells[-1]
        assert (
            "degeneracy sensitivity" in figure_cells[-1]
            or "twist sensitivity" in figure_cells[-1]
        )
        final_source = "".join(cells[-1]["source"])
        assert "claim_ledger" in final_source
        assert "numerical" in final_source or "diagnostics" in final_source


def test_real_soft_inventory_and_full_transfer_sector() -> None:
    records = polar.discover_samples(
        construction="soft", ny_values=(20,), verify_hashes=False, require_complete=True
    )
    assert len(records) == 100
    assert {record.sample_index for record in records} == set(range(100))
    record = records[0]
    with np.load(record.result_path, allow_pickle=False) as saved:
        full_g = np.asarray(saved["G_final"][record.sample_offset])
    active_g, diagnostics = polar.extract_active_covariance(
        full_g,
        nx=20,
        ny=20,
        active_x=polar.SOFT_ACTIVE_X,
        require_product_exterior=False,
    )
    assert active_g.shape == (800, 800)
    assert diagnostics["active_exterior_coupling"] == 0.0
    assert polar.AnalysisConfig().identity_hash != polar.AnalysisConfig(
        construction="soft"
    ).identity_hash
