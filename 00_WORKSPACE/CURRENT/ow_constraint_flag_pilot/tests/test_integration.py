from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

EXPERIMENT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT.parents[2]
SRC = REPO_ROOT / "src"
sys.path[:0] = [str(EXPERIMENT), str(SRC)]

from fgtn.classA_U1FGTN import classA_U1FGTN
from fgtn.diagnostics import TrajectoryActivityRecorder, compute_static_completion


def uniform_model() -> classA_U1FGTN:
    model = classA_U1FGTN(4, 6, DW=False, nshell=1, alpha_1=1, alpha_2=30, trial_orbitals="X")
    model.construct_OW_projectors(nshell=1, DW=False, trial_orbitals="X", dw_truncation=False)
    return model


def recorded(seed: int, *, parallel: bool):
    model = uniform_model()
    recorder = TrajectoryActivityRecorder.from_model(model, cycles=2, samples=2)
    result = model.run_markov_circuit(
        G_history=False,
        progress=False,
        cycles=2,
        samples=2,
        save=False,
        perfect_correction=True,
        init_mode="maxmix",
        sequence="random",
        random_seed=seed,
        parallelize_samples=parallel,
        n_jobs=2,
        backend="threading",
        throttle=False,
        trajectory_weight_observer=recorder if not parallel else None,
    )
    return result, recorder


def test_perfect_correction_identity_calibration_and_serial_parallel_reproducibility():
    serial, recorder = recorded(1234, parallel=False)
    parallel, _ = recorded(1234, parallel=True)
    recorder.assert_complete()
    np.testing.assert_array_equal(recorder.defect, np.abs(recorder.transfer))
    np.testing.assert_allclose(serial["G_final"], parallel["G_final"], atol=1e-14, rtol=1e-14)
    predicted = np.mean(1.0 - recorder.success_probability[recorder.valid])
    observed = np.mean(recorder.defect[recorder.valid])
    attempts = int(np.count_nonzero(recorder.valid))
    sigma = np.sqrt(max(predicted * (1 - predicted), 1e-12) / attempts)
    assert abs(observed - predicted) < 6 * sigma + 1 / attempts


def test_untruncated_uniform_frame_has_exact_completion():
    model = classA_U1FGTN(4, 6, DW=False, nshell=None, alpha_1=1, alpha_2=30, trial_orbitals="X")
    model.construct_OW_projectors(nshell=None, DW=False, trial_orbitals="X", dw_truncation=False)
    result = compute_static_completion(model, meas_slab_only=True)
    assert result.exact_completion_exists
    assert result.completion_constraint_error < 1e-10
    assert result.f_star < 1e-10


def test_revtex_shell_and_pending_note_compile(tmp_path: Path):
    source = EXPERIMENT / "docs" / "constraint_flag_pilot.tex"
    text = source.read_text()
    assert r"\documentclass[aps,prb,onecolumn,nofootinbib,superscriptaddress]{revtex4-2}" in text
    for token in (r"\mathfrak M", r"F_\star", r"\Delta_m", r"\epsilon_{j,t}", "Goldstein"):
        assert token in text
    if shutil.which("pdflatex") is None:
        pytest.skip("pdflatex unavailable")
    command = ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "-output-directory", str(tmp_path), str(source)]
    for _ in range(2):
        subprocess.run(command, cwd=source.parent, capture_output=True, text=True, check=True, timeout=120)
    assert (tmp_path / "constraint_flag_pilot.pdf").is_file()


def test_smoke_runner_writes_atomic_products(tmp_path: Path):
    output = tmp_path / "smoke"
    command = [sys.executable, str(EXPERIMENT / "run_pilot.py"), "smoke", "--workers", "2", "--bootstrap-samples", "20", "--output-root", str(output)]
    subprocess.run(command, cwd=REPO_ROOT, check=True, timeout=600)
    assert (output / "SUCCESS").is_file()
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["config"]["canonical_dynamics_entry_point"] == "classA_U1FGTN.run_markov_circuit"
    assert len(list((output / "stages" / "dynamics" / "cases").glob("*/revisit_raw.npz"))) == 8
    for stem in ("revisit_summary", "bootstrap_contrasts", "static_checkpoint_summary"):
        assert (output / "stages" / "analysis" / f"{stem}.csv").is_file()
        assert (output / "stages" / "analysis" / f"{stem}.parquet").is_file()
    for stem in ("flag_spectral_flow", "marginal_shell_frustration", "retention_profiles", "wall_excesses"):
        assert (output / "stages" / "analysis" / "figures" / f"{stem}.png").is_file()
        assert (output / "stages" / "analysis" / "figures" / f"{stem}.pdf").is_file()
