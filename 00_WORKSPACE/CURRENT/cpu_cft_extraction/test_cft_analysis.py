from __future__ import annotations

import sys
import subprocess
import json
import csv
from pathlib import Path

import numpy as np
import pytest

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parents[0]
SRC = ROOT / "src"
for path in (THIS_DIR, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from cft_analysis import additive_fock_gap, choi_finite_rapidities, fit_campaign, min_abs_gap
from fgtn.classA_U1FGTN import classA_U1FGTN


def test_ceff_fit_synthetic():
    c_eff = 0.75
    f_inf = 1.2
    rows = []
    for L in (12, 16, 20, 24):
        f0 = f_inf - np.pi * c_eff / (6.0 * L * L)
        rows.append({
            "L": L,
            "f0": f0,
            "tangent_fock_gap": 2.0 * np.pi * 0.3 / L,
            "choi_fock_gap": 2.0 * np.pi * 0.4 / L,
        })
    fit = fit_campaign(rows, alpha=1.0)
    assert fit["ceff"] == pytest.approx(c_eff, rel=1e-10, abs=1e-10)
    assert fit["x_tangent_fock"] == pytest.approx(0.3, rel=1e-10, abs=1e-10)
    assert fit["x_choi_fock"] == pytest.approx(0.4, rel=1e-10, abs=1e-10)


def test_choi_endpoint_classification():
    info = choi_finite_rapidities(np.array([-1.0, -0.5, 0.0, 0.5, 1.0]), cycle=10, endpoint_tol=1e-8)
    assert info["endpoint_plus"] == 1
    assert info["endpoint_minus"] == 1
    assert info["finite_count"] == 3
    assert np.all(np.isfinite(info["rapidities"]))


def test_min_abs_gap_skips_floor_and_nonfinite():
    assert min_abs_gap([np.nan, 0.0, -0.2, 0.1], floor=1e-12) == pytest.approx(0.1)


def test_additive_fock_gap_sums_smallest_positive_gaps():
    assert additive_fock_gap([np.nan, 0.0, -0.2, 0.1, 0.3], rank=2, floor=1e-12) == pytest.approx(0.3)


def test_cpu_observers_work_with_choi_tracking():
    model = classA_U1FGTN(Nx=4, Ny=4, DW=True, nshell=1, alpha_1=1.0, alpha_2=30.0, dw_truncation=True)
    weights = []
    tangent_cycles = []
    choi_cycles = []

    def weight_observer(**payload):
        weights.append(payload)

    def lyapunov_observer(**payload):
        tangent_cycles.append(payload["cycle"])

    def choi_observer(**payload):
        choi_cycles.append(payload["cycle"])

    model.run_markov_circuit(
        G_history=False,
        progress=False,
        cycles=1,
        samples=1,
        save=False,
        sequence="raster_y",
        meas_slab_only=True,
        parallelize_samples=False,
        perfect_correction=True,
        track_choi=True,
        choi_observer=choi_observer,
        choi_observer_cycles=[1],
        choi_failure_mode="censor",
        trajectory_weight_observer=weight_observer,
        lyapunov_observer=lyapunov_observer,
    )
    assert weights
    assert tangent_cycles == [1]
    assert choi_cycles == [1]


def test_driver_refuses_postselection():
    script = THIS_DIR / "run_cpu_cft_sweep.py"
    result = subprocess.run(
        [sys.executable, str(script), "--smoke", "--postselect"],
        cwd=str(ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert result.returncode != 0
    assert "excluded from the c_eff extraction" in result.stderr


def test_driver_parallel_smoke_outputs(tmp_path):
    script = THIS_DIR / "run_cpu_cft_sweep.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--smoke",
            "--nx",
            "4",
            "--ny",
            "4",
            "--samples",
            "2",
            "--cycles-factor",
            "0.25",
            "--sample-workers",
            "2",
            "--blas-threads",
            "1",
            "--output-root",
            str(tmp_path),
        ],
        cwd=str(ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    campaigns = [path for path in tmp_path.iterdir() if path.is_dir()]
    assert len(campaigns) == 1
    campaign = campaigns[0]
    manifest = json.loads((campaign / "manifest.json").read_text())
    assert manifest["sample_workers"] == 2
    rows = list(csv.DictReader((campaign / "N4x4" / "trajectory_weights.csv").open()))
    assert sorted({int(row["sample_index"]) for row in rows}) == [0, 1]
    assert sorted({int(row["cycle"]) for row in rows}) == [1]
    tangent = np.load(campaign / "N4x4" / "tangent_lyapunov.npz", allow_pickle=True)
    choi = np.load(campaign / "N4x4" / "choi_rapidity.npz", allow_pickle=True)
    assert tangent["spectra"].shape[0] == 2
    assert choi["rapidities"].shape[0] == 2


def test_driver_benchmark_report(tmp_path):
    script = THIS_DIR / "run_cpu_cft_sweep.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--nx",
            "4",
            "--benchmark-workers",
            "1",
            "2",
            "--benchmark-samples",
            "2",
            "--benchmark-ny",
            "4",
            "--benchmark-cycles",
            "1",
            "--blas-threads",
            "1",
            "--output-root",
            str(tmp_path),
        ],
        cwd=str(ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    reports = list(tmp_path.glob("benchmark_*/benchmark_summary.json"))
    assert len(reports) == 1
    payload = json.loads(reports[0].read_text())
    assert payload["benchmark_kind"] == "sample_worker_core_tuning"
    assert len(payload["rows"]) == 2
    assert payload["recommended"] is not None
