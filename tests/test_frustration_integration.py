from pathlib import Path
import json
import shutil
import subprocess
import sys

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CURRENT = REPO_ROOT / "00_WORKSPACE" / "CURRENT"
sys.path.insert(0, str(CURRENT))

from topological_frustration_diagnostics.run_cpu import build_parser, nshell_values


CLI = CURRENT / "topological_frustration_diagnostics" / "run_cpu.py"
DOCS = CURRENT / "topological_frustration_diagnostics" / "docs"


def _run_cli(*arguments: str, timeout: int = 240) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(CLI), *arguments],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=True,
    )


@pytest.fixture(scope="module")
def smoke_output(tmp_path_factory: pytest.TempPathFactory) -> Path:
    output = tmp_path_factory.mktemp("frustration_diagnostics")
    _run_cli(
        "all",
        "--smoke",
        "--output-dir",
        str(output),
        "--bootstrap-samples",
        "4",
    )
    return output


def test_reference_campaign_defaults_and_explicit_convergence_controls():
    parser = build_parser()
    for mode in ("static", "activity", "spectral", "response"):
        assert nshell_values(parser.parse_args([mode]), mode) == (1,)
        assert nshell_values(parser.parse_args([mode, "--smoke"]), mode) == (1,)
    assert nshell_values(parser.parse_args(["static", "--nshell", "1", "2", "none"]), "static") == (
        1,
        2,
        None,
    )
    assert nshell_values(parser.parse_args(["activity", "--nshell", "1", "2"]), "activity") == (1, 2)


def test_all_smoke_modes_write_expected_schemas_and_finite_core_outputs(smoke_output: Path):
    expected_counts = {
        "static_completion.npz": 2,
        "activity_raw.npz": 2,
        "activity_analysis.npz": 2,
        "spectral_diagnostics.npz": 4,
        "response_raw.npz": 4,
        "response_analysis.npz": 4,
    }
    for filename, count in expected_counts.items():
        assert len(list(smoke_output.rglob(filename))) == count

    summaries = list(smoke_output.rglob("run_summary.json"))
    assert len(summaries) == 12
    response_sequences = {
        json.loads(path.read_text())["config"]["sequence"]
        for path in summaries
        if json.loads(path.read_text())["config"]["diagnostic"] == "response"
    }
    assert response_sequences == {"random", "raster_y"}
    for path in summaries:
        payload = json.loads(path.read_text())
        assert payload["canonical_dynamics_entry_point"] == "classA_U1FGTN.run_markov_circuit"
        assert payload["config"]["alpha_1"] == 1
        assert payload["config"]["alpha_2"] == 30
        assert payload["active_top_layer_indices"]
        if payload["config"]["diagnostic"] != "static":
            assert payload["config"]["rng_streams"] == [
                "initialization",
                "exterior",
                "schedule",
                "dynamics",
            ]
        assert (path.parent / "scalar_metrics.csv").is_file()
        assert len(list((path.parent / "figures").glob("*.png"))) >= 1
        assert len(list((path.parent / "figures").glob("*.pdf"))) >= 1

    for path in smoke_output.rglob("*.npz"):
        with np.load(path) as data:
            assert "G_hist" not in data.files
            assert "G_final" not in data.files
            for key in data.files:
                values = np.asarray(data[key])
                if np.issubdtype(values.dtype, np.number):
                    assert not np.any(np.isinf(values)), f"infinite values in {path}:{key}"

    for path in smoke_output.rglob("static_completion.npz"):
        with np.load(path) as data:
            assert np.isfinite(data["f_star"])
            np.testing.assert_allclose(data["f_star"], data["f_star_formula"], atol=1e-10)

    for path in smoke_output.rglob("activity_raw.npz"):
        with np.load(path) as data:
            np.testing.assert_array_equal(data["defect_X"], np.abs(data["transfer_Y"]))
            assert np.all((data["success_probability"][data["valid"]] >= 0.0))
            assert np.all((data["success_probability"][data["valid"]] <= 1.0))

    for path in smoke_output.rglob("activity_analysis.npz"):
        with np.load(path) as data:
            all_region = list(data["region_names"]).index("all")
            zero_field = int(np.argmin(np.abs(data["s_grid"])))
            assert np.all(np.isfinite(data["theta"][:, :, all_region]))
            np.testing.assert_allclose(data["theta"][:, :, all_region, zero_field], 0.0, atol=1e-14)

    for path in smoke_output.rglob("spectral_diagnostics.npz"):
        with np.load(path) as data:
            assert np.all(np.isfinite(data["lyapunov_gap"]))
            assert np.all(np.isfinite(data["choi_gap"][data["choi_active"]]))
            assert np.all(data["choi_finite_count"][data["choi_active"]] > 0)

    for path in smoke_output.rglob("response_raw.npz"):
        with np.load(path) as data:
            initial_charge = np.sum(data["delta_charge"][:, :, 0], axis=(2, 3))
            np.testing.assert_allclose(initial_charge, 1.0, atol=1e-12)

    for path in smoke_output.rglob("response_analysis.npz"):
        with np.load(path) as data:
            wall_count = data["wall_x"].size
            assert data["velocity_mean"].shape == (wall_count, wall_count)
            assert np.all(np.isfinite(np.diag(data["velocity_mean"])))


def test_analyze_mode_regenerates_every_smoke_figure(smoke_output: Path):
    result = _run_cli(
        "analyze",
        "--input-dir",
        str(smoke_output),
        "--output-dir",
        str(smoke_output),
    )
    report = json.loads(result.stdout[result.stdout.index("{") :])
    assert report["mode"] == "analyze"
    assert report["runs"] == 12
    assert (smoke_output / "analyze_campaign_summary.json").is_file()


def test_revtex_note_defines_diagnostics_and_compiles(tmp_path: Path):
    source = DOCS / "numerical_diagnostics.tex"
    text = source.read_text()
    assert r"\documentclass[aps,prb,onecolumn,nofootinbib,superscriptaddress]{revtex4-2}" in text
    for symbol in (
        r"\Phi",
        r"F_\star",
        r"f_\star(x,y)",
        r"\theta_R",
        r"\Delta_{\mathrm{Choi}}",
        r"\delta n",
    ):
        assert symbol in text

    implementation_text = CLI.read_text() + "\n".join(
        path.read_text() for path in (REPO_ROOT / "src" / "fgtn" / "diagnostics").glob("*.py")
    )
    for output_key in (
        "principal_cosines",
        "overlap_phi",
        "f_star",
        "defect_X",
        "transfer_Y",
        "theta",
        "lyapunov_spectrum",
        "choi_spectrum",
        "velocity_mean",
    ):
        assert output_key in implementation_text

    pdflatex = shutil.which("pdflatex")
    if pdflatex is None:
        pytest.skip("pdflatex is not installed")
    command = [
        pdflatex,
        "-interaction=nonstopmode",
        "-halt-on-error",
        "-output-directory",
        str(tmp_path),
        str(source),
    ]
    for _ in range(2):
        subprocess.run(command, cwd=DOCS, text=True, capture_output=True, check=True, timeout=120)
    assert (tmp_path / "numerical_diagnostics.pdf").is_file()
