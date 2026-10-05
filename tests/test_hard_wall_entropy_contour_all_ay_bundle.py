from __future__ import annotations

import hashlib
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
PARENT = REPO / "00_WORKSPACE/CURRENT/final_production_new_designs"
BUNDLE = PARENT / "16_hard_wall_entropy_contour_all_ay"
NOTEBOOKS = {
    "A": BUNDLE / "run_lane_A_Ny50_Ny60.ipynb",
    "B": BUNDLE / "run_lane_B_Ny30_35_40_45_55.ipynb",
}
REVISION = "hard_wall_entropy_contour_all_ay_nx20_ny30-60_s100_2ny_raster_v3"


def _cell_source(cell: dict) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else str(source)


def _notebook(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _notebook_config(path: Path) -> tuple[dict, dict]:
    for cell in _notebook(path)["cells"]:
        source = _cell_source(cell)
        if cell.get("cell_type") == "code" and "CONFIG = {" in source:
            namespace: dict = {}
            exec(compile(source, str(path), "exec"), namespace)
            return namespace["CONFIG"], namespace
    raise AssertionError(f"configuration cell not found in {path}")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_two_notebooks_share_one_locked_contract_and_disjoint_lanes() -> None:
    config_a, namespace_a = _notebook_config(NOTEBOOKS["A"])
    config_b, namespace_b = _notebook_config(NOTEBOOKS["B"])
    assert config_a == config_b
    assert namespace_a["LANE"] == "A"
    assert namespace_b["LANE"] == "B"
    assert config_a["sampling_revision"] == REVISION
    assert config_a["root_seed"] == 2026091416
    assert config_a["Ny_values"] == [30, 35, 40, 45, 50, 55, 60]
    assert config_a["lane_Ny_values"] == {
        "A": [50, 60],
        "B": [30, 35, 40, 45, 55],
    }
    assert not set(config_a["lane_Ny_values"]["A"]) & set(
        config_a["lane_Ny_values"]["B"]
    )
    assert sorted(
        config_a["lane_Ny_values"]["A"] + config_a["lane_Ny_values"]["B"]
    ) == config_a["Ny_values"]
    assert config_a["execution_batch_size_by_Ny"] == {
        "30": 80,
        "35": 60,
        "40": 40,
        "45": 30,
        "50": 25,
        "55": 20,
        "60": 20,
    }


def test_notebooks_lock_hard_wall_science_and_stream_progress() -> None:
    config, _ = _notebook_config(NOTEBOOKS["A"])
    protocol = config["protocol"]
    assert config["Nx"] == 20
    assert config["samples_per_Ny"] == 100
    assert config["result_shard_size"] == 5
    assert config["cycles_rule"] == "2*Ny"
    assert config["segment_cycles"] == 5
    assert config["device"] == "cuda:0"
    assert config["dtype"] == "complex128"
    assert config["observer"]["endpoint_only"] is True
    assert config["observer"]["entropy_orders"] == [1, 2, 3]
    assert config["observer"]["origin_average"] == "all_periodic_y0_within_trajectory"
    assert config["observer"]["origin_coordinate"] == "relative_dy=(y-y0)_mod_Ny"
    assert config["observer"]["Ay_values"] == "0..Ny//2"
    assert config["observer"]["spatial_contours"] == ["von_neumann"]
    assert (
        config["observer"]["contour_retention"]
        == "trajectory_resolved_y0_average_all_Ay"
    )
    assert config["observer"]["matrix_batch_candidates"] == [8, 16, 32, 64, 80, 128]
    assert config["observer"]["benchmark_maximum_projected_hours"] == 1.0
    assert protocol == {
        "DW": True,
        "domain_wall_interval": [5, 15],
        "dw_truncation": True,
        "meas_slab_only": True,
        "nshell": 1,
        "alpha_1": 1.0,
        "alpha_2": 30.0,
        "filling_frac": 0.5,
        "trial_orbitals": "X",
        "init_mode": "default",
        "sequence": "raster_y",
        "perfect_correction": True,
        "postselect": False,
        "postselect_probability": 0.0,
        "n_a": 0.5,
        "state_representation": "physical_frame",
        "triv_region_local_mode": False,
    }
    for lane, path in NOTEBOOKS.items():
        notebook = _notebook(path)
        joined = "\n".join(_cell_source(cell) for cell in notebook["cells"])
        assert "drive.mount('/content/drive')" in joined
        assert "shutil.copytree(BUNDLE_DRIVE_DIR, LOCAL_BUNDLE_DIR)" in joined
        assert "stdout=subprocess.PIPE" in joined
        assert "os.read(process.stdout.fileno(), 4096)" in joined
        assert "--max-new-execution-batches" in joined
        assert "REPORT_ONLY = False" in joined
        assert "BENCHMARK_ONLY = False" in joined
        assert "--benchmark-only" in joined
        assert "endpoint-$A_y$ progress" in joined
        assert "A100" in joined and "complex128" in joined
        code_cells = [
            _cell_source(cell)
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
        ]
        assert code_cells[-1].strip() == (
            "from google.colab import runtime\n"
            "runtime.unassign()\n"
            "print('done')"
        )
        if lane == "A":
            assert "RUN_ANALYSIS = False" in joined
            assert "analyze_campaign.py" in joined
        else:
            assert "RUN_ANALYSIS" not in joined


def test_bundle_is_registered_and_source_copies_are_canonical() -> None:
    index = json.loads((PARENT / "bundle_index.json").read_text(encoding="utf-8"))
    assert index["standalone_contracts"][BUNDLE.name] == REVISION
    assert BUNDLE.name in index["bundles"]
    assert _sha256(BUNDLE / "src/classA_U1FGTN_gpu.py") == _sha256(
        REPO / "src/fgtn/classA_U1FGTN_gpu.py"
    )
    assert _sha256(BUNDLE / "src/occupied_frame_gpu.py") == _sha256(
        REPO / "src/fgtn/occupied_frame_gpu.py"
    )
    engine_source = (BUNDLE / "src/classA_U1FGTN_gpu.py").read_text(
        encoding="utf-8"
    )
    assert "frame_init_prepared=False" in engine_source
    assert "skipped_prepared_frame" in engine_source
    manifest = json.loads((BUNDLE / "bundle_manifest.json").read_text(encoding="utf-8"))
    for relative, expected in manifest["files"].items():
        path = BUNDLE / relative
        assert path.stat().st_size == expected["bytes"]
        assert _sha256(path) == expected["sha256"]
