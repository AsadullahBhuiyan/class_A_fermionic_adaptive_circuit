from __future__ import annotations

import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]

TARGET_NOTEBOOKS = (
    "00_WORKSPACE/COLAB/colab_lyapunov/notebooks/data_generation/"
    "run_lyapunov_spectra_N20_Ny30-50_S25_cyclesNy.ipynb",
    "00_WORKSPACE/COLAB/colab_lyapunov/notebooks/data_generation/"
    "run_lyapunov_spectra_N20x40_smoke.ipynb",
    "00_WORKSPACE/COLAB/colab_no_feedback_alpha_sweep_transfer/notebooks/data_generation/"
    "run_no_feedback_alpha_sweep_dwtrunc1_N20_Ny20-30-40_S10_cycles2Ny_gpu.ipynb",
    "00_WORKSPACE/COLAB/colab_regularized_choi_transfer_matrix/notebooks/data_generation/"
    "run_particle_choi_transfer_gap_N20_Ny30-50_S25_cycles2Ny_gpu.ipynb",
    "00_WORKSPACE/COLAB/colab_regularized_choi_transfer_matrix/notebooks/data_generation/"
    "run_particle_choi_transfer_spectrum_N20x40_gpu.ipynb",
)

DUPLICATE_ENGINE_BAR_NOTEBOOKS = (
    "00_WORKSPACE/COLAB/colab_charge_fluctuations/notebooks/data_generation/"
    "run_purification_dynamics_maxmix_multi_geometry.ipynb",
    "00_WORKSPACE/COLAB/colab_charge_fluctuations/notebooks/data_generation/"
    "run_streaming_covariance_observables_multi_geometry.ipynb",
    "00_WORKSPACE/COLAB/colab_large_entanglement_scaling_N20/notebooks/"
    "run_slope_vs_cycle_resumable.ipynb",
    "00_WORKSPACE/COLAB/colab_large_entanglement_scaling_N20/notebooks/"
    "run_slope_vs_system_size.ipynb",
    "00_WORKSPACE/COLAB/colab_large_entanglement_scaling_N20/notebooks/"
    "run_slope_vs_system_size_time_scaling.ipynb",
    "00_WORKSPACE/COLAB/colab_partial_post-select/notebooks/"
    "run_partial_postselection_sweep_N20x40_gpu.ipynb",
    "00_WORKSPACE/COLAB/colab_small_system_testing/notebooks/data_generation/"
    "run_covariance_protocol_histories_multi_geometry.ipynb",
    "00_WORKSPACE/COLAB/colab_small_system_testing/notebooks/data_generation/"
    "run_pure_state_markov_snapshots.ipynb",
    "00_WORKSPACE/COLAB/colab_small_system_testing/notebooks/data_generation/"
    "run_purification_maxmix_entropy_contours.ipynb",
    "00_WORKSPACE/COLAB/colab_small_system_testing/notebooks/data_generation/"
    "run_streaming_covariance_observables_multi_geometry.ipynb",
    "00_WORKSPACE/COLAB/colab_small_system_testing/notebooks/data_generation/"
    "run_streaming_covariance_protocol_characterization_multi_geometry.ipynb",
    "00_WORKSPACE/COLAB/colab_small_system_testing/notebooks/data_generation/"
    "run_trace_real_vs_cycle_protocols.ipynb",
)


def _load_notebook(relative_path: str) -> dict:
    path = REPO_ROOT / relative_path
    return json.loads(path.read_text(encoding="utf-8"))


def _cell_source(cell: dict) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else str(source)


def _code_source(relative_path: str) -> str:
    notebook = _load_notebook(relative_path)
    return "\n".join(
        _cell_source(cell)
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )


@pytest.mark.parametrize("relative_path", TARGET_NOTEBOOKS)
def test_target_campaign_has_one_authoritative_progress_bar(relative_path: str) -> None:
    code = _code_source(relative_path)

    assert code.count("from tqdm.auto import tqdm") == 1
    assert code.count("tqdm(") == 1
    assert code.count("campaign_bar = tqdm(") == 1
    assert "unit='config'" in code
    assert "leave=True" in code
    assert "set_postfix_str" in code
    assert "elapsed=0s" in code
    assert "progress=True" not in code
    assert "progress=False" in code


@pytest.mark.parametrize("relative_path", TARGET_NOTEBOOKS)
def test_target_campaign_heartbeat_is_bounded_and_clean(relative_path: str) -> None:
    code = _code_source(relative_path)

    assert "HEARTBEAT_INTERVAL_S = 60.0" in code
    assert "@contextmanager" in code
    assert "while not stop_event.wait(interval_s)" in code
    assert "[heartbeat]" in code
    assert "still running; elapsed=" in code
    assert "stop_event.set()" in code
    assert "worker.join()" in code
    assert "with progress_heartbeat(campaign_bar, label):" in code


@pytest.mark.parametrize("relative_path", TARGET_NOTEBOOKS)
def test_target_notebook_code_cells_compile(relative_path: str) -> None:
    notebook = _load_notebook(relative_path)
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") != "code":
            continue
        compile(_cell_source(cell), f"{relative_path}#cell-{index}", "exec")


@pytest.mark.parametrize("relative_path", DUPLICATE_ENGINE_BAR_NOTEBOOKS)
def test_existing_notebook_bar_does_not_compete_with_engine_bar(relative_path: str) -> None:
    code = _code_source(relative_path)

    assert "from tqdm.auto import tqdm" in code
    assert "tqdm(" in code
    assert "progress=True" not in code
    assert "progress=False" in code


def test_single_configuration_campaign_owns_sample_cycle_bar() -> None:
    relative_path = (
        "00_WORKSPACE/COLAB/colab_large_entanglement_scaling_N20/notebooks/"
        "run_slope_vs_cycle.ipynb"
    )
    code = _code_source(relative_path)

    assert code.count("tqdm(") == 1
    assert "total=SAMPLES * CYCLES" in code
    assert "unit='sample-cycle'" in code
    assert "campaign_bar.update(int(batch_count))" in code
    assert "progress=True" not in code
    assert "progress=False" in code
