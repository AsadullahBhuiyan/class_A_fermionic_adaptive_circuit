from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[1]
PARENT = REPO / "00_WORKSPACE/CURRENT/final_production_new_designs"
BUNDLE = PARENT / "18_hard_wall_purification_alpha_endpoint"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


for candidate in (BUNDLE, BUNDLE / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))
OBSERVER = _load(BUNDLE / "endpoint_spectrum_observer.py", "tested_alpha_endpoint_observer")
RUNNER = _load(BUNDLE / "run_campaign.py", "tested_alpha_endpoint_runner")


def _source(cell: dict) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else str(source)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_locked_grid_lane_partition_and_batching() -> None:
    config = json.loads((BUNDLE / "campaign_config.json").read_text(encoding="utf-8"))
    RUNNER.validate_config(config)
    assert config == RUNNER.expected_config()
    assert config["Ny_values"] == [24, 28, 32, 36, 40, 44, 50]
    assert config["alpha_1_values"] == list(RUNNER.ALPHA_VALUES)
    assert config["execution_batch_size_by_Ny"] == {
        "24": 100,
        "28": 100,
        "32": 100,
        "36": 90,
        "40": 75,
        "44": 60,
        "50": 50,
    }
    assert RUNNER.DYNAMICS_MICROBATCH_SIZE_BY_NY == {
        24: 100,
        28: 100,
        32: 64,
        36: 50,
        40: 40,
        44: 30,
        50: 25,
    }
    all_tasks = RUNNER.expand_execution_batches(config)
    lane_a = RUNNER.expand_execution_batches(config, "A")
    lane_b = RUNNER.expand_execution_batches(config, "B")
    assert len(all_tasks) == 231
    assert len(lane_a) == 116 and len(lane_b) == 115
    assert sum(task.samples for task in lane_a) == 7400
    assert sum(task.samples for task in lane_b) == 7300
    assert len({(task.ny, task.alpha_1) for task in lane_a}) == 74
    assert len({(task.ny, task.alpha_1) for task in lane_b}) == 73
    assert {(task.ny, task.alpha_1) for task in lane_a}.isdisjoint(
        {(task.ny, task.alpha_1) for task in lane_b}
    )
    assert len(RUNNER.all_result_shards(config)) == 2940
    assert len({task.seed for task in all_tasks}) == 231
    identities = {
        (task.ny, task.alpha_1, sample)
        for task in all_tasks
        for sample in task.sample_indices.tolist()
    }
    assert len(identities) == 14_700
    assert lane_a[0].ny == 50 and lane_a[0].alpha_1 == 2.0
    assert lane_b[0].ny == 50 and np.isclose(abs(lane_b[0].alpha_1 - 2.0), 0.025)


def test_centered_transform_caps_and_gaps_match_direct_formula() -> None:
    a = np.asarray([-1.0, -0.8, -0.2, 0.1, 0.7, 1.0])
    values = OBSERVER.spectrum_scalars(a, cycles=50)
    expected = np.asarray(
        [np.inf, -np.arctanh(-0.8) / 50, -np.arctanh(-0.2) / 50,
         -np.arctanh(0.1) / 50, -np.arctanh(0.7) / 50, -np.inf]
    )
    np.testing.assert_allclose(values["rates"][1:-1], expected[1:-1], atol=1e-15)
    assert np.isposinf(values["rates"][0])
    assert np.isneginf(values["rates"][-1])
    assert values["positive_cap_count"] == 1
    assert values["negative_cap_count"] == 1
    assert values["finite_mode_count"] == 4
    assert values["lyapunov_half_gap"] == np.min(np.abs(expected[1:-1]))
    assert values["lyapunov_two_sided_gap"] == (
        np.min(expected[1:-1][expected[1:-1] > 0])
        - np.max(expected[1:-1][expected[1:-1] < 0])
    )

    within_roundoff = np.asarray([-1.0 - 5e-10, 1.0 + 5e-10])
    rates, finite, zero, pole = OBSERVER.centered_to_lyapunov(
        within_roundoff, cycles=10
    )
    assert not finite.any() and zero.tolist() == [False, True]
    assert pole.tolist() == [True, False]
    assert np.isposinf(rates[0]) and np.isneginf(rates[1])


def test_out_of_bounds_spectrum_fails_and_phase_fix_is_deterministic() -> None:
    try:
        OBSERVER.centered_to_lyapunov(np.asarray([1.0 + 2e-9]), cycles=10)
    except FloatingPointError:
        pass
    else:
        raise AssertionError("spectrum outside the tolerance did not fail")
    vectors = np.asarray(
        [[1j, 1 + 1j], [0.1, -3j], [0.0, 0.2]], dtype=np.complex128
    )
    fixed = OBSERVER.phase_fix_columns(vectors)
    pivots = np.argmax(np.abs(fixed), axis=0)
    pivot_values = fixed[pivots, np.arange(fixed.shape[1])]
    np.testing.assert_allclose(pivot_values.imag, 0.0, atol=1e-15)
    assert np.all(pivot_values.real >= 0.0)
    np.testing.assert_allclose(
        np.abs(fixed.conj().T @ fixed), np.abs(vectors.conj().T @ vectors)
    )


def test_endpoint_observer_is_read_only_and_saves_compact_non50_data() -> None:
    ny = 1
    active_count = 22 * ny
    full_count = 2 * 20 * ny
    diagonal = np.linspace(-0.8, 0.8, active_count)
    G = np.zeros((2, full_count, full_count), dtype=np.complex128)
    G[:, np.arange(active_count), np.arange(active_count)] = diagonal
    before = G.copy()
    np.random.seed(1818)
    torch.manual_seed(1818)
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state().clone()
    observer = OBSERVER.EndpointSpectrumObserver(
        nx=20,
        ny=ny,
        cycles=2 * ny,
        samples=2,
        active_indices=torch.arange(active_count),
        full_mode_count=full_count,
        sample_indices=np.asarray([3, 4]),
        sample_chunk=1,
    )
    observer.compute_endpoint(G)
    payload = observer.result_payload(slice(0, 2))
    np.testing.assert_array_equal(G, before)
    after_numpy = np.random.get_state()
    assert after_numpy[0] == numpy_state[0]
    np.testing.assert_array_equal(after_numpy[1], numpy_state[1])
    assert after_numpy[2:] == numpy_state[2:]
    assert torch.equal(torch.get_rng_state(), torch_state)
    assert payload["nearest_centered_eigenvalues"].shape == (2, 16)
    assert payload["nearest_lyapunov_exponents"].shape == (2, 16)
    assert "full_centered_occupation_spectrum" not in payload
    assert "selected_eigenvectors" not in payload


def test_notebooks_are_fixed_a100_lanes_with_native_progress_and_disconnect() -> None:
    for lane in RUNNER.LANES:
        path = BUNDLE / f"run_hard_wall_purification_alpha_endpoint_lane_{lane.lower()}.ipynb"
        notebook = json.loads(path.read_text(encoding="utf-8"))
        source = "\n".join(_source(cell) for cell in notebook["cells"])
        assert f"LANE = '{lane}'" in source
        assert "MAX_NEW_EXECUTION_BATCHES = 1" in source
        assert "gpu_gib < 38.0" in source
        assert "spec.loader.exec_module(runner)" in source
        assert "runner.main(runner_args)" in source
        assert "subprocess.run" not in source
        assert "REPORT_ONLY = False" in source
        final = _source(notebook["cells"][-1])
        assert final == "from google.colab import runtime\nruntime.unassign()\nprint('done')\n"


def test_manifest_and_canonical_sources_are_synchronized() -> None:
    manifest = json.loads((BUNDLE / "deployment_manifest.json").read_text(encoding="utf-8"))
    assert manifest["sampling_revision"] == RUNNER.SAMPLING_REVISION
    assert manifest["lane_contract"]["A"]["execution_batches"] == 116
    assert manifest["lane_contract"]["B"]["execution_batches"] == 115
    for relative, metadata in manifest["files"].items():
        path = BUNDLE / relative
        assert path.stat().st_size == metadata["bytes"]
        assert _sha(path) == metadata["sha256"]
    for filename in ("classA_U1FGTN_gpu.py", "occupied_frame_gpu.py"):
        assert _sha(BUNDLE / "src" / filename) == _sha(REPO / "src/fgtn" / filename)


def test_runner_uses_only_supported_canonical_engine_keywords() -> None:
    supported = set(inspect.signature(RUNNER.classA_U1FGTN_gpu.run_markov_circuit).parameters)
    source = (BUNDLE / "run_campaign.py").read_text(encoding="utf-8")
    import ast

    calls = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "run_markov_circuit"
    ]
    assert len(calls) == 1
    assert {keyword.arg for keyword in calls[0].keywords if keyword.arg is not None} <= supported


def test_final_checkpoint_is_removed_only_after_endpoint_and_shard_verification() -> None:
    source = (BUNDLE / "run_campaign.py").read_text(encoding="utf-8")
    endpoint = source.index("observer.compute_endpoint(G)")
    publish = source.index("publish_result(", endpoint)
    final_verify = source.index("batch result failed final verification", publish)
    remove = source.index("remove_checkpoint(output_root, task)", final_verify)
    assert endpoint < publish < final_verify < remove


def test_cuda_microbatch_is_separate_from_durable_batch_and_progress_is_not_duplicated() -> None:
    task = next(
        task
        for task in RUNNER.expand_execution_batches(RUNNER.expected_config(), "A")
        if task.ny == 50 and task.samples == 50
    )

    class FakeBar:
        def __init__(self) -> None:
            self.count = 0

        def update(self, count: int) -> None:
            self.count += count

    class FakeModel:
        def __init__(self) -> None:
            self.kwargs = None

        def run_markov_circuit(self, **kwargs):
            self.kwargs = kwargs
            observer = kwargs["cycle_observer"]
            for batch_index in range(2):
                for cycle in range(3):
                    observer(
                        cycle=cycle,
                        G=torch.zeros(1),
                        batch_index=batch_index,
                        batch_start=25 * batch_index,
                        batch_count=25,
                    )
            return {
                "state_representation_resolved": "covariance",
                "G_init_prepared": False,
                "exterior_preparation_performed": True,
                "G_final": np.zeros((task.samples, 1, 1), dtype=np.complex128),
            }

    model = FakeModel()
    progress = FakeBar()
    final = RUNNER.run_segment(
        model,
        RUNNER.expected_config(),
        task,
        object(),
        completed_cycle=0,
        segment_cycles=2,
        G_init=None,
        continuing=False,
        progress_bar=progress,
    )
    assert model.kwargs["samples"] == 50
    assert model.kwargs["batch_size"] == 25
    assert final.shape == (50, 1, 1)
    assert progress.count == 2


def test_pre_microbatch_source_identity_is_the_only_accepted_predecessor() -> None:
    current = RUNNER.source_hashes()
    assert RUNNER.source_hashes_are_compatible(current, current)
    assert RUNNER.source_hashes_are_compatible(
        RUNNER.PRE_MICROBATCH_SOURCE_HASHES, current
    )
    changed = dict(RUNNER.PRE_MICROBATCH_SOURCE_HASHES)
    changed["src/classA_U1FGTN_gpu.py"] = "0" * 64
    assert not RUNNER.source_hashes_are_compatible(changed, current)
    changed_current = dict(current)
    changed_current["src/classA_U1FGTN_gpu.py"] = "f" * 64
    assert not RUNNER.source_hashes_are_compatible(
        RUNNER.PRE_MICROBATCH_SOURCE_HASHES, changed_current
    )
