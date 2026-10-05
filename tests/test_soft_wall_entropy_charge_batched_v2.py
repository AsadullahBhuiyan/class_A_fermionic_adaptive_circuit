from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


REPO = Path(__file__).resolve().parents[1]
PARENT = REPO / "00_WORKSPACE/CURRENT/final_production_new_designs"
BUNDLE = PARENT / "10_soft_wall_entropy_charge_batched_v2"
REVISION = "soft_wall_entropy_charge_nx20_ny30-60_s100_2ny_raster_v2_batched_endpoint"


def _load_runner():
    old_path = list(sys.path)
    old_observer = sys.modules.pop("entropy_charge_observer", None)
    try:
        sys.path.insert(0, str(BUNDLE))
        spec = importlib.util.spec_from_file_location(
            "tested_soft_wall_entropy_charge_runner", BUNDLE / "run_campaign.py"
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path[:] = old_path
        if old_observer is not None:
            sys.modules["entropy_charge_observer"] = old_observer
        else:
            sys.modules.pop("entropy_charge_observer", None)


RUNNER = _load_runner()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _notebook_config(path: Path) -> tuple[dict, dict]:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        source = "".join(cell.get("source", []))
        if cell.get("cell_type") == "code" and "CONFIG = {" in source:
            namespace: dict = {}
            exec(compile(source, str(path), "exec"), namespace)
            return namespace["CONFIG"], namespace
    raise AssertionError(f"configuration cell not found in {path}")


def test_soft_contract_is_isolated_and_expands_exact_workload() -> None:
    config = RUNNER.expected_config()
    assert config["sampling_revision"] == REVISION
    assert config["root_seed"] == 2026090305
    assert config["protocol"]["DW"] is True
    assert config["protocol"]["dw_truncation"] is False
    assert config["protocol"]["meas_slab_only"] is False
    assert config["protocol"]["nshell"] == 1
    assert config["protocol"]["perfect_correction"] is True
    assert config["protocol"]["sequence"] == "raster_y"
    assert config["dtype"] == "complex128"
    assert config["cycles_rule"] == "2*Ny"
    tasks = RUNNER.expand_execution_batches(config, lane="A") + RUNNER.expand_execution_batches(
        config, lane="B"
    )
    shards = [shard for task in tasks for shard in RUNNER.result_shards(task)]
    assert len(tasks) == 25
    assert len(shards) == 140
    assert sum(task.sample_count for task in tasks) == 700
    assert all(shard.sample_count == 5 for shard in shards)
    for ny in RUNNER.EXPECTED_NY_VALUES:
        sample_ids = sorted(
            sample
            for shard in shards
            if shard.ny == ny
            for sample in shard.global_sample_indices
        )
        assert sample_ids == list(range(100))


def test_soft_model_uses_full_layer_and_untruncated_ow_support(monkeypatch) -> None:
    captured: dict = {}

    class FakeModel:
        DW_loc = (5, 15)
        dtype = torch.complex128

        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.ny = int(kwargs["Ny"])

        def active_top_layer_indices(self, meas_slab_only: bool):
            assert meas_slab_only is False
            return torch.arange(2 * RUNNER.EXPECTED_NX * self.ny)

    monkeypatch.setattr(RUNNER, "classA_U1FGTN_gpu", FakeModel)
    RUNNER.build_model(RUNNER.expected_config(), ny=30)
    assert captured["DW"] is True
    assert captured["dw_truncation"] is False
    assert captured["nshell"] == 1
    assert captured["alpha_1"] == 1.0
    assert captured["alpha_2"] == 30.0
    assert captured["dtype"] == "complex128"


class _CycleRecorder:
    def __init__(self) -> None:
        self.cycles: list[int] = []

    def __call__(self, *, cycle: int, **_: object) -> None:
        self.cycles.append(int(cycle))


class _SoftFakeModel:
    device = "cpu"

    def __init__(self, task) -> None:
        self.task = task
        self.calls: list[dict] = []

    def run_markov_circuit(self, **kwargs):
        self.calls.append(kwargs)
        observer = kwargs["native_cycle_observer"]
        for cycle in range(int(kwargs["cycles"]) + 1):
            observer(
                cycle=cycle,
                state=object(),
                batch_index=0,
                batch_start=0,
                batch_count=self.task.sample_count,
            )
        return {
            "samples": self.task.sample_count,
            "state_representation_resolved": "physical_frame",
            "covariance_materialization_count": 0,
            "choi_tracked": False,
            "frame_init_prepared": bool(kwargs["frame_init_prepared"]),
            "exterior_preparation_performed": False,
            "exterior_preparation": None,
            "exterior_preparation_mode": None,
            "native_final": {
                "frame": np.zeros(
                    (self.task.sample_count, 2 * RUNNER.EXPECTED_NX * self.task.ny, 1),
                    dtype=np.complex128,
                ),
                "ranks": np.zeros(self.task.sample_count, dtype=np.int64),
            },
        }


def test_segmented_soft_run_maps_cycles_without_exterior_preparation() -> None:
    task = RUNNER.expand_execution_batches(RUNNER.expected_config(), lane="A")[-1]
    model = _SoftFakeModel(task)
    observer = _CycleRecorder()
    native, _ = RUNNER._run_segment(
        model=model,
        config=RUNNER.expected_config(),
        task=task,
        observer=observer,
        segment_start=0,
        frame=None,
        ranks=None,
    )
    assert observer.cycles == [0, 1, 2, 3, 4, 5]
    assert model.calls[-1]["frame_init_prepared"] is False
    assert model.calls[-1]["meas_slab_only"] is False
    RUNNER._run_segment(
        model=model,
        config=RUNNER.expected_config(),
        task=task,
        observer=observer,
        segment_start=5,
        frame=native["frame"],
        ranks=native["ranks"],
    )
    assert observer.cycles == list(range(11))
    assert model.calls[-1]["frame_init_prepared"] is True


def _run_small_soft_engine(model, *, cycles, frame=None, ranks=None, prepared=False):
    return model.run_markov_circuit(
        G_history=False,
        progress=False,
        cycles=cycles,
        postselect=False,
        postselect_probability=0.0,
        perfect_correction=True,
        samples=2,
        init_mode="default",
        frame_init=frame,
        frame_ranks=ranks,
        frame_init_prepared=prepared,
        save=False,
        n_a=0.5,
        sequence="raster_y",
        meas_slab_only=False,
        batch_size=2,
        return_data=True,
        state_representation="physical_frame",
        return_native_state=True,
        require_no_covariance_materialization=True,
        frame_reorthonormalize_interval=1,
    )


def test_real_canonical_soft_continuation_is_bitwise_exact() -> None:
    def new_model():
        return RUNNER.classA_U1FGTN_gpu(
            Nx=4,
            Ny=4,
            DW=True,
            nshell=1,
            filling_frac=0.5,
            alpha_1=1.0,
            alpha_2=30.0,
            trial_orbitals="X",
            dw_truncation=False,
            triv_region_local_mode=False,
            device="cpu",
            dtype="complex128",
            backend="local",
        )

    seed = 84173
    np.random.seed(seed)
    torch.manual_seed(seed)
    uninterrupted = _run_small_soft_engine(new_model(), cycles=6)
    uninterrupted_numpy = np.random.get_state()
    uninterrupted_torch = torch.get_rng_state().clone()

    np.random.seed(seed)
    torch.manual_seed(seed)
    segmented_model = new_model()
    first = _run_small_soft_engine(segmented_model, cycles=3)
    checkpoint_numpy = np.random.get_state()
    checkpoint_torch = torch.get_rng_state().clone()
    np.random.seed(1)
    torch.manual_seed(2)
    np.random.set_state(checkpoint_numpy)
    torch.set_rng_state(checkpoint_torch)
    resumed = _run_small_soft_engine(
        segmented_model,
        cycles=3,
        frame=first["native_final"]["frame"],
        ranks=first["native_final"]["ranks"],
        prepared=True,
    )
    np.testing.assert_array_equal(
        resumed["native_final"]["frame"], uninterrupted["native_final"]["frame"]
    )
    np.testing.assert_array_equal(
        resumed["native_final"]["ranks"], uninterrupted["native_final"]["ranks"]
    )
    assert np.random.get_state()[0] == uninterrupted_numpy[0]
    np.testing.assert_array_equal(np.random.get_state()[1], uninterrupted_numpy[1])
    assert np.random.get_state()[2:] == uninterrupted_numpy[2:]
    assert torch.equal(torch.get_rng_state(), uninterrupted_torch)
    assert first["exterior_preparation"] is None
    assert first["exterior_preparation_performed"] is False
    assert resumed["frame_init_prepared"] is True
    assert resumed["exterior_preparation"] is None
    assert resumed["exterior_preparation_performed"] is False


def test_soft_observer_records_only_charge_during_dynamics(monkeypatch) -> None:
    observer = RUNNER.SoftWallEntropyChargeObserver(
        nx=2, ny=4, physical_cycles=8, sample_ids=[0, 1]
    )

    class State:
        ranks = torch.tensor([7, 9])
        frame = torch.empty(0)

    monkeypatch.setattr(
        torch.linalg,
        "eigvalsh",
        lambda *_a, **_k: (_ for _ in ()).throw(
            AssertionError("endpoint eigensolver ran during dynamics")
        ),
    )
    for cycle in range(9):
        observer(cycle=cycle, state=State(), batch_start=0, batch_count=2)
    observer.validate(require_dynamics=True, require_endpoint=False)
    assert np.array_equal(observer.global_charge[:, 0], [7, 9])


def test_notebooks_are_soft_specific_streamed_and_disconnect() -> None:
    notebooks = {
        "A": BUNDLE / "run_lane_A_Ny40_Ny60.ipynb",
        "B": BUNDLE / "run_lane_B_endpoint_Ny30_35_45_50_55.ipynb",
    }
    configs = []
    for lane, path in notebooks.items():
        config, namespace = _notebook_config(path)
        configs.append(config)
        assert namespace["LANE"] == lane
        notebook = json.loads(path.read_text(encoding="utf-8"))
        joined = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
        assert "soft-wall entropy/charge v2" in joined
        assert "10_soft_wall_entropy_charge_batched_v2" in joined
        assert "soft_wall_entropy_charge_nx20_ny30-60" in joined
        assert "stdout=subprocess.PIPE" in joined
        assert "os.read(process.stdout.fileno(), 4096)" in joined
        assert "REPORT_ONLY = False" in joined
        assert "BENCHMARK_ONLY = False" in joined
        assert "A100" in joined and "complex128" in joined
        code_cells = [
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
        ]
        assert code_cells[-1].strip() == (
            "from google.colab import runtime\n"
            "runtime.unassign()\n"
            "print('done')"
        )
    assert configs[0] == configs[1] == RUNNER.expected_config()


def test_registration_manifest_and_canonical_source_identity() -> None:
    index = json.loads((PARENT / "bundle_index.json").read_text(encoding="utf-8"))
    assert index["standalone_contracts"][BUNDLE.name] == REVISION
    assert BUNDLE.name in index["bundles"]
    assert _sha256(BUNDLE / "src/classA_U1FGTN_gpu.py") == _sha256(
        REPO / "src/fgtn/classA_U1FGTN_gpu.py"
    )
    assert _sha256(BUNDLE / "src/occupied_frame_gpu.py") == _sha256(
        REPO / "src/fgtn/occupied_frame_gpu.py"
    )
    manifest = json.loads((BUNDLE / "bundle_manifest.json").read_text(encoding="utf-8"))
    assert manifest["bundle"] == BUNDLE.name
    assert manifest["sampling_revision"] == REVISION
    for relative, expected in manifest["files"].items():
        path = BUNDLE / relative
        assert path.stat().st_size == expected["bytes"]
        assert _sha256(path) == expected["sha256"]


def test_soft_v2_cannot_accept_hard_v2_identity() -> None:
    hard_revision = (
        "hard_wall_entropy_charge_nx20_ny30-60_s100_2ny_raster_v2_batched_endpoint"
    )
    config = RUNNER.expected_config()
    altered = json.loads(json.dumps(config))
    altered["sampling_revision"] = hard_revision
    altered["protocol"]["dw_truncation"] = True
    altered["protocol"]["meas_slab_only"] = True
    with pytest.raises(ValueError, match="locked campaign contract"):
        RUNNER.validate_config(altered)
