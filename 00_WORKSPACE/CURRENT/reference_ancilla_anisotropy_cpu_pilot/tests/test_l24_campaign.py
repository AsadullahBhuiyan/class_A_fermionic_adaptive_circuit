from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np

EXPERIMENT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT.parents[2]
sys.path[:0] = [str(EXPERIMENT), str(REPO_ROOT / "src")]

from fgtn.classA_U1FGTN import classA_U1FGTN
from launch_tmux import build_tmux_command, choose_cpus, physical_cpus
from reference_probe import ReferencePairObserver, anisotropy_from_matching_time
import run_l24_campaign as campaign
from run_l24_campaign import (
    bracket_details,
    assert_checkpoint_compatible,
    initialize_run,
    load_config,
    model_checkpoint_signature,
    paired_point_gate,
    refinement_separations,
    reset_post_insertion_rngs,
    signature_diff,
    select_equilibration_multiplier,
    select_follow_multiplier,
)


def small_model() -> classA_U1FGTN:
    model = classA_U1FGTN(4, 4, DW=True, nshell=1, alpha_1=1, alpha_2=30, trial_orbitals="X", dw_truncation=True)
    model.construct_OW_projectors(nshell=1, DW=True, trial_orbitals="X", dw_truncation=True)
    return model


def test_locked_l24_conversion_and_support_terminated_flags():
    config = load_config()
    assert config["geometry"]["Ny"] == 24
    assert config["geometry"]["dw_truncation"] is True
    assert config["dynamics"]["meas_slab_only"] is True
    np.testing.assert_allclose(
        anisotropy_from_matching_time(24, 1.0),
        24.0 * np.log1p(np.sqrt(2.0)) / np.pi,
    )


def test_v3_config_and_template_signature_diagnostics():
    config = load_config()
    assert config["schema_version"] == 3
    assert config["reuse"]["partial_reference_branches"] is False
    model = small_model()
    clone = copy.deepcopy(model)
    signature = model_checkpoint_signature(model)
    assert signature_diff(signature, model_checkpoint_signature(clone)) == {}
    changed = copy.deepcopy(signature)
    changed["twist_y"] = 0.125
    differences = signature_diff(changed, signature)
    assert list(differences) == ["twist_y"]
    try:
        assert_checkpoint_compatible({"signature": changed}, clone, "synthetic-task")
    except RuntimeError as exc:
        assert "synthetic-task" in str(exc)
        assert "twist_y" in str(exc)
    else:
        raise AssertionError("Field-level checkpoint mismatch was not rejected.")


def test_model_template_preparation_is_idempotent(monkeypatch):
    model = small_model()
    signature = model_checkpoint_signature(model)
    monkeypatch.setattr(campaign, "_WORKER_MODEL_TEMPLATE", model)
    monkeypatch.setattr(campaign, "_WORKER_MODEL_SIGNATURE", signature)
    monkeypatch.setattr(campaign, "build_model", lambda *_: (_ for _ in ()).throw(AssertionError("rebuilt")))
    assert campaign.prepare_model_template() == signature
    assert campaign.prepare_model_template() == signature


def test_adaptive_refinement_and_invalid_late_crossing():
    summary = {4: {"bracket": {"lower": 6, "upper": 9, "time_star": 7.0}}}
    assert refinement_separations(summary) == [7, 8]
    assert bracket_details([1, 3, 6], [0.1, 0.5, 0.05], 0.3) is None
    bracket = bracket_details([1, 3, 6], [0.8, 0.5, 0.2], 0.35)
    assert bracket is not None
    assert bracket["lower"] == 3 and bracket["upper"] == 6


def test_paired_gate_accepts_zero_shift_and_rejects_large_shift():
    baseline = np.linspace(0.8, 1.2, 32)
    accepted = paired_point_gate(baseline, baseline.copy(), np.random.default_rng(1), 500, 0.5)
    rejected = paired_point_gate(baseline, baseline + 0.2, np.random.default_rng(2), 500, 0.5)
    assert accepted["passed"]
    assert not rejected["passed"]


def test_follow_selector_exercises_initial_and_extension_paths():
    assert select_follow_multiplier({"1_vs_2": {"passed": True}}) == 2
    assert select_follow_multiplier({"1_vs_2": {"passed": False}, "2_vs_3": {"passed": True}}) == 3
    assert select_follow_multiplier({"1_vs_2": {"passed": False}, "2_vs_3": {"passed": False}, "3_vs_4": {"passed": True}}) == 4
    assert select_follow_multiplier({"1_vs_2": {"passed": False}, "2_vs_3": {"passed": False}, "3_vs_4": {"passed": False}}) is None


def test_equilibration_selector_exercises_initial_and_8l_extension_paths():
    eqs = [2, 3, 4, 6]
    initial = {"2_vs_3": {"passed": False}, "3_vs_4": {"passed": True}, "4_vs_6": {"passed": True}}
    assert select_equilibration_multiplier(eqs, initial) == 3
    failed = {key: {"passed": False} for key in ("2_vs_3", "3_vs_4", "4_vs_6")}
    assert select_equilibration_multiplier(eqs, failed) is None
    assert select_equilibration_multiplier(eqs, failed, {"passed": True}) == 6
    assert select_equilibration_multiplier(eqs, failed, {"passed": False}) is None


def test_resume_rejects_source_hash_mismatch(tmp_path):
    initialize_run(tmp_path, resume=False)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["source_sha256"][next(iter(manifest["source_sha256"]))] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))
    try:
        initialize_run(tmp_path, resume=True)
    except RuntimeError as exc:
        assert "source hashes" in str(exc)
    else:
        raise AssertionError("Source-hash mismatch was not rejected.")


def test_post_insertion_rng_reset_is_matched_and_nonmutating():
    template = {
        "rng_states": {
            "initialization": {"token": 1},
            "exterior": {"token": 2},
            "schedule": np.random.default_rng(1).bit_generator.state,
            "dynamics": np.random.default_rng(2).bit_generator.state,
        }
    }
    first = reset_post_insertion_rngs(template, 91)
    second = reset_post_insertion_rngs(copy.deepcopy(template), 91)
    assert first["rng_states"]["schedule"] == second["rng_states"]["schedule"]
    assert first["rng_states"]["dynamics"] == second["rng_states"]["dynamics"]
    assert template["rng_states"]["schedule"] != first["rng_states"]["schedule"]


def test_canonical_checkpoint_resume_repeats_reference_branch():
    model = small_model()
    captured = {}

    def checkpoint_observer(*, cycle, state, **_):
        if cycle == 1:
            captured["state"] = state

    model.run_markov_circuit(
        cycles=1,
        samples=1,
        sequence="random",
        perfect_correction=True,
        G_history=False,
        save=False,
        progress=False,
        random_seed=123,
        state_representation="physical_frame",
        return_native_state=True,
        meas_slab_only=True,
        checkpoint_observer=checkpoint_observer,
    )
    outputs = []
    for _ in range(2):
        branch_model = small_model()
        wall = int(branch_model.DW_loc[0])
        observer = ReferencePairObserver(nx=4, ny=4, tau1=2, tau2=3, follow_cycles=1, first_site=(wall, 0), second_site=(wall, 0), rng=np.random.default_rng(44))
        branch_model.run_markov_circuit(
            cycles=4,
            samples=1,
            sequence="random",
            perfect_correction=True,
            G_history=False,
            save=False,
            progress=False,
            random_seed=123,
            state_representation="physical_frame",
            return_native_state=True,
            native_cycle_observer=observer,
            meas_slab_only=True,
            checkpoint_state=copy.deepcopy(captured["state"]),
        )
        observer.assert_complete()
        outputs.append(observer.payload()["mutual_information"])
    np.testing.assert_array_equal(outputs[0], outputs[1])


def test_same_numa_cpu_selection_and_tmux_command_shape():
    cpus = physical_cpus()
    utilization = {cpu: 0.0 for cpu in cpus}
    selected, metadata = choose_cpus(utilization, required=2, maximum_utilization=35.0)
    assert len(selected) == 2
    assert metadata["selected_mean_utilization_percent"] == 0.0
    command = build_tmux_command("session", "0,1", "python runner.py")
    assert command[:6] == ["tmux", "new-session", "-d", "-s", "session", "taskset"]
    assert command[-2:] == ["-lc", "python runner.py"]
