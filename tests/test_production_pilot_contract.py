from __future__ import annotations

import ast
import hashlib
import io
import json
import math
import sys
import tarfile
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_ROOT = ROOT / "00_WORKSPACE/CURRENT/final_production_ready_figure_scripts"
PRODUCTION = CAMPAIGN_ROOT / "prior_designs"
SHARED = CAMPAIGN_ROOT / "_shared_src"
ACTIVE_BUNDLES = (
    "00_validation",
    "01_bulk_width_gate",
    "02_pure_wall_master",
    "03_chirality_replay",
    "04_maxmix_master",
    "05_scans_and_controls",
    "06_b1_controller_frame",
)
sys.path.insert(0, str(SHARED))

from campaign_cases import expand_cases  # noqa: E402
from b1_controller_frame import b1_cases  # noqa: E402
from gate_analysis import _complete_shard_superset  # noqa: E402
from run_b1_shard import _load_config as load_b1_config  # noqa: E402
from run_h3_shard import find_parent_archive, production_cases  # noqa: E402
from production_runtime import (  # noqa: E402
    LEGACY_AUDIT_SHA256,
    PRIOR_V2_AUDIT_SHA256,
    PRIOR_V2_PRODUCTION_OUTPUT_COLLECTION,
    PRODUCTION_SAMPLES,
    SHARD_SIZE,
    V1_AUDIT_SHA256,
    find_compatible_legacy_archive,
    sha256_file,
    verify_archive_receipt,
)


def _write_parent_archive(
    root: Path, *, case_id: str, shard_index: int = 0
) -> Path:
    output = root / "classA_final_production_outputs" / "02_pure_wall_master"
    output.mkdir(parents=True, exist_ok=True)
    archive_path = output / f"{case_id}_shard-{shard_index}.tar.gz"
    payload = json.dumps(
        {
            "shard_index": shard_index,
            "run_config": {
                "case": {
                    "case_id": case_id,
                    "model": {"Nx": 20, "Ny": 24},
                }
            },
        }
    ).encode("utf-8")
    info = tarfile.TarInfo("manifest.json")
    info.size = len(payload)
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.addfile(info, io.BytesIO(payload))
    digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    receipt_path = archive_path.with_suffix(archive_path.suffix + ".receipt.json")
    receipt_path.write_text(
        json.dumps({"archive": archive_path.name, "archive_sha256": digest})
    )
    return archive_path


def _write_legacy_archive(
    root: Path,
    *,
    case: dict,
    bundle: str = "legacy_test_bundle",
    shard_index: int = 0,
    root_seed: int = 9142,
    engine_hash: str = "e" * 64,
    collection: str = "classA_final_production_outputs",
    declared_samples: int = 25,
    audit_sha256: str = LEGACY_AUDIT_SHA256,
    mutate_manifest=None,
) -> Path:
    output = root / collection / bundle
    output.mkdir(parents=True, exist_ok=True)
    archive_path = output / f"legacy-{shard_index}.tar.gz"
    legacy_case = json.loads(json.dumps(case))
    legacy_case["run"]["samples"] = declared_samples
    seed_raw = f"{root_seed}:{case['case_id']}:{shard_index}".encode()
    shard_seed = int.from_bytes(hashlib.sha256(seed_raw).digest()[:8], "little") % (
        2**63 - 1
    )
    manifest = {
        "bundle": bundle,
        "status": "complete_local",
        "case_id": case["case_id"],
        "shard_index": shard_index,
        "global_sample_indices": list(range(shard_index * 5, shard_index * 5 + 5)),
        "root_seed": root_seed,
        "shard_generator_seed": shard_seed,
        "audit_sha256": audit_sha256,
        "run_config": {
            "case": legacy_case,
            "canonical_engine_sha256": engine_hash,
            "sampling_revision": (
                "production_10sample_v1"
                if "production_10sample_v1" in collection
                else "production_25sample_v1"
            ),
        },
    }
    if mutate_manifest is not None:
        mutate_manifest(manifest)
    payload = json.dumps(manifest).encode()
    info = tarfile.TarInfo("manifest.json")
    info.size = len(payload)
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.addfile(info, io.BytesIO(payload))
    digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    archive_path.with_suffix(archive_path.suffix + ".receipt.json").write_text(
        json.dumps({"archive": archive_path.name, "archive_sha256": digest})
    )
    return archive_path


def test_no_runnable_tmi_campaign_and_pdf_contract_is_retired() -> None:
    runnable_ids: list[str] = []
    for bundle in ACTIVE_BUNDLES:
        config_path = PRODUCTION / bundle / "production_config.json"
        config = json.loads(config_path.read_text())
        if config["bundle"] == "06_b1_controller_frame":
            runnable_ids.extend(case["case_id"] for case in b1_cases(config))
            continue
        kwargs = {}
        if config["bundle"] not in ("00_validation", "01_bulk_width_gate"):
            kwargs["accepted_width"] = 20
        runnable_ids.extend(case["case_id"] for case in expand_cases(config, **kwargs))
    assert not [case_id for case_id in runnable_ids if "tmi" in case_id.lower()]

    pilot = json.loads((CAMPAIGN_ROOT / "pilot_plan.json").read_text())
    assert "tmi" not in json.dumps(pilot).lower()
    tex = (
        ROOT
        / "00_WORKSPACE/CURRENT/experiment_review/numerical_campaign_legacy_working.tex"
    ).read_text()
    assert "Retired tripartite-information scope" in tex
    assert "No new\nTMI trajectory family" in tex


def test_pilot_is_week_bounded_and_flux_uses_one_record_per_protocol() -> None:
    pilot = json.loads((CAMPAIGN_ROOT / "pilot_plan.json").read_text())
    assert pilot["available_a100_hours_reference"] == 1666.0 / 5.3
    assert pilot["planning_caps"]["science_plus_h3_total_gpu_hours"] <= 124.0
    assert pilot["planning_caps"]["science_plus_h3_credits"] <= 1666.0
    flux = pilot["frozen_record_flux_insertion"]
    assert flux["pilot_grid_points"] == 33
    assert flux["replayed_record_indices"] == [0]
    assert flux["replayed_records_per_case"] == 1
    assert flux["science_replayed_records_per_case"] == 1
    assert flux["science_parent_shards"] == [0]
    assert flux["ensemble_frequency_claim"] is False
    assert flux["promotion_order"][0] == "one_record_per_protocol_33_points"
    assert flux["expanded_h3_matrix_default"] is False
    calibration_ids = pilot["calibration"]["03_chirality_replay"]
    assert calibration_ids and all(
        "grid-33_records-1" in case_id for case_id in calibration_ids
    )
    science_ids = pilot["science"]["03_chirality_replay"]
    assert len(science_ids) == 2
    assert all("grid-33_records-1" in case_id for case_id in science_ids)
    science_shards = pilot["profile_case_shard_indices"]["science"]
    assert set(science_shards["02_pure_wall_master"])
    assert all(
        indices == [0, 1]
        for indices in science_shards["02_pure_wall_master"].values()
    )
    assert set(science_shards["03_chirality_replay"])
    assert all(
        indices == [0]
        for indices in science_shards["03_chirality_replay"].values()
    )

    h3_config = json.loads(
        (PRODUCTION / "03_chirality_replay" / "production_config.json").read_text()
    )
    expanded_h3 = expand_cases(h3_config, accepted_width=20)
    selected_h3, selected_parent_shards = production_cases(h3_config, expanded_h3)
    assert selected_parent_shards == [0]
    assert len(selected_h3) == 2
    assert all(case["Ny"] == 40 and case["grid_points"] == 33 for case in selected_h3)
    assert all(case["record_indices"] == [0] for case in selected_h3)


def test_production_sampling_contract_is_ten_in_two_five_trajectory_shards() -> None:
    assert PRODUCTION_SAMPLES == 10
    assert SHARD_SIZE == 5
    for bundle in ACTIVE_BUNDLES:
        config_path = PRODUCTION / bundle / "production_config.json"
        config = json.loads(config_path.read_text())
        assert config["locked_contract"]["samples"] == 10
        assert config["locked_contract"]["sample_shard_size"] == 5
        assert config["sampling_revision"].startswith("production_10sample_v4_occupied_frame")
        if config["bundle"] in ("00_validation", "03_chirality_replay"):
            continue
        cases = (
            b1_cases(config)
            if config["bundle"] == "06_b1_controller_frame"
            else expand_cases(
                config,
                **(
                    {}
                    if config["bundle"] == "01_bulk_width_gate"
                    else {"accepted_width": 20}
                ),
            )
        )
        ordinary = [case for case in cases if int(case["run"]["samples"]) != 1]
        assert ordinary
        assert all(case["run"]["samples"] == 10 for case in ordinary)
        assert all(math.ceil(case["run"]["samples"] / SHARD_SIZE) == 2 for case in ordinary)


def test_fixed_geometry_ten_sample_campaign_has_expected_bundle_shards() -> None:
    configs = {
        path.parent.name: json.loads(path.read_text())
        for path in (PRODUCTION / bundle / "production_config.json" for bundle in ACTIVE_BUNDLES)
    }

    def shard_count(cases: list[dict]) -> int:
        return sum(math.ceil(int(case["run"]["samples"]) / SHARD_SIZE) for case in cases)

    bundle_01 = shard_count(expand_cases(configs["01_bulk_width_gate"]))
    bundle_02 = shard_count(
        expand_cases(configs["02_pure_wall_master"], accepted_width=20)
    )
    bundle_04 = shard_count(
        expand_cases(configs["04_maxmix_master"], accepted_width=20)
    )
    bundle_05 = shard_count(
        expand_cases(
            configs["05_scans_and_controls"],
            accepted_width=20,
            m3_wall_sigma=[0.1, 0.2, 0.3, 0.4, 0.5],
        )
    )
    bundle_06 = shard_count(b1_cases(configs["06_b1_controller_frame"]))
    bundle_03 = 2  # two protocols x one representative record from parent shard 0
    assert (bundle_01, bundle_02, bundle_03, bundle_04, bundle_05, bundle_06) == (
        40,
        60,
        2,
        20,
        374,
        4,
    )
    assert sum((bundle_01, bundle_02, bundle_03, bundle_04, bundle_05, bundle_06)) == 500


def test_s2_adds_exact_pure_support_terminated_mirror() -> None:
    config = json.loads(
        (PRODUCTION / "05_scans_and_controls" / "production_config.json").read_text()
    )
    cases = expand_cases(config, accepted_width=20)
    new = [case for case in cases if "support_terminated_alpha_wall" in case["case_id"]]
    assert len(new) == 24
    assert len({case["case_id"] for case in new}) == 24
    assert all(case["model"]["init_mode"] == "default" for case in new)
    assert all(case["model"]["DW"] is True for case in new)
    assert all(case["model"]["dw_truncation"] is True for case in new)
    assert all(case["model"]["meas_slab_only"] is True for case in new)
    assert all(case["model"]["nshell"] == 1 for case in new)
    assert all(case["model"]["alpha_2"] == 30 for case in new)
    assert all(case["run"]["samples"] == 10 for case in new)
    assert all(case["run"]["cycles"] == 2 * case["model"]["Ny"] for case in new)
    assert all(case["run"]["lyapunov_nvec"] == 16 for case in new)
    assert not any(case["model"]["alpha_1"] == 2 for case in new)
    assert not any("maxmix" in case["case_id"] for case in new)

    old = [
        case for case in cases
        if case["campaign"] == "S2" and "support_terminated_alpha_wall" not in case["case_id"]
    ]
    assert len(old) == 48
    assert all(case["model"]["dw_truncation"] is False for case in old)


def test_strict_legacy_reuse_accepts_only_sample_count_supersets(tmp_path: Path) -> None:
    case = {
        "case_id": "P1_contract_probe",
        "model": {"Nx": 20, "Ny": 24, "nshell": 2},
        "run": {"samples": 25, "cycles": 48},
        "observable_contract": {"version": "probe-v1"},
    }
    bundle = "legacy_test_bundle"
    root_seed = 9142
    engine_hash = "e" * 64

    valid_root = tmp_path / "valid"
    archive = _write_legacy_archive(valid_root, case=case)
    accepted = find_compatible_legacy_archive(
        drive_root=valid_root,
        bundle_name=bundle,
        case=case,
        shard_index=0,
        root_seed=root_seed,
        canonical_engine_sha256=engine_hash,
    )
    assert accepted is not None
    assert accepted["archive"] == str(archive)
    assert accepted["legacy_declared_samples"] == 25
    assert accepted["source_revision"] == "unversioned"

    prior_v2_root = tmp_path / "prior-v2"
    prior_v2_archive = _write_legacy_archive(
        prior_v2_root,
        case=case,
        collection=PRIOR_V2_PRODUCTION_OUTPUT_COLLECTION,
        declared_samples=25,
        audit_sha256=PRIOR_V2_AUDIT_SHA256,
    )
    prior_v2 = find_compatible_legacy_archive(
        drive_root=prior_v2_root,
        bundle_name=bundle,
        case=case,
        shard_index=0,
        root_seed=root_seed,
        canonical_engine_sha256=engine_hash,
    )
    assert prior_v2 is not None
    assert prior_v2["archive"] == str(prior_v2_archive)
    assert prior_v2["source_revision"] == "prior_v2"

    v1_root = tmp_path / "v1-priority"
    v1_archive = _write_legacy_archive(
        v1_root,
        case=case,
        collection="classA_final_production_outputs/production_10sample_v1",
        declared_samples=10,
        audit_sha256=V1_AUDIT_SHA256,
    )
    unversioned_archive = _write_legacy_archive(v1_root, case=case)
    preferred = find_compatible_legacy_archive(
        drive_root=v1_root,
        bundle_name=bundle,
        case=case,
        shard_index=0,
        root_seed=root_seed,
        canonical_engine_sha256=engine_hash,
    )
    assert preferred is not None
    assert preferred["archive"] == str(v1_archive)
    assert preferred["archive"] != str(unversioned_archive)
    assert preferred["source_revision"] == "v1"
    assert preferred["reused_shard_indices"] == [0]

    mutations = {
        "case": lambda manifest: manifest["run_config"]["case"].__setitem__(
            "case_id", "wrong_case"
        ),
        "seed": lambda manifest: manifest.__setitem__("shard_generator_seed", 17),
        "engine": lambda manifest: manifest["run_config"].__setitem__(
            "canonical_engine_sha256", "f" * 64
        ),
        "geometry": lambda manifest: manifest["run_config"]["case"]["model"].__setitem__(
            "Nx", 22
        ),
        "cycles": lambda manifest: manifest["run_config"]["case"]["run"].__setitem__(
            "cycles", 50
        ),
        "observable": lambda manifest: manifest["run_config"]["case"][
            "observable_contract"
        ].__setitem__("version", "wrong"),
        "audit": lambda manifest: manifest.__setitem__("audit_sha256", "0" * 64),
    }
    for name, mutation in mutations.items():
        mismatch_root = tmp_path / name
        _write_legacy_archive(mismatch_root, case=case, mutate_manifest=mutation)
        assert find_compatible_legacy_archive(
            drive_root=mismatch_root,
            bundle_name=bundle,
            case=case,
            shard_index=0,
            root_seed=root_seed,
            canonical_engine_sha256=engine_hash,
        ) is None

    checksum_root = tmp_path / "checksum"
    bad_archive = _write_legacy_archive(checksum_root, case=case)
    receipt_path = bad_archive.with_suffix(bad_archive.suffix + ".receipt.json")
    receipt = json.loads(receipt_path.read_text())
    receipt["archive_sha256"] = "0" * 64
    receipt_path.write_text(json.dumps(receipt))
    assert find_compatible_legacy_archive(
        drive_root=checksum_root,
        bundle_name=bundle,
        case=case,
        shard_index=0,
        root_seed=root_seed,
        canonical_engine_sha256=engine_hash,
    ) is None


@pytest.mark.parametrize("shard_count", [2, 3, 5])
def test_mergers_require_at_least_ten_sample_supersets(shard_count: int) -> None:
    shards = [
        (Path(f"shard-{index}.tar.gz"), {"shard_index": index})
        for index in range(shard_count)
    ]
    complete, indices = _complete_shard_superset(shards)
    assert complete
    assert indices == list(range(shard_count))
    assert 5 * len(indices) >= 10


def test_mergers_reject_one_shard_and_nonconsecutive_supersets() -> None:
    assert _complete_shard_superset([(Path("only.tar.gz"), {"shard_index": 0})])[0] is False
    four = [(Path(f"{index}.tar.gz"), {"shard_index": index}) for index in range(4)]
    assert _complete_shard_superset(four)[0] is True
    broken = [
        (Path("zero.tar.gz"), {"shard_index": 0}),
        (Path("two.tar.gz"), {"shard_index": 2}),
    ]
    assert _complete_shard_superset(broken)[0] is False


def test_current_snapshot_isolated_for_exact_p1_completion() -> None:
    drive_root = ROOT / "00_WORKSPACE/LARGE_RESULTS"
    legacy_root = drive_root / "classA_final_production_outputs/01_bulk_width_gate"
    assert len(list(legacy_root.glob("*.tar.gz"))) == 240
    config = json.loads(
        (PRODUCTION / "01_bulk_width_gate" / "production_config.json").read_text()
    )
    engine_hash = sha256_file(
        PRODUCTION / "01_bulk_width_gate/src/classA_U1FGTN_gpu.py"
    )
    reused = 0
    for case in expand_cases(config):
        for shard_index in range(5):
            reused += find_compatible_legacy_archive(
                drive_root=drive_root,
                bundle_name=config["bundle"],
                case=case,
                shard_index=shard_index,
                root_seed=config["root_seed"],
                canonical_engine_sha256=engine_hash,
            ) is not None
    assert reused == 0
    frozen = PRODUCTION / "01_p1_existing_completion" / "resume_existing_p1.py"
    assert frozen.is_file()
    assert "EXPECTED_SHARDS = 240" in frozen.read_text()


def test_generated_notebooks_default_to_production_report_only() -> None:
    # Standalone 25-sample revisions may extend the root queue without rewriting
    # the immutable v4 bundle-local pilot-plan copies.
    canonical_plan = (PRODUCTION / "00_validation" / "pilot_plan.json").read_bytes()
    for bundle_name in ACTIVE_BUNDLES:
        bundle = PRODUCTION / bundle_name
        assert (bundle / "pilot_plan.json").read_bytes() == canonical_plan
        notebook = json.loads((bundle / "run_production_bundle.ipynb").read_text())
        code = "\n".join(
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )
        assert "RUN_PROFILE = 'production'" in code
        assert "RESUME_REPORT_ONLY = True" in code
        assert "colab_bundle_runner.py" in code
        assert "runner_build_id" in code
        assert "_bundle_sessions" in code
        assert "HEARTBEAT_SECONDS = 60" in code
        assert "runtime.unassign()" in "".join(notebook["cells"][-1]["source"])


def test_independent_bundle_runner_covers_every_runnable_bundle() -> None:
    runner = (CAMPAIGN_ROOT / "colab_bundle_runner.py").read_text()
    assert "drive_storage_guard.py" in runner
    assert "_bundle_sessions" in runner
    assert "--bundle" in runner
    assert "--case-prefix" in runner
    assert "--case-id" in runner
    assert "--preflight-only" in runner
    assert "--resume-report-only" in runner
    assert "sys.executable" in runner and '"-u"' in runner
    assert "production_fixed_geometry_baseline_complete" in runner
    assert "production_m3_bulk_gate_complete_rerun_bundle_for_wall_bracket" in runner
    assert '"fixed_geometry_baseline_manifest.json"' in runner
    assert not (CAMPAIGN_ROOT / "COLAB_LANES").exists()
    assert not (CAMPAIGN_ROOT / "colab_lane_runner.py").exists()
    for bundle in ACTIVE_BUNDLES:
        path = PRODUCTION / bundle / "run_production_bundle.ipynb"
        notebook = json.loads(path.read_text())
        code_cells = [
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        ]
        for index, source in enumerate(code_cells):
            ast.parse(source, filename=f"{path}:code-cell-{index}")
        source = "\n".join(code_cells)
        assert "RUN_PROFILE = 'production'" in source
        assert "subprocess.run(command, check=False)" in source
        assert "saved bundle failure" in source
        assert "HEARTBEAT_SECONDS = 60" in source
        assert "RESUME_REPORT_ONLY = True" in source
        assert "runtime.unassign()" in source

    amendments = (CAMPAIGN_ROOT / "AUDIT_AMENDMENTS.md").read_text()
    assert "A20 — B1 execution is required" in amendments
    assert "scheduling-scope amendment" in amendments
    b1_config = json.loads(
        (PRODUCTION / "06_b1_controller_frame" / "production_config.json").read_text()
    )
    assert b1_config["priority"] == "required_final"


def test_local_cpu_flux_pilot_uses_canonical_cpu_engine_and_one_sample() -> None:
    pilot_script = (
        PRODUCTION
        / "03_chirality_replay/cpu_one_record_pilot/run_cpu_flux_pilot.py"
    ).read_text()
    assert "from fgtn.classA_U1FGTN import classA_U1FGTN" in pilot_script
    assert "classA_U1FGTN_gpu" not in pilot_script
    assert '"--nx", type=int, default=20' in pilot_script
    assert '"--ny", type=int, default=40' in pilot_script
    assert '"samples": 1' in pilot_script
    assert 'parser.add_argument("--grid-points", type=int, default=33)' in pilot_script
    assert '"cycles": int(cycles)' in pilot_script
    assert "trajectory_replay=record" in pilot_script
    assert "successive_covariance_frobenius_per_dimension" in pilot_script
    assert 'SCHEMA = "h3_cpu_one_record_flux_pilot_v2_production_geometry"' in pilot_script
    assert "width = max(1, int(nx) // 4)" in pilot_script
    assert 'config["dw_interval"] = tuple(int(value) for value in wall_centers)' in pilot_script
    assert 'for center in wall_centers:' in pilot_script
    assert '"wall_locations": list(wall_centers)' in pilot_script


def test_gpu_batching_is_enforced_and_derived_stages_are_batched() -> None:
    engine = (ROOT / "src/fgtn/classA_U1FGTN_gpu.py").read_text()
    core = (SHARED / "run_core_shard.py").read_text()
    h3 = (SHARED / "run_h3_shard.py").read_text()
    b1 = (SHARED / "run_b1_shard.py").read_text()
    derived = (SHARED / "fused_chirality_observables.py").read_text()
    contract = "sitewise_rank1_full_trajectory_batch_v1"
    assert "_apply_feedback_sitewise_batched" in engine
    assert contract in engine
    assert contract in core and "ordinary shard did not use" in core
    frame_contract = "padded_variable_rank_frame_v2"
    assert frame_contract in h3 and "H3 replay did not use" in h3
    assert frame_contract in b1 and "B1 did not use" in b1
    assert "batched_eigh_and_packet_propagation_over_complete_shard_v1" in derived
    assert "source_chunk_times_plus_minus_times_complete_shard_v1" in derived


def test_validation_materializes_device_record_buffers_before_numpy() -> None:
    validation = (SHARED / "run_validation_suite.py").read_text()
    replay_materialization = validation.index("replay_writer.validate()")
    replay_numpy = validation.index("replay_writer.conditional_log_probability")
    branch_materialization = validation.index("branch_writer.validate()")
    branch_numpy = validation.index("branch_writer.channel_count")
    assert replay_materialization < replay_numpy
    assert branch_materialization < branch_numpy
    assert 'init_mode="pure"' not in validation
    assert validation.count('init_mode="default"') >= 2
    assert 'nshell=2,\n        device=device,\n        dtype="complex128",\n        backend="local"' in validation
    assert '"suite_version": 2' in validation

    notebook = json.loads(
        (PRODUCTION / "00_validation" / "run_production_bundle.ipynb").read_text()
    )
    notebook_code = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    assert "[SITEWISE GPU SPEEDUP]" in notebook_code
    assert "speedup_over_grouped_reference" in notebook_code


def test_h3_parent_selection_is_exact_and_checksum_verified(tmp_path: Path) -> None:
    wall = _write_parent_archive(
        tmp_path, case_id="MASTER_N20x24_explicit_interface"
    )
    _write_parent_archive(
        tmp_path, case_id="MASTER_N20x24_explicit_interface_matched_trivial"
    )
    case = {
        "case_id": "H3_N20x24_explicit_interface",
        "Nx": 20,
        "Ny": 24,
        "parent_protocol": "explicit_interface",
    }

    selected, manifest = find_parent_archive(
        drive_root=tmp_path, case=case, shard_index=0
    )
    assert selected == wall
    assert verify_archive_receipt(wall)["archive"] == wall.name
    assert manifest["run_config"]["case"]["case_id"] == (
        "MASTER_N20x24_explicit_interface"
    )

    receipt_path = wall.with_suffix(wall.suffix + ".receipt.json")
    receipt = json.loads(receipt_path.read_text())
    receipt["archive_sha256"] = "0" * 64
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(RuntimeError, match="integrity failures"):
        find_parent_archive(drive_root=tmp_path, case=case, shard_index=0)


def test_h3_removes_resumable_checkpoint_only_after_archive_succeeds() -> None:
    h3 = (SHARED / "run_h3_shard.py").read_text()
    archive = h3.index("receipt = archive_run_to_drive(paths)")
    cleanup = h3.index("checkpoint_drive.unlink(missing_ok=True)", archive)
    assert archive < cleanup
    assert '"production_10sample_v4_occupied_frame_cycle_resolved_h3_representative_record"' in (
        PRODUCTION / "03_chirality_replay" / "production_config.json"
    ).read_text()


def test_b1_loader_enforces_the_shared_audit_hash(tmp_path: Path) -> None:
    source = PRODUCTION / "06_b1_controller_frame" / "production_config.json"
    config = json.loads(source.read_text())
    bundle = tmp_path / "06_b1_controller_frame"
    bundle.mkdir()
    path = bundle / "production_config.json"
    path.write_text(json.dumps(config))
    assert load_b1_config(bundle)["audit_sha256"] == config["audit_sha256"]

    config["audit_sha256"] = "0" * 64
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="different audit PDF hash"):
        load_b1_config(bundle)

    analysis = (SHARED / "b1_analysis.py").read_text()
    assert "config = load_config(bundle_root)" in analysis

    gate = (SHARED / "gate_analysis.py").read_text()
    record_spectrum = (SHARED / "record_spectrum_analysis.py").read_text()
    assert "verify_archive_receipt(path)" in gate
    assert "verify_archive_receipt(path)" in record_spectrum
