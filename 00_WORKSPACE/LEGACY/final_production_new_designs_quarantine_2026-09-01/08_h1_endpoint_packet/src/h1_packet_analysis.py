"""Merge and analyze the standalone H1-v3 endpoint-packet archives."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import tarfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from h1_io import sha256_file, write_json_atomic
from h1_packet_runner import AUDIT, BUNDLE, REVISION, load_config
from h1_v3_migration import load_current_verified_ledger
from drive_remote_commit import DriveRemoteCommitter


V3_REVISION = "production_25sample_h1_endpoint_packet_v3"
V3_AUDIT = "69d43b5ef6dbe6766ee1063b783b74fbf60ebe41a0628a598678207b96bfe500"


def _member_bytes(archive: tarfile.TarFile, suffix: str) -> bytes:
    matches = [
        member for member in archive.getmembers()
        if member.name.lstrip("./").endswith(suffix)
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one {suffix!r} member, found {len(matches)}")
    handle = archive.extractfile(matches[0])
    if handle is None:
        raise RuntimeError(f"cannot read {matches[0].name}")
    return handle.read()


def _npz_bytes(raw: bytes) -> dict[str, np.ndarray]:
    with np.load(io.BytesIO(raw), allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def _verify_receipt(path: Path) -> dict[str, Any]:
    receipt_path = path.with_suffix(path.suffix + ".receipt.json")
    if not receipt_path.is_file():
        raise RuntimeError(f"missing H1-v3 archive receipt: {path}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("archive") != path.name or receipt.get("archive_sha256") != sha256_file(path):
        raise RuntimeError(f"H1-v3 archive receipt failed: {path}")
    return receipt


def load_archive(path: Path | str) -> dict[str, Any]:
    path = Path(path)
    receipt = _verify_receipt(path)
    with tarfile.open(path, "r:gz") as archive:
        manifest = json.loads(_member_bytes(archive, "manifest.json").decode("utf-8"))
        raw = {
            "common": _member_bytes(archive, "h1_endpoint_packet/common.npz"),
            "packet": _member_bytes(archive, "h1_endpoint_packet/packet_drift.npz"),
            "profiles": _member_bytes(archive, "h1_endpoint_packet/primary_profiles.npz"),
        }
    if manifest.get("bundle") != BUNDLE:
        raise RuntimeError(f"wrong bundle in {path}")
    revision_identity = (
        manifest.get("sampling_revision"), manifest.get("audit_sha256")
    )
    if revision_identity not in {(REVISION, AUDIT), (V3_REVISION, V3_AUDIT)}:
        raise RuntimeError(f"wrong H1 endpoint-packet revision or audit in {path}")
    declared = {
        row["path"]: row["sha256"]
        for row in manifest["products"]["h1_endpoint_packet"]["files"]
    }
    names = {"common": "common.npz", "packet": "packet_drift.npz", "profiles": "primary_profiles.npz"}
    for key, name in names.items():
        if declared.get(name) != hashlib.sha256(raw[key]).hexdigest():
            raise RuntimeError(f"internal H1-v3 product checksum failed for {path}/{name}")
    common, packet, profiles = (
        _npz_bytes(raw["common"]), _npz_bytes(raw["packet"]), _npz_bytes(raw["profiles"])
    )
    forbidden_fragments = (
        "retarded", "susceptibility", "fourier", "response_ridge",
        "full_covariance", "eigenvectors",
    )
    all_keys = {*common, *packet, *profiles}
    found = sorted(
        key for key in all_keys
        if any(fragment in key.lower() for fragment in forbidden_fragments)
    )
    if found:
        raise RuntimeError(f"retired response products found in {path}: {found}")
    if str(np.asarray(common["schema"]).item()) != "h1_translated_endpoint_packet_v3":
        raise RuntimeError(f"unexpected H1-v3 product schema in {path}")
    return {
        "path": path,
        "receipt": receipt,
        "manifest": manifest,
        "case": manifest["run_config"]["case"],
        "common": common,
        "packet": packet,
        "profiles": profiles,
    }


def merge_archives(
    archive_root: Path | str, *, compatible_v3_archives: list[Path] | None = None
) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    paths = sorted(Path(archive_root).glob("*.tar.gz")) + list(
        compatible_v3_archives or []
    )
    for path in paths:
        try:
            item = load_archive(path)
        except RuntimeError as exc:
            if "wrong bundle" in str(exc) or "wrong H1 endpoint-packet revision" in str(exc):
                continue
            raise
        groups[item["case"]["case_id"]].append(item)
    if len(groups) != 4:
        raise RuntimeError(f"expected four complete H1-v3 cases, found {len(groups)}")

    common_dynamic = (
        "spectral_clip_counts",
        "restricted_correlation_hermiticity_error",
        "conditional_eigenvector_gram_residual",
        "conditional_eigenvector_gram_residual_computed",
        "observer_seconds",
    )
    packet_dynamic = (
        "paired_endpoint_wall_drift",
        "wall_delta_at_fixed_time",
        "oriented_handed_delta",
        "wall_velocity",
        "wall_velocity_r2",
        "oriented_handed_velocity",
        "minimum_wall_retention",
        "raw_packet_total_norm",
        "maximum_raw_packet_norm_error",
        "maximum_raw_packet_norm_drift",
        "maximum_post_normalization_norm_error",
    )
    profile_dynamic = (
        "primary_endpoint_center",
        "primary_endpoint_retention",
        "primary_cut_mean_conditional_profile",
    )
    metadata_keys = (
        "checkpoints", "cut_origins", "modular_times", "wall_x",
        "wall_orientation_signs", "source_width_columns",
        "retention_width_columns", "spectral_clip_eps", "fit_windows",
        "primary_epsilon_index", "primary_source_width_index",
        "primary_retention_width_index", "fixed_time_index", "source_index",
        "actual_dtype", "actual_probability_dtype",
        "raw_norm_warning_tolerance", "raw_norm_hard_failure_tolerance",
        "post_normalization_norm_tolerance", "gram_diagnostic_trigger",
    )
    merged: dict[str, dict[str, Any]] = {}
    for case_id, rows in groups.items():
        by_shard = {int(row["manifest"]["shard_index"]): row for row in rows}
        if set(by_shard) != set(range(5)) or len(rows) != 5:
            raise RuntimeError(f"{case_id}: expected exactly shards 0..4")
        ordered = [by_shard[index] for index in range(5)]
        sample_ids = np.concatenate([
            row["common"]["global_sample_ids"] for row in ordered
        ])
        if sample_ids.tolist() != list(range(25)):
            raise RuntimeError(f"{case_id}: incomplete or unordered sample IDs")
        reference = ordered[0]["common"]
        for row in ordered[1:]:
            for key in metadata_keys:
                if not np.array_equal(row["common"][key], reference[key]):
                    raise RuntimeError(f"{case_id}: shard metadata differs for {key}")
        arrays = {
            key: np.concatenate([row["common"][key] for row in ordered], axis=0)
            for key in common_dynamic
        }
        arrays.update({
            key: np.concatenate([row["packet"][key] for row in ordered], axis=0)
            for key in packet_dynamic
        })
        arrays.update({
            key: np.concatenate([row["profiles"][key] for row in ordered], axis=0)
            for key in profile_dynamic
        })
        merged[case_id] = {
            "case": ordered[0]["case"],
            "sample_ids": sample_ids,
            **{key: np.asarray(reference[key]) for key in metadata_keys},
            **arrays,
            "archives": [str(row["path"]) for row in ordered],
            "trajectory_provenance": [
                row["manifest"].get("trajectory_provenance", {}) for row in ordered
            ],
        }
    wall_reference = next(iter(merged.values()))["wall_x"]
    for case_id, item in merged.items():
        if not np.array_equal(item["wall_x"], wall_reference):
            raise RuntimeError(f"{case_id}: actual model wall coordinates differ")
        if item["wall_orientation_signs"].tolist() != [-1, 1]:
            raise RuntimeError(f"{case_id}: orientation differs from the exact benchmark")
    return merged


def bootstrap_mean_ci(
    values: np.ndarray, *, repetitions: int, seed: int,
) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all() or len(values) < 2:
        raise ValueError("bootstrap values must be a finite trajectory vector")
    rng = np.random.default_rng(int(seed))
    draws = values[
        rng.integers(0, len(values), size=(int(repetitions), len(values)))
    ].mean(axis=1)
    return float(values.mean()), *np.quantile(draws, [0.025, 0.975]).tolist()


def bootstrap_difference_ci(
    left: np.ndarray, right: np.ndarray, *, repetitions: int, seed: int,
) -> tuple[float, float, float]:
    left, right = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    rng = np.random.default_rng(int(seed))
    draws = (
        left[rng.integers(0, len(left), size=(repetitions, len(left)))].mean(axis=1)
        - right[rng.integers(0, len(right), size=(repetitions, len(right)))].mean(axis=1)
    )
    return float(left.mean() - right.mean()), *np.quantile(draws, [0.025, 0.975]).tolist()


def bootstrap_curve_ci(
    values: np.ndarray, *, repetitions: int, seed: int, batch: int = 250,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        raise ValueError("curve bootstrap expects (trajectory,time)")
    rng = np.random.default_rng(int(seed))
    chunks = []
    remaining = int(repetitions)
    while remaining:
        count = min(int(batch), remaining)
        indices = rng.integers(0, values.shape[0], size=(count, values.shape[0]))
        chunks.append(values[indices].mean(axis=1))
        remaining -= count
    draws = np.concatenate(chunks, axis=0)
    return values.mean(axis=0), *np.quantile(draws, [0.025, 0.975], axis=0)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty table {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _case_ids() -> list[str]:
    return [
        f"H1_N20x40_{protocol}_a1-{alpha}"
        for alpha in (1, 3) for protocol in ("hard", "soft")
    ]


def _case_label(item: dict[str, Any]) -> str:
    return f"{item['case']['protocol']}, $\\alpha_1={item['case']['model']['alpha_1']:g}$"


def _primary_indices(item: dict[str, Any]) -> tuple[int, int, int, int]:
    return (
        int(item["primary_epsilon_index"]),
        int(item["primary_source_width_index"]),
        int(item["primary_retention_width_index"]),
        int(item["fixed_time_index"]),
    )


def analyze(
    archive_root: Path | str, output_root: Path | str,
    *, bundle_root: Path | str | None = None,
    drive_root: Path | str | None = None,
) -> dict[str, Any]:
    if bundle_root is None:
        bundle_root = Path(__file__).resolve().parents[1]
    compatible_v3_archives: list[Path] = []
    if drive_root is not None:
        drive_root = Path(drive_root).resolve()
        ledger = load_current_verified_ledger(
            drive_root=drive_root, bundle_root=Path(bundle_root)
        )
        committer = DriveRemoteCommitter(drive_root=drive_root)
        source_root = (
            drive_root
            / "classA_final_production_outputs"
            / V3_REVISION
            / BUNDLE
        )
        cache_root = Path("/content/classA_remote_cache/h1_v3_analysis")
        for row in ledger["accepted_archives"]:
            if not bool(row.get("reuse_in_v4")):
                continue
            visible = source_root / row["archive"]
            if visible.is_file() and sha256_file(visible) == row["archive_sha256"]:
                compatible_v3_archives.append(visible)
                continue
            cache = cache_root / row["archive"]
            if not cache.is_file() or sha256_file(cache) != row["archive_sha256"]:
                committer.download_to(
                    row["archive_remote_commit"]["remote_file_id"], cache
                )
            receipt = cache.with_suffix(cache.suffix + ".receipt.json")
            if not receipt.is_file():
                committer.download_to(
                    row["receipt_remote_commit"]["remote_file_id"], receipt
                )
            compatible_v3_archives.append(cache)
    merged = (
        merge_archives(
            archive_root, compatible_v3_archives=compatible_v3_archives
        )
        if compatible_v3_archives
        else merge_archives(archive_root)
    )
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    config = load_config(bundle_root)
    analysis_config = config["analysis"]
    repetitions = int(analysis_config["bootstrap_repetitions"])
    seed = int(analysis_config["bootstrap_seed"])
    primary_checkpoint = int(analysis_config["primary_checkpoint"])
    equivalence_margin = float(analysis_config["trivial_equivalence_margin"])
    ordered_ids = _case_ids()
    if set(merged) != set(ordered_ids):
        raise RuntimeError(f"H1-v3 case IDs differ from the locked matrix: {sorted(merged)}")

    trajectory_primary: dict[str, np.ndarray] = {}
    trajectory_wall: dict[str, np.ndarray] = {}
    trajectory_velocity: dict[str, np.ndarray] = {}
    summary_rows: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []
    sensitivity_rows: list[dict[str, Any]] = []
    for case_index, case_id in enumerate(ordered_ids):
        item = merged[case_id]
        pe, ps, pr, _ = _primary_indices(item)
        checkpoints = item["checkpoints"].astype(int)
        final_index = int(np.flatnonzero(checkpoints == primary_checkpoint)[0])
        handed_by_cut = item["oriented_handed_delta"][:, :, :, pe, ps, pr]
        handed_by_trajectory_checkpoint = handed_by_cut.mean(axis=2)
        final = handed_by_trajectory_checkpoint[:, final_index]
        trajectory_primary[case_id] = final
        wall_by_cut = item["wall_delta_at_fixed_time"][:, :, :, pe, ps, pr]
        trajectory_wall[case_id] = wall_by_cut.mean(axis=2)[:, final_index]
        velocity_by_cut = item["oriented_handed_velocity"][:, :, :, pe, ps, pr, 0]
        trajectory_velocity[case_id] = velocity_by_cut.mean(axis=2)[:, final_index]
        primary_wall_r2 = item["wall_velocity_r2"][
            :, final_index, :, pe, ps, pr, 0, :
        ]
        wall_median_r2 = np.median(primary_wall_r2, axis=(0, 1))
        center, low, high = bootstrap_mean_ci(
            final, repetitions=repetitions, seed=seed + case_index
        )
        v_center, v_low, v_high = bootstrap_mean_ci(
            trajectory_velocity[case_id],
            repetitions=repetitions,
            seed=seed + 20 + case_index,
        )
        wall_mean = trajectory_wall[case_id].mean(axis=0)
        summary_rows.append({
            "case_id": case_id,
            "protocol": item["case"]["protocol"],
            "alpha_1": item["case"]["model"]["alpha_1"],
            "trajectories": len(final),
            "primary_checkpoint": primary_checkpoint,
            "H_delta_mean": center,
            "H_delta_ci95_low": low,
            "H_delta_ci95_high": high,
            "raw_wall_0_delta_mean": float(wall_mean[0]),
            "raw_wall_1_delta_mean": float(wall_mean[1]),
            "H_velocity_mean": v_center,
            "H_velocity_ci95_low": v_low,
            "H_velocity_ci95_high": v_high,
            "wall_0_velocity_median_r2": float(wall_median_r2[0]),
            "wall_1_velocity_median_r2": float(wall_median_r2[1]),
            "cut_reduction": "mean_40_cuts_within_trajectory",
        })
        for checkpoint_index, checkpoint in enumerate(checkpoints):
            cp_mean, cp_low, cp_high = bootstrap_mean_ci(
                handed_by_trajectory_checkpoint[:, checkpoint_index],
                repetitions=repetitions,
                seed=seed + 100 + 10 * case_index + checkpoint_index,
            )
            checkpoint_rows.append({
                "case_id": case_id,
                "checkpoint": int(checkpoint),
                "H_delta_mean": cp_mean,
                "ci95_low": cp_low,
                "ci95_high": cp_high,
                "is_primary": int(checkpoint) == primary_checkpoint,
            })

        all_delta = item["oriented_handed_delta"][:, final_index].mean(axis=1)
        all_velocity = item["oriented_handed_velocity"][:, final_index].mean(axis=1)
        for eps_index, eps in enumerate(item["spectral_clip_eps"]):
            for source_index, source_width in enumerate(item["source_width_columns"]):
                for retention_index, retention_width in enumerate(item["retention_width_columns"]):
                    d_values = all_delta[:, eps_index, source_index, retention_index]
                    for fit_index, fit_window in enumerate(item["fit_windows"]):
                        v_values = all_velocity[
                            :, eps_index, source_index, retention_index, fit_index
                        ]
                        sensitivity_rows.append({
                            "case_id": case_id,
                            "spectral_clip_eps": float(eps),
                            "source_width_columns": int(source_width),
                            "retention_width_columns": int(retention_width),
                            "fit_window_min": float(fit_window[0]),
                            "fit_window_max": float(fit_window[1]),
                            "is_primary": bool(
                                eps_index == pe and source_index == ps
                                and retention_index == pr and fit_index == 0
                            ),
                            "H_delta_mean": float(d_values.mean()),
                            "H_velocity_mean": float(v_values.mean()),
                            "H_delta_positive": bool(d_values.mean() > 0.0),
                            "H_velocity_positive": bool(v_values.mean() > 0.0),
                        })

    contrast_rows: list[dict[str, Any]] = []
    for contrast_index, protocol in enumerate(("hard", "soft")):
        top = f"H1_N20x40_{protocol}_a1-1"
        trivial = f"H1_N20x40_{protocol}_a1-3"
        center, low, high = bootstrap_difference_ci(
            trajectory_primary[top], trajectory_primary[trivial],
            repetitions=repetitions, seed=seed + 1000 + contrast_index,
        )
        contrast_rows.append({
            "contrast": f"topological_minus_trivial_{protocol}",
            "left_case": top,
            "right_case": trivial,
            "difference": center,
            "ci95_low": low,
            "ci95_high": high,
            "bootstrap": "unpaired_parent_trajectories",
        })

    gates: list[dict[str, Any]] = []
    summary_by_id = {row["case_id"]: row for row in summary_rows}
    contrast_by_protocol = {
        row["contrast"].rsplit("_", 1)[1]: row for row in contrast_rows
    }
    for protocol in ("hard", "soft"):
        top_id = f"H1_N20x40_{protocol}_a1-1"
        trivial_id = f"H1_N20x40_{protocol}_a1-3"
        top_row, trivial_row = summary_by_id[top_id], summary_by_id[trivial_id]
        walls = trajectory_wall[top_id]
        wall_ci = [
            bootstrap_mean_ci(
                walls[:, wall], repetitions=repetitions,
                seed=seed + 2000 + 10 * (protocol == "soft") + wall,
            )
            for wall in range(2)
        ]
        gates.extend([
            {
                "gate": f"{protocol}_topological_H_delta_positive",
                "pass": bool(top_row["H_delta_ci95_low"] > 0.0),
                "detail": f"95% CI [{top_row['H_delta_ci95_low']:.6g},{top_row['H_delta_ci95_high']:.6g}]",
            },
            {
                "gate": f"{protocol}_raw_walls_opposite",
                "pass": bool(wall_ci[0][2] < 0.0 and wall_ci[1][1] > 0.0),
                "detail": (
                    f"wall0 CI [{wall_ci[0][1]:.6g},{wall_ci[0][2]:.6g}], "
                    f"wall1 CI [{wall_ci[1][1]:.6g},{wall_ci[1][2]:.6g}]"
                ),
            },
            {
                "gate": f"{protocol}_topological_minus_trivial_positive",
                "pass": bool(contrast_by_protocol[protocol]["ci95_low"] > 0.0),
                "detail": (
                    f"95% CI [{contrast_by_protocol[protocol]['ci95_low']:.6g},"
                    f"{contrast_by_protocol[protocol]['ci95_high']:.6g}]"
                ),
            },
            {
                "gate": f"{protocol}_trivial_equivalent_to_zero",
                "pass": bool(
                    trivial_row["H_delta_ci95_low"] > -equivalence_margin
                    and trivial_row["H_delta_ci95_high"] < equivalence_margin
                ),
                "detail": (
                    f"95% CI [{trivial_row['H_delta_ci95_low']:.6g},"
                    f"{trivial_row['H_delta_ci95_high']:.6g}] inside "
                    f"[-{equivalence_margin:g},{equivalence_margin:g}]"
                ),
            },
        ])
        top_sensitivity = [
            row for row in sensitivity_rows if row["case_id"] == top_id
        ]
        gates.append({
            "gate": f"{protocol}_sensitivity_sign_stable",
            "pass": bool(all(row["H_delta_positive"] for row in top_sensitivity)),
            "detail": f"{sum(row['H_delta_positive'] for row in top_sensitivity)}/{len(top_sensitivity)} clip/source/retention entries positive (repeated over fit rows)",
        })

    maximum_raw_norm_error = max(
        float(np.max(item["maximum_raw_packet_norm_error"]))
        for item in merged.values()
    )
    maximum_raw_norm_drift = max(
        float(np.max(item["maximum_raw_packet_norm_drift"]))
        for item in merged.values()
    )
    maximum_post_normalization_norm_error = max(
        float(np.max(item["maximum_post_normalization_norm_error"]))
        for item in merged.values()
    )
    warning_tolerance = float(
        config["packet_observer"]["raw_norm_warning_tolerance"]
    )
    hard_failure_tolerance = float(
        config["packet_observer"]["raw_norm_hard_failure_tolerance"]
    )
    post_normalization_tolerance = float(
        config["packet_observer"]["post_normalization_norm_tolerance"]
    )
    warning_case_ids = sorted(
        case_id for case_id, item in merged.items()
        if float(np.max(item["maximum_raw_packet_norm_error"])) > warning_tolerance
    )
    dtype_contract_rows = {
        case_id: {
            "actual_dtype": str(np.asarray(item["actual_dtype"]).item()),
            "actual_probability_dtype": str(
                np.asarray(item["actual_probability_dtype"]).item()
            ),
        }
        for case_id, item in merged.items()
    }
    dtype_contract_pass = all(
        row["actual_dtype"] == "torch.complex128"
        and row["actual_probability_dtype"] == "torch.float64"
        for row in dtype_contract_rows.values()
    )
    raw_norm_error_argmax: dict[str, Any] | None = None
    for case_id, item in merged.items():
        errors = np.abs(item["raw_packet_total_norm"] - 1.0)
        index = tuple(int(value) for value in np.unravel_index(
            int(np.argmax(errors)), errors.shape
        ))
        if (
            raw_norm_error_argmax is not None
            and float(errors[index]) <= float(raw_norm_error_argmax["error"])
        ):
            continue
        (
            sample_index, checkpoint_index, cut_index, epsilon_index,
            source_width_index, wall_index, endpoint_index, time_index,
        ) = index
        raw_norm_error_argmax = {
            "case_id": case_id,
            "array_index": list(index),
            "error": float(errors[index]),
            "global_sample_id": int(item["sample_ids"][sample_index]),
            "checkpoint_cycle": int(item["checkpoints"][checkpoint_index]),
            "cut_origin": int(item["cut_origins"][cut_index]),
            "spectral_clip_epsilon": float(item["spectral_clip_eps"][epsilon_index]),
            "source_width_columns": int(item["source_width_columns"][source_width_index]),
            "wall_x": int(item["wall_x"][wall_index]),
            "endpoint_index": endpoint_index,
            "modular_time": float(item["modular_times"][time_index]),
            "raw_total_norm": float(item["raw_packet_total_norm"][index]),
        }
    conditional_gram_values = np.concatenate([
        item["conditional_eigenvector_gram_residual"][
            item["conditional_eigenvector_gram_residual_computed"].astype(bool)
        ]
        for item in merged.values()
    ])
    maximum_conditional_gram_residual = (
        None
        if conditional_gram_values.size == 0
        else float(np.max(conditional_gram_values))
    )
    minimum_primary_retention = min(
        float(np.min(
            item["minimum_wall_retention"][
                ..., int(item["primary_epsilon_index"]),
                int(item["primary_source_width_index"]),
                int(item["primary_retention_width_index"]), :,
            ]
        ))
        for item in merged.values()
    )
    gates.extend([
        {
            "gate": "primary_retention",
            "pass": bool(
                minimum_primary_retention
                >= float(config["packet_observer"]["minimum_primary_retention"])
            ),
            "detail": f"minimum={minimum_primary_retention:.6g}",
        },
        {
            "gate": "post_normalization_packet_norm_conservation",
            "pass": bool(
                maximum_post_normalization_norm_error
                <= post_normalization_tolerance
            ),
            "detail": (
                f"maximum={maximum_post_normalization_norm_error:.6g}; "
                f"tolerance={post_normalization_tolerance:.6g}"
            ),
        },
        {
            "gate": "raw_packet_norm_hard_failure_ceiling",
            "pass": bool(maximum_raw_norm_error <= hard_failure_tolerance),
            "detail": (
                f"maximum={maximum_raw_norm_error:.6g}; "
                f"hard_ceiling={hard_failure_tolerance:.6g}"
            ),
        },
        {
            "gate": "raw_packet_norm_warning_non_gating",
            "pass": True,
            "detail": (
                f"maximum={maximum_raw_norm_error:.6g}; "
                f"warning_tolerance={warning_tolerance:.6g}; "
                f"warning_cases={warning_case_ids}"
            ),
        },
        {
            "gate": "locked_complex128_probability_float64_dtype",
            "pass": bool(dtype_contract_pass),
            "detail": json.dumps(dtype_contract_rows, sort_keys=True),
        },
    ])

    _write_csv(output_root / "h1_endpoint_summary.csv", summary_rows)
    _write_csv(output_root / "h1_endpoint_checkpoint_summary.csv", checkpoint_rows)
    _write_csv(output_root / "h1_endpoint_contrasts.csv", contrast_rows)
    _write_csv(output_root / "h1_endpoint_sensitivity.csv", sensitivity_rows)
    _write_csv(output_root / "h1_endpoint_gate_ledger.csv", gates)

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["CMU Sans Serif", "DejaVu Sans"],
        "font.size": 8,
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
    })
    styles = {
        ordered_ids[0]: ("#b2182b", "o", "-"),
        ordered_ids[1]: ("#ef8a62", "^", "--"),
        ordered_ids[2]: ("#2166ac", "s", "-"),
        ordered_ids[3]: ("#67a9cf", "D", "--"),
    }
    figure, axis = plt.subplots(figsize=(3.375, 2.7), dpi=300)
    for case_id in ordered_ids:
        rows = [row for row in checkpoint_rows if row["case_id"] == case_id]
        color, marker, linestyle = styles[case_id]
        x = np.asarray([row["checkpoint"] for row in rows])
        y = np.asarray([row["H_delta_mean"] for row in rows])
        low = np.asarray([row["ci95_low"] for row in rows])
        high = np.asarray([row["ci95_high"] for row in rows])
        axis.errorbar(
            x, y, yerr=(y - low, high - y), color=color,
            marker=marker, linestyle=linestyle, linewidth=0.9,
            markersize=3.2, capsize=1.5, label=_case_label(merged[case_id]),
        )
    axis.axhline(0.0, color="0.4", linestyle=":", linewidth=0.7)
    axis.set(xlabel="circuit checkpoint", ylabel=r"endpoint handedness $H_\Delta(\tau=2)$")
    axis.legend(frameon=False, fontsize=6.4, ncol=2)
    figure.tight_layout()
    for suffix in ("pdf", "png"):
        figure.savefig(
            output_root / f"h1_endpoint_handedness_checkpoints.{suffix}",
            dpi=300, bbox_inches="tight",
        )
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(7.05, 2.75), sharey=True, constrained_layout=True)
    for panel, protocol in enumerate(("hard", "soft")):
        case_id = f"H1_N20x40_{protocol}_a1-1"
        item = merged[case_id]
        pe, ps, pr, _ = _primary_indices(item)
        final_index = int(np.flatnonzero(item["checkpoints"] == primary_checkpoint)[0])
        curves = item["paired_endpoint_wall_drift"][
            :, final_index, :, pe, ps, pr
        ].mean(axis=1)
        for wall in range(2):
            mean, low, high = bootstrap_curve_ci(
                curves[:, wall], repetitions=repetitions,
                seed=seed + 3000 + 10 * panel + wall,
            )
            color = ("#2166ac", "#b2182b")[wall]
            axes[panel].plot(
                item["modular_times"], mean, color=color, linewidth=1.0,
                label=f"wall {wall}, $x={int(item['wall_x'][wall])}$",
            )
            axes[panel].fill_between(
                item["modular_times"], low, high, color=color, alpha=0.16, linewidth=0,
            )
        axes[panel].axhline(0.0, color="0.45", linestyle=":", linewidth=0.7)
        axes[panel].axvline(
            float(config["packet_observer"]["fixed_time"]),
            color="0.45", linestyle="--", linewidth=0.7,
        )
        axes[panel].set_title(f"{protocol} wall, $\\alpha_1=1$")
        axes[panel].set_xlabel(r"modular time $\tau$")
        axes[panel].legend(frameon=False, fontsize=6.5)
        axes[panel].text(-0.13, 1.03, f"({chr(97 + panel)})", transform=axes[panel].transAxes, fontweight="bold")
    axes[0].set_ylabel(r"raw paired endpoint drift $D_w(\tau)$")
    for suffix in ("pdf", "png"):
        figure.savefig(
            output_root / f"h1_endpoint_raw_wall_drift.{suffix}",
            dpi=300, bbox_inches="tight",
        )
    plt.close(figure)

    provenance_counts = Counter(
        row.get("mode", "missing")
        for item in merged.values() for row in item["trajectory_provenance"]
    )
    minimum_r2 = float(analysis_config["minimum_velocity_fit_r2"])
    velocity_diagnostics = {}
    for protocol in ("hard", "soft"):
        case_id = f"H1_N20x40_{protocol}_a1-1"
        row = summary_by_id[case_id]
        sensitivity = [
            item for item in sensitivity_rows if item["case_id"] == case_id
        ]
        quality = bool(
            row["wall_0_velocity_median_r2"] >= minimum_r2
            and row["wall_1_velocity_median_r2"] >= minimum_r2
        )
        sign_stable = bool(all(item["H_velocity_positive"] for item in sensitivity))
        velocity_diagnostics[protocol] = {
            "minimum_fit_r2": minimum_r2,
            "wall_median_r2": [
                row["wall_0_velocity_median_r2"],
                row["wall_1_velocity_median_r2"],
            ],
            "fit_quality_pass": quality,
            "fit_window_sign_stable": sign_stable,
            "report_velocity": bool(quality and sign_stable),
        }
    result = {
        "schema": "h1_endpoint_packet_analysis_v4_mixed_server_verified",
        "status": "accepted" if all(row["pass"] for row in gates) else "not_accepted",
        "case_count": 4,
        "samples_per_case": 25,
        "primary_checkpoint": primary_checkpoint,
        "translated_cuts_per_trajectory_checkpoint": 40,
        "primary_estimator": "construct D_w per translated cut; orient wall deltas; average cuts within trajectory; bootstrap 25 parent trajectories",
        "secondary_velocity_interpretation": "modular estimator speed only; not a physical or spectral group velocity",
        "secondary_velocity_diagnostics": velocity_diagnostics,
        "response_products": "omitted",
        "summary_rows": summary_rows,
        "contrast_rows": contrast_rows,
        "gates": gates,
        "minimum_primary_retention": minimum_primary_retention,
        "maximum_raw_packet_norm_error": maximum_raw_norm_error,
        "maximum_raw_packet_norm_drift": maximum_raw_norm_drift,
        "maximum_post_normalization_norm_error": (
            maximum_post_normalization_norm_error
        ),
        "raw_norm_warning_tolerance": warning_tolerance,
        "raw_norm_hard_failure_tolerance": hard_failure_tolerance,
        "post_normalization_norm_tolerance": post_normalization_tolerance,
        "raw_norm_warning_case_ids": warning_case_ids,
        "raw_norm_warning_non_gating": True,
        "raw_norm_error_argmax": raw_norm_error_argmax,
        "dtype_contract": dtype_contract_rows,
        "dtype_contract_pass": bool(dtype_contract_pass),
        "maximum_conditional_eigenvector_gram_residual": (
            maximum_conditional_gram_residual
        ),
        "trajectory_provenance_shard_counts": dict(provenance_counts),
        "outputs": sorted(path.name for path in output_root.iterdir()),
    }
    write_json_atomic(output_root / "h1_endpoint_analysis_summary.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--bundle-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--drive-root", type=Path)
    args = parser.parse_args(argv)
    result = analyze(
        args.archive_root,
        args.output_root,
        bundle_root=args.bundle_root,
        drive_root=args.drive_root,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
