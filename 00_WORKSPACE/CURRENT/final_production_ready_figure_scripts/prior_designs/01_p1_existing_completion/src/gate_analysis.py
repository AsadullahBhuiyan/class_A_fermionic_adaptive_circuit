from __future__ import annotations

import argparse
import io
import json
import math
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from production_runtime import SHARD_SIZE, sha256_file, verify_archive_receipt, write_json_atomic


def _archive_payload(path: Path) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, np.ndarray] | None]:
    verify_archive_receipt(path)
    with tarfile.open(path, "r:gz") as archive:
        members = {member.name.lstrip("./"): member for member in archive.getmembers()}
        manifest_member = members.get("manifest.json")
        if manifest_member is None:
            raise ValueError(f"{path}: missing manifest.json")
        manifest_handle = archive.extractfile(manifest_member)
        if manifest_handle is None:
            raise ValueError(f"{path}: unreadable manifest")
        manifest = json.loads(manifest_handle.read().decode("utf-8"))
        selected_names = [name for name in members if name.endswith("/selected_observables.npz")]
        if len(selected_names) != 1:
            raise ValueError(f"{path}: expected one selected_observables.npz")
        selected_handle = archive.extractfile(members[selected_names[0]])
        if selected_handle is None:
            raise ValueError(f"{path}: unreadable selected observables")
        with np.load(io.BytesIO(selected_handle.read()), allow_pickle=False) as data:
            selected = {key: np.array(data[key], copy=True) for key in data.files}
        tangent_names = [name for name in members if name.endswith("/tangent_qr.npz")]
        tangent = None
        if tangent_names:
            tangent_handle = archive.extractfile(members[tangent_names[0]])
            if tangent_handle is not None:
                with np.load(io.BytesIO(tangent_handle.read()), allow_pickle=False) as data:
                    tangent = {key: np.array(data[key], copy=True) for key in data.files}
    return manifest, selected, tangent


def _complete_groups(root: Path, campaign: str) -> tuple[dict[str, list[tuple]], list[str]]:
    groups: dict[str, list[tuple]] = defaultdict(list)
    errors: list[str] = []
    for path in sorted(root.glob("*.tar.gz")):
        try:
            manifest, selected, tangent = _archive_payload(path)
            case = manifest.get("run_config", {}).get("case", {})
            if case.get("campaign") != campaign:
                continue
            groups[str(case["case_id"])].append((path, manifest, selected, tangent))
        except Exception as exc:
            errors.append(f"{path.name}: {exc}")
    return groups, errors


def _strip_slope(curve: np.ndarray, ny: int) -> np.ndarray:
    ay = np.arange(curve.shape[-1], dtype=np.float64)
    mask = (ay >= 2) & (ay <= ny / 2 - 1)
    x = np.log((ny / math.pi) * np.sin(math.pi * ay[mask] / ny))
    design = np.column_stack((np.ones_like(x), x))
    pinv = np.linalg.pinv(design)
    return np.asarray([(pinv @ row[mask])[1] for row in curve], dtype=np.float64)


def _mean_se(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    return float(np.mean(values)), float(np.std(values, ddof=1) / math.sqrt(values.size))


def _load_b0_width_gate(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing pinned B0 exact-width gate: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema": "B0_exact_domain_wall_gate_reference_v2",
        "status": "accepted",
        "all_requirements_pass": True,
    }
    for key, expected in required.items():
        if payload.get(key) != expected:
            raise ValueError(f"{path}: expected {key}={expected!r}, got {payload.get(key)!r}")
    accepted_nx = payload.get("accepted_Nx")
    if not isinstance(accepted_nx, int) or accepted_nx <= 0:
        raise ValueError(f"{path}: accepted_Nx must be a positive integer")
    for key in ("source_manifest_sha256", "source_config_sha256"):
        value = payload.get(key)
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"{path}: {key} is not a SHA-256 digest")
    return payload


def analyze_width_gate(*, archive_root: Path, output: Path, b0_gate: Path) -> dict[str, Any]:
    b0 = _load_b0_width_gate(b0_gate)
    groups, errors = _complete_groups(archive_root, "W1")
    rows = []
    incomplete = []
    for case_id, shards in sorted(groups.items()):
        indices = sorted(int(item[1]["shard_index"]) for item in shards)
        samples = int(shards[0][1]["run_config"]["case"]["run"]["samples"])
        expected_indices = list(range(math.ceil(samples / SHARD_SIZE)))
        if indices != expected_indices:
            incomplete.append({"case_id": case_id, "shards": indices})
            continue
        case = shards[0][1]["run_config"]["case"]
        nx, ny = int(case["model"]["Nx"]), int(case["model"]["Ny"])
        protocol = case_id.split(f"N{nx}x{ny}_", 1)[1]
        final_cycle = 2 * ny
        strip_key = f"strip_entropy_cycle_{final_cycle:04d}"
        strip = np.concatenate([item[2][strip_key] for item in shards], axis=0)
        slopes = _strip_slope(strip, ny)
        slope_mean, slope_se = _mean_se(slopes)
        opposite_weights = []
        localization_lengths = []
        for _, _, selected, tangent in shards:
            if tangent is None:
                continue
            spectra = tangent["reported_final_window_spectrum"]
            xweight = tangent[f"aligned_frame_x_weight_cycle_{final_cycle:04d}"]
            for sample in range(xweight.shape[0]):
                slow = int(np.argmax(spectra[sample]))
                profile = xweight[sample, slow]
                centers = (max(0, nx // 2 - max(1, nx // 4)), min(nx, nx // 2 + max(1, nx // 4) + 1) - 1)
                wall_mass = np.asarray([profile[(center - 1) % nx] + profile[center] + profile[(center + 1) % nx] for center in centers])
                dominant = int(np.argmax(wall_mass))
                opposite_weights.append(float(wall_mass[1 - dominant] / max(np.sum(profile), 1e-300)))
                distance = np.asarray([min((x - centers[dominant]) % nx, (centers[dominant] - x) % nx) for x in range(nx)])
                mask = (distance >= 1) & (profile > 1e-14)
                if np.count_nonzero(mask) >= 3:
                    slope = np.polyfit(distance[mask], np.log(profile[mask]), 1)[0]
                    localization_lengths.append(float(-2.0 / slope) if slope < 0 else float("inf"))
        rows.append(
            {
                "case_id": case_id,
                "Nx": nx,
                "Ny": ny,
                "protocol": protocol,
                "entropy_slope_mean": slope_mean,
                "entropy_slope_se": slope_se,
                "opposite_wall_weight_upper95": float(np.quantile(opposite_weights, 0.95)) if opposite_weights else float("inf"),
                "localization_length_upper95": float(np.quantile(localization_lengths, 0.95)) if localization_lengths else float("inf"),
                "samples": int(strip.shape[0]),
            }
        )
    required_cases = 24
    if incomplete or len(rows) != required_cases:
        payload = {
            "schema_version": 1,
            "status": "incomplete",
            "complete_case_count": len(rows),
            "required_case_count": required_cases,
            "incomplete": incomplete,
            "archive_errors": errors,
            "exact_B0_transverse_gate": b0,
            "exact_B0_reference_sha256": sha256_file(b0_gate),
        }
        write_json_atomic(output, payload)
        return payload

    threshold = 0.05
    candidates = []
    width_grid = sorted({row["Nx"] for row in rows})
    for nx in width_grid:
        passed = nx == width_grid[0]
        diagnostics = []
        for ny in (48, 80):
            for protocol in sorted({row["protocol"] for row in rows}):
                by_width = {
                    row["Nx"]: row
                    for row in rows
                    if row["Ny"] == ny and row["protocol"] == protocol
                }
                if nx not in by_width:
                    passed = False
                    continue
                current = by_width[nx]
                higher = [width for width in sorted(by_width) if width > nx][:2]
                if len(higher) < 2:
                    passed = False
                for width in higher:
                    other = by_width[width]
                    delta = abs(current["entropy_slope_mean"] - other["entropy_slope_mean"])
                    tolerance = 1.96 * math.sqrt(current["entropy_slope_se"] ** 2 + other["entropy_slope_se"] ** 2)
                    if delta > tolerance:
                        passed = False
                if current["opposite_wall_weight_upper95"] >= threshold:
                    passed = False
                separation = max(1.0, nx / 2)
                if current["localization_length_upper95"] * math.log(ny) >= separation:
                    passed = False
                diagnostics.append(current)
        candidates.append({"Nx": nx, "passed": passed, "diagnostics": diagnostics})
    accepted = next((row["Nx"] for row in candidates if row["passed"]), None)
    b0_accepted = int(b0["accepted_Nx"])
    status = "accepted" if accepted == b0_accepted else "failed_width_gate_disagreement"
    payload = {
        "schema_version": 1,
        "status": status if accepted is not None else "failed_extend_width_grid",
        "accepted_Nx": accepted if accepted == b0_accepted else None,
        "W1_candidate_Nx": accepted,
        "joint_confidence_level": 0.95,
        "opposite_wall_weight_threshold": threshold,
        "exact_B0_transverse_gate": b0,
        "exact_B0_reference_sha256": sha256_file(b0_gate),
        "candidates": candidates,
        "input_archives": {path.name: sha256_file(path) for path in sorted(archive_root.glob("*.tar.gz"))},
        "archive_errors": errors,
    }
    write_json_atomic(output, payload)
    return payload


def analyze_m3_bulk_gate(*, archive_root: Path, output: Path) -> dict[str, Any]:
    groups, errors = _complete_groups(archive_root, "M3_BULK")
    rows = []
    incomplete = []
    for case_id, shards in sorted(groups.items()):
        indices = sorted(int(item[1]["shard_index"]) for item in shards)
        samples = int(shards[0][1]["run_config"]["case"]["run"]["samples"])
        expected_indices = list(range(math.ceil(samples / SHARD_SIZE)))
        if indices != expected_indices:
            incomplete.append({"case_id": case_id, "shards": indices})
            continue
        case = shards[0][1]["run_config"]["case"]
        size = int(case["model"]["Nx"])
        sigma = float(case["run"]["onsite_phase_noise_sigma"])
        selected_all = [item[2] for item in shards]
        obs_cycles = selected_all[0]["cycles"].astype(int).tolist()
        final_index = obs_cycles.index(2 * size)
        marker = np.concatenate([item["real_space_chern"][:, final_index] for item in selected_all])
        gap = np.concatenate([item["purity_gap"][:, final_index] for item in selected_all])
        tangent_gap = []
        for _, _, _, tangent in shards:
            if tangent is not None:
                spectrum = tangent["reported_final_window_spectrum"]
                tangent_gap.extend(np.sort(np.abs(spectrum), axis=1)[:, 0].tolist())
        rows.append(
            {
                "N": size,
                "sigma": sigma,
                "marker_mean": float(np.mean(marker)),
                "purity_gap_mean": float(np.mean(gap)),
                "tangent_gap_mean": float(np.mean(tangent_gap)) if tangent_gap else float("nan"),
                "samples": int(marker.size),
            }
        )
    required = 5 * 29
    if incomplete or len(rows) != required:
        payload = {
            "schema_version": 1,
            "status": "incomplete",
            "complete_case_count": len(rows),
            "required_case_count": required,
            "incomplete": incomplete,
            "archive_errors": errors,
        }
        write_json_atomic(output, payload)
        return payload
    critical = []
    for size in sorted({row["N"] for row in rows}):
        series = sorted((row for row in rows if row["N"] == size), key=lambda row: row["sigma"])
        sigma = np.asarray([row["sigma"] for row in series])
        marker = np.asarray([row["marker_mean"] for row in series])
        gap = np.asarray([row["purity_gap_mean"] for row in series])
        tangent = np.asarray([row["tangent_gap_mean"] for row in series])
        marker_index = int(np.argmax(np.abs(np.gradient(marker, sigma))))
        gap_index = int(np.argmin(gap))
        tangent_index = int(np.nanargmin(tangent))
        estimates = sigma[[marker_index, gap_index, tangent_index]]
        critical.append({"N": size, "estimates": estimates.tolist(), "median": float(np.median(estimates))})
    center = critical[-1]["median"]
    grid = np.asarray(sorted({row["sigma"] for row in rows}))
    nearest = int(np.argmin(np.abs(grid - center)))
    lo, hi = max(0, nearest - 2), min(grid.size, nearest + 3)
    bracket = grid[lo:hi].tolist()
    payload = {
        "schema_version": 1,
        "status": "accepted",
        "bulk_pseudocritical_estimates": critical,
        "wall_sigma_bracket": bracket,
        "bracket_rule": "nearest crossing point plus two neighboring grid points on each side",
        "input_archives": {path.name: sha256_file(path) for path in sorted(archive_root.glob("*.tar.gz"))},
        "archive_errors": errors,
    }
    write_json_atomic(output, payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze completed compact production archives")
    parser.add_argument("gate", choices=("width", "m3_bulk"))
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--b0-gate",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "b0_accepted_width.json",
        help="Pinned accepted B0 exact-domain-wall width reference",
    )
    args = parser.parse_args(argv)
    result = (
        analyze_width_gate(archive_root=args.archive_root, output=args.output, b0_gate=args.b0_gate)
        if args.gate == "width"
        else analyze_m3_bulk_gate(archive_root=args.archive_root, output=args.output)
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] in ("accepted", "passed") else 2


if __name__ == "__main__":
    raise SystemExit(main())
