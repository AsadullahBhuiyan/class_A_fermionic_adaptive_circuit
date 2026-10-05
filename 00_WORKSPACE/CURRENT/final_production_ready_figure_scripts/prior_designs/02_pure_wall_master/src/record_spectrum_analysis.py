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

from production_runtime import (
    PRODUCTION_SAMPLES,
    SHARD_SIZE,
    save_npz_atomic,
    sha256_file,
    verify_archive_receipt,
    write_json_atomic,
)


def _logmeanexp(values: np.ndarray, axis: int = 0) -> np.ndarray:
    maximum = np.max(values, axis=axis, keepdims=True)
    return np.squeeze(maximum, axis=axis) + np.log(
        np.mean(np.exp(values - maximum), axis=axis)
    )


def _read(path: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]] | None:
    verify_archive_receipt(path)
    with tarfile.open(path, "r:gz") as archive:
        members = {item.name.lstrip("./"): item for item in archive.getmembers()}
        manifest_member = members.get("manifest.json")
        if manifest_member is None:
            return None
        handle = archive.extractfile(manifest_member)
        if handle is None:
            return None
        manifest = json.loads(handle.read().decode("utf-8"))
        case = manifest.get("run_config", {}).get("case", {})
        if case.get("campaign") != "R1_RECORD_SPECTRUM":
            return None
        names = [name for name in members if name.endswith("/ordered_record.npz")]
        if len(names) != 1:
            raise ValueError(f"{path}: expected one ordered record")
        record_handle = archive.extractfile(members[names[0]])
        if record_handle is None:
            raise ValueError(f"{path}: unreadable ordered record")
        with np.load(io.BytesIO(record_handle.read()), allow_pickle=False) as data:
            record = {key: np.array(data[key], copy=True) for key in data.files}
        return manifest, record


def analyze(*, archive_root: Path, output_root: Path) -> dict[str, Any]:
    groups: dict[str, list[tuple[Path, dict, dict]]] = defaultdict(list)
    errors = []
    for path in sorted(archive_root.glob("*.tar.gz")):
        try:
            item = _read(path)
            if item is not None:
                manifest, record = item
                case_id = manifest["run_config"]["case"]["case_id"]
                groups[case_id].append((path, manifest, record))
        except Exception as exc:
            errors.append(f"{path.name}: {exc}")
    q_grid = np.linspace(-1.0, 2.0, 121)
    summary_rows = []
    output_root.mkdir(parents=True, exist_ok=True)
    for case_id, shards in sorted(groups.items()):
        indices = sorted(int(item[1]["shard_index"]) for item in shards)
        minimum_shards = math.ceil(PRODUCTION_SAMPLES / SHARD_SIZE)
        if (
            len(indices) < minimum_shards
            or len(indices) != len(set(indices))
            or indices != list(range(indices[-1] + 1))
        ):
            continue
        case = shards[0][1]["run_config"]["case"]
        cycle_info = np.concatenate(
            [item[2]["self_information_per_cycle"] for item in shards], axis=0
        )
        signed = np.concatenate(
            [item[2]["signed_transfer_per_cycle"] for item in shards], axis=0
        )
        absolute = np.concatenate(
            [item[2]["absolute_transfer_per_cycle"] for item in shards], axis=0
        )
        wrong = np.concatenate(
            [item[2]["wrong_outcome_per_cycle"] for item in shards], axis=0
        )
        total_info = np.sum(cycle_info, axis=1)
        duration = int(cycle_info.shape[1])
        scgf = _logmeanexp(-q_grid[:, None] * total_info[None, :], axis=1) / duration
        final_transfer = np.sum(signed, axis=1)
        sectors, counts = np.unique(final_transfer, return_counts=True)
        safe = case_id.replace("/", "_")
        path = output_root / f"{safe}_record_spectrum.npz"
        save_npz_atomic(
            path,
            schema=np.asarray("R1_compact_record_spectrum_v1"),
            q_grid=q_grid,
            scgf=scgf,
            self_information_per_cycle=cycle_info,
            signed_transfer_per_cycle=signed,
            absolute_transfer_per_cycle=absolute,
            wrong_outcome_per_cycle=wrong,
            total_self_information=total_info,
            final_signed_transfer=final_transfer,
            sector_labels=sectors,
            sector_counts=counts,
        )
        summary_rows.append(
            {
                "case_id": case_id,
                "Ny": int(case["model"]["Ny"]),
                "samples": int(total_info.size),
                "mean_self_information_rate": float(np.mean(total_info) / duration),
                "mean_absolute_activity_rate": float(np.mean(np.sum(absolute, axis=1)) / duration),
                "mean_wrong_outcome_rate": float(np.mean(np.sum(wrong, axis=1)) / duration),
                "sector_counts": {str(int(key)): int(value) for key, value in zip(sectors, counts)},
                "product": str(path),
                "sha256": sha256_file(path),
            }
        )
    status = "complete" if len(summary_rows) == 6 else "incomplete"
    payload = {
        "schema": "R1_record_spectrum_analysis_v1",
        "status": status,
        "complete_cases": len(summary_rows),
        "required_cases": 6,
        "rows": summary_rows,
        "archive_errors": errors,
    }
    write_json_atomic(output_root / "R1_record_spectrum_summary.json", payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Merge compact R1 record shards")
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(analyze(archive_root=args.archive_root, output_root=args.output_root), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
