#!/usr/bin/env python3
"""Read and verify wall-diabatized pump result/completion pairs.

This module deliberately does not import the campaign runner.  Analysis must be
able to audit a frozen output collection even after the executable campaign
source has changed.  The completion JSON and the metadata embedded in the NPZ
are therefore treated as the durable interface.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


COMPLETION_SCHEMAS = {
    "wall_diabatic_spectral_pump_completion_v1",
    "wall_diabatized_pump_completion_v1",
    "wall_diabatized_width_sweep_completion_v1",
}
RESULT_SCHEMAS = {
    "wall_diabatic_spectral_pump_result_v1",
    "wall_diabatized_pump_path_v1",
    "wall_diabatized_width_sweep_path_v1",
}


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def scalar(value: Any) -> Any:
    array = np.asarray(value)
    return array.item() if array.ndim == 0 else value


def first_array(saved: Mapping[str, Any], names: Iterable[str], *, required: bool = True) -> np.ndarray | None:
    for name in names:
        if name in saved:
            return np.asarray(saved[name])
    if required:
        raise KeyError(f"missing required NPZ field; accepted aliases={tuple(names)!r}")
    return None


def first_scalar(
    saved: Mapping[str, Any], names: Iterable[str], *, required: bool = True, default: float = float("nan")
) -> float:
    value = first_array(saved, names, required=required)
    return float(scalar(value)) if value is not None else float(default)


@dataclass(frozen=True)
class VerifiedPath:
    result_path: Path
    completion_path: Path
    completion: dict[str, Any]
    metadata: dict[str, Any]
    arrays: dict[str, np.ndarray]

    @property
    def task_id(self) -> str:
        return str(self.metadata["task_id"])


def _result_from_completion(completion_path: Path, completion: Mapping[str, Any]) -> Path:
    record = completion.get("result")
    if not isinstance(record, Mapping) or not isinstance(record.get("name"), str):
        raise ValueError("completion lacks result name")
    declared = record["name"]
    direct = completion_path.with_name(declared)
    if direct.is_file():
        return direct
    matches = list(completion_path.parent.rglob(declared))
    if len(matches) != 1:
        raise ValueError(f"result name resolves to {len(matches)} files: {declared!r}")
    return matches[0]


def verify_completion(completion_path: Path) -> VerifiedPath:
    completion_path = Path(completion_path)
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    schema = str(completion.get("schema", ""))
    if schema not in COMPLETION_SCHEMAS:
        raise ValueError(f"unexpected completion schema {schema!r}")
    result_path = _result_from_completion(completion_path, completion)
    record = completion["result"]
    if int(record.get("bytes", -1)) != result_path.stat().st_size:
        raise ValueError("result byte count does not match completion")
    if str(record.get("sha256", "")) != sha256_path(result_path):
        raise ValueError("result SHA-256 does not match completion")

    with np.load(result_path, allow_pickle=False) as saved:
        result_schema = str(scalar(saved["schema"]))
        if result_schema not in RESULT_SCHEMAS:
            raise ValueError(f"unexpected result schema {result_schema!r}")
        metadata = json.loads(str(scalar(saved["metadata_json"])))
        arrays = {name: np.array(saved[name], copy=True) for name in saved.files}

    for key, value in metadata.items():
        if completion.get(key) != value:
            raise ValueError(f"embedded metadata disagrees with completion for {key}")
    if completion.get("stage") not in (None, "pump", "control", "wall_diabatic_pump"):
        raise ValueError(f"not a pump/control completion: stage={completion.get('stage')!r}")
    for key in ("task_id", "wall", "sample_id"):
        if key not in metadata:
            raise ValueError(f"embedded metadata lacks {key}")

    phi = first_array(arrays, ("phi",))
    q_x = first_array(arrays, ("q_x", "raw_q_x"))
    delta_left = first_array(arrays, ("delta_N_left", "delta_n_left"))
    delta_right = first_array(arrays, ("delta_N_right", "delta_n_right"))
    delta_total = first_array(arrays, ("delta_N_total", "delta_n_total"))
    allowed_ndim = 1 if "direction" in metadata else 2
    if not all(array.ndim == allowed_ndim for array in (phi, q_x, delta_left, delta_right, delta_total)):
        raise ValueError(f"primary charge histories must be {allowed_ndim}-dimensional")
    if len({array.shape for array in (phi, q_x, delta_left, delta_right, delta_total)}) != 1:
        raise ValueError("primary charge histories have unequal shapes")
    if phi.shape[-1] < 2 or not all(np.all(np.isfinite(array)) for array in (phi, q_x, delta_left, delta_right, delta_total)):
        raise ValueError("primary charge histories are empty or non-finite")
    if not np.allclose(q_x, 0.5 * (delta_right - delta_left), rtol=0.0, atol=2e-10):
        raise ValueError("q_x is inconsistent with the regional charges")
    if not np.allclose(delta_total, delta_right + delta_left, rtol=0.0, atol=2e-10):
        raise ValueError("total charge is inconsistent with the regional charges")
    intervals = int(metadata.get("grid_intervals", phi.shape[-1] - 1))
    if phi.shape[-1] != intervals + 1:
        raise ValueError("flux history length disagrees with grid_intervals")
    if allowed_ndim == 2:
        directions = first_array(arrays, ("directions", "direction"))
        sigma = first_array(arrays, ("sigma", "sigmas"))
        if directions.shape != (phi.shape[0],) or sigma.shape != (phi.shape[0],):
            raise ValueError("directions/sigma do not match the directional history axis")
    return VerifiedPath(result_path, completion_path, completion, metadata, arrays)


def discover_verified_paths(
    output_root: Path, *, allow_invalid: bool = False
) -> tuple[list[VerifiedPath], list[dict[str, str]]]:
    output_root = Path(output_root)
    rows: list[VerifiedPath] = []
    invalid: list[dict[str, str]] = []
    for completion_path in sorted(output_root.rglob("*.completion.json")):
        try:
            completion = json.loads(completion_path.read_text(encoding="utf-8"))
        except Exception as exc:
            invalid.append({"path": str(completion_path), "error": f"{type(exc).__name__}: {exc}"})
            continue
        if str(completion.get("schema", "")) not in COMPLETION_SCHEMAS:
            continue
        try:
            rows.append(verify_completion(completion_path))
        except Exception as exc:
            invalid.append({"path": str(completion_path), "error": f"{type(exc).__name__}: {exc}"})
    task_ids = [row.task_id for row in rows]
    duplicates = sorted({task_id for task_id in task_ids if task_ids.count(task_id) > 1})
    if duplicates:
        raise RuntimeError(f"duplicate verified task IDs: {duplicates[:8]}")
    if invalid and not allow_invalid:
        raise RuntimeError(f"found {len(invalid)} invalid wall-diabatized completion pairs")
    return rows, invalid
