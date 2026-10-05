"""Lazy archive reader and completion index for wall-window shards."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Iterator

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _member(archive: tarfile.TarFile, suffix: str) -> tarfile.TarInfo:
    matches = [
        row for row in archive.getmembers()
        if row.name.lstrip("./") == suffix
        or row.name.lstrip("./").endswith("/" + suffix)
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one archive member ending {suffix!r}; found {len(matches)}")
    return matches[0]


def manifest_from_archive(path: Path | str) -> dict[str, Any]:
    with tarfile.open(path, "r:gz") as archive:
        handle = archive.extractfile(_member(archive, "manifest.json"))
        if handle is None:
            raise RuntimeError("manifest member is unreadable")
        return json.load(handle)


def load_npz_member(path: Path | str, member_suffix: str) -> dict[str, np.ndarray]:
    """Load one requested compressed product without extracting or merging the archive."""

    with tarfile.open(path, "r:gz") as archive:
        handle = archive.extractfile(_member(archive, member_suffix))
        if handle is None:
            raise RuntimeError(f"unreadable product {member_suffix}")
        with np.load(io.BytesIO(handle.read()), allow_pickle=False) as data:
            return {key: data[key] for key in data.files}


def iter_case_window_height(
    archives: list[Path | str], *, ay: int
) -> Iterator[dict[str, np.ndarray]]:
    """Yield each immutable shard's requested Ay product in shard order."""

    rows = sorted((manifest_from_archive(path)["shard_index"], Path(path)) for path in archives)
    for _, path in rows:
        yield load_npz_member(path, f"wall_windows/Ay_{int(ay):03d}.npz")


def build_case_index(archive_root: Path | str, *, case_id: str) -> dict[str, Any]:
    root = Path(archive_root)
    found = []
    for archive in sorted(root.glob("*.tar.gz")):
        receipt_path = archive.with_suffix(archive.suffix + ".receipt.json")
        if not receipt_path.is_file():
            continue
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("archive_sha256") != sha256_file(archive):
            raise RuntimeError(f"archive receipt checksum failed: {archive}")
        manifest = manifest_from_archive(archive)
        if manifest.get("case_id") == case_id:
            found.append((int(manifest["shard_index"]), archive, manifest))
    found.sort()
    if [row[0] for row in found] != list(range(5)):
        raise RuntimeError(f"{case_id}: expected exactly shards 0..4")
    sample_ids = [sample for _, _, manifest in found for sample in manifest["global_sample_indices"]]
    if sample_ids != list(range(25)):
        raise RuntimeError(f"{case_id}: exact global sample IDs 0..24 are not complete")
    common_shapes = []
    for _, archive, manifest in found:
        common = load_npz_member(archive, "wall_windows/common.npz")
        if common["global_sample_ids"].tolist() != manifest["global_sample_indices"]:
            raise RuntimeError(f"{archive}: manifest/product sample IDs disagree")
        if not np.isfinite(common["square_correlator"]).all():
            raise RuntimeError(f"{archive}: non-finite squared correlator")
        common_shapes.append(list(common["square_correlator"].shape))
    return {
        "schema": "wall_windows_case_completion_index_v1",
        "case_id": case_id, "complete": True, "sample_ids": sample_ids,
        "shards": [
            {"shard_index": shard, "archive": archive.name, "archive_sha256": sha256_file(archive),
             "run_config_hash": manifest["run_config_hash"]}
            for shard, archive, manifest in found
        ],
        "square_correlator_shapes": common_shapes,
        "merge_policy": "lazy immutable-shard loading; no duplicate monolithic merge",
    }


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    payload = build_case_index(args.archive_root, case_id=args.case_id)
    output = args.output or args.archive_root / f"{args.case_id}_completion_index.json"
    write_json_atomic(output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
