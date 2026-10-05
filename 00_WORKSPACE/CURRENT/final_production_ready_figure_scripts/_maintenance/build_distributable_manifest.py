#!/usr/bin/env python3
"""Write a deterministic file inventory for the uploadable campaign package."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bundle_layout import validate_bundle_layout  # noqa: E402

OUTPUT = ROOT / "DISTRIBUTABLE_MANIFEST.json"
IGNORED_PARTS = {"__pycache__", ".pytest_cache", ".ipynb_checkpoints"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    validate_bundle_layout(ROOT)
    files = {}
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or path == OUTPUT or any(part in IGNORED_PARTS for part in path.parts):
            continue
        files[str(path.relative_to(ROOT))] = {
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
    payload = {
        "schema_version": 1,
        "campaign": "production_10sample_v4_occupied_frame_cycle_resolved",
        "audit_sha256": "d0173317608b2da2a45f6185d15a85237e11265917a0ceed6ff79dc761261de9",
        "bundle_output_overrides": {
            "06_b1_controller_frame": "06_b1_controller_frame_frame_native_v2"
        },
        "file_count_excluding_manifest_and_caches": len(files),
        "files": files,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[built] {OUTPUT.name}: {len(files)} files")


if __name__ == "__main__":
    main()
