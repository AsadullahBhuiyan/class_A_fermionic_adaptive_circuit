#!/usr/bin/env python3
"""Build the clean, independent two-campaign v4 Colab deployment."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUNDLES = ("01_p1_chern_dynamics", "08_h1_endpoint_packet")
ROOT_FILES = (
    "colab_bundle_runner.py",
    "production_runtime.py",
    "drive_remote_commit.py",
    "p1_runtime_hardening.py",
    "server_verified_analysis.py",
)
V4_OPERATIONAL_RELEASE = "v4-drivefs-independent-20260901-r2"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ignore(_: str, names: list[str]) -> set[str]:
    ignored = {
        name
        for name in names
        if name == "__pycache__"
        or name.endswith((".pyc", ".pyo", ".orig", ".rej", "~"))
    }
    return ignored


def build(destination: Path) -> dict:
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError(
            f"refusing to overwrite deployment directory: {destination}"
        )
    destination.mkdir(parents=True)
    for bundle in BUNDLES:
        shutil.copytree(ROOT / bundle, destination / bundle, ignore=_ignore)
    for name in ROOT_FILES:
        source = (
            ROOT
            / ("_shared_src" if name == "production_runtime.py" else "")
            / name
        )
        shutil.copy2(source, destination / name)

    layout = '''"""Layout for the independent P1/H1 v4 deployment."""
from pathlib import Path

NEW_DESIGN_BUNDLES = ("01_p1_chern_dynamics", "08_h1_endpoint_packet")

def bundle_path(root: Path | str, bundle: str) -> Path:
    if bundle not in NEW_DESIGN_BUNDLES:
        raise KeyError(f"unknown v4 bundle {bundle!r}")
    return Path(root) / bundle
'''
    (destination / "bundle_layout.py").write_text(layout, encoding="utf-8")
    index = {
        "schema_version": 4,
        "campaign_parent": "final_production_new_designs_v4",
        "canonical_entry_point": "classA_U1FGTN_gpu.run_markov_circuit",
        "gpu": "NVIDIA A100 40GB",
        "bundles": list(BUNDLES),
        "standalone_contracts": {
            "01_p1_chern_dynamics": "production_25sample_p1_chern_v4",
            "08_h1_endpoint_packet": "production_25sample_h1_endpoint_packet_v4",
        },
        "drive_durability": "google_drive_api_v3_server_size_parent_name_sha256",
    }
    (destination / "bundle_index.json").write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    pilot = json.loads((ROOT / "pilot_plan.json").read_text(encoding="utf-8"))
    for profile in ("calibration", "science"):
        key = f"pilot_{profile}"
        if key in pilot:
            pilot[key] = {
                bundle: pilot[key][bundle]
                for bundle in BUNDLES
                if bundle in pilot[key]
            }
    (destination / "pilot_plan.json").write_text(
        json.dumps(pilot, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (destination / "README.md").write_text(
        "# final_production_new_designs_v4\n\n"
        "Independent P1/H1 deployment. Drive API metadata and SHA-256 are the "
        "durability authority; DriveFS is only a cache. The v3 output folders are "
        "never modified. Open either bundle's generated production notebook.\n",
        encoding="utf-8",
    )

    files = {
        str(path.relative_to(destination)): {
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in sorted(destination.rglob("*"))
        if path.is_file() and path.name != "deployment_manifest.json"
    }
    manifest = {
        "schema": "classA_v4_deployment_manifest_v1",
        "operational_release": V4_OPERATIONAL_RELEASE,
        "campaign_parent": destination.name,
        "bundles": list(BUNDLES),
        "file_count": len(files),
        "files": files,
    }
    (destination / "deployment_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = build(args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
