#!/usr/bin/env python3
"""Synchronize source snapshots for the independent redesigned Colab campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FINAL_ROOT = ROOT / "final_production_new_designs"
SHARED_ROOT = FINAL_ROOT / "_shared_src"
CANONICAL_GPU = ROOT / "src" / "fgtn" / "classA_U1FGTN_gpu.py"
CANONICAL_GPU_FRAME = ROOT / "src" / "fgtn" / "occupied_frame_gpu.py"

BUNDLE_HELPERS = {
    "01_p1_chern_dynamics": [
        SHARED_ROOT / "p1_chern_observables.py",
        SHARED_ROOT / "p1_chern_runner.py",
        SHARED_ROOT / "p1_chern_analysis.py",
        SHARED_ROOT / "drive_storage_guard.py",
        # P1-v4 checkpoints pin this exact helper byte-for-byte.  Operational
        # transport fixes live at the deployment root and are installed by the
        # unhashed wrapper; never overwrite the qualified snapshot here.
        FINAL_ROOT / "01_p1_chern_dynamics" / "src" / "drive_remote_commit.py",
    ],
    "02_wall_cft_windows": [
        SHARED_ROOT / "wall_window_observables.py",
        SHARED_ROOT / "wall_window_runner.py",
        SHARED_ROOT / "wall_window_loader.py",
        SHARED_ROOT / "drive_storage_guard.py",
    ],
    "03_h1_modular_response": [
        FINAL_ROOT / "03_h1_modular_response" / "src" / "h1_io.py",
        FINAL_ROOT / "03_h1_modular_response" / "src" / "h1_record.py",
        FINAL_ROOT / "03_h1_modular_response" / "src" / "h1_modular_observables.py",
        FINAL_ROOT / "03_h1_modular_response" / "src" / "h1_modular_runner.py",
        FINAL_ROOT / "03_h1_modular_response" / "src" / "h1_modular_analysis.py",
        SHARED_ROOT / "drive_storage_guard.py",
    ],
    "04_maxmix_operator_cft": [
        FINAL_ROOT / "04_maxmix_operator_cft" / "src" / "g4_observables.py",
        FINAL_ROOT / "04_maxmix_operator_cft" / "src" / "g4_runner.py",
        FINAL_ROOT / "04_maxmix_operator_cft" / "src" / "g4_analysis.py",
        SHARED_ROOT / "drive_storage_guard.py",
    ],
    "05_pure_tangent_stability": [
        FINAL_ROOT / "05_pure_tangent_stability" / "src" / "g5_tangent_observer.py",
        FINAL_ROOT / "05_pure_tangent_stability" / "src" / "g5_runner.py",
        FINAL_ROOT / "05_pure_tangent_stability" / "src" / "g5_analysis.py",
        SHARED_ROOT / "drive_storage_guard.py",
    ],
    "07_log_gram_alpha_scan": [
        FINAL_ROOT / "07_log_gram_alpha_scan" / "src" / "log_gram_observer.py",
        FINAL_ROOT / "07_log_gram_alpha_scan" / "src" / "log_gram_runner.py",
        SHARED_ROOT / "drive_storage_guard.py",
    ],
    "08_h1_endpoint_packet": [
        FINAL_ROOT / "08_h1_endpoint_packet" / "src" / "h1_io.py",
        FINAL_ROOT / "08_h1_endpoint_packet" / "src" / "h1_record.py",
        FINAL_ROOT / "08_h1_endpoint_packet" / "src" / "h1_packet_observables.py",
        FINAL_ROOT / "08_h1_endpoint_packet" / "src" / "h1_packet_runner.py",
        FINAL_ROOT / "08_h1_endpoint_packet" / "src" / "h1_packet_analysis.py",
        FINAL_ROOT / "08_h1_endpoint_packet" / "src" / "h1_v3_migration.py",
        SHARED_ROOT / "drive_storage_guard.py",
        # H1-v4's saved qualification pins this exact helper.  Keep it
        # self-canonical while root-level wrappers harden transport behavior.
        FINAL_ROOT / "08_h1_endpoint_packet" / "src" / "drive_remote_commit.py",
    ],
}

RETIRED_GENERATED = {
    "continuous_lindblad_gpu.py",
    "run_continuous_lindblad_shard.py",
    "continuous_lindblad_analysis.py",
    "run_m1_shard.py",
    "m1_deterministic_controls.py",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def repository_source(path: Path) -> str:
    return str(path.relative_to(ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle",
        action="append",
        choices=sorted(BUNDLE_HELPERS),
        help="synchronize only this bundle (repeatable); default synchronizes all redesigns",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    selected_bundles = set(args.bundle or BUNDLE_HELPERS)
    for bundle, helpers in BUNDLE_HELPERS.items():
        if bundle not in selected_bundles:
            continue
        src_dir = FINAL_ROOT / bundle / "src"
        src_dir.mkdir(parents=True, exist_ok=True)
        for retired in RETIRED_GENERATED:
            retired_path = src_dir / retired
            if retired_path.exists():
                retired_path.unlink()

        copied: dict[str, dict[str, str]] = {}
        for source in [CANONICAL_GPU, CANONICAL_GPU_FRAME, *helpers]:
            if not source.exists():
                raise FileNotFoundError(source)
            destination = src_dir / source.name
            if source.resolve() != destination.resolve():
                shutil.copy2(source, destination)
            copied[source.name] = {
                "sha256": sha256(destination),
                "repository_source": repository_source(source),
            }

        init_path = src_dir / "__init__.py"
        init_path.write_text(
            '"""Generated, self-contained Colab bundle source."""\n',
            encoding="utf-8",
        )
        copied["__init__.py"] = {
            "sha256": sha256(init_path),
            "repository_source": "generated",
        }
        manifest = {
            "schema_version": 1,
            "canonical_gpu_sha256": sha256(CANONICAL_GPU),
            "files": copied,
            "warning": "Generated copies; edit repository sources and rerun sync_bundle_sources.py.",
        }
        (src_dir / "source_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"[synced] {bundle}: {len(copied)} Python files")


if __name__ == "__main__":
    main()
