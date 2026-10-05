#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
COLAB = ROOT.parent / "COLAB"
FINAL_ROOT = ROOT / "final_production_ready_figure_scripts"
SHARED_ROOT = FINAL_ROOT / "_shared_src"
if str(FINAL_ROOT) not in sys.path:
    sys.path.insert(0, str(FINAL_ROOT))

from bundle_layout import bundle_path  # noqa: E402

CANONICAL_GPU = ROOT / "src" / "fgtn" / "classA_U1FGTN_gpu.py"
CANONICAL_GPU_FRAME = ROOT / "src" / "fgtn" / "occupied_frame_gpu.py"

BUNDLE_HELPERS = {
    "00_validation": [
        COLAB / "colab_lyapunov" / "src" / "lyapunov_observables_gpu.py",
        COLAB / "colab_charge_fluctuations" / "src" / "streaming_covariance_observables_gpu.py",
    ],
    "01_bulk_width_gate": [
        COLAB / "colab_lyapunov" / "src" / "lyapunov_observables_gpu.py",
        COLAB / "colab_charge_fluctuations" / "src" / "streaming_covariance_observables_gpu.py",
    ],
    "02_pure_wall_master": [
        COLAB / "colab_lyapunov" / "src" / "lyapunov_observables_gpu.py",
        COLAB / "colab_charge_fluctuations" / "src" / "streaming_covariance_observables_gpu.py",
        COLAB / "colab_large_entanglement_scaling_N20" / "src" / "strip_entropy_streaming_gpu.py",
        SHARED_ROOT / "g5_manybody_spectrum_analysis.py",
    ],
    "03_chirality_replay": [
        COLAB / "colab_small_system_testing" / "src" / "dynamic_modular_charge_spreading.py",
        COLAB / "colab_charge_fluctuations" / "src" / "streaming_covariance_observables_gpu.py",
    ],
    "04_maxmix_master": [
        COLAB / "colab_lyapunov" / "src" / "lyapunov_observables_gpu.py",
        COLAB / "colab_charge_fluctuations" / "src" / "purification_dynamics_observables_gpu.py",
        COLAB / "colab_charge_fluctuations" / "src" / "streaming_covariance_observables_gpu.py",
    ],
    "05_scans_and_controls": [
        COLAB / "colab_lyapunov" / "src" / "lyapunov_observables_gpu.py",
        COLAB / "colab_charge_fluctuations" / "src" / "streaming_covariance_observables_gpu.py",
        SHARED_ROOT / "s2_entanglement_analysis.py",
    ],
    "06_b1_controller_frame": [
        SHARED_ROOT / "b1_controller_frame.py",
        SHARED_ROOT / "run_b1_shard.py",
        SHARED_ROOT / "b1_analysis.py",
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
    """Return the historical manifest path for CURRENT or COLAB source trees."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path.relative_to(COLAB))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Synchronize generated bundle sources")
    parser.add_argument(
        "--bundle",
        action="append",
        choices=sorted(BUNDLE_HELPERS),
        help="synchronize only this bundle (repeatable); default synchronizes every artifact",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    common = [
        SHARED_ROOT / "production_runtime.py",
        SHARED_ROOT / "record_observables.py",
        SHARED_ROOT / "selected_observables.py",
        SHARED_ROOT / "campaign_cases.py",
        SHARED_ROOT / "tangent_observables.py",
        SHARED_ROOT / "run_core_shard.py",
        SHARED_ROOT / "noise_observables.py",
        SHARED_ROOT / "fused_chirality_observables.py",
        SHARED_ROOT / "h3_twist_observables.py",
        SHARED_ROOT / "run_h3_shard.py",
        SHARED_ROOT / "gate_analysis.py",
        SHARED_ROOT / "run_validation_suite.py",
        SHARED_ROOT / "record_spectrum_analysis.py",
        SHARED_ROOT / "drive_storage_guard.py",
    ]
    selected_bundles = set(args.bundle or BUNDLE_HELPERS)
    for bundle, helpers in BUNDLE_HELPERS.items():
        if bundle not in selected_bundles:
            continue
        src_dir = bundle_path(FINAL_ROOT, bundle) / "src"
        src_dir.mkdir(parents=True, exist_ok=True)
        for retired in RETIRED_GENERATED:
            retired_path = src_dir / retired
            if retired_path.exists():
                retired_path.unlink()
        selected = (
            [CANONICAL_GPU, CANONICAL_GPU_FRAME, *helpers]
            if bundle in (
                "01_p1_chern_dynamics",
                "02_wall_cft_windows",
                "03_h1_modular_response",
                "04_maxmix_operator_cft",
                "05_pure_tangent_stability",
                "07_log_gram_alpha_scan",
            )
            else [CANONICAL_GPU, CANONICAL_GPU_FRAME, *common, *helpers]
        )
        copied = {}
        for source in selected:
            if not source.exists():
                raise FileNotFoundError(source)
            destination = src_dir / source.name
            if source.resolve() != destination.resolve():
                shutil.copy2(source, destination)
            copied[source.name] = {
                "sha256": sha256(destination),
                "repository_source": repository_source(source),
            }
        (src_dir / "__init__.py").write_text(
            '"""Generated, self-contained Colab bundle source."""\n', encoding="utf-8"
        )
        copied["__init__.py"] = {
            "sha256": sha256(src_dir / "__init__.py"),
            "repository_source": "generated",
        }
        manifest = {
            "schema_version": 1,
            "canonical_gpu_sha256": sha256(CANONICAL_GPU),
            "files": copied,
            "warning": "Generated copies; edit repository sources and rerun sync_bundle_sources.py.",
        }
        (src_dir / "source_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"[synced] {bundle}: {len(copied)} Python files")

    if args.bundle:
        return

    # Standalone Colab projects are maintained runnable artifacts too.  Keep
    # their canonical engine and its occupied-frame dependency byte-identical.
    for destination in sorted(COLAB.glob("*/src/classA_U1FGTN_gpu.py")):
        shutil.copy2(CANONICAL_GPU, destination)
        shutil.copy2(CANONICAL_GPU_FRAME, destination.with_name("occupied_frame_gpu.py"))
        print(f"[synced] standalone Colab engine: {destination.parent.parent.name}")

    repository_root = ROOT.parents[1]
    external_engines = (
        repository_root / "00_WORKSPACE/EXTERNAL/OSG/osg_slope_vs_Ny/classA_U1FGTN_gpu.py",
        repository_root / "00_WORKSPACE/EXTERNAL/OSG/osg_entropy_vs_time_maxmix/classA_U1FGTN_gpu.py",
    )
    for destination in external_engines:
        if destination.exists():
            shutil.copy2(CANONICAL_GPU, destination)
            shutil.copy2(CANONICAL_GPU_FRAME, destination.with_name("occupied_frame_gpu.py"))
            print(f"[synced] external engine: {destination.parent.name}")


if __name__ == "__main__":
    main()
