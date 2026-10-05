"""Canonical layout for the preserved prior-design Colab campaign."""

from __future__ import annotations

import json
from pathlib import Path


PRIOR_DESIGN_BUNDLES = (
    "00_validation",
    "01_bulk_width_gate",
    "01_p1_existing_completion",
    "02_pure_wall_master",
    "03_chirality_replay",
    "04_maxmix_master",
    "05_scans_and_controls",
    "06_b1_controller_frame",
)
ALL_BUNDLES = PRIOR_DESIGN_BUNDLES


def bundle_group(bundle: str) -> str:
    if bundle in PRIOR_DESIGN_BUNDLES:
        return "prior_designs"
    raise KeyError(f"unknown prior-design Colab bundle {bundle!r}")


def bundle_relative_path(bundle: str) -> Path:
    return Path(bundle_group(bundle)) / bundle


def bundle_path(root: Path | str, bundle: str) -> Path:
    return Path(root) / bundle_relative_path(bundle)


def validate_bundle_layout(root: Path | str) -> tuple[str, ...]:
    root = Path(root)
    group_root = root / "prior_designs"
    actual = tuple(
        sorted(
            path.name
            for path in group_root.iterdir()
            if path.is_dir() and (path / "production_config.json").is_file()
        )
    )
    if set(actual) != set(PRIOR_DESIGN_BUNDLES):
        missing = sorted(set(PRIOR_DESIGN_BUNDLES) - set(actual))
        unexpected = sorted(set(actual) - set(PRIOR_DESIGN_BUNDLES))
        raise RuntimeError(
            f"prior-design bundle mismatch: missing={missing}, unexpected={unexpected}"
        )
    index = json.loads((root / "bundle_index.json").read_text(encoding="utf-8"))
    if index.get("bundles") != list(PRIOR_DESIGN_BUNDLES):
        raise RuntimeError("bundle_index.json differs from bundle_layout.py")
    return actual
