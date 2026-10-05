"""Canonical flat layout for the independent redesigned Colab campaign."""

from __future__ import annotations

import json
from pathlib import Path


NEW_DESIGN_BUNDLES = (
    "01_p1_chern_dynamics",
    "02_wall_cft_windows",
    "03_h1_modular_response",
    "04_maxmix_operator_cft",
    "05_pure_tangent_stability",
    "07_log_gram_alpha_scan",
    "08_h1_endpoint_packet",
)


def bundle_path(root: Path | str, bundle: str) -> Path:
    if bundle not in NEW_DESIGN_BUNDLES:
        raise KeyError(f"unknown redesigned Colab bundle {bundle!r}")
    return Path(root) / bundle


def validate_bundle_layout(root: Path | str) -> tuple[str, ...]:
    """Validate the complete repository copy of the redesigned campaign."""
    root = Path(root)
    actual = tuple(
        sorted(
            path.name
            for path in root.iterdir()
            if path.is_dir() and (path / "production_config.json").is_file()
        )
    )
    if set(actual) != set(NEW_DESIGN_BUNDLES):
        missing = sorted(set(NEW_DESIGN_BUNDLES) - set(actual))
        unexpected = sorted(set(actual) - set(NEW_DESIGN_BUNDLES))
        raise RuntimeError(
            f"redesigned bundle mismatch: missing={missing}, unexpected={unexpected}"
        )
    for bundle in actual:
        runner = root / bundle / "run_bundle.py"
        if not runner.is_file():
            raise FileNotFoundError(f"missing runner: {runner}")

    index = json.loads((root / "bundle_index.json").read_text(encoding="utf-8"))
    if index.get("bundles") != list(NEW_DESIGN_BUNDLES):
        raise RuntimeError("bundle_index.json differs from bundle_layout.py")
    if set(index.get("standalone_contracts", {})) != set(NEW_DESIGN_BUNDLES):
        raise RuntimeError("every redesigned bundle must declare a standalone contract")
    return actual
