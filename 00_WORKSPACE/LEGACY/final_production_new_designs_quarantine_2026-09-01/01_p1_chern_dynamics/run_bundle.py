#!/usr/bin/env python3
from __future__ import annotations

import os
import importlib.util
import sys
from pathlib import Path


BUNDLE_ROOT = Path(os.path.abspath(__file__)).parent
CAMPAIGN_ROOT = BUNDLE_ROOT.parent
SRC_DIR = BUNDLE_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(CAMPAIGN_ROOT) not in sys.path:
    sys.path.insert(0, str(CAMPAIGN_ROOT))

import p1_chern_runner as _runner
from p1_runtime_hardening import apply_p1_hardening, run_hardened_main


def _load_operational_drive_helper():
    module_name = "_p1_root_operational_drive_remote_commit"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    helper_path = CAMPAIGN_ROOT / "drive_remote_commit.py"
    spec = importlib.util.spec_from_file_location(module_name, helper_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load operational Drive helper: {helper_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


_operational_drive = _load_operational_drive_helper()
apply_p1_hardening(_runner, operational_drive=_operational_drive)


def main(argv: list[str] | None = None) -> int:
    return run_hardened_main(_runner, argv, bundle_root=BUNDLE_ROOT)


if __name__ == "__main__":
    raise SystemExit(main(["--bundle-root", str(BUNDLE_ROOT), *sys.argv[1:]]))
