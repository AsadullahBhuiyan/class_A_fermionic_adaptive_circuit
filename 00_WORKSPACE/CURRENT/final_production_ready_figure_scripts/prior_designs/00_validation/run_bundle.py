#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


BUNDLE_ROOT = Path(__file__).resolve().parent
SRC_DIR = BUNDLE_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from run_validation_suite import main


if __name__ == "__main__":
    raise SystemExit(main(["--bundle-root", str(BUNDLE_ROOT), *sys.argv[1:]]))
