"""Test-path bootstrap for the physical high-level workspace layout."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
for path in (
    ROOT,
    ROOT / "src",
    ROOT / "00_WORKSPACE" / "CURRENT",
    ROOT / "00_WORKSPACE" / "COLAB",
    ROOT / "00_WORKSPACE" / "LARGE_RESULTS",
    ROOT / "00_WORKSPACE" / "LEGACY",
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
