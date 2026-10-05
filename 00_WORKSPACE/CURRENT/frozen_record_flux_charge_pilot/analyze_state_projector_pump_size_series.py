#!/usr/bin/env python3
"""Analyze one verified N20 state-projector-pump size-series campaign."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import analyze_state_projector_pump_s100 as analysis  # noqa: E402
import run_state_projector_pump_size_series as campaign  # noqa: E402


# The original analysis is geometry-generic at fixed Nx=20. Rebind only its
# verifier so the new size-series identity and variable Ny frame dimensions are
# used. No entropy or central-charge observable is loaded or produced here.
analysis.campaign = campaign


def analyze(config: dict[str, Any], output_root: Path) -> dict[str, Any]:
    return analysis.analyze(config, Path(output_root).resolve())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=campaign.DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=campaign.DEFAULT_OUTPUT)
    args = parser.parse_args()
    config = campaign.load_config(args.config.resolve())
    campaign.validate_config(config)
    analyze(config, args.output_root.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
