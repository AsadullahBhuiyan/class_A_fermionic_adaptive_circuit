#!/usr/bin/env python3
"""Read-only progress monitor for the N24x24 S50 RK4 campaign."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import monitor_parent_schrodinger_rk4_s50 as base
import run_parent_schrodinger_rk4_n24x24_s50 as campaign


# The established monitor is geometry-generic once its campaign module is supplied.
base.campaign = campaign


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=campaign.DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=campaign.DEFAULT_OUTPUT)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    report = base.inspect_campaign(args.config.resolve(), args.output_root.resolve())
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text, encoding="utf-8")
    print(text, end="")
    return 1 if report["invalid_or_failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
