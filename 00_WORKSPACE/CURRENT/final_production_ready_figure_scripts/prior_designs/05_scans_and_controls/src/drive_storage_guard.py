#!/usr/bin/env python3
"""Fast, non-destructive storage guard for the production Drive output tree."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DECIMAL_GB = 1_000_000_000
DEFAULT_WORKING_LIMIT_BYTES = 12 * DECIMAL_GB
DEFAULT_ABSOLUTE_EDGE_BYTES = 14 * DECIMAL_GB
DEFAULT_REQUIRED_HEADROOM_BYTES = DECIMAL_GB


def tree_bytes(path: Path | str) -> int:
    """Return bytes currently occupied by regular files below *path*."""
    root = Path(path)
    if not root.exists():
        return 0
    return sum(
        entry.stat().st_size
        for entry in root.rglob("*")
        if entry.is_file()
    )


def storage_status(
    output_root: Path | str,
    *,
    working_limit_bytes: int = DEFAULT_WORKING_LIMIT_BYTES,
    absolute_edge_bytes: int = DEFAULT_ABSOLUTE_EDGE_BYTES,
    required_headroom_bytes: int = DEFAULT_REQUIRED_HEADROOM_BYTES,
) -> dict[str, Any]:
    """Measure active campaign output and decide whether another run may start."""
    root = Path(output_root)
    used = tree_bytes(root)
    children = {
        child.name: tree_bytes(child)
        for child in sorted(root.iterdir(), key=lambda item: item.name)
        if child.is_dir()
    } if root.exists() else {}
    projected = used + int(required_headroom_bytes)
    clear = used < int(absolute_edge_bytes) and projected <= int(working_limit_bytes)
    return {
        "schema_version": 1,
        "output_root": str(root),
        "used_bytes": used,
        "used_gb": used / DECIMAL_GB,
        "required_headroom_bytes": int(required_headroom_bytes),
        "required_headroom_gb": int(required_headroom_bytes) / DECIMAL_GB,
        "projected_bytes": projected,
        "projected_gb": projected / DECIMAL_GB,
        "working_limit_bytes": int(working_limit_bytes),
        "working_limit_gb": int(working_limit_bytes) / DECIMAL_GB,
        "absolute_edge_bytes": int(absolute_edge_bytes),
        "absolute_edge_gb": int(absolute_edge_bytes) / DECIMAL_GB,
        "clear_to_run": clear,
        "bundle_bytes": children,
        "bundle_gb": {name: size / DECIMAL_GB for name, size in children.items()},
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Refuse a production launch when active Drive outputs are too large"
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--working-limit-gb", type=float, default=12.0)
    parser.add_argument("--absolute-edge-gb", type=float, default=14.0)
    parser.add_argument("--required-headroom-gb", type=float, default=1.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.working_limit_gb <= 0 or args.absolute_edge_gb <= 0:
        raise ValueError("storage limits must be positive")
    if args.required_headroom_gb < 0:
        raise ValueError("required headroom cannot be negative")
    if args.working_limit_gb > args.absolute_edge_gb:
        raise ValueError("working limit cannot exceed the absolute edge")
    status = storage_status(
        args.output_root,
        working_limit_bytes=round(args.working_limit_gb * DECIMAL_GB),
        absolute_edge_bytes=round(args.absolute_edge_gb * DECIMAL_GB),
        required_headroom_bytes=round(args.required_headroom_gb * DECIMAL_GB),
    )
    print(json.dumps(status, indent=2, sort_keys=True))
    if status["clear_to_run"]:
        return 0
    print(
        "STORAGE GUARD: move verified output archives to durable storage before "
        "starting another shard. Gate JSON files and unconsumed H3 parents must remain.",
        flush=True,
    )
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
