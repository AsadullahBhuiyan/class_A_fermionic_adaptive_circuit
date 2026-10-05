#!/usr/bin/env python3
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import h1_packet_runner as _runner

from h1_operational import (
    install_operational_hardening,
    lexical_drive_resolve_context,
    operational_main,
)


install_operational_hardening(_runner)


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    drive_root = Path("/content/drive/MyDrive")
    if "--drive-root" in arguments:
        index = arguments.index("--drive-root")
        if index + 1 >= len(arguments):
            raise ValueError("--drive-root requires a value")
        drive_root = Path(arguments[index + 1])
    with lexical_drive_resolve_context(drive_root):
        handled = operational_main(
            arguments, runner=_runner, default_bundle_root=ROOT
        )
        if handled is not None:
            return handled
        return _runner.main(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
