#!/usr/bin/env python3
"""Restore reused figures from checksum-verified original assets; no simulation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

BUNDLE = Path(__file__).resolve().parents[1]
REPO = BUNDLE.parents[4]
ORIGINAL_REPO = Path('/home/abhuiyan/class_A_fermionic_adaptive_circuit')


def source_path(record: dict) -> Path:
    path = Path(record['path'])
    if path.is_relative_to(ORIGINAL_REPO):
        path = REPO / path.relative_to(ORIGINAL_REPO)
    if hashlib.sha256(path.read_bytes()).hexdigest() != record['sha256']:
        raise ValueError(f'Source checksum changed: {path}')
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--figure', help='One Figure_* stem; omit for all reused assets.')
    parser.add_argument('--output-dir', type=Path, default=BUNDLE)
    args = parser.parse_args()
    rows = json.loads((BUNDLE / 'data/existing_assets.json').read_text())
    rows = [row for row in rows if row.get('preserved', True)]
    selected = [row for row in rows if args.figure is None or row['stem'] == args.figure]
    if not selected:
        parser.error(f'No reused figure named {args.figure!r}')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for row in selected:
        if row['stem'] == 'Figure_02_alt_soft_and_hard_walls':
            pdf = BUNDLE / 'data/original_assets/ow_overlap_truncation_schematic.pdf'
            assert hashlib.sha256(pdf.read_bytes()).hexdigest() == row['pdf_source']['sha256']
        else:
            pdf = source_path(row['pdf_source'])
        shutil.copy2(pdf, args.output_dir / f"{row['stem']}.pdf")
        if isinstance(row['png_source'], dict):
            png = source_path(row['png_source'])
            shutil.copy2(png, args.output_dir / f"{row['stem']}.png")
        else:
            subprocess.run(['pdftoppm', '-singlefile', '-r', '300', '-png', str(pdf),
                            str(args.output_dir / row['stem'])], check=True)
        print(row['stem'])


if __name__ == '__main__':
    main()
