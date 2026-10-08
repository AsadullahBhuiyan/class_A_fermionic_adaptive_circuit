"""Regenerate the bundled manuscript figures without running a simulation."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

parser = argparse.ArgumentParser(__doc__)
parser.add_argument('--data-only', action='store_true')
parser.add_argument('--output-dir', type=Path)
args = parser.parse_args()
root = Path(__file__).resolve().parent
output = args.output_dir.resolve() if args.output_dir else root/'regenerated_analysis'
environment = os.environ.copy()
environment.pop('CLASSA_ENGINE_DIR', None)
command = [sys.executable, str(root/'source/analyze_campaign.py'),
           '--output-root', str(root), '--output-dir', str(output)]
if args.data_only:
    command.append('--data-only')
subprocess.run(command, env=environment, check=True)
