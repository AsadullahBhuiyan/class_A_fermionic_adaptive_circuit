#!/bin/bash
# Install Python packages required for OSG GPU jobs.
# Called at the start of run_wrapper.sh before any Python script runs.
# pip skips packages that are already present in the container.
set -euo pipefail

pip install --quiet --user \
    tqdm \
    matplotlib \
    pandas \
    pfapack
