#!/bin/bash
#SBATCH --account=panhaining
#SBATCH --qos=panhaining
#SBATCH --partition=hpg-default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00

set -euo pipefail
module load pytorch/2.8.0
module load texlive/2023
export PATH="/blue/panhaining/ab2398.cornell/.conda/endpoint-pdf-tools/bin:$PATH"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONDONTWRITEBYTECODE=1
export CLASSA_ENGINE_DIR="$1/source/engine"
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/classA-endpoint-matplotlib"
ROOT_RUN="$1"
STAGE="$2"
PYTHON_RUN=/blue/panhaining/ab2398.cornell/.conda/classA-b200/bin/python
case "$STAGE" in
  freeze) exec "$PYTHON_RUN" -u "$ROOT_RUN/source/benchmark.py" --output-root "$ROOT_RUN" --freeze ;;
  analysis) exec "$PYTHON_RUN" -u "$ROOT_RUN/source/analyze_campaign.py" --output-root "$ROOT_RUN" ;;
  *) echo "Unknown CPU stage: $STAGE" >&2; exit 2 ;;
esac
