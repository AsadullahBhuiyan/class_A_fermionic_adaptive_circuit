#!/bin/bash
#SBATCH --account=panhaining
#SBATCH --qos=panhaining
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=112G
#SBATCH --time=24:00:00

set -euo pipefail
module load pytorch/2.8.0
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONDONTWRITEBYTECODE=1
export CLASSA_ENGINE_DIR="$1/source/engine"
ROOT_RUN="$1"
STAGE="$2"
PYTHON_RUN=/blue/panhaining/ab2398.cornell/.conda/classA-b200/bin/python
nvidia-smi
case "$STAGE" in
  validate) exec "$PYTHON_RUN" -u "$ROOT_RUN/source/validate_campaign.py" --output-root "$ROOT_RUN" ;;
  benchmark) exec "$PYTHON_RUN" -u "$ROOT_RUN/source/benchmark.py" --output-root "$ROOT_RUN" ;;
  production) exec "$PYTHON_RUN" -u "$ROOT_RUN/source/run_campaign.py" --output-root "$ROOT_RUN" --worker "$SLURM_ARRAY_TASK_ID" ;;
  *) echo "Unknown GPU stage: $STAGE" >&2; exit 2 ;;
esac
