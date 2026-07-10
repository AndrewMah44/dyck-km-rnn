#!/bin/bash
#SBATCH --job-name=dyck
#SBATCH --output=logs/dyck_%A_%a.out
#SBATCH --error=logs/dyck_%A_%a.err
#SBATCH --array=4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=15G
#SBATCH --time=5:00:00
#SBATCH --mail-type=END,FAIL,ARRAY_TASKS
#SBATCH --mail-user=amah@flatironinstitute.org

module --force purge

# Load whatever CUDA module your cluster recommends, if one is required.
# module load cuda/12.x

source ~/venvs/dyck_rnn/bin/activate

echo "Hostname: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-UNSET}"

echo "=== Slurm GPU allocation ==="
scontrol show job "$SLURM_JOB_ID" | grep -E "Gres=|TresPerNode=|AllocTRES=" || true

echo "=== NVIDIA GPU visibility ==="
nvidia-smi || true

echo "=== JAX installation ==="
python3 - <<'PY'
import os
import sys
import importlib.metadata as md

print("Python:", sys.version)
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))

for package in [
    "jax",
    "jaxlib",
    "jax-cuda12-plugin",
    "jax-cuda12-pjrt",
    "jax-cuda13-plugin",
    "jax-cuda13-pjrt",
]:
    try:
        print(f"{package}: {md.version(package)}")
    except md.PackageNotFoundError:
        pass

import jax
print("JAX backend:", jax.default_backend())
print("JAX devices:", jax.devices())
PY

python3 scripts/training_dyck_km_models.py \
  --config experiments/config_train_dyck_km_rnn.yaml \
  --r "$SLURM_ARRAY_TASK_ID"