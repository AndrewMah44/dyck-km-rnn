#!/usr/bin/bash
#SBATCH --job-name=rnn_init
#SBATCH --output=logs/lam_%A_%a.out
#SBATCH --error=logs/lam_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --gpus-per-task=1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --mem=15G
#SBATCH --time=10:00:00
#SBATCH --mail-type=END,FAIL,ARRAY_TASKS
#SBATCH --mail-user=amah@flatironinstitute.org

module --force purge

source ~/venvs/dyck_rnn/bin/activate

export LD_LIBRARY_PATH="$(
    find "$VIRTUAL_ENV/lib" \
        -path '*/site-packages/nvidia/*/lib' \
        -type d \
        -printf '%p:'
)"

echo "Host: $(hostname)"
echo "Python: $(which python)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

srun --cpu-bind=cores python - <<'PY'
import jax

print("JAX backend:", jax.default_backend())
print("JAX devices:", jax.devices())

if jax.default_backend() != "gpu":
    raise RuntimeError("JAX did not initialize the GPU backend")
PY

srun --cpu-bind=cores python scripts/training_rnn_regularizer_sweep.py \
    --config experiments/config_train_dyck_km_rnn_regularizer_sweep.yaml \
