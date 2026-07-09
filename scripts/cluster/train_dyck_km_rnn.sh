#!/bin/bash
#SBATCH --job-name=dyck
#SBATCH --output=logs/dyck_%A_%a.out
#SBATCH --error=logs/dyck_%A_%a.err
#SBATCH --array=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=15G
#SBATCH --time=5:00:00
#SBATCH --mail-type=END,FAIL,ARRAY_TASKS
#SBATCH --mail-user=amah@flatironinstitute.org
#SBATCH --gpus=1

module --force purge
# module load python   # comment this out unless you know the correct module name
source ~/venvs/dyck_rnn/bin/activate

echo "hostname: $(hostname)"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "SLURM_JOB_PARTITION=$SLURM_JOB_PARTITION"
echo "SLURM_GPUS=$SLURM_GPUS"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

which python
python --version
nvidia-smi

python -c "import jax; print('jax', jax.__version__); print(jax.devices())"

python3 scripts/training_dyck_km_models.py \
  --config experiments/config_train_dyck_km_rnn.yaml \
  --$SLURM_ARRAY_TASK_ID