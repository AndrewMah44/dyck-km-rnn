#!/bin/bash
#SBATCH --job-name=dyck
#SBATCH --output=logs/dyck_%A_%a.out
#SBATCH --error=logs/dyck_%A_%a.err
#SBATCH --array=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=15G
#SBATCH --time=5:00:00
#SBATCH --mail-type=END,FAIL,ARRAY_TASKS
#SBATCH --mail-user=amah@flatironinstitute.org
#SBATCH --partition=genx

module --force purge
# module load python   # comment this out unless you know the correct module name
source ~/venvs/dyck_rnn/bin/activate
export JAX_PLATFORMS=cpu

python3 scripts/training_dyck_km_models.py \
  --config experiments/config_train_dyck_km_rnn.yaml \
  --r $SLURM_ARRAY_TASK_ID
