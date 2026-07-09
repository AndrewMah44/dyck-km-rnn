#!/bin/bash
#SBATCH -p gpu
#SBATCH --job-name=dyck
#SBATCH --output=logs/dyck_%A_%a.out
#SBATCH --error=logs/dyck_%A_%a.err
#SBATCH --array=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=15G                 # Adjust memory as needed
#SBATCH --time=5:00:00            # Max runtime
#SBATCH --mail-type=END,FAIL,ARRAY_TASKS
#SBATCH --mail-user=amah@flatironinstitute.org  # Or your actual email
#SBATCH --partition=genx

# Load your environment
module --force purge
module load python
source ~/venvs/dyck_rnn/bin/activate

# Run the script with the array index
python3 scripts/training_dyck_km_models.py --config experiments/config_train_dyck_km_rnn.yaml --r $SLURM_ARRAY_TASK_ID
