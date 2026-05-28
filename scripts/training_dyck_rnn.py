# Imports
import jax
jax.config.update("jax_enable_x64", True)

import yaml
import argparse
from pathlib import Path
from copy import deepcopy
from dyck_rnn.training.train import train_dyck_rnn

# ====== Load Configuration File ======
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=Path, required=True)
args = parser.parse_args()

with args.config.open("r") as f:
    config = yaml.safe_load(f)

# ====== Set Up Run Name ======
task = config['experiment']['task']
n_runs = config['experiment']['n_runs']
seed = config['experiment']['seed']

if 'comment' in config['experiment']:
    name = task \
        + f'_k{config["data"]["k"]:02}' \
        + f'_m{config["data"]["m"]:02}' \
        + f'_{config["model"]["cell_type"]}' \
        + f'_h{config["model"]["hidden_size"]}' \
        + f'_mlp{config["model"]["readout_depth"]}' \
        + f'_{config['experiment']['comment']}/' 
else:
    name = task \
        + f'_k{config["data"]["k"]:02}' \
        + f'_m{config["data"]["m"]:02}' \
        + f'_{config["model"]["cell_type"]}' \
        + f'_h{config["model"]["hidden_size"]}' \
        + f'_mlp{config["model"]["readout_depth"]}/' \

for run in range(n_runs):
    run_name = name + f'run_{run:02}'
    run_config = deepcopy(config)
    run_config['experiment']['seed'] = seed + run

    train_dyck_rnn(run_name, run_config)

