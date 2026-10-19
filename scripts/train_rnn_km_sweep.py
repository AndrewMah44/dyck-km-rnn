# Imports
import jax
jax.config.update("jax_enable_x64", True)

import yaml
import argparse
from time import time
from pathlib import Path
from copy import deepcopy
from itertools import product
from dyck_rnn.training.train import train_dyck_rnn

# ====== Load Configuration File ======
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=Path, required=True)
args = parser.parse_args()

with args.config.open("r") as f:
    sweep_config = yaml.safe_load(f)

task = sweep_config['experiment']['task']
n_runs = sweep_config['experiment']['n_runs']
seed = sweep_config['experiment']['seed']

# ====== Sweep over hidden size ======
for idx, (k,m) in enumerate(product(
    sweep_config['sweep']['k'],
    sweep_config['sweep']['m'])):
    
    for run in range(n_runs):
        # ====== Set Up Config File ======
        config = deepcopy(sweep_config)
        config['data']['k'] = k
        config['data']['m'] = m
        config['experiment']['seed'] = int(time())

        # ====== Set Up Paths ======
        run_name = task \
            + f'_k{config["data"]["k"]:02}' \
            + f'_m{config["data"]["m"]:02}' \
            + f'_{config["model"]["cell_type"]}' \
            + f'_h{config["model"]["hidden_size"]}' \
            + f'_mlp{config["model"]["readout_depth"]}'  \
            + f'/run_{run:02}'    

        train_dyck_rnn(run_name, config)
        print("\n")