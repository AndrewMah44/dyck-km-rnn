# Imports
import jax
# jax.config.update("jax_enable_x64", True)

import math
import yaml
import argparse
from pathlib import Path
from copy import deepcopy
from dyck_rnn.training.train import train_gaussian_pred_rnn

# ====== Load Configuration File ======
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=Path, required=True)
parser.add_argument("--r", type=int, default=0)

args = parser.parse_args()

with args.config.open("r") as f:
    config = yaml.safe_load(f)

run = args.r

# ====== Set Up Run Name ======
task = config['experiment']['task']
seed = config['experiment']['seed']

name = task \
    + f"_mu{config['data']['mu0']:0.2f}" \
    + f"_tau{config['data']['tau0']:0.2f}" \
    + f"_sigma{config['data']['sigma']:0.2f}"\
    + f"_{config['model']['cell_type']}" \
    + f"_h{config['model']['hidden_size']:02}" \
    + f"_mlp{config['model']['readout_depth']}"

if config['optimizer']['name'] == "adamw":
     name += f"_adamw{config['optimizer']['weight_decay']:0.2f}"

if 'comment' in config['experiment']:
        name += f"_{config['experiment']['comment']}"
        
run_name = name + f'/run_{run:02}'
run_config = deepcopy(config)
run_config['experiment']['seed'] = seed + run

train_gaussian_pred_rnn(run_name, run_config)