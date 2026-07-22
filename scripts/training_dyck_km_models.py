# Imports
import jax
jax.config.update("jax_enable_x64", True)

import math
import yaml
import argparse
from pathlib import Path
from copy import deepcopy
from dyck_rnn.training.train import train_dyck_rnn

# ====== Load Configuration File ======
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=Path, required=True)
parser.add_argument("--r", type=int, default=0)

args = parser.parse_args()

with args.config.open("r") as f:
    config = yaml.safe_load(f)

run = args.r

# ====== Set Up Run Name ======
k = config['data']['k']
m = config['data']['m']
task = config['experiment']['task']
n_runs = config['experiment']['n_runs']
seed = config['experiment']['seed']

# Linear and LSTM hidden sizes are based on Hewitt. 
# GRU is to match number of linear params
if 'hidden_size' not in config['model']:
    if config['model']['cell_type'].lower() == 'linear':
        config['model']['hidden_size'] = math.ceil(6 * m * math.log2(k))

    elif config['model']['cell_type'].lower() in ['lstm', 'gru']:
        config['model']['hidden_size'] = math.ceil(3 * m * math.log2(k))

name = task \
    + f"_k{k:02}_m{m:02}" \
    + f"_{config['model']['cell_type']}" \
    + f"_h{config['model']['hidden_size']}" \
    + f"_mlp{config['model']['readout_depth']}"

if 'lambda' in config['optimizer']:
     name += f"_{config['optimizer']['regularizer']}/"

if 'comment' in config['experiment']:
        name += f"_{config['experiment']['comment']}/"

        
run_name = name + f'run_{run:02}'
run_config = deepcopy(config)
run_config['experiment']['seed'] = seed + run

train_dyck_rnn(run_name, run_config)

