# Imports
import jax
# jax.config.update("jax_enable_x64", True)

import math
import yaml
import argparse
from pathlib import Path
from copy import deepcopy
from dyck_rnn.training.train import train_modular_cumsum_runn

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

# Specify the number of states
if 'n' in config['data']:
    print('Defaulting to n...')
    config['data']['max_length'] = 256

    # Chose linear size to match average time of return on uniform random walk
    if 'hidden_size' not in config['model']:
        if config['model']['cell_type'].lower() == 'linear':
            config['model']['hidden_size'] = config['data']['n']

        elif config['model']['cell_type'].lower() in ['lstm', 'gru']:
            config['model']['hidden_size'] = math.floor(
                config['data']['n'] // 2)

# Match to Dyck KM (recommended)
elif 'k' in config['data'] and 'm' in config['data']:
    k, m = config['data']['k'], config['data']['m']
    print(f'Making n to Dyck-({k}, {m})...')

    # Matching State Size, RNN size, etc.
    config['data']['n'] = math.floor((k**(m+1) - 1) / (k-1))
    config['data']['max_length'] = 4 * m * (m+4)

    # Linear and LSTM hidden sizes are based on Hewitt. 
    if 'hidden_size' not in config['model']:
        if config['model']['cell_type'].lower() == 'linear':
            config['model']['hidden_size'] = math.ceil(6 * m * math.log2(k))

        elif config['model']['cell_type'].lower() in ['lstm', 'gru']:
            config['model']['hidden_size'] = math.ceil(3 * m * math.log2(k))

if 'enforce_stable' in config['optimizer']:
    if config['optimizer']['enforce_stable']:
        stability = '_Stable'
    else:
        stability = ''
else:
    stability = ''
          
name = task \
    + f"_n{config['data']['n']:04}" \
    + f"_{config['model']['cell_type']}" \
    + stability \
    + f"_h{config['model']['hidden_size']:02}" \
    + f"_mlp{config['model']['readout_depth']}"

if 'lambda' in config['optimizer']:
     name += f"_{config['optimizer']['regularizer']}" +\
        f"{config['optimizer']['lambda']:0.2f}"

if 'comment' in config['experiment']:
        name += f"_{config['experiment']['comment']}"
        
run_name = name + f'/run_{run:02}'
run_config = deepcopy(config)
run_config['experiment']['seed'] = seed + run

train_modular_cumsum_runn(run_name, run_config)