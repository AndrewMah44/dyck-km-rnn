# Imports
import jax
jax.config.update("jax_enable_x64", True)

import yaml
import argparse
import jax.numpy as jnp
from pathlib import Path
from copy import deepcopy
from dyck_rnn.training.train import train_dyck_rnn

# ====== Load Configuration File ======
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=Path, required=True)
args = parser.parse_args()

with args.config.open("r") as f:
    sweep_config = yaml.safe_load(f)

# ====== Set Up Run Name ======
k = sweep_config['data']['k']
m = sweep_config['data']['m']
task = sweep_config['experiment']['task']
n_runs = sweep_config['experiment']['n_runs']
seed = sweep_config['experiment']['seed']

if "comment" in sweep_config['experiment']:
    comment = "_" + sweep_config['experiment']['comment']
else:
    comment = ""

# Linear and LSTM hidden sizes are based on Hewitt. 
# GRU is to match number of linear params
if sweep_config['model']['cell_type'].lower() == "linear":
    sweep_config['model']['hidden_size'] = int(jnp.ceil(6 * m * jnp.log2(k)))

elif sweep_config['model']['cell_type'].lower() == "lstm":
    sweep_config['model']['hidden_size'] = int(jnp.ceil(3 * m * jnp.log2(k)))

elif sweep_config['model']['cell_type'].lower() == "gru":
    sweep_config['model']['hidden_size'] = int(jnp.floor(
        0.6 * jnp.ceil(6 * m * jnp.log2(k))))

sweep_name = task \
    + f"_k{sweep_config['data']['k']:02}" \
    + f"_m{sweep_config['data']['m']:02}" \
    + f"_{sweep_config['model']['cell_type'].lower()}" \
    + f"_h{sweep_config['model']['hidden_size']}" \
    + f"_mlp{sweep_config['model']['readout_depth']}" \
    + "_RegularizerSweep" \
    + comment

print(f"Fitting {sweep_name}...")

# ====== Set Up Paths ======
full_run_dir = Path("runs") / sweep_name
full_run_dir.mkdir(parents=True, exist_ok=True)

with (full_run_dir / "sweep_config.yaml").open("w") as f:
    yaml.safe_dump(sweep_config, f)

# ====== Sweep over initial scale ======
min_scale = sweep_config['sweep']['min_scale']
max_scale = sweep_config['sweep']['max_scale']
n_scales = sweep_config['sweep']['n_scales']

for run in range(3):
    init_scales = jnp.linspace(min_scale, max_scale, n_scales)

    final_validation_loss = jnp.zeros(n_scales)
    
    for i, init_scale in enumerate(init_scales):
        run_config = deepcopy(sweep_config)
        run_config['optimizer']['lambda'] = init_scale.item()

        run_name = f"fit_{run:02}_{i:02}"
        loss = train_dyck_rnn(run_name, run_config, run_parent=full_run_dir)

        final_validation_loss = final_validation_loss.at[i].set(loss)

    best_idx = jnp.argmin(final_validation_loss)

    min_scale = init_scales[max(best_idx-1, 0)]
    max_scale = init_scales[min(best_idx+1, n_scales-1)]