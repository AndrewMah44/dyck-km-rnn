# Imports
import jax
jax.config.update("jax_enable_x64", True)

import yaml
import argparse
import jax.numpy as jnp
from pathlib import Path
from copy import deepcopy
from dyck_rnn.training.train import train_gaussian_pred_rnn

# ====== Load Configuration File ======
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=Path, required=True)
args = parser.parse_args()

with args.config.open("r") as f:
    sweep_config = yaml.safe_load(f)

# ====== Set Up Run Name ======
task = sweep_config['experiment']['task']
seed = sweep_config['experiment']['seed']
n_runs = sweep_config['experiment']['n_runs']

if "comment" in sweep_config['experiment']:
    comment = "_" + sweep_config['experiment']['comment']
else:
    comment = ""

sweep_name = task \
    + f"_mu{sweep_config['data']['mu0']:0.2f}" \
    + f"_tau{sweep_config['data']['tau0']:0.2f}" \
    + f"_sigma{sweep_config['data']['sigma']:0.2f}"\
    + f"_{sweep_config['model']['cell_type']}" \
    + f"_h{sweep_config['model']['hidden_size']:02}" \
    + f"_mlp{sweep_config['model']['readout_depth']}" \
    + "_AdamWSweep" \
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

for round in range(3):
    print(f"====== Round {round}: [{min_scale}, {max_scale}] ======")
    weight_decays = jnp.logspace(min_scale, max_scale, n_scales)

    final_validation_loss = jnp.zeros(n_scales)
    
    for i, weight_decay in enumerate(weight_decays):
        run_config = deepcopy(sweep_config)
        run_config['optimizer']['weight_decay'] = weight_decay.item()

        run_name = f"fit_{round:02}_{i:02}"
        loss = train_gaussian_pred_rnn(
            run_name, 
            run_config, 
            run_parent=full_run_dir)

        final_validation_loss = final_validation_loss.at[i].set(loss)

    best_idx = jnp.argmin(final_validation_loss)

    min_scale = weight_decays[max(best_idx-1, 0)]
    max_scale = weight_decays[min(best_idx+1, n_scales-1)]