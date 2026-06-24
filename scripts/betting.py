#%%
# ==== Imports ====
import jax
jax.config.update("jax_enable_x64", True)

import yaml
import argparse
import pickle as pkl
import jax.numpy as jnp
import jax.random as jr
from pathlib import Path
from itertools import product
import matplotlib.pyplot as plt
from jax.scipy.special import kl_div
from dyck_rnn.data.samplers import powerlaw

from dyck_rnn.data.dyck_hmm import dyck_hmm
from dyck_rnn.data.load_model import load_model

eval_config = {
    'experiment': {
        'seed': 324,
        'n_trials': 5
    },
    'run': {
        'task': 'DyckKM',
        'k': [2, 4, 8, 16, 32],
        'm': [2, 4, 8],
        'cell_type': 'Linear',
        'readout_depth': 1,
        'hidden_size': [64],
        'n_runs': 5
    },
    'data': {
        'test_size': 5_000,
        'num_timesteps': 300
    }
}

#%%
k = 2
m = 4
run = 0

# ==== Load Run Configs ====
model_dir = "/Users/amah/Documents/GitHub/dyck-km-rnn/runs/"

# Linear Model
# linear_run_name = "DyckKM_k02_m04_Linear_h24_mlp1"
# linear_run_name = f"DyckKM_k02_m04_Linear_h24_mlp1_smaller_scale/run_{run:02}"
linear_run_name = "DyckKM_k02_m04_linear_h24_mlp1/run_00"

linear_model = load_model(linear_run_name, 
                run_parent_dir = model_dir)

gru_run_name = "DyckKM_k02_m04_lstm_h12_mlp0"
gru_model = load_model(gru_run_name + f'/run_{run:02}', 
                run_parent_dir = model_dir)

with open(model_dir + linear_run_name +  "/config.yaml", "r") as file:
    model_config = yaml.safe_load(file)

# ==== Generate Test Dataset ====
master_key = jr.PRNGKey(eval_config['experiment']['seed'])
data_key, length_key, sample_key = jr.split(master_key,3)

k = model_config['data']['k']
m = model_config['data']['m']
max_length = 10*(4 * m * (m + 4))

DyckHMM = dyck_hmm(k, m)

lengths = powerlaw(
    length_key, 
    15, 
    max_length, 
    model_config['data']['alpha'], 
    shape=(eval_config['data']['test_size'],)
)

states, sequences = DyckHMM.batch_sample_sequence(
    batch_size = eval_config['data']['test_size'], 
    num_timesteps = max_length, 
    min_length = lengths, 
    key = data_key)
mask = (sequences < 2 * DyckHMM.k + 1)

# ==== Run model of data
linear_probs = jax.nn.softmax(jax.vmap(linear_model)(sequences))
gru_probs = jax.nn.softmax(jax.vmap(gru_model)(sequences))

# %%

from jax.scipy.special import rel_entr
alpha = 0.05

p1 = linear_probs.reshape(-1, 2*k+2)
p2 = gru_probs.reshape(-1, 2*k+2)

a = rel_entr(linear_probs, gru_probs).sum(-1)

test = jnp.where(sequences < 2*k, a, 0).sum(1)

#%%
L = jnp.mean(rel_entr(linear_probs, gru_probs).sum(-1)[sequences < 2*k])
-jnp.log(alpha) / L


# %%

