#%%
import jax
jax.config.update("jax_enable_x64", True)

import matplotlib as mpl
# Force Matplotlib to use TrueType fonts instead of Type 3
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

import json
import yaml
import jax.numpy as jnp
import jax.random as jr
from pathlib import Path
import matplotlib.pyplot as plt

from dyck_rnn.data.dyck_hmm import dyck_hmm
from dyck_rnn.data.load_model import load_model
from dyck_rnn.training.losses import kl_div
from dyck_rnn.data.opt_probs import get_opt_probs

#%%
run_dir = Path("/Users/amah/Documents/GitHub/dyck-km-rnn/runs/")
# runname = "DyckKM_k02_m04_linear_h24_mlp1_RegularizerSweep"
# runname = "DyckKM_k02_m04_linear_h24_mlp1_RegularizerSweep_fixed_u_transpose"
# runname = "DyckKM_k02_m04_linear_h24_mlp1_RegularizerSweep_remove_s_scaling2"
# runname = "DyckKM_k02_m04_linear_h24_mlp1_RegularizerSweep_rmv_first_sing_vec"
runname = "DyckKM_k08_m04_linear_h72_mlp1_InitScaleSweep"

validation_loss = []
lam = []
dkl = []

config_file = run_dir / runname / f"sweep_config.yaml"
with open(config_file, "rb") as file:
    configs = yaml.safe_load(file)

k = configs['data']['k']
m = configs['data']['m']
max_length = 30*(4 * m * (m + 4)) 

DyckHMM = dyck_hmm(k, m)

states, sequences = DyckHMM.batch_sample_sequence(
    batch_size = 500, 
    num_timesteps = max_length, 
    min_length = max_length, 
    key = jr.PRNGKey(-1))
states = states[:,:max_length]
sequences = sequences[:,:max_length]
masks = sequences != (2 * k + 1)
opt_probs = get_opt_probs(states, k, m)

for i in range(3):
    for j in range(10):
        print(i, j)

        eval_file = run_dir / runname / f"fit_{i:02}_{j:02}/metrics.json"
        with open(eval_file, "rb") as file:
            evals = json.load(file)

        validation_loss.append(evals['validation_loss_history'][-1])

        config_file = run_dir / runname / f"fit_{i:02}_{j:02}/config.yaml"
        with open(config_file, "rb") as file:
            configs = yaml.safe_load(file)

        lam.append(configs['model']['init_scale'])
        # lam.append(configs['optimizer']['lambda'])

        # model = load_model(runname + f"/fit_{i:02}_{j:02}", 
        #                    run_parent_dir = run_dir)

        # dkl_mat = kl_div(model, opt_probs, sequences, states, masks)
        # dkl.append(
        #     jnp.nanmedian(dkl_mat[:,-50:])
        # )

validation_loss = jnp.array(validation_loss)
lam = jnp.array(lam)
dkl = jnp.array(dkl)

plt.scatter(lam, validation_loss)
plt.axvline(lam[jnp.argmin(validation_loss)])
plt.title(lam[jnp.argmin(validation_loss)])

# %%
