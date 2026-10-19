#%% ==== Imports ====
import jax
jax.config.update("jax_enable_x64", True)

import matplotlib as mpl
# Force Matplotlib to use TrueType fonts instead of Type 3
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

import json
import yaml
import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from pathlib import Path
from functools import partial
import matplotlib.pyplot as plt

from dyck_rnn.data.samplers import powerlaw
from dyck_rnn.data.dyck_hmm import dyck_hmm
from dyck_rnn.data.opt_probs import get_opt_probs
from dyck_rnn.utils.plot import plot_probs_by_condition

from dyck_rnn.training.losses import kl_div
from dyck_rnn.data.load_model import load_model

# ==== Data Set Up ====
# run_name = "DyckKM_k02_m04_linear_h24_mlp0"
# run_name = "DyckKM_k02_m04_linear_h24_mlp1"
# run_name = "DyckKM_k02_m04_linear_h24_mlp1_adamw"
run_name = "DyckKM_k02_m02_linear_Stable_h02_mlp1"

# run_name = "DyckKM_k02_m04_linear_h24_mlp1_svd0.14_fixed_u_tranpose"
# run_name = "DyckKM_k02_m04_linear_h24_mlp1_svd0.14_remove_s_scaling"
# run_name = "DyckKM_k02_m04_linear_h24_mlp1_svd0.14_remove_s_scaling2"
# run_name = "DyckKM_k02_m04_linear_h24_mlp1_svd0.05_remove_s_scaling2"
# run_name = "DyckKM_k02_m04_linear_h24_mlp1_svd0.02_remove_s_scaling2"
# run_name = "DyckKM_k02_m04_linear_h24_mlp1_svd0.20_rmv_first_sing_vec"
# run_name = "DyckKM_k02_m04_linear_h24_mlp1_svd5.00_rmv_first_sing_vec"
# run_name = "DyckKM_k02_m04_linear_h24_mlp1_svd5.00_rmv_first_sing_vec_longer_validation"
# run_name = "DyckKM_k02_m04_linear_h24_mlp1_svd8.14_rmv_first_sing_vec"

# run_name = "DyckKM_k02_m08_linear_h48_mlp1"
# run_name = "DyckKM_k04_m04_linear_h48_mlp1"
# run_name = "DyckKM_k02_m04_gru_h24_mlp0"
# run_name = "DyckKM_k02_m04_linear_h24_mlp1_svd0.29"
# run_name = "DyckKM_k02_m04_linear_h24_mlp1_svd0.50"
# run_name = "DyckKM_k02_m04_linear_h24_mlp1_svd0.29_fixed_u_tranpose"
# run_name = "DyckKM_k02_m04_linear_Stable_h24_mlp1"

eval_config = {
    'experiment': {
        'seed': 1114,
        'n_trials': 5
    },
    'data': {
        'test_size': 5_000,
    }
}

run_dir = Path("/Users/amah/Documents/GitHub/dyck-km-rnn/runs/")
model_dir = run_dir / run_name
with open(model_dir / "run_00/config.yaml", "r") as f:
    fit_config = yaml.safe_load(f)

n_runs = sum(1 for item in model_dir.iterdir() if item.is_dir())

# ==== Generate Test Dataset ====
master_key = jr.PRNGKey(eval_config['experiment']['seed'])
data_key, length_key, sample_key = jr.split(master_key,3)

k = fit_config['data']['k']
m = fit_config['data']['m']
max_length = 75*(4 * m * (m + 4)) 
n_states = int((k ** (m+1) - 1) / (k - 1))

DyckHMM = dyck_hmm(k, m)

lengths = powerlaw(
    length_key, 
    15, 
    100*(4 * m * (m + 4)), 
    0.9, 
    shape=(eval_config['data']['test_size'],)
)

states, sequences = DyckHMM.batch_sample_sequence(
    batch_size = eval_config['data']['test_size'], 
    num_timesteps = max_length, 
    min_length = lengths, 
    key = data_key)
states = states[:,:max_length]
sequences = sequences[:,:max_length]
masks = sequences != (2 * k + 1)

# ===== Load Models =====
models = []
for run in range(n_runs):
    model = load_model(
        run_name + f"/run_{run:02}", 
        run_parent_dir = run_dir)
    models.append(model)

# ===== Functions =====
def plot_model_perf(model_probs, opt_probs, figsize=(6, 5)):
    fig, axes = plt.subplots(k, 2, figsize=figsize)

    for i in range(k):
        # Plot internal states
        internal_state = (states % k == i) \
                & (states < (k ** m - 1) / (k - 1)) \
                & (states != 0)

        axes[i,0] = plot_probs_by_condition(
            axes[i,0], opt_probs, internal_state, color='b')
        axes[i,0] = plot_probs_by_condition(
            axes[i,0], model_probs, internal_state, linestyle='--')
        axes[i,0].set_title(f'Internal {i}')
        
        # Plot leaf states
        leaf_states = (states % k == i) \
            & (states >= (k ** m - 1) / (k - 1)) \
            & (states < (k ** (m+1) - 1) / (k - 1))

        axes[i,1] = plot_probs_by_condition(
            axes[i,1], opt_probs, leaf_states, color='b')   
        axes[i,1] = plot_probs_by_condition(
            axes[i,1], model_probs, leaf_states, linestyle='--')
        axes[i,1].set_title(f'Leaf {i}')

    fig.tight_layout()

    return fig, axes

def plot_d_kl(d_kl_mat):
    fig, axes = plt.subplots(2, 1)

    # D-KL histogram
    axes[0].hist(d_kl_mat.flatten())

    axes[0].set_yscale('log')

    axes[0].set_xlabel('D_KL')
    axes[0].set_ylabel('Count')

    # Median 
    axes[1].plot(
        jnp.arange(1, max_length+1), 
        jnp.nanmedian(d_kl_mat, axis=0))
    axes[1].axvline(4 * m * (m + 4))

    axes[1].set_xscale('log')
    axes[1].set_yscale('log')

    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('D_KL')
    return fig, axes

# ===== Run Through Models =====
opt_probs = get_opt_probs(states, k, m)

for run in range(n_runs):    
    # ==== Generate Data ====
    model = models[run]
    model_probs = jax.nn.softmax(jax.vmap(model)(sequences))

    d_kl_mat = kl_div(model, opt_probs, sequences, states, masks)

    # ==== Figure 1 - Checking Probability outputs ====
    fig, axes = plot_model_perf(model_probs, opt_probs)
    fig.suptitle(run)
    plt.show()

    # ==== Figure 2 - Checking Probability outputs ====
    fig, axes = plot_d_kl(d_kl_mat)
    fig.suptitle(run)

# %%
max_eigval = jnp.zeros(n_runs)
for run in range(n_runs):    
    # ==== Generate Data ====
    model = models[run]

    Wrec = model.rnn.rnn.W_rec
    max_eigval = max_eigval.at[run].set(
        jnp.max(jnp.abs(jnp.linalg.eigvals(Wrec))))

print(max_eigval)
# %%

