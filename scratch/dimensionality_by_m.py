#%%
import yaml
from pathlib import Path
from dyck_rnn.data.load_model import load_model

from dyck_rnn.data.samplers import powerlaw
from dyck_rnn.data.dyck_hmm import dyck_hmm

import jax
import jax.numpy as jnp
import jax.random as jr

eval_config = {
    'experiment': {
        'seed': 1114,
        'n_trials': 5
    },
    'data': {
        'test_size': 500,
    }
}

# %%

# ==== Generate Test Dataset ====
# eval_config['experiment']['seed']
def sample_data(eval_config, DyckHMM):
    master_key = jr.PRNGKey(eval_config['experiment']['seed'])
    data_key, length_key = jr.split(master_key, 2)

    max_length = 4 * m * (m + 4)

    lengths = powerlaw(
        length_key, 
        15, 
        10*(4 * m * (m + 4)), 
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

    return states, sequences, masks


#%%

k, m = 2, 4
n = 3
DyckHMM = dyck_hmm(k, m)

states, sequences, masks = sample_data(
    eval_config, 
    DyckHMM)

PR_04 = jnp.zeros(n)

for run in range(n):
    mdl_04 = load_model(
        run_name = f"DyckKM_k02_m04_linear_h96_mlp1/run_{run:02}",
        run_parent_dir = "/Users/amah/Documents/GitHub/dyck-km-rnn/runs/")

    H = jax.vmap(mdl_04.rnn)(sequences)[masks]
    Hc = H - H.mean(0)

    C = (Hc.T @ Hc) / (Hc.shape[0] - 1)
    evals = jnp.linalg.eigvalsh(C)
    PR_04 = PR_04.at[run].set((jnp.sum(evals) ** 2) / jnp.sum(evals ** 2))

#%%

k, m = 2, 8
DyckHMM = dyck_hmm(k, m)

states, sequences, masks = sample_data(
    eval_config, 
    DyckHMM)

PR_08 = jnp.zeros(5)
for run in range(5):
    mdl_08 = load_model(
        run_name = f"DyckKM_k02_m08_linear_h96_mlp1/run_{run:02}",
        run_parent_dir = "/Users/amah/Documents/GitHub/dyck-km-rnn/runs/")

    H = jax.vmap(mdl_08.rnn)(sequences)[masks]
    Hc = H - H.mean(0)

    C = (Hc.T @ Hc) / (Hc.shape[0] - 1)
    evals = jnp.linalg.eigvalsh(C)
    PR_08 = PR_08.at[run].set((jnp.sum(evals) ** 2) / jnp.sum(evals ** 2))

#%%

k, m = 2, 16
DyckHMM = dyck_hmm(k, m)

states, sequences, masks = sample_data(
    eval_config, 
    DyckHMM)

PR_16 = jnp.zeros(5)
for run in range(5):
    mdl_16 = load_model(
        run_name = f"DyckKM_k02_m16_linear_h96_mlp1/run_{run:02}",
        run_parent_dir = "/Users/amah/Documents/GitHub/dyck-km-rnn/runs/")

    H = jax.vmap(mdl_16.rnn)(sequences)[masks]
    Hc = H - H.mean(0)

    C = (Hc.T @ Hc) / (Hc.shape[0] - 1)
    evals = jnp.linalg.eigvalsh(C)
    PR_16 = PR_16.at[run].set(jnp.sum(evals) ** 2 / jnp.sum(evals ** 2))

#%%
import matplotlib.pyplot as plt

ms = jnp.array([4, 8, 16])
num_states = (2 ** (ms+1) - 1) / (k - 1)

fig, axes = plt.subplots(1, 2, figsize=(6, 3))
axes[0].plot(ms, num_states, 'ko-')
axes[0].set_yscale('log')
axes[0].set_ylabel('Number of States')

axes[1].errorbar(4,  PR_04.mean(0), PR_04.std(0), marker='o')
axes[1].errorbar(8,  PR_08.mean(0), PR_08.std(0), marker='o')
axes[1].errorbar(16, PR_16.mean(0), PR_16.std(0), marker='o')
axes[1].set_ylabel('Hidden Activation PR')

fig.supxlabel('k')
fig.tight_layout()

# axes[1].yscale('log')

#%%


# %%





