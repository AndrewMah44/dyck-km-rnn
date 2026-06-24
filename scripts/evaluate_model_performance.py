#%%
# ==== Imports ====
import jax
jax.config.update("jax_enable_x64", True)

import matplotlib as mpl
# Force Matplotlib to use TrueType fonts instead of Type 3
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

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
        'seed': 1114,
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
        'test_size': 1_000,
    }
}


def close_ages(tokens, states, num_states, k):
    """
    tokens: shape [T], integer token ids
    states: shape [T], integer state id at each token position
            (state before/at processing token t, depending on your convention)
    num_states: total number of possible stack states
    open_token_max: tokens < open_token_max are opens, >= open_token_max are closes
    """

    T = tokens.shape[0]
    is_close = (tokens >= k) & (tokens < 2*k)
    t_idx = jnp.arange(T, dtype=jnp.int32)

    # last_seen[s] = most recent position where state s was observed
    init_last_seen = -jnp.ones((num_states,), dtype=jnp.int32)

    def step(last_seen, x):
        t, s = x
        prev_t = last_seen[s]
        last_seen = last_seen.at[s].set(t)
        return last_seen, prev_t

    _, prev_t = jax.lax.scan(step, init_last_seen, (t_idx, states))
    age = t_idx - prev_t - 1

    return jnp.where(is_close, age, 0)

def get_close_conditional_prob(model_probs, seqs, k):
    p = model_probs.reshape(-1, 2*k + 2)
    s = seqs.flatten()

    close_bool = (s >= k) & (s < 2*k)
    close_idx = jnp.where(close_bool)[0]

    true_close_probs = jnp.take_along_axis(
        p[close_idx-1], 
        jnp.expand_dims(s[close_idx], 1),
        1).flatten()

    total_close_probs = p[close_idx - 1, k:2*k].sum(1)

    confidence = true_close_probs / total_close_probs
    confidence_mat = jnp.full_like(
        s, jnp.nan, dtype=jnp.float64).at[close_idx].set(
            confidence).reshape(
                seqs.shape)
    
    return confidence, confidence_mat
#%%
run_name = "DyckKM_k02_m04_linear_h24_mlp1"
run = 0

model_dir = "/Users/amah/Documents/GitHub/dyck-km-rnn/runs/old2"
run_dir = Path("/Users/amah/Documents/GitHub/dyck-km-rnn/runs/") \
    / run_name / f"run_{run:02}"

# ==== Load Linear Model ====
with open(run_dir / "config.yaml", "r") as file:
    model_config = yaml.safe_load(file)

model = load_model(
    run_name + f"/run_{run:02}", 
    run_parent_dir = model_dir)

# ==== Generate Test Dataset ====
master_key = jr.PRNGKey(eval_config['experiment']['seed'])
data_key, length_key, sample_key = jr.split(master_key,3)

k = model_config['data']['k']
m = model_config['data']['m']
max_length = 50*(4 * m * (m + 4))

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

# ==== Run model of data
model_probs = jax.nn.softmax(jax.vmap(model)(sequences))

# ==== Confidence Metrics ====
conditional_prob, conditional_prob_mat = get_close_conditional_prob(
    model_probs, sequences, k)

num_states = int((k ** (m+1) - 1) / (k - 1))
ages = jax.vmap(close_ages, in_axes=[0,0,None,None])(
    sequences, states, num_states, k)
ages = ages[(sequences >= k) & (sequences < 2*k)]

# ==== Plots ====
fig, ax = plt.subplots(3, 1)

# Confidence histogram
ax[0].hist(conditional_prob)

ax[0].set_xlabel('Conditional Prob.')
ax[0].set_ylabel('N (observations)')

ax[0].set_yscale('log')
ax[0].set_title(f"Mean: {jnp.mean(conditional_prob)}")

# Confidence as a function of age
n_bins = 25
bins = jnp.linspace(0, jnp.max(ages), n_bins)
idx = jnp.digitize(ages, bins)

y = jnp.array(
    [jnp.nanmean(conditional_prob[idx==i]) for i in range(n_bins)])
yerr = jnp.array(
    [jnp.nanstd(conditional_prob[idx==i]) for i in range(n_bins)])
counts = jnp.array(
    [jnp.sum(idx==i) for i in range(n_bins)])

ax[1].errorbar(bins, y, yerr/jnp.sqrt(counts), marker='o', linestyle='None')
ax[1].set_xlabel('Bracket age')
ax[1].set_ylabel('Conditional Prob.')

N_bins = 2**6
binned = conditional_prob_mat.reshape(
    conditional_prob_mat.shape[0], N_bins, -1)
trial_bin_means = jnp.nanmean(binned, axis=2)
counts = jnp.sum(~jnp.isnan(binned), axis=[0,2])

x = jnp.linspace(0, max_length, N_bins)
y = jnp.nanmean(trial_bin_means, axis=0)
yerr = jnp.nanstd(trial_bin_means, axis=0) / jnp.sqrt(counts)

ax[2].plot(x, y)
ax[2].fill_between(x, y+yerr, y-yerr, alpha=0.25)
ax[2].axvline(4 * m * (m + 4))

ax[2].set_xlabel('Sequence position')
ax[2].set_ylabel('Conditional Prob.')


#%%
fig, axes = plt.subplots(3,3, figsize=(10,6))

model_performance("DyckKM_k02_m04_linear_h24_mlp1", axes[:,0], run = 0)
# model_performance("DyckKM_k02_m04_lstm_h12_mlp0", axes[:,2], run = 0)

yl0 = axes[0,1].get_ylim()[1]
yl1 = axes[1,0].get_ylim()[0]
yl2 = axes[2,0].get_ylim()[0]

[ax.set_ylim([0, yl0]) for ax in axes[0,:]]
[ax.set_ylim([yl1, 1.025]) for ax in axes[1,:]]
[ax.set_ylim([yl2, 1.01]) for ax in axes[2,:]]
# [ax.set_xlim([0, 1]) for ax in axes[0,:]]

fig.tight_layout()
# fig.tight_layout()
# fig.savefig('/Users/amah/Desktop/fig1.pdf')

#%%

fig, axes = plt.subplots(3,1, figsize=(3,6))

model_performance("DyckKM_k04_m04_linear_h64_mlp1", axes, run = 0)
# %%
