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
        'test_size': 10_000,
    }
}

# ==== Load Run Configs ====
model_dir = "/Users/amah/Documents/GitHub/dyck-km-rnn/runs/"
run = 0

# ==== Load Linear Model ====
# run_name = "DyckKM_k02_m04_Linear_h24_mlp1"
run_name = "DyckKM_k02_m04_Linear_h24_mlp1_smaller_scale"
# run_name = "DyckKM_k02_m04_Linear_h24_mlp1_even_smaller_scale"
# run_name = "DyckKM_k02_m04_Linear_h24_mlp2"

run_dir = Path("/Users/amah/Documents/GitHub/dyck-km-rnn/runs/") \
    / run_name / f"run_{run:02}"

with open(run_dir / "config.yaml", "r") as file:
    model_config = yaml.safe_load(file)

model = load_model(run_name + f'/run_{run:02}', 
                run_parent_dir = model_dir)

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
model_probs = jax.nn.softmax(jax.vmap(model)(sequences))

# ==== Confidence Metrics ====
def get_close_conditional_prob(model_probs, seqs):
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
                sequences.shape)
    
    return confidence, confidence_mat


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


conditional_prob, conditional_prob_mat = get_close_conditional_prob(
    model_probs, sequences)

num_states = int((k ** (m+1) - 1) / (k - 1))
ages = jax.vmap(close_ages, in_axes=[0,0,None,None])(
    sequences, states, num_states, k)
ages = ages[(sequences >= k) & (sequences < 2*k)]

# ==== Plots ====
fig, axes = plt.subplots(3, 1, figsize=(4, 6))

# Confidence histogram
axes[0].hist(conditional_prob)

axes[0].set_xlabel('Correct Close Conditional Probability')
axes[0].set_ylabel('N (observations)')

axes[0].set_yscale('log')
axes[0].set_title(f"Median: {jnp.median(conditional_prob)}")

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

axes[1].errorbar(bins, y, yerr/jnp.sqrt(counts), marker='o', linestyle='None')
axes[1].set_ylim([0.95, 1.005])
axes[1].set_xlabel('Bracket age')
axes[1].set_ylabel('Confidence')

N_bins = 2**4
binned = conditional_prob_mat.reshape(
    conditional_prob_mat.shape[0], N_bins, -1)
trial_bin_means = jnp.nanmean(binned, axis=2)
counts = jnp.sum(~jnp.isnan(binned), axis=[0,2])

x = jnp.linspace(0, max_length, N_bins)
y = jnp.nanmean(trial_bin_means, axis=0)
yerr = jnp.nanstd(trial_bin_means, axis=0) / jnp.sqrt(counts)

axes[2].plot(x, y)
axes[2].fill_between(x, y+yerr, y-yerr, alpha=0.5)
axes[2].axvline(4 * m * (m + 4))

axes[2].set_xlabel('Sequence position')
axes[2].set_ylabel('Confidence')

fig.tight_layout()
fig.savefig('/Users/amah/Desktop/fig1.pdf')

#%%

h = jax.vmap(model.rnn)(sequences)
h_norm = jnp.linalg.norm(h, axis=2)

plt.hist(h_norm[conditional_prob_mat < 0.1], alpha=0.5, density=True)
plt.hist(h_norm[conditional_prob_mat > 0.1], alpha=0.5, density=True)

# %%

import matplotlib.pyplot as plt

ev = jnp.linalg.eigvals(model.rnn.rnn.W_rec)

# Create figure and axis
fig, ax = plt.subplots()

circle = plt.Circle((0, 0), 1, color='k', fill=False, linewidth=2)

# Add the circle patch to the axis
ax.add_patch(circle)
ax.scatter(jnp.real(ev), jnp.imag(ev))
# Crucial: Keep the aspect ratio equal so it stays a circle
ax.set_aspect('equal')
ax.set_xlim([-1.1, 1.1])
ax.set_ylim([-1.1, 1.1])

# %% 

# Depth 0
x = jnp.arange(2*k + 2)
y = jnp.mean(model_probs[states == 0], axis=0)
yerr = jnp.std(model_probs[states == 0], axis=0)
plt.plot(x, y)
plt.fill_between(x, y+yerr, y-yerr, alpha=0.25)
plt.title('State 0')

# Depth 1
fig, axes = plt.subplots(1, 2, figsize=(6, 2))
for i in range(1, 3):
    y = jnp.mean(model_probs[states == i], axis=0)
    yerr = jnp.std(model_probs[states == i], axis=0)
    axes[i-1].plot(x, y)
    axes[i-1].fill_between(x, y+yerr, y-yerr, alpha=0.25)
    axes[i-1].set_title(f'State {i}' )
fig.tight_layout()

# Depth 2
fig, axes = plt.subplots(1, 4, figsize=(6, 2))
for i, ax in zip(range(3, 7), axes):
    y = jnp.mean(model_probs[states == i], axis=0)
    yerr = jnp.std(model_probs[states == i], axis=0)
    ax.plot(x, y)
    ax.fill_between(x, y+yerr, y-yerr, alpha=0.25)
    ax.set_title(f'State {i}' )

fig.tight_layout()

# Depth 3
fig, axes = plt.subplots(1, 8, figsize=(12, 2))
for i, ax in zip(range(7, 15), axes):
    y = jnp.mean(model_probs[states == i], axis=0)
    yerr = jnp.std(model_probs[states == i], axis=0)
    ax.plot(x, y)
    ax.fill_between(x, y+yerr, y-yerr, alpha=0.25)
    ax.set_title(f'State {i}' )

fig.tight_layout()

# Depth 4
fig, axes = plt.subplots(2, 8, figsize=(12, 5))
for i, ax in zip(range(15, 31), axes.flatten()):
    y = jnp.mean(model_probs[states == i], axis=0)
    yerr = jnp.std(model_probs[states == i], axis=0)
    ax.plot(x, y)
    ax.fill_between(x, y+yerr, y-yerr, alpha=0.25)
    ax.set_title(f'State {i}' )

fig.tight_layout()



# %%
bins = jnp.arange(31) - 0.5
plt.hist(states[conditional_prob_mat < 0.1], bins=bins)
plt.yscale('log')
# %%
bads = model_probs[conditional_prob_mat < 0.1]

templates = jnp.array([
    [0.25, 0.25, 0.5, 0,   0, 0],
    [0.25, 0.25, 0,   0.5, 0, 0]
])

kl1 = jax.scipy.special.rel_entr(templates[0], bads).sum(1)
kl2 = jax.scipy.special.rel_entr(templates[1], bads).sum(1)

plt.scatter(kl1, kl2)

# %%
x = jnp.arange(2*k + 2)
y = bads[(kl1 > 0.5) & (kl2 > 0.5)].mean(0)
yerr = bads[(kl1 > 0.5) & (kl2 > 0.5)].std(0)

plt.plot(x, y)
plt.fill_between(x, y+yerr, y-yerr, alpha=0.25)

# %%

x = jnp.arange(2*k + 2)
y = bads[(kl1 < 0.5)].mean(0)
yerr = bads[(kl1 < 0.5)].std(0)

plt.plot(x, y)
plt.fill_between(x, y+yerr, y-yerr, alpha=0.25)
# %%

x = jnp.arange(2*k + 2)
y = bads[(kl2 < 0.5)].mean(0)
yerr = bads[(kl2 < 0.5)].std(0)

plt.plot(x, y)
plt.fill_between(x, y+yerr, y-yerr, alpha=0.25)

# %%
