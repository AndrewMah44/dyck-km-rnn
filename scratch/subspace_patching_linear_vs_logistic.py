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
from dyck_rnn.data.samplers import powerlaw
from dyck_rnn.utils.get_depth import get_depth

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
        'test_size': 10_000,
    }
}

run_name = "DyckKM_k02_m04_linear_h24_mlp1"
# run_name = "DyckKM_k04_m04_linear_h64_mlp1"
# run_name = "DyckKM_k08_m04_linear_h72_mlp1"

# run_name = "DyckKM_k02_m04_gru_h12_mlp0"
run = 0

model_dir = "/Users/amah/Documents/GitHub/dyck-km-rnn/runs/"
run_dir = Path("/Users/amah/Documents/GitHub/dyck-km-rnn/runs/") \
    / run_name / f"run_{run:02}"

with open(run_dir / "config.yaml", "r") as file:
    model_config = yaml.safe_load(file)

model = load_model(
    run_name + f"/run_{run:02}", 
    run_parent_dir = model_dir)
rnn = model.rnn
readout = model.readout
readout_vmap = jax.vmap(readout)

# ==== Generate Test Dataset ====
master_key = jr.PRNGKey(eval_config['experiment']['seed'])
data_key, length_key, sample_key = jr.split(master_key,3)

k = model_config['data']['k']
m = model_config['data']['m']
max_length = 10 * 4 * m * (m + 4)

DyckHMM = dyck_hmm(k, m)

lengths = powerlaw(
    length_key, 
    15, 
    max_length, 
    model_config['data']['alpha'], 
    shape=(eval_config['data']['test_size'],)
)

states_mat, sequences_mat = DyckHMM.batch_sample_sequence(
    batch_size = eval_config['data']['test_size'], 
    num_timesteps = max_length, 
    min_length = lengths, 
    key = data_key)
mask = (sequences_mat < 2 * k)

states = states_mat[mask]
sequences = sequences_mat[mask]
depth = get_depth(states_mat, k, m)[mask]

# ==== Run model of data ====
model_probs = jax.nn.softmax(jax.vmap(model)(sequences_mat))
hidden_activations = jax.vmap(rnn)(sequences_mat)
H = hidden_activations[mask]

# ==== Depth Linear Decoder ====
from sklearn.linear_model import LinearRegression

linear_regression = LinearRegression(fit_intercept=True).fit(H, depth)
linear_score = linear_regression.score(H, depth)


h = H[states == 1][0]
target_centroid = jnp.mean(H[states == (k ** m - 1) / (k - 1)], axis=0)

v = linear_regression.coef_ / jnp.linalg.norm(linear_regression.coef_)
h_prime_linear = h + jnp.dot(target_centroid - h, v) * v

fig, axes = plt.subplots(3, 2, figsize=(6, 5))
axes[0,0].plot(jax.nn.softmax(readout(h)))
axes[0,0].set_title(f'linear regression \n(R^2 = {linear_score:0.3f})\noriginal')

axes[1,0].plot(jax.nn.softmax(readout(target_centroid)))
axes[1,0].set_title('target')

axes[2,0].plot(jax.nn.softmax(readout(h_prime_linear)))
axes[2,0].set_title('edited')

# ==== Depth Logistic Decoder ====
import scipy
from sklearn.linear_model import LogisticRegression

logistic_regression = LogisticRegression(fit_intercept=True).fit(H, depth)
logistic_score = logistic_regression.score(H, depth)

W = scipy.linalg.orth(logistic_regression.coef_.T)
h_prime_logistic = h + (target_centroid - h) @ W @ W.T

axes[0,1].plot(jax.nn.softmax(readout(h)))
axes[0,1].set_title(f'logistic regression \n(% Correct = {logistic_score:0.3f})\noriginal')

axes[1,1].plot(jax.nn.softmax(readout(target_centroid)))
axes[1,1].set_title('target')

axes[2,1].plot(jax.nn.softmax(readout(h_prime_logistic)))
axes[2,1].set_title('edited')

fig.supylabel('Prob')
fig.supxlabel('Token')
fig.suptitle('GRU')
fig.tight_layout()

# %%
# W = jnp.expand_dims(
#     linear_regression.coef_ / jnp.linalg.norm(linear_regression.coef_),
#     -1)

def subspace_patching(
        key, H, W, original_state, target_state, n_samples = 5_000):
    k1, k2 = jr.split(key, 2)

    # Sample original state 
    H_orig = jr.choice(
        k1, 
        H[states == original_state], 
        shape=(n_samples,))
    
    H_target = jr.choice(
        k2, 
        H[states == target_state], 
        shape=(n_samples,))
    
    H_prime = H_orig + (H_target - H_orig) @ W @ W.T

    return H_orig, H_target, H_prime

def plot_readout(H, ax=None, color='k'):
    if not ax:
        ax = plt.subplot()

    readout = jax.nn.softmax(readout_vmap(H))

    x = range(2*k + 2)
    y = readout.mean(0)
    yerr = readout.std(0)

    ax.plot(x, y, color=color)
    ax.fill_between(x, y+yerr, y-yerr, alpha=0.25, color=color)

    return readout

def plot_subspace_patching(
        key, H, W, original_state, target_state, n_samples = 5_000):
    
    H_orig, H_target, H_prime = subspace_patching(
        key, H, W, original_state, target_state, n_samples)
    
    fig, axes = plt.subplots(3, 1, figsize=(3, 5))
    readout_orig = plot_readout(H_orig, ax=axes[0])
    readout_target = plot_readout(H_target, ax=axes[1])
    readout_prime = plot_readout(H_prime, ax=axes[2])

    return fig, axes, readout_orig, readout_target, readout_prime

W = scipy.linalg.orth(logistic_regression.coef_.T)

for original_state in [1, 3, 7]:
    fig, axes, readout_orig, readout_target, readout_prime = \
        plot_subspace_patching(
            jr.PRNGKey(2), 
            H, 
            W, 
            original_state = original_state, 
            target_state = 15)
    fig.supxlabel('Tokens')
    fig.supylabel('Probs.')
    fig.suptitle(f'{original_state} -> 15')
    fig.tight_layout()

    kl_div_orig = jax.scipy.special.rel_entr(
        readout_orig,
        readout_prime
    ).sum(1)

    kl_div_target = jax.scipy.special.rel_entr(
        readout_target,
        readout_prime
    ).sum(1)

    print(jnp.median(kl_div_orig), jnp.median(kl_div_target))

# %%

for original_state in [2, 4, 8]:
    fig, axes, readout_orig, readout_target, readout_prime = \
        plot_subspace_patching(
            jr.PRNGKey(2), 
            H, 
            W, 
            original_state = original_state, 
            target_state = 30)
    fig.supxlabel('Tokens')
    fig.supylabel('Probs.')
    fig.suptitle(f'{original_state} -> 15')
    fig.tight_layout()

    kl_div_orig = jax.scipy.special.rel_entr(
        readout_orig,
        readout_prime
    ).sum(1)

    kl_div_target = jax.scipy.special.rel_entr(
        readout_target,
        readout_prime
    ).sum(1)

    print(jnp.median(kl_div_orig), jnp.median(kl_div_target))
# %%
key = jr.PRNGKey(1)
D = jnp.linalg.matrix_rank(W)

W_null = jr.normal(key, shape=(rnn.hidden_size, D))
W_null = scipy.linalg.orth(W_null)

for original_state in [1, 3, 7]:
    fig, axes, readout_orig, readout_target, readout_prime = \
        plot_subspace_patching(
            jr.PRNGKey(2), 
            H, 
            W_null, 
            original_state = original_state, 
            target_state = 15)
    fig.supxlabel('Tokens')
    fig.supylabel('Probs.')
    fig.suptitle(f'{original_state} -> 15')
    fig.tight_layout()

    kl_div_orig = jax.scipy.special.rel_entr(
        readout_orig,
        readout_prime
    ).sum(1)

    kl_div_target = jax.scipy.special.rel_entr(
        readout_target,
        readout_prime
    ).sum(1)

    print(jnp.median(kl_div_orig), jnp.median(kl_div_target))

# %%

# Remove depth subspace and see decodable info
