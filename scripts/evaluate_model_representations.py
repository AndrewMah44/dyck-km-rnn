#%% ==== Imports ====
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
from sklearn.decomposition import PCA
from dyck_rnn.data.samplers import powerlaw
from dyck_rnn.utils.get_depth import get_depth

from dyck_rnn.data.dyck_hmm import dyck_hmm
from dyck_rnn.data.load_model import load_model
from dyck_rnn.analysis.representations import get_centroids, get_centroid_drift

# ==== Data Set Up ====
# run_name = "DyckKM_k02_m04_linear_h24_mlp1"
# run_name = "DyckKM_k02_m08_linear_h48_mlp1"
# run_name = "DyckKM_k04_m04_linear_h48_mlp1"
run_name = "DyckKM_k02_m04_linear_h24_mlp1_svd0.14_fixed_u_tranpose"

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
max_length = 25*(4 * m * (m + 4)) 
n_states = int((k ** (m+1) - 1) / (k - 1))

DyckHMM = dyck_hmm(k, m)

lengths = powerlaw(
    length_key, 
    15, 
    100*(4 * m * (m + 4)), 
    0.9, 
    shape=(eval_config['data']['test_size'],)
)

states_mat, sequences = DyckHMM.batch_sample_sequence(
    batch_size = eval_config['data']['test_size'], 
    num_timesteps = max_length, 
    min_length = lengths, 
    key = data_key)
states_mat = states_mat[:,:max_length]
sequences = sequences[:,:max_length]
mask = sequences != (2 * k + 1)

# ==== Load Models ====
models = []
for run in range(n_runs):
    model = load_model(
        run_name + f"/run_{run:02}", 
        run_parent_dir = run_dir)
    models.append(model)

def plot_recurrent_eigenvalues(model):
    Wrec = model.rnn.rnn.W_rec
    evals = jnp.linalg.eigvals(Wrec)

    eval_mag = jnp.abs(evals)
    unstable_evals = eval_mag > 1

    fig, ax = plt.subplots()

    circle = plt.Circle((0,0), 1, 
                    color='k', fill=False, linewidth=1, linestyle='--')

    ax.scatter(jnp.real(evals[~unstable_evals]), 
                jnp.imag(evals[~unstable_evals]))
    ax.scatter(jnp.real(evals[unstable_evals]), 
                jnp.imag(evals[unstable_evals]),
                color='r')

    ax.add_patch(circle)

    ax.set_aspect('equal')

    ax.set_xlim(-1.1,1.1)
    ax.set_ylim(-1.1, 1.1)
    
    ax.set_xlabel('Re(lam)')
    ax.set_ylabel('Im(lam)')

    fig.tight_layout()

    return fig, ax

def plot_centroids(centroids, pca):
    X = pca.transform(centroids)

    fig, axes = plt.subplots(2, 1)

    axes[0].plot(jnp.cumsum(pca.explained_variance_ratio_), 'ko-')
    axes[0].set_xlabel('n_pcs')
    axes[0].set_ylabel('Ratio Variance Explained')

    axes[1].scatter(X[:,0], X[:,1],
                    c = get_depth(jnp.arange(n_states), k, m))
    axes[1].set_xlabel('PC1')
    axes[1].set_ylabel('PC2')

    fig.tight_layout()

    return fig, axes

def plot_centroid_drift(hidden_activity, states, early, late, pca):
    early_centroids, late_centroids, delta = get_centroid_drift(
        hidden_activity, states, n_states, early, late)
    _, s_vals, _ = jnp.linalg.svd(delta)

    X1 = pca.transform(early_centroids)
    X2 = pca.transform(late_centroids)

    fig, axes = plt.subplots(2)

    axes[0].scatter(X1[:,0], X1[:,1], color='k', label='early')
    axes[0].scatter(X2[:,0], X2[:,1], color='r', label='late')
    axes[0].set_xlabel('PC1')
    axes[0].set_ylabel('PC2')

    axes[0].set_title('Mean Displacement: ' + \
                      f'{jnp.linalg.norm(delta, axis=1).mean(0):0.5f}')
    axes[0].legend()

    axes[1].plot(s_vals, 'ko-')
    axes[1].set_xlabel('singular value id')
    axes[1].set_ylabel('singular value')

    fig.tight_layout()

    return fig, axes

def plot_recurrent_eigenvalues(model):
    Wrec = model.rnn.rnn.W_rec
    evals = jnp.linalg.eigvals(Wrec)

    eval_mag = jnp.abs(evals)
    unstable_evals = eval_mag > 1

    fig, ax = plt.subplots()

    circle = plt.Circle((0,0), 1, 
                    color='k', fill=False, linewidth=1, linestyle='--')

    ax.scatter(jnp.real(evals[~unstable_evals]), 
                jnp.imag(evals[~unstable_evals]))
    ax.scatter(jnp.real(evals[unstable_evals]), 
                jnp.imag(evals[unstable_evals]),
                color='r')

    ax.add_patch(circle)

    ax.set_aspect('equal')

    ax.set_xlim(-1.1,1.1)
    ax.set_ylim(-1.1, 1.1)
    
    return fig, ax

#%%

states = states_mat[mask]
trial_no = jnp.cumsum(mask, axis=1)[mask]

for run in range(10):
    figdir = model_dir / f"run_{run:02}/figs"
    figdir.mkdir(exist_ok=True)

    model = models[run]
    rnn = model.rnn
    hidden_activity = jax.vmap(rnn)(sequences)[mask]
    centroids, centroid_pca = get_centroids(hidden_activity, states, n_states)

    # Plot eigenvalues
    eval_fig, _ = plot_recurrent_eigenvalues(model)
    eval_fig.suptitle(f"{run_name}_run{run:02}")
    eval_fig.tight_layout()

    eval_fig.savefig(figdir / "evals.svg")

    # Plot centroids 
    centroid_fig, ax = plot_centroids(centroids, centroid_pca)
    centroid_fig.suptitle(f"{run_name}_run{run:02}")
    centroid_fig.tight_layout()

    centroid_fig.savefig(figdir / "centroids.svg")

    # Plot centroid displacement
    drift_fig, ax = plot_centroid_drift(
        hidden_activity, states, 
        trial_no < 128, trial_no > 10*128,
        centroid_pca)
    drift_fig.suptitle(f"{run_name}_run{run:02}")
    drift_fig.tight_layout()

    drift_fig.savefig(figdir / "centroids_drift.svg")

    plt.close('all')

# %%
