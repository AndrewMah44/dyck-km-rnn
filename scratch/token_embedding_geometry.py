#%% 
# ==== Imports ====
import jax
jax.config.update("jax_enable_x64", True)

import matplotlib as mpl
# Force Matplotlib to use TrueType fonts instead of Type 3
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

import yaml
import pickle as pkl
import jax.numpy as jnp
import jax.random as jr
from tqdm import trange
from pathlib import Path
import matplotlib.pyplot as plt

from dyck_rnn.data.samplers import powerlaw
from dyck_rnn.data.dyck_hmm import dyck_hmm
from dyck_rnn.data.opt_probs import get_opt_probs
from dyck_rnn.utils.get_depth import get_depth

from dyck_rnn.training.losses import kl_div
from dyck_rnn.data.load_model import load_model

# ==== Parameters ====
# RUN_NAME = "DyckKM_k02_m04_linear_h24_mlp1"
RUN_NAME = "DyckKM_k04_m04_linear_h48_mlp1"
# RUN_NAME = "DyckKM_k08_m04_linear_h72_mlp1"

SEED = 1114
TEST_SIZE = 5_000
DKL_THRESH = 2e-2
DKL_BINWIDTH = 50
RUN_DIR = Path("/Users/amah/Documents/GitHub/dyck-km-rnn/runs/")
FIG_DIR = Path.cwd() / "token_geometry"

FIG_DIR.mkdir(exist_ok=True)

# ==== Functions ====
def plot_model_probs(ax, probs, color='k', linestyle='-'):
    x = jnp.arange(2*k + 1)
    y = jnp.median(probs, axis=0)[:-1]
    yerr = jnp.quantile(
        probs, 
        q = jnp.array([0.25, 0.75]),
        axis=0)[:,:-1]

    ax.plot(x, y, color=color, linestyle=linestyle)
    ax.fill_between(x, yerr[0], yerr[1], color=color, alpha=0.25)

    return ax

# ==== Data Set Up ====
model_dir = RUN_DIR / RUN_NAME
with open(model_dir / "run_00/config.yaml", "r") as f:
    fit_config = yaml.safe_load(f)

n_runs = sum(1 for item in model_dir.iterdir() if item.is_dir())

# ==== Generate Test Dataset ====
master_key = jr.PRNGKey(SEED)
data_key, length_key, sample_key = jr.split(master_key,3)

k = fit_config['data']['k']
m = fit_config['data']['m']
train_length = 4 * m * (m + 4)
max_length = train_length
n_states = int((k ** (m+1) - 1) / (k - 1))

DyckHMM = dyck_hmm(k, m)

lengths = powerlaw(
    length_key, 
    15, 
    110 * train_length, 
    0.9, 
    shape=(TEST_SIZE,)
)

states, sequences = DyckHMM.batch_sample_sequence(
    batch_size = TEST_SIZE, 
    num_timesteps = max_length, 
    min_length = lengths, 
    key = data_key)
states = states[:,:max_length]
sequences = sequences[:,:max_length]
masks = sequences != (2 * k + 1)
trial_no = jnp.cumsum(masks, axis=1)

# ===== Load Models =====
models = []
for run in range(n_runs):
    model = load_model(
        RUN_NAME + f"/run_{run:02}", 
        run_parent_dir = RUN_DIR)
    models.append(model)

#%%
opt_probs = get_opt_probs(states, k, m)
depth = jax.vmap(get_depth, in_axes=[0, None, None])(states, k, m)

for run in range(n_runs):
    model = models[run]

    model_probs = jax.nn.softmax(jax.vmap(model)(sequences))

    fig, axes = plt.subplots(k, 2)

    axes[0,0].set_title('Interior')
    axes[0,1].set_title('Leaf')

    for i in range(k):
        state_cond = (states > 0) & (states < n_states) & (states % k == i)

        plot_model_probs(axes[i, 0], 
                        opt_probs[state_cond & (depth < m)])
        plot_model_probs(axes[i, 0], 
                        model_probs[state_cond & (depth < m)], 
                        color='r', linestyle='--')

        plot_model_probs(axes[i, 1], 
                        opt_probs[state_cond & (depth == m)])
        plot_model_probs(axes[i, 1], 
                        model_probs[state_cond & (depth == m)], 
                        color='r', linestyle='--')

    # fig.savefig(FIG_DIR / f"model{run}_performance.pdf")
    # plt.close()

#%%
fig, axes = plt.subplots(2, 5, figsize=(20, 10))

for run, ax in enumerate(axes.ravel()):
    model = models[run]

    A = model.rnn.rnn.W_rec
    B = model.rnn.rnn.W_u @ model.rnn.Win.weight.T

    Bnorm = B / jnp.linalg.norm(B, axis=0)

    S = Bnorm.T @ Bnorm

    im = ax.imshow(S[:-2,:-2])
    fig.colorbar(im, ax=ax)
    ax.set_title(run)

fig.tight_layout()
fig.savefig(FIG_DIR / f"embedding_cosine_sim.pdf")
plt.close()

#%%
fig, axes = plt.subplots(2, 5, figsize=(20, 10))

for run, ax in enumerate(axes.ravel()):
    model = models[run]

    s = jnp.zeros((k,k))
    for i in range(k):
        for j in range(k):
            d = A @ (B[:,i] - B[:,j])
            d = d / jnp.linalg.norm(d)

            e = B[:,k+i] - B[:,k+j]
            e = e / jnp.linalg.norm(e)

            s = s.at[i,j].set(d @ -e)

    im = ax.imshow(s)
    im.set_clim([0.9, 1])
    # fig.colorbar(im, ax=ax)
    ax.set_title(run)

cbar = fig.colorbar(im, ax=axes.ravel(), orientation='vertical', fraction=0.05, pad=0.04)

fig.savefig(FIG_DIR / f"embedding_cancellation.pdf")
plt.close()

# %%
npowers = 10
median_cosine_sim = jnp.zeros(npowers)

for n in range(npowers):
    fig, axes = plt.subplots(2, 5, figsize=(20, 10))

    Ahat = jnp.linalg.matrix_power(A, 2*n + 1)
    s = jnp.zeros((n_runs, k,k))

    for run, ax in enumerate(axes.ravel()):
        model = models[run]

        for i in range(k):
            for j in range(k):
                d = Ahat @ (B[:,i] - B[:,j])
                d = d / jnp.linalg.norm(d)

                e = B[:,k+i] - B[:,k+j]
                e = e / jnp.linalg.norm(e)

                s = s.at[run, i,j].set(d @ -e)

        im = ax.imshow(s[run])
        ax.set_title(run)

    fig.suptitle(2*n + 1)
    cbar = fig.colorbar(
        im, 
        ax=axes.ravel(), 
        orientation='vertical', 
        fraction=0.05, 
        pad=0.04)

    median_cosine_sim = median_cosine_sim.at[n].set(jnp.nanmean(s))

fig = plt.figure()
plt.plot(2*jnp.arange(npowers) + 1, median_cosine_sim, 'ko-')
plt.xticks(2*jnp.arange(npowers) + 1)
# %%    
