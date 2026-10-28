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

from dyck_rnn.training.losses import kl_div
from dyck_rnn.data.load_model import load_model

# ==== Parameters ====
SEED = 1114
TEST_SIZE = 5_000
DKL_THRESH = 2e-2
DKL_BINWIDTH = 50
RUN_NAME = "DyckKM_k02_m04_linear_Stable_h24_mlp1"
RUN_DIR = Path("/Users/amah/Documents/GitHub/dyck-km-rnn/runs/")
FIG_DIR = Path("/Users/amah/Desktop/NeurReps_2026/")

# ==== Functions ====
def plot_model_probs(ax, probs):
    x = jnp.arange(5)
    y = jnp.median(probs, axis=0)[:-1]
    yerr = jnp.quantile(
        probs, 
        q = jnp.array([0.25, 0.75]),
        axis=0)[:,:-1]

    ax.plot(x, y)
    ax.fill_between(x, yerr[0], yerr[1], alpha=0.25)

    return ax

def plot_binned_dkl(ax, dkl, bin_width = 100):
    dkl = dkl.reshape((TEST_SIZE, -1, bin_width))

    x = jnp.arange(0, max_length, bin_width)
    y = jnp.nanmedian(dkl, axis=(0,2))
    yerr = jnp.nanquantile(
        dkl, 
        q = jnp.array([0.25, 0.75]), 
        axis=(0,2))

    ax.plot(x, y)
    ax.fill_between(x, yerr[0], yerr[1], alpha=0.25)

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
max_length = 100 * train_length
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

# ===== Run Through Models =====
opt_probs = get_opt_probs(states, k, m)
dkl_mat = jnp.zeros((
    n_runs, 
    TEST_SIZE,
    max_length))

for run in trange(n_runs):    
    # ==== Generate Data ====
    model = models[run]
    model_probs = jax.nn.softmax(jax.vmap(model)(sequences))

    dkl_mat = dkl_mat.at[run].set(
        kl_div(model, opt_probs, sequences, states, masks)
    )

early_dkl = jnp.nanmedian(dkl_mat[:,:,:train_length], axis=(1,2))
late_dkl = jnp.nanmedian(dkl_mat[:,:,-train_length:], axis=(1,2))

# ==== Pull Example Models ====
non_generalizers = jnp.where(late_dkl > DKL_THRESH)[0]
non_gen_id = 10
non_gen_mdl = models[non_gen_id]

generalizers = jnp.where(late_dkl <= DKL_THRESH)[0]
gen_id = 23
gen_mdl = models[gen_id]

example_state = (states % 2 == 1) & (states < 31)

gen_probs = jax.nn.softmax(jax.vmap(gen_mdl)(sequences))
non_gen_probs = jax.nn.softmax(jax.vmap(non_gen_mdl)(sequences))

# ==== Plot Main Fig ====
fig, axes = plt.subplots(2, 3, figsize=(8, 4.8))

# Early/Late Example Generalizer Performance
plot_model_probs(
    axes[0,0], 
    gen_probs[example_state & (trial_no < train_length)])
plot_model_probs(
    axes[1,0], 
    gen_probs[example_state & (trial_no > max_length - train_length)])

# Early/Late Example NonGeneralizer Performance
plot_model_probs(
    axes[0,1], 
    non_gen_probs[example_state & (trial_no < train_length)])
plot_model_probs(
    axes[1,1], 
    non_gen_probs[example_state & (trial_no > max_length - train_length)])

[ax.set_xlabel('Token') for ax in axes[:,:2].flatten()]
[ax.set_ylabel('Prob.') for ax in axes[:,:2].flatten()]

# Binned DKL for Example Generalizer and Nongeralizer
plot_binned_dkl(axes[0,2], dkl_mat[gen_id], bin_width = DKL_BINWIDTH)
plot_binned_dkl(axes[0,2], dkl_mat[non_gen_id], bin_width = DKL_BINWIDTH)

axes[0,2].set_xscale('log')
axes[0,2].set_yscale('log')

axes[0,2].set_xlabel('Token Position')
axes[0,2].set_ylabel('Positional D_KL')

# Median Early and Late DKL across models
axes[1,2].scatter(early_dkl[generalizers], late_dkl[generalizers])
axes[1,2].scatter(early_dkl[non_generalizers], late_dkl[non_generalizers])

axes[1,2].set_xscale('log')
axes[1,2].set_yscale('log')

axes[0,2].set_xlabel('Early D_KL')
axes[0,2].set_ylabel('Late D_KL')

fig.suptitle(gen_id)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig1_raw.pdf")
plt.close()

# ==== Plot Supfigs ====
# Plot generalizers
for i, id in enumerate(generalizers):
    probs = jax.nn.softmax(jax.vmap(models[id])(sequences))

    fig, axes = plt.subplots(1, 3, figsize=(8, 2.5))
    # Early/Late Example Generalizer Performance
    plot_model_probs(
        axes[0], 
        probs[example_state & (trial_no < train_length)])
    axes[0].set_xlabel('Token')
    axes[0].set_ylabel('Prob.')
    axes[0].set_title('Early')

    plot_model_probs(
        axes[1], 
        probs[example_state & (trial_no > max_length - train_length)])
    axes[1].set_xlabel('Token')
    axes[1].set_ylabel('Prob.')
    axes[0].set_title('Late')

    plot_binned_dkl(axes[2], dkl_mat[id], bin_width = DKL_BINWIDTH)
    axes[2].set_xscale('log')
    axes[2].set_yscale('log')

    axes[2].set_xlabel('Token Position')
    axes[2].set_ylabel('Positional D_KL')

    if id == gen_id:
        fig.suptitle(f'*Generalizer {i} ({id})*')
    else:
        fig.suptitle(f'Generalizer {i} ({id})')
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"supfigs/generalizer_{i}.pdf")
    plt.close()

# Plot non-generalizers
for i, id in enumerate(non_generalizers):
    probs = jax.nn.softmax(jax.vmap(models[id])(sequences))

    fig, axes = plt.subplots(1, 3, figsize=(8, 2.5))
    # Early/Late Example Generalizer Performance
    plot_model_probs(
        axes[0], 
        probs[example_state & (trial_no < train_length)])
    axes[0].set_xlabel('Token')
    axes[0].set_ylabel('Prob.')
    axes[0].set_title('Early')

    plot_model_probs(
        axes[1], 
        probs[example_state & (trial_no > max_length - train_length)])
    axes[1].set_xlabel('Token')
    axes[1].set_ylabel('Prob.')
    axes[0].set_title('Late')

    plot_binned_dkl(axes[2], dkl_mat[id], bin_width = DKL_BINWIDTH)
    axes[2].set_xscale('log')
    axes[2].set_yscale('log')

    axes[2].set_xlabel('Token Position')
    axes[2].set_ylabel('Positional D_KL')

    if id == non_gen_id:
        fig.suptitle(f'*Non-Generalizer {i} ({id})*')
    else:
        fig.suptitle(f'Non-Generalizer {i} ({id})')
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"supfigs/nongeneralizer_{i}.pdf")
    plt.close()

with open(model_dir / "generalizer_ids", "wb") as file:
    pkl.dump((generalizers, non_generalizers), file)

# %%
from scipy.stats import wilcoxon

threshold = 1e-2

ps = jnp.zeros(n_runs)
for i in range(n_runs):
    delta = dkl_mat[i,:,-train_length:] - threshold
    s, p = wilcoxon(delta, axis=None, nan_policy='omit', alternative='greater')

    ps = ps.at[i].set(p)

print(jnp.where(ps < 0.05)[0])
print(jnp.where(ps >= 0.05)[0])

# %%

from scipy.stats import wilcoxon

ps = jnp.zeros(n_runs)
for i in range(n_runs):
    s, p = wilcoxon(
        dkl_mat[i,:,-train_length:] - threshold,
        dkl_mat[i,:,:train_length] - threshold, 
        axis=None, nan_policy='omit', alternative='greater')

    ps = ps.at[i].set(p)

print(jnp.where(ps < 0.05)[0])
print(jnp.where(ps >= 0.05)[0])
