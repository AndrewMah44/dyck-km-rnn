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

from dyck_rnn.training.losses import kl_div
from dyck_rnn.data.load_model import load_model
from dyck_rnn.data.opt_probs import get_opt_probs
from dyck_rnn.training.losses import kl_div2
from dyck_rnn.analysis.representations import \
    get_centroids, get_centroid_drift

# ==== Data Set Up ====
run_name = "DyckKM_k02_m04_linear_h24_mlp1"
# run_name = "DyckKM_k02_m04_linear_h24_mlp1_adamw"

gru_run_name = "DyckKM_k02_m04_GRU_h24_mlp0"

eval_config = {
    'experiment': {
        'seed': 1114,
        'n_trials': 5
    },
    'data': {
        # 'test_size': 5_000,
        'test_size': 500,
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
train_length = 4 * m * (m + 4)

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
trial_no = jnp.cumsum(masks, axis=1)
opt_probs = get_opt_probs(states, k, m)

# ===== Load Models =====
models = []
validation_loss = jnp.zeros(n_runs)

for run in range(n_runs):
    model = load_model(
        run_name + f"/run_{run:02}", 
        run_parent_dir = run_dir)
    models.append(model)

    eval_file = run_dir / run_name / f"run_{run:02}/metrics.json"
    with open(eval_file, "rb") as file:
        evals = json.load(file)

    validation_loss = validation_loss.at[run].set(
        evals['validation_loss_history'][-1]
    )

#%% Plot Early vs Late Representations

fig, axes = plt.subplots(n_runs, 2, figsize=(7.5, 15))
for run in range(n_runs):
    rnn = models[run].rnn

    hidden_activity = jax.vmap(rnn)(sequences)

    centroids, pca = get_centroids(
        hidden_activity[masks], 
        states[masks], 
        n_states
    )

    early_centroids, late_centroids, delta = get_centroid_drift(
        hidden_activity[masks], 
        states[masks], 
        n_states,
        trial_no[masks] < 128,
        trial_no[masks] > 70*128
    )

    X = pca.transform(centroids)
    Xearly = pca.transform(early_centroids)
    Xlate = pca.transform(late_centroids)

    axes[run,0].scatter(X[:,0], X[:,1])

    axes[run,1].scatter(Xearly[:,0], Xearly[:,1])
    axes[run,1].scatter(Xlate[:,0], Xlate[:,1])
fig.tight_layout()

#%% Plot Drift Axis
fig, axes = plt.subplots(n_runs, 1, figsize=(7.5, 15))

for run in range(n_runs):
    model = models[run]
    rnn = models[run].rnn

    hidden_activity = jax.vmap(rnn)(sequences)

    _, pca = get_centroids(
        hidden_activity[masks], 
        states[masks], 
        n_states
    )

    early_centroids, late_centroids, delta = get_centroid_drift(
        hidden_activity[masks], 
        states[masks], 
        n_states,
        trial_no[masks] < 128,
        trial_no[masks] > 70*128
    )

    drift_vector = delta.mean(0) / jnp.linalg.norm(delta.mean(0))
    drift_axis = hidden_activity @ drift_vector

    x = jnp.arange(9600)
    y = jnp.nanmean(jnp.where(masks, drift_axis, jnp.nan), axis=0)
    yerr = jnp.nanstd(jnp.where(masks, drift_axis, jnp.nan), axis=0)

    axes[run].plot(x,y)
    axes[run].fill_between(x, y+yerr, y-yerr, alpha=0.25)

fig.tight_layout()

#%%
xs = jnp.zeros((n_runs, 100))
ys = jnp.zeros((n_runs, 100))

for run in range(5):
    model = models[run]
    rnn = models[run].rnn
    d_kl_mat = kl_div(model, opt_probs, sequences, states, masks)

    hidden_activity = jax.vmap(rnn)(sequences)

    _, pca = get_centroids(
        hidden_activity[masks], 
        states[masks], 
        n_states
    )

    early_centroids, late_centroids, delta = get_centroid_drift(
        hidden_activity[masks], 
        states[masks], 
        n_states,
        trial_no[masks] < 128,
        trial_no[masks] > 70*128
    )

    drift_vector = delta.mean(0) / jnp.linalg.norm(delta.mean(0))
    drift_axis = hidden_activity @ drift_vector

    bnds = jnp.array([0.25, 0.75])
    n_bins = 100
    masked_drift = jnp.where(masks, drift_axis, jnp.nan)

    xlims = jnp.nanquantile(masked_drift, jnp.array([0.01, 0.99]))
    x = jnp.linspace(xlims[0], xlims[1], n_bins)

    idx = jnp.digitize(masked_drift, x)

    y = jnp.array([jnp.nanmedian(d_kl_mat[idx == i]) for i in range(n_bins)])
    yerr = jnp.array([jnp.nanquantile(d_kl_mat[idx == i], q = bnds) 
                    for i in range(n_bins)])

    plt.plot(x, y)
    plt.fill_between(x, y-yerr[:,0], y+yerr[:,1], alpha=0.25)
    plt.yscale('log')
    plt.title(run)
    plt.show()

    xs = xs.at[run].set(x)
    ys = ys.at[run].set(y)


# %%

for run in range(n_runs):
    model = models[run]
    rnn = models[run].rnn
    d_kl_mat = kl_div(model, opt_probs, sequences, states, masks)

    hidden_activity = jax.vmap(rnn)(sequences)

    _, pca = get_centroids(
        hidden_activity[masks], 
        states[masks], 
        n_states
    )

    _, _, delta = get_centroid_drift(
        hidden_activity[masks], 
        states[masks], 
        n_states,
        trial_no[masks] < 128,
        trial_no[masks] > 70*128
    )

    drift_vector = delta.mean(0) / jnp.linalg.norm(delta.mean(0))
    drift_vector = jnp.expand_dims(drift_vector, 1)
    readout = jax.vmap(jax.vmap(model.readout))

    def projection_dkl(h, p, opt_probs, states, masks):
        p = p / jnp.linalg.norm(p)
        P = jnp.eye(24, dtype=h.dtype) - p @ p.T
        h_proj = h @ P
        mdl_probs = readout(h_proj)
        dkl = kl_div2(mdl_probs, opt_probs, states, masks)
        return dkl, h_proj, mdl_probs

    def f(carry, _):
        key = carry
        key, subkey = jr.split(key)
        p = jr.normal(subkey, (24, 1))
        dkl, _, _ = projection_dkl(hidden_activity, p, opt_probs, states, masks)
        return key, dkl

    import time

    key = jr.PRNGKey(1)

    start = time.time()
    _, dkl_mat = jax.lax.scan(f, key, None, 10)
    jax.block_until_ready(dkl_mat)
    end = time.time()
    print(end-start)

    start = time.time()
    _, dkl_mat = jax.lax.scan(f, key, None, 1_000)
    jax.block_until_ready(dkl_mat)
    end = time.time()
    print(end-start)


    random_dkl_ratio = jnp.nanmedian(dkl_mat[:,:,:128], axis=(1,2)) \
        / jnp.nanmedian(dkl_mat[:,:,-128:], axis=(1,2))


    # Drift Projection
    mdl_probs = readout(hidden_activity)
    dkl = kl_div2(mdl_probs, opt_probs, states, masks)
    dkl_ratio_orig = jnp.nanmedian(dkl[:,:128]) \
        / jnp.nanmedian(dkl[:,-128:])

    drift_dkl, _, _ = projection_dkl(
        hidden_activity, drift_vector, opt_probs, states, masks)
    dkl_ratio_drift = jnp.nanmedian(drift_dkl[:,:128]) \
        / jnp.nanmedian(drift_dkl[:,-128:])


    plt.hist(random_dkl_ratio)
    plt.axvline(dkl_ratio_orig, color='k')
    plt.axvline(dkl_ratio_drift, color='r')
    plt.title(run)
    plt.show()

# %%

diff = jnp.zeros(n_runs)
perf = jnp.zeros(n_runs)

for run in range(n_runs):

    model = models[run]
    rnn = models[run].rnn
    readout = jax.vmap(jax.vmap(model.readout))

    hidden_activity = jax.vmap(rnn)(sequences)

    _, pca = get_centroids(
        hidden_activity[masks], 
        states[masks], 
        n_states
    )

    early_centroids, late_centroids, delta = get_centroid_drift(
        hidden_activity[masks], 
        states[masks], 
        n_states,
        trial_no[masks] < 128,
        trial_no[masks] > 70*128
    )

    diff = diff.at[run].set(
        jnp.linalg.norm(early_centroids - late_centroids, axis=0).mean())


    mdl_probs = readout(hidden_activity)
    dkl = kl_div2(mdl_probs, opt_probs, states, masks)

    perf = perf.at[run].set(jnp.nanmedian(dkl))

# %%
# idx = jnp.array([0,1,2,3,6,7,8,9])
plt.scatter(diff, perf)
plt.xscale('log')
plt.yscale('log')

# %%

I = jnp.eye(24)

ubar = jnp.zeros(2*k+2).at[:2*k].set(1/(2*k))
# ubar = jnp.ones(2*k + 2) / (2*k+2)
d = jnp.zeros(n_runs)
drift_d = jnp.zeros((n_runs,))

for run in range(n_runs):
    print(run)

    model = models[run]
    rnn = models[run].rnn
    readout = jax.vmap(jax.vmap(model.readout))

    hidden_activity = jax.vmap(rnn)(sequences)

    _, pca = get_centroids(
        hidden_activity[masks], 
        states[masks], 
        n_states
    )

    early_centroids, late_centroids, delta = get_centroid_drift(
        hidden_activity[masks], 
        states[masks], 
        n_states,
        trial_no[masks] < 128,
        trial_no[masks] > 70*128
    )

    # Theoretical Drift Axis
    # A = jnp.linalg.pinv(I - model.rnn.rnn.W_rec)
    B = model.rnn.rnn.W_u @ model.rnn.Win.weight.T
    # d_theo = A @ B @ ubar
    d_theo = jnp.linalg.solve(I - model.rnn.rnn.W_rec, B @ ubar)
    d_theo = d_theo / jnp.linalg.norm(d_theo)

    # Empirical Drift Axis
    U, S, Vh = jnp.linalg.svd(delta.T)
    d_emp = U[:,0]
    d_emp2 = jnp.mean(delta, axis=0)

    d = d.at[run].set(jnp.abs(d_emp.T @ d_theo))
    drift_d = drift_d.at[run].set(S[0]**2 / jnp.sum(S**2))

plt.plot(d, 'ko-')
