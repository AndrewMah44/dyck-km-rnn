#%% 
# ==== Imports ====
import jax
jax.config.update("jax_enable_x64", True)

import matplotlib as mpl
# Force Matplotlib to use TrueType fonts instead of Type 3
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

from tqdm import trange
import yaml
import pickle as pkl
import jax.numpy as jnp
import jax.random as jr
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from dyck_rnn.data.samplers import powerlaw
from dyck_rnn.data.dyck_hmm import dyck_hmm
from dyck_rnn.data.load_model import load_model

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# ==== Parameters ====
SEED = 1114
TEST_SIZE = 5_000
DKL_THRESH = 2e-2
DKL_BINWIDTH = 50
RUN_NAME = "DyckKM_k02_m04_linear_Stable_h24_mlp1"
RUN_DIR = Path("/Users/amah/Documents/GitHub/dyck-km-rnn/runs/")
FIG_DIR = Path("/Users/amah/Desktop/NeurReps_2026/")

#%% ==== Functions ====
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


def sample_dataset(k, m, size, key):
    data_key, length_key = jr.split(key, 2)

    DyckHMM = dyck_hmm(k, m)

    lengths = powerlaw(
        length_key, 
        15, 
        110 * train_length, 
        0.9, 
        shape=(size,)
    )

    states, sequences = DyckHMM.batch_sample_sequence(
        batch_size = size, 
        num_timesteps = max_length, 
        min_length = lengths, 
        key = data_key)
    
    states = states[:,:max_length]
    sequences = sequences[:,:max_length]
    masks = sequences != (2 * k + 1)
    trial_no = jnp.cumsum(masks, axis=1)

    return states, sequences, masks, trial_no

def early_late_centroids(rnn, seq, mask, states, trial_nos):
    H = jax.vmap(rnn)(seq)[mask]
    pca = PCA(H.shape[1]).fit(H)

    early_centroids = jnp.zeros((n_states - 1, H.shape[1]))
    for state in range(n_states):
        early_centroids = early_centroids.at[state].set(
            jnp.mean(
                H[(states == state) & (trial_nos < train_length)], axis=0)
        )

    late_centroids = jnp.zeros((n_states - 1, H.shape[1]))
    for state in range(n_states):
        late_centroids = late_centroids.at[state].set(
            jnp.mean(
                H[(states == state) & (trial_nos > 99*train_length)], axis=0)
        )

    return early_centroids, late_centroids, pca

from dyck_rnn.training.losses import kl_div, kl_div2
from dyck_rnn.data.opt_probs import get_opt_probs

@jax.jit
def projected_performance(model, projection_vec, 
                          seqs, states, masks, opt_probs):
    # Pull model
    rnn = model.rnn
    readout = jax.vmap(jax.vmap(model.readout))

    # Orthogonal projection matrix
    projection_vec = projection_vec / jnp.linalg.norm(projection_vec)
 
    # ==== Simulate and get performance ====
    H = jax.vmap(rnn)(seqs)
    val = jnp.nanmedian(H[:,:128] @ drift_vector)

    Hproj = H - (H @ projection_vec - val) @ projection_vec.T
    mdl_probs_proj = readout(Hproj)
    dkl_proj = kl_div2(mdl_probs_proj, opt_probs, states, masks)

    return dkl_proj

#%%
# ==== Data Set Up ====
model_dir = RUN_DIR / RUN_NAME
with open(model_dir / "run_00/config.yaml", "r") as f:
    fit_config = yaml.safe_load(f)

n_runs = sum(1 for item in model_dir.iterdir() if item.is_dir())

# ==== Generate Test Dataset ====
master_key = jr.PRNGKey(SEED)
k = fit_config['data']['k']
m = fit_config['data']['m']
n_states = int((k ** (m+1) - 1) / (k - 1))

train_length = 4 * m * (m + 4)
max_length = 100 * train_length

decoder_key, centroid_key, proj_key = jr.split(master_key, 3)

# ===== Load Models =====
models = []
for run in range(n_runs):
    model = load_model(
        RUN_NAME + f"/run_{run:02}", 
        run_parent_dir = RUN_DIR)
    models.append(model)

with open(model_dir / "generalizer_ids", "rb") as file:
    generalizers, non_generalizers = pkl.load(file)

#%% RNN Wrec Eigenvalues

radius = jnp.zeros(n_runs)

for i, mdl in enumerate(models):

    Wrec = mdl.rnn.rnn.W_rec
    evals = jnp.linalg.eigvals(Wrec)
    
    radius = radius.at[i].set(jnp.max(jnp.abs(evals)))
    
#%% State Decoders

decoders = []
scores = jnp.zeros((n_runs, 3))

decoder_states, decoder_sequences, decoder_masks, decoder_trial_no = \
    sample_dataset(k, m, 500, decoder_key)

train_states = decoder_states[:400]
train_sequences = decoder_sequences[:400]
train_masks = decoder_masks[:400]

test_states = decoder_states[400:]
test_sequences = decoder_sequences[400:]
test_masks = decoder_masks[400:]
test_trial_no = decoder_trial_no[400:][test_masks]

y_train = train_states[train_masks]
y_test = test_states[test_masks]

early_test = test_trial_no < train_length
late_test = test_trial_no > 90 * train_length

for idx in range(n_runs):
    model = models[idx]
    rnn = model.rnn

    x_train = jax.vmap(rnn)(train_sequences)[train_masks]
    x_test = jax.vmap(rnn)(test_sequences)[test_masks]

    decoder = make_pipeline(
        StandardScaler(),
        LogisticRegression(fit_intercept=True,
                           max_iter = 1_000)
    ).fit(x_train, y_train)
    decoders.append(decoder)

    # Overall Score
    scores = scores.at[idx,0].set(decoder.score(x_test, y_test))

    # Early Score
    scores = scores.at[idx,1].set(
        decoder.score(x_test[early_test], y_test[early_test])
    )
    
    # Late Score
    scores = scores.at[idx,2].set(
        decoder.score(x_test[late_test], y_test[late_test])
    )
   
    print(f"Modle {idx} Overall Decoder Test Accuracy: {scores[idx,0]:0.4f}")


#%% Centroid Drift

centroid_states, centroid_sequences, centroid_masks, centroid_trial_no = \
    sample_dataset(k, m, 5_000, centroid_key)

S = centroid_states[centroid_masks]
T = centroid_trial_no[centroid_masks]

early_centroid_mat = jnp.zeros((
    n_runs, n_states-1, models[0].rnn.hidden_size
    ))
late_centroid_mat = jnp.zeros((
    n_runs, n_states-1, models[0].rnn.hidden_size
    ))
pcas = []

for i, model in enumerate(models):
    print(i)
    rnn = model.rnn

    early_centroids, late_centroids, pca = early_late_centroids(
        rnn, centroid_sequences, centroid_masks, S, T)

    early_centroid_mat = early_centroid_mat.at[i].set(
        early_centroids
    )

    late_centroid_mat = late_centroid_mat.at[i].set(
        late_centroids
    )

    pcas.append(pca)

#%%

proj_states, proj_sequences, proj_masks, proj_trial_no = \
    sample_dataset(k, m, 500, proj_key)

opt_probs = get_opt_probs(proj_states, k, m)

control_perf = jnp.zeros((n_runs, 2))
proj_perf = jnp.zeros((n_runs, 2))

for run in range(n_runs):  
    print(run)

    # Pull Model
    model = models[run]

    # Get Unperturbed Performance
    dkl_control = kl_div(
        model, opt_probs, proj_sequences, proj_states, proj_masks
    )

    control_perf = control_perf.at[run].set(
        jnp.array([
            jnp.nanmedian(dkl_control[:, :train_length]),
            jnp.nanmedian(dkl_control[:, -train_length:])
            ]
        )
    )

    # Get Projected Performance
    delta = late_centroid_mat[run] - early_centroid_mat[run]
    U, _, _ = jnp.linalg.svd(delta.T)
    drift_vector = jnp.expand_dims(U[:,0], -1)

    dkl_proj = projected_performance(
        model, 
        drift_vector, 
        proj_sequences, 
        proj_states, 
        proj_masks, 
        opt_probs
    )

    proj_perf = proj_perf.at[run].set(
        jnp.array([
            jnp.nanmedian(dkl_proj[:, :train_length]),
            jnp.nanmedian(dkl_proj[:, -train_length:])
            ]
        )
    )

    print(control_perf[run])
    print(proj_perf[run])


fig, axes = plt.subplots(1, 2, figsize=(8, 4))
axes[0].plot([1, 2], control_perf[non_generalizers].T)
axes[0].axhline(2e-2, color='k', linestyle='--')
axes[0].set_yscale('log')

axes[1].plot([1, 2], proj_perf[non_generalizers].T)
axes[1].axhline(2e-2, color='k', linestyle='--')

axes[1].set_yscale('log')
axes[1].set_ylim(axes[0].get_ylim())

fig.tight_layout()

#%%

key = jr.PRNGKey(1)
n_vecs = 250

null_proj = jnp.zeros((n_runs, n_vecs, 2))

# for run in range(n_runs):
for run in non_generalizers:
    print(run)

    model = models[run]
    keys = jr.split(key, n_vecs)

    for i in trange(n_vecs):
        key = keys[i]
        rand_vec = jr.normal(key = key, shape = (model.rnn.hidden_size, 1))
        rand_vec = rand_vec / jnp.linalg.norm(rand_vec)

        dkl_rand = projected_performance(
            model, 
            rand_vec, 
            proj_sequences, 
            proj_states, 
            proj_masks, 
            opt_probs
        )

        null_proj = null_proj.at[run, i, :].set(
            [jnp.nanmedian(dkl_rand[:, :train_length]),
             jnp.nanmedian(dkl_rand[:, -train_length:])]
        )

    _, key = jr.split(key, 2)


for run in non_generalizers:
    n = jnp.sum(null_proj[run, :, 1] <= proj_perf[run,1])

    p = (n+1) / (n_vecs + 1)

    plt.hist(null_proj[run, :, 1])
    plt.axvline(proj_perf[run,1])
    plt.axvline(control_perf[run,1], color='k')
    plt.title([run, p])
    plt.show()

#%%

y1 = jnp.vstack((control_perf[:,0], proj_perf[:,0]))
y2 = jnp.vstack((control_perf[:,1], proj_perf[:,1]))

plt.plot([1, 2], y1[:, non_generalizers])
plt.plot([3, 4], y2[:, non_generalizers])
plt.axhline(2e-2, color='k', linestyle='--')

plt.yscale('log')

# %%

fig, axes = plt.subplots(2, 3, figsize=(14, 6))

# Spectral Radii 
sort_idx = jnp.argsort(radius)
radius_sorted = radius[sort_idx]

gen_x = jnp.where(jnp.isin(sort_idx, generalizers))[0]
nongen_x = jnp.where(jnp.isin(sort_idx, non_generalizers))[0]

axes[0,0].scatter(gen_x, radius_sorted[gen_x], color='r')
axes[0,0].scatter(nongen_x, radius_sorted[nongen_x], color='b')

axes[0,0].set_xlabel('Sorted ID')
axes[0,0].set_ylabel('rho(A)')

# Decoder Performance
gen_x = [0, 1, 2]
gen_y = jnp.nanmedian(scores[generalizers], axis=0)
gen_err = jnp.nanquantile(
    scores[generalizers], 
    q = jnp.array([0.25, 0.75]), 
    axis = 0).T

nongen_x = [4, 5, 6]
nongen_y = jnp.nanmedian(scores[non_generalizers], axis=0)
nongen_err = jnp.nanquantile(
    scores[non_generalizers], 
    q = jnp.array([0.25, 0.75]), 
    axis = 0).T

axes[1,0].bar(gen_x, gen_y, label='Generalizers')
[axes[1,0].plot([x, x], [err[0], err[1]], color='k') 
 for x, err in zip(gen_x, gen_err)]

axes[1,0].bar(nongen_x, nongen_y, label='Non-generalizers')
[axes[1,0].plot([x, x], [err[0], err[1]], color='k') 
 for x, err in zip(nongen_x, nongen_err)]

axes[1,0].set_xticks(gen_x + nongen_x)
axes[1,0].set_xticklabels(
    ['All', 'Early', 'Late']*2)
axes[1,0].legend()

# Centroid Drift
gen_id = 23
non_gen_id = 10

gen_Xearly = pcas[gen_id].transform(early_centroid_mat[gen_id])
gen_Xlate = pcas[gen_id].transform(late_centroid_mat[gen_id])

nongen_Xearly = pcas[non_gen_id].transform(early_centroid_mat[non_gen_id])
nongen_Xlate = pcas[non_gen_id].transform(late_centroid_mat[non_gen_id])

axes[0,1].scatter(gen_Xearly[:,0], gen_Xearly[:,1], label='Early')
axes[0,1].scatter(gen_Xlate[:,0], gen_Xlate[:,1], label='Late')
axes[0,1].legend()

axes[0,1].set_title('Generalizer')
axes[0,1].set_xlabel('PC1')
axes[0,1].set_ylabel('PC2')

axes[1,1].scatter(nongen_Xearly[:,0], nongen_Xearly[:,1])
axes[1,1].scatter(nongen_Xlate[:,0], nongen_Xlate[:,1])

axes[1,1].set_title('Non-Generalizer')
axes[1,1].set_xlabel('PC1')
axes[1,1].set_ylabel('PC2')

fig.tight_layout()

# %%
