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
        'test_size': 5_000,
    }
}

# ==== Generate Test Dataset ====
master_key = jr.PRNGKey(eval_config['experiment']['seed'])
data_key, length_key, sample_key = jr.split(master_key,3)

k = 2
m = 4
max_length = 75*(4 * m * (m + 4)) 

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
mask = sequences != (2 * k + 1)

# %%
run_name = "DyckKM_k02_m04_linear_h24_mlp1"
n = 10

model_dir = "/Users/amah/Documents/GitHub/dyck-km-rnn/runs/"

models = []
for run in range(n):
    model = load_model(
        run_name + f"/run_{run:02}", 
        run_parent_dir = model_dir)
    models.append(model)

#%% RNN Wrec Eigenvalues

fig, axes = plt.subplots(5, 2, figsize=(6,10))

for idx, ax in enumerate(axes.flatten()):
    Wrec = models[idx].rnn.rnn.W_rec
    evals = jnp.linalg.eigvals(Wrec)
    
    eval_mag = jnp.abs(evals)
    unstable_evals = eval_mag > 1

    circle = plt.Circle((0,0), 1, color='k', fill=False, linewidth=1,
                        linestyle='--')
    
    ax.scatter(jnp.real(evals[~unstable_evals]), 
               jnp.imag(evals[~unstable_evals]))
    ax.scatter(jnp.real(evals[unstable_evals]), 
               jnp.imag(evals[unstable_evals]),
               color='r')

    ax.add_patch(circle)

    ax.set_aspect('equal')

    ax.set_xlim(-1.1,1.1)
    ax.set_ylim(-1.1, 1.1)

    if idx in [0, 2, 6]:
        ax.set_title(f"{idx}, Generalizer")
    else:
        ax.set_title(idx)

fig.tight_layout()

#%%

# %% Train Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

decoders = []
scores = jnp.zeros(10)
for idx in range(10):
    model = models[idx]
    rnn = model.rnn

    H = jax.vmap(rnn)(sequences)[mask]
    s = states[mask]

    key = jr.PRNGKey(1)
    train_idx = jr.uniform(key, shape=(H.shape[0])) < 0.8
    test_idx = ~train_idx

    x_train = H[train_idx]
    y_train = s[train_idx]

    x_test = H[test_idx]
    y_test = s[test_idx]

    decoder = make_pipeline(
        StandardScaler(),
        LogisticRegression(fit_intercept=True)
    ).fit(x_train, y_train)

    score = decoder.score(x_test, y_test)
    print(f"Modle {idx} Decoder Test Accuracy: {score:0.4f}")
    
    decoders.append(decoder)
    scores = scores.at[idx].set(score)

plt.bar(range(10), scores)
plt.axhline(1/31, color='r', linestyle='--')
plt.xticks(range(10))
plt.ylabel('State Decoding Accuracy')
plt.xlabel('Model')


# %% PCA Dimensionality and 2D Centroid Projection
from sklearn.decomposition import PCA

pca_fig, pca_axes = plt.subplots(2, 5, figsize=(10, 6))
pca_axes = pca_axes.flatten()

for idx in range(n):
    model = models[idx]
    rnn = model.rnn

    H = jax.vmap(rnn)(sequences)[mask]
    s = states[mask]

    centroids = jnp.array([
        jnp.mean(H[s == i], axis=0) for i in range(31)
    ])

    pca = PCA(24).fit(centroids)
    X = pca.transform(centroids)

    pca_axes[idx].plot(jnp.cumsum(pca.explained_variance_ratio_), 'ko-')
    pca_axes[idx].axhline(0.9)

    fig, axes = plt.subplots(figsize=(5,5))
    axes.scatter(X[:,0], X[:,1], c = get_depth(jnp.arange(31), k, m))
    for i in range(31):
        axes.annotate(
            str(i),
            (X[i,0], X[i,1]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
        )

    if idx in [0, 2, 6]:
        pca_axes[idx].set_title(f"{idx}, Generalizer")
        axes.set_title(f"{idx}, Generalizer")
    elif idx in [4, 5]:
        pca_axes[idx].set_title(f"{idx}, Unstable")
        axes.set_title(f"{idx}, Unstable")
    else:
        pca_axes[idx].set_title(idx)
        axes.set_title(idx)

pca_fig.tight_layout()

# %%

for idx in range(n):
    model = models[idx]
    rnn = model.rnn

    H = jax.vmap(rnn)(sequences)

    early_idx = 4 * m * (m+4)
    late_idx = 74 * early_idx

    early_mask = mask[:,:early_idx]
    early_h = H[:,:early_idx,:][early_mask]
    early_states = states[:,:early_idx][early_mask]
    early_centroid = jnp.array([
        jnp.mean(early_h[early_states == i], axis=0) for i in range(31)
    ])

    late_mask = mask[:,late_idx:]
    late_h = H[:, late_idx:, :][late_mask]
    late_states = states[:,late_idx:][late_mask]
    late_centroid = jnp.array([
        jnp.mean(late_h[late_states == i], axis=0) for i in range(31)
    ])

    pca = PCA(24).fit(H[mask])

    early_X = pca.transform(early_centroid)
    late_X = pca.transform(late_centroid)

    fig, axes = plt.subplots()
    axes.scatter(early_X[:,0], early_X[:,1], label='early', zorder=2)
    axes.scatter(late_X[:,0], late_X[:,1], label='late', zorder=2)

    axes.legend()

    axes.set_xlabel('PC1')
    axes.set_ylabel('PC2')

    for i in range(31):
        axes.plot([early_X[i,0], late_X[i,0]],
                [early_X[i,1], late_X[i,1]],
                color='k',
                zorder=1)

    if idx in [0, 2, 6]:
        axes.set_title(f"{idx}, Generalizer")
    else:
        axes.set_title(idx)


# %%

for idx in range(10):

    model = models[idx]
    rnn = model.rnn

    H = jax.vmap(rnn)(sequences)

    early_idx = 4 * m * (m+4)
    late_idx = 74 * early_idx

    early_mask = mask[:,:early_idx]
    early_h = H[:,:early_idx,:][early_mask]
    early_states = states[:,:early_idx][early_mask]
    early_centroid = jnp.array([
        jnp.mean(early_h[early_states == i], axis=0) for i in range(31)
    ])

    late_mask = mask[:,late_idx:]
    late_h = H[:, late_idx:, :][late_mask]
    late_states = states[:,late_idx:][late_mask]
    late_centroid = jnp.array([
        jnp.mean(late_h[late_states == i], axis=0) for i in range(31)
    ])

    delta = late_centroid - early_centroid

    fig, axes = plt.subplots()
    axes.plot(delta.T)

    axes.set_xlabel('Dimension')
    axes.set_ylabel('Displacement (Late - Early)')

    if idx in [0, 2, 6]:
        axes.set_title(f"{idx}, Generalizer")
    else:
        axes.set_title(idx)
    
    plt.show()

# %%

position_encoding_perf = jnp.zeros(10)
angle_from_position_index = jnp.zeros(10)

key = jr.PRNGKey(1)

for idx in range(10):
    model = models[idx]
    rnn = model.rnn

    H = jax.vmap(rnn)(sequences)

    early_idx = 4 * m * (m+4)
    late_idx = 74 * early_idx

    early_mask = mask[:,:early_idx]
    early_h = H[:,:early_idx,:][early_mask]
    early_states = states[:,:early_idx][early_mask]
    early_centroid = jnp.array([
        jnp.mean(early_h[early_states == i], axis=0) for i in range(31)
    ])

    late_mask = mask[:,late_idx:]
    late_h = H[:, late_idx:, :][late_mask]
    late_states = states[:,late_idx:][late_mask]
    late_centroid = jnp.array([
        jnp.mean(late_h[late_states == i], axis=0) for i in range(31)
    ])

    centroid_delta = late_centroid - early_centroid
    delta = jnp.mean(centroid_delta, axis=0)
    delta = delta / jnp.linalg.norm(delta)

    a = jnp.where(mask, H @ delta, jnp.nan)

    fig, axes = plt.subplots()
    axes.plot(jnp.nanmean(a, axis=0))
    axes.set_ylabel('Activity projected onto Drift Vector')
    axes.set_xlabel('Time')

    if idx in [0, 2, 6]:
        axes.set_title(f"{idx}, Generalizer")
    else:
        axes.set_title(idx)
        
    plt.show()


    from sklearn.linear_model import LinearRegression

    h = H[mask]
    trial_no = jnp.cumsum(mask, axis=1)[mask]
    train = jr.uniform(key = key, shape=(h.shape[0])) < 0.8

    decoder = LinearRegression(fit_intercept=True).fit(
        h[train], 
        trial_no[train])
    position_encoding_perf = position_encoding_perf.at[idx].set(
        decoder.score(h[~train], trial_no[~train])
    )

    a = decoder.coef_
    a = a / jnp.linalg.norm(a)
    # print(idx, jnp.dot(a, delta))
    angle_from_position_index = angle_from_position_index.at[idx].set(
        jnp.dot(a, delta)
    )

    _, key = jr.split(key, 2)

#%%

generalizers = jnp.array([0, 2, 6])
stable_nongeneralizers = jnp.array([1, 3, 7, 8, 9])
unstable_nongeneralizers = jnp.array([4, 5])

plt.bar(generalizers, 
        position_encoding_perf[generalizers], 
        color='r', label='generalizers')
plt.bar(stable_nongeneralizers,
         position_encoding_perf[stable_nongeneralizers], 
         color='k', label='stable nongeneralizers')
plt.bar(unstable_nongeneralizers,
         position_encoding_perf[unstable_nongeneralizers], 
         color='b', label='unstable, nongeneralizers')
plt.legend()
plt.xlabel('Model')
plt.ylabel('Trial Number Decoder R^2')

#%%
for idx in range(10):
    model = models[idx]
    rnn = model.rnn

    H = jax.vmap(rnn)(sequences)
    a = jnp.where(mask, jnp.linalg.norm(H, axis=2), jnp.nan)

    plt.plot(jnp.nanmean(a, axis=0))
    plt.title(idx)
    plt.show()
# %%

# 2. Bin the data using numpy
num_bins = 51
counts, bin_edges = jnp.histogram(
    jnp.arccos(angle_from_position_index), 
    bins=num_bins, 
    range=(0, 2 * jnp.pi))

# 3. Calculate bin centers and widths for the polar plot
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
bin_width = 2 * jnp.pi / num_bins

# 4. Create the polar plot
fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(projection='polar'))

# Plot the bars
bars = ax.bar(
    bin_centers, 
    counts, 
    width=bin_width, 
    bottom=0.0, 
    edgecolor='black', 
    alpha=0.7, 
    color='skyblue'
)

# Optional: Adjust the direction (e.g., 0 degrees at the top / North)
ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)  # Clockwise orientation

plt.title("Angle of drift vector vs. time encoding vector", va='bottom')

# %%
