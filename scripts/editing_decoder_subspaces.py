#%% Set up
import jax
jax.config.update("jax_enable_x64", True)

import yaml
import argparse
import pickle as pkl
import jax.numpy as jnp
import jax.random as jr
from pathlib import Path
from itertools import product
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from jax.scipy.special import kl_div

from dyck_rnn.data.dyck_hmm import dyck_hmm
from dyck_rnn.utils.get_depth import get_depth
from dyck_rnn.data.load_model import load_model

# Experiment
task = "DyckKM"
seed = 324

# Data
k = 2
m = 5
test_size = 50_000
num_timesteps = 84

# Model
cell_type = 'Linear'
hidden_size = 64
mlp_depth = 1

run = 0

# ==== Path Mangement ====
run_parent_dir = Path("/Users/amah/Documents/GitHub/dyck-km-rnn/runs/")
run_name = task \
    + f"_k{k:02}_m{m:02}" \
    + f"_{cell_type}_h{hidden_size}_mlp{mlp_depth}" \
    + f"/run_{run:02}"
run_dir = run_parent_dir / run_name

# ==== RNG Management ====
key = jr.PRNGKey(seed)

# ==== Generate Datasets ====
DyckHMM = dyck_hmm(k, m)

states, sequences = DyckHMM.batch_sample_sequence(
    batch_size = test_size, 
    num_timesteps = num_timesteps, 
    min_length = 15, 
    key = key)
mask = (sequences < 2 * DyckHMM.k).flatten()

states = states.flatten()[mask]
depth = get_depth(states, k, m)

# ==== Load ====
model = load_model(run_name, run_parent_dir=run_parent_dir)
rnn = model.rnn
readout = model.readout

decoder_class = 'logistic'
depth_decoder_file = run_dir / f"decoders/depth_{decoder_class}_decoder.pkl"

with open(depth_decoder_file, "rb") as file:
    decoder_file = pkl.load(file)

with open(run_dir / f"decoders/decoder_config.yaml", "r") as file:
    training_config = yaml.safe_load(file)

#%% ==== Generate Datasets ====
master_key = jr.PRNGKey(training_config['experiment']['seed'])

train_key, test_key = jr.split(master_key, 2)
DyckHMM = dyck_hmm(
    training_config['run']['k'][0],
    training_config['run']['m'][0])

train_states, train_sequences = DyckHMM.batch_sample_sequence(
    batch_size = training_config['data']['train_size'], 
    num_timesteps = training_config['data']['num_timesteps'], 
    min_length = 15, 
    key = train_key)
train_mask = (train_sequences < 2 * DyckHMM.k)
train_states = train_states[train_mask]

test_states, test_sequences = DyckHMM.batch_sample_sequence(
    batch_size = training_config['data']['test_size'], 
    num_timesteps = training_config['data']['num_timesteps'], 
    min_length = 15, 
    key = test_key)
test_mask = (test_sequences < 2 * DyckHMM.k)
test_states = test_states[test_mask]

# ==== RNN Activity ====
train_activity_mat = jax.vmap(rnn)(train_sequences)
train_activity = train_activity_mat[train_mask]

test_activity_mat = jax.vmap(rnn)(test_sequences)
test_activity = test_activity_mat[test_mask]

# ==== Stack Item Decoding ====
def get_stack_item(states, n, k):
    depths = jnp.floor(jnp.log((k - 1) * states + 1) / jnp.log(k))

    start = (k**depths - 1) // (k - 1)
    offset = states - start
    stack_items = (offset // (k**n)) % k

    return jnp.where(depths > n, stack_items, -1).astype(jnp.int32)

def get_full_stack(state, k, m):
    return [get_stack_item(state, n, k).item() for n in range(m)]

#%% ==== Train Stack Decoders ====
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from copy import deepcopy

base_decoder = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                solver="lbfgs",
                max_iter=5000,
                tol=1e-4,
                C=1.0,
            ),
        )

stack_item_decoders = []
for n in range(m):
    train_stack_items = get_stack_item(train_states, n = n, k = k)
    test_stack_items = get_stack_item(test_states, n = n, k = k)

    stack_item_decoder = deepcopy(base_decoder)

    stack_item_decoder.fit(train_activity, train_stack_items)
    print(stack_item_decoder.score(test_activity, test_stack_items))

    stack_item_decoders.append(stack_item_decoder)

#%% ==== Editing Utils ====

# Analysis To Do:
# 1. Editing the stack top changes the next token prediction
# 2. Editing the stack top does not change other features 
#       - depth, other stack positions, etc.
# 3. Show that random subspace editing does not change next token prediction
import scipy

def _subspace_editing(H1, H2, Q):
    return H1 + ((H2 - H1) @ Q) @ Q.T

def subspace_editing_by_state(state1, state2, H, Q, n_samples, key):
    original_key, target_key = jr.split(key, 2)

    orig_state = jnp.where(jnp.isin(train_states, state1))[0]
    target_state = jnp.where(jnp.isin(train_states, state2))[0]

    orig_idx = jr.choice(
        original_key, 
        orig_state, 
        shape=(n_samples,))
    target_idx = jr.choice(
        target_key, 
        target_state, 
        shape=(n_samples,))

    H_orig = H[orig_idx]
    H_target = H[target_idx]
    H_prime = _subspace_editing(H_orig, H_target, Q)

    return H_prime, H_orig, H_target

def plot_next_token_prediction(H, ax, color, linestyle='-', title=None):
    probs = jax.nn.softmax(jax.vmap(readout)(H))

    ax.plot(probs.mean(0), color=color, linestyle=linestyle)
    ax.fill_between(jnp.arange(2*k + 2),
                    probs.mean(0) + probs.std(0),
                    probs.mean(0) - probs.std(0),
                    color=color, alpha=0.2)
    ax.set_title(title)

#%% ==== Editing Stack Content, N = 0. Hand-picked examples ====
Q = scipy.linalg.orth(stack_item_decoders[0][-1].coef_.T)

key1, key2 = jr.split(jr.PRNGKey(1))
n_samples = 5_000

x = jnp.arange(1, 31)

H_prime1, H_orig1, H_target1 = subspace_editing_by_state(
     3, 4, train_activity, Q, n_samples, key1)
H_prime2, H_orig2, H_target2 = subspace_editing_by_state(
     1, 36, train_activity, Q, n_samples, key2)

fig, axes = plt.subplots(3, 2, figsize=(6,4))
plot_next_token_prediction(
    H_orig1, axes[0,0], 'k', title='Mean Original Ouput')

plot_next_token_prediction(
    H_target1, axes[1,0], 'b', title='Mean Target Ouput (Internal)')

plot_next_token_prediction(
    H_prime1, axes[2,0], 'r', linestyle='--', title='Mean Edited Ouput')


plot_next_token_prediction(
    H_orig2, axes[0,1], 'k', title='Mean Original Ouput')

plot_next_token_prediction(
    H_target2, axes[1,1], 'b', title='Mean Target Ouput (Leaf)')

plot_next_token_prediction(
    H_prime2, axes[2,1], 'r', linestyle='--', title='Mean Edited Ouput')
fig.tight_layout()

#%% ==== Editing Stack Content, N = 0. Sampling examples ====

key1, key2, key3, key4 = jr.split(jr.PRNGKey(2), 4)
n_samples = 5_000

x1 = jnp.arange(1, 31)
x2 = jnp.arange(31, 63)

H_prime1, H_orig1, H_target1 = subspace_editing_by_state(
     x1[x1 % 2 == 0], x1[x1 % 2 == 1], train_activity, Q, n_samples, key1)
H_prime2, H_orig2, H_target2 = subspace_editing_by_state(
     x1[x1 % 2 == 0], x2[x2 % 2 == 1], train_activity, Q, n_samples, key2)

fig, axes = plt.subplots(3, 2, figsize=(6,4))
plot_next_token_prediction(
    H_orig1, axes[0,0], 'k', title='Mean Original Ouput')

plot_next_token_prediction(
    H_target1, axes[1,0], 'b', title='Mean Target Ouput (Internal)')

plot_next_token_prediction(
    H_prime1, axes[2,0], 'r', linestyle='--', title='Mean Edited Ouput')

plot_next_token_prediction(
    H_orig2, axes[0,1], 'k', title='Mean Original Ouput')

plot_next_token_prediction(
    H_target2, axes[1,1], 'b', title='Mean Target Ouput (Leaf)')

plot_next_token_prediction(
    H_prime2, axes[2,1], 'r', linestyle='--', title='Mean Edited Ouput')
fig.tight_layout()

# ==============
H_prime1, H_orig1, H_target1 = subspace_editing_by_state(
     x1[x1 % 2 == 1], x1[x1 % 2 == 0], train_activity, Q, n_samples, key3)
H_prime2, H_orig2, H_target2 = subspace_editing_by_state(
     x1[x1 % 2 == 1], x2[x2 % 2 == 0], train_activity, Q, n_samples, key4)

fig, axes = plt.subplots(3, 2, figsize=(6,4))
plot_next_token_prediction(
    H_orig1, axes[0,0], 'k', title='Mean Original Ouput')

plot_next_token_prediction(
    H_target1, axes[1,0], 'b', title='Mean Target Ouput (Internal)')

plot_next_token_prediction(
    H_prime1, axes[2,0], 'r', linestyle='--', title='Mean Edited Ouput')

plot_next_token_prediction(
    H_orig2, axes[0,1], 'k', title='Mean Original Ouput')

plot_next_token_prediction(
    H_target2, axes[1,1], 'b', title='Mean Target Ouput (Leaf)')

plot_next_token_prediction(
    H_prime2, axes[2,1], 'r', linestyle='--', title='Mean Edited Ouput')
fig.tight_layout()

#%% ==== Clustering Sampling Final Probs - Even -> Odd (Internal)====

from sklearn.cluster import KMeans

H_prime1, H_orig1, H_target1 = subspace_editing_by_state(
     x1[x1 % 2 == 0], x1[x1 % 2 == 1], train_activity, Q, n_samples, key1)

target_probs = jax.nn.softmax(jax.vmap(readout)(H_prime1))

idx = KMeans(3, random_state = 5).fit_predict(target_probs)

# fig, axes = plt.subplots(3, 2, figsize=(6,4))
fig, axes = plt.subplots(2, 2, figsize=(6,4))

plot_next_token_prediction(
    H_prime1, axes[0,0], 'k', title='Mean Target Ouput (Leaf)')

for i, ax in enumerate(axes.flatten()[1:]):
    x = jnp.arange(6)
    y = jnp.mean(target_probs[idx == i], axis=0)
    yerr = jnp.std(target_probs[idx == i], axis=0)

    ax.plot(x, y)
    ax.fill_between(x, y+yerr, y-yerr, alpha=0.2)

axes[0,1].set_title(f"Cluster 0 ({jnp.mean(idx == 0):0.4f}) - Bad")
axes[1,0].set_title(f"Cluster 1 ({jnp.mean(idx == 1):0.4f}) - Good")
axes[1,1].set_title(f"Cluster 2 ({jnp.mean(idx == 2):0.4f}) - Bad")

fig.tight_layout()

#%% ==== Clustering Sampling Final Probs - Even -> Odd (Leaf)====

H_prime1, H_orig1, H_target1 = subspace_editing_by_state(
     x1[x1 % 2 == 0], x2[x2 % 2 == 1], train_activity, Q, n_samples, key1)

target_probs = jax.nn.softmax(jax.vmap(readout)(H_prime1))

idx = KMeans(3, random_state = 5).fit_predict(target_probs)

# fig, axes = plt.subplots(3, 2, figsize=(6,4))
fig, axes = plt.subplots(2, 2, figsize=(6,4))

plot_next_token_prediction(
    H_prime1, axes[0,0], 'k', title='Mean Target Ouput (Leaf)')

for i, ax in enumerate(axes.flatten()[1:]):
    x = jnp.arange(6)
    y = jnp.mean(target_probs[idx == i], axis=0)
    yerr = jnp.std(target_probs[idx == i], axis=0)

    ax.plot(x, y)
    ax.fill_between(x, y+yerr, y-yerr, alpha=0.2)

axes[0,1].set_title(f"Cluster 0 ({jnp.mean(idx == 0):0.4f}) - Bad")
axes[1,0].set_title(f"Cluster 1 ({jnp.mean(idx == 1):0.4f}) - Good")
axes[1,1].set_title(f"Cluster 2 ({jnp.mean(idx == 2):0.4f}) - Bad")

fig.tight_layout()


#%%

fig.tight_layout()

cluster = KMeans(n_clusters = 4,
                 random_state=0).fit(readout_prime)

fig, axes = plt.subplots(2, 2, figsize=(6, 4))

x = jnp.arange(6)
for i, ax in enumerate(axes.flatten()):
    y = jnp.mean(readout_prime[cluster.labels_ == i], axis=0)
    yerr = jnp.std(readout_prime[cluster.labels_ == i], axis=0)
    n = jnp.sum(cluster.labels_ == i)

    ax.plot(x, y)
    ax.fill_between(x, y+yerr, y-yerr, alpha=0.25)
    ax.set_xticks(x)
    ax.set_title(f"Cluster {i}: n = {n} ({n / n_samples:0.4f})")

fig.supxlabel('Token')
fig.supylabel('Prob.')
fig.suptitle('Edited Output Clusters')

fig.tight_layout()

#%% Editing Stack Content, N = 1
import scipy

Q = scipy.linalg.orth(stack_item_decoders[1][-1].coef_.T)

key = jr.PRNGKey(1)
n_samples = 5_000

original_key, target_key = jr.split(key, 2)

orig_state = jnp.where(train_states  == 3)[0]
target_state = jnp.where(train_states == 5)[0]

orig_idx = jr.choice(
    original_key, 
    orig_state, 
    shape=(n_samples,))
target_idx = jr.choice(
    target_key, 
    target_state, 
    shape=(n_samples,))

# t = 0
H_orig = train_activity[orig_idx]
H_target = train_activity[target_idx]
H_prime = H_orig + ((H_target - H_orig) @ Q) @ Q.T

readout_orig = jax.nn.softmax(jax.vmap(readout)(H_orig))
readout_target = jax.nn.softmax(jax.vmap(readout)(H_target))
readout_prime = jax.nn.softmax(jax.vmap(readout)(H_prime))

# t = 1
inputs_embedded = rnn.Win(2)

H_orig_1 = jax.vmap(rnn.rnn, in_axes=[0, None])(
    H_orig, inputs_embedded)
H_target_1 = jax.vmap(rnn.rnn, in_axes=[0, None])(
    H_target, inputs_embedded)
H_prime_1 = jax.vmap(rnn.rnn, in_axes=[0, None])(
    H_prime, inputs_embedded)

readout_orig_1 = jax.nn.softmax(jax.vmap(readout)(H_orig_1))
readout_target_1 = jax.nn.softmax(jax.vmap(readout)(H_target_1))
readout_prime_1 = jax.nn.softmax(jax.vmap(readout)(H_prime_1))

fig, axes = plt.subplots(3, 2, figsize=(6,5))
axes[0,0].plot(readout_orig.mean(0), 'k')
axes[0,0].fill_between(jnp.arange(6),
                 readout_orig.mean(0) + readout_orig.std(0),
                 readout_orig.mean(0) - readout_orig.std(0),
                 color='k', alpha=0.2)
axes[0,0].set_title(('t=0\nMean Original Ouput'))

axes[1,0].plot(readout_target.mean(0), 'b')
axes[1,0].fill_between(jnp.arange(6),
                 readout_target.mean(0) + readout_target.std(0),
                 readout_target.mean(0) - readout_target.std(0),
                 color='b', alpha=0.2)
axes[1,0].set_title('Mean Target Ouput')

axes[2,0].plot(readout_prime.mean(0), 'r--')
axes[2,0].fill_between(jnp.arange(6),
                 readout_prime.mean(0) + readout_prime.std(0),
                 readout_prime.mean(0) - readout_prime.std(0),
                 color='r', alpha=0.2)
axes[2,0].set_title('Mean Edited Ouput')



axes[0,1].plot(readout_orig_1.mean(0), 'k')
axes[0,1].fill_between(jnp.arange(6),
                 readout_orig_1.mean(0) + readout_orig_1.std(0),
                 readout_orig_1.mean(0) - readout_orig_1.std(0),
                 color='k', alpha=0.2)
axes[0,1].set_title(('t=1\nMean Original Ouput'))

axes[1,1].plot(readout_target_1.mean(0), 'b')
axes[1,1].fill_between(jnp.arange(6),
                 readout_target_1.mean(0) + readout_target_1.std(0),
                 readout_target_1.mean(0) - readout_target_1.std(0),
                 color='b', alpha=0.2)
axes[1,1].set_title('Mean Target Ouput')

axes[2,1].plot(readout_prime_1.mean(0), 'r--')
axes[2,1].fill_between(jnp.arange(6),
                 readout_prime_1.mean(0) + readout_prime_1.std(0),
                 readout_prime_1.mean(0) - readout_prime_1.std(0),
                 color='r', alpha=0.2)
axes[2,1].set_title('Mean Edited Ouput')
fig.tight_layout()

cluster = KMeans(n_clusters = 3,
                 random_state=0).fit(readout_prime_1)

fig, axes = plt.subplots(2, 2, figsize=(6, 4))

x = jnp.arange(6)
for i, ax in enumerate(axes.flatten()[:3]):
    y = jnp.mean(readout_prime_1[cluster.labels_ == i], axis=0)
    yerr = jnp.std(readout_prime_1[cluster.labels_ == i], axis=0)
    n = jnp.sum(cluster.labels_ == i)

    ax.plot(x, y)
    ax.fill_between(x, y+yerr, y-yerr, alpha=0.25)
    ax.set_xticks(x)
    ax.set_title(f"Cluster {i}: n = {n} ({n / n_samples:0.4f})")

fig.supxlabel('Token')
fig.supylabel('Prob.')
fig.suptitle('Edited Output Clusters')

fig.tight_layout()


#%% Editing Stack Content, N = 2
import scipy

Q = scipy.linalg.orth(stack_item_decoders[2][-1].coef_.T)

key = jr.PRNGKey(1)
n_samples = 5_000

original_key, target_key = jr.split(key, 2)

orig_state = jnp.where(train_states  == 7)[0]
target_state = jnp.where(train_states == 11)[0]

orig_idx = jr.choice(
    original_key, 
    orig_state, 
    shape=(n_samples,))
target_idx = jr.choice(
    target_key, 
    target_state, 
    shape=(n_samples,))

# t = 0
H_orig = train_activity[orig_idx]
H_target = train_activity[target_idx]
H_prime = H_orig + ((H_target - H_orig) @ Q) @ Q.T

readout_orig = jax.nn.softmax(jax.vmap(readout)(H_orig))
readout_target = jax.nn.softmax(jax.vmap(readout)(H_target))
readout_prime = jax.nn.softmax(jax.vmap(readout)(H_prime))

# t = 1
inputs_embedded = rnn.Win(2)

H_orig_1 = jax.vmap(rnn.rnn, in_axes=[0, None])(
    H_orig, inputs_embedded)
H_target_1 = jax.vmap(rnn.rnn, in_axes=[0, None])(
    H_target, inputs_embedded)
H_prime_1 = jax.vmap(rnn.rnn, in_axes=[0, None])(
    H_prime, inputs_embedded)

readout_orig_1 = jax.nn.softmax(jax.vmap(readout)(H_orig_1))
readout_target_1 = jax.nn.softmax(jax.vmap(readout)(H_target_1))
readout_prime_1 = jax.nn.softmax(jax.vmap(readout)(H_prime_1))

# t = 2
inputs_embedded = rnn.Win(2)

H_orig_2 = jax.vmap(rnn.rnn, in_axes=[0, None])(
    H_orig_1, inputs_embedded)
H_target_2 = jax.vmap(rnn.rnn, in_axes=[0, None])(
    H_target_1, inputs_embedded)
H_prime_2 = jax.vmap(rnn.rnn, in_axes=[0, None])(
    H_prime_1, inputs_embedded)

readout_orig_2 = jax.nn.softmax(jax.vmap(readout)(H_orig_2))
readout_target_2 = jax.nn.softmax(jax.vmap(readout)(H_target_2))
readout_prime_2 = jax.nn.softmax(jax.vmap(readout)(H_prime_2))

fig, axes = plt.subplots(3, 3, figsize=(6,5))
axes[0,0].plot(readout_orig.mean(0), 'k')
axes[0,0].fill_between(jnp.arange(6),
                 readout_orig.mean(0) + readout_orig.std(0),
                 readout_orig.mean(0) - readout_orig.std(0),
                 color='k', alpha=0.2)
axes[0,0].set_title(('t=0\nMean Original Ouput'))

axes[1,0].plot(readout_target.mean(0), 'b')
axes[1,0].fill_between(jnp.arange(6),
                 readout_target.mean(0) + readout_target.std(0),
                 readout_target.mean(0) - readout_target.std(0),
                 color='b', alpha=0.2)
axes[1,0].set_title('Mean Target Ouput')

axes[2,0].plot(readout_prime.mean(0), 'r--')
axes[2,0].fill_between(jnp.arange(6),
                 readout_prime.mean(0) + readout_prime.std(0),
                 readout_prime.mean(0) - readout_prime.std(0),
                 color='r', alpha=0.2)
axes[2,0].set_title('Mean Edited Ouput')



axes[0,1].plot(readout_orig_1.mean(0), 'k')
axes[0,1].fill_between(jnp.arange(6),
                 readout_orig_1.mean(0) + readout_orig_1.std(0),
                 readout_orig_1.mean(0) - readout_orig_1.std(0),
                 color='k', alpha=0.2)
axes[0,1].set_title(('t=1\nMean Original Ouput'))

axes[1,1].plot(readout_target_1.mean(0), 'b')
axes[1,1].fill_between(jnp.arange(6),
                 readout_target_1.mean(0) + readout_target_1.std(0),
                 readout_target_1.mean(0) - readout_target_1.std(0),
                 color='b', alpha=0.2)
axes[1,1].set_title('Mean Target Ouput')

axes[2,1].plot(readout_prime_1.mean(0), 'r--')
axes[2,1].fill_between(jnp.arange(6),
                 readout_prime_1.mean(0) + readout_prime_1.std(0),
                 readout_prime_1.mean(0) - readout_prime_1.std(0),
                 color='r', alpha=0.2)
axes[2,1].set_title('Mean Edited Ouput')



axes[0,2].plot(readout_orig_2.mean(0), 'k')
axes[0,2].fill_between(jnp.arange(6),
                 readout_orig_2.mean(0) + readout_orig_2.std(0),
                 readout_orig_2.mean(0) - readout_orig_2.std(0),
                 color='k', alpha=0.2)
axes[0,2].set_title(('t=1\nMean Original Ouput'))

axes[1,2].plot(readout_target_2.mean(0), 'b')
axes[1,2].fill_between(jnp.arange(6),
                 readout_target_2.mean(0) + readout_target_2.std(0),
                 readout_target_2.mean(0) - readout_target_2.std(0),
                 color='b', alpha=0.2)
axes[1,2].set_title('Mean Target Ouput')

axes[2,2].plot(readout_prime_2.mean(0), 'r--')
axes[2,2].fill_between(jnp.arange(6),
                 readout_prime_2.mean(0) + readout_prime_2.std(0),
                 readout_prime_2.mean(0) - readout_prime_2.std(0),
                 color='r', alpha=0.2)
axes[2,2].set_title('Mean Edited Ouput')

fig.tight_layout()

cluster = KMeans(n_clusters = 4,
                 random_state=0).fit(readout_prime_2)

fig, axes = plt.subplots(2, 2, figsize=(6, 4))

x = jnp.arange(6)
for i, ax in enumerate(axes.flatten()):
    y = jnp.mean(readout_prime_2[cluster.labels_ == i], axis=0)
    yerr = jnp.std(readout_prime_2[cluster.labels_ == i], axis=0)
    n = jnp.sum(cluster.labels_ == i)

    ax.plot(x, y)
    ax.fill_between(x, y+yerr, y-yerr, alpha=0.25)
    ax.set_xticks(x)
    ax.set_title(f"Cluster {i}: n = {n} ({n / n_samples:0.4f})")

fig.supxlabel('Token')
fig.supylabel('Prob.')
fig.suptitle('Edited Output Clusters')

fig.tight_layout()





# %%

orig_idx = jr.choice(jr.PRNGKey(1), 
                     jnp.where(states == 1)[0], 
                     shape=(5_000,))
target_idx = jr.choice(jr.PRNGKey(2), 
                     jnp.where(states == 3)[0], 
                     shape=(5_000,))

H_final = H[orig_idx] + ((H[target_idx] - H[orig_idx]) @ Q) @ Q.T


inputs_embedded = rnn.Win(2)


h_next_orig = jax.vmap(rnn.rnn, in_axes=[0, None])(
    H[orig_idx], inputs_embedded)
h_next_prime = jax.vmap(rnn.rnn, in_axes=[0, None])(
    H_final, inputs_embedded)

o1 = jax.nn.softmax(jax.vmap(readout)(h_next_orig))
o2 = jax.nn.softmax(jax.vmap(readout)(h_next_prime))

plt.plot(o1.mean(0))
plt.plot(o2.mean(0))
plt.fill_between(
    jnp.arange(6),
    o2.mean(0) - o2.std(0),
    o2.mean(0) + o2.std(0), alpha=0.15)

