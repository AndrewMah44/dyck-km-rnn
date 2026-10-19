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
from dyck_rnn.data.samplers import powerlaw

from dyck_rnn.data.dyck_hmm import dyck_hmm
from dyck_rnn.utils.get_depth import get_depth
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
        'train_size': 50_000,
        'test_size': 10_000,
    }
}

run_name = "DyckKM_k02_m04_linear_h24_mlp1"
run = 0

# ==== Load Model ====
model_dir = "/Users/amah/Documents/GitHub/dyck-km-rnn/runs/old2"
run_dir = Path(model_dir) / run_name / f"run_{run:02}"

with open(run_dir / "config.yaml", "r") as file:
    model_config = yaml.safe_load(file)

model = load_model(
    run_name + f"/run_{run:02}", 
    run_parent_dir = model_dir)
rnn = model.rnn
readout = model.readout

# ==== Generate Test Dataset ====
k = model_config['data']['k']
m = model_config['data']['m']
max_length = (4 * m * (m + 4))
DyckHMM = dyck_hmm(k, m)

master_key = jr.PRNGKey(eval_config['experiment']['seed'])
train_key, test_key = jr.split(master_key, 2)

# Decoder Training Data
train_data_key, train_length_key = jr.split(train_key, 2)
train_lengths = powerlaw(
    train_length_key, 
    15, 
    max_length, 
    model_config['data']['alpha'], 
    shape=(eval_config['data']['train_size'],)
)

train_states, train_sequences = DyckHMM.batch_sample_sequence(
    batch_size = eval_config['data']['train_size'], 
    num_timesteps = max_length, 
    min_length = train_lengths, 
    key = train_data_key)
train_mask = (train_sequences < 2 * DyckHMM.k)
train_states_vec = train_states[train_mask]

# Decoder Test Data
test_data_key, test_length_key = jr.split(test_key, 2)
test_lengths = powerlaw(
    test_length_key, 
    15, 
    max_length, 
    model_config['data']['alpha'], 
    shape=(eval_config['data']['test_size'],)
)

test_states, test_sequences = DyckHMM.batch_sample_sequence(
    batch_size = eval_config['data']['test_size'], 
    num_timesteps = max_length, 
    min_length = test_lengths, 
    key = test_data_key)
test_mask = (test_sequences < 2 * DyckHMM.k)
test_states_vec = test_states[test_mask]

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

stack_item_decoders = []
for n in range(m):
    train_stack_items = get_stack_item(train_states_vec, n = n, k = k)
    test_stack_items = get_stack_item(test_states_vec, n = n, k = k)

    stack_item_decoder = LogisticRegression(
        solver="lbfgs",
        max_iter=5000,
        tol=1e-4,
        C=1.0)

    stack_item_decoder.fit(train_activity, train_stack_items)
    print(stack_item_decoder.score(test_activity, test_stack_items))

    stack_item_decoders.append(stack_item_decoder)

def get_state_stack(state, decoders):
    return [
        decoder.predict(state.reshape(1,-1)).item() 
        for decoder in decoders]

#%%

Wrec = model.rnn.rnn.W_rec
h = test_activity[jnp.where(test_states_vec == 0)[0][0]]
h2 = Wrec @ h
h3 = Wrec @ h2
h4 = Wrec @ h3
h5 = Wrec @ h4

print(get_state_stack(h, stack_item_decoders))
print(get_state_stack(h2, stack_item_decoders))
print(get_state_stack(h3, stack_item_decoders))
print(get_state_stack(h4, stack_item_decoders))
print(get_state_stack(h5, stack_item_decoders))

#%% ==== Editing Utils ====

# Analysis To Do:
# 1. Editing the stack top changes the next token prediction
# 2. Editing the stack top does not change other features 
#       - depth, other stack positions, etc.
# 3. Show that random subspace editing does not change next token prediction
import scipy

def _subspace_editing(H1, H2, Q):
    return H1 + ((H2 - H1) @ Q) @ Q.T

def subspace_editing_by_state(state1, state2, state_vec, H, Q, n_samples, key):
    original_key, target_key = jr.split(key, 2)

    orig_state = jnp.where(jnp.isin(state_vec, state1))[0]
    target_state = jnp.where(jnp.isin(state_vec, state2))[0]

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
Q = scipy.linalg.orth(stack_item_decoders[0].coef_.T)
n_samples = 5_000
key1, key2 = jr.split(jr.PRNGKey(1))

for state in range(30):
    idx1 = jr.choice(
        key1, 
        jnp.where(test_states_vec == state)[0], 
        (n_samples,))
    idx2 = jr.choice(
        key2, 
        jnp.where(test_states_vec == state+1)[0], 
        (n_samples,))

    h1 = test_activity[idx1]
    h2 = test_activity[idx2]
    hprime = h1 + (h2 - h1) @ Q @ Q.T

    o1 = jax.nn.softmax(jax.vmap(readout)(h1))
    o2 = jax.nn.softmax(jax.vmap(readout)(h2))
    oprime = jax.nn.softmax(jax.vmap(readout)(hprime))

    fig = plt.figure()
    plt.plot(jnp.mean(o1, axis=0), label='orig')
    plt.plot(jnp.mean(o2, axis=0), label='target')
    plt.plot(jnp.mean(oprime, axis=0), 'k--')
    plt.fill_between(jnp.arange(6),
                    jnp.mean(oprime, axis=0) + jnp.std(oprime, axis=0),
                    jnp.mean(oprime, axis=0) - jnp.std(oprime, axis=0),
                    alpha=0.25, color='k')
    plt.legend()
    fig.suptitle([state, state+1])
    plt.show()

    cids = KMeans(4).fit_predict(oprime)

    fig, axes = plt.subplots(2, 2, figsize=(5,5))
    for cluster, ax in enumerate(axes.flatten()):
        ax.plot(oprime[cids == cluster].mean(0))
        ax.set_title(jnp.mean(cids==cluster))

    fig.suptitle([state, state+1])
    fig.tight_layout()
    plt.show()

    key1, key2 = jr.split(key2, 2)

#%%
cond1 = (test_states_vec % 2 == 0) \
    & (test_states_vec > 0) \
    & (test_states_vec < 15)
idx1 = jr.choice(
    key1, 
    jnp.where(cond1)[0], 
    (n_samples,))

cond2 = (test_states_vec % 2 == 1) \
    & (test_states_vec > 0) \
    & (test_states_vec < 15)
idx2 = jr.choice(
    key2, 
    jnp.where(cond2)[0], 
    (n_samples,))

h1 = test_activity[idx1]
h2 = test_activity[idx2]
hprime = h1 + (h2 - h1) @ Q @ Q.T

o1 = jax.nn.softmax(jax.vmap(readout)(h1))
o2 = jax.nn.softmax(jax.vmap(readout)(h2))
oprime = jax.nn.softmax(jax.vmap(readout)(hprime))

fig = plt.figure()
plt.plot(jnp.mean(o1, axis=0), color='b', label='orig')
plt.fill_between(jnp.arange(6),
                jnp.mean(o1, axis=0) + jnp.std(o1, axis=0),
                jnp.mean(o1, axis=0) - jnp.std(o1, axis=0),
                alpha=0.25, color='b')

plt.plot(jnp.mean(o2, axis=0), color='r', label='target')
plt.fill_between(jnp.arange(6),
                jnp.mean(o2, axis=0) + jnp.std(o2, axis=0),
                jnp.mean(o2, axis=0) - jnp.std(o2, axis=0),
                alpha=0.25, color='r')

plt.plot(jnp.mean(oprime, axis=0), 'k--')
plt.fill_between(jnp.arange(6),
                jnp.mean(oprime, axis=0) + jnp.std(oprime, axis=0),
                jnp.mean(oprime, axis=0) - jnp.std(oprime, axis=0),
                alpha=0.25, color='k')
plt.legend()
fig.suptitle([state, state+1])
plt.show()

cid = KMeans(4, random_state = 5).fit_predict(oprime)

fig, axes = plt.subplots(2, 2)
for i, ax in enumerate(axes.flatten()):
    ax.plot(oprime[cid == i].mean(0))
    ax.set_title(jnp.mean(cid==i))
fig.tight_layout()

#%%
from sklearn.decomposition import PCA

good_h = hprime[cid == 0]
bad_h = hprime[cid == 1]

pca = PCA(2).fit(test_activity)

N = 25
x = jnp.linspace(-20, 20, N)
y = jnp.linspace(-20, 20, N)
X, Y = jnp.meshgrid(x, y)
X, Y = X.flatten(), Y.flatten()

p_mat = jnp.zeros((N**2, 6))

for i, (x, y) in enumerate(zip(X, Y)):
    h = pca.inverse_transform(jnp.array([x, y]))
    p_mat = p_mat.at[i].set(
        jax.nn.softmax(readout(h))
    )

x, y = pca.transform(test_activity).T
# plt.imshow(p_mat[:,2].reshape(N, N))

plt.scatter(x, y)
plt.imshow(p_mat[:,2].reshape(N, N), 
           extent=[-20, 20, -20, 20])

#%% ==== Editing Stack Content, N = 0. Sampling examples ====

key1, key2, key3, key4 = jr.split(jr.PRNGKey(2), 4)
n_samples = 10_000

x1 = jnp.arange(1, 15)
x2 = jnp.arange(15, 31)

H_prime1, H_orig1, H_target1 = subspace_editing_by_state(
     x1[x1 % 2 == 0], 
     x1[x1 % 2 == 1], 
     test_states_vec, 
     test_activity, 
     Q, 
     n_samples, 
     key1)
H_prime2, H_orig2, H_target2 = subspace_editing_by_state(
     x1[x1 % 2 == 0], 
     x2[x2 % 2 == 1], 
     test_states_vec, 
     test_activity, 
     Q, 
     n_samples, 
     key2)

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
     x1[x1 % 2 == 1], 
     x1[x1 % 2 == 0], 
     test_states_vec, 
     test_activity, 
     Q, 
     n_samples, 
     key3)
H_prime2, H_orig2, H_target2 = subspace_editing_by_state(
     x1[x1 % 2 == 1], 
     x2[x2 % 2 == 0], 
     test_states_vec, 
     test_activity, 
     Q, 
     n_samples, 
     key4)

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
     x1[x1 % 2 == 1], 
     x1[x1 % 2 == 0], 
     test_states_vec, 
     test_activity, 
     Q, 
     n_samples, 
     key3)

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

