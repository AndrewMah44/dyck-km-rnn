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
from dyck_rnn.utils.plot import plot_probs_by_condition

from dyck_rnn.training.losses import kl_div
from dyck_rnn.data.load_model import load_model

# ==== Data Set Up ====
run_name = "DyckKM_k02_m04_linear_Stable_h24_mlp1"
# run_name = "DyckKM_k04_m04_linear_h48_mlp1"

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
max_length = 4 * m * (m + 4)
n_states = int((k ** (m+1) - 1) / (k - 1))

DyckHMM = dyck_hmm(k, m)

lengths = powerlaw(
    length_key, 
    15, 
    5*(4 * m * (m + 4)), 
    0.9, 
    shape=(eval_config['data']['test_size'],)
)

states_full, sequences = DyckHMM.batch_sample_sequence(
    batch_size = eval_config['data']['test_size'], 
    num_timesteps = max_length, 
    min_length = lengths, 
    key = data_key)
states_full = states_full[:,:max_length]
sequences = sequences[:,:max_length]
masks = sequences != (2 * k + 1)

# ===== Load Models =====
models = []
for run in range(n_runs):
    model = load_model(
        run_name + f"/run_{run:02}", 
        run_parent_dir = run_dir)
    models.append(model)


#%% Functions

from functools import partial

@partial(jax.jit, static_argnames=("k", "m"))
def state_id_to_stack(state_id, k, m):
    """
    Decode a breadth-first state ID into a right-padded stack.

    Root/empty-stack state ID is 0. Token types are 0, ..., k-1.
    """
    tokens_reversed = jnp.full((m,), -1, dtype=jnp.float32)

    def extract_token(i, carry):
        current_id, tokens, depth = carry
        active = current_id > 0

        safe_id = jnp.maximum(current_id, 1)
        parent, token = jnp.divmod(safe_id - 1, k)

        tokens = tokens.at[i].set(
            jnp.where(active, token.astype(jnp.float32), -1)
        )
        current_id = jnp.where(active, parent, 0)
        depth = depth + active.astype(jnp.int32)

        return current_id, tokens, depth

    remaining_id, tokens_reversed, depth = jax.lax.fori_loop(
        0,
        m,
        extract_token,
        (state_id, tokens_reversed, jnp.array(0, dtype=jnp.int32)),
    )

    return tokens_reversed


@partial(jax.jit, static_argnames=("k", "m"))
def stack_to_state_id(stack, k, m):
    """
    Encode a right-padded stack into its breadth-first state ID.

    Expected stack format matches `state_id_to_stack`:
        [top_token, ..., bottom_token, -1, ..., -1]

    The empty stack maps to state ID 0.
    Token types are 0, ..., k - 1.
    """
    stack = jnp.asarray(stack)

    def add_token(i, state_id):
        # Traverse from bottom of the stack toward the top.
        token = stack[m - 1 - i]
        active = token >= 0

        child_id = k * state_id + token.astype(jnp.int32) + 1
        return jnp.where(active, child_id, state_id)

    return jax.lax.fori_loop(
        0,
        m,
        add_token,
        jnp.array(0, dtype=jnp.int32),
    )
# %% 

f = jax.vmap(state_id_to_stack, in_axes=[0, None, None])
stacks = jax.vmap(f, in_axes=[0, None, None])(states_full, k, m)
stacks = stacks.at[states_full == 31].set(jnp.array([-1, -1, -1, -1]))

model = models[0]
rnn = model.rnn
readout = model.readout

H_full = jax.vmap(rnn)(sequences)

#%% Train Decoders

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

decoders = []

for i in range(4):
    decoder = LogisticRegression(fit_intercept=True, max_iter=1_000)

    y = stacks[:,:,i]

    usethese = states_full != n_states
    y = y[usethese]
    X = H_full[usethese]

    train = jr.uniform(jr.PRNGKey(1), shape=(y.shape)) < 0.8

    X_train = X[train]
    y_train = y[train]

    X_test = X[~train]
    y_test = y[~train]

    decoder.fit(X_train, y_train)
    decoders.append(decoder)
    print(f"Position {i} Acc.: {decoder.score(X_test, y_test)}")

#%% Generic Stack Top Patching Does Not Work
i = 0

w = decoders[i].coef_.T
w = w / jnp.linalg.norm(w)

stack_item = stacks[:,:,i]

key1, key2 = jr.split(jr.PRNGKey(1))

depth = jnp.sum(~jnp.isnan(stacks), axis=2)
depth = depth.at[states == 31].set(-1)

H0 = jr.choice(key1, H[(stack_item == 0)], shape=(100,))
H1 = jr.choice(key2, H[(stack_item == 1)], shape=(100,))

H01 = H0 + (H1 - H0) @ w @ w.T
H10 = H1 + (H0 - H1) @ w @ w.T

p0 = jax.nn.softmax(jax.vmap(readout)(H0))
p1 = jax.nn.softmax(jax.vmap(readout)(H1))
p01 = jax.nn.softmax(jax.vmap(readout)(H01))
p10 = jax.nn.softmax(jax.vmap(readout)(H10))

fig, axes = plt.subplots(2, 1)
axes[0].plot(p0.mean(0), 'k')
axes[0].plot(p01.mean(0), 'r')

axes[1].plot(p1.mean(0), 'k')
axes[1].plot(p10.mean(0), 'r')

# %% What is the stacks all agree except on the first
w = decoders[0].coef_.T
w = w / jnp.linalg.norm(w)

n = 1_000

stack1 = jnp.array([0, 0, 1, 0])
stack2 = jnp.array([1, 0, 1, 0])

H1 = H[jnp.all(stacks == stack1, axis=2)][:n]
H2 = H[jnp.all(stacks == stack2, axis=2)][:n]

H12 = H1 + (H2 - H1) @ w @ w.T
H21 = H2 + (H1 - H2) @ w @ w.T

p1 = jax.nn.softmax(jax.vmap(readout)(H1))
p2 = jax.nn.softmax(jax.vmap(readout)(H2))
p12 = jax.nn.softmax(jax.vmap(readout)(H12))
p21 = jax.nn.softmax(jax.vmap(readout)(H21))


fig, axes = plt.subplots(2, 1)
axes[0].plot(p1.mean(0), 'k')
axes[0].plot(p12.mean(0), 'r')

axes[1].plot(p2.mean(0), 'k')
axes[1].plot(p21.mean(0), 'r')

# %% Let's just consider centroid differences

def get_centroid(state_id, H, states):
    h = jnp.where((states == state_id)[..., None], H, jnp.nan)
    return jnp.nanmean(h, axis=(0,1))

# Previous versions used full H matrix for centroids. Now uses a test train 
# split
n_train = 4_000
H_train = H_full[:n_train]
states_train = states_full[:n_train]

# From here on out, call H and states the test split not included in the 
# centroid estimation
H = H_full[n_train:]
states = states_full[n_train:]

# Centroids genearted using training states
centroids = jax.vmap(get_centroid, in_axes=[0, None, None])(
    jnp.arange(n_states), H_train, states_train)
jax.block_until_ready(centroids)

C = centroids - centroids.mean(axis=0, keepdims=True)
_, S, Vh = jnp.linalg.svd(C, full_matrices=False)

exp_var = jnp.cumsum(S**2) / jnp.sum(S**2)
dim = jnp.argmax(exp_var > 0.95) + 1

fig = plt.figure()
plt.plot(exp_var, 'ko-')

plt.xlabel('N Singular Values')
plt.ylabel('Cumulative fraction of variance explained')

#%% Centroid Geometry

norms = jnp.linalg.norm(centroids, axis=1, keepdims=True)
cosine_similarity = jnp.abs((centroids @ centroids.T) / (norms @ norms.T))
centroid_angles = jnp.arccos(cosine_similarity)

dat = centroid_angles[jnp.triu_indices(centroid_angles.shape[0], k=1)]

num_bins = 100
bins = jnp.linspace(0.0, 2 * jnp.pi, num_bins + 1)
counts, _, = jnp.histogram(dat, bins=bins)

widths = jnp.diff(bins)
centers = bins[:-1] + widths / 2

fig = plt.figure()
f = plt.imshow(cosine_similarity[
    jnp.ix_(jnp.argsort(jnp.arange(31) % 2),
    jnp.argsort(jnp.arange(31) % 2))])
plt.colorbar(f, label='Cosine Similarity')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(projection='polar')
bars = ax.bar(centers, counts, 
              width=widths, bottom=0.0, edgecolor='black', alpha=0.7)

ax.set_title("Polar Histogram of Angles", va='bottom', fontsize=14)

fig.tight_layout()
plt.show()

#%% Example Using Full Displacement

def get_hidden_state_probs(h):
    return jax.nn.softmax(jax.vmap(readout)(h))

def full_displacement(source_state, target_state):
    displacement = centroids[target_state] - centroids[source_state]

    H_source = H[states == source_state]
    return H_source + displacement

# Two Example States
source_state = n_states - 1
target_state = 1

# Pull Hidden Activation For Each state
p_source = get_hidden_state_probs(H[states == source_state])
p_target = get_hidden_state_probs(H[states == target_state])

# Perturb H_source using full displacement vector
H_perturb = full_displacement(source_state, target_state)
p_perturb = get_hidden_state_probs(H_perturb)

# Plot
plt.plot(range(2*k + 2), p_source.mean(0), 'k', label='source')
plt.fill_between(range(2*k + 2), 
                 p_source.mean(0) + p_source.std(0),
                 p_source.mean(0) - p_source.std(0),
                 color='k', alpha=0.15)

plt.plot(range(2*k + 2), p_target.mean(0), 'b', label='target')
plt.fill_between(range(2*k + 2), 
                 p_target.mean(0) + p_target.std(0),
                 p_target.mean(0) - p_target.std(0),
                 color='b', alpha=0.15)

plt.plot(range(2*k + 2), p_perturb.mean(0), 'r--', label='perturbed')
plt.fill_between(range(2*k + 2), 
                 p_perturb.mean(0) + p_perturb.std(0),
                 p_perturb.mean(0) - p_perturb.std(0),
                 color='r', alpha=0.15)

plt.xlabel('Token')
plt.ylabel('Prob.')
plt.title('Yay it works!')
plt.legend()

#%% Now let's sweep over all possible sources and targets

def dkl(p, q, eps=1e-8):
    p = jnp.clip(p, eps, 1.0)
    q = jnp.clip(q, eps, 1.0)
    return jnp.sum(p * (jnp.log(p) - jnp.log(q)), axis=-1)

dkl_mat = jnp.zeros((n_states, n_states))

for source in range(n_states):
    for target in range(n_states):
        print(source, target)

        # Pull Hidden Activation For Each state
        p_target = get_hidden_state_probs(H[states == target])

        # Perturb H_source using full displacement vector
        H_perturb = full_displacement(source, target)
        p_perturb = get_hidden_state_probs(H_perturb)

        dkl_mat = dkl_mat.at[source, target].set(
            dkl(p_target.mean(0), p_perturb.mean(0)))

fig = plt.figure()
ax = plt.imshow(dkl_mat)
[plt.axhline(int((k ** (depth+1) - 1) / (k - 1))-0.5) for depth in range(m)]
[plt.axvline(int((k ** (depth+1) - 1) / (k - 1))-0.5) for depth in range(m)]

plt.colorbar(ax)
plt.xlabel('target')
plt.ylabel('source')

plt.title('It\'s not a uniform pattern, but it works sometimes')

#%% Now try using a low dimensional subspace  

# Two Example States
source_state = n_states - 1
target_state = 1

# Displacement Matrix
displacement = centroids[target_state] - centroids[source_state]

# Pull Hidden Activation F                                                                                                                                        or Each state
H_source = H[states == source_state]
p_source = jax.nn.softmax(jax.vmap(readout)(H_source))

H_target = H[states == target_state]
p_target = jax.nn.softmax(jax.vmap(readout)(H_target))

fig, axes = plt.subplots(8, 3, figsize=(10,15))
for d, ax in enumerate(axes.flatten()):
    # Perturb H_source using projected displacement vector
    # Projection depends on right singular vectors (TODO: WHY?)
    Q = Vh[:d].T
    H_perturb_p = H_source + (displacement @ Q) @ Q.T
    p_perturb_p = jax.nn.softmax(jax.vmap(readout)(H_perturb_p))

    # Plot
    ax.plot(range(2*k + 2), p_source.mean(0), 'k')
    ax.fill_between(range(2*k + 2), 
                    p_source.mean(0) + p_source.std(0),
                    p_source.mean(0) - p_source.std(0),
                    color='k', alpha=0.15)

    ax.plot(range(2*k + 2), p_target.mean(0), 'b')
    ax.fill_between(range(2*k + 2), 
                    p_target.mean(0) + p_target.std(0),
                    p_target.mean(0) - p_target.std(0),
                    color='b', alpha=0.15)

    ax.plot(range(2*k + 2), p_perturb_p.mean(0), 'm--')
    ax.fill_between(range(2*k + 2), 
                    p_perturb_p.mean(0) + p_perturb_p.std(0),
                    p_perturb_p.mean(0) - p_perturb_p.std(0),
                    color='m', alpha=0.15)

    ax.set_title(d)

fig.tight_layout()

# %% Depth 3 Examples

def generate_stacks(position, x, y):
    stack = jnp.zeros(3).at[-1].set(-1)
    stack1 = stack.at[position].set(1)
    stack2 = stack.at[position].set(0)

    other_positions = jnp.arange(3) != position
    stack1 = stack1.at[other_positions].set(jnp.array([x, y]))
    stack2 = stack2.at[other_positions].set(jnp.array([x, y]))

    return jnp.append(stack1, jnp.array([-1])), \
        jnp.append(stack2, jnp.array([-1]))


stack_displacement_mat = jnp.zeros((3, k, k, 24))
for position in range(3):
    for i, x in enumerate([0,1]):
        for j, y in enumerate([0,1]):

                stack1, stack2 = generate_stacks(position, x, y)
                state1 = stack_to_state_id(stack1, k, m)
                state2 = stack_to_state_id(stack2, k, m)

                stack_displacement_mat \
                    = stack_displacement_mat.at[position, i, j].set(
                        centroids[state1] - centroids[state2])

stack_displacement_mat = stack_displacement_mat.reshape(3, -1, 24)

fig, axes = plt.subplots(3)

for i, ax in enumerate(axes.flatten()):
    _, S, _ = jnp.linalg.svd(stack_displacement_mat[i])
    ax.plot(S, 'ko-')
    ax.set_yscale('log')

fig.supylabel('Magnitude')
fig.supxlabel('Singular Value')
fig.tight_layout()

# %%

def decode_stack(h):
    if h.ndim == 1:
        h = h.reshape(1,-1)

    stack = jnp.zeros((h.shape[0], m), dtype=jnp.float64)

    for i in jnp.arange(m):
        stack = stack.at[:,i].set(
            decoders[i].predict(h)
        )

    return stack

d = stack_displacement_mat.mean(1)

stack_orig = jnp.array([0., 0., 0., -1.])
state_0 = stack_to_state_id(stack_orig, k, m)

H_orig = H[states == state_0][1]

H_perturb0 = H_orig + d[0]
H_perturb1 = H_orig + d[1]
H_perturb2 = H_orig + d[2]

H_perturb01 = H_orig + d[0] + d[1]
H_perturb02 = H_orig + d[0] + d[2]
H_perturb12 = H_orig + d[1] + d[2]

H_perturb012 = H_orig + d[0] + d[1] + d[2]

decoded_0 = decode_stack(H_perturb0)
decoded_1 = decode_stack(H_perturb1)
decoded_2 = decode_stack(H_perturb2)

decoded_01 = decode_stack(H_perturb01)
decoded_02 = decode_stack(H_perturb02)
decoded_12 = decode_stack(H_perturb12)

decoded_012 = decode_stack(H_perturb012)

print(f"Original Stack:       {stack_orig}")
print(f"Flip Position 0:      {decoded_0}")
print(f"Flip Position 1:      {decoded_1}")
print(f"Flip Position 2:      {decoded_2}")
print(f"Flip Positions 0,1:   {decoded_01}")
print(f"Flip Positions 0,2:   {decoded_02}")
print(f"Flip Positions 1,2:   {decoded_12}")
print(f"Flip Positions 0,1,2: {decoded_012}")

# %%

stack_orig = jnp.array([0., 1., 0., -1.])
state_0 = stack_to_state_id(stack_orig, k, m)

H_orig = H[states == state_0]

for position in range(3):
    sign = 1 - 2*stack_orig[position]
    other_positions = jnp.arange(4) != position
    H_perturb = H_orig + sign * d[position]
    
    stack_perturb = decode_stack(H_perturb)

    # Does editing change the desired position?
    target_effect = jnp.mean(
        stack_perturb[:,position] != stack_orig[position])

    # Does editing leave other positions unchanged?
    offtarget_effect = jnp.mean(
        jnp.any(stack_perturb[:,other_positions] \
                != stack_orig[other_positions], axis=1))

    print(position)
    print(f"Target Change Rate: {target_effect}")
    print(f"Off-Target Change Rate: {offtarget_effect}")

    i = jnp.where(other_positions)[0]
    for p in range(3):
        print(f"   Position {i[p]}: {jnp.mean(
            stack_perturb[:,other_positions][:,p] \
                != stack_orig[other_positions][p])}")

    print("")

# %%

def simulate_rnn(rnn, h0, sequence):
    inputs_embedded = jax.vmap(rnn.Win)(sequence)

    def f(carry, inp):
        # Compute next hidden state
        h = rnn.rnn(inp, carry)

        return h, h
    
    # Scan over time dimension
    _, hiddens = jax.lax.scan(f, h0, inputs_embedded)

    return hiddens


# simulate_rnn(model.rnn, jnp.zeros(24), sequences[0]) - H_full[0]
# %%
h_source = H_full[states_full == 7]
h_target = H_full[states_full == 11][115]

d = centroids[11] - centroids[7]
h_perturb = h_source + d

sequence = jnp.array([0, 2, 1, 3, 2, 2])

H_source = simulate_rnn(rnn, h_source[0], sequence)
H_target = simulate_rnn(rnn, h_target, sequence)
H_perturb = jax.vmap(simulate_rnn, in_axes = [None, 0, None])(
    rnn, h_perturb, sequence)

# %%

p_source = jax.nn.softmax(jax.vmap(model.readout)(H_source))
p_target = jax.nn.softmax(jax.vmap(model.readout)(H_target))
p_perturb = jax.nn.softmax(jax.vmap(jax.vmap(model.readout))(H_perturb))

x = range(6)
y = p_perturb.mean(0)
yerr = p_perturb.std(0)

fig, axes = plt.subplots(6, 1, figsize=(5, 9.5))
for i in range(6):
    axes[i].plot(p_source[i], 'k')
    axes[i].plot(p_target[i], 'b')

    axes[i].plot(x, y[i], 'r--')
    axes[i].fill_between(x, y[i] + yerr[i], y[i] - yerr[i],
                     color = 'r', alpha=0.25)

# %%

kl_source = jax.vmap(jax.scipy.special.rel_entr, in_axes=[0, None])(p_perturb, p_source).sum(-1)
kl_target = jax.vmap(jax.scipy.special.rel_entr, in_axes=[0, None])(p_perturb, p_target).sum(-1)

print(f"Mean Dkl with Source: {kl_source.mean(0)}")
print(f"Mean Dkl with Target:{kl_target.mean(0)}")

# %%

print(jax.scipy.special.rel_entr(p_source, p_target).sum(1))

# %%

ds = {}

# Hacky for now
# ==== Depth 1 ====
ds[1] = jnp.zeros((1, 1, 24))

stack1 = jnp.array([0, -1, -1, -1])
stack2 = jnp.array([1, -1, -1, -1])

state1 = stack_to_state_id(stack1, k, m)
state2 = stack_to_state_id(stack2, k, m)
ds[1] = ds[1].at[0, 0].set(centroids[state2] - centroids[state1])

# ==== Depth 2 ====
ds[2] = jnp.zeros((2, 2, 24))

# Position 0
for x in [0, 1]:
    stack1 = jnp.array([0, x, -1, -1])
    stack2 = jnp.array([1, x, -1, -1])

    state1 = stack_to_state_id(stack1, k, m)
    state2 = stack_to_state_id(stack2, k, m)
    ds[2] = ds[2].at[0,x].set(centroids[state2] - centroids[state1])

# Position 1
for x in [0, 1]:
    stack1 = jnp.array([x, 0, -1, -1])
    stack2 = jnp.array([x, 1, -1, -1])

    state1 = stack_to_state_id(stack1, k, m)
    state2 = stack_to_state_id(stack2, k, m)
    ds[2] = ds[2].at[1,x].set(centroids[state2] - centroids[state1])

# ==== Depth 3 ====
ds[3] = jnp.zeros((3, 4, 24))

# Position 0
c = 0
for x in [0, 1]:
    for y in [0, 1]:
        stack1 = jnp.array([0, x, y, -1])
        stack2 = jnp.array([1, x, y, -1])

        state1 = stack_to_state_id(stack1, k, m)
        state2 = stack_to_state_id(stack2, k, m)
        ds[3] = ds[3].at[0, c].set(centroids[state2] - centroids[state1])
        c+=1

# Position 1
c = 0
for x in [0, 1]:
    for y in [0, 1]:
        stack1 = jnp.array([x, 0, y, -1])
        stack2 = jnp.array([x, 1, y, -1])

        state1 = stack_to_state_id(stack1, k, m)
        state2 = stack_to_state_id(stack2, k, m)
        ds[3] = ds[3].at[1, c].set(centroids[state2] - centroids[state1])
        c += 1

# Position 2
c = 0
for x in [0, 1]:
    for y in [0, 1]:
        stack1 = jnp.array([x, y, 0, -1])
        stack2 = jnp.array([x, y, 1, -1])

        state1 = stack_to_state_id(stack1, k, m)
        state2 = stack_to_state_id(stack2, k, m)
        ds[3] = ds[3].at[2, c].set(centroids[state2] - centroids[state1])
        c += 1

# depth 4
ds[4] = jnp.zeros((4, 8, 24))

# Position 0
c = 0
for x in [0, 1]:
    for y in [0, 1]:
        for z in [0, 1]:
            stack1 = jnp.array([0, x, y, z])
            stack2 = jnp.array([1, x, y, z])

            state1 = stack_to_state_id(stack1, k, m)
            state2 = stack_to_state_id(stack2, k, m)
            ds[4] = ds[4].at[0, c].set(centroids[state2] - centroids[state1])
            c += 1

# Position 1
c = 0
for x in [0, 1]:
    for y in [0, 1]:
        for z in [0, 1]:
            stack1 = jnp.array([x, 0, y, z])
            stack2 = jnp.array([x, 1, y, z])

            state1 = stack_to_state_id(stack1, k, m)
            state2 = stack_to_state_id(stack2, k, m)
            ds[4] = ds[4].at[1, c].set(centroids[state2] - centroids[state1])
            c += 1

# Position 2
c = 0
for x in [0, 1]:
    for y in [0, 1]:
        for z in [0, 1]:
            stack1 = jnp.array([x, y, 0, z])
            stack2 = jnp.array([x, y, 1, z])

            state1 = stack_to_state_id(stack1, k, m)
            state2 = stack_to_state_id(stack2, k, m)
            ds[4] = ds[4].at[2, c].set(centroids[state2] - centroids[state1])
            c += 1

# Position 3
c = 0
for x in [0, 1]:
    for y in [0, 1]:
        for z in [0, 1]:
            stack1 = jnp.array([x, y, z, 0])
            stack2 = jnp.array([x, y, z, 1])

            state1 = stack_to_state_id(stack1, k, m)
            state2 = stack_to_state_id(stack2, k, m)
            ds[4] = ds[4].at[3, c].set(centroids[state2] - centroids[state1])
            c += 1

# %%

fig, axes = plt.subplots(3, 1)
for i, depth in enumerate(range(2, 5)):
    for position in range(depth):
        D = ds[depth][position]

        _, S, _ = jnp.linalg.svd(D)

        axes[i].plot(S, 'o-', label='position')

    axes[i].legend()

# %%
mean_ds = {}
for depth in range(1, 5):
    mean_ds[depth]  = ds[depth].mean(1)

# %%

def cosine(x, y, eps=1e-8):
    return jnp.dot(x, y) / (
        jnp.linalg.norm(x) * jnp.linalg.norm(y) + eps
    )

W_rec = model.rnn.rnn.W_rec

push_alignment = jnp.zeros((3,3))
pop_alignment = jnp.zeros((3,3))


for depth in range(1, m+1):
    for position in range(depth):
        d_propagated = W_rec @ mean_ds[depth][position]

        if depth != m:
            push_alignment = push_alignment.at[depth-1, position].set(
                cosine(
                d_propagated,
                mean_ds[depth + 1][position + 1]
            ))

        if depth != 1:
            pop_alignment = pop_alignment.at[depth-1, position].set(
                cosine(
                d_propagated,
                mean_ds[depth - 1][position - 1]
            ))

print(push_alignment)

print(pop_alignment)

# fig, axes = plt.subplots(2, 1)
# f = axes[0].imshow(push_alignment)
# plt.colorbar(f)

# g = axes[1].imshow(pop_alignment)
# plt.colorbar(g)

# %%

push_alignment = jnp.full((m - 1, m - 1), jnp.nan)
pop_alignment = jnp.full((m - 1, m - 1), jnp.nan)

# Push rows: source depths 1, 2, 3
for depth in range(1, m):
    for position in range(depth):
        propagated = W_rec @ mean_ds[depth][position]

        push_alignment = push_alignment.at[
            depth - 1, position
        ].set(
            cosine(
                propagated,
                mean_ds[depth + 1][position + 1],
            )
        )

# Pop rows: source depths 2, 3, 4
# Position 0 is excluded because it is popped.
for depth in range(2, m + 1):
    for position in range(1, depth):
        propagated = W_rec @ mean_ds[depth][position]

        pop_alignment = pop_alignment.at[
            depth - 2, position - 1
        ].set(
            cosine(
                propagated,
                mean_ds[depth - 1][position - 1],
            )
        )

print(push_alignment)
print(pop_alignment)

# %%

depth = 3
position = 1

test1 = jnp.full((m - 1, m - 1), jnp.nan)
for depth in range(1, m):
    for position in range(depth):
        d1 = mean_ds[depth][position]
        norm1 = jnp.linalg.norm(d1)

        d2 = mean_ds[depth + 1][position + 1]
        norm2 = jnp.linalg.norm(d2)

        test1 = test1.at[
            depth - 1, position
        ].set(
            jnp.acos(jnp.linalg.vecdot(d1, d2) / (norm1 * norm2))
        )

test2 = jnp.full((m - 1, m - 1), jnp.nan)
for depth in range(2, m + 1):
    for position in range(1, depth):
        d1 = mean_ds[depth][position]
        norm1 = jnp.linalg.norm(d1)

        d2 = mean_ds[depth - 1][position - 1]
        norm2 = jnp.linalg.norm(d2)
    
        test2 = test2.at[
            depth - 2, position - 1
        ].set(
            jnp.acos(jnp.linalg.vecdot(d1, d2) / (norm1 * norm2))
        )

print(test1)
print(test2)

# %%

def cosine(x, y, eps=1e-8):
    value = jnp.dot(x, y) / (
        jnp.linalg.norm(x) * jnp.linalg.norm(y) + eps
    )
    return jnp.clip(value, -1.0, 1.0)

for depth in range(2, m):
    for position in range(1, depth):
        d_current = mean_ds[depth][position]
        d_shallower = mean_ds[depth - 1][position - 1]
        d_deeper = mean_ds[depth + 1][position + 1]

        propagated = W_rec @ d_current

        print(
            f"depth={depth}, position={position}",
            "\n  adjacent targets:",
            cosine(d_shallower, d_deeper),
            "\n  A d vs shallower:",
            cosine(propagated, d_shallower),
            "\n  A d vs deeper:",
            cosine(propagated, d_deeper),
        )
# %%
