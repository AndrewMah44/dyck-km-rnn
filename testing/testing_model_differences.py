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

from dyck_rnn.data.dyck_hmm import dyck_hmm
from dyck_rnn.data.load_model import load_model

#%%
run_name = "DyckKM_k02_m04_linear_h24_mlp1"
run = 0

# ==== Load Linear Model ====
model_dir1 = "/Users/amah/Documents/GitHub/dyck-km-rnn/runs/old2"
model1 = load_model(
    run_name + f"/run_00", 
    run_parent_dir = model_dir1)

model_dir2 = "/Users/amah/Documents/GitHub/dyck-km-rnn/runs"
model2 = load_model(
    run_name + f"/run_00", 
    run_parent_dir = model_dir2)

# %%
fig, axes = plt.subplots(2, 1)
axes[0].imshow(model1.rnn.rnn.W_rec)
axes[1].imshow(model2.rnn.rnn.W_rec)

# %%

ev1 = jnp.linalg.eigvals(model1.rnn.rnn.W_rec)
ev2 = jnp.linalg.eigvals(model2.rnn.rnn.W_rec)

fig, axes = plt.subplots(2, 1)
axes[0].scatter(jnp.real(ev1), jnp.imag(ev1), marker='.')
axes[0].axvline(1)

axes[1].scatter(jnp.real(ev2), jnp.imag(ev2), marker='.')
axes[1].axvline(1)

# %%

plt.hist(model1.rnn.rnn.W_u.flatten())
plt.hist(model2.rnn.rnn.W_u.flatten())

# %%

# ==== Generate Test Dataset ====
master_key = jr.PRNGKey(1114)
data_key, length_key, sample_key = jr.split(master_key,3)

k = 2
m = 4
max_length = 50*(4 * m * (m + 4))

DyckHMM = dyck_hmm(k, m)

lengths = powerlaw(
    length_key, 
    15, 
    max_length, 
    0.9, 
    shape = (5_000,)
)

states, sequences = DyckHMM.batch_sample_sequence(
    batch_size = 5_000, 
    num_timesteps = max_length, 
    min_length = lengths, 
    key = data_key)
mask = sequences < 2*k + 1
seq_len = mask.sum(1)
# %%

n = 0
p1 = jax.nn.softmax(model1(sequences[n]))
p2 = jax.nn.softmax(model2(sequences[n]))

d_kl = jax.scipy.special.rel_entr(p1, p2).sum(1)

# %%

plt.plot(d_kl[:seq_len[n]])
plt.yscale('log')
plt.show()

idx = jnp.argmax(d_kl > 0.5)
plt.plot(p1[idx], 'k')
plt.plot(p2[idx], 'r--')

# %%

plt.plot(p2[:,5])
# %%
