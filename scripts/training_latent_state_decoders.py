#%% Imports
# import sys
# sys.path.append('../..')

# from config.load_config import load_paths_config
# from hmmrnn.tasks.dyck_hmm import dyck_hmm
# from hmmrnn.load_trained_model import load_trained_model

import jax
jax.config.update("jax_enable_x64", True)

import yaml
import pickle as pkl
from tqdm import trange 
import jax.numpy as jnp
import jax.random as jr
from pathlib import Path
from sklearn.linear_model import LogisticRegression, LinearRegression

import matplotlib.pyplot as plt

from dyck_rnn.data.dyck_hmm import dyck_hmm
from dyck_rnn.data.get_depth import get_depth
from dyck_rnn.data.load_model import load_model


#%%

# rnndir = paths["rnn_data_dir"] / "Dyck_km_EpsilonSoftened"
# k, m = 2, 5 

# SAVEDIR = "/Users/amah/Desktop/"
# FILENAME = "RNN_MLP_2_5_PowerLaw_64_0.8_" \
#     + "Orthogonal_Adam_GeneratedBatchesSeparately3"

# num_timesteps = 270
# test_data_key = jr.PRNGKey(1114)
# test_size = 10_000

training_config = {
    'experiment': {
        'seed': 1114
    },

    'model': {
        'run_name': 'DyckKM_k02_m05_Linear_h64_mlp1',
        'n_runs': 5
    },

    'data': {
        'num_timesteps': 270,
        'train_size': 10_000,
        'test_size': 2_000}
}

run = 0

run_dir = Path("/Users/amah/Documents/GitHub/dyck-km-rnn/runs/") \
    / training_config['model']['run_name'] / f"run_{run:02}"

with open(run_dir / "config.yaml", "r") as file:
    model_config = yaml.safe_load(file)

model = load_model(training_config['model']['run_name'])
#%%



# Code Blueprint

# Load configuration file
    # final.eqx path
    # run configs

    # Data parameters (number of timesteps, size, etc)
    # Just generate two datasets, training and testing 
        # maybe just run config seed?

# Import util functions from src

# Generate training and testing datasets
master_key = jr.PRNGKey(training_config['experiment']['seed'])

train_key, test_key = jr.split(master_key, 2)
DyckHMM = dyck_hmm(
    model_config['data']['k'],
    model_config['data']['m'])

train_states, train_sequences = DyckHMM.batch_sample_sequence(
    batch_size = training_config['data']['train_size'], 
    num_timesteps = training_config['data']['num_timesteps'], 
    min_length = 15, 
    key = train_key)

test_states, test_sequences = DyckHMM.batch_sample_sequence(
    batch_size = training_config['data']['test_size'], 
    num_timesteps = training_config['data']['num_timesteps'], 
    min_length = 15, 
    key = train_key)


# ==== Depth ====
train_depth_mat = get_depth(
    train_states,
    model_config['data']['k'],
    model_config['data']['m'])
train_depth_vec = train_depth_mat[train_depth_mat >= 0]

test_depth_mat = get_depth(
    test_states,
    model_config['data']['k'],
    model_config['data']['m']).flatten()   
test_depth_vec = test_depth_mat[test_depth_mat >= 0]

#%%

# Pull relevant parameters

# Train decoders

# Metrics: train and test performance
# Controls: Copycat control?

# Save decoders as:
#   runs/<run_folder>/decoders/<decoder_type>_<data>.eqx
# Save metrics as:
#   runs/<run_folder>/decoders/decoder_metrics.json


#%%
# ===== Util Functions =====



reg = LogisticRegression().fit(X_train, y_train)
perf = reg.score(X_test, y_test)

reg = LinearRegression(fit_intercept=True).fit(X_train, y_train)
perf = reg.score(X_test, y_test)




# ===== Load models and data =====
linear_mlp, _ = load_trained_model(f"{rnndir}/{FILENAME}")
rnn = linear_mlp.rnn
readout = jax.vmap(linear_mlp.mlp)

# Load HMM dataset
masked_kl_div = jax.vmap(_masked_kl_div)

DyckHMM = dyck_hmm(k, m, True)

# Generate test set
test_states, test_sequences = DyckHMM.batch_sample_sequence(
    test_size, num_timesteps, 15, key=test_data_key
)
test_mask = (test_sequences < 2*k).astype(jnp.int32)
seq_lengths = jnp.sum(test_mask, axis=1)

opt_posterior = DyckHMM.batch_one_step_prediction(
    test_sequences).transpose(0, 2, 1)

# ===== Simulate RNN and pull relevant variables =====
num_units = rnn.hidden_size

# Pull non-padded trials
usethese = test_sequences.flatten() < 2*k

hidden_states_mat = jax.vmap(rnn)(test_sequences)
hidden_states = hidden_states_mat.reshape(-1, num_units)[usethese]
sequences = test_sequences.flatten()[usethese]

# ===== Pull Relevant Task Variables =====
# Identity of the latent state
latent_state = test_states.flatten()[usethese]

# Depth of the latent state
latent_depth = jax.vmap(get_depth)(latent_state)

# Trial number
trial_num = jnp.cumsum(test_mask, axis=1).flatten()[usethese]

# Max probability token of each state
max_prob_next_token = jnp.argmax(opt_posterior, axis=2).flatten()[usethese]

# Current top of stack
stack_top_by_state = jnp.argmax(DyckHMM.K_dyck.todense().sum(2), axis=0) - 1
stack_top_by_state = stack_top_by_state.at[0].set(0)
stack_top = stack_top_by_state[latent_state]

# ===== Decoding Latent State Features =====
# Decode Stack Depth
rnn_reg_depth_perf, rnn_depth_decoder = \
    train_logistic_regression_decoder(hidden_states, latent_depth)

rnn_linreg_depth_perf, rnn_depth_linear_decoder = \
    train_linear_decoder(hidden_states, latent_depth)

# Decode Top-of-Stack
rnn_reg_stack_top_perf, rnn_stack_top_decoder = \
    train_logistic_regression_decoder(hidden_states, stack_top)

# Decode Latent State
rnn_reg_state_perf, rnn_state_decoder = \
    train_logistic_regression_decoder(hidden_states, latent_state)

# Decode Max Prob Token
rnn_reg_max_prob_perf, rnn_max_prob_decoder = \
    train_logistic_regression_decoder(hidden_states, max_prob_next_token)

# ===== Decoding Previous Tokens =====
hidden_states_mat = jax.vmap(rnn)(test_sequences)

N = 64
prev_token_decoder_perf = jnp.zeros(N)

prev_token_decoders = []
for n_back in trange(1, N+1):
    x = hidden_states_mat[:,n_back:,:].reshape(-1, 64)
    y = test_sequences[:,n_back:].flatten()
    y_shift = test_sequences[:,:-n_back].flatten()

    valid_shifts = (y < 2*k+1) & (y_shift < 2*k+1)
    x = x[valid_shifts]
    y_shift = y_shift[valid_shifts]

    I = int(x.shape[0] * 0.8)
    X_train = x[:I] 
    X_test = x[I:]

    y_train = y_shift[:I]
    y_test = y_shift[I:]

    decoder = LogisticRegression().fit(X_train, y_train)
    prev_token_decoders.append(decoder)
    prev_token_decoder_perf = prev_token_decoder_perf.at[n_back-1].set(
        decoder.score(X_test, y_test))

# ===== Controls =====
# Decoding Latent State Features - Copycat
xpad = jnp.pad(sequences, (64 - 1, 0), constant_values=False)

def get_window(i):
    return jax.lax.dynamic_slice(xpad, (i,), (64,))
copycat_activity = jax.vmap(get_window)(jnp.arange(sequences.shape[0]))

cc_reg_depth_perf, cc_depth_decoder = \
    train_logistic_regression_decoder(copycat_activity, latent_depth)

# Decode Top-of-Stack
cc_reg_stack_top_perf, cc_stack_top_decoder = \
    train_logistic_regression_decoder(copycat_activity, stack_top)

# Decode Latent State
cc_reg_state_perf, cc_state_decoder = \
    train_logistic_regression_decoder(copycat_activity, latent_state)

# Decode Max Prob Token
cc_reg_max_prob_perf, cc_max_prob_decoder = \
    train_logistic_regression_decoder(copycat_activity, max_prob_next_token)
    
# ===== Save Decoders =====
with open(f'{SAVEDIR}/LinearDecoders_{FILENAME}.pkl', "wb") as file:
    pkl.dump({'Prev_tokens': prev_token_decoders,
              'Depth': rnn_depth_decoder,
              'Depth_Linear': rnn_depth_linear_decoder,
              'Stack_top': rnn_stack_top_decoder,
              'State': rnn_state_decoder,
              'Max_Prob_Token': rnn_max_prob_decoder}, file)

with open(f'{SAVEDIR}/LinearDecoders_Copycat.pkl', "wb") as file:
    pkl.dump({'Prev_tokens': prev_token_decoders,
              'Depth': rnn_depth_decoder,
              'Stack_top': rnn_stack_top_decoder,
              'State': rnn_state_decoder,
              'Max_Prob_Token': rnn_max_prob_decoder}, file)
    
# ===== Plots/Tables =====
print("True RNN Activity")
print(f"Depth Decoder (% Correct): {rnn_reg_depth_perf}")
print(f"Stack Top Decoder (% Correct): {rnn_reg_stack_top_perf}")
print(f"State Identity Decoder (% Correct): {rnn_reg_state_perf}")
print(f"Max Prob. Token Decoder (% Correct): {rnn_reg_max_prob_perf}")
print(f"Last Token Decoder (% Correct): {prev_token_decoder_perf[0]}")

print("Copycat Activity")
print(f"Depth Decoder (% Correct): {cc_reg_depth_perf}")
print(f"Stack Top Decoder (% Correct): {cc_reg_stack_top_perf}")
print(f"State Identity Decoder (% Correct): {cc_reg_state_perf}")
print(f"Max Prob. Token Decoder (% Correct): {cc_reg_max_prob_perf}")

plt.plot(range(1, N+1), prev_token_decoder_perf)
plt.xlabel('N tokens back')
plt.ylabel('Decoder performance')

plt.axhline(1/(2*k), color='k', linestyle='--')
