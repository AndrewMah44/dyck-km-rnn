#%% ==== Imports ====
import jax
jax.config.update("jax_enable_x64", True)

import yaml
import argparse
import pickle as pkl
import jax.numpy as jnp
import jax.random as jr
from pathlib import Path
from copy import deepcopy
from itertools import product
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression

from dyck_rnn.data.dyck_hmm import dyck_hmm
from dyck_rnn.utils.get_depth import get_depth
from dyck_rnn.data.load_model import load_model

# ==== Utils ====
def shift_data(n_back, h, seq):
    """
    Returns shifted sequences. 
    """
    n_units = h.shape[-1]

    h_flat = h[:, n_back:, :].reshape(-1, n_units)
    seq_flat = seq[:, n_back:].flatten()
    seq_shift = seq[:, :-n_back].flatten()

    valid_shifts = (seq_flat < 2*k+1) & (seq_shift < 2*k+1)

    h_shifted = h_flat[valid_shifts]
    y_shifted = seq_shift[valid_shifts]

    return h_shifted, y_shifted

# ==== Load Configs ====
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=Path, required=True)
args = parser.parse_args()

with args.config.open("r") as f:
    training_config = yaml.safe_load(f)

# ==== Make prototypic decoder ====
if training_config['run']['scaling']:
    base_decoder = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    solver="lbfgs",
                    max_iter=5000,
                    tol=1e-4,
                    C=1.0,
                ),
            )
else:
    base_decoder = LogisticRegression(
        solver="lbfgs",
        max_iter=5000,
        tol=1e-4,
        C=1.0)
    
# ==== Training Loop ====
for (k, m, hidden_size, run) in product(
    training_config['run']['k'], 
    training_config['run']['m'],
    training_config['run']['hidden_size'],
    jnp.arange(training_config['run']['n_runs'])):

    # ==== Load Run Configs ====
    run_name = f"{training_config['run']['task']}_" \
        + f"k{k:02}_m{m:02}_" \
        + f"{training_config['run']['cell_type']}_" \
        + f"h{hidden_size}_" \
        + f"mlp{training_config['run']['readout_depth']}"

    print("\n" + run_name, run)

    # ==== Set Up Paths ====
    run_dir = Path("/Users/amah/Documents/GitHub/dyck-km-rnn/runs/") \
        / run_name / f"run_{run:02}"

    try:
        with open(run_dir / "config.yaml", "r") as file:
            model_config = yaml.safe_load(file)
    except:
        print(f"{run_dir / "config.yaml"} not found. Skipping...")
        continue

    if training_config['run']['scaling']:
        save_dir = run_dir / 'decoders'
    else:
        save_dir = run_dir / 'decoders_noscaling'

    save_dir.mkdir(parents=True, exist_ok=True)

    with (save_dir / "decoder_config.yaml").open("w") as f:
        yaml.safe_dump(training_config, f)

    # ==== Load Trained Model ====
    # model_dir = "/Users/amah/Documents/GitHub/dyck-km-rnn/runs/"
    try:
        model = load_model(run_name + f'/run_{run:02}')
    except:
        print(f"Model not found at {run_name}/run_{run:02}'. Skipping...")
        continue
    rnn = model.rnn

    # ==== Generate Datasets ====
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
    train_mask = (train_sequences < 2 * DyckHMM.k).flatten()

    test_states, test_sequences = DyckHMM.batch_sample_sequence(
        batch_size = training_config['data']['test_size'], 
        num_timesteps = training_config['data']['num_timesteps'], 
        min_length = 15, 
        key = test_key)
    test_mask = (test_sequences < 2 * DyckHMM.k).flatten()

    # ==== RNN Activity ====
    train_activity_mat = jax.vmap(rnn)(train_sequences)
    train_activity = train_activity_mat.reshape(
            -1, model_config['model']['hidden_size'])[train_mask]

    test_activity_mat = jax.vmap(rnn)(test_sequences)
    test_activity = test_activity_mat.reshape(
            -1, model_config['model']['hidden_size'])[test_mask]

    # ==== Depth Logistic Regression ====
    if (save_dir / 'depth_logistic_decoder.pkl').exists():
        print("Skipping completed depth logistic regression")

    else:
        print("Fitting depth logistic regression")
        # Get train and test depths
        train_depth = get_depth(
            train_states,
            model_config['data']['k'],
            model_config['data']['m']).flatten()
        train_depth = train_depth[train_mask]

        test_depth = get_depth(
            test_states,
            model_config['data']['k'],
            model_config['data']['m']).flatten()   
        test_depth = test_depth[test_mask]

        # Set up logistic regression with standard scaler
        depth_logistic_regression = deepcopy(base_decoder)

        # Fit and evaluate
        depth_logistic_regression.fit(train_activity, train_depth)
        depth_logistic_preformance = depth_logistic_regression.score(
            test_activity, test_depth)

        # Save
        with open(save_dir / 'depth_logistic_decoder.pkl', 'wb') as f:
            pkl.dump({
                'decoder': depth_logistic_regression,
                'percent_correct': depth_logistic_preformance},
                f)

    # ==== Depth Linear Regression ====
    if (save_dir / 'depth_linear_decoder.pkl').exists():
        print("Skipping completed depth linear regression")

    else:
        print("Fitting depth linear regression")
        depth_linear_regression = LinearRegression().fit(
            train_activity, train_depth)
        depth_linear_preformance = depth_linear_regression.score(
            test_activity, test_depth)

        with open(save_dir / 'depth_linear_decoder.pkl', 'wb') as f:
            pkl.dump({
                'decoder': depth_linear_regression,
                'r_squared': depth_linear_preformance}, f)

    # ==== Latent State ====
    if k < 8 and m < 8:
        if (save_dir / 'state_logistic_decoder.pkl').exists():
            print("Skipping completed state logistic regression")

        else:
            # Set up logistic regression with standard scaler
            state_logistic_regression = deepcopy(base_decoder)

            # Fit and evaluate
            state_logistic_regression.fit(
                train_activity, train_states.flatten()[train_mask])
            state_logistic_preformance = state_logistic_regression.score(
                test_activity, test_states.flatten()[test_mask])

            # Save
            with open(save_dir / 'state_logistic_decoder.pkl', 'wb') as f:
                pkl.dump({
                    'decoder': state_logistic_regression,
                    'percent_correct': state_logistic_preformance}, f)
    else:
        print("Skipping state logistic regression. State space too large.")

    # ===== Previous Tokens =====
    if (save_dir / 'prev_token_logistic_decoder.pkl').exists():
        print("Skipping completed previous token logistic regression")

    else:
        print("Fitting previous token logistic regression")
        
        previous_token_regression = []

        k = model_config['data']['k']
        N = 64
        prev_token_decoder_perf = jnp.zeros(N)

        prev_token_decoders = []
        prev_token_performance = []

        for n_back in range(1, N+1):
            # Pull test and train data
            X_train, y_train = shift_data(
                n_back, train_activity_mat, train_sequences)
            X_test, y_test = shift_data(
                n_back, test_activity_mat, test_sequences)

            # Set up logistic regression with standard scaler
            regression = deepcopy(base_decoder)

            # Fit and evaluate
            regression.fit(X_train, y_train)
            perf = regression.score(X_test, y_test)

            # Store
            previous_token_regression.append(regression)
            prev_token_performance.append(perf)
            
        # Save
        with open(save_dir / 'prev_token_logistic_decoder.pkl', "wb") as f:
            pkl.dump({
                'decoders': previous_token_regression,
                'percent_corrects': prev_token_performance}, f)

# # ===== Util Functions =====
# # Max probability token of each state
# max_prob_next_token = jnp.argmax(opt_posterior, axis=2).flatten()[usethese]

# # Current top of stack
# stack_top_by_state = jnp.argmax(DyckHMM.K_dyck.todense().sum(2), axis=0) - 1
# stack_top_by_state = stack_top_by_state.at[0].set(0)
# stack_top = stack_top_by_state[latent_state]