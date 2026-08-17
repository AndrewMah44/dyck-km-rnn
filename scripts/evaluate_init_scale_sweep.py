#%%
# ==== Imports ====
import jax
jax.config.update("jax_enable_x64", True)

import matplotlib as mpl
# Force Matplotlib to use TrueType fonts instead of Type 3
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

import yaml
import jax.numpy as jnp
import jax.random as jr
from pathlib import Path
import matplotlib.pyplot as plt
from jax.scipy.special import rel_entr
from dyck_rnn.data.samplers import powerlaw

from dyck_rnn.data.dyck_hmm import dyck_hmm
from dyck_rnn.data.load_model import load_model
from dyck_rnn.training.losses import pred_loss_func

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
        'test_size': 10_000,
    }
}

# ==== Load Run Configs ====
model_dir = "/Users/amah/Documents/GitHub/dyck-km-rnn/runs/"
run = 0

# ==== Load Linear Model ====
run_name = "DyckKM_k02_m04_Linear_h24_mlp1_InitScaleSweep"
run_dir = Path(model_dir) / run_name
fit_folders = [f.name for f in run_dir.iterdir() if f.is_dir()]
fit_folders.sort()

# ==== Load GRU ====
# gru_run_name = "DyckKM_k02_m04_LSTM_h12_mlp0"
# gru_model = load_model(gru_run_name + f'/run_{run:02}', 
#                 run_parent_dir = model_dir)

def loss_func(model, obs, next_obs, mask, key, inference=False):
    keys = jr.split(key, obs.shape[0])
    pred_loss = jax.vmap(pred_loss_func, 
                        in_axes=[None, 0, 0, 0, 0, None])(
                        model, obs, next_obs, mask, keys, inference)

    return pred_loss.mean()

losses = jnp.zeros((len(fit_folders),))   
#%%
for i, folder in enumerate(fit_folders):
    fit_dir = run_dir / folder

    with open(fit_dir / "config.yaml", "r") as file:
        model_config = yaml.safe_load(file)

    model = load_model(fit_dir, 
                    run_parent_dir = model_dir)

    # ==== Generate Test Dataset ====
    master_key = jr.PRNGKey(1114)
    data_key, length_key, sample_key = jr.split(master_key,3)

    k = model_config['data']['k']
    m = model_config['data']['m']
    max_length = 25 * (4 * m * (m + 4))

    DyckHMM = dyck_hmm(k, m)

    lengths = powerlaw(
        length_key, 
        15, 
        max_length, 
        model_config['data']['alpha'], 
        shape=(eval_config['data']['test_size'],)
    )

    states, sequences = DyckHMM.batch_sample_sequence(
        batch_size = eval_config['data']['test_size'], 
        num_timesteps = max_length, 
        min_length = lengths, 
        key = data_key)
    mask = (sequences < 2 * DyckHMM.k + 1)

    # ==== Run model of data ====
    model_probs = jax.nn.softmax(jax.vmap(model)(sequences))
    # gru_probs = jax.nn.softmax(jax.vmap(gru_model)(sequences))

    # ==== Get loss ====
    x = sequences[:,:-1]
    y = sequences[:,1:]
    mask = x != (2 * k + 1)

    loss = loss_func(
        model, 
        x, 
        y, 
        mask,
        jr.PRNGKey(1),
        inference=True)
    losses = losses.at[i].set(loss)
    
    # ==== Confidence Metrics ====
    def get_close_conditional_prob(model_probs, seqs):
        p = model_probs.reshape(-1, 2*k + 2)
        s = seqs.flatten()

        close_bool = (s >= k) & (s < 2*k)
        close_idx = jnp.where(close_bool)[0]

        true_close_probs = jnp.take_along_axis(
            p[close_idx-1], 
            jnp.expand_dims(s[close_idx], 1),
            1).flatten()

        total_close_probs = p[close_idx - 1, k:2*k].sum(1)

        confidence = true_close_probs / total_close_probs
        confidence_mat = jnp.full_like(
            s, jnp.nan, dtype=jnp.float64).at[close_idx].set(
                confidence).reshape(
                    sequences.shape)
        
        return confidence, confidence_mat

    def close_ages(tokens, states, num_states, k):
        """
        tokens: shape [T], integer token ids
        states: shape [T], integer state id at each token position
                (state before/at processing token t, depending on your convention)
        num_states: total number of possible stack states
        open_token_max: tokens < open_token_max are opens, >= open_token_max are closes
        """

        T = tokens.shape[0]
        is_close = (tokens >= k) & (tokens < 2*k)
        t_idx = jnp.arange(T, dtype=jnp.int32)

        # last_seen[s] = most recent position where state s was observed
        init_last_seen = -jnp.ones((num_states,), dtype=jnp.int32)

        def step(last_seen, x):
            t, s = x
            prev_t = last_seen[s]
            last_seen = last_seen.at[s].set(t)
            return last_seen, prev_t

        _, prev_t = jax.lax.scan(step, init_last_seen, (t_idx, states))
        age = t_idx - prev_t - 1

        return jnp.where(is_close, age, 0)

    conditional_prob, conditional_prob_mat = get_close_conditional_prob(
        model_probs, sequences)

    num_states = int((k ** (m+1) - 1) / (k - 1))
    ages = jax.vmap(close_ages, in_axes=[0,0,None,None])(
        sequences, states, num_states, k)
    ages = ages[(sequences >= k) & (sequences < 2*k)]

    ev = jnp.linalg.eigvals(model.rnn.rnn.W_rec)

    # ==== Plots ====
    fig, axes = plt.subplots(2, 2, figsize=(6,6))
    # Create figure and axis

    circle = plt.Circle((0, 0), 1, color='k', fill=False, linewidth=2)

    # Add the circle patch to the axis
    axes[0,0].add_patch(circle)
    axes[0,0].scatter(jnp.real(ev), jnp.imag(ev))

    axes[0,0].set_aspect('equal')
    axes[0,0].set_xlim([-1.1, 1.1])
    axes[0,0].set_ylim([-1.1, 1.1])

    # Confidence histogram
    axes[0,1].hist(conditional_prob)

    axes[0,1].set_xlabel('Correct Close Conditional Probability')
    axes[0,1].set_ylabel('N (observations)')

    axes[0,1].set_yscale('log')
    axes[0,1].set_title(f"Median: {jnp.median(conditional_prob)}")

    # Confidence as a function of age
    n_bins = 25
    bins = jnp.linspace(0, jnp.max(ages), n_bins)
    idx = jnp.digitize(ages, bins)

    y = jnp.array(
        [jnp.nanmean(conditional_prob[idx==i]) for i in range(n_bins)])
    yerr = jnp.array(
        [jnp.nanstd(conditional_prob[idx==i]) for i in range(n_bins)])
    counts = jnp.array(
        [jnp.sum(idx==i) for i in range(n_bins)])

    axes[1,0].errorbar(bins, y, yerr/jnp.sqrt(counts), marker='o', linestyle='None')
    axes[1,0].set_xlabel('Bracket age')
    axes[1,0].set_ylabel('Confidence')

    N_bins = 2**4
    binned = conditional_prob_mat.reshape(
        conditional_prob_mat.shape[0], N_bins, -1)
    trial_bin_means = jnp.nanmean(binned, axis=2)
    counts = jnp.sum(~jnp.isnan(binned), axis=[0,2])

    x = jnp.linspace(0, max_length, N_bins)
    y = jnp.nanmean(trial_bin_means, axis=0)
    yerr = jnp.nanstd(trial_bin_means, axis=0) / jnp.sqrt(counts)

    axes[1,1].plot(x, y)
    axes[1,1].fill_between(x, y+yerr, y-yerr, alpha=0.5)
    axes[1,1].axvline(4 * m * (m + 4))

    axes[1,1].set_xlabel('Sequence position')
    axes[1,1].set_ylabel('Confidence')

    fig.suptitle(folder)
    fig.tight_layout()
    plt.show()

    # L = jnp.mean(rel_entr(model_probs, gru_probs).sum(-1)[sequences < 2*k])
    # print(-jnp.log(0.05) / L)

#%%
x = jnp.arange(2*k + 2)
y = jnp.mean(gru_probs[states == 1], axis=0)
yerr = jnp.std(gru_probs[states == 1], axis=0)
plt.plot(x, y)
plt.fill_between(x, y+yerr, y-yerr, alpha=0.3)

# %%
x = jnp.arange(2*k + 2)
y = jnp.mean(model_probs[states == 1], axis=0)
yerr = jnp.std(model_probs[states == 1], axis=0)
plt.plot(x, y)
plt.fill_between(x, y+yerr, y-yerr, alpha=0.3)
# %%
