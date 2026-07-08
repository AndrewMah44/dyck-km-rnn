#%%
# ==== Imports ====
import jax
jax.config.update("jax_enable_x64", True)

import matplotlib as mpl
# Force Matplotlib to use TrueType fonts instead of Type 3
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

import json
import yaml
import jax.numpy as jnp
import jax.random as jr
from pathlib import Path
import matplotlib.pyplot as plt
from jax.scipy.special import rel_entr
from dyck_rnn.data.samplers import powerlaw

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
        'test_size': 1_000,
    }
}


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

def get_close_conditional_prob(model_probs, seqs, k):
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
                seqs.shape)
    
    return confidence, confidence_mat
#%%


run_name = "DyckKM_k02_m04_linear_h24_mlp1"
# run_name = "DyckKM_k02_m04_linear_h24_mlp1_adamw"
# run_name = "DyckKM_k02_m04_linear_h24_mlp0"
# run_name = "DyckKM_k02_m04_gru_h24_mlp0"
# run_name = "DyckKM_k02_m04_linear_h24_mlp2"
# run_name = "DyckKM_k02_m04_linear_h64_mlp1"

# ==== Load odel Configs ====
model_dir = "/Users/amah/Documents/GitHub/dyck-km-rnn/runs/"
run_dir = Path("/Users/amah/Documents/GitHub/dyck-km-rnn/runs/")
with open(run_dir /run_name / "run_00/config.yaml", "r") as file:
    model_config = yaml.safe_load(file)

# ==== Generate Test Dataset ====
master_key = jr.PRNGKey(eval_config['experiment']['seed'])
data_key, length_key, sample_key = jr.split(master_key,3)

k = model_config['data']['k']
m = model_config['data']['m']
max_length = 75*(4 * m * (m + 4)) 

DyckHMM = dyck_hmm(k, m, build_full_model=True)

lengths = powerlaw(
    length_key, 
    15, 
    100*(4 * m * (m + 4)), 
    model_config['data']['alpha'], 
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
opt_probs = DyckHMM.batch_one_step_prediction(sequences).transpose(0, 2, 1)

models = []
mean_conditional_prob_mat = jnp.zeros((10,1000,64))
std_conditional_prob_mat = jnp.zeros((10,1000,64))

final_validation_loss = jnp.zeros(10)
for run in range(10):
    model = load_model(
        run_name + f"/run_{run:02}", 
        run_parent_dir = model_dir)
    models.append(model)

    # ==== Run model of data ====
    model_probs = jax.nn.softmax(jax.vmap(model)(sequences))

    # =======================================
    # Figure 1 - Checking Probability outputs
    # =======================================
    d_kl_mat = jnp.where(mask, rel_entr(opt_probs, model_probs).sum(2), jnp.nan)
    d_kl_mat = jnp.where(states!=0, d_kl_mat, jnp.nan)

    fig, axes = plt.subplots(2, k+1, figsize=(10, 6))

    cond = states == 0
    x = jnp.arange(2*k + 1)
    y = jnp.nanmean(model_probs[cond], axis=0)[:-1]
    yerr = jnp.nanstd(model_probs[cond], axis=0)[:-1]
    y_opt = opt_probs[cond].mean(0)

    for i in range(k):
        # ==== Solid Blue States ====
        cond1 = (states % 2 == i) & (states < k ** m - 1) & (states != 0)
        x = jnp.arange(2*k + 1)
        y = jnp.nanmean(model_probs[cond1], axis=0)[:-1]
        yerr = jnp.nanstd(model_probs[cond1], axis=0)[:-1]

        y_opt = opt_probs[cond1].mean(0)

        axes[0,i].plot(x, y_opt[:-1], color='k')
        axes[0,i].plot(x, y)
        axes[0,i].fill_between(x, y+yerr, y-yerr, alpha=0.25)
        axes[0,i].set_title(f'{i}, Non-terminal')

        # ==== Solid Red States ====
        cond2 = (states % 2 == i) & (states >= k ** m - 1) & (states < k ** (m+1) - 1)
        x = jnp.arange(6)[:-1]
        y = jnp.nanmean(model_probs[cond2], axis=0)[:-1]
        yerr = jnp.nanstd(model_probs[cond2], axis=0)[:-1]

        y_opt = opt_probs[cond2].mean(0)

        axes[1,i].plot(x, y_opt[:-1], color='k')
        axes[1,i].plot(x, y)
        axes[1,i].fill_between(x, y+yerr, y-yerr, alpha=0.25)
        axes[1,i].set_title(f'{i}, Terminal')

    # ==== D_KL Histogram ====
    axes[0,k].hist(d_kl_mat.flatten())
    axes[0,k].set_yscale('log')

    axes[0,k].set_xlabel('D_KL')
    axes[0,k].set_ylabel('Count')

    # ==== D_KL By Sequence Position ====
    axes[1,k].plot(jnp.arange(1, max_length+1), jnp.nanmedian(d_kl_mat, axis=0))

    axes[1,k].axvline(4 * m * (m + 4), color='k', linestyle='--')

    axes[1,k].set_yscale('log')
    axes[1,k].set_xscale('log')

    axes[1,k].set_xlabel('Token Position')
    axes[1,k].set_ylabel('Median D_KL')

    fig.suptitle(f'{run}\nRaw Model Outputs')
    fig.tight_layout()

    # ==================================
    # Figure 2 - Close coditional Probs.
    # ==================================
    conditional_prob, conditional_prob_mat = get_close_conditional_prob(
        model_probs, sequences, k)

    num_states = int((k ** (m+1) - 1) / (k - 1))
    ages = jax.vmap(close_ages, in_axes=[0,0,None,None])(
        sequences, states, num_states, k)
    ages = ages[(sequences >= k) & (sequences < 2*k)]

    # ==== Plots ====
    fig, ax = plt.subplots(3, 1)

    # Confidence histogram
    ax[0].hist(conditional_prob)

    ax[0].set_xlabel('Conditional Prob.')
    ax[0].set_ylabel('N (observations)')

    ax[0].set_yscale('log')
    ax[0].set_title(f"Mean: {jnp.mean(conditional_prob)}")

    # Confidence as a function of age
    n_bins = 50
    bins = jnp.linspace(0, jnp.max(ages), n_bins)
    idx = jnp.digitize(ages, bins)

    y = jnp.array(
        [jnp.nanmean(conditional_prob[idx==i]) for i in range(n_bins)])
    yerr = jnp.array(
        [jnp.nanstd(conditional_prob[idx==i]) for i in range(n_bins)])
    counts = jnp.array(
        [jnp.sum(idx==i) for i in range(n_bins)])

    ax[1].errorbar(bins, y, yerr/jnp.sqrt(counts), marker='o', linestyle='None')
    ax[1].set_xlabel('Bracket age')
    ax[1].set_ylabel('Conditional Prob.')

    N_bins = 2**6
    binned = conditional_prob_mat.reshape(
        conditional_prob_mat.shape[0], N_bins, -1)
    trial_bin_means = jnp.nanmean(binned, axis=2)
    counts = jnp.sum(~jnp.isnan(binned), axis=[0,2])

    mean_conditional_prob_mat = mean_conditional_prob_mat.at[run].set(
        trial_bin_means
    )

    x = jnp.linspace(0, max_length, N_bins)
    y = jnp.nanmean(trial_bin_means, axis=0)
    yerr = jnp.nanstd(trial_bin_means, axis=0)

    ax[2].plot(x, y)
    ax[2].fill_between(x, y+yerr, y-yerr, alpha=0.25)
    ax[2].axvline(4 * m * (m + 4))

    ax[2].set_xlabel('Sequence position')
    ax[2].set_ylabel('Conditional Prob.')

    fig.suptitle(f'{run}\nClose Token Conditional Prob.')
    fig.tight_layout()

    with open(run_dir / run_name / f"run_{run:02}/metrics.json", "r") as file:
        a = json.load(file)

    print(f"{run}: {a['validation_loss_history'][-1]}")
    final_validation_loss = final_validation_loss.at[run].set(
        a['validation_loss_history'][-1]
    )


# %% Average Close Conditional Prob over Seeds - by seq position

x = jnp.linspace(0, max_length, N_bins)

for i in range(10):
    y = jnp.nanmean(mean_conditional_prob_mat[i], axis=0)
    yerr = jnp.nanstd(mean_conditional_prob_mat[i], axis=0)

    plt.plot(x, y)
    plt.fill_between(x, y+yerr, y-yerr, alpha=0.25)

plt.ylabel('True Close Conditional Prob.')
plt.xlabel('')

# %% Median D_KL over Seeds - by seq position

d_kl_mats = jnp.zeros((10, 1000, 9600))
for run in range(10):
    model = models[run]
    
    # ==== Run model of data ====
    model_probs = jax.nn.softmax(jax.vmap(model)(sequences))

    d_kl_mat = jnp.where(mask, rel_entr(opt_probs, model_probs).sum(2), jnp.nan)
    d_kl_mat = jnp.where(states!=0, d_kl_mat, jnp.nan)

    d_kl_mats = d_kl_mats.at[run].set(d_kl_mat)

d_kl_mats_binned = d_kl_mats.reshape([10, 1000, -1, (4 * m * (m + 4))])

x = jnp.linspace(1, d_kl_mats.shape[-1], 75)
y = jnp.nanmedian(d_kl_mats_binned, axis=(1,3))
q25, q75 = jnp.nanpercentile(
    d_kl_mats_binned,
    jnp.array([25, 75]),
    axis=(1, 3),
)

for i in range(10):
    plt.plot(x, y[i], label=f'Model {i}')
    plt.fill_between(x, q25[i], q75[i], alpha=0.25)

plt.xlabel('Token Position')
plt.ylabel('Median D_KL')

plt.title(f'Dyck-({k},{m})\nLinear + MLP')
plt.axvline(4 * m * (m+4))
plt.xscale("log")
plt.yscale("log")


# %% Binned Median D_KL over Seeds - in- vs. out-of-distribution

colors=['k', 'b', 'r', 'g', 'm'] * 2
for idx in range(9):
    data1 = d_kl_mats[idx,:,:4 * m * (m + 4)]
    q25, q75 = jnp.nanpercentile(data1, jnp.array([25, 75]))
    plt.errorbar(idx-0.2, 
                jnp.nanmedian(data1), 
                jnp.array([q25, q75])[:,None],
                marker='o',
                color=colors[idx],
                label='Early')

    data2 = d_kl_mats[idx,:,4 * m * (m + 4):]
    q25, q75 = jnp.nanpercentile(data2, jnp.array([25, 75]))
    plt.errorbar(idx+0.2, 
                jnp.nanmedian(data2), 
                jnp.array([q25, q75])[:,None],
                marker='v',
                color=colors[idx],
                label='Late')
      
    plt.plot([idx-0.2, idx+0.2], 
             [jnp.nanmedian(data1),jnp.nanmedian(data2)],
             color=colors[idx])

plt.xticks(range(9))
plt.yscale('log')

plt.ylabel('Median D_KL')
plt.xlabel('Model')
plt.title('Linear + MLP Model')

