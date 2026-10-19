import jax
import jax.numpy as jnp

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