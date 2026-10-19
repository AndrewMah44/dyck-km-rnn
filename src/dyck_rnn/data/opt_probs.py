import jax
import jax.numpy as jnp
from functools import partial

@partial(jax.jit, static_argnames=("k", "m"))
def get_opt_probs_state(state, k: int, m: int):
    state = jnp.asarray(state)
    dtype = jnp.float32
    n_out = 2 * k + 2

    is_leaf = state >= (k**m - 1) / (k - 1)
    is_non_terminal = state < (k ** (m + 1) - 1) / (k - 1)
    is_root = state == 0

    idx = k + (k + state - 1) % k

    probs_root = jnp.zeros((n_out,), dtype=dtype)
    probs_root = probs_root.at[:k].set(0.5 / k)
    probs_root = probs_root.at[-2].set(0.5)

    probs_leaf = jax.nn.one_hot(idx, n_out, dtype=dtype)

    probs_internal = jnp.zeros((n_out,), dtype=dtype)
    probs_internal = probs_internal.at[:k].set(0.5 / k)
    probs_internal = probs_internal.at[idx].set(0.5)

    probs_terminal = jax.nn.one_hot(n_out - 1, n_out, dtype=dtype)

    probs = jnp.where(
        is_root,
        probs_root,
        jnp.where(
            is_leaf & is_non_terminal,
            probs_leaf,
            jnp.where(
                (~is_leaf) & is_non_terminal,
                probs_internal,
                probs_terminal,
            ),
        ),
    )

    return probs

def get_opt_probs_seq(states, k, m):
    return jax.vmap(get_opt_probs_state, in_axes=[0, None, None])(
        states, k, m)

def get_opt_probs(states, k, m):
    return jax.vmap(get_opt_probs_seq, in_axes=[0, None, None])(states, k, m)
    