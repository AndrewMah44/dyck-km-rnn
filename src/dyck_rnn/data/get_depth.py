import jax.numpy as jnp

def get_depth(state, k, m):
    depth = jnp.floor(jnp.log(state*(k - 1) + 1) / jnp.log(k))
    
    return jnp.where(depth < m, depth, -1)
