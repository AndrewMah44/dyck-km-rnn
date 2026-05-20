import jax.random as jr
import jax.numpy as jnp

def powerlaw(key, low, high, alpha, shape):
    # alpha > 0; smaller = heavier tail
    u = jr.uniform(key, shape=shape)
    
    # inverse CDF for truncated power law
    a = low**(1 - alpha)
    b = high**(1 - alpha)
    x = (a + (b - a) * u) ** (1 / (1 - alpha))
    
    return jnp.clip(jnp.round(x), low, high).astype(jnp.int32)