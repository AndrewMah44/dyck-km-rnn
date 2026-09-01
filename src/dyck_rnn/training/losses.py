import jax
import jax.numpy as jnp
from jax.scipy.special import kl_div
from optax import softmax_cross_entropy_with_integer_labels

def pred_loss_func(model, x, y, mask):
    logits = model(x)
    loss = softmax_cross_entropy_with_integer_labels(logits, y) * mask

    return jnp.sum(loss) / jnp.sum(mask)


@jax.jit
def _kl_div(model, sequence, opt_prob):
    log_p = jax.nn.log_softmax(model(sequence))
    return jnp.sum(opt_prob * (jnp.log(opt_prob + 1e-20) - log_p), axis=-1)

@jax.jit
def kl_div(model, opt_probs, sequences, states, masks):
    d_kl_mat = jax.vmap(_kl_div, in_axes=(None, 0, 0))(
        model, sequences, opt_probs)

    valid = masks & (states != 0)
    return jnp.where(valid, d_kl_mat, jnp.nan)

@jax.jit
def _kl_div2(model_probs, opt_prob):
    log_p = jax.nn.log_softmax(model_probs)
    return jnp.sum(opt_prob * (jnp.log(opt_prob + 1e-20) - log_p), axis=-1)

@jax.jit
def kl_div2(model_probs, opt_probs, states, masks):
    d_kl_mat = jax.vmap(_kl_div2)(model_probs, opt_probs)

    valid = masks & (states != 0)
    return jnp.where(valid, d_kl_mat, jnp.nan)