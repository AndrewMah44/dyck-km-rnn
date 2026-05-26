import jax
import jax.numpy as jnp
from jax.scipy.special import kl_div
from optax import softmax_cross_entropy_with_integer_labels

def pred_loss_func(model, x, y, mask):
    logits = model(x)
    loss = softmax_cross_entropy_with_integer_labels(logits, y) * mask

    return jnp.sum(loss) / jnp.sum(mask)

def masked_kl_div(opt_dist, mdl_dist, mask):
    mdl_dist = mdl_dist + 1e-32
    return jnp.sum(kl_div(opt_dist, mdl_dist).sum(-1) * mask) / jnp.sum(mask)
