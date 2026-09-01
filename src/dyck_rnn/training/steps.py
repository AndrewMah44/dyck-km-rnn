import equinox as eqx
import jax.numpy as jnp

def rescale_Wrec(model, rho_max=0.99999, eps = 1e-12):
    rho = jnp.max(jnp.abs(jnp.linalg.eigvals(model.rnn.rnn.W_rec)))

    scale = jnp.minimum(
        1.0,
        rho_max / (rho + eps),
    )

    model = eqx.tree_at(
        where = lambda m: m.rnn.rnn.W_rec,
        pytree = model,
        replace_fn = lambda w: w * scale)

    return model

# A single training step
@eqx.filter_jit
def train_step(loss_func, model, x, y, mask, opt_state, optimizer, 
               enforce_stable):
    loss, grads = eqx.filter_value_and_grad(loss_func)(
        model, x, y, mask)
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)

    if enforce_stable:
        model = rescale_Wrec(model)
        
    return loss, model, opt_state
