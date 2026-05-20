import equinox as eqx

# A single training step
@eqx.filter_jit
def train_step(loss_func, model, obs, next_obs, mask, opt_state, optimizer):
    loss, grads = eqx.filter_value_and_grad(loss_func)(
        model, obs, next_obs, mask)
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)

    return loss, model, opt_state
