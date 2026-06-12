import jax
import equinox as eqx
import jax.random as jr
from dyck_rnn.training.steps import train_step

@eqx.filter_jit
def train_one_epoch(model, 
                    loss_func,
                    epoch_x,
                    epoch_y,
                    epoch_mask,
                    opt_state, 
                    optimizer,
                    key):
    
    model_params, model_static = eqx.partition(
        model, eqx.is_inexact_array
        )

    # Function to scan over mini-batches
    def scan_step(carry, input):
        model_params, opt_state, key = carry
        batch_x, batch_y, batch_mask = input

        # Combine trainable params (scanned) with static params (not scanned)
        scan_model = eqx.combine(model_params, model_static)

        # Do one training step on the batches
        loss, scan_model, opt_state = train_step(
            loss_func, 
            scan_model, 
            batch_x, 
            batch_y, 
            batch_mask, 
            opt_state, 
            optimizer,
            key)
        
        # Split param
        new_params, _ = eqx.partition(
            scan_model, 
            eqx.is_inexact_array)
        
        _, key = jr.split(key, 2)
        return (new_params, opt_state, key), loss

    # Actually scan
    init_carry = (model_params, opt_state, key)
    (final_params, opt_state, _), loss_history = jax.lax.scan(
        scan_step, init_carry, xs = (epoch_x, epoch_y, epoch_mask,))
    
    final_model = eqx.combine(final_params, model_static)
    return final_model, opt_state, loss_history