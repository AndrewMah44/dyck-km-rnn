import jax
import equinox as eqx
import jax.random as jr
from dyck_rnn.training.steps import train_step
from dyck_rnn.data.samplers import powerlaw


@eqx.filter_jit
def train_one_epoch(model, 
                    epoch_lengths,
                    keys,
                    loss_func,
                    sample_func,
                    opt_state, 
                    optimizer,
                    hmm,
                    enforce_stable):

    model_params, model_static = eqx.partition(
        model, eqx.is_inexact_array
        )

    # Function to scan over mini-batches
    def scan_step(carry, input):
        model_params, opt_state = carry
        batch_lengths, batch_key = input

        _, train_sequences = sample_func(batch_lengths, batch_key)
    
        batch_x = train_sequences[:,:,:-1]
        batch_y = train_sequences[:,:,1:]
        batch_mask = batch_x != (2 * hmm.k + 1)

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
            enforce_stable)
        
        # Split param
        new_params, _ = eqx.partition(
            scan_model, 
            eqx.is_inexact_array)
        
        return (new_params, opt_state), loss

    # Actually scan
    init_carry = (model_params, opt_state)
    (final_params, opt_state), loss_history = jax.lax.scan(
        scan_step, init_carry, xs = (epoch_lengths, keys,))
    
    final_model = eqx.combine(final_params, model_static)
    return final_model, opt_state, loss_history