import jax
import equinox as eqx
import jax.random as jr
from dyck_rnn.training.steps import train_step

# train_step(loss_func, model, obs, next_obs, mask, opt_state, optimizer)

@eqx.filter_jit
def train_one_epoch(model, 
                    DyckHMM, 
                    loss_func,
                    batch_size, 
                    batches_per_epoch,
                    num_timesteps, 
                    lengths, 
                    opt_state, 
                    optimizer,
                    *, 
                    key):
    
    model_params, model_static = eqx.partition(
        model, eqx.is_inexact_array
        )

    # Function to scan over mini-batches
    def scan_step(carry, key):
        model_params, opt_state, idx = carry

        # Combine trainable params (scanned) with static params (not scanned)
        scan_model = eqx.combine(model_params, model_static)

        # Generate test sequences
        _, train_sequences = DyckHMM.batch_sample_sequence(
            batch_size, num_timesteps, lengths[idx], key = key)

        # Pull batch sequences and labels
        batch_x = train_sequences[:,:-1]
        batch_y = train_sequences[:,1:]
        batch_mask = batch_x != (2 * DyckHMM.k + 1)

        # Do one training step on the batches
        loss, scan_model, opt_state = train_step(
            loss_func, 
            scan_model, 
            batch_x, 
            batch_y, 
            batch_mask, 
            opt_state, 
            optimizer)
        
        # Split param
        new_params, _ = eqx.partition(
            scan_model, 
            eqx.is_inexact_array)
        
        return (new_params, opt_state, idx+1), loss

    # Actually scan
    init_carry = (model_params, opt_state, 0)
    keys = jr.split(key, batches_per_epoch)
    (final_params, opt_state, _), loss_history = jax.lax.scan(
        scan_step, init_carry, xs=keys)
    
    final_model = eqx.combine(final_params, model_static)
    return final_model, opt_state, loss_history