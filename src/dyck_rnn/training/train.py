import jax
jax.config.update("jax_enable_x64", True)

import time 
import json
import yaml
import optax
import socket
import equinox as eqx
import jax.random as jr
from pathlib import Path
from optax import adam, adamw
from datetime import datetime

from dyck_rnn.data.dyck_hmm import dyck_hmm
from dyck_rnn.data.samplers import powerlaw
from dyck_rnn.data.save_model import save_model
from dyck_rnn.training.losses import pred_loss_func
from dyck_rnn.training.epochs import train_one_epoch
from dyck_rnn.models.rnn import RecurrentSequenceModel
from dyck_rnn.models.transformer import TinyTransformer

def train_dyck_rnn(run_name, config, run_parent="runs"):
    # ====== Set Up Directory ======
    run_dir = Path(run_parent) / run_name
    checkpoint_dir = run_dir / "checkpoints"

    # Stop if model has already been fit
    if (run_dir / "metrics.json").exists() or (run_dir / "final.eqx").exists():
        print(f"Skipping completed run {run_name}")
        return
    
    print(f"Fitting {run_name}...")
    
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ==== PRNGKey Management ====
    master_key = jr.PRNGKey(config['experiment']['seed'])
    validation_key, model_key, training_key = jr.split(master_key, 3)

    # ==== Initalize DyckHMM for Data Generation ====
    k = config['data']['k']
    m = config['data']['m']
    DyckHMM = dyck_hmm(k, m)

    # Bounds taken from Hewitt
    config['data']['max_length'] = 4*m*(m+4)

    sample_func = lambda lengths, key: \
        DyckHMM.batch_sample_sequence(
            config['training']['batch_size'],
            config['data']['max_length'],
            lengths,
            key = key
        )

    # ===== Generate Validation Data =====
    length_key, sample_key = jr.split(validation_key)
    validation_lengths = powerlaw(
        length_key, 
        15, 
        config['data']['max_length'], 
        config['data']['alpha'], 
        shape=(config['training']['validation_size'],)
    )   #min length of 15 ensures unlikely to overlap with training data

    _, validation_sequences = DyckHMM.batch_sample_sequence(
        config['training']['validation_size'], 
        config['data']['max_length'], 
        validation_lengths, 
        key = sample_key
    )

    validation_x = validation_sequences[:,:-1]
    validation_y = validation_sequences[:,1:]
    validation_mask = validation_x != (2 * config['data']['k'] + 1)

    # ==== Initalize model ====
    if config['model']['model_class'].lower() == 'recurrent':
        model = RecurrentSequenceModel(
            cell_type = config['model']['cell_type'],
            vocab_size = 2 * config['data']['k'] + 2, 
            hidden_size = config['model']['hidden_size'], 
            out_size = 2 * config['data']['k'] + 2, 
            readout_depth = config['model']['readout_depth'], 
            rnn_scale = config['model']['init_scale'],
            key = model_key)

        def loss_func(model, obs, next_obs, mask, key, inference=False):
            keys = jr.split(key, obs.shape[0])
            pred_loss = jax.vmap(pred_loss_func, 
                                in_axes=[None, 0, 0, 0, 0, None])(
                                model, obs, next_obs, mask, keys, inference)

            return pred_loss.mean()
    
    elif config['model']['model_class'].lower() == 'transformer':
        model = TinyTransformer(
            vocab_size = 2 * config['data']['k'] + 2, 
            num_blocks = config['model']['num_blocks'], 
            num_heads = config['model']['num_heads'],
            embedding_dim = config['model']['embedding_dim'], 
            hidden_dim = config['model']['hidden_dim'],
            max_length = config['model']['max_length'],        # 450 by default
            key = model_key)
        
        loss_func = pred_loss_func

    else:
        raise ValueError(
            f"Invalid model class: {config['model']['model_class']}")

    # ==== Optimization setup ====
    learning_rate = config['optimizer']['learning_rate']
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),  # Gradient norm clipping
        optax.adam(learning_rate=learning_rate)
        )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    initial_validation_loss = loss_func(
        model, 
        validation_x, 
        validation_y, 
        validation_mask,
        jr.PRNGKey(1),
        inference=True)

    print(f"Initial Validaiton Loss: {initial_validation_loss:0.4f}")

    # ==== Initalize Metrics ====
    validation_loss_history = [initial_validation_loss]
    training_loss_history = []

    # ==== Training Loop ====
    epoch = 1
    counter = 0 
    training_start = time.time()

    while counter < config['training']['max_counter']:
        epoch_start_time = time.time()
        batch_key, length_key, step_key = jr.split(training_key, 3)

        # Sequence lengths
        epoch_lengths = powerlaw(
            length_key, 
            1, 
            config['data']['max_length'], 
            config['data']['alpha'], 
            shape=(config['training']['batches_per_epoch'],
                config['training']['batch_size'],)
        )

        _, train_sequences = jax.vmap(sample_func)(
            epoch_lengths, 
            jr.split(batch_key, 
                     config['training']['batches_per_epoch'])
        )
        epoch_x = train_sequences[:,:,:-1]
        epoch_y = train_sequences[:,:,1:]
        epoch_mask = epoch_x != (2 * DyckHMM.k + 1)

        model, opt_state, loss_history = train_one_epoch(
            model, 
            loss_func,
            epoch_x,
            epoch_y,
            epoch_mask,
            opt_state, 
            optimizer,
            step_key)

        training_loss_history.append(loss_history)

        epoch_validation_loss = loss_func(
            model, 
            validation_x, 
            validation_y, 
            validation_mask,
            jr.PRNGKey(1),
            inference=True)
        
        validation_loss_history.append(epoch_validation_loss)
        
        if epoch_validation_loss > min(validation_loss_history):
            # Decrement learning rate
            learning_rate *= 0.5

            # Re-initalize the optimizer and opt_state
            optimizer = optax.chain(
                optax.clip_by_global_norm(1.0),  # Gradient norm clipping
                optax.adam(learning_rate=learning_rate)
            )

            # Increment counter
            counter += 1
        
        else:
            # If validation loss does achieve new minimum, reset counter
            counter = 0

        epoch_end_time = time.time()
        dt = epoch_end_time - epoch_start_time
        print(f"Epoch {epoch:02d} ({dt:0.2f} s): " \
            + f"{epoch_validation_loss} " \
            + f"(counter = {counter})", flush=True)
        
        _, training_key = jr.split(step_key, 2)

        save_model(model, checkpoint_dir / f'checkpoint_{epoch:02d}')
        epoch += 1

    training_end = time.time()
    total_train_time = training_end - training_start

    # ==== Save Model ====
    save_model(model, run_dir / 'final.eqx')

    # ==== Save Metrics ====
    machine_info = {
        "hostname": socket.gethostname(),
        "backend": jax.default_backend(),
        "devices": [str(d) for d in jax.devices()],
    }

    metrics = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "total_train_time": float(total_train_time),
        "train_loss_history": [
            jax.device_get(x).tolist() for x in training_loss_history
        ],
        "validation_loss_history": [
            float(jax.device_get(x)) for x in validation_loss_history
        ],
        "best_validation_loss": float(
            jax.device_get(min(validation_loss_history))
        ),
        "machine_info": machine_info,
    }

    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Save config
    with (run_dir / "config.yaml").open("w") as f:
        yaml.safe_dump(config, f)
