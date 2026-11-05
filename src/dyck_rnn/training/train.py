import jax
jax.config.update("jax_enable_x64", True)

import time 
import json
import yaml
import optax
import socket
import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from pathlib import Path
from datetime import datetime
from dyck_rnn.data.dyck_hmm import dyck_hmm
from dyck_rnn.data.samplers import powerlaw
from dyck_rnn.data.save_model import save_model
from dyck_rnn.training.losses import pred_loss_func
from dyck_rnn.training.epochs import train_one_epoch
from dyck_rnn.models.rnn import RecurrentSequenceModel
from optax import softmax_cross_entropy_with_integer_labels, squared_error

def length_sampler(config, key):
    if config['experiment']['task'] == 'DyckKM':
        return powerlaw(
            key,
            1,
            config['data']['max_length'],
            config['data']['alpha'],
            shape = (config['training']['batches_per_epoch'],
                     config['training']['batch_size'])
        )
    elif config['experiment']['task'] in ["ModCumSum", "GaussianPred"]:
        return jnp.ones(
            shape = (config['training']['batches_per_epoch'],
                        config['training']['batch_size'],)
        ) * config['data']['max_length']


def train_loop(model, config, 
               validation_loss_func, loss_func, sample_func,
               validation_x, validation_y, validation_mask,
               checkpoint_dir, run_dir, training_key):

    # ==== Set up optimizer ====
    # Whether to enforce stable W_rec by rescaling
    if 'enforce_stable' in config['optimizer']:
        enforce_stable = config['optimizer']['enforce_stable']
    else:
        enforce_stable = False

    # Initalize optimizer (adam or adamw)
    learning_rate = config['optimizer']['learning_rate']
    if config['optimizer']['name'].lower() == 'adam':
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),  # Gradient norm clipping
            optax.adam(learning_rate=learning_rate)
            )
    elif config['optimizer']['name'].lower() == 'adamw':
        if 'weight_decay' in config['optimizer']:
            weight_decay = config['optimizer']['weight_decay']
        else:
            weight_decay = 1e-3
    
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),  # Gradient norm clipping
            optax.adamw(learning_rate=learning_rate, 
                        weight_decay=weight_decay)
        )      

    else:
        raise ValueError(
            f"{config['optimizer']['name']} is not a valid optimizer")
    
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    # ==== Initalize Metrics ====
    initial_validation_loss = validation_loss_func(
        model, 
        validation_x, 
        validation_y,
        validation_mask)

    print(f"Initial Validaiton Loss: {initial_validation_loss:0.4f}",
            flush=True)

    initial_validation_loss = float(jax.device_get(initial_validation_loss))
    validation_loss_history = [initial_validation_loss]

    training_loss_history = []

    # ==== Training Loop ====
    if 'max_epochs' in config['training']:
        max_epochs = config['training']['max_epochs']
    else:
        max_epochs = 0

    epoch = 1
    counter = 0 
    training_start = time.time()

    while counter < config['training']['max_counter'] or epoch < max_epochs:
        epoch_start_time = time.time()
        batch_key, length_key = jr.split(training_key, 2)

        lengths = length_sampler(config, length_key)

        keys = jr.split(batch_key, config['training']['batches_per_epoch'])

        model, opt_state, loss_history = train_one_epoch(
            model, 
            lengths,
            keys,
            loss_func,
            sample_func,
            opt_state, 
            optimizer,
            enforce_stable)

        training_loss_history.append(
            jax.device_get(loss_history)
        )
        
        epoch_validation_loss = validation_loss_func(
            model, 
            validation_x, 
            validation_y,
            validation_mask)
            
        epoch_validation_loss = float(jax.device_get(epoch_validation_loss))
        validation_loss_history.append(epoch_validation_loss)
        
        if epoch_validation_loss > min(validation_loss_history):
            # Decrement learning rate
            learning_rate *= 0.5

            if config['optimizer']['name'].lower() == 'adam':
                optimizer = optax.chain(
                    optax.clip_by_global_norm(1.0),  # Gradient norm clipping
                    optax.adam(learning_rate=learning_rate)
                    )
                
            elif config['optimizer']['name'].lower() == 'adamw':
                if 'weight_decay' in config['optimizer']:
                    weight_decay = config['optimizer']['weight_decay']
                else:
                    weight_decay = 1e-3
            
                optimizer = optax.chain(
                    optax.clip_by_global_norm(1.0),  # Gradient norm clipping
                    optax.adamw(learning_rate=learning_rate, 
                                weight_decay=weight_decay)
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
        
        _, training_key = jr.split(training_key, 2)

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

    return validation_loss_history[-1]

def train_dyck_rnn(run_name, config, run_parent="runs"):
    # ====== Set Up Directory ======
    run_dir = Path(run_parent) / run_name
    checkpoint_dir = run_dir / "checkpoints"

    # Stop if model has already been fit
    if (run_dir / "metrics.json").exists() or (run_dir / "final.eqx").exists():
        print(f"Skipping completed run {run_name}")
        return
    
    print(f"Fitting {run_name}...", flush=True)
    
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ==== PRNGKey Management ====
    master_key = jr.PRNGKey(config['experiment']['seed'])
    validation_key, model_key, training_key = jr.split(master_key, 3)

    # ==== Initalize DyckHMM for Data Generation ====
    k = config['data']['k']
    m = config['data']['m']
    DyckHMM = dyck_hmm(k, m)

    def sample_func(lengths, key):
        _, sequences = DyckHMM.batch_sample_sequence(
            config['training']['batch_size'],
            config['data']['max_length'],
            lengths,
            key = key
        )

        x = sequences[:,:-1]
        y = sequences[:,1:]
        mask = x != (2 * k + 1)

        return x, y, mask

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

    # Reshape to batch validation function
    validation_x = validation_sequences[:,:-1].reshape(
        2**5, -1, config['data']['max_length']-1)
    validation_y = validation_sequences[:,1:].reshape(
        2**5, -1, config['data']['max_length']-1)
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

    else:
        raise ValueError(
            f"Invalid model class: {config['model']['model_class']}")

    # ==== Optimization setup ====
    # Define loss function
    def loss_func(model, obs, next_obs, mask):
        pred_loss = jax.vmap(pred_loss_func, 
                            in_axes=[None, 0, 0, 0])(
                            model, obs, next_obs, mask)

        return pred_loss.mean()

    def validation_loss_func(model, obs, next_obs, masks):
        def f(loss, input):
            x, y, mask = input
            loss += loss_func(model, x, y, mask)

            return loss, loss

        loss_sum, _ = jax.lax.scan(f, 0, xs = (obs, next_obs, masks))
        n_batches = obs.shape[0]

        return loss_sum / n_batches
    
    # Run training loop
    validation_loss_history = train_loop(model, config, 
        validation_loss_func, loss_func, sample_func,
        validation_x, validation_y, validation_mask, 
        checkpoint_dir, run_dir, training_key)

    return validation_loss_history

def train_modular_cumsum_rnn(run_name, config, run_parent="runs"):
    # ====== Set Up Directory ======
    run_dir = Path(run_parent) / run_name
    checkpoint_dir = run_dir / "checkpoints"

    # Stop if model has already been fit
    if (run_dir / "metrics.json").exists() or (run_dir / "final.eqx").exists():
        print(f"Skipping completed run {run_name}")
        return
    
    print(f"Fitting {run_name}...", flush=True)
    
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ==== PRNGKey Management ====
    master_key = jr.PRNGKey(config['experiment']['seed'])
    validation_key, model_key, training_key = jr.split(master_key, 3)

    # ==== Initalize Data Generation ====
    n = config['data']['n']

    def sample_func(lengths, key):
        x = jr.randint(
            key = key,
            minval = 0,
            maxval = n - 1,
            shape = (config['training']['batch_size'], 
                     config['data']['max_length'])
        )

        y = jnp.cumsum(x, axis=1) % n
        mask = None

        return x, y, mask

    # ===== Generate Validation Data =====    
    validation_x = jr.randint(
        key = validation_key,
        minval = 0,
        maxval = n - 1,
        shape = (config['training']['validation_size'],
                 config['data']['max_length'])
    )
    validation_y = jnp.cumsum(validation_x, axis=1) % n
    validation_mask = None

    # Requires validation_size divisible by validation_batch_size.
    validation_x = validation_x.reshape(-1, 2**5, validation_x.shape[-1])
    validation_y = validation_y.reshape(-1, 2**5, validation_x.shape[-1])

    # ==== Initalize model ====
    if config['model']['model_class'].lower() == 'recurrent':
        model = RecurrentSequenceModel(
            cell_type = config['model']['cell_type'],
            vocab_size = n, 
            hidden_size = config['model']['hidden_size'], 
            out_size = n, 
            readout_depth = config['model']['readout_depth'], 
            rnn_scale = config['model']['init_scale'],
            key = model_key)

    else:
        raise ValueError(
            f"Invalid model class: {config['model']['model_class']}")

    # ==== Optimization setup ====
    # Define loss function
    def loss_func(model, x, y, mask=None):
        logits = jax.vmap(model)(x)
        return jnp.mean(
            softmax_cross_entropy_with_integer_labels(logits, y)
        )

    @eqx.filter_jit
    def validation_loss_func(model, x, y, masks=None):
        def f(total, input):
            x, y = input
            return total + loss_func(model, x, y), None

        loss_sum, _ = jax.lax.scan(f, 0, xs = (x, y))
        n_batches = x.shape[0]

        return loss_sum / n_batches
    
    # Run training loop
    validation_loss_history = train_loop(model, config, 
        validation_loss_func, loss_func, sample_func,
        validation_x, validation_y, validation_mask, 
        checkpoint_dir, run_dir, training_key)

    return validation_loss_history

def train_gaussian_pred_rnn(run_name, config, run_parent="runs"):
    # ====== Set Up Directory ======
    run_dir = Path(run_parent) / run_name
    checkpoint_dir = run_dir / "checkpoints"

    # Stop if model has already been fit
    if (run_dir / "metrics.json").exists() or (run_dir / "final.eqx").exists():
        print(f"Skipping completed run {run_name}")
        return
    
    print(f"Fitting {run_name}...", flush=True)
    
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ==== PRNGKey Management ====
    master_key = jr.PRNGKey(config['experiment']['seed'])
    validation_key, model_key, training_key = jr.split(master_key, 3)

    # ==== Initalize Data Generation ====
    mu0 = config['data']['mu0']
    tau0 = config['data']['tau0']
    sigma = config['data']['sigma']

    def sample_func(lengths, key):
        # Sample mu ~ N(mu0, tau0**2)
        mu_key, sample_key = jr.split(key, 2)
        mus = tau0 * jr.normal(
            key = mu_key,
            shape = (config['training']['batch_size'],1)) + mu0

        # Sample x ~ N(mu, sigma**2) (sigma is constant across trials)
        samples = sigma * jr.normal(
            key = sample_key,
            shape = (config['training']['batch_size'], 
                     config['data']['max_length'])
        ) + mus

        x = samples[:,:-1, None]
        y = samples[:,1:, None]
        mask = None

        return x, y, mask

    # ===== Generate Validation Data ===== 
    valiation_mu_key, validation_sample_key = jr.split(validation_key, 2)   
    validation_mus = jr.normal(
        key = valiation_mu_key,
        shape = (config['training']['validation_size'], 1)) + mu0

    validation_samples = sigma * jr.normal(
        key = validation_sample_key,
        shape = (config['training']['validation_size'], 
                    config['data']['max_length'])
    ) + validation_mus

    validation_x = validation_samples[:,:-1, None]
    validation_y = validation_samples[:,1:, None]
    validation_mask = None

    # Requires validation_size divisible by validation_batch_size.
    validation_x = validation_x.reshape(
        -1, 2**5, config['data']['max_length']-1, 1)
    validation_y = validation_y.reshape(
        -1, 2**5, config['data']['max_length']-1, 1)

    # ==== Initalize model ====
    if config['model']['model_class'].lower() == 'recurrent':
        model = RecurrentSequenceModel(
            cell_type = config['model']['cell_type'],
            vocab_size = 1, 
            hidden_size = config['model']['hidden_size'], 
            out_size = 1, 
            readout_depth = config['model']['readout_depth'], 
            input_dtype = "cont",
            rnn_scale = config['model']['init_scale'],
            key = model_key)

    else:
        raise ValueError(
            f"Invalid model class: {config['model']['model_class']}")

    # ==== Optimization setup ====
    # Define loss function
    def loss_func(model, x, y, mask=None):
        yhat = jax.vmap(model)(x)
        return jnp.mean(
            squared_error(y, yhat)
        )

    @eqx.filter_jit
    def validation_loss_func(model, x, y, masks=None):
        def f(total, input):
            x, y = input
            return total + loss_func(model, x, y), None

        loss_sum, _ = jax.lax.scan(f, 0, xs = (x, y))
        n_batches = x.shape[0]

        return loss_sum / n_batches
    
    # Run training loop
    validation_loss_history = train_loop(model, config, 
        validation_loss_func, loss_func, sample_func,
        validation_x, validation_y, validation_mask, 
        checkpoint_dir, run_dir, training_key)

    return validation_loss_history