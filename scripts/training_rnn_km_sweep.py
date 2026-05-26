# Imports
import jax
jax.config.update("jax_enable_x64", True)

import yaml
import argparse
from time import time
from pathlib import Path
from copy import deepcopy
from itertools import product
from dyck_rnn.training.train import train_dyck_rnn

# ====== Load Configuration File ======
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=Path, required=True)
args = parser.parse_args()

with args.config.open("r") as f:
    sweep_config = yaml.safe_load(f)

task = sweep_config['experiment']['task']
n_runs = sweep_config['experiment']['n_runs']
seed = sweep_config['experiment']['seed']

# ====== Sweep over hidden size ======
for idx, (k,m) in enumerate(product(
    sweep_config['sweep']['k'],
    sweep_config['sweep']['m'])):
    
    for run in range(n_runs):
        # ====== Set Up Config File ======
        config = deepcopy(sweep_config)
        config['data']['k'] = k
        config['data']['m'] = m
        config['experiment']['seed'] = int(time())

        # ====== Set Up Paths ======
        run_name = task \
            + f'_k{config["data"]["k"]:02}' \
            + f'_m{config["data"]["m"]:02}' \
            + f'_{config["model"]["cell_type"]}' \
            + f'_h{config["model"]["hidden_size"]}' \
            + f'_mlp{config["model"]["readout_depth"]}'  \
            + f'/run_{run:02}'    

        train_dyck_rnn(run_name, config)
        
        # run_dir = Path("runs") / run_name
        # checkpoint_dir = run_dir / "checkpoints"

        # run_dir.mkdir(parents=True, exist_ok=True)
        # checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # # Check if model has already been fit
        # if (run_dir / "metrics.json").exists() or (run_dir / "final.eqx").exists():
        #     print(f"Skipping completed run {run_name}")
        #     continue

        # print(f"Fitting {run_name}")

        # with (run_dir / "config.yaml").open("w") as f:
        #     yaml.safe_dump(config, f)

        # try:
        #     # ==== PRNGKey Management ====
        #     master_key = jr.PRNGKey(sweep_config['experiment']['seed'])
        #     validation_key, model_key, training_key = jr.split(master_key, 3)

        #     # ==== Initalize DyckHMM for Data Generation ====
        #     DyckHMM = dyck_hmm(config['data']['k'], config['data']['m'])

        #     # ===== Generate Validation Data =====
        #     length_key, sample_key = jr.split(validation_key)
        #     validation_lengths = powerlaw(
        #         length_key, 
        #         15, 
        #         config['data']['max_length'], 
        #         config['data']['alpha'], 
        #         shape=(config['training']['validation_size'],)
        #     )

        #     _, validation_sequences = DyckHMM.batch_sample_sequence(
        #         config['training']['validation_size'], 
        #         config['data']['max_length'], 
        #         validation_lengths, 
        #         key = sample_key
        #     )

        #     validation_x = validation_sequences[:,:-1]
        #     validation_y = validation_sequences[:,1:]
        #     validation_mask = validation_x != (2 * config['data']['k'] + 1)

        #     # ==== Initalize model ====
        #     model = SequenceModel(
        #         cell_type = config['model']['cell_type'],
        #         vocab_size = 2 * config['data']['k'] + 2, 
        #         hidden_size = config['model']['hidden_size'], 
        #         out_size = 2 * config['data']['k'] + 2, 
        #         readout_depth = config['model']['readout_depth'], 
        #         rnn_scale = config['model']['init_scale'],
        #         key = model_key)

        #     # ==== Optimization setup ====
        #     def loss_func(model, obs, next_obs, mask):
        #         pred_loss = jax.vmap(pred_loss_func, 
        #                             in_axes=[None, 0, 0, 0])(
        #                             model, obs, next_obs, mask)

        #         return pred_loss.mean()

        #     learning_rate = config['optimizer']['learning_rate']
        #     optimizer = optax.chain(
        #         optax.clip_by_global_norm(1.0),  # Gradient norm clipping
        #         optax.adam(learning_rate=learning_rate)
        #         )
        #     opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

        #     initial_validation_loss = loss_func(
        #         model, 
        #         validation_x, 
        #         validation_y, 
        #         validation_mask)

        #     print(f"Initial Validaiton Loss: {initial_validation_loss:0.4f}")

        #     # ==== Initalize Metrics ====
        #     validation_loss_history = [initial_validation_loss]
        #     training_loss_history = []

        #     # ==== Training Loop ====
        #     epoch = 1
        #     counter = 0 
        #     training_start = time.time()

        #     while counter < config['training']['max_counter']:
        #         epoch_start_time = time.time()
        #         batch_key, length_key = jr.split(training_key, 2)

        #         # Sequence lengths
        #         epoch_lengths = powerlaw(
        #             length_key, 
        #             config['data']['min_length'], 
        #             config['data']['max_length'], 
        #             config['data']['alpha'], 
        #             shape=(config['training']['batches_per_epoch'],
        #                 config['training']['batch_size'],)
        #         )

        #         # Train one epoch
        #         model, opt_state, loss_history = train_one_epoch(
        #             model, 
        #             DyckHMM, 
        #             loss_func,
        #             config['training']['batch_size'], 
        #             config['training']['batches_per_epoch'],
        #             config['data']['max_length'], 
        #             epoch_lengths, 
        #             opt_state, 
        #             optimizer,
        #             key = batch_key)
                
        #         training_loss_history.append(loss_history)

        #         # Validation loss
        #         epoch_validation_loss = loss_func(
        #             model, 
        #             validation_x, 
        #             validation_y, 
        #             validation_mask)
        #         validation_loss_history.append(epoch_validation_loss)
                
        #         # Early Stopping/Learning Rate Schedule Logic
        #         if epoch_validation_loss > min(validation_loss_history):
        #             # Decrement learning rate
        #             learning_rate *= 0.5

        #             # Re-initalize the optimizer and opt_state
        #             optimizer = optax.chain(
        #                 optax.clip_by_global_norm(1.0),  # Gradient norm clipping
        #                 optax.adam(learning_rate=learning_rate)
        #             )

        #             # Increment counter
        #             counter += 1
                
        #         else:
        #             # If validation loss does achieve new minimum, reset counter
        #             counter = 0

        #         epoch_end_time = time.time()
        #         dt = epoch_end_time - epoch_start_time
        #         print(f"Epoch {epoch:02d} ({dt:0.2f} s): " \
        #             + f"{epoch_validation_loss} " \
        #             + f"(counter = {counter})", flush=True)
                
        #         _, training_key = jr.split(length_key)

        #         # Save checkpoint
        #         save_model(model, 
        #                 checkpoint_dir / (f'checkpoint_{epoch:02d}'))
        #         epoch += 1

        #     training_end = time.time()
        #     total_train_time = training_end - training_start

        #     # ==== Save Model ====
        #     save_model(model, run_dir / 'final.eqx')
        #     print('model saved')

        #     # ==== Save Metrics ====
        #     machine_info = {
        #         "hostname": socket.gethostname(),
        #         "backend": jax.default_backend(),
        #         "devices": [str(d) for d in jax.devices()],
        #     }
            
        #     metrics = {
        #         "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        #         "total_train_time": float(total_train_time),
        #         "train_loss_history": [
        #             jax.device_get(x).tolist() for x in training_loss_history
        #         ],
        #         "validation_loss_history": [
        #             float(jax.device_get(x)) for x in validation_loss_history
        #         ],
        #         "best_validation_loss": float(
        #             jax.device_get(min(validation_loss_history))
        #         ),
        #         "machine_info": machine_info,
        #     }

        #     with open(run_dir / "metrics.json", "w") as f:
        #         json.dump(metrics, f, indent=2)

        # except Exception as e:
        #     print(f"Run failed for {run_name}: {e}")
        #     continue
        
