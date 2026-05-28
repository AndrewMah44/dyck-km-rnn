import jax
jax.config.update("jax_enable_x64", True)

import yaml
import equinox as eqx
import jax.random as jr
import jax.numpy as jnp
from pathlib import Path
from dyck_rnn.models.rnn import SequenceModel

def load_model(run_name, filename='final.eqx', run_dir="runs"):
    # ==== Paths ====
    run_path = Path(run_dir) / run_name

    # ==== Load Configuration File ====
    config_file = run_path / 'config.yaml'
    with config_file.open("r") as f:
        config = yaml.safe_load(f)


    # ==== Make Blank Model with Same Size ====
    blank_model = SequenceModel(
        cell_type = config['model']['cell_type'],
        vocab_size = 2 * config['data']['k'] + 2, 
        hidden_size = config['model']['hidden_size'], 
        out_size = 2 * config['data']['k'] + 2, 
        readout_depth = config['model']['readout_depth'], 
        rnn_scale = config['model']['init_scale'],
        key = jr.PRNGKey(config['experiment']['seed']))

    # Enforce float32 for all parameters (trained with float64)
    blank_model = jax.tree.map(
        lambda x: (
            x.astype(jnp.float32)
            if eqx.is_inexact_array(x)
            else x
        ),
        blank_model,
    )

    # ==== Load Trained Parameters and Combine ====
    loaded_params = eqx.tree_deserialise_leaves(
        run_path / filename, 
        eqx.filter(blank_model, eqx.is_inexact_array))
    
    reconstructed_model = eqx.combine(
        loaded_params, 
        eqx.partition(blank_model, eqx.is_inexact_array)[1])
    
    return reconstructed_model