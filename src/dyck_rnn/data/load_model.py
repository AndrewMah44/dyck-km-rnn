import yaml
import equinox as eqx
import jax.random as jr
from pathlib import Path
from dyck_rnn.models.rnn import SequenceModel

def load_model(run_name, filename='final.eqx'):
    # ==== Paths ====
    run_path = Path("runs") / run_name

    # ==== Load Configuration File ====
    config_file = run_path / 'config.yaml'
    with config_file.open("r") as f:
        config = yaml.safe_load(f)


    # ==== Make Blank Model with Same Size ====
    blank_model = SequenceModel(
        cell_type = config['model']['cell_type'],
        in_size = 2 * config['data']['k'] + 2, 
        hidden_size = config['model']['hidden_size'], 
        out_size = 2 * config['data']['k'] + 2, 
        readout_depth = config['model']['readout_depth'], 
        rnn_scale = config['model']['init_scale'],
        key = jr.PRNGKey(config['experiment']['seed']))

    # ==== Load Trained Parameters and Combine ====
    loaded_params = eqx.tree_deserialise_leaves(
        run_path / filename, 
        eqx.filter(blank_model, eqx.is_inexact_array))
    
    reconstructed_model = eqx.combine(
        loaded_params, 
        eqx.partition(blank_model, eqx.is_inexact_array)[1])
    
    return reconstructed_model