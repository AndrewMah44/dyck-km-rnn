import jax
jax.config.update("jax_enable_x64", True)

import yaml
import equinox as eqx
import jax.random as jr
import jax.numpy as jnp
from pathlib import Path
from dyck_rnn.models.rnn import SequenceModel


def _cast_inexact(tree, dtype):
    return jax.tree.map(
        lambda x: x.astype(dtype) if eqx.is_inexact_array(x) else x,
        tree,
    )

def load_model(
        run_name, 
        filename="final.eqx", 
        run_dir="runs", 
        target_dtype=jnp.float64):
    
    run_path = Path(run_dir) / run_name

    with (run_path / "config.yaml").open("r") as f:
        config = yaml.safe_load(f)

    blank_model = SequenceModel(
        cell_type=config["model"]["cell_type"],
        vocab_size=2 * config["data"]["k"] + 2,
        hidden_size=config["model"]["hidden_size"],
        out_size=2 * config["data"]["k"] + 2,
        readout_depth=config["model"]["readout_depth"],
        rnn_scale=config["model"]["init_scale"],
        key=jr.PRNGKey(config["experiment"]["seed"]),
    )

    # Try loading in the checkpoint's native dtype.
    for dtype in (jnp.float64, jnp.float32):
        try:
            template = _cast_inexact(blank_model, dtype)
            like = eqx.filter(template, eqx.is_inexact_array)
            loaded_params = eqx.tree_deserialise_leaves(
                run_path / filename, like)
            model = eqx.combine(
                loaded_params, 
                eqx.partition(template, eqx.is_inexact_array)[1])
            return _cast_inexact(model, target_dtype)
        except RuntimeError:
            pass
